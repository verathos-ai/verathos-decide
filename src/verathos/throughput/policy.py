"""Throughput-audit policy, the public test schedule, and test sessions.

Once per epoch the auditor drives every listed endpoint at once, at the capacity the endpoint declares, and measures
what it delivers. The test block is drawn from block hashes, so everyone derives it from the chain alone and nobody
knows it before its selection block appears: within an epoch's eligible blocks, block ``b`` draws a hit when
``H(b, hash(b))`` falls below ``1 / (blocks left in the eligible range)``; the first hit of the epoch selects the
test (the last eligible block always hits), so there is exactly one test per epoch, uniformly placed. Traffic starts
``lead_blocks`` later. Following the schedule takes every block of the epoch in order (``Schedule``).

Test traffic carries a session the auditor signs once per endpoint and test, instead of a signature per request.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bittensor_wallet import Keypair
from pydantic import BaseModel, Field

from verathos.protocol import signing
from verathos.protocol.canonical import canonical_json
from verathos.protocol.requests import MAX_SKEW_MS, NonceCache, RequestRejected

BLOCK_TIME_S = 12.0
POLICY_WATCH_S = 60.0  # miners and validators check the committed catalog (and so the policy) this often
SELECTION_PREFIX = b"VERATHOS_THROUGHPUT_TEST_V1"
SESSION_DOMAIN = "verathos/throughput-session/v1"
HDR_SESSION = "X-Verathos-Session"
HDR_SESSION_SIGNATURE = "X-Verathos-Session-Signature"
HDR_REQUEST_ID = "X-Verathos-Request-Id"
_REQUEST_ID = re.compile(r"[0-9a-f]{32}")
_SIGNATURE = re.compile(r"[0-9a-f]{128}")
MAX_SESSIONS = 1024  # verified sessions a miner keeps
MAX_SESSION_S = 300.0


@dataclass(frozen=True)
class AuditPolicy:
    """The ``throughput_audit`` settings of the catalog."""

    mode: str = "enforce"  # enforce | observe (emergency switch: declared capacities count, nothing is gated)
    auditor_hotkey: str = ""  # empty: the subnet owner
    epoch_blocks: int = 180  # the catalog's epoch length
    lead_blocks: int = 5
    max_latency_s: float = 2.0  # the miner declares capacity only at concurrencies answering within this
    warmup_s: float = 3.0  # at least max_latency_s, so a test measures in steady state
    test_s: float = 16.0
    request_timeout_s: float = 20.0
    pass_fraction: float = 0.95
    samples_per_test: int = 10
    capacity_margin: float = 0.95
    qualification_tests: int = 1
    credit_ttl_epochs: int = 2
    rate_growth: float = 2.0  # an endpoint is tested at no more than this times what it last delivered
    min_test_share: float = 0.125  # ... but at least this share of the model's first_test_rps

    def validate(self) -> None:
        if self.mode not in ("enforce", "observe"):
            raise ValueError("mode must be enforce or observe")
        if not (0 < self.pass_fraction <= 1 and 0 < self.capacity_margin <= 1):
            raise ValueError("pass_fraction and capacity_margin must be in (0, 1]")
        if self.max_latency_s <= 0 or self.test_s <= 0 or self.request_timeout_s <= 0:
            raise ValueError("max_latency_s, test_s and request_timeout_s must be positive")
        if self.warmup_s < self.max_latency_s:
            raise ValueError("warmup_s must be at least max_latency_s")
        if self.session_s > MAX_SESSION_S:
            raise ValueError(f"warmup_s + test_s + request_timeout_s must stay under {MAX_SESSION_S - 5:.0f} s")
        if min(self.lead_blocks, self.samples_per_test, self.qualification_tests, self.credit_ttl_epochs) < 1:
            raise ValueError("lead_blocks, samples_per_test, qualification_tests and credit_ttl_epochs must be at "
                             "least 1")
        if self.rate_growth < 1:
            raise ValueError("rate_growth must be at least 1")
        if not 0 < self.min_test_share <= 1:
            raise ValueError("min_test_share must be in (0, 1]")
        if eligible_range(0, self)[1] < eligible_range(0, self)[0]:
            raise ValueError("the epoch is too short for a test")

    @property
    def session_s(self) -> float:
        """How long a test session stays valid after traffic starts: the traffic plus the last request's timeout."""
        return self.warmup_s + self.test_s + self.request_timeout_s + 5.0

    @property
    def traffic_blocks(self) -> int:
        """Blocks the test traffic spans until its last answers are due, rounded up."""
        return math.ceil((self.warmup_s + self.test_s + self.max_latency_s) / BLOCK_TIME_S)


def policy_from(settings: Mapping[str, Any], epoch_blocks: int) -> AuditPolicy:
    unknown = set(settings) - {f.name for f in dataclasses.fields(AuditPolicy)} | ({"epoch_blocks"} & set(settings))
    if unknown:
        raise ValueError(f"unknown throughput_audit settings: {sorted(unknown)}")
    policy = dataclasses.replace(AuditPolicy(), **settings, epoch_blocks=epoch_blocks)
    policy.validate()
    return policy


# ---- schedule ----

def eligible_range(epoch: int, policy: AuditPolicy) -> tuple[int, int]:
    """First and last block of ``epoch`` that may select the test, so that its traffic ends inside the epoch."""
    first = epoch * policy.epoch_blocks + 1
    last = (epoch + 1) * policy.epoch_blocks - 1 - policy.lead_blocks - policy.traffic_blocks - 1
    return first, last


def draws_hit(block: int, block_hash: bytes, policy: AuditPolicy) -> bool:
    """Whether ``block`` draws a hit; the epoch's first hit selects its test."""
    first, last = eligible_range(block // policy.epoch_blocks, policy)
    if not first <= block <= last:
        return False
    digest = hashlib.sha256(SELECTION_PREFIX + block.to_bytes(8, "big") + block_hash).digest()
    return int.from_bytes(digest, "big") * (last - block + 1) < 1 << 256


@dataclass(frozen=True)
class ScheduledTest:
    """A network-wide test selected by ``selection_block``."""

    epoch: int
    selection_block: int
    selection_hash: str
    start_block: int

    @property
    def test_id(self) -> str:
        return hashlib.sha256(f"{self.epoch}:{self.selection_block}:{self.selection_hash}".encode()).hexdigest()

    def held(self, block: int, policy: AuditPolicy) -> bool:
        """Whether ``block`` falls in the test's traffic, with one block of margin on each side for differing block
        observation. Validators hold their canaries meanwhile."""
        return self.start_block - 1 <= block <= self.start_block + policy.traffic_blocks + 1


class Schedule:
    """Follows blocks in order and reports each epoch's test when its selection block appears. A node that starts
    mid-epoch feeds the epoch from its first block (``eligible_range``) before following the head. A new policy
    applies from the next block observed."""

    def __init__(self, policy: AuditPolicy) -> None:
        self.policy = policy
        self._tests: dict[int, ScheduledTest | None] = {}

    def observe(self, block: int, block_hash: bytes) -> ScheduledTest | None:
        """The test this block selects, if any."""
        epoch = block // self.policy.epoch_blocks
        if self._tests.get(epoch) is not None or not draws_hit(block, block_hash, self.policy):
            return None
        test = ScheduledTest(epoch, block, block_hash.hex(), block + self.policy.lead_blocks)
        self._tests[epoch] = test
        for old in [e for e in self._tests if e < epoch - 2]:
            del self._tests[old]
        return test

    def adopt(self, policy: AuditPolicy) -> None:
        """Continue under a new policy; tests already selected stay, so an epoch never gets a second one (a new epoch
        length forgets them)."""
        if policy.epoch_blocks != self.policy.epoch_blocks:
            self._tests = {}
        self.policy = policy

    def held(self, block: int) -> bool:
        """Whether ``block`` is inside a test's traffic (validators hold canaries)."""
        test = self._tests.get(block // self.policy.epoch_blocks)
        return test is not None and test.held(block, self.policy)


# ---- sessions ----

class Session(BaseModel):
    """Authorizes the auditor's test traffic to one endpoint for one test."""

    netuid: int
    auditor: str
    miner_hotkey: str
    endpoint_id: str
    test_id: str
    not_after_ms: int = Field(ge=0)


def session_headers(session: Session, keypair: Keypair) -> dict[str, str]:
    payload = session.model_dump()
    return {HDR_SESSION: base64.urlsafe_b64encode(canonical_json(payload)).decode(),
            HDR_SESSION_SIGNATURE: signing.sign(keypair, SESSION_DOMAIN, payload)}


def read_session(headers: Mapping[str, str]) -> tuple[Session, str] | None:
    """The session and its signature from request headers, or None if absent or malformed."""
    h = {k.lower(): v for k, v in headers.items()}
    raw, signature = h.get(HDR_SESSION.lower()), h.get(HDR_SESSION_SIGNATURE.lower())
    if not raw or not signature:
        return None
    try:
        return Session.model_validate(json.loads(base64.urlsafe_b64decode(raw.encode()))), signature
    except ValueError:
        return None


class SessionGate:
    """A miner's check of test traffic: a session signed by the current auditor (or by the miner itself, for its own
    measurement) for this miner and endpoint, not expired (allowing the same clock skew as signed requests), and a
    request id not seen before in the session. Each session's signature is verified once."""

    def __init__(self, netuid: int, miner_hotkey: str) -> None:
        self.netuid, self.miner_hotkey = netuid, miner_hotkey
        self._verified: dict[str, Session] = {}
        self._request_ids = NonceCache(window_ms=int(MAX_SESSION_S * 1000) + MAX_SKEW_MS)

    def check(self, headers: Mapping[str, str], *, auditor: str | None, endpoint_id: str, now_ms: int) -> tuple[
            Session, str]:
        """-> (session, request id). Raises RequestRejected."""
        found = read_session(headers)
        if found is None:
            raise RequestRejected("malformed test session")
        session, signature = found
        request_id = {k.lower(): v for k, v in headers.items()}.get(HDR_REQUEST_ID.lower(), "")
        if not _REQUEST_ID.fullmatch(request_id):
            raise RequestRejected("malformed request id")
        if not (session.auditor in (auditor, self.miner_hotkey) and session.netuid == self.netuid
                and session.miner_hotkey == self.miner_hotkey and session.endpoint_id == endpoint_id):
            raise RequestRejected("test session is for another auditor, miner or endpoint")
        if now_ms > session.not_after_ms + MAX_SKEW_MS:
            raise RequestRejected("test session expired")
        if self._verified.get(signature) != session:
            if not (_SIGNATURE.fullmatch(signature)
                    and signing.verify(session.auditor, SESSION_DOMAIN, session.model_dump(), signature)):
                raise RequestRejected("bad test session signature")
            live = {s: v for s, v in self._verified.items() if v.not_after_ms + MAX_SKEW_MS >= now_ms}
            self._verified = dict(list(live.items())[-(MAX_SESSIONS - 1):]) | {signature: session}
        if not self._request_ids.check_and_add(f"{session.test_id}:{request_id}", now_ms):
            raise RequestRejected("replayed request")
        return session, request_id
