"""One auditor per hotkey.

The validator of the subnet owner hotkey is the auditor. Two instances of it, as during a move to another machine or
when an old host comes back after its replacement, would both run the throughput tests, publish documents and set
weights, and the hotkey's axon would point at whichever started last. So before an instance with the owner hotkey
starts any of its work, and at the start of every epoch while it runs, it reads the hotkey's axon on chain. Where the
axon points at another address than its own ``public_host:port``, it asks there for the auditor's signed documents,
``GET /v1/status/{hotkey}`` and then ``GET /v1/network``, over TLS accepting the certificate whose pin the catalog names
for the auditor, both within ``TIMEOUT_S`` in total. If either verifies (signed by the hotkey for this netuid within
``STATUS_MAX_AGE_S`` of this machine's clock, ``protocol.status.unverified``), another auditor runs there: the instance
does not start, or stops, and its process manager's restarts are refused while the other one runs. Nothing else counts:
no answer, an error, or a document that is stale or does not verify is a host that is gone, as after a planned move,
and the instance runs.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

import bittensor as bt
import httpx

from verathos.catalog import AuditorUnknown, Catalog
from verathos.protocol import tls
from verathos.protocol.http import request_json
from verathos.protocol.status import (
    MAX_NETWORK_BYTES,
    MAX_STATUS_BYTES,
    NETWORK_DOMAIN,
    NETWORK_PATH,
    STATUS_DOMAIN,
    STATUS_PATH,
    unverified,
)
from verathos.subnet.chain import axon_url, one_line

log = logging.getLogger(__name__)
TIMEOUT_S = 5.0  # both documents in total: a slow or gone host delays a start this long at most
REFUSED = ("another auditor with this hotkey runs at {url}, where the hotkey's axon points: this one does not start; "
           "stop that one first, then start this one")
STOPPED = ("another auditor with this hotkey runs at {url}, where it moved the hotkey's axon: this one stops (no more "
           "tests or weights); stop that one first, then start this one")


async def silent(client: httpx.AsyncClient, url: str, *, hotkey: str, netuid: int, pin: str,
                 timeout_s: float = TIMEOUT_S) -> str | None:
    """Why no auditor with ``hotkey`` answers at ``url`` (a base URL), from the documents asked there within
    ``timeout_s`` in total; None if one answers with a document that verifies."""
    deadline, why = time.monotonic() + timeout_s, []
    for path, domain, limit in ((STATUS_PATH.format(hotkey=hotkey), STATUS_DOMAIN, MAX_STATUS_BYTES),
                                (NETWORK_PATH, NETWORK_DOMAIN, MAX_NETWORK_BYTES)):
        left = deadline - time.monotonic()
        if left <= 0:
            why.append(f"{path}: not asked, no time left of {timeout_s:g} s")
            break
        try:
            document = await request_json(client, "GET", url + path, limit=limit, timeout=left, tls=pin)
        except TimeoutError:
            why.append(f"{path}: no answer within {left:.1f} s")
            continue
        except Exception as exc:  # no document is no answer: only one that verifies stops this instance
            why.append(f"{path}: {one_line(exc).splitlines()[0]}")
            continue
        reason = unverified(document, domain=domain, auditor=hotkey, netuid=netuid, now_ms=int(time.time() * 1000))
        if reason is None:
            return None
        why.append(f"{path}: {reason}")
    return "; ".join(why)


def another_auditor(st: bt.Subtensor, netuid: int, hotkey: str, own: str | None, cat: Catalog, *,
                    client: Callable[[], httpx.AsyncClient] = tls.client, timeout_s: float = TIMEOUT_S) -> str | None:
    """The base URL of ``hotkey``'s axon if another auditor with the hotkey answers there (see the module); None if the
    axon is ``own`` (this instance's ``public_host:port``), serves nothing, or no auditor answers there, which is
    logged."""
    url = axon_url(st, netuid, hotkey)
    if url is None or url == own:
        return None

    async def ask(pin: str) -> str | None:
        async with client() as session:
            return await silent(session, url, hotkey=hotkey, netuid=netuid, pin=pin, timeout_s=timeout_s)

    try:
        why = asyncio.run(ask(cat.auditor_pin()))
    except AuditorUnknown as exc:
        why = str(exc)
    if why is None:
        return url
    log.warning("the subnet owner hotkey's axon points at %s, not at this auditor (%s), and no auditor answers there: "
                "%s", url, own or "no public_host", why)
    return None
