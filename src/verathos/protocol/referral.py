"""Referral codes, and the messages a hotkey signs for them (``protocol.signing``: the domain, a line break and the
canonical JSON of the message, signed with the hotkey's sr25519 key, as hex):

- ``verathos/referral-code/v1``: a code's registration, ``{netuid, hotkey, code, site, issued_ms}``, signed by
  ``hotkey``;
- ``verathos/referral-move/v1``: every code of ``from_hotkey`` moved to ``to_hotkey`` (another hotkey of the same
  coldkey), ``{netuid, from_hotkey, to_hotkey, site, issued_ms}``, signed by ``to_hotkey``;
- ``verathos/referral-view/v1``: a read-only view of a hotkey's referral figures, ``{netuid, hotkey, site,
  expires_ms}``, signed by ``hotkey`` and carried as a token (``view_token``).

``site`` is the origin of the web app the message is for (``https://host``), so a message signed for one site is
refused by another. A registration or a move is taken within ``SIGNED_MAX_AGE_MS`` after it was signed; a view until
it expires, at most ``VIEW_MAX_MS`` after it was signed.

A code is random, ``RANDOM_LENGTH`` characters of ``RANDOM_ALPHABET`` (no 0, o, 1, l or i), or readable: 4 to 24
characters of a-z, digits and single hyphens, starting with a letter. Codes are unique after folding (``fold``): lower
case, hyphens dropped, 0 read as o, 1 as l and 5 as s, so ``a1ice`` cannot shadow ``alice``.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import secrets

from bittensor_wallet import Keypair
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from verathos.protocol import signing
from verathos.protocol.canonical import canonical_json

CODE_DOMAIN = "verathos/referral-code/v1"
MOVE_DOMAIN = "verathos/referral-move/v1"
VIEW_DOMAIN = "verathos/referral-view/v1"
SIGNED_MAX_AGE_MS = 300_000
SIGNED_AHEAD_MS = 60_000  # a message signed by a clock this far ahead is still taken
VIEW_MAX_MS = 86_400_000
RANDOM_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"
RANDOM_LENGTH = 8
READABLE_LENGTH = (4, 24)
_RANDOM = re.compile(f"[{RANDOM_ALPHABET}]{{{RANDOM_LENGTH}}}")
_READABLE = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
_FOLD = str.maketrans({"0": "o", "1": "l", "5": "s"})
_SITE = re.compile(r"https://[a-z0-9.-]+(?::[0-9]{1,5})?")


def well_formed(code: str) -> bool:
    """Whether ``code`` is a random or a readable code, in lower case."""
    if _RANDOM.fullmatch(code):
        return True
    return READABLE_LENGTH[0] <= len(code) <= READABLE_LENGTH[1] and _READABLE.fullmatch(code) is not None


def fold(code: str) -> str:
    """The form codes are unique in."""
    return code.lower().replace("-", "").translate(_FOLD)


def random_code() -> str:
    return "".join(secrets.choice(RANDOM_ALPHABET) for _ in range(RANDOM_LENGTH))


def site_origin(url: str) -> str:
    """The origin a message names for a web app's URL: ``https://host[:port]`` in lower case. Raises ValueError."""
    origin = url.strip().rstrip("/").lower()
    if not _SITE.fullmatch(origin):
        raise ValueError(f"{url!r} is not an https origin (https://host[:port])")
    return origin


class CodeRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    netuid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=64)
    code: str = Field(min_length=1, max_length=READABLE_LENGTH[1])
    site: str = Field(min_length=1, max_length=256)
    issued_ms: int = Field(ge=0)


class SignedRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")
    registration: CodeRegistration
    signature: str = Field(max_length=256)

    @classmethod
    def create(cls, registration: CodeRegistration, keypair: Keypair) -> SignedRegistration:
        return cls(registration=registration,
                   signature=signing.sign(keypair, CODE_DOMAIN, registration.model_dump(mode="json")))

    def verify(self) -> bool:
        """Whether the registration's hotkey signed it."""
        r = self.registration
        return signing.verify(r.hotkey, CODE_DOMAIN, r.model_dump(mode="json"), self.signature)


class CodeMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    netuid: int = Field(ge=0)
    from_hotkey: str = Field(min_length=1, max_length=64)
    to_hotkey: str = Field(min_length=1, max_length=64)
    site: str = Field(min_length=1, max_length=256)
    issued_ms: int = Field(ge=0)


class SignedMove(BaseModel):
    model_config = ConfigDict(extra="forbid")
    move: CodeMove
    signature: str = Field(max_length=256)

    @classmethod
    def create(cls, move: CodeMove, keypair: Keypair) -> SignedMove:
        return cls(move=move, signature=signing.sign(keypair, MOVE_DOMAIN, move.model_dump(mode="json")))

    def verify(self) -> bool:
        """Whether the hotkey the codes move to signed it."""
        m = self.move
        return signing.verify(m.to_hotkey, MOVE_DOMAIN, m.model_dump(mode="json"), self.signature)


def fresh(issued_ms: int, now_ms: int) -> bool:
    """Whether a registration or a move signed at ``issued_ms`` is taken at ``now_ms``."""
    return now_ms - SIGNED_MAX_AGE_MS <= issued_ms <= now_ms + SIGNED_AHEAD_MS


class ReferralView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    netuid: int = Field(ge=0)
    hotkey: str = Field(min_length=1, max_length=64)
    site: str = Field(min_length=1, max_length=256)
    expires_ms: int = Field(ge=0)


def view_token(view: ReferralView, keypair: Keypair) -> str:
    """``<payload>.<signature>``: the view's canonical JSON in unpadded base64url, and the hotkey's signature of it."""
    body = view.model_dump(mode="json")
    payload = base64.urlsafe_b64encode(canonical_json(body)).rstrip(b"=").decode()
    return f"{payload}.{signing.sign(keypair, VIEW_DOMAIN, body)}"


def read_view_token(token: str, *, netuid: int, site: str, now_ms: int) -> ReferralView:
    """The view a token carries, if its hotkey signed it for ``netuid`` and ``site`` and it is valid at ``now_ms``.
    Raises ValueError with the reason."""
    payload, _, signature = token.partition(".")
    try:
        body = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        view = ReferralView.model_validate(body)
    except (binascii.Error, UnicodeDecodeError, ValueError, ValidationError):
        raise ValueError("not a referral view token") from None
    if (view.netuid, view.site) != (netuid, site):
        raise ValueError("the token is for another netuid or site")
    if not now_ms < view.expires_ms <= now_ms + VIEW_MAX_MS:
        raise ValueError("the token has expired, or is valid for longer than a day")
    if not signing.verify(view.hotkey, VIEW_DOMAIN, view.model_dump(mode="json"), signature):
        raise ValueError("the token's signature does not verify against its hotkey")
    return view
