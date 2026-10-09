"""Who calls miners and from where: validator commitments, the catalog's auditor and gateways, the allowlist."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from verathos import catalog
from verathos.protocol.callers import (
    MAX_COMMITMENT_BYTES,
    VALIDATOR_PREFIX,
    ValidatorIdentity,
    allowlist,
    egress_addresses,
    validator_identity,
)
from verathos.protocol.manifest import commitment_value
from verathos.validator.discovery import Neuron

PIN = "0f" * 32
GATEWAY = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty"


def test_egress_addresses_are_public_unicast_literals_in_their_normal_form() -> None:
    assert egress_addresses(["8.8.8.8", "2606:4700::1111"]) == ["8.8.8.8", "2606:4700::1111"]
    for bad, why in ((["10.0.0.1"], "not a public"), (["127.0.0.1"], "not a public"), (["::ffff:8.8.8.8"], "public"),
                     (["2606:4700:0::1111"], "write the address as 2606:4700::1111"), (["8.8.8.8:80"], "not an IP"),
                     (["dns.example"], "not an IP"), (["8.8.8.8", "8.8.8.8"], "unique"),
                     ([f"8.8.8.{i}" for i in range(1, 18)], "at most 16")):
        with pytest.raises(ValueError, match=why):
            egress_addresses(bad)


def test_a_validator_commitment_names_its_pin_and_egress_addresses_in_one_raw_field() -> None:
    identity = ValidatorIdentity(PIN, ("8.8.8.8", "2606:4700::1111"))
    value = identity.commitment()
    assert value.startswith(VALIDATOR_PREFIX) and len(value.encode()) <= MAX_COMMITMENT_BYTES
    assert validator_identity(value) == identity
    assert validator_identity(ValidatorIdentity(None, ("8.8.8.8",)).commitment()) == ValidatorIdentity(None,
                                                                                                        ("8.8.8.8",))
    assert ValidatorIdentity(None, ()).commitment() == VALIDATOR_PREFIX + "- -"
    # the longest IPv6 literal and an IPv4 address fit next to a pin; more do not
    longest = "2606:4700:4700:1111:2606:4700:4700:1112"
    assert validator_identity(ValidatorIdentity(PIN, (longest, "198.211.103.255")).commitment())
    with pytest.raises(ValueError, match="list fewer egress addresses"):
        ValidatorIdentity(PIN, (longest, "198.211.103.255", "198.211.103.254")).commitment()


@pytest.mark.parametrize("value", [
    None, "", "verathos/validator/v1", "verathos/validator/v1 - - -", "verathos/validator/v1 x 8.8.8.8",
    "verathos/validator/v1 - 10.0.0.1", "verathos/validator/v1 - 8.8.8.8,", "verathos/validators/v1 - 8.8.8.8",
    "verathos/validator/v1 D_______________________________________8 -",  # not the canonical encoding
    "verathos/validator/v1 " + "A" * 42 + "! -", commitment_value("0" * 64)])
def test_anything_else_is_no_validator_commitment(value) -> None:
    assert validator_identity(value) is None


def test_the_allowlist_holds_the_addresses_of_the_callers_miners_answer() -> None:
    """The auditor's and the gateways' from the catalog, and those of the validators with a permit and the catalog's
    minimum stake that publish theirs; a light validator, a validator below the stake and a miner add nothing."""
    cat = catalog.Catalog.parse({
        **catalog.load().document(), "auditor": {"tls": PIN, "egress": ["9.9.9.9", "2620:fe::fe"]},
        "gateways": [{"hotkey": GATEWAY, "egress": ["1.1.1.1"]}],
        "scoring": {**catalog.load().document()["scoring"], "validator_min_stake": 1000.0}})

    def validator(uid: int, egress: tuple[str, ...], *, stake: float = 5000.0, permit: bool = True) -> Neuron:
        return Neuron(uid, f"hk{uid}", "0.0.0.0", 0, ValidatorIdentity(None, egress).commitment(), stake, permit)

    neurons = [validator(1, ("8.8.8.8",)), validator(2, ("8.8.4.4", "2001:4860:4860::8888")),
               validator(3, ("4.4.4.4",), stake=999.0), validator(4, ("5.5.5.5",), permit=False),
               Neuron(5, "hk5", "1.2.3.4", 8091, commitment_value("0" * 64), 0.0, False), validator(6, ())]
    assert allowlist(neurons, cat) == ["1.1.1.1", "8.8.4.4", "8.8.8.8", "9.9.9.9", "2001:4860:4860::8888",
                                       "2620:fe::fe"]
    assert allowlist([], catalog.load()) == []


def test_a_catalog_names_the_auditor_and_its_gateways_with_their_addresses() -> None:
    document = catalog.load().document()
    cat = catalog.Catalog.parse({**document, "auditor": {"tls": PIN, "egress": ["9.9.9.9"]},
                                 "gateways": [{"hotkey": GATEWAY, "egress": ["1.1.1.1"]}]})
    assert cat.auditor_pin() == PIN and cat.gateway_hotkeys() == [GATEWAY] and cat.identity_problems() == []
    with pytest.raises(catalog.AuditorUnknown, match="names no TLS pin for the auditor"):
        catalog.load().auditor_pin()
    for changes, why in (({"auditor": {"tls": "0F" * 32}}, "pattern"), ({"auditor": {"egress": ["9.9.9.9"]}}, "tls"),
                         ({"auditor": {"tls": PIN, "egress": ["10.0.0.1"]}}, "public"),
                         ({"gateways": [GATEWAY]}, "valid dictionary"),
                         ({"gateways": [{"hotkey": GATEWAY}, {"hotkey": GATEWAY}]}, "unique"),
                         ({"gateways": [{"hotkey": "5Fnot"}]}, "ss58")):
        with pytest.raises(ValueError, match=why):
            catalog.Catalog.parse({**document, **changes})
    mistyped = catalog.Catalog.parse({**document, "auditor": {"tls": PIN, "egres": []},
                                      "gateways": [{"hotkey": GATEWAY, "ip": "1.1.1.1"}]})
    assert mistyped.unknown_settings() == ["auditor.egres", f"gateways.{GATEWAY}.ip"]


class Chain:
    def __init__(self, commitment: str | None) -> None:
        self.commitment, self.committed = commitment, []

    def get_uid_for_hotkey_on_subnet(self, hotkey: str, netuid: int) -> int:
        return 4

    def get_commitment(self, netuid: int, uid: int) -> str | None:
        return self.commitment

    def set_commitment(self, wallet, netuid: int, value: str):
        self.committed.append(value)
        self.commitment = value
        return SimpleNamespace(success=True)


def test_a_full_validator_commits_its_identity_once_and_never_over_a_manifest() -> None:
    from verathos.validator.chain import publish_identity

    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hk4"))
    identity = ValidatorIdentity(PIN, ("8.8.8.8",))
    chain = Chain(None)
    publish_identity(chain, wallet, 405, identity)
    publish_identity(chain, wallet, 405, identity)
    assert chain.committed == [identity.commitment()]
    moved = ValidatorIdentity(PIN, ("8.8.4.4",))
    publish_identity(chain, wallet, 405, moved)
    assert chain.committed == [identity.commitment(), moved.commitment()]
    mining = Chain(commitment_value("0" * 64))
    with pytest.raises(RuntimeError, match="commits a miner's manifest"):
        publish_identity(mining, wallet, 405, identity)
    assert mining.committed == []
