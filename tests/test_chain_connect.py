"""A miner, validator or gateway that starts while the chain's node refuses connections waits for it."""

from __future__ import annotations

import logging

import bittensor as bt
import pytest
from websockets.exceptions import InvalidHandshake

from verathos.subnet import chain


def test_a_process_connects_to_the_chain_again_while_the_node_refuses_it(monkeypatch, caplog) -> None:
    """A public node refuses connections for a while under load (HTTP 429): the process tries again, one log line per
    try, where it exited, and a supervisord that saw it exit ten times in its first seconds gave up on it."""
    refused = InvalidHandshake("server rejected WebSocket connection: HTTP 429")
    refusals = iter([refused, ConnectionRefusedError(111, "Connection refused"), TimeoutError("timed out"),
                     refused, refused, refused])

    def subtensor(network: str):
        if (refusal := next(refusals, None)) is not None:
            raise refusal
        return f"connected to {network}"

    monkeypatch.setattr(bt, "Subtensor", subtensor)
    waits: list[float] = []
    with caplog.at_level(logging.WARNING, logger="verathos.subnet.chain"):
        assert chain.connect("finney", sleep=waits.append) == "connected to finney"
    assert waits == [5, 15, 30, 60, 60, 60]
    assert caplog.text.count("cannot connect to the finney chain (InvalidHandshake: server rejected WebSocket "
                             "connection: HTTP 429); trying again in") == 4
    assert "Traceback" not in caplog.text


def test_an_error_that_is_no_refusal_by_the_chain_stops_the_start(monkeypatch) -> None:
    def subtensor(network: str):
        raise ValueError(f"{network} is not a network")

    monkeypatch.setattr(bt, "Subtensor", subtensor)
    with pytest.raises(ValueError, match="not a network"):
        chain.connect("finny", sleep=lambda s: pytest.fail("tried again"))


def test_a_network_can_name_several_nodes_the_first_used_while_it_answers(monkeypatch) -> None:
    import bittensor as bt
    from bittensor.core.settings import NETWORK_MAP

    from verathos.protocol.tls import chain_network
    from verathos.subnet import chain as subnet_chain

    assert chain_network("wss://node.example.org:443,finney") == "wss://node.example.org:443,finney"
    assert chain_network("finney") == "finney"
    for refused in ("wss://node.example.org,ws://203.0.113.5:9944", "finney,", "ws://203.0.113.5:9944"):
        with pytest.raises(ValueError, match="wss://"):
            chain_network(refused)
    made = []
    monkeypatch.setattr(bt, "Subtensor", lambda **kw: made.append(kw) or "client")
    assert subnet_chain.subtensor("wss://node.example.org:443, finney") == "client"
    assert made == [{"network": "wss://node.example.org:443", "fallback_endpoints": [NETWORK_MAP["finney"]]}]
    subnet_chain.subtensor("finney")
    assert made[-1] == {"network": "finney"}  # one network: the client as before
