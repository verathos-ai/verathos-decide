"""TLS between machines: identities kept in a state directory, and pinned clients over real TLS on a loopback
port."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from fastapi import FastAPI, Request

from tests.tlsfake import serving
from verathos.protocol import tls


def counting_app() -> tuple[FastAPI, list[str]]:
    """An app that records the paths it was asked for."""
    app, asked = FastAPI(), []

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def anything(path: str, request: Request):
        asked.append(path)
        return {"path": path}

    return app, asked


def test_an_identity_is_created_once_kept_private_and_rotated_by_deleting_it(tmp_path) -> None:
    first = tls.identity(tmp_path / "state")
    assert first.pin == tls.stored_pin(tmp_path / "state") and tls.is_pin(first.pin)
    for path in (first.key_path, first.cert_path):
        assert path.stat().st_mode & 0o777 == 0o600
    assert tls.identity(tmp_path / "state") == first  # reused across restarts
    first.cert_path.unlink()  # a certificate is made again for the same key: the same pin
    assert tls.identity(tmp_path / "state").pin == first.pin
    first.cert_path.write_text("not a certificate")
    assert tls.identity(tmp_path / "state").pin == first.pin
    first.key_path.unlink()  # rotation: a new key, a new certificate, a new pin
    rotated = tls.identity(tmp_path / "state")
    assert rotated.pin != first.pin and tls.stored_pin(tmp_path / "state") == rotated.pin
    assert tls.stored_pin(tmp_path / "nothing") is None and not (tmp_path / "nothing").exists()
    assert tls.curl_pin("00" * 32) == "sha256//" + "A" * 43 + "="


def test_a_pinned_client_accepts_exactly_the_pinned_certificate(tmp_path) -> None:
    """Another self-signed certificate at the same address is refused before any byte of the request is sent; a
    connection checked for one pin never carries a request for another."""
    served, other = tls.identity(tmp_path / "served"), tls.identity(tmp_path / "other")
    app, asked = counting_app()

    async def run(base: str):
        async with tls.client() as client:
            ok = await client.get(base + "/a", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match=f"is not the pinned {other.pin}"):
                await client.get(base + "/b", extensions=tls.use(other.pin))
            again = await client.get(base + "/c", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match="no TLS pin"):
                await client.get(base + "/d")
            with pytest.raises(httpx.UnsupportedProtocol, match="only https"):
                await client.get(base.replace("https", "http") + "/e", extensions=tls.use(served.pin))
            with pytest.raises(httpx.ConnectError, match="an IP address has no CA certificate"):
                await client.get(base + "/f", extensions=tls.use(tls.CA))
            async with client.stream("GET", base + "/g", extensions=tls.use(tls.ANY)) as response:
                seen = tls.peer_pin(response)
                document = await response.aread()
            return ok.json(), again.json(), document, seen

    with serving(app, served) as base:
        ok, again, document, seen = asyncio.run(run(base))
        with tls.sync_client() as client:
            assert client.get(base + "/h", extensions=tls.use(served.pin)).json() == {"path": "h"}
            with pytest.raises(httpx.ConnectError, match="is not the pinned"):
                client.get(base + "/i", extensions=tls.use(other.pin))
    assert ok == {"path": "a"} and again == {"path": "c"} and document == b'{"path":"g"}' and seen == served.pin
    assert asked == ["a", "c", "g", "h"]  # nothing of the refused requests reached the server


def test_a_client_never_takes_its_proxies_from_the_environment(monkeypatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert not tls.client()._mounts and not tls.sync_client()._mounts
