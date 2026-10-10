"""A client that leaves while it sends a request's body, driven at the ASGI level (test clients send bodies whole)."""

from __future__ import annotations

from typing import Any


async def left_mid_body(app: Any, path: str, headers: dict[str, str]) -> int | None:
    """POST to ``path`` with ``headers``, sending part of the body and then leaving. -> the status answered, if any."""
    messages = [{"type": "http.request", "body": b'{"model": "kev-4b", "state": "Hi', "more_body": True},
                {"type": "http.disconnect"}]

    async def receive() -> dict[str, Any]:
        return messages.pop(0) if messages else {"type": "http.disconnect"}

    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "https",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
             "client": ("203.0.113.9", 40000), "server": ("198.51.100.1", 443)}
    await app(scope, receive, send)
    return next((m["status"] for m in sent if m["type"] == "http.response.start"), None)
