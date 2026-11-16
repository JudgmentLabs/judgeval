"""Carries self-reported client identity across legacy stateless HTTP requests."""

from __future__ import annotations

import base64
import functools
import json
import uuid
from typing import Any, Dict, Optional

PREFIX = "judgment-mcp-v1."
MAX_TOKEN_LENGTH = 4096
MAX_INITIALIZE_BYTES = 8192


def read_client_identity(token: Optional[str]) -> Dict[str, str]:
    if not token or not token.startswith(PREFIX) or len(token) > MAX_TOKEN_LENGTH:
        return {}
    try:
        encoded = token[len(PREFIX) :]
        value = json.loads(
            base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        )
        return {
            key: value[key]
            for key in ("name", "version", "protocolVersion")
            if isinstance(value, dict)
            and isinstance(value.get(key), str)
            and value[key]
        }
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return {}


class McpClientIdentityMiddleware:
    """Adds an identity carrier only when the transport supplies no session ID.

    The carrier is telemetry, not an authentication or authorization credential.
    Native stateful sessions and modern requests pass through unchanged.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        headers = dict(scope.get("headers", []))
        if (
            scope["type"] == "http"
            and scope.get("method") in ("GET", "DELETE")
            and headers.get(b"mcp-session-id", b"").startswith(PREFIX.encode("ascii"))
        ):
            await send(
                {
                    "type": "http.response.start",
                    "status": 405,
                    "headers": [(b"allow", b"POST")],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or b"mcp-session-id" in headers
        ):
            await self.app(scope, receive, send)
            return
        body = bytearray()
        token = None
        collecting = True

        async def read() -> Any:
            nonlocal token, collecting
            message = await receive()
            if message["type"] == "http.request" and collecting:
                chunk = message.get("body", b"")
                if len(body) + len(chunk) > MAX_INITIALIZE_BYTES:
                    body.clear()
                    collecting = False
                else:
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        token = _initialize_token(body)
                        body.clear()
                        collecting = False
            return message

        async def write(message: Any) -> None:
            if (
                message["type"] == "http.response.start"
                and token is not None
                and message["status"] == 200
            ):
                response_headers = message.get("headers", [])
                if b"mcp-session-id" not in dict(response_headers):
                    message = {
                        **message,
                        "headers": [
                            *response_headers,
                            (b"mcp-session-id", token.encode("ascii")),
                        ],
                    }
            await send(message)

        await self.app(scope, read, write)


def _initialize_token(body: bytearray) -> Optional[str]:
    try:
        request = json.loads(body)
        if not isinstance(request, dict) or request.get("method") != "initialize":
            return None
        params = request.get("params", {})
        info = params.get("clientInfo", {})
        protocol = params.get("protocolVersion")
        if (
            not isinstance(info, dict)
            or not isinstance(info.get("name"), str)
            or not info["name"]
            or not isinstance(protocol, str)
            or protocol >= "2026-07-28"
        ):
            return None
        value = {
            key: info[key]
            for key in ("name", "version")
            if isinstance(info.get(key), str)
        }
        value.update(protocolVersion=protocol, nonce=str(uuid.uuid4()))
        token = PREFIX + base64.urlsafe_b64encode(
            json.dumps(value).encode()
        ).decode().rstrip("=")
        return token if len(token) <= MAX_TOKEN_LENGTH else None
    except (ValueError, TypeError, AttributeError, RecursionError):
        return None


def instrument_http_app(server: Any) -> None:
    """Installs the carrier at the high-level server's ASGI response boundary."""
    attribute = (
        "http_app"
        if callable(getattr(server, "http_app", None))
        else "streamable_http_app"
    )
    factory = getattr(server, attribute, None)
    if not callable(factory):
        return

    @functools.wraps(factory)
    def application(*args: Any, **kwargs: Any) -> Any:
        app = factory(*args, **kwargs)
        app.add_middleware(McpClientIdentityMiddleware)
        return app

    setattr(server, attribute, application)
