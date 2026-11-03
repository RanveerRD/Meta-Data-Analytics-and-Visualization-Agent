"""Minimal WhatsApp Cloud API client: send text only, no redirects followed."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

_TIMEOUT_S = 10
_MAX_BODY = 4000


class WhatsAppError(RuntimeError):
    """One friendly line; never contains the token."""


_GRAPH_ROOT = "https://graph.facebook.com"


class _Redirected(Exception):
    """Raised instead of following an HTTP redirect."""


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise _Redirected()


def _opener() -> urllib.request.OpenerDirector:
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    opener.add_handler(_NoRedirect())
    return opener


def split_body(body: str, limit: int = _MAX_BODY) -> list[str]:
    """Split a long body at line breaks; hard-cut an overlong single line."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in body.splitlines(keepends=True):
        while len(line) > limit:
            if current_len:
                chunks.append("".join(current))
                current, current_len = [], 0
            chunks.append(line[:limit])
            line = line[limit:]
        if current_len + len(line) > limit and current:
            chunks.append("".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line)
    if current:
        chunks.append("".join(current))
    return chunks or [""]


class WhatsAppClient:
    """Send plain-text WhatsApp messages via the Cloud API."""

    def __init__(self, api_version: str, phone_number_id: str, token: str) -> None:
        self.api_version = api_version
        self.phone_number_id = phone_number_id
        self._token = token

    def send_text(self, to: str, body: str) -> None:
        """Send body as one or more text messages."""
        for chunk in split_body(body):
            self._post(
                {
                    "messaging_product": "whatsapp",
                    "to": to,
                    "type": "text",
                    "text": {"body": chunk},
                }
            )

    def _post(self, payload: dict) -> None:
        url = (
            f"{_GRAPH_ROOT}/{self.api_version}"
            f"/{self.phone_number_id}/messages"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with _opener().open(request, timeout=_TIMEOUT_S) as response:
                response.read()
        except _Redirected:
            raise WhatsAppError(
                "WhatsApp redirected the request; check configuration"
            )
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise WhatsAppError("WhatsApp rejected the API token")
            raise WhatsAppError(f"WhatsApp refused the message (HTTP {exc.code})")
        except (TimeoutError, OSError) as exc:
            if isinstance(exc, TimeoutError) or isinstance(
                getattr(exc, "reason", None), TimeoutError
            ):
                raise WhatsAppError("WhatsApp request timed out")
            raise WhatsAppError("Could not reach WhatsApp to send a message")
