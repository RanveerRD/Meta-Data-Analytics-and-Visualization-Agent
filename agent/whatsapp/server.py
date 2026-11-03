"""WhatsApp webhook server: one session per sender, work in background.

The HTTP layer only verifies, parses, and enqueues; each sender has one
worker thread that runs whole turns one at a time. Clarifications pause
the worker on a queue until that sender's next message arrives.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from agent.clarify import format_options_text, interpret_clarification_reply
from agent.whatsapp.client import WhatsAppClient, WhatsAppError
from agent.whatsapp.config import WhatsAppConfig

log = logging.getLogger(__name__)

_MAX_BODY = 64 * 1024
_DEDUP_KEEP = 500
_OWN_COMMAND_REPLY = "Commands are not available on WhatsApp yet."
_TIMEOUT_LINE = "I did not get an answer, so I stopped this question."
_FAILURE_LINE = "Sorry \u2014 I couldn't finish that question. Please try again."


def _short(value: object) -> str:
    """Last-4 masking for phone numbers and message ids in logs."""
    text = str(value)
    return "\u2026" + text[-4:] if len(text) > 4 else text


class _Abandoned(Exception):
    """A turn stopped cleanly (clarification timeout or send failure)."""


def format_turn_result(result: Any) -> str | None:
    """Compose the WhatsApp reply for a finished turn, or None on failure.

    Sends the answer text plus the verification line. Chart lines name
    files the user cannot see, so they are dropped. Never raises.
    """
    if result is None or getattr(result, "submission", None) is None:
        return None
    lines = [
        line
        for line in (getattr(result, "text", "") or "").splitlines()
        if not line.startswith(("Chart: ", "Chart note: "))
    ]
    text = "\n".join(lines).strip()
    verification = getattr(result, "verification", "") or ""
    if verification:
        text = f"{text}\n{verification}" if text else verification
    return text


class WhatsAppServer:
    """Loopback webhook server with per-sender sessions and workers."""

    def __init__(
        self,
        config: WhatsAppConfig,
        client: WhatsAppClient,
        answer_fn: Any,
        clarify_timeout: float = 300.0,
    ) -> None:
        self.config = config
        self.client = client
        self.answer_fn = answer_fn
        self.clarify_timeout = clarify_timeout
        self.sessions: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.seen: deque[str] = deque(maxlen=_DEDUP_KEEP)
        self.seen_set: set[str] = set()
        self.httpd: ThreadingHTTPServer | None = None

    # -- webhook entry points (fast; agent work stays on workers) --

    def handle_get(self, query: str) -> tuple[int, str, str]:
        """Meta verification handshake."""
        params = parse_qs(query)
        mode = (params.get("hub.mode") or [""])[0]
        token = (params.get("hub.verify_token") or [""])[0]
        challenge = (params.get("hub.challenge") or [""])[0]
        if mode == "subscribe" and hmac.compare_digest(token, self.config.verify_token):
            return 200, "text/plain", challenge
        return 403, "text/plain", "forbidden"

    def handle_post(self, raw: bytes, signature: str | None) -> int:
        """Verify, parse, enqueue. Returns the HTTP status."""
        if not self._valid_signature(raw, signature):
            log.warning("rejected unsigned webhook")
            return 403
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            log.warning("rejected unparseable webhook")
            return 200
        with self.lock:
            for sender, mid, body in self._extract(payload):
                if sender not in self.config.allowed_numbers:
                    continue
                if mid and mid in self.seen_set:
                    log.info("duplicate message %s ignored", _short(mid))
                    continue
                if mid:
                    if len(self.seen) == self.seen.maxlen:
                        self.seen_set.discard(self.seen[0])
                    self.seen.append(mid)
                    self.seen_set.add(mid)
                session = self.sessions.get(sender)
                if session is None:
                    session = {
                        "history": [],
                        "inbox": queue.Queue(),
                        "overtake": deque(),
                    }
                    session["worker"] = threading.Thread(
                        target=self._serve_sender,
                        args=(sender, session),
                        daemon=True,
                        name=f"wa-{_short(sender)}",
                    )
                    self.sessions[sender] = session
                    session["worker"].start()
                session["inbox"].put((time.monotonic(), mid, body))
        return 200

    # -- internals --

    def _valid_signature(self, raw: bytes, signature: str | None) -> bool:
        if not signature or not signature.startswith("sha256="):
            return False
        expected = hmac.new(
            self.config.app_secret.encode("utf-8"), raw, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest("sha256=" + expected, signature)

    @staticmethod
    def _extract(payload: Any) -> list[tuple[str, str, str]]:
        """Pull (sender, message id, text body) triples from a webhook."""
        items: list[tuple[str, str, str]] = []
        if not isinstance(payload, dict):
            return items
        for entry in payload.get("entry", []) or []:
            if not isinstance(entry, dict):
                continue
            for change in entry.get("changes", []) or []:
                if not isinstance(change, dict):
                    continue
                value = change.get("value", {}) or {}
                if not isinstance(value, dict):
                    continue
                for message in value.get("messages", []) or []:
                    if not isinstance(message, dict):
                        continue
                    if message.get("type") != "text":
                        continue
                    body = (message.get("text") or {}).get("body", "")
                    sender = message.get("from", "")
                    if not sender or not isinstance(body, str) or not body:
                        continue
                    items.append((sender, str(message.get("id", "")), body))
        return items

    def _send(self, sender: str, body: str) -> bool:
        try:
            self.client.send_text(sender, body)
        except WhatsAppError as exc:
            log.warning("send to %s failed: %s", _short(sender), exc)
            return False
        return True

    def _next_message(self, session: dict[str, Any]) -> tuple[float, str, str]:
        """Oldest pending message: the overtake stash first, else the inbox."""
        if session["overtake"]:
            return session["overtake"].popleft()
        return session["inbox"].get()

    def _asker(self, sender: str, session: dict[str, Any]) -> Any:
        """ask_clarification over WhatsApp: prompt, then block for a reply."""

        def ask(question: str, options: list[dict[str, Any]]) -> dict[str, str]:
            if not self._send(sender, format_options_text(question, options)):
                raise _Abandoned()
            asked_at = time.monotonic()
            deadline = asked_at + self.clarify_timeout
            typed = False
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    arrived, _mid, body = session["inbox"].get(timeout=remaining)
                except queue.Empty:
                    break
                if arrived < asked_at:
                    session["overtake"].append((arrived, _mid, body))
                    continue
                if typed:
                    # After "0" the next message is free text, exactly like
                    # the console's "type your answer:" input.
                    return {"choice": body.strip()}
                kind, value = interpret_clarification_reply(body, options)
                if kind == "choice":
                    return {"choice": value}
                if kind == "question":
                    return {"user_question": value}
                if body.strip() == "0":
                    typed = True
                if not self._send(sender, value):
                    raise _Abandoned()
            self._send(sender, _TIMEOUT_LINE)
            raise _Abandoned()

        return ask

    def _serve_sender(self, sender: str, session: dict[str, Any]) -> None:
        """One sender's turns, strictly one at a time, forever."""
        while True:
            _ts, _mid, body = self._next_message(session)
            if body.strip().startswith("/"):
                self._send(sender, _OWN_COMMAND_REPLY)
                continue
            snapshot = len(session["history"])
            try:
                reply = self.answer_fn(
                    session["history"], body, self._asker(sender, session)
                )
            except _Abandoned:
                del session["history"][snapshot:]
                continue
            except Exception:
                log.exception("answer failed for %s", _short(sender))
                del session["history"][snapshot:]
                self._send(sender, _FAILURE_LINE)
                continue
            if reply is None:
                del session["history"][snapshot:]
                self._send(sender, _FAILURE_LINE)
                continue
            self._send(sender, reply)

    # -- HTTP routing (thin wrappers; logic stays unit-testable) --

    def dispatch_get(self, path: str, query: str) -> tuple[int, str, str]:
        """Route a GET: (status, content-type, body)."""
        if path != "/webhook":
            return 404, "text/plain", "not found"
        return self.handle_get(query)

    def dispatch_post(
        self, path: str, length: int, raw: bytes, signature: str | None
    ) -> tuple[int, str, str]:
        """Route a POST: (status, content-type, body)."""
        if path != "/webhook":
            return 404, "text/plain", "not found"
        if length > _MAX_BODY:
            return 413, "text/plain", "body too large"
        status = self.handle_post(raw, signature)
        return status, "text/plain", "ok" if status == 200 else "error"

    # -- serving --

    def serve(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        """Bind loopback and serve until interrupted."""

        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "WhatsAppAgent/1"

            def _reply(self, status: int, content_type: str, body: str) -> None:
                raw = body.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                status, content_type, body = outer.dispatch_get(
                    parsed.path, parsed.query
                )
                self._reply(status, content_type, body)

            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = 0
                raw = self.rfile.read(length) if 0 < length <= _MAX_BODY else b""
                status, content_type, body = outer.dispatch_post(
                    urlparse(self.path).path,
                    length,
                    raw,
                    self.headers.get("X-Hub-Signature-256"),
                )
                self._reply(status, content_type, body)

            def log_message(self, *args: object) -> None:
                log.info("%s %s", self.command, self.path.split("?")[0])

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        log.info("WhatsApp webhook listening on %s:%d", host, port)
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.httpd.server_close()
