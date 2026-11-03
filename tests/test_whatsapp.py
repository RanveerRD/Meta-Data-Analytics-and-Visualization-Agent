"""WhatsApp front end: webhook, sessions, clarifications (no real network).

Most tests drive handle_get/handle_post/dispatch directly with a fake
client and stub answers, so they run unprivileged. The two tests that
bind loopback sockets skip when binding is unavailable and run under
one escalated invocation (same precedent as the live-Postgres tests).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib import request as urlrequest

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.whatsapp.client import WhatsAppClient, split_body  # noqa: E402
from agent.whatsapp.config import (  # noqa: E402
    ConfigError,
    WhatsAppConfig,
    load_config,
)
from agent.whatsapp.server import WhatsAppServer, format_turn_result  # noqa: E402

SENDER = "15551234567"
SECRET = "shh-secret"
OPTIONS = [
    {"label": "Alpha", "query_id": "q1", "evidence": ["a"], "explanation": "first"},
    {"label": "Beta", "query_id": "q1", "evidence": ["b"], "explanation": "second"},
]


def _config(**over: Any) -> WhatsAppConfig:
    base: dict[str, Any] = {
        "token": "tok-abc",
        "phone_number_id": "pid-1",
        "app_secret": SECRET,
        "verify_token": "vt-123",
        "allowed_numbers": frozenset({SENDER}),
        "api_version": "v22.0",
        "dataset": "olist",
    }
    base.update(over)
    return WhatsAppConfig(**base)


class _FakeClient:
    def __init__(self) -> None:
        self.sends: list[tuple[str, str]] = []
        self.lock = threading.Lock()
        self.fail: Exception | None = None

    def send_text(self, to: str, body: str) -> None:
        if self.fail is not None:
            raise self.fail
        with self.lock:
            self.sends.append((to, body))

    def bodies(self) -> list[str]:
        with self.lock:
            return [body for _to, body in self.sends]


def _answer(text: str, calls: list | None = None):
    def fn(history: list, question: str, ask_user: Any) -> str:
        if calls is not None:
            calls.append(question)
        return text

    return fn


def _payload(
    sender: str = SENDER,
    body: str = "hi",
    mid: str = "wamid.1",
    mtype: str = "text",
) -> dict:
    message: dict[str, Any] = {"from": sender, "id": mid, "type": mtype}
    if mtype == "text":
        message["text"] = {"body": body}
    return {"entry": [{"changes": [{"value": {"messages": [message]}}]}]}


def _sign(raw: bytes, secret: str = SECRET) -> str:
    return "sha256=" + hmac.new(secret.encode(), raw, hashlib.sha256).hexdigest()


def _post(
    server: WhatsAppServer,
    payload: Any,
    style: str = "good",
    secret: str = SECRET,
) -> int:
    raw = json.dumps(payload).encode()
    if style == "good":
        sig = _sign(raw, secret)
    elif style == "noprefix":
        sig = _sign(raw, secret).removeprefix("sha256=")
    elif style == "wrong":
        sig = "sha256=" + "0" * 64
    else:
        sig = None
    return server.handle_post(raw, sig)


def _wait_until(fn, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if fn():
            return True
        time.sleep(0.05)
    return False


def _server(answer=None, timeout: float = 300.0, **over: Any):
    client = _FakeClient()
    server = WhatsAppServer(
        _config(**over), client, answer or _answer("ok"), clarify_timeout=timeout
    )
    return server, client


def test_handshake():
    server, _client = _server()
    status, _ctype, body = server.handle_get(
        "hub.mode=subscribe&hub.verify_token=vt-123&hub.challenge=abc"
    )
    assert (status, body) == (200, "abc")
    assert server.handle_get(
        "hub.mode=subscribe&hub.verify_token=nope&hub.challenge=abc"
    )[0] == 403
    assert server.handle_get("hub.mode=subscribe")[0] == 403
    assert server.dispatch_get("/nope", "")[0] == 404


def test_signed_post_processed():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    assert _post(server, _payload(body="how many?")) == 200
    assert _wait_until(lambda: client.bodies() == ["done"])
    assert calls == ["how many?"]


def test_signature_variants_rejected():
    for style in ("noprefix", "wrong", "missing"):
        server, client = _server()
        calls: list = []
        server.answer_fn = _answer("done", calls)
        assert _post(server, _payload(), style=style) == 403
        assert calls == [] and client.bodies() == []
    server, _client = _server()
    assert server.handle_post(b"{}", None) == 403


def test_non_allowlisted_sender_ignored():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    assert _post(server, _payload(sender="19998887777")) == 200
    assert calls == [] and client.bodies() == []


def test_duplicate_message_processed_once():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    assert _post(server, _payload(mid="wamid.9")) == 200
    assert _post(server, _payload(mid="wamid.9")) == 200
    assert _wait_until(lambda: len(client.bodies()) == 1)
    time.sleep(0.3)
    assert calls == ["hi"] and len(client.bodies()) == 1


def test_non_text_and_statuses_ignored():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    assert _post(server, _payload(mtype="image")) == 200
    payload = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}
    assert _post(server, payload) == 200
    assert _post(server, {"entry": []}) == 200
    assert calls == [] and client.bodies() == []


def test_command_message_rejected():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    assert _post(server, _payload(body="/chart 1")) == 200
    assert _wait_until(
        lambda: client.bodies() == ["Commands are not available on WhatsApp yet."]
    )
    assert calls == []


def test_split_body():
    text = "".join(f"line {i}\n" for i in range(1000))
    assert len(text) > 4000
    chunks = split_body(text)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    assert "".join(chunks) == text
    assert split_body("x" * 5000) == ["x" * 4000, "x" * 1000]


def _ask_stub(seen: list, reply: str = "picked"):
    def fn(history: list, question: str, ask_user: Any) -> str:
        result = ask_user("Pick one?", OPTIONS)
        seen.append(result)
        return f"{reply}:{json.dumps(result, sort_keys=True)}"

    return fn


def test_clarification_round_trip():
    server, client = _server()
    seen: list = []
    server.answer_fn = _ask_stub(seen)
    assert _post(server, _payload(body="which?", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    prompt = client.bodies()[0]
    assert "1." in prompt and "2." in prompt and "0" in prompt
    assert _post(server, _payload(body="2", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert seen == [{"choice": "Beta"}]
    assert client.bodies()[-1].endswith('"choice": "Beta"}')


def test_zero_then_typed():
    server, client = _server()
    seen: list = []
    server.answer_fn = _ask_stub(seen)
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="0", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert client.bodies()[1] == "type your answer:"
    assert _post(server, _payload(body="hello typed", mid="wamid.c")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 3)
    assert seen == [{"choice": "hello typed"}]


def test_question_about_options():
    server, client = _server()
    seen: list = []
    server.answer_fn = _ask_stub(seen)
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="why these?", mid="wamid.b")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2)
    assert seen == [{"user_question": "why these?"}]


def test_clarification_timeout_abandons_turn():
    server, client = _server(timeout=0.2)

    def fn(history: list, question: str, ask_user: Any) -> str:
        history.append({"role": "note"})
        return ask_user("Pick?", OPTIONS)

    server.answer_fn = fn
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 2, timeout=10)
    assert client.bodies()[1] == (
        "I did not get an answer, so I stopped this question."
    )
    time.sleep(0.4)
    assert len(client.bodies()) == 2
    assert server.sessions[SENDER]["history"] == []


def test_overtake_next_message_first():
    server, _client = _server()
    session = {"history": [], "overtake": deque([("t", "m0", "old")])}
    from queue import Queue

    session["inbox"] = Queue()
    session["inbox"].put(("t2", "m1", "new"))
    assert server._next_message(session) == ("t", "m0", "old")
    assert server._next_message(session) == ("t2", "m1", "new")


def test_missing_config(monkeypatch):
    import agent.whatsapp.config as config_mod

    monkeypatch.setattr(config_mod, "_read_env_file_value", lambda name: "")
    with pytest.raises(ConfigError, match="WHATSAPP_TOKEN"):
        load_config({})
    env = {
        "WHATSAPP_TOKEN": "t",
        "WHATSAPP_PHONE_NUMBER_ID": "p",
        "WHATSAPP_APP_SECRET": "s",
        "WHATSAPP_VERIFY_TOKEN": "v",
        "WHATSAPP_ALLOWED_NUMBERS": "abc",
        "WHATSAPP_API_VERSION": "v22.0",
    }
    with pytest.raises(ConfigError, match="digits only"):
        load_config(env)
    env["WHATSAPP_ALLOWED_NUMBERS"] = "1555"
    with pytest.raises(ConfigError, match="unknown WHATSAPP_DATASET"):
        load_config({**env, "WHATSAPP_DATASET": "nope"})
    with pytest.raises(ConfigError, match="not supported"):
        load_config({**env, "WHATSAPP_DATASET": "all"})
    config = load_config(env)
    assert config.dataset == "olist"
    assert config.allowed_numbers == frozenset({"1555"})


def test_no_blocking_input_reachable(monkeypatch):
    def raising_input(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("input() reached from WhatsApp path")

    monkeypatch.setattr("builtins.input", raising_input)
    server, client = _server()
    seen: list = []
    options = [
        {**OPTIONS[0], "definition_key": "revenue", "last_used": True},
        OPTIONS[1],
    ]

    def fn(history: list, question: str, ask_user: Any) -> str:
        result = ask_user("Pick?", options)
        seen.append(result)
        return "final"

    server.answer_fn = fn
    assert _post(server, _payload(body="q", mid="wamid.a")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert _post(server, _payload(body="1", mid="wamid.b")) == 200
    assert _wait_until(lambda: client.bodies()[-1] == "final")
    assert seen == [{"choice": "Alpha"}]
    assert not any("Save this" in body for body in client.bodies())
    for name in ("server.py", "__main__.py", "client.py", "config.py"):
        source = (
            Path(__file__).resolve().parent.parent / "agent" / "whatsapp" / name
        ).read_text()
        assert "input(" not in source
        assert "offer_definition" not in source


def test_secrets_never_logged(tmp_path, capsys, caplog):
    import agent.whatsapp.client as client_mod

    caplog.set_level(logging.INFO, logger="agent.whatsapp.server")
    server, client = _server()
    server.answer_fn = _answer("ok")
    raw = json.dumps(_payload(body="hi")).encode()
    assert server.handle_post(raw, _sign(raw)) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    assert server.handle_post(raw, "sha256=" + "0" * 64) == 403
    client.fail = client_mod.WhatsAppError("boom")
    assert _post(server, _payload(body="again", mid="wamid.z")) == 200
    assert _wait_until(
        lambda: "boom" in caplog.text or client.bodies() == [], timeout=10
    )
    assert "tok-abc" not in capsys.readouterr().out
    assert "tok-abc" not in capsys.readouterr().err
    for secret in ("tok-abc", SECRET, "vt-123"):
        assert secret not in caplog.text


def test_oversize_body_rejected():
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("done", calls)
    raw = b"x" * (64 * 1024 + 1)
    status, _ctype, _body = server.dispatch_post("/webhook", len(raw), raw, _sign(raw))
    assert status == 413
    assert calls == []


def test_startup_line_and_logging():
    import logging as std_logging

    from agent.whatsapp.__main__ import (
        HOST,
        PORT,
        _LOG_FORMAT,
        configure_logging,
        startup_line,
    )

    assert HOST == "127.0.0.1" and PORT == 8080
    assert "asctime" in _LOG_FORMAT and "levelname" in _LOG_FORMAT
    config = _config()
    assert startup_line(config) == (
        "WhatsApp webhook ready on 127.0.0.1:8080, "
        "dataset=olist, 1 allowed number(s)"
    )
    root = std_logging.getLogger()
    saved = root.handlers[:]
    saved_level = root.level
    try:
        configure_logging()
        installed = [
            h for h in root.handlers
            if isinstance(h, std_logging.StreamHandler)
            and getattr(h.formatter, "_fmt", "") == _LOG_FORMAT
        ]
        assert installed
        assert root.level <= std_logging.INFO
    finally:
        root.handlers = saved
        root.level = saved_level


def test_accepted_message_logs_short_id_only(tmp_path, caplog):
    import logging as std_logging

    caplog.set_level(std_logging.INFO, logger="agent.whatsapp.server")
    server, client = _server()
    server.answer_fn = _answer("ok")
    assert _post(server, _payload(body="hello there")) == 200
    assert _wait_until(lambda: len(client.bodies()) >= 1)
    accepted = [
        r for r in caplog.records if "accepted" in r.getMessage()
    ]
    assert len(accepted) == 1
    assert accepted[0].getMessage() == "message from \u20264567 accepted"
    assert SENDER not in accepted[0].getMessage()
    assert "hello there" not in caplog.text
    for secret in ("tok-abc", SECRET, "vt-123"):
        assert secret not in caplog.text


def test_handler_timeout_default():
    from agent.whatsapp.server import _WebhookHandler

    assert _WebhookHandler.timeout == 10


def test_stalled_connection_closed(monkeypatch):
    import socket as socket_mod

    from agent.whatsapp.server import _WebhookHandler

    try:
        probe = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except OSError:
        pytest.skip("loopback unavailable")
    probe.server_close()
    monkeypatch.setattr(_WebhookHandler, "timeout", 0.5)
    server, _client = _server()
    thread = threading.Thread(
        target=server.serve, kwargs={"port": 0}, daemon=True
    )
    thread.start()
    try:
        assert _wait_until(lambda: server.httpd is not None)
        port = server.httpd.server_port
        sock = socket_mod.create_connection(("127.0.0.1", port), timeout=5)
        sock.settimeout(5)
        try:
            assert sock.recv(1) == b""
        finally:
            sock.close()
    finally:
        if server.httpd is not None:
            server.httpd.shutdown()


def test_format_turn_result():
    submission = {"answer": "x"}
    fake = type("R", (), {"text": "A\nChart: workspace/charts/a.png\nChart note: n\nB", "verification": "verification: 1 matched", "submission": submission})()
    assert format_turn_result(fake) == "A\nB\nverification: 1 matched"
    assert format_turn_result(type("R", (), {"submission": None, "text": "t", "verification": "v"})()) is None
    assert format_turn_result(None) is None


class _FakeGraph:
    """Loopback stand-in for the Graph API; skipped when bind is denied."""

    def __init__(self) -> None:
        self.posts: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                outer.posts.append(
                    {
                        "path": self.path,
                        "auth": self.headers.get("Authorization"),
                        "body": json.loads(raw),
                    }
                )
                payload = b'{"messages": [{"id": "wamid.x"}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        try:
            self.server = HTTPServer(("127.0.0.1", 0), Handler)
        except OSError:
            pytest.skip("loopback unavailable")
        self.base = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()


def test_graph_client_posts(monkeypatch):
    import agent.whatsapp.client as client_mod

    fake = _FakeGraph()
    try:
        monkeypatch.setattr(client_mod, "_GRAPH_ROOT", fake.base)
        client = WhatsAppClient("v22.0", "pid-9", "tok-xyz")
        long_body = "".join(f"line {i}\n" for i in range(600))
        client.send_text("15550001111", long_body)
    finally:
        fake.close()
    assert len(fake.posts) > 1
    assert all(len(p["body"]["text"]["body"]) <= 4000 for p in fake.posts)
    assert "".join(p["body"]["text"]["body"] for p in fake.posts) == long_body
    assert all(p["auth"] == "Bearer tok-xyz" for p in fake.posts)
    assert all(p["path"] == "/v22.0/pid-9/messages" for p in fake.posts)


def test_full_http_round_trip(monkeypatch):
    import urllib.error as urlerror

    try:
        probe = HTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    except OSError:
        pytest.skip("loopback unavailable")
    probe.server_close()
    server, client = _server()
    calls: list = []
    server.answer_fn = _answer("hello back", calls)
    thread = threading.Thread(
        target=server.serve, kwargs={"port": 0}, daemon=True
    )
    thread.start()
    assert _wait_until(lambda: server.httpd is not None)
    port = server.httpd.server_port
    base = f"http://127.0.0.1:{port}"
    try:
        with urlrequest.urlopen(base + "/webhook?hub.mode=subscribe"
                                "&hub.verify_token=vt-123&hub.challenge=zz") as resp:
            assert (resp.status, resp.read()) == (200, b"zz")
        raw = json.dumps(_payload(body="ping")).encode()
        req = urlrequest.Request(
            base + "/webhook", data=raw, method="POST",
            headers={"Content-Type": "application/json",
                     "X-Hub-Signature-256": _sign(raw)},
        )
        with urlrequest.urlopen(req) as resp:
            assert resp.status == 200
    except urlerror.URLError:
        pytest.skip("loopback unavailable")
    finally:
        server.httpd.shutdown()
    assert _wait_until(lambda: client.bodies() == ["hello back"])
    assert calls == ["ping"]
