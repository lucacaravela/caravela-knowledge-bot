"""Unit tests for the WhatsApp webhook module (no real network calls)."""

import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import whatsapp_app


@pytest.fixture(autouse=True)
def whatsapp_env(monkeypatch):
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify-me")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "app-secret")
    monkeypatch.setenv("ALLOWED_PHONES", "+55 11 99999-9999, 5511888888888")
    whatsapp_app._seen_message_ids.clear()
    whatsapp_app._conversations.clear()


@pytest.fixture()
def client():
    return TestClient(whatsapp_app.app)


def _signed(body: dict) -> tuple:
    raw = json.dumps(body).encode()
    sig = "sha256=" + hmac.new(b"app-secret", raw, hashlib.sha256).hexdigest()
    return raw, {"X-Hub-Signature-256": sig, "Content-Type": "application/json"}


def _incoming(phone="5511999999999", text="oi", message_id="wamid.1"):
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "type": "text",
                                    "from": phone,
                                    "id": message_id,
                                    "text": {"body": text},
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }


def test_webhook_verification_handshake(client):
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-me",
            "hub.challenge": "12345",
        },
    )
    assert resp.status_code == 200
    assert resp.text == "12345"


def test_webhook_verification_rejects_wrong_token(client):
    resp = client.get(
        "/webhook",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong",
            "hub.challenge": "12345",
        },
    )
    assert resp.status_code == 403


def test_webhook_rejects_bad_signature(client):
    raw = json.dumps(_incoming()).encode()
    resp = client.post(
        "/webhook",
        content=raw,
        headers={"X-Hub-Signature-256": "sha256=deadbeef"},
    )
    assert resp.status_code == 403


def test_allowed_message_triggers_handler(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app, "handle_question", lambda phone, text: calls.append((phone, text))
    )
    raw, headers = _signed(_incoming(text="quais fintechs vimos?"))
    resp = client.post("/webhook", content=raw, headers=headers)
    assert resp.status_code == 200
    assert calls == [("5511999999999", "quais fintechs vimos?")]


def test_non_allowed_phone_is_ignored(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app, "handle_question", lambda phone, text: calls.append((phone, text))
    )
    raw, headers = _signed(_incoming(phone="5511777777777"))
    resp = client.post("/webhook", content=raw, headers=headers)
    assert resp.status_code == 200
    assert calls == []


def test_duplicate_message_id_processed_once(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app, "handle_question", lambda phone, text: calls.append(text)
    )
    raw, headers = _signed(_incoming(message_id="wamid.dup"))
    client.post("/webhook", content=raw, headers=headers)
    client.post("/webhook", content=raw, headers=headers)
    assert len(calls) == 1


def test_is_allowed_normalizes_formatting():
    assert whatsapp_app.is_allowed("5511999999999")
    assert whatsapp_app.is_allowed("+5511888888888")
    assert not whatsapp_app.is_allowed("5511000000000")


def test_markdown_to_whatsapp():
    text = "## Resumo\n**Bull** é forte.\n- item um\n- item dois\nCusto de \\$10"
    out = whatsapp_app.markdown_to_whatsapp(text)
    assert "*Resumo*" in out
    assert "*Bull*" in out
    assert "• item um" in out
    assert "\\$" not in out


def test_split_message_respects_paragraphs():
    paragraphs = [f"paragrafo {i} " + "x" * 500 for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = whatsapp_app.split_message(text, limit=1200)
    assert all(len(c) <= 1200 for c in chunks)
    assert "".join(c.replace("\n\n", "") for c in chunks).count("paragrafo") == 20


def test_split_message_short_text_single_chunk():
    assert whatsapp_app.split_message("oi") == ["oi"]


def test_twilio_webhook_triggers_handler(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app,
        "handle_question",
        lambda phone, text, sender=None, chunk_limit=0: calls.append((phone, text)),
    )
    resp = client.post(
        "/twilio-webhook",
        data={
            "From": "whatsapp:+5511999999999",
            "Body": "quais fintechs vimos?",
            "MessageSid": "SM123",
        },
    )
    assert resp.status_code == 200
    assert "<Response>" in resp.text
    assert calls == [("5511999999999", "quais fintechs vimos?")]


def test_twilio_webhook_ignores_non_allowed(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app,
        "handle_question",
        lambda phone, text, sender=None, chunk_limit=0: calls.append(phone),
    )
    resp = client.post(
        "/twilio-webhook",
        data={"From": "whatsapp:+5511777777777", "Body": "oi", "MessageSid": "SM124"},
    )
    assert resp.status_code == 200
    assert calls == []


def test_twilio_webhook_dedups_message_sid(client, monkeypatch):
    calls = []
    monkeypatch.setattr(
        whatsapp_app,
        "handle_question",
        lambda phone, text, sender=None, chunk_limit=0: calls.append(text),
    )
    payload = {"From": "whatsapp:+5511999999999", "Body": "oi", "MessageSid": "SM125"}
    client.post("/twilio-webhook", data=payload)
    client.post("/twilio-webhook", data=payload)
    assert len(calls) == 1


def test_send_twilio_message_posts_to_api(monkeypatch):
    import responses as responses_lib

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")
    with responses_lib.RequestsMock() as rsps:
        rsps.post(
            f"{whatsapp_app.TWILIO_API}/Accounts/AC123/Messages.json",
            json={"sid": "SM1"},
        )
        whatsapp_app.send_twilio_message("5511999999999", "ola")
        body = rsps.calls[0].request.body
    assert "whatsapp%3A%2B5511999999999" in body
    assert "ola" in body
