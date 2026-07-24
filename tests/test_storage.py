"""Unit tests for the Supabase storage module, with mocked HTTP responses."""

import pytest
import responses

import storage

BASE = "https://fake.supabase.co/rest/v1"


@pytest.fixture(autouse=True)
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://fake.supabase.co")
    monkeypatch.setenv("SUPABASE_SERVICE_KEY", "service-key")


def test_disabled_without_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL")
    assert not storage.enabled()
    assert storage.list_conversations("a@b.com") == []
    assert storage.create_conversation("a@b.com", "oi") is None
    assert storage.get_email_for_phone("5511999999999") is None


@responses.activate
def test_list_conversations_filters_by_email():
    responses.get(
        f"{BASE}/conversations",
        json=[{"id": "c1", "title": "Celes", "updated_at": "2026-07-24T00:00:00+00:00", "channel": "web"}],
    )
    convs = storage.list_conversations("luca@caravela.capital")
    assert convs[0]["id"] == "c1"
    assert "user_email=eq.luca%40caravela.capital" in responses.calls[0].request.url


@responses.activate
def test_create_conversation_returns_id_and_truncates_title():
    responses.post(f"{BASE}/conversations", json=[{"id": "c9"}])
    long_title = "pergunta muito longa " * 10
    conv_id = storage.create_conversation("a@b.com", long_title)
    assert conv_id == "c9"
    import json as _json

    body = _json.loads(responses.calls[0].request.body)
    assert len(body["title"]) <= storage.TITLE_MAX_CHARS


@responses.activate
def test_load_messages_checks_ownership():
    # Ownership check returns empty -> messages endpoint must not be hit.
    responses.get(f"{BASE}/conversations", json=[])
    result = storage.load_messages("c1", "intruder@evil.com")
    assert result == []
    assert len(responses.calls) == 1


@responses.activate
def test_load_messages_returns_role_content():
    responses.get(f"{BASE}/conversations", json=[{"id": "c1"}])
    responses.get(
        f"{BASE}/messages",
        json=[
            {"role": "user", "content": "oi"},
            {"role": "assistant", "content": "ola"},
        ],
    )
    result = storage.load_messages("c1", "a@b.com")
    assert result == [
        {"role": "user", "content": "oi"},
        {"role": "assistant", "content": "ola"},
    ]


@responses.activate
def test_append_messages_posts_and_touches():
    responses.post(f"{BASE}/messages", json=[{}])
    responses.patch(f"{BASE}/conversations", json=[{}])
    ok = storage.append_messages(
        "c1", [{"role": "user", "content": "oi"}, {"role": "assistant", "content": "ola"}]
    )
    assert ok
    assert len(responses.calls) == 2


@responses.activate
def test_delete_conversation_enforces_owner():
    responses.delete(f"{BASE}/conversations", json=[])
    assert storage.delete_conversation("c1", "a@b.com")
    url = responses.calls[0].request.url
    assert "id=eq.c1" in url
    assert "user_email=eq.a%40b.com" in url


@responses.activate
def test_latest_conversation_respects_max_age():
    from datetime import timedelta

    responses.get(
        f"{BASE}/conversations",
        json=[{"id": "c1", "title": "t", "updated_at": "2020-01-01T00:00:00+00:00"}],
    )
    assert storage.latest_conversation("a@b.com", "whatsapp", max_age=timedelta(hours=2)) is None


@responses.activate
def test_get_email_for_phone():
    responses.get(
        f"{BASE}/phone_mappings", json=[{"user_email": "luca@caravela.capital"}]
    )
    assert storage.get_email_for_phone("5511999818687") == "luca@caravela.capital"


@responses.activate
def test_storage_failures_degrade_gracefully():
    responses.get(f"{BASE}/conversations", status=500)
    assert storage.list_conversations("a@b.com") == []
