"""Persistent chat history on Supabase (PostgREST API) for the Caravela bot.

Talks to Supabase's auto-generated REST API with plain ``requests`` and the
service-role key (server-side only). Chats are private per user: every
query filters by the verified login email.

All functions degrade gracefully: when Supabase is not configured or a
call fails, they return None/[]/False so both apps fall back to the
previous in-memory behavior instead of breaking.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

REQUEST_TIMEOUT = 15
MAX_LOADED_MESSAGES = 200
TITLE_MAX_CHARS = 60


def enabled() -> bool:
    """True when Supabase credentials are configured."""
    return bool(
        os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY")
    )


def _base() -> str:
    return os.environ.get("SUPABASE_URL", "").rstrip("/") + "/rest/v1"


def _headers(write: bool = False) -> dict:
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if write:
        headers["Prefer"] = "return=representation"
    return headers


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_title(question: str) -> str:
    title = " ".join((question or "").split())
    if len(title) > TITLE_MAX_CHARS:
        title = title[: TITLE_MAX_CHARS - 1] + "…"
    return title or "Nova conversa"


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------

def list_conversations(user_email: str, limit: int = 20) -> list:
    """Most recent conversations for one user: [{id, title, updated_at, channel}]."""
    if not enabled():
        return []
    try:
        resp = requests.get(
            f"{_base()}/conversations",
            headers=_headers(),
            params={
                "user_email": f"eq.{user_email}",
                "select": "id,title,updated_at,channel",
                "order": "updated_at.desc",
                "limit": limit,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"[storage] list_conversations failed: {e}")
        return []


def create_conversation(
    user_email: str, title: str, channel: str = "web"
) -> Optional[str]:
    """Create a conversation and return its id (None on failure)."""
    if not enabled():
        return None
    try:
        resp = requests.post(
            f"{_base()}/conversations",
            headers=_headers(write=True),
            json={
                "user_email": user_email,
                "title": make_title(title),
                "channel": channel,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0]["id"] if rows else None
    except Exception as e:
        print(f"[storage] create_conversation failed: {e}")
        return None


def latest_conversation(
    user_email: str, channel: str, max_age: Optional[timedelta] = None
) -> Optional[dict]:
    """The user's most recent conversation on a channel, optionally only if fresh."""
    if not enabled():
        return None
    try:
        resp = requests.get(
            f"{_base()}/conversations",
            headers=_headers(),
            params={
                "user_email": f"eq.{user_email}",
                "channel": f"eq.{channel}",
                "select": "id,title,updated_at",
                "order": "updated_at.desc",
                "limit": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return None
        conv = rows[0]
        if max_age is not None:
            updated = datetime.fromisoformat(conv["updated_at"].replace("Z", "+00:00"))
            if datetime.now(timezone.utc) - updated > max_age:
                return None
        return conv
    except Exception as e:
        print(f"[storage] latest_conversation failed: {e}")
        return None


def delete_conversation(conversation_id: str, user_email: str) -> bool:
    """Delete one conversation (and its messages, via cascade). Ownership enforced."""
    if not enabled():
        return False
    try:
        resp = requests.delete(
            f"{_base()}/conversations",
            headers=_headers(),
            params={
                "id": f"eq.{conversation_id}",
                "user_email": f"eq.{user_email}",
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[storage] delete_conversation failed: {e}")
        return False


def _touch_conversation(conversation_id: str) -> None:
    try:
        requests.patch(
            f"{_base()}/conversations",
            headers=_headers(),
            params={"id": f"eq.{conversation_id}"},
            json={"updated_at": _now_iso()},
            timeout=REQUEST_TIMEOUT,
        )
    except Exception as e:
        print(f"[storage] touch failed: {e}")


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

def load_messages(conversation_id: str, user_email: str) -> list:
    """Messages of one conversation as [{'role', 'content'}], oldest first.

    Ownership is verified: returns [] if the conversation does not belong
    to user_email.
    """
    if not enabled():
        return []
    try:
        owner = requests.get(
            f"{_base()}/conversations",
            headers=_headers(),
            params={
                "id": f"eq.{conversation_id}",
                "user_email": f"eq.{user_email}",
                "select": "id",
                "limit": 1,
            },
            timeout=REQUEST_TIMEOUT,
        )
        owner.raise_for_status()
        if not owner.json():
            return []
        resp = requests.get(
            f"{_base()}/messages",
            headers=_headers(),
            params={
                "conversation_id": f"eq.{conversation_id}",
                "select": "role,content",
                "order": "created_at.asc,id.asc",
                "limit": MAX_LOADED_MESSAGES,
            },
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return [
            {"role": m["role"], "content": m["content"]} for m in resp.json()
        ]
    except Exception as e:
        print(f"[storage] load_messages failed: {e}")
        return []


def append_messages(conversation_id: str, messages: list) -> bool:
    """Append [{'role','content'}] to a conversation and bump updated_at."""
    if not enabled() or not messages:
        return False
    try:
        resp = requests.post(
            f"{_base()}/messages",
            headers=_headers(),
            json=[
                {
                    "conversation_id": conversation_id,
                    "role": m["role"],
                    "content": m["content"],
                }
                for m in messages
            ],
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        _touch_conversation(conversation_id)
        return True
    except Exception as e:
        print(f"[storage] append_messages failed: {e}")
        return False


# ---------------------------------------------------------------------------
# WhatsApp phone mapping
# ---------------------------------------------------------------------------

def get_email_for_phone(phone: str) -> Optional[str]:
    """Resolve a WhatsApp phone (digits only) to a team member's email."""
    if not enabled():
        return None
    try:
        resp = requests.get(
            f"{_base()}/phone_mappings",
            headers=_headers(),
            params={"phone": f"eq.{phone}", "select": "user_email", "limit": 1},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        rows = resp.json()
        return rows[0]["user_email"] if rows else None
    except Exception as e:
        print(f"[storage] get_email_for_phone failed: {e}")
        return None
