"""WhatsApp entry point for the Caravela knowledge bot (Meta Cloud API).

FastAPI webhook that receives WhatsApp messages, validates the sender
against a phone whitelist, sends an instant acknowledgment, runs the same
Claude agent loop used by the Streamlit app, and replies via the Graph API.

Run locally:  uvicorn whatsapp_app:app --reload --port 8000
Deploy:       see render.yaml (service caravela-whatsapp-bot)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

import anthropic
import requests
from fastapi import BackgroundTasks, FastAPI, Query, Request, Response

import agent

GRAPH_API = "https://graph.facebook.com/v21.0"

WHATSAPP_MAX_CHARS = 4_000  # hard limit is 4096; leave margin
HISTORY_TTL = 2 * 60 * 60  # forget a conversation after 2h of silence
HISTORY_MAX_TURNS = 40  # cap stored messages per phone
ACK_TEXT = "Consultando o Affinity..."

app = FastAPI(title="Caravela Knowledge Bot - WhatsApp")

_anthropic_client: Optional[anthropic.Anthropic] = None

# Per-phone conversation state and message dedup (Meta retries webhooks).
_conversations: dict = {}
_conversations_lock = threading.Lock()
_seen_message_ids: dict = {}


def _token() -> str:
    return os.environ.get("WHATSAPP_TOKEN", "")


def _phone_number_id() -> str:
    return os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")


def allowed_phones() -> set:
    """Whitelisted sender phones, digits only (e.g. '5511999999999')."""
    raw = os.environ.get("ALLOWED_PHONES", "")
    return {re.sub(r"\D", "", p) for p in raw.split(",") if re.sub(r"\D", "", p)}


def is_allowed(phone: str) -> bool:
    return re.sub(r"\D", "", phone or "") in allowed_phones()


def markdown_to_whatsapp(text: str) -> str:
    """Convert Claude's markdown to WhatsApp formatting.

    WhatsApp uses *bold*, _italic_, ~strike~ and ```mono```; it has no
    headers or tables, so headers become bold lines and table pipes are
    left as-is (Claude is told to prefer lists on this channel).
    """
    out = text
    # Headers -> bold line.
    out = re.sub(r"^#{1,6}\s*(.+)$", r"*\1*", out, flags=re.MULTILINE)
    # **bold** -> *bold* (do before single-asterisk handling).
    out = re.sub(r"\*\*(.+?)\*\*", r"*\1*", out, flags=re.DOTALL)
    # Markdown bullets: "- item" -> "• item".
    out = re.sub(r"^(\s*)-\s+", r"\1• ", out, flags=re.MULTILINE)
    # Escaped dollars from the web pipeline are not needed here.
    out = out.replace("\\$", "$")
    return out.strip()


def split_message(text: str, limit: int = WHATSAPP_MAX_CHARS) -> list:
    """Split long text into WhatsApp-sized chunks at paragraph boundaries."""
    if len(text) <= limit:
        return [text]
    chunks = []
    current = ""
    for paragraph in text.split("\n\n"):
        while len(paragraph) > limit:  # single huge paragraph
            chunks.append(paragraph[:limit])
            paragraph = paragraph[limit:]
        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) > limit:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def send_whatsapp_message(to: str, text: str) -> None:
    """Send one text message via the Graph API. Logs errors, never raises."""
    try:
        resp = requests.post(
            f"{GRAPH_API}/{_phone_number_id()}/messages",
            headers={"Authorization": f"Bearer {_token()}"},
            json={
                "messaging_product": "whatsapp",
                "to": to,
                "type": "text",
                "text": {"body": text, "preview_url": False},
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            print(f"[whatsapp] send failed {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[whatsapp] send error: {e}")


def _get_history(phone: str) -> list:
    with _conversations_lock:
        now = time.time()
        state = _conversations.get(phone)
        if state is None or now - state["updated"] > HISTORY_TTL:
            state = {"messages": [], "updated": now}
            _conversations[phone] = state
        state["updated"] = now
        return state["messages"]


def _trim_history(messages: list) -> None:
    while len(messages) > HISTORY_MAX_TURNS:
        messages.pop(0)
    # History must start with a user message.
    while messages and messages[0].get("role") != "user":
        messages.pop(0)


def _already_processed(message_id: str) -> bool:
    now = time.time()
    for mid in [m for m, ts in _seen_message_ids.items() if now - ts > 3600]:
        _seen_message_ids.pop(mid, None)
    if message_id in _seen_message_ids:
        return True
    _seen_message_ids[message_id] = now
    return False


def handle_question(
    phone: str,
    question: str,
    sender=None,
    chunk_limit: int = WHATSAPP_MAX_CHARS,
) -> None:
    """Run the agent loop for one incoming message and send the answer.

    Args:
        phone: Sender phone (digits, no 'whatsapp:' prefix).
        question: The user's message text.
        sender: Callable (to, text) used to deliver messages; defaults to
            the Meta Cloud API sender. The Twilio endpoint passes its own.
        chunk_limit: Max characters per outgoing message for this channel.
    """
    global _anthropic_client
    send = sender or send_whatsapp_message
    send(phone, ACK_TEXT)
    try:
        if _anthropic_client is None:
            _anthropic_client = anthropic.Anthropic()
        history = _get_history(phone)
        history.append({"role": "user", "content": question})
        answer = agent.answer_question(_anthropic_client, history)
        history[:] = agent.compact_history(history)
        _trim_history(history)
    except Exception as e:
        cause = getattr(e, "__cause__", None)
        print(
            f"[whatsapp] agent error for {phone}: {type(e).__name__}: {e!r}"
            + (f" (cause: {cause!r})" if cause else "")
        )
        answer = (
            "Desculpe, ocorreu um erro ao processar sua pergunta. "
            "Tente novamente em instantes."
        )
    for chunk in split_message(markdown_to_whatsapp(answer), limit=chunk_limit):
        send(phone, chunk)


def _valid_signature(body: bytes, signature_header: str) -> bool:
    """Check X-Hub-Signature-256 when WHATSAPP_APP_SECRET is configured."""
    secret = os.environ.get("WHATSAPP_APP_SECRET", "")
    if not secret:
        return True  # signature checking disabled
    if not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header[len("sha256="):])


@app.get("/")
def health() -> dict:
    return {"status": "ok", "service": "caravela-knowledge-bot-whatsapp"}


@app.get("/webhook")
def verify_webhook(
    mode: str = Query("", alias="hub.mode"),
    token: str = Query("", alias="hub.verify_token"),
    challenge: str = Query("", alias="hub.challenge"),
):
    """Meta's one-time webhook verification handshake."""
    if mode == "subscribe" and token == os.environ.get("WHATSAPP_VERIFY_TOKEN", ""):
        return Response(content=challenge, media_type="text/plain")
    return Response(status_code=403)


@app.post("/webhook")
async def receive_webhook(request: Request, background: BackgroundTasks):
    """Receive WhatsApp events; answer text messages from allowed phones."""
    body = await request.body()
    if not _valid_signature(body, request.headers.get("X-Hub-Signature-256", "")):
        return Response(status_code=403)

    try:
        payload = await request.json()
    except Exception:
        return {"status": "ignored"}

    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for message in value.get("messages", []) or []:
                if message.get("type") != "text":
                    continue
                phone = message.get("from", "")
                message_id = message.get("id", "")
                text = ((message.get("text") or {}).get("body") or "").strip()
                if not text or not phone:
                    continue
                if message_id and _already_processed(message_id):
                    continue
                if not is_allowed(phone):
                    print(f"[whatsapp] ignored message from non-allowed {phone}")
                    continue
                background.add_task(handle_question, phone, text)

    # Always 200 so Meta does not retry endlessly.
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Twilio WhatsApp (sandbox or production sender)
# ---------------------------------------------------------------------------

TWILIO_API = "https://api.twilio.com/2010-04-01"
TWILIO_MAX_CHARS = 1_500  # Twilio WhatsApp limit is 1600; leave margin


def _twilio_config() -> tuple:
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "")
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    from_number = os.environ.get("TWILIO_WHATSAPP_FROM", "whatsapp:+14155238886")
    return sid, token, from_number


def send_twilio_message(to: str, text: str) -> None:
    """Send one WhatsApp message via Twilio. Logs errors, never raises."""
    sid, token, from_number = _twilio_config()
    try:
        resp = requests.post(
            f"{TWILIO_API}/Accounts/{sid}/Messages.json",
            auth=(sid, token),
            data={
                "From": from_number,
                "To": f"whatsapp:+{re.sub(r'[^0-9]', '', to)}",
                "Body": text,
            },
            timeout=30,
        )
        if resp.status_code >= 400:
            print(f"[twilio] send failed {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        print(f"[twilio] send error: {e}")


@app.post("/twilio-webhook")
async def receive_twilio_webhook(request: Request, background: BackgroundTasks):
    """Receive an incoming WhatsApp message from Twilio (form-encoded)."""
    form = await request.form()
    from_raw = str(form.get("From", ""))  # e.g. 'whatsapp:+5511999999999'
    body = str(form.get("Body", "")).strip()
    message_sid = str(form.get("MessageSid", ""))
    phone = re.sub(r"\D", "", from_raw)

    if not body or not phone:
        return Response(content="<Response></Response>", media_type="text/xml")
    if message_sid and _already_processed(message_sid):
        return Response(content="<Response></Response>", media_type="text/xml")
    if not is_allowed(phone):
        print(f"[twilio] ignored message from non-allowed {phone}")
        return Response(content="<Response></Response>", media_type="text/xml")

    background.add_task(
        handle_question,
        phone,
        body,
        send_twilio_message,
        TWILIO_MAX_CHARS,
    )
    # Empty TwiML: we reply asynchronously via the REST API.
    return Response(content="<Response></Response>", media_type="text/xml")
