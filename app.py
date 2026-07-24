"""Caravela Capital internal knowledge chatbot (Streamlit).

Chat UI + Google login (Streamlit native OIDC auth) in front of a Claude
agentic loop that queries Affinity CRM and Google Drive live.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

load_dotenv()

import anthropic
import streamlit as st

import agent
import storage
from auth_secrets import ensure_auth_secrets, missing_auth_vars

st.set_page_config(page_title="Caravela Knowledge Bot", page_icon="🧭")

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def require_login() -> str:
    """Gate everything behind Google login; return the user's email."""
    if not ensure_auth_secrets():
        st.error(
            "Authentication is not configured. Missing environment variables: "
            + ", ".join(missing_auth_vars())
            + ". See the README for setup instructions."
        )
        st.stop()

    if not st.user.is_logged_in:
        st.title("🧭 Caravela Knowledge Bot")
        st.write("Faça login com sua conta Google da Caravela para continuar.")
        if st.button("Entrar com Google", type="primary"):
            st.login()
        st.stop()

    email = (st.user.email or "").lower()
    allowed_domain = os.environ.get("ALLOWED_DOMAIN", "").lower().lstrip("@")
    allowed_emails = {
        e.strip().lower()
        for e in os.environ.get("ALLOWED_EMAILS", "").split(",")
        if e.strip()
    }
    domain_ok = bool(allowed_domain) and email.endswith("@" + allowed_domain)
    if not (domain_ok or email in allowed_emails):
        st.title("🧭 Caravela Knowledge Bot")
        st.error(
            f"Acesso negado 😕 — a conta **{email or 'desconhecida'}** não "
            f"está autorizada. Faça login com sua conta "
            f"@{allowed_domain or '(ALLOWED_DOMAIN não configurado)'}."
        )
        if st.button("Sair e tentar com outra conta"):
            st.logout()
        st.stop()

    return email


# ---------------------------------------------------------------------------
# Chat state
# ---------------------------------------------------------------------------

def escape_dollars(text: str) -> str:
    """Stop Streamlit from rendering $...$ as LaTeX (e.g. 'R$800' amounts)."""
    return text.replace("$", "\\$")


def links_in_new_tab(text: str) -> str:
    """Turn markdown links and bare URLs into anchors that open in a new tab.

    Clicking a Drive link otherwise navigates away and drops the chat
    session (Streamlit history is per-session).
    """
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        r'<a href="\2" target="_blank">\1</a>',
        text,
    )
    text = re.sub(
        r'(?<![("\'>=\]])(https?://[^\s<>)\]]+)',
        r'<a href="\1" target="_blank">\1</a>',
        text,
    )
    return text


def render_answer(text: str) -> None:
    st.markdown(links_in_new_tab(escape_dollars(text)), unsafe_allow_html=True)


def init_state() -> None:
    if "api_messages" not in st.session_state:
        # Full Messages-API history (incl. tool_use / tool_result blocks).
        st.session_state.api_messages = []
    if "display_messages" not in st.session_state:
        # Simplified (role, text) pairs for rendering.
        st.session_state.display_messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None
    if "conversation_id" not in st.session_state:
        # Supabase conversation id (None until the first answer is saved).
        st.session_state.conversation_id = None


def reset_conversation() -> None:
    st.session_state.api_messages = []
    st.session_state.display_messages = []
    st.session_state.pending_question = None
    st.session_state.conversation_id = None


def open_conversation(conversation_id: str, email: str) -> None:
    """Load a saved conversation into the session."""
    messages = storage.load_messages(conversation_id, email)
    st.session_state.api_messages = [dict(m) for m in messages]
    st.session_state.display_messages = [
        (m["role"], m["content"]) for m in messages
    ]
    st.session_state.pending_question = None
    st.session_state.conversation_id = conversation_id


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(email: str) -> None:
    with st.sidebar:
        st.markdown("### 🧭 Caravela Knowledge Bot")
        st.caption(f"Logado como **{email}**")
        if st.button("Sair", use_container_width=True):
            st.logout()
        if st.button("✨ Nova conversa", use_container_width=True):
            reset_conversation()
            st.rerun()

        if storage.enabled():
            st.divider()
            st.markdown("**Suas conversas**")
            conversations = storage.list_conversations(email)
            if not conversations:
                st.caption("Nenhuma conversa salva ainda.")
            # Long lists scroll inside a fixed-height box instead of
            # stretching the sidebar forever.
            list_box = (
                st.container(height=380) if len(conversations) > 7 else st.container()
            )
            with list_box:
                for conv in conversations:
                    col_open, col_del = st.columns([5, 1])
                    is_current = conv["id"] == st.session_state.conversation_id
                    label = ("▶ " if is_current else "") + conv["title"]
                    if conv.get("channel") == "whatsapp":
                        label = "📱 " + label
                    if col_open.button(
                        label, key=f"conv_{conv['id']}", use_container_width=True
                    ):
                        open_conversation(conv["id"], email)
                        st.rerun()
                    if col_del.button("🗑️", key=f"del_{conv['id']}"):
                        storage.delete_conversation(conv["id"], email)
                        if is_current:
                            reset_conversation()
                        st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def save_turn(email: str, question: str, answer: str) -> None:
    """Persist a finished question/answer pair (no-op if storage is off)."""
    if not storage.enabled():
        return
    if st.session_state.conversation_id is None:
        st.session_state.conversation_id = storage.create_conversation(
            email, title=question, channel="web"
        )
    if st.session_state.conversation_id:
        storage.append_messages(
            st.session_state.conversation_id,
            [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        )


def run_turn(client: anthropic.Anthropic, email: str, question: str) -> None:
    st.session_state.display_messages.append(("user", question))
    st.session_state.api_messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status("Consultando Affinity e Google Drive...", expanded=True) as status:

                def on_tool(name: str, tool_input: dict) -> None:
                    status.write("🔧 " + agent.format_tool_call(name, tool_input))

                answer = agent.answer_question(
                    client, st.session_state.api_messages, on_tool=on_tool
                )
                status.update(label="Pronto ✅", state="complete", expanded=False)
            # Drop raw tool traffic so the next question doesn't pay to
            # re-read it; the final answers carry the substance.
            st.session_state.api_messages = agent.compact_history(
                st.session_state.api_messages
            )
            render_answer(answer)
            st.session_state.display_messages.append(("assistant", answer))
            save_turn(email, question, answer)
        except anthropic.AuthenticationError:
            st.error("Chave da API da Anthropic inválida. Verifique ANTHROPIC_API_KEY.")
        except (anthropic.RateLimitError, anthropic.APIStatusError, anthropic.APIConnectionError):
            st.error(
                "O serviço da Anthropic está indisponível ou sobrecarregado no "
                "momento, mesmo após algumas tentativas. Tente novamente em "
                "alguns instantes."
            )
        except Exception as e:
            st.error(f"Erro inesperado: {e}")


def main() -> None:
    email = require_login()
    init_state()
    render_sidebar(email)

    st.title("🧭 Caravela Knowledge Bot")
    st.caption(
        "Pergunte sobre empresas, setores e documentos internos. As respostas "
        "vêm do Affinity e do Google Drive em tempo real."
    )

    if not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY não está configurada. Veja o README.")
        st.stop()

    client = anthropic.Anthropic()

    for role, text in st.session_state.display_messages:
        with st.chat_message(role):
            if role == "assistant":
                render_answer(text)
            else:
                st.markdown(escape_dollars(text))

    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        run_turn(client, email, pending)

    question = st.chat_input("Ex.: quais empresas de fintech já vimos este ano?")
    if question:
        run_turn(client, email, question)


if __name__ == "__main__":
    # Streamlit runs this script with __name__ == "__main__" on every rerun.
    main()
