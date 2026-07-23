"""Caravela Capital internal knowledge chatbot (Streamlit).

Chat UI + Google login (Streamlit native OIDC auth) in front of a Claude
agentic loop that queries Affinity CRM and Google Drive live.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

import anthropic
import streamlit as st

import agent
from auth_secrets import ensure_auth_secrets, missing_auth_vars

st.set_page_config(page_title="Caravela Knowledge Bot", page_icon="🧭")

EXAMPLE_QUESTIONS = [
    "Quais empresas de healthcare SaaS já vimos, o que elas fazem e o que "
    "devo ter em mente ao falar com empresas parecidas?",
    "What fintech companies in our pipeline are in due diligence right now?",
    "Resuma as últimas notas de reunião sobre empresas de logística.",
]


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


def init_state() -> None:
    if "api_messages" not in st.session_state:
        # Full Messages-API history (incl. tool_use / tool_result blocks).
        st.session_state.api_messages = []
    if "display_messages" not in st.session_state:
        # Simplified (role, text) pairs for rendering.
        st.session_state.display_messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = None


def reset_conversation() -> None:
    st.session_state.api_messages = []
    st.session_state.display_messages = []
    st.session_state.pending_question = None


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(email: str) -> None:
    with st.sidebar:
        st.markdown("### 🧭 Caravela Knowledge Bot")
        st.caption(f"Logado como **{email}**")
        if st.button("Sair", use_container_width=True):
            st.logout()
        if st.button("🗑️ Nova conversa", use_container_width=True):
            reset_conversation()
            st.rerun()
        st.divider()
        st.markdown("**Exemplos de perguntas**")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(q, key=f"example_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def run_turn(client: anthropic.Anthropic, question: str) -> None:
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
            st.markdown(escape_dollars(answer))
            st.session_state.display_messages.append(("assistant", answer))
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
            st.markdown(escape_dollars(text))

    pending = st.session_state.pending_question
    if pending:
        st.session_state.pending_question = None
        run_turn(client, pending)

    question = st.chat_input("Ex.: quais empresas de fintech já vimos este ano?")
    if question:
        run_turn(client, question)


if __name__ == "__main__":
    # Streamlit runs this script with __name__ == "__main__" on every rerun.
    main()
