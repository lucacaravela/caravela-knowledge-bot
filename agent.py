"""Claude agentic loop for the Caravela knowledge bot.

Claude decides which tools to call (Affinity + Google Drive), the app
executes them, results go back to Claude, and the loop continues until
Claude produces a final text answer. Capped at MAX_TOOL_CALLS tool calls
per question.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Callable, Optional

import anthropic

import affinity
import drive

MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
MAX_TOKENS = 8_192
MAX_TOOL_CALLS = 10
MAX_API_RETRIES = 4

SYSTEM_PROMPT = """You are Caravela Capital's internal knowledge assistant. \
Caravela Capital is an early-stage VC fund investing in Latin America. Your \
users are members of the investment team asking about companies the fund has \
seen, sectors, deals, and internal documents.

You have live access to two sources:
1. Affinity CRM (the fund's dealflow database, including notes with synced \
Granola meeting summaries).
2. Google Drive (memos, decks, analyses and other internal documents).

Rules:
- Always ground your answers in tool results. Cite which company or which \
document each claim came from (e.g. "according to the notes on Acme" or \
"per the memo 'Healthcare LatAm 2024'").
- Never guess or invent companies, metrics, or document contents. If the \
tools return nothing, say clearly that nothing was found.
- For sector questions ("what fintech companies have we seen?"): use \
search_pipeline with the Portuguese Setor value (e.g. healthcare -> 'Saúde', \
education -> 'Educação'), then pull notes for the most relevant companies, \
then search Drive for related memos, and only then answer. Use search_orgs \
only when looking up companies by name.
- Dealflow context: the Pipeline list is the fund's dealflow. Its Status \
values run from 'Linkedin'/'New Lead'/'Pré-pipe' (early) through 'Analise \
Preliminar'/'Deep Dive'/'Termsheet' (active work) to 'Won'/'Pass'/'Lost'. \
The 'Investidas' list holds portfolio companies.
- Pass/lost reasons live in CRM COLUMNS, not in notes: 'Motivo lost' \
(short categories, already shown by search_pipeline) and 'Motivo Pass \
Detalhado' (the full pass rationale/email, returned by get_org_details). \
When asked why a company was passed, call get_org_details BEFORE answering; \
never claim a reason is not recorded based on notes alone.
- Be thorough in one pass: when a question needs details or notes for \
several companies, request ALL of them in a single turn (emit multiple tool \
calls together) instead of one per turn. Gather everything you need before \
answering so the user does not have to prompt you again for data you could \
have fetched.
- When a specific field/column the user asked about is empty, say exactly \
that ("a coluna X está vazia para essa empresa") instead of a vague \
'not found'.
- If a Drive search returns nothing, retry with 2 or 3 alternative phrasings \
before concluding there are no documents — try both Portuguese and English \
terms (e.g. "saúde" and "healthcare", "memo" and "tese").
- Answer in the language the user asked in. Users write in Portuguese and \
English interchangeably.
- Style: concise, objective, straight to the point. No emojis, ever. No \
filler, no enthusiasm, no closing offers like "se quiser, posso...". Lead \
with the direct answer, then supporting data. Use compact bullet lists or \
small tables; only include information that answers the question.
- If a piece of information is not in the tool results, state explicitly \
that it is not recorded in the CRM/Drive — never fill the gap with a \
plausible guess. Do not pad answers with general industry knowledge unless \
the user asks for it; if you do include it, label it clearly as general \
knowledge, not internal data.
- Structure for company answers: one-line description, then key data \
(sector, status, owner, dates), then takeaways.
- You have a budget of at most 10 tool calls per question — use them \
deliberately: search broadly first, then drill into only the most relevant \
companies and documents."""

TOOLS = [
    {
        "name": "search_pipeline",
        "description": (
            "Browse Caravela's dealflow (the Affinity 'Pipeline' list), newest "
            "companies first, optionally filtered by sector and/or status. This "
            "is the right tool for sector questions. Matching is accent- and "
            "case-insensitive substring ('saude' matches 'Saúde'). Setor values "
            "are in Portuguese; the main options: Agro, Alimentício, Beleza, "
            "Big Data, Biotecnologia, Construção, Crédito, Cripto/Blockchain, "
            "E-commerce, Educação, Energia, ESG, Fintech, Games, Gestão, "
            "Imobiliario, Jurídico, Logística, Marketing, Mobilidade, Pet, "
            "Produtividade, RH, Saúde, Segurança, Seguros, Serviços, Tecnologia, "
            "Varejo, Vendas, Wealthtech. Status options include: Linkedin, New "
            "Lead, Pré-pipe, Analise Preliminar, Deep Dive, Termsheet, Won, "
            "Pass, Lost, On Hold. Returns name, org id, domain, sector, status, "
            "owners and date added."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sector": {
                    "type": "string",
                    "description": "Optional Setor filter in Portuguese (e.g. 'Saúde', 'Fintech', 'Logística').",
                },
                "status": {
                    "type": "string",
                    "description": "Optional Status filter (e.g. 'Deep Dive', 'Termsheet', 'Won').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max companies to return (default 20, max 50).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_orgs",
        "description": (
            "Search organizations in Affinity CRM by NAME or name keyword. "
            "Returns up to 20 matches with name, domain, organization id and "
            "key field values (sector, status, owner). Use the returned id "
            "with get_org_details and get_notes. Note: this matches company "
            "names, not sectors — for sector questions use search_pipeline."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Name or keyword to search for (e.g. a company name or a sector keyword).",
                },
                "sector": {
                    "type": "string",
                    "description": "Optional sector/industry filter, matched against sector field values (e.g. 'healthcare', 'fintech').",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_org_details",
        "description": (
            "Get EVERY filled-in column for one Affinity organization "
            "(including 'Motivo lost', 'Motivo Pass Detalhado', Descrição, "
            "Blurb, País, Valor do Round, Source of Introduction, enriched "
            "data) plus every list it is on with per-list status. Required "
            "before answering why a company was passed or any question about "
            "its attributes. Call it for several companies in one turn when "
            "the question covers multiple companies."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "org_id": {
                    "type": "integer",
                    "description": "The Affinity organization id (from search_orgs).",
                }
            },
            "required": ["org_id"],
        },
    },
    {
        "name": "get_notes",
        "description": (
            "Get all notes for an Affinity organization, newest first, with "
            "author and date. Notes include synced Granola meeting summaries. "
            "Use this to understand what the team discussed with a company."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "org_id": {
                    "type": "integer",
                    "description": "The Affinity organization id (from search_orgs).",
                }
            },
            "required": ["org_id"],
        },
    },
    {
        "name": "search_drive",
        "description": (
            "Full-text search across Caravela's Google Drive (shared drives). "
            "Returns up to 15 matching files with name, id, type, modified "
            "date and link. If a search returns nothing, retry with 2-3 "
            "alternative phrasings in Portuguese and English."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search terms (e.g. 'healthcare memo', 'tese saúde').",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "read_drive_file",
        "description": (
            "Read the text content of a Google Drive file by id (from "
            "search_drive). Supports Google Docs, Sheets, Slides, PDFs and "
            "plain text files. Output is capped at ~15,000 characters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {
                    "type": "string",
                    "description": "The Drive file id (from search_drive).",
                }
            },
            "required": ["file_id"],
        },
    },
]

_TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "search_pipeline": affinity.search_pipeline,
    "search_orgs": affinity.search_orgs,
    "get_org_details": affinity.get_org_details,
    "get_notes": affinity.get_notes,
    "search_drive": drive.search_drive,
    "read_drive_file": drive.read_drive_file,
}


def execute_tool(name: str, tool_input: dict) -> str:
    """Run one tool and always return a string (never raise)."""
    func = _TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"Unknown tool: {name}"
    try:
        return func(**tool_input)
    except TypeError as e:
        return f"Invalid arguments for {name}: {e}"
    except Exception as e:
        return f"Unexpected error running {name}: {e}"


def _create_with_retry(client: anthropic.Anthropic, **kwargs):
    """Call the Messages API with exponential backoff on rate limits / overload."""
    last_exc: Exception | None = None
    for attempt in range(MAX_API_RETRIES):
        try:
            return client.messages.create(**kwargs)
        except anthropic.RateLimitError as e:
            last_exc = e
        except anthropic.APIStatusError as e:
            if e.status_code >= 500:  # includes 529 overloaded
                last_exc = e
            else:
                raise
        except anthropic.APIConnectionError as e:
            last_exc = e
        delay = min(2**attempt + random.uniform(0, 1), 30)
        time.sleep(delay)
    raise last_exc  # type: ignore[misc]


def _move_cache_marker(api_messages: list) -> None:
    """Keep a single prompt-cache breakpoint on the newest dict-based block.

    Each loop iteration resends the whole conversation; marking the latest
    tool_result block caches everything before it, so re-reads cost ~10% of
    normal input tokens. Older markers are removed (max 4 allowed per call).
    """
    last_block = None
    for msg in api_messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)
                last_block = block
    if last_block is not None:
        last_block["cache_control"] = {"type": "ephemeral"}


def answer_question(
    client: anthropic.Anthropic,
    api_messages: list,
    on_tool: Optional[Callable[[str, dict], None]] = None,
) -> str:
    """Run the agentic loop until Claude produces a final text answer.

    Args:
        client: An Anthropic client.
        api_messages: Full conversation history in Messages API format.
            The last entry must be the new user message. Assistant turns
            (including tool_use blocks) and tool results are appended in
            place so follow-up questions keep their context.
        on_tool: Optional callback invoked as on_tool(tool_name, tool_input)
            before each tool executes — used for the UI status indicator.

    Returns:
        Claude's final text answer.
    """
    tool_calls_used = 0

    while True:
        _move_cache_marker(api_messages)
        response = _create_with_retry(
            client,
            model=MODEL,
            max_tokens=MAX_TOKENS,
            # cache_control here caches the tools + system prompt prefix too.
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            tools=TOOLS,
            messages=api_messages,
        )

        api_messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(
                block.text for block in response.content if block.type == "text"
            )

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            tool_calls_used += 1
            if tool_calls_used > MAX_TOOL_CALLS:
                result = (
                    "Tool call budget exhausted (10 calls per question). "
                    "Answer now with the information gathered so far, and say "
                    "explicitly which parts could not be verified."
                )
            else:
                if on_tool is not None:
                    try:
                        on_tool(block.name, dict(block.input))
                    except Exception:
                        pass
                result = execute_tool(block.name, dict(block.input))
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                }
            )

        api_messages.append({"role": "user", "content": tool_results})


def format_tool_call(name: str, tool_input: dict) -> str:
    """Human-friendly one-liner describing a tool call, for the status UI."""
    args = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in tool_input.items())
    return f"{name}({args})"
