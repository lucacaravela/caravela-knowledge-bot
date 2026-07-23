"""Affinity CRM integration for the Caravela knowledge bot.

Affinity exposes two APIs that share the same API key but use different
auth schemes:

* v2 (https://api.affinity.co/v2) - Bearer token auth. Companies, lists,
  field data.
* v1 (https://api.affinity.co) - HTTP Basic auth with an empty username
  and the API key as the password. Notes are only available here.

Every public function returns a plain string (formatted result or a
readable error message) so a failed call never kills the agent loop.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

V1_BASE = "https://api.affinity.co"
V2_BASE = "https://api.affinity.co/v2"

REQUEST_TIMEOUT = 30

# Caps so tool output does not explode the model context.
MAX_SEARCH_RESULTS = 20
MAX_NOTE_CHARS = 2_000
MAX_TOTAL_NOTE_CHARS = 15_000


def _api_key() -> str:
    key = os.environ.get("AFFINITY_API_KEY", "")
    if not key:
        raise RuntimeError("AFFINITY_API_KEY is not set")
    return key


def _v1_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against the v1 API (Basic auth, empty username)."""
    resp = requests.get(
        f"{V1_BASE}{path}",
        params=params,
        auth=("", _api_key()),
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _v2_get(path: str, params: Optional[dict] = None) -> Any:
    """GET against the v2 API (Bearer token auth)."""
    resp = requests.get(
        f"{V2_BASE}{path}",
        params=params,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _stringify(value: Any) -> str:
    """Flatten an Affinity field value (scalar, dict or list) into text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_stringify(v) for v in value]
        return ", ".join(p for p in parts if p)
    if isinstance(value, dict):
        # Common v2 shapes: {"type": ..., "data": ...}, {"text": ...},
        # {"name": ...}, dropdown options, person refs, etc.
        for key in ("data", "text", "name", "value", "term"):
            if key in value:
                return _stringify(value[key])
        first = value.get("firstName") or value.get("first_name")
        last = value.get("lastName") or value.get("last_name")
        if first or last:
            return " ".join(p for p in (first, last) if p)
        parts = [_stringify(v) for v in value.values()]
        return ", ".join(p for p in parts if p)
    return str(value)


def _extract_key_fields(fields: list) -> dict:
    """Pick sector / stage / status / owner style fields from a v2 field list."""
    wanted = {
        "sector": ("sector", "industry", "setor", "vertical", "category"),
        "stage": ("stage", "estagio", "estágio", "round"),
        "status": ("status",),
        "owner": ("owner", "responsavel", "responsável", "lead"),
    }
    found: dict = {}
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name", "")).lower()
        text = _stringify(field.get("value"))
        if not text:
            continue
        for key, needles in wanted.items():
            if key not in found and any(n in name for n in needles):
                found[key] = text
    return found


def _fetch_v2_fields(org_id: Any) -> list:
    """Fetch v2 field data for one company. Returns [] on any failure."""
    try:
        data = _v2_get(
            f"/companies/{org_id}",
            params={"fieldTypes": ["global", "list", "enriched"]},
        )
        return data.get("fields") or []
    except Exception:
        return []


def search_orgs(query: str, sector: Optional[str] = None) -> str:
    """Search Affinity organizations by name/keyword, optionally filter by sector.

    Args:
        query: Name or keyword to search for.
        sector: Optional sector/industry filter, matched against the
            organization's sector-like field values (case-insensitive).

    Returns:
        A formatted string with up to 20 matches (name, domain, id and key
        field values), or a readable error message.
    """
    try:
        data = _v1_get("/organizations", params={"term": query, "page_size": 50})
        orgs = data.get("organizations") or []
        if not orgs:
            return f"No organizations found in Affinity for query '{query}'."

        results = []
        for org in orgs:
            org_id = org.get("id")
            entry = {
                "id": org_id,
                "name": org.get("name") or "(no name)",
                "domain": org.get("domain")
                or ", ".join(org.get("domains") or [])
                or "-",
            }
            entry["fields"] = _extract_key_fields(_fetch_v2_fields(org_id))
            if sector:
                sector_val = entry["fields"].get("sector", "")
                blob = f"{sector_val} {entry['name']}".lower()
                if sector.lower() not in blob:
                    continue
            results.append(entry)
            if len(results) >= MAX_SEARCH_RESULTS:
                break

        if not results:
            return (
                f"Found {len(orgs)} organizations for '{query}' but none matched "
                f"sector filter '{sector}'. Try without the sector filter."
            )

        lines = [f"Found {len(results)} organizations for '{query}':"]
        for r in results:
            fields = r["fields"]
            extras = "; ".join(
                f"{k}: {v}" for k, v in fields.items()
            ) or "no field data"
            lines.append(
                f"- {r['name']} (id: {r['id']}, domain: {r['domain']}) — {extras}"
            )
        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error while searching '{query}': {e}"
    except Exception as e:
        return f"Error searching Affinity for '{query}': {e}"


def get_org_details(org_id: int) -> str:
    """Get full field data for one organization, plus its Affinity lists.

    Args:
        org_id: The Affinity organization/company id.

    Returns:
        A formatted string with all field values and list memberships
        (including status per list), or a readable error message.
    """
    try:
        company = _v2_get(
            f"/companies/{org_id}",
            params={"fieldTypes": ["global", "list", "enriched"]},
        )
        name = company.get("name") or f"Company {org_id}"
        domain = company.get("domain") or ", ".join(company.get("domains") or [])
        lines = [f"{name} (id: {org_id}, domain: {domain or '-'})"]

        fields = company.get("fields") or []
        if fields:
            lines.append("Fields:")
            for field in fields:
                if not isinstance(field, dict):
                    continue
                fname = field.get("name") or field.get("id") or "?"
                fval = _stringify(field.get("value"))
                if fval:
                    lines.append(f"  - {fname}: {fval}")
        else:
            lines.append("Fields: none available")

        # List memberships and per-list status.
        try:
            entries_data = _v2_get(f"/companies/{org_id}/list-entries")
            entries = entries_data.get("data") or entries_data.get("listEntries") or []
        except Exception:
            entries = []

        if entries:
            lines.append("Lists:")
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                lst = entry.get("list") or {}
                list_name = (
                    lst.get("name")
                    or entry.get("listName")
                    or f"list {entry.get('listId', '?')}"
                )
                status = ""
                for field in entry.get("fields") or []:
                    if isinstance(field, dict) and "status" in str(
                        field.get("name", "")
                    ).lower():
                        status = _stringify(field.get("value"))
                        break
                suffix = f" — status: {status}" if status else ""
                lines.append(f"  - {list_name}{suffix}")
        else:
            lines.append("Lists: not on any list (or list data unavailable)")

        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error fetching details for org {org_id}: {e}"
    except Exception as e:
        return f"Error fetching Affinity details for org {org_id}: {e}"


def get_notes(org_id: int) -> str:
    """Get all notes for an organization (newest first) via the v1 API.

    Notes include synced Granola meeting summaries. Each note is truncated
    and the total output is capped so the model context does not explode.

    Args:
        org_id: The Affinity organization id.

    Returns:
        A formatted string with the notes (author id + date + content),
        or a readable error message.
    """
    try:
        data = _v1_get("/notes", params={"organization_id": org_id})
        notes = data.get("notes") or []
        if not notes:
            return f"No notes found in Affinity for organization {org_id}."

        notes.sort(key=lambda n: n.get("created_at") or "", reverse=True)

        author_cache: dict = {}

        def author_name(creator_id: Any) -> str:
            if creator_id is None:
                return "unknown author"
            if creator_id not in author_cache:
                try:
                    person = _v1_get(f"/persons/{creator_id}")
                    full = " ".join(
                        p
                        for p in (person.get("first_name"), person.get("last_name"))
                        if p
                    )
                    author_cache[creator_id] = full or f"user {creator_id}"
                except Exception:
                    author_cache[creator_id] = f"user {creator_id}"
            return author_cache[creator_id]

        lines = [f"Notes for organization {org_id} (newest first):"]
        total = 0
        shown = 0
        for note in notes:
            content = (note.get("content") or "").strip()
            if not content:
                continue
            if len(content) > MAX_NOTE_CHARS:
                content = content[:MAX_NOTE_CHARS] + " [...note truncated]"
            date = (note.get("created_at") or "")[:10]
            block = f"--- {date} by {author_name(note.get('creator_id'))} ---\n{content}"
            if total + len(block) > MAX_TOTAL_NOTE_CHARS:
                lines.append(
                    f"[{len(notes) - shown} older note(s) omitted to stay within the size limit]"
                )
                break
            lines.append(block)
            total += len(block)
            shown += 1

        if shown == 0:
            return f"Organization {org_id} has {len(notes)} note(s) but all are empty."
        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error fetching notes for org {org_id}: {e}"
    except Exception as e:
        return f"Error fetching Affinity notes for org {org_id}: {e}"
