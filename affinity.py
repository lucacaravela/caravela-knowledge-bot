"""Affinity CRM integration for the Caravela knowledge bot.

Affinity exposes two APIs that share the same API key but use different
auth schemes:

* v2 (https://api.affinity.co/v2) - Bearer token auth. Companies, lists,
  list entries and field data. Note: ``fieldTypes`` only accepts
  ``enriched``, ``global`` and ``relationship-intelligence`` — list-scoped
  fields (Setor, Status, Owners) come via list-entry endpoints instead.
* v1 (https://api.affinity.co) - HTTP Basic auth with an empty username
  and the API key as the password. Notes and name search live here.

Caravela's dealflow is the "Pipeline" list (id 31953), whose entries are
returned newest-first. Its key list fields: Setor (field-394528, Portuguese
dropdown incl. 'Saúde', 'Fintech', ...), Status (field-278853) and Owners
(field-278854).

Every public function returns a plain string (formatted result or a
readable error message) so a failed call never kills the agent loop.
"""

from __future__ import annotations

import os
import time
import unicodedata
from typing import Any, Iterator, Optional

import requests

V1_BASE = "https://api.affinity.co"
V2_BASE = "https://api.affinity.co/v2"

REQUEST_TIMEOUT = 30

PIPELINE_LIST_ID = int(os.environ.get("AFFINITY_PIPELINE_LIST_ID", "31953"))
SETOR_FIELD_ID = "field-394528"
STATUS_FIELD_ID = "field-278853"
OWNERS_FIELD_ID = "field-278854"
MOTIVO_LOST_FIELD_ID = "field-316106"
PORTFOLIO_LIST_ID = int(os.environ.get("AFFINITY_PORTFOLIO_LIST_ID", "73539"))

# Caps so tool output does not explode the model context.
MAX_SEARCH_RESULTS = 20
MAX_NOTE_CHARS = 2_000
MAX_TOTAL_NOTE_CHARS = 15_000
MAX_FIELD_CHARS = 600
MAX_DETAILS_CHARS = 12_000

# Pipeline scan settings: pages of 100, newest entries first.
PIPELINE_PAGE_SIZE = 100
MAX_PIPELINE_SCAN = 3_000
PIPELINE_CACHE_TTL = 900  # seconds

_pipeline_cache: dict = {
    "fetched_at": 0.0,
    "entries": [],
    "next_url": None,
    "exhausted": False,
}


def _reset_pipeline_cache() -> None:
    """Clear the in-memory Pipeline cache (used by tests)."""
    _pipeline_cache.update(
        fetched_at=0.0, entries=[], next_url=None, exhausted=False
    )


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
    return _v2_get_url(f"{V2_BASE}{path}", params)


def _v2_get_url(url: str, params: Optional[dict] = None) -> Any:
    """GET an absolute v2 URL (used for pagination nextUrl links)."""
    resp = requests.get(
        url,
        params=params,
        headers={"Authorization": f"Bearer {_api_key()}"},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _norm(text: Any) -> str:
    """Lowercase and strip accents, so 'saude' matches 'Saúde'."""
    text = str(text or "")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


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
        # person refs, dropdown options, etc.
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


def _field_map(fields: Any) -> dict:
    """Turn a v2 field list into {field name: non-empty string value}."""
    out: dict = {}
    for field in fields or []:
        if not isinstance(field, dict):
            continue
        name = str(field.get("name") or field.get("id") or "?")
        text = _stringify(field.get("value"))
        if text and name not in out:
            out[name] = text
    return out


def _simplify_pipeline_entry(entry: dict) -> dict:
    entity = entry.get("entity") or {}
    fields = _field_map(entity.get("fields"))
    return {
        "org_id": entity.get("id"),
        "name": entity.get("name") or "(no name)",
        "domain": entity.get("domain")
        or ", ".join(entity.get("domains") or [])
        or "-",
        "sector": fields.get("Setor", ""),
        "status": fields.get("Status", ""),
        "owners": fields.get("Owners", ""),
        "motivo_lost": fields.get("Motivo lost", ""),
        "added": (entry.get("createdAt") or "")[:10],
    }


def _iter_pipeline_entries() -> Iterator[dict]:
    """Yield simplified Pipeline entries newest-first, fetching pages lazily.

    Pages already fetched are served from a module-level cache (TTL 15 min)
    shared across questions, so repeated sector searches are cheap.
    """
    cache = _pipeline_cache
    if time.time() - cache["fetched_at"] > PIPELINE_CACHE_TTL:
        cache.update(
            fetched_at=time.time(), entries=[], next_url=None, exhausted=False
        )

    index = 0
    while True:
        while index < len(cache["entries"]):
            yield cache["entries"][index]
            index += 1
        if cache["exhausted"] or len(cache["entries"]) >= MAX_PIPELINE_SCAN:
            return
        if cache["next_url"]:
            data = _v2_get_url(cache["next_url"])
        else:
            data = _v2_get(
                f"/lists/{PIPELINE_LIST_ID}/list-entries",
                params={
                    "limit": PIPELINE_PAGE_SIZE,
                    "fieldIds": [
                        SETOR_FIELD_ID,
                        STATUS_FIELD_ID,
                        OWNERS_FIELD_ID,
                        MOTIVO_LOST_FIELD_ID,
                    ],
                },
            )
        page = data.get("data") or []
        cache["entries"].extend(_simplify_pipeline_entry(e) for e in page)
        cache["next_url"] = (data.get("pagination") or {}).get("nextUrl")
        if not page or not cache["next_url"]:
            cache["exhausted"] = True


def search_pipeline(
    sector: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 20,
) -> str:
    """Browse Caravela's Pipeline dealflow list, newest first, with filters.

    Args:
        sector: Optional Setor filter (accent/case-insensitive substring,
            e.g. 'saude' matches 'Saúde').
        status: Optional Status filter (same matching, e.g. 'deep dive').
        limit: Maximum number of companies to return (default 20).

    Returns:
        A formatted string with matching companies (name, org id, domain,
        sector, status, owners, date added) plus a note on how much of the
        list was scanned, or a readable error message.
    """
    try:
        limit = max(1, min(int(limit or 20), 50))
        matches: list = []
        scanned = 0
        oldest = ""
        for entry in _iter_pipeline_entries():
            scanned += 1
            oldest = entry["added"] or oldest
            if sector and _norm(sector) not in _norm(entry["sector"]):
                continue
            if status and _norm(status) not in _norm(entry["status"]):
                continue
            matches.append(entry)
            if len(matches) >= limit:
                break

        filters = []
        if sector:
            filters.append(f"sector~'{sector}'")
        if status:
            filters.append(f"status~'{status}'")
        filter_desc = " and ".join(filters) or "no filters"

        coverage = (
            f"(scanned the {scanned} most recent Pipeline entries, back to "
            f"{oldest or '?'}; older entries were not scanned)"
        )
        if _pipeline_cache["exhausted"] and scanned >= len(_pipeline_cache["entries"]):
            coverage = f"(scanned the entire Pipeline list, {scanned} entries)"

        if not matches:
            return (
                f"No Pipeline companies matched {filter_desc} {coverage}. "
                "Check the sector spelling — Setor values are in Portuguese "
                "(e.g. 'Saúde', 'Fintech', 'Logística', 'Educação')."
            )

        lines = [f"{len(matches)} Pipeline companies for {filter_desc} {coverage}:"]
        for m in matches:
            details = "; ".join(
                f"{k}: {v}"
                for k, v in (
                    ("setor", m["sector"]),
                    ("status", m["status"]),
                    ("owners", m["owners"]),
                    ("motivo lost", m.get("motivo_lost", "")),
                    ("added", m["added"]),
                )
                if v
            )
            lines.append(
                f"- {m['name']} (id: {m['org_id']}, domain: {m['domain']}) — {details}"
            )
        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error while browsing the Pipeline list: {e}"
    except Exception as e:
        return f"Error browsing the Affinity Pipeline list: {e}"


def list_portfolio() -> str:
    """List every company on the 'Investidas' (portfolio) list.

    Returns:
        A formatted string with each portfolio company's name, org id,
        domain, date added and any filled-in list fields (e.g. Status),
        or a readable error message.
    """
    try:
        lines = []
        url: Optional[str] = None
        count = 0
        while True:
            if url:
                data = _v2_get_url(url)
            else:
                data = _v2_get(
                    f"/lists/{PORTFOLIO_LIST_ID}/list-entries",
                    params={"limit": PIPELINE_PAGE_SIZE},
                )
            for entry in data.get("data") or []:
                entity = entry.get("entity") or {}
                fields = _field_map(entity.get("fields"))
                extras = "; ".join(
                    f"{k}: {v[:80]}" for k, v in fields.items()
                    if k.lower() in ("status", "setor", "sector", "owners")
                )
                domain = entity.get("domain") or "-"
                added = (entry.get("createdAt") or "")[:10]
                suffix = f" — {extras}" if extras else ""
                lines.append(
                    f"- {entity.get('name', '(no name)')} (id: {entity.get('id')}, "
                    f"domain: {domain}, added: {added}){suffix}"
                )
                count += 1
            url = (data.get("pagination") or {}).get("nextUrl")
            if not url or count > 300:
                break
        if not lines:
            return "The Investidas (portfolio) list is empty or unavailable."
        return f"{count} portfolio companies (Investidas list):\n" + "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error listing the portfolio: {e}"
    except Exception as e:
        return f"Error listing the Affinity portfolio: {e}"


def search_persons(query: str) -> str:
    """Search people in Affinity by name, with their organizations.

    Args:
        query: Person name (or part of it) to search for.

    Returns:
        A formatted string with up to 10 matches (name, email, associated
        organizations), or a readable error message.
    """
    try:
        data = _v1_get("/persons", params={"term": query, "page_size": 10})
        persons = data.get("persons") or []
        if not persons:
            return f"No people found in Affinity for '{query}'."

        org_name_cache: dict = {}

        def org_name(org_id: Any) -> str:
            if org_id not in org_name_cache:
                try:
                    org = _v1_get(f"/organizations/{org_id}")
                    org_name_cache[org_id] = org.get("name") or f"org {org_id}"
                except Exception:
                    org_name_cache[org_id] = f"org {org_id}"
            return org_name_cache[org_id]

        lines = [f"Found {len(persons)} people for '{query}':"]
        for i, p in enumerate(persons[:10]):
            name = " ".join(
                x for x in (p.get("first_name"), p.get("last_name")) if x
            ) or "(no name)"
            email = p.get("primary_email") or ", ".join(p.get("emails") or []) or "-"
            org_ids = p.get("organization_ids")
            if org_ids is None and i < 5:
                # Search results omit organizations; fetch the person detail
                # for the top matches only.
                try:
                    org_ids = _v1_get(f"/persons/{p.get('id')}").get(
                        "organization_ids"
                    )
                except Exception:
                    org_ids = None
            org_ids = (org_ids or [])[:5]
            orgs = ", ".join(org_name(oid) for oid in org_ids) or "-"
            lines.append(
                f"- {name} (person id: {p.get('id')}, email: {email}) — "
                f"organizations: {orgs}"
            )
        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Affinity API error while searching people for '{query}': {e}"
    except Exception as e:
        return f"Error searching Affinity people for '{query}': {e}"


def _org_list_entries(org_id: Any) -> list:
    """Fetch v2 list entries (with fields) for one company. [] on failure."""
    try:
        data = _v2_get(f"/companies/{org_id}/list-entries")
        return data.get("data") or []
    except Exception:
        return []


def _extract_key_fields(entries: list) -> dict:
    """Pick sector/status/owner values from a company's list entries."""
    found: dict = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fields = _field_map(entry.get("fields"))
        for name, value in fields.items():
            lname = name.lower()
            if "setor" in lname or "sector" in lname:
                found.setdefault("sector", value)
            elif "status" in lname:
                found.setdefault("status", value)
            elif "owner" in lname:
                found.setdefault("owner", value)
    return found


def search_orgs(query: str, sector: Optional[str] = None) -> str:
    """Search Affinity organizations by name, optionally filter by sector.

    Name/keyword search via the v1 API, enriched with Pipeline field data
    (sector, status, owner) from the v2 API. For sector-wide questions
    prefer search_pipeline, which browses the dealflow list directly.

    Args:
        query: Name or keyword to search for (matches organization names).
        sector: Optional sector filter (accent/case-insensitive).

    Returns:
        A formatted string with up to 20 matches, or a readable error.
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
            entry["fields"] = _extract_key_fields(_org_list_entries(org_id))
            if sector:
                blob = _norm(f"{entry['fields'].get('sector', '')} {entry['name']}")
                if _norm(sector) not in blob:
                    continue
            results.append(entry)
            if len(results) >= MAX_SEARCH_RESULTS:
                break

        if not results:
            return (
                f"Found {len(orgs)} organizations for '{query}' but none matched "
                f"sector filter '{sector}'. Try without the sector filter, or "
                "use search_pipeline for sector-wide questions."
            )

        lines = [f"Found {len(results)} organizations for '{query}':"]
        for r in results:
            extras = "; ".join(
                f"{k}: {v}" for k, v in r["fields"].items()
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
        A formatted string with all non-empty global/enriched field values
        and every list the company is on (with per-list field values such
        as Status and Setor), or a readable error message.
    """
    try:
        company = _v2_get(
            f"/companies/{org_id}",
            params={"fieldTypes": ["global", "enriched", "relationship-intelligence"]},
        )
        name = company.get("name") or f"Company {org_id}"
        domain = company.get("domain") or ", ".join(company.get("domains") or [])
        lines = [f"{name} (id: {org_id}, domain: {domain or '-'})"]

        seen: set = set()

        def add_field_lines(fields: dict, indent: str = "  ") -> None:
            for fname, fval in fields.items():
                key = (fname, fval)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"{indent}- {fname}: {fval[:MAX_FIELD_CHARS]}")

        fields = _field_map(company.get("fields"))
        if fields:
            lines.append("Fields (all filled-in columns):")
            add_field_lines(fields)
        else:
            lines.append("Fields: none filled in")

        entries = _org_list_entries(org_id)
        if entries:
            lines.append("Lists:")
            for entry in entries:
                list_name = (
                    entry.get("listName")
                    or f"list {entry.get('listId', '?')}"
                )
                added = (entry.get("createdAt") or "")[:10]
                lines.append(f"  - {list_name} (added {added or '?'}):")
                add_field_lines(_field_map(entry.get("fields")), indent="      ")
        else:
            lines.append("Lists: not on any list (or list data unavailable)")

        result = "\n".join(lines)
        if len(result) > MAX_DETAILS_CHARS:
            result = result[:MAX_DETAILS_CHARS] + "\n[details truncated]"
        return result
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
        A formatted string with the notes (author + date + content),
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
