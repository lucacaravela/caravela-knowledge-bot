"""Google Drive integration for the Caravela knowledge bot.

Authenticates with a service account (JSON string or file path from env)
using read-only scope, and calls the Drive v3 REST API directly with
``requests`` so behavior is easy to test with mocked HTTP.

Every public function returns a plain string (formatted result or a
readable error message) so a failed call never kills the agent loop.
"""

from __future__ import annotations

import io
import json
import os
from typing import Optional

import requests

DRIVE_API = "https://www.googleapis.com/drive/v3"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

REQUEST_TIMEOUT = 60
MAX_SEARCH_RESULTS = 15
MAX_FILE_CHARS = 15_000

# Google Workspace mime types -> export mime type.
EXPORT_FORMATS = {
    "application/vnd.google-apps.document": "text/plain",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/plain",
}


def _load_credentials():
    """Build service-account credentials from env configuration.

    Reads GOOGLE_SERVICE_ACCOUNT_JSON (full JSON as one line) or
    GOOGLE_SERVICE_ACCOUNT_FILE (path to the JSON key file).
    """
    from google.oauth2 import service_account  # imported lazily for testability

    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    path = os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE", "").strip()
    if raw:
        info = json.loads(raw)
        creds = service_account.Credentials.from_service_account_info(info)
    elif path:
        creds = service_account.Credentials.from_service_account_file(path)
    else:
        raise RuntimeError(
            "Neither GOOGLE_SERVICE_ACCOUNT_JSON nor GOOGLE_SERVICE_ACCOUNT_FILE is set"
        )
    return creds.with_scopes(SCOPES)


def _get_access_token() -> str:
    """Return a fresh OAuth2 access token for the service account."""
    from google.auth.transport.requests import Request  # lazy import

    creds = _load_credentials()
    creds.refresh(Request())
    return creds.token


def _drive_get(path: str, params: Optional[dict] = None, stream: bool = False):
    resp = requests.get(
        f"{DRIVE_API}{path}",
        params=params,
        headers={"Authorization": f"Bearer {_get_access_token()}"},
        timeout=REQUEST_TIMEOUT,
        stream=stream,
    )
    resp.raise_for_status()
    return resp


def search_drive(query: str) -> str:
    """Full-text search across all shared drives the service account can see.

    Args:
        query: Free-text search terms.

    Returns:
        A formatted string with up to 15 matches (name, id, mimeType,
        modified date, link), or a readable error message.
    """
    try:
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        resp = _drive_get(
            "/files",
            params={
                "q": f"fullText contains '{escaped}' and trashed = false",
                "corpora": "allDrives",
                "supportsAllDrives": "true",
                "includeItemsFromAllDrives": "true",
                "pageSize": MAX_SEARCH_RESULTS,
                "fields": "files(id,name,mimeType,modifiedTime,webViewLink)",
            },
        )
        files = resp.json().get("files") or []
        if not files:
            return (
                f"No Google Drive files found for '{query}'. "
                "Try alternative phrasings (Portuguese and English) before "
                "concluding there are no documents."
            )
        lines = [f"Found {len(files)} Drive file(s) for '{query}':"]
        for f in files:
            lines.append(
                "- {name} (id: {id}, type: {mime}, modified: {mod})\n  {link}".format(
                    name=f.get("name", "(no name)"),
                    id=f.get("id", "?"),
                    mime=f.get("mimeType", "?"),
                    mod=(f.get("modifiedTime") or "")[:10],
                    link=f.get("webViewLink", ""),
                )
            )
        return "\n".join(lines)
    except requests.HTTPError as e:
        return f"Google Drive API error while searching '{query}': {e}"
    except Exception as e:
        return f"Error searching Google Drive for '{query}': {e}"


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _truncate(text: str) -> str:
    if len(text) > MAX_FILE_CHARS:
        return (
            text[:MAX_FILE_CHARS]
            + f"\n\n[Content truncated at {MAX_FILE_CHARS} characters]"
        )
    return text


def read_drive_file(file_id: str) -> str:
    """Read the content of a Drive file as plain text.

    Google Docs/Sheets/Slides are exported as text; PDFs are downloaded
    and extracted with pypdf; plain text files are downloaded directly.
    Output is capped at ~15,000 characters.

    Args:
        file_id: The Drive file id (from search_drive).

    Returns:
        The file's text content (possibly truncated), or a readable error
        message.
    """
    try:
        meta = _drive_get(
            f"/files/{file_id}",
            params={"fields": "id,name,mimeType", "supportsAllDrives": "true"},
        ).json()
        name = meta.get("name", file_id)
        mime = meta.get("mimeType", "")

        if mime in EXPORT_FORMATS:
            resp = _drive_get(
                f"/files/{file_id}/export",
                params={"mimeType": EXPORT_FORMATS[mime]},
            )
            text = resp.text
        elif mime == "application/pdf":
            resp = _drive_get(
                f"/files/{file_id}",
                params={"alt": "media", "supportsAllDrives": "true"},
            )
            text = _extract_pdf_text(resp.content)
        elif mime.startswith("text/") or mime in (
            "application/json",
            "application/csv",
        ):
            resp = _drive_get(
                f"/files/{file_id}",
                params={"alt": "media", "supportsAllDrives": "true"},
            )
            text = resp.text
        else:
            return (
                f"File '{name}' has unsupported type '{mime}'. "
                "Only Google Docs/Sheets/Slides, PDFs and plain text files "
                "can be read."
            )

        text = (text or "").strip()
        if not text:
            return f"File '{name}' appears to be empty or contains no extractable text."
        return f"Content of '{name}':\n\n{_truncate(text)}"
    except requests.HTTPError as e:
        return f"Google Drive API error while reading file {file_id}: {e}"
    except Exception as e:
        return f"Error reading Google Drive file {file_id}: {e}"
