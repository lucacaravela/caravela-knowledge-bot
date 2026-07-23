"""Unit tests for the Google Drive module, with mocked HTTP responses."""

import pytest
import responses

import drive


@pytest.fixture(autouse=True)
def fake_token(monkeypatch):
    """Bypass the real service-account token exchange in every test."""
    monkeypatch.setattr(drive, "_get_access_token", lambda: "fake-token")


API = drive.DRIVE_API


@responses.activate
def test_search_drive_returns_matches():
    responses.get(
        f"{API}/files",
        json={
            "files": [
                {
                    "id": "abc123",
                    "name": "Healthcare LatAm Memo",
                    "mimeType": "application/vnd.google-apps.document",
                    "modifiedTime": "2025-03-10T12:00:00Z",
                    "webViewLink": "https://docs.google.com/document/d/abc123",
                }
            ]
        },
    )
    result = drive.search_drive("healthcare memo")
    assert "Healthcare LatAm Memo" in result
    assert "abc123" in result
    assert "2025-03-10" in result


@responses.activate
def test_search_drive_query_params_include_all_drives():
    responses.get(f"{API}/files", json={"files": []})
    drive.search_drive("it's a test")
    request = responses.calls[0].request
    assert "corpora=allDrives" in request.url
    assert "supportsAllDrives=true" in request.url
    assert "includeItemsFromAllDrives=true" in request.url
    # Single quote in the query must be escaped in the q parameter.
    assert "it%5C%27s" in request.url or "it\\'s" in request.url


@responses.activate
def test_search_drive_no_results_suggests_retry():
    responses.get(f"{API}/files", json={"files": []})
    result = drive.search_drive("nada")
    assert "No Google Drive files found" in result
    assert "alternative phrasings" in result


@responses.activate
def test_search_drive_error_returns_string():
    responses.get(f"{API}/files", status=403)
    result = drive.search_drive("query")
    assert "error" in result.lower()
    assert "403" in result


@responses.activate
def test_read_google_doc_exports_plain_text():
    responses.get(
        f"{API}/files/doc1",
        json={
            "id": "doc1",
            "name": "Tese Healthcare",
            "mimeType": "application/vnd.google-apps.document",
        },
    )
    responses.get(f"{API}/files/doc1/export", body="Conteudo do memo.")
    result = drive.read_drive_file("doc1")
    assert "Tese Healthcare" in result
    assert "Conteudo do memo." in result


@responses.activate
def test_read_plain_text_file_truncates():
    responses.get(
        f"{API}/files/txt1",
        json={"id": "txt1", "name": "notes.txt", "mimeType": "text/plain"},
    )
    responses.get(f"{API}/files/txt1", body="y" * (drive.MAX_FILE_CHARS + 100))
    result = drive.read_drive_file("txt1")
    assert "Content truncated" in result


@responses.activate
def test_read_unsupported_type_returns_message():
    responses.get(
        f"{API}/files/zip1",
        json={"id": "zip1", "name": "backup.zip", "mimeType": "application/zip"},
    )
    result = drive.read_drive_file("zip1")
    assert "unsupported type" in result


@responses.activate
def test_read_drive_file_error_returns_string():
    responses.get(f"{API}/files/gone", status=404)
    result = drive.read_drive_file("gone")
    assert "error" in result.lower()


def test_extract_pdf_text_roundtrip():
    """PDF extraction path: build a tiny PDF with pypdf and read it back."""
    import io

    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)

    # A blank page has no text; the function should still return a string.
    text = drive._extract_pdf_text(buf.getvalue())
    assert isinstance(text, str)
