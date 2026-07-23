"""Unit tests for the Affinity module, with mocked HTTP responses."""

import responses

import affinity


V1 = affinity.V1_BASE
V2 = affinity.V2_BASE


def _mock_v2_company(rsps, org_id, fields=None, name="Acme", domain="acme.com"):
    rsps.get(
        f"{V2}/companies/{org_id}",
        json={"id": org_id, "name": name, "domain": domain, "fields": fields or []},
    )


@responses.activate
def test_search_orgs_returns_matches_with_fields():
    responses.get(
        f"{V1}/organizations",
        json={
            "organizations": [
                {"id": 1, "name": "HealthTech Co", "domain": "healthtech.com"},
                {"id": 2, "name": "FinCo", "domain": "finco.com"},
            ]
        },
    )
    _mock_v2_company(
        responses,
        1,
        fields=[
            {"name": "Sector", "value": {"type": "dropdown", "data": {"text": "Healthcare"}}},
            {"name": "Status", "value": "In DD"},
        ],
        name="HealthTech Co",
    )
    _mock_v2_company(responses, 2, fields=[], name="FinCo")

    result = affinity.search_orgs("health")
    assert "HealthTech Co" in result
    assert "id: 1" in result
    assert "sector: Healthcare" in result
    assert "status: In DD" in result
    assert "FinCo" in result


@responses.activate
def test_search_orgs_sector_filter():
    responses.get(
        f"{V1}/organizations",
        json={
            "organizations": [
                {"id": 1, "name": "HealthTech Co", "domain": "healthtech.com"},
                {"id": 2, "name": "FinCo", "domain": "finco.com"},
            ]
        },
    )
    _mock_v2_company(
        responses,
        1,
        fields=[{"name": "Industry", "value": "Healthcare SaaS"}],
        name="HealthTech Co",
    )
    _mock_v2_company(
        responses,
        2,
        fields=[{"name": "Industry", "value": "Fintech"}],
        name="FinCo",
    )

    result = affinity.search_orgs("co", sector="healthcare")
    assert "HealthTech Co" in result
    assert "FinCo" not in result


@responses.activate
def test_search_orgs_no_results():
    responses.get(f"{V1}/organizations", json={"organizations": []})
    result = affinity.search_orgs("nonexistent")
    assert "No organizations found" in result


@responses.activate
def test_search_orgs_http_error_returns_string():
    responses.get(f"{V1}/organizations", status=401)
    result = affinity.search_orgs("anything")
    assert "error" in result.lower()
    assert "401" in result


@responses.activate
def test_get_org_details_includes_fields_and_lists():
    _mock_v2_company(
        responses,
        42,
        fields=[
            {"name": "Sector", "value": {"data": {"text": "Logistics"}}},
            {"name": "Owner", "value": {"firstName": "Ana", "lastName": "Silva"}},
        ],
        name="LogiCo",
        domain="logico.com",
    )
    responses.get(
        f"{V2}/companies/42/list-entries",
        json={
            "data": [
                {
                    "list": {"name": "Dealflow"},
                    "fields": [{"name": "Status", "value": "First Meeting"}],
                }
            ]
        },
    )

    result = affinity.get_org_details(42)
    assert "LogiCo" in result
    assert "Sector: Logistics" in result
    assert "Owner: Ana Silva" in result
    assert "Dealflow" in result
    assert "status: First Meeting" in result


@responses.activate
def test_get_org_details_error_returns_string():
    responses.get(f"{V2}/companies/99", status=404)
    result = affinity.get_org_details(99)
    assert "error" in result.lower()


@responses.activate
def test_get_notes_sorted_newest_first_and_truncated():
    long_content = "x" * (affinity.MAX_NOTE_CHARS + 500)
    responses.get(
        f"{V1}/notes",
        json={
            "notes": [
                {"id": 1, "content": "older note", "created_at": "2024-01-01T00:00:00Z", "creator_id": 7},
                {"id": 2, "content": long_content, "created_at": "2025-06-01T00:00:00Z", "creator_id": 7},
            ]
        },
    )
    responses.get(
        f"{V1}/persons/7",
        json={"first_name": "Bruno", "last_name": "Costa"},
    )

    result = affinity.get_notes(10)
    assert "Bruno Costa" in result
    # Newest note (2025) appears before the older one (2024).
    assert result.index("2025-06-01") < result.index("older note")
    assert "[...note truncated]" in result


@responses.activate
def test_get_notes_empty():
    responses.get(f"{V1}/notes", json={"notes": []})
    result = affinity.get_notes(10)
    assert "No notes found" in result


@responses.activate
def test_get_notes_error_returns_string():
    responses.get(f"{V1}/notes", status=500)
    result = affinity.get_notes(10)
    assert "error" in result.lower()
