"""Unit tests for the Affinity module, with mocked HTTP responses."""

import pytest
import responses

import affinity


V1 = affinity.V1_BASE
V2 = affinity.V2_BASE
PIPELINE = affinity.PIPELINE_LIST_ID


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    affinity._reset_pipeline_cache()
    yield
    affinity._reset_pipeline_cache()


def _entry(org_id, name, sector="", status="", motivo="", created="2026-01-10T00:00:00Z"):
    fields = []
    if sector:
        fields.append(
            {"id": "field-394528", "name": "Setor", "value": {"type": "dropdown", "data": {"text": sector}}}
        )
    if status:
        fields.append(
            {"id": "field-278853", "name": "Status", "value": {"type": "ranked-dropdown", "data": {"text": status}}}
        )
    if motivo:
        fields.append(
            {"id": "field-316106", "name": "Motivo lost", "value": {"type": "dropdown-multi", "data": [{"text": motivo}]}}
        )
    return {
        "id": org_id * 10,
        "listId": PIPELINE,
        "createdAt": created,
        "entity": {"id": org_id, "name": name, "domain": f"{name.lower()}.com", "fields": fields},
    }


def _mock_org_list_entries(rsps, org_id, sector="", status=""):
    fields = []
    if sector:
        fields.append({"name": "Setor", "value": {"data": {"text": sector}}})
    if status:
        fields.append({"name": "Status", "value": {"data": {"text": status}}})
    rsps.get(
        f"{V2}/companies/{org_id}/list-entries",
        json={"data": [{"listId": PIPELINE, "listName": "Pipeline", "fields": fields}]},
    )


# ---------------------------------------------------------------------------
# search_pipeline
# ---------------------------------------------------------------------------

@responses.activate
def test_search_pipeline_filters_by_sector_accent_insensitive():
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={
            "data": [
                _entry(1, "HealthCo", sector="Saúde", status="Deep Dive"),
                _entry(2, "FinCo", sector="Fintech", status="New Lead"),
                _entry(3, "MedCo", sector="Saúde", status="Pass"),
            ],
            "pagination": {"nextUrl": None},
        },
    )
    result = affinity.search_pipeline(sector="saude")
    assert "HealthCo" in result
    assert "MedCo" in result
    assert "FinCo" not in result
    assert "entire Pipeline list" in result


@responses.activate
def test_search_pipeline_includes_motivo_lost():
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={
            "data": [
                _entry(1, "MedCo", sector="Saúde", status="Pass", motivo="Tamanho de Mercado"),
            ],
            "pagination": {"nextUrl": None},
        },
    )
    result = affinity.search_pipeline(sector="saude", status="pass")
    assert "motivo lost: Tamanho de Mercado" in result


@responses.activate
def test_search_pipeline_filters_by_status():
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={
            "data": [
                _entry(1, "HealthCo", sector="Saúde", status="Deep Dive"),
                _entry(2, "FinCo", sector="Fintech", status="New Lead"),
            ],
            "pagination": {"nextUrl": None},
        },
    )
    result = affinity.search_pipeline(status="deep dive")
    assert "HealthCo" in result
    assert "FinCo" not in result


@responses.activate
def test_search_pipeline_paginates_with_next_url():
    next_url = f"{V2}/lists/{PIPELINE}/list-entries?cursor=abc"
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={
            "data": [_entry(1, "FinCo", sector="Fintech")],
            "pagination": {"nextUrl": next_url},
        },
    )
    responses.get(
        next_url,
        json={
            "data": [_entry(2, "HealthCo", sector="Saúde")],
            "pagination": {"nextUrl": None},
        },
    )
    result = affinity.search_pipeline(sector="Saúde")
    assert "HealthCo" in result


@responses.activate
def test_search_pipeline_uses_cache_on_second_call():
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={
            "data": [_entry(1, "HealthCo", sector="Saúde")],
            "pagination": {"nextUrl": None},
        },
    )
    affinity.search_pipeline(sector="Saúde")
    # Second call must not hit the network again (only one mock registered).
    result = affinity.search_pipeline(sector="Fintech")
    assert "No Pipeline companies matched" in result
    assert len(responses.calls) == 1


@responses.activate
def test_search_pipeline_no_match_mentions_portuguese():
    responses.get(
        f"{V2}/lists/{PIPELINE}/list-entries",
        json={"data": [_entry(1, "FinCo", sector="Fintech")], "pagination": {"nextUrl": None}},
    )
    result = affinity.search_pipeline(sector="healthcare")
    assert "No Pipeline companies matched" in result
    assert "Portuguese" in result


@responses.activate
def test_search_pipeline_error_returns_string():
    responses.get(f"{V2}/lists/{PIPELINE}/list-entries", status=500)
    result = affinity.search_pipeline(sector="Saúde")
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# search_orgs
# ---------------------------------------------------------------------------

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
    _mock_org_list_entries(responses, 1, sector="Saúde", status="Deep Dive")
    _mock_org_list_entries(responses, 2)

    result = affinity.search_orgs("health")
    assert "HealthTech Co" in result
    assert "id: 1" in result
    assert "sector: Saúde" in result
    assert "status: Deep Dive" in result
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
    _mock_org_list_entries(responses, 1, sector="Saúde")
    _mock_org_list_entries(responses, 2, sector="Fintech")

    result = affinity.search_orgs("co", sector="saude")
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


# ---------------------------------------------------------------------------
# list_portfolio / search_persons
# ---------------------------------------------------------------------------

@responses.activate
def test_list_portfolio_returns_companies():
    responses.get(
        f"{V2}/lists/{affinity.PORTFOLIO_LIST_ID}/list-entries",
        json={
            "data": [
                {
                    "createdAt": "2021-03-01T00:00:00Z",
                    "entity": {
                        "id": 5,
                        "name": "Mottu",
                        "domain": "mottu.com.br",
                        "fields": [
                            {"name": "Status", "value": {"data": {"text": "Won"}}}
                        ],
                    },
                }
            ],
            "pagination": {"nextUrl": None},
        },
    )
    result = affinity.list_portfolio()
    assert "Mottu" in result
    assert "Status: Won" in result
    assert "1 portfolio companies" in result


@responses.activate
def test_list_portfolio_error_returns_string():
    responses.get(f"{V2}/lists/{affinity.PORTFOLIO_LIST_ID}/list-entries", status=500)
    result = affinity.list_portfolio()
    assert "error" in result.lower()


@responses.activate
def test_search_persons_with_organizations():
    responses.get(
        f"{V1}/persons",
        json={
            "persons": [
                {
                    "id": 9,
                    "first_name": "Fermin",
                    "last_name": "Eguren",
                    "primary_email": "fermin@anngel.mx",
                    "organization_ids": [308645425],
                }
            ]
        },
    )
    responses.get(f"{V1}/organizations/308645425", json={"id": 308645425, "name": "Anngel"})
    result = affinity.search_persons("Fermin")
    assert "Fermin Eguren" in result
    assert "fermin@anngel.mx" in result
    assert "Anngel" in result


@responses.activate
def test_search_persons_no_results():
    responses.get(f"{V1}/persons", json={"persons": []})
    result = affinity.search_persons("Zezinho")
    assert "No people found" in result


# ---------------------------------------------------------------------------
# get_org_details
# ---------------------------------------------------------------------------

@responses.activate
def test_get_org_details_includes_fields_and_lists():
    responses.get(
        f"{V2}/companies/42",
        json={
            "id": 42,
            "name": "LogiCo",
            "domain": "logico.com",
            "fields": [
                {"name": "Blurb", "value": {"type": "text", "data": "Freight marketplace"}},
                {"name": "Description", "value": {"type": "text", "data": None}},
            ],
        },
    )
    responses.get(
        f"{V2}/companies/42/list-entries",
        json={
            "data": [
                {
                    "listId": PIPELINE,
                    "listName": "Pipeline",
                    "createdAt": "2026-05-01T00:00:00Z",
                    "fields": [
                        {"name": "Status", "value": {"data": {"text": "First Meeting"}}},
                        {"name": "Setor", "value": {"data": {"text": "Logística"}}},
                        {"name": "Motivo lost", "value": {"data": {"text": "Round já fechado"}}},
                        {"name": "Blurb", "value": {"data": "Freight marketplace"}},
                    ],
                }
            ]
        },
    )

    result = affinity.get_org_details(42)
    assert "LogiCo" in result
    assert "Blurb: Freight marketplace" in result
    assert "Description" not in result  # empty values are dropped
    assert "Pipeline (added 2026-05-01)" in result
    assert "Status: First Meeting" in result
    assert "Setor: Logística" in result
    # Every non-empty list column is included, deduplicated.
    assert "Motivo lost: Round já fechado" in result
    assert result.count("Freight marketplace") == 1


@responses.activate
def test_get_org_details_error_returns_string():
    responses.get(f"{V2}/companies/99", status=404)
    result = affinity.get_org_details(99)
    assert "error" in result.lower()


# ---------------------------------------------------------------------------
# get_notes
# ---------------------------------------------------------------------------

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
