"""Legacy single-exchange decision routes are not mounted after the Runtime cutover."""

from fastapi.testclient import TestClient


def test_legacy_decision_list_is_unmounted() -> None:
    from api.main import app

    response = TestClient(app, raise_server_exceptions=False).get("/api/decisions")

    assert response.status_code == 404
