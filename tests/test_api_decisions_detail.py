"""Legacy decision-detail presentation is deferred until it can consume the strict Journal."""

from fastapi.testclient import TestClient


def test_legacy_decision_detail_is_unmounted() -> None:
    from api.main import app

    response = TestClient(app, raise_server_exceptions=False).get("/api/decisions/cycle-1")

    assert response.status_code == 404
