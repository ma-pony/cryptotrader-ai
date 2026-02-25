"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    d = r.json()
    assert "decisions_total" in d
    assert "win_rate" in d
    assert "uptime_seconds" in d


def test_journal_log_empty():
    r = client.get("/journal/log")
    assert r.status_code == 200
    assert r.json() == []


def test_journal_show_not_found():
    r = client.get("/journal/nonexistent")
    assert r.status_code == 404
