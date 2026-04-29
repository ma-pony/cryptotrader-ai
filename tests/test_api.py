"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    # Without a real DB, health returns degraded (503) or ok (200)
    assert r.status_code in (200, 503)
    assert r.json()["status"] in ("ok", "degraded")


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    # /metrics 现在返回 Prometheus 文本格式
    assert "text/plain" in r.headers["content-type"]
