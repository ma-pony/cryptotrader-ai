"""FastAPI endpoint tests."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    # The explicit minimal RuntimeConfig is intentionally not activated.
    assert r.status_code == 200
    assert r.json()["status"] == "setup_required"


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    # /metrics 现在返回 Prometheus 文本格式
    assert "text/plain" in r.headers["content-type"]
