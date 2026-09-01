"""FastAPI endpoint tests."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    from cryptotrader.runtime_config.defaults import minimal_runtime_document
    from cryptotrader.runtime_config.models import RuntimeConfigSnapshot

    runtime = app.state.runtime
    app.state.runtime = type(runtime)(
        snapshot=RuntimeConfigSnapshot(1, minimal_runtime_document(), datetime.now(UTC)),
        repository=runtime.repository,
        cycle=None,
    )
    try:
        r = client.get("/health")
    finally:
        app.state.runtime = runtime
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    # /metrics 现在返回 Prometheus 文本格式
    assert "text/plain" in r.headers["content-type"]
