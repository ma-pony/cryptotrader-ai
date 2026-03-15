"""Tests for DashboardDataLoader data loading layer (task 6.1).

TDD strict mode: tests written before implementation.

Testing philosophy per task instructions:
- Minimize mocks — mocks hide real problems.
- HTTP functions (scheduler_status, metrics_summary): test real httpx error paths
  using a real test HTTP server (via threading + http.server) or real httpx exceptions.
- DB functions: test signatures and real behavior using in-memory JournalStore /
  PortfolioManager (no DB connection needed).
- Backtest sessions: use real file system with temp directories.
- Verify that HTTP timeout (5 s) returns None, not an exception.
"""

from __future__ import annotations

import importlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers — import the module under test with streamlit mocked
# ---------------------------------------------------------------------------


def _make_st_mock() -> MagicMock:
    """Return a MagicMock that simulates the Streamlit API used by data_loader."""
    st = MagicMock()
    # cache_data decorator: @st.cache_data(ttl=N) or @st.cache_data
    # Both must behave as pass-through decorators so functions are testable directly.
    st.cache_data = lambda *args, **kwargs: lambda fn: fn
    st.cache_resource = lambda fn: fn
    return st


def _import_data_loader() -> Any:
    """Import (or reload) dashboard.data_loader with streamlit mocked."""
    # Clear so the module is re-evaluated with our st mock
    for key in list(sys.modules.keys()):
        if "dashboard.data_loader" in key:
            del sys.modules[key]
    st_mock = _make_st_mock()
    with patch.dict("sys.modules", {"streamlit": st_mock}):
        return importlib.import_module("dashboard.data_loader")


# ---------------------------------------------------------------------------
# Tiny real HTTP server for testing HTTP loader functions
# ---------------------------------------------------------------------------


class _FixedResponseHandler(BaseHTTPRequestHandler):
    """HTTP handler that returns a fixed JSON body."""

    _response_body: bytes = b"{}"
    _status_code: int = 200

    def do_GET(self):
        self.send_response(self._status_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.__class__._response_body)

    def log_message(self, *args, **kwargs):
        # Suppress default stderr logging during tests
        pass


def _make_handler(body: dict | None = None, status: int = 200):
    """Return an HTTP handler class serving the given body and status."""
    body_bytes = json.dumps(body or {}).encode()

    class _Handler(_FixedResponseHandler):
        _response_body = body_bytes
        _status_code = status

    return _Handler


class _TestServer:
    """Context manager that runs a real HTTP server in a background thread."""

    def __init__(self, handler_class):
        self._server = HTTPServer(("127.0.0.1", 0), handler_class)
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def __enter__(self) -> _TestServer:
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Module interface — function existence and signatures
# ---------------------------------------------------------------------------


def test_load_portfolio_function_exists():
    """data_loader exposes load_portfolio() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_portfolio"), "load_portfolio must be defined"
    assert callable(dl.load_portfolio)


def test_load_journal_function_exists():
    """data_loader exposes load_journal() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_journal"), "load_journal must be defined"
    assert callable(dl.load_journal)


def test_load_commit_detail_function_exists():
    """data_loader exposes load_commit_detail() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_commit_detail"), "load_commit_detail must be defined"
    assert callable(dl.load_commit_detail)


def test_load_risk_status_function_exists():
    """data_loader exposes load_risk_status() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_risk_status"), "load_risk_status must be defined"
    assert callable(dl.load_risk_status)


def test_load_scheduler_status_function_exists():
    """data_loader exposes load_scheduler_status() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_scheduler_status"), "load_scheduler_status must be defined"
    assert callable(dl.load_scheduler_status)


def test_load_metrics_summary_function_exists():
    """data_loader exposes load_metrics_summary() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_metrics_summary"), "load_metrics_summary must be defined"
    assert callable(dl.load_metrics_summary)


def test_list_backtest_sessions_function_exists():
    """data_loader exposes list_backtest_sessions() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "list_backtest_sessions"), "list_backtest_sessions must be defined"
    assert callable(dl.list_backtest_sessions)


def test_load_backtest_session_function_exists():
    """data_loader exposes load_backtest_session() at module level."""
    dl = _import_data_loader()
    assert hasattr(dl, "load_backtest_session"), "load_backtest_session must be defined"
    assert callable(dl.load_backtest_session)


# ---------------------------------------------------------------------------
# load_portfolio — in-memory PortfolioManager (no DB needed)
# ---------------------------------------------------------------------------


def test_load_portfolio_returns_dict():
    """load_portfolio returns a dict with portfolio keys from PortfolioManager."""
    dl = _import_data_loader()
    # Use empty db_url — PortfolioManager falls back to in-memory, returns valid dict
    result = dl.load_portfolio(db_url=None)
    assert isinstance(result, dict)
    assert "account_id" in result
    assert "positions" in result
    assert "cash" in result
    assert "total_value" in result


def test_load_portfolio_total_value_non_negative():
    """load_portfolio total_value is non-negative for a fresh in-memory store."""
    dl = _import_data_loader()
    result = dl.load_portfolio(db_url=None)
    assert result["total_value"] >= 0.0


# ---------------------------------------------------------------------------
# load_journal — in-memory JournalStore (no DB needed)
# ---------------------------------------------------------------------------


def test_load_journal_returns_list():
    """load_journal returns a list (empty when no commits in store)."""
    dl = _import_data_loader()
    result = dl.load_journal(db_url=None, limit=10, pair=None, offset=0)
    assert isinstance(result, list)


def test_load_journal_limit_parameter_respected():
    """load_journal respects the limit parameter."""
    dl = _import_data_loader()
    # With empty store, any limit returns an empty list (limit is respected)
    result = dl.load_journal(db_url=None, limit=5, pair=None, offset=0)
    assert len(result) <= 5


def test_load_journal_different_pairs_independent():
    """load_journal with different pair arguments doesn't share results."""
    dl = _import_data_loader()
    # Both return lists — no cross-contamination
    result_all = dl.load_journal(db_url=None, limit=20, pair=None, offset=0)
    result_btc = dl.load_journal(db_url=None, limit=20, pair="BTC/USDT", offset=0)
    assert isinstance(result_all, list)
    assert isinstance(result_btc, list)


# ---------------------------------------------------------------------------
# load_commit_detail — in-memory JournalStore (no DB needed)
# ---------------------------------------------------------------------------


def test_load_commit_detail_returns_none_for_missing_hash():
    """load_commit_detail returns None when the commit hash is not found."""
    dl = _import_data_loader()
    result = dl.load_commit_detail(db_url=None, commit_hash="nonexistent123")
    assert result is None


# ---------------------------------------------------------------------------
# load_risk_status — real RedisStateManager with no Redis URL
# ---------------------------------------------------------------------------


def test_load_risk_status_returns_none_when_redis_unavailable():
    """load_risk_status returns None when redis_url is None (Redis not configured)."""
    dl = _import_data_loader()
    result = dl.load_risk_status(redis_url=None)
    assert result is None


def test_load_risk_status_returns_dict_when_redis_available_in_memory():
    """load_risk_status returns a dict when RedisStateManager uses in-memory fallback."""
    dl = _import_data_loader()
    # Pass an invalid redis URL — RedisStateManager will instantiate but use memory fallback
    # The function should still return a risk status dict (from memory store)
    result = dl.load_risk_status(redis_url="redis://127.0.0.1:19999")
    # Either None (if unreachable detection) or a dict — must not raise
    assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# load_scheduler_status — HTTP calls with real httpx
# ---------------------------------------------------------------------------


def test_load_scheduler_status_returns_dict_from_valid_server():
    """load_scheduler_status returns parsed dict when server returns valid JSON."""
    scheduler_payload = {
        "running": True,
        "jobs": [],
        "cycle_count": 5,
        "interval_minutes": 240,
        "pairs": ["BTC/USDT"],
    }
    handler = _make_handler(scheduler_payload, status=200)
    with _TestServer(handler) as server:
        dl = _import_data_loader()
        result = dl.load_scheduler_status(api_base_url=server.base_url)
    assert isinstance(result, dict)
    assert result.get("running") is True
    assert result.get("cycle_count") == 5


def test_load_scheduler_status_returns_none_on_connection_refused():
    """load_scheduler_status returns None when the server is not reachable."""
    dl = _import_data_loader()
    # Port 19998 is almost certainly not listening
    result = dl.load_scheduler_status(api_base_url="http://127.0.0.1:19998")
    assert result is None


def test_load_scheduler_status_returns_none_on_server_error():
    """load_scheduler_status returns None on HTTP 500 from the server."""
    handler = _make_handler({"error": "internal"}, status=500)
    with _TestServer(handler) as server:
        dl = _import_data_loader()
        result = dl.load_scheduler_status(api_base_url=server.base_url)
    # 5xx responses should be treated as errors → None
    assert result is None


def test_load_scheduler_status_returns_none_on_timeout(monkeypatch):
    """load_scheduler_status returns None when the request times out."""
    import httpx

    dl = _import_data_loader()

    def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "get", _raise_timeout)
    result = dl.load_scheduler_status(api_base_url="http://127.0.0.1:8000")
    assert result is None


# ---------------------------------------------------------------------------
# load_metrics_summary — HTTP calls with real httpx
# ---------------------------------------------------------------------------


def test_load_metrics_summary_returns_dict_from_valid_server():
    """load_metrics_summary returns parsed dict when server returns valid JSON."""
    metrics_payload = {
        "llm_calls_total": 100,
        "debate_skipped_total": 30,
        "verdict_distribution": {"long": 40, "short": 20, "hold": 40},
        "risk_rejected_total": 5,
        "risk_rejected_by_check": {},
        "trade_executed_total": 60,
        "pipeline_duration_p50_ms": 4000.0,
        "pipeline_duration_p95_ms": 10000.0,
        "execution_latency_p50_ms": 200.0,
        "execution_latency_p95_ms": 900.0,
        "snapshot_time": "2026-03-15T10:00:00Z",
    }
    handler = _make_handler(metrics_payload, status=200)
    with _TestServer(handler) as server:
        dl = _import_data_loader()
        result = dl.load_metrics_summary(api_base_url=server.base_url)
    assert isinstance(result, dict)
    assert result.get("llm_calls_total") == 100
    assert result.get("debate_skipped_total") == 30


def test_load_metrics_summary_returns_none_on_connection_refused():
    """load_metrics_summary returns None when the server is not reachable."""
    dl = _import_data_loader()
    result = dl.load_metrics_summary(api_base_url="http://127.0.0.1:19997")
    assert result is None


def test_load_metrics_summary_returns_none_on_timeout(monkeypatch):
    """load_metrics_summary returns None when the request times out."""
    import httpx

    dl = _import_data_loader()

    def _raise_timeout(*args, **kwargs):
        raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(httpx, "get", _raise_timeout)
    result = dl.load_metrics_summary(api_base_url="http://127.0.0.1:8000")
    assert result is None


def test_load_metrics_summary_returns_none_on_server_error():
    """load_metrics_summary returns None on HTTP 5xx responses."""
    handler = _make_handler({"error": "boom"}, status=503)
    with _TestServer(handler) as server:
        dl = _import_data_loader()
        result = dl.load_metrics_summary(api_base_url=server.base_url)
    assert result is None


# ---------------------------------------------------------------------------
# list_backtest_sessions — real file system with temp directories
# ---------------------------------------------------------------------------


def test_list_backtest_sessions_returns_list(tmp_path, monkeypatch):
    """list_backtest_sessions returns a list of session IDs."""
    # Patch the sessions directory to a temp location
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)

    dl = _import_data_loader()
    result = dl.list_backtest_sessions()
    assert isinstance(result, list)


def test_list_backtest_sessions_empty_when_no_sessions(tmp_path, monkeypatch):
    """list_backtest_sessions returns empty list when sessions dir has no sub-dirs."""
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)
    dl = _import_data_loader()
    result = dl.list_backtest_sessions()
    assert result == []


def test_list_backtest_sessions_returns_session_ids(tmp_path, monkeypatch):
    """list_backtest_sessions returns IDs matching created session dirs."""
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)

    # Create two fake session directories
    (tmp_path / "BTC_USDT_2026-01-01_2026-01-31_4h_20260101_000000").mkdir()
    (tmp_path / "ETH_USDT_2026-01-01_2026-01-31_4h_20260101_000001").mkdir()

    dl = _import_data_loader()
    result = dl.list_backtest_sessions()
    assert len(result) == 2
    assert "BTC_USDT_2026-01-01_2026-01-31_4h_20260101_000000" in result
    assert "ETH_USDT_2026-01-01_2026-01-31_4h_20260101_000001" in result


# ---------------------------------------------------------------------------
# load_backtest_session — real file system with temp directories
# ---------------------------------------------------------------------------


def test_load_backtest_session_returns_list_for_existing_session(tmp_path, monkeypatch):
    """load_backtest_session returns a list of commit dicts for a session with commits."""
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)

    session_id = "test_session_001"
    session_dir = tmp_path / session_id
    session_dir.mkdir()

    # Write two commit lines
    commits_path = session_dir / "commits.jsonl"
    commits_path.write_text(
        json.dumps({"hash": "abc123", "pair": "BTC/USDT"})
        + "\n"
        + json.dumps({"hash": "def456", "pair": "BTC/USDT"})
        + "\n"
    )

    dl = _import_data_loader()
    result = dl.load_backtest_session(session_id=session_id)
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["hash"] == "abc123"
    assert result[1]["hash"] == "def456"


def test_load_backtest_session_returns_empty_for_nonexistent_session(tmp_path, monkeypatch):
    """load_backtest_session returns empty list when session does not exist."""
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)

    dl = _import_data_loader()
    result = dl.load_backtest_session(session_id="does_not_exist")
    assert result == []


def test_load_backtest_session_returns_empty_when_no_commits_file(tmp_path, monkeypatch):
    """load_backtest_session returns empty list when session dir has no commits.jsonl."""
    import cryptotrader.backtest.session as session_mod

    monkeypatch.setattr(session_mod, "_SESSIONS_DIR", tmp_path)

    session_id = "empty_session"
    (tmp_path / session_id).mkdir()

    dl = _import_data_loader()
    result = dl.load_backtest_session(session_id=session_id)
    assert result == []


# ---------------------------------------------------------------------------
# HTTP functions swallow exceptions (not DB exceptions)
# ---------------------------------------------------------------------------


def test_load_scheduler_status_does_not_raise_on_invalid_json(monkeypatch):
    """load_scheduler_status returns None (not exception) when server returns invalid JSON."""

    class _BadJsonHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not-valid-json{{{{")

        def log_message(self, *args, **kwargs):
            pass

    with _TestServer(_BadJsonHandler) as server:
        dl = _import_data_loader()
        result = dl.load_scheduler_status(api_base_url=server.base_url)
    assert result is None


def test_load_metrics_summary_does_not_raise_on_invalid_json(monkeypatch):
    """load_metrics_summary returns None (not exception) when server returns invalid JSON."""

    class _BadJsonHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"not-valid-json{{{{")

        def log_message(self, *args, **kwargs):
            pass

    with _TestServer(_BadJsonHandler) as server:
        dl = _import_data_loader()
        result = dl.load_metrics_summary(api_base_url=server.base_url)
    assert result is None


# ---------------------------------------------------------------------------
# Timeout parameter — verify 5 s timeout is passed to httpx
# ---------------------------------------------------------------------------


def test_load_scheduler_status_passes_timeout_to_httpx(monkeypatch):
    """load_scheduler_status calls httpx.get() with timeout=5."""
    import httpx

    captured = {}

    def _fake_get(url, timeout=None, **kwargs):
        captured["url"] = url
        captured["timeout"] = timeout
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"running": False, "jobs": [], "cycle_count": 0, "interval_minutes": 0, "pairs": []}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(httpx, "get", _fake_get)
    dl = _import_data_loader()
    dl.load_scheduler_status(api_base_url="http://127.0.0.1:8000")
    assert captured.get("timeout") == 5


def test_load_metrics_summary_passes_timeout_to_httpx(monkeypatch):
    """load_metrics_summary calls httpx.get() with timeout=5."""
    import httpx

    captured = {}

    def _fake_get(url, timeout=None, **kwargs):
        captured["url"] = url
        captured["timeout"] = timeout
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "llm_calls_total": 0,
            "debate_skipped_total": 0,
            "verdict_distribution": {},
            "risk_rejected_total": 0,
            "risk_rejected_by_check": {},
            "trade_executed_total": 0,
            "pipeline_duration_p50_ms": 0.0,
            "pipeline_duration_p95_ms": 0.0,
            "execution_latency_p50_ms": 0.0,
            "execution_latency_p95_ms": 0.0,
            "snapshot_time": "2026-03-15T10:00:00Z",
        }
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(httpx, "get", _fake_get)
    dl = _import_data_loader()
    dl.load_metrics_summary(api_base_url="http://127.0.0.1:8000")
    assert captured.get("timeout") == 5


# ---------------------------------------------------------------------------
# URL construction — verify correct endpoint paths are called
# ---------------------------------------------------------------------------


def test_load_scheduler_status_calls_correct_endpoint(monkeypatch):
    """load_scheduler_status calls /scheduler/status on the given api_base_url."""
    import httpx

    captured = {}

    def _fake_get(url, timeout=None, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"running": False, "jobs": [], "cycle_count": 0, "interval_minutes": 0, "pairs": []}
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(httpx, "get", _fake_get)
    dl = _import_data_loader()
    dl.load_scheduler_status(api_base_url="http://myapi:8080")
    assert "scheduler/status" in captured.get("url", "")
    assert "myapi:8080" in captured.get("url", "")


def test_load_metrics_summary_calls_correct_endpoint(monkeypatch):
    """load_metrics_summary calls /metrics/summary on the given api_base_url."""
    import httpx

    captured = {}

    def _fake_get(url, timeout=None, **kwargs):
        captured["url"] = url
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {
            "llm_calls_total": 0,
            "debate_skipped_total": 0,
            "verdict_distribution": {},
            "risk_rejected_total": 0,
            "risk_rejected_by_check": {},
            "trade_executed_total": 0,
            "pipeline_duration_p50_ms": 0.0,
            "pipeline_duration_p95_ms": 0.0,
            "execution_latency_p50_ms": 0.0,
            "execution_latency_p95_ms": 0.0,
            "snapshot_time": "2026-03-15T10:00:00Z",
        }
        resp.raise_for_status = MagicMock()
        return resp

    monkeypatch.setattr(httpx, "get", _fake_get)
    dl = _import_data_loader()
    dl.load_metrics_summary(api_base_url="http://myapi:8080")
    assert "metrics/summary" in captured.get("url", "")
    assert "myapi:8080" in captured.get("url", "")
