"""Cross-language contract tests: assert backend JSON responses contain every
field the frontend zod schemas declare.

When a backend field is renamed or typed differently from what the frontend
``api.schema.ts`` expects, the corresponding assertion fails — catching drift
at CI time instead of when a user loads the dashboard.

Does NOT parse with the real zod (that lives in the frontend test suite). Instead
encodes the frontend's field expectations as a strict set-of-keys check against
the actual FastAPI JSON payload.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from cryptotrader._compat import UTC


@pytest.fixture
def client() -> TestClient:
    from api.main import app

    return TestClient(app, raise_server_exceptions=False)


# Keys declared by the corresponding frontend zod schema. Any key listed here
# MUST be present in the backend JSON response.
_PORTFOLIO_REQUIRED = {
    "equity",
    "cash",
    "positions",
    "pnl_24h",
    "pnl_24h_pct",
    "drawdown",
    "updated_at",
    # Phase 1 additions:
    "sharpe_90d",
    "win_rate",
    "total_trades",
    "realized_pnl_30d",
    # Inception-to-date total return (2026-05-06):
    "total_return",
    "total_return_pct",
    # Mean realized PnL per settled trade (2026-05-06):
    "avg_trade_pnl",
}

_DECISION_LIST_ITEM_REQUIRED = {
    "commit_hash",
    "ts",
    "pair",
    "price",
    "verdict",
    "is_filled",
    "trace_id",
    "pnl",
    "debate_status",
    "reject_reason",
}

_DECISION_DETAIL_REQUIRED = {
    "commit_hash",
    "ts",
    "pair",
    "price",
    "agent_analyses",
    "debate_rounds",
    "verdict",
    "risk_gate",
    "execution",
    "node_timeline",
    "trace_id",
    # Phase 1 additions:
    "debate_turns",
    "debate_gate",
    "consensus_metrics",
    "latency_breakdown",
    "token_usage",
    "pnl",
    "retrospective",
    "debate_skip_reason",
}

_LATENCY_BREAKDOWN_REQUIRED = {
    "data_ms",
    "agents_ms",
    "debate_ms",
    "verdict_ms",
    "risk_ms",
    "execute_ms",
    "other_ms",
    "total_ms",
}

_TOKEN_USAGE_REQUIRED = {
    "input_tokens",
    "output_tokens",
    "cache_hits",
    "calls",
    "cost_usd",
    "by_model",
}

_RISK_REQUIRED = {
    "trade_count_hour",
    "trade_count_day",
    "circuit_breaker",
    "thresholds",
    "redis_available",
    "daily_loss_pct",
    "drawdown_pct",
    "total_exposure_pct",
    "cvar_95",
    "correlation_groups",
    "cooldowns",
    "recent_blocks",
}

_METRICS_REQUIRED = {
    "counters",
    "percentiles",
    "collected_at",
    "llm_calls_24h",
    "llm_cost_24h",
    "cache_hit_rate",
    "decisions_per_day",
    "latency_histogram",
    "cost_14d",
}


class TestPortfolioContract:
    def test_snapshot_contains_every_frontend_field(self, client: TestClient) -> None:
        with (
            patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
            patch("cryptotrader.portfolio.manager.read_portfolio_from_exchange") as read_exc,
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            pm = pm_cls.return_value
            pm.get_portfolio = AsyncMock(
                return_value={"total_value": 10_000.0, "cash": 5_000.0, "positions": {}},
            )
            pm.get_daily_pnl = AsyncMock(return_value=100.0)
            pm.get_drawdown = AsyncMock(return_value=-0.05)
            pm.load_snapshots = AsyncMock(return_value=[])
            read_exc.return_value = None
            js_cls.return_value.log = AsyncMock(return_value=[])
            resp = client.get("/api/portfolio/snapshot")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        missing = _PORTFOLIO_REQUIRED - set(body.keys())
        assert not missing, f"Portfolio response missing frontend-required fields: {missing}"


def _make_full_commit() -> MagicMock:
    """Construct a commit mock populated with every new-field group."""
    from cryptotrader.models import ConsensusMetrics

    commit = MagicMock()
    commit.hash = "c5a8f2e39b1a"
    commit.timestamp = datetime(2026, 4, 24, 10, 32, 8, tzinfo=UTC)
    commit.pair = "BTC/USDT"
    commit.snapshot_summary = {"price": 92810.22}
    commit.analyses = {}
    commit.debate_rounds = 2
    commit.challenges = [
        {
            "round": 1,
            "from": "tech_agent",
            "to": "chain_agent",
            "before": {"direction": "bullish", "confidence": 0.6},
            "after": {"direction": "bullish", "confidence": 0.85},
            "move": "强化",
            "reasoning": "strengthened",
            "new_findings": "whale bought",
            "errored": False,
        },
    ]
    commit.divergence = 0.15
    verdict = MagicMock()
    verdict.action = "long"
    verdict.confidence = 0.78
    verdict.position_scale = 0.6
    verdict.reasoning = "strong consensus"
    verdict.verdict_source = "ai"
    commit.verdict = verdict
    risk_gate = MagicMock()
    risk_gate.passed = True
    risk_gate.rejected_by = ""
    risk_gate.reason = ""
    commit.risk_gate = risk_gate
    commit.order = None
    commit.fill_price = None
    commit.slippage = None
    commit.portfolio_after = {}
    commit.pnl = 1831.2
    commit.retrospective = None
    commit.trace_id = "trace-abc"
    commit.consensus_metrics = ConsensusMetrics(
        strength=0.7,
        mean_score=0.5,
        dispersion=0.15,
        skip_threshold=0.5,
        confusion_threshold=0.05,
    )
    commit.verdict_source = "ai"
    commit.node_trace = []
    commit.debate_skip_reason = "consensus"
    commit.latency_breakdown = {
        "data": 820,
        "agents": 4210,
        "debate": 0,
        "verdict": 1340,
        "risk": 95,
        "execute": 182,
        "other": 0,
        "total": 6647,
    }
    commit.token_usage = {
        "input_tokens": 12840,
        "output_tokens": 3210,
        "cache_hits": 2,
        "calls": 5,
        "cost_usd": 0.168,
        "by_model": {"claude-sonnet-4-6": {"input": 10000, "output": 2500, "calls": 3, "cost_usd": 0.075}},
    }
    return commit


class TestDecisionDetailContract:
    def test_detail_contains_every_frontend_field(self, client: TestClient) -> None:
        commit = _make_full_commit()
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.show = AsyncMock(return_value=commit)
            js_cls.return_value.log = AsyncMock(return_value=[])
            resp = client.get(f"/api/decisions/{commit.hash}")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        missing = _DECISION_DETAIL_REQUIRED - set(body.keys())
        assert not missing, f"Detail response missing: {missing}"

        # Nested contracts
        assert _LATENCY_BREAKDOWN_REQUIRED.issubset(body["latency_breakdown"].keys()), (
            f"latency_breakdown missing: {_LATENCY_BREAKDOWN_REQUIRED - set(body['latency_breakdown'].keys())}"
        )
        assert _TOKEN_USAGE_REQUIRED.issubset(body["token_usage"].keys()), (
            f"token_usage missing: {_TOKEN_USAGE_REQUIRED - set(body['token_usage'].keys())}"
        )

    def test_debate_turn_shape_matches_frontend(self, client: TestClient) -> None:
        commit = _make_full_commit()
        with patch("cryptotrader.journal.store.JournalStore") as js_cls:
            js_cls.return_value.show = AsyncMock(return_value=commit)
            js_cls.return_value.log = AsyncMock(return_value=[])
            resp = client.get(f"/api/decisions/{commit.hash}")
        body = resp.json()
        assert len(body["debate_turns"]) == 1
        turn = body["debate_turns"][0]
        required = {
            "round",
            "from",
            "to",
            "before_direction",
            "before_confidence",
            "after_direction",
            "after_confidence",
            "move",
            "reasoning",
            "new_findings",
            "errored",
        }
        missing = required - set(turn.keys())
        assert not missing, f"debate_turn missing fields: {missing}"


class TestRiskContract:
    def test_status_contains_every_frontend_field(self, client: TestClient) -> None:
        with patch("cryptotrader.risk.state.RedisStateManager") as rsm_cls:
            rsm = rsm_cls.return_value
            rsm.available = True
            rsm.ping = AsyncMock(return_value=True)
            rsm.get_trade_counts = AsyncMock(return_value=(2, 7))
            rsm.is_circuit_breaker_active = AsyncMock(return_value=False)
            rsm.ttl = AsyncMock(return_value=-2)  # no cooldown
            with (
                patch("cryptotrader.portfolio.manager.PortfolioManager") as pm_cls,
                patch("cryptotrader.journal.store.JournalStore") as js_cls,
            ):
                pm_cls.return_value.get_portfolio = AsyncMock(
                    return_value={"total_value": 10_000.0, "positions": {}},
                )
                pm_cls.return_value.load_snapshots = AsyncMock(return_value=[])
                pm_cls.return_value.get_daily_pnl = AsyncMock(return_value=0.0)
                pm_cls.return_value.get_drawdown = AsyncMock(return_value=0.0)
                js_cls.return_value.log = AsyncMock(return_value=[])
                resp = client.get("/api/risk/status")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        missing = _RISK_REQUIRED - set(body.keys())
        assert not missing, f"Risk response missing: {missing}"


class TestMetricsContract:
    def test_summary_contains_every_frontend_field(self, client: TestClient) -> None:
        with (
            patch("api.routes.metrics._sum_counter_samples", return_value=0),
            patch("api.routes.metrics._histogram_quantile", return_value=0.0),
            patch("api.routes.metrics._collect_histogram_buckets", return_value=({}, 0.0)),
            patch("cryptotrader.journal.store.JournalStore") as js_cls,
        ):
            js_cls.return_value.log = AsyncMock(return_value=[])
            resp = client.get("/api/metrics/summary")

        assert resp.status_code == 200, resp.text
        body = resp.json()
        missing = _METRICS_REQUIRED - set(body.keys())
        assert not missing, f"Metrics response missing: {missing}"
