"""Decision journal storage with in-memory fallback."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Any

from cryptotrader.models import DecisionCommit


class JournalStore:
    """Stores decision commits. Uses in-memory list if no database_url."""

    def __init__(self, database_url: str | None = None):
        self._db_url = database_url
        self._memory: list[dict[str, Any]] = []

    def _serialize(self, dc: DecisionCommit) -> dict[str, Any]:
        return {"hash": dc.hash, "parent_hash": dc.parent_hash,
                "timestamp": dc.timestamp.isoformat(), "pair": dc.pair,
                "data": json.loads(json.dumps(asdict(dc), default=str))}

    def _deserialize(self, row: dict[str, Any]) -> DecisionCommit:
        d = row["data"]
        from cryptotrader.models import AgentAnalysis, TradeVerdict, GateResult, Order
        analyses = {k: AgentAnalysis(**v) for k, v in d.get("analyses", ).items()}
        for a in analyses.values():
            if isinstance(a.timestamp, str):
                a.timestamp = datetime.fromisoformat(a.timestamp)
        verdict = TradeVerdict(**d["verdict"]) if d.get("verdict") else None
        risk_gate = GateResult(**d["risk_gate"]) if d.get("risk_gate") else None
        order = None
        if d.get("order"):
            od = d["order"]
            od.pop("status", None)
            order = Order(**od)
        return DecisionCommit(
            hash=d["hash"], parent_hash=d.get("parent_hash"),
            timestamp=datetime.fromisoformat(d["timestamp"]), pair=d["pair"],
            snapshot_summary=d.get("snapshot_summary", {}), analyses=analyses,
            debate_rounds=d.get("debate_rounds", 0), divergence=d.get("divergence", 0.0),
            verdict=verdict, risk_gate=risk_gate, order=order,
            pnl=d.get("pnl"), retrospective=d.get("retrospective"),
        )

    async def commit(self, dc: DecisionCommit) -> None:
        self._memory.append(self._serialize(dc))

    async def log(self, limit: int = 10, pair: str | None = None) -> list[DecisionCommit]:
        rows = self._memory
        if pair:
            rows = [r for r in rows if r["pair"] == pair]
        return [self._deserialize(r) for r in rows[-limit:]]

    async def show(self, hash: str) -> DecisionCommit | None:
        for r in self._memory:
            if r["hash"] == hash:
                return self._deserialize(r)
        return None

    async def update_pnl(self, hash: str, pnl: float, retrospective: str) -> None:
        for r in self._memory:
            if r["hash"] == hash:
                r["data"]["pnl"] = pnl
                r["data"]["retrospective"] = retrospective
                return
