"""POST /analyze endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AnalyzeRequest(BaseModel):
    pair: str = "BTC/USDT"
    exchange: str = "binance"
    mode: str = "paper"


class AnalyzeResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    position_scale: float
    divergence: float
    reasoning: str
    risk_flags: list[str]
    debate_rounds: int


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    from cryptotrader.graph import build_trading_graph

    graph = build_trading_graph()
    result = await graph.ainvoke({
        "messages": [],
        "data": {},
        "metadata": {
            "pair": req.pair,
            "engine": req.mode,
            "exchange_id": req.exchange,
            "timeframe": "1h",
            "ohlcv_limit": 100,
        },
        "debate_round": 0,
        "max_debate_rounds": 3,
        "divergence_scores": [],
    })

    verdict = result.get("data", {}).get("verdict", {})
    analyses = result.get("data", {}).get("analyses", {})
    flags = []
    for a in analyses.values():
        flags.extend(a.get("risk_flags", []))

    return AnalyzeResponse(
        pair=req.pair,
        direction=verdict.get("action", "hold"),
        confidence=verdict.get("confidence", 0),
        position_scale=verdict.get("position_scale", 0),
        divergence=verdict.get("divergence", 0),
        reasoning=verdict.get("reasoning", ""),
        risk_flags=flags,
        debate_rounds=result.get("debate_round", 0),
    )
