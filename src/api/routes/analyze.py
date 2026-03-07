"""POST /analyze endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AnalyzeRequest(BaseModel):
    pair: str = "BTC/USDT"
    exchange: str = "binance"
    mode: str = "paper"
    graph_mode: str = "full"


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
    from cryptotrader.config import load_config
    from cryptotrader.graph import build_debate_graph, build_lite_graph, build_trading_graph

    config = load_config()
    builders = {
        "full": build_trading_graph,
        "lite": build_lite_graph,
        "debate": build_debate_graph,
    }
    if req.graph_mode == "supervisor":
        from cryptotrader.graph import build_supervisor_graph_v2

        graph = build_supervisor_graph_v2()
    else:
        graph = builders.get(req.graph_mode, build_trading_graph)()
    result = await graph.ainvoke(
        {
            "messages": [],
            "data": {},
            "metadata": {
                "pair": req.pair,
                "engine": req.mode,
                "exchange_id": req.exchange,
                "timeframe": config.data.default_timeframe,
                "ohlcv_limit": config.data.ohlcv_limit,
                "analysis_model": config.models.analysis,
                "debate_model": config.models.debate,
                "verdict_model": config.models.verdict,
                "database_url": config.infrastructure.database_url,
                "redis_url": config.infrastructure.redis_url,
                "convergence_threshold": config.debate.convergence_threshold,
                "max_single_pct": config.risk.position.max_single_pct,
                "models": {
                    "tech_agent": config.models.tech_agent,
                    "chain_agent": config.models.chain_agent,
                    "news_agent": config.models.news_agent,
                    "macro_agent": config.models.macro_agent,
                },
            },
            "debate_round": 0,
            "max_debate_rounds": config.debate.max_rounds,
            "divergence_scores": [],
        }
    )

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
