"""POST /analyze endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class AnalyzeRequest(BaseModel):
    pair: str = ""
    exchange: str = ""
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
    if not req.pair:
        req.pair = config.scheduler.pairs[0] if config.scheduler.pairs else "BTC/USDT"
    if not req.exchange:
        req.exchange = config.exchange_id
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
    from cryptotrader.state import build_initial_state

    result = await graph.ainvoke(
        build_initial_state(
            req.pair,
            engine=req.mode,
            exchange_id=req.exchange,
            config=config,
        )
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
