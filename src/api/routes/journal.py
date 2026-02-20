"""Journal API endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/journal")


@router.get("/log")
async def journal_log(limit: int = 10):
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(None)
    commits = await store.log(limit=limit)
    return [{"hash": c.hash, "pair": c.pair, "timestamp": str(c.timestamp),
             "action": c.verdict.action if c.verdict else None} for c in commits]


@router.get("/{hash}")
async def journal_show(hash: str):
    from cryptotrader.journal.store import JournalStore

    store = JournalStore(None)
    commit = await store.show(hash)
    if not commit:
        raise HTTPException(404, "Commit not found")
    return {
        "hash": commit.hash, "pair": commit.pair,
        "timestamp": str(commit.timestamp),
        "debate_rounds": commit.debate_rounds,
        "divergence": commit.divergence,
        "verdict": commit.verdict.action if commit.verdict else None,
        "risk_gate": commit.risk_gate.passed if commit.risk_gate else None,
    }
