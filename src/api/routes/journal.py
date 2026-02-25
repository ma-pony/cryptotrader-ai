"""Journal API endpoints."""

from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/journal")


def _get_store():
    from cryptotrader.journal.store import JournalStore
    return JournalStore(os.environ.get("DATABASE_URL"))


@router.get("/log")
async def journal_log(limit: int = 10):
    store = _get_store()
    commits = await store.log(limit=limit)
    return [{"hash": c.hash, "pair": c.pair, "timestamp": str(c.timestamp),
             "action": c.verdict.action if c.verdict else None} for c in commits]


@router.get("/{hash}")
async def journal_show(hash: str):
    store = _get_store()
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
