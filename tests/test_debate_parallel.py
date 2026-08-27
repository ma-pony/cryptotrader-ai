"""委员会辩论门控与严格错误语义。"""

from types import SimpleNamespace

from cryptotrader.debate.convergence import debate_gate_decision


def _analysis(direction, confidence):
    return {"direction": direction, "confidence": confidence}


def test_gate_skips_strong_consensus_when_enabled():
    config = SimpleNamespace(
        skip_debate=True,
        consensus_skip_threshold=0.5,
        confusion_skip_threshold=0.05,
        confusion_max_dispersion=0.2,
    )
    analyses = {name: _analysis("bullish", 0.8) for name in ("tech", "chain", "news", "macro")}

    skipped, reason, metrics = debate_gate_decision(analyses, config)

    assert skipped is True
    assert reason == "consensus"
    assert metrics["strength"] == 0.8


def test_gate_keeps_debate_for_directional_disagreement():
    config = SimpleNamespace(
        skip_debate=True,
        consensus_skip_threshold=0.5,
        confusion_skip_threshold=0.05,
        confusion_max_dispersion=0.2,
    )
    analyses = {
        "tech": _analysis("bullish", 0.8),
        "chain": _analysis("bearish", 0.8),
        "news": _analysis("bullish", 0.7),
        "macro": _analysis("bearish", 0.7),
    }

    skipped, reason, _ = debate_gate_decision(analyses, config)

    assert skipped is False
    assert reason == ""


def test_gate_can_be_configured_to_always_debate():
    config = SimpleNamespace(
        skip_debate=False,
        consensus_skip_threshold=0.5,
        confusion_skip_threshold=0.05,
        confusion_max_dispersion=0.2,
    )
    analyses = {name: _analysis("bullish", 0.9) for name in ("tech", "chain", "news", "macro")}

    assert debate_gate_decision(analyses, config)[0] is False
