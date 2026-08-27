"""内部辩论的置信度 anti-ratchet 规则。"""

import pytest

from cryptotrader.debate.challenge import apply_anti_ratchet


def test_raise_without_new_evidence_is_capped():
    assert apply_anti_ratchet("bearish", 0.60, "bearish", 0.80, "others agree") == pytest.approx(0.62)


@pytest.mark.parametrize("tag", ["[NEW] datum", "[new] datum", " [New] datum"])
def test_raise_with_new_evidence_tag_passes(tag):
    assert apply_anti_ratchet("bearish", 0.60, "bearish", 0.80, tag) == pytest.approx(0.80)


def test_small_raise_and_lower_confidence_pass():
    assert apply_anti_ratchet("bearish", 0.60, "bearish", 0.62, "") == pytest.approx(0.62)
    assert apply_anti_ratchet("bearish", 0.60, "bearish", 0.40, "") == pytest.approx(0.40)


def test_direction_flip_bypasses_confidence_cap():
    assert apply_anti_ratchet("bearish", 0.60, "bullish", 0.80, "") == pytest.approx(0.80)
