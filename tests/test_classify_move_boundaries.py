"""Boundary cases for _classify_move (Deep Review I-T + C-m3)."""

from __future__ import annotations

from cryptotrader.nodes.debate import _classify_move, _dir_label


class TestClassifyMoveBoundaries:
    def test_exactly_at_strengthen_threshold(self) -> None:
        """delta = 0.05 exactly should count as 强化."""
        assert _classify_move("bullish", 0.60, "bullish", 0.65) == "强化"

    def test_exactly_at_weaken_threshold(self) -> None:
        """delta = -0.05 exactly should count as 弱化."""
        assert _classify_move("bullish", 0.65, "bullish", 0.60) == "弱化"

    def test_just_below_threshold_holds(self) -> None:
        """|delta| < 0.05 should hold."""
        assert _classify_move("bullish", 0.60, "bullish", 0.64) == "保持"

    def test_confidence_above_one_still_classifies(self) -> None:
        """confidence > 1.0 is malformed but should not crash."""
        result = _classify_move("bullish", 0.90, "bullish", 1.10)
        assert result == "强化"  # delta=0.2

    def test_direction_flip_to_neutral(self) -> None:
        result = _classify_move("bullish", 0.6, "neutral", 0.3)
        assert "让步" in result
        assert "中性" in result

    def test_direction_flip_bearish_to_bullish(self) -> None:
        result = _classify_move("bearish", 0.5, "bullish", 0.55)
        assert "让步" in result
        assert "看空" in result
        assert "看多" in result

    def test_empty_before_direction_does_not_crash(self) -> None:
        """Old journal rows may have empty direction strings."""
        # Currently '' != 'bullish' → considered a flip
        result = _classify_move("", 0.5, "bullish", 0.6)
        assert "让步" in result


class TestDirLabel:
    def test_empty_string_passthrough(self) -> None:
        assert _dir_label("") == ""

    def test_unknown_passthrough(self) -> None:
        assert _dir_label("mystery") == "mystery"
