"""Technical analysis agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents._tech_indicators import compute_indicators
from cryptotrader.agents.base import BaseAgent

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder
    from cryptotrader.models import DataSnapshot


class TechAgent(BaseAgent):
    """Technical analysis agent. Overrides _snapshot_to_dict to inject computed indicators."""

    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "") -> None:
        super().__init__(agent_id="tech", prompt_builder=prompt_builder, model=model)

    def _snapshot_to_dict(self, snapshot: DataSnapshot) -> dict:
        # spec 017b P2-3 refactor: inject indicators at dict-build, reuse BaseAgent.analyze().
        snapshot_dict = super()._snapshot_to_dict(snapshot)
        snapshot_dict["indicators"] = compute_indicators(snapshot.market.ohlcv)
        return snapshot_dict
