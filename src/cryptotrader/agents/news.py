"""News sentiment analysis agent — uses tool-calling to actively search news and sentiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cryptotrader.agents.base import ToolAgent
from cryptotrader.agents.data_tools import NEWS_TOOLS

if TYPE_CHECKING:
    from cryptotrader.agents.prompt_builder import PromptBuilder


class NewsAgent(ToolAgent):
    def __init__(self, *, prompt_builder: PromptBuilder, model: str = "", backtest_mode: bool = False) -> None:
        from cryptotrader.agents.skills.tool import load_skill_tool

        super().__init__(
            agent_id="news",
            prompt_builder=prompt_builder,
            tools=[*NEWS_TOOLS, load_skill_tool],
            model=model,
            backtest_mode=backtest_mode,
        )
