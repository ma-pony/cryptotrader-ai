"""Unit tests for api.routes.chat._agent_kind_from_name."""

from __future__ import annotations

from api.routes.chat import _agent_kind_from_name


class TestAgentKindFromName:
    def test_exact_match_tech(self) -> None:
        assert _agent_kind_from_name("tech_agent") == "tech"

    def test_exact_match_chain(self) -> None:
        assert _agent_kind_from_name("chain_agent") == "chain"

    def test_exact_match_news(self) -> None:
        assert _agent_kind_from_name("news_agent") == "news"

    def test_exact_match_macro(self) -> None:
        assert _agent_kind_from_name("macro_agent") == "macro"

    def test_keyword_indicator_to_tech(self) -> None:
        assert _agent_kind_from_name("indicator_analyst") == "tech"

    def test_keyword_whale_to_chain(self) -> None:
        assert _agent_kind_from_name("whale_tracker") == "chain"

    def test_keyword_onchain_to_chain(self) -> None:
        assert _agent_kind_from_name("onchain_scanner") == "chain"

    def test_keyword_sentiment_to_news(self) -> None:
        assert _agent_kind_from_name("sentiment_scorer") == "news"

    def test_keyword_fed_to_macro(self) -> None:
        assert _agent_kind_from_name("fed_watcher") == "macro"

    def test_unknown_returns_other(self) -> None:
        assert _agent_kind_from_name("random_agent_xyz") == "other"

    def test_case_insensitive(self) -> None:
        assert _agent_kind_from_name("MACRO_Agent") == "macro"

    def test_empty_string_returns_other(self) -> None:
        assert _agent_kind_from_name("") == "other"
