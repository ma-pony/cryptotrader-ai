"""Tests for external API response schema validation (task 6.4).

Covers:
- NewsHeadlineResponse and OnchainMetricResponse Pydantic models
- field_validator constraints (non-empty title, etc.)
- Adapter-level schema validation: invalid rows are skipped with logger.warning
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cryptotrader.models import NewsHeadlineResponse, OnchainMetricResponse

# ---------------------------------------------------------------------------
# 1. NewsHeadlineResponse model validation
# ---------------------------------------------------------------------------


class TestNewsHeadlineResponse:
    def test_valid_headline(self):
        h = NewsHeadlineResponse(title="Bitcoin hits new ATH", source="coindesk", published="2026-03-14")
        assert h.title == "Bitcoin hits new ATH"
        assert h.source == "coindesk"

    def test_empty_title_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="title"):
            NewsHeadlineResponse(title="", source="coindesk", published="2026-03-14")

    def test_whitespace_only_title_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="title"):
            NewsHeadlineResponse(title="   ", source="coindesk", published="2026-03-14")

    def test_optional_fields_have_defaults(self):
        h = NewsHeadlineResponse(title="Some news")
        assert h.source == ""
        assert h.published == ""

    def test_title_is_stripped(self):
        # Validator should strip surrounding whitespace before non-empty check
        h = NewsHeadlineResponse(title="  BTC rally  ", source="cointelegraph")
        assert h.title == "BTC rally"

    def test_title_max_length_enforced(self):
        from pydantic import ValidationError

        long_title = "x" * 2001
        with pytest.raises(ValidationError, match="title"):
            NewsHeadlineResponse(title=long_title)

    def test_title_within_max_length(self):
        h = NewsHeadlineResponse(title="x" * 2000)
        assert len(h.title) == 2000


# ---------------------------------------------------------------------------
# 2. OnchainMetricResponse model validation
# ---------------------------------------------------------------------------


class TestOnchainMetricResponse:
    def test_valid_metric(self):
        m = OnchainMetricResponse(metric_name="open_interest", value=12345.67, source="coinglass")
        assert m.metric_name == "open_interest"
        assert m.value == pytest.approx(12345.67)

    def test_empty_metric_name_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="metric_name"):
            OnchainMetricResponse(metric_name="", value=1.0, source="coinglass")

    def test_whitespace_metric_name_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="metric_name"):
            OnchainMetricResponse(metric_name="  ", value=1.0, source="coinglass")

    def test_negative_value_raises(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="value"):
            OnchainMetricResponse(metric_name="open_interest", value=-1.0, source="coinglass")

    def test_zero_value_is_valid(self):
        m = OnchainMetricResponse(metric_name="netflow", value=0.0, source="cryptoquant")
        assert m.value == 0.0

    def test_optional_source_defaults_to_empty(self):
        m = OnchainMetricResponse(metric_name="tvl", value=100.0)
        assert m.source == ""

    def test_metric_name_is_stripped(self):
        m = OnchainMetricResponse(metric_name="  tvl  ", value=100.0)
        assert m.metric_name == "tvl"


# ---------------------------------------------------------------------------
# 3. NewsCollector adapter: schema validation integration
# ---------------------------------------------------------------------------


class TestNewsCollectorSchemaValidation:
    """Integration tests for schema validation in NewsCollector._collect_cryptocompare."""

    @pytest.mark.asyncio
    async def test_articles_with_empty_title_are_skipped(self, caplog):
        """Raw API articles with empty title are skipped; valid ones are returned."""
        raw_articles = [
            {"title": "", "body": "some content", "source": "src1"},
            {"title": "  ", "body": "other content", "source": "src2"},
            {"title": "Valid headline", "body": "good content", "source": "src3"},
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"Data": raw_articles}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        mock_config = MagicMock()
        mock_config.providers.coindesk_api_key = ""

        with (
            patch("cryptotrader.config.load_config", return_value=mock_config),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            from cryptotrader.data.news import NewsCollector

            collector = NewsCollector()
            with caplog.at_level(logging.WARNING, logger="cryptotrader.data.news"):
                articles = await collector._collect_cryptocompare("BTC")

        # Only 1 valid article should be returned
        assert len(articles) == 1
        assert articles[0].title == "Valid headline"

    @pytest.mark.asyncio
    async def test_schema_validation_failure_logs_warning(self, caplog):
        """Invalid rows log a warning with the validation error info."""
        raw_articles = [
            {"title": "", "body": "content", "source": "src"},
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"Data": raw_articles}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        mock_config = MagicMock()
        mock_config.providers.coindesk_api_key = ""

        with (
            patch("cryptotrader.config.load_config", return_value=mock_config),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            from cryptotrader.data.news import NewsCollector

            collector = NewsCollector()
            with caplog.at_level(logging.WARNING, logger="cryptotrader.data.news"):
                await collector._collect_cryptocompare("BTC")

        assert any("schema" in r.message.lower() or "validation" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_all_valid_articles_returned(self, caplog):
        """All valid articles are returned when no schema violations occur."""
        raw_articles = [
            {"title": "Headline 1", "body": "content 1", "source": "src1"},
            {"title": "Headline 2", "body": "content 2", "source": "src2"},
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"Data": raw_articles}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        mock_config = MagicMock()
        mock_config.providers.coindesk_api_key = ""

        with (
            patch("cryptotrader.config.load_config", return_value=mock_config),
            patch("httpx.AsyncClient", return_value=mock_client),
        ):
            from cryptotrader.data.news import NewsCollector

            collector = NewsCollector()
            with caplog.at_level(logging.WARNING, logger="cryptotrader.data.news"):
                articles = await collector._collect_cryptocompare("BTC")

        assert len(articles) == 2
        assert not any("schema" in r.message.lower() or "validation" in r.message.lower() for r in caplog.records)


# ---------------------------------------------------------------------------
# 4. OnchainCollector adapters: schema validation integration
# ---------------------------------------------------------------------------


class TestOnchainProviderSchemaValidation:
    """Integration tests for schema validation in onchain provider fetchers."""

    @pytest.mark.asyncio
    async def test_coinglass_invalid_open_interest_skipped(self, caplog):
        """Negative open_interest from CoinGlass is rejected and defaults to 0."""
        from cryptotrader.data.providers.coinglass import fetch_derivatives

        raw_data = [{"openInterest": -500.0}]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": raw_data}

        mock_resp2 = MagicMock()
        mock_resp2.raise_for_status = MagicMock()
        mock_resp2.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=[mock_resp, mock_resp2])

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            caplog.at_level(logging.WARNING, logger="cryptotrader.data.providers.coinglass"),
        ):
            result = await fetch_derivatives(api_key="test_key", symbol="BTC")  # pragma: allowlist secret

        # Invalid OI should be rejected; result defaults to 0.0
        assert result["open_interest"] == pytest.approx(0.0)
        assert any("schema" in r.message.lower() or "validation" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_whale_alert_empty_hash_skipped(self, caplog):
        """Whale Alert transactions with empty hash field are skipped."""
        from cryptotrader.data.providers.whale_alert import fetch_whale_transfers

        transactions = [
            {"hash": "", "from": {"owner": "binance"}, "to": {"owner": "unknown"}, "amount_usd": 1000000},
            {
                "hash": "abc123",
                "from": {"owner": "binance"},
                "to": {"owner": "unknown"},
                "amount_usd": 2000000,
            },
        ]
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"transactions": transactions}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with (
            patch("httpx.AsyncClient", return_value=mock_client),
            caplog.at_level(logging.WARNING, logger="cryptotrader.data.providers.whale_alert"),
        ):
            result = await fetch_whale_transfers(api_key="test_key")  # pragma: allowlist secret

        # Only the valid transaction (abc123) should be returned
        assert len(result) == 1
        assert result[0]["hash"] == "abc123"
