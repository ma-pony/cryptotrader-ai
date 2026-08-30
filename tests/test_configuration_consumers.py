"""Only real runtime consumers may be advertised as configurable."""

from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import ValidationError, create_model

from cryptotrader.configuration.catalog import PluginConfiguration
from cryptotrader.configuration.fields import LocalizedText
from cryptotrader.configuration.parameters import DefaultMarketSourceParameters
from cryptotrader.runtime_config.models import ExecutionConfig, LlmConfig, NotificationConfig, RiskConfig
from tests.factories.runtime_config import runtime_document


def test_numeric_custom_select_options_are_rejected():
    from pydantic import Field

    model = create_model(
        "NumericOptions",
        value=(
            str,
            Field(
                default="1",
                json_schema_extra={
                    "options": [{"value": 1, "label": {"zh_CN": "一", "en_US": "One"}}],
                },
            ),
        ),
    )
    with pytest.raises(TypeError, match="Unsupported configuration field"):
        PluginConfiguration("numeric", LocalizedText("测试", "Test"), LocalizedText("测试", "Test"), model)


def test_shipped_fields_all_have_bilingual_help_and_advanced_debate_thresholds():
    from cryptotrader.configuration.fields import configuration_fields
    from cryptotrader.configuration.parameters import KronosParameters, LlmCommitteeParameters

    for model in (KronosParameters, LlmCommitteeParameters, DefaultMarketSourceParameters):
        fields = configuration_fields(model)
        assert all(field.description.zh_CN and field.description.en_US for field in fields)
        assert all(field.description.zh_CN != field.description.en_US for field in fields)
    debate = {field.key: field for field in configuration_fields(LlmCommitteeParameters)}
    assert debate["debate.convergence_threshold"].advanced is True


async def test_correlation_reporting_has_no_fictitious_limit():
    from api.routes.risk import _build_correlation_groups

    groups = await _build_correlation_groups(None, portfolio={"positions": {"BTC/USDT": {"amount": 1}}})
    assert groups[0].open == 1
    assert "max" not in groups[0].model_dump()


@pytest.mark.parametrize(
    ("model", "value"),
    [
        (LlmConfig, {"vision_models": ["unused"]}),
        (LlmConfig, {"max_image_bytes": 12}),
        (NotificationConfig, {"telegram": {"enabled": True}}),
        (NotificationConfig, {"events": ["trade"]}),
        (ExecutionConfig, {"allocation_policy": "not-consumed"}),
        (RiskConfig, {"max_stop_loss_pct": 0.2}),
        (RiskConfig, {"token_tax_threshold": 2}),
        (RiskConfig, {"cooldown": {}}),
        (RiskConfig, {"volatility": {}}),
        (RiskConfig, {"exchange": {}}),
        (RiskConfig, {"rate_limit": {}}),
        (RiskConfig, {"loss": {"max_daily_loss_pct": 0.2}}),
        (RiskConfig, {"loss": {"max_cvar_95": 0.2}}),
        (RiskConfig, {"loss": {"cvar_min_returns": 20}}),
        (RiskConfig, {"position": {"max_correlated_positions": 2}}),
        (RiskConfig, {"position": {"max_same_direction_positions": 2}}),
    ],
)
def test_ignored_controls_are_rejected_instead_of_silently_saved(model, value):
    with pytest.raises(ValidationError):
        model.model_validate(value)


def test_monitor_thresholds_are_the_four_enforced_ratio_limits():
    from api.routes.risk import _build_thresholds

    config = runtime_document(
        risk=RiskConfig(
            position={
                "max_single_pct": 0.45,
                "max_total_exposure_pct": 0.8,
                "max_margin_used_pct": 0.35,
            },
            loss={"max_drawdown_pct": 0.12},
        )
    )
    assert _build_thresholds(config).model_dump() == {
        "max_single_pct": 0.45,
        "max_total_exposure_pct": 0.8,
        "max_margin_used_pct": 0.35,
        "max_drawdown_pct": 0.12,
    }


@pytest.mark.parametrize("annotation", [dict[str, str], Any, list[int], tuple[int, ...], Literal[1, 2]])
def test_installed_plugin_rejects_unsupported_form_types_at_declaration(annotation):
    model = create_model("UnsupportedParameters", value=(annotation, ...))
    with pytest.raises(TypeError, match="Unsupported configuration field"):
        PluginConfiguration("unsupported", LocalizedText("测试", "Test"), LocalizedText("测试", "Test"), model)


def test_market_timeframe_and_limit_are_typed_supported_parameters():
    params = DefaultMarketSourceParameters.model_validate({"timeframe": "15m", "limit": 55})
    assert params.timeframe == "15m"
    assert params.limit == 55
    with pytest.raises(ValidationError):
        DefaultMarketSourceParameters.model_validate({"timeframe": "", "limit": 0})


def test_unused_debate_hold_threshold_is_rejected():
    from cryptotrader.configuration.parameters import LlmCommitteeParameters

    with pytest.raises(ValidationError):
        LlmCommitteeParameters.model_validate({"debate": {"divergence_hold_threshold": 0.5}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("enabled", "events", "sends"), [(False, ("daily_summary",), 0), (True, (), 0), (True, ("daily_summary",), 1)]
)
async def test_scheduler_summary_obeys_notification_switch_and_empty_events(monkeypatch, enabled, events, sends):
    from cryptotrader.notifications import WebhookBackend
    from cryptotrader.scheduler import Scheduler

    delivered = []

    async def send(self, event, data):
        delivered.append((event, data))

    monkeypatch.setattr(WebhookBackend, "send", send)
    document = runtime_document(
        notifications=NotificationConfig(webhook_url="https://example.test/summary", enabled=enabled, events=events)
    )
    runtime = SimpleNamespace(snapshot=SimpleNamespace(document=document, revision=1))
    scheduler = Scheduler(document.scheduler, runtime=runtime)
    await scheduler._emit_daily_summary()
    assert len(delivered) == sends
    if sends:
        assert delivered[0][0] == "daily_summary"
        assert delivered[0][1]["config_revision"] == 1
