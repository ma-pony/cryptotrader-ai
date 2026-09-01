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
async def test_scheduler_summary_obeys_notification_switch_and_empty_events(tmp_path, enabled, events, sends):
    from cryptotrader.alerts.service import AlertService
    from cryptotrader.alerts.store import AlertStore
    from cryptotrader.migrations.workbench import migrate_alerts
    from cryptotrader.scheduler import Scheduler

    document = runtime_document(
        notifications=NotificationConfig(webhook_url="https://example.test/summary", enabled=enabled, events=events)
    )
    url = f"sqlite+aiosqlite:///{tmp_path}/summary.db"
    await migrate_alerts(url)
    store = AlertStore(url)

    async def notification_config():
        return document.notifications

    runtime = SimpleNamespace(
        snapshot=SimpleNamespace(document=document, revision=1),
        alerts=AlertService(store, notification_config),
        alert_owner=None,
    )
    scheduler = Scheduler(document.scheduler, runtime=runtime)
    await scheduler._emit_daily_summary()
    await scheduler._emit_daily_summary()
    alerts = await store.list_alerts()
    deliveries = await store.list_deliveries()
    assert len(alerts) == 1
    assert alerts[0].type == "daily_summary"
    assert alerts[0].event_key.startswith("daily_summary:")
    assert "example.test" not in alerts[0].message
    assert len(deliveries) == sends
