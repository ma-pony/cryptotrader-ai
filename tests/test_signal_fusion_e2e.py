"""Cross-layer acceptance tests for dynamic profile fusion and strict failure."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from cryptotrader.cycle_events import NullCycleEventSink
from cryptotrader.decision.engine import DecisionEngine
from cryptotrader.decision.exit_policy import AtrExitPolicy
from cryptotrader.execution.planner import ExecutionPlanner
from cryptotrader.execution.service import ExecutionResult
from cryptotrader.hitl.store import ApprovalStore
from cryptotrader.risk.models import RiskDecision
from cryptotrader.signals.fusion import WeightedSignalFusion
from cryptotrader.signals.models import ComponentSignal, DataRequirements
from cryptotrader.signals.registry import SignalComponentRegistry
from cryptotrader.signals.runner import ComponentRunner
from cryptotrader.trading_cycle import TradingCycle
from tests.factories.signal_fusion import context, profile, request


class _SignalComponent:
    def __init__(self, component_id: str, signal: ComponentSignal | None = None, error: Exception | None = None):
        self.id = component_id
        self.display_name = component_id
        self.description = f"{component_id} acceptance component"
        self.signal = signal
        self.error = error

    def requirements(self) -> DataRequirements:
        return DataRequirements()

    async def evaluate(self, _context):
        if self.error is not None:
            raise self.error
        assert self.signal is not None
        return self.signal


class _Contexts:
    async def collect(self, cycle_request, _requirements):
        return replace(context(), mode=cycle_request.mode)

    async def refresh_execution_state(self, stored_context):
        return stored_context


class _Risk:
    async def check(self, risk_request, _portfolio):
        return RiskDecision(True, risk_request.plan)


class _Executor:
    def __init__(self) -> None:
        self.plans = []

    async def execute(self, plan, _context):
        self.plans.append(plan)
        return ExecutionResult(True, (), None, None)


async def _build_system(tmp_path, *, llm_error: Exception | None = None):
    from cryptotrader.journal.store import CycleJournalStore
    from cryptotrader.profiles.repository import SignalProfileRepository

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'fusion-e2e.db'}"
    profiles = SignalProfileRepository(database_url)
    await profiles.get_or_create(profile())
    components = (
        _SignalComponent("kronos", ComponentSignal("kronos", "long", 0.8, "Kronos long")),
        _SignalComponent(
            "llm_committee",
            ComponentSignal("llm_committee", "short", 0.3, "Committee short"),
            llm_error,
        ),
    )
    registry = SignalComponentRegistry(components)
    executor = _Executor()
    journal = CycleJournalStore(database_url)
    cycle = TradingCycle(
        mode="paper",
        profiles=profiles,
        registry=registry,
        contexts=_Contexts(),
        runner=ComponentRunner(NullCycleEventSink()),
        fusion=WeightedSignalFusion(),
        decisions=DecisionEngine(),
        exits=AtrExitPolicy(),
        approvals=ApprovalStore(database_url),
        risk=_Risk(),
        execution_planner=ExecutionPlanner(0.2),
        executor=executor,
        journal=journal,
        events=NullCycleEventSink(),
    )
    return profiles, registry, cycle, executor, journal


def _profile_payload(kronos: float, llm: float) -> dict:
    return {
        "components": [
            {"component_id": "kronos", "enabled": kronos > 0.0, "weight": kronos},
            {"component_id": "llm_committee", "enabled": llm > 0.0, "weight": llm},
        ],
        "neutral_threshold": 0.2,
        "max_target_ratio": 1.0,
        "atr_stop_multiplier": 2.0,
        "reward_ratio": 2.0,
        "hitl_required": False,
    }


@pytest.mark.asyncio
async def test_full_cycle_uses_web_saved_profile_on_next_cycle(tmp_path):
    from api.main import app

    profiles, registry, cycle, executor, journal = await _build_system(tmp_path)
    old_profiles = getattr(app.state, "signal_profile_repository", None)
    old_registry = getattr(app.state, "signal_registry", None)
    app.state.signal_profile_repository = profiles
    app.state.signal_registry = registry
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            first_save = await client.put("/api/signal-profile", json=_profile_payload(0.6, 0.4))
            assert first_save.status_code == 200
            first = await cycle.run(request())
            assert first.status == "completed"
            assert first.fused_signal is not None
            assert first.fused_signal.score == pytest.approx(0.36)

            second_save = await client.put("/api/signal-profile", json=_profile_payload(1.0, 0.0))
            assert second_save.status_code == 200
            second = await cycle.run(request())
    finally:
        app.state.signal_profile_repository = old_profiles
        app.state.signal_registry = old_registry

    assert second.profile_revision == first.profile_revision + 1
    assert second.fused_signal is not None
    assert second.fused_signal.score == pytest.approx(0.8)
    assert [item.component_id for item in second.component_signals] == ["kronos"]
    assert len(executor.plans) == 2
    assert (await journal.get(second.cycle_id)).profile_revision == second.profile_revision


@pytest.mark.asyncio
async def test_failed_component_records_cycle_and_places_no_order(tmp_path):
    _profiles, _registry, cycle, executor, journal = await _build_system(
        tmp_path,
        llm_error=RuntimeError("offline"),
    )

    outcome = await cycle.run(request())

    assert outcome.status == "component_failed"
    assert executor.plans == []
    record = await journal.get(outcome.cycle_id)
    assert record is not None
    assert record.component_error == {"llm_committee": "RuntimeError: offline"}
