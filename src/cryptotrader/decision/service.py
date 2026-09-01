"""Durable run admission; background ownership is separate from HTTP lifetime."""

import asyncio
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

from cryptotrader.decision.analysis import AnalysisFailure
from cryptotrader.journal.models import DecisionRun, MultiVenueCycleRecord
from cryptotrader.tasks import TaskManagerClosedError


class CapabilityUnavailableError(RuntimeError):
    pass


def configuration_summary(snapshot):
    """Allowlisted business input only; no endpoints, credential refs or tokens."""
    document = snapshot.document
    from cryptotrader.configuration.registry import get_extension_registry

    extensions = get_extension_registry()
    registrations = extensions.components
    market = document.market_data
    market_registration = extensions.market_sources.get(market.source_id)
    market_parameters = dict(market.parameters)
    if market_registration is not None:
        market_parameters = market_registration.configuration.parameter_model.model_validate(
            market_parameters
        ).model_dump(mode="json")

    def component_snapshot(component):
        registration = registrations.get(component.component_id)
        values = dict(component.parameters)
        if registration is not None:
            model = registration.configuration.parameter_model
            values = model.model_validate(values).model_dump(mode="json")
        identity = {}
        if component.component_id == "kronos" and "tokenizer_name" in values:
            # A vocabulary model is not an access token. Store its explicit
            # model identity without relaxing secret-key checks anywhere.
            identity = {"prediction": values["model_name"], "vocabulary": values.pop("tokenizer_name")}
        return {
            "component_id": component.component_id,
            "enabled": component.enabled,
            "weight": component.weight,
            "parameters": values,
            "model_identity": identity,
        }

    return {
        "revision": snapshot.revision,
        "market_data": {"source_id": market.source_id, "timeframe": market.timeframe, "parameters": market_parameters},
        "signals": {
            "components": [component_snapshot(item) for item in document.signals.components],
            "neutral_threshold": document.signals.neutral_threshold,
            "max_target_ratio": document.signals.max_target_ratio,
            "evaluation_interval": document.signals.evaluation_interval,
            "atr_stop_multiplier": document.signals.atr_stop_multiplier,
            "reward_ratio": document.signals.reward_ratio,
        },
        "llm": {
            "models": document.llm.models.model_dump(mode="json"),
            "default_temperature": document.llm.default_temperature,
        },
        "books": [
            {
                "id": item.id,
                "enabled": item.enabled,
                "capital_scope": item.capital_scope,
                "hitl_required": item.hitl_required,
                "allocations": [
                    {
                        "connection_id": allocation.connection_id,
                        "enabled": allocation.enabled,
                        "weight": allocation.weight,
                    }
                    for allocation in item.allocations
                ],
            }
            for item in document.execution.books
        ],
    }


class RunService:
    def __init__(self, *, analysis_provider, journal, task_manager, runtime=None, events=None, clock=None):
        self.analysis_provider = analysis_provider
        self.journal = journal
        self.task_manager = task_manager
        self.clock = clock or (lambda: datetime.now(UTC))
        self.runtime = runtime
        self.events = events

    async def readiness(self):
        from cryptotrader.decision.readiness import (
            CapabilityOut,
            ReadinessOut,
            book_scope,
            component_readiness,
            reason,
            trading_reasons,
        )

        snapshot = await self.runtime.repository.get_or_create()
        components, analysis = await component_readiness(snapshot, self.runtime.repository)
        books = await book_scope(snapshot, self.runtime.repository)
        reasons = list(analysis.reasons) + trading_reasons(snapshot)
        if not any(book.eligible for book in books):
            reasons.append(reason("no_eligible_books", "没有可参与执行的资金池。", "execution.books"))
        latest = await self.journal.list(limit=1)
        return ReadinessOut(
            analysis=analysis,
            trading=CapabilityOut(ready=not reasons, reasons=tuple(reasons)),
            components=components,
            saved_revision=snapshot.revision,
            applied_revision=snapshot.applied_revision,
            apply_error=snapshot.apply_error,
            automation_enabled=snapshot.document.scheduler.automation_enabled,
            latest_run_at=latest[0].created_at if latest else None,
            execution_pairs=snapshot.document.execution.pairs,
        )

    async def trading_scope(self, pair):
        snapshot = await self.runtime.repository.get_or_create()
        return await self._scope(pair, snapshot)

    async def _scope(self, pair, snapshot):
        from cryptotrader.decision.readiness import (
            TradingScopeOut,
            book_scope,
            component_readiness,
            reason,
            trading_reasons,
        )
        from cryptotrader.pair import Pair

        pair = Pair.parse(pair) if isinstance(pair, str) else pair
        books = await book_scope(snapshot, self.runtime.repository)
        _, analysis = await component_readiness(snapshot, self.runtime.repository)
        reasons = list(analysis.reasons) + trading_reasons(snapshot)
        if pair.canonical() not in snapshot.document.execution.pairs:
            reasons.append(reason("pair_out_of_scope", "该品种不在已保存的交易范围内。", "execution.pairs"))
        if not any(book.eligible for book in books):
            reasons.append(reason("no_eligible_books", "没有可参与执行的资金池。", "execution.books"))
        return TradingScopeOut(
            pair=pair.canonical(),
            saved_revision=snapshot.revision,
            ready=not reasons,
            reasons=tuple(reasons),
            books=books,
        )

    async def set_automation(self, enabled, expected_revision):
        snapshot = await self.runtime.repository.get_or_create()
        if snapshot.revision != expected_revision:
            raise ValueError("configuration revision changed")
        document = snapshot.document.model_copy(
            update={"scheduler": snapshot.document.scheduler.model_copy(update={"automation_enabled": enabled})}
        )
        return await self.runtime.apply_automation(document, expected_revision)

    async def start_trading(self, pair, expected_revision, confirmed_book_ids):
        snapshot = await self.runtime.repository.get_or_create()
        if snapshot.revision != expected_revision:
            raise ValueError("configuration revision changed")
        scope = await self._scope(pair, snapshot)
        eligible = tuple(book.book_id for book in scope.books if book.eligible)
        if len(confirmed_book_ids) != len(set(confirmed_book_ids)) or set(confirmed_book_ids) != set(eligible):
            raise ValueError("trading scope changed; confirm all eligible books")
        if not scope.ready:
            raise ValueError("trading is not ready")
        return await self._queue_trading(scope.pair, snapshot, eligible, "manual")

    async def run_automatic(self, pair, source):
        if source not in {"scheduled", "trigger"}:
            raise ValueError("invalid automatic source")
        snapshot = await self.runtime.repository.get_or_create()
        document = snapshot.document
        if not document.scheduler.automation_enabled:
            return None
        if not (document.scheduler.enabled if source == "scheduled" else document.triggers.enabled):
            return None
        scope = await self._scope(pair, snapshot)
        if not scope.ready:
            return None
        return await self._queue_trading(
            scope.pair, snapshot, tuple(book.book_id for book in scope.books if book.eligible), source
        )

    async def _queue_trading(self, pair, snapshot, book_ids, origin):
        from cryptotrader.decision.models import CycleRequest
        from cryptotrader.pair import Pair

        record = MultiVenueCycleRecord(
            str(uuid4()),
            snapshot.revision,
            snapshot.document.market_data.source_id,
            (),
            None,
            None,
            (),
            "queued",
            "not_started",
            False,
            self._now(),
            DecisionRun(pair, "trading", origin, configuration_summary(snapshot), None, None),
        )
        await self.journal.save(record)

        async def work(_interrupt):
            manager = self.task_manager

            class ExecutionStartedSink:
                async def publish(self, event):
                    if event.name == "book_execution_started":
                        manager.get(record.cycle_id).orders_started = True

            try:
                async with self.runtime.execution_lease(
                    pair,
                    expected_revision=snapshot.revision,
                    confirmed_book_ids=book_ids,
                    origin=origin,
                ) as cycle:
                    with self.runtime.events.route(ExecutionStartedSink()):
                        return await cycle.run(
                            CycleRequest(
                                Pair.parse(pair),
                                origin=origin,
                                decision_id=record.cycle_id,
                                confirmed_book_ids=book_ids,
                            )
                        )
            except asyncio.CancelledError:
                await terminal("cancelled", "cancelled")
                raise
            except Exception:
                await terminal("failed", "admission_or_run_failed")

        async def terminal(status, code):
            current = await self.journal.get(record.cycle_id)
            if current is not None and current.cycle_status in {"queued", "running"}:
                await self._fail(current, status, code)

        try:
            self.task_manager.create(
                record.cycle_id, pair, work, origin, on_cancel=lambda: terminal("cancelled", "cancelled")
            )
        except TaskManagerClosedError:
            await self._fail(record, "cancelled", "admission_closed")
            raise
        except Exception:
            await self._fail(record, "failed", "queue_rejected")
            raise
        return record.cycle_id

    async def _check_analysis_readiness(self, expected_revision):
        if self.runtime is not None:
            from cryptotrader.decision.readiness import component_readiness

            saved = await self.runtime.repository.get_or_create()
            if saved.revision != expected_revision:
                raise ValueError("configuration revision changed")
            _, capability = await component_readiness(saved, self.runtime.repository)
            if not capability.ready:
                raise CapabilityUnavailableError("analysis dependencies are not ready")

    async def start_analysis(self, pair, expected_revision: int) -> str:
        await self._check_analysis_readiness(expected_revision)
        # The service instance and its configured component instances are frozen
        # together with the snapshot, before the coroutine is queued.
        snapshot, analysis = await self.analysis_provider(expected_revision)
        created_at = self._now()
        record = MultiVenueCycleRecord(
            str(uuid4()),
            snapshot.revision,
            snapshot.document.market_data.source_id,
            (),
            None,
            None,
            (),
            "queued",
            "not_started",
            False,
            created_at,
            DecisionRun(pair.canonical(), "analysis", "manual", configuration_summary(snapshot), None, None),
        )
        await self.journal.save(record)

        async def work(_interrupt):
            running = replace(record, cycle_status="running")
            try:
                await self.journal.replace(running)
                with (
                    self.events.cycle(record.cycle_id, snapshot.revision) if self.events is not None else nullcontext()
                ):
                    result = await analysis.analyze(pair, snapshot, created_at)
                await self.journal.replace(
                    replace(
                        running,
                        component_signals=result.component_signals,
                        fused_signal=result.fused_signal,
                        target_position=result.target_position,
                        cycle_status="failed" if result.failure else "completed",
                        run=replace(record.run, finished_at=self._now(), failure=result.failure),
                    )
                )
            except asyncio.CancelledError:
                await self._fail(running, "cancelled", "cancelled")
                raise
            except Exception:
                await self._fail(running, "failed", "run_failed")

        try:

            async def cancelled():
                current = await self.journal.get(record.cycle_id)
                if current is not None and current.cycle_status in {"queued", "running"}:
                    await self._fail(current, "cancelled", "cancelled")

            self.task_manager.create(record.cycle_id, pair.canonical(), work, "manual", on_cancel=cancelled)
        except TaskManagerClosedError:
            await self._fail(record, "cancelled", "admission_closed")
            raise
        except Exception:
            await self._fail(record, "failed", "queue_rejected")
            raise
        return record.cycle_id

    async def _fail(self, record, status, code):
        if record.run.mode == "trading" and status == "failed":
            status = "cycle_failed"
        await self.journal.replace(
            replace(
                record,
                cycle_status=status,
                run=replace(
                    record.run,
                    finished_at=self._now(),
                    failure=AnalysisFailure(code=code, stage="runtime", message="运行已停止。未执行交易。"),
                ),
            )
        )

    def _now(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("analysis clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
