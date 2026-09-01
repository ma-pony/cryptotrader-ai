# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""One admission and result persistence path for HTTP, CLI and scripts."""

import asyncio

from cryptotrader.backtest.models import BacktestParams
from cryptotrader.backtest.snapshot import restore_snapshot, safe_snapshot
from cryptotrader.backtest.store import BacktestStore
from cryptotrader.tasks import BackgroundTaskManager
from cryptotrader.venues.models import BacktestCostModel


class BacktestService:
    def __init__(self, *, repository, task_manager, engine_factory=None, registry_provider=None):
        self.repository = repository
        self.task_manager = task_manager
        self.store = BacktestStore(repository.database_url)
        self.engine_factory = engine_factory
        self.registry_provider = registry_provider or self._registries

    async def start(self, params, *, snapshot=None):
        params = BacktestParams.model_validate(params)
        if any(
            getattr(params, key) is None
            for key in ("interval", "initial_equity", "fee_rate", "slippage_bps", "funding_assumption")
        ):
            raise ValueError("启动回测必须明确周期、本金和成本条件")
        current = snapshot or await self.repository.get_existing()
        frozen = await self._frozen_snapshot(params, current)
        replay = restore_snapshot(frozen, current)
        run_id = await self.store.create(params, frozen)
        evidence_records = []

        async def canceled():
            run = await self.store.get(run_id)
            await self.store.update(run_id, "canceled", run.progress, model_evidence=evidence_records)

        async def work(_interrupt):
            nonlocal evidence_records
            from cryptotrader.backtest.engine import BacktestEngine
            from cryptotrader.backtest.evidence import capture_model_evidence

            async def progress(value):
                await self.store.update(run_id, "running", max(0, min(0.99, value)))

            try:
                await progress(0)
                with capture_model_evidence() as evidence:
                    evidence_records = evidence.records
                    signals, markets = await self.registry_provider(replay)
                    engine = (self.engine_factory or BacktestEngine)(
                        pair=params.pair,
                        start=params.start,
                        end=params.end,
                        interval=params.interval,
                        initial_capital=params.initial_equity,
                        snapshot=replay,
                        signal_registry=signals,
                        market_registry=markets,
                        progress_callback=progress,
                        cost_model=BacktestCostModel(
                            params.fee_rate, params.slippage_bps, params.funding_assumption == "available_only"
                        ),
                    )
                    result = await engine.run()
                await self.store.update(run_id, "completed", 1, result, model_evidence=evidence_records)
            except asyncio.CancelledError:
                await canceled()
                raise
            except Exception as error:
                run = await self.store.get(run_id)
                await self.store.update(
                    run_id,
                    "failed",
                    run.progress,
                    error=f"回测失败（{type(error).__name__}），请检查数据源和组件配置。",
                    model_evidence=evidence_records,
                )

        try:
            self.task_manager.create(run_id, params.pair, work, "backtest", on_cancel=canceled)
        except Exception:
            await self.store.update(run_id, "failed", 0, error="回测未能排入任务，请稍后重试。")
            raise
        return run_id

    async def _frozen_snapshot(self, params, current):
        if not params.snapshot_run_id:
            return safe_snapshot(current)
        previous = await self.store.get(params.snapshot_run_id)
        if previous is None:
            raise LookupError("历史回测不存在")
        if previous.config_snapshot is None:
            raise ValueError("历史回测没有可复用的配置快照")
        return previous.config_snapshot

    async def _registries(self, snapshot):
        from cryptotrader.cycle_events import NullCycleEventSink
        from cryptotrader.market_sources.registry import MarketSourceRegistry
        from cryptotrader.runtime_config.repository import LLM_GATEWAY_CREDENTIAL_REF
        from cryptotrader.signals.registry import SignalComponentRegistry

        key = ""
        if any(item.enabled and item.component_id == "llm_committee" for item in snapshot.document.signals.components):
            key = (await self.repository.reveal_token(LLM_GATEWAY_CREDENTIAL_REF)).token
        return (
            SignalComponentRegistry.discover(snapshot.document, NullCycleEventSink(), llm_gateway_key=key),
            MarketSourceRegistry.discover(snapshot.document.market_data),
        )

    async def cancel(self, run_id):
        run = await self.store.get(run_id)
        if run is None:
            raise LookupError("回测不存在")
        if run.status not in {"queued", "running"} or self.task_manager.interrupt(run_id) is None:
            raise ValueError("回测已结束或正在停止")
        # Wait for this run only; never wait for unrelated real order-bearing tasks.
        task = self.task_manager.get(run_id)
        await asyncio.gather(task.task, return_exceptions=True)
        await self.store.update(run_id, "canceled", (await self.store.get(run_id)).progress)

    async def run(self, params, *, snapshot=None):
        run_id = await self.start(params, snapshot=snapshot)
        await self.task_manager.get(run_id).task
        return await self.store.get(run_id)


def configured_service():
    """CLI bootstrap uses the same service without constructing trading accounts."""
    from cryptotrader.bootstrap import BootstrapSettings
    from cryptotrader.runtime_config.repository import RuntimeConfigRepository
    from cryptotrader.runtime_config.secrets import CredentialVault

    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    return BacktestService(repository=repository, task_manager=BackgroundTaskManager())
