"""User-confirmed account actions, entirely independent of models and strategy approvals."""
# ruff: noqa: RUF001 -- User-facing Chinese messages retain Chinese punctuation.

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from uuid import uuid4

from cryptotrader.accounts.models import AccountOperationOut, ExitPlan, OperationOrderOut
from cryptotrader.accounts.store import AccountOperationStore, OperationConflictError, effective_memberships, payload
from cryptotrader.execution.service import VenueExecutionService
from cryptotrader.execution_ownership import ExecutionOwnership, wait_for_owned
from cryptotrader.pair import Pair
from cryptotrader.risk.book_state import BookRiskStateStore
from cryptotrader.runtime_config.repository import RevisionConflict


def connection_in(document, connection_id):
    connection = next((c for c in document.execution.connections if c.id == connection_id), None)
    if connection is None:
        raise LookupError("账户连接不存在或已归档")
    return connection


def ownership(document, connection_id):
    book_id = effective_memberships(document).get(connection_id)
    owner = ExecutionOwnership(document.infrastructure.redis_url)
    return owner.book(book_id) if book_id else owner.connection(connection_id)


def stopped(document, connection_id):
    book_id = effective_memberships(document).get(connection_id)
    if book_id:
        return not next(b for b in document.execution.books if b.id == book_id).enabled
    return not connection_in(document, connection_id).enabled


def removal_ids(before, after):
    old_members = effective_memberships(before)
    new_members = effective_memberships(after)
    removed = {c.id for c in before.execution.connections} - {c.id for c in after.execution.connections}
    return removed | {cid for cid, book_id in old_members.items() if new_members.get(cid) != book_id}


class AccountOperationService:
    def __init__(self, runtime):
        self.runtime = runtime
        self.accounts = runtime.repository.account_store
        self.store = AccountOperationStore(self.accounts.database_url)
        self._tasks = set()
        self._closed = False

    async def get(self, operation_id):
        return await self.store.get(operation_id)

    def _start(self, work):
        task = asyncio.create_task(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def close(self):
        self._closed = True
        if self._tasks:
            await wait_for_owned(asyncio.gather(*self._tasks, return_exceptions=True))

    async def prepare(self, connection_id, pair, kind, expected_revision, confirm_stop):
        if self._closed:
            raise OperationConflictError("人工操作服务正在关闭")
        if not confirm_stop:
            raise ValueError("请先确认停用范围")
        if kind not in {"cancel_orders", "flatten"}:
            raise ValueError("不支持的人工操作")
        pair = Pair.parse(pair).canonical()
        current = await self.runtime.repository.get_existing()
        connection_in(current.document, connection_id)
        if current.revision != expected_revision:
            raise RevisionConflict(expected_revision, current.revision)
        now = datetime.now(UTC)
        operation = AccountOperationOut(
            operation_id=uuid4().hex,
            connection_id=connection_id,
            pair=pair,
            kind=kind,
            status="preparing",
            created_at=now,
            updated_at=now,
        )
        await self.accounts.ensure_tables()
        await self.store.create(operation)
        self._start(self._prepare(operation, current))
        return operation.operation_id

    async def _prepare(self, operation, current):
        try:
            document = current.document
            book_id = effective_memberships(document).get(operation.connection_id)
            execution = document.execution
            if book_id:
                execution = execution.model_copy(
                    update={
                        "books": tuple(replace(b, enabled=False) if b.id == book_id else b for b in execution.books)
                    }
                )
            else:
                execution = execution.model_copy(
                    update={
                        "connections": tuple(
                            replace(c, enabled=False) if c.id == operation.connection_id else c
                            for c in execution.connections
                        )
                    }
                )
            # CAS is persisted before publication; waiting for the pool happens only afterwards.
            await self.runtime.apply_automation(document.model_copy(update={"execution": execution}), current.revision)
            async with self.runtime.account_operation_lease():
                current = await self.runtime.repository.get_existing()
                async with ownership(current.document, operation.connection_id):
                    current = await self.runtime.repository.get_existing()
                    self._assert_stopped(current.document, operation.connection_id, book_id)
                    async with self.runtime.account_session(operation.connection_id) as session:
                        snapshot = await self._read(session)
                        plan = self._plan(operation, snapshot, current.document, 1)
                        await self._refresh_book(plan, snapshot)
                        await self.store.update(
                            operation.model_copy(update={"status": "awaiting_confirmation", "plan": plan}),
                            expected_status="preparing",
                        )
        except asyncio.CancelledError:
            await wait_for_owned(self._fail(operation.operation_id, "操作准备已中断，请重新读取账户"))
            raise
        except Exception:
            await self._fail(operation.operation_id, "退出计划准备失败，请重新读取配置和账户后重试")

    def _plan(self, operation, snapshot, document, version):
        pair = Pair.parse(operation.pair)
        positions = [p for p in snapshot.positions if p.instrument.pair == pair]
        orders = [o for o in snapshot.orders if o.instrument.pair == pair]
        if len(positions) > 1 or any(
            not p.instrument.tradable or p.instrument.market_type != pair.market_type for p in positions
        ):
            raise ValueError("该持仓不可通过标准退出操作处理")
        if any(not o.instrument.tradable or o.instrument.market_type != pair.market_type for o in orders):
            raise ValueError("该挂单不可通过标准退出操作处理")
        if snapshot.completeness:
            raise ValueError("账户事实不完整，不能确认退出数量")
        amount = positions[0].signed_amount if positions else Decimal("0")
        close = abs(amount) if operation.kind == "flatten" else Decimal("0")
        if pair.market_type == "spot" and close:
            available = positions[0].available_amount
            if available is None or amount < 0:
                raise ValueError("现货可用数量未知")
            close = min(close, available)
        book_id = effective_memberships(document).get(operation.connection_id)
        return ExitPlan(
            operation_id=operation.operation_id,
            version=version,
            connection_id=operation.connection_id,
            book_id=book_id,
            capital_scope=snapshot.capital_scope,
            pair=operation.pair,
            kind=operation.kind,
            stopped_scope=[f"book:{book_id}" if book_id else f"connection:{operation.connection_id}"],
            ordinary_order_ids=sorted(o.venue_order_id for o in orders if not o.protection),
            position_amount=amount,
            close_amount=close,
            protection_ids=sorted(o.venue_order_id for o in orders if o.protection),
            snapshot_time=snapshot.observed_at,
        )

    async def execute(self, operation_id, plan_version):
        if self._closed:
            raise OperationConflictError("人工操作服务正在关闭")
        operation = await self.store.get(operation_id)
        if operation.status != "awaiting_confirmation" or operation.plan.version != plan_version:
            raise OperationConflictError("操作状态或计划版本已变化，请重新读取")
        current = await self.runtime.repository.get_existing()
        self._assert_authorized(current.document, operation.plan)
        result = operation.result.model_copy(update={"failure_reason": None})
        executing = operation.model_copy(update={"status": "executing", "result": result})
        await self.store.update(executing, expected_status="awaiting_confirmation", expected_version=plan_version)
        self._start(self._execute(executing))
        return operation_id

    @staticmethod
    def _assert_stopped(document, connection_id, book_id):
        connection_in(document, connection_id)
        if effective_memberships(document).get(connection_id) != book_id or not stopped(document, connection_id):
            raise OperationConflictError("停用范围已变化，请重新准备退出计划")

    def _assert_authorized(self, document, plan):
        from cryptotrader.configuration.catalog import require_environment

        connection = connection_in(document, plan.connection_id)
        scope = require_environment(connection.adapter_id, connection.environment).capital_scope
        self._assert_stopped(document, plan.connection_id, plan.book_id)
        if scope != plan.capital_scope:
            raise OperationConflictError("账户资金性质已变化")
        if scope == "real" and not document.execution.live_order_execution_enabled:
            raise PermissionError("尚未授权真实账户交易")

    @staticmethod
    def _same_facts(left, right):
        return all(
            getattr(left, key) == getattr(right, key)
            for key in (
                "position_amount",
                "close_amount",
                "ordinary_order_ids",
                "protection_ids",
                "capital_scope",
                "book_id",
            )
        )

    async def _read(self, session):
        snapshot = await session.fetch_account()
        if snapshot.connection_id != session.connection_id:
            raise ValueError("账户身份不匹配")
        await self.accounts.ingest(snapshot)
        return snapshot

    async def _save_result(self, operation, result):
        return await self.store.update(operation.model_copy(update={"result": result}), expected_status="executing")

    async def _execute(self, operation):  # noqa: C901 -- Ordered manual safety stages remain explicit.
        plan, result = operation.plan, operation.result.model_copy(deep=True)
        try:
            async with self.runtime.account_operation_lease():
                current = await self.runtime.repository.get_existing()
                async with ownership(current.document, plan.connection_id):
                    current = await self.runtime.repository.get_existing()
                    self._assert_authorized(current.document, plan)
                    async with self.runtime.account_session(plan.connection_id) as session:
                        # Fresh account read also primes adapter-owned cancellation routing in this session.
                        fresh = await self._read(session)
                        observed = self._plan(operation, fresh, current.document, plan.version)
                        self._remaining(result, observed)
                        await self._save_result(operation, result)
                        if not self._same_facts(plan, observed):
                            await self.store.invalidate(operation.operation_id, "持仓或挂单已变化，请重新确认")
                            return
                        needed = {"cancel_order"} if plan.ordinary_order_ids else set()
                        if plan.kind == "flatten":
                            needed |= {"close_position", "cancel_protection"}
                        if not needed <= session.capabilities.exit_operations:
                            raise ValueError("平台不支持所需退出操作")
                        for order_id in plan.ordinary_order_ids:
                            await session.cancel_order(order_id, Pair.parse(plan.pair))
                            result.canceled_order_ids.append(order_id)
                            await self._save_result(operation, result)
                        fresh = await self._read(session)
                        after = self._plan(operation, fresh, current.document, plan.version + 1)
                        self._remaining(result, after)
                        await self._save_result(operation, result)
                        if after.ordinary_order_ids:
                            raise ValueError("普通挂单尚未撤完")
                        if plan.kind == "flatten" and (
                            after.position_amount != plan.position_amount
                            or after.close_amount != plan.close_amount
                            or after.protection_ids != plan.protection_ids
                        ):
                            result.failure_reason = "撤单期间账户数量已变化，请确认新计划"
                            await self.store.update(
                                operation.model_copy(
                                    update={"status": "awaiting_confirmation", "plan": after, "result": result}
                                ),
                                expected_status="executing",
                            )
                            return
                        if plan.kind == "flatten" and plan.close_amount:
                            executor = VenueExecutionService(
                                session,
                                connection=connection_in(current.document, plan.connection_id),
                                live_order_execution_enabled=current.document.execution.live_order_execution_enabled,
                                account_store=self.accounts,
                            )
                            order, audit_failed = await executor.close_frozen(plan)
                            result.orders.append(OperationOrderOut.model_validate(payload(order)))
                            result.reconciliation_required = audit_failed
                            await self._save_result(operation, result)
                        fresh = await self._read(session)
                        final = self._plan(operation, fresh, current.document, plan.version)
                        if plan.kind == "flatten" and final.position_amount == 0:
                            if final.protection_ids:
                                await session.cancel_protection(tuple(final.protection_ids))
                                result.canceled_protection_ids.extend(final.protection_ids)
                                await self._save_result(operation, result)
                            fresh = await self._read(session)
                            final = self._plan(operation, fresh, current.document, plan.version)
                        self._remaining(result, final)
                        await self._save_result(operation, result)
                        fresh = await self.runtime.account_sync.sync_session(session, plan.connection_id)
                        final = self._plan(operation, fresh, current.document, plan.version)
                        self._remaining(result, final)
                        await self._save_result(operation, result)
                        await self._refresh_book(plan, fresh)
                        if final.ordinary_order_ids or (
                            plan.kind == "flatten" and (final.position_amount != 0 or final.protection_ids)
                        ):
                            raise ValueError("退出未完成，保留剩余持仓与保护")
                        if result.reconciliation_required:
                            raise ValueError("实际执行已返回，账本需核对")
                        await self.store.update(
                            operation.model_copy(update={"status": "completed", "result": result}),
                            expected_status="executing",
                        )
        except asyncio.CancelledError:
            await wait_for_owned(self._fail(operation.operation_id, "执行已中断，请核对实际结果；不会自动重试", result))
            raise
        except Exception:
            # Never retry submission. Last receipts remain durable even if subsequent sync fails.
            await self._fail(operation.operation_id, "操作未完成，请刷新账户核对剩余持仓、挂单与保护", result)

    @staticmethod
    def _remaining(result, plan):
        result.remaining_position = plan.position_amount
        result.remaining_order_ids = plan.ordinary_order_ids
        result.remaining_protection_ids = plan.protection_ids
        result.observed_at = plan.snapshot_time

    async def _fail(self, operation_id, reason, result=None):
        operation = await self.store.get(operation_id)
        if operation.status not in {"preparing", "executing"}:
            return
        result = result or operation.result
        result.failure_reason = reason
        result.reconciliation_required = True
        if operation.plan:
            # A best-effort read is facts only, never another attempt to trade.
            try:
                async with self.runtime.account_operation_lease():
                    document = (await self.runtime.repository.get_existing()).document
                    async with ownership(document, operation.connection_id):
                        async with self.runtime.account_session(operation.connection_id) as session:
                            fresh = await self._read(session)
                        self._remaining(result, self._plan(operation, fresh, document, operation.plan.version))
                        await self._refresh_book(operation.plan, fresh)
            except Exception:
                result.remaining_position = None
                result.observed_at = None
        await self.store.update(
            operation.model_copy(update={"status": "failed", "result": result}), expected_status=operation.status
        )

    async def _refresh_book(self, plan, snapshot):
        if plan.book_id:
            document = (await self.runtime.repository.get_existing()).document
            snapshots = [snapshot]
            for cid, bid in effective_memberships(document).items():
                if bid == plan.book_id and cid != plan.connection_id:
                    latest = await self.accounts.latest(cid)
                    if latest is not None:
                        snapshots.append(latest)
                    else:
                        async with self.runtime.account_session(cid) as session:
                            snapshots.append(await self._read(session))
            await BookRiskStateStore(self.accounts).update(plan.book_id, snapshots)

    async def assert_account_removable(self, connection_id):
        """Caller holds application barrier, has drained leases, and holds account ownership."""
        document = (await self.runtime.repository.get_existing()).document
        if not stopped(document, connection_id):
            raise OperationConflictError("请先停用所属资金池或无归属连接")
        async with self.runtime.account_session(connection_id) as session:
            snapshot = await self._read(session)
        if snapshot.completeness or any(p.signed_amount != 0 for p in snapshot.positions) or snapshot.orders:
            raise OperationConflictError("账户仍有持仓、挂单或未知事实，不能移除")

    @asynccontextmanager
    async def removal_guard(self, before, after):
        ids = sorted(removal_ids(before, after))
        async with AsyncExitStack() as stack:
            if ids:
                await self.runtime.wait_for_execution_idle()
            locked = set()
            members = effective_memberships(before)
            for connection_id in ids:
                key = ("book", members[connection_id]) if connection_id in members else ("connection", connection_id)
                if key not in locked:
                    await stack.enter_async_context(ownership(before, connection_id))
                    locked.add(key)
                await self.assert_account_removable(connection_id)
            yield
