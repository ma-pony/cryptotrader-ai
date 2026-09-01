"""Reconcile alerts from committed domain facts, never from raw event payloads."""
# ruff: noqa: RUF001 -- Chinese operator messages retain Chinese punctuation.

from cryptotrader.alerts.models import BusinessAlertEvent


class AlertRecovery:
    def __init__(self, alerts, *, journal, approvals, accounts=None, operations=None, scope_provider=None):
        self.alerts = alerts
        self.store = alerts.store
        self.journal = journal
        self.approvals = approvals
        self.accounts = accounts
        self.operations = operations
        self.scope_provider = scope_provider

    async def reconcile(self):
        await self._decisions()
        await self._approvals()
        scopes = await self.scope_provider() if self.scope_provider else {}
        if self.accounts is not None:
            await self._connections(scopes)
        if self.operations is not None:
            await self._operations(scopes)

    async def _decisions(self):  # noqa: C901 -- one pass maps the closed decision aggregate
        offset = 0
        while records := await self.journal.list(limit=200, offset=offset):
            for record in records:
                if record.run.mode == "backtest" or record.cycle_status in {"queued", "running"}:
                    continue
                for component in record.component_signals:
                    if component.status == "failed":
                        await self.alerts.record(
                            BusinessAlertEvent(
                                event_key=f"component:{record.cycle_id}:{component.component_id}:failed",
                                type="component_failed",
                                occurred_at=record.created_at,
                                decision_id=record.cycle_id,
                                pair=record.run.pair,
                                message="信号组件失败，请查看决策证据；该事项不代表账户已执行交易。",
                            )
                        )
                if record.run.mode != "trading":
                    continue
                for book in record.book_results:
                    common = {
                        "occurred_at": record.created_at,
                        "decision_id": record.cycle_id,
                        "book_id": book.book_id,
                        "capital_scope": book.capital_scope,
                        "pair": record.run.pair,
                    }
                    key = f"book:{record.cycle_id}:{book.book_id}"
                    if (
                        book.proposal
                        and book.proposal.risk.requested_target_exposure != book.proposal.risk.capped_target_exposure
                    ):
                        await self.alerts.record(
                            BusinessAlertEvent(
                                event_key=f"{key}:risk_adjusted",
                                type="risk_adjusted",
                                message="本次目标已按风险限制压低，无需批准该限制。",
                                **common,
                            )
                        )
                    if book.reconciliation_required or book.status in {"failed", "partial"}:
                        message = (
                            "账户账本或执行后核对未完成，请先刷新账户核对；不要重复下单。"
                            if book.reconciliation_required
                            or (
                                book.execution
                                and any(
                                    item.error_operation == "account_ledger"
                                    for item in book.execution.connection_results
                                )
                            )
                            else "执行未完整完成，请查看决策与账户实际结果；通知重试不会重新下单。"
                        )
                        if book.execution is None:
                            message = "执行准备未完成，请查看决策中的风险或连接检查结果。"
                        await self.alerts.record(
                            BusinessAlertEvent(
                                event_key=f"{key}:execution_failed", type="execution_failed", message=message, **common
                            )
                        )
                    if book.execution:
                        for result in book.execution.connection_results:
                            if "protection" in result.error_operation:
                                await self.alerts.record(
                                    BusinessAlertEvent(
                                        event_key=f"{key}:{result.connection_id}:protection_failed",
                                        type="protection_failed",
                                        connection_id=result.connection_id,
                                        message="保护处理失败，请核对现有持仓与保护状态；不要重复下单。",
                                        **common,
                                    )
                                )
            offset += len(records)

    async def _approvals(self):
        for approval in await self.approvals.list_all():
            # Only persisted trading approvals are actionable; never infer execution from approval claims.
            record = await self.journal.get(approval.cycle_id)
            if record is not None and record.run.mode != "trading":
                continue
            event = BusinessAlertEvent(
                event_key=f"approval:{approval.id}:pending",
                type="approval_pending",
                occurred_at=approval.created_at,
                decision_id=approval.cycle_id,
                book_id=approval.book_id,
                capital_scope=approval.proposal.capital_scope,
                pair=approval.proposal.pair.canonical(),
                message="执行计划等待人工审批；已读不会批准或下单。",
            )
            identity = await self.alerts.record(event)
            if approval.status == "invalidated" or (
                approval.status == "pending" and self.approvals.is_expired(approval, self.store.clock())
            ):
                await self.store.resolve(identity, "expired")
            elif approval.status != "pending":
                await self.store.resolve(identity, "approval_processed")

    async def _connections(self, scopes):
        for state in await self.accounts.sync_states():
            connection_id = state["connection_id"]
            success = state["last_success_at"]
            key = f"connection:{connection_id}:{success.isoformat() if success else 'initial'}:failed"
            for alert in await self.store.list_alerts(
                connection_id=connection_id, type="connection_failed", resolution="open"
            ):
                if not state["failure_reason"] or alert.event_key != key:
                    await self.store.resolve(alert.id, "recovered")
            if state["failure_reason"]:
                await self.alerts.record(
                    BusinessAlertEvent(
                        event_key=key,
                        type="connection_failed",
                        occurred_at=state["last_failure_at"],
                        connection_id=connection_id,
                        capital_scope=scopes.get(connection_id),
                        message="账户同步失败，请检查连接配置、权限与历史覆盖范围。",
                    )
                )

    async def _operations(self, scopes):
        operations = await self.operations.list_all()
        for operation in operations:
            if operation.status == "failed" or operation.result.reconciliation_required:
                message = "人工操作未完成，请刷新账户核对；不会自动重试下单。"
                if operation.result.orders:
                    message = "人工操作保留了订单回包，请刷新账户核对实际成交与剩余持仓；不会自动重试下单。"
                await self.alerts.record(
                    BusinessAlertEvent(
                        event_key=f"operation:{operation.operation_id}:failed",
                        type="execution_failed",
                        occurred_at=operation.updated_at,
                        operation_id=operation.operation_id,
                        connection_id=operation.connection_id,
                        book_id=operation.plan.book_id if operation.plan else None,
                        capital_scope=operation.plan.capital_scope
                        if operation.plan
                        else scopes.get(operation.connection_id),
                        pair=operation.pair,
                        message=message,
                    )
                )
        # Resolve only actual completed flatten operations, not revised confirmations or preparation.
        for operation in operations:
            if (
                operation.status != "completed"
                or operation.kind != "flatten"
                or operation.result.reconciliation_required
                or operation.result.remaining_position != 0
            ):
                continue
            for alert in await self.store.list_alerts(connection_id=operation.connection_id, resolution="open"):
                if (
                    alert.type in {"execution_failed", "protection_failed"}
                    and alert.pair == operation.pair
                    and alert.occurred_at <= operation.updated_at
                ):
                    await self.store.resolve(alert.id, "exit_completed")
