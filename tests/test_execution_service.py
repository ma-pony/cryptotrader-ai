"""基于标准化 venue 的多连接执行服务语义。"""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal

import pytest

from cryptotrader.execution.models import ConnectionExecutionPlan
from cryptotrader.pair import Pair
from cryptotrader.venues.models import (
    ConnectionPosition,
    NormalizedOrder,
    OpenVenueState,
    ProtectionState,
    VenueCapabilities,
    VenueQuote,
)
from cryptotrader.venues.models import OrderIntent as VenueOrderIntent
from cryptotrader.venues.protocol import VenueOperationError
from tests.factories.runtime_config import connection

VENUE_PAIR = Pair.parse("BTC/USDT:USDT")
VENUE_CAPABILITIES = VenueCapabilities(
    frozenset({"swap"}),
    native_protection=True,
    hedge_mode=False,
    reduce_only=True,
    supported_order_types=frozenset({"market"}),
)
SPOT_PAIR = Pair.parse("BTC/USDT")
SPOT_CAPABILITIES = VenueCapabilities(
    frozenset({"spot"}),
    native_protection=False,
    hedge_mode=False,
    reduce_only=True,
    supported_order_types=frozenset({"market"}),
)


def _venue_protection(amount: str, *, side: str = "long", protection_id: str = "old") -> ProtectionState:
    return ProtectionState(
        (protection_id,),
        VENUE_PAIR,
        side,
        Decimal(amount),
        Decimal("90") if side == "long" else Decimal("110"),
        Decimal("120") if side == "long" else Decimal("80"),
        True,
        False,
    )


class _VenueSession:
    def __init__(
        self,
        current: str,
        *,
        protections: tuple[ProtectionState, ...] = (),
        failures: tuple[str, ...] = (),
        partial_fill: Decimal | None = None,
        quote: VenueQuote | None = None,
    ) -> None:
        self.connection_id = "paper-a"
        self.capabilities = VENUE_CAPABILITIES
        self.signed_amount = Decimal(current)
        self.protections = protections
        self.failures = list(failures)
        self.partial_fill = partial_fill
        self.quote = quote or VenueQuote(VENUE_PAIR, Decimal("100"), Decimal("100"), Decimal("100"))
        self.orders: list[VenueOrderIntent] = []
        self.calls: list[str] = []
        self._sequence = 0

    def _fail(self, operation: str) -> None:
        if self.failures and self.failures[0] == operation:
            self.failures.pop(0)
            raise VenueOperationError(f"RAW_SECRET_{operation}")

    async def fetch_quote(self, pair):
        self.calls.append("fetch_quote")
        self._fail("fetch_quote")
        return self.quote

    async def normalize_amount(self, pair, base_amount):
        self.calls.append("normalize_amount")
        self._fail("normalize_amount")
        return base_amount

    async def place_order(self, intent):
        self.calls.append("place_order")
        self.orders.append(intent)
        self._fail("place_order")
        self._sequence += 1
        filled = intent.amount if self.partial_fill is None else min(self.partial_fill, intent.amount)
        self.partial_fill = None
        self.signed_amount += filled if intent.side == "buy" else -filled
        status = "filled" if filled == intent.amount else "partially_filled"
        return NormalizedOrder(
            f"order-{self._sequence}",
            intent.pair,
            intent.side,
            intent.order_type,
            intent.amount,
            filled,
            self.quote.last if filled else None,
            status,
            intent.reduce_only,
        )

    async def replace_protection(self, spec):
        self.calls.append("replace_protection")
        self._fail("replace_protection")
        self.protections = (
            ProtectionState(
                (f"new-{self._sequence}",),
                spec.pair,
                spec.position_side,
                spec.amount,
                spec.stop_loss,
                spec.take_profit,
                True,
                False,
            ),
        )
        return self.protections[0]

    async def cancel_protection(self, protection_ids):
        self.calls.append("cancel_protection")
        self._fail("cancel_protection")
        ids = set(protection_ids)
        self.protections = tuple(
            protection for protection in self.protections if ids.isdisjoint(protection.protection_ids)
        )

    async def list_open_state(self, pair):
        self.calls.append("list_open_state")
        self._fail("list_open_state")
        position = ConnectionPosition(pair, self.signed_amount, self.signed_amount * self.quote.last, None)
        return OpenVenueState(position, (), self.protections)


def _venue_plan(
    current: str,
    target: str,
    *,
    planned_current: str | None = None,
    old_protection_ids: tuple[str, ...] = ("old",),
) -> ConnectionExecutionPlan:
    current_amount = Decimal(planned_current if planned_current is not None else current)
    target_amount = Decimal(target)
    side = "buy" if target_amount > current_amount else "sell"
    amount = abs(target_amount - current_amount)
    quote = VenueQuote(VENUE_PAIR, Decimal("100"), Decimal("100"), Decimal("100"))
    current_notional = current_amount * Decimal("100")
    target_notional = target_amount * Decimal("100")
    reduce_only = target_amount == 0 or (
        current_amount * target_amount > 0 and abs(target_amount) < abs(current_amount)
    )
    return ConnectionExecutionPlan(
        book_id="simulation",
        connection_id="paper-a",
        pair=VENUE_PAIR,
        current_signed_notional=current_notional,
        target_signed_notional=target_notional,
        delta_signed_notional=target_notional - current_notional,
        current_signed_amount=current_amount,
        target_signed_amount=target_amount,
        delta_signed_amount=target_amount - current_amount,
        post_fill_signed_amount=target_amount,
        quote=quote,
        execution_price=Decimal("100"),
        amount=amount,
        side=side,
        reduce_only=reduce_only,
        market_type="swap",
        stop_loss=None if target_amount == 0 else Decimal("90") if target_amount > 0 else Decimal("110"),
        take_profit=None if target_amount == 0 else Decimal("120") if target_amount > 0 else Decimal("80"),
        old_protection_ids=old_protection_ids,
        capabilities=VENUE_CAPABILITIES,
    )


@pytest.mark.asyncio
async def test_venue_service_reloads_state_and_uses_platform_protection_replacement_once():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "3"))

    assert result.status == "completed"
    assert session.orders[0].amount == Decimal("1")
    assert session.calls.count("replace_protection") == 1
    assert "cancel_protection" not in session.calls
    assert result.trace == ("pre_read", "place_order", "replace_protection", "reconcile")
    assert result.final_position is not None
    assert result.final_position.protected is True


@pytest.mark.asyncio
async def test_spot_nonflat_execution_reconciles_without_native_protection():
    from cryptotrader.execution.service import VenueExecutionService

    quote = VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("100"), Decimal("99.5"))
    session = _VenueSession("0", quote=quote)
    session.capabilities = SPOT_CAPABILITIES
    plan = replace(
        _venue_plan("0", "1", old_protection_ids=()),
        pair=SPOT_PAIR,
        quote=quote,
        execution_price=Decimal("100"),
        target_signed_notional=Decimal("100"),
        delta_signed_notional=Decimal("100"),
        market_type="spot",
        stop_loss=None,
        take_profit=None,
        capabilities=SPOT_CAPABILITIES,
    )

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "completed"
    assert result.protection is None
    assert result.final_position is not None
    assert result.final_position.protected is False
    assert result.final_position.position.signed_amount == Decimal("1")
    assert "replace_protection" not in session.calls
    assert result.trace == ("pre_read", "place_order", "reconcile")


@pytest.mark.asyncio
async def test_spot_active_protection_fails_precondition_before_quote_or_mutation():
    from cryptotrader.execution.service import VenueExecutionService

    quote = VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("100"), Decimal("99.5"))
    external_oco = ProtectionState(
        ("external-oco",),
        SPOT_PAIR,
        "long",
        Decimal("1"),
        Decimal("90"),
        Decimal("120"),
        True,
        False,
    )
    session = _VenueSession("1", protections=(external_oco,), quote=quote)
    session.capabilities = SPOT_CAPABILITIES
    plan = replace(
        _venue_plan("1", "2", old_protection_ids=()),
        pair=SPOT_PAIR,
        quote=quote,
        execution_price=Decimal("100"),
        market_type="spot",
        stop_loss=None,
        take_profit=None,
        capabilities=SPOT_CAPABILITIES,
    )

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "failed"
    assert result.error_operation == "precondition"
    assert result.requires_attention is True
    assert result.execution_quote is None
    assert result.trace == ("pre_read", "precondition")
    assert session.calls == ["list_open_state"]
    assert session.signed_amount == Decimal("1")
    assert session.protections == (external_oco,)


@pytest.mark.asyncio
async def test_spot_never_validates_or_installs_plan_protection_at_latest_quote():
    from cryptotrader.execution.service import VenueExecutionService

    latest = VenueQuote(SPOT_PAIR, Decimal("130"), Decimal("130"), Decimal("130"))
    capabilities = replace(SPOT_CAPABILITIES, native_protection=True)
    session = _VenueSession("0", quote=latest)
    session.capabilities = capabilities
    plan = replace(
        _venue_plan("0", "1", old_protection_ids=()),
        pair=SPOT_PAIR,
        quote=VenueQuote(SPOT_PAIR, Decimal("100"), Decimal("100"), Decimal("100")),
        market_type="spot",
        capabilities=capabilities,
    )

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "completed"
    assert result.protection is None
    assert "replace_protection" not in session.calls


@pytest.mark.asyncio
async def test_venue_service_flat_target_cancels_old_protection_while_flat():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "0"))

    assert result.status == "completed"
    assert result.trace == ("pre_read", "place_order", "cancel_old_protection", "reconcile")
    assert session.orders[0].reduce_only is True
    assert session.signed_amount == 0
    assert session.protections == ()


@pytest.mark.asyncio
async def test_venue_service_sign_flip_closes_then_opens_without_one_leg_flip():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "-2"))

    assert result.status == "completed"
    assert [(order.side, order.amount, order.reduce_only) for order in session.orders] == [
        ("sell", Decimal("1"), True),
        ("sell", Decimal("2"), False),
    ]
    assert result.trace == (
        "pre_read",
        "close_old_side",
        "reconcile_flat",
        "cancel_old_protection",
        "open_target_side",
        "replace_protection",
        "reconcile",
    )


@pytest.mark.asyncio
async def test_venue_service_open_failure_after_flip_close_stays_safely_flat():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    original_place = session.place_order
    calls = 0

    async def fail_second(intent):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise VenueOperationError("RAW_SECRET_OPEN")
        return await original_place(intent)

    session.place_order = fail_second
    result = await VenueExecutionService(session).execute(_venue_plan("1", "-2"))

    assert result.status == "failed"
    assert session.signed_amount == 0
    assert session.protections == ()
    assert result.requires_attention is False
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_flip_close_transport_error_after_fill_cancels_old_protection_while_flat():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    original_place = session.place_order

    async def fill_then_fail(intent):
        await original_place(intent)
        raise VenueOperationError("RAW_SECRET_CLOSE")

    session.place_order = fill_then_fail
    result = await VenueExecutionService(session).execute(_venue_plan("1", "-2"))

    assert result.status == "failed"
    assert result.requires_attention is False
    assert session.signed_amount == 0
    assert session.protections == ()
    assert result.trace == (
        "pre_read",
        "close_old_side",
        "reconcile_flat",
        "cancel_old_protection",
        "reconcile",
    )
    assert "RAW_SECRET" not in repr(result)


@pytest.mark.asyncio
async def test_partial_risk_reduction_never_reincreases_position():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),), partial_fill=Decimal("0.5"))
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert len(session.orders) == 1
    assert session.signed_amount == Decimal("1.5")
    assert result.compensation.attempted is False


@pytest.mark.asyncio
async def test_ambiguous_derivative_reduction_marks_unprotected_residual_for_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),))
    original_place = session.place_order

    async def fill_then_fail(intent):
        await original_place(intent)
        raise VenueOperationError("RAW_SECRET_REDUCTION")

    session.place_order = fill_then_fail
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert result.final_position is not None
    assert result.final_position.position.signed_amount == Decimal("1")
    assert result.final_position.protected is False
    assert result.requires_attention is True


@pytest.mark.asyncio
async def test_unreadable_state_after_failed_reduction_requires_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession(
        "2",
        protections=(_venue_protection("2"),),
        failures=("place_order", "list_open_state"),
    )
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert result.final_position is None
    assert result.requires_attention is True


@pytest.mark.asyncio
async def test_reduction_replace_transport_error_is_safe_only_with_exact_desired_protection_state():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),))
    original_replace = session.replace_protection

    async def install_then_fail(spec):
        await original_replace(spec)
        raise VenueOperationError("RAW_SECRET_REPLACE_RESPONSE")

    session.replace_protection = install_then_fail
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert result.error_operation == "replace_protection"
    assert result.requires_attention is False
    assert result.compensation.attempted is False
    assert len(session.orders) == 1
    assert session.signed_amount == Decimal("1")
    assert result.final_position is not None
    assert result.final_position.protections == session.protections
    assert result.final_position.protection_ids == ("new-1",)
    assert result.trace == ("pre_read", "place_order", "replace_protection", "reconcile")


@pytest.mark.asyncio
async def test_reduction_replace_transport_error_with_extra_id_in_one_group_requires_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),))
    original_replace = session.replace_protection

    async def install_ambiguous_group_then_fail(spec):
        installed = await original_replace(spec)
        session.protections = (replace(installed, protection_ids=("old", "new")),)
        raise VenueOperationError("RAW_SECRET_REPLACE_RESPONSE")

    session.replace_protection = install_ambiguous_group_then_fail
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert result.error_operation == "replace_protection"
    assert result.requires_attention is True
    assert result.compensation.attempted is False
    assert len(session.orders) == 1
    assert session.signed_amount == Decimal("1")
    assert result.final_position is not None
    assert result.final_position.protection_ids == ("old", "new")


@pytest.mark.asyncio
async def test_reduction_replace_transport_error_with_old_and_new_protection_requires_attention():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("2", protections=(_venue_protection("2"),))
    original_replace = session.replace_protection

    async def retain_old_install_new_then_fail(spec):
        old = session.protections
        await original_replace(spec)
        session.protections = (*old, *session.protections)
        raise VenueOperationError("RAW_SECRET_REPLACE_RESPONSE")

    session.replace_protection = retain_old_install_new_then_fail
    result = await VenueExecutionService(session).execute(_venue_plan("2", "1"))

    assert result.status == "failed"
    assert result.error_operation == "replace_protection"
    assert result.requires_attention is True
    assert result.compensation.attempted is False
    assert len(session.orders) == 1
    assert session.signed_amount == Decimal("1")
    assert result.final_position is not None
    assert len(result.final_position.protections) == 2


@pytest.mark.asyncio
async def test_partial_spot_reduction_does_not_require_native_protection_attention():
    from cryptotrader.execution.service import VenueExecutionService

    quote = VenueQuote(SPOT_PAIR, Decimal("100"), Decimal("100"), Decimal("100"))
    session = _VenueSession("2", partial_fill=Decimal("0.5"), quote=quote)
    session.capabilities = SPOT_CAPABILITIES
    plan = replace(
        _venue_plan("2", "1", old_protection_ids=()),
        pair=SPOT_PAIR,
        quote=quote,
        market_type="spot",
        stop_loss=None,
        take_profit=None,
        capabilities=SPOT_CAPABILITIES,
    )

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "failed"
    assert result.final_position is not None
    assert result.final_position.position.signed_amount == Decimal("1.5")
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_venue_quote_failure_has_safe_category_while_programmer_error_propagates():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("1", protections=(_venue_protection("1"),), failures=("fetch_quote",))
    result = await VenueExecutionService(session).execute(_venue_plan("1", "2"))
    assert result.status == "failed"
    assert result.error_operation == "fetch_quote"
    assert "RAW_SECRET" not in repr(result)

    async def invalid_quote(pair):
        raise ValueError("session contract violated")

    session = _VenueSession("1", protections=(_venue_protection("1"),))
    session.fetch_quote = invalid_quote
    with pytest.raises(ValueError, match="contract violated"):
        await VenueExecutionService(session).execute(_venue_plan("1", "2"))


@pytest.mark.asyncio
async def test_venue_service_executes_the_same_contract_against_real_paper_session():
    from cryptotrader.execution.service import VenueExecutionService
    from cryptotrader.venues.paper import PaperVenueAdapter

    session = await PaperVenueAdapter().connect(connection("paper-a"), None)
    await session.set_quote(VENUE_PAIR, Decimal("100"))
    plan = replace(_venue_plan("0", "1", old_protection_ids=()), capabilities=session.capabilities)

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "completed"
    assert result.final_position is not None
    assert result.final_position.position.signed_amount == Decimal("1")
    assert result.final_position.protected is True
    assert result.trace == ("pre_read", "place_order", "replace_protection", "reconcile")


@pytest.mark.asyncio
async def test_latest_quote_invalidating_protection_fails_closed_before_mutation():
    from cryptotrader.execution.service import VenueExecutionService

    latest = VenueQuote(VENUE_PAIR, Decimal("130"), Decimal("130"), Decimal("130"))
    session = _VenueSession("0", quote=latest)

    with pytest.raises(ValueError, match="geometry"):
        await VenueExecutionService(session).execute(_venue_plan("0", "1", old_protection_ids=()))
    assert session.orders == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("current", "planned_target", "pair", "quote", "expected_trace"),
    [
        (
            "1",
            "1.005",
            VENUE_PAIR,
            VenueQuote(VENUE_PAIR, Decimal("99"), Decimal("101"), Decimal("100")),
            ("pre_read", "replace_protection", "reconcile"),
        ),
        (
            "-1",
            "-1.005",
            VENUE_PAIR,
            VenueQuote(VENUE_PAIR, Decimal("99"), Decimal("101"), Decimal("100")),
            ("pre_read", "replace_protection", "reconcile"),
        ),
        (
            "0.123",
            "0.123005",
            VENUE_PAIR,
            VenueQuote(VENUE_PAIR, Decimal("99.99"), Decimal("100.01"), Decimal("100")),
            ("pre_read", "replace_protection", "reconcile"),
        ),
        (
            "1",
            "1.005",
            SPOT_PAIR,
            VenueQuote(SPOT_PAIR, Decimal("99"), Decimal("101"), Decimal("100")),
            ("pre_read", "reconcile"),
        ),
    ],
)
async def test_target_notional_inside_current_spread_band_is_audited_no_trade(
    current, planned_target, pair, quote, expected_trace
):
    from cryptotrader.execution.service import VenueExecutionService

    is_spot = pair.market_type == "spot"
    protections = (
        ()
        if is_spot
        else (_venue_protection(str(abs(Decimal(current))), side="long" if Decimal(current) > 0 else "short"),)
    )
    session = _VenueSession(current, protections=protections, quote=quote)
    plan = _venue_plan(current, planned_target, old_protection_ids=tuple(p.protection_ids[0] for p in protections))
    if is_spot:
        session.capabilities = SPOT_CAPABILITIES
        plan = replace(
            plan,
            pair=pair,
            quote=VenueQuote(pair, Decimal("100"), Decimal("100"), Decimal("100")),
            market_type="spot",
            stop_loss=None,
            take_profit=None,
            capabilities=SPOT_CAPABILITIES,
        )

    result = await VenueExecutionService(session).execute(plan)

    assert result.status == "completed"
    assert result.orders == ()
    assert result.target_signed_notional == Decimal(planned_target) * Decimal("100")
    assert result.target_signed_amount == Decimal(current)
    assert result.final_position is not None
    assert result.final_position.position.signed_amount == Decimal(current)
    assert result.trace == expected_trace


@pytest.mark.asyncio
async def test_runtime_precision_that_cannot_reach_target_band_fails_before_mutation():
    from cryptotrader.execution.service import VenueExecutionService

    quote = VenueQuote(VENUE_PAIR, Decimal("100"), Decimal("101"), Decimal("100"))
    session = _VenueSession("1", protections=(_venue_protection("1"),), quote=quote)

    async def coarse_precision(pair, amount):
        return Decimal("0.005")

    session.normalize_amount = coarse_precision
    result = await VenueExecutionService(session).execute(_venue_plan("1", "1.02"))

    assert result.status == "failed"
    assert result.error_operation == "incomplete_fill"
    assert result.execution_quote == quote
    assert session.orders == []
    assert session.signed_amount == Decimal("1")
    assert result.requires_attention is False


@pytest.mark.asyncio
async def test_live_execution_gate_refuses_before_any_venue_read_or_write():
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("0")
    session.connection = connection("live-a", environment="live")

    result = await VenueExecutionService(session).execute(_venue_plan("0", "1", old_protection_ids=()))

    assert result.error_operation == "execution_gate"
    assert result.trace == ("execution_gate",)
    assert session.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("environment", ["paper", "demo", "testnet"])
async def test_non_live_environment_is_not_blocked_by_live_execution_gate(environment):
    from cryptotrader.execution.service import VenueExecutionService

    session = _VenueSession("0")
    session.connection = connection(f"{environment}-a", environment=environment)

    result = await VenueExecutionService(session).execute(_venue_plan("0", "1", old_protection_ids=()))

    assert result.status == "completed"
    assert session.calls[0] == "list_open_state"
