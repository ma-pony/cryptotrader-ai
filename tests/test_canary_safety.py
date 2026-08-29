"""Safety contracts for the manually-operated integration canaries."""

from __future__ import annotations

import asyncio
import importlib.util
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from cryptotrader.pair import Pair
from cryptotrader.venues.models import (
    ConnectionPosition,
    NormalizedOrder,
    OpenVenueState,
    ProtectionSpec,
    ProtectionState,
    VenueCapabilities,
    VenueQuote,
)


def _script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_venue_canary_refuses_live_environment():
    venue_canary = _script("venue_canary.py")

    with pytest.raises(venue_canary.CanarySafetyError, match="live connections are read-only in canary"):
        venue_canary.require_simulated_environment("live")


def test_venue_canary_refuses_a_normal_simulated_connection_before_connecting():
    venue_canary = _script("venue_canary.py")

    with pytest.raises(venue_canary.CanarySafetyError, match="canary_only"):
        venue_canary.require_canary_only(type("Connection", (), {"canary_only": False})())


def test_signal_canary_defaults_to_no_execution():
    signal_canary = _script("signal_canary.py")

    options = signal_canary.parse_signal_canary_args(["--pair", "BTC/USDT"])

    assert options.execute is False


def test_cli_does_not_accept_secret_arguments():
    venue_canary = _script("venue_canary.py")

    with pytest.raises(SystemExit):
        venue_canary.parse_venue_canary_args(["--connection", "paper", "--pair", "BTC/USDT", "--secret", "x"])


def test_canary_orders_have_an_exchange_visible_client_identifier_contract():
    from cryptotrader.venues.models import OrderIntent

    intent = OrderIntent(Pair.parse("BTC/USDT"), "buy", Decimal("1"), "market", None, False, "CTABC123O")

    assert intent.client_order_id == "CTABC123O"


@dataclass
class _Session:
    pair: Pair
    fail_open: bool = False
    signed_amount: Decimal = Decimal("0")
    cleaned: bool = False
    order_calls: int = 0
    order_amounts: list[Decimal] = field(default_factory=list)
    protected: bool = False

    connection_id: str = "paper"
    capabilities: VenueCapabilities = field(
        default_factory=lambda: VenueCapabilities(frozenset({"swap"}), True, False, True, frozenset({"market"}))
    )

    async def fetch_portfolio(self, _pair):
        return object()

    async def fetch_quote(self, _pair):
        return VenueQuote(self.pair, Decimal("100"), Decimal("100"), Decimal("100"))

    async def normalize_amount(self, _pair, amount):
        return amount

    async def minimum_amount(self, _pair, _price, _minimum_quote_notional):
        return Decimal("0.1")

    async def place_order(self, intent):
        self.order_calls += 1
        self.order_amounts.append(intent.amount)
        if self.fail_open and not intent.reduce_only:
            raise RuntimeError("open failed")
        self.signed_amount += intent.amount if intent.side == "buy" else -intent.amount
        return NormalizedOrder(
            f"order-{self.order_calls}",
            intent.pair,
            intent.side,
            intent.order_type,
            intent.amount,
            intent.amount,
            Decimal("100"),
            "filled",
            intent.reduce_only,
            intent.client_order_id,
        )

    async def cancel_order(self, _order_id, _pair):
        self.cleaned = True

    async def replace_protection(self, _spec):
        self.protected = True
        return type("Protection", (), {"protection_ids": ("canary",)})()

    async def normalize_protection(self, spec):
        return spec

    async def cancel_protection(self, _ids):
        self.cleaned = True
        self.protected = False

    async def list_open_state(self, _pair):
        protections = ()
        if self.protected:
            from cryptotrader.venues.models import ProtectionState

            protections = (
                ProtectionState(
                    ("canary",), self.pair, "long", self.signed_amount, Decimal("90"), Decimal("110"), True, False
                ),
            )
        return OpenVenueState(ConnectionPosition(self.pair, self.signed_amount, Decimal("0"), None), (), protections)

    async def close(self):
        self.cleaned = True


@pytest.mark.asyncio
async def test_failed_canary_still_runs_cleanup():
    venue_canary = _script("venue_canary.py")
    session = _Session(Pair.parse("BTC/USDT:USDT"), fail_open=True)

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert result["status"] == "failed"
    assert session.cleaned is True


@pytest.mark.asyncio
async def test_canary_uses_small_quote_notional_instead_of_a_fixed_base_amount():
    venue_canary = _script("venue_canary.py")
    session = _Session(Pair.parse("BTC/USDT:USDT"))

    await venue_canary.run_simulated_canary(session, session.pair)

    assert session.order_calls == 2
    assert session.order_amounts[0] == Decimal("0.1")
    assert session.signed_amount == Decimal("0")


@pytest.mark.asyncio
async def test_residual_state_requires_attention_and_audit_protocol_is_separate():
    venue_canary = _script("venue_canary.py")
    session = _Session(Pair.parse("BTC/USDT:USDT"))
    session.signed_amount = Decimal("1")

    result = await venue_canary.inspect_residual(session, session.pair)

    assert result["requires_attention"] is True
    assert result["residual"]["position_nonzero"] is True
    assert venue_canary.audit_in_subprocess.__name__ == "audit_in_subprocess"


@pytest.mark.asyncio
async def test_nonzero_initial_state_is_reported_without_touching_user_position():
    venue_canary = _script("venue_canary.py")
    session = _Session(Pair.parse("BTC/USDT:USDT"), signed_amount=Decimal("1"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert result["requires_attention"] is True
    assert session.order_calls == 0


def test_canary_output_redacts_sensitive_values_and_signal_wiring_is_real():
    venue_canary = _script("venue_canary.py")
    signal_canary = _script("signal_canary.py")

    payload = {"api_key": "visible-secret", "ok": True}  # pragma: allowlist secret
    assert "visible-secret" not in venue_canary.safe_json(payload)  # pragma: allowlist secret
    assert frozenset({"kronos", "llm_committee"}) == signal_canary.REQUIRED_COMPONENT_IDS
    assert "mock" not in Path(signal_canary.__file__).read_text().lower()


def test_signal_canary_forces_each_real_model_call_to_fail_without_fallback(monkeypatch):
    signal_canary = _script("signal_canary.py")
    calls = []

    def build_factory(_config, *, api_key, response_observer=None):
        assert api_key == "gateway-token"  # pragma: allowlist secret
        assert response_observer is None

        def invoke(**kwargs):
            calls.append(kwargs)
            return object()

        return invoke

    monkeypatch.setattr("cryptotrader.agents.base.create_runtime_llm_factory", build_factory)

    factory = signal_canary.strict_llm_factory(object(), "gateway-token")  # pragma: allowlist secret
    factory(model="model-a")

    assert calls == [{"model": "model-a", "with_fallback": False}]


@pytest.mark.asyncio
async def test_subprocess_audit_rejects_nonzero_exit_noise_and_attention(monkeypatch):
    venue_canary = _script("venue_canary.py")

    class Completed:
        returncode = 1
        stdout = (
            '{"status":"completed","requires_attention":false,'
            '"residual":{"position_nonzero":false,"open_orders":false,"protections":false}}'
        )

    monkeypatch.setattr(venue_canary.subprocess, "run", lambda *_args, **_kwargs: Completed())

    result = await venue_canary.audit_in_subprocess("paper", "BTC/USDT")

    assert result["audit_status"] == "failed"
    assert result["requires_attention"] is True


def test_completed_audit_never_erases_main_flow_failure_attention():
    venue_canary = _script("venue_canary.py")
    result = venue_canary.merge_audit_result(
        {"status": "failed", "requires_attention": True, "error_type": "TimeoutError"},
        {"audit_status": "completed", "audit": {"status": "completed", "requires_attention": False}},
    )

    assert result["status"] == "failed"
    assert result["requires_attention"] is True


@dataclass
class _AmbiguousCreateSession(_Session):
    """Exchange accepted the order before the caller saw a timeout."""

    pending: list[NormalizedOrder] = field(default_factory=list)
    cancelled_ids: list[str] = field(default_factory=list)
    partial_fill: Decimal = Decimal("0")
    fail_cancel: bool = False
    external_active: bool = False

    async def place_order(self, intent):
        if not intent.reduce_only:
            self.external_active = True
            remote = NormalizedOrder(
                "canary-open-remote",
                intent.pair,
                intent.side,
                intent.order_type,
                intent.amount,
                self.partial_fill,
                Decimal("100"),
                "open",
                intent.reduce_only,
                intent.client_order_id,
            )
            self.pending.append(remote)
            self.signed_amount += self.partial_fill
            raise TimeoutError("remote order accepted but response timed out")
        return await super().place_order(intent)

    async def cancel_order(self, order_id, pair):
        if self.fail_cancel:
            raise RuntimeError("cancel rejected")
        self.cancelled_ids.append(order_id)
        self.pending = [item for item in self.pending if item.id != order_id]

    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        for item in self.pending:
            if item.id == order_id or item.client_order_id == client_order_id:
                return item
        return None

    async def list_open_state(self, _pair):
        external = ()
        if self.external_active:
            external = (
                NormalizedOrder(
                    "external-order",
                    self.pair,
                    "buy",
                    "limit",
                    Decimal("1"),
                    Decimal("0"),
                    None,
                    "open",
                    False,
                    "externalrun",
                ),
            )
        protections = ()
        if self.protected:
            from cryptotrader.venues.models import ProtectionState

            protections = (
                ProtectionState(
                    ("external-protection",),
                    self.pair,
                    "long",
                    Decimal("1"),
                    Decimal("90"),
                    Decimal("110"),
                    True,
                    False,
                ),
            )
        return OpenVenueState(
            ConnectionPosition(self.pair, self.signed_amount, Decimal("0"), None),
            (*self.pending, *external),
            protections,
        )


@pytest.mark.asyncio
async def test_timeout_after_remote_create_cancels_only_tagged_owned_order():
    venue_canary = _script("venue_canary.py")
    session = _AmbiguousCreateSession(Pair.parse("BTC/USDT:USDT"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_ids == ["canary-open-remote"]
    assert "external-order" not in session.cancelled_ids
    assert result["requires_attention"] is True
    assert result["status"] == "failed"


@pytest.mark.asyncio
async def test_timeout_partial_fill_cancels_pending_then_closes_exact_owned_fill():
    venue_canary = _script("venue_canary.py")
    session = _AmbiguousCreateSession(Pair.parse("BTC/USDT:USDT"), partial_fill=Decimal("0.04"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_ids == ["canary-open-remote"]
    assert session.order_amounts == [Decimal("0.04")]
    assert session.signed_amount == Decimal("0")
    assert result["requires_attention"] is True


@pytest.mark.asyncio
async def test_owned_order_cancel_failure_requires_attention_without_external_cancellation():
    venue_canary = _script("venue_canary.py")
    session = _AmbiguousCreateSession(Pair.parse("BTC/USDT:USDT"), fail_cancel=True)

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_ids == []
    assert result["requires_attention"] is True
    assert result["status"] == "failed"


@dataclass
class _AsyncAcknowledgementSession(_Session):
    queried: int = 0

    async def place_order(self, intent):
        if intent.reduce_only:
            return await super().place_order(intent)
        return NormalizedOrder(
            "ack-order",
            intent.pair,
            intent.side,
            intent.order_type,
            intent.amount,
            Decimal("0"),
            None,
            "open",
            False,
            intent.client_order_id,
        )

    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        self.queried += 1
        assert order_id == "ack-order" or client_order_id
        self.signed_amount = Decimal("0.1")
        return NormalizedOrder(
            "ack-order",
            self.pair,
            "buy",
            "market",
            Decimal("0.1"),
            Decimal("0.1"),
            Decimal("100"),
            "filled",
            False,
            client_order_id or "",
        )


@pytest.mark.asyncio
async def test_async_order_ack_is_reconciled_before_using_owned_fill():
    venue_canary = _script("venue_canary.py")
    session = _AsyncAcknowledgementSession(Pair.parse("BTC/USDT:USDT"))

    await venue_canary.run_simulated_canary(session, session.pair)

    assert session.queried >= 1
    assert session.order_amounts == [Decimal("0.1")]


@dataclass
class _ClientLookupAcknowledgementSession(_AsyncAcknowledgementSession):
    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        if order_id:
            raise RuntimeError("order id not indexed yet")
        return await super().find_order(_pair, order_id=order_id, client_order_id=client_order_id)


@pytest.mark.asyncio
async def test_order_ack_falls_back_to_client_id_when_exchange_has_not_indexed_order_id():
    venue_canary = _script("venue_canary.py")
    session = _ClientLookupAcknowledgementSession(Pair.parse("BTC/USDT:USDT"))

    await venue_canary.run_simulated_canary(session, session.pair)

    assert session.queried >= 1


@dataclass
class _FullyFilledTimeoutSession(_AmbiguousCreateSession):
    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        if client_order_id and client_order_id.endswith("O"):
            self.signed_amount = Decimal("0.1")
            return NormalizedOrder(
                "remote-filled",
                self.pair,
                "buy",
                "market",
                Decimal("0.1"),
                Decimal("0.1"),
                Decimal("100"),
                "filled",
                False,
                client_order_id,
            )
        return None


@pytest.mark.asyncio
async def test_timeout_after_fully_filled_remote_order_uses_client_id_lookup_for_exact_cleanup():
    venue_canary = _script("venue_canary.py")
    session = _FullyFilledTimeoutSession(Pair.parse("BTC/USDT:USDT"))

    await venue_canary.run_simulated_canary(session, session.pair)

    assert session.order_amounts == [Decimal("0.1")]


@dataclass
class _PartialCloseSession(_Session):
    pending: list[NormalizedOrder] = field(default_factory=list)
    cancelled_ids: list[str] = field(default_factory=list)

    async def place_order(self, intent):
        if not intent.reduce_only or intent.client_order_id.endswith("X"):
            return await super().place_order(intent)
        partial = intent.amount / Decimal("2")
        self.order_calls += 1
        self.order_amounts.append(intent.amount)
        self.signed_amount -= partial
        order = NormalizedOrder(
            "close-pending",
            intent.pair,
            "sell",
            "market",
            intent.amount,
            partial,
            Decimal("100"),
            "partial",
            True,
            intent.client_order_id,
        )
        self.pending.append(order)
        return order

    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        return next(
            (item for item in self.pending if item.id == order_id or item.client_order_id == client_order_id), None
        )

    async def cancel_order(self, order_id, _pair):
        self.cancelled_ids.append(order_id)
        self.pending = [item for item in self.pending if item.id != order_id]


@pytest.mark.asyncio
async def test_partial_close_cancels_exact_remainder_then_closes_only_remaining_owned_exposure(monkeypatch):
    venue_canary = _script("venue_canary.py")
    monkeypatch.setattr(venue_canary, "_ORDER_POLL_SECONDS", 0)
    session = _PartialCloseSession(Pair.parse("BTC/USDT:USDT"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_ids == ["close-pending"]
    assert session.order_amounts == [Decimal("0.1"), Decimal("0.1"), Decimal("0.05")]
    assert session.signed_amount == Decimal("0")
    assert result["status"] == "completed"


@dataclass
class _LostProtectionSession(_Session):
    protection_matches: int = 1
    cancelled_protections: list[tuple[str, ...]] = field(default_factory=list)

    async def replace_protection(self, spec):
        self.protected = True
        self._spec = spec
        raise TimeoutError("accepted before response")

    async def normalize_protection(self, spec):
        return ProtectionSpec(spec.pair, spec.position_side, spec.amount, Decimal("98.1"), Decimal("101.9"))

    async def cancel_protection(self, ids):
        self.cancelled_protections.append(tuple(ids))
        self.protected = False

    async def list_open_state(self, _pair):
        protections = (
            tuple(
                ProtectionState(
                    (f"lost-{index}",),
                    self._spec.pair,
                    self._spec.position_side,
                    self._spec.amount,
                    self._spec.stop_loss,
                    self._spec.take_profit,
                    True,
                    False,
                )
                for index in range(self.protection_matches)
            )
            if self.protected
            else ()
        )
        return OpenVenueState(ConnectionPosition(self.pair, self.signed_amount, Decimal("0"), None), (), protections)


@pytest.mark.asyncio
async def test_timeout_after_remote_protection_acceptance_recovers_exactly_one_owned_protection():
    venue_canary = _script("venue_canary.py")
    session = _LostProtectionSession(Pair.parse("BTC/USDT:USDT"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_protections == [("lost-0",)]
    assert result["status"] == "failed"


@dataclass
class _CancelledProtectionSession(_LostProtectionSession):
    async def replace_protection(self, spec):
        self.protected = True
        self._spec = spec
        raise asyncio.CancelledError


@pytest.mark.asyncio
async def test_cancelled_after_remote_protection_acceptance_still_runs_owned_finalizer():
    venue_canary = _script("venue_canary.py")
    session = _CancelledProtectionSession(Pair.parse("BTC/USDT:USDT"))

    with pytest.raises(asyncio.CancelledError):
        await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_protections == [("lost-0",)]
    assert session.cleaned is True


@pytest.mark.asyncio
@pytest.mark.parametrize("matches", [0, 2])
async def test_lost_protection_without_one_exact_match_requires_attention(matches):
    venue_canary = _script("venue_canary.py")
    session = _LostProtectionSession(Pair.parse("BTC/USDT:USDT"), protection_matches=matches)

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.cancelled_protections == []
    assert result["requires_attention"] is True
    assert session.signed_amount == Decimal("0")


@dataclass
class _PendingOpenCancelFailureSession(_Session):
    pending: NormalizedOrder | None = None

    async def place_order(self, intent):
        if intent.reduce_only:
            return await super().place_order(intent)
        self.pending = NormalizedOrder(
            "pending-open",
            intent.pair,
            "buy",
            "market",
            intent.amount,
            Decimal("0"),
            None,
            "open",
            False,
            intent.client_order_id,
        )
        return self.pending

    async def find_order(self, _pair, *, order_id=None, client_order_id=None):
        return self.pending if order_id == "pending-open" or client_order_id == self.pending.client_order_id else None

    async def cancel_order(self, _order_id, _pair):
        raise RuntimeError("cancel failed")


@pytest.mark.asyncio
async def test_unconfirmed_open_cancellation_failure_never_installs_protection(monkeypatch):
    venue_canary = _script("venue_canary.py")
    monkeypatch.setattr(venue_canary, "_ORDER_POLL_SECONDS", 0)
    session = _PendingOpenCancelFailureSession(Pair.parse("BTC/USDT:USDT"))

    result = await venue_canary.run_simulated_canary(session, session.pair)

    assert session.protected is False
    assert result["status"] == "failed"


def test_fill_ledger_never_replaces_a_cumulative_fill_with_stale_smaller_readback():
    venue_canary = _script("venue_canary.py")
    pair = Pair.parse("BTC/USDT:USDT")
    ledger = {}
    filled = NormalizedOrder(
        "open", pair, "buy", "market", Decimal("1"), Decimal("1"), Decimal("100"), "filled", False, "CTLEDGERO"
    )
    stale = NormalizedOrder(
        "open", pair, "buy", "market", Decimal("1"), Decimal("0.5"), Decimal("100"), "partial", False, "CTLEDGERO"
    )

    venue_canary._record_order(ledger, filled)
    venue_canary._record_order(ledger, stale)

    assert venue_canary._owned_exposure(ledger) == Decimal("1")


@pytest.mark.asyncio
async def test_ambiguous_protection_still_cancels_owned_open_order_and_closes_ledger_exposure():
    venue_canary = _script("venue_canary.py")
    pair = Pair.parse("BTC/USDT:USDT")
    session = _LostProtectionSession(pair, protection_matches=2, signed_amount=Decimal("0.1"))
    session._spec = ProtectionSpec(pair, "long", Decimal("0.1"), Decimal("98.1"), Decimal("101.9"))
    session.protected = True
    pending = NormalizedOrder(
        "owned-open", pair, "buy", "market", Decimal("0.1"), Decimal("0.1"), Decimal("100"), "open", False, "CTCOMBOO"
    )
    cancelled: list[str] = []

    async def state(_pair):
        protections = tuple(
            ProtectionState(
                (f"unknown-{index}",), pair, "long", Decimal("0.1"), Decimal("98.1"), Decimal("101.9"), True, False
            )
            for index in range(2)
        )
        orders = () if cancelled else (pending,)
        return OpenVenueState(ConnectionPosition(pair, session.signed_amount, Decimal("0"), None), orders, protections)

    async def cancel(order_id, _pair):
        cancelled.append(order_id)

    session.list_open_state = state
    session.cancel_order = cancel
    ledger = {"CTCOMBOO": pending}
    result = await venue_canary._cleanup_owned(
        session,
        pair,
        allowed_client_order_ids=frozenset({"CTCOMBOO", "CTCOMBOX"}),
        fill_ledger=ledger,
        cleanup_client_order_id="CTCOMBOX",
        protection_ids=(),
        expected_protection=session._spec,
    )

    assert cancelled == ["owned-open"]
    assert session.signed_amount == Decimal("0")
    assert session.cancelled_protections == []
    assert result["requires_attention"] is True
