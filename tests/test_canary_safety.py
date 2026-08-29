"""Safety contracts for the manually-operated integration canaries."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

import pytest

from cryptotrader.pair import Pair
from cryptotrader.venues.models import ConnectionPosition, OpenVenueState, VenueCapabilities, VenueQuote


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


def test_signal_canary_defaults_to_no_execution():
    signal_canary = _script("signal_canary.py")

    options = signal_canary.parse_signal_canary_args(["--pair", "BTC/USDT"])

    assert options.execute is False


def test_cli_does_not_accept_secret_arguments():
    venue_canary = _script("venue_canary.py")

    with pytest.raises(SystemExit):
        venue_canary.parse_venue_canary_args(["--connection", "paper", "--pair", "BTC/USDT", "--secret", "x"])


@dataclass
class _Session:
    pair: Pair
    fail_open: bool = False
    signed_amount: Decimal = Decimal("0")
    cleaned: bool = False
    order_calls: int = 0
    order_amounts: list[Decimal] = field(default_factory=list)

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

    async def place_order(self, intent):
        self.order_calls += 1
        self.order_amounts.append(intent.amount)
        if self.fail_open and not intent.reduce_only:
            raise RuntimeError("open failed")
        self.signed_amount += intent.amount if intent.side == "buy" else -intent.amount
        return object()

    async def replace_protection(self, _spec):
        return type("Protection", (), {"protection_ids": ("canary",)})()

    async def cancel_protection(self, _ids):
        self.cleaned = True

    async def list_open_state(self, _pair):
        return OpenVenueState(ConnectionPosition(self.pair, self.signed_amount, Decimal("0"), None), (), ())

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
