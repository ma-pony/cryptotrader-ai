"""Shared executable contract for both CCXT venue implementations."""

import importlib
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from cryptotrader.pair import Pair
from cryptotrader.portfolio.models import ConnectionPortfolioSnapshot
from cryptotrader.runtime_config.secrets import CredentialPayload
from cryptotrader.venues.ccxt_base import VenueOperationError
from cryptotrader.venues.models import OrderIntent
from tests.contracts.venue_adapter import assert_ccxt_session_contract, assert_venue_contract
from tests.factories.runtime_config import connection
from tests.fakes.ccxt_client import FakeCcxtFactory


@pytest.mark.asyncio
async def test_ccxt_connection_check_reads_an_account_without_trade_side_effects():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]
    client.fetch_balance = AsyncMock(return_value={"total": {"USDT": "100"}})

    await session.check_connection()

    client.fetch_balance.assert_awaited_once()
    assert not any(
        name in {"create_order", "cancel_order", "set_leverage", "set_margin_mode"} for name, _ in client.calls
    )


@pytest.mark.asyncio
async def test_ccxt_connection_check_hides_failed_account_read_payload():
    from cryptotrader.venues.okx import OkxVenueAdapter

    marker = "private-key-marker"
    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]
    client.fetch_balance = AsyncMock(side_effect=RuntimeError(marker))

    with pytest.raises(VenueOperationError) as caught:
        await session.check_connection()

    assert str(caught.value) == "okx-demo: check account failed"
    assert caught.value.code == "account_unavailable"
    assert marker not in str(caught.value)
    assert caught.value.__cause__ is None
    client.fetch_balance.assert_awaited_once()


@pytest.mark.asyncio
async def test_ccxt_connection_check_classifies_authentication_failures_without_payloads():
    from ccxt.base.errors import AuthenticationError

    from cryptotrader.venues.okx import OkxVenueAdapter

    marker = "provider-authentication-payload"
    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]
    client.fetch_balance = AsyncMock(side_effect=AuthenticationError(marker))

    with pytest.raises(VenueOperationError) as caught:
        await session.check_connection()

    assert caught.value.code == "authentication_failed"
    assert marker not in str(caught.value)
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environments"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", ("demo", "live")),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", ("testnet", "demo", "live")),
    ],
)
def test_ccxt_adapters_implement_the_task_4_adapter_contract(adapter_module, adapter_name, adapter_id, environments):
    import importlib

    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    assert_venue_contract(lambda: adapter_class(client_factory=FakeCcxtFactory(adapter_id)), environments)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet"),
    ],
)
async def test_ccxt_adapters_share_portfolio_order_protection_and_close_contract(
    adapter_module, adapter_name, adapter_id, environment
):
    import importlib

    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    fake_factory = FakeCcxtFactory(adapter_id)
    await assert_ccxt_session_contract(lambda: adapter_class(client_factory=fake_factory), environment, fake_factory)


@pytest.mark.asyncio
async def test_ccxt_error_boundary_does_not_expose_raw_exchange_message_or_cause():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    adapter = OkxVenueAdapter(client_factory=fake_factory)
    session = await adapter.connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    async def fail_with_raw_response(*_args):
        raise RuntimeError("raw-response-with-secret")

    fake_factory.clients[-1].fetch_ticker = fail_with_raw_response
    with pytest.raises(VenueOperationError) as caught:
        await session.fetch_quote(Pair.parse("BTC/USDT:USDT"))

    assert str(caught.value) == "okx-demo: fetch quote failed"
    assert caught.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment", "amounts"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo", ("2", "1")),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet", ("0.02", "0.01")),
    ],
)
async def test_ccxt_rejects_simultaneous_nonzero_hedge_legs_before_flat_execution(
    adapter_module, adapter_name, adapter_id, environment, amounts
):
    from dataclasses import replace

    from cryptotrader.execution.service import VenueExecutionService
    from tests.test_execution_service import _venue_plan

    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection("dual", environment, adapter_id=adapter_id, credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]
    pair = Pair.parse("BTC/USDT:USDT")

    async def dual_positions(*_args):
        return [
            {
                "symbol": pair.to_ccxt(),
                "contracts": amounts[0],
                "side": "long",
                "entryPrice": "50000",
                "notional": "1000",
                "info": {"positionIdx": 1, "stopLoss": "0", "takeProfit": "0"},
            },
            {
                "symbol": pair.to_ccxt(),
                "contracts": amounts[1],
                "side": "short",
                "entryPrice": "51000",
                "notional": "500",
                "info": {"positionIdx": 2, "stopLoss": "0", "takeProfit": "0"},
            },
        ]

    client.fetch_positions = dual_positions
    plan = replace(
        _venue_plan("1", "0", old_protection_ids=()),
        connection_id="dual",
        capabilities=session.capabilities,
    )

    result = await VenueExecutionService(session, connection=session.connection).execute(plan)

    assert result.status == "failed"
    assert result.error_operation == "pre_read"
    assert not any(
        name in {"create_order", "private_post_trade_cancel_algos", "private_post_v5_position_trading_stop"}
        for name, _ in client.calls
    )
    assert "secret" not in repr(result).lower()
    assert "raw" not in repr(result).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet"),
    ],
)
async def test_ccxt_keeps_one_nonzero_hedge_leg_supported(adapter_module, adapter_name, adapter_id, environment):
    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection("one-leg", environment, adapter_id=adapter_id, credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    position = await session.fetch_position(Pair.parse("BTC/USDT:USDT"))

    assert position.signed_amount == Decimal("0.02")


@pytest.mark.asyncio
async def test_ccxt_session_has_no_public_client_or_plaintext_credential_surface():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "visible-key",  # pragma: allowlist secret
                "secret": "visible-secret",  # pragma: allowlist secret
                "passphrase": "visible-passphrase",  # pragma: allowlist secret
            }
        ),
    )

    assert not hasattr(session, "client")
    assert "visible-key" not in repr(session)
    assert "visible-secret" not in repr(session)
    assert "visible-passphrase" not in repr(session)


@pytest.mark.asyncio
async def test_precision_and_post_call_shape_failures_are_credential_safe_operation_errors():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "visible-key",  # pragma: allowlist secret
                "secret": "visible-secret",  # pragma: allowlist secret
                "passphrase": "visible-passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]

    def precision_failure(*_args):
        raise RuntimeError("visible-secret raw precision payload")  # pragma: allowlist secret

    client.amount_to_precision = precision_failure
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.1"), "market", None, False)
    with pytest.raises(VenueOperationError, match="normalize amount") as caught:
        await session.place_order(intent)
    assert "visible-secret" not in str(caught.value)
    assert caught.value.__cause__ is None

    client.amount_to_precision = lambda *_args: "0.1"

    async def invalid_quote(*_args):
        return None

    client.fetch_ticker = invalid_quote
    with pytest.raises(VenueOperationError, match="invalid quote response"):
        await session.fetch_quote(Pair.parse("BTC/USDT:USDT"))


@pytest.mark.asyncio
async def test_failed_close_can_be_retried_and_successful_close_is_idempotent():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]
    client.close_failures = 1

    with pytest.raises(VenueOperationError, match="close session failed"):
        await session.close()
    await session.close()
    await session.close()

    assert client.close_calls == 2


@pytest.mark.asyncio
async def test_malformed_open_orders_top_level_shape_is_a_safe_operation_error():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("okx-demo", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    async def malformed_open_orders(*_args):
        return 7

    fake_factory.clients[-1].fetch_open_orders = malformed_open_orders

    with pytest.raises(VenueOperationError, match="invalid open orders response") as caught:
        await session.list_open_state(Pair.parse("BTC/USDT:USDT"))

    assert caught.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet"),
    ],
)
async def test_inverse_contract_is_rejected_before_precision_or_configuration(
    adapter_module, adapter_name, adapter_id, environment
):
    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection("inverse", environment, adapter_id=adapter_id, credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    with pytest.raises(VenueOperationError, match="inverse contracts are unsupported"):
        await session.place_order(OrderIntent(Pair.parse("BTC/USD:BTC"), "buy", Decimal("0.01"), "market", None, False))

    call_names = [name for name, _ in fake_factory.clients[-1].calls]
    assert "set_margin_mode" not in call_names
    assert "set_leverage" not in call_names
    assert "create_order" not in call_names


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment", "expected"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo", Decimal("0.12")),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet", Decimal("0.123")),
    ],
)
async def test_normalize_amount_round_trips_platform_units_back_to_safe_base_amount(
    adapter_module, adapter_name, adapter_id, environment, expected
):
    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection("normalize", environment, adapter_id=adapter_id, credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    normalized = await session.normalize_amount(Pair.parse("BTC/USDT:USDT"), Decimal("0.123456"))

    assert normalized == expected
    assert normalized <= Decimal("0.123456")


@pytest.mark.asyncio
async def test_normalize_amount_rejects_zero_unsafe_rounding_and_inverse_contracts():
    from cryptotrader.venues.okx import OkxVenueAdapter

    fake_factory = FakeCcxtFactory("okx")
    session = await OkxVenueAdapter(client_factory=fake_factory).connect(
        connection("normalize", "demo", adapter_id="okx", credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    client = fake_factory.clients[-1]

    with pytest.raises(VenueOperationError, match="positive finite Decimal"):
        await session.normalize_amount(Pair.parse("BTC/USDT:USDT"), Decimal("0"))

    client.amount_to_precision = lambda *_args: "13"
    with pytest.raises(VenueOperationError, match="unsafe amount normalization"):
        await session.normalize_amount(Pair.parse("BTC/USDT:USDT"), Decimal("0.123456"))

    with pytest.raises(VenueOperationError, match="inverse contracts are unsupported"):
        await session.normalize_amount(Pair.parse("BTC/USD:BTC"), Decimal("0.1"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet"),
    ],
)
async def test_margin_mode_and_leverage_are_configured_once_before_first_market_order(
    adapter_module, adapter_name, adapter_id, environment
):
    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection(
            "configured",
            environment,
            adapter_id=adapter_id,
            credential_ref="credentials",
            margin_mode="cross",
            leverage=7,
        ),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )
    intent = OrderIntent(Pair.parse("BTC/USDT:USDT"), "buy", Decimal("0.1"), "market", None, False)

    await session.place_order(intent)
    await session.place_order(intent)

    calls = fake_factory.clients[-1].calls
    configuration = [(name, payload) for name, payload in calls if name in {"set_margin_mode", "set_leverage"}]
    assert len([item for item in configuration if item[0] == "set_margin_mode"]) == 1
    if adapter_id == "bybit":
        assert len([item for item in configuration if item[0] == "set_leverage"]) == 1
        assert configuration == [
            ("set_margin_mode", ("cross", None, {})),
            ("set_leverage", (7, "BTC/USDT:USDT", {"category": "linear"})),
        ]
    else:
        assert configuration == [
            ("set_margin_mode", ("cross", "BTC/USDT:USDT", {"lever": 7})),
        ]
    first_create = next(index for index, (name, _) in enumerate(calls) if name == "create_order")
    assert all(
        index < first_create for index, (name, _) in enumerate(calls) if name in {"set_margin_mode", "set_leverage"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter_module", "adapter_name", "adapter_id", "environment", "expected_equity"),
    [
        ("cryptotrader.venues.okx", "OkxVenueAdapter", "okx", "demo", "12345.67"),
        ("cryptotrader.venues.bybit", "BybitVenueAdapter", "bybit", "testnet", "23456.78"),
    ],
)
async def test_portfolio_uses_account_equity_and_marks_spot_at_current_quote(
    adapter_module, adapter_name, adapter_id, environment, expected_equity
):
    fake_factory = FakeCcxtFactory(adapter_id)
    adapter_class = getattr(importlib.import_module(adapter_module), adapter_name)
    session = await adapter_class(client_factory=fake_factory).connect(
        connection("portfolio", environment, adapter_id=adapter_id, credential_ref="credentials"),
        CredentialPayload(
            values={
                "api_key": "key",  # pragma: allowlist secret
                "secret": "secret",  # pragma: allowlist secret
                "passphrase": "passphrase",  # pragma: allowlist secret
            }
        ),
    )

    snapshot = await session.fetch_portfolio(Pair.parse("BTC/USDT"))

    assert type(snapshot) is ConnectionPortfolioSnapshot
    assert snapshot.equity == Decimal(expected_equity)
    assert snapshot.balances == {"USDT": Decimal("10000.50"), "BTC": Decimal("0.25")}
    assert snapshot.position.signed_amount == Decimal("0.25")
    assert snapshot.position.signed_notional == Decimal("12500.000")
    assert all(isinstance(value, Decimal) for value in snapshot.balances.values())
