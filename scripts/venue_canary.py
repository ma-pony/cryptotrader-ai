"""Manual simulated-venue canary with a mandatory cleanup and reconnect audit.

This script deliberately has no credential flags.  It loads one connection and
its encrypted credential from the active database runtime configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from cryptotrader.bootstrap import BootstrapSettings
from cryptotrader.cycle_lock import execution_pair_lease
from cryptotrader.execution_ownership import wait_for_owned
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.repository import RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from cryptotrader.venues.registry import VenueAdapterRegistry

_SIMULATED_ENVIRONMENTS = frozenset({"paper", "demo", "testnet"})
_REDACTED = "[redacted]"
_CANARY_QUOTE_NOTIONAL = Decimal("10")
_ORDER_POLL_SECONDS = 0.25
_ORDER_POLL_ATTEMPTS = 3


class CanarySafetyError(RuntimeError):
    """A requested canary action crosses a non-negotiable safety boundary."""


def parse_venue_canary_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simulated venue canary")
    parser.add_argument("--connection", required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--audit", action="store_true", help="internal reconnect audit mode")
    parser.add_argument("--live-read-only", action="store_true", help="only read a live connection state")
    return parser.parse_args(argv)


def require_simulated_environment(environment: str) -> None:
    if environment == "live":
        raise CanarySafetyError("live connections are read-only in canary")
    if environment not in _SIMULATED_ENVIRONMENTS:
        raise CanarySafetyError("canary requires an explicit paper, demo, or testnet connection")


def require_canary_only(connection) -> None:
    if not connection.canary_only:
        raise CanarySafetyError("simulated write canary requires a canary_only connection")


def _safe_value(value: Any, key: str = "") -> Any:
    normalized = key.lower().replace("-", "_")
    if any(token in normalized for token in ("secret", "token", "key", "passphrase", "authorization", "credential")):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(item_key): _safe_value(item, str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item) for item in value]
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat()
    return value


def safe_json(value: Any) -> str:
    return json.dumps(_safe_value(value), ensure_ascii=False, sort_keys=True, default=str)


async def _open_connection(connection, repository):
    registry = VenueAdapterRegistry.discover((connection.adapter_id,))
    credentials = None
    if connection.credential_ref is not None:
        credentials = await repository.reveal_credentials(connection.credential_ref)
    return await registry.require(connection.adapter_id).connect(connection, credentials)


async def _load_target(connection_id: str):
    settings = BootstrapSettings.from_environment()
    repository = RuntimeConfigRepository(settings.database_url, CredentialVault(settings.config_master_key))
    snapshot = await repository.get_existing()
    if not snapshot.operational:
        raise CanarySafetyError("active runtime configuration is required")
    connection = next((item for item in snapshot.document.execution.connections if item.id == connection_id), None)
    if connection is None or not connection.enabled:
        raise CanarySafetyError("configured enabled connection was not found")
    return snapshot, connection, repository


def _residual_from_state(state) -> dict[str, bool]:
    return {
        "position_nonzero": state.position.signed_amount != Decimal("0"),
        "open_orders": bool(state.open_orders),
        "protections": bool(state.protections),
    }


async def inspect_residual(session, pair: Pair) -> dict[str, Any]:
    residual = _residual_from_state(await session.list_open_state(pair))
    return {"residual": residual, "requires_attention": any(residual.values())}


async def _find_owned_order(session, pair: Pair, client_order_id: str, order_id: str | None = None):
    """Read back one exchange-visible client id; never infer ownership from a position."""
    if order_id:
        try:
            order = await session.find_order(pair, order_id=order_id)
            if order is not None and order.client_order_id == client_order_id:
                return order
        except Exception:
            # Exchange order indexes can lag acknowledgement; client id is the
            # independent ownership proof and must still be queried.
            pass
    order = await session.find_order(pair, client_order_id=client_order_id)
    return order if order is not None and order.client_order_id == client_order_id else None


async def _confirmed_order(session, pair: Pair, order, client_order_id: str):  # noqa: C901 - bounded order state machine
    """Return an owned, settled order; pending acknowledgements are cancelled first."""
    if order.client_order_id != client_order_id:
        raise CanarySafetyError("canary order ownership is not confirmed")
    observed = order
    for attempt in range(_ORDER_POLL_ATTEMPTS):
        if observed.status not in {"open", "partial", "partially_filled"}:
            if observed.filled_amount > 0:
                return observed
            raise CanarySafetyError("canary order has no confirmed fill")
        if attempt:
            await asyncio.sleep(_ORDER_POLL_SECONDS)
        observed = await _find_owned_order(session, pair, client_order_id, order.id)
        if observed is None:
            break
        if observed.status not in {"open", "partial", "partially_filled"} and observed.filled_amount > 0:
            return observed
    if observed is not None and observed.status in {"open", "partial", "partially_filled"}:
        await session.cancel_order(observed.id, pair)
        await asyncio.sleep(_ORDER_POLL_SECONDS)
        final = await _find_owned_order(session, pair, client_order_id, observed.id)
        if final is not None and final.status in {"open", "partial", "partially_filled"}:
            raise CanarySafetyError("owned pending order remains open after cancellation")
    if observed is not None and observed.filled_amount > 0:
        return observed
    raise CanarySafetyError("canary order fill is not confirmed")


def _protection_matches(spec: ProtectionSpec, protection) -> bool:
    """Claim a lost protection response only with an exact dedicated-account match."""
    return (
        protection.pair == spec.pair
        and protection.position_side == spec.position_side
        and protection.amount == spec.amount
        and protection.stop_loss == spec.stop_loss
        and protection.take_profit == spec.take_profit
    )


async def _recover_protection_ids(session, pair: Pair, spec: ProtectionSpec) -> tuple[str, ...] | None:
    """Boundedly recover an accepted protection whose response was lost."""
    for attempt in range(_ORDER_POLL_ATTEMPTS):
        state = await session.list_open_state(pair)
        matches = tuple(item for item in state.protections if _protection_matches(spec, item))
        if len(matches) == 1:
            return tuple(matches[0].protection_ids)
        if len(matches) > 1:
            return None
        if attempt + 1 < _ORDER_POLL_ATTEMPTS:
            await asyncio.sleep(_ORDER_POLL_SECONDS)
    return None


def _record_order(ledger: dict[str, Any], order) -> None:
    """Keep cumulative fills monotonic while replacing the order state view."""
    if order is not None and order.client_order_id is not None:
        prior = ledger.get(order.client_order_id)
        if prior is None or order.filled_amount >= prior.filled_amount:
            ledger[order.client_order_id] = order


def _owned_exposure(ledger: Mapping[str, Any]) -> Decimal:
    return sum(
        (order.filled_amount if order.side == "buy" else -order.filled_amount for order in ledger.values()),
        Decimal("0"),
    )


async def _cleanup_owned(  # noqa: C901 - bounded cleanup state machine
    session,
    pair: Pair,
    *,
    allowed_client_order_ids: frozenset[str],
    fill_ledger: dict[str, Any],
    cleanup_client_order_id: str,
    protection_ids: tuple[str, ...],
    expected_protection: ProtectionSpec | None,
) -> dict[str, Any]:
    """Cancel and close only objects positively tagged as belonging to this run."""
    errors: list[str] = []
    attention = False
    try:
        if expected_protection is not None and not protection_ids:
            recovered = await _recover_protection_ids(session, pair, expected_protection)
            if recovered is None:
                # Unknown protections are never guessed/cancelled, but known
                # client-id orders and the dedicated account position still
                # need exact reduce-only cleanup.
                attention = True
                errors.append("UnknownProtectionOwnership")
            else:
                protection_ids = recovered
        state = await session.list_open_state(pair)
        owned_open = tuple(order for order in state.open_orders if order.client_order_id in allowed_client_order_ids)
        order_cancel_failed = False
        for order in owned_open:
            _record_order(fill_ledger, order)
        for order in owned_open:
            try:
                await session.cancel_order(order.id, pair)
            except Exception as error:
                attention = True
                order_cancel_failed = True
                errors.append(type(error).__name__)
        if order_cancel_failed:
            raise CanarySafetyError("owned pending orders could not be cancelled")
        if owned_open:
            confirmed = await session.list_open_state(pair)
            if any(order.client_order_id in allowed_client_order_ids for order in confirmed.open_orders):
                raise CanarySafetyError("owned pending order remains open after cancellation")
        if protection_ids:
            try:
                await session.cancel_protection(protection_ids)
            except Exception as error:
                attention = True
                errors.append(type(error).__name__)
        # Also recover a timed-out acknowledgement that only became visible
        # during finalization.  Each client id is overwritten, never summed.
        for client_order_id in allowed_client_order_ids:
            try:
                observed = await _find_owned_order(session, pair, client_order_id)
            except Exception as error:
                # The ledger may already have a positively confirmed fill;
                # an unavailable historical-order query cannot erase it.
                attention = True
                errors.append(type(error).__name__)
            else:
                _record_order(fill_ledger, observed)
        owned_exposure = _owned_exposure(fill_ledger)
        if owned_exposure > 0:
            state = await session.list_open_state(pair)
            if state.position.signed_amount != owned_exposure:
                attention = True
                errors.append("UnverifiableOwnedExposure")
            else:
                closing = await session.place_order(
                    OrderIntent(pair, "sell", owned_exposure, "market", None, True, cleanup_client_order_id)
                )
                closed = await _confirmed_order(session, pair, closing, cleanup_client_order_id)
                _record_order(fill_ledger, closed)
                owned_exposure = _owned_exposure(fill_ledger)
                if owned_exposure:
                    attention = True
                    errors.append("UnclosedOwnedExposure")
    except Exception as error:
        attention = True
        errors.append(type(error).__name__)
    try:
        residual = await inspect_residual(session, pair)
    except Exception as error:
        residual = {"residual": {"state_unavailable": True}, "requires_attention": True}
        attention = True
        errors.append(type(error).__name__)
    return {**residual, "cleanup_errors": errors, "requires_attention": attention or residual["requires_attention"]}


async def _cleanup_and_close(session, pair: Pair, **kwargs: Any) -> dict[str, Any]:
    """One owned finalizer: cancellation cannot strand an open transport session."""
    cleanup: dict[str, Any]
    try:
        cleanup = await _cleanup_owned(session, pair, **kwargs)
    except BaseException as error:
        cleanup = {"cleanup_errors": [type(error).__name__], "requires_attention": True}
    try:
        await session.close()
    except BaseException as error:
        cleanup["close_error"] = type(error).__name__
        cleanup["requires_attention"] = True
    return cleanup


async def run_simulated_canary(session, pair: Pair, *, close_session: bool = True) -> dict[str, Any]:  # noqa: C901 - bounded safety procedure
    """Run writes only against a session already proven simulated by the caller."""
    result: dict[str, Any] = {"status": "failed", "started_at": datetime.now(UTC), "steps": []}
    canary_write_attempted = False
    client_order_prefix = f"CT{uuid4().hex[:16].upper()}"
    open_client_order_id = f"{client_order_prefix}O"
    close_client_order_id = f"{client_order_prefix}C"
    cleanup_client_order_id = f"{client_order_prefix}X"
    owned_protection_ids: tuple[str, ...] = ()
    expected_protection: ProtectionSpec | None = None
    fill_ledger: dict[str, Any] = {}
    try:
        await session.fetch_portfolio(pair)
        quote = await session.fetch_quote(pair)
        initial = await inspect_residual(session, pair)
        if initial["requires_attention"]:
            raise CanarySafetyError("canary requires initial zero position, orders, and protections")
        result["steps"].append("read_health_balance_open_state")
        minimum_amount = await session.minimum_amount(pair, quote.last, _CANARY_QUOTE_NOTIONAL)
        amount = minimum_amount
        if amount <= 0:
            raise CanarySafetyError("venue did not provide a positive minimum canary amount")
        canary_write_attempted = True
        try:
            raw_opened = await session.place_order(
                OrderIntent(pair, "buy", amount, "market", None, False, open_client_order_id)
            )
        except Exception:
            raw_opened = await _find_owned_order(session, pair, open_client_order_id)
            if raw_opened is None:
                raise
        # A recovered acknowledgement is still subject to the same bounded
        # confirmation/cancellation state machine; never promote it directly.
        opened = await _confirmed_order(session, pair, raw_opened, open_client_order_id)
        _record_order(fill_ledger, opened)
        result["open_order_id"] = opened.id
        result["steps"].append("open_minimum_position")
        expected_protection = await session.normalize_protection(
            ProtectionSpec(
                pair,
                "long",
                opened.filled_amount,
                quote.last * Decimal("0.98"),
                quote.last * Decimal("1.02"),
            )
        )
        protection = await session.replace_protection(expected_protection)
        owned_protection_ids = tuple(protection.protection_ids)
        result["protection_ids"] = owned_protection_ids
        result["steps"].append("install_protection")
        state = await session.list_open_state(pair)
        visible_protection_ids = {item_id for item in state.protections for item_id in item.protection_ids}
        if not set(owned_protection_ids) <= visible_protection_ids:
            raise CanarySafetyError("platform protection was not observable")
        if state.position.signed_amount != opened.filled_amount:
            raise CanarySafetyError("owned exposure cannot be separated from current position")
        closing = await session.place_order(
            OrderIntent(pair, "sell", opened.filled_amount, "market", None, True, close_client_order_id)
        )
        closing = await _confirmed_order(session, pair, closing, close_client_order_id)
        _record_order(fill_ledger, closing)
        if _owned_exposure(fill_ledger) == 0:
            await session.cancel_protection(owned_protection_ids)
        result["steps"].append("reduce_only_close_and_cancel")
        if _owned_exposure(fill_ledger) == 0:
            result["status"] = "completed"
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        if canary_write_attempted:
            cleanup = await wait_for_owned(
                asyncio.create_task(
                    _cleanup_and_close(
                        session,
                        pair,
                        allowed_client_order_ids=frozenset(
                            {open_client_order_id, close_client_order_id, cleanup_client_order_id}
                        ),
                        fill_ledger=fill_ledger,
                        cleanup_client_order_id=cleanup_client_order_id,
                        protection_ids=owned_protection_ids,
                        expected_protection=expected_protection,
                    )
                )
            )
            result.update(cleanup)
        else:
            result.update(await inspect_residual(session, pair))
        if close_session and not canary_write_attempted:
            try:
                await wait_for_owned(asyncio.create_task(session.close()))
            except Exception as error:
                result["close_error"] = type(error).__name__
                result["requires_attention"] = True
        if result["requires_attention"]:
            result["status"] = "failed"
        elif (
            canary_write_attempted
            and result.get("status") == "failed"
            and not result.get("cleanup_errors")
            and "error_type" not in result
        ):
            result["status"] = "completed"
    return _safe_value(result)


async def audit_in_subprocess(connection_id: str, pair: str) -> dict[str, Any]:
    """Reconnect through a new interpreter; it can only perform read operations."""
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, __file__, "--connection", connection_id, "--pair", pair, "--audit"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return {"audit_status": "failed", "requires_attention": True}
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"audit_status": "failed", "requires_attention": True}
    residual = payload.get("residual")
    if (
        not isinstance(payload, dict)
        or payload.get("status") != "completed"
        or payload.get("requires_attention") is not False
        or not isinstance(residual, dict)
        or any(residual.get(key) is not False for key in ("position_nonzero", "open_orders", "protections"))
    ):
        return {"audit_status": "failed", "requires_attention": True}
    return {"audit_status": payload.get("status"), "audit": payload}


def merge_audit_result(result: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Attach audit evidence without allowing a clean audit to erase a failed write flow."""
    merged = {**result, **audit}
    if audit["audit_status"] != "completed":
        merged["status"] = "failed"
        merged["requires_attention"] = True
    else:
        merged["requires_attention"] = bool(result.get("requires_attention", False))
        if result.get("status") != "completed":
            merged["status"] = "failed"
    return merged


async def _main(options: argparse.Namespace) -> dict[str, Any]:
    snapshot, connection, repository = await _load_target(options.connection)
    pair = Pair.parse(options.pair)
    if options.live_read_only:
        if connection.environment != "live":
            raise CanarySafetyError("live read-only mode requires a live connection")
        session = await _open_connection(connection, repository)
        try:
            portfolio = await session.fetch_portfolio(pair)
            state = await session.list_open_state(pair)
            return {
                "status": "completed",
                "environment": connection.environment,
                "connection_id": connection.id,
                "config_revision": snapshot.revision,
                "mode": "live_read_only",
                "capabilities": sorted(session.capabilities.market_types),
                "health": {"portfolio_read": portfolio.connection_id == connection.id, "open_state_read": True},
                "open_state": _residual_from_state(state),
            }
        finally:
            await wait_for_owned(asyncio.create_task(session.close()))
    require_simulated_environment(connection.environment)
    if options.audit:
        session = await _open_connection(connection, repository)
        try:
            result = await inspect_residual(session, pair)
            return {"status": "completed" if not result["requires_attention"] else "failed", **result}
        finally:
            await wait_for_owned(asyncio.create_task(session.close()))
    require_canary_only(connection)
    if pair.market_type != "swap":
        raise CanarySafetyError("simulated write canary requires a swap pair")
    result: dict[str, Any] = {"status": "failed", "requires_attention": True}
    cancellation: asyncio.CancelledError | None = None
    try:
        async with execution_pair_lease(snapshot.document.infrastructure.redis_url, pair.canonical()):
            session = await _open_connection(connection, repository)
            result = await run_simulated_canary(session, pair)
    except asyncio.CancelledError as error:
        # The lease context has already completed its cancellation-owned exit;
        # preserve the original cancellation after independent read-only audit.
        cancellation = error
        result = {"status": "failed", "requires_attention": True, "error_type": type(error).__name__}
    except Exception as error:
        result = {"status": "failed", "requires_attention": True, "error_type": type(error).__name__}
    result.update(
        {"connection_id": connection.id, "environment": connection.environment, "config_revision": snapshot.revision}
    )
    try:
        audit = await wait_for_owned(asyncio.create_task(audit_in_subprocess(connection.id, pair.canonical())))
    except asyncio.CancelledError as error:
        cancellation = cancellation or error
        audit = {"audit_status": "failed", "requires_attention": True}
    merged = merge_audit_result(result, audit)
    if cancellation is not None:
        raise cancellation
    return merged


def main(argv: list[str] | None = None) -> int:
    options = parse_venue_canary_args(argv)
    try:
        result = asyncio.run(_main(options))
    except Exception as error:
        result = {"status": "failed", "requires_attention": True, "error_type": type(error).__name__}
    print(safe_json(result))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
