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
from cryptotrader.pair import Pair
from cryptotrader.runtime_config.repository import RuntimeConfigRepository
from cryptotrader.runtime_config.secrets import CredentialVault
from cryptotrader.venues.models import OrderIntent, ProtectionSpec
from cryptotrader.venues.registry import VenueAdapterRegistry

_SIMULATED_ENVIRONMENTS = frozenset({"paper", "demo", "testnet"})
_REDACTED = "[redacted]"


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


async def _cleanup(session, pair: Pair, *, canary_write_attempted: bool) -> dict[str, Any]:
    errors: list[str] = []
    try:
        state = await session.list_open_state(pair)
        if canary_write_attempted and state.position.signed_amount != Decimal("0"):
            intent = OrderIntent(
                pair,
                "sell" if state.position.signed_amount > 0 else "buy",
                abs(state.position.signed_amount),
                "market",
                None,
                True,
            )
            await session.place_order(intent)
        if canary_write_attempted and state.protections:
            protection_ids = tuple(item for protection in state.protections for item in protection.protection_ids)
            await session.cancel_protection(protection_ids)
    except Exception as error:
        errors.append(type(error).__name__)
    try:
        residual = await inspect_residual(session, pair)
    except Exception as error:
        residual = {"residual": {"state_unavailable": True}, "requires_attention": True}
        errors.append(type(error).__name__)
    return {**residual, "cleanup_errors": errors}


async def run_simulated_canary(session, pair: Pair) -> dict[str, Any]:  # noqa: C901 - bounded safety procedure
    """Run writes only against a session already proven simulated by the caller."""
    result: dict[str, Any] = {"status": "failed", "started_at": datetime.now(UTC), "steps": []}
    canary_write_attempted = False
    client_order_id = f"canary-{uuid4().hex[:24]}"
    owned_order_ids: set[str] = set()
    owned_protection_ids: tuple[str, ...] = ()
    try:
        await session.fetch_portfolio(pair)
        quote = await session.fetch_quote(pair)
        initial = await inspect_residual(session, pair)
        if initial["requires_attention"]:
            raise CanarySafetyError("canary requires initial zero position, orders, and protections")
        result["steps"].append("read_health_balance_open_state")
        amount = await session.minimum_amount(pair, quote.last)
        if amount <= 0:
            raise CanarySafetyError("venue did not provide a positive minimum canary amount")
        canary_write_attempted = True
        opened = await session.place_order(OrderIntent(pair, "buy", amount, "market", None, False, client_order_id))
        if opened.client_order_id != client_order_id or opened.filled_amount <= 0:
            raise CanarySafetyError("canary opening order ownership or fill is not confirmed")
        owned_order_ids.add(opened.id)
        result["open_order_id"] = opened.id
        result["steps"].append("open_minimum_position")
        protection = await session.replace_protection(
            ProtectionSpec(
                pair,
                "long",
                opened.filled_amount,
                quote.last * Decimal("0.98"),
                quote.last * Decimal("1.02"),
            )
        )
        owned_protection_ids = tuple(protection.protection_ids)
        result["protection_ids"] = owned_protection_ids
        result["steps"].append("install_protection")
        state = await session.list_open_state(pair)
        if not state.protections:
            raise CanarySafetyError("platform protection was not observable")
        if state.position.signed_amount != opened.filled_amount:
            raise CanarySafetyError("owned exposure cannot be separated from current position")
        closing = await session.place_order(
            OrderIntent(pair, "sell", opened.filled_amount, "market", None, True, client_order_id)
        )
        if closing.client_order_id != client_order_id or closing.filled_amount != opened.filled_amount:
            raise CanarySafetyError("canary reduce-only close is not confirmed")
        owned_order_ids.add(closing.id)
        await session.cancel_protection(owned_protection_ids)
        result["steps"].append("reduce_only_close_and_cancel")
        result["status"] = "completed"
    except Exception as error:
        result["error_type"] = type(error).__name__
    finally:
        if canary_write_attempted:
            for order_id in owned_order_ids:
                try:
                    await session.cancel_order(order_id, pair)
                except Exception:
                    result["requires_attention"] = True
            if owned_protection_ids:
                try:
                    await session.cancel_protection(owned_protection_ids)
                except Exception:
                    result["requires_attention"] = True
        result.update(await _cleanup(session, pair, canary_write_attempted=False))
        try:
            await session.close()
        except Exception as error:
            result["close_error"] = type(error).__name__
            result["requires_attention"] = True
        if result["requires_attention"]:
            result["status"] = "failed"
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
            await session.close()
    require_simulated_environment(connection.environment)
    session = await _open_connection(connection, repository)
    if options.audit:
        try:
            result = await inspect_residual(session, pair)
            return {"status": "completed" if not result["requires_attention"] else "failed", **result}
        finally:
            await session.close()
    result = await run_simulated_canary(session, pair)
    result.update(
        {"connection_id": connection.id, "environment": connection.environment, "config_revision": snapshot.revision}
    )
    if result["status"] == "completed":
        result.update(await audit_in_subprocess(connection.id, pair.canonical()))
        if result["audit_status"] != "completed":
            result["status"] = "failed"
            result["requires_attention"] = True
    return result


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
