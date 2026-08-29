"""Deterministic strict-JSON boundary coverage."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from cryptotrader.cycle_serialization import json_value


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_venue_capabilities_in_completed_decision_are_stable_across_hash_seeds(seed: int) -> None:
    script = """
import json
from cryptotrader.cycle_serialization import json_value
from cryptotrader.venues.models import VenueCapabilities

capabilities = VenueCapabilities(
    frozenset({"spot", "swap"}),
    native_protection=True,
    hedge_mode=False,
    reduce_only=True,
    supported_order_types=frozenset({"market", "limit"}),
)
decision = {
    "status": "completed",
    "books": [{
        "status": "completed",
        "proposal": {"connection_plans": [{"capabilities": capabilities}]},
    }],
}
print(json.dumps(json_value(decision), separators=(",", ":")))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[1],
        env=os.environ | {"PYTHONHASHSEED": str(seed)},
        check=True,
        capture_output=True,
        text=True,
    )

    capabilities = json.loads(completed.stdout)["books"][0]["proposal"]["connection_plans"][0]["capabilities"]
    assert capabilities == {
        "market_types": ["spot", "swap"],
        "native_protection": True,
        "hedge_mode": False,
        "reduce_only": True,
        "supported_order_types": ["limit", "market"],
    }


def test_set_member_circular_reference_remains_an_error() -> None:
    @dataclass(frozen=True)
    class CircularMember:
        payload: object = field(compare=False, hash=False)

    circular: list[object] = []
    circular.append(circular)

    with pytest.raises(ValueError, match="Circular reference detected"):
        json_value(frozenset({CircularMember(circular)}))
