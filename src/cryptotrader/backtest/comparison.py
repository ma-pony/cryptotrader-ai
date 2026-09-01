# ruff: noqa: RUF001 -- Chinese user-facing messages use Chinese punctuation.
"""Compare experiment conditions before presenting outcomes."""

from dataclasses import dataclass

CONDITIONS = ("pair", "start", "end", "interval", "initial_equity", "fee_rate", "slippage_bps", "funding_assumption")


@dataclass
class BacktestComparison:
    comparable: bool
    condition_differences: dict
    configuration_differences: dict
    left: object
    right: object


def _differences(left, right, keys):
    return {key: {"left": left.get(key), "right": right.get(key)} for key in keys if left.get(key) != right.get(key)}


def compare_runs(left, right):
    conditions = _differences(left.params.model_dump(mode="json"), right.params.model_dump(mode="json"), CONDITIONS)
    for key in CONDITIONS:
        if getattr(left.params, key) is None or getattr(right.params, key) is None:
            conditions[key] = {
                "left": left.params.model_dump(mode="json")[key],
                "right": right.params.model_dump(mode="json")[key],
                "reason": "历史条件缺失，不能判定可比",
            }
    if left.config_snapshot is None or right.config_snapshot is None:
        configurations = {
            "config_snapshot": {
                "left": left.config_snapshot,
                "right": right.config_snapshot,
                "reason": "历史配置快照缺失，不能判定可比",
            }
        }
    else:
        configurations = _differences(
            left.config_snapshot, right.config_snapshot, sorted(set(left.config_snapshot) | set(right.config_snapshot))
        )
    if left.model_evidence != right.model_evidence:
        configurations["model_evidence"] = {"left": left.model_evidence, "right": right.model_evidence}
    for name in ("data_coverage", "unmodeled_costs"):
        a, b = getattr(left.result, name, None), getattr(right.result, name, None)
        if a != b:
            configurations[name] = {"left": a, "right": b}
    if left.incomplete_fields or right.incomplete_fields:
        configurations["incomplete_fields"] = {"left": left.incomplete_fields, "right": right.incomplete_fields}
    return BacktestComparison(
        not conditions
        and not configurations
        and not left.incomplete_fields
        and not right.incomplete_fields
        and left.config_snapshot is not None
        and right.config_snapshot is not None,
        conditions,
        configurations,
        left,
        right,
    )
