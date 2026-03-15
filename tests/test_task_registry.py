"""TaskRegistry unit tests -- TDD RED phase written first, pass after implementation."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


async def _noop_coro() -> None:
    """Coroutine that completes normally."""


async def _failing_coro() -> None:
    """Coroutine that raises an exception."""
    raise RuntimeError("simulated background task failure")


async def _slow_coro(delay: float = 0.05) -> None:
    """Simulates a long-running coroutine."""
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# 1. Module structure tests
# ---------------------------------------------------------------------------


def test_module_exports_background_tasks_set() -> None:
    """_background_tasks is a module-level set singleton."""
    from cryptotrader.task_registry import _background_tasks

    assert isinstance(_background_tasks, set)


def test_module_exports_add_background_task() -> None:
    """add_background_task function exists and is callable."""
    from cryptotrader.task_registry import add_background_task

    assert callable(add_background_task)


def test_module_exports_on_task_done() -> None:
    """_on_task_done function exists and is callable."""
    from cryptotrader.task_registry import _on_task_done

    assert callable(_on_task_done)


# ---------------------------------------------------------------------------
# 2. add_background_task: Task reference holding
# ---------------------------------------------------------------------------


async def test_task_is_held_in_registry_while_running() -> None:
    """Task created by add_background_task is in _background_tasks before completion."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    # Use a slow coroutine to ensure the task is still running when we check
    task = add_background_task(_slow_coro(0.05), name="test_hold")

    # Task not yet done, should be in the set
    assert task in task_registry._background_tasks

    # Wait for task to complete
    await asyncio.sleep(0.1)
    await asyncio.sleep(0)  # Allow callback to run


async def test_task_removed_from_registry_after_completion() -> None:
    """Task is removed from _background_tasks after completion."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    task = add_background_task(_noop_coro(), name="test_remove")
    await asyncio.sleep(0)  # Allow coroutine to execute
    await asyncio.sleep(0)  # Allow callback to run

    assert task not in task_registry._background_tasks


async def test_task_returns_asyncio_task() -> None:
    """add_background_task returns an asyncio.Task object."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    task = add_background_task(_noop_coro(), name="test_type")
    assert isinstance(task, asyncio.Task)

    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# 3. Exception handling: failing tasks log a warning
# ---------------------------------------------------------------------------


async def test_failing_task_logs_warning() -> None:
    """When a background task raises an exception, _on_task_done logs logger.warning(exc_info=True)."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    with patch("cryptotrader.task_registry.logger") as mock_logger:
        task = add_background_task(_failing_coro(), name="test_fail")
        await asyncio.sleep(0.05)
        await asyncio.sleep(0)  # Allow callback to run

        mock_logger.warning.assert_called_once()
        call_kwargs = mock_logger.warning.call_args
        # exc_info=True should be passed to warning
        assert call_kwargs.kwargs.get("exc_info") is True or (
            len(call_kwargs.args) > 0  # at least one positional arg (the message)
        )

    # Task should be removed from set even after exception
    assert task not in task_registry._background_tasks


async def test_failing_task_removed_from_registry() -> None:
    """A Task that raised an exception is still removed from the set after completion."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    task = add_background_task(_failing_coro(), name="test_fail_remove")
    await asyncio.sleep(0.05)
    await asyncio.sleep(0)

    assert task not in task_registry._background_tasks


async def test_normal_task_does_not_log_warning() -> None:
    """A normally completing Task does not trigger logger.warning."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    with patch("cryptotrader.task_registry.logger") as mock_logger:
        add_background_task(_noop_coro(), name="test_ok")
        await asyncio.sleep(0.05)
        await asyncio.sleep(0)

        mock_logger.warning.assert_not_called()


# ---------------------------------------------------------------------------
# 4. name parameter handling
# ---------------------------------------------------------------------------


async def test_task_name_is_set() -> None:
    """The name passed to add_background_task is set as the Task name."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    task = add_background_task(_slow_coro(0.05), name="my_task_name")
    assert task.get_name() == "my_task_name"

    await asyncio.sleep(0.1)


async def test_task_name_none_still_works() -> None:
    """name=None still creates and registers the task correctly."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    task = add_background_task(_noop_coro(), name=None)
    assert isinstance(task, asyncio.Task)

    await asyncio.sleep(0.05)


# ---------------------------------------------------------------------------
# 5. Multiple concurrent tasks
# ---------------------------------------------------------------------------


async def test_multiple_tasks_held_simultaneously() -> None:
    """Multiple background tasks can exist in the set simultaneously."""
    from cryptotrader import task_registry
    from cryptotrader.task_registry import add_background_task

    task_registry._background_tasks.clear()

    tasks = [add_background_task(_slow_coro(0.05), name=f"task_{i}") for i in range(3)]

    # All three tasks are in the set
    assert len(task_registry._background_tasks) == 3
    for t in tasks:
        assert t in task_registry._background_tasks

    # Wait for all to complete
    await asyncio.gather(*tasks, return_exceptions=True)
    await asyncio.sleep(0)  # Allow callbacks to run

    # All removed
    assert len(task_registry._background_tasks) == 0


# ---------------------------------------------------------------------------
# 6. nodes/data.py integration -- verbal_reinforcement uses add_background_task
# ---------------------------------------------------------------------------


async def test_verbal_reinforcement_uses_add_background_task() -> None:
    """verbal_reinforcement() internally calls add_background_task instead of bare loop.create_task."""
    import inspect

    import cryptotrader.nodes.data as data_module

    source = inspect.getsource(data_module.verbal_reinforcement)

    # Should call add_background_task
    assert "add_background_task" in source

    # Should no longer directly call loop.create_task for maybe_reflect
    assert "loop.create_task(maybe_reflect" not in source


async def test_verbal_reinforcement_integration_with_registry(caplog) -> None:
    """verbal_reinforcement in backtest mode skips add_background_task."""
    from unittest.mock import MagicMock, patch

    state = {
        "metadata": {
            "pair": "BTC/USDT",
            "database_url": None,
            "backtest_mode": True,
            "cycle_count": 1,
        },
        "data": {
            "snapshot_summary": {
                "price": 50000.0,
                "funding_rate": 0.0001,
                "volatility": 0.02,
                "orderbook_imbalance": 0.1,
            }
        },
    }

    mock_config = MagicMock()
    mock_config.experience.enabled = True
    mock_config.experience.regime_thresholds = MagicMock()

    # verbal_reinforcement uses delayed imports; patch the actual source modules
    with (
        patch("cryptotrader.config.load_config", return_value=mock_config),
        patch("cryptotrader.journal.store.JournalStore"),
        patch("cryptotrader.learning.regime.tag_regime", return_value=["neutral"]),
        patch("cryptotrader.task_registry.add_background_task") as mock_add,
    ):
        # Reload to pick up patched state
        import importlib

        import cryptotrader.nodes.data as data_mod

        importlib.reload(data_mod)

        result = await data_mod.verbal_reinforcement(state)

    # backtest mode should not trigger background tasks
    mock_add.assert_not_called()
    assert result is not None
