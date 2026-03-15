"""Background task registry to prevent asyncio.Task from being GC-collected.

Python asyncio documentation states: if a Task object has no external reference,
the garbage collector may collect and cancel it at any time. This module provides
a module-level singleton set to hold Task references, removes them via callback
on completion, and logs any uncaught exceptions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

logger = logging.getLogger(__name__)

# Module-level singleton: holds all background Task references to prevent GC collection
_background_tasks: set[asyncio.Task[Any]] = set()


def _on_task_done(task: asyncio.Task[Any], name: str | None) -> None:
    """Task completion callback.

    - Removes the Task reference from _background_tasks
    - If the Task raised an uncaught exception, logs logger.warning(exc_info=True)
    """
    _background_tasks.discard(task)

    if task.cancelled():
        return

    exc = task.exception()
    if exc is not None:
        logger.warning(
            "Background task %r raised an uncaught exception",
            name or task.get_name(),
            exc_info=exc,
        )


def add_background_task(
    coro: Coroutine[Any, Any, Any],
    name: str | None = None,
) -> asyncio.Task[Any]:
    """Create a background Task, hold its reference, and register a completion callback.

    Parameters
    ----------
    coro:
        The coroutine object to schedule.
    name:
        Optional Task name. Readable via Task.get_name() for log tracing.

    Returns
    -------
    asyncio.Task
        The Task object that has been created and added to the event loop.

    Raises
    ------
    RuntimeError
        Raised by asyncio.get_running_loop() if no event loop is running in the
        current thread.
    """
    loop = asyncio.get_running_loop()

    task = loop.create_task(coro, name=name) if name is not None else loop.create_task(coro)

    # Hold reference to prevent GC collection
    _background_tasks.add(task)

    # Register completion callback: remove reference and handle exceptions
    task.add_done_callback(lambda t: _on_task_done(t, name))

    return task
