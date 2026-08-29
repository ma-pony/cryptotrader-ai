"""Background task manager — decouples analysis from HTTP connection lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.decision.models import CycleOutcome

    WorkflowPublisher = Callable[[str, str], Awaitable[object]]

logger = logging.getLogger(__name__)


class TooManyTasksError(Exception):
    pass


class ExecutionInProgressError(Exception):
    pass


@dataclass
class AnalysisTask:
    session_id: str
    pair: str
    trigger_source: str
    task: asyncio.Task[CycleOutcome | None]
    interrupt_event: asyncio.Event
    event_bus: EventBus
    created_at: float = field(default_factory=time.monotonic)
    completed: bool = False
    outcome: CycleOutcome | None = None


class BackgroundTaskManager:
    _instance: BackgroundTaskManager | None = None

    def __init__(
        self,
        *,
        max_concurrent_tasks: int = 5,
        workflow_publisher: WorkflowPublisher | None = None,
    ) -> None:
        self._max_concurrent_tasks = max_concurrent_tasks
        self._workflow_publisher = workflow_publisher
        self._tasks: dict[str, AnalysisTask] = {}

    @classmethod
    def get_instance(
        cls,
        *,
        max_concurrent_tasks: int = 5,
        workflow_publisher: WorkflowPublisher | None = None,
    ) -> BackgroundTaskManager:
        if cls._instance is None:
            cls._instance = cls(
                max_concurrent_tasks=max_concurrent_tasks,
                workflow_publisher=workflow_publisher,
            )
        elif workflow_publisher is not None:
            cls._instance._workflow_publisher = workflow_publisher
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def create(
        self,
        session_id: str,
        pair: str,
        runner: Callable[[asyncio.Event], Awaitable[CycleOutcome | None]],
        trigger_source: str,
        event_bus: EventBus,
    ) -> AnalysisTask:
        existing = self._tasks.get(session_id)
        if existing and not existing.completed and existing.event_bus.execution_started:
            raise ExecutionInProgressError("analysis execution is already in progress")

        active_count = sum(1 for t in self._tasks.values() if not t.completed)
        if active_count >= self._max_concurrent_tasks:
            raise TooManyTasksError(f"Max concurrent tasks ({self._max_concurrent_tasks}) reached")

        if existing and not existing.completed:
            notify_task = asyncio.ensure_future(
                existing.event_bus.publish("session_replaced", {"session_id": session_id})
            )
            notify_task.add_done_callback(lambda _: None)
            existing.interrupt_event.set()
            existing.task.cancel()

        interrupt_event = asyncio.Event()
        task = asyncio.create_task(runner(interrupt_event), name=f"analysis:{session_id}")
        task.add_done_callback(lambda completed_task: self._on_task_done(session_id, completed_task))

        analysis_task = AnalysisTask(
            session_id=session_id,
            pair=pair,
            trigger_source=trigger_source,
            task=task,
            interrupt_event=interrupt_event,
            event_bus=event_bus,
        )
        self._tasks[session_id] = analysis_task

        publisher = self._workflow_publisher
        if publisher is not None:
            broadcast = asyncio.ensure_future(self._broadcast_new_workflow(publisher, session_id, pair, trigger_source))
            broadcast.add_done_callback(lambda _: None)

        return analysis_task

    def get(self, session_id: str) -> AnalysisTask | None:
        return self._tasks.get(session_id)

    def interrupt(self, session_id: str) -> AnalysisTask | None:
        task = self._tasks.get(session_id)
        if task is None or task.completed:
            return None
        if task.event_bus.execution_started:
            return None
        if task.interrupt_event.is_set():
            return None
        task.interrupt_event.set()
        task.task.cancel()
        return task

    async def drain(self) -> None:
        """Stop cancellable analysis and await order-bearing work to terminal state."""
        active = tuple(task for task in self._tasks.values() if not task.completed)
        for analysis in active:
            if not analysis.event_bus.execution_started:
                analysis.interrupt_event.set()
                analysis.task.cancel()
        if active:
            await asyncio.gather(*(analysis.task for analysis in active), return_exceptions=True)

    @staticmethod
    async def _broadcast_new_workflow(
        publisher: WorkflowPublisher,
        session_id: str,
        pair: str,
        trigger_source: str,
    ) -> None:
        try:
            payload = json.dumps(
                {
                    "session_id": session_id,
                    "pair": pair,
                    "trigger_source": trigger_source,
                }
            )
            await publisher("analysis:new_workflow", payload)
        except Exception:
            logger.info("Failed to broadcast new_workflow")

    def _on_task_done(self, session_id: str, completed_task: asyncio.Task[CycleOutcome | None]) -> None:
        analysis_task = self._tasks.get(session_id)
        if analysis_task is None or analysis_task.task is not completed_task:
            return
        analysis_task.completed = True
        if not completed_task.cancelled() and completed_task.exception() is None:
            analysis_task.outcome = completed_task.result()
        duration_ms = int((time.monotonic() - analysis_task.created_at) * 1000)
        logger.info(
            "Analysis task completed: session_id=%s pair=%s duration_ms=%d",
            session_id,
            analysis_task.pair,
            duration_ms,
        )
