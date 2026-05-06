"""Background task manager — decouples analysis from HTTP connection lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from cryptotrader.chat.event_bus import EventBus
    from cryptotrader.config import ChatConfig

logger = logging.getLogger(__name__)


class TooManyTasksError(Exception):
    pass


@dataclass
class AnalysisTask:
    session_id: str
    pair: str
    trigger_source: str
    task: asyncio.Task[None]
    interrupt_event: asyncio.Event
    event_bus: EventBus
    created_at: float = field(default_factory=time.monotonic)
    completed: bool = False
    completed_agents: list[str] = field(default_factory=list)


class BackgroundTaskManager:
    _instance: BackgroundTaskManager | None = None

    def __init__(self, config: ChatConfig) -> None:
        self._config = config
        self._tasks: dict[str, AnalysisTask] = {}

    @classmethod
    def get_instance(cls, config: ChatConfig | None = None) -> BackgroundTaskManager:
        if cls._instance is None:
            if config is None:
                from cryptotrader.config import ChatConfig as ChatCfg

                config = ChatCfg()
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def create(
        self,
        session_id: str,
        pair: str,
        coro: Coroutine[Any, Any, None],
        trigger_source: str,
        event_bus: EventBus,
    ) -> AnalysisTask:
        active_count = sum(1 for t in self._tasks.values() if not t.completed)
        if active_count >= self._config.max_concurrent_tasks:
            raise TooManyTasksError(f"Max concurrent tasks ({self._config.max_concurrent_tasks}) reached")

        existing = self._tasks.get(session_id)
        if existing and not existing.completed:
            notify_task = asyncio.ensure_future(
                existing.event_bus.publish("session_replaced", {"session_id": session_id})
            )
            notify_task.add_done_callback(lambda _: None)
            existing.interrupt_event.set()

        interrupt_event = asyncio.Event()
        task = asyncio.create_task(coro, name=f"analysis:{session_id}")
        task.add_done_callback(lambda _t: self._on_task_done(session_id))

        analysis_task = AnalysisTask(
            session_id=session_id,
            pair=pair,
            trigger_source=trigger_source,
            task=task,
            interrupt_event=interrupt_event,
            event_bus=event_bus,
        )
        self._tasks[session_id] = analysis_task

        broadcast = asyncio.ensure_future(self._broadcast_new_workflow(session_id, pair, trigger_source))
        broadcast.add_done_callback(lambda _: None)

        return analysis_task

    def get(self, session_id: str) -> AnalysisTask | None:
        return self._tasks.get(session_id)

    def interrupt(self, session_id: str) -> bool:
        task = self._tasks.get(session_id)
        if task is None or task.completed:
            return False
        if task.interrupt_event.is_set():
            return False
        task.interrupt_event.set()
        return True

    @staticmethod
    async def _broadcast_new_workflow(
        session_id: str,
        pair: str,
        trigger_source: str,
    ) -> None:
        try:
            from cryptotrader.config import load_config
            from cryptotrader.risk.state import RedisStateManager

            config = load_config()
            state_mgr = RedisStateManager(config.infrastructure.redis_url or None)
            payload = json.dumps(
                {
                    "session_id": session_id,
                    "pair": pair,
                    "trigger_source": trigger_source,
                }
            )
            await state_mgr.publish("analysis:new_workflow", payload)
        except Exception:
            logger.info("Failed to broadcast new_workflow", exc_info=True)

    def _on_task_done(self, session_id: str) -> None:
        task = self._tasks.get(session_id)
        if task is None:
            return
        task.completed = True
        duration_ms = int((time.monotonic() - task.created_at) * 1000)
        logger.info(
            "Analysis task completed: session_id=%s pair=%s duration_ms=%d",
            session_id,
            task.pair,
            duration_ms,
        )
