"""Browser-independent alert recovery and outbox owner."""

import asyncio
import logging
from contextlib import suppress

logger = logging.getLogger(__name__)


class AlertOwner:
    def __init__(self, recovery, delivery, *, interval=5):
        self.recovery = recovery
        self.delivery = delivery
        self.interval = interval
        self._wake = asyncio.Event()
        self._stop = asyncio.Event()
        self._task = None

    def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="business-alerts")

    def refresh(self):
        self._wake.set()

    async def publish(self, event):
        if event.name == "cycle_completed":
            self.refresh()

    async def stop(self):
        self._stop.set()
        self._wake.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _run(self):
        recovered = False
        while not self._stop.is_set():
            self._wake.clear()
            try:
                if not recovered:
                    await self.delivery.store.recover_interrupted()
                    recovered = True
                await self.recovery.reconcile()
            except Exception:
                # Database/observer failure must not mutate or abort the trading owner.
                logger.warning("Business alert recovery failed; will retry")
            try:
                for delivery in await self.delivery.store.list_deliveries(status="pending"):
                    if self._stop.is_set():
                        break
                    await self.delivery.send(delivery.id)
            except Exception:
                logger.warning("Alert outbox processing failed; business state is unchanged")
            if not self._stop.is_set():
                with suppress(TimeoutError):
                    await asyncio.wait_for(self._wake.wait(), self.interval)
