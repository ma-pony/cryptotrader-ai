"""Alert recording and external sending have independent failure boundaries."""

import httpx

from cryptotrader.notifications import WebhookBackend


class AlertService:
    def __init__(self, store, config_provider):
        self.store = store
        self.config_provider = config_provider

    async def record(self, event):
        config = await self.config_provider()
        channel = "webhook" if config.enabled and config.webhook_url and event.type in config.events else None
        return await self.store.record_once(event.event_key, event, channel=channel)

    async def mark_read(self, alert_id):
        return await self.store.mark_read(alert_id)


class DeliveryService:
    def __init__(self, store, config_provider, *, backend_factory=None):
        self.store = store
        self.config_provider = config_provider
        self.backend_factory = backend_factory or (
            lambda config: WebhookBackend(config.webhook_url, config.webhook_timeout)
        )

    async def send(self, delivery_id):
        delivery = await self.store.get_delivery(delivery_id)
        event = await self.store.get_alert(delivery.alert_id)
        config = await self.config_provider()
        if not config.enabled or not config.webhook_url or event.type not in config.events:
            return
        if not await self.store.claim_delivery(delivery_id):
            return
        error = None
        try:
            await self.backend_factory(config).send(event.type, event.model_dump(mode="json"))
        except httpx.HTTPStatusError as exc:
            error = f"Webhook HTTP {exc.response.status_code}"
        except Exception:
            error = "Webhook 连接失败"
        await self.store.finish_delivery(delivery_id, error)

    async def retry(self, delivery_id):
        return await self.store.retry_delivery(delivery_id)
