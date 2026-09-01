"""Notification selection belongs to durable alert services."""

from cryptotrader.runtime_config.models import NotificationConfig


def test_notification_selection_is_explicit_and_strict():
    config = NotificationConfig(enabled=False, events=("connection_failed", "daily_summary"))
    assert config.events == ("connection_failed", "daily_summary")
