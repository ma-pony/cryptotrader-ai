"""Scheduler start/stop tests."""

import pytest
from cryptotrader.scheduler import Scheduler


def test_scheduler_init():
    s = Scheduler(["BTC/USDT"], interval_minutes=60)
    assert s.pairs == ["BTC/USDT"]
    assert s.interval == 3600


def test_scheduler_stop():
    s = Scheduler(["BTC/USDT"])
    s._running = True
    s.stop()
    assert not s._running


def test_scheduler_status():
    s = Scheduler(["BTC/USDT", "ETH/USDT"])
    assert "BTC/USDT" in s.status
    assert "ETH/USDT" in s.status
