"""全局 SignalProfile 的单行持久化与 revision 契约。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.factories.signal_fusion import profile


@pytest.mark.asyncio
async def test_repository_creates_default_and_increments_revision(tmp_path):
    from cryptotrader.profiles.repository import SignalProfileRepository

    url = f"sqlite+aiosqlite:///{tmp_path / 'profile.db'}"
    repository = SignalProfileRepository(url)

    first = await repository.get_or_create(profile(revision=99))
    second = await repository.replace(replace(first, neutral_threshold=0.3))

    assert first.revision == 1
    assert second.revision == 2
    assert (await repository.get()).neutral_threshold == 0.3


@pytest.mark.asyncio
async def test_repository_round_trips_all_component_and_policy_fields(tmp_path):
    from cryptotrader.profiles.repository import SignalProfileRepository

    url = f"sqlite+aiosqlite:///{tmp_path / 'profile.db'}"
    repository = SignalProfileRepository(url)
    expected = profile(
        revision=1,
        kronos=0.7,
        llm=0.3,
        neutral_threshold=0.25,
        max_target_ratio=0.8,
        atr_stop_multiplier=2.5,
        reward_ratio=1.8,
        hitl=True,
    )

    assert await repository.get_or_create(expected) == expected
    assert await SignalProfileRepository(url).get() == expected


@pytest.mark.asyncio
async def test_replace_requires_existing_global_profile(tmp_path):
    from cryptotrader.profiles.repository import SignalProfileRepository

    repository = SignalProfileRepository(f"sqlite+aiosqlite:///{tmp_path / 'profile.db'}")

    with pytest.raises(LookupError, match="global"):
        await repository.replace(profile())
