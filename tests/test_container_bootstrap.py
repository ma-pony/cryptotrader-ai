"""容器镜像必须能在断网首次启动时装配真实 registry。"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
IMAGE_TAG = "cryptotrader-task20-bootstrap-check"
MASTER_KEY = "A" * 43 + "="
_BOOTSTRAP_CHECK = """
import asyncio
import json

from cryptotrader.runtime import build_runtime


async def main():
    runtime = await build_runtime()
    try:
        print(json.dumps({
            "automation_enabled": runtime.snapshot.document.scheduler.automation_enabled,
            "venue_sessions": len(runtime.sessions),
            "cycle_is_none": runtime.cycle is None,
        }))
    finally:
        await runtime.close()


asyncio.run(main())
"""


@pytest.mark.skipif(
    os.environ.get("RUN_CONTAINER_BOOTSTRAP_CHECK") != "1",
    reason="set RUN_CONTAINER_BOOTSTRAP_CHECK=1 to run the Docker bootstrap gate",
)
def test_runtime_image_bootstraps_setup_state_offline_with_real_registry():
    """删掉提示词资源或跳过 registry discovery 会使首次容器启动失败。"""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is unavailable")

    build = subprocess.run(
        ["docker", "build", "--tag", IMAGE_TAG, "."],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr

    runtime = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--env",
            "DATABASE_URL=sqlite+aiosqlite:///:memory:",
            "--env",
            f"CONFIG_MASTER_KEY={MASTER_KEY}",
            IMAGE_TAG,
            "python",
            "-c",
            _BOOTSTRAP_CHECK,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert runtime.returncode == 0, runtime.stderr
    assert runtime.stdout.strip() == ('{"automation_enabled": false, "venue_sessions": 0, "cycle_is_none": true}')
