"""原子写入 + 目录初始化工具。

FR-013: 原子写（temp + os.rename）+ threading.Lock 进程内排他锁。
"""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

# 进程内全局写锁（单进程 single-writer 模型）
_write_lock = threading.Lock()


def atomic_write(path: Path, content: str) -> None:
    """原子写入文件（临时文件 + os.rename）。

    同一进程内通过 _write_lock 排他，防止并发写入冲突。
    跨进程扩展：未来可升级为 fcntl.flock。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.rename(tmp_path, path)
        except Exception:
            # 清理残留临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


def ensure_memory_dirs(base: Path | None = None) -> None:
    """自动创建 agent_memory/ 目录骨架。

    FR-003: cases/ + 4 个 agent 子目录（patterns/ + archive/）。
    """
    from cryptotrader.agents.skills._constants import DEFAULT_AGENT_MEMORY_DIR, VALID_AGENT_IDS

    memory_dir = base or DEFAULT_AGENT_MEMORY_DIR
    # 顶级 cases 目录
    (memory_dir / "cases").mkdir(parents=True, exist_ok=True)
    # 4 个 agent 子目录
    for agent_id in VALID_AGENT_IDS:
        (memory_dir / agent_id / "patterns").mkdir(parents=True, exist_ok=True)
        (memory_dir / agent_id / "archive").mkdir(parents=True, exist_ok=True)


def ensure_skill_dirs(base: Path | None = None) -> None:
    """自动创建 agent_skills/ 目录骨架（initial 5 个 skill 目录）。"""
    from cryptotrader.agents.skills._constants import _INITIAL_SKILL_DIRS, DEFAULT_AGENT_SKILLS_DIR

    skills_dir = base or DEFAULT_AGENT_SKILLS_DIR
    for name in _INITIAL_SKILL_DIRS:
        (skills_dir / name).mkdir(parents=True, exist_ok=True)


def atomic_rename(src: Path, dst: Path) -> None:
    """原子 rename（archive 操作用）。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    with _write_lock:
        os.rename(src, dst)
