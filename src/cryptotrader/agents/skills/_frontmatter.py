"""YAML frontmatter 解析与校验工具。

支持 SKILL.md / pattern_record.md / case_record.md 三种格式的解析。
解析失败抛 CorruptFrontmatterError 含路径与行号。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class CorruptFrontmatterError(ValueError):
    """YAML frontmatter 解析失败或校验不通过。"""

    def __init__(self, msg: str, path: Path | None = None, line: int | None = None) -> None:
        detail = msg
        if path:
            detail = f"{path}: {msg}"
        if line is not None:
            detail = f"{detail} (line {line})"
        super().__init__(detail)
        self.path = path
        self.line = line


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)", re.DOTALL)


def parse_frontmatter(text: str, path: Path | None = None) -> tuple[dict[str, Any], str]:
    """从 markdown 文件内容中提取 YAML frontmatter 与 body。

    Returns:
        (frontmatter_dict, body_str)

    Raises:
        CorruptFrontmatterError: frontmatter 缺失或 YAML 解析失败
    """
    import yaml

    m = _FRONTMATTER_RE.match(text)
    if not m:
        raise CorruptFrontmatterError("缺少 YAML frontmatter（需以 '---' 开头）", path=path, line=1)

    yaml_str = m.group(1)
    body = m.group(2)

    try:
        data = yaml.safe_load(yaml_str)
    except yaml.YAMLError as exc:
        line = None
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            line = exc.problem_mark.line + 2  # +1 for '---' line, +1 for 1-indexed
        raise CorruptFrontmatterError(f"YAML 解析失败: {exc}", path=path, line=line) from exc

    if not isinstance(data, dict):
        raise CorruptFrontmatterError("frontmatter 必须是 YAML mapping", path=path, line=1)

    return data, body


def validate_skill_frontmatter(data: dict[str, Any], path: Path | None = None) -> None:
    """校验 SKILL.md frontmatter 必填字段。

    Required: name, description, scope
    """
    required = ["name", "description", "scope"]
    for field in required:
        if not data.get(field):
            raise CorruptFrontmatterError(f"SKILL.md frontmatter 缺少必填字段: '{field}'", path=path)

    name = data["name"]
    if not re.match(r"^[a-z][a-z0-9-]*$", str(name)):
        raise CorruptFrontmatterError(f"SKILL.md name 格式不合规（必须 kebab-case）: '{name}'", path=path)

    scope = data["scope"]
    if scope != "shared" and not re.match(r"^agent:(tech|chain|news|macro)$", str(scope)):
        raise CorruptFrontmatterError(
            f"SKILL.md scope 格式不合规: '{scope}'（应为 'shared' 或 'agent:<id>'）", path=path
        )


def validate_pattern_frontmatter(data: dict[str, Any], path: Path | None = None) -> None:
    """校验 pattern_record.md frontmatter 必填字段。"""
    required = ["name", "agent", "maturity"]
    for field in required:
        if not data.get(field):
            raise CorruptFrontmatterError(f"pattern frontmatter 缺少必填字段: '{field}'", path=path)


def validate_case_frontmatter(data: dict[str, Any], path: Path | None = None) -> None:
    """校验 case_record.md frontmatter 必填字段。"""
    required = ["cycle_id", "pair", "verdict_action"]
    for field in required:
        if not data.get(field):
            raise CorruptFrontmatterError(f"case frontmatter 缺少必填字段: '{field}'", path=path)


def render_frontmatter(data: dict[str, Any]) -> str:
    """将 dict 渲染为 YAML frontmatter 块（含 --- 分隔符）。"""
    import yaml

    yaml_str = yaml.dump(data, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return f"---\n{yaml_str}---\n"
