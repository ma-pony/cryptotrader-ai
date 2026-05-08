---
name: autoresearch
url: https://github.com/uditgoenka/autoresearch
license: MIT
tier: 3
last_accessed: 2026-05-08
phase_1_complete: true
phase_2_complete: false
---

# autoresearch — uditgoenka

## Architecture Overview

autoresearch 是一个自主迭代框架，将 Karpathy 的 autoresearch 理念推广到任意领域。其核心是一个无界的 modify-verify-keep/discard 循环：agent 修改代码，运行机械化指标验证，如果指标改善则 commit，否则 `git revert`，然后继续下一轮。设计面向 Claude Code、OpenCode 和 OpenAI Codex 三个平台（`.claude/`、`.opencode/`、`.agents/` 各有一套适配）。

技术栈以 Shell（62%）和 Python（38%）为主。持久化完全依赖 git 历史——没有向量数据库、没有外部记忆层；每次迭代都通过读取 `git log` + `git diff` 来重建上下文。

## Prompt Assembly（Phase 1）

autoresearch 的"提示组装"是隐式的、基于 git 的：每次迭代开始前，agent 执行 **read-before-write** 协议——先扫描全代码库上下文与 git 历史，再生成修改。"提示"等于 `[当前目标/指标定义] + [git log 摘要] + [当前 diff]`，没有独立的提示模板文件或动态组装层。

框架提供 11 个专项子命令（`/autoresearch:plan`、`:debug`、`:fix`、`:security`、`:ship`、`:learn`、`:predict`、`:reason`、`:scenario`、`:probe`），每个子命令对应一个 `.md` skill 文件，作为独立的提示规程注入 agent。但各 skill 之间无显式的编排层——它们是并列的入口点，不是流水线阶段。

## Memory ↔ Skill（Phase 1 lite）

N/A — autoresearch 不具备可被检索的结构化记忆，也没有"技能进化"机制。

- **记忆**：完全外包给 git。每次实验以 `experiment:` 前缀 commit；失败实验虽被 revert，但保留在历史中供未来迭代参考。这是一种**只追加、无检索**的记忆——agent 靠自身上下文窗口读取 log，而非向量检索。
- **技能**：skill 文件是静态 Markdown，不会随迭代学习而更新；不存在技能评分、淘汰或版本演化机制。
- **结论**：此项目与 spec 016 的"技能进化"主题仅有表面关联（都有"迭代改进"的概念），但实现路径完全不同。

## Phase 2 Placeholders

- Evolution Algorithm
- Skill Data Structure
- Retrieval
- Evaluation
- Agent ↔ Skill Boundary
- Engineering

## Borrow Recommendations（Phase 1）

1. **Git 作为零依赖审计日志**：`experiment:` commit 前缀 + 失败保留策略是一种极简的迭代溯源方案。cryptotrader-ai 的 backtest session 记录可借鉴该前缀规范，将每次策略变更与 git commit 绑定，无需额外存储。

2. **机械化指标作为迭代终止条件**：autoresearch 要求用户在启动前定义"可机械验证"的目标指标，这与 spec 016 的技能评估需求同构——可将该约束作为 spec 016 任务定义规范的参考，要求每个技能升级附带可量化的验收标准。

## Notes / Open Questions

- 此项目的核心价值在于**循环架构**（modify-verify-keep/discard），而非提示工程或记忆系统。与 spec 016 研究集中的 hermes、openclaw-rl、skillclaw 相比，autoresearch 缺少：向量检索、技能表示层、奖励建模、记忆分级。
- README 的 raw 内容无法直接访问（404），skill 文件同样无法直接读取，故提示模板的具体格式未能核实——上述分析基于仓库首页的渲染内容。
- 如需替换：可考虑 **AutoGen（microsoft/autogen）** 的 `AssistantAgent` + `UserProxyAgent` 配对，或 **OpenHands/SWE-agent**，两者都有更完整的提示组装与状态追踪机制，与 spec 016 的研究焦点更契合。
