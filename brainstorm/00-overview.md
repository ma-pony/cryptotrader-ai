# Brainstorm Overview

Last updated: 2026-05-08

本目录记录 `/spex:brainstorm` 会话的结构化总结，按时间顺序编号。每个会话对应一个 spec（通常也对应一个 feature 分支）。

## 索引

| #   | 日期       | 主题                                          | Spec 目录                                                                 | 状态         |
| --- | ---------- | --------------------------------------------- | ------------------------------------------------------------------------- | ------------ |
| 01  | 2026-04-16 | 前端重写 · LangAlpha 移植 + Crypto 化         | [specs/001-frontend-rewrite-langalpha-port/](../specs/001-frontend-rewrite-langalpha-port/) | Spec Ready，待 `/speckit-plan` |
| 02  | 2026-05-08 | Skill / Memory 进化前序研究（8 项目）         | [specs/016-research-skill-evolution-prior-art/](../specs/016-research-skill-evolution-prior-art/) | Spec ✅ SOUND，Phase 1+2 完成；blocks 017/018 |
| 03  | 2026-05-08 | Spec 017b — Agent Prompt Builder Integration | （待 `/speckit-specify` 创建）                                            | Active brainstorm；6 决策已敲定，待 ship |

## 进行中的依赖链

```
spec 016 Phase 1 → unblocks → spec 017a brainstorm（PromptBuilder 基建，已合并 main）
spec 016 Phase 2 → unblocks → spec 018 brainstorm（skill 进化 v2，待启动）
spec 017a → unblocks → spec 017b brainstorm（4 agent 集成切换，本会话）
```

## 已合并 spec

- **spec 016**：8 项目研究（commit 217b906 等）
- **spec 017a**：PromptBuilder 基建（commit cfd3acc + merge f1e37a9，2026-05-08）

## Open Threads

（暂无 — spec 017b 的 4 项 open threads 在 2026-05-08 spot-check 后均已解决，详见 [#03](03-spec-017b-prompt-builder-integration.md) "Open Threads（已 spot-check 解决）"）

## Parked Ideas

- **autoresearch 项目**（来自 #02）—— 该项目仅含 README，无实际代码可分析；研究价值低，已替换为 microsoft/autogen 项目（#02 Phase 2 已落地）

## 约定

- 每个 session 存为 `NN-<short-name>.md`，NN 与对应的 spec 编号大致对应（split spec 如 017a/017b 使用同一 NN）
- 会话摘要应包含：目标、关键决策、approaches considered、open threads
- spec 的正式内容在 `specs/<NN>-<short-name>/spec.md`，brainstorm 目录只保留过程总结
- 审阅简报放在对应的 spec 目录 `review_brief.md`，不在此目录
