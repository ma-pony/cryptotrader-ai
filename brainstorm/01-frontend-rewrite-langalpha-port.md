# Brainstorm 01 — 前端重写 · LangAlpha 移植 + Crypto 化

**Date**: 2026-04-16
**Feature**: `001-frontend-rewrite-langalpha-port`
**Artifact**: [specs/001-frontend-rewrite-langalpha-port/spec.md](../specs/001-frontend-rewrite-langalpha-port/spec.md)
**Trigger**: `/spex:brainstorm 完全弃用 Streamlit dashboard，照搬 LangAlpha React+Vite+TS 前端，crypto 化所有股票语义`

---

## 目标

把 Streamlit 做的 5 大业务页面全部换成 React 19 + Vite 7 + TS 5.9 的现代 SPA，参考 LangAlpha (https://github.com/ginlix-ai/LangAlpha) 的前端架构，同时把股票语义（SEC/EDGAR/财报）全面替换为加密货币语义（BTC/USDT、资金费率、OI、链上数据）。

## 关键演进（增量讨论 7 节）

### Section 1 — 范围定义
初版想法"完全弃用 Streamlit"用户修正为："系统本质是自动交易 + 回测，Streamlit 5 大业务页面必须 1:1 保留；换的是技术栈，不是功能"。

### Section 2 — 技术栈选型
- 照搬 LangAlpha：React 19.2 + Vite 7 + TS 5.9 strict + pnpm + Tailwind 3 + Radix UI + React Query 5 + Zustand + lightweight-charts 4 + TradingView Widget
- 中英文策略争议：先定"尽可能中文"，后修正为"实现 i18next，zh-CN 默认 + en-US 可选"

### Section 3 — 页面到 FR 映射
- P1：Dashboard / Decisions / Backtest / Risk / Metrics（对齐 Streamlit 5 页）
- P2：ChatAgent / MarketView（新增，保留 LangAlpha 的多代理会话 + 行情图表能力）
- FR 分段：001-019 脚手架 / 100-119 Dashboard / 200-229 Decisions / 300-329 Backtest / 400-409 Risk / 500-509 Metrics / 600-619 ChatAgent / 700-719 MarketView / 800-829 后端 API / 900-915 Streamlit 删除

### Section 4 — 非功能约束
- 性能：首屏 ≤ 2s，主 bundle gzipped ≤ 300KB，权益曲线 1k 点 ≤ 200ms
- 安全：X-API-Key、iframe sandbox、react-markdown + rehype-sanitize、nginx security headers
- 可访问性：WCAG AA、键盘导航、aria-label
- 可维护性：TS strict、ESLint 零警告、`useChatMessages` ≤ 500 行硬限

### Section 5 — 实施分期（11 阶段）
Phase 0 脚手架 → 1 框架 → 2 后端补齐 → 3-7 五大 P1 页面 → 8 e2e + Streamlit 一次性删除 → 9 ChatAgent → 10 MarketView → 11 部署文档

### Section 6 — spex 工作流决策
启用 traits：superpowers + deep-review + teams + worktrees；先 `/spex:brainstorm` → `/speckit-specify` → `/speckit-plan` → `/speckit-implement`，中间插 `/spex:review-spec` 与 `/spex:review-plan`。

### Section 7 — Streamlit 删除硬门槛
用户强化指令："最终所有 Streamlit 代码都需要删除"——新增 FR-908 ~ FR-915，落实到 4 条 `rg` 终态校验命令：

```bash
rg -i streamlit src/ tests/ scripts/   # 必须 0 命中
rg -i streamlit pyproject.toml docker-compose.yml Dockerfile   # 必须 0 命中
rg -i 'src/dashboard' src/ tests/ docs/   # 必须 0 命中
rg -i ':8501' .   # 必须 0 命中（Streamlit 默认端口）
```

`rg` 0 命中是合并 PR 的硬门槛（D-14）。

## 关键修正点（来自用户反馈）

| 修正 | 原计划 | 修正后 |
|------|--------|--------|
| 范围 | 完全弃用 Streamlit，不考虑现有功能 | **1:1 保留 5 大业务页面** |
| 语言 | 纯中文，不做 i18n | i18next 双语，zh-CN 默认 |
| 子代理 | 取消子代理（D-6） | **允许子代理**，依赖 worktrees 防冲突 |
| 删除力度 | 删 `src/dashboard/` 就行 | **物理删除 + `rg` 0 命中校验** |

## 产出

- `specs/001-frontend-rewrite-langalpha-port/spec.md` — 完整规格（~22KB，136 个 FR 引用）
- `specs/001-frontend-rewrite-langalpha-port/checklists/requirements.md` — 质量门槛自检（全部通过）
- `specs/001-frontend-rewrite-langalpha-port/review_brief.md` — 审阅简报
- `.specify/feature.json` — 绑定 feature_directory 供 `/speckit-plan` 等后续命令使用
- 新增 spex/spec-kit 工具链（14 skill + .specify/ 脚手架）
- 分支 `001-frontend-rewrite-langalpha-port` 含 3 个 commit

## 提交记录

```
6d30270 feat(spec): add frontend-rewrite-langalpha-port spec
8bb559c chore: install spex / spec-kit tooling
a5d5aed feat: backtest LLM isolation, prompt hardening, and risk checks refactor
```

## 下一步

- 继续：`/speckit-plan` 生成实施计划，然后 `/spex:review-plan` 评审
- 或：`/speckit-clarify` 补充澄清
- PR 前必须跑完三大闸门：`/spex:review-spec` + `/spex:review-plan` + `/speckit-clarify`

## 经验回顾

1. **用户提前批准可加速收敛**：7 节增量讨论每节用户单独"批准"，避免了最后大返工
2. **"硬门槛"是 spec 最有价值的部分**：FR-915 的 `rg` 命令把"Streamlit 已删除"从主观判断变成可自动校验
3. **worktree 创建前要先看工作树状态**：本次差点把 21 个无关 WIP 扫进 spec commit，幸好提前检测到
4. **pre-commit hook 会改动文件**：detect-secrets 对 sha256 hash 误报、end-of-file-fixer 会自动改行尾——提交后要再 `git add -u` 重提
