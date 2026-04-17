# Review Brief — 前端重写 · LangAlpha 移植 + Crypto 化

**Feature**: `001-frontend-rewrite-langalpha-port`
**Spec**: [spec.md](./spec.md)
**Created**: 2026-04-16
**Source**: `/spex:brainstorm` 7 节增量讨论（每节用户批准）

---

## 一句话摘要

彻底弃用 Streamlit dashboard 的技术栈，照搬 LangAlpha 的 React 19 + Vite 7 + TypeScript 5.9 前端架构，**1:1 保留** Streamlit 5 大业务页面的全部功能，同时把所有股票语义迁移为加密货币语义。

## 为什么做

- Streamlit 已成为 UI/UX 天花板（交互单向、缺少现代前端观测性、无法定制、重渲染开销大）
- 自动交易系统的监控体验需要**可组合、可观测、可扩展**的专业前端
- LangAlpha 已完整验证 React 19 + streamFetch SSE + InlineWidget iframe 沙盒的多代理前端栈，选择性移植比从零搭建更稳
- Streamlit 留下零碎代码会污染后续架构（Docker service、CLI 命令、文档、配置）——**一次性物理删除**是硬门槛

## 范围与优先级

| 业务页面                  | 优先级 | 对应 Streamlit 页         | FR 范围        |
| ------------------------- | ------ | ------------------------- | -------------- |
| Dashboard（总览）         | P1     | Overview                  | FR-100 ~ FR-119 |
| Decisions（实时决策）     | P1     | Live Decisions            | FR-200 ~ FR-229 |
| Backtest（回测）          | P1     | Backtest                  | FR-300 ~ FR-329 |
| Risk（风控状态）          | P1     | Risk Status               | FR-400 ~ FR-409 |
| Metrics（指标）           | P1     | Metrics                   | FR-500 ~ FR-509 |
| ChatAgent（多代理会话）   | P2     | —（新增）                  | FR-600 ~ FR-619 |
| MarketView（行情）         | P2     | —（新增）                  | FR-700 ~ FR-719 |

此外：
- **FR-001 ~ FR-019**：脚手架（pnpm/Vite/TS strict/Tailwind/i18n/React Query/Zustand/streamFetch）
- **FR-800 ~ FR-829**：后端 FastAPI endpoints 按需补齐
- **FR-900 ~ FR-915**：Streamlit 完全物理删除 + `rg -i streamlit 0 命中` 硬校验

## 关键决策（经用户逐条批准）

| # | 决策 | 为什么 |
|---|------|--------|
| D-1 | 选择性移植 LangAlpha，不 git clone 不 submodule | 保持项目自洽，避免 LangAlpha 的股票包袱 |
| D-2 | 前端先行、后端按需补 endpoint | 前端可独立开发演进；后端只补必需接口 |
| D-3 | 完全 crypto 化，删除 SEC/EDGAR/财报相关 | 语义一致性，避免误导 |
| D-4 | Streamlit 5 大页面 1:1 保留 | 自动交易 + 回测是项目本质，不能功能倒退 |
| D-5 | i18next：zh-CN 默认，en-US 可选 | 面向中文用户优先，英文为可选 |
| D-6 | 启用 spex 工作流（superpowers + deep-review + teams + worktrees） | 质量闸门 + 并行加速 |
| D-7 | React Query + Zustand，不用 Redux | 现代轻量，与 LangAlpha 一致 |
| D-8 | streamFetch SSE 客户端，不用 EventSource | 可处理 429/413/404 + header 注入 |
| D-9 | Backtest 长任务用 5s 轮询，不用 SSE | 任务可能 10 min+，连接保活复杂 |
| D-10 | 不做 SSR/PWA/移动端 | 定位是运维 + 研究桌面工具 |
| D-11 | Streamlit 删除一次性 PR + e2e 回归 gate | 一次性切干净，避免遗留 |
| D-12 | 测试少 mock，优先 docker compose 真实栈 | 与项目"最少 mock"长期约束一致 |
| D-13 | **Streamlit 100% 物理删除**，`rg -i streamlit` 必须 0 命中 | 硬门槛；任何历史代码路径必须消失 |
| D-14 | `rg` 4 条终态校验命令，合并 PR 的硬校验 | FR-915 可执行的终态凭证 |

## 审阅者重点检查项

### 1. Content Quality
- [ ] 5 个 P1 user story 的业务描述是否准确描述运维/研究员的真实工作流？
- [ ] 2 个 P2 user story（ChatAgent / MarketView）是否应降级为 P3 或延期？
- [ ] 中英文混用是否恰当（业务术语 vs 技术术语）？

### 2. Requirement Completeness
- [ ] **FR-915 的 4 条 `rg` 命令**是否覆盖所有可能的 Streamlit 残留（代码、配置、脚本、文档、Docker、CI、CLI）？是否需要补充 `:8501` 以外的端口？
- [ ] **后端 endpoints（FR-800 ~ FR-829）** 是否实际存在？哪些需要新增、哪些需要改造？（会在 `/speckit-plan` 阶段逐一核查）
- [ ] **SC-006（主 bundle gzipped ≤ 300KB）** 是否在 React 19 + Radix + lightweight-charts + TradingView Widget 的体量下可行？是否需要动态 import 才能达标？

### 3. 风险与权衡
- [ ] **R-11（spex:teams + worktrees 多 agent 并行写文件冲突）** 是否已充分降级？
- [ ] **Streamlit 删除 PR 的回滚策略**：如果删除后发现某个隐藏依赖（cron、监控脚本），如何快速回滚？
- [ ] **LangAlpha streamFetch 的 429/413 行为**在我们的 FastAPI 后端上是否完全对齐？

### 4. 非功能需求硬指标
- [ ] 首屏 ≤ 2s：在 1k 决策列表 + 100 持仓的真实数据下是否达标？
- [ ] 权益曲线 1k 点 ≤ 200ms：lightweight-charts 实测是否通过？
- [ ] Lighthouse Performance ≥ 90：是否在 docker compose 完整栈下测量？

### 5. Out-of-Scope 红线
- [ ] SSR（Next.js）——**明确不做**，reviewer 如提议请拒绝
- [ ] PWA/离线——**明确不做**
- [ ] 移动端/响应式 < 1280px——**明确不做**
- [ ] 登录/多用户系统——**明确不做**（项目仍是单机自部署工具）

## Spec 完整性自检结果

| 项                               | 状态  |
| -------------------------------- | ----- |
| No [NEEDS CLARIFICATION] markers | ✅ 0 |
| FRs 具体可测试                   | ✅    |
| SCs 可度量（含时间/大小/命中数） | ✅    |
| Acceptance scenarios ≥ 4 per P1  | ✅    |
| Edge cases                       | ✅ 12 |
| Assumptions + Dependencies       | ✅ 14 + 5 |
| 终态硬门槛（FR-915 `rg`）        | ✅ 可执行 |

## 下一步

1. `/speckit-plan` — 生成实施计划（推荐）
2. `/speckit-clarify` — 如有仍需澄清的点
3. `/speckit-checklist` — 生成更细的校验清单
4. `/spex:review-spec` — 额外的多视角 spec 评审

## 相关工件

- [spec.md](./spec.md) — 完整规格
- [checklists/requirements.md](./checklists/requirements.md) — 质量门槛自检
- [../../brainstorm/01-frontend-rewrite-langalpha-port.md](../../brainstorm/01-frontend-rewrite-langalpha-port.md) — 会话总结
