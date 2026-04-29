# Frontend Deep Review Report · frontend-prototype-alignment

**Date**: 2026-04-24
**Scope**: 本会话 13+ 前端文件（新增 Debate 页 + 决策详情 8 节 + Risk/Metrics/Dashboard 组件 + 共享 UI 原子）
**Agents**: Security+A11y · Performance · Architecture+Correctness
**Auto-fix**: disabled（只出报告）
**Status**: ⚠️ **25 问题** — 3 Critical / 14 Important / 8 Minor

---

## 执行摘要

Critical 3 项全部与**稳定性/安全**相关：一个 iframe sandbox 逃逸（inline-widget 用 `doc.write`），两个 1秒 setInterval 产生持续渲染风暴。Important 层面最突出的是**shape drift**：3 处独立的 AgentKind 调色板定义、2 处 DirChip、score-to-direction 阈值不一致。

**跨 agent 重合**：
- **DirChip 双实现**（Architecture Important）+ 本地 MAP 漏 buy/sell（Architecture Important）→ 同一 bug 两视角
- **setInterval 泛滥**（Performance Critical × 2）→ 1/s 全局 + 1/s 页面，同时跑
- **palette/kind 重复**（Architecture Important × 2）→ 3 处独立源

---

## 问题汇总

| 维度 | Critical | Important | Minor | 合计 |
|---|---|---|---|---|
| 🔒 Security + A11y | 1 | 4 | 3 | 8 |
| ⚡ Performance | 2 | 5 | 2 | 9 |
| 🏗️ Architecture + Correctness | 0 | 5 | 3 | 8 |
| **合计** | **3** | **14** | **8** | **25** |

---

## 🔴 Critical 问题（3 个 — 立即修）

### FE-C1 · iframe sandbox 逃逸（XSS）
**文件**: `src/components/inline-widget/inline-widget.tsx:19`
后端 SSE `inline_widgets[].html` 通过 `iframe.contentDocument.write(html)` 注入 — sandbox `allow-scripts` **不** 防护父 JS 上下文写入，脚本在父 origin 执行。
**Fix**: 改用 `<iframe srcdoc={sanitized}>` 或 `URL.createObjectURL(Blob)`，绝不用 `contentDocument.write`。

### FE-C2 · Sidebar setInterval(1s) 全局渲染风暴
**文件**: `src/components/layout/sidebar.tsx:63`
`SidebarFooter` 每秒 `tick((n)=>n+1)` → 整个侧栏 + 9 个 NavLink + footer 每秒重渲 → 会话生命周期内不停。每进入任何页面都吃。
**Fix**: 单一共享 countdown hook；SidebarFooter 精度降到 60s 即可。

### FE-C3 · SchedulerCard setInterval(1s) 页面渲染风暴
**文件**: `src/pages/dashboard/components/scheduler-card.tsx:34`
Dashboard 可见时，与 FE-C2 叠加 = **每秒 2 次独立渲染级联**。月累积上万次无意义 DOM diff。
**Fix**: 与 FE-C2 合并到同一个 `useCountdown` hook。

---

## 🟠 Important 问题（14 个）

### 安全 + A11y（4）
- **FE-I1** `RiskMeter` 进度条无 `role="meter"` / `aria-valuenow`（risk-meter.tsx:44）
- **FE-I2** `decisions-table` `<button aria-selected>` 语义无效（decisions-table.tsx:37）→ 改 `aria-pressed` 或用 `role="grid"` 重构
- **FE-I3** `equity-chart-section` tab 组缺 `aria-controls` + `tabpanel` 伴随元素
- **FE-I4** `text-[10px] text-muted-foreground` 对比度 ~3.4:1 低于 WCAG AA 要求 4.5:1（多处）

### 性能（5）
- **FE-I5** `SectionNav` 滚动监听无 debounce（section-nav.tsx:22）→ 快滚 60次/秒 setState
- **FE-I6** `Sparkline` 路径无 useMemo（sparkline.tsx:21）→ positions 列表 WS tick 每次重算 N 条
- **FE-I7** `toScenario` 在 render body（debate/index.tsx:175）→ 每次父重渲都重建 scenario
- **FE-I8** `AssistantBubble` / `UserBubble` 无 React.memo（message-stream.tsx:53）→ 流式 50 token 期间旧消息重解析 markdown
- **FE-I9** `EquityChart` theme 切换销毁重建（equity-chart.tsx:88）→ 应走 `applyOptions` 增量

### 架构 + 正确性（5）
- **FE-I10** 重复 `DirChip` — debate 本地版 MAP 漏 `buy` / `sell` → verdict.action=`buy` 静默显示 `观望`（dir-chip.tsx:5）
- **FE-I11** Score→direction 阈值不一致：`toScenario` 用 ±0.1，`directionFromScore` 用 ±0.3 → 同一 score 在 Debate 页/详情页显示不同方向
- **FE-I12** `afterDispersion` 默认 0 导致单-turn 轮次 DivergenceMeter **假完美收敛**（debate/index.tsx:90）
- **FE-I13** `isVerdict` regex 第二分支 `/\bverdict\b/` 前 120 字符内都命中 → 非 verdict 消息被误打琥珀辉光
- **FE-I14** 3 处独立 AgentKind 调色板（debate/constants + agent-badge + agent-analysis-grid）→ `agent-analysis-grid` 甚至缺 `verdict` kind

---

## 🟡 Minor 问题（8 个）

### 安全
- **FE-m1** `decision-detail-panel` 两处 `<a href>` 指向内部路由，应用 `<Link>`（SPA 状态丢失）
- **FE-m2** `tradingview-chart.tsx:44` 用 `innerHTML` 塞 i18n 文本（潜在）
- **FE-m3** `DebateTurnCard` div 无 `tabIndex={0}` 不可键盘聚焦

### 性能
- **FE-m4** `PositionRow` WS 订阅依赖 positions 数组引用稳定性；若 refetch 重建引用会断订阅
- **FE-m5** vendor monochunk 187KB gzipped；拆 `lucide-react` / `i18next` / `@tanstack` 可降首屏成本

### 架构
- **FE-m6** `RiskMeter` null 值渲染 0px 绿条 + `invertTone` 死代码
- **FE-m7** `risk/index.tsx` `checksOnline={11}` 硬编码，应来自 API
- **FE-m8** `DebatePage` 无内层 ErrorBoundary，错误冒泡到 AppShell 级导致侧栏坍塌

---

## 修复优先级

**Phase F1（Critical · 立即修，<1h）**：
1. FE-C1 iframe srcdoc 改造（XSS）
2. FE-C2 + FE-C3 合并到共享 `useCountdown` hook

**Phase F2（Important · 半天）**：
3. FE-I10 删除 debate 本地 DirChip，统一用 `ui/dir-chip`
4. FE-I11 抽 `scoreToDirection(score)` 到 `lib/format.ts`
5. FE-I12 `afterDispersion` 默认值修正
6. FE-I13 移除 isVerdict 宽松 fallback
7. FE-I14 抽 `lib/agents.ts` 作为 palette 单点
8. FE-I5 SectionNav debounce 60ms
9. FE-I6 Sparkline `useMemo`
10. FE-I7 DebatePage toScenario `useMemo`
11. FE-I8 AssistantBubble/UserBubble `React.memo`

**Phase F3（Accessibility · 1-2h）**：
12. FE-I1 RiskMeter ARIA
13. FE-I2 decisions-table aria-pressed
14. FE-I3 equity tab panel 补全
15. FE-I4 10px 对比度或字号调整

**Phase F4（Polish · 按需）**：
16. Minor 8 项清理

---

## 报告路径

`.kiro/specs/frontend-prototype-alignment/DEEP_REVIEW_FRONTEND.md`

**下一步建议**：先修 Phase F1 + F2（影响链路完整性与 UX 一致性），F3/F4 可后续处理。
