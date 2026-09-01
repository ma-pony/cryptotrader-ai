# CryptoTrader trading workbench

The product is a modern-minimal, utilitarian operator console. Its macrostructure is a Workbench: one six-item side rail and compact operational surfaces that answer “what is true now, and what is the safe next action?” It keeps the system font stack, semantic HSL tokens, CryptoTrader amber/cyan/trade tokens, Tailwind 3 and both themes.

## Application structure

- The only main entries are 工作台, 决策, 引擎, 账户, 研究 and 系统. Desktop and mobile render the same `SidebarDrawerBody` navigation source.
- 工作台 orders runtime facts and explicit actions before attention items, recent decisions and simulated/real account facts.
- 决策 owns history and decision evidence. Internal LLM debate remains evidence within a decision, not a separate product area.
- 引擎 owns market/context, peer signal components, fusion/risk and automation sources.
- 账户 owns connections, books, account facts, history coverage and income timing.
- 研究 owns market observation, pure analysis and backtests. 系统 owns models, notifications, security, runtime metrics and real Skills/access records.
- Nested detail routes link to the canonical object. Business facts are not copied into a second implementation.

## Forms

- Compact Chinese business headings, with English translations. Two-column desktop controls collapse to one column on narrow screens.
- 14px labels/help, 40px desktop controls and 44px mobile targets. Visible instant focus rings and field-linked errors.
- Section dividers instead of nested cards. Native advanced disclosures retain their children and draft values.
- Numeric drafts can remain empty until corrected. Ratios use explicit percentage labels and convert only at the field boundary.
- Per-section saves show dirty, saving, saved and failed state. Runtime application is distinct from persistence; saving never activates trading.
- Catalog choices replace plugin identifier entry. Credentials have separate write-only inputs and actions.

## Operational surfaces

- Body copy is at least 14px. Desktop actions are at least 40px high; touch actions are at least 44px. Clickable labels remain on one line.
- Dense facts use bordered rows and section dividers. Cards are reserved for a real grouped state or action, never nested for decoration.
- Loading, empty, failed, disabled and successful states say what is known and the next safe action. A failed refresh does not reuse an old green state.
- Missing account values remain unavailable; they are never rendered as zero, profit or loss. Money is shown in its original currency and time-based facts state their `as_of` or covered interval.
- “自动运行已开启” describes the master switch only. Active tasks and enabled schedule/trigger sources require their own backend facts.
- Read operations and ordinary saves never trigger analysis or trading. Only explicit action buttons may do so.
- Narrow data tables own horizontal overflow. Focus-visible rings use the semantic ring token. Both themes retain equivalent hierarchy and contrast.

## Interaction voice

- Chinese labels are practical and specific; platform brands and model names retain their official spelling.
- No marketing hero, footer, decorative gradient, glass surface, invented metric, celebratory motion or parameter JSON editor.
- Silent success is preferred. Errors stay attached to their fields; CAS conflicts retain input and offer reload guidance.

## Verified responsive and focus standard

- At 1440px the permanent side rail and top bar frame a centered content column capped at 76rem. Operational facts, result blocks and tables keep their full hierarchy in both light and dark themes; the page owns vertical scrolling while only an individual wide table may own horizontal scrolling.
- Below 768px the permanent rail becomes a drawer. At the 390px acceptance width, content uses 16px page gutters, configuration grids become one column, action rows wrap, and controls use the 44px mobile height. The document must not become wider than the viewport.
- Body, label and help text remain at least 14px. Status is never conveyed by color alone; semantic borders, text and explicit Chinese labels stay visible in both themes.
- Save actions are keyboard reachable and Enter submits only the current explicit form. Non-destructive approval confirmation initially focuses the confirm action; destructive rejection and account-exit confirmation initially focus cancel. Closing a dialog returns focus to the action that opened it.
- The sticky save bar includes the bottom safe-area inset and may wrap on narrow screens. It must not cover the final field or make the current save state unreachable.

## Existing design sources

- `web/tailwind.config.ts`: font families, semantic theme, spacing and radius utilities.
- `web/src/styles/globals.css`: light/dark palette and additive configuration classes.
- `.superpowers/sdd/2026-08-30-configuration-center/ui-direction.md`: approved preflight and interaction direction.

No new font, token export format, motion dependency, marketing navigation or decorative asset is introduced. `web/src/styles/globals.css` contains the Hallmark app stamp and critique; the existing theme tokens remain the implementation source of truth.
