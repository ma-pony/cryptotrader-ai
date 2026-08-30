# 配置中心重构

Status: approved in conversation on 2026-08-30. The user approved the full proposal after the configuration audit.

## Outcome

用户第一次打开系统，选择 Paper，看到“初始模拟资金（USDT）”而不是参数 JSON。为连接命名、填写金额、保存、只读测试，然后配置模拟资金池与组件权重。每个字段有清晰的名称、单位、默认值和必要说明。用户可以保存后离开，再回来修改；无需启用系统才能访问配置。配置完成后另行检查并启用。第二次访问使用同一组表单，不存在初始化专属的配置死角。

## Global Constraints

- No raw parameter JSON editors or generic key/value escape hatch in any configuration screen, including advanced settings and custom plugins.
- Database runtime configuration and the existing encrypted credential vault remain authoritative; do not add TOML or environment-variable settings.
- Use installed code-owned plugins only. Plugin discovery must describe unconfigured installed plugins without constructing models, calling LLMs, or contacting exchanges.
- Preserve Kronos and the four-agent LLM committee as peer components, configurable weights, and internal debate controls.
- Preserve configurable signal/book HITL, simulated/real capital separation, canary-only exclusion, and the independent default-off live-order execution switch.
- Preserve revision compare-and-swap, runtime apply barriers, credential redaction, and write-only secret inputs. Saving does not activate an inactive system.
- No backward-compatibility layer, raw JSON fallback, workflow engine, new design dependencies, or speculative edge-case framework.
- Chinese user-facing labels must explain business meaning and units; keep the existing English locale functional.
- Keep existing light/dark theme, typography, tokens, and app shell. Scope visual changes to configuration and related forms; no marketing chrome or decorative imagery.
- Do not read or print secret values, change the stable CONFIG_MASTER_KEY, overwrite .env, activate trading, submit orders, merge, or push during this work.

## Information architecture

The configuration family has eight sections: models, signals, market data, venue connections, execution books, risk/HITL, scheduler/triggers, and system/security. Existing operational pages remain operational: risk monitoring and trigger history link to their configuration sections. Existing route entry files may be refactored in place; no compatibility alias routes are required.

Initialize with a checklist using the same domain forms. Make all settings accessible while inactive. Steps have meaningful readiness states, permit revisiting, and preserve unsaved non-secret values during navigation. Save per section with explicit feedback; activate only from a final review action. Do not mistake a visited step for a valid step.

## Forms

- Models: gateway URL/key, role model names with Chinese role labels, fallback, temperature, prompt caching; advanced retry, timeouts, streaming list and per-model cost rows. Retain custom model names because gateways differ. Do not invent model availability or pricing.
- Signals: select installed plugins by display name and description; enabled weights sum to 100%; no silent redistribution. Inline total/remaining indicator. Kronos parameters and committee debate parameters are ordinary typed fields; uncommon fields are in advanced groups.
- Market: installed source selector and its actual parameters. Ordinary text, number, choice and list controls. Credentials must not be exposed as readable ordinary configuration values.
- Venues: select installed platform; only valid environments; Paper initial_equity as a positive USDT amount, no credentials; OKX key/secret/passphrase (all required); Bybit key/secret only. Default Paper amount for a new form is 10000 USDT, editable and not written until save. Account ID generated on create and immutable thereafter; business label remains editable. Show leverage/margin fields only with their actual meaning/capability. Canary-only in advanced settings with clear explanation.
- Books: simulated versus real grouping, eligible enabled connections, no canary allocations, weight total/remaining and allocation preview clearly labelled as an example (not a live balance). Keep stable editor keys. Explain HITL precedence. Do not silently clear allocations when changing scope; make scope fixed for an existing book or require an explicit local draft reset.
- Risk: position, loss, cooldown, volatility, exchange checks and frequency groups; percentages, minutes, seconds and milliseconds explicit. One state per value. Keep approval expiry editable after setup.
- Scheduler: enable, trading-pair list, interval, daily summary time with its actual timezone, trigger engine controls, and existing rule forms with visible validation/request errors.
- System: access security and key rotation, Redis and actual supported notification/observability configuration. Trace consumers; remove dead settings rather than making nonfunctional controls. Do not introduce a new notification integration or telemetry platform to justify a dead field.
- Backtest parameter form: preserve existing workflow, clarify units/date ordering, display request failures, and remove any inert selector. It is not a runtime configuration section.

## Plugin contract

Installed plugin factories carry a small code-owned configuration definition: identity, display name, description, typed parameter model and UI hints. The same definition powers the installed catalog, server validation, and initial form defaults. Nested parameter models produce grouped fields (e.g. debate); arbitrary objects are not accepted as an editor fallback. Support the field kinds needed by shipped plugins: text, integer/number, boolean, choice and string list. Support their declared range, step, default, description, unit and advanced flag. Reject unsupported field definitions at registration with a clear developer error.

Expose GET /api/config/catalog independently of active trading. It must include installed but unconfigured components and market sources, plus supported environments and credential requirements per venue. Never include credential values. Parameter validation must reject unknown keys and bad values before committing a new configuration revision. Default completion for a sparse valid parameter object is normal model defaulting, not a legacy migration.

## Save, validation and verification

Common form primitives own labels, help, units, field errors and focus. Draft state remains separate from the last saved document, but never duplicate a single value as both parsed and JSON text. Validation is inline; server failures are safe, specific and actionable. A successful save and successful runtime application are distinct statuses. Editing again clears obsolete success feedback. Reload/discard warns only when there are unsaved changes; no prompt spam for clean navigation. Route changes preserve draft or ask before discarding. No secrets in localStorage, URLs, previews or error messages.

The platform test performs an actual read-only authenticated account query for real external adapters, and validates the local account for Paper. It does not place/cancel orders or alter leverage/margin. Always close the temporary session. UI reports what was checked, checked time, and a safe failure reason. A saved credential state is not a tested connection. Changes to connection settings or credentials invalidate its previous test result. Read-only authentication cannot prove order-placement permission; do not claim it does.

## Visual system

Use the existing modern-minimal utilitarian console as an app workbench: section navigation, narrow readable form content, restrained separators, and a consistent action/status bar. No gradients, giant headings or nested cards for every field. Common controls are first; advanced controls remain normal controls inside disclosure groups. All fields remain keyboard accessible with instant focus rings and associated errors. Mobile single-column layouts must work at 320, 375, 414 and 768 pixels, without horizontal page overflow. Preserve the current system font stack and semantic theme tokens; no font or animation dependency downloads.

## Acceptance

1. Render and interact with every configuration section without a JSON editor, including plugin parameters.
2. Add a fixture installed plugin without front-end plugin-specific code; its typed fields appear and save correctly.
3. Paper positive amount / OKX required passphrase / platform environment rules reject invalid saves; unknown parameter keys are rejected server-side.
4. Invalid external credentials cannot produce a healthy test result; testing is demonstrably read-only and closes sessions.
5. Save/reload round-trips model, signal, market, risk, scheduling, venue and book changes; no double-state overwrite; concurrent revision failures are actionable and preserve drafts.
6. Inactive setup can save and revisit every section without activation; activation remains explicit and safety gates remain enforced.
7. Existing valid credentials remain encrypted and neither tests nor errors print their values.
8. Keyboard and four mobile widths verified; default/empty/loading/error/success/dirty states make the next action clear.
9. Run backend tests, frontend tests, typecheck, lint/build, and real browser walkthrough. Real provider calls require separately confirmed credentials/environment; no order canaries in this UI refactor.

## Implementation finish checklist (2026-08-30)

- [x] Shared typed configuration center, installed metadata/validation, venue/book workflows and consumed settings implemented in Tasks 1–5.
- [x] Task 6 complete backend/frontend/type/lint/build verification and isolated browser/API evidence recorded in [the durable acceptance report](../../verification/configuration-center/README.md).
- [x] Required viewport sizes, visible errors/focus, mobile dirty actions, light/dark and English inspected; native dialog/select/date tool limitations explicitly recorded.
- [ ] Independent final review of the complete configuration change set closed.
- [ ] Separately authorized existing-database cleanup and deployment, if requested. No activation, merge or push is part of this implementation handoff.
