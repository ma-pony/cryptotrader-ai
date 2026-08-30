# Configuration Center Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace all configuration JSON editors with an understandable, typed and reusable configuration center, including safe initial setup and truthful platform tests.

**Architecture:** Shared domain forms are reused by setup and daily settings. Installed plugin definitions provide typed parameter metadata and server-side validation through one small catalog, while common form controls handle labels, units, errors and state. The existing runtime document, CAS/apply machinery and encrypted vault remain the write path.

**Tech Stack:** Python 3.12/Pydantic/FastAPI, React 19/TypeScript 5.9/Vite 8/Tailwind 3, React Query, existing Radix controls, Vitest/Testing Library and pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-configuration-center.md`

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

## Workspace and verification

Work only in `/Users/rccpony/Projects/cryptotrader-ai/.worktrees/pluggable-signal-fusion`, branch `codex/pluggable-signal-fusion`. Prefix every shell command with `rtk`; use apply_patch for edits. Do not touch the parent checkout or another plan's ledger. Begin from `1abd8583faa69598c2b216b98e5d911a37932146`.

Baseline: frontend 34 files / 195 tests passing; backend configuration and venue API subset 86 tests passing. The first attempted backend command named a nonexistent file; the corrected command below passed.

```sh
rtk .venv/bin/pytest --no-cov -q tests/test_runtime_config_models.py tests/test_runtime_config_api.py tests/test_venue_connections_api.py
rtk /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run
```

The Node command runs with cwd `web`. Other frontend commands use that same absolute Node executable and the corresponding `node_modules/typescript/bin/tsc`, `node_modules/eslint/bin/eslint.js`, `node_modules/vite/bin/vite.js` paths. Run focused tests during iteration and the relevant complete suite before each task commit. Commit only the task's files.

## File ownership

- `src/cryptotrader/configuration/`: installed plugin declarations, parameter models/catalog and validation only; no runtime sessions or UI state.
- `src/api/routes/config.py`: catalog transport and structured safe validation errors on the existing write path.
- `src/cryptotrader/venues/{protocol,ccxt_base,paper}.py`, `src/api/routes/venues.py`: read-only connection check.
- `web/src/components/configuration/`: common field, disclosure, action/status and parameter controls.
- `web/src/pages/settings/forms/`: models, signals, market, risk, scheduling and system domain forms.
- Existing setup/strategy/venues/books page entry files: route-level orchestration only, delegates to shared forms.
- `web/src/pages/settings/index.tsx`: settings section navigation and shared page composition, accessible before activation.
- Existing locales and API schemas/types: same contracts as runtime/backend, no disconnected duplicate validation schema.
- `design.md`, `.hallmark/preflight.json`, `.hallmark/log.json`: one shared design record for configuration pages, not one theme per page.

### Task 1: Installed plugin configuration catalog and parameter validation

**Files:**
- Create: `src/cryptotrader/configuration/__init__.py`, `fields.py`, `catalog.py`, `parameters.py`.
- Modify: signal, market-source and venue registry/factory files where configuration definitions are registered; `src/api/routes/config.py`; parameter consumers in built-in factories.
- Test: `tests/test_configuration_catalog.py`; update affected registry/config API tests and fixture plugins under `tests/factories/`.

**Interfaces:**
- Consumes current `RuntimeConfigDocument`, existing factory entry-point groups and `JsonValueOut` serialization helpers.
- Produces code-owned `PluginConfiguration` (id, label, description, parameter model, environments/credential fields for venue definitions), `configuration_catalog()` and `validate_configuration_parameters(document)`.
- Produces `GET /api/config/catalog`: `{components: PluginDefinition[], venues: PluginDefinition[], market_sources: PluginDefinition[]}`. Each definition has `id`, `label`, `description`, `fields`, `environments`, `credential_fields`. Fields have dot-path `key`, `label`, `description`, `kind`, `default_value` (existing JsonValueOut), `required`, `minimum`, `maximum`, `step`, `unit`, `advanced`, `options`. Options have `value` and `label`. `kind` is `text | number | integer | boolean | select | string_list`; unused limits are null and unused lists empty. Labels/descriptions support the existing Chinese and English locales through a `LocalizedText {zh_CN, en_US}` value.
- Registry factories expose a `.configuration` declaration; built-ins and fixture plugins adopt it directly, no legacy fallback.

- [ ] **Step 1: Write failing catalog/validation tests.** The literal expected IDs include installed but unconfigured plugins; loading the catalog may import definitions but must not invoke factories. Use an installed fake factory that raises if called. Exercise the endpoint, not source text.

```python
async def test_catalog_exposes_paper_funding_field(api_harness):
    response = await api_harness.client.get("/api/config/catalog")
    assert response.status_code == 200
    paper = next(item for item in response.json()["venues"] if item["id"] == "paper")
    assert paper["environments"] == ["paper"]
    assert paper["credential_fields"] == []
    assert next(field for field in paper["fields"] if field["key"] == "initial_equity")["unit"] == "USDT"
```

Also test an unknown parameter and negative initial equity via the config API: 422, unchanged revision, no secret echoed. Cover nested debate values and a custom typed plugin. Definition support is tested by consuming and validating the fields, not asserting catalog counts alone.
- [ ] **Step 2: Run RED:** `rtk .venv/bin/pytest --no-cov -q tests/test_configuration_catalog.py`. Expect endpoint/validation assertions to fail before implementation.
- [ ] **Step 3: Implement the small typed contract.** Parameter models use the current consumed fields/defaults, `extra="forbid"`, safe validation errors, descriptions and UI hints. Generate field descriptors from the typed model; flatten nested models into dot-path groups. Register shipped definitions on code-owned factories. The catalog enumerates the same installed factories as the runtime, including unconfigured entries, without invoking them. Validate configured parameter objects before DB replacement and use typed models in built-in consumers. Do not add a generic recursive form language. Core risk/models remain dedicated forms.

```python
class PaperParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    initial_equity: float = Field(default=10000, gt=0, title="Initial simulated equity")

def validate_configuration_parameters(document: RuntimeConfigDocument) -> None:
    catalog = configuration_catalog()
    for component in document.signals.components:
        catalog.require_component(component.component_id).parameter_model.model_validate(dict(component.parameters))
    catalog.require_market_source(document.market_data.source_id).parameter_model.model_validate(dict(document.market_data.parameters))
    for connection in document.execution.connections:
        definition = catalog.require_venue(connection.adapter_id)
        definition.validate_environment(connection.environment)
        definition.parameter_model.model_validate(dict(connection.parameters))
```

The code illustrates the required validation flow; implement these named catalog lookup methods. Paper defaults fill absent form values, never overwrite a provided amount. Credentials use the encrypted vault; do not expose a news key through an ordinary parameter descriptor—trace this current consumer and carry any required vault extension into the report for Task 4 rather than displaying a plaintext key field.
- [ ] **Step 4: Run GREEN** focused new/registry/config tests, then backend suite. Update fixture factories to the new declaration instead of retaining compatibility. Run Ruff on changed Python files.
- [ ] **Step 5: Commit** `feat(config): describe installed plugins with typed parameter fields`; report exact transport types and RED/GREEN evidence.

### Task 2: Truthful read-only platform connection checks

**Files:**
- Modify: `src/cryptotrader/venues/protocol.py`, `ccxt_base.py`, `paper.py`, `src/api/routes/venues.py`, corresponding fake sessions.
- Test: `tests/test_venue_connections_api.py`, `tests/test_ccxt_venue_contract.py`, adapter tests.

**Interfaces:**
- Produces `VenueSession.check_connection() -> Awaitable[None]`, which raises credential-safe `VenueOperationError` on failed account read.
- Existing test endpoint keeps existing success fields and adds `checked_at` (UTC datetime). Safe failure detail includes an error code for missing credentials, authentication failure or unavailable account/network; no provider raw payloads. Task 3 adds frontend schema support.

- [ ] **Step 1: Write failing tests** using the actual CCXT session and a fake underlying CCXT client: rejected `fetch_balance` must fail the HTTP check; a successful read returns healthy and timestamp; close is called after success and failure; no order, cancel, leverage or margin method is called.

```python
async def test_check_connection_reads_account(session, ccxt_client):
    ccxt_client.fetch_balance = AsyncMock(side_effect=RuntimeError("private-key-marker"))
    with pytest.raises(VenueOperationError) as error:
        await session.check_connection()
    assert "private-key-marker" not in str(error.value)
    ccxt_client.fetch_balance.assert_awaited_once()
```

Use existing adapter fixtures or define focused ones exercising the real session. Also cover the API with a failing session check and assert no healthy response.
- [ ] **Step 2: Run RED** with focused tests; record the missing-read failure.
- [ ] **Step 3: Implement account reads**, with `await self.client.fetch_balance()` behind existing normalized error handling; Paper checks its in-memory account only. Call `await session.check_connection()` before setting success, retain `finally` close, return UTC checked time. No changes to order paths.
- [ ] **Step 4: Run GREEN**, focused API and adapter tests, then backend suite and Ruff.
- [ ] **Step 5: Commit** `fix(venues): authenticate read-only connection checks` and report safe error codes for UI mapping.

### Task 3: Shared configuration controls, draft state and domain forms

**Files:**
- Create: `web/src/components/configuration/{field,section,parameter-fields,save-bar}.tsx`, `web/src/hooks/use-configuration-catalog.ts`, `web/src/hooks/use-configuration-draft.ts`.
- Create: `web/src/pages/settings/forms/{model-settings,signal-settings,market-settings,risk-settings,scheduler-settings,system-settings}.tsx`.
- Modify: `web/src/types/api.schema.ts`, `api.ts`, locales, common test fixtures; add configuration tests beside new components.
- Create: `design.md`, `.hallmark/preflight.json`, shared configuration styles as needed; append only to global CSS.

**Interfaces:**
- Consumes Task 1 catalog and existing `useRuntimeConfig`, `useRuntimeSecrets`.
- Produces `ParameterFields({fields, value, onChange, errors, idPrefix})` for plugin parameters; dot-path access stores nested values without any JSON text representation.
- Produces named domain fields above, each receiving `{value, onChange}` for its runtime-document section, plus catalog/secrets where needed. `SignalSettings` consumes both signals and catalog; no free component-ID entry. `RiskSettings` also receives HITL expiry.
- Produces `useConfigurationDraft` owning a single complete editable document, baseline and dirty state, with section update, discard/reload, validation and explicit save. Existing CAS remains authoritative.

- [ ] **Step 1: Write failing interaction tests**: custom plugin number/choice/bool/nested debate fields save typed values; percentage controls convert 5 to 0.05; risk changes survive advanced toggling; error associates with input; dirty state resets only after a successful save and becomes dirty again on edit; failed requests keep edits. Use current QueryClient and HTTP fixture patterns, not mocked UI components.

```tsx
it('edits a nested plugin threshold without JSON', async () => {
  const onChange = vi.fn();
  render(<ParameterFields fields={debateFields} value={{debate: {max_rounds: 3}}} onChange={onChange} errors={{}} idPrefix="committee" />);
  const input = screen.getByRole('spinbutton', {name: '最大辩论轮数'});
  fireEvent.change(input, {target: {value: '4'}});
  expect(onChange).toHaveBeenLastCalledWith({debate: {max_rounds: 4}});
  expect(screen.queryByRole('textbox', {name: /JSON/i})).not.toBeInTheDocument();
});
```

`debateFields` is a literal catalog fixture in the current backend response shape, not generated by the renderer. Add full save round-trip tests at Task 4 integration.
- [ ] **Step 2: Run RED** using Vitest for new configuration tests.
- [ ] **Step 3: Implement shared controls and six forms.** Use existing tokens/system fonts, readable 14px help, compact headings, consistent 40px controls, explicit units and inline errors. Advanced uses accessible disclosure without animation dependencies. Numeric inputs permit temporary blank drafts and reject invalid save rather than silently turning blanks into zero. Model cost list is rows of model/input/output USD per million tokens; model lists and pairs use removable items plus text entry. For known read-only/supported values use choices; user-defined model names remain editable. Signal weights have a total/remaining indication, no automatic redistribution. Core forms expose currently consumed settings only. Do not wire display-only settings as if functional; report dead consumer gaps precisely.

```ts
export type FieldErrors = Record<string, string>;
export type DomainFormProps<T> = { value: T; onChange: (value: T) => void; errors?: FieldErrors };
export function percentToRatio(percent: number): number { return percent / 100; }
```

Keep credentials separate from ordinary draft values. Include loading, error, saved/applying/failed status text, and discard confirmation for dirty drafts. Preserve non-secret edits across section changes. Use React event-driven updates, stable keys, no duplicated derived state effects.
- [ ] **Step 4: Run GREEN**, all frontend tests, typecheck and ESLint. The new forms must be exercised directly even before routes use them.
- [ ] **Step 5: Commit** `feat(web): add reusable typed configuration forms` and report exported props/signatures for Tasks 4 and 5.

### Task 4: Configuration center, shared setup, and platform/book workflows

**Files:**
- Create: `web/src/pages/settings/index.tsx`, configuration readiness helper/tests.
- Modify: `web/src/pages/setup/index.tsx`, `web/src/pages/strategy/index.tsx`, `web/src/pages/settings/venues/{index,venue-form}.tsx`, `web/src/pages/settings/execution-books/{index,book-form,allocation-preview}.tsx`, `web/src/App.tsx`, sidebar/top-bar navigation, locale files and affected tests.

**Interfaces:**
- Consumes shared forms, catalog, draft state and checked_at/safe check failures from Tasks 1–3.
- Produces daily settings routes for all eight approved sections while preserving operational routes. Setup is a checklist wrapper over the same forms, not a second set of fields.
- A connection test result belongs to saved connection parameters plus credential update timestamp; neither “credentials configured” nor an old test authorizes a new configuration.

- [ ] **Step 1: Write failing end-to-end component tests**: inactive user reaches every config section; Paper form starts with editable 10000 USDT and no credentials; selecting OKX shows only demo/live and required passphrase; setup and daily fields behave identically; saving risk, model and scheduler values survives reload; test success disappears on changed connection or credentials; activation stays a separate explicit action; book-ID editing keeps focus and disabled/canary/wrong-scope connections cannot be allocated.

```tsx
it('keeps saving separate from activation', async () => {
  renderWithRuntime(<SetupPage />, inactiveDocument);
  await user.click(await screen.findByRole('link', {name: '风控与审批'}));
  await user.clear(screen.getByRole('spinbutton', {name: '每日最大亏损'}));
  await user.type(screen.getByRole('spinbutton', {name: '每日最大亏损'}), '3');
  await user.click(screen.getByRole('button', {name: '保存配置'}));
  expect(lastSavedDocument.risk.loss.max_daily_loss_pct).toBe(0.03);
  expect(lastSavedDocument.system.active).toBe(false);
});
```

Implement test harness variables using the current real HTTP mock pattern; assertions must inspect the outgoing document and subsequent server fixture reload. Test missing fields/failure states too.
- [ ] **Step 2: Run RED** on affected setup, strategy, venues/books and routing tests.
- [ ] **Step 3: Wire all shared forms.** Inactive users can access settings without enabling the runtime. Preserve `/strategy` as the actual signal-settings entry if useful, not as a legacy redirect. Group settings navigation coherently with one distinct label. Keep each existing route file thin. Remove the old JSON parsing/text state and duplicate wizard forms, updating tests to new behavior rather than maintaining old labels/hooks. Add labels/help for API Key/API Secret/OKX Passphrase, environment choices from catalog, stable generated IDs, supported Paper amount, advanced canary controls and safe credential rotation feedback. Test only saved connections and show read-only test semantics; do not claim trade permission verified. Books show editable percentages with total/remaining and a separate example allocation panel. Existing book scope/ID are immutable; drafts can be removed before saving. Keep all capital/HITL/live gates unchanged.
- [ ] **Step 4: Validate GREEN**: full frontend suite, backend config/venue suite, typecheck, lint and production build. Exercise failures with fake API responses; no real config writes or credentials in tests.
- [ ] **Step 5: Commit** `refactor(web): unify setup and daily configuration workflows`.

### Task 5: Remaining configuration consumers and operational-form polish

**Files:**
- Modify: `web/src/pages/scheduler/components/{rule-form-dialog,rule-table}.tsx`, scheduler page, risk page configuration links, `web/src/pages/backtest/components/backtest-form.tsx`, related hooks/locales/tests.
- Modify only as consumer tracing requires: `src/cryptotrader/runtime_config/models.py`, config API transport, runtime/scheduler notification assembly and current notification/observability consumers; mirrored frontend types/forms.
- Test: current runtime notification/config contracts and new scheduler/backtest form interaction tests.

**Interfaces:**
- All displayed configuration controls must have a real consumer or be removed. Do not retain ignored configuration fields for compatibility. Existing model/notification consumers are the reference, not old docs.
- Scope explicit removals to dead configuration fields and inert controls. No new notification service, telemetry framework or storage backend.

- [ ] **Step 1: Write failing tests** for scheduler field/request errors, correct trigger-specific required values, backtest start-before-end and failure display, and selected backtest session behavior (remove selector if API has no supported use). Test actual notification/system fields round-trip only if consumed. The production change each test catches must be named in the report.

```tsx
it('does not start a reversed backtest range', async () => {
  renderWithApi(<BacktestForm onRunStarted={vi.fn()} />);
  fireEvent.change(screen.getByLabelText('开始日期'), {target: {value: '2026-08-20'}});
  fireEvent.change(screen.getByLabelText('结束日期'), {target: {value: '2026-08-10'}});
  await user.click(screen.getByRole('button', {name: '启动回测'}));
  expect(screen.getByRole('alert')).toHaveTextContent('结束日期');
  expect(backtestRequests).toHaveLength(0);
});
```

- [ ] **Step 2: Run RED** targeted form/consumer tests.
- [ ] **Step 3: Complete consumer wiring or remove dead controls/fields** with all directly affected tests/types updated. Specifically audit notifications, observability, LLM vision/max image bytes, model timeout roles and market-provider credentials. For a setting with no consumer, delete it from config models/transport/forms and references instead of wiring a speculative feature. If user DB contains a removed field, report before deployment; do not silently edit their database. Add ordinary inline errors for scheduling rule mutations, correct labels and units, and coherent links between monitoring and settings. Replace readable JSON parameter dumps within configuration workflows with labelled summaries; cycle diagnostic payloads outside config scope need not be rewritten.
- [ ] **Step 4: Run GREEN** focused and complete affected suites, typecheck/lint/build; report removed fields with exact consumer evidence.
- [ ] **Step 5: Commit** `fix(config): complete configuration consumers and form feedback`.

### Task 6: Integrated verification and browser acceptance

**Files:**
- Tests and narrowly scoped integration fixes in the above areas only.
- Update: `README.md` / plugin author docs where current configuration or plugin registration instructions changed; approved spec finish checklist and `.hallmark/log.json`.

**Interfaces:**
- Consumes Tasks 1–5 and the approved spec's acceptance checklist.
- Produces verifiable test/build/browser evidence and a clean scoped commit; not activation, merge, push or real order execution.

- [ ] **Step 1: Add any missing cross-page regression test before fixing a discovered defect.** Required integration paths: typed custom installed plugin, setup partial save/revisit, dirty discard, config conflict, saved-versus-applied distinction, missing credential/auth failure, percentage persistence, and disabled/canary scope controls. Preserve meaningful old tests or replace them with new equivalent behavior tests.
- [ ] **Step 2: Run complete backend suite**, frontend suite, TypeScript, ESLint, Vite build, and diff whitespace check. Report warnings honestly. Do not claim complete if failures remain.
- [ ] **Step 3: Build a local UI test runtime without touching saved credentials/data.** Prefer the current local app only after confirming deployment does not require deleting/migrating user data. Otherwise run an isolated temporary fixture-backed application/HTTP test server with literal safe fixture data; do not inject hidden React/browser state. Use the Browser skill for interaction, no alternate browser automation when available. Real API-path persistence can be covered against an isolated test DB.
- [ ] **Step 4: Walk the user scene** at desktop and widths 320/375/414/768: visit every configuration section; add Paper, fill amount, save, verify/revisit; edit model and risk values; open plugin advanced/debate; book allocation preview; trigger error; inspect saved/error/empty states and keyboard focus. Do not activate, place orders, read stored secrets or call paid model endpoints. Capture screenshots and report actual evidence paths; review them visually.
- [ ] **Step 5: Complete scoped docs and commit** `test(config): verify configuration center user journeys`. Independent final review precedes any claim that implementation is finished. Delivery keeps this worktree/branch unless user separately authorizes integration.

## Finish checklist

- [ ] No hand-written JSON in initial or daily settings, advanced plugin settings included.
- [ ] Same parameter definitions drive custom installed plugin fields and server validation.
- [ ] Saved/applying/error/dirty states and field-level errors are visible and truthful.
- [ ] Every exposed setting has a consumer; every approved configuration section remains reachable after initialization.
- [ ] Paper/OKX/Bybit forms only offer supported environment/credential requirements.
- [ ] Authentication check is read-only, reports timestamp, and cannot falsely pass invalid credentials.
- [ ] All trade/HITL/canary/CAS/security barriers remain enforced.
- [ ] Desktop/mobile/keyboard user walkthrough and backend/frontend checks completed.
