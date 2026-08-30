# Task 4 report — shared configuration workflows

## Delivery

Implemented on `codex/pluggable-signal-fusion`, based on `54f1e27181775b4b9bc8baefccca6f5ac34d2daf`. Task 4 only; the controller authorized one required backend admission correction described below. No new dependencies, deployment, push, runtime activation or real credential/account/model/order calls.

- All eight named settings destinations and `/setup` use one mounted `ConfigurationProvider`. Inactive routing permits configuration access. Operational routes remain intact; `/strategy` renders the shared signals form.
- Setup is now a saved-state checklist linking to those exact forms. Activation is a separate explicit action, requires current saved checks/valid sections/required access credentials, refuses unsaved drafts and revision conflicts, and does not change the default-off live execution gate.
- The six Task 3 forms consume catalog, section SaveBar, field errors and current write-only secret metadata. Ordinary drafts survive section navigation, platform writes, credential rotation and reload. Actual document unload warns; explicit discard is confirmed by the existing SaveBar.
- The book overlay owns only `execution.books` and `execution.live_order_execution_enabled`. Dirty/restoration comparisons exclude connections. Derived display and saves take connections and other execution data from the newest baseline. The payload regression covers dirty books → platform rename/create → credential write → book save, while another model section remains dirty.
- Venue forms use installed catalog environments/credentials/parameters. Paper starts at editable 10000 USDT; OKX offers demo/live and requires passphrase. IDs are generated, advanced canary controls remain explicit, and credential fields explain API Key/API Secret/OKX Passphrase. Checks are read-only, for saved values only, and bound to the entire saved connection plus credential timestamp. A repeated check immediately invalidates prior readiness; failed checks cannot revive old success on navigation.
- Books have stable presentation keys, generated draft IDs, disabled saved ID/scope fields, percentage allocation/remaining, invalid connection removal, and a separate illustrative allocation area. Disabled, canary and wrong-scope connections are excluded. Drafts can be removed before save.
- Removed old setup JSON parsing/text state, wizard validators/duplicate forms and two unreferenced strategy form components. Removed files remain recoverable through Git.
- API-access credential acknowledgement updates the existing memory-only authentication key so the next real request uses its `X-API-Key` header. Secret requests remain outside mutation storage, and tokens are absent from shared configuration/query/mutation caches. This consumer correction was explicitly approved by the controller.
- A fresh protected page returning 401 now offers a password field for the existing access key and explicit GET-only unlock/reload. Wrong keys are cleared from the field and memory with safe retry feedback. This does not rotate vault credentials or send a configuration write.
- Added only `GET /api/config/catalog` to the exact read-only commissioning route set. The real dependency previously rejected it with 503 while inactive. Authentication remains enforced, and reads remain blocked during runtime application. Dependency-attached HTTP tests prove inactive 200, protected/no-key 401 and applying 503.

## Files and interfaces

New: `web/src/pages/settings/{index.tsx,configuration-context.tsx,navigation.ts,configuration-workflow.test.tsx}`, `web/src/lib/configuration-readiness{,.test}.ts`, `web/src/test/configuration-workflow.tsx`.

Additional admission files: new `web/src/pages/settings/configuration-access{,.test}.tsx`; changed `web/src/hooks/use-runtime-config.ts`, `src/api/dependencies.py`, `tests/test_configuration_catalog.py` and `tests/test_api_security_hardening.py`.

Changed: `App.tsx`; sidebar/top-bar navigation; `use-configuration-draft{,.test}.tsx` (implementation is `.ts`); `use-runtime-secrets.ts`; venue hook contract tests; setup/strategy pages and tests; settings lifecycle, venue and book pages/forms/tests; allocation preview; English configuration routing tests; SaveBar test label; Chinese/English common/configuration locales; appended configuration-only global CSS.

Deleted: `web/src/pages/strategy/components/{component-weight-card,decision-settings-card}.tsx`.

Integration contracts:

- `useConfiguration()` exposes the existing draft contract, catalog query, non-secret venue drafts, stable book keys, saved check map and `hasUnsaved`.
- `ConfigurationSaveBar` handles one `ConfigurationSection`. `books` extends that existing section union; no new generic merge layer.
- `VenueForm` is controlled through `value`/`onChange`; `onSaved` releases its ordinary draft. `onChecked(ConnectionCheck | undefined)` installs or invalidates saved check evidence. Credential input state stays local and clears on success/failure.
- `connectionFingerprint`, `connectionChecksReady`, `eligibleConnection` and `bookErrors` centralize the checklist/book invariants without network work.
- `workflowHarness` mocks only the HTTP/market-socket boundary and renders the real App/QueryClient/router/shared owner. Writes are asserted against the outgoing full document and the subsequent server fixture.
- `useRuntimeConfig.authenticationRequired` reports initial HTTP 401; `ConfigurationAccess` retains its local retry state while a reload is pending. The key enters only the existing memory store and request header, never query keys/data or mutation variables.

## TDD and test evidence

All commands were run in this worktree, with frontend commands in `web/`. Long-running sessions were retained and polled through final exit; no yielded/partial result is counted as completion.

### Initial behavior RED

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/setup/setup-page.test.tsx src/pages/strategy/strategy-page.test.tsx src/pages/settings/configuration-workflow.test.tsx
```

Exit 1: 3 files failed, 7 tests failed. Old inactive routing/wizard did not expose the shared destinations or controls.

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/settings/venues/venues-page.test.tsx src/pages/settings/execution-books/execution-books-page.test.tsx
```

Exit 1: 2 files failed, 7 tests failed. Missing new platform/book workflows and routed controls.

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/settings/configuration-draft-lifecycle.test.tsx
```

API-key regression RED: exit 1, 1 failed / 3 passed. Subsequent request header was null instead of the acknowledged access key. Corrected at the write acknowledgement boundary.

### Shared-owner GREEN milestone

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/setup/setup-page.test.tsx src/pages/strategy/strategy-page.test.tsx src/pages/settings src/hooks/use-configuration-draft.test.tsx src/hooks/venue-connection-contracts.test.tsx src/hooks/use-runtime-secrets.test.tsx
```

Exit 0: 9 files, 30 tests passed (5.46s). Includes payload preservation across venue/credential writes and API-access header/cache isolation.

Fixture corrections during this stage: the new App harness initially lacked the existing market context; added an offline boundary. Section transitions now await the destination control before interacting with repeated labels. The venue mutation fixture incorrectly echoed request-only `expected_revision` into a strict response connection (29 passed / 1 failed); corrected the fixture to the real response schema and asserted the newly saved editor closes. No production compatibility shim was added.

### Saved check readiness RED/GREEN

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/setup/setup-page.test.tsx src/lib/configuration-readiness.test.ts
```

Exit 1: 1 failed / 5 passed, after correcting the helper test's initially empty generic connection fixture. The real failure proved failed rechecks left activation enabled. Invalidating shared evidence before a recheck fixes this.

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/setup/setup-page.test.tsx src/lib/configuration-readiness.test.ts src/pages/settings/execution-books/execution-books-page.test.tsx
```

Exit 0: 3 files, 9 tests passed (3.66s).

### Final verification

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/typescript/bin/tsc --noEmit
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/eslint/bin/eslint.js .
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vite/bin/vite.js build
```

- Pre-admission full frontend: exit 0, **41 files / 205 tests passed**, 18.86s; no stderr warnings. After the controller-requested admission additions, the same full command passed **42 files / 207 tests**, 10.61s, exit 0 and no stderr warnings.
- TypeScript: exit 0, no diagnostics (rerun after final test edits).
- ESLint: exit 0, no diagnostics. Initial check found three test-only issues (act callback return values and an untyped mock body); fixed.
- Production build: exit 0, Vite 8.0.12, initially 2225 modules/3.32s; final admission build **2226 modules/2.14s**, no warnings. TypeScript and ESLint were also rerun after the admission implementation, both exit 0.
- The first full frontend run was exit 1: 200 passed / 5 failed. Four stale English tests used the removed standalone-page contracts/missing owner; migrated them to the actual routed fixture and current labels. The fifth was initial lazy venue loading exceeding Testing Library's default 1s under parallel load; its first semantic readiness query is now bounded at 5s, with no sleep. A focused rerun then had 19 passed / 1 failed because the new English assertion did not scope two Paper amount labels; scoped it to the new venue form. The final full run above proves all 205 together.

Backend command (worktree root):

```sh
rtk proxy .venv/bin/pytest --no-cov -q tests/test_configuration_catalog.py tests/test_runtime_config_api.py tests/test_runtime_config_models.py tests/test_runtime_config_repository.py tests/test_runtime_config_secrets.py tests/test_venue_connections_api.py tests/test_venue_registry.py tests/test_venue_domain.py tests/test_paper_venue_adapter.py tests/test_okx_venue_adapter.py tests/test_bybit_venue_adapter.py tests/test_ccxt_venue_contract.py
```

Exit 0: **252 passed, 1 warning**, 24.33s. Existing `LangChainPendingDeprecationWarning` from `langgraph/cache/base/__init__.py` about future `allowed_objects` defaults. The subsequent controller-authorized dependency edit was verified by the additional admission/auth suite below.

### Fresh authentication and real inactive catalog admission

```sh
rtk proxy /Users/rccpony/.nvm/versions/node/v24.19.0/bin/node node_modules/vitest/vitest.mjs run src/pages/settings/configuration-access.test.tsx
rtk proxy .venv/bin/pytest --no-cov -q tests/test_configuration_catalog.py -k inactive_catalog
```

Fresh-auth RED: exit 1, 2 failed (missing existing-key entry). GREEN: exit 0, 2 passed, 2.84s; subsequent full frontend includes both tests. Catalog RED: exit 1, 2 failed / 1 passed / 7 deselected; inactive and protected requests incorrectly returned 503, while applying correctly returned 503. GREEN after the one-line admission change: exit 0, 3 passed / 7 deselected / 1 known warning, 5.87s. New catalog fixtures set Redis URL empty to avoid rate-limit connection attempts; they attach the actual API dependency and assert no adapter connection was opened.

```sh
rtk proxy .venv/bin/pytest --no-cov -q tests/test_configuration_catalog.py tests/test_api_security_hardening.py tests/test_misc_coverage.py::TestVerifyApiKey tests/test_runtime_config_api.py tests/test_venue_connections_api.py
rtk proxy .venv/bin/ruff check src/api/dependencies.py tests/test_configuration_catalog.py tests/test_api_security_hardening.py
rtk proxy .venv/bin/ruff format --check src/api/dependencies.py tests/test_configuration_catalog.py tests/test_api_security_hardening.py
```

Admission/auth suite: exit 0, **69 passed / 1 known warning**, 22.17s. Ruff check/format: exit 0, all checks passed / 3 files already formatted (one new import-order issue was corrected).

Two commit hook attempts stopped on detect-secrets false positives: public credential labels/help and fake test data (including the assertion copy). Public TS labels, translation mapping keys and both fake-value occurrences now have explicit false-positive comments; locale keys describe their purpose (`identity/signing/phrase`). No hook was bypassed and no secret-scan baseline was changed.

`rtk git diff --check`: exit 0. Formatting used the existing Prettier executable on the explicit changed/new TypeScript file list; globals.css was not reformatted and remains append-only.

## UI review and remaining boundaries

Frontend-design/Hallmark follow the approved existing app workbench, theme, system fonts, Tailwind 3 tokens and restrained form dividers. No marketing surfaces, generated images or new visual dependency. New navigation has named routes, instant focus outlines, non-wrapping labels and mobile hit targets; forms reuse Task 3 field/error controls. Source-level critique: P4 H4 E4 S5 R5 V4. This is not a rendered viewport/contrast acceptance claim.

Task 6 owns fixture-backed browser acceptance at the specified mobile/desktop widths; no writes or browser experiments were performed against the user's old localhost:5173 deployment. Task 5 owns dead backend consumer/type cleanup (including unused `allocation_policy`), actual HITL expiry enforcement, optional news credential vertical integration and operational scheduler/backtest consumer cleanup. No allocation policy selector, disconnected news control, or Task 5 operational form edit was introduced.
