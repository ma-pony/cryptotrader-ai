# CryptoTrader configuration workbench

The configuration center extends the existing application shell. It keeps the system font stack, semantic HSL tokens, CryptoTrader amber/cyan/trade tokens, Tailwind 3 and both themes.

## Forms

- Compact Chinese business headings, with English translations. Two-column desktop controls collapse to one column on narrow screens.
- 14px labels/help, 40px desktop controls and 44px mobile targets. Visible instant focus rings and field-linked errors.
- Section dividers instead of nested cards. Native advanced disclosures retain their children and draft values.
- Numeric drafts can remain empty until corrected. Ratios use explicit percentage labels and convert only at the field boundary.
- Per-section saves show dirty, saving, saved and failed state. Runtime application is distinct from persistence; saving never activates trading.
- Catalog choices replace plugin identifier entry. Credentials have separate write-only inputs and actions.

## Existing design sources

- `web/tailwind.config.ts`: font families, semantic theme, spacing and radius utilities.
- `web/src/styles/globals.css`: light/dark palette and additive configuration classes.
- `.superpowers/sdd/2026-08-30-configuration-center/ui-direction.md`: approved preflight and interaction direction.

No new font, token export format, motion dependency, marketing navigation or decorative asset is introduced. Configuration browser acceptance belongs to the integrated route task.
