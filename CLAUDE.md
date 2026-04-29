# Spec-Driven Development

This project uses **spec-kit** + **spex** for spec-driven development. The legacy Kiro
workflow (`.kiro/`) has been removed; all historical specs are preserved under `specs/`.

## Project Context

### Paths
- Specs: `specs/` (numbered: `001-*` through `012-*`)

### Active Specifications
- Browse `specs/` directly to see active and historical features
- Each spec lives in `specs/NNN-feature-name/` with `spec.md` + `plan.md` + `tasks.md`
  (newer spec-kit format) or `requirements.md` + `design.md` + `tasks.md` (legacy
  Kiro format, migrated 2026-04-28 — see `010-architecture-review/`,
  `011-dashboard-observability/`, `012-frontend-prototype-alignment/`)

## Development Guidelines
- Think in English, generate responses in Simplified Chinese. All Markdown content
  written to project files (e.g., `spec.md`, `plan.md`, `tasks.md`, review reports)
  MUST be written in Simplified Chinese unless the spec itself declares a different
  target language.

## Workflow (spec-kit + spex)

- Phase 1 (Specification): `/spex:brainstorm` → `/speckit-specify` → `/speckit-plan`
  → `/speckit-tasks`
- Phase 2 (Implementation): `/speckit-implement` (with `spex:teams-orchestrate` for
  parallelism when multiple independent tasks exist)
- Phase 3 (Review): `/spex:review-code` (auto-runs `spex:deep-review` if compliance
  ≥ 95% and trait enabled) → `/spex:stamp` (final gate)
- Status: `/spex:help` for the full command catalog

## Development Rules
- 3-phase approval workflow: Spec → Plan → Tasks → Implementation
- Human review required each phase
- Follow user instructions precisely; within scope, act autonomously: gather necessary
  context and complete requested work end-to-end, asking questions only when essential
  information is missing or instructions are critically ambiguous.
- **Never modify CLAUDE.md automatically** — it is user-maintained, not auto-generated.
