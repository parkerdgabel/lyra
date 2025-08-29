# RFC 0001 Tracking — Phases 0–3

This document tracks acceptance criteria and tasks for RFC 0001: Live-In-Lyra Notebook.

Source RFC: docs/rfcs/0001-live-in-lyra-notebook.md

## Phase 0 — Smooth Basics

Acceptance Criteria

- [x] Typed decode: Client detects and decodes `application/lyra+value` for Frames/Datasets.
- [x] Default table: Frame outputs render as tables by default with schema bar and “Open in Data Viewer”.
- [x] No freezes: Preview UI never blocks; visible timeout with retry and cancel; macOS freeze no longer reproducible.
- [x] Health & cancel: Status bar shows kernel health; cancel stops running previews and cell execution.

Tasks

Frontend

- [x] Add `lyraValueType` + `lyraDecode` helpers with unit coverage where feasible.
- [x] Prefer table viewer for Frame/Dataset; keep Raw JSON toggle.
- [x] Wrap preview in timeout with visible states (loading/timeout/error/retry).
- [x] Table header: schema bar + “Open in Data Viewer”.
- [x] Stop button to interrupt kernel; top progress indicator.

Backend

- [x] `cmd_preview_value` fast-path head rows with sampling; handles Frame and Dataset.
- [x] `cmd_table_*` paths (open/schema/query/stats) robust; basic caching for Frame rows.
- [x] Emit `application/lyra+value` for machine-readable outputs.

Stability

- [x] Manual sanity on macOS: preview Frames, switch to Data Viewer, interrupt long ops.

## Phase 1 — Value‑Native & Tools

Acceptance Criteria

- [x] Viewer registry: Core registry with enable/disable, validation, schema, and details UI.
- [x] Round‑trip actions: Table previews/Data Viewer can insert filters/selects into code.
- [x] Parameter cells: Lightweight widgets bound to symbols; changes mark dependents dirty.
- [x] LSP basics: Definitions/refs/rename wired (F12/Shift+F12/F2).
- [ ] LSP polish: improved completions and hover.

Tasks

- [x] Implement viewer registry APIs and core viewers for Frame/Dataset/JSON table.
- [x] Add “Insert filter” with Key column selection; persist key per output.
- [x] Data Viewer: stats, histograms, null bars, timeouts and inline Retry.
- [x] Parameter cells MVP: parse inline annotations, render controls, update values in cell text, mark dependents dirty.
 - [x] Parameter cells polish: optional prompt for name on insert; auto‑insert missing let bindings for all params.
 - [ ] Parameter cells polish (remaining): warn on name collisions; optional single multi‑field prompt.
 - [x] Parameter cells tests: dedicated UI test page under `ui/tests` for parser and let‑update helpers.
- [x] Introduce LSP endpoint; wire editor for defs/refs/rename (F12/Shift+F12/F2).

## Phase 2 — Reactive Engine

Acceptance Criteria

- [x] Selective recompute: “Run Impacted” using symbol deps; DAG ordering in kernel.
- [x] Cache reuse: Content‑addressed cache (memory + disk) keyed by kernel|schema|input; GUI wiring.
- [ ] Impact visualization: Expose a compact graph/impacted list UI before runs.

Tasks

- [x] Frontend dependency analyzer; “Run Impacted” button and confirm bar.
- [x] Kernel cache with blake3; disk cache under `.lyra-cache` per notebook.
- [x] Add DAG execution option to Run All; mini “Impacted cells” drawer with list.
 - [x] Cache UX: Settings toggle, per‑cell “Run ignoring cache”, cache clear + size cap GC.
 - [x] Cache UX: expose current cache size; per‑run "Ignore cache" toggles wired.
 - [x] Cache UX: add session-level cache salt control (UI + kernel).
 - [ ] Cache UX: persist Run All/Impacted toggles.

## Phase 3 — Data Flow & Viz

Acceptance Criteria

- [x] Linked stats/histograms: Column mini‑histograms/null bars; Stats pane.
- [x] Selection→code: Generate Lyra filters/selects from selections.
- [x] Crossfiltering: Brush overlay with metadata-driven mapping; summary/clear; persistence.
- [ ] Exportables: Table/chart export commands (CSV/Parquet and PNG/SVG).

Tasks

- [x] Crossfilter: Accept `application/lyra+chartmeta` adjacent to images; fallback prompts.
- [ ] Ensure stdlib plots emit chart meta (xColumn/xMin/xMax) consistently.
- [x] Implement `cmd_table_export` (CSV); add Export CSV in Data Viewer.
- [ ] Add chart export actions (PNG/SVG) and stdlib chart meta consistency.

## Notes

- Plugins (Phase 4 preview): Viewer plugin API added with local store; not sandboxed/signed yet.
- Security: Consider plugin sandboxing (iframe/worker) and signing in a follow‑up.
