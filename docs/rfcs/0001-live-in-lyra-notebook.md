---
title: "Live-In-Lyra Notebook: A Value-Native, Reactive, and Smooth Experience"
status: draft
created: 2025-08-28
authors: Lyra Notebook Team
discussion: TBD
---

# Summary

- Problem: The current notebook leaks implementation details (raw JSON, stalls, context switches) and doesn’t feel native to the Lyra language.
- Proposal: Make the notebook “live in Lyra” by treating Lyra values as the core abstraction across compute, UI, and sharing. Ship a value‑native viewer registry, default tabular rendering for Frames, non‑blocking previews, and foundations for a reactive compute graph.
- Outcome: Seamless, fast, and trustworthy workflows where outputs render as type‑appropriate views by default, interactions round‑trip to Lyra code, and recomputation is selective and visible.

# Motivation

- Friction: Raw JSON over tables, modal stalls, unknown kernel state, and flaky previews (e.g., macOS freezes).
- Mental model: Users think in Lyra values (Frame, Dataset, Plot) but the UI often thinks in files or JSON.
- Reliability: No clear guardrails for long‑running or stuck operations; limited feedback during previews.
- Flow: Too many context switches; limited parameterization; hard to go from exploration to shareable outcomes.

# Goals

- Value‑native rendering: Default viewers keyed by type via `application/lyra+value`.
- Smooth previews: Streaming where possible, timeouts/retries, never block the UI.
- Round‑trip UX: Viewer interactions generate Lyra code; code drives viewers.
- Reactive foundations: Track symbol/value dependencies; support selective recompute.
- Performance & trust: Fast first row, cancelable work, visible kernel health.

# Non‑Goals (initial)

- Full collaboration (presence, inline comments, review flows).
- Full DAG UI (start with internal engine; surface graph later).
- Complete export/publishing (defer to later phases).

# User Stories

- EDA: “When I output a Frame, I see a table immediately with schema and stats one click away.”
- Iteration: “Adjusting a parameter recomputes only affected cells and previews fast.”
- Debugging: “If a preview stalls, the UI tells me quickly and lets me retry or cancel.”
- Learning: “Hover docs and completions keep me in flow without leaving the notebook.”
- Sharing: “I can turn a notebook into a small app with input widgets and stable outputs.”

# Design Overview

## Typed Values & MIME

- Primary: `application/lyra+value` for machine‑readable outputs with discriminated unions (`__type`: `Frame`, `Dataset`, `Assoc`, `List`, `Integer`, `Real`, `String`, `Boolean`, `Null`).
- Fallback: `application/json` for plain JSON; always attempt typed decode first.
- Binary: `application/octet-stream` or type‑specific subtypes for media assets.
- Versioning: `x-lyra-version` field; maintain backward‑compatible decoders.

## Viewer Registry

- Core mapping: `Frame → Table`, `Dataset → Table`, `Assoc → Inspector`, `List → List Viewer`, `PlotSpec → Chart`.
- API: `registerViewer(typeId, capabilities, render, actions)`.
- Capabilities: `preview`, `paginate`, `stats`, `actionsToCode`.
- Extensibility: Feature‑flagged plugins; strict type contracts and validation.

## Table Viewer (Frames/Datasets)

- Rendering: Virtualized rows, infinite scroll, sticky header; schema‑aware formatting.
- Interactivity: Sort, filter, select columns; selection inserts Lyra filters/selects into the active cell.
- Stats: Missingness, type counts, min/max, histograms; click‑to‑code.
- Data Viewer: “Open in Data Viewer” via `cmd_table_*` with caching and pushdown.

## Preview Reliability

- Timeouts: Wrap `cmd_preview_value` in short timeout (e.g., 4s); show “Preview timed out” with retry and sample/full toggle.
- Streaming: First rows ASAP; progressively add schema and stats.
- Decode: Detect and decode typed Lyra JSON on client before viewer selection.
- Cancellation: User cancel propagates to kernel; UI remains responsive.

## Reactive Compute Foundations

- Dependency map: Track symbols per cell; hash code + inputs; link outputs to inputs.
- Selective recompute: Recompute only impacted nodes; show impact list before run.
- Cache: Content‑addressed results reused across cells/sessions when inputs match.
- State: Status bar with kernel health and last‑run indicators; safe restart preserves text and cache.

## Editor & Tooling

- LSP: Autocomplete, signatures, hover docs; quick fixes for common errors.
- Run controls: Run cell/selection/to cursor; visible impact estimate.
- Inline diagnostics: Timing, row counts; deeper query plans on demand.
- Parameter cells: Lightweight syntax for widgets bound to symbols, triggering recompute.

# APIs & Protocol

- Kernel emission: Prefer `application/lyra+value`; include type and preview handles for large values.
- Tauri commands: `cmd_preview_value`, `cmd_table_open/schema/query/stats/export` cancelable; `timeoutMs` and `sample` options.
- Client helpers: `lyraValueType(node)`, `lyraDecode(node)`, `invokeWithTimeout(fn, ms)`, `createVirtualTable(handle)`.

# Backend Changes

- Kernel: Ensure Frame/Dataset outputs include preview handles; fast head rows; stabilize typed JSON.
- GUI API: Harden table open/schema/query/stats; sample/full semantics; cache small previews.
- Export: Stub `cmd_table_export` for later (not Phase 0).

# Frontend Changes

- Detection: Prefer `application/lyra+value`; decode typed values before viewer selection.
- Registry: Implement core viewer registry with feature flags; default to table for Frame/Dataset.
- Previews: Non‑blocking preview wrapper with timeouts; visible retries; progressive enhancement.
- Round‑trip: Viewer actions produce Lyra expressions inserted into the active cell.

# Performance & Reliability

- Targets: First‑row ≤ 300ms cached / ≤ 1s live (p95); no UI freezes; all long ops cancelable.
- Guardrails: Timeouts at preview and query layers; backpressure in viewers; clear error surfaces.

# Security & Privacy

- Isolation: Kernel process isolated from UI; structured messages only.
- Limits: Cap preview sizes; sanitize logs/telemetry; opt‑in usage metrics.
- Plugins: Signed/verified viewer plugins; sandboxed execution.

# Telemetry (Opt‑In)

- Metrics: Preview latency, viewer selection rate, timeout frequency, cancel/abort counts, crash rate.
- Privacy: No raw data; only timings, sizes, and type IDs.

# Rollout Plan

## Phase 0 — Smooth Basics (now)

- Typed value decoding; default table viewer for Frame/Dataset.
- Non‑blocking previews with timeout and retry.
- Kernel health indicator; cancel‑run controls.

## Phase 1 — Value‑Native & Tools

- Viewer registry; round‑trip actions.
- LSP basics; parameter cells; run‑selection granularity.

## Phase 2 — Reactive Engine

- Internal DAG and selective recompute; cache reuse; basic impact visualization.

## Phase 3 — Data Flow & Viz

- Linked stats/histograms; selection→code; crossfiltering; exportable visual assets.

## Phase 4 — Share & Extend

- Viewer plugin API; portable bundles; publishing and scheduled runs.

# Success Metrics

- Table default: ≥ 90% of Frame/Dataset outputs render as tables by default.
- Latency: p95 first‑row preview under target; timeout rate < 2%.
- Reliability: UI freeze rate ~0; cancel success rate > 99%; crash rate < 0.1% per session.
- Flow: ≥ 3 actions executed consecutively without context switching (palette/keyboard‑first).

# Risks & Mitigations

- Viewer complexity: Start with core types; gate others behind flags; measure perf.
- Protocol drift: Version and validate `application/lyra+value`; maintain strict decoders.
- Perf regressions: Add performance budgets to CI and dogfood builds; enable fast fallbacks.
- UX overload: Progressive disclosure; defaults stay simple; advanced panes behind “More”.

# Acceptance Criteria (Phase 0)

- Typed decode: Client detects and decodes `application/lyra+value` for Frames/Datasets.
- Default table: Frame outputs render as tables by default with schema bar and “Open in Data Viewer”.
- No freezes: Preview UI never blocks; visible timeout with retry and cancel; macOS freeze no longer reproducible.
- Health & cancel: Status bar shows kernel health; cancel stops running previews and cell execution.

# Appendix A: Initial Tickets (Draft)

- FE: Implement `lyraValueType` + `lyraDecode` with tests.
- FE: Viewer registry core and Frame/Dataset mapping.
- FE: Preview timeout wrapper with UI states (loading/timeout/error/retry).
- FE: Table viewer header with schema bar and “Open in Data Viewer”.
- BE: `cmd_preview_value` fast‑path head rows with sampling.
- BE: Stabilize typed JSON for Frames/Datasets; include `x-lyra-version`.
- BE: Ensure `cmd_table_*` cancelability and basic caching.

