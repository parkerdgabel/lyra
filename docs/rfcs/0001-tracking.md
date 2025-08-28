# RFC 0001 Tracking — Phase 0

This document tracks Phase 0 acceptance criteria and tasks for RFC 0001: Live-In-Lyra Notebook.

Source RFC: docs/rfcs/0001-live-in-lyra-notebook.md

## Acceptance Criteria

- [ ] Typed decode: Client detects and decodes `application/lyra+value` for Frames/Datasets.
- [ ] Default table: Frame outputs render as tables by default with schema bar and “Open in Data Viewer”.
- [ ] No freezes: Preview UI never blocks; visible timeout with retry and cancel; macOS freeze no longer reproducible.
- [ ] Health & cancel: Status bar shows kernel health; cancel stops running previews and cell execution.

## Tasks

Frontend

- [ ] Add `lyraValueType` + `lyraDecode` helpers with unit coverage where feasible.
- [ ] Prefer table viewer for Frame/Dataset; keep Raw JSON toggle.
- [ ] Wrap preview in timeout with visible states (loading/timeout/error/retry).
- [ ] Table header: schema bar + “Open in Data Viewer”.
- [ ] Stop button to interrupt kernel; top progress indicator.

Backend

- [ ] `cmd_preview_value` fast-path head rows with sampling; handles Frame and Dataset.
- [ ] `cmd_table_*` paths (open/schema/query/stats) robust; basic caching for Frame rows.
- [ ] Emit `application/lyra+value` for machine-readable outputs.

Stability

- [ ] Manual sanity on macOS: preview Frames, switch to Data Viewer, interrupt long ops.

## Notes

- Phase 0 is mostly UI/UX plumbing and reliability guardrails; exporting and plugins are explicitly out of scope.

