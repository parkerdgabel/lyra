# Lyra Notebook — Visual Language

This document defines the design system for Lyra Notebook: tokens, components, states, motion, accessibility, and keyboard model. It blends Mathematica’s clear cell structure with VS Code’s calm, discoverable UI.

## Vision
- Calm, focused neutral surfaces with a teal-blue accent.
- Strong state semantics (running/ok/error) and clear hierarchy.
- Discoverable power: inline docs, command palette, sidebars.

## Principles
- Hierarchy: cells are primary, carded with clear input/output.
- State-first: rails, badges, and motion encode state.
- Restraint: color only for state, selection, and primary actions.
- Accessibility: AA+ contrast, keyboard parity, ARIA roles.

## Design Tokens

Colors (light/dark variants are defined in CSS):
- Surfaces: `--surface-1` (app bg), `--surface-2` (cards/panels), `--surface-3` (muted layer)
- Text: `--text-1` (primary), `--text-2` (secondary)
- Accent: `--accent` (brand teal-blue), `--accent-2` (complementary)
- State: `--success`, `--warn`, `--danger`, `--info` (added)
- Chrome: `--border`, `--border-strong`, `--ring` (added)
- Shadows: `--shadow-sm`, `--shadow-md`

Typography
- Body: Inter 14px / 1.5; secondary text uses `--text-2`.
- Code: UI monospace 14px / 1.5 for editors/outputs/usages.
- Scale: 12, 14, 16, 18, 24 for rhythm.

Spacing & Radii
- Spacing scale: 4, 6, 8, 10, 12, 16, 20.
- Radii: 6 (small), 8 (default), 10 (cards), 12 (cells).
- Borders: 1px standard, 2–3px for rails/brackets.

Motion
- Durations: 120–180ms (hover/focus), 200–240ms (panel open/resize).
- Easing: standard ease; prefer transforms.
- Progress: indeterminate top bar for bulk/running; per-cell spinner.

Density Modes
- Comfortable: default paddings, 160px editor height.
- Compact: tightened paddings, 120px editor height, tighter badges.

## Components

Cells
- Structure: gutter (In/Out labels), body (head, editor/text, error panel, outputs).
- Status rail: left 3px rail (idle/border, running/accent pulse, ok/success, error/danger).
- Head: badges (type, language, time, exec count), actions (Run primary, Delete ghost).
- Selection bracket: subtle bracket on far-left when hovered/focused (planned).
- Reorder handle: appears on hover near gutter (planned).

Editor
- Container: muted bg, 1px border, 8–10px radius; focus ring on focus-within.
- Text: transparent textarea + highlight overlay; accent-tinted selection.
- Diagnostics: inline single-line message; underline tokens in overlay (future: squiggles).
- Autocomplete: two-pane popup (list + docs), keyboardable.
- Signature help: bubble near caret with active parameter.
- Hover docs: compact tooltip near pointer.
- Scroll shadows on overflow.

Outputs
- Defaults: wrapped; carded `pre` with subtle shadow; Copy and Show more/Collapse.
- Syntax colors: legible JSON/Lyra value highlighting.
- Variety: prefer text; fall back to first output for non-text; image/chart special-casing later.

Error Panel
- Placement: above outputs, below editor.
- Visual: tinted bg (danger mix), left 3px rail, bold header with dot.
- Actions: Copy error.

Sidebars
- Left: Outline (state chips), Queue (live durations), Problems (diagnostics). Resizable, overlay on mobile, persistent.
- Right: Docs panel: title, usage, summary, insert usage/examples. Resizable and persistent.

Header & Toolbars
- Sticky, blurred, light border. Buttons: neutral by default; primary accent for Run.
- Scroll shadows for overflowing toolbars.

Command Palette
- Centered overlay with input and list; supports commands and symbols; fuzzy ranking; keyboard-first.

Keyboard Model
- Global: Save (Ctrl/Cmd+S), Run All (Ctrl/Cmd+Shift+Enter), Stop (Ctrl/Cmd+.), Add Code (Ctrl/Cmd+Alt+C), Add Text (Ctrl/Cmd+Alt+T), Delete Cell (Ctrl/Cmd+Backspace/Delete), Toggle Sidebar (Ctrl/Cmd+B), Toggle Docs (Ctrl/Cmd+J), Palette (Ctrl/Cmd+K), Quick Doc (Ctrl/Cmd+I).
- Editor: Run cell (Ctrl/Cmd+Enter), autocomplete (Ctrl/Cmd+Space), up/down navigate, Enter/Tab accept, Escape close.

Accessibility
- Contrast AA+ for UI; visible focus ring everywhere; keyboard reach to all controls; ARIA roles on tabs/panels/lists; live region for status.

## Gap Map (spec → current implementation)

Tokens
- `--info`: MISSING in CSS → Added in spec; not yet referenced in UI.
- `--ring`: MISSING in CSS → Added in spec; focus styles still use ad-hoc shadows.
- Output syntax colors: hard-coded hex in CSS (`.out-*`) → Should migrate to tokens.
- Motion durations/easing: scattered literals → Consider motion tokens (e.g., `--dur-fast`, `--dur-med`).
- Spacing/radii scales: used as literals → Consider tokens for common sizes (`--space-*`, `--radius-*`).

Cells
- Selection bracket: NOT IMPLEMENTED → Add left bracket on hover/focus synced with status rail.
- Reorder handle/drag: NOT IMPLEMENTED → Add handle affordance and drag behavior.
- Dirty badge: NOT IMPLEMENTED → Add badge when unsaved edits exist (requires model signal).

Editor
- Diagnostic squiggles: PARTIAL (message only) → Add overlay token-level squiggles.
- Focus ring: PARTIAL → Switch to `--ring` token and unify focus visuals.

Outputs
- Image/chart handling: NOT IMPLEMENTED → Add renderers and cards beyond text.
- Long-output collapse: IMPLEMENTED (line/char thresholds) → OK.
- Copy: IMPLEMENTED → OK.

Sidebars
- Problems severity styles: MINIMAL → Expand severity styling (info/warn/error) using tokens.
- Outline state icons: MINIMAL → Add subtle icons where helpful (keep calm aesthetic).

Header & Toolbars
- Focus states: PARTIAL → Use `--ring` for inputs/selects/buttons consistently.

Command Palette
- Symbol vs command badging: MINIMAL → Consider subtle badges/icons.
- Docs preview in palette: PARTIAL → Optional enhancement.

Accessibility
- ARIA for palette and lists: PARTIAL → Add roles/aria-selected for list items.
- Live regions: PARTIAL (status div only) → Review status announcements for runs/errors.

## Next Steps (implementation-order)
1) Tokens: add `--ring`, `--info`; introduce motion tokens; plan spacing/radius tokens.
2) Focus ring: apply `--ring` across focusable components; remove ad-hoc shadows.
3) Cell bracket + reorder handle: add visuals (no functionality change first, then drag).
4) Problems severity styles: info/warn/error tokens and icons.
5) Output syntax tokens: replace hard-coded hex in `.out-*` with variables.
6) ARIA pass: palette, tabs, lists, status announcements.
7) Optional: palette docs preview and filtering; image/chart output cards.

This doc should evolve with the implementation; keep tokens authoritative and extend with any new states/components.

