# Stdlib Tool & Docs Coverage

This note explains how Lyra exposes every stdlib function to AI tools and how to inspect documentation coverage.

- Discovery: `ToolsList[]` merges three sources in order:
  1) Explicit tool specs you register via `ToolsRegister` or inline `tool_spec!` macros.
  2) Rich default specs for common stdlib functions (strings, lists, etc.).
  3) Fallback builtin cards from `DescribeBuiltins[]` (ensures every builtin is discoverable).
- Export:
  - `ToolsExportOpenAI[]` emits all items from `ToolsList[]` in OpenAI function/tool format.
  - `ToolsExportBundle[]` contains all specs/cards (registered + default + fallback).

Quick checks

- List everything: `ToolsList[]` (optionally filter by tags/effects).
- Describe a function: `ToolsDescribe["Map"]`.
- Export for OpenAI: `ToolsExportOpenAI[]`.
- Export bundle: `ToolsExportBundle[]`.

Improving docs

- Prefer inline `tool_spec!` entries in each module for high-value functions so names, params, tags, and schemas are accurate.
- Add examples to specs; `ToolsDryRun[id, argsAssoc]` validates against schemas and returns normalized args/errors.
- Module guides live under `docs/stdlib/*.md` (e.g., `string.md`, `functional.md`). Expand these with examples and cross-links.

Status

- Functional, math, list, assoc, crypto, image, and audio functions have explicit or default specs; all other builtins are exported via fallback cards.
- As you add more inline specs, `ToolsList`/exports automatically prefer them over fallback cards.
