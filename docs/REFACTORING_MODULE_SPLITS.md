Refactor Plan: Core Module Splits

Goal: Improve code readability, maintainability, and encapsulation without changing public APIs or behavior. Splits target very large files and clarify ownership boundaries (evaluation vs. concurrency vs. rewrite vs. stdlib dispatch, etc.).

Principles
- Preserve public API: keep `pub use` re-exports so external users don’t see breaking changes.
- Small, mechanical moves: extract submodules without semantic changes first.
- Group by responsibility: concurrency primitives together, rewrite functions together, parser stages separated by role, stdlib by domain.
- Add internal `mod` docs and quick rationale at top of each module.

Phased Approach (safe-by-default)
1) Extract submodules (no behavior change), keep re-exports to preserve APIs.
2) Run `cargo test` and docs build; fix imports.
3) Optional: feature-gate dormant code (e.g., numeric tower) to slim release builds.
4) Follow-ups: lint improvements (expect vs unwrap), registry/doc parity tests.

Crate: lyra-runtime

Current pain: `src/eval.rs` is very large and mixes evaluator core, concurrency, rewrite, docs/registry, schema/explain, env, and helpers.

Proposed layout (new files under `crates/lyra-runtime/src/`):
- `lib.rs`: unchanged public surface; `pub mod attrs;` plus `pub mod prelude;` re-exporting types/functions.
- `prelude.rs`: re-exports `Evaluator`, `evaluate`, `set_default_registrar`, and registrar fns.
- `core/`
  - `evaluator.rs`: `Evaluator` struct, core `eval` loop, env helpers, `make_error`, tracing buffers (TRACE_BUF), span handling.
  - `registry.rs`: builtin registration (`register_*` functions), docs registry (`DocEntry`, `set_doc`, `set_doc_examples`, `get_doc*`).
  - `rewrite.rs`: `Replace`, `ReplaceAll`, `ReplaceFirst`, `Thread`, `With`, and any pattern/rewrite helpers that are currently in `eval.rs`.
  - `schema_explain.rs`: `schema_fn`, `explain_fn`, trace emission and collection helpers.
- `concurrency/`
  - `pool.rs`: global `ThreadPool`, `spawn_task`, limiter (`ThreadLimiter`), `Scope` registry.
  - `futures.rs`: `Future`, `Await`, `Cancel`, task registry and lifecycle.
  - `channels.rs`: `ChannelQueue`, `Send`, `Receive`, `TrySend`, `TryReceive`, `CloseChannel`.
  - `actors.rs`: `Actor`, `Tell`, `Ask`, `StopActor`, actor registry.

Public API & wiring:
- `lib.rs` re-exports from `prelude` to keep `use lyra_runtime::{Evaluator, evaluate, set_default_registrar};` working unchanged.
- Registrar helpers (`register_concurrency`, `register_schema`, `register_explain`, etc.) move to `core/registry.rs` and submodules export their own register functions; `registry.rs` calls into them.

Follow-ups:
- Replace lock/condvar `.unwrap()` calls with `.expect("... lock/condvar")` and compute condvar timeouts from remaining deadline.
- Ensure `EndScope` drains/cancels futures created in that scope to avoid leaks.

Crate: lyra-repl

Current pain: `src/main.rs` is very large, combining REPL engine, completion, rendering, config, and handlers.

Proposed layout (`crates/lyra-repl/src/`):
- `main.rs`: CLI bootstrap; wires components together.
- `engine.rs`: REPL loop, evaluation, session state interactions.
- `completion.rs`: completions (rustyline + reedline modes), using-context and assoc key/value contexts.
- `display.rs`: colorized value formatting, truncation config, assoc/list rendering, profile/explain displays.
- `config.rs`: REPL configuration struct and persistence helpers.
- `input.rs`: keybindings, event handlers.
- `helpers.rs`: small utilities (tokenization, caret line display, fuzzy scoring) shared across components.

Public API & wiring:
- `main.rs` imports internal modules; no external API, so internal refactor only.
- Remove or scope `#[allow(dead_code)]` helpers by moving them into modules that use them; otherwise gate behind `#[cfg(test)]` or a `dev-tools` feature.

Follow-ups:
- Replace `expect("external printer")` with a graceful fallback or improved error reporting.

Crate: lyra-stdlib

Current pain: very large domain modules (`dispatch.rs`, `math.rs`, `net.rs`, `io.rs`, `dataset.rs`, `nn.rs`).

Proposed incremental split (examples):
- `math/`
  - `arithmetic.rs` (add, sub, mul, div, pow, exact coercions)
  - `transcendentals.rs` (exp, log, sin, cos, tanh, etc.)
  - `distributions.rs` (probability distributions)
  - `numeric_cast.rs` (coercions and BigReal/string parsing)
- `net/`
  - `http.rs` (HTTP client/server APIs)
  - `ws.rs` (WebSocket)
  - `sse.rs` (Server-sent events)
  - `auth.rs` (auth helpers)
- `dataset/`
  - `io.rs` (CSV/JSON lines read/write)
  - `ops.rs` (transformations: select, filter, join, groupby, etc.)
  - `schema.rs` (schema handling, Describe)
  - `sql.rs` (ExplainSQL, SQL-related plumbing)
- `nn/`
  - `core.rs` (common graph/building blocks)
  - `layers.rs` (layer definitions)
  - `ops.rs` (NN operations/apply)
  - `training.rs` (train/tune)
  - `summary.rs` (summary/property)
- `dispatch.rs`: remains the registration/router; submodules register their functions here.

Public API & wiring:
- Keep `lib.rs` exports stable; `dispatch` imports new submodules and registers functions as before.

Follow-ups:
- Replace panic-prone unwraps in crypto/JWT/encoding paths with error returns (Assoc with `error`+`tag`).
- Document lifecycle for registries (create/use/destroy) and ensure symmetry in code and docs.

Crate: lyra-core

Current pain: numeric tower scaffolding not yet integrated.

Proposed:
- Gate `numeric.rs` subtypes behind `#[cfg(feature = "numeric_tower")]` (or move to `numeric/` folder: `rational.rs`, `complex.rs`, `bigreal.rs`).
- Add module-level docs clarifying Phase 2 intent.

Crate: lyra-parser

Current pain: `src/parser.rs` is a single large file mixing lexer-like utilities, precedence layers, atoms, calls, patterns.

Proposed layout (`crates/lyra-parser/src/`):
- `lib.rs`: public `Parser` API surface and re-exports.
- `lexer.rs`: char-source handling, whitespace/comments, token peeks.
- `parse_expr.rs`: orchestration; precedence climbing between logical/comparison/additive/etc.
- `parse_atoms.rs`: numbers, strings, lists, associations, symbols, pure functions.
- `parse_calls.rs`: calls, indexing, pipelines, map operator forms.
- `parse_patterns.rs`: rules (`->`), conditions (`/;`), alternatives (`|`), blanks and pattern tests.

Public API & wiring:
- Keep `Parser::from_source`, `parse_all`, `parse_all_detailed`, and `parse_all_with_ranges` unchanged.

Follow-ups:
- Replace `unwrap()/unreachable!` with validated code paths + `expect` messages; negative tests for failure paths.
- (Optional) add a fuzz target for parser (crate-level feature `fuzz` gated).

Crate: lyra-compiler

Current pain: `registry.rs` is a large manually-maintained symbol registry, potential drift vs stdlib.

Proposed:
- Keep structure but add a test that checks parity with stdlib `docs/dispatch`—either by scanning symbols or exposing a generated list from stdlib.
- (Optional) generate parts of the registry from stdlib’s dispatch/doc registry to reduce duplication.

Crate: lyra-rewrite

Current: already well-factored.

Optional:
- Extract simple predicate/type-test helpers into a dedicated file.
- Add an optional “explain matching” hook (behind a feature) to emit steps for Explain integration.

Migration Steps (per crate)
1) Create module files and move code blocks (no logic changes); adjust `mod`/`use` paths.
2) Add `pub use` in root modules to preserve public surfaces.
3) Run `cargo test` workspace-wide; fix imports and visibility issues.
4) Replace targeted `unwrap()` with `expect` where invariants are relied upon; add error returns where user input can violate assumptions.
5) Update docs (`README.md`/`docs/DESIGN.md`) only if any user-visible behavior changes (should be none in phase 1).

CI & Lints (follow-up)
- Enable clippy for selected lints (deny `unwrap_used`/`expect_used` in non-test core paths, allow where invariants are guaranteed and documented).
- Keep `dead_code = "warn"` and periodically prune.

Notes
- This plan is scoped to structure; semantic changes are explicitly out-of-scope for phase 1.
- We can stage refactors crate-by-crate to keep diffs reviewable and minimize regression risk.

