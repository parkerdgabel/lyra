# Repository Guidelines

## Project Structure & Modules
- Root Cargo workspace: `Cargo.toml` manages crates.
- Core crates:
  - `crates/lyra-core/`: core `Value`, pretty printer, schema.
  - `crates/lyra-parser/`: parser and lexer (`src/parser.rs`), tests in `tests/`.
  - `crates/lyra-runtime/`: evaluator, attributes, stdlib (`src/eval.rs`, `src/attrs.rs`), tests in `tests/`.
  - `crates/lyra-repl/`: minimal REPL (`src/main.rs`).
- Docs: `docs/DESIGN.md`, user tips in `README.md`.

## Build, Test, and Development
- Build all crates: `cargo build` (use `--release` for optimized binaries).
- Run tests: `cargo test` (workspace-wide). Run per crate: `cargo test -p lyra-runtime`.
- Run REPL: `cargo run -- repl`.
- Quick formatting: `cargo fmt` (uses `rustfmt.toml`).
- Lint (optional): `cargo clippy --all-features -- -D warnings`.

## Coding Style & Naming
- Rust edition defaults; follow standard Rust style (4-space indent, snake_case for functions/vars, CamelCase for types).
- Keep changes minimal and localized. Prefer pure functions and explicit context (no globals).
- Attributes live in `lyra-runtime/src/attrs.rs` (bitflags). Register new builtins in `Evaluator::new` and implement as `fn name(ev: &mut Evaluator, args: Vec<Value>)`.
- Prefer small, composable helpers; mirror existing patterns (e.g., listable threading, orderless sorting, failure associations).

## Testing Guidelines
- Use Rust’s built-in test framework. Place crate-level tests under `crates/<name>/tests/` (e.g., `runtime_smoke.rs`, `concurrency.rs`).
- Add focused tests for new syntax (parser) and semantics (runtime). Example: `cargo test -p lyra-parser`.
- Keep tests deterministic; avoid network and long sleeps. Use small inputs and clear assertions (stringified via pretty printer when helpful).

## Commit & Pull Request Guidelines
- Commits: concise, imperative subject (e.g., "Add Max/Min and Explain steps"), with a short body when rationale helps. Group related changes.
- PRs: include description, motivation, and scope; link issues; add before/after snippets or test output when relevant.
- Checklist: updated docs (`README.md`/`docs/DESIGN.md`) when user-visible behavior changes; added/updated tests; `cargo fmt` and `cargo test` pass.

## Architecture Notes
- Phase-oriented roadmap in `docs/DESIGN.md`. Explain traces and stdlib evolve iteratively; prefer enriching Explain with structured steps and stable association keys.
