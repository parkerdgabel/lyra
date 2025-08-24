# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Rust sources (VM in `vm.rs`, bytecode in `bytecode.rs`, linker/registry in `linker.rs`).
- `src/tree_shaking/`: Tree‑shaking analyzers and pipeline.
- `tests/`: Integration tests; `benches/`: Criterion benchmarks; `examples/`: runnable samples.
- `docs/`: Architecture notes and reports; `scripts/`: helper tooling.

## Build, Test, and Development Commands
- Build: `cargo build` (release: `cargo build --release`).
- Test: `cargo test` (filter: `cargo test <name>`).
- Benchmarks: `cargo bench` (compare results in `benchmark_results/`).
- Lint: `cargo clippy --all-targets -- -D warnings`.
- Format: `cargo fmt --all`.
- Run example: `cargo run --example minimal`.

## Coding Style & Naming Conventions
- Rust 2021, rustfmt enforced (`rustfmt.toml` + pre‑commit).
- Use `snake_case` for functions/vars, `CamelCase` for types, `SCREAMING_SNAKE_CASE` for consts.
- Prefer small, focused modules; keep public APIs in `mod.rs` or top‑level files.
- Error types implement `thiserror::Error`; prefer `Result<T, E>` aliases.

## Testing Guidelines
- Integration tests live under `tests/` (descriptive filenames, e.g., `vm_execution.rs`).
- Co‑locate small unit tests with source via `#[cfg(test)]` when helpful.
- Cover: opcode semantics, bytecode encode/decode, registry dispatch, tree‑shaking passes.
- Run `cargo test` locally; keep tests deterministic and fast.

## Commit & Pull Request Guidelines
- Commits: imperative mood with scope when helpful (e.g., `vm: optimize CallStatic dispatch`).
- PRs: include purpose, key changes, verification steps (commands/output), and linked issues.
- Include benchmarks or before/after numbers for performance work; attach profiler notes if non‑trivial.

## Security & Configuration Tips
- No network access in core VM/bytecode paths; guard feature‑gated integrations.
- Validate indices and operands; favor explicit `u16/u32` bounds checks.
- Use `cargo audit`/`cargo deny` if adding dependencies.
