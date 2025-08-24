Phase 1: Schema + Grammar Checks

Run these lightweight checks locally. They do not require full `cargo test`:

- Schema lints: ensures normalized modules avoid `Value::Dict`, snake_case keys, and Foreign returns.
  Command: `cargo run --bin schema_lints`

- Parser conformance: parses a set of core language samples against the real parser.
  Command: `cargo run --bin parser_conformance`

- Integration smoke: quick Association-return checks for selected stdlib functions.
  Command: `cargo run --bin integration_smoke`

Notes
- Lints currently scope to normalized modules: `numerical/`, `signal/`, `analytics/timeseries.rs`, `number_theory/`, `ai_ml/vector_store.rs`, `ai_ml/embeddings.rs`.
- Expand scope incrementally as modules are normalized.

