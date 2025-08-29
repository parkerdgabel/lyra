Tree Shaking — Implementation Plan

Status: Draft v1
Owner: Core Team
Scope: Minimal binaries via symbol-level and feature-level dead code elimination

Goals
- Produce minimal runtime binaries that only include reachable stdlib functions and optional providers.
- Emit a capabilities manifest derived from the program’s reachable effects (fs, net, db, gpu, process, time).
- Provide a CLI workflow to analyze, configure, and build minimal artifacts.

Non‑Goals (for v1)
- Full IR2/IR3 optimizations or JIT. We focus on reachability and registration shaping.
- Perfect precision for dynamic code that constructs heads at runtime; we handle via conservative widening.

Current State (as of this doc)
- Parser lowers syntactic sugar to symbolic heads (Replace/Rule/Apply/etc.).
- Stdlib registration is module‑level via `register_all`, gated by Cargo features.
- Tools export respects present builtins post‑registration, but registration itself is coarse (module level).
- No `lyra-compiler` crate; no analyzer pipeline yet.

Overview of Approach
1) Analyzer: parse entry sources and collect reachable heads (symbols). Detect dynamic hazards and collect effects.
2) Registration: add selective registration in stdlib so only needed functions are registered.
3) Features & providers: drive Cargo features from analyzer to exclude whole modules/dependencies.
4) Build tooling: CLI (`lyra-shake`) that outputs a manifest and optionally generates a tiny runner using selective registration.
5) Tests and size benchmarks: ensure correctness and quantify size wins.

Crate Changes
- New: `crates/lyra-compiler`
  - `analyzer.rs`: IR0 walker (uses `lyra-parser`) to collect heads and effects.
  - `registry.rs`: static maps Symbol → { moduleFeatures[], effects[] }.
  - `manifest.rs`: types + JSON in/out for {symbols[], features[], capabilities[]}.
  - `bin/lyra-shake.rs`: CLI for analyze/build flows.
- Update: `crates/lyra-stdlib`
  - Add macro `register_if!` and per-module filtered registrars.
  - Add `register_selected(ev: &mut Evaluator, names: &HashSet<&str>)`.
  - Keep `register_all` unchanged for REPL/tests.

Analyzer Details (lyra-compiler)
- Input sources: one or more `.lyra` files or inline program string.
- Parse: use `lyra-parser` → IR0 (Value AST).
- Walk rules:
  - Collect every `Expr[head, …]` where `head` is `Symbol(name)` as a referenced head.
  - For `Rule[lhs, rhs]` or rule-bearing definitions:
    - Include all heads found in `rhs`.
    - Include the outermost head of `lhs` as callable (Down/Up/SubValues roots).
  - Normalize through common syntactic heads (already lowered by parser): Replace, ReplaceAll, Apply, Map, With, Set, etc.
  - Always keep evaluator core heads (see Core Keep Set below).
- Dynamic hazards (conservative widening):
  - If the AST contains patterns that construct symbols dynamically (e.g., parsing strings into code or unknown heads in symbol position), widen to entire modules via configuration flags or heuristics.
- Effects capture:
  - For each referenced head, look up its effects: ["net", "fs", "db", "gpu", "process", "time"].
  - Aggregate to capabilities manifest.
- Output manifest (JSON):
  - symbols: string[] — unique set of referenced heads.
  - features: string[] — derived Cargo feature set for stdlib/providers.
  - capabilities: string[] — aggregated effects tags.

Core Keep Set
- Always include: core evaluator heads and minimal plumbing used by many programs.
- Suggested defaults: `core`, `tools`, `schema`, `explain`, and `DescribeBuiltins` via `register_introspection`.
- Configurable: CLI flag to opt out for extreme minimal builds.

Symbol→Feature & Effects Registry (compiler side)
- Examples (illustrative, maintained centrally in registry.rs):
  - "StringLength" → features: ["string"], effects: []
  - "Split" → ["string"], []
  - "HttpGet" → ["net_https"], ["net"]
  - "SqliteQuery" → ["db_sqlite"], ["fs","db"]
  - "ImageRead" → ["image"], ["fs"]
  - "ParallelMap" → ["concurrency"], ["time"] (budget‑related)
- Keep this single source of truth in `lyra-compiler` and unit‑test it.

Stdlib Registration Changes (symbol‑level)
- New macro in `lyra-stdlib`:
  - `register_if!(ev, filter, "Head", func_impl, attrs)` registers if `filter("Head")` is true.
- Per‑module filtered registrars:
  - `register_string_filtered(ev, filter)`; same for list, math, assoc, functional, etc.
  - Replace existing `ev.register("Name", ...)` with `register_if!` calls.
- New API:
  - `pub fn register_selected(ev: &mut Evaluator, names: &std::collections::HashSet<&str>)`
    - Internally builds a predicate from `names` and calls each `register_*_filtered` behind active Cargo features.
- Backwards compatible:
  - `register_all` calls the unfiltered registrars to preserve current behavior.

CLI Workflows (lyra-shake)
- Analyze only:
  - `lyra-shake analyze path/to/main.lyra -o build/lyra.manifest.json`
  - Prints size‑impact summary and capabilities.
- Build minimal runner:
  - `lyra-shake build path/to/main.lyra --output target/minapp`
  - Steps:
    1) Analyze sources → manifest (symbols, features, capabilities)
    2) Generate `generated/registrar.rs` which exposes `register_minimal(ev)` delegating to `register_selected`
    3) Cargo build with `--no-default-features -F <derived features>` and LTO/z‑opt flags
    4) Emit `capabilities.json` next to the binary

Generated Registrar (sketch)
```
// generated/registrar.rs
use std::collections::HashSet;
pub fn register_minimal(ev: &mut lyra_runtime::Evaluator) {
    let mut keep: HashSet<&'static str> = HashSet::new();
    // filled by analyzer
    keep.extend([
        "StringLength", "Split", // …
    ]);
    lyra_stdlib::register_selected(ev, &keep);
}
```

Build Flags for Size
- `RUSTFLAGS="-C opt-level=z -C strip=symbols -C codegen-units=1"`
- `cargo build --release -Z build-std=std,panic_abort -Z build-std-features=panic_immediate_abort` (optional nightly)
- Cargo profiles can be tuned in workspace if desired.

Testing Strategy
- Unit tests (compiler): analyzer collects expected heads from small programs; manifests compare equal.
- Golden tests: tiny programs built as minimal runners; run and assert outputs.
- Size benchmarks: record binary sizes for representative samples before/after.
- Safety: when runtime hits unknown head, produce a clear Failure suggesting re‑build with widened set.

Dynamic Hazards & Fallbacks
- Detection examples: heads coming from variables of unknown value in call position; string→code eval patterns.
- Policies:
  - `--assume-dynamic none|string|all` controls widening scope.
  - `--keep-modules string,list,…` to override analyzer decisions.
  - On hazard, default to widen to module(s) most likely affected; log a warning.

Capabilities Manifest (JSON)
```
{
  "capabilities": ["net", "fs"],
  "symbols": ["HttpGet", "Split"],
  "features": ["net_https", "string"]
}
```
- Used by packaging/policy to prompt or gate permissions.

Milestones
1) Analyzer MVP: head collection, registry mapping, manifest output
2) Stdlib selective registration for: string, list, math
3) CLI build mode with generated registrar, features wiring
4) Effects/capabilities aggregation and manifest
5) Expand selective registration to assoc, functional, text, ndarray
6) Provider gating: net/db/image/audio/crypto
7) Tests + size baselines + docs polish

Open Questions
- Granularity for evaluator internals in the Core Keep Set.
- How aggressive to be in default hazard handling.
- Registry maintenance ergonomics and CI checks.

References
- See `docs/DESIGN.md` (Compiler and Tree Shaking) for conceptual pipeline.

Quickstart
- Build the CLI: `cargo build -p lyra-compiler --release`
- Analyze a sample: `target/release/lyra-shake analyze examples/tree_shake/sample.lyra`
- Output example:
```
{
  "symbols": ["StringJoin", "StringLength"],
  "features": ["string"],
  "capabilities": []
}
```
