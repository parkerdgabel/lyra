Lyra Language — Design and Implementation Plan

Status: Draft for branch `language-rewrite`
Owner: Core Team
Scope: Full language re-architecture, from parser to packages

1. Vision and Principles
- Symbolic-first: A uniform, expression-based core (Expr[head, args]) with WL-inspired semantics.
- Modern ergonomics: pipelines, dot calls, lambdas, string interpolation, slices.
- Concurrency by design: structured concurrency, data-parallelism, actors, streams.
- Interactive-first: a delightful REPL with Explain, rich rendering, and discoverability.
- Compiles small: tree-shaken binaries with capability manifests and provider pluggability.
- Package ecosystem: reproducible, policy-driven, and workspace-friendly.
- Stdlib that “takes algorithms for granted”: high-level APIs auto-select (or explain) algorithms.

2. Language Semantics
- Expressions: `Expr[head, args...]` where `head: Value` (typically Symbol) and `args: Vec[Value]`.
- Values: Integer, Rational, Real(BigFloat), Complex, String, Boolean, Symbol, List, Association(Object), ByteArray, Date/Quantity (extensible), Foreign.
- Patterns: `_`, `_Type`, `x_`, `__`, `___`, `_?p`, `pat /; cond`, `pat1 | pat2`, `pat..`, `pat...`, `Sequence[...]` (for splicing).
- Rules: `lhs -> rhs` (immediate), `lhs :> rhs` (delayed); `Replace`/`ReplaceAll` APIs.
- Rules: `lhs -> rhs` (immediate), `lhs :> rhs` (delayed); rule operators: `expr /. rule` (ReplaceAll), `expr //. rule` (ReplaceRepeated).
- Attributes (per-Symbol): Hold*, Listable, Flat, Orderless, OneIdentity, NumericFunction, Protected/ReadProtected.
- Definitions: OwnValues, DownValues, UpValues, SubValues; pattern-indexed via discrimination nets; deterministic ordering.
- Exactness and Precision: Integer/Rational exact; Real tracks precision; Complex wraps exact/inexact; explicit coercions.

3. Syntax (Reader/Printer)
- WL-like base: `f[x,y]`, `{a,b}`, `<|"k"->v|>`, `(* comments *)`.
- Modern niceties:
  - Pipelines: `expr |> f[opts] |> g`.
  - Dot calls: `obj.method[a,b]`.
  - WL operator forms: `f @ x` (prefix), `x // f` (postfix), `a ~ f ~ b` (infix), `expr /. rule` (ReplaceAll), `expr //. rule` (ReplaceRepeated).
  - Lambdas: `(x,y) => body` and slots `#1 &`.
  - Interpolation: `"sum={Total[x]}"`.
  - Ranges/slices: `a[[i]]`, `a[[i;;j;;k]]`.
  - Option patterns: Associations for options: `<|method->"auto", tolerance->1e-6|>`.
- Pretty-printer: canonicalization for Orderless, compact vs expanded forms, schema-aware rendering of Associations.

4. Evaluation Model
- Normal-order with attribute-aware control (Hold*, Listable, Flat/Orderless canonicalization, OneIdentity).
- Rule application order: UpValues then DownValues when applicable; hygienic scoping forms (With/Module) for rewrite.
- `Sequence` splicing in calls; `Condition` gating on matches.
- Messages/Diagnostics: namespaced; Failure objects (`<|message, tag, args|>`); Explain pipeline.

5. Concurrency Model
- Structured concurrency with cancellation and resource budgets.
- Scopes & Budgets (implemented):
  - `Scope[<|MaxThreads->n, TimeBudgetMs->ms|>, body]` (HoldAll)
    - Applies a per-scope cooperative cancel token, a thread budget (limiter), and a wall-clock deadline (Instant-based) while evaluating `body`.
    - Workers spawned by `Future`, `ParallelMap`, and `ParallelTable` inherit the scope token, limiter, and deadline.
  - `StartScope[opts] -> ScopeId[id]`, `InScope[ScopeId[id], body]`, `CancelScope[ScopeId[id]]`, `EndScope[ScopeId[id]]` for group cancellation and lifecycle control across statements.
  - Current behavior: naive thread-per-task with a shared limiter; cooperative cancellation checked in evaluator and selected primitives (`BusyWait`). Future work: unify with a work-stealing scheduler, broader cooperative checks, and structured lifetimes/cleanup for scope IDs.
- Primitives:
  - Futures/Tasks: `Future[expr, opts?]`, `Await`, `MapAsync[f, list, opts?]`, `Gather`.
  - Data-parallel: `ParallelMap[f, list, opts?]`, `ParallelTable[expr, spec, opts?]`, `ParallelEvaluate[exprs, opts?]`.
  - Channels: `BoundedChannel[n]`, `Send[ch, v]`, `Receive[ch]`, `CloseChannel[ch]` (bounded, blocking with backpressure).
  - Actors: `Actor[(m)=>body]`, `Tell[actor, msg]`, `Ask[actor, msg]` (reply pattern via internal channel), `StopActor[actor]`.
  - Streams: windowed aggregates, joins, CEP; pull-based with backpressure; composition via pipelines. (planned)
- Scheduler: work-stealing pools; cooperative interrupts; deterministic by default for pure compute.
- Distributed: optional remoting provider (local cluster → remote workers) with serialization of Associations and symbols.

6. REPL Experience
- Multiline editor, context-aware completion, parameter/help hints, fuzzy search.
- Inspectors: Symbols (attributes, values), Definitions (Up/Down/Sub), Environments.
- Rich output: inline tables/plots, schema-aware Association pretty-printer.
- Explain: step-by-step evaluation trace, chosen algorithm/provider, heuristics/complexity estimates.
  - Implemented steps include: `ListableThread`, `Hold`, `FlatFlatten`, `OrderlessSort`, `RuleMatch`, `ParallelDispatch`, `ScopeApply`, `ChannelSend`, `ChannelReceive`.
- Session snapshots/time-travel; `%Out[n]`-style references; variable watch.

7. Compiler and Tree Shaking
- IRs:
  - IR0: Expr AST (parsed forms).
  - IR1: Attribute-aware canonical forms (e.g., flattened, sorted for Orderless; list threading extracted).
  - IR2: Rewrite IR (pattern nets, rule sets, match automata, side-condition hooks).
  - IR3: Numeric SSA for kernels (vectorization, precision, bounds; provider lowering hints).
  - Effect graph: side-effects and capabilities (IO/network/gpu/files) for static analysis & tree shaking.
- Pipeline:
  - Parse → Canonicalize → Infer (shapes/precision/effects) → Optimize (inline/simple rules, constant fold, vectorize listable) → DCE/Tree Shaking → Lower (provider-impl) → Backend.
- Backends: Interpreter, AOT native (LLVM/Rust), WASM; mixed-mode JIT for numeric kernels.
- Artifacts: minimal binaries with only reachable stdlib and providers; manifest embeds capabilities and budgets.

8. Package Management
- Manifests: name, version, deps, features, capabilities, budgets, license, platforms.
- Lockfiles: content-addressed, reproducible, deterministic installs.
- Registries: default + custom mirrors; enterprise support.
- Policies: capability prompts/gates (network, fs, processes, gpu), SBOM, signing, CVE scanning.
- Workspaces/Monorepos: shared tooling; multiple crates/modules.
- CLI: search/publish/audit/sync; scriptable tasks.

9. Stdlib Philosophy — “Algorithms Taken for Granted”
- High-level declarative APIs that auto-select algorithms/providers (with `Method->"auto"` default).
- Options: `Method`, `Tolerance`, `PrecisionGoal`, `MaxIterations`, `Parallel`, `Backend/Provider`.
- Providers: capabilities-checked implementations (BLAS/FFTW/GPU/etc.).
- Schema-first: stable lowerCamelCase keys; `Schema[assoc] -> <|name,version,keys|>`.
- `Explain[expr]`: show method selection & cost model.
- Streaming- & concurrency-aware by default.

10. Performance Engineering
- Hash-consed Exprs; arena allocators; structural interning of Symbols/Strings.
- Packed arrays for homogeneous numerics; SIMD; BLAS/GPU providers.
- Precision tracking; exactness preservation; tuned coercion rules.
- Parallel scheduling; copy-on-write; minimal locking.
- Scoped caches/memoization for pure functions; replay support.

11. Security & Capabilities
- Capability system: per-execution and per-package resource permissions (network, fs, subprocess, gpu, time/memory budgets).
- Sandboxes (WASM/native isolates) with policy files; provenance and signed packages.
- Deterministic builds; artifact verification; SBOM.

12. Workspace & Module Layout (proposed)
- `crates/lyra-core`: Value/Expr, Attributes, Patterns, Rules, Explain, Schema, Failure.
- `crates/lyra-parser`: Lexer/Parser for core syntax + EBNF tests.
- `crates/lyra-rewrite`: Rewrite engine & discrimination nets.
- `crates/lyra-runtime`: Evaluator, scheduler, concurrency primitives, streams.
- `crates/lyra-stdlib`: Standard library (modular; provider-aware; schema builders).
- `crates/lyra-compiler`: Analyzer, IRs, optimizer, tree shaker, backends.
- `crates/lyra-providers`: BLAS/FFTW/GPU/etc. providers (feature-gated).
- `crates/lyra-repl`: Interactive shell toolkit.
- `crates/lyra-pm`: Package manager & registries.
- `tools/*`: lints, formatters, docgen, snapshots.

13. Testing Strategy
- Unit tests per crate (parser, attrs, rewrite, concurrency, providers).
- Golden tests for printer/REPL renders and schema outputs.
- Concurrency determinism tests and deadlock/livelock stress.
- Package PM integration tests: lockfiles, policies, capabilities.
- Performance baselines: vector ops, FFT, Solve-and-simplify microbenchmarks.

14. Phased Implementation Plan

Phase 0 — Bootstrap (Weeks 1–2)
- Deliverables:
  - `lyra-core`: Value/Expr; Attributes stub; Failure; Schema[]; basic pretty-printer.
  - `lyra-parser`: core grammar, parser, printer round-trips; EBNF tests.
  - `lyra-repl`: minimal REPL with multiline, history, `?help` and `Explain` stub.
  - Repo scaffolding, CI (fmt/clippy/tests), docs skeleton.
- Acceptance:
  - Parse/print round-trip for literals, lists, associations, calls, simple patterns.
  - REPL accepts, evaluates `Plus/Times` via stub evaluator; Schema[] works on Associations.

Phase 1 — Evaluation + Concurrency Foundations (Weeks 3–5)
- Deliverables:
  - `lyra-runtime`: attribute-aware evaluator (Hold*, Listable, Orderless), Sequence splicing, Condition.
  - Concurrency primitives: Future/Await, ParallelMap; basic `Scope` with `MaxThreads` and `TimeBudgetMs`; cooperative cancellation points.
  - `lyra-stdlib` v0: lists/strings/numeric basics; schema builders; Explain minimal.
- Acceptance:
  - Correct evaluation for attribute subset; Listable threading tests pass.
  - ParallelMap speedup on CPU; `Scope` throttles concurrency; time budget failures are reported; cancellation/cooperative checks; basic Explain trace.

Phase 2 — Rewrite Engine + Numeric Tower (Weeks 6–9)
- Deliverables:
  - `lyra-rewrite`: discrimination nets; rule indexing; Up/Down/Sub values.
  - Numeric tower: Rational/BigReal/Complex semantics; packed arrays; vectorized listables.
  - Expand stdlib: FFT/filters, regression, timeseries basics.
- Acceptance:
  - Rule-application microbenchmarks; coverage for pattern constructs.
  - Numeric exactness tests; packed-array vectorization demonstrable.

Phase 3 — Analyzer, Tree Shaking, Providers (Weeks 10–13)
- Deliverables:
  - `lyra-compiler`: dep/effect graph; tree shaker; provider interfaces.
  - Providers: at least one BLAS and one FFT provider; CPU default.
  - AOT stub backend (native or WASM) for small programs.
- Acceptance:
  - Small apps compile to minimal binaries; unused stdlib chopped; provider switch via options.

Phase 4 — Packages, Policies, REPL UX (Weeks 14–17)
- Deliverables:
  - `lyra-pm`: manifest/lockfile resolver, install/uninstall, policies (capabilities), signing stub.
  - REPL: doc popovers, completion, inspectors, rich rendering, session snapshots.
- Acceptance:
  - Reproducible installs; capability prompts; publishing to local registry.
  - REPL passes UX smoke (help, inspectors, Explain with chosen algorithms).

Phase 5 — Advanced Stdlib + Distributed + AOT (Weeks 18–24)
- Deliverables:
  - Solve/Integrate/Graph/RAG/ML high-level functions with auto-method selection.
  - Distributed actors/workers; streaming CEP; dashboarding hooks.
  - Harden AOT; WASM target; notebooks/VSCode integration stubs.
- Acceptance:
  - End-to-end demos: streaming analytics, ML pipeline, RAG query; small deployable binaries.

15. Risks and Mitigations
- Scope creep: strictly adhere to phase deliverables; feature flags for experimental work.
- Performance regressions: benchmarks per PR; packed arrays & vectorization early.
- Concurrency complexity: structured primitives only; narrow unsafe/sync surface.
- Package security: capabilities off by default; audit/sbom baked; staged rollouts.
- Provider sprawl: define minimal Provider API; start with few high-value backends.

16. Coding Standards & Conventions
- Stable Association schemas (lowerCamelCase); Schema[] must identify public outputs.
- Attributes as bitflags; symbols interned; structural interning for Exprs.
- Avoid global mutability; explicit context for evaluator; pure by default.
- Consistent error model via Failure associations and diagnostics.

17. Open Questions (track in issues)
- How deep should WL-compatibility go (edge forms, boxes)?
- JIT strategy: immediate or later? Which kernels first?
- Provider ABI stability and versioning.
- Notebook integration format (Jupyter vs custom).

Appendix A — Example Association Schemas
- SpectralResult/v1: <|frequencies, magnitudes, phases?, sampleRate, method|>
- FilterResult/v1: <|filterType, parameters, success, message, filteredSignal|>
- RegressionResult/v1: <|coefficients, rSquared, method, residuals|>

Appendix B — Explain[expr] Minimal Contract
- Returns <|steps: List[Assoc], algorithm?: String, provider?: String, estCost?: Assoc|>.
- Steps include: rule matches, attribute actions (Hold/Listable), provider/lowering notes.

Tiny example (current prototype):

```
Explain[Plus[{1,2,3}, 10]]
=> <|"steps" -> {<|"action" -> "ListableThread", "head" -> Plus, "count" -> 3|>},
      "algorithm" -> "stub", "provider" -> "cpu", "estCost" -> <||>|>

Explain[OrderlessEcho[c, a, b]]
=> <|"steps" -> {<|"action" -> "OrderlessSort", "head" -> OrderlessEcho,
                  "finalOrder" -> {a, b, c}|>},
      "algorithm" -> "stub", "provider" -> "cpu", "estCost" -> <||>|>
```
