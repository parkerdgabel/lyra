Lyra Phase 2 — Tracking TODOs

Scope: Rewrite Engine + Numeric Tower (Weeks 6–9)
Owner: Core Team (assign per task)
Status: In progress

Milestones

M1 — Rewrite Core
- [ ] Discrimination nets: builder, insert/remove, candidate matching API
- [ ] Integrate nets with matcher/engine (fallback to linear when small)
- [ ] DefinitionStore wiring in runtime: Set/Unset/Get for Own/Down/Up/Sub
- [ ] Deterministic lookup order (UpValues → DownValues) with attributes
- [ ] Canonicalization before match (Orderless sort, Flat flatten, OneIdentity)
- [ ] Pattern coverage: BlankSequence/BlankNullSequence/Named*, Alternative precedence, Condition, PatternTest, Repeated/RepeatedNull, Optional, SlotSequence, Sequence splicing
- [ ] Explain hooks: RuleMatch (lhs/rhs), ConditionEvaluated (result, binds)
- [ ] Tests: rule resolution precedence, complex sequence matching, Explain golden
- [ ] Bench: 1k-rule microbench (target ≥5x vs linear scan)

M2 — Numeric Base
- [ ] Value variants (feature-gated initially): Rational, BigReal, Complex
- [ ] Constructors, printers, schema entries for new numeric types
- [ ] Coercions: Integer↔Rational↔Real↔Complex rules; exactness/precision metadata
- [ ] Arithmetic ops: Plus/Times/Abs/Min/Max across mixed numeric types
- [ ] Big number backend adapter (feature: `big-num`) with deterministic behavior
- [ ] Tests: exactness, precision propagation, complex arithmetic basics

M3 — Packed Arrays + Vectorization
- [ ] PackedArray<T> representation + shape metadata
- [ ] Coercion from homogeneous lists; validation utilities
- [ ] Broadcasting semantics with scalars and mixed shapes
- [ ] Vectorized listables for core arithmetic (SIMD or provider hooks)
- [ ] Fallback to list threading for mixed/unsupported types
- [ ] Tests: creation/coercion, broadcasting correctness, perf sanity

M4 — Stdlib Numeric (Phase 2 set)
- [ ] FFT (real/complex) + IFFT, windowing options (schema-first)
- [ ] Filters (FIR/IIR) stubs with validated options and results schema
- [ ] Regression (OLS) with result schema: coefficients, rSquared, residuals
- [ ] Timeseries: RollingMean/Median/Sum, Resample (down/upsample)
- [ ] Tests: known-vector FFT, IFFT roundtrip, regression small datasets, window boundaries

M5 — Performance & Benchmarks
- [ ] Rewrite microbench suite (match rates, depth, nets vs linear)
- [ ] Vectorization benchmarks (f64 buffers vs scalar threading)
- [ ] CI gates for perf regressions (budget-based thresholds)

M6 — Docs & Examples
- [ ] Rewrite API reference and patterns guide
- [ ] Numeric tower & exactness guide; packed arrays usage
- [ ] Examples: rule-based transforms; exact arithmetic; vectorized ops
- [ ] Explain action docs (RuleMatch, ConditionEvaluated) with examples

Cross-Cutting
- [ ] Capability manifests for numeric providers (future Phase 3 tie-in)
- [ ] Error messages and Failures for non-convergent operations
- [ ] Golden snapshots for printers of Rational/Complex

Acceptance Criteria Summary
- Rewrite: deterministic rule selection; >5x speedup on 1k rules
- Numeric: exactness preserved; precision tracked; complex ops correct
- Packed arrays: broadcast semantics correct; vectorized ops faster than scalar threading
- Stdlib: FFT/IFFT roundtrip; regression within tolerance on fixtures
- Explain: stable schema for new steps; tests cover representative flows

Notes / Risks
- Big number backend choice to remain feature-gated initially
- Pattern backtracking bounded; prefer nets narrowing
- Packed array memory layout to stay stable across platforms

Owners & Estimates (fill in)
- M1: ____ (2–3w)
- M2: ____ (1–2w, in parallel after M1 API stabilizes)
- M3: ____ (1–2w)
- M4: ____ (1–2w)
- M5: ____ (ongoing; initial 2–3d)
- M6: ____ (1w overlapping)

