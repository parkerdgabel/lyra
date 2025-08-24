Lyra Core Language Rewrite RFC (Phase 1)

Overview
- Replace ad-hoc runtime types with a uniform Expr-based model and a richer numeric tower.
- Keep Foreign objects only for stateful/external resources.
- Preserve meta-programming via rules, patterns, and attributes.

Core Runtime Types
- Value: single enum covering:
  - Integer, Real, Rational(n,d), BigReal{value,precision}, Complex{re,im}
  - String, Symbol
  - List[Value]
  - Object (Association)
  - Expr{head, args} for Head[args] expressions
  - Rule, Pattern, Quote, PureFunction, Slot
  - LyObj for Foreign resources

Evaluation Semantics (target)
- Normal evaluation honoring Attributes (Hold*, Listable, Flat, Orderless, NumericFunction).
- Rule application precedence: UpValues before DownValues where applicable.
- Sequence splicing during application; Condition (expr /; test) gating rewrites.

Numeric Semantics
- Exactness preserved via Integer/Rational; Complex wraps exact/inexact components.
- BigReal tracks precision; numeric kernels should propagate precision.

Associations
- Public return types use Object with lowerCamelCase keys.
- Schema[] provides best-effort schema detection for Objects.

Migration Plan
- Phase 1 (this PR):
  - Add Value variants: Rational, BigReal, Complex, Expr.
  - Keep legacy variants for compatibility.
  - Provide constructors and serde/hash/eq support.
  - No breaking changes to stdlib callers.
- Phase 2:
  - Parser emits Expr directly for core constructs.
  - Evaluator loop lifts Function/Symbol calls into Expr dispatch.
  - Introduce Attributes on symbols and basic rewrite engine.
- Phase 3:
  - Replace ad-hoc Function(String) with Symbol + DownValues tables.
  - Implement discrimination nets for pattern indexing.

Testing & Lints
- `schema_lints` binary to enforce Association schema conventions.
- `parser_conformance` binary to sanity-check grammar coverage.

Notes
- BigInt dependencies are deferred; placeholder BigReal captures precision without external crates.
- Future work: swap placeholder with real big-int/real types and vectorized packed arrays.

