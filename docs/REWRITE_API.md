Lyra Rewrite API — Sketch (v0)

Goals
- Centralize pattern-based rewriting (rules, definitions, matching) separate from the evaluator/runtime.
- Provide a clean API that the runtime can call for rule application and (later) symbol definitions (Own/Down/Up/Sub values).

Crate: lyra-rewrite
- Modules:
  - rule: Rule, RuleSet, Delayed
  - defs: DefinitionStore { Own, Down, Up, Sub } maps symbol -> RuleSet
  - matcher: match_rule/match_rules, Bindings (initial subset of pattern language)
  - engine: rewrite_once/rewrite_all/rewrite_with_limit (expression traversal, substitution)

API Entry Points
- rule::Rule { lhs, rhs, delayed }
- defs::DefinitionStore with rules(kind, sym) and rules_mut(kind, sym)
- matcher::{match_rule, match_rules}
- engine::{rewrite_once, rewrite_all, rewrite_with_limit}

Reader/Parser Operator Forms
- `expr /. rule` → `ReplaceAll[expr, rule]` (single-pass traversal using matcher; applies across lists/exprs)
- `expr //. rule` → `ReplaceRepeated[expr, rule]` (repeated application to fixed point)
- Named typed blanks: parser accepts WL shorthand like `x_Integer`; matcher binds `x` when type test passes.

Future Work
- Discrimination nets for fast rule indexing
- Full pattern language coverage (BlankSequence, Condition, alternatives with precedence, etc.)
- Integration with evaluator attributes (Hold*, Listable) and value ordering
- UpValues/DownValues/SubValues lookup resolution and deterministic ordering
- Explain hooks for rule selection and side-condition evaluation
