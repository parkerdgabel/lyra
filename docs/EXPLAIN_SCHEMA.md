Explain Step Schema (v0)

Status: Draft, stabilized for current prototype

Overview
- Explain produces an Association with keys:
  - "steps": list of step associations
  - "algorithm": string (stub)
  - "provider": string
  - "estCost": association (reserved)

Step shape
- Each step is an Association with required keys:
  - "action": string — the event kind
  - "head": value — the function head or context symbol
  - "data": association — action-specific fields

Actions and data fields
- ListableThread
  - data.count: integer — number of elements
  - data.argLens: list[integer] — lengths of list arguments (0 for scalars)
- Hold
  - data.held: list[integer] — 1-based positions held
- FlatFlatten
  - data.added: integer — number of inner args flattened
- OrderlessSort
  - data.finalOrder: list[value] — canonical argument order
- RuleMatch
  - head: Replace (Replace/ReplaceAll/ReplaceFirst context)
  - data.lhs: value — matched left-hand pattern
  - data.rhs: value — replacement right-hand side

- ConditionEvaluated
  - head: PatternTest or Condition — the evaluated construct
  - data.result: boolean — whether the predicate/condition evaluated to True
  - data.bindsCount: integer — number of variable bindings in scope (Condition only)
  - data.expr: value — substituted condition expression (Condition only)
  - data.pred: value — predicate head (PatternTest only)
  - data.arg: value — tested argument (PatternTest only)

Notes
- More actions may be added over time (e.g., provider selection, method heuristics).
- Fields are stable; additional keys may appear under data but existing names will not change.

Examples

1) PatternTest inside Replace

```
Explain[Replace[2, _?EvenQ -> 9]]
=> <|"steps" -> {<|"action" -> "ConditionEvaluated", "head" -> PatternTest,
                  "data" -> <|"pred" -> EvenQ, "arg" -> 2, "result" -> True|>|>},
      "algorithm" -> "stub", "provider" -> "cpu", "estCost" -> <||>|>
```

2) Condition in DownValues

```
SetDownValues[f, { f[x_] /; x > 1 -> 0 }];
Explain[f[2]]
=> <|"steps" -> {<|"action" -> "RuleMatch", "head" -> f,
                  "data" -> <|"lhs" -> Condition[f[x_], Greater[x, 1]], "rhs" -> 0|>|>,
                 <|"action" -> "ConditionEvaluated", "head" -> Condition,
                  "data" -> <|"expr" -> Greater[2, 1], "bindsCount" -> 1, "result" -> True|>|>},
      "algorithm" -> "stub", "provider" -> "cpu", "estCost" -> <||>|>
```

