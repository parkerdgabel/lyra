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

Notes
- More actions may be added over time (e.g., provider selection, method heuristics).
- Fields are stable; additional keys may appear under data but existing names will not change.

