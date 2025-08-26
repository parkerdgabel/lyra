# LOGIC

| Function | Usage | Summary |
|---|---|---|
| `And` | `And[args…]` | Logical AND (short-circuit) |
| `Equal` | `Equal[args…]` | Test equality across arguments |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `Greater` | `Greater[args…]` | Strictly decreasing sequence |
| `GreaterEqual` | `GreaterEqual[args…]` | Non-increasing sequence |
| `Less` | `Less[args…]` | Strictly increasing sequence |
| `LessEqual` | `LessEqual[args…]` | Non-decreasing sequence |
| `Not` | `Not[x]` | Logical NOT |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Or` | `Or[args…]` | Logical OR (short-circuit) |

## `And`

- Usage: `And[args…]`
- Summary: Logical AND (short-circuit)
- Examples:
  - `And[True, False]  ==> False`

## `Equal`

- Usage: `Equal[args…]`
- Summary: Test equality across arguments
- Examples:
  - `Equal[1,1,1]  ==> True`

## `GreaterEqual`

- Usage: `GreaterEqual[args…]`
- Summary: Non-increasing sequence
- Examples:
  - `GreaterEqual[3,2,2]  ==> True`

## `Less`

- Usage: `Less[args…]`
- Summary: Strictly increasing sequence
- Examples:
  - `Less[1,2,3]  ==> True`

## `Not`

- Usage: `Not[x]`
- Summary: Logical NOT
- Examples:
  - `Not[True]  ==> False`

## `Or`

- Usage: `Or[args…]`
- Summary: Logical OR (short-circuit)
- Examples:
  - `Or[False, True]  ==> True`
