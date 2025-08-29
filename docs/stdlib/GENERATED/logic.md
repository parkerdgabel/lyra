# LOGIC

| Function | Usage | Summary |
|---|---|---|
| `And` | `And[args…]` | Logical AND (short-circuit) |
| `Do` | `Do[body, n]` | Execute body n times. |
| `Equal` | `Equal[args…]` | Test equality across arguments |
| `EvenQ` | `EvenQ[n]` | Is integer even? |
| `For` | `For[init, test, step, body]` | C-style loop with init/test/step. |
| `Greater` | `Greater[args…]` | Strictly decreasing sequence |
| `GreaterEqual` | `GreaterEqual[args…]` | Non-increasing sequence |
| `Less` | `Less[args…]` | Strictly increasing sequence |
| `LessEqual` | `LessEqual[args…]` | Non-decreasing sequence |
| `Not` | `Not[x]` | Logical NOT |
| `OddQ` | `OddQ[n]` | Is integer odd? |
| `Or` | `Or[args…]` | Logical OR (short-circuit) |
| `While` | `While[test, body]` | Repeat body while test evaluates to True. |

## `And`

- Usage: `And[args…]`
- Summary: Logical AND (short-circuit)
- Tags: logic
- Examples:
  - `And[True, False]  ==> False`

## `Do`

- Usage: `Do[body, n]`
- Summary: Execute body n times.
- Tags: logic, control
- Examples:
  - `i:=0; Do[i:=i+1, 3]; i  ==> 3`

## `Equal`

- Usage: `Equal[args…]`
- Summary: Test equality across arguments
- Tags: logic
- Examples:
  - `Equal[1,1,1]  ==> True`

## `For`

- Usage: `For[init, test, step, body]`
- Summary: C-style loop with init/test/step.
- Tags: logic, control
- Examples:
  - `i:=0; For[i:=0, i<3, i:=i+1, Null]; i  ==> 3`

## `GreaterEqual`

- Usage: `GreaterEqual[args…]`
- Summary: Non-increasing sequence
- Tags: logic
- Examples:
  - `GreaterEqual[3,2,2]  ==> True`

## `Less`

- Usage: `Less[args…]`
- Summary: Strictly increasing sequence
- Tags: logic
- Examples:
  - `Less[1,2,3]  ==> True`

## `Not`

- Usage: `Not[x]`
- Summary: Logical NOT
- Tags: logic
- Examples:
  - `Not[True]  ==> False`

## `Or`

- Usage: `Or[args…]`
- Summary: Logical OR (short-circuit)
- Tags: logic
- Examples:
  - `Or[False, True]  ==> True`

## `While`

- Usage: `While[test, body]`
- Summary: Repeat body while test evaluates to True.
- Tags: logic, control
- Examples:
  - `i:=0; While[i<3, i:=i+1]; i  ==> 3`
