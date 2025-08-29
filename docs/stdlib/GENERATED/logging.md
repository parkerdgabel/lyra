# LOGGING

| Function | Usage | Summary |
|---|---|---|
| `GitLog` | `GitLog[opts?]` | List commits with formatting options |
| `Log` | `Log[x]` | Natural logarithm. Tensor-aware: elementwise on tensors. |
| `Logger` | `Logger[opts?]` | Create/configure a logger (global). |
| `LoginRegistry` | `LoginRegistry[opts?]` | Login to package registry (requires lyra-pm) |
| `LogoutRegistry` | `LogoutRegistry[opts?]` | Logout from package registry (requires lyra-pm) |
| `Logs` | `Logs[id, opts?]` | Stream container logs |
| `WithLogger` | `WithLogger[meta, body]` | Add contextual metadata while evaluating body (held) |

## `GitLog`

- Usage: `GitLog[opts?]`
- Summary: List commits with formatting options
- Tags: git, log
- Examples:
  - `GitLog[<|"Limit"->5|>]  ==> {"<sha>|<author>|...", ...}`

## `Log`

- Usage: `Log[x]`
- Summary: Natural logarithm. Tensor-aware: elementwise on tensors.
- Examples:
  - `Log[E]  ==> 1`
  - `Log[Tensor[{1,E}]]  ==> Tensor[...]`

## `Logger`

- Usage: `Logger[opts?]`
- Summary: Create/configure a logger (global).
- Examples:
  - `logger := Logger[<|"Level"->"debug"|>]  ==> <|__type->"Logger",Name->"default"|>`

## `WithLogger`

- Usage: `WithLogger[meta, body]`
- Summary: Add contextual metadata while evaluating body (held)
- Examples:
  - `WithLogger[<|"requestId"->"abc"|>, LogMessage["info", "ok"]]  ==> True`
