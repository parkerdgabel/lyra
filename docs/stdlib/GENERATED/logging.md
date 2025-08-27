# LOGGING

| Function | Usage | Summary |
|---|---|---|
| `ConfigureLogging` | `ConfigureLogging[opts]` | Configure log level/format/output |
| `GetLogger` | `GetLogger[]` | Get current logger configuration |
| `GitLog` | `GitLog[opts?]` | List commits with formatting options |
| `Log` | `Log[x]` | Natural logarithm |
| `LoginRegistry` | `LoginRegistry[opts?]` | Login to package registry (requires lyra-pm) |
| `LogoutRegistry` | `LogoutRegistry[opts?]` | Logout from package registry (requires lyra-pm) |
| `Logs` | `Logs[id, opts?]` | Stream container logs |
| `SetLogLevel` | `SetLogLevel[level]` | Set global log level |
| `WithLogger` | `WithLogger[meta, body]` | Add contextual metadata while evaluating body (held) |

## `ConfigureLogging`

- Usage: `ConfigureLogging[opts]`
- Summary: Configure log level/format/output
- Examples:
  - `ConfigureLogging[<|"Level"->"debug"|>]  ==> True`

## `GitLog`

- Usage: `GitLog[opts?]`
- Summary: List commits with formatting options
- Tags: git, log
- Examples:
  - `GitLog[<|"Limit"->5|>]  ==> {"<sha>|<author>|...", ...}`

## `Log`

- Usage: `Log[x]`
- Summary: Natural logarithm
- Examples:
  - `Log["info", "service started", <|"port"->8080|>]  ==> True`

## `WithLogger`

- Usage: `WithLogger[meta, body]`
- Summary: Add contextual metadata while evaluating body (held)
- Examples:
  - `WithLogger[<|"requestId"->"abc"|>, Log["info", "ok"]]  ==> True`
