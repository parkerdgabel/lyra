# PROCESS

| Function | Usage | Summary |
|---|---|---|
| `CommandExistsQ` | `CommandExistsQ[cmd]` | Does a command exist in PATH? |
| `Pipe` | `Pipe[cmds]` | Compose processes via pipes |
| `Popen` | `Popen[cmd, args?, opts?]` | Spawn process and return handle |
| `Run` | `Run[cmd, args?, opts?]` | Run a process and capture output |
| `RunContainer` | `RunContainer[]` |  |
| `RuntimeCapabilities` | `RuntimeCapabilities[]` | Supported features and APIs |
| `RuntimeInfo` | `RuntimeInfo[]` | Runtime version and info |
| `Which` | `Which[cmd]` | Resolve command path from PATH |

## `Pipe`

- Usage: `Pipe[cmds]`
- Summary: Compose processes via pipes
- Examples:
  - `Pipe[{{"echo","hello"},{"wc","-c"}}]  ==> <|"Stdout"->"6\n"|>`

## `Run`

- Usage: `Run[cmd, args?, opts?]`
- Summary: Run a process and capture output
- Examples:
  - `Run["echo", {"hi"}]  ==> <|"Status"->0, "Stdout"->"hi\n",...|>`

## `Which`

- Usage: `Which[cmd]`
- Summary: Resolve command path from PATH
- Examples:
  - `Which["sh"]  ==> "/bin/sh"`
