# PROCESS

| Function | Usage | Summary |
|---|---|---|
| `CommandExistsQ` | `CommandExistsQ[cmd]` | Does a command exist in PATH? |
| `KillProcess` | `KillProcess[proc, signal?]` | Send signal to process |
| `Pipe` | `Pipe[cmds]` | Compose processes via pipes |
| `Popen` | `Popen[cmd, args?, opts?]` | Spawn process and return handle |
| `ProcessInfo` | `ProcessInfo[proc]` | Inspect process handle (pid, running, exit) |
| `ReadProcess` | `ReadProcess[proc, opts?]` | Read from process stdout/stderr |
| `Run` | `Run[cmd, args?, opts?]` | Run a process and capture output |
| `RunContainer` | `RunContainer[spec, opts?]` | Run a container image; returns id or result. |
| `RuntimeCapabilities` | `RuntimeCapabilities[]` | Supported features and APIs |
| `RuntimeInfo` | `RuntimeInfo[]` | Runtime version and info |
| `WaitProcess` | `WaitProcess[proc]` | Wait for process to exit |
| `Which` | `Which[cmd]` | Resolve command path from PATH |
| `WriteProcess` | `WriteProcess[proc, data]` | Write to process stdin |

## `Pipe`

- Usage: `Pipe[cmds]`
- Summary: Compose processes via pipes
- Tags: process, proc, os
- Examples:
  - `Pipe[{{"echo","hello"},{"wc","-c"}}]  ==> <|"Stdout"->"6\n"|>`

## `ProcessInfo`

- Usage: `ProcessInfo[proc]`
- Summary: Inspect process handle (pid, running, exit)
- Tags: process, proc, introspect
- Examples:
  - `p := Popen["sleep", {"0.1"}]; ProcessInfo[p]`

## `Run`

- Usage: `Run[cmd, args?, opts?]`
- Summary: Run a process and capture output
- Tags: process, proc, os
- Examples:
  - `Run["echo", {"hi"}]  ==> <|"Status"->0, "Stdout"->"hi\n",...|>`

## `Which`

- Usage: `Which[cmd]`
- Summary: Resolve command path from PATH
- Tags: process, proc, os
- Examples:
  - `Which["sh"]  ==> "/bin/sh"`
