Process Management

Overview
- Spawn processes, capture output, pipeline commands, and stream IO.

Functions
- Run[cmd, args?, opts?] -> <|Status, Stdout, Stderr, TimedOut|> | Failure
  - cmd: String/Symbol; args: List of strings/symbols.
  - opts: <|Cwd->path, Env-><|K->V|>, TimeoutMs->ms, Input->String|>.
- Which[name] -> path|Null
- CommandExistsQ[name] -> Boolean
- Popen[cmd, args?, opts?] -> <|__type:"Process", pid, id|> | Failure
- WriteProcess[proc, data] -> Integer|Failure
- ReadProcess[proc, opts?] -> String|Failure
  - opts: <|Stream->"stdout"|"stderr"|> (reads all available data)
- WaitProcess[proc, timeoutMs?] -> <|Status|> | Failure("Process::timeout")
- KillProcess[proc, signal?] -> True|Failure
- Pipe[{ {cmd,args}, {cmd,args}, ... }] -> Run-like result

Failures
- Process::run, ::spawn, ::io, ::timeout, ::signal, ::notfound.

Examples
- Run["echo", {"hello"}] // returns Stdout -> "hello\n"
- Which["bash"], CommandExistsQ["git"]
- p = Popen["cat"]; WriteProcess[p, "hi"]; ReadProcess[p]; WaitProcess[p]
- Pipe[{{"echo", {"hello"}}, {"sed", {"s/hello/world/"}}}]

Notes
- Env in Run/Popen sets environment variables only for that child process.
- ReadProcess performs a full read of the selected stream; for incremental read, prefer small child processes or future streaming extensions.

