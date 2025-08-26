Logging

Overview
- Structured logging with level control and JSON or text output.
- Per-scope contextual fields via WithLogger.

Functions
- ConfigureLogging[opts] -> True|Failure
  - opts: <|Level->"info", Format->"text"|"json", File-><|Path|>, IncludeTime->True|False, IncludeSpan->True|False|>.
  - If File.Path is set, logs are appended to that file; otherwise stderr.
- Log[level, message, meta?] -> True|Failure
  - level: "trace"|"debug"|"info"|"warn"|"error".
  - message: String or Symbol; meta: assoc of fields.
- WithLogger[contextAssoc, body] (HOLD_ALL)
  - Adds structured fields for all Log calls within body in this thread.
- SetLogLevel[level], GetLogger[]

Failures
- Log::config: invalid configuration.
- Log::emit: write error.

Examples
- ConfigureLogging[<|Level->"info", Format->"json"|>]
- WithLogger[<|requestId->"abc123"|>, Log["info", "hello", <|user->"u1"|>]]
- SetLogLevel["debug"]

