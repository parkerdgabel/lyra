CLI Ergonomics

Overview
- Parse argv into options/flags/rest; interactive prompts; selection; lightweight progress handles.

ArgsParse
- ArgsParse[spec?, argv?] -> <|Options, Flags, Rest|>
- spec:
  - Options: List of <|Name, Short?, Type:"string"|"int"|"float"|"bool", Repeat?|>
  - Flags: List of <|Name, Short?|>
  - RestName: String (label only)
- Examples
  - `ArgsParse(<|Options->{<|Name:"config",Short:"c",Type:"string"|>}, Flags->{<|Name:"verbose",Short:"v"|>}|>, ["-v","-c","conf.toml","file1"])`

Prompts
- Prompt[message, opts?] -> String
- Confirm[message, opts?] -> Bool
- PasswordPrompt[message, opts?] -> String (no masking in the current build)
- Select[message, choices, opts?] -> String | Null
  - choices: List[String | <|name, value|>]

Progress
- ProgressBar[total, opts?] -> handleId
- ProgressAdvance[handleId, n?] -> True
- ProgressFinish[handleId] -> True

Notes
- Prompt/Confirm/PasswordPrompt/Select interact with stdin/stdout and are blocking.
- PasswordPrompt is plaintext for now; masking can be added behind a feature if desired.

