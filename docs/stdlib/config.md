Config & Environment

Overview
- Load .env files, find config files by walking up directories, get XDG paths, expand environment variables.

DotenvLoad
- DotenvLoad[path?, <|Override->Bool|>] -> <|path, loaded|> | Failure
- Loads KEY=VALUE pairs (ignores comments and blanks). If Override is false, existing env vars are preserved.

ConfigFind
- ConfigFind[names?, <|StartDir|>?] -> <|path, name|> | Null
- Walks from StartDir up to root, returning first found name.

XdgDirs
- XdgDirs[<|App|>?] -> <|config_dir, cache_dir, data_dir|>
- Returns user-scoped XDG directories; if App is provided, appends it to each path.

EnvExpand
- EnvExpand[text, <|Vars, Style:"shell"|"windows"|>?] -> String
- Expands ${VAR} and $VAR (shell) or %VAR% (Windows). Vars overrides process env.

Examples
- `DotenvLoad()`
- `ConfigFind(["lyra.toml","config.yaml"])`
- `XdgDirs({App: "mytool"})`
- `EnvExpand("Home=${HOME}/bin", {})`

Failures
- IO::env for dotenv, invalid inputs

