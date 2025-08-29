Config & Environment

Overview
- Load .env files, find config files by walking up directories, get XDG paths, expand environment variables.

DotenvLoad
- DotenvLoad[path?, <|override->Bool|>] -> <|path, loaded|> | Failure
- Loads KEY=VALUE pairs (ignores comments and blanks). If override is false, existing env vars are preserved.

ConfigFind
- ConfigFind[names?, <|startDir|>?] -> <|path, name|> | Null
- Walks from startDir up to root, returning first found name.

XdgDirs
- XdgDirs[<|app|>?] -> <|config_dir, cache_dir, data_dir|>
- Returns user-scoped XDG directories; if app is provided, appends it to each path.

EnvExpand
- EnvExpand[text, <|vars, style:"shell"|"windows"|>?] -> String
- Expands ${VAR} and $VAR (shell) or %VAR% (Windows). vars overrides process env.

Examples
- `DotenvLoad()`
- `ConfigFind(["lyra.toml","config.yaml"])`
- `XdgDirs({app: "mytool"})`
- `EnvExpand("Home=${HOME}/bin", {})`

Failures
- IO::env for dotenv, invalid inputs
