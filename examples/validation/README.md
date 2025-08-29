Lyra Validation Suite

How to run
- Use `lyra-runner --file <script>` (e.g., `lyra-runner --file examples/validation/00-basics.lyra`).
- For CLI scripts that parse argv, pass script args after `--`.

Scripts
- 00-basics.lyra: literals, arithmetic, booleans, comments, grouping
- 01-strings-interpolation.lyra: strings, escaping, interpolation
- 02-lists-assoc.lyra: lists, associations, Part/Get, dot access
- 03-calls-operators.lyra: call forms, pipeline, prefix/postfix/infix, apply, map
- 04-lambdas-and-pure-functions.lyra: lambdas, slots, `&`, immediate calls
- 05-rules-patterns.lyra: blanks, named/typed blanks, PatternTest, alternatives, rules/conditions
- 06-assignments-and-defs.lyra: Set/SetDelayed, function defs with patterns
- 07-functional.lyra: Apply, Map, FoldList, Nest, Try
- 08-strings-regex.lyra: string helpers, regex APIs
- 09-io-files-data.lyra: ReadFile/WriteFile, JSON/YAML/TOML
- 10-bytes-encoding.lyra: TextEncode, Base64, Hex
- 11-fs-glob-archives.lyra: Glob, Zip/Tar/Gzip
- 12-net-http.lyra: HTTP streaming, cached download, retry
- 13-cli-args.lyra: ArgsParse and simple greeting
- 14-config-env-path.lyra: DotenvLoad, EnvExpand, XdgDirs, path ops
- 15-crypto-uuid.lyra: UuidV4
- 16-app-http-etl.lyra: small ETL app to CSV
- 17-concurrency.lyra: Future/Await, ParallelMap

Notes
- Some scripts require network or specific features (e.g., concurrency) enabled when building stdlib.
- Outputs are printed by lyra-runner; expected results are noted in comments.
