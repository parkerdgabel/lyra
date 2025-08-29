# Standard Library Docs

This folder contains focused guides for Lyra's standard library groups.

- Generated API reference per module is available here:
- docs/stdlib/GENERATED/README.md

- Strings: docs/stdlib/string.md
- Functional: docs/stdlib/functional.md
- Data Formats & Encoding: docs/stdlib/data.md
- HTTP/Net: docs/stdlib/net.md
- Filesystem: docs/stdlib/fs.md
- CLI: docs/stdlib/cli.md
- Config & Env: docs/stdlib/config.md
- Coverage & Exports: docs/stdlib/COVERAGE.md
 - Transformers (NN): docs/stdlib/transformer.md

## Canonical vs. Legacy (Quick Reference)

- Function names:
  - Prefer generic, dispatched verbs and canonical heads. Legacy type-prefixed names (e.g., `StringSplit`, `AssocGet`, `SetUnion`, `HttpServer`) are removed from the public registry.
  - Use `Split`, `Get`, `Union/Intersection/Difference`, and `HttpServe` instead.

- Option casing:
  - Prefer lowerCamelCase option keys in examples and new code. The runtime accepts both during migration, but lowerCamelCase is canonical.
  - Examples:
    - Concurrency: `<|maxThreads->2, timeBudgetMs->1000|>`
    - HTTP server: `<|host->"127.0.0.1", port->8080|>`
    - HTTP client: `<|headers-><|"User-Agent"->"lyra"|>, followRedirects->true, timeoutMs->1000|>`
    - Filesystem: `<|parents->true, recursive->true|>`
    - Vector store: `<|name->"mem", dims->8|>`
    - RAG/Search: `<|k->3, style->"markdown", withScores->true|>`

- Pipelines & right-apply:
  - `x // f` is strict right-apply (`f[x]`), `f @ x` is prefix apply. Pipelines `a |> f[args]` inject as first arg.

- Deprecations removed from docs and code:
  - Removed: `Lookup`, `StringSplit`, `StringLength`, `AssocGet/AssocContainsKeyQ`, `SetUnion/SetIntersection/SetDifference`, `ListUnion/ListIntersection/ListDifference`, `HttpServer`.
  - Canonical replacements are shown in all examples and the generated reference.

## LowerCamelCase Examples (explicit)

- HTTP server:
  - `HttpServe[(req)=>RespondText["ok"], <|host->"127.0.0.1", port->0|>]`
- Parallel map:
  - `ParallelMap[(#*2)&, {1,2,3}, <|maxThreads->2|>]`
- Vector store:
  - `VectorStore[<|name->"mem", dims->8|>]`
- RAG:
  - `RAGAnswer["mem", "query", <|k->2|>]`
- Filesystem:
  - `MakeDirectory["/tmp/x", <|parents->true|>]`
