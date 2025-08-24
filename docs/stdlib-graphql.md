Lyra Stdlib: GraphQL Client

Overview
- Minimal GraphQL client with headers/auth, JSON body building, and error handling.
- Shared HTTP runtime and client are used under the hood for efficiency.

Key functions
- `GraphQLClient[endpoint, <|headers -> <|...|>, bearerToken -> "..."|>]` → client object
- `GraphQLQuery[client|endpoint, query, variables?, <|headers, bearerToken, parse -> True, errorOnGraphQLErrors -> True|>]` → Value (parsed JSON by default)
- `GraphQLQueryResponse[client|endpoint, query, variables?, opts?]` → GraphQLResponse object
- `GraphQLIntrospect[client|endpoint]` → schema fragment (minimal)

Headers & Auth patterns
- Set default headers on the client: `GraphQLClient[url, <|headers -> <|"X-Org" -> "acme"|>|>]`
- Override/merge per-call headers: `GraphQLQuery[client, q, vars, <|headers -> <|"X-Request" -> "123"|>|>]`
- Bearer token at client level: `GraphQLClient[url, <|bearerToken -> token|>]`
- Or per-call: `GraphQLQuery[client, q, vars, <|bearerToken -> token|>]`
- Authorization header is set as `Authorization: Bearer <token>` when provided.

GraphQLResponse usage
- `resp = GraphQLQueryResponse[client, "{ viewer { id } }"]`
- `resp["Data"]` → parsed `data`
- `resp["Errors"]` → `errors` list or `Missing`
- `resp["Extensions"]` → extensions if present
- `resp["HasErrors"]` → True if errors present

Examples
- With variables: `GraphQLQuery[client, "query($n:Int){ user(id:$n){name}}", <|"n" -> 1|>]`
- As object: `resp = GraphQLQueryResponse[url, "{ ok }"] ; resp["HasErrors"]`

Notes
- By default `GraphQLQuery` parses JSON. Set `parse -> False` to return raw string.
- When `errorOnGraphQLErrors -> True`, a response containing `errors` raises a runtime error.

Error handling strategies
- Strict: set `errorOnGraphQLErrors -> True` to raise on any `errors` field.
- Lenient: use `GraphQLQueryResponse` and inspect `resp["HasErrors"]` and `resp["Errors"]` without raising.
- Transport: HTTP/network errors surface as runtime errors; wrap with `HTTPRetry` for resilience when appropriate.

Batching patterns
- Simple batching: send an array of operations if your server supports persisted queries or batch execution.
- Client-side batched calls: map over a list of variables and call `GraphQLQuery`/`GraphQLQueryResponse` in parallel using existing async abstractions.
- Example (pseudo-WL): `Map[vars, v \[Function] GraphQLQueryResponse[client, q, v]]`
