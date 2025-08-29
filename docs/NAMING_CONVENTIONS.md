Lyra Stdlib Naming Conventions

Goals

- Consistency: one mental model across modules and domains.
- Predictability: the same verbs and shapes do the same things.
- Brevity: avoid redundant type prefixes and suffixes.
- Extensibility: add new resources and actions without renaming existing ones.

Core Rules

- Head-as-noun: construct or refer to a resource using a noun head with options in an assoc. Prefer `Thing[<|opts|>]` over `ThingCreate`.
- Verb-as-action: operate on a resource using a small set of shared verbs. First argument is the resource. Prefer `Insert[thing, x]` over `ThingInsert`.
- Options last: when present, options are the final assoc argument `<|Key->Value, …|>`.
- Predicates: boolean-returning functions end with `Q` (e.g., `EmptyQ`, `MemberQ`, `SubsetQ`).
- Dispatch over prefixes: avoid `StringLength` and `SetSize`; prefer `Length[x]` with type-based dispatch.
- Stable abbreviations only: `HTTP`, `URL`, `JSON`, `SQL`, `JWT`, `UUID`, `HMAC`, `HKDF`. Avoid bespoke shortenings (e.g., use `PriorityQueue`, not `PQ`).

Resource Lifecycle

- Construct: `Thing[opts]` returns a handle (an expression head or id).
- Start/Stop: `Start[thing]`, `Stop[thing]`. Use `Run[spec, opts]` when starting implies immediate run.
- Open/Close: `Open[thing, opts]`, `Close[thing]` for cursors/streams/channels.
- Info/Properties: `Info[thing]` for metadata. Specific accessors use plural nouns, e.g., `Nodes[graph]`.

Generic Verbs (canonical)

- Data: `Add`, `Insert`, `Upsert`, `Remove`, `Delete`, `Write`, `Reset`.
- Lifecycle: `Open`, `Close`, `Start`, `Stop`, `Restart`, `Pause`, `Unpause`, `Cancel`.
- I/O: `Send`, `Receive`, `TrySend`, `TryReceive`, `Read`, `Fetch`, `Download`.
- Query: `Search`, `Count`, `List`, `Info`.
- Analysis: `Mean`, `Variance`, `PageRank`, etc. (domain-specific functions remain)
- Predicates: `EmptyQ`, `MemberQ`, `SubsetQ`, …

Options (common keys)

- Concurrency: prefer lowerCamelCase options alongside legacy forms: `maxThreads`, `timeBudgetMs`.
- Time: `timeoutMs` (per-call), `timeBudgetMs` (scope-wide/deadline).
- HTTP: `method`, `headers`, `body`, `query`, `followRedirects`, `auth`, etc. (legacy TitleCase still accepted).
- Graphs & collections: `nodes`, `edges`, `priority`, `keyFn`, etc.

Module Guidelines and Examples

Collections (Sets, Bags, Queues, Stacks, PriorityQueues)

- Heads: `Bag[opts]`, `Queue[opts]`, `Stack[opts]`, `PriorityQueue[opts]`.
- Set: use `HashSet[opts]` (avoid conflict with assignment `Set`).
- Operations: `Insert[hashSet, x]`, `Remove[hashSet, x]`, `MemberQ[hashSet, x]`, `Length[hashSet]`, `EmptyQ[hashSet]`.
- Use `Union/Intersection/Difference[set1, set2]` with dispatch; keep `ListUnion` et al. as `Union` on lists.
- Queues/Stacks/PriorityQueues: `Length[q]`, `EmptyQ[q]`, `Peek[q]`, `Pop[q]`, `Insert[pq, x, Priority->p]`.

Text Index

- `Index[opts]` creates or opens an index.
- `Add[index, docs]`, `Search[index, query, opts]`, `Info[index]`.

Graphs

- `Graph[opts]` constructs a graph; `Drop[graph]` deletes.
- `Add/Upsert/Remove[graph, Nodes->…, Edges->…]`.
- Accessors: `Nodes[graph]`, `Edges[graph]`, predicates: `NodeQ[graph, node]`, `EdgeQ[graph, edge]`.
- Algorithms keep established names: `BFS`, `DFS`, `ShortestPaths`, `PageRank`, etc.

Vector Store

- `VectorStore[opts]` head.
- `Upsert[store, items]`, `Search[store, query, opts]`, `Delete[store, ids]`, `Count[store]`, `Reset[store]`.

Filesystem & Archives

- Archives: `Zip[files, opts]`, `Unzip[file, opts]`, `Tar[files, opts]`, `Untar[file, opts]`.
- Watching: `Watch[path, opts]`, `Cancel[watch]`.
- Close things with `Close[handle]` (channels, cursors, streams).

Concurrency

- Channels: `Channel[opts]` (bounded by default), `Send/Receive/TrySend/TryReceive[channel, …]`, `Close[channel]`.
- Futures/tasks: `Future[expr, opts]`, `Await[future]`, `Cancel[future]`.
- Actors: `Actor[handler]`, `Tell[actor, msg]`, `Ask[actor, msg, opts]`, `Stop[actor]`.
- Scopes: `Scope[opts][body]` (avoid imperative `StartScope`, `EndScope`).

Database / SQL

- Connections: `Connect[opts]`, `Disconnect[conn]`, `Ping[conn]`.
- SQL: `Query[conn, sql, opts]` for SELECT; `Execute[conn, sql, opts]` for DDL/DML.
- Cursors: `Cursor[conn, sql, opts]`, `Fetch[cursor]`, `Close[cursor]`.
- Writes: `Insert[table, rows]`, `Upsert[table, rows]`, `Write[conn, dataset, opts]`.

HTTP / Net

- Generic: `HttpRequest[opts]` as canonical (wrappers like `HttpGet` are sugar only).
- Streaming: `HttpStream[opts]`, then `Read[stream]`, `Close[stream]`.
- Servers: `HttpServe[handler, opts]` (canonical). Use `HttpServeRoutes` for route tables. Legacy `HttpServer` is removed.
- Responses: `Respond["Text"|"Json"|"Bytes"|"Html"|"File"|"Redirect"|"NoContent", value, opts]`.

Strings

- Prefer generic `Length`, `Join`, `Split`, `Replace` with dispatch. Keep specialized forms only where semantics differ.
- `Split` is the canonical splitter; legacy `StringSplit` is not registered.
- Regex: keep one canonical name per operation (`RegexMatch`, `RegexFind`, `RegexFindAll`, `RegexReplace`).

Predicates

- Use `...Q` suffix. Examples: `EmptyQ[x]`, `MemberQ[set, x]`, `SubsetQ[a, b]`, `NodeQ[g, v]`.

Migration Notes

- Old names are removed rather than aliased. Update code and tests to the canonical names.
- Prefer generic verbs + dispatch to remove type prefixes in function names.
- Options support both TitleCase and lowerCamelCase during migration; prefer lowerCamelCase for new code (e.g., `timeoutMs`, `maxThreads`).
- Generic Verbs

These verbs are canonical and dispatched by the type of their first argument. They let you write one mental model across modules and data structures.

- Insert[target, value]:
  - Sets: Insert[HashSet[{1,2}], 3]
  - Bags: Insert[Bag[], "x"] (alias of Add)
  - Queues: Insert[Queue[], 10] (enqueue)
  - PriorityQueue: Insert[PriorityQueue[], 5]

- Remove[target, value?]:
  - Sets/Bags: Remove[setOrBag, value]
  - Queues/Stacks/PriorityQueues: Remove[handle] (dequeue/pop)
  - Filesystem paths: Remove["/tmp/file.txt", <|recursive->True|>]

- Add[target, value]:
  - Bags: Add[Bag[], "x"]

- Info[target]:
  - Graphs: Info[Graph[]]  ==> <|nodes->…, edges->…|>

- Length[x]:
  - Lists/Strings/Assocs: usual length semantics
  - Handles: Length[Queue[]]  ==> 0, Length[PriorityQueue[]]  ==> 0

- EmptyQ[x]:
  - Lists/Strings/Assocs and supported handles

- Count[x]:
  - Collections: Count[{1,2,3}]  ==> 3, Count[<|a->1|>]  ==> 1
  - Bag: Count[bag]  ==> total items
  - VectorStore: Count[VectorStore["sqlite:///path.db"]]

- Search[target, query, opts?]:
  - VectorStore: Search[VectorStore[<|name->"vs"|>], {0.1,0.2,0.3}]
  - Text Index: Search[Index["/tmp/idx.sqlite"], "hello"] or TextSearch for convenience

Examples

- Set and Queue
  - s := HashSet[{1,2}]; s := Insert[s, 3]; MemberQ[s, 3]  ==> True
  - q := Queue[]; Insert[q, 1]; Insert[q, 2]; Remove[q]  ==> 1

- Graph
  - g := Graph[]; AddNodes[g, {"a","b"}]; Info[g]  ==> <|nodes->2, edges->0|>

- Archives
  - Zip["/tmp/bundle.zip", {"/tmp/a.txt", "/tmp/dir"}]
  - Tar["/tmp/bundle.tar.gz", {"/tmp/data"}, <|gzip->True|>]
