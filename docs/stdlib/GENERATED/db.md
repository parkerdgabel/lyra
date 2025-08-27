# DB

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `BeginModule` | `BeginModule[name]` | Start a module scope (record exports) |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `ConnectContainers` | `ConnectContainers[opts?]` | Connect to container runtime |
| `ConnectedComponents` | `ConnectedComponents[graph]` | Weakly connected components |
| `ConnectionInfo` | `ConnectionInfo[conn]` | Inspect connection details (dsn, kind, tx) |
| `CursorInfo` | `CursorInfo[cursor]` | Inspect cursor details (dsn, kind, sql, offset) |
| `Disconnect` | `Disconnect[conn]` | Close a database connection |
| `Exec` | `Exec[conn, sql, params?]` | Execute DDL/DML (non-SELECT) |
| `ExecInContainer` | `ExecInContainer[id, cmd, opts?]` | Run a command inside a container |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `Insert` | `Insert[target, value]` | Insert into collection or structure (dispatched) |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Ping` | `Ping[conn]` | Check connectivity for a database connection |
| `RegisterTable` | `RegisterTable[conn, name, rows]` | Register in-memory rows as a table (mock) |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQL` | `SQL[conn, sql, params?]` | Run a SELECT query and return rows |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `TableSimple` | `TableSimple[rows, opts?]` | Render a simple table from rows |
| `UpsertEdges` | `UpsertEdges[graph, edges]` | Insert or update edges |
| `UpsertNodes` | `UpsertNodes[graph, nodes]` | Insert or update nodes |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |

## `Begin`

- Usage: `Begin[conn]`
- Summary: Begin a transaction
- Tags: db, sql, tx
- Examples:
  - `Begin[conn]; Exec[conn, "INSERT ..."]; Commit[conn]`

## `BeginModule`

- Usage: `BeginModule[name]`
- Summary: Start a module scope (record exports)
- Examples:
  - `BeginModule["Main"]  ==> "Main"`

## `Connect`

- Usage: `Connect[dsn|opts]`
- Summary: Open database connection (mock/sqlite/duckdb)
- Tags: db, sql, conn
- Examples:
  - `conn := Connect["mock://"]`
  - `Ping[conn]  ==> True`

## `ConnectionInfo`

- Usage: `ConnectionInfo[conn]`
- Summary: Inspect connection details (dsn, kind, tx)
- Tags: db, conn, introspect
- Examples:
  - `conn := Connect[\"mock://\"]; ConnectionInfo[conn]`

## `CursorInfo`

- Usage: `CursorInfo[cursor]`
- Summary: Inspect cursor details (dsn, kind, sql, offset)
- Tags: db, cursor, introspect
- Examples:
  - `cur := SQLCursor[conn, \"select 1\"]; CursorInfo[cur]`

## `Exec`

- Usage: `Exec[conn, sql, params?]`
- Summary: Execute DDL/DML (non-SELECT)
- Tags: db, sql, query
- Examples:
  - `Exec[conn, "CREATE TABLE x(id INT)"]  ==> <|Status->0|>`

## `ExecInContainer`

- Usage: `ExecInContainer[id, cmd, opts?]`
- Summary: Run a command inside a container
- Examples:
  - `ExecInContainer[cid, {"ls", "/"}]  ==> <|code->0, out->..., err->...|>`

## `Insert`

- Usage: `Insert[target, value]`
- Summary: Insert into collection or structure (dispatched)
- Tags: generic, collection
- Examples:
  - `s := HashSet[{1,2}]; Insert[s, 3]  ==> s'`
  - `q := Queue[]; Insert[q, 5]  ==> q'`
  - `st := Stack[]; Insert[st, "x"]`
  - `pq := PriorityQueue[]; Insert[pq, <|"Key"->1, "Value"->"a"|>]`
  - `g := Graph[]; Insert[g, {"a","b"}]`
  - `Insert[g, <|Src->"a",Dst->"b"|>]`

## `ListTables`

- Usage: `ListTables[conn]`
- Summary: List tables on a connection
- Tags: db, sql, schema
- Examples:
  - `ListTables[conn]  ==> {"t"}`

## `RegisterTable`

- Usage: `RegisterTable[conn, name, rows]`
- Summary: Register in-memory rows as a table (mock)
- Tags: db, sql, table
- Examples:
  - `rows := {<|"id"->1, "name"->"a"|>, <|"id"->2, "name"->"b"|>}`
  - `conn := Connect["mock://"]; RegisterTable[conn, "t", rows]  ==> True`

## `SQL`

- Usage: `SQL[conn, sql, params?]`
- Summary: Run a SELECT query and return rows
- Tags: db, sql, query
- Examples:
  - `SQL[conn, "SELECT * FROM t"]  ==> Dataset[...] `
