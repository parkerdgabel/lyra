# DB

| Function | Usage | Summary |
|---|---|---|
| `Begin` | `Begin[conn]` | Begin a transaction |
| `BeginModule` | `BeginModule[name]` | Start a module scope (record exports) |
| `Commit` | `Commit[conn]` | Commit the current transaction |
| `Connect` | `Connect[dsn|opts]` | Open database connection (mock/sqlite/duckdb) |
| `ConnectContainers` | `ConnectContainers[opts?]` | Connect to container runtime |
| `ConnectedComponents` | `ConnectedComponents[graph]` | Weakly connected components |
| `Exec` | `Exec[conn, sql, params?]` | Execute DDL/DML (non-SELECT) |
| `ExecInContainer` | `ExecInContainer[id, cmd, opts?]` | Run a command inside a container |
| `Fetch` | `Fetch[cursor, limit?]` | Fetch next batch of rows from a cursor |
| `InsertRows` | `InsertRows[conn, table, rows]` | Insert multiple rows (assoc list) into a table |
| `ListTables` | `ListTables[conn]` | List tables on a connection |
| `Rollback` | `Rollback[conn]` | Rollback the current transaction |
| `SQL` | `SQL[conn, sql, params?]` | Run a SELECT query and return rows |
| `SQLCursor` | `SQLCursor[conn, sql, params?]` | Run a query and return a cursor handle |
| `Table` | `Table[conn, name]` | Reference a table as a Dataset |
| `TableSimple` | `TableSimple[rows, opts?]` | Render a simple table from rows |
| `UpsertEdges` | `UpsertEdges[graph, edges]` | Insert or update edges |
| `UpsertNodes` | `UpsertNodes[graph, nodes]` | Insert or update nodes |
| `UpsertRows` | `UpsertRows[conn, table, rows, keys?]` | Upsert rows (assoc list) into a table |

## `Begin`

- Usage: `Begin[conn]`
- Summary: Begin a transaction
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
- Examples:
  - `conn := Connect["mock://"]`
  - `Ping[conn]  ==> True`

## `Exec`

- Usage: `Exec[conn, sql, params?]`
- Summary: Execute DDL/DML (non-SELECT)
- Examples:
  - `Exec[conn, "CREATE TABLE x(id INT)"]  ==> <|Status->0|>`

## `ExecInContainer`

- Usage: `ExecInContainer[id, cmd, opts?]`
- Summary: Run a command inside a container
- Examples:
  - `ExecInContainer[cid, {"ls", "/"}]  ==> <|code->0, out->..., err->...|>`

## `ListTables`

- Usage: `ListTables[conn]`
- Summary: List tables on a connection
- Examples:
  - `ListTables[conn]  ==> {"t"}`

## `SQL`

- Usage: `SQL[conn, sql, params?]`
- Summary: Run a SELECT query and return rows
- Examples:
  - `SQL[conn, "SELECT * FROM t"]  ==> Dataset[...] `

## `Table`

- Usage: `Table[conn, name]`
- Summary: Reference a table as a Dataset
- Examples:
  - `ds := Table[conn, "t"]; Head[ds,1]  ==> {<|...|>}`
