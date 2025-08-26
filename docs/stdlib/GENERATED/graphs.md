# GRAPHS

| Function | Usage | Summary |
|---|---|---|
| `BFS` | `BFS[graph, start, opts?]` | Breadth-first search order |
| `DFS` | `DFS[graph, start, opts?]` | Depth-first search order |
| `GraphCreate` | `GraphCreate[opts?]` | Create a graph handle |
| `GraphInfo` | `GraphInfo[graph]` | Summary and counts for graph |
| `KCore` | `KCore[graph, k]` | Induced subgraph with k-core |
| `KCoreDecomposition` | `KCoreDecomposition[graph]` | K-core index per node |
| `PageRank` | `PageRank[graph, opts?]` | PageRank centrality scores |

## `BFS`

- Usage: `BFS[graph, start, opts?]`
- Summary: Breadth-first search order
- Examples:
  - `BFS[g, "a"]  ==> {"a","b"}`

## `GraphCreate`

- Usage: `GraphCreate[opts?]`
- Summary: Create a graph handle
- Examples:
  - `g := GraphCreate[]; AddNodes[g, {"a","b"}]`
