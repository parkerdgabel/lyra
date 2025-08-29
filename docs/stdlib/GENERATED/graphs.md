# GRAPHS

| Function | Usage | Summary |
|---|---|---|
| `AddEdges` | `AddEdges[graph, edges]` | Add edges with optional keys and weights |
| `AddNodes` | `AddNodes[graph, nodes]` | Add nodes by id and attributes |
| `BFS` | `BFS[graph, start, opts?]` | Breadth-first search order |
| `DFS` | `DFS[graph, start, opts?]` | Depth-first search order |
| `Graph` | `Graph[opts?]` | Create a graph handle |
| `GraphInfo` | `GraphInfo[graph]` | Summary and counts for graph |
| `GraphNetwork` | `GraphNetwork[nodes, edges, opts?]` | Construct a graph network from nodes/edges |
| `KCore` | `KCore[graph, k]` | Induced subgraph with k-core |
| `KCoreDecomposition` | `KCoreDecomposition[graph]` | K-core index per node |
| `MaxFlow` | `MaxFlow[graph, src, dst]` | Maximum flow value and cut |
| `MinimumSpanningTree` | `MinimumSpanningTree[graph]` | Edges in a minimum spanning tree |
| `Neighbors` | `Neighbors[graph, id, opts?]` | Neighbor node ids for a node |
| `PageRank` | `PageRank[graph, opts?]` | PageRank centrality scores |
| `RemoveEdges` | `RemoveEdges[graph, edges]` | Remove edges by id or (src,dst,key) |
| `RemoveNodes` | `RemoveNodes[graph, ids]` | Remove nodes by id |
| `ShortestPaths` | `ShortestPaths[graph, start, opts?]` | Shortest path distances from start |

## `AddEdges`

- Usage: `AddEdges[graph, edges]`
- Summary: Add edges with optional keys and weights
- Tags: graph, mutate
- Examples:
  - `AddEdges[g, {{"a","b"}}]`

## `BFS`

- Usage: `BFS[graph, start, opts?]`
- Summary: Breadth-first search order
- Tags: graph, traversal
- Examples:
  - `BFS[g, "a"]  ==> {"a","b"}`

## `Graph`

- Usage: `Graph[opts?]`
- Summary: Create a graph handle
- Tags: graph, graphs, lifecycle
- Examples:
  - `g := Graph[]; AddNodes[g, {"a","b"}]`
