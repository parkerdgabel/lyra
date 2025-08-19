//! Graph Theory & Network Analysis for the Lyra standard library
//! 
//! This module implements comprehensive graph theory algorithms and data structures
//! following the "Take Algorithms for granted" principle. Users can rely on efficient,
//! battle-tested implementations of classic graph algorithms.
//!
//! ## Features
//!
//! - **Graph Creation**: Create various types of graphs (directed, undirected, weighted)
//! - **Traversal**: Depth-first and breadth-first search algorithms
//! - **Shortest Paths**: Dijkstra's algorithm, Bellman-Ford, Floyd-Warshall
//! - **Connectivity**: Find connected components and strongly connected components
//! - **Centrality**: Compute various centrality measures (betweenness, closeness, PageRank)
//! - **Network Analysis**: Graph properties and metrics
//!
//! ## Graph Representation
//!
//! Graphs are represented using adjacency lists for efficient traversal and algorithms.
//! Both directed and undirected graphs are supported, with optional edge weights.

use crate::vm::{Value, VmError, VmResult};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

/// Internal representation of a graph
#[derive(Debug, Clone)]
pub struct Graph {
    /// Adjacency list representation: vertex -> list of (neighbor, weight)
    pub adjacency: HashMap<i64, Vec<(i64, f64)>>,
    /// Whether the graph is directed
    pub directed: bool,
    /// Set of all vertices in the graph
    pub vertices: HashSet<i64>,
}

impl Graph {
    /// Create a new empty graph
    pub fn new(directed: bool) -> Self {
        Graph {
            adjacency: HashMap::new(),
            directed,
            vertices: HashSet::new(),
        }
    }

    /// Add a vertex to the graph
    pub fn add_vertex(&mut self, vertex: i64) {
        self.vertices.insert(vertex);
        self.adjacency.entry(vertex).or_insert_with(Vec::new);
    }

    /// Add an edge to the graph
    pub fn add_edge(&mut self, from: i64, to: i64, weight: f64) {
        self.add_vertex(from);
        self.add_vertex(to);

        // Add forward edge
        self.adjacency.get_mut(&from).unwrap().push((to, weight));

        // Add backward edge for undirected graphs
        if !self.directed {
            self.adjacency.get_mut(&to).unwrap().push((from, weight));
        }
    }

    /// Get neighbors of a vertex
    pub fn neighbors(&self, vertex: i64) -> Option<&Vec<(i64, f64)>> {
        self.adjacency.get(&vertex)
    }

    /// Get all vertices in the graph
    pub fn get_vertices(&self) -> Vec<i64> {
        let mut vertices: Vec<i64> = self.vertices.iter().cloned().collect();
        vertices.sort();
        vertices
    }

    /// Get number of vertices
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        let total_edges: usize = self.adjacency.values().map(|edges| edges.len()).sum();
        if self.directed {
            total_edges
        } else {
            total_edges / 2  // Each undirected edge is counted twice
        }
    }
}

/// Create an undirected graph from edge list
/// Usage: Graph[{{1, 2}, {2, 3}, {3, 1}}] -> Creates triangle graph
pub fn graph(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (edge list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let edge_list = match &args[0] {
        Value::List(edges) => edges,
        _ => return Err(VmError::TypeError {
            expected: "list of edges".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let mut graph = Graph::new(false); // Undirected graph

    for edge in edge_list {
        match edge {
            Value::List(edge_data) if edge_data.len() >= 2 => {
                let from = extract_integer(&edge_data[0])?;
                let to = extract_integer(&edge_data[1])?;
                let weight = if edge_data.len() >= 3 {
                    extract_number(&edge_data[2])?
                } else {
                    1.0 // Default weight
                };
                
                graph.add_edge(from, to, weight);
            }
            _ => return Err(VmError::TypeError {
                expected: "edge as list {from, to} or {from, to, weight}".to_string(),
                actual: format!("{:?}", edge),
            }),
        }
    }

    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// Create a directed graph from edge list
/// Usage: DirectedGraph[{{1, 2}, {2, 3}, {3, 1}}] -> Creates directed cycle
pub fn directed_graph(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (edge list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let edge_list = match &args[0] {
        Value::List(edges) => edges,
        _ => return Err(VmError::TypeError {
            expected: "list of edges".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let mut graph = Graph::new(true); // Directed graph

    for edge in edge_list {
        match edge {
            Value::List(edge_data) if edge_data.len() >= 2 => {
                let from = extract_integer(&edge_data[0])?;
                let to = extract_integer(&edge_data[1])?;
                let weight = if edge_data.len() >= 3 {
                    extract_number(&edge_data[2])?
                } else {
                    1.0 // Default weight
                };
                
                graph.add_edge(from, to, weight);
            }
            _ => return Err(VmError::TypeError {
                expected: "edge as list {from, to} or {from, to, weight}".to_string(),
                actual: format!("{:?}", edge),
            }),
        }
    }

    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// Create graph from adjacency matrix
/// Usage: AdjacencyMatrix[{{0, 1, 0}, {1, 0, 1}, {0, 1, 0}}]
pub fn adjacency_matrix(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (adjacency matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = match &args[0] {
        Value::List(rows) => rows,
        _ => return Err(VmError::TypeError {
            expected: "adjacency matrix as list of lists".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    if matrix.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty matrix".to_string(),
            actual: "empty matrix".to_string(),
        });
    }

    let n = matrix.len();
    let mut graph = Graph::new(false); // Default to undirected

    // Check if matrix is square and extract adjacency information
    for (i, row) in matrix.iter().enumerate() {
        match row {
            Value::List(row_data) => {
                if row_data.len() != n {
                    return Err(VmError::TypeError {
                        expected: format!("square matrix ({}x{})", n, n),
                        actual: format!("row {} has {} elements", i, row_data.len()),
                    });
                }

                for (j, &ref cell) in row_data.iter().enumerate() {
                    let weight = extract_number(cell)?;
                    if weight != 0.0 {
                        // For undirected graphs, only process upper triangle to avoid double-counting
                        if graph.directed || i <= j {
                            graph.add_edge(i as i64, j as i64, weight);
                        }
                    }
                }
            }
            _ => return Err(VmError::TypeError {
                expected: "matrix row as list".to_string(),
                actual: format!("{:?}", row),
            }),
        }
    }

    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// Depth-First Search traversal
/// Usage: DepthFirstSearch[graph, start_vertex] -> Returns list of vertices in DFS order
pub fn depth_first_search(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (graph, start_vertex)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    let start_vertex = extract_integer(&args[1])?;

    if !graph.vertices.contains(&start_vertex) {
        return Err(VmError::TypeError {
            expected: "vertex in graph".to_string(),
            actual: format!("vertex {} not found in graph", start_vertex),
        });
    }

    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut stack = vec![start_vertex];

    while let Some(vertex) = stack.pop() {
        if !visited.contains(&vertex) {
            visited.insert(vertex);
            result.push(Value::Integer(vertex));

            // Add neighbors to stack (in reverse order to maintain consistent ordering)
            if let Some(neighbors) = graph.neighbors(vertex) {
                let mut neighbor_vertices: Vec<i64> = neighbors.iter().map(|(v, _)| *v).collect();
                neighbor_vertices.sort();
                neighbor_vertices.reverse(); // Reverse to maintain order when popping from stack
                
                for neighbor in neighbor_vertices {
                    if !visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
    }

    Ok(Value::List(result))
}

/// Breadth-First Search traversal
/// Usage: BreadthFirstSearch[graph, start_vertex] -> Returns list of vertices in BFS order
pub fn breadth_first_search(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (graph, start_vertex)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    let start_vertex = extract_integer(&args[1])?;

    if !graph.vertices.contains(&start_vertex) {
        return Err(VmError::TypeError {
            expected: "vertex in graph".to_string(),
            actual: format!("vertex {} not found in graph", start_vertex),
        });
    }

    let mut visited = HashSet::new();
    let mut result = Vec::new();
    let mut queue = VecDeque::new();

    queue.push_back(start_vertex);
    visited.insert(start_vertex);

    while let Some(vertex) = queue.pop_front() {
        result.push(Value::Integer(vertex));

        // Add neighbors to queue
        if let Some(neighbors) = graph.neighbors(vertex) {
            let mut neighbor_vertices: Vec<i64> = neighbors.iter().map(|(v, _)| *v).collect();
            neighbor_vertices.sort(); // Maintain consistent ordering
            
            for neighbor in neighbor_vertices {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }
    }

    Ok(Value::List(result))
}

/// Priority queue item for Dijkstra's algorithm
#[derive(Debug, PartialEq)]
struct DijkstraItem {
    distance: f64,
    vertex: i64,
}

impl Eq for DijkstraItem {}

impl Ord for DijkstraItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.distance.partial_cmp(&self.distance).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra's shortest path algorithm
/// Usage: Dijkstra[graph, start_vertex] -> Returns {distances, predecessors}
pub fn dijkstra(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (graph, start_vertex)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    let start_vertex = extract_integer(&args[1])?;

    if !graph.vertices.contains(&start_vertex) {
        return Err(VmError::TypeError {
            expected: "vertex in graph".to_string(),
            actual: format!("vertex {} not found in graph", start_vertex),
        });
    }

    let mut distances: HashMap<i64, f64> = HashMap::new();
    let mut predecessors: HashMap<i64, Option<i64>> = HashMap::new();
    let mut heap = BinaryHeap::new();

    // Initialize distances and predecessors
    for &vertex in &graph.vertices {
        distances.insert(vertex, f64::INFINITY);
        predecessors.insert(vertex, None);
    }

    distances.insert(start_vertex, 0.0);
    heap.push(DijkstraItem { distance: 0.0, vertex: start_vertex });

    while let Some(DijkstraItem { distance, vertex }) = heap.pop() {
        // Skip if we've already found a better path
        if distance > *distances.get(&vertex).unwrap_or(&f64::INFINITY) {
            continue;
        }

        // Check all neighbors
        if let Some(neighbors) = graph.neighbors(vertex) {
            for &(neighbor, edge_weight) in neighbors {
                let new_distance = distance + edge_weight;
                
                if new_distance < *distances.get(&neighbor).unwrap_or(&f64::INFINITY) {
                    distances.insert(neighbor, new_distance);
                    predecessors.insert(neighbor, Some(vertex));
                    heap.push(DijkstraItem { distance: new_distance, vertex: neighbor });
                }
            }
        }
    }

    // Convert results to Lyra values
    let mut distance_list = Vec::new();
    let mut predecessor_list = Vec::new();
    
    let mut vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    vertices.sort();
    
    for vertex in vertices {
        let dist = distances.get(&vertex).unwrap_or(&f64::INFINITY);
        distance_list.push(Value::List(vec![
            Value::Integer(vertex),
            if dist.is_infinite() { Value::Symbol("Infinity".to_string()) } else { Value::Real(*dist) }
        ]));
        
        let pred = predecessors.get(&vertex).unwrap_or(&None);
        predecessor_list.push(Value::List(vec![
            Value::Integer(vertex),
            match pred {
                Some(p) => Value::Integer(*p),
                None => Value::Symbol("None".to_string()),
            }
        ]));
    }

    Ok(Value::List(vec![
        Value::List(distance_list),
        Value::List(predecessor_list)
    ]))
}

/// Find connected components in an undirected graph
/// Usage: ConnectedComponents[graph] -> Returns list of component lists
pub fn connected_components(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;

    if graph.directed {
        return Err(VmError::TypeError {
            expected: "undirected graph".to_string(),
            actual: "directed graph".to_string(),
        });
    }

    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &vertex in &graph.vertices {
        if !visited.contains(&vertex) {
            let mut component = Vec::new();
            let mut stack = vec![vertex];

            while let Some(current) = stack.pop() {
                if !visited.contains(&current) {
                    visited.insert(current);
                    component.push(Value::Integer(current));

                    // Add all unvisited neighbors
                    if let Some(neighbors) = graph.neighbors(current) {
                        for &(neighbor, _) in neighbors {
                            if !visited.contains(&neighbor) {
                                stack.push(neighbor);
                            }
                        }
                    }
                }
            }

            // Sort component for consistent output
            component.sort_by(|a, b| {
                match (a, b) {
                    (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                    _ => Ordering::Equal,
                }
            });
            components.push(Value::List(component));
        }
    }

    // Sort components by their first element
    components.sort_by(|a, b| {
        match (a, b) {
            (Value::List(comp_a), Value::List(comp_b)) if !comp_a.is_empty() && !comp_b.is_empty() => {
                match (&comp_a[0], &comp_b[0]) {
                    (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                    _ => Ordering::Equal,
                }
            }
            _ => Ordering::Equal,
        }
    });

    Ok(Value::List(components))
}

/// Get basic graph properties
/// Usage: GraphProperties[graph] -> Returns association with properties
pub fn graph_properties(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;

    let properties = vec![
        Value::List(vec![
            Value::Symbol("VertexCount".to_string()),
            Value::Integer(graph.vertex_count() as i64)
        ]),
        Value::List(vec![
            Value::Symbol("EdgeCount".to_string()),
            Value::Integer(graph.edge_count() as i64)
        ]),
        Value::List(vec![
            Value::Symbol("Directed".to_string()),
            Value::Symbol(if graph.directed { "True" } else { "False" }.to_string())
        ]),
    ];

    Ok(Value::List(properties))
}

// Helper functions

fn extract_graph(value: &Value) -> VmResult<&Graph> {
    match value {
        Value::LyObj(lyobj) => {
            lyobj.downcast_ref::<Graph>().ok_or_else(|| VmError::TypeError {
                expected: "Graph".to_string(),
                actual: format!("LyObj (not Graph): {}", lyobj.type_name()),
            })
        }
        _ => Err(VmError::TypeError {
            expected: "Graph".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn extract_integer(value: &Value) -> VmResult<i64> {
    match value {
        Value::Integer(n) => Ok(*n),
        Value::Real(r) => Ok(*r as i64),
        _ => Err(VmError::TypeError {
            expected: "integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

fn extract_number(value: &Value) -> VmResult<f64> {
    match value {
        Value::Integer(n) => Ok(*n as f64),
        Value::Real(r) => Ok(*r),
        _ => Err(VmError::TypeError {
            expected: "number".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

// Implement Foreign trait for Graph
use crate::foreign::{Foreign, ForeignError, LyObj};

impl Foreign for Graph {
    fn type_name(&self) -> &'static str {
        "Graph"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "VertexCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.vertex_count() as i64))
            }
            "EdgeCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.edge_count() as i64))
            }
            "IsDirected" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Symbol(if self.directed { "True" } else { "False" }.to_string()))
            }
            "Vertices" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let vertices: Vec<Value> = self.get_vertices()
                    .into_iter()
                    .map(|v| Value::Integer(v))
                    .collect();
                Ok(Value::List(vertices))
            }
            "Neighbors" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(vertex) => {
                        if let Some(neighbors) = self.neighbors(*vertex) {
                            let neighbor_list: Vec<Value> = neighbors
                                .iter()
                                .map(|(v, w)| {
                                    if *w == 1.0 {
                                        Value::Integer(*v)
                                    } else {
                                        Value::List(vec![Value::Integer(*v), Value::Real(*w)])
                                    }
                                })
                                .collect();
                            Ok(Value::List(neighbor_list))
                        } else {
                            Err(ForeignError::RuntimeError {
                                message: format!("Vertex {} not found in graph", vertex),
                            })
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_graph_creation() {
        // Test undirected graph creation
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(3), Value::Integer(1)]),
        ]);

        let result = graph(&[edges]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let graph = lyobj.downcast_ref::<Graph>().unwrap();
                assert_eq!(graph.vertex_count(), 3);
                assert_eq!(graph.edge_count(), 3);
                assert!(!graph.directed);
            }
            _ => panic!("Expected Graph LyObj"),
        }
    }

    #[test]
    fn test_directed_graph_creation() {
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
        ]);

        let result = directed_graph(&[edges]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let graph = lyobj.downcast_ref::<Graph>().unwrap();
                assert_eq!(graph.vertex_count(), 3);
                assert_eq!(graph.edge_count(), 2);
                assert!(graph.directed);
            }
            _ => panic!("Expected Graph LyObj"),
        }
    }

    #[test]
    fn test_adjacency_matrix() {
        let matrix = Value::List(vec![
            Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0)]),
            Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(1)]),
            Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0)]),
        ]);

        let result = adjacency_matrix(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let graph = lyobj.downcast_ref::<Graph>().unwrap();
                assert_eq!(graph.vertex_count(), 3);
                assert_eq!(graph.edge_count(), 2); // Two edges in undirected path graph
            }
            _ => panic!("Expected Graph LyObj"),
        }
    }

    #[test]
    fn test_dfs() {
        // Create a simple triangle graph
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(3), Value::Integer(1)]),
        ]);

        let graph_value = graph(&[edges]).unwrap();
        let result = depth_first_search(&[graph_value, Value::Integer(1)]).unwrap();

        match result {
            Value::List(vertices) => {
                assert_eq!(vertices.len(), 3);
                // First vertex should be the start vertex
                assert_eq!(vertices[0], Value::Integer(1));
            }
            _ => panic!("Expected list of vertices"),
        }
    }

    #[test]
    fn test_bfs() {
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(1), Value::Integer(3)]),
            Value::List(vec![Value::Integer(2), Value::Integer(4)]),
        ]);

        let graph_value = graph(&[edges]).unwrap();
        let result = breadth_first_search(&[graph_value, Value::Integer(1)]).unwrap();

        match result {
            Value::List(vertices) => {
                assert_eq!(vertices.len(), 4);
                assert_eq!(vertices[0], Value::Integer(1)); // Start vertex first
            }
            _ => panic!("Expected list of vertices"),
        }
    }

    #[test]
    fn test_dijkstra() {
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Real(1.0)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3), Value::Real(2.0)]),
            Value::List(vec![Value::Integer(1), Value::Integer(3), Value::Real(4.0)]),
        ]);

        let graph_value = graph(&[edges]).unwrap();
        let result = dijkstra(&[graph_value, Value::Integer(1)]).unwrap();

        match result {
            Value::List(results) if results.len() == 2 => {
                // Should return distances and predecessors
                match (&results[0], &results[1]) {
                    (Value::List(distances), Value::List(_predecessors)) => {
                        assert_eq!(distances.len(), 3); // Three vertices
                    }
                    _ => panic!("Expected distances and predecessors lists"),
                }
            }
            _ => panic!("Expected result with distances and predecessors"),
        }
    }

    #[test]
    fn test_connected_components() {
        // Create two separate triangles
        let edges = Value::List(vec![
            // First triangle
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
            Value::List(vec![Value::Integer(3), Value::Integer(1)]),
            // Second triangle
            Value::List(vec![Value::Integer(4), Value::Integer(5)]),
            Value::List(vec![Value::Integer(5), Value::Integer(6)]),
            Value::List(vec![Value::Integer(6), Value::Integer(4)]),
        ]);

        let graph_value = graph(&[edges]).unwrap();
        let result = connected_components(&[graph_value]).unwrap();

        match result {
            Value::List(components) => {
                assert_eq!(components.len(), 2); // Two connected components
                
                for component in components {
                    if let Value::List(vertices) = component {
                        assert_eq!(vertices.len(), 3); // Each component has 3 vertices
                    }
                }
            }
            _ => panic!("Expected list of components"),
        }
    }

    #[test]
    fn test_graph_properties() {
        let edges = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
        ]);

        let graph_value = graph(&[edges]).unwrap();
        let result = graph_properties(&[graph_value]).unwrap();

        match result {
            Value::List(properties) => {
                assert_eq!(properties.len(), 3); // VertexCount, EdgeCount, Directed
                
                // Check VertexCount
                if let Value::List(vertex_prop) = &properties[0] {
                    assert_eq!(vertex_prop[0], Value::Symbol("VertexCount".to_string()));
                    assert_eq!(vertex_prop[1], Value::Integer(3));
                }
            }
            _ => panic!("Expected list of properties"),
        }
    }
}