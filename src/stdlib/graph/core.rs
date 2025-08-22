//! Core Graph Theory & Network Analysis for the Lyra standard library
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

    /// Check if there's an edge between two vertices
    pub fn has_edge(&self, from: i64, to: i64) -> bool {
        if let Some(neighbors) = self.neighbors(from) {
            neighbors.iter().any(|(neighbor, _)| *neighbor == to)
        } else {
            false
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

// ===============================
// ADVANCED GRAPH ALGORITHMS (Phase 13B)
// ===============================

/// Find minimum spanning tree using Kruskal's algorithm
/// Usage: MinimumSpanningTree[graph] -> Returns MST as graph
pub fn minimum_spanning_tree(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("MST is only defined for undirected graphs".to_string()));
    }

    // Collect all edges
    let mut edges = Vec::new();
    for (&u, neighbors) in &graph.adjacency {
        for &(v, weight) in neighbors {
            if u < v { // Avoid duplicate edges in undirected graph
                edges.push((weight, u, v));
            }
        }
    }
    
    // Sort edges by weight
    edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    
    // Union-Find data structure for Kruskal's algorithm
    let mut parent: HashMap<i64, i64> = HashMap::new();
    let mut rank: HashMap<i64, usize> = HashMap::new();
    
    for &vertex in &graph.vertices {
        parent.insert(vertex, vertex);
        rank.insert(vertex, 0);
    }
    
    fn find(x: i64, parent: &mut HashMap<i64, i64>) -> i64 {
        if parent[&x] != x {
            let root = find(parent[&x], parent);
            parent.insert(x, root);
        }
        parent[&x]
    }
    
    fn union(x: i64, y: i64, parent: &mut HashMap<i64, i64>, rank: &mut HashMap<i64, usize>) -> bool {
        let root_x = find(x, parent);
        let root_y = find(y, parent);
        
        if root_x == root_y {
            return false; // Already in same component
        }
        
        if rank[&root_x] < rank[&root_y] {
            parent.insert(root_x, root_y);
        } else if rank[&root_x] > rank[&root_y] {
            parent.insert(root_y, root_x);
        } else {
            parent.insert(root_y, root_x);
            rank.insert(root_x, rank[&root_x] + 1);
        }
        
        true
    }
    
    // Build MST
    let mut mst = Graph::new(false);
    let mut mst_edges = Vec::new();
    
    for (weight, u, v) in edges {
        if union(u, v, &mut parent, &mut rank) {
            mst.add_edge(u, v, weight);
            mst_edges.push(vec![Value::Integer(u), Value::Integer(v), Value::Real(weight)]);
            
            if mst_edges.len() == graph.vertices.len() - 1 {
                break; // MST complete
            }
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(mst))))
}

/// Find maximum flow using Ford-Fulkerson algorithm with DFS
/// Usage: MaximumFlow[graph, source, sink] -> Returns maximum flow value
pub fn maximum_flow(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (graph, source, sink)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    let source = extract_integer(&args[1])?;
    let sink = extract_integer(&args[2])?;
    
    if !graph.directed {
        return Err(VmError::Runtime("Maximum flow requires a directed graph".to_string()));
    }
    
    if !graph.vertices.contains(&source) || !graph.vertices.contains(&sink) {
        return Err(VmError::Runtime("Source or sink vertex not found in graph".to_string()));
    }

    // Create capacity matrix
    let mut capacity: HashMap<(i64, i64), f64> = HashMap::new();
    for (&u, neighbors) in &graph.adjacency {
        for &(v, weight) in neighbors {
            capacity.insert((u, v), weight);
        }
    }
    
    // DFS to find augmenting path
    fn dfs_path(current: i64, sink: i64, visited: &mut HashSet<i64>, 
                capacity: &HashMap<(i64, i64), f64>, graph: &Graph) -> Option<Vec<i64>> {
        if current == sink {
            return Some(vec![current]);
        }
        
        visited.insert(current);
        
        if let Some(neighbors) = graph.neighbors(current) {
            for &(next, _) in neighbors {
                if !visited.contains(&next) && capacity.get(&(current, next)).unwrap_or(&0.0) > &0.0 {
                    if let Some(mut path) = dfs_path(next, sink, visited, capacity, graph) {
                        path.insert(0, current);
                        return Some(path);
                    }
                }
            }
        }
        
        None
    }
    
    let mut max_flow = 0.0;
    
    loop {
        let mut visited = HashSet::new();
        if let Some(path) = dfs_path(source, sink, &mut visited, &capacity, graph) {
            // Find minimum capacity along path
            let mut path_flow = f64::INFINITY;
            for window in path.windows(2) {
                let (u, v) = (window[0], window[1]);
                path_flow = path_flow.min(*capacity.get(&(u, v)).unwrap_or(&0.0));
            }
            
            // Update capacities
            for window in path.windows(2) {
                let (u, v) = (window[0], window[1]);
                *capacity.entry((u, v)).or_insert(0.0) -= path_flow;
                *capacity.entry((v, u)).or_insert(0.0) += path_flow;
            }
            
            max_flow += path_flow;
        } else {
            break; // No more augmenting paths
        }
    }
    
    Ok(Value::Real(max_flow))
}

/// Find articulation points (cut vertices) using Tarjan's algorithm
/// Usage: ArticulationPoints[graph] -> Returns list of cut vertices
pub fn articulation_points(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Articulation points are defined for undirected graphs".to_string()));
    }
    
    let mut visited = HashSet::new();
    let mut discovery = HashMap::new();
    let mut low = HashMap::new();
    let mut parent = HashMap::new();
    let mut articulation_points = HashSet::new();
    let mut time = 0;
    
    fn dfs_articulation(u: i64, visited: &mut HashSet<i64>, discovery: &mut HashMap<i64, i64>,
                       low: &mut HashMap<i64, i64>, parent: &mut HashMap<i64, Option<i64>>,
                       articulation_points: &mut HashSet<i64>, time: &mut i64, graph: &Graph) {
        let mut children = 0;
        visited.insert(u);
        discovery.insert(u, *time);
        low.insert(u, *time);
        *time += 1;
        
        if let Some(neighbors) = graph.neighbors(u) {
            for &(v, _) in neighbors {
                if !visited.contains(&v) {
                    children += 1;
                    parent.insert(v, Some(u));
                    dfs_articulation(v, visited, discovery, low, parent, articulation_points, time, graph);
                    
                    low.insert(u, low[&u].min(low[&v]));
                    
                    // Root is articulation point if it has more than one child
                    if parent[&u].is_none() && children > 1 {
                        articulation_points.insert(u);
                    }
                    
                    // Non-root is articulation point if low[v] >= discovery[u]
                    if parent[&u].is_some() && low[&v] >= discovery[&u] {
                        articulation_points.insert(u);
                    }
                } else if Some(v) != parent[&u] {
                    low.insert(u, low[&u].min(discovery[&v]));
                }
            }
        }
    }
    
    // Initialize parent map
    for &vertex in &graph.vertices {
        parent.insert(vertex, None);
    }
    
    // Run DFS from all unvisited vertices
    for &vertex in &graph.vertices {
        if !visited.contains(&vertex) {
            dfs_articulation(vertex, &mut visited, &mut discovery, &mut low, 
                           &mut parent, &mut articulation_points, &mut time, graph);
        }
    }
    
    let mut result: Vec<Value> = articulation_points.into_iter()
        .map(|v| Value::Integer(v))
        .collect();
    result.sort_by_key(|v| match v {
        Value::Integer(i) => *i,
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Find bridges (cut edges) using Tarjan's algorithm
/// Usage: Bridges[graph] -> Returns list of bridge edges
pub fn bridges(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Bridges are defined for undirected graphs".to_string()));
    }
    
    let mut visited = HashSet::new();
    let mut discovery = HashMap::new();
    let mut low = HashMap::new();
    let mut parent = HashMap::new();
    let mut bridges = Vec::new();
    let mut time = 0;
    
    fn dfs_bridges(u: i64, visited: &mut HashSet<i64>, discovery: &mut HashMap<i64, i64>,
                  low: &mut HashMap<i64, i64>, parent: &mut HashMap<i64, Option<i64>>,
                  bridges: &mut Vec<(i64, i64)>, time: &mut i64, graph: &Graph) {
        visited.insert(u);
        discovery.insert(u, *time);
        low.insert(u, *time);
        *time += 1;
        
        if let Some(neighbors) = graph.neighbors(u) {
            for &(v, _) in neighbors {
                if !visited.contains(&v) {
                    parent.insert(v, Some(u));
                    dfs_bridges(v, visited, discovery, low, parent, bridges, time, graph);
                    
                    low.insert(u, low[&u].min(low[&v]));
                    
                    // Edge (u,v) is bridge if low[v] > discovery[u]
                    if low[&v] > discovery[&u] {
                        bridges.push((u.min(v), u.max(v))); // Canonical form
                    }
                } else if Some(v) != parent[&u] {
                    low.insert(u, low[&u].min(discovery[&v]));
                }
            }
        }
    }
    
    // Initialize parent map
    for &vertex in &graph.vertices {
        parent.insert(vertex, None);
    }
    
    // Run DFS from all unvisited vertices
    for &vertex in &graph.vertices {
        if !visited.contains(&vertex) {
            dfs_bridges(vertex, &mut visited, &mut discovery, &mut low, 
                       &mut parent, &mut bridges, &mut time, graph);
        }
    }
    
    // Remove duplicates and convert to result format
    bridges.sort();
    bridges.dedup();
    
    let result: Vec<Value> = bridges.into_iter()
        .map(|(u, v)| Value::List(vec![Value::Integer(u), Value::Integer(v)]))
        .collect();
    
    Ok(Value::List(result))
}

// ===============================
// GRAPH OPTIMIZATION ALGORITHMS
// ===============================

/// Graph coloring using greedy algorithm with Welsh-Powell heuristic
/// Usage: GraphColoring[graph] -> Returns vertex coloring map
pub fn graph_coloring(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Graph coloring is defined for undirected graphs".to_string()));
    }
    
    if graph.vertices.is_empty() {
        return Ok(Value::List(vec![]));
    }
    
    // Welsh-Powell algorithm: sort vertices by degree in descending order
    let mut vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    vertices.sort_by(|&a, &b| {
        let degree_a = graph.neighbors(a).map_or(0, |n| n.len());
        let degree_b = graph.neighbors(b).map_or(0, |n| n.len());
        degree_b.cmp(&degree_a) // Descending order
    });
    
    let mut coloring = HashMap::new();
    let mut color = 0;
    
    // Color vertices greedily
    for &vertex in &vertices {
        if coloring.contains_key(&vertex) {
            continue;
        }
        
        // Find neighbors' colors
        let mut used_colors = HashSet::new();
        if let Some(neighbors) = graph.neighbors(vertex) {
            for &(neighbor, _) in neighbors {
                if let Some(&neighbor_color) = coloring.get(&neighbor) {
                    used_colors.insert(neighbor_color);
                }
            }
        }
        
        // Find smallest available color
        while used_colors.contains(&color) {
            color += 1;
        }
        
        coloring.insert(vertex, color);
    }
    
    // Convert to list of {vertex, color} pairs
    let mut result: Vec<Value> = coloring.into_iter()
        .map(|(vertex, color)| {
            Value::List(vec![Value::Integer(vertex), Value::Integer(color)])
        })
        .collect();
    
    result.sort_by_key(|v| match v {
        Value::List(pair) if pair.len() == 2 => {
            match &pair[0] {
                Value::Integer(i) => *i,
                _ => 0,
            }
        }
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Find Hamiltonian path using backtracking
/// Usage: HamiltonianPath[graph] -> Returns Hamiltonian path if exists, otherwise Missing
pub fn hamiltonian_path(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.vertices.is_empty() {
        return Ok(Value::List(vec![]));
    }
    
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let n = vertices.len();
    
    fn has_hamiltonian_path_from(start: i64, path: &mut Vec<i64>, visited: &mut HashSet<i64>, 
                                target_length: usize, graph: &Graph) -> bool {
        if path.len() == target_length {
            return true;
        }
        
        if let Some(neighbors) = graph.neighbors(start) {
            for &(neighbor, _) in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    path.push(neighbor);
                    
                    if has_hamiltonian_path_from(neighbor, path, visited, target_length, graph) {
                        return true;
                    }
                    
                    // Backtrack
                    path.pop();
                    visited.remove(&neighbor);
                }
            }
        }
        
        false
    }
    
    // Try starting from each vertex
    for &start_vertex in &vertices {
        let mut path = vec![start_vertex];
        let mut visited = HashSet::new();
        visited.insert(start_vertex);
        
        if has_hamiltonian_path_from(start_vertex, &mut path, &mut visited, n, &graph) {
            let result: Vec<Value> = path.into_iter()
                .map(|v| Value::Integer(v))
                .collect();
            return Ok(Value::List(result));
        }
    }
    
    // No Hamiltonian path found
    Ok(Value::Symbol("Missing".to_string()))
}

/// Find Eulerian path using Hierholzer's algorithm
/// Usage: EulerianPath[graph] -> Returns Eulerian path if exists, otherwise Missing
pub fn eulerian_path(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut graph = extract_graph(&args[0])?;
    
    if graph.vertices.is_empty() {
        return Ok(Value::List(vec![]));
    }
    
    // Check if Eulerian path exists
    let mut odd_degree_vertices = Vec::new();
    for &vertex in &graph.vertices {
        let degree = graph.neighbors(vertex).map_or(0, |n| n.len());
        if degree % 2 == 1 {
            odd_degree_vertices.push(vertex);
        }
    }
    
    // Eulerian path exists if there are exactly 0 or 2 vertices with odd degree
    if odd_degree_vertices.len() != 0 && odd_degree_vertices.len() != 2 {
        return Ok(Value::Symbol("Missing".to_string()));
    }
    
    // Choose starting vertex
    let start_vertex = if odd_degree_vertices.is_empty() {
        *graph.vertices.iter().next().unwrap()
    } else {
        odd_degree_vertices[0]
    };
    
    // Hierholzer's algorithm
    let mut path = Vec::new();
    let mut stack = vec![start_vertex];
    let mut edge_used = HashSet::new();
    
    while let Some(current) = stack.last().cloned() {
        if let Some(neighbors) = graph.neighbors(current) {
            let mut found_edge = false;
            for &(neighbor, _) in neighbors {
                let edge_key = if current < neighbor {
                    (current, neighbor)
                } else {
                    (neighbor, current)
                };
                
                if !edge_used.contains(&edge_key) {
                    edge_used.insert(edge_key);
                    stack.push(neighbor);
                    found_edge = true;
                    break;
                }
            }
            
            if !found_edge {
                path.push(stack.pop().unwrap());
            }
        } else {
            path.push(stack.pop().unwrap());
        }
    }
    
    path.reverse();
    
    let result: Vec<Value> = path.into_iter()
        .map(|v| Value::Integer(v))
        .collect();
    
    Ok(Value::List(result))
}

/// Find minimum vertex cover using approximation algorithm
/// Usage: VertexCover[graph] -> Returns approximate minimum vertex cover
pub fn vertex_cover(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Vertex cover is defined for undirected graphs".to_string()));
    }
    
    let mut cover = HashSet::new();
    let mut edges_covered = HashSet::new();
    
    // Collect all edges
    let mut edges = Vec::new();
    for (&u, neighbors) in &graph.adjacency {
        for &(v, _) in neighbors {
            if u < v { // Avoid duplicates in undirected graph
                edges.push((u, v));
            }
        }
    }
    
    // Greedy 2-approximation algorithm
    for (u, v) in edges {
        let edge_key = (u.min(v), u.max(v));
        if !edges_covered.contains(&edge_key) {
            // Add both endpoints to cover (2-approximation)
            cover.insert(u);
            cover.insert(v);
            
            // Mark all edges incident to u and v as covered
            if let Some(u_neighbors) = graph.neighbors(u) {
                for &(neighbor, _) in u_neighbors {
                    edges_covered.insert((u.min(neighbor), u.max(neighbor)));
                }
            }
            if let Some(v_neighbors) = graph.neighbors(v) {
                for &(neighbor, _) in v_neighbors {
                    edges_covered.insert((v.min(neighbor), v.max(neighbor)));
                }
            }
        }
    }
    
    let mut result: Vec<Value> = cover.into_iter()
        .map(|v| Value::Integer(v))
        .collect();
    result.sort_by_key(|v| match v {
        Value::Integer(i) => *i,
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Find maximum independent set using complement of vertex cover
/// Usage: IndependentSet[graph] -> Returns approximate maximum independent set
pub fn independent_set(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Independent set is defined for undirected graphs".to_string()));
    }
    
    // Use complement of vertex cover as approximation for maximum independent set
    let vertex_cover_result = vertex_cover(args)?;
    
    let cover_vertices = match vertex_cover_result {
        Value::List(list) => {
            let mut cover = HashSet::new();
            for item in list {
                if let Value::Integer(v) = item {
                    cover.insert(v);
                }
            }
            cover
        }
        _ => return Err(VmError::Runtime("Invalid vertex cover result".to_string())),
    };
    
    // Independent set is all vertices not in the vertex cover
    let mut independent: Vec<Value> = graph.vertices.iter()
        .filter(|&v| !cover_vertices.contains(v))
        .map(|&v| Value::Integer(v))
        .collect();
    
    independent.sort_by_key(|v| match v {
        Value::Integer(i) => *i,
        _ => 0,
    });
    
    Ok(Value::List(independent))
}

// ===============================
// GRAPH ANALYSIS ALGORITHMS
// ===============================

/// Compute betweenness centrality for all vertices
/// Usage: BetweennessCentrality[graph] -> Returns centrality scores for each vertex
pub fn betweenness_centrality(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let mut centrality = HashMap::new();
    
    // Initialize centrality scores
    for &v in &vertices {
        centrality.insert(v, 0.0);
    }
    
    // Brandes' algorithm for betweenness centrality
    for &s in &vertices {
        let mut stack = Vec::new();
        let mut predecessors: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut sigma: HashMap<i64, f64> = HashMap::new();
        let mut distance: HashMap<i64, i64> = HashMap::new();
        let mut delta: HashMap<i64, f64> = HashMap::new();
        
        // Initialize
        for &v in &vertices {
            predecessors.insert(v, Vec::new());
            sigma.insert(v, 0.0);
            distance.insert(v, -1);
            delta.insert(v, 0.0);
        }
        
        sigma.insert(s, 1.0);
        distance.insert(s, 0);
        
        let mut queue = VecDeque::new();
        queue.push_back(s);
        
        // BFS
        while let Some(v) = queue.pop_front() {
            stack.push(v);
            
            if let Some(neighbors) = graph.neighbors(v) {
                for &(w, _) in neighbors {
                    // First time we found shortest path to w?
                    if distance[&w] < 0 {
                        queue.push_back(w);
                        distance.insert(w, distance[&v] + 1);
                    }
                    
                    // Shortest path to w via v?
                    if distance[&w] == distance[&v] + 1 {
                        sigma.insert(w, sigma[&w] + sigma[&v]);
                        predecessors.get_mut(&w).unwrap().push(v);
                    }
                }
            }
        }
        
        // Accumulation
        while let Some(w) = stack.pop() {
            for &v in &predecessors[&w] {
                let contribution = (sigma[&v] / sigma[&w]) * (1.0 + delta[&w]);
                delta.insert(v, delta[&v] + contribution);
            }
            
            if w != s {
                centrality.insert(w, centrality[&w] + delta[&w]);
            }
        }
    }
    
    // Normalize for undirected graphs
    if !graph.directed {
        for &v in &vertices {
            centrality.insert(v, centrality[&v] / 2.0);
        }
    }
    
    // Convert to result format
    let mut result: Vec<Value> = centrality.into_iter()
        .map(|(vertex, score)| {
            Value::List(vec![Value::Integer(vertex), Value::Real(score)])
        })
        .collect();
    
    result.sort_by_key(|v| match v {
        Value::List(pair) if pair.len() == 2 => {
            match &pair[0] {
                Value::Integer(i) => *i,
                _ => 0,
            }
        }
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Compute closeness centrality for all vertices
/// Usage: ClosenessCentrality[graph] -> Returns centrality scores for each vertex
pub fn closeness_centrality(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let mut centrality = HashMap::new();
    
    for &source in &vertices {
        let distances = shortest_path_distances(&graph, source);
        
        let mut total_distance = 0.0;
        let mut reachable = 0;
        
        for &target in &vertices {
            if let Some(&dist) = distances.get(&target) {
                if dist > 0 && dist < i64::MAX {
                    total_distance += dist as f64;
                    reachable += 1;
                }
            }
        }
        
        let centrality_score = if total_distance > 0.0 && reachable > 0 {
            (reachable as f64) / total_distance
        } else {
            0.0
        };
        
        centrality.insert(source, centrality_score);
    }
    
    // Convert to result format
    let mut result: Vec<Value> = centrality.into_iter()
        .map(|(vertex, score)| {
            Value::List(vec![Value::Integer(vertex), Value::Real(score)])
        })
        .collect();
    
    result.sort_by_key(|v| match v {
        Value::List(pair) if pair.len() == 2 => {
            match &pair[0] {
                Value::Integer(i) => *i,
                _ => 0,
            }
        }
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Helper function to compute shortest path distances from a source vertex
fn shortest_path_distances(graph: &Graph, source: i64) -> HashMap<i64, i64> {
    let mut distances = HashMap::new();
    let mut queue = VecDeque::new();
    
    distances.insert(source, 0);
    queue.push_back(source);
    
    while let Some(current) = queue.pop_front() {
        if let Some(neighbors) = graph.neighbors(current) {
            for &(neighbor, _) in neighbors {
                if !distances.contains_key(&neighbor) {
                    distances.insert(neighbor, distances[&current] + 1);
                    queue.push_back(neighbor);
                }
            }
        }
    }
    
    distances
}

/// Compute PageRank centrality using power iteration
/// Usage: PageRank[graph] -> Returns PageRank scores for each vertex
pub fn pagerank(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (graph, [damping_factor], [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    let damping_factor = if args.len() > 1 {
        match &args[1] {
            Value::Real(d) => *d,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real number for damping factor".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        0.85
    };
    
    let max_iterations = if args.len() > 2 {
        match &args[2] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        100
    };
    
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let n = vertices.len() as f64;
    
    if n == 0.0 {
        return Ok(Value::List(vec![]));
    }
    
    // Initialize PageRank values
    let mut pagerank = HashMap::new();
    for &v in &vertices {
        pagerank.insert(v, 1.0 / n);
    }
    
    // Compute out-degrees
    let mut out_degree = HashMap::new();
    for &v in &vertices {
        let degree = graph.neighbors(v).map_or(0, |neighbors| neighbors.len());
        out_degree.insert(v, degree.max(1)); // Avoid division by zero
    }
    
    // Power iteration
    for _ in 0..max_iterations {
        let mut new_pagerank = HashMap::new();
        
        // Initialize with teleportation probability
        for &v in &vertices {
            new_pagerank.insert(v, (1.0 - damping_factor) / n);
        }
        
        // Add contributions from incoming edges
        for &v in &vertices {
            if let Some(neighbors) = graph.neighbors(v) {
                let contribution = damping_factor * pagerank[&v] / (out_degree[&v] as f64);
                for &(neighbor, _) in neighbors {
                    new_pagerank.insert(neighbor, new_pagerank[&neighbor] + contribution);
                }
            }
        }
        
        pagerank = new_pagerank;
    }
    
    // Convert to result format
    let mut result: Vec<Value> = pagerank.into_iter()
        .map(|(vertex, score)| {
            Value::List(vec![Value::Integer(vertex), Value::Real(score)])
        })
        .collect();
    
    result.sort_by_key(|v| match v {
        Value::List(pair) if pair.len() == 2 => {
            match &pair[0] {
                Value::Integer(i) => *i,
                _ => 0,
            }
        }
        _ => 0,
    });
    
    Ok(Value::List(result))
}

/// Compute HITS (Hyperlink-Induced Topic Search) authority and hub scores
/// Usage: HITS[graph] -> Returns {vertex, authority_score, hub_score} tuples
pub fn hits(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (graph, [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if !graph.directed {
        return Err(VmError::Runtime("HITS algorithm requires a directed graph".to_string()));
    }
    
    let max_iterations = if args.len() > 1 {
        match &args[1] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        100
    };
    
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let n = vertices.len() as f64;
    
    if n == 0.0 {
        return Ok(Value::List(vec![]));
    }
    
    // Initialize authority and hub scores
    let mut authority = HashMap::new();
    let mut hub = HashMap::new();
    
    for &v in &vertices {
        authority.insert(v, 1.0 / n.sqrt());
        hub.insert(v, 1.0 / n.sqrt());
    }
    
    // Power iteration
    for _ in 0..max_iterations {
        let mut new_authority = HashMap::new();
        let mut new_hub = HashMap::new();
        
        // Initialize
        for &v in &vertices {
            new_authority.insert(v, 0.0);
            new_hub.insert(v, 0.0);
        }
        
        // Update authority scores: authority(p) = sum of hub scores of pages linking to p
        for &v in &vertices {
            if let Some(neighbors) = graph.neighbors(v) {
                for &(neighbor, _) in neighbors {
                    new_authority.insert(neighbor, new_authority[&neighbor] + hub[&v]);
                }
            }
        }
        
        // Update hub scores: hub(p) = sum of authority scores of pages p links to
        for &v in &vertices {
            if let Some(neighbors) = graph.neighbors(v) {
                for &(neighbor, _) in neighbors {
                    new_hub.insert(v, new_hub[&v] + authority[&neighbor]);
                }
            }
        }
        
        // Normalize scores
        let auth_norm: f64 = new_authority.values().map(|&x| x * x).sum::<f64>().sqrt();
        let hub_norm: f64 = new_hub.values().map(|&x| x * x).sum::<f64>().sqrt();
        
        if auth_norm > 0.0 {
            for &v in &vertices {
                new_authority.insert(v, new_authority[&v] / auth_norm);
            }
        }
        
        if hub_norm > 0.0 {
            for &v in &vertices {
                new_hub.insert(v, new_hub[&v] / hub_norm);
            }
        }
        
        authority = new_authority;
        hub = new_hub;
    }
    
    // Convert to result format
    let mut result: Vec<Value> = vertices.into_iter()
        .map(|vertex| {
            Value::List(vec![
                Value::Integer(vertex),
                Value::Real(authority[&vertex]),
                Value::Real(hub[&vertex])
            ])
        })
        .collect();
    
    result.sort_by_key(|v| match v {
        Value::List(tuple) if tuple.len() == 3 => {
            match &tuple[0] {
                Value::Integer(i) => *i,
                _ => 0,
            }
        }
        _ => 0,
    });
    
    Ok(Value::List(result))
}
/// Simple community detection using connected components
/// Usage: CommunityDetection[graph] -> Returns communities as list of vertex lists
pub fn community_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = extract_graph(&args[0])?;
    
    if graph.directed {
        return Err(VmError::Runtime("Community detection implemented for undirected graphs".to_string()));
    }
    
    let mut visited = HashSet::new();
    let mut communities = Vec::new();
    
    // Find connected components as communities
    for &vertex in &graph.vertices {
        if !visited.contains(&vertex) {
            let mut component = Vec::new();
            let mut stack = vec![vertex];
            
            while let Some(current) = stack.pop() {
                if !visited.contains(&current) {
                    visited.insert(current);
                    component.push(current);
                    
                    if let Some(neighbors) = graph.neighbors(current) {
                        for &(neighbor, _) in neighbors {
                            if !visited.contains(&neighbor) {
                                stack.push(neighbor);
                            }
                        }
                    }
                }
            }
            
            component.sort();
            communities.push(component);
        }
    }
    
    // Sort communities by size (largest first)
    communities.sort_by(|a, b| b.len().cmp(&a.len()));
    
    // Convert to result format
    let result: Vec<Value> = communities.into_iter()
        .map(|community| {
            let members: Vec<Value> = community.into_iter()
                .map(|v| Value::Integer(v))
                .collect();
            Value::List(members)
        })
        .collect();
    
    Ok(Value::List(result))
}

/// Simple graph isomorphism check using degree sequence and basic structural properties
/// Usage: GraphIsomorphism[graph1, graph2] -> Returns True if potentially isomorphic, False otherwise
pub fn graph_isomorphism(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (graph1, graph2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph1 = extract_graph(&args[0])?;
    let graph2 = extract_graph(&args[1])?;
    
    // Basic necessary conditions for isomorphism
    
    // Same number of vertices
    if graph1.vertices.len() != graph2.vertices.len() {
        return Ok(Value::Integer(0)); // False
    }
    
    // Same directedness
    if graph1.directed != graph2.directed {
        return Ok(Value::Integer(0)); // False
    }
    
    // Count edges
    let edges1 = graph1.adjacency.values()
        .map(|neighbors| neighbors.len())
        .sum::<usize>();
    let edges2 = graph2.adjacency.values()
        .map(|neighbors| neighbors.len())
        .sum::<usize>();
    
    if edges1 != edges2 {
        return Ok(Value::Integer(0)); // False
    }
    
    // Check degree sequences
    let mut degrees1: Vec<usize> = graph1.vertices.iter()
        .map(|&v| graph1.neighbors(v).map_or(0, |n| n.len()))
        .collect();
    let mut degrees2: Vec<usize> = graph2.vertices.iter()
        .map(|&v| graph2.neighbors(v).map_or(0, |n| n.len()))
        .collect();
    
    degrees1.sort();
    degrees2.sort();
    
    if degrees1 != degrees2 {
        return Ok(Value::Integer(0)); // False
    }
    
    // Check degree distribution
    let mut degree_count1 = HashMap::new();
    let mut degree_count2 = HashMap::new();
    
    for degree in degrees1 {
        *degree_count1.entry(degree).or_insert(0) += 1;
    }
    
    for degree in degrees2 {
        *degree_count2.entry(degree).or_insert(0) += 1;
    }
    
    if degree_count1 != degree_count2 {
        return Ok(Value::Integer(0)); // False
    }
    
    // For very small graphs, try to verify more thoroughly
    if graph1.vertices.len() <= 6 {
        // Check triangle counts (simple structural invariant)
        let triangles1 = count_triangles(&graph1);
        let triangles2 = count_triangles(&graph2);
        
        if triangles1 != triangles2 {
            return Ok(Value::Integer(0)); // False
        }
    }
    
    // If all basic tests pass, graphs are potentially isomorphic
    // Note: This is not a complete isomorphism test, just necessary conditions
    Ok(Value::Integer(1)) // True (potentially isomorphic)
}

/// Helper function to count triangles in a graph
fn count_triangles(graph: &Graph) -> usize {
    let mut triangle_count = 0;
    
    for &u in &graph.vertices {
        if let Some(u_neighbors) = graph.neighbors(u) {
            for &(v, _) in u_neighbors {
                if u < v { // Avoid double counting
                    if let Some(v_neighbors) = graph.neighbors(v) {
                        for &(w, _) in v_neighbors {
                            if v < w && graph.has_edge(u, w) {
                                triangle_count += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    
    triangle_count
}
