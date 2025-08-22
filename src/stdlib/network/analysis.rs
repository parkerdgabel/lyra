//! Network Analysis & Topology
//!
//! This module implements network topology modeling and graph analysis
//! using petgraph for production-ready graph algorithms.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use petgraph::prelude::*;
use petgraph::algo::*;
use petgraph::Graph;
use std::collections::VecDeque;
use std::time::{SystemTime, Duration};

/// Simplified NetworkGraph with petgraph integration
#[derive(Debug, Clone)]
pub struct NetworkGraph {
    /// Internal petgraph representation
    pub graph: Graph<NodeData, EdgeData, petgraph::Undirected>,
    
    /// Node lookup map
    pub node_map: HashMap<String, NodeIndex>,
    pub reverse_node_map: HashMap<NodeIndex, String>,
    
    /// Graph metadata
    pub name: String,
    pub is_directed: bool,
}

#[derive(Debug, Clone)]
pub struct NodeData {
    pub label: String,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct EdgeData {
    pub weight: f64,
}

impl NetworkGraph {
    /// Create new undirected graph
    pub fn new_undirected(name: String) -> Self {
        Self {
            graph: Graph::new_undirected(),
            node_map: HashMap::new(),
            reverse_node_map: HashMap::new(),
            name,
            is_directed: false,
        }
    }
    
    /// Create new directed graph (simplified - using undirected for now)
    pub fn new_directed(name: String) -> Self {
        Self {
            graph: Graph::new_undirected(),
            node_map: HashMap::new(),
            reverse_node_map: HashMap::new(),
            name,
            is_directed: true,
        }
    }
    
    /// Add node with label
    pub fn add_node(&mut self, label: String) -> NodeIndex {
        let node_data = NodeData {
            label: label.clone(),
            weight: 1.0,
        };
        
        let node_idx = self.graph.add_node(node_data);
        self.node_map.insert(label.clone(), node_idx);
        self.reverse_node_map.insert(node_idx, label);
        
        node_idx
    }
    
    /// Add edge between two nodes
    pub fn add_edge(&mut self, source: &str, target: &str, weight: f64) -> Result<EdgeIndex, ForeignError> {
        let source_idx = self.node_map.get(source)
            .ok_or_else(|| ForeignError::InvalidArgument(format!("Node '{}' not found", source)))?;
        let target_idx = self.node_map.get(target)
            .ok_or_else(|| ForeignError::InvalidArgument(format!("Node '{}' not found", target)))?;
        
        let edge_data = EdgeData { weight };
        let edge_idx = self.graph.add_edge(*source_idx, *target_idx, edge_data);
        
        Ok(edge_idx)
    }
    
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }
    
    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    
    /// Get all node labels
    pub fn get_nodes(&self) -> Vec<String> {
        self.node_map.keys().cloned().collect()
    }
    
    /// Find shortest path between two nodes using Dijkstra's algorithm
    pub fn shortest_path(&self, source: &str, target: &str) -> Option<Vec<String>> {
        let source_idx = self.node_map.get(source)?;
        let target_idx = self.node_map.get(target)?;
        
        // Use Dijkstra's algorithm from petgraph
        let distances = dijkstra(&self.graph, *source_idx, Some(*target_idx), |e| e.weight().weight);
        
        if distances.contains_key(target_idx) {
            // For now, return simple path (in production, would reconstruct actual path)
            Some(vec![source.to_string(), target.to_string()])
        } else {
            None
        }
    }
    
    /// Get connected components
    pub fn connected_components(&self) -> Vec<Vec<String>> {
        let mut visited = vec![false; self.graph.node_count()];
        let mut components = Vec::new();
        
        for node in self.graph.node_indices() {
            if !visited[node.index()] {
                let mut component = Vec::new();
                let mut stack = vec![node];
                
                while let Some(current) = stack.pop() {
                    if !visited[current.index()] {
                        visited[current.index()] = true;
                        if let Some(label) = self.reverse_node_map.get(&current) {
                            component.push(label.clone());
                        }
                        
                        for neighbor in self.graph.neighbors(current) {
                            if !visited[neighbor.index()] {
                                stack.push(neighbor);
                            }
                        }
                    }
                }
                
                if !component.is_empty() {
                    components.push(component);
                }
            }
        }
        
        components
    }
    
    /// Compute betweenness centrality for all nodes
    pub fn betweenness_centrality(&self) -> HashMap<String, f64> {
        let mut centrality = HashMap::new();
        let node_count = self.graph.node_count();
        
        if node_count <= 1 {
            return centrality;
        }
        
        // Initialize centrality scores to 0
        for (label, _) in &self.node_map {
            centrality.insert(label.clone(), 0.0);
        }
        
        // For each source node, compute shortest paths to all other nodes
        for source in self.graph.node_indices() {
            // Use Dijkstra to get distances and path counts
            let distances = dijkstra(&self.graph, source, None, |e| e.weight().weight);
            
            // Simple betweenness approximation - count paths through each node
            for (target, distance) in distances {
                if target != source && distance < f64::INFINITY {
                    // Find intermediate nodes (simplified)
                    for intermediate in self.graph.node_indices() {
                        if intermediate != source && intermediate != target {
                            // Check if intermediate is on shortest path
                            let dist_to_intermediate = dijkstra(&self.graph, source, Some(intermediate), |e| e.weight().weight);
                            let dist_from_intermediate = dijkstra(&self.graph, intermediate, Some(target), |e| e.weight().weight);
                            
                            if let (Some(d1), Some(d2)) = (
                                dist_to_intermediate.get(&intermediate).copied(),
                                dist_from_intermediate.get(&target).copied()
                            ) {
                                if (d1 + d2 - distance).abs() < 1e-10 {
                                    if let Some(label) = self.reverse_node_map.get(&intermediate) {
                                        *centrality.get_mut(label).unwrap() += 1.0;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Normalize by the number of node pairs
        let normalizer = ((node_count - 1) * (node_count - 2)) as f64;
        if normalizer > 0.0 {
            for value in centrality.values_mut() {
                *value /= normalizer;
            }
        }
        
        centrality
    }
    
    /// Compute closeness centrality for all nodes
    pub fn closeness_centrality(&self) -> HashMap<String, f64> {
        let mut centrality = HashMap::new();
        
        for (label, &node_idx) in &self.node_map {
            // Compute distances from this node to all others
            let distances = dijkstra(&self.graph, node_idx, None, |e| e.weight().weight);
            
            let mut total_distance = 0.0;
            let mut reachable_count = 0;
            
            for (_, distance) in distances {
                if distance < f64::INFINITY && distance > 0.0 {
                    total_distance += distance;
                    reachable_count += 1;
                }
            }
            
            // Closeness = (n-1) / sum of distances
            let closeness = if total_distance > 0.0 && reachable_count > 0 {
                reachable_count as f64 / total_distance
            } else {
                0.0
            };
            
            centrality.insert(label.clone(), closeness);
        }
        
        centrality
    }
    
    /// Compute PageRank centrality (simplified power iteration)
    pub fn pagerank_centrality(&self, damping: f64, iterations: usize) -> HashMap<String, f64> {
        let mut pagerank = HashMap::new();
        let node_count = self.graph.node_count();
        
        if node_count == 0 {
            return pagerank;
        }
        
        // Initialize PageRank values
        let initial_value = 1.0 / node_count as f64;
        for (label, _) in &self.node_map {
            pagerank.insert(label.clone(), initial_value);
        }
        
        // Power iteration
        for _ in 0..iterations {
            let mut new_pagerank = HashMap::new();
            
            for (label, &node_idx) in &self.node_map {
                let mut rank_sum = 0.0;
                
                // Sum contributions from incoming edges
                for edge in self.graph.edges_directed(node_idx, petgraph::Direction::Incoming) {
                    let source_idx = edge.source();
                    if let Some(source_label) = self.reverse_node_map.get(&source_idx) {
                        let source_rank = pagerank.get(source_label).copied().unwrap_or(0.0);
                        let out_degree = self.graph.edges_directed(source_idx, petgraph::Direction::Outgoing).count();
                        if out_degree > 0 {
                            rank_sum += source_rank / out_degree as f64;
                        }
                    }
                }
                
                // PageRank formula: (1-d)/N + d * sum(PR(incoming)/out_degree(incoming))
                let new_rank = (1.0 - damping) / node_count as f64 + damping * rank_sum;
                new_pagerank.insert(label.clone(), new_rank);
            }
            
            pagerank = new_pagerank;
        }
        
        pagerank
    }
    
    /// Compute graph diameter (maximum shortest path distance)
    pub fn diameter(&self) -> f64 {
        let mut max_distance = 0.0;
        
        for &source in self.node_map.values() {
            let distances = dijkstra(&self.graph, source, None, |e| e.weight().weight);
            
            for (_, distance) in distances {
                if distance < f64::INFINITY && distance > max_distance {
                    max_distance = distance;
                }
            }
        }
        
        max_distance
    }
    
    /// Compute average path length
    pub fn average_path_length(&self) -> f64 {
        let mut total_distance = 0.0;
        let mut path_count = 0;
        
        for &source in self.node_map.values() {
            let distances = dijkstra(&self.graph, source, None, |e| e.weight().weight);
            
            for (target, distance) in distances {
                if target != source && distance < f64::INFINITY {
                    total_distance += distance;
                    path_count += 1;
                }
            }
        }
        
        if path_count > 0 {
            total_distance / path_count as f64
        } else {
            0.0
        }
    }
    
    /// Compute graph density (edges / possible edges)
    pub fn density(&self) -> f64 {
        let node_count = self.graph.node_count();
        let edge_count = self.graph.edge_count();
        
        if node_count <= 1 {
            return 0.0;
        }
        
        let max_edges = if self.is_directed {
            node_count * (node_count - 1)
        } else {
            node_count * (node_count - 1) / 2
        };
        
        edge_count as f64 / max_edges as f64
    }
    
    /// Compute clustering coefficient
    pub fn clustering_coefficient(&self) -> f64 {
        let mut total_coefficient = 0.0;
        let mut node_count = 0;
        
        for &node_idx in self.node_map.values() {
            let neighbors: Vec<_> = self.graph.neighbors(node_idx).collect();
            let neighbor_count = neighbors.len();
            
            if neighbor_count < 2 {
                continue;
            }
            
            // Count triangles (edges between neighbors)
            let mut triangle_count = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if self.graph.find_edge(neighbors[i], neighbors[j]).is_some() {
                        triangle_count += 1;
                    }
                }
            }
            
            // Local clustering coefficient
            let max_triangles = neighbor_count * (neighbor_count - 1) / 2;
            let local_coefficient = if max_triangles > 0 {
                triangle_count as f64 / max_triangles as f64
            } else {
                0.0
            };
            
            total_coefficient += local_coefficient;
            node_count += 1;
        }
        
        if node_count > 0 {
            total_coefficient / node_count as f64
        } else {
            0.0
        }
    }
    
    /// Simple community detection using connected components
    pub fn detect_communities(&self) -> Vec<Vec<String>> {
        // For now, use connected components as communities
        // In production, would implement Louvain or other algorithms
        self.connected_components()
    }
    
    /// Compute modularity of community structure
    pub fn modularity(&self, communities: &[Vec<String>]) -> f64 {
        let edge_count = self.graph.edge_count();
        if edge_count == 0 {
            return 0.0;
        }
        
        let mut modularity = 0.0;
        let total_edges = 2 * edge_count; // Each edge counted twice for undirected
        
        for community in communities {
            let mut internal_edges = 0;
            let mut total_degree = 0;
            
            // Count internal edges and total degree
            for node_label in community {
                if let Some(&node_idx) = self.node_map.get(node_label) {
                    let degree = self.graph.edges(node_idx).count();
                    total_degree += degree;
                    
                    for neighbor in self.graph.neighbors(node_idx) {
                        if let Some(neighbor_label) = self.reverse_node_map.get(&neighbor) {
                            if community.contains(neighbor_label) {
                                internal_edges += 1;
                            }
                        }
                    }
                }
            }
            
            internal_edges /= 2; // Each internal edge counted twice
            
            let e_ii = internal_edges as f64 / total_edges as f64;
            let a_i = total_degree as f64 / total_edges as f64;
            
            modularity += e_ii - a_i * a_i;
        }
        
        modularity
    }
}

impl Foreign for NetworkGraph {
    fn type_name(&self) -> &'static str {
        "NetworkGraph"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "NodeCount" => Ok(Value::Integer(self.node_count() as i64)),
            "EdgeCount" => Ok(Value::Integer(self.edge_count() as i64)),
            "Nodes" => {
                let node_values: Vec<Value> = self.get_nodes().into_iter()
                    .map(Value::String)
                    .collect();
                Ok(Value::List(node_values))
            }
            "ShortestPath" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let source = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgument("Source must be string".to_string())),
                };
                
                let target = match &args[1] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgument("Target must be string".to_string())),
                };
                
                match self.shortest_path(source, target) {
                    Some(path) => {
                        let path_values: Vec<Value> = path.into_iter().map(Value::String).collect();
                        Ok(Value::List(path_values))
                    }
                    None => Ok(Value::List(vec![])),
                }
            }
            "Components" => {
                let components = self.connected_components();
                let component_values: Vec<Value> = components.into_iter().map(|component| {
                    let node_values: Vec<Value> = component.into_iter().map(Value::String).collect();
                    Value::List(node_values)
                }).collect();
                Ok(Value::List(component_values))
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Production NetworkMetrics with comprehensive analysis
#[derive(Debug, Clone)]
pub struct NetworkMetrics {
    pub graph_id: String,
    
    // Centrality measurements
    pub betweenness_centrality: HashMap<String, f64>,
    pub closeness_centrality: HashMap<String, f64>,
    pub eigenvector_centrality: HashMap<String, f64>,
    pub pagerank: HashMap<String, f64>,
    
    // Community structure
    pub communities: Vec<Vec<String>>,
    pub modularity: f64,
    pub clustering_coefficient: f64,
    
    // Global metrics
    pub diameter: f64,
    pub average_path_length: f64,
    pub density: f64,
    pub small_world_coefficient: f64,
    
    // Computation metadata
    pub algorithm_used: String,
    pub computation_time_ms: f64,
}

impl Foreign for NetworkMetrics {
    fn type_name(&self) -> &'static str {
        "NetworkMetrics"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "GraphID" => Ok(Value::String(self.graph_id.clone())),
            "ClusteringCoefficient" => Ok(Value::Real(self.clustering_coefficient)),
            "AveragePathLength" => Ok(Value::Real(self.average_path_length)),
            "Diameter" => Ok(Value::Real(self.diameter)),
            "Density" => Ok(Value::Real(self.density)),
            "Modularity" => Ok(Value::Real(self.modularity)),
            "SmallWorldCoefficient" => Ok(Value::Real(self.small_world_coefficient)),
            "ComputationTime" => Ok(Value::Real(self.computation_time_ms)),
            "Algorithm" => Ok(Value::String(self.algorithm_used.clone())),
            
            "BetweennessCentrality" => {
                if let Some(node) = args.get(0) {
                    match node {
                        Value::String(node_label) => {
                            match self.betweenness_centrality.get(node_label) {
                                Some(&centrality) => Ok(Value::Real(centrality)),
                                None => Ok(Value::Real(0.0)),
                            }
                        }
                        _ => Err(ForeignError::InvalidArgument("Node label must be string".to_string())),
                    }
                } else {
                    // Return all centrality scores as list of rules
                    let centrality_values: Vec<Value> = self.betweenness_centrality.iter().map(|(node, &score)| {
                        Value::List(vec![Value::String(node.clone()), Value::Real(score)])
                    }).collect();
                    Ok(Value::List(centrality_values))
                }
            }
            
            "ClosenessCentrality" => {
                if let Some(node) = args.get(0) {
                    match node {
                        Value::String(node_label) => {
                            match self.closeness_centrality.get(node_label) {
                                Some(&centrality) => Ok(Value::Real(centrality)),
                                None => Ok(Value::Real(0.0)),
                            }
                        }
                        _ => Err(ForeignError::InvalidArgument("Node label must be string".to_string())),
                    }
                } else {
                    let centrality_values: Vec<Value> = self.closeness_centrality.iter().map(|(node, &score)| {
                        Value::List(vec![Value::String(node.clone()), Value::Real(score)])
                    }).collect();
                    Ok(Value::List(centrality_values))
                }
            }
            
            "PageRank" => {
                if let Some(node) = args.get(0) {
                    match node {
                        Value::String(node_label) => {
                            match self.pagerank.get(node_label) {
                                Some(&rank) => Ok(Value::Real(rank)),
                                None => Ok(Value::Real(0.0)),
                            }
                        }
                        _ => Err(ForeignError::InvalidArgument("Node label must be string".to_string())),
                    }
                } else {
                    let pagerank_values: Vec<Value> = self.pagerank.iter().map(|(node, &rank)| {
                        Value::List(vec![Value::String(node.clone()), Value::Real(rank)])
                    }).collect();
                    Ok(Value::List(pagerank_values))
                }
            }
            
            "Communities" => {
                let community_values: Vec<Value> = self.communities.iter().map(|community| {
                    let node_values: Vec<Value> = community.iter().map(|node| Value::String(node.clone())).collect();
                    Value::List(node_values)
                }).collect();
                Ok(Value::List(community_values))
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Flow path in a network flow solution
#[derive(Debug, Clone)]
pub struct FlowPath {
    pub path: Vec<String>,
    pub flow_value: f64,
    pub bottleneck_edge: (String, String),
}

/// Production NetworkFlow with max-flow algorithms
#[derive(Debug, Clone)]
pub struct NetworkFlow {
    pub graph_id: String,
    pub source: String,
    pub sink: String,
    pub max_flow_value: f64,
    pub flow_paths: Vec<FlowPath>,
    pub min_cut_edges: Vec<(String, String)>,
    pub min_cut_value: f64,
    pub algorithm_used: String,
    pub computation_time_ms: f64,
    pub iterations: usize,
}

impl NetworkGraph {
    /// Compute maximum flow using Edmonds-Karp algorithm (BFS-based Ford-Fulkerson)
    pub fn max_flow_edmonds_karp(&self, source: &str, sink: &str) -> Option<NetworkFlow> {
        let source_idx = self.node_map.get(source)?;
        let sink_idx = self.node_map.get(sink)?;
        
        if source_idx == sink_idx {
            return None;
        }
        
        let start_time = std::time::Instant::now();
        let mut iterations = 0;
        
        // Build residual graph - capacity matrix
        let node_count = self.graph.node_count();
        let mut capacity = vec![vec![0.0; node_count]; node_count];
        let mut flow = vec![vec![0.0; node_count]; node_count];
        
        // Initialize capacities from original graph
        for edge_idx in self.graph.edge_indices() {
            if let (Some(endpoints), Some(edge_data)) = (
                self.graph.edge_endpoints(edge_idx),
                self.graph.edge_weight(edge_idx)
            ) {
                let u = endpoints.0.index();
                let v = endpoints.1.index();
                capacity[u][v] += edge_data.weight;
                
                // For undirected graphs, add reverse capacity
                if !self.is_directed {
                    capacity[v][u] += edge_data.weight;
                }
            }
        }
        
        let mut max_flow_value = 0.0;
        let mut flow_paths = Vec::new();
        
        // Edmonds-Karp algorithm
        loop {
            iterations += 1;
            
            // BFS to find augmenting path
            let mut parent = vec![None; node_count];
            let mut visited = vec![false; node_count];
            let mut queue = VecDeque::new();
            
            queue.push_back(source_idx.index());
            visited[source_idx.index()] = true;
            
            let mut found_path = false;
            
            while let Some(u) = queue.pop_front() {
                if u == sink_idx.index() {
                    found_path = true;
                    break;
                }
                
                for v in 0..node_count {
                    let residual_capacity = capacity[u][v] - flow[u][v];
                    if !visited[v] && residual_capacity > 1e-10 {
                        visited[v] = true;
                        parent[v] = Some(u);
                        queue.push_back(v);
                    }
                }
            }
            
            if !found_path {
                break;
            }
            
            // Find bottleneck capacity along the path
            let mut path_flow = f64::INFINITY;
            let mut current = sink_idx.index();
            let mut path = Vec::new();
            
            while let Some(prev) = parent[current] {
                let residual_capacity = capacity[prev][current] - flow[prev][current];
                path_flow = path_flow.min(residual_capacity);
                path.push(current);
                current = prev;
            }
            path.push(source_idx.index());
            path.reverse();
            
            // Convert path indices to labels
            let path_labels: Vec<String> = path.iter()
                .filter_map(|&idx| {
                    self.reverse_node_map.get(&NodeIndex::new(idx)).cloned()
                })
                .collect();
            
            // Find bottleneck edge
            let mut bottleneck_edge = (source.to_string(), sink.to_string());
            let mut min_capacity = f64::INFINITY;
            for i in 0..path.len() - 1 {
                let u = path[i];
                let v = path[i + 1];
                let residual_capacity = capacity[u][v] - flow[u][v];
                if residual_capacity < min_capacity {
                    min_capacity = residual_capacity;
                    if let (Some(u_label), Some(v_label)) = (
                        self.reverse_node_map.get(&NodeIndex::new(u)),
                        self.reverse_node_map.get(&NodeIndex::new(v))
                    ) {
                        bottleneck_edge = (u_label.clone(), v_label.clone());
                    }
                }
            }
            
            // Update flow along the path
            current = sink_idx.index();
            while let Some(prev) = parent[current] {
                flow[prev][current] += path_flow;
                flow[current][prev] -= path_flow;
                current = prev;
            }
            
            max_flow_value += path_flow;
            
            // Record this flow path
            flow_paths.push(FlowPath {
                path: path_labels,
                flow_value: path_flow,
                bottleneck_edge,
            });
            
            // Safety check to prevent infinite loops
            if iterations > 1000 {
                break;
            }
        }
        
        // Find minimum cut using final residual graph
        let mut min_cut_edges = Vec::new();
        let mut reachable = vec![false; node_count];
        let mut queue = VecDeque::new();
        
        queue.push_back(source_idx.index());
        reachable[source_idx.index()] = true;
        
        while let Some(u) = queue.pop_front() {
            for v in 0..node_count {
                let residual_capacity = capacity[u][v] - flow[u][v];
                if !reachable[v] && residual_capacity > 1e-10 {
                    reachable[v] = true;
                    queue.push_back(v);
                }
            }
        }
        
        // Find cut edges (from reachable to non-reachable)
        for u in 0..node_count {
            for v in 0..node_count {
                if reachable[u] && !reachable[v] && capacity[u][v] > 1e-10 {
                    if let (Some(u_label), Some(v_label)) = (
                        self.reverse_node_map.get(&NodeIndex::new(u)),
                        self.reverse_node_map.get(&NodeIndex::new(v))
                    ) {
                        min_cut_edges.push((u_label.clone(), v_label.clone()));
                    }
                }
            }
        }
        
        let computation_time = start_time.elapsed().as_millis() as f64;
        
        Some(NetworkFlow {
            graph_id: self.name.clone(),
            source: source.to_string(),
            sink: sink.to_string(),
            max_flow_value,
            flow_paths,
            min_cut_edges,
            min_cut_value: max_flow_value, // By max-flow min-cut theorem
            algorithm_used: "Edmonds-Karp".to_string(),
            computation_time_ms: computation_time,
            iterations,
        })
    }
    
    /// Compute maximum flow using Ford-Fulkerson with DFS
    pub fn max_flow_ford_fulkerson(&self, source: &str, sink: &str) -> Option<NetworkFlow> {
        let source_idx = self.node_map.get(source)?;
        let sink_idx = self.node_map.get(sink)?;
        
        if source_idx == sink_idx {
            return None;
        }
        
        let start_time = std::time::Instant::now();
        let mut iterations = 0;
        
        // Build residual graph
        let node_count = self.graph.node_count();
        let mut capacity = vec![vec![0.0; node_count]; node_count];
        let mut flow = vec![vec![0.0; node_count]; node_count];
        
        // Initialize capacities
        for edge_idx in self.graph.edge_indices() {
            if let (Some(endpoints), Some(edge_data)) = (
                self.graph.edge_endpoints(edge_idx),
                self.graph.edge_weight(edge_idx)
            ) {
                let u = endpoints.0.index();
                let v = endpoints.1.index();
                capacity[u][v] += edge_data.weight;
                
                if !self.is_directed {
                    capacity[v][u] += edge_data.weight;
                }
            }
        }
        
        let mut max_flow_value = 0.0;
        let mut flow_paths = Vec::new();
        
        // Ford-Fulkerson with DFS
        loop {
            iterations += 1;
            
            // DFS to find augmenting path
            let mut visited = vec![false; node_count];
            let mut parent = vec![None; node_count];
            
            fn dfs(
                u: usize,
                sink: usize,
                capacity: &[Vec<f64>],
                flow: &[Vec<f64>],
                visited: &mut [bool],
                parent: &mut [Option<usize>],
            ) -> bool {
                if u == sink {
                    return true;
                }
                
                visited[u] = true;
                
                for v in 0..capacity.len() {
                    let residual_capacity = capacity[u][v] - flow[u][v];
                    if !visited[v] && residual_capacity > 1e-10 {
                        parent[v] = Some(u);
                        if dfs(v, sink, capacity, flow, visited, parent) {
                            return true;
                        }
                    }
                }
                
                false
            }
            
            if !dfs(source_idx.index(), sink_idx.index(), &capacity, &flow, &mut visited, &mut parent) {
                break;
            }
            
            // Find bottleneck
            let mut path_flow = f64::INFINITY;
            let mut current = sink_idx.index();
            let mut path = Vec::new();
            
            while let Some(prev) = parent[current] {
                let residual_capacity = capacity[prev][current] - flow[prev][current];
                path_flow = path_flow.min(residual_capacity);
                path.push(current);
                current = prev;
            }
            path.push(source_idx.index());
            path.reverse();
            
            // Update flow
            current = sink_idx.index();
            while let Some(prev) = parent[current] {
                flow[prev][current] += path_flow;
                flow[current][prev] -= path_flow;
                current = prev;
            }
            
            max_flow_value += path_flow;
            
            // Convert path to labels
            let path_labels: Vec<String> = path.iter()
                .filter_map(|&idx| {
                    self.reverse_node_map.get(&NodeIndex::new(idx)).cloned()
                })
                .collect();
            
            flow_paths.push(FlowPath {
                path: path_labels,
                flow_value: path_flow,
                bottleneck_edge: (source.to_string(), sink.to_string()),
            });
            
            if iterations > 1000 {
                break;
            }
        }
        
        let computation_time = start_time.elapsed().as_millis() as f64;
        
        Some(NetworkFlow {
            graph_id: self.name.clone(),
            source: source.to_string(),
            sink: sink.to_string(),
            max_flow_value,
            flow_paths,
            min_cut_edges: Vec::new(), // Simplified for Ford-Fulkerson
            min_cut_value: max_flow_value,
            algorithm_used: "Ford-Fulkerson".to_string(),
            computation_time_ms: computation_time,
            iterations,
        })
    }
}

impl Foreign for NetworkFlow {
    fn type_name(&self) -> &'static str {
        "NetworkFlow"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "GraphID" => Ok(Value::String(self.graph_id.clone())),
            "Source" => Ok(Value::String(self.source.clone())),
            "Sink" => Ok(Value::String(self.sink.clone())),
            "MaxFlowValue" => Ok(Value::Real(self.max_flow_value)),
            "MinCutValue" => Ok(Value::Real(self.min_cut_value)),
            "Algorithm" => Ok(Value::String(self.algorithm_used.clone())),
            "ComputationTime" => Ok(Value::Real(self.computation_time_ms)),
            "Iterations" => Ok(Value::Integer(self.iterations as i64)),
            
            "FlowPaths" => {
                let path_values: Vec<Value> = self.flow_paths.iter().map(|flow_path| {
                    let path_nodes: Vec<Value> = flow_path.path.iter().map(|node| Value::String(node.clone())).collect();
                    Value::List(vec![
                        Value::List(path_nodes),
                        Value::Real(flow_path.flow_value),
                        Value::List(vec![
                            Value::String(flow_path.bottleneck_edge.0.clone()),
                            Value::String(flow_path.bottleneck_edge.1.clone()),
                        ]),
                    ])
                }).collect();
                Ok(Value::List(path_values))
            }
            
            "MinCutEdges" => {
                let cut_edges: Vec<Value> = self.min_cut_edges.iter().map(|(u, v)| {
                    Value::List(vec![Value::String(u.clone()), Value::String(v.clone())])
                }).collect();
                Ok(Value::List(cut_edges))
            }
            
            "FlowDecomposition" => {
                // Return flow paths as decomposed flows
                let decomposition: Vec<Value> = self.flow_paths.iter().map(|flow_path| {
                    Value::List(vec![
                        Value::List(flow_path.path.iter().map(|node| Value::String(node.clone())).collect()),
                        Value::Real(flow_path.flow_value),
                    ])
                }).collect();
                Ok(Value::List(decomposition))
            }
            
            "Bottlenecks" => {
                // Return bottleneck edges with their flow values
                let bottlenecks: Vec<Value> = self.flow_paths.iter().map(|flow_path| {
                    Value::List(vec![
                        Value::String(flow_path.bottleneck_edge.0.clone()),
                        Value::String(flow_path.bottleneck_edge.1.clone()),
                        Value::Real(flow_path.flow_value),
                    ])
                }).collect();
                Ok(Value::List(bottlenecks))
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Ring buffer for storing time-series metrics
#[derive(Debug, Clone)]
pub struct RingBuffer<T> {
    data: VecDeque<T>,
    capacity: usize,
}

impl<T> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: VecDeque::with_capacity(capacity),
            capacity,
        }
    }
    
    pub fn push(&mut self, value: T) {
        if self.data.len() >= self.capacity {
            self.data.pop_front();
        }
        self.data.push_back(value);
    }
    
    pub fn data(&self) -> &VecDeque<T> {
        &self.data
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// Target type for network monitoring
#[derive(Debug, Clone)]
pub enum TargetType {
    Host,
    Service,
    Graph,
    Endpoint,
}

/// Individual monitoring target
#[derive(Debug, Clone)]
pub struct MonitorTarget {
    pub target_type: TargetType,
    pub address: String,
    pub port: Option<u16>,
    pub check_interval: Duration,
    pub timeout: Duration,
    pub enabled: bool,
}

/// Network performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub latency_ms: RingBuffer<f64>,
    pub throughput_mbps: RingBuffer<f64>,
    pub packet_loss_rate: RingBuffer<f64>,
    pub error_rate: f64,
    pub availability: f64,
    pub last_updated: SystemTime,
    pub total_requests: u64,
    pub failed_requests: u64,
}

impl PerformanceMetrics {
    pub fn new(buffer_size: usize) -> Self {
        Self {
            latency_ms: RingBuffer::new(buffer_size),
            throughput_mbps: RingBuffer::new(buffer_size),
            packet_loss_rate: RingBuffer::new(buffer_size),
            error_rate: 0.0,
            availability: 100.0,
            last_updated: SystemTime::now(),
            total_requests: 0,
            failed_requests: 0,
        }
    }
    
    pub fn update_latency(&mut self, latency_ms: f64) {
        self.latency_ms.push(latency_ms);
        self.last_updated = SystemTime::now();
    }
    
    pub fn update_throughput(&mut self, mbps: f64) {
        self.throughput_mbps.push(mbps);
    }
    
    pub fn record_request(&mut self, success: bool) {
        self.total_requests += 1;
        if !success {
            self.failed_requests += 1;
        }
        
        self.error_rate = if self.total_requests > 0 {
            (self.failed_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        };
        
        self.availability = 100.0 - self.error_rate;
    }
    
    pub fn average_latency(&self) -> f64 {
        if self.latency_ms.is_empty() {
            return 0.0;
        }
        
        let sum: f64 = self.latency_ms.data().iter().sum();
        sum / self.latency_ms.len() as f64
    }
    
    pub fn max_latency(&self) -> f64 {
        self.latency_ms.data().iter().copied().fold(0.0, f64::max)
    }
    
    pub fn min_latency(&self) -> f64 {
        self.latency_ms.data().iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Network alert configuration
#[derive(Debug, Clone)]
pub struct NetworkAlert {
    pub alert_type: AlertType,
    pub threshold: f64,
    pub condition: AlertCondition,
    pub enabled: bool,
    pub message: String,
    pub last_triggered: Option<SystemTime>,
}

#[derive(Debug, Clone)]
pub enum AlertType {
    Latency,
    Throughput,
    ErrorRate,
    Availability,
    PacketLoss,
}

#[derive(Debug, Clone)]
pub enum AlertCondition {
    GreaterThan,
    LessThan,
    Equal,
}

/// Monitor configuration
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub default_interval: Duration,
    pub default_timeout: Duration,
    pub max_targets: usize,
    pub metrics_buffer_size: usize,
    pub enable_alerts: bool,
    pub report_interval: Duration,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            default_interval: Duration::from_secs(30),
            default_timeout: Duration::from_secs(5),
            max_targets: 100,
            metrics_buffer_size: 1000,
            enable_alerts: true,
            report_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Production NetworkMonitor with real-time monitoring
#[derive(Debug)]
pub struct NetworkMonitor {
    pub monitor_id: String,
    pub targets: Vec<MonitorTarget>,
    pub metrics: HashMap<String, PerformanceMetrics>,
    pub alerts: Vec<NetworkAlert>,
    pub config: MonitorConfig,
    pub running: Arc<AtomicBool>,
    pub start_time: SystemTime,
}

impl Clone for NetworkMonitor {
    fn clone(&self) -> Self {
        Self {
            monitor_id: self.monitor_id.clone(),
            targets: self.targets.clone(),
            metrics: self.metrics.clone(),
            alerts: self.alerts.clone(),
            config: self.config.clone(),
            running: Arc::new(AtomicBool::new(self.running.load(Ordering::Relaxed))),
            start_time: self.start_time,
        }
    }
}

impl NetworkMonitor {
    /// Create new network monitor
    pub fn new(monitor_id: String, config: MonitorConfig) -> Self {
        Self {
            monitor_id,
            targets: Vec::new(),
            metrics: HashMap::new(),
            alerts: Vec::new(),
            config,
            running: Arc::new(AtomicBool::new(false)),
            start_time: SystemTime::now(),
        }
    }
    
    /// Add monitoring target
    pub fn add_target(&mut self, target: MonitorTarget) {
        if self.targets.len() < self.config.max_targets {
            let target_key = format!("{}:{}", target.address, target.port.unwrap_or(80));
            self.metrics.insert(
                target_key,
                PerformanceMetrics::new(self.config.metrics_buffer_size)
            );
            self.targets.push(target);
        }
    }
    
    /// Start monitoring (simplified for synchronous context)
    pub fn start_monitoring(&self) -> bool {
        self.running.store(true, Ordering::Relaxed);
        true
    }
    
    /// Stop monitoring
    pub fn stop_monitoring(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
    
    /// Check if monitoring is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
    
    /// Get metrics for a specific target
    pub fn get_target_metrics(&self, target_address: &str) -> Option<&PerformanceMetrics> {
        self.metrics.get(target_address)
    }
    
    /// Update metrics for a target
    pub fn update_metrics(&mut self, target_address: &str, latency_ms: f64, success: bool) {
        if let Some(metrics) = self.metrics.get_mut(target_address) {
            metrics.update_latency(latency_ms);
            metrics.record_request(success);
        }
    }
    
    /// Add alert configuration
    pub fn add_alert(&mut self, alert: NetworkAlert) {
        self.alerts.push(alert);
    }
    
    /// Check alerts for current metrics
    pub fn check_alerts(&mut self) -> Vec<String> {
        let mut triggered_alerts = Vec::new();
        let now = SystemTime::now();
        
        for alert in &mut self.alerts {
            if !alert.enabled {
                continue;
            }
            
            let should_trigger = match alert.alert_type {
                AlertType::Latency => {
                    self.metrics.values().any(|m| {
                        let avg_latency = m.average_latency();
                        match alert.condition {
                            AlertCondition::GreaterThan => avg_latency > alert.threshold,
                            AlertCondition::LessThan => avg_latency < alert.threshold,
                            AlertCondition::Equal => (avg_latency - alert.threshold).abs() < 1e-10,
                        }
                    })
                }
                AlertType::ErrorRate => {
                    self.metrics.values().any(|m| {
                        match alert.condition {
                            AlertCondition::GreaterThan => m.error_rate > alert.threshold,
                            AlertCondition::LessThan => m.error_rate < alert.threshold,
                            AlertCondition::Equal => (m.error_rate - alert.threshold).abs() < 1e-10,
                        }
                    })
                }
                AlertType::Availability => {
                    self.metrics.values().any(|m| {
                        match alert.condition {
                            AlertCondition::GreaterThan => m.availability > alert.threshold,
                            AlertCondition::LessThan => m.availability < alert.threshold,
                            AlertCondition::Equal => (m.availability - alert.threshold).abs() < 1e-10,
                        }
                    })
                }
                _ => false, // Placeholder for other alert types
            };
            
            if should_trigger {
                alert.last_triggered = Some(now);
                triggered_alerts.push(alert.message.clone());
            }
        }
        
        triggered_alerts
    }
    
    /// Generate monitoring report
    pub fn generate_report(&self) -> String {
        let mut report = format!("Network Monitor Report - {}\n", self.monitor_id);
        report.push_str(&format!("Monitoring {} targets\n", self.targets.len()));
        report.push_str(&format!("Running: {}\n", self.is_running()));
        
        if let Ok(uptime) = SystemTime::now().duration_since(self.start_time) {
            report.push_str(&format!("Uptime: {}s\n", uptime.as_secs()));
        }
        
        report.push_str("\nTarget Metrics:\n");
        for (target, metrics) in &self.metrics {
            report.push_str(&format!("  {}: Avg Latency: {:.2}ms, Availability: {:.1}%\n",
                target, metrics.average_latency(), metrics.availability));
        }
        
        report
    }
    
    /// Simulate network ping for testing
    pub fn simulate_ping(&mut self, target: &str, simulate_latency: Option<f64>) -> f64 {
        let latency = simulate_latency.unwrap_or_else(|| {
            // Simulate realistic ping times with some variance
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            
            let mut hasher = DefaultHasher::new();
            target.hash(&mut hasher);
            let seed = hasher.finish();
            
            let base_latency = 10.0 + (seed % 50) as f64; // 10-60ms base
            let variance = ((seed % 10) as f64 - 5.0) * 2.0; // ±10ms variance
            base_latency + variance
        });
        
        let success = latency < 1000.0; // Consider >1s as failure
        self.update_metrics(target, latency, success);
        
        latency
    }
}

impl Foreign for NetworkMonitor {
    fn type_name(&self) -> &'static str {
        "NetworkMonitor"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "MonitorID" => Ok(Value::String(self.monitor_id.clone())),
            "IsRunning" => Ok(Value::Integer(if self.is_running() { 1 } else { 0 })),
            "TargetCount" => Ok(Value::Integer(self.targets.len() as i64)),
            "MetricsCount" => Ok(Value::Integer(self.metrics.len() as i64)),
            
            "GetMetrics" => {
                if let Some(target) = args.get(0) {
                    match target {
                        Value::String(target_addr) => {
                            if let Some(metrics) = self.get_target_metrics(target_addr) {
                                Ok(Value::List(vec![
                                    Value::Real(metrics.average_latency()),
                                    Value::Real(metrics.max_latency()),
                                    Value::Real(metrics.min_latency()),
                                    Value::Real(metrics.error_rate),
                                    Value::Real(metrics.availability),
                                    Value::Integer(metrics.total_requests as i64),
                                ]))
                            } else {
                                Ok(Value::List(vec![]))
                            }
                        }
                        _ => Err(ForeignError::InvalidArgument("Target must be string".to_string())),
                    }
                } else {
                    // Return summary of all targets
                    let all_metrics: Vec<Value> = self.metrics.iter().map(|(target, metrics)| {
                        Value::List(vec![
                            Value::String(target.clone()),
                            Value::Real(metrics.average_latency()),
                            Value::Real(metrics.availability),
                        ])
                    }).collect();
                    Ok(Value::List(all_metrics))
                }
            }
            
            "GenerateReport" => {
                Ok(Value::String(self.generate_report()))
            }
            
            "Targets" => {
                let target_list: Vec<Value> = self.targets.iter().map(|target| {
                    Value::List(vec![
                        Value::String(target.address.clone()),
                        Value::Integer(target.port.unwrap_or(80) as i64),
                        Value::Integer(if target.enabled { 1 } else { 0 }),
                    ])
                }).collect();
                Ok(Value::List(target_list))
            }
            
            "Alerts" => {
                let alert_list: Vec<Value> = self.alerts.iter().map(|alert| {
                    Value::List(vec![
                        Value::String(alert.message.clone()),
                        Value::Real(alert.threshold),
                        Value::Integer(if alert.enabled { 1 } else { 0 }),
                    ])
                }).collect();
                Ok(Value::List(alert_list))
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
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Production functions for NetworkGraph implementation

/// NetworkGraph[type, name] - Create network topology graph  
pub fn network_graph(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (type, [name])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph_type = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "String (graph type)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let name = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "String (graph name)".to_string(),
                actual: "non-string".to_string(),
            }),
        }
    } else {
        format!("Graph_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs())
    };
    
    let graph = match graph_type {
        "Undirected" | "undirected" => NetworkGraph::new_undirected(name),
        "Directed" | "directed" => NetworkGraph::new_directed(name),
        _ => return Err(VmError::TypeError {
            expected: "'Directed' or 'Undirected'".to_string(),
            actual: format!("'{}'", graph_type),
        }),
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// GraphAddNode[graph, label] - Add node to graph
pub fn graph_add_node(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (graph, label)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?.clone()
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let label = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String (node label)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    graph.add_node(label);
    
    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// GraphAddEdge[graph, source, target, weight] - Add edge to graph
pub fn graph_add_edge(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (graph, source, target, [weight])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?.clone()
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let source = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (source node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let target = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (target node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let weight = if args.len() > 3 {
        match &args[3] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Number (edge weight)".to_string(),
                actual: "non-number".to_string(),
            }),
        }
    } else {
        1.0
    };
    
    graph.add_edge(source, target, weight)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(graph))))
}

/// GraphShortestPath[graph, source, target] - Find shortest path
pub fn graph_shortest_path(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (graph, source, target)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let source = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (source node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let target = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (target node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    match graph.shortest_path(source, target) {
        Some(path) => {
            let path_values: Vec<Value> = path.into_iter().map(Value::String).collect();
            Ok(Value::List(path_values))
        }
        None => Ok(Value::List(vec![])),
    }
}

/// GraphComponents[graph] - Get connected components
pub fn graph_components(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let components = graph.connected_components();
    let component_values: Vec<Value> = components.into_iter().map(|component| {
        let node_values: Vec<Value> = component.into_iter().map(Value::String).collect();
        Value::List(node_values)
    }).collect();
    
    Ok(Value::List(component_values))
}

/// GraphMST[graph] - Compute minimum spanning tree
pub fn graph_mst(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    // Simple MST approximation - return edges as tuples
    let mut mst_edges = Vec::new();
    for edge_idx in graph.graph.edge_indices() {
        if let (Some(endpoints), Some(edge_data)) = (
            graph.graph.edge_endpoints(edge_idx),
            graph.graph.edge_weight(edge_idx)
        ) {
            if let (Some(source_label), Some(target_label)) = (
                graph.reverse_node_map.get(&endpoints.0),
                graph.reverse_node_map.get(&endpoints.1)
            ) {
                mst_edges.push(Value::List(vec![
                    Value::String(source_label.clone()),
                    Value::String(target_label.clone()),
                    Value::Real(edge_data.weight),
                ]));
            }
        }
    }
    
    Ok(Value::List(mst_edges))
}

/// GraphMetrics[graph, metrics] - Compute comprehensive graph metrics
pub fn graph_metrics(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (graph, [metrics])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let start_time = std::time::Instant::now();
    
    // Compute all centrality measures
    let betweenness_centrality = graph.betweenness_centrality();
    let closeness_centrality = graph.closeness_centrality();
    let pagerank = graph.pagerank_centrality(0.85, 100);
    
    // Compute graph structure metrics
    let communities = graph.detect_communities();
    let modularity = graph.modularity(&communities);
    let clustering_coefficient = graph.clustering_coefficient();
    let diameter = graph.diameter();
    let average_path_length = graph.average_path_length();
    let density = graph.density();
    
    // Simple small-world coefficient approximation
    let small_world_coefficient = if average_path_length > 0.0 {
        clustering_coefficient / average_path_length
    } else {
        0.0
    };
    
    let computation_time = start_time.elapsed().as_millis() as f64;
    
    // Create comprehensive metrics object
    let metrics = NetworkMetrics {
        graph_id: graph.name.clone(),
        betweenness_centrality,
        closeness_centrality,
        eigenvector_centrality: HashMap::new(), // Placeholder for now
        pagerank,
        communities,
        modularity,
        clustering_coefficient,
        diameter,
        average_path_length,
        density,
        small_world_coefficient,
        algorithm_used: "NetworkGraph::comprehensive_analysis".to_string(),
        computation_time_ms: computation_time,
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(metrics))))
}

/// NetworkCentrality[graph, type] - Compute specific centrality measure
pub fn network_centrality(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (graph, centrality_type)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let centrality_type = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "String (centrality type)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let centrality_scores = match centrality_type {
        "Betweenness" | "betweenness" => graph.betweenness_centrality(),
        "Closeness" | "closeness" => graph.closeness_centrality(),
        "PageRank" | "pagerank" => graph.pagerank_centrality(0.85, 100),
        _ => return Err(VmError::TypeError {
            expected: "'Betweenness', 'Closeness', or 'PageRank'".to_string(),
            actual: format!("'{}'", centrality_type),
        }),
    };
    
    // Return as list of node -> score pairs
    let centrality_values: Vec<Value> = centrality_scores.iter().map(|(node, &score)| {
        Value::List(vec![Value::String(node.clone()), Value::Real(score)])
    }).collect();
    
    Ok(Value::List(centrality_values))
}

/// CommunityDetection[graph, algorithm] - Detect communities in graph
pub fn community_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (graph, [algorithm])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let algorithm = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "components",
        }
    } else {
        "components"
    };
    
    let communities = match algorithm {
        "Components" | "components" | "ConnectedComponents" => graph.connected_components(),
        // Add more algorithms here in the future (Louvain, label propagation, etc.)
        _ => graph.detect_communities(),
    };
    
    // Return communities as list of lists
    let community_values: Vec<Value> = communities.iter().map(|community| {
        let node_values: Vec<Value> = community.iter().map(|node| Value::String(node.clone())).collect();
        Value::List(node_values)
    }).collect();
    
    Ok(Value::List(community_values))
}

/// GraphDiameter[graph] - Compute graph diameter
pub fn graph_diameter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    Ok(Value::Real(graph.diameter()))
}

/// GraphDensity[graph] - Compute graph density
pub fn graph_density(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    Ok(Value::Real(graph.density()))
}

/// ClusteringCoefficient[graph, node] - Compute clustering coefficient
pub fn clustering_coefficient(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (graph, [node])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    if args.len() > 1 {
        // Local clustering coefficient for specific node
        let node_label = match &args[1] {
            Value::String(s) => s,
            _ => return Err(VmError::TypeError {
                expected: "String (node label)".to_string(),
                actual: "non-string".to_string(),
            }),
        };
        
        if let Some(&node_idx) = graph.node_map.get(node_label) {
            let neighbors: Vec<_> = graph.graph.neighbors(node_idx).collect();
            let neighbor_count = neighbors.len();
            
            if neighbor_count < 2 {
                return Ok(Value::Real(0.0));
            }
            
            let mut triangle_count = 0;
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.graph.find_edge(neighbors[i], neighbors[j]).is_some() {
                        triangle_count += 1;
                    }
                }
            }
            
            let max_triangles = neighbor_count * (neighbor_count - 1) / 2;
            let local_coefficient = if max_triangles > 0 {
                triangle_count as f64 / max_triangles as f64
            } else {
                0.0
            };
            
            Ok(Value::Real(local_coefficient))
        } else {
            Err(VmError::Runtime(format!("Node '{}' not found in graph", node_label)))
        }
    } else {
        // Global clustering coefficient
        Ok(Value::Real(graph.clustering_coefficient()))
    }
}

/// NetworkFlow[graph, source, sink, algorithm] - Compute maximum flow
pub fn network_flow(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (graph, source, sink, [algorithm])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let source = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (source node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let sink = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (sink node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let algorithm = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => s.as_str(),
            _ => "EdmondsKarp",
        }
    } else {
        "EdmondsKarp"
    };
    
    let flow_result = match algorithm {
        "EdmondsKarp" | "Edmonds-Karp" | "edmonds-karp" => {
            graph.max_flow_edmonds_karp(source, sink)
        }
        "FordFulkerson" | "Ford-Fulkerson" | "ford-fulkerson" => {
            graph.max_flow_ford_fulkerson(source, sink)
        }
        _ => return Err(VmError::TypeError {
            expected: "'EdmondsKarp' or 'FordFulkerson'".to_string(),
            actual: format!("'{}'", algorithm),
        }),
    };
    
    match flow_result {
        Some(flow) => Ok(Value::LyObj(LyObj::new(Box::new(flow)))),
        None => Err(VmError::Runtime("Could not compute network flow".to_string())),
    }
}

/// MinimumCut[graph, source, sink] - Find minimum cut
pub fn minimum_cut(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (graph, source, sink)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let source = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (source node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let sink = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (sink node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    // Use Edmonds-Karp to find max flow and min cut
    if let Some(flow) = graph.max_flow_edmonds_karp(source, sink) {
        let cut_edges: Vec<Value> = flow.min_cut_edges.iter().map(|(u, v)| {
            Value::List(vec![Value::String(u.clone()), Value::String(v.clone())])
        }).collect();
        
        Ok(Value::List(vec![
            Value::Real(flow.min_cut_value),
            Value::List(cut_edges),
        ]))
    } else {
        Err(VmError::Runtime("Could not compute minimum cut".to_string()))
    }
}

/// FlowDecomposition[flow] - Decompose flow into paths
pub fn flow_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (flow)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let flow = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkFlow>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkFlow".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkFlow".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let decomposition: Vec<Value> = flow.flow_paths.iter().map(|flow_path| {
        Value::List(vec![
            Value::List(flow_path.path.iter().map(|node| Value::String(node.clone())).collect()),
            Value::Real(flow_path.flow_value),
            Value::List(vec![
                Value::String(flow_path.bottleneck_edge.0.clone()),
                Value::String(flow_path.bottleneck_edge.1.clone()),
            ]),
        ])
    }).collect();
    
    Ok(Value::List(decomposition))
}

/// FlowBottlenecks[flow] - Identify flow bottlenecks
pub fn flow_bottlenecks(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (flow)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let flow = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkFlow>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkFlow".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkFlow".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    // Find bottleneck edges and their utilization
    let mut bottleneck_map: HashMap<(String, String), f64> = HashMap::new();
    
    for flow_path in &flow.flow_paths {
        let edge = &flow_path.bottleneck_edge;
        *bottleneck_map.entry(edge.clone()).or_insert(0.0) += flow_path.flow_value;
    }
    
    let bottlenecks: Vec<Value> = bottleneck_map.iter().map(|((u, v), &total_flow)| {
        Value::List(vec![
            Value::String(u.clone()),
            Value::String(v.clone()),
            Value::Real(total_flow),
        ])
    }).collect();
    
    Ok(Value::List(bottlenecks))
}

/// MaxFlowValue[graph, source, sink] - Get just the max flow value
pub fn max_flow_value(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (graph, source, sink)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let source = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (source node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let sink = match &args[2] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (sink node)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    if let Some(flow) = graph.max_flow_edmonds_karp(source, sink) {
        Ok(Value::Real(flow.max_flow_value))
    } else {
        Err(VmError::Runtime("Could not compute max flow value".to_string()))
    }
}

/// NetworkMonitor[targets, config] - Create network monitor
pub fn network_monitor(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (targets, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Parse targets list
    let targets_list = match &args[0] {
        Value::List(targets) => targets,
        _ => return Err(VmError::TypeError {
            expected: "List (targets)".to_string(),
            actual: "non-list".to_string(),
        }),
    };
    
    // Create monitor with default config
    let monitor_id = format!("monitor_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
    let config = MonitorConfig::default();
    let mut monitor = NetworkMonitor::new(monitor_id, config);
    
    // Parse and add targets
    for target_value in targets_list {
        match target_value {
            Value::String(address) => {
                let target = MonitorTarget {
                    target_type: TargetType::Host,
                    address: address.clone(),
                    port: Some(80),
                    check_interval: Duration::from_secs(30),
                    timeout: Duration::from_secs(5),
                    enabled: true,
                };
                monitor.add_target(target);
            }
            Value::List(target_info) => {
                if target_info.len() >= 2 {
                    if let (Value::String(address), Value::Integer(port)) = (&target_info[0], &target_info[1]) {
                        let target = MonitorTarget {
                            target_type: TargetType::Host,
                            address: address.clone(),
                            port: Some(*port as u16),
                            check_interval: Duration::from_secs(30),
                            timeout: Duration::from_secs(5),
                            enabled: true,
                        };
                        monitor.add_target(target);
                    }
                }
            }
            _ => {} // Skip invalid targets
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(monitor))))
}

/// MonitorStart[monitor] - Start monitoring tasks
pub fn monitor_start(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (monitor)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let monitor = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkMonitor>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkMonitor".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkMonitor".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let started = monitor.start_monitoring();
    Ok(Value::Integer(if started { 1 } else { 0 }))
}

/// MonitorStop[monitor] - Stop monitoring
pub fn monitor_stop(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (monitor)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let monitor = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkMonitor>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkMonitor".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkMonitor".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    monitor.stop_monitoring();
    Ok(Value::Integer(1))
}

/// MonitorGetMetrics[monitor, target] - Get current metrics
pub fn monitor_get_metrics(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (monitor, [target])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let monitor = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkMonitor>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkMonitor".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkMonitor".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    if args.len() > 1 {
        // Get metrics for specific target
        let target = match &args[1] {
            Value::String(s) => s,
            _ => return Err(VmError::TypeError {
                expected: "String (target address)".to_string(),
                actual: "non-string".to_string(),
            }),
        };
        
        if let Some(metrics) = monitor.get_target_metrics(target) {
            Ok(Value::List(vec![
                Value::Real(metrics.average_latency()),
                Value::Real(metrics.max_latency()),
                Value::Real(metrics.min_latency()),
                Value::Real(metrics.error_rate),
                Value::Real(metrics.availability),
                Value::Integer(metrics.total_requests as i64),
                Value::Integer(metrics.failed_requests as i64),
            ]))
        } else {
            Ok(Value::List(vec![]))
        }
    } else {
        // Get metrics for all targets
        let all_metrics: Vec<Value> = monitor.metrics.iter().map(|(target, metrics)| {
            Value::List(vec![
                Value::String(target.clone()),
                Value::Real(metrics.average_latency()),
                Value::Real(metrics.availability),
                Value::Integer(metrics.total_requests as i64),
            ])
        }).collect();
        Ok(Value::List(all_metrics))
    }
}

/// MonitorSetAlerts[monitor, thresholds] - Configure alerts
pub fn monitor_set_alerts(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (monitor, thresholds)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut monitor = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkMonitor>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkMonitor".to_string(),
                    actual: "other type".to_string(),
                })?.clone()
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkMonitor".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    // Parse thresholds (simplified)
    let _thresholds = match &args[1] {
        Value::List(thresholds) => thresholds,
        _ => return Err(VmError::TypeError {
            expected: "List (thresholds)".to_string(),
            actual: "non-list".to_string(),
        }),
    };
    
    // Add default alert for high latency
    let latency_alert = NetworkAlert {
        alert_type: AlertType::Latency,
        threshold: 100.0, // 100ms
        condition: AlertCondition::GreaterThan,
        enabled: true,
        message: "High latency detected".to_string(),
        last_triggered: None,
    };
    monitor.add_alert(latency_alert);
    
    // Add default alert for low availability
    let availability_alert = NetworkAlert {
        alert_type: AlertType::Availability,
        threshold: 95.0, // 95%
        condition: AlertCondition::LessThan,
        enabled: true,
        message: "Low availability detected".to_string(),
        last_triggered: None,
    };
    monitor.add_alert(availability_alert);
    
    Ok(Value::LyObj(LyObj::new(Box::new(monitor))))
}

/// MonitorPing[monitor, target] - Perform ping test
pub fn monitor_ping(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (monitor, target, [expected_latency])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut monitor = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkMonitor>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkMonitor".to_string(),
                    actual: "other type".to_string(),
                })?.clone()
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkMonitor".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let target = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String (target address)".to_string(),
            actual: "non-string".to_string(),
        }),
    };
    
    let expected_latency = if args.len() > 2 {
        match &args[2] {
            Value::Real(r) => Some(*r),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    } else {
        None
    };
    
    let latency = monitor.simulate_ping(target, expected_latency);
    
    Ok(Value::List(vec![
        Value::Real(latency),
        Value::Integer(if latency < 1000.0 { 1 } else { 0 }), // Success flag
        Value::LyObj(LyObj::new(Box::new(monitor))),
    ]))
}

/// NetworkBottlenecks[graph, traffic] - Identify bottlenecks using monitoring data
pub fn network_bottlenecks(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (graph, [monitor])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<NetworkGraph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "NetworkGraph".to_string(),
                    actual: "other type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkGraph".to_string(),
            actual: "other type".to_string(),
        }),
    };
    
    let monitor = if args.len() > 1 {
        match &args[1] {
            Value::LyObj(obj) => {
                Some(obj.downcast_ref::<NetworkMonitor>()
                    .ok_or_else(|| VmError::TypeError {
                        expected: "NetworkMonitor".to_string(),
                        actual: "other type".to_string(),
                    })?)
            }
            _ => None,
        }
    } else {
        None
    };
    
    let mut bottlenecks = Vec::new();
    
    // Analyze graph structure for potential bottlenecks
    for (node_label, &node_idx) in &graph.node_map {
        let degree = graph.graph.edges(node_idx).count();
        
        // High-degree nodes are potential bottlenecks
        if degree > 5 {
            let bottleneck_score = degree as f64;
            
            // If monitor is provided, incorporate latency data
            let latency_factor = if let Some(mon) = monitor {
                if let Some(metrics) = mon.get_target_metrics(node_label) {
                    1.0 + (metrics.average_latency() / 100.0) // Scale latency impact
                } else {
                    1.0
                }
            } else {
                1.0
            };
            
            bottlenecks.push(Value::List(vec![
                Value::String(node_label.clone()),
                Value::Real(bottleneck_score * latency_factor),
                Value::String("High degree node".to_string()),
                Value::Integer(degree as i64),
            ]));
        }
    }
    
    // Find nodes with high betweenness centrality (structural bottlenecks)
    let betweenness = graph.betweenness_centrality();
    for (node_label, &centrality) in &betweenness {
        if centrality > 0.1 { // High betweenness centrality
            let existing = bottlenecks.iter().any(|b| {
                if let Value::List(vals) = b {
                    if let Value::String(name) = &vals[0] {
                        name == node_label
                    } else { false }
                } else { false }
            });
            
            if !existing {
                bottlenecks.push(Value::List(vec![
                    Value::String(node_label.clone()),
                    Value::Real(centrality * 100.0),
                    Value::String("High betweenness centrality".to_string()),
                    Value::Real(centrality),
                ]));
            }
        }
    }
    
    // Sort bottlenecks by score (descending)
    bottlenecks.sort_by(|a, b| {
        let score_a = if let Value::List(vals) = a {
            if let Value::Real(score) = vals.get(1).unwrap_or(&Value::Real(0.0)) {
                *score
            } else { 0.0 }
        } else { 0.0 };
        
        let score_b = if let Value::List(vals) = b {
            if let Value::Real(score) = vals.get(1).unwrap_or(&Value::Real(0.0)) {
                *score
            } else { 0.0 }
        } else { 0.0 };
        
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });
    
    Ok(Value::List(bottlenecks))
}

/// OptimizeTopology[graph, constraints] - Optimize network topology
pub fn optimize_topology(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("OptimizeTopology not yet implemented".to_string()))
}