//! Advanced Graph Algorithms for Lyra Standard Library
//!
//! This module implements sophisticated graph algorithms for network analysis,
//! social networks, web ranking, and community detection.

use crate::vm::{Value, VmError, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::graph::Graph;
use std::collections::{HashMap, HashSet, BinaryHeap, VecDeque};
use std::cmp::Ordering;
use std::any::Any;

/// PageRank algorithm implementation for web ranking and authority detection
#[derive(Debug, Clone)]
pub struct PageRankResult {
    /// Map from vertex ID to PageRank score
    pub scores: HashMap<i64, f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final convergence error
    pub error: f64,
}

impl Foreign for PageRankResult {
    fn type_name(&self) -> &'static str {
        "PageRankResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getScore" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let vertex = match &args[0] {
                    Value::Integer(v) => *v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                Ok(self.scores.get(&vertex)
                    .copied()
                    .map(Value::Real)
                    .unwrap_or(Value::Missing))
            }
            "topScores" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let k = match &args[0] {
                    Value::Integer(v) => *v as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let mut sorted_scores: Vec<(i64, f64)> = self.scores.iter()
                    .map(|(&vertex, &score)| (vertex, score))
                    .collect();
                sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                
                let top_k = sorted_scores.into_iter()
                    .take(k)
                    .map(|(vertex, score)| Value::List(vec![
                        Value::Integer(vertex),
                        Value::Real(score),
                    ]))
                    .collect();

                Ok(Value::List(top_k))
            }
            "iterations" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.iterations as i64))
            }
            "error" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.error))
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

/// Community detection result using modularity-based algorithms
#[derive(Debug, Clone)]
pub struct CommunityResult {
    /// Map from vertex ID to community ID
    pub communities: HashMap<i64, usize>,
    /// Modularity score of the partition
    pub modularity: f64,
    /// Number of communities found
    pub num_communities: usize,
}

impl Foreign for CommunityResult {
    fn type_name(&self) -> &'static str {
        "CommunityResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getCommunity" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let vertex = match &args[0] {
                    Value::Integer(v) => *v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                Ok(self.communities.get(&vertex)
                    .copied()
                    .map(|c| Value::Integer(c as i64))
                    .unwrap_or(Value::Missing))
            }
            "getCommunityMembers" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let community_id = match &args[0] {
                    Value::Integer(v) => *v as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let members: Vec<Value> = self.communities.iter()
                    .filter_map(|(&vertex, &comm)| {
                        if comm == community_id {
                            Some(Value::Integer(vertex))
                        } else {
                            None
                        }
                    })
                    .collect();

                Ok(Value::List(members))
            }
            "modularity" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.modularity))
            }
            "numCommunities" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.num_communities as i64))
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

/// Centrality measures result
#[derive(Debug, Clone)]
pub struct CentralityResult {
    /// Map from vertex ID to centrality score
    pub scores: HashMap<i64, f64>,
    /// Type of centrality measure
    pub measure_type: String,
}

impl Foreign for CentralityResult {
    fn type_name(&self) -> &'static str {
        "CentralityResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getScore" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let vertex = match &args[0] {
                    Value::Integer(v) => *v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                Ok(self.scores.get(&vertex)
                    .copied()
                    .map(Value::Real)
                    .unwrap_or(Value::Missing))
            }
            "topScores" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let k = match &args[0] {
                    Value::Integer(v) => *v as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let mut sorted_scores: Vec<(i64, f64)> = self.scores.iter()
                    .map(|(&vertex, &score)| (vertex, score))
                    .collect();
                sorted_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                
                let top_k = sorted_scores.into_iter()
                    .take(k)
                    .map(|(vertex, score)| Value::List(vec![
                        Value::Integer(vertex),
                        Value::Real(score),
                    ]))
                    .collect();

                Ok(Value::List(top_k))
            }
            "measureType" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.measure_type.clone()))
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

/// PageRank algorithm implementation
pub fn page_rank(graph: &Graph, damping: f64, tolerance: f64, max_iterations: usize) -> PageRankResult {
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let n = vertices.len();
    
    if n == 0 {
        return PageRankResult {
            scores: HashMap::new(),
            iterations: 0,
            error: 0.0,
        };
    }

    // Initialize PageRank scores
    let initial_score = 1.0 / n as f64;
    let mut scores: HashMap<i64, f64> = vertices.iter()
        .map(|&v| (v, initial_score))
        .collect();

    // Compute out-degrees for each vertex
    let out_degrees: HashMap<i64, usize> = vertices.iter()
        .map(|&v| {
            let degree = graph.adjacency.get(&v)
                .map(|neighbors| neighbors.len())
                .unwrap_or(0);
            (v, degree)
        })
        .collect();

    // Iterative PageRank computation
    let mut iteration = 0;
    let mut converged = false;

    while iteration < max_iterations && !converged {
        let mut new_scores = HashMap::new();
        let mut max_diff = 0.0;

        for &vertex in &vertices {
            let mut rank = (1.0 - damping) / n as f64;

            // Sum contributions from incoming edges
            for &other_vertex in &vertices {
                if let Some(neighbors) = graph.adjacency.get(&other_vertex) {
                    if neighbors.iter().any(|(neighbor, _)| *neighbor == vertex) {
                        let out_degree = out_degrees.get(&other_vertex).unwrap_or(&0);
                        if *out_degree > 0 {
                            let current_score = scores.get(&other_vertex).unwrap_or(&0.0);
                            rank += damping * current_score / *out_degree as f64;
                        }
                    }
                }
            }

            new_scores.insert(vertex, rank);
            
            // Check convergence
            let old_score = scores.get(&vertex).unwrap_or(&0.0);
            let diff = (rank - old_score).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        scores = new_scores;
        iteration += 1;
        converged = max_diff < tolerance;
    }

    PageRankResult {
        scores,
        iterations: iteration,
        error: if converged { 0.0 } else { tolerance },
    }
}

/// Louvain algorithm for community detection
pub fn louvain_communities(graph: &Graph) -> CommunityResult {
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let n = vertices.len();
    
    if n == 0 {
        return CommunityResult {
            communities: HashMap::new(),
            modularity: 0.0,
            num_communities: 0,
        };
    }

    // Initialize each vertex in its own community
    let mut communities: HashMap<i64, usize> = vertices.iter()
        .enumerate()
        .map(|(i, &v)| (v, i))
        .collect();

    // Calculate total edge weight
    let mut total_weight = 0.0;
    let mut vertex_weights: HashMap<i64, f64> = HashMap::new();
    
    for (&vertex, neighbors) in &graph.adjacency {
        let mut weight = 0.0;
        for &(_, w) in neighbors {
            weight += w;
            total_weight += w;
        }
        vertex_weights.insert(vertex, weight);
    }

    // If undirected graph, we've double-counted edges
    if !graph.directed {
        total_weight /= 2.0;
    }

    // Iterative improvement
    let mut improved = true;
    let mut iteration = 0;
    const MAX_ITERATIONS: usize = 100;

    while improved && iteration < MAX_ITERATIONS {
        improved = false;
        iteration += 1;

        for &vertex in &vertices {
            let current_community = communities[&vertex];
            let mut best_community = current_community;
            let mut best_gain = 0.0;

            // Get neighboring communities
            let mut neighbor_communities = HashSet::new();
            if let Some(neighbors) = graph.adjacency.get(&vertex) {
                for &(neighbor, _) in neighbors {
                    neighbor_communities.insert(communities[&neighbor]);
                }
            }

            // Try moving to each neighboring community
            for &neighbor_community in &neighbor_communities {
                if neighbor_community != current_community {
                    let gain = calculate_modularity_gain(
                        &graph,
                        &communities,
                        &vertex_weights,
                        vertex,
                        current_community,
                        neighbor_community,
                        total_weight,
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best_community = neighbor_community;
                    }
                }
            }

            // Move vertex if improvement found
            if best_community != current_community && best_gain > 1e-6 {
                communities.insert(vertex, best_community);
                improved = true;
            }
        }
    }

    // Relabel communities to be consecutive
    let mut community_map = HashMap::new();
    let mut next_id = 0;
    
    for (vertex, &community) in &communities {
        if !community_map.contains_key(&community) {
            community_map.insert(community, next_id);
            next_id += 1;
        }
        communities.insert(*vertex, community_map[&community]);
    }

    let modularity = calculate_modularity(&graph, &communities, &vertex_weights, total_weight);

    CommunityResult {
        communities,
        modularity,
        num_communities: next_id,
    }
}

/// Calculate modularity gain for moving a vertex between communities
fn calculate_modularity_gain(
    graph: &Graph,
    communities: &HashMap<i64, usize>,
    vertex_weights: &HashMap<i64, f64>,
    vertex: i64,
    from_community: usize,
    to_community: usize,
    total_weight: f64,
) -> f64 {
    // Simplified modularity gain calculation
    // In practice, this would involve computing the change in modularity
    // when moving vertex from one community to another
    
    let vertex_weight = vertex_weights.get(&vertex).unwrap_or(&0.0);
    let mut internal_from = 0.0;
    let mut internal_to = 0.0;

    if let Some(neighbors) = graph.adjacency.get(&vertex) {
        for &(neighbor, weight) in neighbors {
            let neighbor_community = communities[&neighbor];
            if neighbor_community == from_community {
                internal_from += weight;
            } else if neighbor_community == to_community {
                internal_to += weight;
            }
        }
    }

    // Simplified gain calculation
    let gain = (internal_to - internal_from) / (2.0 * total_weight);
    gain - (vertex_weight * vertex_weight) / (4.0 * total_weight * total_weight)
}

/// Calculate overall modularity of a community partition
fn calculate_modularity(
    graph: &Graph,
    communities: &HashMap<i64, usize>,
    vertex_weights: &HashMap<i64, f64>,
    total_weight: f64,
) -> f64 {
    let mut modularity = 0.0;

    // Calculate community internal weights and total weights
    let mut community_internal: HashMap<usize, f64> = HashMap::new();
    let mut community_total: HashMap<usize, f64> = HashMap::new();

    for (&vertex, neighbors) in &graph.adjacency {
        let vertex_community = communities[&vertex];
        let vertex_weight = vertex_weights.get(&vertex).unwrap_or(&0.0);
        
        *community_total.entry(vertex_community).or_insert(0.0) += vertex_weight;

        for &(neighbor, weight) in neighbors {
            if communities[&neighbor] == vertex_community {
                *community_internal.entry(vertex_community).or_insert(0.0) += weight;
            }
        }
    }

    // Calculate modularity
    for (community, &internal) in &community_internal {
        let total = community_total.get(community).unwrap_or(&0.0);
        let internal_fraction = if total_weight > 0.0 { internal / (2.0 * total_weight) } else { 0.0 };
        let total_fraction = if total_weight > 0.0 { total / (2.0 * total_weight) } else { 0.0 };
        
        modularity += internal_fraction - total_fraction * total_fraction;
    }

    modularity
}

/// Betweenness centrality calculation
pub fn betweenness_centrality(graph: &Graph) -> CentralityResult {
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let mut centrality: HashMap<i64, f64> = vertices.iter()
        .map(|&v| (v, 0.0))
        .collect();

    // For each vertex as source
    for &source in &vertices {
        // Compute shortest paths and dependencies
        let mut stack = Vec::new();
        let mut paths: HashMap<i64, i64> = HashMap::new();
        let mut dist: HashMap<i64, f64> = HashMap::new();
        let mut predecessors: HashMap<i64, Vec<i64>> = HashMap::new();

        // Initialize
        for &vertex in &vertices {
            paths.insert(vertex, 0);
            dist.insert(vertex, f64::INFINITY);
            predecessors.insert(vertex, Vec::new());
        }
        
        dist.insert(source, 0.0);
        paths.insert(source, 1);

        // BFS to find shortest paths
        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(vertex) = queue.pop_front() {
            stack.push(vertex);

            if let Some(neighbors) = graph.adjacency.get(&vertex) {
                for &(neighbor, weight) in neighbors {
                    let alt_dist = dist[&vertex] + weight;
                    
                    // Found a shorter path
                    if alt_dist < dist[&neighbor] {
                        queue.push_back(neighbor);
                        dist.insert(neighbor, alt_dist);
                        predecessors.insert(neighbor, vec![vertex]);
                        paths.insert(neighbor, paths[&vertex]);
                    }
                    // Found an equally short path
                    else if (alt_dist - dist[&neighbor]).abs() < 1e-10 {
                        predecessors.get_mut(&neighbor).unwrap().push(vertex);
                        let neighbor_paths = paths[&neighbor] + paths[&vertex];
                        paths.insert(neighbor, neighbor_paths);
                    }
                }
            }
        }

        // Accumulate betweenness from leaves
        let mut dependency: HashMap<i64, f64> = vertices.iter()
            .map(|&v| (v, 0.0))
            .collect();

        while let Some(vertex) = stack.pop() {
            for &predecessor in &predecessors[&vertex] {
                let contrib = (paths[&predecessor] as f64 / paths[&vertex] as f64) * 
                             (1.0 + dependency[&vertex]);
                *dependency.get_mut(&predecessor).unwrap() += contrib;
            }

            if vertex != source {
                *centrality.get_mut(&vertex).unwrap() += dependency[&vertex];
            }
        }
    }

    // Normalize for undirected graphs
    if !graph.directed {
        for (_, score) in centrality.iter_mut() {
            *score /= 2.0;
        }
    }

    CentralityResult {
        scores: centrality,
        measure_type: "betweenness".to_string(),
    }
}

/// Closeness centrality calculation
pub fn closeness_centrality(graph: &Graph) -> CentralityResult {
    let vertices: Vec<i64> = graph.vertices.iter().cloned().collect();
    let mut centrality: HashMap<i64, f64> = HashMap::new();

    for &source in &vertices {
        // Dijkstra's algorithm to find shortest paths
        let mut dist: HashMap<i64, f64> = vertices.iter()
            .map(|&v| (v, if v == source { 0.0 } else { f64::INFINITY }))
            .collect();
        
        let mut visited = HashSet::new();
        let mut heap = BinaryHeap::new();
        heap.push(DistanceState { vertex: source, distance: 0.0 });

        while let Some(DistanceState { vertex, distance }) = heap.pop() {
            if visited.contains(&vertex) {
                continue;
            }
            visited.insert(vertex);

            if let Some(neighbors) = graph.adjacency.get(&vertex) {
                for &(neighbor, weight) in neighbors {
                    let alt_distance = distance + weight;
                    if alt_distance < dist[&neighbor] {
                        dist.insert(neighbor, alt_distance);
                        heap.push(DistanceState { vertex: neighbor, distance: alt_distance });
                    }
                }
            }
        }

        // Calculate closeness centrality
        let total_distance: f64 = dist.values()
            .filter(|&&d| d != f64::INFINITY && d > 0.0)
            .sum();

        let reachable_count = dist.values()
            .filter(|&&d| d != f64::INFINITY)
            .count() as f64 - 1.0; // Exclude source vertex

        let closeness = if total_distance > 0.0 && reachable_count > 0.0 {
            reachable_count / total_distance
        } else {
            0.0
        };

        centrality.insert(source, closeness);
    }

    CentralityResult {
        scores: centrality,
        measure_type: "closeness".to_string(),
    }
}

#[derive(Debug, Clone)]
struct DistanceState {
    vertex: i64,
    distance: f64,
}

impl PartialEq for DistanceState {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for DistanceState {}

impl PartialOrd for DistanceState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for DistanceState {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

// Exported functions for the Lyra VM

/// PageRank[graph, damping, tolerance, maxIterations]
pub fn page_rank_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (graph, damping, tolerance, maxIterations)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Graph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Graph".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Graph".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let damping = match &args[1] {
        Value::Real(d) => *d,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let tolerance = match &args[2] {
        Value::Real(t) => *t,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let max_iterations = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };

    let result = page_rank(graph, damping, tolerance, max_iterations);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// CommunityDetection[graph]
pub fn community_detection_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Graph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Graph".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Graph".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let result = louvain_communities(graph);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// BetweennessCentrality[graph]
pub fn betweenness_centrality_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Graph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Graph".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Graph".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let result = betweenness_centrality(graph);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// ClosenessCentrality[graph]
pub fn closeness_centrality_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (graph)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let graph = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Graph>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Graph".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Graph".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let result = closeness_centrality(graph);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pagerank_simple() {
        let mut graph = Graph::new(true);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0);

        let result = page_rank(&graph, 0.85, 1e-6, 100);
        
        assert_eq!(result.scores.len(), 3);
        assert!(result.iterations > 0);
        
        // In a symmetric cycle, all nodes should have equal PageRank
        let scores: Vec<f64> = result.scores.values().cloned().collect();
        let expected_score = 1.0 / 3.0;
        for &score in &scores {
            assert!((score - expected_score).abs() < 0.01);
        }
    }

    #[test]
    fn test_community_detection_simple() {
        let mut graph = Graph::new(false);
        
        // Create two clusters
        for i in 1..=4 {
            graph.add_vertex(i);
        }
        
        // First cluster: 1-2
        graph.add_edge(1, 2, 1.0);
        
        // Second cluster: 3-4
        graph.add_edge(3, 4, 1.0);
        
        // Weak connection between clusters
        graph.add_edge(2, 3, 0.1);

        let result = louvain_communities(&graph);
        
        assert_eq!(result.communities.len(), 4);
        assert!(result.num_communities >= 2);
        assert!(result.modularity >= 0.0);
    }

    #[test]
    fn test_betweenness_centrality() {
        let mut graph = Graph::new(false);
        
        // Create a path: 1-2-3
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);

        let result = betweenness_centrality(&graph);
        
        assert_eq!(result.scores.len(), 3);
        
        // Vertex 2 should have highest betweenness (it's on the path between 1 and 3)
        let score_1 = result.scores[&1];
        let score_2 = result.scores[&2];
        let score_3 = result.scores[&3];
        
        assert!(score_2 > score_1);
        assert!(score_2 > score_3);
        assert_eq!(score_1, score_3); // Symmetric
    }

    #[test]
    fn test_closeness_centrality() {
        let mut graph = Graph::new(false);
        
        // Create a star graph: 1 connected to 2, 3, 4
        for i in 1..=4 {
            graph.add_vertex(i);
        }
        
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(1, 4, 1.0);

        let result = closeness_centrality(&graph);
        
        assert_eq!(result.scores.len(), 4);
        
        // Vertex 1 should have highest closeness (center of star)
        let score_1 = result.scores[&1];
        let score_2 = result.scores[&2];
        let score_3 = result.scores[&3];
        let score_4 = result.scores[&4];
        
        assert!(score_1 > score_2);
        assert!(score_1 > score_3);
        assert!(score_1 > score_4);
        
        // Other vertices should have equal closeness
        assert!((score_2 - score_3).abs() < 1e-10);
        assert!((score_3 - score_4).abs() < 1e-10);
    }
}