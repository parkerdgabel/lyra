//! Topological Data Analysis Functions
//!
//! Implementation of high-level TDA algorithms:
//! - Topological feature extraction
//! - Mapper algorithm for visualization

use super::{SimplicialComplex, TopologicalFeatures, PersistenceDiagram, value_to_points, compute_betti_numbers_full, vietoris_rips_complex, compute_persistent_homology, Filtration, Simplex};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;
use crate::stdlib::geometry::Point2D;
use std::collections::{HashMap, HashSet};

/// Extract comprehensive topological features from point cloud data
pub fn extract_topological_features(
    points: &[Point2D],
    max_radius: f64,
    max_dimension: usize,
    num_filtration_steps: usize,
) -> TopologicalFeatures {
    // Build filtration at multiple scales
    let mut filtration = Filtration::new();
    
    // Add vertices at filtration value 0
    for i in 0..points.len() {
        filtration.add_simplex(Simplex::new(vec![i]), 0.0);
    }
    
    // Build edges and higher-dimensional simplices at various scales
    let step_size = max_radius / num_filtration_steps as f64;
    
    for step in 1..=num_filtration_steps {
        let radius = step as f64 * step_size;
        let complex = vietoris_rips_complex(points, radius, max_dimension);
        
        // Add new simplices to filtration
        for dim in 1..=max_dimension {
            for simplex in complex.simplices_at_dimension(dim) {
                if !filtration.simplices.iter().any(|(s, _)| s == simplex) {
                    filtration.add_simplex(simplex.clone(), radius);
                }
            }
        }
    }
    
    filtration.sort_by_filtration();
    
    // Compute persistent homology
    let persistence_diagram = compute_persistent_homology(&filtration, max_dimension);
    
    // Build complex at maximum radius for Betti numbers
    let final_complex = vietoris_rips_complex(points, max_radius, max_dimension);
    let betti_numbers = compute_betti_numbers_full(&final_complex);
    let euler_characteristic = final_complex.euler_characteristic();
    
    TopologicalFeatures::new(betti_numbers, persistence_diagram, euler_characteristic)
}

/// Mapper algorithm for topological data visualization
pub fn mapper_algorithm(
    points: &[Point2D],
    filter_function: fn(&Point2D) -> f64,
    num_intervals: usize,
    overlap_percentage: f64,
    clustering_radius: f64,
) -> MapperGraph {
    let n = points.len();
    
    // Step 1: Apply filter function
    let mut filter_values: Vec<(usize, f64)> = points.iter()
        .enumerate()
        .map(|(i, p)| (i, filter_function(p)))
        .collect();
    
    filter_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    
    let min_val = filter_values[0].1;
    let max_val = filter_values[n - 1].1;
    let range = max_val - min_val;
    
    if range == 0.0 {
        return MapperGraph::new(); // Degenerate case
    }
    
    // Step 2: Create overlapping intervals
    let interval_size = range / num_intervals as f64;
    let overlap_size = interval_size * overlap_percentage;
    
    let mut intervals = Vec::new();
    for i in 0..num_intervals {
        let start = min_val + i as f64 * interval_size - overlap_size / 2.0;
        let end = min_val + (i + 1) as f64 * interval_size + overlap_size / 2.0;
        intervals.push((start.max(min_val), end.min(max_val)));
    }
    
    // Step 3: For each interval, find points and cluster them
    let mut mapper_graph = MapperGraph::new();
    let mut interval_clusters: Vec<Vec<usize>> = Vec::new();
    
    for (interval_start, interval_end) in intervals {
        // Find points in this interval
        let points_in_interval: Vec<usize> = filter_values.iter()
            .filter(|(_, val)| *val >= interval_start && *val <= interval_end)
            .map(|(idx, _)| *idx)
            .collect();
        
        if points_in_interval.is_empty() {
            interval_clusters.push(Vec::new());
            continue;
        }
        
        // Cluster points in this interval
        let clusters = cluster_points(&points_in_interval, points, clustering_radius);
        
        // Add clusters as nodes to mapper graph
        for cluster in &clusters {
            mapper_graph.add_node(cluster.clone());
        }
        
        interval_clusters.push(clusters.into_iter().flatten().collect());
    }
    
    // Step 4: Connect overlapping clusters
    for i in 0..(interval_clusters.len() - 1) {
        let current_interval = &interval_clusters[i];
        let next_interval = &interval_clusters[i + 1];
        
        // Find shared points between consecutive intervals
        for &p1 in current_interval {
            for &p2 in next_interval {
                if p1 == p2 {
                    // Find which nodes contain these points
                    let node1 = mapper_graph.find_node_containing_point(p1);
                    let node2 = mapper_graph.find_node_containing_point(p2);
                    
                    if let (Some(n1), Some(n2)) = (node1, node2) {
                        if n1 != n2 {
                            mapper_graph.add_edge(n1, n2);
                        }
                    }
                }
            }
        }
    }
    
    mapper_graph
}

/// Simple clustering algorithm for Mapper
fn cluster_points(point_indices: &[usize], points: &[Point2D], radius: f64) -> Vec<Vec<usize>> {
    let mut clusters = Vec::new();
    let mut visited = vec![false; point_indices.len()];
    
    for (i, &point_idx) in point_indices.iter().enumerate() {
        if visited[i] {
            continue;
        }
        
        let mut cluster = vec![point_idx];
        visited[i] = true;
        
        // Find all points within radius
        for (j, &other_idx) in point_indices.iter().enumerate() {
            if !visited[j] && points[point_idx].distance_to(&points[other_idx]) <= radius {
                cluster.push(other_idx);
                visited[j] = true;
            }
        }
        
        clusters.push(cluster);
    }
    
    clusters
}

/// Mapper graph structure
#[derive(Debug, Clone)]
pub struct MapperGraph {
    pub nodes: Vec<Vec<usize>>, // Each node is a cluster of point indices
    pub edges: Vec<(usize, usize)>, // Edges between node indices
}

impl MapperGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }
    
    pub fn add_node(&mut self, cluster: Vec<usize>) -> usize {
        let node_id = self.nodes.len();
        self.nodes.push(cluster);
        node_id
    }
    
    pub fn add_edge(&mut self, node1: usize, node2: usize) {
        if node1 < self.nodes.len() && node2 < self.nodes.len() && node1 != node2 {
            if !self.edges.contains(&(node1, node2)) && !self.edges.contains(&(node2, node1)) {
                self.edges.push((node1, node2));
            }
        }
    }
    
    pub fn find_node_containing_point(&self, point: usize) -> Option<usize> {
        for (node_id, cluster) in self.nodes.iter().enumerate() {
            if cluster.contains(&point) {
                return Some(node_id);
            }
        }
        None
    }
    
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }
}

/// Default filter functions for Mapper
pub fn height_filter(point: &Point2D) -> f64 {
    point.y
}

pub fn distance_filter(center: Point2D) -> impl Fn(&Point2D) -> f64 {
    move |point: &Point2D| point.distance_to(&center)
}

pub fn density_filter(points: &[Point2D], radius: f64) -> impl Fn(&Point2D) -> f64 + '_ {
    move |point: &Point2D| {
        points.iter()
            .map(|p| if p.distance_to(point) <= radius { 1.0 } else { 0.0 })
            .sum()
    }
}

/// TopologicalFeatures function for Lyra
/// Usage: TopologicalFeatures[points, maxRadius, maxDimension, numSteps]
pub fn topological_features_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (points, maxRadius, maxDimension, numSteps)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    let max_radius = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(r) => *r as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric max radius".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let max_dimension = match &args[2] {
        Value::Integer(d) => *d as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer max dimension".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let num_steps = match &args[3] {
        Value::Integer(n) => *n as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer number of filtration steps".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };

    if points.len() < 2 {
        return Err(VmError::Runtime(
            "Topological features require at least 2 points".to_string()
        ));
    }

    let features = extract_topological_features(&points, max_radius, max_dimension, num_steps);
    Ok(Value::LyObj(LyObj::new(Box::new(features))))
}

/// MapperAlgorithm function for Lyra
/// Usage: MapperAlgorithm[points, filterType, numIntervals, overlapPercentage, clusteringRadius]
pub fn mapper_algorithm_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 5 {
        return Err(VmError::TypeError {
            expected: "5 arguments (points, filterType, numIntervals, overlapPercentage, clusteringRadius)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    let filter_type = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "string filter type".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let num_intervals = match &args[2] {
        Value::Integer(n) => *n as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer number of intervals".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let overlap_percentage = match &args[3] {
        Value::Real(p) => *p,
        Value::Integer(p) => *p as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric overlap percentage".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };

    let clustering_radius = match &args[4] {
        Value::Real(r) => *r,
        Value::Integer(r) => *r as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric clustering radius".to_string(),
            actual: format!("{:?}", args[4]),
        }),
    };

    if points.len() < 2 {
        return Err(VmError::Runtime(
            "Mapper algorithm requires at least 2 points".to_string()
        ));
    }

    // Select filter function based on type
    let filter_function: fn(&Point2D) -> f64 = match filter_type {
        "height" | "y" => height_filter,
        "x" => |p: &Point2D| p.x,
        "distance" => {
            // Use centroid as center for distance filter
            let centroid = Point2D::new(
                points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
                points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
            );
            return Ok(mapper_algorithm_result_to_value(
                &mapper_algorithm(&points, distance_filter(centroid), num_intervals, overlap_percentage, clustering_radius)
            ));
        }
        _ => return Err(VmError::Runtime(
            format!("Unknown filter type: {}", filter_type)
        )),
    };

    let mapper_graph = mapper_algorithm(&points, filter_function, num_intervals, overlap_percentage, clustering_radius);
    Ok(mapper_algorithm_result_to_value(&mapper_graph))
}

/// Convert MapperGraph to Lyra Value
fn mapper_algorithm_result_to_value(graph: &MapperGraph) -> Value {
    let nodes_value: Vec<Value> = graph.nodes.iter()
        .map(|cluster| {
            let cluster_values: Vec<Value> = cluster.iter()
                .map(|&idx| Value::Integer(idx as i64))
                .collect();
            Value::List(cluster_values)
        })
        .collect();

    let edges_value: Vec<Value> = graph.edges.iter()
        .map(|(n1, n2)| {
            Value::List(vec![
                Value::Integer(*n1 as i64),
                Value::Integer(*n2 as i64),
            ])
        })
        .collect();

    Value::List(vec![
        Value::List(nodes_value),  // nodes
        Value::List(edges_value),  // edges
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_topological_features() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.866),
            Point2D::new(0.5, 0.289), // Point inside triangle
        ];
        
        let features = extract_topological_features(&points, 2.0, 2, 5);
        
        // Should have some meaningful features
        assert!(!features.betti_numbers.is_empty());
        assert!(features.num_components > 0);
    }

    #[test]
    fn test_mapper_algorithm_simple() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(2.0, 1.0),
        ];
        
        let graph = mapper_algorithm(&points, height_filter, 3, 0.2, 0.8);
        
        // Should create some nodes and potentially some edges
        assert!(graph.num_nodes() > 0);
    }

    #[test]
    fn test_cluster_points() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(0.1, 0.1),
            Point2D::new(2.0, 0.0),
            Point2D::new(2.1, 0.1),
        ];
        
        let point_indices = vec![0, 1, 2, 3];
        let clusters = cluster_points(&point_indices, &points, 0.5);
        
        // Should form 2 clusters
        assert_eq!(clusters.len(), 2);
        assert!(clusters[0].contains(&0) && clusters[0].contains(&1));
        assert!(clusters[1].contains(&2) && clusters[1].contains(&3));
    }

    #[test]
    fn test_mapper_graph() {
        let mut graph = MapperGraph::new();
        
        let node1 = graph.add_node(vec![0, 1]);
        let node2 = graph.add_node(vec![2, 3]);
        
        graph.add_edge(node1, node2);
        
        assert_eq!(graph.num_nodes(), 2);
        assert_eq!(graph.num_edges(), 1);
        assert_eq!(graph.find_node_containing_point(0), Some(0));
        assert_eq!(graph.find_node_containing_point(2), Some(1));
    }

    #[test]
    fn test_filter_functions() {
        let point = Point2D::new(3.0, 4.0);
        
        assert_eq!(height_filter(&point), 4.0);
        
        let center = Point2D::new(0.0, 0.0);
        let dist_filter = distance_filter(center);
        assert_eq!(dist_filter(&point), 5.0);
    }
}