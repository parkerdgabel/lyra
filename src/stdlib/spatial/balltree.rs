//! Ball Tree Implementation
//!
//! Ball Tree is a space-partitioning data structure that recursively divides points
//! into nested hyperspheres (balls). It excels in high-dimensional spaces where
//! KDTree performance degrades, and works with arbitrary distance metrics.
//!
//! Ball trees are particularly effective for:
//! - High-dimensional nearest neighbor searches (> 20 dimensions)
//! - Non-Euclidean distance metrics
//! - Data with non-uniform distribution

use super::core::{SpatialTree, SpatialTreeType, SpatialTreeStats, DistanceMetric, SearchStats, extract_points, extract_distance_metric, SpatialIndex};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::BinaryHeap;
use std::cmp::Reverse;
use std::time::Instant;

/// Ball tree node structure
#[derive(Debug, Clone)]
struct BallNode {
    /// Center of the ball (centroid of points)
    center: Vec<f64>,
    /// Radius of the ball
    radius: f64,
    /// Point indices stored in this node (for leaf nodes)
    point_indices: Vec<usize>,
    /// Left child node
    left: Option<Box<BallNode>>,
    /// Right child node  
    right: Option<Box<BallNode>>,
    /// Distance from parent center (used for pruning)
    parent_distance: f64,
}

/// Ball Tree implementation
#[derive(Debug, Clone)]
pub struct BallTree {
    /// Root node of the tree
    root: Option<Box<BallNode>>,
    /// Original data points
    data: Vec<Vec<f64>>,
    /// Number of dimensions
    dimensions: usize,
    /// Distance metric
    metric: DistanceMetric,
    /// Tree construction statistics
    stats: SpatialTreeStats,
    /// Maximum leaf size (points per leaf)
    leaf_size: usize,
}

impl BallTree {
    /// Create new BallTree from data points
    pub fn new(data: Vec<Vec<f64>>, metric: DistanceMetric) -> Result<Self, String> {
        Self::with_leaf_size(data, metric, 10)
    }
    
    /// Create BallTree with custom leaf size
    pub fn with_leaf_size(data: Vec<Vec<f64>>, metric: DistanceMetric, leaf_size: usize) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot create BallTree from empty data".to_string());
        }
        
        let dimensions = data[0].len();
        if dimensions == 0 {
            return Err("Points must have at least one dimension".to_string());
        }
        
        // Validate all points have same dimensionality
        for (i, point) in data.iter().enumerate() {
            if point.len() != dimensions {
                return Err(format!("Point {} has {} dimensions, expected {}", i, point.len(), dimensions));
            }
        }
        
        let start_time = Instant::now();
        
        // Create point indices for building
        let indices: Vec<usize> = (0..data.len()).collect();
        
        let mut tree = BallTree {
            root: None,
            data,
            dimensions,
            metric,
            stats: SpatialTreeStats::default(),
            leaf_size,
        };
        
        // Build the tree recursively
        tree.root = tree.build_recursive(indices, 0.0);
        
        // Calculate statistics
        tree.calculate_statistics();
        tree.stats.construction_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(tree)
    }
    
    /// Recursively build the ball tree
    fn build_recursive(&self, indices: Vec<usize>, parent_dist: f64) -> Option<Box<BallNode>> {
        if indices.is_empty() {
            return None;
        }
        
        // Calculate centroid of points
        let center = self.calculate_centroid(&indices);
        
        // Calculate radius (maximum distance from center to any point)
        let radius = indices.iter()
            .map(|&idx| self.metric.distance(&center, &self.data[idx]))
            .fold(0.0, f64::max);
        
        // Create leaf node if we have few enough points
        if indices.len() <= self.leaf_size {
            return Some(Box::new(BallNode {
                center,
                radius,
                point_indices: indices,
                left: None,
                right: None,
                parent_distance: parent_dist,
            }));
        }
        
        // Find the point furthest from centroid
        let furthest_idx = indices.iter()
            .enumerate()
            .max_by(|(_, &a), (_, &b)| {
                let dist_a = self.metric.distance(&center, &self.data[*a]);
                let dist_b = self.metric.distance(&center, &self.data[*b]);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(_, &idx)| idx)
            .unwrap();
        
        // Find the point furthest from the furthest point
        let second_furthest_idx = indices.iter()
            .filter(|&&idx| idx != furthest_idx)
            .max_by(|&&a, &&b| {
                let dist_a = self.metric.distance(&self.data[furthest_idx], &self.data[a]);
                let dist_b = self.metric.distance(&self.data[furthest_idx], &self.data[b]);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .unwrap_or(furthest_idx);
        
        // Split points based on which of the two cluster centers they're closer to
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();
        
        for &idx in &indices {
            if idx == furthest_idx {
                left_indices.push(idx);
            } else if idx == second_furthest_idx {
                right_indices.push(idx);
            } else {
                let dist_to_left = self.metric.distance(&self.data[furthest_idx], &self.data[idx]);
                let dist_to_right = self.metric.distance(&self.data[second_furthest_idx], &self.data[idx]);
                
                if dist_to_left <= dist_to_right {
                    left_indices.push(idx);
                } else {
                    right_indices.push(idx);
                }
            }
        }
        
        // Ensure both sides have at least one point
        if left_indices.is_empty() {
            left_indices.push(right_indices.pop().unwrap());
        } else if right_indices.is_empty() {
            right_indices.push(left_indices.pop().unwrap());
        }
        
        // Recursively build children
        let left_dist = self.metric.distance(&center, &self.data[furthest_idx]);
        let right_dist = self.metric.distance(&center, &self.data[second_furthest_idx]);
        
        let left_child = if !left_indices.is_empty() {
            self.build_recursive(left_indices, left_dist)
        } else {
            None
        };
        
        let right_child = if !right_indices.is_empty() {
            self.build_recursive(right_indices, right_dist)
        } else {
            None
        };
        
        Some(Box::new(BallNode {
            center,
            radius,
            point_indices: Vec::new(), // Internal nodes don't store points
            left: left_child,
            right: right_child,
            parent_distance: parent_dist,
        }))
    }
    
    /// Calculate centroid of points given by indices
    fn calculate_centroid(&self, indices: &[usize]) -> Vec<f64> {
        let mut centroid = vec![0.0; self.dimensions];
        
        for &idx in indices {
            for (i, &coord) in self.data[idx].iter().enumerate() {
                centroid[i] += coord;
            }
        }
        
        let count = indices.len() as f64;
        for coord in &mut centroid {
            *coord /= count;
        }
        
        centroid
    }
    
    /// Calculate tree statistics
    fn calculate_statistics(&mut self) {
        if let Some(ref root) = self.root {
            let (max_depth, min_depth, leaf_count, internal_count) = self.analyze_node(root, 0);
            
            self.stats.max_depth = max_depth;
            self.stats.min_depth = min_depth;
            self.stats.leaf_count = leaf_count;
            self.stats.internal_count = internal_count;
            self.stats.balance_factor = max_depth.saturating_sub(min_depth);
            self.stats.avg_points_per_leaf = if leaf_count > 0 {
                self.data.len() as f64 / leaf_count as f64
            } else {
                0.0
            };
        }
    }
    
    /// Analyze node for statistics
    fn analyze_node(&self, node: &BallNode, depth: usize) -> (usize, usize, usize, usize) {
        match (&node.left, &node.right) {
            (None, None) => {
                // Leaf node
                (depth, depth, 1, 0)
            }
            _ => {
                // Internal node
                let mut max_depth = depth;
                let mut min_depth = depth;
                let mut leaf_count = 0;
                let mut internal_count = 1; // This node
                
                if let Some(ref left) = node.left {
                    let (l_max, l_min, l_leaves, l_internal) = self.analyze_node(left, depth + 1);
                    max_depth = max_depth.max(l_max);
                    min_depth = min_depth.min(l_min);
                    leaf_count += l_leaves;
                    internal_count += l_internal;
                }
                
                if let Some(ref right) = node.right {
                    let (r_max, r_min, r_leaves, r_internal) = self.analyze_node(right, depth + 1);
                    max_depth = max_depth.max(r_max);
                    min_depth = min_depth.min(r_min);
                    leaf_count += r_leaves;
                    internal_count += r_internal;
                }
                
                (max_depth, min_depth, leaf_count, internal_count)
            }
        }
    }
    
    /// K-nearest neighbor search
    fn k_nearest_search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if let Some(ref root) = self.root {
            let mut heap = BinaryHeap::new();
            let mut stats = SearchStats::default();
            
            self.k_nearest_recursive(root, query, k, &mut heap, &mut stats);
            
            // Convert max heap to sorted vector (closest first)
            let mut results = Vec::new();
            while let Some(Reverse((dist, idx))) = heap.pop() {
                results.push((idx, dist));
            }
            results.reverse();
            results
        } else {
            Vec::new()
        }
    }
    
    /// Recursive k-nearest neighbor search
    fn k_nearest_recursive(
        &self,
        node: &BallNode,
        query: &[f64],
        k: usize,
        heap: &mut BinaryHeap<Reverse<(f64, usize)>>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        // Distance from query to ball center
        let center_distance = self.metric.distance(query, &node.center);
        stats.distance_calculations += 1;
        
        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &point_idx in &node.point_indices {
                let distance = self.metric.distance(query, &self.data[point_idx]);
                stats.distance_calculations += 1;
                
                if heap.len() < k {
                    heap.push(Reverse((distance, point_idx)));
                } else if let Some(&Reverse((max_dist, _))) = heap.peek() {
                    if distance < max_dist {
                        heap.pop();
                        heap.push(Reverse((distance, point_idx)));
                    }
                }
            }
            return;
        }
        
        // For internal nodes, check if we need to explore children
        let current_worst_dist = heap.peek().map(|Reverse((d, _))| *d).unwrap_or(f64::INFINITY);
        
        // Check if the ball could contain better points
        let min_possible_distance = (center_distance - node.radius).max(0.0);
        if heap.len() < k || min_possible_distance < current_worst_dist {
            // Determine which child to visit first (closer one)
            let mut children = Vec::new();
            if let Some(ref left) = node.left {
                let left_dist = self.metric.distance(query, &left.center);
                children.push((left_dist, left));
            }
            if let Some(ref right) = node.right {
                let right_dist = self.metric.distance(query, &right.center);
                children.push((right_dist, right));
            }
            
            // Sort by distance to visit closer child first
            children.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            
            for (_, child) in children {
                // Check if this child could possibly contain better points
                let child_center_dist = self.metric.distance(query, &child.center);
                let child_min_dist = (child_center_dist - child.radius).max(0.0);
                
                let current_worst = heap.peek().map(|Reverse((d, _))| *d).unwrap_or(f64::INFINITY);
                if heap.len() < k || child_min_dist < current_worst {
                    self.k_nearest_recursive(child, query, k, heap, stats);
                } else {
                    stats.early_termination = true;
                }
            }
        } else {
            stats.early_termination = true;
        }
    }
    
    /// Radius neighbor search
    fn radius_search(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        if let Some(ref root) = self.root {
            let mut results = Vec::new();
            let mut stats = SearchStats::default();
            
            self.radius_recursive(root, query, radius, &mut results, &mut stats);
            
            // Sort by distance
            results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            results
        } else {
            Vec::new()
        }
    }
    
    /// Recursive radius search
    fn radius_recursive(
        &self,
        node: &BallNode,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(usize, f64)>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        // Distance from query to ball center
        let center_distance = self.metric.distance(query, &node.center);
        stats.distance_calculations += 1;
        
        // If the ball is completely outside the search radius, skip it
        if center_distance - node.radius > radius {
            return;
        }
        
        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &point_idx in &node.point_indices {
                let distance = self.metric.distance(query, &self.data[point_idx]);
                stats.distance_calculations += 1;
                
                if distance <= radius {
                    results.push((point_idx, distance));
                }
            }
            return;
        }
        
        // For internal nodes, recursively search children if they might contain points within radius
        if let Some(ref left) = node.left {
            let left_center_dist = self.metric.distance(query, &left.center);
            if left_center_dist - left.radius <= radius {
                self.radius_recursive(left, query, radius, results, stats);
            }
        }
        
        if let Some(ref right) = node.right {
            let right_center_dist = self.metric.distance(query, &right.center);
            if right_center_dist - right.radius <= radius {
                self.radius_recursive(right, query, radius, results, stats);
            }
        }
    }
}

impl SpatialTree for BallTree {
    fn tree_type(&self) -> SpatialTreeType {
        SpatialTreeType::BallTree
    }
    
    fn len(&self) -> usize {
        self.data.len()
    }
    
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn metric(&self) -> DistanceMetric {
        self.metric
    }
    
    fn k_nearest(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if query.len() != self.dimensions {
            return Vec::new();
        }
        self.k_nearest_search(query, k)
    }
    
    fn radius_neighbors(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)> {
        if query.len() != self.dimensions {
            return Vec::new();
        }
        self.radius_search(query, radius)
    }
    
    fn range_query(&self, min_bounds: &[f64], max_bounds: &[f64]) -> Vec<usize> {
        if min_bounds.len() != self.dimensions || max_bounds.len() != self.dimensions {
            return Vec::new();
        }
        
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_recursive(root, min_bounds, max_bounds, &mut results);
        }
        results
    }
    
    fn data(&self) -> &[Vec<f64>] {
        &self.data
    }
    
    fn statistics(&self) -> SpatialTreeStats {
        self.stats.clone()
    }
}

impl BallTree {
    /// Recursive range query
    fn range_recursive(
        &self,
        node: &BallNode,
        min_bounds: &[f64],
        max_bounds: &[f64],
        results: &mut Vec<usize>,
    ) {
        // Check if ball could intersect with the range
        let mut ball_could_intersect = true;
        for (i, (&center_coord, &radius)) in node.center.iter().zip(std::iter::repeat(node.radius)).enumerate() {
            let min_ball = center_coord - radius;
            let max_ball = center_coord + radius;
            
            if max_ball < min_bounds[i] || min_ball > max_bounds[i] {
                ball_could_intersect = false;
                break;
            }
        }
        
        if !ball_could_intersect {
            return;
        }
        
        // If this is a leaf node, check all points
        if node.left.is_none() && node.right.is_none() {
            for &point_idx in &node.point_indices {
                let point = &self.data[point_idx];
                let in_range = point.iter().enumerate().all(|(i, &coord)| {
                    coord >= min_bounds[i] && coord <= max_bounds[i]
                });
                
                if in_range {
                    results.push(point_idx);
                }
            }
            return;
        }
        
        // Recursively search children
        if let Some(ref left) = node.left {
            self.range_recursive(left, min_bounds, max_bounds, results);
        }
        
        if let Some(ref right) = node.right {
            self.range_recursive(right, min_bounds, max_bounds, results);
        }
    }
}

impl Foreign for BallTree {
    fn type_name(&self) -> &'static str {
        "BallTree"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "Dimensions" => Ok(Value::Integer(self.dimensions() as i64)),
            "LeafSize" => Ok(Value::Integer(self.leaf_size as i64)),
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

// ===============================
// BALLTREE CONSTRUCTOR FUNCTIONS
// ===============================

/// Create BallTree from data points
/// Syntax: BallTree[points, metric, leafSize]
pub fn balltree(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (points, [metric], [leafSize])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let points = extract_points(&args[0])?;
    
    let metric = if args.len() > 1 {
        extract_distance_metric(&args[1])
    } else {
        DistanceMetric::Euclidean
    };
    
    let leaf_size = if args.len() > 2 {
        match &args[2] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for leaf size".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        10
    };
    
    match BallTree::with_leaf_size(points, metric, leaf_size) {
        Ok(tree) => {
            let spatial_index = SpatialIndex::new(Box::new(tree));
            Ok(Value::LyObj(LyObj::new(Box::new(spatial_index))))
        }
        Err(e) => Err(VmError::Runtime(format!("BallTree creation failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_balltree_creation() {
        let points = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Euclidean).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dimensions(), 2);
    }
    
    #[test]
    fn test_balltree_k_nearest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.5, 0.5];
        let results = tree.k_nearest(&query, 2);
        
        assert_eq!(results.len(), 2);
        // Should find points [0,0] and either [1,0] or [0,1] as closest
        assert!(results.iter().any(|(idx, _)| *idx == 0)); // [0,0] should be in results
    }
    
    #[test]
    fn test_balltree_radius_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 3.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.0, 0.0];
        let results = tree.radius_neighbors(&query, 1.5);
        
        // Should find points [0,0], [1,0], and [0,1] within radius 1.5
        assert_eq!(results.len(), 3);
    }
    
    #[test]
    fn test_balltree_range_query() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Euclidean).unwrap();
        let min_bounds = vec![0.5, 0.5];
        let max_bounds = vec![2.5, 2.5];
        let results = tree.range_query(&min_bounds, &max_bounds);
        
        // Should find points [1,1] and [2,2]
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_balltree_high_dimensional() {
        // Test with higher dimensional data where BallTree should outperform KDTree
        let points = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![2.0, 3.0, 4.0, 5.0, 6.0],
            vec![3.0, 4.0, 5.0, 6.0, 7.0],
            vec![4.0, 5.0, 6.0, 7.0, 8.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Euclidean).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dimensions(), 5);
        
        let query = vec![2.5, 3.5, 4.5, 5.5, 6.5];
        let results = tree.k_nearest(&query, 2);
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_balltree_cosine_distance() {
        // Test with cosine distance metric
        let points = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
        ];
        
        let tree = BallTree::new(points, DistanceMetric::Cosine).unwrap();
        let query = vec![1.0, 0.5, 0.0];
        let results = tree.k_nearest(&query, 2);
        
        assert_eq!(results.len(), 2);
        // With cosine distance, [1,1,0] should be closest to [1,0.5,0]
    }
}