//! K-Dimensional Tree (KDTree) Implementation
//!
//! KDTree is a space-partitioning data structure for organizing points in k-dimensional space.
//! It enables efficient nearest neighbor searches, range queries, and other spatial operations.
//! KDTrees work well for low to moderate dimensional data (typically < 20 dimensions).

use super::core::{SpatialTree, SpatialTreeType, SpatialTreeStats, DistanceMetric, SearchStats, extract_points, extract_distance_metric, SpatialIndex};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::time::Instant;

/// KDTree node structure
#[derive(Debug, Clone)]
struct KDNode {
    /// Point index in original data
    point_idx: usize,
    /// Splitting dimension
    dimension: usize,
    /// Splitting value
    split_value: f64,
    /// Left child (points with coord < split_value)
    left: Option<Box<KDNode>>,
    /// Right child (points with coord >= split_value)
    right: Option<Box<KDNode>>,
    /// Bounding box for this node's region
    bbox_min: Vec<f64>,
    bbox_max: Vec<f64>,
}

/// K-Dimensional Tree implementation
#[derive(Debug, Clone)]
pub struct KDTree {
    /// Root node of the tree
    root: Option<Box<KDNode>>,
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

impl KDTree {
    /// Create new KDTree from data points
    pub fn new(data: Vec<Vec<f64>>, metric: DistanceMetric) -> Result<Self, String> {
        Self::with_leaf_size(data, metric, 10)
    }
    
    /// Create KDTree with custom leaf size
    pub fn with_leaf_size(data: Vec<Vec<f64>>, metric: DistanceMetric, leaf_size: usize) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot create KDTree from empty data".to_string());
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
        let mut indices: Vec<usize> = (0..data.len()).collect();
        
        // Calculate global bounding box
        let mut bbox_min = vec![f64::INFINITY; dimensions];
        let mut bbox_max = vec![f64::NEG_INFINITY; dimensions];
        
        for point in &data {
            for (d, &coord) in point.iter().enumerate() {
                bbox_min[d] = bbox_min[d].min(coord);
                bbox_max[d] = bbox_max[d].max(coord);
            }
        }
        
        let mut tree = KDTree {
            root: None,
            data,
            dimensions,
            metric,
            stats: SpatialTreeStats::default(),
            leaf_size,
        };
        
        // Build the tree recursively
        tree.root = tree.build_recursive(&mut indices, 0, &bbox_min, &bbox_max, 0);
        
        // Calculate statistics
        tree.calculate_statistics();
        tree.stats.construction_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(tree)
    }
    
    /// Recursively build the KDTree
    fn build_recursive(
        &self,
        indices: &mut [usize],
        depth: usize,
        bbox_min: &[f64],
        bbox_max: &[f64],
        current_depth: usize,
    ) -> Option<Box<KDNode>> {
        if indices.is_empty() {
            return None;
        }
        
        if indices.len() <= self.leaf_size {
            // Create leaf node with first point as representative
            return Some(Box::new(KDNode {
                point_idx: indices[0],
                dimension: depth % self.dimensions,
                split_value: self.data[indices[0]][depth % self.dimensions],
                left: None,
                right: None,
                bbox_min: bbox_min.to_vec(),
                bbox_max: bbox_max.to_vec(),
            }));
        }
        
        let dimension = depth % self.dimensions;
        
        // Sort points along current dimension
        indices.sort_by(|&a, &b| {
            self.data[a][dimension].partial_cmp(&self.data[b][dimension])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        
        let median_idx = indices.len() / 2;
        let median_point_idx = indices[median_idx];
        let split_value = self.data[median_point_idx][dimension];
        
        // Create bounding boxes for children
        let mut left_bbox_max = bbox_max.to_vec();
        left_bbox_max[dimension] = split_value;
        
        let mut right_bbox_min = bbox_min.to_vec();
        right_bbox_min[dimension] = split_value;
        
        // Split indices
        let (left_indices, right_indices) = indices.split_at_mut(median_idx);
        let right_indices = &mut right_indices[1..]; // Exclude median point
        
        // Recursively build children
        let left_child = if !left_indices.is_empty() {
            self.build_recursive(left_indices, depth + 1, bbox_min, &left_bbox_max, current_depth + 1)
        } else {
            None
        };
        
        let right_child = if !right_indices.is_empty() {
            self.build_recursive(right_indices, depth + 1, &right_bbox_min, bbox_max, current_depth + 1)
        } else {
            None
        };
        
        Some(Box::new(KDNode {
            point_idx: median_point_idx,
            dimension,
            split_value,
            left: left_child,
            right: right_child,
            bbox_min: bbox_min.to_vec(),
            bbox_max: bbox_max.to_vec(),
        }))
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
    fn analyze_node(&self, node: &KDNode, depth: usize) -> (usize, usize, usize, usize) {
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
        node: &KDNode,
        query: &[f64],
        k: usize,
        heap: &mut BinaryHeap<Reverse<(f64, usize)>>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        // Calculate distance to current node's point
        let distance = self.metric.distance(query, &self.data[node.point_idx]);
        stats.distance_calculations += 1;
        
        // Add to heap if we need more points or this is closer
        if heap.len() < k {
            heap.push(Reverse((distance, node.point_idx)));
        } else if let Some(&Reverse((max_dist, _))) = heap.peek() {
            if distance < max_dist {
                heap.pop();
                heap.push(Reverse((distance, node.point_idx)));
            }
        }
        
        // Determine which child to visit first
        let query_coord = query[node.dimension];
        let (first_child, second_child) = if query_coord < node.split_value {
            (&node.left, &node.right)
        } else {
            (&node.right, &node.left)
        };
        
        // Visit first child
        if let Some(ref child) = first_child {
            self.k_nearest_recursive(child, query, k, heap, stats);
        }
        
        // Check if we need to visit second child
        let current_worst_dist = heap.peek().map(|Reverse((d, _))| *d).unwrap_or(f64::INFINITY);
        let plane_distance = (query_coord - node.split_value).abs();
        
        if heap.len() < k || plane_distance < current_worst_dist {
            if let Some(ref child) = second_child {
                self.k_nearest_recursive(child, query, k, heap, stats);
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
        node: &KDNode,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(usize, f64)>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        // Check if current node's point is within radius
        let distance = self.metric.distance(query, &self.data[node.point_idx]);
        stats.distance_calculations += 1;
        
        if distance <= radius {
            results.push((node.point_idx, distance));
        }
        
        // Check which children to visit
        let query_coord = query[node.dimension];
        let plane_distance = (query_coord - node.split_value).abs();
        
        if query_coord < node.split_value {
            // Visit left child
            if let Some(ref left) = node.left {
                self.radius_recursive(left, query, radius, results, stats);
            }
            // Visit right child if plane distance <= radius
            if plane_distance <= radius {
                if let Some(ref right) = node.right {
                    self.radius_recursive(right, query, radius, results, stats);
                }
            }
        } else {
            // Visit right child
            if let Some(ref right) = node.right {
                self.radius_recursive(right, query, radius, results, stats);
            }
            // Visit left child if plane distance <= radius
            if plane_distance <= radius {
                if let Some(ref left) = node.left {
                    self.radius_recursive(left, query, radius, results, stats);
                }
            }
        }
    }
}

impl SpatialTree for KDTree {
    fn tree_type(&self) -> SpatialTreeType {
        SpatialTreeType::KDTree
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

impl KDTree {
    /// Recursive range query
    fn range_recursive(
        &self,
        node: &KDNode,
        min_bounds: &[f64],
        max_bounds: &[f64],
        results: &mut Vec<usize>,
    ) {
        // Check if current point is in range
        let point = &self.data[node.point_idx];
        let in_range = point.iter().enumerate().all(|(i, &coord)| {
            coord >= min_bounds[i] && coord <= max_bounds[i]
        });
        
        if in_range {
            results.push(node.point_idx);
        }
        
        // Check which children to visit
        let dim = node.dimension;
        
        // Visit left child if range overlaps with left side
        if min_bounds[dim] < node.split_value {
            if let Some(ref left) = node.left {
                self.range_recursive(left, min_bounds, max_bounds, results);
            }
        }
        
        // Visit right child if range overlaps with right side
        if max_bounds[dim] >= node.split_value {
            if let Some(ref right) = node.right {
                self.range_recursive(right, min_bounds, max_bounds, results);
            }
        }
    }
}

impl Foreign for KDTree {
    fn type_name(&self) -> &'static str {
        "KDTree"
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

// Needed for k-nearest neighbor heap
use std::collections::BinaryHeap;
use std::cmp::Reverse;

// ===============================
// KDTREE CONSTRUCTOR FUNCTIONS
// ===============================

/// Create KDTree from data points
/// Syntax: KDTree[points, metric, leafSize]
pub fn kdtree(args: &[Value]) -> VmResult<Value> {
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
    
    match KDTree::with_leaf_size(points, metric, leaf_size) {
        Ok(tree) => {
            let spatial_index = SpatialIndex::new(Box::new(tree));
            Ok(Value::LyObj(LyObj::new(Box::new(spatial_index))))
        }
        Err(e) => Err(VmError::Runtime(format!("KDTree creation failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kdtree_creation() {
        let points = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let tree = KDTree::new(points, DistanceMetric::Euclidean).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dimensions(), 2);
    }
    
    #[test]
    fn test_kdtree_k_nearest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];
        
        let tree = KDTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.5, 0.5];
        let results = tree.k_nearest(&query, 2);
        
        assert_eq!(results.len(), 2);
        // Should find points [0,0] and either [1,0] or [0,1] as closest
        assert!(results.iter().any(|(idx, _)| *idx == 0)); // [0,0] should be in results
    }
    
    #[test]
    fn test_kdtree_radius_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 3.0],
        ];
        
        let tree = KDTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.0, 0.0];
        let results = tree.radius_neighbors(&query, 1.5);
        
        // Should find points [0,0], [1,0], and [0,1] within radius 1.5
        assert_eq!(results.len(), 3);
    }
    
    #[test]
    fn test_kdtree_range_query() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        
        let tree = KDTree::new(points, DistanceMetric::Euclidean).unwrap();
        let min_bounds = vec![0.5, 0.5];
        let max_bounds = vec![2.5, 2.5];
        let results = tree.range_query(&min_bounds, &max_bounds);
        
        // Should find points [1,1] and [2,2]
        assert_eq!(results.len(), 2);
    }
}