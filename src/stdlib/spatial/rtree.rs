//! R-Tree Implementation  
//!
//! R-Tree is a tree data structure used for spatial access methods, specifically for
//! indexing multi-dimensional information such as geographical coordinates, rectangles, 
//! or polygons. It excels at spatial queries involving rectangles and regions.
//!
//! R-trees are particularly effective for:
//! - GIS applications with rectangular regions
//! - Range queries over multi-dimensional rectangles  
//! - Spatial joins and intersection queries
//! - Database spatial indexing

use super::core::{SpatialTree, SpatialTreeType, SpatialTreeStats, DistanceMetric, SearchStats, extract_points, extract_distance_metric, SpatialIndex, BoundingBox};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::time::Instant;

/// Maximum number of entries in a node
const MAX_ENTRIES: usize = 8;
/// Minimum number of entries in a node (typically MAX/2)
const MIN_ENTRIES: usize = 4;

/// R-tree node structure
#[derive(Debug, Clone)]
struct RTreeNode {
    /// Bounding box that encompasses all child bounding boxes
    bbox: BoundingBox,
    /// Child nodes (for internal nodes)
    children: Vec<RTreeNode>,
    /// Point indices (for leaf nodes only)
    point_indices: Vec<usize>,
    /// Whether this is a leaf node
    is_leaf: bool,
}

/// R-Tree implementation for spatial indexing of rectangles
#[derive(Debug, Clone)]
pub struct RTree {
    /// Root node of the tree
    root: Option<RTreeNode>,
    /// Original data points (treated as point rectangles)
    data: Vec<Vec<f64>>,
    /// Number of dimensions
    dimensions: usize,
    /// Distance metric (for point queries)
    metric: DistanceMetric,
    /// Tree construction statistics
    stats: SpatialTreeStats,
}

impl RTreeNode {
    /// Create new leaf node
    fn new_leaf(bbox: BoundingBox) -> Self {
        Self {
            bbox,
            children: Vec::new(),
            point_indices: Vec::new(),
            is_leaf: true,
        }
    }
    
    /// Create new internal node
    fn new_internal(bbox: BoundingBox) -> Self {
        Self {
            bbox,
            children: Vec::new(),
            point_indices: Vec::new(),
            is_leaf: false,
        }
    }
    
    /// Get area of bounding box
    fn area(&self) -> f64 {
        self.bbox.volume()
    }
    
    /// Calculate area increase if bbox were expanded to include other
    fn area_increase(&self, other: &BoundingBox) -> f64 {
        let current_area = self.area();
        let expanded_bbox = self.expand_to_include(other);
        expanded_bbox.volume() - current_area
    }
    
    /// Create bounding box that includes this node's bbox and the other bbox
    fn expand_to_include(&self, other: &BoundingBox) -> BoundingBox {
        let min: Vec<f64> = self.bbox.min.iter()
            .zip(other.min.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        
        let max: Vec<f64> = self.bbox.max.iter()
            .zip(other.max.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        
        BoundingBox::new(min, max).unwrap()
    }
}

impl RTree {
    /// Create new RTree from data points
    pub fn new(data: Vec<Vec<f64>>, metric: DistanceMetric) -> Result<Self, String> {
        if data.is_empty() {
            return Err("Cannot create RTree from empty data".to_string());
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
        
        let mut tree = RTree {
            root: None,
            data,
            dimensions,
            metric,
            stats: SpatialTreeStats::default(),
        };
        
        // Build the tree by inserting each point
        tree.build_tree();
        
        // Calculate statistics
        tree.calculate_statistics();
        tree.stats.construction_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        
        Ok(tree)
    }
    
    /// Build the R-tree by inserting all points
    fn build_tree(&mut self) {
        // Collect the data first to avoid borrowing issues
        let data_points: Vec<(usize, Vec<f64>)> = self.data.iter().enumerate()
            .map(|(i, point)| (i, point.clone()))
            .collect();
            
        for (i, point) in data_points {
            // Create point bounding box (min = max = point)
            let bbox = BoundingBox::new(point.clone(), point.clone()).unwrap();
            self.insert(bbox, i);
        }
    }
    
    /// Insert a bounding box with associated point index
    fn insert(&mut self, bbox: BoundingBox, point_idx: usize) {
        if self.root.is_some() {
            let overflow = {
                let root = self.root.as_mut().unwrap();
                self.insert_recursive(root, bbox, point_idx)
            };
            
            // Handle root overflow by creating new root
            if let Some((bbox1, node1, bbox2, node2)) = overflow {
                let mut new_root = RTreeNode::new_internal(
                    self.union_bbox(&bbox1, &bbox2)
                );
                
                let mut child1 = RTreeNode::new_internal(bbox1);
                child1.children = node1;
                child1.is_leaf = child1.children.is_empty();
                
                let mut child2 = RTreeNode::new_internal(bbox2);
                child2.children = node2;
                child2.is_leaf = child2.children.is_empty();
                
                new_root.children = vec![child1, child2];
                self.root = Some(new_root);
            }
        } else {
            // Create first node
            let mut leaf = RTreeNode::new_leaf(bbox.clone());
            leaf.point_indices.push(point_idx);
            self.root = Some(leaf);
        }
    }
    
    /// Recursive insertion that returns overflow split if needed
    fn insert_recursive(
        &self,
        node: &mut RTreeNode,
        bbox: BoundingBox,
        point_idx: usize,
    ) -> Option<(BoundingBox, Vec<RTreeNode>, BoundingBox, Vec<RTreeNode>)> {
        if node.is_leaf {
            // Insert into leaf node
            node.point_indices.push(point_idx);
            node.bbox = self.union_bbox(&node.bbox, &bbox);
            
            // Check for overflow
            if node.point_indices.len() > MAX_ENTRIES {
                self.split_leaf_node(node)
            } else {
                None
            }
        } else {
            // Choose child with minimum area increase
            let best_child_idx = self.choose_child(node, &bbox);
            
            let overflow = self.insert_recursive(&mut node.children[best_child_idx], bbox, point_idx);
            
            // Update bounding box
            node.bbox = self.calculate_node_bbox(node);
            
            // Handle child overflow
            if let Some((bbox1, nodes1, bbox2, nodes2)) = overflow {
                // Replace child with split results
                node.children.remove(best_child_idx);
                
                let mut child1 = RTreeNode::new_internal(bbox1);
                child1.children = nodes1;
                child1.is_leaf = child1.children.is_empty();
                
                let mut child2 = RTreeNode::new_internal(bbox2);
                child2.children = nodes2;
                child2.is_leaf = child2.children.is_empty();
                
                node.children.insert(best_child_idx, child1);
                node.children.insert(best_child_idx + 1, child2);
                
                // Check for overflow
                if node.children.len() > MAX_ENTRIES {
                    self.split_internal_node(node)
                } else {
                    None
                }
            } else {
                None
            }
        }
    }
    
    /// Choose best child for insertion
    fn choose_child(&self, node: &RTreeNode, bbox: &BoundingBox) -> usize {
        node.children.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let increase_a = a.area_increase(bbox);
                let increase_b = b.area_increase(bbox);
                increase_a.partial_cmp(&increase_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
    
    /// Split leaf node using quadratic algorithm
    fn split_leaf_node(
        &self,
        node: &mut RTreeNode,
    ) -> Option<(BoundingBox, Vec<RTreeNode>, BoundingBox, Vec<RTreeNode>)> {
        let indices = std::mem::take(&mut node.point_indices);
        
        // Find seeds (most wasteful pair)
        let (seed1, seed2) = self.pick_seeds_leaf(&indices);
        
        let mut group1 = vec![seed1];
        let mut group2 = vec![seed2];
        
        let mut remaining: Vec<usize> = indices.into_iter()
            .filter(|&idx| idx != seed1 && idx != seed2)
            .collect();
        
        // Distribute remaining points
        while !remaining.is_empty() {
            // Ensure minimum entries
            if group1.len() + remaining.len() == MIN_ENTRIES {
                group1.extend(remaining.drain(..));
                break;
            }
            if group2.len() + remaining.len() == MIN_ENTRIES {
                group2.extend(remaining.drain(..));
                break;
            }
            
            // Pick next entry
            let next_idx = self.pick_next_leaf(&group1, &group2, &remaining);
            let idx = remaining.remove(next_idx);
            
            // Add to group with least area increase
            let bbox1 = self.calculate_bbox_for_indices(&group1);
            let bbox2 = self.calculate_bbox_for_indices(&group2);
            let point_bbox = BoundingBox::new(self.data[idx].clone(), self.data[idx].clone()).unwrap();
            
            let increase1 = bbox1.volume() - self.union_bbox(&bbox1, &point_bbox).volume();
            let increase2 = bbox2.volume() - self.union_bbox(&bbox2, &point_bbox).volume();
            
            if increase1 < increase2 {
                group1.push(idx);
            } else {
                group2.push(idx);
            }
        }
        
        // Create new leaf nodes
        let bbox1 = self.calculate_bbox_for_indices(&group1);
        let bbox2 = self.calculate_bbox_for_indices(&group2);
        
        let mut leaf1 = RTreeNode::new_leaf(bbox1.clone());
        leaf1.point_indices = group1;
        
        let mut leaf2 = RTreeNode::new_leaf(bbox2.clone());
        leaf2.point_indices = group2;
        
        Some((bbox1, vec![leaf1], bbox2, vec![leaf2]))
    }
    
    /// Split internal node
    fn split_internal_node(
        &self,
        node: &mut RTreeNode,
    ) -> Option<(BoundingBox, Vec<RTreeNode>, BoundingBox, Vec<RTreeNode>)> {
        let children = std::mem::take(&mut node.children);
        
        // Simple split for now - could be improved with better algorithms
        let mid = children.len() / 2;
        let (group1, group2) = children.split_at(mid);
        
        let bbox1 = self.calculate_bbox_for_children(group1);
        let bbox2 = self.calculate_bbox_for_children(group2);
        
        Some((bbox1, group1.to_vec(), bbox2, group2.to_vec()))
    }
    
    /// Pick seeds for quadratic split (leaf nodes)
    fn pick_seeds_leaf(&self, indices: &[usize]) -> (usize, usize) {
        let mut max_waste = f64::NEG_INFINITY;
        let mut best_pair = (indices[0], indices[1]);
        
        for (i, &idx1) in indices.iter().enumerate() {
            for &idx2 in indices.iter().skip(i + 1) {
                let bbox1 = BoundingBox::new(self.data[idx1].clone(), self.data[idx1].clone()).unwrap();
                let bbox2 = BoundingBox::new(self.data[idx2].clone(), self.data[idx2].clone()).unwrap();
                let union = self.union_bbox(&bbox1, &bbox2);
                
                let waste = union.volume() - bbox1.volume() - bbox2.volume();
                if waste > max_waste {
                    max_waste = waste;
                    best_pair = (idx1, idx2);
                }
            }
        }
        
        best_pair
    }
    
    /// Pick next entry for quadratic split (leaf nodes)
    fn pick_next_leaf(&self, group1: &[usize], group2: &[usize], remaining: &[usize]) -> usize {
        let bbox1 = self.calculate_bbox_for_indices(group1);
        let bbox2 = self.calculate_bbox_for_indices(group2);
        
        let mut max_difference = f64::NEG_INFINITY;
        let mut best_idx = 0;
        
        for (i, &idx) in remaining.iter().enumerate() {
            let point_bbox = BoundingBox::new(self.data[idx].clone(), self.data[idx].clone()).unwrap();
            
            let increase1 = self.union_bbox(&bbox1, &point_bbox).volume() - bbox1.volume();
            let increase2 = self.union_bbox(&bbox2, &point_bbox).volume() - bbox2.volume();
            
            let difference = (increase1 - increase2).abs();
            if difference > max_difference {
                max_difference = difference;
                best_idx = i;
            }
        }
        
        best_idx
    }
    
    /// Calculate bounding box for a set of point indices
    fn calculate_bbox_for_indices(&self, indices: &[usize]) -> BoundingBox {
        if indices.is_empty() {
            return BoundingBox::new(vec![0.0; self.dimensions], vec![0.0; self.dimensions]).unwrap();
        }
        
        let points: Vec<Vec<f64>> = indices.iter().map(|&i| self.data[i].clone()).collect();
        BoundingBox::from_points(&points).unwrap()
    }
    
    /// Calculate bounding box for a set of child nodes
    fn calculate_bbox_for_children(&self, children: &[RTreeNode]) -> BoundingBox {
        if children.is_empty() {
            return BoundingBox::new(vec![0.0; self.dimensions], vec![0.0; self.dimensions]).unwrap();
        }
        
        let mut min = children[0].bbox.min.clone();
        let mut max = children[0].bbox.max.clone();
        
        for child in children.iter().skip(1) {
            for (i, (&child_min, &child_max)) in child.bbox.min.iter().zip(child.bbox.max.iter()).enumerate() {
                min[i] = min[i].min(child_min);
                max[i] = max[i].max(child_max);
            }
        }
        
        BoundingBox::new(min, max).unwrap()
    }
    
    /// Calculate bounding box for a node based on its contents
    fn calculate_node_bbox(&self, node: &RTreeNode) -> BoundingBox {
        if node.is_leaf {
            self.calculate_bbox_for_indices(&node.point_indices)
        } else {
            self.calculate_bbox_for_children(&node.children)
        }
    }
    
    /// Union of two bounding boxes
    fn union_bbox(&self, bbox1: &BoundingBox, bbox2: &BoundingBox) -> BoundingBox {
        let min: Vec<f64> = bbox1.min.iter()
            .zip(bbox2.min.iter())
            .map(|(a, b)| a.min(*b))
            .collect();
        
        let max: Vec<f64> = bbox1.max.iter()
            .zip(bbox2.max.iter())
            .map(|(a, b)| a.max(*b))
            .collect();
        
        BoundingBox::new(min, max).unwrap()
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
    fn analyze_node(&self, node: &RTreeNode, depth: usize) -> (usize, usize, usize, usize) {
        if node.is_leaf {
            (depth, depth, 1, 0)
        } else {
            let mut max_depth = depth;
            let mut min_depth = usize::MAX;
            let mut leaf_count = 0;
            let mut internal_count = 1; // This node
            
            for child in &node.children {
                let (c_max, c_min, c_leaves, c_internal) = self.analyze_node(child, depth + 1);
                max_depth = max_depth.max(c_max);
                min_depth = min_depth.min(c_min);
                leaf_count += c_leaves;
                internal_count += c_internal;
            }
            
            (max_depth, min_depth, leaf_count, internal_count)
        }
    }
    
    /// K-nearest neighbor search
    fn k_nearest_search(&self, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        if let Some(ref root) = self.root {
            let mut candidates = Vec::new();
            let mut stats = SearchStats::default();
            
            self.k_nearest_recursive(root, query, k, &mut candidates, &mut stats);
            
            // Sort by distance and take k closest
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            candidates.truncate(k);
            candidates
        } else {
            Vec::new()
        }
    }
    
    /// Recursive k-nearest neighbor search
    fn k_nearest_recursive(
        &self,
        node: &RTreeNode,
        query: &[f64],
        k: usize,
        candidates: &mut Vec<(usize, f64)>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        if node.is_leaf {
            // Check all points in leaf
            for &point_idx in &node.point_indices {
                let distance = self.metric.distance(query, &self.data[point_idx]);
                stats.distance_calculations += 1;
                candidates.push((point_idx, distance));
            }
        } else {
            // Sort children by minimum distance to query point
            let mut child_distances: Vec<(usize, f64)> = node.children.iter()
                .enumerate()
                .map(|(i, child)| {
                    let min_dist = self.min_distance_to_bbox(query, &child.bbox);
                    (i, min_dist)
                })
                .collect();
            
            child_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            
            // Visit children in order of increasing minimum distance
            for (child_idx, _) in child_distances {
                self.k_nearest_recursive(&node.children[child_idx], query, k, candidates, stats);
            }
        }
    }
    
    /// Calculate minimum distance from point to bounding box
    fn min_distance_to_bbox(&self, point: &[f64], bbox: &BoundingBox) -> f64 {
        let mut distance_sq = 0.0;
        
        for (i, &coord) in point.iter().enumerate() {
            if coord < bbox.min[i] {
                distance_sq += (coord - bbox.min[i]).powi(2);
            } else if coord > bbox.max[i] {
                distance_sq += (coord - bbox.max[i]).powi(2);
            }
            // If coord is within [min, max], distance contribution is 0
        }
        
        distance_sq.sqrt()
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
        node: &RTreeNode,
        query: &[f64],
        radius: f64,
        results: &mut Vec<(usize, f64)>,
        stats: &mut SearchStats,
    ) {
        stats.nodes_visited += 1;
        
        // Check if bounding box could contain points within radius
        let min_dist = self.min_distance_to_bbox(query, &node.bbox);
        if min_dist > radius {
            return; // Prune this subtree
        }
        
        if node.is_leaf {
            // Check all points in leaf
            for &point_idx in &node.point_indices {
                let distance = self.metric.distance(query, &self.data[point_idx]);
                stats.distance_calculations += 1;
                
                if distance <= radius {
                    results.push((point_idx, distance));
                }
            }
        } else {
            // Recursively search children
            for child in &node.children {
                self.radius_recursive(child, query, radius, results, stats);
            }
        }
    }
}

impl SpatialTree for RTree {
    fn tree_type(&self) -> SpatialTreeType {
        SpatialTreeType::RTree
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
        
        let query_bbox = BoundingBox::new(min_bounds.to_vec(), max_bounds.to_vec());
        if query_bbox.is_err() {
            return Vec::new();
        }
        let query_bbox = query_bbox.unwrap();
        
        let mut results = Vec::new();
        if let Some(ref root) = self.root {
            self.range_recursive(root, &query_bbox, &mut results);
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

impl RTree {
    /// Recursive range query
    fn range_recursive(
        &self,
        node: &RTreeNode,
        query_bbox: &BoundingBox,
        results: &mut Vec<usize>,
    ) {
        // Check if node's bounding box intersects with query range
        if !node.bbox.intersects(query_bbox) {
            return; // Prune this subtree
        }
        
        if node.is_leaf {
            // Check all points in leaf
            for &point_idx in &node.point_indices {
                let point = &self.data[point_idx];
                let in_range = point.iter().enumerate().all(|(i, &coord)| {
                    coord >= query_bbox.min[i] && coord <= query_bbox.max[i]
                });
                
                if in_range {
                    results.push(point_idx);
                }
            }
        } else {
            // Recursively search children
            for child in &node.children {
                self.range_recursive(child, query_bbox, results);
            }
        }
    }
}

impl Foreign for RTree {
    fn type_name(&self) -> &'static str {
        "RTree"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "Dimensions" => Ok(Value::Integer(self.dimensions() as i64)),
            "MaxEntries" => Ok(Value::Integer(MAX_ENTRIES as i64)),
            "MinEntries" => Ok(Value::Integer(MIN_ENTRIES as i64)),
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
// RTREE CONSTRUCTOR FUNCTIONS
// ===============================

/// Create RTree from data points
/// Syntax: RTree[points, metric]
pub fn rtree(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (points, [metric])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let points = extract_points(&args[0])?;
    
    let metric = if args.len() > 1 {
        extract_distance_metric(&args[1])
    } else {
        DistanceMetric::Euclidean
    };
    
    match RTree::new(points, metric) {
        Ok(tree) => {
            let spatial_index = SpatialIndex::new(Box::new(tree));
            Ok(Value::LyObj(LyObj::new(Box::new(spatial_index))))
        }
        Err(e) => Err(VmError::Runtime(format!("RTree creation failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_rtree_creation() {
        let points = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let tree = RTree::new(points, DistanceMetric::Euclidean).unwrap();
        assert_eq!(tree.len(), 4);
        assert_eq!(tree.dimensions(), 2);
    }
    
    #[test]
    fn test_rtree_k_nearest() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
            vec![0.0, 2.0],
        ];
        
        let tree = RTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.5, 0.5];
        let results = tree.k_nearest(&query, 2);
        
        assert_eq!(results.len(), 2);
        // Should find points [0,0] and either [1,0] or [0,1] as closest
        assert!(results.iter().any(|(idx, _)| *idx == 0)); // [0,0] should be in results
    }
    
    #[test]
    fn test_rtree_radius_search() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![3.0, 3.0],
        ];
        
        let tree = RTree::new(points, DistanceMetric::Euclidean).unwrap();
        let query = vec![0.0, 0.0];
        let results = tree.radius_neighbors(&query, 1.5);
        
        // Should find points [0,0], [1,0], and [0,1] within radius 1.5
        assert_eq!(results.len(), 3);
    }
    
    #[test]
    fn test_rtree_range_query() {
        let points = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
        ];
        
        let tree = RTree::new(points, DistanceMetric::Euclidean).unwrap();
        let min_bounds = vec![0.5, 0.5];
        let max_bounds = vec![2.5, 2.5];
        let results = tree.range_query(&min_bounds, &max_bounds);
        
        // Should find points [1,1] and [2,2]
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_rtree_large_dataset() {
        // Test with larger dataset to trigger splits
        let mut points = Vec::new();
        for i in 0..20 {
            for j in 0..20 {
                points.push(vec![i as f64, j as f64]);
            }
        }
        
        let tree = RTree::new(points, DistanceMetric::Euclidean).unwrap();
        assert_eq!(tree.len(), 400);
        
        // Test k-nearest on large dataset
        let query = vec![10.0, 10.0];
        let results = tree.k_nearest(&query, 5);
        assert_eq!(results.len(), 5);
        
        // Test range query
        let min_bounds = vec![5.0, 5.0];
        let max_bounds = vec![15.0, 15.0];
        let range_results = tree.range_query(&min_bounds, &max_bounds);
        assert_eq!(range_results.len(), 121); // 11x11 grid
    }
    
    #[test]
    fn test_bounding_box_operations() {
        let bbox1 = BoundingBox::new(vec![0.0, 0.0], vec![2.0, 2.0]).unwrap();
        let bbox2 = BoundingBox::new(vec![1.0, 1.0], vec![3.0, 3.0]).unwrap();
        
        // Test intersection
        assert!(bbox1.intersects(&bbox2));
        
        // Test containment
        assert!(bbox1.contains(&vec![1.0, 1.0]));
        assert!(!bbox1.contains(&vec![3.0, 3.0]));
        
        // Test volume
        assert_eq!(bbox1.volume(), 4.0); // 2x2
    }
}