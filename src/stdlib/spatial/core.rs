//! Core Spatial Data Structure Infrastructure
//!
//! Provides the foundational traits, enums, and utility functions for spatial data structures.
//! This module defines the common interface that all spatial trees implement.

use crate::foreign::{Foreign, ForeignError};
use crate::vm::{Value};
use std::any::Any;
use std::collections::HashMap;

/// Distance metrics for spatial operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
    /// Minkowski distance with custom p
    Minkowski(f64),
    /// Haversine distance for geographic coordinates
    Haversine,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
}

impl DistanceMetric {
    /// Calculate distance between two points
    pub fn distance(&self, a: &[f64], b: &[f64]) -> f64 {
        if a.len() != b.len() {
            return f64::INFINITY;
        }

        match self {
            DistanceMetric::Euclidean => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum::<f64>()
                    .sqrt()
            }
            DistanceMetric::Manhattan => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .sum()
            }
            DistanceMetric::Chebyshev => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0, f64::max)
            }
            DistanceMetric::Minkowski(p) => {
                if *p == 0.0 {
                    return f64::INFINITY;
                }
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(*p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
            DistanceMetric::Haversine => {
                if a.len() != 2 || b.len() != 2 {
                    return f64::INFINITY;
                }
                haversine_distance(a[0], a[1], b[0], b[1])
            }
            DistanceMetric::Cosine => {
                let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f64>();
                let norm_a = a.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                let norm_b = b.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
                
                if norm_a == 0.0 || norm_b == 0.0 {
                    1.0 // Maximum distance for zero vectors
                } else {
                    1.0 - (dot_product / (norm_a * norm_b))
                }
            }
        }
    }

    /// Calculate squared distance (faster when exact distance not needed)
    pub fn distance_squared(&self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            DistanceMetric::Euclidean => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).powi(2))
                    .sum()
            }
            _ => self.distance(a, b).powi(2),
        }
    }
}

/// Haversine distance calculation for geographic coordinates (lat, lon in degrees)
fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    const R: f64 = 6371.0; // Earth's radius in kilometers
    
    let lat1_rad = lat1.to_radians();
    let lat2_rad = lat2.to_radians();
    let delta_lat = (lat2 - lat1).to_radians();
    let delta_lon = (lon2 - lon1).to_radians();
    
    let a = (delta_lat / 2.0).sin().powi(2) + 
            lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().asin();
    
    R * c
}

/// Spatial tree types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialTreeType {
    KDTree,
    BallTree,
    RTree,
    QuadTree,
    Octree,
    VPTree,
    CoverTree,
}

impl SpatialTreeType {
    /// Get tree type name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            SpatialTreeType::KDTree => "KDTree",
            SpatialTreeType::BallTree => "BallTree",
            SpatialTreeType::RTree => "RTree",
            SpatialTreeType::QuadTree => "QuadTree",
            SpatialTreeType::Octree => "Octree",
            SpatialTreeType::VPTree => "VPTree",
            SpatialTreeType::CoverTree => "CoverTree",
        }
    }
}

/// Common trait for all spatial tree implementations
pub trait SpatialTree: Foreign + std::fmt::Debug {
    /// Get the spatial tree type
    fn tree_type(&self) -> SpatialTreeType;
    
    /// Get number of points in the tree
    fn len(&self) -> usize;
    
    /// Check if tree is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get dimensionality of the space
    fn dimensions(&self) -> usize;
    
    /// Get distance metric used by the tree
    fn metric(&self) -> DistanceMetric;
    
    /// Find k nearest neighbors to query point
    fn k_nearest(&self, query: &[f64], k: usize) -> Vec<(usize, f64)>;
    
    /// Find all neighbors within radius
    fn radius_neighbors(&self, query: &[f64], radius: f64) -> Vec<(usize, f64)>;
    
    /// Find all neighbors within rectangular range
    fn range_query(&self, min_bounds: &[f64], max_bounds: &[f64]) -> Vec<usize>;
    
    /// Get the underlying data points
    fn data(&self) -> &[Vec<f64>];
    
    /// Get tree statistics (depth, balance factor, etc.)
    fn statistics(&self) -> SpatialTreeStats;
}

/// Spatial tree statistics for analysis and optimization
#[derive(Debug, Clone)]
pub struct SpatialTreeStats {
    /// Maximum depth of the tree
    pub max_depth: usize,
    /// Minimum depth to a leaf
    pub min_depth: usize,
    /// Number of leaf nodes
    pub leaf_count: usize,
    /// Number of internal nodes
    pub internal_count: usize,
    /// Balance factor (max_depth - min_depth)
    pub balance_factor: usize,
    /// Average points per leaf
    pub avg_points_per_leaf: f64,
    /// Tree construction time in milliseconds
    pub construction_time_ms: f64,
}

impl Default for SpatialTreeStats {
    fn default() -> Self {
        Self {
            max_depth: 0,
            min_depth: 0,
            leaf_count: 0,
            internal_count: 0,
            balance_factor: 0,
            avg_points_per_leaf: 0.0,
            construction_time_ms: 0.0,
        }
    }
}

/// Bounding box for spatial queries
#[derive(Debug, Clone)]
pub struct BoundingBox {
    /// Minimum coordinates for each dimension
    pub min: Vec<f64>,
    /// Maximum coordinates for each dimension
    pub max: Vec<f64>,
}

impl BoundingBox {
    /// Create new bounding box
    pub fn new(min: Vec<f64>, max: Vec<f64>) -> Result<Self, String> {
        if min.len() != max.len() {
            return Err("Min and max must have same dimensionality".to_string());
        }
        
        for (mn, mx) in min.iter().zip(max.iter()) {
            if mn > mx {
                return Err("Min coordinates must be <= max coordinates".to_string());
            }
        }
        
        Ok(BoundingBox { min, max })
    }
    
    /// Create bounding box from points
    pub fn from_points(points: &[Vec<f64>]) -> Option<Self> {
        if points.is_empty() {
            return None;
        }
        
        let dims = points[0].len();
        let mut min = vec![f64::INFINITY; dims];
        let mut max = vec![f64::NEG_INFINITY; dims];
        
        for point in points {
            if point.len() != dims {
                return None; // Inconsistent dimensions
            }
            
            for (i, &coord) in point.iter().enumerate() {
                min[i] = min[i].min(coord);
                max[i] = max[i].max(coord);
            }
        }
        
        Some(BoundingBox { min, max })
    }
    
    /// Check if point is inside bounding box
    pub fn contains(&self, point: &[f64]) -> bool {
        if point.len() != self.min.len() {
            return false;
        }
        
        point.iter().enumerate().all(|(i, &coord)| {
            coord >= self.min[i] && coord <= self.max[i]
        })
    }
    
    /// Check if two bounding boxes intersect
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        if self.min.len() != other.min.len() {
            return false;
        }
        
        self.min.iter().zip(other.max.iter()).all(|(&mn, &mx)| mn <= mx) &&
        self.max.iter().zip(other.min.iter()).all(|(&mx, &mn)| mx >= mn)
    }
    
    /// Get volume of bounding box
    pub fn volume(&self) -> f64 {
        self.min.iter().zip(self.max.iter())
            .map(|(mn, mx)| mx - mn)
            .product()
    }
    
    /// Get center point of bounding box
    pub fn center(&self) -> Vec<f64> {
        self.min.iter().zip(self.max.iter())
            .map(|(mn, mx)| (mn + mx) / 2.0)
            .collect()
    }
}

/// Nearest neighbor search result
#[derive(Debug, Clone)]
pub struct NearestNeighborResult {
    /// Indices of nearest neighbors
    pub indices: Vec<usize>,
    /// Distances to nearest neighbors
    pub distances: Vec<f64>,
    /// Query point
    pub query_point: Vec<f64>,
    /// Tree type used for search
    pub tree_type: SpatialTreeType,
    /// Search statistics
    pub search_stats: SearchStats,
}

/// Search performance statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    /// Number of distance calculations performed
    pub distance_calculations: usize,
    /// Number of nodes visited
    pub nodes_visited: usize,
    /// Search time in microseconds
    pub search_time_us: f64,
    /// Whether early termination was used
    pub early_termination: bool,
}

impl Default for SearchStats {
    fn default() -> Self {
        Self {
            distance_calculations: 0,
            nodes_visited: 0,
            search_time_us: 0.0,
            early_termination: false,
        }
    }
}

/// Generic spatial index container
#[derive(Debug)]
pub struct SpatialIndex {
    /// The actual spatial tree implementation
    tree: Box<dyn SpatialTree>,
    /// Index metadata
    metadata: HashMap<String, String>,
}

impl SpatialIndex {
    /// Create new spatial index
    pub fn new(tree: Box<dyn SpatialTree>) -> Self {
        Self {
            tree,
            metadata: HashMap::new(),
        }
    }
    
    /// Get reference to underlying tree
    pub fn tree(&self) -> &dyn SpatialTree {
        self.tree.as_ref()
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &HashMap<String, String> {
        &self.metadata
    }
    
    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl Clone for SpatialIndex {
    fn clone(&self) -> Self {
        // For now, we'll create a new tree from the data
        // This is not the most efficient, but works for the interface
        let data = self.tree.data().to_vec();
        let metric = self.tree.metric();
        
        // Create a new KDTree as default (in practice, we'd want to preserve the original type)
        let new_tree = super::kdtree::KDTree::new(data, metric).unwrap();
        
        Self {
            tree: Box::new(new_tree),
            metadata: self.metadata.clone(),
        }
    }
}

impl Foreign for SpatialIndex {
    fn type_name(&self) -> &'static str {
        "SpatialIndex"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "TreeType" => Ok(Value::String(self.tree.tree_type().as_str().to_string())),
            "Length" => Ok(Value::Integer(self.tree.len() as i64)),
            "Dimensions" => Ok(Value::Integer(self.tree.dimensions() as i64)),
            "Metric" => Ok(Value::String(format!("{:?}", self.tree.metric()))),
            "IsEmpty" => Ok(Value::Integer(if self.tree.is_empty() { 1 } else { 0 })),
            "KNearest" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let query = extract_point(&args[0])?;
                let k = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                let results = self.tree.k_nearest(&query, k);
                let result_list: Vec<Value> = results.iter()
                    .map(|&(idx, dist)| {
                        Value::List(vec![
                            Value::Integer(idx as i64),
                            Value::Real(dist)
                        ])
                    })
                    .collect();
                Ok(Value::List(result_list))
            }
            "RadiusNeighbors" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let query = extract_point(&args[0])?;
                let radius = match &args[1] {
                    Value::Real(r) => *r,
                    Value::Integer(i) => *i as f64,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Number".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                let results = self.tree.radius_neighbors(&query, radius);
                let result_list: Vec<Value> = results.iter()
                    .map(|&(idx, dist)| {
                        Value::List(vec![
                            Value::Integer(idx as i64),
                            Value::Real(dist)
                        ])
                    })
                    .collect();
                Ok(Value::List(result_list))
            }
            "Statistics" => {
                let stats = self.tree.statistics();
                let stats_list = vec![
                    Value::List(vec![Value::String("MaxDepth".to_string()), Value::Integer(stats.max_depth as i64)]),
                    Value::List(vec![Value::String("MinDepth".to_string()), Value::Integer(stats.min_depth as i64)]),
                    Value::List(vec![Value::String("LeafCount".to_string()), Value::Integer(stats.leaf_count as i64)]),
                    Value::List(vec![Value::String("InternalCount".to_string()), Value::Integer(stats.internal_count as i64)]),
                    Value::List(vec![Value::String("BalanceFactor".to_string()), Value::Integer(stats.balance_factor as i64)]),
                    Value::List(vec![Value::String("AvgPointsPerLeaf".to_string()), Value::Real(stats.avg_points_per_leaf)]),
                    Value::List(vec![Value::String("ConstructionTimeMs".to_string()), Value::Real(stats.construction_time_ms)]),
                ];
                Ok(Value::List(stats_list))
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

// ===============================
// UTILITY FUNCTIONS
// ===============================

/// Extract point coordinates from Value
pub fn extract_point(value: &Value) -> Result<Vec<f64>, ForeignError> {
    match value {
        Value::List(list) => {
            let mut point = Vec::new();
            for coord in list {
                match coord {
                    Value::Real(r) => point.push(*r),
                    Value::Integer(i) => point.push(*i as f64),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "extract_point".to_string(),
                        expected: "numeric value".to_string(),
                        actual: format!("{:?}", coord),
                    }),
                }
            }
            Ok(point)
        }
        _ => Err(ForeignError::InvalidArgumentType {
            method: "extract_point".to_string(),
            expected: "List of coordinates".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract multiple points from Value
pub fn extract_points(value: &Value) -> Result<Vec<Vec<f64>>, ForeignError> {
    match value {
        Value::List(point_list) => {
            let mut points = Vec::new();
            for point_val in point_list {
                points.push(extract_point(point_val)?);
            }
            Ok(points)
        }
        _ => Err(ForeignError::InvalidArgumentType {
            method: "extract_points".to_string(),
            expected: "List of points".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract distance metric from Value
pub fn extract_distance_metric(value: &Value) -> DistanceMetric {
    match value {
        Value::String(s) => match s.as_str() {
            "Euclidean" => DistanceMetric::Euclidean,
            "Manhattan" => DistanceMetric::Manhattan,
            "Chebyshev" => DistanceMetric::Chebyshev,
            "Haversine" => DistanceMetric::Haversine,
            "Cosine" => DistanceMetric::Cosine,
            _ => DistanceMetric::Euclidean,
        },
        _ => DistanceMetric::Euclidean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_distance_metrics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        // Test Euclidean distance
        let euclidean = DistanceMetric::Euclidean.distance(&a, &b);
        assert!((euclidean - 5.196152422706632).abs() < 1e-10);
        
        // Test Manhattan distance
        let manhattan = DistanceMetric::Manhattan.distance(&a, &b);
        assert_eq!(manhattan, 9.0);
        
        // Test Chebyshev distance
        let chebyshev = DistanceMetric::Chebyshev.distance(&a, &b);
        assert_eq!(chebyshev, 3.0);
        
        // Test squared distance optimization
        let squared = DistanceMetric::Euclidean.distance_squared(&a, &b);
        assert_eq!(squared, 27.0);
    }
    
    #[test]
    fn test_bounding_box() {
        let points = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![0.0, 5.0],
        ];
        
        let bbox = BoundingBox::from_points(&points).unwrap();
        assert_eq!(bbox.min, vec![0.0, 2.0]);
        assert_eq!(bbox.max, vec![3.0, 5.0]);
        
        // Test containment
        assert!(bbox.contains(&vec![1.0, 3.0]));
        assert!(!bbox.contains(&vec![4.0, 3.0]));
        
        // Test volume
        assert_eq!(bbox.volume(), 9.0); // (3-0) * (5-2) = 9
        
        // Test center
        assert_eq!(bbox.center(), vec![1.5, 3.5]);
    }
    
    #[test]
    fn test_haversine_distance() {
        // Test distance between New York and London (approximately)
        let ny = vec![40.7128, -74.0060];
        let london = vec![51.5074, -0.1278];
        
        let distance = DistanceMetric::Haversine.distance(&ny, &london);
        // Should be approximately 5585 km
        assert!((distance - 5585.0).abs() < 100.0);
    }
}