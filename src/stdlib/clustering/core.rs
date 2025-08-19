//! Core Clustering Data Structures and Utilities
//!
//! Provides fundamental clustering types including ClusterData Foreign objects,
//! distance metrics, and unified result structures.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Distance metrics for clustering algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Manhattan distance (L1 norm)  
    Manhattan,
    /// Cosine distance (1 - cosine similarity)
    Cosine,
    /// Minkowski distance with custom p
    Minkowski(f64),
    /// Hamming distance for categorical data
    Hamming,
    /// Jaccard distance for binary data
    Jaccard,
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
            DistanceMetric::Minkowski(p) => {
                if *p == 0.0 {
                    return f64::INFINITY;
                }
                a.iter().zip(b.iter())
                    .map(|(x, y)| (x - y).abs().powf(*p))
                    .sum::<f64>()
                    .powf(1.0 / p)
            }
            DistanceMetric::Hamming => {
                a.iter().zip(b.iter())
                    .map(|(x, y)| if (x - y).abs() < f64::EPSILON { 0.0 } else { 1.0 })
                    .sum()
            }
            DistanceMetric::Jaccard => {
                let intersection = a.iter().zip(b.iter())
                    .map(|(x, y)| if *x > 0.0 && *y > 0.0 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                let union = a.iter().zip(b.iter())
                    .map(|(x, y)| if *x > 0.0 || *y > 0.0 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                
                if union == 0.0 {
                    0.0
                } else {
                    1.0 - (intersection / union)
                }
            }
        }
    }
}

/// Core ClusterData Foreign object for representing clustered data
#[derive(Debug, Clone)]
pub struct ClusterData {
    /// Data points (n_samples x n_features)
    pub data: Vec<Vec<f64>>,
    /// Cluster labels for each data point
    pub labels: Vec<i32>,
    /// Cluster centroids 
    pub centroids: Vec<Vec<f64>>,
    /// Number of clusters
    pub n_clusters: usize,
    /// Distance metric used
    pub distance_metric: DistanceMetric,
    /// Algorithm metadata
    pub metadata: HashMap<String, String>,
    /// Cluster statistics
    pub inertia: f64,
    /// Silhouette score
    pub silhouette_score: f64,
}

impl ClusterData {
    /// Create new ClusterData
    pub fn new(
        data: Vec<Vec<f64>>, 
        labels: Vec<i32>, 
        centroids: Vec<Vec<f64>>,
        distance_metric: DistanceMetric
    ) -> Self {
        let n_clusters = centroids.len();
        let inertia = Self::calculate_inertia(&data, &labels, &centroids, distance_metric);
        let silhouette_score = Self::calculate_silhouette_score(&data, &labels, distance_metric);
        
        ClusterData {
            data,
            labels,
            centroids,
            n_clusters,
            distance_metric,
            metadata: HashMap::new(),
            inertia,
            silhouette_score,
        }
    }

    /// Calculate within-cluster sum of squares (inertia)
    fn calculate_inertia(
        data: &[Vec<f64>], 
        labels: &[i32], 
        centroids: &[Vec<f64>],
        metric: DistanceMetric
    ) -> f64 {
        data.iter()
            .zip(labels.iter())
            .map(|(point, &label)| {
                if label >= 0 && (label as usize) < centroids.len() {
                    metric.distance(point, &centroids[label as usize]).powi(2)
                } else {
                    0.0 // Noise point
                }
            })
            .sum()
    }

    /// Calculate silhouette score for clustering quality
    fn calculate_silhouette_score(
        data: &[Vec<f64>], 
        labels: &[i32],
        metric: DistanceMetric
    ) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let n = data.len();
        let mut silhouette_scores = Vec::with_capacity(n);

        for i in 0..n {
            let label_i = labels[i];
            if label_i < 0 {
                continue; // Skip noise points
            }

            // Calculate a(i): average distance to points in same cluster
            let mut same_cluster_distances = Vec::new();
            for j in 0..n {
                if i != j && labels[j] == label_i {
                    same_cluster_distances.push(metric.distance(&data[i], &data[j]));
                }
            }

            let a_i = if same_cluster_distances.is_empty() {
                0.0
            } else {
                same_cluster_distances.iter().sum::<f64>() / same_cluster_distances.len() as f64
            };

            // Calculate b(i): minimum average distance to points in other clusters
            let unique_labels: std::collections::HashSet<_> = labels.iter()
                .filter(|&&l| l >= 0 && l != label_i)
                .collect();

            let mut min_avg_distance = f64::INFINITY;
            for &other_label in unique_labels {
                let mut other_cluster_distances = Vec::new();
                for j in 0..n {
                    if labels[j] == other_label {
                        other_cluster_distances.push(metric.distance(&data[i], &data[j]));
                    }
                }

                if !other_cluster_distances.is_empty() {
                    let avg_distance = other_cluster_distances.iter().sum::<f64>() 
                        / other_cluster_distances.len() as f64;
                    min_avg_distance = min_avg_distance.min(avg_distance);
                }
            }

            let b_i = min_avg_distance;

            // Calculate silhouette score for point i
            let s_i = if a_i == 0.0 && b_i == 0.0 {
                0.0
            } else {
                (b_i - a_i) / a_i.max(b_i)
            };

            silhouette_scores.push(s_i);
        }

        if silhouette_scores.is_empty() {
            0.0
        } else {
            silhouette_scores.iter().sum::<f64>() / silhouette_scores.len() as f64
        }
    }

    /// Get number of data points
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get dimensionality of data
    pub fn n_features(&self) -> usize {
        self.data.first().map(|v| v.len()).unwrap_or(0)
    }

    /// Get points in specific cluster
    pub fn cluster_points(&self, cluster_id: i32) -> Vec<&Vec<f64>> {
        self.data.iter()
            .zip(self.labels.iter())
            .filter_map(|(point, &label)| {
                if label == cluster_id { Some(point) } else { None }
            })
            .collect()
    }

    /// Get cluster sizes
    pub fn cluster_sizes(&self) -> HashMap<i32, usize> {
        let mut sizes = HashMap::new();
        for &label in &self.labels {
            *sizes.entry(label).or_insert(0) += 1;
        }
        sizes
    }
}

impl Foreign for ClusterData {
    fn type_name(&self) -> &'static str {
        "ClusterData"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "NClusters" => Ok(Value::Integer(self.n_clusters as i64)),
            "NFeatures" => Ok(Value::Integer(self.n_features() as i64)),
            "Labels" => {
                let labels: Vec<Value> = self.labels.iter()
                    .map(|&l| Value::Integer(l as i64))
                    .collect();
                Ok(Value::List(labels))
            }
            "Centroids" => {
                let centroids: Vec<Value> = self.centroids.iter()
                    .map(|centroid| {
                        let coords: Vec<Value> = centroid.iter()
                            .map(|&c| Value::Real(c))
                            .collect();
                        Value::List(coords)
                    })
                    .collect();
                Ok(Value::List(centroids))
            }
            "Data" => {
                let data: Vec<Value> = self.data.iter()
                    .map(|point| {
                        let coords: Vec<Value> = point.iter()
                            .map(|&c| Value::Real(c))
                            .collect();
                        Value::List(coords)
                    })
                    .collect();
                Ok(Value::List(data))
            }
            "Inertia" => Ok(Value::Real(self.inertia)),
            "SilhouetteScore" => Ok(Value::Real(self.silhouette_score)),
            "DistanceMetric" => Ok(Value::String(format!("{:?}", self.distance_metric))),
            "ClusterSizes" => {
                let sizes = self.cluster_sizes();
                let size_list: Vec<Value> = sizes.iter()
                    .map(|(&label, &size)| {
                        Value::List(vec![
                            Value::Integer(label as i64),
                            Value::Integer(size as i64)
                        ])
                    })
                    .collect();
                Ok(Value::List(size_list))
            }
            "ClusterPoints" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let cluster_id = match &args[0] {
                    Value::Integer(i) => *i as i32,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let cluster_points = self.cluster_points(cluster_id);
                let points: Vec<Value> = cluster_points.iter()
                    .map(|point| {
                        let coords: Vec<Value> = point.iter()
                            .map(|&c| Value::Real(c))
                            .collect();
                        Value::List(coords)
                    })
                    .collect();
                Ok(Value::List(points))
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

/// Distance matrix for precomputed distances
#[derive(Debug, Clone)]
pub struct DistanceMatrix {
    /// Symmetric distance matrix
    pub matrix: Vec<Vec<f64>>,
    /// Size of matrix (n x n)
    pub size: usize,
    /// Distance metric used
    pub metric: DistanceMetric,
}

impl DistanceMatrix {
    /// Create distance matrix from data points
    pub fn from_data(data: &[Vec<f64>], metric: DistanceMetric) -> Self {
        let n = data.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in i+1..n {
                let distance = metric.distance(&data[i], &data[j]);
                matrix[i][j] = distance;
                matrix[j][i] = distance;
            }
        }

        DistanceMatrix {
            matrix,
            size: n,
            metric,
        }
    }

    /// Get distance between two points
    pub fn get(&self, i: usize, j: usize) -> f64 {
        if i < self.size && j < self.size {
            self.matrix[i][j]
        } else {
            f64::INFINITY
        }
    }

    /// Get k nearest neighbors for point i
    pub fn k_nearest(&self, i: usize, k: usize) -> Vec<(usize, f64)> {
        if i >= self.size {
            return Vec::new();
        }

        let mut distances: Vec<(usize, f64)> = (0..self.size)
            .filter(|&j| j != i)
            .map(|j| (j, self.matrix[i][j]))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances
    }
}

impl Foreign for DistanceMatrix {
    fn type_name(&self) -> &'static str {
        "DistanceMatrix"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Size" => Ok(Value::Integer(self.size as i64)),
            "Metric" => Ok(Value::String(format!("{:?}", self.metric))),
            "Get" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                let i = match &args[0] {
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let j = match &args[1] {
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };

                Ok(Value::Real(self.get(i, j)))
            }
            "KNearest" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                let i = match &args[0] {
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let k = match &args[1] {
                    Value::Integer(k_val) => *k_val as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };

                let neighbors = self.k_nearest(i, k);
                let result: Vec<Value> = neighbors.iter()
                    .map(|&(idx, dist)| {
                        Value::List(vec![
                            Value::Integer(idx as i64),
                            Value::Real(dist)
                        ])
                    })
                    .collect();
                Ok(Value::List(result))
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
// CORE CLUSTERING FUNCTIONS
// ===============================

/// Create ClusterData from data points, labels, and centroids
/// Syntax: ClusterData[data, labels, centroids, distance_metric]
pub fn cluster_data(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (data, labels, centroids, [distance_metric])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract data points
    let data = extract_data_matrix(&args[0])?;
    
    // Extract labels
    let labels = match &args[1] {
        Value::List(list) => {
            let mut label_vec = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => label_vec.push(*i as i32),
                    _ => return Err(VmError::TypeError {
                        expected: "integer list for labels".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            label_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List of integers for labels".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Extract centroids
    let centroids = extract_data_matrix(&args[2])?;

    // Extract distance metric
    let distance_metric = if args.len() == 4 {
        match &args[3] {
            Value::String(s) => match s.as_str() {
                "Euclidean" => DistanceMetric::Euclidean,
                "Manhattan" => DistanceMetric::Manhattan,
                "Cosine" => DistanceMetric::Cosine,
                "Hamming" => DistanceMetric::Hamming,
                "Jaccard" => DistanceMetric::Jaccard,
                _ => DistanceMetric::Euclidean,
            },
            _ => DistanceMetric::Euclidean,
        }
    } else {
        DistanceMetric::Euclidean
    };

    let cluster_data = ClusterData::new(data, labels, centroids, distance_metric);
    Ok(Value::LyObj(LyObj::new(Box::new(cluster_data))))
}

/// Create DistanceMatrix from data points
/// Syntax: DistanceMatrix[data, distance_metric]
pub fn distance_matrix(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (data, [distance_metric])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let data = extract_data_matrix(&args[0])?;

    let distance_metric = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => match s.as_str() {
                "Euclidean" => DistanceMetric::Euclidean,
                "Manhattan" => DistanceMetric::Manhattan,
                "Cosine" => DistanceMetric::Cosine,
                "Hamming" => DistanceMetric::Hamming,
                "Jaccard" => DistanceMetric::Jaccard,
                _ => DistanceMetric::Euclidean,
            },
            _ => DistanceMetric::Euclidean,
        }
    } else {
        DistanceMetric::Euclidean
    };

    let dist_matrix = DistanceMatrix::from_data(&data, distance_metric);
    Ok(Value::LyObj(LyObj::new(Box::new(dist_matrix))))
}

/// Utility function to extract 2D data matrix from Value
pub fn extract_data_matrix(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(outer_list) => {
            let mut data = Vec::new();
            for row in outer_list {
                match row {
                    Value::List(inner_list) => {
                        let mut point = Vec::new();
                        for coord in inner_list {
                            match coord {
                                Value::Real(r) => point.push(*r),
                                Value::Integer(i) => point.push(*i as f64),
                                _ => return Err(VmError::TypeError {
                                    expected: "numeric value".to_string(),
                                    actual: format!("{:?}", coord),
                                }),
                            }
                        }
                        data.push(point);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of lists (matrix)".to_string(),
                        actual: format!("list containing {:?}", row),
                    }),
                }
            }
            Ok(data)
        }
        _ => Err(VmError::TypeError {
            expected: "List of lists (matrix)".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Utility function to extract ClusterData from Value
pub fn extract_cluster_data(value: &Value) -> VmResult<&ClusterData> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<ClusterData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "ClusterData".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "ClusterData".to_string(),
            actual: format!("{:?}", value),
        }),
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

        // Test Cosine distance
        let cosine = DistanceMetric::Cosine.distance(&a, &b);
        assert!(cosine >= 0.0 && cosine <= 1.0);
    }

    #[test]
    fn test_cluster_data_creation() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 3.0],
            vec![8.0, 9.0],
            vec![9.0, 10.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![
            vec![1.5, 2.5],
            vec![8.5, 9.5],
        ];

        let cluster_data = ClusterData::new(data, labels, centroids, DistanceMetric::Euclidean);
        
        assert_eq!(cluster_data.len(), 4);
        assert_eq!(cluster_data.n_clusters, 2);
        assert_eq!(cluster_data.n_features(), 2);
        assert!(cluster_data.inertia > 0.0);
    }

    #[test]
    fn test_distance_matrix_creation() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
        ];

        let dist_matrix = DistanceMatrix::from_data(&data, DistanceMetric::Euclidean);
        
        assert_eq!(dist_matrix.size, 3);
        assert_eq!(dist_matrix.get(0, 0), 0.0);
        assert!((dist_matrix.get(0, 1) - 1.4142135623730951).abs() < 1e-10);
        assert!((dist_matrix.get(0, 2) - 2.8284271247461903).abs() < 1e-10);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![2.0, 0.0],
        ];

        let dist_matrix = DistanceMatrix::from_data(&data, DistanceMetric::Euclidean);
        let neighbors = dist_matrix.k_nearest(0, 2);
        
        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].0, 1); // Nearest is point 1
        assert_eq!(neighbors[1].0, 2); // Second nearest is point 2
    }

    #[test]
    fn test_silhouette_score_calculation() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![11.0, 10.0],
        ];
        let labels = vec![0, 0, 1, 1];
        let centroids = vec![
            vec![0.5, 0.0],
            vec![10.5, 10.0],
        ];

        let cluster_data = ClusterData::new(data, labels, centroids, DistanceMetric::Euclidean);
        
        // Should have good silhouette score for well-separated clusters
        assert!(cluster_data.silhouette_score > 0.5);
    }

    #[test]
    fn test_cluster_data_function() {
        let data_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        let labels_value = Value::List(vec![Value::Integer(0), Value::Integer(1)]);
        let centroids_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);

        let result = cluster_data(&[data_value, labels_value, centroids_value]).unwrap();

        match result {
            Value::LyObj(obj) => {
                let cluster_data = obj.downcast_ref::<ClusterData>().unwrap();
                assert_eq!(cluster_data.len(), 2);
                assert_eq!(cluster_data.n_clusters, 2);
            }
            _ => panic!("Expected ClusterData object"),
        }
    }
}