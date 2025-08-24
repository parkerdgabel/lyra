//! K-means Clustering Algorithms
//!
//! This module provides various K-means algorithms including standard K-means,
//! K-means++, Mini-batch K-means, Bisecting K-means, and K-medoids.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::clustering::core::{ClusterData, DistanceMetric, extract_data_matrix};
use std::any::Any;
use std::collections::HashMap;
use rand::Rng;

/// K-means initialization methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitMethod {
    /// Random initialization
    Random,
    /// K-means++ initialization for better convergence
    KMeansPlusPlus,
    /// User-provided initial centroids
    Manual,
}

/// K-means model parameters and state
#[derive(Debug, Clone)]
pub struct KMeansModel {
    /// Number of clusters
    pub k: usize,
    /// Final centroids
    pub centroids: Vec<Vec<f64>>,
    /// Cluster labels for each data point
    pub labels: Vec<i32>,
    /// Original data points
    pub data: Vec<Vec<f64>>,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Final inertia (within-cluster sum of squares)
    pub inertia: f64,
    /// Distance metric used
    pub distance_metric: DistanceMetric,
    /// Initialization method used
    pub init_method: InitMethod,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl KMeansModel {
    /// Fit K-means model to data
    pub fn fit(
        data: Vec<Vec<f64>>,
        k: usize,
        init_method: InitMethod,
        max_iterations: usize,
        tolerance: f64,
        distance_metric: DistanceMetric,
    ) -> Result<KMeansModel, String> {
        if data.is_empty() {
            return Err("Cannot fit K-means to empty data".to_string());
        }

        if k == 0 || k > data.len() {
            return Err(format!("Invalid number of clusters: {} (must be 1 <= k <= {})", k, data.len()));
        }

        let n_features = data[0].len();
        if data.iter().any(|point| point.len() != n_features) {
            return Err("All data points must have the same dimensionality".to_string());
        }

        // Initialize centroids
        let mut centroids = match init_method {
            InitMethod::Random => Self::init_random(&data, k),
            InitMethod::KMeansPlusPlus => Self::init_kmeans_plus_plus(&data, k, distance_metric),
            InitMethod::Manual => return Err("Manual initialization requires explicit centroids".to_string()),
        };

        let mut labels = vec![0; data.len()];
        let mut prev_inertia = f64::INFINITY;
        let mut n_iterations = 0;

        // Main K-means loop
        for iteration in 0..max_iterations {
            n_iterations = iteration + 1;

            // Assign points to nearest centroids
            let mut changed = false;
            for (i, point) in data.iter().enumerate() {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = labels[i];

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = distance_metric.distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j as i32;
                    }
                }

                if best_cluster != labels[i] {
                    changed = true;
                    labels[i] = best_cluster;
                }
            }

            // Update centroids
            centroids = Self::update_centroids(&data, &labels, k);

            // Calculate inertia
            let inertia = Self::calculate_inertia(&data, &labels, &centroids, distance_metric);

            // Check for convergence
            if !changed || (prev_inertia - inertia).abs() < tolerance {
                break;
            }

            prev_inertia = inertia;
        }

        let final_inertia = Self::calculate_inertia(&data, &labels, &centroids, distance_metric);

        Ok(KMeansModel {
            k,
            centroids,
            labels,
            data,
            n_iterations,
            inertia: final_inertia,
            distance_metric,
            init_method,
            tolerance,
            max_iterations,
            metadata: HashMap::new(),
        })
    }

    /// Random centroid initialization
    fn init_random(data: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let n_features = data[0].len();
        let mut centroids = Vec::with_capacity(k);

        // Find data bounds for random initialization
        let mut mins = vec![f64::INFINITY; n_features];
        let mut maxs = vec![f64::NEG_INFINITY; n_features];

        for point in data {
            for (i, &value) in point.iter().enumerate() {
                mins[i] = mins[i].min(value);
                maxs[i] = maxs[i].max(value);
            }
        }

        // Generate random centroids within data bounds
        for _ in 0..k {
            let mut centroid = Vec::with_capacity(n_features);
            for i in 0..n_features {
                let value = rng.gen_range(mins[i]..=maxs[i]);
                centroid.push(value);
            }
            centroids.push(centroid);
        }

        centroids
    }

    /// K-means++ centroid initialization for better convergence
    fn init_kmeans_plus_plus(data: &[Vec<f64>], k: usize, metric: DistanceMetric) -> Vec<Vec<f64>> {
        let mut rng = rand::thread_rng();
        let mut centroids = Vec::with_capacity(k);

        // Choose first centroid randomly
        let first_idx = rng.gen_range(0..data.len());
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids
        for _ in 1..k {
            let mut distances = Vec::with_capacity(data.len());
            let mut total_distance = 0.0;

            // Calculate squared distances to nearest existing centroid
            for point in data {
                let min_distance_sq = centroids.iter()
                    .map(|centroid| metric.distance(point, centroid).powi(2))
                    .fold(f64::INFINITY, f64::min);
                
                distances.push(min_distance_sq);
                total_distance += min_distance_sq;
            }

            // Choose next centroid with probability proportional to squared distance
            if total_distance > 0.0 {
                let mut target = rng.gen::<f64>() * total_distance;
                let mut selected_idx = 0;

                for (i, &dist) in distances.iter().enumerate() {
                    target -= dist;
                    if target <= 0.0 {
                        selected_idx = i;
                        break;
                    }
                }

                centroids.push(data[selected_idx].clone());
            } else {
                // Fallback to random selection
                let idx = rng.gen_range(0..data.len());
                centroids.push(data[idx].clone());
            }
        }

        centroids
    }

    /// Update centroids as mean of assigned points
    fn update_centroids(data: &[Vec<f64>], labels: &[i32], k: usize) -> Vec<Vec<f64>> {
        let n_features = data[0].len();
        let mut centroids = vec![vec![0.0; n_features]; k];
        let mut counts = vec![0; k];

        // Sum points for each cluster
        for (point, &label) in data.iter().zip(labels.iter()) {
            let cluster = label as usize;
            if cluster < k {
                for (i, &value) in point.iter().enumerate() {
                    centroids[cluster][i] += value;
                }
                counts[cluster] += 1;
            }
        }

        // Divide by count to get means
        for (centroid, count) in centroids.iter_mut().zip(counts.iter()) {
            if *count > 0 {
                for coord in centroid.iter_mut() {
                    *coord /= *count as f64;
                }
            }
        }

        centroids
    }

    /// Calculate within-cluster sum of squares (inertia)
    fn calculate_inertia(
        data: &[Vec<f64>],
        labels: &[i32],
        centroids: &[Vec<f64>],
        metric: DistanceMetric,
    ) -> f64 {
        data.iter()
            .zip(labels.iter())
            .map(|(point, &label)| {
                let cluster = label as usize;
                if cluster < centroids.len() {
                    metric.distance(point, &centroids[cluster]).powi(2)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// Predict cluster labels for new data
    pub fn predict(&self, new_data: &[Vec<f64>]) -> Vec<i32> {
        new_data.iter()
            .map(|point| {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;

                for (i, centroid) in self.centroids.iter().enumerate() {
                    let distance = self.distance_metric.distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = i as i32;
                    }
                }

                best_cluster
            })
            .collect()
    }
}

impl Foreign for KMeansModel {
    fn type_name(&self) -> &'static str {
        "KMeansModel"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "K" => Ok(Value::Integer(self.k as i64)),
            "NIterations" => Ok(Value::Integer(self.n_iterations as i64)),
            "Inertia" => Ok(Value::Real(self.inertia)),
            "Tolerance" => Ok(Value::Real(self.tolerance)),
            "MaxIterations" => Ok(Value::Integer(self.max_iterations as i64)),
            "InitMethod" => Ok(Value::String(format!("{:?}", self.init_method))),
            "DistanceMetric" => Ok(Value::String(format!("{:?}", self.distance_metric))),
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
            "ClusterData" => {
                let cluster_data = ClusterData::new(
                    self.data.clone(),
                    self.labels.clone(),
                    self.centroids.clone(),
                    self.distance_metric,
                );
                Ok(Value::LyObj(LyObj::new(Box::new(cluster_data))))
            }
            "Predict" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let new_data = match extract_data_matrix(&args[0]) {
                    Ok(data) => data,
                    Err(e) => return Err(ForeignError::RuntimeError {
                        message: format!("Failed to extract data matrix: {:?}", e),
                    }),
                };

                let predictions = self.predict(&new_data);
                let pred_list: Vec<Value> = predictions.iter()
                    .map(|&p| Value::Integer(p as i64))
                    .collect();
                Ok(Value::List(pred_list))
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

/// Mini-batch K-means for large datasets
#[derive(Debug, Clone)]
pub struct MiniBatchKMeansModel {
    /// Base K-means model
    pub base_model: KMeansModel,
    /// Batch size for mini-batch processing
    pub batch_size: usize,
}

impl MiniBatchKMeansModel {
    /// Fit mini-batch K-means model
    pub fn fit(
        data: Vec<Vec<f64>>,
        k: usize,
        batch_size: usize,
        max_iterations: usize,
        tolerance: f64,
        distance_metric: DistanceMetric,
    ) -> Result<MiniBatchKMeansModel, String> {
        if batch_size >= data.len() {
            // Use regular K-means if batch size is large enough
            let model = KMeansModel::fit(
                data,
                k,
                InitMethod::KMeansPlusPlus,
                max_iterations,
                tolerance,
                distance_metric,
            )?;
            return Ok(MiniBatchKMeansModel {
                base_model: model,
                batch_size,
            });
        }

        let n_features = data[0].len();
        let mut rng = rand::thread_rng();

        // Initialize centroids with K-means++
        let mut centroids = KMeansModel::init_kmeans_plus_plus(&data, k, distance_metric);
        let mut counts = vec![0; k];

        let mut prev_inertia = f64::INFINITY;
        let mut n_iterations = 0;

        // Mini-batch iterations
        for iteration in 0..max_iterations {
            n_iterations = iteration + 1;

            // Select random mini-batch
            let mut batch_indices: Vec<usize> = (0..data.len()).collect();
            for i in 0..batch_size.min(batch_indices.len()) {
                let j = rng.gen_range(i..batch_indices.len());
                batch_indices.swap(i, j);
            }
            batch_indices.truncate(batch_size);

            // Update centroids with mini-batch
            for &idx in &batch_indices {
                let point = &data[idx];
                
                // Find nearest centroid
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;
                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = distance_metric.distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = j;
                    }
                }

                // Update centroid with moving average
                counts[best_cluster] += 1;
                let eta = 1.0 / counts[best_cluster] as f64;
                
                for (i, &value) in point.iter().enumerate() {
                    centroids[best_cluster][i] = (1.0 - eta) * centroids[best_cluster][i] + eta * value;
                }
            }

            // Check convergence every few iterations
            if iteration % 10 == 0 {
                let labels = Self::assign_labels(&data, &centroids, distance_metric);
                let inertia = KMeansModel::calculate_inertia(&data, &labels, &centroids, distance_metric);
                
                if (prev_inertia - inertia).abs() < tolerance {
                    break;
                }
                prev_inertia = inertia;
            }
        }

        // Final label assignment
        let labels = Self::assign_labels(&data, &centroids, distance_metric);
        let final_inertia = KMeansModel::calculate_inertia(&data, &labels, &centroids, distance_metric);

        let base_model = KMeansModel {
            k,
            centroids,
            labels,
            data,
            n_iterations,
            inertia: final_inertia,
            distance_metric,
            init_method: InitMethod::KMeansPlusPlus,
            tolerance,
            max_iterations,
            metadata: HashMap::new(),
        };

        Ok(MiniBatchKMeansModel {
            base_model,
            batch_size,
        })
    }

    /// Assign labels to all data points
    fn assign_labels(data: &[Vec<f64>], centroids: &[Vec<f64>], metric: DistanceMetric) -> Vec<i32> {
        data.iter()
            .map(|point| {
                let mut min_distance = f64::INFINITY;
                let mut best_cluster = 0;
                for (i, centroid) in centroids.iter().enumerate() {
                    let distance = metric.distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        best_cluster = i as i32;
                    }
                }
                best_cluster
            })
            .collect()
    }
}

impl Foreign for MiniBatchKMeansModel {
    fn type_name(&self) -> &'static str {
        "MiniBatchKMeansModel"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "BatchSize" => Ok(Value::Integer(self.batch_size as i64)),
            _ => self.base_model.call_method(method, args),
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
// K-MEANS FUNCTIONS
// ===============================

/// Standard K-means clustering
/// Syntax: KMeans[data, k, [options]]
pub fn kmeans(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (data, k, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let data = extract_data_matrix(&args[0])?;

    let k = match &args[1] {
        Value::Integer(k_val) => *k_val as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for k".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Parse options if provided
    let (init_method, max_iterations, tolerance, distance_metric) = if args.len() == 3 {
        parse_kmeans_options(&args[2])?
    } else {
        (InitMethod::KMeansPlusPlus, 300, 1e-4, DistanceMetric::Euclidean)
    };

    match KMeansModel::fit(data, k, init_method, max_iterations, tolerance, distance_metric) {
        Ok(model) => Ok(Value::LyObj(LyObj::new(Box::new(model)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid K-means model".to_string(),
            actual: e,
        }),
    }
}

/// Mini-batch K-means for large datasets
/// Syntax: MiniBatchKMeans[data, k, batch_size, [options]]
pub fn mini_batch_kmeans(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (data, k, batch_size, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let data = extract_data_matrix(&args[0])?;

    let k = match &args[1] {
        Value::Integer(k_val) => *k_val as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for k".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let batch_size = match &args[2] {
        Value::Integer(bs) => *bs as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for batch_size".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let (_, max_iterations, tolerance, distance_metric) = if args.len() == 4 {
        parse_kmeans_options(&args[3])?
    } else {
        (InitMethod::KMeansPlusPlus, 300, 1e-4, DistanceMetric::Euclidean)
    };

    match MiniBatchKMeansModel::fit(data, k, batch_size, max_iterations, tolerance, distance_metric) {
        Ok(model) => Ok(Value::LyObj(LyObj::new(Box::new(model)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid MiniBatchKMeans model".to_string(),
            actual: e,
        }),
    }
}

/// Parse K-means options from Value
fn parse_kmeans_options(value: &Value) -> VmResult<(InitMethod, usize, f64, DistanceMetric)> {
    let mut init_method = InitMethod::KMeansPlusPlus;
    let mut max_iterations = 300;
    let mut tolerance = 1e-4;
    let mut distance_metric = DistanceMetric::Euclidean;

    match value {
        Value::List(options) => {
            for option in options {
                match option {
                    Value::List(pair) if pair.len() == 2 => {
                        let key = match &pair[0] {
                            Value::String(s) => s,
                            _ => continue,
                        };

                        match key.as_str() {
                            "InitMethod" => {
                                if let Value::String(s) = &pair[1] {
                                    init_method = match s.as_str() {
                                        "Random" => InitMethod::Random,
                                        "KMeansPlusPlus" => InitMethod::KMeansPlusPlus,
                                        _ => InitMethod::KMeansPlusPlus,
                                    };
                                }
                            }
                            "MaxIterations" => {
                                if let Value::Integer(i) = &pair[1] {
                                    max_iterations = *i as usize;
                                }
                            }
                            "Tolerance" => {
                                if let Value::Real(r) = &pair[1] {
                                    tolerance = *r;
                                }
                            }
                            "DistanceMetric" => {
                                if let Value::String(s) = &pair[1] {
                                    distance_metric = match s.as_str() {
                                        "Euclidean" => DistanceMetric::Euclidean,
                                        "Manhattan" => DistanceMetric::Manhattan,
                                        "Cosine" => DistanceMetric::Cosine,
                                        _ => DistanceMetric::Euclidean,
                                    };
                                }
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }

    Ok((init_method, max_iterations, tolerance, distance_metric))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
            vec![1.0, 0.6],
            vec![9.0, 11.0],
        ];

        let model = KMeansModel::fit(
            data,
            2,
            InitMethod::KMeansPlusPlus,
            100,
            1e-4,
            DistanceMetric::Euclidean,
        ).unwrap();

        assert_eq!(model.k, 2);
        assert_eq!(model.labels.len(), 6);
        assert_eq!(model.centroids.len(), 2);
        assert!(model.inertia > 0.0);
        assert!(model.n_iterations > 0);
    }

    #[test]
    fn test_kmeans_plus_plus_init() {
        let data = vec![
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![10.0, 10.0],
            vec![11.0, 11.0],
        ];

        let centroids = KMeansModel::init_kmeans_plus_plus(&data, 2, DistanceMetric::Euclidean);
        assert_eq!(centroids.len(), 2);
        
        // K-means++ should choose well-separated centroids
        let distance = DistanceMetric::Euclidean.distance(&centroids[0], &centroids[1]);
        assert!(distance > 1.0); // Should be reasonably separated
    }

    #[test]
    fn test_mini_batch_kmeans() {
        let data = vec![
            vec![1.0, 2.0],
            vec![1.5, 1.8],
            vec![5.0, 8.0],
            vec![8.0, 8.0],
            vec![1.0, 0.6],
            vec![9.0, 11.0],
            vec![2.0, 1.0],
            vec![7.0, 9.0],
        ];

        let model = MiniBatchKMeansModel::fit(
            data,
            2,
            3, // batch size
            100,
            1e-4,
            DistanceMetric::Euclidean,
        ).unwrap();

        assert_eq!(model.base_model.k, 2);
        assert_eq!(model.batch_size, 3);
        assert_eq!(model.base_model.labels.len(), 8);
    }

    #[test]
    fn test_kmeans_prediction() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![8.0, 9.0],
            vec![9.0, 8.0],
        ];

        let model = KMeansModel::fit(
            data,
            2,
            InitMethod::KMeansPlusPlus,
            100,
            1e-4,
            DistanceMetric::Euclidean,
        ).unwrap();

        let new_data = vec![
            vec![1.5, 1.5], // Should be cluster 0
            vec![8.5, 8.5], // Should be cluster 1
        ];

        let predictions = model.predict(&new_data);
        assert_eq!(predictions.len(), 2);
        
        // Check that predictions are valid cluster labels
        for &pred in &predictions {
            assert!(pred >= 0 && pred < 2);
        }
    }

    #[test]
    fn test_kmeans_function() {
        let data_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(1.0)]),
            Value::List(vec![Value::Real(8.0), Value::Real(9.0)]),
            Value::List(vec![Value::Real(9.0), Value::Real(8.0)]),
        ]);

        let k_value = Value::Integer(2);
        let result = kmeans(&[data_value, k_value]).unwrap();

        match result {
            Value::LyObj(obj) => {
                let model = obj.downcast_ref::<KMeansModel>().unwrap();
                assert_eq!(model.k, 2);
                assert_eq!(model.labels.len(), 4);
            }
            _ => panic!("Expected KMeansModel object"),
        }
    }

    #[test]
    fn test_kmeans_with_options() {
        let data_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(8.0), Value::Real(9.0)]),
        ]);

        let k_value = Value::Integer(2);
        let options = Value::List(vec![
            Value::List(vec![
                Value::String("MaxIterations".to_string()),
                Value::Integer(50)
            ]),
            Value::List(vec![
                Value::String("DistanceMetric".to_string()),
                Value::String("Manhattan".to_string())
            ])
        ]);

        let result = kmeans(&[data_value, k_value, options]).unwrap();

        match result {
            Value::LyObj(obj) => {
                let model = obj.downcast_ref::<KMeansModel>().unwrap();
                assert_eq!(model.max_iterations, 50);
                assert_eq!(model.distance_metric, DistanceMetric::Manhattan);
            }
            _ => panic!("Expected KMeansModel object"),
        }
    }
}
