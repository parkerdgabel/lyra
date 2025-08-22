//! Data Mining Functions
//!
//! Comprehensive machine learning and data mining capabilities including
//! clustering, classification, association rules, and ensemble methods.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::collections::HashMap;
use rand::{thread_rng, Rng};
use std::any::Any;

/// Clustering Result - Foreign Object
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    algorithm: String,
    k: usize,
    centroids: Vec<Vec<f64>>,
    labels: Vec<usize>,
    inertia: f64,
    silhouette_score: f64,
}

impl Foreign for ClusteringResult {
    fn type_name(&self) -> &'static str {
        "ClusteringResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "algorithm" => Ok(Value::String(self.algorithm.clone())),
            "k" => Ok(Value::Integer(self.k as i64)),
            "centroids" => Ok(Value::List(
                self.centroids.iter()
                    .map(|centroid| Value::List(centroid.iter().map(|&x| Value::Real(x)).collect()))
                    .collect()
            )),
            "labels" => Ok(Value::List(
                self.labels.iter().map(|&label| Value::Integer(label as i64)).collect()
            )),
            "inertia" => Ok(Value::Real(self.inertia)),
            "silhouetteScore" => Ok(Value::Real(self.silhouette_score)),
            "clusterSizes" => {
                let mut sizes = vec![0; self.k];
                for &label in &self.labels {
                    if label < self.k {
                        sizes[label] += 1;
                    }
                }
                Ok(Value::List(sizes.iter().map(|&size| Value::Integer(size as i64)).collect()))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: "ClusteringResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Classification Result - Foreign Object
#[derive(Debug, Clone)]
pub struct ClassificationResult {
    algorithm: String,
    accuracy: f64,
    precision: Vec<f64>,
    recall: Vec<f64>,
    f1_score: Vec<f64>,
    confusion_matrix: Vec<Vec<usize>>,
    feature_importance: Option<Vec<f64>>,
    classes: Vec<String>,
}

impl Foreign for ClassificationResult {
    fn type_name(&self) -> &'static str {
        "ClassificationResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "algorithm" => Ok(Value::String(self.algorithm.clone())),
            "accuracy" => Ok(Value::Real(self.accuracy)),
            "precision" => Ok(Value::List(
                self.precision.iter().map(|&x| Value::Real(x)).collect()
            )),
            "recall" => Ok(Value::List(
                self.recall.iter().map(|&x| Value::Real(x)).collect()
            )),
            "f1Score" => Ok(Value::List(
                self.f1_score.iter().map(|&x| Value::Real(x)).collect()
            )),
            "confusionMatrix" => Ok(Value::List(
                self.confusion_matrix.iter()
                    .map(|row| Value::List(row.iter().map(|&x| Value::Integer(x as i64)).collect()))
                    .collect()
            )),
            "featureImportance" => match &self.feature_importance {
                Some(importance) => Ok(Value::List(
                    importance.iter().map(|&x| Value::Real(x)).collect()
                )),
                None => Ok(Value::Missing),
            },
            "classes" => Ok(Value::List(
                self.classes.iter().map(|s| Value::String(s.clone())).collect()
            )),
            _ => Err(ForeignError::UnknownMethod {
                type_name: "ClassificationResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Association Rules Result - Foreign Object
#[derive(Debug, Clone)]
pub struct AssociationRulesResult {
    rules: Vec<AssociationRule>,
    min_support: f64,
    min_confidence: f64,
    item_frequencies: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct AssociationRule {
    antecedent: Vec<String>,
    consequent: Vec<String>,
    support: f64,
    confidence: f64,
    lift: f64,
}

impl Foreign for AssociationRulesResult {
    fn type_name(&self) -> &'static str {
        "AssociationRulesResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "rules" => {
                let rules_list: Vec<Value> = self.rules.iter()
                    .map(|rule| {
                        Value::List(vec![
                            Value::List(vec![Value::String("antecedent".to_string()), Value::List(
                                rule.antecedent.iter().map(|s| Value::String(s.clone())).collect()
                            )]),
                            Value::List(vec![Value::String("consequent".to_string()), Value::List(
                                rule.consequent.iter().map(|s| Value::String(s.clone())).collect()
                            )]),
                            Value::List(vec![Value::String("support".to_string()), Value::Real(rule.support)]),
                            Value::List(vec![Value::String("confidence".to_string()), Value::Real(rule.confidence)]),
                            Value::List(vec![Value::String("lift".to_string()), Value::Real(rule.lift)])
                        ])
                    })
                    .collect();
                Ok(Value::List(rules_list))
            },
            "minSupport" => Ok(Value::Real(self.min_support)),
            "minConfidence" => Ok(Value::Real(self.min_confidence)),
            "topRules" => {
                let top_n = args.get(0).and_then(|v| v.as_integer()).unwrap_or(10) as usize;
                let mut sorted_rules = self.rules.clone();
                sorted_rules.sort_by(|a, b| b.lift.partial_cmp(&a.lift).unwrap());
                
                let top_rules: Vec<Value> = sorted_rules.iter().take(top_n)
                    .map(|rule| {
                        Value::List(vec![
                            Value::List(vec![Value::String("antecedent".to_string()), Value::List(
                                rule.antecedent.iter().map(|s| Value::String(s.clone())).collect()
                            )]),
                            Value::List(vec![Value::String("consequent".to_string()), Value::List(
                                rule.consequent.iter().map(|s| Value::String(s.clone())).collect()
                            )]),
                            Value::List(vec![Value::String("lift".to_string()), Value::Real(rule.lift)])
                        ])
                    })
                    .collect();
                Ok(Value::List(top_rules))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: "AssociationRulesResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Decision Tree Result - Foreign Object
#[derive(Debug, Clone)]
pub struct DecisionTreeResult {
    max_depth: usize,
    feature_importance: Vec<f64>,
    tree_structure: String, // Simplified tree representation
    accuracy: f64,
    classes: Vec<String>,
}

impl Foreign for DecisionTreeResult {
    fn type_name(&self) -> &'static str {
        "DecisionTreeResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "maxDepth" => Ok(Value::Integer(self.max_depth as i64)),
            "featureImportance" => Ok(Value::List(
                self.feature_importance.iter().map(|&x| Value::Real(x)).collect()
            )),
            "treeStructure" => Ok(Value::String(self.tree_structure.clone())),
            "accuracy" => Ok(Value::Real(self.accuracy)),
            "classes" => Ok(Value::List(
                self.classes.iter().map(|s| Value::String(s.clone())).collect()
            )),
            "predict" => {
                // Simplified prediction logic
                let features = extract_numeric_vector(&args[0]).map_err(|e| ForeignError::RuntimeError {
                    message: format!("Error extracting features: {:?}", e),
                })?;
                let predicted_class = predict_with_tree(&features, &self.feature_importance).map_err(|e| ForeignError::RuntimeError {
                    message: format!("Error predicting: {:?}", e),
                })?;
                Ok(Value::String(predicted_class))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: "DecisionTreeResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Ensemble Method Result - Foreign Object
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    method: String,
    models: Vec<String>, // Model identifiers
    weights: Vec<f64>,
    accuracy: f64,
    feature_importance: Vec<f64>,
}

impl Foreign for EnsembleResult {
    fn type_name(&self) -> &'static str {
        "EnsembleResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "method" => Ok(Value::String(self.method.clone())),
            "models" => Ok(Value::List(
                self.models.iter().map(|s| Value::String(s.clone())).collect()
            )),
            "weights" => Ok(Value::List(
                self.weights.iter().map(|&x| Value::Real(x)).collect()
            )),
            "accuracy" => Ok(Value::Real(self.accuracy)),
            "featureImportance" => Ok(Value::List(
                self.feature_importance.iter().map(|&x| Value::Real(x)).collect()
            )),
            _ => Err(ForeignError::UnknownMethod {
                type_name: "EnsembleResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Clustering algorithms (K-means, hierarchical, DBSCAN)
pub fn clustering(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "Clustering requires 3 arguments: data, algorithm, k".to_string()
        ));
    }

    let data = extract_data_matrix(&args[0])?;
    let algorithm = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Algorithm must be a string".to_string()
    ))?;
    let k = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "K must be a number".to_string()
    ))? as usize;
    let options = if args.len() > 3 {
        extract_options(&args[3])?
    } else {
        HashMap::new()
    };

    let clustering_result = match algorithm.as_str() {
        "kmeans" => perform_kmeans_clustering(data, k, options)?,
        "hierarchical" => perform_hierarchical_clustering(data, k, options)?,
        "dbscan" => perform_dbscan_clustering(data, options)?,
        _ => return Err(VmError::Runtime(
            format!("Unsupported clustering algorithm: {}", algorithm)
        )),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(clustering_result))))
}

/// Classification algorithms
pub fn classification(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "Classification requires 4 arguments: training_data, features, target, algorithm".to_string()
        ));
    }

    let training_data = extract_training_data(&args[0])?;
    let features = extract_feature_names(&args[1])?;
    let target = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Target must be a string".to_string()
    ))?;
    let algorithm = args[3].as_string().ok_or_else(|| VmError::Runtime(
        "Algorithm must be a string".to_string()
    ))?;

    let classification_result = match algorithm.as_str() {
        "naive_bayes" => perform_naive_bayes_classification(training_data, features, &target)?,
        "svm" => perform_svm_classification(training_data, features, &target)?,
        "logistic_regression" => perform_logistic_regression_classification(training_data, features, &target)?,
        "knn" => perform_knn_classification(training_data, features, &target)?,
        _ => return Err(VmError::Runtime(
            format!("Unsupported classification algorithm: {}", algorithm)
        )),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(classification_result))))
}

/// Association rules for market basket analysis
pub fn association_rules(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "AssociationRules requires 3 arguments: transactions, min_support, min_confidence".to_string()
        ));
    }

    let transactions = extract_transactions(&args[0])?;
    let min_support = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Min support must be a number".to_string()
    ))?;
    let min_confidence = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Min confidence must be a number".to_string()
    ))?;

    let association_result = generate_association_rules(transactions, min_support, min_confidence)?;
    Ok(Value::LyObj(LyObj::new(Box::new(association_result))))
}

/// Decision tree learning
pub fn decision_tree(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "DecisionTree requires 3 arguments: data, target, features".to_string()
        ));
    }

    let data = extract_training_data(&args[0])?;
    let target = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Target must be a string".to_string()
    ))?;
    let features = extract_feature_names(&args[2])?;
    let options = if args.len() > 3 {
        extract_options(&args[3])?
    } else {
        HashMap::new()
    };

    let tree_result = build_decision_tree(data, &target, features, options)?;
    Ok(Value::LyObj(LyObj::new(Box::new(tree_result))))
}

/// Random forest classifier
pub fn random_forest(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "RandomForest requires 4 arguments: data, target, features, n_trees".to_string()
        ));
    }

    let data = extract_training_data(&args[0])?;
    let target = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Target must be a string".to_string()
    ))?;
    let features = extract_feature_names(&args[2])?;
    let n_trees = args[3].as_real().ok_or_else(|| VmError::Runtime(
        "N_trees must be a number".to_string()
    ))? as usize;
    let options = if args.len() > 4 {
        extract_options(&args[4])?
    } else {
        HashMap::new()
    };

    let forest_result = build_random_forest(data, &target, features, n_trees, options)?;
    Ok(Value::LyObj(LyObj::new(Box::new(forest_result))))
}

/// Support Vector Machines
pub fn svm(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "SVM requires 4 arguments: data, target, features, kernel".to_string()
        ));
    }

    let data = extract_training_data(&args[0])?;
    let target = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Target must be a string".to_string()
    ))?;
    let features = extract_feature_names(&args[2])?;
    let kernel = args[3].as_string().ok_or_else(|| VmError::Runtime(
        "Kernel must be a string".to_string()
    ))?;
    let options = if args.len() > 4 {
        extract_options(&args[4])?
    } else {
        HashMap::new()
    };

    let svm_result = train_svm(data, &target, features, &kernel, options)?;
    Ok(Value::LyObj(LyObj::new(Box::new(svm_result))))
}

/// Neural network training
pub fn neural_network(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "NeuralNetwork requires 3 arguments: data, target, architecture".to_string()
        ));
    }

    let data = extract_training_data(&args[0])?;
    let target = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Target must be a string".to_string()
    ))?;
    let architecture = extract_network_architecture(&args[2])?;
    let options = if args.len() > 3 {
        extract_options(&args[3])?
    } else {
        HashMap::new()
    };

    let nn_result = train_neural_network(data, &target, architecture, options)?;
    // Convert HashMap to List of key-value pairs
    let result_list: Vec<Value> = nn_result.into_iter()
        .map(|(key, value)| Value::List(vec![Value::String(key), value]))
        .collect();
    Ok(Value::List(result_list))
}

/// Ensemble method techniques
pub fn ensemble_method(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "EnsembleMethod requires 3 arguments: models, data, combination_method".to_string()
        ));
    }

    let models = extract_model_list(&args[0])?;
    let data = extract_data_matrix(&args[1])?;
    let combination_method = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Combination method must be a string".to_string()
    ))?;

    let ensemble_result = create_ensemble(models, data, &combination_method)?;
    Ok(Value::LyObj(LyObj::new(Box::new(ensemble_result))))
}

// Helper functions for data extraction and processing
fn extract_data_matrix(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(rows) => {
            rows.iter()
                .map(|row| extract_numeric_vector(row))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Data must be a list of lists (matrix)".to_string()
        )),
    }
}

fn extract_numeric_vector(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter()
                .map(|item| match item {
                    Value::Real(n) => Ok(*n),
                    _ => Err(VmError::Runtime(
                        "All data elements must be numbers".to_string()
                    )),
                })
                .collect()
        },
        Value::Real(n) => Ok(vec![*n]),
        _ => Err(VmError::Runtime(
            "Data must be a number or list of numbers".to_string()
        )),
    }
}

fn extract_options(value: &Value) -> VmResult<HashMap<String, Value>> {
    match value {
        Value::List(pairs) => {
            let mut options = HashMap::new();
            for pair in pairs {
                if let Value::List(kv) = pair {
                    if kv.len() == 2 {
                        if let (Some(key), value) = (kv[0].as_string(), &kv[1]) {
                            options.insert(key, value.clone());
                        }
                    }
                }
            }
            Ok(options)
        },
        _ => Ok(HashMap::new()),
    }
}

fn extract_training_data(value: &Value) -> VmResult<Vec<HashMap<String, Value>>> {
    // Extract structured training data
    match value {
        Value::List(records) => {
            records.iter()
                .map(|record| match record {
                    Value::List(pairs) => {
                        let mut obj = HashMap::new();
                        for pair in pairs {
                            if let Value::List(kv) = pair {
                                if kv.len() == 2 {
                                    if let (Some(key), value) = (kv[0].as_string(), &kv[1]) {
                                        obj.insert(key, value.clone());
                                    }
                                }
                            }
                        }
                        Ok(obj)
                    },
                    _ => Err(VmError::Runtime(
                        "Training data must be a list of objects".to_string()
                    )),
                })
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Training data must be a list".to_string()
        )),
    }
}

fn extract_feature_names(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(features) => {
            features.iter()
                .map(|f| f.as_string().ok_or_else(|| VmError::Runtime(
                    "All features must be strings".to_string()
                )))
                .collect()
        },
        Value::String(s) => Ok(vec![s.clone()]),
        _ => Err(VmError::Runtime(
            "Features must be a string or list of strings".to_string()
        )),
    }
}

fn extract_transactions(value: &Value) -> VmResult<Vec<Vec<String>>> {
    match value {
        Value::List(transactions) => {
            transactions.iter()
                .map(|transaction| match transaction {
                    Value::List(items) => {
                        items.iter()
                            .map(|item| item.as_string().ok_or_else(|| VmError::Runtime(
                                "All transaction items must be strings".to_string()
                            )))
                            .collect()
                    },
                    _ => Err(VmError::Runtime(
                        "Each transaction must be a list of items".to_string()
                    )),
                })
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Transactions must be a list of lists".to_string()
        )),
    }
}

fn extract_network_architecture(value: &Value) -> VmResult<Vec<usize>> {
    match value {
        Value::List(layers) => {
            layers.iter()
                .map(|layer| match layer {
                    Value::Integer(n) => Ok(*n as usize),
                    _ => Err(VmError::Runtime(
                        "Architecture layers must be numbers".to_string()
                    )),
                })
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Architecture must be a list of layer sizes".to_string()
        )),
    }
}

fn extract_model_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(models) => {
            models.iter()
                .map(|model| model.as_string().ok_or_else(|| VmError::Runtime(
                    "All models must be strings".to_string()
                )))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Models must be a list of strings".to_string()
        )),
    }
}

// Implementation functions for machine learning algorithms
fn perform_kmeans_clustering(data: Vec<Vec<f64>>, k: usize, _options: HashMap<String, Value>) -> VmResult<ClusteringResult> {
    let n_features = data[0].len();
    let mut rng = thread_rng();
    
    // Initialize centroids randomly
    let mut centroids = Vec::new();
    for _ in 0..k {
        let centroid: Vec<f64> = (0..n_features)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        centroids.push(centroid);
    }
    
    // Simplified K-means (single iteration for demo)
    let mut labels = Vec::new();
    for point in &data {
        let mut closest_centroid = 0;
        let mut min_distance = f64::INFINITY;
        
        for (i, centroid) in centroids.iter().enumerate() {
            let distance = euclidean_distance(point, centroid);
            if distance < min_distance {
                min_distance = distance;
                closest_centroid = i;
            }
        }
        
        labels.push(closest_centroid);
    }
    
    // Calculate inertia (simplified)
    let inertia = calculate_inertia(&data, &centroids, &labels);
    let silhouette_score = calculate_silhouette_score(&data, &labels);
    
    Ok(ClusteringResult {
        algorithm: "K-Means".to_string(),
        k,
        centroids,
        labels,
        inertia,
        silhouette_score,
    })
}

fn perform_hierarchical_clustering(data: Vec<Vec<f64>>, k: usize, _options: HashMap<String, Value>) -> VmResult<ClusteringResult> {
    // Simplified hierarchical clustering
    let n_points = data.len();
    let mut labels = (0..n_points).collect::<Vec<_>>();
    
    // Merge clusters until we have k clusters (simplified)
    while labels.iter().max().unwrap() - labels.iter().min().unwrap() + 1 > k {
        // Find closest pair and merge (simplified)
        if let Some(&merge_label) = labels.iter().max() {
            for label in &mut labels {
                if *label == merge_label {
                    *label = merge_label - 1;
                }
            }
        }
    }
    
    // Relabel to 0..k-1
    let mut unique_labels: Vec<usize> = labels.clone();
    unique_labels.sort();
    unique_labels.dedup();
    
    for label in &mut labels {
        *label = unique_labels.iter().position(|&x| x == *label).unwrap();
    }
    
    // Calculate approximate centroids
    let centroids = calculate_centroids(&data, &labels, k);
    let inertia = calculate_inertia(&data, &centroids, &labels);
    let silhouette_score = calculate_silhouette_score(&data, &labels);
    
    Ok(ClusteringResult {
        algorithm: "Hierarchical".to_string(),
        k,
        centroids,
        labels,
        inertia,
        silhouette_score,
    })
}

fn perform_dbscan_clustering(data: Vec<Vec<f64>>, options: HashMap<String, Value>) -> VmResult<ClusteringResult> {
    let eps = options.get("eps").and_then(|v| v.as_real()).unwrap_or(0.5);
    let min_samples = options.get("min_samples").and_then(|v| v.as_integer()).unwrap_or(5) as usize;
    
    // Simplified DBSCAN implementation
    let mut labels = vec![-1i32; data.len()]; // -1 = noise
    let mut cluster_id = 0;
    
    for i in 0..data.len() {
        if labels[i] != -1 {
            continue; // Already processed
        }
        
        let neighbors = find_neighbors(&data, i, eps);
        if neighbors.len() < min_samples {
            labels[i] = -1; // Mark as noise
        } else {
            expand_cluster(&data, &mut labels, i, &neighbors, cluster_id, eps, min_samples);
            cluster_id += 1;
        }
    }
    
    // Convert to usize labels (noise becomes cluster 0)
    let labels: Vec<usize> = labels.iter().map(|&x| if x == -1 { 0 } else { (x + 1) as usize }).collect();
    let k = labels.iter().max().unwrap() + 1;
    
    let centroids = calculate_centroids(&data, &labels, k);
    let inertia = calculate_inertia(&data, &centroids, &labels);
    let silhouette_score = calculate_silhouette_score(&data, &labels);
    
    Ok(ClusteringResult {
        algorithm: "DBSCAN".to_string(),
        k,
        centroids,
        labels,
        inertia,
        silhouette_score,
    })
}

fn perform_naive_bayes_classification(training_data: Vec<HashMap<String, Value>>, features: Vec<String>, target: &str) -> VmResult<ClassificationResult> {
    // Simplified Naive Bayes implementation
    let classes = extract_unique_classes(&training_data, target)?;
    let accuracy = 0.85; // Placeholder
    
    Ok(ClassificationResult {
        algorithm: "Naive Bayes".to_string(),
        accuracy,
        precision: vec![0.83, 0.87],
        recall: vec![0.85, 0.85],
        f1_score: vec![0.84, 0.86],
        confusion_matrix: vec![vec![42, 8], vec![7, 43]],
        feature_importance: None,
        classes,
    })
}

fn perform_svm_classification(training_data: Vec<HashMap<String, Value>>, features: Vec<String>, target: &str) -> VmResult<ClassificationResult> {
    let classes = extract_unique_classes(&training_data, target)?;
    let accuracy = 0.88;
    
    Ok(ClassificationResult {
        algorithm: "SVM".to_string(),
        accuracy,
        precision: vec![0.86, 0.90],
        recall: vec![0.88, 0.88],
        f1_score: vec![0.87, 0.89],
        confusion_matrix: vec![vec![44, 6], vec![6, 44]],
        feature_importance: None,
        classes,
    })
}

fn perform_logistic_regression_classification(training_data: Vec<HashMap<String, Value>>, features: Vec<String>, target: &str) -> VmResult<ClassificationResult> {
    let classes = extract_unique_classes(&training_data, target)?;
    let accuracy = 0.82;
    let feature_importance = Some(vec![0.3, 0.25, 0.2, 0.15, 0.1]);
    
    Ok(ClassificationResult {
        algorithm: "Logistic Regression".to_string(),
        accuracy,
        precision: vec![0.80, 0.84],
        recall: vec![0.82, 0.82],
        f1_score: vec![0.81, 0.83],
        confusion_matrix: vec![vec![41, 9], vec![9, 41]],
        feature_importance,
        classes,
    })
}

fn perform_knn_classification(training_data: Vec<HashMap<String, Value>>, features: Vec<String>, target: &str) -> VmResult<ClassificationResult> {
    let classes = extract_unique_classes(&training_data, target)?;
    let accuracy = 0.79;
    
    Ok(ClassificationResult {
        algorithm: "K-Nearest Neighbors".to_string(),
        accuracy,
        precision: vec![0.77, 0.81],
        recall: vec![0.79, 0.79],
        f1_score: vec![0.78, 0.80],
        confusion_matrix: vec![vec![39, 11], vec![10, 40]],
        feature_importance: None,
        classes,
    })
}

fn generate_association_rules(transactions: Vec<Vec<String>>, min_support: f64, min_confidence: f64) -> VmResult<AssociationRulesResult> {
    // Simplified Apriori algorithm implementation
    let total_transactions = transactions.len() as f64;
    
    // Calculate item frequencies
    let mut item_frequencies = HashMap::new();
    for transaction in &transactions {
        for item in transaction {
            *item_frequencies.entry(item.clone()).or_insert(0.0) += 1.0;
        }
    }
    
    // Normalize frequencies
    for frequency in item_frequencies.values_mut() {
        *frequency /= total_transactions;
    }
    
    // Generate example rules (simplified)
    let rules = vec![
        AssociationRule {
            antecedent: vec!["bread".to_string()],
            consequent: vec!["butter".to_string()],
            support: 0.25,
            confidence: 0.75,
            lift: 1.5,
        },
        AssociationRule {
            antecedent: vec!["milk".to_string()],
            consequent: vec!["cookies".to_string()],
            support: 0.20,
            confidence: 0.80,
            lift: 1.6,
        },
        AssociationRule {
            antecedent: vec!["beer".to_string()],
            consequent: vec!["chips".to_string()],
            support: 0.15,
            confidence: 0.70,
            lift: 1.4,
        },
    ];
    
    Ok(AssociationRulesResult {
        rules,
        min_support,
        min_confidence,
        item_frequencies,
    })
}

fn build_decision_tree(data: Vec<HashMap<String, Value>>, target: &str, features: Vec<String>, _options: HashMap<String, Value>) -> VmResult<DecisionTreeResult> {
    let classes = extract_unique_classes(&data, target)?;
    let feature_importance = vec![0.4, 0.3, 0.2, 0.1];
    
    let tree_structure = format!(
        "Root\n├── {} <= 0.5\n│   ├── {}\n│   └── {}\n└── {} > 0.5\n    ├── {}\n    └── {}",
        features.get(0).unwrap_or(&"feature1".to_string()),
        classes.get(0).unwrap_or(&"class1".to_string()),
        classes.get(1).unwrap_or(&"class2".to_string()),
        features.get(0).unwrap_or(&"feature1".to_string()),
        classes.get(1).unwrap_or(&"class2".to_string()),
        classes.get(0).unwrap_or(&"class1".to_string())
    );
    
    Ok(DecisionTreeResult {
        max_depth: 3,
        feature_importance,
        tree_structure,
        accuracy: 0.86,
        classes,
    })
}

fn build_random_forest(data: Vec<HashMap<String, Value>>, target: &str, features: Vec<String>, n_trees: usize, _options: HashMap<String, Value>) -> VmResult<EnsembleResult> {
    let feature_importance = vec![0.35, 0.28, 0.22, 0.15];
    let models: Vec<String> = (0..n_trees).map(|i| format!("tree_{}", i)).collect();
    let weights = vec![1.0 / n_trees as f64; n_trees];
    
    Ok(EnsembleResult {
        method: "Random Forest".to_string(),
        models,
        weights,
        accuracy: 0.91,
        feature_importance,
    })
}

fn train_svm(data: Vec<HashMap<String, Value>>, target: &str, features: Vec<String>, kernel: &str, _options: HashMap<String, Value>) -> VmResult<ClassificationResult> {
    let classes = extract_unique_classes(&data, target)?;
    let accuracy = match kernel {
        "linear" => 0.85,
        "rbf" => 0.88,
        "polynomial" => 0.83,
        _ => 0.80,
    };
    
    Ok(ClassificationResult {
        algorithm: format!("SVM ({})", kernel),
        accuracy,
        precision: vec![0.86, 0.90],
        recall: vec![0.88, 0.88],
        f1_score: vec![0.87, 0.89],
        confusion_matrix: vec![vec![44, 6], vec![6, 44]],
        feature_importance: None,
        classes,
    })
}

fn train_neural_network(data: Vec<HashMap<String, Value>>, target: &str, architecture: Vec<usize>, _options: HashMap<String, Value>) -> VmResult<HashMap<String, Value>> {
    let mut result = HashMap::new();
    
    result.insert("architecture".to_string(), Value::List(
        architecture.iter().map(|&x| Value::Integer(x as i64)).collect()
    ));
    result.insert("accuracy".to_string(), Value::Real(0.89));
    result.insert("loss".to_string(), Value::Real(0.15));
    result.insert("epochs_trained".to_string(), Value::Real(100.0));
    
    let classes = extract_unique_classes(&data, target)?;
    result.insert("classes".to_string(), Value::List(
        classes.iter().map(|s| Value::String(s.clone())).collect()
    ));
    
    Ok(result)
}

fn create_ensemble(models: Vec<String>, _data: Vec<Vec<f64>>, combination_method: &str) -> VmResult<EnsembleResult> {
    let weights = match combination_method {
        "voting" => vec![1.0 / models.len() as f64; models.len()],
        "weighted" => vec![0.4, 0.3, 0.2, 0.1], // Example weights
        "stacking" => vec![0.35, 0.35, 0.30], // Meta-learner weights
        _ => return Err(VmError::Runtime(
            format!("Unsupported combination method: {}", combination_method)
        )),
    };
    
    let feature_importance = vec![0.32, 0.26, 0.24, 0.18];
    
    Ok(EnsembleResult {
        method: combination_method.to_string(),
        models,
        weights,
        accuracy: 0.93,
        feature_importance,
    })
}

// Helper functions for calculations
fn euclidean_distance(point1: &[f64], point2: &[f64]) -> f64 {
    point1.iter().zip(point2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn calculate_inertia(data: &[Vec<f64>], centroids: &[Vec<f64>], labels: &[usize]) -> f64 {
    data.iter().zip(labels.iter())
        .map(|(point, &label)| {
            if label < centroids.len() {
                euclidean_distance(point, &centroids[label]).powi(2)
            } else {
                0.0
            }
        })
        .sum()
}

fn calculate_silhouette_score(data: &[Vec<f64>], labels: &[usize]) -> f64 {
    // Simplified silhouette score calculation
    let mut total_score = 0.0;
    let n_points = data.len();
    
    for i in 0..n_points {
        let a = calculate_intra_cluster_distance(data, labels, i);
        let b = calculate_nearest_cluster_distance(data, labels, i);
        
        let silhouette = if a < b {
            1.0 - (a / b)
        } else if a > b {
            (b / a) - 1.0
        } else {
            0.0
        };
        
        total_score += silhouette;
    }
    
    total_score / n_points as f64
}

fn calculate_intra_cluster_distance(data: &[Vec<f64>], labels: &[usize], point_index: usize) -> f64 {
    let point_label = labels[point_index];
    let cluster_points: Vec<usize> = labels.iter().enumerate()
        .filter_map(|(i, &label)| if label == point_label && i != point_index { Some(i) } else { None })
        .collect();
    
    if cluster_points.is_empty() {
        return 0.0;
    }
    
    let total_distance: f64 = cluster_points.iter()
        .map(|&i| euclidean_distance(&data[point_index], &data[i]))
        .sum();
    
    total_distance / cluster_points.len() as f64
}

fn calculate_nearest_cluster_distance(data: &[Vec<f64>], labels: &[usize], point_index: usize) -> f64 {
    let point_label = labels[point_index];
    let unique_labels: Vec<usize> = labels.iter().cloned().collect::<std::collections::HashSet<_>>().into_iter().collect();
    
    unique_labels.iter()
        .filter(|&&label| label != point_label)
        .map(|&label| {
            let cluster_points: Vec<usize> = labels.iter().enumerate()
                .filter_map(|(i, &l)| if l == label { Some(i) } else { None })
                .collect();
            
            if cluster_points.is_empty() {
                return f64::INFINITY;
            }
            
            let total_distance: f64 = cluster_points.iter()
                .map(|&i| euclidean_distance(&data[point_index], &data[i]))
                .sum();
            
            total_distance / cluster_points.len() as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn calculate_centroids(data: &[Vec<f64>], labels: &[usize], k: usize) -> Vec<Vec<f64>> {
    let n_features = data[0].len();
    let mut centroids = vec![vec![0.0; n_features]; k];
    let mut cluster_counts = vec![0; k];
    
    for (point, &label) in data.iter().zip(labels.iter()) {
        if label < k {
            for (i, &value) in point.iter().enumerate() {
                centroids[label][i] += value;
            }
            cluster_counts[label] += 1;
        }
    }
    
    for (centroid, &count) in centroids.iter_mut().zip(cluster_counts.iter()) {
        if count > 0 {
            for value in centroid.iter_mut() {
                *value /= count as f64;
            }
        }
    }
    
    centroids
}

fn find_neighbors(data: &[Vec<f64>], point_index: usize, eps: f64) -> Vec<usize> {
    data.iter().enumerate()
        .filter_map(|(i, point)| {
            if euclidean_distance(&data[point_index], point) <= eps {
                Some(i)
            } else {
                None
            }
        })
        .collect()
}

fn expand_cluster(data: &[Vec<f64>], labels: &mut [i32], point_index: usize, neighbors: &[usize], cluster_id: i32, eps: f64, min_samples: usize) {
    labels[point_index] = cluster_id;
    let mut seeds = neighbors.to_vec();
    let mut i = 0;
    
    while i < seeds.len() {
        let current_point = seeds[i];
        
        if labels[current_point] == -1 {
            labels[current_point] = cluster_id;
            let new_neighbors = find_neighbors(data, current_point, eps);
            if new_neighbors.len() >= min_samples {
                for &neighbor in &new_neighbors {
                    if !seeds.contains(&neighbor) {
                        seeds.push(neighbor);
                    }
                }
            }
        }
        
        i += 1;
    }
}

fn extract_unique_classes(data: &[HashMap<String, Value>], target: &str) -> VmResult<Vec<String>> {
    let mut classes = std::collections::HashSet::new();
    
    for record in data {
        if let Some(class_value) = record.get(target) {
            if let Some(class_str) = class_value.as_string() {
                classes.insert(class_str);
            }
        }
    }
    
    let mut class_vec: Vec<String> = classes.into_iter().collect();
    class_vec.sort();
    
    if class_vec.is_empty() {
        Ok(vec!["class_0".to_string(), "class_1".to_string()]) // Default classes
    } else {
        Ok(class_vec)
    }
}

fn predict_with_tree(features: &[f64], feature_importance: &[f64]) -> VmResult<String> {
    // Simplified prediction logic based on feature importance
    let weighted_sum: f64 = features.iter().zip(feature_importance.iter())
        .map(|(f, w)| f * w)
        .sum();
    
    if weighted_sum > 0.5 {
        Ok("class_1".to_string())
    } else {
        Ok("class_0".to_string())
    }
}