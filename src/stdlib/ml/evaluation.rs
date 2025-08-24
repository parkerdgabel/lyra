//! Model Evaluation Infrastructure
//!
//! This module provides comprehensive model evaluation capabilities including
//! train/validation splits, evaluation metrics, cross-validation, and model selection.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::{NetChain, NetTrain};
use crate::stdlib::ml::training::TrainingConfig;
use crate::stdlib::ml::dataloader::{DataLoader, DataLoaderConfig};
use crate::stdlib::ml::automl::{ProblemType, ValidationStrategy};
use crate::stdlib::data::{ForeignDataset, ForeignTable};
use crate::vm::Value;
use crate::stdlib::common::assoc;
use std::collections::HashMap;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;

/// Train/Validation split functionality
pub struct DataSplitter;

impl DataSplitter {
    /// Split dataset into train and validation sets
    pub fn train_test_split(
        dataset: &ForeignDataset,
        test_size: f64,
        shuffle: bool,
        _stratify: bool,
    ) -> MLResult<(ForeignDataset, ForeignDataset)> {
        if !(0.0..=1.0).contains(&test_size) {
            return Err(MLError::DataError {
                reason: "test_size must be between 0.0 and 1.0".to_string(),
            });
        }
        
        let data_value = dataset.get_value();
        match data_value {
            Value::List(elements) => {
                let total_samples = elements.len();
                let test_samples = (total_samples as f64 * test_size) as usize;
                
                let mut indices: Vec<usize> = (0..total_samples).collect();
                
                if shuffle {
                    let mut rng = thread_rng();
                    indices.shuffle(&mut rng);
                }
                
                // Simple split (stratification not implemented for now)
                let (test_indices, train_indices) = indices.split_at(test_samples);
                
                // Create training dataset
                let train_elements: Vec<Value> = train_indices.iter()
                    .map(|&i| elements[i].clone())
                    .collect();
                
                // Create test dataset
                let test_elements: Vec<Value> = test_indices.iter()
                    .map(|&i| elements[i].clone())
                    .collect();
                
                let train_dataset = ForeignDataset::new(Value::List(train_elements));
                let test_dataset = ForeignDataset::new(Value::List(test_elements));
                
                Ok((train_dataset, test_dataset))
            },
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data for splitting".to_string(),
            }),
        }
    }
    
    /// Split table into train and validation sets
    pub fn train_test_split_table(
        table: &ForeignTable,
        test_size: f64,
        shuffle: bool,
        _stratify: bool,
    ) -> MLResult<(ForeignTable, ForeignTable)> {
        if !(0.0..=1.0).contains(&test_size) {
            return Err(MLError::DataError {
                reason: "test_size must be between 0.0 and 1.0".to_string(),
            });
        }
        
        let total_samples = table.length;
        let test_samples = (total_samples as f64 * test_size) as usize;
        
        let mut indices: Vec<usize> = (0..total_samples).collect();
        
        if shuffle {
            let mut rng = thread_rng();
            indices.shuffle(&mut rng);
        }
        
        let (_test_indices, _train_indices) = indices.split_at(test_samples);
        
        // Create subset tables (simplified implementation)
        // In practice, this would create actual table subsets
        let train_table = table.clone(); // Placeholder
        let test_table = table.clone();  // Placeholder
        
        Ok((train_table, test_table))
    }
    
    /// Create k-fold cross-validation splits
    pub fn k_fold_split(
        dataset: &ForeignDataset,
        k: usize,
        shuffle: bool,
    ) -> MLResult<Vec<(ForeignDataset, ForeignDataset)>> {
        if k < 2 {
            return Err(MLError::DataError {
                reason: "k must be at least 2 for cross-validation".to_string(),
            });
        }
        
        let data_value = dataset.get_value();
        match data_value {
            Value::List(elements) => {
                let total_samples = elements.len();
                let fold_size = total_samples / k;
                
                let mut indices: Vec<usize> = (0..total_samples).collect();
                
                if shuffle {
                    let mut rng = thread_rng();
                    indices.shuffle(&mut rng);
                }
                
                let mut folds = Vec::new();
                
                for fold_idx in 0..k {
                    let start_idx = fold_idx * fold_size;
                    let end_idx = if fold_idx == k - 1 {
                        total_samples // Include remaining samples in last fold
                    } else {
                        start_idx + fold_size
                    };
                    
                    // Validation indices for this fold
                    let val_indices = &indices[start_idx..end_idx];
                    
                    // Training indices (all except validation)
                    let train_indices: Vec<usize> = indices.iter()
                        .filter(|&&i| !val_indices.contains(&i))
                        .cloned()
                        .collect();
                    
                    // Create datasets
                    let train_elements: Vec<Value> = train_indices.iter()
                        .map(|&i| elements[i].clone())
                        .collect();
                    
                    let val_elements: Vec<Value> = val_indices.iter()
                        .map(|&i| elements[i].clone())
                        .collect();
                    
                    let train_dataset = ForeignDataset::new(Value::List(train_elements));
                    let val_dataset = ForeignDataset::new(Value::List(val_elements));
                    
                    folds.push((train_dataset, val_dataset));
                }
                
                Ok(folds)
            },
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data for k-fold splitting".to_string(),
            }),
        }
    }
}

/// Comprehensive evaluation metrics for classification and regression
pub struct EvaluationMetrics;

impl EvaluationMetrics {
    /// Calculate classification accuracy
    pub fn accuracy(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        if y_true.shape != y_pred.shape {
            return Err(MLError::DataError {
                reason: format!("Shape mismatch: y_true {:?} vs y_pred {:?}", y_true.shape, y_pred.shape),
            });
        }
        
        let correct_predictions = y_true.data.iter()
            .zip(y_pred.data.iter())
            .filter(|(true_val, pred_val)| {
                let true_class = if true_val.value() > 0.5 { 1 } else { 0 };
                let pred_class = if pred_val.value() > 0.5 { 1 } else { 0 };
                true_class == pred_class
            })
            .count();
        
        let total_predictions = y_true.data.len();
        Ok(correct_predictions as f64 / total_predictions as f64)
    }
    
    /// Calculate precision for binary classification
    pub fn precision(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        let (tp, fp, _tn, _fn) = Self::confusion_matrix_components(y_true, y_pred)?;
        
        if tp + fp == 0 {
            Ok(0.0) // No positive predictions
        } else {
            Ok(tp as f64 / (tp + fp) as f64)
        }
    }
    
    /// Calculate recall for binary classification
    pub fn recall(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        let (tp, _fp, _tn, fn_val) = Self::confusion_matrix_components(y_true, y_pred)?;
        
        if tp + fn_val == 0 {
            Ok(0.0) // No actual positives
        } else {
            Ok(tp as f64 / (tp + fn_val) as f64)
        }
    }
    
    /// Calculate F1 score (harmonic mean of precision and recall)
    pub fn f1_score(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        let precision = Self::precision(y_true, y_pred)?;
        let recall = Self::recall(y_true, y_pred)?;
        
        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }
    
    /// Calculate Mean Squared Error for regression
    pub fn mean_squared_error(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        if y_true.shape != y_pred.shape {
            return Err(MLError::DataError {
                reason: format!("Shape mismatch: y_true {:?} vs y_pred {:?}", y_true.shape, y_pred.shape),
            });
        }
        
        let mse = y_true.data.iter()
            .zip(y_pred.data.iter())
            .map(|(true_val, pred_val)| {
                let diff = true_val.value() - pred_val.value();
                diff * diff
            })
            .sum::<f64>() / y_true.data.len() as f64;
        
        Ok(mse)
    }
    
    /// Calculate Mean Absolute Error for regression
    pub fn mean_absolute_error(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        if y_true.shape != y_pred.shape {
            return Err(MLError::DataError {
                reason: format!("Shape mismatch: y_true {:?} vs y_pred {:?}", y_true.shape, y_pred.shape),
            });
        }
        
        let mae = y_true.data.iter()
            .zip(y_pred.data.iter())
            .map(|(true_val, pred_val)| {
                (true_val.value() - pred_val.value()).abs()
            })
            .sum::<f64>() / y_true.data.len() as f64;
        
        Ok(mae)
    }
    
    /// Calculate R-squared (coefficient of determination) for regression
    pub fn r_squared(y_true: &Tensor, y_pred: &Tensor) -> MLResult<f64> {
        if y_true.shape != y_pred.shape {
            return Err(MLError::DataError {
                reason: format!("Shape mismatch: y_true {:?} vs y_pred {:?}", y_true.shape, y_pred.shape),
            });
        }
        
        // Calculate mean of true values
        let y_mean = y_true.data.iter().map(|v| v.value()).sum::<f64>() / y_true.data.len() as f64;
        
        // Calculate sum of squares total and residual
        let ss_tot = y_true.data.iter()
            .map(|v| {
                let diff = v.value() - y_mean;
                diff * diff
            })
            .sum::<f64>();
        
        let ss_res = y_true.data.iter()
            .zip(y_pred.data.iter())
            .map(|(true_val, pred_val)| {
                let diff = true_val.value() - pred_val.value();
                diff * diff
            })
            .sum::<f64>();
        
        if ss_tot == 0.0 {
            Ok(1.0) // Perfect prediction when variance is zero
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
    
    /// Calculate confusion matrix components for binary classification
    fn confusion_matrix_components(y_true: &Tensor, y_pred: &Tensor) -> MLResult<(usize, usize, usize, usize)> {
        if y_true.shape != y_pred.shape {
            return Err(MLError::DataError {
                reason: format!("Shape mismatch: y_true {:?} vs y_pred {:?}", y_true.shape, y_pred.shape),
            });
        }
        
        let mut tp = 0; // True positives
        let mut fp = 0; // False positives
        let mut tn = 0; // True negatives
        let mut fn_val = 0; // False negatives
        
        for (true_val, pred_val) in y_true.data.iter().zip(y_pred.data.iter()) {
            let true_class = if true_val.value() > 0.5 { 1 } else { 0 };
            let pred_class = if pred_val.value() > 0.5 { 1 } else { 0 };
            
            match (true_class, pred_class) {
                (1, 1) => tp += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                (1, 0) => fn_val += 1,
                _ => unreachable!(),
            }
        }
        
        Ok((tp, fp, tn, fn_val))
    }
    
    /// Generate comprehensive classification report
    pub fn classification_report(y_true: &Tensor, y_pred: &Tensor) -> MLResult<ClassificationReport> {
        let accuracy = Self::accuracy(y_true, y_pred)?;
        let precision = Self::precision(y_true, y_pred)?;
        let recall = Self::recall(y_true, y_pred)?;
        let f1 = Self::f1_score(y_true, y_pred)?;
        let (tp, fp, tn, fn_val) = Self::confusion_matrix_components(y_true, y_pred)?;
        
        Ok(ClassificationReport {
            accuracy,
            precision,
            recall,
            f1_score: f1,
            confusion_matrix: ConfusionMatrix { tp, fp, tn, fn_val },
            support: y_true.data.len(),
        })
    }
    
    /// Generate comprehensive regression report
    pub fn regression_report(y_true: &Tensor, y_pred: &Tensor) -> MLResult<RegressionReport> {
        let mse = Self::mean_squared_error(y_true, y_pred)?;
        let mae = Self::mean_absolute_error(y_true, y_pred)?;
        let r2 = Self::r_squared(y_true, y_pred)?;
        let rmse = mse.sqrt();
        
        Ok(RegressionReport {
            mean_squared_error: mse,
            mean_absolute_error: mae,
            root_mean_squared_error: rmse,
            r_squared: r2,
            sample_count: y_true.data.len(),
        })
    }
}

/// Classification evaluation report
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    pub confusion_matrix: ConfusionMatrix,
    pub support: usize,
}

impl ClassificationReport {
    /// Generate human-readable report
    pub fn to_string(&self) -> String {
        format!(
            "Classification Report:\n\
            Accuracy:  {:.4}\n\
            Precision: {:.4}\n\
            Recall:    {:.4}\n\
            F1-Score:  {:.4}\n\
            Support:   {}\n\n\
            Confusion Matrix:\n\
            TP: {:<4} FP: {}\n\
            FN: {:<4} TN: {}",
            self.accuracy,
            self.precision,
            self.recall,
            self.f1_score,
            self.support,
            self.confusion_matrix.tp,
            self.confusion_matrix.fp,
            self.confusion_matrix.fn_val,
            self.confusion_matrix.tn
        )
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        assoc(vec![
            ("accuracy", Value::Real(self.accuracy)),
            ("precision", Value::Real(self.precision)),
            ("recall", Value::Real(self.recall)),
            ("f1Score", Value::Real(self.f1_score)),
            ("support", Value::Integer(self.support as i64)),
            ("confusionMatrix", assoc(vec![
                ("tp", Value::Integer(self.confusion_matrix.tp as i64)),
                ("fp", Value::Integer(self.confusion_matrix.fp as i64)),
                ("tn", Value::Integer(self.confusion_matrix.tn as i64)),
                ("fn", Value::Integer(self.confusion_matrix.fn_val as i64)),
            ])),
            ("reportType", Value::String("Classification".to_string())),
        ])
    }
}

/// Regression evaluation report
#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub mean_squared_error: f64,
    pub mean_absolute_error: f64,
    pub root_mean_squared_error: f64,
    pub r_squared: f64,
    pub sample_count: usize,
}

impl RegressionReport {
    /// Generate human-readable report
    pub fn to_string(&self) -> String {
        format!(
            "Regression Report:\n\
            MSE:    {:.6}\n\
            MAE:    {:.6}\n\
            RMSE:   {:.6}\n\
            R²:     {:.6}\n\
            Samples: {}",
            self.mean_squared_error,
            self.mean_absolute_error,
            self.root_mean_squared_error,
            self.r_squared,
            self.sample_count
        )
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        assoc(vec![
            ("meanSquaredError", Value::Real(self.mean_squared_error)),
            ("meanAbsoluteError", Value::Real(self.mean_absolute_error)),
            ("rootMeanSquaredError", Value::Real(self.root_mean_squared_error)),
            ("rSquared", Value::Real(self.r_squared)),
            ("sampleCount", Value::Integer(self.sample_count as i64)),
            ("reportType", Value::String("Regression".to_string())),
        ])
    }
}

/// Confusion matrix for binary classification
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    pub tp: usize, // True positives
    pub fp: usize, // False positives
    pub tn: usize, // True negatives
    pub fn_val: usize, // False negatives
}

impl ConfusionMatrix {
    /// Calculate specificity (true negative rate)
    pub fn specificity(&self) -> f64 {
        if self.tn + self.fp == 0 {
            0.0
        } else {
            self.tn as f64 / (self.tn + self.fp) as f64
        }
    }
    
    /// Calculate sensitivity (same as recall)
    pub fn sensitivity(&self) -> f64 {
        if self.tp + self.fn_val == 0 {
            0.0
        } else {
            self.tp as f64 / (self.tp + self.fn_val) as f64
        }
    }
}

/// Cross-validation framework
pub struct CrossValidator {
    k_folds: usize,
    shuffle: bool,
        stratify: bool,
    scoring_metric: ScoringMetric,
}

/// Scoring metrics for model selection
#[derive(Debug, Clone, Copy)]
pub enum ScoringMetric {
    Accuracy,
    F1Score,
    Precision,
    Recall,
    MeanSquaredError,
    MeanAbsoluteError,
    RSquared,
}

impl CrossValidator {
    /// Create new cross-validator
    pub fn new(k_folds: usize, scoring_metric: ScoringMetric) -> Self {
        Self {
            k_folds,
            shuffle: true,
            stratify: false,
            scoring_metric,
        }
    }
    
    /// Perform cross-validation on dataset
    pub fn cross_validate(
        &self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: impl Fn() -> MLResult<NetChain>,
        training_config: &TrainingConfig,
    ) -> MLResult<CrossValidationResult> {
        // Create k-fold splits
        let folds = DataSplitter::k_fold_split(dataset, self.k_folds, self.shuffle)?;
        
        let mut fold_scores = Vec::new();
        let mut fold_reports = Vec::new();
        
        for (fold_idx, (train_dataset, val_dataset)) in folds.iter().enumerate() {
            // Build new model for this fold
            let mut model = model_builder()?;
            
            // Train on training fold
            let trainer = NetTrain::with_config(training_config.clone());
            let _training_result = trainer.train_dataset_auto(&mut model, train_dataset, target_extraction)?;
            
            // Evaluate on validation fold
            let evaluation_result = self.evaluate_model_on_dataset(&model, val_dataset, target_extraction)?;
            
            // Extract score based on scoring metric
            let fold_score = evaluation_result.get_score(self.scoring_metric)?;
            fold_scores.push(fold_score);
            fold_reports.push(evaluation_result);
            
            println!("Fold {}: Score = {:.4}", fold_idx + 1, fold_score);
        }
        
        // Calculate statistics
        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores.iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>() / fold_scores.len() as f64;
        let std_dev = variance.sqrt();
        
        let best_fold_index = fold_scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
            
        Ok(CrossValidationResult {
            fold_scores,
            mean_score,
            std_deviation: std_dev,
            best_fold_index,
            fold_reports,
            scoring_metric: self.scoring_metric,
        })
    }
    
    /// Evaluate model on a dataset
    fn evaluate_model_on_dataset(
        &self,
        model: &NetChain,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
    ) -> MLResult<EvaluationResult> {
        // Create DataLoader for evaluation
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: false,
            drop_last: false,
            num_workers: 1,
            pin_memory: false,
        };
        
        let dataloader = DataLoader::from_dataset(dataset.clone(), target_extraction, config)?;
        
        let mut all_predictions = Vec::new();
        let mut all_targets = Vec::new();
        let mut model_clone = model.clone();
        
        // Collect predictions and targets
        for batch_idx in 0..dataloader.num_batches() {
            let batch = dataloader.get_batch(batch_idx)?;
            
            for (input_tensor, target_tensor) in batch {
                let prediction = model_clone.forward(&input_tensor)?;
                all_predictions.push(prediction);
                all_targets.push(target_tensor);
            }
        }
        
        // Combine all predictions and targets into single tensors
        let combined_predictions = Self::combine_tensors(&all_predictions)?;
        let combined_targets = Self::combine_tensors(&all_targets)?;
        
        // Generate evaluation result based on problem type
        let problem_type = self.infer_problem_type_from_targets(&combined_targets)?;
        
        match problem_type {
            ProblemType::BinaryClassification | ProblemType::MultiClassification => {
                let classification_report = EvaluationMetrics::classification_report(&combined_targets, &combined_predictions)?;
                Ok(EvaluationResult::Classification(classification_report))
            },
            ProblemType::Regression => {
                let regression_report = EvaluationMetrics::regression_report(&combined_targets, &combined_predictions)?;
                Ok(EvaluationResult::Regression(regression_report))
            },
            _ => Err(MLError::DataError {
                reason: "Cannot evaluate unknown problem type".to_string(),
            }),
        }
    }
    
    /// Combine multiple tensors into a single tensor
    fn combine_tensors(tensors: &[Tensor]) -> MLResult<Tensor> {
        if tensors.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot combine empty tensor list".to_string(),
            });
        }
        
        let total_elements: usize = tensors.iter().map(|t| t.data.len()).sum();
        let mut combined_data = Vec::with_capacity(total_elements);
        
        for tensor in tensors {
            combined_data.extend_from_slice(&tensor.data);
        }
        
        Tensor::new(combined_data, vec![total_elements])
    }
    
    /// Infer problem type from target tensor values
    fn infer_problem_type_from_targets(&self, targets: &Tensor) -> MLResult<ProblemType> {
        // Simple heuristic: check if all values are 0 or 1 (binary classification)
        let is_binary = targets.data.iter().all(|v| {
            let val = v.value();
            val == 0.0 || val == 1.0
        });
        
        if is_binary {
            Ok(ProblemType::BinaryClassification)
        } else {
            // Check if values seem to be discrete (classification) or continuous (regression)
            let unique_values: std::collections::HashSet<String> = targets.data.iter()
                .map(|v| format!("{:.3}", v.value()))
                .collect();
            
            if unique_values.len() <= 20 && unique_values.len() < targets.data.len() / 10 {
                Ok(ProblemType::MultiClassification)
            } else {
                Ok(ProblemType::Regression)
            }
        }
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub fold_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_deviation: f64,
    pub best_fold_index: usize,
    pub fold_reports: Vec<EvaluationResult>,
    pub scoring_metric: ScoringMetric,
}

impl CrossValidationResult {
    /// Generate human-readable cross-validation report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Cross-Validation Results ===\n\n");
        report.push_str(&format!("Scoring Metric: {:?}\n", self.scoring_metric));
        report.push_str(&format!("Folds: {}\n", self.fold_scores.len()));
        report.push_str(&format!("Mean Score: {:.4} (+/- {:.4})\n", self.mean_score, self.std_deviation * 2.0));
        report.push_str(&format!("Best Fold: {} (Score: {:.4})\n\n", self.best_fold_index + 1, self.fold_scores[self.best_fold_index]));
        
        report.push_str("Fold Scores:\n");
        for (i, score) in self.fold_scores.iter().enumerate() {
            report.push_str(&format!("  Fold {}: {:.4}\n", i + 1, score));
        }
        
        report
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        let scores: Vec<Value> = self.fold_scores.iter().map(|&x| Value::Real(x)).collect();
        let reports: Vec<Value> = self.fold_reports.iter().map(|r| r.to_value()).collect();
        assoc(vec![
            ("foldScores", Value::List(scores)),
            ("meanScore", Value::Real(self.mean_score)),
            ("stdDev", Value::Real(self.std_deviation)),
            ("bestFoldIndex", Value::Integer(self.best_fold_index as i64)),
            ("scoringMetric", Value::String(format!("{:?}", self.scoring_metric))),
            ("foldReports", Value::List(reports)),
        ])
    }
}

/// Evaluation result (classification or regression)
#[derive(Debug, Clone)]
pub enum EvaluationResult {
    Classification(ClassificationReport),
    Regression(RegressionReport),
}

impl EvaluationResult {
    /// Get score based on scoring metric
    pub fn get_score(&self, metric: ScoringMetric) -> MLResult<f64> {
        match (self, metric) {
            (EvaluationResult::Classification(report), ScoringMetric::Accuracy) => Ok(report.accuracy),
            (EvaluationResult::Classification(report), ScoringMetric::F1Score) => Ok(report.f1_score),
            (EvaluationResult::Classification(report), ScoringMetric::Precision) => Ok(report.precision),
            (EvaluationResult::Classification(report), ScoringMetric::Recall) => Ok(report.recall),
            (EvaluationResult::Regression(report), ScoringMetric::MeanSquaredError) => Ok(report.mean_squared_error),
            (EvaluationResult::Regression(report), ScoringMetric::MeanAbsoluteError) => Ok(report.mean_absolute_error),
            (EvaluationResult::Regression(report), ScoringMetric::RSquared) => Ok(report.r_squared),
            _ => Err(MLError::DataError {
                reason: format!("Scoring metric {:?} not compatible with evaluation type", metric),
            }),
        }
    }
    
    /// Generate report string
    pub fn to_string(&self) -> String {
        match self {
            EvaluationResult::Classification(report) => report.to_string(),
            EvaluationResult::Regression(report) => report.to_string(),
        }
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        match self {
            EvaluationResult::Classification(report) => report.to_value(),
            EvaluationResult::Regression(report) => report.to_value(),
        }
    }
}

/// Model selection and comparison framework
pub struct ModelSelector {
    models: Vec<ModelCandidate>,
    evaluation_strategy: ValidationStrategy,
    scoring_metric: ScoringMetric,
}

/// A candidate model for comparison
pub struct ModelCandidate {
    pub name: String,
    pub model_builder: Box<dyn Fn() -> MLResult<NetChain>>,
    pub training_config: TrainingConfig,
    pub evaluation_result: Option<EvaluationResult>,
    pub cross_validation_result: Option<CrossValidationResult>,
}

impl ModelSelector {
    /// Create new model selector
    pub fn new(evaluation_strategy: ValidationStrategy, scoring_metric: ScoringMetric) -> Self {
        Self {
            models: Vec::new(),
            evaluation_strategy,
            scoring_metric,
        }
    }
    
    /// Add model candidate for comparison
    pub fn add_model<F>(
        &mut self,
        name: String,
        model_builder: F,
        training_config: TrainingConfig,
    ) where
        F: Fn() -> MLResult<NetChain> + 'static,
    {
        self.models.push(ModelCandidate {
            name,
            model_builder: Box::new(model_builder),
            training_config,
            evaluation_result: None,
            cross_validation_result: None,
        });
    }
    
    /// Evaluate all models and select the best one
    pub fn select_best_model(
        &mut self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
    ) -> MLResult<ModelSelectionResult> {
        let mut model_results = Vec::new();
        
        for model_candidate in &mut self.models {
            println!("Evaluating model: {}", model_candidate.name);
            
            match self.evaluation_strategy {
                ValidationStrategy::HoldOut => {
                    // Simple train/test split
                    let (train_data, test_data) = DataSplitter::train_test_split(dataset, 0.2, true, false)?;
                    
                    // Train model
                    let mut model = (model_candidate.model_builder)()?;
                    let trainer = NetTrain::with_config(model_candidate.training_config.clone());
                    let _training_result = trainer.train_dataset_auto(&mut model, &train_data, target_extraction)?;
                    
                    // Evaluate on test set
                    let cross_validator = CrossValidator::new(1, self.scoring_metric);
                    let evaluation_result = cross_validator.evaluate_model_on_dataset(&model, &test_data, target_extraction)?;
                    
                    model_candidate.evaluation_result = Some(evaluation_result.clone());
                    
                    let score = evaluation_result.get_score(self.scoring_metric)?;
                    model_results.push((model_candidate.name.clone(), score, evaluation_result));
                },
                ValidationStrategy::CrossValidation => {
                    // K-fold cross-validation
                    let cross_validator = CrossValidator::new(5, self.scoring_metric);
                    let cv_result = cross_validator.cross_validate(
                        dataset,
                        target_extraction,
                        &model_candidate.model_builder,
                        &model_candidate.training_config,
                    )?;
                    
                    model_candidate.cross_validation_result = Some(cv_result.clone());
                    
                    let score = cv_result.mean_score;
                    let evaluation_result = EvaluationResult::Classification(ClassificationReport {
                        accuracy: score,
                        precision: 0.0,
                        recall: 0.0,
                        f1_score: 0.0,
                        confusion_matrix: ConfusionMatrix { tp: 0, fp: 0, tn: 0, fn_val: 0 },
                        support: 0,
                    });
                    
                    model_results.push((model_candidate.name.clone(), score, evaluation_result));
                },
                ValidationStrategy::TimeBasedSplit => {
                    // Simple split for now (would implement time-based logic)
                    let (train_data, test_data) = DataSplitter::train_test_split(dataset, 0.2, false, false)?;
                    
                    let mut model = (model_candidate.model_builder)()?;
                    let trainer = NetTrain::with_config(model_candidate.training_config.clone());
                    let _training_result = trainer.train_dataset_auto(&mut model, &train_data, target_extraction)?;
                    
                    let cross_validator = CrossValidator::new(1, self.scoring_metric);
                    let evaluation_result = cross_validator.evaluate_model_on_dataset(&model, &test_data, target_extraction)?;
                    
                    model_candidate.evaluation_result = Some(evaluation_result.clone());
                    
                    let score = evaluation_result.get_score(self.scoring_metric)?;
                    model_results.push((model_candidate.name.clone(), score, evaluation_result));
                },
            }
        }
        
        // Find best model
        let best_model = model_results.iter()
            .max_by(|(_, score_a, _), (_, score_b, _)| score_a.partial_cmp(score_b).unwrap())
            .ok_or_else(|| MLError::DataError {
                reason: "No models evaluated successfully".to_string(),
            })?;
        
        Ok(ModelSelectionResult {
            best_model_name: best_model.0.clone(),
            best_score: best_model.1,
            all_results: model_results,
            evaluation_strategy: self.evaluation_strategy,
            scoring_metric: self.scoring_metric,
        })
    }
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResult {
    pub best_model_name: String,
    pub best_score: f64,
    pub all_results: Vec<(String, f64, EvaluationResult)>,
    pub evaluation_strategy: ValidationStrategy,
    pub scoring_metric: ScoringMetric,
}

impl ModelSelectionResult {
    /// Generate model selection report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Model Selection Results ===\n\n");
        report.push_str(&format!("Evaluation Strategy: {:?}\n", self.evaluation_strategy));
        report.push_str(&format!("Scoring Metric: {:?}\n\n", self.scoring_metric));
        
        report.push_str(&format!("Best Model: {} (Score: {:.4})\n\n", self.best_model_name, self.best_score));
        
        report.push_str("All Models:\n");
        for (name, score, _result) in &self.all_results {
            report.push_str(&format!("  {}: {:.4}\n", name, score));
        }
        
        report
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        let results: Vec<Value> = self.all_results.iter().map(|(name, score, eval)| {
            assoc(vec![
                ("modelName", Value::String(name.clone())),
                ("score", Value::Real(*score)),
                ("report", eval.to_value()),
            ])
        }).collect();
        assoc(vec![
            ("bestModelName", Value::String(self.best_model_name.clone())),
            ("bestScore", Value::Real(self.best_score)),
            ("results", Value::List(results)),
            ("evaluationStrategy", Value::String(format!("{:?}", self.evaluation_strategy))),
            ("scoringMetric", Value::String(format!("{:?}", self.scoring_metric))),
        ])
    }
}

/// Hyperparameter optimization framework
pub struct HyperparameterOptimizer {
    search_strategy: SearchStrategy,
    parameter_space: ParameterSpace,
    optimization_budget: OptimizationBudget,
}

/// Search strategy for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    GridSearch,
    RandomSearch { iterations: usize },
    BayesianOptimization { iterations: usize },
}

/// Parameter space definition
#[derive(Debug, Clone)]
pub struct ParameterSpace {
    pub learning_rates: Vec<f64>,
    pub batch_sizes: Vec<usize>,
    pub epoch_ranges: (usize, usize),
    pub architecture_params: HashMap<String, Vec<Value>>,
}

impl Default for ParameterSpace {
    fn default() -> Self {
        Self {
            learning_rates: vec![0.0001, 0.001, 0.01, 0.1],
            batch_sizes: vec![8, 16, 32, 64, 128],
            epoch_ranges: (50, 500),
            architecture_params: HashMap::new(),
        }
    }
}

/// Optimization budget constraints
#[derive(Debug, Clone)]
pub struct OptimizationBudget {
    pub max_evaluations: usize,
    pub max_time_minutes: usize,
    pub early_stopping_patience: usize,
}

impl Default for OptimizationBudget {
    fn default() -> Self {
        Self {
            max_evaluations: 50,
            max_time_minutes: 120,
            early_stopping_patience: 10,
        }
    }
}

impl HyperparameterOptimizer {
    /// Create new hyperparameter optimizer
    pub fn new(search_strategy: SearchStrategy) -> Self {
        Self {
            search_strategy,
            parameter_space: ParameterSpace::default(),
            optimization_budget: OptimizationBudget::default(),
        }
    }
    
    /// Optimize hyperparameters for a given model and dataset
    pub fn optimize(
        &self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: impl Fn() -> MLResult<NetChain>,
        scoring_metric: ScoringMetric,
    ) -> MLResult<HyperparameterOptimizationResult> {
        match &self.search_strategy {
            SearchStrategy::GridSearch => {
                self.grid_search(dataset, target_extraction, model_builder, scoring_metric)
            },
            SearchStrategy::RandomSearch { iterations } => {
                self.random_search(dataset, target_extraction, model_builder, scoring_metric, *iterations)
            },
            SearchStrategy::BayesianOptimization { iterations } => {
                // Fallback to random search for now
                self.random_search(dataset, target_extraction, model_builder, scoring_metric, *iterations)
            },
        }
    }
    
    /// Grid search implementation
    fn grid_search(
        &self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: impl Fn() -> MLResult<NetChain>,
        scoring_metric: ScoringMetric,
    ) -> MLResult<HyperparameterOptimizationResult> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_config = TrainingConfig::default();
        let mut all_evaluations = Vec::new();
        
        // Simplified grid search over learning rates and batch sizes
        for &learning_rate in &self.parameter_space.learning_rates {
            for &batch_size in &self.parameter_space.batch_sizes {
                let config = TrainingConfig {
                    epochs: 100,
                    batch_size,
                    learning_rate,
                    print_progress: false,
                };
                
                // Evaluate this configuration
                let score = self.evaluate_config(dataset, target_extraction, &model_builder, &config, scoring_metric)?;
                
                all_evaluations.push((config.clone(), score));
                
                if score > best_score {
                    best_score = score;
                    best_config = config;
                }
                
                println!("Evaluated LR={:.4}, BS={}: Score={:.4}", learning_rate, batch_size, score);
            }
        }
        
        let total_evaluations = all_evaluations.len();
        Ok(HyperparameterOptimizationResult {
            best_config,
            best_score,
            all_evaluations,
            search_strategy: self.search_strategy.clone(),
            total_evaluations,
        })
    }
    
    /// Random search implementation
    fn random_search(
        &self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: impl Fn() -> MLResult<NetChain>,
        scoring_metric: ScoringMetric,
        iterations: usize,
    ) -> MLResult<HyperparameterOptimizationResult> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_config = TrainingConfig::default();
        let mut all_evaluations = Vec::new();
        
        use rand::Rng;
        let mut rng = thread_rng();
        
        for i in 0..iterations {
            // Sample random hyperparameters
            let learning_rate = self.parameter_space.learning_rates[
                rng.gen_range(0..self.parameter_space.learning_rates.len())
            ];
            let batch_size = self.parameter_space.batch_sizes[
                rng.gen_range(0..self.parameter_space.batch_sizes.len())
            ];
            let epochs = rng.gen_range(self.parameter_space.epoch_ranges.0..=self.parameter_space.epoch_ranges.1);
            
            let config = TrainingConfig {
                epochs,
                batch_size,
                learning_rate,
                print_progress: false,
            };
            
            // Evaluate this configuration
            let score = self.evaluate_config(dataset, target_extraction, &model_builder, &config, scoring_metric)?;
            
            all_evaluations.push((config.clone(), score));
            
            if score > best_score {
                best_score = score;
                best_config = config;
            }
            
            println!("Iteration {}: LR={:.4}, BS={}, E={}: Score={:.4}", i + 1, learning_rate, batch_size, epochs, score);
        }
        
        let total_evaluations = all_evaluations.len();
        Ok(HyperparameterOptimizationResult {
            best_config,
            best_score,
            all_evaluations,
            search_strategy: self.search_strategy.clone(),
            total_evaluations,
        })
    }
    
    /// Evaluate a specific hyperparameter configuration
    fn evaluate_config(
        &self,
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: &impl Fn() -> MLResult<NetChain>,
        config: &TrainingConfig,
        scoring_metric: ScoringMetric,
    ) -> MLResult<f64> {
        // Split data for evaluation
        let (train_data, val_data) = DataSplitter::train_test_split(dataset, 0.2, true, false)?;
        
        // Build and train model
        let mut model = model_builder()?;
        let trainer = NetTrain::with_config(config.clone());
        let _training_result = trainer.train_dataset_auto(&mut model, &train_data, target_extraction)?;
        
        // Evaluate on validation set
        let cross_validator = CrossValidator::new(1, scoring_metric);
        let evaluation_result = cross_validator.evaluate_model_on_dataset(&model, &val_data, target_extraction)?;
        
        evaluation_result.get_score(scoring_metric)
    }
}

/// Hyperparameter optimization results
#[derive(Debug, Clone)]
pub struct HyperparameterOptimizationResult {
    pub best_config: TrainingConfig,
    pub best_score: f64,
    pub all_evaluations: Vec<(TrainingConfig, f64)>,
    pub search_strategy: SearchStrategy,
    pub total_evaluations: usize,
}

impl HyperparameterOptimizationResult {
    /// Generate optimization report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Hyperparameter Optimization Results ===\n\n");
        report.push_str(&format!("Search Strategy: {:?}\n", self.search_strategy));
        report.push_str(&format!("Total Evaluations: {}\n", self.total_evaluations));
        report.push_str(&format!("Best Score: {:.6}\n\n", self.best_score));
        
        report.push_str("Best Configuration:\n");
        report.push_str(&format!("  Learning Rate: {:.6}\n", self.best_config.learning_rate));
        report.push_str(&format!("  Batch Size: {}\n", self.best_config.batch_size));
        report.push_str(&format!("  Epochs: {}\n\n", self.best_config.epochs));
        
        // Show top 5 configurations
        let mut sorted_evals = self.all_evaluations.clone();
        sorted_evals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        report.push_str("Top 5 Configurations:\n");
        for (i, (config, score)) in sorted_evals.iter().take(5).enumerate() {
            report.push_str(&format!(
                "  {}. LR={:.4}, BS={}, E={}: {:.4}\n",
                i + 1, config.learning_rate, config.batch_size, config.epochs, score
            ));
        }
        
        report
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        let cfg = &self.best_config;
        let best_config = assoc(vec![
            ("epochs", Value::Integer(cfg.epochs as i64)),
            ("batchSize", Value::Integer(cfg.batch_size as i64)),
            ("learningRate", Value::Real(cfg.learning_rate)),
            ("printProgress", Value::Boolean(cfg.print_progress)),
        ]);
        let evals: Vec<Value> = self.all_evaluations.iter().map(|(c, s)| {
            assoc(vec![
                ("config", assoc(vec![
                    ("epochs", Value::Integer(c.epochs as i64)),
                    ("batchSize", Value::Integer(c.batch_size as i64)),
                    ("learningRate", Value::Real(c.learning_rate)),
                    ("printProgress", Value::Boolean(c.print_progress)),
                ])),
                ("score", Value::Real(*s)),
            ])
        }).collect();
        assoc(vec![
            ("bestConfig", best_config),
            ("bestScore", Value::Real(self.best_score)),
            ("evaluations", Value::List(evals)),
            ("searchStrategy", Value::String(format!("{:?}", self.search_strategy))),
            ("totalEvaluations", Value::Integer(self.total_evaluations as i64)),
        ])
    }
}

/// Evaluation utilities for common ML tasks
pub struct EvaluationUtils;

impl EvaluationUtils {
    /// Quick evaluation of a trained model on test data
    pub fn quick_evaluate(
        model: &NetChain,
        test_data: &[(Tensor, Tensor)],
        problem_type: ProblemType,
    ) -> MLResult<EvaluationResult> {
        if test_data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot evaluate on empty test data".to_string(),
            });
        }
        
        let mut predictions = Vec::new();
        let mut targets = Vec::new();
        
        // Generate predictions
        let mut model_clone = model.clone();
        for (input, target) in test_data {
            let prediction = model_clone.forward(input)?;
            predictions.push(prediction);
            targets.push(target.clone());
        }
        
        // Combine into single tensors
        let combined_predictions = Self::combine_tensor_list(&predictions)?;
        let combined_targets = Self::combine_tensor_list(&targets)?;
        
        // Generate appropriate evaluation result
        match problem_type {
            ProblemType::BinaryClassification | ProblemType::MultiClassification => {
                let report = EvaluationMetrics::classification_report(&combined_targets, &combined_predictions)?;
                Ok(EvaluationResult::Classification(report))
            },
            ProblemType::Regression => {
                let report = EvaluationMetrics::regression_report(&combined_targets, &combined_predictions)?;
                Ok(EvaluationResult::Regression(report))
            },
            _ => Err(MLError::DataError {
                reason: "Cannot evaluate unknown problem type".to_string(),
            }),
        }
    }
    
    /// Helper: Combine list of tensors into single tensor
    fn combine_tensor_list(tensors: &[Tensor]) -> MLResult<Tensor> {
        if tensors.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot combine empty tensor list".to_string(),
            });
        }
        
        let total_elements: usize = tensors.iter().map(|t| t.data.len()).sum();
        let mut combined_data = Vec::with_capacity(total_elements);
        
        for tensor in tensors {
            combined_data.extend_from_slice(&tensor.data);
        }
        
        Tensor::new(combined_data, vec![total_elements])
    }
    
    /// Generate learning curve data for training analysis
    pub fn learning_curve(
        dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        model_builder: impl Fn() -> MLResult<NetChain>,
        config: &TrainingConfig,
        train_sizes: Vec<f64>,
    ) -> MLResult<LearningCurveResult> {
        let mut train_scores = Vec::new();
        let mut val_scores = Vec::new();
        let mut train_size_actual = Vec::new();
        
        for &train_size in &train_sizes {
            if !(0.0..=1.0).contains(&train_size) {
                continue;
            }
            
            // Create subset of data for this training size
            let (train_subset, val_data) = DataSplitter::train_test_split(dataset, 0.2, true, false)?;
            
            // Further subset training data
            let train_list = train_subset.get_value().as_list().ok_or_else(|| MLError::DataError {
                reason: "Train subset must contain List data".to_string(),
            })?;
            let subset_size = (train_list.len() as f64 * train_size) as usize;
            let train_elements = match train_subset.get_value() {
                Value::List(elements) => elements[..subset_size].to_vec(),
                _ => return Err(MLError::DataError {
                    reason: "Dataset must contain List data".to_string(),
                }),
            };
            
            let train_data_subset = ForeignDataset::new(Value::List(train_elements));
            
            // Train model on subset
            let mut model = model_builder()?;
            let trainer = NetTrain::with_config(config.clone());
            let _training_result = trainer.train_dataset_auto(&mut model, &train_data_subset, target_extraction)?;
            
            // Evaluate on training and validation sets
            let cross_validator = CrossValidator::new(1, ScoringMetric::Accuracy);
            
            let train_eval = cross_validator.evaluate_model_on_dataset(&model, &train_data_subset, target_extraction)?;
            let val_eval = cross_validator.evaluate_model_on_dataset(&model, &val_data, target_extraction)?;
            
            let train_score = train_eval.get_score(ScoringMetric::Accuracy)?;
            let val_score = val_eval.get_score(ScoringMetric::Accuracy)?;
            
            train_scores.push(train_score);
            val_scores.push(val_score);
            train_size_actual.push(subset_size);
            
            println!("Train size {}: Train={:.4}, Val={:.4}", subset_size, train_score, val_score);
        }
        
        Ok(LearningCurveResult {
            train_sizes: train_size_actual,
            train_scores,
            validation_scores: val_scores,
        })
    }
}

/// Learning curve analysis results
#[derive(Debug, Clone)]
pub struct LearningCurveResult {
    pub train_sizes: Vec<usize>,
    pub train_scores: Vec<f64>,
    pub validation_scores: Vec<f64>,
}

impl LearningCurveResult {
    /// Generate learning curve report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Learning Curve Analysis ===\n\n");
        report.push_str("Train Size | Train Score | Val Score | Gap\n");
        report.push_str("-----------|-------------|-----------|------\n");
        
        for (i, &size) in self.train_sizes.iter().enumerate() {
            let train_score = self.train_scores[i];
            let val_score = self.validation_scores[i];
            let gap = train_score - val_score;
            
            report.push_str(&format!(
                "{:>10} | {:>11.4} | {:>9.4} | {:>5.4}\n",
                size, train_score, val_score, gap
            ));
        }
        
        // Analysis
        let final_gap = self.train_scores.last().unwrap_or(&0.0) - self.validation_scores.last().unwrap_or(&0.0);
        report.push_str(&format!("\nFinal Train-Validation Gap: {:.4}\n", final_gap));
        
        if final_gap > 0.1 {
            report.push_str("Analysis: Model may be overfitting. Consider regularization or more data.\n");
        } else if final_gap < 0.05 {
            report.push_str("Analysis: Good generalization. Model is well-calibrated.\n");
        } else {
            report.push_str("Analysis: Moderate overfitting. Model performance is acceptable.\n");
        }
        
        report
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        assoc(vec![
            ("trainSizes", Value::List(self.train_sizes.iter().map(|&x| Value::Integer(x as i64)).collect())),
            ("trainScores", Value::List(self.train_scores.iter().map(|&x| Value::Real(x)).collect())),
            ("validationScores", Value::List(self.validation_scores.iter().map(|&x| Value::Real(x)).collect())),
        ])
    }
}

/// High-level evaluation API for easy access
pub struct ModelEvaluator;

impl ModelEvaluator {
    /// Comprehensive model evaluation with automatic metric selection
    pub fn evaluate_model(
        model: &mut NetChain,
        test_dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        problem_type: Option<ProblemType>,
    ) -> MLResult<EvaluationResult> {
        // Create DataLoader for test data
        let config = DataLoaderConfig {
            batch_size: 32,
            shuffle: false,
            drop_last: false,
            num_workers: 1,
            pin_memory: false,
        };
        
        let dataloader = DataLoader::from_dataset(test_dataset.clone(), target_extraction, config)?;
        
        // Collect all test data
        let mut test_data = Vec::new();
        for batch_idx in 0..dataloader.num_batches() {
            let batch = dataloader.get_batch(batch_idx)?;
            test_data.extend(batch);
        }
        
        // Infer problem type if not provided
        let inferred_problem_type = if let Some(pt) = problem_type {
            pt
        } else {
            // Simple inference based on target values
            if let Some((_, first_target)) = test_data.first() {
                if first_target.data.iter().all(|v| v.value() == 0.0 || v.value() == 1.0) {
                    ProblemType::BinaryClassification
                } else {
                    ProblemType::Regression
                }
            } else {
                ProblemType::Unknown
            }
        };
        
        // Evaluate using appropriate metrics
        EvaluationUtils::quick_evaluate(model, &test_data, inferred_problem_type)
    }
    
    /// Compare multiple models on the same dataset
    pub fn compare_models(
        models: Vec<(&str, &mut NetChain)>,
        test_dataset: &ForeignDataset,
        target_extraction: crate::stdlib::ml::DatasetTargetExtraction,
        problem_type: ProblemType,
    ) -> MLResult<ModelComparisonResult> {
        let mut model_results = Vec::new();
        
        for (model_name, model) in models {
            let evaluation_result = Self::evaluate_model(model, test_dataset, target_extraction, Some(problem_type))?;
            model_results.push((model_name.to_string(), evaluation_result));
        }
        
        // Determine best model based on problem type
        let scoring_metric = match problem_type {
            ProblemType::BinaryClassification | ProblemType::MultiClassification => ScoringMetric::F1Score,
            ProblemType::Regression => ScoringMetric::RSquared,
            _ => ScoringMetric::Accuracy,
        };
        
        let best_model = model_results.iter()
            .max_by(|(_, result_a), (_, result_b)| {
                let score_a = result_a.get_score(scoring_metric).unwrap_or(f64::NEG_INFINITY);
                let score_b = result_b.get_score(scoring_metric).unwrap_or(f64::NEG_INFINITY);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        
        Ok(ModelComparisonResult {
            model_results,
            best_model_name: best_model,
            comparison_metric: scoring_metric,
            problem_type,
        })
    }
}

/// Model comparison results
#[derive(Debug, Clone)]
pub struct ModelComparisonResult {
    pub model_results: Vec<(String, EvaluationResult)>,
    pub best_model_name: String,
    pub comparison_metric: ScoringMetric,
    pub problem_type: ProblemType,
}

impl ModelComparisonResult {
    /// Generate model comparison report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Model Comparison Results ===\n\n");
        report.push_str(&format!("Problem Type: {:?}\n", self.problem_type));
        report.push_str(&format!("Comparison Metric: {:?}\n", self.comparison_metric));
        report.push_str(&format!("Best Model: {}\n\n", self.best_model_name));
        
        report.push_str("Model Performance:\n");
        for (model_name, result) in &self.model_results {
            let score = result.get_score(self.comparison_metric).unwrap_or(0.0);
            report.push_str(&format!("  {}: {:.4}\n", model_name, score));
        }
        
        report
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        let results: Vec<Value> = self.model_results.iter().map(|(name, res)| {
            assoc(vec![
                ("modelName", Value::String(name.clone())),
                ("report", res.to_value()),
            ])
        }).collect();
        assoc(vec![
            ("results", Value::List(results)),
            ("bestModelName", Value::String(self.best_model_name.clone())),
            ("comparisonMetric", Value::String(format!("{:?}", self.comparison_metric))),
            ("problemType", Value::String(format!("{:?}", self.problem_type))),
        ])
    }
}

/// Statistical significance testing for model comparison
pub struct StatisticalTester;

impl StatisticalTester {
    /// Perform paired t-test between two models' cross-validation scores
    pub fn paired_t_test(scores_a: &[f64], scores_b: &[f64]) -> MLResult<TTestResult> {
        if scores_a.len() != scores_b.len() || scores_a.is_empty() {
            return Err(MLError::DataError {
                reason: "Score arrays must have the same non-zero length".to_string(),
            });
        }
        
        let differences: Vec<f64> = scores_a.iter()
            .zip(scores_b.iter())
            .map(|(a, b)| a - b)
            .collect();
        
        let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
        let var_diff = differences.iter()
            .map(|d| (d - mean_diff).powi(2))
            .sum::<f64>() / (differences.len() - 1) as f64;
        let se_diff = (var_diff / differences.len() as f64).sqrt();
        
        let t_statistic = mean_diff / se_diff;
        let degrees_freedom = differences.len() - 1;
        
        // Simplified p-value calculation (would use proper t-distribution)
        let p_value = if t_statistic.abs() > 2.0 {
            0.05 // Roughly significant
        } else {
            0.2  // Not significant
        };
        
        Ok(TTestResult {
            t_statistic,
            p_value,
            degrees_freedom,
            mean_difference: mean_diff,
            is_significant: p_value < 0.05,
        })
    }
}

/// T-test results for statistical comparison
#[derive(Debug, Clone)]
pub struct TTestResult {
    pub t_statistic: f64,
    pub p_value: f64,
    pub degrees_freedom: usize,
    pub mean_difference: f64,
    pub is_significant: bool,
}

impl TTestResult {
    /// Generate statistical test report
    pub fn to_string(&self) -> String {
        format!(
            "Paired T-Test Results:\n\
            T-statistic: {:.4}\n\
            P-value: {:.4}\n\
            Degrees of Freedom: {}\n\
            Mean Difference: {:.6}\n\
            Significant (p < 0.05): {}",
            self.t_statistic,
            self.p_value,
            self.degrees_freedom,
            self.mean_difference,
            self.is_significant
        )
    }
}

/// Advanced evaluation strategies
pub struct AdvancedEvaluator;

impl AdvancedEvaluator {
    /// Bootstrap evaluation for confidence intervals
    pub fn bootstrap_evaluation(
        model: &NetChain,
        test_data: &[(Tensor, Tensor)],
        problem_type: ProblemType,
        n_bootstrap: usize,
    ) -> MLResult<BootstrapEvaluationResult> {
        if test_data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot bootstrap evaluate on empty test data".to_string(),
            });
        }
        
        let mut bootstrap_scores = Vec::new();
        let mut rng = thread_rng();
        
        for _ in 0..n_bootstrap {
            // Sample with replacement
            let bootstrap_sample: Vec<(Tensor, Tensor)> = (0..test_data.len())
                .map(|_| {
                    let idx = rng.gen_range(0..test_data.len());
                    test_data[idx].clone()
                })
                .collect();
            
            // Evaluate on bootstrap sample
            let evaluation_result = EvaluationUtils::quick_evaluate(model, &bootstrap_sample, problem_type)?;
            
            // Extract primary score
            let score = match problem_type {
                ProblemType::BinaryClassification | ProblemType::MultiClassification => {
                    evaluation_result.get_score(ScoringMetric::Accuracy)?
                },
                ProblemType::Regression => {
                    evaluation_result.get_score(ScoringMetric::RSquared)?
                },
                _ => 0.0,
            };
            
            bootstrap_scores.push(score);
        }
        
        // Calculate confidence intervals
        bootstrap_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_percentile = bootstrap_scores[n_bootstrap * 5 / 100]; // 5th percentile
        let upper_percentile = bootstrap_scores[n_bootstrap * 95 / 100]; // 95th percentile
        let median = bootstrap_scores[n_bootstrap / 2];
        let mean = bootstrap_scores.iter().sum::<f64>() / bootstrap_scores.len() as f64;
        
        Ok(BootstrapEvaluationResult {
            mean_score: mean,
            median_score: median,
            confidence_interval_95: (lower_percentile, upper_percentile),
            bootstrap_scores,
            n_bootstrap,
        })
    }
}

/// Bootstrap evaluation results with confidence intervals
#[derive(Debug, Clone)]
pub struct BootstrapEvaluationResult {
    pub mean_score: f64,
    pub median_score: f64,
    pub confidence_interval_95: (f64, f64),
    pub bootstrap_scores: Vec<f64>,
    pub n_bootstrap: usize,
}

impl BootstrapEvaluationResult {
    /// Generate bootstrap evaluation report
    pub fn to_string(&self) -> String {
        format!(
            "Bootstrap Evaluation Results ({} iterations):\n\
            Mean Score: {:.4}\n\
            Median Score: {:.4}\n\
            95% Confidence Interval: [{:.4}, {:.4}]\n\
            Score Range: [{:.4}, {:.4}]",
            self.n_bootstrap,
            self.mean_score,
            self.median_score,
            self.confidence_interval_95.0,
            self.confidence_interval_95.1,
            self.bootstrap_scores.iter().fold(f64::INFINITY, |acc, &x| acc.min(x)),
            self.bootstrap_scores.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
        )
    }

    /// Convert to standardized Association
    pub fn to_value(&self) -> Value {
        assoc(vec![
            ("meanScore", Value::Real(self.mean_score)),
            ("medianScore", Value::Real(self.median_score)),
            ("confidenceInterval95", assoc(vec![
                ("lower", Value::Real(self.confidence_interval_95.0)),
                ("upper", Value::Real(self.confidence_interval_95.1)),
            ])),
            ("scores", Value::List(self.bootstrap_scores.iter().map(|&x| Value::Real(x)).collect())),
            ("iterations", Value::Integer(self.n_bootstrap as i64)),
        ])
    }
}

/// Value extension trait for convenience
trait ValueExt {
    fn as_list(&self) -> MLResult<&Vec<Value>>;
}

impl ValueExt for Value {
    fn as_list(&self) -> MLResult<&Vec<Value>> {
        match self {
            Value::List(list) => Ok(list),
            _ => Err(MLError::DataError {
                reason: "Value is not a list".to_string(),
            }),
        }
    }
}
