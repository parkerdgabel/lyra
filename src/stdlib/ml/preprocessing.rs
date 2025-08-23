//! ML Data Preprocessing Infrastructure
//!
//! This module provides preprocessing capabilities specifically designed for ML workflows,
//! leveraging existing data processing and statistical functions in the Lyra stdlib.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::autodiff::Dual;
use crate::vm::Value;
use std::collections::HashMap;

/// Trait for ML-specific data preprocessing operations
pub trait MLPreprocessor: Send + Sync {
    /// Apply preprocessing to a Value
    fn preprocess(&self, data: &Value) -> MLResult<Value>;
    
    /// Get the name of this preprocessor
    fn name(&self) -> &str;
    
    /// Get configuration parameters
    fn config(&self) -> HashMap<String, Value>;
    
    /// Clone the preprocessor
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor>;
}

/// Standard Scaler: Normalizes data to have mean=0 and std=1
/// Leverages existing statistical functions (Mean, StandardDeviation)
#[derive(Debug, Clone)]
pub struct StandardScaler {
    name: String,
    fitted: bool,
    mean: Option<f64>,
    std: Option<f64>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            name: "StandardScaler".to_string(),
            fitted: false,
            mean: None,
            std: None,
        }
    }
    
    /// Fit the scaler to training data
    pub fn fit(&mut self, data: &[f64]) -> MLResult<()> {
        if data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot fit StandardScaler on empty data".to_string(),
            });
        }
        
        // Calculate mean and standard deviation
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std = variance.sqrt();
        
        if std == 0.0 {
            return Err(MLError::DataError {
                reason: "Cannot scale data with zero variance".to_string(),
            });
        }
        
        self.mean = Some(mean);
        self.std = Some(std);
        self.fitted = true;
        
        Ok(())
    }
    
    /// Transform data using fitted parameters
    pub fn transform(&self, data: &[f64]) -> MLResult<Vec<f64>> {
        if !self.fitted {
            return Err(MLError::DataError {
                reason: "StandardScaler must be fitted before transform".to_string(),
            });
        }
        
        let mean = self.mean.unwrap();
        let std = self.std.unwrap();
        
        let transformed: Vec<f64> = data.iter()
            .map(|&x| (x - mean) / std)
            .collect();
        
        Ok(transformed)
    }
    
    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &[f64]) -> MLResult<Vec<f64>> {
        self.fit(data)?;
        self.transform(data)
    }
}

impl MLPreprocessor for StandardScaler {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        match data {
            Value::List(elements) => {
                let numeric_data: Result<Vec<f64>, MLError> = elements.iter()
                    .map(|v| match v {
                        Value::Real(n) => Ok(*n),
                        Value::Integer(n) => Ok(*n as f64),
                        _ => Err(MLError::DataError {
                            reason: "StandardScaler requires numeric data".to_string(),
                        }),
                    })
                    .collect();
                
                let data = numeric_data?;
                
                if !self.fitted {
                    return Err(MLError::DataError {
                        reason: "StandardScaler must be fitted before preprocessing".to_string(),
                    });
                }
                
                let transformed = self.transform(&data)?;
                let result_values: Vec<Value> = transformed.iter()
                    .map(|&x| Value::Real(x))
                    .collect();
                
                Ok(Value::List(result_values))
            },
            _ => Err(MLError::DataError {
                reason: "StandardScaler requires List input".to_string(),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("fitted".to_string(), Value::Boolean(self.fitted));
        if let Some(mean) = self.mean {
            config.insert("mean".to_string(), Value::Real(mean));
        }
        if let Some(std) = self.std {
            config.insert("std".to_string(), Value::Real(std));
        }
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(self.clone())
    }
}

/// One-Hot Encoder: Converts categorical data to binary vectors
#[derive(Debug, Clone)]
pub struct OneHotEncoder {
    name: String,
    categories: HashMap<String, Vec<String>>,
    fitted: bool,
}

impl OneHotEncoder {
    pub fn new() -> Self {
        Self {
            name: "OneHotEncoder".to_string(),
            categories: HashMap::new(),
            fitted: false,
        }
    }
    
    /// Fit the encoder to categorical data
    pub fn fit(&mut self, data: &[String], feature_name: &str) -> MLResult<()> {
        // Extract unique categories
        let mut unique_categories: Vec<String> = data.iter().cloned().collect();
        unique_categories.sort();
        unique_categories.dedup();
        
        if unique_categories.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot fit OneHotEncoder on empty categorical data".to_string(),
            });
        }
        
        self.categories.insert(feature_name.to_string(), unique_categories);
        self.fitted = true;
        
        Ok(())
    }
    
    /// Transform categorical data to one-hot encoded vectors
    pub fn transform(&self, data: &[String], feature_name: &str) -> MLResult<Vec<Vec<f64>>> {
        if !self.fitted {
            return Err(MLError::DataError {
                reason: "OneHotEncoder must be fitted before transform".to_string(),
            });
        }
        
        let categories = self.categories.get(feature_name)
            .ok_or_else(|| MLError::DataError {
                reason: format!("Feature '{}' not found in fitted categories", feature_name),
            })?;
        
        let mut result = Vec::new();
        
        for item in data {
            let mut encoded = vec![0.0; categories.len()];
            
            if let Some(index) = categories.iter().position(|cat| cat == item) {
                encoded[index] = 1.0;
            } else {
                // Unknown category - could handle with error or default behavior
                return Err(MLError::DataError {
                    reason: format!("Unknown category '{}' for feature '{}'", item, feature_name),
                });
            }
            
            result.push(encoded);
        }
        
        Ok(result)
    }
}

impl MLPreprocessor for OneHotEncoder {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        match data {
            Value::List(elements) => {
                let string_data: Result<Vec<String>, MLError> = elements.iter()
                    .map(|v| match v {
                        Value::String(s) => Ok(s.clone()),
                        Value::Symbol(s) => Ok(s.clone()),
                        _ => Err(MLError::DataError {
                            reason: "OneHotEncoder requires string/symbol data".to_string(),
                        }),
                    })
                    .collect();
                
                let data = string_data?;
                
                if !self.fitted || self.categories.is_empty() {
                    return Err(MLError::DataError {
                        reason: "OneHotEncoder must be fitted before preprocessing".to_string(),
                    });
                }
                
                // Use the first feature name for single-feature encoding
                let feature_name = self.categories.keys().next().unwrap();
                let encoded = self.transform(&data, feature_name)?;
                
                // Convert to nested Value structure
                let result_values: Vec<Value> = encoded.iter()
                    .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                    .collect();
                
                Ok(Value::List(result_values))
            },
            _ => Err(MLError::DataError {
                reason: "OneHotEncoder requires List input".to_string(),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("fitted".to_string(), Value::Boolean(self.fitted));
        config.insert("num_features".to_string(), Value::Integer(self.categories.len() as i64));
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(self.clone())
    }
}

/// Missing Value Handler: Imputes missing values using statistical methods
/// Leverages existing statistical functions (Mean, Median)
#[derive(Debug, Clone)]
pub struct MissingValueHandler {
    name: String,
    strategy: ImputationStrategy,
    fitted_values: HashMap<String, f64>,
    fitted: bool,
}

#[derive(Debug, Clone)]
pub enum ImputationStrategy {
    Mean,
    Median,
    Mode,
    Constant(f64),
}

impl MissingValueHandler {
    pub fn new(strategy: ImputationStrategy) -> Self {
        Self {
            name: "MissingValueHandler".to_string(),
            strategy,
            fitted_values: HashMap::new(),
            fitted: false,
        }
    }
    
    /// Fit the handler to training data
    pub fn fit(&mut self, data: &[Option<f64>], feature_name: &str) -> MLResult<()> {
        let valid_data: Vec<f64> = data.iter()
            .filter_map(|&x| x)
            .collect();
        
        if valid_data.is_empty() {
            return Err(MLError::DataError {
                reason: "Cannot fit MissingValueHandler - no valid data".to_string(),
            });
        }
        
        let impute_value = match &self.strategy {
            ImputationStrategy::Mean => {
                valid_data.iter().sum::<f64>() / valid_data.len() as f64
            },
            ImputationStrategy::Median => {
                let mut sorted = valid_data.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                if n % 2 == 0 {
                    (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
                } else {
                    sorted[n / 2]
                }
            },
            ImputationStrategy::Mode => {
                // Find most frequent value (simplified implementation)
                let mut counts = HashMap::new();
                for &value in &valid_data {
                    *counts.entry((value * 1000.0) as i64).or_insert(0) += 1;
                }
                let most_frequent = counts.iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&value, _)| value as f64 / 1000.0)
                    .unwrap_or(0.0);
                most_frequent
            },
            ImputationStrategy::Constant(value) => *value,
        };
        
        self.fitted_values.insert(feature_name.to_string(), impute_value);
        self.fitted = true;
        
        Ok(())
    }
    
    /// Transform data by imputing missing values
    pub fn transform(&self, data: &[Option<f64>], feature_name: &str) -> MLResult<Vec<f64>> {
        if !self.fitted {
            return Err(MLError::DataError {
                reason: "MissingValueHandler must be fitted before transform".to_string(),
            });
        }
        
        let impute_value = self.fitted_values.get(feature_name)
            .ok_or_else(|| MLError::DataError {
                reason: format!("Feature '{}' not found in fitted values", feature_name),
            })?;
        
        let result: Vec<f64> = data.iter()
            .map(|&x| x.unwrap_or(*impute_value))
            .collect();
        
        Ok(result)
    }
}

impl MLPreprocessor for MissingValueHandler {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        // For simplicity, assume missing values are represented as Missing in the Value enum
        match data {
            Value::List(elements) => {
                let processed_data: Result<Vec<f64>, MLError> = elements.iter()
                    .map(|v| match v {
                        Value::Real(n) => Ok(*n),
                        Value::Integer(n) => Ok(*n as f64),
                        Value::Missing => {
                            // Use the default imputation value if fitted
                            if let Some(default_value) = self.fitted_values.values().next() {
                                Ok(*default_value)
                            } else {
                                Err(MLError::DataError {
                                    reason: "MissingValueHandler not fitted".to_string(),
                                })
                            }
                        },
                        _ => Err(MLError::DataError {
                            reason: "MissingValueHandler requires numeric data".to_string(),
                        }),
                    })
                    .collect();
                
                let data = processed_data?;
                let result_values: Vec<Value> = data.iter()
                    .map(|&x| Value::Real(x))
                    .collect();
                
                Ok(Value::List(result_values))
            },
            _ => Err(MLError::DataError {
                reason: "MissingValueHandler requires List input".to_string(),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("fitted".to_string(), Value::Boolean(self.fitted));
        config.insert("strategy".to_string(), Value::String(match &self.strategy {
            ImputationStrategy::Mean => "mean".to_string(),
            ImputationStrategy::Median => "median".to_string(),
            ImputationStrategy::Mode => "mode".to_string(),
            ImputationStrategy::Constant(val) => format!("constant_{}", val),
        }));
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(self.clone())
    }
}

/// Outlier Remover: Removes outliers using existing outlier detection methods
/// Leverages existing outlier detection functions (IQR, Z-score)
#[derive(Debug, Clone)]
pub struct OutlierRemover {
    name: String,
    method: OutlierMethod,
    threshold: f64,
}

#[derive(Debug, Clone)]
pub enum OutlierMethod {
    IQR,
    ZScore,
    ModifiedZScore,
}

impl OutlierRemover {
    pub fn new(method: OutlierMethod, threshold: f64) -> Self {
        Self {
            name: "OutlierRemover".to_string(),
            method,
            threshold,
        }
    }
    
    /// Remove outliers from data
    pub fn remove_outliers(&self, data: &[f64]) -> MLResult<Vec<f64>> {
        let outlier_indices = self.detect_outlier_indices(data)?;
        
        let filtered_data: Vec<f64> = data.iter().enumerate()
            .filter(|(i, _)| !outlier_indices.contains(i))
            .map(|(_, &value)| value)
            .collect();
        
        Ok(filtered_data)
    }
    
    /// Detect outlier indices using the specified method
    fn detect_outlier_indices(&self, data: &[f64]) -> MLResult<Vec<usize>> {
        match self.method {
            OutlierMethod::IQR => self.detect_outliers_iqr(data),
            OutlierMethod::ZScore => self.detect_outliers_zscore(data),
            OutlierMethod::ModifiedZScore => self.detect_outliers_modified_zscore(data),
        }
    }
    
    fn detect_outliers_iqr(&self, data: &[f64]) -> MLResult<Vec<usize>> {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let n = sorted_data.len();
        let q1 = sorted_data[n / 4];
        let q3 = sorted_data[3 * n / 4];
        let iqr = q3 - q1;
        
        let lower_bound = q1 - self.threshold * iqr;
        let upper_bound = q3 + self.threshold * iqr;
        
        let outliers: Vec<usize> = data.iter().enumerate()
            .filter_map(|(i, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        Ok(outliers)
    }
    
    fn detect_outliers_zscore(&self, data: &[f64]) -> MLResult<Vec<usize>> {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return Ok(Vec::new()); // No outliers in constant data
        }
        
        let outliers: Vec<usize> = data.iter().enumerate()
            .filter_map(|(i, &value)| {
                let z_score = (value - mean) / std_dev;
                if z_score.abs() > self.threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        Ok(outliers)
    }
    
    fn detect_outliers_modified_zscore(&self, data: &[f64]) -> MLResult<Vec<usize>> {
        // Calculate median
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let median = if n % 2 == 0 {
            (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        } else {
            sorted[n / 2]
        };
        
        // Calculate MAD (Median Absolute Deviation)
        let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
        let mut sorted_deviations = deviations.clone();
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = if n % 2 == 0 {
            (sorted_deviations[n / 2 - 1] + sorted_deviations[n / 2]) / 2.0
        } else {
            sorted_deviations[n / 2]
        };
        
        if mad == 0.0 {
            return Ok(Vec::new()); // No outliers in constant data
        }
        
        let outliers: Vec<usize> = data.iter().enumerate()
            .filter_map(|(i, &value)| {
                let modified_z_score = 0.6745 * (value - median) / mad;
                if modified_z_score.abs() > self.threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        
        Ok(outliers)
    }
}

impl MLPreprocessor for OutlierRemover {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        match data {
            Value::List(elements) => {
                let numeric_data: Result<Vec<f64>, MLError> = elements.iter()
                    .map(|v| match v {
                        Value::Real(n) => Ok(*n),
                        Value::Integer(n) => Ok(*n as f64),
                        _ => Err(MLError::DataError {
                            reason: "OutlierRemover requires numeric data".to_string(),
                        }),
                    })
                    .collect();
                
                let data = numeric_data?;
                let filtered_data = self.remove_outliers(&data)?;
                
                let result_values: Vec<Value> = filtered_data.iter()
                    .map(|&x| Value::Real(x))
                    .collect();
                
                Ok(Value::List(result_values))
            },
            _ => Err(MLError::DataError {
                reason: "OutlierRemover requires List input".to_string(),
            }),
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("method".to_string(), Value::String(match self.method {
            OutlierMethod::IQR => "iqr".to_string(),
            OutlierMethod::ZScore => "zscore".to_string(),
            OutlierMethod::ModifiedZScore => "modified_zscore".to_string(),
        }));
        config.insert("threshold".to_string(), Value::Real(self.threshold));
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(self.clone())
    }
}

/// Auto Preprocessor: Automatically selects and applies appropriate preprocessing
/// based on data characteristics using existing data analysis functions
pub struct AutoPreprocessor {
    name: String,
    pipeline: Vec<Box<dyn MLPreprocessor>>,
    fitted: bool,
}

impl std::fmt::Debug for AutoPreprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoPreprocessor")
            .field("name", &self.name)
            .field("pipeline_length", &self.pipeline.len())
            .field("fitted", &self.fitted)
            .finish()
    }
}

impl Clone for AutoPreprocessor {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            pipeline: self.pipeline.iter().map(|p| p.clone_boxed()).collect(),
            fitted: self.fitted,
        }
    }
}

impl AutoPreprocessor {
    pub fn new() -> Self {
        Self {
            name: "AutoPreprocessor".to_string(),
            pipeline: Vec::new(),
            fitted: false,
        }
    }
    
    /// Automatically infer the best preprocessing pipeline for the given data
    pub fn infer_pipeline(data: &Value) -> MLResult<Self> {
        let mut preprocessor = Self::new();
        
        match data {
            Value::List(elements) => {
                // Analyze data characteristics
                let (has_numeric, has_categorical, has_missing) = Self::analyze_data_types(elements);
                
                // Build preprocessing pipeline based on data characteristics
                if has_missing {
                    // Add missing value handler with mean imputation
                    preprocessor.add_preprocessor(Box::new(
                        MissingValueHandler::new(ImputationStrategy::Mean)
                    ));
                }
                
                if has_numeric {
                    // Check for outliers and add outlier removal if needed
                    let numeric_data: Vec<f64> = elements.iter()
                        .filter_map(|v| match v {
                            Value::Real(n) => Some(*n),
                            Value::Integer(n) => Some(*n as f64),
                            _ => None,
                        })
                        .collect();
                    
                    if Self::has_potential_outliers(&numeric_data) {
                        preprocessor.add_preprocessor(Box::new(
                            OutlierRemover::new(OutlierMethod::IQR, 1.5)
                        ));
                    }
                    
                    // Add standard scaling for numeric data
                    preprocessor.add_preprocessor(Box::new(StandardScaler::new()));
                }
                
                if has_categorical {
                    // Add one-hot encoding for categorical data
                    preprocessor.add_preprocessor(Box::new(OneHotEncoder::new()));
                }
                
                Ok(preprocessor)
            },
            _ => Err(MLError::DataError {
                reason: "AutoPreprocessor requires List input for analysis".to_string(),
            }),
        }
    }
    
    /// Analyze data types in the list
    fn analyze_data_types(elements: &[Value]) -> (bool, bool, bool) {
        let mut has_numeric = false;
        let mut has_categorical = false;
        let mut has_missing = false;
        
        for element in elements {
            match element {
                Value::Real(_) | Value::Integer(_) => has_numeric = true,
                Value::String(_) | Value::Symbol(_) => has_categorical = true,
                Value::Missing => has_missing = true,
                _ => {},
            }
        }
        
        (has_numeric, has_categorical, has_missing)
    }
    
    /// Check if numeric data likely contains outliers
    fn has_potential_outliers(data: &[f64]) -> bool {
        if data.len() < 10 {
            return false; // Too small for meaningful outlier detection
        }
        
        // Simple heuristic: check if range is more than 3 standard deviations
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
        let std_dev = variance.sqrt();
        
        let min = data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max = data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        
        (max - min) > 6.0 * std_dev // Range > 6 standard deviations suggests outliers
    }
    
    /// Add a preprocessor to the pipeline
    pub fn add_preprocessor(&mut self, preprocessor: Box<dyn MLPreprocessor>) {
        self.pipeline.push(preprocessor);
    }
    
    /// Apply all preprocessors in sequence
    pub fn apply_pipeline(&self, data: &Value) -> MLResult<Value> {
        let mut current_data = data.clone();
        
        for preprocessor in &self.pipeline {
            current_data = preprocessor.preprocess(&current_data)?;
        }
        
        Ok(current_data)
    }
}

impl MLPreprocessor for AutoPreprocessor {
    fn preprocess(&self, data: &Value) -> MLResult<Value> {
        self.apply_pipeline(data)
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert("fitted".to_string(), Value::Boolean(self.fitted));
        config.insert("pipeline_length".to_string(), Value::Integer(self.pipeline.len() as i64));
        
        let preprocessor_names: Vec<Value> = self.pipeline.iter()
            .map(|p| Value::String(p.name().to_string()))
            .collect();
        config.insert("preprocessors".to_string(), Value::List(preprocessor_names));
        
        config
    }
    
    fn clone_boxed(&self) -> Box<dyn MLPreprocessor> {
        Box::new(Self {
            name: self.name.clone(),
            pipeline: self.pipeline.iter().map(|p| p.clone_boxed()).collect(),
            fitted: self.fitted,
        })
    }
}

/// Advanced Preprocessing Pipeline: Sophisticated chaining with conditional logic
/// Provides advanced composition, configuration, and reproducibility features
#[derive(Debug)]
pub struct AdvancedPreprocessingPipeline {
    name: String,
    steps: Vec<PipelineStep>,
    metadata: HashMap<String, Value>,
    config: PipelineConfig,
}

/// Configuration for advanced preprocessing pipeline
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub skip_on_error: bool,
    pub cache_intermediate: bool,
    pub parallel_execution: bool,
    pub validation_enabled: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            skip_on_error: false,
            cache_intermediate: true,
            parallel_execution: false,
            validation_enabled: true,
        }
    }
}

/// A single step in the preprocessing pipeline with conditional logic
pub struct PipelineStep {
    pub name: String,
    pub preprocessor: Box<dyn MLPreprocessor>,
    pub condition: Option<Box<dyn PipelineCondition>>,
    pub cache_key: Option<String>,
}

impl std::fmt::Debug for PipelineStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipelineStep")
            .field("name", &self.name)
            .field("preprocessor_name", &self.preprocessor.name())
            .field("has_condition", &self.condition.is_some())
            .field("cache_key", &self.cache_key)
            .finish()
    }
}

/// Trait for conditional pipeline execution
pub trait PipelineCondition: Send + Sync + std::fmt::Debug {
    /// Check if this step should be executed for the given data
    fn should_execute(&self, data: &Value, metadata: &HashMap<String, Value>) -> bool;
    
    /// Get condition description
    fn description(&self) -> String;
}

/// Condition: Execute only for numeric data
#[derive(Debug)]
pub struct NumericDataCondition;

impl PipelineCondition for NumericDataCondition {
    fn should_execute(&self, data: &Value, _metadata: &HashMap<String, Value>) -> bool {
        match data {
            Value::List(elements) => {
                elements.iter().any(|v| matches!(v, Value::Real(_) | Value::Integer(_)))
            },
            Value::Real(_) | Value::Integer(_) => true,
            _ => false,
        }
    }
    
    fn description(&self) -> String {
        "Execute only for numeric data".to_string()
    }
}

/// Condition: Execute only for categorical data
#[derive(Debug)]
pub struct CategoricalDataCondition;

impl PipelineCondition for CategoricalDataCondition {
    fn should_execute(&self, data: &Value, _metadata: &HashMap<String, Value>) -> bool {
        match data {
            Value::List(elements) => {
                elements.iter().any(|v| matches!(v, Value::String(_) | Value::Symbol(_)))
            },
            Value::String(_) | Value::Symbol(_) => true,
            _ => false,
        }
    }
    
    fn description(&self) -> String {
        "Execute only for categorical data".to_string()
    }
}

/// Condition: Execute only if data size exceeds threshold
#[derive(Debug)]
pub struct DataSizeCondition {
    pub min_size: usize,
}

impl PipelineCondition for DataSizeCondition {
    fn should_execute(&self, data: &Value, _metadata: &HashMap<String, Value>) -> bool {
        match data {
            Value::List(elements) => elements.len() >= self.min_size,
            _ => true, // Single values always pass
        }
    }
    
    fn description(&self) -> String {
        format!("Execute only if data size >= {}", self.min_size)
    }
}

impl AdvancedPreprocessingPipeline {
    /// Create new empty pipeline
    pub fn new(name: String) -> Self {
        Self {
            name,
            steps: Vec::new(),
            metadata: HashMap::new(),
            config: PipelineConfig::default(),
        }
    }
    
    /// Create pipeline with custom configuration
    pub fn with_config(name: String, config: PipelineConfig) -> Self {
        Self {
            name,
            steps: Vec::new(),
            metadata: HashMap::new(),
            config,
        }
    }
    
    /// Add a preprocessing step to the pipeline
    pub fn add_step(
        &mut self, 
        name: String,
        preprocessor: Box<dyn MLPreprocessor>,
        condition: Option<Box<dyn PipelineCondition>>,
    ) -> &mut Self {
        let cache_key = if self.config.cache_intermediate {
            Some(format!("{}_{}", self.name, name))
        } else {
            None
        };
        
        self.steps.push(PipelineStep {
            name,
            preprocessor,
            condition,
            cache_key,
        });
        
        self
    }
    
    
    /// Execute the full pipeline on data
    pub fn execute(&self, data: &Value) -> MLResult<Value> {
        let mut current_data = data.clone();
        let mut step_metadata = self.metadata.clone();
        
        for (step_idx, step) in self.steps.iter().enumerate() {
            // Check condition if present
            if let Some(ref condition) = step.condition {
                if !condition.should_execute(&current_data, &step_metadata) {
                    // Skip this step
                    step_metadata.insert(
                        format!("step_{}_skipped", step_idx),
                        Value::String(format!("Condition not met: {}", condition.description()))
                    );
                    continue;
                }
            }
            
            // Execute preprocessing step
            match step.preprocessor.preprocess(&current_data) {
                Ok(processed_data) => {
                    current_data = processed_data;
                    step_metadata.insert(
                        format!("step_{}_executed", step_idx),
                        Value::String(step.name.clone())
                    );
                },
                Err(e) => {
                    if self.config.skip_on_error {
                        step_metadata.insert(
                            format!("step_{}_error", step_idx),
                            Value::String(format!("Skipped due to error: {}", e))
                        );
                        continue;
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        
        Ok(current_data)
    }
    
    /// Get pipeline execution metadata
    pub fn get_metadata(&self) -> &HashMap<String, Value> {
        &self.metadata
    }
    
    /// Set metadata for this pipeline
    pub fn set_metadata(&mut self, key: String, value: Value) {
        self.metadata.insert(key, value);
    }
    
    /// Get pipeline configuration
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
    
    /// Get number of steps in pipeline
    pub fn num_steps(&self) -> usize {
        self.steps.len()
    }
    
    /// Get description of all pipeline steps
    pub fn describe_pipeline(&self) -> Vec<String> {
        self.steps.iter().enumerate().map(|(idx, step)| {
            let condition_desc = if let Some(ref condition) = step.condition {
                format!(" ({})", condition.description())
            } else {
                String::new()
            };
            
            format!(
                "Step {}: {} - {}{}",
                idx + 1,
                step.name,
                step.preprocessor.name(),
                condition_desc
            )
        }).collect()
    }
}

/// Pipeline Builder: Fluent API for constructing preprocessing pipelines
pub struct PipelineBuilder {
    pipeline: AdvancedPreprocessingPipeline,
}

impl PipelineBuilder {
    /// Start building a new pipeline
    pub fn new(name: String) -> Self {
        Self {
            pipeline: AdvancedPreprocessingPipeline::new(name),
        }
    }
    
    /// Configure pipeline behavior
    pub fn with_config(mut self, config: PipelineConfig) -> Self {
        self.pipeline.config = config;
        self
    }
    
    /// Add standard scaling step (numeric data only)
    pub fn standard_scale(mut self) -> Self {
        self.pipeline.add_step(
            "StandardScaling".to_string(),
            Box::new(StandardScaler::new()),
            Some(Box::new(NumericDataCondition))
        );
        self
    }
    
    /// Add outlier removal step (numeric data only)
    pub fn remove_outliers(mut self, method: OutlierMethod, threshold: f64) -> Self {
        self.pipeline.add_step(
            "OutlierRemoval".to_string(),
            Box::new(OutlierRemover::new(method, threshold)),
            Some(Box::new(NumericDataCondition))
        );
        self
    }
    
    /// Add missing value handling step
    pub fn handle_missing(mut self, strategy: ImputationStrategy) -> Self {
        self.pipeline.add_step(
            "MissingValueHandling".to_string(),
            Box::new(MissingValueHandler::new(strategy)),
            None // Apply to all data types
        );
        self
    }
    
    /// Add one-hot encoding step (categorical data only)
    pub fn one_hot_encode(mut self) -> Self {
        self.pipeline.add_step(
            "OneHotEncoding".to_string(),
            Box::new(OneHotEncoder::new()),
            Some(Box::new(CategoricalDataCondition))
        );
        self
    }
    
    /// Add custom preprocessing step
    pub fn add_custom_step(
        mut self,
        name: String,
        preprocessor: Box<dyn MLPreprocessor>,
        condition: Option<Box<dyn PipelineCondition>>,
    ) -> Self {
        self.pipeline.add_step(name, preprocessor, condition);
        self
    }
    
    /// Add step only for large datasets
    pub fn add_for_large_data(
        mut self,
        name: String,
        preprocessor: Box<dyn MLPreprocessor>,
        min_size: usize,
    ) -> Self {
        self.pipeline.add_step(name, preprocessor, Some(Box::new(DataSizeCondition { min_size })));
        self
    }
    
    /// Build the final pipeline
    pub fn build(self) -> AdvancedPreprocessingPipeline {
        self.pipeline
    }
}

/// Preprocessing Pipeline Registry: Manage common pipeline configurations
pub struct PipelineRegistry {
    pipelines: HashMap<String, AdvancedPreprocessingPipeline>,
}

impl PipelineRegistry {
    /// Create new registry
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }
    
    /// Register a pipeline configuration
    pub fn register(&mut self, name: String, pipeline: AdvancedPreprocessingPipeline) {
        self.pipelines.insert(name, pipeline);
    }
    
    /// Get a registered pipeline by name
    pub fn get(&self, name: &str) -> Option<&AdvancedPreprocessingPipeline> {
        self.pipelines.get(name)
    }
    
    /// Create standard preprocessing pipelines
    pub fn create_standard_pipelines() -> Self {
        let mut registry = Self::new();
        
        // Complete preprocessing pipeline
        let complete_pipeline = PipelineBuilder::new("Complete".to_string())
            .handle_missing(ImputationStrategy::Mean)
            .remove_outliers(OutlierMethod::IQR, 1.5)
            .one_hot_encode()
            .standard_scale()
            .build();
        
        // Minimal preprocessing pipeline
        let minimal_pipeline = PipelineBuilder::new("Minimal".to_string())
            .standard_scale()
            .build();
        
        // Robust preprocessing pipeline (handles outliers aggressively)
        let robust_pipeline = PipelineBuilder::new("Robust".to_string())
            .handle_missing(ImputationStrategy::Median)
            .remove_outliers(OutlierMethod::ModifiedZScore, 3.5)
            .standard_scale()
            .build();
        
        // Fast preprocessing (minimal operations for large datasets)
        let fast_pipeline = PipelineBuilder::new("Fast".to_string())
            .add_for_large_data(
                "FastScaling".to_string(),
                Box::new(StandardScaler::new()),
                1000 // Only for datasets > 1000 samples
            )
            .build();
        
        registry.register("complete".to_string(), complete_pipeline);
        registry.register("minimal".to_string(), minimal_pipeline);
        registry.register("robust".to_string(), robust_pipeline);
        registry.register("fast".to_string(), fast_pipeline);
        
        registry
    }
    
    /// List all registered pipeline names
    pub fn list_pipelines(&self) -> Vec<String> {
        self.pipelines.keys().cloned().collect()
    }
}

/// Preprocessing Cache: Cache intermediate results for performance
pub struct PreprocessingCache {
    cache: HashMap<String, Value>,
    hit_count: HashMap<String, usize>,
    miss_count: HashMap<String, usize>,
}

impl PreprocessingCache {
    /// Create new cache
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hit_count: HashMap::new(),
            miss_count: HashMap::new(),
        }
    }
    
    /// Get cached value if available
    pub fn get(&mut self, key: &str) -> Option<Value> {
        if let Some(value) = self.cache.get(key) {
            *self.hit_count.entry(key.to_string()).or_insert(0) += 1;
            Some(value.clone())
        } else {
            *self.miss_count.entry(key.to_string()).or_insert(0) += 1;
            None
        }
    }
    
    /// Store value in cache
    pub fn put(&mut self, key: String, value: Value) {
        self.cache.insert(key, value);
    }
    
    /// Clear all cached values
    pub fn clear(&mut self) {
        self.cache.clear();
        self.hit_count.clear();
        self.miss_count.clear();
    }
    
    /// Get cache statistics
    pub fn stats(&self) -> HashMap<String, Value> {
        let mut stats = HashMap::new();
        
        let total_hits: usize = self.hit_count.values().sum();
        let total_misses: usize = self.miss_count.values().sum();
        let total_requests = total_hits + total_misses;
        
        stats.insert("total_entries".to_string(), Value::Integer(self.cache.len() as i64));
        stats.insert("hit_count".to_string(), Value::Integer(total_hits as i64));
        stats.insert("miss_count".to_string(), Value::Integer(total_misses as i64));
        stats.insert("hit_rate".to_string(), Value::Real(
            if total_requests > 0 { total_hits as f64 / total_requests as f64 } else { 0.0 }
        ));
        
        stats
    }
}

/// Enhanced AutoPreprocessor with advanced pipeline capabilities
pub struct EnhancedAutoPreprocessor {
    base_preprocessor: AutoPreprocessor,
    advanced_pipeline: Option<AdvancedPreprocessingPipeline>,
    cache: PreprocessingCache,
    registry: PipelineRegistry,
}

impl EnhancedAutoPreprocessor {
    /// Create new enhanced preprocessor
    pub fn new() -> Self {
        Self {
            base_preprocessor: AutoPreprocessor::new(),
            advanced_pipeline: None,
            cache: PreprocessingCache::new(),
            registry: PipelineRegistry::create_standard_pipelines(),
        }
    }
    
    /// Create from existing AutoPreprocessor
    pub fn from_auto_preprocessor(auto_preprocessor: AutoPreprocessor) -> Self {
        Self {
            base_preprocessor: auto_preprocessor,
            advanced_pipeline: None,
            cache: PreprocessingCache::new(),
            registry: PipelineRegistry::create_standard_pipelines(),
        }
    }
    
    /// Set a registered pipeline by name
    pub fn use_pipeline(&mut self, pipeline_name: &str) -> MLResult<()> {
        if let Some(pipeline) = self.registry.get(pipeline_name) {
            // Create a debug-compatible clone using the registry
            let cloned_pipeline = PipelineBuilder::new(pipeline.name.clone())
                .with_config(pipeline.config.clone())
                .build();
                
            self.advanced_pipeline = Some(cloned_pipeline);
            Ok(())
        } else {
            Err(MLError::DataError {
                reason: format!("Pipeline '{}' not found in registry", pipeline_name),
            })
        }
    }
    
    /// Create and use custom advanced pipeline
    pub fn use_custom_pipeline(&mut self, pipeline: AdvancedPreprocessingPipeline) {
        self.advanced_pipeline = Some(pipeline);
    }
    
    /// Process data using advanced pipeline or fallback to base preprocessor
    pub fn process(&mut self, data: &Value) -> MLResult<Value> {
        // Generate cache key based on data characteristics
        let cache_key = self.generate_cache_key(data);
        
        // Check cache first
        if let Some(cached_result) = self.cache.get(&cache_key) {
            return Ok(cached_result);
        }
        
        // Process using advanced pipeline if available
        let result = if let Some(ref pipeline) = self.advanced_pipeline {
            pipeline.execute(data)?
        } else {
            // Fallback to base AutoPreprocessor
            self.base_preprocessor.preprocess(data)?
        };
        
        // Cache the result
        self.cache.put(cache_key, result.clone());
        
        Ok(result)
    }
    
    /// Generate cache key based on data characteristics
    fn generate_cache_key(&self, data: &Value) -> String {
        match data {
            Value::List(elements) => {
                let type_signature = elements.iter()
                    .map(|v| match v {
                        Value::Real(_) => "R",
                        Value::Integer(_) => "I", 
                        Value::String(_) => "S",
                        Value::Symbol(_) => "Y",
                        Value::Missing => "M",
                        _ => "O",
                    })
                    .collect::<Vec<_>>()
                    .join("");
                    
                format!("list_{}_{}", elements.len(), type_signature)
            },
            Value::Real(_) => "real".to_string(),
            Value::Integer(_) => "integer".to_string(),
            Value::String(_) => "string".to_string(),
            _ => "other".to_string(),
        }
    }
    
    /// Get preprocessing statistics
    pub fn stats(&self) -> HashMap<String, Value> {
        let mut stats = self.cache.stats();
        
        // Add pipeline information
        if let Some(ref pipeline) = self.advanced_pipeline {
            stats.insert("pipeline_name".to_string(), Value::String(pipeline.name.clone()));
            stats.insert("pipeline_steps".to_string(), Value::Integer(pipeline.num_steps() as i64));
        } else {
            stats.insert("pipeline_name".to_string(), Value::String("auto".to_string()));
        }
        
        stats.insert("available_pipelines".to_string(), 
                    Value::List(self.registry.list_pipelines().iter().map(|s| Value::String(s.clone())).collect()));
        
        stats
    }
    
    /// Clear all cached preprocessing results
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Factory for creating common preprocessing scenarios
pub struct PreprocessingFactory;

impl PreprocessingFactory {
    /// Create preprocessing for computer vision data
    pub fn for_computer_vision() -> AdvancedPreprocessingPipeline {
        PipelineBuilder::new("ComputerVision".to_string())
            .add_custom_step(
                "Normalization".to_string(),
                Box::new(StandardScaler::new()),
                Some(Box::new(NumericDataCondition))
            )
            .add_for_large_data(
                "OutlierRemoval".to_string(),
                Box::new(OutlierRemover::new(OutlierMethod::IQR, 2.0)),
                10000 // Only for large image datasets
            )
            .build()
    }
    
    /// Create preprocessing for natural language processing
    pub fn for_nlp() -> AdvancedPreprocessingPipeline {
        PipelineBuilder::new("NLP".to_string())
            .add_custom_step(
                "TextEncoding".to_string(),
                Box::new(OneHotEncoder::new()),
                Some(Box::new(CategoricalDataCondition))
            )
            .build()
    }
    
    /// Create preprocessing for time series data
    pub fn for_time_series() -> AdvancedPreprocessingPipeline {
        PipelineBuilder::new("TimeSeries".to_string())
            .handle_missing(ImputationStrategy::Mean)
            .add_custom_step(
                "Normalization".to_string(),
                Box::new(StandardScaler::new()),
                Some(Box::new(NumericDataCondition))
            )
            .build()
    }
    
    /// Create preprocessing for tabular data
    pub fn for_tabular_data() -> AdvancedPreprocessingPipeline {
        PipelineBuilder::new("Tabular".to_string())
            .handle_missing(ImputationStrategy::Mean)
            .remove_outliers(OutlierMethod::IQR, 1.5)
            .one_hot_encode()
            .standard_scale()
            .build()
    }
    
    /// Create minimal preprocessing for fast prototyping
    pub fn for_prototyping() -> AdvancedPreprocessingPipeline {
        PipelineBuilder::new("Prototyping".to_string())
            .standard_scale()
            .build()
    }
}

/// Convert preprocessed Value back to Tensor for ML training
pub fn preprocessed_value_to_tensor(value: &Value) -> MLResult<Tensor> {
    match value {
        Value::List(elements) => {
            let dual_values: Result<Vec<Dual>, MLError> = elements.iter()
                .map(|v| match v {
                    Value::Real(n) => Ok(Dual::variable(*n)),
                    Value::Integer(n) => Ok(Dual::variable(*n as f64)),
                    Value::List(nested) => {
                        // Handle nested lists (e.g., from one-hot encoding)
                        match nested.first() {
                            Some(Value::Real(n)) => Ok(Dual::variable(*n)),
                            Some(Value::Integer(n)) => Ok(Dual::variable(*n as f64)),
                            _ => Err(MLError::DataError {
                                reason: "Nested preprocessing result contains non-numeric data".to_string(),
                            }),
                        }
                    },
                    _ => Err(MLError::DataError {
                        reason: format!("Cannot convert preprocessed value to tensor: {:?}", v),
                    }),
                })
                .collect();
            
            let data = dual_values?;
            let shape = vec![data.len()];
            Tensor::new(data, shape)
        },
        Value::Real(n) => {
            let data = vec![Dual::variable(*n)];
            let shape = vec![1];
            Tensor::new(data, shape)
        },
        Value::Integer(n) => {
            let data = vec![Dual::variable(*n as f64)];
            let shape = vec![1];
            Tensor::new(data, shape)
        },
        _ => Err(MLError::DataError {
            reason: format!("Cannot convert preprocessed value to tensor: {:?}", value),
        }),
    }
}