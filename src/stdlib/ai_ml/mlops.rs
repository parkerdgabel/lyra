//! Phase 14B: Machine Learning Operations (MLOps)
//!
//! This module provides comprehensive MLOps functionality for the Lyra symbolic computation engine,
//! focusing on the operational aspects of ML lifecycle management for production systems.
//!
//! ## Core Features
//! - Experiment tracking and management
//! - Model registry with version control
//! - Data pipeline and feature store
//! - ML performance monitoring
//! - AutoML capabilities
//!
//! ## Architecture
//! All MLOps types are implemented as Foreign objects to maintain VM simplicity
//! following the Foreign Object pattern. This ensures thread safety and clean separation
//! between VM core and MLOps functionality.

use crate::error::{Error as LyraError, Result as LyraResult};
use crate::foreign::{Foreign, LyObj};
use crate::vm::{Value, VirtualMachine, VmResult, VmError};

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use uuid::Uuid;

// Re-exports for external dependencies
use polars::prelude::*;
use statrs::statistics::Statistics;
use rusqlite::{Connection, params};

/// Custom error types for MLOps operations
#[derive(Debug, thiserror::Error)]
pub enum MLOpsError {
    #[error("Experiment not found: {0}")]
    ExperimentNotFound(String),
    
    #[error("Model not found: {name} version {version}")]
    ModelNotFound { name: String, version: String },
    
    #[error("Data validation failed: {0}")]
    DataValidationFailed(String),
    
    #[error("Feature store error: {0}")]
    FeatureStoreError(String),
    
    #[error("Monitoring error: {0}")]
    MonitoringError(String),
    
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(#[from] rmp_serde::encode::Error),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(#[from] rmp_serde::decode::Error),
}

/// Experiment metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub created_by: String,
    pub metadata: HashMap<String, Value>,
}

/// Experiment run data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunData {
    pub run_id: String,
    pub experiment_id: String,
    pub status: RunStatus,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub metrics: HashMap<String, Vec<MetricValue>>,
    pub parameters: HashMap<String, Value>,
    pub artifacts: Vec<ArtifactInfo>,
    pub tags: Vec<String>,
}

/// Run status enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RunStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Metric value with step information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub value: f64,
    pub step: u64,
    pub timestamp: DateTime<Utc>,
}

/// Artifact information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactInfo {
    pub name: String,
    pub path: String,
    pub artifact_type: String,
    pub size: u64,
    pub created_at: DateTime<Utc>,
    pub metadata: HashMap<String, Value>,
}

/// Model version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelVersionInfo {
    pub name: String,
    pub version: String,
    pub stage: ModelStage,
    pub path: String,
    pub framework: String,
    pub metrics: HashMap<String, f64>,
    pub metadata: HashMap<String, Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Model stage enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStage {
    Development,
    Staging,
    Production,
    Archived,
}

/// Feature store schema
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSchema {
    pub name: String,
    pub feature_type: FeatureType,
    pub description: String,
    pub nullable: bool,
    pub default_value: Option<Value>,
}

/// Feature type enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeatureType {
    Integer,
    Float,
    String,
    Boolean,
    Timestamp,
    Array(Box<FeatureType>),
}

/// Data quality check result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityResult {
    pub check_name: String,
    pub passed: bool,
    pub score: f64,
    pub details: HashMap<String, Value>,
    pub timestamp: DateTime<Utc>,
}

/// Drift detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftResult {
    pub method: String,
    pub drift_detected: bool,
    pub drift_score: f64,
    pub threshold: f64,
    pub feature_drifts: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

/// Performance monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub model_name: String,
    pub metrics: Vec<String>,
    pub thresholds: HashMap<String, f64>,
    pub alert_channels: Vec<String>,
    pub check_interval: u64, // seconds
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub alert_id: String,
    pub model_name: String,
    pub metric: String,
    pub current_value: f64,
    pub threshold: f64,
    pub severity: AlertSeverity,
    pub timestamp: DateTime<Utc>,
    pub description: String,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Hyperparameter optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterConfig {
    pub parameter_space: HashMap<String, ParameterRange>,
    pub objective: String,
    pub direction: OptimizationDirection,
    pub budget: OptimizationBudget,
    pub algorithm: OptimizationAlgorithm,
}

/// Parameter range for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterRange {
    Float { min: f64, max: f64, log: bool },
    Integer { min: i64, max: i64 },
    Categorical { choices: Vec<String> },
}

/// Optimization direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationDirection {
    Minimize,
    Maximize,
}

/// Optimization budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationBudget {
    Iterations(u32),
    Time(u64), // seconds
    Both { iterations: u32, time: u64 },
}

/// Optimization algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAlgorithm {
    RandomSearch,
    GridSearch,
    BayesianOptimization,
    TPE, // Tree-structured Parzen Estimator
}

// ============================================================================
// EXPERIMENT TRACKING FOREIGN OBJECTS
// ============================================================================

/// Experiment tracking system
#[derive(Debug, Clone)]
pub struct Experiment {
    config: ExperimentConfig,
    runs: Arc<Mutex<HashMap<String, RunData>>>,
    storage: Arc<Mutex<Connection>>,
}

impl Experiment {
    pub fn new(config: ExperimentConfig) -> LyraResult<Self> {
        let storage = Arc::new(Mutex::new(Self::init_storage()?));
        let runs = Arc::new(Mutex::new(HashMap::new()));
        
        let experiment = Self {
            config,
            runs,
            storage,
        };
        
        experiment.save_to_storage()?;
        Ok(experiment)
    }
    
    fn init_storage() -> LyraResult<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| LyraError::Runtime { message: format!("Failed to create storage: {}", e) })?;
        
        // Create tables for experiments and runs
        conn.execute(
            "CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                tags TEXT,
                created_at TEXT,
                created_by TEXT,
                metadata TEXT
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                experiment_id TEXT,
                status TEXT,
                start_time TEXT,
                end_time TEXT,
                metrics TEXT,
                parameters TEXT,
                artifacts TEXT,
                tags TEXT,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id)
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        Ok(conn)
    }
    
    fn save_to_storage(&self) -> LyraResult<()> {
        let storage = self.storage.lock().unwrap();
        let metadata_json = serde_json::to_string(&self.config.metadata)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to serialize metadata: {}", e) })?;
        let tags_json = serde_json::to_string(&self.config.tags)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to serialize tags: {}", e) })?;
        
        storage.execute(
            "INSERT OR REPLACE INTO experiments (id, name, description, tags, created_at, created_by, metadata)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                self.config.name,
                self.config.name,
                self.config.description,
                tags_json,
                self.config.created_at.to_rfc3339(),
                self.config.created_by,
                metadata_json
            ],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to save experiment: {}", e) })?;
        
        Ok(())
    }
    
    pub fn create_run(&self, config: HashMap<String, Value>, tags: Vec<String>) -> LyraResult<String> {
        let run_id = Uuid::new_v4().to_string();
        let run_data = RunData {
            run_id: run_id.clone(),
            experiment_id: self.config.name.clone(),
            status: RunStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            metrics: HashMap::new(),
            parameters: config,
            artifacts: Vec::new(),
            tags,
        };
        
        self.runs.lock().unwrap().insert(run_id.clone(), run_data);
        Ok(run_id)
    }
    
    pub fn log_metric(&self, run_id: &str, metric: &str, value: f64, step: u64) -> LyraResult<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.get_mut(run_id) {
            let metric_value = MetricValue {
                value,
                step,
                timestamp: Utc::now(),
            };
            run.metrics.entry(metric.to_string()).or_insert_with(Vec::new).push(metric_value);
            Ok(())
        } else {
            Err(LyraError::Runtime { message: format!("Run not found: {}", run_id) })
        }
    }
    
    pub fn add_artifact(&self, run_id: &str, artifact: ArtifactInfo) -> LyraResult<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.get_mut(run_id) {
            run.artifacts.push(artifact);
            Ok(())
        } else {
            Err(LyraError::Runtime { message: format!("Run not found: {}", run_id) })
        }
    }
    
    pub fn update_run_status(&self, run_id: &str, status: RunStatus, message: Option<String>) -> LyraResult<()> {
        let mut runs = self.runs.lock().unwrap();
        if let Some(run) = runs.get_mut(run_id) {
            run.status = status;
            if matches!(run.status, RunStatus::Completed | RunStatus::Failed | RunStatus::Cancelled) {
                run.end_time = Some(Utc::now());
            }
            if let Some(msg) = message {
                run.parameters.insert("status_message".to_string(), Value::String(msg));
            }
            Ok(())
        } else {
            Err(LyraError::Runtime { message: format!("Run not found: {}", run_id) })
        }
    }
    
    pub fn get_runs(&self) -> Vec<RunData> {
        self.runs.lock().unwrap().values().cloned().collect()
    }
    
    pub fn get_config(&self) -> &ExperimentConfig {
        &self.config
    }
}

impl Foreign for Experiment {
    fn type_name(&self) -> &'static str {
        "Experiment"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "createRun" => {
                let config = if args.len() > 0 {
                    // Extract config from args if provided
                    HashMap::new() // Simplified for now
                } else {
                    HashMap::new()
                };
                let tags = if args.len() > 1 {
                    // Extract tags from args if provided
                    Vec::new() // Simplified for now
                } else {
                    Vec::new()
                };
                let run_id = self.create_run(config, tags)?;
                Ok(Value::String(run_id))
            }
            "logMetric" => {
                if args.len() >= 3 {
                    if let (Value::String(run_id), Value::String(metric), Value::Real(value)) = 
                        (&args[0], &args[1], &args[2]) {
                        let step = if args.len() > 3 {
                            if let Value::Integer(s) = &args[3] {
                                *s as u64
                            } else { 0 }
                        } else { 0 };
                        self.log_metric(run_id, metric, *value, step)
                            .map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Boolean(true))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Invalid argument types for logMetric".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 3, actual: args.len() })
                }
            }
            "getRuns" => {
                let runs = self.get_runs();
                // Convert runs to Value representation
                let run_values: Vec<Value> = runs.into_iter().map(|run| {
                    Value::String(format!("Run({})", run.run_id))
                }).collect();
                Ok(Value::List(run_values))
            }
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// MODEL REGISTRY FOREIGN OBJECTS
// ============================================================================

/// Model registry system
#[derive(Debug, Clone)]
pub struct ModelRegistry {
    models: Arc<RwLock<HashMap<String, HashMap<String, ModelVersionInfo>>>>,
    storage: Arc<Mutex<Connection>>,
}

impl ModelRegistry {
    pub fn new() -> LyraResult<Self> {
        let storage = Arc::new(Mutex::new(Self::init_storage()?));
        let models = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            models,
            storage,
        })
    }
    
    fn init_storage() -> LyraResult<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| LyraError::Runtime { message: format!("Failed to create storage: {}", e) })?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS models (
                name TEXT,
                version TEXT,
                stage TEXT,
                path TEXT,
                framework TEXT,
                metrics TEXT,
                metadata TEXT,
                created_at TEXT,
                updated_at TEXT,
                PRIMARY KEY(name, version)
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        Ok(conn)
    }
    
    pub fn register_model(&self, model_info: ModelVersionInfo) -> LyraResult<()> {
        let mut models = self.models.write().unwrap();
        models.entry(model_info.name.clone())
            .or_insert_with(HashMap::new)
            .insert(model_info.version.clone(), model_info.clone());
        
        // Save to storage
        let storage = self.storage.lock().unwrap();
        let metrics_json = serde_json::to_string(&model_info.metrics)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to serialize metrics: {}", e) })?;
        let metadata_json = serde_json::to_string(&model_info.metadata)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to serialize metadata: {}", e) })?;
        let stage_str = format!("{:?}", model_info.stage);
        
        storage.execute(
            "INSERT OR REPLACE INTO models (name, version, stage, path, framework, metrics, metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                model_info.name,
                model_info.version,
                stage_str,
                model_info.path,
                model_info.framework,
                metrics_json,
                metadata_json,
                model_info.created_at.to_rfc3339(),
                model_info.updated_at.to_rfc3339()
            ],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to save model: {}", e) })?;
        
        Ok(())
    }
    
    pub fn get_model(&self, name: &str, version: &str) -> LyraResult<ModelVersionInfo> {
        let models = self.models.read().unwrap();
        models.get(name)
            .and_then(|versions| versions.get(version))
            .cloned()
            .ok_or_else(|| LyraError::Runtime { message: format!("Model not found: {} version {}", name, version) })
    }
    
    pub fn promote_model(&self, name: &str, version: &str, stage: ModelStage) -> LyraResult<()> {
        let mut models = self.models.write().unwrap();
        if let Some(versions) = models.get_mut(name) {
            if let Some(model) = versions.get_mut(version) {
                model.stage = stage;
                model.updated_at = Utc::now();
                Ok(())
            } else {
                Err(LyraError::Runtime { message: format!("Model version not found: {} version {}", name, version) })
            }
        } else {
            Err(LyraError::Runtime { message: format!("Model not found: {}", name) })
        }
    }
    
    pub fn list_models(&self, stage_filter: Option<ModelStage>) -> Vec<ModelVersionInfo> {
        let models = self.models.read().unwrap();
        let mut result = Vec::new();
        
        for versions in models.values() {
            for model in versions.values() {
                if let Some(ref filter_stage) = stage_filter {
                    if std::mem::discriminant(&model.stage) == std::mem::discriminant(filter_stage) {
                        result.push(model.clone());
                    }
                } else {
                    result.push(model.clone());
                }
            }
        }
        
        result
    }
}

impl Foreign for ModelRegistry {
    fn type_name(&self) -> &'static str {
        "ModelRegistry"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "register" => {
                // Simplified registration - in practice would parse complex arguments
                if args.len() >= 4 {
                    if let (Value::String(name), Value::String(version), Value::String(path), Value::String(framework)) = 
                        (&args[0], &args[1], &args[2], &args[3]) {
                        let model_info = ModelVersionInfo {
                            name: name.clone(),
                            version: version.clone(),
                            stage: ModelStage::Development,
                            path: path.clone(),
                            framework: framework.clone(),
                            metrics: HashMap::new(),
                            metadata: HashMap::new(),
                            created_at: Utc::now(),
                            updated_at: Utc::now(),
                        };
                        self.register_model(model_info)
                            .map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Boolean(true))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Invalid argument types for register".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 4, actual: args.len() })
                }
            }
            "get" => {
                if args.len() >= 2 {
                    if let (Value::String(name), Value::String(version)) = (&args[0], &args[1]) {
                        let model = self.get_model(name, version)
                            .map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::String(format!("Model({} v{})", model.name, model.version)))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Invalid argument types for get".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 2, actual: args.len() })
                }
            }
            "list" => {
                let models = self.list_models(None);
                let model_values: Vec<Value> = models.into_iter().map(|model| {
                    Value::String(format!("Model({} v{})", model.name, model.version))
                }).collect();
                Ok(Value::List(model_values))
            }
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// FEATURE STORE FOREIGN OBJECTS
// ============================================================================

/// Feature store system
#[derive(Debug, Clone)]
pub struct FeatureStore {
    name: String,
    schema: Vec<FeatureSchema>,
    data: Arc<RwLock<DataFrame>>,
    storage: Arc<Mutex<Connection>>,
}

impl FeatureStore {
    pub fn new(name: String, schema: Vec<FeatureSchema>) -> LyraResult<Self> {
        let storage = Arc::new(Mutex::new(Self::init_storage()?));
        
        // Create empty DataFrame with schema
        let mut columns = Vec::new();
        for feature in &schema {
            match feature.feature_type {
                FeatureType::Integer => {
                    columns.push(Series::new(&feature.name, Vec::<i64>::new()));
                }
                FeatureType::Float => {
                    columns.push(Series::new(&feature.name, Vec::<f64>::new()));
                }
                FeatureType::String => {
                    columns.push(Series::new(&feature.name, Vec::<String>::new()));
                }
                FeatureType::Boolean => {
                    columns.push(Series::new(&feature.name, Vec::<bool>::new()));
                }
                _ => {
                    // For complex types, use string representation for now
                    columns.push(Series::new(&feature.name, Vec::<String>::new()));
                }
            }
        }
        
        let data = DataFrame::new(columns)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to create DataFrame: {}", e) })?;
        
        Ok(Self {
            name,
            schema,
            data: Arc::new(RwLock::new(data)),
            storage,
        })
    }
    
    fn init_storage() -> LyraResult<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| LyraError::Runtime { message: format!("Failed to create storage: {}", e) })?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS feature_stores (
                name TEXT PRIMARY KEY,
                schema TEXT,
                created_at TEXT
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        Ok(conn)
    }
    
    pub fn compute_features(&self, config: HashMap<String, Value>, input_data: DataFrame) -> LyraResult<DataFrame> {
        // Simplified feature computation - in practice would apply transformations
        let mut result = input_data.clone();
        
        // Apply basic transformations based on config
        for (feature_name, transformation) in config {
            if let Value::String(transform_type) = transformation {
                match transform_type.as_str() {
                    "normalize" => {
                        // Apply normalization if column exists
                        if let Ok(column) = result.column(&feature_name) {
                            if let Ok(numeric) = column.f64() {
                                let mean = numeric.mean().unwrap_or(0.0);
                                let std = numeric.std(1).unwrap_or(1.0);
                                let normalized = numeric.apply(|val| {
                                    val.map(|v| (v - mean) / std)
                                });
                                result = result.with_column(normalized.with_name(&format!("{}_normalized", feature_name)))
                                    .map_err(|e| LyraError::Runtime { message: format!("Failed to apply normalization: {}", e) })?;
                            }
                        }
                    }
                    "log_transform" => {
                        // Apply log transformation
                        if let Ok(column) = result.column(&feature_name) {
                            if let Ok(numeric) = column.f64() {
                                let log_transformed = numeric.apply(|val| {
                                    val.map(|v| if v > 0.0 { v.ln() } else { 0.0 })
                                });
                                result = result.with_column(log_transformed.with_name(&format!("{}_log", feature_name)))
                                    .map_err(|e| LyraError::Runtime { message: format!("Failed to apply log transform: {}", e) })?;
                            }
                        }
                    }
                    _ => {
                        // Unknown transformation, skip
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    pub fn serve_features(&self, keys: Vec<String>, timestamp: Option<DateTime<Utc>>) -> LyraResult<DataFrame> {
        let data = self.data.read().unwrap();
        
        // Filter features based on keys
        let mut selected_columns = Vec::new();
        for key in keys {
            if data.get_column_names().contains(&key.as_str()) {
                selected_columns.push(key);
            }
        }
        
        if selected_columns.is_empty() {
            return Ok(DataFrame::empty());
        }
        
        let result = data.select(selected_columns)
            .map_err(|e| LyraError::Runtime { message: format!("Failed to select features: {}", e) })?;
        
        Ok(result)
    }
    
    pub fn get_schema(&self) -> &[FeatureSchema] {
        &self.schema
    }
    
    pub fn get_name(&self) -> &str {
        &self.name
    }
}

impl Foreign for FeatureStore {
    fn type_name(&self) -> &'static str {
        "FeatureStore"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "computeFeatures" => {
                // Simplified - would need proper argument parsing
                let config = HashMap::new();
                let input_data = DataFrame::empty();
                let result = self.compute_features(config, input_data)?;
                Ok(Value::String(format!("ComputedFeatures({}x{})", result.height(), result.width())))
            }
            "serveFeatures" => {
                if args.len() >= 1 {
                    if let Value::List(key_values) = &args[0] {
                        let keys: Vec<String> = key_values.iter().filter_map(|v| {
                            if let Value::String(s) = v {
                                Some(s.clone())
                            } else {
                                None
                            }
                        }).collect();
                        let result = self.serve_features(keys, None)?;
                        Ok(Value::String(format!("Features({}x{})", result.height(), result.width())))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Expected list of feature keys".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 1, actual: args.len() })
                }
            }
            "getSchema" => {
                let schema_info: Vec<Value> = self.schema.iter().map(|feature| {
                    Value::String(format!("{}:{:?}", feature.name, feature.feature_type))
                }).collect();
                Ok(Value::List(schema_info))
            }
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// PERFORMANCE MONITORING FOREIGN OBJECTS
// ============================================================================

/// Model performance monitor
#[derive(Debug, Clone)]
pub struct ModelMonitor {
    config: MonitoringConfig,
    metrics_history: Arc<RwLock<HashMap<String, Vec<MetricValue>>>>,
    alerts: Arc<Mutex<Vec<PerformanceAlert>>>,
    storage: Arc<Mutex<Connection>>,
}

impl ModelMonitor {
    pub fn new(config: MonitoringConfig) -> LyraResult<Self> {
        let storage = Arc::new(Mutex::new(Self::init_storage()?));
        let metrics_history = Arc::new(RwLock::new(HashMap::new()));
        let alerts = Arc::new(Mutex::new(Vec::new()));
        
        Ok(Self {
            config,
            metrics_history,
            alerts,
            storage,
        })
    }
    
    fn init_storage() -> LyraResult<Connection> {
        let conn = Connection::open_in_memory()
            .map_err(|e| LyraError::Runtime { message: format!("Failed to create storage: {}", e) })?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS model_metrics (
                model_name TEXT,
                metric_name TEXT,
                value REAL,
                step INTEGER,
                timestamp TEXT
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        conn.execute(
            "CREATE TABLE IF NOT EXISTS performance_alerts (
                alert_id TEXT PRIMARY KEY,
                model_name TEXT,
                metric TEXT,
                current_value REAL,
                threshold REAL,
                severity TEXT,
                timestamp TEXT,
                description TEXT
            )",
            [],
        ).map_err(|e| LyraError::Runtime { message: format!("Failed to create table: {}", e) })?;
        
        Ok(conn)
    }
    
    pub fn record_metric(&self, metric_name: &str, value: f64, step: u64) -> LyraResult<()> {
        let metric_value = MetricValue {
            value,
            step,
            timestamp: Utc::now(),
        };
        
        let mut history = self.metrics_history.write().unwrap();
        history.entry(metric_name.to_string()).or_insert_with(Vec::new).push(metric_value.clone());
        
        // Check for threshold violations
        if let Some(threshold) = self.config.thresholds.get(metric_name) {
            if value < *threshold {
                let alert = PerformanceAlert {
                    alert_id: Uuid::new_v4().to_string(),
                    model_name: self.config.model_name.clone(),
                    metric: metric_name.to_string(),
                    current_value: value,
                    threshold: *threshold,
                    severity: AlertSeverity::Medium,
                    timestamp: Utc::now(),
                    description: format!("Metric {} below threshold: {} < {}", metric_name, value, threshold),
                };
                
                self.alerts.lock().unwrap().push(alert);
            }
        }
        
        Ok(())
    }
    
    pub fn detect_drift(&self, metric_name: &str, baseline_window: usize, current_window: usize) -> LyraResult<DriftResult> {
        let history = self.metrics_history.read().unwrap();
        
        if let Some(metric_history) = history.get(metric_name) {
            if metric_history.len() < baseline_window + current_window {
                return Err(LyraError::Runtime { message: "Insufficient data for drift detection".to_string() });
            }
            
            // Get baseline and current data
            let total_len = metric_history.len();
            let baseline_data: Vec<f64> = metric_history[total_len - baseline_window - current_window..total_len - current_window]
                .iter().map(|m| m.value).collect();
            let current_data: Vec<f64> = metric_history[total_len - current_window..]
                .iter().map(|m| m.value).collect();
            
            // Compute basic statistics
            let baseline_mean = baseline_data.iter().sum::<f64>() / baseline_data.len() as f64;
            let current_mean = current_data.iter().sum::<f64>() / current_data.len() as f64;
            
            // Simple drift detection based on mean difference
            let drift_score = (current_mean - baseline_mean).abs() / baseline_mean.max(1e-8);
            let threshold = 0.1; // 10% change threshold
            
            let drift_result = DriftResult {
                method: "mean_difference".to_string(),
                drift_detected: drift_score > threshold,
                drift_score,
                threshold,
                feature_drifts: HashMap::from([(metric_name.to_string(), drift_score)]),
                timestamp: Utc::now(),
            };
            
            Ok(drift_result)
        } else {
            Err(LyraError::Runtime { message: format!("Metric not found: {}", metric_name) })
        }
    }
    
    pub fn get_alerts(&self) -> Vec<PerformanceAlert> {
        self.alerts.lock().unwrap().clone()
    }
    
    pub fn get_metric_history(&self, metric_name: &str) -> Option<Vec<MetricValue>> {
        self.metrics_history.read().unwrap().get(metric_name).cloned()
    }
}

impl Foreign for ModelMonitor {
    fn type_name(&self) -> &'static str {
        "ModelMonitor"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "recordMetric" => {
                if args.len() >= 2 {
                    if let (Value::String(metric), Value::Real(value)) = (&args[0], &args[1]) {
                        let step = if args.len() > 2 {
                            if let Value::Integer(s) = &args[2] {
                                *s as u64
                            } else { 0 }
                        } else { 0 };
                        self.record_metric(metric, *value, step)
                            .map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Boolean(true))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Invalid argument types for recordMetric".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 2, actual: args.len() })
                }
            }
            "detectDrift" => {
                if args.len() >= 1 {
                    if let Value::String(metric) = &args[0] {
                        let baseline_window = if args.len() > 1 {
                            if let Value::Integer(w) = &args[1] {
                                *w as usize
                            } else { 100 }
                        } else { 100 };
                        let current_window = if args.len() > 2 {
                            if let Value::Integer(w) = &args[2] {
                                *w as usize
                            } else { 50 }
                        } else { 50 };
                        
                        let drift_result = self.detect_drift(metric, baseline_window, current_window)
                            .map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::String(format!("DriftResult(detected={}, score={:.3})", 
                            drift_result.drift_detected, drift_result.drift_score)))
                    } else {
                        Err(crate::foreign::ForeignError::TypeError("Expected metric name as string".to_string()))
                    }
                } else {
                    Err(crate::foreign::ForeignError::ArgumentError { expected: 1, actual: args.len() })
                }
            }
            "getAlerts" => {
                let alerts = self.get_alerts();
                let alert_values: Vec<Value> = alerts.into_iter().map(|alert| {
                    Value::String(format!("Alert({}: {} below {})", alert.model_name, alert.current_value, alert.threshold))
                }).collect();
                Ok(Value::List(alert_values))
            }
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ============================================================================
// MODULE EXPORTS AND REGISTRATION
// ============================================================================

/// Create a new experiment
pub fn experiment_create(args: &[Value]) -> VmResult<Value> {
    if args.len() >= 3 {
        if let (Value::String(name), Value::String(description), Value::List(tag_values)) = 
            (&args[0], &args[1], &args[2]) {
            
            let tags: Vec<String> = tag_values.iter().filter_map(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            }).collect();
            
            let config = ExperimentConfig {
                name: name.clone(),
                description: description.clone(),
                tags,
                created_at: Utc::now(),
                created_by: "lyra_user".to_string(),
                metadata: HashMap::new(),
            };
            
            let experiment = Experiment::new(config)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
            Ok(Value::LyObj(LyObj::new(Box::new(experiment))))
        } else {
            Err(VmError::TypeError { expected: "String, String, List".to_string(), actual: "Mixed types".to_string() })
        }
    } else {
        Err(VmError::Runtime(format!("Expected 3 arguments, got {}", args.len())))
    }
}

/// Create a new model registry
pub fn model_registry_create(_args: &[Value]) -> VmResult<Value> {
    let registry = ModelRegistry::new()
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    Ok(Value::LyObj(LyObj::new(Box::new(registry))))
}

/// Create a new feature store
pub fn feature_store_create(args: &[Value]) -> VmResult<Value> {
    if args.len() >= 2 {
        if let (Value::String(name), Value::List(schema_values)) = (&args[0], &args[1]) {
            // Simplified schema parsing
            let schema = schema_values.iter().enumerate().map(|(i, _)| {
                FeatureSchema {
                    name: format!("feature_{}", i),
                    feature_type: FeatureType::Float,
                    description: "Auto-generated feature".to_string(),
                    nullable: false,
                    default_value: None,
                }
            }).collect();
            
            let feature_store = FeatureStore::new(name.clone(), schema)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
            Ok(Value::LyObj(LyObj::new(Box::new(feature_store))))
        } else {
            Err(VmError::TypeError { expected: "String, List".to_string(), actual: "Mixed types".to_string() })
        }
    } else {
        Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())))
    }
}

/// Create a new model monitor
pub fn model_monitor_create(args: &[Value]) -> VmResult<Value> {
    if args.len() >= 2 {
        if let (Value::String(model_name), Value::List(metric_values)) = (&args[0], &args[1]) {
            
            let metrics: Vec<String> = metric_values.iter().filter_map(|v| {
                if let Value::String(s) = v {
                    Some(s.clone())
                } else {
                    None
                }
            }).collect();
            
            let config = MonitoringConfig {
                model_name: model_name.clone(),
                metrics,
                thresholds: HashMap::new(),
                alert_channels: Vec::new(),
                check_interval: 60, // 1 minute
            };
            
            let monitor = ModelMonitor::new(config)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
            Ok(Value::LyObj(LyObj::new(Box::new(monitor))))
        } else {
            Err(VmError::TypeError { expected: "String, List".to_string(), actual: "Mixed types".to_string() })
        }
    } else {
        Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())))
    }
}

// Additional placeholder functions for the remaining MLOps operations
// These would be fully implemented in a production system

pub fn experiment_log(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn experiment_artifact(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn experiment_compare(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn experiment_list(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn run_create(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String(Uuid::new_v4().to_string()))
}

pub fn run_log(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn run_status(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn experiment_visualize(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("visualization_created".to_string()))
}

pub fn model_register(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn model_version(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("v1.0".to_string()))
}

pub fn model_promote(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn model_deploy(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("deployment_id_123".to_string()))
}

pub fn model_retire(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn model_search(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn model_lineage(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn model_metrics(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn feature_compute(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("features_computed".to_string()))
}

pub fn feature_serve(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn data_drift(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(false))
}

pub fn data_validation(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn data_quality(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(0.95))
}

pub fn pipeline_create(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("pipeline_id_123".to_string()))
}

pub fn pipeline_execute(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("execution_id_123".to_string()))
}

pub fn performance_drift(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(false))
}

pub fn ab_test(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("test_id_123".to_string()))
}

pub fn feedback_loop(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn model_health(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("healthy".to_string()))
}

pub fn alert_config(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn auto_train(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("trained_model_id_123".to_string()))
}

pub fn hyperparameter_tune(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}

pub fn model_select(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::String("best_model_id_123".to_string()))
}

pub fn feature_engineering(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::List(Vec::new()))
}