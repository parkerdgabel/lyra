//! MLOps Integration: Experiment Tracking and Model Registry
//!
//! This module provides MLOps capabilities including experiment tracking,
//! model versioning, artifact management, and deployment preparation.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::NetChain;
use crate::stdlib::ml::training::{TrainingConfig, TrainingResult};
use crate::stdlib::ml::automl::{AutoMLResult, ProblemType, ModelComplexity};
use crate::stdlib::ml::evaluation::{EvaluationResult, CrossValidationResult};
use crate::vm::Value;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};

/// Experiment tracking system for ML workflows
pub struct ExperimentTracker {
    experiments: HashMap<String, Experiment>,
    current_experiment: Option<String>,
    storage_backend: StorageBackend,
}

/// Storage backend for experiments and models
#[derive(Debug, Clone)]
pub enum StorageBackend {
    Memory,          // In-memory storage (default)
    FileSystem { base_path: String },
    Database { connection_string: String },
}

impl Default for StorageBackend {
    fn default() -> Self {
        StorageBackend::Memory
    }
}

impl ExperimentTracker {
    /// Create new experiment tracker
    pub fn new(storage_backend: StorageBackend) -> Self {
        Self {
            experiments: HashMap::new(),
            current_experiment: None,
            storage_backend,
        }
    }
    
    /// Start a new experiment
    pub fn start_experiment(
        &mut self,
        name: String,
        description: String,
        tags: Vec<String>,
    ) -> MLResult<String> {
        let experiment_id = self.generate_experiment_id(&name);
        
        let experiment = Experiment {
            id: experiment_id.clone(),
            name,
            description,
            tags,
            start_time: SystemTime::now(),
            end_time: None,
            status: ExperimentStatus::Running,
            hyperparameters: HashMap::new(),
            metrics: HashMap::new(),
            artifacts: Vec::new(),
            metadata: HashMap::new(),
            model_versions: Vec::new(),
        };
        
        self.experiments.insert(experiment_id.clone(), experiment);
        self.current_experiment = Some(experiment_id.clone());
        
        println!("Started experiment: {}", experiment_id);
        Ok(experiment_id)
    }
    
    /// Log hyperparameters for current experiment
    pub fn log_hyperparameters(&mut self, hyperparameters: HashMap<String, Value>) -> MLResult<()> {
        let experiment_id = self.get_current_experiment_id()?;
        
        if let Some(experiment) = self.experiments.get_mut(&experiment_id) {
            experiment.hyperparameters.extend(hyperparameters);
            Ok(())
        } else {
            Err(MLError::DataError {
                reason: format!("Experiment {} not found", experiment_id),
            })
        }
    }
    
    /// Log metrics for current experiment
    pub fn log_metrics(&mut self, metrics: HashMap<String, f64>) -> MLResult<()> {
        let experiment_id = self.get_current_experiment_id()?;
        
        if let Some(experiment) = self.experiments.get_mut(&experiment_id) {
            for (metric_name, value) in metrics {
                let step = experiment.metrics.len();
                experiment.metrics
                    .entry(metric_name)
                    .or_insert_with(Vec::new)
                    .push(MetricValue {
                        value,
                        timestamp: SystemTime::now(),
                        step,
                    });
            }
            Ok(())
        } else {
            Err(MLError::DataError {
                reason: format!("Experiment {} not found", experiment_id),
            })
        }
    }
    
    /// Log training results
    pub fn log_training_result(&mut self, training_result: &TrainingResult) -> MLResult<()> {
        let mut metrics = HashMap::new();
        metrics.insert("final_loss".to_string(), training_result.final_loss);
        metrics.insert("epochs_completed".to_string(), training_result.epochs_completed as f64);
        
        if let Some(&best_loss) = training_result.loss_history.iter().min_by(|a, b| a.partial_cmp(b).unwrap()) {
            metrics.insert("best_loss".to_string(), best_loss);
        }
        
        self.log_metrics(metrics)
    }
    
    /// Log evaluation results
    pub fn log_evaluation_result(&mut self, evaluation_result: &EvaluationResult) -> MLResult<()> {
        let mut metrics = HashMap::new();
        
        match evaluation_result {
            EvaluationResult::Classification(report) => {
                metrics.insert("accuracy".to_string(), report.accuracy);
                metrics.insert("precision".to_string(), report.precision);
                metrics.insert("recall".to_string(), report.recall);
                metrics.insert("f1_score".to_string(), report.f1_score);
            },
            EvaluationResult::Regression(report) => {
                metrics.insert("mse".to_string(), report.mean_squared_error);
                metrics.insert("mae".to_string(), report.mean_absolute_error);
                metrics.insert("rmse".to_string(), report.root_mean_squared_error);
                metrics.insert("r_squared".to_string(), report.r_squared);
            },
        }
        
        self.log_metrics(metrics)
    }
    
    /// Register a trained model with the current experiment
    pub fn register_model(
        &mut self,
        model: &NetChain,
        model_name: String,
        model_metadata: HashMap<String, Value>,
    ) -> MLResult<String> {
        let experiment_id = self.get_current_experiment_id()?;
        let model_version = self.generate_model_version(&model_name);
        
        let model_artifact = ModelArtifact {
            model_id: model_version.clone(),
            model_name: model_name.clone(),
            experiment_id: experiment_id.clone(),
            created_at: SystemTime::now(),
            model_type: ModelType::NetChain,
            metadata: model_metadata,
            file_path: None, // Would store actual file path in real implementation
            checksum: self.calculate_model_checksum(model)?,
            size_bytes: self.estimate_model_size(model),
        };
        
        if let Some(experiment) = self.experiments.get_mut(&experiment_id) {
            let size_bytes = model_artifact.size_bytes;
            experiment.model_versions.push(model_artifact);
            
            // Add artifact reference
            experiment.artifacts.push(ArtifactReference {
                artifact_type: ArtifactType::Model,
                artifact_id: model_version.clone(),
                file_path: format!("models/{}.model", model_version),
                size_bytes,
                created_at: SystemTime::now(),
            });
        }
        
        println!("Registered model: {} (version: {})", model_name, model_version);
        Ok(model_version)
    }
    
    /// End current experiment
    pub fn end_experiment(&mut self, status: ExperimentStatus) -> MLResult<ExperimentSummary> {
        let experiment_id = self.get_current_experiment_id()?;
        
        // Generate summary before modifying experiment
        let summary = if let Some(experiment) = self.experiments.get(&experiment_id) {
            let mut temp_experiment = experiment.clone();
            temp_experiment.end_time = Some(SystemTime::now());
            temp_experiment.status = status;
            self.generate_experiment_summary(&temp_experiment)?
        } else {
            return Err(MLError::DataError {
                reason: format!("Experiment {} not found", experiment_id),
            });
        };
        
        // Now modify the actual experiment
        if let Some(experiment) = self.experiments.get_mut(&experiment_id) {
            experiment.end_time = Some(SystemTime::now());
            experiment.status = status;
        }
        
        self.current_experiment = None;
        Ok(summary)
    }
    
    /// Get experiment by ID
    pub fn get_experiment(&self, experiment_id: &str) -> Option<&Experiment> {
        self.experiments.get(experiment_id)
    }
    
    /// List all experiments
    pub fn list_experiments(&self) -> Vec<&Experiment> {
        self.experiments.values().collect()
    }
    
    /// Search experiments by tags or metadata
    pub fn search_experiments(&self, query: &ExperimentQuery) -> Vec<&Experiment> {
        self.experiments.values()
            .filter(|exp| self.matches_query(exp, query))
            .collect()
    }
    
    /// Compare multiple experiments
    pub fn compare_experiments(&self, experiment_ids: &[String]) -> MLResult<ExperimentComparison> {
        let experiments: Result<Vec<&Experiment>, MLError> = experiment_ids.iter()
            .map(|id| self.experiments.get(id).ok_or_else(|| MLError::DataError {
                reason: format!("Experiment {} not found", id),
            }))
            .collect();
        
        let experiments = experiments?;
        
        Ok(ExperimentComparison {
            experiments: experiments.iter().map(|&exp| exp.clone()).collect(),
            comparison_metrics: self.extract_comparison_metrics(&experiments),
            best_experiment_id: self.find_best_experiment(&experiments)?,
        })
    }
    
    // Helper methods
    
    fn get_current_experiment_id(&self) -> MLResult<String> {
        self.current_experiment.clone().ok_or_else(|| MLError::DataError {
            reason: "No active experiment. Call start_experiment() first.".to_string(),
        })
    }
    
    fn generate_experiment_id(&self, name: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("{}_{}", name.replace(' ', "_").to_lowercase(), timestamp)
    }
    
    fn generate_model_version(&self, model_name: &str) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("{}_v{}", model_name.replace(' ', "_").to_lowercase(), timestamp)
    }
    
    fn calculate_model_checksum(&self, _model: &NetChain) -> MLResult<String> {
        // Simplified checksum calculation
        Ok(format!("checksum_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()))
    }
    
    fn estimate_model_size(&self, _model: &NetChain) -> usize {
        // Simplified size estimation
        1024 * 1024 // 1MB default
    }
    
    fn generate_experiment_summary(&self, experiment: &Experiment) -> MLResult<ExperimentSummary> {
        let duration = experiment.end_time
            .unwrap_or(SystemTime::now())
            .duration_since(experiment.start_time)
            .unwrap()
            .as_secs();
        
        // Extract final metrics
        let final_metrics: HashMap<String, f64> = experiment.metrics.iter()
            .map(|(name, values)| {
                let final_value = values.last().map(|v| v.value).unwrap_or(0.0);
                (name.clone(), final_value)
            })
            .collect();
        
        Ok(ExperimentSummary {
            experiment_id: experiment.id.clone(),
            name: experiment.name.clone(),
            status: experiment.status,
            duration_seconds: duration,
            final_metrics,
            model_count: experiment.model_versions.len(),
            artifact_count: experiment.artifacts.len(),
        })
    }
    
    fn matches_query(&self, experiment: &Experiment, query: &ExperimentQuery) -> bool {
        if let Some(ref name_filter) = query.name_contains {
            if !experiment.name.to_lowercase().contains(&name_filter.to_lowercase()) {
                return false;
            }
        }
        
        if let Some(ref tag_filter) = query.has_tag {
            if !experiment.tags.contains(tag_filter) {
                return false;
            }
        }
        
        if let Some(status_filter) = query.status {
            if experiment.status != status_filter {
                return false;
            }
        }
        
        true
    }
    
    fn extract_comparison_metrics(&self, experiments: &[&Experiment]) -> HashMap<String, Vec<f64>> {
        let mut comparison_metrics = HashMap::new();
        
        // Find common metrics across all experiments
        for experiment in experiments {
            for (metric_name, metric_values) in &experiment.metrics {
                if let Some(final_value) = metric_values.last() {
                    comparison_metrics
                        .entry(metric_name.clone())
                        .or_insert_with(Vec::new)
                        .push(final_value.value);
                }
            }
        }
        
        comparison_metrics
    }
    
    fn find_best_experiment(&self, experiments: &[&Experiment]) -> MLResult<String> {
        // Simple heuristic: find experiment with best final loss
        let best_experiment = experiments.iter()
            .max_by(|exp_a, exp_b| {
                let score_a = exp_a.metrics.get("accuracy")
                    .or_else(|| exp_a.metrics.get("r_squared"))
                    .and_then(|values| values.last())
                    .map(|v| v.value)
                    .unwrap_or(0.0);
                
                let score_b = exp_b.metrics.get("accuracy")
                    .or_else(|| exp_b.metrics.get("r_squared"))
                    .and_then(|values| values.last())
                    .map(|v| v.value)
                    .unwrap_or(0.0);
                
                score_a.partial_cmp(&score_b).unwrap()
            })
            .ok_or_else(|| MLError::DataError {
                reason: "No experiments to compare".to_string(),
            })?;
        
        Ok(best_experiment.id.clone())
    }
}

/// Individual experiment record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    pub start_time: SystemTime,
    pub end_time: Option<SystemTime>,
    pub status: ExperimentStatus,
    pub hyperparameters: HashMap<String, Value>,
    pub metrics: HashMap<String, Vec<MetricValue>>,
    pub artifacts: Vec<ArtifactReference>,
    pub metadata: HashMap<String, Value>,
    pub model_versions: Vec<ModelArtifact>,
}

/// Experiment status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Cancelled,
}

/// Metric value with timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub value: f64,
    pub timestamp: SystemTime,
    pub step: usize,
}

/// Artifact reference in experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactReference {
    pub artifact_type: ArtifactType,
    pub artifact_id: String,
    pub file_path: String,
    pub size_bytes: usize,
    pub created_at: SystemTime,
}

/// Types of artifacts that can be tracked
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ArtifactType {
    Model,
    Dataset,
    Preprocessor,
    Report,
    Plot,
    Config,
}

/// Experiment query for searching
#[derive(Debug, Clone)]
pub struct ExperimentQuery {
    pub name_contains: Option<String>,
    pub has_tag: Option<String>,
    pub status: Option<ExperimentStatus>,
    pub metric_threshold: Option<(String, f64)>, // (metric_name, min_value)
}

/// Experiment summary for reporting
#[derive(Debug, Clone)]
pub struct ExperimentSummary {
    pub experiment_id: String,
    pub name: String,
    pub status: ExperimentStatus,
    pub duration_seconds: u64,
    pub final_metrics: HashMap<String, f64>,
    pub model_count: usize,
    pub artifact_count: usize,
}

impl ExperimentSummary {
    /// Generate experiment summary report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Experiment Summary ===\n\n");
        report.push_str(&format!("Experiment: {} ({})\n", self.name, self.experiment_id));
        report.push_str(&format!("Status: {:?}\n", self.status));
        report.push_str(&format!("Duration: {}s\n", self.duration_seconds));
        report.push_str(&format!("Models: {}\n", self.model_count));
        report.push_str(&format!("Artifacts: {}\n\n", self.artifact_count));
        
        report.push_str("Final Metrics:\n");
        for (metric_name, value) in &self.final_metrics {
            report.push_str(&format!("  {}: {:.6}\n", metric_name, value));
        }
        
        report
    }
}

/// Experiment comparison results
#[derive(Debug, Clone)]
pub struct ExperimentComparison {
    pub experiments: Vec<Experiment>,
    pub comparison_metrics: HashMap<String, Vec<f64>>,
    pub best_experiment_id: String,
}

impl ExperimentComparison {
    /// Generate experiment comparison report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Experiment Comparison ===\n\n");
        report.push_str(&format!("Comparing {} experiments\n", self.experiments.len()));
        report.push_str(&format!("Best Experiment: {}\n\n", self.best_experiment_id));
        
        // Show metric comparison table
        for (metric_name, values) in &self.comparison_metrics {
            report.push_str(&format!("{}:\n", metric_name));
            for (i, experiment) in self.experiments.iter().enumerate() {
                let value = values.get(i).unwrap_or(&0.0);
                report.push_str(&format!("  {}: {:.4}\n", experiment.name, value));
            }
            report.push_str("\n");
        }
        
        report
    }
}

/// Model registry for versioning and deployment
pub struct ModelRegistry {
    models: HashMap<String, RegisteredModel>,
    storage_backend: StorageBackend,
}

impl ModelRegistry {
    /// Create new model registry
    pub fn new(storage_backend: StorageBackend) -> Self {
        Self {
            models: HashMap::new(),
            storage_backend,
        }
    }
    
    /// Register a new model or new version of existing model
    pub fn register_model(
        &mut self,
        model: &NetChain,
        model_name: String,
        version_metadata: ModelVersionMetadata,
    ) -> MLResult<String> {
        let model_version = format!("{}_{}", model_name, version_metadata.version);
        
        let model_artifact = ModelArtifact {
            model_id: model_version.clone(),
            model_name: model_name.clone(),
            experiment_id: version_metadata.experiment_id.clone(),
            created_at: SystemTime::now(),
            model_type: ModelType::NetChain,
            metadata: version_metadata.to_value_map(),
            file_path: Some(format!("registry/models/{}.model", model_version)),
            checksum: self.calculate_checksum(model)?,
            size_bytes: self.estimate_size(model),
        };
        
        // Get or create registered model
        let registered_model = self.models.entry(model_name.clone()).or_insert_with(|| {
            RegisteredModel {
                name: model_name.clone(),
                description: version_metadata.description.clone(),
                created_at: SystemTime::now(),
                updated_at: SystemTime::now(),
                versions: Vec::new(),
                tags: version_metadata.tags.clone(),
                latest_version: None,
            }
        });
        
        // Add new version
        registered_model.versions.push(model_artifact);
        registered_model.latest_version = Some(model_version.clone());
        registered_model.updated_at = SystemTime::now();
        
        println!("Registered model version: {}", model_version);
        Ok(model_version)
    }
    
    /// Get model by name and version
    pub fn get_model(&self, model_name: &str, version: Option<&str>) -> Option<&ModelArtifact> {
        if let Some(registered_model) = self.models.get(model_name) {
            if let Some(version_id) = version {
                registered_model.versions.iter()
                    .find(|v| v.model_id.contains(version_id))
            } else {
                // Get latest version
                registered_model.versions.last()
            }
        } else {
            None
        }
    }
    
    /// List all registered models
    pub fn list_models(&self) -> Vec<&RegisteredModel> {
        self.models.values().collect()
    }
    
    /// Promote model to production
    pub fn promote_to_production(
        &mut self,
        model_name: &str,
        version: Option<&str>,
        deployment_metadata: DeploymentMetadata,
    ) -> MLResult<String> {
        let model_artifact = self.get_model(model_name, version)
            .ok_or_else(|| MLError::DataError {
                reason: format!("Model {} (version {:?}) not found", model_name, version),
            })?;
        
        let deployment_id = format!("prod_{}_{}", 
            model_artifact.model_id, 
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        );
        
        // In a real implementation, this would trigger deployment processes
        println!("Promoted model {} to production: {}", model_artifact.model_id, deployment_id);
        
        Ok(deployment_id)
    }
    
    // Helper methods
    
    fn calculate_checksum(&self, _model: &NetChain) -> MLResult<String> {
        Ok(format!("checksum_{}", SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()))
    }
    
    fn estimate_size(&self, _model: &NetChain) -> usize {
        1024 * 1024 // 1MB default
    }
}

/// Registered model with version history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredModel {
    pub name: String,
    pub description: String,
    pub created_at: SystemTime,
    pub updated_at: SystemTime,
    pub versions: Vec<ModelArtifact>,
    pub tags: Vec<String>,
    pub latest_version: Option<String>,
}

/// Model artifact with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArtifact {
    pub model_id: String,
    pub model_name: String,
    pub experiment_id: String,
    pub created_at: SystemTime,
    pub model_type: ModelType,
    pub metadata: HashMap<String, Value>,
    pub file_path: Option<String>,
    pub checksum: String,
    pub size_bytes: usize,
}

/// Model type classification
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelType {
    NetChain,
    NetGraph,
    Custom,
}

/// Metadata for model versions
#[derive(Debug, Clone)]
pub struct ModelVersionMetadata {
    pub version: String,
    pub description: String,
    pub experiment_id: String,
    pub tags: Vec<String>,
    pub performance_metrics: HashMap<String, f64>,
    pub training_config: TrainingConfig,
    pub problem_type: ProblemType,
}

impl ModelVersionMetadata {
    /// Convert to Value map for storage
    fn to_value_map(&self) -> HashMap<String, Value> {
        let mut map = HashMap::new();
        map.insert("version".to_string(), Value::String(self.version.clone()));
        map.insert("description".to_string(), Value::String(self.description.clone()));
        map.insert("experiment_id".to_string(), Value::String(self.experiment_id.clone()));
        map.insert("problem_type".to_string(), Value::String(format!("{:?}", self.problem_type)));
        map.insert("learning_rate".to_string(), Value::Real(self.training_config.learning_rate));
        map.insert("batch_size".to_string(), Value::Integer(self.training_config.batch_size as i64));
        map.insert("epochs".to_string(), Value::Integer(self.training_config.epochs as i64));
        
        for (metric_name, value) in &self.performance_metrics {
            map.insert(format!("metric_{}", metric_name), Value::Real(*value));
        }
        
        map
    }
}

/// Deployment metadata for production models
#[derive(Debug, Clone)]
pub struct DeploymentMetadata {
    pub deployment_environment: String,
    pub deployment_timestamp: SystemTime,
    pub deployment_config: HashMap<String, Value>,
    pub health_check_endpoint: Option<String>,
    pub rollback_version: Option<String>,
}

/// Experiment and model lifecycle management
pub struct MLOpsManager {
    experiment_tracker: ExperimentTracker,
    model_registry: ModelRegistry,
    active_deployments: HashMap<String, DeploymentInfo>,
}

impl MLOpsManager {
    /// Create new MLOps manager
    pub fn new(storage_backend: StorageBackend) -> Self {
        Self {
            experiment_tracker: ExperimentTracker::new(storage_backend.clone()),
            model_registry: ModelRegistry::new(storage_backend),
            active_deployments: HashMap::new(),
        }
    }
    
    /// Complete ML experiment workflow with tracking
    pub fn run_tracked_experiment(
        &mut self,
        experiment_name: String,
        experiment_description: String,
        automl_workflow: impl FnOnce() -> MLResult<AutoMLResult>,
    ) -> MLResult<TrackedExperimentResult> {
        // Start experiment tracking
        let experiment_id = self.experiment_tracker.start_experiment(
            experiment_name.clone(),
            experiment_description,
            vec!["automl".to_string()],
        )?;
        
        // Run ML workflow
        let automl_result = match automl_workflow() {
            Ok(result) => {
                // Log success
                self.experiment_tracker.log_training_result(&result.training_result)?;
                
                // Extract and log hyperparameters
                let hyperparameters = self.extract_hyperparameters_from_config(&result.training_config);
                self.experiment_tracker.log_hyperparameters(hyperparameters)?;
                
                // Register model
                let model_metadata = self.create_model_metadata(&result);
                let model_version = self.experiment_tracker.register_model(
                    &result.network,
                    format!("{}_model", experiment_name),
                    model_metadata,
                )?;
                
                self.experiment_tracker.end_experiment(ExperimentStatus::Completed)?;
                result
            },
            Err(e) => {
                // Log failure
                self.experiment_tracker.end_experiment(ExperimentStatus::Failed)?;
                return Err(e);
            }
        };
        
        // Get experiment summary
        let experiment_summary = self.experiment_tracker.get_experiment(&experiment_id)
            .map(|exp| ExperimentSummary {
                experiment_id: exp.id.clone(),
                name: exp.name.clone(),
                status: exp.status,
                duration_seconds: exp.end_time.unwrap_or(SystemTime::now())
                    .duration_since(exp.start_time).unwrap().as_secs(),
                final_metrics: exp.metrics.iter()
                    .map(|(name, values)| (name.clone(), values.last().map(|v| v.value).unwrap_or(0.0)))
                    .collect(),
                model_count: exp.model_versions.len(),
                artifact_count: exp.artifacts.len(),
            })
            .unwrap();
        
        Ok(TrackedExperimentResult {
            experiment_id,
            automl_result,
            experiment_summary,
        })
    }
    
    /// Deploy model to production environment
    pub fn deploy_model(
        &mut self,
        model_name: &str,
        version: Option<&str>,
        deployment_environment: String,
    ) -> MLResult<String> {
        // Get model from registry
        let model_artifact = self.model_registry.get_model(model_name, version)
            .ok_or_else(|| MLError::DataError {
                reason: format!("Model {} (version {:?}) not found in registry", model_name, version),
            })?.clone();
        
        // Create deployment metadata
        let deployment_metadata = DeploymentMetadata {
            deployment_environment: deployment_environment.clone(),
            deployment_timestamp: SystemTime::now(),
            deployment_config: HashMap::new(),
            health_check_endpoint: Some(format!("/health/{}", model_artifact.model_id)),
            rollback_version: None,
        };
        
        // Register deployment
        let deployment_id = self.model_registry.promote_to_production(
            model_name,
            version,
            deployment_metadata.clone(),
        )?;
        
        // Track active deployment
        self.active_deployments.insert(deployment_id.clone(), DeploymentInfo {
            deployment_id: deployment_id.clone(),
            model_id: model_artifact.model_id.clone(),
            environment: deployment_environment,
            deployed_at: SystemTime::now(),
            status: DeploymentStatus::Active,
            metadata: deployment_metadata,
        });
        
        Ok(deployment_id)
    }
    
    /// Get experiment tracker for direct access
    pub fn experiment_tracker(&mut self) -> &mut ExperimentTracker {
        &mut self.experiment_tracker
    }
    
    /// Get model registry for direct access
    pub fn model_registry(&mut self) -> &mut ModelRegistry {
        &mut self.model_registry
    }
    
    /// List active deployments
    pub fn list_deployments(&self) -> Vec<&DeploymentInfo> {
        self.active_deployments.values().collect()
    }
    
    // Helper methods
    
    fn extract_hyperparameters_from_config(&self, config: &TrainingConfig) -> HashMap<String, Value> {
        let mut hyperparameters = HashMap::new();
        hyperparameters.insert("learning_rate".to_string(), Value::Real(config.learning_rate));
        hyperparameters.insert("batch_size".to_string(), Value::Integer(config.batch_size as i64));
        hyperparameters.insert("epochs".to_string(), Value::Integer(config.epochs as i64));
        hyperparameters.insert("print_progress".to_string(), Value::Boolean(config.print_progress));
        hyperparameters
    }
    
    fn create_model_metadata(&self, automl_result: &AutoMLResult) -> HashMap<String, Value> {
        let mut metadata = HashMap::new();
        metadata.insert("problem_type".to_string(), Value::String(format!("{:?}", automl_result.data_analysis.problem_type)));
        metadata.insert("feature_count".to_string(), Value::Integer(automl_result.data_analysis.feature_count as i64));
        metadata.insert("sample_count".to_string(), Value::Integer(automl_result.data_analysis.sample_count as i64));
        metadata.insert("architecture_type".to_string(), Value::String(format!("{:?}", automl_result.model_recommendation.architecture_type)));
        metadata.insert("parameter_count".to_string(), Value::Integer(automl_result.model_recommendation.parameter_count as i64));
        metadata.insert("final_loss".to_string(), Value::Real(automl_result.training_result.final_loss));
        metadata
    }
}

/// Deployment information
#[derive(Debug, Clone)]
pub struct DeploymentInfo {
    pub deployment_id: String,
    pub model_id: String,
    pub environment: String,
    pub deployed_at: SystemTime,
    pub status: DeploymentStatus,
    pub metadata: DeploymentMetadata,
}

/// Deployment status
#[derive(Debug, Clone, Copy)]
pub enum DeploymentStatus {
    Active,
    Inactive,
    Failed,
    RolledBack,
}

/// Result of tracked experiment execution
#[derive(Debug)]
pub struct TrackedExperimentResult {
    pub experiment_id: String,
    pub automl_result: AutoMLResult,
    pub experiment_summary: ExperimentSummary,
}

impl TrackedExperimentResult {
    /// Generate comprehensive tracked experiment report
    pub fn to_string(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Tracked Experiment Results ===\n\n");
        report.push_str(&self.experiment_summary.to_string());
        report.push_str("\n\n");
        report.push_str(&self.automl_result.generate_report());
        
        report
    }
}

/// MLOps utilities for common operations
pub struct MLOpsUtils;

impl MLOpsUtils {
    /// Quick experiment tracking for simple workflows
    pub fn quick_track_automl(
        experiment_name: String,
        automl_result: AutoMLResult,
    ) -> MLResult<String> {
        let mut tracker = ExperimentTracker::new(StorageBackend::Memory);
        
        let experiment_id = tracker.start_experiment(
            experiment_name,
            "Quick AutoML experiment".to_string(),
            vec!["automl".to_string(), "quick".to_string()],
        )?;
        
        // Log results
        tracker.log_training_result(&automl_result.training_result)?;
        
        let hyperparameters = HashMap::from([
            ("learning_rate".to_string(), Value::Real(automl_result.training_config.learning_rate)),
            ("batch_size".to_string(), Value::Integer(automl_result.training_config.batch_size as i64)),
            ("epochs".to_string(), Value::Integer(automl_result.training_config.epochs as i64)),
        ]);
        tracker.log_hyperparameters(hyperparameters)?;
        
        tracker.register_model(&automl_result.network, "automl_model".to_string(), HashMap::new())?;
        tracker.end_experiment(ExperimentStatus::Completed)?;
        
        Ok(experiment_id)
    }
    
    /// Export experiment data for external analysis
    pub fn export_experiment_data(
        experiment: &Experiment,
        format: ExportFormat,
    ) -> MLResult<String> {
        match format {
            ExportFormat::JSON => {
                // Simplified JSON export
                Ok(format!("{{\"id\": \"{}\", \"name\": \"{}\", \"status\": \"{:?}\"}}", 
                          experiment.id, experiment.name, experiment.status))
            },
            ExportFormat::CSV => {
                // Simplified CSV export of metrics
                let mut csv = String::new();
                csv.push_str("metric_name,final_value\n");
                for (metric_name, values) in &experiment.metrics {
                    if let Some(final_value) = values.last() {
                        csv.push_str(&format!("{},{}\n", metric_name, final_value.value));
                    }
                }
                Ok(csv)
            },
            ExportFormat::Report => {
                // Generate comprehensive report
                Ok(format!(
                    "Experiment Report\n\
                    ================\n\
                    ID: {}\n\
                    Name: {}\n\
                    Status: {:?}\n\
                    Models: {}\n\
                    Artifacts: {}",
                    experiment.id,
                    experiment.name,
                    experiment.status,
                    experiment.model_versions.len(),
                    experiment.artifacts.len()
                ))
            },
        }
    }
}

/// Export formats for experiment data
#[derive(Debug, Clone, Copy)]
pub enum ExportFormat {
    JSON,
    CSV,
    Report,
}

/// Integration with external MLOps platforms
pub struct ExternalMLOpsIntegration;

impl ExternalMLOpsIntegration {
    /// Export experiment to MLflow format
    pub fn export_to_mlflow(experiment: &Experiment) -> MLResult<String> {
        // Placeholder for MLflow integration
        Ok(format!("MLflow export for experiment {}", experiment.id))
    }
    
    /// Export experiment to Weights & Biases format
    pub fn export_to_wandb(experiment: &Experiment) -> MLResult<String> {
        // Placeholder for W&B integration
        Ok(format!("W&B export for experiment {}", experiment.id))
    }
    
    /// Export model to ONNX format
    pub fn export_model_to_onnx(model: &NetChain) -> MLResult<String> {
        // Placeholder for ONNX export
        Ok("ONNX model export (placeholder)".to_string())
    }
    
    /// Create Docker deployment configuration
    pub fn create_docker_config(
        model_artifact: &ModelArtifact,
        deployment_config: &DeploymentMetadata,
    ) -> MLResult<String> {
        // Generate Dockerfile content
        let dockerfile = format!(
            "FROM python:3.9-slim\n\
            COPY {} /app/model\n\
            WORKDIR /app\n\
            EXPOSE 8080\n\
            CMD [\"python\", \"serve_model.py\"]",
            model_artifact.file_path.as_deref().unwrap_or("model.pkl")
        );
        
        Ok(dockerfile)
    }
}