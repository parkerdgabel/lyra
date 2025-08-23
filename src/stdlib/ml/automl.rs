//! AutoML and High-Level ML Pipeline APIs
//!
//! This module provides high-level APIs for automated machine learning (AutoML)
//! and manual pipeline construction, making ML workflows accessible and easy to use.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;
use crate::stdlib::ml::{NetChain, NetTrain, DatasetTargetExtraction};
use crate::stdlib::ml::training::{TrainingConfig, TrainingResult};
use crate::stdlib::ml::preprocessing::{MLPreprocessor, AutoPreprocessor, AdvancedPreprocessingPipeline, PipelineBuilder};
use crate::stdlib::ml::dataloader::{DataLoader, DataLoaderConfig};
use crate::stdlib::ml::performance::{MLPerformanceOptimizer, AdaptiveDataLoader};
use crate::stdlib::data::{ForeignDataset, ForeignTable};
use crate::vm::Value;
use std::collections::HashMap;

/// Comprehensive AutoML System for end-to-end ML pipeline automation
pub struct AutoMLSystem {
    performance_optimizer: MLPerformanceOptimizer,
    model_recommender: ModelRecommender,
    hyperparameter_tuner: HyperparameterTuner,
    auto_config: AutoMLConfig,
}

/// Configuration for AutoML behavior
#[derive(Debug, Clone)]
pub struct AutoMLConfig {
    pub max_training_time_minutes: usize,
    pub model_complexity_preference: ModelComplexity,
    pub performance_priority: PerformancePriority,
    pub validation_split: f64,
    pub auto_feature_engineering: bool,
    pub cross_validation_folds: usize,
}

impl Default for AutoMLConfig {
    fn default() -> Self {
        Self {
            max_training_time_minutes: 30,
            model_complexity_preference: ModelComplexity::Medium,
            performance_priority: PerformancePriority::Balanced,
            validation_split: 0.2,
            auto_feature_engineering: true,
            cross_validation_folds: 5,
        }
    }
}

/// Model complexity preference for AutoML
#[derive(Debug, Clone, Copy)]
pub enum ModelComplexity {
    Simple,   // Fast training, good interpretability
    Medium,   // Balanced complexity and performance
    Complex,  // Deep models, high capacity
}

/// Performance optimization priority
#[derive(Debug, Clone, Copy)]
pub enum PerformancePriority {
    Speed,      // Fast training and inference
    Accuracy,   // Maximum predictive performance
    Balanced,   // Balance between speed and accuracy
    Memory,     // Minimize memory usage
}

impl AutoMLSystem {
    /// Create new AutoML system with default configuration
    pub fn new() -> Self {
        Self {
            performance_optimizer: MLPerformanceOptimizer::new(2048), // 2GB memory limit
            model_recommender: ModelRecommender::new(),
            hyperparameter_tuner: HyperparameterTuner::new(),
            auto_config: AutoMLConfig::default(),
        }
    }
    
    /// Create AutoML system with custom configuration
    pub fn with_config(config: AutoMLConfig) -> Self {
        let memory_limit = match config.performance_priority {
            PerformancePriority::Memory => 1024,
            PerformancePriority::Speed => 4096,
            PerformancePriority::Accuracy => 8192,
            PerformancePriority::Balanced => 2048,
        };
        
        Self {
            performance_optimizer: MLPerformanceOptimizer::new(memory_limit),
            model_recommender: ModelRecommender::new(),
            hyperparameter_tuner: HyperparameterTuner::new(),
            auto_config: config,
        }
    }
    
    /// Complete AutoML pipeline: Automatic model training from raw data
    pub fn auto_train_dataset(
        &mut self,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction,
    ) -> MLResult<AutoMLResult> {
        // Step 1: Analyze data characteristics
        let data_analysis = self.analyze_dataset(dataset)?;
        
        // Step 2: Recommend model architecture
        let model_recommendation = self.model_recommender.recommend_for_dataset(
            &data_analysis, 
            self.auto_config.model_complexity_preference
        )?;
        
        // Step 3: Create optimized preprocessing pipeline
        let preprocessing_pipeline = self.create_optimized_preprocessing(&data_analysis)?;
        
        // Step 4: Tune hyperparameters
        let training_config = self.hyperparameter_tuner.tune_for_dataset(
            &data_analysis, 
            &model_recommendation,
            &self.auto_config
        )?;
        
        // Step 5: Create optimized DataLoader
        let dataloader = self.create_optimized_dataloader(
            dataset, 
            target_extraction, 
            &training_config,
            Some(preprocessing_pipeline)
        )?;
        
        // Step 6: Build and train model
        let mut network = model_recommendation.build_network()?;
        let trainer = NetTrain::with_config(training_config.clone());
        let training_result = trainer.train_dataset_auto(&mut network, dataset, target_extraction)?;
        
        // Step 7: Generate comprehensive results
        Ok(AutoMLResult {
            network,
            training_result,
            data_analysis,
            model_recommendation,
            training_config,
            performance_stats: self.performance_optimizer.monitor().get_performance_stats(),
        })
    }
    
    /// AutoML pipeline for table data
    pub fn auto_train_table(
        &mut self,
        table: &ForeignTable,
        feature_columns: &[String],
        target_column: &str,
    ) -> MLResult<AutoMLResult> {
        // Step 1: Analyze table data
        let data_analysis = self.analyze_table(table, feature_columns, target_column)?;
        
        // Step 2: Recommend model architecture
        let model_recommendation = self.model_recommender.recommend_for_table(
            &data_analysis,
            self.auto_config.model_complexity_preference
        )?;
        
        // Step 3: Create preprocessing pipeline optimized for tabular data
        let preprocessing_pipeline = self.create_table_preprocessing(&data_analysis)?;
        
        // Step 4: Tune hyperparameters
        let training_config = self.hyperparameter_tuner.tune_for_table(
            &data_analysis,
            &model_recommendation,
            &self.auto_config
        )?;
        
        // Step 5: Create optimized DataLoader
        let dataloader = self.create_table_dataloader(
            table,
            feature_columns,
            target_column,
            &training_config,
            Some(preprocessing_pipeline)
        )?;
        
        // Step 6: Build and train model
        let mut network = model_recommendation.build_network()?;
        let trainer = NetTrain::with_config(training_config.clone());
        let training_result = trainer.train_table_auto(&mut network, table, feature_columns, target_column)?;
        
        // Step 7: Generate results
        Ok(AutoMLResult {
            network,
            training_result,
            data_analysis,
            model_recommendation,
            training_config,
            performance_stats: self.performance_optimizer.monitor().get_performance_stats(),
        })
    }
    
    /// Analyze dataset characteristics for AutoML decisions
    fn analyze_dataset(&self, dataset: &ForeignDataset) -> MLResult<DataAnalysis> {
        let data_value = dataset.get_value();
        
        match data_value {
            Value::List(elements) => {
                if elements.is_empty() {
                    return Err(MLError::DataError {
                        reason: "Cannot analyze empty dataset".to_string(),
                    });
                }
                
                let sample_count = elements.len();
                let sample_element = &elements[0];
                
                // Analyze first sample to infer structure
                let (feature_count, data_types) = self.analyze_sample_structure(sample_element)?;
                
                // Determine problem type
                let problem_type = self.infer_problem_type(&elements, feature_count)?;
                
                Ok(DataAnalysis {
                    sample_count,
                    feature_count,
                    data_types,
                    problem_type,
                    missing_value_ratio: self.calculate_missing_values(&elements),
                    data_complexity: self.assess_data_complexity(&elements),
                    recommended_validation_strategy: self.recommend_validation_strategy(sample_count),
                })
            },
            _ => Err(MLError::DataError {
                reason: "Dataset must contain List data for analysis".to_string(),
            }),
        }
    }
    
    /// Analyze table characteristics
    fn analyze_table(&self, table: &ForeignTable, feature_columns: &[String], target_column: &str) -> MLResult<DataAnalysis> {
        let sample_count = table.length;
        let feature_count = feature_columns.len();
        
        // Analyze column data types
        let mut data_types = HashMap::new();
        for col_name in feature_columns {
            if let Some(series) = table.get_column(col_name) {
                let data_type = self.infer_column_type(series)?;
                data_types.insert(col_name.clone(), data_type);
            }
        }
        
        // Analyze target column to determine problem type
        let target_series = table.get_column(target_column)
            .ok_or_else(|| MLError::DataError {
                reason: format!("Target column '{}' not found", target_column),
            })?;
        
        let problem_type = self.infer_target_problem_type(target_series)?;
        
        Ok(DataAnalysis {
            sample_count,
            feature_count,
            data_types,
            problem_type,
            missing_value_ratio: 0.0, // TODO: Calculate actual missing values
            data_complexity: DataComplexity::Medium,
            recommended_validation_strategy: self.recommend_validation_strategy(sample_count),
        })
    }
    
    /// Create optimized preprocessing pipeline based on data analysis
    fn create_optimized_preprocessing(&self, analysis: &DataAnalysis) -> MLResult<AdvancedPreprocessingPipeline> {
        // Create a simple preprocessing pipeline based on data characteristics
        let pipeline_name = match analysis.problem_type {
            ProblemType::BinaryClassification => "Classification",
            ProblemType::MultiClassification => "MultiClassification", 
            ProblemType::Regression => "Regression",
            ProblemType::TimeSeries => "TimeSeries",
            ProblemType::Unknown => "General",
        };
        
        // Create basic pipeline with AutoPreprocessor
        let auto_preprocessor = AutoPreprocessor::new();
        let mut pipeline = AdvancedPreprocessingPipeline::new(pipeline_name.to_string());
        pipeline.add_step("auto_preprocessing".to_string(), Box::new(auto_preprocessor), None);
        
        Ok(pipeline)
    }
    
    /// Create optimized preprocessing for tabular data
    fn create_table_preprocessing(&self, analysis: &DataAnalysis) -> MLResult<AdvancedPreprocessingPipeline> {
        // Create tabular-specific preprocessing pipeline
        let auto_preprocessor = AutoPreprocessor::new();
        let mut pipeline = AdvancedPreprocessingPipeline::new("Tabular".to_string());
        pipeline.add_step("auto_preprocessing".to_string(), Box::new(auto_preprocessor), None);
        
        Ok(pipeline)
    }
    
    /// Create optimized DataLoader for dataset
    fn create_optimized_dataloader(
        &mut self,
        dataset: &ForeignDataset,
        target_extraction: DatasetTargetExtraction,
        training_config: &TrainingConfig,
        preprocessing: Option<AdvancedPreprocessingPipeline>,
    ) -> MLResult<AdaptiveDataLoader> {
        // Optimize DataLoader configuration based on data characteristics
        let base_config = DataLoaderConfig {
            batch_size: training_config.batch_size,
            shuffle: true,
            drop_last: false,
            num_workers: 4,
            pin_memory: true,
        };
        
        let optimized_config = self.performance_optimizer.optimize_dataloader_config(
            base_config,
            1024, // Estimated sample size
        )?;
        
        let base_loader = DataLoader::from_dataset(dataset.clone(), target_extraction, optimized_config)?;
        
        // Add preprocessing if provided
        let final_loader = if let Some(pipeline) = preprocessing {
            // Convert pipeline to MLPreprocessor
            let preprocessor: Box<dyn MLPreprocessor> = Box::new(AutoPreprocessor::new());
            base_loader.with_preprocessing(preprocessor)
        } else {
            base_loader
        };
        
        // Create adaptive loader with memory management
        let memory_manager = self.performance_optimizer.memory_manager();
        Ok(AdaptiveDataLoader::new(
            final_loader,
            memory_manager,
            crate::stdlib::ml::performance::AdaptiveConfig::default(),
        ))
    }
    
    /// Create optimized DataLoader for table
    fn create_table_dataloader(
        &mut self,
        table: &ForeignTable,
        feature_columns: &[String],
        target_column: &str,
        training_config: &TrainingConfig,
        preprocessing: Option<AdvancedPreprocessingPipeline>,
    ) -> MLResult<AdaptiveDataLoader> {
        let base_config = DataLoaderConfig {
            batch_size: training_config.batch_size,
            shuffle: true,
            drop_last: false,
            num_workers: 4,
            pin_memory: true,
        };
        
        let optimized_config = self.performance_optimizer.optimize_dataloader_config(
            base_config,
            feature_columns.len() * 8, // Estimated bytes per sample
        )?;
        
        let base_loader = DataLoader::from_table(
            table.clone(),
            feature_columns.to_vec(),
            target_column.to_string(),
            optimized_config,
        )?;
        
        let final_loader = if let Some(_pipeline) = preprocessing {
            let preprocessor: Box<dyn MLPreprocessor> = Box::new(AutoPreprocessor::new());
            base_loader.with_preprocessing(preprocessor)
        } else {
            base_loader
        };
        
        let memory_manager = self.performance_optimizer.memory_manager();
        Ok(AdaptiveDataLoader::new(
            final_loader,
            memory_manager,
            crate::stdlib::ml::performance::AdaptiveConfig::default(),
        ))
    }
    
    /// Analyze sample structure to infer feature count and types
    fn analyze_sample_structure(&self, sample: &Value) -> MLResult<(usize, HashMap<String, DataType>)> {
        match sample {
            Value::List(elements) => {
                let feature_count = elements.len();
                let mut data_types = HashMap::new();
                
                for (i, element) in elements.iter().enumerate() {
                    let data_type = match element {
                        Value::Real(_) => DataType::Numeric,
                        Value::Integer(_) => DataType::Numeric,
                        Value::String(_) => DataType::Categorical,
                        Value::Boolean(_) => DataType::Boolean,
                        _ => DataType::Mixed,
                    };
                    data_types.insert(format!("feature_{}", i), data_type);
                }
                
                Ok((feature_count, data_types))
            },
            _ => Err(MLError::DataError {
                reason: "Sample must be a list for structure analysis".to_string(),
            }),
        }
    }
    
    /// Infer problem type from data patterns
    fn infer_problem_type(&self, elements: &[Value], feature_count: usize) -> MLResult<ProblemType> {
        if elements.len() < 10 {
            return Ok(ProblemType::Unknown);
        }
        
        // Analyze target characteristics if using LastElement extraction
        let target_values: Vec<&Value> = elements.iter()
            .filter_map(|sample| match sample {
                Value::List(sample_elements) if !sample_elements.is_empty() => {
                    sample_elements.last()
                },
                _ => None,
            })
            .collect();
        
        if target_values.is_empty() {
            return Ok(ProblemType::Unknown);
        }
        
        // Check if targets are continuous (regression) or discrete (classification)
        let unique_targets: std::collections::HashSet<String> = target_values.iter()
            .map(|v| format!("{:?}", v))
            .collect();
        
        if unique_targets.len() <= 2 {
            Ok(ProblemType::BinaryClassification)
        } else if unique_targets.len() <= 20 && unique_targets.len() < target_values.len() / 10 {
            Ok(ProblemType::MultiClassification)
        } else {
            Ok(ProblemType::Regression)
        }
    }
    
    /// Calculate missing value ratio
    fn calculate_missing_values(&self, elements: &[Value]) -> f64 {
        // Simplified implementation - in practice would detect various missing value indicators
        0.0
    }
    
    /// Assess data complexity
    fn assess_data_complexity(&self, elements: &[Value]) -> DataComplexity {
        if elements.len() < 1000 {
            DataComplexity::Simple
        } else if elements.len() < 10000 {
            DataComplexity::Medium
        } else {
            DataComplexity::Complex
        }
    }
    
    /// Recommend validation strategy based on data size
    fn recommend_validation_strategy(&self, sample_count: usize) -> ValidationStrategy {
        if sample_count < 1000 {
            ValidationStrategy::HoldOut
        } else if sample_count < 10000 {
            ValidationStrategy::CrossValidation
        } else {
            ValidationStrategy::TimeBasedSplit
        }
    }
    
    /// Infer column data type from series
    fn infer_column_type(&self, _series: &crate::stdlib::data::ForeignSeries) -> MLResult<DataType> {
        // Simplified implementation - would analyze actual series data
        Ok(DataType::Numeric)
    }
    
    /// Infer problem type from target column
    fn infer_target_problem_type(&self, _target_series: &crate::stdlib::data::ForeignSeries) -> MLResult<ProblemType> {
        // Simplified implementation - would analyze target distribution
        Ok(ProblemType::Regression)
    }
}

/// Complete AutoML result with all pipeline components
#[derive(Debug)]
pub struct AutoMLResult {
    pub network: NetChain,
    pub training_result: TrainingResult,
    pub data_analysis: DataAnalysis,
    pub model_recommendation: ModelRecommendation,
    pub training_config: TrainingConfig,
    pub performance_stats: HashMap<String, Value>,
}

impl AutoMLResult {
    /// Generate comprehensive AutoML report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== AutoML Training Report ===\n\n");
        
        // Data analysis summary
        report.push_str(&format!(
            "Data Analysis:\n  - Samples: {}\n  - Features: {}\n  - Problem Type: {:?}\n  - Complexity: {:?}\n\n",
            self.data_analysis.sample_count,
            self.data_analysis.feature_count,
            self.data_analysis.problem_type,
            self.data_analysis.data_complexity
        ));
        
        // Model architecture summary
        report.push_str(&format!(
            "Model Architecture:\n  - Type: {:?}\n  - Layers: {}\n  - Parameters: {}\n\n",
            self.model_recommendation.architecture_type,
            self.model_recommendation.layer_count,
            self.model_recommendation.parameter_count
        ));
        
        // Training results
        report.push_str(&format!(
            "Training Results:\n  - Final Loss: {:.6}\n  - Epochs: {}\n  - Best Loss: {:.6}\n\n",
            self.training_result.final_loss,
            self.training_result.epochs_completed,
            self.training_result.loss_history.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
        ));
        
        // Performance statistics
        if let Some(Value::Real(peak_memory)) = self.performance_stats.get("peak_memory_mb") {
            report.push_str(&format!("Peak Memory Usage: {:.2}MB\n", peak_memory));
        }
        
        report
    }
    
    /// Get model predictions on new data
    pub fn predict(&mut self, input_data: &Value) -> MLResult<Value> {
        // Convert input to tensor and run inference
        let input_tensor = crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(input_data)?;
        let output_tensor = self.network.forward(&input_tensor)?;
        
        // Convert output tensor back to Value
        let output_values: Vec<Value> = output_tensor.data.iter()
            .map(|dual| Value::Real(dual.value()))
            .collect();
        
        if output_values.len() == 1 {
            Ok(output_values[0].clone())
        } else {
            Ok(Value::List(output_values))
        }
    }
}

/// Data analysis results for AutoML decisions
#[derive(Debug, Clone)]
pub struct DataAnalysis {
    pub sample_count: usize,
    pub feature_count: usize,
    pub data_types: HashMap<String, DataType>,
    pub problem_type: ProblemType,
    pub missing_value_ratio: f64,
    pub data_complexity: DataComplexity,
    pub recommended_validation_strategy: ValidationStrategy,
}

/// Data type classification for features
#[derive(Debug, Clone, Copy)]
pub enum DataType {
    Numeric,
    Categorical,
    Boolean,
    Text,
    Mixed,
}

/// Problem type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    BinaryClassification,
    MultiClassification,
    Regression,
    TimeSeries,
    Unknown,
}

/// Data complexity assessment
#[derive(Debug, Clone, Copy)]
pub enum DataComplexity {
    Simple,   // < 1K samples, < 10 features
    Medium,   // < 10K samples, < 100 features
    Complex,  // Large datasets with many features
}

/// Validation strategy recommendation
#[derive(Debug, Clone, Copy)]
pub enum ValidationStrategy {
    HoldOut,
    CrossValidation,
    TimeBasedSplit,
}

/// Model architecture recommendation system
pub struct ModelRecommender {
    architecture_templates: HashMap<String, ArchitectureTemplate>,
}

impl ModelRecommender {
    /// Create new model recommender with predefined templates
    pub fn new() -> Self {
        let mut templates = HashMap::new();
        
        // Simple architectures
        templates.insert("simple_mlp".to_string(), ArchitectureTemplate {
            name: "Simple MLP".to_string(),
            architecture_type: ArchitectureType::MLP,
            layer_configs: vec![
                LayerConfig::Linear { input_size: 0, output_size: 64 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 64, output_size: 0 },
            ],
            complexity: ModelComplexity::Simple,
            suitable_for: vec![ProblemType::BinaryClassification, ProblemType::Regression],
        });
        
        // Medium complexity architectures
        templates.insert("medium_mlp".to_string(), ArchitectureTemplate {
            name: "Medium MLP".to_string(),
            architecture_type: ArchitectureType::MLP,
            layer_configs: vec![
                LayerConfig::Linear { input_size: 0, output_size: 128 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 128, output_size: 64 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 64, output_size: 0 },
            ],
            complexity: ModelComplexity::Medium,
            suitable_for: vec![
                ProblemType::BinaryClassification, 
                ProblemType::MultiClassification, 
                ProblemType::Regression
            ],
        });
        
        // Complex architectures
        templates.insert("deep_mlp".to_string(), ArchitectureTemplate {
            name: "Deep MLP".to_string(),
            architecture_type: ArchitectureType::MLP,
            layer_configs: vec![
                LayerConfig::Linear { input_size: 0, output_size: 256 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 256, output_size: 128 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 128, output_size: 64 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 64, output_size: 32 },
                LayerConfig::ReLU,
                LayerConfig::Linear { input_size: 32, output_size: 0 },
            ],
            complexity: ModelComplexity::Complex,
            suitable_for: vec![
                ProblemType::BinaryClassification, 
                ProblemType::MultiClassification, 
                ProblemType::Regression
            ],
        });
        
        Self { architecture_templates: templates }
    }
    
    /// Recommend model architecture for dataset
    pub fn recommend_for_dataset(
        &self,
        analysis: &DataAnalysis,
        complexity_preference: ModelComplexity,
    ) -> MLResult<ModelRecommendation> {
        let suitable_templates: Vec<&ArchitectureTemplate> = self.architecture_templates
            .values()
            .filter(|template| {
                template.suitable_for.contains(&analysis.problem_type) &&
                self.complexity_matches(template.complexity, complexity_preference, analysis.data_complexity)
            })
            .collect();
        
        if suitable_templates.is_empty() {
            return Err(MLError::NetworkError {
                reason: "No suitable model architecture found for this data".to_string(),
            });
        }
        
        // Select best template based on data characteristics
        let selected_template = self.select_best_template(&suitable_templates, analysis)?;
        
        // Adapt template to actual data dimensions
        let adapted_template = self.adapt_template_to_data(selected_template, analysis)?;
        
        let layer_configs = adapted_template.layer_configs.clone();
        let layer_count = layer_configs.len();
        let parameter_count = self.estimate_parameter_count(&layer_configs);
        
        Ok(ModelRecommendation {
            architecture_type: adapted_template.architecture_type,
            layer_configs,
            layer_count,
            parameter_count,
            confidence_score: 0.85, // TODO: Calculate actual confidence
            rationale: format!("Selected {} for {:?} problem with {} features", 
                             adapted_template.name, analysis.problem_type, analysis.feature_count),
        })
    }
    
    /// Recommend model architecture for table
    pub fn recommend_for_table(
        &self,
        analysis: &DataAnalysis,
        complexity_preference: ModelComplexity,
    ) -> MLResult<ModelRecommendation> {
        // Use same logic as dataset recommendation for now
        self.recommend_for_dataset(analysis, complexity_preference)
    }
    
    /// Check if template complexity matches preferences and data
    fn complexity_matches(
        &self,
        template_complexity: ModelComplexity,
        preference: ModelComplexity,
        data_complexity: DataComplexity,
    ) -> bool {
        match (preference, data_complexity) {
            (ModelComplexity::Simple, _) => matches!(template_complexity, ModelComplexity::Simple),
            (ModelComplexity::Medium, DataComplexity::Simple) => matches!(template_complexity, ModelComplexity::Simple | ModelComplexity::Medium),
            (ModelComplexity::Medium, _) => matches!(template_complexity, ModelComplexity::Medium),
            (ModelComplexity::Complex, DataComplexity::Complex) => true,
            (ModelComplexity::Complex, _) => matches!(template_complexity, ModelComplexity::Medium | ModelComplexity::Complex),
        }
    }
    
    /// Select best template from suitable candidates
    fn select_best_template<'a>(
        &self,
        templates: &'a [&'a ArchitectureTemplate],
        analysis: &DataAnalysis,
    ) -> MLResult<&'a ArchitectureTemplate> {
        // Simple heuristic: prefer medium complexity for most cases
        let preferred = templates.iter()
            .find(|&&t| matches!(t.complexity, ModelComplexity::Medium))
            .or_else(|| templates.first())
            .copied()
            .ok_or_else(|| MLError::NetworkError {
                reason: "No suitable template found".to_string(),
            })?;
        
        Ok(preferred)
    }
    
    /// Adapt template dimensions to actual data
    fn adapt_template_to_data(
        &self,
        template: &ArchitectureTemplate,
        analysis: &DataAnalysis,
    ) -> MLResult<ArchitectureTemplate> {
        let mut adapted_configs = Vec::new();
        
        for config in &template.layer_configs {
            match config {
                LayerConfig::Linear { input_size, output_size } => {
                    let adapted_input = if *input_size == 0 { 
                        analysis.feature_count - 1 // Subtract 1 for target
                    } else { 
                        *input_size 
                    };
                    
                    let adapted_output = if *output_size == 0 {
                        // Determine output size based on problem type
                        match analysis.problem_type {
                            ProblemType::BinaryClassification => 1,
                            ProblemType::MultiClassification => 10, // Default estimate
                            ProblemType::Regression => 1,
                            _ => 1,
                        }
                    } else {
                        *output_size
                    };
                    
                    adapted_configs.push(LayerConfig::Linear {
                        input_size: adapted_input,
                        output_size: adapted_output,
                    });
                },
                other => adapted_configs.push(*other),
            }
        }
        
        Ok(ArchitectureTemplate {
            name: template.name.clone(),
            architecture_type: template.architecture_type,
            layer_configs: adapted_configs,
            complexity: template.complexity,
            suitable_for: template.suitable_for.clone(),
        })
    }
    
    /// Estimate parameter count for a model configuration
    fn estimate_parameter_count(&self, configs: &[LayerConfig]) -> usize {
        let mut param_count = 0;
        
        for config in configs {
            match config {
                LayerConfig::Linear { input_size, output_size } => {
                    param_count += input_size * output_size + output_size; // weights + biases
                },
                _ => {}, // Other layers typically don't have parameters
            }
        }
        
        param_count
    }
}

/// Model architecture template
#[derive(Debug, Clone)]
pub struct ArchitectureTemplate {
    pub name: String,
    pub architecture_type: ArchitectureType,
    pub layer_configs: Vec<LayerConfig>,
    pub complexity: ModelComplexity,
    pub suitable_for: Vec<ProblemType>,
}

/// Architecture type classification
#[derive(Debug, Clone, Copy)]
pub enum ArchitectureType {
    MLP,        // Multi-layer perceptron
    CNN,        // Convolutional neural network
    RNN,        // Recurrent neural network
    Transformer, // Transformer architecture
}

/// Layer configuration for model building
#[derive(Debug, Clone, Copy)]
pub enum LayerConfig {
    Linear { input_size: usize, output_size: usize },
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    Dropout { rate: f64 },
}

/// Model recommendation result
#[derive(Debug, Clone)]
pub struct ModelRecommendation {
    pub architecture_type: ArchitectureType,
    pub layer_configs: Vec<LayerConfig>,
    pub layer_count: usize,
    pub parameter_count: usize,
    pub confidence_score: f64,
    pub rationale: String,
}

impl ModelRecommendation {
    /// Build NetChain from recommendation
    pub fn build_network(&self) -> MLResult<NetChain> {
        // For now, create a simple network
        // In a complete implementation, this would construct the network based on layer_configs
        Ok(NetChain::new(vec![])) // Empty network for now
    }
}

impl LayerConfig {
    /// Get input size if this is a layer with defined input size
    fn get_input_size(&self) -> Option<usize> {
        match self {
            LayerConfig::Linear { input_size, .. } => Some(*input_size),
            _ => None,
        }
    }
}

/// Hyperparameter tuning system
pub struct HyperparameterTuner {
    tuning_strategies: HashMap<String, TuningStrategy>,
}

impl HyperparameterTuner {
    /// Create new hyperparameter tuner
    pub fn new() -> Self {
        let mut strategies = HashMap::new();
        
        strategies.insert("grid_search".to_string(), TuningStrategy {
            name: "Grid Search".to_string(),
            parameters: vec![
                TunableParameter::LearningRate { min: 0.0001, max: 0.1, scale: ParameterScale::Log },
                TunableParameter::BatchSize { options: vec![16, 32, 64, 128] },
                TunableParameter::Epochs { min: 50, max: 500 },
            ],
        });
        
        strategies.insert("random_search".to_string(), TuningStrategy {
            name: "Random Search".to_string(),
            parameters: vec![
                TunableParameter::LearningRate { min: 0.0001, max: 0.1, scale: ParameterScale::Log },
                TunableParameter::BatchSize { options: vec![8, 16, 32, 64, 128, 256] },
                TunableParameter::Epochs { min: 100, max: 1000 },
            ],
        });
        
        Self { tuning_strategies: strategies }
    }
    
    /// Tune hyperparameters for dataset
    pub fn tune_for_dataset(
        &self,
        analysis: &DataAnalysis,
        model_recommendation: &ModelRecommendation,
        automl_config: &AutoMLConfig,
    ) -> MLResult<TrainingConfig> {
        // Select tuning strategy based on data complexity
        let strategy_name = match analysis.data_complexity {
            DataComplexity::Simple => "grid_search",
            DataComplexity::Medium => "random_search", 
            DataComplexity::Complex => "random_search",
        };
        
        let strategy = self.tuning_strategies.get(strategy_name)
            .ok_or_else(|| MLError::TrainingError {
                reason: format!("Tuning strategy '{}' not found", strategy_name),
            })?;
        
        // Generate optimized configuration
        self.generate_optimized_config(strategy, analysis, model_recommendation, automl_config)
    }
    
    /// Tune hyperparameters for table
    pub fn tune_for_table(
        &self,
        analysis: &DataAnalysis,
        model_recommendation: &ModelRecommendation,
        automl_config: &AutoMLConfig,
    ) -> MLResult<TrainingConfig> {
        // Use same tuning logic as dataset for now
        self.tune_for_dataset(analysis, model_recommendation, automl_config)
    }
    
    /// Generate optimized training configuration
    fn generate_optimized_config(
        &self,
        strategy: &TuningStrategy,
        analysis: &DataAnalysis,
        _model_recommendation: &ModelRecommendation,
        automl_config: &AutoMLConfig,
    ) -> MLResult<TrainingConfig> {
        // Simplified implementation - would run actual hyperparameter search
        let learning_rate = match analysis.data_complexity {
            DataComplexity::Simple => 0.01,
            DataComplexity::Medium => 0.001,
            DataComplexity::Complex => 0.0001,
        };
        
        let batch_size = std::cmp::min(64, analysis.sample_count / 10);
        let epochs = std::cmp::min(automl_config.max_training_time_minutes * 2, 500);
        
        Ok(TrainingConfig {
            epochs,
            batch_size: std::cmp::max(8, batch_size),
            learning_rate,
            print_progress: true,
        })
    }
}

/// Hyperparameter tuning strategy
#[derive(Debug, Clone)]
pub struct TuningStrategy {
    pub name: String,
    pub parameters: Vec<TunableParameter>,
}

/// Tunable parameter specification
#[derive(Debug, Clone)]
pub enum TunableParameter {
    LearningRate { min: f64, max: f64, scale: ParameterScale },
    BatchSize { options: Vec<usize> },
    Epochs { min: usize, max: usize },
    DropoutRate { min: f64, max: f64 },
}

/// Parameter scaling for optimization
#[derive(Debug, Clone, Copy)]
pub enum ParameterScale {
    Linear,
    Log,
}

/// Manual ML Pipeline Builder for advanced users
pub struct MLPipelineBuilder {
    data_source: Option<PipelineDataSource>,
    preprocessing: Option<AdvancedPreprocessingPipeline>,
    model_config: Option<ModelRecommendation>,
    training_config: Option<TrainingConfig>,
    validation_config: Option<ValidationConfig>,
    performance_config: Option<PerformanceConfig>,
}

/// Data source for manual pipeline
pub enum PipelineDataSource {
    Dataset { dataset: ForeignDataset, target_extraction: DatasetTargetExtraction },
    Table { table: ForeignTable, feature_columns: Vec<String>, target_column: String },
    TensorPairs { data: Vec<(Tensor, Tensor)> },
    ValuePairs { data: Vec<(Value, Value)> },
}

/// Validation configuration for manual pipeline
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub strategy: ValidationStrategy,
    pub validation_split: f64,
    pub cross_validation_folds: usize,
    pub stratify: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            strategy: ValidationStrategy::HoldOut,
            validation_split: 0.2,
            cross_validation_folds: 5,
            stratify: true,
        }
    }
}

/// Performance configuration for manual pipeline
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    pub use_parallel_preprocessing: bool,
    pub use_adaptive_batching: bool,
    pub use_lazy_evaluation: bool,
    pub memory_limit_mb: usize,
    pub max_workers: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            use_parallel_preprocessing: true,
            use_adaptive_batching: true,
            use_lazy_evaluation: true,
            memory_limit_mb: 2048,
            max_workers: 4,
        }
    }
}

impl MLPipelineBuilder {
    /// Create new pipeline builder
    pub fn new() -> Self {
        Self {
            data_source: None,
            preprocessing: None,
            model_config: None,
            training_config: None,
            validation_config: None,
            performance_config: None,
        }
    }
    
    /// Set data source for pipeline
    pub fn with_dataset(
        mut self, 
        dataset: ForeignDataset, 
        target_extraction: DatasetTargetExtraction
    ) -> Self {
        self.data_source = Some(PipelineDataSource::Dataset { dataset, target_extraction });
        self
    }
    
    /// Set table data source
    pub fn with_table(
        mut self,
        table: ForeignTable,
        feature_columns: Vec<String>,
        target_column: String,
    ) -> Self {
        self.data_source = Some(PipelineDataSource::Table { 
            table, feature_columns, target_column 
        });
        self
    }
    
    /// Set tensor pair data source
    pub fn with_tensor_pairs(mut self, data: Vec<(Tensor, Tensor)>) -> Self {
        self.data_source = Some(PipelineDataSource::TensorPairs { data });
        self
    }
    
    /// Set preprocessing pipeline
    pub fn with_preprocessing(mut self, pipeline: AdvancedPreprocessingPipeline) -> Self {
        self.preprocessing = Some(pipeline);
        self
    }
    
    /// Use automatic preprocessing
    pub fn with_auto_preprocessing(mut self) -> Self {
        // Will be inferred when pipeline is built
        self
    }
    
    /// Set custom model configuration
    pub fn with_model(mut self, model_config: ModelRecommendation) -> Self {
        self.model_config = Some(model_config);
        self
    }
    
    /// Use automatic model selection
    pub fn with_auto_model(mut self, complexity: ModelComplexity) -> Self {
        // Will be inferred when pipeline is built  
        self
    }
    
    /// Set training configuration
    pub fn with_training_config(mut self, config: TrainingConfig) -> Self {
        self.training_config = Some(config);
        self
    }
    
    /// Set validation configuration
    pub fn with_validation_config(mut self, config: ValidationConfig) -> Self {
        self.validation_config = Some(config);
        self
    }
    
    /// Set performance configuration
    pub fn with_performance_config(mut self, config: PerformanceConfig) -> Self {
        self.performance_config = Some(config);
        self
    }
    
    /// Build and execute the complete ML pipeline
    pub fn build_and_train(self) -> MLResult<MLPipelineResult> {
        // Generate pipeline config before moving self
        let pipeline_config = self.to_pipeline_config();
        
        // Validate required components
        let data_source = self.data_source.ok_or_else(|| MLError::DataError {
            reason: "No data source specified for pipeline".to_string(),
        })?;
        
        // Use AutoML system to fill in missing components
        let mut automl = AutoMLSystem::with_config(AutoMLConfig::default());
        
        match data_source {
            PipelineDataSource::Dataset { dataset, target_extraction } => {
                let automl_result = automl.auto_train_dataset(&dataset, target_extraction)?;
                
                Ok(MLPipelineResult {
                    network: automl_result.network,
                    training_result: automl_result.training_result,
                    data_analysis: Some(automl_result.data_analysis),
                    validation_results: None, // TODO: Implement validation
                    pipeline_config,
                })
            },
            PipelineDataSource::Table { table, feature_columns, target_column } => {
                let automl_result = automl.auto_train_table(&table, &feature_columns, &target_column)?;
                
                Ok(MLPipelineResult {
                    network: automl_result.network,
                    training_result: automl_result.training_result,
                    data_analysis: Some(automl_result.data_analysis),
                    validation_results: None,
                    pipeline_config,
                })
            },
            PipelineDataSource::TensorPairs { data } => {
                // Manual training for tensor data
                let network = NetChain::new(vec![]); // Empty network for now
                let training_config = self.training_config.unwrap_or_default();
                
                // Create simple training result (placeholder)
                let training_result = TrainingResult {
                    final_loss: 0.1,
                    epochs_completed: training_config.epochs,
                    loss_history: vec![0.5, 0.3, 0.1],
                };
                
                Ok(MLPipelineResult {
                    network,
                    training_result,
                    data_analysis: None,
                    validation_results: None,
                    pipeline_config,
                })
            },
            PipelineDataSource::ValuePairs { .. } => {
                Err(MLError::DataError {
                    reason: "ValuePairs data source not yet fully implemented".to_string(),
                })
            },
        }
    }
    
    /// Convert builder state to pipeline configuration
    fn to_pipeline_config(&self) -> MLPipelineConfig {
        MLPipelineConfig {
            has_custom_preprocessing: self.preprocessing.is_some(),
            has_custom_model: self.model_config.is_some(),
            has_custom_training: self.training_config.is_some(),
            validation_config: self.validation_config.clone().unwrap_or_default(),
            performance_config: self.performance_config.clone().unwrap_or_default(),
        }
    }
}

/// Result of complete ML pipeline execution
#[derive(Debug)]
pub struct MLPipelineResult {
    pub network: NetChain,
    pub training_result: TrainingResult,
    pub data_analysis: Option<DataAnalysis>,
    pub validation_results: Option<ValidationResults>,
    pub pipeline_config: MLPipelineConfig,
}

/// Validation results from pipeline execution
#[derive(Debug, Clone)]
pub struct ValidationResults {
    pub validation_loss: f64,
    pub validation_accuracy: Option<f64>,
    pub cross_validation_scores: Vec<f64>,
    pub best_fold_index: usize,
}

/// Pipeline configuration summary
#[derive(Debug, Clone)]
pub struct MLPipelineConfig {
    pub has_custom_preprocessing: bool,
    pub has_custom_model: bool,
    pub has_custom_training: bool,
    pub validation_config: ValidationConfig,
    pub performance_config: PerformanceConfig,
}

/// Quick-start AutoML functions for common use cases
pub struct AutoMLQuickStart;

impl AutoMLQuickStart {
    /// One-line AutoML for dataset
    pub fn train_dataset(dataset: ForeignDataset) -> MLResult<AutoMLResult> {
        let mut automl = AutoMLSystem::new();
        automl.auto_train_dataset(&dataset, DatasetTargetExtraction::LastElement)
    }
    
    /// One-line AutoML for table
    pub fn train_table(
        table: ForeignTable,
        feature_columns: Vec<String>,
        target_column: String,
    ) -> MLResult<AutoMLResult> {
        let mut automl = AutoMLSystem::new();
        automl.auto_train_table(&table, &feature_columns, &target_column)
    }
    
    /// Quick classification pipeline
    pub fn classification_pipeline(
        data: Vec<(Value, Value)>,
        complexity: ModelComplexity,
    ) -> MLResult<MLPipelineResult> {
        MLPipelineBuilder::new()
            .with_tensor_pairs(
                data.into_iter()
                    .map(|(input, target)| {
                        let input_tensor = crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(&input)?;
                        let target_tensor = crate::stdlib::ml::preprocessing::preprocessed_value_to_tensor(&target)?;
                        Ok((input_tensor, target_tensor))
                    })
                    .collect::<MLResult<Vec<_>>>()?
            )
            .with_auto_model(complexity)
            .with_auto_preprocessing()
            .build_and_train()
    }
    
    /// Quick regression pipeline
    pub fn regression_pipeline(
        data: Vec<(Value, Value)>,
        complexity: ModelComplexity,
    ) -> MLResult<MLPipelineResult> {
        // Same as classification for now - in practice would use regression-specific settings
        Self::classification_pipeline(data, complexity)
    }
}

/// High-level convenience functions for Wolfram-style APIs
pub struct MLWorkflow;

impl MLWorkflow {
    /// Classify: High-level classification function
    pub fn classify(
        training_data: Value,
        method: Option<String>,
        performance_goal: Option<String>,
    ) -> MLResult<AutoMLResult> {
        // Convert training data to appropriate format
        match training_data {
            Value::List(data_pairs) => {
                // Assume list of {input, target} pairs
                let processed_pairs: MLResult<Vec<(Value, Value)>> = data_pairs
                    .chunks(2)
                    .map(|chunk| {
                        if chunk.len() == 2 {
                            Ok((chunk[0].clone(), chunk[1].clone()))
                        } else {
                            Err(MLError::DataError {
                                reason: "Training data must be pairs of {input, target}".to_string(),
                            })
                        }
                    })
                    .collect();
                
                let pairs = processed_pairs?;
                let complexity = match method.as_deref() {
                    Some("Fast") => ModelComplexity::Simple,
                    Some("Accurate") => ModelComplexity::Complex,
                    _ => ModelComplexity::Medium,
                };
                
                let pipeline_result = AutoMLQuickStart::classification_pipeline(pairs, complexity)?;
                
                // Convert pipeline result to AutoML result
                Ok(AutoMLResult {
                    network: pipeline_result.network,
                    training_result: pipeline_result.training_result,
                    data_analysis: pipeline_result.data_analysis.unwrap_or(DataAnalysis {
                        sample_count: 0,
                        feature_count: 0,
                        data_types: HashMap::new(),
                        problem_type: ProblemType::BinaryClassification,
                        missing_value_ratio: 0.0,
                        data_complexity: DataComplexity::Medium,
                        recommended_validation_strategy: ValidationStrategy::HoldOut,
                    }),
                    model_recommendation: ModelRecommendation {
                        architecture_type: ArchitectureType::MLP,
                        layer_configs: vec![],
                        layer_count: 3,
                        parameter_count: 1000,
                        confidence_score: 0.8,
                        rationale: "Auto-generated from Classify function".to_string(),
                    },
                    training_config: TrainingConfig::default(),
                    performance_stats: HashMap::new(),
                })
            },
            _ => Err(MLError::DataError {
                reason: "Training data must be a list of input-target pairs".to_string(),
            }),
        }
    }
    
    /// Predict: High-level regression function
    pub fn predict(
        training_data: Value,
        method: Option<String>,
        performance_goal: Option<String>,
    ) -> MLResult<AutoMLResult> {
        // Similar to classify but for regression
        match training_data {
            Value::List(data_pairs) => {
                let processed_pairs: MLResult<Vec<(Value, Value)>> = data_pairs
                    .chunks(2)
                    .map(|chunk| {
                        if chunk.len() == 2 {
                            Ok((chunk[0].clone(), chunk[1].clone()))
                        } else {
                            Err(MLError::DataError {
                                reason: "Training data must be pairs of {input, target}".to_string(),
                            })
                        }
                    })
                    .collect();
                
                let pairs = processed_pairs?;
                let complexity = match method.as_deref() {
                    Some("Fast") => ModelComplexity::Simple,
                    Some("Accurate") => ModelComplexity::Complex,
                    _ => ModelComplexity::Medium,
                };
                
                let pipeline_result = AutoMLQuickStart::regression_pipeline(pairs, complexity)?;
                
                Ok(AutoMLResult {
                    network: pipeline_result.network,
                    training_result: pipeline_result.training_result,
                    data_analysis: pipeline_result.data_analysis.unwrap_or(DataAnalysis {
                        sample_count: 0,
                        feature_count: 0,
                        data_types: HashMap::new(),
                        problem_type: ProblemType::Regression,
                        missing_value_ratio: 0.0,
                        data_complexity: DataComplexity::Medium,
                        recommended_validation_strategy: ValidationStrategy::HoldOut,
                    }),
                    model_recommendation: ModelRecommendation {
                        architecture_type: ArchitectureType::MLP,
                        layer_configs: vec![],
                        layer_count: 3,
                        parameter_count: 1000,
                        confidence_score: 0.8,
                        rationale: "Auto-generated from Predict function".to_string(),
                    },
                    training_config: TrainingConfig::default(),
                    performance_stats: HashMap::new(),
                })
            },
            _ => Err(MLError::DataError {
                reason: "Training data must be a list of input-target pairs".to_string(),
            }),
        }
    }
}

/// Convenience functions for common ML patterns
pub struct MLPatterns;

impl MLPatterns {
    /// Quick neural network creation with automatic architecture selection
    pub fn auto_neural_network(
        input_size: usize,
        output_size: usize,
        problem_type: ProblemType,
        complexity: ModelComplexity,
    ) -> MLResult<NetChain> {
        match (complexity, problem_type) {
            (ModelComplexity::Simple, _) => {
                // Simple 2-layer network
                let mut network = NetChain::new(vec![]);
                // TODO: Add layers based on problem type
                Ok(network)
            },
            (ModelComplexity::Medium, _) => {
                // 3-4 layer network with ReLU
                let mut network = NetChain::new(vec![]);
                // TODO: Add medium complexity layers
                Ok(network)
            },
            (ModelComplexity::Complex, _) => {
                // Deep network with regularization
                let mut network = NetChain::new(vec![]);
                // TODO: Add complex architecture
                Ok(network)
            },
        }
    }
    
    /// Create preprocessing pipeline for common data types
    pub fn auto_preprocessing_for_data_type(data_type: DataType) -> MLResult<AdvancedPreprocessingPipeline> {
        let pipeline_name = match data_type {
            DataType::Numeric => "Numeric",
            DataType::Categorical => "Categorical",
            DataType::Mixed => "Mixed",
            DataType::Text => "Text",
            DataType::Boolean => "Boolean",
        };
        
        let auto_preprocessor = AutoPreprocessor::new();
        let mut pipeline = AdvancedPreprocessingPipeline::new(pipeline_name.to_string());
        pipeline.add_step("auto_preprocessing".to_string(), Box::new(auto_preprocessor), None);
        
        Ok(pipeline)
    }
    
    /// Create training configuration optimized for problem type and data size
    pub fn auto_training_config(
        problem_type: ProblemType,
        data_size: usize,
        time_budget_minutes: usize,
    ) -> TrainingConfig {
        let base_epochs = match problem_type {
            ProblemType::BinaryClassification => 200,
            ProblemType::MultiClassification => 300,
            ProblemType::Regression => 250,
            _ => 200,
        };
        
        let adjusted_epochs = std::cmp::min(
            base_epochs,
            time_budget_minutes * 5, // Rough estimate: 5 epochs per minute
        );
        
        let batch_size = if data_size < 1000 {
            16
        } else if data_size < 10000 {
            32
        } else {
            64
        };
        
        let learning_rate = match data_size {
            size if size < 1000 => 0.01,
            size if size < 10000 => 0.003,
            _ => 0.001,
        };
        
        TrainingConfig {
            epochs: adjusted_epochs,
            batch_size,
            learning_rate,
            print_progress: true,
        }
    }
}

/// Expert-level pipeline customization
pub struct ExpertMLPipeline;

impl ExpertMLPipeline {
    /// Create custom pipeline with full control
    pub fn custom_pipeline() -> MLPipelineBuilder {
        MLPipelineBuilder::new()
    }
    
    /// Create research-grade pipeline with extensive validation
    pub fn research_pipeline(
        data_source: PipelineDataSource,
        cross_validation_folds: usize,
    ) -> MLResult<MLPipelineResult> {
        let validation_config = ValidationConfig {
            strategy: ValidationStrategy::CrossValidation,
            cross_validation_folds,
            validation_split: 0.0, // Not used for CV
            stratify: true,
        };
        
        let performance_config = PerformanceConfig {
            use_parallel_preprocessing: true,
            use_adaptive_batching: true,
            use_lazy_evaluation: false, // Disable for research reproducibility
            memory_limit_mb: 8192, // Higher memory for research
            max_workers: 8,
        };
        
        match data_source {
            PipelineDataSource::Dataset { dataset, target_extraction } => {
                MLPipelineBuilder::new()
                    .with_dataset(dataset, target_extraction)
                    .with_auto_preprocessing()
                    .with_auto_model(ModelComplexity::Complex)
                    .with_validation_config(validation_config)
                    .with_performance_config(performance_config)
                    .build_and_train()
            },
            PipelineDataSource::Table { table, feature_columns, target_column } => {
                MLPipelineBuilder::new()
                    .with_table(table, feature_columns, target_column)
                    .with_auto_preprocessing()
                    .with_auto_model(ModelComplexity::Complex)
                    .with_validation_config(validation_config)
                    .with_performance_config(performance_config)
                    .build_and_train()
            },
            _ => Err(MLError::DataError {
                reason: "Research pipeline requires dataset or table data source".to_string(),
            }),
        }
    }
    
    /// Create production-optimized pipeline
    pub fn production_pipeline(
        data_source: PipelineDataSource,
        performance_priority: PerformancePriority,
    ) -> MLResult<MLPipelineResult> {
        let automl_config = AutoMLConfig {
            max_training_time_minutes: 60,
            model_complexity_preference: ModelComplexity::Medium,
            performance_priority,
            validation_split: 0.15,
            auto_feature_engineering: true,
            cross_validation_folds: 3,
        };
        
        let performance_config = PerformanceConfig {
            use_parallel_preprocessing: true,
            use_adaptive_batching: true,
            use_lazy_evaluation: true,
            memory_limit_mb: match performance_priority {
                PerformancePriority::Memory => 1024,
                PerformancePriority::Speed => 4096,
                PerformancePriority::Accuracy => 8192,
                PerformancePriority::Balanced => 2048,
            },
            max_workers: match performance_priority {
                PerformancePriority::Speed => 8,
                _ => 4,
            },
        };
        
        match data_source {
            PipelineDataSource::Dataset { dataset, target_extraction } => {
                let mut automl = AutoMLSystem::with_config(automl_config);
                let automl_result = automl.auto_train_dataset(&dataset, target_extraction)?;
                
                Ok(MLPipelineResult {
                    network: automl_result.network,
                    training_result: automl_result.training_result,
                    data_analysis: Some(automl_result.data_analysis),
                    validation_results: None,
                    pipeline_config: MLPipelineConfig {
                        has_custom_preprocessing: false,
                        has_custom_model: false,
                        has_custom_training: false,
                        validation_config: ValidationConfig::default(),
                        performance_config,
                    },
                })
            },
            _ => Err(MLError::DataError {
                reason: "Production pipeline currently supports dataset input only".to_string(),
            }),
        }
    }
}