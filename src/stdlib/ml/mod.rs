//! Machine Learning Framework for Lyra
//!
//! This module implements a complete ML framework modeled on Wolfram Language's
//! neural network APIs, including NetChain, NetGraph, and comprehensive layer types.

pub mod layers;
pub mod netchain;
pub mod netgraph;
pub mod encoders;
pub mod decoders;
pub mod training;
pub mod losses;
pub mod optimizers;
pub mod metrics;
pub mod preprocessing;
pub mod dataloader;
pub mod performance;
pub mod automl;
pub mod evaluation;
pub mod mlops;
pub mod wrapper;
pub mod bio_integration;
pub mod quantum_bridge;

// Re-export main types for convenience
pub use netchain::NetChain;
pub use netgraph::NetGraph;
pub use layers::*;  // This includes the Tensor type from layers.rs
pub use training::{NetTrain, DatasetTargetExtraction};
pub use preprocessing::{MLPreprocessor, StandardScaler, OneHotEncoder, MissingValueHandler, OutlierRemover, AutoPreprocessor, ImputationStrategy, OutlierMethod, AdvancedPreprocessingPipeline, PipelineBuilder, PipelineRegistry, PreprocessingFactory, EnhancedAutoPreprocessor};
pub use dataloader::{DataLoader, DataLoaderConfig, DataLoaderFactory, StreamingDataLoader};
pub use performance::{LazyTensor, ParallelPreprocessingPipeline, AdaptiveDataLoader, StreamingPreprocessor, MLPerformanceOptimizer, MLPerformanceMonitor};
pub use automl::{AutoMLSystem, AutoMLResult, MLPipelineBuilder, MLPipelineResult, AutoMLQuickStart, MLWorkflow, MLPatterns, ModelComplexity, PerformancePriority, ProblemType, DataType, ValidationStrategy};
pub use evaluation::{DataSplitter, EvaluationMetrics, CrossValidator, ModelSelector, HyperparameterOptimizer, ModelEvaluator, ClassificationReport, RegressionReport, CrossValidationResult, EvaluationResult, ScoringMetric};
pub use mlops::{ExperimentTracker, ModelRegistry, MLOpsManager, Experiment, ExperimentStatus, ModelArtifact, DeploymentInfo, TrackedExperimentResult, ExperimentSummary, MLOpsUtils};
pub use bio_integration::{SequenceEncoding, SequencePreprocessor, SequenceDataset, BioMLWorkflow};
pub use quantum_bridge::{QuantumFeatureMap, QuantumDataEncoder, EncodingType, NormalizationStrategy};

/// Result type for ML operations
pub type MLResult<T> = Result<T, MLError>;

/// Error types for ML operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum MLError {
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch { expected: Vec<usize>, actual: Vec<usize> },
    
    #[error("Invalid layer configuration: {reason}")]
    InvalidLayer { reason: String },
    
    #[error("Training error: {reason}")]
    TrainingError { reason: String },
    
    #[error("Network construction error: {reason}")]
    NetworkError { reason: String },
    
    #[error("Data processing error: {reason}")]
    DataError { reason: String },
    
    #[error("Autodiff error: {0}")]
    AutodiffError(#[from] crate::stdlib::autodiff::AutodiffError),
    
    #[error("Gradient computation error: {reason}")]
    GradientComputationError { reason: String },
}

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate ML stdlib functions
pub fn register_ml_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();

    // Spatial layer wrapper functions
    f.insert("FlattenLayer".to_string(), wrapper::flatten_layer as fn(&[Value]) -> VmResult<Value>);
    f.insert("ReshapeLayer".to_string(), wrapper::reshape_layer as fn(&[Value]) -> VmResult<Value>);
    f.insert("PermuteLayer".to_string(), wrapper::permute_layer as fn(&[Value]) -> VmResult<Value>);
    f.insert("TransposeLayer".to_string(), wrapper::transpose_layer as fn(&[Value]) -> VmResult<Value>);

    // Layer composition
    f.insert("Sequential".to_string(), wrapper::sequential_layer as fn(&[Value]) -> VmResult<Value>);
    f.insert("Identity".to_string(), wrapper::identity_layer as fn(&[Value]) -> VmResult<Value>);

    // Training / network constructors
    f.insert("NetTrain".to_string(), wrapper::net_train as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetChain".to_string(), wrapper::net_chain as fn(&[Value]) -> VmResult<Value>);
    f.insert("CreateTrainingConfig".to_string(), wrapper::create_training_config as fn(&[Value]) -> VmResult<Value>);

    // Tensor utilities
    f.insert("TensorShape".to_string(), wrapper::tensor_shape as fn(&[Value]) -> VmResult<Value>);
    f.insert("TensorRank".to_string(), wrapper::tensor_rank as fn(&[Value]) -> VmResult<Value>);
    f.insert("TensorSize".to_string(), wrapper::tensor_size as fn(&[Value]) -> VmResult<Value>);

    // Bio-ML integration
    f.insert("BiologicalSequenceToML".to_string(), bio_integration::biological_sequence_to_ml as fn(&[Value]) -> VmResult<Value>);
    f.insert("SequenceDataset".to_string(), bio_integration::sequence_dataset as fn(&[Value]) -> VmResult<Value>);
    f.insert("TrainSequenceClassifier".to_string(), bio_integration::train_sequence_classifier as fn(&[Value]) -> VmResult<Value>);

    // Quantum-ML bridge
    f.insert("QuantumFeatureMap".to_string(), quantum_bridge::quantum_feature_map as fn(&[Value]) -> VmResult<Value>);
    f.insert("QuantumDataEncoder".to_string(), quantum_bridge::quantum_data_encoder as fn(&[Value]) -> VmResult<Value>);
    f.insert("EncodeToQuantumState".to_string(), quantum_bridge::encode_to_quantum_state as fn(&[Value]) -> VmResult<Value>);
    f.insert("ParameterizedGate".to_string(), quantum_bridge::parameterized_gate as fn(&[Value]) -> VmResult<Value>);
    f.insert("VariationalCircuit".to_string(), quantum_bridge::variational_circuit as fn(&[Value]) -> VmResult<Value>);
    f.insert("PauliObservable".to_string(), quantum_bridge::pauli_observable as fn(&[Value]) -> VmResult<Value>);
    f.insert("QuantumGradientComputer".to_string(), quantum_bridge::quantum_gradient_computer as fn(&[Value]) -> VmResult<Value>);

    // Evaluation & preprocessing
    f.insert("TrainTestSplit".to_string(), wrapper::train_test_split as fn(&[Value]) -> VmResult<Value>);
    f.insert("ClassificationReport".to_string(), wrapper::classification_report as fn(&[Value]) -> VmResult<Value>);
    f.insert("RegressionReport".to_string(), wrapper::regression_report as fn(&[Value]) -> VmResult<Value>);
    f.insert("StandardScale".to_string(), wrapper::standard_scale as fn(&[Value]) -> VmResult<Value>);
    f.insert("OneHotEncode".to_string(), wrapper::one_hot_encode as fn(&[Value]) -> VmResult<Value>);

    // AutoML quick starts
    f.insert("AutoMLQuickStart".to_string(), wrapper::automl_quick_start_dataset as fn(&[Value]) -> VmResult<Value>);
    f.insert("AutoMLQuickStartTable".to_string(), wrapper::automl_quick_start_table as fn(&[Value]) -> VmResult<Value>);

    // MLOps basic wrappers
    f.insert("ExperimentStart".to_string(), wrapper::experiment_start as fn(&[Value]) -> VmResult<Value>);
    f.insert("ExperimentLogMetrics".to_string(), wrapper::experiment_log_metrics as fn(&[Value]) -> VmResult<Value>);
    f.insert("ExperimentEnd".to_string(), wrapper::experiment_end as fn(&[Value]) -> VmResult<Value>);
    f.insert("CrossValidate".to_string(), wrapper::cross_validate as fn(&[Value]) -> VmResult<Value>);

    // Layers
    f.insert("Linear".to_string(), wrapper::linear as fn(&[Value]) -> VmResult<Value>);
    f.insert("ReLU".to_string(), wrapper::relu as fn(&[Value]) -> VmResult<Value>);
    f.insert("Sigmoid".to_string(), wrapper::sigmoid as fn(&[Value]) -> VmResult<Value>);
    f.insert("Tanh".to_string(), wrapper::tanh as fn(&[Value]) -> VmResult<Value>);
    f.insert("Softmax".to_string(), wrapper::softmax as fn(&[Value]) -> VmResult<Value>);
    f.insert("Conv2D".to_string(), wrapper::conv2d as fn(&[Value]) -> VmResult<Value>);
    f.insert("MaxPool".to_string(), wrapper::max_pool as fn(&[Value]) -> VmResult<Value>);
    f.insert("AvgPool".to_string(), wrapper::avg_pool as fn(&[Value]) -> VmResult<Value>);
    f.insert("Dropout".to_string(), wrapper::dropout as fn(&[Value]) -> VmResult<Value>);
    f.insert("BatchNorm".to_string(), wrapper::batch_norm as fn(&[Value]) -> VmResult<Value>);

    // Table-based cross validation
    f.insert("CrossValidateTable".to_string(), wrapper::cross_validate_table as fn(&[Value]) -> VmResult<Value>);

    // NetGraph builder
    f.insert("NetGraph".to_string(), wrapper::net_graph as fn(&[Value]) -> VmResult<Value>);
    f.insert("AIForward".to_string(), wrapper::ai_forward as fn(&[Value]) -> VmResult<Value>);

    f
}
