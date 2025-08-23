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
}