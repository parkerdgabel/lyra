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

// Re-export main types for convenience
pub use netchain::NetChain;
pub use netgraph::NetGraph;
pub use layers::*;
pub use training::NetTrain;

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