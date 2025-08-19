//! Automatic Differentiation System for Lyra
//!
//! This module provides comprehensive automatic differentiation capabilities
//! for symbolic computation, enabling gradient computation and optimization.
//!
//! Features:
//! - Forward-mode automatic differentiation via dual numbers
//! - Reverse-mode automatic differentiation via computation graphs
//! - Higher-order derivatives and Hessian computation
//! - Integration with Lyra's symbolic computation system

pub mod dual;
pub mod graph;
pub mod context;
pub mod operations;
pub mod foreign;

pub use dual::Dual;
pub use graph::{ComputationGraph, NodeId, Operation};
pub use context::{GradientContext, AutodiffMode, Variable};
pub use operations::AutodiffOps;

/// Result type for autodiff operations
pub type AutodiffResult<T> = Result<T, AutodiffError>;

/// Errors that can occur during automatic differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum AutodiffError {
    /// Variable not found in gradient context
    VariableNotFound { name: String },
    /// Incompatible dimensions for matrix operations
    DimensionMismatch { expected: Vec<usize>, actual: Vec<usize> },
    /// Operation not supported in current autodiff mode
    UnsupportedOperation { operation: String, mode: String },
    /// Gradient computation failed
    GradientComputationFailed { reason: String },
    /// Invalid dual number operation
    InvalidDualOperation { reason: String },
    /// Computation graph error
    GraphError { reason: String },
}

impl std::fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AutodiffError::VariableNotFound { name } => {
                write!(f, "Variable '{}' not found in gradient context", name)
            }
            AutodiffError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {:?}, got {:?}", expected, actual)
            }
            AutodiffError::UnsupportedOperation { operation, mode } => {
                write!(f, "Operation '{}' not supported in {} mode", operation, mode)
            }
            AutodiffError::GradientComputationFailed { reason } => {
                write!(f, "Gradient computation failed: {}", reason)
            }
            AutodiffError::InvalidDualOperation { reason } => {
                write!(f, "Invalid dual number operation: {}", reason)
            }
            AutodiffError::GraphError { reason } => {
                write!(f, "Computation graph error: {}", reason)
            }
        }
    }
}

impl std::error::Error for AutodiffError {}

/// Constants for automatic differentiation
pub mod constants {
    /// Default epsilon for numerical differentiation fallback
    pub const DEFAULT_EPSILON: f64 = 1e-8;
    
    /// Maximum computation graph depth to prevent stack overflow
    pub const MAX_GRAPH_DEPTH: usize = 10000;
    
    /// Default memory pool size for gradient computation
    pub const DEFAULT_GRADIENT_POOL_SIZE: usize = 1024;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_autodiff_error_display() {
        let error = AutodiffError::VariableNotFound {
            name: "x".to_string(),
        };
        assert_eq!(error.to_string(), "Variable 'x' not found in gradient context");
        
        let error = AutodiffError::DimensionMismatch {
            expected: vec![2, 3],
            actual: vec![3, 2],
        };
        assert_eq!(error.to_string(), "Dimension mismatch: expected [2, 3], got [3, 2]");
    }
}