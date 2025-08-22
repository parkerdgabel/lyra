//! Unified error handling utilities for Lyra standard library
//!
//! This module provides consistent error handling across all stdlib modules,
//! including conversion utilities between different error types and
//! standardized error messages.

use crate::vm::VmError;
use crate::foreign::ForeignError;
use thiserror::Error;

/// Unified error type that can represent errors from any stdlib domain
#[derive(Error, Debug, Clone, PartialEq)]
pub enum UnifiedError {
    #[error("Argument error: {message}")]
    ArgumentError { message: String },
    
    #[error("Type error: expected {expected}, got {actual}")]
    TypeError { expected: String, actual: String },
    
    #[error("Index out of bounds: {index} not in range [0, {max}]")]
    IndexOutOfBounds { index: isize, max: usize },
    
    #[error("Invalid operation: {operation} not supported for {type_name}")]
    UnsupportedOperation { operation: String, type_name: String },
    
    #[error("Runtime error: {message}")]
    RuntimeError { message: String },
    
    #[error("Math error: {message}")]
    MathError { message: String },
    
    #[error("Algorithm error: {message}")]
    AlgorithmError { message: String },
    
    #[error("Validation error: {message}")]
    ValidationError { message: String },
}

/// Trait for converting between different error types used in stdlib
pub trait ErrorConversion {
    /// Convert ForeignError to VmError
    fn foreign_to_vm_error(err: ForeignError) -> VmError;
    
    /// Convert UnifiedError to VmError
    fn unified_to_vm_error(err: UnifiedError) -> VmError;
    
    /// Convert UnifiedError to ForeignError
    fn unified_to_foreign_error(err: UnifiedError) -> ForeignError;
}

impl ErrorConversion for () {
    fn foreign_to_vm_error(err: ForeignError) -> VmError {
        match err {
            ForeignError::UnknownMethod { method, type_name } => {
                VmError::Runtime(format!("Unknown method '{}' for type '{}'", method, type_name))
            }
            ForeignError::InvalidArity { method, expected, actual } => {
                VmError::Runtime(format!(
                    "Invalid arity for method '{}': expected {}, got {}", 
                    method, expected, actual
                ))
            }
            ForeignError::InvalidArgumentType { method, expected, actual } => {
                VmError::Runtime(format!(
                    "Invalid argument type for method '{}': expected {}, got {}", 
                    method, expected, actual
                ))
            }
            ForeignError::IndexOutOfBounds { index, bounds } => {
                VmError::Runtime(format!("Index out of bounds: {} not in range {}", index, bounds))
            }
            ForeignError::RuntimeError { message } => {
                VmError::Runtime(message)
            }
            ForeignError::ArgumentError { expected, actual } => {
                VmError::Runtime(format!("Argument error: expected {}, got {}", expected, actual))
            }
            ForeignError::TypeError { expected, actual } => {
                VmError::TypeError { expected, actual }
            }
            ForeignError::InvalidArgument(message) => {
                VmError::Runtime(message)
            }
        }
    }
    
    fn unified_to_vm_error(err: UnifiedError) -> VmError {
        match err {
            UnifiedError::ArgumentError { message } => {
                VmError::Runtime(format!("Argument error: {}", message))
            }
            UnifiedError::TypeError { expected, actual } => {
                VmError::TypeError { expected, actual }
            }
            UnifiedError::IndexOutOfBounds { index, max } => {
                VmError::Runtime(format!("Index {} out of bounds [0, {}]", index, max))
            }
            UnifiedError::UnsupportedOperation { operation, type_name } => {
                VmError::Runtime(format!("Operation '{}' not supported for type '{}'", operation, type_name))
            }
            UnifiedError::RuntimeError { message } => {
                VmError::Runtime(message)
            }
            UnifiedError::MathError { message } => {
                VmError::Runtime(format!("Math error: {}", message))
            }
            UnifiedError::AlgorithmError { message } => {
                VmError::Runtime(format!("Algorithm error: {}", message))
            }
            UnifiedError::ValidationError { message } => {
                VmError::Runtime(format!("Validation error: {}", message))
            }
        }
    }
    
    fn unified_to_foreign_error(err: UnifiedError) -> ForeignError {
        match err {
            UnifiedError::ArgumentError { message } => {
                ForeignError::InvalidArgument(message)
            }
            UnifiedError::TypeError { expected, actual } => {
                ForeignError::TypeError { expected, actual }
            }
            UnifiedError::IndexOutOfBounds { index, max } => {
                ForeignError::IndexOutOfBounds { 
                    index: index.to_string(), 
                    bounds: format!("[0, {}]", max)
                }
            }
            UnifiedError::UnsupportedOperation { operation, type_name } => {
                ForeignError::UnknownMethod { 
                    method: operation, 
                    type_name 
                }
            }
            _ => ForeignError::RuntimeError { 
                message: err.to_string() 
            }
        }
    }
}

/// Convenience macros for creating common error types
#[macro_export]
macro_rules! argument_error {
    ($msg:expr) => {
        crate::stdlib::common::errors::UnifiedError::ArgumentError {
            message: $msg.to_string()
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        crate::stdlib::common::errors::UnifiedError::ArgumentError {
            message: format!($fmt, $($arg)*)
        }
    };
}

#[macro_export]
macro_rules! type_error {
    ($expected:expr, $actual:expr) => {
        crate::stdlib::common::errors::UnifiedError::TypeError {
            expected: $expected.to_string(),
            actual: $actual.to_string()
        }
    };
}

#[macro_export]
macro_rules! math_error {
    ($msg:expr) => {
        crate::stdlib::common::errors::UnifiedError::MathError {
            message: $msg.to_string()
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        crate::stdlib::common::errors::UnifiedError::MathError {
            message: format!($fmt, $($arg)*)
        }
    };
}

#[macro_export]
macro_rules! algorithm_error {
    ($msg:expr) => {
        crate::stdlib::common::errors::UnifiedError::AlgorithmError {
            message: $msg.to_string()
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        crate::stdlib::common::errors::UnifiedError::AlgorithmError {
            message: format!($fmt, $($arg)*)
        }
    };
}

#[macro_export]
macro_rules! runtime_error {
    ($msg:expr) => {
        crate::stdlib::common::errors::UnifiedError::RuntimeError {
            message: $msg.to_string()
        }
    };
    ($fmt:expr, $($arg:tt)*) => {
        crate::stdlib::common::errors::UnifiedError::RuntimeError {
            message: format!($fmt, $($arg)*)
        }
    };
}

/// Utility function to convert any error to a VmError
pub fn to_vm_error<E: ToString>(error: E) -> VmError {
    VmError::Runtime(error.to_string())
}

/// Result type alias for stdlib operations
pub type StdlibResult<T> = Result<T, UnifiedError>;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_conversions() {
        let foreign_err = ForeignError::TypeError {
            expected: "Number".to_string(),
            actual: "String".to_string(),
        };
        
        let vm_err = <()>::foreign_to_vm_error(foreign_err);
        match vm_err {
            VmError::TypeError { expected, actual } => {
                assert_eq!(expected, "Number");
                assert_eq!(actual, "String");
            }
            _ => panic!("Wrong error type converted"),
        }
    }
    
    #[test]
    fn test_unified_error_creation() {
        let err = argument_error!("Invalid argument count: {}", 3);
        match err {
            UnifiedError::ArgumentError { message } => {
                assert_eq!(message, "Invalid argument count: 3");
            }
            _ => panic!("Wrong error type created"),
        }
    }
}