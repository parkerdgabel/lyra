//! ML Framework Wrapper Functions for Stdlib Integration
//!
//! This module provides wrapper functions that bridge ML layer implementations
//! with the standard library function system, enabling tree-shaking optimized imports.

use crate::{
    vm::{Value, VmResult},
    stdlib::ml::{
        layers::{FlattenLayer, ReshapeLayer, PermuteLayer, Layer},
        MLError, MLResult, Tensor,
    },
    stdlib::autodiff::Dual,
};

/// Convert a stdlib Value to a Tensor for ML operations
fn value_to_tensor(value: &Value) -> MLResult<Tensor> {
    match value {
        Value::LyObj(_obj) => {
            // Try to extract tensor from LyObj
            // For now, this is a placeholder - we would need proper type checking
            Err(MLError::DataError {
                reason: "LyObj tensor conversion not yet implemented".to_string(),
            })
        },
        Value::List(elements) => {
            // Handle nested lists to create multidimensional tensors
            // This is a simplified implementation
            let dual_values: Result<Vec<Dual>, _> = elements.iter()
                .map(|v| match v {
                    Value::Real(n) => Ok(Dual::constant(*n)),
                    Value::Integer(n) => Ok(Dual::constant(*n as f64)),
                    Value::List(_) => Err(MLError::DataError {
                        reason: "Nested list conversion not yet implemented".to_string(),
                    }),
                    _ => Err(MLError::DataError {
                        reason: format!("Cannot convert {:?} to tensor", v),
                    }),
                })
                .collect();
            
            let data = dual_values?;
            let shape = vec![data.len()];
            Tensor::new(data, shape)
        },
        Value::Real(n) => {
            // Convert single number to 0D tensor (scalar)
            let data = vec![Dual::constant(*n)];
            let shape = vec![];
            Tensor::new(data, shape)
        },
        Value::Integer(n) => {
            // Convert single integer to 0D tensor (scalar)
            let data = vec![Dual::constant(*n as f64)];
            let shape = vec![];
            Tensor::new(data, shape)
        },
        _ => Err(MLError::DataError {
            reason: format!("Cannot convert {:?} to tensor", value),
        }),
    }
}

/// Convert a Tensor back to a stdlib Value
fn tensor_to_value(tensor: &Tensor) -> Value {
    // For now, we'll convert back to a simple list representation
    // In a full implementation, we might want to use LyObj with a proper foreign tensor
    let values: Vec<Value> = tensor.data.iter()
        .map(|dual| Value::Real(dual.value))
        .collect();
    Value::List(values)
}

/// Extract integer list from Value for shape specifications
fn value_to_shape(value: &Value) -> MLResult<Vec<i32>> {
    match value {
        Value::List(elements) => {
            elements.iter()
                .map(|v| match v {
                    Value::Real(n) => Ok(*n as i32),
                    Value::Integer(n) => Ok(*n as i32),
                    _ => Err(MLError::DataError {
                        reason: format!("Shape must be a list of integers, got {:?}", v),
                    }),
                })
                .collect()
        },
        Value::Real(n) => Ok(vec![*n as i32]),
        Value::Integer(n) => Ok(vec![*n as i32]),
        _ => Err(MLError::DataError {
            reason: format!("Shape must be a list of integers, got {:?}", value),
        }),
    }
}

/// Extract usize list from Value for permutation specifications
fn value_to_usize_list(value: &Value) -> MLResult<Vec<usize>> {
    match value {
        Value::List(elements) => {
            elements.iter()
                .map(|v| match v {
                    Value::Real(n) => {
                        if *n >= 0.0 && *n == n.floor() {
                            Ok(*n as usize)
                        } else {
                            Err(MLError::DataError {
                                reason: format!("Permutation indices must be non-negative integers, got {}", n),
                            })
                        }
                    },
                    Value::Integer(n) => {
                        if *n >= 0 {
                            Ok(*n as usize)
                        } else {
                            Err(MLError::DataError {
                                reason: format!("Permutation indices must be non-negative integers, got {}", n),
                            })
                        }
                    },
                    _ => Err(MLError::DataError {
                        reason: format!("Permutation must be a list of integers, got {:?}", v),
                    }),
                })
                .collect()
        },
        _ => Err(MLError::DataError {
            reason: format!("Permutation must be a list of integers, got {:?}", value),
        }),
    }
}

/// Handle ML errors by converting to VM errors
fn handle_ml_error(error: MLError) -> crate::vm::VmError {
    match error {
        MLError::ShapeMismatch { expected, actual } => {
            crate::vm::VmError::TypeError {
                expected: format!("tensor with shape {:?}", expected),
                actual: format!("tensor with shape {:?}", actual),
            }
        },
        MLError::InvalidLayer { reason } => {
            crate::vm::VmError::TypeError {
                expected: "valid layer configuration".to_string(),
                actual: reason,
            }
        },
        MLError::DataError { reason } => {
            crate::vm::VmError::TypeError {
                expected: "valid tensor data".to_string(),
                actual: reason,
            }
        },
        MLError::AutodiffError(e) => {
            crate::vm::VmError::TypeError {
                expected: "valid autodiff operation".to_string(),
                actual: format!("Autodiff error: {:?}", e),
            }
        },
        _ => crate::vm::VmError::TypeError {
            expected: "successful ML operation".to_string(),
            actual: format!("ML operation failed: {:?}", error),
        },
    }
}

// ============================================================================
// SPATIAL LAYER WRAPPER FUNCTIONS
// ============================================================================

/// FlattenLayer[tensor] - Flatten tensor from dimension 1 onwards
/// FlattenLayer[tensor, start_dim] - Flatten from specified start dimension
/// FlattenLayer[tensor, start_dim, end_dim] - Flatten specified dimension range
pub fn flatten_layer(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 3 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1, 2, or 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    
    let layer = match args.len() {
        1 => FlattenLayer::from_dim(1), // Default: flatten from dim 1
        2 => {
            let start_dim = match &args[1] {
                Value::Real(n) => *n as usize,
                Value::Integer(n) => *n as usize,
                _ => return Err(crate::vm::VmError::TypeError {
                    expected: "number".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            };
            FlattenLayer::from_dim(start_dim)
        },
        3 => {
            let start_dim = match &args[1] {
                Value::Real(n) => *n as usize,
                Value::Integer(n) => *n as usize,
                _ => return Err(crate::vm::VmError::TypeError {
                    expected: "number".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            };
            let end_dim = match &args[2] {
                Value::Real(n) => *n as usize,
                Value::Integer(n) => *n as usize,
                _ => return Err(crate::vm::VmError::TypeError {
                    expected: "number".to_string(),
                    actual: format!("{:?}", args[2]),
                }),
            };
            FlattenLayer::from_range(start_dim, end_dim)
        },
        _ => unreachable!(),
    };

    let result = layer.forward(&tensor).map_err(handle_ml_error)?;
    Ok(tensor_to_value(&result))
}

/// ReshapeLayer[tensor, shape] - Reshape tensor to new shape (supports -1 for inference)
pub fn reshape_layer(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(crate::vm::VmError::TypeError {
            expected: "2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    let shape = value_to_shape(&args[1]).map_err(handle_ml_error)?;
    
    let layer = ReshapeLayer::new(shape);
    let result = layer.forward(&tensor).map_err(handle_ml_error)?;
    Ok(tensor_to_value(&result))
}

/// PermuteLayer[tensor, permutation] - Reorder tensor dimensions according to permutation
pub fn permute_layer(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(crate::vm::VmError::TypeError {
            expected: "2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    let permutation = value_to_usize_list(&args[1]).map_err(handle_ml_error)?;
    
    let layer = PermuteLayer::new(permutation);
    let result = layer.forward(&tensor).map_err(handle_ml_error)?;
    Ok(tensor_to_value(&result))
}

/// TransposeLayer[tensor] - Special case of PermuteLayer for 2D transpose (swap dimensions)
pub fn transpose_layer(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    
    if tensor.shape.len() != 2 {
        return Err(crate::vm::VmError::TypeError {
            expected: "2D tensor".to_string(),
            actual: format!("{}D tensor", tensor.shape.len()),
        });
    }
    
    let layer = PermuteLayer::new(vec![1, 0]); // Swap first two dimensions
    let result = layer.forward(&tensor).map_err(handle_ml_error)?;
    Ok(tensor_to_value(&result))
}

// ============================================================================
// LAYER COMPOSITION FUNCTIONS
// ============================================================================

/// Sequential[layers...] - Compose multiple layers sequentially
/// This would apply each layer in order to the input
pub fn sequential_layer(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(crate::vm::VmError::TypeError {
            expected: "at least 1 argument".to_string(),
            actual: "0 arguments".to_string(),
        });
    }

    // For now, just return the first argument (placeholder implementation)
    // A full implementation would need to handle layer composition
    Ok(args[0].clone())
}

/// Identity[tensor] - Identity layer (returns input unchanged)
pub fn identity_layer(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Identity function - just return the input
    Ok(args[0].clone())
}

// ============================================================================
// SHAPE UTILITY FUNCTIONS
// ============================================================================

/// TensorShape[tensor] - Get the shape of a tensor
pub fn tensor_shape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    let shape_values: Vec<Value> = tensor.shape.iter()
        .map(|&dim| Value::Real(dim as f64))
        .collect();
    
    Ok(Value::List(shape_values))
}

/// TensorRank[tensor] - Get the number of dimensions of a tensor
pub fn tensor_rank(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    Ok(Value::Real(tensor.shape.len() as f64))
}

/// TensorSize[tensor] - Get the total number of elements in a tensor
pub fn tensor_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::TypeError {
            expected: "1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = value_to_tensor(&args[0]).map_err(handle_ml_error)?;
    let total_size: usize = tensor.shape.iter().product();
    Ok(Value::Real(total_size as f64))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_layer_wrapper() {
        // Test 1 argument version (default flatten from dim 0 for 1D tensor)
        let input_value = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), 
            Value::Real(4.0), Value::Real(5.0), Value::Real(6.0)
        ]);
        
        // For 1D tensor, use start_dim = 0
        let result = flatten_layer(&[input_value, Value::Real(0.0)]).unwrap();
        if let Value::List(result_values) = result {
            assert_eq!(result_values.len(), 6);
        } else {
            panic!("Expected List result");
        }
    }

    #[test]
    fn test_reshape_layer_wrapper() {
        let input_value = Value::List(vec![
            Value::Real(0.0), Value::Real(1.0), Value::Real(2.0),
            Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        let shape_value = Value::List(vec![Value::Real(3.0), Value::Real(2.0)]);
        
        let result = reshape_layer(&[input_value, shape_value]).unwrap();
        if let Value::List(result_values) = result {
            assert_eq!(result_values.len(), 6);
        } else {
            panic!("Expected List result");
        }
    }

    #[test]
    fn test_permute_layer_wrapper() {
        // Test error handling since current Value representation has limitations
        // In a real implementation, this would work with proper tensor storage
        let input_value = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ]);
        // This permutation should fail for 1D input
        let permutation_value = Value::List(vec![Value::Integer(1), Value::Integer(0)]);
        let result = permute_layer(&[input_value, permutation_value]);
        
        // Should get an error because 1D tensor can't be permuted with 2D permutation
        assert!(result.is_err());
        if let Err(err) = result {
            match err {
                crate::vm::VmError::TypeError { expected, actual } => {
                    assert!(actual.contains("Permutation length 2 does not match input rank 1"));
                }
                _ => panic!("Expected TypeError"),
            }
        }
    }

    #[test]
    fn test_tensor_shape_wrapper() {
        let input_value = Value::List(vec![
            Value::Real(0.0), Value::Real(1.0), Value::Real(2.0),
            Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        
        let result = tensor_shape(&[input_value]).unwrap();
        if let Value::List(shape_list) = result {
            assert_eq!(shape_list.len(), 1);
            assert_eq!(shape_list[0], Value::Real(6.0));
        } else {
            panic!("Expected List result");
        }
    }

    #[test]
    fn test_error_handling() {
        // Test invalid arguments
        let result = flatten_layer(&[]);
        assert!(result.is_err());
        
        // Test wrong argument types
        let result = flatten_layer(&[Value::Symbol("invalid".to_string())]);
        assert!(result.is_err());
    }
}