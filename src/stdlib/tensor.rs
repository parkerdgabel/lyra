//! Tensor operations for the Lyra standard library
//!
//! This module provides comprehensive tensor operations compatible with
//! Wolfram Language, including creation, manipulation, and arithmetic operations.

use crate::vm::{Value, VmError, VmResult};
use ndarray::{ArrayD, IxDyn};

/// Create an array (tensor) from nested lists
/// Usage: Array[{1, 2, 3}] -> 1D tensor, Array[{{1, 2}, {3, 4}}] -> 2D tensor
pub fn array(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => {
            // Convert nested list structure to tensor
            let (shape, data) = extract_shape_and_data(list)?;
            let tensor = ArrayD::from_shape_vec(IxDyn(&shape), data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            Ok(Value::Tensor(tensor))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Helper function to extract shape and flattened data from nested lists
fn extract_shape_and_data(list: &[Value]) -> VmResult<(Vec<usize>, Vec<f64>)> {
    if list.is_empty() {
        return Ok((vec![0], vec![]));
    }

    // Determine if this is a flat list of numbers or nested lists
    let first_element = &list[0];
    match first_element {
        Value::Integer(_) | Value::Real(_) => {
            // Flat list of numbers -> 1D tensor
            let mut data = Vec::new();
            for value in list {
                match value {
                    Value::Integer(n) => data.push(*n as f64),
                    Value::Real(r) => data.push(*r),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric values".to_string(),
                        actual: format!("mixed types in list: {:?}", value),
                    }),
                }
            }
            Ok((vec![list.len()], data))
        }
        Value::List(inner_list) => {
            // Nested lists -> multi-dimensional tensor
            let mut all_data = Vec::new();
            let inner_len = inner_list.len();
            
            // Verify all sublists have the same length
            for value in list {
                match value {
                    Value::List(sublist) => {
                        if sublist.len() != inner_len {
                            return Err(VmError::TypeError {
                                expected: "consistent sublist lengths".to_string(),
                                actual: format!("inconsistent lengths: {} vs {}", inner_len, sublist.len()),
                            });
                        }
                        let (_, mut subdata) = extract_shape_and_data(sublist)?;
                        all_data.append(&mut subdata);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "nested lists".to_string(),
                        actual: format!("mixed types: {:?}", value),
                    }),
                }
            }
            
            // Build shape: outer length + inner shape
            let (inner_shape, _) = extract_shape_and_data(inner_list)?;
            let mut shape = vec![list.len()];
            shape.extend(inner_shape);
            
            Ok((shape, all_data))
        }
        _ => Err(VmError::TypeError {
            expected: "numeric values or nested lists".to_string(),
            actual: format!("unsupported type: {:?}", first_element),
        }),
    }
}

/// Get the dimensions (shape) of a tensor
/// Usage: ArrayDimensions[Array[{{1, 2}, {3, 4}}]] -> {2, 2}
pub fn array_dimensions(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Tensor(tensor) => {
            let dims: Vec<Value> = tensor.shape()
                .iter()
                .map(|&dim| Value::Integer(dim as i64))
                .collect();
            Ok(Value::List(dims))
        }
        _ => Err(VmError::TypeError {
            expected: "Tensor".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get the rank (number of dimensions) of a tensor
/// Usage: ArrayRank[Array[{{1, 2}, {3, 4}}]] -> 2
pub fn array_rank(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Tensor(tensor) => Ok(Value::Integer(tensor.ndim() as i64)),
        _ => Err(VmError::TypeError {
            expected: "Tensor".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Reshape a tensor to new dimensions
/// Usage: ArrayReshape[Array[{1, 2, 3, 4}], {2, 2}] -> {{1, 2}, {3, 4}}
pub fn array_reshape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => return Err(VmError::TypeError {
            expected: "Tensor".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let new_shape = match &args[1] {
        Value::List(dims) => {
            let mut shape = Vec::new();
            for dim in dims {
                match dim {
                    Value::Integer(n) => {
                        if *n < 0 {
                            return Err(VmError::TypeError {
                                expected: "positive dimension".to_string(),
                                actual: format!("negative dimension: {}", n),
                            });
                        }
                        shape.push(*n as usize);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "integer dimensions".to_string(),
                        actual: format!("non-integer dimension: {:?}", dim),
                    }),
                }
            }
            shape
        }
        _ => return Err(VmError::TypeError {
            expected: "List of dimensions".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Verify the total number of elements matches
    let old_size = tensor.len();
    let new_size: usize = new_shape.iter().product();
    if old_size != new_size {
        return Err(VmError::TypeError {
            expected: format!("shape with {} elements", old_size),
            actual: format!("shape with {} elements", new_size),
        });
    }

    let reshaped = tensor.clone().into_shape_with_order(IxDyn(&new_shape))
        .map_err(|e| VmError::TypeError {
            expected: "valid reshape operation".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(reshaped))
}

/// Flatten a tensor to a 1D array
/// Usage: ArrayFlatten[Array[{{1, 2}, {3, 4}}]] -> {1, 2, 3, 4}
pub fn array_flatten(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Tensor(tensor) => {
            let flattened = tensor.clone().into_shape_with_order(IxDyn(&[tensor.len()]))
                .map_err(|e| VmError::TypeError {
                    expected: "valid flatten operation".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            Ok(Value::Tensor(flattened))
        }
        _ => Err(VmError::TypeError {
            expected: "Tensor".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Helper function for tensor arithmetic operations
/// Returns the result of a + b where a and b can be tensors or scalars
pub fn tensor_add(a: &Value, b: &Value) -> VmResult<Value> {
    match (a, b) {
        // Tensor + Tensor
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            let result = &broadcast_a + &broadcast_b;
            Ok(Value::Tensor(result))
        }
        
        // Tensor + Scalar (Real)
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = tensor + &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Real) + Tensor
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = &scalar_tensor + tensor;
            Ok(Value::Tensor(result))
        }
        
        // Tensor + Scalar (Integer - promote to Real)
        (Value::Tensor(tensor), Value::Integer(int_val)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = tensor + &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Integer) + Tensor
        (Value::Integer(int_val), Value::Tensor(tensor)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = &scalar_tensor + tensor;
            Ok(Value::Tensor(result))
        }
        
        // All other combinations are type errors
        _ => Err(VmError::TypeError {
            expected: "Tensor and Tensor, Tensor and Number, or Number and Tensor".to_string(),
            actual: format!("{:?} and {:?}", a, b),
        }),
    }
}

/// Helper function for tensor subtraction operations  
/// Returns the result of a - b where a and b can be tensors or scalars
pub fn tensor_sub(a: &Value, b: &Value) -> VmResult<Value> {
    match (a, b) {
        // Tensor - Tensor
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            let result = &broadcast_a - &broadcast_b;
            Ok(Value::Tensor(result))
        }
        
        // Tensor - Scalar (Real)
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = tensor - &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Real) - Tensor
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = &scalar_tensor - tensor;
            Ok(Value::Tensor(result))
        }
        
        // Tensor - Scalar (Integer - promote to Real)
        (Value::Tensor(tensor), Value::Integer(int_val)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = tensor - &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Integer) - Tensor
        (Value::Integer(int_val), Value::Tensor(tensor)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = &scalar_tensor - tensor;
            Ok(Value::Tensor(result))
        }
        
        // All other combinations are type errors
        _ => Err(VmError::TypeError {
            expected: "Tensor and Tensor, Tensor and Number, or Number and Tensor".to_string(),
            actual: format!("{:?} and {:?}", a, b),
        }),
    }
}

/// Helper function for tensor multiplication operations
/// Returns the result of a * b where a and b can be tensors or scalars (element-wise)
pub fn tensor_mul(a: &Value, b: &Value) -> VmResult<Value> {
    match (a, b) {
        // Tensor * Tensor
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            let result = &broadcast_a * &broadcast_b;
            Ok(Value::Tensor(result))
        }
        
        // Tensor * Scalar (Real)
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = tensor * &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Real) * Tensor
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = &scalar_tensor * tensor;
            Ok(Value::Tensor(result))
        }
        
        // Tensor * Scalar (Integer - promote to Real)
        (Value::Tensor(tensor), Value::Integer(int_val)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = tensor * &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Integer) * Tensor
        (Value::Integer(int_val), Value::Tensor(tensor)) => {
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = &scalar_tensor * tensor;
            Ok(Value::Tensor(result))
        }
        
        // All other combinations are type errors
        _ => Err(VmError::TypeError {
            expected: "Tensor and Tensor, Tensor and Number, or Number and Tensor".to_string(),
            actual: format!("{:?} and {:?}", a, b),
        }),
    }
}

/// Helper function for tensor division operations
/// Returns the result of a / b where a and b can be tensors or scalars
pub fn tensor_div(a: &Value, b: &Value) -> VmResult<Value> {
    match (a, b) {
        // Tensor / Tensor
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            // Check for division by zero
            for &val in tensor_b.iter() {
                if val == 0.0 {
                    return Err(VmError::DivisionByZero);
                }
            }
            
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            let result = &broadcast_a / &broadcast_b;
            Ok(Value::Tensor(result))
        }
        
        // Tensor / Scalar (Real)
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            if *scalar == 0.0 {
                return Err(VmError::DivisionByZero);
            }
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = tensor / &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Real) / Tensor
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            // Check for division by zero in tensor
            for &val in tensor.iter() {
                if val == 0.0 {
                    return Err(VmError::DivisionByZero);
                }
            }
            let scalar_tensor = broadcast_scalar_to_tensor(*scalar, tensor);
            let result = &scalar_tensor / tensor;
            Ok(Value::Tensor(result))
        }
        
        // Tensor / Scalar (Integer - promote to Real)
        (Value::Tensor(tensor), Value::Integer(int_val)) => {
            if *int_val == 0 {
                return Err(VmError::DivisionByZero);
            }
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = tensor / &scalar_tensor;
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Integer) / Tensor
        (Value::Integer(int_val), Value::Tensor(tensor)) => {
            // Check for division by zero in tensor
            for &val in tensor.iter() {
                if val == 0.0 {
                    return Err(VmError::DivisionByZero);
                }
            }
            let scalar = *int_val as f64;
            let scalar_tensor = broadcast_scalar_to_tensor(scalar, tensor);
            let result = &scalar_tensor / tensor;
            Ok(Value::Tensor(result))
        }
        
        // All other combinations are type errors
        _ => Err(VmError::TypeError {
            expected: "Tensor and Tensor, Tensor and Number, or Number and Tensor".to_string(),
            actual: format!("{:?} and {:?}", a, b),
        }),
    }
}

/// Helper function for tensor power operations
/// Returns the result of a ^ b where a and b can be tensors or scalars
pub fn tensor_pow(a: &Value, b: &Value) -> VmResult<Value> {
    match (a, b) {
        // Tensor ^ Tensor
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            // Element-wise power operation using zip
            let result = ndarray::Zip::from(&broadcast_a)
                .and(&broadcast_b)
                .map_collect(|&base, &exp| base.powf(exp));
            Ok(Value::Tensor(result))
        }
        
        // Tensor ^ Scalar (Real)
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            let result = tensor.mapv(|x| x.powf(*scalar));
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Real) ^ Tensor
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            let result = tensor.mapv(|x| scalar.powf(x));
            Ok(Value::Tensor(result))
        }
        
        // Tensor ^ Scalar (Integer - promote to Real)
        (Value::Tensor(tensor), Value::Integer(int_val)) => {
            let exp = *int_val as f64;
            let result = tensor.mapv(|x| x.powf(exp));
            Ok(Value::Tensor(result))
        }
        
        // Scalar (Integer) ^ Tensor
        (Value::Integer(int_val), Value::Tensor(tensor)) => {
            let base = *int_val as f64;
            let result = tensor.mapv(|x| base.powf(x));
            Ok(Value::Tensor(result))
        }
        
        // All other combinations are type errors
        _ => Err(VmError::TypeError {
            expected: "Tensor and Tensor, Tensor and Number, or Number and Tensor".to_string(),
            actual: format!("{:?} and {:?}", a, b),
        }),
    }
}

/// Helper function to broadcast two tensors to compatible shapes
/// Returns (broadcasted_a, broadcasted_b) or error if incompatible
fn broadcast_tensors(a: &ArrayD<f64>, b: &ArrayD<f64>) -> VmResult<(ArrayD<f64>, ArrayD<f64>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // If shapes are identical, no broadcasting needed
    if a_shape == b_shape {
        return Ok((a.clone(), b.clone()));
    }
    
    // Determine the output shape using numpy broadcasting rules
    let max_ndim = a_shape.len().max(b_shape.len());
    let mut output_shape = vec![1; max_ndim];
    
    // Work backwards from the trailing dimensions
    for i in 0..max_ndim {
        let a_dim = if i < a_shape.len() {
            a_shape[a_shape.len() - 1 - i]
        } else {
            1
        };
        
        let b_dim = if i < b_shape.len() {
            b_shape[b_shape.len() - 1 - i]
        } else {
            1
        };
        
        // Check if dimensions are compatible
        if a_dim == b_dim {
            output_shape[max_ndim - 1 - i] = a_dim;
        } else if a_dim == 1 {
            output_shape[max_ndim - 1 - i] = b_dim;
        } else if b_dim == 1 {
            output_shape[max_ndim - 1 - i] = a_dim;
        } else {
            return Err(VmError::TypeError {
                expected: "broadcastable shapes".to_string(),
                actual: format!("cannot broadcast shapes {:?} and {:?}", a_shape, b_shape),
            });
        }
    }
    
    // Broadcast both tensors to the output shape
    let a_broadcast = broadcast_tensor_to_shape(a, &output_shape)?;
    let b_broadcast = broadcast_tensor_to_shape(b, &output_shape)?;
    
    Ok((a_broadcast, b_broadcast))
}

/// Helper function to broadcast a tensor to a specific shape
fn broadcast_tensor_to_shape(tensor: &ArrayD<f64>, target_shape: &[usize]) -> VmResult<ArrayD<f64>> {
    let current_shape = tensor.shape();
    
    // If already the right shape, return a clone
    if current_shape == target_shape {
        return Ok(tensor.clone());
    }
    
    // Use ndarray's broadcast method
    match tensor.broadcast(IxDyn(target_shape)) {
        Some(broadcasted) => {
            // Convert broadcasted view to owned array
            Ok(broadcasted.to_owned())
        }
        None => Err(VmError::TypeError {
            expected: "valid broadcasting operation".to_string(),
            actual: format!("cannot broadcast shape {:?} to {:?}", current_shape, target_shape),
        })
    }
}

/// Helper function to broadcast a scalar to a tensor shape
fn broadcast_scalar_to_tensor(scalar: f64, tensor: &ArrayD<f64>) -> ArrayD<f64> {
    // Create a new tensor with the same shape as the input tensor, filled with the scalar value
    ArrayD::from_elem(tensor.shape(), scalar)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_array_creation_from_list() {
        // RED phase: These tests should fail initially
        let result = array(&[Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ])]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[3]);
                assert_eq!(tensor[[0]], 1.0);
                assert_eq!(tensor[[1]], 2.0);
                assert_eq!(tensor[[2]], 3.0);
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_array_creation_2d() {
        // Test creating 2D tensor from nested lists
        let nested_list = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ]);
        
        let result = array(&[nested_list]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[2, 2]);
                assert_eq!(tensor[[0, 0]], 1.0);
                assert_eq!(tensor[[0, 1]], 2.0);
                assert_eq!(tensor[[1, 0]], 3.0);
                assert_eq!(tensor[[1, 1]], 4.0);
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_array_dimensions() {
        // Create a test tensor
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        let result = array_dimensions(&[Value::Tensor(tensor)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::List(dims) => {
                assert_eq!(dims.len(), 2);
                assert_eq!(dims[0], Value::Integer(2));
                assert_eq!(dims[1], Value::Integer(3));
            }
            _ => panic!("Expected list of dimensions"),
        }
    }

    #[test]
    fn test_array_rank() {
        // Create a test tensor
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 3, 4]), vec![0.0; 24]).unwrap();
        
        let result = array_rank(&[Value::Tensor(tensor)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Integer(rank) => assert_eq!(rank, 3),
            _ => panic!("Expected integer rank"),
        }
    }

    #[test]
    fn test_array_reshape() {
        // Create a 1D tensor
        let tensor = ArrayD::from_shape_vec(IxDyn(&[4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let new_shape = Value::List(vec![Value::Integer(2), Value::Integer(2)]);
        
        let result = array_reshape(&[Value::Tensor(tensor), new_shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(reshaped) => {
                assert_eq!(reshaped.shape(), &[2, 2]);
                assert_eq!(reshaped[[0, 0]], 1.0);
                assert_eq!(reshaped[[0, 1]], 2.0);
                assert_eq!(reshaped[[1, 0]], 3.0);
                assert_eq!(reshaped[[1, 1]], 4.0);
            }
            _ => panic!("Expected reshaped tensor"),
        }
    }

    #[test]
    fn test_array_flatten() {
        // Create a 2D tensor
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        let result = array_flatten(&[Value::Tensor(tensor)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(flattened) => {
                assert_eq!(flattened.shape(), &[4]);
                assert_eq!(flattened[[0]], 1.0);
                assert_eq!(flattened[[1]], 2.0);
                assert_eq!(flattened[[2]], 3.0);
                assert_eq!(flattened[[3]], 4.0);
            }
            _ => panic!("Expected flattened tensor"),
        }
    }

    #[test]
    fn test_array_wrong_args() {
        // Test error cases
        assert!(array(&[]).is_err());
        assert!(array(&[Value::Integer(1), Value::Integer(2)]).is_err());
        assert!(array_dimensions(&[]).is_err());
        assert!(array_rank(&[]).is_err());
    }

    #[test]
    fn test_array_wrong_types() {
        // Test type errors
        assert!(array(&[Value::String("not a list".to_string())]).is_err());
        assert!(array_dimensions(&[Value::Integer(42)]).is_err());
        assert!(array_rank(&[Value::Boolean(true)]).is_err());
    }

    // ===== TENSOR ARITHMETIC TESTS (TDD - RED PHASE) =====

    #[test]
    fn test_tensor_add_same_shape() {
        // Create two 1D tensors with same shape
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![4.0, 5.0, 6.0]).unwrap();
        
        let result = tensor_add(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                assert_eq!(result_tensor[[0]], 5.0); // 1 + 4
                assert_eq!(result_tensor[[1]], 7.0); // 2 + 5  
                assert_eq!(result_tensor[[2]], 9.0); // 3 + 6
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_add_2d_same_shape() {
        // Create two 2D tensors with same shape
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let result = tensor_add(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 6.0);  // 1 + 5
                assert_eq!(result_tensor[[0, 1]], 8.0);  // 2 + 6
                assert_eq!(result_tensor[[1, 0]], 10.0); // 3 + 7
                assert_eq!(result_tensor[[1, 1]], 12.0); // 4 + 8
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test] 
    fn test_tensor_add_scalar_to_tensor() {
        // Add scalar to tensor
        let tensor = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let scalar = 10.0;
        
        let result = tensor_add(&Value::Tensor(tensor), &Value::Real(scalar));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                assert_eq!(result_tensor[[0]], 11.0); // 1 + 10
                assert_eq!(result_tensor[[1]], 12.0); // 2 + 10
                assert_eq!(result_tensor[[2]], 13.0); // 3 + 10
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_add_tensor_to_scalar() {
        // Add tensor to scalar (commutative)
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let scalar = 5.0;
        
        let result = tensor_add(&Value::Real(scalar), &Value::Tensor(tensor));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 6.0); // 5 + 1
                assert_eq!(result_tensor[[0, 1]], 7.0); // 5 + 2
                assert_eq!(result_tensor[[1, 0]], 8.0); // 5 + 3
                assert_eq!(result_tensor[[1, 1]], 9.0); // 5 + 4
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_add_broadcasting() {
        // Test broadcasting: [2, 3] + [3] -> [2, 3]
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
        
        let result = tensor_add(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 3]);
                // First row: [1,2,3] + [10,20,30] = [11,22,33]
                assert_eq!(result_tensor[[0, 0]], 11.0);
                assert_eq!(result_tensor[[0, 1]], 22.0); 
                assert_eq!(result_tensor[[0, 2]], 33.0);
                // Second row: [4,5,6] + [10,20,30] = [14,25,36]
                assert_eq!(result_tensor[[1, 0]], 14.0);
                assert_eq!(result_tensor[[1, 1]], 25.0);
                assert_eq!(result_tensor[[1, 2]], 36.0);
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_add_incompatible_shapes() {
        // Test incompatible shapes that can't be broadcast
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[2, 3]), vec![1.0; 6]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0; 4]).unwrap();
        
        let result = tensor_add(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_sub_basic() {
        // Test tensor subtraction
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![10.0, 8.0, 6.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        
        let result = tensor_sub(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 9.0);  // 10 - 1
                assert_eq!(result_tensor[[1]], 6.0);  // 8 - 2
                assert_eq!(result_tensor[[2]], 3.0);  // 6 - 3
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_mul_basic() {
        // Test element-wise tensor multiplication
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![5.0, 6.0, 7.0]).unwrap();
        
        let result = tensor_mul(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 10.0); // 2 * 5
                assert_eq!(result_tensor[[1]], 18.0); // 3 * 6
                assert_eq!(result_tensor[[2]], 28.0); // 4 * 7
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_div_basic() {
        // Test element-wise tensor division
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![12.0, 15.0, 20.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![3.0, 5.0, 4.0]).unwrap();
        
        let result = tensor_div(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 4.0); // 12 / 3
                assert_eq!(result_tensor[[1]], 3.0); // 15 / 5
                assert_eq!(result_tensor[[2]], 5.0); // 20 / 4
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_div_by_zero() {
        // Test division by zero error
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 1.0]).unwrap();
        
        let result = tensor_div(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_pow_basic() {
        // Test element-wise tensor power
        let tensor_a = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 3.0, 4.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(IxDyn(&[3]), vec![2.0, 2.0, 2.0]).unwrap();
        
        let result = tensor_pow(&Value::Tensor(tensor_a), &Value::Tensor(tensor_b));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 4.0);  // 2^2
                assert_eq!(result_tensor[[1]], 9.0);  // 3^2
                assert_eq!(result_tensor[[2]], 16.0); // 4^2
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_arithmetic_with_integers() {
        // Test tensor arithmetic with Integer values (should be promoted to Real)
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        
        let result = tensor_add(&Value::Tensor(tensor), &Value::Integer(5));
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 6.0); // 1 + 5
                assert_eq!(result_tensor[[1]], 7.0); // 2 + 5
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_arithmetic_type_errors() {
        // Test operations with unsupported types
        let tensor = ArrayD::from_shape_vec(IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        
        // String + tensor should error
        let result = tensor_add(&Value::String("hello".to_string()), &Value::Tensor(tensor.clone()));
        assert!(result.is_err());
        
        // Tensor + list should error
        let result = tensor_add(&Value::Tensor(tensor.clone()), &Value::List(vec![Value::Integer(1)]));
        assert!(result.is_err());
    }
}