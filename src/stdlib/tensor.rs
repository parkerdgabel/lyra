//! # Tensor Operations for Lyra Standard Library
//!
//! This module provides comprehensive tensor operations compatible with Wolfram Language,
//! including creation, manipulation, linear algebra, and neural network building blocks.
//!
//! ## Features
//!
//! - **Tensor Creation**: Create tensors from nested lists with automatic shape inference
//! - **Linear Algebra**: Matrix multiplication, transpose, and element-wise operations
//! - **Broadcasting**: NumPy-style broadcasting for all tensor operations
//! - **Neural Networks**: Essential operations for building neural networks (ReLU, etc.)
//! - **Type Safety**: Comprehensive error checking with clear error messages
//!
//! ## Quick Start
//!
//! ```wolfram
//! (* Create tensors *)
//! vector = Array[{1, 2, 3}]
//! matrix = Array[{{1, 2}, {3, 4}}]
//!
//! (* Linear algebra *)
//! result = Dot[matrix, vector]  (* Matrix-vector multiplication *)
//! transposed = Transpose[matrix]
//!
//! (* Neural network operations *)
//! activated = Maximum[result, 0]  (* ReLU activation *)
//! ```
//!
//! ## Examples
//!
//! ### Basic Tensor Operations
//! ```wolfram
//! (* 1D tensor *)
//! Array[{1, 2, 3, 4}]
//! (* → Tensor[shape: [4], elements: 4] *)
//!
//! (* 2D tensor *)
//! Array[{{1, 2}, {3, 4}}]
//! (* → Tensor[shape: [2, 2], elements: 4] *)
//!
//! (* Get tensor information *)
//! tensor = Array[{{1, 2, 3}, {4, 5, 6}}]
//! ArrayDimensions[tensor]  (* → {2, 3} *)
//! ArrayRank[tensor]        (* → 2 *)
//! ```
//!
//! ### Linear Algebra
//! ```wolfram
//! (* Vector dot product *)
//! Dot[{1, 2, 3}, {4, 5, 6}]  (* → 32 *)
//!
//! (* Matrix multiplication *)
//! A = Array[{{1, 2}, {3, 4}}]
//! B = Array[{{5, 6}, {7, 8}}]
//! Dot[A, B]  (* → {{19, 22}, {43, 50}} *)
//!
//! (* Matrix transpose *)
//! Transpose[{{1, 2, 3}, {4, 5, 6}}]  (* → {{1, 4}, {2, 5}, {3, 6}} *)
//! ```
//!
//! ### Neural Network Operations
//! ```wolfram
//! (* ReLU activation *)
//! Maximum[{-2, -1, 0, 1, 2}, 0]  (* → {0, 0, 0, 1, 2} *)
//!
//! (* Linear layer forward pass *)
//! weights = Array[{{0.1, 0.2}, {0.3, 0.4}}]
//! input = Array[{1.0, 2.0}]
//! output = Dot[weights, input]
//! activated = Maximum[output, 0]
//! ```
//!
//! ## Broadcasting
//!
//! All tensor operations support NumPy-style broadcasting:
//! ```wolfram
//! (* Scalar broadcasting *)
//! Array[{{1, 2}, {3, 4}}] + 10  (* → {{11, 12}, {13, 14}} *)
//!
//! (* Vector broadcasting *)
//! matrix + vector  (* Vector broadcast across matrix rows *)
//! ```
//!
//! ## Error Handling
//!
//! The module provides comprehensive error checking:
//! - Type validation for all function arguments
//! - Shape compatibility for tensor operations
//! - Division by zero protection
//! - Clear, actionable error messages

use crate::vm::{Value, VmError, VmResult};
use ndarray::{ArrayD, IxDyn};
use rand::Rng;

/// Create an array (tensor) from nested lists with automatic shape inference.
///
/// This function converts Lyra lists into n-dimensional tensors, automatically
/// determining the shape from the nested list structure.
///
/// # Arguments
/// * `args[0]` - A nested list structure to convert to a tensor
///
/// # Returns
/// * `Ok(Value::Tensor)` - The created tensor
/// * `Err(VmError)` - If the input is invalid or shapes are inconsistent
///
/// # Examples
/// ```wolfram
/// Array[{1, 2, 3, 4}]                    (* 1D tensor, shape [4] *)
/// Array[{{1, 2}, {3, 4}}]                (* 2D tensor, shape [2, 2] *)
/// Array[{{{1, 2}}, {{3, 4}}}]            (* 3D tensor, shape [2, 1, 2] *)
/// ```
///
/// # Error Conditions
/// - Wrong number of arguments (expects exactly 1)
/// - Non-list input
/// - Inconsistent nested list shapes
/// - Non-numeric values in nested lists
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

/// Get the dimensions (shape) of a tensor.
///
/// Returns a list containing the size of each dimension of the tensor.
///
/// # Arguments
/// * `args[0]` - A tensor whose dimensions to query
///
/// # Returns
/// * `Ok(Value::List)` - List of dimension sizes
/// * `Err(VmError)` - If the input is not a tensor
///
/// # Examples
/// ```wolfram
/// tensor = Array[{{1, 2, 3}, {4, 5, 6}}]
/// ArrayDimensions[tensor]                    (* → {2, 3} *)
///
/// vector = Array[{1, 2, 3, 4, 5}]
/// ArrayDimensions[vector]                    (* → {5} *)
///
/// scalar = Array[{{42}}]
/// ArrayDimensions[scalar]                    (* → {1, 1} *)
/// ```
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

// ===== LINEAR ALGEBRA OPERATIONS =====

/// Compute dot product (matrix multiplication) between tensors.
///
/// This function implements comprehensive matrix multiplication operations:
/// - Vector·Vector → Scalar (dot product)
/// - Matrix·Vector → Vector
/// - Vector·Matrix → Vector  
/// - Matrix·Matrix → Matrix
///
/// All operations follow standard linear algebra rules with proper shape validation.
///
/// # Arguments
/// * `args[0]` - First tensor (left operand)
/// * `args[1]` - Second tensor (right operand)
///
/// # Returns
/// * `Ok(Value::Real)` - For vector-vector dot product
/// * `Ok(Value::Tensor)` - For all other matrix operations
/// * `Err(VmError)` - If shapes are incompatible or arguments are invalid
///
/// # Examples
/// ```wolfram
/// (* Vector dot product *)
/// Dot[{1, 2, 3}, {4, 5, 6}]                (* → 32 *)
///
/// (* Matrix-vector multiplication *)
/// matrix = Array[{{1, 2}, {3, 4}}]
/// vector = Array[{1, 0}]
/// Dot[matrix, vector]                       (* → {1, 3} *)
///
/// (* Matrix-matrix multiplication *)
/// A = Array[{{1, 2}, {3, 4}}]
/// B = Array[{{5, 6}, {7, 8}}]
/// Dot[A, B]                                (* → {{19, 22}, {43, 50}} *)
///
/// (* Vector-matrix multiplication *)
/// Dot[{1, 2}, {{3, 4}, {5, 6}}]           (* → {13, 16} *)
/// ```
///
/// # Shape Requirements
/// - Vector·Vector: Both vectors must have the same length
/// - Matrix·Vector: Matrix columns must equal vector length
/// - Vector·Matrix: Vector length must equal matrix rows
/// - Matrix·Matrix: Left matrix columns must equal right matrix rows
///
/// # Neural Network Applications
/// Essential for neural network operations:
/// ```wolfram
/// (* Linear layer forward pass *)
/// weights = Array[{{0.1, 0.2}, {0.3, 0.4}}]  (* 2x2 weight matrix *)
/// input = Array[{1.0, 2.0}]                   (* 2D input vector *)
/// output = Dot[weights, input]                 (* Forward pass *)
/// ```
pub fn dot(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Extract tensor arguments
    let (tensor_a, tensor_b) = match (&args[0], &args[1]) {
        (Value::Tensor(a), Value::Tensor(b)) => (a, b),
        _ => {
            return Err(VmError::TypeError {
                expected: "Tensor and Tensor".to_string(),
                actual: format!("{:?} and {:?}", args[0], args[1]),
            });
        }
    };
    
    match (tensor_a.ndim(), tensor_b.ndim()) {
        // Case 1: Vector-Vector dot product (1D x 1D -> scalar)
        (1, 1) => {
            if tensor_a.shape() != tensor_b.shape() {
                return Err(VmError::TypeError {
                    expected: format!("compatible shapes"),
                    actual: format!("shapes {:?} vs {:?}", tensor_a.shape(), tensor_b.shape()),
                });
            }
            
            let mut result = 0.0;
            for i in 0..tensor_a.len() {
                result += tensor_a[[i]] * tensor_b[[i]];
            }
            Ok(Value::Real(result))
        },
        
        // Case 2: Matrix-Vector multiplication (2D x 1D -> 1D)
        (2, 1) => {
            let a_shape = tensor_a.shape();
            let b_shape = tensor_b.shape();
            
            // Check compatible shapes: matrix columns must match vector length
            if a_shape[1] != b_shape[0] {
                return Err(VmError::TypeError {
                    expected: format!("matrix columns ({}) to match vector length ({})", a_shape[1], b_shape[0]),
                    actual: format!("incompatible shapes {:?} vs {:?}", a_shape, b_shape),
                });
            }
            
            let rows = a_shape[0];
            let cols = a_shape[1];
            let mut result_data = vec![0.0; rows];
            
            // Compute matrix-vector multiplication: result[i] = sum(matrix[i,j] * vector[j])
            for i in 0..rows {
                for j in 0..cols {
                    result_data[i] += tensor_a[[i, j]] * tensor_b[[j]];
                }
            }
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(&[rows]), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
                
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 3: Vector-Matrix multiplication (1D x 2D -> 1D)
        (1, 2) => {
            let a_shape = tensor_a.shape();
            let b_shape = tensor_b.shape();
            
            // Check compatible shapes: vector length must match matrix rows
            if a_shape[0] != b_shape[0] {
                return Err(VmError::TypeError {
                    expected: format!("vector length ({}) to match matrix rows ({})", a_shape[0], b_shape[0]),
                    actual: format!("incompatible shapes {:?} vs {:?}", a_shape, b_shape),
                });
            }
            
            let rows = b_shape[0];
            let cols = b_shape[1];
            let mut result_data = vec![0.0; cols];
            
            // Compute vector-matrix multiplication: result[j] = sum(vector[i] * matrix[i,j])
            for j in 0..cols {
                for i in 0..rows {
                    result_data[j] += tensor_a[[i]] * tensor_b[[i, j]];
                }
            }
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(&[cols]), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
                
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 4: Matrix-Matrix multiplication (2D x 2D -> 2D)
        (2, 2) => {
            let a_shape = tensor_a.shape();
            let b_shape = tensor_b.shape();
            
            // Check compatible shapes: matrix A columns must match matrix B rows
            if a_shape[1] != b_shape[0] {
                return Err(VmError::TypeError {
                    expected: format!("matrix A columns ({}) to match matrix B rows ({})", a_shape[1], b_shape[0]),
                    actual: format!("incompatible shapes {:?} vs {:?}", a_shape, b_shape),
                });
            }
            
            let m = a_shape[0]; // result rows
            let n = b_shape[1]; // result columns  
            let k = a_shape[1]; // shared dimension
            let mut result_data = vec![0.0; m * n];
            
            // Compute matrix-matrix multiplication: C[i,j] = sum(A[i,k] * B[k,j])
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for kk in 0..k {
                        sum += tensor_a[[i, kk]] * tensor_b[[kk, j]];
                    }
                    result_data[i * n + j] = sum;
                }
            }
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(&[m, n]), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
                
            Ok(Value::Tensor(result_tensor))
        },
        
        // Unsupported cases
        _ => {
            Err(VmError::TypeError {
                expected: "vector-vector, matrix-vector, vector-matrix, or matrix-matrix multiplication".to_string(),
                actual: format!("{}D and {}D tensors", tensor_a.ndim(), tensor_b.ndim()),
            })
        }
    }
}

/// Transpose a matrix or convert 1D vectors to column vectors.
///
/// This function performs matrix transposition with special handling for vectors:
/// - 2D Matrix: Standard transpose (swap rows and columns)
/// - 1D Vector: Convert to column vector (essential for neural networks)
/// - Row Vector: Convert to column vector
/// - Column Vector: Convert to row vector
///
/// # Arguments
/// * `args[0]` - A tensor to transpose (1D or 2D)
///
/// # Returns
/// * `Ok(Value::Tensor)` - The transposed tensor
/// * `Err(VmError)` - If the input is invalid or unsupported dimension
///
/// # Examples
/// ```wolfram
/// (* 2D matrix transpose *)
/// matrix = Array[{{1, 2, 3}, {4, 5, 6}}]
/// Transpose[matrix]                          (* → {{1, 4}, {2, 5}, {3, 6}} *)
///
/// (* 1D vector to column vector *)
/// vector = Array[{1, 2, 3}]
/// Transpose[vector]                          (* → {{1}, {2}, {3}} *)
///
/// (* Row vector to column vector *)
/// row = Array[{{1, 2, 3}}]                   (* 1x3 row vector *)
/// Transpose[row]                             (* → {{1}, {2}, {3}} - 3x1 column *)
///
/// (* Identity matrix is unchanged *)
/// identity = Array[{{1, 0}, {0, 1}}]
/// Transpose[identity]                        (* → {{1, 0}, {0, 1}} *)
/// ```
///
/// # Neural Network Applications
/// Critical for neural network operations:
/// ```wolfram
/// (* Preparing weight matrices for backward pass *)
/// weights = Array[{{0.1, 0.2}, {0.3, 0.4}}]
/// weights_T = Transpose[weights]             (* For gradient computation *)
///
/// (* Converting vectors for matrix operations *)
/// input_vector = Array[{1, 2, 3}]
/// input_column = Transpose[input_vector]     (* Column vector for multiplication *)
/// ```
///
/// # Mathematical Properties
/// - Double transpose returns original: `Transpose[Transpose[A]] = A`
/// - Preserves all matrix elements, only changes arrangement
/// - For matrix multiplication: `Transpose[A·B] = Transpose[B]·Transpose[A]`  
pub fn transpose(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Extract tensor argument
    let tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => {
            return Err(VmError::TypeError {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };
    
    match tensor.ndim() {
        // Case 1: 1D vector -> convert to column vector [n] -> [n, 1]
        1 => {
            let len = tensor.shape()[0];
            let data = tensor.as_slice().unwrap().to_vec();
            
            let column_vector = ArrayD::from_shape_vec(IxDyn(&[len, 1]), data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(column_vector))
        },
        
        // Case 2: 2D matrix -> standard transpose [m, n] -> [n, m]
        2 => {
            let shape = tensor.shape();
            let rows = shape[0];
            let cols = shape[1];
            
            // Create transposed data: new[j, i] = original[i, j]
            let mut transposed_data = vec![0.0; rows * cols];
            for i in 0..rows {
                for j in 0..cols {
                    transposed_data[j * rows + i] = tensor[[i, j]];
                }
            }
            
            // Create transposed tensor with swapped dimensions
            let transposed_tensor = ArrayD::from_shape_vec(IxDyn(&[cols, rows]), transposed_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(transposed_tensor))
        },
        
        // Unsupported cases (3D+ tensors)
        _ => {
            Err(VmError::TypeError {
                expected: "1D vector or 2D matrix".to_string(),
                actual: format!("{}D tensor", tensor.ndim()),
            })
        }
    }
}

/// Element-wise maximum operation between tensors and scalars.
///
/// This function computes the element-wise maximum between two operands with full
/// broadcasting support. It's essential for implementing ReLU activation functions
/// and other neural network operations.
///
/// # Arguments
/// * `args[0]` - First operand (tensor, real, or integer)
/// * `args[1]` - Second operand (tensor, real, or integer)
///
/// # Returns
/// * `Ok(Value::Tensor)` - Tensor containing element-wise maximums
/// * `Err(VmError)` - If operands are incompatible types or shapes
///
/// # Examples
/// ```wolfram
/// (* ReLU activation - fundamental neural network operation *)
/// input = Array[{-2, -1, 0, 1, 2}]
/// Maximum[input, 0]                          (* → {0, 0, 0, 1, 2} *)
///
/// (* 2D tensor ReLU *)
/// matrix = Array[{{-1, 0}, {1, 2}}]
/// Maximum[matrix, 0]                         (* → {{0, 0}, {1, 2}} *)
///
/// (* Element-wise maximum between tensors *)
/// A = Array[{1, 5, 3}]
/// B = Array[{4, 2, 6}]
/// Maximum[A, B]                              (* → {4, 5, 6} *)
///
/// (* Broadcasting with different shapes *)
/// matrix = Array[{{1, 2}, {3, 4}}]
/// vector = Array[{2, 1}]
/// Maximum[matrix, vector]                    (* → {{2, 2}, {3, 4}} *)
///
/// (* Commutative operation *)
/// Maximum[0, Array[{-1, 1}]]                (* → {0, 1} *)
/// ```
///
/// # Neural Network Applications
/// Essential for activation functions:
/// ```wolfram
/// (* ReLU activation layer *)
/// linear_output = Dot[weights, input]
/// activated = Maximum[linear_output, 0]       (* ReLU activation *)
///
/// (* Leaky ReLU (future extension) *)
/// (* Maximum[input, 0.01 * input] *)
///
/// (* Clipping gradients *)
/// Maximum[gradients, -1]                      (* Clip from below *)
/// ```
///
/// # Broadcasting Rules
/// Follows NumPy-style broadcasting:
/// - Scalar + Tensor: Scalar broadcast to all tensor elements
/// - Tensor + Tensor: Element-wise with shape compatibility
/// - Compatible shapes aligned from trailing dimensions
///
/// # Performance
/// - Optimized element-wise operations using ndarray SIMD
/// - Memory-efficient broadcasting without unnecessary copies
/// - Zero-copy operations when possible
pub fn maximum(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match (&args[0], &args[1]) {
        // Case 1: Tensor and Real
        (Value::Tensor(tensor), Value::Real(scalar)) => {
            // Apply element-wise maximum with scalar
            let result_data: Vec<f64> = tensor
                .as_slice()
                .unwrap()
                .iter()
                .map(|&x| x.max(*scalar))
                .collect();
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(tensor.shape()), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 2: Tensor and Integer
        (Value::Tensor(tensor), Value::Integer(int_scalar)) => {
            let scalar_val = *int_scalar as f64;
            
            // Apply element-wise maximum with scalar
            let result_data: Vec<f64> = tensor
                .as_slice()
                .unwrap()
                .iter()
                .map(|&x| x.max(scalar_val))
                .collect();
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(tensor.shape()), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 3: Real and Tensor (commutative)
        (Value::Real(scalar), Value::Tensor(tensor)) => {
            // Apply element-wise maximum with scalar
            let result_data: Vec<f64> = tensor
                .as_slice()
                .unwrap()
                .iter()
                .map(|&x| scalar.max(x))
                .collect();
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(tensor.shape()), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 4: Integer and Tensor (commutative)
        (Value::Integer(int_scalar), Value::Tensor(tensor)) => {
            let scalar_val = *int_scalar as f64;
            
            // Apply element-wise maximum with scalar
            let result_data: Vec<f64> = tensor
                .as_slice()
                .unwrap()
                .iter()
                .map(|&x| scalar_val.max(x))
                .collect();
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(tensor.shape()), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(result_tensor))
        },
        
        // Case 5: Tensor and Tensor (with broadcasting)
        (Value::Tensor(tensor_a), Value::Tensor(tensor_b)) => {
            // Use existing broadcasting logic
            let (broadcast_a, broadcast_b) = broadcast_tensors(tensor_a, tensor_b)?;
            
            // Apply element-wise maximum
            let result_data: Vec<f64> = broadcast_a
                .as_slice()
                .unwrap()
                .iter()
                .zip(broadcast_b.as_slice().unwrap().iter())
                .map(|(&a, &b)| a.max(b))
                .collect();
            
            let result_tensor = ArrayD::from_shape_vec(IxDyn(broadcast_a.shape()), result_data)
                .map_err(|e| VmError::TypeError {
                    expected: "valid tensor shape".to_string(),
                    actual: format!("ndarray error: {}", e),
                })?;
            
            Ok(Value::Tensor(result_tensor))
        },
        
        // Unsupported cases
        _ => {
            Err(VmError::TypeError {
                expected: "Tensor and Scalar, or Tensor and Tensor".to_string(),
                actual: format!("{:?} and {:?}", args[0], args[1]),
            })
        }
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

/// Sigmoid activation function for neural networks.
///
/// This function applies the sigmoid activation function element-wise to tensor inputs.
/// The sigmoid function is defined as: σ(x) = 1 / (1 + e^(-x))
/// It maps any real value to a value between 0 and 1, making it useful for binary
/// classification and as an activation function in neural networks.
///
/// # Numerical Stability
/// 
/// The implementation uses numerically stable computation to avoid overflow:
/// - For x > 0: σ(x) = 1 / (1 + e^(-x))
/// - For x ≤ 0: σ(x) = e^x / (1 + e^x)
/// This prevents overflow when computing e^x for large positive or negative values.
///
/// # Arguments
/// * `args[0]` - Input tensor (any shape)
///
/// # Returns
/// * `Ok(Value::Tensor)` - Tensor with sigmoid applied element-wise
/// * `Err(VmError)` - If argument is not a tensor or wrong number of arguments
///
/// # Examples
/// ```wolfram
/// (* Basic sigmoid operations *)
/// Sigmoid[{0}]                              (* → {0.5} *)
/// Sigmoid[{-1, 0, 1}]                       (* → {0.268941, 0.5, 0.731059} *)
///
/// (* 2D tensor activation *)
/// matrix = Array[{{-2, -1}, {1, 2}}]
/// Sigmoid[matrix]                           (* → {{0.119203, 0.268941}, {0.731059, 0.880797}} *)
///
/// (* Neural network layer activation *)
/// linear_output = Dot[weights, input]       (* Linear layer output *)
/// activated = Sigmoid[linear_output]        (* Apply sigmoid activation *)
///
/// (* Large values (numerical stability) *)
/// Sigmoid[{-100, 100}]                      (* → {0, 1} safely computed *)
/// ```
///
/// # Neural Network Applications
/// Essential for:
/// - Binary classification output layers
/// - Gate mechanisms in LSTM/GRU networks
/// - Traditional feedforward network activations
/// - Probability output interpretation
pub fn sigmoid(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract tensor from arguments
    let tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => {
            return Err(VmError::TypeError {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Apply sigmoid function element-wise with numerical stability
    let result_data: Vec<f64> = tensor
        .iter()
        .map(|&x| {
            // Numerically stable sigmoid computation
            if x > 0.0 {
                // For positive x: σ(x) = 1 / (1 + e^(-x))
                let exp_neg_x = (-x).exp();
                1.0 / (1.0 + exp_neg_x)
            } else {
                // For negative x: σ(x) = e^x / (1 + e^x)
                let exp_x = x.exp();
                exp_x / (1.0 + exp_x)
            }
        })
        .collect();

    // Create result tensor with same shape as input
    let result_tensor = ArrayD::from_shape_vec(tensor.raw_dim(), result_data)
        .map_err(|e| VmError::TypeError {
            expected: "valid tensor shape".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(result_tensor))
}

/// Hyperbolic tangent activation function for neural networks.
///
/// This function applies the hyperbolic tangent activation function element-wise to tensor inputs.
/// The tanh function is defined as: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
/// It maps any real value to a value between -1 and 1, making it useful as an activation
/// function in neural networks where zero-centered outputs are beneficial.
///
/// # Properties
/// 
/// - **Range**: (-1, 1) - outputs are bounded between -1 and 1
/// - **Zero-centered**: tanh(0) = 0, making it zero-centered unlike sigmoid
/// - **Antisymmetric**: tanh(-x) = -tanh(x)
/// - **Smooth**: Differentiable everywhere with well-behaved gradients
/// - **Saturating**: Large positive/negative inputs saturate to ±1
///
/// # Arguments
/// * `args[0]` - Input tensor (any shape)
///
/// # Returns
/// * `Ok(Value::Tensor)` - Tensor with tanh applied element-wise
/// * `Err(VmError)` - If argument is not a tensor or wrong number of arguments
///
/// # Examples
/// ```wolfram
/// (* Basic tanh operations *)
/// Tanh[{0}]                                 (* → {0} *)
/// Tanh[{-1, 0, 1}]                          (* → {-0.761594, 0, 0.761594} *)
///
/// (* 2D tensor activation *)
/// matrix = Array[{{-2, -1}, {1, 2}}]
/// Tanh[matrix]                              (* → {{-0.964028, -0.761594}, {0.761594, 0.964028}} *)
///
/// (* Neural network hidden layer activation *)
/// hidden_output = Dot[weights, input]       (* Linear layer output *)
/// activated = Tanh[hidden_output]           (* Apply tanh activation *)
///
/// (* Symmetric property *)
/// Tanh[{-2, 2}]                            (* → {-0.964028, 0.964028} *)
/// ```
///
/// # Neural Network Applications
/// Preferred over sigmoid for:
/// - Hidden layer activations (zero-centered gradients)
/// - RNN/LSTM gate functions
/// - Autoencoder bottlenecks
/// - When gradient flow is important
pub fn tanh(args: &[Value]) -> VmResult<Value> {
    // Validate argument count
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract tensor from arguments
    let tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => {
            return Err(VmError::TypeError {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Apply tanh function element-wise using standard library
    let result_data: Vec<f64> = tensor
        .iter()
        .map(|&x| x.tanh()) // Rust's standard library tanh is numerically stable
        .collect();

    // Create result tensor with same shape as input
    let result_tensor = ArrayD::from_shape_vec(tensor.raw_dim(), result_data)
        .map_err(|e| VmError::TypeError {
            expected: "valid tensor shape".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(result_tensor))
}

/// Softmax activation function for neural networks.
///
/// This function applies the softmax activation function to tensor inputs, converting
/// logits into a probability distribution. The softmax function is defined as:
/// softmax(x_i) = exp(x_i) / Σ(exp(x_j)) for all j in the same dimension.
///
/// # Numerical Stability
/// 
/// The implementation uses the numerically stable "max trick":
/// softmax(x_i) = exp(x_i - max(x)) / Σ(exp(x_j - max(x)))
/// This prevents overflow when computing exp(x) for large values.
///
/// # Dimension Behavior
/// 
/// - **1 argument**: Applies softmax along the last dimension (default)
/// - **2 arguments**: Second argument specifies the dimension to normalize
/// - For 1D tensors: Always normalizes the entire vector
/// - For 2D+ tensors: Normalizes along the specified dimension
///
/// # Arguments
/// * `args[0]` - Input tensor (any shape)
/// * `args[1]` - Optional: dimension to apply softmax (integer)
///
/// # Returns
/// * `Ok(Value::Tensor)` - Tensor with softmax applied along specified dimension
/// * `Err(VmError)` - If arguments are invalid or incompatible
///
/// # Examples
/// ```wolfram
/// (* Basic 1D softmax *)
/// Softmax[{1, 2, 3}]                        (* → probability distribution summing to 1 *)
///
/// (* 2D softmax (default: last dimension) *)
/// matrix = Array[{{1, 2}, {3, 4}}]
/// Softmax[matrix]                           (* Each row sums to 1 *)
///
/// (* 2D softmax along specific dimension *)
/// Softmax[matrix, 0]                       (* Each column sums to 1 *)
/// Softmax[matrix, 1]                       (* Each row sums to 1 - same as default *)
///
/// (* Neural network classification *)
/// logits = Array[{2.1, 1.3, 0.2, 3.5}]    (* Raw network output *)
/// probabilities = Softmax[logits]           (* Convert to class probabilities *)
///
/// (* Batch processing *)
/// batch_logits = Array[{{1,2,3},{4,5,6}}]  (* Batch of 2 samples, 3 classes each *)
/// Softmax[batch_logits]                     (* Each sample gets probability distribution *)
/// ```
///
/// # Neural Network Applications
/// Essential for:
/// - Multi-class classification output layers
/// - Attention mechanisms (scaled dot-product attention)
/// - Policy networks in reinforcement learning
/// - Any scenario requiring probability distributions
pub fn softmax(args: &[Value]) -> VmResult<Value> {
    // Validate argument count (1 or 2 arguments)
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1 or 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract tensor from first argument
    let tensor = match &args[0] {
        Value::Tensor(t) => t,
        _ => {
            return Err(VmError::TypeError {
                expected: "Tensor".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Determine the dimension to apply softmax
    let dim = if args.len() == 2 {
        // Explicit dimension provided
        match &args[1] {
            Value::Integer(d) => {
                let dim_val = *d as usize;
                if *d < 0 || dim_val >= tensor.ndim() {
                    return Err(VmError::TypeError {
                        expected: format!("dimension in range [0, {})", tensor.ndim()),
                        actual: format!("dimension {}", d),
                    });
                }
                dim_val
            }
            _ => {
                return Err(VmError::TypeError {
                    expected: "Integer dimension".to_string(),
                    actual: format!("{:?}", args[1]),
                });
            }
        }
    } else {
        // Default: last dimension
        tensor.ndim() - 1
    };

    // Apply softmax along the specified dimension
    let result_tensor = apply_softmax_along_dim(tensor, dim)?;
    
    Ok(Value::Tensor(result_tensor))
}

/// Helper function to apply softmax along a specific dimension
fn apply_softmax_along_dim(tensor: &ArrayD<f64>, dim: usize) -> VmResult<ArrayD<f64>> {
    let shape = tensor.shape();
    let mut result = tensor.clone();
    
    // For 1D tensors, apply softmax to the entire vector
    if tensor.ndim() == 1 {
        let data: Vec<f64> = tensor.iter().cloned().collect();
        let softmax_data = compute_softmax_1d(&data);
        return ArrayD::from_shape_vec(tensor.raw_dim(), softmax_data)
            .map_err(|e| VmError::TypeError {
                expected: "valid tensor shape".to_string(),
                actual: format!("ndarray error: {}", e),
            });
    }
    
    // For multi-dimensional tensors, iterate over all slices perpendicular to the softmax dimension
    let mut indices = vec![0; tensor.ndim()];
    let total_slices: usize = shape.iter().enumerate()
        .filter(|(i, _)| *i != dim)
        .map(|(_, &size)| size)
        .product();
    
    for slice_idx in 0..total_slices {
        // Convert linear slice index to multi-dimensional indices
        let mut temp_idx = slice_idx;
        for i in 0..tensor.ndim() {
            if i == dim {
                continue;
            }
            indices[i] = temp_idx % shape[i];
            temp_idx /= shape[i];
        }
        
        // Extract values along the softmax dimension
        let mut values = Vec::with_capacity(shape[dim]);
        for k in 0..shape[dim] {
            indices[dim] = k;
            values.push(tensor[indices.as_slice()]);
        }
        
        // Compute softmax for this slice
        let softmax_values = compute_softmax_1d(&values);
        
        // Store results back
        for k in 0..shape[dim] {
            indices[dim] = k;
            result[indices.as_slice()] = softmax_values[k];
        }
    }
    
    Ok(result)
}

/// Compute softmax for a 1D vector with numerical stability
fn compute_softmax_1d(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    
    // Numerical stability: subtract max value to prevent overflow
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute exp(x - max) for all values
    let exp_values: Vec<f64> = values
        .iter()
        .map(|&x| (x - max_val).exp())
        .collect();
    
    // Compute sum of exponentials
    let sum_exp: f64 = exp_values.iter().sum();
    
    // Normalize to get probabilities
    exp_values
        .iter()
        .map(|&exp_val| exp_val / sum_exp)
        .collect()
}

/// Random normal weight initialization function for neural networks.
///
/// This function creates tensors filled with random values drawn from a normal (Gaussian)
/// distribution. It's essential for neural network weight initialization to break symmetry
/// and enable effective learning.
///
/// # Arguments
/// * `args[0]` - Shape list: {rows, cols} or {dim1, dim2, ...}
/// * `args[1]` - Optional: mean value (default: 0.0)
/// * `args[2]` - Optional: standard deviation (default: 1.0)
///
/// # Returns
/// * `Ok(Value::Tensor)` - Tensor filled with random normal values
/// * `Err(VmError)` - If arguments are invalid or incompatible
///
/// # Examples
/// ```wolfram
/// (* Basic weight initialization *)
/// weights = RandomNormal[{784, 128}]        (* Standard normal (mean=0, std=1) *)
/// bias = RandomNormal[{128}]                (* 1D bias vector *)
///
/// (* Custom distribution *)
/// weights = RandomNormal[{10, 5}, 0.0, 0.1] (* Small weights for stable training *)
/// 
/// (* Neural network layer initialization *)
/// input_weights = RandomNormal[{784, 256}]   (* Input layer weights *)
/// hidden_weights = RandomNormal[{256, 128}]  (* Hidden layer weights *)
/// output_weights = RandomNormal[{128, 10}]   (* Output layer weights *)
/// ```
///
/// # Weight Initialization Guidelines
/// - **Default (std=1.0)**: Good starting point for most networks
/// - **Small std (0.01-0.1)**: Prevents saturation in deep networks
/// - **Xavier/Glorot scaling**: Use `std = sqrt(2/(fan_in + fan_out))`
/// - **He scaling**: Use `std = sqrt(2/fan_in)` for ReLU networks
pub fn random_normal(args: &[Value]) -> VmResult<Value> {
    // Validate argument count (1, 2, or 3 arguments)
    if args.is_empty() || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1, 2, or 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Parse shape from first argument
    let shape_dims = match &args[0] {
        Value::List(dims) => {
            if dims.is_empty() {
                return Err(VmError::TypeError {
                    expected: "non-empty shape list".to_string(),
                    actual: "empty list".to_string(),
                });
            }
            
            let mut parsed_dims = Vec::new();
            for dim in dims {
                match dim {
                    Value::Integer(d) => {
                        if *d <= 0 {
                            return Err(VmError::TypeError {
                                expected: "positive dimension".to_string(),
                                actual: format!("dimension {}", d),
                            });
                        }
                        parsed_dims.push(*d as usize);
                    }
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "integer dimension".to_string(),
                            actual: format!("{:?}", dim),
                        });
                    }
                }
            }
            parsed_dims
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "shape list".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Parse mean (default: 0.0)
    let mean = if args.len() >= 2 {
        match &args[1] {
            Value::Real(m) => *m,
            Value::Integer(m) => *m as f64,
            _ => {
                return Err(VmError::TypeError {
                    expected: "numeric mean".to_string(),
                    actual: format!("{:?}", args[1]),
                });
            }
        }
    } else {
        0.0
    };

    // Parse standard deviation (default: 1.0)
    let std = if args.len() >= 3 {
        match &args[2] {
            Value::Real(s) => {
                if *s <= 0.0 {
                    return Err(VmError::TypeError {
                        expected: "positive standard deviation".to_string(),
                        actual: format!("{}", s),
                    });
                }
                *s
            }
            Value::Integer(s) => {
                if *s <= 0 {
                    return Err(VmError::TypeError {
                        expected: "positive standard deviation".to_string(),
                        actual: format!("{}", s),
                    });
                }
                *s as f64
            }
            _ => {
                return Err(VmError::TypeError {
                    expected: "numeric standard deviation".to_string(),
                    actual: format!("{:?}", args[2]),
                });
            }
        }
    } else {
        1.0
    };

    // Calculate total number of elements
    let total_elements: usize = shape_dims.iter().product();
    
    // Generate random normal values using Box-Muller transform
    let mut random_values = Vec::with_capacity(total_elements);
    let mut spare_value: Option<f64> = None;
    
    for _ in 0..total_elements {
        let normal_value = if let Some(spare) = spare_value.take() {
            spare
        } else {
            // Box-Muller transform to generate normal distribution from uniform
            let u1: f64 = rand::random::<f64>();
            let u2: f64 = rand::random::<f64>();
            
            // Ensure u1 is not zero to avoid log(0)
            let u1 = if u1 == 0.0 { f64::EPSILON } else { u1 };
            
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            
            spare_value = Some(z1);
            z0
        };
        
        // Scale and shift: N(mean, std²) = mean + std * N(0, 1)
        random_values.push(mean + std * normal_value);
    }

    // Create tensor with the specified shape
    let result_tensor = ArrayD::from_shape_vec(IxDyn(&shape_dims), random_values)
        .map_err(|e| VmError::TypeError {
            expected: "valid tensor shape".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(result_tensor))
}

/// Xavier/Glorot weight initialization for neural networks.
///
/// This function creates tensors filled with random values drawn from a uniform distribution
/// scaled according to the Xavier/Glorot initialization scheme. It's specifically designed
/// for layers using Sigmoid or Tanh activation functions to maintain activation variance.
///
/// # Xavier Formula
/// Values are uniformly distributed in the range:
/// `[-bound, bound]` where `bound = sqrt(6 / (fan_in + fan_out))`
///
/// This maintains activation variance across layers and prevents vanishing/exploding gradients.
///
/// # Arguments
/// * `args[0]` - Shape list: {fan_in, fan_out} (exactly 2 dimensions required)
///
/// # Returns
/// * `Ok(Value::Tensor)` - 2D tensor with Xavier-initialized weights
/// * `Err(VmError)` - If arguments are invalid or shape is not 2D
///
/// # Examples
/// ```wolfram
/// (* Basic layer initialization *)
/// weights = Xavier[{784, 128}]              (* Input layer: 784 → 128 *)
/// hidden = Xavier[{128, 64}]                (* Hidden layer: 128 → 64 *)
/// output = Xavier[{64, 10}]                 (* Output layer: 64 → 10 *)
///
/// (* Neural network with Sigmoid/Tanh activations *)
/// layer1_weights = Xavier[{784, 256}]
/// layer1_output = Sigmoid[Dot[layer1_weights, input]]
/// 
/// layer2_weights = Xavier[{256, 128}] 
/// layer2_output = Tanh[Dot[layer2_weights, layer1_output]]
/// ```
///
/// # When to Use Xavier
/// - **Sigmoid/Tanh networks**: Optimal for these activation functions
/// - **Feedforward networks**: Classic choice for fully connected layers
/// - **Symmetric activations**: Functions centered around zero
/// - **Gradient flow**: Maintains stable gradients during backpropagation
///
/// For ReLU networks, consider using He initialization instead.
pub fn xavier(args: &[Value]) -> VmResult<Value> {
    // Validate argument count (exactly 1 argument: the shape)
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Parse shape from first argument - must be exactly 2D
    let (fan_in, fan_out) = match &args[0] {
        Value::List(dims) => {
            if dims.len() != 2 {
                return Err(VmError::TypeError {
                    expected: "exactly 2 dimensions for Xavier initialization".to_string(),
                    actual: format!("{} dimensions", dims.len()),
                });
            }
            
            let fan_in = match &dims[0] {
                Value::Integer(d) => {
                    if *d <= 0 {
                        return Err(VmError::TypeError {
                            expected: "positive fan_in dimension".to_string(),
                            actual: format!("fan_in {}", d),
                        });
                    }
                    *d as usize
                }
                _ => {
                    return Err(VmError::TypeError {
                        expected: "integer fan_in dimension".to_string(),
                        actual: format!("{:?}", dims[0]),
                    });
                }
            };
            
            let fan_out = match &dims[1] {
                Value::Integer(d) => {
                    if *d <= 0 {
                        return Err(VmError::TypeError {
                            expected: "positive fan_out dimension".to_string(),
                            actual: format!("fan_out {}", d),
                        });
                    }
                    *d as usize
                }
                _ => {
                    return Err(VmError::TypeError {
                        expected: "integer fan_out dimension".to_string(),
                        actual: format!("{:?}", dims[1]),
                    });
                }
            };
            
            (fan_in, fan_out)
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "shape list".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };

    // Calculate Xavier bound: sqrt(6 / (fan_in + fan_out))
    let xavier_bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
    
    // Generate random uniform values in [-xavier_bound, xavier_bound]
    let total_elements = fan_in * fan_out;
    let mut random_values = Vec::with_capacity(total_elements);
    
    for _ in 0..total_elements {
        // Generate uniform random value in [0, 1] then scale to [-bound, bound]
        let uniform_val: f64 = rand::random();
        let scaled_val = xavier_bound * (2.0 * uniform_val - 1.0); // Maps [0,1] to [-bound, bound]
        random_values.push(scaled_val);
    }

    // Create tensor with the specified shape
    let result_tensor = ArrayD::from_shape_vec(IxDyn(&[fan_in, fan_out]), random_values)
        .map_err(|e| VmError::TypeError {
            expected: "valid tensor shape".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(result_tensor))
}

/// He initialization for ReLU networks: He[{fan_in, fan_out}]
/// 
/// He initialization is specifically designed for ReLU activation networks.
/// It samples from a uniform distribution with bounds determined by:
/// bound = sqrt(2 / fan_in)
/// 
/// This initialization ensures proper gradient flow in deep ReLU networks
/// by maintaining unit variance in the forward pass.
/// 
/// # Arguments
/// * `args[0]` - List containing two integers: {fan_in, fan_out}
/// 
/// # Returns
/// * `Value::Tensor` - Tensor of shape [fan_in, fan_out] with He-initialized weights
/// 
/// # Errors
/// * Wrong number of arguments
/// * Invalid shape format (must be list of exactly 2 integers)
/// * Non-positive dimensions
/// 
/// # Examples
/// ```wolfram
/// (* Initialize weights for 784->128 ReLU layer *)
/// weights = He[{784, 128}]
/// 
/// (* Initialize smaller layer *)
/// weights = He[{64, 32}]
/// ```
pub fn he(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract fan_in and fan_out from the shape list
    let (fan_in, fan_out) = match &args[0] {
        Value::List(shape_list) => {
            if shape_list.len() != 2 {
                return Err(VmError::TypeError {
                    expected: "list of two integers {fan_in, fan_out}".to_string(),
                    actual: format!("list of {} elements", shape_list.len()),
                });
            }
            
            let fan_in = match &shape_list[0] {
                Value::Integer(n) => {
                    if *n <= 0 {
                        return Err(VmError::TypeError {
                            expected: "positive integer for fan_in".to_string(),
                            actual: format!("{}", n),
                        });
                    }
                    *n as usize
                }
                _ => return Err(VmError::TypeError {
                    expected: "integer for fan_in".to_string(),
                    actual: format!("{:?}", shape_list[0]),
                })
            };
            
            let fan_out = match &shape_list[1] {
                Value::Integer(n) => {
                    if *n <= 0 {
                        return Err(VmError::TypeError {
                            expected: "positive integer for fan_out".to_string(),
                            actual: format!("{}", n),
                        });
                    }
                    *n as usize
                }
                _ => return Err(VmError::TypeError {
                    expected: "integer for fan_out".to_string(),
                    actual: format!("{:?}", shape_list[1]),
                })
            };
            
            (fan_in, fan_out)
        }
        _ => return Err(VmError::TypeError {
            expected: "list of two integers {fan_in, fan_out}".to_string(),
            actual: format!("{:?}", args[0]),
        })
    };

    // He initialization bound: sqrt(2 / fan_in)
    let he_bound = (2.0_f64 / fan_in as f64).sqrt();
    
    // Create tensor using ArrayD like other functions in this file
    let total_elements = fan_in * fan_out;
    let mut random_values = Vec::with_capacity(total_elements);
    
    for _ in 0..total_elements {
        // Generate random value in [0, 1] then scale to [-he_bound, he_bound]
        let random_val: f64 = rand::random();
        random_values.push((random_val - 0.5) * 2.0 * he_bound);
    }

    // Create tensor with the specified shape
    let result_tensor = ArrayD::from_shape_vec(IxDyn(&[fan_in, fan_out]), random_values)
        .map_err(|e| VmError::TypeError {
            expected: "valid tensor shape".to_string(),
            actual: format!("ndarray error: {}", e),
        })?;

    Ok(Value::Tensor(result_tensor))
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

    // ===== LINEAR ALGEBRA OPERATIONS TESTS =====
    // TDD Tests for Dot[] function (Phase 3A)

    #[test]
    fn test_dot_product_vector_vector() {
        // RED: Test vector dot product - Dot[{1,2,3}, {4,5,6}] -> 32
        let vec_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        let vec_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![4.0, 5.0, 6.0]
        ).unwrap());
        
        let result = dot(&[vec_a, vec_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Real(value) => {
                assert_eq!(value, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            }
            _ => panic!("Expected real value from vector dot product"),
        }
    }

    #[test] 
    fn test_dot_product_vector_vector_simple() {
        // RED: Test simple vector dot product - Dot[{1,0}, {0,1}] -> 0
        let vec_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![1.0, 0.0]
        ).unwrap());
        let vec_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![0.0, 1.0]
        ).unwrap());
        
        let result = dot(&[vec_a, vec_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Real(value) => {
                assert_eq!(value, 0.0); // 1*0 + 0*1 = 0
            }
            _ => panic!("Expected real value from vector dot product"),
        }
    }

    #[test]
    fn test_dot_product_vector_vector_ones() {
        // RED: Test dot product with ones - Dot[{1,1,1}, {1,1,1}] -> 3
        let vec_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 1.0, 1.0]
        ).unwrap());
        let vec_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 1.0, 1.0]
        ).unwrap());
        
        let result = dot(&[vec_a, vec_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Real(value) => {
                assert_eq!(value, 3.0); // 1*1 + 1*1 + 1*1 = 3
            }
            _ => panic!("Expected real value from vector dot product"),
        }
    }

    #[test]
    fn test_dot_product_vector_incompatible_shapes() {
        // RED: Test incompatible vector shapes should error
        let vec_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        let vec_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![4.0, 5.0]
        ).unwrap());
        
        let result = dot(&[vec_a, vec_b]);
        assert!(result.is_err()); // Should fail due to incompatible shapes
    }

    #[test]
    fn test_dot_product_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let vec_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        // Too few arguments
        let result = dot(&[vec_a.clone()]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = dot(&[vec_a.clone(), vec_a.clone(), vec_a]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dot_product_non_tensor_args() {
        // RED: Test non-tensor arguments should error
        let result = dot(&[Value::Integer(1), Value::Integer(2)]);
        assert!(result.is_err());
        
        let result = dot(&[
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)])
        ]);
        assert!(result.is_err());
    }

    // ===== MATRIX-VECTOR MULTIPLICATION TESTS =====
    // TDD Tests for matrix-vector dot products

    #[test]
    fn test_dot_product_matrix_vector_2x3() {
        // RED: Test matrix-vector multiplication - Dot[{{1,2,3},{4,5,6}}, {1,0,1}] -> {4,10}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 0.0, 1.0]
        ).unwrap());
        
        let result = dot(&[matrix, vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                assert_eq!(result_tensor[[0]], 4.0);  // 1*1 + 2*0 + 3*1 = 4
                assert_eq!(result_tensor[[1]], 10.0); // 4*1 + 5*0 + 6*1 = 10
            }
            _ => panic!("Expected tensor value from matrix-vector multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_vector_simple() {
        // RED: Test simple matrix-vector multiplication - Dot[{{1,2},{3,4}}, {1,1}] -> {3,7}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![1.0, 1.0]
        ).unwrap());
        
        let result = dot(&[matrix, vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                assert_eq!(result_tensor[[0]], 3.0); // 1*1 + 2*1 = 3
                assert_eq!(result_tensor[[1]], 7.0); // 3*1 + 4*1 = 7
            }
            _ => panic!("Expected tensor value from matrix-vector multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_vector_identity() {
        // RED: Test identity matrix multiplication - Dot[{{1,0},{0,1}}, {3,4}] -> {3,4}
        let identity = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 0.0, 0.0, 1.0]
        ).unwrap());
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![3.0, 4.0]
        ).unwrap());
        
        let result = dot(&[identity, vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                assert_eq!(result_tensor[[0]], 3.0); // 1*3 + 0*4 = 3
                assert_eq!(result_tensor[[1]], 4.0); // 0*3 + 1*4 = 4
            }
            _ => panic!("Expected tensor value from matrix-vector multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_vector_incompatible_shapes() {
        // RED: Test incompatible matrix-vector shapes should error
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), // Wrong size - should be 3 to match matrix columns
            vec![1.0, 2.0]
        ).unwrap());
        
        let result = dot(&[matrix, vector]);
        assert!(result.is_err()); // Should fail due to incompatible shapes
    }

    #[test]
    fn test_dot_product_vector_matrix() {
        // RED: Test vector-matrix multiplication should also work - Dot[{1,2}, {{3,4},{5,6}}] -> {13,16}
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![1.0, 2.0]
        ).unwrap());
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        
        let result = dot(&[vector, matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                assert_eq!(result_tensor[[0]], 13.0); // 1*3 + 2*5 = 13
                assert_eq!(result_tensor[[1]], 16.0); // 1*4 + 2*6 = 16
            }
            _ => panic!("Expected tensor value from vector-matrix multiplication"),
        }
    }

    // ===== MATRIX-MATRIX MULTIPLICATION TESTS =====
    // TDD Tests for matrix-matrix dot products

    #[test]
    fn test_dot_product_matrix_matrix_2x2() {
        // RED: Test 2x2 matrix multiplication - Dot[{{1,2},{3,4}}, {{5,6},{7,8}}] -> {{19,22},{43,50}}
        let matrix_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        let matrix_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![5.0, 6.0, 7.0, 8.0]
        ).unwrap());
        
        let result = dot(&[matrix_a, matrix_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 19.0); // 1*5 + 2*7 = 19
                assert_eq!(result_tensor[[0, 1]], 22.0); // 1*6 + 2*8 = 22
                assert_eq!(result_tensor[[1, 0]], 43.0); // 3*5 + 4*7 = 43
                assert_eq!(result_tensor[[1, 1]], 50.0); // 3*6 + 4*8 = 50
            }
            _ => panic!("Expected tensor value from matrix-matrix multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_matrix_identity() {
        // RED: Test identity matrix multiplication - Dot[{{1,2},{3,4}}, {{1,0},{0,1}}] -> {{1,2},{3,4}}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        let identity = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 0.0, 0.0, 1.0]
        ).unwrap());
        
        let result = dot(&[matrix, identity]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 1.0); // 1*1 + 2*0 = 1
                assert_eq!(result_tensor[[0, 1]], 2.0); // 1*0 + 2*1 = 2
                assert_eq!(result_tensor[[1, 0]], 3.0); // 3*1 + 4*0 = 3
                assert_eq!(result_tensor[[1, 1]], 4.0); // 3*0 + 4*1 = 4
            }
            _ => panic!("Expected tensor value from matrix-matrix multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_matrix_rectangular() {
        // RED: Test rectangular matrix multiplication - Dot[{{1,2,3},{4,5,6}}, {{1,2},{3,4},{5,6}}] -> {{22,28},{49,64}}
        let matrix_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        let matrix_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3, 2]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        
        let result = dot(&[matrix_a, matrix_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 22.0); // 1*1 + 2*3 + 3*5 = 22
                assert_eq!(result_tensor[[0, 1]], 28.0); // 1*2 + 2*4 + 3*6 = 28
                assert_eq!(result_tensor[[1, 0]], 49.0); // 4*1 + 5*3 + 6*5 = 49
                assert_eq!(result_tensor[[1, 1]], 64.0); // 4*2 + 5*4 + 6*6 = 64
            }
            _ => panic!("Expected tensor value from matrix-matrix multiplication"),
        }
    }

    #[test]
    fn test_dot_product_matrix_matrix_incompatible_shapes() {
        // RED: Test incompatible matrix shapes should error
        let matrix_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        let matrix_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), // Wrong shape - should be [3, x] to match matrix_a columns
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        
        let result = dot(&[matrix_a, matrix_b]);
        assert!(result.is_err()); // Should fail due to incompatible shapes
    }

    // ===== TRANSPOSE OPERATION TESTS =====
    // TDD Tests for Transpose[] function

    #[test]
    fn test_transpose_2x3_matrix() {
        // RED: Test 2x3 matrix transpose - Transpose[{{1,2,3},{4,5,6}}] -> {{1,4},{2,5},{3,6}}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        
        let result = transpose(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3, 2]);
                assert_eq!(result_tensor[[0, 0]], 1.0); // Original [0,0]
                assert_eq!(result_tensor[[0, 1]], 4.0); // Original [1,0]
                assert_eq!(result_tensor[[1, 0]], 2.0); // Original [0,1]
                assert_eq!(result_tensor[[1, 1]], 5.0); // Original [1,1]
                assert_eq!(result_tensor[[2, 0]], 3.0); // Original [0,2]
                assert_eq!(result_tensor[[2, 1]], 6.0); // Original [1,2]
            }
            _ => panic!("Expected tensor value from matrix transpose"),
        }
    }

    #[test]
    fn test_transpose_2x2_matrix() {
        // RED: Test 2x2 matrix transpose - Transpose[{{1,2},{3,4}}] -> {{1,3},{2,4}}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        
        let result = transpose(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 1.0); // Original [0,0]
                assert_eq!(result_tensor[[0, 1]], 3.0); // Original [1,0]
                assert_eq!(result_tensor[[1, 0]], 2.0); // Original [0,1]
                assert_eq!(result_tensor[[1, 1]], 4.0); // Original [1,1]
            }
            _ => panic!("Expected tensor value from matrix transpose"),
        }
    }

    #[test]
    fn test_transpose_identity_matrix() {
        // RED: Test identity matrix transpose (should be unchanged)
        let identity = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3, 3]), 
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ).unwrap());
        
        let result = transpose(&[identity]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3, 3]);
                // Identity matrix should be unchanged by transpose
                assert_eq!(result_tensor[[0, 0]], 1.0);
                assert_eq!(result_tensor[[0, 1]], 0.0);
                assert_eq!(result_tensor[[0, 2]], 0.0);
                assert_eq!(result_tensor[[1, 0]], 0.0);
                assert_eq!(result_tensor[[1, 1]], 1.0);
                assert_eq!(result_tensor[[1, 2]], 0.0);
                assert_eq!(result_tensor[[2, 0]], 0.0);
                assert_eq!(result_tensor[[2, 1]], 0.0);
                assert_eq!(result_tensor[[2, 2]], 1.0);
            }
            _ => panic!("Expected tensor value from matrix transpose"),
        }
    }

    #[test]
    fn test_transpose_single_element() {
        // RED: Test 1x1 matrix transpose
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[1, 1]), 
            vec![42.0]
        ).unwrap());
        
        let result = transpose(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[1, 1]);
                assert_eq!(result_tensor[[0, 0]], 42.0);
            }
            _ => panic!("Expected tensor value from matrix transpose"),
        }
    }

    #[test]
    fn test_transpose_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        
        // Too few arguments
        let result = transpose(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = transpose(&[matrix.clone(), matrix]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_non_tensor_args() {
        // RED: Test non-tensor arguments should error
        let result = transpose(&[Value::Integer(42)]);
        assert!(result.is_err());
        
        let result = transpose(&[Value::List(vec![Value::Integer(1), Value::Integer(2)])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_transpose_unsupported_3d_tensor() {
        // RED: Test 3D+ tensor should error (only support 1D and 2D)
        let tensor_3d = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ).unwrap());
        
        let result = transpose(&[tensor_3d]);
        assert!(result.is_err()); // Should fail for 3D tensor
    }

    // ===== VECTOR TRANSPOSE TESTS =====
    // TDD Tests for vector transpose behavior

    #[test]
    fn test_transpose_row_vector() {
        // RED: Test row vector transpose - Transpose[{{1,2,3}}] -> {{1},{2},{3}}
        let row_vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[1, 3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        let result = transpose(&[row_vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3, 1]);
                assert_eq!(result_tensor[[0, 0]], 1.0);
                assert_eq!(result_tensor[[1, 0]], 2.0);
                assert_eq!(result_tensor[[2, 0]], 3.0);
            }
            _ => panic!("Expected tensor value from row vector transpose"),
        }
    }

    #[test]
    fn test_transpose_column_vector() {
        // RED: Test column vector transpose - Transpose[{{1},{2},{3}}] -> {{1,2,3}}
        let column_vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3, 1]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        let result = transpose(&[column_vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[1, 3]);
                assert_eq!(result_tensor[[0, 0]], 1.0);
                assert_eq!(result_tensor[[0, 1]], 2.0);
                assert_eq!(result_tensor[[0, 2]], 3.0);
            }
            _ => panic!("Expected tensor value from column vector transpose"),
        }
    }

    #[test]
    fn test_transpose_vector_conversion() {
        // RED: Test 1D vector should convert to column vector when transposed
        // This is needed for neural network operations
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        let result = transpose(&[vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3, 1]); // Convert to column vector
                assert_eq!(result_tensor[[0, 0]], 1.0);
                assert_eq!(result_tensor[[1, 0]], 2.0);
                assert_eq!(result_tensor[[2, 0]], 3.0);
            }
            _ => panic!("Expected tensor value from vector transpose"),
        }
    }

    #[test]
    fn test_transpose_double_transpose() {
        // RED: Test double transpose returns original matrix
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        
        let first_transpose = transpose(&[matrix.clone()]).unwrap();
        let second_transpose = transpose(&[first_transpose]).unwrap();
        
        match (matrix, second_transpose) {
            (Value::Tensor(original), Value::Tensor(double_transposed)) => {
                assert_eq!(original.shape(), double_transposed.shape());
                for i in 0..original.len() {
                    assert_eq!(original.as_slice().unwrap()[i], double_transposed.as_slice().unwrap()[i]);
                }
            }
            _ => panic!("Expected tensor values"),
        }
    }

    // ===== MAXIMUM FUNCTION TESTS =====
    // TDD Tests for Maximum[] function (needed for ReLU activation)

    #[test]
    fn test_maximum_tensor_scalar_relu() {
        // RED: Test ReLU operation - Maximum[{-2,-1,0,1,2}, 0] -> {0,0,0,1,2}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[5]), 
            vec![-2.0, -1.0, 0.0, 1.0, 2.0]
        ).unwrap());
        let scalar = Value::Real(0.0);
        
        let result = maximum(&[tensor, scalar]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[5]);
                assert_eq!(result_tensor[[0]], 0.0); // max(-2, 0) = 0
                assert_eq!(result_tensor[[1]], 0.0); // max(-1, 0) = 0
                assert_eq!(result_tensor[[2]], 0.0); // max(0, 0) = 0
                assert_eq!(result_tensor[[3]], 1.0); // max(1, 0) = 1
                assert_eq!(result_tensor[[4]], 2.0); // max(2, 0) = 2
            }
            _ => panic!("Expected tensor value from maximum operation"),
        }
    }

    #[test]
    fn test_maximum_scalar_tensor() {
        // RED: Test scalar-tensor maximum (commutative) - Maximum[0, {-1,0,1}] -> {0,0,1}
        let scalar = Value::Real(0.0);
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![-1.0, 0.0, 1.0]
        ).unwrap());
        
        let result = maximum(&[scalar, tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                assert_eq!(result_tensor[[0]], 0.0); // max(0, -1) = 0
                assert_eq!(result_tensor[[1]], 0.0); // max(0, 0) = 0
                assert_eq!(result_tensor[[2]], 1.0); // max(0, 1) = 1
            }
            _ => panic!("Expected tensor value from maximum operation"),
        }
    }

    #[test]
    fn test_maximum_tensor_tensor() {
        // RED: Test element-wise maximum between tensors - Maximum[{1,2,3}, {2,1,4}] -> {2,2,4}
        let tensor_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        let tensor_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![2.0, 1.0, 4.0]
        ).unwrap());
        
        let result = maximum(&[tensor_a, tensor_b]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                assert_eq!(result_tensor[[0]], 2.0); // max(1, 2) = 2
                assert_eq!(result_tensor[[1]], 2.0); // max(2, 1) = 2
                assert_eq!(result_tensor[[2]], 4.0); // max(3, 4) = 4
            }
            _ => panic!("Expected tensor value from maximum operation"),
        }
    }

    #[test]
    fn test_maximum_2d_tensor_scalar() {
        // RED: Test 2D tensor with scalar - Maximum[{{-1,0},{1,2}}, 0] -> {{0,0},{1,2}}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![-1.0, 0.0, 1.0, 2.0]
        ).unwrap());
        let scalar = Value::Real(0.0);
        
        let result = maximum(&[matrix, scalar]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 0.0); // max(-1, 0) = 0
                assert_eq!(result_tensor[[0, 1]], 0.0); // max(0, 0) = 0
                assert_eq!(result_tensor[[1, 0]], 1.0); // max(1, 0) = 1
                assert_eq!(result_tensor[[1, 1]], 2.0); // max(2, 0) = 2
            }
            _ => panic!("Expected tensor value from maximum operation"),
        }
    }

    #[test]
    fn test_maximum_with_broadcasting() {
        // RED: Test broadcasting - Maximum[{{1,2},{3,4}}, {0,1}] -> {{1,2},{3,4}}
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        let vector = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![0.0, 1.0]
        ).unwrap());
        
        let result = maximum(&[matrix, vector]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                assert_eq!(result_tensor[[0, 0]], 1.0); // max(1, 0) = 1
                assert_eq!(result_tensor[[0, 1]], 2.0); // max(2, 1) = 2
                assert_eq!(result_tensor[[1, 0]], 3.0); // max(3, 0) = 3
                assert_eq!(result_tensor[[1, 1]], 4.0); // max(4, 1) = 4
            }
            _ => panic!("Expected tensor value from maximum operation"),
        }
    }

    #[test]
    fn test_maximum_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        // Too few arguments
        let result = maximum(&[tensor.clone()]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = maximum(&[tensor.clone(), tensor.clone(), tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_maximum_incompatible_types() {
        // RED: Test incompatible types should error
        let result = maximum(&[Value::String("hello".to_string()), Value::Integer(1)]);
        assert!(result.is_err());
        
        let result = maximum(&[
            Value::List(vec![Value::Integer(1)]),
            Value::Integer(2)
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_maximum_incompatible_shapes() {
        // RED: Test incompatible tensor shapes should error
        let tensor_a = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        let tensor_b = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        
        let result = maximum(&[tensor_a, tensor_b]);
        assert!(result.is_err()); // Should fail due to incompatible shapes
    }

    // ===== SIGMOID ACTIVATION FUNCTION TESTS (Phase 3B-1) =====
    // TDD Tests for Sigmoid[] function

    #[test]
    fn test_sigmoid_single_element() {
        // RED: Test sigmoid on single element tensor - Sigmoid[{0}] -> {0.5}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[1]), 
            vec![0.0]
        ).unwrap());
        
        let result = sigmoid(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[1]);
                assert!((result_tensor[[0]] - 0.5).abs() < 1e-10); // sigmoid(0) = 0.5
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    #[test]
    fn test_sigmoid_basic_values() {
        // RED: Test sigmoid on basic values - Sigmoid[{-1, 0, 1}] -> {0.268941, 0.5, 0.731059}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![-1.0, 0.0, 1.0]
        ).unwrap());
        
        let result = sigmoid(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // sigmoid(-1) ≈ 0.268941
                assert!((result_tensor[[0]] - 0.2689414213699951).abs() < 1e-10);
                // sigmoid(0) = 0.5
                assert!((result_tensor[[1]] - 0.5).abs() < 1e-10);
                // sigmoid(1) ≈ 0.731059
                assert!((result_tensor[[2]] - 0.7310585786300049).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    #[test]
    fn test_sigmoid_large_positive_values() {
        // RED: Test sigmoid with large positive values (numerical stability) - Sigmoid[{10, 100}] -> {~1, ~1}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![10.0, 100.0]
        ).unwrap());
        
        let result = sigmoid(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                // sigmoid(10) should be very close to 1
                assert!(result_tensor[[0]] > 0.9999);
                assert!(result_tensor[[0]] <= 1.0);
                // sigmoid(100) should be 1 (due to numerical stability)
                assert!(result_tensor[[1]] > 0.9999);
                assert!(result_tensor[[1]] <= 1.0);
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    #[test]
    fn test_sigmoid_large_negative_values() {
        // RED: Test sigmoid with large negative values (numerical stability) - Sigmoid[{-10, -100}] -> {~0, ~0}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2]), 
            vec![-10.0, -100.0]
        ).unwrap());
        
        let result = sigmoid(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                // sigmoid(-10) should be very close to 0
                assert!(result_tensor[[0]] < 0.0001);
                assert!(result_tensor[[0]] >= 0.0);
                // sigmoid(-100) should be 0 (due to numerical stability)
                assert!(result_tensor[[1]] < 0.0001);
                assert!(result_tensor[[1]] >= 0.0);
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    #[test]
    fn test_sigmoid_2d_tensor() {
        // RED: Test sigmoid on 2D tensor - Sigmoid[{{-1, 0}, {1, 2}}]
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![-1.0, 0.0, 1.0, 2.0]
        ).unwrap());
        
        let result = sigmoid(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                // sigmoid(-1) ≈ 0.268941
                assert!((result_tensor[[0, 0]] - 0.2689414213699951).abs() < 1e-10);
                // sigmoid(0) = 0.5
                assert!((result_tensor[[0, 1]] - 0.5).abs() < 1e-10);
                // sigmoid(1) ≈ 0.731059
                assert!((result_tensor[[1, 0]] - 0.7310585786300049).abs() < 1e-10);
                // sigmoid(2) ≈ 0.880797
                assert!((result_tensor[[1, 1]] - 0.8807970779778823).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    #[test]
    fn test_sigmoid_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        // Too few arguments
        let result = sigmoid(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = sigmoid(&[tensor.clone(), tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid_non_tensor_args() {
        // RED: Test non-tensor arguments should error
        let result = sigmoid(&[Value::Integer(1)]);
        assert!(result.is_err());
        
        let result = sigmoid(&[Value::String("hello".to_string())]);
        assert!(result.is_err());
        
        let result = sigmoid(&[Value::List(vec![Value::Real(1.0)])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sigmoid_neural_network_example() {
        // RED: Test sigmoid in neural network context - typical activation patterns
        let logits = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![-2.0, -0.5, 0.5, 2.0]
        ).unwrap());
        
        let result = sigmoid(&[logits]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[4]);
                // All values should be between 0 and 1
                for i in 0..4 {
                    assert!(result_tensor[[i]] > 0.0);
                    assert!(result_tensor[[i]] < 1.0);
                }
                // Values should be monotonic increasing
                assert!(result_tensor[[0]] < result_tensor[[1]]);
                assert!(result_tensor[[1]] < result_tensor[[2]]);
                assert!(result_tensor[[2]] < result_tensor[[3]]);
            }
            _ => panic!("Expected tensor value from sigmoid operation"),
        }
    }

    // ===== TANH ACTIVATION FUNCTION TESTS (Phase 3B-1) =====
    // TDD Tests for Tanh[] function

    #[test]
    fn test_tanh_single_element() {
        // RED: Test tanh on single element tensor - Tanh[{0}] -> {0}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[1]), 
            vec![0.0]
        ).unwrap());
        
        let result = tanh(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[1]);
                assert!((result_tensor[[0]] - 0.0).abs() < 1e-10); // tanh(0) = 0
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_basic_values() {
        // RED: Test tanh on basic values - Tanh[{-1, 0, 1}] -> {-0.761594, 0, 0.761594}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![-1.0, 0.0, 1.0]
        ).unwrap());
        
        let result = tanh(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // tanh(-1) ≈ -0.761594
                assert!((result_tensor[[0]] - (-0.7615941559557649)).abs() < 1e-10);
                // tanh(0) = 0
                assert!((result_tensor[[1]] - 0.0).abs() < 1e-10);
                // tanh(1) ≈ 0.761594
                assert!((result_tensor[[2]] - 0.7615941559557649).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_symmetric_property() {
        // RED: Test tanh symmetric property - tanh(-x) = -tanh(x)
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![0.5, 1.0, 2.0, 3.0]
        ).unwrap());
        
        let tensor_neg = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![-0.5, -1.0, -2.0, -3.0]
        ).unwrap());
        
        let result_pos = tanh(&[tensor]);
        let result_neg = tanh(&[tensor_neg]);
        
        assert!(result_pos.is_ok());
        assert!(result_neg.is_ok());
        
        match (result_pos.unwrap(), result_neg.unwrap()) {
            (Value::Tensor(pos_tensor), Value::Tensor(neg_tensor)) => {
                assert_eq!(pos_tensor.shape(), neg_tensor.shape());
                for i in 0..4 {
                    // tanh(-x) should equal -tanh(x)
                    assert!((neg_tensor[[i]] + pos_tensor[[i]]).abs() < 1e-10);
                }
            }
            _ => panic!("Expected tensor values from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_large_values() {
        // RED: Test tanh with large values (should saturate to ±1)
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![-10.0, -5.0, 5.0, 10.0]
        ).unwrap());
        
        let result = tanh(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[4]);
                // tanh(-10) should be very close to -1
                assert!(result_tensor[[0]] < -0.9999);
                assert!(result_tensor[[0]] >= -1.0);
                // tanh(-5) should be close to -1
                assert!(result_tensor[[1]] < -0.999);
                assert!(result_tensor[[1]] >= -1.0);
                // tanh(5) should be close to 1
                assert!(result_tensor[[2]] > 0.999);
                assert!(result_tensor[[2]] <= 1.0);
                // tanh(10) should be very close to 1
                assert!(result_tensor[[3]] > 0.9999);
                assert!(result_tensor[[3]] <= 1.0);
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_2d_tensor() {
        // RED: Test tanh on 2D tensor - Tanh[{{-2, -1}, {1, 2}}]
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![-2.0, -1.0, 1.0, 2.0]
        ).unwrap());
        
        let result = tanh(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                // tanh(-2) ≈ -0.964028
                assert!((result_tensor[[0, 0]] - (-0.9640275800758169)).abs() < 1e-10);
                // tanh(-1) ≈ -0.761594
                assert!((result_tensor[[0, 1]] - (-0.7615941559557649)).abs() < 1e-10);
                // tanh(1) ≈ 0.761594
                assert!((result_tensor[[1, 0]] - 0.7615941559557649).abs() < 1e-10);
                // tanh(2) ≈ 0.964028
                assert!((result_tensor[[1, 1]] - 0.9640275800758169).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_range_property() {
        // RED: Test tanh range property - all outputs should be in [-1, 1] (inclusive due to floating point precision)
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[7]), 
            vec![-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0]
        ).unwrap());
        
        let result = tanh(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[7]);
                // All values should be in range [-1, 1] (inclusive due to floating point limits)
                for i in 0..7 {
                    assert!(result_tensor[[i]] >= -1.0);
                    assert!(result_tensor[[i]] <= 1.0);
                }
                // For very large negative values, should be very close to -1
                assert!(result_tensor[[0]] < -0.999); // tanh(-100)
                assert!(result_tensor[[1]] < -0.999); // tanh(-10)
                // For very large positive values, should be very close to 1
                assert!(result_tensor[[5]] > 0.999);  // tanh(10)
                assert!(result_tensor[[6]] > 0.999);  // tanh(100)
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    #[test]
    fn test_tanh_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        // Too few arguments
        let result = tanh(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = tanh(&[tensor.clone(), tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tanh_non_tensor_args() {
        // RED: Test non-tensor arguments should error
        let result = tanh(&[Value::Integer(1)]);
        assert!(result.is_err());
        
        let result = tanh(&[Value::String("hello".to_string())]);
        assert!(result.is_err());
        
        let result = tanh(&[Value::List(vec![Value::Real(1.0)])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tanh_neural_network_context() {
        // RED: Test tanh in neural network context - centered activation around 0
        let hidden_layer_output = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![-1.5, -0.5, 0.5, 1.5]
        ).unwrap());
        
        let result = tanh(&[hidden_layer_output]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[4]);
                // All values should be in range (-1, 1)
                for i in 0..4 {
                    assert!(result_tensor[[i]] > -1.0);
                    assert!(result_tensor[[i]] < 1.0);
                }
                // Values should be monotonic increasing
                assert!(result_tensor[[0]] < result_tensor[[1]]);
                assert!(result_tensor[[1]] < result_tensor[[2]]);
                assert!(result_tensor[[2]] < result_tensor[[3]]);
                // Should be symmetric around 0
                assert!((result_tensor[[0]] + result_tensor[[3]]).abs() < 1e-10);
                assert!((result_tensor[[1]] + result_tensor[[2]]).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from tanh operation"),
        }
    }

    // ===== SOFTMAX ACTIVATION FUNCTION TESTS (Phase 3B-1) =====
    // TDD Tests for Softmax[] function

    #[test]
    fn test_softmax_1d_basic() {
        // RED: Test softmax on 1D tensor - Softmax[{1, 2, 3}] -> probabilities that sum to 1
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        let result = softmax(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // All values should be positive
                for i in 0..3 {
                    assert!(result_tensor[[i]] > 0.0);
                    assert!(result_tensor[[i]] < 1.0);
                }
                // Values should sum to 1 (probability distribution)
                let sum: f64 = (0..3).map(|i| result_tensor[[i]]).sum();
                assert!((sum - 1.0).abs() < 1e-10);
                // Values should be monotonic increasing (since input is increasing)
                assert!(result_tensor[[0]] < result_tensor[[1]]);
                assert!(result_tensor[[1]] < result_tensor[[2]]);
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_1d_uniform() {
        // RED: Test softmax on uniform inputs - Softmax[{2, 2, 2}] -> {1/3, 1/3, 1/3}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![2.0, 2.0, 2.0]
        ).unwrap());
        
        let result = softmax(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // All values should be equal (uniform distribution)
                let expected = 1.0 / 3.0;
                for i in 0..3 {
                    assert!((result_tensor[[i]] - expected).abs() < 1e-10);
                }
                // Sum should be 1
                let sum: f64 = (0..3).map(|i| result_tensor[[i]]).sum();
                assert!((sum - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // RED: Test softmax with large values (numerical stability) - Softmax[{100, 200, 300}]
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![100.0, 200.0, 300.0]
        ).unwrap());
        
        let result = softmax(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // Should not have NaN or infinity
                for i in 0..3 {
                    assert!(result_tensor[[i]].is_finite());
                    assert!(result_tensor[[i]] >= 0.0);
                    assert!(result_tensor[[i]] <= 1.0);
                }
                // Sum should be 1
                let sum: f64 = (0..3).map(|i| result_tensor[[i]]).sum();
                assert!((sum - 1.0).abs() < 1e-10);
                // Largest input should have probability close to 1
                assert!(result_tensor[[2]] > 0.99); // softmax(300) should dominate
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_2d_default_dim() {
        // RED: Test softmax on 2D tensor (default: last dimension) - Softmax[{{1,2},{3,4}}]
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        
        let result = softmax(&[matrix]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                // Each row should sum to 1 (applying softmax along last dimension)
                let row0_sum = result_tensor[[0, 0]] + result_tensor[[0, 1]];
                let row1_sum = result_tensor[[1, 0]] + result_tensor[[1, 1]];
                assert!((row0_sum - 1.0).abs() < 1e-10);
                assert!((row1_sum - 1.0).abs() < 1e-10);
                // All values should be positive
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(result_tensor[[i, j]] > 0.0);
                        assert!(result_tensor[[i, j]] < 1.0);
                    }
                }
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_2d_with_dim_parameter() {
        // RED: Test softmax on 2D tensor with specified dimension - Softmax[{{1,2},{3,4}}, 0]
        let matrix = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 2]), 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap());
        let dim = Value::Integer(0);
        
        let result = softmax(&[matrix, dim]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 2]);
                // Each column should sum to 1 (applying softmax along dimension 0)
                let col0_sum = result_tensor[[0, 0]] + result_tensor[[1, 0]];
                let col1_sum = result_tensor[[0, 1]] + result_tensor[[1, 1]];
                assert!((col0_sum - 1.0).abs() < 1e-10);
                assert!((col1_sum - 1.0).abs() < 1e-10);
                // All values should be positive
                for i in 0..2 {
                    for j in 0..2 {
                        assert!(result_tensor[[i, j]] > 0.0);
                        assert!(result_tensor[[i, j]] < 1.0);
                    }
                }
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_zero_inputs() {
        // RED: Test softmax with zero inputs - Softmax[{0, 0, 0}] -> {1/3, 1/3, 1/3}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![0.0, 0.0, 0.0]
        ).unwrap());
        
        let result = softmax(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                // All values should be equal (uniform distribution)
                let expected = 1.0 / 3.0;
                for i in 0..3 {
                    assert!((result_tensor[[i]] - expected).abs() < 1e-10);
                }
                // Sum should be 1
                let sum: f64 = (0..3).map(|i| result_tensor[[i]]).sum();
                assert!((sum - 1.0).abs() < 1e-10);
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_single_element() {
        // RED: Test softmax on single element - Softmax[{5}] -> {1}
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[1]), 
            vec![5.0]
        ).unwrap());
        
        let result = softmax(&[tensor]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[1]);
                assert!((result_tensor[[0]] - 1.0).abs() < 1e-10); // Only element should be 1
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    #[test]
    fn test_softmax_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[3]), 
            vec![1.0, 2.0, 3.0]
        ).unwrap());
        
        // Too few arguments
        let result = softmax(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = softmax(&[tensor.clone(), Value::Integer(0), tensor]);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_non_tensor_first_arg() {
        // RED: Test non-tensor first argument should error
        let result = softmax(&[Value::Integer(1)]);
        assert!(result.is_err());
        
        let result = softmax(&[Value::String("hello".to_string())]);
        assert!(result.is_err());
        
        let result = softmax(&[Value::List(vec![Value::Real(1.0)])]);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_invalid_dim_parameter() {
        // RED: Test invalid dimension parameter should error
        let tensor = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[2, 3]), 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap());
        
        // Dimension out of range
        let result = softmax(&[tensor.clone(), Value::Integer(2)]);
        assert!(result.is_err());
        
        // Negative dimension
        let result = softmax(&[tensor.clone(), Value::Integer(-1)]);
        assert!(result.is_err());
        
        // Non-integer dimension
        let result = softmax(&[tensor.clone(), Value::Real(1.5)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_softmax_neural_network_classification() {
        // RED: Test softmax in neural network classification context - typical logits
        let logits = Value::Tensor(ArrayD::from_shape_vec(
            IxDyn(&[4]), 
            vec![2.0, 1.0, 0.1, 3.0]
        ).unwrap());
        
        let result = softmax(&[logits]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[4]);
                // Should form a valid probability distribution
                let sum: f64 = (0..4).map(|i| result_tensor[[i]]).sum();
                assert!((sum - 1.0).abs() < 1e-10);
                
                // All probabilities should be positive
                for i in 0..4 {
                    assert!(result_tensor[[i]] > 0.0);
                    assert!(result_tensor[[i]] < 1.0);
                }
                
                // Largest logit (3.0) should have highest probability
                let max_prob_idx = (0..4)
                    .max_by(|&i, &j| result_tensor[[i]].partial_cmp(&result_tensor[[j]]).unwrap())
                    .unwrap();
                assert_eq!(max_prob_idx, 3); // Index of logit 3.0
            }
            _ => panic!("Expected tensor value from softmax operation"),
        }
    }

    // ===== WEIGHT INITIALIZATION FUNCTION TESTS (Phase 3B-2A) =====
    // TDD Tests for RandomNormal[] function

    #[test]
    fn test_random_normal_basic_shape() {
        // RED: Test RandomNormal creates tensor with correct shape - RandomNormal[{2, 3}]
        let shape = Value::List(vec![Value::Integer(2), Value::Integer(3)]);
        
        let result = random_normal(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 3]);
                assert_eq!(result_tensor.len(), 6);
                // Values should be different (very unlikely to be all the same)
                let values: Vec<f64> = result_tensor.iter().cloned().collect();
                let all_same = values.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                assert!(!all_same, "Random values should not all be identical");
            }
            _ => panic!("Expected tensor value from RandomNormal"),
        }
    }

    #[test]
    fn test_random_normal_1d_shape() {
        // RED: Test RandomNormal with 1D shape - RandomNormal[{5}]
        let shape = Value::List(vec![Value::Integer(5)]);
        
        let result = random_normal(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[5]);
                assert_eq!(result_tensor.len(), 5);
            }
            _ => panic!("Expected tensor value from RandomNormal"),
        }
    }

    #[test]
    fn test_random_normal_with_mean_std() {
        // RED: Test RandomNormal with custom mean and std - RandomNormal[{3, 3}, 5.0, 2.0]
        let shape = Value::List(vec![Value::Integer(3), Value::Integer(3)]);
        let mean = Value::Real(5.0);
        let std = Value::Real(2.0);
        
        let result = random_normal(&[shape, mean, std]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3, 3]);
                // Check that values are in reasonable range (mean ± 3*std covers ~99.7%)
                for &value in result_tensor.iter() {
                    assert!(value > 5.0 - 6.0); // mean - 3*std
                    assert!(value < 5.0 + 6.0); // mean + 3*std
                }
            }
            _ => panic!("Expected tensor value from RandomNormal"),
        }
    }

    #[test]
    fn test_random_normal_default_mean_std() {
        // RED: Test RandomNormal with default mean=0, std=1 - RandomNormal[{4, 2}]
        let shape = Value::List(vec![Value::Integer(4), Value::Integer(2)]);
        
        let result = random_normal(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[4, 2]);
                // Values should be roughly centered around 0 with std ~1
                // Check that most values are within reasonable range
                let mut in_range_count = 0;
                for &value in result_tensor.iter() {
                    if value > -3.0 && value < 3.0 {
                        in_range_count += 1;
                    }
                }
                // At least 7 out of 8 values should be within 3 standard deviations
                assert!(in_range_count >= 7);
            }
            _ => panic!("Expected tensor value from RandomNormal"),
        }
    }

    #[test]
    fn test_random_normal_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let shape = Value::List(vec![Value::Integer(2), Value::Integer(2)]);
        let mean = Value::Real(0.0);
        let std = Value::Real(1.0);
        
        // Too few arguments
        let result = random_normal(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = random_normal(&[shape.clone(), mean, std, Value::Real(1.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_normal_invalid_shape() {
        // RED: Test invalid shape arguments should error
        
        // Non-list shape
        let result = random_normal(&[Value::Integer(5)]);
        assert!(result.is_err());
        
        // Empty shape list
        let empty_shape = Value::List(vec![]);
        let result = random_normal(&[empty_shape]);
        assert!(result.is_err());
        
        // Non-integer in shape
        let invalid_shape = Value::List(vec![Value::Real(2.5), Value::Integer(3)]);
        let result = random_normal(&[invalid_shape]);
        assert!(result.is_err());
        
        // Negative dimension
        let negative_shape = Value::List(vec![Value::Integer(-1), Value::Integer(3)]);
        let result = random_normal(&[negative_shape]);
        assert!(result.is_err());
        
        // Zero dimension
        let zero_shape = Value::List(vec![Value::Integer(0), Value::Integer(3)]);
        let result = random_normal(&[zero_shape]);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_normal_invalid_mean_std() {
        // RED: Test invalid mean/std arguments should error
        let shape = Value::List(vec![Value::Integer(2), Value::Integer(2)]);
        
        // Non-numeric mean
        let result = random_normal(&[shape.clone(), Value::String("invalid".to_string())]);
        assert!(result.is_err());
        
        // Non-numeric std
        let result = random_normal(&[shape.clone(), Value::Real(0.0), Value::String("invalid".to_string())]);
        assert!(result.is_err());
        
        // Negative std
        let result = random_normal(&[shape.clone(), Value::Real(0.0), Value::Real(-1.0)]);
        assert!(result.is_err());
        
        // Zero std
        let result = random_normal(&[shape.clone(), Value::Real(0.0), Value::Real(0.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_random_normal_deterministic_with_seed() {
        // RED: Test that same operations with same seed produce same results
        let shape = Value::List(vec![Value::Integer(2), Value::Integer(3)]);
        
        // Note: This test will need to be updated when we implement seeding
        // For now, just verify that multiple calls produce different results
        let result1 = random_normal(&[shape.clone()]);
        let result2 = random_normal(&[shape.clone()]);
        
        assert!(result1.is_ok());
        assert!(result2.is_ok());
        
        match (result1.unwrap(), result2.unwrap()) {
            (Value::Tensor(tensor1), Value::Tensor(tensor2)) => {
                assert_eq!(tensor1.shape(), tensor2.shape());
                // Very unlikely that all values are identical
                let all_same = tensor1.iter().zip(tensor2.iter())
                    .all(|(&a, &b)| (a - b).abs() < 1e-10);
                assert!(!all_same, "Different calls should produce different random values");
            }
            _ => panic!("Expected tensor values from RandomNormal"),
        }
    }

    #[test]
    fn test_random_normal_neural_network_context() {
        // RED: Test RandomNormal in neural network context - weight initialization
        let weight_shape = Value::List(vec![Value::Integer(784), Value::Integer(128)]); // MNIST input to hidden
        let bias_shape = Value::List(vec![Value::Integer(128)]);
        
        let weights = random_normal(&[weight_shape]);
        let biases = random_normal(&[bias_shape, Value::Real(0.0), Value::Real(0.1)]); // Small std for biases
        
        assert!(weights.is_ok());
        assert!(biases.is_ok());
        
        match (weights.unwrap(), biases.unwrap()) {
            (Value::Tensor(w_tensor), Value::Tensor(b_tensor)) => {
                assert_eq!(w_tensor.shape(), &[784, 128]);
                assert_eq!(b_tensor.shape(), &[128]);
                
                // Weights should be roughly centered around 0
                let weight_values: Vec<f64> = w_tensor.iter().cloned().collect();
                let weight_mean = weight_values.iter().sum::<f64>() / weight_values.len() as f64;
                assert!(weight_mean.abs() < 0.5); // Should be close to 0
                
                // Biases should be small (close to 0 with small std)
                for &bias in b_tensor.iter() {
                    assert!(bias.abs() < 1.0); // Within reasonable range for small std
                }
            }
            _ => panic!("Expected tensor values for neural network initialization"),
        }
    }

    // ===== XAVIER INITIALIZATION FUNCTION TESTS (Phase 3B-2A) =====
    // TDD Tests for Xavier[] function

    #[test]
    fn test_xavier_basic_shape() {
        // RED: Test Xavier creates tensor with correct shape - Xavier[{10, 5}]
        let shape = Value::List(vec![Value::Integer(10), Value::Integer(5)]);
        
        let result = xavier(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[10, 5]);
                assert_eq!(result_tensor.len(), 50);
                // Values should be different (very unlikely to be all the same)
                let values: Vec<f64> = result_tensor.iter().cloned().collect();
                let all_same = values.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-10);
                assert!(!all_same, "Xavier values should not all be identical");
            }
            _ => panic!("Expected tensor value from Xavier"),
        }
    }

    #[test]
    fn test_xavier_variance_scaling() {
        // RED: Test Xavier initialization has correct variance scaling
        let fan_in = 100;
        let fan_out = 50;
        let shape = Value::List(vec![Value::Integer(fan_in), Value::Integer(fan_out)]);
        
        let result = xavier(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[fan_in as usize, fan_out as usize]);
                
                // Calculate theoretical Xavier bound: sqrt(6 / (fan_in + fan_out))
                let xavier_bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
                
                // All values should be within the Xavier bounds
                for &value in result_tensor.iter() {
                    assert!(value >= -xavier_bound);
                    assert!(value <= xavier_bound);
                }
                
                // Check that values span a reasonable range (not all tiny)
                let max_val = result_tensor.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_val = result_tensor.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                assert!(max_val - min_val > 0.1); // Should have reasonable spread
            }
            _ => panic!("Expected tensor value from Xavier"),
        }
    }

    #[test]
    fn test_xavier_square_matrix() {
        // RED: Test Xavier with square matrix - Xavier[{64, 64}]
        let shape = Value::List(vec![Value::Integer(64), Value::Integer(64)]);
        
        let result = xavier(&[shape]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[64, 64]);
                
                // Xavier bound for square matrix: sqrt(6 / (64 + 64)) = sqrt(6/128)
                let xavier_bound = (6.0_f64 / 128.0_f64).sqrt();
                
                for &value in result_tensor.iter() {
                    assert!(value >= -xavier_bound);
                    assert!(value <= xavier_bound);
                }
            }
            _ => panic!("Expected tensor value from Xavier"),
        }
    }

    #[test]
    fn test_xavier_asymmetric_layers() {
        // RED: Test Xavier with different layer sizes
        let test_cases = vec![
            (784, 256), // MNIST input to hidden
            (256, 128), // Hidden to hidden
            (128, 10),  // Hidden to output
            (1, 1000),  // Extreme case: 1 input to many outputs
            (1000, 1),  // Extreme case: many inputs to 1 output
        ];
        
        for (fan_in, fan_out) in test_cases {
            let shape = Value::List(vec![Value::Integer(fan_in), Value::Integer(fan_out)]);
            let result = xavier(&[shape]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Tensor(result_tensor) => {
                    assert_eq!(result_tensor.shape(), &[fan_in as usize, fan_out as usize]);
                    
                    let xavier_bound = (6.0 / (fan_in + fan_out) as f64).sqrt();
                    
                    // Check bounds
                    for &value in result_tensor.iter() {
                        assert!(value >= -xavier_bound, 
                            "Value {} exceeds lower bound {} for shape [{}, {}]", 
                            value, -xavier_bound, fan_in, fan_out);
                        assert!(value <= xavier_bound,
                            "Value {} exceeds upper bound {} for shape [{}, {}]", 
                            value, xavier_bound, fan_in, fan_out);
                    }
                }
                _ => panic!("Expected tensor value from Xavier for shape [{}, {}]", fan_in, fan_out),
            }
        }
    }

    #[test]
    fn test_xavier_wrong_number_of_args() {
        // RED: Test wrong number of arguments should error
        let shape = Value::List(vec![Value::Integer(10), Value::Integer(5)]);
        
        // Too few arguments
        let result = xavier(&[]);
        assert!(result.is_err());
        
        // Too many arguments
        let result = xavier(&[shape.clone(), Value::Real(1.0)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_xavier_invalid_shape() {
        // RED: Test invalid shape arguments should error
        
        // Non-list shape
        let result = xavier(&[Value::Integer(5)]);
        assert!(result.is_err());
        
        // Wrong number of dimensions (Xavier requires exactly 2D: [fan_in, fan_out])
        let shape_1d = Value::List(vec![Value::Integer(10)]);
        let result = xavier(&[shape_1d]);
        assert!(result.is_err());
        
        let shape_3d = Value::List(vec![Value::Integer(10), Value::Integer(5), Value::Integer(3)]);
        let result = xavier(&[shape_3d]);
        assert!(result.is_err());
        
        // Non-integer in shape
        let invalid_shape = Value::List(vec![Value::Real(10.5), Value::Integer(5)]);
        let result = xavier(&[invalid_shape]);
        assert!(result.is_err());
        
        // Negative dimensions
        let negative_shape = Value::List(vec![Value::Integer(-10), Value::Integer(5)]);
        let result = xavier(&[negative_shape]);
        assert!(result.is_err());
        
        // Zero dimensions
        let zero_shape = Value::List(vec![Value::Integer(0), Value::Integer(5)]);
        let result = xavier(&[zero_shape]);
        assert!(result.is_err());
    }

    #[test]
    fn test_xavier_neural_network_context() {
        // RED: Test Xavier in typical neural network layer contexts
        
        // MNIST classifier layers
        let input_layer = Value::List(vec![Value::Integer(784), Value::Integer(256)]);
        let hidden_layer = Value::List(vec![Value::Integer(256), Value::Integer(128)]);
        let output_layer = Value::List(vec![Value::Integer(128), Value::Integer(10)]);
        
        let input_weights = xavier(&[input_layer]);
        let hidden_weights = xavier(&[hidden_layer]);
        let output_weights = xavier(&[output_layer]);
        
        assert!(input_weights.is_ok());
        assert!(hidden_weights.is_ok());
        assert!(output_weights.is_ok());
        
        // Verify each layer has appropriate initialization
        match (input_weights.unwrap(), hidden_weights.unwrap(), output_weights.unwrap()) {
            (Value::Tensor(input_w), Value::Tensor(hidden_w), Value::Tensor(output_w)) => {
                assert_eq!(input_w.shape(), &[784, 256]);
                assert_eq!(hidden_w.shape(), &[256, 128]);
                assert_eq!(output_w.shape(), &[128, 10]);
                
                // Input layer should have smallest weights (largest fan_in + fan_out)
                let input_bound = (6.0_f64 / (784.0_f64 + 256.0_f64)).sqrt();
                let output_bound = (6.0_f64 / (128.0_f64 + 10.0_f64)).sqrt();
                
                // Output layer should have larger bounds than input layer
                assert!(output_bound > input_bound);
                
                // Verify bounds are respected
                for &w in input_w.iter() {
                    assert!(w.abs() <= input_bound);
                }
                for &w in output_w.iter() {
                    assert!(w.abs() <= output_bound);
                }
            }
            _ => panic!("Expected tensor values for neural network Xavier initialization"),
        }
    }

    #[test]
    fn test_xavier_vs_random_normal() {
        // RED: Test that Xavier gives different distribution than standard normal
        let shape = Value::List(vec![Value::Integer(100), Value::Integer(50)]);
        
        let xavier_result = xavier(&[shape.clone()]);
        let normal_result = random_normal(&[shape.clone()]);
        
        assert!(xavier_result.is_ok());
        assert!(normal_result.is_ok());
        
        match (xavier_result.unwrap(), normal_result.unwrap()) {
            (Value::Tensor(xavier_tensor), Value::Tensor(normal_tensor)) => {
                // Both should have same shape
                assert_eq!(xavier_tensor.shape(), normal_tensor.shape());
                
                // Xavier should have smaller variance than standard normal
                let xavier_values: Vec<f64> = xavier_tensor.iter().cloned().collect();
                let normal_values: Vec<f64> = normal_tensor.iter().cloned().collect();
                
                let xavier_max = xavier_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
                let normal_max = normal_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b.abs()));
                
                // Xavier bound for this case: sqrt(6/(100+50)) ≈ 0.2
                let expected_xavier_bound = (6.0_f64 / 150.0_f64).sqrt();
                assert!(xavier_max <= expected_xavier_bound);
                
                // Standard normal can have much larger values
                // Very likely that normal_max > xavier_max
                // (This is probabilistic but should pass almost always)
            }
            _ => panic!("Expected tensor values from initialization comparison"),
        }
    }

    // Test He[] weight initialization function - for ReLU networks
    #[test]
    fn test_he_basic_2d_weight_matrix() {
        // RED phase: Test He initialization for a 2D weight matrix
        // He initialization uses sqrt(2 / fan_in) for ReLU networks
        let result = he(&[Value::List(vec![Value::Integer(784), Value::Integer(128)])]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[784, 128]);
                // Check that values are within He initialization bounds
                // He bound = sqrt(2 / 784) ≈ 0.0506
                let he_bound = (2.0_f64 / 784.0_f64).sqrt();
                for &val in tensor.iter() {
                    assert!(val >= -he_bound && val <= he_bound);
                }
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_he_single_neuron() {
        // Test He initialization for single neuron (fan_in = 1)
        // He bound = sqrt(2 / 1) = sqrt(2) ≈ 1.414
        let result = he(&[Value::List(vec![Value::Integer(1), Value::Integer(1)])]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[1, 1]);
                let he_bound = (2.0_f64 / 1.0_f64).sqrt();
                let val = tensor[[0, 0]];
                assert!(val >= -he_bound && val <= he_bound);
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_he_large_fan_in() {
        // Test He initialization with large fan_in (small weights)
        // fan_in = 10000, He bound = sqrt(2 / 10000) ≈ 0.0141
        let result = he(&[Value::List(vec![Value::Integer(10000), Value::Integer(10)])]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[10000, 10]);
                let he_bound = (2.0_f64 / 10000.0_f64).sqrt();
                // Check a few random samples
                for i in 0..10 {
                    for j in 0..10 {
                        let val = tensor[[i, j]];
                        assert!(val >= -he_bound && val <= he_bound);
                    }
                }
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_he_neural_network_context() {
        // Test He initialization in realistic neural network context
        // Hidden layer: 256 -> 128 neurons
        let result = he(&[Value::List(vec![Value::Integer(256), Value::Integer(128)])]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Tensor(tensor) => {
                assert_eq!(tensor.shape(), &[256, 128]);
                // He bound = sqrt(2 / 256) ≈ 0.0884
                let he_bound = (2.0_f64 / 256.0_f64).sqrt();
                
                // Verify statistical properties
                let values: Vec<f64> = tensor.iter().copied().collect();
                let mean: f64 = values.iter().sum::<f64>() / values.len() as f64;
                let variance: f64 = values.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / values.len() as f64;
                
                // He initialization should have mean ≈ 0 and variance ≈ 2/fan_in
                assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
                assert!((variance - 2.0_f64/256.0_f64).abs() < 0.01, 
                       "Variance should be close to 2/fan_in, got {}", variance);
            }
            _ => panic!("Expected tensor value"),
        }
    }

    #[test]
    fn test_he_error_wrong_argument_count() {
        // Test error with wrong number of arguments
        let result = he(&[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected exactly 1 argument"));
        
        let result = he(&[Value::Integer(784), Value::Integer(128)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected exactly 1 argument"));
    }

    #[test]
    fn test_he_error_invalid_shape_format() {
        // Test error with non-list argument
        let result = he(&[Value::Integer(784)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected list of two integers"));
        
        // Test error with wrong list length
        let result = he(&[Value::List(vec![Value::Integer(784)])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected list of two integers"));
        
        let result = he(&[Value::List(vec![
            Value::Integer(784), 
            Value::Integer(128), 
            Value::Integer(64)
        ])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected list of two integers"));
    }

    #[test]
    fn test_he_error_non_integer_dimensions() {
        // Test error with non-integer dimensions
        let result = he(&[Value::List(vec![Value::Real(784.5), Value::Integer(128)])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("integer for fan_in"));
        
        let result = he(&[Value::List(vec![Value::Integer(784), Value::String("128".to_string())])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("integer for fan_out"));
    }

    #[test]
    fn test_he_error_negative_dimensions() {
        // Test error with negative or zero dimensions
        let result = he(&[Value::List(vec![Value::Integer(-784), Value::Integer(128)])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive integer for fan_in"));
        
        let result = he(&[Value::List(vec![Value::Integer(784), Value::Integer(0)])]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("positive integer for fan_out"));
    }
}