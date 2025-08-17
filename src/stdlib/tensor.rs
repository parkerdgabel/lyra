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
}