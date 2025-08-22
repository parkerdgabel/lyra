//! Basic Sparse Matrix Operations
//!
//! This module provides fundamental operations on sparse matrices including
//! arithmetic, norms, slicing, and element access.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;
use super::core::{GenericSparseMatrix, extract_sparse_matrix};
use std::collections::HashMap;

/// Add two sparse matrices
pub fn sparse_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix_a = extract_sparse_matrix(&args[0])?;
    let matrix_b = extract_sparse_matrix(&args[1])?;

    // Check dimensions match
    let shape_a = matrix_a.matrix().shape();
    let shape_b = matrix_b.matrix().shape();
    
    if shape_a != shape_b {
        return Err(VmError::Runtime(format!(
            "Matrix dimensions do not match: {:?} vs {:?}", 
            shape_a, shape_b
        )));
    }

    // For now, implement a simple addition by converting to triplets
    let triplets_a = matrix_a.matrix().triplets();
    let triplets_b = matrix_b.matrix().triplets();
    
    // Combine triplets and sum values for same coordinates
    let mut combined = HashMap::new();
    
    for (i, j, val) in triplets_a {
        *combined.entry((i, j)).or_insert(0.0) += val;
    }
    
    for (i, j, val) in triplets_b {
        *combined.entry((i, j)).or_insert(0.0) += val;
    }
    
    // Convert back to triplets, filtering out zeros
    let result_triplets: Vec<(usize, usize, f64)> = combined
        .into_iter()
        .filter(|&(_, val)| val.abs() > 1e-12)
        .map(|((i, j), val)| (i, j, val))
        .collect();
    
    // Create a new COO matrix with the result
    let coo_matrix = super::coo::COOMatrix::from_triplets(
        result_triplets,
        shape_a
    );
    
    let generic_matrix = GenericSparseMatrix::new(Box::new(coo_matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(generic_matrix))))
}

/// Multiply two sparse matrices
pub fn sparse_multiply(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix_a = extract_sparse_matrix(&args[0])?;
    let matrix_b = extract_sparse_matrix(&args[1])?;

    let shape_a = matrix_a.matrix().shape();
    let shape_b = matrix_b.matrix().shape();
    
    if shape_a.1 != shape_b.0 {
        return Err(VmError::Runtime(format!(
            "Cannot multiply matrices: ({}, {}) x ({}, {})", 
            shape_a.0, shape_a.1, shape_b.0, shape_b.1
        )));
    }

    // Simple matrix multiplication using triplets
    let triplets_a = matrix_a.matrix().triplets();
    let triplets_b = matrix_b.matrix().triplets();
    
    // Build row index for A and column index for B for efficient lookup
    let mut rows_a: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for (i, j, val) in triplets_a {
        rows_a.entry(i).or_default().push((j, val));
    }
    
    let mut cols_b: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    for (i, j, val) in triplets_b {
        cols_b.entry(j).or_default().push((i, val));
    }
    
    // Compute result
    let mut result = HashMap::new();
    
    for (i, row_a) in rows_a {
        for (j, col_b) in &cols_b {
            let mut sum = 0.0;
            for &(k_a, val_a) in &row_a {
                for &(k_b, val_b) in col_b {
                    if k_a == k_b {
                        sum += val_a * val_b;
                    }
                }
            }
            if sum.abs() > 1e-12 {
                result.insert((i, *j), sum);
            }
        }
    }
    
    let result_triplets: Vec<(usize, usize, f64)> = result
        .into_iter()
        .map(|((i, j), val)| (i, j, val))
        .collect();
    
    let result_shape = (shape_a.0, shape_b.1);
    let coo_matrix = super::coo::COOMatrix::from_triplets(result_triplets, result_shape);
    
    let generic_matrix = GenericSparseMatrix::new(Box::new(coo_matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(generic_matrix))))
}

/// Transpose a sparse matrix
pub fn sparse_transpose(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = extract_sparse_matrix(&args[0])?;
    let shape = matrix.matrix().shape();
    let triplets = matrix.matrix().triplets();
    
    // Transpose by swapping i and j coordinates
    let transposed_triplets: Vec<(usize, usize, f64)> = triplets
        .into_iter()
        .map(|(i, j, val)| (j, i, val))
        .collect();
    
    let transposed_shape = (shape.1, shape.0);
    let coo_matrix = super::coo::COOMatrix::from_triplets(transposed_triplets, transposed_shape);
    
    let generic_matrix = GenericSparseMatrix::new(Box::new(coo_matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(generic_matrix))))
}

/// Get sparse matrix shape
pub fn sparse_shape(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = extract_sparse_matrix(&args[0])?;
    let shape = matrix.matrix().shape();
    
    Ok(Value::List(vec![
        Value::Integer(shape.0 as i64),
        Value::Integer(shape.1 as i64),
    ]))
}

/// Get number of non-zero elements
pub fn sparse_nnz(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = extract_sparse_matrix(&args[0])?;
    let nnz = matrix.matrix().nnz();
    
    Ok(Value::Integer(nnz as i64))
}

/// Get matrix density (nnz / total_elements)
pub fn sparse_density(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = extract_sparse_matrix(&args[0])?;
    let shape = matrix.matrix().shape();
    let nnz = matrix.matrix().nnz();
    let total_elements = shape.0 * shape.1;
    
    if total_elements == 0 {
        return Ok(Value::Real(0.0));
    }
    
    let density = nnz as f64 / total_elements as f64;
    Ok(Value::Real(density))
}

/// Convert sparse matrix to dense representation
pub fn sparse_to_dense(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = extract_sparse_matrix(&args[0])?;
    let shape = matrix.matrix().shape();
    let triplets = matrix.matrix().triplets();
    
    // Create dense matrix initialized to zero
    let mut dense = vec![vec![0.0; shape.1]; shape.0];
    
    // Fill in non-zero values
    for (i, j, val) in triplets {
        if i < shape.0 && j < shape.1 {
            dense[i][j] = val;
        }
    }
    
    // Convert to Lyra Value format
    let rows: Vec<Value> = dense
        .into_iter()
        .map(|row| {
            let values: Vec<Value> = row.into_iter().map(Value::Real).collect();
            Value::List(values)
        })
        .collect();
    
    Ok(Value::List(rows))
}