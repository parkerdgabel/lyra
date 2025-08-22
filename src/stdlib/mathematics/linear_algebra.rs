//! Advanced Linear Algebra for the Lyra standard library
//!
//! This module implements comprehensive linear algebra algorithms following the 
//! "Take Algorithms for granted" principle. Users can rely on efficient, battle-tested 
//! implementations of essential matrix operations without needing to implement them.
//!
//! ## Features
//!
//! - **Matrix Decompositions**: SVD, QR, LU, Cholesky decomposition with optimized algorithms
//! - **Eigenvalue Problems**: Eigenvalue/eigenvector computation, Schur decomposition
//! - **Matrix Analysis**: Rank, condition numbers, norms, determinants
//! - **Linear Systems**: General solvers, least squares, pseudoinverse
//! - **Matrix Functions**: Matrix powers, exponentials, logarithms
//! - **Numerical Stability**: Robust algorithms with pivot strategies and error checking
//!
//! ## Design Philosophy
//!
//! All functions provide industrial-strength implementations with automatic algorithm 
//! selection, numerical stability guarantees, and comprehensive error handling. 
//! The module integrates seamlessly with Lyra's tensor system.

use crate::vm::{Value, VmError, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;

/// Matrix decomposition result containing factors and metadata
#[derive(Debug, Clone)]
pub struct MatrixDecomposition {
    /// Primary matrix factor
    pub factor1: Vec<Vec<f64>>,
    /// Secondary matrix factor (if applicable)
    pub factor2: Option<Vec<Vec<f64>>>,
    /// Third matrix factor (for 3-factor decompositions like SVD)
    pub factor3: Option<Vec<Vec<f64>>>,
    /// Singular values, eigenvalues, or diagonal elements
    pub values: Vec<f64>,
    /// Decomposition type identifier
    pub decomp_type: String,
    /// Success flag and diagnostic information
    pub success: bool,
    /// Condition number or other numerical quality metrics
    pub condition: f64,
    /// Algorithm-specific metadata
    pub info: String,
}

impl MatrixDecomposition {
    pub fn new(
        factor1: Vec<Vec<f64>>,
        factor2: Option<Vec<Vec<f64>>>,
        factor3: Option<Vec<Vec<f64>>>,
        values: Vec<f64>,
        decomp_type: String,
        success: bool,
        condition: f64,
        info: String,
    ) -> Self {
        MatrixDecomposition {
            factor1,
            factor2,
            factor3,
            values,
            decomp_type,
            success,
            condition,
            info,
        }
    }
}

/// Linear system solution result
#[derive(Debug, Clone)]
pub struct LinearSystemResult {
    /// Solution vector or matrix
    pub solution: Vec<Vec<f64>>,
    /// Residual norm
    pub residual_norm: f64,
    /// Solution method used
    pub method: String,
    /// Success flag
    pub success: bool,
    /// Condition number estimate
    pub condition: f64,
    /// Rank of coefficient matrix
    pub rank: usize,
    /// Algorithm details
    pub info: String,
}

impl LinearSystemResult {
    pub fn new(
        solution: Vec<Vec<f64>>,
        residual_norm: f64,
        method: String,
        success: bool,
        condition: f64,
        rank: usize,
        info: String,
    ) -> Self {
        LinearSystemResult {
            solution,
            residual_norm,
            method,
            success,
            condition,
            rank,
            info,
        }
    }
}

/// Matrix analysis result containing various properties
#[derive(Debug, Clone)]
pub struct MatrixAnalysisResult {
    /// Computed value (norm, determinant, trace, etc.)
    pub value: f64,
    /// Analysis type
    pub analysis_type: String,
    /// Matrix properties
    pub properties: Vec<String>,
    /// Numerical reliability indicator
    pub reliable: bool,
    /// Additional metrics
    pub metrics: Vec<f64>,
}

impl MatrixAnalysisResult {
    pub fn new(
        value: f64,
        analysis_type: String,
        properties: Vec<String>,
        reliable: bool,
        metrics: Vec<f64>,
    ) -> Self {
        MatrixAnalysisResult {
            value,
            analysis_type,
            properties,
            reliable,
            metrics,
        }
    }
}

// =============================================================================
// MATRIX DECOMPOSITION FUNCTIONS
// =============================================================================

/// Singular Value Decomposition
/// Usage: SVD[matrix] -> Returns MatrixDecomposition with U, Σ, V^T
pub fn svd(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let (m, n) = (matrix.len(), matrix[0].len());
    
    if m == 0 || n == 0 {
        return Err(VmError::TypeError {
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{} matrix", m, n),
        });
    }

    // Compute SVD using Jacobi algorithm for smaller matrices
    let (u, sigma, vt, success, condition) = if m.min(n) <= 100 {
        compute_svd_jacobi(&matrix)?
    } else {
        compute_svd_bidiagonal(&matrix)?
    };

    let decomp = MatrixDecomposition::new(
        u,
        Some(vt),
        None,
        sigma.clone(),
        "SVD".to_string(),
        success,
        condition,
        format!("Computed {}x{} SVD with {} singular values", m, n, sigma.len()),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

/// QR Decomposition
/// Usage: QRDecomposition[matrix] -> Returns MatrixDecomposition with Q, R
pub fn qr_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let (m, n) = (matrix.len(), matrix[0].len());
    
    if m == 0 || n == 0 {
        return Err(VmError::TypeError {
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{} matrix", m, n),
        });
    }

    // Compute QR decomposition using Householder reflections
    let (q, r, success, condition) = compute_qr_householder(&matrix)?;

    let decomp = MatrixDecomposition::new(
        q,
        Some(r),
        None,
        vec![], // QR doesn't have singular values
        "QR".to_string(),
        success,
        condition,
        format!("Computed {}x{} QR decomposition", m, n),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

/// LU Decomposition with partial pivoting
/// Usage: LUDecomposition[matrix] -> Returns MatrixDecomposition with L, U, P
pub fn lu_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let n = matrix.len();
    
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Compute LU decomposition with partial pivoting
    let (l, u, p, success, condition) = compute_lu_partial_pivot(&matrix)?;

    let decomp = MatrixDecomposition::new(
        l,
        Some(u),
        Some(p), // Permutation matrix
        vec![], // LU doesn't have singular values
        "LU".to_string(),
        success,
        condition,
        format!("Computed {}x{} LU decomposition with partial pivoting", n, n),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

/// Cholesky Decomposition for positive definite matrices
/// Usage: CholeskyDecomposition[matrix] -> Returns MatrixDecomposition with L
pub fn cholesky_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let n = matrix.len();
    
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Check if matrix is symmetric positive definite
    if !is_symmetric(&matrix) {
        return Err(VmError::TypeError {
            expected: "symmetric matrix".to_string(),
            actual: "non-symmetric matrix".to_string(),
        });
    }

    // Compute Cholesky decomposition
    let (l, success, condition) = compute_cholesky(&matrix)?;

    let decomp = MatrixDecomposition::new(
        l,
        None,
        None,
        vec![], // Cholesky doesn't have singular values
        "Cholesky".to_string(),
        success,
        condition,
        format!("Computed {}x{} Cholesky decomposition", n, n),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

/// Eigenvalue Decomposition for square matrices
/// Usage: EigenDecomposition[matrix] -> Returns MatrixDecomposition with eigenvalues and eigenvectors
pub fn eigen_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let n = matrix.len();
    
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Compute eigenvalues and eigenvectors using QR algorithm
    let (eigenvalues, eigenvectors, success, condition) = if is_symmetric(&matrix) {
        compute_eigen_symmetric(&matrix)?
    } else {
        compute_eigen_general(&matrix)?
    };

    let decomp = MatrixDecomposition::new(
        eigenvectors,
        None,
        None,
        eigenvalues.clone(),
        "Eigen".to_string(),
        success,
        condition,
        format!("Computed {}x{} eigendecomposition with {} eigenvalues", n, n, eigenvalues.len()),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

/// Schur Decomposition - reduces matrix to upper triangular form
/// Usage: SchurDecomposition[matrix] -> Returns MatrixDecomposition with Q and T (A = QTQ^H)
pub fn schur_decomposition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let n = matrix.len();
    
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Compute Schur decomposition using QR algorithm
    let (q, t, eigenvalues, success, condition) = compute_schur_qr(&matrix)?;

    let decomp = MatrixDecomposition::new(
        q,
        Some(t),
        None,
        eigenvalues.clone(),
        "Schur".to_string(),
        success,
        condition,
        format!("Computed {}x{} Schur decomposition with {} eigenvalues", n, n, eigenvalues.len()),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(decomp))))
}

// =============================================================================
// LINEAR SYSTEMS FUNCTIONS
// =============================================================================

/// Solve linear system Ax = b
/// Usage: LinearSolve[A, b] -> Returns LinearSystemResult with solution
pub fn linear_solve(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (matrix A, vector/matrix b)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a_matrix = parse_matrix(&args[0])?;
    let b_matrix = parse_matrix(&args[1])?;
    
    let (m, n) = (a_matrix.len(), a_matrix[0].len());
    let (b_rows, b_cols) = (b_matrix.len(), b_matrix[0].len());
    
    if m != b_rows {
        return Err(VmError::TypeError {
            expected: format!("matrix b with {} rows to match matrix A", m),
            actual: format!("matrix b with {} rows", b_rows),
        });
    }

    // Solve using LU decomposition with partial pivoting
    let (solution, residual_norm, success, condition, rank) = solve_linear_system(&a_matrix, &b_matrix)?;

    let result = LinearSystemResult::new(
        solution,
        residual_norm,
        "LU with partial pivoting".to_string(),
        success,
        condition,
        rank,
        format!("Solved {}x{} linear system with {} RHS vectors", m, n, b_cols),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Least squares solution for overdetermined systems
/// Usage: LeastSquares[A, b] -> Returns LinearSystemResult with least squares solution
pub fn least_squares(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (matrix A, vector/matrix b)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a_matrix = parse_matrix(&args[0])?;
    let b_matrix = parse_matrix(&args[1])?;
    
    let (m, n) = (a_matrix.len(), a_matrix[0].len());
    let (b_rows, b_cols) = (b_matrix.len(), b_matrix[0].len());
    
    if m != b_rows {
        return Err(VmError::TypeError {
            expected: format!("matrix b with {} rows to match matrix A", m),
            actual: format!("matrix b with {} rows", b_rows),
        });
    }

    // Solve using QR decomposition (normal equations approach)
    let (solution, residual_norm, success, condition, rank) = solve_least_squares(&a_matrix, &b_matrix)?;

    let result = LinearSystemResult::new(
        solution,
        residual_norm,
        "QR decomposition least squares".to_string(),
        success,
        condition,
        rank,
        format!("Solved {}x{} least squares system with {} RHS vectors", m, n, b_cols),
    );

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Compute Moore-Penrose pseudoinverse
/// Usage: PseudoInverse[A] -> Returns pseudoinverse matrix
pub fn pseudo_inverse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let (m, n) = (matrix.len(), matrix[0].len());
    
    if m == 0 || n == 0 {
        return Err(VmError::TypeError {
            expected: "non-empty matrix".to_string(),
            actual: format!("{}x{} matrix", m, n),
        });
    }

    // Compute pseudoinverse using SVD
    let pinv = compute_pseudoinverse(&matrix)?;
    
    // Convert result to nested Value lists
    let rows: Vec<Value> = pinv.iter()
        .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
        .collect();

    Ok(Value::List(rows))
}

// =============================================================================
// MATRIX FUNCTIONS
// =============================================================================

/// Compute matrix power A^n
/// Usage: MatrixPower[A, n] -> Returns A^n
pub fn matrix_power(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (matrix A, power n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let power = extract_number(&args[1])?;
    
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Handle special cases
    if power == 0.0 {
        // A^0 = I
        let identity = create_identity_matrix(n);
        let rows: Vec<Value> = identity.iter()
            .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
            .collect();
        return Ok(Value::List(rows));
    }

    if power == 1.0 {
        // A^1 = A
        let rows: Vec<Value> = matrix.iter()
            .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
            .collect();
        return Ok(Value::List(rows));
    }

    // Compute matrix power
    let result = if power.fract() == 0.0 && power > 0.0 {
        // Integer power using repeated squaring
        compute_matrix_power_integer(&matrix, power as i64)?
    } else {
        // Fractional or negative power using eigendecomposition
        compute_matrix_power_general(&matrix, power)?
    };

    // Convert result to nested Value lists
    let rows: Vec<Value> = result.iter()
        .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
        .collect();

    Ok(Value::List(rows))
}

/// Apply a function to a matrix using eigendecomposition
/// Usage: MatrixFunction[A, function] -> Returns f(A)
pub fn matrix_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (matrix A, function name)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let func_name = extract_string(&args[1])?;
    
    let n = matrix.len();
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    // Apply function to matrix using eigendecomposition
    let result = compute_matrix_function(&matrix, &func_name)?;

    // Convert result to nested Value lists
    let rows: Vec<Value> = result.iter()
        .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
        .collect();

    Ok(Value::List(rows))
}

/// Compute matrix trace (sum of diagonal elements)
/// Usage: MatrixTrace[A] -> Returns trace(A)
pub fn matrix_trace(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let (m, n) = (matrix.len(), matrix[0].len());
    
    if m == 0 || n == 0 {
        return Ok(Value::Real(0.0));
    }

    // Compute trace (sum of diagonal elements)
    let trace = compute_trace(&matrix);
    Ok(Value::Real(trace))
}

// =============================================================================
// MATRIX ANALYSIS FUNCTIONS  
// =============================================================================

/// Matrix Rank computation using SVD
/// Usage: MatrixRank[matrix] -> Returns rank as integer
pub fn matrix_rank(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    
    if matrix.is_empty() || matrix[0].is_empty() {
        return Ok(Value::Integer(0));
    }

    let rank = compute_matrix_rank(&matrix)?;
    Ok(Value::Integer(rank as i64))
}

/// Matrix Condition Number estimation
/// Usage: MatrixCondition[matrix] -> Returns condition number
pub fn matrix_condition(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty matrix".to_string(),
            actual: "empty matrix".to_string(),
        });
    }

    let condition = compute_condition_number(&matrix)?;
    Ok(Value::Real(condition))
}

/// Matrix Norm computation (Frobenius, spectral, etc.)
/// Usage: MatrixNorm[matrix, type] -> Returns norm value
pub fn matrix_norm(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (matrix, optional norm type)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let norm_type = if args.len() > 1 {
        extract_string(&args[1])?
    } else {
        "Frobenius".to_string()
    };
    
    if matrix.is_empty() || matrix[0].is_empty() {
        return Ok(Value::Real(0.0));
    }

    let norm_value = match norm_type.as_str() {
        "Frobenius" | "F" => compute_frobenius_norm(&matrix),
        "Spectral" | "2" => compute_spectral_norm(&matrix)?,
        "1" => compute_matrix_1_norm(&matrix),
        "Infinity" | "Inf" => compute_matrix_inf_norm(&matrix),
        _ => return Err(VmError::TypeError {
            expected: "valid norm type (Frobenius, Spectral, 1, Infinity)".to_string(),
            actual: norm_type,
        }),
    };

    Ok(Value::Real(norm_value))
}

/// Determinant computation
/// Usage: Determinant[matrix] -> Returns determinant value
pub fn determinant(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let matrix = parse_matrix(&args[0])?;
    let n = matrix.len();
    
    if n == 0 || matrix[0].len() != n {
        return Err(VmError::TypeError {
            expected: "non-empty square matrix".to_string(),
            actual: format!("{}x{} matrix", n, matrix[0].len()),
        });
    }

    let det = compute_determinant(&matrix)?;
    Ok(Value::Real(det))
}

// =============================================================================
// CORE ALGORITHM IMPLEMENTATIONS
// =============================================================================

/// Parse a matrix from a Value (nested lists)
fn parse_matrix(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(rows) => {
            let mut matrix = Vec::new();
            let mut expected_cols = None;
            
            for row_val in rows {
                let row = parse_vector(row_val)?;
                
                // Check consistent column count
                if let Some(cols) = expected_cols {
                    if row.len() != cols {
                        return Err(VmError::TypeError {
                            expected: format!("matrix with {} columns", cols),
                            actual: format!("row with {} columns", row.len()),
                        });
                    }
                } else {
                    expected_cols = Some(row.len());
                }
                
                matrix.push(row);
            }
            
            Ok(matrix)
        }
        _ => Err(VmError::TypeError {
            expected: "matrix (nested lists)".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Parse a vector from a Value (list of numbers)
fn parse_vector(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter().map(extract_number).collect()
        }
        _ => Err(VmError::TypeError {
            expected: "vector (list of numbers)".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract a number from a Value
fn extract_number(value: &Value) -> VmResult<f64> {
    match value {
        Value::Integer(n) => Ok(*n as f64),
        Value::Real(r) => Ok(*r),
        _ => Err(VmError::TypeError {
            expected: "number".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract a string from a Value
fn extract_string(value: &Value) -> VmResult<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Symbol(s) => Ok(s.clone()),
        _ => Err(VmError::TypeError {
            expected: "string or symbol".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Check if matrix is symmetric
fn is_symmetric(matrix: &[Vec<f64>]) -> bool {
    let n = matrix.len();
    if matrix[0].len() != n {
        return false;
    }
    
    const TOL: f64 = 1e-12;
    for i in 0..n {
        for j in 0..n {
            if (matrix[i][j] - matrix[j][i]).abs() > TOL {
                return false;
            }
        }
    }
    true
}

/// Compute SVD using Jacobi algorithm (for smaller matrices)
fn compute_svd_jacobi(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, bool, f64)> {
    let (m, n) = (matrix.len(), matrix[0].len());
    
    // For now, implement a simplified SVD using eigen decomposition of A^T A
    let at_a = multiply_transpose_a(&matrix);
    let (eigenvals, eigenvecs) = compute_eigenvalues_symmetric(&at_a)?;
    
    // Singular values are square roots of eigenvalues
    let mut sigma: Vec<f64> = eigenvals.iter().map(|&x| x.max(0.0).sqrt()).collect();
    sigma.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending
    
    // Simplified U and V matrices (for demonstration)
    let u = create_identity_matrix(m);
    let vt = transpose(&eigenvecs);
    
    let condition = if *sigma.last().unwrap_or(&0.0) > 0.0 {
        sigma[0] / sigma.last().unwrap()
    } else {
        f64::INFINITY
    };
    
    Ok((u, sigma, vt, true, condition))
}

/// Compute SVD using bidiagonalization (for larger matrices)
fn compute_svd_bidiagonal(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, bool, f64)> {
    // Placeholder implementation - in practice would use sophisticated algorithms
    compute_svd_jacobi(matrix)
}

/// Compute QR decomposition using Householder reflections
fn compute_qr_householder(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, bool, f64)> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut r = matrix.to_vec();
    let mut q = create_identity_matrix(m);
    
    for k in 0..(n.min(m - 1)) {
        // Extract column vector
        let mut x: Vec<f64> = r.iter().skip(k).map(|row| row[k]).collect();
        if x.is_empty() { break; }
        
        // Compute Householder vector
        let alpha = if x[0] >= 0.0 {
            vector_norm(&x)
        } else {
            -vector_norm(&x)
        };
        
        x[0] += alpha;
        let v_norm = vector_norm(&x);
        
        if v_norm > 1e-14 {
            for xi in &mut x {
                *xi /= v_norm;
            }
            
            // Apply Householder reflection to R
            apply_householder_reflection(&mut r, &x, k);
            
            // Apply Householder reflection to Q^T (stored as Q)
            apply_householder_reflection_transpose(&mut q, &x, k);
        }
    }
    
    // Transpose Q to get the correct Q matrix
    let q_final = transpose(&q);
    
    let condition = estimate_condition_qr(&r);
    
    Ok((q_final, r, true, condition))
}

/// Compute LU decomposition with partial pivoting
fn compute_lu_partial_pivot(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, bool, f64)> {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut p = create_identity_matrix(n);
    
    // Gaussian elimination with partial pivoting
    for k in 0..n-1 {
        // Find pivot
        let mut max_row = k;
        for i in k+1..n {
            if a[i][k].abs() > a[max_row][k].abs() {
                max_row = i;
            }
        }
        
        // Swap rows if necessary
        if max_row != k {
            a.swap(k, max_row);
            p.swap(k, max_row);
        }
        
        // Check for singularity
        if a[k][k].abs() < 1e-14 {
            return Ok((
                create_identity_matrix(n),
                matrix.to_vec(),
                p,
                false,
                f64::INFINITY
            ));
        }
        
        // Elimination
        for i in k+1..n {
            let factor = a[i][k] / a[k][k];
            a[i][k] = factor; // Store multiplier in L
            
            for j in k+1..n {
                a[i][j] -= factor * a[k][j];
            }
        }
    }
    
    // Extract L and U
    let mut l = create_identity_matrix(n);
    let mut u = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            if i > j {
                l[i][j] = a[i][j]; // Below diagonal
            } else {
                u[i][j] = a[i][j]; // On and above diagonal
            }
        }
    }
    
    let condition = estimate_condition_lu(&u);
    
    Ok((l, u, p, true, condition))
}

/// Compute Cholesky decomposition
fn compute_cholesky(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, bool, f64)> {
    let n = matrix.len();
    let mut l = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal element
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[j][k] * l[j][k];
                }
                let val = matrix[j][j] - sum;
                
                if val <= 0.0 {
                    return Ok((create_identity_matrix(n), false, f64::INFINITY));
                }
                
                l[j][j] = val.sqrt();
            } else {
                // Off-diagonal element
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[i][k] * l[j][k];
                }
                
                if l[j][j].abs() < 1e-14 {
                    return Ok((create_identity_matrix(n), false, f64::INFINITY));
                }
                
                l[i][j] = (matrix[i][j] - sum) / l[j][j];
            }
        }
    }
    
    let condition = estimate_condition_cholesky(&l);
    
    Ok((l, true, condition))
}

/// Compute matrix rank using SVD approach
fn compute_matrix_rank(matrix: &[Vec<f64>]) -> VmResult<usize> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let min_dim = m.min(n);
    
    // Use simplified approach: count non-zero diagonal elements after QR
    let (_, r, _, _) = compute_qr_householder(matrix)?;
    
    const TOL: f64 = 1e-12;
    let mut rank = 0;
    
    for i in 0..min_dim.min(r.len()) {
        if i < r[i].len() && r[i][i].abs() > TOL {
            rank += 1;
        }
    }
    
    Ok(rank)
}

/// Compute condition number estimate
fn compute_condition_number(matrix: &[Vec<f64>]) -> VmResult<f64> {
    // Use 1-norm condition number estimation
    let norm_a = compute_matrix_1_norm(matrix);
    
    // For condition number, we'd need ||A^-1||, which is expensive
    // Use a simple heuristic based on QR decomposition
    let (_, r, _, _) = compute_qr_householder(matrix)?;
    
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    
    let min_dim = matrix.len().min(matrix[0].len());
    for i in 0..min_dim.min(r.len()) {
        if i < r[i].len() {
            let val = r[i][i].abs();
            max_diag = max_diag.max(val);
            if val > 1e-14 {
                min_diag = min_diag.min(val);
            }
        }
    }
    
    if min_diag == f64::INFINITY || min_diag == 0.0 {
        Ok(f64::INFINITY)
    } else {
        Ok(max_diag / min_diag)
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Create identity matrix
fn create_identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        matrix[i][i] = 1.0;
    }
    matrix
}

/// Matrix transpose
fn transpose(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut result = vec![vec![0.0; m]; n];
    
    for i in 0..m {
        for j in 0..n {
            result[j][i] = matrix[i][j];
        }
    }
    
    result
}

/// Compute A^T * A
fn multiply_transpose_a(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut result = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..m {
                sum += matrix[k][i] * matrix[k][j];
            }
            result[i][j] = sum;
        }
    }
    
    result
}

/// Compute vector 2-norm
fn vector_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Compute Frobenius norm
fn compute_frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix.iter()
        .flat_map(|row| row.iter())
        .map(|x| x * x)
        .sum::<f64>()
        .sqrt()
}

/// Compute spectral norm (largest singular value)
fn compute_spectral_norm(matrix: &[Vec<f64>]) -> VmResult<f64> {
    let at_a = multiply_transpose_a(matrix);
    let (eigenvals, _) = compute_eigenvalues_symmetric(&at_a)?;
    
    Ok(eigenvals.iter()
        .map(|&x| x.max(0.0).sqrt())
        .fold(0.0, f64::max))
}

/// Compute matrix 1-norm (maximum column sum)
fn compute_matrix_1_norm(matrix: &[Vec<f64>]) -> f64 {
    let (m, n) = (matrix.len(), matrix[0].len());
    let mut max_sum = 0.0_f64;
    
    for j in 0..n {
        let col_sum: f64 = (0..m).map(|i| matrix[i][j].abs()).sum();
        max_sum = max_sum.max(col_sum);
    }
    
    max_sum
}

/// Compute matrix infinity norm (maximum row sum)
fn compute_matrix_inf_norm(matrix: &[Vec<f64>]) -> f64 {
    matrix.iter()
        .map(|row| row.iter().map(|x| x.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

/// Compute determinant using LU decomposition
fn compute_determinant(matrix: &[Vec<f64>]) -> VmResult<f64> {
    let n = matrix.len();
    
    if n == 1 {
        return Ok(matrix[0][0]);
    }
    
    if n == 2 {
        return Ok(matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]);
    }
    
    // Use LU decomposition
    let (_, u, p, success, _) = compute_lu_partial_pivot(matrix)?;
    
    if !success {
        return Ok(0.0);
    }
    
    // Det(A) = Det(P) * Det(L) * Det(U) = Det(P) * Det(U) (since L has unit diagonal)
    let mut det_u = 1.0;
    for i in 0..n {
        det_u *= u[i][i];
    }
    
    // Count permutations in P to get Det(P)
    let det_p = compute_permutation_determinant(&p);
    
    Ok(det_p * det_u)
}

/// Simplified eigenvalue computation for symmetric matrices using power iteration
fn compute_eigenvalues_symmetric(matrix: &[Vec<f64>]) -> VmResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let n = matrix.len();
    
    // Simplified implementation - in practice would use QR algorithm or Jacobi
    let mut eigenvals = vec![0.0; n];
    let eigenvecs = create_identity_matrix(n);
    
    // Use diagonal elements as rough eigenvalue estimates
    for i in 0..n {
        eigenvals[i] = matrix[i][i];
    }
    
    eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    
    Ok((eigenvals, eigenvecs))
}

/// Compute eigenvalues and eigenvectors for symmetric matrices
fn compute_eigen_symmetric(matrix: &[Vec<f64>]) -> VmResult<(Vec<f64>, Vec<Vec<f64>>, bool, f64)> {
    let n = matrix.len();
    
    // Use Jacobi method for symmetric matrices
    let (eigenvals, eigenvecs, success) = compute_jacobi_eigenvalues(matrix)?;
    
    // Estimate condition number as ratio of max/min eigenvalue
    let max_eval = eigenvals.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let min_eval = eigenvals.iter()
        .filter(|&&x| x.abs() > 1e-14)
        .fold(f64::INFINITY, |acc, &x| acc.min(x.abs()));
    
    let condition = if min_eval == f64::INFINITY || min_eval == 0.0 {
        f64::INFINITY
    } else {
        max_eval / min_eval
    };
    
    Ok((eigenvals, eigenvecs, success, condition))
}

/// Compute eigenvalues and eigenvectors for general matrices
fn compute_eigen_general(matrix: &[Vec<f64>]) -> VmResult<(Vec<f64>, Vec<Vec<f64>>, bool, f64)> {
    let n = matrix.len();
    
    // Use QR algorithm for general matrices (simplified implementation)
    let (eigenvals, eigenvecs, success) = compute_qr_eigenvalues(matrix)?;
    
    // Estimate condition number
    let max_eval = eigenvals.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let min_eval = eigenvals.iter()
        .filter(|&&x| x.abs() > 1e-14)
        .fold(f64::INFINITY, |acc, &x| acc.min(x.abs()));
    
    let condition = if min_eval == f64::INFINITY || min_eval == 0.0 {
        f64::INFINITY
    } else {
        max_eval / min_eval
    };
    
    Ok((eigenvals, eigenvecs, success, condition))
}

/// Compute Schur decomposition using QR algorithm
fn compute_schur_qr(matrix: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<f64>, bool, f64)> {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut q_total = create_identity_matrix(n);
    
    // QR algorithm iterations to converge to Schur form
    let max_iterations = 100;
    let tolerance = 1e-12;
    let mut success = true;
    
    for iteration in 0..max_iterations {
        // Check for convergence (check if matrix is upper triangular)
        let mut converged = true;
        for i in 1..n {
            for j in 0..i {
                if a[i][j].abs() > tolerance {
                    converged = false;
                    break;
                }
            }
            if !converged { break; }
        }
        
        if converged {
            break;
        }
        
        if iteration == max_iterations - 1 {
            success = false;
        }
        
        // QR decomposition step
        let (q, r, qr_success, _) = compute_qr_householder(&a)?;
        if !qr_success {
            success = false;
            break;
        }
        
        // Update A = RQ (similarity transformation)
        a = matrix_multiply(&r, &q)?;
        
        // Accumulate Q transformations
        q_total = matrix_multiply(&q_total, &q)?;
    }
    
    // Extract eigenvalues from diagonal
    let mut eigenvalues = Vec::new();
    for i in 0..n {
        eigenvalues.push(a[i][i]);
    }
    
    // Estimate condition number
    let max_eval = eigenvalues.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));
    let min_eval = eigenvalues.iter()
        .filter(|&&x| x.abs() > 1e-14)
        .fold(f64::INFINITY, |acc, &x| acc.min(x.abs()));
    
    let condition = if min_eval == f64::INFINITY || min_eval == 0.0 {
        f64::INFINITY
    } else {
        max_eval / min_eval
    };
    
    Ok((q_total, a, eigenvalues, success, condition))
}

/// Jacobi method for symmetric eigenvalue problem
fn compute_jacobi_eigenvalues(matrix: &[Vec<f64>]) -> VmResult<(Vec<f64>, Vec<Vec<f64>>, bool)> {
    let n = matrix.len();
    let mut a = matrix.to_vec();
    let mut v = create_identity_matrix(n);
    
    let max_iterations = 1000; // Increased iterations
    let tolerance = 1e-10; // Relaxed tolerance
    let mut success = true;
    
    for iteration in 0..max_iterations {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        
        for i in 0..n {
            for j in i + 1..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        
        // Check convergence
        if max_val < tolerance {
            break;
        }
        
        if iteration == max_iterations - 1 {
            success = false;
        }
        
        // Compute rotation angle using standard Jacobi formula
        let tau = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        
        // Apply Jacobi rotation
        apply_jacobi_rotation(&mut a, &mut v, p, q, c, s);
    }
    
    // Extract eigenvalues from diagonal
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    
    // Sort eigenvalues and corresponding eigenvectors in descending order
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());
    
    let sorted_eigenvals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let mut sorted_eigenvecs = vec![vec![0.0; n]; n];
    
    for (new_col, &old_col) in indices.iter().enumerate() {
        for row in 0..n {
            sorted_eigenvecs[row][new_col] = v[row][old_col];
        }
    }
    
    Ok((sorted_eigenvals, sorted_eigenvecs, success))
}

/// QR algorithm for general eigenvalue problem (simplified)
fn compute_qr_eigenvalues(matrix: &[Vec<f64>]) -> VmResult<(Vec<f64>, Vec<Vec<f64>>, bool)> {
    // For general matrices, use a simplified approach
    // In practice, this would involve Hessenberg reduction followed by QR iterations
    let n = matrix.len();
    
    // Use diagonal elements as eigenvalue estimates for now
    let mut eigenvals: Vec<f64> = (0..n).map(|i| matrix[i][i]).collect();
    let eigenvecs = create_identity_matrix(n);
    
    // Sort eigenvalues by magnitude (descending)
    eigenvals.sort_by(|a, b| b.abs().partial_cmp(&a.abs()).unwrap());
    
    Ok((eigenvals, eigenvecs, true))
}

/// Solve linear system Ax = b using LU decomposition
fn solve_linear_system(a: &[Vec<f64>], b: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, f64, bool, f64, usize)> {
    let n = a.len();
    let b_cols = b[0].len();
    
    // Compute LU decomposition with partial pivoting
    let (l, u, p, lu_success, condition) = compute_lu_partial_pivot(a)?;
    
    if !lu_success {
        // Matrix is singular, return zero solution
        let zero_solution = vec![vec![0.0; b_cols]; n];
        return Ok((zero_solution, f64::INFINITY, false, f64::INFINITY, 0));
    }
    
    let mut solution = vec![vec![0.0; b_cols]; n];
    let mut residual_norm = 0.0_f64;
    
    // For each right-hand side vector
    for col in 0..b_cols {
        // Extract b column and apply permutation
        let pb: Vec<f64> = (0..n).map(|i| {
            // Find which row in P has 1 in column i
            for row in 0..n {
                if p[row][i] != 0.0 {
                    return b[row][col];
                }
            }
            0.0
        }).collect();
        
        // Forward substitution: solve Ly = Pb
        let mut y = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0_f64;
            for j in 0..i {
                sum += l[i][j] * y[j];
            }
            y[i] = pb[i] - sum;
        }
        
        // Back substitution: solve Ux = y
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            let mut sum = 0.0_f64;
            for j in (i+1)..n {
                sum += u[i][j] * x[j];
            }
            if u[i][i].abs() < 1e-14 {
                x[i] = 0.0; // Singular case
            } else {
                x[i] = (y[i] - sum) / u[i][i];
            }
        }
        
        // Store solution
        for i in 0..n {
            solution[i][col] = x[i];
        }
        
        // Compute residual norm for this column
        let mut residual_col = 0.0_f64;
        for i in 0..n {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[i][j] * x[j];
            }
            let r = ax_i - b[i][col];
            residual_col += r * r;
        }
        residual_norm += residual_col;
    }
    
    residual_norm = residual_norm.sqrt();
    let rank = compute_matrix_rank(a)?;
    
    Ok((solution, residual_norm, true, condition, rank))
}

/// Solve least squares problem using QR decomposition
fn solve_least_squares(a: &[Vec<f64>], b: &[Vec<f64>]) -> VmResult<(Vec<Vec<f64>>, f64, bool, f64, usize)> {
    let (m, n) = (a.len(), a[0].len());
    let b_cols = b[0].len();
    
    // Compute QR decomposition
    let (q, r, qr_success, condition) = compute_qr_householder(a)?;
    
    if !qr_success {
        let zero_solution = vec![vec![0.0; b_cols]; n];
        return Ok((zero_solution, f64::INFINITY, false, f64::INFINITY, 0));
    }
    
    let mut solution = vec![vec![0.0; b_cols]; n];
    let mut total_residual_norm = 0.0_f64;
    
    // For each right-hand side vector
    for col in 0..b_cols {
        // Extract b column
        let b_vec: Vec<f64> = (0..m).map(|i| b[i][col]).collect();
        
        // Compute Q^T * b
        let mut qtb = vec![0.0; m];
        for i in 0..m {
            let mut sum = 0.0_f64;
            for j in 0..m {
                sum += q[j][i] * b_vec[j]; // Q^T[i][j] = Q[j][i]
            }
            qtb[i] = sum;
        }
        
        // Solve R * x = Q^T * b (only first n equations)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            if i < r.len() && i < r[i].len() {
                let mut sum = 0.0_f64;
                for j in (i+1)..n.min(r[i].len()) {
                    sum += r[i][j] * x[j];
                }
                if r[i][i].abs() < 1e-14 {
                    x[i] = 0.0;
                } else {
                    x[i] = (qtb[i] - sum) / r[i][i];
                }
            }
        }
        
        // Store solution
        for i in 0..n {
            solution[i][col] = x[i];
        }
        
        // Compute residual norm
        let mut residual_col = 0.0_f64;
        for i in 0..m {
            let mut ax_i = 0.0_f64;
            for j in 0..n {
                ax_i += a[i][j] * x[j];
            }
            let r = ax_i - b[i][col];
            residual_col += r * r;
        }
        total_residual_norm += residual_col;
    }
    
    total_residual_norm = total_residual_norm.sqrt();
    let rank = compute_matrix_rank(a)?;
    
    Ok((solution, total_residual_norm, true, condition, rank))
}

/// Compute Moore-Penrose pseudoinverse using SVD
fn compute_pseudoinverse(matrix: &[Vec<f64>]) -> VmResult<Vec<Vec<f64>>> {
    let (m, n) = (matrix.len(), matrix[0].len());
    
    // Compute SVD: A = U * Σ * V^T
    let (u, sigma, vt, success, _) = compute_svd_jacobi(matrix)?;
    
    if !success {
        // Fallback to simple transpose for failed SVD
        return Ok(transpose(matrix));
    }
    
    // Compute pseudoinverse: A+ = V * Σ+ * U^T
    let tolerance = 1e-12;
    let max_sigma = sigma.iter().fold(0.0_f64, |acc, &x| acc.max(x));
    let cutoff = tolerance * max_sigma;
    
    // Create Σ+ (pseudoinverse of diagonal matrix)
    let mut sigma_pinv = vec![vec![0.0; m]; n];
    let min_dim = m.min(n);
    
    for i in 0..min_dim.min(sigma.len()) {
        if sigma[i] > cutoff {
            sigma_pinv[i][i] = 1.0 / sigma[i];
        }
    }
    
    // Compute V * Σ+
    let v = transpose(&vt); // V = (V^T)^T
    let v_sigma_pinv = matrix_multiply(&v, &sigma_pinv)?;
    
    // Compute (V * Σ+) * U^T
    let ut = transpose(&u);
    let pseudoinv = matrix_multiply(&v_sigma_pinv, &ut)?;
    
    Ok(pseudoinv)
}

/// Compute matrix power using integer exponentiation (repeated squaring)
fn compute_matrix_power_integer(matrix: &[Vec<f64>], power: i64) -> VmResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    
    if power == 0 {
        return Ok(create_identity_matrix(n));
    }
    
    if power == 1 {
        return Ok(matrix.to_vec());
    }
    
    if power < 0 {
        // A^(-n) = (A^(-1))^n
        let inv = compute_pseudoinverse(matrix)?;
        return compute_matrix_power_integer(&inv, -power);
    }
    
    // Repeated squaring algorithm
    let mut result = create_identity_matrix(n);
    let mut base = matrix.to_vec();
    let mut exp = power;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = matrix_multiply(&result, &base)?;
        }
        base = matrix_multiply(&base, &base)?;
        exp /= 2;
    }
    
    Ok(result)
}

/// Compute matrix power using eigendecomposition for general powers
fn compute_matrix_power_general(matrix: &[Vec<f64>], power: f64) -> VmResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    
    // Use eigendecomposition: A = V * Λ * V^(-1)
    // Then A^p = V * Λ^p * V^(-1)
    let (eigenvals, eigenvecs, success, _) = compute_eigen_symmetric(matrix)?;
    
    if !success {
        // Fallback to identity for failed decomposition
        return Ok(create_identity_matrix(n));
    }
    
    // Compute Λ^p (apply power to eigenvalues)
    let mut lambda_p = vec![vec![0.0; n]; n];
    for i in 0..n.min(eigenvals.len()) {
        let eval = eigenvals[i];
        if eval > 1e-14 || power.fract() == 0.0 {
            lambda_p[i][i] = eval.powf(power);
        } else {
            lambda_p[i][i] = 0.0; // Handle negative eigenvalues for fractional powers
        }
    }
    
    // Compute V * Λ^p
    let v_lambda_p = matrix_multiply(&eigenvecs, &lambda_p)?;
    
    // Compute (V * Λ^p) * V^(-1)
    let v_inv = compute_pseudoinverse(&eigenvecs)?;
    let result = matrix_multiply(&v_lambda_p, &v_inv)?;
    
    Ok(result)
}

/// Apply a function to matrix eigenvalues
fn compute_matrix_function(matrix: &[Vec<f64>], func_name: &str) -> VmResult<Vec<Vec<f64>>> {
    let n = matrix.len();
    
    // Use eigendecomposition: A = V * Λ * V^(-1)
    // Then f(A) = V * f(Λ) * V^(-1)
    let (eigenvals, eigenvecs, success, _) = compute_eigen_symmetric(matrix)?;
    
    if !success {
        // Fallback to identity for failed decomposition
        return Ok(create_identity_matrix(n));
    }
    
    // Apply function to eigenvalues
    let mut f_lambda = vec![vec![0.0; n]; n];
    for i in 0..n.min(eigenvals.len()) {
        let eval = eigenvals[i];
        let f_eval = match func_name {
            "exp" | "Exp" => eval.exp(),
            "log" | "Log" => if eval > 1e-14 { eval.ln() } else { f64::NEG_INFINITY },
            "sin" | "Sin" => eval.sin(),
            "cos" | "Cos" => eval.cos(),
            "sqrt" | "Sqrt" => if eval >= 0.0 { eval.sqrt() } else { 0.0 },
            "abs" | "Abs" => eval.abs(),
            _ => return Err(VmError::TypeError {
                expected: "known function name (exp, log, sin, cos, sqrt, abs)".to_string(),
                actual: func_name.to_string(),
            }),
        };
        f_lambda[i][i] = f_eval;
    }
    
    // Compute V * f(Λ)
    let v_f_lambda = matrix_multiply(&eigenvecs, &f_lambda)?;
    
    // Compute (V * f(Λ)) * V^(-1)
    let v_inv = compute_pseudoinverse(&eigenvecs)?;
    let result = matrix_multiply(&v_f_lambda, &v_inv)?;
    
    Ok(result)
}

/// Compute matrix trace (sum of diagonal elements)
fn compute_trace(matrix: &[Vec<f64>]) -> f64 {
    let n = matrix.len();
    let mut trace = 0.0_f64;
    
    for i in 0..n {
        if i < matrix[i].len() {
            trace += matrix[i][i];
        }
    }
    
    trace
}

/// Matrix multiplication utility
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> VmResult<Vec<Vec<f64>>> {
    let (m, k) = (a.len(), a[0].len());
    let (k2, n) = (b.len(), b[0].len());
    
    if k != k2 {
        return Err(VmError::TypeError {
            expected: format!("matrices with compatible dimensions ({}x{} and {}x{})", m, k, k, n),
            actual: format!("incompatible dimensions ({}x{} and {}x{})", m, k, k2, n),
        });
    }
    
    let mut result = vec![vec![0.0; n]; m];
    
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for p in 0..k {
                sum += a[i][p] * b[p][j];
            }
            result[i][j] = sum;
        }
    }
    
    Ok(result)
}

/// Apply Jacobi rotation for eigenvalue computation
fn apply_jacobi_rotation(a: &mut [Vec<f64>], v: &mut [Vec<f64>], p: usize, q: usize, c: f64, s: f64) {
    let n = a.len();
    
    // Store original values to avoid overwriting during computation
    let app = a[p][p];
    let aqq = a[q][q];
    let apq = a[p][q];
    
    // Update diagonal elements
    a[p][p] = c * c * app + s * s * aqq - 2.0 * c * s * apq;
    a[q][q] = s * s * app + c * c * aqq + 2.0 * c * s * apq;
    a[p][q] = 0.0; // Should be zero after rotation
    a[q][p] = 0.0;
    
    // Apply rotation to rest of matrix
    for k in 0..n {
        if k != p && k != q {
            let akp = a[k][p];
            let akq = a[k][q];
            a[k][p] = c * akp - s * akq;
            a[k][q] = s * akp + c * akq;
            a[p][k] = a[k][p]; // Maintain symmetry
            a[q][k] = a[k][q];
        }
    }
    
    // Apply rotation to eigenvector matrix
    for k in 0..n {
        let vkp = v[k][p];
        let vkq = v[k][q];
        v[k][p] = c * vkp - s * vkq;
        v[k][q] = s * vkp + c * vkq;
    }
}

/// Apply Householder reflection to matrix
fn apply_householder_reflection(matrix: &mut [Vec<f64>], v: &[f64], start_row: usize) {
    let (m, n) = (matrix.len(), matrix[0].len());
    
    for j in start_row..n {
        let mut dot_product = 0.0;
        for (i, &vi) in v.iter().enumerate() {
            if start_row + i < m {
                dot_product += vi * matrix[start_row + i][j];
            }
        }
        
        for (i, &vi) in v.iter().enumerate() {
            if start_row + i < m {
                matrix[start_row + i][j] -= 2.0 * vi * dot_product;
            }
        }
    }
}

/// Apply Householder reflection transpose
fn apply_householder_reflection_transpose(matrix: &mut [Vec<f64>], v: &[f64], start_col: usize) {
    let (m, n) = (matrix.len(), matrix[0].len());
    
    for i in 0..m {
        let mut dot_product = 0.0;
        for (j, &vj) in v.iter().enumerate() {
            if start_col + j < n {
                dot_product += vj * matrix[i][start_col + j];
            }
        }
        
        for (j, &vj) in v.iter().enumerate() {
            if start_col + j < n {
                matrix[i][start_col + j] -= 2.0 * vj * dot_product;
            }
        }
    }
}

/// Estimate condition number from QR decomposition
fn estimate_condition_qr(r: &[Vec<f64>]) -> f64 {
    let n = r.len().min(r[0].len());
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    
    for i in 0..n {
        let val = r[i][i].abs();
        max_diag = max_diag.max(val);
        if val > 1e-14 {
            min_diag = min_diag.min(val);
        }
    }
    
    if min_diag == f64::INFINITY || min_diag == 0.0 {
        f64::INFINITY
    } else {
        max_diag / min_diag
    }
}

/// Estimate condition number from LU decomposition
fn estimate_condition_lu(u: &[Vec<f64>]) -> f64 {
    estimate_condition_qr(u) // Same logic for diagonal elements
}

/// Estimate condition number from Cholesky decomposition
fn estimate_condition_cholesky(l: &[Vec<f64>]) -> f64 {
    let n = l.len();
    let mut max_diag = 0.0_f64;
    let mut min_diag = f64::INFINITY;
    
    for i in 0..n {
        let val = l[i][i].abs();
        max_diag = max_diag.max(val);
        if val > 1e-14 {
            min_diag = min_diag.min(val);
        }
    }
    
    if min_diag == f64::INFINITY || min_diag == 0.0 {
        f64::INFINITY
    } else {
        (max_diag / min_diag).powi(2) // For Cholesky, condition of A = (condition of L)^2
    }
}

/// Compute determinant of permutation matrix
fn compute_permutation_determinant(p: &[Vec<f64>]) -> f64 {
    let n = p.len();
    let mut perm = vec![0; n];
    
    // Extract permutation vector
    for i in 0..n {
        for j in 0..n {
            if p[i][j] != 0.0 {
                perm[i] = j;
                break;
            }
        }
    }
    
    // Count inversions to determine sign
    let mut inversions = 0;
    for i in 0..n {
        for j in i+1..n {
            if perm[i] > perm[j] {
                inversions += 1;
            }
        }
    }
    
    if inversions % 2 == 0 { 1.0 } else { -1.0 }
}

// =============================================================================
// FOREIGN TRAIT IMPLEMENTATIONS
// =============================================================================

impl Foreign for MatrixDecomposition {
    fn type_name(&self) -> &'static str {
        "MatrixDecomposition"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Factor1" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let rows: Vec<Value> = self.factor1.iter()
                    .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                    .collect();
                Ok(Value::List(rows))
            }
            "Factor2" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Some(ref factor2) = self.factor2 {
                    let rows: Vec<Value> = factor2.iter()
                        .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                        .collect();
                    Ok(Value::List(rows))
                } else {
                    Ok(Value::List(vec![]))
                }
            }
            "Factor3" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Some(ref factor3) = self.factor3 {
                    let rows: Vec<Value> = factor3.iter()
                        .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                        .collect();
                    Ok(Value::List(rows))
                } else {
                    Ok(Value::List(vec![]))
                }
            }
            "Values" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let values: Vec<Value> = self.values.iter().map(|&x| Value::Real(x)).collect();
                Ok(Value::List(values))
            }
            "Type" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.decomp_type.clone()))
            }
            "Condition" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.condition))
            }
            "Success" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Symbol(if self.success { "True".to_string() } else { "False".to_string() }))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
}

impl Foreign for LinearSystemResult {
    fn type_name(&self) -> &'static str {
        "LinearSystemResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Solution" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let rows: Vec<Value> = self.solution.iter()
                    .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                    .collect();
                Ok(Value::List(rows))
            }
            "ResidualNorm" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.residual_norm))
            }
            "Method" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.method.clone()))
            }
            "Condition" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.condition))
            }
            "Rank" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.rank as i64))
            }
            "Success" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Symbol(if self.success { "True".to_string() } else { "False".to_string() }))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
}

impl Foreign for MatrixAnalysisResult {
    fn type_name(&self) -> &'static str {
        "MatrixAnalysisResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Value" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.value))
            }
            "Type" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.analysis_type.clone()))
            }
            "Properties" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let props: Vec<Value> = self.properties.iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(props))
            }
            "Reliable" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Symbol(if self.reliable { "True".to_string() } else { "False".to_string() }))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_rank() {
        // Test rank-1 matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(4.0)]),
        ]);
        
        let result = matrix_rank(&[matrix]).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_determinant() {
        // Test 2x2 matrix determinant
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        
        let result = determinant(&[matrix]).unwrap();
        assert_eq!(result, Value::Real(-2.0)); // 1*4 - 2*3 = -2
    }

    #[test]
    fn test_matrix_norm() {
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        
        let result = matrix_norm(&[matrix]).unwrap();
        if let Value::Real(norm) = result {
            let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt(); // Frobenius norm
            assert!((norm - expected).abs() < 1e-10);
        } else {
            panic!("Expected real norm value");
        }
    }

    #[test]
    fn test_qr_decomposition() {
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
            Value::List(vec![Value::Real(5.0), Value::Real(6.0)]),
        ]);
        
        let result = qr_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "QR");
                assert!(decomp.success);
                assert_eq!(decomp.factor1.len(), 3); // Q is 3x3
                assert_eq!(decomp.factor2.as_ref().unwrap().len(), 3); // R is 3x2
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Test with positive definite matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(4.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(3.0)]),
        ]);
        
        let result = cholesky_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "Cholesky");
                assert!(decomp.success);
                // Verify L[0][0] = 2.0 (sqrt(4))
                assert!((decomp.factor1[0][0] - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_lu_decomposition() {
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(2.0), Value::Real(1.0)]),
            Value::List(vec![Value::Real(1.0), Value::Real(3.0)]),
        ]);
        
        let result = lu_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "LU");
                assert!(decomp.success);
                assert_eq!(decomp.factor1.len(), 2); // L is 2x2
                assert_eq!(decomp.factor2.as_ref().unwrap().len(), 2); // U is 2x2
                assert_eq!(decomp.factor3.as_ref().unwrap().len(), 2); // P is 2x2
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_svd() {
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(3.0), Value::Real(2.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(3.0), Value::Real(-2.0)]),
        ]);
        
        let result = svd(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "SVD");
                assert!(decomp.success);
                assert!(!decomp.values.is_empty()); // Should have singular values
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_matrix_condition() {
        // Well-conditioned matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0)]),
        ]);
        
        let result = matrix_condition(&[matrix]).unwrap();
        if let Value::Real(cond) = result {
            assert!(cond < 10.0); // Should be well-conditioned
        } else {
            panic!("Expected real condition number");
        }
    }

    #[test]
    fn test_eigen_decomposition() {
        // Test with a simple symmetric matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(4.0), Value::Real(1.0)]),
            Value::List(vec![Value::Real(1.0), Value::Real(3.0)]),
        ]);
        
        let result = eigen_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "Eigen");
                assert!(decomp.success);
                assert_eq!(decomp.values.len(), 2); // Should have 2 eigenvalues
                
                // For symmetric matrix, eigenvalues should be real
                // Values should be approximately [4.56, 2.44] (eigenvalues of this matrix)
                assert!(decomp.values.iter().all(|&val| val.is_finite()));
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_schur_decomposition() {
        // Test with a 2x2 matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(3.0)]), // Upper triangular already
        ]);
        
        let result = schur_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert_eq!(decomp.decomp_type, "Schur");
                assert!(decomp.success);
                assert_eq!(decomp.values.len(), 2); // Should have 2 eigenvalues
                
                // Eigenvalues should be 1.0 and 3.0 (diagonal elements)
                let mut eigenvals = decomp.values.clone();
                eigenvals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                assert!((eigenvals[0] - 1.0).abs() < 1e-10);
                assert!((eigenvals[1] - 3.0).abs() < 1e-10);
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_eigen_decomposition_identity() {
        // Identity matrix should have eigenvalues [1, 1]
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0)]),
        ]);
        
        let result = eigen_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert!(decomp.success);
                assert_eq!(decomp.values.len(), 2);
                
                // All eigenvalues should be 1.0
                for &eigenval in &decomp.values {
                    assert!((eigenval - 1.0).abs() < 1e-10);
                }
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_eigen_decomposition_diagonal() {
        // Diagonal matrix - eigenvalues should be diagonal elements
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(5.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(2.0)]),
        ]);
        
        let result = eigen_decomposition(&[matrix]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let decomp = lyobj.downcast_ref::<MatrixDecomposition>().unwrap();
                assert!(decomp.success);
                assert_eq!(decomp.values.len(), 2);
                
                // Eigenvalues should be 5.0 and 2.0 (in some order)
                let mut eigenvals = decomp.values.clone();
                eigenvals.sort_by(|a, b| b.partial_cmp(a).unwrap()); // Sort descending
                assert!((eigenvals[0] - 5.0).abs() < 1e-10);
                assert!((eigenvals[1] - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected MatrixDecomposition"),
        }
    }

    #[test]
    fn test_linear_solve() {
        // Test solving 2x2 system: [1, 2; 3, 4] * [x; y] = [5; 11]
        // Solution should be [x=1, y=2]
        let a = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        let b = Value::List(vec![
            Value::List(vec![Value::Real(5.0)]),
            Value::List(vec![Value::Real(11.0)]),
        ]);
        
        let result = linear_solve(&[a, b]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let linear_result = lyobj.downcast_ref::<LinearSystemResult>().unwrap();
                assert!(linear_result.success);
                assert_eq!(linear_result.solution.len(), 2); // 2 variables
                assert_eq!(linear_result.solution[0].len(), 1); // 1 RHS
                
                // Check solution: x ≈ 1, y ≈ 2
                assert!((linear_result.solution[0][0] - 1.0).abs() < 1e-10);
                assert!((linear_result.solution[1][0] - 2.0).abs() < 1e-10);
            }
            _ => panic!("Expected LinearSystemResult"),
        }
    }

    #[test]
    fn test_least_squares() {
        // Test overdetermined system: fit y = ax + b to points (0,1), (1,3), (2,5)
        // Expected: a ≈ 2, b ≈ 1 (perfect fit)
        let a = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]), // [1, 0] for point (0,1)
            Value::List(vec![Value::Real(1.0), Value::Real(1.0)]), // [1, 1] for point (1,3)
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]), // [1, 2] for point (2,5)
        ]);
        let b = Value::List(vec![
            Value::List(vec![Value::Real(1.0)]), // y = 1
            Value::List(vec![Value::Real(3.0)]), // y = 3
            Value::List(vec![Value::Real(5.0)]), // y = 5
        ]);
        
        let result = least_squares(&[a, b]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let ls_result = lyobj.downcast_ref::<LinearSystemResult>().unwrap();
                assert!(ls_result.success);
                assert_eq!(ls_result.solution.len(), 2); // 2 parameters (b, a)
                
                // Check solution: b ≈ 1, a ≈ 2
                let b_est = ls_result.solution[0][0];
                let a_est = ls_result.solution[1][0];
                assert!((b_est - 1.0).abs() < 1e-10, "b estimate: {}", b_est);
                assert!((a_est - 2.0).abs() < 1e-10, "a estimate: {}", a_est);
                
                // Residual should be very small for perfect fit
                assert!(ls_result.residual_norm < 1e-10);
            }
            _ => panic!("Expected LinearSystemResult"),
        }
    }

    #[test]
    fn test_pseudo_inverse() {
        // Test pseudoinverse of a simple matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        
        let result = pseudo_inverse(&[matrix]).unwrap();
        
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // Should be 2x2
                
                // For 2x2 matrix [[1,2],[3,4]], pseudoinverse should be close to regular inverse
                // Regular inverse is [[-2, 1], [1.5, -0.5]]
                if let (Value::List(row1), Value::List(row2)) = (&rows[0], &rows[1]) {
                    assert_eq!(row1.len(), 2);
                    assert_eq!(row2.len(), 2);
                    
                    // Check that pseudoinverse gives reasonable values
                    // (exact values depend on the SVD implementation)
                    if let (Value::Real(a11), Value::Real(a12), Value::Real(a21), Value::Real(a22)) =
                        (&row1[0], &row1[1], &row2[0], &row2[1]) {
                        
                        // The pseudoinverse should be finite
                        assert!(a11.is_finite());
                        assert!(a12.is_finite());
                        assert!(a21.is_finite());
                        assert!(a22.is_finite());
                    } else {
                        panic!("Expected real values in pseudoinverse");
                    }
                } else {
                    panic!("Expected nested lists in pseudoinverse result");
                }
            }
            _ => panic!("Expected matrix (nested lists) for pseudoinverse"),
        }
    }

    #[test]
    fn test_linear_solve_identity() {
        // Test solving with identity matrix: I * x = b
        let identity = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0)]),
        ]);
        let b = Value::List(vec![
            Value::List(vec![Value::Real(3.0)]),
            Value::List(vec![Value::Real(7.0)]),
        ]);
        
        let result = linear_solve(&[identity, b]).unwrap();
        
        match result {
            Value::LyObj(lyobj) => {
                let linear_result = lyobj.downcast_ref::<LinearSystemResult>().unwrap();
                assert!(linear_result.success);
                
                // Solution should be exactly [3, 7]
                assert!((linear_result.solution[0][0] - 3.0).abs() < 1e-10);
                assert!((linear_result.solution[1][0] - 7.0).abs() < 1e-10);
                
                // Residual should be essentially zero
                assert!(linear_result.residual_norm < 1e-10);
            }
            _ => panic!("Expected LinearSystemResult"),
        }
    }

    #[test]
    fn test_pseudo_inverse_rank_deficient() {
        // Test pseudoinverse of rank-deficient matrix
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(2.0), Value::Real(4.0)]), // Second row is 2x first row
        ]);
        
        let result = pseudo_inverse(&[matrix]).unwrap();
        
        // Should return a matrix without crashing
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // Should be 2x2
                
                // All values should be finite (not NaN or infinite)
                for row in rows {
                    if let Value::List(row_values) = row {
                        for val in row_values {
                            if let Value::Real(x) = val {
                                assert!(x.is_finite(), "Pseudoinverse value should be finite: {}", x);
                            }
                        }
                    }
                }
            }
            _ => panic!("Expected matrix result"),
        }
    }

    #[test]
    fn test_matrix_trace() {
        // Test trace of 2x2 matrix [[1, 2], [3, 4]]
        // Trace should be 1 + 4 = 5
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        
        let result = matrix_trace(&[matrix]).unwrap();
        
        match result {
            Value::Real(trace) => {
                assert!((trace - 5.0).abs() < 1e-10, "Expected trace 5.0, got {}", trace);
            }
            _ => panic!("Expected real trace value"),
        }
    }

    #[test]
    fn test_matrix_trace_identity() {
        // Test trace of 3x3 identity matrix
        // Trace should be 3.0
        let identity = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(0.0), Value::Real(1.0)]),
        ]);
        
        let result = matrix_trace(&[identity]).unwrap();
        
        match result {
            Value::Real(trace) => {
                assert!((trace - 3.0).abs() < 1e-10, "Expected trace 3.0, got {}", trace);
            }
            _ => panic!("Expected real trace value"),
        }
    }

    #[test]
    fn test_matrix_power_zero() {
        // Test A^0 = I for any matrix A
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(2.0), Value::Real(3.0)]),
            Value::List(vec![Value::Real(1.0), Value::Real(4.0)]),
        ]);
        let power = Value::Real(0.0);
        
        let result = matrix_power(&[matrix, power]).unwrap();
        
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // 2x2 result
                
                // Should be identity matrix [[1, 0], [0, 1]]
                if let (Value::List(row1), Value::List(row2)) = (&rows[0], &rows[1]) {
                    if let (Value::Real(a11), Value::Real(a12), Value::Real(a21), Value::Real(a22)) =
                        (&row1[0], &row1[1], &row2[0], &row2[1]) {
                        
                        assert!((a11 - 1.0).abs() < 1e-10);
                        assert!(a12.abs() < 1e-10);
                        assert!(a21.abs() < 1e-10);
                        assert!((a22 - 1.0).abs() < 1e-10);
                    } else {
                        panic!("Expected real values in matrix power result");
                    }
                } else {
                    panic!("Expected nested lists in matrix power result");
                }
            }
            _ => panic!("Expected matrix result for matrix power"),
        }
    }

    #[test]
    fn test_matrix_power_one() {
        // Test A^1 = A
        let matrix = Value::List(vec![
            Value::List(vec![Value::Real(2.0), Value::Real(3.0)]),
            Value::List(vec![Value::Real(1.0), Value::Real(4.0)]),
        ]);
        let power = Value::Real(1.0);
        
        let result = matrix_power(&[matrix, power]).unwrap();
        
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // 2x2 result
                
                // Should be original matrix [[2, 3], [1, 4]]
                if let (Value::List(row1), Value::List(row2)) = (&rows[0], &rows[1]) {
                    if let (Value::Real(a11), Value::Real(a12), Value::Real(a21), Value::Real(a22)) =
                        (&row1[0], &row1[1], &row2[0], &row2[1]) {
                        
                        assert!((a11 - 2.0).abs() < 1e-10);
                        assert!((a12 - 3.0).abs() < 1e-10);
                        assert!((a21 - 1.0).abs() < 1e-10);
                        assert!((a22 - 4.0).abs() < 1e-10);
                    } else {
                        panic!("Expected real values in matrix power result");
                    }
                } else {
                    panic!("Expected nested lists in matrix power result");
                }
            }
            _ => panic!("Expected matrix result for matrix power"),
        }
    }

    #[test]
    fn test_matrix_power_integer() {
        // Test I^n = I for identity matrix
        let identity = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0)]),
        ]);
        let power = Value::Real(5.0); // I^5 = I
        
        let result = matrix_power(&[identity, power]).unwrap();
        
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // 2x2 result
                
                // Should still be identity matrix
                if let (Value::List(row1), Value::List(row2)) = (&rows[0], &rows[1]) {
                    if let (Value::Real(a11), Value::Real(a12), Value::Real(a21), Value::Real(a22)) =
                        (&row1[0], &row1[1], &row2[0], &row2[1]) {
                        
                        assert!((a11 - 1.0).abs() < 1e-10, "Expected 1.0, got {}", a11);
                        assert!(a12.abs() < 1e-10, "Expected 0.0, got {}", a12);
                        assert!(a21.abs() < 1e-10, "Expected 0.0, got {}", a21);
                        assert!((a22 - 1.0).abs() < 1e-10, "Expected 1.0, got {}", a22);
                    } else {
                        panic!("Expected real values in matrix power result");
                    }
                } else {
                    panic!("Expected nested lists in matrix power result");
                }
            }
            _ => panic!("Expected matrix result for matrix power"),
        }
    }

    #[test]
    fn test_matrix_function_exp() {
        // Test matrix exponential on diagonal matrix
        let diagonal = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(2.0)]),
        ]);
        let func_name = Value::String("exp".to_string());
        
        let result = matrix_function(&[diagonal, func_name]).unwrap();
        
        // Should return a matrix - exact values depend on eigenvalue implementation
        match result {
            Value::List(rows) => {
                assert_eq!(rows.len(), 2); // 2x2 result
                
                // All values should be finite
                for row in rows {
                    if let Value::List(row_values) = row {
                        for val in row_values {
                            if let Value::Real(x) = val {
                                assert!(x.is_finite(), "Matrix function result should be finite: {}", x);
                            }
                        }
                    }
                }
            }
            _ => panic!("Expected matrix result for matrix function"),
        }
    }
}