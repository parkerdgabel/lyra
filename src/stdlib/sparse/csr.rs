//! Compressed Sparse Row (CSR) Matrix Format
//!
//! CSR format stores a sparse matrix using three arrays:
//! - `data`: Non-zero values
//! - `indices`: Column indices for each non-zero value
//! - `indptr`: Row pointers indicating where each row starts in data/indices
//!
//! This format is highly efficient for:
//! - Matrix-vector multiplication
//! - Row-wise operations
//! - Iteration over rows

use super::core::{SparseMatrix, SparseFormat, GenericSparseMatrix};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Compressed Sparse Row matrix implementation
#[derive(Debug, Clone)]
pub struct CSRMatrix {
    /// Non-zero values in row-major order
    pub data: Vec<f64>,
    /// Column indices for each non-zero value
    pub indices: Vec<usize>,
    /// Row pointers (length = rows + 1)
    pub indptr: Vec<usize>,
    /// Matrix dimensions
    pub shape: (usize, usize),
}

impl CSRMatrix {
    /// Create new CSR matrix from arrays
    pub fn new(
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
        shape: (usize, usize),
    ) -> Result<Self, String> {
        // Validate inputs
        if data.len() != indices.len() {
            return Err("data and indices must have same length".to_string());
        }
        
        if indptr.len() != shape.0 + 1 {
            return Err("indptr length must be rows + 1".to_string());
        }
        
        if let Some(&max_col) = indices.iter().max() {
            if max_col >= shape.1 {
                return Err("column index exceeds matrix dimensions".to_string());
            }
        }
        
        if *indptr.last().unwrap() != data.len() {
            return Err("indptr[-1] must equal data length".to_string());
        }
        
        // Check indptr is non-decreasing
        for i in 1..indptr.len() {
            if indptr[i] < indptr[i-1] {
                return Err("indptr must be non-decreasing".to_string());
            }
        }
        
        Ok(CSRMatrix {
            data,
            indices,
            indptr,
            shape,
        })
    }
    
    /// Create CSR matrix from triplets (row, col, value)
    pub fn from_triplets(
        triplets: Vec<(usize, usize, f64)>,
        shape: (usize, usize),
    ) -> Self {
        let mut row_data: Vec<Vec<(usize, f64)>> = vec![Vec::new(); shape.0];
        
        // Group by row and sort by column
        for (row, col, val) in triplets {
            if val != 0.0 && row < shape.0 && col < shape.1 {
                row_data[row].push((col, val));
            }
        }
        
        // Sort each row by column index
        for row in &mut row_data {
            row.sort_by_key(|&(col, _)| col);
        }
        
        // Build CSR arrays
        let mut data = Vec::new();
        let mut indices = Vec::new();
        let mut indptr = vec![0];
        
        for row in row_data {
            for (col, val) in row {
                data.push(val);
                indices.push(col);
            }
            indptr.push(data.len());
        }
        
        CSRMatrix {
            data,
            indices,
            indptr,
            shape,
        }
    }
    
    /// Create CSR matrix from dense matrix
    pub fn from_dense(matrix: &[Vec<f64>]) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };
        
        let mut triplets = Vec::new();
        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if val != 0.0 {
                    triplets.push((i, j, val));
                }
            }
        }
        
        Self::from_triplets(triplets, (rows, cols))
    }
    
    /// Get row slice (column indices and values)
    pub fn row(&self, row: usize) -> Option<(&[usize], &[f64])> {
        if row >= self.shape.0 {
            return None;
        }
        
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        
        Some((&self.indices[start..end], &self.data[start..end]))
    }
    
    /// Get element at (row, col)
    pub fn get_element(&self, row: usize, col: usize) -> f64 {
        if row >= self.shape.0 || col >= self.shape.1 {
            return 0.0;
        }
        
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        
        // Binary search for column index
        match self.indices[start..end].binary_search(&col) {
            Ok(pos) => self.data[start + pos],
            Err(_) => 0.0,
        }
    }
    
    /// Set element at (row, col) - Note: This is inefficient for CSR
    pub fn set_element(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.shape.0 || col >= self.shape.1 {
            return;
        }
        
        let start = self.indptr[row];
        let end = self.indptr[row + 1];
        
        match self.indices[start..end].binary_search(&col) {
            Ok(pos) => {
                if value == 0.0 {
                    // Remove element
                    self.data.remove(start + pos);
                    self.indices.remove(start + pos);
                    // Update indptr
                    for i in (row + 1)..self.indptr.len() {
                        self.indptr[i] -= 1;
                    }
                } else {
                    // Update existing element
                    self.data[start + pos] = value;
                }
            }
            Err(pos) => {
                if value != 0.0 {
                    // Insert new element
                    self.data.insert(start + pos, value);
                    self.indices.insert(start + pos, col);
                    // Update indptr
                    for i in (row + 1)..self.indptr.len() {
                        self.indptr[i] += 1;
                    }
                }
            }
        }
    }
    
    /// Matrix-vector multiplication
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.shape.1 {
            panic!("Vector length must match matrix columns");
        }
        
        let mut result = vec![0.0; self.shape.0];
        
        for i in 0..self.shape.0 {
            let start = self.indptr[i];
            let end = self.indptr[i + 1];
            
            for k in start..end {
                result[i] += self.data[k] * x[self.indices[k]];
            }
        }
        
        result
    }
    
    /// Transpose to CSC format
    pub fn transpose(&self) -> super::csc::CSCMatrix {
        // Create CSC from this CSR's data by transposing (row-ordered becomes column-ordered)
        let triplets: Vec<(usize, usize, f64)> = self.triplets()
            .into_iter()
            .map(|(r, c, v)| (c, r, v))
            .collect();
        
        super::csc::CSCMatrix::from_triplets(triplets, (self.shape.1, self.shape.0))
    }
    
    /// Eliminate zeros from the matrix
    pub fn eliminate_zeros(&mut self) {
        let mut new_data = Vec::new();
        let mut new_indices = Vec::new();
        let mut new_indptr = vec![0];
        
        for row in 0..self.shape.0 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for k in start..end {
                if self.data[k] != 0.0 {
                    new_data.push(self.data[k]);
                    new_indices.push(self.indices[k]);
                }
            }
            new_indptr.push(new_data.len());
        }
        
        self.data = new_data;
        self.indices = new_indices;
        self.indptr = new_indptr;
    }
    
    /// Sort indices within each row
    pub fn sort_indices(&mut self) {
        for row in 0..self.shape.0 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            if end > start {
                let mut row_data: Vec<(usize, f64)> = (start..end)
                    .map(|k| (self.indices[k], self.data[k]))
                    .collect();
                
                row_data.sort_by_key(|&(col, _)| col);
                
                for (i, (col, val)) in row_data.into_iter().enumerate() {
                    self.indices[start + i] = col;
                    self.data[start + i] = val;
                }
            }
        }
    }
}

impl SparseMatrix for CSRMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::CSR
    }
    
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    fn nnz(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, row: usize, col: usize) -> f64 {
        self.get_element(row, col)
    }
    
    fn set(&mut self, row: usize, col: usize, value: f64) {
        self.set_element(row, col, value)
    }
    
    fn triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::with_capacity(self.data.len());
        
        for row in 0..self.shape.0 {
            let start = self.indptr[row];
            let end = self.indptr[row + 1];
            
            for k in start..end {
                triplets.push((row, self.indices[k], self.data[k]));
            }
        }
        
        triplets
    }
    
    fn convert_to(&self, format: SparseFormat) -> Box<dyn SparseMatrix> {
        match format {
            SparseFormat::CSR => Box::new(self.clone()),
            SparseFormat::CSC => Box::new(self.transpose()),
            SparseFormat::COO => {
                let triplets = self.triplets();
                Box::new(super::coo::COOMatrix::from_triplets(triplets, self.shape))
            }
            SparseFormat::DOK => {
                let triplets = self.triplets();
                Box::new(super::dok::DOKMatrix::from_triplets(triplets, self.shape))
            }
            SparseFormat::LIL => {
                let triplets = self.triplets();
                Box::new(super::lil::LILMatrix::from_triplets(triplets, self.shape))
            }
        }
    }
}

impl Foreign for CSRMatrix {
    fn type_name(&self) -> &'static str {
        "CSRMatrix"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Data" => {
                let data: Vec<Value> = self.data.iter().map(|&v| Value::Real(v)).collect();
                Ok(Value::List(data))
            }
            "Indices" => {
                let indices: Vec<Value> = self.indices.iter().map(|&i| Value::Integer(i as i64)).collect();
                Ok(Value::List(indices))
            }
            "Indptr" => {
                let indptr: Vec<Value> = self.indptr.iter().map(|&i| Value::Integer(i as i64)).collect();
                Ok(Value::List(indptr))
            }
            "MatVec" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let vector = match &args[0] {
                    Value::List(list) => {
                        let mut vec = Vec::new();
                        for item in list {
                            match item {
                                Value::Real(r) => vec.push(*r),
                                Value::Integer(i) => vec.push(*i as f64),
                                _ => return Err(ForeignError::InvalidArgumentType {
                                    method: method.to_string(),
                                    expected: "numeric vector".to_string(),
                                    actual: format!("{:?}", item),
                                }),
                            }
                        }
                        vec
                    }
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let result = self.matvec(&vector);
                let result_values: Vec<Value> = result.iter().map(|&v| Value::Real(v)).collect();
                Ok(Value::List(result_values))
            }
            "Row" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let row = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.row(row) {
                    Some((cols, vals)) => {
                        let col_vals: Vec<Value> = cols.iter().zip(vals.iter())
                            .map(|(&c, &v)| Value::List(vec![Value::Integer(c as i64), Value::Real(v)]))
                            .collect();
                        Ok(Value::List(col_vals))
                    }
                    None => Ok(Value::List(vec![])),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// ===============================
// CSR MATRIX CONSTRUCTOR FUNCTION
// ===============================

/// Create CSR matrix from data, indices, indptr arrays
/// Syntax: CSRMatrix[data, indices, indptr, shape]
pub fn csr_matrix(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (data, indices, indptr, shape)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Extract data array
    let data = match &args[0] {
        Value::List(list) => {
            let mut data_vec = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => data_vec.push(*r),
                    Value::Integer(i) => data_vec.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric list for data".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            data_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List for data".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Extract indices array
    let indices = match &args[1] {
        Value::List(list) => {
            let mut indices_vec = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => indices_vec.push(*i as usize),
                    _ => return Err(VmError::TypeError {
                        expected: "integer list for indices".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            indices_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List for indices".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // Extract indptr array
    let indptr = match &args[2] {
        Value::List(list) => {
            let mut indptr_vec = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => indptr_vec.push(*i as usize),
                    _ => return Err(VmError::TypeError {
                        expected: "integer list for indptr".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            indptr_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List for indptr".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    // Extract shape
    let shape = match &args[3] {
        Value::List(list) if list.len() == 2 => {
            let rows = match &list[0] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "integer for rows".to_string(),
                    actual: format!("{:?}", list[0]),
                }),
            };
            let cols = match &list[1] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "integer for cols".to_string(),
                    actual: format!("{:?}", list[1]),
                }),
            };
            (rows, cols)
        }
        _ => return Err(VmError::TypeError {
            expected: "List of 2 integers for shape".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    match CSRMatrix::new(data, indices, indptr, shape) {
        Ok(matrix) => {
            let sparse = GenericSparseMatrix::new(Box::new(matrix));
            Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
        }
        Err(e) => Err(VmError::Runtime(format!("CSR matrix creation failed: {}", e))),
    }
}

/// Create CSR matrix from triplets
/// Syntax: CSRFromTriplets[triplets, shape]
pub fn csr_from_triplets(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (triplets, shape)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let triplets = super::core::extract_triplets(&args[0])?;
    
    let shape = match &args[1] {
        Value::List(list) if list.len() == 2 => {
            let rows = match &list[0] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "integer for rows".to_string(),
                    actual: format!("{:?}", list[0]),
                }),
            };
            let cols = match &list[1] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "integer for cols".to_string(),
                    actual: format!("{:?}", list[1]),
                }),
            };
            (rows, cols)
        }
        _ => return Err(VmError::TypeError {
            expected: "List of 2 integers for shape".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let matrix = CSRMatrix::from_triplets(triplets, shape);
    let sparse = GenericSparseMatrix::new(Box::new(matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
}

/// Create CSR matrix from dense matrix
/// Syntax: CSRFromDense[matrix]
pub fn csr_from_dense(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let dense_matrix = super::core::extract_numeric_matrix(&args[0])?;
    let matrix = CSRMatrix::from_dense(&dense_matrix);
    let sparse = GenericSparseMatrix::new(Box::new(matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_csr_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 2, 1, 2];
        let indptr = vec![0, 2, 4];
        let shape = (2, 3);
        
        let csr = CSRMatrix::new(data, indices, indptr, shape).unwrap();
        
        assert_eq!(csr.shape(), (2, 3));
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(1, 2), 4.0);
        assert_eq!(csr.get(0, 1), 0.0);
    }
    
    #[test]
    fn test_csr_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (1, 2, 4.0)];
        let csr = CSRMatrix::from_triplets(triplets, (2, 3));
        
        assert_eq!(csr.shape(), (2, 3));
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(1, 2), 4.0);
    }
    
    #[test]
    fn test_csr_matvec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 2, 1, 2];
        let indptr = vec![0, 2, 4];
        let shape = (2, 3);
        
        let csr = CSRMatrix::new(data, indices, indptr, shape).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let result = csr.matvec(&x);
        
        assert_eq!(result, vec![7.0, 18.0]); // [1*1 + 2*3, 3*2 + 4*3]
    }
    
    #[test]
    fn test_csr_from_dense() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 4.0],
        ];
        
        let csr = CSRMatrix::from_dense(&dense);
        assert_eq!(csr.shape(), (2, 3));
        assert_eq!(csr.nnz(), 4);
        assert_eq!(csr.get(0, 0), 1.0);
        assert_eq!(csr.get(0, 2), 2.0);
        assert_eq!(csr.get(1, 1), 3.0);
        assert_eq!(csr.get(1, 2), 4.0);
    }
    
    #[test]
    fn test_csr_row_access() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let indices = vec![0, 2, 1, 2];
        let indptr = vec![0, 2, 4];
        let shape = (2, 3);
        
        let csr = CSRMatrix::new(data, indices, indptr, shape).unwrap();
        
        let (cols, vals) = csr.row(0).unwrap();
        assert_eq!(cols, &[0, 2]);
        assert_eq!(vals, &[1.0, 2.0]);
        
        let (cols, vals) = csr.row(1).unwrap();
        assert_eq!(cols, &[1, 2]);
        assert_eq!(vals, &[3.0, 4.0]);
    }
}