//! Coordinate (COO) Matrix Format
//!
//! COO format stores a sparse matrix using three arrays:
//! - `data`: Non-zero values
//! - `row`: Row indices for each non-zero value
//! - `col`: Column indices for each non-zero value
//!
//! This format is highly efficient for:
//! - Matrix construction from triplets
//! - Format conversion
//! - Simple iteration over non-zero elements

use super::core::{SparseMatrix, SparseFormat, GenericSparseMatrix};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Coordinate matrix implementation
#[derive(Debug, Clone)]
pub struct COOMatrix {
    /// Non-zero values
    pub data: Vec<f64>,
    /// Row indices for each non-zero value
    pub row: Vec<usize>,
    /// Column indices for each non-zero value
    pub col: Vec<usize>,
    /// Matrix dimensions
    pub shape: (usize, usize),
    /// Whether duplicates have been eliminated
    pub duplicates_eliminated: bool,
}

impl COOMatrix {
    /// Create new COO matrix from arrays
    pub fn new(
        data: Vec<f64>,
        row: Vec<usize>,
        col: Vec<usize>,
        shape: (usize, usize),
    ) -> Result<Self, String> {
        // Validate inputs
        if data.len() != row.len() || data.len() != col.len() {
            return Err("data, row, and col must have same length".to_string());
        }
        
        // Check bounds
        if let Some(&max_row) = row.iter().max() {
            if max_row >= shape.0 {
                return Err("row index exceeds matrix dimensions".to_string());
            }
        }
        
        if let Some(&max_col) = col.iter().max() {
            if max_col >= shape.1 {
                return Err("column index exceeds matrix dimensions".to_string());
            }
        }
        
        Ok(COOMatrix {
            data,
            row,
            col,
            shape,
            duplicates_eliminated: false,
        })
    }
    
    /// Create COO matrix from triplets (row, col, value)
    pub fn from_triplets(
        triplets: Vec<(usize, usize, f64)>,
        shape: (usize, usize),
    ) -> Self {
        let mut data = Vec::new();
        let mut row = Vec::new();
        let mut col = Vec::new();
        
        for (r, c, val) in triplets {
            if val != 0.0 && r < shape.0 && c < shape.1 {
                data.push(val);
                row.push(r);
                col.push(c);
            }
        }
        
        COOMatrix {
            data,
            row,
            col,
            shape,
            duplicates_eliminated: false,
        }
    }
    
    /// Create COO matrix from dense matrix
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
    
    /// Get element at (row, col)
    pub fn get_element(&self, target_row: usize, target_col: usize) -> f64 {
        if target_row >= self.shape.0 || target_col >= self.shape.1 {
            return 0.0;
        }
        
        let mut sum = 0.0;
        for i in 0..self.data.len() {
            if self.row[i] == target_row && self.col[i] == target_col {
                sum += self.data[i];
            }
        }
        sum
    }
    
    /// Set element at (row, col) - Note: This may create duplicates
    pub fn set_element(&mut self, target_row: usize, target_col: usize, value: f64) {
        if target_row >= self.shape.0 || target_col >= self.shape.1 {
            return;
        }
        
        // Find existing entry
        for i in 0..self.data.len() {
            if self.row[i] == target_row && self.col[i] == target_col {
                if value == 0.0 {
                    // Remove entry
                    self.data.remove(i);
                    self.row.remove(i);
                    self.col.remove(i);
                } else {
                    // Update entry
                    self.data[i] = value;
                }
                return;
            }
        }
        
        // Add new entry if value is non-zero
        if value != 0.0 {
            self.data.push(value);
            self.row.push(target_row);
            self.col.push(target_col);
            self.duplicates_eliminated = false;
        }
    }
    
    /// Eliminate duplicate entries by summing them
    pub fn eliminate_duplicates(&mut self) {
        if self.duplicates_eliminated {
            return;
        }
        
        let mut entries: HashMap<(usize, usize), f64> = HashMap::new();
        
        for i in 0..self.data.len() {
            let key = (self.row[i], self.col[i]);
            *entries.entry(key).or_insert(0.0) += self.data[i];
        }
        
        // Rebuild arrays without duplicates
        self.data.clear();
        self.row.clear();
        self.col.clear();
        
        for ((r, c), val) in entries {
            if val != 0.0 {
                self.data.push(val);
                self.row.push(r);
                self.col.push(c);
            }
        }
        
        self.duplicates_eliminated = true;
    }
    
    /// Eliminate zeros from the matrix
    pub fn eliminate_zeros(&mut self) {
        let mut i = 0;
        while i < self.data.len() {
            if self.data[i] == 0.0 {
                self.data.remove(i);
                self.row.remove(i);
                self.col.remove(i);
            } else {
                i += 1;
            }
        }
    }
    
    /// Sort entries by (row, col) order
    pub fn sort_indices(&mut self) {
        let mut indices: Vec<usize> = (0..self.data.len()).collect();
        indices.sort_by(|&a, &b| {
            (self.row[a], self.col[a]).cmp(&(self.row[b], self.col[b]))
        });
        
        let old_data = std::mem::take(&mut self.data);
        let old_row = std::mem::take(&mut self.row);
        let old_col = std::mem::take(&mut self.col);
        
        for &i in &indices {
            self.data.push(old_data[i]);
            self.row.push(old_row[i]);
            self.col.push(old_col[i]);
        }
    }
    
    /// Matrix-vector multiplication
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        if x.len() != self.shape.1 {
            panic!("Vector length must match matrix columns");
        }
        
        let mut result = vec![0.0; self.shape.0];
        
        for i in 0..self.data.len() {
            result[self.row[i]] += self.data[i] * x[self.col[i]];
        }
        
        result
    }
    
    /// Transpose to another COO matrix
    pub fn transpose(&self) -> COOMatrix {
        COOMatrix {
            data: self.data.clone(),
            row: self.col.clone(),
            col: self.row.clone(),
            shape: (self.shape.1, self.shape.0),
            duplicates_eliminated: self.duplicates_eliminated,
        }
    }
    
    /// Convert to CSR format
    pub fn to_csr(&self) -> super::csr::CSRMatrix {
        let triplets = self.triplets();
        super::csr::CSRMatrix::from_triplets(triplets, self.shape)
    }
    
    /// Convert to CSC format
    pub fn to_csc(&self) -> super::csc::CSCMatrix {
        let triplets = self.triplets();
        super::csc::CSCMatrix::from_triplets(triplets, self.shape)
    }
}

impl SparseMatrix for COOMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::COO
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
        (0..self.data.len())
            .map(|i| (self.row[i], self.col[i], self.data[i]))
            .collect()
    }
    
    fn convert_to(&self, format: SparseFormat) -> Box<dyn SparseMatrix> {
        match format {
            SparseFormat::COO => Box::new(self.clone()),
            SparseFormat::CSR => Box::new(self.to_csr()),
            SparseFormat::CSC => Box::new(self.to_csc()),
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

impl Foreign for COOMatrix {
    fn type_name(&self) -> &'static str {
        "COOMatrix"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Data" => {
                let data: Vec<Value> = self.data.iter().map(|&v| Value::Real(v)).collect();
                Ok(Value::List(data))
            }
            "Row" => {
                let row: Vec<Value> = self.row.iter().map(|&i| Value::Integer(i as i64)).collect();
                Ok(Value::List(row))
            }
            "Col" => {
                let col: Vec<Value> = self.col.iter().map(|&i| Value::Integer(i as i64)).collect();
                Ok(Value::List(col))
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
            "HasDuplicates" => {
                Ok(Value::Integer(if self.duplicates_eliminated { 0 } else { 1 }))
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
// COO MATRIX CONSTRUCTOR FUNCTIONS
// ===============================

/// Create COO matrix from data, row, col arrays
/// Syntax: COOMatrix[data, row, col, shape]
pub fn coo_matrix(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (data, row, col, shape)".to_string(),
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
    
    // Extract row array
    let row = match &args[1] {
        Value::List(list) => {
            let mut row_vec = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => row_vec.push(*i as usize),
                    _ => return Err(VmError::TypeError {
                        expected: "integer list for row".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            row_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List for row".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // Extract col array
    let col = match &args[2] {
        Value::List(list) => {
            let mut col_vec = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => col_vec.push(*i as usize),
                    _ => return Err(VmError::TypeError {
                        expected: "integer list for col".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            col_vec
        }
        _ => return Err(VmError::TypeError {
            expected: "List for col".to_string(),
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
    
    match COOMatrix::new(data, row, col, shape) {
        Ok(matrix) => {
            let sparse = GenericSparseMatrix::new(Box::new(matrix));
            Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
        }
        Err(e) => Err(VmError::Runtime(format!("COO matrix creation failed: {}", e))),
    }
}

/// Create COO matrix from triplets
/// Syntax: COOFromTriplets[triplets, shape]
pub fn coo_from_triplets(args: &[Value]) -> VmResult<Value> {
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
    
    let matrix = COOMatrix::from_triplets(triplets, shape);
    let sparse = GenericSparseMatrix::new(Box::new(matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
}

/// Create COO matrix from dense matrix
/// Syntax: COOFromDense[matrix]
pub fn coo_from_dense(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (matrix)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let dense_matrix = super::core::extract_numeric_matrix(&args[0])?;
    let matrix = COOMatrix::from_dense(&dense_matrix);
    let sparse = GenericSparseMatrix::new(Box::new(matrix));
    Ok(Value::LyObj(LyObj::new(Box::new(sparse))))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_coo_creation() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row = vec![0, 0, 1, 1];
        let col = vec![0, 2, 1, 2];
        let shape = (2, 3);
        
        let coo = COOMatrix::new(data, row, col, shape).unwrap();
        
        assert_eq!(coo.shape(), (2, 3));
        assert_eq!(coo.nnz(), 4);
        assert_eq!(coo.get(0, 0), 1.0);
        assert_eq!(coo.get(0, 2), 2.0);
        assert_eq!(coo.get(1, 1), 3.0);
        assert_eq!(coo.get(1, 2), 4.0);
        assert_eq!(coo.get(0, 1), 0.0);
    }
    
    #[test]
    fn test_coo_from_triplets() {
        let triplets = vec![(0, 0, 1.0), (0, 2, 2.0), (1, 1, 3.0), (1, 2, 4.0)];
        let coo = COOMatrix::from_triplets(triplets, (2, 3));
        
        assert_eq!(coo.shape(), (2, 3));
        assert_eq!(coo.nnz(), 4);
        assert_eq!(coo.get(0, 0), 1.0);
        assert_eq!(coo.get(0, 2), 2.0);
        assert_eq!(coo.get(1, 1), 3.0);
        assert_eq!(coo.get(1, 2), 4.0);
    }
    
    #[test]
    fn test_coo_matvec() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row = vec![0, 0, 1, 1];
        let col = vec![0, 2, 1, 2];
        let shape = (2, 3);
        
        let coo = COOMatrix::new(data, row, col, shape).unwrap();
        let x = vec![1.0, 2.0, 3.0];
        let result = coo.matvec(&x);
        
        assert_eq!(result, vec![7.0, 18.0]); // [1*1 + 2*3, 3*2 + 4*3]
    }
    
    #[test]
    fn test_coo_duplicates() {
        let data = vec![1.0, 2.0, 3.0];
        let row = vec![0, 0, 0];
        let col = vec![0, 0, 1];
        let shape = (2, 2);
        
        let mut coo = COOMatrix::new(data, row, col, shape).unwrap();
        assert_eq!(coo.get(0, 0), 3.0); // 1.0 + 2.0
        assert_eq!(coo.get(0, 1), 3.0);
        
        coo.eliminate_duplicates();
        assert_eq!(coo.nnz(), 2);
        assert_eq!(coo.get(0, 0), 3.0);
        assert_eq!(coo.get(0, 1), 3.0);
    }
    
    #[test]
    fn test_coo_from_dense() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 4.0],
        ];
        
        let coo = COOMatrix::from_dense(&dense);
        assert_eq!(coo.shape(), (2, 3));
        assert_eq!(coo.nnz(), 4);
        assert_eq!(coo.get(0, 0), 1.0);
        assert_eq!(coo.get(0, 2), 2.0);
        assert_eq!(coo.get(1, 1), 3.0);
        assert_eq!(coo.get(1, 2), 4.0);
    }
    
    #[test]
    fn test_coo_transpose() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let row = vec![0, 0, 1, 1];
        let col = vec![0, 2, 1, 2];
        let shape = (2, 3);
        
        let coo = COOMatrix::new(data, row, col, shape).unwrap();
        let transposed = coo.transpose();
        
        assert_eq!(transposed.shape(), (3, 2));
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(2, 0), 2.0);
        assert_eq!(transposed.get(1, 1), 3.0);
        assert_eq!(transposed.get(2, 1), 4.0);
    }
}