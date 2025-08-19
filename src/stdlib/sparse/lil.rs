//! List of Lists (LIL) Matrix Format
//!
//! LIL format stores a sparse matrix using two lists for each row:
//! - Column indices of non-zero elements
//! - Corresponding values
//!
//! This format is highly efficient for:
//! - Row-based matrix construction
//! - Fast row operations
//! - Incremental building

use super::core::{SparseMatrix, SparseFormat};
use crate::foreign::{Foreign, ForeignError};
use crate::vm::Value;
use std::any::Any;

/// List of Lists matrix implementation
#[derive(Debug, Clone)]
pub struct LILMatrix {
    /// For each row: list of (column_index, value) pairs
    pub rows: Vec<Vec<(usize, f64)>>,
    /// Matrix dimensions
    pub shape: (usize, usize),
}

impl LILMatrix {
    /// Create new LIL matrix
    pub fn new(shape: (usize, usize)) -> Self {
        LILMatrix {
            rows: vec![Vec::new(); shape.0],
            shape,
        }
    }
    
    /// Create LIL matrix from triplets
    pub fn from_triplets(
        triplets: Vec<(usize, usize, f64)>,
        shape: (usize, usize),
    ) -> Self {
        let mut matrix = LILMatrix::new(shape);
        
        for (row, col, val) in triplets {
            if val != 0.0 && row < shape.0 && col < shape.1 {
                matrix.rows[row].push((col, val));
            }
        }
        
        // Sort each row by column index
        for row in &mut matrix.rows {
            row.sort_by_key(|&(col, _)| col);
        }
        
        matrix
    }
}

impl SparseMatrix for LILMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::LIL
    }
    
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    fn nnz(&self) -> usize {
        self.rows.iter().map(|row| row.len()).sum()
    }
    
    fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.shape.0 || col >= self.shape.1 {
            return 0.0;
        }
        
        for &(c, v) in &self.rows[row] {
            if c == col {
                return v;
            }
        }
        0.0
    }
    
    fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.shape.0 || col >= self.shape.1 {
            return;
        }
        
        let row_data = &mut self.rows[row];
        
        // Find existing entry
        for (i, &(c, _)) in row_data.iter().enumerate() {
            if c == col {
                if value == 0.0 {
                    row_data.remove(i);
                } else {
                    row_data[i] = (col, value);
                }
                return;
            }
        }
        
        // Add new entry if non-zero
        if value != 0.0 {
            row_data.push((col, value));
            row_data.sort_by_key(|&(c, _)| c);
        }
    }
    
    fn triplets(&self) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::new();
        for (row_idx, row) in self.rows.iter().enumerate() {
            for &(col, val) in row {
                triplets.push((row_idx, col, val));
            }
        }
        triplets
    }
    
    fn convert_to(&self, format: SparseFormat) -> Box<dyn SparseMatrix> {
        let triplets = self.triplets();
        match format {
            SparseFormat::LIL => Box::new(self.clone()),
            SparseFormat::COO => Box::new(super::coo::COOMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::CSR => Box::new(super::csr::CSRMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::CSC => Box::new(super::csc::CSCMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::DOK => Box::new(super::dok::DOKMatrix::from_triplets(triplets, self.shape)),
        }
    }
}

impl Foreign for LILMatrix {
    fn type_name(&self) -> &'static str {
        "LILMatrix"
    }
    
    fn call_method(&self, _method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        // Placeholder implementation
        Err(ForeignError::UnknownMethod {
            type_name: self.type_name().to_string(),
            method: _method.to_string(),
        })
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}