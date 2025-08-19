//! Dictionary of Keys (DOK) Matrix Format
//!
//! DOK format stores a sparse matrix using a HashMap where keys are (row, col)
//! tuples and values are the non-zero matrix elements.
//!
//! This format is highly efficient for:
//! - Incremental matrix construction
//! - Random access to elements
//! - Fast insertion and deletion

use super::core::{SparseMatrix, SparseFormat};
use crate::foreign::{Foreign, ForeignError};
use crate::vm::Value;
use std::any::Any;
use std::collections::HashMap;

/// Dictionary of Keys matrix implementation
#[derive(Debug, Clone)]
pub struct DOKMatrix {
    /// Dictionary mapping (row, col) -> value
    pub data: HashMap<(usize, usize), f64>,
    /// Matrix dimensions
    pub shape: (usize, usize),
}

impl DOKMatrix {
    /// Create new DOK matrix
    pub fn new(shape: (usize, usize)) -> Self {
        DOKMatrix {
            data: HashMap::new(),
            shape,
        }
    }
    
    /// Create DOK matrix from triplets
    pub fn from_triplets(
        triplets: Vec<(usize, usize, f64)>,
        shape: (usize, usize),
    ) -> Self {
        let mut matrix = DOKMatrix::new(shape);
        
        for (row, col, val) in triplets {
            if val != 0.0 && row < shape.0 && col < shape.1 {
                matrix.data.insert((row, col), val);
            }
        }
        
        matrix
    }
}

impl SparseMatrix for DOKMatrix {
    fn format(&self) -> SparseFormat {
        SparseFormat::DOK
    }
    
    fn shape(&self) -> (usize, usize) {
        self.shape
    }
    
    fn nnz(&self) -> usize {
        self.data.len()
    }
    
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data.get(&(row, col)).copied().unwrap_or(0.0)
    }
    
    fn set(&mut self, row: usize, col: usize, value: f64) {
        if row >= self.shape.0 || col >= self.shape.1 {
            return;
        }
        
        if value == 0.0 {
            self.data.remove(&(row, col));
        } else {
            self.data.insert((row, col), value);
        }
    }
    
    fn triplets(&self) -> Vec<(usize, usize, f64)> {
        self.data.iter()
            .map(|(&(r, c), &v)| (r, c, v))
            .collect()
    }
    
    fn convert_to(&self, format: SparseFormat) -> Box<dyn SparseMatrix> {
        let triplets = self.triplets();
        match format {
            SparseFormat::DOK => Box::new(self.clone()),
            SparseFormat::COO => Box::new(super::coo::COOMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::CSR => Box::new(super::csr::CSRMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::CSC => Box::new(super::csc::CSCMatrix::from_triplets(triplets, self.shape)),
            SparseFormat::LIL => Box::new(super::lil::LILMatrix::from_triplets(triplets, self.shape)),
        }
    }
}

impl Foreign for DOKMatrix {
    fn type_name(&self) -> &'static str {
        "DOKMatrix"
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