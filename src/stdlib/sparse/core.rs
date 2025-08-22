//! Core Sparse Matrix Infrastructure
//!
//! Provides the foundational traits, enums, and utility functions for sparse matrix operations.
//! This module defines the common interface that all sparse matrix formats implement.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Sparse matrix storage format types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseFormat {
    /// Compressed Sparse Row format
    CSR,
    /// Compressed Sparse Column format  
    CSC,
    /// Coordinate (triplet) format
    COO,
    /// Dictionary of Keys format
    DOK,
    /// List of Lists format
    LIL,
}

impl SparseFormat {
    /// Get format name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            SparseFormat::CSR => "CSR",
            SparseFormat::CSC => "CSC", 
            SparseFormat::COO => "COO",
            SparseFormat::DOK => "DOK",
            SparseFormat::LIL => "LIL",
        }
    }

    /// Parse format from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CSR" => Some(SparseFormat::CSR),
            "CSC" => Some(SparseFormat::CSC),
            "COO" => Some(SparseFormat::COO),
            "DOK" => Some(SparseFormat::DOK),
            "LIL" => Some(SparseFormat::LIL),
            _ => None,
        }
    }
}

/// Common trait for all sparse matrix implementations
pub trait SparseMatrix: Foreign + std::fmt::Debug {
    /// Get the sparse format type
    fn format(&self) -> SparseFormat;
    
    /// Get matrix dimensions (rows, cols)
    fn shape(&self) -> (usize, usize);
    
    /// Get number of non-zero elements
    fn nnz(&self) -> usize;
    
    /// Get density (fraction of non-zero elements)
    fn density(&self) -> f64 {
        let (rows, cols) = self.shape();
        if rows == 0 || cols == 0 {
            0.0
        } else {
            self.nnz() as f64 / (rows * cols) as f64
        }
    }
    
    /// Check if matrix is square
    fn is_square(&self) -> bool {
        let (rows, cols) = self.shape();
        rows == cols
    }
    
    /// Get element at (row, col), returns 0.0 if not present
    fn get(&self, row: usize, col: usize) -> f64;
    
    /// Set element at (row, col)
    fn set(&mut self, row: usize, col: usize, value: f64);
    
    /// Convert to dense matrix representation
    fn to_dense(&self) -> Vec<Vec<f64>> {
        let (rows, cols) = self.shape();
        let mut dense = vec![vec![0.0; cols]; rows];
        
        for i in 0..rows {
            for j in 0..cols {
                dense[i][j] = self.get(i, j);
            }
        }
        
        dense
    }
    
    /// Get all non-zero elements as (row, col, value) triplets
    fn triplets(&self) -> Vec<(usize, usize, f64)>;
    
    /// Convert to another sparse format
    fn convert_to(&self, format: SparseFormat) -> Box<dyn SparseMatrix>;
}

/// Sparse matrix metadata for optimization hints
#[derive(Debug, Clone)]
pub struct SparseMetadata {
    /// Whether the matrix is symmetric
    pub symmetric: bool,
    /// Whether the matrix is positive definite
    pub positive_definite: bool,
    /// Whether the matrix is triangular (upper, lower, or none)
    pub triangular: Option<TriangularType>,
    /// Whether the matrix has sorted indices
    pub sorted_indices: bool,
    /// User-defined properties
    pub properties: HashMap<String, String>,
}

impl Default for SparseMetadata {
    fn default() -> Self {
        Self {
            symmetric: false,
            positive_definite: false,
            triangular: None,
            sorted_indices: true,
            properties: HashMap::new(),
        }
    }
}

/// Triangular matrix types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriangularType {
    Upper,
    Lower,
}

/// Generic sparse matrix container that can hold any format
#[derive(Debug)]
pub struct GenericSparseMatrix {
    /// The actual sparse matrix implementation
    matrix: Box<dyn SparseMatrix>,
    /// Matrix metadata
    metadata: SparseMetadata,
}

impl GenericSparseMatrix {
    /// Create new generic sparse matrix
    pub fn new(matrix: Box<dyn SparseMatrix>) -> Self {
        Self {
            matrix,
            metadata: SparseMetadata::default(),
        }
    }
    
    /// Create with metadata
    pub fn with_metadata(matrix: Box<dyn SparseMatrix>, metadata: SparseMetadata) -> Self {
        Self {
            matrix,
            metadata,
        }
    }
    
    /// Get reference to underlying matrix
    pub fn matrix(&self) -> &dyn SparseMatrix {
        self.matrix.as_ref()
    }
    
    /// Get mutable reference to underlying matrix
    pub fn matrix_mut(&mut self) -> &mut dyn SparseMatrix {
        self.matrix.as_mut()
    }
    
    /// Get reference to metadata
    pub fn metadata(&self) -> &SparseMetadata {
        &self.metadata
    }
    
    /// Get mutable reference to metadata
    pub fn metadata_mut(&mut self) -> &mut SparseMetadata {
        &mut self.metadata
    }
}

impl Clone for GenericSparseMatrix {
    fn clone(&self) -> Self {
        // Clone using the format-specific conversion
        let format = self.matrix.format();
        let cloned_matrix = self.matrix.convert_to(format);
        Self {
            matrix: cloned_matrix,
            metadata: self.metadata.clone(),
        }
    }
}

impl Foreign for GenericSparseMatrix {
    fn type_name(&self) -> &'static str {
        "SparseMatrix"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Format" => Ok(Value::String(self.matrix.format().as_str().to_string())),
            "Shape" => {
                let (rows, cols) = self.matrix.shape();
                Ok(Value::List(vec![
                    Value::Integer(rows as i64),
                    Value::Integer(cols as i64)
                ]))
            }
            "Rows" => Ok(Value::Integer(self.matrix.shape().0 as i64)),
            "Cols" => Ok(Value::Integer(self.matrix.shape().1 as i64)),
            "NNZ" => Ok(Value::Integer(self.matrix.nnz() as i64)),
            "Density" => Ok(Value::Real(self.matrix.density())),
            "IsSquare" => Ok(Value::Integer(if self.matrix.is_square() { 1 } else { 0 })),
            "Get" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
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
                
                let col = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                Ok(Value::Real(self.matrix.get(row, col)))
            }
            "ToDense" => {
                let dense = self.matrix.to_dense();
                let rows: Vec<Value> = dense.iter()
                    .map(|row| {
                        let vals: Vec<Value> = row.iter()
                            .map(|&val| Value::Real(val))
                            .collect();
                        Value::List(vals)
                    })
                    .collect();
                Ok(Value::List(rows))
            }
            "Triplets" => {
                let triplets = self.matrix.triplets();
                let result: Vec<Value> = triplets.iter()
                    .map(|&(row, col, val)| {
                        Value::List(vec![
                            Value::Integer(row as i64),
                            Value::Integer(col as i64),
                            Value::Real(val)
                        ])
                    })
                    .collect();
                Ok(Value::List(result))
            }
            "ConvertTo" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let format_str = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let format = SparseFormat::from_str(format_str)
                    .ok_or_else(|| ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Valid sparse format (CSR, CSC, COO, DOK, LIL)".to_string(),
                        actual: format_str.clone(),
                    })?;
                
                let converted = self.matrix.convert_to(format);
                let new_sparse = GenericSparseMatrix::with_metadata(converted, self.metadata.clone());
                Ok(Value::LyObj(LyObj::new(Box::new(new_sparse))))
            }
            "IsSymmetric" => Ok(Value::Integer(if self.metadata.symmetric { 1 } else { 0 })),
            "IsPositiveDefinite" => Ok(Value::Integer(if self.metadata.positive_definite { 1 } else { 0 })),
            "IsTriangular" => match self.metadata.triangular {
                Some(TriangularType::Upper) => Ok(Value::String("Upper".to_string())),
                Some(TriangularType::Lower) => Ok(Value::String("Lower".to_string())),
                None => Ok(Value::String("None".to_string())),
            },
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
// UTILITY FUNCTIONS
// ===============================

/// Extract sparse matrix from Value
pub fn extract_sparse_matrix(value: &Value) -> VmResult<&GenericSparseMatrix> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<GenericSparseMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "SparseMatrix".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "SparseMatrix".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract sparse matrix from Value and clone it for modification
/// Since LyObj doesn't support mutable access, we clone and return a new matrix
pub fn extract_sparse_matrix_for_modification(value: &Value) -> VmResult<GenericSparseMatrix> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<GenericSparseMatrix>()
                .cloned()
                .ok_or_else(|| VmError::TypeError {
                    expected: "SparseMatrix".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "SparseMatrix".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract 2D numeric matrix from Value
pub fn extract_numeric_matrix(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(outer_list) => {
            let mut matrix = Vec::new();
            for row in outer_list {
                match row {
                    Value::List(inner_list) => {
                        let mut row_vec = Vec::new();
                        for element in inner_list {
                            match element {
                                Value::Real(r) => row_vec.push(*r),
                                Value::Integer(i) => row_vec.push(*i as f64),
                                _ => return Err(VmError::TypeError {
                                    expected: "numeric value".to_string(),
                                    actual: format!("{:?}", element),
                                }),
                            }
                        }
                        matrix.push(row_vec);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of lists (matrix)".to_string(),
                        actual: format!("list containing {:?}", row),
                    }),
                }
            }
            Ok(matrix)
        }
        _ => Err(VmError::TypeError {
            expected: "List of lists (matrix)".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract triplets (row, col, val) from Value
pub fn extract_triplets(value: &Value) -> VmResult<Vec<(usize, usize, f64)>> {
    match value {
        Value::List(triplet_list) => {
            let mut triplets = Vec::new();
            for triplet in triplet_list {
                match triplet {
                    Value::List(elements) if elements.len() == 3 => {
                        let row = match &elements[0] {
                            Value::Integer(i) => *i as usize,
                            _ => return Err(VmError::TypeError {
                                expected: "integer row index".to_string(),
                                actual: format!("{:?}", elements[0]),
                            }),
                        };
                        
                        let col = match &elements[1] {
                            Value::Integer(i) => *i as usize,
                            _ => return Err(VmError::TypeError {
                                expected: "integer column index".to_string(),
                                actual: format!("{:?}", elements[1]),
                            }),
                        };
                        
                        let val = match &elements[2] {
                            Value::Real(r) => *r,
                            Value::Integer(i) => *i as f64,
                            _ => return Err(VmError::TypeError {
                                expected: "numeric value".to_string(),
                                actual: format!("{:?}", elements[2]),
                            }),
                        };
                        
                        triplets.push((row, col, val));
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "triplet [row, col, value]".to_string(),
                        actual: format!("{:?}", triplet),
                    }),
                }
            }
            Ok(triplets)
        }
        _ => Err(VmError::TypeError {
            expected: "List of triplets".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sparse_format_conversion() {
        assert_eq!(SparseFormat::from_str("CSR"), Some(SparseFormat::CSR));
        assert_eq!(SparseFormat::from_str("csc"), Some(SparseFormat::CSC));
        assert_eq!(SparseFormat::from_str("COO"), Some(SparseFormat::COO));
        assert_eq!(SparseFormat::from_str("invalid"), None);
    }
    
    #[test]
    fn test_sparse_format_as_str() {
        assert_eq!(SparseFormat::CSR.as_str(), "CSR");
        assert_eq!(SparseFormat::CSC.as_str(), "CSC");
        assert_eq!(SparseFormat::COO.as_str(), "COO");
        assert_eq!(SparseFormat::DOK.as_str(), "DOK");
        assert_eq!(SparseFormat::LIL.as_str(), "LIL");
    }
    
    #[test]
    fn test_sparse_metadata_default() {
        let metadata = SparseMetadata::default();
        assert!(!metadata.symmetric);
        assert!(!metadata.positive_definite);
        assert!(metadata.triangular.is_none());
        assert!(metadata.sorted_indices);
        assert!(metadata.properties.is_empty());
    }
}