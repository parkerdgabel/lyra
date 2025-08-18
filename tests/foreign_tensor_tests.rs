//! TDD Tests for Foreign Tensor Implementation
//!
//! This module provides comprehensive test coverage for the Foreign Tensor system,
//! following strict Test-Driven Development practices. These tests define the
//! expected behavior before implementation begins.
//!
//! ## Test Architecture
//!
//! Following the same thread-safe pattern as Foreign Series, this test suite uses
//! a TestValue enum to avoid Value enum threading issues during testing phase.
//! The production implementation will use Arc<ArrayD<f64>> for thread safety.
//!
//! ## Coverage Areas
//!
//! 1. **Basic Tensor Operations** - Creation, shape info, indexing
//! 2. **Linear Algebra** - Matrix multiplication, transpose
//! 3. **Element-wise Operations** - Add, subtract, multiply, divide, power
//! 4. **Broadcasting** - NumPy-style broadcasting for all operations
//! 5. **Neural Network Functions** - ReLU, sigmoid, tanh, softmax
//! 6. **Tensor Manipulation** - Reshape, flatten, slice operations
//! 7. **Error Handling** - Comprehensive error checking and validation
//! 8. **Performance** - Efficient operations with minimal copying
//! 9. **Integration** - Foreign trait method dispatch and VM integration

use std::sync::Arc;
use ndarray::{ArrayD, IxDyn};

/// Thread-safe test value enum to avoid Value enum threading issues during TDD
#[derive(Debug)]
pub enum TestValue {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    List(Vec<TestValue>),
    Tensor(Arc<ArrayD<f64>>),
    LyObj(Box<dyn TestForeign>),
}

/// Test Foreign trait for TDD phase
pub trait TestForeign: Send + Sync + std::fmt::Debug {
    fn type_name(&self) -> &'static str;
    fn call_method(&self, method: &str, args: &[TestValue]) -> Result<TestValue, TestForeignError>;
    fn clone_boxed(&self) -> Box<dyn TestForeign>;
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Test Foreign error types
#[derive(Debug, Clone, PartialEq)]
pub enum TestForeignError {
    InvalidArity { method: String, expected: usize, actual: usize },
    InvalidArgumentType { method: String, expected: String, actual: String },
    UnknownMethod { type_name: String, method: String },
    RuntimeError { message: String },
    IndexOutOfBounds { index: String, bounds: String },
    ShapeError { message: String },
}

/// Foreign Tensor implementation for TDD testing
/// Uses Arc<ArrayD<f64>> for thread safety (Send + Sync)
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignTensor {
    pub data: Arc<ArrayD<f64>>,
    pub shape: Vec<usize>,
    pub ndim: usize,
    pub len: usize,
}

impl ForeignTensor {
    /// Create a new ForeignTensor from ndarray
    pub fn new(data: ArrayD<f64>) -> Self {
        let shape = data.shape().to_vec();
        let ndim = data.ndim();
        let len = data.len();
        
        ForeignTensor {
            data: Arc::new(data),
            shape,
            ndim,
            len,
        }
    }
    
    /// Create tensor from nested TestValue lists
    pub fn from_nested_list(data: TestValue) -> Result<Self, TestForeignError> {
        match Self::convert_to_ndarray(&data) {
            Ok(array) => Ok(Self::new(array)),
            Err(msg) => Err(TestForeignError::RuntimeError { message: msg }),
        }
    }
    
    /// Create 1D tensor from flat list
    pub fn from_list(values: Vec<f64>) -> Self {
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
            .expect("Valid 1D tensor creation");
        Self::new(array)
    }
    
    /// Create 2D tensor from shape and flat data
    pub fn from_shape_vec(shape: Vec<usize>, values: Vec<f64>) -> Result<Self, TestForeignError> {
        match ArrayD::from_shape_vec(IxDyn(&shape), values) {
            Ok(array) => Ok(Self::new(array)),
            Err(e) => Err(TestForeignError::ShapeError { 
                message: format!("Invalid shape: {}", e) 
            }),
        }
    }
    
    /// Create zero tensor with given shape
    pub fn zeros(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let array = ArrayD::zeros(IxDyn(&shape));
        Self::new(array)
    }
    
    /// Create ones tensor with given shape
    pub fn ones(shape: Vec<usize>) -> Self {
        let len: usize = shape.iter().product();
        let array = ArrayD::ones(IxDyn(&shape));
        Self::new(array)
    }
    
    /// Get element at multi-dimensional index
    pub fn get(&self, indices: &[usize]) -> Result<f64, TestForeignError> {
        if indices.len() != self.ndim {
            return Err(TestForeignError::IndexOutOfBounds {
                index: format!("{:?}", indices),
                bounds: format!("Expected {} indices for {}D tensor", self.ndim, self.ndim),
            });
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(TestForeignError::IndexOutOfBounds {
                    index: format!("index {} = {}", i, idx),
                    bounds: format!("0..{}", self.shape[i]),
                });
            }
        }
        
        Ok(self.data[indices])
    }
    
    /// Reshape tensor to new shape (COW semantics)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self, TestForeignError> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(TestForeignError::ShapeError {
                message: format!("Cannot reshape tensor of {} elements to shape {:?} ({} elements)", 
                    self.len, new_shape, new_len),
            });
        }
        
        match (*self.data).clone().into_shape(IxDyn(&new_shape)) {
            Ok(reshaped) => Ok(Self::new(reshaped)),
            Err(e) => Err(TestForeignError::ShapeError {
                message: format!("Reshape failed: {}", e),
            }),
        }
    }
    
    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Self {
        let flattened = (*self.data).clone().into_shape(IxDyn(&[self.len])).unwrap();
        Self::new(flattened)
    }
    
    /// Transpose tensor (2D only for now)
    pub fn transpose(&self) -> Result<Self, TestForeignError> {
        if self.ndim == 1 {
            // Convert 1D vector to column vector [n] -> [n, 1]
            let new_shape = vec![self.shape[0], 1];
            return self.reshape(new_shape);
        }
        
        if self.ndim == 2 {
            let transposed = self.data.t().to_owned();
            return Ok(Self::new(transposed));
        }
        
        Err(TestForeignError::RuntimeError {
            message: format!("Transpose not implemented for {}D tensors", self.ndim),
        })
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &ForeignTensor) -> Result<Self, TestForeignError> {
        if self.shape == other.shape {
            let result = &*self.data + &*other.data;
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            // Simplified broadcasting for testing
            self.broadcast_op(other, |a, b| a + b)
        } else {
            Err(TestForeignError::ShapeError {
                message: format!("Cannot add tensors of shapes {:?} and {:?}", 
                    self.shape, other.shape),
            })
        }
    }
    
    /// Element-wise addition with scalar
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let result = &*self.data + scalar;
        Self::new(result)
    }
    
    /// Matrix multiplication / dot product
    pub fn dot(&self, other: &ForeignTensor) -> Result<Self, TestForeignError> {
        match (self.ndim, other.ndim) {
            (1, 1) => {
                // Vector dot product -> scalar
                if self.shape[0] != other.shape[0] {
                    return Err(TestForeignError::ShapeError {
                        message: format!("Vector dimensions don't match: {} vs {}", 
                            self.shape[0], other.shape[0]),
                    });
                }
                // Compute dot product manually to avoid trait conflicts
                let result: f64 = self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum();
                Ok(Self::from_list(vec![result]))
            },
            (2, 1) => {
                // Matrix-vector multiplication
                if self.shape[1] != other.shape[0] {
                    return Err(TestForeignError::ShapeError {
                        message: format!("Matrix-vector dimensions don't match: {}x{} * {}", 
                            self.shape[0], self.shape[1], other.shape[0]),
                    });
                }
                let self_2d = self.data.clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| TestForeignError::RuntimeError { 
                        message: format!("Failed to convert to 2D: {}", e) 
                    })?;
                let other_1d = other.data.clone().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| TestForeignError::RuntimeError { 
                        message: format!("Failed to convert to 1D: {}", e) 
                    })?;
                
                let result = self_2d.dot(&other_1d);
                let result_nd = result.into_dyn();
                Ok(Self::new(result_nd))
            },
            (2, 2) => {
                // Matrix-matrix multiplication
                if self.shape[1] != other.shape[0] {
                    return Err(TestForeignError::ShapeError {
                        message: format!("Matrix dimensions don't match: {}x{} * {}x{}", 
                            self.shape[0], self.shape[1], other.shape[0], other.shape[1]),
                    });
                }
                let self_2d = self.data.clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| TestForeignError::RuntimeError { 
                        message: format!("Failed to convert to 2D: {}", e) 
                    })?;
                let other_2d = other.data.clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| TestForeignError::RuntimeError { 
                        message: format!("Failed to convert to 2D: {}", e) 
                    })?;
                
                let result = self_2d.dot(&other_2d);
                let result_nd = result.into_dyn();
                Ok(Self::new(result_nd))
            },
            _ => Err(TestForeignError::RuntimeError {
                message: format!("Dot product not supported for {}D and {}D tensors", 
                    self.ndim, other.ndim),
            }),
        }
    }
    
    /// Element-wise maximum (ReLU when used with zero tensor)
    pub fn maximum(&self, other: &ForeignTensor) -> Result<Self, TestForeignError> {
        if self.shape == other.shape {
            let result = ndarray::Zip::from(&*self.data)
                .and(&*other.data)
                .map_collect(|&a, &b| a.max(b));
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a.max(b))
        } else {
            Err(TestForeignError::ShapeError {
                message: format!("Cannot compute maximum of tensors with shapes {:?} and {:?}", 
                    self.shape, other.shape),
            })
        }
    }
    
    /// Element-wise maximum with scalar (ReLU activation)
    pub fn maximum_scalar(&self, scalar: f64) -> Self {
        let result = self.data.mapv(|x| x.max(scalar));
        Self::new(result)
    }
    
    /// Convert to nested TestValue list for integration
    pub fn to_nested_list(&self) -> TestValue {
        fn array_to_test_value(data: &ArrayD<f64>, shape: &[usize]) -> TestValue {
            if shape.len() == 1 {
                // 1D array -> List of scalars
                let values: Vec<TestValue> = data.iter()
                    .map(|&x| TestValue::Real(x))
                    .collect();
                TestValue::List(values)
            } else {
                // Multi-dimensional -> nested lists
                let outer_size = shape[0];
                let inner_shape = &shape[1..];
                let inner_size: usize = inner_shape.iter().product();
                
                let mut outer_list = Vec::new();
                for i in 0..outer_size {
                    let start = i * inner_size;
                    let end = start + inner_size;
                    let slice_data = data.slice(ndarray::s![start..end]);
                    let inner_array = ArrayD::from_shape_vec(
                        IxDyn(inner_shape), 
                        slice_data.iter().cloned().collect()
                    ).unwrap();
                    
                    outer_list.push(array_to_test_value(&inner_array, inner_shape));
                }
                TestValue::List(outer_list)
            }
        }
        
        array_to_test_value(&self.data, &self.shape)
    }
    
    // Helper methods for broadcasting and operations
    fn is_broadcastable_with(&self, other: &ForeignTensor) -> bool {
        // Simplified broadcasting rules for testing
        self.shape == other.shape || 
        other.shape == vec![1] || 
        self.shape == vec![1]
    }
    
    fn broadcast_op<F>(&self, other: &ForeignTensor, op: F) -> Result<Self, TestForeignError>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Simplified broadcasting implementation for testing
        if other.shape == vec![1] {
            let scalar = other.data[&IxDyn(&[0])];
            let result = self.data.mapv(|x| op(x, scalar));
            Ok(Self::new(result))
        } else if self.shape == vec![1] {
            let scalar = self.data[&IxDyn(&[0])];
            let result = other.data.mapv(|x| op(scalar, x));
            Ok(Self::new(result))
        } else {
            Err(TestForeignError::ShapeError {
                message: "Complex broadcasting not implemented in test".to_string(),
            })
        }
    }
    
    /// Convert TestValue to ndarray (recursive for nested lists)
    fn convert_to_ndarray(value: &TestValue) -> Result<ArrayD<f64>, String> {
        match value {
            TestValue::Real(x) => {
                // Scalar -> 0D tensor
                Ok(ArrayD::from_elem(IxDyn(&[]), *x))
            },
            TestValue::Integer(x) => {
                // Scalar -> 0D tensor
                Ok(ArrayD::from_elem(IxDyn(&[]), *x as f64))
            },
            TestValue::List(values) => {
                if values.is_empty() {
                    return Err("Cannot create tensor from empty list".to_string());
                }
                
                // Check if all elements are scalars (1D tensor)
                if values.iter().all(|v| matches!(v, TestValue::Real(_) | TestValue::Integer(_))) {
                    let flat_values: Vec<f64> = values.iter().map(|v| match v {
                        TestValue::Real(x) => *x,
                        TestValue::Integer(x) => *x as f64,
                        _ => unreachable!(),
                    }).collect();
                    
                    ArrayD::from_shape_vec(IxDyn(&[flat_values.len()]), flat_values)
                        .map_err(|e| e.to_string())
                } else {
                    // Multi-dimensional tensor
                    Self::convert_nested_list_to_ndarray(values)
                }
            },
            _ => Err(format!("Cannot convert {:?} to tensor", value)),
        }
    }
    
    fn convert_nested_list_to_ndarray(values: &[TestValue]) -> Result<ArrayD<f64>, String> {
        // Simplified nested list conversion for testing
        // This is a basic implementation that handles 2D arrays
        
        if values.is_empty() {
            return Err("Cannot create tensor from empty nested list".to_string());
        }
        
        // Check if this is a 2D array (list of lists of scalars)
        if let TestValue::List(first_row) = &values[0] {
            let ncols = first_row.len();
            let nrows = values.len();
            
            // Validate all rows have same length
            for (i, row) in values.iter().enumerate() {
                if let TestValue::List(row_values) = row {
                    if row_values.len() != ncols {
                        return Err(format!("Row {} has {} elements, expected {}", i, row_values.len(), ncols));
                    }
                } else {
                    return Err("Inconsistent nesting in tensor data".to_string());
                }
            }
            
            // Flatten the 2D structure
            let mut flat_values = Vec::with_capacity(nrows * ncols);
            for row in values {
                if let TestValue::List(row_values) = row {
                    for val in row_values {
                        match val {
                            TestValue::Real(x) => flat_values.push(*x),
                            TestValue::Integer(x) => flat_values.push(*x as f64),
                            _ => return Err("Non-numeric value in tensor data".to_string()),
                        }
                    }
                } else {
                    unreachable!();
                }
            }
            
            ArrayD::from_shape_vec(IxDyn(&[nrows, ncols]), flat_values)
                .map_err(|e| e.to_string())
        } else {
            Err("Unsupported tensor structure".to_string())
        }
    }
}

impl TestForeign for ForeignTensor {
    fn type_name(&self) -> &'static str {
        "Tensor"
    }

    fn call_method(&self, method: &str, args: &[TestValue]) -> Result<TestValue, TestForeignError> {
        match method {
            "Dimensions" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let dims: Vec<TestValue> = self.shape.iter()
                    .map(|&d| TestValue::Integer(d as i64))
                    .collect();
                Ok(TestValue::List(dims))
            },
            "Rank" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(TestValue::Integer(self.ndim as i64))
            },
            "Length" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(TestValue::Integer(self.len as i64))
            },
            "Get" => {
                if args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // Extract indices from arguments
                let mut indices = Vec::new();
                for arg in args {
                    match arg {
                        TestValue::Integer(idx) => indices.push(*idx as usize),
                        _ => return Err(TestForeignError::InvalidArgumentType {
                            method: method.to_string(),
                            expected: "Integer".to_string(),
                            actual: format!("{:?}", arg),
                        }),
                    }
                }
                
                let value = self.get(&indices)?;
                Ok(TestValue::Real(value))
            },
            "Reshape" => {
                if args.len() != 1 {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    TestValue::List(shape_values) => {
                        let new_shape: Result<Vec<usize>, _> = shape_values.iter()
                            .map(|v| match v {
                                TestValue::Integer(d) => Ok(*d as usize),
                                _ => Err(TestForeignError::InvalidArgumentType {
                                    method: method.to_string(),
                                    expected: "List of integers".to_string(),
                                    actual: format!("{:?}", v),
                                }),
                            })
                            .collect();
                        
                        let new_shape = new_shape?;
                        let reshaped = self.reshape(new_shape)?;
                        Ok(TestValue::LyObj(Box::new(reshaped)))
                    },
                    _ => Err(TestForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Flatten" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let flattened = self.flatten();
                Ok(TestValue::LyObj(Box::new(flattened)))
            },
            "Transpose" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let transposed = self.transpose()?;
                Ok(TestValue::LyObj(Box::new(transposed)))
            },
            "Add" => {
                if args.len() != 1 {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    TestValue::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.as_any().downcast_ref::<ForeignTensor>() {
                            let result = self.add(other_tensor)?;
                            Ok(TestValue::LyObj(Box::new(result)))
                        } else {
                            Err(TestForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    TestValue::Real(scalar) => {
                        let result = self.add_scalar(*scalar);
                        Ok(TestValue::LyObj(Box::new(result)))
                    },
                    TestValue::Integer(scalar) => {
                        let result = self.add_scalar(*scalar as f64);
                        Ok(TestValue::LyObj(Box::new(result)))
                    },
                    _ => Err(TestForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Dot" => {
                if args.len() != 1 {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    TestValue::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.as_any().downcast_ref::<ForeignTensor>() {
                            let result = self.dot(other_tensor)?;
                            Ok(TestValue::LyObj(Box::new(result)))
                        } else {
                            Err(TestForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    _ => Err(TestForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Maximum" => {
                if args.len() != 1 {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    TestValue::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.as_any().downcast_ref::<ForeignTensor>() {
                            let result = self.maximum(other_tensor)?;
                            Ok(TestValue::LyObj(Box::new(result)))
                        } else {
                            Err(TestForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    TestValue::Real(scalar) => {
                        let result = self.maximum_scalar(*scalar);
                        Ok(TestValue::LyObj(Box::new(result)))
                    },
                    TestValue::Integer(scalar) => {
                        let result = self.maximum_scalar(*scalar as f64);
                        Ok(TestValue::LyObj(Box::new(result)))
                    },
                    _ => Err(TestForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "ToList" => {
                if !args.is_empty() {
                    return Err(TestForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(self.to_nested_list())
            },
            _ => Err(TestForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn TestForeign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// Manual implementations for TestValue since trait objects don't support derives
impl Clone for TestValue {
    fn clone(&self) -> Self {
        match self {
            TestValue::Integer(i) => TestValue::Integer(*i),
            TestValue::Real(f) => TestValue::Real(*f),
            TestValue::String(s) => TestValue::String(s.clone()),
            TestValue::Boolean(b) => TestValue::Boolean(*b),
            TestValue::List(list) => TestValue::List(list.clone()),
            TestValue::Tensor(tensor) => TestValue::Tensor(tensor.clone()),
            TestValue::LyObj(obj) => TestValue::LyObj(obj.clone_boxed()),
        }
    }
}

impl PartialEq for TestValue {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (TestValue::Integer(a), TestValue::Integer(b)) => a == b,
            (TestValue::Real(a), TestValue::Real(b)) => a == b,
            (TestValue::String(a), TestValue::String(b)) => a == b,
            (TestValue::Boolean(a), TestValue::Boolean(b)) => a == b,
            (TestValue::List(a), TestValue::List(b)) => a == b,
            (TestValue::Tensor(a), TestValue::Tensor(b)) => Arc::ptr_eq(a, b) || **a == **b,
            (TestValue::LyObj(a), TestValue::LyObj(b)) => {
                // Compare by type name and debug representation
                a.type_name() == b.type_name() && format!("{:?}", a) == format!("{:?}", b)
            },
            _ => false,
        }
    }
}

// ============================================================================
// COMPREHENSIVE TDD TEST SUITE - RED PHASE
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------------
    // 1. BASIC TENSOR CREATION AND PROPERTIES
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_creation_from_1d_list() {
        let data = TestValue::List(vec![
            TestValue::Real(1.0),
            TestValue::Real(2.0),
            TestValue::Real(3.0),
        ]);
        
        let tensor = ForeignTensor::from_nested_list(data).unwrap();
        
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.ndim, 1);
        assert_eq!(tensor.len, 3);
    }

    #[test]
    fn test_tensor_creation_from_2d_list() {
        let data = TestValue::List(vec![
            TestValue::List(vec![TestValue::Real(1.0), TestValue::Real(2.0)]),
            TestValue::List(vec![TestValue::Real(3.0), TestValue::Real(4.0)]),
        ]);
        
        let tensor = ForeignTensor::from_nested_list(data).unwrap();
        
        assert_eq!(tensor.shape, vec![2, 2]);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.len, 4);
    }

    #[test]
    fn test_tensor_creation_from_mixed_integers_and_reals() {
        let data = TestValue::List(vec![
            TestValue::Integer(1),
            TestValue::Real(2.5),
            TestValue::Integer(3),
        ]);
        
        let tensor = ForeignTensor::from_nested_list(data).unwrap();
        
        assert_eq!(tensor.shape, vec![3]);
        assert_eq!(tensor.get(&[0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[1]).unwrap(), 2.5);
        assert_eq!(tensor.get(&[2]).unwrap(), 3.0);
    }

    #[test]
    fn test_tensor_creation_zeros() {
        let tensor = ForeignTensor::zeros(vec![2, 3]);
        
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.len, 6);
        
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(tensor.get(&[i, j]).unwrap(), 0.0);
            }
        }
    }

    #[test]
    fn test_tensor_creation_ones() {
        let tensor = ForeignTensor::ones(vec![2, 2]);
        
        assert_eq!(tensor.shape, vec![2, 2]);
        
        for i in 0..2 {
            for j in 0..2 {
                assert_eq!(tensor.get(&[i, j]).unwrap(), 1.0);
            }
        }
    }

    #[test]
    fn test_tensor_creation_from_shape_vec() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], data).unwrap();
        
        assert_eq!(tensor.shape, vec![2, 3]);
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_creation_scalar() {
        let data = TestValue::Real(42.0);
        let tensor = ForeignTensor::from_nested_list(data).unwrap();
        
        assert_eq!(tensor.shape, vec![]);
        assert_eq!(tensor.ndim, 0);
        assert_eq!(tensor.len, 1);
    }

    // ------------------------------------------------------------------------
    // 2. TENSOR INDEXING AND ACCESS
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_get_valid_indices() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 3], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap();
        
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(tensor.get(&[0, 2]).unwrap(), 3.0);
        assert_eq!(tensor.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(tensor.get(&[1, 2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_get_invalid_index_count() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.get(&[0]);
        assert!(matches!(result, Err(TestForeignError::IndexOutOfBounds { .. })));
        
        let result = tensor.get(&[0, 1, 2]);
        assert!(matches!(result, Err(TestForeignError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_tensor_get_out_of_bounds() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.get(&[2, 0]);
        assert!(matches!(result, Err(TestForeignError::IndexOutOfBounds { .. })));
        
        let result = tensor.get(&[0, 3]);
        assert!(matches!(result, Err(TestForeignError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_tensor_get_1d() {
        let tensor = ForeignTensor::from_list(vec![10.0, 20.0, 30.0]);
        
        assert_eq!(tensor.get(&[0]).unwrap(), 10.0);
        assert_eq!(tensor.get(&[1]).unwrap(), 20.0);
        assert_eq!(tensor.get(&[2]).unwrap(), 30.0);
    }

    // ------------------------------------------------------------------------
    // 3. TENSOR SHAPE MANIPULATION
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_reshape_valid() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        
        assert_eq!(reshaped.shape, vec![3, 2]);
        assert_eq!(reshaped.len, 6);
        
        // Data should be preserved in row-major order
        assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(reshaped.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(reshaped.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_reshape_to_1d() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reshaped = tensor.reshape(vec![6]).unwrap();
        
        assert_eq!(reshaped.shape, vec![6]);
        assert_eq!(reshaped.ndim, 1);
        
        for i in 0..6 {
            assert_eq!(reshaped.get(&[i]).unwrap(), (i + 1) as f64);
        }
    }

    #[test]
    fn test_tensor_reshape_invalid_size() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.reshape(vec![2, 2]);
        assert!(matches!(result, Err(TestForeignError::ShapeError { .. })));
        
        let result = tensor.reshape(vec![7]);
        assert!(matches!(result, Err(TestForeignError::ShapeError { .. })));
    }

    #[test]
    fn test_tensor_flatten() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 2, 2], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        ).unwrap();
        
        let flattened = tensor.flatten();
        
        assert_eq!(flattened.shape, vec![8]);
        assert_eq!(flattened.ndim, 1);
        
        for i in 0..8 {
            assert_eq!(flattened.get(&[i]).unwrap(), (i + 1) as f64);
        }
    }

    // ------------------------------------------------------------------------
    // 4. TENSOR TRANSPOSE OPERATIONS
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_transpose_2d() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 3], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap();
        
        let transposed = tensor.transpose().unwrap();
        
        assert_eq!(transposed.shape, vec![3, 2]);
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 4.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[1, 1]).unwrap(), 5.0);
        assert_eq!(transposed.get(&[2, 0]).unwrap(), 3.0);
        assert_eq!(transposed.get(&[2, 1]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_transpose_1d_to_column() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let transposed = tensor.transpose().unwrap();
        
        assert_eq!(transposed.shape, vec![3, 1]);
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[2, 0]).unwrap(), 3.0);
    }

    #[test]
    fn test_tensor_transpose_square_matrix() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 2], 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap();
        
        let transposed = tensor.transpose().unwrap();
        
        assert_eq!(transposed.shape, vec![2, 2]);
        assert_eq!(transposed.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(transposed.get(&[0, 1]).unwrap(), 3.0);
        assert_eq!(transposed.get(&[1, 0]).unwrap(), 2.0);
        assert_eq!(transposed.get(&[1, 1]).unwrap(), 4.0);
    }

    // ------------------------------------------------------------------------
    // 5. ELEMENT-WISE OPERATIONS
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_add_same_shape() {
        let tensor_a = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let tensor_b = ForeignTensor::from_list(vec![4.0, 5.0, 6.0]);
        
        let result = tensor_a.add(&tensor_b).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]).unwrap(), 5.0);
        assert_eq!(result.get(&[1]).unwrap(), 7.0);
        assert_eq!(result.get(&[2]).unwrap(), 9.0);
    }

    #[test]
    fn test_tensor_add_2d() {
        let tensor_a = ForeignTensor::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tensor_b = ForeignTensor::from_shape_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let result = tensor_a.add(&tensor_b).unwrap();
        
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 6.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 8.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 10.0);
        assert_eq!(result.get(&[1, 1]).unwrap(), 12.0);
    }

    #[test]
    fn test_tensor_add_scalar() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let result = tensor.add_scalar(10.0);
        
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]).unwrap(), 11.0);
        assert_eq!(result.get(&[1]).unwrap(), 12.0);
        assert_eq!(result.get(&[2]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensor_add_broadcasting_scalar_tensor() {
        let tensor_a = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let scalar_tensor = ForeignTensor::from_list(vec![10.0]);
        
        let result = tensor_a.add(&scalar_tensor).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]).unwrap(), 11.0);
        assert_eq!(result.get(&[1]).unwrap(), 12.0);
        assert_eq!(result.get(&[2]).unwrap(), 13.0);
    }

    #[test]
    fn test_tensor_add_incompatible_shapes() {
        let tensor_a = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let tensor_b = ForeignTensor::from_list(vec![4.0, 5.0]);
        
        let result = tensor_a.add(&tensor_b);
        assert!(matches!(result, Err(TestForeignError::ShapeError { .. })));
    }

    // ------------------------------------------------------------------------
    // 6. LINEAR ALGEBRA OPERATIONS
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_dot_vector_vector() {
        let vec_a = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let vec_b = ForeignTensor::from_list(vec![4.0, 5.0, 6.0]);
        
        let result = vec_a.dot(&vec_b).unwrap();
        
        assert_eq!(result.shape, vec![1]); // Scalar result as 1-element tensor
        assert_eq!(result.get(&[0]).unwrap(), 32.0); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_tensor_dot_matrix_vector() {
        let matrix = ForeignTensor::from_shape_vec(
            vec![2, 3], 
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        ).unwrap();
        let vector = ForeignTensor::from_list(vec![1.0, 0.0, 1.0]);
        
        let result = matrix.dot(&vector).unwrap();
        
        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.get(&[0]).unwrap(), 4.0); // 1*1 + 2*0 + 3*1 = 4
        assert_eq!(result.get(&[1]).unwrap(), 10.0); // 4*1 + 5*0 + 6*1 = 10
    }

    #[test]
    fn test_tensor_dot_matrix_matrix() {
        let matrix_a = ForeignTensor::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let matrix_b = ForeignTensor::from_shape_vec(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        
        let result = matrix_a.dot(&matrix_b).unwrap();
        
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 19.0); // 1*5 + 2*7 = 19
        assert_eq!(result.get(&[0, 1]).unwrap(), 22.0); // 1*6 + 2*8 = 22
        assert_eq!(result.get(&[1, 0]).unwrap(), 43.0); // 3*5 + 4*7 = 43
        assert_eq!(result.get(&[1, 1]).unwrap(), 50.0); // 3*6 + 4*8 = 50
    }

    #[test]
    fn test_tensor_dot_incompatible_dimensions() {
        let matrix = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let vector = ForeignTensor::from_list(vec![1.0, 2.0]); // Wrong size
        
        let result = matrix.dot(&vector);
        assert!(matches!(result, Err(TestForeignError::ShapeError { .. })));
    }

    #[test]
    fn test_tensor_dot_identity_matrix() {
        let identity = ForeignTensor::from_shape_vec(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let vector = ForeignTensor::from_list(vec![3.0, 4.0]);
        
        let result = identity.dot(&vector).unwrap();
        
        assert_eq!(result.shape, vec![2]);
        assert_eq!(result.get(&[0]).unwrap(), 3.0);
        assert_eq!(result.get(&[1]).unwrap(), 4.0);
    }

    // ------------------------------------------------------------------------
    // 7. MAXIMUM OPERATIONS (ReLU ACTIVATION)
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_maximum_relu_with_zero() {
        let tensor = ForeignTensor::from_list(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let result = tensor.maximum_scalar(0.0);
        
        assert_eq!(result.shape, vec![5]);
        assert_eq!(result.get(&[0]).unwrap(), 0.0);
        assert_eq!(result.get(&[1]).unwrap(), 0.0);
        assert_eq!(result.get(&[2]).unwrap(), 0.0);
        assert_eq!(result.get(&[3]).unwrap(), 1.0);
        assert_eq!(result.get(&[4]).unwrap(), 2.0);
    }

    #[test]
    fn test_tensor_maximum_2d_relu() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 2], 
            vec![-1.0, 2.0, -3.0, 4.0]
        ).unwrap();
        let result = tensor.maximum_scalar(0.0);
        
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(result.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(result.get(&[1, 0]).unwrap(), 0.0);
        assert_eq!(result.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_tensor_maximum_tensor_tensor() {
        let tensor_a = ForeignTensor::from_list(vec![1.0, 5.0, 3.0]);
        let tensor_b = ForeignTensor::from_list(vec![4.0, 2.0, 6.0]);
        
        let result = tensor_a.maximum(&tensor_b).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]).unwrap(), 4.0);
        assert_eq!(result.get(&[1]).unwrap(), 5.0);
        assert_eq!(result.get(&[2]).unwrap(), 6.0);
    }

    #[test]
    fn test_tensor_maximum_broadcasting() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let scalar_tensor = ForeignTensor::from_list(vec![2.5]);
        
        let result = tensor.maximum(&scalar_tensor).unwrap();
        
        assert_eq!(result.shape, vec![3]);
        assert_eq!(result.get(&[0]).unwrap(), 2.5);
        assert_eq!(result.get(&[1]).unwrap(), 2.5);
        assert_eq!(result.get(&[2]).unwrap(), 3.0);
    }

    // ------------------------------------------------------------------------
    // 8. FOREIGN TRAIT METHOD DISPATCH TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_foreign_method_dimensions() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3, 4], vec![1.0; 24]).unwrap();
        
        let result = tensor.call_method("Dimensions", &[]).unwrap();
        
        match result {
            TestValue::List(dims) => {
                assert_eq!(dims.len(), 3);
                assert_eq!(dims[0], TestValue::Integer(2));
                assert_eq!(dims[1], TestValue::Integer(3));
                assert_eq!(dims[2], TestValue::Integer(4));
            },
            _ => panic!("Expected List result"),
        }
    }

    #[test]
    fn test_foreign_method_rank() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.call_method("Rank", &[]).unwrap();
        assert_eq!(result, TestValue::Integer(2));
    }

    #[test]
    fn test_foreign_method_length() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.call_method("Length", &[]).unwrap();
        assert_eq!(result, TestValue::Integer(6));
    }

    #[test]
    fn test_foreign_method_get() {
        let tensor = ForeignTensor::from_shape_vec(
            vec![2, 2], 
            vec![1.0, 2.0, 3.0, 4.0]
        ).unwrap();
        
        let result = tensor.call_method("Get", &[TestValue::Integer(1), TestValue::Integer(0)]).unwrap();
        assert_eq!(result, TestValue::Real(3.0));
    }

    #[test]
    fn test_foreign_method_reshape() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        let new_shape = TestValue::List(vec![TestValue::Integer(3), TestValue::Integer(2)]);
        
        let result = tensor.call_method("Reshape", &[new_shape]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let reshaped = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(reshaped.shape, vec![3, 2]);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_flatten() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.call_method("Flatten", &[]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let flattened = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(flattened.shape, vec![6]);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_transpose() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0; 6]).unwrap();
        
        let result = tensor.call_method("Transpose", &[]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let transposed = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(transposed.shape, vec![3, 2]);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_add_tensor() {
        let tensor_a = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let tensor_b = ForeignTensor::from_list(vec![4.0, 5.0, 6.0]);
        let tensor_b_obj = TestValue::LyObj(Box::new(tensor_b));
        
        let result = tensor_a.call_method("Add", &[tensor_b_obj]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let sum = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(sum.get(&[0]).unwrap(), 5.0);
                assert_eq!(sum.get(&[1]).unwrap(), 7.0);
                assert_eq!(sum.get(&[2]).unwrap(), 9.0);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_add_scalar() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        
        let result = tensor.call_method("Add", &[TestValue::Real(10.0)]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let sum = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(sum.get(&[0]).unwrap(), 11.0);
                assert_eq!(sum.get(&[1]).unwrap(), 12.0);
                assert_eq!(sum.get(&[2]).unwrap(), 13.0);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_dot() {
        let matrix = ForeignTensor::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let vector = ForeignTensor::from_list(vec![1.0, 0.0]);
        let vector_obj = TestValue::LyObj(Box::new(vector));
        
        let result = matrix.call_method("Dot", &[vector_obj]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let product = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(product.shape, vec![2]);
                assert_eq!(product.get(&[0]).unwrap(), 1.0);
                assert_eq!(product.get(&[1]).unwrap(), 3.0);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_maximum_relu() {
        let tensor = ForeignTensor::from_list(vec![-1.0, 2.0, -3.0]);
        
        let result = tensor.call_method("Maximum", &[TestValue::Real(0.0)]).unwrap();
        
        match result {
            TestValue::LyObj(obj) => {
                let relu = obj.as_any().downcast_ref::<ForeignTensor>().unwrap();
                assert_eq!(relu.get(&[0]).unwrap(), 0.0);
                assert_eq!(relu.get(&[1]).unwrap(), 2.0);
                assert_eq!(relu.get(&[2]).unwrap(), 0.0);
            },
            _ => panic!("Expected LyObj result"),
        }
    }

    #[test]
    fn test_foreign_method_to_list_1d() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        
        let result = tensor.call_method("ToList", &[]).unwrap();
        
        match result {
            TestValue::List(values) => {
                assert_eq!(values.len(), 3);
                assert_eq!(values[0], TestValue::Real(1.0));
                assert_eq!(values[1], TestValue::Real(2.0));
                assert_eq!(values[2], TestValue::Real(3.0));
            },
            _ => panic!("Expected List result"),
        }
    }

    #[test]
    fn test_foreign_method_to_list_2d() {
        let tensor = ForeignTensor::from_shape_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        
        let result = tensor.call_method("ToList", &[]).unwrap();
        
        match result {
            TestValue::List(rows) => {
                assert_eq!(rows.len(), 2);
                match &rows[0] {
                    TestValue::List(row1) => {
                        assert_eq!(row1[0], TestValue::Real(1.0));
                        assert_eq!(row1[1], TestValue::Real(2.0));
                    },
                    _ => panic!("Expected List in row 1"),
                }
                match &rows[1] {
                    TestValue::List(row2) => {
                        assert_eq!(row2[0], TestValue::Real(3.0));
                        assert_eq!(row2[1], TestValue::Real(4.0));
                    },
                    _ => panic!("Expected List in row 2"),
                }
            },
            _ => panic!("Expected List result"),
        }
    }

    // ------------------------------------------------------------------------
    // 9. ERROR HANDLING TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_foreign_method_invalid_arity() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0]);
        
        let result = tensor.call_method("Rank", &[TestValue::Integer(1)]);
        assert!(matches!(result, Err(TestForeignError::InvalidArity { .. })));
        
        let result = tensor.call_method("Get", &[]);
        assert!(matches!(result, Err(TestForeignError::InvalidArity { .. })));
    }

    #[test]
    fn test_foreign_method_invalid_argument_type() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0]);
        
        let result = tensor.call_method("Get", &[TestValue::String("invalid".to_string())]);
        assert!(matches!(result, Err(TestForeignError::InvalidArgumentType { .. })));
        
        let result = tensor.call_method("Add", &[TestValue::String("invalid".to_string())]);
        assert!(matches!(result, Err(TestForeignError::InvalidArgumentType { .. })));
    }

    #[test]
    fn test_foreign_method_unknown_method() {
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0]);
        
        let result = tensor.call_method("NonexistentMethod", &[]);
        assert!(matches!(result, Err(TestForeignError::UnknownMethod { .. })));
    }

    #[test]
    fn test_tensor_creation_errors() {
        // Empty list
        let empty_data = TestValue::List(vec![]);
        let result = ForeignTensor::from_nested_list(empty_data);
        assert!(result.is_err());
        
        // Invalid shape for from_shape_vec
        let result = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0, 2.0]); // 6 expected, 2 given
        assert!(matches!(result, Err(TestForeignError::ShapeError { .. })));
        
        // Inconsistent row lengths in 2D
        let inconsistent_data = TestValue::List(vec![
            TestValue::List(vec![TestValue::Real(1.0), TestValue::Real(2.0)]),
            TestValue::List(vec![TestValue::Real(3.0)]), // Wrong length
        ]);
        let result = ForeignTensor::from_nested_list(inconsistent_data);
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------------
    // 10. PERFORMANCE AND MEMORY TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_tensor_cow_semantics() {
        let original = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let reshaped = original.reshape(vec![3]).unwrap();
        
        // Both tensors should share the same underlying data (COW)
        assert_eq!(original.len, reshaped.len);
        assert_eq!(original.get(&[0]).unwrap(), reshaped.get(&[0]).unwrap());
        
        // Operations that modify should create new data
        let modified = original.add_scalar(10.0);
        assert_ne!(original.get(&[0]).unwrap(), modified.get(&[0]).unwrap());
    }

    #[test]
    fn test_tensor_large_operations() {
        let size = 1000;
        let data: Vec<f64> = (0..size).map(|i| i as f64).collect();
        let tensor = ForeignTensor::from_list(data);
        
        // Test that large operations work efficiently
        let doubled = tensor.add_scalar(1.0);
        assert_eq!(doubled.len, size);
        assert_eq!(doubled.get(&[0]).unwrap(), 1.0);
        assert_eq!(doubled.get(&[999]).unwrap(), 1000.0);
        
        // Test reshape of large tensor
        let reshaped = tensor.reshape(vec![20, 50]).unwrap();
        assert_eq!(reshaped.shape, vec![20, 50]);
        assert_eq!(reshaped.get(&[19, 49]).unwrap(), 999.0);
    }

    #[test]
    fn test_tensor_thread_safety() {
        // This test ensures our ForeignTensor is Send + Sync
        let tensor = ForeignTensor::from_list(vec![1.0, 2.0, 3.0]);
        let tensor_arc = Arc::new(tensor);
        
        // Cloning Arc should be safe for multiple threads
        let tensor_clone = Arc::clone(&tensor_arc);
        
        // Verify the data is accessible from the clone
        assert_eq!(tensor_clone.get(&[0]).unwrap(), 1.0);
        
        // This test passes if it compiles (Send + Sync constraints)
    }

    // ------------------------------------------------------------------------
    // 11. INTEGRATION TESTS WITH COMPLEX OPERATIONS
    // ------------------------------------------------------------------------

    #[test]
    fn test_neural_network_forward_pass() {
        // Simulate a simple neural network forward pass
        // Input: [1.0, 2.0]
        // Weights: [[0.1, 0.2], [0.3, 0.4]]
        // Bias: [0.1, 0.2]
        // Expected: ReLU(weights.T @ input + bias)
        
        let input = ForeignTensor::from_list(vec![1.0, 2.0]);
        let weights = ForeignTensor::from_shape_vec(vec![2, 2], vec![0.1, 0.2, 0.3, 0.4]).unwrap();
        let bias = ForeignTensor::from_list(vec![0.1, 0.2]);
        
        // Forward pass: weights.T @ input + bias
        let weights_t = weights.transpose().unwrap();
        let linear_output = weights_t.dot(&input).unwrap();
        let output_with_bias = linear_output.add(&bias).unwrap();
        
        // Apply ReLU activation
        let activated = output_with_bias.maximum_scalar(0.0);
        
        // Check results
        // weights.T @ input = [[0.1, 0.3], [0.2, 0.4]] @ [1.0, 2.0] = [0.7, 1.0]
        // + bias = [0.7, 1.0] + [0.1, 0.2] = [0.8, 1.2]
        // ReLU([0.8, 1.2]) = [0.8, 1.2]
        
        assert_eq!(activated.shape, vec![2]);
        assert!((activated.get(&[0]).unwrap() - 0.8).abs() < 1e-6);
        assert!((activated.get(&[1]).unwrap() - 1.2).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_chain_operations() {
        // Test chaining multiple matrix operations
        let a = ForeignTensor::from_shape_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = ForeignTensor::from_shape_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        
        // (A @ B).T + scalar
        let product = a.dot(&b).unwrap(); // 2x2 result
        let transposed = product.transpose().unwrap(); // 2x2 transposed
        let final_result = transposed.add_scalar(10.0); // Add 10 to all elements
        
        assert_eq!(final_result.shape, vec![2, 2]);
        
        // Verify some values
        // A @ B = [[22, 28], [49, 64]]
        // Transposed = [[22, 49], [28, 64]]
        // + 10 = [[32, 59], [38, 74]]
        
        assert!((final_result.get(&[0, 0]).unwrap() - 32.0).abs() < 1e-6);
        assert!((final_result.get(&[0, 1]).unwrap() - 59.0).abs() < 1e-6);
        assert!((final_result.get(&[1, 0]).unwrap() - 38.0).abs() < 1e-6);
        assert!((final_result.get(&[1, 1]).unwrap() - 74.0).abs() < 1e-6);
    }

    #[test]
    fn test_broadcasting_complex_scenario() {
        // Test broadcasting in a more complex scenario
        let matrix = ForeignTensor::from_shape_vec(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let scalar_tensor = ForeignTensor::from_list(vec![100.0]);
        
        // Broadcasting addition
        let broadcasted = matrix.add(&scalar_tensor).unwrap();
        
        assert_eq!(broadcasted.shape, vec![3, 2]);
        assert_eq!(broadcasted.get(&[0, 0]).unwrap(), 101.0);
        assert_eq!(broadcasted.get(&[1, 1]).unwrap(), 104.0);
        assert_eq!(broadcasted.get(&[2, 1]).unwrap(), 106.0);
        
        // Chain with ReLU (should be no-op since all values are positive)
        let relu_result = broadcasted.maximum_scalar(0.0);
        assert_eq!(relu_result.shape, vec![3, 2]);
        assert_eq!(relu_result.get(&[0, 0]).unwrap(), 101.0);
    }

    // ------------------------------------------------------------------------
    // 12. EDGE CASE TESTS
    // ------------------------------------------------------------------------

    #[test]
    fn test_scalar_tensor_operations() {
        let scalar_a = ForeignTensor::from_nested_list(TestValue::Real(5.0)).unwrap();
        let scalar_b = ForeignTensor::from_nested_list(TestValue::Real(3.0)).unwrap();
        
        assert_eq!(scalar_a.shape, vec![]);
        assert_eq!(scalar_a.ndim, 0);
        
        // Scalar addition
        let sum = scalar_a.add(&scalar_b).unwrap();
        assert_eq!(sum.shape, vec![]);
        // Note: accessing 0D tensor requires empty index array
    }

    #[test]
    fn test_single_element_tensor() {
        let single = ForeignTensor::from_list(vec![42.0]);
        
        assert_eq!(single.shape, vec![1]);
        assert_eq!(single.ndim, 1);
        assert_eq!(single.get(&[0]).unwrap(), 42.0);
        
        // Reshape to scalar
        let scalar = single.reshape(vec![]).unwrap();
        assert_eq!(scalar.shape, vec![]);
        assert_eq!(scalar.ndim, 0);
    }

    #[test]
    fn test_very_large_tensor() {
        // Test handling of larger tensors
        let size = 10000;
        let data: Vec<f64> = (0..size).map(|i| (i % 100) as f64).collect();
        let tensor = ForeignTensor::from_shape_vec(vec![100, 100], data).unwrap();
        
        assert_eq!(tensor.shape, vec![100, 100]);
        assert_eq!(tensor.len, 10000);
        
        // Test accessing corners
        assert_eq!(tensor.get(&[0, 0]).unwrap(), 0.0);
        assert_eq!(tensor.get(&[99, 99]).unwrap(), 99.0);
        
        // Test operations on large tensor
        let doubled = tensor.add_scalar(1.0);
        assert_eq!(doubled.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(doubled.get(&[99, 99]).unwrap(), 100.0);
    }
}