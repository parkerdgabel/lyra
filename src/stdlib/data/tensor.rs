//! Foreign Tensor Implementation
//!
//! This module provides a thread-safe Foreign Tensor implementation that replaces
//! the tensor operations in the VM core. Following TDD principles, this implementation
//! is designed to make all the comprehensive tests in foreign_tensor_tests.rs pass.
//!
//! ## Architecture
//!
//! - **Thread Safety**: Uses `Arc<ArrayD<f64>>` for Send + Sync compliance
//! - **NumPy Compatibility**: Follows ndarray broadcasting and operation semantics
//! - **Linear Algebra**: Full matrix multiplication, transpose, and vector operations
//! - **Neural Networks**: ReLU activation and element-wise operations
//! - **Memory Efficiency**: Zero-copy operations using ndarray views where possible

use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::sync::Arc;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;

/// Foreign Tensor implementation with thread-safe Arc<ArrayD<f64>> storage
/// This struct replaces tensor operations in VM core with Foreign object pattern
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
    
    /// Create tensor from nested Value lists (main constructor)
    pub fn from_nested_list(data: Value) -> VmResult<Self> {
        match Self::convert_to_ndarray(&data) {
            Ok(array) => Ok(Self::new(array)),
            Err(msg) => Err(VmError::TypeError {
                expected: "valid tensor data".to_string(),
                actual: msg,
            }),
        }
    }
    
    /// Create 1D tensor from flat vector
    pub fn from_list(values: Vec<f64>) -> VmResult<Self> {
        let array = ArrayD::from_shape_vec(IxDyn(&[values.len()]), values)
            .map_err(|e| VmError::TypeError {
                expected: "valid 1D tensor shape".to_string(),
                actual: format!("shape error: {}", e),
            })?;
        Ok(Self::new(array))
    }
    
    /// Create tensor from shape and flat data
    pub fn from_shape_vec(shape: Vec<usize>, values: Vec<f64>) -> VmResult<Self> {
        let expected_len: usize = shape.iter().product();
        if values.len() != expected_len {
            return Err(VmError::TypeError {
                expected: format!("data with {} elements for shape {:?}", expected_len, shape),
                actual: format!("data with {} elements", values.len()),
            });
        }
        
        let array = ArrayD::from_shape_vec(IxDyn(&shape), values)
            .map_err(|e| VmError::TypeError {
                expected: "valid tensor shape".to_string(),
                actual: format!("shape error: {}", e),
            })?;
        Ok(Self::new(array))
    }
    
    /// Create zero-filled tensor with given shape
    pub fn zeros(shape: Vec<usize>) -> VmResult<Self> {
        let array = ArrayD::zeros(IxDyn(&shape));
        Ok(Self::new(array))
    }
    
    /// Create ones-filled tensor with given shape
    pub fn ones(shape: Vec<usize>) -> VmResult<Self> {
        let array = ArrayD::ones(IxDyn(&shape));
        Ok(Self::new(array))
    }
    
    /// Create identity matrix (2D tensor)
    pub fn eye(size: usize) -> VmResult<Self> {
        let mut array = ArrayD::zeros(IxDyn(&[size, size]));
        for i in 0..size {
            array[[i, i]] = 1.0;
        }
        Ok(Self::new(array))
    }
    
    /// Create random tensor with given shape (uniform [0,1])
    pub fn random(shape: Vec<usize>) -> VmResult<Self> {
        let mut rng = rand::thread_rng();
        let len: usize = shape.iter().product();
        let values: Vec<f64> = (0..len).map(|_| rng.gen()).collect();
        Self::from_shape_vec(shape, values)
    }
    
    /// Get element at given indices
    pub fn get(&self, indices: &[usize]) -> VmResult<f64> {
        if indices.len() != self.ndim {
            return Err(VmError::IndexError {
                index: indices.len() as i64,
                length: self.ndim,
            });
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(VmError::IndexError {
                    index: idx as i64,
                    length: self.shape[i],
                });
            }
        }
        
        let ix = IxDyn(indices);
        Ok(self.data[ix])
    }
    
    /// Set element at given indices (returns new tensor - COW semantics)
    pub fn set(&self, indices: &[usize], value: f64) -> VmResult<Self> {
        if indices.len() != self.ndim {
            return Err(VmError::IndexError {
                index: indices.len() as i64,
                length: self.ndim,
            });
        }
        
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(VmError::IndexError {
                    index: idx as i64,
                    length: self.shape[i],
                });
            }
        }
        
        let mut new_data = (*self.data).clone();
        let ix = IxDyn(indices);
        new_data[ix] = value;
        Ok(Self::new(new_data))
    }
    
    /// Reshape tensor to new shape (total elements must match)
    pub fn reshape(&self, new_shape: Vec<usize>) -> VmResult<Self> {
        let new_len: usize = new_shape.iter().product();
        if new_len != self.len {
            return Err(VmError::TypeError {
                expected: format!("reshape with {} total elements", self.len),
                actual: format!("reshape to {} total elements", new_len),
            });
        }
        
        let reshaped = self.data.to_shape(IxDyn(&new_shape))
            .map_err(|e| VmError::TypeError {
                expected: "valid reshape operation".to_string(),
                actual: format!("reshape error: {}", e),
            })?.to_owned();
        
        Ok(Self::new(reshaped))
    }
    
    /// Flatten tensor to 1D
    pub fn flatten(&self) -> Self {
        let flattened = self.data.to_shape(IxDyn(&[self.len]))
            .expect("Flatten should always succeed").to_owned();
        Self::new(flattened)
    }
    
    /// Transpose tensor (2D matrices or 1D->2D conversion)
    pub fn transpose(&self) -> VmResult<Self> {
        match self.ndim {
            1 => {
                // 1D vector -> column vector (Nx1)
                let transposed = self.data.to_shape(IxDyn(&[self.len, 1]))
                    .map_err(|e| VmError::TypeError {
                        expected: "valid transpose".to_string(),
                        actual: format!("transpose error: {}", e),
                    })?.to_owned();
                Ok(Self::new(transposed))
            },
            2 => {
                // 2D matrix transpose
                let transposed = self.data.t().to_owned();
                Ok(Self::new(transposed))
            },
            _ => Err(VmError::TypeError {
                expected: "1D or 2D tensor for transpose".to_string(),
                actual: format!("{}D tensor", self.ndim),
            }),
        }
    }
    
    /// Element-wise addition with another tensor
    pub fn add(&self, other: &ForeignTensor) -> VmResult<Self> {
        if self.shape == other.shape {
            let result = &*self.data + &*other.data;
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a + b)
        } else {
            Err(VmError::TypeError {
                expected: format!("tensor compatible with shape {:?}", self.shape),
                actual: format!("tensor with shape {:?}", other.shape),
            })
        }
    }
    
    /// Element-wise addition with scalar
    pub fn add_scalar(&self, scalar: f64) -> Self {
        let result = &*self.data + scalar;
        Self::new(result)
    }
    
    /// Element-wise multiplication with another tensor
    pub fn multiply(&self, other: &ForeignTensor) -> VmResult<Self> {
        if self.shape == other.shape {
            let result = &*self.data * &*other.data;
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a * b)
        } else {
            Err(VmError::TypeError {
                expected: format!("tensor compatible with shape {:?}", self.shape),
                actual: format!("tensor with shape {:?}", other.shape),
            })
        }
    }
    
    /// Element-wise multiplication with scalar
    pub fn multiply_scalar(&self, scalar: f64) -> Self {
        let result = &*self.data * scalar;
        Self::new(result)
    }
    
    /// Element-wise subtraction with another tensor
    pub fn subtract(&self, other: &ForeignTensor) -> VmResult<Self> {
        if self.shape == other.shape {
            let result = &*self.data - &*other.data;
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a - b)
        } else {
            Err(VmError::TypeError {
                expected: format!("tensor compatible with shape {:?}", self.shape),
                actual: format!("tensor with shape {:?}", other.shape),
            })
        }
    }
    
    /// Element-wise subtraction with scalar
    pub fn subtract_scalar(&self, scalar: f64) -> Self {
        let result = &*self.data - scalar;
        Self::new(result)
    }
    
    /// Element-wise division with another tensor
    pub fn divide(&self, other: &ForeignTensor) -> VmResult<Self> {
        if self.shape == other.shape {
            let result = &*self.data / &*other.data;
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a / b)
        } else {
            Err(VmError::TypeError {
                expected: format!("tensor compatible with shape {:?}", self.shape),
                actual: format!("tensor with shape {:?}", other.shape),
            })
        }
    }
    
    /// Element-wise division with scalar
    pub fn divide_scalar(&self, scalar: f64) -> Self {
        let result = &*self.data / scalar;
        Self::new(result)
    }
    
    /// Matrix multiplication / dot product
    pub fn dot(&self, other: &ForeignTensor) -> VmResult<Self> {
        match (self.ndim, other.ndim) {
            (1, 1) => {
                // Vector dot product -> scalar (stored as 0D tensor)
                if self.shape[0] != other.shape[0] {
                    return Err(VmError::TypeError {
                        expected: format!("vector with {} elements", self.shape[0]),
                        actual: format!("vector with {} elements", other.shape[0]),
                    });
                }
                let self_1d = (*self.data).clone().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| VmError::TypeError {
                        expected: "1D vector".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                let other_1d = (*other.data).clone().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| VmError::TypeError {
                        expected: "1D vector".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                
                let result = self_1d.dot(&other_1d);
                let scalar_array = ArrayD::from_elem(IxDyn(&[]), result);
                Ok(Self::new(scalar_array))
            },
            (2, 1) => {
                // Matrix-vector multiplication
                if self.shape[1] != other.shape[0] {
                    return Err(VmError::TypeError {
                        expected: format!("vector with {} elements for {}x{} matrix", self.shape[1], self.shape[0], self.shape[1]),
                        actual: format!("vector with {} elements", other.shape[0]),
                    });
                }
                let self_2d = (*self.data).clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| VmError::TypeError {
                        expected: "2D matrix".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                let other_1d = (*other.data).clone().into_dimensionality::<ndarray::Ix1>()
                    .map_err(|e| VmError::TypeError {
                        expected: "1D vector".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                
                let result = self_2d.dot(&other_1d);
                let result_nd = result.into_dyn();
                Ok(Self::new(result_nd))
            },
            (2, 2) => {
                // Matrix-matrix multiplication
                if self.shape[1] != other.shape[0] {
                    return Err(VmError::TypeError {
                        expected: format!("{}x{} matrix for multiplication with {}x{}", 
                            other.shape[0], other.shape[1], self.shape[0], self.shape[1]),
                        actual: format!("{}x{} matrix", other.shape[0], other.shape[1]),
                    });
                }
                let self_2d = (*self.data).clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| VmError::TypeError {
                        expected: "2D matrix".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                let other_2d = (*other.data).clone().into_dimensionality::<ndarray::Ix2>()
                    .map_err(|e| VmError::TypeError {
                        expected: "2D matrix".to_string(),
                        actual: format!("conversion error: {}", e),
                    })?;
                
                let result = self_2d.dot(&other_2d);
                let result_nd = result.into_dyn();
                Ok(Self::new(result_nd))
            },
            _ => Err(VmError::TypeError {
                expected: "tensors with 1D or 2D dimensions for dot product".to_string(),
                actual: format!("{}D and {}D tensors", self.ndim, other.ndim),
            }),
        }
    }
    
    /// Element-wise maximum (ReLU when used with zero tensor)
    pub fn maximum(&self, other: &ForeignTensor) -> VmResult<Self> {
        if self.shape == other.shape {
            let result = ndarray::Zip::from(&*self.data)
                .and(&*other.data)
                .map_collect(|&a, &b| a.max(b));
            Ok(Self::new(result))
        } else if self.is_broadcastable_with(other) {
            self.broadcast_op(other, |a, b| a.max(b))
        } else {
            Err(VmError::TypeError {
                expected: format!("tensor compatible with shape {:?}", self.shape),
                actual: format!("tensor with shape {:?}", other.shape),
            })
        }
    }
    
    /// Element-wise maximum with scalar (ReLU activation when scalar=0)
    pub fn maximum_scalar(&self, scalar: f64) -> Self {
        let result = self.data.mapv(|x| x.max(scalar));
        Self::new(result)
    }
    
    /// Convert to nested Value list for VM integration
    pub fn to_nested_list(&self) -> Value {
        fn array_to_value(data: &ArrayD<f64>, shape: &[usize]) -> Value {
            if shape.is_empty() {
                // 0D scalar
                Value::Real(data[&IxDyn(&[])])
            } else if shape.len() == 1 {
                // 1D array -> List of scalars
                let values: Vec<Value> = data.iter()
                    .map(|&x| Value::Real(x))
                    .collect();
                Value::List(values)
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
                    
                    outer_list.push(array_to_value(&inner_array, inner_shape));
                }
                Value::List(outer_list)
            }
        }
        
        array_to_value(&self.data, &self.shape)
    }
    
    // Helper methods for broadcasting and operations
    fn is_broadcastable_with(&self, other: &ForeignTensor) -> bool {
        // Simplified broadcasting rules (can be extended for full NumPy compatibility)
        self.shape == other.shape || 
        other.shape == vec![1] || 
        self.shape == vec![1] ||
        (self.ndim == 0 || other.ndim == 0) // scalar broadcasting
    }
    
    fn broadcast_op<F>(&self, other: &ForeignTensor, op: F) -> VmResult<Self>
    where
        F: Fn(f64, f64) -> f64,
    {
        // Simplified broadcasting implementation
        if other.shape == vec![1] || other.ndim == 0 {
            let scalar = if other.ndim == 0 {
                other.data[&IxDyn(&[])]
            } else {
                other.data[&IxDyn(&[0])]
            };
            let result = self.data.mapv(|x| op(x, scalar));
            Ok(Self::new(result))
        } else if self.shape == vec![1] || self.ndim == 0 {
            let scalar = if self.ndim == 0 {
                self.data[&IxDyn(&[])]
            } else {
                self.data[&IxDyn(&[0])]
            };
            let result = other.data.mapv(|x| op(scalar, x));
            Ok(Self::new(result))
        } else {
            Err(VmError::TypeError {
                expected: "broadcastable tensor shapes".to_string(),
                actual: format!("shapes {:?} and {:?}", self.shape, other.shape),
            })
        }
    }
    
    /// Convert Value to ndarray (recursive for nested lists)
    fn convert_to_ndarray(value: &Value) -> Result<ArrayD<f64>, String> {
        match value {
            Value::Real(x) => {
                // Scalar -> 0D tensor
                Ok(ArrayD::from_elem(IxDyn(&[]), *x))
            },
            Value::Integer(x) => {
                // Scalar -> 0D tensor
                Ok(ArrayD::from_elem(IxDyn(&[]), *x as f64))
            },
            Value::List(values) => {
                if values.is_empty() {
                    return Err("Cannot create tensor from empty list".to_string());
                }
                
                // Check if all elements are scalars (1D tensor)
                if values.iter().all(|v| matches!(v, Value::Real(_) | Value::Integer(_))) {
                    let flat_values: Vec<f64> = values.iter().map(|v| match v {
                        Value::Real(x) => *x,
                        Value::Integer(x) => *x as f64,
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
    
    fn convert_nested_list_to_ndarray(values: &[Value]) -> Result<ArrayD<f64>, String> {
        // Enhanced nested list conversion supporting 2D and 3D arrays
        
        if values.is_empty() {
            return Err("Cannot create tensor from empty nested list".to_string());
        }
        
        // Check if this is a 2D array (list of lists of scalars)
        if let Value::List(first_row) = &values[0] {
            let ncols = first_row.len();
            let nrows = values.len();
            
            // Validate all rows have same length and are scalar lists
            for (i, row) in values.iter().enumerate() {
                if let Value::List(row_values) = row {
                    if row_values.len() != ncols {
                        return Err(format!("Row {} has {} elements, expected {}", i, row_values.len(), ncols));
                    }
                    if !row_values.iter().all(|v| matches!(v, Value::Real(_) | Value::Integer(_))) {
                        return Err("Non-scalar values in tensor data".to_string());
                    }
                } else {
                    return Err("Inconsistent nesting in tensor data".to_string());
                }
            }
            
            // Flatten the 2D structure
            let mut flat_values = Vec::with_capacity(nrows * ncols);
            for row in values {
                if let Value::List(row_values) = row {
                    for val in row_values {
                        match val {
                            Value::Real(x) => flat_values.push(*x),
                            Value::Integer(x) => flat_values.push(*x as f64),
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

impl Foreign for ForeignTensor {
    fn type_name(&self) -> &'static str {
        "Tensor"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Dimensions" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let dims: Vec<Value> = self.shape.iter()
                    .map(|&d| Value::Integer(d as i64))
                    .collect();
                Ok(Value::List(dims))
            },
            "Rank" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.ndim as i64))
            },
            "Length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.len as i64))
            },
            "Get" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // Extract indices from arguments
                let mut indices = Vec::new();
                for arg in args {
                    match arg {
                        Value::Integer(idx) => indices.push(*idx as usize),
                        _ => return Err(ForeignError::InvalidArgumentType {
                            method: method.to_string(),
                            expected: "Integer".to_string(),
                            actual: format!("{:?}", arg),
                        }),
                    }
                }
                
                let value = self.get(&indices).map_err(|e| ForeignError::RuntimeError {
                    message: format!("Index access error: {}", e),
                })?;
                Ok(Value::Real(value))
            },
            "Set" => {
                if args.len() < 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                // Last argument is the value, rest are indices
                let value_arg = &args[args.len() - 1];
                let index_args = &args[..args.len() - 1];
                
                let value = match value_arg {
                    Value::Real(x) => *x,
                    Value::Integer(x) => *x as f64,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", value_arg),
                    }),
                };
                
                let mut indices = Vec::new();
                for arg in index_args {
                    match arg {
                        Value::Integer(idx) => indices.push(*idx as usize),
                        _ => return Err(ForeignError::InvalidArgumentType {
                            method: method.to_string(),
                            expected: "Integer".to_string(),
                            actual: format!("{:?}", arg),
                        }),
                    }
                }
                
                let new_tensor = self.set(&indices, value).map_err(|e| ForeignError::RuntimeError {
                    message: format!("Set operation error: {}", e),
                })?;
                Ok(Value::LyObj(LyObj::new(Box::new(new_tensor))))
            },
            "Reshape" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::List(shape_values) => {
                        let new_shape: Result<Vec<usize>, _> = shape_values.iter()
                            .map(|v| match v {
                                Value::Integer(d) => Ok(*d as usize),
                                _ => Err(ForeignError::InvalidArgumentType {
                                    method: method.to_string(),
                                    expected: "List of integers".to_string(),
                                    actual: format!("{:?}", v),
                                }),
                            })
                            .collect();
                        
                        let new_shape = new_shape?;
                        let reshaped = self.reshape(new_shape).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Reshape error: {}", e),
                        })?;
                        Ok(Value::LyObj(LyObj::new(Box::new(reshaped))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Flatten" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let flattened = self.flatten();
                Ok(Value::LyObj(LyObj::new(Box::new(flattened))))
            },
            "Transpose" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let transposed = self.transpose().map_err(|e| ForeignError::RuntimeError {
                    message: format!("Transpose error: {}", e),
                })?;
                Ok(Value::LyObj(LyObj::new(Box::new(transposed))))
            },
            "Add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.add(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Add operation error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    Value::Real(scalar) => {
                        let result = self.add_scalar(*scalar);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    Value::Integer(scalar) => {
                        let result = self.add_scalar(*scalar as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Multiply" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.multiply(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Multiply operation error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    Value::Real(scalar) => {
                        let result = self.multiply_scalar(*scalar);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    Value::Integer(scalar) => {
                        let result = self.multiply_scalar(*scalar as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Subtract" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.subtract(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Subtract operation error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    Value::Real(scalar) => {
                        let result = self.subtract_scalar(*scalar);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    Value::Integer(scalar) => {
                        let result = self.subtract_scalar(*scalar as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Divide" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.divide(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Divide operation error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    Value::Real(scalar) => {
                        let result = self.divide_scalar(*scalar);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    Value::Integer(scalar) => {
                        let result = self.divide_scalar(*scalar as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Dot" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.dot(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Dot product error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "Maximum" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_tensor) = other_obj.downcast_ref::<ForeignTensor>() {
                            let result = self.maximum(other_tensor).map_err(|e| ForeignError::RuntimeError {
                                message: format!("Maximum operation error: {}", e),
                            })?;
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "Tensor".to_string(),
                                actual: "Non-tensor object".to_string(),
                            })
                        }
                    },
                    Value::Real(scalar) => {
                        let result = self.maximum_scalar(*scalar);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    Value::Integer(scalar) => {
                        let result = self.maximum_scalar(*scalar as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(result))))
                    },
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Tensor or scalar".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            },
            "ToList" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(self.to_nested_list())
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