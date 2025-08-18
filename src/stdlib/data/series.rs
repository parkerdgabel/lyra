use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::sync::Arc;

/// Series data type for typed vectors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeriesType {
    Int64,
    Float64,
    Bool,
    String,
}

/// Thread-safe Series implementation for Foreign objects
/// Uses Arc instead of Rc to be Send + Sync compliant
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignSeries {
    pub data: Arc<Vec<Value>>,
    pub dtype: SeriesType,
    pub length: usize,
}

impl ForeignSeries {
    /// Create a new ForeignSeries with type validation
    pub fn new(data: Vec<Value>, dtype: SeriesType) -> VmResult<Self> {
        // Validate all data matches the specified type
        for value in &data {
            if !Self::value_matches_type(value, &dtype) {
                return Err(VmError::TypeError {
                    expected: format!("{:?}", dtype),
                    actual: format!("{:?}", value),
                });
            }
        }
        
        Ok(ForeignSeries {
            length: data.len(),
            data: Arc::new(data),
            dtype,
        })
    }
    
    /// Create a new ForeignSeries with automatic type inference
    pub fn infer(data: Vec<Value>) -> VmResult<Self> {
        if data.is_empty() {
            return Err(VmError::TypeError {
                expected: "non-empty data".to_string(),
                actual: "empty data".to_string(),
            });
        }
        
        // Infer type from first non-Missing element
        let dtype = data.iter()
            .find(|v| !matches!(v, Value::Missing))
            .map(|v| Self::infer_type(v))
            .unwrap_or(SeriesType::String); // Default to String if all Missing
            
        Self::new(data, dtype)
    }
    
    
    /// Get a value at index (bounds-checked)
    pub fn get(&self, index: usize) -> VmResult<&Value> {
        if index >= self.length {
            return Err(VmError::IndexError {
                index: index as i64,
                length: self.length,
            });
        }
        Ok(&self.data[index])
    }
    
    /// Create a new Series with a modified value (COW semantics)
    pub fn with_value_at(&self, index: usize, value: Value) -> VmResult<Self> {
        if index >= self.length {
            return Err(VmError::IndexError {
                index: index as i64,
                length: self.length,
            });
        }
        
        // Validate the new value matches the series type
        if !Self::value_matches_type(&value, &self.dtype) {
            return Err(VmError::TypeError {
                expected: format!("{:?}", self.dtype),
                actual: format!("{:?}", value),
            });
        }
        
        // Clone the data (COW)
        let mut new_data = (*self.data).clone();
        new_data[index] = value;
        
        Ok(ForeignSeries {
            data: Arc::new(new_data),
            dtype: self.dtype.clone(),
            length: self.length,
        })
    }
    
    /// Append a value to create a new Series (COW semantics)
    pub fn append(&self, value: Value) -> VmResult<Self> {
        // Validate the value matches the series type
        if !Self::value_matches_type(&value, &self.dtype) {
            return Err(VmError::TypeError {
                expected: format!("{:?}", self.dtype),
                actual: format!("{:?}", value),
            });
        }
        
        // Clone and extend the data (COW)
        let mut new_data = (*self.data).clone();
        new_data.push(value);
        
        Ok(ForeignSeries {
            data: Arc::new(new_data),
            dtype: self.dtype.clone(),
            length: self.length + 1,
        })
    }
    
    /// Create a view/slice of this Series (COW optimization)
    pub fn slice(&self, start: usize, end: usize) -> VmResult<Self> {
        if start > end || end > self.length {
            return Err(VmError::IndexError {
                index: end as i64,
                length: self.length,
            });
        }
        
        // Create a new vector with the sliced data
        let sliced_data = self.data[start..end].to_vec();
        
        Ok(ForeignSeries {
            data: Arc::new(sliced_data),
            dtype: self.dtype.clone(),
            length: end - start,
        })
    }
    
    /// Check if a value matches the expected type
    fn value_matches_type(value: &Value, dtype: &SeriesType) -> bool {
        match (value, dtype) {
            (Value::Integer(_), SeriesType::Int64) => true,
            (Value::Real(_), SeriesType::Float64) => true,
            (Value::String(_), SeriesType::String) => true,
            (Value::Boolean(_), SeriesType::Bool) => true,
            (Value::Missing, _) => true, // Missing is compatible with all types
            _ => false,
        }
    }
    
    /// Infer the type of a single value
    fn infer_type(value: &Value) -> SeriesType {
        match value {
            Value::Integer(_) => SeriesType::Int64,
            Value::Real(_) => SeriesType::Float64,
            Value::String(_) => SeriesType::String,
            Value::Boolean(_) => SeriesType::Bool,
            Value::Missing => SeriesType::String, // Default for Missing
            _ => SeriesType::String, // Default for complex types
        }
    }
    
    /// Get the underlying data as Value vector for integration
    pub fn to_values(&self) -> Vec<Value> {
        (*self.data).clone()
    }
    
    /// Create a new series filled with a repeated value
    pub fn filled(value: Value, length: usize) -> VmResult<Self> {
        let dtype = Self::infer_type(&value);
        let data = vec![value; length];
        Self::new(data, dtype)
    }
    
    /// Create a range series from start to end (exclusive)
    pub fn range(start: i64, end: i64) -> VmResult<Self> {
        if start >= end {
            return Err(VmError::TypeError {
                expected: "start < end".to_string(),
                actual: format!("start={}, end={}", start, end),
            });
        }
        
        let data: Vec<Value> = (start..end).map(Value::Integer).collect();
        Self::new(data, SeriesType::Int64)
    }
    
    /// Create a zeros series of given length
    pub fn zeros(length: usize) -> VmResult<Self> {
        let data = vec![Value::Integer(0); length];
        Self::new(data, SeriesType::Int64)
    }
    
    /// Create a ones series of given length
    pub fn ones(length: usize) -> VmResult<Self> {
        let data = vec![Value::Integer(1); length];
        Self::new(data, SeriesType::Int64)
    }
}

impl Foreign for ForeignSeries {
    fn type_name(&self) -> &'static str {
        "Series"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.length as i64))
            }
            "Type" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let type_name = match self.dtype {
                    SeriesType::Int64 => "Integer",
                    SeriesType::Float64 => "Real",
                    SeriesType::String => "String",
                    SeriesType::Bool => "Boolean",
                };
                Ok(Value::String(type_name.to_string()))
            }
            "Get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(index) => {
                        let idx = *index as usize;
                        let value = self.get(idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Get operation error: {}", e),
                        })?;
                        Ok(value.clone())
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "Set" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(index) => {
                        let idx = *index as usize;
                        let new_series = self.with_value_at(idx, args[1].clone()).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Set operation error: {}", e),
                        })?;
                        Ok(Value::LyObj(LyObj::new(Box::new(new_series))))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "Append" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let new_series = self.append(args[0].clone()).map_err(|e| ForeignError::RuntimeError {
                    message: format!("Append operation error: {}", e),
                })?;
                Ok(Value::LyObj(LyObj::new(Box::new(new_series))))
            }
            "Slice" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                match (&args[0], &args[1]) {
                    (Value::Integer(start), Value::Integer(end)) => {
                        let start_idx = *start as usize;
                        let end_idx = *end as usize;
                        let sliced_series = self.slice(start_idx, end_idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Slice operation error: {}", e),
                        })?;
                        Ok(Value::LyObj(LyObj::new(Box::new(sliced_series))))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args),
                    }),
                }
            }
            "ToList" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let values = self.to_values();
                Ok(Value::List(values))
            }
            "IsEmpty" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.length == 0))
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