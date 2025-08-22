// Example: Creating a Custom TimeSeries Foreign Object
// This example demonstrates how to create a sophisticated Foreign object
// that integrates with Lyra's VM and provides rich functionality.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// A time series data structure for financial and scientific data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Time stamps (as Unix timestamps)
    timestamps: Vec<i64>,
    /// Data values
    values: Vec<f64>,
    /// Optional metadata
    metadata: HashMap<String, String>,
    /// Thread-safe access control
    #[serde(skip)]
    lock: Arc<RwLock<()>>,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(timestamps: Vec<i64>, values: Vec<f64>) -> VmResult<Self> {
        if timestamps.len() != values.len() {
            return Err(VmError::Runtime(
                "Timestamps and values must have same length".to_string()
            ));
        }
        
        Ok(TimeSeries {
            timestamps,
            values,
            metadata: HashMap::new(),
            lock: Arc::new(RwLock::new(())),
        })
    }
    
    /// Create from Lyra values
    pub fn from_lyra_values(timestamps: &[Value], values: &[Value]) -> VmResult<Self> {
        let mut ts_vec = Vec::new();
        let mut val_vec = Vec::new();
        
        // Convert timestamps
        for ts in timestamps {
            match ts {
                Value::Integer(n) => ts_vec.push(*n),
                Value::Real(f) => ts_vec.push(*f as i64),
                _ => return Err(VmError::TypeError {
                    expected: "Integer or Real timestamp".to_string(),
                    actual: format!("{:?}", ts),
                }),
            }
        }
        
        // Convert values
        for val in values {
            match val {
                Value::Integer(n) => val_vec.push(*n as f64),
                Value::Real(f) => val_vec.push(*f),
                _ => return Err(VmError::TypeError {
                    expected: "numeric value".to_string(),
                    actual: format!("{:?}", val),
                }),
            }
        }
        
        Self::new(ts_vec, val_vec)
    }
    
    /// Get length
    pub fn length(&self) -> usize {
        self.values.len()
    }
    
    /// Get value at index
    pub fn get(&self, index: usize) -> Option<(i64, f64)> {
        if index < self.length() {
            Some((self.timestamps[index], self.values[index]))
        } else {
            None
        }
    }
    
    /// Slice the time series
    pub fn slice(&self, start: usize, end: usize) -> VmResult<TimeSeries> {
        if start >= self.length() || end > self.length() || start >= end {
            return Err(VmError::IndexError {
                index: start as i64,
                length: self.length(),
            });
        }
        
        Ok(TimeSeries {
            timestamps: self.timestamps[start..end].to_vec(),
            values: self.values[start..end].to_vec(),
            metadata: self.metadata.clone(),
            lock: Arc::new(RwLock::new(())),
        })
    }
    
    /// Resample the time series
    pub fn resample(&self, window_size: usize, method: &str) -> VmResult<TimeSeries> {
        if window_size == 0 {
            return Err(VmError::Runtime("Window size must be positive".to_string()));
        }
        
        let mut new_timestamps = Vec::new();
        let mut new_values = Vec::new();
        
        for chunk in self.values.chunks(window_size) {
            if !chunk.is_empty() {
                let start_idx = (new_timestamps.len() * window_size).min(self.timestamps.len() - 1);
                new_timestamps.push(self.timestamps[start_idx]);
                
                let aggregated = match method {
                    "mean" => chunk.iter().sum::<f64>() / chunk.len() as f64,
                    "sum" => chunk.iter().sum(),
                    "max" => chunk.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                    "min" => chunk.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                    _ => return Err(VmError::Runtime(format!("Unknown method: {}", method))),
                };
                
                new_values.push(aggregated);
            }
        }
        
        Ok(TimeSeries {
            timestamps: new_timestamps,
            values: new_values,
            metadata: self.metadata.clone(),
            lock: Arc::new(RwLock::new(())),
        })
    }
    
    /// Apply a mathematical operation to all values
    pub fn apply_operation<F>(&self, op: F) -> TimeSeries 
    where
        F: Fn(f64) -> f64 + Send + Sync,
    {
        TimeSeries {
            timestamps: self.timestamps.clone(),
            values: self.values.iter().map(|&v| op(v)).collect(),
            metadata: self.metadata.clone(),
            lock: Arc::new(RwLock::new(())),
        }
    }
}

impl Foreign for TimeSeries {
    fn type_name(&self) -> &'static str {
        "TimeSeries"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let _guard = self.lock.read().map_err(|_| ForeignError::RuntimeError {
            message: "Failed to acquire read lock".to_string(),
        })?;
        
        match method {
            // Basic operations
            "length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.length() as i64))
            }
            
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let index = match &args[0] {
                    Value::Integer(n) => *n as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.get(index) {
                    Some((timestamp, value)) => {
                        Ok(Value::List(vec![
                            Value::Integer(timestamp),
                            Value::Real(value),
                        ]))
                    }
                    None => Err(ForeignError::IndexOutOfBounds {
                        index: index.to_string(),
                        bounds: format!("0..{}", self.length()),
                    }),
                }
            }
            
            "slice" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let start = match &args[0] {
                    Value::Integer(n) => *n as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let end = match &args[1] {
                    Value::Integer(n) => *n as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                match self.slice(start, end) {
                    Ok(sliced) => Ok(Value::LyObj(LyObj::new(Box::new(sliced)))),
                    Err(vm_err) => Err(ForeignError::from(vm_err)),
                }
            }
            
            "resample" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let window_size = match &args[0] {
                    Value::Integer(n) => *n as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let method_name = match &args[1] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                match self.resample(window_size, method_name) {
                    Ok(resampled) => Ok(Value::LyObj(LyObj::new(Box::new(resampled)))),
                    Err(vm_err) => Err(ForeignError::from(vm_err)),
                }
            }
            
            // Mathematical operations
            "square" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let squared = self.apply_operation(|x| x * x);
                Ok(Value::LyObj(LyObj::new(Box::new(squared))))
            }
            
            "sqrt" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let sqrt_ts = self.apply_operation(|x| x.sqrt());
                Ok(Value::LyObj(LyObj::new(Box::new(sqrt_ts))))
            }
            
            "add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::Real(x) => {
                        let added = self.apply_operation(|v| v + x);
                        Ok(Value::LyObj(LyObj::new(Box::new(added))))
                    }
                    Value::Integer(x) => {
                        let x_f = *x as f64;
                        let added = self.apply_operation(|v| v + x_f);
                        Ok(Value::LyObj(LyObj::new(Box::new(added))))
                    }
                    Value::LyObj(other_obj) => {
                        if let Some(other_ts) = other_obj.downcast_ref::<TimeSeries>() {
                            if self.length() != other_ts.length() {
                                return Err(ForeignError::RuntimeError {
                                    message: "TimeSeries must have same length for addition".to_string(),
                                });
                            }
                            
                            let new_values: Vec<f64> = self.values.iter()
                                .zip(other_ts.values.iter())
                                .map(|(a, b)| a + b)
                                .collect();
                            
                            let result = TimeSeries {
                                timestamps: self.timestamps.clone(),
                                values: new_values,
                                metadata: self.metadata.clone(),
                                lock: Arc::new(RwLock::new(())),
                            };
                            
                            Ok(Value::LyObj(LyObj::new(Box::new(result))))
                        } else {
                            Err(ForeignError::TypeError {
                                expected: "TimeSeries".to_string(),
                                actual: other_obj.type_name().to_string(),
                            })
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Real, Integer, or TimeSeries".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            
            // Statistics
            "mean" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                if self.values.is_empty() {
                    return Ok(Value::Missing);
                }
                
                let sum: f64 = self.values.iter().sum();
                Ok(Value::Real(sum / self.values.len() as f64))
            }
            
            "std" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                if self.values.is_empty() {
                    return Ok(Value::Missing);
                }
                
                let mean = self.values.iter().sum::<f64>() / self.values.len() as f64;
                let variance = self.values.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / self.values.len() as f64;
                    
                Ok(Value::Real(variance.sqrt()))
            }
            
            // Metadata operations
            "setMetadata" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let value = match &args[1] {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                
                let mut new_ts = self.clone();
                new_ts.metadata.insert(key, value);
                Ok(Value::LyObj(LyObj::new(Box::new(new_ts))))
            }
            
            "getMetadata" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.metadata.get(key) {
                    Some(value) => Ok(Value::String(value.clone())),
                    None => Ok(Value::Missing),
                }
            }
            
            // Conversion operations
            "toList" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                let pairs: Vec<Value> = self.timestamps.iter()
                    .zip(self.values.iter())
                    .map(|(&ts, &val)| Value::List(vec![
                        Value::Integer(ts),
                        Value::Real(val),
                    ]))
                    .collect();
                
                Ok(Value::List(pairs))
            }
            
            // Introspection
            "__methods__" => {
                let methods = vec![
                    "length", "get", "slice", "resample",
                    "square", "sqrt", "add", "mean", "std",
                    "setMetadata", "getMetadata", "toList"
                ];
                Ok(Value::List(
                    methods.into_iter()
                        .map(|s| Value::String(s.to_string()))
                        .collect()
                ))
            }
            
            "__signature__" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let method_name = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let signature = match method_name.as_str() {
                    "length" => "() -> Integer",
                    "get" => "(Integer) -> List[Integer, Real]",
                    "slice" => "(Integer, Integer) -> TimeSeries",
                    "resample" => "(Integer, String) -> TimeSeries",
                    "square" => "() -> TimeSeries",
                    "sqrt" => "() -> TimeSeries",
                    "add" => "(Real | Integer | TimeSeries) -> TimeSeries",
                    "mean" => "() -> Real",
                    "std" => "() -> Real",
                    "setMetadata" => "(String, String) -> TimeSeries",
                    "getMetadata" => "(String) -> String | Missing",
                    "toList" => "() -> List[List[Integer, Real]]",
                    _ => return Ok(Value::Missing),
                };
                
                Ok(Value::String(signature.to_string()))
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
    
    fn serialize(&self) -> Result<Vec<u8>, ForeignError> {
        serde_json::to_vec(self).map_err(|e| ForeignError::RuntimeError {
            message: format!("Serialization failed: {}", e),
        })
    }
    
    fn deserialize(data: &[u8]) -> Result<Box<dyn Foreign>, ForeignError>
    where
        Self: Sized
    {
        let mut instance: TimeSeries = serde_json::from_slice(data)
            .map_err(|e| ForeignError::RuntimeError {
                message: format!("Deserialization failed: {}", e),
            })?;
        
        // Reinitialize the lock since it's not serialized
        instance.lock = Arc::new(RwLock::new(()));
        
        Ok(Box::new(instance))
    }
}

// Thread safety implementation
unsafe impl Send for TimeSeries {}
unsafe impl Sync for TimeSeries {}

// Stdlib function for creating TimeSeries objects
use crate::stdlib::StdlibFunction;

pub fn create_time_series(args: &[Value]) -> VmResult<Value> {
    match args {
        // TimeSeries[timestamps, values]
        [Value::List(timestamps), Value::List(values)] => {
            let ts = TimeSeries::from_lyra_values(timestamps, values)?;
            Ok(Value::LyObj(LyObj::new(Box::new(ts))))
        }
        
        // TimeSeries[length] - create empty series
        [Value::Integer(length)] => {
            let len = *length as usize;
            if len == 0 {
                return Err(VmError::Runtime("Length must be positive".to_string()));
            }
            
            let timestamps = (0..len as i64).collect();
            let values = vec![0.0; len];
            let ts = TimeSeries::new(timestamps, values)?;
            Ok(Value::LyObj(LyObj::new(Box::new(ts))))
        }
        
        // TimeSeries[] - create empty series
        [] => {
            let ts = TimeSeries::new(Vec::new(), Vec::new())?;
            Ok(Value::LyObj(LyObj::new(Box::new(ts))))
        }
        
        _ => Err(VmError::TypeError {
            expected: "TimeSeries[timestamps, values], TimeSeries[length], or TimeSeries[]".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

// Registration function for stdlib
pub fn register_timeseries_functions() -> Vec<(&'static str, StdlibFunction)> {
    vec![
        ("TimeSeries", create_time_series),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_creation() {
        let timestamps = vec![1, 2, 3, 4, 5];
        let values = vec![1.0, 4.0, 9.0, 16.0, 25.0];
        let ts = TimeSeries::new(timestamps, values).unwrap();
        
        assert_eq!(ts.length(), 5);
        assert_eq!(ts.get(2), Some((3, 9.0)));
    }
    
    #[test]
    fn test_method_calls() {
        let ts = TimeSeries::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        
        // Test length method
        let result = ts.call_method("length", &[]).unwrap();
        assert_eq!(result, Value::Integer(3));
        
        // Test get method
        let result = ts.call_method("get", &[Value::Integer(1)]).unwrap();
        assert_eq!(result, Value::List(vec![Value::Integer(2), Value::Real(2.0)]));
        
        // Test mean method
        let result = ts.call_method("mean", &[]).unwrap();
        assert_eq!(result, Value::Real(2.0));
    }
    
    #[test]
    fn test_mathematical_operations() {
        let ts = TimeSeries::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        
        // Test square operation
        let result = ts.call_method("square", &[]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(squared_ts) = obj.downcast_ref::<TimeSeries>() {
                assert_eq!(squared_ts.values, vec![1.0, 4.0, 9.0]);
            } else {
                panic!("Expected TimeSeries object");
            }
        } else {
            panic!("Expected LyObj result");
        }
    }
    
    #[test]
    fn test_error_handling() {
        let ts = TimeSeries::new(vec![1, 2], vec![1.0, 2.0]).unwrap();
        
        // Test invalid method
        let result = ts.call_method("invalid_method", &[]);
        assert!(matches!(result, Err(ForeignError::UnknownMethod { .. })));
        
        // Test invalid arity
        let result = ts.call_method("length", &[Value::Integer(1)]);
        assert!(matches!(result, Err(ForeignError::InvalidArity { .. })));
        
        // Test out of bounds access
        let result = ts.call_method("get", &[Value::Integer(5)]);
        assert!(matches!(result, Err(ForeignError::IndexOutOfBounds { .. })));
    }
    
    #[test]
    fn test_serialization() {
        let ts = TimeSeries::new(vec![1, 2, 3], vec![1.0, 2.0, 3.0]).unwrap();
        
        // Test serialization
        let serialized = ts.serialize().unwrap();
        assert!(!serialized.is_empty());
        
        // Test deserialization
        let deserialized = TimeSeries::deserialize(&serialized).unwrap();
        let concrete_ts = deserialized.as_any().downcast_ref::<TimeSeries>().unwrap();
        
        assert_eq!(concrete_ts.timestamps, vec![1, 2, 3]);
        assert_eq!(concrete_ts.values, vec![1.0, 2.0, 3.0]);
    }
    
    #[test]
    fn test_stdlib_integration() {
        // Test TimeSeries creation from stdlib function
        let timestamps = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let values = Value::List(vec![
            Value::Real(1.0),
            Value::Real(2.0),
            Value::Real(3.0),
        ]);
        
        let result = create_time_series(&[timestamps, values]).unwrap();
        
        if let Value::LyObj(obj) = result {
            assert_eq!(obj.type_name(), "TimeSeries");
            if let Some(ts) = obj.downcast_ref::<TimeSeries>() {
                assert_eq!(ts.length(), 3);
            }
        } else {
            panic!("Expected LyObj result");
        }
    }
}