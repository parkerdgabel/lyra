use lyra::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
    stdlib::data::SeriesType,
};
use std::any::Any;
use std::sync::Arc;

/// Test-only Value type that is Send + Sync for TDD tests
#[derive(Debug, Clone, PartialEq)]
pub enum TestValue {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    Missing,
}

impl TestValue {
    /// Convert TestValue to real Value for integration
    pub fn to_value(&self) -> Value {
        match self {
            TestValue::Integer(i) => Value::Integer(*i),
            TestValue::Real(f) => Value::Real(*f),
            TestValue::String(s) => Value::String(s.clone()),
            TestValue::Boolean(b) => Value::Boolean(*b),
            TestValue::Missing => Value::Missing,
        }
    }
    
    /// Convert real Value to TestValue for testing
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Integer(i) => TestValue::Integer(*i),
            Value::Real(f) => TestValue::Real(*f),
            Value::String(s) => TestValue::String(s.clone()),
            Value::Boolean(b) => TestValue::Boolean(*b),
            Value::Missing => TestValue::Missing,
            _ => TestValue::String(format!("Unsupported: {:?}", value)),
        }
    }
}

/// Foreign Series implementation that will replace Value::Series
/// This struct mirrors the existing Series but is Send + Sync for Foreign objects
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignSeries {
    pub data: Arc<Vec<TestValue>>,
    pub dtype: SeriesType,
    pub length: usize,
}

impl ForeignSeries {
    /// Create a new ForeignSeries with type validation
    pub fn new(data: Vec<TestValue>, dtype: SeriesType) -> VmResult<Self> {
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
    pub fn infer(data: Vec<TestValue>) -> VmResult<Self> {
        if data.is_empty() {
            return Err(VmError::TypeError {
                expected: "non-empty data".to_string(),
                actual: "empty data".to_string(),
            });
        }
        
        // Infer type from first non-Missing element
        let dtype = data.iter()
            .find(|v| !matches!(v, TestValue::Missing))
            .map(|v| Self::infer_type(v))
            .unwrap_or(SeriesType::String); // Default to String if all Missing
            
        Self::new(data, dtype)
    }
    
    /// Get a value at index (bounds-checked)
    pub fn get(&self, index: usize) -> VmResult<&TestValue> {
        if index >= self.length {
            return Err(VmError::IndexError {
                index: index as i64,
                length: self.length,
            });
        }
        Ok(&self.data[index])
    }
    
    /// Create a new Series with a modified value (COW semantics)
    pub fn with_value_at(&self, index: usize, value: TestValue) -> VmResult<Self> {
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
    pub fn append(&self, value: TestValue) -> VmResult<Self> {
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
    fn value_matches_type(value: &TestValue, dtype: &SeriesType) -> bool {
        match (value, dtype) {
            (TestValue::Integer(_), SeriesType::Int64) => true,
            (TestValue::Real(_), SeriesType::Float64) => true,
            (TestValue::String(_), SeriesType::String) => true,
            (TestValue::Boolean(_), SeriesType::Bool) => true,
            (TestValue::Missing, _) => true, // Missing is compatible with all types
            _ => false,
        }
    }
    
    /// Infer the type of a single value
    fn infer_type(value: &TestValue) -> SeriesType {
        match value {
            TestValue::Integer(_) => SeriesType::Int64,
            TestValue::Real(_) => SeriesType::Float64,
            TestValue::String(_) => SeriesType::String,
            TestValue::Boolean(_) => SeriesType::Bool,
            TestValue::Missing => SeriesType::String, // Default for Missing
        }
    }
    
    /// Get the underlying data as Value vector for integration
    pub fn to_values(&self) -> Vec<Value> {
        self.data.iter().map(|v| v.to_value()).collect()
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
                        Ok(value.to_value())
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
                        let test_value = TestValue::from_value(&args[1]);
                        let new_series = self.with_value_at(idx, test_value).map_err(|e| ForeignError::RuntimeError {
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
                let test_value = TestValue::from_value(&args[0]);
                let new_series = self.append(test_value).map_err(|e| ForeignError::RuntimeError {
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

// ==========================================
// TDD Tests for Foreign Series Implementation
// ==========================================

/// Helper function to create a simple test series
fn create_test_integer_series() -> ForeignSeries {
    ForeignSeries::new(
        vec![TestValue::Integer(1), TestValue::Integer(2), TestValue::Integer(3)],
        SeriesType::Int64
    ).unwrap()
}

/// Helper function to create a mixed type series for testing
fn create_test_string_series() -> ForeignSeries {
    ForeignSeries::new(
        vec![
            TestValue::String("hello".to_string()),
            TestValue::String("world".to_string()),
            TestValue::String("test".to_string())
        ],
        SeriesType::String
    ).unwrap()
}

/// Helper function to create a series with missing values
fn create_test_series_with_missing() -> ForeignSeries {
    ForeignSeries::new(
        vec![TestValue::Integer(1), TestValue::Missing, TestValue::Integer(3)],
        SeriesType::Int64
    ).unwrap()
}

// ==========================================
// Basic Foreign Series Tests
// ==========================================

#[test]
fn test_foreign_series_creation() {
    let series = create_test_integer_series();
    
    assert_eq!(series.length, 3);
    assert_eq!(series.dtype, SeriesType::Int64);
    assert_eq!(*series.get(0).unwrap(), TestValue::Integer(1));
    assert_eq!(*series.get(1).unwrap(), TestValue::Integer(2));
    assert_eq!(*series.get(2).unwrap(), TestValue::Integer(3));
}

#[test]
fn test_foreign_series_type_inference() {
    // Test integer inference
    let int_series = ForeignSeries::infer(
        vec![TestValue::Integer(1), TestValue::Integer(2)]
    ).unwrap();
    assert_eq!(int_series.dtype, SeriesType::Int64);
    
    // Test string inference
    let str_series = ForeignSeries::infer(
        vec![TestValue::String("a".to_string()), TestValue::String("b".to_string())]
    ).unwrap();
    assert_eq!(str_series.dtype, SeriesType::String);
    
    // Test boolean inference
    let bool_series = ForeignSeries::infer(
        vec![TestValue::Boolean(true), TestValue::Boolean(false)]
    ).unwrap();
    assert_eq!(bool_series.dtype, SeriesType::Bool);
}

#[test]
fn test_foreign_series_type_validation() {
    // Should fail when mixing types
    let result = ForeignSeries::new(
        vec![TestValue::Integer(1), TestValue::String("hello".to_string())],
        SeriesType::Int64
    );
    assert!(result.is_err());
}

#[test]
fn test_foreign_series_empty_creation() {
    let result = ForeignSeries::infer(vec![]);
    assert!(result.is_err());
}

#[test]
fn test_foreign_series_with_missing_values() {
    let series = create_test_series_with_missing();
    
    assert_eq!(series.length, 3);
    assert_eq!(*series.get(0).unwrap(), TestValue::Integer(1));
    assert_eq!(*series.get(1).unwrap(), TestValue::Missing);
    assert_eq!(*series.get(2).unwrap(), TestValue::Integer(3));
}

#[test]
fn test_foreign_series_get_bounds_checking() {
    let series = create_test_integer_series();
    
    // Valid access
    assert!(series.get(0).is_ok());
    assert!(series.get(2).is_ok());
    
    // Invalid access
    assert!(series.get(3).is_err());
    assert!(series.get(100).is_err());
}

#[test]
fn test_foreign_series_with_value_at() {
    let series = create_test_integer_series();
    
    // Valid modification
    let new_series = series.with_value_at(1, TestValue::Integer(42)).unwrap();
    assert_eq!(*new_series.get(0).unwrap(), TestValue::Integer(1));
    assert_eq!(*new_series.get(1).unwrap(), TestValue::Integer(42));
    assert_eq!(*new_series.get(2).unwrap(), TestValue::Integer(3));
    
    // Original should be unchanged
    assert_eq!(*series.get(1).unwrap(), TestValue::Integer(2));
    
    // Invalid index
    assert!(series.with_value_at(10, TestValue::Integer(42)).is_err());
    
    // Invalid type
    assert!(series.with_value_at(1, TestValue::String("invalid".to_string())).is_err());
}

#[test]
fn test_foreign_series_append() {
    let series = create_test_integer_series();
    
    // Valid append
    let new_series = series.append(TestValue::Integer(4)).unwrap();
    assert_eq!(new_series.length, 4);
    assert_eq!(*new_series.get(3).unwrap(), TestValue::Integer(4));
    
    // Original should be unchanged
    assert_eq!(series.length, 3);
    
    // Invalid type append
    assert!(series.append(TestValue::String("invalid".to_string())).is_err());
}

#[test]
fn test_foreign_series_slice() {
    let series = create_test_integer_series();
    
    // Valid slice
    let sliced = series.slice(1, 3).unwrap();
    assert_eq!(sliced.length, 2);
    assert_eq!(*sliced.get(0).unwrap(), TestValue::Integer(2));
    assert_eq!(*sliced.get(1).unwrap(), TestValue::Integer(3));
    
    // Empty slice
    let empty_slice = series.slice(1, 1).unwrap();
    assert_eq!(empty_slice.length, 0);
    
    // Invalid slice bounds
    assert!(series.slice(2, 1).is_err()); // start > end
    assert!(series.slice(0, 10).is_err()); // end > length
}

// ==========================================
// Foreign Trait Method Tests
// ==========================================

#[test]
fn test_foreign_series_length_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("Length", &[]).unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_foreign_series_type_method() {
    let int_series = create_test_integer_series();
    let int_obj = LyObj::new(Box::new(int_series));
    
    let result = int_obj.call_method("Type", &[]).unwrap();
    assert_eq!(result, Value::String("Integer".to_string()));
    
    let str_series = create_test_string_series();
    let str_obj = LyObj::new(Box::new(str_series));
    
    let result = str_obj.call_method("Type", &[]).unwrap();
    assert_eq!(result, Value::String("String".to_string()));
}

#[test]
fn test_foreign_series_get_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("Get", &[Value::Integer(1)]).unwrap();
    assert_eq!(result, Value::Integer(2));
    
    // Test bounds checking
    let result = series_obj.call_method("Get", &[Value::Integer(10)]);
    assert!(result.is_err());
    
    // Test invalid argument type
    let result = series_obj.call_method("Get", &[Value::String("invalid".to_string())]);
    assert!(result.is_err());
}

#[test]
fn test_foreign_series_set_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("Set", &[Value::Integer(1), Value::Integer(42)]).unwrap();
    
    match result {
        Value::LyObj(new_series_obj) => {
            let get_result = new_series_obj.call_method("Get", &[Value::Integer(1)]).unwrap();
            assert_eq!(get_result, Value::Integer(42));
        }
        _ => panic!("Expected LyObj result"),
    }
    
    // Test original unchanged
    let original_result = series_obj.call_method("Get", &[Value::Integer(1)]).unwrap();
    assert_eq!(original_result, Value::Integer(2));
}

#[test]
fn test_foreign_series_append_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("Append", &[Value::Integer(4)]).unwrap();
    
    match result {
        Value::LyObj(new_series_obj) => {
            let length_result = new_series_obj.call_method("Length", &[]).unwrap();
            assert_eq!(length_result, Value::Integer(4));
            
            let get_result = new_series_obj.call_method("Get", &[Value::Integer(3)]).unwrap();
            assert_eq!(get_result, Value::Integer(4));
        }
        _ => panic!("Expected LyObj result"),
    }
}

#[test]
fn test_foreign_series_slice_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("Slice", &[Value::Integer(1), Value::Integer(3)]).unwrap();
    
    match result {
        Value::LyObj(sliced_obj) => {
            let length_result = sliced_obj.call_method("Length", &[]).unwrap();
            assert_eq!(length_result, Value::Integer(2));
            
            let get_result = sliced_obj.call_method("Get", &[Value::Integer(0)]).unwrap();
            assert_eq!(get_result, Value::Integer(2));
        }
        _ => panic!("Expected LyObj result"),
    }
}

#[test]
fn test_foreign_series_to_list_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("ToList", &[]).unwrap();
    
    match result {
        Value::List(values) => {
            assert_eq!(values.len(), 3);
            assert_eq!(values[0], Value::Integer(1));
            assert_eq!(values[1], Value::Integer(2));
            assert_eq!(values[2], Value::Integer(3));
        }
        _ => panic!("Expected List result"),
    }
}

#[test]
fn test_foreign_series_is_empty_method() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    let result = series_obj.call_method("IsEmpty", &[]).unwrap();
    assert_eq!(result, Value::Boolean(false));
    
    // Test with empty slice
    let slice_result = series_obj.call_method("Slice", &[Value::Integer(1), Value::Integer(1)]).unwrap();
    match slice_result {
        Value::LyObj(empty_obj) => {
            let empty_result = empty_obj.call_method("IsEmpty", &[]).unwrap();
            assert_eq!(empty_result, Value::Boolean(true));
        }
        _ => panic!("Expected LyObj result"),
    }
}

// ==========================================
// Error Handling Tests
// ==========================================

#[test]
fn test_foreign_series_method_error_handling() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    // Test unknown method
    let result = series_obj.call_method("UnknownMethod", &[]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::UnknownMethod { method, .. } => {
            assert_eq!(method, "UnknownMethod");
        }
        _ => panic!("Expected UnknownMethod error"),
    }
    
    // Test invalid arity
    let result = series_obj.call_method("Length", &[Value::Integer(1)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::InvalidArity { method, expected, actual } => {
            assert_eq!(method, "Length");
            assert_eq!(expected, 0);
            assert_eq!(actual, 1);
        }
        _ => panic!("Expected InvalidArity error"),
    }
}

// ==========================================
// Integration Tests
// ==========================================

#[test]
fn test_foreign_series_value_integration() {
    let series = create_test_integer_series();
    let series_value = Value::LyObj(LyObj::new(Box::new(series)));
    
    // Test that it can be cloned
    let cloned_value = series_value.clone();
    assert_eq!(series_value, cloned_value);
    
    // Test that it can be pattern matched
    match series_value {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "Series");
        }
        _ => panic!("Expected LyObj variant"),
    }
}

#[test]
fn test_foreign_series_type_safety() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    // Verify type name
    assert_eq!(series_obj.type_name(), "Series");
    
    // Verify safe downcasting
    let foreign_ref = series_obj.as_foreign();
    let series_ref = foreign_ref.as_any().downcast_ref::<ForeignSeries>();
    assert!(series_ref.is_some());
    
    let series_ref = series_ref.unwrap();
    assert_eq!(series_ref.length, 3);
    assert_eq!(series_ref.dtype, SeriesType::Int64);
}

// ==========================================
// Performance Tests
// ==========================================

#[test]
fn test_foreign_series_large_operations() {
    // Create a large series for performance testing
    let large_data: Vec<TestValue> = (0..10000).map(|i| TestValue::Integer(i)).collect();
    let large_series = ForeignSeries::new(large_data, SeriesType::Int64).unwrap();
    let series_obj = LyObj::new(Box::new(large_series));
    
    let start = std::time::Instant::now();
    
    // Perform operations on large series
    let _length = series_obj.call_method("Length", &[]).unwrap();
    let _first = series_obj.call_method("Get", &[Value::Integer(0)]).unwrap();
    let _last = series_obj.call_method("Get", &[Value::Integer(9999)]).unwrap();
    let _slice = series_obj.call_method("Slice", &[Value::Integer(1000), Value::Integer(2000)]).unwrap();
    
    let duration = start.elapsed();
    
    // Should complete operations quickly (< 5ms)
    assert!(duration.as_millis() < 5, "Series operations too slow: {:?}", duration);
}

// ==========================================
// Edge Cases and Boundary Tests
// ==========================================

#[test]
fn test_foreign_series_edge_cases() {
    // Test single element series
    let single_series = ForeignSeries::new(
        vec![TestValue::Integer(42)],
        SeriesType::Int64
    ).unwrap();
    let single_obj = LyObj::new(Box::new(single_series));
    
    let length_result = single_obj.call_method("Length", &[]).unwrap();
    assert_eq!(length_result, Value::Integer(1));
    
    let get_result = single_obj.call_method("Get", &[Value::Integer(0)]).unwrap();
    assert_eq!(get_result, Value::Integer(42));
    
    // Test slice at boundaries
    let slice_result = single_obj.call_method("Slice", &[Value::Integer(0), Value::Integer(1)]).unwrap();
    match slice_result {
        Value::LyObj(sliced_obj) => {
            let sliced_length = sliced_obj.call_method("Length", &[]).unwrap();
            assert_eq!(sliced_length, Value::Integer(1));
        }
        _ => panic!("Expected LyObj result"),
    }
}

#[test]
fn test_foreign_series_cow_semantics() {
    let series = create_test_integer_series();
    let series_obj = LyObj::new(Box::new(series));
    
    // Create multiple modifications
    let modified1 = series_obj.call_method("Set", &[Value::Integer(0), Value::Integer(100)]).unwrap();
    let modified2 = series_obj.call_method("Set", &[Value::Integer(1), Value::Integer(200)]).unwrap();
    
    // Original should be unchanged
    let original_val = series_obj.call_method("Get", &[Value::Integer(0)]).unwrap();
    assert_eq!(original_val, Value::Integer(1));
    
    // Modifications should be independent
    match (modified1, modified2) {
        (Value::LyObj(obj1), Value::LyObj(obj2)) => {
            let val1 = obj1.call_method("Get", &[Value::Integer(0)]).unwrap();
            let val2 = obj2.call_method("Get", &[Value::Integer(1)]).unwrap();
            
            assert_eq!(val1, Value::Integer(100));
            assert_eq!(val2, Value::Integer(200));
        }
        _ => panic!("Expected LyObj results"),
    }
}