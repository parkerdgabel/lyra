use lyra::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::Value,
};
use std::any::Any;

/// Mock Foreign implementation for testing
#[derive(Debug, Clone)]
struct MockTable {
    name: String,
    rows: usize,
    columns: Vec<String>,
}

impl Foreign for MockTable {
    fn type_name(&self) -> &'static str {
        "MockTable"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "rows" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.rows as i64))
            }
            "columns" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.columns.len() as i64))
            }
            "get" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                // Mock implementation: return row + column as integer
                match (&args[0], &args[1]) {
                    (Value::Integer(row), Value::Integer(col)) => {
                        if *row < 0 || *col < 0 || *row as usize >= self.rows || *col as usize >= self.columns.len() {
                            return Err(ForeignError::IndexOutOfBounds {
                                index: format!("({}, {})", row, col),
                                bounds: format!("({}, {})", self.rows, self.columns.len()),
                            });
                        }
                        Ok(Value::Integer(row + col))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args),
                    }),
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

impl PartialEq for MockTable {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.rows == other.rows && self.columns == other.columns
    }
}

// ==========================================
// Core Foreign Trait Tests
// ==========================================

#[test]
fn test_foreign_trait_type_name() {
    let table = MockTable {
        name: "test_table".to_string(),
        rows: 10,
        columns: vec!["col1".to_string(), "col2".to_string()],
    };

    assert_eq!(table.type_name(), "MockTable");
}

#[test]
fn test_foreign_trait_method_call_success() {
    let table = MockTable {
        name: "test_table".to_string(),
        rows: 5,
        columns: vec!["a".to_string(), "b".to_string(), "c".to_string()],
    };

    // Test rows() method
    let result = table.call_method("rows", &[]).unwrap();
    assert_eq!(result, Value::Integer(5));

    // Test columns() method  
    let result = table.call_method("columns", &[]).unwrap();
    assert_eq!(result, Value::Integer(3));

    // Test get(row, col) method
    let result = table.call_method("get", &[Value::Integer(1), Value::Integer(2)]).unwrap();
    assert_eq!(result, Value::Integer(3)); // 1 + 2
}

#[test]
fn test_foreign_trait_method_call_errors() {
    let table = MockTable {
        name: "test_table".to_string(),
        rows: 3,
        columns: vec!["x".to_string()],
    };

    // Test unknown method
    let result = table.call_method("unknown_method", &[]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::UnknownMethod { type_name, method } => {
            assert_eq!(type_name, "MockTable");
            assert_eq!(method, "unknown_method");
        }
        _ => panic!("Expected UnknownMethod error"),
    }

    // Test invalid arity
    let result = table.call_method("rows", &[Value::Integer(1)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::InvalidArity { method, expected, actual } => {
            assert_eq!(method, "rows");
            assert_eq!(expected, 0);
            assert_eq!(actual, 1);
        }
        _ => panic!("Expected InvalidArity error"),
    }

    // Test invalid argument type
    let result = table.call_method("get", &[Value::String("invalid".to_string()), Value::Integer(0)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::InvalidArgumentType { method, expected, actual: _ } => {
            assert_eq!(method, "get");
            assert_eq!(expected, "Integer");
        }
        _ => panic!("Expected InvalidArgumentType error"),
    }

    // Test index out of bounds
    let result = table.call_method("get", &[Value::Integer(10), Value::Integer(0)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::IndexOutOfBounds { index, bounds } => {
            assert_eq!(index, "(10, 0)");
            assert_eq!(bounds, "(3, 1)");
        }
        _ => panic!("Expected IndexOutOfBounds error"),
    }
}

#[test]
fn test_foreign_trait_clone_boxed() {
    let table = MockTable {
        name: "original".to_string(),
        rows: 2,
        columns: vec!["col".to_string()],
    };

    let cloned = table.clone_boxed();
    
    // Should be able to call methods on cloned object
    let result = cloned.call_method("rows", &[]).unwrap();
    assert_eq!(result, Value::Integer(2));
}

#[test]
fn test_foreign_trait_as_any_downcasting() {
    let table = MockTable {
        name: "test".to_string(),
        rows: 1,
        columns: vec![],
    };

    let any_ref = table.as_any();
    let downcast = any_ref.downcast_ref::<MockTable>();
    
    assert!(downcast.is_some());
    let downcast_table = downcast.unwrap();
    assert_eq!(downcast_table.name, "test");
    assert_eq!(downcast_table.rows, 1);
}

// ==========================================
// LyObj Wrapper Tests  
// ==========================================

#[test]
fn test_lyobj_creation() {
    let table = MockTable {
        name: "wrapped".to_string(),
        rows: 4,
        columns: vec!["a".to_string(), "b".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Should be able to access type name
    assert_eq!(lyobj.type_name(), "MockTable");
}

#[test]
fn test_lyobj_method_delegation() {
    let table = MockTable {
        name: "test".to_string(),
        rows: 7,
        columns: vec!["x".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Method calls should be delegated to the wrapped object
    let result = lyobj.call_method("rows", &[]).unwrap();
    assert_eq!(result, Value::Integer(7));
    
    let result = lyobj.call_method("columns", &[]).unwrap();
    assert_eq!(result, Value::Integer(1));
}

#[test]
fn test_lyobj_clone() {
    let table = MockTable {
        name: "cloneable".to_string(),
        rows: 3,
        columns: vec!["test".to_string()],
    };

    let lyobj1 = LyObj::new(Box::new(table));
    let lyobj2 = lyobj1.clone();
    
    // Both objects should work independently
    let result1 = lyobj1.call_method("rows", &[]).unwrap();
    let result2 = lyobj2.call_method("rows", &[]).unwrap();
    
    assert_eq!(result1, Value::Integer(3));
    assert_eq!(result2, Value::Integer(3));
}

#[test]
fn test_lyobj_partial_eq() {
    let table1 = MockTable {
        name: "same".to_string(),
        rows: 2,
        columns: vec!["col".to_string()],
    };
    let table2 = MockTable {
        name: "same".to_string(),
        rows: 2,
        columns: vec!["col".to_string()],
    };
    let table3 = MockTable {
        name: "different".to_string(),
        rows: 2,
        columns: vec!["col".to_string()],
    };

    let lyobj1 = LyObj::new(Box::new(table1));
    let lyobj2 = LyObj::new(Box::new(table2));
    let lyobj3 = LyObj::new(Box::new(table3));
    
    assert_eq!(lyobj1, lyobj2);
    assert_ne!(lyobj1, lyobj3);
}

#[test]
fn test_lyobj_debug_display() {
    let table = MockTable {
        name: "debug_test".to_string(),
        rows: 1,
        columns: vec![],
    };

    let lyobj = LyObj::new(Box::new(table));
    let debug_str = format!("{:?}", lyobj);
    
    // Should include type name in debug output
    assert!(debug_str.contains("MockTable"));
}

// ==========================================
// Value Enum Integration Tests
// ==========================================

#[test]
fn test_value_lyobj_variant() {
    let table = MockTable {
        name: "in_value".to_string(),
        rows: 5,
        columns: vec!["test".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    let value = Value::LyObj(lyobj);
    
    // Should be able to pattern match on Value
    match value {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "MockTable");
        }
        _ => panic!("Expected LyObj variant"),
    }
}

#[test]
fn test_value_lyobj_equality() {
    let table1 = MockTable {
        name: "equal".to_string(),
        rows: 1,
        columns: vec![],
    };
    let table2 = MockTable {
        name: "equal".to_string(),
        rows: 1,
        columns: vec![],
    };

    let value1 = Value::LyObj(LyObj::new(Box::new(table1)));
    let value2 = Value::LyObj(LyObj::new(Box::new(table2)));
    
    assert_eq!(value1, value2);
}

#[test]
fn test_value_lyobj_clone() {
    let table = MockTable {
        name: "clone_test".to_string(),
        rows: 2,
        columns: vec!["a".to_string()],
    };

    let value1 = Value::LyObj(LyObj::new(Box::new(table)));
    let value2 = value1.clone();
    
    // Both values should be equal
    assert_eq!(value1, value2);
    
    // Both should work independently
    if let (Value::LyObj(obj1), Value::LyObj(obj2)) = (&value1, &value2) {
        let result1 = obj1.call_method("rows", &[]).unwrap();
        let result2 = obj2.call_method("rows", &[]).unwrap();
        assert_eq!(result1, result2);
    } else {
        panic!("Expected LyObj values");
    }
}

// ==========================================
// Error Handling Tests
// ==========================================

#[test]
fn test_foreign_error_display() {
    let errors = vec![
        ForeignError::UnknownMethod {
            type_name: "TestType".to_string(),
            method: "unknown".to_string(),
        },
        ForeignError::InvalidArity {
            method: "test_method".to_string(),
            expected: 2,
            actual: 1,
        },
        ForeignError::InvalidArgumentType {
            method: "typed_method".to_string(),
            expected: "Integer".to_string(),
            actual: "String".to_string(),
        },
        ForeignError::IndexOutOfBounds {
            index: "5".to_string(),
            bounds: "0..3".to_string(),
        },
        ForeignError::RuntimeError {
            message: "Something went wrong".to_string(),
        },
    ];

    for error in errors {
        let error_str = format!("{}", error);
        assert!(!error_str.is_empty());
        // Each error should have a meaningful message
        assert!(error_str.len() > 10);
    }
}

#[test]
fn test_foreign_error_debug() {
    let error = ForeignError::UnknownMethod {
        type_name: "TestType".to_string(),
        method: "test".to_string(),
    };

    let debug_str = format!("{:?}", error);
    assert!(debug_str.contains("UnknownMethod"));
    assert!(debug_str.contains("TestType"));
    assert!(debug_str.contains("test"));
}

// ==========================================
// Integration Tests with VM Operations
// ==========================================

#[test]
fn test_lyobj_in_vm_stack() {
    // This test will verify that LyObj values work correctly in VM operations
    // like stack push/pop, function calls, etc.
    let table = MockTable {
        name: "vm_test".to_string(),
        rows: 3,
        columns: vec!["id".to_string(), "name".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    let value = Value::LyObj(lyobj);
    
    // Should be able to clone for stack operations
    let cloned = value.clone();
    assert_eq!(value, cloned);
    
    // Should maintain object identity through stack operations
    if let Value::LyObj(obj) = &cloned {
        let result = obj.call_method("rows", &[]).unwrap();
        assert_eq!(result, Value::Integer(3));
    }
}

#[test]
fn test_lyobj_method_call_with_complex_args() {
    let table = MockTable {
        name: "complex_args".to_string(),
        rows: 2,
        columns: vec!["data".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Test method calls with various argument types
    let args = vec![
        Value::Integer(0),
        Value::Integer(0),
    ];
    
    let result = lyobj.call_method("get", &args).unwrap();
    assert_eq!(result, Value::Integer(0)); // 0 + 0
}

// ==========================================
// Performance and Memory Tests
// ==========================================

#[test]
fn test_lyobj_memory_efficiency() {
    use std::mem;
    
    // LyObj should be reasonably sized (single Box pointer)
    let lyobj_size = mem::size_of::<LyObj>();
    let box_size = mem::size_of::<Box<dyn Foreign>>();
    
    assert_eq!(lyobj_size, box_size);
    
    // Value enum should still be reasonably sized with LyObj variant
    let value_size = mem::size_of::<Value>();
    println!("Value enum size: {} bytes", value_size);
    assert!(value_size <= 128); // Value enum is larger due to complex variants like Table/Tensor
}

#[test]
fn test_lyobj_method_call_performance() {
    let table = MockTable {
        name: "perf_test".to_string(),
        rows: 1000,
        columns: vec!["col1".to_string(), "col2".to_string()],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Method calls should be reasonably fast
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let _result = lyobj.call_method("get", &[
            Value::Integer(i % 1000),
            Value::Integer(i % 2),
        ]).unwrap();
    }
    let duration = start.elapsed();
    
    // Should complete 1000 method calls in reasonable time (< 10ms)
    assert!(duration.as_millis() < 10, "Method calls took too long: {:?}", duration);
}

// ==========================================
// Type Safety Tests
// ==========================================

#[test]
fn test_foreign_type_safety() {
    let table = MockTable {
        name: "type_safety".to_string(),
        rows: 1,
        columns: vec![],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Should be able to safely downcast
    let foreign_ref: &dyn Foreign = lyobj.as_foreign();
    let any_ref = foreign_ref.as_any();
    let table_ref = any_ref.downcast_ref::<MockTable>();
    
    assert!(table_ref.is_some());
    assert_eq!(table_ref.unwrap().name, "type_safety");
    
    // Should fail to downcast to wrong type
    #[derive(Debug)]
    struct OtherType;
    let wrong_cast = any_ref.downcast_ref::<OtherType>();
    assert!(wrong_cast.is_none());
}

#[test]
fn test_foreign_thread_safety() {
    // Foreign objects should be Send + Sync if the underlying type is
    // This test verifies the trait bounds work correctly
    let table = MockTable {
        name: "thread_test".to_string(),
        rows: 1,
        columns: vec![],
    };

    let lyobj = LyObj::new(Box::new(table));
    
    // Should be able to send across threads (if MockTable is Send)
    std::thread::spawn(move || {
        let result = lyobj.call_method("rows", &[]).unwrap();
        assert_eq!(result, Value::Integer(1));
    }).join().unwrap();
}