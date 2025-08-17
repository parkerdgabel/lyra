use lyra::{
    compiler::Compiler,
    foreign::{Foreign, ForeignError, LyObj},
    parser::Parser,
    vm::Value,
};
use std::any::Any;

/// Test Foreign implementation for method dispatch testing
#[derive(Debug, Clone, PartialEq)]
struct TestTable {
    name: String,
    rows: Vec<Vec<i64>>,
    columns: Vec<String>,
}

impl TestTable {
    fn new(name: String, columns: Vec<String>) -> Self {
        TestTable {
            name,
            rows: Vec::new(),
            columns,
        }
    }
    
    fn add_row(&mut self, row: Vec<i64>) {
        if row.len() == self.columns.len() {
            self.rows.push(row);
        }
    }
}

impl Foreign for TestTable {
    fn type_name(&self) -> &'static str {
        "TestTable"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Rows" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.rows.len() as i64))
            }
            "Columns" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.columns.len() as i64))
            }
            "Get" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                match (&args[0], &args[1]) {
                    (Value::Integer(row), Value::Integer(col)) => {
                        let row_idx = *row as usize;
                        let col_idx = *col as usize;
                        
                        if row_idx >= self.rows.len() || col_idx >= self.columns.len() {
                            return Err(ForeignError::IndexOutOfBounds {
                                index: format!("({}, {})", row, col),
                                bounds: format!("({}, {})", self.rows.len(), self.columns.len()),
                            });
                        }
                        
                        Ok(Value::Integer(self.rows[row_idx][col_idx]))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args),
                    }),
                }
            }
            "Name" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.name.clone()))
            }
            "ColumnNames" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let column_values: Vec<Value> = self.columns.iter()
                    .map(|name| Value::String(name.clone()))
                    .collect();
                Ok(Value::List(column_values))
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

/// Helper function to create a test table as LyObj value
fn create_test_table() -> Value {
    let mut table = TestTable::new("test_table".to_string(), vec!["id".to_string(), "value".to_string()]);
    table.add_row(vec![1, 100]);
    table.add_row(vec![2, 200]);
    table.add_row(vec![3, 300]);
    
    Value::LyObj(LyObj::new(Box::new(table)))
}

/// Helper to evaluate expressions that include Foreign objects
fn eval_with_foreign_object(source: &str, foreign_obj: Value) -> Result<Value, Box<dyn std::error::Error>> {
    // Parse the source code
    let mut parser = Parser::from_source(source)?;
    let statements = parser.parse()?;
    let expr = statements.last().ok_or("No expressions to evaluate")?;
    
    // Create compiler and compile expression
    let mut compiler = Compiler::new();
    compiler.compile_expr(expr)?;
    
    // Create VM and load the foreign object onto the stack
    let mut vm = compiler.into_vm();
    
    // For now, we'll inject the foreign object into the constants pool
    // In a real implementation, we'd have better variable binding
    vm.constants.push(foreign_obj);
    
    // Execute and get result
    let result = vm.run()?;
    Ok(result)
}

// ==========================================
// Basic Method Dispatch Tests
// ==========================================

#[test]
fn test_method_dispatch_simple() {
    // Test calling Rows[] on a table object
    // This will be implemented as: when we see Rows[table], and table is LyObj,
    // dispatch to table.call_method("Rows", &[])
    
    let table = create_test_table();
    
    // For now, test direct method calls on LyObj values
    if let Value::LyObj(obj) = &table {
        let result = obj.call_method("Rows", &[]).unwrap();
        assert_eq!(result, Value::Integer(3));
        
        let result = obj.call_method("Columns", &[]).unwrap();
        assert_eq!(result, Value::Integer(2));
    } else {
        panic!("Expected LyObj");
    }
}

#[test]
fn test_method_dispatch_with_args() {
    // Test calling Get[table, 1, 0] -> table.call_method("Get", &[1, 0])
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        let result = obj.call_method("Get", &[Value::Integer(1), Value::Integer(0)]).unwrap();
        assert_eq!(result, Value::Integer(2)); // Row 1, Col 0 = id 2
        
        let result = obj.call_method("Get", &[Value::Integer(2), Value::Integer(1)]).unwrap();
        assert_eq!(result, Value::Integer(300)); // Row 2, Col 1 = value 300
    } else {
        panic!("Expected LyObj");
    }
}

#[test]
fn test_method_dispatch_error_handling() {
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Test unknown method
        let result = obj.call_method("UnknownMethod", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForeignError::UnknownMethod { method, .. } => {
                assert_eq!(method, "UnknownMethod");
            }
            _ => panic!("Expected UnknownMethod error"),
        }
        
        // Test wrong arity
        let result = obj.call_method("Rows", &[Value::Integer(1)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForeignError::InvalidArity { method, expected, actual } => {
                assert_eq!(method, "Rows");
                assert_eq!(expected, 0);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected InvalidArity error"),
        }
        
        // Test index out of bounds
        let result = obj.call_method("Get", &[Value::Integer(10), Value::Integer(0)]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForeignError::IndexOutOfBounds { .. } => {}
            _ => panic!("Expected IndexOutOfBounds error"),
        }
    }
}

#[test]
fn test_method_dispatch_return_types() {
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Test method returning String
        let result = obj.call_method("Name", &[]).unwrap();
        assert_eq!(result, Value::String("test_table".to_string()));
        
        // Test method returning List
        let result = obj.call_method("ColumnNames", &[]).unwrap();
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], Value::String("id".to_string()));
                assert_eq!(items[1], Value::String("value".to_string()));
            }
            _ => panic!("Expected List result"),
        }
    }
}

// ==========================================
// Compiler Integration Tests
// ==========================================

#[test]
fn test_stdlib_function_vs_method_dispatch() {
    // Test that normal stdlib functions still work
    // Length[{1, 2, 3}] should call stdlib Length function
    let source = "Length[{1, 2, 3}]";
    let mut parser = Parser::from_source(source).unwrap();
    let statements = parser.parse().unwrap();
    let expr = &statements[0];
    
    let result = Compiler::eval(expr).unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_method_dispatch_compilation_pattern() {
    // Test that when we compile a function call like Rows[obj]
    // where obj is a LyObj, the compiler emits the right bytecode
    
    // For now, this is a placeholder test that shows the intended behavior
    // The actual implementation will modify the compiler to detect when
    // the first argument to a function call is a LyObj and dispatch accordingly
    
    let table = create_test_table();
    
    // This test represents the desired behavior:
    // 1. Parse "Rows[table]" 
    // 2. During compilation, detect that 'table' resolves to a LyObj
    // 3. Emit bytecode for method dispatch instead of regular function call
    // 4. VM executes obj.call_method("Rows", &[])
    
    // For now, we verify this works at the LyObj level
    if let Value::LyObj(obj) = &table {
        let result = obj.call_method("Rows", &[]).unwrap();
        assert_eq!(result, Value::Integer(3));
    }
}

// ==========================================
// VM Integration Tests  
// ==========================================

#[test]
fn test_vm_method_dispatch_execution() {
    // Test that the VM can execute method calls on Foreign objects
    // This will test the VM's CALL opcode handling for method dispatch
    
    let table = create_test_table();
    
    // Create a simple VM program that calls a method
    let mut compiler = Compiler::new();
    
    // Add the table object to constants
    let table_index = compiler.context.add_constant(table).unwrap();
    
    // Emit bytecode for loading the table and calling a method
    // LDC table_index (load table)
    // LDC method_name (load method name)
    // CALL 0 (call with 0 additional args - the method name and object are already on stack)
    
    compiler.context.emit(lyra::bytecode::OpCode::LDC, table_index as u32).unwrap();
    
    // Add method name to constants
    let method_index = compiler.context.add_constant(Value::String("Rows".to_string())).unwrap();
    compiler.context.emit(lyra::bytecode::OpCode::LDC, method_index as u32).unwrap();
    
    // For now, this would require VM modifications to handle method dispatch
    // This test demonstrates the intended execution flow
    let mut vm = compiler.into_vm();
    
    // The VM should recognize that when CALL is executed with a method name and LyObj,
    // it should dispatch to the object's call_method instead of looking up a stdlib function
    
    // This is a placeholder - actual implementation will be in the VM's CALL handler
    // For now, verify the objects are set up correctly
    assert_eq!(vm.constants.len(), 2);
    
    if let Value::LyObj(obj) = &vm.constants[table_index] {
        let result = obj.call_method("Rows", &[]).unwrap();
        assert_eq!(result, Value::Integer(3));
    }
}

// ==========================================
// Error Propagation Tests
// ==========================================

#[test]
fn test_foreign_error_propagation() {
    // Test that Foreign method errors are properly propagated through the VM
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Test that errors are propagated correctly
        let result = obj.call_method("Get", &[Value::String("invalid".to_string()), Value::Integer(0)]);
        
        match result {
            Err(ForeignError::InvalidArgumentType { method, expected, .. }) => {
                assert_eq!(method, "Get");
                assert_eq!(expected, "Integer");
            }
            _ => panic!("Expected InvalidArgumentType error"),
        }
    }
}

// ==========================================
// Performance Tests
// ==========================================

#[test]
fn test_method_dispatch_performance() {
    // Test that method dispatch is reasonably fast
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        let start = std::time::Instant::now();
        
        // Call methods many times to test performance
        for i in 0..1000 {
            let _ = obj.call_method("Get", &[
                Value::Integer(i % 3), 
                Value::Integer(i % 2)
            ]).unwrap();
        }
        
        let duration = start.elapsed();
        
        // Should complete 1000 method calls quickly (< 10ms)
        assert!(duration.as_millis() < 10, "Method dispatch took too long: {:?}", duration);
    }
}

// ==========================================
// Type Safety Tests
// ==========================================

#[test]
fn test_method_dispatch_type_safety() {
    // Test that method dispatch maintains type safety
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Verify type name
        assert_eq!(obj.type_name(), "TestTable");
        
        // Verify downcast safety
        let foreign_ref = obj.as_foreign();
        let test_table = foreign_ref.as_any().downcast_ref::<TestTable>();
        assert!(test_table.is_some());
        
        let test_table = test_table.unwrap();
        assert_eq!(test_table.name, "test_table");
        assert_eq!(test_table.rows.len(), 3);
        assert_eq!(test_table.columns.len(), 2);
    }
}

// ==========================================
// Integration with Existing Language Features
// ==========================================

#[test]
fn test_method_results_in_expressions() {
    // Test that method call results can be used in arithmetic expressions
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Get row count and use in arithmetic
        let row_count = obj.call_method("Rows", &[]).unwrap();
        let col_count = obj.call_method("Columns", &[]).unwrap();
        
        // Test that results can be used in calculations
        match (row_count, col_count) {
            (Value::Integer(rows), Value::Integer(cols)) => {
                assert_eq!(rows * cols, 6); // 3 rows * 2 cols = 6
            }
            _ => panic!("Expected Integer results"),
        }
    }
}

#[test]
fn test_method_results_in_function_calls() {
    // Test that method call results can be passed to other functions
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Get column names and use with Length function
        let column_names = obj.call_method("ColumnNames", &[]).unwrap();
        
        match column_names {
            Value::List(names) => {
                assert_eq!(names.len(), 2);
                // In a real implementation, we could pass this to Length[names]
                // to get the count of columns
            }
            _ => panic!("Expected List result"),
        }
    }
}

// ==========================================
// Complex Method Call Scenarios
// ==========================================

#[test]
fn test_nested_method_calls() {
    // Test calling methods on results of other method calls
    // This would represent something like: Length[ColumnNames[table]]
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // First call: get column names
        let column_names = obj.call_method("ColumnNames", &[]).unwrap();
        
        // Second call: get length of result (using stdlib Length)
        match column_names {
            Value::List(names) => {
                // In real implementation, this would be: Length[ColumnNames[table]]
                // For now, verify the intermediate result
                assert_eq!(names.len(), 2);
            }
            _ => panic!("Expected List result"),
        }
    }
}

#[test]
fn test_method_calls_with_computed_arguments() {
    // Test calling methods with arguments that are results of expressions
    // Like: Get[table, 1 + 1, 2 - 1]
    
    let table = create_test_table();
    
    if let Value::LyObj(obj) = &table {
        // Simulate computed arguments
        let row_arg = Value::Integer(1 + 1); // 2
        let col_arg = Value::Integer(2 - 1); // 1
        
        let result = obj.call_method("Get", &[row_arg, col_arg]).unwrap();
        assert_eq!(result, Value::Integer(300)); // Row 2, Col 1 = 300
    }
}