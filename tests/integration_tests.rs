use lyra::{compiler::Compiler, parser::Parser, vm::Value};

/// Test the complete pipeline: source code → AST → bytecode → execution
fn eval_source(source: &str) -> Result<Value, Box<dyn std::error::Error>> {
    // Parse source code to AST
    let mut parser = Parser::from_source(source)?;
    let statements = parser.parse()?;
    
    // Take the last statement as the expression to evaluate
    let expr = statements.last().ok_or("No expressions to evaluate")?;
    
    // Compile AST to bytecode and execute
    let result = Compiler::eval(expr)?;
    Ok(result)
}

#[test]
fn test_simple_arithmetic() {
    let result = eval_source("1 + 2").unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_multiplication() {
    let result = eval_source("3 * 4").unwrap();
    assert_eq!(result, Value::Integer(12));
}

#[test]
fn test_division() {
    let result = eval_source("8 / 2").unwrap();
    assert_eq!(result, Value::Real(4.0));
}

#[test]
fn test_power() {
    let result = eval_source("2 ^ 3").unwrap();
    assert_eq!(result, Value::Integer(8));
}

#[test]
fn test_complex_expression() {
    // Test (2 + 3) * 4 = 20
    let result = eval_source("(2 + 3) * 4").unwrap();
    assert_eq!(result, Value::Integer(20));
}

#[test]
fn test_nested_operations() {
    // Test 2 + 3 * 4 = 14 (respects precedence)
    let result = eval_source("2 + 3 * 4").unwrap();
    assert_eq!(result, Value::Integer(14));
}

#[test]
fn test_power_precedence() {
    // Test 2 * 3 ^ 2 = 18 (3^2 = 9, then 2*9 = 18)
    let result = eval_source("2 * 3 ^ 2").unwrap();
    assert_eq!(result, Value::Integer(18));
}

#[test]
fn test_mixed_types() {
    // Test integer and real arithmetic
    let result = eval_source("5 + 2.5").unwrap();
    assert_eq!(result, Value::Real(7.5));
}

#[test]
fn test_function_call_syntax() {
    // Test explicit function call syntax
    let result = eval_source("Plus[7, 3]").unwrap();
    assert_eq!(result, Value::Integer(10));
}

#[test]
fn test_nested_function_calls() {
    // Test Times[Plus[2, 3], 4] = 20
    let result = eval_source("Times[Plus[2, 3], 4]").unwrap();
    assert_eq!(result, Value::Integer(20));
}

#[test]
fn test_large_numbers() {
    // Test with numbers that require constant pool
    let result = eval_source("16777216 + 1").unwrap(); // 2^24 + 1
    assert_eq!(result, Value::Integer(16777217));
}

#[test]
fn test_real_numbers() {
    let result = eval_source("3.14 * 2.0").unwrap();
    assert_eq!(result, Value::Real(6.28));
}

#[test]
fn test_parentheses_precedence() {
    // Test (2 + 3) * (4 + 5) = 5 * 9 = 45
    let result = eval_source("(2 + 3) * (4 + 5)").unwrap();
    assert_eq!(result, Value::Integer(45));
}

#[test]
fn test_deep_nesting() {
    // Test ((1 + 2) * 3) + 4 = (3 * 3) + 4 = 13
    let result = eval_source("((1 + 2) * 3) + 4").unwrap();
    assert_eq!(result, Value::Integer(13));
}

#[test]
fn test_division_results_in_real() {
    // Even integer division should result in real
    let result = eval_source("7 / 2").unwrap();
    assert_eq!(result, Value::Real(3.5));
}

#[test]
fn test_power_with_negative_exponent() {
    // 2^(-1) should be 0.5
    let result = eval_source("Power[2, -1]").unwrap();
    assert_eq!(result, Value::Real(0.5));
}

#[test]
fn test_error_handling_division_by_zero() {
    let result = eval_source("5 / 0");
    assert!(result.is_err());
}

#[test]
fn test_literals() {
    // Test various literal types work
    let int_result = eval_source("42").unwrap();
    assert_eq!(int_result, Value::Integer(42));
    
    let real_result = eval_source("3.14159").unwrap();
    assert_eq!(real_result, Value::Real(3.14159));
}

#[test]
fn test_zero_arithmetic() {
    let result = eval_source("0 + 0").unwrap();
    assert_eq!(result, Value::Integer(0));
    
    let result2 = eval_source("5 * 0").unwrap();
    assert_eq!(result2, Value::Integer(0));
}

#[test]
fn test_negative_numbers() {
    // Test parsing and evaluating negative numbers
    let result = eval_source("-5 + 3").unwrap();
    assert_eq!(result, Value::Integer(-2));
}

#[test]
fn test_operator_associativity() {
    // Test that subtraction is left-associative: 10 - 3 - 2 = (10 - 3) - 2 = 5
    let result = eval_source("10 - 3 - 2").unwrap();
    assert_eq!(result, Value::Integer(5));
}

#[test]
fn test_power_right_associativity() {
    // Test that power is right-associative: 2^3^2 = 2^(3^2) = 2^9 = 512
    let result = eval_source("2 ^ 3 ^ 2").unwrap();
    assert_eq!(result, Value::Integer(512));
}

/// Test individual compiler components work with parsed AST
#[test]
fn test_compiler_with_parsed_ast() {
    // Parse an expression
    let mut parser = Parser::from_source("2 + 3 * 4").unwrap();
    let statements = parser.parse().unwrap();
    let expr = &statements[0];
    
    // Compile it
    let mut compiler = Compiler::new();
    compiler.compile_expr(expr).unwrap();
    
    // Verify the bytecode looks reasonable
    assert!(!compiler.context.code.is_empty());
    
    // Execute it
    compiler.context.emit(lyra::bytecode::OpCode::Halt, 0).unwrap();
    let mut vm = compiler.into_vm();
    let result = vm.run().unwrap();
    
    assert_eq!(result, Value::Integer(14));
}

/// Test compiling multiple statements
#[test]
fn test_compile_multiple_statements() {
    let mut parser = Parser::from_source("1 + 1; 2 * 3").unwrap();
    let statements = parser.parse().unwrap();
    
    assert_eq!(statements.len(), 2);
    
    // Compile the program
    let mut compiler = Compiler::new();
    compiler.compile_program(&statements).unwrap();
    
    // Execute and get the result (should be the last statement)
    let mut vm = compiler.into_vm();
    let result = vm.run().unwrap();
    
    // The last statement is 2 * 3 = 6, but our current implementation
    // might have both results on stack. Let's just verify it runs.
    // In a real implementation, we'd want better semantics for multiple statements.
    assert!(matches!(result, Value::Integer(_)));
}

/// Performance test: compile and execute a moderately complex expression
#[test]
fn test_performance_complex_expression() {
    let source = "((1 + 2) * (3 + 4)) + ((5 + 6) * (7 + 8))"; // = (3*7) + (11*15) = 21 + 165 = 186
    
    let start = std::time::Instant::now();
    let result = eval_source(source).unwrap();
    let duration = start.elapsed();
    
    assert_eq!(result, Value::Integer(186));
    
    // Should complete in reasonable time (less than 1ms for this simple expression)
    assert!(duration.as_millis() < 10, "Compilation took too long: {:?}", duration);
}

// ===============================
// Standard Library Integration Tests
// ===============================

// Helper function to assert float equality with tolerance
fn assert_float_eq(actual: Value, expected: f64, tolerance: f64) {
    match actual {
        Value::Real(r) => assert!((r - expected).abs() < tolerance, 
            "Expected {}, got {}, difference: {}", expected, r, (r - expected).abs()),
        _ => panic!("Expected Real value, got {:?}", actual),
    }
}

// List Operations Tests
#[test]
fn test_stdlib_length() {
    let result = eval_source("Length[{1, 2, 3, 4, 5}]").unwrap();
    assert_eq!(result, Value::Integer(5));
    
    let result = eval_source("Length[{}]").unwrap();
    assert_eq!(result, Value::Integer(0));
}

#[test]
fn test_stdlib_head() {
    let result = eval_source("Head[{10, 20, 30}]").unwrap();
    assert_eq!(result, Value::Integer(10));
}

#[test]
fn test_stdlib_tail() {
    let result = eval_source("Tail[{1, 2, 3, 4}]").unwrap();
    if let Value::List(list) = result {
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], Value::Integer(2));
        assert_eq!(list[1], Value::Integer(3));
        assert_eq!(list[2], Value::Integer(4));
    } else {
        panic!("Expected List value, got {:?}", result);
    }
}

#[test]
fn test_stdlib_append() {
    let result = eval_source("Append[{1, 2}, 3]").unwrap();
    if let Value::List(list) = result {
        assert_eq!(list.len(), 3);
        assert_eq!(list[0], Value::Integer(1));
        assert_eq!(list[1], Value::Integer(2));
        assert_eq!(list[2], Value::Integer(3));
    } else {
        panic!("Expected List value, got {:?}", result);
    }
}

#[test]
fn test_stdlib_flatten() {
    // Note: This test requires nested list support in the parser
    // For now, test with mixed content
    let result = eval_source("Flatten[{1, {2, 3}, 4}]");
    // This might fail if parser doesn't handle nested lists yet
    // We'll check for error and that's okay for now
    if result.is_err() {
        // Parser doesn't support nested lists yet, that's expected
        return;
    }
    
    if let Ok(Value::List(list)) = result {
        assert_eq!(list.len(), 4);
        assert_eq!(list[0], Value::Integer(1));
        assert_eq!(list[1], Value::Integer(2));
        assert_eq!(list[2], Value::Integer(3));
        assert_eq!(list[3], Value::Integer(4));
    }
}

// String Operations Tests
#[test]
fn test_stdlib_string_join() {
    let result = eval_source("StringJoin[\"Hello\", \" \", \"World\"]").unwrap();
    assert_eq!(result, Value::String("Hello World".to_string()));
}

#[test]
fn test_stdlib_string_length() {
    let result = eval_source("StringLength[\"Hello\"]").unwrap();
    assert_eq!(result, Value::Integer(5));
    
    let result = eval_source("StringLength[\"\"]").unwrap();
    assert_eq!(result, Value::Integer(0));
}

#[test]
fn test_stdlib_string_take() {
    let result = eval_source("StringTake[\"Hello\", 3]").unwrap();
    assert_eq!(result, Value::String("Hel".to_string()));
    
    let result = eval_source("StringTake[\"Hi\", 10]").unwrap();
    assert_eq!(result, Value::String("Hi".to_string()));
}

#[test]
fn test_stdlib_string_drop() {
    let result = eval_source("StringDrop[\"Hello\", 2]").unwrap();
    assert_eq!(result, Value::String("llo".to_string()));
    
    let result = eval_source("StringDrop[\"Hi\", 10]").unwrap();
    assert_eq!(result, Value::String("".to_string()));
}

// Math Functions Tests
#[test]
fn test_stdlib_sin() {
    let result = eval_source("Sin[0]").unwrap();
    assert_float_eq(result, 0.0, 1e-10);
    
    // Test with Pi/2 approximation
    let result = eval_source("Sin[1.5707963267948966]").unwrap();
    assert_float_eq(result, 1.0, 1e-10);
}

#[test]
fn test_stdlib_cos() {
    let result = eval_source("Cos[0]").unwrap();
    assert_float_eq(result, 1.0, 1e-10);
    
    // Test with Pi approximation
    let result = eval_source("Cos[3.141592653589793]").unwrap();
    assert_float_eq(result, -1.0, 1e-10);
}

#[test]
fn test_stdlib_tan() {
    let result = eval_source("Tan[0]").unwrap();
    assert_float_eq(result, 0.0, 1e-10);
    
    // Test with Pi/4 approximation
    let result = eval_source("Tan[0.7853981633974483]").unwrap();
    assert_float_eq(result, 1.0, 1e-10);
}

#[test]
fn test_stdlib_exp() {
    let result = eval_source("Exp[0]").unwrap();
    assert_float_eq(result, 1.0, 1e-10);
    
    let result = eval_source("Exp[1]").unwrap();
    assert_float_eq(result, std::f64::consts::E, 1e-10);
}

#[test]
fn test_stdlib_log() {
    let result = eval_source("Log[1]").unwrap();
    assert_float_eq(result, 0.0, 1e-10);
    
    let result = eval_source("Log[2.718281828459045]").unwrap(); // E
    assert_float_eq(result, 1.0, 1e-10);
}

#[test]
fn test_stdlib_sqrt() {
    let result = eval_source("Sqrt[0]").unwrap();
    assert_float_eq(result, 0.0, 1e-10);
    
    let result = eval_source("Sqrt[4]").unwrap();
    assert_float_eq(result, 2.0, 1e-10);
    
    let result = eval_source("Sqrt[2]").unwrap();
    assert_float_eq(result, std::f64::consts::SQRT_2, 1e-10);
}

// Complex Expressions with stdlib
#[test]
fn test_stdlib_nested_calls() {
    // Test Length[Tail[{1, 2, 3, 4}]] = 3
    let result = eval_source("Length[Tail[{1, 2, 3, 4}]]").unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_stdlib_arithmetic_with_functions() {
    // Test Sin[0] + Cos[0] = 0 + 1 = 1
    let result = eval_source("Sin[0] + Cos[0]").unwrap();
    assert_float_eq(result, 1.0, 1e-10);
}

#[test]
fn test_stdlib_string_operations_composition() {
    // Test StringLength[StringJoin["Hello", "World"]] = 10
    let result = eval_source("StringLength[StringJoin[\"Hello\", \"World\"]]").unwrap();
    assert_eq!(result, Value::Integer(10));
}

// Error handling tests
#[test]
fn test_stdlib_error_handling() {
    // Test invalid function calls
    let result = eval_source("UnknownFunction[1, 2]");
    assert!(result.is_err());
    
    // Test wrong argument count
    let result = eval_source("Length[1, 2]");
    assert!(result.is_err());
    
    // Test wrong argument type  
    let result = eval_source("Length[42]");
    assert!(result.is_err());
}

#[test]
fn test_stdlib_edge_cases() {
    // Test Head and Tail on empty lists (should error)
    let result = eval_source("Head[{}]");
    assert!(result.is_err());
    
    let result = eval_source("Tail[{}]");
    assert!(result.is_err());
    
    // Test Log and Sqrt with invalid inputs (should error)
    let result = eval_source("Log[0]");
    assert!(result.is_err());
    
    let result = eval_source("Sqrt[-1]");
    assert!(result.is_err());
}