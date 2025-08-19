use lyra::compiler::Compiler;
use lyra::lexer::Lexer;
use lyra::parser::Parser;
use lyra::vm::{VirtualMachine, Value};
use std::collections::HashMap;

#[test]
fn test_runtime_parameter_type_mismatch() {
    // Test: Function call with wrong parameter type fails at runtime
    let mut compiler = Compiler::new();
    
    // Define typed function: add[x: Integer, y: Integer]: Integer = x + y
    let func_source = "add[x: Integer, y: Integer]: Integer = x + y";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with wrong type: add[1, 3.14] - second arg is Real, not Integer
    let call_source = "add[1, 3.14]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should fail with type validation error
    let result = vm.run();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Type error"));
    assert!(error_msg.contains("parameter y"));
    assert!(error_msg.contains("expected"));
    assert!(error_msg.contains("Integer"));
    assert!(error_msg.contains("Real"));
}

#[test]
fn test_runtime_arity_validation() {
    // Test: Function call with wrong number of arguments fails
    let mut compiler = Compiler::new();
    
    // Define function: multiply[x: Integer, y: Integer]: Integer = x * y
    let func_source = "multiply[x: Integer, y: Integer]: Integer = x * y";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with wrong arity: multiply[5] - missing second argument
    let call_source = "multiply[5]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    let result = vm.run();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Type error"));
    assert!(error_msg.contains("2 parameters"));
    assert!(error_msg.contains("1 arguments"));
}

#[test] 
fn test_successful_integer_to_real_coercion() {
    // Test: Integer arguments automatically coerce to Real parameters
    let mut compiler = Compiler::new();
    
    // Define function accepting Real: scale[x: Real]: Real = x * 2.5
    let func_source = "scale[x: Real]: Real = x * 2.5";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with Integer (should auto-coerce): scale[4]
    let call_source = "scale[4]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    let result = vm.run().unwrap();
    // Integer 4 coerced to Real 4.0, then 4.0 * 2.5 = 10.0
    assert_eq!(result, Value::Real(10.0));
}

#[test]
fn test_invalid_string_to_numeric_coercion() {
    // Test: Invalid coercions fail with clear error messages
    let mut compiler = Compiler::new();
    
    // Define numeric function: square[x: Integer]: Integer = x * x
    let func_source = "square[x: Integer]: Integer = x * x";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with String: square["hello"]
    let call_source = "square[\"hello\"]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    let result = vm.run();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    println!("String coercion error: {}", error_msg);
    assert!(error_msg.contains("type mismatch") || error_msg.contains("Type error"));
    assert!(error_msg.contains("String") && error_msg.contains("Integer"));
}

#[test]
fn test_gradual_typing_partial_validation() {
    // Test: Functions with mixed typed/untyped parameters validate only typed ones
    let mut compiler = Compiler::new();
    
    // Define partially typed function: partial[x: Integer, y] = x + 1
    let func_source = "partial[x: Integer, y] = x + 1";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with correct first type, any second type: partial[42, "anything"]
    let call_source = "partial[42, \"anything\"]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should succeed - only x is type-checked, y can be anything
    let result = vm.run().unwrap();
    assert_eq!(result, Value::Integer(43));
}

#[test]
fn test_gradual_typing_typed_parameter_validation() {
    // Test: Typed parameters in mixed functions are still validated
    let mut compiler = Compiler::new();
    
    // Define partially typed function: partial[x: Integer, y] = x + 1
    let func_source = "partial[x: Integer, y] = x + 1";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with wrong first type: partial[3.14, "anything"] - x should be Integer
    let call_source = "partial[3.14, \"anything\"]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should fail - x must be Integer even in partially typed function
    let result = vm.run();
    match result {
        Ok(val) => {
            println!("Gradual typing test unexpectedly succeeded with result: {:?}", val);
            panic!("Expected type error but got success");
        }
        Err(error) => {
            let error_msg = error.to_string();
            println!("Gradual typing test failed as expected: {}", error_msg);
            assert!(error_msg.contains("type mismatch") || error_msg.contains("Type error"));
            assert!(error_msg.contains("parameter x"));
            assert!(error_msg.contains("Integer"));
            assert!(error_msg.contains("Real"));
        }
    }
}

#[test]
fn test_untyped_function_no_validation() {
    // Test: Completely untyped functions bypass all type checking
    let mut compiler = Compiler::new();
    
    // Define untyped function: anything[a, b] = a
    let func_source = "anything[a, b] = a";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with any types: anything["hello", 42]
    let call_source = "anything[\"hello\", 42]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should succeed - no type checking for untyped functions
    let result = vm.run().unwrap();
    assert_eq!(result, Value::String("hello".to_string()));
}

#[test]
fn test_return_type_validation() {
    // Test: Function return type is validated at runtime
    let mut compiler = Compiler::new();
    
    // Define function with return type: bad[x: Integer]: Integer = "not an integer"
    let func_source = "bad[x: Integer]: Integer = \"not an integer\"";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call function: bad[42]
    let call_source = "bad[42]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should fail - return type mismatch
    let result = vm.run();
    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    println!("Return type error: {}", error_msg);
    assert!(error_msg.contains("return type mismatch") || error_msg.contains("Return type mismatch"));
    assert!(error_msg.contains("Integer"));
    assert!(error_msg.contains("String"));
}

#[test]
fn test_nested_type_validation() {
    // Test: Complex type expressions are validated correctly
    let mut compiler = Compiler::new();
    
    // Define function with list type: sumlist[nums: List[Integer]]: Integer = Plus @@ nums
    let func_source = "sumlist[nums: List[Integer]]: Integer = Length[nums]";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with invalid list: sumlist[{1, 2, "oops"}] - contains String
    let call_source = "sumlist[{1, 2, \"oops\"}]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should fail - list contains non-Integer element
    let result = vm.run();
    match result {
        Ok(val) => {
            println!("Test unexpectedly succeeded with result: {:?}", val);
            panic!("Expected error but got success");
        }
        Err(error) => {
            let error_msg = error.to_string();
            println!("Test failed as expected with error: {}", error_msg);
            assert!(error_msg.contains("type mismatch") || error_msg.contains("Type error"));
            assert!(error_msg.contains("List[Integer]") || error_msg.contains("Integer"));
            assert!(error_msg.contains("String"));
        }
    }
}

#[test]
fn test_successful_complex_type_validation() {
    // Test: Valid complex types work correctly
    let mut compiler = Compiler::new();
    
    // Define function: process[data: List[Real]]: Real = Length[data]
    let func_source = "process[data: List[Real]]: Real = Length[data]";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call with valid list: process[{1.0, 2.5, 3.7}]
    let call_source = "process[{1.0, 2.5, 3.7}]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    vm.load_type_metadata(
        compiler.context.type_metadata.clone(),
        compiler.context.enhanced_metadata.clone(),
        compiler.context.user_functions.clone(),
    );
    
    // Should succeed
    let result = vm.run().unwrap();
    assert_eq!(result, Value::Real(3.0)); // Length of 3-element list
}