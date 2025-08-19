//! Test compile-time type checking and validation
//! Tests for type checking function calls, assignments, and operations

use lyra::compiler::{Compiler, CompilerError};
use lyra::parser::Parser;
use lyra::lexer::Lexer;

#[test]
fn test_valid_typed_function_call() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: add[x: Integer, y: Integer]: Integer
    let signature_source = "add[x: Integer, y: Integer]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should succeed: add[1, 2] with Integer arguments
    let call_source = "add[1, 2]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_ok());
}

#[test]
fn test_invalid_typed_function_arity() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: add[x: Integer, y: Integer]: Integer
    let signature_source = "add[x: Integer, y: Integer]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should fail: add[1] with wrong arity
    let call_source = "add[1]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    if let Err(CompilerError::InvalidArity { function, expected, actual }) = result {
        assert_eq!(function, "add");
        assert_eq!(expected, 2);
        assert_eq!(actual, 1);
    } else {
        panic!("Expected InvalidArity error");
    }
}

#[test]
fn test_invalid_typed_function_argument_types() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: add[x: Integer, y: Integer]: Integer
    let signature_source = "add[x: Integer, y: Integer]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should fail: add["hello", 2] with String instead of Integer
    let call_source = "add[\"hello\", 2]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    match result {
        Err(CompilerError::UnsupportedExpression(msg)) => {
            assert!(msg.contains("Type mismatch"));
            assert!(msg.contains("expected Integer"));
            assert!(msg.contains("got String"));
        }
        _ => panic!("Expected type mismatch error"),
    }
}

#[test]
fn test_type_coercion_integer_to_real() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: multiply[x: Real, y: Real]: Real
    let signature_source = "multiply[x: Real, y: Real]: Real";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should succeed: multiply[1, 2.5] with Integer coerced to Real
    let call_source = "multiply[1, 2.5]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_ok());
}

#[test]
fn test_list_type_checking() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: process[data: List[Integer]]: Integer
    let signature_source = "process[data: List[Integer]]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should succeed: process[{1, 2, 3}] with List[Integer]
    let call_source = "process[{1, 2, 3}]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_ok());
}

#[test]
fn test_invalid_list_type_checking() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: process[data: List[Integer]]: Integer
    let signature_source = "process[data: List[Integer]]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // This should fail: process[{1, "hello", 3}] with mixed list
    let call_source = "process[{1, \"hello\", 3}]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    match result {
        Err(CompilerError::UnsupportedExpression(msg)) => {
            assert!(msg.contains("Type mismatch"));
            assert!(msg.contains("List[Integer]"));
        }
        _ => panic!("Expected type mismatch error"),
    }
}

#[test]
fn test_nested_type_checking() {
    let mut compiler = Compiler::new();
    
    // Register a typed function: transform[data: Map[String, List[Integer]]]: Boolean
    let signature_source = "transform[data: Map[String, List[Integer]]]: Boolean";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Test with a wrong type - pass a simple List instead of Map
    let call_source = "transform[{1, 2, 3}]"; // List[Integer] instead of Map[String, List[Integer]]
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    // This should fail because List[Integer] is not compatible with Map[String, List[Integer]]
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    match result {
        Err(CompilerError::UnsupportedExpression(msg)) => {
            assert!(msg.contains("Type mismatch"));
        }
        _ => panic!("Expected type mismatch error"),
    }
}

#[test]
fn test_arithmetic_type_checking() {
    let mut compiler = Compiler::new();
    
    // Test that arithmetic operations are type-checked
    let source = "Plus[1, \"hello\"]"; // Should fail: Integer + String
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&expr);
    assert!(result.is_err());
    match result {
        Err(CompilerError::UnsupportedExpression(msg)) => {
            assert!(msg.contains("Cannot perform arithmetic"));
        }
        _ => panic!("Expected arithmetic type error"),
    }
}

#[test]
fn test_assignment_basic_compilation() {
    let mut compiler = Compiler::new();
    
    // Test basic assignment compilation (no strict typing yet)
    let assignment = "x = 42";
    let mut lexer = Lexer::new(assignment);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let assign_expr = parser.parse_expression().unwrap();
    
    // Basic assignments should compile successfully
    let result = compiler.compile_expr_with_type_checking(&assign_expr);
    assert!(result.is_ok());
}