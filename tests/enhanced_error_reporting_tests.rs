//! Test enhanced error reporting for type checking
//! Tests for detailed error messages with suggestions and context

use lyra::compiler::{Compiler, CompilerError};
use lyra::parser::Parser;
use lyra::lexer::Lexer;

#[test]
fn test_detailed_type_mismatch_error() {
    let mut compiler = Compiler::new();
    
    // Register a typed function
    let signature_source = "compute[input: List[Integer], threshold: Real]: Boolean";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Call with wrong types
    let call_source = "compute[\"hello\", 42]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    if let Err(CompilerError::UnsupportedExpression(msg)) = result {
        // Should include parameter name, expected type, actual type
        assert!(msg.contains("input"));
        assert!(msg.contains("List[Integer]"));
        assert!(msg.contains("String"));
        println!("Error message: {}", msg);
    } else {
        panic!("Expected detailed type mismatch error");
    }
}

#[test]
fn test_arity_error_with_function_signature() {
    let mut compiler = Compiler::new();
    
    // Register a typed function
    let signature_source = "process[data: List[String], config: Map[String, Integer], verbose: Boolean]: Result[Data]";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Call with wrong arity
    let call_source = "process[myData, myConfig]"; // Missing third argument
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    if let Err(CompilerError::InvalidArity { function, expected, actual }) = result {
        assert_eq!(function, "process");
        assert_eq!(expected, 3);
        assert_eq!(actual, 2);
        println!("Arity error for function: {} (expected {}, got {})", function, expected, actual);
    } else {
        panic!("Expected InvalidArity error");
    }
}

#[test]
fn test_arithmetic_type_error_with_context() {
    let mut compiler = Compiler::new();
    
    // Try to add incompatible types
    let source = "Plus[42, \"hello\", 3.14]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&expr);
    assert!(result.is_err());
    if let Err(CompilerError::UnsupportedExpression(msg)) = result {
        // Should mention the operation and problematic type
        assert!(msg.contains("arithmetic"));
        assert!(msg.contains("String"));
        println!("Arithmetic error message: {}", msg);
    } else {
        panic!("Expected arithmetic type error");
    }
}

#[test]
fn test_list_type_compatibility_error() {
    let mut compiler = Compiler::new();
    
    // Register function expecting specific list type
    let signature_source = "analyze[numbers: List[Real]]: Statistics";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Call with incompatible list type
    let call_source = "analyze[{\"a\", \"b\", \"c\"}]"; // List[String] instead of List[Real]
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    if let Err(CompilerError::UnsupportedExpression(msg)) = result {
        assert!(msg.contains("numbers"));
        assert!(msg.contains("List[Real]"));
        assert!(msg.contains("List[String]"));
        println!("List compatibility error: {}", msg);
    } else {
        panic!("Expected list type compatibility error");
    }
}

#[test]
fn test_coercion_success_message() {
    let mut compiler = Compiler::new();
    
    // Register function expecting Real
    let signature_source = "calculate[value: Real]: Real";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Call with Integer (should be coerced to Real)
    let call_source = "calculate[42]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    // This should succeed due to Integer -> Real coercion
    assert!(result.is_ok());
}

#[test]
fn test_nested_type_structure_error() {
    let mut compiler = Compiler::new();
    
    // Register function with deeply nested type
    let signature_source = "transform[data: Map[String, List[Optional[Integer]]]]: Boolean";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Call with wrong nested structure
    let call_source = "transform[{1, 2, 3}]"; // Simple list instead of complex Map
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    assert!(result.is_err());
    if let Err(CompilerError::UnsupportedExpression(msg)) = result {
        assert!(msg.contains("Map[String, List[Optional[Integer]]]"));
        assert!(msg.contains("List[Integer]"));
        println!("Nested type error: {}", msg);
    } else {
        panic!("Expected nested type structure error");
    }
}

#[test]
fn test_unknown_function_helpful_message() {
    let mut compiler = Compiler::new();
    
    // Call unknown function (should compile successfully but could provide warnings)
    let call_source = "unknownFunction[1, 2, 3]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let result = compiler.compile_expr_with_type_checking(&call_expr);
    // Unknown functions should still compile (dynamic dispatch)
    assert!(result.is_ok());
}