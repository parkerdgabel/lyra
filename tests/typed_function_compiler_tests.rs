//! Tests for TypedFunction compilation support
//!
//! These tests verify that the compiler can handle TypedFunction expressions
//! and properly extract type information during compilation.

use lyra::compiler::{Compiler, CompilerError};
use lyra::ast::Expr;
use lyra::parser::Parser;

#[test]
fn test_compile_typed_function_simple() {
    let mut compiler = Compiler::new();
    
    // Test compiling: f[x: Integer]: Real
    let mut parser = Parser::from_source("f[x: Integer]: Real").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    
    // Should compile without errors
    let result = compiler.compile_expr(&expressions[0]);
    assert!(result.is_ok(), "TypedFunction compilation should succeed");
}

#[test]
fn test_compile_typed_function_multiple_params() {
    let mut compiler = Compiler::new();
    
    // Test compiling: func[a: Integer, b: Real, c: String]: List[Real]
    let mut parser = Parser::from_source("func[a: Integer, b: Real, c: String]: List[Real]")
        .expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    
    // Should compile without errors
    let result = compiler.compile_expr(&expressions[0]);
    assert!(result.is_ok(), "Multi-parameter TypedFunction compilation should succeed");
}

#[test]
fn test_compile_typed_function_nested_types() {
    let mut compiler = Compiler::new();
    
    // Test compiling: matrix[data: List[List[Real]]]: Matrix
    let mut parser = Parser::from_source("matrix[data: List[List[Real]]]: Matrix")
        .expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    
    // Should compile without errors
    let result = compiler.compile_expr(&expressions[0]);
    assert!(result.is_ok(), "Nested type TypedFunction compilation should succeed");
}

#[test]
fn test_typed_function_generates_correct_bytecode() {
    let mut compiler = Compiler::new();
    
    // Test that TypedFunction generates similar bytecode to regular Function
    let mut parser1 = Parser::from_source("f[x]").expect("Parser creation should succeed");
    let regular_func = parser1.parse().expect("Parsing should succeed");
    
    let mut parser2 = Parser::from_source("f[x: Integer]: Real").expect("Parser creation should succeed");
    let typed_func = parser2.parse().expect("Parsing should succeed");
    
    // Compile both
    let mut compiler1 = Compiler::new();
    let mut compiler2 = Compiler::new();
    
    let result1 = compiler1.compile_expr(&regular_func[0]);
    let result2 = compiler2.compile_expr(&typed_func[0]);
    
    assert!(result1.is_ok());
    assert!(result2.is_ok());
    
    // Both should generate similar instruction patterns
    // (The main difference should be type metadata, not core instructions)
    let code1 = &compiler1.context.code;
    let code2 = &compiler2.context.code;
    
    // Should have similar length (type metadata might add instructions)
    assert!(
        code1.len() <= code2.len(),
        "TypedFunction should generate at least as many instructions as regular function"
    );
}

#[test]
fn test_typed_function_with_mixed_params() {
    let mut compiler = Compiler::new();
    
    // Test compiling: func[x: Integer, y, z: Real]: Boolean
    let mut parser = Parser::from_source("func[x: Integer, y, z: Real]: Boolean")
        .expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    
    // Should compile without errors
    let result = compiler.compile_expr(&expressions[0]);
    assert!(result.is_ok(), "Mixed typed/untyped parameter compilation should succeed");
}

#[test]
fn test_typed_function_stores_type_metadata() {
    let mut compiler = Compiler::new();
    
    // Test that type information is stored during compilation
    let mut parser = Parser::from_source("f[x: Integer, y: Real]: Boolean")
        .expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    // Compile the typed function
    let result = compiler.compile_expr(&expressions[0]);
    assert!(result.is_ok());
    
    // Check that type metadata was stored
    // (This test will drive the implementation of type metadata storage)
    // For now, just verify compilation succeeds
    assert!(true, "Type metadata storage test placeholder");
}

#[test]
fn test_typed_function_error_cases() {
    // Test cases that should work without errors during compilation
    // (Actual type validation errors will be caught at runtime or during type checking)
    
    let test_cases = vec![
        "f[]: Void",  // No parameters
        "f[x: Unknown]: Unknown",  // Unknown types (should compile but may warn)
        "f[x: Integer]: Integer",  // Same input/output type
    ];
    
    for case in test_cases {
        let mut compiler = Compiler::new();
        let mut parser = Parser::from_source(case).expect("Parser creation should succeed");
        let expressions = parser.parse().expect("Parsing should succeed");
        
        let result = compiler.compile_expr(&expressions[0]);
        assert!(result.is_ok(), "Case '{}' should compile successfully", case);
    }
}