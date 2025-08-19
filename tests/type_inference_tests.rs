//! Test type inference from expressions
//! Tests for inferring types from literals and operations

use lyra::compiler::Compiler;
use lyra::ast::Expr;
use lyra::parser::Parser;
use lyra::lexer::Lexer;

#[test]
fn test_integer_literal_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from integer literal: 42
    let source = "42";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("Integer".to_string()));
}

#[test]
fn test_real_literal_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from real literal: 3.14
    let source = "3.14";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("Real".to_string()));
}

#[test]
fn test_string_literal_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from string literal: "hello"
    let source = "\"hello\"";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("String".to_string()));
}

#[test]
fn test_list_literal_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from homogeneous list: {1, 2, 3}
    let source = "{1, 2, 3}";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("List[Integer]".to_string()));
}

#[test]
fn test_mixed_list_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from mixed list: {1, 2.5, 3}
    let source = "{1, 2.5, 3}";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    // Should infer to List[Real] since Real can represent both Integer and Real
    assert_eq!(inferred_type, Some("List[Real]".to_string()));
}

#[test]
fn test_heterogeneous_list_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from heterogeneous list: {1, "hello", 3.14}
    let source = "{1, \"hello\", 3.14}";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    // Should infer to a union type or a generic "Any" type
    assert_eq!(inferred_type, Some("List[Union[Integer, Real, String]]".to_string()));
}

#[test]
fn test_nested_list_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from nested list: {{1, 2}, {3, 4}}
    let source = "{{1, 2}, {3, 4}}";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("List[List[Integer]]".to_string()));
}

#[test]
fn test_arithmetic_operation_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from arithmetic: Plus[1, 2]
    let source = "Plus[1, 2]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    assert_eq!(inferred_type, Some("Integer".to_string()));
}

#[test]
fn test_mixed_arithmetic_type_inference() {
    let mut compiler = Compiler::new();
    
    // Test type inference from mixed arithmetic: Plus[1, 2.5]
    let source = "Plus[1, 2.5]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&expr);
    // Integer + Real = Real
    assert_eq!(inferred_type, Some("Real".to_string()));
}

#[test]
fn test_typed_function_call_type_inference() {
    let mut compiler = Compiler::new();
    
    // First register a typed function: add[x: Integer, y: Integer]: Integer
    let signature_source = "add[x: Integer, y: Integer]: Integer";
    let mut lexer = Lexer::new(signature_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let signature_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&signature_expr).unwrap();
    
    // Now test type inference from function call: add[1, 2]
    let call_source = "add[1, 2]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    
    let inferred_type = compiler.infer_expression_type(&call_expr);
    assert_eq!(inferred_type, Some("Integer".to_string()));
}