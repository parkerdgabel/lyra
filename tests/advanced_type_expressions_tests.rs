//! Test advanced type expression parsing and extraction
//! Tests for Map, Tuple, Optional, Union, Array, and Generic types

use lyra::compiler::Compiler;
use lyra::ast::{Expr, Symbol};
use lyra::parser::Parser;
use lyra::lexer::Lexer;

#[test]
fn test_map_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Map[String, Integer] type
    let source = "Map[String, Integer]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let map_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&map_type_expr);
    assert_eq!(extracted, Some("Map[String, Integer]".to_string()));
}

#[test]
fn test_nested_map_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Map[String, List[Integer]] type
    let source = "Map[String, List[Integer]]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let nested_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&nested_type_expr);
    assert_eq!(extracted, Some("Map[String, List[Integer]]".to_string()));
}

#[test]
fn test_tuple_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Tuple[Integer, String, Real] type
    let source = "Tuple[Integer, String, Real]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let tuple_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&tuple_type_expr);
    assert_eq!(extracted, Some("Tuple[Integer, String, Real]".to_string()));
}

#[test]
fn test_optional_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Optional[Integer] type
    let source = "Optional[Integer]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let optional_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&optional_type_expr);
    assert_eq!(extracted, Some("Optional[Integer]".to_string()));
}

#[test]
fn test_union_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Union[Integer, String] type
    let source = "Union[Integer, String]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let union_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&union_type_expr);
    assert_eq!(extracted, Some("Union[Integer, String]".to_string()));
}

#[test]
fn test_array_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Array[Real, 3] type
    let source = "Array[Real, 3]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let array_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&array_type_expr);
    assert_eq!(extracted, Some("Array[Real, 3]".to_string()));
}

#[test]
fn test_generic_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test generic type T
    let source = "T";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let generic_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&generic_type_expr);
    assert_eq!(extracted, Some("T".to_string()));
}

#[test]
fn test_complex_nested_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test deeply nested type: List[Map[String, Optional[Integer]]]
    let source = "List[Map[String, Optional[Integer]]]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let complex_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&complex_type_expr);
    assert_eq!(extracted, Some("List[Map[String, Optional[Integer]]]".to_string()));
}

#[test]
fn test_function_type_extraction() {
    let mut compiler = Compiler::new();
    
    // Test Function[Integer, String] -> Boolean type
    let source = "Function[Integer, String, Boolean]"; // For now, use existing Function syntax
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let function_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&function_type_expr);
    assert_eq!(extracted, Some("Function[Integer, String, Boolean]".to_string()));
}

#[test]
fn test_typed_function_with_advanced_types() {
    let mut compiler = Compiler::new();
    
    // Parse: processData[input: Map[String, List[Integer]], config: Optional[Config]]: Result[Data]
    let source = "processData[input: Map[String, List[Integer]], config: Optional[Config]]: Result[Data]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    // Compile and check that advanced types are extracted
    compiler.compile_expr(&expr).unwrap();
    
    let signature = compiler.context.get_enhanced_type_signature("processData").unwrap();
    assert_eq!(signature.name, "processData");
    assert_eq!(signature.param_count(), 2);
    assert!(signature.is_typed);
    assert_eq!(signature.get_param_type("input"), Some("Map[String, List[Integer]]"));
    assert_eq!(signature.get_param_type("config"), Some("Optional[Config]"));
    assert_eq!(signature.get_return_type(), Some("Result[Data]"));
}