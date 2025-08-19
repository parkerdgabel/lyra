//! Test type extraction from TypedFunction expressions

use lyra::compiler::{Compiler, EnhancedFunctionSignature};
use lyra::ast::{Expr, Symbol};
use lyra::parser::Parser;
use lyra::lexer::Lexer;

#[test]
fn test_enhanced_signature_creation() {
    // Test creating an enhanced signature with actual type information
    let sig = EnhancedFunctionSignature {
        name: "add".to_string(),
        params: vec![
            ("x".to_string(), Some("Integer".to_string())),
            ("y".to_string(), Some("Integer".to_string())),
        ],
        return_type: Some("Integer".to_string()),
        is_typed: true,
        location: Some((1, 1)),
    };
    
    assert_eq!(sig.name, "add");
    assert_eq!(sig.param_count(), 2);
    assert!(sig.is_fully_typed());
    assert_eq!(sig.get_param_type("x"), Some("Integer"));
    assert_eq!(sig.get_param_type("y"), Some("Integer"));
    assert_eq!(sig.get_return_type(), Some("Integer"));
}

#[test]
fn test_mixed_typed_untyped_parameters() {
    let sig = EnhancedFunctionSignature {
        name: "func".to_string(),
        params: vec![
            ("x".to_string(), Some("Integer".to_string())),
            ("y".to_string(), None), // Untyped parameter
            ("z".to_string(), Some("Real".to_string())),
        ],
        return_type: None,
        is_typed: false,
        location: None,
    };
    
    assert_eq!(sig.param_count(), 3);
    assert!(!sig.is_fully_typed());
    assert_eq!(sig.typed_param_count(), 2);
    assert_eq!(sig.untyped_param_count(), 1);
    assert_eq!(sig.get_param_type("x"), Some("Integer"));
    assert_eq!(sig.get_param_type("y"), None);
    assert_eq!(sig.get_param_type("z"), Some("Real"));
}

#[test]
fn test_type_extraction_from_simple_expressions() {
    let mut compiler = Compiler::new();
    
    // Test extraction from Symbol expressions
    let integer_sym = Expr::Symbol(Symbol { name: "Integer".to_string() });
    assert_eq!(compiler.extract_type_from_expr(&integer_sym), Some("Integer".to_string()));
    
    let real_sym = Expr::Symbol(Symbol { name: "Real".to_string() });
    assert_eq!(compiler.extract_type_from_expr(&real_sym), Some("Real".to_string()));
    
    let string_sym = Expr::Symbol(Symbol { name: "String".to_string() });
    assert_eq!(compiler.extract_type_from_expr(&string_sym), Some("String".to_string()));
}

#[test]
fn test_type_extraction_from_complex_expressions() {
    let mut compiler = Compiler::new();
    
    // Test List[Integer] type
    let source = "List[Integer]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let list_type_expr = parser.parse_expression().unwrap();
    
    let extracted = compiler.extract_type_from_expr(&list_type_expr);
    assert_eq!(extracted, Some("List[Integer]".to_string()));
}

#[test]
fn test_typed_function_compilation_with_extraction() {
    let mut compiler = Compiler::new();
    
    // Parse: f[x: Integer, y: Real]: Boolean
    let source = "f[x: Integer, y: Real]: Boolean";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    // Compile and check that types are extracted
    compiler.compile_expr(&expr).unwrap();
    
    let signature = compiler.context.get_enhanced_type_signature("f").unwrap();
    assert_eq!(signature.name, "f");
    assert_eq!(signature.param_count(), 2);
    assert!(signature.is_fully_typed());
    assert_eq!(signature.get_param_type("x"), Some("Integer"));
    assert_eq!(signature.get_param_type("y"), Some("Real"));
    assert_eq!(signature.get_return_type(), Some("Boolean"));
}

#[test]
fn test_function_validation_with_types() {
    let mut compiler = Compiler::new();
    
    // Register a typed function
    let sig = EnhancedFunctionSignature {
        name: "add".to_string(),
        params: vec![
            ("x".to_string(), Some("Integer".to_string())),
            ("y".to_string(), Some("Integer".to_string())),
        ],
        return_type: Some("Integer".to_string()),
        is_typed: true,
        location: None,
    };
    compiler.register_enhanced_signature(sig);
    
    // Test validation - correct arity and types should pass
    let result = compiler.validate_enhanced_function_call("add", 2, &["Integer", "Integer"]);
    assert!(result.is_ok());
    
    // Test validation - wrong arity should fail
    let result = compiler.validate_enhanced_function_call("add", 1, &["Integer"]);
    assert!(result.is_err());
    
    // Test validation - wrong types should fail
    let result = compiler.validate_enhanced_function_call("add", 2, &["String", "Integer"]);
    assert!(result.is_err());
}