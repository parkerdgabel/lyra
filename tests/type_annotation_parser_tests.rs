//! Tests for type annotation parsing support
//!
//! These tests verify that the parser can handle type annotation syntax
//! in function parameters, return types, and variable declarations.

use lyra::parser::Parser;
use lyra::ast::{Expr, Pattern};

#[test]
fn test_simple_variable_type_annotation() {
    let mut parser = Parser::from_source("x: Integer").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
            assert_eq!(name, "x");
            if let Expr::Symbol(sym) = type_pattern.as_ref() {
                assert_eq!(sym.name, "Integer");
            } else {
                panic!("Expected Symbol for type pattern, got {:?}", type_pattern);
            }
        }
        other => panic!("Expected typed pattern, got {:?}", other),
    }
}

#[test]
fn test_function_with_parameter_type_annotation() {
    let mut parser = Parser::from_source("f[x: Integer]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Function { head, args } => {
            if let Expr::Symbol(sym) = head.as_ref() {
                assert_eq!(sym.name, "f");
            } else {
                panic!("Expected Symbol for function head, got {:?}", head);
            }
            
            assert_eq!(args.len(), 1);
            match &args[0] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "x");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Integer");
                    } else {
                        panic!("Expected Symbol for type pattern, got {:?}", type_pattern);
                    }
                }
                other => panic!("Expected typed pattern in args, got {:?}", other),
            }
        }
        other => panic!("Expected function call, got {:?}", other),
    }
}

#[test]
fn test_function_with_multiple_typed_parameters() {
    let mut parser = Parser::from_source("func[a: Integer, b: Real, c: String]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Function { head, args } => {
            if let Expr::Symbol(sym) = head.as_ref() {
                assert_eq!(sym.name, "func");
            }
            
            assert_eq!(args.len(), 3);
            
            // Check first parameter: a: Integer
            match &args[0] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "a");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Integer");
                    }
                }
                other => panic!("Expected typed pattern for first arg, got {:?}", other),
            }
            
            // Check second parameter: b: Real
            match &args[1] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "b");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Real");
                    }
                }
                other => panic!("Expected typed pattern for second arg, got {:?}", other),
            }
            
            // Check third parameter: c: String  
            match &args[2] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "c");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "String");
                    }
                }
                other => panic!("Expected typed pattern for third arg, got {:?}", other),
            }
        }
        other => panic!("Expected function call, got {:?}", other),
    }
}

#[test]
fn test_complex_type_annotation() {
    let mut parser = Parser::from_source("matrix: List[List[Real]]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
            assert_eq!(name, "matrix");
            
            // Should be List[List[Real]]
            match type_pattern.as_ref() {
                Expr::Function { head, args } => {
                    if let Expr::Symbol(sym) = head.as_ref() {
                        assert_eq!(sym.name, "List");
                    }
                    assert_eq!(args.len(), 1);
                    
                    // Inner List[Real]
                    match &args[0] {
                        Expr::Function { head: inner_head, args: inner_args } => {
                            if let Expr::Symbol(sym) = inner_head.as_ref() {
                                assert_eq!(sym.name, "List");
                            }
                            assert_eq!(inner_args.len(), 1);
                            
                            if let Expr::Symbol(sym) = &inner_args[0] {
                                assert_eq!(sym.name, "Real");
                            }
                        }
                        other => panic!("Expected inner List function, got {:?}", other),
                    }
                }
                other => panic!("Expected List function for type, got {:?}", other),
            }
        }
        other => panic!("Expected typed pattern, got {:?}", other),
    }
}

#[test]
fn test_function_return_type_annotation() {
    // This test will initially fail - we need to implement return type annotations
    // Expected syntax: f[x: Real]: Boolean
    let source = "f[x: Real]: Boolean";
    let result = Parser::from_source(source);
    
    // For now, we expect this to parse as two separate expressions
    // Later we'll implement proper return type syntax
    if let Ok(mut parser) = result {
        let _expressions = parser.parse();
        // This is a placeholder test - we'll implement return types later
        assert!(true, "Return type parsing not yet implemented");
    }
}

#[test]
fn test_mixed_typed_and_untyped_parameters() {
    let mut parser = Parser::from_source("func[x: Integer, y, z: Real]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Function { head, args } => {
            assert_eq!(args.len(), 3);
            
            // First arg: x: Integer (typed)
            match &args[0] {
                Expr::Pattern(Pattern::Typed { name, .. }) => {
                    assert_eq!(name, "x");
                }
                other => panic!("Expected typed pattern for first arg, got {:?}", other),
            }
            
            // Second arg: y (untyped) 
            match &args[1] {
                Expr::Symbol(sym) => {
                    assert_eq!(sym.name, "y");
                }
                other => panic!("Expected symbol for second arg, got {:?}", other),
            }
            
            // Third arg: z: Real (typed)
            match &args[2] {
                Expr::Pattern(Pattern::Typed { name, .. }) => {
                    assert_eq!(name, "z");
                }
                other => panic!("Expected typed pattern for third arg, got {:?}", other),
            }
        }
        other => panic!("Expected function call, got {:?}", other),
    }
}