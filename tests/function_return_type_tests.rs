//! Tests for function return type annotation parsing
//!
//! These tests verify that the parser can handle function return type syntax
//! like `f[x: Real]: Boolean`

use lyra::parser::Parser;
use lyra::ast::{Expr, Pattern};

#[test]
fn test_function_with_return_type_annotation() {
    let mut parser = Parser::from_source("f[x: Real]: Boolean").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        // For now, this will be parsed as a function definition with return type
        // We need to extend the AST to support typed function definitions
        Expr::TypedFunction { head, params, return_type } => {
            if let Expr::Symbol(sym) = head.as_ref() {
                assert_eq!(sym.name, "f");
            } else {
                panic!("Expected Symbol for function head, got {:?}", head);
            }
            
            assert_eq!(params.len(), 1);
            match &params[0] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "x");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Real");
                    }
                }
                other => panic!("Expected typed pattern in params, got {:?}", other),
            }
            
            if let Expr::Symbol(sym) = return_type.as_ref() {
                assert_eq!(sym.name, "Boolean");
            } else {
                panic!("Expected Symbol for return type, got {:?}", return_type);
            }
        }
        other => panic!("Expected typed function, got {:?}", other),
    }
}

#[test]
fn test_complex_function_with_return_type() {
    let mut parser = Parser::from_source("func[a: Integer, b: Real]: List[String]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::TypedFunction { head, params, return_type } => {
            if let Expr::Symbol(sym) = head.as_ref() {
                assert_eq!(sym.name, "func");
            }
            
            assert_eq!(params.len(), 2);
            
            // Check first parameter: a: Integer
            match &params[0] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "a");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Integer");
                    }
                }
                other => panic!("Expected typed pattern for first param, got {:?}", other),
            }
            
            // Check second parameter: b: Real  
            match &params[1] {
                Expr::Pattern(Pattern::Typed { name, type_pattern }) => {
                    assert_eq!(name, "b");
                    if let Expr::Symbol(sym) = type_pattern.as_ref() {
                        assert_eq!(sym.name, "Real");
                    }
                }
                other => panic!("Expected typed pattern for second param, got {:?}", other),
            }
            
            // Check return type: List[String]
            match return_type.as_ref() {
                Expr::Function { head: ret_head, args: ret_args } => {
                    if let Expr::Symbol(sym) = ret_head.as_ref() {
                        assert_eq!(sym.name, "List");
                    }
                    assert_eq!(ret_args.len(), 1);
                    if let Expr::Symbol(sym) = &ret_args[0] {
                        assert_eq!(sym.name, "String");
                    }
                }
                other => panic!("Expected List function for return type, got {:?}", other),
            }
        }
        other => panic!("Expected typed function, got {:?}", other),
    }
}

#[test]
fn test_function_without_return_type_still_works() {
    // Ensure we don't break existing function parsing
    let mut parser = Parser::from_source("f[x: Real]").expect("Parser creation should succeed");
    let expressions = parser.parse().expect("Parsing should succeed");
    
    assert_eq!(expressions.len(), 1);
    match &expressions[0] {
        Expr::Function { head, args } => {
            if let Expr::Symbol(sym) = head.as_ref() {
                assert_eq!(sym.name, "f");
            }
            assert_eq!(args.len(), 1);
        }
        other => panic!("Expected regular function, got {:?}", other),
    }
}