//! Tests for type annotation lexing support
//!
//! These tests verify that the lexer can handle type annotation syntax
//! like `x: Integer`, `f[x: Real]: Boolean`, etc.

use lyra::lexer::{Lexer, TokenKind};

#[test]
fn test_type_annotation_colon() {
    let mut lexer = Lexer::new("x: Integer");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    assert_eq!(tokens.len(), 4); // x, :, Integer, EOF
    assert_eq!(tokens[0].kind, TokenKind::Symbol("x".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Colon);
    assert_eq!(tokens[2].kind, TokenKind::Symbol("Integer".to_string()));
}

#[test]
fn test_function_return_type_annotation() {
    let mut lexer = Lexer::new("f[x: Real]: Boolean");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Expected: f, [, x, :, Real, ], :, Boolean, EOF
    assert_eq!(tokens.len(), 9);
    assert_eq!(tokens[0].kind, TokenKind::Symbol("f".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[2].kind, TokenKind::Symbol("x".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::Colon);
    assert_eq!(tokens[4].kind, TokenKind::Symbol("Real".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::RightBracket);
    assert_eq!(tokens[6].kind, TokenKind::Colon);
    assert_eq!(tokens[7].kind, TokenKind::Symbol("Boolean".to_string()));
}

#[test]
fn test_multiple_parameter_types() {
    let mut lexer = Lexer::new("func[a: Integer, b: Real, c: String]");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Expected: func, [, a, :, Integer, ,, b, :, Real, ,, c, :, String, ], EOF
    assert_eq!(tokens.len(), 15);
    
    // Check parameter annotations
    assert_eq!(tokens[2].kind, TokenKind::Symbol("a".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::Colon);
    assert_eq!(tokens[4].kind, TokenKind::Symbol("Integer".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::Comma);
    
    assert_eq!(tokens[6].kind, TokenKind::Symbol("b".to_string()));
    assert_eq!(tokens[7].kind, TokenKind::Colon);
    assert_eq!(tokens[8].kind, TokenKind::Symbol("Real".to_string()));
    assert_eq!(tokens[9].kind, TokenKind::Comma);
    
    assert_eq!(tokens[10].kind, TokenKind::Symbol("c".to_string()));
    assert_eq!(tokens[11].kind, TokenKind::Colon);
    assert_eq!(tokens[12].kind, TokenKind::Symbol("String".to_string()));
}

#[test]
fn test_complex_type_annotations() {
    let mut lexer = Lexer::new("matrix: List[List[Real]]");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Expected: matrix, :, List, [, List, [, Real, ], ], EOF
    assert_eq!(tokens.len(), 10);
    assert_eq!(tokens[0].kind, TokenKind::Symbol("matrix".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Colon);
    assert_eq!(tokens[2].kind, TokenKind::Symbol("List".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[4].kind, TokenKind::Symbol("List".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[6].kind, TokenKind::Symbol("Real".to_string()));
    assert_eq!(tokens[7].kind, TokenKind::RightBracket);
    assert_eq!(tokens[8].kind, TokenKind::RightBracket);
}

#[test]
fn test_optional_type_annotation() {
    let mut lexer = Lexer::new("value: Option[Integer]");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Expected: value, :, Option, [, Integer, ], EOF
    assert_eq!(tokens.len(), 7);
    assert_eq!(tokens[0].kind, TokenKind::Symbol("value".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Colon);
    assert_eq!(tokens[2].kind, TokenKind::Symbol("Option".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[4].kind, TokenKind::Symbol("Integer".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::RightBracket);
}

#[test]
fn test_result_type_annotation() {
    let mut lexer = Lexer::new("result: Result[String, Integer]");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Expected: result, :, Result, [, String, ,, Integer, ], EOF
    assert_eq!(tokens.len(), 9);
    assert_eq!(tokens[0].kind, TokenKind::Symbol("result".to_string()));
    assert_eq!(tokens[1].kind, TokenKind::Colon);
    assert_eq!(tokens[2].kind, TokenKind::Symbol("Result".to_string()));
    assert_eq!(tokens[3].kind, TokenKind::LeftBracket);
    assert_eq!(tokens[4].kind, TokenKind::Symbol("String".to_string()));
    assert_eq!(tokens[5].kind, TokenKind::Comma);
    assert_eq!(tokens[6].kind, TokenKind::Symbol("Integer".to_string()));
    assert_eq!(tokens[7].kind, TokenKind::RightBracket);
}

#[test]
fn test_type_annotation_with_existing_wolfram_syntax() {
    // Make sure type annotations don't interfere with existing syntax
    let mut lexer = Lexer::new("x := y + z; f[a: Real] := a^2");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    // Should properly distinguish between := (SetDelayed) and : (type annotation)
    assert!(tokens.iter().any(|t| t.kind == TokenKind::SetDelayed));
    assert!(tokens.iter().any(|t| t.kind == TokenKind::Colon));
    
    // Count the different colon types
    let colon_count = tokens.iter().filter(|t| t.kind == TokenKind::Colon).count();
    let set_delayed_count = tokens.iter().filter(|t| t.kind == TokenKind::SetDelayed).count();
    
    assert_eq!(colon_count, 1); // One type annotation
    assert_eq!(set_delayed_count, 2); // Two delayed assignments
}