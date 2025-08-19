//! Debug test to see what tokens are actually being generated

use lyra::lexer::{Lexer, TokenKind};

#[test]
fn debug_simple_type_annotation() {
    let mut lexer = Lexer::new("x: Integer");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    println!("Tokens for 'x: Integer':");
    for (i, token) in tokens.iter().enumerate() {
        println!("  {}: {:?}", i, token.kind);
    }
    println!("Total tokens: {}", tokens.len());
}

#[test]
fn debug_function_type_annotation() {
    let mut lexer = Lexer::new("f[x: Real]: Boolean");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    println!("Tokens for 'f[x: Real]: Boolean':");
    for (i, token) in tokens.iter().enumerate() {
        println!("  {}: {:?}", i, token.kind);
    }
    println!("Total tokens: {}", tokens.len());
}