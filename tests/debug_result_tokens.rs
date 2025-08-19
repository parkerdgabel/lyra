//! Debug test to see the tokens for Result type annotation

use lyra::lexer::Lexer;

#[test]
fn debug_result_type_annotation() {
    let mut lexer = Lexer::new("result: Result[String, Integer]");
    let tokens = lexer.tokenize().expect("Lexing should succeed");
    
    println!("Tokens for 'result: Result[String, Integer]':");
    for (i, token) in tokens.iter().enumerate() {
        println!("  {}: {:?}", i, token.kind);
    }
    println!("Total tokens: {}", tokens.len());
}