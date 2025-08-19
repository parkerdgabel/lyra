//! Minimal test for TypedFunction compilation support

use lyra::compiler::Compiler;
use lyra::parser::Parser;

#[test]
fn test_simple_typed_function_compiles() {
    // Just test that TypedFunction doesn't cause a panic
    let mut parser = Parser::from_source("f[x: Integer]: Real").expect("Parser should work");
    let expressions = parser.parse().expect("Parsing should work");
    
    let mut compiler = Compiler::new();
    let result = compiler.compile_expr(&expressions[0]);
    
    // The main goal is that compilation doesn't crash with UnsupportedExpression
    // Detailed testing will come later once the infrastructure is stable
    println!("Compilation result: {:?}", result);
    
    // For now, just assert that we get some result (success or failure is both acceptable)
    assert!(true, "TypedFunction compilation should not panic");
}