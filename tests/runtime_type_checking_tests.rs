use lyra::compiler::Compiler;
use lyra::lexer::Lexer;
use lyra::parser::Parser;
use lyra::vm::{VirtualMachine, Value};

#[test]
fn test_typed_function_metadata_storage() {
    // Test: TypedFunction metadata is stored correctly
    let mut compiler = Compiler::new();
    
    // Define a typed function: f[x: Integer]: Integer = 42
    let func_source = "f[x: Integer]: Integer = 42";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    
    // This should now compile successfully (GREEN - first step)
    let result = compiler.compile_expr(&func_expr);
    assert!(result.is_ok());
    
    // Check that metadata was stored
    assert!(compiler.context.has_enhanced_metadata("f"));
    let metadata = compiler.context.get_enhanced_type_signature("f").unwrap();
    assert_eq!(metadata.name, "f");
    assert_eq!(metadata.params.len(), 1);
    assert_eq!(metadata.params[0].0, "x");
    assert_eq!(metadata.params[0].1, Some("Integer".to_string()));
    assert_eq!(metadata.return_type, Some("Integer".to_string()));
    assert!(metadata.is_typed);
}

#[test]
fn test_untyped_function_metadata_storage() {
    // Test: For now, skip untyped function assignments - they're not implemented yet
    // This test will be implemented when we add support for regular function assignments
    // TODO: Implement untyped function assignment support in future iterations
}

#[test] 
fn test_typed_function_compilation_only() {
    // Test: Complex typed functions compile without execution
    let mut compiler = Compiler::new();
    
    // Define a typed function with multiple parameters: add[x: Integer, y: Real]: Real
    let func_source = "add[x: Integer, y: Real]: Real";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    
    // This should compile successfully
    let result = compiler.compile_expr(&func_expr);
    assert!(result.is_ok());
    
    // Check metadata
    assert!(compiler.context.has_enhanced_metadata("add"));
    let metadata = compiler.context.get_enhanced_type_signature("add").unwrap();
    assert_eq!(metadata.name, "add");
    assert_eq!(metadata.params.len(), 2);
    assert_eq!(metadata.params[0].0, "x");
    assert_eq!(metadata.params[0].1, Some("Integer".to_string()));
    assert_eq!(metadata.params[1].0, "y");
    assert_eq!(metadata.params[1].1, Some("Real".to_string()));
    assert_eq!(metadata.return_type, Some("Real".to_string()));
    assert!(metadata.is_typed);
}

#[test]
fn test_mixed_typed_and_untyped_parameters() {
    // Test: Functions with some typed and some untyped parameters
    let mut compiler = Compiler::new();
    
    // Define a function with mixed parameters: mixed[x: Integer, y] = 42
    let func_source = "mixed[x: Integer, y] = 42";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    
    // This should compile successfully
    let result = compiler.compile_expr(&func_expr);
    assert!(result.is_ok());
    
    // Check metadata
    assert!(compiler.context.has_enhanced_metadata("mixed"));
    let metadata = compiler.context.get_enhanced_type_signature("mixed").unwrap();
    assert_eq!(metadata.name, "mixed");
    assert_eq!(metadata.params.len(), 2);
    assert_eq!(metadata.params[0].0, "x");
    assert_eq!(metadata.params[0].1, Some("Integer".to_string()));
    assert_eq!(metadata.params[1].0, "y");
    assert_eq!(metadata.params[1].1, None); // Untyped parameter
    assert!(!metadata.is_typed); // Not fully typed because of mixed parameters
}