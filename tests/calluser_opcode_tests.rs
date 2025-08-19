use lyra::compiler::Compiler;
use lyra::lexer::Lexer;
use lyra::parser::Parser;
use lyra::bytecode::OpCode;
use std::collections::HashMap;

#[test]
fn test_calluser_opcode_generation() {
    // Test: User-defined function calls generate CallUser opcodes
    let mut compiler = Compiler::new();
    
    // First define a typed function: f[x: Integer]: Integer = 42
    let func_source = "f[x: Integer]: Integer = 42";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Now call the function: f[1]
    let call_source = "f[1]";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    // Check that CallUser opcode was generated
    let has_call_user = compiler.context.code.iter().any(|inst| inst.opcode == OpCode::CallUser);
    assert!(has_call_user, "CallUser opcode should be generated for user-defined function calls");
}

#[test]
fn test_stdlib_vs_user_function_dispatch() {
    // Test: Stdlib functions generate CallStatic, user functions generate CallUser
    let mut compiler = Compiler::new();
    
    // Define user function: myFunc[x] = 42
    let func_source = "myFunc[x] = 42";
    let mut lexer = Lexer::new(func_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func_expr).unwrap();
    
    // Call user function: myFunc[1]
    let user_call_source = "myFunc[1]";
    let mut lexer = Lexer::new(user_call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let user_call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&user_call_expr).unwrap();
    
    // Call stdlib function: Length[{1, 2, 3}]
    let stdlib_call_source = "Length[{1, 2, 3}]";
    let mut lexer = Lexer::new(stdlib_call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let stdlib_call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&stdlib_call_expr).unwrap();
    
    // Check opcodes
    let has_call_user = compiler.context.code.iter().any(|inst| inst.opcode == OpCode::CallUser);
    let has_call_static = compiler.context.code.iter().any(|inst| inst.opcode == OpCode::CallStatic);
    
    assert!(has_call_user, "User functions should generate CallUser opcodes");
    assert!(has_call_static, "Stdlib functions should generate CallStatic opcodes");
}

#[test]
fn test_type_metadata_storage_integration() {
    // Test: Type metadata is properly stored and accessible
    let mut compiler = Compiler::new();
    
    // Define multiple typed functions
    let func1_source = "add[x: Integer, y: Integer]: Integer = 42";
    let mut lexer = Lexer::new(func1_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func1_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func1_expr).unwrap();
    
    let func2_source = "scale[x: Real]: Real = 2.0";
    let mut lexer = Lexer::new(func2_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let func2_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&func2_expr).unwrap();
    
    // Verify metadata storage
    assert!(compiler.context.has_enhanced_metadata("add"));
    assert!(compiler.context.has_enhanced_metadata("scale"));
    
    let add_meta = compiler.context.get_enhanced_type_signature("add").unwrap();
    assert_eq!(add_meta.params.len(), 2);
    assert_eq!(add_meta.params[0].1, Some("Integer".to_string()));
    assert_eq!(add_meta.params[1].1, Some("Integer".to_string()));
    assert_eq!(add_meta.return_type, Some("Integer".to_string()));
    
    let scale_meta = compiler.context.get_enhanced_type_signature("scale").unwrap();
    assert_eq!(scale_meta.params.len(), 1);
    assert_eq!(scale_meta.params[0].1, Some("Real".to_string()));
    assert_eq!(scale_meta.return_type, Some("Real".to_string()));
}