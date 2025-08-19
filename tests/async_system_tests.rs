use lyra::compiler::Compiler;
use lyra::lexer::Lexer;
use lyra::parser::Parser;
use lyra::vm::{VirtualMachine, Value};

#[test]
fn test_promise_creation() {
    // Test: Promise stdlib function can create Future values
    let mut compiler = Compiler::new();
    
    // Create a promise: result = Promise[42]
    let source = "result = Promise[42]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&expr).unwrap();
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return a Future value
    let result = vm.run();
    assert!(result.is_ok());
    let value = result.unwrap();
    
    // Check that we got a Future LyObj back
    match value {
        Value::LyObj(ly_obj) => {
            // Should be a Future type
            assert_eq!(ly_obj.type_name(), "Future");
            
            // Call resolve method to get the contained value
            let resolved = ly_obj.call_method("resolve", &[]).unwrap();
            assert_eq!(resolved, Value::Integer(42));
        }
        _ => panic!("Expected Future LyObj, got {:?}", value),
    }
}

#[test]
fn test_await_resolution() {
    // Test: Await stdlib function can resolve futures
    let mut compiler = Compiler::new();
    
    // Create a promise and await it: result = Await[Promise[100]]
    let source = "result = Await[Promise[100]]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&expr).unwrap();
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return the resolved value (100)
    let result = vm.run();
    assert!(result.is_ok());
    let value = result.unwrap();
    assert_eq!(value, Value::Integer(100));
}

#[test]
fn test_future_type_validation() {
    // Test: Future objects work with basic Promise creation
    let mut compiler = Compiler::new();
    
    // Create a promise and store it: result = Promise[42]
    let source = "result = Promise[42]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&expr).unwrap();
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().unwrap();
    compiler.compile_expr(&call_expr).unwrap();
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return a Future LyObj
    let result = vm.run();
    if let Err(ref error) = result {
        println!("VM execution failed: {:?}", error);
    }
    assert!(result.is_ok());
    let value = result.unwrap();
    
    match value {
        Value::LyObj(ly_obj) => {
            assert_eq!(ly_obj.type_name(), "Future");
            let resolved = ly_obj.call_method("resolve", &[]).unwrap();
            assert_eq!(resolved, Value::Integer(42));
        }
        _ => panic!("Expected Future[Integer], got {:?}", value),
    }
}