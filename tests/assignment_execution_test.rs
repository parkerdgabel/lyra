use lyra::{
    ast::{Expr, Number}, 
    parser::Parser,
    lexer::Lexer,
    compiler::Compiler,
    vm::{VirtualMachine, Value},
};

/// Helper function to parse statements
fn parse_statements(input: &str) -> Result<Vec<Expr>, Box<dyn std::error::Error>> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    Ok(parser.parse()?)
}

/// Test immediate assignment compilation and execution
#[test]
fn test_immediate_assignment_execution() {
    // Parse and compile: x = 42
    let statements = parse_statements("x = 42").expect("Failed to parse assignment");
    let assignment = &statements[0];
    
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Check that x is stored in global_symbols with value 42
    assert!(vm.global_symbols.contains_key("x"));
    assert_eq!(vm.global_symbols.get("x"), Some(&Value::Integer(42)));
    
    // Verify delayed_definitions is empty for immediate assignment
    assert!(!vm.delayed_definitions.contains_key("x"));
}

/// Test delayed assignment compilation and execution
#[test]
fn test_delayed_assignment_execution() {
    // Parse and compile: y := 100
    let statements = parse_statements("y := 100").expect("Failed to parse delayed assignment");
    let assignment = &statements[0];
    
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile delayed assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Check that y is stored in delayed_definitions
    assert!(vm.delayed_definitions.contains_key("y"));
    if let Some(expr) = vm.delayed_definitions.get("y") {
        assert!(matches!(expr, Expr::Number(Number::Integer(100))));
    } else {
        panic!("Expected delayed definition for y");
    }
    
    // Verify global_symbols is empty for delayed assignment
    assert!(!vm.global_symbols.contains_key("y"));
}

/// Test symbol resolution after assignment
#[test] 
fn test_symbol_resolution_after_assignment() {
    // First assign: x = 42
    let assignment_stmt = parse_statements("x = 42").expect("Failed to parse assignment");
    let assignment = &assignment_stmt[0];
    
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Now test symbol resolution: compile just "x"
    let symbol_stmt = parse_statements("x").expect("Failed to parse symbol");
    let symbol_expr = &symbol_stmt[0];
    
    let mut compiler2 = Compiler::new();
    compiler2.compile_expr(symbol_expr).expect("Failed to compile symbol");
    
    // Create new VM but transfer the global symbols
    let mut vm2 = VirtualMachine::new();
    vm2.global_symbols = vm.global_symbols.clone();
    vm2.load(compiler2.context.code, compiler2.context.constants);
    
    // Execute symbol loading
    while vm2.ip < vm2.code.len() {
        vm2.step().expect("VM execution failed");
    }
    
    // Check that the stack contains the resolved value 42
    assert_eq!(vm2.stack.len(), 1);
    assert_eq!(vm2.stack[0], Value::Integer(42));
}

/// Test multiple assignments
#[test]
fn test_multiple_assignments() {
    let statements = parse_statements("x = 10").expect("Failed to parse assignment");
    let assignment1 = &statements[0];
    
    // Set up VM and compiler for first assignment
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment1).expect("Failed to compile first assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute first assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Second assignment: y = 20
    let statements2 = parse_statements("y = 20").expect("Failed to parse second assignment");
    let assignment2 = &statements2[0];
    
    let mut compiler2 = Compiler::new();
    compiler2.compile_expr(assignment2).expect("Failed to compile second assignment");
    
    // Create new VM with previous global symbols
    let mut vm2 = VirtualMachine::new();
    vm2.global_symbols = vm.global_symbols.clone();
    vm2.load(compiler2.context.code, compiler2.context.constants);
    
    // Execute second assignment
    while vm2.ip < vm2.code.len() {
        vm2.step().expect("VM execution failed");
    }
    
    // Check both assignments persisted
    assert_eq!(vm2.global_symbols.get("x"), Some(&Value::Integer(10)));
    assert_eq!(vm2.global_symbols.get("y"), Some(&Value::Integer(20)));
}

/// Test that delayed assignment stores the right expression
#[test]
fn test_delayed_assignment_expression_storage() {
    // Parse more complex delayed assignment: z := 5 + 3
    let statements = parse_statements("z := 8").expect("Failed to parse delayed assignment");
    let assignment = &statements[0];
    
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile delayed assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Verify the expression is stored correctly
    assert!(vm.delayed_definitions.contains_key("z"));
    if let Some(stored_expr) = vm.delayed_definitions.get("z") {
        // The stored expression should be the number 8
        assert!(matches!(stored_expr, Expr::Number(Number::Integer(8))));
    } else {
        panic!("Expected delayed definition for z");
    }
}