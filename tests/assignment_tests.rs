use lyra::{
    ast::{Expr, Number, Symbol}, 
    compiler::Compiler, 
    vm::{VirtualMachine, Value},
    parser::parse_statements,
};

/// Test immediate assignment compilation and execution
#[test]
fn test_immediate_assignment_basic() {
    // Parse assignment: x = 42
    let statements = parse_statements("x = 42").expect("Failed to parse assignment");
    assert_eq!(statements.len(), 1);
    
    let assignment = &statements[0];
    if let Expr::Assignment { lhs, rhs, delayed } = assignment {
        assert!(!delayed); // Should be immediate assignment
        assert!(matches!(lhs.as_ref(), Expr::Symbol(_)));
        assert!(matches!(rhs.as_ref(), Expr::Number(Number::Integer(42))));
    } else {
        panic!("Expected Assignment expression");
    }
    
    // Test compilation and execution
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Check that x is in global_symbols with value 42
    assert!(vm.global_symbols.contains_key("x"));
    assert_eq!(vm.global_symbols.get("x"), Some(&Value::Integer(42)));
}

/// Test delayed assignment compilation
#[test]
fn test_delayed_assignment_basic() {
    // Parse assignment: x := 42
    let statements = parse_statements("x := 42").expect("Failed to parse delayed assignment");
    assert_eq!(statements.len(), 1);
    
    let assignment = &statements[0];
    if let Expr::Assignment { lhs, rhs, delayed } = assignment {
        assert!(delayed); // Should be delayed assignment
        assert!(matches!(lhs.as_ref(), Expr::Symbol(_)));
        assert!(matches!(rhs.as_ref(), Expr::Number(Number::Integer(42))));
    } else {
        panic!("Expected Assignment expression");
    }
    
    // Test compilation and execution
    let mut compiler = Compiler::new();
    compiler.compile_expr(assignment).expect("Failed to compile delayed assignment");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code, compiler.context.constants);
    
    // Execute the assignment
    while vm.ip < vm.code.len() {
        vm.step().expect("VM execution failed");
    }
    
    // Check that x is in delayed_definitions
    assert!(vm.delayed_definitions.contains_key("x"));
    if let Some(expr) = vm.delayed_definitions.get("x") {
        assert!(matches!(expr, Expr::Number(Number::Integer(42))));
    } else {
        panic!("Expected delayed definition for x");
    }
}

/// Test symbol resolution after immediate assignment
#[test] 
fn test_symbol_resolution_immediate() {
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
    
    // Now test symbol resolution: compile just "x" and see if it resolves to 42
    let symbol_stmt = parse_statements("x").expect("Failed to parse symbol");
    let symbol_expr = &symbol_stmt[0];
    
    let mut compiler2 = Compiler::new();
    compiler2.compile_expr(symbol_expr).expect("Failed to compile symbol");
    
    // Transfer the resolved symbols to the new VM
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
    let statements = parse_statements("x = 10; y = 20; z := x + y").expect("Failed to parse multiple assignments");
    assert_eq!(statements.len(), 3);
    
    let mut compiler = Compiler::new();
    let mut vm = VirtualMachine::new();
    
    // Compile and execute each statement
    for stmt in statements {
        // Reset compiler context for each statement (in reality they'd be cumulative)
        let mut stmt_compiler = Compiler::new();
        // Transfer previous constants and code
        stmt_compiler.context.constants = compiler.context.constants.clone();
        
        stmt_compiler.compile_expr(&stmt).expect("Failed to compile statement");
        
        let mut stmt_vm = VirtualMachine::new();
        // Transfer global symbols from previous executions
        stmt_vm.global_symbols = vm.global_symbols.clone();
        stmt_vm.delayed_definitions = vm.delayed_definitions.clone();
        
        stmt_vm.load(stmt_compiler.context.code, stmt_compiler.context.constants);
        
        // Execute
        while stmt_vm.ip < stmt_vm.code.len() {
            stmt_vm.step().expect("VM execution failed");
        }
        
        // Update VM state for next iteration
        vm.global_symbols = stmt_vm.global_symbols.clone();
        vm.delayed_definitions = stmt_vm.delayed_definitions.clone();
        compiler.context = stmt_compiler.context;
    }
    
    // Check final state
    assert_eq!(vm.global_symbols.get("x"), Some(&Value::Integer(10)));
    assert_eq!(vm.global_symbols.get("y"), Some(&Value::Integer(20)));
    assert!(vm.delayed_definitions.contains_key("z"));
}

/// Test invalid assignment patterns
#[test]
fn test_invalid_assignment_patterns() {
    let mut compiler = Compiler::new();
    
    // Try to compile assignment with non-symbol LHS (should fail for now)
    let invalid_assignment = Expr::Assignment {
        lhs: Box::new(Expr::Number(Number::Integer(42))),
        rhs: Box::new(Expr::Number(Number::Integer(10))),
        delayed: false,
    };
    
    let result = compiler.compile_expr(&invalid_assignment);
    assert!(result.is_err());
    
    // Check error message contains expected text
    if let Err(err) = result {
        let error_msg = format!("{}", err);
        assert!(error_msg.contains("Complex assignment patterns not yet supported"));
    }
}

/// Test parser correctly identifies assignment operators
#[test]
fn test_assignment_parsing() {
    // Test immediate assignment
    let immediate = parse_statements("f[x_] = x^2").expect("Failed to parse function definition");
    if let Expr::Assignment { delayed, .. } = &immediate[0] {
        assert!(!delayed);
    } else {
        panic!("Expected Assignment");
    }
    
    // Test delayed assignment  
    let delayed = parse_statements("g[x_] := RandomReal[]").expect("Failed to parse delayed function definition");
    if let Expr::Assignment { delayed, .. } = &delayed[0] {
        assert!(delayed);
    } else {
        panic!("Expected Assignment");
    }
}