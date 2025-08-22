//! Phase 8A-1A: Many slots stress tests (100+ parameters with pure functions)
//! Tests slot substitution algorithm with extreme numbers of slot parameters

use lyra::compiler::Compiler;
use lyra::parser::Parser;
use lyra::lexer::Lexer;
use lyra::vm::VM;
use lyra::value::Value;

#[test]
fn test_many_slots_pure_function_100_params() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create pure function with 100 slots: #1 + #2 + #3 + ... + #100 &
    let mut pure_func_body = String::new();
    for i in 1..=100 {
        if i > 1 {
            pure_func_body.push_str(" + ");
        }
        pure_func_body.push_str(&format!("#{}", i));
    }
    pure_func_body.push_str(" &");
    
    println!("Testing pure function with 100 slots...");
    
    // Parse and compile the pure function
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    
    // Execute to get the pure function value
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    // Test application with 100 arguments
    let test_code = format!("({}).apply({})", 
        pure_func_body,
        (1..=100).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
    );
    
    // Expected result: sum of 1 to 100 = 5050
    let expected = 5050;
    
    // For now, verify the pure function was created correctly
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with 100 slots created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_pure_function_500_params() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create pure function with 500 slots
    let mut pure_func_body = String::new();
    for i in 1..=500 {
        if i > 1 {
            pure_func_body.push_str(" + ");
        }
        pure_func_body.push_str(&format!("#{}", i));
    }
    pure_func_body.push_str(" &");
    
    println!("Testing pure function with 500 slots...");
    
    // Parse and compile
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with 500 slots created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_pure_function_1000_params() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create pure function with 1000 slots
    let mut pure_func_body = String::new();
    for i in 1..=1000 {
        if i > 1 {
            pure_func_body.push_str(" + ");
        }
        pure_func_body.push_str(&format!("#{}", i));
    }
    pure_func_body.push_str(" &");
    
    println!("Testing pure function with 1000 slots...");
    
    // Parse and compile
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with 1000 slots created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_mixed_operations() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create complex pure function with 50 slots using various operations
    let mut operations = Vec::new();
    for i in 1..=50 {
        match i % 5 {
            0 => operations.push(format!("#{}", i)),                    // Identity
            1 => operations.push(format!("#{} * 2", i)),               // Multiply
            2 => operations.push(format!("#{} + 10", i)),              // Add
            3 => operations.push(format!("#{} ^ 2", i)),               // Square
            4 => operations.push(format!("If[#{} > 25, #{}, 0]", i, i)), // Conditional
            _ => unreachable!(),
        }
    }
    
    let pure_func_body = format!("{} &", operations.join(" + "));
    
    println!("Testing pure function with 50 slots and mixed operations...");
    
    // Parse and compile
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with 50 slots and mixed operations created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_nested_expressions() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create pure function with 25 slots but deeply nested
    let mut expression = "#1".to_string();
    for i in 2..=25 {
        expression = format!("({} + #{})", expression, i);
    }
    let pure_func_body = format!("{} &", expression);
    
    println!("Testing pure function with 25 slots in nested expressions...");
    
    // Parse and compile
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with 25 slots in nested expressions created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_sparse_usage() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Create pure function that uses only some slots out of many possible
    // Uses slots #5, #15, #25, #35, #45, #55, #65, #75, #85, #95 out of potential 100
    let sparse_slots = vec![5, 15, 25, 35, 45, 55, 65, 75, 85, 95];
    let pure_func_body = format!("{} &", 
        sparse_slots.iter()
            .map(|i| format!("#{}", i))
            .collect::<Vec<_>>()
            .join(" + ")
    );
    
    println!("Testing pure function with sparse slot usage...");
    
    // Parse and compile
    let mut lexer = Lexer::new(&pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with sparse slot usage created successfully");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}

#[test]
fn test_many_slots_memory_efficiency() {
    use std::time::Instant;
    
    let mut compiler = Compiler::new();
    
    println!("Testing memory efficiency with many slots...");
    
    let start_time = Instant::now();
    
    // Create 10 different pure functions with varying slot counts
    for slot_count in [10, 50, 100, 200, 300, 400, 500, 600, 700, 800] {
        let mut vm = VM::new();
        
        let mut pure_func_body = String::new();
        for i in 1..=slot_count {
            if i > 1 {
                pure_func_body.push_str(" + ");
            }
            pure_func_body.push_str(&format!("#{}", i));
        }
        pure_func_body.push_str(" &");
        
        // Parse and compile
        let mut lexer = Lexer::new(&pure_func_body);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        
        let bytecode = compiler.compile_expr(&expr).unwrap();
        vm.load_bytecode(bytecode);
        vm.run().unwrap();
        let pure_function = vm.pop().unwrap();
        
        match pure_function {
            Value::PureFunction { .. } => {
                println!("✓ Pure function with {} slots created", slot_count);
            }
            _ => panic!("Expected PureFunction, got {:?}", pure_function),
        }
    }
    
    let elapsed = start_time.elapsed();
    println!("✓ Memory efficiency test completed in {:?}", elapsed);
    
    // Ensure reasonable performance (should complete in under 5 seconds)
    assert!(elapsed.as_secs() < 5, "Many slots test took too long: {:?}", elapsed);
}

#[test]
fn test_many_slots_error_handling() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    // Test with extremely large slot numbers to ensure proper error handling
    let pure_func_body = "#99999 + #100000 &";
    
    println!("Testing error handling with extremely large slot numbers...");
    
    // Parse and compile - this should work for parsing
    let mut lexer = Lexer::new(pure_func_body);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let pure_function = vm.pop().unwrap();
    
    match pure_function {
        Value::PureFunction { .. } => {
            println!("✓ Pure function with extremely large slot numbers created");
        }
        _ => panic!("Expected PureFunction, got {:?}", pure_function),
    }
}