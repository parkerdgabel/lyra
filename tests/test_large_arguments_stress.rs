//! Phase 8A-1B: Large arguments stress tests (massive tensors, deep data structures)
//! Tests system performance and memory handling with extremely large data structures

use lyra::compiler::Compiler;
use lyra::parser::Parser;
use lyra::lexer::Lexer;
use lyra::vm::VM;
use lyra::value::Value;
use std::time::Instant;

#[test]
fn test_large_tensor_creation() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing large tensor creation...");
    
    // Test 1: Large 1D tensor (10,000 elements)
    let large_1d = format!("Array[{}]", 
        (1..=10000).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
    );
    
    let start = Instant::now();
    let mut lexer = Lexer::new(&large_1d);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Large 1D tensor (10K elements) created in {:?}", elapsed);
    
    // Verify it's a tensor
    match result {
        Value::Tensor(_) => println!("  ✓ Successfully created tensor"),
        _ => panic!("Expected tensor, got {:?}", result),
    }
}

#[test]
fn test_large_2d_tensor() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing large 2D tensor creation...");
    
    // Test 2: Large 2D tensor (100x100 = 10,000 elements)
    let mut rows = Vec::new();
    for i in 1..=100 {
        let row = (1..=100).map(|j| (i * 100 + j).to_string()).collect::<Vec<_>>().join(", ");
        rows.push(format!("{{{}}}", row));
    }
    let large_2d = format!("Array[{{{}}}]", rows.join(", "));
    
    let start = Instant::now();
    let mut lexer = Lexer::new(&large_2d);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Large 2D tensor (100x100) created in {:?}", elapsed);
    
    match result {
        Value::Tensor(_) => println!("  ✓ Successfully created 2D tensor"),
        _ => panic!("Expected tensor, got {:?}", result),
    }
}

#[test]
fn test_deeply_nested_lists() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing deeply nested list structures...");
    
    // Test 3: Deeply nested lists (1000 levels deep)
    let mut nested = "1".to_string();
    for _ in 0..1000 {
        nested = format!("{{{}}}", nested);
    }
    
    let start = Instant::now();
    let mut lexer = Lexer::new(&nested);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Deeply nested list (1000 levels) created in {:?}", elapsed);
    
    match result {
        Value::List(_) => println!("  ✓ Successfully created deeply nested list"),
        _ => panic!("Expected list, got {:?}", result),
    }
}

#[test]
fn test_large_string_arguments() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing large string arguments...");
    
    // Test 4: Very large string (1MB)
    let large_string = "a".repeat(1_000_000);
    let string_expr = format!("\"{}\"", large_string);
    
    let start = Instant::now();
    let mut lexer = Lexer::new(&string_expr);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Large string (1MB) processed in {:?}", elapsed);
    
    match result {
        Value::String(s) => {
            assert_eq!(s.len(), 1_000_000);
            println!("  ✓ Successfully created large string");
        }
        _ => panic!("Expected string, got {:?}", result),
    }
}

#[test]
fn test_complex_nested_structures() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing complex nested data structures...");
    
    // Test 5: Complex nested structure (mix of lists, tensors, strings)
    let complex_structure = r#"
    {
        Array[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}],
        {"nested", {"deeply", {"very", {"extremely", "deep"}}}},
        Array[{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}],
        "large_string_component_here",
        {
            Array[{{10, 20}, {30, 40}}],
            {"another", "level", "of", "nesting"}
        }
    }
    "#;
    
    let start = Instant::now();
    let mut lexer = Lexer::new(complex_structure);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Complex nested structure created in {:?}", elapsed);
    
    match result {
        Value::List(_) => println!("  ✓ Successfully created complex structure"),
        _ => panic!("Expected list, got {:?}", result),
    }
}

#[test]
fn test_large_function_arguments() {
    let mut compiler = Compiler::new();
    let mut vm = VM::new();
    
    println!("Testing functions with large argument lists...");
    
    // Test 6: Function call with many large arguments
    let large_args: Vec<String> = (1..=100).map(|i| {
        if i % 3 == 0 {
            format!("Array[{}]", (1..=100).map(|j| (i * j).to_string()).collect::<Vec<_>>().join(", "))
        } else if i % 2 == 0 {
            format!("\"string_arg_{}\"", "x".repeat(1000))
        } else {
            (1..=50).map(|j| j.to_string()).collect::<Vec<_>>().join(", ")
        }
    }).collect();
    
    let function_call = format!("Length[{{{}}}]", large_args.join(", "));
    
    let start = Instant::now();
    let mut lexer = Lexer::new(&function_call);
    let tokens = lexer.tokenize().unwrap();
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().unwrap();
    
    let bytecode = compiler.compile_expr(&expr).unwrap();
    vm.load_bytecode(bytecode);
    vm.run().unwrap();
    let result = vm.pop().unwrap();
    
    let elapsed = start.elapsed();
    println!("✓ Function with large arguments processed in {:?}", elapsed);
    
    match result {
        Value::Integer(len) => {
            assert_eq!(len, 100);
            println!("  ✓ Function correctly processed {} arguments", len);
        }
        _ => panic!("Expected integer, got {:?}", result),
    }
}

#[test]
fn test_memory_pressure_with_large_data() {
    println!("Testing memory pressure with multiple large data structures...");
    
    let start = Instant::now();
    let mut compilers = Vec::new();
    let mut vms = Vec::new();
    
    // Create multiple VMs with large data to test memory pressure
    for i in 0..10 {
        let mut compiler = Compiler::new();
        let mut vm = VM::new();
        
        // Each VM gets a large tensor
        let tensor_size = 1000;
        let large_tensor = format!("Array[{}]", 
            (1..=tensor_size).map(|j| (i * tensor_size + j).to_string())
                .collect::<Vec<_>>().join(", ")
        );
        
        let mut lexer = Lexer::new(&large_tensor);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        
        let bytecode = compiler.compile_expr(&expr).unwrap();
        vm.load_bytecode(bytecode);
        vm.run().unwrap();
        let _result = vm.pop().unwrap();
        
        compilers.push(compiler);
        vms.push(vm);
    }
    
    let elapsed = start.elapsed();
    println!("✓ Memory pressure test with {} VMs completed in {:?}", vms.len(), elapsed);
    println!("  ✓ All {} large tensors created successfully", vms.len());
}

#[test]
fn test_performance_scaling() {
    println!("Testing performance scaling with increasing data sizes...");
    
    let sizes = vec![100, 500, 1000, 2000, 5000];
    
    for size in sizes {
        let mut compiler = Compiler::new();
        let mut vm = VM::new();
        
        let large_list = format!("{{{}}}",
            (1..=size).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")
        );
        
        let start = Instant::now();
        let mut lexer = Lexer::new(&large_list);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expr = parser.parse_expression().unwrap();
        
        let bytecode = compiler.compile_expr(&expr).unwrap();
        vm.load_bytecode(bytecode);
        vm.run().unwrap();
        let _result = vm.pop().unwrap();
        
        let elapsed = start.elapsed();
        println!("  Size {}: {:?}", size, elapsed);
        
        // Performance should scale reasonably (not exponentially)
        assert!(elapsed.as_millis() < (size as u128) * 2, 
                "Performance degraded too much for size {}", size);
    }
    
    println!("✓ Performance scaling test completed - linear scaling maintained");
}