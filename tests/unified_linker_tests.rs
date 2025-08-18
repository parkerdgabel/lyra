use lyra::{
    bytecode::{Instruction, OpCode},
    compiler::{Compiler, CompilerError},
    ast::{Expr, Symbol, Number},
    vm::{Value, VirtualMachine},
    linker::{FunctionRegistry, LinkerError},
    stdlib::StandardLibrary,
};

/// Phase 4B.1: Unified Linker Architecture Tests (RED PHASE)
/// 
/// These tests define the expected behavior of a unified function resolution system
/// that handles BOTH Foreign methods AND stdlib functions through static dispatch.
///
/// **Target Architecture:**
/// ```
/// Unified Function Index Space:
/// ├─ 0-31:   Foreign Methods (Series.Length, Tensor.Add, etc.)
/// ├─ 32-63:  Stdlib Functions (Plus, Sin, Cos, etc.)  
/// ├─ 64-95:  User Functions (future)
/// └─ 96+:    Extended Functions
/// ```

#[test]
fn test_unified_registry_combines_foreign_and_stdlib() {
    // RED PHASE: This should fail because unified registry doesn't exist yet
    
    // Create unified registry that includes BOTH Foreign methods AND stdlib functions
    let unified_registry = create_unified_registry().unwrap();
    
    // Should have Foreign methods (previously 0-31)
    assert!(unified_registry.has_function("Series::Length"));
    assert!(unified_registry.has_function("Tensor::Add"));
    assert!(unified_registry.has_function("Table::RowCount"));
    
    // Should ALSO have stdlib functions (new: 32-63)
    assert!(unified_registry.has_function("Plus"));
    assert!(unified_registry.has_function("Sin"));
    assert!(unified_registry.has_function("Length")); // stdlib Length
    assert!(unified_registry.has_function("StringJoin"));
    
    // Should have total function count: Foreign (32) + Stdlib (~30) = ~62 functions
    let total_functions = unified_registry.get_total_function_count();
    assert!(total_functions >= 60); // At least 60 functions total
    assert!(total_functions <= 100); // Reasonable upper bound
}

#[test]
fn test_unified_function_indices_no_conflicts() {
    // RED PHASE: Should fail because index space management doesn't exist yet
    
    let unified_registry = create_unified_registry().unwrap();
    
    // Foreign methods should occupy indices 0-31
    let series_length_index = unified_registry.get_function_index("Series::Length").unwrap();
    assert!(series_length_index <= 31, "Foreign method should be in 0-31 range");
    
    let tensor_add_index = unified_registry.get_function_index("Tensor::Add").unwrap();
    assert!(tensor_add_index <= 31, "Foreign method should be in 0-31 range");
    
    // Stdlib functions should occupy indices 32+
    let plus_index = unified_registry.get_function_index("Plus").unwrap();
    assert!(plus_index >= 32, "Stdlib function should be in 32+ range");
    assert!(plus_index < 64, "Stdlib function should be in 32-63 range");
    
    let sin_index = unified_registry.get_function_index("Sin").unwrap();
    assert!(sin_index >= 32, "Stdlib function should be in 32+ range");
    assert!(sin_index < 64, "Stdlib function should be in 32-63 range");
    
    // No two functions should have the same index
    assert_ne!(series_length_index, plus_index);
    assert_ne!(tensor_add_index, sin_index);
    assert_ne!(plus_index, sin_index);
}

#[test]
fn test_compiler_emits_call_static_for_stdlib_functions() {
    // RED PHASE: Should fail because compiler doesn't emit CALL_STATIC for stdlib yet
    
    let mut compiler = Compiler::new(); // Should use unified registry
    
    // Arithmetic operations should use CALL_STATIC, not CALL
    // Note: Plus[2, 3] is handled by compiler as ADD opcode currently,
    // but Sin[x] should go through function registry
    let sin_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
        args: vec![Expr::Number(Number::Real(1.0))],
    };
    
    compiler.compile_expr(&sin_expr).unwrap();
    
    // Should emit CALL_STATIC instruction, not CALL
    let last_instruction = compiler.context.code.last().unwrap();
    assert_eq!(last_instruction.opcode, OpCode::CALL_STATIC, 
              "Stdlib functions should use CALL_STATIC for static dispatch");
    
    // Should have function index >= 32 (stdlib range)
    let (function_index, argc) = last_instruction.decode_call_static();
    assert!(function_index >= 32, "Sin should have stdlib function index >= 32");
    assert_eq!(argc, 1, "Sin takes 1 argument");
}

#[test]
fn test_vm_handles_stdlib_call_static() {
    // RED PHASE: Should fail because VM doesn't handle stdlib CALL_STATIC yet
    
    let mut vm = VirtualMachine::new(); // Should use unified registry
    
    // Create a CALL_STATIC instruction for stdlib function (Sin)
    let sin_index = 32; // Assuming Sin gets index 32 (first stdlib function)
    let call_static_instruction = Instruction::new_call_static(sin_index, 1).unwrap();
    
    // Push argument for Sin[1.0]
    vm.push(Value::Real(1.0));
    
    // Execute CALL_STATIC instruction
    // This should resolve to stdlib Sin function and execute it
    let result = vm.execute_instruction(&call_static_instruction);
    assert!(result.is_ok(), "VM should handle stdlib CALL_STATIC");
    
    // Result should be sin(1.0) ≈ 0.8414
    let result_value = vm.pop().unwrap();
    match result_value {
        Value::Real(val) => {
            assert!((val - 0.8414).abs() < 0.001, "Sin[1.0] should ≈ 0.8414");
        }
        _ => panic!("Sin should return Real value"),
    }
}

#[test]
fn test_eliminate_old_call_opcode() {
    // RED PHASE: Should fail because compiler still emits CALL for some functions
    
    let mut compiler = Compiler::new();
    
    // ALL function calls should use CALL_STATIC, never CALL
    let test_functions = vec![
        ("Sin", vec![Expr::Number(Number::Real(1.0))]),
        ("Length", vec![Expr::List(vec![Expr::Number(Number::Integer(1))])]),
        ("StringJoin", vec![
            Expr::String("hello".to_string()),
            Expr::String("world".to_string())
        ]),
    ];
    
    for (func_name, args) in test_functions {
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: func_name.to_string() })),
            args,
        };
        
        compiler.compile_expr(&expr).unwrap();
        
        // Check that NO CALL opcodes were emitted
        for instruction in &compiler.context.code {
            assert_ne!(instruction.opcode, OpCode::CALL, 
                      "Function {} should use CALL_STATIC, not CALL", func_name);
        }
        
        // At least one CALL_STATIC should be emitted
        let call_static_count = compiler.context.code.iter()
            .filter(|inst| inst.opcode == OpCode::CALL_STATIC)
            .count();
        assert!(call_static_count > 0, 
               "Function {} should emit at least one CALL_STATIC", func_name);
        
        compiler.context.code.clear(); // Reset for next test
    }
}

#[test]
fn test_performance_improvement_for_stdlib() {
    // RED PHASE: Should fail because stdlib functions don't have static dispatch yet
    
    use std::time::Instant;
    
    // Create large dataset for meaningful performance measurement
    let iterations = 10000;
    
    // Test new static dispatch system
    let start = Instant::now();
    for _ in 0..iterations {
        let result = execute_stdlib_function_static("Sin", &[Value::Real(1.0)]);
        assert!(result.is_ok());
    }
    let static_duration = start.elapsed();
    
    // Test old dynamic dispatch system (if we still had it)
    let start = Instant::now();
    for _ in 0..iterations {
        let result = execute_stdlib_function_dynamic("Sin", &[Value::Real(1.0)]);
        assert!(result.is_ok());
    }
    let dynamic_duration = start.elapsed();
    
    // Static dispatch should be significantly faster
    let speedup = dynamic_duration.as_nanos() as f64 / static_duration.as_nanos() as f64;
    assert!(speedup >= 10.0, 
           "Static dispatch should be ≥10x faster than dynamic dispatch, got {:.1}x", 
           speedup);
    
    println!("📊 STDLIB PERFORMANCE:");
    println!("  Static dispatch:  {:6.2}μs", static_duration.as_micros());
    println!("  Dynamic dispatch: {:6.2}μs", dynamic_duration.as_micros());
    println!("  🎯 Speedup:       {:.1}x", speedup);
}

#[test]
fn test_end_to_end_stdlib_static_dispatch() {
    // RED PHASE: Should fail because end-to-end system doesn't exist yet
    
    // Test complete pipeline: source → compile → execute with static dispatch
    let mut compiler = Compiler::new();
    
    // Complex expression using multiple stdlib functions
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
        args: vec![
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Cos".to_string() })),
                args: vec![Expr::Number(Number::Real(1.0))],
            }
        ],
    };
    
    // Compile expression
    compiler.compile_expr(&expr).unwrap();
    
    // ALL instructions should be CALL_STATIC (no CALL opcodes)
    for instruction in &compiler.context.code {
        if instruction.opcode == OpCode::CALL {
            panic!("Found CALL opcode in compiled code - should be CALL_STATIC only");
        }
    }
    
    // Execute in VM
    let mut vm = compiler.into_vm();
    let result = vm.run().unwrap();
    
    // Should get Sin[Cos[1.0]] ≈ Sin[0.5403] ≈ 0.5150
    match result {
        Value::Real(val) => {
            assert!((val - 0.5150).abs() < 0.001, 
                   "Sin[Cos[1.0]] should ≈ 0.5150, got {}", val);
        }
        _ => panic!("Expected Real result from Sin[Cos[1.0]]"),
    }
}

// ===== HELPER FUNCTIONS (TO BE IMPLEMENTED) =====

/// Create unified registry with both Foreign methods and stdlib functions
fn create_unified_registry() -> Result<UnifiedFunctionRegistry, LinkerError> {
    // This function doesn't exist yet - will be implemented in Phase 4B.2
    todo!("UnifiedFunctionRegistry not implemented yet")
}

/// Execute stdlib function using new static dispatch system
fn execute_stdlib_function_static(name: &str, args: &[Value]) -> Result<Value, LinkerError> {
    // This function doesn't exist yet - will be implemented in Phase 4B.4
    todo!("Static dispatch for stdlib functions not implemented yet")
}

/// Execute stdlib function using old dynamic dispatch system (for comparison)
fn execute_stdlib_function_dynamic(name: &str, args: &[Value]) -> Result<Value, LinkerError> {
    // This simulates the old HashMap-based lookup system
    let stdlib = StandardLibrary::new();
    if let Some(func) = stdlib.get_function(name) {
        func(args).map_err(|vm_err| LinkerError::RegistryError {
            message: format!("VM error: {:?}", vm_err),
        })
    } else {
        Err(LinkerError::FunctionNotFound {
            type_name: "Stdlib".to_string(),
            method_name: name.to_string(),
        })
    }
}

// ===== TYPES TO BE IMPLEMENTED =====

/// Unified function registry that handles both Foreign methods and stdlib functions
/// This will replace the current split system
struct UnifiedFunctionRegistry {
    // To be implemented in Phase 4B.2
}

impl UnifiedFunctionRegistry {
    fn has_function(&self, _name: &str) -> bool {
        todo!("UnifiedFunctionRegistry::has_function not implemented yet")
    }
    
    fn get_function_index(&self, _name: &str) -> Result<u16, LinkerError> {
        todo!("UnifiedFunctionRegistry::get_function_index not implemented yet")
    }
    
    fn get_total_function_count(&self) -> usize {
        todo!("UnifiedFunctionRegistry::get_total_function_count not implemented yet")
    }
}