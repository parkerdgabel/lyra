use lyra::{
    bytecode::{Instruction, OpCode},
    compiler::{CompilerContext, CompilerError, CompilerResult},
    ast::Expr,
    linker::{registry::create_global_registry, FunctionRegistry},
    foreign::{Foreign, ForeignError, LyObj},
    vm::Value,
    stdlib::data::series::{ForeignSeries, SeriesType},
};
use std::any::Any;

/// Test implementation for compile-time function resolution
/// This should replace the current runtime dynamic dispatch with compile-time resolved static calls

#[test]
fn test_compile_time_method_resolution_basic() {
    // RED PHASE: This test should initially fail because compile-time resolution doesn't exist yet
    
    // Create a simple Series method call: series.Length[]
    let series = ForeignSeries::new(vec![Value::Integer(1), Value::Integer(2)], SeriesType::Int64).unwrap();
    let lyobj = LyObj::new(Box::new(series));
    
    // This should be resolved at compile-time, not runtime
    let mut compiler = TestCompiler::new();
    
    // Method call: obj.Length[] - should resolve to static function call
    let result = compiler.compile_method_call("Length", &lyobj, &[]);
    
    // Should succeed and generate CALL_STATIC instead of CALL
    assert!(result.is_ok());
    let bytecode = result.unwrap();
    
    // Should emit CALL_STATIC opcode with function registry index
    assert_eq!(bytecode.len(), 1);
    match bytecode[0].opcode {
        OpCode::CallStatic => {
            // Should contain the function registry index for Series::Length
            let function_index = bytecode[0].operand;
            assert!(function_index < 32); // We have 32 registered functions
        }
        _ => panic!("Expected CALL_STATIC opcode, got {:?}", bytecode[0].opcode),
    }
}

#[test]
fn test_compile_time_method_validation() {
    // RED PHASE: Should fail because method validation doesn't exist yet
    
    let series = ForeignSeries::new(vec![Value::Integer(1)], SeriesType::Int64).unwrap();
    let lyobj = LyObj::new(Box::new(series));
    
    let mut compiler = TestCompiler::new();
    
    // Valid method should compile successfully
    let valid_result = compiler.compile_method_call("Length", &lyobj, &[]);
    assert!(valid_result.is_ok());
    
    // Invalid method should fail at compile-time
    let invalid_result = compiler.compile_method_call("NonExistentMethod", &lyobj, &[]);
    assert!(invalid_result.is_err());
    
    match invalid_result.unwrap_err() {
        CompilerError::UnknownMethod { type_name, method } => {
            assert_eq!(type_name, "Series");
            assert_eq!(method, "NonExistentMethod");
        }
        _ => panic!("Expected UnknownMethod error"),
    }
}

#[test]
fn test_compile_time_arity_validation() {
    // RED PHASE: Should fail because arity validation doesn't exist yet
    
    let series = ForeignSeries::new(vec![Value::Integer(1)], SeriesType::Int64).unwrap();
    let lyobj = LyObj::new(Box::new(series));
    
    let mut compiler = TestCompiler::new();
    
    // Length method expects 0 arguments
    let valid_result = compiler.compile_method_call("Length", &lyobj, &[]);
    assert!(valid_result.is_ok());
    
    // Length method with wrong arity should fail at compile-time
    let invalid_result = compiler.compile_method_call("Length", &lyobj, &[Value::Integer(1)]);
    assert!(invalid_result.is_err());
    
    match invalid_result.unwrap_err() {
        CompilerError::InvalidMethodArity { method, expected, actual } => {
            assert_eq!(method, "Length");
            assert_eq!(expected, 0);
            assert_eq!(actual, 1);
        }
        _ => panic!("Expected InvalidMethodArity error"),
    }
}

#[test]
fn test_all_foreign_types_resolve() {
    // RED PHASE: Should fail because type resolution doesn't exist yet
    
    let mut compiler = TestCompiler::new();
    
    // Test Series methods
    let series = ForeignSeries::new(vec![Value::Integer(1)], SeriesType::Int64).unwrap();
    let series_obj = LyObj::new(Box::new(series));
    
    assert!(compiler.compile_method_call("Length", &series_obj, &[]).is_ok());
    assert!(compiler.compile_method_call("Type", &series_obj, &[]).is_ok());
    assert!(compiler.compile_method_call("ToList", &series_obj, &[]).is_ok());
    
    // Test Tensor methods (when we get to Tensor)
    // This demonstrates the system should work for all Foreign types
    // TODO: Add Tensor, Table, Dataset tests once available
}

#[test]
fn test_performance_improvement_measurement() {
    // RED PHASE: Should fail because performance infrastructure doesn't exist yet
    
    use std::time::Instant;
    
    let series = ForeignSeries::new((0..1000).map(Value::Integer).collect(), SeriesType::Int64).unwrap();
    let lyobj = LyObj::new(Box::new(series));
    
    let mut compiler = TestCompiler::new();
    
    // Measure compile-time resolution performance
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = compiler.compile_method_call("Length", &lyobj, &[]);
    }
    let compile_time_duration = start.elapsed();
    
    // This should be significantly faster than runtime dispatch
    // We'll measure actual improvement in later phases
    assert!(compile_time_duration.as_micros() < 10000); // Should be very fast
}

/// Test compiler implementation for compile-time method resolution
/// This is the interface we want to implement
struct TestCompiler {
    context: CompilerContext,
    registry: FunctionRegistry,
}

impl TestCompiler {
    fn new() -> Self {
        TestCompiler {
            context: CompilerContext::new(),
            registry: create_global_registry().expect("Failed to create function registry"),
        }
    }
    
    /// Compile a method call with compile-time resolution
    /// This should replace the current runtime dispatch system
    fn compile_method_call(
        &mut self, 
        method: &str, 
        obj: &LyObj, 
        args: &[Value]
    ) -> Result<Vec<Instruction>, CompilerError> {
        // 1. Determine the object type (Series, Tensor, etc.)
        let type_name = obj.type_name();
        
        // 2. Check if method exists in registry
        if !self.registry.has_method(type_name, method) {
            return Err(CompilerError::UnknownMethod {
                type_name: type_name.to_string(),
                method: method.to_string(),
            });
        }
        
        // For now, we'll use a simple approach where we simulate static resolution
        // by checking arity and generating a placeholder CALL_STATIC instruction
        
        // For Series.Length method, we expect 0 arguments
        let expected_arity = match (type_name, method) {
            ("Series", "Length") => 0,
            ("Series", "Type") => 0,
            ("Series", "ToList") => 0,
            ("Series", "IsEmpty") => 0,
            ("Series", "Get") => 1,
            ("Series", "Append") => 1,
            ("Series", "Set") => 2,
            ("Series", "Slice") => 2,
            _ => {
                return Err(CompilerError::UnknownMethod {
                    type_name: type_name.to_string(),
                    method: method.to_string(),
                });
            }
        };
            
        // Validate argument count matches expected arity
        if args.len() != expected_arity {
            return Err(CompilerError::InvalidMethodArity {
                method: method.to_string(),
                expected: expected_arity,
                actual: args.len(),
            });
        }
        
        // Generate CALL_STATIC instruction with a placeholder function index
        // In a real implementation, this would be resolved from the Function Registry
        let function_index = match (type_name, method) {
            ("Series", "Length") => 0,
            ("Series", "Type") => 1,
            ("Series", "ToList") => 2,
            ("Series", "IsEmpty") => 3,
            ("Series", "Get") => 4,
            ("Series", "Append") => 5,
            ("Series", "Set") => 6,
            ("Series", "Slice") => 7,
            _ => 0, // Should not reach here due to check above
        };
        let argc = args.len() as u8;
        
        let instruction = Instruction::new_call_static(function_index, argc)
            .map_err(|_| CompilerError::TooManyConstants)?; // Reuse existing error type
            
        Ok(vec![instruction])
    }
}

// Error types are now implemented in the main CompilerError enum

#[test]
fn test_call_static_opcode_exists() {
    // RED PHASE: This should fail because CALL_STATIC doesn't exist yet
    
    // This test ensures we add the CALL_STATIC opcode to the bytecode system
    let instruction = Instruction::new_call_static(5, 2); // function_index=5, argc=2
    assert!(instruction.is_ok());
    
    let inst = instruction.unwrap();
    assert_eq!(inst.opcode, OpCode::CallStatic);
    
    let (function_index, argc) = inst.decode_call_static();
    assert_eq!(function_index, 5);
    assert_eq!(argc, 2);
}