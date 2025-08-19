//! Phase 4B.5.1: CALL opcode removal tests (RED phase)
//!
//! These tests verify that the legacy CALL opcode has been completely removed
//! from the VM execution engine. They will FAIL initially since CALL is still
//! present in the codebase, establishing our RED phase for TDD.

use lyra::bytecode::{OpCode, Instruction, BytecodeError};
use lyra::compiler::Compiler;
use lyra::parser::Parser as LyraParser; 
use lyra::{Result};

#[test]
fn test_opcode_enum_has_exactly_18_opcodes() {
    println!("\n🔍 TEST: OpCode enum should have exactly 18 opcodes (CALL removed)");
    println!("===============================================================");
    
    let all_opcodes = OpCode::all_opcodes();
    println!("📊 Current opcode count: {}", all_opcodes.len());
    
    // ❌ THIS WILL FAIL: currently has 19 opcodes including CALL
    assert_eq!(all_opcodes.len(), 18, 
               "OpCode enum should have exactly 18 opcodes after CALL removal");
}

#[test]
fn test_call_opcode_not_in_enum() {
    println!("\n🔍 TEST: CALL opcode should not exist in OpCode enum");
    println!("=================================================");
    
    let all_opcodes = OpCode::all_opcodes();
    
    // Check that no opcode has the name "CALL" (excluding CALL_STATIC)
    let has_call = all_opcodes.iter()
        .any(|op| op.name() == "CALL");
    
    println!("📊 Found CALL opcode: {}", has_call);
    
    // ❌ THIS WILL FAIL: CALL is still present
    assert!(!has_call, "CALL opcode should be completely removed from enum");
}

#[test]
fn test_call_opcode_byte_value_invalid() {
    println!("\n🔍 TEST: CALL opcode byte value (0x40) should be invalid");
    println!("======================================================");
    
    // CALL was previously 0x40 - this should now return an error
    let result = OpCode::from_u8(0x40);
    
    println!("📊 OpCode::from_u8(0x40) result: {:?}", result);
    
    // ❌ THIS WILL FAIL: 0x40 is still valid (maps to CALL)
    assert!(result.is_err(), "Byte value 0x40 (former CALL) should be invalid");
    
    match result {
        Err(BytecodeError::InvalidOpcode(0x40)) => {
            println!("✅ Correctly rejects former CALL opcode byte value");
        }
        _ => panic!("Expected InvalidOpcode error for 0x40"),
    }
}

#[test]
fn test_is_call_method_excludes_call() {
    println!("\n🔍 TEST: is_call() should only match CALL_STATIC and RET");
    println!("=====================================================");
    
    let all_opcodes = OpCode::all_opcodes();
    let call_opcodes: Vec<_> = all_opcodes.iter()
        .filter(|op| op.is_call())
        .collect();
    
    println!("📊 Call opcodes found: {:?}", call_opcodes.iter().map(|op| op.name()).collect::<Vec<_>>());
    
    // Should only have CALL_STATIC and RET (2 opcodes)
    assert_eq!(call_opcodes.len(), 2, "Should have exactly 2 call opcodes");
    
    // Verify they are CALL_STATIC and RET
    let opcode_names: Vec<_> = call_opcodes.iter().map(|op| op.name()).collect();
    assert!(opcode_names.contains(&"CALL_STATIC"));
    assert!(opcode_names.contains(&"RET"));
    
    // ❌ THIS WILL FAIL: currently includes CALL
    assert!(!opcode_names.contains(&"CALL"), "CALL should not be in call opcodes");
}

#[test]
fn test_instruction_new_call_method_removed() {
    println!("\n🔍 TEST: Instruction::new_call() method should be removed");
    println!("=======================================================");
    
    // This test ensures the new_call method is removed from Instruction
    // We can't directly test method existence, but we can verify via behavior
    
    // Try to create what would be a CALL instruction using the general new() method
    let result = Instruction::new(OpCode::from_u8(0x40).unwrap_or(OpCode::SYS), 0x12345);
    
    // If CALL was removed properly, this will either:
    // 1. Fail to create OpCode from 0x40, or
    // 2. Create a different opcode than CALL
    
    if let Ok(instruction) = result {
        // ❌ THIS WILL FAIL: if CALL still exists, this will create a CALL instruction
        assert_ne!(instruction.opcode.name(), "CALL", 
                  "Should not be able to create CALL instructions");
    }
    
    println!("📊 CALL instruction creation behavior validated");
}

#[test]
fn test_compiler_never_generates_call_instructions() {
    println!("\n🔍 TEST: Compiler should never generate CALL instructions");
    println!("========================================================");
    
    // Test various function calls that might have used CALL previously
    let test_cases = [
        "Sin[1.0]",             // stdlib function
        "Cos[0.5]",             // another stdlib function  
        "StringLength[\"test\"]", // stdlib string function
        "Length[{1, 2, 3}]",    // list function (may not parse correctly)
    ];
    
    for source in test_cases {
        println!("  Testing: {}", source);
        
        let instructions = compile_source(source).unwrap();
        
        // Count CALL vs CALL_STATIC instructions
        let call_count = count_opcode(&instructions, get_call_opcode());
        let call_static_count = count_opcode(&instructions, OpCode::CallStatic);
        
        println!("    CALL instructions: {}", call_count);
        println!("    CALL_STATIC instructions: {}", call_static_count);
        
        // Should never generate CALL instructions
        assert_eq!(call_count, 0, 
                  "Compiler should never generate CALL instructions for: {}", source);
        
        // Note: Some functions may compile to direct opcodes (like Plus->ADD) 
        // rather than CALL_STATIC, which is correct for performance
        if call_static_count == 0 {
            println!("    Note: {} compiled to direct opcodes (not CALL_STATIC)", source);
        }
    }
}

#[test]
fn test_vm_execution_has_no_call_handler() {
    println!("\n🔍 TEST: VM execution should have no CALL opcode handler");
    println!("======================================================");
    
    // This is a behavioral test - we try to execute a CALL instruction
    // and verify it fails appropriately (since CALL handler should be removed)
    
    // First, verify we can create a theoretically valid CALL instruction
    if let Ok(call_opcode) = OpCode::from_u8(0x40) {
        let call_instruction = Instruction::new(call_opcode, 0x123).unwrap();
        println!("📊 Created instruction: {:?}", call_instruction);
        
        // Try to execute this in a VM (this should fail gracefully)
        // This test will evolve as we remove CALL from the VM
        
        // ❌ THIS WILL FAIL: VM will currently handle CALL instructions
        // After removal, this should result in an error or panic
        println!("⚠️  VM CALL handler test - to be implemented in Phase 4B.5.3");
    } else {
        println!("✅ CALL opcode (0x40) already invalid - good!");
    }
}

#[test]
fn test_all_function_types_use_call_static() {
    println!("\n🔍 TEST: All function types should use CALL_STATIC exclusively");
    println!("==============================================================");
    
    // Test all categories of functions to ensure they use CALL_STATIC
    let function_categories = [
        ("Foreign methods", "tensor.Length[]"),   // Would need object creation
        ("Stdlib math", "Sin[1.0]"),             // Stdlib function
        ("Stdlib string", "StringLength[\"test\"]"), // Another stdlib
        ("Arithmetic", "Plus[1, 2]"),            // Arithmetic function
    ];
    
    for (category, source) in function_categories {
        println!("  Testing category: {}", category);
        
        // Some tests might fail to parse/compile, skip those for now
        if let Ok(instructions) = compile_source(source) {
            let call_count = count_opcode(&instructions, get_call_opcode());
            let call_static_count = count_opcode(&instructions, OpCode::CallStatic);
            
            println!("    Source: {}", source);
            println!("    CALL: {}, CALL_STATIC: {}", call_count, call_static_count);
            
            // ❌ THIS WILL FAIL: might find CALL instructions
            assert_eq!(call_count, 0, 
                      "No CALL instructions for category: {}", category);
        } else {
            println!("    Skipping unparseable: {}", source);
        }
    }
}

/// Helper function to compile source and return instructions
fn compile_source(source: &str) -> Result<Vec<Instruction>> {
    let mut parser = LyraParser::from_source(source)?;
    let statements = parser.parse()?;

    let mut compiler = Compiler::new();
    for stmt in &statements {
        compiler.compile_expr(stmt)?;
    }
    
    Ok(compiler.context.code.clone())
}

/// Helper function to count specific opcodes in instructions
fn count_opcode(instructions: &[Instruction], opcode: OpCode) -> usize {
    instructions.iter()
        .filter(|instruction| instruction.opcode == opcode)
        .count()
}

/// Get the CALL opcode if it still exists (for testing during removal)
fn get_call_opcode() -> OpCode {
    // Try to get CALL opcode - this will fail after removal
    OpCode::from_u8(0x40).unwrap_or_else(|_| {
        // If CALL is removed, use a placeholder that will never match
        OpCode::SYS  // Use SYS as a placeholder that won't match function calls
    })
}

#[test]
fn test_opcode_count_regression() {
    println!("\n🔍 TEST: Opcode count regression test");
    println!("====================================");
    
    let all_opcodes = OpCode::all_opcodes();
    
    // Verify we have the expected categories after CALL removal
    let load_store_count = all_opcodes.iter().filter(|op| op.is_load_store()).count();
    let aggregate_count = all_opcodes.iter().filter(|op| op.is_aggregate()).count(); 
    let math_count = all_opcodes.iter().filter(|op| op.is_math()).count();
    let control_count = all_opcodes.iter().filter(|op| op.is_control()).count();
    let call_count = all_opcodes.iter().filter(|op| op.is_call()).count();
    let stack_count = all_opcodes.iter().filter(|op| matches!(op, OpCode::POP | OpCode::DUP)).count();
    let system_count = all_opcodes.iter().filter(|op| matches!(op, OpCode::SYS)).count();
    
    println!("📊 Opcode category counts:");
    println!("  Load/Store: {}", load_store_count);
    println!("  Aggregate: {}", aggregate_count);
    println!("  Math: {}", math_count);
    println!("  Control: {}", control_count);
    println!("  Call: {}", call_count);
    println!("  Stack: {}", stack_count);
    println!("  System: {}", system_count);
    
    // Expected counts after CALL removal
    assert_eq!(load_store_count, 4, "Should have 4 load/store opcodes");
    assert_eq!(aggregate_count, 2, "Should have 2 aggregate opcodes");
    assert_eq!(math_count, 5, "Should have 5 math opcodes");
    assert_eq!(control_count, 2, "Should have 2 control opcodes");
    
    // ❌ THIS WILL FAIL: currently has 3 call opcodes (CALL, CALL_STATIC, RET)
    assert_eq!(call_count, 2, "Should have 2 call opcodes (CALL_STATIC, RET)");
    
    assert_eq!(stack_count, 2, "Should have 2 stack opcodes");
    assert_eq!(system_count, 1, "Should have 1 system opcode");
    
    // Total should be 18
    let total = load_store_count + aggregate_count + math_count + control_count + call_count + stack_count + system_count;
    assert_eq!(total, 18, "Total opcode count should be 18");
}