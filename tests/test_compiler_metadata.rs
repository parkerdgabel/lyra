//! Test the compiler's type metadata integration

use lyra::compiler::{Compiler, CompilerContext, SimpleFunctionSignature};

#[test]
fn test_compiler_context_type_metadata() {
    let mut context = CompilerContext::new();
    
    // Create a function signature
    let sig = SimpleFunctionSignature {
        name: "add".to_string(),
        param_count: 2,
        is_typed: true,
    };
    
    // Register the signature
    context.register_type_signature(sig);
    
    // Verify it's stored
    assert!(context.has_type_metadata("add"));
    assert!(!context.has_type_metadata("multiply"));
    
    let retrieved = context.get_type_signature("add").unwrap();
    assert_eq!(retrieved.name, "add");
    assert_eq!(retrieved.param_count, 2);
    assert!(retrieved.is_typed);
}

#[test]
fn test_compiler_type_registry() {
    let mut compiler = Compiler::new();
    
    // Create a function signature
    let sig = SimpleFunctionSignature {
        name: "test_func".to_string(),
        param_count: 1,
        is_typed: true,
    };
    
    // Register via the compiler's convenience method
    compiler.register_type_signature(sig);
    
    // Verify it's stored in context
    assert!(compiler.context.has_type_metadata("test_func"));
    
    // Test validation
    let result = compiler.validate_function_call("test_func", 1);
    assert!(result.is_ok());
    
    let result = compiler.validate_function_call("test_func", 2);
    assert!(result.is_err());
    
    // Check stats
    let stats_str = compiler.type_metadata_stats();
    assert!(stats_str.contains("1 functions"));
    assert!(stats_str.contains("1 typed"));
}

#[test]
fn test_compiler_multiple_functions() {
    let mut compiler = Compiler::new();
    
    // Add typed function
    let typed_sig = SimpleFunctionSignature {
        name: "typed_func".to_string(),
        param_count: 1,
        is_typed: true,
    };
    compiler.register_type_signature(typed_sig);
    
    // Add untyped function
    let untyped_sig = SimpleFunctionSignature {
        name: "untyped_func".to_string(),
        param_count: 1,
        is_typed: false,
    };
    compiler.register_type_signature(untyped_sig);
    
    // Check stats
    let stats_str = compiler.type_metadata_stats();
    assert!(stats_str.contains("2 functions"));
    assert!(stats_str.contains("1 typed"));
    assert!(stats_str.contains("1 untyped"));
}