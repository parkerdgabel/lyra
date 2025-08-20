use lyra::error::{LyraError, LyraResult};
use lyra::vm::VmError;
use lyra::foreign::ForeignError;
use lyra::compiler::CompilerError;

/// Test that the unified LyraError can handle all error types with proper chaining
#[test]
fn test_lyra_error_from_vm_error() {
    let vm_error = VmError::Runtime("Test runtime error".to_string());
    let lyra_error = LyraError::from(vm_error);
    
    // Should be able to access the original VM error
    if let LyraError::Vm(original) = lyra_error {
        assert!(matches!(original, VmError::Runtime(_)));
    } else {
        panic!("Expected LyraError::Vm variant");
    }
}

#[test]
fn test_lyra_error_from_foreign_error() {
    let foreign_error = ForeignError::UnknownMethod {
        type_name: "TestType".to_string(),
        method: "TestMethod".to_string(),
    };
    let lyra_error = LyraError::from(foreign_error.clone());
    
    // Should be able to access the original Foreign error
    if let LyraError::Foreign(original) = lyra_error {
        assert_eq!(original, foreign_error);
    } else {
        panic!("Expected LyraError::Foreign variant");
    }
}

#[test]
fn test_lyra_error_from_compiler_error() {
    let compiler_error = CompilerError::UnknownFunction("TestFunc".to_string());
    let lyra_error = LyraError::from(compiler_error.clone());
    
    // Should be able to access the original Compiler error
    if let LyraError::Compiler(original) = lyra_error {
        assert_eq!(original, compiler_error);
    } else {
        panic!("Expected LyraError::Compiler variant");
    }
}

#[test]
fn test_lyra_error_chaining() {
    let vm_error = VmError::TypeError {
        expected: "Integer".to_string(),
        actual: "String".to_string(),
    };
    let lyra_error = LyraError::from(vm_error);
    
    // Should preserve error chain information
    let error_string = format!("{}", lyra_error);
    assert!(error_string.contains("Type error"));
    assert!(error_string.contains("expected Integer"));
    assert!(error_string.contains("got String"));
}

#[test]
fn test_lyra_error_context() {
    let base_error = VmError::DivisionByZero;
    let lyra_error = LyraError::from(base_error).with_context("while evaluating expression");
    
    // Should include context information
    let error_string = format!("{}", lyra_error);
    assert!(error_string.contains("while evaluating expression"));
    assert!(error_string.contains("Division by zero"));
}

#[test]
fn test_lyra_result_type_alias() {
    // Should be able to use LyraResult as a type alias
    fn test_function() -> LyraResult<i32> {
        Ok(42)
    }
    
    let result = test_function();
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_lyra_error_conversion_from_io_error() {
    use std::io::{Error as IoError, ErrorKind};
    
    let io_error = IoError::new(ErrorKind::NotFound, "File not found");
    let lyra_error = LyraError::from(io_error);
    
    // Should convert IO errors properly
    if let LyraError::Io(_) = lyra_error {
        // Success
    } else {
        panic!("Expected LyraError::Io variant");
    }
}

#[test]
fn test_error_display_formatting() {
    let errors = vec![
        LyraError::Parse {
            message: "Unexpected token".to_string(),
            position: 42,
        },
        LyraError::Runtime {
            message: "Test runtime error".to_string(),
        },
        LyraError::Type {
            expected: "Integer".to_string(),
            actual: "String".to_string(),
        },
    ];
    
    for error in errors {
        let display_string = format!("{}", error);
        assert!(!display_string.is_empty());
        // Each error should have a meaningful display format
    }
}

#[test]
fn test_error_categorization() {
    let runtime_error = LyraError::Runtime {
        message: "Test error".to_string(),
    };
    
    // Should be able to categorize errors
    assert!(runtime_error.is_runtime_error());
    assert!(!runtime_error.is_parse_error());
    assert!(!runtime_error.is_type_error());
    
    let type_error = LyraError::Type {
        expected: "Integer".to_string(),
        actual: "String".to_string(),
    };
    
    assert!(type_error.is_type_error());
    assert!(!type_error.is_runtime_error());
}

#[test]
fn test_error_recovery_information() {
    let parse_error = LyraError::Parse {
        message: "Expected bracket".to_string(),
        position: 15,
    };
    
    // Should provide recovery suggestions
    let suggestions = parse_error.recovery_suggestions();
    assert!(!suggestions.is_empty());
    
    // Should include helpful hints for common errors
    assert!(suggestions.iter().any(|s| s.contains("bracket")));
}