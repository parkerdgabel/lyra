//! Integration tests for Result and Option types in Lyra's stdlib
//! 
//! These tests verify that Result and Option types work correctly 
//! with the VM and standard library integration.

use lyra::{
    stdlib::StandardLibrary,
    vm::{Value, VirtualMachine},
    compiler::Compiler,
    parser::Parser,
    lexer::Lexer,
};

#[test]
fn test_result_constructors_integration() {
    let mut vm = VirtualMachine::new();
    let stdlib = StandardLibrary::new();
    
    // Test Ok constructor
    let ok_fn = stdlib.get_function("Ok").expect("Ok function should be registered");
    let ok_result = ok_fn(&[Value::Integer(42)]).expect("Ok constructor should work");
    
    // Verify it's a proper Result
    if let Value::List(items) = &ok_result {
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::Symbol("Ok".to_string()));
        assert_eq!(items[1], Value::Integer(42));
    } else {
        panic!("Ok result should be a list");
    }
    
    // Test Error constructor
    let error_fn = stdlib.get_function("Error").expect("Error function should be registered");
    let error_result = error_fn(&[Value::String("test error".to_string())]).expect("Error constructor should work");
    
    // Verify it's a proper Result
    if let Value::List(items) = &error_result {
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::Symbol("Error".to_string()));
        assert_eq!(items[1], Value::String("test error".to_string()));
    } else {
        panic!("Error result should be a list");
    }
}

#[test]
fn test_option_constructors_integration() {
    let mut vm = VirtualMachine::new();
    let stdlib = StandardLibrary::new();
    
    // Test Some constructor
    let some_fn = stdlib.get_function("Some").expect("Some function should be registered");
    let some_result = some_fn(&[Value::Integer(42)]).expect("Some constructor should work");
    
    // Verify it's a proper Option
    if let Value::List(items) = &some_result {
        assert_eq!(items.len(), 2);
        assert_eq!(items[0], Value::Symbol("Some".to_string()));
        assert_eq!(items[1], Value::Integer(42));
    } else {
        panic!("Some result should be a list");
    }
    
    // Test None constructor
    let none_fn = stdlib.get_function("None").expect("None function should be registered");
    let none_result = none_fn(&[]).expect("None constructor should work");
    
    // Verify it's a proper Option
    if let Value::List(items) = &none_result {
        assert_eq!(items.len(), 1);
        assert_eq!(items[0], Value::Symbol("None".to_string()));
    } else {
        panic!("None result should be a list");
    }
}

#[test]
fn test_result_methods_integration() {
    let stdlib = StandardLibrary::new();
    
    // Create test values
    let ok_result = stdlib.get_function("Ok").unwrap()(&[Value::Integer(42)]).unwrap();
    let error_result = stdlib.get_function("Error").unwrap()(&[Value::String("error".to_string())]).unwrap();
    
    // Test ResultIsOk
    let is_ok_fn = stdlib.get_function("ResultIsOk").expect("ResultIsOk should be registered");
    assert_eq!(is_ok_fn(&[ok_result.clone()]).unwrap(), Value::Boolean(true));
    assert_eq!(is_ok_fn(&[error_result.clone()]).unwrap(), Value::Boolean(false));
    
    // Test ResultIsError
    let is_error_fn = stdlib.get_function("ResultIsError").expect("ResultIsError should be registered");
    assert_eq!(is_error_fn(&[ok_result.clone()]).unwrap(), Value::Boolean(false));
    assert_eq!(is_error_fn(&[error_result.clone()]).unwrap(), Value::Boolean(true));
    
    // Test ResultUnwrap
    let unwrap_fn = stdlib.get_function("ResultUnwrap").expect("ResultUnwrap should be registered");
    assert_eq!(unwrap_fn(&[ok_result.clone()]).unwrap(), Value::Integer(42));
    assert!(unwrap_fn(&[error_result.clone()]).is_err());
    
    // Test ResultUnwrapOr
    let unwrap_or_fn = stdlib.get_function("ResultUnwrapOr").expect("ResultUnwrapOr should be registered");
    assert_eq!(unwrap_or_fn(&[ok_result.clone(), Value::Integer(0)]).unwrap(), Value::Integer(42));
    assert_eq!(unwrap_or_fn(&[error_result.clone(), Value::Integer(0)]).unwrap(), Value::Integer(0));
}

#[test]
fn test_option_methods_integration() {
    let stdlib = StandardLibrary::new();
    
    // Create test values
    let some_option = stdlib.get_function("Some").unwrap()(&[Value::Integer(42)]).unwrap();
    let none_option = stdlib.get_function("None").unwrap()(&[]).unwrap();
    
    // Test OptionIsSome
    let is_some_fn = stdlib.get_function("OptionIsSome").expect("OptionIsSome should be registered");
    assert_eq!(is_some_fn(&[some_option.clone()]).unwrap(), Value::Boolean(true));
    assert_eq!(is_some_fn(&[none_option.clone()]).unwrap(), Value::Boolean(false));
    
    // Test OptionIsNone
    let is_none_fn = stdlib.get_function("OptionIsNone").expect("OptionIsNone should be registered");
    assert_eq!(is_none_fn(&[some_option.clone()]).unwrap(), Value::Boolean(false));
    assert_eq!(is_none_fn(&[none_option.clone()]).unwrap(), Value::Boolean(true));
    
    // Test OptionUnwrap
    let unwrap_fn = stdlib.get_function("OptionUnwrap").expect("OptionUnwrap should be registered");
    assert_eq!(unwrap_fn(&[some_option.clone()]).unwrap(), Value::Integer(42));
    assert!(unwrap_fn(&[none_option.clone()]).is_err());
    
    // Test OptionUnwrapOr
    let unwrap_or_fn = stdlib.get_function("OptionUnwrapOr").expect("OptionUnwrapOr should be registered");
    assert_eq!(unwrap_or_fn(&[some_option.clone(), Value::Integer(0)]).unwrap(), Value::Integer(42));
    assert_eq!(unwrap_or_fn(&[none_option.clone(), Value::Integer(0)]).unwrap(), Value::Integer(0));
}

#[test]
fn test_result_option_type_safety() {
    let stdlib = StandardLibrary::new();
    
    // Test that Result functions reject non-Result values
    let is_ok_fn = stdlib.get_function("ResultIsOk").unwrap();
    assert_eq!(is_ok_fn(&[Value::Integer(42)]).unwrap(), Value::Boolean(false));
    assert_eq!(is_ok_fn(&[Value::String("test".to_string())]).unwrap(), Value::Boolean(false));
    
    // Test that Option functions reject non-Option values  
    let is_some_fn = stdlib.get_function("OptionIsSome").unwrap();
    assert_eq!(is_some_fn(&[Value::Integer(42)]).unwrap(), Value::Boolean(false));
    assert_eq!(is_some_fn(&[Value::String("test".to_string())]).unwrap(), Value::Boolean(false));
    
    // Test error handling for invalid inputs
    let unwrap_fn = stdlib.get_function("ResultUnwrap").unwrap();
    assert!(unwrap_fn(&[Value::Integer(42)]).is_err());
    
    let option_unwrap_fn = stdlib.get_function("OptionUnwrap").unwrap();
    assert!(option_unwrap_fn(&[Value::Integer(42)]).is_err());
}

#[test]
fn test_result_option_arity_checking() {
    let stdlib = StandardLibrary::new();
    
    // Test Ok with wrong arity
    let ok_fn = stdlib.get_function("Ok").unwrap();
    assert!(ok_fn(&[]).is_err()); // Too few args
    assert!(ok_fn(&[Value::Integer(1), Value::Integer(2)]).is_err()); // Too many args
    
    // Test None with wrong arity
    let none_fn = stdlib.get_function("None").unwrap();
    assert!(none_fn(&[Value::Integer(1)]).is_err()); // Too many args
    
    // Test ResultIsOk with wrong arity
    let is_ok_fn = stdlib.get_function("ResultIsOk").unwrap();
    assert!(is_ok_fn(&[]).is_err()); // Too few args
    assert!(is_ok_fn(&[Value::Integer(1), Value::Integer(2)]).is_err()); // Too many args
    
    // Test ResultUnwrapOr with wrong arity
    let unwrap_or_fn = stdlib.get_function("ResultUnwrapOr").unwrap();
    assert!(unwrap_or_fn(&[Value::Integer(1)]).is_err()); // Too few args
    assert!(unwrap_or_fn(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)]).is_err()); // Too many args
}