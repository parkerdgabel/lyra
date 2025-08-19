//! Test that Result and Option functions are available in StandardLibrary

use lyra::stdlib::StandardLibrary;

#[test]
fn test_result_option_functions_registered() {
    let stdlib = StandardLibrary::new();
    let function_names = stdlib.function_names();
    
    // Result constructors
    assert!(function_names.iter().any(|name| name.as_str() == "Ok"), "Ok function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "Error"), "Error function should be registered");
    
    // Option constructors
    assert!(function_names.iter().any(|name| name.as_str() == "Some"), "Some function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "None"), "None function should be registered");
    
    // Result methods
    assert!(function_names.iter().any(|name| name.as_str() == "ResultIsOk"), "ResultIsOk function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "ResultIsError"), "ResultIsError function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "ResultUnwrap"), "ResultUnwrap function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "ResultUnwrapOr"), "ResultUnwrapOr function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "ResultMap"), "ResultMap function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "ResultAndThen"), "ResultAndThen function should be registered");
    
    // Option methods
    assert!(function_names.iter().any(|name| name.as_str() == "OptionIsSome"), "OptionIsSome function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "OptionIsNone"), "OptionIsNone function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "OptionUnwrap"), "OptionUnwrap function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "OptionUnwrapOr"), "OptionUnwrapOr function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "OptionMap"), "OptionMap function should be registered");
    assert!(function_names.iter().any(|name| name.as_str() == "OptionAndThen"), "OptionAndThen function should be registered");
}

#[test]
fn test_function_lookup_works() {
    let stdlib = StandardLibrary::new();
    
    // Test that we can successfully lookup Result/Option functions
    assert!(stdlib.get_function("Ok").is_some(), "Ok function should be retrievable");
    assert!(stdlib.get_function("Error").is_some(), "Error function should be retrievable");
    assert!(stdlib.get_function("Some").is_some(), "Some function should be retrievable");
    assert!(stdlib.get_function("None").is_some(), "None function should be retrievable");
    assert!(stdlib.get_function("ResultIsOk").is_some(), "ResultIsOk function should be retrievable");
    assert!(stdlib.get_function("OptionIsSome").is_some(), "OptionIsSome function should be retrievable");
}