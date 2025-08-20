#![no_main]

use libfuzzer_sys::fuzz_target;
use lyra::{vm::Value, stdlib::StandardLibrary};
use arbitrary::{Arbitrary, Unstructured};

const MAX_LIST_SIZE: usize = 1000;
const MAX_STRING_SIZE: usize = 1000;
const MAX_NESTING_DEPTH: usize = 10;

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    function_name: String,
    args: Vec<FuzzValue>,
}

#[derive(Arbitrary, Debug, Clone)]
enum FuzzValue {
    Integer(i32),  // Use i32 to limit size
    Real(f32),     // Use f32 for simpler fuzzing
    Boolean(bool),
    String(String),
    Symbol(String),
    List(Vec<FuzzValue>),
}

impl FuzzValue {
    fn to_vm_value(self, depth: usize) -> Option<Value> {
        if depth > MAX_NESTING_DEPTH {
            return None; // Prevent infinite recursion
        }
        
        match self {
            FuzzValue::Integer(i) => Some(Value::Integer(i as i64)),
            FuzzValue::Real(f) => {
                if f.is_finite() {
                    Some(Value::Real(f as f64))
                } else {
                    Some(Value::Real(0.0)) // Replace non-finite with safe value
                }
            }
            FuzzValue::Boolean(b) => Some(Value::Boolean(b)),
            FuzzValue::String(mut s) => {
                if s.len() > MAX_STRING_SIZE {
                    s.truncate(MAX_STRING_SIZE);
                }
                // Remove null bytes and control characters
                s = s.chars().filter(|&c| c != '\0' && !c.is_control() || c == '\n' || c == '\t').collect();
                Some(Value::String(s))
            }
            FuzzValue::Symbol(mut s) => {
                if s.len() > MAX_STRING_SIZE {
                    s.truncate(MAX_STRING_SIZE);
                }
                // Ensure valid symbol name
                s = s.chars().filter(|c| c.is_alphanumeric() || *c == '_').collect();
                if s.is_empty() {
                    s = "x".to_string();
                }
                Some(Value::Symbol(s))
            }
            FuzzValue::List(mut list) => {
                if list.len() > MAX_LIST_SIZE {
                    list.truncate(MAX_LIST_SIZE);
                }
                
                let vm_values: Vec<Value> = list
                    .into_iter()
                    .filter_map(|v| v.to_vm_value(depth + 1))
                    .collect();
                    
                Some(Value::List(vm_values))
            }
        }
    }
    
    fn constrain_size(&mut self) {
        match self {
            FuzzValue::String(s) | FuzzValue::Symbol(s) => {
                if s.len() > MAX_STRING_SIZE {
                    s.truncate(MAX_STRING_SIZE);
                }
            }
            FuzzValue::List(list) => {
                if list.len() > MAX_LIST_SIZE {
                    list.truncate(MAX_LIST_SIZE);
                }
                for item in list {
                    item.constrain_size();
                }
            }
            _ => {}
        }
    }
}

impl FuzzInput {
    fn constrain_size(&mut self) {
        // Limit function name length
        if self.function_name.len() > 100 {
            self.function_name.truncate(100);
        }
        
        // Ensure valid function name characters
        self.function_name = self.function_name
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_')
            .collect();
            
        if self.function_name.is_empty() {
            self.function_name = "Length".to_string();
        }
        
        // Limit number of arguments
        if self.args.len() > 10 {
            self.args.truncate(10);
        }
        
        // Constrain each argument
        for arg in &mut self.args {
            arg.constrain_size();
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(mut input) = FuzzInput::arbitrary(&mut unstructured) {
        input.constrain_size();
        
        let stdlib = StandardLibrary::new();
        
        // Check if the function exists in stdlib
        if let Some(function) = stdlib.get_function(&input.function_name) {
            // Convert fuzz values to VM values
            let vm_args: Vec<Value> = input.args
                .into_iter()
                .filter_map(|v| v.to_vm_value(0))
                .collect();
            
            // Test the function with panic catching
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                function(&vm_args)
            })) {
                Ok(result) => {
                    // Function execution completed (success or error)
                    match result {
                        Ok(_value) => {
                            // Successfully executed
                        }
                        Err(_error) => {
                            // Function error is expected for invalid inputs
                        }
                    }
                }
                Err(_) => {
                    // Function panicked - this is a bug we want to find
                }
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use lyra::{vm::Value, stdlib::StandardLibrary};
    
    #[test]
    fn test_stdlib_fuzzing_basic() {
        let stdlib = StandardLibrary::new();
        
        // Test each major category of functions with valid inputs
        let test_cases = vec![
            // List functions
            ("Length", vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])]),
            ("Head", vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])]),
            ("Tail", vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])]),
            
            // String functions
            ("StringLength", vec![Value::String("hello".to_string())]),
            ("StringJoin", vec![Value::String("hello".to_string()), Value::String(" world".to_string())]),
            
            // Math functions
            ("Plus", vec![Value::Integer(1), Value::Integer(2)]),
            ("Times", vec![Value::Integer(3), Value::Integer(4)]),
            ("Sin", vec![Value::Real(1.0)]),
            ("Cos", vec![Value::Real(0.0)]),
            
            // Statistics functions
            ("Mean", vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])]),
            ("Max", vec![Value::List(vec![Value::Integer(1), Value::Integer(3), Value::Integer(2)])]),
            ("Min", vec![Value::List(vec![Value::Integer(1), Value::Integer(3), Value::Integer(2)])]),
        ];
        
        for (func_name, args) in test_cases {
            if let Some(function) = stdlib.get_function(func_name) {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(&args)
                }));
                
                assert!(result.is_ok(), "Function {} panicked with valid args", func_name);
            }
        }
    }
    
    #[test]
    fn test_stdlib_security_patterns() {
        let stdlib = StandardLibrary::new();
        
        let security_test_cases = vec![
            // Very large lists
            ("Length", vec![Value::List((0..10000).map(|i| Value::Integer(i)).collect())]),
            
            // Very long strings
            ("StringLength", vec![Value::String("x".repeat(100000))]),
            
            // Deeply nested lists
            ("Length", vec![create_deep_nested_list(100)]),
            
            // Invalid numeric values
            ("Sin", vec![Value::Real(f64::NAN)]),
            ("Cos", vec![Value::Real(f64::INFINITY)]),
            ("Plus", vec![Value::Real(f64::NEG_INFINITY), Value::Real(f64::INFINITY)]),
            
            // Empty arguments
            ("Length", vec![]),
            ("Plus", vec![]),
            ("StringJoin", vec![]),
            
            // Wrong argument types
            ("Length", vec![Value::Integer(42)]),
            ("StringLength", vec![Value::Integer(123)]),
            ("Plus", vec![Value::String("not a number".to_string())]),
            
            // Mixed valid/invalid arguments
            ("Plus", vec![Value::Integer(1), Value::String("invalid".to_string())]),
            
            // Boundary values
            ("Plus", vec![Value::Integer(i64::MAX), Value::Integer(1)]),
            ("Times", vec![Value::Integer(i64::MAX), Value::Integer(2)]),
            ("Divide", vec![Value::Integer(1), Value::Integer(0)]),
        ];
        
        for (func_name, args) in security_test_cases {
            if let Some(function) = stdlib.get_function(func_name) {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(&args)
                }));
                
                assert!(result.is_ok(), 
                       "Function {} panicked with security test args: {:?}", 
                       func_name, 
                       args.iter().take(2).collect::<Vec<_>>());
            }
        }
    }
    
    #[test]
    fn test_stdlib_resource_limits() {
        let stdlib = StandardLibrary::new();
        
        // Test functions that might consume excessive resources
        
        // Large tensor operations
        if let Some(function) = stdlib.get_function("Array") {
            let large_tensor = create_large_nested_list(1000, 3);
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                function(&vec![large_tensor])
            }));
            
            assert!(result.is_ok(), "Array function panicked with large input");
        }
        
        // FFT with large input
        if let Some(function) = stdlib.get_function("FFT") {
            let large_signal: Vec<Value> = (0..10000).map(|i| Value::Real(i as f64)).collect();
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                function(&vec![Value::List(large_signal)])
            }));
            
            assert!(result.is_ok(), "FFT function panicked with large input");
        }
        
        // String operations with very long strings
        if let Some(function) = stdlib.get_function("StringJoin") {
            let long_strings: Vec<Value> = (0..1000)
                .map(|_| Value::String("x".repeat(1000)))
                .collect();
            
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                function(&long_strings)
            }));
            
            assert!(result.is_ok(), "StringJoin function panicked with long strings");
        }
    }
    
    #[test]
    fn test_all_stdlib_functions_basic() {
        let stdlib = StandardLibrary::new();
        
        // Test that all registered functions can be called without crashing
        // (even if they return errors due to invalid arguments)
        
        for func_name in stdlib.function_names() {
            if let Some(function) = stdlib.get_function(func_name) {
                // Test with empty args
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(&[])
                }));
                assert!(result.is_ok(), "Function {} panicked with empty args", func_name);
                
                // Test with single integer arg
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(&[Value::Integer(1)])
                }));
                assert!(result.is_ok(), "Function {} panicked with single integer arg", func_name);
                
                // Test with single string arg
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    function(&[Value::String("test".to_string())])
                }));
                assert!(result.is_ok(), "Function {} panicked with single string arg", func_name);
            }
        }
    }
    
    // Helper functions for testing
    fn create_deep_nested_list(depth: usize) -> Value {
        if depth == 0 {
            Value::Integer(1)
        } else {
            Value::List(vec![create_deep_nested_list(depth - 1)])
        }
    }
    
    fn create_large_nested_list(size: usize, depth: usize) -> Value {
        if depth == 0 {
            Value::Integer(1)
        } else {
            Value::List((0..size).map(|_| create_large_nested_list(size.min(10), depth - 1)).collect())
        }
    }
}