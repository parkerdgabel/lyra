//! Tests for Developer Tools & Debugging System
//!
//! This test suite validates all developer experience tools including debugging,
//! performance measurement, error handling, testing framework, logging, and introspection.

use lyra::stdlib::StandardLibrary;
use lyra::vm::{Value, VmError};

#[test]
fn test_developer_tools_functions_registered() {
    let stdlib = StandardLibrary::new();
    
    // Test debugging tools
    assert!(stdlib.get_function("Inspect").is_some());
    assert!(stdlib.get_function("Debug").is_some());
    assert!(stdlib.get_function("Trace").is_some());
    assert!(stdlib.get_function("DebugBreak").is_some());
    assert!(stdlib.get_function("StackTrace").is_some());
    
    // Test performance tools
    assert!(stdlib.get_function("Timing").is_some());
    assert!(stdlib.get_function("MemoryUsage").is_some());
    assert!(stdlib.get_function("ProfileFunction").is_some());
    assert!(stdlib.get_function("Benchmark").is_some());
    assert!(stdlib.get_function("BenchmarkCompare").is_some());
    
    // Test error handling
    assert!(stdlib.get_function("Try").is_some());
    assert!(stdlib.get_function("Assert").is_some());
    assert!(stdlib.get_function("Validate").is_some());
    assert!(stdlib.get_function("ErrorMessage").is_some());
    assert!(stdlib.get_function("ThrowError").is_some());
    
    // Test testing framework
    assert!(stdlib.get_function("Test").is_some());
    assert!(stdlib.get_function("TestSuite").is_some());
    assert!(stdlib.get_function("MockData").is_some());
    assert!(stdlib.get_function("BenchmarkSuite").is_some());
    assert!(stdlib.get_function("TestReport").is_some());
    
    // Test logging system
    assert!(stdlib.get_function("Log").is_some());
    assert!(stdlib.get_function("LogLevel").is_some());
    assert!(stdlib.get_function("LogToFile").is_some());
    assert!(stdlib.get_function("LogFilter").is_some());
    assert!(stdlib.get_function("LogHistory").is_some());
    
    // Test introspection
    assert!(stdlib.get_function("FunctionInfo").is_some());
    assert!(stdlib.get_function("FunctionList").is_some());
    assert!(stdlib.get_function("Help").is_some());
    assert!(stdlib.get_function("TypeOf").is_some());
    assert!(stdlib.get_function("SizeOf").is_some());
    assert!(stdlib.get_function("Dependencies").is_some());
}

// ============================================================================
// DEBUGGING TOOLS TESTS
// ============================================================================

#[test]
fn test_inspect_function() {
    let stdlib = StandardLibrary::new();
    let inspect_fn = stdlib.get_function("Inspect").unwrap();
    
    // Test inspecting an integer
    let result = inspect_fn(&[Value::Integer(42)]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "InspectionResult");
            
            // Test getting type info
            let type_info = obj.call_method("getType", &[]).unwrap();
            match type_info {
                Value::String(s) => assert_eq!(s, "Integer"),
                _ => panic!("Expected string type info"),
            }
        }
        _ => panic!("Expected InspectionResult object"),
    }
}

#[test]
fn test_inspect_list() {
    let stdlib = StandardLibrary::new();
    let inspect_fn = stdlib.get_function("Inspect").unwrap();
    
    let list = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::String("test".to_string()),
    ]);
    
    let result = inspect_fn(&[list]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "InspectionResult");
            
            let metadata = obj.call_method("getMetadata", &[]).unwrap();
            match metadata {
                Value::List(items) => {
                    assert!(!items.is_empty());
                    // Should contain length and depth metadata
                }
                _ => panic!("Expected metadata list"),
            }
        }
        _ => panic!("Expected InspectionResult object"),
    }
}

#[test]
fn test_debug_session() {
    let stdlib = StandardLibrary::new();
    let debug_fn = stdlib.get_function("Debug").unwrap();
    
    let result = debug_fn(&[Value::String("test_expression".to_string())]).unwrap();
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 2);
            // First item should be the original expression
            match &items[0] {
                Value::String(s) => assert_eq!(s, "test_expression"),
                _ => panic!("Expected original expression"),
            }
            // Second item should be the debug session
            match &items[1] {
                Value::LyObj(obj) => assert_eq!(obj.type_name(), "DebugSession"),
                _ => panic!("Expected DebugSession object"),
            }
        }
        _ => panic!("Expected list with expression and debug session"),
    }
}

#[test]
fn test_trace_execution() {
    let stdlib = StandardLibrary::new();
    let trace_fn = stdlib.get_function("Trace").unwrap();
    
    let result = trace_fn(&[Value::Integer(42)]).unwrap();
    match result {
        Value::Integer(n) => assert_eq!(n, 42),
        _ => panic!("Expected traced expression to be returned"),
    }
}

#[test]
fn test_debug_break() {
    let stdlib = StandardLibrary::new();
    let debug_break_fn = stdlib.get_function("DebugBreak").unwrap();
    
    let result = debug_break_fn(&[Value::String("x > 10".to_string())]).unwrap();
    match result {
        Value::Boolean(b) => assert!(b),
        _ => panic!("Expected boolean true for successful breakpoint"),
    }
}

#[test]
fn test_stack_trace() {
    let stdlib = StandardLibrary::new();
    let stack_trace_fn = stdlib.get_function("StackTrace").unwrap();
    
    let result = stack_trace_fn(&[]).unwrap();
    match result {
        Value::List(stack) => {
            assert!(!stack.is_empty());
            // Should contain at least some stack frames
            for frame in stack {
                match frame {
                    Value::String(_) => {}, // Valid stack frame
                    _ => panic!("Expected string stack frame"),
                }
            }
        }
        _ => panic!("Expected list of stack frames"),
    }
}

// ============================================================================
// PERFORMANCE TOOLS TESTS
// ============================================================================

#[test]
fn test_timing_function() {
    let stdlib = StandardLibrary::new();
    let timing_fn = stdlib.get_function("Timing").unwrap();
    
    let result = timing_fn(&[Value::Integer(42)]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "TimingResult");
            
            // Test getting duration
            let duration = obj.call_method("getDuration", &[]).unwrap();
            match duration {
                Value::Real(d) => assert!(d >= 0.0),
                _ => panic!("Expected real duration"),
            }
            
            // Test getting result
            let result = obj.call_method("getResult", &[]).unwrap();
            match result {
                Value::Integer(n) => assert_eq!(n, 42),
                _ => panic!("Expected original result"),
            }
        }
        _ => panic!("Expected TimingResult object"),
    }
}

#[test]
fn test_memory_usage() {
    let stdlib = StandardLibrary::new();
    let memory_fn = stdlib.get_function("MemoryUsage").unwrap();
    
    let result = memory_fn(&[Value::String("test".to_string())]).unwrap();
    match result {
        Value::Integer(size) => assert!(size > 0),
        _ => panic!("Expected integer memory size"),
    }
}

#[test]
fn test_profile_function() {
    let stdlib = StandardLibrary::new();
    let profile_fn = stdlib.get_function("ProfileFunction").unwrap();
    
    let result = profile_fn(&[
        Value::Function("TestFunction".to_string()),
        Value::Integer(10),
        Value::Integer(20),
    ]).unwrap();
    
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "TimingResult");
        }
        _ => panic!("Expected TimingResult object"),
    }
}

#[test]
fn test_benchmark() {
    let stdlib = StandardLibrary::new();
    let benchmark_fn = stdlib.get_function("Benchmark").unwrap();
    
    let result = benchmark_fn(&[
        Value::String("test_expression".to_string()),
        Value::Integer(10),
    ]).unwrap();
    
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "BenchmarkResult");
            
            // Test getting iterations
            let iterations = obj.call_method("getIterations", &[]).unwrap();
            match iterations {
                Value::Integer(n) => assert_eq!(n, 10),
                _ => panic!("Expected 10 iterations"),
            }
            
            // Test getting average time
            let avg_time = obj.call_method("getAvgTime", &[]).unwrap();
            match avg_time {
                Value::Real(t) => assert!(t > 0.0),
                _ => panic!("Expected positive average time"),
            }
        }
        _ => panic!("Expected BenchmarkResult object"),
    }
}

#[test]
fn test_benchmark_compare() {
    let stdlib = StandardLibrary::new();
    let benchmark_compare_fn = stdlib.get_function("BenchmarkCompare").unwrap();
    
    let result = benchmark_compare_fn(&[
        Value::String("expr1".to_string()),
        Value::String("expr2".to_string()),
    ]).unwrap();
    
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            // Should contain two times and a ratio
            match (&items[0], &items[1], &items[2]) {
                (Value::Real(t1), Value::Real(t2), Value::Real(ratio)) => {
                    assert!(*t1 > 0.0);
                    assert!(*t2 > 0.0);
                    assert!(*ratio > 0.0);
                }
                _ => panic!("Expected real time values and ratio"),
            }
        }
        _ => panic!("Expected list with timing comparison"),
    }
}

// ============================================================================
// ERROR HANDLING TESTS
// ============================================================================

#[test]
fn test_try_catch() {
    let stdlib = StandardLibrary::new();
    let try_fn = stdlib.get_function("Try").unwrap();
    
    let result = try_fn(&[
        Value::String("risky_operation".to_string()),
        Value::String("fallback_value".to_string()),
    ]).unwrap();
    
    // Should return one of the expressions (success case mocked)
    match result {
        Value::String(_) => {}, // Valid result
        _ => panic!("Expected string result from try-catch"),
    }
}

#[test]
fn test_assert_success() {
    let stdlib = StandardLibrary::new();
    let assert_fn = stdlib.get_function("Assert").unwrap();
    
    let result = assert_fn(&[Value::Boolean(true)]).unwrap();
    match result {
        Value::Boolean(true) => {}, // Assertion passed
        _ => panic!("Expected true for successful assertion"),
    }
}

#[test]
fn test_assert_failure() {
    let stdlib = StandardLibrary::new();
    let assert_fn = stdlib.get_function("Assert").unwrap();
    
    let result = assert_fn(&[Value::Boolean(false)]);
    assert!(result.is_err());
}

#[test]
fn test_assert_with_message() {
    let stdlib = StandardLibrary::new();
    let assert_fn = stdlib.get_function("Assert").unwrap();
    
    let result = assert_fn(&[
        Value::Boolean(false),
        Value::String("Custom error message".to_string()),
    ]);
    
    assert!(result.is_err());
    if let Err(VmError::Runtime(msg)) = result {
        assert!(msg.contains("Custom error message"));
    } else {
        panic!("Expected runtime error with custom message");
    }
}

#[test]
fn test_validate_integer() {
    let stdlib = StandardLibrary::new();
    let validate_fn = stdlib.get_function("Validate").unwrap();
    
    // Test positive validation
    let result = validate_fn(&[
        Value::Integer(5),
        Value::String("positive".to_string()),
    ]).unwrap();
    
    match result {
        Value::Boolean(true) => {}, // Valid positive number
        _ => panic!("Expected true for positive validation"),
    }
    
    // Test negative validation
    let result = validate_fn(&[
        Value::Integer(-5),
        Value::String("positive".to_string()),
    ]).unwrap();
    
    match result {
        Value::Boolean(false) => {}, // Invalid for positive rule
        _ => panic!("Expected false for negative number with positive rule"),
    }
}

#[test]
fn test_validate_string() {
    let stdlib = StandardLibrary::new();
    let validate_fn = stdlib.get_function("Validate").unwrap();
    
    // Test email validation
    let result = validate_fn(&[
        Value::String("test@example.com".to_string()),
        Value::String("email".to_string()),
    ]).unwrap();
    
    match result {
        Value::Boolean(true) => {}, // Valid email
        _ => panic!("Expected true for email validation"),
    }
}

#[test]
fn test_error_message() {
    let stdlib = StandardLibrary::new();
    let error_message_fn = stdlib.get_function("ErrorMessage").unwrap();
    
    let result = error_message_fn(&[Value::String("Test error".to_string())]).unwrap();
    match result {
        Value::String(msg) => assert_eq!(msg, "Test error"),
        _ => panic!("Expected error message string"),
    }
}

#[test]
fn test_throw_error() {
    let stdlib = StandardLibrary::new();
    let throw_error_fn = stdlib.get_function("ThrowError").unwrap();
    
    let result = throw_error_fn(&[Value::String("Custom error".to_string())]);
    assert!(result.is_err());
    if let Err(VmError::Runtime(msg)) = result {
        assert!(msg.contains("Custom error"));
    } else {
        panic!("Expected runtime error");
    }
}

// ============================================================================
// TESTING FRAMEWORK TESTS
// ============================================================================

#[test]
fn test_test_function_success() {
    let stdlib = StandardLibrary::new();
    let test_fn = stdlib.get_function("Test").unwrap();
    
    let result = test_fn(&[Value::Integer(42), Value::Integer(42)]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "TestResult");
            
            let passed = obj.call_method("passed", &[]).unwrap();
            match passed {
                Value::Boolean(true) => {}, // Test passed
                _ => panic!("Expected test to pass"),
            }
        }
        _ => panic!("Expected TestResult object"),
    }
}

#[test]
fn test_test_function_failure() {
    let stdlib = StandardLibrary::new();
    let test_fn = stdlib.get_function("Test").unwrap();
    
    let result = test_fn(&[Value::Integer(42), Value::Integer(24)]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "TestResult");
            
            let passed = obj.call_method("passed", &[]).unwrap();
            match passed {
                Value::Boolean(false) => {}, // Test failed
                _ => panic!("Expected test to fail"),
            }
        }
        _ => panic!("Expected TestResult object"),
    }
}

#[test]
fn test_test_suite() {
    let stdlib = StandardLibrary::new();
    let test_suite_fn = stdlib.get_function("TestSuite").unwrap();
    
    let tests = Value::List(vec![
        Value::String("test1".to_string()),
        Value::String("test2".to_string()),
        Value::String("test3".to_string()),
    ]);
    
    let result = test_suite_fn(&[tests]).unwrap();
    match result {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "TestSuite");
            
            let test_count = obj.call_method("getTestCount", &[]).unwrap();
            match test_count {
                Value::Integer(n) => assert_eq!(n, 3),
                _ => panic!("Expected 3 tests"),
            }
            
            let success_rate = obj.call_method("getSuccessRate", &[]).unwrap();
            match success_rate {
                Value::Real(rate) => assert!(rate >= 0.0 && rate <= 1.0),
                _ => panic!("Expected success rate between 0 and 1"),
            }
        }
        _ => panic!("Expected TestSuite object"),
    }
}

#[test]
fn test_mock_data() {
    let stdlib = StandardLibrary::new();
    let mock_data_fn = stdlib.get_function("MockData").unwrap();
    
    let result = mock_data_fn(&[
        Value::String("Integer".to_string()),
        Value::Integer(5),
    ]).unwrap();
    
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 5);
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::Integer(n) => assert_eq!(*n, i as i64),
                    _ => panic!("Expected integer mock data"),
                }
            }
        }
        _ => panic!("Expected list of mock data"),
    }
}

#[test]
fn test_mock_data_strings() {
    let stdlib = StandardLibrary::new();
    let mock_data_fn = stdlib.get_function("MockData").unwrap();
    
    let result = mock_data_fn(&[
        Value::String("String".to_string()),
        Value::Integer(3),
    ]).unwrap();
    
    match result {
        Value::List(items) => {
            assert_eq!(items.len(), 3);
            for (i, item) in items.iter().enumerate() {
                match item {
                    Value::String(s) => assert_eq!(s, &format!("item_{}", i)),
                    _ => panic!("Expected string mock data"),
                }
            }
        }
        _ => panic!("Expected list of mock data"),
    }
}

// ============================================================================
// LOGGING SYSTEM TESTS
// ============================================================================

#[test]
fn test_log_function() {
    let stdlib = StandardLibrary::new();
    let log_fn = stdlib.get_function("Log").unwrap();
    
    let result = log_fn(&[
        Value::String("info".to_string()),
        Value::String("Test message".to_string()),
    ]).unwrap();
    
    match result {
        Value::Boolean(true) => {}, // Logging succeeded
        _ => panic!("Expected true for successful logging"),
    }
}

#[test]
fn test_log_level() {
    let stdlib = StandardLibrary::new();
    let log_level_fn = stdlib.get_function("LogLevel").unwrap();
    
    let result = log_level_fn(&[Value::String("debug".to_string())]).unwrap();
    match result {
        Value::String(level) => assert_eq!(level, "debug"),
        _ => panic!("Expected log level string"),
    }
}

// ============================================================================
// INTROSPECTION TESTS
// ============================================================================

#[test]
fn test_type_of() {
    let stdlib = StandardLibrary::new();
    let type_of_fn = stdlib.get_function("TypeOf").unwrap();
    
    // Test integer type
    let result = type_of_fn(&[Value::Integer(42)]).unwrap();
    match result {
        Value::String(type_name) => assert_eq!(type_name, "Integer"),
        _ => panic!("Expected type name string"),
    }
    
    // Test string type
    let result = type_of_fn(&[Value::String("test".to_string())]).unwrap();
    match result {
        Value::String(type_name) => assert_eq!(type_name, "String"),
        _ => panic!("Expected type name string"),
    }
    
    // Test list type
    let result = type_of_fn(&[Value::List(vec![Value::Integer(1), Value::Integer(2)])]).unwrap();
    match result {
        Value::String(type_name) => assert!(type_name.starts_with("List[")),
        _ => panic!("Expected list type name"),
    }
}

#[test]
fn test_size_of() {
    let stdlib = StandardLibrary::new();
    let size_of_fn = stdlib.get_function("SizeOf").unwrap();
    
    let result = size_of_fn(&[Value::Integer(42)]).unwrap();
    match result {
        Value::Integer(size) => assert_eq!(size, 8), // Size of i64
        _ => panic!("Expected integer size"),
    }
    
    let result = size_of_fn(&[Value::String("test".to_string())]).unwrap();
    match result {
        Value::Integer(size) => assert_eq!(size, 4), // Length of string
        _ => panic!("Expected string size"),
    }
}

#[test]
fn test_function_info() {
    let stdlib = StandardLibrary::new();
    let function_info_fn = stdlib.get_function("FunctionInfo").unwrap();
    
    let result = function_info_fn(&[Value::String("Length".to_string())]).unwrap();
    match result {
        Value::String(info) => assert!(info.contains("Returns the number of elements")),
        _ => panic!("Expected function info string"),
    }
    
    let result = function_info_fn(&[Value::String("UnknownFunction".to_string())]).unwrap();
    match result {
        Value::String(info) => assert!(info.contains("Function not found")),
        _ => panic!("Expected not found message"),
    }
}

#[test]
fn test_function_list() {
    let stdlib = StandardLibrary::new();
    let function_list_fn = stdlib.get_function("FunctionList").unwrap();
    
    // Test wildcard pattern
    let result = function_list_fn(&[Value::String("*".to_string())]).unwrap();
    match result {
        Value::List(functions) => assert!(!functions.is_empty()),
        _ => panic!("Expected list of functions"),
    }
    
    // Test prefix pattern
    let result = function_list_fn(&[Value::String("String*".to_string())]).unwrap();
    match result {
        Value::List(functions) => {
            for func in functions {
                match func {
                    Value::String(name) => assert!(name.starts_with("String")),
                    _ => panic!("Expected string function name"),
                }
            }
        }
        _ => panic!("Expected list of matching functions"),
    }
}

#[test]
fn test_help_general() {
    let stdlib = StandardLibrary::new();
    let help_fn = stdlib.get_function("Help").unwrap();
    
    let result = help_fn(&[]).unwrap();
    match result {
        Value::String(help_text) => assert!(help_text.contains("General help displayed")),
        _ => panic!("Expected help text"),
    }
}

#[test]
fn test_help_topic() {
    let stdlib = StandardLibrary::new();
    let help_fn = stdlib.get_function("Help").unwrap();
    
    let result = help_fn(&[Value::String("Debugging".to_string())]).unwrap();
    match result {
        Value::String(help_text) => assert!(help_text.contains("Help for Debugging displayed")),
        _ => panic!("Expected debugging help text"),
    }
    
    let result = help_fn(&[Value::String("Performance".to_string())]).unwrap();
    match result {
        Value::String(help_text) => assert!(help_text.contains("Help for Performance displayed")),
        _ => panic!("Expected performance help text"),
    }
}

// ============================================================================
// INTEGRATION TESTS
// ============================================================================

#[test]
fn test_complete_debugging_workflow() {
    let stdlib = StandardLibrary::new();
    
    // Start with inspection
    let inspect_fn = stdlib.get_function("Inspect").unwrap();
    let data = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
    let inspection = inspect_fn(&[data.clone()]).unwrap();
    
    // Move to debugging
    let debug_fn = stdlib.get_function("Debug").unwrap();
    let debug_session = debug_fn(&[data.clone()]).unwrap();
    
    // Add timing
    let timing_fn = stdlib.get_function("Timing").unwrap();
    let timing_result = timing_fn(&[data]).unwrap();
    
    // All should succeed
    assert!(matches!(inspection, Value::LyObj(_)));
    assert!(matches!(debug_session, Value::List(_)));
    assert!(matches!(timing_result, Value::LyObj(_)));
}

#[test]
fn test_complete_testing_workflow() {
    let stdlib = StandardLibrary::new();
    
    // Generate mock data
    let mock_data_fn = stdlib.get_function("MockData").unwrap();
    let test_data = mock_data_fn(&[
        Value::String("Integer".to_string()),
        Value::Integer(3),
    ]).unwrap();
    
    // Run individual tests
    let test_fn = stdlib.get_function("Test").unwrap();
    match &test_data {
        Value::List(items) => {
            if let Value::Integer(n) = &items[0] {
                let test_result = test_fn(&[Value::Integer(*n), Value::Integer(0)]).unwrap();
                assert!(matches!(test_result, Value::LyObj(_)));
            }
        }
        _ => panic!("Expected mock data list"),
    }
    
    // Run test suite
    let test_suite_fn = stdlib.get_function("TestSuite").unwrap();
    let suite_result = test_suite_fn(&[test_data]).unwrap();
    assert!(matches!(suite_result, Value::LyObj(_)));
}

#[test]
fn test_error_handling_workflow() {
    let stdlib = StandardLibrary::new();
    
    // Test assertion failure
    let assert_fn = stdlib.get_function("Assert").unwrap();
    let result = assert_fn(&[
        Value::Boolean(false),
        Value::String("Test assertion".to_string()),
    ]);
    assert!(result.is_err());
    
    // Test custom error throwing
    let throw_error_fn = stdlib.get_function("ThrowError").unwrap();
    let result = throw_error_fn(&[Value::String("Custom error".to_string())]);
    assert!(result.is_err());
    
    // Test error message extraction
    let error_message_fn = stdlib.get_function("ErrorMessage").unwrap();
    let result = error_message_fn(&[Value::String("Error text".to_string())]).unwrap();
    match result {
        Value::String(msg) => assert_eq!(msg, "Error text"),
        _ => panic!("Expected error message"),
    }
}

#[test]
fn test_performance_measurement_workflow() {
    let stdlib = StandardLibrary::new();
    
    // Start with timing
    let timing_fn = stdlib.get_function("Timing").unwrap();
    let expr = Value::String("test_expression".to_string());
    let timing_result = timing_fn(&[expr.clone()]).unwrap();
    
    // Move to benchmarking
    let benchmark_fn = stdlib.get_function("Benchmark").unwrap();
    let benchmark_result = benchmark_fn(&[expr.clone(), Value::Integer(5)]).unwrap();
    
    // Compare performance
    let benchmark_compare_fn = stdlib.get_function("BenchmarkCompare").unwrap();
    let compare_result = benchmark_compare_fn(&[expr.clone(), expr]).unwrap();
    
    // All should succeed and return appropriate objects
    assert!(matches!(timing_result, Value::LyObj(_)));
    assert!(matches!(benchmark_result, Value::LyObj(_)));
    assert!(matches!(compare_result, Value::List(_)));
}

#[test]
fn test_introspection_workflow() {
    let stdlib = StandardLibrary::new();
    
    // Get help
    let help_fn = stdlib.get_function("Help").unwrap();
    let help_result = help_fn(&[]).unwrap();
    assert!(matches!(help_result, Value::String(_)));
    
    // List functions
    let function_list_fn = stdlib.get_function("FunctionList").unwrap();
    let functions = function_list_fn(&[Value::String("*".to_string())]).unwrap();
    assert!(matches!(functions, Value::List(_)));
    
    // Get function info
    let function_info_fn = stdlib.get_function("FunctionInfo").unwrap();
    let info = function_info_fn(&[Value::String("Length".to_string())]).unwrap();
    assert!(matches!(info, Value::String(_)));
    
    // Check types
    let type_of_fn = stdlib.get_function("TypeOf").unwrap();
    let type_info = type_of_fn(&[Value::Integer(42)]).unwrap();
    match type_info {
        Value::String(t) => assert_eq!(t, "Integer"),
        _ => panic!("Expected type string"),
    }
}

// ============================================================================
// ADDITIONAL FUNCTION TESTS
// ============================================================================

#[test]
fn test_log_to_file() {
    let stdlib = StandardLibrary::new();
    let log_to_file_fn = stdlib.get_function("LogToFile").unwrap();
    
    let result = log_to_file_fn(&[Value::String("app.log".to_string())]).unwrap();
    match result {
        Value::String(filename) => assert_eq!(filename, "app.log"),
        _ => panic!("Expected filename string"),
    }
}

#[test]
fn test_log_filter() {
    let stdlib = StandardLibrary::new();
    let log_filter_fn = stdlib.get_function("LogFilter").unwrap();
    
    let result = log_filter_fn(&[Value::String("ERROR".to_string())]).unwrap();
    match result {
        Value::String(pattern) => assert_eq!(pattern, "ERROR"),
        _ => panic!("Expected filter pattern string"),
    }
}

#[test]
fn test_log_history() {
    let stdlib = StandardLibrary::new();
    let log_history_fn = stdlib.get_function("LogHistory").unwrap();
    
    let result = log_history_fn(&[]).unwrap();
    match result {
        Value::List(entries) => {
            assert!(!entries.is_empty());
            // Check that entries have proper structure
            for entry in entries {
                match entry {
                    Value::List(fields) => {
                        assert_eq!(fields.len(), 3); // level, message, timestamp
                    }
                    _ => panic!("Expected log entry as list"),
                }
            }
        }
        _ => panic!("Expected list of log entries"),
    }
}

#[test]
fn test_dependencies() {
    let stdlib = StandardLibrary::new();
    let dependencies_fn = stdlib.get_function("Dependencies").unwrap();
    
    let result = dependencies_fn(&[Value::String("Map".to_string())]).unwrap();
    match result {
        Value::List(deps) => {
            assert!(!deps.is_empty());
            // Should contain expected dependencies for Map
            for dep in deps {
                match dep {
                    Value::String(_) => {}, // Valid dependency name
                    _ => panic!("Expected string dependency name"),
                }
            }
        }
        _ => panic!("Expected list of dependencies"),
    }
}

#[test]
fn test_test_report() {
    let stdlib = StandardLibrary::new();
    let test_suite_fn = stdlib.get_function("TestSuite").unwrap();
    let test_report_fn = stdlib.get_function("TestReport").unwrap();
    
    // First create a test suite
    let tests = Value::List(vec![
        Value::String("test1".to_string()),
        Value::String("test2".to_string()),
    ]);
    
    let suite = test_suite_fn(&[tests]).unwrap();
    
    // Then generate a report
    let report = test_report_fn(&[suite]).unwrap();
    match report {
        Value::String(msg) => assert!(msg.contains("Test report generated")),
        _ => panic!("Expected report generation message"),
    }
}

#[test]
fn test_benchmark_suite() {
    let stdlib = StandardLibrary::new();
    let benchmark_suite_fn = stdlib.get_function("BenchmarkSuite").unwrap();
    
    let benchmarks = Value::List(vec![
        Value::String("expr1".to_string()),
        Value::String("expr2".to_string()),
        Value::String("expr3".to_string()),
    ]);
    
    let result = benchmark_suite_fn(&[benchmarks]).unwrap();
    match result {
        Value::List(results) => {
            assert_eq!(results.len(), 3);
            for result in results {
                match result {
                    Value::LyObj(obj) => {
                        assert_eq!(obj.type_name(), "BenchmarkResult");
                    }
                    _ => panic!("Expected BenchmarkResult objects"),
                }
            }
        }
        _ => panic!("Expected list of benchmark results"),
    }
}

#[test]
fn test_complete_developer_workflow() {
    let stdlib = StandardLibrary::new();
    
    // 1. Start with help
    let help_fn = stdlib.get_function("Help").unwrap();
    let help_result = help_fn(&[]).unwrap();
    assert!(matches!(help_result, Value::String(_)));
    
    // 2. List available functions
    let function_list_fn = stdlib.get_function("FunctionList").unwrap();
    let functions = function_list_fn(&[Value::String("*".to_string())]).unwrap();
    assert!(matches!(functions, Value::List(_)));
    
    // 3. Inspect some data
    let inspect_fn = stdlib.get_function("Inspect").unwrap();
    let data = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
    let inspection = inspect_fn(&[data.clone()]).unwrap();
    assert!(matches!(inspection, Value::LyObj(_)));
    
    // 4. Time an operation
    let timing_fn = stdlib.get_function("Timing").unwrap();
    let timing_result = timing_fn(&[data.clone()]).unwrap();
    assert!(matches!(timing_result, Value::LyObj(_)));
    
    // 5. Test equality
    let test_fn = stdlib.get_function("Test").unwrap();
    let test_result = test_fn(&[Value::Integer(42), Value::Integer(42)]).unwrap();
    assert!(matches!(test_result, Value::LyObj(_)));
    
    // 6. Log the results
    let log_fn = stdlib.get_function("Log").unwrap();
    let log_result = log_fn(&[
        Value::String("info".to_string()),
        Value::String("Workflow completed successfully".to_string()),
    ]).unwrap();
    assert!(matches!(log_result, Value::Boolean(true)));
}

#[test]
fn test_error_handling_integration() {
    let stdlib = StandardLibrary::new();
    
    // Test assertion failure
    let assert_fn = stdlib.get_function("Assert").unwrap();
    let result = assert_fn(&[
        Value::Boolean(false),
        Value::String("Integration test assertion".to_string()),
    ]);
    assert!(result.is_err());
    
    // Test validation
    let validate_fn = stdlib.get_function("Validate").unwrap();
    let validation_result = validate_fn(&[
        Value::Integer(10),
        Value::String("positive".to_string()),
    ]).unwrap();
    assert!(matches!(validation_result, Value::Boolean(true)));
    
    // Test negative validation  
    let validation_result = validate_fn(&[
        Value::Integer(-5),
        Value::String("positive".to_string()),
    ]).unwrap();
    assert!(matches!(validation_result, Value::Boolean(false)));
}

#[test]
fn test_function_count_verification() {
    let stdlib = StandardLibrary::new();
    
    // Count developer tools functions
    let developer_tools_functions = [
        // Debugging Tools (5)
        "Inspect", "Debug", "Trace", "DebugBreak", "StackTrace",
        // Performance Tools (5)
        "Timing", "MemoryUsage", "ProfileFunction", "Benchmark", "BenchmarkCompare",
        // Error Handling (5)
        "Try", "Assert", "Validate", "ErrorMessage", "ThrowError",
        // Testing Framework (5)
        "Test", "TestSuite", "MockData", "BenchmarkSuite", "TestReport",
        // Logging System (5)
        "Log", "LogLevel", "LogToFile", "LogFilter", "LogHistory",
        // Introspection & Reflection (6)
        "FunctionInfo", "FunctionList", "Help", "TypeOf", "SizeOf", "Dependencies",
    ];
    
    // Verify all 31 functions are registered
    for function_name in &developer_tools_functions {
        assert!(
            stdlib.get_function(function_name).is_some(),
            "Function {} should be registered",
            function_name
        );
    }
    
    // Total should be 31 functions (exceeds the 25+ requirement)
    assert_eq!(developer_tools_functions.len(), 31);
}