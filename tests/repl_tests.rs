//! REPL Integration Tests
//! 
//! Tests for the interactive Read-Eval-Print Loop functionality

use lyra::repl::{ReplEngine, ReplResult, PerformanceStats};
use lyra::vm::Value;

#[test]
fn test_repl_engine_creation() {
    // Test that we can create a new REPL engine
    let repl = ReplEngine::new();
    assert!(repl.is_ok());
}

#[test]
fn test_basic_expression_evaluation() {
    // Test basic arithmetic evaluation
    let mut repl = ReplEngine::new().unwrap();
    
    let result = repl.evaluate_line("2 + 3");
    assert!(result.is_ok());
    
    let output = result.unwrap();
    assert!(output.result.contains("5"));
}

#[test]
fn test_variable_assignment() {
    // Test variable assignment and retrieval
    let mut repl = ReplEngine::new().unwrap();
    
    // Assign variable
    let result = repl.evaluate_line("x = 5");
    if let Err(e) = &result {
        println!("Assignment error: {:?}", e);
    }
    assert!(result.is_ok());
    
    // Use variable
    let result = repl.evaluate_line("x + 3");
    if let Err(e) = &result {
        println!("Variable use error: {:?}", e);
    }
    assert!(result.is_ok());
    if let Ok(ref res) = result {
        println!("Variable use result: {}", res.result);
    }
    assert!(result.unwrap().result.contains("8"));
}

#[test]
fn test_function_definitions() {
    // Test function definition and calling
    let mut repl = ReplEngine::new().unwrap();
    
    // Define function
    let result = repl.evaluate_line("f[x_] := x^2");
    assert!(result.is_ok());
    
    // Call function
    let result = repl.evaluate_line("f[4]");
    assert!(result.is_ok());
    assert!(result.unwrap().result.contains("16"));
}

#[test]
fn test_pattern_matching() {
    // Test pattern matching capabilities
    let mut repl = ReplEngine::new().unwrap();
    
    let result = repl.evaluate_line("expr = Plus[x, Times[2, y]]");
    assert!(result.is_ok());
    
    let result = repl.evaluate_line("expr /. x -> 5");
    assert!(result.is_ok());
}

#[test]
fn test_meta_commands() {
    // Test REPL meta commands
    let mut repl = ReplEngine::new().unwrap();
    
    // Test history command
    let result = repl.evaluate_line("%history");
    assert!(result.is_ok());
    
    // Test performance command
    let result = repl.evaluate_line("%perf");
    assert!(result.is_ok());
    
    // Test clear command
    let result = repl.evaluate_line("%clear");
    assert!(result.is_ok());
}

#[test]
fn test_performance_tracking() {
    // Test that performance statistics are collected
    let mut repl = ReplEngine::new().unwrap();
    
    // Execute expressions that should trigger performance tracking
    let _ = repl.evaluate_line("Sin[Pi/2]");
    let _ = repl.evaluate_line("expr = Plus[1, 2, 3]; expr /. Plus -> Times");
    
    let stats = repl.get_performance_stats();
    assert!(stats.pattern_matching_calls > 0 || stats.rule_application_calls > 0);
}

#[test]
fn test_error_handling() {
    // Test error handling for invalid expressions
    let mut repl = ReplEngine::new().unwrap();
    
    let result = repl.evaluate_line("invalid syntax here");
    assert!(result.is_err());
    
    // Should continue working after error
    let result = repl.evaluate_line("2 + 2");
    assert!(result.is_ok());
}

#[test]
fn test_multi_line_support() {
    // Test multi-line expression support
    let mut repl = ReplEngine::new().unwrap();
    
    let result = repl.evaluate_line("expr = Plus[\n  Times[2, x],\n  Power[y, 2]\n]");
    assert!(result.is_ok());
}

#[test]
fn test_symbolic_computation_showcase() {
    // Test expressions that showcase symbolic computation
    let mut repl = ReplEngine::new().unwrap();
    
    // Mathematical functions
    let result = repl.evaluate_line("Sin[Pi/2]");
    assert!(result.is_ok());
    
    // Complex symbolic expressions  
    let result = repl.evaluate_line("Expand[(x + y)^2]");
    assert!(result.is_ok());
    
    // Rule application
    let result = repl.evaluate_line("x^2 + 2*x + 1 /. x -> 3");
    assert!(result.is_ok());
}

#[test]
fn test_list_operations() {
    // Test list operations in REPL
    let mut repl = ReplEngine::new().unwrap();
    
    let result = repl.evaluate_line("lst = {1, 2, 3, 4, 5}");
    assert!(result.is_ok());
    
    let result = repl.evaluate_line("Length[lst]");
    assert!(result.is_ok());
    assert!(result.unwrap().result.contains("5"));
    
    let result = repl.evaluate_line("Head[lst]");
    assert!(result.is_ok());
    assert!(result.unwrap().result.contains("1"));
}