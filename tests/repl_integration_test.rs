//! Integration tests for the REPL in realistic usage scenarios

use lyra::repl::ReplEngine;

#[test]
fn test_full_repl_session() {
    // Test a complete REPL session with various features
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Basic arithmetic
    let result = repl.evaluate_line("2 + 3").expect("Failed to evaluate arithmetic");
    assert!(result.result.contains("5"));
    
    // Variable assignment
    let result = repl.evaluate_line("x = 10").expect("Failed to assign variable");
    assert!(result.result.contains("10"));
    
    // Use variable in expression
    let result = repl.evaluate_line("x * 2").expect("Failed to use variable");
    assert!(result.result.contains("20"));
    
    // Mathematical function
    let result = repl.evaluate_line("Sin[0]").expect("Failed to evaluate Sin");
    assert!(result.result.contains("0"));
    
    // List operations
    let result = repl.evaluate_line("lst = {1, 2, 3}").expect("Failed to create list");
    assert!(result.result.contains("{1, 2, 3}"));
    
    let result = repl.evaluate_line("Length[lst]").expect("Failed to get length");
    assert!(result.result.contains("3"));
    
    // Check history
    let result = repl.evaluate_line("%history").expect("Failed to show history");
    assert!(result.result.contains("Command History"));
    
    // Check variables
    let result = repl.evaluate_line("%vars").expect("Failed to show variables");
    assert!(result.result.contains("x = 10"));
    assert!(result.result.contains("lst = {1, 2, 3}"));
    
    // Performance stats
    let result = repl.evaluate_line("%perf").expect("Failed to show performance");
    assert!(result.result.contains("Performance Statistics"));
}

#[test]
fn test_repl_error_recovery() {
    // Test that REPL recovers from errors gracefully
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Valid expression
    let result = repl.evaluate_line("x = 5");
    assert!(result.is_ok());
    
    // Invalid expression
    let result = repl.evaluate_line("invalid syntax here");
    assert!(result.is_err());
    
    // Should still work after error
    let result = repl.evaluate_line("y = x + 1");
    assert!(result.is_ok());
    assert!(result.unwrap().result.contains("6"));
}

#[test]
fn test_repl_variable_persistence() {
    // Test that variables persist across evaluations
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Set multiple variables
    repl.evaluate_line("a = 1").expect("Failed to set a");
    repl.evaluate_line("b = 2").expect("Failed to set b");
    repl.evaluate_line("c = 3").expect("Failed to set c");
    
    // Use all variables
    let result = repl.evaluate_line("a + b + c").expect("Failed to use variables");
    assert!(result.result.contains("6"));
    
    // Check that they're all stored
    let vars_result = repl.evaluate_line("%vars").expect("Failed to show vars");
    assert!(vars_result.result.contains("a = 1"));
    assert!(vars_result.result.contains("b = 2"));
    assert!(vars_result.result.contains("c = 3"));
}

#[test]
fn test_repl_meta_commands() {
    // Test all meta commands work correctly
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Execute some expressions to populate history and stats
    repl.evaluate_line("x = 42").expect("Failed to set variable");
    repl.evaluate_line("y = x * 2").expect("Failed to calculate");
    
    // Test meta commands
    let help_result = repl.evaluate_line("%help").expect("Failed to show help");
    assert!(help_result.result.contains("Meta Commands"));
    
    let history_result = repl.evaluate_line("%history").expect("Failed to show history");
    assert!(history_result.result.contains("x = 42"));
    
    let vars_result = repl.evaluate_line("%vars").expect("Failed to show variables");
    assert!(vars_result.result.contains("x = 42"));
    assert!(vars_result.result.contains("y = 84"));
    
    let perf_result = repl.evaluate_line("%perf").expect("Failed to show performance");
    assert!(perf_result.result.contains("Performance Statistics"));
    
    // Test timing toggle
    let timing_off = repl.evaluate_line("%timing off").expect("Failed to turn timing off");
    assert!(timing_off.result.contains("Timing display disabled"));
    
    let timing_on = repl.evaluate_line("%timing on").expect("Failed to turn timing on");
    assert!(timing_on.result.contains("Timing display enabled"));
    
    // Test performance toggle
    let perf_off = repl.evaluate_line("%perf off").expect("Failed to turn perf off");
    assert!(perf_off.result.contains("Performance display disabled"));
    
    let perf_on = repl.evaluate_line("%perf on").expect("Failed to turn perf on");
    assert!(perf_on.result.contains("Performance display enabled"));
    
    // Test clear
    let clear_result = repl.evaluate_line("%clear").expect("Failed to clear session");
    assert!(clear_result.result.contains("Session cleared"));
    
    // Variables should be gone
    let vars_after_clear = repl.evaluate_line("%vars").expect("Failed to show vars after clear");
    assert!(vars_after_clear.result.contains("No variables defined"));
}

#[test]
fn test_repl_complex_expressions() {
    // Test more complex symbolic computation
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Complex arithmetic expression
    let result = repl.evaluate_line("2^3 + 4*5 - 6/2").expect("Failed complex arithmetic");
    assert!(result.result.contains("25")); // 8 + 20 - 3 = 25
    
    // Nested function calls
    let result = repl.evaluate_line("Sin[Cos[0]]").expect("Failed nested functions");
    // Should work without error
    assert!(!result.result.is_empty());
    
    // List with variables
    repl.evaluate_line("n = 5").expect("Failed to set n");
    let result = repl.evaluate_line("range = {1, 2, n}").expect("Failed list with variables");
    assert!(result.result.contains("{1, 2, 5}"));
}

#[test]
fn test_repl_performance_tracking() {
    // Test that performance tracking works
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Execute expressions that should trigger performance tracking
    repl.evaluate_line("Sin[Pi/2]").expect("Failed Sin evaluation");
    repl.evaluate_line("Complex[1, 2]").expect("Failed Complex evaluation");
    
    // Check performance stats
    let stats = repl.get_performance_stats();
    assert!(stats.evaluation_time.as_nanos() > 0);
    
    let perf_result = repl.evaluate_line("%perf").expect("Failed to get performance");
    assert!(perf_result.result.contains("Performance Statistics"));
    assert!(perf_result.result.contains("Total Evaluation Time"));
}

#[test]
fn test_repl_assignment_types() {
    // Test different types of assignments
    let mut repl = ReplEngine::new().expect("Failed to create REPL");
    
    // Integer assignment
    let result = repl.evaluate_line("int_var = 42").expect("Failed int assignment");
    assert!(result.result.contains("42"));
    
    // String assignment  
    let result = repl.evaluate_line("str_var = \"hello\"").expect("Failed string assignment");
    assert!(result.result.contains("\"hello\""));
    
    // List assignment
    let result = repl.evaluate_line("list_var = {1, 2, 3}").expect("Failed list assignment");
    assert!(result.result.contains("{1, 2, 3}"));
    
    // Expression assignment
    repl.evaluate_line("x = 5").expect("Failed to set x");
    let result = repl.evaluate_line("expr_var = x + 10").expect("Failed expr assignment");
    assert!(result.result.contains("15"));
    
    // Check all variables exist
    let vars_result = repl.evaluate_line("%vars").expect("Failed to show variables");
    assert!(vars_result.result.contains("int_var = 42"));
    assert!(vars_result.result.contains("str_var = \"hello\""));
    assert!(vars_result.result.contains("list_var = {1, 2, 3}"));
    assert!(vars_result.result.contains("expr_var = 15"));
}