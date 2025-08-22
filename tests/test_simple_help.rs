//! Simple test to verify enhanced help system is working

use lyra::repl::{ReplEngine, ReplResult};
use lyra::repl::config::ReplConfig;

#[test]
fn test_enhanced_help_integration() {
    let mut repl = ReplEngine::new_with_config(ReplConfig::default()).expect("Failed to create REPL");
    
    // Test that enhanced help commands don't crash the system
    
    // Test help command usage
    let result = repl.evaluate_line("?").unwrap();
    println!("Help usage result: {}", result.result);
    assert!(result.result.contains("Usage"));
    
    // Test search command usage
    let result = repl.evaluate_line("??").unwrap();
    println!("Search usage result: {}", result.result);
    assert!(result.result.contains("Usage"));
    
    // Test function help (should work or show suggestions)
    let result = repl.evaluate_line("?Sin").unwrap();
    println!("Function help result: {}", result.result);
    assert!(!result.result.is_empty());
    
    // Test search functionality
    let result = repl.evaluate_line("??math").unwrap();
    println!("Math search result: {}", result.result);
    assert!(!result.result.is_empty());
    
    // Test that regular computation still works
    let result = repl.evaluate_line("2 + 3").unwrap();
    println!("Arithmetic result: {}", result.result);
    
    println!("✅ All enhanced help system tests passed!");
}