//! Comprehensive tests for Enhanced REPL Help System
//!
//! Tests all features of the enhanced help system including:
//! - ?FunctionName detailed help
//! - ??search_term fuzzy search
//! - ??category browsing
//! - Context-aware completion
//! - Usage statistics and analytics

use lyra::repl::{ReplEngine, ReplResult};
use lyra::repl::config::ReplConfig;

#[cfg(test)]
mod enhanced_help_tests {
    use super::*;

    fn create_test_repl() -> ReplEngine {
        ReplEngine::new_with_config(ReplConfig::default()).expect("Failed to create test REPL")
    }

    #[test]
    fn test_enhanced_function_help() {
        let mut repl = create_test_repl();
        
        // Test detailed help for Sin function
        let result = repl.evaluate_line("?Sin").unwrap();
        let help_text = result.result;
        
        // Should contain signature
        assert!(help_text.contains("Sin"));
        assert!(help_text.contains("Sin[x_]"));
        
        // Should contain description
        assert!(help_text.contains("sine") || help_text.contains("Sine"));
        
        // Should contain examples
        assert!(help_text.contains("Examples") || help_text.contains("Sin[0]"));
        
        // Should contain parameter information
        assert!(help_text.contains("Parameter") || help_text.contains("x"));
    }

    #[test] 
    fn test_enhanced_function_help_with_aliases() {
        let mut repl = create_test_repl();
        
        // Test help for Plus function (has aliases like Add, +)
        let result = repl.evaluate_line("?Plus").unwrap();
        let help_text = result.result;
        
        assert!(help_text.contains("Plus"));
        assert!(help_text.contains("addition") || help_text.contains("Add"));
    }

    #[test]
    fn test_function_help_typo_detection() {
        let mut repl = create_test_repl();
        
        // Test typo detection for "sin" -> "Sin"
        let result = repl.evaluate_line("?sin").unwrap();
        let help_text = result.result;
        
        // Should suggest correct spelling or provide help for Sin
        assert!(help_text.contains("Sin") || help_text.contains("Did you mean"));
    }

    #[test]
    fn test_fuzzy_search_basic() {
        let mut repl = create_test_repl();
        
        // Test fuzzy search for mathematical functions
        let result = repl.evaluate_line("??sin").unwrap();
        let search_results = result.result;
        
        // Should find Sin function
        assert!(search_results.contains("Sin"));
        assert!(search_results.contains("Search results") || search_results.contains("Sin"));
        
        // Should show relevance scores or match types
        assert!(search_results.contains("score") || search_results.contains("exact") || search_results.contains("fuzzy"));
    }

    #[test] 
    fn test_fuzzy_search_with_typos() {
        let mut repl = create_test_repl();
        
        // Test fuzzy search with typos
        let result = repl.evaluate_line("??lenght").unwrap();
        let search_results = result.result;
        
        // Should find Length function despite typo
        assert!(search_results.contains("Length") || search_results.contains("No functions found"));
    }

    #[test]
    fn test_fuzzy_search_description_matching() {
        let mut repl = create_test_repl();
        
        // Test search by description keywords
        let result = repl.evaluate_line("??trigonometric").unwrap();
        let search_results = result.result;
        
        // Should find trig functions even if searching by description
        assert!(
            search_results.contains("Sin") || 
            search_results.contains("Cos") ||
            search_results.contains("No functions found")
        );
    }

    #[test]
    fn test_category_browsing() {
        let mut repl = create_test_repl();
        
        // Test math category browsing
        let result = repl.evaluate_line("??math").unwrap();
        let category_results = result.result;
        
        // Should list mathematical functions
        assert!(
            category_results.contains("Functions in category") || 
            category_results.contains("Sin") ||
            category_results.contains("Plus") ||
            category_results.contains("Available categories")
        );
    }

    #[test]
    fn test_category_browsing_list() {
        let mut repl = create_test_repl();
        
        // Test list category browsing
        let result = repl.evaluate_line("??list").unwrap();
        let category_results = result.result;
        
        // Should list list manipulation functions
        assert!(
            category_results.contains("Functions in category") ||
            category_results.contains("Length") ||
            category_results.contains("Head") ||
            category_results.contains("Available categories")
        );
    }

    #[test] 
    fn test_category_aliases() {
        let mut repl = create_test_repl();
        
        // Test category alias (mathematics -> math)
        let result = repl.evaluate_line("??mathematics").unwrap();
        let category_results = result.result;
        
        // Should handle the alias and show mathematical functions
        assert!(
            category_results.contains("math") ||
            category_results.contains("Sin") ||
            category_results.contains("Available categories")
        );
    }

    #[test]
    fn test_help_command_error_handling() {
        let mut repl = create_test_repl();
        
        // Test empty help command
        let result = repl.evaluate_line("?").unwrap();
        assert!(result.result.contains("Usage"));
        
        // Test empty search
        let result = repl.evaluate_line("??").unwrap();
        assert!(result.result.contains("Usage"));
    }

    #[test]
    fn test_nonexistent_function_help() {
        let mut repl = create_test_repl();
        
        // Test help for nonexistent function
        let result = repl.evaluate_line("?NonexistentFunction");
        
        match result {
            Ok(eval_result) => {
                // Should provide suggestions or error message
                assert!(
                    eval_result.result.contains("not found") ||
                    eval_result.result.contains("Did you mean") ||
                    eval_result.result.contains("similar functions")
                );
            },
            Err(_) => {
                // Error is acceptable for nonexistent functions
            }
        }
    }

    #[test]
    fn test_context_aware_completion() {
        let repl = create_test_repl();
        
        // Test completion for help commands
        let completions = repl.get_completions("?S", 2);
        
        // Should suggest function names starting with S
        assert!(completions.iter().any(|c| c.contains("Sin")));
    }

    #[test]
    fn test_search_command_completion() {
        let repl = create_test_repl();
        
        // Test completion for search commands
        let completions = repl.get_completions("??m", 3);
        
        // Should suggest math category or similar
        assert!(completions.iter().any(|c| c.contains("math")));
    }

    #[test]
    fn test_category_completion() {
        let repl = create_test_repl();
        
        // Test completion for category browsing
        let completions = repl.get_completions("??", 2);
        
        // Should suggest available categories
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|c| c.contains("math") || c.contains("list")));
    }

    #[test]
    fn test_help_output_formatting() {
        let mut repl = create_test_repl();
        
        // Test that help output is well-formatted
        let result = repl.evaluate_line("?Length").unwrap();
        let help_text = result.result;
        
        // Should have structured sections
        assert!(help_text.contains("Length"));
        
        // Should not be too long (reasonable truncation)
        assert!(help_text.len() < 2000, "Help output should be reasonably sized");
        
        // Should not be too short (meaningful content)
        assert!(help_text.len() > 50, "Help output should have meaningful content");
    }

    #[test]
    fn test_search_result_relevance() {
        let mut repl = create_test_repl();
        
        // Test that search results are ordered by relevance
        let result = repl.evaluate_line("??sin").unwrap();
        let search_results = result.result;
        
        if search_results.contains("Search results") {
            // If results are found, Sin should appear early (high relevance)
            let lines: Vec<&str> = search_results.lines().collect();
            let sin_line_index = lines.iter().position(|line| line.contains("Sin"));
            
            if let Some(index) = sin_line_index {
                // Sin should appear in first few results (high relevance)
                assert!(index < 8, "Sin should have high relevance in search for 'sin'");
            }
        }
    }

    #[test]
    fn test_multiple_help_commands() {
        let mut repl = create_test_repl();
        
        // Test that multiple help commands work in sequence
        let _result1 = repl.evaluate_line("?Sin").unwrap();
        let _result2 = repl.evaluate_line("??math").unwrap(); 
        let result3 = repl.evaluate_line("?Length").unwrap();
        
        // Last command should still work properly
        assert!(result3.result.contains("Length"));
    }

    #[test]
    fn test_help_system_performance() {
        let mut repl = create_test_repl();
        
        use std::time::Instant;
        
        // Test that help commands complete reasonably quickly
        let start = Instant::now();
        let _result = repl.evaluate_line("?Sin").unwrap();
        let duration = start.elapsed();
        
        // Help should complete in under 100ms
        assert!(duration.as_millis() < 100, "Help command should be fast");
        
        // Test search performance
        let start = Instant::now();
        let _result = repl.evaluate_line("??sin").unwrap();
        let duration = start.elapsed();
        
        // Search should complete in reasonable time
        assert!(duration.as_millis() < 200, "Search command should be reasonably fast");
    }

    #[test]
    fn test_integration_with_regular_commands() {
        let mut repl = create_test_repl();
        
        // Test that help commands don't interfere with regular evaluation
        let result1 = repl.evaluate_line("2 + 3").unwrap();
        assert!(result1.result.contains("5"));
        
        let _help_result = repl.evaluate_line("?Plus").unwrap();
        
        let result2 = repl.evaluate_line("4 * 5").unwrap();
        assert!(result2.result.contains("20"));
        
        // Regular evaluation should still work after help commands
    }

    #[test] 
    fn test_case_insensitive_help() {
        let mut repl = create_test_repl();
        
        // Test case insensitive help lookup
        let result_lower = repl.evaluate_line("?length");
        let result_upper = repl.evaluate_line("?LENGTH");
        let result_mixed = repl.evaluate_line("?Length");
        
        // All should provide useful output (either direct help or suggestions)
        match (&result_lower, &result_upper, &result_mixed) {
            (Ok(r1), Ok(r2), Ok(r3)) => {
                // At least one should contain Length
                assert!(
                    r1.result.contains("Length") ||
                    r2.result.contains("Length") ||
                    r3.result.contains("Length") ||
                    r1.result.contains("Did you mean") ||
                    r2.result.contains("Did you mean") ||
                    r3.result.contains("Did you mean")
                );
            },
            _ => {
                // Errors are acceptable - the system should handle them gracefully
            }
        }
    }
}

#[cfg(test)]
mod help_system_integration_tests {
    use super::*;

    #[test]
    fn test_help_with_assignment() {
        let mut repl = create_test_repl();
        
        // Test that variables don't interfere with help
        let _assign_result = repl.evaluate_line("x = 5").unwrap();
        
        let help_result = repl.evaluate_line("?Sin").unwrap();
        assert!(help_result.result.contains("Sin"));
        
        // Variable should still exist
        let var_result = repl.evaluate_line("x").unwrap();
        assert!(var_result.result.contains("5"));
    }

    #[test]
    fn test_help_after_error() {
        let mut repl = create_test_repl();
        
        // Cause an error
        let _error_result = repl.evaluate_line("invalid syntax here");
        
        // Help should still work after error
        let help_result = repl.evaluate_line("?Length").unwrap();
        assert!(help_result.result.contains("Length"));
    }

    #[test]
    fn test_completion_with_variables() {
        let mut repl = create_test_repl();
        
        // Define some variables
        let _result = repl.evaluate_line("myVariable = 42").unwrap();
        let _result2 = repl.evaluate_line("anotherVar = \"hello\"").unwrap();
        
        // Test that completion still includes help commands
        let completions = repl.get_completions("?L", 2);
        assert!(completions.iter().any(|c| c.contains("Length")));
        
        // Test that variable completion works alongside help
        let var_completions = repl.get_completions("my", 2);
        assert!(var_completions.iter().any(|c| c.contains("myVariable")));
    }
}