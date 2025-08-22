//! Basic integration test for Enhanced REPL Help System
//!
//! Focuses on testing the basic functionality without complex dependencies

#[cfg(test)]
mod basic_help_tests {
    use lyra::repl::{ReplEngine, ReplResult};
    use lyra::repl::config::ReplConfig;

    fn create_test_repl() -> ReplResult<ReplEngine> {
        ReplEngine::new_with_config(ReplConfig::default())
    }

    #[test]
    fn test_help_command_basic() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test basic help command
        let result = repl.evaluate_line("?");
        assert!(result.is_ok());
        
        let help_result = result.unwrap();
        assert!(help_result.result.contains("Usage"));
    }

    #[test] 
    fn test_search_command_basic() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test basic search command
        let result = repl.evaluate_line("??");
        assert!(result.is_ok());
        
        let search_result = result.unwrap();
        assert!(search_result.result.contains("Usage"));
    }

    #[test]
    fn test_function_help_attempt() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test function help - should either work or gracefully handle missing function
        let result = repl.evaluate_line("?Sin");
        assert!(result.is_ok());
        
        let help_result = result.unwrap();
        // Should contain either help for Sin or suggestions/error message
        assert!(!help_result.result.is_empty());
    }

    #[test]
    fn test_fuzzy_search_attempt() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test fuzzy search - should work or gracefully handle
        let result = repl.evaluate_line("??sin");
        assert!(result.is_ok());
        
        let search_result = result.unwrap();
        // Should contain either search results or no results message
        assert!(!search_result.result.is_empty());
    }

    #[test]
    fn test_category_browse_attempt() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test category browsing - should work or show available categories
        let result = repl.evaluate_line("??math");
        assert!(result.is_ok());
        
        let category_result = result.unwrap();
        // Should contain either category functions or available categories
        assert!(!category_result.result.is_empty());
    }

    #[test]
    fn test_regular_evaluation_still_works() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test that regular evaluation is not broken by help system
        let result = repl.evaluate_line("1 + 2");
        assert!(result.is_ok());
        
        // After help command
        let _help_result = repl.evaluate_line("?Sin");
        
        // Regular evaluation should still work
        let result2 = repl.evaluate_line("3 + 4");
        assert!(result2.is_ok());
    }

    #[test]
    fn test_completion_basic() {
        let repl = create_test_repl().expect("Failed to create REPL");
        
        // Test that completion system works
        let completions = repl.get_completions("?S", 2);
        
        // Should return some completions (even if empty list)
        // The key is that it doesn't crash
        assert!(completions.len() <= 10); // Should be limited as per our implementation
    }

    #[test]
    fn test_help_system_robustness() {
        let mut repl = create_test_repl().expect("Failed to create REPL");
        
        // Test various edge cases
        let edge_cases = vec![
            "?",
            "??", 
            "?NonExistentFunction",
            "??nonexistent",
            "?123invalid",
            "??$%^invalid",
        ];
        
        for case in edge_cases {
            let result = repl.evaluate_line(case);
            // Should either succeed or fail gracefully
            match result {
                Ok(eval_result) => {
                    // If successful, should have some content
                    assert!(!eval_result.result.is_empty());
                },
                Err(_) => {
                    // Errors are acceptable for invalid input
                }
            }
        }
    }
}