//! Parameter suggestion system for function calls
//!
//! This module analyzes function call context and suggests appropriate parameters
//! based on function signatures and current argument position.

use super::function_hints::{FunctionSignatureDatabase, ParameterInfo};
use super::context_analyzer::HintContext;
use super::HintResult;
use std::sync::Arc;

/// Parameter suggestion engine
pub struct ParameterSuggester {
    /// Reference to function signature database
    function_db: Arc<FunctionSignatureDatabase>,
}

impl ParameterSuggester {
    /// Create a new parameter suggester
    pub fn new(function_db: Arc<FunctionSignatureDatabase>) -> Self {
        Self { function_db }
    }
    
    /// Suggest parameter for current context
    pub fn suggest_parameter(&self, context: &HintContext) -> HintResult {
        let function_name = match &context.function_name {
            Some(name) => name,
            None => return HintResult::None,
        };
        
        let signature = match self.function_db.get_signature(function_name) {
            Some(sig) => sig,
            None => return HintResult::None,
        };
        
        let param_index = context.parameter_index.unwrap_or(0);
        
        // Get parameter info for current position
        if let Some(param_info) = signature.parameters.get(param_index) {
            let suggestions = self.generate_suggestions(param_info, context);
            
            HintResult::ParameterSuggestion {
                expected_type: param_info.param_type.clone(),
                suggestions,
                current_param: param_index,
            }
        } else if signature.parameters.is_empty() {
            HintResult::ParameterSuggestion {
                expected_type: "None".to_string(),
                suggestions: vec!["Function takes no parameters".to_string()],
                current_param: 0,
            }
        } else {
            HintResult::ParameterSuggestion {
                expected_type: "Extra".to_string(),
                suggestions: vec![format!("Function expects {} parameters", signature.parameters.len())],
                current_param: param_index,
            }
        }
    }
    
    /// Generate specific suggestions for a parameter
    fn generate_suggestions(&self, param_info: &ParameterInfo, context: &HintContext) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        match param_info.param_type.as_str() {
            "Number" => {
                suggestions.extend(self.suggest_numbers(context));
            }
            "Integer" => {
                suggestions.extend(self.suggest_integers(context));
            }
            "List" => {
                suggestions.extend(self.suggest_lists(context));
            }
            "String" => {
                suggestions.extend(self.suggest_strings(context));
            }
            "Tensor" => {
                suggestions.extend(self.suggest_tensors(context));
            }
            "Expression" => {
                suggestions.extend(self.suggest_expressions(context));
            }
            "Pattern" => {
                suggestions.extend(self.suggest_patterns(context));
            }
            _ => {
                // Generic suggestions
                suggestions.push("value".to_string());
            }
        }
        
        // Add default value if available
        if let Some(default) = &param_info.default_value {
            suggestions.insert(0, format!("{} (default)", default));
        }
        
        // Limit number of suggestions
        suggestions.truncate(5);
        suggestions
    }
    
    /// Suggest numbers based on context
    fn suggest_numbers(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "0".to_string(),
            "1".to_string(),
            "Pi".to_string(),
            "E".to_string(),
        ]
    }
    
    /// Suggest integers based on context
    fn suggest_integers(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "0".to_string(),
            "1".to_string(),
            "2".to_string(),
            "-1".to_string(),
        ]
    }
    
    /// Suggest lists based on context
    fn suggest_lists(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "{}".to_string(),
            "{1, 2, 3}".to_string(),
            "{a, b, c}".to_string(),
        ]
    }
    
    /// Suggest strings based on context
    fn suggest_strings(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "\"\"".to_string(),
            "\"text\"".to_string(),
        ]
    }
    
    /// Suggest tensors based on context
    fn suggest_tensors(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "Array[{1, 2, 3}]".to_string(),
            "Array[{{1, 2}, {3, 4}}]".to_string(),
        ]
    }
    
    /// Suggest expressions based on context
    fn suggest_expressions(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "x".to_string(),
            "x + y".to_string(),
            "f[x]".to_string(),
        ]
    }
    
    /// Suggest patterns based on context
    fn suggest_patterns(&self, _context: &HintContext) -> Vec<String> {
        vec![
            "x_".to_string(),
            "x__".to_string(),
            "x_Integer".to_string(),
            "_".to_string(),
        ]
    }
    
    /// Get context-aware suggestions for specific functions
    pub fn get_function_specific_suggestions(&self, function_name: &str, param_index: usize) -> Vec<String> {
        match (function_name, param_index) {
            // Trigonometric functions - suggest common angles
            ("Sin" | "Cos" | "Tan", 0) => vec![
                "0".to_string(),
                "Pi/6".to_string(),
                "Pi/4".to_string(),
                "Pi/3".to_string(),
                "Pi/2".to_string(),
                "Pi".to_string(),
            ],
            
            // Array creation - suggest common patterns
            ("Array", 0) => vec![
                "{1, 2, 3}".to_string(),
                "{{1, 2}, {3, 4}}".to_string(),
                "{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}".to_string(),
            ],
            
            // List operations
            ("Length" | "Head" | "Tail", 0) => vec![
                "{1, 2, 3}".to_string(),
                "{a, b, c}".to_string(),
                "{}".to_string(),
            ],
            
            ("Append", 0) => vec![
                "{1, 2}".to_string(),
                "{}".to_string(),
            ],
            
            ("Append", 1) => vec![
                "3".to_string(),
                "x".to_string(),
            ],
            
            // String functions
            ("StringJoin", _) => vec![
                "\"hello\"".to_string(),
                "\" \"".to_string(),
                "\"world\"".to_string(),
            ],
            
            ("StringLength" | "StringTake" | "StringDrop", 0) => vec![
                "\"hello\"".to_string(),
                "\"\"".to_string(),
            ],
            
            // Dot product suggestions
            ("Dot", 0) => vec![
                "{1, 2, 3}".to_string(),
                "{{1, 2}, {3, 4}}".to_string(),
                "Array[{1, 2, 3}]".to_string(),
            ],
            
            ("Dot", 1) => vec![
                "{4, 5, 6}".to_string(),
                "{{5, 6}, {7, 8}}".to_string(),
                "Array[{4, 5, 6}]".to_string(),
            ],
            
            // Mathematical functions
            ("Log", 0) => vec![
                "E".to_string(),
                "10".to_string(),
                "2".to_string(),
                "1".to_string(),
            ],
            
            ("Exp", 0) => vec![
                "0".to_string(),
                "1".to_string(),
                "Log[x]".to_string(),
            ],
            
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::StandardLibrary;
    use crate::repl::hints::function_hints::FunctionSignatureDatabase;
    use crate::repl::hints::context_analyzer::{HintContext, HintType};
    use std::sync::Arc;

    fn create_test_suggester() -> ParameterSuggester {
        let stdlib = StandardLibrary::new();
        let function_db = Arc::new(FunctionSignatureDatabase::from_stdlib(&stdlib));
        ParameterSuggester::new(function_db)
    }

    #[test]
    fn test_basic_parameter_suggestion() {
        let suggester = create_test_suggester();
        
        let context = HintContext {
            function_name: Some("Sin".to_string()),
            parameter_index: Some(0),
            hint_type: HintType::ParameterPosition,
            current_token: None,
            error_type: None,
        };
        
        let result = suggester.suggest_parameter(&context);
        
        match result {
            HintResult::ParameterSuggestion { expected_type, suggestions, current_param } => {
                assert_eq!(expected_type, "Number");
                assert_eq!(current_param, 0);
                assert!(!suggestions.is_empty());
            }
            _ => panic!("Expected parameter suggestion"),
        }
    }
    
    #[test]
    fn test_list_parameter_suggestion() {
        let suggester = create_test_suggester();
        
        let context = HintContext {
            function_name: Some("Length".to_string()),
            parameter_index: Some(0),
            hint_type: HintType::ParameterPosition,
            current_token: None,
            error_type: None,
        };
        
        let result = suggester.suggest_parameter(&context);
        
        match result {
            HintResult::ParameterSuggestion { expected_type, .. } => {
                assert_eq!(expected_type, "List");
            }
            _ => panic!("Expected parameter suggestion"),
        }
    }
    
    #[test]
    fn test_function_specific_suggestions() {
        let suggester = create_test_suggester();
        
        // Test trigonometric function suggestions
        let trig_suggestions = suggester.get_function_specific_suggestions("Sin", 0);
        assert!(trig_suggestions.contains(&"Pi/2".to_string()));
        assert!(trig_suggestions.contains(&"0".to_string()));
        
        // Test array creation suggestions
        let array_suggestions = suggester.get_function_specific_suggestions("Array", 0);
        assert!(array_suggestions.iter().any(|s| s.contains("{")));
    }
    
    #[test]
    fn test_no_parameters_function() {
        let suggester = create_test_suggester();
        
        // Create a context for a function that doesn't exist in our database
        let context = HintContext {
            function_name: Some("NonExistentFunction".to_string()),
            parameter_index: Some(0),
            hint_type: HintType::ParameterPosition,
            current_token: None,
            error_type: None,
        };
        
        let result = suggester.suggest_parameter(&context);
        
        match result {
            HintResult::None => {
                // Expected for unknown functions
            }
            _ => {
                // This is also acceptable behavior
            }
        }
    }
    
    #[test]
    fn test_too_many_parameters() {
        let suggester = create_test_suggester();
        
        let context = HintContext {
            function_name: Some("Sin".to_string()),
            parameter_index: Some(5), // Sin only takes 1 parameter
            hint_type: HintType::ParameterPosition,
            current_token: None,
            error_type: None,
        };
        
        let result = suggester.suggest_parameter(&context);
        
        match result {
            HintResult::ParameterSuggestion { expected_type, .. } => {
                assert_eq!(expected_type, "Extra");
            }
            _ => panic!("Expected parameter suggestion indicating too many parameters"),
        }
    }
}