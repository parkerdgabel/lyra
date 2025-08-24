#![allow(unused_variables)]
//! Enhanced error context hints with intelligent recovery suggestions
//!
//! This module analyzes errors and provides context-aware hints to help users
//! understand and fix common mistakes.

use std::collections::HashMap;
use crate::lexer::{Lexer, TokenKind};

/// Enhanced error context analyzer
pub struct ErrorContextAnalyzer {
    /// Common error patterns and their solutions
    error_patterns: HashMap<String, ErrorPattern>,
    /// Function name suggestions for typos
    function_suggestions: HashMap<String, Vec<String>>,
    /// Common syntax error fixes
    syntax_fixes: HashMap<String, Vec<SyntaxFix>>,
}

/// Error pattern with contextual solutions
#[derive(Debug, Clone)]
pub struct ErrorPattern {
    /// Pattern identifier
    pub pattern_id: String,
    /// Human-readable error description
    pub description: String,
    /// Possible causes
    pub causes: Vec<String>,
    /// Suggested solutions
    pub solutions: Vec<Solution>,
    /// Example of correct usage
    pub correct_examples: Vec<String>,
    /// Severity level (1=info, 5=critical)
    pub severity: u8,
}

/// Solution for an error pattern
#[derive(Debug, Clone)]
pub struct Solution {
    /// Brief solution description
    pub description: String,
    /// Detailed explanation
    pub explanation: String,
    /// Code example showing the fix
    pub fix_example: String,
    /// Automatic fix (if possible)
    pub auto_fix: Option<String>,
}

/// Syntax fix suggestion
#[derive(Debug, Clone)]
pub struct SyntaxFix {
    /// Description of the fix
    pub description: String,
    /// Original problematic text
    pub original: String,
    /// Suggested replacement
    pub replacement: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
}

/// Error context hint result
#[derive(Debug, Clone)]
pub struct ErrorContextHint {
    /// Type of error detected
    pub error_type: ErrorType,
    /// Primary error message
    pub primary_message: String,
    /// Additional context information
    pub context_info: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<ErrorSuggestion>,
    /// "Did you mean?" alternatives
    pub did_you_mean: Vec<String>,
    /// Related documentation links
    pub see_also: Vec<String>,
}

/// Type of error detected
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorType {
    /// Function name typo or unknown function
    UnknownFunction,
    /// Incorrect number of arguments
    ArgumentCount,
    /// Type mismatch in arguments
    ArgumentType,
    /// Syntax error (brackets, quotes, etc.)
    SyntaxError,
    /// Pattern matching error
    PatternError,
    /// Mathematical domain error
    DomainError,
    /// General runtime error
    RuntimeError,
}

/// Specific error suggestion
#[derive(Debug, Clone)]
pub struct ErrorSuggestion {
    /// Brief description
    pub description: String,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Suggested code change
    pub suggested_code: Option<String>,
    /// Explanation of why this might help
    pub rationale: String,
}

impl ErrorContextAnalyzer {
    /// Create a new error context analyzer
    pub fn new() -> Self {
        let mut analyzer = Self {
            error_patterns: HashMap::new(),
            function_suggestions: HashMap::new(),
            syntax_fixes: HashMap::new(),
        };
        
        analyzer.populate_error_patterns();
        analyzer.build_function_suggestions();
        analyzer.populate_syntax_fixes();
        
        analyzer
    }
    
    /// Analyze an error and provide contextual hints
    pub fn analyze_error(&self, error_message: &str, input: &str, cursor_pos: Option<usize>) -> ErrorContextHint {
        // Determine error type
        let error_type = self.classify_error(error_message, input);
        
        // Generate suggestions based on error type
        let suggestions = self.generate_suggestions(&error_type, error_message, input);
        
        // Find "did you mean" alternatives
        let did_you_mean = self.find_did_you_mean_suggestions(&error_type, input);
        
        // Extract context information
        let context_info = self.extract_context_info(&error_type, input, cursor_pos);
        
        // Find related documentation
        let see_also = self.find_related_documentation(&error_type, input);
        
        ErrorContextHint {
            error_type,
            primary_message: self.generate_friendly_message(error_message),
            context_info,
            suggestions,
            did_you_mean,
            see_also,
        }
    }
    
    /// Classify the type of error
    fn classify_error(&self, error_message: &str, input: &str) -> ErrorType {
        let error_lower = error_message.to_lowercase();
        
        if error_lower.contains("unknown function") || error_lower.contains("undefined symbol") {
            ErrorType::UnknownFunction
        } else if error_lower.contains("wrong number of arguments") || error_lower.contains("arity") {
            ErrorType::ArgumentCount
        } else if error_lower.contains("type error") || error_lower.contains("type mismatch") {
            ErrorType::ArgumentType
        } else if error_lower.contains("syntax error") || self.has_syntax_errors(input) {
            ErrorType::SyntaxError
        } else if error_lower.contains("pattern") || error_lower.contains("match") {
            ErrorType::PatternError
        } else if error_lower.contains("domain") || error_lower.contains("division by zero") {
            ErrorType::DomainError
        } else {
            ErrorType::RuntimeError
        }
    }
    
    /// Check for syntax errors in input
    fn has_syntax_errors(&self, input: &str) -> bool {
        // Check bracket balance
        let mut bracket_count = 0;
        let mut paren_count = 0;
        let mut brace_count = 0;
        let mut in_string = false;
        let mut escape_next = false;
        
        for ch in input.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            
            match ch {
                '\\' if in_string => escape_next = true,
                '"' => in_string = !in_string,
                '[' if !in_string => bracket_count += 1,
                ']' if !in_string => bracket_count -= 1,
                '(' if !in_string => paren_count += 1,
                ')' if !in_string => paren_count -= 1,
                '{' if !in_string => brace_count += 1,
                '}' if !in_string => brace_count -= 1,
                _ => {}
            }
        }
        
        bracket_count != 0 || paren_count != 0 || brace_count != 0 || in_string
    }
    
    /// Generate suggestions based on error type
    fn generate_suggestions(&self, error_type: &ErrorType, error_message: &str, input: &str) -> Vec<ErrorSuggestion> {
        match error_type {
            ErrorType::UnknownFunction => self.suggest_function_alternatives(input),
            ErrorType::ArgumentCount => self.suggest_argument_fixes(input),
            ErrorType::ArgumentType => self.suggest_type_fixes(input),
            ErrorType::SyntaxError => self.suggest_syntax_fixes(input),
            ErrorType::PatternError => self.suggest_pattern_fixes(input),
            ErrorType::DomainError => self.suggest_domain_fixes(error_message, input),
            ErrorType::RuntimeError => self.suggest_runtime_fixes(error_message, input),
        }
    }
    
    /// Suggest function name alternatives
    fn suggest_function_alternatives(&self, input: &str) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();
        
        // Try to extract function name from input
        if let Some(func_name) = self.extract_function_name(input) {
            if let Some(alternatives) = self.function_suggestions.get(&func_name.to_lowercase()) {
                for alt in alternatives {
                    suggestions.push(ErrorSuggestion {
                        description: format!("Did you mean '{}'?", alt),
                        confidence: self.calculate_similarity(&func_name, alt),
                        suggested_code: Some(input.replace(&func_name, alt)),
                        rationale: format!("'{}' is similar to '{}' and is a valid function", alt, func_name),
                    });
                }
            }
        }
        
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(3); // Top 3 suggestions
        
        suggestions
    }
    
    /// Suggest argument count fixes
    fn suggest_argument_fixes(&self, input: &str) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();
        
        if let Some(func_name) = self.extract_function_name(input) {
            // Look up correct function signature
            if let Some(correct_usage) = self.get_function_usage(&func_name) {
                suggestions.push(ErrorSuggestion {
                    description: format!("Check {} usage", func_name),
                    confidence: 0.9,
                    suggested_code: Some(correct_usage.clone()),
                    rationale: format!("{} expects a specific number of arguments", func_name),
                });
            }
        }
        
        suggestions
    }
    
    /// Suggest type fixes
    fn suggest_type_fixes(&self, input: &str) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();
        
        // Common type conversion suggestions
        if input.contains("\"") && input.contains("+") {
            suggestions.push(ErrorSuggestion {
                description: "Convert string to number".to_string(),
                confidence: 0.8,
                suggested_code: Some("Use ToNumber[string] to convert".to_string()),
                rationale: "String operations require consistent types".to_string(),
            });
        }
        
        suggestions
    }
    
    /// Suggest syntax fixes
    fn suggest_syntax_fixes(&self, input: &str) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();
        
        // Check for common bracket issues
        let bracket_count = input.chars().filter(|&c| c == '[').count() - 
                           input.chars().filter(|&c| c == ']').count();
        
        if bracket_count > 0 {
            suggestions.push(ErrorSuggestion {
                description: format!("Add {} closing bracket(s)", bracket_count),
                confidence: 0.95,
                suggested_code: Some(format!("{}{}", input, "]".repeat(bracket_count))),
                rationale: "Unmatched opening brackets need to be closed".to_string(),
            });
        } else if bracket_count < 0 {
            let extra_brackets = (bracket_count as i32).abs() as usize;
            suggestions.push(ErrorSuggestion {
                description: format!("Remove {} extra closing bracket(s)", extra_brackets),
                confidence: 0.9,
                suggested_code: None,
                rationale: "Extra closing brackets should be removed".to_string(),
            });
        }
        
        // Check for string quote issues
        let quote_count = input.chars().filter(|&c| c == '"').count();
        if quote_count % 2 != 0 {
            suggestions.push(ErrorSuggestion {
                description: "Add missing closing quote".to_string(),
                confidence: 0.9,
                suggested_code: Some(format!("{}\"", input)),
                rationale: "Strings must be properly quoted".to_string(),
            });
        }
        
        suggestions
    }
    
    /// Suggest pattern fixes
    fn suggest_pattern_fixes(&self, _input: &str) -> Vec<ErrorSuggestion> {
        vec![
            ErrorSuggestion {
                description: "Check pattern syntax".to_string(),
                confidence: 0.7,
                suggested_code: Some("Use x_ for pattern variables".to_string()),
                rationale: "Pattern variables need underscore suffix".to_string(),
            }
        ]
    }
    
    /// Suggest domain fixes
    fn suggest_domain_fixes(&self, error_message: &str, input: &str) -> Vec<ErrorSuggestion> {
        let mut suggestions = Vec::new();
        
        if error_message.contains("division by zero") {
            suggestions.push(ErrorSuggestion {
                description: "Check for zero divisor".to_string(),
                confidence: 0.95,
                suggested_code: Some("Add condition: If[denominator != 0, division, \"undefined\"]".to_string()),
                rationale: "Division by zero is undefined in mathematics".to_string(),
            });
        }
        
        if input.contains("Log[") && error_message.contains("domain") {
            suggestions.push(ErrorSuggestion {
                description: "Ensure positive argument for Log".to_string(),
                confidence: 0.9,
                suggested_code: Some("Log[Abs[x]] or check that x > 0".to_string()),
                rationale: "Logarithm requires positive real arguments".to_string(),
            });
        }
        
        suggestions
    }
    
    /// Suggest runtime fixes
    fn suggest_runtime_fixes(&self, _error_message: &str, _input: &str) -> Vec<ErrorSuggestion> {
        vec![
            ErrorSuggestion {
                description: "Check input values".to_string(),
                confidence: 0.6,
                suggested_code: None,
                rationale: "Runtime errors often result from unexpected input values".to_string(),
            }
        ]
    }
    
    /// Find "did you mean" suggestions
    fn find_did_you_mean_suggestions(&self, error_type: &ErrorType, input: &str) -> Vec<String> {
        if *error_type == ErrorType::UnknownFunction {
            if let Some(func_name) = self.extract_function_name(input) {
                return self.find_similar_function_names(&func_name);
            }
        }
        Vec::new()
    }
    
    /// Extract context information
    fn extract_context_info(&self, error_type: &ErrorType, _input: &str, cursor_pos: Option<usize>) -> Vec<String> {
        let mut info = Vec::new();
        
        match error_type {
            ErrorType::SyntaxError => {
                info.push("Check brackets, parentheses, and quotes are balanced".to_string());
                if let Some(pos) = cursor_pos {
                    info.push(format!("Error near position {}", pos));
                }
            },
            ErrorType::UnknownFunction => {
                info.push("Make sure function name is spelled correctly".to_string());
                info.push("Use 'functions' command to see available functions".to_string());
            },
            ErrorType::ArgumentCount => {
                info.push("Check function documentation for correct usage".to_string());
            },
            _ => {}
        }
        
        info
    }
    
    /// Find related documentation
    fn find_related_documentation(&self, error_type: &ErrorType, input: &str) -> Vec<String> {
        let mut docs = Vec::new();
        
        if let Some(func_name) = self.extract_function_name(input) {
            docs.push(format!("See documentation for {}", func_name));
        }
        
        match error_type {
            ErrorType::SyntaxError => docs.push("See syntax reference".to_string()),
            ErrorType::PatternError => docs.push("See pattern matching guide".to_string()),
            _ => {}
        }
        
        docs
    }
    
    /// Generate friendly error message
    fn generate_friendly_message(&self, error_message: &str) -> String {
        // Make error messages more user-friendly
        if error_message.contains("parse error") {
            "There's a syntax error in your input".to_string()
        } else if error_message.contains("unknown") {
            "I don't recognize that function name".to_string()
        } else if error_message.contains("type") {
            "The arguments don't have the expected types".to_string()
        } else {
            error_message.to_string()
        }
    }
    
    /// Extract function name from input
    fn extract_function_name(&self, input: &str) -> Option<String> {
        // Simple extraction - look for pattern: Name[
        let mut lexer = Lexer::new(input);
        if let Ok(tokens) = lexer.tokenize() {
            for (i, token) in tokens.iter().enumerate() {
                if let TokenKind::Symbol(name) = &token.kind {
                    // Check if followed by opening bracket
                    if i + 1 < tokens.len() {
                        if let TokenKind::LeftBracket = tokens[i + 1].kind {
                            return Some(name.clone());
                        }
                    }
                }
            }
        }
        None
    }
    
    /// Calculate string similarity
    fn calculate_similarity(&self, a: &str, b: &str) -> f32 {
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        
        // Simple Levenshtein-like distance
        let max_len = a_lower.len().max(b_lower.len());
        if max_len == 0 {
            return 1.0;
        }
        
        let distance = self.levenshtein_distance(&a_lower, &b_lower);
        1.0 - (distance as f32 / max_len as f32)
    }
    
    /// Calculate Levenshtein distance
    fn levenshtein_distance(&self, a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let a_len = a_chars.len();
        let b_len = b_chars.len();
        
        if a_len == 0 { return b_len; }
        if b_len == 0 { return a_len; }
        
        let mut matrix = vec![vec![0; b_len + 1]; a_len + 1];
        
        for i in 0..=a_len {
            matrix[i][0] = i;
        }
        for j in 0..=b_len {
            matrix[0][j] = j;
        }
        
        for i in 1..=a_len {
            for j in 1..=b_len {
                let cost = if a_chars[i-1] == b_chars[j-1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i-1][j] + 1)
                    .min(matrix[i][j-1] + 1)
                    .min(matrix[i-1][j-1] + cost);
            }
        }
        
        matrix[a_len][b_len]
    }
    
    /// Find similar function names
    fn find_similar_function_names(&self, func_name: &str) -> Vec<String> {
        let common_functions = vec![
            "Sin", "Cos", "Tan", "Log", "Exp", "Sqrt", "Abs",
            "Length", "Head", "Tail", "Append", "Map", "Apply",
            "StringJoin", "StringLength", "StringSplit",
            "Array", "Dot", "Transpose", "Maximum", "Minimum",
        ];
        
        let mut suggestions = Vec::new();
        for func in common_functions {
            let similarity = self.calculate_similarity(func_name, func);
            if similarity > 0.4 { // Similarity threshold
                suggestions.push((func.to_string(), similarity));
            }
        }
        
        suggestions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.into_iter().map(|(name, _)| name).take(3).collect()
    }
    
    /// Get function usage example
    fn get_function_usage(&self, func_name: &str) -> Option<String> {
        match func_name {
            "Sin" => Some("Sin[x] - where x is a number in radians".to_string()),
            "Length" => Some("Length[list] - where list is a list".to_string()),
            "Dot" => Some("Dot[a, b] - where a and b are compatible tensors".to_string()),
            _ => None,
        }
    }
    
    /// Populate error patterns
    fn populate_error_patterns(&mut self) {
        // Common error patterns would be populated here
        // This is a simplified version showing the structure
    }
    
    /// Build function suggestions
    fn build_function_suggestions(&mut self) {
        // Build mapping of common typos to correct function names
        self.function_suggestions.insert("sin".to_string(), vec!["Sin".to_string()]);
        self.function_suggestions.insert("cos".to_string(), vec!["Cos".to_string()]);
        self.function_suggestions.insert("length".to_string(), vec!["Length".to_string()]);
        self.function_suggestions.insert("len".to_string(), vec!["Length".to_string()]);
        self.function_suggestions.insert("size".to_string(), vec!["Length".to_string()]);
        self.function_suggestions.insert("count".to_string(), vec!["Length".to_string()]);
    }
    
    /// Populate syntax fixes
    fn populate_syntax_fixes(&mut self) {
        // Common syntax fixes would be populated here
    }
}

impl Default for ErrorContextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_classification() {
        let analyzer = ErrorContextAnalyzer::new();
        
        let error_type = analyzer.classify_error("unknown function", "Sine[0]");
        assert_eq!(error_type, ErrorType::UnknownFunction);
        
        let error_type = analyzer.classify_error("wrong number of arguments", "Sin[1, 2]");
        assert_eq!(error_type, ErrorType::ArgumentCount);
    }
    
    #[test]
    fn test_function_name_extraction() {
        let analyzer = ErrorContextAnalyzer::new();
        
        let func_name = analyzer.extract_function_name("Sin[0]");
        assert_eq!(func_name, Some("Sin".to_string()));
        
        let func_name = analyzer.extract_function_name("Length[{1, 2, 3}]");
        assert_eq!(func_name, Some("Length".to_string()));
    }
    
    #[test]
    fn test_syntax_error_detection() {
        let analyzer = ErrorContextAnalyzer::new();
        
        assert!(analyzer.has_syntax_errors("Sin[0"));  // Missing closing bracket
        assert!(analyzer.has_syntax_errors("\"hello"));  // Missing closing quote
        assert!(!analyzer.has_syntax_errors("Sin[0]"));  // Valid syntax
    }
    
    #[test]
    fn test_similarity_calculation() {
        let analyzer = ErrorContextAnalyzer::new();
        
        let similarity = analyzer.calculate_similarity("Sin", "sin");
        assert!(similarity > 0.8);
        
        let similarity = analyzer.calculate_similarity("Length", "len");
        assert!(similarity > 0.3);
    }
    
    #[test]
    fn test_did_you_mean_suggestions() {
        let analyzer = ErrorContextAnalyzer::new();
        
        let suggestions = analyzer.find_similar_function_names("sine");
        assert!(suggestions.contains(&"Sin".to_string()));
        
        let suggestions = analyzer.find_similar_function_names("lenght");
        assert!(suggestions.contains(&"Length".to_string()));
    }
}
