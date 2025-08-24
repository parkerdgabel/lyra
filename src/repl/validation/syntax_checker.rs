#![allow(unused_variables)]
//! Lightweight syntax validation for Lyra expressions
//!
//! This module provides basic syntax validation that can detect common errors
//! without requiring full parsing, suitable for real-time validation feedback.

use crate::lexer::{Lexer, Token, TokenKind};

/// Lightweight syntax checker for Lyra expressions
pub struct SyntaxChecker {
    // No state needed for stateless syntax checking
}

/// Result of syntax validation
#[derive(Debug, Clone, PartialEq)]
pub struct SyntaxResult {
    /// Whether the syntax is valid
    pub is_valid: bool,
    /// Error message if invalid
    pub error_message: String,
    /// Suggestion for fixing the error
    pub suggestion: Option<String>,
    /// Position of the error in the input
    pub error_position: Option<usize>,
}

impl SyntaxChecker {
    /// Create a new syntax checker
    pub fn new() -> Self {
        Self {}
    }
    
    /// Check syntax of the input expression
    pub fn check_syntax(&self, input: &str) -> Result<SyntaxResult, String> {
        let trimmed = input.trim();
        
        // Empty input is valid
        if trimmed.is_empty() {
            return Ok(SyntaxResult {
                is_valid: true,
                error_message: String::new(),
                suggestion: None,
                error_position: None,
            });
        }
        
        // Try to tokenize
        let mut lexer = Lexer::new(trimmed);
        let tokens = match lexer.tokenize() {
            Ok(tokens) => tokens,
            Err(err) => {
                return Ok(SyntaxResult {
                    is_valid: false,
                    error_message: format!("Tokenization error: {}", err),
                    suggestion: Some("Check for invalid characters or malformed tokens".to_string()),
                    error_position: None,
                });
            }
        };
        
        // Perform lightweight syntax checks
        self.analyze_token_sequence(&tokens, trimmed)
    }
    
    /// Analyze token sequence for common syntax errors
    fn analyze_token_sequence(&self, tokens: &[Token], input: &str) -> Result<SyntaxResult, String> {
        if tokens.is_empty() {
            return Ok(SyntaxResult {
                is_valid: true,
                error_message: String::new(),
                suggestion: None,
                error_position: None,
            });
        }
        
        // Check for consecutive operators
        if let Some(error) = self.check_consecutive_operators(tokens) {
            return Ok(error);
        }
        
        // Check for invalid operator sequences
        if let Some(error) = self.check_operator_placement(tokens) {
            return Ok(error);
        }
        
        // Check for malformed patterns
        if let Some(error) = self.check_pattern_syntax(tokens) {
            return Ok(error);
        }
        
        // Check for invalid function calls
        if let Some(error) = self.check_function_syntax(tokens) {
            return Ok(error);
        }
        
        // Check for rule syntax issues
        if let Some(error) = self.check_rule_syntax(tokens) {
            return Ok(error);
        }
        
        // If we get here, syntax looks valid
        Ok(SyntaxResult {
            is_valid: true,
            error_message: String::new(),
            suggestion: None,
            error_position: None,
        })
    }
    
    /// Check for consecutive operators that are not valid
    fn check_consecutive_operators(&self, tokens: &[Token]) -> Option<SyntaxResult> {
        for window in tokens.windows(2) {
            if let [first, second] = window {
                if self.is_binary_operator(&first.kind) && self.is_binary_operator(&second.kind) {
                    // Some consecutive operators are valid (like ++, --, etc.)
                    if !self.is_valid_operator_combination(&first.kind, &second.kind) {
                        return Some(SyntaxResult {
                            is_valid: false,
                            error_message: format!(
                                "Invalid consecutive operators: {} {}", 
                                self.token_display(&first.kind),
                                self.token_display(&second.kind)
                            ),
                            suggestion: Some("Remove one operator or add operand between them".to_string()),
                            error_position: Some(second.position),
                        });
                    }
                }
            }
        }
        None
    }
    
    /// Check for operators in wrong positions
    fn check_operator_placement(&self, tokens: &[Token]) -> Option<SyntaxResult> {
        // Check for operators at the beginning (except unary operators)
        if let Some(first) = tokens.first() {
            if self.is_binary_operator(&first.kind) && !self.is_valid_prefix_operator(&first.kind) {
                return Some(SyntaxResult {
                    is_valid: false,
                    error_message: format!(
                        "Expression cannot start with operator '{}'", 
                        self.token_display(&first.kind)
                    ),
                    suggestion: Some("Add operand before operator".to_string()),
                    error_position: Some(first.position),
                });
            }
        }
        
        // Check for operators after opening brackets
        for window in tokens.windows(2) {
            if let [first, second] = window {
                match (&first.kind, &second.kind) {
                    (TokenKind::LeftBracket, op) | (TokenKind::LeftParen, op) | (TokenKind::LeftBrace, op) 
                        if self.is_binary_operator(op) && !self.is_valid_prefix_operator(op) => {
                        return Some(SyntaxResult {
                            is_valid: false,
                            error_message: format!(
                                "Operator '{}' cannot immediately follow opening bracket", 
                                self.token_display(op)
                            ),
                            suggestion: Some("Add operand after opening bracket".to_string()),
                            error_position: Some(second.position),
                        });
                    }
                    _ => {}
                }
            }
        }
        
        None
    }
    
    /// Check for malformed pattern syntax
    fn check_pattern_syntax(&self, tokens: &[Token]) -> Option<SyntaxResult> {
        for window in tokens.windows(2) {
            if let [first, second] = window {
                match (&first.kind, &second.kind) {
                    // Check for invalid blank sequences
                    (TokenKind::BlankNullSequence, TokenKind::Blank) |
                    (TokenKind::BlankSequence, TokenKind::BlankNullSequence) => {
                        return Some(SyntaxResult {
                            is_valid: false,
                            error_message: "Invalid blank pattern sequence".to_string(),
                            suggestion: Some("Use single blank pattern or correct sequence".to_string()),
                            error_position: Some(second.position),
                        });
                    }
                    _ => {}
                }
            }
        }
        None
    }
    
    /// Check for invalid function call syntax
    fn check_function_syntax(&self, tokens: &[Token]) -> Option<SyntaxResult> {
        for window in tokens.windows(2) {
            if let [first, second] = window {
                match (&first.kind, &second.kind) {
                    // Check for number followed by bracket (should be multiplication)
                    (TokenKind::Integer(_), TokenKind::LeftBracket) |
                    (TokenKind::Real(_), TokenKind::LeftBracket) => {
                        return Some(SyntaxResult {
                            is_valid: false,
                            error_message: "Invalid syntax: number followed by bracket".to_string(),
                            suggestion: Some("Use * for multiplication or separate with space".to_string()),
                            error_position: Some(second.position),
                        });
                    }
                    _ => {}
                }
            }
        }
        None
    }
    
    /// Check for rule syntax issues
    fn check_rule_syntax(&self, tokens: &[Token]) -> Option<SyntaxResult> {
        for i in 0..tokens.len() {
            let token = &tokens[i];
            match &token.kind {
                TokenKind::Rule | TokenKind::RuleDelayed => {
                    // Rule should have something before and after
                    if i == 0 {
                        return Some(SyntaxResult {
                            is_valid: false,
                            error_message: "Rule operator cannot be at the beginning".to_string(),
                            suggestion: Some("Add pattern before rule operator".to_string()),
                            error_position: Some(token.position),
                        });
                    }
                    if i == tokens.len() - 1 {
                        // This is handled by multiline detection, so it's not necessarily an error
                        // but we can provide a hint
                        return Some(SyntaxResult {
                            is_valid: true, // Not an error, just incomplete
                            error_message: String::new(),
                            suggestion: Some("Rule needs replacement expression".to_string()),
                            error_position: None,
                        });
                    }
                }
                _ => {}
            }
        }
        None
    }
    
    /// Check if a token kind represents a binary operator
    fn is_binary_operator(&self, kind: &TokenKind) -> bool {
        matches!(kind,
            TokenKind::Plus | TokenKind::Minus | TokenKind::Times | TokenKind::Divide |
            TokenKind::Power | TokenKind::Modulo | TokenKind::Equal | TokenKind::NotEqual |
            TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual |
            TokenKind::And | TokenKind::Or | TokenKind::Rule | TokenKind::RuleDelayed |
            TokenKind::Set | TokenKind::SetDelayed | TokenKind::ReplaceAll | TokenKind::ReplaceRepeated
        )
    }
    
    /// Check if an operator can validly appear at the beginning (unary operators)
    fn is_valid_prefix_operator(&self, kind: &TokenKind) -> bool {
        matches!(kind,
            TokenKind::Plus | TokenKind::Minus | TokenKind::Not
        )
    }
    
    /// Check if two consecutive operators form a valid combination
    fn is_valid_operator_combination(&self, first: &TokenKind, second: &TokenKind) -> bool {
        match (first, second) {
            // Some combinations might be valid in specific contexts
            // For now, be conservative and reject most consecutive operators
            (TokenKind::Plus, TokenKind::Plus) => false, // ++ not supported in Lyra
            (TokenKind::Minus, TokenKind::Minus) => false, // -- not supported 
            (TokenKind::Times, TokenKind::Times) => false, // ** should be ^
            _ => false, // Most consecutive operators are invalid
        }
    }
    
    /// Get display string for a token kind
    fn token_display(&self, kind: &TokenKind) -> String {
        match kind {
            TokenKind::Plus => "+".to_string(),
            TokenKind::Minus => "-".to_string(),
            TokenKind::Times => "*".to_string(),
            TokenKind::Divide => "/".to_string(),
            TokenKind::Power => "^".to_string(),
            TokenKind::Modulo => "mod".to_string(),
            TokenKind::Equal => "==".to_string(),
            TokenKind::NotEqual => "!=".to_string(),
            TokenKind::Less => "<".to_string(),
            TokenKind::LessEqual => "<=".to_string(),
            TokenKind::Greater => ">".to_string(),
            TokenKind::GreaterEqual => ">=".to_string(),
            TokenKind::And => "&&".to_string(),
            TokenKind::Or => "||".to_string(),
            TokenKind::Not => "!".to_string(),
            TokenKind::Rule => "->".to_string(),
            TokenKind::RuleDelayed => ":>".to_string(),
            TokenKind::Set => "=".to_string(),
            TokenKind::SetDelayed => ":=".to_string(),
            TokenKind::ReplaceAll => "/.".to_string(),
            TokenKind::ReplaceRepeated => "//.".to_string(),
            TokenKind::Blank => "_".to_string(),
            TokenKind::BlankSequence => "__".to_string(),
            TokenKind::BlankNullSequence => "___".to_string(),
            _ => format!("{:?}", kind),
        }
    }
}

impl Default for SyntaxChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_syntax() {
        let checker = SyntaxChecker::new();
        
        // Simple valid expressions
        let result = checker.check_syntax("42").unwrap();
        assert!(result.is_valid);
        
        let result = checker.check_syntax("x + y").unwrap();
        assert!(result.is_valid);
        
        let result = checker.check_syntax("Sin[Pi/2]").unwrap();
        assert!(result.is_valid);
        
        let result = checker.check_syntax("f[x_] := x^2").unwrap();
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_consecutive_operators() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("x + + y").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_message.contains("consecutive operators"));
        
        let result = checker.check_syntax("a * / b").unwrap();
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_operator_at_beginning() {
        let checker = SyntaxChecker::new();
        
        // Valid unary operators
        let result = checker.check_syntax("+x").unwrap();
        assert!(result.is_valid);
        
        let result = checker.check_syntax("-y").unwrap();
        assert!(result.is_valid);
        
        // Invalid operators at beginning
        let result = checker.check_syntax("*x").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_message.contains("cannot start with"));
        
        let result = checker.check_syntax("/y").unwrap();
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_operator_after_bracket() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("Sin[+x]").unwrap();
        assert!(result.is_valid); // Unary plus is valid
        
        let result = checker.check_syntax("f[*x]").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_message.contains("cannot immediately follow"));
    }
    
    #[test]
    fn test_invalid_blank_patterns() {
        let checker = SyntaxChecker::new();
        
        // This test might need adjustment based on actual token patterns
        // For now, test the basic framework
        let result = checker.check_syntax("x___y_").unwrap();
        // The exact behavior depends on how the lexer handles this
    }
    
    #[test]
    fn test_number_bracket_syntax() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("2[x]").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_message.contains("number followed by bracket"));
        
        let result = checker.check_syntax("3.14[y]").unwrap();
        assert!(!result.is_valid);
    }
    
    #[test]
    fn test_rule_syntax() {
        let checker = SyntaxChecker::new();
        
        // Valid rule
        let result = checker.check_syntax("x -> x^2").unwrap();
        assert!(result.is_valid);
        
        // Rule at beginning (invalid)
        let result = checker.check_syntax("-> x^2").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_message.contains("cannot be at the beginning"));
    }
    
    #[test]
    fn test_empty_input() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("").unwrap();
        assert!(result.is_valid);
        
        let result = checker.check_syntax("   ").unwrap();
        assert!(result.is_valid);
    }
    
    #[test]
    fn test_tokenization_errors() {
        let checker = SyntaxChecker::new();
        
        // Test with potentially problematic input
        // This depends on what the lexer considers invalid
        let result = checker.check_syntax("###invalid###");
        // Result depends on lexer behavior - either error or valid tokens
        assert!(result.is_ok()); // Should not panic
    }
    
    #[test]
    fn test_error_positions() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("x + + y").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_position.is_some());
        
        let result = checker.check_syntax("*invalid").unwrap();
        assert!(!result.is_valid);
        assert!(result.error_position.is_some());
    }
    
    #[test]
    fn test_suggestions() {
        let checker = SyntaxChecker::new();
        
        let result = checker.check_syntax("2[x]").unwrap();
        assert!(!result.is_valid);
        assert!(result.suggestion.is_some());
        assert!(result.suggestion.unwrap().contains("multiplication"));
        
        let result = checker.check_syntax("x + + y").unwrap();
        assert!(!result.is_valid);
        assert!(result.suggestion.is_some());
    }
}
