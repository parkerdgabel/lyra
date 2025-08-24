#![allow(unused_variables)]
//! Multiline expression detection for Lyra REPL
//!
//! This module detects when expressions are incomplete and require multiline continuation,
//! providing intelligent hints for appropriate continuation.

use crate::lexer::{Lexer, TokenKind};

/// Multiline expression detector
pub struct MultilineDetector {
    // No state needed for stateless detection
}

/// Result of multiline detection analysis
#[derive(Debug, Clone, PartialEq)]
pub struct MultilineResult {
    /// Whether the expression needs continuation
    pub needs_continuation: bool,
    /// Reason for continuation requirement
    pub reason: String,
    /// Suggested continuation text
    pub suggested_continuation: Option<String>,
    /// Detected continuation type
    pub continuation_type: ContinuationType,
}

/// Types of continuation that can be detected
#[derive(Debug, Clone, PartialEq)]
pub enum ContinuationType {
    /// No continuation needed
    None,
    /// Continuation due to unclosed brackets
    UnclosedBrackets,
    /// Continuation due to trailing operators
    TrailingOperator,
    /// Continuation due to incomplete function definition
    FunctionDefinition,
    /// Continuation due to incomplete rule
    IncompleteRule,
    /// Continuation due to incomplete assignment
    IncompleteAssignment,
    /// Continuation due to incomplete list/expression
    IncompleteExpression,
}

impl MultilineDetector {
    /// Create a new multiline detector
    pub fn new() -> Self {
        Self {}
    }
    
    /// Detect if expression needs multiline continuation
    pub fn detect_multiline(&self, input: &str) -> Result<MultilineResult, String> {
        let trimmed = input.trim();
        
        // Empty input doesn't need continuation
        if trimmed.is_empty() {
            return Ok(MultilineResult {
                needs_continuation: false,
                reason: String::new(),
                suggested_continuation: None,
                continuation_type: ContinuationType::None,
            });
        }
        
        // Try to tokenize the input
        let mut lexer = Lexer::new(trimmed);
        let tokens = match lexer.tokenize() {
            Ok(tokens) => tokens,
            Err(_) => {
                // If tokenization fails, we might need continuation
                return self.detect_from_text_analysis(trimmed);
            }
        };
        
        // Analyze tokens for continuation patterns
        self.analyze_tokens(&tokens, trimmed)
    }
    
    /// Analyze tokens to detect multiline patterns
    fn analyze_tokens(&self, tokens: &[crate::lexer::Token], input: &str) -> Result<MultilineResult, String> {
        if tokens.is_empty() {
            return Ok(MultilineResult {
                needs_continuation: false,
                reason: String::new(),
                suggested_continuation: None,
                continuation_type: ContinuationType::None,
            });
        }
        
        // Check for trailing operators
        if let Some(last_token) = tokens.last() {
            match &last_token.kind {
                TokenKind::Plus | TokenKind::Minus | TokenKind::Times | TokenKind::Divide 
                | TokenKind::Power | TokenKind::Modulo => {
                    return Ok(MultilineResult {
                        needs_continuation: true,
                        reason: "Trailing arithmetic operator".to_string(),
                        suggested_continuation: Some("    ".to_string()),
                        continuation_type: ContinuationType::TrailingOperator,
                    });
                }
                TokenKind::Rule | TokenKind::RuleDelayed => {
                    return Ok(MultilineResult {
                        needs_continuation: true,
                        reason: "Incomplete rule definition".to_string(),
                        suggested_continuation: Some("    ".to_string()),
                        continuation_type: ContinuationType::IncompleteRule,
                    });
                }
                TokenKind::Set | TokenKind::SetDelayed => {
                    return Ok(MultilineResult {
                        needs_continuation: true,
                        reason: "Incomplete assignment".to_string(),
                        suggested_continuation: Some("    ".to_string()),
                        continuation_type: ContinuationType::IncompleteAssignment,
                    });
                }
                TokenKind::ReplaceAll | TokenKind::ReplaceRepeated => {
                    return Ok(MultilineResult {
                        needs_continuation: true,
                        reason: "Incomplete replacement rule".to_string(),
                        suggested_continuation: Some("    ".to_string()),
                        continuation_type: ContinuationType::IncompleteRule,
                    });
                }
                TokenKind::Comma => {
                    return Ok(MultilineResult {
                        needs_continuation: true,
                        reason: "Trailing comma in expression".to_string(),
                        suggested_continuation: Some("    ".to_string()),
                        continuation_type: ContinuationType::IncompleteExpression,
                    });
                }
                _ => {}
            }
        }
        
        // Check for function definition patterns
        if self.is_function_definition_pattern(&tokens) {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Incomplete function definition".to_string(),
                suggested_continuation: Some("    ".to_string()),
                continuation_type: ContinuationType::FunctionDefinition,
            });
        }
        
        // Check bracket balance (this should be caught by bracket matcher, but as backup)
        if self.has_unclosed_brackets(&tokens) {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Unclosed brackets".to_string(),
                suggested_continuation: Some("    ".to_string()),
                continuation_type: ContinuationType::UnclosedBrackets,
            });
        }
        
        // No continuation needed
        Ok(MultilineResult {
            needs_continuation: false,
            reason: String::new(),
            suggested_continuation: None,
            continuation_type: ContinuationType::None,
        })
    }
    
    /// Detect continuation needs from text analysis when tokenization fails
    fn detect_from_text_analysis(&self, input: &str) -> Result<MultilineResult, String> {
        // Simple heuristics for when lexer fails
        
        // Check for unclosed quotes
        if input.chars().filter(|&c| c == '"').count() % 2 == 1 {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Unclosed string quote".to_string(),
                suggested_continuation: None,
                continuation_type: ContinuationType::IncompleteExpression,
            });
        }
        
        // Count bracket types
        let open_parens = input.chars().filter(|&c| c == '(').count();
        let close_parens = input.chars().filter(|&c| c == ')').count();
        let open_brackets = input.chars().filter(|&c| c == '[').count();
        let close_brackets = input.chars().filter(|&c| c == ']').count();
        let open_braces = input.chars().filter(|&c| c == '{').count();
        let close_braces = input.chars().filter(|&c| c == '}').count();
        
        if open_parens > close_parens || open_brackets > close_brackets || open_braces > close_braces {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Unclosed brackets detected".to_string(),
                suggested_continuation: Some("    ".to_string()),
                continuation_type: ContinuationType::UnclosedBrackets,
            });
        }
        
        // Check for trailing operators (simple string-based check)
        let trimmed = input.trim();
        if trimmed.ends_with('+') || trimmed.ends_with('-') || trimmed.ends_with('*') 
           || trimmed.ends_with('/') || trimmed.ends_with('^') {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Expression ends with operator".to_string(),
                suggested_continuation: Some("    ".to_string()),
                continuation_type: ContinuationType::TrailingOperator,
            });
        }
        
        // Check for rule operators
        if trimmed.ends_with("->") || trimmed.ends_with(":>") || trimmed.ends_with("/.") {
            return Ok(MultilineResult {
                needs_continuation: true,
                reason: "Incomplete rule or replacement".to_string(),
                suggested_continuation: Some("    ".to_string()),
                continuation_type: ContinuationType::IncompleteRule,
            });
        }
        
        // No clear continuation pattern detected
        Ok(MultilineResult {
            needs_continuation: false,
            reason: String::new(),
            suggested_continuation: None,
            continuation_type: ContinuationType::None,
        })
    }
    
    /// Check if tokens represent a function definition pattern
    fn is_function_definition_pattern(&self, tokens: &[crate::lexer::Token]) -> bool {
        // Look for patterns like: Symbol[pattern_] := or Symbol[pattern_] =
        if tokens.len() < 4 {
            return false;
        }
        
        for window in tokens.windows(4) {
            if let [first, second, third, fourth] = window {
                match (&first.kind, &second.kind, &third.kind, &fourth.kind) {
                    (TokenKind::Symbol(_), TokenKind::LeftBracket, _, TokenKind::SetDelayed) |
                    (TokenKind::Symbol(_), TokenKind::LeftBracket, _, TokenKind::Set) => {
                        return true;
                    }
                    _ => {}
                }
            }
        }
        
        false
    }
    
    /// Check if there are unclosed brackets in the token stream
    fn has_unclosed_brackets(&self, tokens: &[crate::lexer::Token]) -> bool {
        let mut depth: i32 = 0;
        
        for token in tokens {
            match &token.kind {
                TokenKind::LeftParen | TokenKind::LeftBracket | TokenKind::LeftBrace 
                | TokenKind::LeftDoubleBracket => {
                    depth += 1;
                }
                TokenKind::RightParen | TokenKind::RightBracket | TokenKind::RightBrace 
                | TokenKind::RightDoubleBracket => {
                    depth -= 1;
                }
                _ => {}
            }
        }
        
        depth > 0
    }
    
    /// Generate appropriate indentation for continuation
    pub fn generate_continuation_indent(&self, input: &str, continuation_type: ContinuationType) -> String {
        match continuation_type {
            ContinuationType::None => String::new(),
            ContinuationType::UnclosedBrackets => {
                // Calculate nesting depth and add appropriate indentation
                let depth = self.calculate_bracket_depth(input);
                "    ".repeat(depth)
            }
            ContinuationType::TrailingOperator => "    ".to_string(),
            ContinuationType::FunctionDefinition => "    ".to_string(),
            ContinuationType::IncompleteRule => "    ".to_string(),
            ContinuationType::IncompleteAssignment => "    ".to_string(),
            ContinuationType::IncompleteExpression => "    ".to_string(),
        }
    }
    
    /// Calculate the current bracket nesting depth
    fn calculate_bracket_depth(&self, input: &str) -> usize {
        let mut depth: i32 = 0;
        let mut in_string = false;
        
        for ch in input.chars() {
            match ch {
                '"' => in_string = !in_string,
                '(' | '[' | '{' if !in_string => depth += 1,
                ')' | ']' | '}' if !in_string => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        
        depth.max(0) as usize
    }
}

impl Default for MultilineDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_continuation_needed() {
        let detector = MultilineDetector::new();
        
        // Simple complete expressions
        let result = detector.detect_multiline("42").unwrap();
        assert!(!result.needs_continuation);
        
        let result = detector.detect_multiline("Sin[Pi/2]").unwrap();
        assert!(!result.needs_continuation);
        
        let result = detector.detect_multiline("x + y").unwrap();
        assert!(!result.needs_continuation);
    }
    
    #[test]
    fn test_trailing_operators() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_multiline("x +").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::TrailingOperator);
        
        let result = detector.detect_multiline("Sin[x] *").unwrap();
        assert!(result.needs_continuation);
        
        let result = detector.detect_multiline("a / b -").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_incomplete_rules() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_multiline("x ->").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::IncompleteRule);
        
        let result = detector.detect_multiline("pattern :>").unwrap();
        assert!(result.needs_continuation);
        
        let result = detector.detect_multiline("expr /.").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_incomplete_assignments() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_multiline("f[x_] :=").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::IncompleteAssignment);
        
        let result = detector.detect_multiline("var =").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_trailing_comma() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_multiline("{1, 2, 3,").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::IncompleteExpression);
        
        let result = detector.detect_multiline("f[a, b,").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_unclosed_brackets_fallback() {
        let detector = MultilineDetector::new();
        
        // Test text-based analysis when tokenization might fail
        let result = detector.detect_from_text_analysis("Sin[").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::UnclosedBrackets);
        
        let result = detector.detect_from_text_analysis("(1 + 2").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_unclosed_strings() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_from_text_analysis("\"hello").unwrap();
        assert!(result.needs_continuation);
        assert_eq!(result.continuation_type, ContinuationType::IncompleteExpression);
        
        let result = detector.detect_from_text_analysis("Print[\"test").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_bracket_depth_calculation() {
        let detector = MultilineDetector::new();
        
        assert_eq!(detector.calculate_bracket_depth(""), 0);
        assert_eq!(detector.calculate_bracket_depth("("), 1);
        assert_eq!(detector.calculate_bracket_depth("(("), 2);
        assert_eq!(detector.calculate_bracket_depth("Sin[Cos["), 2);
        assert_eq!(detector.calculate_bracket_depth("Sin[Cos[x]]"), 0);
    }
    
    #[test]
    fn test_continuation_indentation() {
        let detector = MultilineDetector::new();
        
        let indent = detector.generate_continuation_indent("(", ContinuationType::UnclosedBrackets);
        assert_eq!(indent, "    ");
        
        let indent = detector.generate_continuation_indent("((", ContinuationType::UnclosedBrackets);
        assert_eq!(indent, "        ");
        
        let indent = detector.generate_continuation_indent("x +", ContinuationType::TrailingOperator);
        assert_eq!(indent, "    ");
    }
    
    #[test]
    fn test_function_definition_detection() {
        let detector = MultilineDetector::new();
        
        // This would require proper tokenization to work fully
        // For now, test the basic pattern matching logic
        let result = detector.detect_multiline("f[x_] :=").unwrap();
        assert!(result.needs_continuation);
    }
    
    #[test]
    fn test_empty_input() {
        let detector = MultilineDetector::new();
        
        let result = detector.detect_multiline("").unwrap();
        assert!(!result.needs_continuation);
        
        let result = detector.detect_multiline("   ").unwrap();
        assert!(!result.needs_continuation);
    }
}
