//! Bracket and quote balance checking for Lyra expressions
//!
//! This module provides sophisticated bracket and quote matching that understands
//! Lyra's syntax including special bracket types and string contexts.

use std::collections::VecDeque;

/// Bracket and quote balance checker
pub struct BracketMatcher {
    // No state needed for stateless bracket matching
}

/// Result of bracket/quote balance checking
#[derive(Debug, Clone, PartialEq)]
pub struct BalanceResult {
    /// Whether brackets and quotes are balanced
    pub is_balanced: bool,
    /// Error message if not balanced
    pub error_message: String,
    /// Suggested fix if available
    pub suggested_fix: Option<String>,
    /// Position of the unmatched bracket/quote
    pub error_position: Option<usize>,
}

/// Types of brackets/quotes that can be matched
#[derive(Debug, Clone, Copy, PartialEq)]
enum BracketType {
    /// Round parentheses ()
    Paren,
    /// Square brackets []
    Bracket,
    /// Curly braces {}
    Brace,
    /// Double square brackets [[]]
    DoubleBracket,
    /// String quotes ""
    StringQuote,
    /// Context backticks ``
    ContextQuote,
}

/// Stack entry for tracking nested brackets
#[derive(Debug, Clone)]
struct StackEntry {
    bracket_type: BracketType,
    position: usize,
    char: char,
}

impl BracketMatcher {
    /// Create a new bracket matcher
    pub fn new() -> Self {
        Self {}
    }
    
    /// Check if brackets and quotes are balanced in the input
    pub fn check_balance(&self, input: &str) -> Result<BalanceResult, String> {
        let mut stack = VecDeque::new();
        let mut in_string = false;
        let mut string_quote_pos = None;
        let mut chars: Vec<char> = input.chars().collect();
        let mut i = 0;
        
        while i < chars.len() {
            let ch = chars[i];
            
            match ch {
                '"' => {
                    if in_string {
                        // Closing string quote
                        in_string = false;
                        string_quote_pos = None;
                    } else {
                        // Opening string quote
                        in_string = true;
                        string_quote_pos = Some(i);
                    }
                }
                '`' if !in_string => {
                    // Context symbols - simplified handling for now
                    // TODO: Implement proper context symbol parsing
                }
                '(' if !in_string => {
                    stack.push_back(StackEntry {
                        bracket_type: BracketType::Paren,
                        position: i,
                        char: ch,
                    });
                }
                ')' if !in_string => {
                    match stack.pop_back() {
                        Some(entry) if entry.bracket_type == BracketType::Paren => {
                            // Matched correctly
                        }
                        Some(entry) => {
                            return Ok(BalanceResult {
                                is_balanced: false,
                                error_message: format!(
                                    "Mismatched bracket: expected '{}' but found ')'", 
                                    self.closing_bracket(entry.bracket_type)
                                ),
                                suggested_fix: Some(format!(
                                    "Change ')' to '{}' or add matching '('", 
                                    self.closing_bracket(entry.bracket_type)
                                )),
                                error_position: Some(i),
                            });
                        }
                        None => {
                            return Ok(BalanceResult {
                                is_balanced: false,
                                error_message: "Unmatched closing parenthesis ')'".to_string(),
                                suggested_fix: Some("Add matching '(' or remove ')'".to_string()),
                                error_position: Some(i),
                            });
                        }
                    }
                }
                '[' if !in_string => {
                    // Check for double bracket [[
                    if i + 1 < chars.len() && chars[i + 1] == '[' {
                        stack.push_back(StackEntry {
                            bracket_type: BracketType::DoubleBracket,
                            position: i,
                            char: ch,
                        });
                        i += 1; // Skip the second [
                    } else {
                        stack.push_back(StackEntry {
                            bracket_type: BracketType::Bracket,
                            position: i,
                            char: ch,
                        });
                    }
                }
                ']' if !in_string => {
                    // Check for double bracket ]]
                    if i + 1 < chars.len() && chars[i + 1] == ']' {
                        match stack.pop_back() {
                            Some(entry) if entry.bracket_type == BracketType::DoubleBracket => {
                                i += 1; // Skip the second ]
                            }
                            Some(entry) => {
                                return Ok(BalanceResult {
                                    is_balanced: false,
                                    error_message: format!(
                                        "Mismatched bracket: expected '{}' but found ']]'", 
                                        self.closing_bracket(entry.bracket_type)
                                    ),
                                    suggested_fix: Some("Check bracket nesting".to_string()),
                                    error_position: Some(i),
                                });
                            }
                            None => {
                                return Ok(BalanceResult {
                                    is_balanced: false,
                                    error_message: "Unmatched closing double bracket ']]'".to_string(),
                                    suggested_fix: Some("Add matching '[['".to_string()),
                                    error_position: Some(i),
                                });
                            }
                        }
                    } else {
                        match stack.pop_back() {
                            Some(entry) if entry.bracket_type == BracketType::Bracket => {
                                // Matched correctly
                            }
                            Some(entry) => {
                                return Ok(BalanceResult {
                                    is_balanced: false,
                                    error_message: format!(
                                        "Mismatched bracket: expected '{}' but found ']'", 
                                        self.closing_bracket(entry.bracket_type)
                                    ),
                                    suggested_fix: Some("Check bracket types".to_string()),
                                    error_position: Some(i),
                                });
                            }
                            None => {
                                return Ok(BalanceResult {
                                    is_balanced: false,
                                    error_message: "Unmatched closing bracket ']'".to_string(),
                                    suggested_fix: Some("Add matching '['".to_string()),
                                    error_position: Some(i),
                                });
                            }
                        }
                    }
                }
                '{' if !in_string => {
                    stack.push_back(StackEntry {
                        bracket_type: BracketType::Brace,
                        position: i,
                        char: ch,
                    });
                }
                '}' if !in_string => {
                    match stack.pop_back() {
                        Some(entry) if entry.bracket_type == BracketType::Brace => {
                            // Matched correctly
                        }
                        Some(entry) => {
                            return Ok(BalanceResult {
                                is_balanced: false,
                                error_message: format!(
                                    "Mismatched bracket: expected '{}' but found '}}'", 
                                    self.closing_bracket(entry.bracket_type)
                                ),
                                suggested_fix: Some("Check bracket types".to_string()),
                                error_position: Some(i),
                            });
                        }
                        None => {
                            return Ok(BalanceResult {
                                is_balanced: false,
                                error_message: "Unmatched closing brace '}'".to_string(),
                                suggested_fix: Some("Add matching '{'".to_string()),
                                error_position: Some(i),
                            });
                        }
                    }
                }
                _ => {
                    // Other characters are ignored for bracket matching
                }
            }
            
            i += 1;
        }
        
        // Check for unclosed strings
        if in_string {
            return Ok(BalanceResult {
                is_balanced: false,
                error_message: "Unclosed string quote".to_string(),
                suggested_fix: Some("Add closing quote '\"'".to_string()),
                error_position: string_quote_pos,
            });
        }
        
        // Check for unclosed brackets
        if let Some(entry) = stack.back() {
            let closing = self.closing_bracket(entry.bracket_type.clone());
            return Ok(BalanceResult {
                is_balanced: false,
                error_message: format!("Unclosed bracket '{}'", entry.char),
                suggested_fix: Some(format!("Add closing bracket '{}'", closing)),
                error_position: Some(entry.position),
            });
        }
        
        // Everything is balanced
        Ok(BalanceResult {
            is_balanced: true,
            error_message: String::new(),
            suggested_fix: None,
            error_position: None,
        })
    }
    
    /// Get the closing bracket character for a bracket type
    fn closing_bracket(&self, bracket_type: BracketType) -> char {
        match bracket_type {
            BracketType::Paren => ')',
            BracketType::Bracket => ']',
            BracketType::Brace => '}',
            BracketType::DoubleBracket => ']', // ]] is two chars, simplified
            BracketType::StringQuote => '"',
            BracketType::ContextQuote => '`',
        }
    }
    
    /// Count the depth of nesting at a given position
    pub fn get_nesting_depth(&self, input: &str, position: usize) -> usize {
        let mut depth: i32 = 0;
        let mut in_string = false;
        
        for (i, ch) in input.chars().enumerate() {
            if i >= position {
                break;
            }
            
            match ch {
                '"' => in_string = !in_string,
                '(' | '[' | '{' if !in_string => depth += 1,
                ')' | ']' | '}' if !in_string => depth = depth.saturating_sub(1),
                _ => {}
            }
        }
        
        depth.max(0) as usize
    }
    
    /// Suggest appropriate indentation based on nesting
    pub fn suggest_indentation(&self, input: &str) -> String {
        let depth = self.get_nesting_depth(input, input.len());
        " ".repeat(depth * 4) // 4 spaces per nesting level
    }
}

impl Default for BracketMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_balanced_brackets() {
        let matcher = BracketMatcher::new();
        
        // Simple balanced cases
        let result = matcher.check_balance("()").unwrap();
        assert!(result.is_balanced);
        
        let result = matcher.check_balance("[]").unwrap();
        assert!(result.is_balanced);
        
        let result = matcher.check_balance("{}").unwrap();
        assert!(result.is_balanced);
        
        // Complex nested cases
        let result = matcher.check_balance("Sin[Cos[x]]").unwrap();
        assert!(result.is_balanced);
        
        let result = matcher.check_balance("{[1, 2], (3, 4)}").unwrap();
        assert!(result.is_balanced);
    }
    
    #[test]
    fn test_unbalanced_brackets() {
        let matcher = BracketMatcher::new();
        
        // Missing closing brackets
        let result = matcher.check_balance("(").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_message.contains("Unclosed"));
        
        let result = matcher.check_balance("Sin[x").unwrap();
        assert!(!result.is_balanced);
        
        // Extra closing brackets
        let result = matcher.check_balance(")").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_message.contains("Unmatched"));
        
        let result = matcher.check_balance("Sin[x]]").unwrap();
        assert!(!result.is_balanced);
    }
    
    #[test]
    fn test_mismatched_brackets() {
        let matcher = BracketMatcher::new();
        
        let result = matcher.check_balance("(]").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_message.contains("Mismatched"));
        
        let result = matcher.check_balance("{)").unwrap();
        assert!(!result.is_balanced);
    }
    
    #[test]
    fn test_string_handling() {
        let matcher = BracketMatcher::new();
        
        // Brackets inside strings should be ignored
        let result = matcher.check_balance("\"hello [world]\"").unwrap();
        assert!(result.is_balanced);
        
        // Unclosed string
        let result = matcher.check_balance("\"hello").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_message.contains("Unclosed string"));
        
        // Brackets with strings
        let result = matcher.check_balance("Print[\"hello\"]").unwrap();
        assert!(result.is_balanced);
    }
    
    #[test]
    fn test_double_brackets() {
        let matcher = BracketMatcher::new();
        
        // Balanced double brackets
        let result = matcher.check_balance("expr[[1, 2]]").unwrap();
        assert!(result.is_balanced);
        
        // Unbalanced double brackets
        let result = matcher.check_balance("expr[[1, 2]").unwrap();
        assert!(!result.is_balanced);
    }
    
    #[test]
    fn test_nesting_depth() {
        let matcher = BracketMatcher::new();
        
        assert_eq!(matcher.get_nesting_depth("", 0), 0);
        assert_eq!(matcher.get_nesting_depth("(", 1), 1);
        assert_eq!(matcher.get_nesting_depth("((", 2), 2);
        assert_eq!(matcher.get_nesting_depth("(()", 3), 1);
        assert_eq!(matcher.get_nesting_depth("Sin[Cos[", 8), 2);
    }
    
    #[test]
    fn test_indentation_suggestion() {
        let matcher = BracketMatcher::new();
        
        assert_eq!(matcher.suggest_indentation(""), "");
        assert_eq!(matcher.suggest_indentation("("), "    ");
        assert_eq!(matcher.suggest_indentation("(("), "        ");
        assert_eq!(matcher.suggest_indentation("Sin[Cos["), "        ");
    }
    
    #[test]
    fn test_error_positions() {
        let matcher = BracketMatcher::new();
        
        let result = matcher.check_balance("Sin[x)").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_position.is_some());
        
        let result = matcher.check_balance("\"unclosed").unwrap();
        assert!(!result.is_balanced);
        assert!(result.error_position.is_some());
    }
}