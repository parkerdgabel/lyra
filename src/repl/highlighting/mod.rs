//! Real-time syntax highlighting for Lyra REPL
//!
//! This module provides comprehensive syntax highlighting capabilities for the Lyra REPL,
//! including token-based highlighting, configurable themes, and performance optimization.

use crate::lexer::{Lexer, Token, TokenKind};
use crate::repl::config::ReplConfig;
use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::Mutex;

pub mod themes;

use themes::{ColorTheme, HighlightColor};

/// Syntax highlighter for Lyra expressions
pub struct LyraSyntaxHighlighter {
    /// Current color theme
    theme: ColorTheme,
    /// Cache for highlighted expressions to improve performance (with interior mutability)
    highlight_cache: Mutex<HashMap<String, String>>,
    /// Maximum cache size to prevent memory growth
    max_cache_size: usize,
}

impl LyraSyntaxHighlighter {
    /// Create a new syntax highlighter with the given configuration
    pub fn new(config: &ReplConfig) -> Self {
        let theme = ColorTheme::from_config(config);
        
        Self {
            theme,
            highlight_cache: Mutex::new(HashMap::new()),
            max_cache_size: 1000, // Configurable cache size
        }
    }
    
    /// Update the highlighter's configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.theme = ColorTheme::from_config(config);
        // Clear cache when theme changes
        if let Ok(mut cache) = self.highlight_cache.lock() {
            cache.clear();
        }
    }
    
    /// Highlight a complete line of Lyra code
    pub fn highlight_line<'l>(&self, line: &'l str) -> Cow<'l, str> {
        // Check cache first for performance
        if let Ok(cache) = self.highlight_cache.lock() {
            if let Some(cached) = cache.get(line) {
                return Cow::Owned(cached.clone());
            }
        }
        
        // If line is too long, don't cache to prevent memory issues
        let should_cache = line.len() < 1000;
        
        let highlighted = self.highlight_tokens(line);
        
        // Cache the result if appropriate
        if should_cache {
            if let Ok(mut cache) = self.highlight_cache.lock() {
                // Manage cache size
                if cache.len() >= self.max_cache_size {
                    cache.clear();
                }
                cache.insert(line.to_string(), highlighted.clone());
            }
        }
        
        Cow::Owned(highlighted)
    }
    
    /// Core highlighting logic using token-based approach
    fn highlight_tokens(&self, line: &str) -> String {
        let mut result = String::with_capacity(line.len() * 2); // Estimate with color codes
        let mut lexer = Lexer::new(line);
        
        // Get all tokens from the lexer
        match lexer.tokenize() {
            Ok(tokens) => {
                let mut last_end = 0;
                
                for token in tokens {
                    let token_start = token.position;
                    let token_end = token.position + token.length;
                    
                    // Add any whitespace/content between tokens
                    if token_start > last_end {
                        result.push_str(&line[last_end..token_start]);
                    }
                    
                    // Apply highlighting based on token type
                    let colored_token = self.colorize_token(&token, line);
                    result.push_str(&colored_token);
                    
                    last_end = token_end;
                }
                
                // Add any remaining content
                if last_end < line.len() {
                    result.push_str(&line[last_end..]);
                }
            }
            Err(_) => {
                // On lexer error, highlight entire line as error
                let error_colored = self.theme.apply_color(line, HighlightColor::Error);
                result.push_str(&error_colored);
            }
        }
        
        result
    }
    
    /// Apply appropriate color to a token based on its type
    fn colorize_token(&self, token: &Token, line: &str) -> String {
        let token_start = token.position;
        let token_end = token.position + token.length;
        let token_text = &line[token_start..token_end];
        
        let color = match &token.kind {
            TokenKind::Integer(_) | TokenKind::Real(_) | TokenKind::Rational(_, _) 
            | TokenKind::Complex(_, _) | TokenKind::BigInt(_) | TokenKind::BigDecimal(_) 
            | TokenKind::HexInteger(_) => HighlightColor::Number,
            
            TokenKind::String(_) | TokenKind::InterpolatedString(_) => HighlightColor::String,
            
            TokenKind::Symbol(name) => {
                if self.is_stdlib_function(name) {
                    HighlightColor::Function
                } else {
                    HighlightColor::Variable
                }
            }
            
            TokenKind::ContextSymbol(_) => HighlightColor::Variable,
            
            TokenKind::LeftBracket | TokenKind::RightBracket | TokenKind::LeftBrace 
            | TokenKind::RightBrace | TokenKind::LeftParen | TokenKind::RightParen
            | TokenKind::LeftDoubleBracket | TokenKind::RightDoubleBracket => HighlightColor::Bracket,
            
            TokenKind::Comma | TokenKind::Semicolon | TokenKind::Dot | TokenKind::Colon 
            | TokenKind::Question => HighlightColor::Punctuation,
            
            TokenKind::Plus | TokenKind::Minus | TokenKind::Times | TokenKind::Divide 
            | TokenKind::Modulo | TokenKind::Power | TokenKind::Equal | TokenKind::NotEqual
            | TokenKind::Less | TokenKind::LessEqual | TokenKind::Greater | TokenKind::GreaterEqual
            | TokenKind::And | TokenKind::Or | TokenKind::Not | TokenKind::Pipeline
            | TokenKind::Postfix | TokenKind::Prefix | TokenKind::Alternative => HighlightColor::Operator,
            
            TokenKind::Set | TokenKind::SetDelayed | TokenKind::Rule | TokenKind::RuleDelayed
            | TokenKind::ReplaceAll | TokenKind::ReplaceRepeated | TokenKind::Arrow 
            | TokenKind::Range | TokenKind::Condition => HighlightColor::Operator,
            
            TokenKind::LeftAssoc | TokenKind::RightAssoc => HighlightColor::Operator,
            
            TokenKind::Blank | TokenKind::BlankSequence | TokenKind::BlankNullSequence => HighlightColor::Variable,
            
            TokenKind::StringJoin | TokenKind::Backtick | TokenKind::Slot | TokenKind::NumberedSlot(_) | TokenKind::PureFunction => HighlightColor::Operator,
            
            TokenKind::Comment(_) => HighlightColor::Comment,
            
            TokenKind::Whitespace | TokenKind::Eof => return token_text.to_string(), // No highlighting
        };
        
        self.theme.apply_color(token_text, color)
    }
    
    /// Check if a symbol is a standard library function
    fn is_stdlib_function(&self, name: &str) -> bool {
        // Common mathematical and list functions
        matches!(name, 
            "Sin" | "Cos" | "Tan" | "Log" | "Exp" | "Sqrt" | "Abs" | "Max" | "Min" |
            "Plus" | "Times" | "Power" | "Subtract" | "Divide" | "Mod" |
            "Length" | "First" | "Last" | "Rest" | "Take" | "Drop" | "Append" |
            "Map" | "Apply" | "Select" | "Cases" | "Count" | "Sort" | "Reverse" |
            "Range" | "Table" | "Sum" | "Product" | "Mean" | "Variance" |
            "Random" | "RandomReal" | "RandomInteger" | "SeedRandom" |
            "If" | "Switch" | "Which" | "Do" | "For" | "While" |
            "Print" | "Echo" | "ToString" | "ToExpression"
        )
    }
    
    /// Check if character highlighting should be enabled at the given position
    pub fn should_highlight_char(&self, line: &str, pos: usize) -> bool {
        // Enable bracket matching for brackets and parentheses
        if pos >= line.len() {
            return false;
        }
        
        let ch = line.chars().nth(pos).unwrap_or('\0');
        matches!(ch, '[' | ']' | '{' | '}' | '(' | ')')
    }
    
    /// Get statistics about the highlighter for debugging
    pub fn get_stats(&self) -> HighlighterStats {
        let cache_size = self.highlight_cache.lock()
            .map(|cache| cache.len())
            .unwrap_or(0);
            
        HighlighterStats {
            cache_size,
            max_cache_size: self.max_cache_size,
            current_theme: self.theme.name().to_string(),
        }
    }
    
    /// Clear the highlighting cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.highlight_cache.lock() {
            cache.clear();
        }
    }
}

/// Statistics about the highlighter's performance and state
#[derive(Debug, Clone)]
pub struct HighlighterStats {
    pub cache_size: usize,
    pub max_cache_size: usize,
    pub current_theme: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::config::ReplConfig;

    #[test]
    fn test_basic_highlighting() {
        let config = ReplConfig::default();
        let mut highlighter = LyraSyntaxHighlighter::new(&config);
        
        // Test number highlighting
        let result = highlighter.highlight_line("42");
        assert!(result.contains("42")); // Should contain the number
        
        // Test function highlighting
        let result = highlighter.highlight_line("Sin[x]");
        assert!(result.contains("Sin")); // Should contain the function name
        
        // Test string highlighting
        let result = highlighter.highlight_line("\"hello\"");
        assert!(result.contains("hello")); // Should contain the string content
    }
    
    #[test]
    fn test_complex_expression_highlighting() {
        let config = ReplConfig::default();
        let mut highlighter = LyraSyntaxHighlighter::new(&config);
        
        let expr = "Sin[Pi/2] + Cos[0] * Log[E]";
        let result = highlighter.highlight_line(expr);
        
        // Should highlight function names, numbers, and operators
        assert!(result.len() >= expr.len()); // Result should be at least as long (due to color codes)
    }
    
    #[test]
    fn test_highlighting_cache() {
        let config = ReplConfig::default();
        let mut highlighter = LyraSyntaxHighlighter::new(&config);
        
        let expr = "Sin[x]";
        
        // First highlighting
        let result1 = highlighter.highlight_line(expr);
        assert_eq!(highlighter.get_stats().cache_size, 1);
        
        // Second highlighting should use cache
        let result2 = highlighter.highlight_line(expr);
        assert_eq!(result1, result2);
        assert_eq!(highlighter.get_stats().cache_size, 1);
    }
    
    #[test]
    fn test_bracket_highlighting() {
        let config = ReplConfig::default();
        let highlighter = LyraSyntaxHighlighter::new(&config);
        
        // Test bracket character highlighting
        assert!(highlighter.should_highlight_char("[", 0));
        assert!(highlighter.should_highlight_char("]", 0));
        assert!(highlighter.should_highlight_char("{", 0));
        assert!(highlighter.should_highlight_char("}", 0));
        assert!(!highlighter.should_highlight_char("a", 0));
    }
    
    #[test]
    fn test_stdlib_function_recognition() {
        let config = ReplConfig::default();
        let highlighter = LyraSyntaxHighlighter::new(&config);
        
        // Test known stdlib functions
        assert!(highlighter.is_stdlib_function("Sin"));
        assert!(highlighter.is_stdlib_function("Length"));
        assert!(highlighter.is_stdlib_function("Map"));
        
        // Test unknown functions
        assert!(!highlighter.is_stdlib_function("MyCustomFunction"));
        assert!(!highlighter.is_stdlib_function("x"));
    }
    
    #[test]
    fn test_error_highlighting() {
        let config = ReplConfig::default();
        let mut highlighter = LyraSyntaxHighlighter::new(&config);
        
        // Test malformed expression
        let malformed = "Sin[unclosed";
        let result = highlighter.highlight_line(malformed);
        
        // Should still produce some output even with lexer errors
        assert!(!result.is_empty());
    }
}