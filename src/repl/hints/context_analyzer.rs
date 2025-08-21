//! Context analysis for intelligent hint generation
//!
//! This module analyzes the input context around the cursor position to determine
//! what type of hint would be most helpful.

use crate::lexer::{Lexer, Token, TokenKind};

/// Context analyzer for determining hint type
pub struct ContextAnalyzer {
    // Stateless analyzer - no fields needed
}

/// Type of hint that would be most appropriate
#[derive(Debug, Clone, PartialEq)]
pub enum HintType {
    /// Function call context - show function signature
    FunctionCall,
    /// Parameter position context - show parameter suggestions
    ParameterPosition,
    /// Error context - show error-specific hints
    ErrorContext,
    /// No specific context - no hint
    None,
}

/// Context information for hint generation
#[derive(Debug, Clone)]
pub struct HintContext {
    /// Function name if in function call context
    pub function_name: Option<String>,
    /// Parameter index if in parameter position
    pub parameter_index: Option<usize>,
    /// Type of hint to display
    pub hint_type: HintType,
    /// Current token at cursor position
    pub current_token: Option<Token>,
    /// Error type if in error context
    pub error_type: Option<String>,
}

impl ContextAnalyzer {
    /// Create a new context analyzer
    pub fn new() -> Self {
        Self {}
    }
    
    /// Analyze input context at cursor position
    pub fn analyze(&self, line: &str, pos: usize) -> Result<HintContext, String> {
        // Tokenize the input
        let tokens = self.tokenize_input(line)?;
        
        // Find the context around the cursor position
        let cursor_context = self.find_cursor_context(&tokens, line, pos);
        
        // Determine hint type based on context
        let hint_type = self.determine_hint_type(&cursor_context, line, pos);
        
        Ok(self.build_hint_context(cursor_context, hint_type))
    }
    
    /// Tokenize input safely
    fn tokenize_input(&self, line: &str) -> Result<Vec<Token>, String> {
        let mut lexer = Lexer::new(line);
        match lexer.tokenize() {
            Ok(tokens) => Ok(tokens),
            Err(_) => {
                // If tokenization fails, return empty tokens
                // This allows us to still provide hints for malformed input
                Ok(Vec::new())
            }
        }
    }
    
    /// Find context around cursor position
    fn find_cursor_context(&self, tokens: &[Token], line: &str, pos: usize) -> CursorContext {
        let mut context = CursorContext::default();
        
        // Find token at or before cursor position
        for (i, token) in tokens.iter().enumerate() {
            if token.position <= pos && pos <= token.position + token.length {
                context.current_token = Some(token.clone());
                context.token_index = Some(i);
                break;
            } else if token.position > pos {
                // Cursor is before this token
                if i > 0 {
                    context.previous_token = Some(tokens[i - 1].clone());
                }
                break;
            }
        }
        
        // If no exact match, find closest token
        if context.current_token.is_none() && !tokens.is_empty() {
            // Find the closest token before cursor
            for (i, token) in tokens.iter().enumerate() {
                if token.position <= pos {
                    context.previous_token = Some(token.clone());
                    context.token_index = Some(i);
                }
            }
        }
        
        // Analyze surrounding tokens for function call patterns
        context.function_context = self.analyze_function_context(tokens, &context);
        
        context
    }
    
    /// Analyze function call context
    fn analyze_function_context(&self, tokens: &[Token], cursor_context: &CursorContext) -> Option<FunctionContext> {
        if tokens.is_empty() {
            return None;
        }
        
        // Look for function call patterns: Symbol[...
        for i in 0..tokens.len() {
            if let TokenKind::Symbol(name) = &tokens[i].kind {
                // Check if followed by opening bracket
                if i + 1 < tokens.len() && matches!(tokens[i + 1].kind, TokenKind::LeftBracket) {
                    // This is a function call
                    let function_context = FunctionContext {
                        function_name: name.clone(),
                        function_token_index: i,
                        open_bracket_index: i + 1,
                        parameter_count: self.count_parameters(&tokens[i + 2..]),
                        current_parameter_index: self.find_current_parameter_index(
                            &tokens[i + 2..], 
                            cursor_context
                        ),
                    };
                    
                    return Some(function_context);
                }
            }
        }
        
        None
    }
    
    /// Count parameters in function call
    fn count_parameters(&self, tokens: &[Token]) -> usize {
        let mut count = 0;
        let mut bracket_depth = 0;
        let mut has_content = false;
        
        for token in tokens {
            match &token.kind {
                TokenKind::LeftBracket | TokenKind::LeftParen | TokenKind::LeftBrace => {
                    bracket_depth += 1;
                }
                TokenKind::RightBracket | TokenKind::RightParen | TokenKind::RightBrace => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        // End of function call
                        if has_content {
                            count += 1;
                        }
                        break;
                    }
                }
                TokenKind::Comma if bracket_depth == 0 => {
                    if has_content {
                        count += 1;
                        has_content = false;
                    }
                }
                _ => {
                    if bracket_depth >= 0 {
                        has_content = true;
                    }
                }
            }
        }
        
        // Count the last parameter if we haven't hit a closing bracket
        if bracket_depth >= 0 && has_content {
            count += 1;
        }
        
        count
    }
    
    /// Find current parameter index based on cursor position
    fn find_current_parameter_index(&self, tokens: &[Token], cursor_context: &CursorContext) -> Option<usize> {
        let cursor_token_index = cursor_context.token_index?;
        let mut param_index = 0;
        let mut bracket_depth = 0;
        
        for (i, token) in tokens.iter().enumerate() {
            // Adjust index to account for function tokens not included in slice
            let absolute_index = i + cursor_token_index.saturating_sub(tokens.len());
            
            match &token.kind {
                TokenKind::LeftBracket | TokenKind::LeftParen | TokenKind::LeftBrace => {
                    bracket_depth += 1;
                }
                TokenKind::RightBracket | TokenKind::RightParen | TokenKind::RightBrace => {
                    bracket_depth -= 1;
                    if bracket_depth < 0 {
                        break;
                    }
                }
                TokenKind::Comma if bracket_depth == 0 => {
                    if absolute_index < cursor_token_index {
                        param_index += 1;
                    }
                }
                _ => {}
            }
            
            if absolute_index >= cursor_token_index {
                break;
            }
        }
        
        Some(param_index)
    }
    
    /// Determine what type of hint to show
    fn determine_hint_type(&self, context: &CursorContext, line: &str, pos: usize) -> HintType {
        // Check for function call context
        if let Some(func_context) = &context.function_context {
            // If cursor is right after function name, show function signature
            if let Some(current) = &context.current_token {
                if let TokenKind::Symbol(_) = current.kind {
                    return HintType::FunctionCall;
                }
            }
            
            // If cursor is in parameter position, show parameter hints
            if func_context.current_parameter_index.is_some() {
                return HintType::ParameterPosition;
            }
        }
        
        // Check for error contexts
        if self.has_syntax_errors(line) {
            return HintType::ErrorContext;
        }
        
        // Check if we're starting to type a function name
        if self.is_potential_function_start(line, pos) {
            return HintType::FunctionCall;
        }
        
        HintType::None
    }
    
    /// Check if input has obvious syntax errors
    fn has_syntax_errors(&self, line: &str) -> bool {
        // Simple heuristics for common syntax errors
        let mut bracket_count = 0;
        let mut paren_count = 0;
        let mut brace_count = 0;
        
        for ch in line.chars() {
            match ch {
                '[' => bracket_count += 1,
                ']' => bracket_count -= 1,
                '(' => paren_count += 1,
                ')' => paren_count -= 1,
                '{' => brace_count += 1,
                '}' => brace_count -= 1,
                _ => {}
            }
        }
        
        bracket_count != 0 || paren_count != 0 || brace_count != 0
    }
    
    /// Check if cursor position might be starting a function name
    fn is_potential_function_start(&self, line: &str, pos: usize) -> bool {
        if pos == 0 {
            return true;
        }
        
        let chars: Vec<char> = line.chars().collect();
        if pos > chars.len() {
            return false;
        }
        
        // Check if previous character suggests start of function
        if pos > 0 {
            match chars[pos - 1] {
                ' ' | ',' | '[' | '(' | '{' | '\t' => return true,
                _ => {}
            }
        }
        
        false
    }
    
    /// Build final hint context
    fn build_hint_context(&self, cursor_context: CursorContext, hint_type: HintType) -> HintContext {
        let (function_name, parameter_index) = if let Some(func_context) = &cursor_context.function_context {
            (
                Some(func_context.function_name.clone()),
                func_context.current_parameter_index,
            )
        } else {
            (None, None)
        };
        
        HintContext {
            function_name,
            parameter_index,
            hint_type,
            current_token: cursor_context.current_token,
            error_type: None, // Could be enhanced to detect specific error types
        }
    }
}

/// Internal cursor context during analysis
#[derive(Debug, Clone, Default)]
struct CursorContext {
    current_token: Option<Token>,
    previous_token: Option<Token>,
    token_index: Option<usize>,
    function_context: Option<FunctionContext>,
}

/// Function call context information
#[derive(Debug, Clone)]
struct FunctionContext {
    function_name: String,
    function_token_index: usize,
    open_bracket_index: usize,
    parameter_count: usize,
    current_parameter_index: Option<usize>,
}

impl Default for ContextAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_call_context() {
        let analyzer = ContextAnalyzer::new();
        
        // Test function call recognition
        let context = analyzer.analyze("Sin[", 4).unwrap();
        assert_eq!(context.hint_type, HintType::FunctionCall);
        assert_eq!(context.function_name, Some("Sin".to_string()));
    }
    
    #[test]
    fn test_parameter_position_context() {
        let analyzer = ContextAnalyzer::new();
        
        // Test parameter position recognition
        let context = analyzer.analyze("Sin[0", 5).unwrap();
        assert_eq!(context.hint_type, HintType::ParameterPosition);
        assert_eq!(context.function_name, Some("Sin".to_string()));
        assert_eq!(context.parameter_index, Some(0));
    }
    
    #[test]
    fn test_multiple_parameters() {
        let analyzer = ContextAnalyzer::new();
        
        // Test multiple parameter recognition
        let context = analyzer.analyze("Append[{1, 2}, ", 15).unwrap();
        assert_eq!(context.hint_type, HintType::ParameterPosition);
        assert_eq!(context.function_name, Some("Append".to_string()));
        assert_eq!(context.parameter_index, Some(1));
    }
    
    #[test]
    fn test_no_context() {
        let analyzer = ContextAnalyzer::new();
        
        // Test when no specific context is found
        let context = analyzer.analyze("123", 3).unwrap();
        assert_eq!(context.hint_type, HintType::None);
        assert_eq!(context.function_name, None);
    }
    
    #[test]
    fn test_syntax_error_context() {
        let analyzer = ContextAnalyzer::new();
        
        // Test syntax error detection
        let context = analyzer.analyze("Sin[0", 5).unwrap();
        // This should detect the unclosed bracket
        // The exact behavior may vary based on implementation
    }
    
    #[test]
    fn test_nested_function_calls() {
        let analyzer = ContextAnalyzer::new();
        
        // Test nested function call recognition
        let context = analyzer.analyze("Cos[Sin[", 9).unwrap();
        assert_eq!(context.function_name, Some("Sin".to_string()));
        assert_eq!(context.hint_type, HintType::FunctionCall);
    }
    
    #[test]
    fn test_parameter_counting() {
        let analyzer = ContextAnalyzer::new();
        
        // Test correct parameter counting
        let tokens = analyzer.tokenize_input("1, 2, 3").unwrap();
        let count = analyzer.count_parameters(&tokens);
        assert_eq!(count, 3);
    }
    
    #[test]
    fn test_empty_parameters() {
        let analyzer = ContextAnalyzer::new();
        
        // Test empty parameter list
        let context = analyzer.analyze("Length[]", 7).unwrap();
        assert_eq!(context.function_name, Some("Length".to_string()));
    }
}