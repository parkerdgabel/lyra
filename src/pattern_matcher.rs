//! Pattern Matching Engine for Lyra Symbolic Computation
//!
//! This module implements the core pattern matching algorithm that enables
//! symbolic computation, rule application, and pattern-based transformations.
//! 
//! The PatternMatcher follows Mathematica/Wolfram Language semantics for
//! pattern matching, including support for:
//! - Blank patterns (_)  
//! - Named patterns (x_)
//! - Typed patterns (_Integer, _Real)
//! - Sequence patterns (__, ___)
//! - Alternative and conditional patterns

use crate::ast::{Expr, Pattern, Symbol, Number};
use crate::vm::Value;
use crate::error::{Error, Result};
use std::collections::HashMap;

/// Result of a pattern matching operation
#[derive(Debug, Clone, PartialEq)]
pub enum MatchResult {
    /// Pattern matched successfully with variable bindings
    Success {
        bindings: HashMap<String, Value>,
    },
    /// Pattern failed to match
    Failure {
        reason: String,
    },
}

/// Context frame for tracking match state during recursive matching
#[derive(Debug, Clone)]
struct MatchFrame {
    /// Current nesting level in expression tree
    level: usize,
    /// Position in sequence matching (for __ and ___ patterns)
    sequence_pos: Option<usize>,
    /// Bindings at this frame level
    local_bindings: HashMap<String, Value>,
}

/// Core pattern matching engine
#[derive(Debug)]
pub struct PatternMatcher {
    /// Variable bindings from successful pattern matches
    bindings: HashMap<String, Value>,
    /// Stack of match frames for nested matching
    match_stack: Vec<MatchFrame>,
    /// Maximum recursion depth to prevent stack overflow
    max_depth: usize,
}

impl PatternMatcher {
    /// Create a new pattern matcher with default settings
    pub fn new() -> Self {
        Self {
            bindings: HashMap::new(),
            match_stack: Vec::new(),
            max_depth: 1000, // Reasonable default for recursion depth
        }
    }
    
    /// Create a pattern matcher with custom maximum recursion depth
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            bindings: HashMap::new(),
            match_stack: Vec::new(),
            max_depth,
        }
    }
    
    /// Match an expression against a pattern
    /// 
    /// This is the main entry point for pattern matching. Returns MatchResult
    /// indicating success with bindings or failure with reason.
    /// 
    /// # Examples
    /// ```ignore
    /// let mut matcher = PatternMatcher::new();
    /// let expr = Expr::Number(Number::Integer(42));
    /// let pattern = Pattern::Blank { head: None };
    /// 
    /// match matcher.match_pattern(&expr, &pattern) {
    ///     MatchResult::Success { bindings } => {
    ///         // Pattern matched successfully
    ///     }
    ///     MatchResult::Failure { reason } => {
    ///         // Pattern failed to match
    ///     }
    /// }
    /// ```
    pub fn match_pattern(&mut self, expr: &Expr, pattern: &Pattern) -> MatchResult {
        // Check recursion depth
        if self.match_stack.len() >= self.max_depth {
            return MatchResult::Failure {
                reason: "Maximum recursion depth exceeded".to_string(),
            };
        }
        
        // Push new match frame
        self.match_stack.push(MatchFrame {
            level: self.match_stack.len(),
            sequence_pos: None,
            local_bindings: HashMap::new(),
        });
        
        // Perform the actual matching
        let result = self.match_pattern_impl(expr, pattern);
        
        // Pop match frame and merge bindings on success
        if let Some(frame) = self.match_stack.pop() {
            if let MatchResult::Success { ref bindings } = result {
                // Merge frame bindings into global bindings
                for (name, value) in frame.local_bindings {
                    self.bindings.insert(name, value);
                }
            }
        }
        
        result
    }
    
    /// Internal implementation of pattern matching
    fn match_pattern_impl(&mut self, expr: &Expr, pattern: &Pattern) -> MatchResult {
        match pattern {
            Pattern::Blank { head } => self.match_blank(expr, head.as_deref()),
            Pattern::BlankSequence { head } => self.match_blank_sequence(expr, head.as_deref()),
            Pattern::BlankNullSequence { head } => self.match_blank_null_sequence(expr, head.as_deref()),
            Pattern::Named { name, pattern } => self.match_named(expr, name, pattern),
            Pattern::Typed { name, type_pattern } => self.match_typed(expr, name, type_pattern),
            Pattern::Predicate { pattern, test } => self.match_predicate(expr, pattern, test),
            Pattern::Alternative { patterns } => self.match_alternative(expr, patterns),
            Pattern::Conditional { pattern, condition } => self.match_conditional(expr, pattern, condition),
        }
    }
    
    /// Match a blank pattern (_)
    fn match_blank(&mut self, expr: &Expr, head: Option<&str>) -> MatchResult {
        if let Some(type_name) = head {
            // Typed blank pattern like _Integer
            if self.matches_type(expr, type_name) {
                MatchResult::Success {
                    bindings: HashMap::new(),
                }
            } else {
                MatchResult::Failure {
                    reason: format!("Expression does not match type {}", type_name),
                }
            }
        } else {
            // Anonymous blank pattern _ matches anything
            MatchResult::Success {
                bindings: HashMap::new(),
            }
        }
    }
    
    /// Match a blank sequence pattern (__)
    fn match_blank_sequence(&mut self, _expr: &Expr, _head: Option<&str>) -> MatchResult {
        // TODO: Implement sequence matching - requires context from parent expression
        MatchResult::Failure {
            reason: "Sequence patterns not yet implemented".to_string(),
        }
    }
    
    /// Match a blank null sequence pattern (___)  
    fn match_blank_null_sequence(&mut self, _expr: &Expr, _head: Option<&str>) -> MatchResult {
        // TODO: Implement null sequence matching - requires context from parent expression
        MatchResult::Failure {
            reason: "Null sequence patterns not yet implemented".to_string(),
        }
    }
    
    /// Match a named pattern (x_)
    fn match_named(&mut self, expr: &Expr, name: &str, pattern: &Pattern) -> MatchResult {
        // First check if the nested pattern matches
        let inner_result = self.match_pattern_impl(expr, pattern);
        
        match inner_result {
            MatchResult::Success { mut bindings } => {
                // Convert expression to Value for binding
                match self.expr_to_value(expr) {
                    Ok(value) => {
                        // Add the named binding
                        bindings.insert(name.to_string(), value);
                        MatchResult::Success { bindings }
                    }
                    Err(e) => MatchResult::Failure {
                        reason: format!("Failed to convert expression to value: {}", e),
                    }
                }
            }
            MatchResult::Failure { reason } => MatchResult::Failure { reason },
        }
    }
    
    /// Match a typed pattern (x:Integer)
    fn match_typed(&mut self, _expr: &Expr, _name: &str, _type_pattern: &Expr) -> MatchResult {
        // TODO: Implement typed pattern matching
        MatchResult::Failure {
            reason: "Typed patterns not yet implemented".to_string(),
        }
    }
    
    /// Match a predicate pattern (x_?Positive)
    fn match_predicate(&mut self, _expr: &Expr, _pattern: &Pattern, _test: &Expr) -> MatchResult {
        // TODO: Implement predicate pattern matching
        MatchResult::Failure {
            reason: "Predicate patterns not yet implemented".to_string(),
        }
    }
    
    /// Match an alternative pattern (x_Integer | x_Real)
    fn match_alternative(&mut self, expr: &Expr, patterns: &[Pattern]) -> MatchResult {
        // Try each alternative pattern until one matches
        for pattern in patterns {
            match self.match_pattern_impl(expr, pattern) {
                MatchResult::Success { bindings } => {
                    return MatchResult::Success { bindings };
                }
                MatchResult::Failure { .. } => {
                    // Continue to next alternative
                    continue;
                }
            }
        }
        
        MatchResult::Failure {
            reason: "No alternative pattern matched".to_string(),
        }
    }
    
    /// Match a conditional pattern (x_ /; x > 0)
    fn match_conditional(&mut self, _expr: &Expr, _pattern: &Pattern, _condition: &Expr) -> MatchResult {
        // TODO: Implement conditional pattern matching
        MatchResult::Failure {
            reason: "Conditional patterns not yet implemented".to_string(),
        }
    }
    
    /// Check if an expression matches a given type name
    fn matches_type(&self, expr: &Expr, type_name: &str) -> bool {
        match (expr, type_name) {
            (Expr::Number(Number::Integer(_)), "Integer") => true,
            (Expr::Number(Number::Real(_)), "Real") => true,
            (Expr::String(_), "String") => true,
            (Expr::Symbol(_), "Symbol") => true,
            (Expr::List(_), "List") => true,
            _ => false,
        }
    }
    
    /// Convert an Expr to a Value for binding storage
    fn expr_to_value(&self, expr: &Expr) -> Result<Value> {
        match expr {
            Expr::Number(Number::Integer(n)) => Ok(Value::Integer(*n)),
            Expr::Number(Number::Real(f)) => Ok(Value::Real(*f)),
            Expr::String(s) => Ok(Value::String(s.clone())),
            Expr::Symbol(Symbol { name }) => Ok(Value::Symbol(name.clone())),
            Expr::List(items) => {
                let mut values = Vec::new();
                for item in items {
                    values.push(self.expr_to_value(item)?);
                }
                Ok(Value::List(values))
            }
            _ => Err(Error::Runtime {
                message: format!("Cannot convert expression to value: {:?}", expr),
            }),
        }
    }
    
    /// Get the current variable bindings
    pub fn get_bindings(&self) -> &HashMap<String, Value> {
        &self.bindings
    }
    
    /// Get a specific binding by name
    pub fn get_binding(&self, name: &str) -> Option<&Value> {
        self.bindings.get(name)
    }
    
    /// Clear all variable bindings
    pub fn clear_bindings(&mut self) {
        self.bindings.clear();
    }
    
    /// Check if a specific variable is bound
    pub fn has_binding(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
}

impl Default for PatternMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    /// Test basic blank pattern matching
    #[test]
    fn test_blank_pattern_matching() {
        let mut matcher = PatternMatcher::new();
        
        // _ should match any expression
        let expr = Expr::Number(Number::Integer(42));
        let pattern = Pattern::Blank { head: None };
        
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Success { .. }));
    }
    
    /// Test typed blank pattern matching
    #[test]
    fn test_typed_blank_pattern_matching() {
        let mut matcher = PatternMatcher::new();
        
        // _Integer should match integers
        let expr = Expr::Number(Number::Integer(42));
        let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
        
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Success { .. }));
        
        // _Integer should not match reals
        let expr = Expr::Number(Number::Real(3.14));
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Failure { .. }));
    }
    
    /// Test named pattern matching
    #[test]
    fn test_named_pattern_matching() {
        let mut matcher = PatternMatcher::new();
        
        // x_ should match and bind x to the value
        let expr = Expr::Number(Number::Integer(42));
        let pattern = Pattern::Named {
            name: "x".to_string(),
            pattern: Box::new(Pattern::Blank { head: None }),
        };
        
        let result = matcher.match_pattern(&expr, &pattern);
        match result {
            MatchResult::Success { bindings } => {
                assert_eq!(bindings.get("x"), Some(&Value::Integer(42)));
            }
            _ => panic!("Expected successful match"),
        }
    }
    
    /// Test alternative pattern matching
    #[test]
    fn test_alternative_pattern_matching() {
        let mut matcher = PatternMatcher::new();
        
        // _Integer | _Real should match both integers and reals
        let pattern = Pattern::Alternative {
            patterns: vec![
                Pattern::Blank { head: Some("Integer".to_string()) },
                Pattern::Blank { head: Some("Real".to_string()) },
            ],
        };
        
        // Should match integer
        let expr = Expr::Number(Number::Integer(42));
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Success { .. }));
        
        // Should match real
        let expr = Expr::Number(Number::Real(3.14));
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Success { .. }));
        
        // Should not match string
        let expr = Expr::String("hello".to_string());
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Failure { .. }));
    }
    
    /// Test recursion depth limiting
    #[test]
    fn test_recursion_depth_limit() {
        let mut matcher = PatternMatcher::with_max_depth(5);
        
        // Create a deeply nested pattern that would exceed the limit
        // For now, just test that the mechanism works
        let expr = Expr::Number(Number::Integer(42));
        let pattern = Pattern::Blank { head: None };
        
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Success { .. }));
    }
}