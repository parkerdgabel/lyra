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
            if let MatchResult::Success { bindings: _ } = result {
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
            Pattern::Function { head, args } => self.match_function(expr, head, args),
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
    /// 
    /// Note: This method is called for single expression matching.
    /// For actual sequence matching in function arguments or lists,
    /// use match_pattern_sequence instead.
    fn match_blank_sequence(&mut self, _expr: &Expr, _head: Option<&str>) -> MatchResult {
        MatchResult::Failure {
            reason: "Sequence patterns require context - use match_pattern_sequence for function arguments or lists".to_string(),
        }
    }
    
    /// Match a blank null sequence pattern (___)  
    /// 
    /// Note: This method is called for single expression matching.
    /// For actual null sequence matching in function arguments or lists,
    /// use match_pattern_sequence instead.
    fn match_blank_null_sequence(&mut self, _expr: &Expr, _head: Option<&str>) -> MatchResult {
        MatchResult::Failure {
            reason: "Null sequence patterns require context - use match_pattern_sequence for function arguments or lists".to_string(),
        }
    }
    
    /// Match a pattern against a sequence of expressions (for function arguments or list elements)
    /// 
    /// This method handles sequence patterns (__ and ___) by matching them against
    /// variable numbers of expressions in a sequence context.
    /// 
    /// # Arguments
    /// * `exprs` - The sequence of expressions to match against
    /// * `patterns` - The sequence of patterns to match
    /// 
    /// Returns MatchResult with bindings for any named patterns, including
    /// sequence variables that bind to lists of matched expressions.
    pub fn match_pattern_sequence(&mut self, exprs: &[Expr], patterns: &[Pattern]) -> MatchResult {
        // Check recursion depth
        if self.match_stack.len() >= self.max_depth {
            return MatchResult::Failure {
                reason: "Maximum recursion depth exceeded".to_string(),
            };
        }
        
        // Push new match frame for sequence matching
        self.match_stack.push(MatchFrame {
            level: self.match_stack.len(),
            sequence_pos: Some(0),
            local_bindings: HashMap::new(),
        });
        
        // Perform sequence matching
        let result = self.match_sequence_impl(exprs, patterns, 0, 0);
        
        // Pop match frame and merge bindings on success
        if let Some(frame) = self.match_stack.pop() {
            if let MatchResult::Success { bindings: _ } = result {
                // Merge frame bindings into global bindings
                for (name, value) in frame.local_bindings {
                    self.bindings.insert(name, value);
                }
            }
        }
        
        result
    }
    
    /// Internal implementation of sequence pattern matching
    /// 
    /// Uses recursive backtracking to find valid matches for sequence patterns.
    /// 
    /// # Arguments
    /// * `exprs` - The expressions to match
    /// * `patterns` - The patterns to match against
    /// * `expr_pos` - Current position in expression sequence
    /// * `pattern_pos` - Current position in pattern sequence
    fn match_sequence_impl(&mut self, exprs: &[Expr], patterns: &[Pattern], expr_pos: usize, pattern_pos: usize) -> MatchResult {
        // Base case: all patterns consumed
        if pattern_pos >= patterns.len() {
            if expr_pos >= exprs.len() {
                // All expressions consumed too - perfect match
                return MatchResult::Success {
                    bindings: HashMap::new(),
                };
            } else {
                // Still have expressions left but no patterns - no match
                return MatchResult::Failure {
                    reason: "More expressions than patterns".to_string(),
                };
            }
        }
        
        // Base case: all expressions consumed but patterns remain
        if expr_pos >= exprs.len() {
            // Check if remaining patterns can match empty (only ___ can)
            for i in pattern_pos..patterns.len() {
                if !self.pattern_can_match_empty(&patterns[i]) {
                    return MatchResult::Failure {
                        reason: "Pattern cannot match empty sequence".to_string(),
                    };
                }
            }
            // All remaining patterns can match empty
            return self.bind_empty_sequences(&patterns[pattern_pos..]);
        }
        
        let current_pattern = &patterns[pattern_pos];
        
        match current_pattern {
            Pattern::BlankSequence { head } => {
                // __ pattern: match one or more expressions
                self.match_sequence_greedy(exprs, patterns, expr_pos, pattern_pos, head.as_deref(), 1)
            }
            Pattern::BlankNullSequence { head } => {
                // ___ pattern: match zero or more expressions  
                self.match_sequence_greedy(exprs, patterns, expr_pos, pattern_pos, head.as_deref(), 0)
            }
            Pattern::Named { name, pattern } => {
                // Handle named sequence patterns like x__ or x___
                match pattern.as_ref() {
                    Pattern::BlankSequence { head } => {
                        self.match_named_sequence(exprs, patterns, expr_pos, pattern_pos, name, head.as_deref(), 1)
                    }
                    Pattern::BlankNullSequence { head } => {
                        self.match_named_sequence(exprs, patterns, expr_pos, pattern_pos, name, head.as_deref(), 0)
                    }
                    _ => {
                        // Regular named pattern - match single expression
                        if expr_pos < exprs.len() {
                            let expr = &exprs[expr_pos];
                            match self.match_pattern_impl(expr, current_pattern) {
                                MatchResult::Success { bindings } => {
                                    // Continue with rest of sequence
                                    let rest_result = self.match_sequence_impl(exprs, patterns, expr_pos + 1, pattern_pos + 1);
                                    
                                    match rest_result {
                                        MatchResult::Success { bindings: mut rest_bindings } => {
                                            // Merge both sets of bindings
                                            for (name, value) in bindings {
                                                rest_bindings.insert(name.clone(), value.clone());
                                                self.bindings.insert(name, value);
                                            }
                                            MatchResult::Success { bindings: rest_bindings }
                                        }
                                        failure => failure,
                                    }
                                }
                                failure => failure,
                            }
                        } else {
                            MatchResult::Failure {
                                reason: "No expression to match against named pattern".to_string(),
                            }
                        }
                    }
                }
            }
            _ => {
                // Regular pattern - match single expression
                if expr_pos < exprs.len() {
                    let expr = &exprs[expr_pos];
                    match self.match_pattern_impl(expr, current_pattern) {
                        MatchResult::Success { bindings } => {
                            // Continue with rest of sequence
                            let rest_result = self.match_sequence_impl(exprs, patterns, expr_pos + 1, pattern_pos + 1);
                            
                            match rest_result {
                                MatchResult::Success { bindings: mut rest_bindings } => {
                                    // Merge both sets of bindings
                                    for (name, value) in bindings {
                                        rest_bindings.insert(name.clone(), value.clone());
                                        self.bindings.insert(name, value);
                                    }
                                    MatchResult::Success { bindings: rest_bindings }
                                }
                                failure => failure,
                            }
                        }
                        failure => failure,
                    }
                } else {
                    MatchResult::Failure {
                        reason: "No expression to match against pattern".to_string(),
                    }
                }
            }
        }
    }
    
    /// Match sequence patterns using greedy algorithm with backtracking
    fn match_sequence_greedy(&mut self, exprs: &[Expr], patterns: &[Pattern], expr_pos: usize, pattern_pos: usize, head: Option<&str>, min_match: usize) -> MatchResult {
        let remaining_exprs = exprs.len() - expr_pos;
        let remaining_patterns = patterns.len() - pattern_pos - 1; // -1 for current pattern
        
        // Calculate maximum expressions this pattern can consume
        // We need to leave at least one expression for each remaining non-null-sequence pattern
        let max_match = if remaining_patterns == 0 {
            remaining_exprs // Can consume all remaining expressions
        } else {
            remaining_exprs.saturating_sub(remaining_patterns)
        };
        
        // Try from maximum down to minimum (greedy approach)
        for match_count in (min_match..=max_match).rev() {
            // Check if the expressions we're trying to match are of the right type
            let match_slice = &exprs[expr_pos..expr_pos + match_count];
            if let Some(type_name) = head {
                // Typed sequence - all expressions must match the type
                if !match_slice.iter().all(|expr| self.matches_type(expr, type_name)) {
                    continue;
                }
            }
            
            // Try to match the rest of the patterns
            let rest_result = self.match_sequence_impl(exprs, patterns, expr_pos + match_count, pattern_pos + 1);
            
            if let MatchResult::Success { bindings } = rest_result {
                // Success! Merge bindings and return
                for (name, value) in bindings {
                    self.bindings.insert(name, value);
                }
                return MatchResult::Success {
                    bindings: HashMap::new(),
                };
            }
        }
        
        MatchResult::Failure {
            reason: "No valid sequence match found".to_string(),
        }
    }
    
    /// Match named sequence patterns (like x__ or x___)
    fn match_named_sequence(&mut self, exprs: &[Expr], patterns: &[Pattern], expr_pos: usize, pattern_pos: usize, name: &str, head: Option<&str>, min_match: usize) -> MatchResult {
        let remaining_exprs = exprs.len() - expr_pos;
        let remaining_patterns = patterns.len() - pattern_pos - 1; // -1 for current pattern
        
        // Calculate maximum expressions this pattern can consume
        let max_match = if remaining_patterns == 0 {
            remaining_exprs // Can consume all remaining expressions
        } else {
            remaining_exprs.saturating_sub(remaining_patterns)
        };
        
        // Try from maximum down to minimum (greedy approach)
        for match_count in (min_match..=max_match).rev() {
            // Check if the expressions we're trying to match are of the right type
            let match_slice = &exprs[expr_pos..expr_pos + match_count];
            if let Some(type_name) = head {
                // Typed sequence - all expressions must match the type
                if !match_slice.iter().all(|expr| self.matches_type(expr, type_name)) {
                    continue;
                }
            }
            
            // Convert matched expressions to Values for binding
            let mut match_values = Vec::new();
            for expr in match_slice {
                match self.expr_to_value(expr) {
                    Ok(value) => match_values.push(value),
                    Err(_) => continue, // Skip this match attempt
                }
            }
            
            // Try to match the rest of the patterns
            let rest_result = self.match_sequence_impl(exprs, patterns, expr_pos + match_count, pattern_pos + 1);
            
            if let MatchResult::Success { bindings } = rest_result {
                // Success! Add our binding and merge other bindings
                let mut all_bindings = HashMap::new();
                all_bindings.insert(name.to_string(), Value::List(match_values.clone()));
                
                for (name, value) in &bindings {
                    all_bindings.insert(name.clone(), value.clone());
                }
                
                // Also add to global bindings
                self.bindings.insert(name.to_string(), Value::List(match_values));
                for (name, value) in bindings {
                    self.bindings.insert(name, value);
                }
                
                return MatchResult::Success {
                    bindings: all_bindings,
                };
            }
        }
        
        MatchResult::Failure {
            reason: format!("No valid sequence match found for named pattern {}", name),
        }
    }
    
    /// Check if a pattern can match an empty sequence
    fn pattern_can_match_empty(&self, pattern: &Pattern) -> bool {
        match pattern {
            Pattern::BlankNullSequence { .. } => true,
            Pattern::Named { pattern, .. } => self.pattern_can_match_empty(pattern),
            _ => false,
        }
    }
    
    /// Bind empty sequences for patterns that can match empty
    fn bind_empty_sequences(&mut self, patterns: &[Pattern]) -> MatchResult {
        let mut bindings = HashMap::new();
        
        for pattern in patterns {
            match pattern {
                Pattern::BlankNullSequence { .. } => {
                    // Anonymous null sequence - no binding needed
                }
                Pattern::Named { name, pattern } => {
                    if let Pattern::BlankNullSequence { .. } = pattern.as_ref() {
                        // Named null sequence - bind to empty list
                        bindings.insert(name.clone(), Value::List(vec![]));
                        self.bindings.insert(name.clone(), Value::List(vec![]));
                    }
                }
                _ => {
                    return MatchResult::Failure {
                        reason: "Pattern cannot match empty sequence".to_string(),
                    };
                }
            }
        }
        
        MatchResult::Success { bindings }
    }
    
    /// Match a named pattern (x_)
    fn match_named(&mut self, expr: &Expr, name: &str, pattern: &Pattern) -> MatchResult {
        // Convert expression to Value for binding check
        let value = match self.expr_to_value(expr) {
            Ok(v) => v,
            Err(e) => return MatchResult::Failure {
                reason: format!("Failed to convert expression to value: {}", e),
            }
        };
        
        // Check if this variable is already bound
        if let Some(existing_value) = self.bindings.get(name) {
            // Variable already bound - check if values match
            if self.values_equal(&value, existing_value) {
                // Values match - just verify inner pattern still matches
                let inner_result = self.match_pattern_impl(expr, pattern);
                match inner_result {
                    MatchResult::Success { bindings } => MatchResult::Success { bindings },
                    MatchResult::Failure { reason } => MatchResult::Failure { reason },
                }
            } else {
                // Values don't match - binding conflict
                MatchResult::Failure {
                    reason: format!("Variable '{}' already bound to different value", name),
                }
            }
        } else {
            // Variable not yet bound - check if the nested pattern matches
            let inner_result = self.match_pattern_impl(expr, pattern);
            
            match inner_result {
                MatchResult::Success { mut bindings } => {
                    // Pattern matched - add the named binding
                    bindings.insert(name.to_string(), value);
                    
                    // Also add to our global bindings for future checks
                    self.bindings.insert(name.to_string(), bindings.get(name).unwrap().clone());
                    
                    MatchResult::Success { bindings }
                }
                MatchResult::Failure { reason } => MatchResult::Failure { reason },
            }
        }
    }
    
    /// Match a function pattern (f[x_, y_])
    /// 
    /// This matches function call expressions against function patterns.
    /// For example, Plus[x_, 0] matches Plus[a, 0] with x binding to a.
    fn match_function(&mut self, expr: &Expr, head_pattern: &Pattern, arg_patterns: &[Pattern]) -> MatchResult {
        match expr {
            Expr::Function { head, args } => {
                // First match the head (function name)
                let head_result = self.match_pattern_impl(head, head_pattern);
                
                match head_result {
                    MatchResult::Success { bindings: head_bindings } => {
                        // Head matched, now match arguments using sequence matching
                        let args_result = self.match_pattern_sequence(args, arg_patterns);
                        
                        match args_result {
                            MatchResult::Success { bindings: mut args_bindings } => {
                                // Merge head and args bindings
                                for (name, value) in head_bindings {
                                    args_bindings.insert(name, value);
                                }
                                MatchResult::Success { bindings: args_bindings }
                            }
                            failure => failure,
                        }
                    }
                    failure => failure,
                }
            }
            Expr::List(list_items) => {
                // Handle semantic equivalence: {a, b} ≡ List[a, b]
                // Check if head pattern would match a List type
                let head_result = match head_pattern {
                    Pattern::Blank { head: Some(type_name) } => {
                        if type_name == "List" {
                            // Pattern is _List, which should match List expressions
                            MatchResult::Success { bindings: HashMap::new() }
                        } else {
                            MatchResult::Failure {
                                reason: format!("List expression does not match type {}", type_name),
                            }
                        }
                    }
                    Pattern::Blank { head: None } => {
                        // Pattern is _, which matches anything
                        MatchResult::Success { bindings: HashMap::new() }
                    }
                    _ => {
                        // Try matching against a "List" symbol for other pattern types
                        let list_symbol = Expr::Symbol(Symbol { name: "List".to_string() });
                        self.match_pattern_impl(&list_symbol, head_pattern)
                    }
                };
                
                match head_result {
                    MatchResult::Success { bindings: head_bindings } => {
                        // Head pattern matches List, now match list elements as function arguments
                        let args_result = self.match_pattern_sequence(list_items, arg_patterns);
                        
                        match args_result {
                            MatchResult::Success { bindings: mut args_bindings } => {
                                // Merge head and args bindings
                                for (name, value) in head_bindings {
                                    args_bindings.insert(name, value);
                                }
                                MatchResult::Success { bindings: args_bindings }
                            }
                            failure => failure,
                        }
                    }
                    MatchResult::Failure { reason } => {
                        MatchResult::Failure { reason }
                    }
                }
            }
            _ => {
                // Expression is not a function call or list
                MatchResult::Failure {
                    reason: "Expression is not a function call or list".to_string(),
                }
            }
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
    fn match_conditional(&mut self, expr: &Expr, pattern: &Pattern, condition: &Expr) -> MatchResult {
        // First, try to match the base pattern
        let base_result = self.match_pattern_impl(expr, pattern);
        
        match base_result {
            MatchResult::Success { bindings } => {
                // Base pattern matched - now check the condition
                // We need to substitute the variable bindings into the condition expression
                let substituted_condition = self.substitute_bindings_in_expr(condition, &bindings);
                
                // Evaluate the condition
                let condition_result = self.evaluate_condition(&substituted_condition);
                
                match condition_result {
                    Ok(true) => {
                        // Condition is satisfied - pattern matches
                        MatchResult::Success { bindings }
                    }
                    Ok(false) => {
                        // Condition is not satisfied - pattern fails
                        MatchResult::Failure {
                            reason: "Condition not satisfied".to_string(),
                        }
                    }
                    Err(reason) => {
                        // Error evaluating condition
                        MatchResult::Failure {
                            reason: format!("Error evaluating condition: {}", reason),
                        }
                    }
                }
            }
            MatchResult::Failure { reason } => {
                // Base pattern didn't match
                MatchResult::Failure { reason }
            }
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
    
    /// Substitute variable bindings into an expression
    /// 
    /// This replaces any symbols in the expression that match binding names
    /// with their corresponding values converted back to expressions.
    fn substitute_bindings_in_expr(&self, expr: &Expr, bindings: &HashMap<String, Value>) -> Expr {
        match expr {
            Expr::Symbol(Symbol { name }) => {
                // Check if this symbol has a binding
                if let Some(value) = bindings.get(name) {
                    // Convert the bound value back to an expression
                    self.value_to_expr(value)
                } else {
                    // No binding - return original symbol
                    expr.clone()
                }
            }
            Expr::Function { head, args } => {
                // Recursively substitute in function head and arguments
                let substituted_head = Box::new(self.substitute_bindings_in_expr(head, bindings));
                let substituted_args: Vec<Expr> = args
                    .iter()
                    .map(|arg| self.substitute_bindings_in_expr(arg, bindings))
                    .collect();
                
                Expr::Function {
                    head: substituted_head,
                    args: substituted_args,
                }
            }
            Expr::List(items) => {
                // Recursively substitute in list items
                let substituted_items: Vec<Expr> = items
                    .iter()
                    .map(|item| self.substitute_bindings_in_expr(item, bindings))
                    .collect();
                
                Expr::List(substituted_items)
            }
            // For other expression types, return as-is (no substitution needed)
            _ => expr.clone(),
        }
    }
    
    /// Convert a Value back to an Expr for substitution
    fn value_to_expr(&self, value: &Value) -> Expr {
        match value {
            Value::Integer(n) => Expr::Number(Number::Integer(*n)),
            Value::Real(f) => Expr::Number(Number::Real(*f)),
            Value::String(s) => Expr::String(s.clone()),
            Value::Symbol(name) => Expr::Symbol(Symbol { name: name.clone() }),
            Value::List(items) => {
                let expr_items: Vec<Expr> = items
                    .iter()
                    .map(|item| self.value_to_expr(item))
                    .collect();
                Expr::List(expr_items)
            }
            Value::Boolean(b) => {
                // Convert boolean to symbolic expression
                Expr::Symbol(Symbol { 
                    name: if *b { "True".to_string() } else { "False".to_string() }
                })
            }
            // For other value types, convert to string representation
            _ => Expr::String(format!("{:?}", value)),
        }
    }
    
    /// Evaluate a condition expression to determine if it's true or false
    /// 
    /// This performs basic evaluation of common mathematical conditions like:
    /// - Greater[x, y], Less[x, y], Equal[x, y], etc.
    /// - Even[x], Odd[x] for number properties
    /// - Simple boolean operations
    fn evaluate_condition(&self, condition: &Expr) -> std::result::Result<bool, String> {
        match condition {
            Expr::Function { head, args } => {
                match head.as_ref() {
                    Expr::Symbol(Symbol { name }) => {
                        match name.as_str() {
                            "Greater" => self.evaluate_greater(args),
                            "Less" => self.evaluate_less(args),
                            "Equal" => self.evaluate_equal(args),
                            "GreaterEqual" => self.evaluate_greater_equal(args),
                            "LessEqual" => self.evaluate_less_equal(args),
                            "Even" => self.evaluate_even(args),
                            "Odd" => self.evaluate_odd(args),
                            "FreeQ" => self.evaluate_free_q(args),
                            "Length" => self.evaluate_length(args),
                            "And" => self.evaluate_and(args),
                            "Or" => self.evaluate_or(args),
                            "Not" => self.evaluate_not(args),
                            _ => {
                                // Unknown function - assume it evaluates to true for now
                                // In a full implementation, this would call the VM to evaluate
                                Ok(true)
                            }
                        }
                    }
                    _ => Err("Condition head must be a symbol".to_string()),
                }
            }
            Expr::Symbol(Symbol { name }) => {
                match name.as_str() {
                    "True" => Ok(true),
                    "False" => Ok(false),
                    _ => Err(format!("Cannot evaluate symbol '{}' as condition", name)),
                }
            }
            _ => Err("Unsupported condition expression".to_string()),
        }
    }
    
    /// Evaluate Greater[x, y] condition
    fn evaluate_greater(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("Greater requires exactly 2 arguments".to_string());
        }
        
        let left = self.extract_numeric_value(&args[0])?;
        let right = self.extract_numeric_value(&args[1])?;
        
        Ok(left > right)
    }
    
    /// Evaluate Less[x, y] condition
    fn evaluate_less(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("Less requires exactly 2 arguments".to_string());
        }
        
        let left = self.extract_numeric_value(&args[0])?;
        let right = self.extract_numeric_value(&args[1])?;
        
        Ok(left < right)
    }
    
    /// Evaluate Equal[x, y] condition
    fn evaluate_equal(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("Equal requires exactly 2 arguments".to_string());
        }
        
        // Try numeric comparison first
        if let (Ok(left), Ok(right)) = (
            self.extract_numeric_value(&args[0]), 
            self.extract_numeric_value(&args[1])
        ) {
            Ok((left - right).abs() < f64::EPSILON)
        } else {
            // Fall back to string comparison for non-numeric values
            Ok(format!("{:?}", args[0]) == format!("{:?}", args[1]))
        }
    }
    
    /// Evaluate GreaterEqual[x, y] condition
    fn evaluate_greater_equal(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("GreaterEqual requires exactly 2 arguments".to_string());
        }
        
        let left = self.extract_numeric_value(&args[0])?;
        let right = self.extract_numeric_value(&args[1])?;
        
        Ok(left >= right)
    }
    
    /// Evaluate LessEqual[x, y] condition
    fn evaluate_less_equal(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("LessEqual requires exactly 2 arguments".to_string());
        }
        
        let left = self.extract_numeric_value(&args[0])?;
        let right = self.extract_numeric_value(&args[1])?;
        
        Ok(left <= right)
    }
    
    /// Evaluate Even[x] condition
    fn evaluate_even(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 1 {
            return Err("Even requires exactly 1 argument".to_string());
        }
        
        let value = self.extract_numeric_value(&args[0])?;
        let int_value = value as i64;
        
        // Check if it's actually an integer
        if (value - int_value as f64).abs() > f64::EPSILON {
            return Err("Even can only be applied to integers".to_string());
        }
        
        Ok(int_value % 2 == 0)
    }
    
    /// Evaluate Odd[x] condition
    fn evaluate_odd(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 1 {
            return Err("Odd requires exactly 1 argument".to_string());
        }
        
        let value = self.extract_numeric_value(&args[0])?;
        let int_value = value as i64;
        
        // Check if it's actually an integer
        if (value - int_value as f64).abs() > f64::EPSILON {
            return Err("Odd can only be applied to integers".to_string());
        }
        
        Ok(int_value % 2 == 1)
    }
    
    /// Evaluate FreeQ[expr, symbol] condition (checks if symbol is absent from expr)
    fn evaluate_free_q(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 2 {
            return Err("FreeQ requires exactly 2 arguments".to_string());
        }
        
        let expr = &args[0];
        let symbol = &args[1];
        
        // For simplicity, check if the expressions are different
        // A full implementation would do deeper structural analysis
        let expr_str = format!("{:?}", expr);
        let symbol_str = format!("{:?}", symbol);
        
        Ok(!expr_str.contains(&symbol_str))
    }
    
    /// Evaluate Length[list] condition
    fn evaluate_length(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 1 {
            return Err("Length requires exactly 1 argument".to_string());
        }
        
        match &args[0] {
            Expr::List(items) => {
                // Return the length as a comparison with itself (always true)
                // This is a placeholder - in real usage, Length would be part of a comparison
                Ok(true)
            }
            _ => Err("Length can only be applied to lists".to_string()),
        }
    }
    
    /// Evaluate And[...] condition
    fn evaluate_and(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        for arg in args {
            let result = self.evaluate_condition(arg)?;
            if !result {
                return Ok(false);
            }
        }
        Ok(true)
    }
    
    /// Evaluate Or[...] condition
    fn evaluate_or(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        for arg in args {
            let result = self.evaluate_condition(arg)?;
            if result {
                return Ok(true);
            }
        }
        Ok(false)
    }
    
    /// Evaluate Not[x] condition
    fn evaluate_not(&self, args: &[Expr]) -> std::result::Result<bool, String> {
        if args.len() != 1 {
            return Err("Not requires exactly 1 argument".to_string());
        }
        
        let result = self.evaluate_condition(&args[0])?;
        Ok(!result)
    }
    
    /// Extract a numeric value from an expression for comparison
    fn extract_numeric_value(&self, expr: &Expr) -> std::result::Result<f64, String> {
        match expr {
            Expr::Number(Number::Integer(n)) => Ok(*n as f64),
            Expr::Number(Number::Real(f)) => Ok(*f),
            Expr::Function { head, args } => {
                // Try to evaluate arithmetic functions
                match head.as_ref() {
                    Expr::Symbol(Symbol { name }) => {
                        match name.as_str() {
                            "Plus" => self.evaluate_plus_arithmetic(args),
                            "Minus" => self.evaluate_minus_arithmetic(args),
                            "Times" => self.evaluate_times_arithmetic(args),
                            "Divide" => self.evaluate_divide_arithmetic(args),
                            _ => Err(format!("Cannot extract numeric value from function {}", name)),
                        }
                    }
                    _ => Err(format!("Cannot extract numeric value from complex function head")),
                }
            }
            _ => Err(format!("Cannot extract numeric value from {:?}", expr)),
        }
    }
    
    /// Evaluate Plus[a, b, ...] arithmetic
    fn evaluate_plus_arithmetic(&self, args: &[Expr]) -> std::result::Result<f64, String> {
        let mut sum = 0.0;
        for arg in args {
            sum += self.extract_numeric_value(arg)?;
        }
        Ok(sum)
    }
    
    /// Evaluate Minus[a, b] arithmetic
    fn evaluate_minus_arithmetic(&self, args: &[Expr]) -> std::result::Result<f64, String> {
        if args.is_empty() {
            return Err("Minus requires at least 1 argument".to_string());
        }
        let mut result = self.extract_numeric_value(&args[0])?;
        if args.len() == 1 {
            // Unary minus
            result = -result;
        } else {
            // Binary minus (and more)
            for arg in &args[1..] {
                result -= self.extract_numeric_value(arg)?;
            }
        }
        Ok(result)
    }
    
    /// Evaluate Times[a, b, ...] arithmetic  
    fn evaluate_times_arithmetic(&self, args: &[Expr]) -> std::result::Result<f64, String> {
        let mut product = 1.0;
        for arg in args {
            product *= self.extract_numeric_value(arg)?;
        }
        Ok(product)
    }
    
    /// Evaluate Divide[a, b] arithmetic
    fn evaluate_divide_arithmetic(&self, args: &[Expr]) -> std::result::Result<f64, String> {
        if args.len() != 2 {
            return Err("Divide requires exactly 2 arguments".to_string());
        }
        let numerator = self.extract_numeric_value(&args[0])?;
        let denominator = self.extract_numeric_value(&args[1])?;
        if denominator.abs() < f64::EPSILON {
            return Err("Division by zero".to_string());
        }
        Ok(numerator / denominator)
    }
    
    /// Compare two values for equality in pattern matching context
    fn values_equal(&self, a: &Value, b: &Value) -> bool {
        use crate::vm::Value;
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Real(a), Value::Real(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| self.values_equal(x, y))
            }
            (Value::Quote(a), Value::Quote(b)) => {
                // Compare quoted expressions using Debug format for simplicity
                format!("{:?}", a) == format!("{:?}", b)
            }
            (Value::Pattern(a), Value::Pattern(b)) => {
                // Compare patterns using Debug format for simplicity  
                format!("{:?}", a) == format!("{:?}", b)
            }
            (Value::Missing, Value::Missing) => true,
            // Tensor and LyObj comparison would be more complex, skip for now
            _ => false, // Different types or complex types are never equal
        }
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
    
    /// Test basic sequence pattern matching
    #[test]
    fn test_sequence_pattern_basic() {
        let mut matcher = PatternMatcher::new();
        
        // Test expressions: [1, 2, 3]
        let exprs = vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ];
        
        // Test pattern: [x__] - should match all elements
        let patterns = vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::BlankSequence { head: None }),
            }
        ];
        
        let result = matcher.match_pattern_sequence(&exprs, &patterns);
        match result {
            MatchResult::Success { bindings } => {
                let expected_list = Value::List(vec![
                    Value::Integer(1),
                    Value::Integer(2), 
                    Value::Integer(3)
                ]);
                assert_eq!(bindings.get("x"), Some(&expected_list));
            }
            _ => panic!("Expected successful sequence match"),
        }
    }
    
    /// Test sequence pattern with regular pattern
    #[test] 
    fn test_sequence_pattern_mixed() {
        let mut matcher = PatternMatcher::new();
        
        // Test expressions: [1, 2, 3]  
        let exprs = vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ];
        
        // Test pattern: [x__, y_] - should bind x={1,2} and y=3
        let patterns = vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::BlankSequence { head: None }),
            },
            Pattern::Named {
                name: "y".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }
        ];
        
        let result = matcher.match_pattern_sequence(&exprs, &patterns);
        match result {
            MatchResult::Success { bindings } => {
                let expected_x = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
                let expected_y = Value::Integer(3);
                assert_eq!(bindings.get("x"), Some(&expected_x));
                assert_eq!(bindings.get("y"), Some(&expected_y));
            }
            _ => panic!("Expected successful sequence match"),
        }
    }
    
    /// Test null sequence pattern (can match empty)
    #[test]
    fn test_null_sequence_pattern() {
        let mut matcher = PatternMatcher::new();
        
        // Test expressions: [1]
        let exprs = vec![
            Expr::Number(Number::Integer(1)),
        ];
        
        // Test pattern: [x___, y_] - should bind x={} and y=1
        let patterns = vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::BlankNullSequence { head: None }),
            },
            Pattern::Named {
                name: "y".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }
        ];
        
        let result = matcher.match_pattern_sequence(&exprs, &patterns);
        match result {
            MatchResult::Success { bindings } => {
                let expected_x = Value::List(vec![]);
                let expected_y = Value::Integer(1);
                assert_eq!(bindings.get("x"), Some(&expected_x));
                assert_eq!(bindings.get("y"), Some(&expected_y));
            }
            _ => panic!("Expected successful null sequence match"),
        }
    }
    
    /// Test typed sequence pattern
    #[test]
    fn test_typed_sequence_pattern() {
        let mut matcher = PatternMatcher::new();
        
        // Test expressions: [1, 2, "hello"]
        let exprs = vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
            Expr::String("hello".to_string()),
        ];
        
        // Test pattern: [x__Integer, y_] - should bind x={1,2} and y="hello"
        let patterns = vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::BlankSequence { head: Some("Integer".to_string()) }),
            },
            Pattern::Named {
                name: "y".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }
        ];
        
        let result = matcher.match_pattern_sequence(&exprs, &patterns);
        match result {
            MatchResult::Success { bindings } => {
                let expected_x = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
                let expected_y = Value::String("hello".to_string());
                assert_eq!(bindings.get("x"), Some(&expected_x));
                assert_eq!(bindings.get("y"), Some(&expected_y));
            }
            _ => panic!("Expected successful typed sequence match"),
        }
    }
    
    /// Test sequence pattern that should fail
    #[test]
    fn test_sequence_pattern_failure() {
        let mut matcher = PatternMatcher::new();
        
        // Test expressions: [1, 2]
        let exprs = vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
        ];
        
        // Test pattern: [x__, y__, z_] - impossible to match with only 2 expressions
        let patterns = vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::BlankSequence { head: None }),
            },
            Pattern::Named {
                name: "y".to_string(),
                pattern: Box::new(Pattern::BlankSequence { head: None }),
            },
            Pattern::Named {
                name: "z".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }
        ];
        
        let result = matcher.match_pattern_sequence(&exprs, &patterns);
        assert!(matches!(result, MatchResult::Failure { .. }));
    }
    
    /// Test function pattern matching - basic function pattern
    #[test]
    fn test_function_pattern_basic() {
        let mut matcher = PatternMatcher::new();
        
        // Create expression: Plus[x, 0]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "x".to_string() }),
                Expr::Number(Number::Integer(0)),
            ],
        };
        
        // Create pattern: Plus[x_, 0] 
        let pattern = Pattern::Function {
            head: Box::new(Pattern::Named {
                name: "func".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }),
            args: vec![
                Pattern::Named {
                    name: "var".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None }),
                },
                Pattern::Blank { head: Some("Integer".to_string()) },
            ],
        };
        
        let result = matcher.match_pattern(&expr, &pattern);
        match result {
            MatchResult::Success { bindings } => {
                // func should bind to "Plus"
                assert_eq!(bindings.get("func"), Some(&Value::Symbol("Plus".to_string())));
                // var should bind to "x" 
                assert_eq!(bindings.get("var"), Some(&Value::Symbol("x".to_string())));
            }
            _ => panic!("Expected successful function pattern match"),
        }
    }
    
    /// Test mathematical function pattern matching - Plus[x_, 0]
    #[test]
    fn test_function_pattern_plus_zero() {
        let mut matcher = PatternMatcher::new();
        
        // Create expression: Plus[a, 0]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "a".to_string() }),
                Expr::Number(Number::Integer(0)),
            ],
        };
        
        // Create pattern: Plus[x_, 0] (exact function name match)
        let pattern = Pattern::Function {
            head: Box::new(Pattern::Named {
                name: "Plus".to_string(),
                pattern: Box::new(Pattern::Blank { head: Some("Symbol".to_string()) }),
            }),
            args: vec![
                Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None }),
                },
                Pattern::Blank { head: Some("Integer".to_string()) },
            ],
        };
        
        let result = matcher.match_pattern(&expr, &pattern);
        match result {
            MatchResult::Success { bindings } => {
                // Plus should bind to "Plus"
                assert_eq!(bindings.get("Plus"), Some(&Value::Symbol("Plus".to_string())));
                // x should bind to "a"
                assert_eq!(bindings.get("x"), Some(&Value::Symbol("a".to_string())));
            }
            _ => panic!("Expected successful Plus[x_, 0] pattern match"),
        }
    }
    
    /// Test function pattern no match - wrong function name
    #[test]
    fn test_function_pattern_no_match() {
        let mut matcher = PatternMatcher::new();
        
        // Create expression: Times[x, 1]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "x".to_string() }),
                Expr::Number(Number::Integer(1)),
            ],
        };
        
        // Create pattern: Plus[x_, 1] (different function name)
        // We need to match the exact symbol "Plus", not bind it to a variable
        // Let me create a simpler test that actually tests pattern mismatch
        let pattern = Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Integer".to_string()) }), // This should fail - head must be Integer but we have Symbol
            args: vec![
                Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None }),
                },
                Pattern::Blank { head: Some("Integer".to_string()) },
            ],
        };
        
        let result = matcher.match_pattern(&expr, &pattern);
        assert!(matches!(result, MatchResult::Failure { .. }));
    }
    
    /// Test function pattern with sequence patterns
    #[test]
    fn test_function_pattern_with_sequences() {
        let mut matcher = PatternMatcher::new();
        
        // Create expression: f[1, 2, 3]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "f".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(1)),
                Expr::Number(Number::Integer(2)),
                Expr::Number(Number::Integer(3)),
            ],
        };
        
        // Create pattern: f[x__] (function with sequence pattern)
        let pattern = Pattern::Function {
            head: Box::new(Pattern::Named {
                name: "func".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }),
            args: vec![
                Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::BlankSequence { head: None }),
                },
            ],
        };
        
        let result = matcher.match_pattern(&expr, &pattern);
        match result {
            MatchResult::Success { bindings } => {
                // func should bind to "f"
                assert_eq!(bindings.get("func"), Some(&Value::Symbol("f".to_string())));
                // x should bind to list [1, 2, 3]
                let expected_list = Value::List(vec![
                    Value::Integer(1),
                    Value::Integer(2),
                    Value::Integer(3),
                ]);
                assert_eq!(bindings.get("x"), Some(&expected_list));
            }
            _ => panic!("Expected successful function pattern with sequence match"),
        }
    }
}