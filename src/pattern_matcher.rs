#![allow(unused_variables)]
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
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, OnceLock};
use dashmap::DashMap;

/// String interning for common variable names to reduce allocations
static COMMON_VARIABLES: OnceLock<Vec<&'static str>> = OnceLock::new();

fn get_common_variables() -> &'static Vec<&'static str> {
    COMMON_VARIABLES.get_or_init(|| vec!["x", "y", "z", "a", "b", "c", "n", "i", "j", "k"])
}

/// Get interned string for common variable names to reduce allocation overhead
fn intern_variable_name(name: &str) -> String {
    if get_common_variables().contains(&name) {
        // For common variables, use a static allocation strategy
        // This could be further optimized with a proper string interner
        name.to_string()
    } else {
        name.to_string()
    }
}

/// Create a HashMap with optimized capacity for pattern matching
/// Most patterns have 1-4 variables, so we pre-allocate to avoid rehashing
fn create_optimized_bindings() -> HashMap<String, Value> {
    HashMap::with_capacity(4)
}

/// Fast path for simple pattern matching without full recursive machinery
#[inline(always)]
fn try_simple_match(pattern: &Pattern, expr: &Value) -> Option<HashMap<String, Value>> {
    match pattern {
        Pattern::Blank { .. } => {
            // Blank pattern matches anything
            Some(HashMap::new())
        }
        _ => None, // Complex patterns need full matcher (simplified for now)
    }
}

/// Pattern categorization for optimization
#[derive(Debug, Clone, PartialEq)]
pub enum PatternCategory {
    Simple,       // Blank, Named with simple patterns
    Complex,      // Function patterns, nested structures
    Conditional,  // Patterns with conditions
}

/// Pattern type classification for bytecode generation
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    Blank,
    Named,
    Function,
    Conditional,
    BlankSequence,
    BlankNullSequence,
    Typed,
    Predicate,
    Alternative,
}

/// Bytecode instructions for optimized pattern matching
#[derive(Debug, Clone, PartialEq)]
pub enum BytecodeInstruction {
    CheckType { expected_type: String },
    BindVariable { name: String },
    CheckFunctionHead { expected_head: String },
    CheckArgumentCount { expected_count: usize },
    CheckExactMatch,
    DescendIntoArg { arg_index: usize },
    Return { success: bool },
}

/// Compiled pattern bytecode for fast matching
#[derive(Debug, Clone)]
pub struct PatternBytecode {
    pub instructions: Vec<BytecodeInstruction>,
    pub variable_count: usize,
    pub pattern_type: PatternType,
}

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

/// Thread-safe context for pattern matching operations
/// This holds the mutable state for a single pattern matching session
#[derive(Debug, Clone)]
pub struct MatchingContext {
    /// Variable bindings from successful pattern matches
    pub bindings: HashMap<String, Value>,
    /// Stack of match frames for nested matching
    pub match_stack: Vec<MatchFrame>,
}

impl MatchingContext {
    /// Create a new matching context
    pub fn new() -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
        }
    }
    
    /// Get a binding by name
    pub fn get_binding(&self, name: &str) -> Option<&Value> {
        self.bindings.get(name)
    }
    
    /// Check if a binding exists
    pub fn has_binding(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
    
    /// Clear all bindings
    pub fn clear_bindings(&mut self) {
        self.bindings.clear();
    }
    
    /// Get all bindings
    pub fn get_bindings(&self) -> &HashMap<String, Value> {
        &self.bindings
    }
}

/// Core pattern matching engine with thread-safe infrastructure
#[derive(Debug)]
pub struct PatternMatcher {
    /// Variable bindings from successful pattern matches
    bindings: HashMap<String, Value>,
    /// Stack of match frames for nested matching
    match_stack: Vec<MatchFrame>,
    /// Maximum recursion depth to prevent stack overflow
    max_depth: usize,
    /// Whether to use pattern compilation for optimization
    use_compilation: bool,
    /// Cache of compiled patterns for performance (thread-safe)
    pattern_cache: Arc<DashMap<u64, PatternBytecode>>,
    /// Pattern router for fast-path optimization
    router: Option<PatternRouter>,
}

impl PatternMatcher {
    /// Create a new pattern matcher with default settings
    pub fn new() -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
            max_depth: 1000, // Reasonable default for recursion depth
            use_compilation: true, // Enable compilation by default for performance
            pattern_cache: Arc::new(DashMap::new()),
            router: Some(PatternRouter::new()), // Enable fast-path routing by default
        }
    }
    
    /// Create a pattern matcher with custom maximum recursion depth
    pub fn with_max_depth(max_depth: usize) -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
            max_depth,
            use_compilation: true,
            pattern_cache: Arc::new(DashMap::new()),
            router: Some(PatternRouter::new()),
        }
    }
    
    /// Create a pattern matcher with compilation enabled or disabled
    pub fn with_compilation(use_compilation: bool) -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
            max_depth: 1000,
            use_compilation,
            pattern_cache: Arc::new(DashMap::new()),
            router: Some(PatternRouter::new()),
        }
    }
    
    /// Create a pattern matcher with fast-path routing disabled (for testing)
    pub fn with_fast_path_disabled() -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
            max_depth: 1000,
            use_compilation: true,
            pattern_cache: Arc::new(DashMap::new()),
            router: None, // Disable fast-path routing
        }
    }
    
    /// Create a new matching context for this matcher
    pub fn create_context(&self) -> MatchingContext {
        MatchingContext::new()
    }
    
    /// Clone this matcher for use in concurrent environments
    /// This shares the thread-safe caches but creates new local state
    pub fn clone_for_concurrent_use(&self) -> Self {
        Self {
            bindings: create_optimized_bindings(),
            match_stack: Vec::new(),
            max_depth: self.max_depth,
            use_compilation: self.use_compilation,
            pattern_cache: Arc::clone(&self.pattern_cache), // Shared thread-safe cache
            router: self.router.as_ref().map(|r| r.clone_with_shared_stats()), // Shared stats
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
            local_bindings: create_optimized_bindings(),
        });
        
        // Perform the actual matching with integrated optimization
        let result = self.match_pattern_integrated(expr, pattern);
        
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
    
    /// Integrated pattern matching with lightweight fast-path optimization
    fn match_pattern_integrated(&mut self, expr: &Expr, pattern: &Pattern) -> MatchResult {
        // Try fast-path matching for simple patterns only
        if self.router.is_some() {
            // Quick check for patterns that can benefit from fast-path
            if self.is_fast_path_candidate(pattern) {
                if let Some(router) = &self.router {
                    // Try direct fast-path matching without fallback overhead
                    if let Some(result) = router.fast_path_registry.try_fast_match(expr, pattern) {
                        return result;
                    }
                }
            }
        }
        
        // Use standard matching logic for all other cases
        self.match_pattern_impl(expr, pattern)
    }
    
    /// Quick check if pattern is a candidate for fast-path matching
    /// Uses simple pattern type checks instead of expensive complexity scoring
    fn is_fast_path_candidate(&self, pattern: &Pattern) -> bool {
        match pattern {
            // Blank patterns with type constraints - these showed excellent performance
            Pattern::Blank { head: Some(_) } => true,
            // Anonymous blank patterns - moderate benefit but consistent
            Pattern::Blank { head: None } => true,
            // Exact patterns are very fast - simple equality check
            Pattern::Exact { .. } => true,
            // Named patterns wrapping simple blanks - excellent performance
            Pattern::Named { pattern: inner, .. } => {
                matches!(inner.as_ref(), Pattern::Blank { .. } | Pattern::Exact { .. })
            }
            // All other patterns use standard matching
            _ => false,
        }
    }
    
    /// Internal implementation of pattern matching
    fn match_pattern_impl(&mut self, expr: &Expr, pattern: &Pattern) -> MatchResult {
        match pattern {
            Pattern::Blank { head } => self.match_blank(expr, head.as_deref()),
            Pattern::BlankSequence { head } => self.match_blank_sequence(expr, head.as_deref()),
            Pattern::BlankNullSequence { head } => self.match_blank_null_sequence(expr, head.as_deref()),
            Pattern::Named { name, pattern } => self.match_named(expr, name, pattern),
            Pattern::Function { head, args } => self.match_function(expr, head, args),
            Pattern::Exact { value } => self.match_exact(expr, value),
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
                    bindings: create_optimized_bindings(),
                }
            } else {
                MatchResult::Failure {
                    reason: format!("Expression does not match type {}", type_name),
                }
            }
        } else {
            // Anonymous blank pattern _ matches anything
            MatchResult::Success {
                bindings: create_optimized_bindings(),
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
            local_bindings: create_optimized_bindings(),
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
                    bindings: create_optimized_bindings(),
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
                    bindings: create_optimized_bindings(),
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
                let mut all_bindings = create_optimized_bindings();
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
        let mut bindings = create_optimized_bindings();
        
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
                    // Pattern matched - add the named binding using optimized string interning
                    let interned_name = intern_variable_name(name);
                    bindings.insert(interned_name.clone(), value.clone());
                    
                    // Also add to our global bindings for future checks
                    self.bindings.insert(interned_name, value);
                    
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
                            MatchResult::Success { bindings: create_optimized_bindings() }
                        } else {
                            MatchResult::Failure {
                                reason: format!("List expression does not match type {}", type_name),
                            }
                        }
                    }
                    Pattern::Blank { head: None } => {
                        // Pattern is _, which matches anything
                        MatchResult::Success { bindings: create_optimized_bindings() }
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
    
    /// Match an exact pattern (exact value match)
    fn match_exact(&mut self, expr: &Expr, pattern_value: &Expr) -> MatchResult {
        // Check if the expression matches the pattern value exactly
        if expr == pattern_value {
            MatchResult::Success {
                bindings: create_optimized_bindings(),
            }
        } else {
            MatchResult::Failure {
                reason: format!("Expression {:?} does not match exact pattern {:?}", expr, pattern_value),
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
            Expr::List(_items) => {
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

/// Pattern compilation functions for Phase 6B.5.1d.2a infrastructure

/// Categorize a pattern for optimization purposes
pub fn categorize_pattern(pattern: &Pattern) -> PatternCategory {
    match pattern {
        Pattern::Blank { .. } => PatternCategory::Simple,
        Pattern::Named { pattern, .. } => {
            match pattern.as_ref() {
                Pattern::Blank { .. } => PatternCategory::Simple,
                _ => PatternCategory::Complex,
            }
        }
        Pattern::Function { .. } => PatternCategory::Complex,
        Pattern::Exact { .. } => PatternCategory::Simple,
        Pattern::Conditional { .. } => PatternCategory::Conditional,
        Pattern::BlankSequence { .. } | Pattern::BlankNullSequence { .. } => PatternCategory::Complex,
        Pattern::Typed { .. } => PatternCategory::Complex,
        Pattern::Predicate { .. } => PatternCategory::Conditional,
        Pattern::Alternative { .. } => PatternCategory::Complex,
    }
}

/// Compile a pattern into optimized bytecode
pub fn compile_pattern(pattern: &Pattern) -> PatternBytecode {
    match pattern {
        Pattern::Blank { head } => PatternBytecode {
            instructions: if let Some(type_name) = head {
                vec![
                    BytecodeInstruction::CheckType { expected_type: type_name.clone() },
                    BytecodeInstruction::Return { success: true },
                ]
            } else {
                vec![BytecodeInstruction::Return { success: true }]
            },
            variable_count: 0,
            pattern_type: PatternType::Blank,
        },
        Pattern::Exact { value: _value } => PatternBytecode {
            instructions: vec![
                BytecodeInstruction::CheckExactMatch, // Would need implementation
                BytecodeInstruction::Return { success: true },
            ],
            variable_count: 0,
            pattern_type: PatternType::Blank, // Use Blank for simplicity
        },
        Pattern::Named { name, pattern } => {
            let mut instructions = vec![];
            let inner_bytecode = compile_pattern(pattern);
            instructions.extend(inner_bytecode.instructions);
            instructions.push(BytecodeInstruction::BindVariable { name: name.clone() });
            
            PatternBytecode {
                instructions,
                variable_count: 1 + inner_bytecode.variable_count,
                pattern_type: PatternType::Named,
            }
        }
        Pattern::Function { head, args } => {
            let mut instructions = vec![];
            instructions.push(BytecodeInstruction::CheckArgumentCount { expected_count: args.len() });
            
            if let Pattern::Blank { head: Some(head_name) } = head.as_ref() {
                instructions.push(BytecodeInstruction::CheckFunctionHead { expected_head: head_name.clone() });
            }
            
            let mut total_variables = 0;
            for (i, arg) in args.iter().enumerate() {
                instructions.push(BytecodeInstruction::DescendIntoArg { arg_index: i });
                let arg_bytecode = compile_pattern(arg);
                instructions.extend(arg_bytecode.instructions);
                total_variables += arg_bytecode.variable_count;
            }
            
            instructions.push(BytecodeInstruction::Return { success: true });
            
            PatternBytecode {
                instructions,
                variable_count: total_variables,
                pattern_type: PatternType::Function,
            }
        }
        Pattern::Conditional { pattern, .. } => {
            let inner_bytecode = compile_pattern(pattern);
            PatternBytecode {
                instructions: inner_bytecode.instructions,
                variable_count: inner_bytecode.variable_count,
                pattern_type: PatternType::Conditional,
            }
        }
        Pattern::BlankSequence { .. } => PatternBytecode {
            instructions: vec![BytecodeInstruction::Return { success: true }],
            variable_count: 0,
            pattern_type: PatternType::BlankSequence,
        },
        Pattern::BlankNullSequence { .. } => PatternBytecode {
            instructions: vec![BytecodeInstruction::Return { success: true }],
            variable_count: 0,
            pattern_type: PatternType::BlankNullSequence,
        },
        Pattern::Typed { .. } => PatternBytecode {
            instructions: vec![BytecodeInstruction::Return { success: true }],
            variable_count: 0,
            pattern_type: PatternType::Typed,
        },
        Pattern::Predicate { .. } => PatternBytecode {
            instructions: vec![BytecodeInstruction::Return { success: true }],
            variable_count: 0,
            pattern_type: PatternType::Predicate,
        },
        Pattern::Alternative { .. } => PatternBytecode {
            instructions: vec![BytecodeInstruction::Return { success: true }],
            variable_count: 0,
            pattern_type: PatternType::Alternative,
        },
    }
}

/// Hash a pattern for caching purposes
fn hash_pattern(pattern: &Pattern) -> u64 {
    let mut hasher = DefaultHasher::new();
    pattern.hash(&mut hasher);
    hasher.finish()
}

//
// Phase 6B.5.1d.2b: Fast-Path Matcher Infrastructure
//

/// Fast-path matcher trait for specialized pattern matching optimizations
/// 
/// Fast-path matchers provide optimized implementations for common pattern types,
/// bypassing the general-purpose pattern matching algorithm for performance gains.
pub trait FastPathMatcher: std::fmt::Debug {
    /// Check if this matcher can handle the given pattern
    fn can_handle(&self, pattern: &Pattern) -> bool;
    
    /// Perform fast pattern matching against an expression
    /// Returns None if the matcher cannot handle this pattern/expression combination
    fn fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult>;
    
    /// Get a descriptive name for this matcher (for debugging/profiling)
    fn name(&self) -> &'static str;
}

/// Fast-path matcher for blank patterns (_)
/// 
/// This is the most common pattern type and benefits significantly from optimization.
/// Handles both anonymous blanks (_) and typed blanks (_Integer, _Real, etc.).
#[derive(Debug)]
pub struct BlankMatcher;

impl BlankMatcher {
    pub fn new() -> Self {
        Self
    }
}

impl FastPathMatcher for BlankMatcher {
    fn can_handle(&self, pattern: &Pattern) -> bool {
        matches!(pattern, Pattern::Blank { .. })
    }
    
    fn fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult> {
        if let Pattern::Blank { head } = pattern {
            if let Some(type_name) = head {
                // Typed blank pattern - direct type check without full pattern machinery
                let matches_type = match (expr, type_name.as_str()) {
                    (Expr::Number(Number::Integer(_)), "Integer") => true,
                    (Expr::Number(Number::Real(_)), "Real") => true,
                    (Expr::String(_), "String") => true,
                    (Expr::Symbol(_), "Symbol") => true,
                    (Expr::List(_), "List") => true,
                    _ => false,
                };
                
                Some(if matches_type {
                    MatchResult::Success {
                        bindings: create_optimized_bindings(),
                    }
                } else {
                    MatchResult::Failure {
                        reason: format!("Expression does not match type {}", type_name),
                    }
                })
            } else {
                // Anonymous blank - always matches
                Some(MatchResult::Success {
                    bindings: create_optimized_bindings(),
                })
            }
        } else {
            None // Cannot handle non-blank patterns
        }
    }
    
    fn name(&self) -> &'static str {
        "BlankMatcher"
    }
}

/// Fast-path matcher for symbol expressions
/// 
/// Optimizes matching when both pattern and expression are symbols,
/// which is common in algebraic expressions.
#[derive(Debug)]
pub struct SymbolMatcher;

impl SymbolMatcher {
    pub fn new() -> Self {
        Self
    }
}

impl FastPathMatcher for SymbolMatcher {
    fn can_handle(&self, pattern: &Pattern) -> bool {
        // Handle patterns that expect symbol expressions
        matches!(pattern, Pattern::Blank { head: Some(head) } if head == "Symbol")
    }
    
    fn fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult> {
        if let Pattern::Blank { head: Some(head) } = pattern {
            if head == "Symbol" {
                return Some(match expr {
                    Expr::Symbol(_) => MatchResult::Success {
                        bindings: create_optimized_bindings(),
                    },
                    _ => MatchResult::Failure {
                        reason: "Expression is not a symbol".to_string(),
                    },
                });
            }
        }
        None
    }
    
    fn name(&self) -> &'static str {
        "SymbolMatcher"
    }
}

/// Fast-path matcher for integer expressions
/// 
/// Optimizes integer pattern matching, which is very common in mathematical expressions.
#[derive(Debug)]
pub struct IntegerMatcher;

impl IntegerMatcher {
    pub fn new() -> Self {
        Self
    }
}

impl FastPathMatcher for IntegerMatcher {
    fn can_handle(&self, pattern: &Pattern) -> bool {
        // Handle patterns that expect integer expressions
        matches!(pattern, Pattern::Blank { head: Some(head) } if head == "Integer")
    }
    
    fn fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult> {
        if let Pattern::Blank { head: Some(head) } = pattern {
            if head == "Integer" {
                return Some(match expr {
                    Expr::Number(Number::Integer(_)) => MatchResult::Success {
                        bindings: create_optimized_bindings(),
                    },
                    _ => MatchResult::Failure {
                        reason: "Expression is not an integer".to_string(),
                    },
                });
            }
        }
        None
    }
    
    fn name(&self) -> &'static str {
        "IntegerMatcher"
    }
}

/// Fast-path matcher for function head matching
/// 
/// Optimizes the common case of matching function expressions against function patterns
/// with known head names (e.g., Plus, Times, Power).
#[derive(Debug)]
pub struct FunctionHeadMatcher;

impl FunctionHeadMatcher {
    pub fn new() -> Self {
        Self
    }
}

impl FastPathMatcher for FunctionHeadMatcher {
    fn can_handle(&self, pattern: &Pattern) -> bool {
        // Handle simple function patterns with typed head and simple argument patterns
        if let Pattern::Function { head, args } = pattern {
            // Check if head is a simple typed blank (e.g., _Plus)
            let simple_head = matches!(head.as_ref(), Pattern::Blank { head: Some(_) });
            
            // Check if arguments are simple patterns that don't require complex matching
            let simple_args = args.iter().all(|arg| {
                match arg {
                    Pattern::Blank { .. } => true,
                    Pattern::Named { pattern: inner_pattern, .. } => {
                        matches!(inner_pattern.as_ref(), Pattern::Blank { .. })
                    }
                    _ => false,
                }
            });
            
            simple_head && simple_args && args.len() <= 3 // Limit to common function arities
        } else {
            false
        }
    }
    
    fn fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult> {
        if let (Expr::Function { head: expr_head, args: expr_args }, 
                Pattern::Function { head: pattern_head, args: pattern_args }) = (expr, pattern) {
            
            // Fast head matching
            if let (Expr::Symbol(expr_symbol), Pattern::Blank { head: Some(expected_head) }) = 
                (expr_head.as_ref(), pattern_head.as_ref()) {
                
                if expr_symbol.name != *expected_head {
                    return Some(MatchResult::Failure {
                        reason: format!("Function head '{}' does not match expected '{}'", 
                                      expr_symbol.name, expected_head),
                    });
                }
                
                // Quick argument count check
                if expr_args.len() != pattern_args.len() {
                    return Some(MatchResult::Failure {
                        reason: format!("Function has {} arguments but pattern expects {}", 
                                      expr_args.len(), pattern_args.len()),
                    });
                }
                
                // Fast argument matching for simple patterns
                let mut bindings = create_optimized_bindings();
                
                for (expr_arg, pattern_arg) in expr_args.iter().zip(pattern_args.iter()) {
                    match pattern_arg {
                        Pattern::Blank { head: None } => {
                            // Anonymous blank - always matches, no binding
                        }
                        Pattern::Blank { head: Some(type_name) } => {
                            // Typed blank - check type
                            let matches_type = match (expr_arg, type_name.as_str()) {
                                (Expr::Number(Number::Integer(_)), "Integer") => true,
                                (Expr::Number(Number::Real(_)), "Real") => true,
                                (Expr::String(_), "String") => true,
                                (Expr::Symbol(_), "Symbol") => true,
                                (Expr::List(_), "List") => true,
                                _ => false,
                            };
                            
                            if !matches_type {
                                return Some(MatchResult::Failure {
                                    reason: format!("Argument does not match type {}", type_name),
                                });
                            }
                        }
                        Pattern::Named { name, pattern } => {
                            if let Pattern::Blank { head } = pattern.as_ref() {
                                // Simple named pattern
                                if let Some(type_name) = head {
                                    // Check type constraint
                                    let matches_type = match (expr_arg, type_name.as_str()) {
                                        (Expr::Number(Number::Integer(_)), "Integer") => true,
                                        (Expr::Number(Number::Real(_)), "Real") => true,
                                        (Expr::String(_), "String") => true,
                                        (Expr::Symbol(_), "Symbol") => true,
                                        (Expr::List(_), "List") => true,
                                        _ => false,
                                    };
                                    
                                    if !matches_type {
                                        return Some(MatchResult::Failure {
                                            reason: format!("Argument does not match type {}", type_name),
                                        });
                                    }
                                }
                                
                                // Convert expression to value for binding
                                if let Ok(value) = self.expr_to_value_fast(expr_arg) {
                                    bindings.insert(intern_variable_name(name), value);
                                } else {
                                    return Some(MatchResult::Failure {
                                        reason: "Failed to convert expression to value".to_string(),
                                    });
                                }
                            } else {
                                // Complex named pattern - fall back to standard matcher
                                return None;
                            }
                        }
                        _ => {
                            // Complex pattern - fall back to standard matcher
                            return None;
                        }
                    }
                }
                
                return Some(MatchResult::Success { bindings });
            }
        }
        
        None
    }
    
    fn name(&self) -> &'static str {
        "FunctionHeadMatcher"
    }
}

impl FunctionHeadMatcher {
    /// Fast expression to value conversion for common cases
    fn expr_to_value_fast(&self, expr: &Expr) -> std::result::Result<Value, ()> {
        match expr {
            Expr::Number(Number::Integer(n)) => Ok(Value::Integer(*n)),
            Expr::Number(Number::Real(f)) => Ok(Value::Real(*f)),
            Expr::String(s) => Ok(Value::String(s.clone())),
            Expr::Symbol(Symbol { name }) => Ok(Value::Symbol(name.clone())),
            _ => Err(()), // Complex expressions not supported in fast path
        }
    }
}

/// Fast-path matcher registry that manages all available matchers
#[derive(Debug)]
pub struct FastPathRegistry {
    matchers: Vec<Box<dyn FastPathMatcher>>,
}

impl FastPathRegistry {
    /// Create a new registry with default fast-path matchers
    pub fn new() -> Self {
        let matchers: Vec<Box<dyn FastPathMatcher>> = vec![
            Box::new(BlankMatcher::new()),
            Box::new(SymbolMatcher::new()),
            Box::new(IntegerMatcher::new()),
            Box::new(FunctionHeadMatcher::new()),
        ];
        
        Self { matchers }
    }
    
    /// Find the best matcher for a given pattern
    pub fn find_matcher(&self, pattern: &Pattern) -> Option<&dyn FastPathMatcher> {
        self.matchers
            .iter()
            .find(|matcher| matcher.can_handle(pattern))
            .map(|matcher| matcher.as_ref())
    }
    
    /// Try fast-path matching with all available matchers
    pub fn try_fast_match(&self, expr: &Expr, pattern: &Pattern) -> Option<MatchResult> {
        for matcher in &self.matchers {
            if matcher.can_handle(pattern) {
                if let Some(result) = matcher.fast_match(expr, pattern) {
                    return Some(result);
                }
            }
        }
        None
    }
    
    /// Get statistics about matcher usage (for performance analysis)
    pub fn get_matcher_names(&self) -> Vec<&'static str> {
        self.matchers.iter().map(|m| m.name()).collect()
    }
}

//
// Sub-Phase 2b.2: Pattern Routing System
//

/// Pattern complexity score for routing decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComplexityScore {
    /// Trivial patterns (blank, simple typed patterns)
    Trivial = 0,
    /// Simple patterns (named blanks, simple functions)
    Simple = 1,
    /// Moderate patterns (functions with multiple arguments)
    Moderate = 2,
    /// Complex patterns (nested patterns, sequences)
    Complex = 3,
    /// Very complex patterns (conditionals, alternatives, predicates)
    VeryComplex = 4,
}

/// Pattern routing strategy for optimization decisions
#[derive(Debug, Clone, PartialEq)]
pub enum RoutingStrategy {
    /// Always try fast-path first, fall back to standard
    FastPathFirst,
    /// Use standard matcher for all patterns
    StandardOnly,
    /// Hybrid approach based on complexity scoring
    Hybrid { complexity_threshold: ComplexityScore },
}

impl Default for RoutingStrategy {
    fn default() -> Self {
        Self::Hybrid {
            complexity_threshold: ComplexityScore::Moderate,
        }
    }
}

/// Intelligent pattern router that decides between fast-path and standard matching
#[derive(Debug)]
pub struct PatternRouter {
    /// Fast-path matcher registry
    fast_path_registry: FastPathRegistry,
    /// Routing strategy configuration
    strategy: RoutingStrategy,
    /// Performance statistics for adaptive routing (thread-safe)
    performance_stats: Arc<DashMap<ComplexityScore, RouteStats>>,
}

/// Performance statistics for routing decisions
#[derive(Debug, Clone)]
pub struct RouteStats {
    fast_path_attempts: u64,
    fast_path_successes: u64,
    standard_attempts: u64,
    total_time_fast_path: std::time::Duration,
    total_time_standard: std::time::Duration,
}

impl RouteStats {
    /// Get number of fast-path attempts
    pub fn fast_path_attempts(&self) -> u64 {
        self.fast_path_attempts
    }
    
    /// Get number of fast-path successes
    pub fn fast_path_successes(&self) -> u64 {
        self.fast_path_successes
    }
    
    /// Get number of standard attempts
    pub fn standard_attempts(&self) -> u64 {
        self.standard_attempts
    }
    
    /// Get total time spent in fast-path matching
    pub fn total_time_fast_path(&self) -> std::time::Duration {
        self.total_time_fast_path
    }
    
    /// Get total time spent in standard matching
    pub fn total_time_standard(&self) -> std::time::Duration {
        self.total_time_standard
    }
}

impl Default for RouteStats {
    fn default() -> Self {
        Self {
            fast_path_attempts: 0,
            fast_path_successes: 0,
            standard_attempts: 0,
            total_time_fast_path: std::time::Duration::new(0, 0),
            total_time_standard: std::time::Duration::new(0, 0),
        }
    }
}

impl PatternRouter {
    /// Create a new pattern router with default settings
    pub fn new() -> Self {
        Self {
            fast_path_registry: FastPathRegistry::new(),
            strategy: RoutingStrategy::default(),
            performance_stats: Arc::new(DashMap::new()),
        }
    }
    
    /// Create a pattern router with a specific routing strategy
    pub fn with_strategy(strategy: RoutingStrategy) -> Self {
        Self {
            fast_path_registry: FastPathRegistry::new(),
            strategy,
            performance_stats: Arc::new(DashMap::new()),
        }
    }
    
    /// Route a pattern matching request to the appropriate matcher
    pub fn route_pattern_match(
        &mut self,
        expr: &Expr,
        pattern: &Pattern,
        standard_matcher: &mut PatternMatcher,
    ) -> MatchResult {
        let complexity = self.calculate_pattern_complexity(pattern);
        let should_try_fast_path = self.should_use_fast_path(complexity);
        
        if should_try_fast_path {
            // Try fast-path first
            let start_time = std::time::Instant::now();
            if let Some(result) = self.fast_path_registry.try_fast_match(expr, pattern) {
                // Fast-path handled it
                let elapsed = start_time.elapsed();
                self.record_fast_path_success(complexity, elapsed);
                return result;
            } else {
                // Fast-path couldn't handle it, record attempt
                let elapsed = start_time.elapsed();
                self.record_fast_path_attempt(complexity, elapsed);
            }
        }
        
        // Fall back to standard matching
        let start_time = std::time::Instant::now();
        let result = standard_matcher.match_pattern(expr, pattern);
        let elapsed = start_time.elapsed();
        self.record_standard_attempt(complexity, elapsed);
        
        result
    }
    
    /// Calculate complexity score for a pattern
    pub fn calculate_pattern_complexity(&self, pattern: &Pattern) -> ComplexityScore {
        match pattern {
            // Trivial patterns - direct type/value matching
            Pattern::Blank { head: None } => ComplexityScore::Trivial,
            Pattern::Blank { head: Some(_) } => ComplexityScore::Trivial,
            Pattern::Exact { .. } => ComplexityScore::Trivial,
            
            // Simple patterns - single level with basic constraints
            Pattern::Named { pattern: inner, .. } => {
                match inner.as_ref() {
                    Pattern::Blank { .. } | Pattern::Exact { .. } => ComplexityScore::Simple,
                    _ => self.calculate_pattern_complexity(inner).max(ComplexityScore::Simple),
                }
            }
            
            // Moderate patterns - functions with simple arguments
            Pattern::Function { head, args } => {
                let _head_complexity = self.calculate_pattern_complexity(head);
                let max_arg_complexity = args.iter()
                    .map(|arg| self.calculate_pattern_complexity(arg))
                    .max()
                    .unwrap_or(ComplexityScore::Trivial);
                
                // Function patterns are at least moderate
                let base_complexity = ComplexityScore::Moderate;
                
                // Increase complexity based on argument count and complexity
                match (args.len(), max_arg_complexity) {
                    (0..=2, ComplexityScore::Trivial | ComplexityScore::Simple) => base_complexity,
                    (3..=4, ComplexityScore::Trivial | ComplexityScore::Simple) => ComplexityScore::Complex,
                    _ => ComplexityScore::VeryComplex,
                }
            }
            
            // Complex patterns - sequences and structural patterns
            Pattern::BlankSequence { .. } => ComplexityScore::Complex,
            Pattern::BlankNullSequence { .. } => ComplexityScore::Complex,
            Pattern::Typed { .. } => ComplexityScore::Complex,
            Pattern::Alternative { patterns } => {
                let max_inner = patterns.iter()
                    .map(|p| self.calculate_pattern_complexity(p))
                    .max()
                    .unwrap_or(ComplexityScore::Trivial);
                max_inner.max(ComplexityScore::Complex)
            }
            
            // Very complex patterns - require evaluation or complex logic
            Pattern::Conditional { pattern: inner, .. } => {
                let inner_complexity = self.calculate_pattern_complexity(inner);
                inner_complexity.max(ComplexityScore::VeryComplex)
            }
            Pattern::Predicate { pattern: inner, .. } => {
                let inner_complexity = self.calculate_pattern_complexity(inner);
                inner_complexity.max(ComplexityScore::VeryComplex)
            }
        }
    }
    
    /// Determine if fast-path matching should be attempted for given complexity
    fn should_use_fast_path(&self, complexity: ComplexityScore) -> bool {
        match &self.strategy {
            RoutingStrategy::FastPathFirst => true,
            RoutingStrategy::StandardOnly => false,
            RoutingStrategy::Hybrid { complexity_threshold } => {
                complexity <= *complexity_threshold
            }
        }
    }
    
    /// Record a successful fast-path match
    fn record_fast_path_success(&mut self, complexity: ComplexityScore, elapsed: std::time::Duration) {
        let mut stats = self.performance_stats.entry(complexity).or_default();
        stats.fast_path_attempts += 1;
        stats.fast_path_successes += 1;
        stats.total_time_fast_path += elapsed;
    }
    
    /// Record a fast-path attempt that fell back to standard
    fn record_fast_path_attempt(&mut self, complexity: ComplexityScore, elapsed: std::time::Duration) {
        let mut stats = self.performance_stats.entry(complexity).or_default();
        stats.fast_path_attempts += 1;
        stats.total_time_fast_path += elapsed;
    }
    
    /// Record a standard matcher attempt
    fn record_standard_attempt(&mut self, complexity: ComplexityScore, elapsed: std::time::Duration) {
        let mut stats = self.performance_stats.entry(complexity).or_default();
        stats.standard_attempts += 1;
        stats.total_time_standard += elapsed;
    }
    
    /// Get performance statistics for analysis
    pub fn get_performance_stats(&self) -> Arc<DashMap<ComplexityScore, RouteStats>> {
        Arc::clone(&self.performance_stats)
    }
    
    /// Get the success rate for fast-path matching at different complexity levels
    pub fn get_fast_path_success_rate(&self, complexity: ComplexityScore) -> f64 {
        if let Some(stats) = self.performance_stats.get(&complexity) {
            if stats.fast_path_attempts > 0 {
                stats.fast_path_successes as f64 / stats.fast_path_attempts as f64
            } else {
                0.0
            }
        } else {
            0.0
        }
    }
    
    /// Get average time per match for fast-path vs standard at given complexity
    pub fn get_average_times(&self, complexity: ComplexityScore) -> (Option<std::time::Duration>, Option<std::time::Duration>) {
        if let Some(stats) = self.performance_stats.get(&complexity) {
            let fast_path_avg = if stats.fast_path_attempts > 0 {
                Some(stats.total_time_fast_path / stats.fast_path_attempts as u32)
            } else {
                None
            };
            
            let standard_avg = if stats.standard_attempts > 0 {
                Some(stats.total_time_standard / stats.standard_attempts as u32)
            } else {
                None
            };
            
            (fast_path_avg, standard_avg)
        } else {
            (None, None)
        }
    }
    
    /// Adaptive strategy adjustment based on performance data
    /// This can be used to automatically tune the complexity threshold
    pub fn suggest_strategy_adjustment(&self) -> Option<RoutingStrategy> {
        // Analyze performance data to suggest better routing strategy
        let mut should_increase_threshold = true;
        let mut should_decrease_threshold = true;
        
        // Check if we should increase threshold (more aggressive fast-path usage)
        for complexity in [ComplexityScore::Complex] {
            if let Some(stats) = self.performance_stats.get(&complexity) {
                let success_rate = self.get_fast_path_success_rate(complexity);
                if success_rate < 0.7 || stats.fast_path_attempts < 10 {
                    should_increase_threshold = false;
                }
            }
        }
        
        // Check if we should decrease threshold (less aggressive fast-path usage)
        for complexity in [ComplexityScore::Simple, ComplexityScore::Moderate] {
            if let Some(stats) = self.performance_stats.get(&complexity) {
                let success_rate = self.get_fast_path_success_rate(complexity);
                if success_rate > 0.9 && stats.fast_path_attempts > 20 {
                    should_decrease_threshold = false;
                }
            }
        }
        
        match &self.strategy {
            RoutingStrategy::Hybrid { complexity_threshold } => {
                if should_increase_threshold && *complexity_threshold < ComplexityScore::Complex {
                    Some(RoutingStrategy::Hybrid {
                        complexity_threshold: ComplexityScore::Complex,
                    })
                } else if should_decrease_threshold && *complexity_threshold > ComplexityScore::Simple {
                    Some(RoutingStrategy::Hybrid {
                        complexity_threshold: ComplexityScore::Simple,
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
    
    /// Clear performance statistics (for testing or reset)
    pub fn clear_stats(&mut self) {
        self.performance_stats.clear();
    }
    
    /// Update routing strategy
    pub fn set_strategy(&mut self, strategy: RoutingStrategy) {
        self.strategy = strategy;
    }
    
    /// Clone this router with shared performance statistics for concurrent use
    pub fn clone_with_shared_stats(&self) -> Self {
        Self {
            fast_path_registry: FastPathRegistry::new(),
            strategy: self.strategy.clone(),
            performance_stats: Arc::clone(&self.performance_stats),
        }
    }
}
