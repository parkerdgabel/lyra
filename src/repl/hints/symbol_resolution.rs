//! Symbol resolution and tracking for intelligent hints
//!
//! This module provides comprehensive symbol tracking including variables,
//! user-defined functions, and scope management for context-aware hints.

use crate::lexer::{Lexer, TokenKind};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Symbol resolution engine for tracking variables and functions
pub struct SymbolResolver {
    /// Global symbol table
    global_symbols: Arc<RwLock<SymbolTable>>,
    /// Stack of local scopes
    scope_stack: Vec<SymbolTable>,
    /// Function definitions
    function_definitions: HashMap<String, FunctionDefinition>,
    /// Variable assignments and their types
    variable_assignments: HashMap<String, VariableInfo>,
    /// Pattern definitions
    pattern_definitions: HashMap<String, PatternInfo>,
    /// Symbol usage statistics
    usage_stats: HashMap<String, UsageStats>,
}

/// Symbol table for a specific scope
#[derive(Debug, Clone)]
pub struct SymbolTable {
    /// Symbols in this scope
    symbols: HashMap<String, Symbol>,
    /// Parent scope (if any)
    parent: Option<Box<SymbolTable>>,
    /// Scope type (global, function, block)
    scope_type: ScopeType,
}

/// Type of scope
#[derive(Debug, Clone, PartialEq)]
pub enum ScopeType {
    /// Global scope
    Global,
    /// Function scope
    Function(String),
    /// Block scope (e.g., Module, With)
    Block,
    /// Pattern scope
    Pattern,
}

/// Symbol information
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Inferred value type
    pub value_type: Option<String>,
    /// Definition location
    pub definition_location: Option<SourceLocation>,
    /// Usage count
    pub usage_count: usize,
    /// Last used
    pub last_used: Option<std::time::Instant>,
}

/// Type of symbol
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolType {
    /// Variable
    Variable,
    /// User-defined function
    Function,
    /// Pattern variable
    PatternVariable,
    /// Module or context
    Module,
    /// Built-in function
    BuiltinFunction,
}

/// Function definition information
#[derive(Debug, Clone)]
pub struct FunctionDefinition {
    /// Function name
    pub name: String,
    /// Parameter patterns
    pub parameters: Vec<String>,
    /// Function body (if available)
    pub body: Option<String>,
    /// Return type hint
    pub return_type: Option<String>,
    /// Definition source
    pub definition_source: String,
    /// Number of parameters
    pub arity: usize,
}

/// Variable information
#[derive(Debug, Clone)]
pub struct VariableInfo {
    /// Variable name
    pub name: String,
    /// Assigned value representation
    pub value: String,
    /// Inferred type
    pub inferred_type: String,
    /// Assignment location
    pub assignment_location: Option<SourceLocation>,
    /// Is mutable
    pub is_mutable: bool,
}

/// Pattern information
#[derive(Debug, Clone)]
pub struct PatternInfo {
    /// Pattern expression
    pub pattern: String,
    /// Variables captured by this pattern
    pub captured_variables: Vec<String>,
    /// Pattern type (blank, condition, etc.)
    pub pattern_type: PatternType,
    /// Usage examples
    pub examples: Vec<String>,
}

/// Type of pattern
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    /// Blank pattern (x_)
    Blank,
    /// Blank sequence (x__)
    BlankSequence,
    /// Typed blank (x_Integer)
    TypedBlank(String),
    /// Condition pattern (x_?condition)
    Condition,
    /// Named pattern
    Named,
}

/// Source location information
#[derive(Debug, Clone)]
pub struct SourceLocation {
    /// Line number
    pub line: usize,
    /// Column number
    pub column: usize,
    /// Source file (if applicable)
    pub file: Option<String>,
}

/// Usage statistics for symbols
#[derive(Debug, Clone, Default)]
pub struct UsageStats {
    /// Total usage count
    pub total_uses: usize,
    /// First used time
    pub first_used: Option<std::time::Instant>,
    /// Last used time
    pub last_used: Option<std::time::Instant>,
    /// Common usage contexts
    pub contexts: Vec<String>,
}

/// Symbol suggestion result
#[derive(Debug, Clone)]
pub struct SymbolSuggestion {
    /// Suggested symbol name
    pub name: String,
    /// Symbol type
    pub symbol_type: SymbolType,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Why this symbol is suggested
    pub rationale: String,
    /// Example usage
    pub example_usage: Option<String>,
}

/// Symbol resolution result
#[derive(Debug, Clone)]
pub struct SymbolResolutionResult {
    /// Resolved symbol (if found)
    pub symbol: Option<Symbol>,
    /// Suggestions if not found
    pub suggestions: Vec<SymbolSuggestion>,
    /// Scope information
    pub scope_info: ScopeInfo,
    /// Conflicts or warnings
    pub warnings: Vec<String>,
}

/// Information about current scope
#[derive(Debug, Clone)]
pub struct ScopeInfo {
    /// Current scope type
    pub scope_type: ScopeType,
    /// Available symbols in current scope
    pub available_symbols: Vec<String>,
    /// Parent scopes
    pub parent_scopes: Vec<ScopeType>,
}

impl SymbolResolver {
    /// Create a new symbol resolver
    pub fn new() -> Self {
        Self {
            global_symbols: Arc::new(RwLock::new(SymbolTable::new(ScopeType::Global))),
            scope_stack: Vec::new(),
            function_definitions: HashMap::new(),
            variable_assignments: HashMap::new(),
            pattern_definitions: HashMap::new(),
            usage_stats: HashMap::new(),
        }
    }
    
    /// Analyze input and update symbol tables
    pub fn analyze_input(&mut self, input: &str) -> Result<(), String> {
        // Tokenize input to find symbol definitions and usage
        let mut lexer = Lexer::new(input);
        let tokens = lexer.tokenize().map_err(|e| format!("Tokenization error: {:?}", e))?;
        
        // Look for function definitions (f[x_] = expr or f[x_] := expr)
        self.extract_function_definitions(&tokens, input)?;
        
        // Look for variable assignments (x = value)
        self.extract_variable_assignments(&tokens, input)?;
        
        // Look for pattern definitions
        self.extract_pattern_definitions(&tokens, input)?;
        
        // Update usage statistics
        self.update_usage_statistics(&tokens)?;
        
        Ok(())
    }
    
    /// Resolve a symbol in current context
    pub fn resolve_symbol(&self, symbol_name: &str) -> SymbolResolutionResult {
        // Check local scopes first (stack)
        for scope in self.scope_stack.iter().rev() {
            if let Some(symbol) = scope.symbols.get(symbol_name) {
                return SymbolResolutionResult {
                    symbol: Some(symbol.clone()),
                    suggestions: Vec::new(),
                    scope_info: self.get_current_scope_info(),
                    warnings: Vec::new(),
                };
            }
        }
        
        // Check global scope
        if let Ok(global) = self.global_symbols.read() {
            if let Some(symbol) = global.symbols.get(symbol_name) {
                return SymbolResolutionResult {
                    symbol: Some(symbol.clone()),
                    suggestions: Vec::new(),
                    scope_info: self.get_current_scope_info(),
                    warnings: Vec::new(),
                };
            }
        }
        
        // Symbol not found - generate suggestions
        let suggestions = self.generate_symbol_suggestions(symbol_name);
        let warnings = self.check_for_warnings(symbol_name);
        
        SymbolResolutionResult {
            symbol: None,
            suggestions,
            scope_info: self.get_current_scope_info(),
            warnings,
        }
    }
    
    /// Get suggestions for symbol completion
    pub fn get_completion_suggestions(&self, prefix: &str, context: &str) -> Vec<SymbolSuggestion> {
        let mut suggestions = Vec::new();
        
        // Get symbols from all scopes
        let all_symbols = self.collect_all_symbols();
        
        // Filter by prefix and rank by relevance
        for symbol in all_symbols {
            if symbol.name.starts_with(prefix) {
                let confidence = self.calculate_completion_confidence(&symbol, prefix, context);
                if confidence > 0.1 { // Minimum confidence threshold
                    suggestions.push(SymbolSuggestion {
                        name: symbol.name.clone(),
                        symbol_type: symbol.symbol_type.clone(),
                        confidence,
                        rationale: self.generate_completion_rationale(&symbol, context),
                        example_usage: self.generate_example_usage(&symbol),
                    });
                }
            }
        }
        
        // Sort by confidence (highest first)
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to top 10 suggestions
        suggestions.truncate(10);
        
        suggestions
    }
    
    /// Add a variable to current scope
    pub fn add_variable(&mut self, name: String, value_type: Option<String>) {
        let symbol = Symbol {
            name: name.clone(),
            symbol_type: SymbolType::Variable,
            value_type,
            definition_location: None,
            usage_count: 0,
            last_used: None,
        };
        
        if let Some(current_scope) = self.scope_stack.last_mut() {
            current_scope.symbols.insert(name, symbol);
        } else {
            // Add to global scope
            if let Ok(mut global) = self.global_symbols.write() {
                global.symbols.insert(name, symbol);
            }
        }
    }
    
    /// Add a function definition
    pub fn add_function(&mut self, definition: FunctionDefinition) {
        let symbol = Symbol {
            name: definition.name.clone(),
            symbol_type: SymbolType::Function,
            value_type: definition.return_type.clone(),
            definition_location: None,
            usage_count: 0,
            last_used: None,
        };
        
        // Add to current scope
        if let Some(current_scope) = self.scope_stack.last_mut() {
            current_scope.symbols.insert(definition.name.clone(), symbol);
        } else {
            if let Ok(mut global) = self.global_symbols.write() {
                global.symbols.insert(definition.name.clone(), symbol);
            }
        }
        
        // Store detailed function information
        self.function_definitions.insert(definition.name.clone(), definition);
    }
    
    /// Enter a new scope
    pub fn enter_scope(&mut self, scope_type: ScopeType) {
        let new_scope = SymbolTable::new(scope_type);
        self.scope_stack.push(new_scope);
    }
    
    /// Exit current scope
    pub fn exit_scope(&mut self) {
        self.scope_stack.pop();
    }
    
    /// Get all variables in current scope
    pub fn get_variables_in_scope(&self) -> Vec<Symbol> {
        let mut variables = Vec::new();
        
        // Collect from scope stack
        for scope in &self.scope_stack {
            for symbol in scope.symbols.values() {
                if symbol.symbol_type == SymbolType::Variable {
                    variables.push(symbol.clone());
                }
            }
        }
        
        // Collect from global scope
        if let Ok(global) = self.global_symbols.read() {
            for symbol in global.symbols.values() {
                if symbol.symbol_type == SymbolType::Variable {
                    variables.push(symbol.clone());
                }
            }
        }
        
        variables
    }
    
    /// Get all user-defined functions
    pub fn get_user_functions(&self) -> Vec<&FunctionDefinition> {
        self.function_definitions.values().collect()
    }
    
    /// Check for naming conflicts
    pub fn check_naming_conflicts(&self, name: &str) -> Vec<String> {
        let mut conflicts = Vec::new();
        
        // Check if name conflicts with built-in functions
        let builtin_functions = vec![
            "Sin", "Cos", "Tan", "Log", "Exp", "Sqrt", "Abs",
            "Length", "Head", "Tail", "Map", "Apply", "Select",
            "Array", "Dot", "Transpose", "Maximum", "Minimum",
        ];
        
        if builtin_functions.contains(&name) {
            conflicts.push(format!("'{}' conflicts with built-in function", name));
        }
        
        // Check if already defined in current scope
        if let Some(current_scope) = self.scope_stack.last() {
            if current_scope.symbols.contains_key(name) {
                conflicts.push(format!("'{}' already defined in current scope", name));
            }
        }
        
        conflicts
    }
    
    /// Get symbol usage statistics
    pub fn get_symbol_stats(&self, symbol_name: &str) -> Option<&UsageStats> {
        self.usage_stats.get(symbol_name)
    }
    
    /// Record symbol usage
    pub fn record_usage(&mut self, symbol_name: &str, context: &str) {
        let stats = self.usage_stats.entry(symbol_name.to_string()).or_default();
        stats.total_uses += 1;
        stats.last_used = Some(std::time::Instant::now());
        if stats.first_used.is_none() {
            stats.first_used = Some(std::time::Instant::now());
        }
        if !stats.contexts.contains(&context.to_string()) {
            stats.contexts.push(context.to_string());
        }
    }
    
    // Helper methods
    
    fn extract_function_definitions(&mut self, tokens: &[crate::lexer::Token], input: &str) -> Result<(), String> {
        // Look for patterns like: name[params_] = expr or name[params_] := expr
        for (i, token) in tokens.iter().enumerate() {
            if let TokenKind::Symbol(name) = &token.kind {
                // Check if followed by [, parameters, ], and assignment
                if i + 4 < tokens.len() {
                    if let (TokenKind::LeftBracket, TokenKind::Symbol(param), TokenKind::RightBracket, assign_op) = 
                        (&tokens[i+1].kind, &tokens[i+2].kind, &tokens[i+3].kind, &tokens[i+4].kind) {
                        
                        if matches!(assign_op, TokenKind::Set | TokenKind::SetDelayed) {
                            let function_def = FunctionDefinition {
                                name: name.clone(),
                                parameters: vec![param.clone()],
                                body: Some(input.to_string()), // Simplified - would need proper parsing
                                return_type: None,
                                definition_source: input.to_string(),
                                arity: 1,
                            };
                            
                            self.add_function(function_def);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_variable_assignments(&mut self, tokens: &[crate::lexer::Token], input: &str) -> Result<(), String> {
        // Look for patterns like: name = value
        for (i, token) in tokens.iter().enumerate() {
            if let TokenKind::Symbol(name) = &token.kind {
                if i + 1 < tokens.len() {
                    if let TokenKind::Set = tokens[i+1].kind {
                        // Found variable assignment
                        let var_info = VariableInfo {
                            name: name.clone(),
                            value: input.to_string(), // Simplified
                            inferred_type: "Unknown".to_string(), // Would need type inference
                            assignment_location: None,
                            is_mutable: true,
                        };
                        
                        self.variable_assignments.insert(name.clone(), var_info);
                        self.add_variable(name.clone(), None);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn extract_pattern_definitions(&mut self, _tokens: &[crate::lexer::Token], _input: &str) -> Result<(), String> {
        // Pattern extraction would be implemented here
        // This is a placeholder for the pattern matching implementation
        Ok(())
    }
    
    fn update_usage_statistics(&mut self, tokens: &[crate::lexer::Token]) -> Result<(), String> {
        for token in tokens {
            if let TokenKind::Symbol(name) = &token.kind {
                self.record_usage(name, "expression");
            }
        }
        Ok(())
    }
    
    fn generate_symbol_suggestions(&self, symbol_name: &str) -> Vec<SymbolSuggestion> {
        let mut suggestions = Vec::new();
        let all_symbols = self.collect_all_symbols();
        
        // Find similar symbols using string similarity
        for symbol in all_symbols {
            let similarity = self.calculate_string_similarity(symbol_name, &symbol.name);
            if similarity > 0.4 { // Similarity threshold
                suggestions.push(SymbolSuggestion {
                    name: symbol.name.clone(),
                    symbol_type: symbol.symbol_type.clone(),
                    confidence: similarity,
                    rationale: format!("Similar to '{}'", symbol_name),
                    example_usage: self.generate_example_usage(&symbol),
                });
            }
        }
        
        // Sort by confidence
        suggestions.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        suggestions.truncate(5);
        
        suggestions
    }
    
    fn check_for_warnings(&self, _symbol_name: &str) -> Vec<String> {
        // Check for common issues and generate warnings
        Vec::new() // Placeholder
    }
    
    fn get_current_scope_info(&self) -> ScopeInfo {
        let scope_type = self.scope_stack.last()
            .map(|s| s.scope_type.clone())
            .unwrap_or(ScopeType::Global);
        
        let available_symbols = self.collect_all_symbols()
            .into_iter()
            .map(|s| s.name)
            .collect();
        
        let parent_scopes = self.scope_stack.iter()
            .map(|s| s.scope_type.clone())
            .collect();
        
        ScopeInfo {
            scope_type,
            available_symbols,
            parent_scopes,
        }
    }
    
    fn collect_all_symbols(&self) -> Vec<Symbol> {
        let mut symbols = Vec::new();
        
        // Collect from scope stack
        for scope in &self.scope_stack {
            symbols.extend(scope.symbols.values().cloned());
        }
        
        // Collect from global scope
        if let Ok(global) = self.global_symbols.read() {
            symbols.extend(global.symbols.values().cloned());
        }
        
        symbols
    }
    
    fn calculate_completion_confidence(&self, symbol: &Symbol, prefix: &str, context: &str) -> f32 {
        let mut confidence = 0.0;
        
        // Base confidence from prefix match
        if symbol.name.starts_with(prefix) {
            confidence += 0.5;
        }
        
        // Boost for usage frequency
        if let Some(stats) = self.usage_stats.get(&symbol.name) {
            confidence += (stats.total_uses as f32 * 0.01).min(0.3);
        }
        
        // Context relevance
        if context.contains("function") && symbol.symbol_type == SymbolType::Function {
            confidence += 0.2;
        }
        if context.contains("variable") && symbol.symbol_type == SymbolType::Variable {
            confidence += 0.2;
        }
        
        confidence.min(1.0)
    }
    
    fn generate_completion_rationale(&self, symbol: &Symbol, _context: &str) -> String {
        match symbol.symbol_type {
            SymbolType::Variable => format!("Variable '{}'", symbol.name),
            SymbolType::Function => format!("User-defined function '{}'", symbol.name),
            SymbolType::PatternVariable => format!("Pattern variable '{}'", symbol.name),
            SymbolType::Module => format!("Module '{}'", symbol.name),
            SymbolType::BuiltinFunction => format!("Built-in function '{}'", symbol.name),
        }
    }
    
    fn generate_example_usage(&self, symbol: &Symbol) -> Option<String> {
        match symbol.symbol_type {
            SymbolType::Variable => Some(symbol.name.clone()),
            SymbolType::Function => {
                if let Some(func_def) = self.function_definitions.get(&symbol.name) {
                    Some(format!("{}[{}]", symbol.name, func_def.parameters.join(", ")))
                } else {
                    Some(format!("{}[args]", symbol.name))
                }
            },
            _ => None,
        }
    }
    
    fn calculate_string_similarity(&self, a: &str, b: &str) -> f32 {
        // Simple similarity calculation (Jaro-Winkler could be better)
        let a_lower = a.to_lowercase();
        let b_lower = b.to_lowercase();
        
        if a_lower == b_lower {
            return 1.0;
        }
        
        let max_len = a_lower.len().max(b_lower.len());
        if max_len == 0 {
            return 1.0;
        }
        
        let common_chars = a_lower.chars()
            .zip(b_lower.chars())
            .take_while(|(a, b)| a == b)
            .count();
        
        common_chars as f32 / max_len as f32
    }
}

impl SymbolTable {
    /// Create a new symbol table
    pub fn new(scope_type: ScopeType) -> Self {
        Self {
            symbols: HashMap::new(),
            parent: None,
            scope_type,
        }
    }
    
    /// Get symbol from this scope
    pub fn get_symbol(&self, name: &str) -> Option<&Symbol> {
        self.symbols.get(name)
    }
    
    /// Add symbol to this scope
    pub fn add_symbol(&mut self, symbol: Symbol) {
        self.symbols.insert(symbol.name.clone(), symbol);
    }
}

impl Default for SymbolResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_resolver_creation() {
        let resolver = SymbolResolver::new();
        let scope_info = resolver.get_current_scope_info();
        assert_eq!(scope_info.scope_type, ScopeType::Global);
    }
    
    #[test]
    fn test_variable_addition() {
        let mut resolver = SymbolResolver::new();
        resolver.add_variable("x".to_string(), Some("Number".to_string()));
        
        let result = resolver.resolve_symbol("x");
        assert!(result.symbol.is_some());
        assert_eq!(result.symbol.unwrap().name, "x");
    }
    
    #[test]
    fn test_function_definition() {
        let mut resolver = SymbolResolver::new();
        let func_def = FunctionDefinition {
            name: "myFunc".to_string(),
            parameters: vec!["x".to_string()],
            body: Some("x + 1".to_string()),
            return_type: None,
            definition_source: "myFunc[x_] = x + 1".to_string(),
            arity: 1,
        };
        
        resolver.add_function(func_def);
        
        let result = resolver.resolve_symbol("myFunc");
        assert!(result.symbol.is_some());
        assert_eq!(result.symbol.unwrap().symbol_type, SymbolType::Function);
    }
    
    #[test]
    fn test_scope_management() {
        let mut resolver = SymbolResolver::new();
        
        // Add to global scope
        resolver.add_variable("global_var".to_string(), None);
        
        // Enter function scope
        resolver.enter_scope(ScopeType::Function("test".to_string()));
        resolver.add_variable("local_var".to_string(), None);
        
        // Both should be visible
        assert!(resolver.resolve_symbol("global_var").symbol.is_some());
        assert!(resolver.resolve_symbol("local_var").symbol.is_some());
        
        // Exit scope
        resolver.exit_scope();
        
        // Only global should be visible
        assert!(resolver.resolve_symbol("global_var").symbol.is_some());
        assert!(resolver.resolve_symbol("local_var").symbol.is_none());
    }
    
    #[test]
    fn test_completion_suggestions() {
        let mut resolver = SymbolResolver::new();
        resolver.add_variable("variable1".to_string(), None);
        resolver.add_variable("variable2".to_string(), None);
        resolver.add_variable("otherName".to_string(), None);
        
        let suggestions = resolver.get_completion_suggestions("var", "");
        assert_eq!(suggestions.len(), 2); // Should match variable1 and variable2
        
        for suggestion in suggestions {
            assert!(suggestion.name.starts_with("var"));
        }
    }
    
    #[test]
    fn test_naming_conflicts() {
        let resolver = SymbolResolver::new();
        let conflicts = resolver.check_naming_conflicts("Sin");
        assert!(!conflicts.is_empty());
        assert!(conflicts[0].contains("built-in function"));
    }
}