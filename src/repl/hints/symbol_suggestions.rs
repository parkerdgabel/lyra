//! Context-aware symbol suggestion engine
//!
//! This module provides intelligent symbol suggestions by analyzing the current
//! context and combining information from symbol resolution and pattern matching.

use super::symbol_resolution::{SymbolResolver, SymbolType, SymbolSuggestion};
use super::pattern_matching::{PatternAnalyzer, PatternType};
use crate::lexer::{Lexer, TokenKind};
use std::collections::HashMap;

/// Context-aware symbol suggestion engine
pub struct SymbolSuggestionEngine {
    /// Symbol resolver for tracking variables and functions
    symbol_resolver: SymbolResolver,
    /// Pattern analyzer for pattern-based suggestions
    pattern_analyzer: PatternAnalyzer,
    /// Context analysis cache
    context_cache: HashMap<String, AnalysisContext>,
    /// Suggestion ranking weights
    ranking_weights: SuggestionRankingWeights,
    /// Configuration
    config: SuggestionConfig,
}

/// Configuration for symbol suggestions
#[derive(Debug, Clone)]
pub struct SuggestionConfig {
    /// Maximum number of suggestions to return
    pub max_suggestions: usize,
    /// Minimum confidence threshold
    pub min_confidence: f32,
    /// Enable context-aware ranking
    pub context_aware_ranking: bool,
    /// Enable pattern-based suggestions
    pub pattern_suggestions: bool,
    /// Enable type-based filtering
    pub type_filtering: bool,
    /// Enable usage-based ranking
    pub usage_based_ranking: bool,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 10,
            min_confidence: 0.2,
            context_aware_ranking: true,
            pattern_suggestions: true,
            type_filtering: true,
            usage_based_ranking: true,
        }
    }
}

/// Weights for ranking suggestions
#[derive(Debug, Clone)]
pub struct SuggestionRankingWeights {
    /// Weight for prefix matching
    pub prefix_match: f32,
    /// Weight for usage frequency
    pub usage_frequency: f32,
    /// Weight for context relevance
    pub context_relevance: f32,
    /// Weight for type compatibility
    pub type_compatibility: f32,
    /// Weight for recency of use
    pub recency: f32,
    /// Weight for pattern matching
    pub pattern_match: f32,
}

impl Default for SuggestionRankingWeights {
    fn default() -> Self {
        Self {
            prefix_match: 0.3,
            usage_frequency: 0.2,
            context_relevance: 0.25,
            type_compatibility: 0.15,
            recency: 0.05,
            pattern_match: 0.05,
        }
    }
}

/// Analysis context for suggestions
#[derive(Debug, Clone)]
pub struct AnalysisContext {
    /// Current input being analyzed
    pub input: String,
    /// Cursor position
    pub cursor_position: usize,
    /// Detected context type
    pub context_type: ContextType,
    /// Expected type (if any)
    pub expected_type: Option<String>,
    /// Function being called (if in function call)
    pub function_context: Option<FunctionContext>,
    /// Pattern context (if in pattern)
    pub pattern_context: Option<PatternContext>,
    /// Local variables in scope
    pub local_variables: Vec<String>,
    /// Available functions
    pub available_functions: Vec<String>,
}

/// Type of context being analyzed
#[derive(Debug, Clone, PartialEq)]
pub enum ContextType {
    /// General expression context
    Expression,
    /// Function call context
    FunctionCall,
    /// Function parameter context
    FunctionParameter,
    /// Pattern matching context
    PatternMatch,
    /// Variable assignment context
    VariableAssignment,
    /// Rule construction context
    RuleConstruction,
    /// List/array context
    ListContext,
    /// String context
    StringContext,
}

/// Function call context information
#[derive(Debug, Clone)]
pub struct FunctionContext {
    /// Function name being called
    pub function_name: String,
    /// Current parameter index
    pub parameter_index: usize,
    /// Expected parameter type
    pub expected_parameter_type: Option<String>,
    /// Parameter description
    pub parameter_description: Option<String>,
}

/// Pattern matching context information
#[derive(Debug, Clone)]
pub struct PatternContext {
    /// Type of pattern being constructed
    pub pattern_type: Option<PatternType>,
    /// Pattern being built so far
    pub partial_pattern: String,
    /// Variables already captured
    pub captured_variables: Vec<String>,
    /// Expected completion
    pub expected_completion: Option<String>,
}

/// Enhanced symbol suggestion with context
#[derive(Debug, Clone)]
pub struct EnhancedSymbolSuggestion {
    /// Base symbol suggestion
    pub base_suggestion: SymbolSuggestion,
    /// Context-specific information
    pub context_info: String,
    /// Type compatibility score
    pub type_compatibility: f32,
    /// Usage-based score
    pub usage_score: f32,
    /// Pattern relevance score
    pub pattern_relevance: f32,
    /// Final ranked score
    pub final_score: f32,
    /// Additional details
    pub details: SuggestionDetails,
}

/// Additional details for suggestions
#[derive(Debug, Clone)]
pub struct SuggestionDetails {
    /// Last used time (relative description)
    pub last_used: Option<String>,
    /// Usage frequency description
    pub usage_frequency: Option<String>,
    /// Type information
    pub type_info: Option<String>,
    /// Example usage in current context
    pub context_example: Option<String>,
    /// Related symbols
    pub related_symbols: Vec<String>,
}

/// Suggestion analysis result
#[derive(Debug, Clone)]
pub struct SuggestionAnalysisResult {
    /// Context analysis
    pub context: AnalysisContext,
    /// Generated suggestions
    pub suggestions: Vec<EnhancedSymbolSuggestion>,
    /// Pattern-based suggestions
    pub pattern_suggestions: Vec<String>,
    /// Context-specific hints
    pub context_hints: Vec<String>,
    /// Warnings or issues
    pub warnings: Vec<String>,
}

impl SymbolSuggestionEngine {
    /// Create a new symbol suggestion engine
    pub fn new() -> Self {
        Self {
            symbol_resolver: SymbolResolver::new(),
            pattern_analyzer: PatternAnalyzer::new(),
            context_cache: HashMap::new(),
            ranking_weights: SuggestionRankingWeights::default(),
            config: SuggestionConfig::default(),
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: SuggestionConfig) -> Self {
        Self {
            symbol_resolver: SymbolResolver::new(),
            pattern_analyzer: PatternAnalyzer::new(),
            context_cache: HashMap::new(),
            ranking_weights: SuggestionRankingWeights::default(),
            config,
        }
    }
    
    /// Update symbol information from input
    pub fn update_symbols(&mut self, input: &str) -> Result<(), String> {
        self.symbol_resolver.analyze_input(input)
    }
    
    /// Generate context-aware suggestions
    pub fn get_suggestions(&mut self, input: &str, cursor_pos: usize, prefix: &str) -> SuggestionAnalysisResult {
        // Analyze context
        let context = self.analyze_context(input, cursor_pos);
        
        // Get base symbol suggestions
        let base_suggestions = self.symbol_resolver.get_completion_suggestions(prefix, &context.input);
        
        // Enhance suggestions with context information
        let enhanced_suggestions = self.enhance_suggestions(base_suggestions, &context);
        
        // Get pattern-based suggestions if applicable
        let pattern_suggestions = if self.config.pattern_suggestions {
            self.get_pattern_suggestions(&context)
        } else {
            Vec::new()
        };
        
        // Generate context hints
        let context_hints = self.generate_context_hints(&context);
        
        // Check for warnings
        let warnings = self.check_context_warnings(&context);
        
        SuggestionAnalysisResult {
            context,
            suggestions: enhanced_suggestions,
            pattern_suggestions,
            context_hints,
            warnings,
        }
    }
    
    /// Get suggestions for specific context type
    pub fn get_context_specific_suggestions(&self, context_type: ContextType, additional_info: &str) -> Vec<String> {
        match context_type {
            ContextType::FunctionCall => {
                self.symbol_resolver.get_user_functions()
                    .iter()
                    .map(|f| f.name.clone())
                    .collect()
            },
            ContextType::FunctionParameter => {
                self.get_parameter_suggestions(additional_info)
            },
            ContextType::PatternMatch => {
                self.pattern_analyzer.suggest_pattern_variables(additional_info)
            },
            ContextType::VariableAssignment => {
                self.symbol_resolver.get_variables_in_scope()
                    .iter()
                    .map(|v| v.name.clone())
                    .collect()
            },
            ContextType::ListContext => {
                vec!["Map".to_string(), "Select".to_string(), "Apply".to_string(), "Length".to_string()]
            },
            ContextType::StringContext => {
                vec!["StringJoin".to_string(), "StringLength".to_string(), "StringSplit".to_string()]
            },
            _ => Vec::new(),
        }
    }
    
    /// Get suggestions for variable names
    pub fn suggest_variable_names(&self, context: &str, value_type: Option<&str>) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Context-based suggestions
        if context.contains("count") || context.contains("number") {
            suggestions.extend(vec!["count".to_string(), "n".to_string(), "num".to_string()]);
        }
        
        if context.contains("list") || context.contains("array") {
            suggestions.extend(vec!["list".to_string(), "array".to_string(), "items".to_string()]);
        }
        
        if context.contains("result") || context.contains("output") {
            suggestions.extend(vec!["result".to_string(), "output".to_string(), "value".to_string()]);
        }
        
        // Type-based suggestions
        if let Some(vtype) = value_type {
            match vtype.to_lowercase().as_str() {
                "integer" | "number" => {
                    suggestions.extend(vec!["i".to_string(), "j".to_string(), "k".to_string(), "index".to_string()]);
                },
                "real" | "float" => {
                    suggestions.extend(vec!["x".to_string(), "y".to_string(), "z".to_string(), "value".to_string()]);
                },
                "string" => {
                    suggestions.extend(vec!["text".to_string(), "str".to_string(), "name".to_string()]);
                },
                "list" => {
                    suggestions.extend(vec!["items".to_string(), "elements".to_string(), "data".to_string()]);
                },
                _ => {}
            }
        }
        
        // Common variable names
        suggestions.extend(vec![
            "temp".to_string(), "tmp".to_string(), "aux".to_string(),
            "input".to_string(), "output".to_string(), "result".to_string(),
        ]);
        
        // Remove duplicates and sort
        suggestions.sort();
        suggestions.dedup();
        
        suggestions
    }
    
    /// Get function name suggestions
    pub fn suggest_function_names(&self, intent: &str, return_type: Option<&str>) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Intent-based suggestions
        let intent_lower = intent.to_lowercase();
        
        if intent_lower.contains("math") || intent_lower.contains("calculate") {
            suggestions.extend(vec![
                "calculate".to_string(), "compute".to_string(), "eval".to_string(),
                "process".to_string(), "transform".to_string(),
            ]);
        }
        
        if intent_lower.contains("list") || intent_lower.contains("array") {
            suggestions.extend(vec![
                "process".to_string(), "filter".to_string(), "transform".to_string(),
                "reduce".to_string(), "accumulate".to_string(),
            ]);
        }
        
        if intent_lower.contains("string") || intent_lower.contains("text") {
            suggestions.extend(vec![
                "format".to_string(), "parse".to_string(), "convert".to_string(),
                "clean".to_string(), "normalize".to_string(),
            ]);
        }
        
        // Return type based suggestions
        if let Some(rtype) = return_type {
            match rtype.to_lowercase().as_str() {
                "boolean" | "bool" => {
                    suggestions.extend(vec!["is".to_string(), "check".to_string(), "test".to_string()]);
                },
                "number" | "integer" | "real" => {
                    suggestions.extend(vec!["count".to_string(), "sum".to_string(), "calc".to_string()]);
                },
                "string" => {
                    suggestions.extend(vec!["format".to_string(), "toString".to_string(), "stringify".to_string()]);
                },
                "list" => {
                    suggestions.extend(vec!["collect".to_string(), "gather".to_string(), "build".to_string()]);
                },
                _ => {}
            }
        }
        
        suggestions.sort();
        suggestions.dedup();
        suggestions
    }
    
    /// Record symbol usage for learning
    pub fn record_symbol_usage(&mut self, symbol: &str, context: &str) {
        self.symbol_resolver.record_usage(symbol, context);
    }
    
    /// Get usage statistics for a symbol
    pub fn get_symbol_usage_stats(&self, symbol: &str) -> Option<String> {
        if let Some(stats) = self.symbol_resolver.get_symbol_stats(symbol) {
            Some(format!("Used {} times", stats.total_uses))
        } else {
            None
        }
    }
    
    /// Clear suggestion cache
    pub fn clear_cache(&mut self) {
        self.context_cache.clear();
    }
    
    // Helper methods
    
    fn analyze_context(&mut self, input: &str, cursor_pos: usize) -> AnalysisContext {
        // Check cache first
        let cache_key = format!("{}:{}", input, cursor_pos);
        if let Some(cached) = self.context_cache.get(&cache_key) {
            return cached.clone();
        }
        
        let context_type = self.determine_context_type(input, cursor_pos);
        let expected_type = self.infer_expected_type(input, cursor_pos, &context_type);
        let function_context = self.analyze_function_context(input, cursor_pos);
        let pattern_context = self.analyze_pattern_context(input, cursor_pos);
        
        let context = AnalysisContext {
            input: input.to_string(),
            cursor_position: cursor_pos,
            context_type,
            expected_type,
            function_context,
            pattern_context,
            local_variables: self.symbol_resolver.get_variables_in_scope()
                .iter().map(|v| v.name.clone()).collect(),
            available_functions: self.symbol_resolver.get_user_functions()
                .iter().map(|f| f.name.clone()).collect(),
        };
        
        // Cache the result
        self.context_cache.insert(cache_key, context.clone());
        
        context
    }
    
    fn determine_context_type(&self, input: &str, cursor_pos: usize) -> ContextType {
        // Simple heuristics to determine context type
        let before_cursor = &input[..cursor_pos.min(input.len())];
        
        if before_cursor.contains("[") && !before_cursor.contains("]") {
            if before_cursor.matches("[").count() > before_cursor.matches("]").count() {
                return ContextType::FunctionParameter;
            }
        }
        
        if before_cursor.contains("_") {
            return ContextType::PatternMatch;
        }
        
        if before_cursor.contains("=") && !before_cursor.contains("==") {
            return ContextType::VariableAssignment;
        }
        
        if before_cursor.contains("->") || before_cursor.contains(":>") {
            return ContextType::RuleConstruction;
        }
        
        if before_cursor.contains("{") && !before_cursor.contains("}") {
            return ContextType::ListContext;
        }
        
        if before_cursor.contains("\"") && before_cursor.matches("\"").count() % 2 == 1 {
            return ContextType::StringContext;
        }
        
        // Check if we're in a function call
        if let Ok(lexer) = Lexer::new(before_cursor).tokenize() {
            // Look for pattern: Symbol[
            for (i, token) in lexer.iter().enumerate() {
                if let TokenKind::Symbol(_) = token.kind {
                    if i + 1 < lexer.len() {
                        if let TokenKind::LeftBracket = lexer[i + 1].kind {
                            return ContextType::FunctionCall;
                        }
                    }
                }
            }
        }
        
        ContextType::Expression
    }
    
    fn infer_expected_type(&self, input: &str, cursor_pos: usize, context_type: &ContextType) -> Option<String> {
        match context_type {
            ContextType::FunctionParameter => {
                // Would need function signature lookup
                Some("Any".to_string())
            },
            ContextType::ListContext => Some("Any".to_string()),
            ContextType::StringContext => Some("String".to_string()),
            _ => None,
        }
    }
    
    fn analyze_function_context(&self, input: &str, cursor_pos: usize) -> Option<FunctionContext> {
        // Simplified function context analysis
        let before_cursor = &input[..cursor_pos.min(input.len())];
        
        // Look for function call pattern
        if let Some(bracket_pos) = before_cursor.rfind('[') {
            if let Some(start) = before_cursor[..bracket_pos].rfind(|c: char| !c.is_alphanumeric() && c != '_') {
                let function_name = before_cursor[start + 1..bracket_pos].to_string();
                
                // Count commas to determine parameter index
                let params_part = &before_cursor[bracket_pos + 1..];
                let parameter_index = params_part.matches(',').count();
                
                return Some(FunctionContext {
                    function_name,
                    parameter_index,
                    expected_parameter_type: None, // Would need signature lookup
                    parameter_description: None,
                });
            }
        }
        
        None
    }
    
    fn analyze_pattern_context(&self, input: &str, cursor_pos: usize) -> Option<PatternContext> {
        let before_cursor = &input[..cursor_pos.min(input.len())];
        
        if before_cursor.contains("_") {
            let pattern_analysis = self.pattern_analyzer.analyze_pattern(before_cursor);
            
            return Some(PatternContext {
                pattern_type: pattern_analysis.pattern_type,
                partial_pattern: before_cursor.to_string(),
                captured_variables: pattern_analysis.captured_variables,
                expected_completion: None,
            });
        }
        
        None
    }
    
    fn enhance_suggestions(&self, base_suggestions: Vec<SymbolSuggestion>, context: &AnalysisContext) -> Vec<EnhancedSymbolSuggestion> {
        let mut enhanced = Vec::new();
        
        for suggestion in base_suggestions {
            let type_compatibility = self.calculate_type_compatibility(&suggestion, context);
            let usage_score = self.calculate_usage_score(&suggestion);
            let pattern_relevance = self.calculate_pattern_relevance(&suggestion, context);
            
            let final_score = self.calculate_final_score(
                &suggestion,
                type_compatibility,
                usage_score,
                pattern_relevance,
                context,
            );
            
            if final_score >= self.config.min_confidence {
                enhanced.push(EnhancedSymbolSuggestion {
                    context_info: self.generate_context_info(&suggestion, context),
                    type_compatibility,
                    usage_score,
                    pattern_relevance,
                    final_score,
                    details: self.generate_suggestion_details(&suggestion),
                    base_suggestion: suggestion,
                });
            }
        }
        
        // Sort by final score
        enhanced.sort_by(|a, b| b.final_score.partial_cmp(&a.final_score).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit to max suggestions
        enhanced.truncate(self.config.max_suggestions);
        
        enhanced
    }
    
    fn get_pattern_suggestions(&self, context: &AnalysisContext) -> Vec<String> {
        if let Some(pattern_context) = &context.pattern_context {
            self.pattern_analyzer.suggest_pattern_variables(&pattern_context.partial_pattern)
        } else {
            Vec::new()
        }
    }
    
    fn generate_context_hints(&self, context: &AnalysisContext) -> Vec<String> {
        let mut hints = Vec::new();
        
        match context.context_type {
            ContextType::FunctionCall => {
                hints.push("Type function name and parameters".to_string());
            },
            ContextType::FunctionParameter => {
                if let Some(func_ctx) = &context.function_context {
                    hints.push(format!("Parameter {} for {}", func_ctx.parameter_index + 1, func_ctx.function_name));
                }
            },
            ContextType::PatternMatch => {
                hints.push("Use _ for blank patterns, __ for sequences".to_string());
            },
            ContextType::VariableAssignment => {
                hints.push("Assign value to variable".to_string());
            },
            _ => {}
        }
        
        hints
    }
    
    fn check_context_warnings(&self, context: &AnalysisContext) -> Vec<String> {
        let warnings = Vec::new();
        
        // Check for potential naming conflicts
        if context.context_type == ContextType::VariableAssignment {
            // Would check against builtin functions
        }
        
        warnings
    }
    
    fn get_parameter_suggestions(&self, function_name: &str) -> Vec<String> {
        // Would look up function signature and suggest appropriate parameter types
        match function_name {
            "Sin" | "Cos" | "Tan" => vec!["angle_in_radians".to_string()],
            "Length" => vec!["list_or_array".to_string()],
            "Map" => vec!["function".to_string(), "list".to_string()],
            _ => vec!["parameter".to_string()],
        }
    }
    
    fn calculate_type_compatibility(&self, suggestion: &SymbolSuggestion, context: &AnalysisContext) -> f32 {
        if let Some(expected_type) = &context.expected_type {
            // Simplified type compatibility
            match (&suggestion.symbol_type, expected_type.as_str()) {
                (SymbolType::Function, "Function") => 1.0,
                (SymbolType::Variable, "Any") => 0.8,
                (SymbolType::Variable, _) => 0.6,
                _ => 0.5,
            }
        } else {
            0.5
        }
    }
    
    fn calculate_usage_score(&self, suggestion: &SymbolSuggestion) -> f32 {
        if let Some(stats) = self.symbol_resolver.get_symbol_stats(&suggestion.name) {
            (stats.total_uses as f32 * 0.1).min(1.0)
        } else {
            0.0
        }
    }
    
    fn calculate_pattern_relevance(&self, _suggestion: &SymbolSuggestion, context: &AnalysisContext) -> f32 {
        if context.pattern_context.is_some() {
            0.8 // High relevance in pattern context
        } else {
            0.1
        }
    }
    
    fn calculate_final_score(&self, suggestion: &SymbolSuggestion, type_compat: f32, usage: f32, pattern: f32, _context: &AnalysisContext) -> f32 {
        let weights = &self.ranking_weights;
        
        weights.prefix_match * suggestion.confidence +
        weights.usage_frequency * usage +
        weights.type_compatibility * type_compat +
        weights.pattern_match * pattern
    }
    
    fn generate_context_info(&self, suggestion: &SymbolSuggestion, context: &AnalysisContext) -> String {
        match context.context_type {
            ContextType::FunctionCall => format!("Function: {}", suggestion.name),
            ContextType::VariableAssignment => format!("Variable: {}", suggestion.name),
            ContextType::PatternMatch => format!("Pattern variable: {}", suggestion.name),
            _ => format!("Symbol: {}", suggestion.name),
        }
    }
    
    fn generate_suggestion_details(&self, suggestion: &SymbolSuggestion) -> SuggestionDetails {
        SuggestionDetails {
            last_used: self.symbol_resolver.get_symbol_stats(&suggestion.name)
                .and_then(|stats| stats.last_used.map(|_| "recently".to_string())),
            usage_frequency: self.symbol_resolver.get_symbol_stats(&suggestion.name)
                .map(|stats| format!("{} times", stats.total_uses)),
            type_info: match suggestion.symbol_type {
                SymbolType::Function => Some("Function".to_string()),
                SymbolType::Variable => Some("Variable".to_string()),
                _ => None,
            },
            context_example: suggestion.example_usage.clone(),
            related_symbols: Vec::new(), // Would be populated with related symbols
        }
    }
}

impl Default for SymbolSuggestionEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_suggestion_engine_creation() {
        let engine = SymbolSuggestionEngine::new();
        assert_eq!(engine.config.max_suggestions, 10);
    }
    
    #[test]
    fn test_context_type_detection() {
        let engine = SymbolSuggestionEngine::new();
        
        assert_eq!(engine.determine_context_type("x = ", 4), ContextType::VariableAssignment);
        assert_eq!(engine.determine_context_type("Sin[", 4), ContextType::FunctionParameter);
        assert_eq!(engine.determine_context_type("x_", 2), ContextType::PatternMatch);
    }
    
    #[test]
    fn test_variable_name_suggestions() {
        let engine = SymbolSuggestionEngine::new();
        
        let suggestions = engine.suggest_variable_names("count items", Some("Integer"));
        assert!(suggestions.contains(&"count".to_string()));
        assert!(suggestions.contains(&"n".to_string()));
    }
    
    #[test]
    fn test_function_name_suggestions() {
        let engine = SymbolSuggestionEngine::new();
        
        let suggestions = engine.suggest_function_names("calculate math", Some("Number"));
        assert!(suggestions.contains(&"calculate".to_string()));
        assert!(suggestions.contains(&"compute".to_string()));
    }
    
    #[test]
    fn test_context_specific_suggestions() {
        let engine = SymbolSuggestionEngine::new();
        
        let suggestions = engine.get_context_specific_suggestions(ContextType::ListContext, "");
        assert!(suggestions.contains(&"Map".to_string()));
        assert!(suggestions.contains(&"Length".to_string()));
    }
}