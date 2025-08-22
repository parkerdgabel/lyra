//! Smart hints system for Lyra REPL
//!
//! This module provides intelligent, context-aware hints including:
//! - Function signature display
//! - Parameter type suggestions  
//! - Usage examples
//! - Context-sensitive help

use crate::repl::config::ReplConfig;
use crate::stdlib::StandardLibrary;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub mod function_hints;
pub mod parameter_hints;
pub mod context_analyzer;
pub mod documentation;
pub mod examples;
pub mod error_context;
pub mod symbol_resolution;
pub mod pattern_matching;
pub mod symbol_suggestions;

use function_hints::FunctionSignatureDatabase;
use parameter_hints::ParameterSuggester;
use context_analyzer::ContextAnalyzer;
use documentation::DocumentationDatabase;
use examples::ExamplesDatabase;
use error_context::ErrorContextAnalyzer;
use symbol_resolution::SymbolResolver;
use pattern_matching::{PatternAnalyzer, PatternAnalysisResult};
use symbol_suggestions::{SymbolSuggestionEngine, SuggestionConfig};

/// Smart hint engine for Lyra REPL with comprehensive features
pub struct LyraHintEngine {
    /// Function signature database
    function_db: Arc<FunctionSignatureDatabase>,
    /// Parameter suggestion engine
    parameter_suggester: ParameterSuggester,
    /// Context analyzer for position-aware hints
    context_analyzer: ContextAnalyzer,
    /// Rich documentation database
    documentation_db: Arc<DocumentationDatabase>,
    /// Interactive examples database
    examples_db: Arc<ExamplesDatabase>,
    /// Error context analyzer
    error_analyzer: ErrorContextAnalyzer,
    /// Symbol resolution engine
    symbol_resolver: Arc<Mutex<SymbolResolver>>,
    /// Pattern matching analyzer
    pattern_analyzer: Arc<PatternAnalyzer>,
    /// Context-aware symbol suggestion engine
    symbol_suggestion_engine: Arc<Mutex<SymbolSuggestionEngine>>,
    /// Hint cache for performance
    hint_cache: Mutex<HashMap<String, CachedHint>>,
    /// Configuration
    config: HintConfig,
    /// Maximum cache size
    max_cache_size: usize,
}

/// Configuration for hint system
#[derive(Debug, Clone)]
pub struct HintConfig {
    /// Enable function signature hints
    pub show_function_signatures: bool,
    /// Enable parameter suggestions
    pub show_parameter_hints: bool,
    /// Enable usage examples
    pub show_examples: bool,
    /// Enable rich documentation
    pub show_documentation: bool,
    /// Enable error analysis hints
    pub show_error_analysis: bool,
    /// Enable symbol resolution hints
    pub show_symbol_resolution: bool,
    /// Enable pattern matching hints
    pub show_pattern_hints: bool,
    /// Maximum example difficulty level (1-5)
    pub max_example_difficulty: u8,
    /// Maximum hint length in characters
    pub max_hint_length: usize,
    /// Hint display timeout in milliseconds
    pub hint_timeout_ms: u64,
}

impl Default for HintConfig {
    fn default() -> Self {
        Self {
            show_function_signatures: true,
            show_parameter_hints: true,
            show_examples: true,
            show_documentation: true,
            show_error_analysis: true,
            show_symbol_resolution: true,
            show_pattern_hints: true,
            max_example_difficulty: 3,
            max_hint_length: 200,
            hint_timeout_ms: 100,
        }
    }
}

impl HintConfig {
    /// Create hint config from REPL config
    pub fn from_repl_config(config: &ReplConfig) -> Self {
        // For now use defaults, but this can be extended with specific hint settings
        let hint_config = Self::default();
        
        // Respect display settings
        if !config.display.colors {
            // Could adjust hint formatting for non-color terminals
        }
        
        hint_config
    }
}

/// Cached hint with metadata
#[derive(Debug, Clone)]
struct CachedHint {
    /// The hint text
    hint: String,
    /// Context hash for cache validation
    context_hash: u64,
    /// Cache timestamp
    timestamp: std::time::Instant,
}

/// Result of hint analysis
#[derive(Debug, Clone)]
pub enum HintResult {
    /// Function signature hint
    FunctionSignature {
        function_name: String,
        signature: String,
        description: String,
    },
    /// Parameter suggestion hint
    ParameterSuggestion {
        expected_type: String,
        suggestions: Vec<String>,
        current_param: usize,
    },
    /// Usage example hint
    UsageExample {
        example: String,
        description: String,
    },
    /// Error context hint
    ErrorContext {
        error_message: String,
        suggestion: String,
    },
    /// Symbol resolution hint
    SymbolResolution {
        symbol_name: String,
        symbol_type: String,
        suggestions: Vec<String>,
    },
    /// Pattern matching hint
    PatternMatch {
        pattern_type: String,
        suggestions: Vec<String>,
        examples: Vec<String>,
    },
    /// No hint available
    None,
}

impl LyraHintEngine {
    /// Create a new hint engine with comprehensive features
    pub fn new(stdlib: &StandardLibrary, config: &ReplConfig) -> Self {
        let hint_config = HintConfig::from_repl_config(config);
        let function_db = Arc::new(FunctionSignatureDatabase::from_stdlib(stdlib));
        let documentation_db = Arc::new(DocumentationDatabase::new());
        let examples_db = Arc::new(ExamplesDatabase::new());
        let error_analyzer = ErrorContextAnalyzer::new();
        let symbol_resolver = Arc::new(Mutex::new(SymbolResolver::new()));
        let pattern_analyzer = Arc::new(PatternAnalyzer::new());
        let suggestion_config = SuggestionConfig::default();
        let symbol_suggestion_engine = Arc::new(Mutex::new(SymbolSuggestionEngine::with_config(suggestion_config)));
        
        Self {
            function_db: function_db.clone(),
            parameter_suggester: ParameterSuggester::new(function_db.clone()),
            context_analyzer: ContextAnalyzer::new(),
            documentation_db,
            examples_db,
            error_analyzer,
            symbol_resolver,
            pattern_analyzer,
            symbol_suggestion_engine,
            hint_cache: Mutex::new(HashMap::new()),
            config: hint_config,
            max_cache_size: 1000,
        }
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.config = HintConfig::from_repl_config(config);
    }
    
    /// Generate hint for given input and cursor position
    pub fn generate_hint(&self, line: &str, pos: usize) -> Option<String> {
        // Check cache first
        let cache_key = format!("{}:{}", line, pos);
        if let Some(cached) = self.check_cache(&cache_key) {
            return Some(cached);
        }
        
        // Analyze the input context
        let context = match self.context_analyzer.analyze(line, pos) {
            Ok(ctx) => ctx,
            Err(_) => return None,
        };
        
        // Generate appropriate hint based on context
        let hint_result = match context.hint_type {
            context_analyzer::HintType::FunctionCall => {
                self.generate_function_hint(&context)
            }
            context_analyzer::HintType::ParameterPosition => {
                self.generate_parameter_hint(&context)
            }
            context_analyzer::HintType::ErrorContext => {
                self.generate_error_hint(&context)
            }
            context_analyzer::HintType::None => HintResult::None,
        };
        
        // Convert hint result to string
        let hint_text = self.format_hint(hint_result)?;
        
        // Cache the result
        self.cache_hint(cache_key, &hint_text);
        
        Some(hint_text)
    }
    
    /// Generate function signature hint
    fn generate_function_hint(&self, context: &context_analyzer::HintContext) -> HintResult {
        if !self.config.show_function_signatures {
            return HintResult::None;
        }
        
        if let Some(function_name) = &context.function_name {
            if let Some(signature) = self.function_db.get_signature(function_name) {
                return HintResult::FunctionSignature {
                    function_name: function_name.clone(),
                    signature: signature.signature.clone(),
                    description: signature.description.clone(),
                };
            }
        }
        
        HintResult::None
    }
    
    /// Generate parameter suggestion hint
    fn generate_parameter_hint(&self, context: &context_analyzer::HintContext) -> HintResult {
        if !self.config.show_parameter_hints {
            return HintResult::None;
        }
        
        self.parameter_suggester.suggest_parameter(context)
    }
    
    /// Generate error context hint
    fn generate_error_hint(&self, context: &context_analyzer::HintContext) -> HintResult {
        if !self.config.show_error_analysis {
            return HintResult::None;
        }
        
        // Use enhanced error analysis
        match &context.error_type {
            Some(error) => {
                // For now, return basic hint - can be enhanced with error_analyzer
                HintResult::ErrorContext {
                    error_message: error.clone(),
                    suggestion: "Check syntax and function parameters".to_string(),
                }
            },
            None => HintResult::None,
        }
    }
    
    /// Generate symbol resolution hint
    fn generate_symbol_hint(&self, symbol_name: &str, line: &str, pos: usize) -> HintResult {
        if !self.config.show_symbol_resolution {
            return HintResult::None;
        }
        
        if let Ok(resolver) = self.symbol_resolver.lock() {
            let resolution_result = resolver.resolve_symbol(symbol_name);
            
            if let Some(symbol) = resolution_result.symbol {
                return HintResult::SymbolResolution {
                    symbol_name: symbol.name.clone(),
                    symbol_type: format!("{:?}", symbol.symbol_type),
                    suggestions: vec![symbol.name.clone()],
                };
            } else if !resolution_result.suggestions.is_empty() {
                let suggestions: Vec<String> = resolution_result.suggestions
                    .into_iter()
                    .map(|s| s.name)
                    .collect();
                
                return HintResult::SymbolResolution {
                    symbol_name: symbol_name.to_string(),
                    symbol_type: "Unknown".to_string(),
                    suggestions,
                };
            }
        }
        
        HintResult::None
    }
    
    /// Generate pattern matching hint
    fn generate_pattern_hint(&self, pattern: &str) -> HintResult {
        if !self.config.show_pattern_hints {
            return HintResult::None;
        }
        
        let analysis = self.pattern_analyzer.analyze_pattern(pattern);
        
        if let Some(pattern_type) = analysis.pattern_type {
            let examples = self.pattern_analyzer
                .get_pattern_examples(&pattern_type)
                .into_iter()
                .map(|ex| ex.pattern)
                .collect();
            
            return HintResult::PatternMatch {
                pattern_type: format!("{:?}", pattern_type),
                suggestions: analysis.captured_variables,
                examples,
            };
        }
        
        HintResult::None
    }
    
    /// Format hint result into display string
    fn format_hint(&self, hint_result: HintResult) -> Option<String> {
        match hint_result {
            HintResult::FunctionSignature { function_name, signature, description } => {
                let mut hint = format!("{}: {}", function_name, signature);
                
                if self.config.show_documentation && !description.is_empty() {
                    hint = format!("{} - {}", hint, description);
                }
                
                // Add example if requested and available
                if self.config.show_examples {
                    if let Some(examples) = self.examples_db.get_function_examples(&function_name) {
                        if let Some(first_example) = examples.quickstart.first() {
                            if first_example.difficulty <= self.config.max_example_difficulty {
                                hint = format!("{} (e.g., {})", hint, first_example.code);
                            }
                        }
                    }
                }
                
                Some(self.truncate_hint(hint))
            }
            HintResult::ParameterSuggestion { expected_type, suggestions, current_param } => {
                let hint = if suggestions.is_empty() {
                    format!("Parameter {}: {}", current_param + 1, expected_type)
                } else {
                    format!("Parameter {}: {} (try: {})", 
                            current_param + 1, 
                            expected_type, 
                            suggestions.join(", "))
                };
                
                Some(self.truncate_hint(hint))
            }
            HintResult::UsageExample { example, description } => {
                let hint = format!("{}: {}", description, example);
                Some(self.truncate_hint(hint))
            }
            HintResult::ErrorContext { error_message, suggestion } => {
                let hint = format!("Error: {} - {}", error_message, suggestion);
                Some(self.truncate_hint(hint))
            }
            HintResult::SymbolResolution { symbol_name, symbol_type, suggestions } => {
                let hint = if suggestions.len() == 1 && suggestions[0] == symbol_name {
                    format!("{}: {}", symbol_name, symbol_type)
                } else {
                    format!("Unknown symbol '{}' - did you mean: {}", symbol_name, suggestions.join(", "))
                };
                Some(self.truncate_hint(hint))
            }
            HintResult::PatternMatch { pattern_type, suggestions, examples } => {
                let mut hint = format!("Pattern ({})", pattern_type);
                if !suggestions.is_empty() {
                    hint = format!("{} - captures: {}", hint, suggestions.join(", "));
                }
                if !examples.is_empty() && examples.len() <= 2 {
                    hint = format!("{} - examples: {}", hint, examples.join(", "));
                }
                Some(self.truncate_hint(hint))
            }
            HintResult::None => None,
        }
    }
    
    /// Truncate hint to maximum length
    fn truncate_hint(&self, hint: String) -> String {
        if hint.len() <= self.config.max_hint_length {
            hint
        } else {
            let mut truncated = hint.chars().take(self.config.max_hint_length - 3).collect::<String>();
            truncated.push_str("...");
            truncated
        }
    }
    
    /// Check cache for existing hint
    fn check_cache(&self, key: &str) -> Option<String> {
        let cache = self.hint_cache.lock().ok()?;
        let cached = cache.get(key)?;
        
        // Check if cache entry is still valid (within timeout)
        let elapsed = cached.timestamp.elapsed();
        if elapsed.as_millis() > self.config.hint_timeout_ms as u128 {
            return None;
        }
        
        Some(cached.hint.clone())
    }
    
    /// Cache a hint result
    fn cache_hint(&self, key: String, hint: &str) {
        if let Ok(mut cache) = self.hint_cache.lock() {
            // Clean cache if it's getting too large
            if cache.len() >= self.max_cache_size {
                cache.clear();
            }
            
            cache.insert(key, CachedHint {
                hint: hint.to_string(),
                context_hash: self.calculate_context_hash(hint),
                timestamp: std::time::Instant::now(),
            });
        }
    }
    
    /// Calculate context hash for cache validation
    fn calculate_context_hash(&self, hint: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        hint.hash(&mut hasher);
        hasher.finish()
    }
    
    /// Clear hint cache
    pub fn clear_cache(&self) {
        if let Ok(mut cache) = self.hint_cache.lock() {
            cache.clear();
        }
    }
    
    /// Get comprehensive function information
    pub fn get_function_info(&self, function_name: &str) -> Option<ComprehensiveFunctionInfo> {
        let signature = self.function_db.get_signature(function_name)?;
        let documentation = self.documentation_db.get_documentation(function_name);
        let examples = self.examples_db.get_function_examples(function_name);
        
        Some(ComprehensiveFunctionInfo {
            signature: signature.clone(),
            documentation: documentation.cloned(),
            examples: examples.cloned(),
            formatted_signature: format!("{}: {}", signature.name, signature.signature),
        })
    }
    
    /// Analyze error and provide helpful hints
    pub fn analyze_error(&self, error_message: &str, input: &str) -> Option<ErrorHintResult> {
        if !self.config.show_error_analysis {
            return None;
        }
        
        let error_hint = self.error_analyzer.analyze_error(error_message, input, None);
        
        Some(ErrorHintResult {
            error_type: format!("{:?}", error_hint.error_type),
            friendly_message: error_hint.primary_message,
            suggestions: error_hint.suggestions.iter()
                .map(|s| s.description.clone())
                .collect(),
            did_you_mean: error_hint.did_you_mean,
            context_info: error_hint.context_info,
        })
    }
    
    /// Search examples by keyword
    pub fn search_examples(&self, keyword: &str) -> Vec<String> {
        let results = self.examples_db.search_examples(keyword);
        results.into_iter()
            .take(5) // Limit to top 5 results
            .map(|example| {
                format!("{}: {}", example.title, example.code)
            })
            .collect()
    }
    
    /// Get learning paths
    pub fn get_learning_paths(&self) -> Vec<String> {
        self.examples_db.get_all_learning_paths()
            .into_iter()
            .map(|path| format!("{}: {}", path.name, path.description))
            .collect()
    }
    
    /// Get cache statistics
    pub fn get_cache_stats(&self) -> HintCacheStats {
        if let Ok(cache) = self.hint_cache.lock() {
            HintCacheStats {
                entries: cache.len(),
                max_size: self.max_cache_size,
            }
        } else {
            HintCacheStats::default()
        }
    }
    
    /// Update symbol information from input
    pub fn update_symbols(&self, input: &str) -> Result<(), String> {
        if let Ok(mut resolver) = self.symbol_resolver.lock() {
            resolver.analyze_input(input)
        } else {
            Err("Failed to acquire symbol resolver lock".to_string())
        }
    }
    
    /// Get symbol suggestions for completion
    pub fn get_symbol_suggestions(&self, input: &str, cursor_pos: usize, prefix: &str) -> Vec<String> {
        if let Ok(mut engine) = self.symbol_suggestion_engine.lock() {
            let result = engine.get_suggestions(input, cursor_pos, prefix);
            result.suggestions
                .into_iter()
                .map(|s| s.base_suggestion.name)
                .collect()
        } else {
            Vec::new()
        }
    }
    
    /// Analyze pattern and provide hints
    pub fn analyze_pattern(&self, pattern: &str) -> PatternAnalysisResult {
        self.pattern_analyzer.analyze_pattern(pattern)
    }
    
    /// Get pattern construction suggestions
    pub fn get_pattern_suggestions(&self, context: &str, intent: &str) -> Vec<String> {
        self.pattern_analyzer.get_pattern_suggestions(context, intent)
            .into_iter()
            .map(|template| format!("{}: {}", template.name, template.template))
            .collect()
    }
    
    /// Validate pattern syntax
    pub fn validate_pattern(&self, pattern: &str) -> Vec<String> {
        let errors = self.pattern_analyzer.validate_pattern_syntax(pattern);
        errors.into_iter()
            .map(|error| format!("{}: {}", error.error_type, error.message))
            .collect()
    }
    
    /// Get rule construction hints
    pub fn get_rule_hints(&self, pattern: &str) -> Vec<String> {
        self.pattern_analyzer.get_rule_construction_hints(pattern, "")
            .into_iter()
            .map(|hint| format!("{} -> {}", hint.pattern, hint.replacement))
            .collect()
    }
    
    /// Record symbol usage for learning
    pub fn record_symbol_usage(&self, symbol: &str, context: &str) {
        if let Ok(mut resolver) = self.symbol_resolver.lock() {
            resolver.record_usage(symbol, context);
        }
    }
    
    /// Get available symbols in current scope
    pub fn get_available_symbols(&self) -> Vec<String> {
        if let Ok(resolver) = self.symbol_resolver.lock() {
            let mut symbols = Vec::new();
            
            // Get variables
            symbols.extend(resolver.get_variables_in_scope().iter().map(|v| v.name.clone()));
            
            // Get user-defined functions
            symbols.extend(resolver.get_user_functions().iter().map(|f| f.name.clone()));
            
            symbols.sort();
            symbols.dedup();
            symbols
        } else {
            Vec::new()
        }
    }
}

/// Statistics about hint cache
#[derive(Debug, Clone, Default)]
pub struct HintCacheStats {
    pub entries: usize,
    pub max_size: usize,
}

/// Comprehensive function information
#[derive(Debug, Clone)]
pub struct ComprehensiveFunctionInfo {
    /// Function signature
    pub signature: function_hints::FunctionSignature,
    /// Rich documentation
    pub documentation: Option<documentation::DocumentationEntry>,
    /// Interactive examples
    pub examples: Option<examples::FunctionExamples>,
    /// Formatted signature string
    pub formatted_signature: String,
}

/// Error hint result
#[derive(Debug, Clone)]
pub struct ErrorHintResult {
    /// Type of error
    pub error_type: String,
    /// User-friendly error message
    pub friendly_message: String,
    /// Suggested fixes
    pub suggestions: Vec<String>,
    /// "Did you mean" alternatives
    pub did_you_mean: Vec<String>,
    /// Additional context information
    pub context_info: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::StandardLibrary;
    use crate::repl::config::ReplConfig;

    #[test]
    fn test_hint_engine_creation() {
        let stdlib = StandardLibrary::new();
        let config = ReplConfig::default();
        let engine = LyraHintEngine::new(&stdlib, &config);
        
        let stats = engine.get_cache_stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.max_size, 1000);
    }
    
    #[test]
    fn test_hint_config_from_repl() {
        let config = ReplConfig::default();
        let hint_config = HintConfig::from_repl_config(&config);
        
        assert!(hint_config.show_function_signatures);
        assert!(hint_config.show_parameter_hints);
        assert!(hint_config.show_examples);
    }
    
    #[test]
    fn test_cache_functionality() {
        let stdlib = StandardLibrary::new();
        let config = ReplConfig::default();
        let engine = LyraHintEngine::new(&stdlib, &config);
        
        // Test cache miss
        let hint1 = engine.generate_hint("Sin[", 4);
        
        // Test cache hit (same input)
        let hint2 = engine.generate_hint("Sin[", 4);
        
        // Both should return the same result (or None)
        assert_eq!(hint1, hint2);
    }
    
    #[test]
    fn test_hint_truncation() {
        let mut config = HintConfig::default();
        config.max_hint_length = 20;
        
        let stdlib = StandardLibrary::new();
        let repl_config = ReplConfig::default();
        let mut engine = LyraHintEngine::new(&stdlib, &repl_config);
        engine.config = config;
        
        let long_hint = "This is a very long hint that should be truncated".to_string();
        let truncated = engine.truncate_hint(long_hint);
        
        assert!(truncated.len() <= 20);
        assert!(truncated.ends_with("..."));
    }
}