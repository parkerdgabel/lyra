//! Auto-completion support for the REPL
//! 
//! Provides intelligent tab completion for:
//! - StandardLibrary function names
//! - User-defined variables from the environment
//! - Meta commands (%help, %history, etc.)
//! - File paths for load/save operations

use rustyline::completion::{Completer, FilenameCompleter, Pair};
use rustyline::{Context, Result as RustylineResult, Helper};
use rustyline::hint::Hinter;
use rustyline::highlight::Highlighter;
use rustyline::validate::Validator;
use crate::stdlib::StandardLibrary;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use crate::vm::Value;
use crate::repl::hints::symbol_suggestions::{SymbolSuggestionEngine, ContextType};

/// Custom completer for the Lyra REPL
pub struct LyraCompleter {
    /// StandardLibrary function names for completion
    function_names: Vec<String>,
    /// File completer for file path completion
    file_completer: FilenameCompleter,
    /// Meta commands available in REPL
    meta_commands: Vec<String>,
    /// Symbol suggestion engine for context-aware completion
    symbol_engine: SymbolSuggestionEngine,
    /// Bracket/quote completion configuration
    smart_completion_config: SmartCompletionConfig,
}

/// Configuration for smart bracket and quote completion
#[derive(Debug, Clone)]
pub struct SmartCompletionConfig {
    /// Enable automatic bracket completion
    pub enable_bracket_completion: bool,
    /// Enable automatic quote completion
    pub enable_quote_completion: bool,
    /// Enable context-aware suggestions
    pub enable_context_suggestions: bool,
    /// Enable function parameter hints
    pub enable_parameter_hints: bool,
}

impl Default for SmartCompletionConfig {
    fn default() -> Self {
        Self {
            enable_bracket_completion: true,
            enable_quote_completion: true,
            enable_context_suggestions: true,
            enable_parameter_hints: true,
        }
    }
}

/// Result of smart completion analysis
#[derive(Debug, Clone)]
pub struct SmartCompletionResult {
    /// The suggested completion text
    pub completion: String,
    /// Whether to auto-insert closing brackets/quotes
    pub auto_close: Option<String>,
    /// Context information for the suggestion
    pub context_info: Option<String>,
    /// Parameter hints for function calls
    pub parameter_hints: Vec<String>,
}

impl LyraCompleter {
    /// Create a new completer with StandardLibrary function names
    pub fn new() -> Self {
        let stdlib = StandardLibrary::new();
        let function_names: Vec<String> = stdlib.function_names().into_iter().cloned().collect();
        
        let meta_commands = vec![
            // Basic REPL commands
            "%help".to_string(),
            "%history".to_string(),
            "%perf".to_string(),
            "%clear".to_string(),
            "%vars".to_string(),
            "%timing".to_string(),
            
            // File operations
            "%load".to_string(),
            "%save".to_string(),
            "%run".to_string(),
            "%include".to_string(),
            "%examples".to_string(),
            
            // Documentation commands
            "%doc".to_string(),
            "%search".to_string(),
            "%categories".to_string(),
            
            // Benchmarking commands
            "%benchmark".to_string(),
            "%profile".to_string(),
            "%compare".to_string(),
            "%memory".to_string(),
            
            // Session management
            "%export".to_string(),
            "%import".to_string(),
            "%reset".to_string(),
            "%checkpoint".to_string(),
            "%restore".to_string(),
            
            // Helper management
            "%helper-info".to_string(),
            "%helper-reload".to_string(),
            
            // Debug commands
            "%debug".to_string(),
            "%step".to_string(),
            "%continue".to_string(),
            "%stop".to_string(),
            "%inspect".to_string(),
            "%tree".to_string(),
            "%debug-status".to_string(),
            
            // Performance profiling commands
            "%profile".to_string(),
            "%prof".to_string(),
            "%profile-report".to_string(),
            "%prof-report".to_string(),
            "%preport".to_string(),
            "%profile-summary".to_string(),
            "%prof-summary".to_string(),
            "%psummary".to_string(),
            "%profile-visualize".to_string(),
            "%prof-viz".to_string(),
            "%pviz".to_string(),
            "%profile-export".to_string(),
            "%prof-export".to_string(),
            "%pexport".to_string(),
            "%profile-stop".to_string(),
            "%prof-stop".to_string(),
            "%pstop".to_string(),
            
            // Baseline management commands
            "%baseline-set".to_string(),
            "%bset".to_string(),
            "%baseline-compare".to_string(),
            "%bcompare".to_string(),
            "%baseline-info".to_string(),
            "%binfo".to_string(),
            "%baseline-clear".to_string(),
            "%bclear".to_string(),
            
            // Regression detection commands
            "%regression-detect".to_string(),
            "%regress".to_string(),
            "%rdetect".to_string(),
            "%regression-status".to_string(),
            "%rstatus".to_string(),
            
            // Benchmark commands
            "%benchmark-run".to_string(),
            "%bench-run".to_string(),
            "%brun".to_string(),
            "%benchmark-suite".to_string(),
            "%bench-suite".to_string(),
            "%bsuite".to_string(),
            "%benchmark-category".to_string(),
            "%bench-cat".to_string(),
            "%bcat".to_string(),
            "%benchmark-compare".to_string(),
            "%bench-cmp".to_string(),
            "%bcmp".to_string(),
            "%benchmark-list".to_string(),
            "%bench-list".to_string(),
            "%blist".to_string(),
            "%benchmark-history".to_string(),
            "%bench-hist".to_string(),
            "%bhist".to_string(),
        ];

        Self {
            function_names,
            file_completer: FilenameCompleter::new(),
            meta_commands,
            symbol_engine: SymbolSuggestionEngine::new(),
            smart_completion_config: SmartCompletionConfig::default(),
        }
    }

    /// Get completions for function names
    pub fn complete_functions(&self, text: &str) -> Vec<Pair> {
        self.function_names
            .iter()
            .filter(|name| name.to_lowercase().starts_with(&text.to_lowercase()))
            .map(|name| Pair {
                display: name.clone(),
                replacement: name.clone(),
            })
            .collect()
    }

    /// Get completions for meta commands
    pub fn complete_meta_commands(&self, text: &str) -> Vec<Pair> {
        self.meta_commands
            .iter()
            .filter(|cmd| cmd.to_lowercase().starts_with(&text.to_lowercase()))
            .map(|cmd| Pair {
                display: cmd.clone(),
                replacement: cmd.clone(),
            })
            .collect()
    }

    /// Get completions for variables from the environment
    pub fn complete_variables(&self, text: &str, environment: &HashMap<String, Value>) -> Vec<Pair> {
        environment
            .keys()
            .filter(|var| var.to_lowercase().starts_with(&text.to_lowercase()))
            .map(|var| Pair {
                display: var.clone(),
                replacement: var.clone(),
            })
            .collect()
    }

    /// Update function names (for dynamic function registration)
    pub fn update_function_names(&mut self, new_functions: Vec<String>) {
        self.function_names = new_functions;
        self.function_names.sort();
    }
    
    /// Update the symbol engine with new input for learning
    pub fn update_symbols(&mut self, input: &str) -> Result<(), String> {
        self.symbol_engine.update_symbols(input)
    }
    
    /// Get smart completion suggestions with context analysis
    pub fn get_smart_completion(&mut self, line: &str, pos: usize) -> SmartCompletionResult {
        // Find the word being completed
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace() || "()[]{},".contains(c))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        let word = &line[start..pos];
        
        // Get context-aware suggestions from symbol engine
        let suggestion_result = self.symbol_engine.get_suggestions(line, pos, word);
        
        // Determine auto-completion based on context
        let auto_close = self.determine_auto_close(line, pos, &suggestion_result.context.context_type);
        
        // Extract parameter hints if in function context
        let parameter_hints = if let Some(func_ctx) = &suggestion_result.context.function_context {
            self.get_parameter_hints(&func_ctx.function_name, func_ctx.parameter_index)
        } else {
            Vec::new()
        };
        
        // Get the best completion suggestion
        let completion = if let Some(best_suggestion) = suggestion_result.suggestions.first() {
            best_suggestion.base_suggestion.name.clone()
        } else if !suggestion_result.pattern_suggestions.is_empty() {
            suggestion_result.pattern_suggestions[0].clone()
        } else {
            word.to_string()
        };
        
        // Generate context info
        let context_info = if let Some(best_suggestion) = suggestion_result.suggestions.first() {
            Some(best_suggestion.context_info.clone())
        } else if !suggestion_result.context_hints.is_empty() {
            Some(suggestion_result.context_hints[0].clone())
        } else {
            None
        };
        
        SmartCompletionResult {
            completion,
            auto_close,
            context_info,
            parameter_hints,
        }
    }
    
    /// Determine what should be auto-closed based on context
    fn determine_auto_close(&self, line: &str, pos: usize, context_type: &ContextType) -> Option<String> {
        if !self.smart_completion_config.enable_bracket_completion && 
           !self.smart_completion_config.enable_quote_completion {
            return None;
        }
        
        let before_cursor = &line[..pos];
        let after_cursor = &line[pos..];
        
        match context_type {
            ContextType::FunctionCall => {
                // If we're completing a function name and there's no '[' after it
                if !after_cursor.trim_start().starts_with('[') {
                    Some("[".to_string())
                } else {
                    None
                }
            },
            ContextType::StringContext => {
                // If we're in a string context and there's an unmatched quote
                if self.smart_completion_config.enable_quote_completion {
                    let quote_count = before_cursor.matches('"').count();
                    if quote_count % 2 == 1 && !after_cursor.starts_with('"') {
                        Some("\"".to_string())
                    } else {
                        None
                    }
                } else {
                    None
                }
            },
            ContextType::ListContext => {
                // If we're in a list and there's no closing brace
                let open_braces = before_cursor.matches('{').count();
                let close_braces = before_cursor.matches('}').count();
                if open_braces > close_braces && !after_cursor.trim_start().starts_with('}') {
                    Some("}".to_string())
                } else {
                    None
                }
            },
            _ => None,
        }
    }
    
    /// Get parameter hints for a function
    fn get_parameter_hints(&self, function_name: &str, current_param: usize) -> Vec<String> {
        if !self.smart_completion_config.enable_parameter_hints {
            return Vec::new();
        }
        
        // Basic parameter hints for common functions
        let hints = match function_name {
            "Sin" | "Cos" | "Tan" => vec!["angle_in_radians".to_string()],
            "Map" => vec!["function".to_string(), "list".to_string()],
            "Select" => vec!["list".to_string(), "criteria_function".to_string()],
            "Apply" => vec!["function".to_string(), "arguments_list".to_string()],
            "Function" => vec!["variables".to_string(), "body".to_string()],
            "Plot" => vec!["function".to_string(), "range".to_string()],
            "Solve" => vec!["equation".to_string(), "variable".to_string()],
            "D" => vec!["expression".to_string(), "variable".to_string()],
            "Integrate" => vec!["expression".to_string(), "variable".to_string(), "limits".to_string()],
            "Length" => vec!["list_or_string".to_string()],
            "Join" => vec!["list1".to_string(), "list2".to_string()],
            "Take" => vec!["list".to_string(), "count".to_string()],
            "Drop" => vec!["list".to_string(), "count".to_string()],
            "Range" => vec!["start".to_string(), "end".to_string(), "step".to_string()],
            "RandomInteger" => vec!["range_or_max".to_string()],
            "RandomReal" => vec!["range_or_max".to_string()],
            _ => vec!["parameter".to_string()],
        };
        
        // Return hint for current parameter if available
        if current_param < hints.len() {
            vec![format!("Parameter {}: {}", current_param + 1, hints[current_param])]
        } else {
            vec!["Additional parameter".to_string()]
        }
    }
    
    /// Get enhanced completions using symbol engine
    pub fn get_enhanced_completions(&mut self, line: &str, pos: usize, word: &str) -> Vec<Pair> {
        let suggestion_result = self.symbol_engine.get_suggestions(line, pos, word);
        let mut candidates = Vec::new();
        
        // Add enhanced symbol suggestions
        for suggestion in suggestion_result.suggestions {
            let display = if self.smart_completion_config.enable_context_suggestions {
                format!("{} - {}", suggestion.base_suggestion.name, suggestion.context_info)
            } else {
                suggestion.base_suggestion.name.clone()
            };
            
            candidates.push(Pair {
                display,
                replacement: suggestion.base_suggestion.name,
            });
        }
        
        // Add pattern suggestions
        for pattern in suggestion_result.pattern_suggestions {
            candidates.push(Pair {
                display: format!("{} (pattern)", pattern),
                replacement: pattern,
            });
        }
        
        // Add context-specific suggestions
        let context_suggestions = self.symbol_engine.get_context_specific_suggestions(
            suggestion_result.context.context_type,
            &suggestion_result.context.input
        );
        
        for suggestion in context_suggestions {
            candidates.push(Pair {
                display: format!("{} (context)", suggestion),
                replacement: suggestion,
            });
        }
        
        candidates
    }
    
    /// Record symbol usage for learning
    pub fn record_usage(&mut self, symbol: &str, context: &str) {
        self.symbol_engine.record_symbol_usage(symbol, context);
    }
    
    /// Update smart completion configuration
    pub fn update_config(&mut self, config: SmartCompletionConfig) {
        self.smart_completion_config = config;
    }
}

impl Completer for LyraCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &Context<'_>,
    ) -> RustylineResult<(usize, Vec<Self::Candidate>)> {
        // Note: This method can't use enhanced completions because it requires &self
        // Enhanced completions are available through get_enhanced_completions method
        
        // Find the word being completed
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace() || "()[]{},".contains(c))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        let word = &line[start..pos];
        
        let mut candidates = Vec::new();

        // Meta commands (start with %)
        if word.starts_with('%') {
            candidates.extend(self.complete_meta_commands(word));
        }
        // File paths (for meta commands like %load, %save)
        else if self.should_complete_file_path(line, pos) {
            match self.file_completer.complete(line, pos, _ctx) {
                Ok((start_pos, file_candidates)) => {
                    return Ok((start_pos, file_candidates));
                }
                Err(_) => {} // Fall through to other completion types
            }
        }
        // Function names and variables
        else {
            // Complete function names
            candidates.extend(self.complete_functions(word));
            
            // Note: Variable completion requires access to the environment
            // This will be handled in the integration with ReplEngine
        }

        // Sort candidates by relevance (exact prefix match first, then alphabetical)
        candidates.sort_by(|a, b| {
            let a_exact = a.replacement.to_lowercase() == word.to_lowercase();
            let b_exact = b.replacement.to_lowercase() == word.to_lowercase();
            
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.replacement.cmp(&b.replacement),
            }
        });

        Ok((start, candidates))
    }

}

impl LyraCompleter {
    /// Check if we should complete file paths based on context
    fn should_complete_file_path(&self, line: &str, pos: usize) -> bool {
        // Look for meta commands that expect file paths
        let file_commands = ["%load", "%save", "%run", "%import", "%export"];
        
        for cmd in &file_commands {
            if let Some(cmd_pos) = line.find(cmd) {
                let after_cmd = cmd_pos + cmd.len();
                if pos > after_cmd && line[after_cmd..pos].trim_start().len() > 0 {
                    return true;
                }
            }
        }
        false
    }
}

/// Shared completer wrapper that can access the environment from ReplEngine
pub struct SharedLyraCompleter {
    /// Base completer with function names and meta commands
    base_completer: Arc<Mutex<LyraCompleter>>,
    /// Shared reference to the environment for variable completion
    environment: Arc<Mutex<HashMap<String, Value>>>,
}

impl SharedLyraCompleter {
    /// Create a new shared completer with environment reference
    pub fn new(environment: Arc<Mutex<HashMap<String, Value>>>) -> Self {
        Self {
            base_completer: Arc::new(Mutex::new(LyraCompleter::new())),
            environment,
        }
    }
    
    /// Get enhanced completions with context awareness
    pub fn get_enhanced_completions(&self, line: &str, pos: usize) -> Vec<Pair> {
        if let Ok(mut completer) = self.base_completer.lock() {
            // Find the word being completed
            let start = line[..pos]
                .rfind(|c: char| c.is_whitespace() || "()[]{},".contains(c))
                .map(|i| i + 1)
                .unwrap_or(0);
            
            let word = &line[start..pos];
            
            // Get enhanced completions from the symbol engine
            let mut candidates = completer.get_enhanced_completions(line, pos, word);
            
            // Add variable completions from environment
            if let Ok(env) = self.environment.lock() {
                candidates.extend(completer.complete_variables(word, &env));
            }
            
            // Sort and return
            candidates.sort_by(|a, b| {
                let a_exact = a.replacement.to_lowercase() == word.to_lowercase();
                let b_exact = b.replacement.to_lowercase() == word.to_lowercase();
                
                match (a_exact, b_exact) {
                    (true, false) => std::cmp::Ordering::Less,
                    (false, true) => std::cmp::Ordering::Greater,
                    _ => a.replacement.cmp(&b.replacement),
                }
            });
            
            candidates
        } else {
            Vec::new()
        }
    }
    
    /// Get smart completion with auto-close suggestions
    pub fn get_smart_completion(&self, line: &str, pos: usize) -> Option<SmartCompletionResult> {
        if let Ok(mut completer) = self.base_completer.lock() {
            Some(completer.get_smart_completion(line, pos))
        } else {
            None
        }
    }
    
    /// Update symbols for learning
    pub fn update_symbols(&self, input: &str) -> Result<(), String> {
        if let Ok(mut completer) = self.base_completer.lock() {
            completer.update_symbols(input)
        } else {
            Err("Failed to acquire completer lock".to_string())
        }
    }
    
    /// Record symbol usage
    pub fn record_usage(&self, symbol: &str, context: &str) {
        if let Ok(mut completer) = self.base_completer.lock() {
            completer.record_usage(symbol, context);
        }
    }
    
    /// Update smart completion configuration
    pub fn update_config(&self, config: SmartCompletionConfig) {
        if let Ok(mut completer) = self.base_completer.lock() {
            completer.update_config(config);
        }
    }
}

impl Completer for SharedLyraCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> RustylineResult<(usize, Vec<Self::Candidate>)> {
        // Find the word being completed
        let start = line[..pos]
            .rfind(|c: char| c.is_whitespace() || "()[]{},".contains(c))
            .map(|i| i + 1)
            .unwrap_or(0);
        
        let word = &line[start..pos];
        
        let mut candidates = Vec::new();

        if let Ok(completer) = self.base_completer.lock() {
            // Meta commands (start with %)
            if word.starts_with('%') {
                candidates.extend(completer.complete_meta_commands(word));
            }
            // File paths (for meta commands like %load, %save)
            else if completer.should_complete_file_path(line, pos) {
                return completer.complete(line, pos, ctx);
            }
            // Function names and variables - use fallback for rustyline compatibility
            else {
                // Complete function names
                candidates.extend(completer.complete_functions(word));
                
                // Complete variables from shared environment
                if let Ok(env) = self.environment.lock() {
                    candidates.extend(completer.complete_variables(word, &env));
                }
            }
        }

        // Sort candidates by relevance (exact prefix match first, then alphabetical)
        candidates.sort_by(|a, b| {
            let a_exact = a.replacement.to_lowercase() == word.to_lowercase();
            let b_exact = b.replacement.to_lowercase() == word.to_lowercase();
            
            match (a_exact, b_exact) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => a.replacement.cmp(&b.replacement),
            }
        });

        Ok((start, candidates))
    }
}

impl Helper for SharedLyraCompleter {}

impl Hinter for SharedLyraCompleter {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &Context<'_>) -> Option<Self::Hint> {
        None
    }
}

impl Highlighter for SharedLyraCompleter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> std::borrow::Cow<'l, str> {
        std::borrow::Cow::Borrowed(line)
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        false
    }
}

impl Validator for SharedLyraCompleter {}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_completion() {
        let completer = LyraCompleter::new();
        let completions = completer.complete_functions("Si");
        
        // Should include Sin, Sign, etc.
        assert!(completions.iter().any(|c| c.replacement == "Sin"));
        assert!(completions.iter().any(|c| c.replacement == "Sign"));
    }

    #[test]
    fn test_meta_command_completion() {
        let completer = LyraCompleter::new();
        let completions = completer.complete_meta_commands("%h");
        
        // Should include %help, %history
        assert!(completions.iter().any(|c| c.replacement == "%help"));
        assert!(completions.iter().any(|c| c.replacement == "%history"));
    }

    #[test]
    fn test_variable_completion() {
        let completer = LyraCompleter::new();
        let mut env = HashMap::new();
        env.insert("myVariable".to_string(), Value::Integer(42));
        env.insert("myOtherVar".to_string(), Value::String("test".to_string()));
        
        let completions = completer.complete_variables("my", &env);
        
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().any(|c| c.replacement == "myVariable"));
        assert!(completions.iter().any(|c| c.replacement == "myOtherVar"));
    }

    #[test]
    fn test_case_insensitive_completion() {
        let completer = LyraCompleter::new();
        let completions = completer.complete_functions("sin");
        
        // Should find Sin even with lowercase input
        assert!(completions.iter().any(|c| c.replacement == "Sin"));
    }

    #[test]
    fn test_file_path_detection() {
        let completer = LyraCompleter::new();
        
        assert!(completer.should_complete_file_path("%load /path/to/", 12));
        assert!(completer.should_complete_file_path("%save script.ly", 15));
        assert!(!completer.should_complete_file_path("Sin[x]", 6));
    }
    
    #[test]
    fn test_smart_completion_config() {
        let mut completer = LyraCompleter::new();
        let config = SmartCompletionConfig {
            enable_bracket_completion: false,
            enable_quote_completion: false,
            enable_context_suggestions: true,
            enable_parameter_hints: true,
        };
        
        completer.update_config(config.clone());
        assert_eq!(completer.smart_completion_config.enable_bracket_completion, false);
        assert_eq!(completer.smart_completion_config.enable_quote_completion, false);
    }
    
    #[test]
    fn test_smart_completion_bracket_detection() {
        let completer = LyraCompleter::new();
        
        // Test function call context - should suggest opening bracket
        let auto_close = completer.determine_auto_close("Si", 2, &ContextType::FunctionCall);
        assert_eq!(auto_close, Some("[".to_string()));
        
        // Test function call with existing bracket - should not suggest
        let auto_close = completer.determine_auto_close("Sin[", 4, &ContextType::FunctionCall);
        assert_eq!(auto_close, None);
    }
    
    #[test]
    fn test_smart_completion_quote_detection() {
        let completer = LyraCompleter::new();
        
        // Test string context with unmatched quote
        let auto_close = completer.determine_auto_close("message = \"hello", 16, &ContextType::StringContext);
        assert_eq!(auto_close, Some("\"".to_string()));
        
        // Test string context with matched quotes
        let auto_close = completer.determine_auto_close("message = \"hello\"", 17, &ContextType::StringContext);
        assert_eq!(auto_close, None);
    }
    
    #[test]
    fn test_parameter_hints() {
        let completer = LyraCompleter::new();
        
        let hints = completer.get_parameter_hints("Sin", 0);
        assert_eq!(hints, vec!["Parameter 1: angle_in_radians"]);
        
        let hints = completer.get_parameter_hints("Map", 1);
        assert_eq!(hints, vec!["Parameter 2: list"]);
        
        let hints = completer.get_parameter_hints("UnknownFunction", 0);
        assert_eq!(hints, vec!["Parameter 1: parameter"]);
    }
    
    #[test]
    fn test_smart_completion_result() {
        let mut completer = LyraCompleter::new();
        
        // Test smart completion for function call
        let result = completer.get_smart_completion("Si", 2);
        assert!(!result.completion.is_empty());
        
        // Test that context info is generated
        if result.context_info.is_some() {
            assert!(!result.context_info.unwrap().is_empty());
        }
    }
    
    #[test]
    fn test_enhanced_completions() {
        let mut completer = LyraCompleter::new();
        
        // Update symbols to learn from input
        let _ = completer.update_symbols("x = 42; f[x_] := x^2");
        
        // Test enhanced completions
        let candidates = completer.get_enhanced_completions("f[", 2, "f");
        
        // Should have at least some suggestions
        assert!(!candidates.is_empty());
    }
    
    #[test]
    fn test_shared_completer_enhanced_functionality() {
        let env = Arc::new(Mutex::new(HashMap::new()));
        env.lock().unwrap().insert("testVar".to_string(), Value::Integer(42));
        
        let shared_completer = SharedLyraCompleter::new(env);
        
        // Test enhanced completions
        let candidates = shared_completer.get_enhanced_completions("testV", 5);
        assert!(candidates.iter().any(|c| c.replacement == "testVar"));
        
        // Test smart completion
        let smart_result = shared_completer.get_smart_completion("testV", 5);
        assert!(smart_result.is_some());
        
        // Test symbol updates
        let result = shared_completer.update_symbols("newVar = 123");
        assert!(result.is_ok());
        
        // Test usage recording
        shared_completer.record_usage("testVar", "variable assignment");
    }
    
    #[test]
    fn test_list_context_auto_close() {
        let completer = LyraCompleter::new();
        
        // Test list context with unmatched brace
        let auto_close = completer.determine_auto_close("{1, 2, 3", 8, &ContextType::ListContext);
        assert_eq!(auto_close, Some("}".to_string()));
        
        // Test list context with matched braces
        let auto_close = completer.determine_auto_close("{1, 2, 3}", 9, &ContextType::ListContext);
        assert_eq!(auto_close, None);
    }
    
    #[test]
    fn test_completion_with_disabled_features() {
        let mut completer = LyraCompleter::new();
        let config = SmartCompletionConfig {
            enable_bracket_completion: false,
            enable_quote_completion: false,
            enable_context_suggestions: false,
            enable_parameter_hints: false,
        };
        completer.update_config(config);
        
        // Should not suggest auto-close when disabled
        let auto_close = completer.determine_auto_close("Si", 2, &ContextType::FunctionCall);
        assert_eq!(auto_close, None);
        
        // Should not provide parameter hints when disabled
        let hints = completer.get_parameter_hints("Sin", 0);
        assert!(hints.is_empty());
    }
}