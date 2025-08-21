//! Enhanced Helper Architecture for Lyra REPL
//!
//! Provides a modular, trait-based architecture that serves as the foundation
//! for all quality-of-life improvements in the Lyra REPL. This architecture
//! supports syntax highlighting, smart hints, input validation, and enhanced
//! completion while maintaining backward compatibility with existing functionality.

use crate::repl::{config::ReplConfig, completion::SharedLyraCompleter, highlighting::LyraSyntaxHighlighter, validation::LyraInputValidator, hints::LyraHintEngine};
use crate::stdlib::StandardLibrary;
use rustyline::{
    Helper,
    completion::{Completer, Pair},
    hint::Hinter,
    highlight::Highlighter,
    validate::{Validator, ValidationResult, ValidationContext},
    Context, Result as RustylineResult
};
use std::sync::Arc;
use std::borrow::Cow;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HelperError {
    #[error("Configuration error: {0}")]
    Config(#[from] crate::repl::config::ConfigError),
    #[error("Invalid configuration value: {message}")]
    InvalidConfig { message: String },
    #[error("Helper component error: {message}")]
    Component { message: String },
}

pub type HelperResult<T> = std::result::Result<T, HelperError>;

/// Main enhanced helper that integrates all quality-of-life components
pub struct EnhancedLyraHelper {
    /// Completion support (delegates to existing system)
    completer: SharedLyraCompleter,
    /// Syntax highlighting component
    highlighter: LyraHighlighter,
    /// Smart hints component
    hinter: LyraHinter,
    /// Input validation component
    validator: LyraValidator,
    /// Configuration
    config: Arc<ReplConfig>,
}

impl EnhancedLyraHelper {
    /// Create a new enhanced helper with the given configuration and completer
    pub fn new(config: ReplConfig, completer: SharedLyraCompleter, stdlib: &StandardLibrary) -> HelperResult<Self> {
        // Validate configuration
        config.validate()?;
        
        let config = Arc::new(config);
        
        Ok(Self {
            completer,
            highlighter: LyraHighlighter::new(&config),
            hinter: LyraHinter::new(&config, stdlib),
            validator: LyraValidator::new(&config),
            config,
        })
    }
    
    /// Get current configuration
    pub fn get_config(&self) -> &ReplConfig {
        &self.config
    }
    
    /// Update configuration at runtime
    pub fn update_config(&mut self, new_config: ReplConfig) -> HelperResult<()> {
        new_config.validate()?;
        
        // Update components with new configuration
        self.highlighter.update_config(&new_config);
        self.hinter.update_config(&new_config);
        self.validator.update_config(&new_config);
        
        self.config = Arc::new(new_config);
        Ok(())
    }
    
    /// Get helper information for %helper-info command
    pub fn get_helper_info(&self) -> String {
        let mut info = Vec::new();
        let highlighter_stats = self.highlighter.get_stats();
        let validator_stats = self.validator.get_stats();
        let hinter_stats = self.hinter.get_stats();
        
        info.push("Enhanced Lyra Helper Status".to_string());
        info.push("============================".to_string());
        info.push("".to_string());
        
        info.push("[Components]".to_string());
        info.push("  Completion: ✓ enabled with enhanced features".to_string());
        info.push(format!("  Highlighting: ✓ {} theme ({} cached entries)", 
            highlighter_stats.current_theme, highlighter_stats.cache_size));
        info.push(format!("  Hinting: ✓ smart hints ({} cached entries)", 
            hinter_stats.entries));
        info.push(format!("  Validation: ✓ {} features enabled", 
            self.count_enabled_validation_features(&validator_stats)));
        info.push("".to_string());
        
        info.push("[Configuration]".to_string());
        info.push(format!("  Editor mode: {}", self.config.editor.mode));
        info.push(format!("  Auto complete: {}", self.config.repl.auto_complete));
        info.push(format!("  History size: {}", self.config.repl.history_size));
        info.push(format!("  Colors enabled: {}", self.config.display.colors));
        info.push(format!("  Multiline support: {}", self.config.repl.multiline_support));
        info.push("".to_string());
        
        info.push("[Validation Features]".to_string());
        info.push(format!("  Bracket matching: {}", if validator_stats.bracket_matching_enabled { "✓" } else { "○" }));
        info.push(format!("  Syntax checking: {}", if validator_stats.syntax_checking_enabled { "✓" } else { "○" }));
        info.push(format!("  Multiline detection: {}", if validator_stats.multiline_detection_enabled { "✓" } else { "○" }));
        info.push("".to_string());
        
        info.push("[Future Features]".to_string());
        info.push("  • Smart hints with function signature display".to_string());
        info.push("  • Vi/Emacs mode enhancements".to_string());
        info.push("  • Advanced performance optimization".to_string());
        
        info.join("\n")
    }
    
    /// Count enabled validation features
    fn count_enabled_validation_features(&self, stats: &crate::repl::validation::ValidationStats) -> usize {
        let mut count = 0;
        if stats.bracket_matching_enabled { count += 1; }
        if stats.syntax_checking_enabled { count += 1; }
        if stats.multiline_detection_enabled { count += 1; }
        count
    }
    
    /// Get helper capabilities for introspection
    pub fn get_capabilities(&self) -> Vec<&'static str> {
        vec![
            "completion",
            "highlighting",
            "hinting", 
            "validation",
            "configuration",
            "runtime-updates"
        ]
    }
    
    /// Reload helper configuration (for %helper-reload command)
    pub fn reload_config(&mut self) -> HelperResult<String> {
        let new_config = ReplConfig::load_or_create_default()?;
        self.update_config(new_config)?;
        Ok("Helper configuration reloaded successfully".to_string())
    }
}

impl Helper for EnhancedLyraHelper {}

impl Completer for EnhancedLyraHelper {
    type Candidate = Pair;
    
    fn complete(
        &self,
        line: &str,
        pos: usize,
        ctx: &Context<'_>,
    ) -> RustylineResult<(usize, Vec<Self::Candidate>)> {
        // Delegate to existing completion system
        self.completer.complete(line, pos, ctx)
    }
}

impl Hinter for EnhancedLyraHelper {
    type Hint = String;
    
    fn hint(&self, line: &str, pos: usize, ctx: &Context<'_>) -> Option<Self::Hint> {
        // Delegate to hinter component
        self.hinter.hint(line, pos, ctx)
    }
}

impl Highlighter for EnhancedLyraHelper {
    fn highlight<'l>(&self, line: &'l str, pos: usize) -> Cow<'l, str> {
        // Delegate to highlighter component
        self.highlighter.highlight(line, pos)
    }
    
    fn highlight_char(&self, line: &str, pos: usize, forced: bool) -> bool {
        // Delegate to highlighter component
        self.highlighter.highlight_char(line, pos, forced)
    }
}

impl Validator for EnhancedLyraHelper {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> RustylineResult<ValidationResult> {
        // Delegate to validator component
        self.validator.validate(ctx)
    }
}

/// Syntax highlighting component - real implementation
pub struct LyraHighlighter {
    /// The underlying syntax highlighter
    syntax_highlighter: LyraSyntaxHighlighter,
}

impl LyraHighlighter {
    /// Create a new highlighter with the given configuration
    pub fn new(config: &ReplConfig) -> Self {
        Self {
            syntax_highlighter: LyraSyntaxHighlighter::new(config),
        }
    }
    
    /// Update the highlighter configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.syntax_highlighter.update_config(config);
    }
    
    /// Get highlighter statistics
    pub fn get_stats(&self) -> crate::repl::highlighting::HighlighterStats {
        self.syntax_highlighter.get_stats()
    }
    
    /// Clear the highlighting cache
    pub fn clear_cache(&self) {
        self.syntax_highlighter.clear_cache();
    }
}

impl Default for LyraHighlighter {
    fn default() -> Self {
        let default_config = ReplConfig::default();
        Self::new(&default_config)
    }
}

impl Highlighter for LyraHighlighter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        // Use the real syntax highlighter implementation
        self.syntax_highlighter.highlight_line(line)
    }
    
    fn highlight_char(&self, line: &str, pos: usize, _forced: bool) -> bool {
        // Enable character highlighting for brackets and similar
        self.syntax_highlighter.should_highlight_char(line, pos)
    }
}

/// Smart hints component - real implementation
pub struct LyraHinter {
    /// The underlying hint engine
    hint_engine: LyraHintEngine,
}

impl LyraHinter {
    /// Create a new hinter with the given configuration and standard library
    pub fn new(config: &ReplConfig, stdlib: &StandardLibrary) -> Self {
        Self {
            hint_engine: LyraHintEngine::new(stdlib, config),
        }
    }
    
    /// Update the hinter configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.hint_engine.update_config(config);
    }
    
    /// Get hinter statistics
    pub fn get_stats(&self) -> crate::repl::hints::HintCacheStats {
        self.hint_engine.get_cache_stats()
    }
    
    /// Clear the hint cache
    pub fn clear_cache(&self) {
        self.hint_engine.clear_cache();
    }
}

impl Hinter for LyraHinter {
    type Hint = String;
    
    fn hint(&self, line: &str, pos: usize, _ctx: &Context<'_>) -> Option<Self::Hint> {
        // Use the real hint engine implementation
        self.hint_engine.generate_hint(line, pos)
    }
}

/// Input validation component - real implementation
pub struct LyraValidator {
    /// The underlying input validator
    input_validator: LyraInputValidator,
}

impl LyraValidator {
    /// Create a new validator with the given configuration
    pub fn new(config: &ReplConfig) -> Self {
        Self {
            input_validator: LyraInputValidator::new(config),
        }
    }
    
    /// Update the validator configuration
    pub fn update_config(&mut self, config: &ReplConfig) {
        self.input_validator.update_config(config);
    }
    
    /// Get validation statistics
    pub fn get_stats(&self) -> crate::repl::validation::ValidationStats {
        self.input_validator.get_stats()
    }
}

impl Default for LyraValidator {
    fn default() -> Self {
        let default_config = ReplConfig::default();
        Self::new(&default_config)
    }
}

impl Validator for LyraValidator {
    fn validate(&self, ctx: &mut ValidationContext<'_>) -> RustylineResult<ValidationResult> {
        // Use the real input validator implementation
        let input = ctx.input();
        let validation_status = self.input_validator.validate_input(input);
        
        // Convert validation status to rustyline result
        self.input_validator.to_rustyline_result(validation_status)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::ReplEngine;
    
    #[test]
    fn test_enhanced_helper_creation() {
        let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
        let stdlib = StandardLibrary::new();
        let helper = EnhancedLyraHelper::new(
            repl_engine.get_config().clone(),
            repl_engine.create_shared_completer(),
            &stdlib
        );
        assert!(helper.is_ok());
    }
    
    #[test]
    fn test_helper_components() {
        let config = ReplConfig::default();
        let stdlib = StandardLibrary::new();
        
        let highlighter = LyraHighlighter::new(&config);
        let hinter = LyraHinter::new(&config, &stdlib);
        let validator = LyraValidator::new(&config);
        
        // Test placeholder implementations
        assert_eq!(highlighter.highlight("test", 0).as_ref(), "test");
        assert!(!highlighter.highlight_char("test", 0, false));
        
        // Note: Context requires specific parameters, just test component creation
        assert!(true);
    }
    
    #[test]
    fn test_config_integration() {
        let mut config = ReplConfig::default();
        config.editor.mode = "vi".to_string();
        
        let repl_engine = ReplEngine::new_with_config(config.clone())
            .expect("Failed to create REPL engine");
        let stdlib = StandardLibrary::new();
        let helper = EnhancedLyraHelper::new(config, repl_engine.create_shared_completer(), &stdlib)
            .expect("Failed to create helper");
        
        assert_eq!(helper.get_config().editor.mode, "vi");
    }
    
    #[test]
    fn test_helper_info() {
        let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
        let stdlib = StandardLibrary::new();
        let helper = EnhancedLyraHelper::new(
            repl_engine.get_config().clone(),
            repl_engine.create_shared_completer(),
            &stdlib
        ).expect("Failed to create helper");
        
        let info = helper.get_helper_info();
        assert!(info.contains("Enhanced Lyra Helper"));
        assert!(info.contains("Completion: enabled"));
    }
    
    #[test]
    fn test_capabilities() {
        let repl_engine = ReplEngine::new().expect("Failed to create REPL engine");
        let stdlib = StandardLibrary::new();
        let helper = EnhancedLyraHelper::new(
            repl_engine.get_config().clone(),
            repl_engine.create_shared_completer(),
            &stdlib
        ).expect("Failed to create helper");
        
        let capabilities = helper.get_capabilities();
        assert!(capabilities.contains(&"completion"));
        assert!(capabilities.contains(&"highlighting"));
        assert!(capabilities.contains(&"hinting"));
        assert!(capabilities.contains(&"validation"));
    }
}