//! REPL Configuration System
//! 
//! Provides comprehensive configuration management for the Lyra REPL including:
//! - Cross-platform configuration directories
//! - TOML-based configuration with sensible defaults  
//! - Environment variable overrides
//! - Configuration validation and error handling
//! - Thread-safe operations

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::env;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("TOML parsing error: {0}")]
    TomlParse(#[from] toml::de::Error),
    #[error("TOML serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Path error: {0}")]
    Path(String),
}

pub type ConfigResult<T> = std::result::Result<T, ConfigError>;

/// REPL subsystem configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplSubConfig {
    /// Maximum number of history entries to retain
    pub history_size: usize,
    /// Remove duplicate entries from history
    pub remove_duplicates: bool,
    /// Show execution timing information
    pub show_timing: bool,
    /// Show performance optimization information
    pub show_performance: bool,
    /// Enable auto-completion
    pub auto_complete: bool,
    /// Enable multiline input support
    pub multiline_support: bool,
}

impl Default for ReplSubConfig {
    fn default() -> Self {
        Self {
            history_size: 10000,
            remove_duplicates: true,
            show_timing: false,
            show_performance: false,
            auto_complete: true,
            multiline_support: true,
        }
    }
}

/// Display subsystem configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Enable colored output
    pub colors: bool,
    /// Enable Unicode character support
    pub unicode_support: bool,
    /// Maximum length of output before truncation
    pub max_output_length: usize,
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            colors: true,
            unicode_support: true,
            max_output_length: 1000,
        }
    }
}

/// Editor subsystem configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EditorConfig {
    /// Editor mode: "emacs" or "vi"
    pub mode: String,
    /// External editor command (supports ${EDITOR} substitution)
    pub external_editor: String,
}

impl Default for EditorConfig {
    fn default() -> Self {
        let default_editor = env::var("EDITOR").unwrap_or_else(|_| "vim".to_string());
        Self {
            mode: "emacs".to_string(),
            external_editor: default_editor,
        }
    }
}

/// Validation subsystem configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable bracket/quote balance checking
    pub enable_bracket_matching: bool,
    /// Enable syntax validation
    pub enable_syntax_checking: bool,
    /// Enable multiline detection
    pub enable_multiline_detection: bool,
    /// Show helpful validation hints
    pub show_validation_hints: bool,
    /// Validation delay in milliseconds
    pub validation_delay_ms: u64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_bracket_matching: true,
            enable_syntax_checking: true,
            enable_multiline_detection: true,
            show_validation_hints: true,
            validation_delay_ms: 50, // Quick feedback
        }
    }
}

/// Complete REPL configuration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplConfig {
    pub repl: ReplSubConfig,
    pub display: DisplayConfig,
    pub editor: EditorConfig,
    pub validation: ValidationConfig,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            repl: ReplSubConfig::default(),
            display: DisplayConfig::default(),
            editor: EditorConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl ReplConfig {
    /// Load configuration from file, falling back to defaults for missing values
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> ConfigResult<Self> {
        let content = fs::read_to_string(path.as_ref())?;
        let mut config: Self = toml::from_str(&content)?;
        
        // Apply environment variable overrides
        config.apply_env_overrides();
        
        // Validate the configuration
        config.validate()?;
        
        Ok(config)
    }
    
    /// Load configuration from the default location, creating if necessary
    pub fn load_or_create_default() -> ConfigResult<Self> {
        let config_path = Self::get_config_file_path()?;
        
        if config_path.exists() {
            Self::load_from_file(&config_path)
        } else {
            // Create default config and save it
            let config = Self::default();
            
            // Ensure config directory exists
            if let Some(parent) = config_path.parent() {
                fs::create_dir_all(parent)?;
            }
            
            config.save_to_file(&config_path)?;
            Ok(config)
        }
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> ConfigResult<()> {
        let toml_content = toml::to_string_pretty(self)?;
        fs::write(path.as_ref(), toml_content)?;
        Ok(())
    }
    
    /// Apply environment variable overrides
    pub fn apply_env_overrides(&mut self) {
        // REPL configuration overrides
        if let Ok(val) = env::var("LYRA_HISTORY_SIZE") {
            if let Ok(size) = val.parse::<usize>() {
                self.repl.history_size = size;
            }
        }
        
        if let Ok(val) = env::var("LYRA_REMOVE_DUPLICATES") {
            if let Ok(remove) = val.parse::<bool>() {
                self.repl.remove_duplicates = remove;
            }
        }
        
        if let Ok(val) = env::var("LYRA_SHOW_TIMING") {
            if let Ok(show) = val.parse::<bool>() {
                self.repl.show_timing = show;
            }
        }
        
        if let Ok(val) = env::var("LYRA_SHOW_PERFORMANCE") {
            if let Ok(show) = val.parse::<bool>() {
                self.repl.show_performance = show;
            }
        }
        
        if let Ok(val) = env::var("LYRA_AUTO_COMPLETE") {
            if let Ok(complete) = val.parse::<bool>() {
                self.repl.auto_complete = complete;
            }
        }
        
        if let Ok(val) = env::var("LYRA_MULTILINE_SUPPORT") {
            if let Ok(multiline) = val.parse::<bool>() {
                self.repl.multiline_support = multiline;
            }
        }
        
        // Display configuration overrides
        if let Ok(val) = env::var("LYRA_COLORS") {
            if let Ok(colors) = val.parse::<bool>() {
                self.display.colors = colors;
            }
        }
        
        if let Ok(val) = env::var("LYRA_UNICODE_SUPPORT") {
            if let Ok(unicode) = val.parse::<bool>() {
                self.display.unicode_support = unicode;
            }
        }
        
        if let Ok(val) = env::var("LYRA_MAX_OUTPUT_LENGTH") {
            if let Ok(length) = val.parse::<usize>() {
                self.display.max_output_length = length;
            }
        }
        
        // Editor configuration overrides
        if let Ok(mode) = env::var("LYRA_EDITOR_MODE") {
            if mode == "emacs" || mode == "vi" {
                self.editor.mode = mode;
            }
        }
        
        if let Ok(editor) = env::var("LYRA_EXTERNAL_EDITOR") {
            self.editor.external_editor = editor;
        }
        
        // Also support standard EDITOR environment variable
        if let Ok(editor) = env::var("EDITOR") {
            if self.editor.external_editor.contains("${EDITOR}") {
                self.editor.external_editor = self.editor.external_editor.replace("${EDITOR}", &editor);
            } else if self.editor.external_editor == "vim" {
                // Default case - replace with actual EDITOR value
                self.editor.external_editor = editor;
            }
        }
        
        // Validation configuration overrides
        if let Ok(val) = env::var("LYRA_ENABLE_BRACKET_MATCHING") {
            if let Ok(enable) = val.parse::<bool>() {
                self.validation.enable_bracket_matching = enable;
            }
        }
        
        if let Ok(val) = env::var("LYRA_ENABLE_SYNTAX_CHECKING") {
            if let Ok(enable) = val.parse::<bool>() {
                self.validation.enable_syntax_checking = enable;
            }
        }
        
        if let Ok(val) = env::var("LYRA_ENABLE_MULTILINE_DETECTION") {
            if let Ok(enable) = val.parse::<bool>() {
                self.validation.enable_multiline_detection = enable;
            }
        }
        
        if let Ok(val) = env::var("LYRA_SHOW_VALIDATION_HINTS") {
            if let Ok(show) = val.parse::<bool>() {
                self.validation.show_validation_hints = show;
            }
        }
        
        if let Ok(val) = env::var("LYRA_VALIDATION_DELAY_MS") {
            if let Ok(delay) = val.parse::<u64>() {
                self.validation.validation_delay_ms = delay;
            }
        }
    }
    
    /// Validate configuration values
    pub fn validate(&self) -> ConfigResult<()> {
        // Validate history size
        if self.repl.history_size == 0 {
            return Err(ConfigError::Validation(
                "History size must be greater than 0".to_string()
            ));
        }
        
        if self.repl.history_size > 1_000_000 {
            return Err(ConfigError::Validation(
                "History size must be less than 1,000,000 for performance reasons".to_string()
            ));
        }
        
        // Validate max output length
        if self.display.max_output_length == 0 {
            return Err(ConfigError::Validation(
                "Max output length must be greater than 0".to_string()
            ));
        }
        
        if self.display.max_output_length > 100_000 {
            return Err(ConfigError::Validation(
                "Max output length must be less than 100,000 for performance reasons".to_string()
            ));
        }
        
        // Validate editor mode
        if self.editor.mode != "emacs" && self.editor.mode != "vi" {
            return Err(ConfigError::Validation(
                format!("Editor mode must be 'emacs' or 'vi', got '{}'", self.editor.mode)
            ));
        }
        
        // Validate external editor is not empty
        if self.editor.external_editor.trim().is_empty() {
            return Err(ConfigError::Validation(
                "External editor command cannot be empty".to_string()
            ));
        }
        
        // Validate validation delay
        if self.validation.validation_delay_ms > 5000 {
            return Err(ConfigError::Validation(
                "Validation delay must be less than 5000ms for responsive feedback".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get the cross-platform configuration directory
    pub fn get_config_dir() -> ConfigResult<PathBuf> {
        dirs::config_dir()
            .map(|dir| dir.join("lyra"))
            .ok_or_else(|| ConfigError::Path(
                "Unable to determine config directory".to_string()
            ))
    }
    
    /// Get the configuration file path
    pub fn get_config_file_path() -> ConfigResult<PathBuf> {
        Ok(Self::get_config_dir()?.join("config.toml"))
    }
    
    /// Get the history file path
    pub fn get_history_file_path() -> ConfigResult<PathBuf> {
        // Try environment variable first
        if let Ok(path) = env::var("LYRA_HISTORY_FILE") {
            return Ok(PathBuf::from(path));
        }
        
        // Fall back to standard location
        dirs::home_dir()
            .map(|dir| dir.join(".lyra_history"))
            .ok_or_else(|| ConfigError::Path(
                "Unable to determine home directory".to_string()
            ))
    }
    
    /// Get configuration directory from environment or default
    pub fn get_config_dir_with_env() -> ConfigResult<PathBuf> {
        if let Ok(config_dir) = env::var("LYRA_CONFIG_DIR") {
            Ok(PathBuf::from(config_dir))
        } else {
            Self::get_config_dir()
        }
    }
}

/// History-specific configuration derived from main config
#[derive(Debug, Clone)]
pub struct HistoryConfig {
    pub size: usize,
    pub remove_duplicates: bool,
    pub session_isolation: bool,
}

impl HistoryConfig {
    /// Create history config from main REPL config
    pub fn from_repl_config(config: &ReplConfig) -> Self {
        Self {
            size: config.repl.history_size,
            remove_duplicates: config.repl.remove_duplicates,
            session_isolation: env::var("LYRA_SESSION_ISOLATION")
                .map(|v| v.parse().unwrap_or(false))
                .unwrap_or(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_default_config() {
        let config = ReplConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = ReplConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: ReplConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(config, parsed);
    }

    #[test]
    fn test_env_variable_override() {
        env::set_var("LYRA_HISTORY_SIZE", "5000");
        env::set_var("LYRA_COLORS", "false");
        
        let mut config = ReplConfig::default();
        config.apply_env_overrides();
        
        assert_eq!(config.repl.history_size, 5000);
        assert_eq!(config.display.colors, false);
        
        env::remove_var("LYRA_HISTORY_SIZE");
        env::remove_var("LYRA_COLORS");
    }

    #[test]
    fn test_validation_errors() {
        let mut config = ReplConfig::default();
        
        // Test invalid history size
        config.repl.history_size = 0;
        assert!(config.validate().is_err());
        
        // Test invalid editor mode
        config = ReplConfig::default();
        config.editor.mode = "invalid".to_string();
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("test_config.toml");
        
        let config = ReplConfig::default();
        config.save_to_file(&config_path).unwrap();
        
        let loaded_config = ReplConfig::load_from_file(&config_path).unwrap();
        assert_eq!(config, loaded_config);
    }

    #[test]
    fn test_partial_config_loading() {
        let temp_dir = TempDir::new().unwrap();
        let config_path = temp_dir.path().join("partial_config.toml");
        
        let partial_content = r#"
[repl]
history_size = 5000

[display]
colors = false
"#;
        
        fs::write(&config_path, partial_content).unwrap();
        let config = ReplConfig::load_from_file(&config_path).unwrap();
        
        // Check overridden values
        assert_eq!(config.repl.history_size, 5000);
        assert_eq!(config.display.colors, false);
        
        // Check default values are preserved
        assert_eq!(config.repl.remove_duplicates, true);
        assert_eq!(config.editor.mode, "emacs");
    }

    #[test]
    fn test_cross_platform_paths() {
        // These should not panic and should return valid paths
        let config_dir = ReplConfig::get_config_dir();
        assert!(config_dir.is_ok());
        
        let history_path = ReplConfig::get_history_file_path();
        assert!(history_path.is_ok());
        
        let config_file_path = ReplConfig::get_config_file_path();
        assert!(config_file_path.is_ok());
    }
}