//! Export Engine Trait and Core Types
//! 
//! Defines the interface for export engines and common export functionality

use crate::repl::export::SessionSnapshot;
use std::path::PathBuf;
use std::collections::HashMap;
use thiserror::Error;
use serde::{Serialize, Deserialize};

/// Export engine trait that all exporters must implement
pub trait ExportEngine: Send + Sync {
    /// Export a session snapshot to the specified output path
    fn export(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
        output_path: &PathBuf,
    ) -> ExportResult<()>;
    
    /// Preview the export without writing to file
    fn preview(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> ExportResult<String>;
    
    /// Validate that the configuration is valid for this exporter
    fn validate_config(&self, config: &ExportConfig) -> ExportResult<()>;
    
    /// Get the default file extension for this format
    fn default_extension(&self) -> &'static str;
    
    /// Get the MIME type for this format
    fn mime_type(&self) -> &'static str;
    
    /// Get a human-readable description of this format
    fn description(&self) -> &'static str;
    
    /// Check if this exporter supports the given feature
    fn supports_feature(&self, feature: ExportFeature) -> bool;
    
    /// Get format-specific configuration schema
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({})
    }
}

/// Supported export formats
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExportFormat {
    /// Jupyter Notebook (.ipynb)
    Jupyter,
    /// LaTeX document (.tex)
    LaTeX,
    /// HTML document (.html)
    Html,
    /// Markdown document (.md)
    Markdown,
    /// Plain text (.txt)
    Text,
    /// JSON format (.json)
    Json,
    /// PDF (requires LaTeX)
    Pdf,
}

impl ExportFormat {
    /// Get all available formats
    pub fn all() -> Vec<ExportFormat> {
        vec![
            ExportFormat::Jupyter,
            ExportFormat::LaTeX,
            ExportFormat::Html,
            ExportFormat::Markdown,
            ExportFormat::Text,
            ExportFormat::Json,
            ExportFormat::Pdf,
        ]
    }
    
    /// Get the default file extension
    pub fn default_extension(self) -> &'static str {
        match self {
            ExportFormat::Jupyter => "ipynb",
            ExportFormat::LaTeX => "tex",
            ExportFormat::Html => "html",
            ExportFormat::Markdown => "md",
            ExportFormat::Text => "txt",
            ExportFormat::Json => "json",
            ExportFormat::Pdf => "pdf",
        }
    }
    
    /// Get the MIME type
    pub fn mime_type(self) -> &'static str {
        match self {
            ExportFormat::Jupyter => "application/x-ipynb+json",
            ExportFormat::LaTeX => "application/x-latex",
            ExportFormat::Html => "text/html",
            ExportFormat::Markdown => "text/markdown",
            ExportFormat::Text => "text/plain",
            ExportFormat::Json => "application/json",
            ExportFormat::Pdf => "application/pdf",
        }
    }
    
    /// Get a human-readable description
    pub fn description(self) -> &'static str {
        match self {
            ExportFormat::Jupyter => "Jupyter Notebook with interactive cells",
            ExportFormat::LaTeX => "LaTeX document with mathematical formatting",
            ExportFormat::Html => "HTML document with syntax highlighting",
            ExportFormat::Markdown => "Markdown document with code blocks",
            ExportFormat::Text => "Plain text format",
            ExportFormat::Json => "Raw JSON session data",
            ExportFormat::Pdf => "PDF document (via LaTeX)",
        }
    }
}

impl std::fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExportFormat::Jupyter => write!(f, "jupyter"),
            ExportFormat::LaTeX => write!(f, "latex"),
            ExportFormat::Html => write!(f, "html"),
            ExportFormat::Markdown => write!(f, "markdown"),
            ExportFormat::Text => write!(f, "text"),
            ExportFormat::Json => write!(f, "json"),
            ExportFormat::Pdf => write!(f, "pdf"),
        }
    }
}

impl std::str::FromStr for ExportFormat {
    type Err = ExportError;
    
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "jupyter" | "ipynb" => Ok(ExportFormat::Jupyter),
            "latex" | "tex" => Ok(ExportFormat::LaTeX),
            "html" | "htm" => Ok(ExportFormat::Html),
            "markdown" | "md" => Ok(ExportFormat::Markdown),
            "text" | "txt" => Ok(ExportFormat::Text),
            "json" => Ok(ExportFormat::Json),
            "pdf" => Ok(ExportFormat::Pdf),
            _ => Err(ExportError::InvalidFormat { format: s.to_string() }),
        }
    }
}

/// Export features that different engines may or may not support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExportFeature {
    /// Syntax highlighting in code blocks
    SyntaxHighlighting,
    /// Mathematical expression rendering
    MathRendering,
    /// Embedded images and plots
    EmbeddedImages,
    /// Interactive elements
    Interactive,
    /// Metadata preservation
    Metadata,
    /// Performance data inclusion
    PerformanceData,
    /// Error information
    ErrorDetails,
    /// Cross-references and links
    CrossReferences,
    /// Custom styling/themes
    CustomStyling,
    /// Table of contents
    TableOfContents,
}

/// Configuration for export operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Include execution timing information
    pub include_timing: bool,
    /// Include error details in output
    pub include_errors: bool,
    /// Include session metadata
    pub include_metadata: bool,
    /// Include environment variables
    pub include_environment: bool,
    /// Enable syntax highlighting
    pub syntax_highlighting: bool,
    /// Mathematical rendering style
    pub math_rendering: MathRenderingStyle,
    /// Code block style
    pub code_style: CodeBlockStyle,
    /// Maximum output length per cell (None for unlimited)
    pub max_output_length: Option<usize>,
    /// Include performance profiling data
    pub include_performance_data: bool,
    /// Custom template path (format-specific)
    pub custom_template: Option<PathBuf>,
    /// Additional format-specific options
    pub format_options: HashMap<String, serde_json::Value>,
    /// Author information
    pub author: Option<String>,
    /// Document title
    pub title: Option<String>,
    /// Export timestamp
    pub include_export_timestamp: bool,
    /// Compression settings
    pub compression: CompressionSettings,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_timing: true,
            include_errors: true,
            include_metadata: true,
            include_environment: false,
            syntax_highlighting: true,
            math_rendering: MathRenderingStyle::LaTeX,
            code_style: CodeBlockStyle::Highlighted,
            max_output_length: Some(10000),
            include_performance_data: false,
            custom_template: None,
            format_options: HashMap::new(),
            author: None,
            title: None,
            include_export_timestamp: true,
            compression: CompressionSettings::default(),
        }
    }
}

/// Mathematical rendering style options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MathRenderingStyle {
    /// LaTeX mathematical notation
    LaTeX,
    /// MathML format
    MathML,
    /// ASCII mathematical notation
    ASCII,
    /// Unicode mathematical symbols
    Unicode,
    /// No special math rendering
    None,
}

/// Code block styling options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CodeBlockStyle {
    /// Syntax highlighted with colors
    Highlighted,
    /// Plain text code blocks
    Plain,
    /// Minimal formatting
    Minimal,
    /// Rich formatting with line numbers
    Rich,
}

/// Compression settings for exports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9, algorithm dependent)
    pub level: u8,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            algorithm: CompressionAlgorithm::Gzip,
            level: 6,
        }
    }
}

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    /// Gzip compression
    Gzip,
    /// Bzip2 compression
    Bzip2,
    /// LZMA compression
    Lzma,
}

/// Export operation errors
#[derive(Error, Debug)]
pub enum ExportError {
    #[error("Unsupported export format: {format:?}")]
    UnsupportedFormat { format: ExportFormat },
    
    #[error("Invalid format string: {format}")]
    InvalidFormat { format: String },
    
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },
    
    #[error("Serialization error: {message}")]
    Serialization { message: String },
    
    #[error("Template error: {message}")]
    Template { message: String },
    
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Validation error: {message}")]
    Validation { message: String },
    
    #[error("Feature not supported: {feature:?} for format {format:?}")]
    FeatureNotSupported {
        feature: ExportFeature,
        format: ExportFormat,
    },
    
    #[error("Conversion error: {message}")]
    Conversion { message: String },
    
    #[error("External tool error: {tool}: {message}")]
    ExternalTool { tool: String, message: String },
    
    #[error("Dependency missing: {dependency}")]
    MissingDependency { dependency: String },
}

/// Result type for export operations
pub type ExportResult<T> = Result<T, ExportError>;

/// Export progress callback
pub type ProgressCallback = Box<dyn Fn(f64, &str) + Send + Sync>;

/// Advanced export options for complex exports
pub struct AdvancedExportOptions {
    /// Progress callback for long-running exports
    pub progress_callback: Option<ProgressCallback>,
    /// Custom CSS/styling for HTML exports
    pub custom_css: Option<String>,
    /// Custom JavaScript for interactive exports
    pub custom_javascript: Option<String>,
    /// Include source maps for debugging
    pub include_source_maps: bool,
    /// Minify output where applicable
    pub minify_output: bool,
    /// External resource handling
    pub resource_handling: ResourceHandling,
}

impl Default for AdvancedExportOptions {
    fn default() -> Self {
        Self {
            progress_callback: None,
            custom_css: None,
            custom_javascript: None,
            include_source_maps: false,
            minify_output: false,
            resource_handling: ResourceHandling::Embed,
        }
    }
}

/// How to handle external resources (images, etc.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceHandling {
    /// Embed resources directly in the output
    Embed,
    /// Link to external resources
    Link,
    /// Copy resources to output directory
    Copy,
    /// Ignore external resources
    Ignore,
}

/// Export validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the export is valid
    pub is_valid: bool,
    /// List of warnings (non-fatal issues)
    pub warnings: Vec<String>,
    /// List of errors (fatal issues)
    pub errors: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

impl ValidationResult {
    /// Create a successful validation result
    pub fn success() -> Self {
        Self {
            is_valid: true,
            warnings: Vec::new(),
            errors: Vec::new(),
            suggestions: Vec::new(),
        }
    }
    
    /// Create a failed validation result
    pub fn failure(errors: Vec<String>) -> Self {
        Self {
            is_valid: false,
            warnings: Vec::new(),
            errors,
            suggestions: Vec::new(),
        }
    }
    
    /// Add a warning to the validation result
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// Add an error to the validation result
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }
    
    /// Add a suggestion to the validation result
    pub fn add_suggestion(&mut self, suggestion: String) {
        self.suggestions.push(suggestion);
    }
    
    /// Format the validation result as a human-readable string
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        
        if self.is_valid {
            report.push_str("✅ Export validation passed\n");
        } else {
            report.push_str("❌ Export validation failed\n");
        }
        
        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for error in &self.errors {
                report.push_str(&format!("  • {}\n", error));
            }
        }
        
        if !self.warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in &self.warnings {
                report.push_str(&format!("  • {}\n", warning));
            }
        }
        
        if !self.suggestions.is_empty() {
            report.push_str("\nSuggestions:\n");
            for suggestion in &self.suggestions {
                report.push_str(&format!("  • {}\n", suggestion));
            }
        }
        
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_export_format_parsing() {
        assert_eq!("jupyter".parse::<ExportFormat>().unwrap(), ExportFormat::Jupyter);
        assert_eq!("ipynb".parse::<ExportFormat>().unwrap(), ExportFormat::Jupyter);
        assert_eq!("latex".parse::<ExportFormat>().unwrap(), ExportFormat::LaTeX);
        assert_eq!("html".parse::<ExportFormat>().unwrap(), ExportFormat::Html);
        
        assert!("unknown".parse::<ExportFormat>().is_err());
    }
    
    #[test]
    fn test_export_format_properties() {
        assert_eq!(ExportFormat::Jupyter.default_extension(), "ipynb");
        assert_eq!(ExportFormat::LaTeX.mime_type(), "application/x-latex");
        assert!(ExportFormat::Html.description().contains("HTML"));
    }
    
    #[test]
    fn test_export_config_default() {
        let config = ExportConfig::default();
        assert!(config.include_timing);
        assert!(config.include_errors);
        assert!(config.syntax_highlighting);
        assert_eq!(config.math_rendering, MathRenderingStyle::LaTeX);
    }
    
    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::success();
        assert!(result.is_valid);
        
        result.add_warning("Test warning".to_string());
        assert!(result.is_valid); // warnings don't make it invalid
        
        result.add_error("Test error".to_string());
        assert!(!result.is_valid); // errors do make it invalid
        
        let report = result.format_report();
        assert!(report.contains("❌"));
        assert!(report.contains("Test warning"));
        assert!(report.contains("Test error"));
    }
    
    #[test]
    fn test_compression_settings() {
        let settings = CompressionSettings::default();
        assert!(!settings.enabled);
        assert_eq!(settings.algorithm, CompressionAlgorithm::Gzip);
        assert_eq!(settings.level, 6);
    }
    
    #[test]
    fn test_advanced_export_options() {
        let options = AdvancedExportOptions::default();
        assert!(options.progress_callback.is_none());
        assert!(!options.include_source_maps);
        assert_eq!(options.resource_handling, ResourceHandling::Embed);
    }
}