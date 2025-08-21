//! Jupyter Notebook Exporter
//! 
//! Exports Lyra REPL sessions to Jupyter notebook format (.ipynb)

use crate::repl::export::{
    SessionSnapshot, SessionEntry, CellType,
};
use crate::repl::export::engine::{
    ExportEngine, ExportFormat, ExportConfig, ExportResult, ExportError, ExportFeature,
};
use crate::vm::Value;
use std::path::PathBuf;
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;
use std::fs;

/// Jupyter notebook exporter
pub struct JupyterExporter {
    /// Template configuration
    template_config: JupyterTemplateConfig,
}

impl JupyterExporter {
    /// Create a new Jupyter exporter
    pub fn new() -> Self {
        Self {
            template_config: JupyterTemplateConfig::default(),
        }
    }
    
    /// Create exporter with custom template configuration
    pub fn with_template_config(template_config: JupyterTemplateConfig) -> Self {
        Self {
            template_config,
        }
    }
    
    /// Convert session snapshot to Jupyter notebook
    fn convert_to_notebook(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> ExportResult<JupyterNotebook> {
        let mut cells = Vec::new();
        
        // Add title cell if specified
        if let Some(ref title) = config.title {
            cells.push(JupyterCell {
                cell_type: JupyterCellType::Markdown,
                source: vec![format!("# {}", title)],
                metadata: JupyterCellMetadata::default(),
                execution_count: None,
                outputs: None,
                id: Some("title-cell".to_string()),
            });
        }
        
        // Add session metadata cell if enabled
        if config.include_metadata {
            let metadata_content = self.format_session_metadata(&snapshot.metadata, config);
            cells.push(JupyterCell {
                cell_type: JupyterCellType::Markdown,
                source: vec![metadata_content],
                metadata: JupyterCellMetadata::default(),
                execution_count: None,
                outputs: None,
                id: Some("metadata-cell".to_string()),
            });
        }
        
        // Convert session entries to cells
        for (index, entry) in snapshot.entries.iter().enumerate() {
            let cell = self.convert_entry_to_cell(entry, config, index)?;
            cells.push(cell);
        }
        
        // Add environment variables cell if enabled
        if config.include_environment && !snapshot.environment.is_empty() {
            let env_content = self.format_environment(&snapshot.environment);
            cells.push(JupyterCell {
                cell_type: JupyterCellType::Markdown,
                source: vec![env_content],
                metadata: JupyterCellMetadata::default(),
                execution_count: None,
                outputs: None,
                id: Some("environment-cell".to_string()),
            });
        }
        
        // Create notebook metadata
        let notebook_metadata = self.create_notebook_metadata(snapshot, config);
        
        Ok(JupyterNotebook {
            cells,
            metadata: notebook_metadata,
            nbformat: 4,
            nbformat_minor: 5,
        })
    }
    
    /// Convert a session entry to a Jupyter cell
    fn convert_entry_to_cell(
        &self,
        entry: &SessionEntry,
        config: &ExportConfig,
        index: usize,
    ) -> ExportResult<JupyterCell> {
        match entry.cell_type {
            CellType::Code => self.convert_code_cell(entry, config, index),
            CellType::Meta => self.convert_meta_cell(entry, config, index),
            CellType::Markdown => self.convert_markdown_cell(entry, config, index),
            CellType::Raw => self.convert_raw_cell(entry, config, index),
        }
    }
    
    /// Convert a code entry to a Jupyter code cell
    fn convert_code_cell(
        &self,
        entry: &SessionEntry,
        config: &ExportConfig,
        index: usize,
    ) -> ExportResult<JupyterCell> {
        let mut outputs = Vec::new();
        
        // Add execution result output
        if entry.has_output() {
            let output_text = self.format_output_value(&entry.output, config);
            outputs.push(JupyterOutput {
                output_type: "execute_result".to_string(),
                execution_count: entry.execution_count,
                data: {
                    let mut data = HashMap::new();
                    data.insert("text/plain".to_string(), serde_json::Value::Array(
                        output_text.lines().map(|line| serde_json::Value::String(line.to_string())).collect()
                    ));
                    data
                },
                metadata: HashMap::new(),
                text: None,
                name: None,
                ename: None,
                evalue: None,
                traceback: None,
            });
        }
        
        // Add error output if present
        if let Some(ref error) = entry.error {
            if config.include_errors {
                outputs.push(JupyterOutput {
                    output_type: "error".to_string(),
                    execution_count: None,
                    data: HashMap::new(),
                    metadata: HashMap::new(),
                    text: None,
                    name: None,
                    ename: Some("LyraError".to_string()),
                    evalue: Some(error.clone()),
                    traceback: Some(vec![error.clone()]),
                });
            }
        }
        
        // Add timing information as a comment
        let mut source_lines = vec![entry.input.clone()];
        if config.include_timing {
            if let Some(exec_time) = entry.execution_time {
                source_lines.push(format!("# Execution time: {}", 
                    entry.formatted_execution_time()));
            }
        }
        
        // Create cell metadata
        let mut cell_metadata = JupyterCellMetadata::default();
        if config.include_timing || config.include_metadata {
            let mut lyra_metadata = HashMap::new();
            
            if let Some(exec_time) = entry.execution_time {
                lyra_metadata.insert("execution_time_ms".to_string(), 
                    serde_json::Value::Real(serde_json::Number::from(exec_time.as_millis() as u64)));
            }
            
            lyra_metadata.insert("timestamp".to_string(), 
                serde_json::Value::String(entry.formatted_timestamp()));
            
            if entry.error.is_some() {
                lyra_metadata.insert("has_error".to_string(), serde_json::Value::Bool(true));
            }
            
            cell_metadata.lyra = Some(lyra_metadata);
        }
        
        Ok(JupyterCell {
            cell_type: JupyterCellType::Code,
            source: source_lines,
            metadata: cell_metadata,
            execution_count: entry.execution_count,
            outputs: Some(outputs),
            id: Some(entry.cell_id.clone()),
        })
    }
    
    /// Convert a meta command entry to a Jupyter cell
    fn convert_meta_cell(
        &self,
        entry: &SessionEntry,
        config: &ExportConfig,
        _index: usize,
    ) -> ExportResult<JupyterCell> {
        // Meta commands are converted to markdown cells with code formatting
        let content = format!(
            "**Meta Command:**\n\n```\n{}\n```\n\n**Output:**\n\n```\n{}\n```",
            entry.input,
            self.format_output_value(&entry.output, config)
        );
        
        let mut cell_metadata = JupyterCellMetadata::default();
        if config.include_metadata {
            let mut lyra_metadata = HashMap::new();
            lyra_metadata.insert("cell_type".to_string(), 
                serde_json::Value::String("meta".to_string()));
            lyra_metadata.insert("original_command".to_string(), 
                serde_json::Value::String(entry.input.clone()));
            cell_metadata.lyra = Some(lyra_metadata);
        }
        
        Ok(JupyterCell {
            cell_type: JupyterCellType::Markdown,
            source: vec![content],
            metadata: cell_metadata,
            execution_count: None,
            outputs: None,
            id: Some(entry.cell_id.clone()),
        })
    }
    
    /// Convert a markdown entry to a Jupyter markdown cell
    fn convert_markdown_cell(
        &self,
        entry: &SessionEntry,
        _config: &ExportConfig,
        _index: usize,
    ) -> ExportResult<JupyterCell> {
        Ok(JupyterCell {
            cell_type: JupyterCellType::Markdown,
            source: vec![entry.input.clone()],
            metadata: JupyterCellMetadata::default(),
            execution_count: None,
            outputs: None,
            id: Some(entry.cell_id.clone()),
        })
    }
    
    /// Convert a raw entry to a Jupyter raw cell
    fn convert_raw_cell(
        &self,
        entry: &SessionEntry,
        _config: &ExportConfig,
        _index: usize,
    ) -> ExportResult<JupyterCell> {
        Ok(JupyterCell {
            cell_type: JupyterCellType::Raw,
            source: vec![entry.input.clone()],
            metadata: JupyterCellMetadata::default(),
            execution_count: None,
            outputs: None,
            id: Some(entry.cell_id.clone()),
        })
    }
    
    /// Format a Value for display in Jupyter
    fn format_output_value(&self, value: &Value, config: &ExportConfig) -> String {
        let output_str = self.value_to_string(value);
        
        if let Some(max_length) = config.max_output_length {
            if output_str.len() > max_length {
                format!("{}...\n\n[Output truncated to {} characters]", 
                       &output_str[..max_length], max_length)
            } else {
                output_str
            }
        } else {
            output_str
        }
    }
    
    /// Convert Value to string representation
    fn value_to_string(&self, value: &Value) -> String {
        match value {
            Value::Integer(n) => n.to_string(),
            Value::Real(f) => f.to_string(),
            Value::String(s) => s.clone(),
            Value::Symbol(s) => s.clone(),
            Value::Boolean(b) => b.to_string(),
            Value::Missing => "Missing".to_string(),
            Value::Function(name) => format!("Function[{}]", name),
            Value::List(items) => {
                let items_str: Vec<String> = items.iter()
                    .map(|item| self.value_to_string(item))
                    .collect();
                format!("{{{}}}", items_str.join(", "))
            },
            Value::LyObj(obj) => format!("{}[...]", obj.type_name()),
            Value::Quote(expr) => format!("Hold[{:?}]", expr),
            Value::Pattern(pat) => format!("Pattern[{:?}]", pat),
            Value::Rule { lhs, rhs } => format!("{} -> {}", 
                self.value_to_string(lhs), 
                self.value_to_string(rhs)),
        }
    }
    
    /// Format session metadata for display
    fn format_session_metadata(
        &self,
        metadata: &crate::repl::export::SessionMetadata,
        _config: &ExportConfig,
    ) -> String {
        format!(
            "## Session Information\n\n\
             **Session ID:** {}\n\n\
             **Lyra Version:** {}\n\n\
             **Start Time:** {}\n\n\
             **End Time:** {}\n\n\
             **Duration:** {}\n\n\
             **Working Directory:** {}\n\n",
            metadata.session_id,
            metadata.lyra_version,
            metadata.formatted_start_time(),
            metadata.formatted_end_time(),
            metadata.formatted_duration(),
            metadata.working_directory.display()
        )
    }
    
    /// Format environment variables for display
    fn format_environment(&self, environment: &HashMap<String, Value>) -> String {
        let mut content = String::from("## Environment Variables\n\n");
        
        let mut vars: Vec<_> = environment.iter().collect();
        vars.sort_by_key(|(name, _)| *name);
        
        content.push_str("| Variable | Value | Type |\n");
        content.push_str("|----------|--------|------|\n");
        
        for (name, value) in vars {
            let value_str = self.value_to_string(value);
            let display_value = if value_str.len() > 50 {
                format!("{}...", &value_str[..47])
            } else {
                value_str
            };
            
            let type_name = match value {
                Value::Integer(_) => "Integer",
                Value::Real(_) => "Real",
                Value::String(_) => "String",
                Value::Boolean(_) => "Boolean",
                Value::List(_) => "List",
                Value::Symbol(_) => "Symbol",
                Value::Function(_) => "Function",
                Value::Missing => "Missing",
                Value::LyObj(_) => "LyObj",
                Value::Quote(_) => "Quote",
                Value::Pattern(_) => "Pattern", 
                Value::Rule { .. } => "Rule",
            };
            
            content.push_str(&format!("| `{}` | `{}` | {} |\n", name, display_value, type_name));
        }
        
        content
    }
    
    /// Create notebook-level metadata
    fn create_notebook_metadata(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> JupyterMetadata {
        let mut metadata = JupyterMetadata {
            kernelspec: JupyterKernelSpec {
                display_name: "Lyra".to_string(),
                language: "lyra".to_string(),
                name: "lyra".to_string(),
            },
            language_info: JupyterLanguageInfo {
                name: "lyra".to_string(),
                version: snapshot.metadata.lyra_version.clone(),
                mimetype: "text/x-lyra".to_string(),
                file_extension: ".lyra".to_string(),
                pygments_lexer: None,
                codemirror_mode: Some("lyra".to_string()),
                nbconvert_exporter: None,
            },
            lyra_export: None,
        };
        
        // Add Lyra-specific export metadata
        if config.include_metadata {
            let stats = snapshot.statistics();
            metadata.lyra_export = Some(LyraExportMetadata {
                export_version: "1.0".to_string(),
                session_id: snapshot.metadata.session_id.clone(),
                export_timestamp: chrono::Utc::now().to_rfc3339(),
                original_session_start: snapshot.metadata.formatted_start_time(),
                original_session_end: snapshot.metadata.formatted_end_time(),
                total_cells: stats.total_entries,
                code_cells: stats.code_entries,
                meta_cells: stats.meta_entries,
                error_cells: stats.error_entries,
                success_rate: stats.success_rate,
                total_execution_time_ms: stats.total_execution_time.as_millis() as u64,
                lyra_version: snapshot.metadata.lyra_version.clone(),
                export_config: ExportConfigSummary {
                    include_timing: config.include_timing,
                    include_errors: config.include_errors,
                    include_metadata: config.include_metadata,
                    include_environment: config.include_environment,
                    syntax_highlighting: config.syntax_highlighting,
                },
            });
        }
        
        metadata
    }
}

impl Default for JupyterExporter {
    fn default() -> Self {
        Self::new()
    }
}

impl ExportEngine for JupyterExporter {
    fn export(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
        output_path: &PathBuf,
    ) -> ExportResult<()> {
        let notebook = self.convert_to_notebook(snapshot, config)?;
        let json = serde_json::to_string_pretty(&notebook)
            .map_err(|e| ExportError::Serialization { 
                message: format!("Failed to serialize notebook: {}", e) 
            })?;
        
        fs::write(output_path, json)
            .map_err(|e| ExportError::Io { source: e })?;
        
        Ok(())
    }
    
    fn preview(
        &self,
        snapshot: &SessionSnapshot,
        config: &ExportConfig,
    ) -> ExportResult<String> {
        let notebook = self.convert_to_notebook(snapshot, config)?;
        serde_json::to_string_pretty(&notebook)
            .map_err(|e| ExportError::Serialization { 
                message: format!("Failed to serialize notebook preview: {}", e) 
            })
    }
    
    fn validate_config(&self, config: &ExportConfig) -> ExportResult<()> {
        // Check if custom template exists
        if let Some(ref template_path) = config.custom_template {
            if !template_path.exists() {
                return Err(ExportError::Configuration {
                    message: format!("Custom template not found: {}", template_path.display()),
                });
            }
        }
        
        // Validate max output length
        if let Some(max_length) = config.max_output_length {
            if max_length == 0 {
                return Err(ExportError::Configuration {
                    message: "max_output_length cannot be zero".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    fn default_extension(&self) -> &'static str {
        "ipynb"
    }
    
    fn mime_type(&self) -> &'static str {
        "application/x-ipynb+json"
    }
    
    fn description(&self) -> &'static str {
        "Jupyter Notebook with interactive code and output cells"
    }
    
    fn supports_feature(&self, feature: ExportFeature) -> bool {
        match feature {
            ExportFeature::SyntaxHighlighting => true,
            ExportFeature::MathRendering => true,
            ExportFeature::EmbeddedImages => true,
            ExportFeature::Interactive => true,
            ExportFeature::Metadata => true,
            ExportFeature::PerformanceData => true,
            ExportFeature::ErrorDetails => true,
            ExportFeature::CrossReferences => false,
            ExportFeature::CustomStyling => true,
            ExportFeature::TableOfContents => false,
        }
    }
    
    fn config_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "include_timing": {
                    "type": "boolean",
                    "description": "Include execution timing information"
                },
                "include_errors": {
                    "type": "boolean", 
                    "description": "Include error details in output"
                },
                "include_metadata": {
                    "type": "boolean",
                    "description": "Include session metadata"
                },
                "include_environment": {
                    "type": "boolean",
                    "description": "Include environment variables"
                },
                "max_output_length": {
                    "type": ["integer", "null"],
                    "description": "Maximum output length per cell (null for unlimited)"
                },
                "title": {
                    "type": ["string", "null"],
                    "description": "Notebook title"
                },
                "author": {
                    "type": ["string", "null"],
                    "description": "Author name"
                }
            }
        })
    }
}

/// Configuration for Jupyter template customization
#[derive(Debug, Clone)]
pub struct JupyterTemplateConfig {
    /// Custom cell templates
    pub cell_templates: HashMap<CellType, String>,
    /// Custom CSS for styling
    pub custom_css: Option<String>,
    /// Additional notebook metadata
    pub additional_metadata: HashMap<String, serde_json::Value>,
}

impl Default for JupyterTemplateConfig {
    fn default() -> Self {
        Self {
            cell_templates: HashMap::new(),
            custom_css: None,
            additional_metadata: HashMap::new(),
        }
    }
}

/// Jupyter notebook structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterNotebook {
    /// List of cells
    pub cells: Vec<JupyterCell>,
    /// Notebook metadata
    pub metadata: JupyterMetadata,
    /// Notebook format version
    pub nbformat: u32,
    /// Notebook format minor version
    pub nbformat_minor: u32,
}

/// Jupyter notebook cell
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterCell {
    /// Cell type
    pub cell_type: JupyterCellType,
    /// Cell source code/content
    pub source: Vec<String>,
    /// Cell metadata
    pub metadata: JupyterCellMetadata,
    /// Execution count (code cells only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_count: Option<usize>,
    /// Cell outputs (code cells only)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Vec<JupyterOutput>>,
    /// Cell ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

/// Jupyter cell types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JupyterCellType {
    Code,
    Markdown,
    Raw,
}

/// Jupyter cell metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterCellMetadata {
    /// Lyra-specific metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lyra: Option<HashMap<String, serde_json::Value>>,
    /// Additional metadata
    #[serde(flatten)]
    pub additional: HashMap<String, serde_json::Value>,
}

impl Default for JupyterCellMetadata {
    fn default() -> Self {
        Self {
            lyra: None,
            additional: HashMap::new(),
        }
    }
}

/// Jupyter cell output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterOutput {
    /// Output type
    pub output_type: String,
    /// Execution count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_count: Option<usize>,
    /// Output data
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub data: HashMap<String, serde_json::Value>,
    /// Output metadata
    #[serde(skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Text output (for stream outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<Vec<String>>,
    /// Stream name (for stream outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Error name (for error outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ename: Option<String>,
    /// Error value (for error outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evalue: Option<String>,
    /// Error traceback (for error outputs)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub traceback: Option<Vec<String>>,
}

/// Jupyter notebook metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterMetadata {
    /// Kernel specification
    pub kernelspec: JupyterKernelSpec,
    /// Language information
    pub language_info: JupyterLanguageInfo,
    /// Lyra export metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lyra_export: Option<LyraExportMetadata>,
}

/// Jupyter kernel specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterKernelSpec {
    /// Display name
    pub display_name: String,
    /// Language name
    pub language: String,
    /// Kernel name
    pub name: String,
}

/// Jupyter language information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JupyterLanguageInfo {
    /// Language name
    pub name: String,
    /// Language version
    pub version: String,
    /// MIME type
    pub mimetype: String,
    /// File extension
    pub file_extension: String,
    /// Pygments lexer
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pygments_lexer: Option<String>,
    /// CodeMirror mode
    #[serde(skip_serializing_if = "Option::is_none")]
    pub codemirror_mode: Option<String>,
    /// NBConvert exporter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nbconvert_exporter: Option<String>,
}

/// Lyra-specific export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LyraExportMetadata {
    /// Export format version
    pub export_version: String,
    /// Original session ID
    pub session_id: String,
    /// Export timestamp
    pub export_timestamp: String,
    /// Original session start time
    pub original_session_start: String,
    /// Original session end time
    pub original_session_end: String,
    /// Total number of cells
    pub total_cells: usize,
    /// Number of code cells
    pub code_cells: usize,
    /// Number of meta cells
    pub meta_cells: usize,
    /// Number of error cells
    pub error_cells: usize,
    /// Success rate percentage
    pub success_rate: f64,
    /// Total execution time in milliseconds
    pub total_execution_time_ms: u64,
    /// Lyra version used
    pub lyra_version: String,
    /// Export configuration summary
    pub export_config: ExportConfigSummary,
}

/// Summary of export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfigSummary {
    /// Include timing information
    pub include_timing: bool,
    /// Include errors
    pub include_errors: bool,
    /// Include metadata
    pub include_metadata: bool,
    /// Include environment
    pub include_environment: bool,
    /// Syntax highlighting enabled
    pub syntax_highlighting: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::repl::export::session_snapshot::*;
    use std::time::SystemTime;
    
    fn create_test_snapshot() -> SessionSnapshot {
        let mut snapshot = SessionSnapshot::new(SessionMetadata::new(
            "test-session".to_string(),
            "0.1.0".to_string(),
        ));
        
        let entry = SessionEntry::new(
            "cell-1".to_string(),
            CellType::Code,
            "x = 5 + 3".to_string(),
            Value::Integer(8),
        );
        
        snapshot.add_entry(entry);
        snapshot
    }
    
    #[test]
    fn test_jupyter_exporter_creation() {
        let exporter = JupyterExporter::new();
        assert_eq!(exporter.default_extension(), "ipynb");
        assert_eq!(exporter.mime_type(), "application/x-ipynb+json");
    }
    
    #[test]
    fn test_supports_features() {
        let exporter = JupyterExporter::new();
        assert!(exporter.supports_feature(ExportFeature::SyntaxHighlighting));
        assert!(exporter.supports_feature(ExportFeature::MathRendering));
        assert!(exporter.supports_feature(ExportFeature::Interactive));
        assert!(!exporter.supports_feature(ExportFeature::CrossReferences));
    }
    
    #[test]
    fn test_convert_to_notebook() {
        let exporter = JupyterExporter::new();
        let snapshot = create_test_snapshot();
        let config = ExportConfig::default();
        
        let notebook = exporter.convert_to_notebook(&snapshot, &config).unwrap();
        
        assert_eq!(notebook.nbformat, 4);
        assert_eq!(notebook.nbformat_minor, 5);
        assert!(!notebook.cells.is_empty());
        assert_eq!(notebook.metadata.language_info.name, "lyra");
    }
    
    #[test]
    fn test_convert_code_cell() {
        let exporter = JupyterExporter::new();
        let entry = SessionEntry::new(
            "cell-1".to_string(),
            CellType::Code,
            "x = 42".to_string(),
            Value::Integer(42),
        );
        let config = ExportConfig::default();
        
        let cell = exporter.convert_code_cell(&entry, &config, 0).unwrap();
        
        assert!(matches!(cell.cell_type, JupyterCellType::Code));
        assert_eq!(cell.source[0], "x = 42");
        assert!(cell.outputs.is_some());
        assert_eq!(cell.id, Some("cell-1".to_string()));
    }
    
    #[test]
    fn test_convert_meta_cell() {
        let exporter = JupyterExporter::new();
        let entry = SessionEntry::new(
            "cell-1".to_string(),
            CellType::Meta,
            "%help".to_string(),
            Value::String("Help text".to_string()),
        );
        let config = ExportConfig::default();
        
        let cell = exporter.convert_meta_cell(&entry, &config, 0).unwrap();
        
        assert!(matches!(cell.cell_type, JupyterCellType::Markdown));
        assert!(cell.source[0].contains("Meta Command"));
        assert!(cell.source[0].contains("%help"));
    }
    
    #[test]
    fn test_format_output_value() {
        let exporter = JupyterExporter::new();
        let config = ExportConfig::default();
        
        let short_output = exporter.format_output_value(&Value::Integer(42), &config);
        assert_eq!(short_output, "42");
        
        let long_string = "a".repeat(20000);
        let mut truncated_config = config.clone();
        truncated_config.max_output_length = Some(100);
        let truncated_output = exporter.format_output_value(&Value::String(long_string), &truncated_config);
        assert!(truncated_output.contains("..."));
        assert!(truncated_output.contains("[Output truncated"));
    }
    
    #[test]
    fn test_preview() {
        let exporter = JupyterExporter::new();
        let snapshot = create_test_snapshot();
        let config = ExportConfig::default();
        
        let preview = exporter.preview(&snapshot, &config).unwrap();
        assert!(preview.contains("\"nbformat\": 4"));
        assert!(preview.contains("\"cells\""));
        assert!(preview.contains("\"metadata\""));
    }
    
    #[test]
    fn test_validate_config() {
        let exporter = JupyterExporter::new();
        
        // Valid config
        let config = ExportConfig::default();
        assert!(exporter.validate_config(&config).is_ok());
        
        // Invalid config - zero max output length
        let mut invalid_config = config.clone();
        invalid_config.max_output_length = Some(0);
        assert!(exporter.validate_config(&invalid_config).is_err());
    }
    
    #[test]
    fn test_config_schema() {
        let exporter = JupyterExporter::new();
        let schema = exporter.config_schema();
        
        assert!(schema.is_object());
        assert!(schema["properties"]["include_timing"]["type"] == "boolean");
        assert!(schema["properties"]["max_output_length"]["type"].as_array().unwrap().contains(&serde_json::Value::String("integer".to_string())));
    }
    
    #[test]
    fn test_error_cell_handling() {
        let exporter = JupyterExporter::new();
        let mut entry = SessionEntry::new(
            "cell-1".to_string(),
            CellType::Code,
            "invalid syntax".to_string(),
            Value::String("".to_string()),
        );
        entry.error = Some("Parse error: unexpected token".to_string());
        
        let config = ExportConfig::default();
        let cell = exporter.convert_code_cell(&entry, &config, 0).unwrap();
        
        assert!(cell.outputs.is_some());
        let outputs = cell.outputs.unwrap();
        assert!(outputs.iter().any(|output| output.output_type == "error"));
    }
    
    #[test]
    fn test_metadata_inclusion() {
        let exporter = JupyterExporter::new();
        let snapshot = create_test_snapshot();
        let mut config = ExportConfig::default();
        config.include_metadata = true;
        config.title = Some("Test Notebook".to_string());
        
        let notebook = exporter.convert_to_notebook(&snapshot, &config).unwrap();
        
        // Should have title cell and metadata cell
        let markdown_cells: Vec<_> = notebook.cells.iter()
            .filter(|cell| matches!(cell.cell_type, JupyterCellType::Markdown))
            .collect();
        
        assert!(markdown_cells.len() >= 2); // title + metadata
        assert!(notebook.metadata.lyra_export.is_some());
    }
}