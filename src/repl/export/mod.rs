//! Export Module
//! 
//! Provides comprehensive export capabilities for Lyra REPL sessions
//! Supports multiple formats: Jupyter notebooks, LaTeX, HTML, and more

pub mod engine;
pub mod html;
pub mod jupyter;
pub mod latex;
pub mod session_snapshot;

pub use engine::{ExportEngine, ExportFormat, ExportConfig, ExportResult, ExportError, ExportFeature};
pub use html::HtmlExporter;
pub use jupyter::{JupyterExporter, JupyterCell, JupyterMetadata, JupyterNotebook};
pub use latex::LaTeXExporter;
pub use session_snapshot::{SessionSnapshot, SessionEntry, SessionMetadata, CellType};

use crate::repl::history::HistoryEntry;
use crate::vm::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

/// Unified export manager that handles all export formats
pub struct ExportManager {
    /// Export configuration
    config: ExportConfig,
    /// Registered export engines
    engines: HashMap<ExportFormat, Box<dyn ExportEngine>>,
}

impl ExportManager {
    /// Create a new export manager with default configuration
    pub fn new() -> Self {
        let mut manager = Self {
            config: ExportConfig::default(),
            engines: HashMap::new(),
        };
        
        // Register default exporters
        manager.register_engine(ExportFormat::Jupyter, Box::new(JupyterExporter::new()));
        manager.register_engine(ExportFormat::LaTeX, Box::new(LaTeXExporter::new()));
        manager.register_engine(ExportFormat::Html, Box::new(HtmlExporter::new()));
        
        manager
    }
    
    /// Register a new export engine
    pub fn register_engine(&mut self, format: ExportFormat, engine: Box<dyn ExportEngine>) {
        self.engines.insert(format, engine);
    }
    
    /// Export session to specified format
    pub fn export_session(
        &self,
        snapshot: &SessionSnapshot,
        format: ExportFormat,
        output_path: &PathBuf,
    ) -> ExportResult<()> {
        if let Some(engine) = self.engines.get(&format) {
            engine.export(snapshot, &self.config, output_path)
        } else {
            Err(ExportError::UnsupportedFormat { format })
        }
    }
    
    /// Get list of supported export formats
    pub fn supported_formats(&self) -> Vec<ExportFormat> {
        self.engines.keys().cloned().collect()
    }
    
    /// Update export configuration
    pub fn update_config(&mut self, config: ExportConfig) {
        self.config = config;
    }
    
    /// Get current export configuration
    pub fn config(&self) -> &ExportConfig {
        &self.config
    }
    
    /// Create session snapshot from REPL history and state
    pub fn create_snapshot(
        &self,
        history: &[HistoryEntry],
        environment: &HashMap<String, Value>,
        metadata: SessionMetadata,
    ) -> SessionSnapshot {
        let entries = history
            .iter()
            .enumerate()
            .map(|(index, entry)| SessionEntry {
                cell_id: format!("cell-{}", index),
                cell_type: if entry.input.trim().starts_with('%') {
                    CellType::Meta
                } else {
                    CellType::Code
                },
                input: entry.input.clone(),
                output: Value::String(entry.output.clone()),
                execution_count: Some(index + 1),
                timestamp: entry.timestamp,
                execution_time: Some(entry.execution_time),
                error: None, // HistoryEntry doesn't have error field, will be None
            })
            .collect();
        
        SessionSnapshot {
            entries,
            environment: environment.clone(),
            metadata,
        }
    }
    
    /// Preview export without writing to file
    pub fn preview_export(
        &self,
        snapshot: &SessionSnapshot,
        format: ExportFormat,
    ) -> ExportResult<String> {
        if let Some(engine) = self.engines.get(&format) {
            engine.preview(snapshot, &self.config)
        } else {
            Err(ExportError::UnsupportedFormat { format })
        }
    }
    
    /// Validate export configuration
    pub fn validate_config(&self, format: ExportFormat) -> ExportResult<()> {
        if let Some(engine) = self.engines.get(&format) {
            engine.validate_config(&self.config)
        } else {
            Err(ExportError::UnsupportedFormat { format })
        }
    }
    
    /// Get export statistics
    pub fn export_stats(&self, snapshot: &SessionSnapshot) -> ExportStats {
        ExportStats {
            total_cells: snapshot.entries.len(),
            code_cells: snapshot.entries.iter().filter(|e| matches!(e.cell_type, CellType::Code)).count(),
            meta_cells: snapshot.entries.iter().filter(|e| matches!(e.cell_type, CellType::Meta)).count(),
            error_cells: snapshot.entries.iter().filter(|e| e.error.is_some()).count(),
            total_execution_time: snapshot.entries.iter()
                .filter_map(|e| e.execution_time)
                .sum(),
            session_duration: snapshot.metadata.end_time
                .duration_since(snapshot.metadata.start_time)
                .unwrap_or_default(),
            environment_variables: snapshot.environment.len(),
            lyra_version: snapshot.metadata.lyra_version.clone(),
        }
    }
}

impl Default for ExportManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about an export session
#[derive(Debug, Clone)]
pub struct ExportStats {
    /// Total number of cells
    pub total_cells: usize,
    /// Number of code cells
    pub code_cells: usize,
    /// Number of meta command cells
    pub meta_cells: usize,
    /// Number of cells with errors
    pub error_cells: usize,
    /// Total execution time for all cells
    pub total_execution_time: Duration,
    /// Duration of the entire session
    pub session_duration: Duration,
    /// Number of environment variables
    pub environment_variables: usize,
    /// Lyra version used
    pub lyra_version: String,
}

impl ExportStats {
    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_cells == 0 {
            100.0
        } else {
            ((self.total_cells - self.error_cells) as f64 / self.total_cells as f64) * 100.0
        }
    }
    
    /// Get average execution time per cell
    pub fn average_execution_time(&self) -> Duration {
        if self.code_cells == 0 {
            Duration::ZERO
        } else {
            self.total_execution_time / self.code_cells as u32
        }
    }
    
    /// Format statistics as human-readable string
    pub fn format_summary(&self) -> String {
        format!(
            "Export Summary:\n\
             • Total cells: {}\n\
             • Code cells: {} | Meta cells: {}\n\
             • Success rate: {:.1}%\n\
             • Session duration: {:.2}s\n\
             • Average execution time: {:.3}ms\n\
             • Environment variables: {}\n\
             • Lyra version: {}",
            self.total_cells,
            self.code_cells,
            self.meta_cells,
            self.success_rate(),
            self.session_duration.as_secs_f64(),
            self.average_execution_time().as_millis(),
            self.environment_variables,
            self.lyra_version
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;
    
    fn create_test_snapshot() -> SessionSnapshot {
        SessionSnapshot {
            entries: vec![
                SessionEntry {
                    cell_id: "cell-0".to_string(),
                    cell_type: CellType::Code,
                    input: "x = 5 + 3".to_string(),
                    output: Value::Integer(8),
                    execution_count: Some(1),
                    timestamp: SystemTime::now(),
                    execution_time: Some(Duration::from_millis(10)),
                    error: None,
                },
                SessionEntry {
                    cell_id: "cell-1".to_string(),
                    cell_type: CellType::Meta,
                    input: "%help".to_string(),
                    output: Value::String("Help text".to_string()),
                    execution_count: Some(2),
                    timestamp: SystemTime::now(),
                    execution_time: Some(Duration::from_millis(5)),
                    error: None,
                },
            ],
            environment: {
                let mut env = HashMap::new();
                env.insert("x".to_string(), Value::Integer(8));
                env
            },
            metadata: SessionMetadata {
                session_id: "test-session".to_string(),
                start_time: SystemTime::now(),
                end_time: SystemTime::now(),
                lyra_version: "0.1.0".to_string(),
                user: Some("test-user".to_string()),
                working_directory: PathBuf::from("/test"),
                configuration: HashMap::new(),
            },
        }
    }
    
    #[test]
    fn test_export_manager_creation() {
        let manager = ExportManager::new();
        assert!(manager.supported_formats().contains(&ExportFormat::Jupyter));
    }
    
    #[test]
    fn test_create_snapshot() {
        let manager = ExportManager::new();
        let history = vec![
            HistoryEntry {
                input: "x = 42".to_string(),
                output: Some(Value::Integer(42)),
                timestamp: SystemTime::now(),
                execution_time: Some(Duration::from_millis(10)),
                error: None,
            }
        ];
        let environment = HashMap::new();
        let metadata = SessionMetadata {
            session_id: "test".to_string(),
            start_time: SystemTime::now(),
            end_time: SystemTime::now(),
            lyra_version: "0.1.0".to_string(),
            user: None,
            working_directory: PathBuf::new(),
            configuration: HashMap::new(),
        };
        
        let snapshot = manager.create_snapshot(&history, &environment, metadata);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].input, "x = 42");
    }
    
    #[test]
    fn test_export_stats() {
        let manager = ExportManager::new();
        let snapshot = create_test_snapshot();
        let stats = manager.export_stats(&snapshot);
        
        assert_eq!(stats.total_cells, 2);
        assert_eq!(stats.code_cells, 1);
        assert_eq!(stats.meta_cells, 1);
        assert_eq!(stats.error_cells, 0);
        assert_eq!(stats.success_rate(), 100.0);
        assert_eq!(stats.environment_variables, 1);
    }
    
    #[test]
    fn test_export_stats_with_errors() {
        let mut snapshot = create_test_snapshot();
        snapshot.entries[0].error = Some("Test error".to_string());
        
        let manager = ExportManager::new();
        let stats = manager.export_stats(&snapshot);
        
        assert_eq!(stats.error_cells, 1);
        assert_eq!(stats.success_rate(), 50.0);
    }
    
    #[test]
    fn test_stats_formatting() {
        let manager = ExportManager::new();
        let snapshot = create_test_snapshot();
        let stats = manager.export_stats(&snapshot);
        let summary = stats.format_summary();
        
        assert!(summary.contains("Total cells: 2"));
        assert!(summary.contains("Success rate: 100.0%"));
        assert!(summary.contains("Environment variables: 1"));
    }
}