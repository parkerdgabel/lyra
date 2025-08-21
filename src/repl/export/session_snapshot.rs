//! Session Snapshot Module
//! 
//! Captures complete REPL session state for export operations

use crate::vm::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, TimeZone};

/// Complete snapshot of a REPL session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSnapshot {
    /// All session entries (input/output pairs)
    pub entries: Vec<SessionEntry>,
    /// Current environment variables
    pub environment: HashMap<String, Value>,
    /// Session metadata
    pub metadata: SessionMetadata,
}

impl SessionSnapshot {
    /// Create a new empty session snapshot
    pub fn new(metadata: SessionMetadata) -> Self {
        Self {
            entries: Vec::new(),
            environment: HashMap::new(),
            metadata,
        }
    }
    
    /// Add a new entry to the session
    pub fn add_entry(&mut self, entry: SessionEntry) {
        self.entries.push(entry);
    }
    
    /// Update the environment
    pub fn update_environment(&mut self, environment: HashMap<String, Value>) {
        self.environment = environment;
    }
    
    /// Get entries by type
    pub fn entries_by_type(&self, cell_type: CellType) -> Vec<&SessionEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.cell_type == cell_type)
            .collect()
    }
    
    /// Get entries with errors
    pub fn error_entries(&self) -> Vec<&SessionEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.error.is_some())
            .collect()
    }
    
    /// Get total session duration
    pub fn total_duration(&self) -> Duration {
        self.metadata.end_time
            .duration_since(self.metadata.start_time)
            .unwrap_or_default()
    }
    
    /// Get total execution time for all code cells
    pub fn total_execution_time(&self) -> Duration {
        self.entries
            .iter()
            .filter(|entry| matches!(entry.cell_type, CellType::Code))
            .filter_map(|entry| entry.execution_time)
            .sum()
    }
    
    /// Count entries by type
    pub fn count_by_type(&self, cell_type: CellType) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.cell_type == cell_type)
            .count()
    }
    
    /// Get session statistics
    pub fn statistics(&self) -> SessionStatistics {
        let total_entries = self.entries.len();
        let code_entries = self.count_by_type(CellType::Code);
        let meta_entries = self.count_by_type(CellType::Meta);
        let error_entries = self.error_entries().len();
        let success_entries = total_entries - error_entries;
        
        SessionStatistics {
            total_entries,
            code_entries,
            meta_entries,
            error_entries,
            success_entries,
            success_rate: if total_entries > 0 {
                (success_entries as f64 / total_entries as f64) * 100.0
            } else {
                100.0
            },
            total_duration: self.total_duration(),
            total_execution_time: self.total_execution_time(),
            average_execution_time: if code_entries > 0 {
                self.total_execution_time() / code_entries as u32
            } else {
                Duration::ZERO
            },
            environment_variables: self.environment.len(),
        }
    }
    
    /// Validate snapshot integrity
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult {
            is_valid: true,
            issues: Vec::new(),
        };
        
        // Check for duplicate cell IDs
        let mut cell_ids = std::collections::HashSet::new();
        for entry in &self.entries {
            if cell_ids.contains(&entry.cell_id) {
                result.issues.push(ValidationIssue {
                    level: ValidationLevel::Error,
                    message: format!("Duplicate cell ID: {}", entry.cell_id),
                    suggestion: Some("Ensure all cell IDs are unique".to_string()),
                });
                result.is_valid = false;
            }
            cell_ids.insert(&entry.cell_id);
        }
        
        // Check for invalid timestamps
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.timestamp < self.metadata.start_time {
                result.issues.push(ValidationIssue {
                    level: ValidationLevel::Warning,
                    message: format!("Entry {} timestamp before session start", i),
                    suggestion: Some("Check timestamp accuracy".to_string()),
                });
            }
        }
        
        // Check for excessive execution times
        for entry in &self.entries {
            if let Some(exec_time) = entry.execution_time {
                if exec_time > Duration::from_secs(300) { // 5 minutes
                    result.issues.push(ValidationIssue {
                        level: ValidationLevel::Warning,
                        message: format!("Very long execution time in cell {}: {:.2}s", 
                                       entry.cell_id, exec_time.as_secs_f64()),
                        suggestion: Some("Consider performance optimization".to_string()),
                    });
                }
            }
        }
        
        // Check session duration sanity
        if self.total_duration() > Duration::from_secs(86400) { // 24 hours
            result.issues.push(ValidationIssue {
                level: ValidationLevel::Warning,
                message: "Session duration exceeds 24 hours".to_string(),
                suggestion: Some("Consider breaking into multiple sessions".to_string()),
            });
        }
        
        result
    }
    
    /// Filter entries by time range
    pub fn filter_by_time_range(
        &self,
        start: SystemTime,
        end: SystemTime,
    ) -> Vec<&SessionEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.timestamp >= start && entry.timestamp <= end)
            .collect()
    }
    
    /// Get entry by cell ID
    pub fn get_entry_by_id(&self, cell_id: &str) -> Option<&SessionEntry> {
        self.entries
            .iter()
            .find(|entry| entry.cell_id == cell_id)
    }
    
    /// Merge with another session snapshot
    pub fn merge(&mut self, other: SessionSnapshot) -> Result<(), String> {
        // Validate that sessions can be merged
        if self.metadata.session_id == other.metadata.session_id {
            return Err("Cannot merge sessions with the same ID".to_string());
        }
        
        // Check for overlapping time ranges
        if self.metadata.start_time < other.metadata.end_time &&
           other.metadata.start_time < self.metadata.end_time {
            return Err("Sessions have overlapping time ranges".to_string());
        }
        
        // Merge entries with updated cell IDs to avoid conflicts
        let offset = self.entries.len();
        for (i, mut entry) in other.entries.into_iter().enumerate() {
            entry.cell_id = format!("merged-{}-{}", offset + i, entry.cell_id);
            if let Some(ref mut exec_count) = entry.execution_count {
                *exec_count += offset;
            }
            self.entries.push(entry);
        }
        
        // Merge environments (other session takes precedence for conflicts)
        for (key, value) in other.environment {
            self.environment.insert(key, value);
        }
        
        // Update metadata
        self.metadata.end_time = self.metadata.end_time.max(other.metadata.end_time);
        self.metadata.session_id = format!("{}-merged-{}", 
                                          self.metadata.session_id, 
                                          other.metadata.session_id);
        
        Ok(())
    }
}

/// Individual session entry (input/output pair)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    /// Unique identifier for this cell
    pub cell_id: String,
    /// Type of cell (code, meta command, etc.)
    pub cell_type: CellType,
    /// Input text/code
    pub input: String,
    /// Output value
    pub output: Value,
    /// Execution count (for Jupyter compatibility)
    pub execution_count: Option<usize>,
    /// Timestamp when this entry was created
    pub timestamp: SystemTime,
    /// Time taken to execute (if applicable)
    pub execution_time: Option<Duration>,
    /// Error message if execution failed
    pub error: Option<String>,
}

impl SessionEntry {
    /// Create a new session entry
    pub fn new(
        cell_id: String,
        cell_type: CellType,
        input: String,
        output: Value,
    ) -> Self {
        Self {
            cell_id,
            cell_type,
            input,
            output,
            execution_count: None,
            timestamp: SystemTime::now(),
            execution_time: None,
            error: None,
        }
    }
    
    /// Check if this entry represents a successful execution
    pub fn is_success(&self) -> bool {
        self.error.is_none()
    }
    
    /// Check if this entry has output
    pub fn has_output(&self) -> bool {
        !matches!(self.output, Value::String(ref s) if s.is_empty())
    }
    
    /// Get formatted timestamp
    pub fn formatted_timestamp(&self) -> String {
        if let Ok(datetime) = self.timestamp.duration_since(SystemTime::UNIX_EPOCH) {
            let dt = Utc.timestamp_opt(datetime.as_secs() as i64, 0)
                .single()
                .unwrap_or_else(|| Utc::now());
            dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
        } else {
            "Unknown time".to_string()
        }
    }
    
    /// Get execution time as formatted string
    pub fn formatted_execution_time(&self) -> String {
        if let Some(duration) = self.execution_time {
            if duration.as_millis() < 1000 {
                format!("{:.1}ms", duration.as_millis() as f64)
            } else {
                format!("{:.2}s", duration.as_secs_f64())
            }
        } else {
            "N/A".to_string()
        }
    }
    
    /// Get truncated output for preview
    pub fn truncated_output(&self, max_length: usize) -> String {
        let output_str = self.value_to_string(&self.output);
        if output_str.len() <= max_length {
            output_str
        } else {
            format!("{}...", &output_str[..max_length])
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
}

/// Type of session entry
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CellType {
    /// Regular code execution
    Code,
    /// Meta command (starts with %)
    Meta,
    /// Markdown/documentation cell
    Markdown,
    /// Raw text
    Raw,
}

impl std::fmt::Display for CellType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellType::Code => write!(f, "code"),
            CellType::Meta => write!(f, "meta"),
            CellType::Markdown => write!(f, "markdown"),
            CellType::Raw => write!(f, "raw"),
        }
    }
}

/// Session metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetadata {
    /// Unique session identifier
    pub session_id: String,
    /// Session start time
    pub start_time: SystemTime,
    /// Session end time
    pub end_time: SystemTime,
    /// Lyra version used
    pub lyra_version: String,
    /// User information (if available)
    pub user: Option<String>,
    /// Working directory
    pub working_directory: PathBuf,
    /// Configuration used during session
    pub configuration: HashMap<String, serde_json::Value>,
}

impl SessionMetadata {
    /// Create new session metadata
    pub fn new(session_id: String, lyra_version: String) -> Self {
        let now = SystemTime::now();
        Self {
            session_id,
            start_time: now,
            end_time: now,
            lyra_version,
            user: None,
            working_directory: std::env::current_dir().unwrap_or_default(),
            configuration: HashMap::new(),
        }
    }
    
    /// Update end time to current time
    pub fn update_end_time(&mut self) {
        self.end_time = SystemTime::now();
    }
    
    /// Get session duration
    pub fn duration(&self) -> Duration {
        self.end_time
            .duration_since(self.start_time)
            .unwrap_or_default()
    }
    
    /// Get formatted start time
    pub fn formatted_start_time(&self) -> String {
        format_system_time(self.start_time)
    }
    
    /// Get formatted end time
    pub fn formatted_end_time(&self) -> String {
        format_system_time(self.end_time)
    }
    
    /// Get formatted duration
    pub fn formatted_duration(&self) -> String {
        let duration = self.duration();
        let hours = duration.as_secs() / 3600;
        let minutes = (duration.as_secs() % 3600) / 60;
        let seconds = duration.as_secs() % 60;
        
        if hours > 0 {
            format!("{}h {}m {}s", hours, minutes, seconds)
        } else if minutes > 0 {
            format!("{}m {}s", minutes, seconds)
        } else {
            format!("{}.{:03}s", seconds, duration.subsec_millis())
        }
    }
}

/// Session statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionStatistics {
    /// Total number of entries
    pub total_entries: usize,
    /// Number of code entries
    pub code_entries: usize,
    /// Number of meta command entries
    pub meta_entries: usize,
    /// Number of entries with errors
    pub error_entries: usize,
    /// Number of successful entries
    pub success_entries: usize,
    /// Success rate as percentage
    pub success_rate: f64,
    /// Total session duration
    pub total_duration: Duration,
    /// Total execution time for code cells
    pub total_execution_time: Duration,
    /// Average execution time per code cell
    pub average_execution_time: Duration,
    /// Number of environment variables
    pub environment_variables: usize,
}

impl SessionStatistics {
    /// Format statistics as human-readable string
    pub fn format_summary(&self) -> String {
        format!(
            "Session Statistics:\n\
             • Total entries: {}\n\
             • Code cells: {} | Meta commands: {}\n\
             • Success rate: {:.1}%\n\
             • Session duration: {}\n\
             • Total execution time: {}\n\
             • Average execution time: {}\n\
             • Environment variables: {}",
            self.total_entries,
            self.code_entries,
            self.meta_entries,
            self.success_rate,
            format_duration(self.total_duration),
            format_duration(self.total_execution_time),
            format_duration(self.average_execution_time),
            self.environment_variables
        )
    }
}

/// Validation result for session snapshots
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the snapshot is valid
    pub is_valid: bool,
    /// List of validation issues
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Get errors from validation issues
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.level == ValidationLevel::Error)
            .collect()
    }
    
    /// Get warnings from validation issues
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.level == ValidationLevel::Warning)
            .collect()
    }
    
    /// Format validation result as report
    pub fn format_report(&self) -> String {
        let mut report = String::new();
        
        if self.is_valid {
            report.push_str("✅ Session snapshot validation passed\n");
        } else {
            report.push_str("❌ Session snapshot validation failed\n");
        }
        
        let errors = self.errors();
        let warnings = self.warnings();
        
        if !errors.is_empty() {
            report.push_str("\nErrors:\n");
            for error in errors {
                report.push_str(&format!("  • {}\n", error.message));
                if let Some(ref suggestion) = error.suggestion {
                    report.push_str(&format!("    Suggestion: {}\n", suggestion));
                }
            }
        }
        
        if !warnings.is_empty() {
            report.push_str("\nWarnings:\n");
            for warning in warnings {
                report.push_str(&format!("  • {}\n", warning.message));
                if let Some(ref suggestion) = warning.suggestion {
                    report.push_str(&format!("    Suggestion: {}\n", suggestion));
                }
            }
        }
        
        report
    }
}

/// Individual validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level
    pub level: ValidationLevel,
    /// Issue description
    pub message: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Validation issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLevel {
    /// Non-fatal issue
    Warning,
    /// Fatal issue that prevents export
    Error,
}

/// Format SystemTime as human-readable string
fn format_system_time(time: SystemTime) -> String {
    if let Ok(datetime) = time.duration_since(SystemTime::UNIX_EPOCH) {
        let dt = Utc.timestamp_opt(datetime.as_secs() as i64, 0)
            .single()
            .unwrap_or_else(|| Utc::now());
        dt.format("%Y-%m-%d %H:%M:%S UTC").to_string()
    } else {
        "Unknown time".to_string()
    }
}

/// Format Duration as human-readable string
fn format_duration(duration: Duration) -> String {
    let total_seconds = duration.as_secs();
    let hours = total_seconds / 3600;
    let minutes = (total_seconds % 3600) / 60;
    let seconds = total_seconds % 60;
    let millis = duration.subsec_millis();
    
    if hours > 0 {
        format!("{}h {}m {}s", hours, minutes, seconds)
    } else if minutes > 0 {
        format!("{}m {}s", minutes, seconds)
    } else if seconds > 0 {
        format!("{}.{:03}s", seconds, millis)
    } else {
        format!("{}ms", millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn create_test_metadata() -> SessionMetadata {
        SessionMetadata::new("test-session".to_string(), "0.1.0".to_string())
    }
    
    fn create_test_entry(id: &str, input: &str) -> SessionEntry {
        SessionEntry::new(
            id.to_string(),
            CellType::Code,
            input.to_string(),
            Value::Integer(42),
        )
    }
    
    #[test]
    fn test_session_snapshot_creation() {
        let metadata = create_test_metadata();
        let snapshot = SessionSnapshot::new(metadata);
        
        assert_eq!(snapshot.entries.len(), 0);
        assert_eq!(snapshot.environment.len(), 0);
        assert_eq!(snapshot.metadata.session_id, "test-session");
    }
    
    #[test]
    fn test_add_entry() {
        let mut snapshot = SessionSnapshot::new(create_test_metadata());
        let entry = create_test_entry("cell-1", "x = 42");
        
        snapshot.add_entry(entry);
        assert_eq!(snapshot.entries.len(), 1);
        assert_eq!(snapshot.entries[0].cell_id, "cell-1");
    }
    
    #[test]
    fn test_entries_by_type() {
        let mut snapshot = SessionSnapshot::new(create_test_metadata());
        
        let mut code_entry = create_test_entry("cell-1", "x = 42");
        code_entry.cell_type = CellType::Code;
        
        let mut meta_entry = create_test_entry("cell-2", "%help");
        meta_entry.cell_type = CellType::Meta;
        
        snapshot.add_entry(code_entry);
        snapshot.add_entry(meta_entry);
        
        let code_entries = snapshot.entries_by_type(CellType::Code);
        let meta_entries = snapshot.entries_by_type(CellType::Meta);
        
        assert_eq!(code_entries.len(), 1);
        assert_eq!(meta_entries.len(), 1);
        assert_eq!(code_entries[0].cell_id, "cell-1");
        assert_eq!(meta_entries[0].cell_id, "cell-2");
    }
    
    #[test]
    fn test_error_entries() {
        let mut snapshot = SessionSnapshot::new(create_test_metadata());
        
        let mut success_entry = create_test_entry("cell-1", "x = 42");
        success_entry.error = None;
        
        let mut error_entry = create_test_entry("cell-2", "invalid syntax");
        error_entry.error = Some("Parse error".to_string());
        
        snapshot.add_entry(success_entry);
        snapshot.add_entry(error_entry);
        
        let error_entries = snapshot.error_entries();
        assert_eq!(error_entries.len(), 1);
        assert_eq!(error_entries[0].cell_id, "cell-2");
    }
    
    #[test]
    fn test_session_statistics() {
        let mut snapshot = SessionSnapshot::new(create_test_metadata());
        
        let mut code_entry = create_test_entry("cell-1", "x = 42");
        code_entry.cell_type = CellType::Code;
        code_entry.execution_time = Some(Duration::from_millis(100));
        
        let mut meta_entry = create_test_entry("cell-2", "%help");
        meta_entry.cell_type = CellType::Meta;
        meta_entry.execution_time = Some(Duration::from_millis(50));
        
        let mut error_entry = create_test_entry("cell-3", "invalid");
        error_entry.cell_type = CellType::Code;
        error_entry.error = Some("Error".to_string());
        
        snapshot.add_entry(code_entry);
        snapshot.add_entry(meta_entry);
        snapshot.add_entry(error_entry);
        
        let stats = snapshot.statistics();
        assert_eq!(stats.total_entries, 3);
        assert_eq!(stats.code_entries, 2);
        assert_eq!(stats.meta_entries, 1);
        assert_eq!(stats.error_entries, 1);
        assert_eq!(stats.success_entries, 2);
        assert!((stats.success_rate - 66.67).abs() < 0.1);
    }
    
    #[test]
    fn test_session_entry_formatting() {
        let entry = create_test_entry("cell-1", "x = 42");
        
        assert!(entry.is_success());
        assert!(entry.has_output());
        assert!(!entry.formatted_timestamp().is_empty());
        
        let truncated = entry.truncated_output(5);
        assert!(truncated.len() <= 8); // 5 chars + "..."
    }
    
    #[test]
    fn test_session_metadata_formatting() {
        let metadata = create_test_metadata();
        
        assert!(!metadata.formatted_start_time().is_empty());
        assert!(!metadata.formatted_end_time().is_empty());
        assert!(!metadata.formatted_duration().is_empty());
    }
    
    #[test]
    fn test_validation() {
        let mut snapshot = SessionSnapshot::new(create_test_metadata());
        
        // Add entry with duplicate ID
        snapshot.add_entry(create_test_entry("cell-1", "x = 42"));
        snapshot.add_entry(create_test_entry("cell-1", "y = 24")); // duplicate ID
        
        let validation = snapshot.validate();
        assert!(!validation.is_valid);
        assert!(!validation.errors().is_empty());
    }
    
    #[test]
    fn test_duration_formatting() {
        assert_eq!(format_duration(Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(Duration::from_secs(5)), "5.000s");
        assert_eq!(format_duration(Duration::from_secs(65)), "1m 5s");
        assert_eq!(format_duration(Duration::from_secs(3665)), "1h 1m 5s");
    }
    
    #[test]
    fn test_merge_sessions() {
        let mut snapshot1 = SessionSnapshot::new(create_test_metadata());
        snapshot1.add_entry(create_test_entry("cell-1", "x = 1"));
        
        let mut metadata2 = create_test_metadata();
        metadata2.session_id = "other-session".to_string();
        let mut snapshot2 = SessionSnapshot::new(metadata2);
        snapshot2.add_entry(create_test_entry("cell-1", "y = 2"));
        
        assert!(snapshot1.merge(snapshot2).is_ok());
        assert_eq!(snapshot1.entries.len(), 2);
        assert!(snapshot1.metadata.session_id.contains("merged"));
    }
}