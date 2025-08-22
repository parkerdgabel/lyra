//! File Operations for REPL
//! 
//! Provides file loading, saving, and execution capabilities for the Lyra REPL.
//! Supports loading demo scripts, saving sessions, and file inclusion.

use crate::ast::Expr;
use crate::parser::Parser;
use crate::compiler::Compiler;
use crate::vm::Value;
use crate::repl::{ReplError, ReplResult, EvaluationResult, HistoryEntry};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

/// Session state for export/import functionality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub variables: HashMap<String, SerializableValue>,
    pub history: Vec<SerializableHistoryEntry>,
    pub line_number: usize,
    pub show_performance: bool,
    pub show_timing: bool,
    pub timestamp: SystemTime,
    pub version: String,
}

/// Serializable version of Value for JSON export
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializableValue {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    Boolean(bool),
    List(Vec<SerializableValue>),
    // Complex types like LyObj are not serializable
    Complex(String), // Store as string representation
}

/// Serializable version of HistoryEntry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableHistoryEntry {
    pub line_number: usize,
    pub input: String,
    pub output: String,
    pub execution_time_ms: f64,
    pub timestamp: SystemTime,
}

impl From<&Value> for SerializableValue {
    fn from(value: &Value) -> Self {
        match value {
            Value::Integer(n) => SerializableValue::Integer(*n),
            Value::Real(f) => SerializableValue::Real(*f),
            Value::String(s) => SerializableValue::String(s.clone()),
            Value::Symbol(s) => SerializableValue::Symbol(s.clone()),
            Value::Boolean(b) => SerializableValue::Boolean(*b),
            Value::List(items) => {
                let serializable_items: Vec<SerializableValue> = items.iter()
                    .map(SerializableValue::from)
                    .collect();
                SerializableValue::List(serializable_items)
            }
            // For complex types, store as string representation
            _ => SerializableValue::Complex(format!("{:?}", value)),
        }
    }
}

impl SerializableValue {
    pub fn to_value(&self) -> Value {
        match self {
            SerializableValue::Integer(n) => Value::Integer(*n),
            SerializableValue::Real(f) => Value::Real(*f),
            SerializableValue::String(s) => Value::String(s.clone()),
            SerializableValue::Symbol(s) => Value::Symbol(s.clone()),
            SerializableValue::Boolean(b) => Value::Boolean(*b),
            SerializableValue::List(items) => {
                let values: Vec<Value> = items.iter()
                    .map(|item| item.to_value())
                    .collect();
                Value::List(values)
            }
            SerializableValue::Complex(s) => Value::Symbol(s.clone()),
        }
    }
}

impl From<&HistoryEntry> for SerializableHistoryEntry {
    fn from(entry: &HistoryEntry) -> Self {
        SerializableHistoryEntry {
            line_number: entry.line_number,
            input: entry.input.clone(),
            output: entry.output.clone(),
            execution_time_ms: entry.execution_time.as_secs_f64() * 1000.0,
            timestamp: entry.timestamp,
        }
    }
}

impl SerializableHistoryEntry {
    pub fn to_history_entry(&self) -> HistoryEntry {
        use crate::repl::history::CommandCategory;
        use std::collections::HashSet;
        
        HistoryEntry {
            line_number: self.line_number,
            input: self.input.clone(),
            output: self.output.clone(),
            execution_time: Duration::from_secs_f64(self.execution_time_ms / 1000.0),
            timestamp: self.timestamp,
            tags: HashSet::new(),
            session_id: None,
            success: true, // Assume success for old entries
            category: CommandCategory::Unknown,
        }
    }
}

pub struct FileOperations;

impl FileOperations {
    /// Load and execute a Lyra script file
    pub fn load_file(file_path: &str, current_dir: Option<&Path>) -> ReplResult<Vec<EvaluationResult>> {
        let path = Self::resolve_path(file_path, current_dir)?;
        
        // Read file content
        let source = fs::read_to_string(&path).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to read file '{}': {}", path.display(), e),
        })?;

        // Parse and execute the file
        Self::execute_source(&source, Some(path.to_string_lossy().to_string()))
    }

    /// Execute a file without loading results into session
    pub fn run_file(file_path: &str, current_dir: Option<&Path>) -> ReplResult<Vec<EvaluationResult>> {
        // Same as load_file for now, but could be extended to not affect session state
        Self::load_file(file_path, current_dir)
    }

    /// Include file content (parse but don't execute immediately)
    pub fn include_file(file_path: &str, current_dir: Option<&Path>) -> ReplResult<Vec<Expr>> {
        let path = Self::resolve_path(file_path, current_dir)?;
        
        // Read file content
        let source = fs::read_to_string(&path).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to read file '{}': {}", path.display(), e),
        })?;

        // Parse the file to AST
        let mut parser = Parser::from_source(&source).map_err(|e| ReplError::ParseError {
            message: format!("Failed to parse file '{}': {}", path.display(), e),
        })?;

        let statements = parser.parse().map_err(|e| ReplError::ParseError {
            message: format!("Parse error in file '{}': {}", path.display(), e),
        })?;

        Ok(statements)
    }

    /// Save session state to a JSON file
    pub fn save_session(
        variables: &HashMap<String, Value>,
        history: &[HistoryEntry],
        line_number: usize,
        show_performance: bool,
        show_timing: bool,
        file_path: &str,
        current_dir: Option<&Path>,
    ) -> ReplResult<String> {
        let path = Self::resolve_path(file_path, current_dir)?;

        // Convert to serializable format
        let serializable_variables: HashMap<String, SerializableValue> = variables.iter()
            .map(|(k, v)| (k.clone(), SerializableValue::from(v)))
            .collect();

        let serializable_history: Vec<SerializableHistoryEntry> = history.iter()
            .map(SerializableHistoryEntry::from)
            .collect();

        let session_state = SessionState {
            variables: serializable_variables,
            history: serializable_history,
            line_number,
            show_performance,
            show_timing,
            timestamp: SystemTime::now(),
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        // Serialize to JSON
        let json = serde_json::to_string_pretty(&session_state).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to serialize session: {}", e),
        })?;

        // Write to file
        fs::write(&path, json).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to write session to '{}': {}", path.display(), e),
        })?;

        Ok(format!("Session saved to '{}'", path.display()))
    }

    /// Load session state from a JSON file
    pub fn load_session(
        file_path: &str,
        current_dir: Option<&Path>,
    ) -> ReplResult<(HashMap<String, Value>, Vec<HistoryEntry>, usize, bool, bool)> {
        let path = Self::resolve_path(file_path, current_dir)?;

        // Read file content
        let json = fs::read_to_string(&path).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to read session file '{}': {}", path.display(), e),
        })?;

        // Deserialize from JSON
        let session_state: SessionState = serde_json::from_str(&json).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to parse session file '{}': {}", path.display(), e),
        })?;

        // Convert back to runtime types
        let variables: HashMap<String, Value> = session_state.variables.iter()
            .map(|(k, v)| (k.clone(), v.to_value()))
            .collect();

        let history: Vec<HistoryEntry> = session_state.history.iter()
            .map(|entry| entry.to_history_entry())
            .collect();

        Ok((
            variables,
            history,
            session_state.line_number,
            session_state.show_performance,
            session_state.show_timing,
        ))
    }

    /// Execute source code and return evaluation results
    fn execute_source(source: &str, file_name: Option<String>) -> ReplResult<Vec<EvaluationResult>> {
        let start_time = Instant::now();

        // Parse the source
        let mut parser = Parser::from_source(source).map_err(|e| ReplError::ParseError {
            message: format!("Parse error{}: {}", 
                file_name.as_ref().map(|f| format!(" in '{}'", f)).unwrap_or_default(), 
                e
            ),
        })?;

        let statements = parser.parse().map_err(|e| ReplError::ParseError {
            message: format!("Parse error{}: {}", 
                file_name.as_ref().map(|f| format!(" in '{}'", f)).unwrap_or_default(), 
                e
            ),
        })?;

        // Execute each statement
        let mut results = Vec::new();
        for (i, statement) in statements.iter().enumerate() {
            let stmt_start = Instant::now();
            
            // Compile and evaluate
            let value = Compiler::eval(statement).map_err(|e| ReplError::CompilationError {
                message: format!("Compilation error{} (statement {}): {}", 
                    file_name.as_ref().map(|f| format!(" in '{}'", f)).unwrap_or_default(),
                    i + 1,
                    e
                ),
            })?;

            let execution_time = stmt_start.elapsed();
            let result_str = Self::format_value(&value);

            results.push(EvaluationResult {
                result: result_str,
                value: Some(value),
                performance_info: None,
                execution_time,
            });
        }

        Ok(results)
    }

    /// Format a value for display (simplified version)
    fn format_value(value: &Value) -> String {
        match value {
            Value::Integer(n) => n.to_string(),
            Value::Real(f) => {
                if f.fract() == 0.0 {
                    format!("{:.1}", f)
                } else {
                    f.to_string()
                }
            }
            Value::String(s) => format!("\"{}\"", s),
            Value::Symbol(s) => s.clone(),
            Value::List(items) => {
                let formatted_items: Vec<String> = items.iter().map(Self::format_value).collect();
                format!("{{{}}}", formatted_items.join(", "))
            }
            Value::Function(name) => format!("Function[{}]", name),
            Value::Boolean(b) => if *b { "True" } else { "False" }.to_string(),
            Value::Missing => "Missing[]".to_string(),
            Value::Object(obj) => format!("Object[{:?}]", obj),
            Value::LyObj(obj) => format!("{}[...]", obj.type_name()),
            Value::Quote(expr) => format!("Hold[{:?}]", expr),
            Value::Pattern(pattern) => format!("{}", pattern),
            Value::Rule { lhs, rhs } => format!("{} -> {}", Self::format_value(lhs), Self::format_value(rhs)),
            Value::PureFunction { body } => format!("{} &", Self::format_value(body)),
            Value::Slot { number } => match number {
                Some(n) => format!("#{}", n),
                None => "#".to_string(),
            },
        }
    }

    /// Resolve file path relative to current directory or absolute
    fn resolve_path(file_path: &str, current_dir: Option<&Path>) -> ReplResult<PathBuf> {
        let path = PathBuf::from(file_path);
        
        if path.is_absolute() {
            return Ok(path);
        }

        // Try relative to current directory first
        if let Some(current) = current_dir {
            let resolved = current.join(&path);
            if resolved.exists() {
                return Ok(resolved);
            }
        }

        // Try relative to current working directory
        let cwd_path = std::env::current_dir()
            .map_err(|e| ReplError::RuntimeError {
                message: format!("Failed to get current directory: {}", e),
            })?
            .join(&path);

        if cwd_path.exists() {
            return Ok(cwd_path);
        }

        // Try relative to examples directory
        let examples_path = std::env::current_dir()
            .map_err(|e| ReplError::RuntimeError {
                message: format!("Failed to get current directory: {}", e),
            })?
            .join("examples")
            .join(&path);

        if examples_path.exists() {
            return Ok(examples_path);
        }

        // File not found in any location
        Err(ReplError::RuntimeError {
            message: format!("File not found: '{}' (searched in current directory, working directory, and examples directory)", file_path),
        })
    }

    /// List available example files
    pub fn list_examples() -> ReplResult<Vec<String>> {
        let examples_dir = std::env::current_dir()
            .map_err(|e| ReplError::RuntimeError {
                message: format!("Failed to get current directory: {}", e),
            })?
            .join("examples");

        if !examples_dir.exists() {
            return Ok(vec!["No examples directory found".to_string()]);
        }

        let mut files = Vec::new();
        
        // Read directory entries
        let entries = fs::read_dir(&examples_dir).map_err(|e| ReplError::RuntimeError {
            message: format!("Failed to read examples directory: {}", e),
        })?;

        for entry in entries {
            let entry = entry.map_err(|e| ReplError::RuntimeError {
                message: format!("Failed to read directory entry: {}", e),
            })?;

            let path = entry.path();
            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "lyra" || extension == "ly" {
                        if let Some(file_name) = path.file_name() {
                            files.push(file_name.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }

        files.sort();
        Ok(files)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::{tempdir, NamedTempFile};

    #[test]
    fn test_serializable_value_conversion() {
        let original = Value::Integer(42);
        let serializable = SerializableValue::from(&original);
        let converted_back = serializable.to_value();
        
        if let Value::Integer(n) = converted_back {
            assert_eq!(n, 42);
        } else {
            panic!("Conversion failed");
        }
    }

    #[test]
    fn test_execute_source() {
        let source = "x = 5\ny = x + 3";
        let results = FileOperations::execute_source(source, None).unwrap();
        
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].result, "5");
        assert_eq!(results[1].result, "8");
    }

    #[test]
    fn test_save_and_load_session() {
        let temp_file = NamedTempFile::new().unwrap();
        let file_path = temp_file.path().to_str().unwrap();

        // Create test data
        let mut variables = HashMap::new();
        variables.insert("x".to_string(), Value::Integer(42));
        variables.insert("y".to_string(), Value::String("test".to_string()));

        let history = vec![
            HistoryEntry {
                line_number: 1,
                input: "x = 42".to_string(),
                output: "42".to_string(),
                execution_time: Duration::from_millis(10),
                timestamp: SystemTime::now(),
                tags: std::collections::HashSet::new(),
                session_id: None,
                success: true,
                category: crate::repl::history::CommandCategory::VariableAssignment,
            }
        ];

        // Save session
        let result = FileOperations::save_session(
            &variables, 
            &history, 
            2, 
            true, 
            false, 
            file_path, 
            None
        );
        assert!(result.is_ok());

        // Load session
        let (loaded_vars, loaded_history, line_num, show_perf, show_timing) = 
            FileOperations::load_session(file_path, None).unwrap();

        assert_eq!(loaded_vars.len(), 2);
        assert_eq!(loaded_history.len(), 1);
        assert_eq!(line_num, 2);
        assert_eq!(show_perf, true);
        assert_eq!(show_timing, false);
    }

    #[test]
    fn test_resolve_path() {
        // Test absolute path
        let abs_path = if cfg!(windows) { "C:\\test.lyra" } else { "/test.lyra" };
        let resolved = FileOperations::resolve_path(abs_path, None);
        // This will fail because the file doesn't exist, but the path should be recognized as absolute
        assert!(resolved.is_err()); // File not found is expected

        // Test relative path resolution logic (without requiring actual files)
        let result = FileOperations::resolve_path("nonexistent.lyra", None);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("File not found"));
    }
}