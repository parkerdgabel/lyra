//! System Integration & Environment Operations
//!
//! This module provides comprehensive system integration capabilities:
//! - Environment variable operations
//! - File system operations and metadata
//! - Process management and execution
//! - Path manipulation utilities
//! - System information and diagnostics
//!
//! All functions are designed to be cross-platform (Windows, macOS, Linux)
//! and include proper error handling for security and robustness.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmError, VmResult};
use std::env;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};
use sysinfo::{System, PidExt, SystemExt, ProcessExt, UserExt, CpuExt, RefreshKind, ProcessRefreshKind};
use std::any::Any;

/// Error types for system operations
#[derive(Debug, Clone)]
pub enum SystemError {
    PermissionDenied(String),
    NotFound(String),
    InvalidPath(String),
    ProcessError(String),
    EnvironmentError(String),
    IoError(String),
}

impl fmt::Display for SystemError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SystemError::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            SystemError::NotFound(msg) => write!(f, "Not found: {}", msg),
            SystemError::InvalidPath(msg) => write!(f, "Invalid path: {}", msg),
            SystemError::ProcessError(msg) => write!(f, "Process error: {}", msg),
            SystemError::EnvironmentError(msg) => write!(f, "Environment error: {}", msg),
            SystemError::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

/// Foreign object for running process management
#[derive(Debug, Clone)]
pub struct ProcessHandle {
    pid: u32,
    command: String,
    args: Vec<String>,
    start_time: SystemTime,
}

impl ProcessHandle {
    pub fn new(pid: u32, command: String, args: Vec<String>) -> Self {
        Self {
            pid,
            command,
            args,
            start_time: SystemTime::now(),
        }
    }
    
    pub fn pid(&self) -> u32 {
        self.pid
    }
    
    pub fn command(&self) -> &str {
        &self.command
    }
    
    pub fn is_running(&self) -> bool {
        let mut system = System::new_with_specifics(RefreshKind::new().with_processes(ProcessRefreshKind::new()));
        system.refresh_processes();
        system.process(sysinfo::Pid::from_u32(self.pid)).is_some()
    }
    
    pub fn kill(&self) -> Result<(), SystemError> {
        let mut system = System::new_with_specifics(RefreshKind::new().with_processes(ProcessRefreshKind::new()));
        system.refresh_processes();
        
        if let Some(process) = system.process(sysinfo::Pid::from_u32(self.pid)) {
            if process.kill() {
                Ok(())
            } else {
                Err(SystemError::ProcessError(format!("Failed to kill process {}", self.pid)))
            }
        } else {
            Err(SystemError::NotFound(format!("Process {} not found", self.pid)))
        }
    }
}

impl Foreign for ProcessHandle {
    fn type_name(&self) -> &'static str {
        "ProcessHandle"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "pid" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.pid as i64))
            }
            "command" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.command.clone()))
            }
            "isRunning" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_running()))
            }
            "kill" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.kill() {
                    Ok(()) => Ok(Value::Boolean(true)),
                    Err(e) => Err(ForeignError::RuntimeError { message: e.to_string() }),
                }
            }
            "uptime" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let duration = self.start_time.elapsed()
                    .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                Ok(Value::Real(duration.as_secs_f64()))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object for directory contents with metadata
#[derive(Debug, Clone)]
pub struct DirectoryListing {
    path: PathBuf,
    entries: Vec<DirectoryEntry>,
}

#[derive(Debug, Clone)]
pub struct DirectoryEntry {
    name: String,
    path: PathBuf,
    is_file: bool,
    is_directory: bool,
    size: Option<u64>,
    modified: Option<SystemTime>,
}

impl DirectoryListing {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, SystemError> {
        let path = path.as_ref().to_path_buf();
        let mut entries = Vec::new();
        
        let read_dir = fs::read_dir(&path)
            .map_err(|e| SystemError::IoError(format!("Failed to read directory {}: {}", path.display(), e)))?;
        
        for entry in read_dir {
            let entry = entry.map_err(|e| SystemError::IoError(e.to_string()))?;
            let entry_path = entry.path();
            let metadata = entry_path.metadata().ok();
            
            entries.push(DirectoryEntry {
                name: entry.file_name().to_string_lossy().to_string(),
                path: entry_path.clone(),
                is_file: metadata.as_ref().map(|m| m.is_file()).unwrap_or(false),
                is_directory: metadata.as_ref().map(|m| m.is_dir()).unwrap_or(false),
                size: metadata.as_ref().and_then(|m| if m.is_file() { Some(m.len()) } else { None }),
                modified: metadata.as_ref().and_then(|m| m.modified().ok()),
            });
        }
        
        // Sort by name for consistent ordering
        entries.sort_by(|a, b| a.name.cmp(&b.name));
        
        Ok(Self { path, entries })
    }
    
    pub fn entries(&self) -> &[DirectoryEntry] {
        &self.entries
    }
    
    pub fn files(&self) -> Vec<&DirectoryEntry> {
        self.entries.iter().filter(|e| e.is_file).collect()
    }
    
    pub fn directories(&self) -> Vec<&DirectoryEntry> {
        self.entries.iter().filter(|e| e.is_directory).collect()
    }
}

impl Foreign for DirectoryListing {
    fn type_name(&self) -> &'static str {
        "DirectoryListing"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "path" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.path.to_string_lossy().to_string()))
            }
            "count" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.entries.len() as i64))
            }
            "fileCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.files().len() as i64))
            }
            "directoryCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.directories().len() as i64))
            }
            "names" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let names: Vec<Value> = self.entries.iter()
                    .map(|e| Value::String(e.name.clone()))
                    .collect();
                Ok(Value::List(names))
            }
            "paths" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let paths: Vec<Value> = self.entries.iter()
                    .map(|e| Value::String(e.path.to_string_lossy().to_string()))
                    .collect();
                Ok(Value::List(paths))
            }
            "fileNames" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let names: Vec<Value> = self.files().iter()
                    .map(|e| Value::String(e.name.clone()))
                    .collect();
                Ok(Value::List(names))
            }
            "directoryNames" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let names: Vec<Value> = self.directories().iter()
                    .map(|e| Value::String(e.name.clone()))
                    .collect();
                Ok(Value::List(names))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object for file metadata and properties
#[derive(Debug, Clone)]
pub struct FileMetadata {
    path: PathBuf,
    size: u64,
    is_file: bool,
    is_directory: bool,
    is_symlink: bool,
    modified: Option<SystemTime>,
    accessed: Option<SystemTime>,
    created: Option<SystemTime>,
    permissions: Option<fs::Permissions>,
}

impl FileMetadata {
    pub fn new(path: impl AsRef<Path>) -> Result<Self, SystemError> {
        let path = path.as_ref().to_path_buf();
        let metadata = fs::metadata(&path)
            .map_err(|e| SystemError::IoError(format!("Failed to get metadata for {}: {}", path.display(), e)))?;
        
        Ok(Self {
            path,
            size: metadata.len(),
            is_file: metadata.is_file(),
            is_directory: metadata.is_dir(),
            is_symlink: metadata.file_type().is_symlink(),
            modified: metadata.modified().ok(),
            accessed: metadata.accessed().ok(),
            created: metadata.created().ok(),
            permissions: Some(metadata.permissions()),
        })
    }
}

impl Foreign for FileMetadata {
    fn type_name(&self) -> &'static str {
        "FileMetadata"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "path" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.path.to_string_lossy().to_string()))
            }
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.size as i64))
            }
            "isFile" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_file))
            }
            "isDirectory" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_directory))
            }
            "isSymlink" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_symlink))
            }
            "modifiedTime" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.modified {
                    Some(time) => {
                        let timestamp = time.duration_since(UNIX_EPOCH)
                            .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Integer(timestamp.as_secs() as i64))
                    }
                    None => Ok(Value::Missing),
                }
            }
            "accessedTime" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.accessed {
                    Some(time) => {
                        let timestamp = time.duration_since(UNIX_EPOCH)
                            .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Integer(timestamp.as_secs() as i64))
                    }
                    None => Ok(Value::Missing),
                }
            }
            "createdTime" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.created {
                    Some(time) => {
                        let timestamp = time.duration_since(UNIX_EPOCH)
                            .map_err(|e| ForeignError::RuntimeError { message: e.to_string() })?;
                        Ok(Value::Integer(timestamp.as_secs() as i64))
                    }
                    None => Ok(Value::Missing),
                }
            }
            "readOnly" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match &self.permissions {
                    Some(perms) => Ok(Value::Boolean(perms.readonly())),
                    None => Ok(Value::Missing),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Foreign object for system metrics and information
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    system: Arc<Mutex<System>>,
}

impl SystemMetrics {
    pub fn new() -> Self {
        Self {
            system: Arc::new(Mutex::new(System::new_with_specifics(RefreshKind::everything()))),
        }
    }
    
    pub fn refresh(&self) {
        if let Ok(mut sys) = self.system.lock() {
            sys.refresh_all();
        }
    }
}

impl Foreign for SystemMetrics {
    fn type_name(&self) -> &'static str {
        "SystemMetrics"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "refresh" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                self.refresh();
                Ok(Value::Boolean(true))
            }
            "cpuUsage" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Ok(sys) = self.system.lock() {
                    let usage = sys.global_cpu_info().cpu_usage();
                    Ok(Value::Real(usage as f64))
                } else {
                    Err(ForeignError::RuntimeError { message: "Failed to lock system".to_string() })
                }
            }
            "memoryUsed" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Ok(sys) = self.system.lock() {
                    Ok(Value::Integer(sys.used_memory() as i64))
                } else {
                    Err(ForeignError::RuntimeError { message: "Failed to lock system".to_string() })
                }
            }
            "memoryTotal" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Ok(sys) = self.system.lock() {
                    Ok(Value::Integer(sys.total_memory() as i64))
                } else {
                    Err(ForeignError::RuntimeError { message: "Failed to lock system".to_string() })
                }
            }
            "processCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Ok(sys) = self.system.lock() {
                    Ok(Value::Integer(sys.processes().len() as i64))
                } else {
                    Err(ForeignError::RuntimeError { message: "Failed to lock system".to_string() })
                }
            }
            "uptime" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                if let Ok(sys) = self.system.lock() {
                    Ok(Value::Integer(sys.uptime() as i64))
                } else {
                    Err(ForeignError::RuntimeError { message: "Failed to lock system".to_string() })
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Utility functions for safe path operations
fn validate_path(path: &str) -> Result<PathBuf, SystemError> {
    let path = Path::new(path);
    
    // Check for directory traversal attacks
    if path.components().any(|component| {
        matches!(component, std::path::Component::ParentDir)
    }) {
        return Err(SystemError::InvalidPath(
            "Directory traversal not allowed".to_string()
        ));
    }
    
    Ok(path.to_path_buf())
}

fn sanitize_command_args(command: &str, args: &[String]) -> Result<(String, Vec<String>), SystemError> {
    // Basic command injection prevention
    if command.contains(';') || command.contains('&') || command.contains('|') {
        return Err(SystemError::ProcessError(
            "Command injection characters not allowed".to_string()
        ));
    }
    
    for arg in args {
        if arg.contains(';') || arg.contains('&') || arg.contains('|') {
            return Err(SystemError::ProcessError(
                "Command injection characters not allowed in arguments".to_string()
            ));
        }
    }
    
    Ok((command.to_string(), args.to_vec()))
}

// Environment Variable Operations (5 functions)

/// Get environment variable value
pub fn environment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "Environment[var_name]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let var_name = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    match env::var(var_name) {
        Ok(value) => Ok(Value::String(value)),
        Err(_) => Ok(Value::Missing),
    }
}

/// Set environment variable
pub fn set_environment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "SetEnvironment[var_name, value]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let var_name = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let value = match &args[1] {
        Value::String(s) => s,
        Value::Integer(n) => &n.to_string(),
        Value::Real(f) => &f.to_string(),
        Value::Boolean(b) => if *b { "true" } else { "false" },
        _ => return Err(VmError::TypeError {
            expected: "string, integer, real, or boolean".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    env::set_var(var_name, value);
    Ok(Value::Boolean(true))
}

/// Unset/remove environment variable
pub fn unset_environment(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "UnsetEnvironment[var_name]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let var_name = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    env::remove_var(var_name);
    Ok(Value::Boolean(true))
}

/// List all environment variables
pub fn environment_list(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "EnvironmentList[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let mut env_vars = Vec::new();
    for (key, value) in env::vars() {
        let pair = vec![Value::String(key), Value::String(value)];
        env_vars.push(Value::List(pair));
    }
    
    Ok(Value::List(env_vars))
}

/// Get system information
pub fn system_info(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "SystemInfo[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let mut system = System::new_with_specifics(RefreshKind::everything());
    system.refresh_all();
    
    let info = vec![
        Value::List(vec![Value::String("os_name".to_string()), Value::String(system.name().unwrap_or("Unknown".to_string()))]),
        Value::List(vec![Value::String("os_version".to_string()), Value::String(system.os_version().unwrap_or("Unknown".to_string()))]),
        Value::List(vec![Value::String("hostname".to_string()), Value::String(system.host_name().unwrap_or("Unknown".to_string()))]),
        Value::List(vec![Value::String("architecture".to_string()), Value::String(env::consts::ARCH.to_string())]),
        Value::List(vec![Value::String("cpu_count".to_string()), Value::Integer(system.cpus().len() as i64)]),
        Value::List(vec![Value::String("total_memory".to_string()), Value::Integer(system.total_memory() as i64)]),
        Value::List(vec![Value::String("used_memory".to_string()), Value::Integer(system.used_memory() as i64)]),
        Value::List(vec![Value::String("uptime".to_string()), Value::Integer(system.uptime() as i64)]),
    ];
    
    Ok(Value::List(info))
}

// File System Operations (8 functions)

/// Check if file exists
pub fn file_exists(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FileExists[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::Boolean(path_buf.exists() && path_buf.is_file()))
}

/// Check if directory exists
pub fn directory_exists(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "DirectoryExists[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::Boolean(path_buf.exists() && path_buf.is_dir()))
}

/// List directory contents
pub fn directory_list(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "DirectoryList[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let listing = DirectoryListing::new(path_buf)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(listing))))
}

/// Create directory (with parents if needed)
pub fn create_directory(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "CreateDirectory[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    match fs::create_dir_all(&path_buf) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to create directory {}: {}", path, e))),
    }
}

/// Delete file
pub fn delete_file(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "DeleteFile[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !path_buf.exists() {
        return Err(VmError::Runtime(format!("File does not exist: {}", path)));
    }
    
    if !path_buf.is_file() {
        return Err(VmError::Runtime(format!("Path is not a file: {}", path)));
    }
    
    match fs::remove_file(&path_buf) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to delete file {}: {}", path, e))),
    }
}

/// Delete directory
pub fn delete_directory(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "DeleteDirectory[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !path_buf.exists() {
        return Err(VmError::Runtime(format!("Directory does not exist: {}", path)));
    }
    
    if !path_buf.is_dir() {
        return Err(VmError::Runtime(format!("Path is not a directory: {}", path)));
    }
    
    match fs::remove_dir_all(&path_buf) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to delete directory {}: {}", path, e))),
    }
}

/// Copy file
pub fn copy_file(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "CopyFile[source, destination]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let source = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let destination = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let source_path = validate_path(source)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    let dest_path = validate_path(destination)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !source_path.exists() {
        return Err(VmError::Runtime(format!("Source file does not exist: {}", source)));
    }
    
    if !source_path.is_file() {
        return Err(VmError::Runtime(format!("Source is not a file: {}", source)));
    }
    
    // Create destination directory if it doesn't exist
    if let Some(dest_dir) = dest_path.parent() {
        if !dest_dir.exists() {
            fs::create_dir_all(dest_dir)
                .map_err(|e| VmError::Runtime(format!("Failed to create destination directory: {}", e)))?;
        }
    }
    
    match fs::copy(&source_path, &dest_path) {
        Ok(bytes_copied) => Ok(Value::Integer(bytes_copied as i64)),
        Err(e) => Err(VmError::Runtime(format!("Failed to copy file from {} to {}: {}", source, destination, e))),
    }
}

/// Move/rename file
pub fn move_file(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "MoveFile[source, destination]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let source = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let destination = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let source_path = validate_path(source)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    let dest_path = validate_path(destination)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !source_path.exists() {
        return Err(VmError::Runtime(format!("Source file does not exist: {}", source)));
    }
    
    // Create destination directory if it doesn't exist
    if let Some(dest_dir) = dest_path.parent() {
        if !dest_dir.exists() {
            fs::create_dir_all(dest_dir)
                .map_err(|e| VmError::Runtime(format!("Failed to create destination directory: {}", e)))?;
        }
    }
    
    match fs::rename(&source_path, &dest_path) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to move file from {} to {}: {}", source, destination, e))),
    }
}

// File Information Functions (7 functions)

/// Get file size in bytes
pub fn file_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FileSize[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let metadata = FileMetadata::new(path_buf)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::Integer(metadata.size as i64))
}

/// Get file modification time
pub fn file_modification_time(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FileModificationTime[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let metadata = FileMetadata::new(path_buf)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    match metadata.modified {
        Some(time) => {
            let timestamp = time.duration_since(UNIX_EPOCH)
                .map_err(|e| VmError::Runtime(e.to_string()))?;
            Ok(Value::Integer(timestamp.as_secs() as i64))
        }
        None => Ok(Value::Missing),
    }
}

/// Get file permissions (Foreign object)
pub fn file_permissions(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FilePermissions[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let metadata = FileMetadata::new(path_buf)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::LyObj(LyObj::new(Box::new(metadata))))
}

/// Set file permissions
pub fn set_file_permissions(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "SetFilePermissions[path, permissions]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let readonly = match &args[1] {
        Value::Boolean(b) => *b,
        _ => return Err(VmError::TypeError {
            expected: "boolean (readonly flag)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !path_buf.exists() {
        return Err(VmError::Runtime(format!("File does not exist: {}", path)));
    }
    
    let metadata = fs::metadata(&path_buf)
        .map_err(|e| VmError::Runtime(format!("Failed to get metadata: {}", e)))?;
    
    let mut permissions = metadata.permissions();
    permissions.set_readonly(readonly);
    
    match fs::set_permissions(&path_buf, permissions) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to set permissions: {}", e))),
    }
}

/// Check if path is file
pub fn is_file(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "IsFile[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::Boolean(path_buf.is_file()))
}

/// Check if path is directory
pub fn is_directory(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "IsDirectory[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    Ok(Value::Boolean(path_buf.is_dir()))
}

/// Check if path is symbolic link
pub fn is_symbolic_link(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "IsSymbolicLink[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let is_symlink = path_buf.symlink_metadata()
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    
    Ok(Value::Boolean(is_symlink))
}

// Process Management Functions (6 functions)

/// Execute command and get output
pub fn run_command(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "RunCommand[command] or RunCommand[command, args]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let command = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let command_args = if args.len() > 1 {
        match &args[1] {
            Value::List(list) => {
                let mut string_args = Vec::new();
                for arg in list {
                    match arg {
                        Value::String(s) => string_args.push(s.clone()),
                        _ => return Err(VmError::TypeError {
                            expected: "list of strings".to_string(),
                            actual: format!("list containing {:?}", arg),
                        }),
                    }
                }
                string_args
            }
            _ => return Err(VmError::TypeError {
                expected: "list".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        Vec::new()
    };
    
    let (clean_command, clean_args) = sanitize_command_args(command, &command_args)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let output = Command::new(&clean_command)
        .args(&clean_args)
        .output()
        .map_err(|e| VmError::Runtime(format!("Failed to execute command: {}", e)))?;
    
    let result = vec![
        Value::List(vec![Value::String("stdout".to_string()), Value::String(String::from_utf8_lossy(&output.stdout).to_string())]),
        Value::List(vec![Value::String("stderr".to_string()), Value::String(String::from_utf8_lossy(&output.stderr).to_string())]),
        Value::List(vec![Value::String("exit_code".to_string()), Value::Integer(output.status.code().unwrap_or(-1) as i64)]),
        Value::List(vec![Value::String("success".to_string()), Value::Boolean(output.status.success())]),
    ];
    
    Ok(Value::List(result))
}

/// Start process asynchronously
pub fn process_start(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "ProcessStart[command] or ProcessStart[command, args] or ProcessStart[command, args, options]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let command = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let command_args = if args.len() > 1 {
        match &args[1] {
            Value::List(list) => {
                let mut string_args = Vec::new();
                for arg in list {
                    match arg {
                        Value::String(s) => string_args.push(s.clone()),
                        _ => return Err(VmError::TypeError {
                            expected: "list of strings".to_string(),
                            actual: format!("list containing {:?}", arg),
                        }),
                    }
                }
                string_args
            }
            _ => return Err(VmError::TypeError {
                expected: "list".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        Vec::new()
    };
    
    let (clean_command, clean_args) = sanitize_command_args(command, &command_args)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let mut child = Command::new(&clean_command)
        .args(&clean_args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| VmError::Runtime(format!("Failed to start process: {}", e)))?;
    
    let pid = child.id();
    let handle = ProcessHandle::new(pid, clean_command, clean_args);
    
    // Spawn the child process in a separate thread to avoid blocking
    thread::spawn(move || {
        let _ = child.wait();
    });
    
    Ok(Value::LyObj(LyObj::new(Box::new(handle))))
}

/// List running processes
pub fn process_list(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "ProcessList[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let mut system = System::new_with_specifics(RefreshKind::new().with_processes(ProcessRefreshKind::everything()));
    system.refresh_processes();
    
    let mut processes = Vec::new();
    for (pid, process) in system.processes() {
        let process_info = vec![
            Value::List(vec![Value::String("pid".to_string()), Value::Integer(pid.as_u32() as i64)]),
            Value::List(vec![Value::String("name".to_string()), Value::String(process.name().to_string())]),
            Value::List(vec![Value::String("cpu_usage".to_string()), Value::Real(process.cpu_usage() as f64)]),
            Value::List(vec![Value::String("memory".to_string()), Value::Integer(process.memory() as i64)]),
        ];
        processes.push(Value::List(process_info));
    }
    
    Ok(Value::List(processes))
}

/// Terminate process by PID
pub fn process_kill(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "ProcessKill[pid]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let pid = match &args[0] {
        Value::Integer(n) => *n as u32,
        _ => return Err(VmError::TypeError {
            expected: "integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut system = System::new_with_specifics(RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new()));
    system.refresh_processes();
    
    if let Some(process) = system.process(sysinfo::Pid::from_u32(pid)) {
        if process.kill() {
            Ok(Value::Boolean(true))
        } else {
            Err(VmError::Runtime(format!("Failed to kill process {}", pid)))
        }
    } else {
        Err(VmError::Runtime(format!("Process {} not found", pid)))
    }
}

/// Get current process ID
pub fn current_pid(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "CurrentPID[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    Ok(Value::Integer(std::process::id() as i64))
}

/// Check if process exists
pub fn process_exists(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "ProcessExists[pid]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let pid = match &args[0] {
        Value::Integer(n) => *n as u32,
        _ => return Err(VmError::TypeError {
            expected: "integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut system = System::new_with_specifics(RefreshKind::new().with_processes(sysinfo::ProcessRefreshKind::new()));
    system.refresh_processes();
    
    Ok(Value::Boolean(system.process(sysinfo::Pid::from_u32(pid)).is_some()))
}

// Path Operations (7 functions)

/// Get absolute path
pub fn absolute_path(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "AbsolutePath[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = Path::new(path);
    let absolute = path_buf.canonicalize()
        .or_else(|_| {
            // If canonicalize fails, try to make absolute manually
            let current_dir = env::current_dir()
                .map_err(|e| VmError::Runtime(format!("Failed to get current directory: {}", e)))?;
            Ok(current_dir.join(path_buf))
        })
        .map_err(|e: VmError| e)?;
    
    Ok(Value::String(absolute.to_string_lossy().to_string()))
}

/// Get relative path from one location to another
pub fn relative_path(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "RelativePath[from, to]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let from = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let to = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let from_path = Path::new(from);
    let to_path = Path::new(to);
    
    // This is a simplified implementation
    // A full implementation would handle all edge cases
    match pathdiff::diff_paths(to_path, from_path) {
        Some(relative) => Ok(Value::String(relative.to_string_lossy().to_string())),
        None => Ok(Value::String(to.clone())), // Fallback to absolute path
    }
}

/// Join path components
pub fn path_join(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::TypeError {
            expected: "PathJoin[part1, part2, ...]".to_string(),
            actual: "got 0 arguments".to_string(),
        });
    }
    
    let mut path = PathBuf::new();
    for arg in args {
        match arg {
            Value::String(s) => path.push(s),
            _ => return Err(VmError::TypeError {
                expected: "string".to_string(),
                actual: format!("{:?}", arg),
            }),
        }
    }
    
    Ok(Value::String(path.to_string_lossy().to_string()))
}

/// Split path into components
pub fn path_split(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "PathSplit[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = Path::new(path);
    let components: Vec<Value> = path_buf
        .components()
        .map(|component| Value::String(component.as_os_str().to_string_lossy().to_string()))
        .collect();
    
    Ok(Value::List(components))
}

/// Extract filename from path
pub fn file_name(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FileName[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = Path::new(path);
    match path_buf.file_name() {
        Some(name) => Ok(Value::String(name.to_string_lossy().to_string())),
        None => Ok(Value::Missing),
    }
}

/// Extract file extension from path
pub fn file_extension(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "FileExtension[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = Path::new(path);
    match path_buf.extension() {
        Some(ext) => Ok(Value::String(format!(".{}", ext.to_string_lossy()))),
        None => Ok(Value::Missing),
    }
}

/// Extract directory name from path
pub fn directory_name(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "DirectoryName[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = Path::new(path);
    match path_buf.parent() {
        Some(parent) => Ok(Value::String(parent.to_string_lossy().to_string())),
        None => Ok(Value::Missing),
    }
}

// System Information Functions (6 functions)

/// Get current working directory
pub fn current_directory(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "CurrentDirectory[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    match env::current_dir() {
        Ok(dir) => Ok(Value::String(dir.to_string_lossy().to_string())),
        Err(e) => Err(VmError::Runtime(format!("Failed to get current directory: {}", e))),
    }
}

/// Change working directory
pub fn set_current_directory(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "SetCurrentDirectory[path]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "string".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path_buf = validate_path(path)
        .map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if !path_buf.exists() {
        return Err(VmError::Runtime(format!("Directory does not exist: {}", path)));
    }
    
    if !path_buf.is_dir() {
        return Err(VmError::Runtime(format!("Path is not a directory: {}", path)));
    }
    
    match env::set_current_dir(&path_buf) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to change directory: {}", e))),
    }
}

/// Get user home directory
pub fn home_directory(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "HomeDirectory[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    match dirs::home_dir() {
        Some(dir) => Ok(Value::String(dir.to_string_lossy().to_string())),
        None => Ok(Value::Missing),
    }
}

/// Get system temp directory
pub fn temp_directory(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "TempDirectory[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    let temp_dir = env::temp_dir();
    Ok(Value::String(temp_dir.to_string_lossy().to_string()))
}

/// Get current username
pub fn current_user(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "CurrentUser[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    match env::var("USER").or_else(|_| env::var("USERNAME")) {
        Ok(user) => Ok(Value::String(user)),
        Err(_) => {
            // Try using system info
            let mut system = System::new_with_specifics(RefreshKind::new().with_users_list());
            system.refresh_users_list();
            
            if let Some(user) = system.users().first() {
                Ok(Value::String(user.name().to_string()))
            } else {
                Ok(Value::String("unknown".to_string()))
            }
        }
    }
}

/// Get system architecture
pub fn system_architecture(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "SystemArchitecture[]".to_string(),
            actual: format!("got {} arguments", args.len()),
        });
    }
    
    Ok(Value::String(env::consts::ARCH.to_string()))
}

// Add a simple implementation of pathdiff functionality
mod pathdiff {
    use std::path::{Path, PathBuf, Component};
    
    pub fn diff_paths<P: AsRef<Path>, Q: AsRef<Path>>(path: P, base: Q) -> Option<PathBuf> {
        let path = path.as_ref();
        let base = base.as_ref();
        
        if path.is_absolute() != base.is_absolute() {
            if path.is_absolute() {
                Some(path.to_path_buf())
            } else {
                None
            }
        } else {
            let mut ita = path.components();
            let mut itb = base.components();
            let mut comps: Vec<Component> = vec![];
            loop {
                match (ita.next(), itb.next()) {
                    (None, None) => break,
                    (Some(a), None) => {
                        comps.push(a);
                        comps.extend(ita.by_ref());
                        break;
                    }
                    (None, _) => comps.push(Component::ParentDir),
                    (Some(a), Some(b)) if comps.is_empty() && a == b => (),
                    (Some(a), Some(_)) => {
                        comps.push(Component::ParentDir);
                        for _ in itb {
                            comps.push(Component::ParentDir);
                        }
                        comps.push(a);
                        comps.extend(ita.by_ref());
                        break;
                    }
                }
            }
            Some(comps.iter().map(|c| c.as_os_str()).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use std::fs::File;
    use std::io::Write;
    
    #[test]
    fn test_environment_operations() {
        // Test setting and getting environment variable
        let result = set_environment(&[
            Value::String("TEST_VAR".to_string()),
            Value::String("test_value".to_string())
        ]);
        assert!(result.is_ok());
        
        let result = environment(&[Value::String("TEST_VAR".to_string())]);
        assert_eq!(result.unwrap(), Value::String("test_value".to_string()));
        
        // Test unsetting
        let result = unset_environment(&[Value::String("TEST_VAR".to_string())]);
        assert!(result.is_ok());
        
        let result = environment(&[Value::String("TEST_VAR".to_string())]);
        assert_eq!(result.unwrap(), Value::Missing);
    }
    
    #[test]
    fn test_file_operations() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("test.txt");
        
        // Create a test file
        let mut file = File::create(&file_path).unwrap();
        writeln!(file, "test content").unwrap();
        
        // Test file_exists
        let result = file_exists(&[Value::String(file_path.to_string_lossy().to_string())]);
        assert_eq!(result.unwrap(), Value::Boolean(true));
        
        // Test file_size
        let result = file_size(&[Value::String(file_path.to_string_lossy().to_string())]);
        assert!(matches!(result.unwrap(), Value::Integer(_)));
        
        // Test is_file
        let result = is_file(&[Value::String(file_path.to_string_lossy().to_string())]);
        assert_eq!(result.unwrap(), Value::Boolean(true));
        
        // Test is_directory
        let result = is_directory(&[Value::String(file_path.to_string_lossy().to_string())]);
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }
    
    #[test]
    fn test_directory_operations() {
        let temp_dir = TempDir::new().unwrap();
        let dir_path = temp_dir.path().join("test_dir");
        
        // Test create_directory
        let result = create_directory(&[Value::String(dir_path.to_string_lossy().to_string())]);
        assert!(result.is_ok());
        
        // Test directory_exists
        let result = directory_exists(&[Value::String(dir_path.to_string_lossy().to_string())]);
        assert_eq!(result.unwrap(), Value::Boolean(true));
        
        // Test is_directory
        let result = is_directory(&[Value::String(dir_path.to_string_lossy().to_string())]);
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_path_operations() {
        // Test path_join
        let result = path_join(&[
            Value::String("home".to_string()),
            Value::String("user".to_string()),
            Value::String("documents".to_string()),
        ]);
        assert!(result.is_ok());
        
        // Test file_name
        let result = file_name(&[Value::String("/path/to/file.txt".to_string())]);
        assert_eq!(result.unwrap(), Value::String("file.txt".to_string()));
        
        // Test file_extension
        let result = file_extension(&[Value::String("file.txt".to_string())]);
        assert_eq!(result.unwrap(), Value::String(".txt".to_string()));
        
        // Test directory_name
        let result = directory_name(&[Value::String("/path/to/file.txt".to_string())]);
        assert_eq!(result.unwrap(), Value::String("/path/to".to_string()));
    }
    
    #[test]
    fn test_system_info_operations() {
        // Test current_directory
        let result = current_directory(&[]);
        assert!(result.is_ok());
        
        // Test current_user
        let result = current_user(&[]);
        assert!(result.is_ok());
        
        // Test system_architecture
        let result = system_architecture(&[]);
        assert!(result.is_ok());
        
        // Test temp_directory
        let result = temp_directory(&[]);
        assert!(result.is_ok());
        
        // Test home_directory
        let result = home_directory(&[]);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_process_operations() {
        // Test current_pid
        let result = current_pid(&[]);
        assert!(matches!(result.unwrap(), Value::Integer(_)));
        
        // Test process_exists with current PID
        let current_pid_val = current_pid(&[]).unwrap();
        let result = process_exists(&[current_pid_val]);
        assert_eq!(result.unwrap(), Value::Boolean(true));
        
        // Test run_command
        let result = run_command(&[
            Value::String("echo".to_string()),
            Value::List(vec![Value::String("hello".to_string())]),
        ]);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_security_validation() {
        // Test directory traversal prevention
        let result = validate_path("../etc/passwd");
        assert!(result.is_err());
        
        // Test command injection prevention
        let result = sanitize_command_args("ls; rm -rf /", &[]);
        assert!(result.is_err());
        
        let result = sanitize_command_args("ls", &["file.txt; rm -rf /".to_string()]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_foreign_objects() {
        let temp_dir = TempDir::new().unwrap();
        
        // Test DirectoryListing foreign object
        let listing = DirectoryListing::new(temp_dir.path()).unwrap();
        let lyobj = LyObj::new(Box::new(listing));
        
        // Test method calls
        let result = lyobj.call_method("count", &[]);
        assert!(result.is_ok());
        
        let result = lyobj.call_method("names", &[]);
        assert!(result.is_ok());
        
        // Test ProcessHandle foreign object
        let handle = ProcessHandle::new(12345, "test".to_string(), vec!["arg1".to_string()]);
        let lyobj = LyObj::new(Box::new(handle));
        
        let result = lyobj.call_method("pid", &[]);
        assert_eq!(result.unwrap(), Value::Integer(12345));
        
        let result = lyobj.call_method("command", &[]);
        assert_eq!(result.unwrap(), Value::String("test".to_string()));
    }
    
    #[test]
    fn test_error_handling() {
        // Test invalid arguments
        let result = environment(&[]);
        assert!(result.is_err());
        
        let result = file_exists(&[Value::Integer(123)]);
        assert!(result.is_err());
        
        // Test non-existent file operations
        let result = file_size(&[Value::String("/nonexistent/file.txt".to_string())]);
        assert!(result.is_err());
        
        let result = delete_file(&[Value::String("/nonexistent/file.txt".to_string())]);
        assert!(result.is_err());
    }
}