//! Utilities Module for Lyra Standard Library
//!
//! This module provides comprehensive utility functionality including:
//! - I/O operations (file handling, system interaction)
//! - System utilities (environment, process management)
//! - Time and date operations (temporal calculations)
//! - Developer tools (debugging, profiling, testing)
//! - Documentation generation and management
//! - Result and Option types for error handling
//! - Security operations (cryptography, validation)

pub mod system;
pub mod temporal;
pub mod developer;
pub mod documentation;
pub mod result;
pub mod security;
pub mod serialization;
pub mod config;
pub mod cache;

// Re-export all utility functions for convenience
pub use system::*;
pub use temporal::*;
pub use developer::*;
pub use documentation::*;
pub use result::*;
pub use security::*;
pub use serialization::*;
pub use config::*;
pub use cache::*;

use crate::vm::{Value, VmResult};

/// Register all utility functions for the standard library
pub fn register_utilities_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = std::collections::HashMap::new();
    
    // I/O operations
    functions.extend(register_io_functions());
    
    // System utilities
    functions.extend(register_system_functions());
    
    // Temporal operations
    functions.extend(register_temporal_functions());
    
    // Developer tools
    functions.extend(register_developer_functions());
    
    // Documentation functions
    functions.extend(register_documentation_functions());
    
    // Result and Option types
    functions.extend(register_result_functions());
    
    // Security operations
    functions.extend(security::register_security_functions());
    
    functions
}

// Function registration helpers for utilities modules
fn register_io_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = std::collections::HashMap::new();
    
    // Core I/O functions
    functions.insert("Import".to_string(), crate::stdlib::io::import as fn(&[Value]) -> VmResult<Value>);
    // functions.insert("Export".to_string(), crate::stdlib::io::export);
    
    functions
}

fn register_system_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    std::collections::HashMap::new() // TODO: Implement when system functions are defined
}

fn register_temporal_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    std::collections::HashMap::new() // TODO: Implement when temporal functions are defined
}

fn register_developer_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    std::collections::HashMap::new() // TODO: Implement when developer functions are defined
}

fn register_documentation_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    std::collections::HashMap::new() // TODO: Implement when documentation functions are defined
}

fn register_result_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = std::collections::HashMap::new();
    
    // Result constructors
    functions.insert("Ok".to_string(), result::ok_constructor as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Error".to_string(), result::error_constructor as fn(&[Value]) -> VmResult<Value>);
    
    // Option constructors
    functions.insert("Some".to_string(), result::some_constructor as fn(&[Value]) -> VmResult<Value>);
    functions.insert("None".to_string(), result::none_constructor as fn(&[Value]) -> VmResult<Value>);
    
    // Result operations
    functions.insert("ResultIsOk".to_string(), result::result_is_ok as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ResultIsError".to_string(), result::result_is_error as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ResultUnwrap".to_string(), result::result_unwrap as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ResultUnwrapOr".to_string(), result::result_unwrap_or as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ResultMap".to_string(), result::result_map as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ResultAndThen".to_string(), result::result_and_then as fn(&[Value]) -> VmResult<Value>);
    
    // Option operations
    functions.insert("OptionIsSome".to_string(), result::option_is_some as fn(&[Value]) -> VmResult<Value>);
    functions.insert("OptionIsNone".to_string(), result::option_is_none as fn(&[Value]) -> VmResult<Value>);
    functions.insert("OptionUnwrap".to_string(), result::option_unwrap as fn(&[Value]) -> VmResult<Value>);
    functions.insert("OptionUnwrapOr".to_string(), result::option_unwrap_or as fn(&[Value]) -> VmResult<Value>);
    functions.insert("OptionMap".to_string(), result::option_map as fn(&[Value]) -> VmResult<Value>);
    functions.insert("OptionAndThen".to_string(), result::option_and_then as fn(&[Value]) -> VmResult<Value>);
    
    functions
}
