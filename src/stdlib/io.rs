//! I/O functions for the Lyra standard library
//! 
//! This module provides Import/Export functionality with comprehensive format support.

use crate::vm::{Value, VmResult, VmError};

/// Import[filename] - Load data from external file
/// Import[filename, format] - Load data with specific format
/// 
/// Examples:
/// - `Import["data.json"]` → Load JSON data with auto-detection
/// - `Import["data.csv", "CSV"]` → Load CSV with explicit format
/// - `Import["config.txt"]` → Load text file
pub fn import(args: &[Value]) -> VmResult<Value> {
    crate::io::import(args)
}

/// Export[data, filename] - Save data to external file  
/// Export[data, filename, format] - Save data with specific format
/// 
/// Examples:
/// - `Export[data, "output.json"]` → Save as JSON with auto-detection
/// - `Export[data, "output.csv", "CSV"]` → Save as CSV with explicit format
/// - `Export[text, "output.txt"]` → Save as plain text
pub fn export(args: &[Value]) -> VmResult<Value> {
    crate::io::export(args)
}