//! String Operations Module
//!
//! This module provides comprehensive string processing capabilities including:
//! - Basic string operations (join, length, take, drop)
//! - Advanced string processing (regex, templating, encoding)
//! - Case transformations and formatting
//! - String validation and manipulation

pub mod basic;
pub mod advanced;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Register all string functions with the standard library
pub fn register_string_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();

    // Basic string operations (from basic.rs)
    functions.insert("StringJoin".to_string(), basic::string_join as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringLength".to_string(), basic::string_length as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringTake".to_string(), basic::string_take as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringDrop".to_string(), basic::string_drop as fn(&[Value]) -> VmResult<Value>);

    // Advanced string operations (from advanced.rs)
    functions.insert("StringTemplate".to_string(), advanced::string_template as fn(&[Value]) -> VmResult<Value>);
    functions.insert("RegularExpression".to_string(), advanced::regular_expression as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringMatch".to_string(), advanced::string_match as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringExtract".to_string(), advanced::string_extract as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringReplace".to_string(), advanced::string_replace as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringSplit".to_string(), advanced::string_split as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringTrim".to_string(), advanced::string_trim as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringContains".to_string(), advanced::string_contains as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringStartsWith".to_string(), advanced::string_starts_with as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringEndsWith".to_string(), advanced::string_ends_with as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringReverse".to_string(), advanced::string_reverse as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StringRepeat".to_string(), advanced::string_repeat as fn(&[Value]) -> VmResult<Value>);
    
    // Case transformation functions
    functions.insert("ToUpperCase".to_string(), advanced::to_upper_case as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ToLowerCase".to_string(), advanced::to_lower_case as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TitleCase".to_string(), advanced::title_case as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CamelCase".to_string(), advanced::camel_case as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SnakeCase".to_string(), advanced::snake_case as fn(&[Value]) -> VmResult<Value>);
    
    // Encoding/decoding functions
    functions.insert("Base64Encode".to_string(), advanced::base64_encode as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Base64Decode".to_string(), advanced::base64_decode as fn(&[Value]) -> VmResult<Value>);
    functions.insert("URLEncode".to_string(), advanced::url_encode as fn(&[Value]) -> VmResult<Value>);
    functions.insert("URLDecode".to_string(), advanced::url_decode as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HTMLEscape".to_string(), advanced::html_escape as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HTMLUnescape".to_string(), advanced::html_unescape as fn(&[Value]) -> VmResult<Value>);
    functions.insert("JSONEscape".to_string(), advanced::json_escape as fn(&[Value]) -> VmResult<Value>);
    
    // String formatting
    functions.insert("StringFormat".to_string(), advanced::string_format as fn(&[Value]) -> VmResult<Value>);

    functions
}

// Re-export all public functions for convenience
pub use basic::*;
pub use advanced::*;