//! Compression algorithms for Lyra standard library
//!
//! This module implements basic compression and encoding algorithms.

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;

/// Run-length encoding
pub fn run_length_encode(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let input = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("RunLengthEncode expects a string".to_string())),
    };
    
    if input.is_empty() {
        return Ok(Value::String("".to_string()));
    }
    
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut current_char = chars[0];
    let mut count = 1;
    
    for &ch in &chars[1..] {
        if ch == current_char {
            count += 1;
        } else {
            result.push_str(&format!("{}{}", count, current_char));
            current_char = ch;
            count = 1;
        }
    }
    
    // Don't forget the last run
    result.push_str(&format!("{}{}", count, current_char));
    
    Ok(Value::String(result))
}

/// Run-length decoding
pub fn run_length_decode(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let input = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("RunLengthDecode expects a string".to_string())),
    };
    
    if input.is_empty() {
        return Ok(Value::String("".to_string()));
    }
    
    let mut result = String::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    
    while i < chars.len() {
        // Read count
        let mut count_str = String::new();
        while i < chars.len() && chars[i].is_ascii_digit() {
            count_str.push(chars[i]);
            i += 1;
        }
        
        if count_str.is_empty() {
            return Err(VmError::Runtime("Invalid RLE format: expected count".to_string()));
        }
        
        let count: usize = count_str.parse()
            .map_err(|_| VmError::Runtime("Invalid RLE format: invalid count".to_string()))?;
        
        // Read character
        if i >= chars.len() {
            return Err(VmError::Runtime("Invalid RLE format: expected character".to_string()));
        }
        
        let ch = chars[i];
        i += 1;
        
        // Repeat character count times
        for _ in 0..count {
            result.push(ch);
        }
    }
    
    Ok(Value::String(result))
}

/// Register compression functions
pub fn register_compression_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("RunLengthEncode".to_string(), run_length_encode);
    functions.insert("RunLengthDecode".to_string(), run_length_decode);
}

/// Get documentation for compression algorithms
pub fn get_compression_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("RunLengthEncode".to_string(), "RunLengthEncode[string] - Encode string using run-length encoding. E.g., \"aaabbc\" → \"3a2b1c\".".to_string());
    docs.insert("RunLengthDecode".to_string(), "RunLengthDecode[encoded] - Decode run-length encoded string. E.g., \"3a2b1c\" → \"aaabbc\".".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_run_length_encode_basic() {
        let args = vec![Value::String("aaabbc".to_string())];
        let result = run_length_encode(&args).unwrap();
        assert_eq!(result, Value::String("3a2b1c".to_string()));
    }
    
    #[test]
    fn test_run_length_encode_single_chars() {
        let args = vec![Value::String("abcde".to_string())];
        let result = run_length_encode(&args).unwrap();
        assert_eq!(result, Value::String("1a1b1c1d1e".to_string()));
    }
    
    #[test]
    fn test_run_length_encode_empty() {
        let args = vec![Value::String("".to_string())];
        let result = run_length_encode(&args).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }
    
    #[test]
    fn test_run_length_encode_long_run() {
        let args = vec![Value::String("aaaaaaaaaa".to_string())];
        let result = run_length_encode(&args).unwrap();
        assert_eq!(result, Value::String("10a".to_string()));
    }
    
    #[test]
    fn test_run_length_decode_basic() {
        let args = vec![Value::String("3a2b1c".to_string())];
        let result = run_length_decode(&args).unwrap();
        assert_eq!(result, Value::String("aaabbc".to_string()));
    }
    
    #[test]
    fn test_run_length_decode_single_chars() {
        let args = vec![Value::String("1a1b1c1d1e".to_string())];
        let result = run_length_decode(&args).unwrap();
        assert_eq!(result, Value::String("abcde".to_string()));
    }
    
    #[test]
    fn test_run_length_decode_empty() {
        let args = vec![Value::String("".to_string())];
        let result = run_length_decode(&args).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }
    
    #[test]
    fn test_run_length_decode_long_run() {
        let args = vec![Value::String("10a".to_string())];
        let result = run_length_decode(&args).unwrap();
        assert_eq!(result, Value::String("aaaaaaaaaa".to_string()));
    }
    
    #[test]
    fn test_run_length_roundtrip() {
        let test_cases = vec![
            "aaabbc",
            "abcde",
            "aabbccddee",
            "xxxxxxxxxx",
            "a",
            "ab",
            "",
        ];
        
        for test_case in test_cases {
            let encode_args = vec![Value::String(test_case.to_string())];
            let encoded = run_length_encode(&encode_args).unwrap();
            
            if let Value::String(encoded_str) = encoded {
                let decode_args = vec![Value::String(encoded_str)];
                let decoded = run_length_decode(&decode_args).unwrap();
                assert_eq!(decoded, Value::String(test_case.to_string()));
            } else {
                panic!("Expected string result from encoding");
            }
        }
    }
    
    #[test]
    fn test_run_length_decode_invalid_format() {
        // Invalid: no count
        let args = vec![Value::String("a".to_string())];
        assert!(run_length_decode(&args).is_err());
        
        // Invalid: count without character
        let args = vec![Value::String("3".to_string())];
        assert!(run_length_decode(&args).is_err());
        
        // Invalid: non-numeric count
        let args = vec![Value::String("xa".to_string())];
        assert!(run_length_decode(&args).is_err());
    }
    
    #[test]
    fn test_run_length_encode_invalid_type() {
        let args = vec![Value::Integer(42)];
        assert!(run_length_encode(&args).is_err());
    }
    
    #[test]
    fn test_run_length_decode_invalid_type() {
        let args = vec![Value::Integer(42)];
        assert!(run_length_decode(&args).is_err());
    }
}