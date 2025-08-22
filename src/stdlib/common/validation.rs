//! Parameter validation utilities for Lyra standard library functions
//!
//! This module provides consistent parameter validation across all stdlib functions,
//! ensuring type safety and proper argument handling with clear error messages.

use crate::vm::Value;
use thiserror::Error;

/// Validation error type for parameter checking
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ValidationError {
    #[error("Argument count error: expected {expected}, got {actual}")]
    ArgumentCount { expected: usize, actual: usize },
    
    #[error("Argument count error: expected {min} to {max} arguments, got {actual}")]
    ArgumentRange { min: usize, max: usize, actual: usize },
    
    #[error("Type error: expected {expected} at position {position}, got {actual}")]
    TypeError { expected: String, actual: String, position: usize },
    
    #[error("Value error: {message} at position {position}")]
    ValueError { message: String, position: usize },
    
    #[error("List error: expected non-empty list at position {position}")]
    EmptyList { position: usize },
    
    #[error("Range error: {message}")]
    RangeError { message: String },
}

/// Validate exact argument count
pub fn validate_args(args: &[Value], expected: usize) -> Result<(), ValidationError> {
    if args.len() != expected {
        Err(ValidationError::ArgumentCount {
            expected,
            actual: args.len(),
        })
    } else {
        Ok(())
    }
}

/// Validate argument count within a range
pub fn validate_args_range(args: &[Value], min: usize, max: usize) -> Result<(), ValidationError> {
    let actual = args.len();
    if actual < min || actual > max {
        Err(ValidationError::ArgumentRange { min, max, actual })
    } else {
        Ok(())
    }
}

/// Validate argument type at specific position
pub fn validate_type<'a>(args: &'a [Value], position: usize, expected_type: &str) -> Result<&'a Value, ValidationError> {
    if position >= args.len() {
        return Err(ValidationError::ArgumentCount {
            expected: position + 1,
            actual: args.len(),
        });
    }
    
    let value = &args[position];
    let actual_type = match value {
        Value::Integer(_) => "Integer",
        Value::Real(_) => "Real", 
        Value::String(_) => "String",
        Value::Symbol(_) => "Symbol",
        Value::List(_) => "List",
        Value::Boolean(_) => "Boolean",
        Value::Function(_) => "Function",
        Value::Missing => "Missing",
        Value::Object(_) => "Object",
        Value::LyObj(_) => "LyObj",
        Value::Quote(_) => "Quote",
        Value::Pattern(_) => "Pattern",
        Value::Rule { .. } => "Rule",
        Value::PureFunction { .. } => "PureFunction",
        Value::Slot { .. } => "Slot",
    };
    
    // Check if the actual type matches expected or is compatible
    let type_matches = match expected_type {
        "Number" => matches!(value, Value::Integer(_) | Value::Real(_)),
        "List" => matches!(value, Value::List(_)),
        "String" => matches!(value, Value::String(_)),
        "Symbol" => matches!(value, Value::Symbol(_)),
        "Integer" => matches!(value, Value::Integer(_)),
        "Real" => matches!(value, Value::Real(_)),
        "Object" => matches!(value, Value::LyObj(_)),
        _ => actual_type == expected_type,
    };
    
    if type_matches {
        Ok(value)
    } else {
        Err(ValidationError::TypeError {
            expected: expected_type.to_string(),
            actual: actual_type.to_string(),
            position,
        })
    }
}

/// Extract integer from Value with validation
pub fn extract_integer(args: &[Value], position: usize) -> Result<i64, ValidationError> {
    let value = validate_type(args, position, "Integer")?;
    match value {
        Value::Integer(i) => Ok(*i),
        _ => unreachable!("Type validation should have caught this"),
    }
}

/// Extract real number from Value with validation (accepts both Integer and Real)
pub fn extract_number(args: &[Value], position: usize) -> Result<f64, ValidationError> {
    let value = validate_type(args, position, "Number")?;
    match value {
        Value::Integer(i) => Ok(*i as f64),
        Value::Real(r) => Ok(*r),
        _ => unreachable!("Type validation should have caught this"),
    }
}

/// Extract string from Value with validation
pub fn extract_string(args: &[Value], position: usize) -> Result<&str, ValidationError> {
    let value = validate_type(args, position, "String")?;
    match value {
        Value::String(s) => Ok(s),
        _ => unreachable!("Type validation should have caught this"),
    }
}

/// Extract list from Value with validation
pub fn extract_list(args: &[Value], position: usize) -> Result<&[Value], ValidationError> {
    let value = validate_type(args, position, "List")?;
    match value {
        Value::List(list) => Ok(list),
        _ => unreachable!("Type validation should have caught this"),
    }
}

/// Extract non-empty list from Value with validation
pub fn extract_non_empty_list(args: &[Value], position: usize) -> Result<&[Value], ValidationError> {
    let list = extract_list(args, position)?;
    if list.is_empty() {
        Err(ValidationError::EmptyList { position })
    } else {
        Ok(list)
    }
}

/// Extract symbol from Value with validation
pub fn extract_symbol(args: &[Value], position: usize) -> Result<&str, ValidationError> {
    let value = validate_type(args, position, "Symbol")?;
    match value {
        Value::Symbol(s) => Ok(s),
        _ => unreachable!("Type validation should have caught this"),
    }
}

/// Validate that a number is positive
pub fn validate_positive(value: f64, position: usize) -> Result<f64, ValidationError> {
    if value > 0.0 {
        Ok(value)
    } else {
        Err(ValidationError::ValueError {
            message: format!("expected positive number, got {}", value),
            position,
        })
    }
}

/// Validate that a number is non-negative
pub fn validate_non_negative(value: f64, position: usize) -> Result<f64, ValidationError> {
    if value >= 0.0 {
        Ok(value)
    } else {
        Err(ValidationError::ValueError {
            message: format!("expected non-negative number, got {}", value),
            position,
        })
    }
}

/// Validate that an integer is within a range
pub fn validate_integer_range(value: i64, min: i64, max: i64, position: usize) -> Result<i64, ValidationError> {
    if value >= min && value <= max {
        Ok(value)
    } else {
        Err(ValidationError::RangeError {
            message: format!("value {} at position {} not in range [{}, {}]", value, position, min, max),
        })
    }
}

/// Validate that a list has a specific length
pub fn validate_list_length(list: &[Value], expected: usize, position: usize) -> Result<(), ValidationError> {
    if list.len() != expected {
        Err(ValidationError::ValueError {
            message: format!("expected list of length {}, got {}", expected, list.len()),
            position,
        })
    } else {
        Ok(())
    }
}

/// Validate that all elements in a list are numbers
pub fn validate_numeric_list(list: &[Value], position: usize) -> Result<Vec<f64>, ValidationError> {
    let mut numbers = Vec::with_capacity(list.len());
    for (i, value) in list.iter().enumerate() {
        match value {
            Value::Integer(n) => numbers.push(*n as f64),
            Value::Real(n) => numbers.push(*n),
            _ => {
                let actual_type = match value {
                    Value::String(_) => "String",
                    Value::Symbol(_) => "Symbol", 
                    Value::List(_) => "List",
                    Value::LyObj(_) => "Object",
                    _ => "Unknown",
                };
                return Err(ValidationError::TypeError {
                    expected: "Number".to_string(),
                    actual: actual_type.to_string(),
                    position,
                });
            }
        }
    }
    Ok(numbers)
}

/// Validate that all elements in a list are integers
pub fn validate_integer_list(list: &[Value], position: usize) -> Result<Vec<i64>, ValidationError> {
    let mut integers = Vec::with_capacity(list.len());
    for (i, value) in list.iter().enumerate() {
        match value {
            Value::Integer(n) => integers.push(*n),
            _ => {
                let actual_type = match value {
                    Value::Real(_) => "Real",
                    Value::String(_) => "String",
                    Value::Symbol(_) => "Symbol",
                    Value::List(_) => "List", 
                    Value::LyObj(_) => "Object",
                    _ => "Unknown",
                };
                return Err(ValidationError::TypeError {
                    expected: "Integer".to_string(),
                    actual: actual_type.to_string(),
                    position,
                });
            }
        }
    }
    Ok(integers)
}

/// Macro for validating arguments with custom error messages
#[macro_export]
macro_rules! validate {
    ($condition:expr, $error:expr) => {
        if !$condition {
            return Err($error);
        }
    };
    ($args:expr, $expected:expr) => {
        $crate::stdlib::common::validation::validate_args($args, $expected)?
    };
    ($args:expr, $min:expr, $max:expr) => {
        $crate::stdlib::common::validation::validate_args_range($args, $min, $max)?
    };
}

/// Convenience macros for extracting and validating common argument patterns
#[macro_export]
macro_rules! extract_args {
    ($args:expr; $($name:ident: $type:ident at $pos:expr),+ $(,)?) => {
        $(
            let $name = extract_args!(@extract $args, $pos, $type)?;
        )+
    };
    
    (@extract $args:expr, $pos:expr, integer) => {
        $crate::stdlib::common::validation::extract_integer($args, $pos)
    };
    (@extract $args:expr, $pos:expr, number) => {
        $crate::stdlib::common::validation::extract_number($args, $pos)
    };
    (@extract $args:expr, $pos:expr, string) => {
        $crate::stdlib::common::validation::extract_string($args, $pos)
    };
    (@extract $args:expr, $pos:expr, list) => {
        $crate::stdlib::common::validation::extract_list($args, $pos)
    };
    (@extract $args:expr, $pos:expr, non_empty_list) => {
        $crate::stdlib::common::validation::extract_non_empty_list($args, $pos)
    };
    (@extract $args:expr, $pos:expr, symbol) => {
        $crate::stdlib::common::validation::extract_symbol($args, $pos)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_args() {
        let args = vec![Value::Integer(1), Value::Real(2.0)];
        
        // Should pass with correct count
        assert!(validate_args(&args, 2).is_ok());
        
        // Should fail with wrong count
        assert!(validate_args(&args, 1).is_err());
        assert!(validate_args(&args, 3).is_err());
    }
    
    #[test]
    fn test_validate_type() {
        let args = vec![Value::Integer(42), Value::String("hello".to_string())];
        
        // Should pass with correct types
        assert!(validate_type(&args, 0, "Integer").is_ok());
        assert!(validate_type(&args, 0, "Number").is_ok()); // Integer is also Number
        assert!(validate_type(&args, 1, "String").is_ok());
        
        // Should fail with wrong types
        assert!(validate_type(&args, 0, "String").is_err());
        assert!(validate_type(&args, 1, "Integer").is_err());
    }
    
    #[test]
    fn test_extract_functions() {
        let args = vec![
            Value::Integer(42),
            Value::Real(3.14),
            Value::String("test".to_string()),
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        ];
        
        assert_eq!(extract_integer(&args, 0).unwrap(), 42);
        assert_eq!(extract_number(&args, 1).unwrap(), 3.14);
        assert_eq!(extract_number(&args, 0).unwrap(), 42.0); // Integer as number
        assert_eq!(extract_string(&args, 2).unwrap(), "test");
        assert_eq!(extract_list(&args, 3).unwrap().len(), 2);
    }
    
    #[test]
    fn test_validate_numeric_list() {
        let list = vec![Value::Integer(1), Value::Real(2.5), Value::Integer(3)];
        let numbers = validate_numeric_list(&list, 0).unwrap();
        assert_eq!(numbers, vec![1.0, 2.5, 3.0]);
        
        let mixed_list = vec![Value::Integer(1), Value::String("not a number".to_string())];
        assert!(validate_numeric_list(&mixed_list, 0).is_err());
    }
}