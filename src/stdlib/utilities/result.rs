//! Result and Option types for production-ready error handling
//!
//! This module implements Result[T, E] and Option[T] types as standard library
//! constructs, following Rust-like semantics but with Wolfram Language syntax.
//!
//! Examples:
//! ```lyra
//! (* Creating results *)
//! success = Ok[42]
//! failure = Error["Something went wrong"]
//! 
//! (* Pattern matching *)
//! result.match[{
//!   Ok[value_] :> Print["Success: " <> ToString[value]],
//!   Error[msg_] :> Print["Error: " <> msg]
//! }]
//! 
//! (* Chaining operations *)
//! result.map[x => x * 2].unwrapOr[0]
//! ```

use crate::vm::{Value, VmResult, VmError};


/// Create a Result with Ok variant
/// Usage: Ok[value]
pub fn ok_constructor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "Ok expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    // Create a list with tag "Ok" and the value
    Ok(Value::List(vec![
        Value::Symbol("Ok".to_string()),
        args[0].clone(),
    ]))
}

/// Create a Result with Error variant  
/// Usage: Error[message]
pub fn error_constructor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "Error expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    // Create a list with tag "Error" and the error value
    Ok(Value::List(vec![
        Value::Symbol("Error".to_string()),
        args[0].clone(),
    ]))
}

/// Create an Option with Some variant
/// Usage: Some[value] 
pub fn some_constructor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "Some expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    Ok(Value::List(vec![
        Value::Symbol("Some".to_string()),
        args[0].clone(),
    ]))
}

/// Create an Option with None variant
/// Usage: None[]
pub fn none_constructor(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime(format!(
            "None expects no arguments, got {}", args.len()
        )));
    }
    
    Ok(Value::List(vec![
        Value::Symbol("None".to_string()),
    ]))
}

/// Check if a Result is Ok
/// Usage: ResultIsOk[result]
pub fn result_is_ok(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "ResultIsOk expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 1 => {
            if let Value::Symbol(tag) = &items[0] {
                Ok(Value::Boolean(tag == "Ok"))
            } else {
                Ok(Value::Boolean(false))
            }
        }
        _ => Ok(Value::Boolean(false)),
    }
}

/// Check if a Result is Error
/// Usage: ResultIsError[result]
pub fn result_is_error(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "ResultIsError expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 1 => {
            if let Value::Symbol(tag) = &items[0] {
                Ok(Value::Boolean(tag == "Error"))
            } else {
                Ok(Value::Boolean(false))
            }
        }
        _ => Ok(Value::Boolean(false)),
    }
}

/// Unwrap a Result, panicking on Error
/// Usage: ResultUnwrap[result]
pub fn result_unwrap(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "ResultUnwrap expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Ok" => Ok(items[1].clone()),
                    "Error" => Err(VmError::Runtime(format!(
                        "Called ResultUnwrap on Error value: {:?}", items[1]
                    ))),
                    _ => Err(VmError::Runtime(
                        "ResultUnwrap called on non-Result value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "ResultUnwrap called on non-Result value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "ResultUnwrap called on non-Result value".to_string()
        )),
    }
}

/// Unwrap a Result or return a default value
/// Usage: ResultUnwrapOr[result, default]
pub fn result_unwrap_or(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "ResultUnwrapOr expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Ok" => Ok(items[1].clone()),
                    "Error" => Ok(args[1].clone()),
                    _ => Ok(args[1].clone()),
                }
            } else {
                Ok(args[1].clone())
            }
        }
        _ => Ok(args[1].clone()),
    }
}

/// Map a function over a Result's Ok value
/// Usage: ResultMap[result, function]
pub fn result_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "ResultMap expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Ok" => {
                        // Apply function to the Ok value
                        // TODO: Implement function application
                        // For now, just return the original result
                        Ok(args[0].clone())
                    }
                    "Error" => Ok(args[0].clone()), // Pass through errors
                    _ => Err(VmError::Runtime(
                        "ResultMap called on non-Result value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "ResultMap called on non-Result value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "ResultMap called on non-Result value".to_string()
        )),
    }
}

/// Chain a function that returns a Result
/// Usage: ResultAndThen[result, function]
pub fn result_and_then(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "ResultAndThen expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Ok" => {
                        // Apply function to the Ok value - function should return Result
                        // TODO: Implement function application
                        // For now, just return the original result
                        Ok(args[0].clone())
                    }
                    "Error" => Ok(args[0].clone()), // Pass through errors
                    _ => Err(VmError::Runtime(
                        "ResultAndThen called on non-Result value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "ResultAndThen called on non-Result value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "ResultAndThen called on non-Result value".to_string()
        )),
    }
}

/// Check if an Option is Some
/// Usage: OptionIsSome[option]
pub fn option_is_some(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "OptionIsSome expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if !items.is_empty() => {
            if let Value::Symbol(tag) = &items[0] {
                Ok(Value::Boolean(tag == "Some"))
            } else {
                Ok(Value::Boolean(false))
            }
        }
        _ => Ok(Value::Boolean(false)),
    }
}

/// Check if an Option is None
/// Usage: OptionIsNone[option]  
pub fn option_is_none(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "OptionIsNone expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if !items.is_empty() => {
            if let Value::Symbol(tag) = &items[0] {
                Ok(Value::Boolean(tag == "None"))
            } else {
                Ok(Value::Boolean(false))
            }
        }
        _ => Ok(Value::Boolean(false)),
    }
}

/// Unwrap an Option, panicking on None
/// Usage: OptionUnwrap[option]
pub fn option_unwrap(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "OptionUnwrap expects exactly 1 argument, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Some" => Ok(items[1].clone()),
                    "None" => Err(VmError::Runtime(
                        "Called OptionUnwrap on None value".to_string()
                    )),
                    _ => Err(VmError::Runtime(
                        "OptionUnwrap called on non-Option value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "OptionUnwrap called on non-Option value".to_string()
                ))
            }
        }
        Value::List(items) if items.len() == 1 => {
            if let Value::Symbol(tag) = &items[0] {
                if tag == "None" {
                    Err(VmError::Runtime(
                        "Called OptionUnwrap on None value".to_string()
                    ))
                } else {
                    Err(VmError::Runtime(
                        "OptionUnwrap called on non-Option value".to_string()
                    ))
                }
            } else {
                Err(VmError::Runtime(
                    "OptionUnwrap called on non-Option value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "OptionUnwrap called on non-Option value".to_string()
        )),
    }
}

/// Unwrap an Option or return a default value
/// Usage: OptionUnwrapOr[option, default]
pub fn option_unwrap_or(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "OptionUnwrapOr expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Some" => Ok(items[1].clone()),
                    "None" => Ok(args[1].clone()),
                    _ => Ok(args[1].clone()),
                }
            } else {
                Ok(args[1].clone())
            }
        }
        Value::List(items) if items.len() == 1 => {
            if let Value::Symbol(tag) = &items[0] {
                if tag == "None" {
                    Ok(args[1].clone())
                } else {
                    Ok(args[1].clone())
                }
            } else {
                Ok(args[1].clone())
            }
        }
        _ => Ok(args[1].clone()),
    }
}

/// Map a function over an Option's Some value
/// Usage: OptionMap[option, function]  
pub fn option_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "OptionMap expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Some" => {
                        // Apply function to the Some value
                        // TODO: Implement function application
                        // For now, just return the original option
                        Ok(args[0].clone())
                    }
                    "None" => Ok(args[0].clone()), // Pass through None
                    _ => Err(VmError::Runtime(
                        "OptionMap called on non-Option value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "OptionMap called on non-Option value".to_string()
                ))
            }
        }
        Value::List(items) if items.len() == 1 => {
            if let Value::Symbol(tag) = &items[0] {
                if tag == "None" {
                    Ok(args[0].clone()) // Pass through None
                } else {
                    Err(VmError::Runtime(
                        "OptionMap called on non-Option value".to_string()
                    ))
                }
            } else {
                Err(VmError::Runtime(
                    "OptionMap called on non-Option value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "OptionMap called on non-Option value".to_string()
        )),
    }
}

/// Chain a function that returns an Option
/// Usage: OptionAndThen[option, function]
pub fn option_and_then(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "OptionAndThen expects exactly 2 arguments, got {}", args.len()
        )));
    }
    
    match &args[0] {
        Value::List(items) if items.len() >= 2 => {
            if let Value::Symbol(tag) = &items[0] {
                match tag.as_str() {
                    "Some" => {
                        // Apply function to the Some value - function should return Option
                        // TODO: Implement function application
                        // For now, just return the original option
                        Ok(args[0].clone())
                    }
                    "None" => Ok(args[0].clone()), // Pass through None
                    _ => Err(VmError::Runtime(
                        "OptionAndThen called on non-Option value".to_string()
                    )),
                }
            } else {
                Err(VmError::Runtime(
                    "OptionAndThen called on non-Option value".to_string()
                ))
            }
        }
        Value::List(items) if items.len() == 1 => {
            if let Value::Symbol(tag) = &items[0] {
                if tag == "None" {
                    Ok(args[0].clone()) // Pass through None
                } else {
                    Err(VmError::Runtime(
                        "OptionAndThen called on non-Option value".to_string()
                    ))
                }
            } else {
                Err(VmError::Runtime(
                    "OptionAndThen called on non-Option value".to_string()
                ))
            }
        }
        _ => Err(VmError::Runtime(
            "OptionAndThen called on non-Option value".to_string()
        )),
    }
}

/// Helper function to check if a value is a Result type
pub fn is_result(value: &Value) -> bool {
    match value {
        Value::List(items) if !items.is_empty() => {
            if let Value::Symbol(tag) = &items[0] {
                tag == "Ok" || tag == "Error"
            } else {
                false
            }
        }
        _ => false,
    }
}

/// Helper function to check if a value is an Option type  
pub fn is_option(value: &Value) -> bool {
    match value {
        Value::List(items) if !items.is_empty() => {
            if let Value::Symbol(tag) = &items[0] {
                tag == "Some" || tag == "None"
            } else {
                false
            }
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_result_constructors() {
        let ok_result = ok_constructor(&[Value::Integer(42)]).unwrap();
        assert!(is_result(&ok_result));
        
        let error_result = error_constructor(&[Value::String("error".to_string())]).unwrap();
        assert!(is_result(&error_result));
    }

    #[test]
    fn test_option_constructors() {
        let some_option = some_constructor(&[Value::Integer(42)]).unwrap();
        assert!(is_option(&some_option));
        
        let none_option = none_constructor(&[]).unwrap();
        assert!(is_option(&none_option));
    }

    #[test]
    fn test_result_is_ok() {
        let ok_result = ok_constructor(&[Value::Integer(42)]).unwrap();
        let is_ok = result_is_ok(&[ok_result]).unwrap();
        assert_eq!(is_ok, Value::Boolean(true));
        
        let error_result = error_constructor(&[Value::String("error".to_string())]).unwrap();
        let is_ok = result_is_ok(&[error_result]).unwrap();
        assert_eq!(is_ok, Value::Boolean(false));
    }

    #[test]
    fn test_result_unwrap() {
        let ok_result = ok_constructor(&[Value::Integer(42)]).unwrap();
        let unwrapped = result_unwrap(&[ok_result]).unwrap();
        assert_eq!(unwrapped, Value::Integer(42));
        
        let error_result = error_constructor(&[Value::String("error".to_string())]).unwrap();
        let result = result_unwrap(&[error_result]);
        assert!(result.is_err());
    }

    #[test]
    fn test_result_unwrap_or() {
        let ok_result = ok_constructor(&[Value::Integer(42)]).unwrap();
        let unwrapped = result_unwrap_or(&[ok_result, Value::Integer(0)]).unwrap();
        assert_eq!(unwrapped, Value::Integer(42));
        
        let error_result = error_constructor(&[Value::String("error".to_string())]).unwrap();
        let unwrapped = result_unwrap_or(&[error_result, Value::Integer(0)]).unwrap();
        assert_eq!(unwrapped, Value::Integer(0));
    }

    #[test]
    fn test_option_is_some() {
        let some_option = some_constructor(&[Value::Integer(42)]).unwrap();
        let is_some = option_is_some(&[some_option]).unwrap();
        assert_eq!(is_some, Value::Boolean(true));
        
        let none_option = none_constructor(&[]).unwrap();
        let is_some = option_is_some(&[none_option]).unwrap();
        assert_eq!(is_some, Value::Boolean(false));
    }

    #[test]
    fn test_option_unwrap() {
        let some_option = some_constructor(&[Value::Integer(42)]).unwrap();
        let unwrapped = option_unwrap(&[some_option]).unwrap();
        assert_eq!(unwrapped, Value::Integer(42));
        
        let none_option = none_constructor(&[]).unwrap();
        let result = option_unwrap(&[none_option]);
        assert!(result.is_err());
    }

    #[test]
    fn test_option_unwrap_or() {
        let some_option = some_constructor(&[Value::Integer(42)]).unwrap();
        let unwrapped = option_unwrap_or(&[some_option, Value::Integer(0)]).unwrap();
        assert_eq!(unwrapped, Value::Integer(42));
        
        let none_option = none_constructor(&[]).unwrap();
        let unwrapped = option_unwrap_or(&[none_option, Value::Integer(0)]).unwrap();
        assert_eq!(unwrapped, Value::Integer(0));
    }
}