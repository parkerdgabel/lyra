//! String operations for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};

/// Join multiple strings together
/// Usage: StringJoin["Hello", " ", "World"] -> "Hello World"
pub fn string_join(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Ok(Value::String(String::new()));
    }

    let mut result = String::new();
    for arg in args {
        match arg {
            Value::String(s) => result.push_str(s),
            _ => {
                return Err(VmError::TypeError {
                    expected: "String".to_string(),
                    actual: format!("{:?}", arg),
                })
            }
        }
    }

    Ok(Value::String(result))
}

/// Get the length of a string
/// Usage: StringLength["Hello"] -> 5
pub fn string_length(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::String(s) => Ok(Value::Integer(s.chars().count() as i64)),
        _ => Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Take the first n characters from a string
/// Usage: StringTake["Hello", 3] -> "Hel"
pub fn string_take(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let string = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(VmError::TypeError {
                expected: "String as first argument".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
    };

    let n = match &args[1] {
        Value::Integer(n) => *n,
        _ => {
            return Err(VmError::TypeError {
                expected: "Integer as second argument".to_string(),
                actual: format!("{:?}", args[1]),
            })
        }
    };

    if n < 0 {
        return Err(VmError::TypeError {
            expected: "non-negative integer".to_string(),
            actual: format!("negative integer: {}", n),
        });
    }

    let chars: Vec<char> = string.chars().collect();
    let take_count = std::cmp::min(n as usize, chars.len());
    let result: String = chars.iter().take(take_count).collect();

    Ok(Value::String(result))
}

/// Drop the first n characters from a string
/// Usage: StringDrop["Hello", 2] -> "llo"
pub fn string_drop(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let string = match &args[0] {
        Value::String(s) => s,
        _ => {
            return Err(VmError::TypeError {
                expected: "String as first argument".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
    };

    let n = match &args[1] {
        Value::Integer(n) => *n,
        _ => {
            return Err(VmError::TypeError {
                expected: "Integer as second argument".to_string(),
                actual: format!("{:?}", args[1]),
            })
        }
    };

    if n < 0 {
        return Err(VmError::TypeError {
            expected: "non-negative integer".to_string(),
            actual: format!("negative integer: {}", n),
        });
    }

    let chars: Vec<char> = string.chars().collect();
    let drop_count = std::cmp::min(n as usize, chars.len());
    let result: String = chars.iter().skip(drop_count).collect();

    Ok(Value::String(result))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_string_join_basic() {
        let args = vec![
            Value::String("Hello".to_string()),
            Value::String(" ".to_string()),
            Value::String("World".to_string()),
        ];
        let result = string_join(&args).unwrap();
        assert_eq!(result, Value::String("Hello World".to_string()));
    }

    #[test]
    fn test_string_join_empty() {
        let result = string_join(&[]).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_string_join_single() {
        let args = vec![Value::String("Hello".to_string())];
        let result = string_join(&args).unwrap();
        assert_eq!(result, Value::String("Hello".to_string()));
    }

    #[test]
    fn test_string_join_wrong_type() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(42)];
        assert!(string_join(&args).is_err());
    }

    #[test]
    fn test_string_length_basic() {
        let args = vec![Value::String("Hello".to_string())];
        let result = string_length(&args).unwrap();
        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_string_length_empty() {
        let args = vec![Value::String("".to_string())];
        let result = string_length(&args).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_string_length_unicode() {
        let args = vec![Value::String("🦀💎".to_string())];
        let result = string_length(&args).unwrap();
        assert_eq!(result, Value::Integer(2)); // Two Unicode characters
    }

    #[test]
    fn test_string_length_wrong_args() {
        assert!(string_length(&[]).is_err());
        assert!(
            string_length(&[Value::String("".to_string()), Value::String("".to_string())]).is_err()
        );
    }

    #[test]
    fn test_string_length_wrong_type() {
        let args = vec![Value::Integer(42)];
        assert!(string_length(&args).is_err());
    }

    #[test]
    fn test_string_take_basic() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(3)];
        let result = string_take(&args).unwrap();
        assert_eq!(result, Value::String("Hel".to_string()));
    }

    #[test]
    fn test_string_take_zero() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(0)];
        let result = string_take(&args).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_string_take_more_than_length() {
        let args = vec![Value::String("Hi".to_string()), Value::Integer(10)];
        let result = string_take(&args).unwrap();
        assert_eq!(result, Value::String("Hi".to_string()));
    }

    #[test]
    fn test_string_take_negative() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(-1)];
        assert!(string_take(&args).is_err());
    }

    #[test]
    fn test_string_take_wrong_args() {
        assert!(string_take(&[]).is_err());
        assert!(string_take(&[Value::String("".to_string())]).is_err());
        assert!(string_take(&[
            Value::String("".to_string()),
            Value::Integer(1),
            Value::Integer(2)
        ])
        .is_err());
    }

    #[test]
    fn test_string_take_wrong_types() {
        let args1 = vec![Value::Integer(42), Value::Integer(1)];
        assert!(string_take(&args1).is_err());

        let args2 = vec![
            Value::String("Hello".to_string()),
            Value::String("1".to_string()),
        ];
        assert!(string_take(&args2).is_err());
    }

    #[test]
    fn test_string_drop_basic() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(2)];
        let result = string_drop(&args).unwrap();
        assert_eq!(result, Value::String("llo".to_string()));
    }

    #[test]
    fn test_string_drop_zero() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(0)];
        let result = string_drop(&args).unwrap();
        assert_eq!(result, Value::String("Hello".to_string()));
    }

    #[test]
    fn test_string_drop_all() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(5)];
        let result = string_drop(&args).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_string_drop_more_than_length() {
        let args = vec![Value::String("Hi".to_string()), Value::Integer(10)];
        let result = string_drop(&args).unwrap();
        assert_eq!(result, Value::String("".to_string()));
    }

    #[test]
    fn test_string_drop_negative() {
        let args = vec![Value::String("Hello".to_string()), Value::Integer(-1)];
        assert!(string_drop(&args).is_err());
    }

    #[test]
    fn test_string_drop_unicode() {
        let args = vec![Value::String("🦀💎🔧".to_string()), Value::Integer(1)];
        let result = string_drop(&args).unwrap();
        assert_eq!(result, Value::String("💎🔧".to_string()));
    }
}
