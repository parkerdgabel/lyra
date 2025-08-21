//! List operations for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};

/// Get the length of a list
/// Usage: Length[{1, 2, 3}] -> 3
pub fn length(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => Ok(Value::Integer(list.len() as i64)),
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get the first element of a list
/// Usage: Head[{1, 2, 3}] -> 1
pub fn head(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => {
            if list.is_empty() {
                Err(VmError::TypeError {
                    expected: "non-empty list".to_string(),
                    actual: "empty list".to_string(),
                })
            } else {
                Ok(list[0].clone())
            }
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get all elements except the first (tail of a list)
/// Usage: Tail[{1, 2, 3}] -> {2, 3}
pub fn tail(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => {
            if list.is_empty() {
                Err(VmError::TypeError {
                    expected: "non-empty list".to_string(),
                    actual: "empty list".to_string(),
                })
            } else {
                Ok(Value::List(list[1..].to_vec()))
            }
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Append an element to a list
/// Usage: Append[{1, 2}, 3] -> {1, 2, 3}
pub fn append(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => {
            let mut new_list = list.clone();
            new_list.push(args[1].clone());
            Ok(Value::List(new_list))
        }
        _ => Err(VmError::TypeError {
            expected: "List as first argument".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Flatten nested lists one level deep
/// Usage: Flatten[{{1, 2}, {3, 4}}] -> {1, 2, 3, 4}
pub fn flatten(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(list) => {
            let mut flattened = Vec::new();
            for item in list {
                match item {
                    Value::List(inner_list) => {
                        flattened.extend(inner_list.clone());
                    }
                    other => {
                        flattened.push(other.clone());
                    }
                }
            }
            Ok(Value::List(flattened))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Apply a function to each element of a list
/// Usage: Map[f, {1, 2, 3}] -> {f[1], f[2], f[3]}
/// Note: This is a simplified version - full implementation requires function evaluation
pub fn map(_args: &[Value]) -> VmResult<Value> {
    // TODO: Implement once we have function evaluation in the VM
    Err(VmError::TypeError {
        expected: "Map not yet implemented".to_string(),
        actual: "function evaluation required".to_string(),
    })
}

/// Apply a function to a list of arguments
/// Usage: Apply[Plus, {1, 2, 3}] -> Plus[1, 2, 3]
/// Note: This is a simplified version - full implementation requires function evaluation
pub fn apply(_args: &[Value]) -> VmResult<Value> {
    // TODO: Implement once we have function evaluation in the VM
    Err(VmError::TypeError {
        expected: "Apply not yet implemented".to_string(),
        actual: "function evaluation required".to_string(),
    })
}

/// Sum all elements in a list
/// Usage: Total[{1, 2, 3, 4}] -> 10
pub fn total(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::List(items) => {
            if items.is_empty() {
                return Ok(Value::Integer(0));
            }

            // Determine if we're working with integers or reals
            let mut sum_int: i64 = 0;
            let mut sum_real: f64 = 0.0;
            let mut has_real = false;

            for item in items {
                match item {
                    Value::Integer(n) => {
                        if has_real {
                            sum_real += *n as f64;
                        } else {
                            sum_int += n;
                        }
                    }
                    Value::Real(r) => {
                        if !has_real {
                            // Convert accumulated integer sum to real
                            sum_real = sum_int as f64 + r;
                            has_real = true;
                        } else {
                            sum_real += r;
                        }
                    }
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "List of numbers".to_string(),
                            actual: format!("List containing {:?}", item),
                        });
                    }
                }
            }

            if has_real {
                Ok(Value::Real(sum_real))
            } else {
                Ok(Value::Integer(sum_int))
            }
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_length_basic() {
        let list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let result = length(&[list]).unwrap();
        assert_eq!(result, Value::Integer(3));
    }

    #[test]
    fn test_length_empty_list() {
        let list = Value::List(vec![]);
        let result = length(&[list]).unwrap();
        assert_eq!(result, Value::Integer(0));
    }

    #[test]
    fn test_length_wrong_args() {
        assert!(length(&[]).is_err());
        assert!(length(&[Value::Integer(1), Value::Integer(2)]).is_err());
    }

    #[test]
    fn test_length_wrong_type() {
        assert!(length(&[Value::Integer(42)]).is_err());
    }

    #[test]
    fn test_head_basic() {
        let list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let result = head(&[list]).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_head_empty_list() {
        let list = Value::List(vec![]);
        assert!(head(&[list]).is_err());
    }

    #[test]
    fn test_head_wrong_args() {
        assert!(head(&[]).is_err());
        assert!(head(&[Value::Integer(1), Value::Integer(2)]).is_err());
    }

    #[test]
    fn test_tail_basic() {
        let list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let result = tail(&[list]).unwrap();
        assert_eq!(
            result,
            Value::List(vec![Value::Integer(2), Value::Integer(3)])
        );
    }

    #[test]
    fn test_tail_single_element() {
        let list = Value::List(vec![Value::Integer(42)]);
        let result = tail(&[list]).unwrap();
        assert_eq!(result, Value::List(vec![]));
    }

    #[test]
    fn test_tail_empty_list() {
        let list = Value::List(vec![]);
        assert!(tail(&[list]).is_err());
    }

    #[test]
    fn test_append_basic() {
        let list = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let elem = Value::Integer(3);
        let result = append(&[list, elem]).unwrap();
        assert_eq!(
            result,
            Value::List(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3),
            ])
        );
    }

    #[test]
    fn test_append_to_empty() {
        let list = Value::List(vec![]);
        let elem = Value::Integer(42);
        let result = append(&[list, elem]).unwrap();
        assert_eq!(result, Value::List(vec![Value::Integer(42)]));
    }

    #[test]
    fn test_append_wrong_args() {
        assert!(append(&[]).is_err());
        assert!(append(&[Value::Integer(1)]).is_err());
        assert!(append(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)]).is_err());
    }

    #[test]
    fn test_flatten_basic() {
        let list = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ]);
        let result = flatten(&[list]).unwrap();
        assert_eq!(
            result,
            Value::List(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(4),
            ])
        );
    }

    #[test]
    fn test_flatten_mixed() {
        let list = Value::List(vec![
            Value::Integer(1),
            Value::List(vec![Value::Integer(2), Value::Integer(3)]),
            Value::Integer(4),
        ]);
        let result = flatten(&[list]).unwrap();
        assert_eq!(
            result,
            Value::List(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(4),
            ])
        );
    }

    #[test]
    fn test_flatten_empty() {
        let list = Value::List(vec![]);
        let result = flatten(&[list]).unwrap();
        assert_eq!(result, Value::List(vec![]));
    }

    #[test]
    fn test_map_not_implemented() {
        // Map requires function evaluation, so should return error for now
        let list = Value::List(vec![Value::Integer(1)]);
        let func = Value::Function("f".to_string());
        assert!(map(&[func, list]).is_err());
    }

    #[test]
    fn test_apply_not_implemented() {
        // Apply requires function evaluation, so should return error for now
        let func = Value::Function("Plus".to_string());
        let list = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        assert!(apply(&[func, list]).is_err());
    }
}
