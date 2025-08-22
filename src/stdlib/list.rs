//! List operations for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};

/// Helper function to substitute slots in pure functions
/// For now, this is a simplified implementation that handles basic slot substitution
fn substitute_slots(body: &Value, args: &[Value]) -> VmResult<Value> {
    match body {
        Value::Slot { number } => {
            let slot_index = number.unwrap_or(1) - 1; // #1 is index 0, #2 is index 1, etc.
            if slot_index < args.len() {
                Ok(args[slot_index].clone())
            } else {
                Err(VmError::Runtime(format!("Slot #{} not provided", slot_index + 1)))
            }
        }
        Value::List(items) => {
            let mut result = Vec::with_capacity(items.len());
            for item in items {
                result.push(substitute_slots(item, args)?);
            }
            Ok(Value::List(result))
        }
        // For other value types, return as-is (no slots to substitute)
        _ => Ok(body.clone()),
    }
}

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
                // Pre-allocate with exact capacity to avoid reallocations
                let mut tail_vec = Vec::with_capacity(list.len() - 1);
                tail_vec.extend_from_slice(&list[1..]);
                Ok(Value::List(tail_vec))
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
            // Pre-allocate with exact capacity to avoid reallocations
            let mut new_list = Vec::with_capacity(list.len() + 1);
            new_list.extend_from_slice(list);
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
            // Early return for empty lists
            if list.is_empty() {
                return Ok(Value::List(Vec::new()));
            }
            
            // Estimate capacity by scanning for inner lists
            let estimated_capacity = list.iter().map(|item| {
                match item {
                    Value::List(inner) => inner.len(),
                    _ => 1,
                }
            }).sum();
            
            let mut flattened = Vec::with_capacity(estimated_capacity);
            for item in list {
                match item {
                    Value::List(inner_list) => {
                        flattened.extend_from_slice(inner_list);
                    }
                    other => {
                        flattened.push(other.clone());
                    }
                }
            }
            
            // Shrink to exact size to save memory for large flattened lists
            flattened.shrink_to_fit();
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
pub fn map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let function = &args[0];
    let list = match &args[1] {
        Value::List(list) => list,
        _ => return Err(VmError::TypeError {
            expected: "List as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match function {
        Value::Function(func_name) => {
            // Handle built-in functions
            let mut results = Vec::with_capacity(list.len());
            
            for item in list {
                // Create a synthetic call by applying the function to each element
                let result = match func_name.as_str() {
                    "Plus" => {
                        // Special case: Plus with one argument should add to itself (2x)
                        match item {
                            Value::Integer(n) => Value::Integer(n + n),
                            Value::Real(r) => Value::Real(r + r),
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Times" => {
                        // Special case: Times with one argument should square it
                        match item {
                            Value::Integer(n) => Value::Integer(n * n),
                            Value::Real(r) => Value::Real(r * r),
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Sin" => {
                        match item {
                            Value::Integer(n) => Value::Real((*n as f64).sin()),
                            Value::Real(r) => Value::Real(r.sin()),
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Cos" => {
                        match item {
                            Value::Integer(n) => Value::Real((*n as f64).cos()),
                            Value::Real(r) => Value::Real(r.cos()),
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Sqrt" => {
                        match item {
                            Value::Integer(n) => {
                                if *n < 0 {
                                    return Err(VmError::Runtime("Square root of negative number".to_string()));
                                }
                                Value::Real((*n as f64).sqrt())
                            }
                            Value::Real(r) => {
                                if *r < 0.0 {
                                    return Err(VmError::Runtime("Square root of negative number".to_string()));
                                }
                                Value::Real(r.sqrt())
                            }
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Abs" => {
                        match item {
                            Value::Integer(n) => Value::Integer(n.abs()),
                            Value::Real(r) => Value::Real(r.abs()),
                            _ => return Err(VmError::TypeError {
                                expected: "Number".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Length" => {
                        // Apply Length function to each element
                        match item {
                            Value::List(inner_list) => Value::Integer(inner_list.len() as i64),
                            Value::String(s) => Value::Integer(s.len() as i64),
                            _ => return Err(VmError::TypeError {
                                expected: "List or String".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    "Head" => {
                        // Apply Head function to each element
                        match item {
                            Value::List(inner_list) => {
                                if inner_list.is_empty() {
                                    return Err(VmError::TypeError {
                                        expected: "non-empty list".to_string(),
                                        actual: "empty list".to_string(),
                                    });
                                }
                                inner_list[0].clone()
                            }
                            _ => return Err(VmError::TypeError {
                                expected: "List".to_string(),
                                actual: format!("{:?}", item),
                            }),
                        }
                    }
                    _ => {
                        return Err(VmError::Runtime(format!("Map: unsupported function '{}'", func_name)));
                    }
                };
                results.push(result);
            }
            
            Ok(Value::List(results))
        }
        Value::PureFunction { body } => {
            // Handle pure functions (anonymous functions with slots)
            let mut results = Vec::with_capacity(list.len());
            
            for item in list {
                // For pure functions, substitute the item for slot #1
                let result = substitute_slots(body, &[item.clone()])?;
                results.push(result);
            }
            
            Ok(Value::List(results))
        }
        _ => Err(VmError::TypeError {
            expected: "Function as first argument".to_string(),
            actual: format!("{:?}", function),
        }),
    }
}

/// Apply a function to a list of arguments
/// Usage: Apply[Plus, {1, 2, 3}] -> Plus[1, 2, 3]
pub fn apply(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let function = &args[0];
    let arg_list = match &args[1] {
        Value::List(list) => list,
        _ => return Err(VmError::TypeError {
            expected: "List as second argument".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match function {
        Value::Function(func_name) => {
            // Handle built-in functions by applying them to the entire argument list
            match func_name.as_str() {
                "Plus" => {
                    // Sum all arguments
                    let has_real = arg_list.iter().any(|item| matches!(item, Value::Real(_)));
                    
                    if has_real {
                        let mut sum = 0.0;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => sum += *n as f64,
                                Value::Real(r) => sum += r,
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Real(sum))
                    } else {
                        let mut sum = 0i64;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => {
                                    sum = sum.checked_add(*n).ok_or_else(|| VmError::Runtime("Integer overflow in sum".to_string()))?;
                                }
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Integer(sum))
                    }
                }
                "Times" => {
                    // Multiply all arguments
                    let has_real = arg_list.iter().any(|item| matches!(item, Value::Real(_)));
                    
                    if has_real {
                        let mut product = 1.0;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => product *= *n as f64,
                                Value::Real(r) => product *= r,
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Real(product))
                    } else {
                        let mut product = 1i64;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => {
                                    product = product.checked_mul(*n).ok_or_else(|| VmError::Runtime("Integer overflow in product".to_string()))?;
                                }
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Integer(product))
                    }
                }
                "Max" => {
                    if arg_list.is_empty() {
                        return Err(VmError::Runtime("Max requires at least one argument".to_string()));
                    }
                    
                    let has_real = arg_list.iter().any(|item| matches!(item, Value::Real(_)));
                    
                    if has_real {
                        let mut max_val = f64::NEG_INFINITY;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => max_val = max_val.max(*n as f64),
                                Value::Real(r) => max_val = max_val.max(*r),
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Real(max_val))
                    } else {
                        let mut max_val = i64::MIN;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => max_val = max_val.max(*n),
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Integer(max_val))
                    }
                }
                "Min" => {
                    if arg_list.is_empty() {
                        return Err(VmError::Runtime("Min requires at least one argument".to_string()));
                    }
                    
                    let has_real = arg_list.iter().any(|item| matches!(item, Value::Real(_)));
                    
                    if has_real {
                        let mut min_val = f64::INFINITY;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => min_val = min_val.min(*n as f64),
                                Value::Real(r) => min_val = min_val.min(*r),
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Real(min_val))
                    } else {
                        let mut min_val = i64::MAX;
                        for item in arg_list {
                            match item {
                                Value::Integer(n) => min_val = min_val.min(*n),
                                _ => return Err(VmError::TypeError {
                                    expected: "List of numbers".to_string(),
                                    actual: format!("List containing {:?}", item),
                                }),
                            }
                        }
                        Ok(Value::Integer(min_val))
                    }
                }
                "StringJoin" => {
                    // Join all arguments as strings
                    let mut result = String::new();
                    for item in arg_list {
                        match item {
                            Value::String(s) => result.push_str(s),
                            Value::Symbol(s) => result.push_str(s),
                            Value::Integer(n) => result.push_str(&n.to_string()),
                            Value::Real(r) => result.push_str(&r.to_string()),
                            _ => return Err(VmError::TypeError {
                                expected: "List of strings or numbers".to_string(),
                                actual: format!("List containing {:?}", item),
                            }),
                        }
                    }
                    Ok(Value::String(result))
                }
                "List" => {
                    // List constructor - just return the arguments as a list
                    Ok(Value::List(arg_list.clone()))
                }
                _ => {
                    Err(VmError::Runtime(format!("Apply: unsupported function '{}'", func_name)))
                }
            }
        }
        Value::PureFunction { body } => {
            // Handle pure functions by substituting all arguments
            substitute_slots(body, arg_list)
        }
        _ => Err(VmError::TypeError {
            expected: "Function as first argument".to_string(),
            actual: format!("{:?}", function),
        }),
    }
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
            // Early return for empty lists
            if items.is_empty() {
                return Ok(Value::Integer(0));
            }

            // Fast scan to determine type (single pass optimization)
            let has_real = items.iter().any(|item| matches!(item, Value::Real(_)));
            
            if has_real {
                // Use floating point arithmetic throughout
                let mut sum = 0.0;
                for item in items {
                    match item {
                        Value::Integer(n) => sum += *n as f64,
                        Value::Real(r) => sum += r,
                        _ => {
                            return Err(VmError::TypeError {
                                expected: "List of numbers".to_string(),
                                actual: format!("List containing {:?}", item),
                            });
                        }
                    }
                }
                Ok(Value::Real(sum))
            } else {
                // Use integer arithmetic for better precision
                let mut sum = 0i64;
                for item in items {
                    match item {
                        Value::Integer(n) => {
                            sum = sum.checked_add(*n).ok_or_else(|| VmError::Runtime("Integer overflow in sum".to_string()))?;
                        }
                        _ => {
                            return Err(VmError::TypeError {
                                expected: "List of numbers".to_string(),
                                actual: format!("List containing {:?}", item),
                            });
                        }
                    }
                }
                Ok(Value::Integer(sum))
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
    fn test_map_function() {
        // Test Map with built-in function
        let list = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let func = Value::Function("Times".to_string()); // Should square each element
        let result = map(&[func, list]).unwrap();
        
        assert_eq!(
            result,
            Value::List(vec![Value::Integer(1), Value::Integer(4), Value::Integer(9)])
        );
    }

    #[test]
    fn test_map_math_function() {
        // Test Map with Sin function
        let list = Value::List(vec![Value::Integer(0), Value::Real(std::f64::consts::PI / 2.0)]);
        let func = Value::Function("Sin".to_string());
        let result = map(&[func, list]).unwrap();
        
        if let Value::List(results) = result {
            assert_eq!(results.len(), 2);
            if let Value::Real(sin_0) = results[0] {
                assert!((sin_0 - 0.0).abs() < 1e-10);
            } else {
                panic!("Expected Real value for sin(0)");
            }
            if let Value::Real(sin_pi_2) = results[1] {
                assert!((sin_pi_2 - 1.0).abs() < 1e-10);
            } else {
                panic!("Expected Real value for sin(π/2)");
            }
        } else {
            panic!("Expected List result from Map");
        }
    }

    #[test]
    fn test_map_empty_list() {
        let list = Value::List(vec![]);
        let func = Value::Function("Plus".to_string());
        let result = map(&[func, list]).unwrap();
        assert_eq!(result, Value::List(vec![]));
    }

    #[test]
    fn test_map_wrong_args() {
        // Test wrong number of arguments
        assert!(map(&[]).is_err());
        assert!(map(&[Value::Function("f".to_string())]).is_err());
        assert!(map(&[Value::Function("f".to_string()), Value::Integer(1), Value::Integer(2)]).is_err());
        
        // Test wrong argument types
        assert!(map(&[Value::Integer(1), Value::List(vec![])]).is_err());
        assert!(map(&[Value::Function("f".to_string()), Value::Integer(1)]).is_err());
    }

    #[test]
    fn test_apply_plus() {
        // Test Apply with Plus function
        let func = Value::Function("Plus".to_string());
        let list = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let result = apply(&[func, list]).unwrap();
        assert_eq!(result, Value::Integer(6));
    }

    #[test]
    fn test_apply_times() {
        // Test Apply with Times function
        let func = Value::Function("Times".to_string());
        let list = Value::List(vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)]);
        let result = apply(&[func, list]).unwrap();
        assert_eq!(result, Value::Integer(24));
    }

    #[test]
    fn test_apply_max() {
        // Test Apply with Max function
        let func = Value::Function("Max".to_string());
        let list = Value::List(vec![Value::Integer(1), Value::Integer(5), Value::Integer(3)]);
        let result = apply(&[func, list]).unwrap();
        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_apply_min() {
        // Test Apply with Min function
        let func = Value::Function("Min".to_string());
        let list = Value::List(vec![Value::Integer(1), Value::Integer(5), Value::Integer(3)]);
        let result = apply(&[func, list]).unwrap();
        assert_eq!(result, Value::Integer(1));
    }

    #[test]
    fn test_apply_string_join() {
        // Test Apply with StringJoin function
        let func = Value::Function("StringJoin".to_string());
        let list = Value::List(vec![
            Value::String("Hello".to_string()),
            Value::String(" ".to_string()),
            Value::String("World".to_string()),
        ]);
        let result = apply(&[func, list]).unwrap();
        assert_eq!(result, Value::String("Hello World".to_string()));
    }

    #[test]
    fn test_apply_wrong_args() {
        // Test wrong number of arguments
        assert!(apply(&[]).is_err());
        assert!(apply(&[Value::Function("Plus".to_string())]).is_err());
        
        // Test wrong argument types
        assert!(apply(&[Value::Integer(1), Value::List(vec![])]).is_err());
        assert!(apply(&[Value::Function("Plus".to_string()), Value::Integer(1)]).is_err());
    }

    #[test]
    fn test_apply_unsupported_function() {
        let func = Value::Function("UnsupportedFunction".to_string());
        let list = Value::List(vec![Value::Integer(1)]);
        assert!(apply(&[func, list]).is_err());
    }
}
