//! Pattern matching and rules for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};
use crate::pattern_matcher::{PatternMatcher, MatchResult};
use crate::ast::Pattern;

/// Test whether an expression matches a pattern
/// Usage: MatchQ[expr, pattern] returns True if expr matches pattern, False otherwise
/// 
/// Examples:
/// - MatchQ[2, _] → True (any expression matches blank pattern)
/// - MatchQ[2, _Integer] → True (integer matches integer pattern)
/// - MatchQ[2, _Real] → False (integer doesn't match real pattern)
/// - MatchQ["hello", _String] → True (string matches string pattern)
pub fn match_q(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let expr = &args[0];
    let pattern_value = &args[1];

    // Extract pattern from Value::Pattern
    let pattern = match pattern_value {
        Value::Pattern(p) => p,
        _ => {
            return Err(VmError::TypeError {
                expected: "pattern as second argument".to_string(),
                actual: format!("got {:?}", pattern_value),
            });
        }
    };

    // Convert Value to Expr for pattern matching
    let expr_ast = value_to_expr(expr)?;

    // Perform pattern matching
    let mut matcher = PatternMatcher::new();
    let result = matcher.match_pattern(&expr_ast, pattern);

    // Return boolean result
    match result {
        MatchResult::Success { .. } => Ok(Value::Boolean(true)),
        MatchResult::Failure { .. } => Ok(Value::Boolean(false)),
    }
}

/// Convert a VM Value to an AST Expr for pattern matching
/// This is a helper function to bridge the VM and AST representations
fn value_to_expr(value: &Value) -> VmResult<crate::ast::Expr> {
    match value {
        Value::Integer(n) => Ok(crate::ast::Expr::Number(crate::ast::Number::Integer(*n))),
        Value::Real(f) => Ok(crate::ast::Expr::Number(crate::ast::Number::Real(*f))),
        Value::String(s) => Ok(crate::ast::Expr::String(s.clone())),
        Value::Symbol(s) => Ok(crate::ast::Expr::Symbol(crate::ast::Symbol { name: s.clone() })),
        Value::List(items) => {
            let mut expr_items = Vec::new();
            for item in items {
                expr_items.push(value_to_expr(item)?);
            }
            Ok(crate::ast::Expr::List(expr_items))
        }
        _ => Err(VmError::TypeError {
            expected: "basic value types for pattern matching".to_string(),
            actual: format!("unsupported value type: {:?}", value),
        }),
    }
}

/// Extract elements from a list that match a pattern
/// Usage: Cases[list, pattern] returns all elements that match the pattern
/// 
/// Examples:
/// - Cases[{1, 2, "a", 3}, _Integer] → {1, 2, 3}
/// - Cases[{1, 2.5, 3}, _Real] → {2.5}
pub fn cases(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let list_value = &args[0];
    let pattern_value = &args[1];

    // Extract list from first argument
    let list = match list_value {
        Value::List(items) => items,
        _ => {
            return Err(VmError::TypeError {
                expected: "list as first argument".to_string(),
                actual: format!("got {:?}", list_value),
            });
        }
    };

    // Extract pattern from second argument
    let pattern = match pattern_value {
        Value::Pattern(p) => p,
        _ => {
            return Err(VmError::TypeError {
                expected: "pattern as second argument".to_string(),
                actual: format!("got {:?}", pattern_value),
            });
        }
    };

    // Find matching elements
    let mut matches = Vec::new();
    let mut matcher = PatternMatcher::new();

    for item in list {
        // Convert item to Expr for pattern matching
        let item_expr = value_to_expr(item)?;
        
        // Test if item matches pattern
        let result = matcher.match_pattern(&item_expr, pattern);
        
        if let MatchResult::Success { .. } = result {
            matches.push(item.clone());
        }
    }

    Ok(Value::List(matches))
}

/// Count elements in a list that match a pattern  
/// Usage: Count[list, pattern] returns the number of elements that match the pattern
/// 
/// Examples:
/// - Count[{1, 2, "a", 3, "b"}, _String] → 2
/// - Count[{1, 2, 3}, _Integer] → 3
pub fn count_pattern(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let list_value = &args[0];
    let pattern_value = &args[1];

    // Extract list from first argument
    let list = match list_value {
        Value::List(items) => items,
        _ => {
            return Err(VmError::TypeError {
                expected: "list as first argument".to_string(),
                actual: format!("got {:?}", list_value),
            });
        }
    };

    // Extract pattern from second argument
    let pattern = match pattern_value {
        Value::Pattern(p) => p,
        _ => {
            return Err(VmError::TypeError {
                expected: "pattern as second argument".to_string(),
                actual: format!("got {:?}", pattern_value),
            });
        }
    };

    // Count matching elements
    let mut count = 0;
    let mut matcher = PatternMatcher::new();

    for item in list {
        // Convert item to Expr for pattern matching
        let item_expr = value_to_expr(item)?;
        
        // Test if item matches pattern
        let result = matcher.match_pattern(&item_expr, pattern);
        
        if let MatchResult::Success { .. } = result {
            count += 1;
        }
    }

    Ok(Value::Integer(count))
}

/// Find positions of elements that match a pattern or value
/// Usage: Position[list, pattern] returns list of positions where matches occur
/// 
/// Examples:
/// - Position[{a, b, a, c}, a] → {{1}, {3}} (1-indexed positions)
/// - Position[{1, "a", 2}, _String] → {{2}}
pub fn position(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let list_value = &args[0];
    let search_value = &args[1];

    // Extract list from first argument
    let list = match list_value {
        Value::List(items) => items,
        _ => {
            return Err(VmError::TypeError {
                expected: "list as first argument".to_string(),
                actual: format!("got {:?}", list_value),
            });
        }
    };

    let mut positions = Vec::new();

    // Handle pattern matching if second argument is a pattern
    if let Value::Pattern(pattern) = search_value {
        let mut matcher = PatternMatcher::new();
        
        for (index, item) in list.iter().enumerate() {
            // Convert item to Expr for pattern matching
            let item_expr = value_to_expr(item)?;
            
            // Test if item matches pattern
            let result = matcher.match_pattern(&item_expr, pattern);
            
            if let MatchResult::Success { .. } = result {
                // Add 1-indexed position as a list (Mathematica style)
                positions.push(Value::List(vec![Value::Integer((index + 1) as i64)]));
            }
        }
    } else {
        // Handle exact value matching
        for (index, item) in list.iter().enumerate() {
            if item == search_value {
                // Add 1-indexed position as a list (Mathematica style)
                positions.push(Value::List(vec![Value::Integer((index + 1) as i64)]));
            }
        }
    }

    Ok(Value::List(positions))
}

/// Create a replacement rule
/// Usage: Rule[x_, x^2] creates a rule that replaces any pattern x_ with x^2
/// Note: This is a placeholder implementation - full pattern matching requires more VM integration
pub fn rule(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // For now, we just return a special "Rule" function value
    // Full implementation would require pattern compilation and matching
    Ok(Value::Function(format!(
        "Rule[{:?}, {:?}]",
        args[0], args[1]
    )))
}

/// Create a delayed replacement rule
/// Usage: RuleDelayed[x_, RandomReal[]] creates a rule that evaluates RandomReal[] each time
/// Note: This is a placeholder implementation
pub fn rule_delayed(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // For now, we just return a special "RuleDelayed" function value
    Ok(Value::Function(format!(
        "RuleDelayed[{:?}, {:?}]",
        args[0], args[1]
    )))
}

/// Apply replacement rules to an expression
/// Usage: ReplaceAll[expr, rules] or expr /. rules
/// Note: This is a placeholder implementation - full pattern matching requires VM integration
pub fn replace_all(_args: &[Value]) -> VmResult<Value> {
    // TODO: Implement full pattern matching and replacement
    // This requires:
    // 1. Pattern compilation from AST patterns to matchable forms
    // 2. Pattern matching engine that can bind variables
    // 3. Expression evaluation with bindings
    // 4. Integration with the VM for function calls

    Err(VmError::TypeError {
        expected: "ReplaceAll not yet implemented".to_string(),
        actual: "pattern matching engine required".to_string(),
    })
}

/// Apply replacement rules repeatedly to an expression
/// Usage: ReplaceRepeated[expr, rules] or expr //. rules
/// Note: This is a placeholder implementation
pub fn replace_repeated(_args: &[Value]) -> VmResult<Value> {
    // TODO: Implement repeated rule application
    // This applies rules repeatedly until no more changes occur
    
    Err(VmError::TypeError {
        expected: "ReplaceRepeated not yet implemented".to_string(),
        actual: "pattern matching engine required".to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_rule_basic() {
        let lhs = Value::Symbol("x_".to_string());
        let rhs = Value::Symbol("x^2".to_string());
        let result = rule(&[lhs, rhs]).unwrap();

        // Should return a function representing the rule
        match result {
            Value::Function(rule_str) => {
                assert!(rule_str.contains("Rule"));
                assert!(rule_str.contains("x_"));
                assert!(rule_str.contains("x^2"));
            }
            _ => panic!("Expected Function value, got {:?}", result),
        }
    }

    #[test]
    fn test_rule_wrong_args() {
        assert!(rule(&[]).is_err());
        assert!(rule(&[Value::Integer(1)]).is_err());
        assert!(rule(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)]).is_err());
    }

    #[test]
    fn test_rule_delayed_basic() {
        let lhs = Value::Symbol("x_".to_string());
        let rhs = Value::Symbol("RandomReal[]".to_string());
        let result = rule_delayed(&[lhs, rhs]).unwrap();

        // Should return a function representing the delayed rule
        match result {
            Value::Function(rule_str) => {
                assert!(rule_str.contains("RuleDelayed"));
                assert!(rule_str.contains("x_"));
                assert!(rule_str.contains("RandomReal"));
            }
            _ => panic!("Expected Function value, got {:?}", result),
        }
    }

    #[test]
    fn test_rule_delayed_wrong_args() {
        assert!(rule_delayed(&[]).is_err());
        assert!(rule_delayed(&[Value::Integer(1)]).is_err());
        assert!(rule_delayed(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)]).is_err());
    }

    #[test]
    fn test_replace_all_not_implemented() {
        // ReplaceAll requires full pattern matching, so should return error for now
        let expr = Value::Symbol("x".to_string());
        let rule = Value::Function("x -> x^2".to_string());
        assert!(replace_all(&[expr, rule]).is_err());
    }

    #[test]
    fn test_rule_with_different_types() {
        // Rules should work with any value types
        let lhs = Value::Integer(1);
        let rhs = Value::String("one".to_string());
        let result = rule(&[lhs, rhs]).unwrap();

        match result {
            Value::Function(rule_str) => {
                assert!(rule_str.contains("Rule"));
            }
            _ => panic!("Expected Function value"),
        }
    }

    #[test]
    fn test_rule_delayed_with_lists() {
        // Test with more complex data types
        let lhs = Value::List(vec![Value::Symbol("x_".to_string())]);
        let rhs = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let result = rule_delayed(&[lhs, rhs]).unwrap();

        match result {
            Value::Function(rule_str) => {
                assert!(rule_str.contains("RuleDelayed"));
            }
            _ => panic!("Expected Function value"),
        }
    }

    #[test]
    fn test_match_q_blank_pattern() {
        use crate::ast::Pattern;
        
        // Test _ pattern matches any value
        let expr = Value::Integer(42);
        let pattern = Value::Pattern(Pattern::Blank { head: None });
        let result = match_q(&[expr, pattern]).unwrap();
        
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_match_q_typed_pattern() {
        use crate::ast::Pattern;
        
        // Test _Integer pattern
        let expr = Value::Integer(42);
        let pattern = Value::Pattern(Pattern::Blank { head: Some("Integer".to_string()) });
        let result = match_q(&[expr.clone(), pattern]).unwrap();
        
        assert_eq!(result, Value::Boolean(true));
        
        // Test _Real pattern with integer (should fail)
        let pattern = Value::Pattern(Pattern::Blank { head: Some("Real".to_string()) });
        let result = match_q(&[expr, pattern]).unwrap();
        
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_match_q_string_pattern() {
        use crate::ast::Pattern;
        
        // Test _String pattern
        let expr = Value::String("hello".to_string());
        let pattern = Value::Pattern(Pattern::Blank { head: Some("String".to_string()) });
        let result = match_q(&[expr, pattern]).unwrap();
        
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_match_q_wrong_args() {
        // Test wrong number of arguments
        assert!(match_q(&[]).is_err());
        assert!(match_q(&[Value::Integer(1)]).is_err());
        assert!(match_q(&[Value::Integer(1), Value::Integer(2), Value::Integer(3)]).is_err());
    }

    #[test]
    fn test_match_q_non_pattern_arg() {
        // Test second argument is not a pattern
        let expr = Value::Integer(42);
        let non_pattern = Value::String("not a pattern".to_string());
        let result = match_q(&[expr, non_pattern]);
        
        assert!(result.is_err());
    }
}
