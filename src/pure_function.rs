#![allow(unused_variables)]
/// Pure Function Slot Substitution Implementation
/// 
/// This module implements the core algorithm for substituting slots in pure functions
/// with provided arguments. This follows Wolfram Language semantics for pure functions.

use crate::vm::Value;
use crate::ast::{Expr, Number, Symbol};
use crate::unified_errors::{LyraUnifiedError, LyraResult};

/// Substitute slots in a pure function with the provided arguments
/// 
/// This is the main entry point for pure function evaluation. It takes a pure function
/// Value and a list of arguments, then substitutes all slot placeholders with the
/// corresponding arguments.
/// 
/// # Arguments
/// * `pure_function` - The Value::PureFunction containing the AST body
/// * `args` - The arguments to substitute for slots (#, #1, #2, etc.)
/// 
/// # Returns
/// * `Ok(Value)` - The result after slot substitution and evaluation
/// * `Err(LyraUnifiedError)` - Error if substitution fails
/// 
/// # Wolfram Language Semantics
/// - `#` refers to the first argument (equivalent to `#1`)
/// - `#n` refers to the nth argument (1-indexed)
/// - If both `#` and numbered slots are used, `#` defaults to `#1`
/// - Slot indices must be within the bounds of provided arguments
/// 
/// # Examples
/// ```
/// // # + 1 & applied to [5] -> 6
/// // #1 + #2 & applied to [3, 4] -> 7
/// // # * # & applied to [3] -> 9
/// ```
pub fn substitute_slots(pure_function: &Value, args: &[Value]) -> LyraResult<Value> {
    match pure_function {
        Value::PureFunction { body } => {
            // Extract the AST from the body Value
            let ast_body = value_to_expr(body)?;
            
            // Perform slot substitution on the AST
            let substituted_ast = substitute_slots_in_expr(&ast_body, args)?;
            
            // Convert back to Value and evaluate if possible
            let result = expr_to_value(&substituted_ast)?;
            Ok(result)
        }
        _ => Err(LyraUnifiedError::Runtime {
            message: "Expected PureFunction value".to_string(),
            context: crate::unified_errors::RuntimeContext {
                current_function: Some("substitute_slots".to_string()),
                call_stack_depth: 0,
                local_variables: vec![],
                evaluation_mode: "pure_function_substitution".to_string(),
            },
            recoverable: false,
        })
    }
}

/// Recursively substitute slots in an AST expression
/// 
/// This function traverses the AST tree and replaces any Slot nodes with the
/// corresponding argument values. It handles both numbered and unnumbered slots
/// according to Wolfram Language semantics.
/// 
/// # Arguments
/// * `expr` - The AST expression to process
/// * `args` - The arguments to substitute for slots
/// 
/// # Returns
/// * `Ok(Expr)` - The expression with slots substituted
/// * `Err(LyraUnifiedError)` - Error if slot index is out of bounds or other issues
pub fn substitute_slots_in_expr(expr: &Expr, args: &[Value]) -> LyraResult<Expr> {
    substitute_slots_in_expr_with_depth(expr, args, 0)
}

/// Internal function with recursion depth tracking to prevent stack overflow
fn substitute_slots_in_expr_with_depth(expr: &Expr, args: &[Value], depth: usize) -> LyraResult<Expr> {
    // Handle edge case: prevent stack overflow from deeply nested expressions
    const MAX_RECURSION_DEPTH: usize = 1000;
    if depth > MAX_RECURSION_DEPTH {
        return Err(LyraUnifiedError::Runtime {
            message: format!("Maximum recursion depth ({}) exceeded during slot substitution", MAX_RECURSION_DEPTH),
            context: crate::unified_errors::RuntimeContext {
                current_function: Some("substitute_slots_in_expr_with_depth".to_string()),
                call_stack_depth: depth,
                local_variables: vec![],
                evaluation_mode: "pure_function_substitution".to_string(),
            },
            recoverable: false,
        });
    }
    match expr {
        // Handle slot substitution
        Expr::Slot { number } => {
            // Handle edge case: no arguments provided
            if args.is_empty() {
                return Err(LyraUnifiedError::Runtime {
                    message: "Pure function requires arguments but none were provided".to_string(),
                    context: crate::unified_errors::RuntimeContext {
                        current_function: Some("substitute_slots_in_expr".to_string()),
                        call_stack_depth: 0,
                        local_variables: vec![],
                        evaluation_mode: "pure_function_substitution".to_string(),
                    },
                    recoverable: false,
                });
            }
            
            let slot_index = match number {
                Some(n) => {
                    // Handle edge cases for numbered slots
                    if *n == 0 {
                        return Err(LyraUnifiedError::Runtime {
                            message: "Slot indices must be >= 1, found slot #0".to_string(),
                            context: crate::unified_errors::RuntimeContext {
                                current_function: Some("substitute_slots_in_expr".to_string()),
                                call_stack_depth: 0,
                                local_variables: vec![],
                                evaluation_mode: "pure_function_substitution".to_string(),
                            },
                            recoverable: false,
                        });
                    }
                    
                    // Handle edge case: very large slot numbers (potential DoS protection)
                    if *n > 10000 {
                        return Err(LyraUnifiedError::Runtime {
                            message: format!("Slot index #{} exceeds maximum allowed slot number (10000)", n),
                            context: crate::unified_errors::RuntimeContext {
                                current_function: Some("substitute_slots_in_expr".to_string()),
                                call_stack_depth: 0,
                                local_variables: vec![],
                                evaluation_mode: "pure_function_substitution".to_string(),
                            },
                            recoverable: false,
                        });
                    }
                    
                    *n - 1 // Convert from 1-indexed to 0-indexed
                }
                None => 0, // # defaults to first argument (#1)
            };
            
            // Check bounds with descriptive error messages
            if slot_index >= args.len() {
                let slot_display = match number {
                    Some(n) => format!("#{}", n),
                    None => "#".to_string(),
                };
                return Err(LyraUnifiedError::Runtime {
                    message: format!(
                        "Slot {} requires argument at position {}, but only {} arguments provided",
                        slot_display, slot_index + 1, args.len()
                    ),
                    context: crate::unified_errors::RuntimeContext {
                        current_function: Some("substitute_slots_in_expr".to_string()),
                        call_stack_depth: 0,
                        local_variables: vec![],
                        evaluation_mode: "pure_function_substitution".to_string(),
                    },
                    recoverable: false,
                });
            }
            
            // Convert the argument Value to an Expr for substitution
            // Handle edge case: argument conversion failures gracefully
            match value_to_expr(&args[slot_index]) {
                Ok(expr) => Ok(expr),
                Err(e) => {
                    return Err(LyraUnifiedError::Runtime {
                        message: format!("Failed to convert argument {} to expression: {}", slot_index + 1, e),
                        context: crate::unified_errors::RuntimeContext {
                            current_function: Some("substitute_slots_in_expr".to_string()),
                            call_stack_depth: 0,
                            local_variables: vec![],
                            evaluation_mode: "pure_function_substitution".to_string(),
                        },
                        recoverable: false,
                    });
                }
            }
        }
        
        // Handle function calls (including operations like Plus, Times, etc.)
        Expr::Function { head, args: call_args } => {
            // Handle edge case: too many function arguments (potential DoS protection)
            if call_args.len() > 1000 {
                return Err(LyraUnifiedError::Runtime {
                    message: format!("Function call has {} arguments, exceeding maximum of 1000", call_args.len()),
                    context: crate::unified_errors::RuntimeContext {
                        current_function: Some("substitute_slots_in_expr_with_depth".to_string()),
                        call_stack_depth: depth,
                        local_variables: vec![],
                        evaluation_mode: "pure_function_substitution".to_string(),
                    },
                    recoverable: false,
                });
            }
            
            // Substitute slots in the head (function name)
            let substituted_head = substitute_slots_in_expr_with_depth(head, args, depth + 1)?;
            
            // Substitute slots in all arguments
            let mut substituted_args = Vec::new();
            for arg in call_args {
                substituted_args.push(substitute_slots_in_expr_with_depth(arg, args, depth + 1)?);
            }
            
            // Try to evaluate mathematical operations if possible
            if let Expr::Symbol(Symbol { name }) = &substituted_head {
                if let Some(result) = evaluate_operation(name, &substituted_args)? {
                    return Ok(result);
                }
            }
            
            Ok(Expr::Function {
                head: Box::new(substituted_head),
                args: substituted_args,
            })
        }
        
        // Handle lists (e.g., {#, # + 1, # * 2})
        Expr::List(elements) => {
            // Handle edge case: very large lists (potential DoS protection)
            if elements.len() > 100000 {
                return Err(LyraUnifiedError::Runtime {
                    message: format!("List has {} elements, exceeding maximum of 100000", elements.len()),
                    context: crate::unified_errors::RuntimeContext {
                        current_function: Some("substitute_slots_in_expr_with_depth".to_string()),
                        call_stack_depth: depth,
                        local_variables: vec![],
                        evaluation_mode: "pure_function_substitution".to_string(),
                    },
                    recoverable: false,
                });
            }
            
            let mut substituted_elements = Vec::new();
            for element in elements {
                substituted_elements.push(substitute_slots_in_expr_with_depth(element, args, depth + 1)?);
            }
            
            Ok(Expr::List(substituted_elements))
        }
        
        // Handle nested pure functions (rare but possible)
        Expr::PureFunction { body, max_slot: _ } => {
            // For nested pure functions, we don't substitute slots in the inner function
            // The inner function maintains its own slot scope
            Ok(expr.clone())
        }
        
        // Handle assignments 
        Expr::Assignment { lhs, rhs, delayed } => {
            let substituted_lhs = substitute_slots_in_expr_with_depth(lhs, args, depth + 1)?;
            let substituted_rhs = substitute_slots_in_expr_with_depth(rhs, args, depth + 1)?;
            
            Ok(Expr::Assignment {
                lhs: Box::new(substituted_lhs),
                rhs: Box::new(substituted_rhs),
                delayed: *delayed,
            })
        }
        
        // Handle rules (e.g., # -> # + 1, x_ :> # + x)
        Expr::Rule { lhs, rhs, delayed } => {
            let substituted_lhs = substitute_slots_in_expr_with_depth(lhs, args, depth + 1)?;
            let substituted_rhs = substitute_slots_in_expr_with_depth(rhs, args, depth + 1)?;
            
            Ok(Expr::Rule {
                lhs: Box::new(substituted_lhs),
                rhs: Box::new(substituted_rhs),
                delayed: *delayed,
            })
        }
        
        // Literals don't need substitution - pass through unchanged
        Expr::Number(_) | Expr::String(_) | Expr::Symbol(_) => {
            Ok(expr.clone())
        }
        
        // Handle other expression types by recursively processing their children
        _ => {
            // For any other expression types, we pass them through unchanged
            // This includes things like Pattern, Quote, etc. that don't contain slots
            Ok(expr.clone())
        }
    }
}

/// Convert a Value to an AST Expr for processing
/// 
/// This function converts runtime Values back to AST expressions so we can
/// perform slot substitution at the AST level.
fn value_to_expr(value: &Value) -> LyraResult<Expr> {
    match value {
        Value::Integer(n) => Ok(Expr::Number(Number::Integer(*n))),
        Value::Real(f) => Ok(Expr::Number(Number::Real(*f))),
        Value::String(s) => Ok(Expr::String(s.clone())),
        Value::Symbol(s) => Ok(Expr::Symbol(Symbol { name: s.clone() })),
        Value::Boolean(b) => {
            // Boolean is not in AST, represent as symbol True/False
            Ok(Expr::Symbol(Symbol { 
                name: if *b { "True".to_string() } else { "False".to_string() } 
            }))
        }
        Value::List(elements) => {
            let mut ast_elements = Vec::new();
            for element in elements {
                ast_elements.push(value_to_expr(element)?);
            }
            Ok(Expr::List(ast_elements))
        }
        Value::PureFunction { body } => {
            let body_expr = value_to_expr(body)?;
            Ok(Expr::PureFunction {
                body: Box::new(body_expr),
                max_slot: None, // Will be calculated if needed
            })
        }
        Value::Slot { number } => Ok(Expr::Slot { number: *number }),
        Value::Quote(expr) => {
            // Extract the expression from the quote
            Ok((**expr).clone())
        },
        _ => {
            // For other Value types, we create a symbol representation
            // This allows complex values to be referenced in expressions
            Ok(Expr::Symbol(Symbol { name: format!("Value[{:?}]", value) }))
        }
    }
}

/// Convert an AST Expr back to a Value for VM execution
/// 
/// This function converts the AST with substituted slots back to a Value
/// that can be executed or returned by the VM.
fn expr_to_value(expr: &Expr) -> LyraResult<Value> {
    match expr {
        Expr::Number(Number::Integer(n)) => Ok(Value::Integer(*n)),
        Expr::Number(Number::Real(f)) => Ok(Value::Real(*f)),
        Expr::String(s) => Ok(Value::String(s.clone())),
        Expr::Symbol(Symbol { name }) => Ok(Value::Symbol(name.clone())),
        Expr::List(elements) => {
            let mut value_elements = Vec::new();
            for element in elements {
                value_elements.push(expr_to_value(element)?);
            }
            Ok(Value::List(value_elements))
        }
        Expr::Function { head, args } => {
            // Convert function calls to List representation for VM
            let head_value = expr_to_value(head)?;
            let mut call_elements = vec![head_value];
            for arg in args {
                call_elements.push(expr_to_value(arg)?);
            }
            Ok(Value::List(call_elements))
        }
        Expr::PureFunction { body, max_slot: _ } => {
            let body_value = expr_to_value(body)?;
            Ok(Value::PureFunction {
                body: Box::new(body_value),
            })
        }
        Expr::Slot { number } => Ok(Value::Slot { number: *number }),
        _ => {
            // For complex expressions that can't be directly converted,
            // we represent them as quoted expressions
            Ok(Value::Quote(Box::new(expr.clone())))
        }
    }
}

/// Try to evaluate a mathematical operation if both operands are literals
/// 
/// This function handles Wolfram Language mathematical operations like Plus, Times, etc.
/// when all arguments are literal values that can be computed immediately.
fn evaluate_operation(operation_name: &str, args: &[Expr]) -> LyraResult<Option<Expr>> {
    match operation_name {
        "Plus" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Integer(a + b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a + b))))
                    }
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(*a as f64 + b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a + *b as f64))))
                    }
                    _ => Ok(None), // Can't evaluate
                }
            } else {
                Ok(None)
            }
        }
        "Times" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Integer(a * b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a * b))))
                    }
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(*a as f64 * b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a * *b as f64))))
                    }
                    _ => Ok(None), // Can't evaluate
                }
            } else {
                Ok(None)
            }
        }
        "Subtract" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Integer(a - b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a - b))))
                    }
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Real(b))) => {
                        Ok(Some(Expr::Number(Number::Real(*a as f64 - b))))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Integer(b))) => {
                        Ok(Some(Expr::Number(Number::Real(a - *b as f64))))
                    }
                    _ => Ok(None), // Can't evaluate
                }
            } else {
                Ok(None)
            }
        }
        "Divide" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Integer(b))) => {
                        if *b == 0 {
                            Err(LyraUnifiedError::Runtime {
                                message: "Division by zero".to_string(),
                                context: crate::unified_errors::RuntimeContext {
                                    current_function: Some("evaluate_operation".to_string()),
                                    call_stack_depth: 0,
                                    local_variables: vec![],
                                    evaluation_mode: "pure_function_evaluation".to_string(),
                                },
                                recoverable: false,
                            })
                        } else {
                            Ok(Some(Expr::Number(Number::Integer(a / b))))
                        }
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Real(b))) => {
                        if *b == 0.0 {
                            Err(LyraUnifiedError::Runtime {
                                message: "Division by zero".to_string(),
                                context: crate::unified_errors::RuntimeContext {
                                    current_function: Some("evaluate_operation".to_string()),
                                    call_stack_depth: 0,
                                    local_variables: vec![],
                                    evaluation_mode: "pure_function_evaluation".to_string(),
                                },
                                recoverable: false,
                            })
                        } else {
                            Ok(Some(Expr::Number(Number::Real(a / b))))
                        }
                    }
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Real(b))) => {
                        if *b == 0.0 {
                            Err(LyraUnifiedError::Runtime {
                                message: "Division by zero".to_string(),
                                context: crate::unified_errors::RuntimeContext {
                                    current_function: Some("evaluate_operation".to_string()),
                                    call_stack_depth: 0,
                                    local_variables: vec![],
                                    evaluation_mode: "pure_function_evaluation".to_string(),
                                },
                                recoverable: false,
                            })
                        } else {
                            Ok(Some(Expr::Number(Number::Real(*a as f64 / b))))
                        }
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Integer(b))) => {
                        if *b == 0 {
                            Err(LyraUnifiedError::Runtime {
                                message: "Division by zero".to_string(),
                                context: crate::unified_errors::RuntimeContext {
                                    current_function: Some("evaluate_operation".to_string()),
                                    call_stack_depth: 0,
                                    local_variables: vec![],
                                    evaluation_mode: "pure_function_evaluation".to_string(),
                                },
                                recoverable: false,
                            })
                        } else {
                            Ok(Some(Expr::Number(Number::Real(a / *b as f64))))
                        }
                    }
                    _ => Ok(None), // Can't evaluate
                }
            } else {
                Ok(None)
            }
        }
        "Greater" => {
            if args.len() == 2 {
                match (&args[0], &args[1]) {
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Integer(b))) => {
                        let result = if *a > *b { "True" } else { "False" };
                        Ok(Some(Expr::Symbol(Symbol { name: result.to_string() })))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Real(b))) => {
                        let result = if *a > *b { "True" } else { "False" };
                        Ok(Some(Expr::Symbol(Symbol { name: result.to_string() })))
                    }
                    (Expr::Number(Number::Integer(a)), Expr::Number(Number::Real(b))) => {
                        let result = if (*a as f64) > *b { "True" } else { "False" };
                        Ok(Some(Expr::Symbol(Symbol { name: result.to_string() })))
                    }
                    (Expr::Number(Number::Real(a)), Expr::Number(Number::Integer(b))) => {
                        let result = if *a > (*b as f64) { "True" } else { "False" };
                        Ok(Some(Expr::Symbol(Symbol { name: result.to_string() })))
                    }
                    _ => Ok(None), // Can't evaluate
                }
            } else {
                Ok(None)
            }
        }
        _ => Ok(None), // Unknown operation, can't evaluate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_basic_slot_substitution() {
        // Test Plus[#, 1] with argument 5 -> 6
        let body = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Slot { number: None },
                Expr::Number(Number::Integer(1)),
            ],
        }));
        
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(5)];
        
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(6));
    }
    
    #[test]
    fn test_numbered_slot_substitution() {
        // Test Plus[#1, #2] with arguments [3, 4] -> 7
        let body = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Slot { number: Some(1) },
                Expr::Slot { number: Some(2) },
            ],
        }));
        
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(3), Value::Integer(4)];
        
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(7));
    }
    
    #[test]
    fn test_slot_out_of_bounds_error() {
        // Test #2 with only one argument should error
        let body = Value::Quote(Box::new(Expr::Slot { number: Some(2) }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(5)];
        
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires argument at position 2"));
    }
    
    #[test]
    fn test_no_arguments_error() {
        // Test # with no arguments should error
        let body = Value::Quote(Box::new(Expr::Slot { number: None }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![];
        
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("requires arguments but none were provided"));
    }
    
    #[test] 
    fn test_zero_slot_error() {
        // Test #0 should error (slots must be >= 1)
        let body = Value::Quote(Box::new(Expr::Slot { number: Some(0) }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(5)];
        
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Slot indices must be >= 1"));
    }
    
    #[test]
    fn test_very_large_slot_error() {
        // Test #10001 should error (exceeds maximum)
        let body = Value::Quote(Box::new(Expr::Slot { number: Some(10001) }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(5)];
        
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum allowed slot number"));
    }
    
    #[test]
    fn test_mixed_value_types() {
        // Test slot substitution with different value types
        let body = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Slot { number: Some(1) }, // Integer
                Expr::Slot { number: Some(2) }, // Real
            ],
        }));
        
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(5), Value::Real(3.14)];
        
        let result = substitute_slots(&pure_func, &args).unwrap();
        // Should be 5.0 + 3.14 = 8.14
        assert_eq!(result, Value::Real(8.14));
    }
    
    #[test]
    fn test_nested_list_substitution() {
        // Test deeply nested list with slots
        let body = Value::Quote(Box::new(Expr::List(vec![
            Expr::Slot { number: None },
            Expr::List(vec![
                Expr::Slot { number: Some(1) },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Number(Number::Integer(1)),
                    ],
                },
            ])
        ])));
        
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(10)];
        
        let result = substitute_slots(&pure_func, &args).unwrap();
        let expected = Value::List(vec![
            Value::Integer(10),
            Value::List(vec![
                Value::Integer(10),
                Value::Integer(11), // 10 + 1
            ])
        ]);
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_constant_pure_function() {
        // Test pure function with no slots (constant function)
        let body = Value::Quote(Box::new(Expr::Number(Number::Integer(42))));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        let args = vec![Value::Integer(999)]; // Argument should be ignored
        
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(42));
    }
}
