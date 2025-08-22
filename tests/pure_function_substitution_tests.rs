/// Pure Function Slot Substitution Tests
/// 
/// This module defines the expected behavior of slot substitution in pure functions
/// following Test-Driven Development principles. These tests will drive the 
/// implementation of the slot substitution algorithm.

use lyra::vm::Value;
use lyra::ast::Expr;

/// Test suite for basic slot substitution functionality
#[cfg(test)]
mod basic_substitution_tests {
    use super::*;

    #[test]
    fn test_single_slot_substitution() {
        // Test case: # + 1 & applied to 5 should become 5 + 1 = 6
        // Pure function: PureFunction { body: BinaryOp(Slot, Add, Integer(1)) }
        // Arguments: [5]
        // Expected: 6
        
        let pure_func = create_pure_function_add_one();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(6));
    }

    #[test] 
    fn test_numbered_slot_substitution() {
        // Test case: #1 + #2 & applied to (3, 7) should become 3 + 7 = 10
        // Pure function: PureFunction { body: BinaryOp(Slot(1), Add, Slot(2)) }
        // Arguments: [3, 7]
        // Expected: 10
        
        let pure_func = create_pure_function_add_two_args();
        let args = vec![Value::Integer(3), Value::Integer(7)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(10));
    }

    #[test]
    fn test_multiple_same_slot_substitution() {
        // Test case: # * # & applied to 4 should become 4 * 4 = 16
        // Pure function: PureFunction { body: BinaryOp(Slot, Multiply, Slot) }
        // Arguments: [4]
        // Expected: 16
        
        let pure_func = create_pure_function_square();
        let args = vec![Value::Integer(4)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(16));
    }

    #[test]
    fn test_slot_with_no_arguments_error() {
        // Test case: # + 1 & applied to no arguments should error
        let pure_func = create_pure_function_add_one();
        let args = vec![];
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not enough arguments"));
    }

    #[test]
    fn test_numbered_slot_out_of_bounds_error() {
        // Test case: #2 & applied to [5] should error (only 1 argument, but requesting #2)
        let pure_func = create_pure_function_second_arg();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("slot index out of bounds"));
    }
}

/// Test suite for nested slot substitution
#[cfg(test)]
mod nested_substitution_tests {
    use super::*;

    #[test]
    fn test_nested_list_slot_substitution() {
        // Test case: {#, # + 1, # * 2} & applied to 3 should become {3, 4, 6}
        let pure_func = create_pure_function_nested_list();
        let args = vec![Value::Integer(3)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        let expected = Value::List(vec![
            Value::Integer(3),
            Value::Integer(4), 
            Value::Integer(6)
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_nested_function_call_substitution() {
        // Test case: Sin[#] & applied to π/2 should become Sin[π/2]
        let pure_func = create_pure_function_sin_call();
        let args = vec![Value::Real(std::f64::consts::PI / 2.0)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        
        // We expect a function call Value::FunctionCall or similar
        // This will need to be evaluated later by the VM
        assert!(matches!(result, Value::List(_))); // Function calls are represented as lists
    }

    #[test]
    fn test_deeply_nested_slot_substitution() {
        // Test case: {{#, #}, {# + 1, # * 2}} & applied to 5
        // Expected: {{5, 5}, {6, 10}}
        let pure_func = create_pure_function_deeply_nested();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        
        let expected = Value::List(vec![
            Value::List(vec![Value::Integer(5), Value::Integer(5)]),
            Value::List(vec![Value::Integer(6), Value::Integer(10)])
        ]);
        assert_eq!(result, expected);
    }
}

/// Test suite for complex slot patterns
#[cfg(test)]
mod complex_substitution_tests {
    use super::*;

    #[test]
    fn test_mixed_numbered_and_unnumbered_slots() {
        // Test case: # + #1 + #2 & applied to (10, 20) should become 10 + 10 + 20 = 40
        // # defaults to #1 when numbered slots are present
        let pure_func = create_pure_function_mixed_slots();
        let args = vec![Value::Integer(10), Value::Integer(20)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(40));
    }

    #[test]
    fn test_out_of_order_numbered_slots() {
        // Test case: #3 + #1 + #2 & applied to (1, 2, 3) should become 3 + 1 + 2 = 6
        let pure_func = create_pure_function_out_of_order_slots();
        let args = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(6));
    }

    #[test]
    fn test_slot_in_binary_operation() {
        // Test case: # > 0 & applied to -5 should become -5 > 0 = False
        let pure_func = create_pure_function_comparison();
        let args = vec![Value::Integer(-5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }

    #[test]
    fn test_slot_with_string_argument() {
        // Test case: StringLength[#] & applied to "hello" should substitute correctly
        let pure_func = create_pure_function_string_length();
        let args = vec![Value::String("hello".to_string())];
        let result = substitute_slots(&pure_func, &args).unwrap();
        
        // The result should be a function call that when evaluated gives 5
        // For now, we just check that substitution happened without error
        assert!(result != Value::Missing);
    }
}

/// Test suite for edge cases and error conditions
#[cfg(test)]
mod edge_case_tests {
    use super::*;

    #[test]
    fn test_pure_function_with_no_slots() {
        // Test case: 42 & applied to anything should return 42 (constant function)
        let pure_func = create_pure_function_constant();
        let args = vec![Value::Integer(999)]; // Argument should be ignored
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_empty_pure_function_body() {
        // Test case: empty body should handle gracefully
        let pure_func = create_pure_function_empty();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args);
        // This might be an error or return Missing, depending on implementation
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    fn test_slot_substitution_with_different_value_types() {
        // Test case: # & applied to various types should work
        let pure_func = create_pure_function_identity();
        
        // Test with integer
        let result1 = substitute_slots(&pure_func, &vec![Value::Integer(42)]).unwrap();
        assert_eq!(result1, Value::Integer(42));
        
        // Test with string
        let result2 = substitute_slots(&pure_func, &vec![Value::String("test".to_string())]).unwrap();
        assert_eq!(result2, Value::String("test".to_string()));
        
        // Test with list
        let list_arg = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let result3 = substitute_slots(&pure_func, &vec![list_arg.clone()]).unwrap();
        assert_eq!(result3, list_arg);
    }

    #[test]
    fn test_very_large_slot_number() {
        // Test case: #1000 & should error gracefully with reasonable arguments
        let pure_func = create_pure_function_large_slot_number();
        let args = vec![Value::Integer(1)]; // Only 1 argument for slot #1000
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("slot index out of bounds"));
    }
}

/// Test suite for performance and stress testing
#[cfg(test)]
mod performance_tests {
    use super::*;

    #[test]
    fn test_deeply_nested_substitution_performance() {
        // Test case: very deep nesting should complete in reasonable time
        let pure_func = create_deeply_nested_pure_function(100); // 100 levels deep
        let args = vec![Value::Integer(1)];
        
        let start = std::time::Instant::now();
        let result = substitute_slots(&pure_func, &args);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 1000); // Should complete in under 1 second
    }

    #[test]
    fn test_many_slots_substitution_performance() {
        // Test case: many slots should substitute efficiently
        let pure_func = create_pure_function_many_slots(50); // 50 different slots
        let args: Vec<Value> = (1..=50).map(|i| Value::Integer(i)).collect();
        
        let start = std::time::Instant::now();
        let result = substitute_slots(&pure_func, &args);
        let duration = start.elapsed();
        
        assert!(result.is_ok());
        assert!(duration.as_millis() < 500); // Should complete quickly
    }
}

// =============================================================================
// Helper Functions for Creating Test Pure Functions
// =============================================================================

/// Create a pure function that adds 1 to its argument: Plus[#, 1] &
fn create_pure_function_add_one() -> Value {
    use lyra::ast::{Expr, Symbol, Number};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Slot { number: None },
            Expr::Number(Number::Integer(1)),
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function that adds two arguments: Plus[#1, #2] &
fn create_pure_function_add_two_args() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Slot { number: Some(1) },
            Expr::Slot { number: Some(2) },
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function that squares its argument: Times[#, #] &
fn create_pure_function_square() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
        args: vec![
            Expr::Slot { number: None },
            Expr::Slot { number: None },
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function that accesses the second argument: #2 &
fn create_pure_function_second_arg() -> Value {
    let body = Value::Quote(Box::new(Expr::Slot { number: Some(2) }));
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with nested list: {#, Plus[#, 1], Times[#, 2]} &
fn create_pure_function_nested_list() -> Value {
    use lyra::ast::{Expr, Symbol, Number};
    
    let body = Value::Quote(Box::new(Expr::List(vec![
        Expr::Slot { number: None },
        Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Slot { number: None },
                Expr::Number(Number::Integer(1)),
            ],
        },
        Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
            args: vec![
                Expr::Slot { number: None },
                Expr::Number(Number::Integer(2)),
            ],
        },
    ])));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with function call: Sin[#] &
fn create_pure_function_sin_call() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
        args: vec![Expr::Slot { number: None }],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a deeply nested pure function: {{#, #}, {Plus[#, 1], Times[#, 2]}} &
fn create_pure_function_deeply_nested() -> Value {
    use lyra::ast::{Expr, Symbol, Number};
    
    let body = Value::Quote(Box::new(Expr::List(vec![
        Expr::List(vec![
            Expr::Slot { number: None },
            Expr::Slot { number: None },
        ]),
        Expr::List(vec![
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1)),
                ],
            },
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2)),
                ],
            },
        ]),
    ])));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with mixed slots: Plus[Plus[#, #1], #2] &
fn create_pure_function_mixed_slots() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None }, // # defaults to #1
                    Expr::Slot { number: Some(1) },
                ],
            },
            Expr::Slot { number: Some(2) },
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with out-of-order slots: Plus[Plus[#3, #1], #2] &
fn create_pure_function_out_of_order_slots() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(3) },
                    Expr::Slot { number: Some(1) },
                ],
            },
            Expr::Slot { number: Some(2) },
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with comparison: Greater[#, 0] &
fn create_pure_function_comparison() -> Value {
    use lyra::ast::{Expr, Symbol, Number};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
        args: vec![
            Expr::Slot { number: None },
            Expr::Number(Number::Integer(0)),
        ],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with string operation: StringLength[#] &
fn create_pure_function_string_length() -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "StringLength".to_string() })),
        args: vec![Expr::Slot { number: None }],
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

/// Create a constant pure function: 42 &
fn create_pure_function_constant() -> Value {
    use lyra::ast::{Expr, Number};
    
    let body = Value::Quote(Box::new(Expr::Number(Number::Integer(42))));
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with empty body (Missing value)
fn create_pure_function_empty() -> Value {
    let body = Value::Missing;
    Value::PureFunction { body: Box::new(body) }
}

/// Create an identity pure function: # &
fn create_pure_function_identity() -> Value {
    let body = Value::Quote(Box::new(Expr::Slot { number: None }));
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with large slot number: #1000 &
fn create_pure_function_large_slot_number() -> Value {
    let body = Value::Quote(Box::new(Expr::Slot { number: Some(1000) }));
    Value::PureFunction { body: Box::new(body) }
}

/// Create a deeply nested pure function for performance testing
fn create_deeply_nested_pure_function(depth: usize) -> Value {
    use lyra::ast::{Expr, Symbol, Number};
    
    let mut current_expr = Expr::Slot { number: None };
    
    // Create nested Plus operations
    for i in 0..depth {
        current_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                current_expr,
                Expr::Number(Number::Integer(i as i64)),
            ],
        };
    }
    
    let body = Value::Quote(Box::new(current_expr));
    Value::PureFunction { body: Box::new(body) }
}

/// Create a pure function with many slots for performance testing
fn create_pure_function_many_slots(count: usize) -> Value {
    use lyra::ast::{Expr, Symbol};
    
    let mut args = Vec::new();
    for i in 1..=count {
        args.push(Expr::Slot { number: Some(i) });
    }
    
    let body = Value::Quote(Box::new(Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args,
    }));
    
    Value::PureFunction { body: Box::new(body) }
}

// =============================================================================
// Core Slot Substitution Function (To Be Implemented)
// =============================================================================

/// Substitute slots in a pure function with the provided arguments
/// 
/// This delegates to the actual implementation in the pure_function module.
/// 
/// # Arguments
/// * `pure_function` - The pure function Value containing the AST body
/// * `args` - The arguments to substitute for slots
/// 
/// # Returns
/// * `Ok(Value)` - The result after slot substitution
/// * `Err(String)` - Error if substitution fails
fn substitute_slots(pure_function: &Value, args: &[Value]) -> Result<Value, String> {
    lyra::pure_function::substitute_slots(pure_function, args)
        .map_err(|e| e.to_string())
}

/// Test suite for arithmetic operations with slots
#[cfg(test)]
mod arithmetic_tests {
    use super::*;

    #[test]
    fn test_addition_with_integers() {
        let pure_func = create_pure_function_add_one();
        let args = vec![Value::Integer(42)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(43));
    }

    #[test]
    fn test_addition_with_reals() {
        let pure_func = create_pure_function_add_one();
        let args = vec![Value::Real(3.14)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Real(4.14));
    }

    #[test]
    fn test_multiplication_square() {
        let pure_func = create_pure_function_square();
        let args = vec![Value::Integer(7)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(49));
    }

    #[test]
    fn test_multiplication_with_reals() {
        let pure_func = create_pure_function_square();
        let args = vec![Value::Real(2.5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Real(6.25));
    }

    #[test]
    fn test_two_argument_addition() {
        let pure_func = create_pure_function_add_two_args();
        let args = vec![Value::Integer(15), Value::Integer(27)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_mixed_types_addition() {
        let pure_func = create_pure_function_add_two_args();
        let args = vec![Value::Integer(10), Value::Real(5.5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Real(15.5));
    }

    #[test]
    fn test_comparison_greater_true() {
        let pure_func = create_pure_function_comparison();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Symbol("True".to_string()));
    }

    #[test]
    fn test_comparison_greater_false() {
        let pure_func = create_pure_function_comparison();
        let args = vec![Value::Integer(-3)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Symbol("False".to_string()));
    }

    #[test]
    fn test_zero_comparison() {
        let pure_func = create_pure_function_comparison();
        let args = vec![Value::Integer(0)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Symbol("False".to_string()));
    }
}

/// Test suite for complex slot patterns
#[cfg(test)]
mod slot_pattern_tests {
    use super::*;

    #[test]
    fn test_out_of_order_slot_usage() {
        let pure_func = create_pure_function_out_of_order_slots();
        let args = vec![Value::Integer(10), Value::Integer(20), Value::Integer(30)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // #3 + #1 + #2 = 30 + 10 + 20 = 60
        assert_eq!(result, Value::Integer(60));
    }

    #[test]
    fn test_mixed_numbered_and_unnumbered_slots() {
        let pure_func = create_pure_function_mixed_slots();
        let args = vec![Value::Integer(5), Value::Integer(10)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // Plus[Plus[#, #1], #2] = Plus[Plus[5, 5], 10] = 20
        assert_eq!(result, Value::Integer(20));
    }

    #[test]
    fn test_repeated_slot_usage() {
        // Create: Times[#, Plus[#, 1]] & (multiply by self plus one)
        use lyra::ast::{Expr, Symbol, Number};
        let body = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
            args: vec![
                Expr::Slot { number: None },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Number(Number::Integer(1)),
                    ],
                },
            ],
        }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        
        let args = vec![Value::Integer(4)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // 4 * (4 + 1) = 4 * 5 = 20
        assert_eq!(result, Value::Integer(20));
    }

    #[test]
    fn test_slot_in_nested_context() {
        // Create: {#, {#, Plus[#, #]}} &
        use lyra::ast::{Expr, Symbol};
        let body = Value::Quote(Box::new(Expr::List(vec![
            Expr::Slot { number: None },
            Expr::List(vec![
                Expr::Slot { number: None },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Slot { number: None },
                    ],
                },
            ]),
        ])));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        
        let args = vec![Value::Integer(7)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        let expected = Value::List(vec![
            Value::Integer(7),
            Value::List(vec![
                Value::Integer(7),
                Value::Integer(14), // 7 + 7
            ]),
        ]);
        assert_eq!(result, expected);
    }
}

/// Test suite for error conditions and edge cases
#[cfg(test)]
mod error_condition_tests {
    use super::*;

    #[test]
    fn test_slot_index_zero_error() {
        // Create function with slot #0
        use lyra::ast::Expr;
        let body = Value::Quote(Box::new(Expr::Slot { number: Some(0) }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Slot indices must be >= 1"));
    }

    #[test]
    fn test_very_large_slot_number_error() {
        let pure_func = create_pure_function_large_slot_number();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires argument at position"));
    }

    #[test]
    fn test_no_arguments_provided() {
        let pure_func = create_pure_function_identity();
        let args = vec![];
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires arguments but none were provided"));
    }

    #[test]
    fn test_insufficient_arguments_for_numbered_slot() {
        let pure_func = create_pure_function_second_arg();
        let args = vec![Value::Integer(1)]; // Only 1 argument, but need #2
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires argument at position 2"));
    }

    #[test]
    fn test_mixed_slot_insufficient_args() {
        let pure_func = create_pure_function_mixed_slots();
        let args = vec![Value::Integer(5)]; // Need at least 2 arguments
        let result = substitute_slots(&pure_func, &args);
        assert!(result.is_err());
    }
}

/// Test suite for different value types as arguments
#[cfg(test)]
mod value_type_tests {
    use super::*;

    #[test]
    fn test_string_argument() {
        let pure_func = create_pure_function_identity();
        let args = vec![Value::String("hello".to_string())];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[test]
    fn test_boolean_argument() {
        let pure_func = create_pure_function_identity();
        let args = vec![Value::Boolean(true)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }

    #[test]
    fn test_list_argument() {
        let pure_func = create_pure_function_identity();
        let list_arg = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        let args = vec![list_arg.clone()];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, list_arg);
    }

    #[test]
    fn test_nested_list_argument() {
        let pure_func = create_pure_function_identity();
        let nested_list = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ]);
        let args = vec![nested_list.clone()];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, nested_list);
    }

    #[test]
    fn test_symbol_argument() {
        let pure_func = create_pure_function_identity();
        let args = vec![Value::Symbol("TestSymbol".to_string())];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Symbol("TestSymbol".to_string()));
    }

    #[test]
    fn test_missing_argument() {
        let pure_func = create_pure_function_identity();
        let args = vec![Value::Missing];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Missing);
    }
}

/// Test suite for complex nested expressions
#[cfg(test)]
mod complex_expression_tests {
    use super::*;

    #[test]
    fn test_nested_list_with_operations() {
        let pure_func = create_pure_function_nested_list();
        let args = vec![Value::Integer(6)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        let expected = Value::List(vec![
            Value::Integer(6),
            Value::Integer(7),  // 6 + 1
            Value::Integer(12), // 6 * 2
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_deeply_nested_structure() {
        let pure_func = create_pure_function_deeply_nested();
        let args = vec![Value::Integer(3)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        let expected = Value::List(vec![
            Value::List(vec![Value::Integer(3), Value::Integer(3)]),
            Value::List(vec![Value::Integer(4), Value::Integer(6)]), // 3+1, 3*2
        ]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_function_call_substitution() {
        let pure_func = create_pure_function_sin_call();
        let args = vec![Value::Real(1.57)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // Should return a function call list: [Sin, 1.57]
        if let Value::List(elements) = result {
            assert_eq!(elements.len(), 2);
            assert_eq!(elements[0], Value::Symbol("Sin".to_string()));
            assert_eq!(elements[1], Value::Real(1.57));
        } else {
            panic!("Expected function call to be converted to list");
        }
    }

    #[test]
    fn test_multiple_nested_operations() {
        // Create: Plus[Times[#, 2], Plus[#, 3]] & (2*x + x+3)
        use lyra::ast::{Expr, Symbol, Number};
        let body = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Number(Number::Integer(2)),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Number(Number::Integer(3)),
                    ],
                },
            ],
        }));
        let pure_func = Value::PureFunction { body: Box::new(body) };
        
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // 2*5 + (5+3) = 10 + 8 = 18
        assert_eq!(result, Value::Integer(18));
    }
}

/// Test suite for constant and special functions
#[cfg(test)]
mod special_function_tests {
    use super::*;

    #[test]
    fn test_constant_function() {
        let pure_func = create_pure_function_constant();
        let args = vec![Value::Integer(999)]; // Argument should be ignored
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_constant_function_multiple_args() {
        let pure_func = create_pure_function_constant();
        let args = vec![Value::Integer(1), Value::String("test".to_string()), Value::Boolean(true)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Integer(42)); // Still constant
    }

    #[test]
    fn test_empty_function_body() {
        let pure_func = create_pure_function_empty();
        let args = vec![Value::Integer(5)];
        let result = substitute_slots(&pure_func, &args).unwrap();
        assert_eq!(result, Value::Missing);
    }

    #[test]
    fn test_string_length_function_call() {
        let pure_func = create_pure_function_string_length();
        let args = vec![Value::String("hello".to_string())];
        let result = substitute_slots(&pure_func, &args).unwrap();
        // Should return a function call list: [StringLength, "hello"]
        if let Value::List(elements) = result {
            assert_eq!(elements.len(), 2);
            assert_eq!(elements[0], Value::Symbol("StringLength".to_string()));
            assert_eq!(elements[1], Value::String("hello".to_string()));
        } else {
            panic!("Expected function call to be converted to list");
        }
    }
}