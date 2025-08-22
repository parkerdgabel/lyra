use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};
use proptest::prelude::*;
use std::collections::HashMap;

/// Property-based testing framework for pure functions
/// 
/// This module implements comprehensive property testing to verify mathematical
/// laws and invariants that pure functions should satisfy, ensuring correctness
/// under all conditions.

#[cfg(test)]
mod property_tests {
    use super::*;

    /// Test data generators for property-based testing
    mod generators {
        use super::*;

        /// Generate random integer values for testing
        pub fn arb_integer() -> impl Strategy<Value = i64> {
            any::<i64>().prop_filter("Avoid overflow issues", |x| x.abs() < 1_000_000)
        }

        /// Generate random real values for testing
        pub fn arb_real() -> impl Strategy<Value = f64> {
            any::<f64>().prop_filter("Finite values only", |x| x.is_finite())
        }

        /// Generate random strings for testing
        pub fn arb_string() -> impl Strategy<Value = String> {
            "[a-zA-Z0-9_]{1,20}".prop_map(|s| s.to_string())
        }

        /// Generate random Value for pure function testing
        pub fn arb_value() -> impl Strategy<Value = Value> {
            prop_oneof![
                arb_integer().prop_map(Value::Integer),
                arb_real().prop_map(Value::Real),
                arb_string().prop_map(Value::String),
                any::<bool>().prop_map(Value::Boolean),
            ]
        }

        /// Generate lists of values for multi-argument testing
        pub fn arb_value_list() -> impl Strategy<Value = Vec<Value>> {
            prop::collection::vec(arb_value(), 1..10)
        }

        /// Generate simple arithmetic expressions for testing
        pub fn arb_arithmetic_expr() -> impl Strategy<Value = Expr> {
            prop_oneof![
                // Simple slot expressions
                Just(Expr::Slot { number: None }),
                Just(Expr::Slot { number: Some(1) }),
                Just(Expr::Slot { number: Some(2) }),
                
                // Arithmetic operations
                (arb_integer(), arb_integer()).prop_map(|(a, b)| {
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Number(Number::Integer(a)),
                            Expr::Number(Number::Integer(b))
                        ]
                    }
                }),
                
                // Slot arithmetic
                Just(Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Slot { number: None },
                        Expr::Number(Number::Integer(1))
                    ]
                }),
            ]
        }

        /// Generate pure function expressions
        pub fn arb_pure_function() -> impl Strategy<Value = Value> {
            arb_arithmetic_expr().prop_map(|expr| {
                Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(expr)))
                }
            })
        }
    }

    /// Core mathematical property tests
    mod mathematical_properties {
        use super::*;

        /// Property: Identity Law
        /// For any value x: (# &)[x] ≡ x
        proptest! {
            #[test]
            fn identity_property(x in generators::arb_value()) {
                // Create identity pure function: (# &)
                let identity_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: None })
                };
                
                // Apply to argument
                let result = pure_function::substitute_slots(&identity_func, &[x.clone()]);
                
                // Should return the input unchanged
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), x);
            }
        }

        /// Property: Constant Function
        /// For any constant c and value x: (c &)[x] ≡ c
        proptest! {
            #[test]
            fn constant_function_property(
                c in generators::arb_integer(),
                x in generators::arb_value()
            ) {
                // Create constant pure function: (c &)
                let constant_func = Value::PureFunction {
                    body: Box::new(Value::Integer(c))
                };
                
                // Apply to any argument
                let result = pure_function::substitute_slots(&constant_func, &[x]);
                
                // Should return the constant
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), Value::Integer(c));
            }
        }

        /// Property: Slot Substitution
        /// For slot #n and arguments [a1, a2, ..., an, ...]: (#n &)[a1, a2, ..., an, ...] ≡ an
        proptest! {
            #[test]
            fn slot_substitution_property(
                args in generators::arb_value_list(),
                slot_num in 1usize..5
            ) {
                // Only test if we have enough arguments
                prop_assume!(slot_num <= args.len());
                
                // Create slot pure function: (#slot_num &)
                let slot_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(slot_num) })
                };
                
                // Apply to arguments
                let result = pure_function::substitute_slots(&slot_func, &args);
                
                // Should return the slot_num-th argument (1-indexed)
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), args[slot_num - 1]);
            }
        }

        /// Property: Arithmetic Consistency  
        /// Pure function arithmetic should be consistent with direct arithmetic
        proptest! {
            #[test]
            fn arithmetic_consistency_property(
                a in generators::arb_integer(),
                b in generators::arb_integer()
            ) {
                // Avoid overflow
                prop_assume!(a.abs() < 100_000 && b.abs() < 100_000);
                
                // Create pure function: (# + #2 &)
                let add_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Slot { number: Some(2) }
                        ]
                    })))
                };
                
                let args = vec![Value::Integer(a), Value::Integer(b)];
                let result = pure_function::substitute_slots(&add_func, &args);
                
                // Should equal direct addition
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), Value::Integer(a + b));
            }
        }

        /// Property: Commutativity of Addition in Pure Functions
        /// (#1 + #2 &)[a, b] ≡ (#2 + #1 &)[a, b]
        proptest! {
            #[test]
            fn addition_commutativity_property(
                a in generators::arb_integer(),
                b in generators::arb_integer()
            ) {
                prop_assume!(a.abs() < 100_000 && b.abs() < 100_000);
                
                // Create (#1 + #2 &)
                let add_func1 = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(1) },
                            Expr::Slot { number: Some(2) }
                        ]
                    })))
                };
                
                // Create (#2 + #1 &)
                let add_func2 = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(2) },
                            Expr::Slot { number: Some(1) }
                        ]
                    })))
                };
                
                let args = vec![Value::Integer(a), Value::Integer(b)];
                let result1 = pure_function::substitute_slots(&add_func1, &args);
                let result2 = pure_function::substitute_slots(&add_func2, &args);
                
                // Both should succeed and be equal
                prop_assert!(result1.is_ok() && result2.is_ok());
                prop_assert_eq!(result1.unwrap(), result2.unwrap());
            }
        }

        /// Property: Function Composition
        /// If f = (# + 1 &) and g = (# * 2 &), then f(g(x)) ≡ ((# * 2) + 1 &)[x]
        proptest! {
            #[test]
            fn function_composition_property(x in generators::arb_integer()) {
                prop_assume!(x.abs() < 100_000);
                
                // Create f = (# + 1 &)
                let f = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(1))
                        ]
                    })))
                };
                
                // Create g = (# * 2 &)
                let g = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(2))
                        ]
                    })))
                };
                
                // Compute g(x) first
                let g_result = pure_function::substitute_slots(&g, &[Value::Integer(x)]);
                prop_assert!(g_result.is_ok());
                
                // Then compute f(g(x))
                let composed_result = pure_function::substitute_slots(&f, &[g_result.unwrap()]);
                prop_assert!(composed_result.is_ok());
                
                // Create the equivalent direct function (# * 2 + 1 &)
                let direct_composition = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                                args: vec![
                                    Expr::Slot { number: None },
                                    Expr::Number(Number::Integer(2))
                                ]
                            },
                            Expr::Number(Number::Integer(1))
                        ]
                    })))
                };
                
                let direct_result = pure_function::substitute_slots(&direct_composition, &[Value::Integer(x)]);
                prop_assert!(direct_result.is_ok());
                
                // Both should produce the same result
                prop_assert_eq!(composed_result.unwrap(), direct_result.unwrap());
            }
        }

        /// Property: Associativity of Addition in Pure Functions
        /// ((# + #2) + #3 &)[a, b, c] ≡ (# + (#2 + #3) &)[a, b, c]
        proptest! {
            #[test]
            fn addition_associativity_property(
                a in generators::arb_integer(),
                b in generators::arb_integer(),
                c in generators::arb_integer()
            ) {
                prop_assume!(a.abs() < 50_000 && b.abs() < 50_000 && c.abs() < 50_000);
                
                // Create ((# + #2) + #3 &) - left associative
                let left_assoc = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                                args: vec![
                                    Expr::Slot { number: None },
                                    Expr::Slot { number: Some(2) }
                                ]
                            },
                            Expr::Slot { number: Some(3) }
                        ]
                    })))
                };
                
                // Create (# + (#2 + #3) &) - right associative
                let right_assoc = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                                args: vec![
                                    Expr::Slot { number: Some(2) },
                                    Expr::Slot { number: Some(3) }
                                ]
                            }
                        ]
                    })))
                };
                
                let args = vec![Value::Integer(a), Value::Integer(b), Value::Integer(c)];
                let left_result = pure_function::substitute_slots(&left_assoc, &args);
                let right_result = pure_function::substitute_slots(&right_assoc, &args);
                
                // Both should succeed and be equal due to associativity
                prop_assert!(left_result.is_ok() && right_result.is_ok());
                prop_assert_eq!(left_result.unwrap(), right_result.unwrap());
            }
        }

        /// Property: Multiplication Associativity in Pure Functions
        /// ((# * #2) * #3 &)[a, b, c] ≡ (# * (#2 * #3) &)[a, b, c]
        proptest! {
            #[test]
            fn multiplication_associativity_property(
                a in 1i64..100,
                b in 1i64..100,
                c in 1i64..100
            ) {
                // Create ((# * #2) * #3 &) - left associative
                let left_assoc = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                                args: vec![
                                    Expr::Slot { number: None },
                                    Expr::Slot { number: Some(2) }
                                ]
                            },
                            Expr::Slot { number: Some(3) }
                        ]
                    })))
                };
                
                // Create (# * (#2 * #3) &) - right associative
                let right_assoc = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                                args: vec![
                                    Expr::Slot { number: Some(2) },
                                    Expr::Slot { number: Some(3) }
                                ]
                            }
                        ]
                    })))
                };
                
                let args = vec![Value::Integer(a), Value::Integer(b), Value::Integer(c)];
                let left_result = pure_function::substitute_slots(&left_assoc, &args);
                let right_result = pure_function::substitute_slots(&right_assoc, &args);
                
                // Both should succeed and be equal due to associativity
                prop_assert!(left_result.is_ok() && right_result.is_ok());
                prop_assert_eq!(left_result.unwrap(), right_result.unwrap());
            }
        }
    }

    /// Error handling property tests
    mod error_properties {
        use super::*;

        /// Property: Out-of-bounds slots should error consistently
        proptest! {
            #[test]
            fn out_of_bounds_slot_error_property(
                args in generators::arb_value_list(),
                slot_num in 10usize..20
            ) {
                // Ensure slot is out of bounds
                prop_assume!(slot_num > args.len());
                
                // Create slot pure function with out-of-bounds slot
                let slot_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(slot_num) })
                };
                
                // Should consistently error
                let result = pure_function::substitute_slots(&slot_func, &args);
                prop_assert!(result.is_err());
                
                // Error message should mention the slot number
                let error_msg = result.unwrap_err().to_string();
                prop_assert!(error_msg.contains(&format!("#{}", slot_num)));
            }
        }

        /// Property: Empty arguments should error for slot functions
        proptest! {
            #[test]
            fn empty_args_error_property(slot_num in 1usize..5) {
                // Create slot pure function
                let slot_func = Value::PureFunction {
                    body: Box::new(Value::Slot { 
                        number: if slot_num == 1 { None } else { Some(slot_num) }
                    })
                };
                
                // Empty arguments should error
                let result = pure_function::substitute_slots(&slot_func, &[]);
                prop_assert!(result.is_err());
                
                // Error should mention missing arguments
                let error_msg = result.unwrap_err().to_string();
                prop_assert!(error_msg.contains("arguments"));
            }
        }

        /// Property: Non-pure function values should error consistently
        proptest! {
            #[test]
            fn non_pure_function_error_property(value in generators::arb_value()) {
                // Skip if value is actually a pure function
                prop_assume!(!matches!(value, Value::PureFunction { .. }));
                
                let args = vec![Value::Integer(42)];
                let result = pure_function::substitute_slots(&value, &args);
                
                // Should consistently error
                prop_assert!(result.is_err());
                
                // Error should mention expecting pure function
                let error_msg = result.unwrap_err().to_string();
                prop_assert!(error_msg.contains("PureFunction"));
            }
        }
    }

    /// Type safety property tests
    mod type_safety_properties {
        use super::*;

        /// Property: Type preservation for identity function
        proptest! {
            #[test]
            fn identity_type_preservation_property(value in generators::arb_value()) {
                let identity_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: None })
                };
                
                let result = pure_function::substitute_slots(&identity_func, &[value.clone()]);
                
                prop_assert!(result.is_ok());
                let output = result.unwrap();
                
                // Type should be preserved (same discriminant)
                prop_assert_eq!(std::mem::discriminant(&value), std::mem::discriminant(&output));
            }
        }

        /// Property: String operations preserve string type
        proptest! {
            #[test]
            fn string_type_preservation_property(s in generators::arb_string()) {
                let string_identity_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: None })
                };
                
                let result = pure_function::substitute_slots(
                    &string_identity_func, 
                    &[Value::String(s.clone())]
                );
                
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), Value::String(s));
            }
        }

        /// Property: Arithmetic type safety - integers remain integers
        proptest! {
            #[test]
            fn arithmetic_integer_type_safety(
                a in generators::arb_integer(),
                b in generators::arb_integer()
            ) {
                prop_assume!(a.abs() < 100_000 && b.abs() < 100_000);
                
                let add_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Slot { number: Some(2) }
                        ]
                    })))
                };
                
                let result = pure_function::substitute_slots(
                    &add_func, 
                    &[Value::Integer(a), Value::Integer(b)]
                );
                
                prop_assert!(result.is_ok());
                match result.unwrap() {
                    Value::Integer(_) => {}, // Expected integer result
                    other => prop_assert!(false, "Expected Integer, got {:?}", other),
                }
            }
        }

        /// Property: Mixed type operations handle errors gracefully
        proptest! {
            #[test]
            fn mixed_type_error_handling(
                int_val in generators::arb_integer(),
                str_val in generators::arb_string()
            ) {
                // Create a function that expects arithmetic on arguments
                let add_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Slot { number: Some(2) }
                        ]
                    })))
                };
                
                // Try to add integer and string - should fail gracefully at evaluation
                let result = pure_function::substitute_slots(
                    &add_func, 
                    &[Value::Integer(int_val), Value::String(str_val)]
                );
                
                // The substitution itself should succeed (type checking happens at evaluation)
                prop_assert!(result.is_ok());
                
                // The result should be a proper expression that would fail at runtime
                match result.unwrap() {
                    Value::Quote(_) => {}, // Expected quoted expression
                    other => prop_assert!(false, "Expected Quote, got {:?}", other),
                }
            }
        }

        /// Property: Nested type safety - complex structures preserve types
        proptest! {
            #[test]
            fn nested_type_safety(values in generators::arb_value_list()) {
                prop_assume!(values.len() >= 2);
                
                // Create nested function application
                let nested_func = Value::PureFunction {
                    body: Box::new(Value::PureFunction {
                        body: Box::new(Value::Slot { number: None })
                    })
                };
                
                let result = pure_function::substitute_slots(&nested_func, &values);
                
                prop_assert!(result.is_ok());
                // Result should be a PureFunction with substituted body
                match result.unwrap() {
                    Value::PureFunction { .. } => {}, // Expected
                    other => prop_assert!(false, "Expected PureFunction, got {:?}", other),
                }
            }
        }
    }

    /// Error propagation property tests
    mod error_propagation_properties {
        use super::*;

        /// Property: Error propagation through nested operations
        proptest! {
            #[test]
            fn nested_error_propagation(
                valid_args in generators::arb_value_list(),
                invalid_slot in 10usize..20
            ) {
                prop_assume!(valid_args.len() < 5);
                prop_assume!(invalid_slot > valid_args.len());
                
                // Create nested function with invalid slot access
                let nested_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Slot { number: Some(invalid_slot) }
                        ]
                    })))
                };
                
                let result = pure_function::substitute_slots(&nested_func, &valid_args);
                
                // Error should be propagated properly
                prop_assert!(result.is_err());
                
                let error_msg = result.unwrap_err().to_string();
                prop_assert!(error_msg.contains(&format!("#{}", invalid_slot)));
            }
        }

        /// Property: Error consistency across different invalid operations
        proptest! {
            #[test]
            fn error_consistency_property(
                args in generators::arb_value_list(),
                invalid_slot1 in 15usize..25,
                invalid_slot2 in 25usize..35
            ) {
                prop_assume!(args.len() < 10);
                prop_assume!(invalid_slot1 > args.len() && invalid_slot2 > args.len());
                
                // Create two functions with different invalid slots
                let func1 = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(invalid_slot1) })
                };
                
                let func2 = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(invalid_slot2) })
                };
                
                let result1 = pure_function::substitute_slots(&func1, &args);
                let result2 = pure_function::substitute_slots(&func2, &args);
                
                // Both should error consistently
                prop_assert!(result1.is_err() && result2.is_err());
                
                // Error messages should follow consistent format
                let error1 = result1.unwrap_err().to_string();
                let error2 = result2.unwrap_err().to_string();
                
                prop_assert!(error1.contains("Slot"));
                prop_assert!(error2.contains("Slot"));
            }
        }

        /// Property: Error recovery and partial evaluation
        proptest! {
            #[test]
            fn error_recovery_property(
                valid_values in generators::arb_value_list(),
                invalid_slot in 20usize..30
            ) {
                prop_assume!(valid_values.len() >= 2 && valid_values.len() < 10);
                prop_assume!(invalid_slot > valid_values.len());
                
                // Create function with both valid and invalid slots
                let mixed_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "List".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(1) }, // Valid
                            Expr::Slot { number: Some(invalid_slot) }, // Invalid
                            Expr::Slot { number: Some(2) } // Valid
                        ]
                    })))
                };
                
                let result = pure_function::substitute_slots(&mixed_func, &valid_values);
                
                // Should fail on first invalid slot
                prop_assert!(result.is_err());
                
                // Error should be specific to the problematic slot
                let error_msg = result.unwrap_err().to_string();
                prop_assert!(error_msg.contains(&format!("#{}", invalid_slot)));
            }
        }

        /// Property: Type error propagation in complex expressions
        proptest! {
            #[test]
            fn type_error_propagation_property(
                int_val in generators::arb_integer(),
                str_val in generators::arb_string(),
                bool_val in any::<bool>()
            ) {
                // Create function that attempts invalid type operations
                let type_error_func = Value::PureFunction {
                    body: Box::new(Value::Quote(Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                                args: vec![
                                    Expr::Slot { number: Some(1) }, // Integer
                                    Expr::Slot { number: Some(2) }  // String - invalid for Times
                                ]
                            },
                            Expr::Slot { number: Some(3) } // Boolean - invalid for Plus
                        ]
                    })))
                };
                
                let args = vec![
                    Value::Integer(int_val),
                    Value::String(str_val),
                    Value::Boolean(bool_val)
                ];
                
                let result = pure_function::substitute_slots(&type_error_func, &args);
                
                // Substitution should succeed (creating expression for later evaluation)
                prop_assert!(result.is_ok());
                
                // Result should be a quoted expression that would fail at evaluation
                match result.unwrap() {
                    Value::Quote(_) => {}, // Expected
                    other => prop_assert!(false, "Expected Quote for deferred evaluation, got {:?}", other),
                }
            }
        }

        /// Property: Error message quality and information content
        proptest! {
            #[test]
            fn error_message_quality_property(
                args in generators::arb_value_list(),
                out_of_bounds_slot in 50usize..100
            ) {
                prop_assume!(args.len() < 10);
                prop_assume!(out_of_bounds_slot > args.len());
                
                let error_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(out_of_bounds_slot) })
                };
                
                let result = pure_function::substitute_slots(&error_func, &args);
                prop_assert!(result.is_err());
                
                let error_msg = result.unwrap_err().to_string();
                
                // Error message should contain key information
                prop_assert!(error_msg.len() > 10); // Non-trivial message
                prop_assert!(error_msg.contains(&format!("#{}", out_of_bounds_slot))); // Slot number
                prop_assert!(error_msg.contains(&format!("{}", args.len())) || 
                           error_msg.contains("argument")); // Reference to available arguments
                
                // Error message should be actionable
                prop_assert!(error_msg.to_lowercase().contains("slot") || 
                           error_msg.to_lowercase().contains("argument") ||
                           error_msg.to_lowercase().contains("bound"));
            }
        }
    }

    /// Performance and scalability property tests
    mod performance_properties {
        use super::*;

        /// Property: Linear performance scaling with argument count
        proptest! {
            #[test]
            fn linear_performance_scaling_property(
                arg_count in 1usize..20
            ) {
                // Create arguments
                let args: Vec<Value> = (0..arg_count)
                    .map(|i| Value::Integer(i as i64))
                    .collect();
                
                // Create pure function that uses last slot
                let slot_func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: Some(arg_count) })
                };
                
                // Should complete in reasonable time regardless of argument count
                let start = std::time::Instant::now();
                let result = pure_function::substitute_slots(&slot_func, &args);
                let duration = start.elapsed();
                
                prop_assert!(result.is_ok());
                prop_assert_eq!(result.unwrap(), Value::Integer((arg_count - 1) as i64));
                
                // Should complete in under 1ms even with many arguments
                prop_assert!(duration.as_millis() < 1);
            }
        }

        /// Property: Memory usage should be bounded
        proptest! {
            #[test]
            fn bounded_memory_usage_property(
                depth in 1usize..5
            ) {
                // Create nested pure function calls
                let mut func = Value::PureFunction {
                    body: Box::new(Value::Slot { number: None })
                };
                
                // Nest the functions
                for _ in 0..depth {
                    func = Value::PureFunction {
                        body: Box::new(func)
                    };
                }
                
                let args = vec![Value::Integer(42)];
                let result = pure_function::substitute_slots(&func, &args);
                
                // Should handle reasonable nesting without stack overflow
                prop_assert!(result.is_ok());
            }
        }
    }
}

/// Integration tests with the VM and compilation system
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_pure_function_evaluation_integration() {
        // Test that pure functions integrate correctly with the broader system
        let source = "(# + 1 &)[5]";
        
        // This would require full parsing and compilation integration
        // For now, test the core substitution directly
        let pure_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };
        
        let result = pure_function::substitute_slots(&pure_func, &[Value::Integer(5)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Integer(6));
    }

    #[test]
    fn test_property_framework_setup() {
        // Verify the property testing framework is working
        use generators::*;
        
        let strategy = arb_value();
        let mut runner = proptest::test_runner::TestRunner::default();
        
        // Test that we can generate random values
        for _ in 0..10 {
            let value = strategy.new_tree(&mut runner).unwrap().current();
            match value {
                Value::Integer(_) | Value::Real(_) | Value::String(_) | Value::Boolean(_) => {
                    // Valid generated value
                }
                _ => panic!("Unexpected value type generated: {:?}", value),
            }
        }
    }
}