//! Integration tests for pure functions with pattern matching system
//!
//! This module tests the interaction between pure functions and Lyra's pattern
//! matching capabilities, ensuring seamless integration and correct behavior
//! across the entire language system.

use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};

#[cfg(test)]
mod pattern_matching_integration {
    use super::*;

    /// Test pure functions with simple pattern matching
    #[test]
    fn test_pure_function_simple_pattern_match() {
        // Test: (# > 0 &) applied to various values
        let predicate_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(0))
                ]
            })))
        };

        // Test positive number
        let result = pure_function::substitute_slots(&predicate_func, &[Value::Integer(5)]);
        assert!(result.is_ok());
        
        // Test negative number
        let result = pure_function::substitute_slots(&predicate_func, &[Value::Integer(-3)]);
        assert!(result.is_ok());
        
        // Test zero
        let result = pure_function::substitute_slots(&predicate_func, &[Value::Integer(0)]);
        assert!(result.is_ok());
    }

    /// Test pure functions with list pattern matching
    #[test]
    fn test_pure_function_list_patterns() {
        // Test: (Head[#] &) - extract first element
        let head_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Head".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Test with list
        let test_list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3)
        ]);

        let result = pure_function::substitute_slots(&head_func, &[test_list]);
        assert!(result.is_ok());
        
        // Result should be quoted expression for Head[{1, 2, 3}]
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with nested pattern matching
    #[test]
    fn test_pure_function_nested_patterns() {
        // Test: (Length[Tail[#]] &) - length of tail
        let nested_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Length".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Tail".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_list = Value::List(vec![
            Value::String("first".to_string()),
            Value::String("second".to_string()),
            Value::String("third".to_string())
        ]);

        let result = pure_function::substitute_slots(&nested_func, &[test_list]);
        assert!(result.is_ok());
        
        // Should create proper nested expression
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected quoted expression
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with conditional patterns
    #[test]
    fn test_pure_function_conditional_patterns() {
        // Test: (If[# > 0, #, -#] &) - absolute value function
        let abs_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(0))
                        ]
                    },
                    Expr::Slot { number: None },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        // Test positive number
        let result_pos = pure_function::substitute_slots(&abs_func, &[Value::Integer(5)]);
        assert!(result_pos.is_ok());

        // Test negative number
        let result_neg = pure_function::substitute_slots(&abs_func, &[Value::Integer(-7)]);
        assert!(result_neg.is_ok());

        // Both should produce valid quoted expressions
        match (result_pos.unwrap(), result_neg.unwrap()) {
            (Value::Quote(_), Value::Quote(_)) => {}, // Expected
            (a, b) => panic!("Expected both Quote, got {:?} and {:?}", a, b),
        }
    }

    /// Test pure functions with multiple pattern arguments
    #[test]
    fn test_pure_function_multiple_pattern_args() {
        // Test: (Match[#1, #2] &) - pattern matching with two arguments
        let match_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Match".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        let pattern = Value::Symbol("x_".to_string());
        let expression = Value::Integer(42);

        let result = pure_function::substitute_slots(&match_func, &[expression, pattern]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with rule-based patterns
    #[test]
    fn test_pure_function_rule_patterns() {
        // Test: (ReplaceAll[#, x_ -> x + 1] &) - rule application
        let rule_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ReplaceAll".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Rule".to_string() })),
                        args: vec![
                            Expr::Symbol(Symbol { name: "x_".to_string() }),
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                                args: vec![
                                    Expr::Symbol(Symbol { name: "x".to_string() }),
                                    Expr::Number(Number::Integer(1))
                                ]
                            }
                        ]
                    }
                ]
            })))
        };

        let test_expr = Value::Integer(10);
        let result = pure_function::substitute_slots(&rule_func, &[test_expr]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with pattern guards
    #[test]
    fn test_pure_function_pattern_guards() {
        // Test: (Cases[#, {x_ /; x > 0 -> "positive", _ -> "non-positive"}] &)
        let cases_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Cases".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::List(vec![
                        Expr::Function {
                            head: Box::new(Expr::Symbol(Symbol { name: "Rule".to_string() })),
                            args: vec![
                                Expr::Function {
                                    head: Box::new(Expr::Symbol(Symbol { name: "Condition".to_string() })),
                                    args: vec![
                                        Expr::Symbol(Symbol { name: "x_".to_string() }),
                                        Expr::Function {
                                            head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                                            args: vec![
                                                Expr::Symbol(Symbol { name: "x".to_string() }),
                                                Expr::Number(Number::Integer(0))
                                            ]
                                        }
                                    ]
                                },
                                Expr::String("positive".to_string())
                            ]
                        },
                        Expr::Function {
                            head: Box::new(Expr::Symbol(Symbol { name: "Rule".to_string() })),
                            args: vec![
                                Expr::Symbol(Symbol { name: "_".to_string() }),
                                Expr::String("non-positive".to_string())
                            ]
                        }
                    ])
                ]
            })))
        };

        // Test positive number
        let result_pos = pure_function::substitute_slots(&cases_func, &[Value::Integer(5)]);
        assert!(result_pos.is_ok());

        // Test negative number
        let result_neg = pure_function::substitute_slots(&cases_func, &[Value::Integer(-2)]);
        assert!(result_neg.is_ok());

        // Both should produce valid quoted expressions
        match (result_pos.unwrap(), result_neg.unwrap()) {
            (Value::Quote(_), Value::Quote(_)) => {}, // Expected
            (a, b) => panic!("Expected both Quote, got {:?} and {:?}", a, b),
        }
    }

    /// Test pure functions with recursive pattern matching
    #[test]
    fn test_pure_function_recursive_patterns() {
        // Test factorial-like pattern: (If[# <= 1, 1, # * factorial[# - 1]] &)
        let factorial_like = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "LessEqual".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(1))
                        ]
                    },
                    Expr::Number(Number::Integer(1)),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "factorial".to_string() })),
                                args: vec![
                                    Expr::Function {
                                        head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                                        args: vec![
                                            Expr::Slot { number: None },
                                            Expr::Number(Number::Integer(1))
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            })))
        };

        let result = pure_function::substitute_slots(&factorial_like, &[Value::Integer(3)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with complex data structure patterns
    #[test]
    fn test_pure_function_complex_structure_patterns() {
        // Test: (First[Cases[#, _Integer]] &) - extract first integer from list
        let extract_int_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "First".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Cases".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Symbol(Symbol { name: "_Integer".to_string() })
                        ]
                    }
                ]
            })))
        };

        let mixed_list = Value::List(vec![
            Value::String("hello".to_string()),
            Value::Integer(42),
            Value::Real(3.14),
            Value::Integer(7)
        ]);

        let result = pure_function::substitute_slots(&extract_int_func, &[mixed_list]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with pattern replacement chains
    #[test]
    fn test_pure_function_pattern_replacement_chains() {
        // Test: (# /. {x_ :> x + 1} /. {y_ :> y * 2} &) - chained replacements
        let chain_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ReplaceAll".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ReplaceAll".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::List(vec![
                                Expr::Function {
                                    head: Box::new(Expr::Symbol(Symbol { name: "RuleDelayed".to_string() })),
                                    args: vec![
                                        Expr::Symbol(Symbol { name: "x_".to_string() }),
                                        Expr::Function {
                                            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                                            args: vec![
                                                Expr::Symbol(Symbol { name: "x".to_string() }),
                                                Expr::Number(Number::Integer(1))
                                            ]
                                        }
                                    ]
                                }
                            ])
                        ]
                    },
                    Expr::List(vec![
                        Expr::Function {
                            head: Box::new(Expr::Symbol(Symbol { name: "RuleDelayed".to_string() })),
                            args: vec![
                                Expr::Symbol(Symbol { name: "y_".to_string() }),
                                Expr::Function {
                                    head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                                    args: vec![
                                        Expr::Symbol(Symbol { name: "y".to_string() }),
                                        Expr::Number(Number::Integer(2))
                                    ]
                                }
                            ]
                        }
                    ])
                ]
            })))
        };

        let test_value = Value::Integer(5);
        let result = pure_function::substitute_slots(&chain_func, &[test_value]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure function error handling with invalid patterns
    #[test]
    fn test_pure_function_invalid_pattern_handling() {
        // Test pattern with invalid slot reference
        let invalid_pattern_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Match".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(5) } // Out of bounds
                ]
            })))
        };

        let result = pure_function::substitute_slots(&invalid_pattern_func, &[
            Value::Integer(42),
            Value::String("pattern".to_string())
        ]);

        // Should error due to out-of-bounds slot
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("#5"));
    }

    /// Test integration with Map function and patterns
    #[test]
    fn test_pure_function_map_pattern_integration() {
        // Test: Map[(# * 2 &), list] equivalent pattern
        let map_like_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Map".to_string() })),
                args: vec![
                    Expr::PureFunction {
                        body: Box::new(Expr::Function {
                            head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                            args: vec![
                                Expr::Slot { number: None },
                                Expr::Number(Number::Integer(2))
                            ]
                        })
                    },
                    Expr::Slot { number: None }
                ]
            })))
        };

        let test_list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3)
        ]);

        let result = pure_function::substitute_slots(&map_like_func, &[test_list]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pattern-based function composition
    #[test]
    fn test_pure_function_pattern_composition() {
        // Test: (f[g[#]] &) where f and g are pattern-based functions
        let composed_pattern_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "f".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "g".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_value = Value::Symbol("x".to_string());
        let result = pure_function::substitute_slots(&composed_pattern_func, &[test_value]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with pattern-based conditionals
    #[test]
    fn test_pure_function_pattern_conditionals() {
        // Test: (Which[IntegerQ[#], "integer", ListQ[#], "list", True, "other"] &)
        let which_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Which".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "IntegerQ".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::String("integer".to_string()),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ListQ".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::String("list".to_string()),
                    Expr::Boolean(true),
                    Expr::String("other".to_string())
                ]
            })))
        };

        // Test with integer
        let result_int = pure_function::substitute_slots(&which_func, &[Value::Integer(42)]);
        assert!(result_int.is_ok());

        // Test with list
        let test_list = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let result_list = pure_function::substitute_slots(&which_func, &[test_list]);
        assert!(result_list.is_ok());

        // Test with string
        let result_str = pure_function::substitute_slots(&which_func, &[Value::String("test".to_string())]);
        assert!(result_str.is_ok());

        // All should produce valid quoted expressions
        match (result_int.unwrap(), result_list.unwrap(), result_str.unwrap()) {
            (Value::Quote(_), Value::Quote(_), Value::Quote(_)) => {}, // Expected
            (a, b, c) => panic!("Expected all Quote, got {:?}, {:?}, {:?}", a, b, c),
        }
    }
}

/// Integration tests with VM evaluation
#[cfg(test)]
mod vm_integration {
    use super::*;

    #[test]
    fn test_pattern_integration_framework_setup() {
        // Verify the pattern integration testing framework is working
        let identity_func = Value::PureFunction {
            body: Box::new(Value::Slot { number: None })
        };

        let test_values = vec![
            Value::Integer(42),
            Value::String("test".to_string()),
            Value::Boolean(true),
            Value::List(vec![Value::Integer(1), Value::Integer(2)])
        ];

        for value in test_values {
            let result = pure_function::substitute_slots(&identity_func, &[value.clone()]);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), value);
        }
    }

    #[test]
    fn test_pattern_expression_validity() {
        // Test that pattern-based pure functions create valid expressions
        let pattern_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "MatchQ".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Symbol(Symbol { name: "_Integer".to_string() })
                ]
            })))
        };

        let result = pure_function::substitute_slots(&pattern_func, &[Value::Integer(42)]);
        assert!(result.is_ok());
        
        // Verify the resulting expression structure
        match result.unwrap() {
            Value::Quote(expr) => {
                match *expr {
                    Expr::Function { head, args } => {
                        // Verify function structure
                        assert!(matches!(*head, Expr::Symbol(_)));
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }
}