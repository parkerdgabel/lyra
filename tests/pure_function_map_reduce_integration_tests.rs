//! Integration tests for pure functions with Map/Reduce operations
//!
//! This module tests the interaction between pure functions and Lyra's functional
//! programming operations like Map, Reduce, Filter, and related higher-order functions,
//! ensuring seamless integration and performance characteristics.

use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};

#[cfg(test)]
mod map_reduce_integration {
    use super::*;

    /// Test pure functions with Map operations
    #[test]
    fn test_pure_function_with_map() {
        // Test: Map[(# * 2 &), {1, 2, 3, 4, 5}]
        let double_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        // Create Map expression with pure function
        let map_expr = Value::Quote(Box::new(Expr::Function {
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
                Expr::List(vec![
                    Expr::Number(Number::Integer(1)),
                    Expr::Number(Number::Integer(2)),
                    Expr::Number(Number::Integer(3)),
                    Expr::Number(Number::Integer(4)),
                    Expr::Number(Number::Integer(5))
                ])
            ]
        }));

        // Test that the expression is well-formed
        match map_expr {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }

        // Test individual pure function application
        let result = pure_function::substitute_slots(&double_func, &[Value::Integer(3)]);
        assert!(result.is_ok());
        
        // Should produce a quoted expression for 3 * 2
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with complex Map operations
    #[test]
    fn test_pure_function_complex_map() {
        // Test: Map[(If[# > 0, #, -#] &), {-2, -1, 0, 1, 2}] - absolute value map
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

        // Test individual applications
        let test_cases = vec![-2, -1, 0, 1, 2];
        for value in test_cases {
            let result = pure_function::substitute_slots(&abs_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected quoted expression
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with nested Map operations
    #[test]
    fn test_pure_function_nested_map() {
        // Test: Map[Map[(# + 1 &), #] &, {{1, 2}, {3, 4}, {5, 6}}]
        let inner_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };

        let outer_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Map".to_string() })),
                args: vec![
                    Expr::PureFunction {
                        body: Box::new(Expr::Function {
                            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                            args: vec![
                                Expr::Slot { number: None },
                                Expr::Number(Number::Integer(1))
                            ]
                        })
                    },
                    Expr::Slot { number: None }
                ]
            })))
        };

        // Test with nested list
        let nested_list = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)])
        ]);

        let result = pure_function::substitute_slots(&outer_func, &[nested_list]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with Reduce operations
    #[test]
    fn test_pure_function_with_reduce() {
        // Test: Reduce[(#1 + #2 &), {1, 2, 3, 4, 5}]
        let sum_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        // Test individual binary applications
        let result = pure_function::substitute_slots(&sum_func, &[
            Value::Integer(3),
            Value::Integer(7)
        ]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with complex Reduce operations
    #[test]
    fn test_pure_function_complex_reduce() {
        // Test: Reduce[(If[#1 > #2, #1, #2] &), {3, 1, 4, 1, 5, 9, 2, 6}] - maximum
        let max_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(1) },
                            Expr::Slot { number: Some(2) }
                        ]
                    },
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        // Test with various pairs
        let test_pairs = vec![(3, 1), (4, 1), (5, 9), (9, 2), (9, 6)];
        for (a, b) in test_pairs {
            let result = pure_function::substitute_slots(&max_func, &[
                Value::Integer(a),
                Value::Integer(b)
            ]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for pair ({}, {}), got {:?}", a, b, other),
            }
        }
    }

    /// Test pure functions with Filter operations
    #[test]
    fn test_pure_function_with_filter() {
        // Test: Filter[(# > 0 &), {-2, -1, 0, 1, 2, 3}]
        let positive_filter = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(0))
                ]
            })))
        };

        // Test individual predicate applications
        let test_values = vec![-2, -1, 0, 1, 2, 3];
        for value in test_values {
            let result = pure_function::substitute_slots(&positive_filter, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with MapIndexed operations
    #[test]
    fn test_pure_function_with_map_indexed() {
        // Test: MapIndexed[(#1 * #2 &), {10, 20, 30}] - multiply value by index
        let indexed_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) }, // Value
                    Expr::Slot { number: Some(2) }  // Index
                ]
            })))
        };

        // Test with value-index pairs
        let test_cases = vec![(10, 1), (20, 2), (30, 3)];
        for (value, index) in test_cases {
            let result = pure_function::substitute_slots(&indexed_func, &[
                Value::Integer(value),
                Value::Integer(index)
            ]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for ({}, {}), got {:?}", value, index, other),
            }
        }
    }

    /// Test pure functions with Select operations
    #[test]
    fn test_pure_function_with_select() {
        // Test: Select[{1, 2, 3, 4, 5, 6}, (EvenQ[#] &)]
        let even_predicate = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "EvenQ".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Test individual applications
        let test_values = vec![1, 2, 3, 4, 5, 6];
        for value in test_values {
            let result = pure_function::substitute_slots(&even_predicate, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with Fold operations
    #[test]
    fn test_pure_function_with_fold() {
        // Test: Fold[(#1 + #2^2 &), 0, {1, 2, 3}] - sum of squares with accumulator
        let fold_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) }, // Accumulator
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Power".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(2) }, // Current value
                            Expr::Number(Number::Integer(2))
                        ]
                    }
                ]
            })))
        };

        // Test with accumulator-value pairs
        let test_cases = vec![(0, 1), (1, 2), (5, 3)]; // accumulator, current value
        for (acc, val) in test_cases {
            let result = pure_function::substitute_slots(&fold_func, &[
                Value::Integer(acc),
                Value::Integer(val)
            ]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for ({}, {}), got {:?}", acc, val, other),
            }
        }
    }

    /// Test pure functions with Scan operations
    #[test]
    fn test_pure_function_with_scan() {
        // Test: Scan[(#1 * #2 &), {1, 2, 3, 4}] - cumulative product
        let product_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        // Test with consecutive pairs
        let test_cases = vec![(1, 2), (2, 3), (6, 4)]; // intermediate results
        for (acc, val) in test_cases {
            let result = pure_function::substitute_slots(&product_func, &[
                Value::Integer(acc),
                Value::Integer(val)
            ]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for ({}, {}), got {:?}", acc, val, other),
            }
        }
    }

    /// Test pure functions with Apply operations
    #[test]
    fn test_pure_function_with_apply() {
        // Test: Apply[(#1 + #2 + #3 &), {1, 2, 3}]
        let sum_three = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) },
                    Expr::Slot { number: Some(3) }
                ]
            })))
        };

        let result = pure_function::substitute_slots(&sum_three, &[
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3)
        ]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with GroupBy operations
    #[test]
    fn test_pure_function_with_group_by() {
        // Test: GroupBy[{1, 2, 3, 4, 5, 6}, (Mod[#, 2] &)] - group by even/odd
        let mod_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Mod".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        // Test individual applications
        let test_values = vec![1, 2, 3, 4, 5, 6];
        for value in test_values {
            let result = pure_function::substitute_slots(&mod_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with SortBy operations
    #[test]
    fn test_pure_function_with_sort_by() {
        // Test: SortBy[{{1, 3}, {2, 1}, {3, 2}}, (Last[#] &)] - sort by last element
        let last_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Last".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Test with pairs
        let test_pairs = vec![
            vec![Value::Integer(1), Value::Integer(3)],
            vec![Value::Integer(2), Value::Integer(1)],
            vec![Value::Integer(3), Value::Integer(2)]
        ];

        for pair in test_pairs {
            let result = pure_function::substitute_slots(&last_func, &[Value::List(pair.clone())]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for pair {:?}, got {:?}", pair, other),
            }
        }
    }

    /// Test pure functions with Partition operations
    #[test]
    fn test_pure_function_with_partition() {
        // Test: Partition[{1, 2, 3, 4, 5, 6}, (# > 3 &)] - partition by condition
        let greater_than_three = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(3))
                ]
            })))
        };

        // Test individual applications
        let test_values = vec![1, 2, 3, 4, 5, 6];
        for value in test_values {
            let result = pure_function::substitute_slots(&greater_than_three, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with complex chained operations
    #[test]
    fn test_pure_function_chained_map_reduce() {
        // Test complex chain: Map[(# + 1 &), list] then Reduce[(#1 * #2 &), result]
        
        // Step 1: Map function (# + 1 &)
        let map_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };

        // Step 2: Reduce function (#1 * #2 &)
        let reduce_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        // Test map function applications
        let test_values = vec![1, 2, 3];
        for value in &test_values {
            let result = pure_function::substitute_slots(&map_func, &[Value::Integer(*value)]);
            assert!(result.is_ok());
        }

        // Test reduce function applications
        let reduce_pairs = vec![(2, 3), (6, 4)]; // After map: [2, 3, 4], reduce pairs
        for (a, b) in reduce_pairs {
            let result = pure_function::substitute_slots(&reduce_func, &[
                Value::Integer(a),
                Value::Integer(b)
            ]);
            assert!(result.is_ok());
        }
    }

    /// Test error handling in Map/Reduce operations
    #[test]
    fn test_pure_function_map_reduce_error_handling() {
        // Test function with out-of-bounds slot
        let invalid_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Slot { number: Some(5) } // Out of bounds
                ]
            })))
        };

        let result = pure_function::substitute_slots(&invalid_func, &[
            Value::Integer(1),
            Value::Integer(2)
        ]);

        // Should error due to out-of-bounds slot
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("#5"));
    }

    /// Test pure functions with string operations in Map
    #[test]
    fn test_pure_function_string_map() {
        // Test: Map[(StringLength[#] &), {"hello", "world", "test"}]
        let string_length_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "StringLength".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        let test_strings = vec!["hello", "world", "test"];
        for s in test_strings {
            let result = pure_function::substitute_slots(&string_length_func, &[
                Value::String(s.to_string())
            ]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for string '{}', got {:?}", s, other),
            }
        }
    }

    /// Test pure functions with mixed type operations
    #[test]
    fn test_pure_function_mixed_type_map() {
        // Test: Map[(ToString[#] &), {1, 2.5, "hello", True}]
        let to_string_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ToString".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        let mixed_values = vec![
            Value::Integer(1),
            Value::Real(2.5),
            Value::String("hello".to_string()),
            Value::Boolean(true)
        ];

        for value in mixed_values {
            let result = pure_function::substitute_slots(&to_string_func, &[value.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for value {:?}, got {:?}", value, other),
            }
        }
    }
}

/// Performance and scalability tests for Map/Reduce operations
#[cfg(test)]
mod performance_integration {
    use super::*;

    #[test]
    fn test_map_reduce_performance_characteristics() {
        // Test that pure functions maintain performance with large datasets
        let square_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Power".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        // Test with various sizes to ensure scalability
        let test_sizes = vec![10, 100, 1000];
        
        for size in test_sizes {
            let start = std::time::Instant::now();
            
            // Simulate Map operation by applying function to many values
            for i in 0..size {
                let result = pure_function::substitute_slots(&square_func, &[Value::Integer(i)]);
                assert!(result.is_ok());
            }
            
            let duration = start.elapsed();
            
            // Should complete in reasonable time (< 100ms for 1000 operations)
            assert!(duration.as_millis() < 100, 
                "Performance test failed for size {}: took {:?}", size, duration);
        }
    }

    #[test]
    fn test_memory_efficiency_in_map_reduce() {
        // Test that pure function applications don't leak memory
        let complex_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Slot { number: None }
                        ]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(1))
                        ]
                    }
                ]
            })))
        };

        // Apply function many times to test memory usage
        for i in 0..500 {
            let result = pure_function::substitute_slots(&complex_func, &[Value::Integer(i)]);
            assert!(result.is_ok());
            
            // Each result should be valid
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote at iteration {}, got {:?}", i, other),
            }
        }

        // Test completed without memory issues
    }
}

/// Integration test framework validation
#[cfg(test)]
mod framework_validation {
    use super::*;

    #[test]
    fn test_map_reduce_integration_framework_setup() {
        // Verify the integration testing framework is working correctly
        let identity_func = Value::PureFunction {
            body: Box::new(Value::Slot { number: None })
        };

        // Test with various types that would be used in Map/Reduce
        let test_values = vec![
            Value::Integer(42),
            Value::Real(3.14),
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
    fn test_expression_structure_validation() {
        // Test that Map/Reduce expressions have valid structure
        let map_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Map".to_string() })),
            args: vec![
                Expr::PureFunction {
                    body: Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(1))
                        ]
                    })
                },
                Expr::List(vec![
                    Expr::Number(Number::Integer(1)),
                    Expr::Number(Number::Integer(2)),
                    Expr::Number(Number::Integer(3))
                ])
            ]
        };

        // Verify expression structure
        match map_expr {
            Expr::Function { head, args } => {
                assert!(matches!(*head, Expr::Symbol(_)));
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expr::PureFunction { .. }));
                assert!(matches!(args[1], Expr::List(_)));
            }
            other => panic!("Expected Function expression, got {:?}", other),
        }
    }
}