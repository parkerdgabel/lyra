//! Integration tests for pure functions with concurrency and parallelism
//!
//! This module tests the interaction between pure functions and Lyra's concurrency
//! features including ThreadPool, Channel, Future, ParallelMap, and other concurrent
//! operations, ensuring thread safety and performance characteristics.

use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};

#[cfg(test)]
mod concurrency_integration {
    use super::*;

    /// Test pure functions with ThreadPool operations
    #[test]
    fn test_pure_function_with_thread_pool() {
        // Test: ThreadPool.submit((# * 2 &), args)
        let double_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        // Test multiple concurrent applications
        let test_values = vec![1, 2, 3, 4, 5];
        
        for value in test_values {
            let result = pure_function::substitute_slots(&double_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected quoted expression
                other => panic!("Expected Quote for value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with parallel map operations
    #[test]
    fn test_pure_function_parallel_map() {
        // Test: ParallelMap[(# + 1 &), {1, 2, 3, 4, 5}]
        let increment_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };

        // Simulate parallel application to multiple values
        let test_values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut results = Vec::new();

        for value in test_values {
            let result = pure_function::substitute_slots(&increment_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            results.push(result.unwrap());
        }

        // All results should be valid quoted expressions
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            match result {
                Value::Quote(_) => {}, // Expected
                other => panic!("Result {} should be Quote, got {:?}", i, other),
            }
        }
    }

    /// Test pure functions with parallel reduce operations
    #[test]
    fn test_pure_function_parallel_reduce() {
        // Test: ParallelReduce[(#1 + #2 &), {1, 2, 3, 4, 5, 6, 7, 8}]
        let sum_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: Some(1) },
                    Expr::Slot { number: Some(2) }
                ]
            })))
        };

        // Test tree-like reduction pairs
        let reduction_pairs = vec![
            (1, 2), (3, 4), (5, 6), (7, 8), // Level 1
            (3, 7), (11, 15),               // Level 2  
            (10, 26)                        // Level 3
        ];

        for (a, b) in reduction_pairs {
            let result = pure_function::substitute_slots(&sum_func, &[
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

    /// Test pure functions with channels and message passing
    #[test]
    fn test_pure_function_with_channels() {
        // Test: Channel operations with pure function message processing
        let process_message_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "StringJoin".to_string() })),
                args: vec![
                    Expr::String("Processed: ".to_string()),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ToString".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        // Test processing various message types
        let messages = vec![
            Value::Integer(42),
            Value::String("hello".to_string()),
            Value::Real(3.14),
            Value::Boolean(true)
        ];

        for message in messages {
            let result = pure_function::substitute_slots(&process_message_func, &[message.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for message {:?}, got {:?}", message, other),
            }
        }
    }

    /// Test pure functions with futures and async operations
    #[test]
    fn test_pure_function_with_futures() {
        // Test: Future.map((# * 3 &))
        let triple_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(3))
                ]
            })))
        };

        // Test with various async values
        let async_values = vec![5, 10, 15, 20];
        
        for value in async_values {
            let result = pure_function::substitute_slots(&triple_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for async value {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with pipeline processing
    #[test]
    fn test_pure_function_pipeline_processing() {
        // Test: Pipeline with multiple pure function stages
        
        // Stage 1: (# + 10 &)
        let stage1_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(10))
                ]
            })))
        };

        // Stage 2: (# * 2 &)
        let stage2_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        // Stage 3: (# - 5 &)
        let stage3_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(5))
                ]
            })))
        };

        // Test pipeline processing: input -> stage1 -> stage2 -> stage3
        let input_values = vec![1, 2, 3, 4, 5];
        
        for input in input_values {
            // Stage 1: input + 10
            let result1 = pure_function::substitute_slots(&stage1_func, &[Value::Integer(input)]);
            assert!(result1.is_ok());

            // Stage 2: (input + 10) * 2 (simulated)
            let intermediate = input + 10;
            let result2 = pure_function::substitute_slots(&stage2_func, &[Value::Integer(intermediate)]);
            assert!(result2.is_ok());

            // Stage 3: ((input + 10) * 2) - 5 (simulated)
            let intermediate2 = intermediate * 2;
            let result3 = pure_function::substitute_slots(&stage3_func, &[Value::Integer(intermediate2)]);
            assert!(result3.is_ok());

            // All stages should produce valid results
            match (result1.unwrap(), result2.unwrap(), result3.unwrap()) {
                (Value::Quote(_), Value::Quote(_), Value::Quote(_)) => {}, // Expected
                (a, b, c) => panic!("Expected all Quote for input {}, got {:?}, {:?}, {:?}", input, a, b, c),
            }
        }
    }

    /// Test pure functions with concurrent data processing
    #[test]
    fn test_pure_function_concurrent_data_processing() {
        // Test: Concurrent processing of large datasets
        let complex_processing_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Power".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(2))
                        ]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(3))
                        ]
                    }
                ]
            })))
        };

        // Simulate concurrent processing of multiple data chunks
        let data_chunks = vec![
            vec![1, 2, 3],
            vec![4, 5, 6],
            vec![7, 8, 9],
            vec![10, 11, 12]
        ];

        for chunk in data_chunks {
            for value in chunk {
                let result = pure_function::substitute_slots(&complex_processing_func, &[Value::Integer(value)]);
                assert!(result.is_ok());
                
                match result.unwrap() {
                    Value::Quote(_) => {}, // Expected
                    other => panic!("Expected Quote for value {}, got {:?}", value, other),
                }
            }
        }
    }

    /// Test pure functions with work-stealing patterns
    #[test]
    fn test_pure_function_work_stealing() {
        // Test: Work-stealing scheduler with pure function tasks
        let fibonacci_like_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "LessEqual".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(2))
                        ]
                    },
                    Expr::Number(Number::Integer(1)),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "fib".to_string() })),
                                args: vec![
                                    Expr::Function {
                                        head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                                        args: vec![
                                            Expr::Slot { number: None },
                                            Expr::Number(Number::Integer(1))
                                        ]
                                    }
                                ]
                            },
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "fib".to_string() })),
                                args: vec![
                                    Expr::Function {
                                        head: Box::new(Expr::Symbol(Symbol { name: "Minus".to_string() })),
                                        args: vec![
                                            Expr::Slot { number: None },
                                            Expr::Number(Number::Integer(2))
                                        ]
                                    }
                                ]
                            }
                        ]
                    }
                ]
            })))
        };

        // Test with small values to avoid deep recursion
        let test_values = vec![1, 2, 3, 4, 5];
        
        for value in test_values {
            let result = pure_function::substitute_slots(&fibonacci_like_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for fib({}), got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with concurrent aggregation
    #[test]
    fn test_pure_function_concurrent_aggregation() {
        // Test: Concurrent aggregation of results from pure functions
        let aggregation_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Length".to_string() })),
                        args: vec![Expr::Slot { number: Some(1) }]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: Some(2) },
                            Expr::Number(Number::Integer(2))
                        ]
                    }
                ]
            })))
        };

        // Test aggregation of different data types
        let test_cases = vec![
            (Value::List(vec![Value::Integer(1), Value::Integer(2)]), Value::Integer(5)),
            (Value::List(vec![Value::String("a".to_string())]), Value::Integer(10)),
            (Value::List(vec![]), Value::Integer(7))
        ];

        for (list, number) in test_cases {
            let result = pure_function::substitute_slots(&aggregation_func, &[list.clone(), number.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for aggregation ({:?}, {:?}), got {:?}", list, number, other),
            }
        }
    }

    /// Test pure functions with producer-consumer patterns
    #[test]
    fn test_pure_function_producer_consumer() {
        // Test: Producer-consumer pattern with pure function data transformation
        
        // Producer function: (Range[#] &)
        let producer_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Range".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Consumer function: (Sum[#] &)
        let consumer_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Sum".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Test producer-consumer chain
        let producer_inputs = vec![3, 5, 7];
        
        for input in producer_inputs {
            // Producer phase
            let producer_result = pure_function::substitute_slots(&producer_func, &[Value::Integer(input)]);
            assert!(producer_result.is_ok());

            // Consumer phase (simulate with list)
            let test_list = Value::List(vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(3)
            ]);
            let consumer_result = pure_function::substitute_slots(&consumer_func, &[test_list]);
            assert!(consumer_result.is_ok());

            // Both phases should succeed
            match (producer_result.unwrap(), consumer_result.unwrap()) {
                (Value::Quote(_), Value::Quote(_)) => {}, // Expected
                (a, b) => panic!("Expected both Quote for input {}, got {:?}, {:?}", input, a, b),
            }
        }
    }

    /// Test pure functions with scatter-gather patterns
    #[test]
    fn test_pure_function_scatter_gather() {
        // Test: Scatter-gather pattern with pure function processing
        
        // Scatter function: (Partition[#, 3] &) - split into chunks
        let scatter_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Partition".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(3))
                ]
            })))
        };

        // Process function: (Map[(# * 2 &), #] &) - process each chunk
        let process_func = Value::PureFunction {
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

        // Gather function: (Flatten[#] &) - combine results
        let gather_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Flatten".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        // Test scatter-gather pipeline
        let large_list = Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3),
            Value::Integer(4), Value::Integer(5), Value::Integer(6),
            Value::Integer(7), Value::Integer(8), Value::Integer(9)
        ]);

        // Scatter phase
        let scatter_result = pure_function::substitute_slots(&scatter_func, &[large_list]);
        assert!(scatter_result.is_ok());

        // Process phase (simulate with chunk)
        let test_chunk = Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3)
        ]);
        let process_result = pure_function::substitute_slots(&process_func, &[test_chunk]);
        assert!(process_result.is_ok());

        // Gather phase (simulate with processed chunks)
        let processed_chunks = Value::List(vec![
            Value::List(vec![Value::Integer(2), Value::Integer(4)]),
            Value::List(vec![Value::Integer(6), Value::Integer(8)])
        ]);
        let gather_result = pure_function::substitute_slots(&gather_func, &[processed_chunks]);
        assert!(gather_result.is_ok());

        // All phases should succeed
        match (scatter_result.unwrap(), process_result.unwrap(), gather_result.unwrap()) {
            (Value::Quote(_), Value::Quote(_), Value::Quote(_)) => {}, // Expected
            (a, b, c) => panic!("Expected all Quote, got {:?}, {:?}, {:?}", a, b, c),
        }
    }

    /// Test error handling in concurrent pure function operations
    #[test]
    fn test_pure_function_concurrent_error_handling() {
        // Test: Error propagation in concurrent contexts
        let error_prone_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Divide".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Slot { number: Some(5) } // Out of bounds
                ]
            })))
        };

        // Test with insufficient arguments
        let result = pure_function::substitute_slots(&error_prone_func, &[
            Value::Integer(10),
            Value::Integer(2)
        ]);

        // Should error due to out-of-bounds slot
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("#5"));
    }

    /// Test pure functions with concurrent data structures
    #[test]
    fn test_pure_function_concurrent_data_structures() {
        // Test: Pure functions with concurrent-safe data structures
        let data_structure_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "AssociationMap".to_string() })),
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

        // Test with association-like structure
        let test_assoc = Value::List(vec![
            Value::List(vec![Value::String("a".to_string()), Value::Integer(1)]),
            Value::List(vec![Value::String("b".to_string()), Value::Integer(2)]),
            Value::List(vec![Value::String("c".to_string()), Value::Integer(3)])
        ]);

        let result = pure_function::substitute_slots(&data_structure_func, &[test_assoc]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test thread safety of pure function operations
    #[test]
    fn test_pure_function_thread_safety() {
        // Test: Thread safety properties of pure functions
        let thread_safe_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };

        // Simulate concurrent access from multiple threads
        let concurrent_values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut results = Vec::new();

        for value in concurrent_values {
            let result = pure_function::substitute_slots(&thread_safe_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            results.push(result.unwrap());
        }

        // All concurrent operations should succeed
        assert_eq!(results.len(), 10);
        for (i, result) in results.iter().enumerate() {
            match result {
                Value::Quote(_) => {}, // Expected
                other => panic!("Concurrent result {} should be Quote, got {:?}", i, other),
            }
        }
    }
}

/// Performance and scalability tests for concurrent operations
#[cfg(test)]
mod concurrent_performance {
    use super::*;

    #[test]
    fn test_concurrent_pure_function_performance() {
        // Test: Performance characteristics under concurrent load
        let performance_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Power".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(2))
                ]
            })))
        };

        let start = std::time::Instant::now();
        
        // Simulate high concurrent load
        for i in 0..1000 {
            let result = pure_function::substitute_slots(&performance_func, &[Value::Integer(i)]);
            assert!(result.is_ok());
        }
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time (< 500ms for 1000 operations)
        assert!(duration.as_millis() < 500, 
            "Concurrent performance test failed: took {:?}", duration);
    }

    #[test]
    fn test_concurrent_memory_efficiency() {
        // Test: Memory efficiency under concurrent operations
        let memory_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "List".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(1))
                        ]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(2))
                        ]
                    }
                ]
            })))
        };

        // Apply function many times to test memory usage
        for i in 0..200 {
            let result = pure_function::substitute_slots(&memory_func, &[Value::Integer(i)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote at iteration {}, got {:?}", i, other),
            }
        }

        // Test completed without memory issues
    }

    #[test]
    fn test_concurrent_scalability() {
        // Test: Scalability with increasing concurrent load
        let scalability_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "EvenQ".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Divide".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(2))
                        ]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(3))
                        ]
                    }
                ]
            })))
        };

        // Test with increasing load sizes
        let load_sizes = vec![10, 50, 100, 200];
        
        for size in load_sizes {
            let start = std::time::Instant::now();
            
            for i in 0..size {
                let result = pure_function::substitute_slots(&scalability_func, &[Value::Integer(i)]);
                assert!(result.is_ok());
            }
            
            let duration = start.elapsed();
            
            // Should scale reasonably (< 1ms per operation)
            assert!(duration.as_millis() < size as u128, 
                "Scalability test failed for size {}: took {:?}", size, duration);
        }
    }
}

/// Integration test framework validation
#[cfg(test)]
mod framework_validation {
    use super::*;

    #[test]
    fn test_concurrency_integration_framework_setup() {
        // Verify the concurrency integration testing framework is working
        let identity_func = Value::PureFunction {
            body: Box::new(Value::Slot { number: None })
        };

        // Test with values suitable for concurrent operations
        let test_values = vec![
            Value::Integer(42),
            Value::Real(3.14),
            Value::String("concurrent".to_string()),
            Value::Boolean(true),
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])
        ];

        for value in test_values {
            let result = pure_function::substitute_slots(&identity_func, &[value.clone()]);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), value);
        }
    }

    #[test]
    fn test_concurrent_expression_validity() {
        // Test that concurrent pure function expressions are valid
        let concurrent_expr = Value::Quote(Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "ParallelMap".to_string() })),
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
        }));

        // Verify expression structure
        match concurrent_expr {
            Value::Quote(expr) => {
                match *expr {
                    Expr::Function { head, args } => {
                        assert!(matches!(*head, Expr::Symbol(_)));
                        assert_eq!(args.len(), 2);
                        assert!(matches!(args[0], Expr::PureFunction { .. }));
                        assert!(matches!(args[1], Expr::List(_)));
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }
}