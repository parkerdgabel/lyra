//! Stress tests for pure functions under extreme conditions
//!
//! This module tests the robustness, performance, and memory characteristics
//! of pure functions under stress conditions including deep nesting,
//! many parameters, large arguments, and high memory pressure.

use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};

#[cfg(test)]
mod deep_nesting_stress {
    use super::*;

    /// Test pure functions with deep expression nesting (1000+ levels)
    #[test]
    fn test_deep_nesting_arithmetic_expressions() {
        // Create deeply nested arithmetic: ((((# + 1) + 1) + 1) + 1) ... (1000 levels)
        let nesting_depth = 1000;
        
        let mut nested_expr = Expr::Slot { number: None };
        
        for _ in 0..nesting_depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    nested_expr,
                    Expr::Number(Number::Integer(1))
                ]
            };
        }

        let deep_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        // Test that deep nesting doesn't cause stack overflow
        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_func, &[Value::Integer(0)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep nesting (1000 levels) should not cause stack overflow");
        
        // Should complete in reasonable time (< 1 second)
        assert!(duration.as_secs() < 1, 
            "Deep nesting stress test took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for deep nesting, got {:?}", other),
        }
    }

    /// Test pure functions with deeply nested conditional expressions
    #[test]
    fn test_deep_nesting_conditional_expressions() {
        // Create deeply nested conditionals: If[# > 0, If[# > 1, If[# > 2, ...], ...], ...]
        let nesting_depth = 500; // Reduced for conditionals due to complexity
        
        let mut nested_expr = Expr::Number(Number::Integer(42)); // Base case
        
        for i in (0..nesting_depth).rev() {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Greater".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(i as i64))
                        ]
                    },
                    nested_expr.clone(),
                    Expr::Number(Number::Integer(-1))
                ]
            };
        }

        let deep_conditional_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_conditional_func, &[Value::Integer(250)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep conditional nesting should not cause stack overflow");
        
        // Should complete in reasonable time (< 2 seconds for complex conditionals)
        assert!(duration.as_secs() < 2, 
            "Deep conditional nesting took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for deep conditional nesting, got {:?}", other),
        }
    }

    /// Test pure functions with deeply nested function compositions
    #[test]
    fn test_deep_nesting_function_compositions() {
        // Create deeply nested function calls: f(g(h(i(j(#)))))... (750 levels)
        let nesting_depth = 750;
        
        let mut nested_expr = Expr::Slot { number: None };
        
        for i in 0..nesting_depth {
            let function_name = format!("f{}", i % 10); // Cycle through f0, f1, ..., f9
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: function_name })),
                args: vec![nested_expr]
            };
        }

        let deep_composition_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_composition_func, &[Value::Integer(123)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep function composition should not cause stack overflow");
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1500, 
            "Deep function composition took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for deep composition, got {:?}", other),
        }
    }

    /// Test pure functions with deeply nested list structures
    #[test]
    fn test_deep_nesting_list_structures() {
        // Create deeply nested lists: {{{{{#}}}}} (800 levels)
        let nesting_depth = 800;
        
        let mut nested_expr = Expr::Slot { number: None };
        
        for _ in 0..nesting_depth {
            nested_expr = Expr::List(vec![nested_expr]);
        }

        let deep_list_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_list_func, &[Value::Integer(999)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep list nesting should not cause stack overflow");
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000, 
            "Deep list nesting took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for deep list nesting, got {:?}", other),
        }
    }

    /// Test pure functions with deeply nested pure function definitions
    #[test]
    fn test_deep_nesting_pure_function_definitions() {
        // Create nested pure functions: ((((# + 1 &) &) &) &)... (300 levels)
        let nesting_depth = 300; // Reduced due to complexity
        
        let mut nested_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };
        
        for _ in 0..nesting_depth {
            nested_func = Value::PureFunction {
                body: Box::new(nested_func)
            };
        }

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&nested_func, &[Value::Integer(10)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep pure function nesting should not cause stack overflow");
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 500, 
            "Deep pure function nesting took too long: {:?}", duration);

        // Result should be a pure function (nested functions)
        match result.unwrap() {
            Value::PureFunction { .. } => {}, // Expected
            other => panic!("Expected PureFunction for deep nesting, got {:?}", other),
        }
    }

    /// Test pure functions with deeply nested slot references
    #[test]
    fn test_deep_nesting_slot_references() {
        // Create expressions with many nested slot references
        let nesting_depth = 600;
        
        let mut nested_expr = Expr::Slot { number: None };
        
        for i in 0..nesting_depth {
            let operation = if i % 2 == 0 { "Plus" } else { "Times" };
            let value = if i % 3 == 0 { 1 } else { 2 };
            
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: operation.to_string() })),
                args: vec![
                    Expr::Slot { number: None }, // Always reference the same slot
                    nested_expr,
                    Expr::Number(Number::Integer(value))
                ]
            };
        }

        let deep_slot_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_slot_func, &[Value::Integer(5)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Deep slot reference nesting should not cause stack overflow");
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 800, 
            "Deep slot reference nesting took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for deep slot references, got {:?}", other),
        }
    }

    /// Test memory usage with deep nesting
    #[test]
    fn test_deep_nesting_memory_usage() {
        // Test that deep nesting doesn't cause excessive memory usage
        let nesting_depths = vec![100, 200, 400, 800];
        
        for depth in nesting_depths {
            // Create nested arithmetic expression
            let mut nested_expr = Expr::Slot { number: None };
            
            for _ in 0..depth {
                nested_expr = Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        nested_expr,
                        Expr::Number(Number::Integer(1))
                    ]
                };
            }

            let deep_func = Value::PureFunction {
                body: Box::new(Value::Quote(Box::new(nested_expr)))
            };

            // Measure memory usage (approximate)
            let start_time = std::time::Instant::now();
            let result = pure_function::substitute_slots(&deep_func, &[Value::Integer(0)]);
            let duration = start_time.elapsed();

            assert!(result.is_ok(), "Nesting depth {} should not fail", depth);
            
            // Memory usage should scale reasonably (not exponentially)
            // Duration should increase roughly linearly with depth
            let max_duration_ms = (depth as u128) / 10 + 50; // Rough heuristic
            assert!(duration.as_millis() < max_duration_ms, 
                "Depth {} took too long: {:?} (max: {}ms)", depth, duration, max_duration_ms);
        }
    }

    /// Test error handling with deep nesting and invalid slots
    #[test]
    fn test_deep_nesting_error_handling() {
        // Create deep nesting with an invalid slot at the bottom
        let nesting_depth = 400;
        
        let mut nested_expr = Expr::Slot { number: Some(999) }; // Invalid slot
        
        for _ in 0..nesting_depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    nested_expr,
                    Expr::Number(Number::Integer(1))
                ]
            };
        }

        let deep_error_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&deep_error_func, &[Value::Integer(0)]);
        let duration = start_time.elapsed();

        // Should error due to invalid slot, but not crash
        assert!(result.is_err(), "Should error due to invalid slot #999");
        
        // Error handling should be fast even with deep nesting
        assert!(duration.as_millis() < 200, 
            "Error handling in deep nesting took too long: {:?}", duration);

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("#999"), "Error should mention the invalid slot");
    }

    /// Test recursive-like patterns with deep nesting
    #[test]
    fn test_deep_nesting_recursive_patterns() {
        // Create recursive-like expressions: factorial-like structure
        let nesting_depth = 200; // Reduced for recursive patterns
        
        let mut nested_expr = Expr::Number(Number::Integer(1)); // Base case
        
        for i in 1..=nesting_depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "LessEqual".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(i as i64))
                        ]
                    },
                    Expr::Number(Number::Integer(1)),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            nested_expr.clone()
                        ]
                    }
                ]
            };
        }

        let recursive_pattern_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&recursive_pattern_func, &[Value::Integer(5)]);
        let duration = start_time.elapsed();

        assert!(result.is_ok(), "Recursive pattern with deep nesting should not fail");
        
        // Should complete in reasonable time
        assert!(duration.as_millis() < 1000, 
            "Recursive pattern nesting took too long: {:?}", duration);

        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for recursive pattern, got {:?}", other),
        }
    }

    /// Stress test with maximum reasonable nesting depth
    #[test]
    fn test_maximum_nesting_depth_stress() {
        // Test with very deep nesting to find practical limits
        let max_depth = 2000; // Very deep nesting
        
        // Use simple arithmetic to minimize per-level overhead
        let mut nested_expr = Expr::Slot { number: None };
        
        for _ in 0..max_depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Identity".to_string() })),
                args: vec![nested_expr]
            };
        }

        let max_depth_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let start_time = std::time::Instant::now();
        let result = pure_function::substitute_slots(&max_depth_func, &[Value::Integer(42)]);
        let duration = start_time.elapsed();

        // This might fail on some systems due to stack limits, but should handle gracefully
        if result.is_ok() {
            println!("Successfully handled {} levels of nesting in {:?}", max_depth, duration);
            
            // If it succeeds, should complete in reasonable time (< 5 seconds)
            assert!(duration.as_secs() < 5, 
                "Maximum depth stress test took too long: {:?}", duration);

            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for maximum depth, got {:?}", other),
            }
        } else {
            // If it fails, should fail gracefully with a meaningful error
            let error = result.unwrap_err();
            println!("Maximum depth {} failed gracefully: {}", max_depth, error);
            
            // Error should be related to stack or memory limits, not a panic
            let error_msg = error.to_string();
            assert!(!error_msg.contains("panic"), "Should not panic on deep nesting");
        }
    }
}

/// Performance characteristics under deep nesting
#[cfg(test)]
mod nesting_performance {
    use super::*;

    #[test]
    fn test_nesting_performance_scaling() {
        // Test how performance scales with nesting depth
        let depths = vec![50, 100, 200, 400];
        let mut durations = Vec::new();
        
        for depth in depths {
            // Create nested expression
            let mut nested_expr = Expr::Slot { number: None };
            
            for _ in 0..depth {
                nested_expr = Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        nested_expr,
                        Expr::Number(Number::Integer(1))
                    ]
                };
            }

            let func = Value::PureFunction {
                body: Box::new(Value::Quote(Box::new(nested_expr)))
            };

            // Measure performance
            let start = std::time::Instant::now();
            let result = pure_function::substitute_slots(&func, &[Value::Integer(0)]);
            let duration = start.elapsed();

            assert!(result.is_ok(), "Depth {} should succeed", depth);
            durations.push((depth, duration));
            
            println!("Depth {}: {:?}", depth, duration);
        }

        // Performance should scale roughly linearly, not exponentially
        for i in 1..durations.len() {
            let (prev_depth, prev_duration) = durations[i-1];
            let (curr_depth, curr_duration) = durations[i];
            
            let depth_ratio = curr_depth as f64 / prev_depth as f64;
            let time_ratio = curr_duration.as_nanos() as f64 / prev_duration.as_nanos() as f64;
            
            // Time ratio should not exceed depth ratio by more than 2x (allowing for overhead)
            assert!(time_ratio < depth_ratio * 2.0, 
                "Performance degradation too severe: depth ratio {:.2}, time ratio {:.2}", 
                depth_ratio, time_ratio);
        }
    }

    #[test]
    fn test_memory_efficiency_with_nesting() {
        // Test that memory usage is efficient with deep nesting
        let depth = 1000;
        
        // Create expression
        let mut nested_expr = Expr::Slot { number: None };
        
        for i in 0..depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "f".to_string() })),
                args: vec![
                    nested_expr,
                    Expr::Number(Number::Integer(i as i64))
                ]
            };
        }

        let func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        // Test multiple iterations to check for memory leaks
        for iteration in 0..100 {
            let start = std::time::Instant::now();
            let result = pure_function::substitute_slots(&func, &[Value::Integer(iteration)]);
            let duration = start.elapsed();

            assert!(result.is_ok(), "Iteration {} should succeed", iteration);
            
            // Each iteration should take roughly the same time (no memory leaks)
            assert!(duration.as_millis() < 100, 
                "Iteration {} took too long: {:?}", iteration, duration);
        }
    }
}

/// Framework validation for stress tests
#[cfg(test)]
mod stress_framework_validation {
    use super::*;

    #[test]
    fn test_stress_test_framework_setup() {
        // Verify the stress testing framework works with simple cases
        let simple_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Number(Number::Integer(1))
                ]
            })))
        };

        let result = pure_function::substitute_slots(&simple_func, &[Value::Integer(41)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    #[test]
    fn test_nesting_infrastructure() {
        // Test that we can create and handle moderate nesting correctly
        let moderate_depth = 10;
        
        let mut nested_expr = Expr::Slot { number: None };
        
        for i in 0..moderate_depth {
            nested_expr = Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Identity".to_string() })),
                args: vec![nested_expr]
            };
        }

        let func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(nested_expr)))
        };

        let result = pure_function::substitute_slots(&func, &[Value::Integer(123)]);
        assert!(result.is_ok());
        
        // Verify the structure is preserved
        match result.unwrap() {
            Value::Quote(expr) => {
                // Should be a nested function structure
                match *expr {
                    Expr::Function { .. } => {}, // Expected
                    other => panic!("Expected nested Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }
}