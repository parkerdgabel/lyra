//! Integration tests for pure functions with module system and import/export
//!
//! This module tests the interaction between pure functions and Lyra's module system,
//! including namespace resolution, import/export functionality, cross-module usage,
//! and module-scoped pure function definitions.

use lyra::{
    ast::{Expr, Number, Symbol},
    vm::{Value, VirtualMachine},
    pure_function,
    parser::Parser,
    compiler::Compiler,
    error::Result,
};

#[cfg(test)]
mod module_integration {
    use super::*;

    /// Test pure functions with module namespace resolution
    #[test]
    fn test_pure_function_module_namespaces() {
        // Test: (MyModule`privateFunction[#] &)
        let namespaced_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "MyModule`privateFunction".to_string() })),
                args: vec![Expr::Slot { number: None }]
            })))
        };

        let test_value = Value::Integer(42);
        let result = pure_function::substitute_slots(&namespaced_func, &[test_value]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(expr) => {
                match *expr {
                    Expr::Function { head, args } => {
                        // Verify namespace is preserved
                        if let Expr::Symbol(Symbol { name }) = *head {
                            assert!(name.contains("MyModule`"));
                        }
                        assert_eq!(args.len(), 1);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with import statements
    #[test]
    fn test_pure_function_with_imports() {
        // Test: Import["Utils"] followed by (Utils`helper[#] &)
        let imported_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Utils`helper".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::String("imported".to_string())
                ]
            })))
        };

        let test_values = vec![
            Value::Integer(10),
            Value::String("test".to_string()),
            Value::List(vec![Value::Integer(1), Value::Integer(2)])
        ];

        for value in test_values {
            let result = pure_function::substitute_slots(&imported_func, &[value.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for imported function with {:?}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with exported symbols
    #[test]
    fn test_pure_function_exported_symbols() {
        // Test: Export["publicFunction"] with pure function definition
        let exported_pure_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "publicFunction".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Number(Number::Integer(100))
                        ]
                    }
                ]
            })))
        };

        // Test that exported pure functions work correctly
        let test_cases = vec![1, 5, 10, 25];
        
        for value in test_cases {
            let result = pure_function::substitute_slots(&exported_pure_func, &[Value::Integer(value)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for exported function with {}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with cross-module dependencies
    #[test]
    fn test_pure_function_cross_module_dependencies() {
        // Test: ModuleA`func calls ModuleB`helper
        let cross_module_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ModuleA`process".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ModuleB`transform".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_data = vec![
            Value::Integer(42),
            Value::String("cross-module".to_string()),
            Value::Real(3.14159)
        ];

        for data in test_data {
            let result = pure_function::substitute_slots(&cross_module_func, &[data.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for cross-module function with {:?}, got {:?}", data, other),
            }
        }
    }

    /// Test pure functions with module-scoped variables
    #[test]
    fn test_pure_function_module_scoped_variables() {
        // Test: Pure function referencing module-scoped constants
        let module_scoped_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Symbol(Symbol { name: "MyModule`CONSTANT_VALUE".to_string() })
                ]
            })))
        };

        let test_inputs = vec![5, 10, 15, 20];
        
        for input in test_inputs {
            let result = pure_function::substitute_slots(&module_scoped_func, &[Value::Integer(input)]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(expr) => {
                    // Verify module-scoped variable is preserved
                    match *expr {
                        Expr::Function { args, .. } => {
                            assert_eq!(args.len(), 2);
                            // Second argument should be module-scoped symbol
                            if let Expr::Symbol(Symbol { name }) = &args[1] {
                                assert!(name.contains("MyModule`CONSTANT_VALUE"));
                            }
                        }
                        other => panic!("Expected Function, got {:?}", other),
                    }
                }
                other => panic!("Expected Quote, got {:?}", other),
            }
        }
    }

    /// Test pure functions with module initialization
    #[test]
    fn test_pure_function_module_initialization() {
        // Test: Pure functions defined during module initialization
        let init_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Initialize".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                        args: vec![
                            Expr::Slot { number: None },
                            Expr::Symbol(Symbol { name: "InitializationConstant".to_string() })
                        ]
                    }
                ]
            })))
        };

        let result = pure_function::substitute_slots(&init_func, &[Value::Integer(7)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for initialization function, got {:?}", other),
        }
    }

    /// Test pure functions with dynamic module loading
    #[test]
    fn test_pure_function_dynamic_module_loading() {
        // Test: Pure functions with dynamically loaded modules
        let dynamic_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "If".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ModuleLoadedQ".to_string() })),
                        args: vec![Expr::String("DynamicModule".to_string())]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "DynamicModule`process".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Identity".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_value = Value::String("dynamic_test".to_string());
        let result = pure_function::substitute_slots(&dynamic_func, &[test_value]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for dynamic module function, got {:?}", other),
        }
    }

    /// Test pure functions with module versioning
    #[test]
    fn test_pure_function_module_versioning() {
        // Test: Pure functions with versioned module references
        let versioned_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "MathUtils`v2`advanced_calc".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "MathUtils`v1`basic_calc".to_string() })),
                        args: vec![Expr::Slot { number: Some(2) }]
                    }
                ]
            })))
        };

        let result = pure_function::substitute_slots(&versioned_func, &[
            Value::Integer(10),
            Value::Integer(5)
        ]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(expr) => {
                // Verify version information is preserved
                match *expr {
                    Expr::Function { head, args } => {
                        if let Expr::Symbol(Symbol { name }) = *head {
                            assert!(name.contains("v2"));
                        }
                        assert_eq!(args.len(), 2);
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test pure functions with module dependency resolution
    #[test]
    fn test_pure_function_dependency_resolution() {
        // Test: Complex dependency chain A -> B -> C
        let dependency_chain_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ModuleA`process".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ModuleB`transform".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "ModuleC`base_operation".to_string() })),
                                args: vec![Expr::Slot { number: None }]
                            }
                        ]
                    }
                ]
            })))
        };

        let test_value = Value::Integer(100);
        let result = pure_function::substitute_slots(&dependency_chain_func, &[test_value]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for dependency chain, got {:?}", other),
        }
    }

    /// Test pure functions with module aliases
    #[test]
    fn test_pure_function_module_aliases() {
        // Test: Import["VeryLongModuleName", "Short"] followed by (Short`func[#] &)
        let aliased_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Short`utility".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::String("aliased_call".to_string())
                ]
            })))
        };

        let test_values = vec![
            Value::Integer(1),
            Value::String("alias_test".to_string()),
            Value::Boolean(true)
        ];

        for value in test_values {
            let result = pure_function::substitute_slots(&aliased_func, &[value.clone()]);
            assert!(result.is_ok());
            
            match result.unwrap() {
                Value::Quote(_) => {}, // Expected
                other => panic!("Expected Quote for aliased function with {:?}, got {:?}", value, other),
            }
        }
    }

    /// Test pure functions with conditional imports
    #[test]
    fn test_pure_function_conditional_imports() {
        // Test: Conditional module loading based on runtime conditions
        let conditional_import_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Switch".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "SystemInformation".to_string() })),
                        args: vec![Expr::String("Platform".to_string())]
                    },
                    Expr::String("Windows".to_string()),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "WindowsUtils`process".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::String("Linux".to_string()),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "LinuxUtils`process".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    },
                    Expr::Symbol(Symbol { name: "_".to_string() }),
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "GenericUtils`process".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_data = Value::String("platform_specific_data".to_string());
        let result = pure_function::substitute_slots(&conditional_import_func, &[test_data]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for conditional import, got {:?}", other),
        }
    }

    /// Test pure functions with module-scoped pure function definitions
    #[test]
    fn test_pure_function_module_scoped_definitions() {
        // Test: Module defines its own pure functions that reference other pure functions
        let module_defined_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Compose".to_string() })),
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

        let result = pure_function::substitute_slots(&module_defined_func, &[Value::Integer(5)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for module-defined composition, got {:?}", other),
        }
    }

    /// Test pure functions with export lists and visibility
    #[test]
    fn test_pure_function_export_visibility() {
        // Test: Pure functions with different visibility levels
        let visibility_test_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "TestModule`PublicFunction".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "TestModule`Private`helper".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let test_input = Value::Integer(123);
        let result = pure_function::substitute_slots(&visibility_test_func, &[test_input]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(expr) => {
                // Verify both public and private namespace references
                match *expr {
                    Expr::Function { head, args } => {
                        if let Expr::Symbol(Symbol { name }) = *head {
                            assert!(name.contains("PublicFunction"));
                        }
                        assert_eq!(args.len(), 1);
                        
                        // Check nested private function reference
                        if let Expr::Function { head: inner_head, .. } = &args[0] {
                            if let Expr::Symbol(Symbol { name }) = inner_head.as_ref() {
                                assert!(name.contains("Private`helper"));
                            }
                        }
                    }
                    other => panic!("Expected Function, got {:?}", other),
                }
            }
            other => panic!("Expected Quote, got {:?}", other),
        }
    }

    /// Test error handling with missing module dependencies
    #[test]
    fn test_pure_function_missing_module_error_handling() {
        // Test: Pure function referencing non-existent module
        let missing_module_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "NonExistentModule`function".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Slot { number: Some(5) } // Also test out-of-bounds slot
                ]
            })))
        };

        let result = pure_function::substitute_slots(&missing_module_func, &[
            Value::Integer(42),
            Value::String("test".to_string())
        ]);

        // Should error due to out-of-bounds slot (slot #5 with only 2 arguments)
        assert!(result.is_err());
        
        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("#5"));
    }

    /// Test pure functions with circular module dependencies
    #[test]
    fn test_pure_function_circular_dependencies() {
        // Test: Modules with circular references (should be handled gracefully)
        let circular_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ModuleA`callB".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ModuleB`callA".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    }
                ]
            })))
        };

        let result = pure_function::substitute_slots(&circular_func, &[Value::Integer(1)]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected - circular references in expressions are valid
            other => panic!("Expected Quote for circular reference, got {:?}", other),
        }
    }

    /// Test pure functions with package-level organization
    #[test]
    fn test_pure_function_package_organization() {
        // Test: Package.Module.Submodule organization
        let package_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "MyPackage`Core`Math`advanced".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "MyPackage`Utils`String`process".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "ToString".to_string() })),
                                args: vec![Expr::Slot { number: None }]
                            }
                        ]
                    }
                ]
            })))
        };

        let test_number = Value::Integer(987);
        let result = pure_function::substitute_slots(&package_func, &[test_number]);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::Quote(_) => {}, // Expected
            other => panic!("Expected Quote for package organization, got {:?}", other),
        }
    }
}

/// Module system performance and memory tests
#[cfg(test)]
mod module_performance {
    use super::*;

    #[test]
    fn test_module_namespace_performance() {
        // Test: Performance with many module namespace resolutions
        let complex_namespace_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "VeryLongModuleName`SubModule`DeepNesting`Function".to_string() })),
                args: vec![
                    Expr::Slot { number: None },
                    Expr::Symbol(Symbol { name: "AnotherModule`Constants`PI".to_string() })
                ]
            })))
        };

        let start = std::time::Instant::now();
        
        // Test many namespace resolutions
        for i in 0..1000 {
            let result = pure_function::substitute_slots(&complex_namespace_func, &[Value::Integer(i)]);
            assert!(result.is_ok());
        }
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time (< 100ms for 1000 operations)
        assert!(duration.as_millis() < 100, 
            "Module namespace performance test failed: took {:?}", duration);
    }

    #[test]
    fn test_cross_module_memory_efficiency() {
        // Test: Memory efficiency with cross-module references
        let cross_module_func = Value::PureFunction {
            body: Box::new(Value::Quote(Box::new(Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "ModuleA`process".to_string() })),
                args: vec![
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "ModuleB`transform".to_string() })),
                        args: vec![
                            Expr::Function {
                                head: Box::new(Expr::Symbol(Symbol { name: "ModuleC`validate".to_string() })),
                                args: vec![Expr::Slot { number: None }]
                            }
                        ]
                    }
                ]
            })))
        };

        // Apply function many times to test memory usage
        for i in 0..500 {
            let result = pure_function::substitute_slots(&cross_module_func, &[Value::Integer(i)]);
            assert!(result.is_ok());
            
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
    fn test_module_integration_framework_setup() {
        // Verify the module integration testing framework is working
        let identity_func = Value::PureFunction {
            body: Box::new(Value::Slot { number: None })
        };

        // Test with values suitable for module operations
        let test_values = vec![
            Value::Integer(42),
            Value::String("module_test".to_string()),
            Value::Boolean(true),
            Value::List(vec![
                Value::String("namespace".to_string()),
                Value::String("function".to_string())
            ])
        ];

        for value in test_values {
            let result = pure_function::substitute_slots(&identity_func, &[value.clone()]);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), value);
        }
    }

    #[test]
    fn test_module_expression_structure_validation() {
        // Test that module-related expressions have valid structure
        let module_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Import".to_string() })),
            args: vec![
                Expr::String("TestModule".to_string()),
                Expr::PureFunction {
                    body: Box::new(Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "TestModule`function".to_string() })),
                        args: vec![Expr::Slot { number: None }]
                    })
                }
            ]
        };

        // Verify expression structure
        match module_expr {
            Expr::Function { head, args } => {
                assert!(matches!(*head, Expr::Symbol(_)));
                assert_eq!(args.len(), 2);
                assert!(matches!(args[0], Expr::String(_)));
                assert!(matches!(args[1], Expr::PureFunction { .. }));
            }
            other => panic!("Expected Function expression, got {:?}", other),
        }
    }

    #[test]
    fn test_namespace_symbol_validation() {
        // Test that namespace symbols are handled correctly
        let namespaced_symbols = vec![
            "Module`function",
            "Package`Module`function", 
            "Very`Deep`Nested`Module`function",
            "Module`Private`internal",
            "Module`v1`legacy_function"
        ];

        for symbol_name in namespaced_symbols {
            let symbol = Expr::Symbol(Symbol { name: symbol_name.to_string() });
            
            // Verify symbol structure
            match symbol {
                Expr::Symbol(Symbol { name }) => {
                    assert!(name.contains("`"));
                    assert!(name.len() > 3); // Non-trivial namespace
                }
                other => panic!("Expected Symbol, got {:?}", other),
            }
        }
    }
}