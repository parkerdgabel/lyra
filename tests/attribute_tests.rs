use lyra::ast::{Expr, Symbol, Number};
use lyra::compiler::Compiler;
use lyra::linker::{FunctionRegistry, FunctionAttribute, FunctionSignature};
use lyra::vm::{Value, VirtualMachine, VmError};

/// Test module for Function Attribute system (Hold, Listable, Orderless)
/// Following TDD approach - these tests should FAIL initially (RED phase)

#[cfg(test)]
mod function_attribute_metadata_tests {
    use super::*;
    
    /// Test that FunctionAttribute enum and FunctionSignature work correctly
    #[test]
    fn test_function_attribute_creation() {
        // Test creating function attributes
        let hold_attr = FunctionAttribute::Hold(vec![1, 2]);
        let listable_attr = FunctionAttribute::Listable;
        let orderless_attr = FunctionAttribute::Orderless;
        
        // Test creating function signature with attributes
        let signature = FunctionSignature::with_attributes(
            "TestFunction",
            "test_method", 
            2,
            vec![hold_attr.clone(), listable_attr, orderless_attr.clone()]
        );
        
        // Test attribute query methods
        assert!(signature.has_attribute(&hold_attr));
        assert!(signature.is_listable());
        assert!(signature.is_orderless());
        
        // Test Hold position extraction
        assert_eq!(signature.get_hold_positions(), Some(&vec![1, 2]));
        
        // This test should PASS - confirming the metadata system works
        println!("✅ FunctionAttribute metadata system operational!");
    }
    
    /// Test different attribute combinations
    #[test]
    fn test_stdlib_function_attributes() {
        // Test math function with Listable attribute (like Sin)
        let sin_signature = FunctionSignature::with_attributes(
            "Math",
            "Sin", 
            1,
            vec![FunctionAttribute::Listable, FunctionAttribute::Protected]
        );
        
        assert!(sin_signature.is_listable());
        assert!(sin_signature.has_attribute(&FunctionAttribute::Protected));
        assert!(!sin_signature.is_orderless());
        
        // Test arithmetic function with Orderless + Listable (like Plus)
        let plus_signature = FunctionSignature::with_attributes(
            "Arithmetic",
            "Plus",
            2, 
            vec![FunctionAttribute::Orderless, FunctionAttribute::Listable]
        );
        
        assert!(plus_signature.is_orderless());
        assert!(plus_signature.is_listable());
        assert_eq!(plus_signature.get_hold_positions(), None);
        
        println!("✅ Stdlib function attribute metadata works!");
    }
    
    /// Test the AttributeRegistry system integration
    #[test]
    fn test_attribute_registry_system() {
        // Create a function registry with attribute support
        let mut registry = FunctionRegistry::new();
        
        // Register some functions with attributes
        registry.register_function_attributes("Sin", vec![
            FunctionAttribute::Listable,
            FunctionAttribute::Protected,
        ]);
        
        registry.register_function_attributes("Plus", vec![
            FunctionAttribute::Orderless,
            FunctionAttribute::Listable,
            FunctionAttribute::Protected,
        ]);
        
        registry.register_function_attributes("If", vec![
            FunctionAttribute::Hold(vec![2, 3]),
            FunctionAttribute::Protected,
        ]);
        
        // Test getting functions by attribute
        let listable_functions = registry.get_listable_functions();
        assert!(listable_functions.contains(&"Sin".to_string()));
        assert!(listable_functions.contains(&"Plus".to_string()));
        assert!(!listable_functions.contains(&"If".to_string()));
        
        let orderless_functions = registry.get_orderless_functions();
        assert!(orderless_functions.contains(&"Plus".to_string()));
        assert!(!orderless_functions.contains(&"Sin".to_string()));
        
        // Test getting attributes for specific functions
        let sin_attributes = registry.get_function_attributes("Sin");
        assert!(sin_attributes.contains(&FunctionAttribute::Listable));
        assert!(sin_attributes.contains(&FunctionAttribute::Protected));
        assert!(!sin_attributes.contains(&FunctionAttribute::Orderless));
        
        // Test Hold functions
        let hold_functions = registry.get_hold_functions();
        assert_eq!(hold_functions.len(), 1);
        assert_eq!(hold_functions[0].0, "If");
        assert_eq!(hold_functions[0].1, vec![2, 3]);
        
        // Test attribute checking
        assert!(registry.function_has_attribute("Sin", &FunctionAttribute::Listable));
        assert!(!registry.function_has_attribute("Sin", &FunctionAttribute::Orderless));
        
        // Test registry statistics
        let (functions_count, attributes_count) = registry.get_attribute_stats();
        assert_eq!(functions_count, 3); // Sin, Plus, If
        assert!(attributes_count >= 3); // Listable, Orderless, Hold, Protected
        
        println!("✅ AttributeRegistry system works correctly!");
    }
    
    /// Test stdlib attribute pre-registration
    #[test]
    fn test_stdlib_attribute_preregistration() {
        let mut registry = FunctionRegistry::new();
        
        // Register standard library attributes
        registry.register_stdlib_attributes();
        
        // Verify math functions are Listable
        assert!(registry.function_has_attribute("Sin", &FunctionAttribute::Listable));
        assert!(registry.function_has_attribute("Cos", &FunctionAttribute::Listable));
        assert!(registry.function_has_attribute("Sqrt", &FunctionAttribute::Listable));
        
        // Verify arithmetic functions are Orderless and Listable
        assert!(registry.function_has_attribute("Plus", &FunctionAttribute::Orderless));
        assert!(registry.function_has_attribute("Plus", &FunctionAttribute::Listable));
        assert!(registry.function_has_attribute("Times", &FunctionAttribute::Orderless));
        
        // Verify Hold functions
        let hold_functions = registry.get_hold_functions();
        let setdelayed_hold = hold_functions.iter().find(|(name, _)| name == "SetDelayed");
        assert!(setdelayed_hold.is_some());
        assert_eq!(setdelayed_hold.unwrap().1, vec![1]);
        
        let if_hold = hold_functions.iter().find(|(name, _)| name == "If");
        assert!(if_hold.is_some());
        assert_eq!(if_hold.unwrap().1, vec![2, 3]);
        
        // Verify all functions are Protected
        assert!(registry.function_has_attribute("Sin", &FunctionAttribute::Protected));
        assert!(registry.function_has_attribute("Plus", &FunctionAttribute::Protected));
        assert!(registry.function_has_attribute("Length", &FunctionAttribute::Protected));
        
        let (functions_count, _) = registry.get_attribute_stats();
        assert!(functions_count >= 10); // Should have registered many stdlib functions
        
        println!("✅ Stdlib attribute pre-registration works!");
    }
}

#[cfg(test)]
mod hold_attribute_tests {
    use super::*;
    
    /// Test that Hold[1] prevents evaluation of the first argument
    #[test]
    fn test_hold_first_argument() {
        // SETUP: Create a function with Hold[1] attribute
        // This should fail initially - FunctionAttribute doesn't exist yet
        // let hold_attr = FunctionAttribute::Hold(vec![1]);
        // let signature = FunctionSignature {
        //     type_name: "TestFunction".to_string(),
        //     method_name: "test_hold".to_string(),
        //     arity: 2,
        //     arg_types: None,
        //     attributes: vec![hold_attr],
        //     function: UnifiedFunction::Stdlib(test_hold_function),
        // };
        // registry.register_function_signature("TestHold", signature).unwrap();
        
        // TEST: Compile function call where first argument should not be evaluated
        let mut compiler = Compiler::new();
        
        // This expression should NOT evaluate 1+1 because of Hold[1] 
        // TestHold[1+1, 2+2] should receive (unevaluated: 1+1, evaluated: 4)
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "TestHold".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(1)),
                        Expr::Number(Number::Integer(1)),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(2)),
                        Expr::Number(Number::Integer(2)),
                    ],
                },
            ],
        };
        
        // This should compile successfully with Hold attribute processing
        let result = compiler.compile_expr(&expr);
        assert!(result.is_ok(), "Hold attribute compilation should succeed");
        
        // This is a placeholder test - actual Hold behavior not yet implemented
        // This assertion will fail initially, confirming RED phase
        assert!(false, "Hold behavior not yet implemented - should fail in RED phase");
    }
    
    /// Test that Hold[2,3] prevents evaluation of second and third arguments
    #[test] 
    fn test_hold_multiple_arguments() {
        // SETUP: Function with Hold[2,3] - hold arguments 2 and 3
        // This should fail initially - attribute system doesn't exist
        // let hold_attr = FunctionAttribute::Hold(vec![2, 3]);
        // registry.register_with_attributes("TestHoldMultiple", signature);
        
        let mut compiler = Compiler::new();
        
        // TestHoldMultiple[1+1, 2+2, 3+3, 4+4]
        // Should evaluate: arg1=2, arg4=8
        // Should hold: arg2=(2+2), arg3=(3+3)
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "TestHoldMultiple".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(1)),
                        Expr::Number(Number::Integer(1)),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(2)),
                        Expr::Number(Number::Integer(2)),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(3)),
                        Expr::Number(Number::Integer(3)),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                    args: vec![
                        Expr::Number(Number::Integer(4)),
                        Expr::Number(Number::Integer(4)),
                    ],
                },
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        assert!(result.is_ok(), "Multiple Hold attribute compilation should succeed");
        
        // This will fail until attribute system is implemented
        assert!(false, "Multiple Hold test - should fail in RED phase");
    }
    
    /// Test that Hold attribute works with nested expressions
    #[test]
    fn test_hold_nested_expressions() {
        // SETUP: Function with Hold[1] and nested expression
        let mut compiler = Compiler::new();
        
        // TestHold[f[x], g[y]] - f[x] should not be evaluated due to Hold[1]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "TestHold".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "f".to_string() })),
                    args: vec![Expr::Symbol(Symbol { name: "x".to_string() })],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "g".to_string() })),
                    args: vec![Expr::Symbol(Symbol { name: "y".to_string() })],
                },
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This should fail initially - no Hold processing implemented
        assert!(false, "Nested Hold test - should fail in RED phase");
    }
    
    /// Test Hold attribute with Symbol expressions that would normally evaluate
    #[test]
    fn test_hold_symbol_evaluation() {
        // SETUP: Symbol that has a value assigned
        let mut compiler = Compiler::new();
        
        // If x=5, then TestHold[x, y] with Hold[1] should receive (Symbol: x, Value: y_value)
        // First argument should remain as Symbol, not evaluate to 5
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "TestHold".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "x".to_string() }),
                Expr::Symbol(Symbol { name: "y".to_string() }),
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This will fail until Hold attribute implementation exists
        assert!(false, "Symbol Hold test - should fail in RED phase");
    }
}

#[cfg(test)]
mod listable_attribute_tests {
    use super::*;
    
    /// Test that Listable attribute automatically threads over lists
    #[test]
    fn test_listable_single_list_argument() {
        // SETUP: Math function with Listable attribute (like Sin)
        // This should fail initially - Listable attribute doesn't exist
        // let listable_attr = FunctionAttribute::Listable;
        // Sin should have Listable attribute: Sin[{1,2,3}] → {Sin[1], Sin[2], Sin[3]}
        
        let mut compiler = Compiler::new();
        
        // Sin[{1.0, 2.0, 3.0}] should become {Sin[1.0], Sin[2.0], Sin[3.0]}
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
            args: vec![
                Expr::List(vec![
                    Expr::Number(Number::Real(1.0)),
                    Expr::Number(Number::Real(2.0)), 
                    Expr::Number(Number::Real(3.0)),
                ]),
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        assert!(result.is_ok(), "Listable compilation should succeed");
        
        // This will fail until Listable is implemented
        assert!(false, "Listable test - should fail in RED phase");
    }
    
    /// Test Listable with multiple list arguments of same length
    #[test]
    fn test_listable_multiple_lists() {
        let mut compiler = Compiler::new();
        
        // Plus[{1,2,3}, {4,5,6}] should become {Plus[1,4], Plus[2,5], Plus[3,6]} = {5,7,9}
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::List(vec![
                    Expr::Number(Number::Integer(1)),
                    Expr::Number(Number::Integer(2)),
                    Expr::Number(Number::Integer(3)),
                ]),
                Expr::List(vec![
                    Expr::Number(Number::Integer(4)),
                    Expr::Number(Number::Integer(5)),
                    Expr::Number(Number::Integer(6)),
                ]),
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This should fail initially - no Listable processing
        assert!(false, "Multiple Listable test - should fail in RED phase");
    }
    
    /// Test Listable with mixed scalar and list arguments
    #[test]
    fn test_listable_mixed_arguments() {
        let mut compiler = Compiler::new();
        
        // Plus[{1,2,3}, 10] should become {Plus[1,10], Plus[2,10], Plus[3,10]} = {11,12,13}
        // Scalar argument should be broadcast to match list length
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::List(vec![
                    Expr::Number(Number::Integer(1)),
                    Expr::Number(Number::Integer(2)),
                    Expr::Number(Number::Integer(3)),
                ]),
                Expr::Number(Number::Integer(10)),
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This should fail initially
        assert!(false, "Mixed Listable test - should fail in RED phase");
    }
}

#[cfg(test)]
mod orderless_attribute_tests {
    use super::*;
    
    /// Test that Orderless attribute sorts arguments deterministically
    #[test]
    fn test_orderless_sorts_arguments() {
        // SETUP: Plus should have Orderless attribute
        // This should fail initially - Orderless attribute doesn't exist
        // let orderless_attr = FunctionAttribute::Orderless;
        // Plus should have Orderless: Plus[b, a] should be equivalent to Plus[a, b]
        
        let mut compiler = Compiler::new();
        
        // Plus[b, a] should be canonicalized to Plus[a, b] at compile time
        let expr1 = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "b".to_string() }),
                Expr::Symbol(Symbol { name: "a".to_string() }),
            ],
        };
        
        let expr2 = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Symbol(Symbol { name: "a".to_string() }),
                Expr::Symbol(Symbol { name: "b".to_string() }),
            ],
        };
        
        let result1 = compiler.compile_expr(&expr1);
        let result2 = compiler.compile_expr(&expr2);
        
        // Both should compile successfully, but until Orderless is implemented
        // they won't produce identical bytecode
        assert!(result1.is_ok() && result2.is_ok(), "Orderless compilation should succeed");
        
        // This will fail until Orderless is implemented
        assert!(false, "Orderless test - should fail in RED phase");
    }
    
    /// Test Orderless with numeric arguments 
    #[test]
    fn test_orderless_numeric_sorting() {
        let mut compiler = Compiler::new();
        
        // Plus[3, 1, 2] should be canonicalized to Plus[1, 2, 3]
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(3)),
                Expr::Number(Number::Integer(1)), 
                Expr::Number(Number::Integer(2)),
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This should fail initially
        assert!(false, "Numeric Orderless test - should fail in RED phase");
    }
    
    /// Test Orderless with complex expressions
    #[test]
    fn test_orderless_complex_expressions() {
        let mut compiler = Compiler::new();
        
        // Plus[f[x], a, g[y]] should be sorted by some canonical ordering
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "f".to_string() })),
                    args: vec![Expr::Symbol(Symbol { name: "x".to_string() })],
                },
                Expr::Symbol(Symbol { name: "a".to_string() }),
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "g".to_string() })),
                    args: vec![Expr::Symbol(Symbol { name: "y".to_string() })],
                },
            ],
        };
        
        let result = compiler.compile_expr(&expr);
        
        // This should fail initially
        assert!(false, "Complex Orderless test - should fail in RED phase");
    }
}

// Helper test functions would be implemented once the attribute system exists
// Currently commented out to avoid compilation errors in RED phase

// fn test_hold_function(args: &[Value]) -> Result<Value, VmError> {
//     // This function would receive both evaluated and unevaluated arguments
//     // and could inspect which ones were held
//     Ok(Value::Integer(args.len() as i64))
// }

// fn test_hold_multiple_function(args: &[Value]) -> Result<Value, VmError> {
//     // This function would receive mix of evaluated and unevaluated arguments  
//     // based on Hold[2,3] specification
//     Ok(Value::Integer(args.len() as i64))
// }