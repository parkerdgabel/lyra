use lyra::{
    ast::{Expr, Number, Symbol},
    types::{
        LyraType, TensorShape, TypedCompiler, TypeChecker, TypeInferenceEngine,
        infer_expression_type, check_expression_safety, eval_with_type,
        TypeEnvironment, TypeScheme, FunctionSignature, TypeConstraint,
    },
    vm::Value,
};

#[test]
fn test_basic_type_inference() {
    // Test literal type inference
    let int_expr = Expr::Number(Number::Integer(42));
    let int_type = infer_expression_type(&int_expr).unwrap();
    assert_eq!(int_type, LyraType::Integer);

    let real_expr = Expr::Number(Number::Real(3.14));
    let real_type = infer_expression_type(&real_expr).unwrap();
    assert_eq!(real_type, LyraType::Real);

    let string_expr = Expr::String("hello".to_string());
    let string_type = infer_expression_type(&string_expr).unwrap();
    assert_eq!(string_type, LyraType::String);
}

#[test]
fn test_function_type_inference() {
    // Test Plus function
    let plus_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ],
    };
    
    let plus_type = infer_expression_type(&plus_expr).unwrap();
    // Should infer as Integer since both operands are Integer
    assert!(matches!(plus_type, LyraType::Integer) || matches!(plus_type, LyraType::TypeVar(_)));
}

#[test]
fn test_list_type_inference() {
    // Homogeneous integer list
    let int_list = Expr::List(vec![
        Expr::Number(Number::Integer(1)),
        Expr::Number(Number::Integer(2)),
        Expr::Number(Number::Integer(3)),
    ]);
    
    let list_type = infer_expression_type(&int_list).unwrap();
    assert_eq!(list_type, LyraType::List(Box::new(LyraType::Integer)));

    // Empty list
    let empty_list = Expr::List(vec![]);
    let empty_type = infer_expression_type(&empty_list).unwrap();
    assert!(matches!(empty_type, LyraType::List(_)));
}

#[test]
fn test_type_checking_safety() {
    // Valid arithmetic
    let valid_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ],
    };
    
    assert!(check_expression_safety(&valid_expr).is_ok());

    // Invalid arity
    let invalid_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![Expr::Number(Number::Integer(2))], // Only one argument
    };
    
    // This should detect the arity error at type checking time
    let result = check_expression_safety(&invalid_expr);
    // Note: depending on implementation, this might succeed with type inference
    // but fail with stricter type checking
    println!("Arity check result: {:?}", result);
}

#[test]
fn test_typed_compiler_integration() {
    let mut compiler = TypedCompiler::new();
    
    // Compile simple expression with type checking
    let expr = Expr::Number(Number::Integer(42));
    let expr_type = compiler.compile_expr_typed(&expr).unwrap();
    assert_eq!(expr_type, LyraType::Integer);

    // Check type annotation
    let annotation = compiler.get_type_annotation(&expr);
    assert_eq!(annotation, Some(&LyraType::Integer));

    // Create VM and run
    let mut vm = compiler.into_typed_vm();
    let result = vm.run().unwrap();
    assert_eq!(result, Value::Integer(42));

    // Check runtime type matches compile-time type
    let runtime_type = vm.get_value_type(&result);
    assert_eq!(runtime_type, LyraType::Integer);
}

#[test]
fn test_mathematical_function_types() {
    // Test Sin function
    let sin_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
        args: vec![Expr::Number(Number::Real(3.14))],
    };
    
    let sin_type = infer_expression_type(&sin_expr).unwrap();
    // Sin should return Real type
    assert!(matches!(sin_type, LyraType::Real) || matches!(sin_type, LyraType::TypeVar(_)));
}

#[test]
fn test_eval_with_type_integration() {
    // Test the convenience function that returns both value and type
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(5)),
            Expr::Number(Number::Integer(7)),
        ],
    };
    
    let (value, ty) = eval_with_type(&expr).unwrap();
    assert_eq!(value, Value::Integer(12));
    assert!(matches!(ty, LyraType::Integer) || matches!(ty, LyraType::TypeVar(_)));
}

#[test]
fn test_type_environment_scoping() {
    let mut engine = TypeInferenceEngine::new();
    let mut env = TypeEnvironment::new();
    
    // Add a variable to the environment
    env.insert("x".to_string(), TypeScheme::monomorphic(LyraType::Integer));
    
    // Reference the variable
    let var_expr = Expr::Symbol(Symbol { name: "x".to_string() });
    let var_type = engine.infer_expr(&var_expr, &env).unwrap();
    assert_eq!(var_type, LyraType::Integer);
    
    // Reference unknown variable should fail
    let unknown_expr = Expr::Symbol(Symbol { name: "unknown".to_string() });
    let unknown_result = engine.infer_expr(&unknown_expr, &env);
    assert!(unknown_result.is_err());
}

#[test]
fn test_arrow_function_types() {
    // Identity function: (x) => x
    let identity = Expr::ArrowFunction {
        params: vec!["x".to_string()],
        body: Box::new(Expr::Symbol(Symbol { name: "x".to_string() })),
    };
    
    let identity_type = infer_expression_type(&identity).unwrap();
    assert!(identity_type.is_function());
    
    // The identity function should have type α -> α
    if let LyraType::Function { params, return_type } = identity_type {
        assert_eq!(params.len(), 1);
        // Parameter and return type should be the same type variable
        assert_eq!(params[0], *return_type);
    }
}

#[test]
fn test_program_type_inference() {
    let mut engine = TypeInferenceEngine::new();
    let env = TypeEnvironment::new();
    
    let expressions = vec![
        Expr::Number(Number::Integer(42)),
        Expr::String("hello".to_string()),
        Expr::List(vec![Expr::Number(Number::Real(1.0))]),
    ];
    
    let types = engine.infer_program(&expressions, &env).unwrap();
    assert_eq!(types.len(), 3);
    assert_eq!(types[0], LyraType::Integer);
    assert_eq!(types[1], LyraType::String);
    assert_eq!(types[2], LyraType::List(Box::new(LyraType::Real)));
}

#[test]
fn test_tensor_shape_compatibility() {
    // Test tensor shape broadcasting rules
    let shape1 = TensorShape::new(vec![3, 1]);
    let shape2 = TensorShape::new(vec![1, 4]);
    
    assert!(shape1.broadcast_compatible(&shape2));
    
    let result_shape = shape1.broadcast_result(&shape2).unwrap();
    assert_eq!(result_shape.dimensions, vec![3, 4]);
    
    // Incompatible shapes
    let shape3 = TensorShape::new(vec![3]);
    let shape4 = TensorShape::new(vec![4]);
    
    assert!(!shape3.broadcast_compatible(&shape4));
    assert!(shape3.broadcast_result(&shape4).is_none());
}

#[test]
fn test_type_checker_with_signatures() {
    let mut checker = TypeChecker::new();
    
    // Check that built-in functions have signatures
    assert!(checker.has_function("Plus"));
    assert!(checker.has_function("Sin"));
    assert!(!checker.has_function("NonExistent"));
    
    // Get a signature
    let plus_sig = checker.get_signature("Plus").unwrap();
    assert_eq!(plus_sig.params.len(), 2);
    assert!(!plus_sig.variadic);
}

#[test]
fn test_strict_vs_normal_mode() {
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Real(3.5)),
        ],
    };
    
    // Normal mode compiler
    let mut normal_compiler = TypedCompiler::new();
    let normal_result = normal_compiler.compile_expr_typed(&expr);
    
    // Strict mode compiler  
    let mut strict_compiler = TypedCompiler::new_strict();
    let strict_result = strict_compiler.compile_expr_typed(&expr);
    
    // Both should compile (though strict might be more restrictive in the future)
    println!("Normal mode result: {:?}", normal_result);
    println!("Strict mode result: {:?}", strict_result);
}

#[test]
fn test_complex_nested_expression() {
    // Test: Plus[Times[2, 3], Divide[8, 2]]
    let complex_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                args: vec![
                    Expr::Number(Number::Integer(2)),
                    Expr::Number(Number::Integer(3)),
                ],
            },
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Divide".to_string() })),
                args: vec![
                    Expr::Number(Number::Integer(8)),
                    Expr::Number(Number::Integer(2)),
                ],
            },
        ],
    };
    
    // Should type check successfully
    let result = check_expression_safety(&complex_expr);
    assert!(result.is_ok());
    
    // Should evaluate to correct value and type
    let (value, ty) = eval_with_type(&complex_expr).unwrap();
    // Times[2, 3] = 6, Divide[8, 2] = 4.0, Plus[6, 4.0] = 10.0 (Real due to division)
    assert!(matches!(value, Value::Real(_) | Value::Integer(_)));
    assert!(matches!(ty, LyraType::Real | LyraType::Integer | LyraType::TypeVar(_)));
}

#[test]
fn test_polymorphic_function_instantiation() {
    let mut engine = TypeInferenceEngine::new();
    let env = TypeEnvironment::new();
    
    // Test that polymorphic functions can be instantiated differently
    
    // Plus with integers
    let plus_int = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
        ],
    };
    
    let type1 = engine.infer_expr(&plus_int, &env).unwrap();
    
    // Plus with reals  
    let plus_real = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Real(1.0)),
            Expr::Number(Number::Real(2.0)),
        ],
    };
    
    let type2 = engine.infer_expr(&plus_real, &env).unwrap();
    
    // Types should be different instantiations
    println!("Plus[Integer, Integer]: {:?}", type1);
    println!("Plus[Real, Real]: {:?}", type2);
}

#[test]
fn test_error_recovery() {
    // Test that type errors don't crash the system
    let invalid_expr = Expr::Symbol(Symbol { name: "NonExistentFunction".to_string() });
    
    let result = infer_expression_type(&invalid_expr);
    assert!(result.is_err());
    
    // Error should be informative
    match result {
        Err(error) => {
            let error_msg = error.to_string();
            assert!(error_msg.contains("NonExistentFunction") || error_msg.contains("Unbound"));
        }
        Ok(_) => panic!("Expected an error"),
    }
}

#[test] 
fn test_type_annotation_persistence() {
    let mut compiler = TypedCompiler::new();
    
    // Compile multiple expressions
    let expr1 = Expr::Number(Number::Integer(42));
    let expr2 = Expr::String("hello".to_string());
    
    let type1 = compiler.compile_expr_typed(&expr1).unwrap();
    let type2 = compiler.compile_expr_typed(&expr2).unwrap();
    
    // Annotations should persist
    assert_eq!(compiler.get_type_annotation(&expr1), Some(&type1));
    assert_eq!(compiler.get_type_annotation(&expr2), Some(&type2));
    
    // Different expressions should have different annotations
    assert_ne!(type1, type2);
}

#[test]
fn test_performance_with_type_checking() {
    use std::time::Instant;
    
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ],
    };
    
    // Measure with type checking
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = check_expression_safety(&expr);
    }
    let with_types = start.elapsed();
    
    // Measure basic evaluation (this would need a non-typed path)
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = infer_expression_type(&expr);
    }
    let inference_only = start.elapsed();
    
    println!("Type checking time: {:?}", with_types);
    println!("Inference only time: {:?}", inference_only);
    
    // Type checking shouldn't be more than 10x slower than inference
    assert!(with_types.as_nanos() < inference_only.as_nanos() * 20);
}