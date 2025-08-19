use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lyra::{
    ast::{Expr, Number, Symbol},
    types::{
        TypedCompiler, TypeChecker, TypeInferenceEngine, 
        infer_expression_type, check_expression_safety,
        eval_with_type, TypeEnvironment, TypeScheme, LyraType,
    },
    compiler::Compiler,
};

fn create_arithmetic_expr(depth: usize) -> Expr {
    if depth == 0 {
        Expr::Number(Number::Integer(42))
    } else {
        Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                create_arithmetic_expr(depth - 1),
                create_arithmetic_expr(depth - 1),
            ],
        }
    }
}

fn create_list_expr(size: usize) -> Expr {
    let elements = (0..size)
        .map(|i| Expr::Number(Number::Integer(i as i64)))
        .collect();
    Expr::List(elements)
}

fn create_nested_function_expr(depth: usize) -> Expr {
    if depth == 0 {
        Expr::Symbol(Symbol { name: "x".to_string() })
    } else {
        Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
            args: vec![create_nested_function_expr(depth - 1)],
        }
    }
}

fn benchmark_type_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_inference");
    
    // Simple expressions
    let simple_exprs = vec![
        ("integer", Expr::Number(Number::Integer(42))),
        ("real", Expr::Number(Number::Real(3.14))),
        ("string", Expr::String("hello".to_string())),
        ("symbol", Expr::Symbol(Symbol { name: "x".to_string() })),
    ];
    
    for (name, expr) in simple_exprs {
        group.bench_with_input(
            BenchmarkId::new("simple", name),
            &expr,
            |b, expr| {
                b.iter(|| {
                    black_box(infer_expression_type(black_box(expr)).unwrap())
                })
            },
        );
    }
    
    // Arithmetic expressions of varying depth
    for depth in [1, 3, 5, 7].iter() {
        let expr = create_arithmetic_expr(*depth);
        group.bench_with_input(
            BenchmarkId::new("arithmetic", depth),
            &expr,
            |b, expr| {
                b.iter(|| {
                    black_box(infer_expression_type(black_box(expr)).unwrap())
                })
            },
        );
    }
    
    // List expressions of varying size
    for size in [10, 50, 100, 500].iter() {
        let expr = create_list_expr(*size);
        group.bench_with_input(
            BenchmarkId::new("list", size),
            &expr,
            |b, expr| {
                b.iter(|| {
                    black_box(infer_expression_type(black_box(expr)).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_type_checking(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_checking");
    
    // Valid arithmetic expressions
    let valid_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ],
    };
    
    group.bench_function("valid_arithmetic", |b| {
        b.iter(|| {
            black_box(check_expression_safety(black_box(&valid_expr)).unwrap())
        })
    });
    
    // Mathematical function call
    let math_expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
        args: vec![Expr::Number(Number::Real(3.14))],
    };
    
    group.bench_function("math_function", |b| {
        b.iter(|| {
            black_box(check_expression_safety(black_box(&math_expr)).unwrap())
        })
    });
    
    // Nested function calls
    for depth in [2, 4, 6].iter() {
        let expr = create_nested_function_expr(*depth);
        group.bench_with_input(
            BenchmarkId::new("nested", depth),
            &expr,
            |b, expr| {
                b.iter(|| {
                    // This might fail for unbound variables, so we'll use inference instead
                    black_box(infer_expression_type(black_box(expr)).unwrap_or(LyraType::Unknown))
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_typed_compilation(c: &mut Criterion) {
    let mut group = c.benchmark_group("typed_compilation");
    
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(5)),
            Expr::Number(Number::Integer(7)),
        ],
    };
    
    group.bench_function("with_types", |b| {
        b.iter(|| {
            let mut compiler = TypedCompiler::new();
            black_box(compiler.compile_expr_typed(black_box(&expr)).unwrap())
        })
    });
    
    group.bench_function("without_types", |b| {
        b.iter(|| {
            let mut compiler = Compiler::new();
            black_box(compiler.compile_expr(black_box(&expr)).unwrap())
        })
    });
    
    // Comparison: typed vs untyped compilation
    group.bench_function("typed_vs_untyped", |b| {
        b.iter(|| {
            // Typed compilation
            let mut typed_compiler = TypedCompiler::new();
            let _typed_result = black_box(typed_compiler.compile_expr_typed(black_box(&expr)).unwrap());
            
            // Untyped compilation  
            let mut untyped_compiler = Compiler::new();
            let _untyped_result = black_box(untyped_compiler.compile_expr(black_box(&expr)).unwrap());
        })
    });
    
    group.finish();
}

fn benchmark_eval_with_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("eval_with_types");
    
    let expr = Expr::Function {
        head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
        args: vec![
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ],
    };
    
    group.bench_function("eval_with_type", |b| {
        b.iter(|| {
            black_box(eval_with_type(black_box(&expr)).unwrap())
        })
    });
    
    group.bench_function("eval_without_type", |b| {
        b.iter(|| {
            black_box(Compiler::eval(black_box(&expr)).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_type_environment(c: &mut Criterion) {
    let mut group = c.benchmark_group("type_environment");
    
    // Environment with many bindings
    let mut env = TypeEnvironment::new();
    for i in 0..100 {
        env.insert(
            format!("var{}", i),
            TypeScheme::monomorphic(if i % 2 == 0 { LyraType::Integer } else { LyraType::Real }),
        );
    }
    
    let var_expr = Expr::Symbol(Symbol { name: "var50".to_string() });
    
    group.bench_function("lookup_in_large_env", |b| {
        b.iter(|| {
            let mut engine = TypeInferenceEngine::new();
            black_box(engine.infer_expr(black_box(&var_expr), black_box(&env)).unwrap())
        })
    });
    
    // Environment copying for scoping
    group.bench_function("env_extension", |b| {
        b.iter(|| {
            let extended = black_box(&env).extend(
                "new_var".to_string(),
                TypeScheme::monomorphic(LyraType::String),
            );
            black_box(extended)
        })
    });
    
    group.finish();
}

fn benchmark_complex_expressions(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_expressions");
    
    // Complex nested expression: Plus[Times[2, 3], Divide[Sin[3.14], Cos[1.57]]]
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
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
                        args: vec![Expr::Number(Number::Real(3.14))],
                    },
                    Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Cos".to_string() })),
                        args: vec![Expr::Number(Number::Real(1.57))],
                    },
                ],
            },
        ],
    };
    
    group.bench_function("complex_inference", |b| {
        b.iter(|| {
            black_box(infer_expression_type(black_box(&complex_expr)).unwrap())
        })
    });
    
    group.bench_function("complex_checking", |b| {
        b.iter(|| {
            black_box(check_expression_safety(black_box(&complex_expr)).unwrap())
        })
    });
    
    group.bench_function("complex_eval", |b| {
        b.iter(|| {
            black_box(eval_with_type(black_box(&complex_expr)).unwrap())
        })
    });
    
    group.finish();
}

fn benchmark_arrow_functions(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_functions");
    
    // Identity function: (x) => x
    let identity = Expr::ArrowFunction {
        params: vec!["x".to_string()],
        body: Box::new(Expr::Symbol(Symbol { name: "x".to_string() })),
    };
    
    group.bench_function("identity_inference", |b| {
        b.iter(|| {
            black_box(infer_expression_type(black_box(&identity)).unwrap())
        })
    });
    
    // Complex arrow function: (x, y) => Plus[Times[x, x], Times[y, y]]
    let complex_arrow = Expr::ArrowFunction {
        params: vec!["x".to_string(), "y".to_string()],
        body: Box::new(Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                    args: vec![
                        Expr::Symbol(Symbol { name: "x".to_string() }),
                        Expr::Symbol(Symbol { name: "x".to_string() }),
                    ],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Times".to_string() })),
                    args: vec![
                        Expr::Symbol(Symbol { name: "y".to_string() }),
                        Expr::Symbol(Symbol { name: "y".to_string() }),
                    ],
                },
            ],
        }),
    };
    
    group.bench_function("complex_arrow_inference", |b| {
        b.iter(|| {
            black_box(infer_expression_type(black_box(&complex_arrow)).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_type_inference,
    benchmark_type_checking,
    benchmark_typed_compilation,
    benchmark_eval_with_types,
    benchmark_type_environment,
    benchmark_complex_expressions,
    benchmark_arrow_functions
);
criterion_main!(benches);