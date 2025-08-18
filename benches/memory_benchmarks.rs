use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use lyra::{
    ast::{Expr, Pattern},
    pattern_matcher::PatternMatcher,
    rules_engine::{RuleEngine, Rule}
};

/// Memory allocation benchmark for pattern matching operations
fn memory_pattern_matching_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pattern_matching");
    
    // Basic blank pattern - minimal allocations expected (baseline)
    group.bench_function("memory_basic_blank_pattern", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::new(), Expr::integer(42), Pattern::Blank { head: None })
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Typed blank pattern - String allocation for type checking
    group.bench_function("memory_typed_blank_pattern", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::new(), Expr::integer(42), Pattern::Blank { head: Some("Integer".to_string()) })
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Variable binding - HashMap allocation for variable storage
    group.bench_function("memory_variable_binding", |b| {
        b.iter_batched(
            || {
                (
                    PatternMatcher::new(),
                    Expr::integer(42),
                    Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None })
                    }
                )
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Conditional pattern - heavy allocations: condition evaluation + variable binding
    group.bench_function("memory_conditional_pattern_simple", |b| {
        b.iter_batched(
            || {
                let condition = Expr::function(
                    Expr::symbol("Greater"),
                    vec![Expr::symbol("x"), Expr::integer(0)]
                );
                let pattern = Pattern::Conditional {
                    pattern: Box::new(Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None })
                    }),
                    condition: Box::new(condition)
                };
                (PatternMatcher::new(), Expr::integer(5), pattern)
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Complex conditional pattern - maximum allocations: multiple variables + arithmetic evaluation
    group.bench_function("memory_conditional_pattern_complex", |b| {
        b.iter_batched(
            || {
                let condition = Expr::function(
                    Expr::symbol("Equal"),
                    vec![
                        Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("a"), Expr::symbol("b")]),
                        Expr::integer(10)
                    ]
                );
                let pattern = Pattern::Conditional {
                    pattern: Box::new(Pattern::Function {
                        head: Box::new(Pattern::Blank { head: Some("List".to_string()) }),
                        args: vec![
                            Pattern::Named {
                                name: "a".to_string(),
                                pattern: Box::new(Pattern::Blank { head: None })
                            },
                            Pattern::Named {
                                name: "b".to_string(),
                                pattern: Box::new(Pattern::Blank { head: None })
                            }
                        ]
                    }),
                    condition: Box::new(condition)
                };
                (PatternMatcher::new(), Expr::list(vec![Expr::integer(3), Expr::integer(7)]), pattern)
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Sequence pattern allocation analysis - Vec operations for sequence matching
    group.bench_function("memory_sequence_pattern_matching", |b| {
        b.iter_batched(
            || {
                (
                    PatternMatcher::new(),
                    Expr::list(vec![
                        Expr::integer(1), Expr::integer(2), Expr::integer(3), Expr::integer(4)
                    ]),
                    Pattern::BlankSequence { head: None }
                )
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Memory allocation benchmark for rule engine operations
fn memory_rule_engine_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_rule_engine");
    
    // Simple rule application - baseline: variable binding + expression creation
    group.bench_function("memory_simple_rule_application", |b| {
        b.iter_batched(
            || {
                let rule = Rule::immediate(
                    Pattern::Named {
                        name: "x".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None })
                    },
                    Expr::function(Expr::symbol("Power"), vec![Expr::symbol("x"), Expr::integer(2)])
                );
                (RuleEngine::new(), Expr::integer(42), rule)
            },
            |(mut engine, expr, rule)| {
                engine.apply_rule(black_box(&expr), black_box(&rule)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    // Mathematical simplification - function pattern matching + variable substitution
    group.bench_function("memory_mathematical_simplification", |b| {
        b.iter_batched(
            || {
                let rule = Rule::immediate(
                    Pattern::Function {
                        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                        args: vec![
                            Pattern::Named {
                                name: "x".to_string(),
                                pattern: Box::new(Pattern::Blank { head: None })
                            },
                            Pattern::Blank { head: Some("Integer".to_string()) }
                        ]
                    },
                    Expr::symbol("x")
                );
                let expr = Expr::function(
                    Expr::symbol("Plus"),
                    vec![Expr::symbol("x"), Expr::integer(0)]
                );
                (RuleEngine::new(), expr, rule)
            },
            |(mut engine, expr, rule)| {
                engine.apply_rule(black_box(&expr), black_box(&rule)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    // Conditional rule application - high allocation: condition evaluation + binding
    group.bench_function("memory_conditional_rule_application", |b| {
        b.iter_batched(
            || {
                let rule = Rule::immediate(
                    Pattern::Conditional {
                        pattern: Box::new(Pattern::Named {
                            name: "x".to_string(),
                            pattern: Box::new(Pattern::Blank { head: None })
                        }),
                        condition: Box::new(Expr::function(
                            Expr::symbol("Greater"),
                            vec![Expr::symbol("x"), Expr::integer(0)]
                        ))
                    },
                    Expr::function(Expr::symbol("Positive"), vec![Expr::symbol("x")])
                );
                (RuleEngine::new(), Expr::integer(5), rule)
            },
            |(mut engine, expr, rule)| {
                engine.apply_rule(black_box(&expr), black_box(&rule)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    // Multiple rules sequential - memory accumulation analysis
    group.bench_function("memory_multiple_rules_sequential", |b| {
        let mut engine = RuleEngine::new();
        let expr = Expr::function(
            Expr::symbol("Plus"),
            vec![
                Expr::function(Expr::symbol("Times"), vec![Expr::symbol("x"), Expr::integer(0)]),
                Expr::integer(5)
            ]
        );
        let rules = vec![
            Rule::immediate(
                Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                    args: vec![
                        Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                        Pattern::Blank { head: Some("Integer".to_string()) }
                    ]
                },
                Expr::integer(0)
            ),
            Rule::immediate(
                Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                    args: vec![
                        Pattern::Blank { head: Some("Integer".to_string()) },
                        Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }
                    ]
                },
                Expr::symbol("x")
            ),
            Rule::immediate(
                Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                    args: vec![
                        Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                        Pattern::Blank { head: Some("Integer".to_string()) }
                    ]
                },
                Expr::symbol("x")
            )
        ];
        
        b.iter_batched(
            || {
                let expr = Expr::function(
                    Expr::symbol("Plus"),
                    vec![
                        Expr::function(Expr::symbol("Times"), vec![Expr::symbol("x"), Expr::integer(0)]),
                        Expr::integer(5)
                    ]
                );
                let rules = vec![
                    Rule::immediate(
                        Pattern::Function {
                            head: Box::new(Pattern::Blank { head: Some("Times".to_string()) }),
                            args: vec![
                                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                                Pattern::Blank { head: Some("Integer".to_string()) }
                            ]
                        },
                        Expr::integer(0)
                    ),
                    Rule::immediate(
                        Pattern::Function {
                            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                            args: vec![
                                Pattern::Blank { head: Some("Integer".to_string()) },
                                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) }
                            ]
                        },
                        Expr::symbol("x")
                    ),
                    Rule::immediate(
                        Pattern::Function {
                            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                            args: vec![
                                Pattern::Named { name: "x".to_string(), pattern: Box::new(Pattern::Blank { head: None }) },
                                Pattern::Blank { head: Some("Integer".to_string()) }
                            ]
                        },
                        Expr::symbol("x")
                    )
                ];
                (RuleEngine::new(), expr, rules)
            },
            |(mut engine, expr, rules)| {
                engine.apply_rules(black_box(&expr), black_box(&rules)).unwrap()
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Memory allocation benchmark for expression creation and manipulation
fn memory_expression_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_expression");
    
    // Integer expression creation - minimal allocations (no heap allocation)
    group.bench_function("memory_integer_creation", |b| {
        b.iter(|| {
            black_box(Expr::integer(42))
        });
    });
    
    // Symbol expression creation - String allocation for symbol name
    group.bench_function("memory_symbol_creation", |b| {
        b.iter(|| {
            black_box(Expr::symbol("x"))
        });
    });
    
    // Function expression creation - Vec allocation + recursive expressions
    group.bench_function("memory_function_creation", |b| {
        b.iter(|| {
            black_box(Expr::function(
                Expr::symbol("Plus"),
                vec![Expr::integer(1), Expr::integer(2), Expr::integer(3)]
            ))
        });
    });
    
    // List expression creation - Vec allocation for list elements
    group.bench_function("memory_list_creation", |b| {
        b.iter(|| {
            black_box(Expr::list(vec![
                Expr::integer(1), Expr::integer(2), Expr::integer(3), Expr::integer(4), Expr::integer(5)
            ]))
        });
    });
    
    // Complex nested expression - maximum allocations
    group.bench_function("memory_complex_nested_expression", |b| {
        b.iter(|| {
            black_box(Expr::function(
                Expr::symbol("Plus"),
                vec![
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![
                            Expr::integer(2),
                            Expr::function(
                                Expr::symbol("Power"),
                                vec![Expr::symbol("x"), Expr::integer(2)]
                            )
                        ]
                    ),
                    Expr::function(
                        Expr::symbol("Times"),
                        vec![Expr::integer(3), Expr::symbol("x")]
                    ),
                    Expr::integer(1)
                ]
            ))
        });
    });
    
    group.finish();
}


criterion_group!(
    memory_benches,
    memory_pattern_matching_benchmark,
    memory_rule_engine_benchmark,
    memory_expression_benchmark
);
criterion_main!(memory_benches);