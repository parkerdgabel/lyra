//! Performance Benchmarks for Fast-Path Pattern Matching System
//! 
//! Phase 6B.5.1d.2b.4: Performance Validation & Benchmarking
//! 
//! This benchmark suite validates the performance improvements achieved by the
//! integrated fast-path pattern matching system, targeting 15-25% improvement.
//!
//! System Components Tested:
//! 1. Fast-path matchers (Sub-Phase 2b.1) 
//! 2. Pattern routing system (Sub-Phase 2b.2)
//! 3. Integration with compilation system (Sub-Phase 2b.3)

use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use lyra::{
    ast::{Expr, Pattern},
    pattern_matcher::PatternMatcher,
};

/// Benchmark fast-path vs standard pattern matching performance
fn fast_path_vs_standard_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("fast_path_vs_standard");
    
    // Trivial blank pattern - should show maximum fast-path benefit
    group.bench_function("trivial_blank_standard", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::with_fast_path_disabled(), Expr::integer(42), Pattern::Blank { head: None })
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("trivial_blank_fast_path", |b| {
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
    
    // Typed blank pattern - common fast-path case
    group.bench_function("typed_blank_standard", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::with_fast_path_disabled(), Expr::integer(42), Pattern::Blank { head: Some("Integer".to_string()) })
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("typed_blank_fast_path", |b| {
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
    
    // Named pattern - should benefit from fast-path
    group.bench_function("named_pattern_standard", |b| {
        b.iter_batched(
            || {
                (
                    PatternMatcher::with_fast_path_disabled(),
                    Expr::symbol("test"),
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
    
    group.bench_function("named_pattern_fast_path", |b| {
        b.iter_batched(
            || {
                (
                    PatternMatcher::new(),
                    Expr::symbol("test"),
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
    
    // Simple function pattern - fast-path eligible
    group.bench_function("simple_function_standard", |b| {
        b.iter_batched(
            || {
                let expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("x"), Expr::integer(0)]);
                let pattern = Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                    args: vec![
                        Pattern::Named {
                            name: "var".to_string(),
                            pattern: Box::new(Pattern::Blank { head: None }),
                        },
                        Pattern::Blank { head: Some("Integer".to_string()) },
                    ],
                };
                (PatternMatcher::with_fast_path_disabled(), expr, pattern)
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("simple_function_fast_path", |b| {
        b.iter_batched(
            || {
                let expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("x"), Expr::integer(0)]);
                let pattern = Pattern::Function {
                    head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
                    args: vec![
                        Pattern::Named {
                            name: "var".to_string(),
                            pattern: Box::new(Pattern::Blank { head: None }),
                        },
                        Pattern::Blank { head: Some("Integer".to_string()) },
                    ],
                };
                (PatternMatcher::new(), expr, pattern)
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Benchmark mixed workload representing realistic usage patterns
fn mixed_workload_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    
    // Create a mixed workload of patterns that should benefit from fast-path optimization
    let create_mixed_workload = || {
        vec![
            // 40% trivial patterns (should get maximum fast-path benefit)
            (Expr::integer(42), Pattern::Blank { head: None }),
            (Expr::integer(100), Pattern::Blank { head: None }),
            (Expr::symbol("x"), Pattern::Blank { head: None }),
            (Expr::real(3.14), Pattern::Blank { head: None }),
            
            // 30% typed patterns (should get good fast-path benefit)
            (Expr::integer(42), Pattern::Blank { head: Some("Integer".to_string()) }),
            (Expr::symbol("test"), Pattern::Blank { head: Some("Symbol".to_string()) }),
            (Expr::real(2.5), Pattern::Blank { head: Some("Real".to_string()) }),
            
            // 20% named patterns (should get moderate fast-path benefit)
            (
                Expr::symbol("var"),
                Pattern::Named {
                    name: "x".to_string(),
                    pattern: Box::new(Pattern::Blank { head: None })
                }
            ),
            (
                Expr::integer(5),
                Pattern::Named {
                    name: "y".to_string(),
                    pattern: Box::new(Pattern::Blank { head: Some("Integer".to_string()) })
                }
            ),
            
            // 10% complex patterns (should fall back to standard matching)
            (
                Expr::integer(10),
                Pattern::Conditional {
                    pattern: Box::new(Pattern::Named {
                        name: "z".to_string(),
                        pattern: Box::new(Pattern::Blank { head: None }),
                    }),
                    condition: Box::new(Expr::function(
                        Expr::symbol("Greater"),
                        vec![Expr::symbol("z"), Expr::integer(0)]
                    )),
                }
            ),
        ]
    };
    
    group.bench_function("mixed_workload_standard", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::with_fast_path_disabled(), create_mixed_workload())
            },
            |(mut matcher, workload)| {
                for (expr, pattern) in workload {
                    let _ = matcher.match_pattern(black_box(&expr), black_box(&pattern));
                    matcher.clear_bindings();
                }
            },
            BatchSize::SmallInput
        );
    });
    
    group.bench_function("mixed_workload_fast_path", |b| {
        b.iter_batched(
            || {
                (PatternMatcher::new(), create_mixed_workload())
            },
            |(mut matcher, workload)| {
                for (expr, pattern) in workload {
                    let _ = matcher.match_pattern(black_box(&expr), black_box(&pattern));
                    matcher.clear_bindings();
                }
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Benchmark pattern complexity routing efficiency
fn routing_efficiency_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("routing_efficiency");
    
    // Test patterns at different complexity levels
    let complexity_patterns = vec![
        // Trivial complexity - should be routed to fast-path immediately
        ("trivial", Pattern::Blank { head: None }),
        
        // Simple complexity - should be routed to fast-path
        ("simple", Pattern::Named {
            name: "x".to_string(),
            pattern: Box::new(Pattern::Blank { head: None }),
        }),
        
        // Moderate complexity - routing decision based on strategy
        ("moderate", Pattern::Function {
            head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
            args: vec![
                Pattern::Blank { head: None },
                Pattern::Blank { head: None },
            ],
        }),
        
        // Complex complexity - should be routed to standard matching
        ("complex", Pattern::BlankSequence { head: None }),
        
        // Very complex - should definitely use standard matching
        ("very_complex", Pattern::Conditional {
            pattern: Box::new(Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }),
            condition: Box::new(Expr::symbol("True")),
        }),
    ];
    
    let expr = Expr::integer(42);
    
    for (complexity_name, pattern) in complexity_patterns {
        group.bench_function(&format!("{}_routing_overhead", complexity_name), |b| {
            b.iter_batched(
                || {
                    (PatternMatcher::new(), expr.clone(), pattern.clone())
                },
                |(mut matcher, expr, pattern)| {
                    matcher.match_pattern(black_box(&expr), black_box(&pattern))
                },
                BatchSize::SmallInput
            );
        });
    }
    
    group.finish();
}

/// Benchmark pattern compilation integration
fn compilation_integration_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_integration");
    
    // Test compilation with and without fast-path routing
    let test_pattern = Pattern::Function {
        head: Box::new(Pattern::Blank { head: Some("Plus".to_string()) }),
        args: vec![
            Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            },
            Pattern::Blank { head: Some("Integer".to_string()) },
        ],
    };
    
    let expr = Expr::function(Expr::symbol("Plus"), vec![Expr::symbol("y"), Expr::integer(5)]);
    
    // Standard matcher with compilation disabled
    group.bench_function("no_compilation_no_fast_path", |b| {
        b.iter_batched(
            || {
                let mut matcher = PatternMatcher::with_fast_path_disabled();
                // Compilation is controlled by the matcher configuration
                (matcher, expr.clone(), test_pattern.clone())
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    // Fast-path system with compilation integration
    group.bench_function("compilation_with_fast_path", |b| {
        b.iter_batched(
            || {
                let matcher = PatternMatcher::new(); // Uses both compilation and fast-path
                (matcher, expr.clone(), test_pattern.clone())
            },
            |(mut matcher, expr, pattern)| {
                matcher.match_pattern(black_box(&expr), black_box(&pattern))
            },
            BatchSize::SmallInput
        );
    });
    
    group.finish();
}

/// Performance regression test to ensure fast-path doesn't slow down complex patterns
fn regression_prevention_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_prevention");
    
    // Complex patterns that should NOT use fast-path but should have minimal routing overhead
    let complex_patterns = vec![
        Pattern::BlankSequence { head: None },
        Pattern::Alternative {
            patterns: vec![
                Pattern::Blank { head: Some("Integer".to_string()) },
                Pattern::Blank { head: Some("Real".to_string()) },
            ],
        },
        Pattern::Conditional {
            pattern: Box::new(Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None }),
            }),
            condition: Box::new(Expr::function(
                Expr::symbol("Greater"),
                vec![Expr::symbol("x"), Expr::integer(0)]
            )),
        },
    ];
    
    let expr = Expr::integer(42);
    
    for (i, pattern) in complex_patterns.into_iter().enumerate() {
        group.bench_function(&format!("complex_pattern_{}_standard", i), |b| {
            b.iter_batched(
                || {
                    (PatternMatcher::with_fast_path_disabled(), expr.clone(), pattern.clone())
                },
                |(mut matcher, expr, pattern)| {
                    matcher.match_pattern(black_box(&expr), black_box(&pattern))
                },
                BatchSize::SmallInput
            );
        });
        
        group.bench_function(&format!("complex_pattern_{}_with_routing", i), |b| {
            b.iter_batched(
                || {
                    (PatternMatcher::new(), expr.clone(), pattern.clone())
                },
                |(mut matcher, expr, pattern)| {
                    matcher.match_pattern(black_box(&expr), black_box(&pattern))
                },
                BatchSize::SmallInput
            );
        });
    }
    
    group.finish();
}

criterion_group!(
    fast_path_benches,
    fast_path_vs_standard_benchmark,
    mixed_workload_benchmark,
    routing_efficiency_benchmark,
    compilation_integration_benchmark,
    regression_prevention_benchmark
);

criterion_main!(fast_path_benches);