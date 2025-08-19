//! Performance Regression Detection Benchmarks
//!
//! This module provides automated performance regression detection for critical
//! paths in the Lyra system. It establishes baseline performance metrics and
//! can detect when changes impact performance negatively.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize, BenchmarkId};
use lyra::{
    vm::VirtualMachine,
    parser::Parser,
    compiler::Compiler,
    stdlib::StandardLibrary,
    ast::Expr,
    vm::Value,
    repl::ReplEngine,
};
use std::time::Instant;

/// Critical path benchmarks that must maintain performance
fn critical_path_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("critical_paths");
    group.sample_size(1000); // High sample size for accuracy
    
    // Parser critical paths
    let parser_test_cases = vec![
        ("arithmetic", "2 + 3 * 4 - 1 / 2"),
        ("function_call", "Length[{1, 2, 3, 4, 5}]"),
        ("nested_expr", "Head[Tail[Rest[{1, 2, 3, 4, 5}]]]"),
        ("pattern_def", "f[x_] := x^2 + 3*x + 1"),
        ("rule_apply", "expr /. x -> 5"),
        ("list_ops", "Flatten[{{1, 2}, {3, 4}, {5, 6}}]"),
        ("math_expr", "Sin[Pi/4] + Cos[Pi/6] * Tan[Pi/3]"),
    ];
    
    for (name, source) in parser_test_cases {
        group.bench_function(BenchmarkId::new("parser", name), |b| {
            b.iter(|| {
                let mut parser = Parser::from_source(black_box(source)).unwrap();
                parser.parse().unwrap()
            });
        });
    }
    
    group.finish();
}

/// VM execution critical paths  
fn vm_execution_critical_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_critical_paths");
    group.sample_size(1000);
    
    let execution_test_cases = vec![
        ("basic_math", "2 + 3 * 4"),
        ("stdlib_call", "Length[{1, 2, 3, 4, 5}]"),
        ("chained_calls", "Head[Tail[{1, 2, 3, 4}]]"),
        ("math_functions", "Sin[0.5] + Cos[0.5]"),
        ("list_creation", "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}"),
    ];
    
    for (name, source) in execution_test_cases {
        group.bench_function(BenchmarkId::new("vm_execution", name), |b| {
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let mut compiler = Compiler::new();
            let bytecode = compiler.compile(&statements).unwrap();
            
            b.iter(|| {
                let mut vm = VirtualMachine::new();
                vm.execute(black_box(&bytecode))
            });
        });
    }
    
    group.finish();
}

/// Compilation performance critical paths
fn compilation_critical_paths(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_critical_paths");
    group.sample_size(500);
    
    let compilation_test_cases = vec![
        ("simple", "x + y"),
        ("moderate", "f[x_] := x^2 + 3*x + 1"),
        ("complex", "Integrate[x^2 + 3*x + 1, x]"),
        ("pattern_heavy", "expr /. {x -> a, y -> b, z -> c}"),
        ("nested_functions", "Map[Function[x, x^2], Range[1, 10]]"),
    ];
    
    for (name, source) in compilation_test_cases {
        group.bench_function(BenchmarkId::new("compilation", name), |b| {
            let mut parser = Parser::from_source(source).unwrap();
            let ast = parser.parse().unwrap();
            
            b.iter(|| {
                let mut compiler = Compiler::new();
                compiler.compile(black_box(&ast))
            });
        });
    }
    
    group.finish();
}

/// Standard library function performance baselines
fn stdlib_performance_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("stdlib_baselines");
    group.sample_size(2000);
    
    let stdlib_functions = vec![
        ("Length", vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])]),
        ("Head", vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])]),
        ("Tail", vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])]),
        ("Append", vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::Integer(3)
        ]),
        ("Map", vec![
            Value::Symbol("Identity".to_string()),
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])
        ]),
    ];
    
    let stdlib = StandardLibrary::new().unwrap();
    
    for (func_name, args) in stdlib_functions {
        group.bench_function(BenchmarkId::new("stdlib", func_name), |b| {
            b.iter(|| {
                stdlib.call_function(black_box(func_name), black_box(&args))
            });
        });
    }
    
    group.finish();
}

/// REPL performance baselines (interactive usage)
fn repl_performance_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("repl_baselines");
    group.sample_size(200); // Lower sample size for interactive scenarios
    
    let repl_interactions = vec![
        "2 + 3",
        "Length[{1, 2, 3}]",
        "x = 5",
        "f[x_] := x^2",
        "f[10]",
        "Plot[Sin[x], {x, 0, 2*Pi}]", // More complex interaction
    ];
    
    for input in repl_interactions {
        group.bench_function(BenchmarkId::new("repl", input), |b| {
            b.iter_batched(
                || ReplEngine::new(),
                |mut repl| {
                    repl.evaluate(black_box(input))
                },
                BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

/// Memory usage regression detection
fn memory_regression_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_regression");
    group.sample_size(100); // Lower sample for memory-intensive tests
    
    // Large data structure handling
    group.bench_function("large_list_handling", |b| {
        b.iter_batched(
            || {
                // Create large list
                let mut elements = Vec::new();
                for i in 0..10000 {
                    elements.push(Value::Integer(i));
                }
                Value::List(elements)
            },
            |large_list| {
                // Operations that shouldn't cause memory issues
                match &large_list {
                    Value::List(elements) => {
                        black_box(elements.len());
                        black_box(elements.first());
                        black_box(elements.last());
                        // Test that we can iterate without issues
                        let sum: i64 = elements.iter()
                            .filter_map(|v| match v {
                                Value::Integer(i) => Some(*i),
                                _ => None,
                            })
                            .take(1000)
                            .sum();
                        black_box(sum);
                    }
                    _ => {}
                }
            },
            BatchSize::LargeInput,
        );
    });
    
    // Deep nesting test
    group.bench_function("deep_nesting_handling", |b| {
        b.iter_batched(
            || {
                // Create deeply nested structure
                let mut current = Value::Integer(0);
                for _ in 0..100 {
                    current = Value::List(vec![current]);
                }
                current
            },
            |nested_structure| {
                // Navigate through the structure
                let mut current = &nested_structure;
                let mut depth = 0;
                while let Value::List(elements) = current {
                    if let Some(first) = elements.first() {
                        current = first;
                        depth += 1;
                        if depth > 50 { break; } // Prevent infinite loops
                    } else {
                        break;
                    }
                }
                black_box(depth)
            },
            BatchSize::SmallInput,
        );
    });
    
    group.finish();
}

/// Pattern matching regression detection
fn pattern_matching_regression_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_regression");
    group.sample_size(500);
    
    let pattern_test_cases = vec![
        ("simple_var", "f[x_]", "f[5]"),
        ("typed_pattern", "g[x_Integer]", "g[42]"),
        ("sequence_pattern", "h[x__, y_]", "h[1, 2, 3, 4]"),
        ("nested_pattern", "i[f[x_], y_]", "i[f[3], 7]"),
        ("conditional_pattern", "j[x_?Positive]", "j[5]"),
    ];
    
    for (name, pattern_src, test_src) in pattern_test_cases {
        group.bench_function(BenchmarkId::new("pattern", name), |b| {
            let mut parser = Parser::from_source(pattern_src).unwrap();
            let pattern = parser.parse().unwrap();
            
            let mut test_parser = Parser::from_source(test_src).unwrap();
            let test_expr = test_parser.parse().unwrap();
            
            b.iter(|| {
                // Simulate pattern matching
                match (&pattern[0], &test_expr[0]) {
                    (Expr::Function { head: p_head, args: p_args },
                     Expr::Function { head: t_head, args: t_args }) => {
                        black_box(p_head == t_head);
                        black_box(p_args.len() <= t_args.len());
                    }
                    _ => {}
                }
            });
        });
    }
    
    group.finish();
}

/// Comprehensive end-to-end regression test
fn end_to_end_regression_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_regression");
    group.sample_size(100);
    
    let complete_programs = vec![
        ("basic_computation", r#"
            f[x_] := x^2 + 3*x + 1;
            result = f[5];
            Length[{result, f[10], f[15]}]
        "#),
        ("list_processing", r#"
            data = Range[1, 100];
            filtered = Select[data, # > 50 &];
            Length[filtered]
        "#),
        ("mathematical", r#"
            result = Sin[Pi/4] + Cos[Pi/6] * Tan[Pi/3];
            Round[result, 0.001]
        "#),
    ];
    
    for (name, program) in complete_programs {
        group.bench_function(BenchmarkId::new("end_to_end", name), |b| {
            b.iter(|| {
                let start = Instant::now();
                
                let mut parser = Parser::from_source(black_box(program)).unwrap();
                let ast = parser.parse().unwrap();
                
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile(&ast).unwrap();
                
                let mut vm = VirtualMachine::new();
                let _result = vm.execute(&bytecode);
                
                black_box(start.elapsed())
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    regression_detection,
    critical_path_benchmarks,
    vm_execution_critical_paths,
    compilation_critical_paths,
    stdlib_performance_baselines,
    repl_performance_baselines,
    memory_regression_detection,
    pattern_matching_regression_detection,
    end_to_end_regression_test
);

criterion_main!(regression_detection);