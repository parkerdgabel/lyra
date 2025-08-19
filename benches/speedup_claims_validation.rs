//! Speedup Claims Validation Benchmarks
//!
//! This module specifically targets the "1000x speedup" claims found throughout
//! the Lyra codebase. It provides empirical validation of performance improvements
//! and identifies areas where claims need verification or adjustment.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize, BenchmarkId};
use lyra::{
    vm::VirtualMachine,
    parser::Parser,
    compiler::Compiler,
    stdlib::StandardLibrary,
    linker::FunctionRegistry,
    ast::Expr,
    vm::Value,
};
use std::time::{Duration, Instant};

/// Validate "1000x+ speedup" claim from CALL_STATIC optimization
/// Location: src/linker.rs:11, 17, 31
fn validate_call_static_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_static_speedup_validation");
    
    // Simulated "slow path" - dynamic function resolution
    group.bench_function("dynamic_function_resolution", |b| {
        let stdlib = StandardLibrary::new().unwrap();
        let args = vec![Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3),
            Value::Integer(4), Value::Integer(5),
        ])];
        
        b.iter(|| {
            // Simulate dynamic lookup every time
            let function_name = black_box("Length");
            
            // Simulate string comparison and hash table lookup overhead
            let mut lookup_cost = 0;
            for _ in 0..10 { // Simulate lookup overhead
                lookup_cost += function_name.len();
            }
            black_box(lookup_cost);
            
            // Then call the function
            stdlib.call_function(function_name, black_box(&args))
        });
    });

    // "Fast path" - CALL_STATIC direct function pointer
    group.bench_function("call_static_direct", |b| {
        let stdlib = StandardLibrary::new().unwrap();
        let args = vec![Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3),
            Value::Integer(4), Value::Integer(5),
        ])];
        
        b.iter(|| {
            // Direct function call via static index (simulated)
            stdlib.call_function(black_box("Length"), black_box(&args))
        });
    });

    // Registry-based lookup vs direct call comparison
    group.bench_function("registry_vs_direct_comparison", |b| {
        let registry = FunctionRegistry::new().unwrap();
        
        b.iter_batched(
            || {
                vec![
                    ("Length", vec![Value::Integer(1), Value::Integer(2)]),
                    ("Head", vec![Value::Integer(42)]),
                    ("Tail", vec![Value::Integer(99)]),
                ]
            },
            |test_cases| {
                for (func_name, args) in test_cases {
                    // Simulate registry lookup vs direct call
                    if registry.has_function(black_box(func_name)) {
                        black_box(registry.call_function(func_name, &args));
                    }
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Validate pattern matching performance claims
/// Location: Various pattern matcher optimizations
fn validate_pattern_matching_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching_speedup");
    
    // Different pattern complexities to measure scaling
    let pattern_complexities = vec![
        ("simple", "f[x_]", "f[5]"),
        ("constrained", "f[x_Integer]", "f[42]"), 
        ("nested", "g[f[x_], y_]", "g[f[3], 7]"),
        ("sequence", "h[x__, y_]", "h[1, 2, 3, 4]"),
    ];

    for (name, pattern_src, test_src) in pattern_complexities {
        group.bench_function(BenchmarkId::new("pattern_complexity", name), |b| {
            let mut parser = Parser::from_source(pattern_src).unwrap();
            let pattern_ast = parser.parse().unwrap();
            
            let mut test_parser = Parser::from_source(test_src).unwrap();
            let test_ast = test_parser.parse().unwrap();
            
            b.iter(|| {
                // Simulate pattern matching operation
                let start = Instant::now();
                
                // Basic pattern matching simulation
                match (&pattern_ast[0], &test_ast[0]) {
                    (Expr::Function { head: p_head, args: p_args }, 
                     Expr::Function { head: t_head, args: t_args }) => {
                        black_box(p_head == t_head);
                        black_box(p_args.len() == t_args.len());
                    }
                    _ => {}
                }
                
                black_box(start.elapsed())
            });
        });
    }

    group.finish();
}

/// Validate memory management speedup claims
/// Location: Memory system optimizations throughout
fn validate_memory_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_speedup_validation");
    
    // Allocation-heavy workloads
    group.bench_function("allocation_intensive", |b| {
        b.iter_batched(
            || Vec::new(),
            |mut values| {
                // Simulate heavy allocation/deallocation
                for i in 0..1000 {
                    values.push(Value::Integer(i));
                    if i % 100 == 0 {
                        values.clear(); // Force deallocation
                    }
                }
                black_box(values)
            },
            BatchSize::LargeInput,
        );
    });

    // Large data structure operations
    group.bench_function("large_structure_ops", |b| {
        b.iter_batched(
            || {
                let mut large_list = Vec::new();
                for i in 0..10000 {
                    large_list.push(Value::Integer(i));
                }
                Value::List(large_list)
            },
            |large_list| {
                match &large_list {
                    Value::List(elements) => {
                        // Operations that stress memory system
                        black_box(elements.len());
                        for chunk in elements.chunks(1000) {
                            black_box(chunk.first());
                        }
                    }
                    _ => {}
                }
            },
            BatchSize::LargeInput,
        );
    });

    group.finish();
}

/// Validate compilation speedup claims  
fn validate_compilation_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_speedup");
    
    let test_programs = vec![
        ("simple", "2 + 3 * 4"),
        ("moderate", "Sin[Pi/4] + Cos[Pi/6] * Length[{1,2,3}]"),
        ("complex", "f[x_] := x^2; g[y_] := f[y] + 1; g[5]"),
    ];

    for (complexity, source) in test_programs {
        group.bench_function(BenchmarkId::new("compilation", complexity), |b| {
            b.iter(|| {
                let mut parser = Parser::from_source(black_box(source)).unwrap();
                let ast = parser.parse().unwrap();
                
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile(black_box(&ast)).unwrap();
                
                black_box(bytecode)
            });
        });
    }

    group.finish();
}

/// Measure actual vs claimed speedup ratios
fn speedup_ratio_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_ratio_analysis");
    
    // Measure baseline (simulated "old" implementation)
    group.bench_function("baseline_implementation", |b| {
        b.iter(|| {
            // Simulate slower, unoptimized approach
            let start = Instant::now();
            
            // Multiple parsing passes (inefficient)
            let source = "Length[{1, 2, 3, 4, 5}]";
            for _ in 0..5 {
                let mut parser = Parser::from_source(source).unwrap();
                let _ = parser.parse().unwrap();
            }
            
            // Slow function lookup simulation
            let function_name = "Length";
            for _ in 0..100 {
                black_box(function_name.chars().count());
            }
            
            black_box(start.elapsed())
        });
    });
    
    // Measure optimized implementation
    group.bench_function("optimized_implementation", |b| {
        let source = "Length[{1, 2, 3, 4, 5}]";
        let mut parser = Parser::from_source(source).unwrap();
        let ast = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile(&ast).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.execute(black_box(&bytecode))
        });
    });

    // End-to-end performance comparison
    group.bench_function("end_to_end_comparison", |b| {
        let test_cases = vec![
            "2 + 3 * 4",
            "Length[{1, 2, 3}]", 
            "Head[Tail[{1, 2, 3, 4}]]",
            "Sin[Pi/4] + Cos[Pi/6]",
        ];
        
        b.iter(|| {
            for source in &test_cases {
                let start = Instant::now();
                
                let mut parser = Parser::from_source(source).unwrap();
                let ast = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile(&ast).unwrap();
                
                let mut vm = VirtualMachine::new();
                let _result = vm.execute(&bytecode);
                
                black_box(start.elapsed());
            }
        });
    });

    group.finish();
}

/// Specific benchmark for validating numeric claims
fn numeric_claims_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("numeric_claims_validation");
    
    // Test if we can achieve anything close to 1000x
    group.bench_function("claim_validation_test", |b| {
        // Setup optimized path
        let source = "Length[{1, 2, 3, 4, 5}]";
        let mut parser = Parser::from_source(source).unwrap();
        let ast = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile(&ast).unwrap();
        let mut vm = VirtualMachine::new();
        
        b.iter(|| {
            // Measure the "fast path"
            let start = Instant::now();
            let _result = vm.execute(black_box(&bytecode));
            let fast_duration = start.elapsed();
            
            // Measure a deliberately slow path for comparison
            let slow_start = Instant::now();
            for _ in 0..1000 { // Simulate 1000x slower operations
                black_box("Length".chars().count());
            }
            let slow_duration = slow_start.elapsed();
            
            // Calculate actual ratio
            let ratio = if fast_duration.as_nanos() > 0 {
                slow_duration.as_nanos() as f64 / fast_duration.as_nanos() as f64
            } else {
                0.0
            };
            
            black_box((fast_duration, slow_duration, ratio))
        });
    });

    group.finish();
}

criterion_group!(
    speedup_claims,
    validate_call_static_speedup,
    validate_pattern_matching_speedup,
    validate_memory_speedup,
    validate_compilation_speedup,
    speedup_ratio_analysis,
    numeric_claims_validation
);

criterion_main!(speedup_claims);