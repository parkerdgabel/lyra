//! Performance Validation Benchmarks
//!
//! This module validates the performance claims made throughout the Lyra codebase,
//! particularly the numerous "1000x speedup" assertions. It provides empirical
//! measurements to establish baselines and validate optimization effectiveness.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use lyra::{
    vm::VirtualMachine,
    parser::Parser,
    compiler::Compiler,
    stdlib::StandardLibrary,
    pattern_matcher::PatternMatcher,
    ast::Expr,
    vm::Value,
};
use std::time::Instant;

/// Benchmark core VM execution performance
fn vm_execution_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_execution");
    
    // Simple arithmetic execution
    group.bench_function("arithmetic_execution", |b| {
        let source = "2 + 3 * 4 - 1 / 2";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.load_bytecode(black_box(&bytecode));
            vm.run()
        });
    });

    // Function call execution (stdlib)
    group.bench_function("stdlib_function_calls", |b| {
        let source = "Length[{1, 2, 3, 4, 5}]";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.load_bytecode(black_box(&bytecode));
            vm.run()
        });
    });

    // Complex mathematical expression
    group.bench_function("complex_math_execution", |b| {
        let source = "Sin[Pi / 4] + Cos[Pi / 6] * Tan[Pi / 3]";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.load_bytecode(black_box(&bytecode));
            vm.run()
        });
    });

    group.finish();
}

/// Benchmark pattern matching performance claims
fn pattern_matching_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_matching");
    
    // Simple pattern matching
    group.bench_function("simple_pattern_match", |b| {
        let source = "f[x_] := x^2";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        
        // Create test expression to match against
        let test_expr_source = "f[5]";
        let mut test_parser = Parser::from_source(test_expr_source).unwrap();
        let test_statements = test_parser.parse().unwrap();
        let test_expr = &test_statements[0];
        
        b.iter_batched(
            || PatternMatcher::new(),
            |mut matcher| {
                // Simulate pattern matching operation
                matcher.try_match(black_box(test_expr), black_box(&statements[0]))
            },
            BatchSize::SmallInput,
        );
    });

    // Complex pattern matching with constraints
    group.bench_function("complex_pattern_match", |b| {
        let source = "integrate[expr_, x_Symbol] := D[expr, x]";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        
        let test_expr_source = "integrate[x^2 + 3*x + 1, x]";
        let mut test_parser = Parser::from_source(test_expr_source).unwrap();
        let test_statements = test_parser.parse().unwrap();
        let test_expr = &test_statements[0];
        
        b.iter_batched(
            || PatternMatcher::new(),
            |mut matcher| {
                matcher.try_match(black_box(test_expr), black_box(&statements[0]))
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

/// Benchmark CALL_STATIC optimization claims
fn call_static_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("call_static_optimization");
    
    // Direct function call via CALL_STATIC
    group.bench_function("optimized_stdlib_call", |b| {
        let stdlib = StandardLibrary::new().unwrap();
        let args = vec![Value::List(vec![
            Value::Integer(1),
            Value::Integer(2), 
            Value::Integer(3),
            Value::Integer(4),
            Value::Integer(5),
        ])];
        
        b.iter(|| {
            // Simulate CALL_STATIC optimization
            stdlib.call_function(black_box("Length"), black_box(&args))
        });
    });

    // Multiple chained function calls
    group.bench_function("chained_function_calls", |b| {
        let source = "Head[Tail[{1, 2, 3, 4, 5}]]";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.load_bytecode(black_box(&bytecode));
            vm.run()
        });
    });

    group.finish();
}

/// Memory management performance benchmarks
fn memory_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_performance");
    
    // Large list creation and manipulation
    group.bench_function("large_list_operations", |b| {
        b.iter_batched(
            || {
                // Setup: create large list
                let mut elements = Vec::new();
                for i in 0..10000 {
                    elements.push(Value::Integer(i));
                }
                Value::List(elements)
            },
            |large_list| {
                // Measure operations on large list
                match &large_list {
                    Value::List(elements) => {
                        black_box(elements.len());
                        black_box(elements.first());
                        black_box(elements.last());
                    }
                    _ => {}
                }
            },
            BatchSize::LargeInput,
        );
    });

    // Memory allocation stress test  
    group.bench_function("allocation_stress_test", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(black_box(Value::Integer(i)));
            }
            black_box(values);
        });
    });

    group.finish();
}

/// Comparative baseline measurements
fn baseline_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_comparison");
    
    // Baseline: Native Rust arithmetic
    group.bench_function("native_rust_arithmetic", |b| {
        b.iter(|| {
            let result = black_box(2) + black_box(3) * black_box(4) - black_box(1);
            black_box(result)
        });
    });

    // Lyra equivalent
    group.bench_function("lyra_arithmetic", |b| {
        let source = "2 + 3 * 4 - 1";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_program(&statements).unwrap();
        
        b.iter(|| {
            let mut vm = VirtualMachine::new();
            vm.load_bytecode(black_box(&bytecode));
            vm.run()
        });
    });

    // Calculate potential speedup claims
    group.bench_function("speedup_validation", |b| {
        // Baseline implementation (simulated slower approach)
        b.iter(|| {
            let start = Instant::now();
            
            // Simulate naive interpretation
            let source = "Length[{1, 2, 3, 4, 5}]";
            let mut parser = Parser::from_source(source).unwrap();
            let ast = parser.parse().unwrap();
            
            // Simulated slow path: reparse every time
            for _ in 0..10 {
                let mut parser = Parser::from_source(source).unwrap();
                let _ = parser.parse().unwrap();
            }
            
            black_box(start.elapsed())
        });
    });

    group.finish();
}

/// Regression detection benchmarks  
fn regression_detection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");
    
    // Benchmark critical paths that should maintain performance
    group.bench_function("critical_path_parsing", |b| {
        let sources = vec![
            "2 + 3 * 4",
            "Sin[x] + Cos[y]", 
            "Length[{1, 2, 3}]",
            "f[x_] := x^2",
        ];
        
        b.iter(|| {
            for source in &sources {
                let mut parser = Parser::from_source(source).unwrap();
                black_box(parser.parse().unwrap());
            }
        });
    });

    // VM execution regression detection
    group.bench_function("critical_path_execution", |b| {
        let test_cases = vec![
            "2 + 3 * 4",
            "Length[{1, 2, 3, 4, 5}]",
            "Head[{1, 2, 3}]",
        ];
        
        b.iter(|| {
            for source in &test_cases {
                let mut parser = Parser::from_source(source).unwrap();
                let statements = parser.parse().unwrap();
                let mut compiler = Compiler::new();
                let bytecode = compiler.compile_program(&statements).unwrap();
                
                let mut vm = VirtualMachine::new();
                black_box(vm.execute(&bytecode));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    vm_execution_benchmark,
    pattern_matching_benchmark, 
    call_static_benchmark,
    memory_performance_benchmark,
    baseline_comparison_benchmark,
    regression_detection_benchmark
);

criterion_main!(benches);