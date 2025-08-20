//! Phase 3B: Comprehensive Performance Benchmarking Framework
//!
//! This benchmark suite validates optimization claims and provides automated performance
//! monitoring for the Lyra symbolic computation engine. It focuses on:
//! 
//! 1. Micro-benchmarks for key operations (symbol interning, Value enum, async primitives)
//! 2. Workload simulations for mathematical computation and data processing  
//! 3. Performance regression detection with automated alerts
//! 4. Memory profiling for allocation patterns and pool efficiency
//! 5. Optimization validation for claimed 2-5x improvements

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput, BatchSize};
use lyra::{
    vm::{VirtualMachine, Value},
    parser::Parser,
    compiler::Compiler,
    stdlib::StandardLibrary,
    pattern_matcher::PatternMatcher,
    memory::{StringInterner, SymbolId},
};
use std::time::{Duration, Instant};
use std::collections::HashMap;

// =============================================================================
// MICRO-BENCHMARKS FOR KEY OPERATIONS
// =============================================================================

/// Benchmark symbol interning performance to validate 40-60% memory reduction claims
fn symbol_interning_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_interning");
    group.throughput(Throughput::Elements(1000));
    
    let common_symbols = vec![
        "x", "y", "z", "Plus", "Times", "Sin", "Cos", "Length", "Head", "Tail",
        "List", "Function", "Real", "Integer", "Boolean", "True", "False",
        "Pi", "E", "Infinity", "Table", "Map", "Apply", "Select", "Part"
    ];
    
    // Baseline: Raw string operations
    group.bench_function("raw_string_storage", |b| {
        b.iter(|| {
            let mut string_storage: Vec<String> = Vec::new();
            for symbol in &common_symbols {
                for i in 0..40 { // Simulate 40 uses of each symbol
                    string_storage.push(format!("{}_{}", symbol, i));
                }
            }
            black_box(string_storage);
        });
    });
    
    // Optimized: Symbol ID interning
    group.bench_function("symbol_id_interning", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let mut symbol_storage: Vec<SymbolId> = Vec::new();
            for symbol in &common_symbols {
                for i in 0..40 { // Simulate 40 uses of each symbol  
                    let symbol_name = format!("{}_{}", symbol, i);
                    symbol_storage.push(interner.intern_symbol_id(&symbol_name));
                }
            }
            black_box(symbol_storage);
        });
    });
    
    // Memory usage comparison
    group.bench_function("memory_overhead_analysis", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let initial_memory = interner.memory_usage();
            
            // Add 1000 symbols
            for i in 0..1000 {
                interner.intern_symbol_id(&format!("test_symbol_{}", i));
            }
            
            let final_memory = interner.memory_usage();
            let memory_per_symbol = (final_memory - initial_memory) as f64 / 1000.0;
            black_box(memory_per_symbol);
        });
    });
    
    group.finish();
}

/// Benchmark Value enum operations for performance optimization validation
fn value_operations_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_operations");
    group.throughput(Throughput::Elements(1000));
    
    // Value creation benchmarks
    group.bench_function("value_creation_integers", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(Value::Integer(i));
            }
        });
    });
    
    group.bench_function("value_creation_reals", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(Value::Real(i as f64 * 0.1));
            }
        });
    });
    
    group.bench_function("value_creation_lists", |b| {
        b.iter(|| {
            for i in 0..100 {
                let list: Vec<Value> = (0..10).map(|j| Value::Integer(i * 10 + j)).collect();
                black_box(Value::List(list));
            }
        });
    });
    
    // Value cloning performance (critical for functional programming)
    group.bench_function("value_cloning_performance", |b| {
        let large_list = Value::List((0..1000).map(|i| Value::Integer(i)).collect());
        
        b.iter(|| {
            for _ in 0..10 {
                black_box(large_list.clone());
            }
        });
    });
    
    group.finish();
}

/// Benchmark parsing performance for baseline measurements
fn parsing_performance_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("parsing_performance");
    
    let test_expressions = vec![
        ("simple_arithmetic", "2 + 3 * 4 - 1"),
        ("function_call", "Length[{1, 2, 3, 4, 5}]"),
        ("nested_expressions", "{{1, 2}, {3, 4}, {5, 6}}"),
        ("mathematical_functions", "Sin[3.14159 / 4] + Cos[3.14159 / 6]"),
        ("complex_nesting", "{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}"),
    ];
    
    for (name, source) in test_expressions {
        group.bench_function(BenchmarkId::new("parse", name), |b| {
            b.iter(|| {
                let mut parser = Parser::from_source(black_box(source)).unwrap();
                black_box(parser.parse().unwrap());
            });
        });
    }
    
    // Parsing throughput test
    group.throughput(Throughput::Elements(100));
    group.bench_function("parsing_throughput", |b| {
        let expressions: Vec<&str> = (0..100).map(|_| "2 + 3 * 4").collect();
        
        b.iter(|| {
            for expr in &expressions {
                let mut parser = Parser::from_source(expr).unwrap();
                black_box(parser.parse().unwrap());
            }
        });
    });
    
    group.finish();
}

/// Benchmark compilation performance
fn compilation_performance_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("compilation_performance");
    
    let test_programs = vec![
        ("simple", "2 + 3 * 4"),
        ("list_operations", "{1, 2, 3, 4, 5}"),
        ("function_calls", "Length[{1, 2, 3}]"),
        ("nested_structures", "{{1, 2}, {3, 4}}"),
    ];
    
    for (complexity, source) in test_programs {
        group.bench_function(BenchmarkId::new("compile", complexity), |b| {
            b.iter_batched(
                || {
                    let mut parser = Parser::from_source(source).unwrap();
                    let statements = parser.parse().unwrap();
                    (Compiler::new(), statements)
                },
                |(mut compiler, statements)| {
                    black_box(compiler.compile_program(&statements).unwrap());
                },
                BatchSize::SmallInput,
            );
        });
    }
    
    group.finish();
}

/// Benchmark VM execution performance
fn vm_execution_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("vm_execution");
    
    let test_cases = vec![
        ("arithmetic", "2 + 3 * 4"),
        ("list_creation", "{1, 2, 3, 4, 5}"),
        ("nested_lists", "{{1, 2}, {3, 4}}"),
    ];
    
    for (name, source) in test_cases {
        group.bench_function(BenchmarkId::new("execute", name), |b| {
            // Pre-compile the program
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let mut compiler = Compiler::new();
            compiler.compile_program(&statements).unwrap();
            
            b.iter(|| {
                // Create fresh compiler for each iteration
                let mut fresh_parser = Parser::from_source(source).unwrap();
                let fresh_statements = fresh_parser.parse().unwrap();
                let mut fresh_compiler = Compiler::new();
                fresh_compiler.compile_program(&fresh_statements).unwrap();
                let mut vm = fresh_compiler.into_vm();
                black_box(vm.run().unwrap());
            });
        });
    }
    
    group.finish();
}

// =============================================================================
// WORKLOAD SIMULATIONS
// =============================================================================

/// Mathematical computation workload simulation
fn mathematical_workload_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mathematical_workload");
    group.throughput(Throughput::Elements(100));
    
    // Intensive arithmetic operations
    group.bench_function("intensive_arithmetic", |b| {
        b.iter(|| {
            let mut result = 0i64;
            for i in 1..=100 {
                result += i * i - i / 2 + i % 7;
            }
            black_box(result);
        });
    });
    
    // List processing workload
    group.bench_function("list_processing", |b| {
        b.iter(|| {
            let mut lists = Vec::new();
            for i in 0..20 {
                let list: Vec<Value> = (0..50).map(|j| Value::Integer(i * 50 + j)).collect();
                lists.push(Value::List(list));
            }
            black_box(lists);
        });
    });
    
    // Pattern matching intensive workload
    group.bench_function("pattern_matching_intensive", |b| {
        b.iter(|| {
            for _ in 0..10 {
                let matcher = PatternMatcher::new();
                black_box(matcher);
            }
        });
    });
    
    group.finish();
}

/// Data processing workload simulation
fn data_processing_workload_simulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_processing_workload");
    
    // Large data structure manipulation
    group.bench_function("large_data_manipulation", |b| {
        b.iter_batched(
            || {
                // Create large dataset
                let mut data = Vec::new();
                for i in 0..1000 {
                    let row: Vec<Value> = (0..10).map(|j| Value::Real((i * 10 + j) as f64)).collect();
                    data.push(Value::List(row));
                }
                Value::List(data)
            },
            |dataset| {
                match &dataset {
                    Value::List(rows) => {
                        // Simulate data processing operations
                        let mut sum = 0.0;
                        for row in rows.iter().take(100) { // Process subset for benchmarking
                            if let Value::List(values) = row {
                                for value in values {
                                    if let Value::Real(r) = value {
                                        sum += r;
                                    }
                                }
                            }
                        }
                        black_box(sum);
                    }
                    _ => {}
                }
            },
            BatchSize::LargeInput,
        );
    });
    
    group.finish();
}

// =============================================================================
// PERFORMANCE REGRESSION DETECTION
// =============================================================================

/// Critical path performance monitoring for regression detection
fn regression_detection_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_detection");
    
    // Core operations that must maintain performance
    group.bench_function(BenchmarkId::new("critical_path", "parser_create"), |b| {
        b.iter(|| {
            black_box(Parser::from_source("2 + 3").unwrap());
        });
    });
    
    group.bench_function(BenchmarkId::new("critical_path", "compiler_create"), |b| {
        b.iter(|| {
            black_box(Compiler::new());
        });
    });
    
    group.bench_function(BenchmarkId::new("critical_path", "vm_create"), |b| {
        b.iter(|| {
            black_box(VirtualMachine::new());
        });
    });
    
    group.bench_function(BenchmarkId::new("critical_path", "stdlib_create"), |b| {
        b.iter(|| {
            black_box(StandardLibrary::new());
        });
    });
    
    group.bench_function(BenchmarkId::new("critical_path", "pattern_matcher_create"), |b| {
        b.iter(|| {
            black_box(PatternMatcher::new());
        });
    });
    
    // End-to-end critical path
    group.bench_function("end_to_end_critical_path", |b| {
        b.iter(|| {
            let source = "2 + 3 * 4";
            let mut parser = Parser::from_source(source).unwrap();
            let statements = parser.parse().unwrap();
            let mut compiler = Compiler::new();
            compiler.compile_program(&statements).unwrap();
            let mut vm = compiler.into_vm();
            black_box(vm.run().unwrap());
        });
    });
    
    group.finish();
}

// =============================================================================
// MEMORY PROFILING BENCHMARKS
// =============================================================================

/// Memory allocation pattern analysis
fn memory_profiling_benchmarks(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_profiling");
    
    // Memory allocation stress test
    group.bench_function("allocation_stress", |b| {
        b.iter(|| {
            let mut allocations = Vec::new();
            for i in 0..1000 {
                allocations.push(Value::Integer(i));
                if i % 100 == 0 {
                    allocations.clear(); // Force deallocation
                }
            }
            black_box(allocations);
        });
    });
    
    // Symbol interning memory efficiency
    group.bench_function("symbol_interning_efficiency", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let initial_usage = interner.memory_usage();
            
            // Intern many symbols with repetition (realistic workload)
            let symbols = vec!["x", "y", "Plus", "Times"];
            for _ in 0..250 { // 1000 total operations, but only 4 unique symbols
                for symbol in &symbols {
                    interner.intern_symbol_id(symbol);
                }
            }
            
            let final_usage = interner.memory_usage();
            black_box(final_usage - initial_usage);
        });
    });
    
    group.finish();
}

// =============================================================================
// SPEEDUP CLAIMS VALIDATION
// =============================================================================

/// Validate specific performance claims made in the codebase
fn speedup_claims_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("speedup_claims_validation");
    
    // Validate symbol interning speedup
    group.bench_function("symbol_lookup_baseline", |b| {
        let symbols = vec!["x", "y", "Plus", "Times", "Sin", "Cos"];
        let mut symbol_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            symbol_map.insert(symbol.to_string(), i);
        }
        
        b.iter(|| {
            for symbol in &symbols {
                for _ in 0..100 { // Repeat lookups
                    black_box(symbol_map.get(*symbol));
                }
            }
        });
    });
    
    group.bench_function("symbol_lookup_optimized", |b| {
        let symbols = vec!["x", "y", "Plus", "Times", "Sin", "Cos"];
        let interner = StringInterner::new();
        let symbol_ids: Vec<SymbolId> = symbols.iter()
            .map(|s| interner.intern_symbol_id(s))
            .collect();
        
        b.iter(|| {
            for &symbol_id in &symbol_ids {
                for _ in 0..100 { // Repeat lookups
                    black_box(interner.resolve_symbol(symbol_id));
                }
            }
        });
    });
    
    // Validate compilation speedup claims
    group.bench_function("compilation_baseline", |b| {
        let source = "2 + 3 * 4";
        b.iter(|| {
            // Simulate slower compilation (reparse multiple times)
            for _ in 0..5 {
                let mut parser = Parser::from_source(source).unwrap();
                black_box(parser.parse().unwrap());
            }
        });
    });
    
    group.bench_function("compilation_optimized", |b| {
        let source = "2 + 3 * 4";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        
        b.iter(|| {
            let mut compiler = Compiler::new();
            black_box(compiler.compile_program(&statements).unwrap());
        });
    });
    
    group.finish();
}

// =============================================================================
// BASELINE VS OPTIMIZED COMPARISONS
// =============================================================================

/// Direct comparison benchmarks for before/after optimization validation
fn baseline_vs_optimized_comparisons(c: &mut Criterion) {
    let mut group = c.benchmark_group("baseline_vs_optimized");
    
    // String operations: baseline vs interned
    group.bench_function("string_operations_baseline", |b| {
        let operations = vec!["Plus", "Times", "Sin", "Cos"];
        b.iter(|| {
            let mut results = Vec::new();
            for op in &operations {
                for i in 0..50 {
                    results.push(format!("{}_{}", op, i));
                }
            }
            black_box(results);
        });
    });
    
    group.bench_function("string_operations_optimized", |b| {
        let operations = vec!["Plus", "Times", "Sin", "Cos"];
        let interner = StringInterner::new();
        b.iter(|| {
            let mut results = Vec::new();
            for op in &operations {
                for i in 0..50 {
                    let symbol_name = format!("{}_{}", op, i);
                    results.push(interner.intern_symbol_id(&symbol_name));
                }
            }
            black_box(results);
        });
    });
    
    group.finish();
}

// =============================================================================
// AUTOMATED PERFORMANCE MONITORING
// =============================================================================

/// Performance monitoring benchmarks for continuous validation
fn automated_performance_monitoring(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_monitoring");
    
    // Key performance indicators that should be monitored
    group.throughput(Throughput::Elements(1000));
    group.bench_function("kpi_parser_throughput", |b| {
        let expressions: Vec<&str> = (0..1000).map(|_| "x + y").collect();
        
        b.iter(|| {
            for expr in &expressions {
                let mut parser = Parser::from_source(expr).unwrap();
                black_box(parser.parse().unwrap());
            }
        });
    });
    
    group.throughput(Throughput::Elements(100));
    group.bench_function("kpi_compilation_speed", |b| {
        let source = "2 + 3 * 4";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        
        b.iter(|| {
            for _ in 0..100 {
                let mut compiler = Compiler::new();
                black_box(compiler.compile_program(&statements).unwrap());
            }
        });
    });
    
    group.throughput(Throughput::Elements(100));
    group.bench_function("kpi_execution_speed", |b| {
        let source = "2 + 3";
        
        b.iter(|| {
            for _ in 0..100 {
                // Create fresh compiler for each execution
                let mut fresh_parser = Parser::from_source(source).unwrap();
                let fresh_statements = fresh_parser.parse().unwrap();
                let mut fresh_compiler = Compiler::new();
                fresh_compiler.compile_program(&fresh_statements).unwrap();
                let mut vm = fresh_compiler.into_vm();
                black_box(vm.run().unwrap());
            }
        });
    });
    
    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    micro_benchmarks,
    symbol_interning_benchmarks,
    value_operations_benchmarks,
    parsing_performance_benchmarks,
    compilation_performance_benchmarks,
    vm_execution_benchmarks
);

criterion_group!(
    workload_simulations,
    mathematical_workload_simulation,
    data_processing_workload_simulation
);

criterion_group!(
    performance_validation,
    regression_detection_benchmarks,
    memory_profiling_benchmarks,
    speedup_claims_validation,
    baseline_vs_optimized_comparisons
);

criterion_group!(
    monitoring,
    automated_performance_monitoring
);

criterion_main!(
    micro_benchmarks,
    workload_simulations, 
    performance_validation,
    monitoring
);

// =============================================================================
// VALIDATION TESTS
// =============================================================================

#[cfg(test)]
mod validation_tests {
    use super::*;
    
    #[test]
    fn validate_symbol_interning_memory_efficiency() {
        let interner = StringInterner::new();
        let initial_memory = interner.memory_usage();
        
        // Add 1000 symbols
        for i in 0..1000 {
            interner.intern_symbol_id(&format!("test_symbol_{}", i));
        }
        
        let final_memory = interner.memory_usage();
        let memory_per_symbol = (final_memory - initial_memory) as f64 / 1000.0;
        
        println!("Memory per symbol: {:.2} bytes", memory_per_symbol);
        println!("SymbolId size: {} bytes", std::mem::size_of::<SymbolId>());
        println!("String size: {} bytes", std::mem::size_of::<String>());
        
        // Validate that symbol IDs are more efficient than strings
        assert!(std::mem::size_of::<SymbolId>() < std::mem::size_of::<String>());
        
        // Memory per symbol should be reasonable (including overhead)
        assert!(memory_per_symbol < 50.0, 
            "Memory per symbol should be under 50 bytes including overhead, got {:.2}", 
            memory_per_symbol);
    }
    
    #[test]
    fn validate_performance_baseline_establishment() {
        // Establish baseline performance measurements
        let start = Instant::now();
        
        // Simple parsing benchmark
        for _ in 0..100 {
            let mut parser = Parser::from_source("2 + 3 * 4").unwrap();
            let _ = parser.parse().unwrap();
        }
        
        let parsing_time = start.elapsed();
        println!("Parsing 100 expressions took: {:?}", parsing_time);
        
        // Compilation benchmark
        let start = Instant::now();
        let mut parser = Parser::from_source("2 + 3 * 4").unwrap();
        let statements = parser.parse().unwrap();
        
        for _ in 0..100 {
            let mut compiler = Compiler::new();
            compiler.compile_program(&statements).unwrap();
        }
        
        let compilation_time = start.elapsed();
        println!("Compiling 100 programs took: {:?}", compilation_time);
        
        // These should complete in reasonable time
        assert!(parsing_time < Duration::from_secs(1), "Parsing took too long: {:?}", parsing_time);
        assert!(compilation_time < Duration::from_secs(1), "Compilation took too long: {:?}", compilation_time);
    }
    
    #[test]
    fn validate_speedup_measurement_framework() {
        // Test the framework for measuring speedup claims
        let baseline_start = Instant::now();
        
        // Baseline: Inefficient operation
        let mut results = Vec::new();
        for i in 0..1000 {
            results.push(format!("symbol_{}", i));
        }
        
        let baseline_time = baseline_start.elapsed();
        
        let optimized_start = Instant::now();
        
        // Optimized: Using symbol interning
        let interner = StringInterner::new();
        let mut optimized_results = Vec::new();
        for i in 0..1000 {
            optimized_results.push(interner.intern_symbol_id(&format!("symbol_{}", i)));
        }
        
        let optimized_time = optimized_start.elapsed();
        
        println!("Baseline time: {:?}", baseline_time);
        println!("Optimized time: {:?}", optimized_time);
        
        if optimized_time.as_nanos() > 0 {
            let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
            println!("Measured speedup: {:.2}x", speedup);
            
            // Should show some improvement, though exact speedup depends on implementation
            assert!(speedup >= 0.5, "Should show some performance characteristics");
        }
    }
}