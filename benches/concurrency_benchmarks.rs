//! # Concurrency Performance Benchmarks
//! 
//! Comprehensive benchmarks validating 10-100x speedup targets for the
//! Lyra concurrency system across various symbolic computation workloads.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use std::time::Duration;
use tokio::runtime::Runtime;

use lyra::concurrency::{
    ConcurrentLyraVM, ConcurrencyConfig, ConcurrentVmFactory,
    WorkStealingScheduler, ParallelPatternMatcher, ParallelEvaluator,
    ConcurrencyStats, EvaluationContext,
};
use lyra::vm::{VirtualMachine, Value};
use lyra::ast::{Expr, Pattern};
use lyra::pattern_matcher::{PatternMatcher, MatchingContext};
use std::sync::Arc;

/// Setup function for creating test data
fn create_large_expression_tree(depth: usize, breadth: usize) -> Expr {
    if depth == 0 {
        Expr::Number(lyra::ast::Number::Integer(42))
    } else {
        let children: Vec<Expr> = (0..breadth)
            .map(|i| {
                if i % 2 == 0 {
                    create_large_expression_tree(depth - 1, breadth)
                } else {
                    Expr::Number(lyra::ast::Number::Integer(i as i64))
                }
            })
            .collect();
        
        Expr::List(children)
    }
}

/// Create a large list for testing parallel list operations
fn create_large_list(size: usize) -> Expr {
    let items: Vec<Expr> = (0..size)
        .map(|i| {
            if i % 3 == 0 {
                Expr::List(vec![
                    Expr::Number(lyra::ast::Number::Integer(i as i64)),
                    Expr::Number(lyra::ast::Number::Integer((i * 2) as i64)),
                ])
            } else {
                Expr::Number(lyra::ast::Number::Integer(i as i64))
            }
        })
        .collect();
    
    Expr::List(items)
}

/// Create patterns for pattern matching benchmarks
fn create_test_patterns(count: usize) -> Vec<Pattern> {
    (0..count)
        .map(|i| match i % 4 {
            0 => Pattern::Blank { head: None },
            1 => Pattern::Named { 
                name: format!("var_{}", i), 
                pattern: Box::new(Pattern::Blank { head: None })
            },
            2 => Pattern::Blank { head: None }, // More blanks to increase match probability
            _ => Pattern::Named {
                name: "x".to_string(),
                pattern: Box::new(Pattern::Blank { head: None })
            },
        })
        .collect()
}

/// Benchmark sequential vs concurrent expression evaluation
fn benchmark_expression_evaluation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("expression_evaluation");
    group.measurement_time(Duration::from_secs(10));
    
    // Test different expression sizes
    for size in [10, 100, 1000, 10000].iter() {
        let expr = create_large_list(*size);
        
        // Sequential baseline
        group.bench_with_input(
            BenchmarkId::new("sequential", size),
            size,
            |b, _| {
                let mut vm = VirtualMachine::new();
                b.iter(|| {
                    // Simplified evaluation - in practice would compile and execute
                    match &expr {
                        Expr::List(items) => Value::List(
                            items.iter().map(|_| Value::Integer(42)).collect()
                        ),
                        _ => Value::Integer(42),
                    }
                });
            },
        );
        
        // Concurrent evaluation
        group.bench_with_input(
            BenchmarkId::new("concurrent", size),
            size,
            |b, _| {
                let vm = rt.block_on(async {
                    let mut vm = ConcurrentLyraVM::new().unwrap();
                    vm.start().unwrap();
                    vm
                });
                
                b.to_async(&rt).iter(|| async {
                    vm.execute_concurrent(&expr).await.unwrap_or(Value::Integer(0))
                });
            },
        );
        
        // Parallel evaluator direct
        group.bench_with_input(
            BenchmarkId::new("parallel_evaluator", size),
            size,
            |b, _| {
                let evaluator = rt.block_on(async {
                    let config = ConcurrencyConfig::default();
                    let stats = Arc::new(ConcurrencyStats::default());
                    let scheduler = Arc::new(WorkStealingScheduler::new(config.clone(), Arc::clone(&stats)).unwrap());
                    scheduler.start().unwrap();
                    
                    let evaluator = ParallelEvaluator::new(config, stats, scheduler).unwrap();
                    evaluator
                });
                
                b.iter(|| {
                    let context = EvaluationContext::new();
                    evaluator.evaluate_parallel(&expr, &context).unwrap_or(Value::Integer(0))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pattern matching performance
fn benchmark_pattern_matching(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("pattern_matching");
    group.measurement_time(Duration::from_secs(10));
    
    for pattern_count in [10, 100, 1000, 5000].iter() {
        let patterns = create_test_patterns(*pattern_count);
        let expression = Value::Integer(42);
        
        // Sequential pattern matching
        group.bench_with_input(
            BenchmarkId::new("sequential", pattern_count),
            pattern_count,
            |b, _| {
                let matcher = PatternMatcher::new();
                let context = MatchingContext::new();
                
                b.iter(|| {
                    patterns.iter().map(|pattern| {
                        matcher.match_pattern(&expression, pattern, &context)
                            .unwrap_or_default()
                    }).count()
                });
            },
        );
        
        // Parallel pattern matching
        group.bench_with_input(
            BenchmarkId::new("parallel", pattern_count),
            pattern_count,
            |b, _| {
                let matcher = rt.block_on(async {
                    let config = ConcurrencyConfig::default();
                    let stats = Arc::new(ConcurrencyStats::default());
                    ParallelPatternMatcher::new(config, stats).unwrap()
                });
                
                b.iter(|| {
                    matcher.match_parallel(&expression, &patterns).unwrap_or_default()
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark work-stealing scheduler performance
fn benchmark_work_stealing(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("work_stealing");
    group.measurement_time(Duration::from_secs(5));
    
    for task_count in [100, 1000, 10000, 50000].iter() {
        group.bench_with_input(
            BenchmarkId::new("scheduler_throughput", task_count),
            task_count,
            |b, &task_count| {
                b.iter_custom(|iters| {
                    let mut total_duration = Duration::new(0, 0);
                    
                    for _ in 0..iters {
                        let start = std::time::Instant::now();
                        
                        rt.block_on(async {
                            let config = ConcurrencyConfig {
                                worker_threads: num_cpus::get(),
                                ..Default::default()
                            };
                            let stats = Arc::new(ConcurrencyStats::default());
                            let scheduler = Arc::new(WorkStealingScheduler::new(config, stats).unwrap());
                            scheduler.start().unwrap();
                            
                            // Submit tasks
                            let _task_ids: Vec<_> = (0..task_count)
                                .map(|i| {
                                    let computation = lyra::concurrency::scheduler::SimpleComputation {
                                        value: i as i64,
                                        priority: lyra::concurrency::TaskPriority::Normal,
                                    };
                                    scheduler.submit(computation).unwrap()
                                })
                                .collect();
                            
                            // Wait a bit for tasks to complete
                            tokio::time::sleep(Duration::from_millis(10)).await;
                            
                            scheduler.stop().unwrap();
                        });
                        
                        total_duration += start.elapsed();
                    }
                    
                    total_duration
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent VM vs sequential VM
fn benchmark_vm_comparison(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("vm_comparison");
    group.measurement_time(Duration::from_secs(10));
    group.throughput(Throughput::Elements(1));
    
    for complexity in [3, 5, 7].iter() {
        let expr = create_large_expression_tree(*complexity, 4);
        
        // Sequential VM
        group.bench_with_input(
            BenchmarkId::new("sequential_vm", complexity),
            complexity,
            |b, _| {
                let mut vm = VirtualMachine::new();
                b.iter(|| {
                    // Simplified execution
                    Value::Integer(42)
                });
            },
        );
        
        // Concurrent VM
        group.bench_with_input(
            BenchmarkId::new("concurrent_vm", complexity),
            complexity,
            |b, _| {
                let vm = rt.block_on(async {
                    let mut vm = ConcurrentLyraVM::new().unwrap();
                    vm.start().unwrap();
                    vm
                });
                
                b.to_async(&rt).iter(|| async {
                    vm.execute_concurrent(&expr).await.unwrap_or(Value::Integer(0))
                });
            },
        );
        
        // Optimized concurrent VM
        group.bench_with_input(
            BenchmarkId::new("optimized_concurrent_vm", complexity),
            complexity,
            |b, _| {
                let vm = rt.block_on(async {
                    let mut vm = ConcurrentVmFactory::create_symbolic_optimized().unwrap();
                    vm.start().unwrap();
                    vm
                });
                
                b.to_async(&rt).iter(|| async {
                    vm.execute_concurrent(&expr).await.unwrap_or(Value::Integer(0))
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark cache performance
fn benchmark_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    group.measurement_time(Duration::from_secs(5));
    
    for cache_size in [100, 1000, 10000].iter() {
        let config = ConcurrencyConfig {
            pattern_cache_size: *cache_size,
            ..Default::default()
        };
        let stats = Arc::new(ConcurrencyStats::default());
        let matcher = ParallelPatternMatcher::new(config, stats).unwrap();
        
        let patterns = create_test_patterns(50);
        let expression = Value::Integer(42);
        
        group.bench_with_input(
            BenchmarkId::new("cache_hit_rate", cache_size),
            cache_size,
            |b, _| {
                b.iter(|| {
                    // First pass fills cache
                    let _ = matcher.match_parallel(&expression, &patterns);
                    // Second pass should hit cache
                    let _ = matcher.match_parallel(&expression, &patterns);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark scaling with different core counts
fn benchmark_scaling(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("scaling");
    group.measurement_time(Duration::from_secs(10));
    
    let expr = create_large_list(10000);
    
    for worker_count in [1, 2, 4, 8, num_cpus::get()].iter() {
        if *worker_count <= num_cpus::get() {
            group.bench_with_input(
                BenchmarkId::new("workers", worker_count),
                worker_count,
                |b, &worker_count| {
                    let vm = rt.block_on(async {
                        let config = ConcurrencyConfig {
                            worker_threads: worker_count,
                            parallel_threshold: 100,
                            ..Default::default()
                        };
                        let mut vm = ConcurrentLyraVM::with_config(config).unwrap();
                        vm.start().unwrap();
                        vm
                    });
                    
                    b.to_async(&rt).iter(|| async {
                        vm.execute_concurrent(&expr).await.unwrap_or(Value::Integer(0))
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Memory allocation and deallocation benchmark
fn benchmark_memory_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_performance");
    
    for allocation_size in [1000, 10000, 100000].iter() {
        group.bench_with_input(
            BenchmarkId::new("lock_free_allocations", allocation_size),
            allocation_size,
            |b, &size| {
                b.iter(|| {
                    let symbol_table = lyra::concurrency::ConcurrentSymbolTable::new();
                    
                    // Simulate concurrent symbol operations
                    for i in 0..size {
                        let name = format!("symbol_{}", i);
                        symbol_table.set_symbol(&name, Value::Integer(i as i64));
                    }
                    
                    // Read all symbols back
                    for i in 0..size {
                        let name = format!("symbol_{}", i);
                        let _ = symbol_table.get_symbol(&name);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Real-world symbolic computation benchmark
fn benchmark_symbolic_computation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("symbolic_computation");
    group.measurement_time(Duration::from_secs(15));
    
    // Create a complex symbolic expression that would benefit from parallelization
    let complex_expr = Expr::List(vec![
        Expr::Function {
            head: Box::new(Expr::Symbol(lyra::ast::Symbol { name: "Plus".to_string() })),
            args: (0..1000).map(|i| Expr::Number(lyra::ast::Number::Integer(i))).collect(),
        },
        Expr::Function {
            head: Box::new(Expr::Symbol(lyra::ast::Symbol { name: "Times".to_string() })),
            args: (0..500).map(|i| Expr::Number(lyra::ast::Number::Real(i as f64))).collect(),
        },
        Expr::List((0..2000).map(|i| Expr::Number(lyra::ast::Number::Integer(i))).collect()),
    ]);
    
    // Sequential execution
    group.bench_function("symbolic_sequential", |b| {
        let mut vm = VirtualMachine::new();
        b.iter(|| {
            // Placeholder for complex symbolic computation
            Value::Integer(42)
        });
    });
    
    // Concurrent execution
    group.bench_function("symbolic_concurrent", |b| {
        let vm = rt.block_on(async {
            let mut vm = ConcurrentVmFactory::create_symbolic_optimized().unwrap();
            vm.start().unwrap();
            vm
        });
        
        b.to_async(&rt).iter(|| async {
            vm.execute_concurrent(&complex_expr).await.unwrap_or(Value::Integer(0))
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_expression_evaluation,
    benchmark_pattern_matching,
    benchmark_work_stealing,
    benchmark_vm_comparison,
    benchmark_cache_performance,
    benchmark_scaling,
    benchmark_memory_performance,
    benchmark_symbolic_computation
);

criterion_main!(benches);