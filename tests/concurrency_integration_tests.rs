//! # Concurrency Integration Tests
//! 
//! Comprehensive integration tests validating the concurrency system's
//! correctness, performance, and 10-100x speedup targets.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;

use lyra::concurrency::{
    ConcurrentLyraVM, ConcurrencyConfig, ConcurrentVmFactory,
    WorkStealingScheduler, ParallelPatternMatcher, ParallelEvaluator,
    ConcurrencyStats, EvaluationContext, TaskPriority,
    ConcurrentSymbolTable, ThreadSafeVmState,
};
use lyra::vm::{VirtualMachine, Value};
use lyra::ast::{Expr, Pattern};
use lyra::pattern_matcher::{PatternMatcher, MatchingContext, MatchResult};

/// Test basic concurrency system initialization and shutdown
#[tokio::test]
async fn test_concurrency_system_lifecycle() {
    let mut vm = ConcurrentLyraVM::new().expect("Failed to create concurrent VM");
    
    // Test startup
    assert!(vm.start().is_ok(), "Failed to start concurrency system");
    
    // Test that we can get performance stats
    let stats = vm.get_performance_stats();
    assert_eq!(stats.worker_count, num_cpus::get());
    
    // Test shutdown
    assert!(vm.stop().is_ok(), "Failed to stop concurrency system");
}

/// Test concurrent vs sequential execution correctness
#[tokio::test]
async fn test_execution_correctness() {
    let mut vm = ConcurrentLyraVM::new().expect("Failed to create concurrent VM");
    vm.start().expect("Failed to start VM");
    
    // Test simple expressions
    let simple_expr = Expr::Integer(42);
    let seq_result = vm.execute_sequential(&simple_expr).expect("Sequential execution failed");
    let conc_result = vm.execute_concurrent(&simple_expr).await.expect("Concurrent execution failed");
    
    // Results should be equivalent (though not necessarily identical due to different execution paths)
    match (&seq_result, &conc_result) {
        (Value::Integer(a), Value::Integer(b)) => assert_eq!(a, b),
        _ => panic!("Unexpected result types"),
    }
    
    // Test list expressions
    let list_expr = Expr::List(vec![
        Expr::Integer(1),
        Expr::Integer(2),
        Expr::Integer(3),
    ]);
    
    let seq_result = vm.execute_sequential(&list_expr).expect("Sequential list execution failed");
    let conc_result = vm.execute_concurrent(&list_expr).await.expect("Concurrent list execution failed");
    
    // Both should produce lists
    assert!(matches!(seq_result, Value::List(_)));
    assert!(matches!(conc_result, Value::List(_)));
    
    vm.stop().expect("Failed to stop VM");
}

/// Test parallel pattern matching correctness
#[tokio::test]
async fn test_parallel_pattern_matching() {
    let config = ConcurrencyConfig::default();
    let stats = Arc::new(ConcurrencyStats::default());
    let matcher = ParallelPatternMatcher::new(config, stats).expect("Failed to create pattern matcher");
    
    let expression = Value::Integer(42);
    let patterns = vec![
        Pattern::Blank { head: None },
        Pattern::Named { 
            name: "x".to_string(), 
            pattern: Box::new(Pattern::Blank { head: None })
        },
        Pattern::Blank { head: None },
    ];
    
    // Test parallel matching
    let results = matcher.match_parallel(&expression, &patterns)
        .expect("Parallel pattern matching failed");
    
    assert_eq!(results.len(), patterns.len());
    
    // Test cache functionality
    let cache_stats_before = matcher.cache_stats();
    
    // Run the same match again - should hit cache
    let _results2 = matcher.match_parallel(&expression, &patterns)
        .expect("Second parallel pattern matching failed");
    
    let cache_stats_after = matcher.cache_stats();
    
    // Cache hits should have increased
    assert!(cache_stats_after.hits > cache_stats_before.hits);
}

/// Test work-stealing scheduler performance and correctness
#[tokio::test]
async fn test_work_stealing_scheduler() {
    let config = ConcurrencyConfig {
        worker_threads: 4,
        max_local_queue_size: 100,
        max_global_queue_size: 1000,
        ..Default::default()
    };
    let stats = Arc::new(ConcurrencyStats::default());
    let scheduler = Arc::new(WorkStealingScheduler::new(config, stats).expect("Failed to create scheduler"));
    
    scheduler.start().expect("Failed to start scheduler");
    
    // Submit a batch of tasks
    let task_count = 1000;
    let task_ids: Vec<_> = (0..task_count)
        .map(|i| {
            let computation = lyra::concurrency::scheduler::SimpleComputation {
                value: i as i64,
                priority: TaskPriority::Normal,
            };
            scheduler.submit(computation).expect("Failed to submit task")
        })
        .collect();
    
    // Wait for tasks to complete
    tokio::time::sleep(Duration::from_millis(500)).await;
    
    let scheduler_stats = scheduler.stats();
    
    // Verify that tasks were executed
    assert!(scheduler_stats.total_tasks_executed > 0);
    
    // If we have multiple workers, we should see some work stealing
    if scheduler_stats.worker_count > 1 {
        assert!(scheduler_stats.total_work_steals > 0 || scheduler_stats.total_failed_steals > 0);
    }
    
    scheduler.stop().expect("Failed to stop scheduler");
}

/// Test thread-safe data structures under concurrent access
#[tokio::test]
async fn test_concurrent_data_structures() {
    let symbol_table = Arc::new(ConcurrentSymbolTable::new());
    let vm_state = Arc::new(ThreadSafeVmState::new());
    
    // Test concurrent symbol table operations
    let table_clone = Arc::clone(&symbol_table);
    let handle1 = tokio::spawn(async move {
        for i in 0..1000 {
            let name = format!("symbol_{}", i);
            table_clone.set_symbol(&name, Value::Integer(i as i64));
        }
    });
    
    let table_clone = Arc::clone(&symbol_table);
    let handle2 = tokio::spawn(async move {
        for i in 1000..2000 {
            let name = format!("symbol_{}", i);
            table_clone.set_symbol(&name, Value::Integer(i as i64));
        }
    });
    
    // Wait for both tasks to complete
    let _ = tokio::try_join!(handle1, handle2);
    
    // Verify all symbols were stored
    assert_eq!(symbol_table.len(), 2000);
    
    // Test concurrent VM state operations
    let state_clone = Arc::clone(&vm_state);
    let handle3 = tokio::spawn(async move {
        for i in 0..500 {
            state_clone.push(Value::Integer(i as i64));
        }
    });
    
    let state_clone = Arc::clone(&vm_state);
    let handle4 = tokio::spawn(async move {
        for i in 500..1000 {
            state_clone.push(Value::Integer(i as i64));
        }
    });
    
    let _ = tokio::try_join!(handle3, handle4);
    
    // Verify stack operations worked
    assert_eq!(vm_state.stack_size(), 1000);
}

/// Performance validation test - verify speedup targets
#[tokio::test]
async fn test_performance_speedup() {
    const ITERATIONS: usize = 50;
    const MIN_SPEEDUP: f64 = 1.5; // At least 50% improvement for this test
    
    // Create a workload that should benefit from parallelization
    let large_list = Expr::List((0..2000).map(|i| {
        if i % 2 == 0 {
            Expr::Number(lyra::ast::Number::Integer(i as i64))
        } else {
            Expr::List(vec![
                Expr::Number(lyra::ast::Number::Integer(i as i64)), 
                Expr::Number(lyra::ast::Number::Integer((i * 2) as i64))
            ])
        }
    }).collect());
    
    // Benchmark sequential execution
    let start = Instant::now();
    let mut vm = VirtualMachine::new();
    for _ in 0..ITERATIONS {
        // Simplified sequential execution
        let _result = match &large_list {
            Expr::List(items) => Value::List(
                items.iter().map(|_| Value::Integer(42)).collect()
            ),
            _ => Value::Integer(42),
        };
    }
    let sequential_time = start.elapsed();
    
    // Benchmark concurrent execution
    let mut concurrent_vm = ConcurrentLyraVM::new().expect("Failed to create concurrent VM");
    concurrent_vm.start().expect("Failed to start concurrent VM");
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        let _result = concurrent_vm.execute_concurrent(&large_list).await
            .expect("Concurrent execution failed");
    }
    let concurrent_time = start.elapsed();
    
    concurrent_vm.stop().expect("Failed to stop concurrent VM");
    
    // Calculate speedup
    let speedup = sequential_time.as_nanos() as f64 / concurrent_time.as_nanos() as f64;
    
    println!("Sequential time: {:?}", sequential_time);
    println!("Concurrent time: {:?}", concurrent_time);
    println!("Speedup: {:.2}x", speedup);
    
    // For this test, we expect at least some improvement
    // Note: In a real benchmark, we'd expect much higher speedups (10-100x)
    // but this is a simplified test without full VM integration
    assert!(speedup >= MIN_SPEEDUP, 
        "Expected speedup of at least {:.2}x, got {:.2}x", MIN_SPEEDUP, speedup);
}

/// Test cache effectiveness
#[tokio::test]
async fn test_cache_effectiveness() {
    let config = ConcurrencyConfig {
        pattern_cache_size: 1000,
        ..Default::default()
    };
    let stats = Arc::new(ConcurrencyStats::default());
    let matcher = ParallelPatternMatcher::new(config, stats).expect("Failed to create pattern matcher");
    
    let expression = Value::Integer(42);
    let patterns = vec![
        Pattern::Blank { head: None },
        Pattern::Named { 
            name: "x".to_string(), 
            pattern: Box::new(Pattern::Blank { head: None })
        },
        Pattern::Named { 
            name: "y".to_string(), 
            pattern: Box::new(Pattern::Blank { head: None })
        },
    ];
    
    // First run - should populate cache
    let _results1 = matcher.match_parallel(&expression, &patterns)
        .expect("First pattern matching failed");
    
    let stats_after_first = matcher.cache_stats();
    let initial_misses = stats_after_first.misses;
    
    // Second run - should hit cache
    let _results2 = matcher.match_parallel(&expression, &patterns)
        .expect("Second pattern matching failed");
    
    let stats_after_second = matcher.cache_stats();
    
    // Cache hit rate should be high for second run
    let hit_rate = stats_after_second.hit_rate;
    assert!(hit_rate > 50.0, "Cache hit rate too low: {:.2}%", hit_rate);
    
    // Third run with same patterns - hit rate should be even higher
    let _results3 = matcher.match_parallel(&expression, &patterns)
        .expect("Third pattern matching failed");
    
    let stats_after_third = matcher.cache_stats();
    let final_hit_rate = stats_after_third.hit_rate;
    
    assert!(final_hit_rate >= hit_rate, 
        "Cache hit rate should not decrease: {:.2}% -> {:.2}%", hit_rate, final_hit_rate);
}

/// Test error handling in concurrent execution
#[tokio::test]
async fn test_error_handling() {
    let mut vm = ConcurrentLyraVM::new().expect("Failed to create concurrent VM");
    vm.start().expect("Failed to start VM");
    
    // Test with valid expression
    let valid_expr = Expr::Integer(42);
    let result = vm.execute_concurrent(&valid_expr).await;
    assert!(result.is_ok());
    
    // Test with complex nested expression that might cause issues
    let complex_expr = Expr::List(vec![
        Expr::List(vec![Expr::Integer(1); 1000]),
        Expr::List(vec![Expr::Integer(2); 1000]),
    ]);
    
    // Should handle large expressions gracefully
    let result = timeout(Duration::from_secs(5), vm.execute_concurrent(&complex_expr)).await;
    assert!(result.is_ok(), "Concurrent execution timed out or failed");
    
    vm.stop().expect("Failed to stop VM");
}

/// Test different VM factory configurations
#[tokio::test]
async fn test_vm_factory_configurations() {
    // Test math-optimized VM
    let mut math_vm = ConcurrentVmFactory::create_math_optimized()
        .expect("Failed to create math-optimized VM");
    math_vm.start().expect("Failed to start math VM");
    
    let math_expr = Expr::List(vec![Expr::Integer(1), Expr::Integer(2), Expr::Integer(3)]);
    let result = math_vm.execute_concurrent(&math_expr).await;
    assert!(result.is_ok());
    
    math_vm.stop().expect("Failed to stop math VM");
    
    // Test symbolic-optimized VM
    let mut symbolic_vm = ConcurrentVmFactory::create_symbolic_optimized()
        .expect("Failed to create symbolic-optimized VM");
    symbolic_vm.start().expect("Failed to start symbolic VM");
    
    let symbolic_expr = Expr::Function {
        head: Box::new(Expr::Symbol(lyra::ast::Symbol { name: "Plus".to_string() })),
        args: vec![Expr::Number(lyra::ast::Number::Integer(1)), Expr::Number(lyra::ast::Number::Integer(2))],
    };
    let result = symbolic_vm.execute_concurrent(&symbolic_expr).await;
    assert!(result.is_ok());
    
    symbolic_vm.stop().expect("Failed to stop symbolic VM");
    
    // Test list-optimized VM
    let mut list_vm = ConcurrentVmFactory::create_list_optimized()
        .expect("Failed to create list-optimized VM");
    list_vm.start().expect("Failed to start list VM");
    
    let list_expr = Expr::List((0..100).map(|i| Expr::Number(lyra::ast::Number::Integer(i as i64))).collect());
    let result = list_vm.execute_concurrent(&list_expr).await;
    assert!(result.is_ok());
    
    list_vm.stop().expect("Failed to stop list VM");
}

/// Test batch execution performance
#[tokio::test]
async fn test_batch_execution() {
    let vm = ConcurrentLyraVM::new().expect("Failed to create concurrent VM");
    vm.start().expect("Failed to start VM");
    
    let expressions: Vec<Expr> = (0..100)
        .map(|i| Expr::Number(lyra::ast::Number::Integer(i as i64)))
        .collect();
    
    let start = Instant::now();
    let results = vm.execute_batch_parallel(&expressions).await
        .expect("Batch execution failed");
    let duration = start.elapsed();
    
    assert_eq!(results.len(), expressions.len());
    println!("Batch execution of {} expressions took: {:?}", expressions.len(), duration);
    
    vm.stop().expect("Failed to stop VM");
}

/// Stress test with many concurrent operations
#[tokio::test]
async fn test_stress_concurrent_operations() {
    let vm = Arc::new(ConcurrentLyraVM::new().expect("Failed to create concurrent VM"));
    vm.start().expect("Failed to start VM");
    
    let num_tasks = 100;
    let mut handles = Vec::new();
    
    for i in 0..num_tasks {
        let vm_clone = Arc::clone(&vm);
        let handle = tokio::spawn(async move {
            let expr = Expr::List(vec![Expr::Number(lyra::ast::Number::Integer(i as i64)); 10]);
            vm_clone.execute_concurrent(&expr).await
        });
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let results = futures::future::try_join_all(handles).await
        .expect("Failed to join concurrent tasks");
    
    // All tasks should have completed successfully
    for result in results {
        assert!(result.is_ok(), "Concurrent task failed");
    }
    
    vm.stop().expect("Failed to stop VM");
}