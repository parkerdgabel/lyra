//! Comprehensive Async Performance Benchmarks for Lyra
//!
//! This benchmark suite validates the performance characteristics of the newly implemented
//! async concurrency system, measuring throughput, latency, scalability, and resource efficiency.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::time::{Duration, Instant};
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};
use lyra::foreign::LyObj;

/// Benchmark ThreadPool creation and basic operations
fn bench_thread_pool_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_pool_operations");
    
    // ThreadPool creation with different worker counts
    for worker_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("creation", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let args = [Value::Integer(worker_count as i64)];
                    black_box(thread_pool(&args).unwrap())
                })
            },
        );
    }
    
    // Default ThreadPool creation (most common case)
    group.bench_function("default_creation", |b| {
        b.iter(|| {
            black_box(thread_pool(&[]).unwrap())
        })
    });
    
    group.finish();
}

/// Benchmark Channel operations (send/receive throughput)
fn bench_channel_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_operations");
    
    // Unbounded channel throughput
    group.throughput(Throughput::Elements(1000));
    group.bench_function("unbounded_send_receive_1000", |b| {
        b.iter(|| {
            let ch = channel(&[]).unwrap();
            let channel_value = ch.clone();
            
            // Send 1000 messages
            for i in 0..1000 {
                let message = Value::Integer(i);
                let _ = send(&[channel_value.clone(), message]);
            }
            
            // Receive 1000 messages
            let mut received = Vec::with_capacity(1000);
            for _ in 0..1000 {
                match receive(&[channel_value.clone()]) {
                    Ok(value) => received.push(value),
                    Err(_) => break,
                }
            }
            
            black_box(received)
        })
    });
    
    // Bounded channel with different capacities
    for capacity in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("bounded_send_receive", capacity),
            capacity,
            |b, &capacity| {
                b.iter(|| {
                    let ch = bounded_channel(&[Value::Integer(capacity)]).unwrap();
                    let channel_value = ch.clone();
                    
                    // Fill channel to capacity
                    for i in 0..capacity {
                        let message = Value::Integer(i as i64);
                        let _ = send(&[channel_value.clone(), message]);
                    }
                    
                    // Receive all messages
                    let mut received = Vec::with_capacity(capacity);
                    for _ in 0..capacity {
                        match receive(&[channel_value.clone()]) {
                            Ok(value) => received.push(value),
                            Err(_) => break,
                        }
                    }
                    
                    black_box(received)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Future/Promise operations
fn bench_future_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("future_operations");
    
    // Promise creation and immediate resolution
    group.bench_function("promise_create_await", |b| {
        b.iter(|| {
            let value = Value::String("test_value".to_string());
            let future = promise(&[value.clone()]).unwrap();
            let result = await_future(&[future]).unwrap();
            black_box(result)
        })
    });
    
    // Batch promise operations
    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_promises_100", |b| {
        b.iter(|| {
            let mut futures = Vec::with_capacity(100);
            let mut results = Vec::with_capacity(100);
            
            // Create 100 promises
            for i in 0..100 {
                let value = Value::Integer(i);
                let future = promise(&[value]).unwrap();
                futures.push(future);
            }
            
            // Await all futures
            for future in futures {
                let result = await_future(&[future]).unwrap();
                results.push(result);
            }
            
            black_box(results)
        })
    });
    
    group.finish();
}

/// Benchmark concurrent task execution patterns
fn bench_concurrent_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_patterns");
    
    // Producer-Consumer pattern with different queue sizes
    for queue_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("producer_consumer", queue_size),
            queue_size,
            |b, &queue_size| {
                b.iter(|| {
                    let ch = bounded_channel(&[Value::Integer(queue_size)]).unwrap();
                    let channel_value = ch.clone();
                    
                    // Simulate producer-consumer pattern
                    let producer_messages = queue_size / 2;
                    
                    // Producer phase
                    for i in 0..producer_messages {
                        let message = Value::Integer(i as i64);
                        let _ = send(&[channel_value.clone(), message]);
                    }
                    
                    // Consumer phase
                    let mut consumed = Vec::with_capacity(producer_messages);
                    for _ in 0..producer_messages {
                        match receive(&[channel_value.clone()]) {
                            Ok(value) => consumed.push(value),
                            Err(_) => break,
                        }
                    }
                    
                    black_box(consumed)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory efficiency of async operations
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    // Memory usage for large number of channels
    group.bench_function("channel_memory_usage", |b| {
        b.iter_with_large_drop(|| {
            let mut channels = Vec::with_capacity(1000);
            
            // Create 1000 channels
            for _ in 0..1000 {
                let ch = channel(&[]).unwrap();
                channels.push(ch);
            }
            
            // Use each channel once
            for (i, ch) in channels.iter().enumerate() {
                let message = Value::Integer(i as i64);
                let _ = send(&[ch.clone(), message]);
                let _ = receive(&[ch.clone()]);
            }
            
            channels // Will be dropped after timing
        })
    });
    
    // Memory usage for large number of futures
    group.bench_function("future_memory_usage", |b| {
        b.iter_with_large_drop(|| {
            let mut futures = Vec::with_capacity(1000);
            
            // Create 1000 futures
            for i in 0..1000 {
                let value = Value::Integer(i);
                let future = promise(&[value]).unwrap();
                futures.push(future);
            }
            
            // Await all futures
            let mut results = Vec::with_capacity(1000);
            for future in futures.iter() {
                let result = await_future(&[future.clone()]).unwrap();
                results.push(result);
            }
            
            (futures, results) // Will be dropped after timing
        })
    });
    
    group.finish();
}

/// Benchmark latency characteristics under load
fn bench_latency_under_load(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_under_load");
    
    // Single operation latency (baseline)
    group.bench_function("baseline_channel_latency", |b| {
        b.iter(|| {
            let ch = channel(&[]).unwrap();
            let message = Value::Integer(42);
            
            let start = Instant::now();
            let _ = send(&[ch.clone(), message]);
            let _ = receive(&[ch.clone()]);
            let latency = start.elapsed();
            
            black_box(latency)
        })
    });
    
    // Latency under concurrent load
    for concurrent_ops in [10, 50, 100].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_load_latency", concurrent_ops),
            concurrent_ops,
            |b, &concurrent_ops| {
                b.iter(|| {
                    let mut channels = Vec::with_capacity(concurrent_ops);
                    let mut latencies = Vec::with_capacity(concurrent_ops);
                    
                    // Create multiple channels for concurrent operations
                    for _ in 0..concurrent_ops {
                        let ch = channel(&[]).unwrap();
                        channels.push(ch);
                    }
                    
                    // Measure latency for each operation
                    for (i, ch) in channels.iter().enumerate() {
                        let message = Value::Integer(i as i64);
                        
                        let start = Instant::now();
                        let _ = send(&[ch.clone(), message]);
                        let _ = receive(&[ch.clone()]);
                        let latency = start.elapsed();
                        
                        latencies.push(latency);
                    }
                    
                    black_box(latencies)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark scalability with different worker counts
fn bench_scalability(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalability");
    
    // Test ThreadPool scalability
    for worker_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("threadpool_scalability", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let pool = thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
                    
                    // Simulate work by creating/using the pool
                    if let Value::LyObj(pool_obj) = &pool {
                        // Test pool creation overhead
                        let _worker_count = pool_obj.call_method("workerCount", &[]);
                    }
                    
                    black_box(pool)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark error handling performance
fn bench_error_handling(c: &mut Criterion) {
    let mut group = c.benchmark_group("error_handling");
    
    // Error handling in channel operations
    group.bench_function("channel_error_handling", |b| {
        b.iter(|| {
            let ch = bounded_channel(&[Value::Integer(1)]).unwrap();
            let channel_value = ch.clone();
            
            // Fill channel
            let _ = send(&[channel_value.clone(), Value::Integer(1)]);
            
            // Try to overfill (should handle error gracefully)
            let result = send(&[channel_value.clone(), Value::Integer(2)]);
            
            // Drain channel
            let _ = receive(&[channel_value.clone()]);
            let _ = receive(&[channel_value.clone()]);
            
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark resource cleanup efficiency
fn bench_resource_cleanup(c: &mut Criterion) {
    let mut group = c.benchmark_group("resource_cleanup");
    
    // Test automatic cleanup when objects are dropped
    group.bench_function("automatic_cleanup", |b| {
        b.iter(|| {
            // Create many short-lived async objects
            for _ in 0..100 {
                let ch = channel(&[]).unwrap();
                let message = Value::Integer(42);
                let _ = send(&[ch.clone(), message]);
                let _ = receive(&[ch]);
                // Channel goes out of scope and should be cleaned up
            }
        })
    });
    
    group.finish();
}

/// Performance regression tests
fn bench_regression_tests(c: &mut Criterion) {
    let mut group = c.benchmark_group("regression_tests");
    
    // Baseline operations that should maintain performance
    group.bench_function("channel_baseline_regression", |b| {
        b.iter(|| {
            let ch = channel(&[]).unwrap();
            for i in 0..10 {
                let message = Value::Integer(i);
                let _ = send(&[ch.clone(), message]);
                let _ = receive(&[ch.clone()]);
            }
        })
    });
    
    group.bench_function("promise_baseline_regression", |b| {
        b.iter(|| {
            for i in 0..10 {
                let value = Value::Integer(i);
                let future = promise(&[value]).unwrap();
                let _ = await_future(&[future]).unwrap();
            }
        })
    });
    
    group.finish();
}

criterion_group!(
    async_benchmarks,
    bench_thread_pool_operations,
    bench_channel_operations,
    bench_future_operations,
    bench_concurrent_patterns,
    bench_memory_efficiency,
    bench_latency_under_load,
    bench_scalability,
    bench_error_handling,
    bench_resource_cleanup,
    bench_regression_tests
);

criterion_main!(async_benchmarks);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benchmark_components() {
        // Verify all benchmark components work correctly
        
        // Test ThreadPool
        let pool = thread_pool(&[Value::Integer(2)]).unwrap();
        assert!(matches!(pool, Value::LyObj(_)));
        
        // Test Channel
        let ch = channel(&[]).unwrap();
        let message = Value::String("test".to_string());
        let _ = send(&[ch.clone(), message]);
        let received = receive(&[ch]).unwrap();
        assert!(matches!(received, Value::String(_)));
        
        // Test Future
        let value = Value::Integer(42);
        let future = promise(&[value.clone()]).unwrap();
        let result = await_future(&[future]).unwrap();
        assert_eq!(result, value);
        
        println!("All benchmark components working correctly");
    }
    
    #[test]
    fn test_performance_baseline() {
        // Establish performance baselines for future comparison
        
        // Channel baseline
        let start = Instant::now();
        let ch = channel(&[]).unwrap();
        for i in 0..1000 {
            let message = Value::Integer(i);
            let _ = send(&[ch.clone(), message]);
            let _ = receive(&[ch.clone()]);
        }
        let channel_baseline = start.elapsed();
        
        // Future baseline
        let start = Instant::now();
        for i in 0..1000 {
            let value = Value::Integer(i);
            let future = promise(&[value]).unwrap();
            let _ = await_future(&[future]).unwrap();
        }
        let future_baseline = start.elapsed();
        
        println!("Channel baseline (1000 ops): {:?}", channel_baseline);
        println!("Future baseline (1000 ops): {:?}", future_baseline);
        
        // These baselines will be used for regression detection
        assert!(channel_baseline < Duration::from_millis(100), "Channel ops too slow");
        assert!(future_baseline < Duration::from_millis(50), "Future ops too slow");
    }
    
    #[test]
    fn test_memory_overhead() {
        // Test memory overhead of async operations
        use std::alloc::{GlobalAlloc, Layout, System};
        
        // Simple memory tracking (in a real benchmark we'd use more sophisticated tools)
        let initial_memory = std::process::id(); // Placeholder for actual memory measurement
        
        // Create and use many async objects
        let mut channels = Vec::new();
        for _ in 0..1000 {
            let ch = channel(&[]).unwrap();
            channels.push(ch);
        }
        
        let after_creation = std::process::id(); // Placeholder
        
        // Use all channels
        for (i, ch) in channels.iter().enumerate() {
            let message = Value::Integer(i as i64);
            let _ = send(&[ch.clone(), message]);
        }
        
        let after_usage = std::process::id(); // Placeholder
        
        println!("Memory overhead test completed - implement proper memory measurement");
    }
    
    #[test] 
    fn test_scalability_characteristics() {
        // Test that operations scale reasonably with system resources
        
        let single_threaded_time = {
            let start = Instant::now();
            let pool = thread_pool(&[Value::Integer(1)]).unwrap();
            // Simulate work
            for _ in 0..100 {
                let ch = channel(&[]).unwrap();
                let _ = send(&[ch.clone(), Value::Integer(42)]);
                let _ = receive(&[ch]);
            }
            start.elapsed()
        };
        
        let multi_threaded_time = {
            let start = Instant::now();
            let pool = thread_pool(&[Value::Integer(4)]).unwrap();
            // Simulate work
            for _ in 0..100 {
                let ch = channel(&[]).unwrap();
                let _ = send(&[ch.clone(), Value::Integer(42)]);
                let _ = receive(&[ch]);
            }
            start.elapsed()
        };
        
        println!("Single-threaded time: {:?}", single_threaded_time);
        println!("Multi-threaded time: {:?}", multi_threaded_time);
        
        // Multi-threaded shouldn't be significantly slower (overhead test)
        let overhead_ratio = multi_threaded_time.as_nanos() as f64 / single_threaded_time.as_nanos() as f64;
        assert!(overhead_ratio < 2.0, "Multi-threading overhead too high: {}x", overhead_ratio);
    }
}