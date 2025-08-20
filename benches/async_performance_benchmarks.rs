use std::time::{Duration, Instant};
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{create_thread_pool, parallel};
use lyra::foreign::LyObj;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// Baseline performance benchmarks for async operations
// These will be used to validate the 2-5x performance improvements

/// CPU-bound task for benchmarking - expensive factorial calculation
fn factorial_task(n: i64) -> Value {
    if n <= 1 {
        Value::Integer(1)
    } else {
        let mut result = 1i64;
        for i in 2..=n {
            result = result.saturating_mul(i);
        }
        Value::Integer(result)
    }
}

/// Memory-bound task - large vector operations
fn memory_bound_task(size: usize) -> Value {
    let mut vec = Vec::with_capacity(size);
    for i in 0..size {
        vec.push(i as i64);
    }
    
    // Perform some memory-intensive operations
    vec.sort();
    vec.reverse();
    let sum: i64 = vec.iter().sum();
    
    Value::Integer(sum)
}

/// Mixed workload task - combination of CPU and memory operations
fn mixed_workload_task(cpu_factor: i64, memory_factor: usize) -> Value {
    // CPU work
    let mut cpu_result = 1i64;
    for i in 1..=cpu_factor {
        cpu_result = cpu_result.saturating_mul(i % 100 + 1);
    }
    
    // Memory work
    let mut vec = Vec::with_capacity(memory_factor);
    for i in 0..memory_factor {
        vec.push((i as i64 * cpu_result) % 1000);
    }
    vec.sort();
    
    Value::Integer(vec.len() as i64 + cpu_result % 1000)
}

fn bench_thread_pool_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("thread_pool_creation");
    
    for worker_count in [1, 2, 4, 8, 16].iter() {
        group.bench_with_input(
            BenchmarkId::new("create_pool", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    black_box(create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_cpu_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_cpu_bound");
    group.throughput(Throughput::Elements(1000));
    
    // Create test data - list of numbers for factorial calculation
    let test_data: Vec<Value> = (10..20).map(|i| Value::Integer(i)).collect();
    let function = Value::Function("Factorial".to_string());
    
    for worker_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("factorial_parallel", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
                    let args = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool];
                    black_box(parallel(&args).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_memory_bound(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_memory_bound");
    group.throughput(Throughput::Elements(100));
    
    // Create test data - various memory allocation sizes
    let test_data: Vec<Value> = (1000..2000).step_by(100).map(|i| Value::Integer(i)).collect();
    let function = Value::Function("MemoryBound".to_string());
    
    for worker_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("memory_bound_parallel", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
                    let args = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool];
                    black_box(parallel(&args).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_parallel_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_mixed_workload");
    group.throughput(Throughput::Elements(50));
    
    // Create test data - mixed CPU/memory parameters
    let test_data: Vec<Value> = (0..50).map(|i| {
        Value::List(vec![
            Value::Integer(10 + i % 20), // CPU factor
            Value::Integer(500 + i * 50), // Memory factor
        ])
    }).collect();
    let function = Value::Function("MixedWorkload".to_string());
    
    for worker_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("mixed_workload_parallel", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
                    let args = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool];
                    black_box(parallel(&args).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_task_submission_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("task_submission_rate");
    group.throughput(Throughput::Elements(1000));
    
    for worker_count in [1, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("submit_1000_tasks", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    if let Value::LyObj(pool) = create_thread_pool(&[Value::Integer(*worker_count as i64)]).unwrap() {
                        let mut task_ids = Vec::new();
                        
                        // Submit 1000 lightweight tasks
                        for i in 0..1000 {
                            let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(i), Value::Integer(1)];
                            if let Ok(Value::Integer(task_id)) = pool.call_method("Submit", &submit_args) {
                                task_ids.push(task_id);
                            }
                        }
                        
                        // Wait for all tasks to complete
                        for task_id in task_ids {
                            loop {
                                if let Ok(Value::Boolean(true)) = pool.call_method("IsCompleted", &[Value::Integer(task_id)]) {
                                    let _ = pool.call_method("GetResult", &[Value::Integer(task_id)]);
                                    break;
                                }
                                std::thread::sleep(Duration::from_millis(1));
                            }
                        }
                        
                        black_box(())
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn bench_load_balancing(c: &mut Criterion) {
    let mut group = c.benchmark_group("load_balancing");
    
    // Test with uneven workload distribution
    let uneven_data: Vec<Value> = (0..100).map(|i| {
        // Some tasks are much more expensive than others
        let factor = if i % 10 == 0 { 100 } else { 1 };
        Value::Integer(factor)
    }).collect();
    
    let function = Value::Function("Factorial".to_string());
    
    for worker_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("uneven_workload", worker_count),
            worker_count,
            |b, &worker_count| {
                b.iter(|| {
                    let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
                    let args = vec![Value::List(vec![function.clone(), Value::List(uneven_data.clone())]), pool];
                    black_box(parallel(&args).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_chunk_size_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_size_impact");
    
    // Large dataset to test chunking effectiveness
    let large_data: Vec<Value> = (0..10000).map(|i| Value::Integer(i % 100)).collect();
    let function = Value::Function("Add".to_string());
    
    // Compare performance with different data sizes (affects chunking)
    for data_size in [100, 1000, 10000].iter() {
        let test_data: Vec<Value> = large_data[0..*data_size].to_vec();
        
        group.bench_with_input(
            BenchmarkId::new("chunk_performance", data_size),
            data_size,
            |b, _data_size| {
                b.iter(|| {
                    let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
                    let args = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool];
                    black_box(parallel(&args).unwrap())
                })
            },
        );
    }
    
    group.finish();
}

fn bench_latency_vs_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_vs_throughput");
    
    // Single task latency
    group.bench_function("single_task_latency", |b| {
        b.iter(|| {
            let pool = create_thread_pool(&[Value::Integer(1)]).unwrap();
            if let Value::LyObj(pool_obj) = pool {
                let start = Instant::now();
                
                let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(42), Value::Integer(1)];
                if let Ok(Value::Integer(task_id)) = pool_obj.call_method("Submit", &submit_args) {
                    loop {
                        if let Ok(Value::Boolean(true)) = pool_obj.call_method("IsCompleted", &[Value::Integer(task_id)]) {
                            let _ = pool_obj.call_method("GetResult", &[Value::Integer(task_id)]);
                            break;
                        }
                        std::thread::sleep(Duration::from_millis(1));
                    }
                }
                
                black_box(start.elapsed())
            }
        })
    });
    
    // Batch throughput
    group.throughput(Throughput::Elements(100));
    group.bench_function("batch_throughput", |b| {
        b.iter(|| {
            let test_data: Vec<Value> = (0..100).map(|i| Value::Integer(i)).collect();
            let function = Value::Function("Add".to_string());
            let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
            let args = vec![Value::List(vec![function, Value::List(test_data)]), pool];
            black_box(parallel(&args).unwrap())
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_thread_pool_creation,
    bench_parallel_cpu_bound,
    bench_parallel_memory_bound,
    bench_parallel_mixed_workload,
    bench_task_submission_rate,
    bench_load_balancing,
    bench_chunk_size_impact,
    bench_latency_vs_throughput
);

criterion_main!(benches);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_baseline_performance_metrics() {
        // Verify that the benchmark functions work correctly
        let test_data: Vec<Value> = (1..10).map(|i| Value::Integer(i)).collect();
        let function = Value::Function("Add".to_string());
        let pool = create_thread_pool(&[Value::Integer(2)]).unwrap();
        let args = vec![Value::List(vec![function, Value::List(test_data)]), pool];
        
        let start = Instant::now();
        let result = parallel(&args).unwrap();
        let elapsed = start.elapsed();
        
        println!("Baseline parallel execution took: {:?}", elapsed);
        
        if let Value::List(results) = result {
            assert_eq!(results.len(), 9);
            println!("Processed {} items successfully", results.len());
        }
    }
    
    #[test]
    fn test_busy_waiting_detection() {
        // This test will help us measure the impact of busy-waiting
        let pool = create_thread_pool(&[Value::Integer(1)]).unwrap();
        
        if let Value::LyObj(pool_obj) = pool {
            let start = Instant::now();
            
            // Submit a simple task
            let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(1), Value::Integer(1)];
            if let Ok(Value::Integer(task_id)) = pool_obj.call_method("Submit", &submit_args) {
                // Measure how long we spend waiting
                let wait_start = Instant::now();
                loop {
                    if let Ok(Value::Boolean(true)) = pool_obj.call_method("IsCompleted", &[Value::Integer(task_id)]) {
                        let _ = pool_obj.call_method("GetResult", &[Value::Integer(task_id)]);
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
                let wait_time = wait_start.elapsed();
                
                println!("Time spent waiting for task completion: {:?}", wait_time);
                println!("Total task execution time: {:?}", start.elapsed());
                
                // The wait time should be much shorter after optimization
                assert!(wait_time.as_millis() >= 1, "Should have some wait time due to busy-waiting");
            }
        }
    }
    
    #[test]
    fn test_load_balancing_baseline() {
        // Test with uneven workload to measure load balancing effectiveness
        let uneven_data: Vec<Value> = (0..20).map(|i| {
            // Alternate between light and heavy tasks
            let factor = if i % 2 == 0 { 5 } else { 15 };
            Value::Integer(factor)
        }).collect();
        
        let function = Value::Function("Factorial".to_string());
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
        let args = vec![Value::List(vec![function, Value::List(uneven_data)]), pool];
        
        let start = Instant::now();
        let _result = parallel(&args).unwrap();
        let elapsed = start.elapsed();
        
        println!("Uneven workload execution time: {:?}", elapsed);
        
        // After work-stealing optimization, this should be much faster
        // as heavy tasks won't block entire worker threads
    }
}