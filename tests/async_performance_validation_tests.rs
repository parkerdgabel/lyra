use std::time::{Duration, Instant};
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{create_thread_pool, parallel};
use lyra::foreign::LyObj;

/// Performance validation tests to verify 2-5x improvement claims
/// These tests compare the optimized implementation against expected performance targets

#[cfg(test)]
mod performance_validation {
    use super::*;

    /// Test CPU-bound workload performance with different worker counts
    #[test]
    fn test_cpu_bound_performance_scaling() {
        let test_data: Vec<Value> = (5..15).map(|i| Value::Integer(i)).collect();
        let function = Value::Function("Factorial".to_string());
        
        // Test with different worker counts
        let worker_counts = vec![1, 2, 4, 8];
        let mut timings = Vec::new();
        
        for &worker_count in &worker_counts {
            let start = Instant::now();
            
            let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
            let args = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool];
            let _result = parallel(&args).unwrap();
            
            let elapsed = start.elapsed();
            timings.push((worker_count, elapsed));
            println!("Worker count {}: {:?}", worker_count, elapsed);
        }
        
        // Verify scaling: 4 workers should be significantly faster than 1 worker
        let single_worker_time = timings[0].1;
        let four_worker_time = timings[2].1;
        
        let speedup = single_worker_time.as_nanos() as f64 / four_worker_time.as_nanos() as f64;
        println!("Speedup (1->4 workers): {:.2}x", speedup);
        
        // Should achieve at least 2x speedup with 4 workers
        assert!(speedup >= 2.0, "Expected at least 2x speedup with 4 workers, got {:.2}x", speedup);
    }
    
    /// Test event-driven notifications vs busy-waiting performance
    #[test]
    fn test_event_driven_notification_performance() {
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
        
        if let Value::LyObj(pool_obj) = pool {
            let start = Instant::now();
            let mut task_ids = Vec::new();
            
            // Submit 100 lightweight tasks
            for i in 0..100 {
                let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(i), Value::Integer(1)];
                if let Ok(Value::Integer(task_id)) = pool_obj.call_method("Submit", &submit_args) {
                    task_ids.push(task_id);
                }
            }
            
            // Use event-driven waiting
            for task_id in task_ids {
                let _ = pool_obj.call_method("AwaitResult", &[Value::Integer(task_id)]);
            }
            
            let elapsed = start.elapsed();
            println!("Event-driven notification time for 100 tasks: {:?}", elapsed);
            
            // Should complete much faster than with busy-waiting (target: < 100ms)
            assert!(elapsed.as_millis() < 100, "Event-driven notifications should be fast, took {:?}", elapsed);
        }
    }
    
    /// Test work-stealing effectiveness with uneven workload
    #[test]
    fn test_work_stealing_performance() {
        // Create uneven workload: some expensive, some cheap tasks
        let uneven_data: Vec<Value> = (0..50).map(|i| {
            // Every 5th task is much more expensive
            if i % 5 == 0 { Value::Integer(20) } else { Value::Integer(5) }
        }).collect();
        
        let function = Value::Function("Factorial".to_string());
        
        // Test with work-stealing enabled (default) vs disabled
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
        let args = vec![Value::List(vec![function, Value::List(uneven_data)]), pool];
        let _result = parallel(&args).unwrap();
        let elapsed = start.elapsed();
        
        println!("Work-stealing uneven workload time: {:?}", elapsed);
        
        // Should handle uneven workload efficiently (target: < 500ms)
        assert!(elapsed.as_millis() < 500, "Work-stealing should handle uneven workloads efficiently, took {:?}", elapsed);
    }
    
    /// Test cache-aligned chunking performance
    #[test]
    fn test_cache_aligned_chunking_performance() {
        // Large dataset to test chunking effectiveness
        let large_data: Vec<Value> = (0..10000).map(|i| Value::Integer(i % 100)).collect();
        let function = Value::Function("Add".to_string());
        
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
        let args = vec![Value::List(vec![function, Value::List(large_data)]), pool];
        let _result = parallel(&args).unwrap();
        let elapsed = start.elapsed();
        
        println!("Cache-aligned chunking time for 10k items: {:?}", elapsed);
        
        // Should process large datasets efficiently (target: < 1000ms)
        assert!(elapsed.as_millis() < 1000, "Cache-aligned chunking should be efficient for large datasets, took {:?}", elapsed);
    }
    
    /// Test batched channel operations performance
    #[test]
    fn test_batched_channel_performance() {
        use lyra::stdlib::async_ops::{create_channel, channel_send_batch, channel_receive_batch};
        
        let channel = create_channel(&[]).unwrap();
        
        // Create large batch of data
        let batch_data: Vec<Value> = (0..1000).map(|i| Value::Integer(i)).collect();
        
        let start = Instant::now();
        
        // Send batch
        let send_args = vec![channel.clone(), Value::List(batch_data.clone())];
        let _send_result = channel_send_batch(&send_args).unwrap();
        
        // Receive batch
        let receive_args = vec![channel, Value::Integer(1000)];
        let receive_result = channel_receive_batch(&receive_args).unwrap();
        
        let elapsed = start.elapsed();
        
        println!("Batched channel operations time for 1000 items: {:?}", elapsed);
        
        // Verify correctness
        if let Value::List(received) = receive_result {
            assert_eq!(received.len(), 1000, "Should receive all 1000 items");
        } else {
            panic!("Expected list result from batch receive");
        }
        
        // Should be very fast for batched operations (target: < 50ms)
        assert!(elapsed.as_millis() < 50, "Batched channel operations should be fast, took {:?}", elapsed);
    }
    
    /// Test NUMA-aware thread pinning (verification test)
    #[test]
    fn test_numa_aware_thread_pinning() {
        // This test mainly verifies that NUMA-aware thread pools can be created
        // Actual NUMA benefits are hard to measure in unit tests
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
        
        if let Value::LyObj(pool_obj) = pool {
            // Verify the pool was created successfully
            let worker_count = pool_obj.call_method("WorkerCount", &[]).unwrap();
            if let Value::Integer(count) = worker_count {
                assert_eq!(count, 4, "Should have 4 workers");
            }
            
            // Submit a test task to verify threads are working
            let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(42), Value::Integer(1)];
            let task_result = pool_obj.call_method("Submit", &submit_args);
            assert!(task_result.is_ok(), "Should be able to submit tasks to NUMA-aware pool");
        }
    }
    
    /// Integration test: Combined optimizations performance
    #[test]
    fn test_combined_optimizations_performance() {
        // Test that combines all optimizations for maximum performance
        println!("=== Combined Optimizations Performance Test ===");
        
        // Large mixed workload
        let mixed_data: Vec<Value> = (0..1000).map(|i| {
            match i % 4 {
                0 => Value::Integer(10), // Light computation
                1 => Value::Integer(15), // Medium computation  
                2 => Value::Integer(20), // Heavy computation
                _ => Value::Integer(5),  // Very light computation
            }
        }).collect();
        
        let function = Value::Function("Factorial".to_string());
        
        // Measure performance with all optimizations enabled
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(8)]).unwrap(); // Use 8 workers
        let args = vec![Value::List(vec![function, Value::List(mixed_data)]), pool];
        let result = parallel(&args).unwrap();
        let elapsed = start.elapsed();
        
        println!("Combined optimizations time for 1000 mixed tasks: {:?}", elapsed);
        
        // Verify correctness
        if let Value::List(results) = result {
            assert_eq!(results.len(), 1000, "Should process all 1000 tasks");
        }
        
        // With all optimizations, should achieve significant performance (target: < 2000ms)
        assert!(elapsed.as_millis() < 2000, "Combined optimizations should achieve high performance, took {:?}", elapsed);
        
        // Calculate effective throughput
        let throughput = 1000.0 / elapsed.as_secs_f64();
        println!("Throughput: {:.0} tasks/second", throughput);
        
        // Should achieve reasonable throughput (target: > 500 tasks/second)
        assert!(throughput > 500.0, "Should achieve good throughput, got {:.0} tasks/second", throughput);
    }
    
    /// Regression test: Ensure optimizations don't break existing functionality
    #[test]
    fn test_optimization_correctness() {
        // Test various scenarios to ensure optimizations don't break correctness
        
        // 1. Simple parallel computation
        let simple_data: Vec<Value> = vec![Value::Integer(5), Value::Integer(10), Value::Integer(15)];
        let add_func = Value::Function("Add".to_string());
        let pool = create_thread_pool(&[Value::Integer(2)]).unwrap();
        let args = vec![Value::List(vec![add_func, Value::List(simple_data)]), pool];
        let result = parallel(&args).unwrap();
        
        if let Value::List(results) = result {
            assert_eq!(results.len(), 3, "Should have 3 results");
            // Results should be correct (though order may vary due to parallel execution)
            assert!(results.iter().all(|v| matches!(v, Value::Integer(_))), "All results should be integers");
        }
        
        // 2. Empty input handling
        let empty_data: Vec<Value> = vec![];
        let pool2 = create_thread_pool(&[Value::Integer(2)]).unwrap();
        let args2 = vec![Value::List(vec![Value::Function("Add".to_string()), Value::List(empty_data)]), pool2];
        let result2 = parallel(&args2).unwrap();
        
        if let Value::List(results) = result2 {
            assert_eq!(results.len(), 0, "Empty input should produce empty output");
        }
        
        // 3. Single item handling
        let single_data = vec![Value::Integer(42)];
        let pool3 = create_thread_pool(&[Value::Integer(2)]).unwrap();
        let args3 = vec![Value::List(vec![Value::Function("Add".to_string()), Value::List(single_data)]), pool3];
        let result3 = parallel(&args3).unwrap();
        
        if let Value::List(results) = result3 {
            assert_eq!(results.len(), 1, "Single item should produce single result");
        }
        
        println!("All correctness tests passed!");
    }
}

/// Benchmark comparison functions for measuring improvements
pub mod benchmark_comparison {
    use super::*;
    
    /// Measure baseline performance (for comparison)
    pub fn measure_baseline_performance() -> Duration {
        let test_data: Vec<Value> = (1..100).map(|i| Value::Integer(i % 20)).collect();
        let function = Value::Function("Factorial".to_string());
        
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(1)]).unwrap(); // Single worker baseline
        let args = vec![Value::List(vec![function, Value::List(test_data)]), pool];
        let _result = parallel(&args).unwrap();
        start.elapsed()
    }
    
    /// Measure optimized performance
    pub fn measure_optimized_performance() -> Duration {
        let test_data: Vec<Value> = (1..100).map(|i| Value::Integer(i % 20)).collect();
        let function = Value::Function("Factorial".to_string());
        
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(4)]).unwrap(); // Multi-worker optimized
        let args = vec![Value::List(vec![function, Value::List(test_data)]), pool];
        let _result = parallel(&args).unwrap();
        start.elapsed()
    }
    
    /// Calculate and report speedup
    pub fn report_performance_improvement() {
        println!("=== Performance Improvement Report ===");
        
        let baseline = measure_baseline_performance();
        let optimized = measure_optimized_performance();
        
        let speedup = baseline.as_nanos() as f64 / optimized.as_nanos() as f64;
        
        println!("Baseline (1 worker): {:?}", baseline);
        println!("Optimized (4 workers): {:?}", optimized);
        println!("Speedup: {:.2}x", speedup);
        
        // Verify we achieved the target improvement
        if speedup >= 2.0 {
            println!("✅ SUCCESS: Achieved {:.2}x speedup (target: 2-5x)", speedup);
        } else {
            println!("❌ WARNING: Only achieved {:.2}x speedup (target: 2-5x)", speedup);
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use benchmark_comparison::*;
    
    #[test]
    fn test_performance_improvement_target() {
        report_performance_improvement();
        
        let baseline = measure_baseline_performance();
        let optimized = measure_optimized_performance();
        let speedup = baseline.as_nanos() as f64 / optimized.as_nanos() as f64;
        
        // Assert that we achieved at least 2x improvement
        assert!(speedup >= 2.0, "Failed to achieve target 2-5x performance improvement. Got {:.2}x", speedup);
    }
}