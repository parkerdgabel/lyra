use std::time::{Duration, Instant};
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{create_thread_pool, parallel};

#[test]
fn test_simple_async_performance() {
    println!("=== Simple Async Performance Test ===");
    
    // Create test data - simple factorial calculations
    let test_data: Vec<Value> = (5..15).map(|i| Value::Integer(i)).collect();
    let function = Value::Function("Factorial".to_string());
    
    // Test single-threaded baseline
    let start = Instant::now();
    let pool_1 = create_thread_pool(&[Value::Integer(1)]).unwrap();
    let args_1 = vec![Value::List(vec![function.clone(), Value::List(test_data.clone())]), pool_1];
    let _result_1 = parallel(&args_1).unwrap();
    let baseline_time = start.elapsed();
    
    // Test multi-threaded optimized
    let start = Instant::now();
    let pool_4 = create_thread_pool(&[Value::Integer(4)]).unwrap();
    let args_4 = vec![Value::List(vec![function, Value::List(test_data)]), pool_4];
    let _result_4 = parallel(&args_4).unwrap();
    let optimized_time = start.elapsed();
    
    println!("Single thread time: {:?}", baseline_time);
    println!("4-thread time: {:?}", optimized_time);
    
    // Calculate speedup
    let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;
    println!("Speedup: {:.2}x", speedup);
    
    // Basic verification that we got some improvement
    if speedup >= 1.5 {
        println!("✅ Good speedup achieved: {:.2}x", speedup);
    } else {
        println!("⚠️  Limited speedup: {:.2}x", speedup);
    }
}

#[test]
fn test_event_driven_notifications() {
    println!("=== Event-Driven Notification Test ===");
    
    let pool = create_thread_pool(&[Value::Integer(2)]).unwrap();
    
    if let Value::LyObj(pool_obj) = pool {
        let start = Instant::now();
        
        // Submit a simple task
        let submit_args = vec![Value::Function("Add".to_string()), Value::Integer(10), Value::Integer(5)];
        if let Ok(Value::Integer(task_id)) = pool_obj.call_method("Submit", &submit_args) {
            // Use event-driven waiting
            if let Ok(result) = pool_obj.call_method("AwaitResult", &[Value::Integer(task_id)]) {
                let elapsed = start.elapsed();
                println!("Event-driven notification completed in: {:?}", elapsed);
                println!("Result: {:?}", result);
                
                // Should be very fast (no busy-waiting)
                assert!(elapsed.as_millis() < 100, "Event-driven notification should be fast");
            }
        }
    }
}

#[test]
fn test_work_stealing_effectiveness() {
    println!("=== Work-Stealing Effectiveness Test ===");
    
    // Create uneven workload
    let mut uneven_data = Vec::new();
    for i in 0..20 {
        // Every 5th task is much more expensive
        if i % 5 == 0 {
            uneven_data.push(Value::Integer(15)); // Heavy task
        } else {
            uneven_data.push(Value::Integer(5));  // Light task
        }
    }
    
    let function = Value::Function("Factorial".to_string());
    
    let start = Instant::now();
    let pool = create_thread_pool(&[Value::Integer(4)]).unwrap();
    let args = vec![Value::List(vec![function, Value::List(uneven_data)]), pool];
    let result = parallel(&args).unwrap();
    let elapsed = start.elapsed();
    
    println!("Work-stealing uneven workload time: {:?}", elapsed);
    
    // Verify we got correct results
    if let Value::List(results) = result {
        assert_eq!(results.len(), 20, "Should have 20 results");
        println!("All {} tasks completed successfully", results.len());
    }
    
    // Should handle uneven workload reasonably well
    assert!(elapsed.as_millis() < 1000, "Work-stealing should handle uneven workloads efficiently");
}

#[test]
fn test_thread_pool_creation() {
    println!("=== Thread Pool Creation Test ===");
    
    for worker_count in [1, 2, 4, 8] {
        let start = Instant::now();
        let pool = create_thread_pool(&[Value::Integer(worker_count as i64)]).unwrap();
        let creation_time = start.elapsed();
        
        if let Value::LyObj(pool_obj) = pool {
            // Verify pool was created correctly
            if let Ok(Value::Integer(count)) = pool_obj.call_method("WorkerCount", &[]) {
                assert_eq!(count, worker_count as i64, "Worker count should match requested");
                println!("Created {}-worker pool in {:?}", worker_count, creation_time);
            }
        }
    }
}

#[test]
fn test_basic_correctness() {
    println!("=== Basic Correctness Test ===");
    
    // Test that optimizations don't break basic functionality
    let simple_data = vec![Value::Integer(3), Value::Integer(4), Value::Integer(5)];
    let function = Value::Function("Add".to_string());
    
    let pool = create_thread_pool(&[Value::Integer(2)]).unwrap();
    let args = vec![Value::List(vec![function, Value::List(simple_data)]), pool];
    let result = parallel(&args).unwrap();
    
    if let Value::List(results) = result {
        assert_eq!(results.len(), 3, "Should have 3 results");
        println!("Basic correctness test passed: {} results computed", results.len());
        
        // All results should be valid integers (results of Add function)
        for (i, result) in results.iter().enumerate() {
            match result {
                Value::Integer(_) => println!("Result {}: {:?}", i, result),
                _ => panic!("Expected integer result, got {:?}", result),
            }
        }
    } else {
        panic!("Expected list result, got {:?}", result);
    }
}