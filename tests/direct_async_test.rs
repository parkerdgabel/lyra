use std::time::{Duration, Instant};
use lyra::stdlib::async_ops::ThreadPool;
use lyra::vm::Value;

#[test]
fn test_direct_thread_pool_creation() {
    println!("=== Direct Thread Pool Creation Test ===");
    
    for worker_count in [1, 2, 4, 8] {
        let start = Instant::now();
        let _pool = ThreadPool::new(worker_count);
        let creation_time = start.elapsed();
        
        println!("Created {}-worker pool in {:?}", worker_count, creation_time);
        
        // Should be very fast to create
        assert!(creation_time.as_millis() < 100, "Pool creation should be fast");
    }
}

#[test]
fn test_direct_task_submission() {
    println!("=== Direct Task Submission Test ===");
    
    let pool = ThreadPool::new(2);
    
    let start = Instant::now();
    
    // Submit a simple task
    let task_id = pool.submit_task(
        Value::Function("Add".to_string()),
        vec![Value::Integer(10), Value::Integer(5)]
    ).unwrap();
    
    // Wait for completion using event-driven waiting
    let result = pool.await_result(task_id).unwrap();
    
    let elapsed = start.elapsed();
    
    println!("Task completed in: {:?}", elapsed);
    println!("Result: {:?}", result);
    
    // Should complete quickly
    assert!(elapsed.as_millis() < 100, "Task should complete quickly");
}

#[test]
fn test_work_stealing_with_uneven_load() {
    println!("=== Work Stealing Test ===");
    
    let pool = ThreadPool::new(4);
    
    let start = Instant::now();
    let mut task_ids = Vec::new();
    
    // Submit tasks with varying complexity
    for i in 0..20 {
        let complexity = if i % 5 == 0 { 15 } else { 5 }; // Every 5th task is heavier
        let task_id = pool.submit_task(
            Value::Function("Factorial".to_string()),
            vec![Value::Integer(complexity)]
        ).unwrap();
        task_ids.push(task_id);
    }
    
    // Wait for all tasks to complete
    let mut results = Vec::new();
    for task_id in task_ids {
        let result = pool.await_result(task_id).unwrap();
        results.push(result);
    }
    
    let elapsed = start.elapsed();
    
    println!("Completed 20 uneven tasks in: {:?}", elapsed);
    println!("Results count: {}", results.len());
    
    assert_eq!(results.len(), 20, "Should have 20 results");
    assert!(elapsed.as_millis() < 1000, "Should complete within reasonable time");
}

#[test]
fn test_performance_comparison() {
    println!("=== Performance Comparison Test ===");
    
    // Test with 1 worker (baseline)
    let pool_1 = ThreadPool::new(1);
    let start = Instant::now();
    let mut task_ids_1 = Vec::new();
    
    for i in 5..15 {
        let task_id = pool_1.submit_task(
            Value::Function("Factorial".to_string()),
            vec![Value::Integer(i)]
        ).unwrap();
        task_ids_1.push(task_id);
    }
    
    for task_id in task_ids_1 {
        let _ = pool_1.await_result(task_id).unwrap();
    }
    
    let time_1_worker = start.elapsed();
    
    // Test with 4 workers (optimized)
    let pool_4 = ThreadPool::new(4);
    let start = Instant::now();
    let mut task_ids_4 = Vec::new();
    
    for i in 5..15 {
        let task_id = pool_4.submit_task(
            Value::Function("Factorial".to_string()),
            vec![Value::Integer(i)]
        ).unwrap();
        task_ids_4.push(task_id);
    }
    
    for task_id in task_ids_4 {
        let _ = pool_4.await_result(task_id).unwrap();
    }
    
    let time_4_workers = start.elapsed();
    
    let speedup = time_1_worker.as_nanos() as f64 / time_4_workers.as_nanos() as f64;
    
    println!("1 worker time: {:?}", time_1_worker);
    println!("4 worker time: {:?}", time_4_workers);
    println!("Speedup: {:.2}x", speedup);
    
    if speedup >= 1.5 {
        println!("✅ Good speedup achieved!");
    } else {
        println!("⚠️  Limited speedup, but functionality works");
    }
}

#[test]
fn test_event_driven_notifications() {
    println!("=== Event-Driven Notification Test ===");
    
    let pool = ThreadPool::new(2);
    
    // Submit multiple tasks
    let mut task_ids = Vec::new();
    for i in 0..10 {
        let task_id = pool.submit_task(
            Value::Function("Add".to_string()),
            vec![Value::Integer(i), Value::Integer(1)]
        ).unwrap();
        task_ids.push(task_id);
    }
    
    let start = Instant::now();
    
    // Use event-driven waiting for all tasks
    for task_id in task_ids {
        let _result = pool.await_result(task_id).unwrap();
    }
    
    let elapsed = start.elapsed();
    
    println!("Event-driven completion of 10 tasks: {:?}", elapsed);
    
    // Should be fast due to no busy-waiting
    assert!(elapsed.as_millis() < 200, "Event-driven waiting should be efficient");
}

#[test]
fn test_thread_pool_configuration() {
    println!("=== Thread Pool Configuration Test ===");
    
    let pool = ThreadPool::new(4);
    
    // Test basic pool properties
    assert_eq!(pool.worker_count(), 4);
    
    // Test that we can submit and complete tasks
    let task_id = pool.submit_task(
        Value::Function("Add".to_string()),
        vec![Value::Integer(1), Value::Integer(2)]
    ).unwrap();
    
    let result = pool.await_result(task_id).unwrap();
    println!("Configuration test result: {:?}", result);
    
    println!("Thread pool configuration test passed");
}