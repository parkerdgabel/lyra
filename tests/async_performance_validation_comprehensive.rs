//! Comprehensive Async Performance Validation Tests
//!
//! This test suite validates that our async performance benchmarking infrastructure
//! works correctly and provides meaningful performance measurements.

use std::time::{Duration, Instant};
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};

#[test]
fn test_async_functionality_baseline_performance() {
    println!("🚀 Running Async Functionality Baseline Performance Tests");
    
    // Test basic async operations performance
    test_threadpool_baseline_performance();
    test_channel_baseline_performance();
    test_future_baseline_performance();
    test_concurrent_patterns_baseline();
}

fn test_threadpool_baseline_performance() {
    println!("📊 Testing ThreadPool Baseline Performance");
    
    let iterations = 100;
    let mut creation_times = Vec::with_capacity(iterations);
    
    // Measure ThreadPool creation time
    for _ in 0..iterations {
        let start = Instant::now();
        let pool = thread_pool(&[]).unwrap();
        let duration = start.elapsed();
        creation_times.push(duration);
        
        // Verify the pool was created successfully
        assert!(matches!(pool, Value::LyObj(_)));
    }
    
    let avg_creation_time = creation_times.iter().sum::<Duration>() / iterations as u32;
    println!("  ✓ ThreadPool creation average time: {:?}", avg_creation_time);
    
    // ThreadPool creation should be fast (< 1ms typically)
    assert!(avg_creation_time < Duration::from_millis(10), 
           "ThreadPool creation too slow: {:?}", avg_creation_time);
    
    // Test with different worker counts
    for worker_count in [1, 2, 4, 8] {
        let start = Instant::now();
        let pool = thread_pool(&[Value::Integer(worker_count)]).unwrap();
        let duration = start.elapsed();
        
        println!("  ✓ ThreadPool creation ({} workers): {:?}", worker_count, duration);
        assert!(matches!(pool, Value::LyObj(_)));
        assert!(duration < Duration::from_millis(50), 
               "ThreadPool creation with {} workers too slow: {:?}", worker_count, duration);
        
        // Test worker count query
        if let Value::LyObj(pool_obj) = &pool {
            let start = Instant::now();
            let result = pool_obj.call_method("workerCount", &[]);
            let query_duration = start.elapsed();
            
            println!("    ✓ Worker count query time: {:?}", query_duration);
            assert!(query_duration < Duration::from_millis(1), 
                   "Worker count query too slow: {:?}", query_duration);
            
            if let Ok(Value::Integer(count)) = result {
                assert_eq!(count, worker_count, "Worker count mismatch");
            }
        }
    }
}

fn test_channel_baseline_performance() {
    println!("📊 Testing Channel Baseline Performance");
    
    // Unbounded channel creation performance
    let iterations = 1000;
    let mut creation_times = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let start = Instant::now();
        let ch = channel(&[]).unwrap();
        let duration = start.elapsed();
        creation_times.push(duration);
        
        assert!(matches!(ch, Value::LyObj(_)));
    }
    
    let avg_creation_time = creation_times.iter().sum::<Duration>() / iterations as u32;
    println!("  ✓ Unbounded channel creation average time: {:?}", avg_creation_time);
    assert!(avg_creation_time < Duration::from_millis(1), 
           "Channel creation too slow: {:?}", avg_creation_time);
    
    // Bounded channel creation performance
    for capacity in [10, 100, 1000] {
        let start = Instant::now();
        let ch = bounded_channel(&[Value::Integer(capacity)]).unwrap();
        let duration = start.elapsed();
        
        println!("  ✓ Bounded channel creation (capacity {}): {:?}", capacity, duration);
        assert!(matches!(ch, Value::LyObj(_)));
        assert!(duration < Duration::from_millis(5), 
               "Bounded channel creation too slow: {:?}", duration);
    }
    
    // Send/Receive performance
    let ch = channel(&[]).unwrap();
    let mut send_receive_times = Vec::with_capacity(1000);
    
    for i in 0..1000 {
        let message = Value::Integer(i);
        
        let start = Instant::now();
        send(&[ch.clone(), message]).unwrap();
        let received = receive(&[ch.clone()]).unwrap();
        let duration = start.elapsed();
        
        send_receive_times.push(duration);
        assert_eq!(received, Value::Integer(i));
    }
    
    let avg_send_receive_time = send_receive_times.iter().sum::<Duration>() / 1000;
    println!("  ✓ Send/Receive average time: {:?}", avg_send_receive_time);
    assert!(avg_send_receive_time < Duration::from_millis(1), 
           "Send/Receive too slow: {:?}", avg_send_receive_time);
    
    // Batch throughput test
    let start = Instant::now();
    for i in 0..10000 {
        let message = Value::Integer(i);
        send(&[ch.clone(), message]).unwrap();
    }
    
    for _ in 0..10000 {
        receive(&[ch.clone()]).unwrap();
    }
    let batch_duration = start.elapsed();
    
    let throughput = 10000.0 / batch_duration.as_secs_f64();
    println!("  ✓ Channel throughput: {:.2} ops/sec", throughput);
    assert!(throughput > 10000.0, "Channel throughput too low: {:.2} ops/sec", throughput);
}

fn test_future_baseline_performance() {
    println!("📊 Testing Future Baseline Performance");
    
    // Promise creation and await performance
    let iterations = 1000;
    let mut promise_await_times = Vec::with_capacity(iterations);
    
    for i in 0..iterations {
        let value = Value::Integer(i);
        
        let start = Instant::now();
        let future = promise(&[value.clone()]).unwrap();
        let result = await_future(&[future]).unwrap();
        let duration = start.elapsed();
        
        promise_await_times.push(duration);
        assert_eq!(result, value);
    }
    
    let avg_promise_await_time = promise_await_times.iter().sum::<Duration>() / iterations as u32;
    println!("  ✓ Promise create/await average time: {:?}", avg_promise_await_time);
    assert!(avg_promise_await_time < Duration::from_millis(1), 
           "Promise create/await too slow: {:?}", avg_promise_await_time);
    
    // Test with different value types
    let test_values = vec![
        ("integer", Value::Integer(42)),
        ("string", Value::String("test_string".to_string())),
        ("list", Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])),
    ];
    
    for (value_type, test_value) in test_values {
        let start = Instant::now();
        let future = promise(&[test_value.clone()]).unwrap();
        let result = await_future(&[future]).unwrap();
        let duration = start.elapsed();
        
        println!("  ✓ Promise create/await for {}: {:?}", value_type, duration);
        assert_eq!(result, test_value);
        assert!(duration < Duration::from_millis(5), 
               "Promise create/await for {} too slow: {:?}", value_type, duration);
    }
    
    // Batch future performance
    let batch_size = 100;
    let start = Instant::now();
    
    let mut futures = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let value = Value::Integer(i);
        let future = promise(&[value]).unwrap();
        futures.push(future);
    }
    
    let mut results = Vec::with_capacity(batch_size);
    for future in futures {
        let result = await_future(&[future]).unwrap();
        results.push(result);
    }
    
    let batch_duration = start.elapsed();
    let batch_throughput = batch_size as f64 / batch_duration.as_secs_f64();
    
    println!("  ✓ Future batch throughput ({}): {:.2} ops/sec", batch_size, batch_throughput);
    assert_eq!(results.len(), batch_size);
    assert!(batch_throughput > 1000.0, "Future batch throughput too low: {:.2} ops/sec", batch_throughput);
}

fn test_concurrent_patterns_baseline() {
    println!("📊 Testing Concurrent Patterns Baseline Performance");
    
    // Producer-Consumer pattern performance
    for queue_size in [10, 100, 1000] {
        let start = Instant::now();
        
        let ch = bounded_channel(&[Value::Integer(queue_size)]).unwrap();
        
        // Producer phase
        let producer_items = queue_size / 2;
        for i in 0..producer_items {
            let message = Value::Integer(i);
            send(&[ch.clone(), message]).unwrap();
        }
        
        // Consumer phase
        let mut consumed = Vec::new();
        for _ in 0..producer_items {
            let received = receive(&[ch.clone()]).unwrap();
            consumed.push(received);
        }
        
        let duration = start.elapsed();
        let throughput = (producer_items * 2) as f64 / duration.as_secs_f64(); // *2 for send+receive
        
        println!("  ✓ Producer-Consumer pattern (queue={}): {:?}, {:.2} ops/sec", 
                queue_size, duration, throughput);
        assert_eq!(consumed.len(), producer_items);
        assert!(throughput > 100.0, "Producer-Consumer throughput too low: {:.2} ops/sec", throughput);
    }
    
    // Multi-channel coordination
    let start = Instant::now();
    
    let ch1 = channel(&[]).unwrap();
    let ch2 = channel(&[]).unwrap();
    let ch3 = channel(&[]).unwrap();
    
    // Send to multiple channels
    for i in 0..100 {
        let message = Value::Integer(i);
        send(&[ch1.clone(), message.clone()]).unwrap();
        send(&[ch2.clone(), message.clone()]).unwrap();
        send(&[ch3.clone(), message]).unwrap();
    }
    
    // Receive from all channels
    let mut all_results = Vec::new();
    for _ in 0..100 {
        all_results.push(receive(&[ch1.clone()]).unwrap());
        all_results.push(receive(&[ch2.clone()]).unwrap());
        all_results.push(receive(&[ch3.clone()]).unwrap());
    }
    
    let multi_channel_duration = start.elapsed();
    let multi_channel_throughput = 600.0 / multi_channel_duration.as_secs_f64(); // 300 sends + 300 receives
    
    println!("  ✓ Multi-channel coordination: {:?}, {:.2} ops/sec", 
            multi_channel_duration, multi_channel_throughput);
    assert_eq!(all_results.len(), 300);
    assert!(multi_channel_throughput > 1000.0, 
           "Multi-channel throughput too low: {:.2} ops/sec", multi_channel_throughput);
}

#[test]
fn test_scalability_characteristics() {
    println!("🚀 Testing Scalability Characteristics");
    
    // Test ThreadPool scalability
    let baseline_work = || -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();
        
        // Simulate work with channels
        for _ in 0..100 {
            let ch = channel(&[])?;
            let message = Value::Integer(42);
            send(&[ch.clone(), message])?;
            receive(&[ch])?;
        }
        
        Ok(start.elapsed())
    };
    
    // Baseline with single thread
    let single_thread_time = {
        let _pool = thread_pool(&[Value::Integer(1)]).unwrap();
        baseline_work().unwrap()
    };
    
    println!("  ✓ Baseline time (1 worker): {:?}", single_thread_time);
    
    // Test with multiple workers
    for worker_count in [2, 4, 8] {
        let multi_thread_time = {
            let _pool = thread_pool(&[Value::Integer(worker_count)]).unwrap();
            baseline_work().unwrap()
        };
        
        let speedup = single_thread_time.as_secs_f64() / multi_thread_time.as_secs_f64();
        let overhead_ratio = multi_thread_time.as_secs_f64() / single_thread_time.as_secs_f64();
        
        println!("  ✓ Time with {} workers: {:?} (speedup: {:.2}x, overhead: {:.2}x)", 
                worker_count, multi_thread_time, speedup, overhead_ratio);
        
        // Multi-threading shouldn't add excessive overhead
        assert!(overhead_ratio < 3.0, 
               "Multi-threading overhead too high with {} workers: {:.2}x", 
               worker_count, overhead_ratio);
    }
}

#[test]
fn test_memory_efficiency_baseline() {
    println!("🚀 Testing Memory Efficiency Baseline");
    
    // Test memory efficiency with many objects
    let large_scale_test = || -> Result<(), Box<dyn std::error::Error>> {
        let mut channels = Vec::new();
        let mut futures = Vec::new();
        
        // Create many async objects
        for i in 0..1000 {
            // Create channels
            let ch = channel(&[])?;
            channels.push(ch);
            
            // Create futures
            let value = Value::Integer(i);
            let future = promise(&[value])?;
            futures.push(future);
        }
        
        // Use all channels
        for (i, ch) in channels.iter().enumerate() {
            let message = Value::Integer(i as i64);
            send(&[ch.clone(), message])?;
            receive(&[ch.clone()])?;
        }
        
        // Use all futures
        let mut results = Vec::new();
        for future in futures {
            let result = await_future(&[future])?;
            results.push(result);
        }
        
        assert_eq!(channels.len(), 1000);
        assert_eq!(results.len(), 1000);
        
        Ok(())
    };
    
    let start = Instant::now();
    large_scale_test().unwrap();
    let duration = start.elapsed();
    
    println!("  ✓ Large scale test (1000 channels + 1000 futures): {:?}", duration);
    
    // Should complete reasonably quickly
    assert!(duration < Duration::from_secs(5), 
           "Large scale test too slow: {:?}", duration);
}

#[test]
fn test_error_handling_performance() {
    println!("🚀 Testing Error Handling Performance");
    
    // Test bounded channel overflow error handling
    let iterations = 100;
    let mut error_handling_times = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let ch = bounded_channel(&[Value::Integer(1)]).unwrap();
        
        // Fill the channel
        let message1 = Value::Integer(1);
        send(&[ch.clone(), message1]).unwrap();
        
        // Try to overflow
        let message2 = Value::Integer(2);
        let start = Instant::now();
        let _result = send(&[ch.clone(), message2]); // May fail
        let duration = start.elapsed();
        
        error_handling_times.push(duration);
        
        // Clean up
        receive(&[ch.clone()]).unwrap();
    }
    
    let avg_error_handling_time = error_handling_times.iter().sum::<Duration>() / iterations as u32;
    println!("  ✓ Error handling average time: {:?}", avg_error_handling_time);
    
    // Error handling should be fast
    assert!(avg_error_handling_time < Duration::from_millis(5), 
           "Error handling too slow: {:?}", avg_error_handling_time);
}

#[test]
fn test_latency_distribution() {
    println!("🚀 Testing Latency Distribution Characteristics");
    
    // Collect latency samples for channel operations
    let iterations = 1000;
    let mut latencies = Vec::with_capacity(iterations);
    let ch = channel(&[]).unwrap();
    
    for i in 0..iterations {
        let message = Value::Integer(i);
        
        let start = Instant::now();
        send(&[ch.clone(), message]).unwrap();
        receive(&[ch.clone()]).unwrap();
        let duration = start.elapsed();
        
        latencies.push(duration);
    }
    
    // Sort for percentile calculation
    latencies.sort();
    
    let mean = latencies.iter().sum::<Duration>() / iterations as u32;
    let median = latencies[iterations / 2];
    let p95 = latencies[(iterations as f64 * 0.95) as usize];
    let p99 = latencies[(iterations as f64 * 0.99) as usize];
    
    println!("  ✓ Channel operation latency distribution:");
    println!("    Mean: {:?}", mean);
    println!("    Median: {:?}", median);
    println!("    P95: {:?}", p95);
    println!("    P99: {:?}", p99);
    
    // Latency characteristics
    assert!(mean < Duration::from_millis(1), "Mean latency too high: {:?}", mean);
    assert!(p95 < Duration::from_millis(5), "P95 latency too high: {:?}", p95);
    assert!(p99 < Duration::from_millis(10), "P99 latency too high: {:?}", p99);
    
    // Check latency consistency (P99 shouldn't be much higher than median)
    let consistency_ratio = p99.as_nanos() as f64 / median.as_nanos() as f64;
    println!("  ✓ Latency consistency ratio (P99/Median): {:.2}", consistency_ratio);
    assert!(consistency_ratio < 10.0, 
           "Latency too inconsistent: P99 is {:.2}x median", consistency_ratio);
}

#[test]
fn test_performance_regression_detection() {
    println!("🚀 Testing Performance Regression Detection");
    
    // Establish baseline performance metrics
    let baseline_metrics = establish_baseline_metrics();
    
    // Run the same tests again and compare
    let current_metrics = establish_baseline_metrics();
    
    // Compare performance (allowing for some variance)
    let variance_threshold = 0.2; // 20% variance allowed
    
    for (operation, baseline_time) in baseline_metrics {
        if let Some(current_time) = current_metrics.get(&operation) {
            let variance = (current_time.as_secs_f64() - baseline_time.as_secs_f64()).abs() 
                         / baseline_time.as_secs_f64();
            
            println!("  ✓ {} - Baseline: {:?}, Current: {:?}, Variance: {:.2}%", 
                    operation, baseline_time, current_time, variance * 100.0);
            
            assert!(variance < variance_threshold, 
                   "Performance regression detected in {}: {:.2}% variance", 
                   operation, variance * 100.0);
        }
    }
}

fn establish_baseline_metrics() -> std::collections::HashMap<String, Duration> {
    let mut metrics = std::collections::HashMap::new();
    
    // ThreadPool creation
    let start = Instant::now();
    let _pool = thread_pool(&[]).unwrap();
    metrics.insert("threadpool_creation".to_string(), start.elapsed());
    
    // Channel send/receive
    let start = Instant::now();
    let ch = channel(&[]).unwrap();
    let message = Value::Integer(42);
    send(&[ch.clone(), message]).unwrap();
    receive(&[ch]).unwrap();
    metrics.insert("channel_send_receive".to_string(), start.elapsed());
    
    // Future create/await
    let start = Instant::now();
    let value = Value::Integer(42);
    let future = promise(&[value]).unwrap();
    await_future(&[future]).unwrap();
    metrics.insert("future_create_await".to_string(), start.elapsed());
    
    metrics
}