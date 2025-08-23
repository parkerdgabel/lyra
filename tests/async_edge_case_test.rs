//! Edge Case and Error Resilience Test for Async Operations

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};
    use lyra::vm::Value;
    use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};

    #[test]
    fn test_edge_case_validation() {
        println!("🔥 Edge Case Validation Test");
        
        // Test 1: Empty operations
        println!("  📊 Testing Empty Operations");
        
        // ThreadPool with 0 workers (should handle gracefully)
        let pool_result = thread_pool(&[Value::Integer(0)]);
        match pool_result {
            Ok(_) => println!("    ✓ ThreadPool(0) handled gracefully"),
            Err(e) => println!("    ✓ ThreadPool(0) properly rejected: {:?}", e),
        }
        
        // Very large worker count
        let large_pool_result = thread_pool(&[Value::Integer(1000)]);
        match large_pool_result {
            Ok(_) => println!("    ✓ ThreadPool(1000) accepted"),
            Err(e) => println!("    ⚠ ThreadPool(1000) rejected: {:?}", e),
        }
        
        // Test 2: Boundary channel operations
        println!("  📊 Testing Boundary Channel Operations");
        
        let ch = channel(&[]).unwrap();
        
        // Rapid send/receive cycles
        let rapid_cycles = 1000;
        let start = Instant::now();
        for i in 0..rapid_cycles {
            let msg = Value::Integer(i);
            send(&[ch.clone(), msg]).unwrap();
        }
        
        for _ in 0..rapid_cycles {
            receive(&[ch.clone()]).unwrap();
        }
        let rapid_time = start.elapsed();
        
        let rapid_throughput = (rapid_cycles * 2) as f64 / rapid_time.as_secs_f64();
        println!("    ✓ Rapid cycles: {:.2} ops/sec", rapid_throughput);
        assert!(rapid_throughput > 1000.0, "Rapid throughput too low");
        
        // Test 3: Promise edge cases  
        println!("  📊 Testing Promise Edge Cases");
        
        // Different value types
        let edge_values = vec![
            Value::Integer(i64::MAX),
            Value::Integer(i64::MIN),
            Value::String("".to_string()), // Empty string
            Value::String("very long string that might cause issues with memory management or copying".repeat(100)),
            Value::List(vec![]), // Empty list
            Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]), // Normal list
            Value::List(vec![Value::List(vec![Value::Integer(1)])]), // Nested list
        ];
        
        for (i, value) in edge_values.iter().enumerate() {
            let start = Instant::now();
            let future = promise(&[value.clone()]).unwrap();
            let result = await_future(&[future]).unwrap();
            let edge_time = start.elapsed();
            
            assert_eq!(result, *value, "Promise result mismatch for value {}", i);
            println!("    ✓ Edge value {}: {:?}", i, edge_time);
        }
        
        println!("  ✅ All edge cases handled correctly");
    }
    
    #[test]
    fn test_error_resilience() {
        println!("🔥 Error Resilience Test");
        
        // Test 1: Invalid argument types
        println!("  📊 Testing Invalid Argument Types");
        
        // ThreadPool with wrong type
        let bad_pool = thread_pool(&[Value::String("not_a_number".to_string())]);
        assert!(bad_pool.is_err(), "ThreadPool should reject string argument");
        println!("    ✓ ThreadPool properly rejects string argument");
        
        // Send with wrong argument count
        let send_wrong_args = send(&[Value::Integer(42)]);
        assert!(send_wrong_args.is_err(), "Send should require 2 arguments");
        println!("    ✓ Send properly rejects wrong argument count");
        
        // Receive with non-channel
        let receive_wrong_type = receive(&[Value::Integer(42)]);
        assert!(receive_wrong_type.is_err(), "Receive should require channel");
        println!("    ✓ Receive properly rejects non-channel argument");
        
        // Test 2: Bounded channel overflow handling
        println!("  📊 Testing Bounded Channel Overflow");
        
        let small_capacity = 5;
        let bounded_ch = bounded_channel(&[Value::Integer(small_capacity)]).unwrap();
        
        // Fill the channel to capacity
        let mut successful_sends = 0;
        for i in 0..small_capacity + 10 { // Try to overfill
            let message = Value::Integer(i);
            match send(&[bounded_ch.clone(), message]) {
                Ok(_) => successful_sends += 1,
                Err(_) => break, // Expected when full
            }
        }
        
        println!("    ✓ Successful sends before overflow: {}", successful_sends);
        assert!(successful_sends <= small_capacity + 5, "Too many sends accepted");
        
        // Drain the channel
        let mut received_count = 0;
        while let Ok(_) = receive(&[bounded_ch.clone()]) {
            received_count += 1;
            if received_count >= successful_sends {
                break;
            }
        }
        
        println!("    ✓ Successfully drained {} messages", received_count);
        
        // Test 3: Stress resilience
        println!("  📊 Testing Stress Resilience");
        
        let stress_iterations = 5000;
        let mut error_count = 0;
        
        let start = Instant::now();
        for i in 0..stress_iterations {
            // Create and use async objects rapidly
            match channel(&[]) {
                Ok(ch) => {
                    let message = Value::Integer(i);
                    if send(&[ch.clone(), message]).is_ok() {
                        let _ = receive(&[ch]);
                    } else {
                        error_count += 1;
                    }
                },
                Err(_) => error_count += 1,
            }
            
            // Promise stress
            let value = Value::Integer(i % 100);
            match promise(&[value.clone()]) {
                Ok(future) => {
                    if await_future(&[future]).is_err() {
                        error_count += 1;
                    }
                },
                Err(_) => error_count += 1,
            }
        }
        
        let stress_time = start.elapsed();
        let error_rate = error_count as f64 / (stress_iterations * 2) as f64;
        let stress_throughput = (stress_iterations * 4) as f64 / stress_time.as_secs_f64();
        
        println!("    ✓ Stress test: {:.2} ops/sec, {:.2}% error rate", 
                stress_throughput, error_rate * 100.0);
        
        assert!(error_rate < 0.01, "Error rate too high: {:.2}%", error_rate * 100.0);
        assert!(stress_throughput > 1000.0, "Stress throughput too low: {:.2}", stress_throughput);
        
        println!("  ✅ All error resilience tests passed");
    }
    
    #[test]
    fn test_concurrent_edge_cases() {
        println!("🔥 Concurrent Edge Cases Test");
        
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::thread;
        
        // Test concurrent channel access
        let ch = Arc::new(channel(&[]).unwrap());
        let success_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));
        
        let thread_count = 4;
        let ops_per_thread = 100;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let ch_ref = Arc::clone(&ch);
            let success_ref = Arc::clone(&success_count);
            let error_ref = Arc::clone(&error_count);
            
            let handle = thread::spawn(move || {
                for i in 0..ops_per_thread {
                    let message = Value::Integer(thread_id * 1000 + i);
                    
                    // Send-receive cycle
                    match send(&[ch_ref.as_ref().clone(), message.clone()]) {
                        Ok(_) => {
                            match receive(&[ch_ref.as_ref().clone()]) {
                                Ok(received) => {
                                    if received == message {
                                        success_ref.fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        error_ref.fetch_add(1, Ordering::Relaxed);
                                    }
                                },
                                Err(_) => { error_ref.fetch_add(1, Ordering::Relaxed); },
                            }
                        },
                        Err(_) => { error_ref.fetch_add(1, Ordering::Relaxed); },
                    }
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }
        
        let total_successes = success_count.load(Ordering::Relaxed);
        let total_errors = error_count.load(Ordering::Relaxed);
        let expected_ops = thread_count * ops_per_thread;
        
        let concurrent_success_rate = total_successes as f64 / expected_ops as f64 * 100.0;
        
        println!("  ✓ Concurrent operations: {}/{} successful ({:.1}%)",
                total_successes, expected_ops, concurrent_success_rate);
        
        assert!(concurrent_success_rate > 95.0, 
               "Concurrent success rate too low: {:.1}%", concurrent_success_rate);
        
        println!("  ✅ Concurrent edge cases handled correctly");
    }
}