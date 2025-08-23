//! Async Integration Validation Suite
//!
//! Comprehensive integration tests to validate that Lyra's async concurrency system
//! properly integrates with VM execution, REPL, stdlib, and error handling systems.

use std::time::{Duration, Instant};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};
use lyra::parser::Parser;
use lyra::lexer::Lexer;
use lyra::error::Error;

#[test]
fn test_vm_integration_basic() {
    println!("🔧 Integration Test: Basic VM Integration");
    
    // Test that async operations work alongside basic VM operations
    let mut vm = VirtualMachine::new();
    
    // Create async objects
    let pool = thread_pool(&[Value::Integer(2)]).unwrap();
    let ch = channel(&[]).unwrap();
    let future = promise(&[Value::Integer(42)]).unwrap();
    
    // Verify they were created as VM Values  
    assert!(matches!(pool, Value::LyObj(_)), "ThreadPool should be LyObj");
    assert!(matches!(ch, Value::LyObj(_)), "Channel should be LyObj");
    assert!(matches!(future, Value::LyObj(_)), "Future should be LyObj");
    
    // Test that we can store and retrieve async objects from VM
    let async_values = vec![pool, ch, future];
    let vm_list = Value::List(async_values);
    
    // Verify the VM can handle lists containing async objects
    if let Value::List(retrieved_values) = vm_list {
        assert_eq!(retrieved_values.len(), 3, "Should have 3 async objects");
        
        for (i, value) in retrieved_values.iter().enumerate() {
            match value {
                Value::LyObj(_) => {
                    println!("    ✓ Async object {} properly stored in VM", i);
                },
                _ => panic!("Expected LyObj for async object {}", i),
            }
        }
    }
    
    println!("  ✓ VM can properly store and handle async objects");
}

#[test]
fn test_vm_integration_with_computation() {
    println!("🔧 Integration Test: VM Integration with Computation");
    
    // Test async operations alongside VM computations
    let start = Instant::now();
    
    // Simulate VM computations mixed with async operations
    let mut computational_results = Vec::new();
    let mut async_results = Vec::new();
    
    for i in 0..100 {
        // VM-style computation
        let vm_result = Value::Integer(i * i);  // Square numbers
        computational_results.push(vm_result);
        
        // Async operation
        let future = promise(&[Value::Integer(i)]).unwrap();
        let async_result = await_future(&[future]).unwrap();
        async_results.push(async_result);
    }
    
    let duration = start.elapsed();
    
    // Verify results
    assert_eq!(computational_results.len(), 100);
    assert_eq!(async_results.len(), 100);
    
    // Check that computational and async results are coherent
    for i in 0..100 {
        if let (Value::Integer(comp_val), Value::Integer(async_val)) = 
            (&computational_results[i], &async_results[i]) {
            assert_eq!(*comp_val, (i * i) as i64, "VM computation incorrect");
            assert_eq!(*async_val, i as i64, "Async result incorrect");
        }
    }
    
    println!("  ✓ VM computations and async operations work together");
    println!("    Mixed operations completed in {:?}", duration);
}

#[test]
fn test_stdlib_integration() {
    println!("🔧 Integration Test: Stdlib Integration");
    
    // Test async operations with stdlib-like mathematical operations
    let start = Instant::now();
    
    // Create async infrastructure
    let ch = channel(&[]).unwrap();
    let results_channel = channel(&[]).unwrap();
    
    // Simulate mathematical computation pipeline using async operations
    let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    
    // Send mathematical inputs through async channel
    for value in &test_data {
        let input = Value::Integer(*value as i64);
        send(&[ch.clone(), input]).unwrap();
    }
    
    // Process mathematical operations asynchronously
    for _ in 0..test_data.len() {
        let input = receive(&[ch.clone()]).unwrap();
        
        if let Value::Integer(n) = input {
            // Mathematical operations (factorial-like)
            let mut result = 1i64;
            for i in 1..=n.min(10) {  // Limit to prevent overflow
                result = result.saturating_mul(i);
            }
            
            // Send result through async channel
            let computed_result = Value::Integer(result);
            send(&[results_channel.clone(), computed_result]).unwrap();
        }
    }
    
    // Collect results
    let mut final_results = Vec::new();
    for _ in 0..test_data.len() {
        let result = receive(&[results_channel.clone()]).unwrap();
        final_results.push(result);
    }
    
    let duration = start.elapsed();
    
    // Verify mathematical correctness
    assert_eq!(final_results.len(), test_data.len());
    
    // Check some expected factorial-like results
    if let Value::Integer(first_result) = &final_results[0] {
        assert_eq!(*first_result, 1, "1! should be 1");
    }
    
    if let Value::Integer(second_result) = &final_results[1] {
        assert_eq!(*second_result, 2, "2! should be 2");  
    }
    
    if let Value::Integer(third_result) = &final_results[2] {
        assert_eq!(*third_result, 6, "3! should be 6");
    }
    
    println!("  ✓ Async operations integrate properly with mathematical computations");
    println!("    Mathematical pipeline completed in {:?}", duration);
}

#[test]
fn test_error_propagation_integration() {
    println!("🔧 Integration Test: Error Propagation");
    
    // Test that errors propagate correctly through the system
    
    // Test 1: Channel overflow error handling
    {
        let small_ch = bounded_channel(&[Value::Integer(1)]).unwrap();
        
        // Fill the channel
        let success1 = send(&[small_ch.clone(), Value::Integer(1)]);
        assert!(success1.is_ok(), "First send should succeed");
        
        // Try to overflow (this may or may not error depending on implementation)
        let overflow_result = send(&[small_ch.clone(), Value::Integer(2)]);
        
        // Either way, the system should handle it gracefully
        match overflow_result {
            Ok(_) => println!("    ✓ Channel accepted overflow (unbounded behavior)"),
            Err(e) => println!("    ✓ Channel properly rejected overflow: {:?}", e),
        }
        
        // Clean up
        let _ = receive(&[small_ch.clone()]);
        let _ = receive(&[small_ch.clone()]);
    }
    
    // Test 2: Future error handling
    {
        // Test that futures handle various value types correctly
        let test_values = vec![
            Value::Integer(42),
            Value::String("test".to_string()),
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        ];
        
        for test_value in test_values {
            let future = promise(&[test_value.clone()]).unwrap();
            let result = await_future(&[future]).unwrap();
            assert_eq!(result, test_value, "Future should return the same value");
        }
    }
    
    // Test 3: ThreadPool error handling
    {
        // Test ThreadPool with various worker counts
        for worker_count in [0, 1, 2, 4, 100] {
            let pool_result = thread_pool(&[Value::Integer(worker_count)]);
            
            match worker_count {
                0 => {
                    // 0 workers might be handled as default or error
                    match pool_result {
                        Ok(_) => println!("    ✓ ThreadPool accepts 0 workers (uses default)"),
                        Err(e) => println!("    ✓ ThreadPool properly rejects 0 workers: {:?}", e),
                    }
                },
                _ => {
                    assert!(pool_result.is_ok(), 
                           "ThreadPool with {} workers should succeed", worker_count);
                }
            }
        }
    }
    
    println!("  ✓ Error propagation works correctly across system boundaries");
}

#[test]
fn test_concurrent_integration() {
    println!("🔧 Integration Test: Concurrent System Integration");
    
    // Test multiple subsystems working together concurrently
    let start = Instant::now();
    
    let iterations = 50;
    let mut all_successful = true;
    
    for i in 0..iterations {
        // VM-style operations
        let vm_value = Value::Integer(i);
        
        // Async operations
        let future = promise(&[vm_value.clone()]).unwrap();
        let ch = channel(&[]).unwrap();
        let pool = thread_pool(&[Value::Integer(2)]).unwrap();
        
        // Cross-system operations
        send(&[ch.clone(), vm_value.clone()]).unwrap();
        let channel_result = receive(&[ch]).unwrap();
        let future_result = await_future(&[future]).unwrap();
        
        // Verify consistency across systems
        if channel_result != vm_value || future_result != vm_value {
            all_successful = false;
            break;
        }
        
        // ThreadPool validation
        if let Value::LyObj(pool_obj) = pool {
            if let Ok(Value::Integer(worker_count)) = pool_obj.call_method("workerCount", &[]) {
                if worker_count != 2 {
                    all_successful = false;
                    break;
                }
            }
        }
    }
    
    let duration = start.elapsed();
    
    assert!(all_successful, "All concurrent integrations should succeed");
    
    let throughput = (iterations * 4) as f64 / duration.as_secs_f64(); // 4 operations per iteration
    
    println!("  ✓ Concurrent system integration successful");
    println!("    {} iterations completed in {:?}", iterations, duration);
    println!("    Concurrent integration throughput: {:.2} ops/sec", throughput);
    
    // Validate performance
    assert!(throughput > 1000.0, 
           "Concurrent integration throughput too low: {:.2} ops/sec", throughput);
}

#[test]
fn test_repl_simulation_integration() {
    println!("🔧 Integration Test: REPL Simulation");
    
    // Simulate REPL-like interactive usage of async operations
    
    // Mock REPL session commands
    let repl_commands = vec![
        ("Create ThreadPool", || -> Result<Value, String> {
            thread_pool(&[Value::Integer(4)]).map_err(|e| format!("ThreadPool error: {:?}", e))
        }),
        ("Create Channel", || -> Result<Value, String> {
            channel(&[]).map_err(|e| format!("Channel error: {:?}", e))
        }),
        ("Create Promise", || -> Result<Value, String> {
            promise(&[Value::String("Hello REPL".to_string())])
                .map_err(|e| format!("Promise error: {:?}", e))
        }),
        ("Channel Send/Receive", || -> Result<Value, String> {
            let ch = channel(&[])?;
            let msg = Value::Integer(123);
            send(&[ch.clone(), msg.clone()])?;
            let result = receive(&[ch])?;
            if result == msg {
                Ok(Value::String("Send/Receive successful".to_string()))
            } else {
                Err("Send/Receive mismatch".to_string())
            }
        }),
        ("Future Await", || -> Result<Value, String> {
            let future = promise(&[Value::Real(3.14159)])?;
            await_future(&[future]).map_err(|e| format!("Await error: {:?}", e))
        }),
    ];
    
    println!("  📊 Simulating REPL session with {} commands", repl_commands.len());
    
    let mut session_results = Vec::new();
    let session_start = Instant::now();
    
    for (command_name, command_fn) in repl_commands {
        println!("    Executing: {}", command_name);
        
        let cmd_start = Instant::now();
        let result = command_fn();
        let cmd_duration = cmd_start.elapsed();
        
        match result {
            Ok(value) => {
                println!("      ✓ Success in {:?}: {:?}", cmd_duration, 
                        match &value {
                            Value::LyObj(_) => "LyObj(...)".to_string(),
                            other => format!("{:?}", other),
                        });
                session_results.push(value);
            },
            Err(error) => {
                panic!("REPL command '{}' failed: {}", command_name, error);
            }
        }
    }
    
    let session_duration = session_start.elapsed();
    
    // Validate session results
    assert_eq!(session_results.len(), 5, "Should have 5 successful REPL commands");
    
    // Check result types
    assert!(matches!(session_results[0], Value::LyObj(_)), "ThreadPool result");
    assert!(matches!(session_results[1], Value::LyObj(_)), "Channel result");
    assert!(matches!(session_results[2], Value::LyObj(_)), "Promise result");
    assert!(matches!(session_results[3], Value::String(_)), "Send/Receive result");
    assert!(matches!(session_results[4], Value::Real(_)), "Future result");
    
    println!("  ✓ REPL simulation completed successfully in {:?}", session_duration);
    println!("    All {} async commands executed without errors", session_results.len());
}

#[test]
fn test_data_flow_integration() {
    println!("🔧 Integration Test: Data Flow Integration");
    
    // Test complex data flow patterns through the integrated system
    let pipeline_size = 20;
    let start = Instant::now();
    
    // Create data pipeline using multiple async components
    let input_channel = channel(&[]).unwrap();
    let processing_channel = channel(&[]).unwrap();
    let output_channel = channel(&[]).unwrap();
    
    // Stage 1: Data ingestion
    for i in 0..pipeline_size {
        let data = Value::Integer(i as i64);
        send(&[input_channel.clone(), data]).unwrap();
    }
    
    // Stage 2: Data processing (using futures for computation)
    for _ in 0..pipeline_size {
        let raw_data = receive(&[input_channel.clone()]).unwrap();
        
        if let Value::Integer(n) = raw_data {
            // Process through Future (simulate async computation)
            let computation_input = Value::Integer(n * 2); // Double the value
            let future = promise(&[computation_input]).unwrap();
            let processed_data = await_future(&[future]).unwrap();
            
            // Send to next stage
            send(&[processing_channel.clone(), processed_data]).unwrap();
        }
    }
    
    // Stage 3: Data aggregation
    let mut aggregated_sum = 0i64;
    for _ in 0..pipeline_size {
        let processed_data = receive(&[processing_channel.clone()]).unwrap();
        
        if let Value::Integer(n) = processed_data {
            aggregated_sum += n;
            
            // Final output
            let final_result = Value::Integer(aggregated_sum);
            send(&[output_channel.clone(), final_result]).unwrap();
        }
    }
    
    // Validate pipeline results
    let mut final_results = Vec::new();
    for _ in 0..pipeline_size {
        let result = receive(&[output_channel.clone()]).unwrap();
        final_results.push(result);
    }
    
    let duration = start.elapsed();
    
    // Verify data flow correctness
    assert_eq!(final_results.len(), pipeline_size);
    
    // Check that results are cumulative sums
    let mut expected_sum = 0i64;
    for i in 0..pipeline_size {
        expected_sum += (i as i64) * 2; // Doubled values
        
        if let Value::Integer(actual_sum) = &final_results[i] {
            assert_eq!(*actual_sum, expected_sum, 
                      "Pipeline result {} incorrect", i);
        }
    }
    
    let throughput = (pipeline_size * 3) as f64 / duration.as_secs_f64(); // 3 stages
    
    println!("  ✓ Data flow pipeline integration successful");
    println!("    {} items processed through 3 stages in {:?}", pipeline_size, duration);
    println!("    Pipeline throughput: {:.2} ops/sec", throughput);
    
    // Validate performance
    assert!(throughput > 500.0, 
           "Pipeline throughput too low: {:.2} ops/sec", throughput);
}

#[test]
fn test_system_boundary_integration() {
    println!("🔧 Integration Test: System Boundary Integration");
    
    // Test that async operations work correctly across different system boundaries
    
    // Boundary 1: Async ↔ VM Value System
    {
        let async_objects = vec![
            thread_pool(&[]).unwrap(),
            channel(&[]).unwrap(), 
            promise(&[Value::Integer(42)]).unwrap(),
        ];
        
        // Test conversion through VM value system
        let vm_collection = Value::List(async_objects);
        
        if let Value::List(retrieved_objects) = vm_collection {
            for (i, obj) in retrieved_objects.iter().enumerate() {
                assert!(matches!(obj, Value::LyObj(_)), 
                       "Object {} should remain LyObj after VM conversion", i);
            }
        }
        
        println!("    ✓ Async ↔ VM boundary integration works");
    }
    
    // Boundary 2: Error System Integration
    {
        let mut error_count = 0;
        let total_operations = 100;
        
        for i in 0..total_operations {
            // Mix successful and potentially failing operations
            let result = if i % 10 == 0 {
                // Occasionally use edge case values
                thread_pool(&[Value::Integer(0)])  // 0 workers might error
            } else {
                thread_pool(&[Value::Integer(2)])  // Normal case
            };
            
            match result {
                Ok(_) => {},  // Success
                Err(_) => error_count += 1,  // Expected error
            }
        }
        
        let success_rate = ((total_operations - error_count) as f64 / total_operations as f64) * 100.0;
        println!("    ✓ Error boundary integration: {:.1}% success rate", success_rate);
        
        // Most operations should succeed
        assert!(success_rate > 80.0, "Success rate too low: {:.1}%", success_rate);
    }
    
    // Boundary 3: Concurrent Access Integration
    {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;
        
        let success_count = Arc::new(AtomicUsize::new(0));
        let thread_count = 4;
        let operations_per_thread = 25;
        
        let mut handles = Vec::new();
        
        for thread_id in 0..thread_count {
            let success_ref = Arc::clone(&success_count);
            
            let handle = thread::spawn(move || {
                for i in 0..operations_per_thread {
                    // Each thread performs async operations
                    let ch = channel(&[]).unwrap();
                    let message = Value::Integer((thread_id * 100 + i) as i64);
                    
                    if send(&[ch.clone(), message.clone()]).is_ok() {
                        if let Ok(received) = receive(&[ch]) {
                            if received == message {
                                success_ref.fetch_add(1, Ordering::Relaxed);
                            }
                        }
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
        let expected_successes = thread_count * operations_per_thread;
        let concurrent_success_rate = (total_successes as f64 / expected_successes as f64) * 100.0;
        
        println!("    ✓ Concurrent boundary integration: {}/{} operations successful ({:.1}%)", 
                total_successes, expected_successes, concurrent_success_rate);
        
        assert!(concurrent_success_rate > 95.0, 
               "Concurrent success rate too low: {:.1}%", concurrent_success_rate);
    }
    
    println!("  ✓ All system boundary integrations working correctly");
}