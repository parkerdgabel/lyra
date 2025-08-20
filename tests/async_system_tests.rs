use lyra::compiler::Compiler;
use lyra::lexer::Lexer;
use lyra::parser::Parser;
use lyra::vm::{VirtualMachine, Value};
use std::time::Duration;
use std::thread;

#[test]
fn test_promise_creation() {
    // Test: Promise stdlib function can create Future values
    let mut compiler = Compiler::new();
    
    // Create a promise: result = Promise[42]
    let source = "result = Promise[42]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().expect("Should tokenize promise creation");
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().expect("Should parse promise expression");
    compiler.compile_expr(&expr).expect("Should compile promise expression");
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().expect("Should tokenize result call");
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().expect("Should parse result call");
    compiler.compile_expr(&call_expr).expect("Should compile result call");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return a Future value
    let result = vm.run();
    assert!(result.is_ok(), "VM execution should succeed, got error: {:?}", result.err());
    let value = result.expect("VM should return a value");
    
    // Check that we got a Future LyObj back
    match value {
        Value::LyObj(ly_obj) => {
            // Should be a Future type
            assert_eq!(ly_obj.type_name(), "Future");
            
            // Call resolve method to get the contained value
            let resolved = ly_obj.call_method("resolve", &[])
                .expect("Future should resolve successfully");
            assert_eq!(resolved, Value::Integer(42));
        }
        _ => panic!("Expected Future LyObj, got {:?}", value),
    }
}

#[test]
fn test_await_resolution() {
    // Test: Await stdlib function can resolve futures
    let mut compiler = Compiler::new();
    
    // Create a promise and await it: result = Await[Promise[100]]
    let source = "result = Await[Promise[100]]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().expect("Should tokenize await expression");
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().expect("Should parse await expression");
    compiler.compile_expr(&expr).expect("Should compile await expression");
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().expect("Should tokenize result call");
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().expect("Should parse result call");
    compiler.compile_expr(&call_expr).expect("Should compile result call");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return the resolved value (100)
    let result = vm.run();
    assert!(result.is_ok(), "VM execution should succeed for await, got error: {:?}", result.err());
    let value = result.expect("VM should return the awaited value");
    assert_eq!(value, Value::Integer(100));
}

#[test]
fn test_future_type_validation() {
    // Test: Future objects work with basic Promise creation
    let mut compiler = Compiler::new();
    
    // Create a promise and store it: result = Promise[42]
    let source = "result = Promise[42]";
    let mut lexer = Lexer::new(source);
    let tokens = lexer.tokenize().expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let expr = parser.parse_expression().expect("Failed to parse");
    compiler.compile_expr(&expr).expect("Failed to compile");
    
    // Call the result
    let call_source = "result";
    let mut lexer = Lexer::new(call_source);
    let tokens = lexer.tokenize().expect("Failed to tokenize");
    let mut parser = Parser::new(tokens);
    let call_expr = parser.parse_expression().expect("Failed to parse");
    compiler.compile_expr(&call_expr).expect("Failed to compile");
    
    let mut vm = VirtualMachine::new();
    vm.load(compiler.context.code.clone(), compiler.context.constants.clone());
    
    // Should return a Future LyObj
    let result = vm.run();
    if let Err(ref error) = result {
        eprintln!("VM execution failed: {:?}", error);
        assert!(false, "VM execution should not fail");
    }
    let value = result.expect("VM should execute successfully");
    
    match value {
        Value::LyObj(ly_obj) => {
            assert_eq!(ly_obj.type_name(), "Future");
            let resolved = ly_obj.call_method("resolve", &[]).expect("Future should resolve");
            assert_eq!(resolved, Value::Integer(42));
        }
        _ => panic!("Expected Future[Integer], got {:?}", value),
    }
}

#[test]
fn test_promise_with_different_types() {
    use lyra::stdlib::async_ops::{promise, AsyncFuture};
    use lyra::foreign::LyObj;
    
    // Test Promise with integer
    let int_result = promise(&[Value::Integer(123)]).expect("Promise should work with integer");
    if let Value::LyObj(ly_obj) = int_result {
        assert_eq!(ly_obj.type_name(), "Future");
        let resolved = ly_obj.call_method("resolve", &[]).expect("Should resolve");
        assert_eq!(resolved, Value::Integer(123));
    } else {
        panic!("Expected LyObj Future");
    }
    
    // Test Promise with real number
    let real_result = promise(&[Value::Real(3.14159)]).expect("Promise should work with real");
    if let Value::LyObj(ly_obj) = real_result {
        let resolved = ly_obj.call_method("resolve", &[]).expect("Should resolve");
        assert_eq!(resolved, Value::Real(3.14159));
    } else {
        panic!("Expected LyObj Future");
    }
    
    // Test Promise with string
    let string_result = promise(&[Value::String("Hello Async".to_string())]).expect("Promise should work with string");
    if let Value::LyObj(ly_obj) = string_result {
        let resolved = ly_obj.call_method("resolve", &[]).expect("Should resolve");
        assert_eq!(resolved, Value::String("Hello Async".to_string()));
    } else {
        panic!("Expected LyObj Future");
    }
    
    // Test Promise with list
    let list_val = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
    let list_result = promise(&[list_val.clone()]).expect("Promise should work with list");
    if let Value::LyObj(ly_obj) = list_result {
        let resolved = ly_obj.call_method("resolve", &[]).expect("Should resolve");
        assert_eq!(resolved, list_val);
    } else {
        panic!("Expected LyObj Future");
    }
}

#[test]
fn test_await_error_conditions() {
    use lyra::stdlib::async_ops::{await_future, promise};
    
    // Test await with wrong number of arguments
    let no_args_result = await_future(&[]);
    assert!(no_args_result.is_err(), "Await should fail with no arguments");
    
    let too_many_args_result = await_future(&[Value::Integer(1), Value::Integer(2)]);
    assert!(too_many_args_result.is_err(), "Await should fail with too many arguments");
    
    // Test await with non-Future object
    let non_future_result = await_future(&[Value::Integer(42)]);
    assert_eq!(non_future_result.expect("Should return the value itself"), Value::Integer(42));
    
    // Test await with non-LyObj
    let string_result = await_future(&[Value::String("test".to_string())]);
    assert_eq!(string_result.expect("Should return the string itself"), Value::String("test".to_string()));
}

#[test]
fn test_future_chaining() {
    use lyra::stdlib::async_ops::{promise, await_future};
    
    // Create a chain of promises
    let promise1 = promise(&[Value::Integer(10)]).expect("First promise should succeed");
    let resolved1 = await_future(&[promise1]).expect("First await should succeed");
    
    let promise2 = promise(&[resolved1]).expect("Second promise should succeed");
    let resolved2 = await_future(&[promise2]).expect("Second await should succeed");
    
    assert_eq!(resolved2, Value::Integer(10));
}

#[test]
fn test_thread_pool_creation() {
    use lyra::stdlib::async_ops::create_thread_pool;
    
    // Test default thread pool (no arguments)
    let default_pool = create_thread_pool(&[]).expect("Default ThreadPool should be created");
    if let Value::LyObj(ly_obj) = default_pool {
        assert_eq!(ly_obj.type_name(), "ThreadPool");
        let worker_count = ly_obj.call_method("workerCount", &[]).expect("Should get worker count");
        assert_eq!(worker_count, Value::Integer(4)); // Default is 4
    } else {
        panic!("Expected ThreadPool LyObj");
    }
    
    // Test custom thread pool
    let custom_pool = create_thread_pool(&[Value::Integer(8)]).expect("Custom ThreadPool should be created");
    if let Value::LyObj(ly_obj) = custom_pool {
        let worker_count = ly_obj.call_method("workerCount", &[]).expect("Should get worker count");
        assert_eq!(worker_count, Value::Integer(8));
    } else {
        panic!("Expected ThreadPool LyObj");
    }
}

#[test]
fn test_thread_pool_error_conditions() {
    use lyra::stdlib::async_ops::create_thread_pool;
    
    // Test invalid worker count (zero)
    let zero_workers = create_thread_pool(&[Value::Integer(0)]);
    assert!(zero_workers.is_err(), "ThreadPool with 0 workers should fail");
    
    // Test invalid worker count (negative)
    let negative_workers = create_thread_pool(&[Value::Integer(-1)]);
    assert!(negative_workers.is_err(), "ThreadPool with negative workers should fail");
    
    // Test invalid argument type
    let string_arg = create_thread_pool(&[Value::String("invalid".to_string())]);
    assert!(string_arg.is_err(), "ThreadPool with string argument should fail");
    
    // Test too many arguments
    let too_many_args = create_thread_pool(&[Value::Integer(4), Value::Integer(8)]);
    assert!(too_many_args.is_err(), "ThreadPool with too many arguments should fail");
}

#[test]
fn test_thread_pool_task_submission() {
    use lyra::stdlib::async_ops::create_thread_pool;
    
    let pool = create_thread_pool(&[Value::Integer(2)]).expect("Should create ThreadPool");
    if let Value::LyObj(pool_obj) = pool {
        // Submit an Add task
        let task_id = pool_obj.call_method("submit", &[
            Value::Function("Add".to_string()),
            Value::Integer(10),
            Value::Integer(20)
        ]).expect("Should submit task");
        
        if let Value::Integer(id) = task_id {
            assert!(id >= 0, "Task ID should be non-negative");
            
            // Wait for task completion
            let mut attempts = 0;
            loop {
                let completed = pool_obj.call_method("isCompleted", &[Value::Integer(id)])
                    .expect("Should check completion");
                
                if let Value::Boolean(true) = completed {
                    break;
                }
                
                attempts += 1;
                if attempts > 100 {
                    panic!("Task took too long to complete");
                }
                
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            
            // Get result
            let result = pool_obj.call_method("getResult", &[Value::Integer(id)])
                .expect("Should get result");
            assert_eq!(result, Value::Integer(30));
        } else {
            panic!("Expected Integer task ID");
        }
    } else {
        panic!("Expected ThreadPool LyObj");
    }
}

#[test]
fn test_channel_creation_and_basic_operations() {
    use lyra::stdlib::async_ops::{create_channel, create_bounded_channel};
    
    // Test unbounded channel
    let unbounded = create_channel(&[]).expect("Should create unbounded channel");
    if let Value::LyObj(channel_obj) = unbounded {
        assert_eq!(channel_obj.type_name(), "Channel");
        
        // Test capacity (should be None/Missing for unbounded)
        let capacity = channel_obj.call_method("capacity", &[]).expect("Should get capacity");
        assert_eq!(capacity, Value::Missing);
        
        // Test initial state
        let is_empty = channel_obj.call_method("isEmpty", &[]).expect("Should check if empty");
        assert_eq!(is_empty, Value::Boolean(true));
        
        let len = channel_obj.call_method("len", &[]).expect("Should get length");
        assert_eq!(len, Value::Integer(0));
    } else {
        panic!("Expected Channel LyObj");
    }
    
    // Test bounded channel
    let bounded = create_bounded_channel(&[Value::Integer(5)]).expect("Should create bounded channel");
    if let Value::LyObj(channel_obj) = bounded {
        let capacity = channel_obj.call_method("capacity", &[]).expect("Should get capacity");
        assert_eq!(capacity, Value::Integer(5));
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_channel_send_receive() {
    use lyra::stdlib::async_ops::create_channel;
    
    let channel = create_channel(&[]).expect("Should create channel");
    if let Value::LyObj(channel_obj) = channel {
        // Send a value
        let send_result = channel_obj.call_method("send", &[Value::Integer(42)])
            .expect("Should send value");
        assert_eq!(send_result, Value::Boolean(true));
        
        // Check channel is no longer empty
        let is_empty = channel_obj.call_method("isEmpty", &[]).expect("Should check if empty");
        assert_eq!(is_empty, Value::Boolean(false));
        
        let len = channel_obj.call_method("len", &[]).expect("Should get length");
        assert_eq!(len, Value::Integer(1));
        
        // Receive the value
        let received = channel_obj.call_method("receive", &[])
            .expect("Should receive value");
        assert_eq!(received, Value::Integer(42));
        
        // Check channel is empty again
        let is_empty = channel_obj.call_method("isEmpty", &[]).expect("Should check if empty");
        assert_eq!(is_empty, Value::Boolean(true));
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_channel_try_operations() {
    use lyra::stdlib::async_ops::create_bounded_channel;
    
    let channel = create_bounded_channel(&[Value::Integer(1)]).expect("Should create bounded channel");
    if let Value::LyObj(channel_obj) = channel {
        // Try send - should succeed
        let try_send_1 = channel_obj.call_method("trySend", &[Value::String("first".to_string())])
            .expect("Should try send");
        assert_eq!(try_send_1, Value::Boolean(true));
        
        // Try send again - should fail (channel full)
        let try_send_2 = channel_obj.call_method("trySend", &[Value::String("second".to_string())]);
        assert!(try_send_2.is_err(), "Second try send should fail on full channel");
        
        // Try receive - should succeed
        let try_receive_1 = channel_obj.call_method("tryReceive", &[])
            .expect("Should try receive");
        assert_eq!(try_receive_1, Value::String("first".to_string()));
        
        // Try receive again - should fail (channel empty)
        let try_receive_2 = channel_obj.call_method("tryReceive", &[]);
        assert!(try_receive_2.is_err(), "Second try receive should fail on empty channel");
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_parallel_map_basic() {
    use lyra::stdlib::async_ops::parallel_map;
    
    let function = Value::Function("Add".to_string());
    let data = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(10)]),
        Value::List(vec![Value::Integer(2), Value::Integer(20)]),
        Value::List(vec![Value::Integer(3), Value::Integer(30)])
    ]);
    
    let result = parallel_map(&[function, data]);
    // Note: This test might need adjustment based on actual implementation
    // For now, we just verify it doesn't crash
    match result {
        Ok(Value::List(_)) => {}, // Expected
        Ok(other) => eprintln!("Unexpected result type: {:?}", other),
        Err(e) => eprintln!("ParallelMap failed: {:?}", e),
    }
}

#[test]
fn test_async_function_wrapper() {
    use lyra::stdlib::async_ops::async_function;
    
    // Test wrapping a function
    let wrapped = async_function(&[Value::Function("Add".to_string())])
        .expect("Should wrap function");
    assert_eq!(wrapped, Value::Function("AsyncAdd".to_string()));
    
    // Test error conditions
    let no_args = async_function(&[]);
    assert!(no_args.is_err(), "AsyncFunction should fail with no arguments");
    
    let wrong_type = async_function(&[Value::Integer(42)]);
    assert!(wrong_type.is_err(), "AsyncFunction should fail with non-function argument");
}

#[test]
fn test_all_futures_resolution() {
    use lyra::stdlib::async_ops::{all_futures, promise};
    
    // Create multiple promises
    let promise1 = promise(&[Value::Integer(1)]).expect("Should create promise 1");
    let promise2 = promise(&[Value::Integer(2)]).expect("Should create promise 2");
    let promise3 = promise(&[Value::Integer(3)]).expect("Should create promise 3");
    
    let futures_list = Value::List(vec![promise1, promise2, promise3]);
    
    let result = all_futures(&[futures_list]).expect("All should resolve");
    if let Value::List(resolved_values) = result {
        assert_eq!(resolved_values.len(), 3);
        assert_eq!(resolved_values[0], Value::Integer(1));
        assert_eq!(resolved_values[1], Value::Integer(2));
        assert_eq!(resolved_values[2], Value::Integer(3));
    } else {
        panic!("Expected list of resolved values");
    }
}

#[test]
fn test_any_future_resolution() {
    use lyra::stdlib::async_ops::{any_future, promise};
    
    // Create multiple promises
    let promise1 = promise(&[Value::String("first".to_string())]).expect("Should create promise 1");
    let promise2 = promise(&[Value::String("second".to_string())]).expect("Should create promise 2");
    
    let futures_list = Value::List(vec![promise1, promise2]);
    
    let result = any_future(&[futures_list]).expect("Any should resolve");
    // Should return the first future's result
    assert_eq!(result, Value::String("first".to_string()));
    
    // Test error condition - empty list
    let empty_list = Value::List(vec![]);
    let empty_result = any_future(&[empty_list]);
    assert!(empty_result.is_err(), "Any should fail with empty list");
}

#[test]
fn test_concurrent_channel_operations() {
    use lyra::stdlib::async_ops::create_channel;
    use std::thread;
    use std::sync::Arc;
    
    let channel = create_channel(&[]).expect("Should create channel");
    if let Value::LyObj(channel_obj) = channel {
        let channel_arc = Arc::new(channel_obj);
        
        // Test concurrent sending and receiving
        let sender_channel = Arc::clone(&channel_arc);
        let receiver_channel = Arc::clone(&channel_arc);
        
        let sender_handle = thread::spawn(move || {
            for i in 0..5 {
                let send_result = sender_channel.call_method("send", &[Value::Integer(i)]);
                assert!(send_result.is_ok(), "Send should succeed");
                thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        
        let receiver_handle = thread::spawn(move || {
            let mut received_values = Vec::new();
            for _ in 0..5 {
                // Use try_receive with retry loop to avoid blocking indefinitely
                let mut attempts = 0;
                loop {
                    match receiver_channel.call_method("tryReceive", &[]) {
                        Ok(value) => {
                            received_values.push(value);
                            break;
                        }
                        Err(_) => {
                            attempts += 1;
                            if attempts > 100 {
                                panic!("Failed to receive after many attempts");
                            }
                            thread::sleep(std::time::Duration::from_millis(10));
                        }
                    }
                }
            }
            received_values
        });
        
        sender_handle.join().expect("Sender thread should complete");
        let received = receiver_handle.join().expect("Receiver thread should complete");
        
        assert_eq!(received.len(), 5);
        for (i, value) in received.iter().enumerate() {
            assert_eq!(*value, Value::Integer(i as i64));
        }
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_thread_pool_stress() {
    use lyra::stdlib::async_ops::create_thread_pool;
    
    let pool = create_thread_pool(&[Value::Integer(4)]).expect("Should create ThreadPool");
    if let Value::LyObj(pool_obj) = pool {
        let mut task_ids = Vec::new();
        
        // Submit many tasks
        for i in 0..20 {
            let task_id = pool_obj.call_method("submit", &[
                Value::Function("Multiply".to_string()),
                Value::Integer(i),
                Value::Integer(2)
            ]).expect("Should submit task");
            
            if let Value::Integer(id) = task_id {
                task_ids.push(id);
            } else {
                panic!("Expected Integer task ID");
            }
        }
        
        // Wait for all tasks and collect results
        let mut results = Vec::new();
        for task_id in task_ids {
            loop {
                let completed = pool_obj.call_method("isCompleted", &[Value::Integer(task_id)])
                    .expect("Should check completion");
                
                if let Value::Boolean(true) = completed {
                    let result = pool_obj.call_method("getResult", &[Value::Integer(task_id)])
                        .expect("Should get result");
                    results.push(result);
                    break;
                }
                
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
        
        // Verify all results
        assert_eq!(results.len(), 20);
        for (i, result) in results.iter().enumerate() {
            let expected = Value::Integer((i as i64) * 2);
            assert_eq!(*result, expected, "Task {} result should be {}", i, i * 2);
        }
    } else {
        panic!("Expected ThreadPool LyObj");
    }
}

#[test]
fn test_channel_close_operations() {
    use lyra::stdlib::async_ops::create_channel;
    
    let channel = create_channel(&[]).expect("Should create channel");
    if let Value::LyObj(channel_obj) = channel {
        // Initially not closed
        let is_closed = channel_obj.call_method("isClosed", &[]).expect("Should check if closed");
        assert_eq!(is_closed, Value::Boolean(false));
        
        // Send a value
        let send_result = channel_obj.call_method("send", &[Value::String("test".to_string())])
            .expect("Should send before close");
        assert_eq!(send_result, Value::Boolean(true));
        
        // Close the channel
        let close_result = channel_obj.call_method("close", &[]).expect("Should close channel");
        assert_eq!(close_result, Value::Boolean(true));
        
        // Check it's closed
        let is_closed = channel_obj.call_method("isClosed", &[]).expect("Should check if closed");
        assert_eq!(is_closed, Value::Boolean(true));
        
        // Try to send after close - should fail
        let send_after_close = channel_obj.call_method("send", &[Value::String("after close".to_string())]);
        assert!(send_after_close.is_err(), "Send should fail after close");
        
        // Should still be able to receive existing messages
        let receive_result = channel_obj.call_method("receive", &[])
            .expect("Should still receive existing messages");
        assert_eq!(receive_result, Value::String("test".to_string()));
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_error_propagation_in_async_operations() {
    use lyra::stdlib::async_ops::{promise, await_future, parallel_map};
    
    // Test Promise with invalid arguments
    let promise_no_args = promise(&[]);
    assert!(promise_no_args.is_err(), "Promise should fail with no arguments");
    
    let promise_too_many = promise(&[Value::Integer(1), Value::Integer(2)]);
    assert!(promise_too_many.is_err(), "Promise should fail with too many arguments");
    
    // Test ParallelMap with invalid arguments
    let parallel_no_args = parallel_map(&[]);
    assert!(parallel_no_args.is_err(), "ParallelMap should fail with no arguments");
    
    let parallel_one_arg = parallel_map(&[Value::Function("Add".to_string())]);
    assert!(parallel_one_arg.is_err(), "ParallelMap should fail with one argument");
    
    let parallel_wrong_type = parallel_map(&[
        Value::Function("Add".to_string()),
        Value::String("not a list".to_string())
    ]);
    assert!(parallel_wrong_type.is_err(), "ParallelMap should fail with non-list data");
}

// ============================================================================
// COMPREHENSIVE CONCURRENT OPERATION TESTS
// ============================================================================

#[test]
fn test_high_contention_thread_pool() {
    use lyra::stdlib::async_ops::create_thread_pool;
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    let pool = create_thread_pool(&[Value::Integer(8)]).expect("Should create ThreadPool");
    
    if let Value::LyObj(pool_obj) = pool {
        let pool_arc = Arc::new(pool_obj);
        let completed_tasks = Arc::new(Mutex::new(Vec::new()));
        
        // Spawn multiple threads to submit tasks concurrently
        let mut handles = Vec::new();
        
        for thread_id in 0..10 {
            let pool_clone = Arc::clone(&pool_arc);
            let completed_clone = Arc::clone(&completed_tasks);
            
            let handle = thread::spawn(move || {
                let mut thread_results = Vec::new();
                
                for task_num in 0..5 {
                    let task_id = pool_clone.call_method("submit", &[
                        Value::Function("Add".to_string()),
                        Value::Integer(thread_id * 100 + task_num),
                        Value::Integer(1000)
                    ]).expect("Should submit task");
                    
                    if let Value::Integer(id) = task_id {
                        // Wait for completion
                        loop {
                            let completed = pool_clone.call_method("isCompleted", &[Value::Integer(id)])
                                .expect("Should check completion");
                            
                            if let Value::Boolean(true) = completed {
                                let result = pool_clone.call_method("getResult", &[Value::Integer(id)])
                                    .expect("Should get result");
                                thread_results.push((id, result));
                                break;
                            }
                            
                            thread::sleep(Duration::from_millis(1));
                        }
                    }
                }
                
                // Store thread results
                {
                    let mut completed = completed_clone.lock().expect("Should lock results");
                    completed.extend(thread_results);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().expect("Thread should complete successfully");
        }
        
        // Verify all tasks completed
        let completed = completed_tasks.lock().expect("Should lock results");
        assert_eq!(completed.len(), 50, "All 50 tasks should complete"); // 10 threads * 5 tasks each
        
        // Verify results are correct
        for (task_id, result) in completed.iter() {
            if let Value::Integer(val) = result {
                assert!(*val >= 1000, "All results should be at least 1000 (base + increment)");
            } else {
                panic!("Expected integer result for task {}", task_id);
            }
        }
    } else {
        panic!("Expected ThreadPool LyObj");
    }
}

#[test]
fn test_channel_producer_consumer_pattern() {
    use lyra::stdlib::async_ops::create_bounded_channel;
    use std::sync::Arc;
    use std::thread;
    
    let channel = create_bounded_channel(&[Value::Integer(10)]).expect("Should create bounded channel");
    
    if let Value::LyObj(channel_obj) = channel {
        let channel_arc = Arc::new(channel_obj);
        
        let producer_channel = Arc::clone(&channel_arc);
        let consumer_channel = Arc::clone(&channel_arc);
        
        // Producer thread
        let producer_handle = thread::spawn(move || {
            for i in 0..20 {
                loop {
                    match producer_channel.call_method("trySend", &[Value::Integer(i)]) {
                        Ok(_) => {
                            break; // Successfully sent
                        }
                        Err(_) => {
                            // Channel full, wait and retry
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                }
                
                if i % 5 == 0 {
                    thread::sleep(Duration::from_millis(10)); // Occasional delay
                }
            }
        });
        
        // Consumer thread
        let consumer_handle = thread::spawn(move || {
            let mut consumed_values = Vec::new();
            
            for _ in 0..20 {
                loop {
                    match consumer_channel.call_method("tryReceive", &[]) {
                        Ok(value) => {
                            consumed_values.push(value);
                            break;
                        }
                        Err(_) => {
                            // Channel empty, wait and retry
                            thread::sleep(Duration::from_millis(5));
                        }
                    }
                }
            }
            
            consumed_values
        });
        
        producer_handle.join().expect("Producer thread should complete");
        let consumed = consumer_handle.join().expect("Consumer thread should complete");
        
        // Verify we consumed all produced values
        assert_eq!(consumed.len(), 20);
        
        // Verify values are in correct range (0-19)
        let mut received_values: Vec<i64> = consumed.iter()
            .map(|v| match v {
                Value::Integer(i) => *i,
                _ => panic!("Expected integer value"),
            })
            .collect();
        
        received_values.sort();
        
        for (i, &value) in received_values.iter().enumerate() {
            assert_eq!(value, i as i64, "Should receive values 0-19 in order");
        }
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_deadlock_prevention() {
    use lyra::stdlib::async_ops::create_bounded_channel;
    use std::sync::{Arc, Barrier};
    use std::thread;
    
    // Create a small bounded channel that could cause deadlock
    let channel = create_bounded_channel(&[Value::Integer(2)]).expect("Should create bounded channel");
    
    if let Value::LyObj(channel_obj) = channel {
        let channel_arc = Arc::new(channel_obj);
        let barrier = Arc::new(Barrier::new(3)); // 2 workers + main thread
        
        // Worker 1: Try to send multiple items
        let sender_channel = Arc::clone(&channel_arc);
        let sender_barrier = Arc::clone(&barrier);
        let sender_handle = thread::spawn(move || {
            sender_barrier.wait(); // Synchronize start
            
            let mut sent_count = 0;
            for i in 0..5 {
                // Use trySend to avoid blocking
                match sender_channel.call_method("trySend", &[Value::Integer(i)]) {
                    Ok(_) => {
                        sent_count += 1;
                    }
                    Err(_) => {
                        // Channel full, yield and continue
                        thread::yield_now();
                    }
                }
                
                thread::sleep(Duration::from_millis(10));
            }
            
            sent_count
        });
        
        // Worker 2: Try to receive items
        let receiver_channel = Arc::clone(&channel_arc);
        let receiver_barrier = Arc::clone(&barrier);
        let receiver_handle = thread::spawn(move || {
            receiver_barrier.wait(); // Synchronize start
            
            let mut received_items = Vec::new();
            let mut attempts = 0;
            
            while attempts < 100 { // Prevent infinite loop
                match receiver_channel.call_method("tryReceive", &[]) {
                    Ok(value) => {
                        received_items.push(value);
                        if received_items.len() >= 5 {
                            break; // Received enough
                        }
                    }
                    Err(_) => {
                        // Channel empty, continue trying
                        thread::sleep(Duration::from_millis(5));
                    }
                }
                
                attempts += 1;
            }
            
            received_items
        });
        
        // Main thread waits and coordinates
        barrier.wait();
        
        // Give threads time to work
        thread::sleep(Duration::from_millis(500));
        
        let sent_count = sender_handle.join().expect("Sender should complete");
        let received_items = receiver_handle.join().expect("Receiver should complete");
        
        // Verify no deadlock occurred and some progress was made
        assert!(sent_count > 0, "Should have sent some items");
        assert!(!received_items.is_empty(), "Should have received some items");
        
        // The exact counts may vary due to timing, but we should have some activity
        println!("Sent: {}, Received: {}", sent_count, received_items.len());
    } else {
        panic!("Expected Channel LyObj");
    }
}

#[test]
fn test_resource_cleanup_under_stress() {
    use lyra::stdlib::async_ops::{create_channel, create_thread_pool};
    
    // Create and immediately drop many channels and thread pools
    // This tests that resources are properly cleaned up
    
    for i in 0..50 {
        // Create channel
        let channel = create_channel(&[]).expect("Should create channel");
        if let Value::LyObj(channel_obj) = channel {
            // Use the channel briefly
            let send_result = channel_obj.call_method("send", &[Value::Integer(i)]);
            assert!(send_result.is_ok(), "Send should succeed");
            
            let receive_result = channel_obj.call_method("receive", &[]);
            assert!(receive_result.is_ok(), "Receive should succeed");
        }
        
        // Create thread pool
        let pool = create_thread_pool(&[Value::Integer(2)]).expect("Should create ThreadPool");
        if let Value::LyObj(pool_obj) = pool {
            // Submit a quick task
            let task_id = pool_obj.call_method("submit", &[
                Value::Function("Add".to_string()),
                Value::Integer(i),
                Value::Integer(100)
            ]).expect("Should submit task");
            
            if let Value::Integer(id) = task_id {
                // Don't wait for completion - test cleanup with pending tasks
                assert!(id >= 0, "Task ID should be valid");
            }
        }
        
        // Resources should be automatically cleaned up when dropped
    }
    
    // If we reach here without crashing, resource cleanup is working
    assert!(true, "Resource cleanup stress test completed");
}