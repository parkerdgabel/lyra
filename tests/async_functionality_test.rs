//! Test for basic async functionality
//! 
//! This test validates that our core async operations work correctly

use lyra::vm::Value;
use lyra::foreign::LyObj;
use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};

#[test]
fn test_threadpool_creation() {
    // Test default ThreadPool creation
    let result = thread_pool(&[]);
    match result {
        Ok(Value::LyObj(_)) => println!("✓ Default ThreadPool created successfully"),
        Ok(other) => panic!("Expected LyObj, got {:?}", other),
        Err(e) => panic!("ThreadPool creation failed: {:?}", e),
    }
    
    // Test ThreadPool with specified worker count
    let worker_count = Value::Integer(8);
    let result = thread_pool(&[worker_count]);
    match result {
        Ok(Value::LyObj(_)) => println!("✓ ThreadPool with 8 workers created successfully"),
        Ok(other) => panic!("Expected LyObj, got {:?}", other),
        Err(e) => panic!("ThreadPool creation failed: {:?}", e),
    }
}

#[test]
fn test_channel_creation() {
    // Test unbounded channel
    let result = channel(&[]);
    match result {
        Ok(Value::LyObj(_)) => println!("✓ Unbounded Channel created successfully"),
        Ok(other) => panic!("Expected LyObj, got {:?}", other),
        Err(e) => panic!("Channel creation failed: {:?}", e),
    }
    
    // Test bounded channel
    let capacity = Value::Integer(100);
    let result = bounded_channel(&[capacity]);
    match result {
        Ok(Value::LyObj(_)) => println!("✓ BoundedChannel with capacity 100 created successfully"),
        Ok(other) => panic!("Expected LyObj, got {:?}", other),
        Err(e) => panic!("BoundedChannel creation failed: {:?}", e),
    }
}

#[test]
fn test_promise_creation() {
    let value = Value::String("Hello World".to_string());
    let result = promise(&[value]);
    match result {
        Ok(Value::LyObj(_)) => println!("✓ Promise created successfully"),
        Ok(other) => panic!("Expected LyObj, got {:?}", other),
        Err(e) => panic!("Promise creation failed: {:?}", e),
    }
}

#[test]
fn test_channel_send_receive() {
    // Create a channel
    if let Ok(Value::LyObj(ch)) = channel(&[]) {
        let channel_value = Value::LyObj(ch.clone());
        let message = Value::String("Test Message".to_string());
        
        // Test send
        let send_result = send(&[channel_value.clone(), message.clone()]);
        match send_result {
            Ok(Value::Symbol(s)) if s == "Ok" => println!("✓ Message sent successfully"),
            Ok(other) => panic!("Expected Ok symbol, got {:?}", other),
            Err(e) => panic!("Send failed: {:?}", e),
        }
        
        // Test receive
        let receive_result = receive(&[channel_value]);
        match receive_result {
            Ok(Value::String(s)) if s == "Test Message" => println!("✓ Message received successfully"),
            Ok(other) => panic!("Expected test message, got {:?}", other),
            Err(e) => panic!("Receive failed: {:?}", e),
        }
    } else {
        panic!("Failed to create channel for send/receive test");
    }
}

#[test]
fn test_future_await() {
    let value = Value::Integer(42);
    if let Ok(Value::LyObj(promise_obj)) = promise(&[value]) {
        let promise_value = Value::LyObj(promise_obj);
        
        // Test await
        let await_result = await_future(&[promise_value]);
        match await_result {
            Ok(Value::Integer(42)) => println!("✓ Future await returned correct value"),
            Ok(other) => panic!("Expected integer 42, got {:?}", other),
            Err(e) => panic!("Await failed: {:?}", e),
        }
    } else {
        panic!("Failed to create promise for await test");
    }
}