//! Simple async functionality validation test

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use lyra::vm::Value;
    use lyra::stdlib::async_ops::{thread_pool, channel, promise, send, receive, await_future};

    #[test]
    fn test_basic_async_operations() {
        println!("🚀 Testing Basic Async Operations");
        
        // Test 1: ThreadPool creation
        let start = Instant::now();
        let pool = thread_pool(&[]).unwrap();
        let pool_time = start.elapsed();
        
        assert!(matches!(pool, Value::LyObj(_)));
        println!("  ✓ ThreadPool created in {:?}", pool_time);
        
        // Test 2: Channel send/receive
        let start = Instant::now();
        let ch = channel(&[]).unwrap();
        let message = Value::Integer(42);
        send(&[ch.clone(), message.clone()]).unwrap();
        let received = receive(&[ch]).unwrap();
        let channel_time = start.elapsed();
        
        assert_eq!(received, message);
        println!("  ✓ Channel send/receive completed in {:?}", channel_time);
        
        // Test 3: Promise/await
        let start = Instant::now();
        let value = Value::String("test".to_string());
        let future = promise(&[value.clone()]).unwrap();
        let result = await_future(&[future]).unwrap();
        let promise_time = start.elapsed();
        
        assert_eq!(result, value);
        println!("  ✓ Promise/await completed in {:?}", promise_time);
        
        println!("  ✅ All basic async operations working correctly");
    }
    
    #[test]
    fn test_async_throughput() {
        println!("🚀 Testing Async Throughput");
        
        let iterations = 100;
        
        // Channel throughput test
        let start = Instant::now();
        let ch = channel(&[]).unwrap();
        
        for i in 0..iterations {
            let message = Value::Integer(i);
            send(&[ch.clone(), message]).unwrap();
        }
        
        for _ in 0..iterations {
            receive(&[ch.clone()]).unwrap();
        }
        
        let duration = start.elapsed();
        let throughput = (iterations * 2) as f64 / duration.as_secs_f64();
        
        println!("  ✓ Channel throughput: {:.2} ops/sec", throughput);
        assert!(throughput > 1000.0, "Channel throughput too low: {:.2} ops/sec", throughput);
        
        // Promise throughput test
        let start = Instant::now();
        for i in 0..iterations {
            let value = Value::Integer(i);
            let future = promise(&[value.clone()]).unwrap();
            let result = await_future(&[future]).unwrap();
            assert_eq!(result, Value::Integer(i));
        }
        
        let duration = start.elapsed();
        let promise_throughput = (iterations * 2) as f64 / duration.as_secs_f64();
        
        println!("  ✓ Promise throughput: {:.2} ops/sec", promise_throughput);
        assert!(promise_throughput > 500.0, "Promise throughput too low: {:.2} ops/sec", promise_throughput);
        
        println!("  ✅ Async throughput validation passed");
    }
}