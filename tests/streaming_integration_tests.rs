//! Integration tests for streaming and event-driven architecture functionality

use lyra::vm::{Value, VM};
use lyra::foreign::LyObj;
use lyra::stdlib::streaming::*;
use lyra::error::LyraError;
use std::collections::HashMap;
use std::time::Duration;

#[tokio::test]
async fn test_kafka_producer_consumer_integration() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create Kafka producer
    let producer_result = vm.eval("KafkaProducer[{\"localhost:9092\"}, \"test-topic\"]").unwrap();
    assert!(matches!(producer_result, Value::LyObj(_)));
    
    // Create Kafka consumer
    let consumer_result = vm.eval("KafkaConsumer[{\"localhost:9092\"}, \"test-topic\", \"test-group\"]").unwrap();
    assert!(matches!(consumer_result, Value::LyObj(_)));
}

#[test]
fn test_event_store_publish_replay() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create event store
    let store_result = vm.eval("EventStore[\"test-store\", 4, 86400000]").unwrap();
    assert!(matches!(store_result, Value::LyObj(_)));
    
    // Store the event store in VM for method calls
    vm.set_variable("store".to_string(), store_result);
    
    // Publish an event
    let publish_result = vm.eval("store.publish(\"user_action\", \"user-123\", \"clicked\")").unwrap();
    assert!(matches!(publish_result, Value::String(_)));
    
    // Replay events
    let replay_result = vm.eval("store.replay(0, 9999999999, Missing)").unwrap();
    assert!(matches!(replay_result, Value::List(_)));
    
    if let Value::List(events) = replay_result {
        assert!(events.len() > 0);
    }
}

#[test]
fn test_message_queue_send_receive() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create message queue
    let queue_result = vm.eval("MessageQueue[\"test-queue\"]").unwrap();
    assert!(matches!(queue_result, Value::LyObj(_)));
    
    vm.set_variable("queue".to_string(), queue_result);
    
    // Send a message
    let send_result = vm.eval("queue.send(\"hello world\", 1, Missing)").unwrap();
    assert!(matches!(send_result, Value::String(_)));
    
    // Receive the message
    let receive_result = vm.eval("queue.receive(5000, 1)").unwrap();
    assert!(matches!(receive_result, Value::List(_)));
    
    if let Value::List(messages) = receive_result {
        assert_eq!(messages.len(), 1);
    }
}

#[test]
fn test_topic_publish_subscribe() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create topic
    let topic_result = vm.eval("TopicCreate[\"test-topic\", 4, 3]").unwrap();
    assert!(matches!(topic_result, Value::LyObj(_)));
    
    vm.set_variable("topic".to_string(), topic_result);
    
    // Publish to topic
    let publish_result = vm.eval("topic.publish(\"test message\", \"key1\")").unwrap();
    assert!(matches!(publish_result, Value::String(_)));
    
    // Check topic properties
    let name_result = vm.eval("topic.name()").unwrap();
    assert_eq!(name_result, Value::String("test-topic".to_string()));
    
    let partitions_result = vm.eval("topic.partitions()").unwrap();
    assert_eq!(partitions_result, Value::Number(4.0));
}

#[test]
fn test_websocket_server_client() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create WebSocket server
    let server_result = vm.eval("WebSocketServer[8080, \"/ws\"]").unwrap();
    assert!(matches!(server_result, Value::LyObj(_)));
    
    vm.set_variable("server".to_string(), server_result);
    
    // Create WebSocket client
    let client_result = vm.eval("WebSocketClient[\"ws://localhost:8080\", {\"protocol1\"}]").unwrap();
    assert!(matches!(client_result, Value::LyObj(_)));
    
    vm.set_variable("client".to_string(), client_result);
    
    // Connect client
    let connect_result = vm.eval("client.connect()").unwrap();
    assert_eq!(connect_result, Value::Boolean(true));
    
    // Send message
    let send_result = vm.eval("client.send(\"hello websocket\")").unwrap();
    assert_eq!(send_result, Value::Boolean(true));
    
    // Broadcast from server
    let broadcast_result = vm.eval("server.broadcast(\"broadcast message\", Missing)").unwrap();
    assert!(matches!(broadcast_result, Value::Number(_)));
}

#[test]
fn test_server_sent_events() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create SSE endpoint
    let sse_result = vm.eval("ServerSentEvents[\"/events\", \"json\"]").unwrap();
    assert!(matches!(sse_result, Value::LyObj(_)));
    
    vm.set_variable("sse".to_string(), sse_result);
    
    // Send event
    let send_result = vm.eval("sse.send(\"event data\", \"update\")").unwrap();
    assert!(matches!(send_result, Value::Number(_)));
    
    // Check endpoint
    let endpoint_result = vm.eval("sse.endpoint()").unwrap();
    assert_eq!(endpoint_result, Value::String("/events".to_string()));
}

#[test]
fn test_grpc_server_client() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create gRPC server
    let server_result = vm.eval("gRPCServer[50051]").unwrap();
    assert!(matches!(server_result, Value::LyObj(_)));
    
    vm.set_variable("grpc_server".to_string(), server_result);
    
    // Create gRPC client
    let client_result = vm.eval("gRPCClient[\"localhost:50051\", \"TestService\"]").unwrap();
    assert!(matches!(client_result, Value::LyObj(_)));
    
    vm.set_variable("grpc_client".to_string(), client_result);
    
    // Start server
    let start_result = vm.eval("grpc_server.start()").unwrap();
    assert_eq!(start_result, Value::Boolean(true));
    
    // Make client call
    let call_result = vm.eval("grpc_client.call(\"TestMethod\", \"request data\")").unwrap();
    assert!(matches!(call_result, Value::String(_)));
}

#[test]
fn test_window_aggregate_processing() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create window aggregator
    let window_result = vm.eval("WindowAggregate[\"tumbling\", 60000, \"sum\"]").unwrap();
    assert!(matches!(window_result, Value::LyObj(_)));
    
    vm.set_variable("window".to_string(), window_result);
    
    // Process values
    let process_result1 = vm.eval("window.process(10, 1000)").unwrap();
    assert!(matches!(process_result1, Value::List(_)));
    
    let process_result2 = vm.eval("window.process(20, 2000)").unwrap();
    assert!(matches!(process_result2, Value::List(_)));
    
    // Update watermark
    let watermark_result = vm.eval("window.updateWatermark(100000)").unwrap();
    assert_eq!(watermark_result, Value::Boolean(true));
}

#[test]
fn test_stream_join_operations() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create stream join
    let join_result = vm.eval("StreamJoin[\"user_id\", 300000]").unwrap();
    assert!(matches!(join_result, Value::LyObj(_)));
    
    vm.set_variable("join".to_string(), join_result);
    
    // Process left stream
    let left_result = vm.eval("join.processLeft(\"left_data\", \"user123\", 1000)").unwrap();
    assert!(matches!(left_result, Value::List(_)));
    
    // Process right stream
    let right_result = vm.eval("join.processRight(\"right_data\", \"user123\", 1100)").unwrap();
    assert!(matches!(right_result, Value::List(_)));
    
    if let Value::List(joins) = right_result {
        // Should have found a matching join
        assert!(joins.len() > 0);
    }
}

#[test]
fn test_complex_event_processing() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create CEP engine
    let cep_result = vm.eval("ComplexEventProcessing[]").unwrap();
    assert!(matches!(cep_result, Value::LyObj(_)));
    
    vm.set_variable("cep".to_string(), cep_result);
    
    // Process events
    let event_result = vm.eval("cep.processEvent(\"login\", \"user123\", 1000)").unwrap();
    assert!(matches!(event_result, Value::List(_)));
}

#[test]
fn test_backpressure_control() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create backpressure controller
    let bp_result = vm.eval("BackpressureControl[\"adaptive\"]").unwrap();
    assert!(matches!(bp_result, Value::LyObj(_)));
    
    vm.set_variable("bp".to_string(), bp_result);
    
    // Update pressure
    let update_result = vm.eval("bp.updatePressure(700, 1000)").unwrap();
    assert_eq!(update_result, Value::Boolean(true));
    
    // Check if backpressure should be applied
    let should_apply_result = vm.eval("bp.shouldApplyBackpressure()").unwrap();
    assert_eq!(should_apply_result, Value::Boolean(true)); // 0.7 >= 0.7 threshold
    
    // Get metrics
    let metrics_result = vm.eval("bp.getMetrics()").unwrap();
    assert!(matches!(metrics_result, Value::List(_)));
}

#[test]
fn test_connection_pool() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create connection pool
    let pool_result = vm.eval("ConnectionPool[\"websocket\", 10]").unwrap();
    assert!(matches!(pool_result, Value::LyObj(_)));
}

#[test]
fn test_dead_letter_queue() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create dead letter queue
    let dlq_result = vm.eval("DeadLetterQueue[\"test-queue\", 3, 1000]").unwrap();
    assert!(matches!(dlq_result, Value::LyObj(_)));
}

#[test]
fn test_streaming_error_handling() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Test invalid arguments
    let invalid_kafka_result = vm.eval("KafkaProducer[{}, \"topic\"]");
    assert!(invalid_kafka_result.is_err());
    
    let invalid_queue_result = vm.eval("MessageQueue[]");
    assert!(invalid_queue_result.is_err());
    
    let invalid_window_result = vm.eval("WindowAggregate[\"unknown\", 1000, \"sum\"]");
    assert!(invalid_window_result.is_err());
}

#[test]
fn test_streaming_performance() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create message queue
    let queue_result = vm.eval("MessageQueue[\"perf-test\"]").unwrap();
    vm.set_variable("queue".to_string(), queue_result);
    
    // Send many messages
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let expr = format!("queue.send(\"message-{}\", 1, Missing)", i);
        vm.eval(&expr).unwrap();
    }
    let send_duration = start.elapsed();
    
    // Receive many messages
    let start = std::time::Instant::now();
    let receive_result = vm.eval("queue.receive(10000, 1000)").unwrap();
    let receive_duration = start.elapsed();
    
    println!("Send 1000 messages: {:?}", send_duration);
    println!("Receive 1000 messages: {:?}", receive_duration);
    
    if let Value::List(messages) = receive_result {
        assert_eq!(messages.len(), 1000);
    }
}

#[test]
fn test_high_throughput_windowing() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create window aggregator
    let window_result = vm.eval("WindowAggregate[\"global\", 1000, \"count\"]").unwrap();
    vm.set_variable("window".to_string(), window_result);
    
    // Process many values
    let start = std::time::Instant::now();
    for i in 0..1000 {
        let expr = format!("window.process({}, {})", i, i * 10);
        vm.eval(&expr).unwrap();
    }
    let processing_duration = start.elapsed();
    
    println!("Process 1000 values through window: {:?}", processing_duration);
    
    // Trigger window computation
    let watermark_result = vm.eval("window.updateWatermark(999999)").unwrap();
    assert_eq!(watermark_result, Value::Boolean(true));
}

#[test]
fn test_concurrent_streaming_operations() {
    use std::thread;
    use std::sync::Arc;
    
    let vm = Arc::new(std::sync::Mutex::new(VM::new()));
    
    // Register streaming functions
    {
        let mut vm_lock = vm.lock().unwrap();
        let streaming_functions = streaming::register_functions();
        for (name, func) in streaming_functions {
            vm_lock.register_builtin_function(name, func);
        }
        
        // Create shared message queue
        let queue_result = vm_lock.eval("MessageQueue[\"concurrent-test\"]").unwrap();
        vm_lock.set_variable("queue".to_string(), queue_result);
    }
    
    // Spawn multiple threads to send messages
    let mut handles = vec![];
    for thread_id in 0..5 {
        let vm_clone = Arc::clone(&vm);
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let mut vm_lock = vm_clone.lock().unwrap();
                let expr = format!("queue.send(\"thread-{}-msg-{}\", 1, Missing)", thread_id, i);
                vm_lock.eval(&expr).unwrap();
            }
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify all messages were sent
    let mut vm_lock = vm.lock().unwrap();
    let stats_result = vm_lock.eval("queue.stats()").unwrap();
    
    if let Value::List(stats_list) = stats_result {
        for stat in stats_list {
            if let Value::List(pair) = stat {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::Number(value)) = (&pair[0], &pair[1]) {
                        if key == "messages_sent" {
                            assert_eq!(*value, 500.0); // 5 threads * 100 messages
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn test_streaming_memory_usage() {
    let mut vm = VM::new();
    
    // Register streaming functions
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    
    // Create various streaming objects
    let _kafka_producer = vm.eval("KafkaProducer[{\"localhost:9092\"}, \"test\"]").unwrap();
    let _kafka_consumer = vm.eval("KafkaConsumer[{\"localhost:9092\"}, \"test\", \"group\"]").unwrap();
    let _event_store = vm.eval("EventStore[\"store\", 4, 86400000]").unwrap();
    let _message_queue = vm.eval("MessageQueue[\"queue\"]").unwrap();
    let _topic = vm.eval("TopicCreate[\"topic\", 4, 3]").unwrap();
    let _websocket_server = vm.eval("WebSocketServer[8080, \"/ws\"]").unwrap();
    let _websocket_client = vm.eval("WebSocketClient[\"ws://localhost:8080\", {}]").unwrap();
    let _sse = vm.eval("ServerSentEvents[\"/events\", \"json\"]").unwrap();
    let _grpc_server = vm.eval("gRPCServer[50051]").unwrap();
    let _grpc_client = vm.eval("gRPCClient[\"localhost:50051\", \"Service\"]").unwrap();
    let _window_agg = vm.eval("WindowAggregate[\"tumbling\", 60000, \"sum\"]").unwrap();
    let _stream_join = vm.eval("StreamJoin[\"key\", 300000]").unwrap();
    let _cep = vm.eval("ComplexEventProcessing[]").unwrap();
    let _backpressure = vm.eval("BackpressureControl[\"adaptive\"]").unwrap();
    let _connection_pool = vm.eval("ConnectionPool[\"websocket\", 10]").unwrap();
    let _dlq = vm.eval("DeadLetterQueue[\"queue\", 3, 1000]").unwrap();
    
    // All objects should be created successfully without excessive memory usage
    // In a real test, we'd measure actual memory usage here
}

// Helper function to create a VM with streaming functions registered
fn create_streaming_vm() -> VM {
    let mut vm = VM::new();
    let streaming_functions = streaming::register_functions();
    for (name, func) in streaming_functions {
        vm.register_builtin_function(name, func);
    }
    vm
}

#[test]
fn test_streaming_vm_helper() {
    let vm = create_streaming_vm();
    
    // Test that streaming functions are available
    let kafka_result = vm.eval("KafkaProducer[{\"localhost:9092\"}, \"test\"]");
    assert!(kafka_result.is_ok());
}