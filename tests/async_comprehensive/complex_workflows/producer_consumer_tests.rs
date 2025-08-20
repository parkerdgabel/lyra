//! Producer-Consumer Workflow Tests
//! 
//! Tests for producer-consumer patterns including bounded channels, multiple producers,
//! multiple consumers, flow control, and complex coordination scenarios.

use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex, Barrier, Condvar};
use std::thread;
use std::collections::VecDeque;

#[cfg(test)]
mod producer_consumer_tests {
    use super::*;

    #[test]
    fn test_single_producer_single_consumer() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test basic single producer, single consumer pattern
        
        let buffer_size = 10;
        let items_to_produce = 100;
        
        let (mut producer, mut consumer) = ProducerConsumer::new(buffer_size);
        
        let producer_handle = thread::spawn(move || {
            let mut produced_count = 0;
            for i in 0..items_to_produce {
                producer.produce(i).unwrap();
                produced_count += 1;
                
                if i % 10 == 0 {
                    thread::sleep(Duration::from_millis(1)); // Simulate work
                }
            }
            producer.close();
            produced_count
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            while let Ok(item) = consumer.consume() {
                consumed_items.push(item);
                
                if consumed_items.len() % 10 == 0 {
                    thread::sleep(Duration::from_millis(1)); // Simulate work
                }
            }
            consumed_items
        });
        
        let produced_count = producer_handle.join().unwrap();
        let consumed_items = consumer_handle.join().unwrap();
        
        assert_eq!(produced_count, items_to_produce);
        assert_eq!(consumed_items.len(), items_to_produce);
        
        // Verify order preservation
        for (i, &item) in consumed_items.iter().enumerate() {
            assert_eq!(item, i as i64);
        }
    }

    #[test]
    fn test_multiple_producers_single_consumer() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test multiple producers feeding single consumer
        
        let buffer_size = 20;
        let num_producers = 4;
        let items_per_producer = 25;
        
        let (producers, mut consumer) = ProducerConsumer::new_multi_producer(buffer_size, num_producers);
        let barrier = Arc::new(Barrier::new(num_producers + 1)); // +1 for main thread
        
        let mut producer_handles = Vec::new();
        
        for (producer_id, mut producer) in producers.into_iter().enumerate() {
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier_clone.wait(); // Synchronize start
                
                let mut produced_items = Vec::new();
                for i in 0..items_per_producer {
                    let item = (producer_id as i64 * 1000) + i as i64;
                    producer.produce(item).unwrap();
                    produced_items.push(item);
                    
                    if i % 5 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
                producer.close();
                produced_items
            });
            
            producer_handles.push(handle);
        }
        
        barrier.wait(); // Start all producers
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            while let Ok(item) = consumer.consume() {
                consumed_items.push(item);
            }
            consumed_items
        });
        
        // Wait for all producers to complete
        let mut all_produced_items = Vec::new();
        for handle in producer_handles {
            let produced_items = handle.join().unwrap();
            all_produced_items.extend(produced_items);
        }
        
        let consumed_items = consumer_handle.join().unwrap();
        
        // Verify counts
        assert_eq!(all_produced_items.len(), num_producers * items_per_producer);
        assert_eq!(consumed_items.len(), num_producers * items_per_producer);
        
        // Verify all items were consumed (order may vary)
        let mut produced_sorted = all_produced_items;
        let mut consumed_sorted = consumed_items;
        produced_sorted.sort();
        consumed_sorted.sort();
        assert_eq!(produced_sorted, consumed_sorted);
    }

    #[test]
    fn test_single_producer_multiple_consumers() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test single producer with multiple consumers
        
        let buffer_size = 15;
        let num_consumers = 3;
        let total_items = 90; // Divisible by number of consumers
        
        let (mut producer, consumers) = ProducerConsumer::new_multi_consumer(buffer_size, num_consumers);
        let barrier = Arc::new(Barrier::new(num_consumers + 1)); // +1 for main thread
        
        let producer_handle = thread::spawn(move || {
            let mut produced_items = Vec::new();
            for i in 0..total_items {
                producer.produce(i).unwrap();
                produced_items.push(i);
                
                if i % 10 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            producer.close();
            produced_items
        });
        
        let mut consumer_handles = Vec::new();
        
        for (consumer_id, mut consumer) in consumers.into_iter().enumerate() {
            let barrier_clone = Arc::clone(&barrier);
            
            let handle = thread::spawn(move || {
                barrier_clone.wait(); // Synchronize start
                
                let mut consumed_items = Vec::new();
                while let Ok(item) = consumer.consume() {
                    consumed_items.push((consumer_id, item));
                    
                    if consumed_items.len() % 5 == 0 {
                        thread::sleep(Duration::from_millis(1));
                    }
                }
                consumed_items
            });
            
            consumer_handles.push(handle);
        }
        
        barrier.wait(); // Start all consumers
        
        let produced_items = producer_handle.join().unwrap();
        
        // Collect all consumed items
        let mut all_consumed_items = Vec::new();
        for handle in consumer_handles {
            let consumed_items = handle.join().unwrap();
            all_consumed_items.extend(consumed_items);
        }
        
        // Verify counts
        assert_eq!(produced_items.len(), total_items);
        assert_eq!(all_consumed_items.len(), total_items);
        
        // Verify each item was consumed exactly once
        let mut consumed_values: Vec<_> = all_consumed_items.iter().map(|(_, item)| *item).collect();
        consumed_values.sort();
        assert_eq!(consumed_values, (0..total_items as i64).collect::<Vec<_>>());
        
        // Verify work distribution among consumers
        let mut consumer_counts = vec![0; num_consumers];
        for (consumer_id, _) in all_consumed_items {
            consumer_counts[consumer_id] += 1;
        }
        
        // Each consumer should have processed some items
        for count in consumer_counts {
            assert!(count > 0);
            assert!(count <= total_items); // Upper bound check
        }
    }

    #[test]
    fn test_producer_consumer_with_priorities() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test producer-consumer with priority queuing
        
        let buffer_size = 20;
        let (mut producer, mut consumer) = ProducerConsumer::new_with_priorities(buffer_size);
        
        let producer_handle = thread::spawn(move || {
            // Produce items with different priorities
            for i in 0..30 {
                let priority = match i % 3 {
                    0 => Priority::High,
                    1 => Priority::Medium,
                    2 => Priority::Low,
                    _ => unreachable!(),
                };
                
                producer.produce_with_priority(i, priority).unwrap();
                
                if i % 5 == 0 {
                    thread::sleep(Duration::from_millis(2));
                }
            }
            producer.close();
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            while let Ok((item, priority)) = consumer.consume_with_priority() {
                consumed_items.push((item, priority));
                thread::sleep(Duration::from_millis(1));
            }
            consumed_items
        });
        
        producer_handle.join().unwrap();
        let consumed_items = consumer_handle.join().unwrap();
        
        assert_eq!(consumed_items.len(), 30);
        
        // Verify priority ordering: high priority items should generally come first
        let high_priority_positions: Vec<_> = consumed_items.iter()
            .enumerate()
            .filter(|(_, (_, priority))| matches!(priority, Priority::High))
            .map(|(pos, _)| pos)
            .collect();
        
        let low_priority_positions: Vec<_> = consumed_items.iter()
            .enumerate()
            .filter(|(_, (_, priority))| matches!(priority, Priority::Low))
            .map(|(pos, _)| pos)
            .collect();
        
        // Most high priority items should come before most low priority items
        if !high_priority_positions.is_empty() && !low_priority_positions.is_empty() {
            let avg_high_pos = high_priority_positions.iter().sum::<usize>() as f64 / high_priority_positions.len() as f64;
            let avg_low_pos = low_priority_positions.iter().sum::<usize>() as f64 / low_priority_positions.len() as f64;
            assert!(avg_high_pos < avg_low_pos);
        }
    }

    #[test]
    fn test_producer_consumer_flow_control() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test flow control mechanisms to prevent overwhelming
        
        let buffer_size = 5; // Small buffer to test flow control
        let (mut producer, mut consumer) = ProducerConsumer::new_with_flow_control(buffer_size);
        
        let metrics = Arc::new(Mutex::new(FlowControlMetrics::new()));
        let metrics_producer = Arc::clone(&metrics);
        let metrics_consumer = Arc::clone(&metrics);
        
        let producer_handle = thread::spawn(move || {
            for i in 0..50 {
                let start = Instant::now();
                
                match producer.try_produce(i) {
                    Ok(()) => {
                        metrics_producer.lock().unwrap().successful_produces += 1;
                    }
                    Err(ProducerError::BufferFull) => {
                        // Flow control kicked in, wait a bit
                        thread::sleep(Duration::from_millis(10));
                        producer.produce(i).unwrap(); // Blocking produce
                        metrics_producer.lock().unwrap().blocked_produces += 1;
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
                
                let elapsed = start.elapsed();
                if elapsed > Duration::from_millis(5) {
                    metrics_producer.lock().unwrap().slow_produces += 1;
                }
            }
            producer.close();
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            while let Ok(item) = consumer.consume() {
                consumed_items.push(item);
                
                // Simulate variable consumer speed
                if item % 10 == 0 {
                    thread::sleep(Duration::from_millis(20)); // Slow down occasionally
                    metrics_consumer.lock().unwrap().slow_consumes += 1;
                }
            }
            consumed_items
        });
        
        producer_handle.join().unwrap();
        let consumed_items = consumer_handle.join().unwrap();
        
        assert_eq!(consumed_items.len(), 50);
        
        let final_metrics = metrics.lock().unwrap();
        
        // Verify flow control was exercised
        assert!(final_metrics.blocked_produces > 0);
        assert!(final_metrics.slow_produces > 0);
        assert!(final_metrics.slow_consumes > 0);
        
        println!("Flow control metrics: {:?}", *final_metrics);
    }

    #[test]
    fn test_producer_consumer_batching() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test batch processing for efficiency
        
        let buffer_size = 100;
        let batch_size = 10;
        let total_items = 200;
        
        let (mut producer, mut consumer) = ProducerConsumer::new_with_batching(buffer_size, batch_size);
        
        let producer_handle = thread::spawn(move || {
            let mut batch = Vec::new();
            for i in 0..total_items {
                batch.push(i);
                
                if batch.len() == batch_size || i == total_items - 1 {
                    producer.produce_batch(batch.clone()).unwrap();
                    batch.clear();
                }
            }
            producer.close();
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut all_consumed = Vec::new();
            let mut batch_count = 0;
            
            while let Ok(batch) = consumer.consume_batch() {
                batch_count += 1;
                all_consumed.extend(batch);
            }
            
            (all_consumed, batch_count)
        });
        
        producer_handle.join().unwrap();
        let (consumed_items, batch_count) = consumer_handle.join().unwrap();
        
        assert_eq!(consumed_items.len(), total_items);
        assert_eq!(batch_count, total_items / batch_size); // Should be exactly 20 batches
        
        // Verify order preservation within and across batches
        for (i, &item) in consumed_items.iter().enumerate() {
            assert_eq!(item, i as i64);
        }
    }

    #[test]
    fn test_producer_consumer_with_transforms() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test producer-consumer with data transformation pipeline
        
        let buffer_size = 20;
        let (mut producer, mut consumer) = ProducerConsumer::new_with_transform(
            buffer_size,
            |x: i64| x * 2,        // Double the input
            |x: i64| x + 100,      // Add 100
        );
        
        let producer_handle = thread::spawn(move || {
            for i in 1..=50 {
                producer.produce(i).unwrap();
            }
            producer.close();
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            while let Ok(item) = consumer.consume() {
                consumed_items.push(item);
            }
            consumed_items
        });
        
        producer_handle.join().unwrap();
        let consumed_items = consumer_handle.join().unwrap();
        
        assert_eq!(consumed_items.len(), 50);
        
        // Verify transformation: (x * 2) + 100
        for (i, &item) in consumed_items.iter().enumerate() {
            let input = (i + 1) as i64;
            let expected = (input * 2) + 100;
            assert_eq!(item, expected);
        }
    }

    #[test]
    fn test_producer_consumer_graceful_shutdown() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test graceful shutdown with proper cleanup
        
        let buffer_size = 10;
        let (mut producer, mut consumer) = ProducerConsumer::new_with_shutdown(buffer_size);
        
        let shutdown_signal = Arc::new(Mutex::new(false));
        let shutdown_producer = Arc::clone(&shutdown_signal);
        let shutdown_consumer = Arc::clone(&shutdown_signal);
        
        let producer_handle = thread::spawn(move || {
            let mut produced_count = 0;
            for i in 0..1000 {
                if *shutdown_producer.lock().unwrap() {
                    break;
                }
                
                if producer.produce(i).is_ok() {
                    produced_count += 1;
                }
                
                thread::sleep(Duration::from_millis(1));
            }
            producer.shutdown_graceful(Duration::from_millis(100));
            produced_count
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            loop {
                match consumer.consume_with_timeout(Duration::from_millis(50)) {
                    Ok(item) => consumed_items.push(item),
                    Err(ConsumeError::Timeout) => {
                        if *shutdown_consumer.lock().unwrap() {
                            break;
                        }
                    }
                    Err(ConsumeError::Closed) => break,
                }
            }
            consumed_items
        });
        
        // Let it run for a bit, then signal shutdown
        thread::sleep(Duration::from_millis(50));
        *shutdown_signal.lock().unwrap() = true;
        
        let produced_count = producer_handle.join().unwrap();
        let consumed_items = consumer_handle.join().unwrap();
        
        // Should have produced and consumed some items
        assert!(produced_count > 0);
        assert!(consumed_items.len() > 0);
        
        // Consumer should have gotten most or all produced items
        assert!(consumed_items.len() <= produced_count);
        assert!(consumed_items.len() >= produced_count - buffer_size); // Allow for items in buffer
        
        println!("Graceful shutdown: produced {}, consumed {}", produced_count, consumed_items.len());
    }

    #[test]
    fn test_producer_consumer_error_handling() {
        // RED: Will fail until ProducerConsumer is implemented
        // Test error handling and recovery
        
        let buffer_size = 10;
        let (mut producer, mut consumer) = ProducerConsumer::new_with_error_handling(buffer_size);
        
        let producer_handle = thread::spawn(move || {
            let mut error_count = 0;
            for i in 0..100 {
                // Simulate occasional errors
                if i % 13 == 0 {
                    match producer.produce_with_error_simulation(i) {
                        Ok(()) => {}
                        Err(ProducerError::SimulatedFailure) => {
                            error_count += 1;
                            // Retry without error simulation
                            producer.produce(i).unwrap();
                        }
                        Err(e) => panic!("Unexpected error: {:?}", e),
                    }
                } else {
                    producer.produce(i).unwrap();
                }
            }
            producer.close();
            error_count
        });
        
        let consumer_handle = thread::spawn(move || {
            let mut consumed_items = Vec::new();
            let mut error_count = 0;
            
            while let result = consumer.consume() {
                match result {
                    Ok(item) => consumed_items.push(item),
                    Err(ConsumeError::Closed) => break,
                    Err(ConsumeError::CorruptedData) => {
                        error_count += 1;
                        continue; // Skip corrupted item
                    }
                    Err(e) => panic!("Unexpected error: {:?}", e),
                }
            }
            
            (consumed_items, error_count)
        });
        
        let producer_errors = producer_handle.join().unwrap();
        let (consumed_items, consumer_errors) = consumer_handle.join().unwrap();
        
        // Should have handled errors gracefully
        assert!(producer_errors > 0); // Should have encountered some errors
        assert_eq!(consumed_items.len(), 100); // All items should be consumed eventually
        
        // Verify data integrity despite errors
        for (i, &item) in consumed_items.iter().enumerate() {
            assert_eq!(item, i as i64);
        }
        
        println!("Error handling: producer errors {}, consumer errors {}", producer_errors, consumer_errors);
    }
}

// Placeholder types and implementations (RED phase - will fail compilation)

struct ProducerConsumer<T> {
    _phantom: std::marker::PhantomData<T>,
}

struct Producer<T> {
    _phantom: std::marker::PhantomData<T>,
}

struct Consumer<T> {
    _phantom: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Priority {
    High,
    Medium,
    Low,
}

#[derive(Debug)]
enum ProducerError {
    BufferFull,
    Closed,
    SimulatedFailure,
}

#[derive(Debug)]
enum ConsumeError {
    Timeout,
    Closed,
    CorruptedData,
}

#[derive(Debug)]
struct FlowControlMetrics {
    successful_produces: usize,
    blocked_produces: usize,
    slow_produces: usize,
    slow_consumes: usize,
}

impl FlowControlMetrics {
    fn new() -> Self {
        Self {
            successful_produces: 0,
            blocked_produces: 0,
            slow_produces: 0,
            slow_consumes: 0,
        }
    }
}

impl<T> ProducerConsumer<T> {
    fn new(_buffer_size: usize) -> (Producer<T>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new not yet implemented")
    }
    
    fn new_multi_producer(_buffer_size: usize, _num_producers: usize) -> (Vec<Producer<T>>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new_multi_producer not yet implemented")
    }
    
    fn new_multi_consumer(_buffer_size: usize, _num_consumers: usize) -> (Producer<T>, Vec<Consumer<T>>) {
        unimplemented!("ProducerConsumer::new_multi_consumer not yet implemented")
    }
    
    fn new_with_priorities(_buffer_size: usize) -> (Producer<T>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new_with_priorities not yet implemented")
    }
    
    fn new_with_flow_control(_buffer_size: usize) -> (Producer<T>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new_with_flow_control not yet implemented")
    }
    
    fn new_with_batching(_buffer_size: usize, _batch_size: usize) -> (Producer<Vec<T>>, Consumer<Vec<T>>) {
        unimplemented!("ProducerConsumer::new_with_batching not yet implemented")
    }
    
    fn new_with_transform<F1, F2, U>(_buffer_size: usize, _transform1: F1, _transform2: F2) -> (Producer<T>, Consumer<U>)
    where 
        F1: Fn(T) -> T + Send + Sync + 'static,
        F2: Fn(T) -> U + Send + Sync + 'static,
    {
        unimplemented!("ProducerConsumer::new_with_transform not yet implemented")
    }
    
    fn new_with_shutdown(_buffer_size: usize) -> (Producer<T>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new_with_shutdown not yet implemented")
    }
    
    fn new_with_error_handling(_buffer_size: usize) -> (Producer<T>, Consumer<T>) {
        unimplemented!("ProducerConsumer::new_with_error_handling not yet implemented")
    }
}

impl<T> Producer<T> {
    fn produce(&mut self, _item: T) -> Result<(), ProducerError> {
        unimplemented!("Producer::produce not yet implemented")
    }
    
    fn try_produce(&mut self, _item: T) -> Result<(), ProducerError> {
        unimplemented!("Producer::try_produce not yet implemented")
    }
    
    fn produce_with_priority(&mut self, _item: T, _priority: Priority) -> Result<(), ProducerError> {
        unimplemented!("Producer::produce_with_priority not yet implemented")
    }
    
    fn produce_batch(&mut self, _items: Vec<T>) -> Result<(), ProducerError> {
        unimplemented!("Producer::produce_batch not yet implemented")
    }
    
    fn produce_with_error_simulation(&mut self, _item: T) -> Result<(), ProducerError> {
        unimplemented!("Producer::produce_with_error_simulation not yet implemented")
    }
    
    fn close(&mut self) {
        unimplemented!("Producer::close not yet implemented")
    }
    
    fn shutdown_graceful(&mut self, _timeout: Duration) {
        unimplemented!("Producer::shutdown_graceful not yet implemented")
    }
}

impl<T> Consumer<T> {
    fn consume(&mut self) -> Result<T, ConsumeError> {
        unimplemented!("Consumer::consume not yet implemented")
    }
    
    fn consume_with_priority(&mut self) -> Result<(T, Priority), ConsumeError> {
        unimplemented!("Consumer::consume_with_priority not yet implemented")
    }
    
    fn consume_batch(&mut self) -> Result<Vec<T>, ConsumeError> {
        unimplemented!("Consumer::consume_batch not yet implemented")
    }
    
    fn consume_with_timeout(&mut self, _timeout: Duration) -> Result<T, ConsumeError> {
        unimplemented!("Consumer::consume_with_timeout not yet implemented")
    }
}