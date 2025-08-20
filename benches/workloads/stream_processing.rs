//! Stream Processing Workload Simulations
//!
//! High-throughput data streaming scenarios with backpressure handling,
//! pipeline processing, and real-time data transformation.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::vm::Value;
use std::sync::{Arc, Mutex};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use std::collections::VecDeque;

/// Simulate streaming data source
fn generate_stream_data(count: usize, batch_size: usize) -> Vec<Vec<Value>> {
    (0..count).map(|batch_id| {
        (0..batch_size).map(|i| {
            Value::List(vec![
                Value::Integer((batch_id * batch_size + i) as i64),  // Timestamp
                Value::Real((i as f64 * 0.1).sin()),               // Sensor value
                Value::String(format!("device_{}", i % 10)),       // Device ID
                Value::Boolean(i % 2 == 0),                        // Status flag
            ])
        }).collect()
    }).collect()
}

/// Benchmark streaming data ingestion
fn streaming_data_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_ingestion");
    group.throughput(Throughput::Elements(10000));
    
    for batch_size in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("batch_ingestion", batch_size),
            batch_size,
            |b, &batch_size| {
                let stream_data = generate_stream_data(100, batch_size);
                
                b.iter(|| {
                    let (sender, receiver) = mpsc::channel();
                    let mut handles = vec![];
                    
                    // Data producer (simulates streaming source)
                    let data_clone = stream_data.clone();
                    let producer = thread::spawn(move || {
                        for batch in data_clone {
                            sender.send(batch).unwrap();
                            // Simulate variable ingestion rate
                            thread::sleep(Duration::from_micros(10));
                        }
                    });
                    handles.push(producer);
                    
                    // Data consumer (ingestion processor)
                    let consumer = thread::spawn(move || {
                        let mut ingested_count = 0;
                        let mut buffer = Vec::new();
                        
                        while let Ok(batch) = receiver.recv() {
                            buffer.extend(batch);
                            ingested_count += buffer.len();
                            
                            // Process buffer when it gets large enough
                            if buffer.len() >= 1000 {
                                // Simulate processing
                                buffer.clear();
                            }
                        }
                        
                        ingested_count + buffer.len()
                    });
                    
                    let total_ingested = consumer.join().unwrap();
                    producer.join().unwrap();
                    
                    black_box(total_ingested);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark stream processing pipeline
fn stream_processing_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_processing_pipeline");
    group.throughput(Throughput::Elements(5000));
    
    group.bench_function("multi_stage_pipeline", |b| {
        let stream_data = generate_stream_data(50, 100);
        
        b.iter(|| {
            let (input_sender, input_receiver) = mpsc::channel();
            let (filter_sender, filter_receiver) = mpsc::channel();
            let (transform_sender, transform_receiver) = mpsc::channel();
            let (output_sender, output_receiver) = mpsc::channel();
            
            let mut handles = vec![];
            
            // Stage 1: Data source
            let data_clone = stream_data.clone();
            let source = thread::spawn(move || {
                for batch in data_clone {
                    for record in batch {
                        input_sender.send(record).unwrap();
                    }
                }
            });
            handles.push(source);
            
            // Stage 2: Filter stage
            let filter = thread::spawn(move || {
                let mut filtered_count = 0;
                while let Ok(record) = input_receiver.recv() {
                    if let Value::List(fields) = &record {
                        // Filter: only pass records where sensor value > 0
                        if fields.len() >= 2 {
                            if let Value::Real(sensor_value) = &fields[1] {
                                if *sensor_value > 0.0 {
                                    filter_sender.send(record).unwrap();
                                    filtered_count += 1;
                                }
                            }
                        }
                    }
                }
                filtered_count
            });
            handles.push(filter);
            
            // Stage 3: Transform stage
            let transform = thread::spawn(move || {
                let mut transformed_count = 0;
                while let Ok(record) = filter_receiver.recv() {
                    if let Value::List(mut fields) = record {
                        // Transform: normalize sensor value and add computed field
                        if fields.len() >= 2 {
                            if let Value::Real(sensor_value) = &fields[1] {
                                // Normalize to 0-1 range
                                let normalized = (sensor_value + 1.0) / 2.0;
                                fields[1] = Value::Real(normalized);
                                
                                // Add computed field (square of normalized value)
                                fields.push(Value::Real(normalized * normalized));
                                
                                transform_sender.send(Value::List(fields)).unwrap();
                                transformed_count += 1;
                            }
                        }
                    }
                }
                transformed_count
            });
            handles.push(transform);
            
            // Stage 4: Output stage (aggregation)
            let output = thread::spawn(move || {
                let mut output_count = 0;
                let mut sum = 0.0;
                
                while let Ok(record) = transform_receiver.recv() {
                    if let Value::List(fields) = &record {
                        if fields.len() >= 2 {
                            if let Value::Real(value) = &fields[1] {
                                sum += value;
                                output_count += 1;
                            }
                        }
                    }
                }
                
                output_sender.send((output_count, sum)).unwrap();
                (output_count, sum)
            });
            
            let (count, total_sum) = output.join().unwrap();
            
            // Wait for other stages
            for handle in handles {
                handle.join().unwrap();
            }
            
            // Collect final result
            let final_result = output_receiver.recv().unwrap();
            
            black_box((count, total_sum, final_result));
        });
    });
    
    group.finish();
}

/// Benchmark backpressure handling
fn backpressure_handling_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("backpressure_handling");
    
    for buffer_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("bounded_queue_backpressure", buffer_size),
            buffer_size,
            |b, &buffer_size| {
                b.iter(|| {
                    let (sender, receiver) = mpsc::sync_channel(buffer_size);
                    let mut handles = vec![];
                    
                    // Fast producer
                    let fast_producer = thread::spawn(move || {
                        let mut sent_count = 0;
                        for i in 0..1000 {
                            let data = Value::Integer(i);
                            match sender.try_send(data) {
                                Ok(_) => sent_count += 1,
                                Err(mpsc::TrySendError::Full(_)) => {
                                    // Backpressure: queue is full, simulate retry
                                    thread::sleep(Duration::from_micros(1));
                                    if let Ok(_) = sender.send(Value::Integer(i)) {
                                        sent_count += 1;
                                    }
                                }
                                Err(_) => break,
                            }
                        }
                        sent_count
                    });
                    handles.push(fast_producer);
                    
                    // Slow consumer
                    let slow_consumer = thread::spawn(move || {
                        let mut consumed_count = 0;
                        while let Ok(_data) = receiver.recv() {
                            consumed_count += 1;
                            // Simulate slow processing
                            thread::sleep(Duration::from_micros(2));
                        }
                        consumed_count
                    });
                    
                    let sent = fast_producer.join().unwrap();
                    let consumed = slow_consumer.join().unwrap();
                    
                    black_box((sent, consumed));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark real-time stream analytics
fn realtime_stream_analytics(c: &mut Criterion) {
    let mut group = c.benchmark_group("realtime_analytics");
    
    group.bench_function("sliding_window_aggregation", |b| {
        let stream_data = generate_stream_data(100, 50);
        
        b.iter(|| {
            let window_size = 10;
            let mut sliding_window = VecDeque::new();
            let mut analytics_results = Vec::new();
            
            for batch in &stream_data {
                for record in batch {
                    if let Value::List(fields) = record {
                        if fields.len() >= 2 {
                            if let Value::Real(sensor_value) = &fields[1] {
                                // Add to sliding window
                                sliding_window.push_back(*sensor_value);
                                
                                // Maintain window size
                                if sliding_window.len() > window_size {
                                    sliding_window.pop_front();
                                }
                                
                                // Compute window statistics
                                if sliding_window.len() == window_size {
                                    let sum: f64 = sliding_window.iter().sum();
                                    let mean = sum / window_size as f64;
                                    
                                    let variance: f64 = sliding_window.iter()
                                        .map(|x| (x - mean).powi(2))
                                        .sum::<f64>() / window_size as f64;
                                    
                                    let min = sliding_window.iter()
                                        .fold(f64::INFINITY, |a, &b| a.min(b));
                                    let max = sliding_window.iter()
                                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                    
                                    analytics_results.push((mean, variance, min, max));
                                }
                            }
                        }
                    }
                }
            }
            
            black_box(analytics_results);
        });
    });
    
    group.bench_function("event_pattern_detection", |b| {
        let stream_data = generate_stream_data(100, 50);
        
        b.iter(|| {
            let mut pattern_matches = Vec::new();
            let mut recent_events = VecDeque::new();
            let pattern_window = 5;
            
            for batch in &stream_data {
                for record in batch {
                    if let Value::List(fields) = record {
                        recent_events.push_back(fields.clone());
                        
                        // Maintain window size
                        if recent_events.len() > pattern_window {
                            recent_events.pop_front();
                        }
                        
                        // Pattern: sequence of increasing sensor values
                        if recent_events.len() == pattern_window {
                            let mut is_increasing = true;
                            let mut prev_value = None;
                            
                            for event in &recent_events {
                                if event.len() >= 2 {
                                    if let Value::Real(sensor_value) = &event[1] {
                                        if let Some(prev) = prev_value {
                                            if sensor_value <= &prev {
                                                is_increasing = false;
                                                break;
                                            }
                                        }
                                        prev_value = Some(*sensor_value);
                                    }
                                }
                            }
                            
                            if is_increasing {
                                pattern_matches.push(recent_events.clone());
                            }
                        }
                    }
                }
            }
            
            black_box(pattern_matches);
        });
    });
    
    group.finish();
}

/// Benchmark parallel stream processing
fn parallel_stream_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_stream_processing");
    
    for worker_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_stream_workers", worker_count),
            worker_count,
            |b, &worker_count| {
                let stream_data = generate_stream_data(50, 100);
                
                b.iter(|| {
                    let (input_sender, input_receiver) = mpsc::channel();
                    let input_receiver = Arc::new(Mutex::new(input_receiver));
                    let results = Arc::new(Mutex::new(Vec::new()));
                    let mut handles = vec![];
                    
                    // Data source
                    let data_clone = stream_data.clone();
                    let source = thread::spawn(move || {
                        for batch in data_clone {
                            for record in batch {
                                input_sender.send(record).unwrap();
                            }
                        }
                    });
                    handles.push(source);
                    
                    // Parallel workers
                    for worker_id in 0..worker_count {
                        let receiver_clone = Arc::clone(&input_receiver);
                        let results_clone = Arc::clone(&results);
                        
                        let worker = thread::spawn(move || {
                            let mut worker_results = Vec::new();
                            
                            loop {
                                let record = {
                                    let receiver = receiver_clone.lock().unwrap();
                                    receiver.try_recv()
                                };
                                
                                match record {
                                    Ok(data) => {
                                        // Process the record
                                        if let Value::List(fields) = &data {
                                            if fields.len() >= 2 {
                                                if let Value::Real(sensor_value) = &fields[1] {
                                                    // Simple processing: square the sensor value
                                                    let processed = sensor_value * sensor_value;
                                                    worker_results.push((worker_id, processed));
                                                }
                                            }
                                        }
                                    }
                                    Err(mpsc::TryRecvError::Empty) => {
                                        // No data available, brief wait
                                        thread::sleep(Duration::from_micros(1));
                                    }
                                    Err(mpsc::TryRecvError::Disconnected) => {
                                        // Source finished
                                        break;
                                    }
                                }
                            }
                            
                            results_clone.lock().unwrap().extend(worker_results);
                        });
                        handles.push(worker);
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    let final_results = results.lock().unwrap().clone();
                    black_box(final_results);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark stream latency and throughput
fn stream_latency_throughput_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("stream_latency_throughput");
    
    group.bench_function("end_to_end_latency", |b| {
        b.iter(|| {
            let (sender, receiver) = mpsc::channel();
            let mut latencies = Vec::new();
            
            let processor = thread::spawn(move || {
                while let Ok((data, timestamp)) = receiver.recv() {
                    let processing_start = Instant::now();
                    
                    // Simulate processing
                    if let Value::List(fields) = &data {
                        if fields.len() >= 2 {
                            if let Value::Real(sensor_value) = &fields[1] {
                                // Simple computation
                                let _result = sensor_value.sin() + sensor_value.cos();
                            }
                        }
                    }
                    
                    let processing_end = Instant::now();
                    let latency = processing_end.duration_since(timestamp);
                    latencies.push(latency);
                }
                latencies
            });
            
            // Send data with timestamps
            for i in 0..1000 {
                let timestamp = Instant::now();
                let data = Value::List(vec![
                    Value::Integer(i),
                    Value::Real((i as f64 * 0.01).sin()),
                ]);
                sender.send((data, timestamp)).unwrap();
            }
            
            drop(sender);
            let measured_latencies = processor.join().unwrap();
            
            black_box(measured_latencies);
        });
    });
    
    group.throughput(Throughput::Elements(10000));
    group.bench_function("high_throughput_processing", |b| {
        b.iter(|| {
            let (sender, receiver) = mpsc::sync_channel(1000);
            let mut handles = vec![];
            
            // High-rate producer
            let producer = thread::spawn(move || {
                let mut sent = 0;
                for i in 0..10000 {
                    let data = Value::List(vec![
                        Value::Integer(i),
                        Value::Real((i as f64 * 0.001).sin()),
                        Value::String(format!("stream_{}", i % 100)),
                    ]);
                    
                    if sender.send(data).is_ok() {
                        sent += 1;
                    }
                }
                sent
            });
            handles.push(producer);
            
            // High-throughput consumer
            let consumer = thread::spawn(move || {
                let mut processed = 0;
                let mut batch = Vec::new();
                const BATCH_SIZE: usize = 100;
                
                while let Ok(data) = receiver.recv() {
                    batch.push(data);
                    
                    if batch.len() >= BATCH_SIZE {
                        // Process batch
                        for record in &batch {
                            if let Value::List(fields) = record {
                                if fields.len() >= 2 {
                                    if let Value::Real(value) = &fields[1] {
                                        // Simulate computation
                                        let _result = value * 2.0 + 1.0;
                                        processed += 1;
                                    }
                                }
                            }
                        }
                        batch.clear();
                    }
                }
                
                // Process remaining items
                processed += batch.len();
                processed
            });
            
            let sent = producer.join().unwrap();
            let processed = consumer.join().unwrap();
            
            black_box((sent, processed));
        });
    });
    
    group.finish();
}

criterion_group!(
    stream_processing_benchmarks,
    streaming_data_ingestion,
    stream_processing_pipeline,
    backpressure_handling_benchmark,
    realtime_stream_analytics,
    parallel_stream_processing,
    stream_latency_throughput_benchmark
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_stream_data_generation() {
        let stream_data = generate_stream_data(5, 10);
        assert_eq!(stream_data.len(), 5); // 5 batches
        
        for batch in &stream_data {
            assert_eq!(batch.len(), 10); // 10 records per batch
            
            for record in batch {
                if let Value::List(fields) = record {
                    assert_eq!(fields.len(), 4); // timestamp, sensor, device, status
                    assert!(matches!(fields[0], Value::Integer(_)));
                    assert!(matches!(fields[1], Value::Real(_)));
                    assert!(matches!(fields[2], Value::String(_)));
                    assert!(matches!(fields[3], Value::Boolean(_)));
                }
            }
        }
    }
    
    #[test]
    fn validate_stream_pipeline() {
        let (input_sender, input_receiver) = mpsc::channel();
        let (output_sender, output_receiver) = mpsc::channel();
        
        // Simple pipeline stage
        let processor = thread::spawn(move || {
            let mut count = 0;
            while let Ok(record) = input_receiver.recv() {
                if let Value::List(fields) = &record {
                    if fields.len() >= 2 {
                        if let Value::Real(value) = &fields[1] {
                            if *value > 0.0 {
                                output_sender.send(record).unwrap();
                                count += 1;
                            }
                        }
                    }
                }
            }
            count
        });
        
        // Send test data
        for i in 0..10 {
            let record = Value::List(vec![
                Value::Integer(i),
                Value::Real(if i % 2 == 0 { 0.5 } else { -0.5 }),
            ]);
            input_sender.send(record).unwrap();
        }
        drop(input_sender);
        
        let processed_count = processor.join().unwrap();
        
        // Collect output
        drop(output_sender);
        let mut output_count = 0;
        while let Ok(_) = output_receiver.recv() {
            output_count += 1;
        }
        
        assert_eq!(processed_count, output_count);
        assert_eq!(output_count, 5); // Half the records have positive values
    }
    
    #[test]
    fn validate_sliding_window() {
        let mut window = VecDeque::new();
        let window_size = 3;
        
        for i in 1..=10 {
            window.push_back(i as f64);
            
            if window.len() > window_size {
                window.pop_front();
            }
            
            if window.len() == window_size {
                let sum: f64 = window.iter().sum();
                let mean = sum / window_size as f64;
                
                // Verify window contains correct values
                assert_eq!(window.len(), window_size);
                
                // For i=3, window should be [1, 2, 3], mean = 2
                if i == 3 {
                    assert_eq!(mean, 2.0);
                }
            }
        }
    }
    
    #[test]
    fn validate_backpressure_handling() {
        let (sender, receiver) = mpsc::sync_channel(2); // Small buffer
        
        // Fast producer
        let producer = thread::spawn(move || {
            let mut sent = 0;
            for i in 0..10 {
                match sender.try_send(i) {
                    Ok(_) => sent += 1,
                    Err(mpsc::TrySendError::Full(_)) => {
                        // Backpressure detected
                        thread::sleep(Duration::from_millis(1));
                        if sender.send(i).is_ok() {
                            sent += 1;
                        }
                    }
                    Err(_) => break,
                }
            }
            sent
        });
        
        // Slow consumer
        let consumer = thread::spawn(move || {
            let mut received = 0;
            while let Ok(_data) = receiver.recv() {
                received += 1;
                thread::sleep(Duration::from_millis(2)); // Slow processing
            }
            received
        });
        
        let sent = producer.join().unwrap();
        let received = consumer.join().unwrap();
        
        assert_eq!(sent, received);
        assert_eq!(sent, 10); // All data should eventually be processed
    }
    
    #[test]
    fn validate_parallel_stream_processing() {
        let (sender, receiver) = mpsc::channel();
        let receiver = Arc::new(Mutex::new(receiver));
        let results = Arc::new(Mutex::new(Vec::new()));
        
        // Send test data
        let test_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        for value in &test_data {
            sender.send(*value).unwrap();
        }
        drop(sender);
        
        // Parallel workers
        let mut handles = vec![];
        for worker_id in 0..3 {
            let receiver_clone = Arc::clone(&receiver);
            let results_clone = Arc::clone(&results);
            
            let handle = thread::spawn(move || {
                let mut worker_results = Vec::new();
                
                while let Ok(data) = {
                    let receiver = receiver_clone.lock().unwrap();
                    receiver.try_recv()
                } {
                    // Process: square the value
                    worker_results.push((worker_id, data * data));
                }
                
                results_clone.lock().unwrap().extend(worker_results);
            });
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_results = results.lock().unwrap();
        
        // Should have processed all data
        assert_eq!(final_results.len(), test_data.len());
        
        // Verify results (each value should be squared)
        let mut processed_values: Vec<i32> = final_results.iter()
            .map(|(_, result)| *result)
            .collect();
        processed_values.sort();
        
        let expected: Vec<i32> = test_data.iter().map(|x| x * x).collect();
        assert_eq!(processed_values, expected);
    }
}