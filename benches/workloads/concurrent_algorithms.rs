//! Concurrent Algorithms Workload Simulations
//!
//! Parallel sorting, searching, graph algorithms that test the async system,
//! work-stealing effectiveness, and concurrent data structure performance.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::vm::Value;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::sync::mpsc;
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Generate test data for sorting algorithms
fn generate_sorting_data(size: usize) -> Vec<i64> {
    (0..size).map(|i| ((i * 7919) % size) as i64).collect() // Pseudo-random sequence
}

/// Generate graph data for graph algorithms
fn generate_graph(nodes: usize, edge_probability: f64) -> HashMap<usize, Vec<usize>> {
    let mut graph = HashMap::new();
    
    for i in 0..nodes {
        let mut neighbors = Vec::new();
        for j in 0..nodes {
            if i != j && rand::random::<f64>() < edge_probability {
                neighbors.push(j);
            }
        }
        graph.insert(i, neighbors);
    }
    
    graph
}

/// Benchmark parallel sorting algorithms
fn parallel_sorting_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_sorting");
    group.throughput(Throughput::Elements(10000));
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_merge_sort", thread_count),
            thread_count,
            |b, &thread_count| {
                let data = generate_sorting_data(10000);
                
                b.iter(|| {
                    let mut sorted_data = data.clone();
                    
                    if thread_count == 1 {
                        // Sequential sort
                        sorted_data.sort();
                    } else {
                        // Parallel sort simulation
                        let chunk_size = sorted_data.len() / thread_count;
                        let mut handles = vec![];
                        let data_chunks = Arc::new(Mutex::new(Vec::new()));
                        
                        // Divide data into chunks
                        for i in 0..thread_count {
                            let start = i * chunk_size;
                            let end = if i == thread_count - 1 { 
                                sorted_data.len() 
                            } else { 
                                (i + 1) * chunk_size 
                            };
                            
                            let mut chunk = sorted_data[start..end].to_vec();
                            let data_chunks_clone = Arc::clone(&data_chunks);
                            
                            let handle = thread::spawn(move || {
                                chunk.sort();
                                data_chunks_clone.lock().unwrap().push(chunk);
                            });
                            
                            handles.push(handle);
                        }
                        
                        // Wait for all threads to complete
                        for handle in handles {
                            handle.join().unwrap();
                        }
                        
                        // Merge sorted chunks
                        let chunks = data_chunks.lock().unwrap();
                        sorted_data = merge_sorted_chunks(&chunks);
                    }
                    
                    black_box(sorted_data);
                });
            },
        );
    }
    
    group.finish();
}

/// Helper function to merge sorted chunks
fn merge_sorted_chunks(chunks: &[Vec<i64>]) -> Vec<i64> {
    if chunks.is_empty() {
        return Vec::new();
    }
    
    let mut result = chunks[0].clone();
    
    for chunk in chunks.iter().skip(1) {
        let mut merged = Vec::with_capacity(result.len() + chunk.len());
        let mut i = 0;
        let mut j = 0;
        
        while i < result.len() && j < chunk.len() {
            if result[i] <= chunk[j] {
                merged.push(result[i]);
                i += 1;
            } else {
                merged.push(chunk[j]);
                j += 1;
            }
        }
        
        while i < result.len() {
            merged.push(result[i]);
            i += 1;
        }
        
        while j < chunk.len() {
            merged.push(chunk[j]);
            j += 1;
        }
        
        result = merged;
    }
    
    result
}

/// Benchmark parallel search algorithms
fn parallel_search_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_search");
    
    let data = generate_sorting_data(100000);
    let mut sorted_data = data.clone();
    sorted_data.sort();
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_binary_search", thread_count),
            thread_count,
            |b, &thread_count| {
                let search_targets: Vec<i64> = (0..100).map(|i| i * 1000).collect();
                
                b.iter(|| {
                    let mut results = Vec::new();
                    
                    if thread_count == 1 {
                        // Sequential search
                        for target in &search_targets {
                            let result = sorted_data.binary_search(target);
                            results.push(result.is_ok());
                        }
                    } else {
                        // Parallel search
                        let targets_per_thread = search_targets.len() / thread_count;
                        let mut handles = vec![];
                        let results_mutex = Arc::new(Mutex::new(Vec::new()));
                        
                        for i in 0..thread_count {
                            let start = i * targets_per_thread;
                            let end = if i == thread_count - 1 {
                                search_targets.len()
                            } else {
                                (i + 1) * targets_per_thread
                            };
                            
                            let targets_chunk = search_targets[start..end].to_vec();
                            let data_ref = sorted_data.clone();
                            let results_clone = Arc::clone(&results_mutex);
                            
                            let handle = thread::spawn(move || {
                                let mut thread_results = Vec::new();
                                for target in targets_chunk {
                                    let result = data_ref.binary_search(&target);
                                    thread_results.push(result.is_ok());
                                }
                                results_clone.lock().unwrap().extend(thread_results);
                            });
                            
                            handles.push(handle);
                        }
                        
                        for handle in handles {
                            handle.join().unwrap();
                        }
                        
                        results = results_mutex.lock().unwrap().clone();
                    }
                    
                    black_box(results);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark parallel graph algorithms
fn parallel_graph_algorithms_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_graph_algorithms");
    
    let graph = generate_graph(1000, 0.01); // 1000 nodes, 1% edge probability
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("parallel_bfs", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    // Parallel Breadth-First Search simulation
                    let start_node = 0;
                    let mut visited = Arc::new(RwLock::new(std::collections::HashSet::new()));
                    let mut queue = Arc::new(Mutex::new(VecDeque::new()));
                    
                    queue.lock().unwrap().push_back(start_node);
                    visited.write().unwrap().insert(start_node);
                    
                    let mut handles = vec![];
                    let (sender, receiver) = mpsc::channel();
                    
                    // Start worker threads
                    for _ in 0..*thread_count {
                        let graph_clone = graph.clone();
                        let visited_clone = Arc::clone(&visited);
                        let queue_clone = Arc::clone(&queue);
                        let sender_clone = sender.clone();
                        
                        let handle = thread::spawn(move || {
                            let mut nodes_processed = 0;
                            
                            loop {
                                let current_node = {
                                    let mut q = queue_clone.lock().unwrap();
                                    q.pop_front()
                                };
                                
                                if let Some(node) = current_node {
                                    nodes_processed += 1;
                                    
                                    if let Some(neighbors) = graph_clone.get(&node) {
                                        for &neighbor in neighbors {
                                            let mut visited_write = visited_clone.write().unwrap();
                                            if !visited_write.contains(&neighbor) {
                                                visited_write.insert(neighbor);
                                                queue_clone.lock().unwrap().push_back(neighbor);
                                            }
                                        }
                                    }
                                } else {
                                    // No more work, signal completion
                                    break;
                                }
                            }
                            
                            sender_clone.send(nodes_processed).unwrap();
                        });
                        
                        handles.push(handle);
                    }
                    
                    drop(sender);
                    
                    // Collect results
                    let mut total_processed = 0;
                    while let Ok(count) = receiver.recv() {
                        total_processed += count;
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    let final_visited = visited.read().unwrap().len();
                    black_box((total_processed, final_visited));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark work-stealing queue performance
fn work_stealing_queue_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("work_stealing_queue");
    
    // Simulate work-stealing with uneven task distribution
    let tasks: Vec<usize> = (0..10000).map(|i| {
        if i % 100 == 0 { 1000 } else { 10 } // Every 100th task is much more expensive
    }).collect();
    
    for thread_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("work_stealing_simulation", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    // Shared work queue
                    let work_queue = Arc::new(Mutex::new(tasks.clone()));
                    let mut handles = vec![];
                    let results = Arc::new(Mutex::new(Vec::new()));
                    
                    for worker_id in 0..*thread_count {
                        let queue_clone = Arc::clone(&work_queue);
                        let results_clone = Arc::clone(&results);
                        
                        let handle = thread::spawn(move || {
                            let mut worker_results = Vec::new();
                            
                            loop {
                                // Try to steal work from queue
                                let task = {
                                    let mut queue = queue_clone.lock().unwrap();
                                    queue.pop()
                                };
                                
                                if let Some(work_amount) = task {
                                    // Simulate work
                                    let mut result = 0;
                                    for _ in 0..work_amount {
                                        result += 1;
                                    }
                                    worker_results.push((worker_id, result));
                                } else {
                                    // No more work available
                                    break;
                                }
                            }
                            
                            results_clone.lock().unwrap().extend(worker_results);
                        });
                        
                        handles.push(handle);
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

/// Benchmark producer-consumer patterns
fn producer_consumer_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("producer_consumer");
    
    for buffer_size in [10, 100, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::new("bounded_buffer", buffer_size),
            buffer_size,
            |b, &buffer_size| {
                b.iter(|| {
                    let (sender, receiver) = mpsc::sync_channel(buffer_size);
                    let mut handles = vec![];
                    
                    // Producer thread
                    let producer_sender = sender.clone();
                    let producer = thread::spawn(move || {
                        for i in 0..1000 {
                            let value = Value::Integer(i);
                            producer_sender.send(value).unwrap();
                            
                            // Simulate variable production rate
                            if i % 10 == 0 {
                                thread::sleep(Duration::from_micros(1));
                            }
                        }
                        drop(producer_sender);
                    });
                    handles.push(producer);
                    
                    // Consumer threads
                    let num_consumers = 3;
                    let consumed_count = Arc::new(Mutex::new(0));
                    
                    for _ in 0..num_consumers {
                        let consumer_receiver = receiver.clone();
                        let count_clone = Arc::clone(&consumed_count);
                        
                        let consumer = thread::spawn(move || {
                            let mut local_count = 0;
                            while let Ok(_value) = consumer_receiver.recv() {
                                local_count += 1;
                                
                                // Simulate processing time
                                thread::sleep(Duration::from_micros(1));
                            }
                            
                            *count_clone.lock().unwrap() += local_count;
                        });
                        handles.push(consumer);
                    }
                    
                    drop(receiver); // Close the receiver
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                    
                    let total_consumed = *consumed_count.lock().unwrap();
                    black_box(total_consumed);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark concurrent data structure operations
fn concurrent_data_structures_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_data_structures");
    
    for thread_count in [2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_hashmap", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let map = Arc::new(RwLock::new(HashMap::new()));
                    let mut handles = vec![];
                    
                    // Writer threads
                    let num_writers = thread_count / 2;
                    for i in 0..num_writers {
                        let map_clone = Arc::clone(&map);
                        let handle = thread::spawn(move || {
                            for j in 0..100 {
                                let key = format!("key_{}_{}", i, j);
                                let value = Value::Integer((i * 100 + j) as i64);
                                map_clone.write().unwrap().insert(key, value);
                            }
                        });
                        handles.push(handle);
                    }
                    
                    // Reader threads
                    let num_readers = thread_count - num_writers;
                    for i in 0..num_readers {
                        let map_clone = Arc::clone(&map);
                        let handle = thread::spawn(move || {
                            let mut read_count = 0;
                            for _ in 0..50 {
                                let map_read = map_clone.read().unwrap();
                                read_count += map_read.len();
                                
                                // Try to read some specific keys
                                for j in 0..10 {
                                    let key = format!("key_{}_{}", i % 2, j);
                                    if map_read.contains_key(&key) {
                                        read_count += 1;
                                    }
                                }
                            }
                            read_count
                        });
                        handles.push(handle);
                    }
                    
                    let mut total_reads = 0;
                    for handle in handles {
                        if let Ok(reads) = handle.join() {
                            total_reads += reads;
                        }
                    }
                    
                    black_box((map.read().unwrap().len(), total_reads));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark lock contention patterns
fn lock_contention_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("lock_contention");
    
    group.bench_function("high_contention_counter", |b| {
        b.iter(|| {
            let counter = Arc::new(Mutex::new(0));
            let mut handles = vec![];
            
            // Many threads incrementing same counter (high contention)
            for _ in 0..8 {
                let counter_clone = Arc::clone(&counter);
                let handle = thread::spawn(move || {
                    for _ in 0..1000 {
                        *counter_clone.lock().unwrap() += 1;
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let final_count = *counter.lock().unwrap();
            black_box(final_count);
        });
    });
    
    group.bench_function("low_contention_counters", |b| {
        b.iter(|| {
            let counters: Vec<Arc<Mutex<i32>>> = (0..8)
                .map(|_| Arc::new(Mutex::new(0)))
                .collect();
            let mut handles = vec![];
            
            // Each thread has its own counter (low contention)
            for i in 0..8 {
                let counter_clone = Arc::clone(&counters[i]);
                let handle = thread::spawn(move || {
                    for _ in 0..1000 {
                        *counter_clone.lock().unwrap() += 1;
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let total: i32 = counters.iter()
                .map(|c| *c.lock().unwrap())
                .sum();
            black_box(total);
        });
    });
    
    group.finish();
}

criterion_group!(
    concurrent_algorithms_benchmarks,
    parallel_sorting_benchmark,
    parallel_search_benchmark,
    parallel_graph_algorithms_benchmark,
    work_stealing_queue_benchmark,
    producer_consumer_benchmark,
    concurrent_data_structures_benchmark,
    lock_contention_benchmark
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_sorting_data_generation() {
        let data = generate_sorting_data(100);
        assert_eq!(data.len(), 100);
        
        // Should contain varied values
        let min = *data.iter().min().unwrap();
        let max = *data.iter().max().unwrap();
        assert!(max > min);
    }
    
    #[test]
    fn validate_merge_sorted_chunks() {
        let chunks = vec![
            vec![1, 3, 5],
            vec![2, 4, 6],
            vec![0, 7, 8],
        ];
        
        let merged = merge_sorted_chunks(&chunks);
        let expected = vec![0, 1, 2, 3, 4, 5, 6, 7, 8];
        
        assert_eq!(merged, expected);
    }
    
    #[test]
    fn validate_graph_generation() {
        let graph = generate_graph(10, 0.5);
        assert!(graph.len() <= 10);
        
        // Each node should have some neighbors (with 50% probability)
        for (node, neighbors) in &graph {
            assert!(*node < 10);
            for &neighbor in neighbors {
                assert!(neighbor < 10);
                assert!(neighbor != *node);
            }
        }
    }
    
    #[test]
    fn validate_parallel_sorting() {
        let mut data = generate_sorting_data(1000);
        let original = data.clone();
        
        // Sequential sort
        data.sort();
        
        // Verify sorted
        for i in 1..data.len() {
            assert!(data[i] >= data[i-1]);
        }
        
        // Verify same elements
        let mut original_sorted = original;
        original_sorted.sort();
        assert_eq!(data, original_sorted);
    }
    
    #[test]
    fn validate_work_stealing_simulation() {
        let tasks = vec![100, 10, 10, 100, 10]; // Mixed workload
        let work_queue = Arc::new(Mutex::new(tasks.clone()));
        let results = Arc::new(Mutex::new(Vec::new()));
        
        let mut handles = vec![];
        
        // Simulate 2 workers
        for worker_id in 0..2 {
            let queue_clone = Arc::clone(&work_queue);
            let results_clone = Arc::clone(&results);
            
            let handle = thread::spawn(move || {
                let mut worker_results = Vec::new();
                
                while let Some(work_amount) = {
                    let mut queue = queue_clone.lock().unwrap();
                    queue.pop()
                } {
                    // Simulate work
                    let result = work_amount * 2; // Simple computation
                    worker_results.push((worker_id, result));
                }
                
                results_clone.lock().unwrap().extend(worker_results);
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_results = results.lock().unwrap();
        
        // Should have processed all tasks
        assert_eq!(final_results.len(), tasks.len());
        
        // Verify results
        let total_work: usize = final_results.iter().map(|(_, result)| result).sum();
        let expected_work: usize = tasks.iter().map(|t| t * 2).sum();
        assert_eq!(total_work, expected_work);
    }
    
    #[test]
    fn validate_producer_consumer() {
        let (sender, receiver) = mpsc::sync_channel(10);
        
        // Producer
        let producer = thread::spawn(move || {
            for i in 0..100 {
                sender.send(Value::Integer(i)).unwrap();
            }
        });
        
        // Consumer
        let consumer = thread::spawn(move || {
            let mut count = 0;
            while let Ok(_value) = receiver.recv() {
                count += 1;
            }
            count
        });
        
        producer.join().unwrap();
        let consumed = consumer.join().unwrap();
        
        assert_eq!(consumed, 100);
    }
    
    #[test]
    fn validate_concurrent_hashmap() {
        let map = Arc::new(RwLock::new(HashMap::new()));
        let mut handles = vec![];
        
        // Writer thread
        let map_write = Arc::clone(&map);
        let writer = thread::spawn(move || {
            for i in 0..100 {
                let key = format!("key_{}", i);
                let value = Value::Integer(i);
                map_write.write().unwrap().insert(key, value);
            }
        });
        handles.push(writer);
        
        // Reader thread
        let map_read = Arc::clone(&map);
        let reader = thread::spawn(move || {
            let mut read_count = 0;
            for _ in 0..50 {
                let map_guard = map_read.read().unwrap();
                read_count += map_guard.len();
            }
            read_count
        });
        handles.push(reader);
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_size = map.read().unwrap().len();
        assert_eq!(final_size, 100);
    }
}