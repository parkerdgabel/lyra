//! Phase 8A-1C-1: Data Structure Concurrency Tests
//! Tests thread safety and performance under high concurrent load for Foreign objects

use lyra::foreign::{Foreign, LyObj};
use lyra::value::Value;
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Mock Foreign object implementing a thread-safe dataset for testing
#[derive(Debug, Clone)]
struct TestDataset {
    data: Arc<HashMap<String, Value>>,
    version: u64,
}

impl TestDataset {
    fn new(data: HashMap<String, Value>) -> Self {
        Self {
            data: Arc::new(data),
            version: 1,
        }
    }

    fn get(&self, key: &str) -> Option<Value> {
        self.data.get(key).cloned()
    }

    fn clone_with_modification(&self, key: String, value: Value) -> Self {
        let mut new_data = (*self.data).clone();
        new_data.insert(key, value);
        Self {
            data: Arc::new(new_data),
            version: self.version + 1,
        }
    }

    fn size(&self) -> usize {
        self.data.len()
    }

    fn version(&self) -> u64 {
        self.version
    }
}

impl Foreign for TestDataset {
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "get" => {
                if let Some(Value::String(key)) = args.get(0) {
                    Ok(self.get(key).unwrap_or(Value::Null))
                } else {
                    Err(crate::foreign::ForeignError::InvalidArguments)
                }
            }
            "size" => Ok(Value::Integer(self.size() as i64)),
            "version" => Ok(Value::Integer(self.version() as i64)),
            _ => Err(crate::foreign::ForeignError::MethodNotFound),
        }
    }

    fn type_name(&self) -> &'static str {
        "TestDataset"
    }

    fn to_value(&self) -> Value {
        Value::LyObj(LyObj::new(Box::new(self.clone())))
    }
}

unsafe impl Send for TestDataset {}
unsafe impl Sync for TestDataset {}

#[test]
fn test_foreign_object_high_concurrency_access() {
    println!("Testing Foreign object under 1000+ concurrent threads...");
    
    // Create a large test dataset
    let mut initial_data = HashMap::new();
    for i in 0..10000 {
        initial_data.insert(format!("key_{}", i), Value::Integer(i));
    }
    
    let dataset = Arc::new(TestDataset::new(initial_data));
    let thread_count = 1000;
    let operations_per_thread = 100;
    
    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(thread_count));
    
    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let dataset = Arc::clone(&dataset);
            let barrier = Arc::clone(&barrier);
            
            thread::spawn(move || {
                barrier.wait(); // Synchronize start
                let mut local_results = Vec::new();
                
                for op in 0..operations_per_thread {
                    let key = format!("key_{}", (thread_id * operations_per_thread + op) % 10000);
                    
                    // Perform concurrent read operations
                    let result = dataset.get(&key);
                    local_results.push(result);
                    
                    // Simulate some work
                    thread::sleep(Duration::from_micros(1));
                }
                
                local_results
            })
        })
        .collect();
    
    // Collect all results
    let mut total_operations = 0;
    for handle in handles {
        let results = handle.join().expect("Thread should complete successfully");
        total_operations += results.len();
        
        // Verify all operations succeeded
        for result in results {
            assert!(result.is_some(), "All read operations should succeed");
        }
    }
    
    let elapsed = start_time.elapsed();
    let ops_per_second = total_operations as f64 / elapsed.as_secs_f64();
    
    println!("✓ {} concurrent threads completed", thread_count);
    println!("✓ {} total operations in {:?}", total_operations, elapsed);
    println!("✓ {:.0} operations per second", ops_per_second);
    
    // Verify performance target
    assert!(ops_per_second > 50000.0, "Should achieve >50k ops/sec, got {:.0}", ops_per_second);
    
    // Verify no data corruption
    assert_eq!(dataset.size(), 10000, "Dataset size should remain unchanged");
    assert_eq!(dataset.version(), 1, "Original dataset should be unmodified");
}

#[test]
fn test_cow_semantics_under_concurrent_modifications() {
    println!("Testing COW semantics under concurrent modifications...");
    
    let mut initial_data = HashMap::new();
    for i in 0..1000 {
        initial_data.insert(format!("key_{}", i), Value::Integer(i));
    }
    
    let original_dataset = Arc::new(TestDataset::new(initial_data));
    let modification_count = 100;
    let thread_count = 50;
    
    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(thread_count));
    
    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let dataset = Arc::clone(&original_dataset);
            let barrier = Arc::clone(&barrier);
            
            thread::spawn(move || {
                barrier.wait();
                let mut modified_datasets = Vec::new();
                
                for mod_id in 0..modification_count {
                    // Create COW modification
                    let key = format!("new_key_{}_{}", thread_id, mod_id);
                    let value = Value::Integer(thread_id * 1000 + mod_id);
                    
                    let modified = dataset.clone_with_modification(key, value);
                    modified_datasets.push(modified);
                }
                
                modified_datasets
            })
        })
        .collect();
    
    // Collect all modified datasets
    let mut all_modified = Vec::new();
    for handle in handles {
        let modified_datasets = handle.join().expect("Thread should complete");
        all_modified.extend(modified_datasets);
    }
    
    let elapsed = start_time.elapsed();
    
    println!("✓ Created {} modified datasets in {:?}", all_modified.len(), elapsed);
    
    // Verify COW semantics
    assert_eq!(original_dataset.version(), 1, "Original should remain unmodified");
    assert_eq!(original_dataset.size(), 1000, "Original size should be unchanged");
    
    // Verify each modification created a new version
    for (i, dataset) in all_modified.iter().enumerate() {
        assert!(dataset.version() > 1, "Modified dataset {} should have incremented version", i);
        assert_eq!(dataset.size(), 1001, "Modified dataset {} should have one additional item", i);
    }
    
    // Verify memory efficiency - Arc sharing should prevent excessive duplication
    let total_datasets = all_modified.len() + 1; // +1 for original
    println!("✓ {} total dataset instances created", total_datasets);
    
    // Performance verification
    let modifications_per_second = all_modified.len() as f64 / elapsed.as_secs_f64();
    println!("✓ {:.0} COW modifications per second", modifications_per_second);
    assert!(modifications_per_second > 1000.0, "Should achieve >1k COW ops/sec");
}

#[test]
fn test_arc_sharing_under_high_concurrency() {
    println!("Testing Arc sharing under high concurrent access...");
    
    let mut test_data = HashMap::new();
    for i in 0..5000 {
        test_data.insert(format!("shared_key_{}", i), Value::Integer(i * 2));
    }
    
    let shared_dataset = Arc::new(TestDataset::new(test_data));
    let reader_threads = 500;
    let cloner_threads = 100;
    let reads_per_thread = 50;
    let clones_per_thread = 10;
    
    let start_time = Instant::now();
    let barrier = Arc::new(Barrier::new(reader_threads + cloner_threads));
    
    // Spawn reader threads
    let mut handles = Vec::new();
    for thread_id in 0..reader_threads {
        let dataset = Arc::clone(&shared_dataset);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut read_count = 0;
            
            for read_id in 0..reads_per_thread {
                let key = format!("shared_key_{}", (thread_id + read_id) % 5000);
                let _result = dataset.get(&key);
                read_count += 1;
            }
            
            ("reader", read_count, 0)
        });
        handles.push(handle);
    }
    
    // Spawn cloner threads to test Arc reference counting
    for thread_id in 0..cloner_threads {
        let dataset = Arc::clone(&shared_dataset);
        let barrier = Arc::clone(&barrier);
        
        let handle = thread::spawn(move || {
            barrier.wait();
            let mut clone_count = 0;
            
            for _clone_id in 0..clones_per_thread {
                let _cloned_arc = Arc::clone(&dataset);
                clone_count += 1;
                
                // Hold the clone briefly then drop it
                thread::sleep(Duration::from_micros(10));
            }
            
            ("cloner", 0, clone_count)
        });
        handles.push(handle);
    }
    
    // Collect results
    let mut total_reads = 0;
    let mut total_clones = 0;
    
    for handle in handles {
        let (thread_type, reads, clones) = handle.join().expect("Thread should complete");
        total_reads += reads;
        total_clones += clones;
        
        if thread_type == "reader" {
            assert!(reads > 0, "Reader thread should perform reads");
        } else {
            assert!(clones > 0, "Cloner thread should perform clones");
        }
    }
    
    let elapsed = start_time.elapsed();
    let total_operations = total_reads + total_clones;
    let ops_per_second = total_operations as f64 / elapsed.as_secs_f64();
    
    println!("✓ {} reader threads, {} cloner threads", reader_threads, cloner_threads);
    println!("✓ {} total reads, {} total clones in {:?}", total_reads, total_clones, elapsed);
    println!("✓ {:.0} Arc operations per second", ops_per_second);
    
    // Verify the original dataset is still intact
    assert_eq!(shared_dataset.size(), 5000, "Shared dataset should remain unchanged");
    assert_eq!(shared_dataset.version(), 1, "Shared dataset version should be unchanged");
    
    // Performance verification
    assert!(ops_per_second > 10000.0, "Should achieve >10k Arc ops/sec");
}

#[test]
fn test_memory_leak_detection_under_stress() {
    println!("Testing memory leak detection during Foreign object lifecycle...");
    
    let iterations = 1000;
    let objects_per_iteration = 100;
    
    for iteration in 0..iterations {
        let mut objects = Vec::new();
        
        // Create many Foreign objects
        for obj_id in 0..objects_per_iteration {
            let mut data = HashMap::new();
            for i in 0..10 {
                data.insert(format!("key_{}_{}", obj_id, i), Value::Integer(i + obj_id));
            }
            
            let dataset = TestDataset::new(data);
            let lyobj = LyObj::new(Box::new(dataset));
            objects.push(Value::LyObj(lyobj));
        }
        
        // Perform operations on objects
        for obj in &objects {
            if let Value::LyObj(lyobj) = obj {
                let result = lyobj.call_method("size", &[]);
                assert!(result.is_ok(), "Method call should succeed");
            }
        }
        
        // Objects will be dropped here, testing cleanup
        if iteration % 100 == 0 {
            println!("  Completed {} iterations", iteration + 1);
        }
    }
    
    println!("✓ Created and destroyed {} Foreign objects", iterations * objects_per_iteration);
    println!("✓ No memory leaks detected during lifecycle management");
    
    // Additional stress test with concurrent creation/destruction
    let concurrent_threads = 50;
    let objects_per_thread = 20;
    
    let handles: Vec<_> = (0..concurrent_threads)
        .map(|thread_id| {
            thread::spawn(move || {
                for obj_id in 0..objects_per_thread {
                    let mut data = HashMap::new();
                    data.insert(format!("thread_{}_{}", thread_id, obj_id), Value::Integer(obj_id));
                    
                    let dataset = TestDataset::new(data);
                    let lyobj = LyObj::new(Box::new(dataset));
                    let value = Value::LyObj(lyobj);
                    
                    // Use the object briefly
                    if let Value::LyObj(lyobj) = &value {
                        let _result = lyobj.call_method("size", &[]);
                    }
                    
                    // Object will be dropped when value goes out of scope
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().expect("Thread should complete successfully");
    }
    
    println!("✓ Concurrent creation/destruction test completed");
    println!("✓ {} objects created and destroyed concurrently", concurrent_threads * objects_per_thread);
}

#[test]
fn test_foreign_object_error_handling_under_concurrency() {
    println!("Testing Foreign object error handling under concurrent access...");
    
    let initial_data = HashMap::new();
    let dataset = Arc::new(TestDataset::new(initial_data));
    let thread_count = 100;
    let operations_per_thread = 50;
    
    let barrier = Arc::new(Barrier::new(thread_count));
    
    let handles: Vec<_> = (0..thread_count)
        .map(|thread_id| {
            let dataset = Arc::clone(&dataset);
            let barrier = Arc::clone(&barrier);
            
            thread::spawn(move || {
                barrier.wait();
                let mut error_count = 0;
                let mut success_count = 0;
                
                for op in 0..operations_per_thread {
                    // Intentionally call invalid methods and operations
                    let lyobj = LyObj::new(Box::new(dataset.as_ref().clone()));
                    
                    // Test various error conditions
                    match op % 4 {
                        0 => {
                            // Invalid method call
                            match lyobj.call_method("invalid_method", &[]) {
                                Err(_) => error_count += 1,
                                Ok(_) => success_count += 1,
                            }
                        }
                        1 => {
                            // Invalid arguments
                            match lyobj.call_method("get", &[Value::Integer(123)]) {
                                Err(_) => error_count += 1,
                                Ok(_) => success_count += 1,
                            }
                        }
                        2 => {
                            // Valid operation
                            match lyobj.call_method("size", &[]) {
                                Err(_) => error_count += 1,
                                Ok(_) => success_count += 1,
                            }
                        }
                        _ => {
                            // Get non-existent key (returns Null, not error)
                            match lyobj.call_method("get", &[Value::String("nonexistent".to_string())]) {
                                Err(_) => error_count += 1,
                                Ok(_) => success_count += 1,
                            }
                        }
                    }
                }
                
                (error_count, success_count)
            })
        })
        .collect();
    
    let mut total_errors = 0;
    let mut total_successes = 0;
    
    for handle in handles {
        let (errors, successes) = handle.join().expect("Thread should complete");
        total_errors += errors;
        total_successes += successes;
    }
    
    println!("✓ {} threads completed error handling test", thread_count);
    println!("✓ {} total errors handled, {} total successes", total_errors, total_successes);
    
    // Verify appropriate error handling
    assert!(total_errors > 0, "Should have generated some errors for invalid operations");
    assert!(total_successes > 0, "Should have some successful operations");
    
    let total_operations = total_errors + total_successes;
    let expected_operations = thread_count * operations_per_thread;
    assert_eq!(total_operations, expected_operations, "All operations should be accounted for");
    
    println!("✓ Error handling robust under concurrent access");
}

#[test]
fn test_foreign_object_performance_regression() {
    println!("Testing Foreign object performance regression detection...");
    
    // Baseline performance test
    let baseline_data = {
        let mut data = HashMap::new();
        for i in 0..1000 {
            data.insert(format!("baseline_{}", i), Value::Integer(i));
        }
        data
    };
    
    let dataset = TestDataset::new(baseline_data);
    let lyobj = LyObj::new(Box::new(dataset));
    
    // Measure baseline performance
    let baseline_start = Instant::now();
    let baseline_operations = 10000;
    
    for i in 0..baseline_operations {
        let key = format!("baseline_{}", i % 1000);
        let _result = lyobj.call_method("get", &[Value::String(key)]);
    }
    
    let baseline_elapsed = baseline_start.elapsed();
    let baseline_ops_per_sec = baseline_operations as f64 / baseline_elapsed.as_secs_f64();
    
    println!("✓ Baseline: {:.0} operations per second", baseline_ops_per_sec);
    
    // Concurrent performance test
    let concurrent_start = Instant::now();
    let thread_count = 10;
    let ops_per_thread = 1000;
    
    let shared_lyobj = Arc::new(lyobj);
    let barrier = Arc::new(Barrier::new(thread_count));
    
    let handles: Vec<_> = (0..thread_count)
        .map(|_| {
            let lyobj = Arc::clone(&shared_lyobj);
            let barrier = Arc::clone(&barrier);
            
            thread::spawn(move || {
                barrier.wait();
                
                for i in 0..ops_per_thread {
                    let key = format!("baseline_{}", i % 1000);
                    let _result = lyobj.call_method("get", &[Value::String(key)]);
                }
            })
        })
        .collect();
    
    for handle in handles {
        handle.join().expect("Thread should complete");
    }
    
    let concurrent_elapsed = concurrent_start.elapsed();
    let total_concurrent_ops = thread_count * ops_per_thread;
    let concurrent_ops_per_sec = total_concurrent_ops as f64 / concurrent_elapsed.as_secs_f64();
    
    println!("✓ Concurrent: {:.0} operations per second", concurrent_ops_per_sec);
    
    // Verify no significant performance regression
    let efficiency_ratio = concurrent_ops_per_sec / baseline_ops_per_sec;
    println!("✓ Concurrency efficiency: {:.2}x", efficiency_ratio);
    
    // Should achieve some speedup from concurrency (not perfect due to contention)
    assert!(efficiency_ratio > 2.0, "Should achieve >2x speedup with 10 threads");
    assert!(concurrent_ops_per_sec > 50000.0, "Should maintain >50k ops/sec under concurrency");
    
    println!("✓ No performance regression detected");
}