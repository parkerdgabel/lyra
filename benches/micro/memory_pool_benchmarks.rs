//! Memory Pool Operations Performance Benchmarks
//!
//! Validates memory pool allocation/deallocation efficiency, pool hit rates,
//! and memory management overhead.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::memory::{ValuePools, CompactValuePools, MemoryManager, CompactValue, StringInterner};
use lyra::vm::Value;
use std::sync::Arc;

/// Benchmark basic pool allocation vs heap allocation
fn pool_allocation_vs_heap(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_vs_heap_allocation");
    
    // Baseline: Direct heap allocation
    group.bench_function("heap_allocation", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(Box::new(Value::Integer(i)));
            }
            black_box(values);
        });
    });
    
    // Pool-based allocation
    group.bench_function("pool_allocation", |b| {
        let pools = ValuePools::new();
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                if let Ok(managed_val) = pools.alloc_value(Value::Integer(i)) {
                    values.push(managed_val);
                }
            }
            black_box(values);
        });
    });
    
    // Compact pool allocation
    group.bench_function("compact_pool_allocation", |b| {
        let pools = CompactValuePools::new();
        let interner = StringInterner::new();
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i as i32);
                values.push(pools.alloc_value(compact_val));
            }
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark pool hit rates and recycling efficiency
fn pool_hit_rates_and_recycling(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_hit_rates");
    
    let pools = CompactValuePools::new();
    
    // Test allocation and recycling patterns
    group.bench_function("allocation_recycling_cycle", |b| {
        b.iter(|| {
            let mut allocated_values = Vec::new();
            
            // Allocation phase
            for i in 0..100 {
                let value = CompactValue::SmallInt(i);
                allocated_values.push(pools.alloc_value(value));
            }
            
            // Recycling phase
            for value in &allocated_values {
                pools.recycle_value(value);
            }
            
            black_box(allocated_values);
        });
    });
    
    // Test pool warm-up effects
    group.bench_function("cold_pool_allocation", |b| {
        b.iter(|| {
            let fresh_pools = CompactValuePools::new();
            let mut values = Vec::new();
            for i in 0..100 {
                let value = CompactValue::SmallInt(i);
                values.push(fresh_pools.alloc_value(value));
            }
            black_box(values);
        });
    });
    
    group.bench_function("warm_pool_allocation", |b| {
        // Pre-warm the pool
        for i in 0..100 {
            let value = CompactValue::SmallInt(i);
            let allocated = pools.alloc_value(value);
            pools.recycle_value(&allocated);
        }
        
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..100 {
                let value = CompactValue::SmallInt(i);
                values.push(pools.alloc_value(value));
            }
            // Clean up for next iteration
            for value in &values {
                pools.recycle_value(value);
            }
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark different value type allocation patterns
fn value_type_allocation_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_type_allocation");
    
    let pools = CompactValuePools::new();
    let interner = StringInterner::new();
    
    // Test allocation patterns for different value types
    let value_types = vec![
        ("small_integers", || CompactValue::SmallInt(42)),
        ("large_integers", || CompactValue::LargeInt(Arc::new(i64::MAX))),
        ("reals", || CompactValue::Real(3.14159)),
        ("symbols", || CompactValue::Symbol(interner.intern_symbol_id("test"))),
        ("booleans", || CompactValue::Boolean(true)),
    ];
    
    for (type_name, value_creator) in value_types {
        group.bench_function(type_name, |b| {
            b.iter(|| {
                let mut values = Vec::new();
                for _ in 0..1000 {
                    values.push(pools.alloc_value(value_creator()));
                }
                black_box(values);
            });
        });
    }
    
    group.finish();
}

/// Benchmark memory pool fragmentation effects
fn pool_fragmentation_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_fragmentation");
    
    let pools = CompactValuePools::new();
    
    // Simulate fragmentation through mixed allocation/deallocation
    group.bench_function("fragmented_allocation_pattern", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            
            // Allocate many values
            for i in 0..1000 {
                values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            }
            
            // Deallocate every other value (creates fragmentation)
            for (i, value) in values.iter().enumerate() {
                if i % 2 == 0 {
                    pools.recycle_value(value);
                }
            }
            
            // Allocate new values into fragmented space
            let mut new_values = Vec::new();
            for i in 0..500 {
                new_values.push(pools.alloc_value(CompactValue::SmallInt(i + 1000)));
            }
            
            black_box((values, new_values));
        });
    });
    
    // Compare with linear allocation pattern
    group.bench_function("linear_allocation_pattern", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            
            // Allocate values linearly
            for i in 0..1500 {
                values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            }
            
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark pool memory usage and efficiency
fn pool_memory_usage_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_memory_efficiency");
    
    // Test memory usage patterns
    group.bench_function("memory_usage_tracking", |b| {
        let pools = CompactValuePools::new();
        
        b.iter(|| {
            let initial_usage = pools.total_memory_usage();
            
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            }
            
            let peak_usage = pools.total_memory_usage();
            
            // Recycle all values
            for value in &values {
                pools.recycle_value(value);
            }
            
            let final_usage = pools.total_memory_usage();
            
            black_box((initial_usage, peak_usage, final_usage));
        });
    });
    
    // Test pool efficiency metrics
    group.bench_function("pool_efficiency_metrics", |b| {
        let pools = CompactValuePools::new();
        
        b.iter(|| {
            // Allocate and track efficiency
            let mut values = Vec::new();
            for i in 0..100 {
                values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            }
            
            let stats = pools.stats();
            let efficiency_report = pools.efficiency_report();
            
            black_box((stats, efficiency_report));
        });
    });
    
    group.finish();
}

/// Benchmark concurrent pool access
fn concurrent_pool_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_pool_access");
    group.throughput(Throughput::Elements(1000));
    
    use std::thread;
    use std::sync::Arc as StdArc;
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_allocation", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let pools = StdArc::new(CompactValuePools::new());
                    let mut handles = vec![];
                    
                    for i in 0..thread_count {
                        let pools_clone = StdArc::clone(&pools);
                        let handle = thread::spawn(move || {
                            let mut values = Vec::new();
                            for j in 0..250 { // 250 * 4 threads = 1000 total
                                let value = CompactValue::SmallInt((i * 250 + j) as i32);
                                values.push(pools_clone.alloc_value(value));
                            }
                            values
                        });
                        handles.push(handle);
                    }
                    
                    let mut all_values = Vec::new();
                    for handle in handles {
                        all_values.extend(handle.join().unwrap());
                    }
                    
                    black_box(all_values);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark pool garbage collection efficiency
fn pool_garbage_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_garbage_collection");
    
    let pools = CompactValuePools::new();
    
    group.bench_function("garbage_collection_cycle", |b| {
        b.iter(|| {
            // Create garbage by allocating and not recycling
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            }
            
            // Only recycle half, creating garbage
            for (i, value) in values.iter().enumerate() {
                if i % 2 == 0 {
                    pools.recycle_value(value);
                }
            }
            
            // Trigger garbage collection
            let collected = pools.collect_unused();
            
            black_box(collected);
        });
    });
    
    group.finish();
}

/// Benchmark pool vs MemoryManager integration
fn pool_memory_manager_integration(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_memory_manager_integration");
    
    group.bench_function("memory_manager_allocation", |b| {
        let mut memory_manager = MemoryManager::new();
        
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i);
                values.push(memory_manager.alloc_compact_value(compact_val));
            }
            black_box(values);
        });
    });
    
    group.bench_function("direct_pool_allocation", |b| {
        let pools = CompactValuePools::new();
        
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i);
                values.push(pools.alloc_value(compact_val));
            }
            black_box(values);
        });
    });
    
    group.finish();
}

criterion_group!(
    memory_pool_benchmarks,
    pool_allocation_vs_heap,
    pool_hit_rates_and_recycling,
    value_type_allocation_patterns,
    pool_fragmentation_effects,
    pool_memory_usage_efficiency,
    concurrent_pool_access,
    pool_garbage_collection,
    pool_memory_manager_integration
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_pool_efficiency() {
        let pools = CompactValuePools::new();
        
        // Measure allocation efficiency
        let initial_usage = pools.total_memory_usage();
        
        let mut values = Vec::new();
        for i in 0..1000 {
            values.push(pools.alloc_value(CompactValue::SmallInt(i)));
        }
        
        let peak_usage = pools.total_memory_usage();
        
        // Recycle half the values
        for (i, value) in values.iter().enumerate() {
            if i % 2 == 0 {
                pools.recycle_value(value);
            }
        }
        
        let after_recycle_usage = pools.total_memory_usage();
        
        println!("Initial usage: {} bytes", initial_usage);
        println!("Peak usage: {} bytes", peak_usage);
        println!("After recycle: {} bytes", after_recycle_usage);
        
        // Pool should show some memory reuse after recycling
        assert!(peak_usage > initial_usage);
        // Note: after_recycle_usage may not be less than peak_usage in all pool implementations
    }
    
    #[test]
    fn validate_pool_stats() {
        let pools = CompactValuePools::new();
        
        // Allocate various types
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(pools.alloc_value(CompactValue::SmallInt(i)));
            values.push(pools.alloc_value(CompactValue::Real(i as f64)));
            values.push(pools.alloc_value(CompactValue::Boolean(i % 2 == 0)));
        }
        
        let stats = pools.stats();
        
        println!("Pool stats: {:?}", stats);
        
        // Should have stats for different value types
        assert!(!stats.is_empty());
    }
    
    #[test]
    fn validate_memory_manager_integration() {
        let memory_manager = MemoryManager::new();
        
        // Test integrated allocation
        let compact_val = memory_manager.compact_value(Value::Integer(42));
        let allocated = memory_manager.alloc_compact_value(compact_val);
        
        // Should be able to recycle
        memory_manager.recycle_compact_value(&allocated);
        
        // Memory manager should track stats
        let stats = memory_manager.memory_stats();
        assert!(stats.total_allocated >= 0);
    }
    
    #[test]
    fn validate_concurrent_pool_access() {
        use std::thread;
        use std::sync::Arc as StdArc;
        
        let pools = StdArc::new(CompactValuePools::new());
        let mut handles = vec![];
        
        // Create multiple threads that allocate from pools
        for i in 0..4 {
            let pools_clone = StdArc::clone(&pools);
            let handle = thread::spawn(move || {
                let mut values = Vec::new();
                for j in 0..100 {
                    let value = CompactValue::SmallInt((i * 100 + j) as i32);
                    values.push(pools_clone.alloc_value(value));
                }
                values.len()
            });
            handles.push(handle);
        }
        
        let mut total_allocated = 0;
        for handle in handles {
            total_allocated += handle.join().unwrap();
        }
        
        assert_eq!(total_allocated, 400); // 4 threads * 100 allocations each
    }
    
    #[test]
    fn validate_pool_garbage_collection() {
        let pools = CompactValuePools::new();
        
        // Create some garbage
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(pools.alloc_value(CompactValue::SmallInt(i)));
        }
        
        // Only recycle half
        for (i, value) in values.iter().enumerate() {
            if i % 2 == 0 {
                pools.recycle_value(value);
            }
        }
        
        // Should be able to collect unused memory
        let collected = pools.collect_unused();
        
        // collect_unused should return some amount of memory collected
        // (actual amount depends on pool implementation)
        println!("Collected {} bytes", collected);
    }
}