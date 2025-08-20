//! Memory Usage and Allocation Profiling Benchmarks
//!
//! Comprehensive memory profiling benchmarks for allocation patterns, pool efficiency,
//! cache performance, and memory fragmentation analysis.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use lyra::{
    vm::Value,
    memory::{MemoryManager, CompactValue, StringInterner, ValuePools, CompactValuePools},
};
use std::sync::Arc;
use std::collections::HashMap;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Custom allocator for tracking allocations
struct TrackingAllocator {
    inner: System,
    allocated: AtomicUsize,
    deallocated: AtomicUsize,
    allocation_count: AtomicUsize,
    deallocation_count: AtomicUsize,
}

impl TrackingAllocator {
    const fn new() -> Self {
        Self {
            inner: System,
            allocated: AtomicUsize::new(0),
            deallocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
            deallocation_count: AtomicUsize::new(0),
        }
    }
    
    fn reset_stats(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.deallocated.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
        self.deallocation_count.store(0, Ordering::Relaxed);
    }
    
    fn get_stats(&self) -> (usize, usize, usize, usize) {
        (
            self.allocated.load(Ordering::Relaxed),
            self.deallocated.load(Ordering::Relaxed),
            self.allocation_count.load(Ordering::Relaxed),
            self.deallocation_count.load(Ordering::Relaxed),
        )
    }
    
    fn net_allocated(&self) -> usize {
        let (allocated, deallocated, _, _) = self.get_stats();
        allocated.saturating_sub(deallocated)
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = self.inner.alloc(layout);
        if !ptr.is_null() {
            self.allocated.fetch_add(layout.size(), Ordering::Relaxed);
            self.allocation_count.fetch_add(1, Ordering::Relaxed);
        }
        ptr
    }
    
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.inner.dealloc(ptr, layout);
        self.deallocated.fetch_add(layout.size(), Ordering::Relaxed);
        self.deallocation_count.fetch_add(1, Ordering::Relaxed);
    }
}

// Note: We can't actually replace the global allocator in benchmarks easily,
// so we'll simulate memory tracking through other means

/// Benchmark allocation patterns with different strategies
fn allocation_patterns_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_patterns");
    group.throughput(Throughput::Elements(10000));
    
    // Pattern 1: Many small allocations
    group.bench_function("many_small_allocations", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..10000 {
                values.push(black_box(Value::Integer(i)));
            }
            black_box(values);
        });
    });
    
    // Pattern 2: Few large allocations
    group.bench_function("few_large_allocations", |b| {
        b.iter(|| {
            let mut large_lists = Vec::new();
            for _ in 0..100 {
                let large_list: Vec<Value> = (0..100)
                    .map(|i| Value::Integer(i))
                    .collect();
                large_lists.push(black_box(Value::List(large_list)));
            }
            black_box(large_lists);
        });
    });
    
    // Pattern 3: Mixed allocation sizes
    group.bench_function("mixed_allocation_sizes", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                match i % 4 {
                    0 => values.push(Value::Integer(i)),
                    1 => values.push(Value::String(format!("string_{}", i))),
                    2 => {
                        let list: Vec<Value> = (0..i%20).map(|j| Value::Integer(j)).collect();
                        values.push(Value::List(list));
                    }
                    _ => values.push(Value::Real(i as f64)),
                }
            }
            black_box(values);
        });
    });
    
    // Pattern 4: Frequent allocation/deallocation cycles
    group.bench_function("allocation_deallocation_cycles", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let temp_values: Vec<Value> = (0..10)
                    .map(|i| Value::Integer(i))
                    .collect();
                black_box(temp_values); // Goes out of scope and deallocates
            }
        });
    });
    
    group.finish();
}

/// Benchmark pool efficiency and hit rates
fn pool_efficiency_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_efficiency");
    
    // Pool vs heap allocation comparison
    group.bench_function("heap_allocation", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(black_box(Box::new(Value::Integer(i))));
            }
            black_box(values);
        });
    });
    
    group.bench_function("pool_allocation", |b| {
        let pools = ValuePools::new();
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                if let Ok(managed_val) = pools.alloc_value(Value::Integer(i)) {
                    values.push(black_box(managed_val));
                }
            }
            black_box(values);
        });
    });
    
    group.bench_function("compact_pool_allocation", |b| {
        let pools = CompactValuePools::new();
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i as i32);
                values.push(black_box(pools.alloc_value(compact_val)));
            }
            black_box(values);
        });
    });
    
    // Test pool hit rates with recycling
    group.bench_function("pool_with_recycling", |b| {
        let pools = CompactValuePools::new();
        
        // Pre-warm the pool
        let mut initial_values = Vec::new();
        for i in 0..100 {
            let val = CompactValue::SmallInt(i);
            initial_values.push(pools.alloc_value(val));
        }
        
        // Recycle half to create available pool entries
        for (i, value) in initial_values.iter().enumerate() {
            if i % 2 == 0 {
                pools.recycle_value(value);
            }
        }
        
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i as i32);
                let allocated = pools.alloc_value(compact_val);
                values.push(allocated);
                
                // Recycle some values to maintain pool
                if i % 10 == 0 && !values.is_empty() {
                    let idx = i % values.len();
                    pools.recycle_value(&values[idx]);
                }
            }
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark cache performance with different access patterns
fn cache_performance_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance");
    
    let data_size = 10000;
    
    // Sequential access (cache-friendly)
    group.bench_function("sequential_access", |b| {
        let data: Vec<Value> = (0..data_size)
            .map(|i| Value::Integer(i))
            .collect();
        
        b.iter(|| {
            let mut sum = 0i64;
            for value in &data {
                if let Value::Integer(i) = value {
                    sum += i;
                }
            }
            black_box(sum);
        });
    });
    
    // Random access (cache-unfriendly)
    group.bench_function("random_access", |b| {
        let data: Vec<Value> = (0..data_size)
            .map(|i| Value::Integer(i))
            .collect();
        
        // Generate pseudo-random indices
        let indices: Vec<usize> = (0..data_size)
            .map(|i| (i * 7919) % data_size)
            .collect();
        
        b.iter(|| {
            let mut sum = 0i64;
            for &idx in &indices {
                if let Value::Integer(i) = &data[idx] {
                    sum += i;
                }
            }
            black_box(sum);
        });
    });
    
    // Strided access (partially cache-friendly)
    group.bench_function("strided_access", |b| {
        let data: Vec<Value> = (0..data_size)
            .map(|i| Value::Integer(i))
            .collect();
        
        b.iter(|| {
            let mut sum = 0i64;
            let stride = 8;
            let mut i = 0;
            while i < data.len() {
                if let Value::Integer(val) = &data[i] {
                    sum += val;
                }
                i += stride;
            }
            black_box(sum);
        });
    });
    
    // Compact values cache performance
    group.bench_function("compact_sequential_access", |b| {
        let interner = StringInterner::new();
        let data: Vec<CompactValue> = (0..data_size)
            .map(|i| CompactValue::SmallInt(i as i32))
            .collect();
        
        b.iter(|| {
            let mut sum = 0i64;
            for value in &data {
                if let CompactValue::SmallInt(i) = value {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark memory fragmentation effects
fn memory_fragmentation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_fragmentation");
    
    // Fragmentation through mixed allocation/deallocation
    group.bench_function("fragmentation_simulation", |b| {
        b.iter(|| {
            let mut allocations = Vec::new();
            
            // Phase 1: Allocate many objects
            for i in 0..1000 {
                allocations.push(black_box(Value::Integer(i)));
                allocations.push(black_box(Value::String(format!("str_{}", i))));
            }
            
            // Phase 2: Deallocate every other object (creates fragmentation)
            let mut fragmented = Vec::new();
            for (i, value) in allocations.into_iter().enumerate() {
                if i % 2 == 0 {
                    // Keep this allocation
                    fragmented.push(value);
                }
                // Odd indices are dropped, creating holes
            }
            
            // Phase 3: Allocate new objects into fragmented space
            for i in 0..500 {
                fragmented.push(black_box(Value::Real(i as f64)));
            }
            
            black_box(fragmented);
        });
    });
    
    // Pool-based allocation reduces fragmentation
    group.bench_function("pool_fragmentation_resistance", |b| {
        let pools = CompactValuePools::new();
        
        b.iter(|| {
            let mut allocations = Vec::new();
            
            // Phase 1: Allocate many objects
            for i in 0..1000 {
                let val1 = CompactValue::SmallInt(i as i32);
                let val2 = CompactValue::Real(i as f64);
                allocations.push(pools.alloc_value(val1));
                allocations.push(pools.alloc_value(val2));
            }
            
            // Phase 2: Recycle every other object
            for (i, value) in allocations.iter().enumerate() {
                if i % 2 == 1 {
                    pools.recycle_value(value);
                }
            }
            
            // Phase 3: Allocate new objects (should reuse recycled memory)
            let mut new_allocations = Vec::new();
            for i in 0..500 {
                let val = CompactValue::SmallInt((i + 1000) as i32);
                new_allocations.push(pools.alloc_value(val));
            }
            
            black_box((allocations, new_allocations));
        });
    });
    
    group.finish();
}

/// Benchmark memory usage with different data structures
fn memory_usage_comparison_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_comparison");
    
    // Compare memory overhead of different approaches
    for size in [100, 1000, 10000].iter() {
        group.bench_with_input(
            BenchmarkId::new("vec_of_values", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let values: Vec<Value> = (0..size)
                        .map(|i| Value::Integer(i))
                        .collect();
                    black_box(values);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("vec_of_compact_values", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let values: Vec<CompactValue> = (0..size)
                        .map(|i| CompactValue::SmallInt(i as i32))
                        .collect();
                    black_box(values);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("memory_managed_values", size),
            size,
            |b, &size| {
                let mut memory_manager = MemoryManager::new();
                b.iter(|| {
                    let mut values = Vec::new();
                    for i in 0..size {
                        let compact_val = CompactValue::SmallInt(i as i32);
                        values.push(memory_manager.alloc_compact_value(compact_val));
                    }
                    black_box(values);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark string interning memory efficiency
fn string_interning_memory_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning_memory");
    
    let test_strings: Vec<String> = (0..1000)
        .map(|i| format!("test_string_{:04}", i % 100)) // Many duplicates
        .collect();
    
    group.bench_function("without_interning", |b| {
        b.iter(|| {
            let mut string_values = Vec::new();
            for s in &test_strings {
                string_values.push(black_box(Value::String(s.clone())));
            }
            black_box(string_values);
        });
    });
    
    group.bench_function("with_string_interning", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let mut interned_values = Vec::new();
            for s in &test_strings {
                let symbol_id = interner.intern_symbol_id(s);
                interned_values.push(black_box(CompactValue::Symbol(symbol_id)));
            }
            black_box(interned_values);
        });
    });
    
    // Memory usage over time
    group.bench_function("memory_growth_without_interning", |b| {
        b.iter(|| {
            let mut all_strings = Vec::new();
            for batch in 0..10 {
                for i in 0..100 {
                    let s = format!("batch_{}_string_{}", batch, i % 20);
                    all_strings.push(black_box(Value::String(s)));
                }
            }
            black_box(all_strings);
        });
    });
    
    group.bench_function("memory_growth_with_interning", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let mut all_symbols = Vec::new();
            for batch in 0..10 {
                for i in 0..100 {
                    let s = format!("batch_{}_string_{}", batch, i % 20);
                    let symbol_id = interner.intern_symbol_id(&s);
                    all_symbols.push(black_box(CompactValue::Symbol(symbol_id)));
                }
            }
            black_box(all_symbols);
        });
    });
    
    group.finish();
}

/// Benchmark garbage collection and cleanup performance
fn garbage_collection_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("garbage_collection");
    
    group.bench_function("pool_garbage_collection", |b| {
        let pools = CompactValuePools::new();
        
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
            let collected = black_box(pools.collect_unused());
            black_box(collected);
        });
    });
    
    group.bench_function("memory_manager_cleanup", |b| {
        b.iter(|| {
            let mut memory_manager = MemoryManager::new();
            
            // Allocate many values
            let mut values = Vec::new();
            for i in 0..1000 {
                let compact_val = CompactValue::SmallInt(i);
                values.push(memory_manager.alloc_compact_value(compact_val));
            }
            
            // Recycle some values
            for (i, value) in values.iter().enumerate() {
                if i % 3 == 0 {
                    memory_manager.recycle_compact_value(value);
                }
            }
            
            // Trigger garbage collection
            let collected = black_box(memory_manager.collect_garbage());
            black_box(collected);
        });
    });
    
    group.finish();
}

/// Benchmark memory allocation under concurrent access
fn concurrent_memory_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_memory");
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_pool_allocation", thread_count),
            thread_count,
            |b, &thread_count| {
                use std::thread;
                use std::sync::Arc as StdArc;
                
                b.iter(|| {
                    let pools = StdArc::new(CompactValuePools::new());
                    let mut handles = vec![];
                    
                    for i in 0..thread_count {
                        let pools_clone = StdArc::clone(&pools);
                        let handle = thread::spawn(move || {
                            let mut values = Vec::new();
                            for j in 0..250 { // 250 * threads = 1000+ total allocations
                                let val = CompactValue::SmallInt((i * 250 + j) as i32);
                                values.push(pools_clone.alloc_value(val));
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

criterion_group!(
    memory_profiling_benchmarks,
    allocation_patterns_benchmark,
    pool_efficiency_benchmark,
    cache_performance_benchmark,
    memory_fragmentation_benchmark,
    memory_usage_comparison_benchmark,
    string_interning_memory_benchmark,
    garbage_collection_benchmark,
    concurrent_memory_benchmark
);

criterion_main!(memory_profiling_benchmarks);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_tracking_allocator() {
        let allocator = TrackingAllocator::new();
        allocator.reset_stats();
        
        let initial_stats = allocator.get_stats();
        assert_eq!(initial_stats, (0, 0, 0, 0));
        
        // Note: We can't easily test the allocator without replacing the global allocator
        // This is just to verify the struct compiles and has the right interface
    }
    
    #[test]
    fn test_allocation_patterns() {
        // Test basic allocation patterns work
        let mut values = Vec::new();
        for i in 0..100 {
            values.push(Value::Integer(i));
        }
        assert_eq!(values.len(), 100);
        
        // Test mixed allocation sizes
        let mut mixed_values = Vec::new();
        for i in 0..50 {
            match i % 3 {
                0 => mixed_values.push(Value::Integer(i)),
                1 => mixed_values.push(Value::String(format!("test_{}", i))),
                _ => {
                    let list: Vec<Value> = (0..i%5).map(|j| Value::Integer(j)).collect();
                    mixed_values.push(Value::List(list));
                }
            }
        }
        assert_eq!(mixed_values.len(), 50);
    }
    
    #[test]
    fn test_pool_efficiency() {
        let pools = CompactValuePools::new();
        
        // Test basic allocation
        let mut values = Vec::new();
        for i in 0..100 {
            let val = CompactValue::SmallInt(i);
            values.push(pools.alloc_value(val));
        }
        assert_eq!(values.len(), 100);
        
        // Test recycling
        for value in &values[0..50] {
            pools.recycle_value(value);
        }
        
        // Pool should be able to handle this without issues
        let stats = pools.stats();
        assert!(!stats.is_empty());
    }
    
    #[test]
    fn test_memory_manager_integration() {
        let mut memory_manager = MemoryManager::new();
        
        let initial_stats = memory_manager.memory_stats();
        
        // Allocate some values
        let mut values = Vec::new();
        for i in 0..100 {
            let compact_val = CompactValue::SmallInt(i);
            values.push(memory_manager.alloc_compact_value(compact_val));
        }
        
        let peak_stats = memory_manager.memory_stats();
        assert!(peak_stats.total_allocated >= initial_stats.total_allocated);
        
        // Recycle values
        for value in &values {
            memory_manager.recycle_compact_value(value);
        }
        
        // Should be able to collect garbage
        let collected = memory_manager.collect_garbage();
        // Note: collected amount depends on implementation
    }
    
    #[test]
    fn test_string_interning_efficiency() {
        let interner = StringInterner::new();
        
        let test_strings = vec!["x", "y", "Plus", "Times", "x", "y", "Plus"];
        let mut symbol_ids = Vec::new();
        
        for s in &test_strings {
            symbol_ids.push(interner.intern_symbol_id(s));
        }
        
        // Duplicate strings should have same symbol IDs
        assert_eq!(symbol_ids[0], symbol_ids[4]); // "x"
        assert_eq!(symbol_ids[1], symbol_ids[5]); // "y"
        assert_eq!(symbol_ids[2], symbol_ids[6]); // "Plus"
        
        // Different strings should have different IDs
        assert_ne!(symbol_ids[0], symbol_ids[1]);
        assert_ne!(symbol_ids[0], symbol_ids[2]);
    }
    
    #[test]
    fn test_cache_access_patterns() {
        let data: Vec<Value> = (0..1000)
            .map(|i| Value::Integer(i))
            .collect();
        
        // Sequential access
        let mut sum1 = 0i64;
        for value in &data {
            if let Value::Integer(i) = value {
                sum1 += i;
            }
        }
        
        // Random access (pseudo-random)
        let mut sum2 = 0i64;
        for i in 0..data.len() {
            let idx = (i * 7919) % data.len();
            if let Value::Integer(val) = &data[idx] {
                sum2 += val;
            }
        }
        
        // Both should produce valid results
        assert!(sum1 > 0);
        assert!(sum2 > 0);
        
        // Sequential access sum should be the sum of 0..999
        let expected_sum = (0..1000).sum::<i64>();
        assert_eq!(sum1, expected_sum);
    }
    
    #[test]
    fn test_fragmentation_simulation() {
        // Simulate fragmentation through allocation/deallocation pattern
        let mut allocations = Vec::new();
        
        // Allocate many objects
        for i in 0..100 {
            allocations.push(Value::Integer(i));
            allocations.push(Value::String(format!("str_{}", i)));
        }
        
        // Create fragmentation by keeping only even indices
        let mut fragmented = Vec::new();
        for (i, value) in allocations.into_iter().enumerate() {
            if i % 2 == 0 {
                fragmented.push(value);
            }
        }
        
        // Should have half the original allocations
        assert_eq!(fragmented.len(), 100); // 200 original / 2 = 100
        
        // Add new allocations
        for i in 0..50 {
            fragmented.push(Value::Real(i as f64));
        }
        
        assert_eq!(fragmented.len(), 150);
    }
}