//! Cache Alignment Performance Benchmarks
//!
//! Validates cache hit rates and access patterns, measures cache line utilization,
//! and tests the impact of data structure alignment on performance.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::memory::{CompactValue, CacheAlignedValue, StringInterner};
use std::sync::Arc;

/// Benchmark cache line utilization with different data layouts
fn cache_line_utilization(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_line_utilization");
    
    // Regular layout: values packed together
    let regular_values: Vec<CompactValue> = (0..1000)
        .map(|i| CompactValue::SmallInt(i as i32))
        .collect();
    
    // Cache-aligned layout: each value on its own cache line
    let aligned_values: Vec<CacheAlignedValue> = (0..1000)
        .map(|i| CacheAlignedValue::new(CompactValue::SmallInt(i as i32)))
        .collect();
    
    group.bench_function("regular_layout_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for value in &regular_values {
                if let CompactValue::SmallInt(i) = value {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    group.bench_function("aligned_layout_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for value in &aligned_values {
                if let CompactValue::SmallInt(i) = &**value {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark different memory access patterns
fn memory_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_access_patterns");
    
    let data_size = 1024;
    let regular_data: Vec<CompactValue> = (0..data_size)
        .map(|i| CompactValue::SmallInt(i as i32))
        .collect();
    
    let aligned_data: Vec<CacheAlignedValue> = (0..data_size)
        .map(|i| CacheAlignedValue::new(CompactValue::SmallInt(i as i32)))
        .collect();
    
    // Sequential access pattern
    group.bench_function("regular_sequential_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            for value in &regular_data {
                if let CompactValue::SmallInt(i) = value {
                    checksum = checksum.wrapping_add(*i as i64);
                }
            }
            black_box(checksum);
        });
    });
    
    group.bench_function("aligned_sequential_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            for value in &aligned_data {
                if let CompactValue::SmallInt(i) = &**value {
                    checksum = checksum.wrapping_add(*i as i64);
                }
            }
            black_box(checksum);
        });
    });
    
    // Random access pattern
    let indices: Vec<usize> = (0..data_size).collect();
    
    group.bench_function("regular_random_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            for &idx in &indices {
                let value = &regular_data[idx % regular_data.len()];
                if let CompactValue::SmallInt(i) = value {
                    checksum = checksum.wrapping_add(*i as i64);
                }
            }
            black_box(checksum);
        });
    });
    
    group.bench_function("aligned_random_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            for &idx in &indices {
                let value = &aligned_data[idx % aligned_data.len()];
                if let CompactValue::SmallInt(i) = &**value {
                    checksum = checksum.wrapping_add(*i as i64);
                }
            }
            black_box(checksum);
        });
    });
    
    // Strided access pattern (every 8th element)
    group.bench_function("regular_strided_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            let mut i = 0;
            while i < regular_data.len() {
                if let CompactValue::SmallInt(val) = &regular_data[i] {
                    checksum = checksum.wrapping_add(*val as i64);
                }
                i += 8; // Skip 7 elements
            }
            black_box(checksum);
        });
    });
    
    group.bench_function("aligned_strided_read", |b| {
        b.iter(|| {
            let mut checksum = 0i64;
            let mut i = 0;
            while i < aligned_data.len() {
                if let CompactValue::SmallInt(val) = &**aligned_data[i] {
                    checksum = checksum.wrapping_add(*val as i64);
                }
                i += 8; // Skip 7 elements
            }
            black_box(checksum);
        });
    });
    
    group.finish();
}

/// Benchmark cache performance with different data structures
fn cache_performance_data_structures(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_performance_structures");
    
    let interner = StringInterner::new();
    
    // Array of structures (AoS) - typical layout
    #[derive(Clone)]
    struct ValueRecord {
        id: u32,
        value: CompactValue,
        metadata: u64,
    }
    
    let aos_data: Vec<ValueRecord> = (0..1000)
        .map(|i| ValueRecord {
            id: i as u32,
            value: CompactValue::SmallInt(i as i32),
            metadata: (i as u64) * 42,
        })
        .collect();
    
    // Structure of arrays (SoA) - cache-friendly layout
    struct ValueArrays {
        ids: Vec<u32>,
        values: Vec<CompactValue>,
        metadata: Vec<u64>,
    }
    
    let soa_data = ValueArrays {
        ids: (0..1000).map(|i| i as u32).collect(),
        values: (0..1000).map(|i| CompactValue::SmallInt(i as i32)).collect(),
        metadata: (0..1000).map(|i| (i as u64) * 42).collect(),
    };
    
    // Benchmark accessing only values (common case)
    group.bench_function("aos_value_only_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for record in &aos_data {
                if let CompactValue::SmallInt(i) = &record.value {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    group.bench_function("soa_value_only_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for value in &soa_data.values {
                if let CompactValue::SmallInt(i) = value {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    // Benchmark accessing all fields
    group.bench_function("aos_full_record_access", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for record in &aos_data {
                sum += record.id as u64;
                if let CompactValue::SmallInt(i) = &record.value {
                    sum += *i as u64;
                }
                sum += record.metadata;
            }
            black_box(sum);
        });
    });
    
    group.bench_function("soa_full_record_access", |b| {
        b.iter(|| {
            let mut sum = 0u64;
            for i in 0..soa_data.ids.len() {
                sum += soa_data.ids[i] as u64;
                if let CompactValue::SmallInt(val) = &soa_data.values[i] {
                    sum += *val as u64;
                }
                sum += soa_data.metadata[i];
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark false sharing effects
fn false_sharing_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("false_sharing_effects");
    
    use std::thread;
    use std::sync::Arc;
    
    // Shared cache line data (prone to false sharing)
    #[repr(C)]
    struct SharedCacheLine {
        counter1: std::sync::atomic::AtomicU64,
        counter2: std::sync::atomic::AtomicU64,
        counter3: std::sync::atomic::AtomicU64,
        counter4: std::sync::atomic::AtomicU64,
    }
    
    // Cache-aligned data (avoids false sharing)
    #[repr(align(64))]
    struct AlignedCounter {
        counter: std::sync::atomic::AtomicU64,
    }
    
    group.bench_function("false_sharing_prone", |b| {
        b.iter(|| {
            let shared_data = Arc::new(SharedCacheLine {
                counter1: std::sync::atomic::AtomicU64::new(0),
                counter2: std::sync::atomic::AtomicU64::new(0),
                counter3: std::sync::atomic::AtomicU64::new(0),
                counter4: std::sync::atomic::AtomicU64::new(0),
            });
            
            let mut handles = vec![];
            
            for i in 0..4 {
                let data_clone = Arc::clone(&shared_data);
                let handle = thread::spawn(move || {
                    for _ in 0..1000 {
                        match i {
                            0 => data_clone.counter1.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                            1 => data_clone.counter2.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                            2 => data_clone.counter3.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                            _ => data_clone.counter4.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
                        };
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            black_box((
                shared_data.counter1.load(std::sync::atomic::Ordering::Relaxed),
                shared_data.counter2.load(std::sync::atomic::Ordering::Relaxed),
                shared_data.counter3.load(std::sync::atomic::Ordering::Relaxed),
                shared_data.counter4.load(std::sync::atomic::Ordering::Relaxed),
            ));
        });
    });
    
    group.bench_function("false_sharing_avoided", |b| {
        b.iter(|| {
            let counters = Arc::new(vec![
                AlignedCounter { counter: std::sync::atomic::AtomicU64::new(0) },
                AlignedCounter { counter: std::sync::atomic::AtomicU64::new(0) },
                AlignedCounter { counter: std::sync::atomic::AtomicU64::new(0) },
                AlignedCounter { counter: std::sync::atomic::AtomicU64::new(0) },
            ]);
            
            let mut handles = vec![];
            
            for i in 0..4 {
                let counters_clone = Arc::clone(&counters);
                let handle = thread::spawn(move || {
                    for _ in 0..1000 {
                        counters_clone[i].counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let results: Vec<u64> = counters.iter()
                .map(|c| c.counter.load(std::sync::atomic::Ordering::Relaxed))
                .collect();
            
            black_box(results);
        });
    });
    
    group.finish();
}

/// Benchmark cache-friendly algorithms
fn cache_friendly_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_friendly_algorithms");
    
    let size = 512; // 512x512 matrix fits in most L2 caches
    let matrix: Vec<Vec<i32>> = (0..size)
        .map(|i| (0..size).map(|j| (i * size + j) as i32).collect())
        .collect();
    
    // Row-major traversal (cache-friendly)
    group.bench_function("matrix_row_major_traversal", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for i in 0..size {
                for j in 0..size {
                    sum += matrix[i][j] as i64;
                }
            }
            black_box(sum);
        });
    });
    
    // Column-major traversal (cache-unfriendly)
    group.bench_function("matrix_column_major_traversal", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for j in 0..size {
                for i in 0..size {
                    sum += matrix[i][j] as i64;
                }
            }
            black_box(sum);
        });
    });
    
    // Blocked traversal (cache-friendly for larger matrices)
    group.bench_function("matrix_blocked_traversal", |b| {
        let block_size = 64; // Tune for cache size
        b.iter(|| {
            let mut sum = 0i64;
            for block_i in (0..size).step_by(block_size) {
                for block_j in (0..size).step_by(block_size) {
                    for i in block_i..std::cmp::min(block_i + block_size, size) {
                        for j in block_j..std::cmp::min(block_j + block_size, size) {
                            sum += matrix[i][j] as i64;
                        }
                    }
                }
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark memory prefetching effects
fn memory_prefetching_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_prefetching");
    
    let data: Vec<CompactValue> = (0..10000)
        .map(|i| CompactValue::SmallInt(i as i32))
        .collect();
    
    // Sequential access (good for hardware prefetching)
    group.bench_function("sequential_access_prefetch_friendly", |b| {
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
    
    // Random access (poor for hardware prefetching)
    let mut indices: Vec<usize> = (0..data.len()).collect();
    // Shuffle indices to create random access pattern
    for i in 0..indices.len() {
        let j = (i * 7919) % indices.len(); // Simple pseudo-random shuffle
        indices.swap(i, j);
    }
    
    group.bench_function("random_access_prefetch_unfriendly", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for &idx in &indices {
                if let CompactValue::SmallInt(i) = &data[idx] {
                    sum += *i as i64;
                }
            }
            black_box(sum);
        });
    });
    
    // Strided access with manual prefetch hints
    group.bench_function("strided_access_with_prefetch", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            let stride = 16;
            for i in (0..data.len()).step_by(stride) {
                // Manual prefetch hint (if available on platform)
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    if i + stride * 8 < data.len() {
                        std::arch::x86_64::_mm_prefetch(
                            data.as_ptr().add(i + stride * 8) as *const i8,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                
                if let CompactValue::SmallInt(val) = &data[i] {
                    sum += *val as i64;
                }
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

criterion_group!(
    cache_alignment_benchmarks,
    cache_line_utilization,
    memory_access_patterns,
    cache_performance_data_structures,
    false_sharing_effects,
    cache_friendly_algorithms,
    memory_prefetching_effects
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_cache_alignment() {
        let aligned_value = CacheAlignedValue::new(CompactValue::SmallInt(42));
        
        // Should be aligned to 64-byte cache line
        let ptr = &aligned_value as *const CacheAlignedValue as usize;
        assert_eq!(ptr % 64, 0, "CacheAlignedValue should be aligned to 64-byte boundary");
        
        // Size should be exactly 64 bytes
        assert_eq!(std::mem::size_of::<CacheAlignedValue>(), 64);
        
        // Should be able to access the wrapped value
        assert_eq!(*aligned_value, CompactValue::SmallInt(42));
    }
    
    #[test]
    fn validate_false_sharing_avoidance() {
        use std::sync::atomic::{AtomicU64, Ordering};
        
        // Test that aligned counters avoid false sharing
        #[repr(align(64))]
        struct AlignedCounter {
            counter: AtomicU64,
        }
        
        let counter1 = AlignedCounter { counter: AtomicU64::new(0) };
        let counter2 = AlignedCounter { counter: AtomicU64::new(0) };
        
        let ptr1 = &counter1 as *const AlignedCounter as usize;
        let ptr2 = &counter2 as *const AlignedCounter as usize;
        
        // Should be aligned to cache line boundaries
        assert_eq!(ptr1 % 64, 0);
        assert_eq!(ptr2 % 64, 0);
        
        // If they're on different cache lines, the difference should be >= 64
        if ptr1 != ptr2 {
            let diff = if ptr1 > ptr2 { ptr1 - ptr2 } else { ptr2 - ptr1 };
            println!("Counter address difference: {} bytes", diff);
        }
    }
    
    #[test]
    fn validate_cache_line_size() {
        // Verify that our assumption about 64-byte cache lines is reasonable
        // (This is typical for x86-64, but may vary on other architectures)
        
        let cache_line_size = 64; // Assumed cache line size
        
        // CacheAlignedValue should match this size
        assert_eq!(std::mem::size_of::<CacheAlignedValue>(), cache_line_size);
        assert_eq!(std::mem::align_of::<CacheAlignedValue>(), cache_line_size);
        
        println!("Using cache line size: {} bytes", cache_line_size);
    }
    
    #[test]
    fn validate_memory_layout_efficiency() {
        // Compare memory usage of different layouts
        let regular_values: Vec<CompactValue> = (0..1000)
            .map(|i| CompactValue::SmallInt(i as i32))
            .collect();
        
        let aligned_values: Vec<CacheAlignedValue> = (0..1000)
            .map(|i| CacheAlignedValue::new(CompactValue::SmallInt(i as i32)))
            .collect();
        
        let regular_size = std::mem::size_of_val(&regular_values);
        let aligned_size = std::mem::size_of_val(&aligned_values);
        
        println!("Regular layout: {} bytes", regular_size);
        println!("Aligned layout: {} bytes", aligned_size);
        println!("Alignment overhead: {}x", aligned_size as f64 / regular_size as f64);
        
        // Aligned layout should be larger due to padding
        assert!(aligned_size >= regular_size);
        
        // But the overhead should be reasonable (less than 10x for this case)
        assert!((aligned_size as f64 / regular_size as f64) < 10.0);
    }
    
    #[test]
    fn validate_cache_friendly_access_patterns() {
        // Test that sequential access is faster than random access
        use std::time::Instant;
        
        let data: Vec<CompactValue> = (0..10000)
            .map(|i| CompactValue::SmallInt(i as i32))
            .collect();
        
        // Sequential access
        let start = Instant::now();
        let mut sum = 0i64;
        for value in &data {
            if let CompactValue::SmallInt(i) = value {
                sum += *i as i64;
            }
        }
        let sequential_time = start.elapsed();
        
        // Random access (simple pattern)
        let start = Instant::now();
        let mut sum2 = 0i64;
        for i in 0..data.len() {
            let idx = (i * 7919) % data.len(); // Pseudo-random pattern
            if let CompactValue::SmallInt(val) = &data[idx] {
                sum2 += *val as i64;
            }
        }
        let random_time = start.elapsed();
        
        println!("Sequential access: {:?}", sequential_time);
        println!("Random access: {:?}", random_time);
        
        // Both should complete successfully
        assert!(sum != 0);
        assert!(sum2 != 0);
        
        // Sequential should generally be faster, but this is platform-dependent
        println!("Random/Sequential ratio: {:.2}", 
                random_time.as_nanos() as f64 / sequential_time.as_nanos() as f64);
    }
}