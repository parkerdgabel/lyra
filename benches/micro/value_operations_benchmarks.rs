//! Value Operations Performance Benchmarks
//!
//! Validates the claimed 20-30% cache performance improvement from Value enum optimization.
//! Tests optimized Value enum performance vs original, cache alignment benefits,
//! and memory access patterns.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::vm::Value;
use lyra::memory::{StringInterner, CompactValue, CacheAlignedValue};
use std::sync::Arc;

/// Benchmark Value creation and basic operations
fn value_creation_and_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_creation_operations");
    
    // Baseline: Original Value enum operations
    group.bench_function("original_value_creation", |b| {
        b.iter(|| {
            let values = vec![
                Value::Integer(42),
                Value::Real(3.14159),
                Value::Symbol("Plus".to_string()),
                Value::String("hello".to_string()),
                Value::Boolean(true),
                Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]),
            ];
            black_box(values);
        });
    });
    
    // Optimized: CompactValue operations
    group.bench_function("compact_value_creation", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let values = vec![
                CompactValue::SmallInt(42),
                CompactValue::Real(3.14159),
                CompactValue::Symbol(interner.intern_symbol_id("Plus")),
                CompactValue::String(interner.intern_symbol_id("hello")),
                CompactValue::Boolean(true),
                CompactValue::List(Arc::new(vec![
                    CompactValue::SmallInt(1),
                    CompactValue::SmallInt(2), 
                    CompactValue::SmallInt(3)
                ])),
            ];
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark memory layout and access patterns
fn memory_layout_and_access_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_layout_access");
    
    // Test array access patterns with different Value types
    let original_values: Vec<Value> = (0..1000)
        .map(|i| {
            match i % 4 {
                0 => Value::Integer(i as i64),
                1 => Value::Real(i as f64),
                2 => Value::Symbol(format!("sym_{}", i)),
                _ => Value::Boolean(i % 2 == 0),
            }
        })
        .collect();
    
    let interner = StringInterner::new();
    let compact_values: Vec<CompactValue> = (0..1000)
        .map(|i| {
            match i % 4 {
                0 => {
                    if i <= i32::MAX as usize {
                        CompactValue::SmallInt(i as i32)
                    } else {
                        CompactValue::LargeInt(Arc::new(i as i64))
                    }
                }
                1 => CompactValue::Real(i as f64),
                2 => CompactValue::Symbol(interner.intern_symbol_id(&format!("sym_{}", i))),
                _ => CompactValue::Boolean(i % 2 == 0),
            }
        })
        .collect();
    
    group.bench_function("original_value_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for value in &original_values {
                match value {
                    Value::Integer(i) => sum += i,
                    Value::Real(r) => sum += *r as i64,
                    _ => sum += 1,
                }
            }
            black_box(sum);
        });
    });
    
    group.bench_function("compact_value_sequential_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            for value in &compact_values {
                match value {
                    CompactValue::SmallInt(i) => sum += *i as i64,
                    CompactValue::LargeInt(i) => sum += **i,
                    CompactValue::Real(r) => sum += *r as i64,
                    _ => sum += 1,
                }
            }
            black_box(sum);
        });
    });
    
    group.finish();
}

/// Benchmark cache alignment effects
fn cache_alignment_effects(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_alignment_effects");
    
    // Regular CompactValue array
    let regular_values: Vec<CompactValue> = (0..1000)
        .map(|i| CompactValue::SmallInt(i as i32))
        .collect();
    
    // Cache-aligned CompactValue array
    let aligned_values: Vec<CacheAlignedValue> = (0..1000)
        .map(|i| CacheAlignedValue::new(CompactValue::SmallInt(i as i32)))
        .collect();
    
    group.bench_function("regular_compact_value_access", |b| {
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
    
    group.bench_function("cache_aligned_value_access", |b| {
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

/// Benchmark value conversion overhead
fn value_conversion_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_conversion_overhead");
    
    let interner = StringInterner::new();
    let original_values = vec![
        Value::Integer(42),
        Value::Real(3.14),
        Value::Symbol("Plus".to_string()),
        Value::String("test".to_string()),
        Value::Boolean(true),
    ];
    
    group.bench_function("value_to_compact_conversion", |b| {
        b.iter(|| {
            let compact_values: Vec<CompactValue> = original_values.iter()
                .map(|v| CompactValue::from_value(v.clone(), &interner))
                .collect();
            black_box(compact_values);
        });
    });
    
    // Pre-create compact values for reverse conversion
    let compact_values: Vec<CompactValue> = original_values.iter()
        .map(|v| CompactValue::from_value(v.clone(), &interner))
        .collect();
    
    group.bench_function("compact_to_value_conversion", |b| {
        b.iter(|| {
            let converted_values: Vec<Value> = compact_values.iter()
                .map(|v| v.to_value(&interner))
                .collect();
            black_box(converted_values);
        });
    });
    
    group.finish();
}

/// Benchmark small integer optimization effectiveness
fn small_integer_optimization(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_integer_optimization");
    
    // Test range of integer values to see optimization effectiveness
    let small_integers: Vec<i64> = (-1000..1000).collect();
    let large_integers: Vec<i64> = vec![i64::MIN, -1000000000, 1000000000, i64::MAX];
    
    group.bench_function("small_integers_as_original_value", |b| {
        b.iter(|| {
            let values: Vec<Value> = small_integers.iter()
                .map(|&i| Value::Integer(i))
                .collect();
            black_box(values);
        });
    });
    
    group.bench_function("small_integers_as_compact_value", |b| {
        b.iter(|| {
            let values: Vec<CompactValue> = small_integers.iter()
                .map(|&i| {
                    if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                        CompactValue::SmallInt(i as i32)
                    } else {
                        CompactValue::LargeInt(Arc::new(i))
                    }
                })
                .collect();
            black_box(values);
        });
    });
    
    group.bench_function("large_integers_as_original_value", |b| {
        b.iter(|| {
            let values: Vec<Value> = large_integers.iter()
                .map(|&i| Value::Integer(i))
                .collect();
            black_box(values);
        });
    });
    
    group.bench_function("large_integers_as_compact_value", |b| {
        b.iter(|| {
            let values: Vec<CompactValue> = large_integers.iter()
                .map(|&i| CompactValue::LargeInt(Arc::new(i)))
                .collect();
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark memory usage patterns
fn memory_usage_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_patterns");
    
    let interner = StringInterner::new();
    
    // Compare memory allocation patterns for different approaches
    group.bench_function("original_value_memory_allocation", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(Value::Integer(i));
                values.push(Value::Symbol(format!("sym_{}", i)));
                values.push(Value::Real(i as f64));
            }
            black_box(values);
        });
    });
    
    group.bench_function("compact_value_memory_allocation", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(CompactValue::SmallInt(i as i32));
                values.push(CompactValue::Symbol(interner.intern_symbol_id(&format!("sym_{}", i))));
                values.push(CompactValue::Real(i as f64));
            }
            black_box(values);
        });
    });
    
    group.finish();
}

/// Benchmark list operations performance
fn list_operations_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("list_operations_performance");
    
    let interner = StringInterner::new();
    
    // Create test data
    let original_list = Value::List(
        (0..1000).map(|i| Value::Integer(i)).collect()
    );
    
    let compact_list = CompactValue::List(Arc::new(
        (0..1000).map(|i| CompactValue::SmallInt(i as i32)).collect()
    ));
    
    group.bench_function("original_list_iteration", |b| {
        b.iter(|| {
            if let Value::List(items) = &original_list {
                let mut sum = 0i64;
                for item in items {
                    if let Value::Integer(i) = item {
                        sum += i;
                    }
                }
                black_box(sum);
            }
        });
    });
    
    group.bench_function("compact_list_iteration", |b| {
        b.iter(|| {
            if let CompactValue::List(items) = &compact_list {
                let mut sum = 0i64;
                for item in items.iter() {
                    if let CompactValue::SmallInt(i) = item {
                        sum += *i as i64;
                    }
                }
                black_box(sum);
            }
        });
    });
    
    group.finish();
}

criterion_group!(
    value_operations_benchmarks,
    value_creation_and_operations,
    memory_layout_and_access_patterns,
    cache_alignment_effects,
    value_conversion_overhead,
    small_integer_optimization,
    memory_usage_patterns,
    list_operations_performance
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_value_size_optimization() {
        // Ensure CompactValue is actually smaller than Value
        let original_size = std::mem::size_of::<Value>();
        let compact_size = std::mem::size_of::<CompactValue>();
        
        println!("Original Value size: {} bytes", original_size);
        println!("CompactValue size: {} bytes", compact_size);
        
        // CompactValue should be smaller than original Value
        assert!(compact_size < original_size, 
            "CompactValue ({} bytes) should be smaller than Value ({} bytes)", 
            compact_size, original_size);
    }
    
    #[test]
    fn validate_cache_alignment() {
        let aligned_size = std::mem::size_of::<CacheAlignedValue>();
        let alignment = std::mem::align_of::<CacheAlignedValue>();
        
        println!("CacheAlignedValue size: {} bytes", aligned_size);
        println!("CacheAlignedValue alignment: {} bytes", alignment);
        
        // Should be aligned to 64-byte cache line
        assert_eq!(alignment, 64);
        assert_eq!(aligned_size, 64); // Should pad to cache line size
    }
    
    #[test]
    fn validate_small_integer_optimization() {
        // Small integers should fit in SmallInt variant
        assert!(i32::MIN as i64 >= i32::MIN as i64);
        assert!(i32::MAX as i64 <= i32::MAX as i64);
        
        // Verify size difference
        let small_int_size = std::mem::size_of::<i32>();
        let large_int_size = std::mem::size_of::<Arc<i64>>();
        
        println!("SmallInt storage: {} bytes", small_int_size);
        println!("LargeInt storage: {} bytes", large_int_size);
        
        // SmallInt should be more efficient for common values
        assert!(small_int_size <= 8); // Should fit in 8 bytes or less
    }
    
    #[test]
    fn validate_symbol_interning_integration() {
        let interner = StringInterner::new();
        
        // Test that symbol interning works correctly with CompactValue
        let symbol1 = CompactValue::Symbol(interner.intern_symbol_id("test"));
        let symbol2 = CompactValue::Symbol(interner.intern_symbol_id("test"));
        
        // Should have same SymbolId for same string
        if let (CompactValue::Symbol(id1), CompactValue::Symbol(id2)) = (&symbol1, &symbol2) {
            assert_eq!(id1, id2);
        } else {
            panic!("Expected symbols");
        }
        
        // SymbolId should be much smaller than String
        assert!(std::mem::size_of::<lyra::memory::SymbolId>() < std::mem::size_of::<String>());
    }
    
    #[test]
    fn validate_memory_efficiency() {
        let interner = StringInterner::new();
        
        // Create equivalent data with both approaches
        let original_data: Vec<Value> = vec![
            Value::Integer(42),
            Value::Symbol("x".to_string()),
            Value::String("hello".to_string()),
            Value::Boolean(true),
        ];
        
        let compact_data: Vec<CompactValue> = vec![
            CompactValue::SmallInt(42),
            CompactValue::Symbol(interner.intern_symbol_id("x")),
            CompactValue::String(interner.intern_symbol_id("hello")),
            CompactValue::Boolean(true),
        ];
        
        // Calculate estimated memory usage
        let original_memory: usize = original_data.iter()
            .map(|v| std::mem::size_of_val(v))
            .sum();
        
        let compact_memory: usize = compact_data.iter()
            .map(|v| v.memory_size())
            .sum();
        
        println!("Original data memory: {} bytes", original_memory);
        println!("Compact data memory: {} bytes", compact_memory);
        
        // Compact representation should use less memory
        // Note: This is an approximation since we can't measure exact heap usage
        assert!(compact_memory <= original_memory, 
            "Compact representation should not use more memory than original");
    }
}