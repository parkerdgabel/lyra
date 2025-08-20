//! Benchmarks for memory optimization system
//!
//! This benchmark suite validates the memory efficiency improvements
//! implemented in Phase 2B: Memory & Symbol Optimization.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId, BatchSize};
use lyra::memory::{
    StringInterner, SymbolId, CompactValue, CompactValuePools, CacheAlignedValue,
    MemoryManager,
};
use lyra::vm::Value;
use std::collections::HashMap;

/// Benchmark string interning performance
fn bench_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");
    
    // Test data
    let common_symbols = vec![
        "x", "y", "z", "Plus", "Times", "Sin", "Cos", "Log", "Exp",
        "List", "Head", "Tail", "Length", "Map", "Apply", "Function",
    ];
    
    let custom_symbols: Vec<String> = (0..1000)
        .map(|i| format!("symbol_{}", i))
        .collect();
    
    // Benchmark legacy string storage (baseline)
    group.bench_function("legacy_string_storage", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for symbol in &common_symbols {
                strings.push(symbol.to_string());
            }
            for symbol in &custom_symbols {
                strings.push(symbol.clone());
            }
            strings
        });
    });
    
    // Benchmark new symbol ID interning
    group.bench_function("symbol_id_interning", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let mut symbol_ids = Vec::new();
            for symbol in &common_symbols {
                symbol_ids.push(interner.intern_symbol_id(symbol));
            }
            for symbol in &custom_symbols {
                symbol_ids.push(interner.intern_symbol_id(symbol));
            }
            symbol_ids
        });
    });
    
    // Benchmark symbol resolution
    let interner = StringInterner::new();
    let symbol_ids: Vec<SymbolId> = common_symbols.iter()
        .map(|s| interner.intern_symbol_id(s))
        .collect();
    
    group.bench_function("symbol_resolution", |b| {
        b.iter(|| {
            let mut resolved = Vec::new();
            for &id in &symbol_ids {
                resolved.push(interner.resolve_symbol(id));
            }
            resolved
        });
    });
    
    group.finish();
}

/// Benchmark CompactValue vs regular Value memory efficiency
fn bench_compact_value(c: &mut Criterion) {
    let mut group = c.benchmark_group("compact_value");
    
    let interner = StringInterner::new();
    
    // Test data
    let test_values = vec![
        Value::Integer(42),
        Value::Integer(i64::MAX),
        Value::Real(3.14159),
        Value::Symbol("x".to_string()),
        Value::String("hello world".to_string()),
        Value::Boolean(true),
        Value::Missing,
        Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]),
    ];
    
    // Benchmark regular Value creation and manipulation
    group.bench_function("regular_value_operations", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for test_value in &test_values {
                values.push(test_value.clone());
            }
            
            // Simulate some operations
            let mut processed = Vec::new();
            for value in values {
                match value {
                    Value::Integer(i) => processed.push(Value::Integer(i * 2)),
                    Value::Real(r) => processed.push(Value::Real(r * 2.0)),
                    other => processed.push(other),
                }
            }
            processed
        });
    });
    
    // Benchmark CompactValue creation and manipulation
    group.bench_function("compact_value_operations", |b| {
        b.iter(|| {
            let mut compact_values = Vec::new();
            for test_value in &test_values {
                compact_values.push(CompactValue::from_value(test_value.clone(), &interner));
            }
            
            // Simulate some operations
            let mut processed = Vec::new();
            for compact_value in compact_values {
                match compact_value {
                    CompactValue::SmallInt(i) => processed.push(CompactValue::SmallInt(i * 2)),
                    CompactValue::LargeInt(i) => processed.push(CompactValue::LargeInt(std::sync::Arc::new(*i * 2))),
                    CompactValue::Real(r) => processed.push(CompactValue::Real(r * 2.0)),
                    other => processed.push(other),
                }
            }
            processed
        });
    });
    
    // Benchmark round-trip conversion
    group.bench_function("round_trip_conversion", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for test_value in &test_values {
                let compact = CompactValue::from_value(test_value.clone(), &interner);
                let back_to_regular = compact.to_value(&interner);
                results.push(back_to_regular);
            }
            results
        });
    });
    
    group.finish();
}

/// Benchmark memory pool performance
fn bench_memory_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pools");
    
    let pools = CompactValuePools::new();
    
    // Test data for different value types
    let small_ints: Vec<CompactValue> = (0..1000)
        .map(|i| CompactValue::SmallInt(i))
        .collect();
    
    let large_ints: Vec<CompactValue> = (0..100)
        .map(|i| CompactValue::LargeInt(std::sync::Arc::new(i64::MAX - i as i64)))
        .collect();
    
    let reals: Vec<CompactValue> = (0..1000)
        .map(|i| CompactValue::Real(std::f64::consts::PI * i as f64))
        .collect();
    
    // Benchmark small integer pool performance
    group.bench_function("small_int_pool", |b| {
        b.iter(|| {
            let mut allocated = Vec::new();
            for &value in &small_ints {
                allocated.push(pools.alloc_value(value));
            }
            allocated
        });
    });
    
    // Benchmark large integer pool performance
    group.bench_function("large_int_pool", |b| {
        b.iter(|| {
            let mut allocated = Vec::new();
            for value in &large_ints {
                allocated.push(pools.alloc_value(value.clone()));
            }
            allocated
        });
    });
    
    // Benchmark real number pool performance
    group.bench_function("real_pool", |b| {
        b.iter(|| {
            let mut allocated = Vec::new();
            for &value in &reals {
                allocated.push(pools.alloc_value(value));
            }
            allocated
        });
    });
    
    // Benchmark pool statistics and efficiency
    group.bench_function("pool_statistics", |b| {
        b.iter(|| {
            pools.stats()
        });
    });
    
    group.finish();
}

/// Benchmark cache-aligned data structures
fn bench_cache_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("cache_alignment");
    
    let interner = StringInterner::new();
    
    // Create test data
    let values: Vec<CompactValue> = (0..1000)
        .map(|i| {
            if i % 3 == 0 {
                CompactValue::SmallInt(i)
            } else if i % 3 == 1 {
                CompactValue::Real(i as f64 * 3.14)
            } else {
                CompactValue::Symbol(interner.intern_symbol_id(&format!("sym_{}", i)))
            }
        })
        .collect();
    
    // Benchmark regular CompactValue access patterns
    group.bench_function("regular_value_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            let mut real_sum = 0.0f64;
            for value in &values {
                match value {
                    CompactValue::SmallInt(i) => sum += *i as i64,
                    CompactValue::Real(r) => real_sum += r,
                    _ => {}
                }
            }
            (sum, real_sum)
        });
    });
    
    // Benchmark cache-aligned value access patterns
    let aligned_values: Vec<CacheAlignedValue> = values.iter()
        .map(|v| CacheAlignedValue::new(v.clone()))
        .collect();
    
    group.bench_function("cache_aligned_access", |b| {
        b.iter(|| {
            let mut sum = 0i64;
            let mut real_sum = 0.0f64;
            for aligned_value in &aligned_values {
                match &aligned_value.value {
                    CompactValue::SmallInt(i) => sum += *i as i64,
                    CompactValue::Real(r) => real_sum += r,
                    _ => {}
                }
            }
            (sum, real_sum)
        });
    });
    
    group.finish();
}

/// Benchmark overall memory manager performance
fn bench_memory_manager(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_manager");
    
    // Benchmark memory manager creation and basic operations
    group.bench_function("manager_creation", |b| {
        b.iter(|| {
            MemoryManager::new()
        });
    });
    
    let manager = MemoryManager::new();
    
    // Benchmark symbol interning through manager
    group.bench_function("manager_symbol_interning", |b| {
        b.iter_batched(
            || vec!["x", "y", "Plus", "Times", "custom_symbol_1", "custom_symbol_2"],
            |symbols| {
                let mut ids = Vec::new();
                for symbol in symbols {
                    ids.push(manager.intern_symbol(symbol));
                }
                ids
            },
            BatchSize::SmallInput
        );
    });
    
    // Benchmark value conversion through manager
    let test_values = vec![
        Value::Integer(42),
        Value::Real(3.14159),
        Value::Symbol("test".to_string()),
        Value::String("hello".to_string()),
        Value::Boolean(true),
        Value::Missing,
    ];
    
    group.bench_function("manager_value_conversion", |b| {
        b.iter(|| {
            let mut compact_values = Vec::new();
            for value in &test_values {
                compact_values.push(manager.compact_value(value.clone()));
            }
            compact_values
        });
    });
    
    // Benchmark memory statistics collection
    group.bench_function("memory_statistics", |b| {
        b.iter(|| {
            manager.memory_stats()
        });
    });
    
    // Benchmark efficiency report generation
    group.bench_function("efficiency_report", |b| {
        b.iter(|| {
            manager.efficiency_report()
        });
    });
    
    group.finish();
}

/// Memory usage comparison benchmark
fn bench_memory_usage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage_comparison");
    
    // Create large datasets for memory usage comparison
    let dataset_sizes = [100, 1000, 10000];
    
    for &size in &dataset_sizes {
        group.bench_with_input(
            BenchmarkId::new("regular_values", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut values = Vec::with_capacity(size);
                    for i in 0..size {
                        match i % 5 {
                            0 => values.push(Value::Integer(i as i64)),
                            1 => values.push(Value::Real(i as f64)),
                            2 => values.push(Value::Symbol(format!("sym_{}", i))),
                            3 => values.push(Value::String(format!("str_{}", i))),
                            4 => values.push(Value::Boolean(i % 2 == 0)),
                            _ => unreachable!(),
                        }
                    }
                    
                    // Calculate memory usage estimate
                    values.len() * std::mem::size_of::<Value>()
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("compact_values", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let interner = StringInterner::new();
                    let mut values = Vec::with_capacity(size);
                    for i in 0..size {
                        match i % 5 {
                            0 => values.push(CompactValue::SmallInt(i as i32)),
                            1 => values.push(CompactValue::Real(i as f64)),
                            2 => values.push(CompactValue::Symbol(interner.intern_symbol_id(&format!("sym_{}", i)))),
                            3 => values.push(CompactValue::String(interner.intern_symbol_id(&format!("str_{}", i)))),
                            4 => values.push(CompactValue::Boolean(i % 2 == 0)),
                            _ => unreachable!(),
                        }
                    }
                    
                    // Calculate memory usage estimate
                    values.len() * std::mem::size_of::<CompactValue>()
                });
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_string_interning,
    bench_compact_value,
    bench_memory_pools,
    bench_cache_alignment,
    bench_memory_manager,
    bench_memory_usage_comparison
);

criterion_main!(benches);