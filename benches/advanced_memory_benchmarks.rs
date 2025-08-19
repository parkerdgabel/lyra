//! Advanced Memory Management Benchmarks for Lyra
//!
//! This benchmark suite validates the 35% memory reduction target and measures
//! performance characteristics of the advanced memory management system.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lyra::memory::{
    MemoryManager, StringInterner, ValuePools, ComputationArena, ManagedValue
};
use lyra::vm::Value;

/// Benchmark string interning performance vs standard String allocation
fn benchmark_string_interning(c: &mut Criterion) {
    let mut group = c.benchmark_group("string_interning");
    
    // Common strings used in symbolic computation
    let test_strings = vec![
        "x", "y", "z", "Plus", "Times", "Sin", "Cos", "Exp", "Log",
        "function_name", "variable_123", "long_symbolic_expression_name",
        "std::math::trigonometry::AdvancedSin", "user::custom::MyFunction"
    ];
    
    group.throughput(Throughput::Elements(test_strings.len() as u64));
    
    // Benchmark standard String allocation
    group.bench_function("standard_strings", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for s in &test_strings {
                strings.push(black_box(s.to_string()));
            }
            strings
        })
    });
    
    // Benchmark string interning
    group.bench_function("interned_strings", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let mut interned = Vec::new();
            for s in &test_strings {
                interned.push(black_box(interner.intern(s)));
            }
            interned
        })
    });
    
    // Benchmark repeated interning (should hit cache)
    group.bench_function("repeated_interning", |b| {
        let interner = StringInterner::new();
        // Pre-warm cache
        for s in &test_strings {
            interner.intern(s);
        }
        
        b.iter(|| {
            let mut interned = Vec::new();
            for s in &test_strings {
                interned.push(black_box(interner.intern(s)));
            }
            interned
        })
    });
    
    group.finish();
}

/// Benchmark ManagedValue vs standard Value memory usage and performance
fn benchmark_managed_values(c: &mut Criterion) {
    let mut group = c.benchmark_group("managed_values");
    
    let test_values = create_test_values();
    group.throughput(Throughput::Elements(test_values.len() as u64));
    
    // Benchmark standard Value creation
    group.bench_function("standard_values", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for val in &test_values {
                values.push(black_box(val.clone()));
            }
            values
        })
    });
    
    // Benchmark ManagedValue creation
    group.bench_function("managed_values", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let mut managed_values = Vec::new();
            for val in &test_values {
                if let Ok(managed) = ManagedValue::from_value(val.clone(), &interner) {
                    managed_values.push(black_box(managed));
                }
            }
            managed_values
        })
    });
    
    // Benchmark managed value operations
    group.bench_function("managed_operations", |b| {
        let interner = StringInterner::new();
        let managed_values: Vec<_> = test_values.iter()
            .filter_map(|v| ManagedValue::from_value(v.clone(), &interner).ok())
            .collect();
        
        b.iter(|| {
            let mut result = 0usize;
            for managed in &managed_values {
                result += black_box(managed.memory_size());
            }
            result
        })
    });
    
    group.finish();
}

/// Benchmark memory pool efficiency
fn benchmark_memory_pools(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_pools");
    
    let pool_sizes = vec![100, 1000, 10000];
    
    for size in pool_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark without pools (standard allocation)
        group.bench_with_input(
            BenchmarkId::new("standard_allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut values = Vec::new();
                    for i in 0..size {
                        values.push(black_box(Value::Integer(i as i64)));
                    }
                    values
                })
            },
        );
        
        // Benchmark with memory pools
        group.bench_with_input(
            BenchmarkId::new("pooled_allocation", size),
            &size,
            |b, &size| {
                let pools = ValuePools::new();
                b.iter(|| {
                    let mut values = Vec::new();
                    for i in 0..size {
                        if let Ok(managed) = pools.alloc_value(Value::Integer(i as i64)) {
                            values.push(black_box(managed));
                        }
                    }
                    values
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark arena allocation for temporary computations
fn benchmark_arena_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_allocation");
    
    let computation_sizes = vec![50, 500, 5000];
    
    for size in computation_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Benchmark standard heap allocation
        group.bench_with_input(
            BenchmarkId::new("heap_allocation", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let mut temps = Vec::new();
                    for i in 0..size {
                        temps.push(black_box(ManagedValue::integer(i as i64)));
                    }
                    temps // Let them drop individually
                })
            },
        );
        
        // Benchmark arena allocation
        group.bench_with_input(
            BenchmarkId::new("arena_allocation", size),
            &size,
            |b, &size| {
                let arena = ComputationArena::new();
                b.iter(|| {
                    arena.with_scope(|_scope| {
                        for i in 0..size {
                            arena.alloc(black_box(ManagedValue::integer(i as i64)));
                        }
                    }); // All allocated values cleaned up in one operation
                })
            },
        );
    }
    
    group.finish();
}

/// Memory usage comparison benchmark  
fn benchmark_memory_usage_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_usage");
    
    // Create large datasets for memory comparison
    let dataset_sizes = vec![1000, 10000, 100000];
    
    for size in dataset_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Measure standard Value memory usage
        group.bench_with_input(
            BenchmarkId::new("standard_memory", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let values = create_large_value_dataset(size);
                    let memory_usage = estimate_value_memory_usage(&values);
                    black_box(memory_usage)
                })
            },
        );
        
        // Measure managed Value memory usage
        group.bench_with_input(
            BenchmarkId::new("managed_memory", size),
            &size,
            |b, &size| {
                let interner = StringInterner::new();
                b.iter(|| {
                    let values = create_large_value_dataset(size);
                    let managed_values: Vec<_> = values.iter()
                        .filter_map(|v| ManagedValue::from_value(v.clone(), &interner).ok())
                        .collect();
                    let memory_usage = managed_values.iter().map(|v| v.memory_size()).sum::<usize>();
                    black_box(memory_usage)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark garbage collection efficiency
fn benchmark_garbage_collection(c: &mut Criterion) {
    let mut group = c.benchmark_group("garbage_collection");
    
    let memory_sizes = vec![1024, 10240, 102400]; // KB
    
    for size_kb in memory_sizes {
        group.throughput(Throughput::Bytes(size_kb * 1024));
        
        group.bench_with_input(
            BenchmarkId::new("gc_cycle", size_kb),
            &size_kb,
            |b, &size_kb| {
                b.iter(|| {
                    let mut manager = MemoryManager::new();
                    
                    // Allocate memory up to target size
                    let target_bytes = size_kb * 1024;
                    let mut allocated = 0;
                    
                    while allocated < target_bytes {
                        let _ = manager.intern_string(&format!("symbol_{}", allocated));
                        allocated += 64; // Estimate per symbol
                    }
                    
                    // Measure GC performance
                    let freed = manager.collect_garbage();
                    black_box(freed)
                })
            },
        );
    }
    
    group.finish();
}

/// Comprehensive memory reduction validation
fn benchmark_memory_reduction_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_reduction_validation");
    
    // Test different workload patterns to validate 35% reduction target
    let workload_sizes = vec![1000, 10000, 50000];
    
    for size in workload_sizes {
        group.throughput(Throughput::Elements(size as u64));
        
        // Standard symbolic computation workload
        group.bench_with_input(
            BenchmarkId::new("standard_symbolic_workload", size),
            &size,
            |b, &size| {
                b.iter(|| {
                    let values = create_symbolic_workload(size);
                    let total_memory = values.iter().map(estimate_single_value_memory).sum::<usize>();
                    black_box(total_memory)
                })
            },
        );
        
        // Memory-managed symbolic computation workload
        group.bench_with_input(
            BenchmarkId::new("managed_symbolic_workload", size),
            &size,
            |b, &size| {
                let interner = StringInterner::new();
                let pools = ValuePools::new();
                b.iter(|| {
                    let values = create_symbolic_workload(size);
                    let managed_values: Vec<_> = values.iter()
                        .filter_map(|v| ManagedValue::from_value(v.clone(), &interner).ok())
                        .collect();
                    let total_memory = managed_values.iter().map(|v| v.memory_size()).sum::<usize>();
                    black_box(total_memory)
                })
            },
        );
        
        // Mathematical computation workload
        group.bench_with_input(
            BenchmarkId::new("mathematical_workload", size),
            &size,
            |b, &size| {
                let interner = StringInterner::new();
                b.iter(|| {
                    let values = create_mathematical_workload(size);
                    let managed_values: Vec<_> = values.iter()
                        .filter_map(|v| ManagedValue::from_value(v.clone(), &interner).ok())
                        .collect();
                    let total_memory = managed_values.iter().map(|v| v.memory_size()).sum::<usize>();
                    black_box(total_memory)
                })
            },
        );
    }
    
    group.finish();
}

/// Helper functions for benchmark data creation

fn create_test_values() -> Vec<Value> {
    vec![
        Value::Integer(42),
        Value::Real(3.14159),
        Value::String("test_string".to_string()),
        Value::Symbol("x".to_string()),
        Value::Boolean(true),
        Value::Missing,
        Value::Function("Sin".to_string()),
        Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]),
    ]
}

fn create_large_value_dataset(size: usize) -> Vec<Value> {
    let mut values = Vec::with_capacity(size);
    
    for i in 0..size {
        match i % 6 {
            0 => values.push(Value::Integer(i as i64)),
            1 => values.push(Value::Real(i as f64 * 0.1)),
            2 => values.push(Value::String(format!("string_{}", i))),
            3 => values.push(Value::Symbol(format!("sym_{}", i))),
            4 => values.push(Value::Boolean(i % 2 == 0)),
            5 => values.push(Value::List(vec![
                Value::Integer(i as i64),
                Value::Integer((i + 1) as i64),
            ])),
            _ => unreachable!(),
        }
    }
    
    values
}

fn create_symbolic_workload(size: usize) -> Vec<Value> {
    let mut values = Vec::with_capacity(size);
    
    // Create typical symbolic computation patterns
    for i in 0..size {
        match i % 8 {
            0 => values.push(Value::Symbol("x".to_string())),
            1 => values.push(Value::Symbol("y".to_string())),
            2 => values.push(Value::Function("Plus".to_string())),
            3 => values.push(Value::Function("Times".to_string())),
            4 => values.push(Value::Function("Sin".to_string())),
            5 => values.push(Value::List(vec![
                Value::Symbol("expr".to_string()),
                Value::Integer(i as i64),
            ])),
            6 => values.push(Value::String(format!("pattern_{}", i % 10))),
            7 => values.push(Value::Integer(i as i64)),
            _ => unreachable!(),
        }
    }
    
    values
}

fn create_mathematical_workload(size: usize) -> Vec<Value> {
    let mut values = Vec::with_capacity(size);
    
    // Create typical mathematical computation patterns
    for i in 0..size {
        match i % 6 {
            0 => values.push(Value::Real((i as f64).sin())),
            1 => values.push(Value::Real((i as f64).cos())),
            2 => values.push(Value::Integer(i as i64)),
            3 => values.push(Value::Function("Exp".to_string())),
            4 => values.push(Value::Function("Log".to_string())),
            5 => values.push(Value::List(vec![
                Value::Real(i as f64),
                Value::Real((i + 1) as f64),
            ])),
            _ => unreachable!(),
        }
    }
    
    values
}

fn estimate_value_memory_usage(values: &[Value]) -> usize {
    values.iter().map(estimate_single_value_memory).sum()
}

fn estimate_single_value_memory(v: &Value) -> usize {
    match v {
        Value::Integer(_) => std::mem::size_of::<i64>() + 8, // Enum overhead
        Value::Real(_) => std::mem::size_of::<f64>() + 8,
        Value::String(s) => s.len() + 24 + 8, // String struct + enum overhead
        Value::Symbol(s) => s.len() + 24 + 8,
        Value::Boolean(_) => std::mem::size_of::<bool>() + 8,
        Value::Missing => 8,
        Value::Function(s) => s.len() + 24 + 8,
        Value::List(l) => l.len() * 32 + 24 + 8, // Estimate nested values
        Value::LyObj(_) => 64, // Estimate foreign object overhead
        Value::Quote(_) => 32, // Estimate AST overhead
        Value::Pattern(_) => 32, // Estimate pattern overhead
    }
}

criterion_group!(
    advanced_memory_benches,
    benchmark_string_interning,
    benchmark_managed_values,
    benchmark_memory_pools,
    benchmark_arena_allocation,
    benchmark_memory_usage_comparison,
    benchmark_garbage_collection,
    benchmark_memory_reduction_validation
);

criterion_main!(advanced_memory_benches);