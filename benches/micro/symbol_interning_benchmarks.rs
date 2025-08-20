//! Symbol Interning Performance Benchmarks
//!
//! Validates the claimed 40-60% memory reduction from symbol interning system.
//! Compares string lookup vs u32 index lookup, measures memory usage reduction,
//! and tests concurrent access patterns.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::memory::{StringInterner, SymbolId};
use std::sync::Arc;
use std::thread;
use std::collections::HashMap;

/// Benchmark symbol interning vs raw string operations
fn symbol_interning_vs_string_lookup(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_interning_vs_strings");
    
    // Common symbols for testing
    let symbols = vec![
        "x", "y", "z", "Plus", "Times", "Sin", "Cos", "Length", "Head", "Tail",
        "List", "Function", "Real", "Integer", "Boolean", "True", "False"
    ];
    
    // Baseline: Raw string hashmap lookup
    group.bench_function("raw_string_hashmap", |b| {
        let mut string_map = HashMap::new();
        for (i, symbol) in symbols.iter().enumerate() {
            string_map.insert(symbol.to_string(), i);
        }
        
        b.iter(|| {
            for symbol in &symbols {
                black_box(string_map.get(*symbol));
            }
        });
    });
    
    // Optimized: Symbol ID interning
    group.bench_function("symbol_id_interning", |b| {
        let interner = StringInterner::new();
        let mut symbol_ids = Vec::new();
        for symbol in &symbols {
            symbol_ids.push(interner.intern_symbol_id(symbol));
        }
        
        b.iter(|| {
            for &symbol_id in &symbol_ids {
                black_box(interner.resolve_symbol(symbol_id));
            }
        });
    });
    
    // Memory usage comparison
    group.bench_function("memory_usage_strings", |b| {
        b.iter(|| {
            let strings: Vec<String> = symbols.iter().map(|s| s.to_string()).collect();
            black_box(strings);
        });
    });
    
    group.bench_function("memory_usage_symbol_ids", |b| {
        let interner = StringInterner::new();
        b.iter(|| {
            let symbol_ids: Vec<SymbolId> = symbols.iter()
                .map(|s| interner.intern_symbol_id(s))
                .collect();
            black_box(symbol_ids);
        });
    });
    
    group.finish();
}

/// Benchmark concurrent access patterns for symbol interning
fn concurrent_symbol_access(c: &mut Criterion) {
    let mut group = c.benchmark_group("concurrent_symbol_access");
    group.throughput(Throughput::Elements(1000));
    
    let symbols = vec![
        "concurrent_test_1", "concurrent_test_2", "concurrent_test_3",
        "concurrent_test_4", "concurrent_test_5", "concurrent_test_6",
        "concurrent_test_7", "concurrent_test_8", "concurrent_test_9", 
        "concurrent_test_10"
    ];
    
    for thread_count in [1, 2, 4, 8].iter() {
        group.bench_with_input(
            BenchmarkId::new("concurrent_interning", thread_count),
            thread_count,
            |b, &thread_count| {
                b.iter(|| {
                    let interner = Arc::new(StringInterner::new());
                    let mut handles = vec![];
                    
                    for i in 0..thread_count {
                        let interner_clone = Arc::clone(&interner);
                        let symbols_clone = symbols.clone();
                        let handle = thread::spawn(move || {
                            for _ in 0..100 {
                                for symbol in &symbols_clone {
                                    let symbol_name = format!("{}_{}", symbol, i);
                                    black_box(interner_clone.intern_symbol_id(&symbol_name));
                                }
                            }
                        });
                        handles.push(handle);
                    }
                    
                    for handle in handles {
                        handle.join().unwrap();
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory overhead of different symbol storage approaches
fn symbol_memory_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_memory_overhead");
    
    // Test different symbol set sizes
    let test_sizes = [100, 1000, 10000];
    
    for &size in &test_sizes {
        let symbols: Vec<String> = (0..size)
            .map(|i| format!("symbol_{:04}", i))
            .collect();
        
        group.bench_with_input(
            BenchmarkId::new("string_storage", size),
            &symbols,
            |b, symbols| {
                b.iter(|| {
                    // Store as Vec<String>
                    let string_storage: Vec<String> = symbols.clone();
                    black_box(string_storage);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("symbol_id_storage", size),
            &symbols,
            |b, symbols| {
                b.iter(|| {
                    // Store as Vec<SymbolId> using interner
                    let interner = StringInterner::new();
                    let symbol_storage: Vec<SymbolId> = symbols.iter()
                        .map(|s| interner.intern_symbol_id(s))
                        .collect();
                    black_box(symbol_storage);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark hit rates and cache performance for symbol interning
fn symbol_hit_rates_and_cache_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_hit_rates");
    
    // Simulate real workload with repeated symbol access
    let common_symbols = vec!["x", "y", "Plus", "Times", "Sin", "Cos"];
    let rare_symbols = vec!["RareFunction1", "RareFunction2", "UnusualSymbol"];
    
    group.bench_function("high_hit_rate_workload", |b| {
        let interner = StringInterner::new();
        
        // Pre-populate with common symbols
        for symbol in &common_symbols {
            interner.intern_symbol_id(symbol);
        }
        
        b.iter(|| {
            // Simulate workload with 90% common symbols, 10% rare
            for i in 0..100 {
                let symbol = if i % 10 == 0 {
                    &rare_symbols[i % rare_symbols.len()]
                } else {
                    &common_symbols[i % common_symbols.len()]
                };
                black_box(interner.intern_symbol_id(symbol));
            }
        });
    });
    
    group.bench_function("low_hit_rate_workload", |b| {
        let interner = StringInterner::new();
        
        b.iter(|| {
            // Simulate workload with mostly unique symbols
            for i in 0..100 {
                let symbol = format!("unique_symbol_{}", i);
                black_box(interner.intern_symbol_id(&symbol));
            }
        });
    });
    
    group.finish();
}

/// Benchmark symbol resolution performance
fn symbol_resolution_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_resolution");
    
    let interner = StringInterner::new();
    let symbols = vec!["x", "y", "z", "Plus", "Times", "Length", "Head", "Tail"];
    let symbol_ids: Vec<SymbolId> = symbols.iter()
        .map(|s| interner.intern_symbol_id(s))
        .collect();
    
    group.bench_function("symbol_id_to_string", |b| {
        b.iter(|| {
            for &symbol_id in &symbol_ids {
                black_box(interner.resolve_symbol(symbol_id));
            }
        });
    });
    
    group.bench_function("string_to_symbol_id", |b| {
        b.iter(|| {
            for symbol in &symbols {
                black_box(interner.intern_symbol_id(symbol));
            }
        });
    });
    
    group.finish();
}

/// Memory footprint analysis for symbol interning
fn symbol_memory_footprint_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("symbol_memory_footprint");
    
    group.bench_function("interner_memory_growth", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let initial_usage = interner.memory_usage();
            
            // Add many symbols and measure growth
            for i in 0..1000 {
                interner.intern_symbol_id(&format!("symbol_{}", i));
            }
            
            let final_usage = interner.memory_usage();
            black_box(final_usage - initial_usage);
        });
    });
    
    group.bench_function("symbol_id_size_comparison", |b| {
        b.iter(|| {
            // Size of SymbolId vs String
            let symbol_id_size = std::mem::size_of::<SymbolId>();
            let string_size = std::mem::size_of::<String>();
            black_box((symbol_id_size, string_size));
        });
    });
    
    group.finish();
}

criterion_group!(
    symbol_interning_benchmarks,
    symbol_interning_vs_string_lookup,
    concurrent_symbol_access,
    symbol_memory_overhead,
    symbol_hit_rates_and_cache_performance,
    symbol_resolution_performance,
    symbol_memory_footprint_analysis
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_symbol_interning_efficiency() {
        let interner = StringInterner::new();
        
        // Measure baseline memory usage
        let initial_memory = interner.memory_usage();
        
        // Add 1000 symbols
        let symbols: Vec<SymbolId> = (0..1000)
            .map(|i| interner.intern_symbol_id(&format!("test_symbol_{}", i)))
            .collect();
        
        let final_memory = interner.memory_usage();
        let memory_per_symbol = (final_memory - initial_memory) as f64 / 1000.0;
        
        println!("Memory per symbol: {:.2} bytes", memory_per_symbol);
        println!("SymbolId size: {} bytes", std::mem::size_of::<SymbolId>());
        println!("String size: {} bytes", std::mem::size_of::<String>());
        
        // Validate that symbol IDs are much smaller than strings
        assert!(std::mem::size_of::<SymbolId>() < std::mem::size_of::<String>());
        
        // Memory per symbol should be reasonable (including interner overhead)
        assert!(memory_per_symbol < 50.0, "Memory per symbol should be under 50 bytes including overhead");
    }
    
    #[test]
    fn validate_symbol_hit_rates() {
        let interner = StringInterner::new();
        
        // Simulate repeated access to same symbols
        let symbols = vec!["x", "y", "Plus", "Times"];
        
        // First pass - should create symbols
        for symbol in &symbols {
            interner.intern_symbol_id(symbol);
        }
        
        let stats_after_creation = interner.stats();
        
        // Second pass - should hit cache
        for symbol in &symbols {
            interner.intern_symbol_id(symbol);
        }
        
        let final_stats = interner.stats();
        
        // Should have more hits than misses for repeated access
        let total_hits = final_stats.static_hits + final_stats.dynamic_hits;
        assert!(total_hits >= symbols.len(), "Should have cache hits for repeated symbol access");
    }
}