//! Comprehensive Micro-Benchmarks Suite
//!
//! This benchmark suite provides focused micro-benchmarks for validating
//! performance claims about hot path operations in Lyra.

use criterion::criterion_main;

mod micro;

use micro::symbol_interning_benchmarks::symbol_interning_benchmarks;
use micro::value_operations_benchmarks::value_operations_benchmarks;
use micro::memory_pool_benchmarks::memory_pool_benchmarks;
use micro::async_operations_benchmarks::async_operations_benchmarks;
use micro::cache_alignment_benchmarks::cache_alignment_benchmarks;

criterion_main!(
    symbol_interning_benchmarks,
    value_operations_benchmarks,
    memory_pool_benchmarks,
    async_operations_benchmarks,
    cache_alignment_benchmarks
);