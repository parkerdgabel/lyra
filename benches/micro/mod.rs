//! Micro-benchmarks for specific hot path operations
//! 
//! This module provides focused micro-benchmarks for validating the performance
//! of individual operations that are claimed to have significant optimizations.

pub mod symbol_interning_benchmarks;
pub mod value_operations_benchmarks;
pub mod memory_pool_benchmarks;
pub mod async_operations_benchmarks;
pub mod cache_alignment_benchmarks;