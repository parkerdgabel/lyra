//! Comprehensive Workload Simulations Suite
//!
//! This benchmark suite provides real-world workload simulations that reflect
//! actual usage patterns for symbolic computation systems like Lyra.

use criterion::criterion_main;

mod workloads;

use workloads::mathematical_computation::mathematical_computation_benchmarks;
use workloads::data_processing::data_processing_benchmarks;
use workloads::scientific_computing::scientific_computing_benchmarks;
use workloads::concurrent_algorithms::concurrent_algorithms_benchmarks;
use workloads::stream_processing::stream_processing_benchmarks;

criterion_main!(
    mathematical_computation_benchmarks,
    data_processing_benchmarks,
    scientific_computing_benchmarks,
    concurrent_algorithms_benchmarks,
    stream_processing_benchmarks
);