//! Scientific Computing Workload Simulations
//!
//! Matrix operations, signal processing, and numerical computation workloads
//! that stress the VM's mathematical capabilities and memory management.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::{
    vm::{VirtualMachine, Value},
    parser::Parser,
    compiler::Compiler,
    memory::{MemoryManager, CompactValue, StringInterner},
};
use std::sync::Arc;

/// Generate test matrices of various sizes
fn generate_matrix(rows: usize, cols: usize) -> Value {
    let matrix_data: Vec<Value> = (0..rows).map(|i| {
        let row: Vec<Value> = (0..cols).map(|j| {
            Value::Real((i * cols + j) as f64)
        }).collect();
        Value::List(row)
    }).collect();
    
    Value::List(matrix_data)
}

/// Generate complex matrices for testing
fn generate_complex_matrix(rows: usize, cols: usize) -> Value {
    let matrix_data: Vec<Value> = (0..rows).map(|i| {
        let row: Vec<Value> = (0..cols).map(|j| {
            // Create complex numbers as lists [real, imaginary]
            Value::List(vec![
                Value::Real((i * cols + j) as f64),
                Value::Real(((i + j) % 3) as f64),
            ])
        }).collect();
        Value::List(row)
    }).collect();
    
    Value::List(matrix_data)
}

/// Benchmark matrix operations
fn matrix_operations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_operations");
    
    for size in [10, 50, 100, 200].iter() {
        let matrix_a = generate_matrix(*size, *size);
        let matrix_b = generate_matrix(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("matrix_addition", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate matrix addition
                    if let (Value::List(a_rows), Value::List(b_rows)) = (&matrix_a, &matrix_b) {
                        let mut result_rows = Vec::new();
                        
                        for (a_row, b_row) in a_rows.iter().zip(b_rows.iter()) {
                            if let (Value::List(a_cols), Value::List(b_cols)) = (a_row, b_row) {
                                let mut result_row = Vec::new();
                                
                                for (a_val, b_val) in a_cols.iter().zip(b_cols.iter()) {
                                    if let (Value::Real(a), Value::Real(b)) = (a_val, b_val) {
                                        result_row.push(Value::Real(a + b));
                                    }
                                }
                                
                                result_rows.push(Value::List(result_row));
                            }
                        }
                        
                        black_box(Value::List(result_rows));
                    }
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_multiplication", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate matrix multiplication (simplified)
                    if let (Value::List(a_rows), Value::List(b_rows)) = (&matrix_a, &matrix_b) {
                        let mut result_rows = Vec::new();
                        
                        for a_row in a_rows {
                            if let Value::List(a_cols) = a_row {
                                let mut result_row = Vec::new();
                                
                                // For each column in B
                                for col_idx in 0..a_cols.len() {
                                    let mut sum = 0.0;
                                    
                                    // Dot product
                                    for (row_idx, a_val) in a_cols.iter().enumerate() {
                                        if let Value::Real(a) = a_val {
                                            if let Value::List(b_row) = &b_rows[row_idx] {
                                                if let Value::Real(b) = &b_row[col_idx] {
                                                    sum += a * b;
                                                }
                                            }
                                        }
                                    }
                                    
                                    result_row.push(Value::Real(sum));
                                }
                                
                                result_rows.push(Value::List(result_row));
                            }
                        }
                        
                        black_box(Value::List(result_rows));
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark signal processing operations
fn signal_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_processing");
    
    for size in [1000, 5000, 10000, 20000].iter() {
        // Generate synthetic signal data
        let signal: Vec<Value> = (0..*size).map(|i| {
            let t = i as f64 / *size as f64;
            let amplitude = (2.0 * std::f64::consts::PI * 5.0 * t).sin() + 
                           0.5 * (2.0 * std::f64::consts::PI * 15.0 * t).sin() +
                           0.1 * rand::random::<f64>(); // Add noise
            Value::Real(amplitude)
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("signal_filtering", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simple moving average filter
                    let window_size = 5;
                    let mut filtered = Vec::new();
                    
                    for i in 0..signal.len() {
                        let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
                        let end = std::cmp::min(i + window_size / 2 + 1, signal.len());
                        
                        let mut sum = 0.0;
                        let mut count = 0;
                        
                        for j in start..end {
                            if let Value::Real(val) = &signal[j] {
                                sum += val;
                                count += 1;
                            }
                        }
                        
                        if count > 0 {
                            filtered.push(Value::Real(sum / count as f64));
                        }
                    }
                    
                    black_box(filtered);
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("signal_fft_simulation", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate FFT computation (simplified DFT)
                    let mut fft_result = Vec::new();
                    let n = signal.len();
                    
                    for k in 0..std::cmp::min(n, 100) { // Limit to first 100 frequencies
                        let mut real_sum = 0.0;
                        let mut imag_sum = 0.0;
                        
                        for (i, sample) in signal.iter().enumerate() {
                            if let Value::Real(val) = sample {
                                let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                                real_sum += val * angle.cos();
                                imag_sum += val * angle.sin();
                            }
                        }
                        
                        // Store as complex number [real, imag]
                        fft_result.push(Value::List(vec![
                            Value::Real(real_sum),
                            Value::Real(imag_sum),
                        ]));
                    }
                    
                    black_box(fft_result);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark numerical integration
fn numerical_integration_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("numerical_integration");
    
    // Define functions to integrate
    let functions = vec![
        ("polynomial", |x: f64| x * x + 2.0 * x + 1.0),
        ("trigonometric", |x: f64| x.sin() + x.cos()),
        ("exponential", |x: f64| (-x * x).exp()),
        ("rational", |x: f64| 1.0 / (1.0 + x * x)),
    ];
    
    for (name, func) in functions {
        for steps in [1000, 5000, 10000].iter() {
            group.bench_with_input(
                BenchmarkId::new(format!("{}_integration", name), steps),
                steps,
                |b, &steps| {
                    b.iter(|| {
                        // Trapezoidal rule integration
                        let a = 0.0;
                        let b = 1.0;
                        let h = (b - a) / steps as f64;
                        
                        let mut sum = func(a) + func(b);
                        for i in 1..steps {
                            let x = a + i as f64 * h;
                            sum += 2.0 * func(x);
                        }
                        
                        let result = h * sum / 2.0;
                        black_box(result);
                    });
                },
            );
        }
    }
    
    group.finish();
}

/// Benchmark linear algebra operations
fn linear_algebra_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear_algebra");
    
    for size in [50, 100, 200].iter() {
        let matrix = generate_matrix(*size, *size);
        
        group.bench_with_input(
            BenchmarkId::new("matrix_transpose", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Matrix transpose
                    if let Value::List(rows) = &matrix {
                        let mut transposed = Vec::new();
                        
                        if !rows.is_empty() {
                            if let Value::List(first_row) = &rows[0] {
                                for col_idx in 0..first_row.len() {
                                    let mut new_row = Vec::new();
                                    
                                    for row in rows {
                                        if let Value::List(row_data) = row {
                                            if col_idx < row_data.len() {
                                                new_row.push(row_data[col_idx].clone());
                                            }
                                        }
                                    }
                                    
                                    transposed.push(Value::List(new_row));
                                }
                            }
                        }
                        
                        black_box(Value::List(transposed));
                    }
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("matrix_determinant_simulation", size),
            size,
            |b, _| {
                b.iter(|| {
                    // Simulate determinant calculation (simplified)
                    if let Value::List(rows) = &matrix {
                        let mut det = 1.0;
                        
                        // Simplified: just multiply diagonal elements
                        for (i, row) in rows.iter().enumerate() {
                            if let Value::List(row_data) = row {
                                if i < row_data.len() {
                                    if let Value::Real(val) = &row_data[i] {
                                        det *= val;
                                    }
                                }
                            }
                        }
                        
                        black_box(det);
                    }
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark statistical computations
fn statistical_computations_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_computations");
    
    for size in [1000, 10000, 50000, 100000].iter() {
        // Generate random data
        let data: Vec<Value> = (0..*size).map(|_| {
            Value::Real(rand::random::<f64>() * 100.0)
        }).collect();
        
        group.bench_with_input(
            BenchmarkId::new("descriptive_statistics", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut sum = 0.0;
                    let mut sum_sq = 0.0;
                    let mut min_val = f64::INFINITY;
                    let mut max_val = f64::NEG_INFINITY;
                    let mut count = 0;
                    
                    for value in &data {
                        if let Value::Real(val) = value {
                            sum += val;
                            sum_sq += val * val;
                            min_val = min_val.min(*val);
                            max_val = max_val.max(*val);
                            count += 1;
                        }
                    }
                    
                    let mean = sum / count as f64;
                    let variance = (sum_sq / count as f64) - (mean * mean);
                    let std_dev = variance.sqrt();
                    
                    black_box((mean, variance, std_dev, min_val, max_val));
                });
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("correlation_computation", size),
            size,
            |b, _| {
                // Generate second dataset for correlation
                let data_y: Vec<Value> = (0..*size).map(|i| {
                    // Correlated with data but with noise
                    if let Value::Real(x) = &data[i] {
                        Value::Real(x * 0.8 + rand::random::<f64>() * 20.0)
                    } else {
                        Value::Real(0.0)
                    }
                }).collect();
                
                b.iter(|| {
                    // Compute Pearson correlation
                    let mut sum_x = 0.0;
                    let mut sum_y = 0.0;
                    let mut sum_xy = 0.0;
                    let mut sum_x2 = 0.0;
                    let mut sum_y2 = 0.0;
                    let mut n = 0;
                    
                    for (x_val, y_val) in data.iter().zip(data_y.iter()) {
                        if let (Value::Real(x), Value::Real(y)) = (x_val, y_val) {
                            sum_x += x;
                            sum_y += y;
                            sum_xy += x * y;
                            sum_x2 += x * x;
                            sum_y2 += y * y;
                            n += 1;
                        }
                    }
                    
                    let n_f = n as f64;
                    let numerator = n_f * sum_xy - sum_x * sum_y;
                    let denominator = ((n_f * sum_x2 - sum_x * sum_x) * (n_f * sum_y2 - sum_y * sum_y)).sqrt();
                    
                    let correlation = if denominator != 0.0 {
                        numerator / denominator
                    } else {
                        0.0
                    };
                    
                    black_box(correlation);
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark memory-efficient scientific computing
fn memory_efficient_scientific_computing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficient_scientific");
    
    group.bench_function("compact_matrix_operations", |b| {
        let mut memory_manager = MemoryManager::new();
        
        b.iter(|| {
            let size = 100;
            let mut compact_matrices = Vec::new();
            
            // Create matrices using CompactValue
            for matrix_id in 0..3 {
                let mut matrix_rows = Vec::new();
                
                for i in 0..size {
                    let mut row = Vec::new();
                    for j in 0..size {
                        let value = CompactValue::Real((matrix_id * size * size + i * size + j) as f64);
                        row.push(value);
                    }
                    matrix_rows.push(CompactValue::List(Arc::new(row)));
                }
                
                let matrix = CompactValue::List(Arc::new(matrix_rows));
                compact_matrices.push(memory_manager.alloc_compact_value(matrix));
            }
            
            // Simulate matrix addition
            if compact_matrices.len() >= 2 {
                // Access and process matrices (simplified)
                for matrix in &compact_matrices {
                    // Simulate processing
                    black_box(matrix.memory_size());
                }
            }
            
            // Clean up
            for matrix in &compact_matrices {
                memory_manager.recycle_compact_value(matrix);
            }
            
            black_box(compact_matrices);
        });
    });
    
    group.bench_function("traditional_matrix_operations", |b| {
        b.iter(|| {
            let size = 100;
            let mut matrices = Vec::new();
            
            // Create matrices using traditional Value
            for matrix_id in 0..3 {
                let mut matrix_rows = Vec::new();
                
                for i in 0..size {
                    let mut row = Vec::new();
                    for j in 0..size {
                        row.push(Value::Real((matrix_id * size * size + i * size + j) as f64));
                    }
                    matrix_rows.push(Value::List(row));
                }
                
                matrices.push(Value::List(matrix_rows));
            }
            
            // Simulate matrix processing
            for matrix in &matrices {
                // Access matrix elements (simplified)
                if let Value::List(rows) = matrix {
                    black_box(rows.len());
                }
            }
            
            black_box(matrices);
        });
    });
    
    group.finish();
}

criterion_group!(
    scientific_computing_benchmarks,
    matrix_operations_benchmark,
    signal_processing_benchmark,
    numerical_integration_benchmark,
    linear_algebra_benchmark,
    statistical_computations_benchmark,
    memory_efficient_scientific_computing
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_matrix_generation() {
        let matrix = generate_matrix(3, 3);
        
        if let Value::List(rows) = matrix {
            assert_eq!(rows.len(), 3);
            
            for row in rows {
                if let Value::List(cols) = row {
                    assert_eq!(cols.len(), 3);
                    for col in cols {
                        assert!(matches!(col, Value::Real(_)));
                    }
                }
            }
        } else {
            panic!("Expected matrix to be a list");
        }
    }
    
    #[test]
    fn validate_matrix_operations() {
        let matrix_a = generate_matrix(2, 2);
        let matrix_b = generate_matrix(2, 2);
        
        // Test matrix addition
        if let (Value::List(a_rows), Value::List(b_rows)) = (&matrix_a, &matrix_b) {
            let mut result_rows = Vec::new();
            
            for (a_row, b_row) in a_rows.iter().zip(b_rows.iter()) {
                if let (Value::List(a_cols), Value::List(b_cols)) = (a_row, b_row) {
                    let mut result_row = Vec::new();
                    
                    for (a_val, b_val) in a_cols.iter().zip(b_cols.iter()) {
                        if let (Value::Real(a), Value::Real(b)) = (a_val, b_val) {
                            result_row.push(Value::Real(a + b));
                        }
                    }
                    
                    result_rows.push(Value::List(result_row));
                }
            }
            
            let result = Value::List(result_rows);
            
            // Verify result structure
            if let Value::List(result_rows) = result {
                assert_eq!(result_rows.len(), 2);
                for row in result_rows {
                    if let Value::List(cols) = row {
                        assert_eq!(cols.len(), 2);
                    }
                }
            }
        }
    }
    
    #[test]
    fn validate_signal_processing() {
        let signal_size = 1000;
        let signal: Vec<Value> = (0..signal_size).map(|i| {
            let t = i as f64 / signal_size as f64;
            let amplitude = (2.0 * std::f64::consts::PI * 5.0 * t).sin();
            Value::Real(amplitude)
        }).collect();
        
        // Test moving average filter
        let window_size = 5;
        let mut filtered = Vec::new();
        
        for i in 0..signal.len() {
            let start = if i >= window_size / 2 { i - window_size / 2 } else { 0 };
            let end = std::cmp::min(i + window_size / 2 + 1, signal.len());
            
            let mut sum = 0.0;
            let mut count = 0;
            
            for j in start..end {
                if let Value::Real(val) = &signal[j] {
                    sum += val;
                    count += 1;
                }
            }
            
            if count > 0 {
                filtered.push(Value::Real(sum / count as f64));
            }
        }
        
        assert_eq!(filtered.len(), signal.len());
        
        // Filtered signal should have same structure
        for sample in filtered {
            assert!(matches!(sample, Value::Real(_)));
        }
    }
    
    #[test]
    fn validate_statistical_computations() {
        let data: Vec<Value> = vec![
            Value::Real(1.0),
            Value::Real(2.0),
            Value::Real(3.0),
            Value::Real(4.0),
            Value::Real(5.0),
        ];
        
        // Compute mean
        let mut sum = 0.0;
        let mut count = 0;
        
        for value in &data {
            if let Value::Real(val) = value {
                sum += val;
                count += 1;
            }
        }
        
        let mean = sum / count as f64;
        assert_eq!(mean, 3.0); // Mean of 1,2,3,4,5 is 3
        
        // Compute variance
        let mut sum_sq_diff = 0.0;
        for value in &data {
            if let Value::Real(val) = value {
                let diff = val - mean;
                sum_sq_diff += diff * diff;
            }
        }
        
        let variance = sum_sq_diff / count as f64;
        assert_eq!(variance, 2.0); // Variance of 1,2,3,4,5 is 2
    }
    
    #[test]
    fn validate_numerical_integration() {
        // Test trapezoidal rule on simple function: f(x) = x^2
        let func = |x: f64| x * x;
        let a = 0.0;
        let b = 1.0;
        let steps = 1000;
        let h = (b - a) / steps as f64;
        
        let mut sum = func(a) + func(b);
        for i in 1..steps {
            let x = a + i as f64 * h;
            sum += 2.0 * func(x);
        }
        
        let result = h * sum / 2.0;
        
        // Analytical result for integral of x^2 from 0 to 1 is 1/3
        let expected = 1.0 / 3.0;
        let error = (result - expected).abs();
        
        println!("Numerical: {}, Analytical: {}, Error: {}", result, expected, error);
        
        // Should be reasonably close with 1000 steps
        assert!(error < 0.001);
    }
}