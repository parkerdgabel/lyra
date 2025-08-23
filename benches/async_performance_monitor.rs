//! Async Performance Monitor
//!
//! Comprehensive performance monitoring and analysis for Lyra's async concurrency system.
//! Provides automated performance validation, regression detection, and optimization guidance.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::fs::File;
use std::io::Write;
use lyra::vm::{Value, VmResult};
use lyra::stdlib::async_ops::{thread_pool, channel, bounded_channel, promise, send, receive, await_future};

/// Performance metric collection
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub mean_duration: Duration,
    pub median_duration: Duration,
    pub p95_duration: Duration,
    pub p99_duration: Duration,
    pub throughput_ops_per_sec: f64,
    pub memory_overhead_mb: f64,
    pub success_rate: f64,
    pub samples: usize,
}

/// Performance monitor for async operations
pub struct AsyncPerformanceMonitor {
    metrics: HashMap<String, Vec<Duration>>,
    errors: HashMap<String, usize>,
    start_time: Instant,
}

impl AsyncPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
            errors: HashMap::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Run comprehensive performance analysis
    pub fn run_comprehensive_analysis(&mut self) -> Result<Vec<PerformanceMetrics>, Box<dyn std::error::Error>> {
        println!("🚀 Starting Comprehensive Async Performance Analysis");
        
        // Run individual performance tests
        self.test_threadpool_performance()?;
        self.test_channel_performance()?;
        self.test_future_performance()?;
        self.test_concurrent_patterns_performance()?;
        self.test_scalability_performance()?;
        self.test_memory_efficiency()?;
        self.test_latency_characteristics()?;
        self.test_error_handling_performance()?;
        
        // Generate comprehensive metrics
        let metrics = self.calculate_metrics();
        
        // Generate performance report
        self.generate_performance_report(&metrics)?;
        
        Ok(metrics)
    }
    
    /// Test ThreadPool performance characteristics
    fn test_threadpool_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing ThreadPool Performance");
        
        // Test creation performance
        self.measure_operation("threadpool_creation_default", 100, || {
            thread_pool(&[])
        })?;
        
        for worker_count in [1, 2, 4, 8, 16] {
            let operation_name = format!("threadpool_creation_{}_workers", worker_count);
            self.measure_operation(&operation_name, 50, || {
                thread_pool(&[Value::Integer(worker_count)])
            })?;
        }
        
        // Test worker count query performance
        let pool = thread_pool(&[Value::Integer(4)])?;
        if let Value::LyObj(pool_obj) = &pool {
            self.measure_operation("threadpool_worker_count_query", 1000, || {
                pool_obj.call_method("workerCount", &[])
            })?;
        }
        
        Ok(())
    }
    
    /// Test Channel performance characteristics
    fn test_channel_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Channel Performance");
        
        // Unbounded channel performance
        self.measure_operation("channel_creation_unbounded", 1000, || {
            channel(&[])
        })?;
        
        // Bounded channel performance
        for capacity in [10, 100, 1000, 10000] {
            let operation_name = format!("channel_creation_bounded_{}", capacity);
            self.measure_operation(&operation_name, 100, || {
                bounded_channel(&[Value::Integer(capacity)])
            })?;
        }
        
        // Send/Receive throughput tests
        let ch = channel(&[])?;
        
        // Single message send/receive latency
        self.measure_operation("channel_single_send_receive", 1000, || {
            let message = Value::Integer(42);
            send(&[ch.clone(), message]).and_then(|_| {
                receive(&[ch.clone()])
            })
        })?;
        
        // Batch send/receive throughput
        for batch_size in [10, 100, 1000] {
            let operation_name = format!("channel_batch_send_receive_{}", batch_size);
            self.measure_operation(&operation_name, 10, || {
                for i in 0..batch_size {
                    let message = Value::Integer(i);
                    send(&[ch.clone(), message])?;
                }
                for _ in 0..batch_size {
                    receive(&[ch.clone()])?;
                }
                Ok(Value::Integer(batch_size))
            })?;
        }
        
        Ok(())
    }
    
    /// Test Future/Promise performance characteristics
    fn test_future_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Future/Promise Performance");
        
        // Promise creation and immediate resolution
        self.measure_operation("promise_create_await", 1000, || {
            let value = Value::Integer(42);
            let future = promise(&[value])?;
            await_future(&[future])
        })?;
        
        // Different value types
        let test_values = vec![
            ("integer", Value::Integer(42)),
            ("string", Value::String("test_string".to_string())),
            ("list", Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])),
        ];
        
        for (value_type, value) in test_values {
            let operation_name = format!("promise_create_await_{}", value_type);
            self.measure_operation(&operation_name, 100, || {
                let future = promise(&[value.clone()])?;
                await_future(&[future])
            })?;
        }
        
        // Batch future operations
        for batch_size in [10, 50, 100] {
            let operation_name = format!("promise_batch_{}", batch_size);
            self.measure_operation(&operation_name, 10, || {
                let mut futures = Vec::new();
                
                // Create all futures
                for i in 0..batch_size {
                    let value = Value::Integer(i);
                    let future = promise(&[value])?;
                    futures.push(future);
                }
                
                // Await all futures
                let mut results = Vec::new();
                for future in futures {
                    let result = await_future(&[future])?;
                    results.push(result);
                }
                
                Ok(Value::List(results))
            })?;
        }
        
        Ok(())
    }
    
    /// Test concurrent usage patterns
    fn test_concurrent_patterns_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Concurrent Patterns Performance");
        
        // Producer-Consumer pattern
        for queue_size in [10, 100, 1000] {
            let operation_name = format!("producer_consumer_pattern_{}", queue_size);
            self.measure_operation(&operation_name, 10, || {
                let ch = bounded_channel(&[Value::Integer(queue_size)])?;
                
                // Producer phase
                let producer_items = queue_size / 2;
                for i in 0..producer_items {
                    let message = Value::Integer(i);
                    send(&[ch.clone(), message])?;
                }
                
                // Consumer phase
                let mut consumed = Vec::new();
                for _ in 0..producer_items {
                    let received = receive(&[ch.clone()])?;
                    consumed.push(received);
                }
                
                Ok(Value::List(consumed))
            })?;
        }
        
        // Multi-channel coordination
        self.measure_operation("multi_channel_coordination", 50, || {
            let ch1 = channel(&[])?;
            let ch2 = channel(&[])?;
            let ch3 = channel(&[])?;
            
            // Send messages to multiple channels
            for i in 0..10 {
                let message = Value::Integer(i);
                send(&[ch1.clone(), message.clone()])?;
                send(&[ch2.clone(), message.clone()])?;
                send(&[ch3.clone(), message])?;
            }
            
            // Receive from all channels
            let mut results = Vec::new();
            for _ in 0..10 {
                results.push(receive(&[ch1.clone()])?);
                results.push(receive(&[ch2.clone()])?);
                results.push(receive(&[ch3.clone()])?);
            }
            
            Ok(Value::List(results))
        })?;
        
        Ok(())
    }
    
    /// Test scalability with different configurations
    fn test_scalability_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Scalability Performance");
        
        // ThreadPool scalability
        let baseline_time = {
            let start = Instant::now();
            let _pool = thread_pool(&[Value::Integer(1)])?;
            // Simulate some work
            for _ in 0..100 {
                let ch = channel(&[])?;
                let message = Value::Integer(42);
                send(&[ch.clone(), message])?;
                receive(&[ch])?;
            }
            start.elapsed()
        };
        
        for worker_count in [2, 4, 8, 16] {
            let operation_name = format!("scalability_test_{}_workers", worker_count);
            let worker_count_val = worker_count;
            self.measure_operation(&operation_name, 5, || {
                let _pool = thread_pool(&[Value::Integer(worker_count_val)])?;
                // Same work as baseline
                for _ in 0..100 {
                    let ch = channel(&[])?;
                    let message = Value::Integer(42);
                    send(&[ch.clone(), message])?;
                    receive(&[ch])?;
                }
                Ok(Value::Integer(worker_count_val))
            })?;
        }
        
        println!("Baseline time (1 worker): {:?}", baseline_time);
        
        Ok(())
    }
    
    /// Test memory efficiency
    fn test_memory_efficiency(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Memory Efficiency");
        
        // Memory overhead for channels
        self.measure_operation("memory_channel_overhead", 10, || {
            let mut channels = Vec::new();
            
            // Create many channels
            for _ in 0..1000 {
                let ch = channel(&[])?;
                channels.push(ch);
            }
            
            // Use each channel once
            for (i, ch) in channels.iter().enumerate() {
                let message = Value::Integer(i as i64);
                send(&[ch.clone(), message])?;
                receive(&[ch.clone()])?;
            }
            
            Ok(Value::Integer(channels.len() as i64))
        })?;
        
        // Memory overhead for futures
        self.measure_operation("memory_future_overhead", 10, || {
            let mut futures = Vec::new();
            
            // Create many futures
            for i in 0..1000 {
                let value = Value::Integer(i);
                let future = promise(&[value])?;
                futures.push(future);
            }
            
            // Await all futures
            let mut results = Vec::new();
            for future in futures {
                let result = await_future(&[future])?;
                results.push(result);
            }
            
            Ok(Value::List(results))
        })?;
        
        Ok(())
    }
    
    /// Test latency characteristics
    fn test_latency_characteristics(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Latency Characteristics");
        
        // Minimum latency test
        self.measure_operation("minimum_latency_channel", 1000, || {
            let ch = channel(&[])?;
            let message = Value::Integer(42);
            send(&[ch.clone(), message])?;
            receive(&[ch])
        })?;
        
        self.measure_operation("minimum_latency_future", 1000, || {
            let value = Value::Integer(42);
            let future = promise(&[value])?;
            await_future(&[future])
        })?;
        
        // Latency under load
        for concurrent_ops in [10, 50, 100] {
            let operation_name = format!("latency_under_load_{}", concurrent_ops);
            self.measure_operation(&operation_name, 20, || {
                let mut channels = Vec::new();
                
                // Create multiple channels
                for _ in 0..concurrent_ops {
                    let ch = channel(&[])?;
                    channels.push(ch);
                }
                
                // Perform operations on all channels
                for (i, ch) in channels.iter().enumerate() {
                    let message = Value::Integer(i as i64);
                    send(&[ch.clone(), message])?;
                    receive(&[ch.clone()])?;
                }
                
                Ok(Value::Integer(concurrent_ops))
            })?;
        }
        
        Ok(())
    }
    
    /// Test error handling performance
    fn test_error_handling_performance(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        println!("📊 Testing Error Handling Performance");
        
        // Channel overflow error handling
        self.measure_operation("channel_overflow_error_handling", 100, || {
            let ch = bounded_channel(&[Value::Integer(1)])?;
            
            // Fill the channel
            let message1 = Value::Integer(1);
            send(&[ch.clone(), message1])?;
            
            // Try to overflow (should handle error gracefully)
            let message2 = Value::Integer(2);
            let _result = send(&[ch.clone(), message2]); // May fail, that's expected
            
            // Clean up
            receive(&[ch.clone()])?;
            
            Ok(Value::Integer(1))
        })?;
        
        Ok(())
    }
    
    /// Helper method to measure operation performance
    fn measure_operation<F, T>(&mut self, operation_name: &str, iterations: usize, mut operation: F) -> Result<(), Box<dyn std::error::Error>> 
    where
        F: FnMut() -> Result<T, Box<dyn std::error::Error>>,
    {
        let mut durations = Vec::with_capacity(iterations);
        let mut error_count = 0;
        
        for _ in 0..iterations {
            let start = Instant::now();
            match operation() {
                Ok(_) => {
                    durations.push(start.elapsed());
                }
                Err(_) => {
                    error_count += 1;
                }
            }
        }
        
        self.metrics.insert(operation_name.to_string(), durations);
        if error_count > 0 {
            self.errors.insert(operation_name.to_string(), error_count);
        }
        
        Ok(())
    }
    
    /// Calculate comprehensive metrics from collected data
    fn calculate_metrics(&self) -> Vec<PerformanceMetrics> {
        let mut metrics = Vec::new();
        
        for (operation, durations) in &self.metrics {
            if durations.is_empty() {
                continue;
            }
            
            let mut sorted_durations = durations.clone();
            sorted_durations.sort();
            
            let mean = Duration::from_nanos(
                (sorted_durations.iter().map(|d| d.as_nanos()).sum::<u128>() / durations.len() as u128) as u64
            );
            
            let median = sorted_durations[durations.len() / 2];
            let p95 = sorted_durations[(durations.len() as f64 * 0.95) as usize];
            let p99 = sorted_durations[(durations.len() as f64 * 0.99) as usize];
            
            let throughput = if mean.as_secs_f64() > 0.0 {
                1.0 / mean.as_secs_f64()
            } else {
                0.0
            };
            
            let error_count = self.errors.get(operation).unwrap_or(&0);
            let success_rate = (durations.len() as f64) / (durations.len() + error_count) as f64 * 100.0;
            
            metrics.push(PerformanceMetrics {
                operation: operation.clone(),
                mean_duration: mean,
                median_duration: median,
                p95_duration: p95,
                p99_duration: p99,
                throughput_ops_per_sec: throughput,
                memory_overhead_mb: 0.0, // TODO: Implement memory measurement
                success_rate,
                samples: durations.len(),
            });
        }
        
        metrics
    }
    
    /// Generate comprehensive performance report
    fn generate_performance_report(&self, metrics: &[PerformanceMetrics]) -> Result<(), Box<dyn std::error::Error>> {
        let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S").to_string();
        let filename = format!("async_performance_report_{}.md", timestamp);
        
        let mut file = File::create(&filename)?;
        
        writeln!(file, "# Lyra Async Performance Report")?;
        writeln!(file, "*Generated on: {}*\n", chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"))?;
        
        writeln!(file, "## Executive Summary\n")?;
        writeln!(file, "- **Total Operations Tested**: {}", metrics.len())?;
        writeln!(file, "- **Test Duration**: {:?}", self.start_time.elapsed())?;
        writeln!(file, "- **Total Errors**: {}\n", self.errors.values().sum::<usize>())?;
        
        writeln!(file, "## Performance Results\n")?;
        writeln!(file, "| Operation | Mean (μs) | Median (μs) | P95 (μs) | P99 (μs) | Throughput (ops/s) | Success Rate (%) |")?;
        writeln!(file, "|-----------|-----------|-------------|----------|----------|-------------------|------------------|")?;
        
        for metric in metrics {
            writeln!(file, "| {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |",
                metric.operation,
                metric.mean_duration.as_micros() as f64 / 1000.0,
                metric.median_duration.as_micros() as f64 / 1000.0,
                metric.p95_duration.as_micros() as f64 / 1000.0,
                metric.p99_duration.as_micros() as f64 / 1000.0,
                metric.throughput_ops_per_sec,
                metric.success_rate
            )?;
        }
        
        // Performance analysis
        writeln!(file, "\n## Performance Analysis\n")?;
        
        // Identify best performing operations
        let mut sorted_by_throughput = metrics.to_vec();
        sorted_by_throughput.sort_by(|a, b| b.throughput_ops_per_sec.partial_cmp(&a.throughput_ops_per_sec).unwrap());
        
        writeln!(file, "### Top 5 Highest Throughput Operations:\n")?;
        for (i, metric) in sorted_by_throughput.iter().take(5).enumerate() {
            writeln!(file, "{}. **{}**: {:.2} ops/s", i + 1, metric.operation, metric.throughput_ops_per_sec)?;
        }
        
        // Identify operations with concerning latency
        let high_latency_ops: Vec<_> = metrics.iter()
            .filter(|m| m.p99_duration.as_millis() > 10)
            .collect();
        
        if !high_latency_ops.is_empty() {
            writeln!(file, "\n### Operations with High P99 Latency (>10ms):\n")?;
            for metric in high_latency_ops {
                writeln!(file, "- **{}**: {:.2}ms", metric.operation, metric.p99_duration.as_millis())?;
            }
        }
        
        // Error analysis
        if !self.errors.is_empty() {
            writeln!(file, "\n### Error Analysis:\n")?;
            for (operation, error_count) in &self.errors {
                writeln!(file, "- **{}**: {} errors", operation, error_count)?;
            }
        }
        
        writeln!(file, "\n## Recommendations\n")?;
        writeln!(file, "- Monitor high-latency operations for optimization opportunities")?;
        writeln!(file, "- Investigate operations with low success rates")?;
        writeln!(file, "- Consider caching for frequently used operations")?;
        writeln!(file, "- Profile memory usage for operations with high overhead")?;
        
        println!("📊 Performance report generated: {}", filename);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_monitor() {
        let mut monitor = AsyncPerformanceMonitor::new();
        
        // Test a simple measurement
        monitor.measure_operation("test_operation", 10, || {
            let ch = channel(&[])?;
            let message = Value::Integer(42);
            send(&[ch.clone(), message])?;
            receive(&[ch])
        }).unwrap();
        
        let metrics = monitor.calculate_metrics();
        assert!(!metrics.is_empty());
        
        let test_metric = metrics.iter().find(|m| m.operation == "test_operation").unwrap();
        assert_eq!(test_metric.samples, 10);
        assert!(test_metric.success_rate >= 90.0);
    }
    
    #[test]
    fn test_metrics_calculation() {
        let mut monitor = AsyncPerformanceMonitor::new();
        
        // Add some test data
        let durations = vec![
            Duration::from_millis(1),
            Duration::from_millis(2),
            Duration::from_millis(3),
            Duration::from_millis(4),
            Duration::from_millis(5),
        ];
        
        monitor.metrics.insert("test".to_string(), durations);
        
        let metrics = monitor.calculate_metrics();
        let test_metric = &metrics[0];
        
        assert_eq!(test_metric.operation, "test");
        assert_eq!(test_metric.median_duration, Duration::from_millis(3));
        assert!(test_metric.throughput_ops_per_sec > 0.0);
    }
}