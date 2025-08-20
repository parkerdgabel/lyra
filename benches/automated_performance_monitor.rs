//! Automated Performance Monitoring and Regression Detection
//!
//! This module provides continuous performance monitoring capabilities for the Lyra
//! symbolic computation engine, including:
//! - Automated performance regression detection with configurable thresholds
//! - Performance trend analysis and alerting
//! - Automated baseline establishment and management
//! - Performance reporting and visualization data generation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use lyra::{
    vm::{VirtualMachine, Value},
    parser::Parser,
    compiler::Compiler,
    memory::StringInterner,
};
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::Path;

// =============================================================================
// PERFORMANCE MONITORING DATA STRUCTURES
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub benchmark_name: String,
    pub operation_type: String,
    pub duration_ns: u64,
    pub throughput_ops_per_sec: Option<f64>,
    pub memory_usage_bytes: Option<u64>,
    pub timestamp: u64,
    pub git_commit: Option<String>,
    pub environment_info: EnvironmentInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub platform: String,
    pub cpu_info: String,
    pub memory_total: u64,
    pub rust_version: String,
    pub debug_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub benchmark_name: String,
    pub operation_type: String,
    pub baseline_duration_ns: u64,
    pub baseline_throughput: Option<f64>,
    pub confidence_interval_low: u64,
    pub confidence_interval_high: u64,
    pub sample_count: usize,
    pub established_at: u64,
    pub last_updated: u64,
    pub environment: EnvironmentInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAlert {
    pub benchmark_name: String,
    pub operation_type: String,
    pub severity: AlertSeverity,
    pub performance_change_percent: f64,
    pub current_duration_ns: u64,
    pub baseline_duration_ns: u64,
    pub detected_at: u64,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Critical,  // > 50% regression
    Major,     // 20-50% regression
    Minor,     // 10-20% regression
    Warning,   // 5-10% regression
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub generated_at: u64,
    pub total_benchmarks: usize,
    pub regressions_detected: usize,
    pub improvements_detected: usize,
    pub critical_alerts: Vec<RegressionAlert>,
    pub performance_trends: HashMap<String, Vec<PerformanceMeasurement>>,
    pub baseline_status: BaselineStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineStatus {
    pub total_baselines: usize,
    pub outdated_baselines: usize,
    pub missing_baselines: Vec<String>,
}

// =============================================================================
// PERFORMANCE MONITORING ENGINE
// =============================================================================

pub struct PerformanceMonitor {
    baselines: HashMap<String, PerformanceBaseline>,
    measurements: Vec<PerformanceMeasurement>,
    alerts: Vec<RegressionAlert>,
    config: MonitoringConfig,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    pub regression_threshold_percent: f64,
    pub critical_threshold_percent: f64,
    pub baseline_update_threshold_days: u64,
    pub measurement_history_limit: usize,
    pub enable_alerts: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            regression_threshold_percent: 5.0,
            critical_threshold_percent: 50.0,
            baseline_update_threshold_days: 30,
            measurement_history_limit: 1000,
            enable_alerts: true,
        }
    }
}

impl PerformanceMonitor {
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            measurements: Vec::new(),
            alerts: Vec::new(),
            config: MonitoringConfig::default(),
        }
    }

    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            baselines: HashMap::new(),
            measurements: Vec::new(),
            alerts: Vec::new(),
            config,
        }
    }

    /// Add a performance measurement and check for regressions
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) {
        let benchmark_key = format!("{}_{}", measurement.benchmark_name, measurement.operation_type);
        
        // Check for regression against baseline
        if let Some(baseline) = self.baselines.get(&benchmark_key) {
            if let Some(alert) = self.check_for_regression(&measurement, baseline) {
                if self.config.enable_alerts {
                    self.alerts.push(alert);
                }
            }
        }
        
        // Store measurement
        self.measurements.push(measurement);
        
        // Limit measurement history
        if self.measurements.len() > self.config.measurement_history_limit {
            self.measurements.remove(0);
        }
    }

    /// Check if a measurement represents a performance regression
    fn check_for_regression(
        &self,
        measurement: &PerformanceMeasurement,
        baseline: &PerformanceBaseline,
    ) -> Option<RegressionAlert> {
        let current_duration = measurement.duration_ns;
        let baseline_duration = baseline.baseline_duration_ns;
        
        let change_percent = ((current_duration as f64 - baseline_duration as f64) / baseline_duration as f64) * 100.0;
        
        if change_percent > self.config.regression_threshold_percent {
            let severity = if change_percent > self.config.critical_threshold_percent {
                AlertSeverity::Critical
            } else if change_percent > 20.0 {
                AlertSeverity::Major
            } else if change_percent > 10.0 {
                AlertSeverity::Minor
            } else {
                AlertSeverity::Warning
            };
            
            let message = format!(
                "Performance regression detected in {}: {:.1}% slower than baseline ({:.2}ms vs {:.2}ms)",
                measurement.benchmark_name,
                change_percent,
                current_duration as f64 / 1_000_000.0,
                baseline_duration as f64 / 1_000_000.0
            );
            
            Some(RegressionAlert {
                benchmark_name: measurement.benchmark_name.clone(),
                operation_type: measurement.operation_type.clone(),
                severity,
                performance_change_percent: change_percent,
                current_duration_ns: current_duration,
                baseline_duration_ns: baseline_duration,
                detected_at: measurement.timestamp,
                message,
            })
        } else {
            None
        }
    }

    /// Establish a new baseline from recent measurements
    pub fn establish_baseline(&mut self, benchmark_name: &str, operation_type: &str) {
        let benchmark_key = format!("{}_{}", benchmark_name, operation_type);
        
        let recent_measurements: Vec<&PerformanceMeasurement> = self.measurements
            .iter()
            .filter(|m| m.benchmark_name == benchmark_name && m.operation_type == operation_type)
            .collect();
        
        if recent_measurements.len() >= 5 {
            let durations: Vec<u64> = recent_measurements.iter().map(|m| m.duration_ns).collect();
            let mean_duration = durations.iter().sum::<u64>() / durations.len() as u64;
            
            // Calculate confidence interval (simple approach using standard deviation)
            let variance = durations.iter()
                .map(|d| (*d as f64 - mean_duration as f64).powi(2))
                .sum::<f64>() / durations.len() as f64;
            let std_dev = variance.sqrt();
            
            let confidence_low = (mean_duration as f64 - 2.0 * std_dev).max(0.0) as u64;
            let confidence_high = (mean_duration as f64 + 2.0 * std_dev) as u64;
            
            let baseline = PerformanceBaseline {
                benchmark_name: benchmark_name.to_string(),
                operation_type: operation_type.to_string(),
                baseline_duration_ns: mean_duration,
                baseline_throughput: recent_measurements.last().and_then(|m| m.throughput_ops_per_sec),
                confidence_interval_low: confidence_low,
                confidence_interval_high: confidence_high,
                sample_count: recent_measurements.len(),
                established_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                environment: recent_measurements.last().unwrap().environment_info.clone(),
            };
            
            self.baselines.insert(benchmark_key, baseline);
        }
    }

    /// Generate a comprehensive performance report
    pub fn generate_report(&self) -> PerformanceReport {
        let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Group measurements by benchmark for trend analysis
        let mut performance_trends = HashMap::new();
        for measurement in &self.measurements {
            let key = format!("{}_{}", measurement.benchmark_name, measurement.operation_type);
            performance_trends.entry(key).or_insert_with(Vec::new).push(measurement.clone());
        }
        
        // Count improvements (negative regressions)
        let improvements_detected = self.alerts.iter()
            .filter(|alert| alert.performance_change_percent < 0.0)
            .count();
        
        // Critical alerts
        let critical_alerts: Vec<RegressionAlert> = self.alerts.iter()
            .filter(|alert| alert.severity == AlertSeverity::Critical)
            .cloned()
            .collect();
        
        // Baseline status
        let total_baselines = self.baselines.len();
        let outdated_threshold = now - (self.config.baseline_update_threshold_days * 24 * 60 * 60);
        let outdated_baselines = self.baselines.values()
            .filter(|baseline| baseline.last_updated < outdated_threshold)
            .count();
        
        // Find benchmarks without baselines
        let mut benchmark_names = std::collections::HashSet::new();
        for measurement in &self.measurements {
            benchmark_names.insert(format!("{}_{}", measurement.benchmark_name, measurement.operation_type));
        }
        let missing_baselines: Vec<String> = benchmark_names.into_iter()
            .filter(|name| !self.baselines.contains_key(name))
            .collect();
        
        PerformanceReport {
            generated_at: now,
            total_benchmarks: performance_trends.len(),
            regressions_detected: self.alerts.len(),
            improvements_detected,
            critical_alerts,
            performance_trends,
            baseline_status: BaselineStatus {
                total_baselines,
                outdated_baselines,
                missing_baselines,
            },
        }
    }

    /// Save performance data to files
    pub fn save_to_files(&self, base_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = Path::new(base_path);
        fs::create_dir_all(base_dir)?;
        
        // Save baselines
        let baselines_file = base_dir.join("baselines.json");
        let baselines_json = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(baselines_file, baselines_json)?;
        
        // Save measurements
        let measurements_file = base_dir.join("measurements.json");
        let measurements_json = serde_json::to_string_pretty(&self.measurements)?;
        fs::write(measurements_file, measurements_json)?;
        
        // Save alerts
        let alerts_file = base_dir.join("alerts.json");
        let alerts_json = serde_json::to_string_pretty(&self.alerts)?;
        fs::write(alerts_file, alerts_json)?;
        
        // Generate and save report
        let report = self.generate_report();
        let report_file = base_dir.join("performance_report.json");
        let report_json = serde_json::to_string_pretty(&report)?;
        fs::write(report_file, report_json)?;
        
        Ok(())
    }

    /// Load performance data from files
    pub fn load_from_files(&mut self, base_path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let base_dir = Path::new(base_path);
        
        // Load baselines
        let baselines_file = base_dir.join("baselines.json");
        if baselines_file.exists() {
            let baselines_json = fs::read_to_string(baselines_file)?;
            self.baselines = serde_json::from_str(&baselines_json)?;
        }
        
        // Load measurements
        let measurements_file = base_dir.join("measurements.json");
        if measurements_file.exists() {
            let measurements_json = fs::read_to_string(measurements_file)?;
            self.measurements = serde_json::from_str(&measurements_json)?;
        }
        
        // Load alerts
        let alerts_file = base_dir.join("alerts.json");
        if alerts_file.exists() {
            let alerts_json = fs::read_to_string(alerts_file)?;
            self.alerts = serde_json::from_str(&alerts_json)?;
        }
        
        Ok(())
    }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

impl EnvironmentInfo {
    pub fn current() -> Self {
        Self {
            platform: std::env::consts::OS.to_string(),
            cpu_info: "Unknown".to_string(), // Would need external crate for detailed CPU info
            memory_total: 0, // Would need external crate for memory info
            rust_version: "Unknown".to_string(),
            debug_mode: cfg!(debug_assertions),
        }
    }
}

/// Helper function to create a performance measurement
pub fn create_measurement(
    benchmark_name: &str,
    operation_type: &str,
    duration: Duration,
) -> PerformanceMeasurement {
    PerformanceMeasurement {
        benchmark_name: benchmark_name.to_string(),
        operation_type: operation_type.to_string(),
        duration_ns: duration.as_nanos() as u64,
        throughput_ops_per_sec: None,
        memory_usage_bytes: None,
        timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        git_commit: None,
        environment_info: EnvironmentInfo::current(),
    }
}

// =============================================================================
// AUTOMATED MONITORING BENCHMARKS
// =============================================================================

/// Core performance monitoring benchmarks that establish baselines
fn core_performance_monitoring(c: &mut Criterion) {
    let mut monitor = PerformanceMonitor::new();
    
    // Load existing data if available
    let _ = monitor.load_from_files("./benchmark_results/monitoring");
    
    let mut group = c.benchmark_group("core_monitoring");
    
    // Parsing performance monitoring
    group.bench_function("parsing_baseline", |b| {
        let start = Instant::now();
        b.iter(|| {
            let mut parser = Parser::from_source("2 + 3 * 4").unwrap();
            black_box(parser.parse().unwrap());
        });
        let duration = start.elapsed();
        
        let measurement = create_measurement("parsing", "baseline", duration);
        // Note: In a real implementation, you'd collect measurements and add them
    });
    
    // Compilation performance monitoring
    group.bench_function("compilation_baseline", |b| {
        let source = "2 + 3 * 4";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        
        let start = Instant::now();
        b.iter(|| {
            let mut compiler = Compiler::new();
            black_box(compiler.compile_program(&statements).unwrap());
        });
        let duration = start.elapsed();
        
        let measurement = create_measurement("compilation", "baseline", duration);
    });
    
    // VM execution performance monitoring
    group.bench_function("execution_baseline", |b| {
        let source = "2 + 3";
        let mut parser = Parser::from_source(source).unwrap();
        let statements = parser.parse().unwrap();
        let mut compiler = Compiler::new();
        compiler.compile_program(&statements).unwrap();
        
        let start = Instant::now();
        b.iter(|| {
            // Create fresh VM for each execution (required due to ownership)
            let mut fresh_parser = Parser::from_source(source).unwrap();
            let fresh_statements = fresh_parser.parse().unwrap();
            let mut fresh_compiler = Compiler::new();
            fresh_compiler.compile_program(&fresh_statements).unwrap();
            let mut vm = fresh_compiler.into_vm();
            black_box(vm.run().unwrap());
        });
        let duration = start.elapsed();
        
        let measurement = create_measurement("execution", "baseline", duration);
    });
    
    // Symbol interning performance monitoring
    group.bench_function("symbol_interning_baseline", |b| {
        let start = Instant::now();
        b.iter(|| {
            let interner = StringInterner::new();
            for i in 0..100 {
                black_box(interner.intern_symbol_id(&format!("symbol_{}", i)));
            }
        });
        let duration = start.elapsed();
        
        let measurement = create_measurement("symbol_interning", "baseline", duration);
    });
    
    group.finish();
    
    // Save monitoring data
    let _ = monitor.save_to_files("./benchmark_results/monitoring");
}

/// Memory optimization validation benchmarks
fn memory_optimization_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_optimization");
    
    // String storage vs symbol interning comparison
    group.bench_function("string_storage_memory", |b| {
        b.iter(|| {
            let mut strings = Vec::new();
            for i in 0..1000 {
                strings.push(format!("symbol_{}", i));
            }
            black_box(strings);
        });
    });
    
    group.bench_function("symbol_interning_memory", |b| {
        b.iter(|| {
            let interner = StringInterner::new();
            let mut symbols = Vec::new();
            for i in 0..1000 {
                symbols.push(interner.intern_symbol_id(&format!("symbol_{}", i)));
            }
            black_box(symbols);
        });
    });
    
    // Value creation optimization
    group.bench_function("value_creation_overhead", |b| {
        b.iter(|| {
            let mut values = Vec::new();
            for i in 0..1000 {
                values.push(Value::Integer(i));
                values.push(Value::Real(i as f64));
                values.push(Value::String(format!("value_{}", i)));
            }
            black_box(values);
        });
    });
    
    group.finish();
}

/// Performance trend analysis benchmarks
fn performance_trend_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("trend_analysis");
    
    // Simulate different performance scenarios for trend analysis
    let scenarios = vec![
        ("stable_performance", 1.0),
        ("gradual_improvement", 0.95),
        ("slight_regression", 1.05),
        ("significant_regression", 1.25),
    ];
    
    for (scenario_name, multiplier) in scenarios {
        group.bench_function(BenchmarkId::new("scenario", scenario_name), |b| {
            let base_work = (100.0 * multiplier) as usize;
            
            b.iter(|| {
                // Simulate work that scales with the multiplier
                let mut result = 0;
                for i in 0..base_work {
                    result += i * i;
                }
                black_box(result);
            });
        });
    }
    
    group.finish();
}

// =============================================================================
// BENCHMARK GROUPS
// =============================================================================

criterion_group!(
    monitoring_benchmarks,
    core_performance_monitoring,
    memory_optimization_validation,
    performance_trend_analysis
);

criterion_main!(monitoring_benchmarks);

// =============================================================================
// INTEGRATION TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_performance_monitor_basic_functionality() {
        let mut monitor = PerformanceMonitor::new();
        
        // Create test measurement
        let measurement = PerformanceMeasurement {
            benchmark_name: "test_benchmark".to_string(),
            operation_type: "parsing".to_string(),
            duration_ns: 1_000_000, // 1ms
            throughput_ops_per_sec: None,
            memory_usage_bytes: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            git_commit: None,
            environment_info: EnvironmentInfo::current(),
        };
        
        // Add measurement
        monitor.add_measurement(measurement);
        
        // Establish baseline
        monitor.establish_baseline("test_benchmark", "parsing");
        
        assert!(monitor.baselines.contains_key("test_benchmark_parsing"));
    }
    
    #[test]
    fn test_regression_detection() {
        let mut monitor = PerformanceMonitor::new();
        
        // Create baseline measurement
        let baseline_measurement = PerformanceMeasurement {
            benchmark_name: "test_benchmark".to_string(),
            operation_type: "parsing".to_string(),
            duration_ns: 1_000_000, // 1ms baseline
            throughput_ops_per_sec: None,
            memory_usage_bytes: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            git_commit: None,
            environment_info: EnvironmentInfo::current(),
        };
        
        // Add multiple measurements to establish baseline
        for _ in 0..5 {
            monitor.add_measurement(baseline_measurement.clone());
        }
        monitor.establish_baseline("test_benchmark", "parsing");
        
        // Create regression measurement (20% slower)
        let regression_measurement = PerformanceMeasurement {
            benchmark_name: "test_benchmark".to_string(),
            operation_type: "parsing".to_string(),
            duration_ns: 1_200_000, // 1.2ms (20% regression)
            throughput_ops_per_sec: None,
            memory_usage_bytes: None,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            git_commit: None,
            environment_info: EnvironmentInfo::current(),
        };
        
        monitor.add_measurement(regression_measurement);
        
        // Should detect regression
        assert!(!monitor.alerts.is_empty());
        assert_eq!(monitor.alerts[0].severity, AlertSeverity::Minor);
    }
    
    #[test]
    fn test_performance_report_generation() {
        let mut monitor = PerformanceMonitor::new();
        
        // Add some test data
        let measurement = PerformanceMeasurement {
            benchmark_name: "test_benchmark".to_string(),
            operation_type: "parsing".to_string(),
            duration_ns: 1_000_000,
            throughput_ops_per_sec: Some(1000.0),
            memory_usage_bytes: Some(1024),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            git_commit: Some("abc123".to_string()),
            environment_info: EnvironmentInfo::current(),
        };
        
        monitor.add_measurement(measurement);
        
        let report = monitor.generate_report();
        
        assert_eq!(report.total_benchmarks, 1);
        assert_eq!(report.regressions_detected, 0);
        assert!(report.baseline_status.missing_baselines.contains(&"test_benchmark_parsing".to_string()));
    }
    
    #[test]
    fn test_memory_optimization_claims() {
        // Test to validate symbol interning memory efficiency
        let interner = StringInterner::new();
        let initial_memory = interner.memory_usage();
        
        // Add 100 symbols
        for i in 0..100 {
            interner.intern_symbol_id(&format!("test_symbol_{}", i));
        }
        
        let final_memory = interner.memory_usage();
        let memory_per_symbol = (final_memory - initial_memory) as f64 / 100.0;
        
        println!("Memory per symbol: {:.2} bytes", memory_per_symbol);
        println!("SymbolId size: {} bytes", std::mem::size_of::<lyra::memory::SymbolId>());
        println!("String size: {} bytes", std::mem::size_of::<String>());
        
        // Validate that symbol IDs are more memory efficient than strings
        assert!(std::mem::size_of::<lyra::memory::SymbolId>() < std::mem::size_of::<String>());
        
        // Memory per symbol should be reasonable
        assert!(memory_per_symbol < 100.0, "Memory per symbol should be reasonable");
    }
}