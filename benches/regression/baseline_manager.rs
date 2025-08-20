//! Baseline Performance Management
//!
//! Manages performance baselines, stores historical data, and provides
//! reference points for regression detection.

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;
use std::fs;
use std::time::{SystemTime, UNIX_EPOCH};

/// Performance measurement data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub benchmark_name: String,
    pub duration_ns: u64,
    pub throughput: Option<f64>,
    pub memory_usage: Option<usize>,
    pub timestamp: u64,
    pub commit_hash: Option<String>,
    pub environment: EnvironmentInfo,
}

/// System environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub os: String,
    pub cpu_model: String,
    pub cpu_cores: usize,
    pub memory_total: usize,
    pub rust_version: String,
    pub build_type: String, // debug, release
}

/// Performance baseline for a specific benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    pub benchmark_name: String,
    pub baseline_duration_ns: u64,
    pub baseline_throughput: Option<f64>,
    pub baseline_memory: Option<usize>,
    pub confidence_interval: (f64, f64), // 95% confidence interval
    pub sample_count: usize,
    pub established_at: u64,
    pub last_updated: u64,
    pub environment: EnvironmentInfo,
}

/// Statistical performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentile_95: f64,
    pub percentile_99: f64,
}

/// Manages performance baselines and historical data
pub struct BaselineManager {
    baselines: HashMap<String, PerformanceBaseline>,
    historical_data: Vec<PerformanceMeasurement>,
    baseline_file: String,
    history_file: String,
}

impl BaselineManager {
    /// Create a new baseline manager
    pub fn new(baseline_file: &str, history_file: &str) -> Self {
        let mut manager = Self {
            baselines: HashMap::new(),
            historical_data: Vec::new(),
            baseline_file: baseline_file.to_string(),
            history_file: history_file.to_string(),
        };
        
        manager.load_baselines();
        manager.load_historical_data();
        manager
    }
    
    /// Load existing baselines from file
    fn load_baselines(&mut self) {
        if Path::new(&self.baseline_file).exists() {
            if let Ok(content) = fs::read_to_string(&self.baseline_file) {
                if let Ok(baselines) = serde_json::from_str::<HashMap<String, PerformanceBaseline>>(&content) {
                    self.baselines = baselines;
                    println!("Loaded {} performance baselines", self.baselines.len());
                }
            }
        }
    }
    
    /// Load historical performance data
    fn load_historical_data(&mut self) {
        if Path::new(&self.history_file).exists() {
            if let Ok(content) = fs::read_to_string(&self.history_file) {
                if let Ok(data) = serde_json::from_str::<Vec<PerformanceMeasurement>>(&content) {
                    self.historical_data = data;
                    println!("Loaded {} historical measurements", self.historical_data.len());
                }
            }
        }
    }
    
    /// Save baselines to file
    pub fn save_baselines(&self) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.baselines)?;
        fs::write(&self.baseline_file, content)?;
        Ok(())
    }
    
    /// Save historical data to file
    pub fn save_historical_data(&self) -> Result<(), Box<dyn std::error::Error>> {
        let content = serde_json::to_string_pretty(&self.historical_data)?;
        fs::write(&self.history_file, content)?;
        Ok(())
    }
    
    /// Add a new performance measurement
    pub fn add_measurement(&mut self, measurement: PerformanceMeasurement) {
        self.historical_data.push(measurement);
        
        // Keep only last 10000 measurements to prevent unbounded growth
        if self.historical_data.len() > 10000 {
            self.historical_data.drain(0..self.historical_data.len() - 10000);
        }
    }
    
    /// Establish a new baseline for a benchmark
    pub fn establish_baseline(&mut self, benchmark_name: &str, min_samples: usize) -> Result<(), String> {
        let recent_measurements: Vec<&PerformanceMeasurement> = self.historical_data.iter()
            .filter(|m| m.benchmark_name == benchmark_name)
            .rev()
            .take(min_samples * 2) // Take more samples for better statistics
            .collect();
        
        if recent_measurements.len() < min_samples {
            return Err(format!("Not enough samples for {}: {} < {}", 
                              benchmark_name, recent_measurements.len(), min_samples));
        }
        
        let durations: Vec<f64> = recent_measurements.iter()
            .map(|m| m.duration_ns as f64)
            .collect();
        
        let stats = self.calculate_stats(&durations);
        
        // Calculate 95% confidence interval
        let margin_of_error = 1.96 * stats.std_dev / (durations.len() as f64).sqrt();
        let confidence_interval = (stats.mean - margin_of_error, stats.mean + margin_of_error);
        
        let baseline = PerformanceBaseline {
            benchmark_name: benchmark_name.to_string(),
            baseline_duration_ns: stats.median as u64,
            baseline_throughput: recent_measurements.first().and_then(|m| m.throughput),
            baseline_memory: recent_measurements.first().and_then(|m| m.memory_usage),
            confidence_interval,
            sample_count: recent_measurements.len(),
            established_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            environment: recent_measurements.first().unwrap().environment.clone(),
        };
        
        self.baselines.insert(benchmark_name.to_string(), baseline);
        println!("Established baseline for {}: {:.2}ns ±{:.2}ns", 
                benchmark_name, stats.median, margin_of_error);
        
        Ok(())
    }
    
    /// Get baseline for a benchmark
    pub fn get_baseline(&self, benchmark_name: &str) -> Option<&PerformanceBaseline> {
        self.baselines.get(benchmark_name)
    }
    
    /// Calculate statistical summary
    fn calculate_stats(&self, values: &[f64]) -> PerformanceStats {
        if values.is_empty() {
            return PerformanceStats {
                mean: 0.0, median: 0.0, std_dev: 0.0, min: 0.0, max: 0.0,
                percentile_95: 0.0, percentile_99: 0.0,
            };
        }
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = sorted_values[sorted_values.len() / 2];
        
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        
        let min = sorted_values[0];
        let max = sorted_values[sorted_values.len() - 1];
        
        let percentile_95 = sorted_values[(sorted_values.len() as f64 * 0.95) as usize];
        let percentile_99 = sorted_values[(sorted_values.len() as f64 * 0.99) as usize];
        
        PerformanceStats {
            mean, median, std_dev, min, max, percentile_95, percentile_99,
        }
    }
    
    /// Get performance trend for a benchmark
    pub fn get_performance_trend(&self, benchmark_name: &str, days: u64) -> Vec<&PerformanceMeasurement> {
        let cutoff_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - (days * 24 * 3600);
        
        self.historical_data.iter()
            .filter(|m| m.benchmark_name == benchmark_name && m.timestamp >= cutoff_time)
            .collect()
    }
    
    /// Update baseline with new measurements
    pub fn update_baseline(&mut self, benchmark_name: &str) -> Result<(), String> {
        self.establish_baseline(benchmark_name, 10) // Minimum 10 samples for update
    }
    
    /// Get summary of all baselines
    pub fn get_baseline_summary(&self) -> HashMap<String, (u64, f64)> {
        self.baselines.iter()
            .map(|(name, baseline)| {
                let margin = baseline.confidence_interval.1 - baseline.confidence_interval.0;
                (name.clone(), (baseline.baseline_duration_ns, margin))
            })
            .collect()
    }
    
    /// Clean old historical data
    pub fn clean_old_data(&mut self, days_to_keep: u64) {
        let cutoff_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() - (days_to_keep * 24 * 3600);
        
        let initial_count = self.historical_data.len();
        self.historical_data.retain(|m| m.timestamp >= cutoff_time);
        let removed_count = initial_count - self.historical_data.len();
        
        if removed_count > 0 {
            println!("Cleaned {} old measurements (older than {} days)", removed_count, days_to_keep);
        }
    }
    
    /// Get current system environment info
    pub fn get_current_environment() -> EnvironmentInfo {
        EnvironmentInfo {
            os: std::env::consts::OS.to_string(),
            cpu_model: "Unknown".to_string(), // Would need platform-specific code
            cpu_cores: num_cpus::get(),
            memory_total: 0, // Would need platform-specific code
            rust_version: env!("RUSTC_VERSION").to_string(),
            build_type: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_baseline_manager_creation() {
        let baseline_file = NamedTempFile::new().unwrap();
        let history_file = NamedTempFile::new().unwrap();
        
        let manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        assert_eq!(manager.baselines.len(), 0);
        assert_eq!(manager.historical_data.len(), 0);
    }
    
    #[test]
    fn test_add_measurement_and_establish_baseline() {
        let baseline_file = NamedTempFile::new().unwrap();
        let history_file = NamedTempFile::new().unwrap();
        
        let mut manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        let env = BaselineManager::get_current_environment();
        
        // Add measurements
        for i in 0..20 {
            let measurement = PerformanceMeasurement {
                benchmark_name: "test_benchmark".to_string(),
                duration_ns: 1000000 + (i as u64 * 1000), // Simulate some variation
                throughput: Some(1000.0),
                memory_usage: Some(1024 * 1024),
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                commit_hash: Some("abc123".to_string()),
                environment: env.clone(),
            };
            manager.add_measurement(measurement);
        }
        
        // Establish baseline
        let result = manager.establish_baseline("test_benchmark", 10);
        assert!(result.is_ok());
        
        // Check that baseline was created
        let baseline = manager.get_baseline("test_benchmark");
        assert!(baseline.is_some());
        
        let baseline = baseline.unwrap();
        assert_eq!(baseline.benchmark_name, "test_benchmark");
        assert!(baseline.baseline_duration_ns > 0);
        assert_eq!(baseline.sample_count, 20);
    }
    
    #[test]
    fn test_calculate_stats() {
        let baseline_file = NamedTempFile::new().unwrap();
        let history_file = NamedTempFile::new().unwrap();
        
        let manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = manager.calculate_stats(&values);
        
        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert!(stats.std_dev > 0.0);
    }
    
    #[test]
    fn test_save_and_load_baselines() {
        let baseline_file = NamedTempFile::new().unwrap();
        let history_file = NamedTempFile::new().unwrap();
        
        let mut manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        let env = BaselineManager::get_current_environment();
        
        // Add some measurements and establish baseline
        for i in 0..15 {
            let measurement = PerformanceMeasurement {
                benchmark_name: "save_test".to_string(),
                duration_ns: 2000000 + (i as u64 * 500),
                throughput: None,
                memory_usage: None,
                timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                commit_hash: None,
                environment: env.clone(),
            };
            manager.add_measurement(measurement);
        }
        
        manager.establish_baseline("save_test", 10).unwrap();
        
        // Save data
        manager.save_baselines().unwrap();
        manager.save_historical_data().unwrap();
        
        // Create new manager and load data
        let mut new_manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        assert_eq!(new_manager.baselines.len(), 1);
        assert_eq!(new_manager.historical_data.len(), 15);
        
        let baseline = new_manager.get_baseline("save_test");
        assert!(baseline.is_some());
    }
    
    #[test]
    fn test_performance_trend() {
        let baseline_file = NamedTempFile::new().unwrap();
        let history_file = NamedTempFile::new().unwrap();
        
        let mut manager = BaselineManager::new(
            baseline_file.path().to_str().unwrap(),
            history_file.path().to_str().unwrap(),
        );
        
        let env = BaselineManager::get_current_environment();
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs();
        
        // Add measurements from different time periods
        for i in 0..10 {
            let measurement = PerformanceMeasurement {
                benchmark_name: "trend_test".to_string(),
                duration_ns: 1000000,
                throughput: None,
                memory_usage: None,
                timestamp: current_time - (i as u64 * 24 * 3600), // i days ago
                commit_hash: None,
                environment: env.clone(),
            };
            manager.add_measurement(measurement);
        }
        
        // Get trend for last 5 days
        let trend = manager.get_performance_trend("trend_test", 5);
        assert_eq!(trend.len(), 6); // 0, 1, 2, 3, 4, 5 days ago = 6 measurements
        
        // Get trend for last 15 days (should include all)
        let trend = manager.get_performance_trend("trend_test", 15);
        assert_eq!(trend.len(), 10);
    }
}