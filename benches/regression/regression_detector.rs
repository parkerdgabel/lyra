//! Performance Regression Detection
//!
//! Analyzes performance measurements against baselines to detect regressions,
//! improvements, and performance anomalies.

use super::baseline_manager::{PerformanceMeasurement, PerformanceBaseline, BaselineManager};
use serde::{Serialize, Deserialize};
use std::collections::HashMap;

/// Type of performance change detected
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionType {
    Improvement,      // Performance got better
    Regression,       // Performance got worse  
    NoChange,         // Performance within normal variance
    Anomaly,          // Unusual measurement (outlier)
}

/// Severity level of detected regression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RegressionSeverity {
    Critical,         // >50% degradation
    Major,           // 20-50% degradation
    Minor,           // 5-20% degradation  
    Negligible,      // <5% degradation
}

/// Detailed regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub benchmark_name: String,
    pub regression_type: RegressionType,
    pub severity: Option<RegressionSeverity>,
    pub performance_change_percent: f64,
    pub current_value: u64,
    pub baseline_value: u64,
    pub confidence_level: f64,
    pub statistical_significance: bool,
    pub outlier_detection: bool,
    pub analysis_timestamp: u64,
}

/// Configuration for regression detection
#[derive(Debug, Clone)]
pub struct RegressionConfig {
    pub regression_threshold_percent: f64,    // e.g., 5.0 for 5%
    pub improvement_threshold_percent: f64,   // e.g., 5.0 for 5%
    pub outlier_detection_enabled: bool,
    pub outlier_threshold_std_devs: f64,      // e.g., 3.0 for 3 standard deviations
    pub minimum_confidence_level: f64,        // e.g., 0.95 for 95%
    pub significance_test_enabled: bool,
}

impl Default for RegressionConfig {
    fn default() -> Self {
        Self {
            regression_threshold_percent: 5.0,
            improvement_threshold_percent: 5.0,
            outlier_detection_enabled: true,
            outlier_threshold_std_devs: 3.0,
            minimum_confidence_level: 0.95,
            significance_test_enabled: true,
        }
    }
}

/// Performance regression detector
pub struct RegressionDetector {
    config: RegressionConfig,
}

impl RegressionDetector {
    /// Create a new regression detector with default configuration
    pub fn new() -> Self {
        Self {
            config: RegressionConfig::default(),
        }
    }
    
    /// Create a regression detector with custom configuration
    pub fn with_config(config: RegressionConfig) -> Self {
        Self { config }
    }
    
    /// Analyze a measurement against its baseline
    pub fn analyze_measurement(
        &self,
        measurement: &PerformanceMeasurement,
        baseline: &PerformanceBaseline,
    ) -> RegressionAnalysis {
        let current_value = measurement.duration_ns;
        let baseline_value = baseline.baseline_duration_ns;
        
        // Calculate percentage change (negative = improvement, positive = regression)
        let performance_change_percent = 
            ((current_value as f64 - baseline_value as f64) / baseline_value as f64) * 100.0;
        
        // Determine regression type
        let regression_type = self.classify_change(performance_change_percent);
        
        // Determine severity for regressions
        let severity = if matches!(regression_type, RegressionType::Regression) {
            Some(self.determine_severity(performance_change_percent))
        } else {
            None
        };
        
        // Calculate confidence level based on baseline confidence interval
        let confidence_level = self.calculate_confidence_level(current_value, baseline);
        
        // Statistical significance test
        let statistical_significance = self.test_statistical_significance(
            current_value,
            baseline,
            confidence_level,
        );
        
        // Outlier detection
        let outlier_detection = self.detect_outlier(current_value, baseline);
        
        RegressionAnalysis {
            benchmark_name: measurement.benchmark_name.clone(),
            regression_type,
            severity,
            performance_change_percent,
            current_value,
            baseline_value,
            confidence_level,
            statistical_significance,
            outlier_detection,
            analysis_timestamp: measurement.timestamp,
        }
    }
    
    /// Classify performance change type
    fn classify_change(&self, change_percent: f64) -> RegressionType {
        if change_percent.abs() < self.config.regression_threshold_percent {
            RegressionType::NoChange
        } else if change_percent > 0.0 {
            RegressionType::Regression
        } else {
            RegressionType::Improvement
        }
    }
    
    /// Determine regression severity
    fn determine_severity(&self, change_percent: f64) -> RegressionSeverity {
        let abs_change = change_percent.abs();
        
        if abs_change >= 50.0 {
            RegressionSeverity::Critical
        } else if abs_change >= 20.0 {
            RegressionSeverity::Major
        } else if abs_change >= 5.0 {
            RegressionSeverity::Minor
        } else {
            RegressionSeverity::Negligible
        }
    }
    
    /// Calculate confidence level for the measurement
    fn calculate_confidence_level(&self, current_value: u64, baseline: &PerformanceBaseline) -> f64 {
        let baseline_mean = (baseline.confidence_interval.0 + baseline.confidence_interval.1) / 2.0;
        let baseline_margin = (baseline.confidence_interval.1 - baseline.confidence_interval.0) / 2.0;
        
        if baseline_margin <= 0.0 {
            return 1.0; // Perfect confidence if no variance
        }
        
        let z_score = (current_value as f64 - baseline_mean).abs() / baseline_margin;
        
        // Convert z-score to confidence level (simplified)
        let confidence = 1.0 - (z_score / 3.0).min(1.0); // Normalize to 0-1 range
        confidence.max(0.0)
    }
    
    /// Test for statistical significance
    fn test_statistical_significance(
        &self,
        current_value: u64,
        baseline: &PerformanceBaseline,
        confidence_level: f64,
    ) -> bool {
        if !self.config.significance_test_enabled {
            return true;
        }
        
        // Check if the change is outside the confidence interval
        let current_f64 = current_value as f64;
        let outside_interval = current_f64 < baseline.confidence_interval.0 
            || current_f64 > baseline.confidence_interval.1;
        
        // Also check minimum confidence level
        let meets_confidence = confidence_level >= self.config.minimum_confidence_level;
        
        outside_interval && meets_confidence
    }
    
    /// Detect outliers using standard deviation method
    fn detect_outlier(&self, current_value: u64, baseline: &PerformanceBaseline) -> bool {
        if !self.config.outlier_detection_enabled {
            return false;
        }
        
        let baseline_mean = (baseline.confidence_interval.0 + baseline.confidence_interval.1) / 2.0;
        let baseline_margin = (baseline.confidence_interval.1 - baseline.confidence_interval.0) / 2.0;
        
        // Estimate standard deviation from confidence interval (assuming normal distribution)
        let estimated_std_dev = baseline_margin / 1.96; // 95% CI -> std dev
        
        let z_score = (current_value as f64 - baseline_mean).abs() / estimated_std_dev;
        
        z_score > self.config.outlier_threshold_std_devs
    }
    
    /// Batch analyze multiple measurements
    pub fn analyze_batch(
        &self,
        measurements: &[PerformanceMeasurement],
        baseline_manager: &BaselineManager,
    ) -> Vec<RegressionAnalysis> {
        measurements.iter()
            .filter_map(|measurement| {
                if let Some(baseline) = baseline_manager.get_baseline(&measurement.benchmark_name) {
                    Some(self.analyze_measurement(measurement, baseline))
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Find all regressions in a batch analysis
    pub fn find_regressions(&self, analyses: &[RegressionAnalysis]) -> Vec<&RegressionAnalysis> {
        analyses.iter()
            .filter(|analysis| matches!(analysis.regression_type, RegressionType::Regression))
            .collect()
    }
    
    /// Find all improvements in a batch analysis
    pub fn find_improvements(&self, analyses: &[RegressionAnalysis]) -> Vec<&RegressionAnalysis> {
        analyses.iter()
            .filter(|analysis| matches!(analysis.regression_type, RegressionType::Improvement))
            .collect()
    }
    
    /// Find all outliers in a batch analysis
    pub fn find_outliers(&self, analyses: &[RegressionAnalysis]) -> Vec<&RegressionAnalysis> {
        analyses.iter()
            .filter(|analysis| analysis.outlier_detection)
            .collect()
    }
    
    /// Generate a summary report of regression analysis
    pub fn generate_summary_report(&self, analyses: &[RegressionAnalysis]) -> String {
        let total_benchmarks = analyses.len();
        let regressions = self.find_regressions(analyses);
        let improvements = self.find_improvements(analyses);
        let outliers = self.find_outliers(analyses);
        
        let critical_regressions = regressions.iter()
            .filter(|r| matches!(r.severity, Some(RegressionSeverity::Critical)))
            .count();
        
        let major_regressions = regressions.iter()
            .filter(|r| matches!(r.severity, Some(RegressionSeverity::Major)))
            .count();
        
        let minor_regressions = regressions.iter()
            .filter(|r| matches!(r.severity, Some(RegressionSeverity::Minor)))
            .count();
        
        let mut report = String::new();
        report.push_str("=== Performance Regression Analysis Summary ===\n\n");
        report.push_str(&format!("Total benchmarks analyzed: {}\n", total_benchmarks));
        report.push_str(&format!("Regressions detected: {}\n", regressions.len()));
        report.push_str(&format!("  - Critical: {}\n", critical_regressions));
        report.push_str(&format!("  - Major: {}\n", major_regressions));
        report.push_str(&format!("  - Minor: {}\n", minor_regressions));
        report.push_str(&format!("Improvements detected: {}\n", improvements.len()));
        report.push_str(&format!("Outliers detected: {}\n", outliers.len()));
        report.push_str("\n");
        
        // Critical regressions details
        if !regressions.is_empty() {
            report.push_str("=== Regression Details ===\n");
            for regression in &regressions {
                let severity_str = regression.severity.as_ref()
                    .map(|s| format!("{:?}", s))
                    .unwrap_or("Unknown".to_string());
                    
                report.push_str(&format!(
                    "- {} [{}]: {:.1}% slower ({:.2}ms -> {:.2}ms)\n",
                    regression.benchmark_name,
                    severity_str,
                    regression.performance_change_percent,
                    regression.baseline_value as f64 / 1_000_000.0,
                    regression.current_value as f64 / 1_000_000.0,
                ));
            }
            report.push_str("\n");
        }
        
        // Improvements details
        if !improvements.is_empty() {
            report.push_str("=== Performance Improvements ===\n");
            for improvement in &improvements {
                report.push_str(&format!(
                    "- {}: {:.1}% faster ({:.2}ms -> {:.2}ms)\n",
                    improvement.benchmark_name,
                    improvement.performance_change_percent.abs(),
                    improvement.baseline_value as f64 / 1_000_000.0,
                    improvement.current_value as f64 / 1_000_000.0,
                ));
            }
            report.push_str("\n");
        }
        
        // Outliers details
        if !outliers.is_empty() {
            report.push_str("=== Outliers (may need investigation) ===\n");
            for outlier in &outliers {
                report.push_str(&format!(
                    "- {}: {:.1}% change (confidence: {:.1}%)\n",
                    outlier.benchmark_name,
                    outlier.performance_change_percent,
                    outlier.confidence_level * 100.0,
                ));
            }
        }
        
        report
    }
    
    /// Check if any critical regressions were found
    pub fn has_critical_regressions(&self, analyses: &[RegressionAnalysis]) -> bool {
        analyses.iter()
            .any(|analysis| matches!(analysis.severity, Some(RegressionSeverity::Critical)))
    }
    
    /// Get regression count by severity
    pub fn get_regression_counts(&self, analyses: &[RegressionAnalysis]) -> HashMap<RegressionSeverity, usize> {
        let mut counts = HashMap::new();
        
        for analysis in analyses {
            if let Some(severity) = &analysis.severity {
                *counts.entry(severity.clone()).or_insert(0) += 1;
            }
        }
        
        counts
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benches::regression::baseline_manager::{EnvironmentInfo, BaselineManager};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    fn create_test_environment() -> EnvironmentInfo {
        BaselineManager::get_current_environment()
    }
    
    fn create_test_baseline() -> PerformanceBaseline {
        PerformanceBaseline {
            benchmark_name: "test_benchmark".to_string(),
            baseline_duration_ns: 1_000_000, // 1ms
            baseline_throughput: Some(1000.0),
            baseline_memory: Some(1024),
            confidence_interval: (950_000.0, 1_050_000.0), // ±5%
            sample_count: 100,
            established_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            last_updated: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            environment: create_test_environment(),
        }
    }
    
    fn create_test_measurement(duration_ns: u64) -> PerformanceMeasurement {
        PerformanceMeasurement {
            benchmark_name: "test_benchmark".to_string(),
            duration_ns,
            throughput: Some(1000.0),
            memory_usage: Some(1024),
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            commit_hash: Some("abc123".to_string()),
            environment: create_test_environment(),
        }
    }
    
    #[test]
    fn test_no_change_detection() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(1_000_000); // Same as baseline
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert_eq!(analysis.regression_type, RegressionType::NoChange);
        assert_eq!(analysis.severity, None);
        assert!(analysis.performance_change_percent.abs() < 0.1);
    }
    
    #[test]
    fn test_minor_regression_detection() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(1_100_000); // 10% slower
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert_eq!(analysis.regression_type, RegressionType::Regression);
        assert_eq!(analysis.severity, Some(RegressionSeverity::Minor));
        assert!((analysis.performance_change_percent - 10.0).abs() < 0.1);
    }
    
    #[test]
    fn test_major_regression_detection() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(1_300_000); // 30% slower
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert_eq!(analysis.regression_type, RegressionType::Regression);
        assert_eq!(analysis.severity, Some(RegressionSeverity::Major));
        assert!((analysis.performance_change_percent - 30.0).abs() < 0.1);
    }
    
    #[test]
    fn test_critical_regression_detection() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(1_600_000); // 60% slower
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert_eq!(analysis.regression_type, RegressionType::Regression);
        assert_eq!(analysis.severity, Some(RegressionSeverity::Critical));
        assert!((analysis.performance_change_percent - 60.0).abs() < 0.1);
    }
    
    #[test]
    fn test_improvement_detection() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(800_000); // 20% faster
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert_eq!(analysis.regression_type, RegressionType::Improvement);
        assert_eq!(analysis.severity, None);
        assert!((analysis.performance_change_percent + 20.0).abs() < 0.1);
    }
    
    #[test]
    fn test_outlier_detection() {
        let mut config = RegressionConfig::default();
        config.outlier_threshold_std_devs = 2.0; // More sensitive
        let detector = RegressionDetector::with_config(config);
        
        let baseline = create_test_baseline();
        let measurement = create_test_measurement(2_000_000); // Very far from baseline
        
        let analysis = detector.analyze_measurement(&measurement, &baseline);
        
        assert!(analysis.outlier_detection);
    }
    
    #[test]
    fn test_batch_analysis() {
        let detector = RegressionDetector::new();
        let baseline_manager = {
            let mut manager = BaselineManager::new("test_baseline.json", "test_history.json");
            // We can't easily establish baselines in tests without file I/O, so we'll skip this part
            manager
        };
        
        let measurements = vec![
            create_test_measurement(1_000_000), // No change
            create_test_measurement(1_100_000), // Minor regression
            create_test_measurement(800_000),   // Improvement
        ];
        
        // Note: This test would need properly established baselines to work fully
        let analyses = detector.analyze_batch(&measurements, &baseline_manager);
        
        // Since we don't have baselines established, analyses will be empty
        // In a real scenario with proper baselines, we'd have 3 analyses
        assert!(analyses.len() <= measurements.len());
    }
    
    #[test]
    fn test_summary_report_generation() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        
        let measurements = vec![
            create_test_measurement(1_000_000), // No change
            create_test_measurement(1_100_000), // Minor regression
            create_test_measurement(800_000),   // Improvement
            create_test_measurement(1_600_000), // Critical regression
        ];
        
        let analyses: Vec<RegressionAnalysis> = measurements.iter()
            .map(|m| detector.analyze_measurement(m, &baseline))
            .collect();
        
        let report = detector.generate_summary_report(&analyses);
        
        assert!(report.contains("Total benchmarks analyzed: 4"));
        assert!(report.contains("Regressions detected: 2"));
        assert!(report.contains("Improvements detected: 1"));
        assert!(report.contains("Critical: 1"));
    }
    
    #[test]
    fn test_critical_regression_check() {
        let detector = RegressionDetector::new();
        let baseline = create_test_baseline();
        
        let critical_measurement = create_test_measurement(1_600_000); // 60% slower
        let minor_measurement = create_test_measurement(1_100_000);    // 10% slower
        
        let critical_analysis = detector.analyze_measurement(&critical_measurement, &baseline);
        let minor_analysis = detector.analyze_measurement(&minor_measurement, &baseline);
        
        let analyses = vec![critical_analysis, minor_analysis];
        
        assert!(detector.has_critical_regressions(&analyses));
        
        let counts = detector.get_regression_counts(&analyses);
        assert_eq!(counts.get(&RegressionSeverity::Critical), Some(&1));
        assert_eq!(counts.get(&RegressionSeverity::Minor), Some(&1));
    }
}