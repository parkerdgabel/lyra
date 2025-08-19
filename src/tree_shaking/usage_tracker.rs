//! Usage Tracking System
//!
//! Tracks function usage patterns across the stdlib to inform tree-shaking decisions.

use crate::modules::registry::ModuleRegistry;
use super::TreeShakeError;
use std::collections::{HashMap, HashSet};
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};

/// Tracks function usage patterns across the entire stdlib
pub struct UsageTracker {
    /// Per-function usage statistics
    function_stats: HashMap<String, FunctionUsageStats>,
    
    /// Module-level usage statistics
    module_stats: HashMap<String, ModuleUsageStats>,
    
    /// Call graph tracking
    call_graph: HashMap<String, HashSet<String>>,
    
    /// Session-based tracking
    session_data: SessionTrackingData,
    
    /// Global usage statistics
    global_stats: UsageStats,
    
    /// Configuration
    config: UsageTrackerConfig,
}

/// Usage statistics for a specific function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionUsageStats {
    /// Function name
    pub function_name: String,
    
    /// Module the function belongs to
    pub module_name: String,
    
    /// Total number of times called
    pub call_count: u64,
    
    /// Number of unique calling functions
    pub unique_callers: usize,
    
    /// Average calls per session
    pub avg_calls_per_session: f64,
    
    /// Peak calls per time window
    pub peak_calls_per_window: u64,
    
    /// Time-based usage patterns
    pub usage_patterns: UsagePatterns,
    
    /// Performance impact metrics
    pub performance_impact: PerformanceImpact,
    
    /// First time this function was called
    pub first_called: Option<SystemTime>,
    
    /// Last time this function was called
    pub last_called: Option<SystemTime>,
    
    /// Whether function is actively used
    pub is_active: bool,
    
    /// Critical path indicators
    pub is_on_critical_path: bool,
}

/// Module-level usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleUsageStats {
    /// Module name
    pub module_name: String,
    
    /// Total calls to all functions in this module
    pub total_calls: u64,
    
    /// Number of active functions in this module
    pub active_functions: usize,
    
    /// Number of unused functions in this module
    pub unused_functions: usize,
    
    /// Module load frequency
    pub load_frequency: f64,
    
    /// Average session usage
    pub avg_session_usage: f64,
    
    /// Import patterns
    pub import_patterns: ImportPatterns,
    
    /// Dependencies on other modules
    pub module_dependencies: HashSet<String>,
    
    /// Modules that depend on this one
    pub dependent_modules: HashSet<String>,
}

/// Time-based usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePatterns {
    /// Hourly usage distribution (24 buckets)
    pub hourly_distribution: Vec<u64>,
    
    /// Daily usage over time
    pub daily_usage: Vec<u64>,
    
    /// Seasonal patterns (if applicable)
    pub seasonal_patterns: Vec<f64>,
    
    /// Usage spikes detection
    pub usage_spikes: Vec<UsageSpike>,
    
    /// Trending direction
    pub trend: UsageTrend,
}

/// Performance impact of a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Total execution time across all calls
    pub total_execution_time: Duration,
    
    /// Average execution time per call
    pub avg_execution_time: Duration,
    
    /// Memory allocations caused by this function
    pub memory_allocations: u64,
    
    /// I/O operations performed
    pub io_operations: u64,
    
    /// CPU usage percentage during execution
    pub cpu_usage_percent: f64,
    
    /// Performance bottleneck score (0.0 to 1.0)
    pub bottleneck_score: f64,
}

/// Import patterns for modules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPatterns {
    /// Full module imports
    pub full_imports: u64,
    
    /// Selective function imports
    pub selective_imports: HashMap<String, u64>,
    
    /// Wildcard imports
    pub wildcard_imports: u64,
    
    /// Re-exports from this module
    pub re_exports: HashMap<String, u64>,
    
    /// Most commonly imported functions
    pub popular_functions: Vec<(String, u64)>,
}

/// Usage spike detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSpike {
    /// When the spike occurred
    pub timestamp: SystemTime,
    
    /// Peak value during spike
    pub peak_value: u64,
    
    /// Duration of spike
    pub duration: Duration,
    
    /// Suspected cause
    pub suspected_cause: Option<String>,
}

/// Usage trend analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UsageTrend {
    /// Usage is increasing
    Increasing(f64),
    
    /// Usage is stable
    Stable,
    
    /// Usage is decreasing
    Decreasing(f64),
    
    /// Usage is sporadic/unpredictable
    Sporadic,
    
    /// Not enough data to determine trend
    Insufficient,
}

/// Session-based tracking data
#[derive(Debug, Clone)]
pub struct SessionTrackingData {
    /// Current session ID
    pub current_session: String,
    
    /// Session start time
    pub session_start: SystemTime,
    
    /// Functions called in current session
    pub session_functions: HashSet<String>,
    
    /// Call sequence in current session
    pub call_sequence: Vec<(String, SystemTime)>,
    
    /// Session statistics
    pub session_stats: HashMap<String, SessionStats>,
}

/// Statistics for a specific session
#[derive(Debug, Clone)]
pub struct SessionStats {
    /// Session ID
    pub session_id: String,
    
    /// Duration of session
    pub duration: Duration,
    
    /// Functions used in this session
    pub functions_used: HashSet<String>,
    
    /// Total function calls
    pub total_calls: u64,
    
    /// Unique function calls
    pub unique_calls: usize,
    
    /// Performance metrics for this session
    pub performance_metrics: SessionPerformanceMetrics,
}

/// Performance metrics for a session
#[derive(Debug, Clone)]
pub struct SessionPerformanceMetrics {
    /// Total execution time
    pub total_execution_time: Duration,
    
    /// Memory peak usage
    pub peak_memory_usage: u64,
    
    /// Average CPU utilization
    pub avg_cpu_utilization: f64,
    
    /// Number of cache misses
    pub cache_misses: u64,
}

/// Global usage statistics across all functions and modules
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total number of function calls tracked
    pub total_function_calls: u64,
    
    /// Total number of unique functions called
    pub unique_functions_called: usize,
    
    /// Total number of active modules
    pub active_modules: usize,
    
    /// Total number of sessions tracked
    pub total_sessions: usize,
    
    /// Average session duration
    pub avg_session_duration: Duration,
    
    /// Most called functions (top 10)
    pub most_called_functions: Vec<(String, u64)>,
    
    /// Least called functions
    pub least_called_functions: Vec<(String, u64)>,
    
    /// Functions never called
    pub never_called_functions: Vec<String>,
    
    /// Critical path functions
    pub critical_path_functions: Vec<String>,
    
    /// Performance bottleneck functions
    pub bottleneck_functions: Vec<String>,
    
    /// Time when tracking started
    pub tracking_start_time: Option<SystemTime>,
    
    /// Time when statistics were last updated
    pub last_updated: Option<SystemTime>,
}

/// Configuration for usage tracking
#[derive(Debug, Clone)]
pub struct UsageTrackerConfig {
    /// Enable detailed performance tracking
    pub enable_performance_tracking: bool,
    
    /// Maximum number of sessions to track
    pub max_sessions: usize,
    
    /// Time window for spike detection (seconds)
    pub spike_detection_window: u64,
    
    /// Minimum call threshold for active functions
    pub active_function_threshold: u64,
    
    /// Enable real-time analysis
    pub enable_realtime_analysis: bool,
    
    /// Sampling rate for performance metrics (0.0 to 1.0)
    pub performance_sampling_rate: f64,
    
    /// Maximum call graph depth to track
    pub max_call_graph_depth: u32,
}

impl Default for UsageTrackerConfig {
    fn default() -> Self {
        UsageTrackerConfig {
            enable_performance_tracking: true,
            max_sessions: 1000,
            spike_detection_window: 60,
            active_function_threshold: 10,
            enable_realtime_analysis: true,
            performance_sampling_rate: 0.1,
            max_call_graph_depth: 10,
        }
    }
}

impl UsageTracker {
    /// Create a new usage tracker
    pub fn new() -> Self {
        UsageTracker {
            function_stats: HashMap::new(),
            module_stats: HashMap::new(),
            call_graph: HashMap::new(),
            session_data: SessionTrackingData {
                current_session: format!("session_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
                session_start: SystemTime::now(),
                session_functions: HashSet::new(),
                call_sequence: Vec::new(),
                session_stats: HashMap::new(),
            },
            global_stats: UsageStats::default(),
            config: UsageTrackerConfig::default(),
        }
    }
    
    /// Create a usage tracker with custom configuration
    pub fn with_config(config: UsageTrackerConfig) -> Self {
        let mut tracker = Self::new();
        tracker.config = config;
        tracker
    }
    
    /// Track stdlib usage patterns by analyzing the module registry
    pub fn track_stdlib_usage(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        // Start tracking time
        if self.global_stats.tracking_start_time.is_none() {
            self.global_stats.tracking_start_time = Some(SystemTime::now());
        }
        
        // Analyze all modules in the registry
        for namespace in module_registry.list_modules() {
            if let Some(module) = module_registry.get_module(&namespace) {
                self.analyze_module_usage(&namespace, &module)?;
            }
        }
        
        // Update global statistics
        self.update_global_stats();
        
        // Detect usage patterns
        self.detect_usage_patterns()?;
        
        // Identify critical paths
        self.identify_critical_paths()?;
        
        Ok(())
    }
    
    /// Record a function call
    pub fn record_function_call(&mut self, function_name: &str, module_name: &str, caller: Option<&str>) {
        let now = SystemTime::now();
        
        // Update function statistics
        let stats = self.function_stats.entry(function_name.to_string())
            .or_insert_with(|| FunctionUsageStats {
                function_name: function_name.to_string(),
                module_name: module_name.to_string(),
                call_count: 0,
                unique_callers: 0,
                avg_calls_per_session: 0.0,
                peak_calls_per_window: 0,
                usage_patterns: UsagePatterns {
                    hourly_distribution: vec![0; 24],
                    daily_usage: Vec::new(),
                    seasonal_patterns: Vec::new(),
                    usage_spikes: Vec::new(),
                    trend: UsageTrend::Insufficient,
                },
                performance_impact: PerformanceImpact {
                    total_execution_time: Duration::from_nanos(0),
                    avg_execution_time: Duration::from_nanos(0),
                    memory_allocations: 0,
                    io_operations: 0,
                    cpu_usage_percent: 0.0,
                    bottleneck_score: 0.0,
                },
                first_called: Some(now),
                last_called: Some(now),
                is_active: true,
                is_on_critical_path: false,
            });
        
        stats.call_count += 1;
        stats.last_called = Some(now);
        
        // Update call graph
        if let Some(caller_name) = caller {
            self.call_graph.entry(caller_name.to_string())
                .or_insert_with(HashSet::new)
                .insert(function_name.to_string());
        }
        
        // Update session data
        self.session_data.session_functions.insert(function_name.to_string());
        self.session_data.call_sequence.push((function_name.to_string(), now));
        
        // Update module statistics
        let module_stats = self.module_stats.entry(module_name.to_string())
            .or_insert_with(|| ModuleUsageStats {
                module_name: module_name.to_string(),
                total_calls: 0,
                active_functions: 0,
                unused_functions: 0,
                load_frequency: 0.0,
                avg_session_usage: 0.0,
                import_patterns: ImportPatterns {
                    full_imports: 0,
                    selective_imports: HashMap::new(),
                    wildcard_imports: 0,
                    re_exports: HashMap::new(),
                    popular_functions: Vec::new(),
                },
                module_dependencies: HashSet::new(),
                dependent_modules: HashSet::new(),
            });
        
        module_stats.total_calls += 1;
        
        // Update global statistics
        self.global_stats.total_function_calls += 1;
        
        // Update unique functions count
        self.global_stats.unique_functions_called = self.function_stats.len();
    }
    
    /// Get usage statistics for a specific function
    pub fn get_function_stats(&self, function_name: &str) -> Option<&FunctionUsageStats> {
        self.function_stats.get(function_name)
    }
    
    /// Get usage statistics for a specific module
    pub fn get_module_stats(&self, module_name: &str) -> Option<&ModuleUsageStats> {
        self.module_stats.get(module_name)
    }
    
    /// Get global usage statistics
    pub fn stats(&self) -> &UsageStats {
        &self.global_stats
    }
    
    /// Get functions that are never called
    pub fn get_unused_functions(&self) -> Vec<String> {
        self.global_stats.never_called_functions.clone()
    }
    
    /// Get most frequently called functions
    pub fn get_most_called_functions(&self, limit: usize) -> Vec<(String, u64)> {
        let mut functions: Vec<_> = self.function_stats.iter()
            .map(|(name, stats)| (name.clone(), stats.call_count))
            .collect();
        
        functions.sort_by(|a, b| b.1.cmp(&a.1));
        functions.truncate(limit);
        functions
    }
    
    /// Get functions on critical paths
    pub fn get_critical_path_functions(&self) -> Vec<String> {
        self.function_stats.values()
            .filter(|stats| stats.is_on_critical_path)
            .map(|stats| stats.function_name.clone())
            .collect()
    }
    
    /// Get performance bottleneck functions
    pub fn get_bottleneck_functions(&self, threshold: f64) -> Vec<String> {
        self.function_stats.values()
            .filter(|stats| stats.performance_impact.bottleneck_score > threshold)
            .map(|stats| stats.function_name.clone())
            .collect()
    }
    
    /// Start a new tracking session
    pub fn start_new_session(&mut self) -> String {
        let session_id = format!("session_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        
        // Save current session if it exists
        if !self.session_data.session_functions.is_empty() {
            let duration = SystemTime::now().duration_since(self.session_data.session_start)
                .unwrap_or(Duration::from_secs(0));
            
            let session_stats = SessionStats {
                session_id: self.session_data.current_session.clone(),
                duration,
                functions_used: self.session_data.session_functions.clone(),
                total_calls: self.session_data.call_sequence.len() as u64,
                unique_calls: self.session_data.session_functions.len(),
                performance_metrics: SessionPerformanceMetrics {
                    total_execution_time: duration,
                    peak_memory_usage: 0,
                    avg_cpu_utilization: 0.0,
                    cache_misses: 0,
                },
            };
            
            self.session_data.session_stats.insert(self.session_data.current_session.clone(), session_stats);
        }
        
        // Start new session
        self.session_data.current_session = session_id.clone();
        self.session_data.session_start = SystemTime::now();
        self.session_data.session_functions.clear();
        self.session_data.call_sequence.clear();
        
        session_id
    }
    
    /// Export usage data for analysis
    pub fn export_usage_data(&self) -> UsageExportData {
        UsageExportData {
            function_stats: self.function_stats.clone(),
            module_stats: self.module_stats.clone(),
            global_stats: self.global_stats.clone(),
            export_timestamp: SystemTime::now(),
        }
    }
    
    // Private helper methods
    
    fn analyze_module_usage(&mut self, namespace: &str, module: &crate::modules::Module) -> Result<(), TreeShakeError> {
        // Analyze exports and their usage patterns
        for export in module.exports.values() {
            let function_name = &export.export_name;
            
            // Initialize function stats if not exists
            if !self.function_stats.contains_key(function_name) {
                let stats = FunctionUsageStats {
                    function_name: function_name.clone(),
                    module_name: namespace.to_string(),
                    call_count: 0,
                    unique_callers: 0,
                    avg_calls_per_session: 0.0,
                    peak_calls_per_window: 0,
                    usage_patterns: UsagePatterns {
                        hourly_distribution: vec![0; 24],
                        daily_usage: Vec::new(),
                        seasonal_patterns: Vec::new(),
                        usage_spikes: Vec::new(),
                        trend: UsageTrend::Insufficient,
                    },
                    performance_impact: PerformanceImpact {
                        total_execution_time: Duration::from_nanos(0),
                        avg_execution_time: Duration::from_nanos(0),
                        memory_allocations: 0,
                        io_operations: 0,
                        cpu_usage_percent: 0.0,
                        bottleneck_score: 0.0,
                    },
                    first_called: None,
                    last_called: None,
                    is_active: false,
                    is_on_critical_path: false,
                };
                
                self.function_stats.insert(function_name.clone(), stats);
            }
        }
        
        Ok(())
    }
    
    fn update_global_stats(&mut self) {
        self.global_stats.unique_functions_called = self.function_stats.len();
        self.global_stats.active_modules = self.module_stats.len();
        self.global_stats.total_sessions = self.session_data.session_stats.len();
        
        // Calculate average session duration
        if !self.session_data.session_stats.is_empty() {
            let total_duration: Duration = self.session_data.session_stats.values()
                .map(|stats| stats.duration)
                .sum();
            self.global_stats.avg_session_duration = total_duration / self.session_data.session_stats.len() as u32;
        }
        
        // Update most/least called functions
        self.global_stats.most_called_functions = self.get_most_called_functions(10);
        
        // Find never called functions
        self.global_stats.never_called_functions = self.function_stats.iter()
            .filter(|(_, stats)| stats.call_count == 0)
            .map(|(name, _)| name.clone())
            .collect();
        
        self.global_stats.last_updated = Some(SystemTime::now());
    }
    
    fn detect_usage_patterns(&mut self) -> Result<(), TreeShakeError> {
        // Analyze usage patterns for each function
        for stats in self.function_stats.values_mut() {
            // Determine usage trend
            if stats.call_count == 0 {
                stats.usage_patterns.trend = UsageTrend::Insufficient;
            } else if stats.call_count < 10 {
                stats.usage_patterns.trend = UsageTrend::Sporadic;
            } else {
                // For now, mark as stable - in a real implementation,
                // this would analyze call patterns over time
                stats.usage_patterns.trend = UsageTrend::Stable;
            }
            
            // Mark as active if above threshold
            stats.is_active = stats.call_count >= self.config.active_function_threshold;
        }
        
        Ok(())
    }
    
    fn identify_critical_paths(&mut self) -> Result<(), TreeShakeError> {
        // Identify functions on critical paths based on call graph
        let mut critical_functions = HashSet::new();
        
        // For now, mark frequently called functions as critical
        for (name, stats) in &self.function_stats {
            if stats.call_count > 100 {  // Threshold for critical path
                critical_functions.insert(name.clone());
            }
        }
        
        // Update critical path flags
        for (name, stats) in self.function_stats.iter_mut() {
            stats.is_on_critical_path = critical_functions.contains(name);
        }
        
        self.global_stats.critical_path_functions = critical_functions.into_iter().collect();
        
        Ok(())
    }
}

/// Exported usage data for external analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageExportData {
    pub function_stats: HashMap<String, FunctionUsageStats>,
    pub module_stats: HashMap<String, ModuleUsageStats>,
    pub global_stats: UsageStats,
    pub export_timestamp: SystemTime,
}

impl Default for UsageTracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_tracker_creation() {
        let tracker = UsageTracker::new();
        assert_eq!(tracker.stats().total_function_calls, 0);
        assert_eq!(tracker.stats().unique_functions_called, 0);
    }

    #[test]
    fn test_record_function_call() {
        let mut tracker = UsageTracker::new();
        
        tracker.record_function_call("test_function", "std::math", None);
        
        assert_eq!(tracker.stats().total_function_calls, 1);
        assert_eq!(tracker.stats().unique_functions_called, 1);
        
        let stats = tracker.get_function_stats("test_function");
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().call_count, 1);
        assert_eq!(stats.unwrap().module_name, "std::math");
    }

    #[test]
    fn test_multiple_function_calls() {
        let mut tracker = UsageTracker::new();
        
        tracker.record_function_call("func1", "std::math", None);
        tracker.record_function_call("func2", "std::list", None);
        tracker.record_function_call("func1", "std::math", Some("func2"));
        
        assert_eq!(tracker.stats().total_function_calls, 3);
        assert_eq!(tracker.stats().unique_functions_called, 2);
        
        let func1_stats = tracker.get_function_stats("func1").unwrap();
        assert_eq!(func1_stats.call_count, 2);
    }

    #[test]
    fn test_most_called_functions() {
        let mut tracker = UsageTracker::new();
        
        // Call functions with different frequencies
        for _ in 0..5 {
            tracker.record_function_call("func1", "std::math", None);
        }
        for _ in 0..3 {
            tracker.record_function_call("func2", "std::list", None);
        }
        for _ in 0..1 {
            tracker.record_function_call("func3", "std::string", None);
        }
        
        let most_called = tracker.get_most_called_functions(2);
        assert_eq!(most_called.len(), 2);
        assert_eq!(most_called[0].0, "func1");
        assert_eq!(most_called[0].1, 5);
        assert_eq!(most_called[1].0, "func2");
        assert_eq!(most_called[1].1, 3);
    }

    #[test]
    fn test_session_tracking() {
        let mut tracker = UsageTracker::new();
        
        tracker.record_function_call("func1", "std::math", None);
        tracker.record_function_call("func2", "std::list", None);
        
        let new_session = tracker.start_new_session();
        assert!(new_session.starts_with("session_"));
        
        tracker.record_function_call("func3", "std::string", None);
        
        // First session should be saved
        assert_eq!(tracker.session_data.session_stats.len(), 1);
    }

    #[test]
    fn test_usage_trends() {
        let mut tracker = UsageTracker::new();
        
        // Record some calls
        for _ in 0..5 {
            tracker.record_function_call("active_func", "std::math", None);
        }
        
        tracker.detect_usage_patterns().unwrap();
        
        let stats = tracker.get_function_stats("active_func").unwrap();
        assert!(matches!(stats.usage_patterns.trend, UsageTrend::Sporadic));
    }

    #[test]
    fn test_usage_export() {
        let mut tracker = UsageTracker::new();
        
        tracker.record_function_call("func1", "std::math", None);
        tracker.record_function_call("func2", "std::list", None);
        
        let export_data = tracker.export_usage_data();
        assert_eq!(export_data.function_stats.len(), 2);
        assert!(export_data.function_stats.contains_key("func1"));
        assert!(export_data.function_stats.contains_key("func2"));
    }

    #[test]
    fn test_config_customization() {
        let config = UsageTrackerConfig {
            active_function_threshold: 20,
            enable_performance_tracking: false,
            ..Default::default()
        };
        
        let tracker = UsageTracker::with_config(config);
        assert_eq!(tracker.config.active_function_threshold, 20);
        assert!(!tracker.config.enable_performance_tracking);
    }
}