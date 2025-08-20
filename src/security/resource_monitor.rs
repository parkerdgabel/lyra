//! Resource monitoring and exhaustion protection

use super::{SecurityError, SecurityResult, SecurityConfig};
use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, AtomicI64, Ordering}};
use std::time::{Duration, Instant};

/// Resource usage statistics
#[derive(Debug, Clone)]
pub struct ResourceStats {
    pub memory_used: u64,
    pub memory_peak: u64,
    pub cpu_time_ms: u64,
    pub operations_count: u64,
    pub active_contexts: u64,
    pub violations_count: u64,
    pub last_violation: Option<Instant>,
}

/// Per-operation resource tracking
#[derive(Debug)]
struct OperationStats {
    memory_usage: AtomicI64,
    cpu_time_ms: AtomicU64,
    call_count: AtomicU64,
    peak_memory: AtomicU64,
    violation_count: AtomicU64,
}

impl OperationStats {
    fn new() -> Self {
        Self {
            memory_usage: AtomicI64::new(0),
            cpu_time_ms: AtomicU64::new(0),
            call_count: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            violation_count: AtomicU64::new(0),
        }
    }
    
    fn update_memory(&self, delta: i64) -> u64 {
        let new_usage = self.memory_usage.fetch_add(delta, Ordering::SeqCst) + delta;
        let current_usage = new_usage.max(0) as u64;
        
        // Update peak if necessary
        let mut peak = self.peak_memory.load(Ordering::SeqCst);
        while current_usage > peak {
            match self.peak_memory.compare_exchange_weak(
                peak, 
                current_usage, 
                Ordering::SeqCst, 
                Ordering::SeqCst
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
        
        current_usage
    }
    
    fn add_cpu_time(&self, cpu_time_ms: u64) {
        self.cpu_time_ms.fetch_add(cpu_time_ms, Ordering::SeqCst);
    }
    
    fn increment_call_count(&self) {
        self.call_count.fetch_add(1, Ordering::SeqCst);
    }
    
    fn increment_violations(&self) {
        self.violation_count.fetch_add(1, Ordering::SeqCst);
    }
}

/// Resource monitor with configurable limits and alerting
pub struct ResourceMonitor {
    config: SecurityConfig,
    global_memory: AtomicU64,
    global_cpu_time: AtomicU64,
    global_operations: AtomicU64,
    global_violations: AtomicU64,
    peak_memory: AtomicU64,
    operation_stats: Arc<std::sync::RwLock<HashMap<String, Arc<OperationStats>>>>,
    context_stats: Arc<std::sync::RwLock<HashMap<String, Arc<OperationStats>>>>,
    start_time: Instant,
    last_violation: Arc<std::sync::RwLock<Option<Instant>>>,
}

impl ResourceMonitor {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        Ok(Self {
            config: config.clone(),
            global_memory: AtomicU64::new(0),
            global_cpu_time: AtomicU64::new(0),
            global_operations: AtomicU64::new(0),
            global_violations: AtomicU64::new(0),
            peak_memory: AtomicU64::new(0),
            operation_stats: Arc::new(std::sync::RwLock::new(HashMap::new())),
            context_stats: Arc::new(std::sync::RwLock::new(HashMap::new())),
            start_time: Instant::now(),
            last_violation: Arc::new(std::sync::RwLock::new(None)),
        })
    }
    
    pub fn track_usage(&self, operation: &str, memory_delta: i64, cpu_time_ms: u64) -> SecurityResult<()> {
        // Update global counters
        let new_memory = if memory_delta >= 0 {
            self.global_memory.fetch_add(memory_delta as u64, Ordering::SeqCst) + memory_delta as u64
        } else {
            self.global_memory.fetch_sub((-memory_delta) as u64, Ordering::SeqCst) - (-memory_delta) as u64
        };
        
        self.global_cpu_time.fetch_add(cpu_time_ms, Ordering::SeqCst);
        self.global_operations.fetch_add(1, Ordering::SeqCst);
        
        // Update peak memory
        let mut peak = self.peak_memory.load(Ordering::SeqCst);
        while new_memory > peak {
            match self.peak_memory.compare_exchange_weak(
                peak, 
                new_memory, 
                Ordering::SeqCst, 
                Ordering::SeqCst
            ) {
                Ok(_) => break,
                Err(new_peak) => peak = new_peak,
            }
        }
        
        // Update operation-specific stats
        {
            let operation_stats = self.operation_stats.read().unwrap();
            if let Some(stats) = operation_stats.get(operation) {
                let op_memory = stats.update_memory(memory_delta);
                stats.add_cpu_time(cpu_time_ms);
                stats.increment_call_count();
                
                // Check per-operation memory limit
                if op_memory > self.config.max_memory_per_context {
                    stats.increment_violations();
                    self.record_violation();
                    return Err(SecurityError::ResourceLimitExceeded {
                        resource: format!("operation_{}_memory", operation),
                        current: op_memory,
                        limit: self.config.max_memory_per_context,
                    });
                }
            } else {
                drop(operation_stats);
                let mut operation_stats = self.operation_stats.write().unwrap();
                let stats = Arc::new(OperationStats::new());
                stats.update_memory(memory_delta);
                stats.add_cpu_time(cpu_time_ms);
                stats.increment_call_count();
                operation_stats.insert(operation.to_string(), stats);
            }
        }
        
        // Check global memory limit
        if new_memory > self.config.max_memory_per_context * 10 { // Global limit is 10x per-context
            self.record_violation();
            return Err(SecurityError::ResourceLimitExceeded {
                resource: "global_memory".to_string(),
                current: new_memory,
                limit: self.config.max_memory_per_context * 10,
            });
        }
        
        // Check CPU time limit
        if cpu_time_ms > self.config.max_cpu_time_ms {
            self.record_violation();
            return Err(SecurityError::ResourceLimitExceeded {
                resource: "cpu_time".to_string(),
                current: cpu_time_ms,
                limit: self.config.max_cpu_time_ms,
            });
        }
        
        Ok(())
    }
    
    pub fn track_context_usage(&self, context_id: &str, memory_delta: i64, cpu_time_ms: u64) -> SecurityResult<()> {
        let context_stats = self.context_stats.read().unwrap();
        if let Some(stats) = context_stats.get(context_id) {
            let context_memory = stats.update_memory(memory_delta);
            stats.add_cpu_time(cpu_time_ms);
            stats.increment_call_count();
            
            // Check per-context memory limit
            if context_memory > self.config.max_memory_per_context {
                stats.increment_violations();
                self.record_violation();
                return Err(SecurityError::ResourceLimitExceeded {
                    resource: format!("context_{}_memory", context_id),
                    current: context_memory,
                    limit: self.config.max_memory_per_context,
                });
            }
            
            // Check per-context CPU time
            let total_cpu = stats.cpu_time_ms.load(Ordering::SeqCst);
            if total_cpu > self.config.max_cpu_time_ms {
                stats.increment_violations();
                self.record_violation();
                return Err(SecurityError::ResourceLimitExceeded {
                    resource: format!("context_{}_cpu", context_id),
                    current: total_cpu,
                    limit: self.config.max_cpu_time_ms,
                });
            }
        } else {
            drop(context_stats);
            let mut context_stats = self.context_stats.write().unwrap();
            let stats = Arc::new(OperationStats::new());
            stats.update_memory(memory_delta);
            stats.add_cpu_time(cpu_time_ms);
            stats.increment_call_count();
            context_stats.insert(context_id.to_string(), stats);
        }
        
        Ok(())
    }
    
    fn record_violation(&self) {
        self.global_violations.fetch_add(1, Ordering::SeqCst);
        let mut last_violation = self.last_violation.write().unwrap();
        *last_violation = Some(Instant::now());
    }
    
    pub fn get_stats(&self) -> ResourceStats {
        let last_violation = self.last_violation.read().unwrap().clone();
        
        ResourceStats {
            memory_used: self.global_memory.load(Ordering::SeqCst),
            memory_peak: self.peak_memory.load(Ordering::SeqCst),
            cpu_time_ms: self.global_cpu_time.load(Ordering::SeqCst),
            operations_count: self.global_operations.load(Ordering::SeqCst),
            active_contexts: self.context_stats.read().unwrap().len() as u64,
            violations_count: self.global_violations.load(Ordering::SeqCst),
            last_violation,
        }
    }
    
    pub fn get_operation_stats(&self, operation: &str) -> Option<(u64, u64, u64, u64)> {
        let operation_stats = self.operation_stats.read().unwrap();
        operation_stats.get(operation).map(|stats| {
            (
                stats.memory_usage.load(Ordering::SeqCst).max(0) as u64,
                stats.cpu_time_ms.load(Ordering::SeqCst),
                stats.call_count.load(Ordering::SeqCst),
                stats.violation_count.load(Ordering::SeqCst),
            )
        })
    }
    
    pub fn get_context_stats(&self, context_id: &str) -> Option<(u64, u64, u64, u64)> {
        let context_stats = self.context_stats.read().unwrap();
        context_stats.get(context_id).map(|stats| {
            (
                stats.memory_usage.load(Ordering::SeqCst).max(0) as u64,
                stats.cpu_time_ms.load(Ordering::SeqCst),
                stats.call_count.load(Ordering::SeqCst),
                stats.violation_count.load(Ordering::SeqCst),
            )
        })
    }
    
    pub fn reset_stats(&self) -> SecurityResult<()> {
        self.global_memory.store(0, Ordering::SeqCst);
        self.global_cpu_time.store(0, Ordering::SeqCst);
        self.global_operations.store(0, Ordering::SeqCst);
        self.global_violations.store(0, Ordering::SeqCst);
        self.peak_memory.store(0, Ordering::SeqCst);
        
        {
            let mut operation_stats = self.operation_stats.write().unwrap();
            operation_stats.clear();
        }
        
        {
            let mut context_stats = self.context_stats.write().unwrap();
            context_stats.clear();
        }
        
        {
            let mut last_violation = self.last_violation.write().unwrap();
            *last_violation = None;
        }
        
        Ok(())
    }
    
    pub fn is_healthy(&self) -> bool {
        let stats = self.get_stats();
        
        // Check if memory usage is reasonable
        let memory_ok = stats.memory_used < self.config.max_memory_per_context * 5;
        
        // Check if we haven't had too many violations recently
        let violations_ok = if let Some(last_violation) = stats.last_violation {
            last_violation.elapsed() > Duration::from_secs(60) || stats.violations_count < 10
        } else {
            true
        };
        
        memory_ok && violations_ok
    }
    
    pub fn get_health_report(&self) -> String {
        let stats = self.get_stats();
        let uptime = self.start_time.elapsed();
        
        format!(
            "Resource Monitor Health Report\n\
             Uptime: {:.2?}\n\
             Memory Used: {} MB\n\
             Peak Memory: {} MB\n\
             CPU Time: {} ms\n\
             Operations: {}\n\
             Active Contexts: {}\n\
             Violations: {}\n\
             Last Violation: {}\n\
             Healthy: {}",
            uptime,
            stats.memory_used / 1_000_000,
            stats.memory_peak / 1_000_000,
            stats.cpu_time_ms,
            stats.operations_count,
            stats.active_contexts,
            stats.violations_count,
            stats.last_violation.map_or("Never".to_string(), |t| format!("{:.2?} ago", t.elapsed())),
            self.is_healthy()
        )
    }
}

/// Resource tracking macro for convenient use in operations
#[macro_export]
macro_rules! track_resource_usage {
    ($monitor:expr, $operation:expr, $memory:expr, $cpu:expr) => {
        $monitor.track_usage($operation, $memory, $cpu)?;
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_resource_monitor_creation() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config);
        assert!(monitor.is_ok());
    }
    
    #[test]
    fn test_track_usage_basic() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.track_usage("test_op", 1000, 100).is_ok());
        
        let stats = monitor.get_stats();
        assert_eq!(stats.memory_used, 1000);
        assert_eq!(stats.cpu_time_ms, 100);
        assert_eq!(stats.operations_count, 1);
    }
    
    #[test]
    fn test_memory_limit_exceeded() {
        let mut config = SecurityConfig::default();
        config.max_memory_per_context = 1000;
        
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        // Should succeed
        assert!(monitor.track_usage("test_op", 500, 10).is_ok());
        
        // Should fail - exceeds per-operation limit
        assert!(monitor.track_usage("test_op", 600, 10).is_err());
    }
    
    #[test]
    fn test_cpu_limit_exceeded() {
        let mut config = SecurityConfig::default();
        config.max_cpu_time_ms = 1000;
        
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        // Should fail - exceeds CPU limit
        assert!(monitor.track_usage("test_op", 100, 1500).is_err());
    }
    
    #[test]
    fn test_operation_stats() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.track_usage("op1", 1000, 100).is_ok());
        assert!(monitor.track_usage("op1", 500, 50).is_ok());
        assert!(monitor.track_usage("op2", 200, 20).is_ok());
        
        let op1_stats = monitor.get_operation_stats("op1").unwrap();
        assert_eq!(op1_stats.0, 1500); // memory
        assert_eq!(op1_stats.1, 150);  // cpu time
        assert_eq!(op1_stats.2, 2);    // call count
        
        let op2_stats = monitor.get_operation_stats("op2").unwrap();
        assert_eq!(op2_stats.0, 200);
        assert_eq!(op2_stats.1, 20);
        assert_eq!(op2_stats.2, 1);
    }
    
    #[test]
    fn test_context_tracking() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.track_context_usage("ctx1", 1000, 100).is_ok());
        assert!(monitor.track_context_usage("ctx1", 500, 50).is_ok());
        assert!(monitor.track_context_usage("ctx2", 200, 20).is_ok());
        
        let ctx1_stats = monitor.get_context_stats("ctx1").unwrap();
        assert_eq!(ctx1_stats.0, 1500); // memory
        assert_eq!(ctx1_stats.1, 150);  // cpu time
        assert_eq!(ctx1_stats.2, 2);    // call count
    }
    
    #[test]
    fn test_peak_memory_tracking() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.track_usage("test_op", 1000, 10).is_ok());
        assert!(monitor.track_usage("test_op", 2000, 10).is_ok());
        assert!(monitor.track_usage("test_op", -1500, 10).is_ok());
        
        let stats = monitor.get_stats();
        assert_eq!(stats.memory_used, 1500);
        assert_eq!(stats.memory_peak, 3000);
    }
    
    #[test]
    fn test_health_check() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.is_healthy());
        
        // Add some reasonable usage
        assert!(monitor.track_usage("test_op", 1000, 100).is_ok());
        assert!(monitor.is_healthy());
        
        let report = monitor.get_health_report();
        assert!(report.contains("Healthy: true"));
    }
    
    #[test]
    fn test_reset_stats() {
        let config = SecurityConfig::default();
        let monitor = ResourceMonitor::new(&config).unwrap();
        
        assert!(monitor.track_usage("test_op", 1000, 100).is_ok());
        
        let stats_before = monitor.get_stats();
        assert_ne!(stats_before.memory_used, 0);
        
        assert!(monitor.reset_stats().is_ok());
        
        let stats_after = monitor.get_stats();
        assert_eq!(stats_after.memory_used, 0);
        assert_eq!(stats_after.operations_count, 0);
    }
}