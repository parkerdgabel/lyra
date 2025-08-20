//! Sandboxing implementation for untrusted code execution

use super::{SecurityError, SecurityResult, SecurityConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, Instant};

/// Sandbox capabilities and restrictions
#[derive(Debug, Clone)]
pub struct SandboxCapabilities {
    pub allow_file_io: bool,
    pub allow_network: bool,
    pub allow_system_calls: bool,
    pub allow_subprocess: bool,
    pub max_memory: u64,
    pub max_cpu_time: Duration,
    pub max_operations: u64,
    pub allowed_functions: Option<Vec<String>>,
    pub blocked_functions: Vec<String>,
}

impl Default for SandboxCapabilities {
    fn default() -> Self {
        Self {
            allow_file_io: false,
            allow_network: false,
            allow_system_calls: false,
            allow_subprocess: false,
            max_memory: 64 * 1024 * 1024, // 64MB
            max_cpu_time: Duration::from_secs(10),
            max_operations: 100_000,
            allowed_functions: None, // None means all non-blocked functions are allowed
            blocked_functions: vec![
                "System".to_string(),
                "Run".to_string(),
                "Import".to_string(),
                "Export".to_string(),
                "FileOpen".to_string(),
                "URLFetch".to_string(),
                "ReadList".to_string(),
                "WriteString".to_string(),
            ],
        }
    }
}

/// Execution context for sandboxed operations
#[derive(Debug)]
pub struct SandboxContext {
    id: String,
    capabilities: SandboxCapabilities,
    start_time: Instant,
    memory_used: u64,
    operations_count: u64,
    is_active: bool,
    violations: Vec<String>,
}

impl SandboxContext {
    fn new(id: String, capabilities: SandboxCapabilities) -> Self {
        Self {
            id,
            capabilities,
            start_time: Instant::now(),
            memory_used: 0,
            operations_count: 0,
            is_active: true,
            violations: Vec::new(),
        }
    }
    
    fn check_function_allowed(&self, function_name: &str) -> SecurityResult<()> {
        // Check if function is explicitly blocked
        if self.capabilities.blocked_functions.contains(&function_name.to_string()) {
            return Err(SecurityError::SandboxViolation {
                operation: function_name.to_string(),
                reason: "Function is blocked in sandbox".to_string(),
            });
        }
        
        // If there's an allow list, check if function is in it
        if let Some(ref allowed) = self.capabilities.allowed_functions {
            if !allowed.contains(&function_name.to_string()) {
                return Err(SecurityError::SandboxViolation {
                    operation: function_name.to_string(),
                    reason: "Function not in sandbox allow list".to_string(),
                });
            }
        }
        
        Ok(())
    }
    
    fn check_resource_limits(&mut self) -> SecurityResult<()> {
        // Check CPU time limit
        if self.start_time.elapsed() > self.capabilities.max_cpu_time {
            self.is_active = false;
            return Err(SecurityError::SandboxViolation {
                operation: "execution_time".to_string(),
                reason: format!("CPU time limit exceeded: {:?}", self.capabilities.max_cpu_time),
            });
        }
        
        // Check memory limit
        if self.memory_used > self.capabilities.max_memory {
            self.is_active = false;
            return Err(SecurityError::SandboxViolation {
                operation: "memory_usage".to_string(),
                reason: format!("Memory limit exceeded: {} > {}", 
                               self.memory_used, self.capabilities.max_memory),
            });
        }
        
        // Check operations limit
        if self.operations_count > self.capabilities.max_operations {
            self.is_active = false;
            return Err(SecurityError::SandboxViolation {
                operation: "operation_count".to_string(),
                reason: format!("Operations limit exceeded: {} > {}", 
                               self.operations_count, self.capabilities.max_operations),
            });
        }
        
        Ok(())
    }
    
    fn record_violation(&mut self, violation: String) {
        self.violations.push(violation);
        self.is_active = false;
    }
    
    fn track_operation(&mut self, memory_delta: i64) -> SecurityResult<()> {
        self.operations_count += 1;
        
        if memory_delta > 0 {
            self.memory_used += memory_delta as u64;
        } else {
            self.memory_used = self.memory_used.saturating_sub((-memory_delta) as u64);
        }
        
        self.check_resource_limits()
    }
}

/// Sandbox manager for creating and managing execution contexts
pub struct SandboxManager {
    config: SecurityConfig,
    active_contexts: Arc<RwLock<HashMap<String, Arc<Mutex<SandboxContext>>>>>,
    default_capabilities: SandboxCapabilities,
}

impl SandboxManager {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        let default_capabilities = SandboxCapabilities {
            max_memory: config.max_memory_per_context,
            max_cpu_time: Duration::from_millis(config.max_cpu_time_ms),
            ..SandboxCapabilities::default()
        };
        
        Ok(Self {
            config: config.clone(),
            active_contexts: Arc::new(RwLock::new(HashMap::new())),
            default_capabilities,
        })
    }
    
    pub fn create_context(&self, context_id: &str, capabilities: Option<SandboxCapabilities>) -> SecurityResult<()> {
        let capabilities = capabilities.unwrap_or_else(|| self.default_capabilities.clone());
        let context = SandboxContext::new(context_id.to_string(), capabilities);
        
        let mut contexts = self.active_contexts.write().unwrap();
        contexts.insert(context_id.to_string(), Arc::new(Mutex::new(context)));
        
        Ok(())
    }
    
    pub fn execute<F, R>(&self, context_id: &str, operation: F) -> SecurityResult<R>
    where 
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Create context if it doesn't exist
        if !self.context_exists(context_id) {
            self.create_context(context_id, None)?;
        }
        
        let contexts = self.active_contexts.read().unwrap();
        let context_arc = contexts.get(context_id)
            .ok_or_else(|| SecurityError::SandboxViolation {
                operation: "context_lookup".to_string(),
                reason: format!("Context {} not found", context_id),
            })?
            .clone();
        drop(contexts);
        
        // Check if context is still active
        {
            let context = context_arc.lock().unwrap();
            if !context.is_active {
                return Err(SecurityError::SandboxViolation {
                    operation: "context_state".to_string(),
                    reason: "Context is no longer active due to previous violations".to_string(),
                });
            }
        }
        
        // Execute operation with timeout
        let start_time = Instant::now();
        let max_duration = {
            let context = context_arc.lock().unwrap();
            context.capabilities.max_cpu_time
        };
        
        // Spawn execution in separate thread for timeout control
        let (tx, rx) = std::sync::mpsc::channel();
        let context_for_thread = context_arc.clone();
        
        thread::spawn(move || {
            let result = operation();
            let _ = tx.send(result);
        });
        
        // Wait for result with timeout
        let result = rx.recv_timeout(max_duration)
            .map_err(|_| SecurityError::SandboxViolation {
                operation: "execution_timeout".to_string(),
                reason: format!("Operation timed out after {:?}", max_duration),
            })?;
        
        // Track the operation
        {
            let mut context = context_arc.lock().unwrap();
            context.track_operation(0)?; // Basic operation tracking
        }
        
        Ok(result)
    }
    
    pub fn check_function_call(&self, context_id: &str, function_name: &str) -> SecurityResult<()> {
        let contexts = self.active_contexts.read().unwrap();
        if let Some(context_arc) = contexts.get(context_id) {
            let context = context_arc.lock().unwrap();
            context.check_function_allowed(function_name)
        } else {
            Err(SecurityError::SandboxViolation {
                operation: "function_call_check".to_string(),
                reason: format!("Context {} not found", context_id),
            })
        }
    }
    
    pub fn track_memory_usage(&self, context_id: &str, memory_delta: i64) -> SecurityResult<()> {
        let contexts = self.active_contexts.read().unwrap();
        if let Some(context_arc) = contexts.get(context_id) {
            let mut context = context_arc.lock().unwrap();
            context.track_operation(memory_delta)
        } else {
            Err(SecurityError::SandboxViolation {
                operation: "memory_tracking".to_string(),
                reason: format!("Context {} not found", context_id),
            })
        }
    }
    
    pub fn get_context_stats(&self, context_id: &str) -> Option<(u64, u64, Duration, bool)> {
        let contexts = self.active_contexts.read().unwrap();
        contexts.get(context_id).map(|context_arc| {
            let context = context_arc.lock().unwrap();
            (
                context.memory_used,
                context.operations_count,
                context.start_time.elapsed(),
                context.is_active,
            )
        })
    }
    
    pub fn get_context_violations(&self, context_id: &str) -> Vec<String> {
        let contexts = self.active_contexts.read().unwrap();
        if let Some(context_arc) = contexts.get(context_id) {
            let context = context_arc.lock().unwrap();
            context.violations.clone()
        } else {
            vec![]
        }
    }
    
    pub fn destroy_context(&self, context_id: &str) -> SecurityResult<()> {
        let mut contexts = self.active_contexts.write().unwrap();
        contexts.remove(context_id);
        Ok(())
    }
    
    pub fn list_active_contexts(&self) -> Vec<String> {
        let contexts = self.active_contexts.read().unwrap();
        contexts.keys().cloned().collect()
    }
    
    pub fn cleanup_inactive_contexts(&self) -> SecurityResult<u32> {
        let mut contexts = self.active_contexts.write().unwrap();
        let mut removed_count = 0;
        
        let inactive_contexts: Vec<String> = contexts.iter()
            .filter_map(|(id, context_arc)| {
                let context = context_arc.lock().unwrap();
                if !context.is_active {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect();
        
        for context_id in inactive_contexts {
            contexts.remove(&context_id);
            removed_count += 1;
        }
        
        Ok(removed_count)
    }
    
    fn context_exists(&self, context_id: &str) -> bool {
        let contexts = self.active_contexts.read().unwrap();
        contexts.contains_key(context_id)
    }
    
    pub fn set_context_capabilities(&self, context_id: &str, capabilities: SandboxCapabilities) -> SecurityResult<()> {
        let contexts = self.active_contexts.read().unwrap();
        if let Some(context_arc) = contexts.get(context_id) {
            let mut context = context_arc.lock().unwrap();
            context.capabilities = capabilities;
            Ok(())
        } else {
            Err(SecurityError::SandboxViolation {
                operation: "set_capabilities".to_string(),
                reason: format!("Context {} not found", context_id),
            })
        }
    }
    
    /// Execute code with strict isolation
    pub fn execute_isolated<F, R>(&self, operation: F) -> SecurityResult<R>
    where 
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let context_id = format!("isolated_{}", uuid::Uuid::new_v4());
        
        // Create highly restricted context
        let strict_capabilities = SandboxCapabilities {
            allow_file_io: false,
            allow_network: false,
            allow_system_calls: false,
            allow_subprocess: false,
            max_memory: 16 * 1024 * 1024, // 16MB
            max_cpu_time: Duration::from_secs(5),
            max_operations: 10_000,
            allowed_functions: Some(vec![
                "Plus".to_string(),
                "Times".to_string(),
                "Power".to_string(),
                "List".to_string(),
                "Length".to_string(),
            ]),
            blocked_functions: vec![
                "System".to_string(),
                "Run".to_string(),
                "Import".to_string(),
                "Export".to_string(),
                "FileOpen".to_string(),
                "URLFetch".to_string(),
                "ReadList".to_string(),
                "WriteString".to_string(),
                "Eval".to_string(),
                "ToExpression".to_string(),
            ],
        };
        
        self.create_context(&context_id, Some(strict_capabilities))?;
        
        let result = self.execute(&context_id, operation);
        
        // Always cleanup isolated contexts
        let _ = self.destroy_context(&context_id);
        
        result
    }
}

/// Sandbox enforcement macro for function calls
#[macro_export]
macro_rules! sandbox_check {
    ($sandbox:expr, $context:expr, $function:expr) => {
        $sandbox.check_function_call($context, $function)?;
    };
}

/// Memory tracking macro for sandbox
#[macro_export]
macro_rules! sandbox_track_memory {
    ($sandbox:expr, $context:expr, $delta:expr) => {
        $sandbox.track_memory_usage($context, $delta)?;
    };
}

// Add uuid dependency to Cargo.toml for unique context IDs
// This is a placeholder - in real implementation we'd use a proper UUID library
mod uuid {
    pub struct Uuid;
    impl Uuid {
        pub fn new_v4() -> Self { Self }
    }
    impl std::fmt::Display for Uuid {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{:x}", std::ptr::addr_of!(*self) as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sandbox_manager_creation() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config);
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_create_context() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        assert!(manager.create_context("test_ctx", None).is_ok());
        assert!(manager.context_exists("test_ctx"));
    }
    
    #[test]
    fn test_execute_simple_operation() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        let result = manager.execute("test_ctx", || {
            42 + 8
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 50);
    }
    
    #[test]
    fn test_function_call_checking() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        assert!(manager.create_context("test_ctx", None).is_ok());
        
        // Allowed function
        assert!(manager.check_function_call("test_ctx", "Plus").is_ok());
        
        // Blocked function
        assert!(manager.check_function_call("test_ctx", "System").is_err());
    }
    
    #[test]
    fn test_memory_tracking() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        assert!(manager.create_context("test_ctx", None).is_ok());
        
        // Track some memory usage
        assert!(manager.track_memory_usage("test_ctx", 1000).is_ok());
        assert!(manager.track_memory_usage("test_ctx", 500).is_ok());
        
        let stats = manager.get_context_stats("test_ctx").unwrap();
        assert_eq!(stats.0, 1500); // memory used
        assert_eq!(stats.1, 2);    // operation count
    }
    
    #[test]
    fn test_memory_limit_enforcement() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        let capabilities = SandboxCapabilities {
            max_memory: 1000,
            ..SandboxCapabilities::default()
        };
        
        assert!(manager.create_context("test_ctx", Some(capabilities)).is_ok());
        
        // Should succeed
        assert!(manager.track_memory_usage("test_ctx", 500).is_ok());
        
        // Should fail - exceeds limit
        assert!(manager.track_memory_usage("test_ctx", 600).is_err());
    }
    
    #[test]
    fn test_context_cleanup() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        assert!(manager.create_context("test_ctx1", None).is_ok());
        assert!(manager.create_context("test_ctx2", None).is_ok());
        
        let active_contexts = manager.list_active_contexts();
        assert_eq!(active_contexts.len(), 2);
        
        assert!(manager.destroy_context("test_ctx1").is_ok());
        
        let active_contexts = manager.list_active_contexts();
        assert_eq!(active_contexts.len(), 1);
    }
    
    #[test]
    fn test_isolated_execution() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        let result = manager.execute_isolated(|| {
            // Simple computation that should be allowed
            let mut sum = 0;
            for i in 1..=10 {
                sum += i;
            }
            sum
        });
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 55);
    }
    
    #[test]
    fn test_context_violations_tracking() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        let capabilities = SandboxCapabilities {
            max_operations: 2,
            ..SandboxCapabilities::default()
        };
        
        assert!(manager.create_context("test_ctx", Some(capabilities)).is_ok());
        
        // First two operations should succeed
        assert!(manager.track_memory_usage("test_ctx", 100).is_ok());
        assert!(manager.track_memory_usage("test_ctx", 100).is_ok());
        
        // Third operation should fail
        assert!(manager.track_memory_usage("test_ctx", 100).is_err());
        
        let violations = manager.get_context_violations("test_ctx");
        assert!(!violations.is_empty());
    }
    
    #[test]
    fn test_capabilities_modification() {
        let config = SecurityConfig::default();
        let manager = SandboxManager::new(&config).unwrap();
        
        assert!(manager.create_context("test_ctx", None).is_ok());
        
        let new_capabilities = SandboxCapabilities {
            max_memory: 2000,
            allowed_functions: Some(vec!["OnlyThis".to_string()]),
            ..SandboxCapabilities::default()
        };
        
        assert!(manager.set_context_capabilities("test_ctx", new_capabilities).is_ok());
        
        // Now only "OnlyThis" should be allowed
        assert!(manager.check_function_call("test_ctx", "OnlyThis").is_ok());
        assert!(manager.check_function_call("test_ctx", "Plus").is_err());
    }
}