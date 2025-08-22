//! Security module providing comprehensive security hardening for Lyra
//!
//! This module implements:
//! - Rate limiting and DoS prevention
//! - Resource monitoring and exhaustion protection
//! - Security event logging and audit trails
//! - Input validation and sanitization
//! - Sandboxing for untrusted code execution
//! - Memory safety enhancements

pub mod audit;
pub mod rate_limiter;
pub mod resource_monitor;
pub mod sandbox;
pub mod validation;

use std::fmt;

/// Security-related errors
#[derive(Debug, Clone)]
pub enum SecurityError {
    /// Rate limit exceeded
    RateLimitExceeded {
        operation: String,
        limit: u64,
        window: u64,
    },
    /// Resource limit exceeded
    ResourceLimitExceeded {
        resource: String,
        current: u64,
        limit: u64,
    },
    /// Invalid input detected
    InvalidInput {
        input_type: String,
        reason: String,
    },
    /// Sandbox violation
    SandboxViolation {
        operation: String,
        reason: String,
    },
    /// Security policy violation
    PolicyViolation {
        policy: String,
        violation: String,
    },
    /// Audit log error
    AuditError {
        message: String,
    },
}

impl fmt::Display for SecurityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityError::RateLimitExceeded { operation, limit, window } => {
                write!(f, "Rate limit exceeded for operation '{}': {} requests per {} seconds", 
                       operation, limit, window)
            }
            SecurityError::ResourceLimitExceeded { resource, current, limit } => {
                write!(f, "Resource limit exceeded for '{}': {} > {}", resource, current, limit)
            }
            SecurityError::InvalidInput { input_type, reason } => {
                write!(f, "Invalid input for '{}': {}", input_type, reason)
            }
            SecurityError::SandboxViolation { operation, reason } => {
                write!(f, "Sandbox violation in operation '{}': {}", operation, reason)
            }
            SecurityError::PolicyViolation { policy, violation } => {
                write!(f, "Security policy '{}' violated: {}", policy, violation)
            }
            SecurityError::AuditError { message } => {
                write!(f, "Audit error: {}", message)
            }
        }
    }
}

impl std::error::Error for SecurityError {}

/// Result type for security operations
pub type SecurityResult<T> = Result<T, SecurityError>;

/// Security configuration
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Maximum memory usage per execution context (in bytes)
    pub max_memory_per_context: u64,
    /// Maximum CPU time per operation (in milliseconds)
    pub max_cpu_time_ms: u64,
    /// Maximum tensor dimensions
    pub max_tensor_dimensions: usize,
    /// Maximum tensor size (total elements)
    pub max_tensor_size: usize,
    /// Maximum string length
    pub max_string_length: usize,
    /// Maximum list length
    pub max_list_length: usize,
    /// Global rate limit (operations per second)
    pub global_rate_limit: u64,
    /// Per-operation rate limits
    pub operation_rate_limits: std::collections::HashMap<String, u64>,
    /// Enable audit logging
    pub enable_audit_logging: bool,
    /// Enable resource monitoring
    pub enable_resource_monitoring: bool,
    /// Enable sandboxing
    pub enable_sandboxing: bool,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        let mut operation_rate_limits = std::collections::HashMap::new();
        
        // Default rate limits for expensive operations
        operation_rate_limits.insert("tensor_multiply".to_string(), 100);
        operation_rate_limits.insert("matrix_inverse".to_string(), 50);
        operation_rate_limits.insert("fft".to_string(), 10);
        operation_rate_limits.insert("ml_train".to_string(), 5);
        operation_rate_limits.insert("file_io".to_string(), 20);
        
        Self {
            max_memory_per_context: 1024 * 1024 * 1024, // 1GB
            max_cpu_time_ms: 30000, // 30 seconds
            max_tensor_dimensions: 8,
            max_tensor_size: 100_000_000, // 100M elements
            max_string_length: 1_000_000, // 1M characters
            max_list_length: 1_000_000, // 1M elements
            global_rate_limit: 1000, // 1000 ops/sec globally
            operation_rate_limits,
            enable_audit_logging: true,
            enable_resource_monitoring: true,
            enable_sandboxing: true,
        }
    }
}

/// Main security manager coordinating all security components
pub struct SecurityManager {
    config: SecurityConfig,
    rate_limiter: rate_limiter::RateLimiter,
    resource_monitor: resource_monitor::ResourceMonitor,
    audit_logger: audit::AuditLogger,
    sandbox_manager: sandbox::SandboxManager,
}

impl SecurityManager {
    /// Create a new security manager with the given configuration
    pub fn new(config: SecurityConfig) -> SecurityResult<Self> {
        let rate_limiter = rate_limiter::RateLimiter::new(&config)?;
        let resource_monitor = resource_monitor::ResourceMonitor::new(&config)?;
        let audit_logger = audit::AuditLogger::new(&config)?;
        let sandbox_manager = sandbox::SandboxManager::new(&config)?;
        
        Ok(Self {
            config,
            rate_limiter,
            resource_monitor,
            audit_logger,
            sandbox_manager,
        })
    }
    
    /// Check if an operation is allowed by rate limiting
    pub fn check_rate_limit(&self, operation: &str, user_id: Option<&str>) -> SecurityResult<()> {
        self.rate_limiter.check_rate(operation, user_id)
    }
    
    /// Monitor resource usage for an operation
    pub fn track_resource_usage(&self, operation: &str, memory_delta: i64, cpu_time_ms: u64) -> SecurityResult<()> {
        self.resource_monitor.track_usage(operation, memory_delta, cpu_time_ms)
    }
    
    /// Log a security event
    pub fn log_security_event(&self, event: audit::SecurityEvent) -> SecurityResult<()> {
        self.audit_logger.log_event(event)
    }
    
    /// Validate input parameters
    pub fn validate_input<T>(&self, input: &T, input_type: &str) -> SecurityResult<()> 
    where T: validation::Validatable {
        validation::validate_input(input, input_type, &self.config)
    }
    
    /// Execute code in a sandbox
    pub fn execute_sandboxed<F, R>(&self, context_id: &str, operation: F) -> SecurityResult<R>
    where 
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        self.sandbox_manager.execute(context_id, operation)
    }
    
    /// Get current resource usage statistics
    pub fn get_resource_stats(&self) -> resource_monitor::ResourceStats {
        self.resource_monitor.get_stats()
    }
    
    /// Get security configuration
    pub fn config(&self) -> &SecurityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_manager_creation() {
        let config = SecurityConfig::default();
        let manager = SecurityManager::new(config);
        assert!(manager.is_ok());
    }
    
    #[test]
    fn test_default_config() {
        let config = SecurityConfig::default();
        assert_eq!(config.max_memory_per_context, 1024 * 1024 * 1024);
        assert_eq!(config.max_tensor_dimensions, 8);
        assert!(config.enable_audit_logging);
    }
    
    #[test]
    fn test_security_error_display() {
        let error = SecurityError::RateLimitExceeded {
            operation: "test_op".to_string(),
            limit: 100,
            window: 60,
        };
        let display = format!("{}", error);
        assert!(display.contains("Rate limit exceeded"));
        assert!(display.contains("test_op"));
    }
}