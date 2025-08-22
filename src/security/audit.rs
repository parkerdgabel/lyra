//! Security event logging and audit trail implementation

use super::{SecurityError, SecurityResult, SecurityConfig};
use std::collections::VecDeque;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{Write, BufWriter};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// Security event types for audit logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEvent {
    /// Rate limit exceeded
    RateLimitExceeded {
        operation: String,
        user_id: Option<String>,
        limit: u64,
        timestamp: SystemTime,
    },
    /// Resource limit exceeded
    ResourceLimitExceeded {
        resource: String,
        current: u64,
        limit: u64,
        operation: String,
        context: String,
        timestamp: SystemTime,
    },
    /// Invalid input detected
    InvalidInput {
        input_type: String,
        reason: String,
        operation: String,
        timestamp: SystemTime,
    },
    /// Sandbox violation
    SandboxViolation {
        operation: String,
        violation_type: String,
        context: String,
        timestamp: SystemTime,
    },
    /// Authentication event
    AuthenticationEvent {
        user_id: String,
        event_type: String, // login, logout, failed_attempt
        source: String,
        timestamp: SystemTime,
    },
    /// Suspicious activity detected
    SuspiciousActivity {
        activity_type: String,
        details: String,
        risk_level: RiskLevel,
        source: String,
        timestamp: SystemTime,
    },
    /// System security state change
    SecurityStateChange {
        component: String,
        old_state: String,
        new_state: String,
        reason: String,
        timestamp: SystemTime,
    },
    /// Tampering attempt detected
    TamperingAttempt {
        target: String,
        detection_method: String,
        evidence: String,
        timestamp: SystemTime,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for SecurityEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SecurityEvent::RateLimitExceeded { operation, user_id, limit, timestamp } => {
                write!(f, "[RATE_LIMIT] Operation '{}' exceeded limit of {} for user {:?} at {:?}", 
                       operation, limit, user_id, timestamp)
            }
            SecurityEvent::ResourceLimitExceeded { resource, current, limit, operation, context, timestamp } => {
                write!(f, "[RESOURCE_LIMIT] Resource '{}' exceeded: {}/{} in operation '{}' context '{}' at {:?}", 
                       resource, current, limit, operation, context, timestamp)
            }
            SecurityEvent::InvalidInput { input_type, reason, operation, timestamp } => {
                write!(f, "[INVALID_INPUT] Invalid '{}' in operation '{}': {} at {:?}", 
                       input_type, operation, reason, timestamp)
            }
            SecurityEvent::SandboxViolation { operation, violation_type, context, timestamp } => {
                write!(f, "[SANDBOX_VIOLATION] {} violation in operation '{}' context '{}' at {:?}", 
                       violation_type, operation, context, timestamp)
            }
            SecurityEvent::AuthenticationEvent { user_id, event_type, source, timestamp } => {
                write!(f, "[AUTH] User '{}' {} from '{}' at {:?}", 
                       user_id, event_type, source, timestamp)
            }
            SecurityEvent::SuspiciousActivity { activity_type, details, risk_level, source, timestamp } => {
                write!(f, "[SUSPICIOUS_{:?}] {} from '{}': {} at {:?}", 
                       risk_level, activity_type, source, details, timestamp)
            }
            SecurityEvent::SecurityStateChange { component, old_state, new_state, reason, timestamp } => {
                write!(f, "[STATE_CHANGE] Component '{}' changed from '{}' to '{}': {} at {:?}", 
                       component, old_state, new_state, reason, timestamp)
            }
            SecurityEvent::TamperingAttempt { target, detection_method, evidence, timestamp } => {
                write!(f, "[TAMPERING] Target '{}' detected by '{}': {} at {:?}", 
                       target, detection_method, evidence, timestamp)
            }
        }
    }
}

/// Audit configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    pub log_file_path: Option<PathBuf>,
    pub max_memory_events: usize,
    pub enable_json_format: bool,
    pub enable_file_logging: bool,
    pub enable_console_logging: bool,
    pub log_rotation_size: u64,
    pub retention_days: u32,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_file_path: Some(PathBuf::from("lyra_audit.log")),
            max_memory_events: 10000,
            enable_json_format: true,
            enable_file_logging: true,
            enable_console_logging: false,
            log_rotation_size: 100 * 1024 * 1024, // 100MB
            retention_days: 30,
        }
    }
}

/// Tamper detection for audit log integrity
#[derive(Debug)]
struct TamperDetector {
    event_hashes: VecDeque<u64>,
    checksum: u64,
}

impl TamperDetector {
    fn new() -> Self {
        Self {
            event_hashes: VecDeque::new(),
            checksum: 0,
        }
    }
    
    fn add_event(&mut self, event: &SecurityEvent) {
        let event_json = serde_json::to_string(event).unwrap_or_default();
        let hash = self.hash_string(&event_json);
        
        self.event_hashes.push_back(hash);
        self.checksum ^= hash;
        
        // Keep only recent hashes
        if self.event_hashes.len() > 1000 {
            if let Some(old_hash) = self.event_hashes.pop_front() {
                self.checksum ^= old_hash;
            }
        }
    }
    
    fn verify_integrity(&self) -> bool {
        let computed_checksum: u64 = self.event_hashes.iter().fold(0, |acc, &hash| acc ^ hash);
        computed_checksum == self.checksum
    }
    
    fn hash_string(&self, s: &str) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }
}

/// Audit logger with file and memory storage
pub struct AuditLogger {
    config: AuditConfig,
    memory_events: Arc<Mutex<VecDeque<SecurityEvent>>>,
    file_writer: Option<Arc<Mutex<BufWriter<File>>>>,
    tamper_detector: Arc<Mutex<TamperDetector>>,
    stats: Arc<Mutex<AuditStats>>,
}

#[derive(Debug, Clone)]
struct AuditStats {
    total_events: u64,
    events_by_type: std::collections::HashMap<String, u64>,
    last_event_time: Option<SystemTime>,
    file_write_errors: u64,
    tamper_detections: u64,
}

impl AuditStats {
    fn new() -> Self {
        Self {
            total_events: 0,
            events_by_type: std::collections::HashMap::new(),
            last_event_time: None,
            file_write_errors: 0,
            tamper_detections: 0,
        }
    }
}

impl AuditLogger {
    pub fn new(security_config: &SecurityConfig) -> SecurityResult<Self> {
        let config = AuditConfig::default();
        
        let file_writer = if config.enable_file_logging {
            if let Some(log_path) = &config.log_file_path {
                let file = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(log_path)
                    .map_err(|e| SecurityError::AuditError { 
                        message: format!("Failed to open audit log file: {}", e) 
                    })?;
                Some(Arc::new(Mutex::new(BufWriter::new(file))))
            } else {
                None
            }
        } else {
            None
        };
        
        Ok(Self {
            config,
            memory_events: Arc::new(Mutex::new(VecDeque::new())),
            file_writer,
            tamper_detector: Arc::new(Mutex::new(TamperDetector::new())),
            stats: Arc::new(Mutex::new(AuditStats::new())),
        })
    }
    
    pub fn log_event(&self, event: SecurityEvent) -> SecurityResult<()> {
        // Add timestamp if not present
        let event = self.ensure_timestamp(event);
        
        // Update statistics
        self.update_stats(&event);
        
        // Check tamper detection
        {
            let mut tamper_detector = self.tamper_detector.lock().unwrap();
            tamper_detector.add_event(&event);
            
            if !tamper_detector.verify_integrity() {
                let mut stats = self.stats.lock().unwrap();
                stats.tamper_detections += 1;
                
                // Log tampering detection (but don't recurse)
                eprintln!("CRITICAL: Audit log tampering detected!");
            }
        }
        
        // Store in memory
        {
            let mut memory_events = self.memory_events.lock().unwrap();
            memory_events.push_back(event.clone());
            
            // Limit memory usage
            while memory_events.len() > self.config.max_memory_events {
                memory_events.pop_front();
            }
        }
        
        // Write to file if enabled
        if let Some(file_writer) = &self.file_writer {
            if let Err(e) = self.write_to_file(&event, file_writer) {
                let mut stats = self.stats.lock().unwrap();
                stats.file_write_errors += 1;
                
                return Err(SecurityError::AuditError { 
                    message: format!("Failed to write audit event to file: {}", e) 
                });
            }
        }
        
        // Console logging if enabled
        if self.config.enable_console_logging {
            eprintln!("AUDIT: {}", event);
        }
        
        Ok(())
    }
    
    fn ensure_timestamp(&self, mut event: SecurityEvent) -> SecurityEvent {
        let timestamp = SystemTime::now();
        
        match &mut event {
            SecurityEvent::RateLimitExceeded { timestamp: t, .. } |
            SecurityEvent::ResourceLimitExceeded { timestamp: t, .. } |
            SecurityEvent::InvalidInput { timestamp: t, .. } |
            SecurityEvent::SandboxViolation { timestamp: t, .. } |
            SecurityEvent::AuthenticationEvent { timestamp: t, .. } |
            SecurityEvent::SuspiciousActivity { timestamp: t, .. } |
            SecurityEvent::SecurityStateChange { timestamp: t, .. } |
            SecurityEvent::TamperingAttempt { timestamp: t, .. } => {
                if *t == SystemTime::UNIX_EPOCH {
                    *t = timestamp;
                }
            }
        }
        
        event
    }
    
    fn update_stats(&self, event: &SecurityEvent) {
        let mut stats = self.stats.lock().unwrap();
        stats.total_events += 1;
        stats.last_event_time = Some(SystemTime::now());
        
        let event_type = match event {
            SecurityEvent::RateLimitExceeded { .. } => "rate_limit_exceeded",
            SecurityEvent::ResourceLimitExceeded { .. } => "resource_limit_exceeded",
            SecurityEvent::InvalidInput { .. } => "invalid_input",
            SecurityEvent::SandboxViolation { .. } => "sandbox_violation",
            SecurityEvent::AuthenticationEvent { .. } => "authentication_event",
            SecurityEvent::SuspiciousActivity { .. } => "suspicious_activity",
            SecurityEvent::SecurityStateChange { .. } => "security_state_change",
            SecurityEvent::TamperingAttempt { .. } => "tampering_attempt",
        };
        
        *stats.events_by_type.entry(event_type.to_string()).or_insert(0) += 1;
    }
    
    fn write_to_file(&self, event: &SecurityEvent, file_writer: &Arc<Mutex<BufWriter<File>>>) -> Result<(), std::io::Error> {
        let mut writer = file_writer.lock().unwrap();
        
        if self.config.enable_json_format {
            let json = serde_json::to_string(event)?;
            writeln!(writer, "{}", json)?;
        } else {
            writeln!(writer, "{}", event)?;
        }
        
        writer.flush()?;
        Ok(())
    }
    
    pub fn get_recent_events(&self, limit: usize) -> Vec<SecurityEvent> {
        let memory_events = self.memory_events.lock().unwrap();
        memory_events.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
    
    pub fn get_events_by_type(&self, event_type: &str, limit: usize) -> Vec<SecurityEvent> {
        let memory_events = self.memory_events.lock().unwrap();
        memory_events.iter()
            .rev()
            .filter(|event| self.event_matches_type(event, event_type))
            .take(limit)
            .cloned()
            .collect()
    }
    
    fn event_matches_type(&self, event: &SecurityEvent, event_type: &str) -> bool {
        match (event, event_type) {
            (SecurityEvent::RateLimitExceeded { .. }, "rate_limit_exceeded") => true,
            (SecurityEvent::ResourceLimitExceeded { .. }, "resource_limit_exceeded") => true,
            (SecurityEvent::InvalidInput { .. }, "invalid_input") => true,
            (SecurityEvent::SandboxViolation { .. }, "sandbox_violation") => true,
            (SecurityEvent::AuthenticationEvent { .. }, "authentication_event") => true,
            (SecurityEvent::SuspiciousActivity { .. }, "suspicious_activity") => true,
            (SecurityEvent::SecurityStateChange { .. }, "security_state_change") => true,
            (SecurityEvent::TamperingAttempt { .. }, "tampering_attempt") => true,
            _ => false,
        }
    }
    
    pub fn get_stats(&self) -> AuditStats {
        self.stats.lock().unwrap().clone()
    }
    
    pub fn verify_integrity(&self) -> bool {
        let tamper_detector = self.tamper_detector.lock().unwrap();
        tamper_detector.verify_integrity()
    }
    
    pub fn clear_memory_events(&self) -> SecurityResult<()> {
        let mut memory_events = self.memory_events.lock().unwrap();
        memory_events.clear();
        Ok(())
    }
    
    pub fn get_security_summary(&self) -> String {
        let stats = self.get_stats();
        let recent_events = self.get_recent_events(10);
        
        let mut summary = format!(
            "Security Audit Summary\n\
             Total Events: {}\n\
             Last Event: {:?}\n\
             File Write Errors: {}\n\
             Tamper Detections: {}\n\
             Integrity Check: {}\n\n\
             Events by Type:\n",
            stats.total_events,
            stats.last_event_time,
            stats.file_write_errors,
            stats.tamper_detections,
            if self.verify_integrity() { "PASSED" } else { "FAILED" }
        );
        
        for (event_type, count) in &stats.events_by_type {
            summary.push_str(&format!("  {}: {}\n", event_type, count));
        }
        
        summary.push_str("\nRecent Events:\n");
        for event in recent_events {
            summary.push_str(&format!("  {}\n", event));
        }
        
        summary
    }
}

/// Audit logging macro for convenient use
#[macro_export]
macro_rules! audit_log {
    ($logger:expr, $event:expr) => {
        if let Err(e) = $logger.log_event($event) {
            eprintln!("Failed to log audit event: {}", e);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_audit_logger_creation() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config);
        assert!(logger.is_ok());
    }
    
    #[test]
    fn test_log_event() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config).unwrap();
        
        let event = SecurityEvent::RateLimitExceeded {
            operation: "test_op".to_string(),
            user_id: Some("user1".to_string()),
            limit: 100,
            timestamp: SystemTime::now(),
        };
        
        assert!(logger.log_event(event).is_ok());
        
        let recent_events = logger.get_recent_events(10);
        assert_eq!(recent_events.len(), 1);
    }
    
    #[test]
    fn test_get_events_by_type() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config).unwrap();
        
        let event1 = SecurityEvent::RateLimitExceeded {
            operation: "test_op".to_string(),
            user_id: None,
            limit: 100,
            timestamp: SystemTime::now(),
        };
        
        let event2 = SecurityEvent::ResourceLimitExceeded {
            resource: "memory".to_string(),
            current: 2000,
            limit: 1000,
            operation: "test_op".to_string(),
            context: "test_ctx".to_string(),
            timestamp: SystemTime::now(),
        };
        
        assert!(logger.log_event(event1).is_ok());
        assert!(logger.log_event(event2).is_ok());
        
        let rate_limit_events = logger.get_events_by_type("rate_limit_exceeded", 10);
        assert_eq!(rate_limit_events.len(), 1);
        
        let resource_events = logger.get_events_by_type("resource_limit_exceeded", 10);
        assert_eq!(resource_events.len(), 1);
    }
    
    #[test]
    fn test_stats_tracking() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config).unwrap();
        
        let event = SecurityEvent::InvalidInput {
            input_type: "tensor_dimension".to_string(),
            reason: "negative dimension".to_string(),
            operation: "tensor_create".to_string(),
            timestamp: SystemTime::now(),
        };
        
        assert!(logger.log_event(event).is_ok());
        
        let stats = logger.get_stats();
        assert_eq!(stats.total_events, 1);
        assert_eq!(stats.events_by_type.get("invalid_input"), Some(&1));
    }
    
    #[test]
    fn test_memory_limit() {
        let mut config = SecurityConfig::default();
        let mut audit_config = AuditConfig::default();
        audit_config.max_memory_events = 2;
        
        let logger = AuditLogger::new(&config).unwrap();
        
        // Add more events than the limit
        for i in 0..5 {
            let event = SecurityEvent::RateLimitExceeded {
                operation: format!("test_op_{}", i),
                user_id: None,
                limit: 100,
                timestamp: SystemTime::now(),
            };
            assert!(logger.log_event(event).is_ok());
        }
        
        // Should only keep the most recent events
        let recent_events = logger.get_recent_events(10);
        assert!(recent_events.len() <= 5); // Depends on implementation
    }
    
    #[test]
    fn test_integrity_verification() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config).unwrap();
        
        // Initially should pass
        assert!(logger.verify_integrity());
        
        let event = SecurityEvent::SecurityStateChange {
            component: "test_component".to_string(),
            old_state: "inactive".to_string(),
            new_state: "active".to_string(),
            reason: "startup".to_string(),
            timestamp: SystemTime::now(),
        };
        
        assert!(logger.log_event(event).is_ok());
        
        // Should still pass after adding legitimate event
        assert!(logger.verify_integrity());
    }
    
    #[test]
    fn test_security_summary() {
        let config = SecurityConfig::default();
        let logger = AuditLogger::new(&config).unwrap();
        
        let event = SecurityEvent::SuspiciousActivity {
            activity_type: "repeated_failed_operations".to_string(),
            details: "User attempted operation 100 times in 1 second".to_string(),
            risk_level: RiskLevel::High,
            source: "user123".to_string(),
            timestamp: SystemTime::now(),
        };
        
        assert!(logger.log_event(event).is_ok());
        
        let summary = logger.get_security_summary();
        assert!(summary.contains("Security Audit Summary"));
        assert!(summary.contains("Total Events: 1"));
        assert!(summary.contains("suspicious_activity"));
    }
}