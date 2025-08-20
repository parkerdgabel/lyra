use lyra::security::{SecurityManager, SecurityConfig, SecurityError};
use lyra::security::audit::{SecurityEvent, RiskLevel};
use lyra::security::validation::{validate_tensor_dimensions, validate_string, validate_file_path};
use lyra::vm::Value;
use std::sync::Arc;

#[test]
fn test_security_manager_creation() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config);
    assert!(manager.is_ok());
}

#[test]
fn test_rate_limiting_basic() {
    let mut config = SecurityConfig::default();
    config.global_rate_limit = 2;
    
    let manager = SecurityManager::new(config).unwrap();
    
    // First two operations should succeed
    assert!(manager.check_rate_limit("test_op", None).is_ok());
    assert!(manager.check_rate_limit("test_op", None).is_ok());
    
    // Third should fail
    assert!(manager.check_rate_limit("test_op", None).is_err());
}

#[test]
fn test_operation_specific_rate_limiting() {
    let mut config = SecurityConfig::default();
    config.operation_rate_limits.insert("limited_op".to_string(), 1);
    config.global_rate_limit = 100; // High global limit
    
    let manager = SecurityManager::new(config).unwrap();
    
    // First operation should succeed
    assert!(manager.check_rate_limit("limited_op", None).is_ok());
    
    // Second should fail due to operation-specific limit
    assert!(manager.check_rate_limit("limited_op", None).is_err());
    
    // Different operation should still work
    assert!(manager.check_rate_limit("other_op", None).is_ok());
}

#[test]
fn test_resource_monitoring() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    // Track some resource usage
    assert!(manager.track_resource_usage("test_op", 1000, 100).is_ok());
    assert!(manager.track_resource_usage("test_op", 500, 50).is_ok());
    
    let stats = manager.get_resource_stats();
    assert_eq!(stats.memory_used, 1500);
    assert_eq!(stats.cpu_time_ms, 150);
    assert_eq!(stats.operations_count, 2);
}

#[test]
fn test_resource_limit_enforcement() {
    let mut config = SecurityConfig::default();
    config.max_memory_per_context = 1000;
    
    let manager = SecurityManager::new(config).unwrap();
    
    // Should succeed
    assert!(manager.track_resource_usage("test_op", 500, 10).is_ok());
    
    // Should fail - exceeds memory limit  
    assert!(manager.track_resource_usage("test_op", 600, 10).is_err());
}

#[test]
fn test_audit_logging() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    let event = SecurityEvent::RateLimitExceeded {
        operation: "test_op".to_string(),
        user_id: Some("user1".to_string()),
        limit: 100,
        timestamp: std::time::SystemTime::now(),
    };
    
    assert!(manager.log_security_event(event).is_ok());
}

#[test]
fn test_input_validation_tensors() {
    let config = SecurityConfig::default();
    
    // Valid dimensions
    assert!(validate_tensor_dimensions(&[10, 20, 30], &config).is_ok());
    
    // Too many dimensions
    let too_many_dims: Vec<usize> = (0..20).collect();
    assert!(validate_tensor_dimensions(&too_many_dims, &config).is_err());
    
    // Zero dimension
    assert!(validate_tensor_dimensions(&[10, 0, 30], &config).is_err());
    
    // Too large total size
    assert!(validate_tensor_dimensions(&[100_000, 100_000], &config).is_err());
}

#[test]
fn test_input_validation_strings() {
    let config = SecurityConfig::default();
    
    // Valid string
    assert!(validate_string("Hello, World!", &config).is_ok());
    
    // String too long
    let long_string = "a".repeat(config.max_string_length + 1);
    assert!(validate_string(&long_string, &config).is_err());
    
    // String with null bytes
    assert!(validate_string("hello\0world", &config).is_err());
}

#[test]
fn test_file_path_validation() {
    // Valid relative paths
    assert!(validate_file_path("data/file.txt").is_ok());
    assert!(validate_file_path("subdir/another.dat").is_ok());
    
    // Directory traversal
    assert!(validate_file_path("../etc/passwd").is_err());
    assert!(validate_file_path("data/../../../secret").is_err());
    
    // Absolute paths
    assert!(validate_file_path("/etc/passwd").is_err());
    assert!(validate_file_path("C:\\Windows\\system32\\config").is_err());
    
    // System paths
    assert!(validate_file_path("somewhere/etc/passwd").is_err());
}

#[test]
fn test_sandbox_basic_operation() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    let result = manager.execute_sandboxed("test_context", || {
        42 + 8
    });
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 50);
}

#[test]
fn test_value_validation() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    // Valid values
    assert!(manager.validate_input(&Value::Integer(42), "integer").is_ok());
    assert!(manager.validate_input(&Value::String("hello".to_string()), "string").is_ok());
    
    // Invalid values
    assert!(manager.validate_input(&Value::Real(f64::NAN), "real").is_err());
    
    let long_string = "a".repeat(manager.config().max_string_length + 1);
    assert!(manager.validate_input(&Value::String(long_string), "string").is_err());
}

#[test] 
fn test_security_event_types() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    let events = vec![
        SecurityEvent::RateLimitExceeded {
            operation: "test".to_string(),
            user_id: None,
            limit: 100,
            timestamp: std::time::SystemTime::now(),
        },
        SecurityEvent::ResourceLimitExceeded {
            resource: "memory".to_string(),
            current: 2000,
            limit: 1000,
            operation: "test".to_string(),
            context: "ctx".to_string(),
            timestamp: std::time::SystemTime::now(),
        },
        SecurityEvent::InvalidInput {
            input_type: "tensor".to_string(),
            reason: "invalid dimensions".to_string(),
            operation: "tensor_create".to_string(),
            timestamp: std::time::SystemTime::now(),
        },
        SecurityEvent::SandboxViolation {
            operation: "file_access".to_string(),
            violation_type: "unauthorized_access".to_string(),
            context: "sandbox_1".to_string(),
            timestamp: std::time::SystemTime::now(),
        },
        SecurityEvent::SuspiciousActivity {
            activity_type: "repeated_failures".to_string(),
            details: "100 failed operations in 1 second".to_string(),
            risk_level: RiskLevel::High,
            source: "user123".to_string(),
            timestamp: std::time::SystemTime::now(),
        },
    ];
    
    for event in events {
        assert!(manager.log_security_event(event).is_ok());
    }
}

#[test]
fn test_security_config_defaults() {
    let config = SecurityConfig::default();
    
    assert_eq!(config.max_memory_per_context, 1024 * 1024 * 1024);
    assert_eq!(config.max_tensor_dimensions, 8);
    assert_eq!(config.global_rate_limit, 1000);
    assert!(config.enable_audit_logging);
    assert!(config.enable_resource_monitoring);
    assert!(config.enable_sandboxing);
}

#[test]
fn test_concurrent_security_operations() {
    use std::thread;
    use std::sync::Arc;
    
    let config = SecurityConfig::default();
    let manager = Arc::new(SecurityManager::new(config).unwrap());
    
    let handles: Vec<_> = (0..10).map(|i| {
        let manager = manager.clone();
        thread::spawn(move || {
            // Each thread performs some security operations
            let _ = manager.check_rate_limit(&format!("op_{}", i), Some(&format!("user_{}", i)));
            let _ = manager.track_resource_usage(&format!("op_{}", i), 100, 10);
            
            let event = SecurityEvent::ResourceLimitExceeded {
                resource: "test".to_string(),
                current: 100,
                limit: 50,
                operation: format!("op_{}", i),
                context: format!("ctx_{}", i),
                timestamp: std::time::SystemTime::now(),
            };
            let _ = manager.log_security_event(event);
        })
    }).collect();
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Check that all operations were tracked
    let stats = manager.get_resource_stats();
    assert!(stats.operations_count >= 10);
}

#[test]
fn test_dos_protection() {
    let mut config = SecurityConfig::default();
    config.global_rate_limit = 5;
    config.max_memory_per_context = 1000;
    
    let manager = SecurityManager::new(config).unwrap();
    
    // Simulate DoS attack with rapid requests
    let mut success_count = 0;
    let mut failure_count = 0;
    
    for _ in 0..20 {
        match manager.check_rate_limit("attack_op", Some("attacker")) {
            Ok(_) => success_count += 1,
            Err(_) => failure_count += 1,
        }
    }
    
    // Should have rate limited most requests
    assert!(failure_count > success_count);
    assert!(success_count <= 5); // Global limit
    
    // Simulate memory exhaustion attack
    let result = manager.track_resource_usage("memory_bomb", 2000, 10);
    assert!(result.is_err()); // Should exceed memory limit
}

#[test]
fn test_input_sanitization() {
    use lyra::security::validation::{sanitize_string, validate_user_input};
    
    let config = SecurityConfig::default();
    
    // Test string sanitization
    assert_eq!(sanitize_string("hello\0world"), "helloworld");
    assert_eq!(sanitize_string("normal text"), "normal text");
    assert_eq!(sanitize_string("line1\nline2"), "line1\nline2");
    
    // Test user input validation
    assert!(validate_user_input("x + y", &config).is_ok());
    assert!(validate_user_input("eval(malicious_code)", &config).is_err());
    assert!(validate_user_input("import os; os.system('rm -rf /')", &config).is_err());
}

#[test]
fn test_security_error_types() {
    // Test different security error types
    let rate_limit_error = SecurityError::RateLimitExceeded {
        operation: "test".to_string(),
        limit: 100,
        window: 60,
    };
    
    let resource_error = SecurityError::ResourceLimitExceeded {
        resource: "memory".to_string(),
        current: 2000,
        limit: 1000,
    };
    
    let input_error = SecurityError::InvalidInput {
        input_type: "tensor".to_string(),
        reason: "invalid dimensions".to_string(),
    };
    
    let sandbox_error = SecurityError::SandboxViolation {
        operation: "file_access".to_string(),
        reason: "unauthorized access".to_string(),
    };
    
    // Test error display
    let rate_limit_msg = format!("{}", rate_limit_error);
    assert!(rate_limit_msg.contains("Rate limit exceeded"));
    
    let resource_msg = format!("{}", resource_error);
    assert!(resource_msg.contains("Resource limit exceeded"));
    
    let input_msg = format!("{}", input_error);
    assert!(input_msg.contains("Invalid input"));
    
    let sandbox_msg = format!("{}", sandbox_error);
    assert!(sandbox_msg.contains("Sandbox violation"));
}

#[test]
fn test_security_integration() {
    // Test the integration of all security components
    let mut config = SecurityConfig::default();
    config.global_rate_limit = 10;
    config.max_memory_per_context = 10000;
    config.enable_audit_logging = true;
    
    let manager = SecurityManager::new(config).unwrap();
    
    // Simulate a realistic workflow
    for i in 0..15 {
        let operation = format!("workflow_step_{}", i);
        
        // Check rate limit
        let rate_check = manager.check_rate_limit(&operation, Some("user1"));
        
        if rate_check.is_ok() {
            // Track resource usage
            let _ = manager.track_resource_usage(&operation, 100, 10);
            
            // Validate some input
            let test_value = Value::Integer(i);
            let _ = manager.validate_input(&test_value, "step_input");
            
            // Execute in sandbox
            let result = manager.execute_sandboxed("workflow_context", || {
                i * 2
            });
            
            if let Ok(value) = result {
                assert_eq!(value, i * 2);
            }
        } else {
            // Log the rate limit violation
            let event = SecurityEvent::RateLimitExceeded {
                operation: operation.clone(),
                user_id: Some("user1".to_string()),
                limit: 10,
                timestamp: std::time::SystemTime::now(),
            };
            let _ = manager.log_security_event(event);
        }
    }
    
    // Check final state
    let stats = manager.get_resource_stats();
    assert!(stats.operations_count <= 10); // Rate limited
    assert!(stats.memory_used <= 10000); // Within limits
}

#[test]
fn test_nested_security_validation() {
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    // Test deeply nested list validation
    fn create_nested_list(depth: usize) -> Value {
        if depth == 0 {
            Value::Integer(1)
        } else {
            Value::List(vec![create_nested_list(depth - 1)])
        }
    }
    
    // Should handle reasonable nesting
    let reasonable_nested = create_nested_list(5);
    assert!(manager.validate_input(&reasonable_nested, "nested_list").is_ok());
    
    // Should reject excessive nesting
    let excessive_nested = create_nested_list(50);
    assert!(manager.validate_input(&excessive_nested, "nested_list").is_err());
}

#[test]
fn test_security_performance() {
    // Test that security checks don't significantly impact performance
    use std::time::Instant;
    
    let config = SecurityConfig::default();
    let manager = SecurityManager::new(config).unwrap();
    
    let start = Instant::now();
    
    // Perform many security operations
    for i in 0..1000 {
        let _ = manager.check_rate_limit("perf_test", Some(&format!("user_{}", i % 10)));
        let _ = manager.track_resource_usage("perf_test", 10, 1);
        let _ = manager.validate_input(&Value::Integer(i), "perf_input");
    }
    
    let duration = start.elapsed();
    
    // Should complete reasonably quickly (adjust threshold as needed)
    assert!(duration.as_millis() < 1000, "Security operations took too long: {:?}", duration);
}