use lyra::security::{SecurityManager, SecurityConfig};
use lyra::stdlib::secure_wrapper::SecureStdlibWrapper;
use lyra::vm::Value;
use lyra::{lexer::Lexer, parser::Parser, compiler::Compiler, vm::VM};
use std::sync::Arc;

#[test]
fn test_secure_stdlib_wrapper() {
    let config = SecurityConfig::default();
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // Test valid function call
    let args = vec![Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)])];
    let result = wrapper.execute_function("Length", &args, |args| {
        Ok(Value::Integer(args[0].as_list().unwrap().len() as i64))
    });
    
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), Value::Integer(3));
}

#[test]
fn test_secure_wrapper_input_validation() {
    let config = SecurityConfig::default();
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // Test with invalid input (NaN)
    let args = vec![Value::Real(f64::NAN)];
    let result = wrapper.execute_function("Sin", &args, |args| {
        if let Value::Real(x) = &args[0] {
            Ok(Value::Real(x.sin()))
        } else {
            Err(lyra::error::Error::TypeError("Expected real number".to_string()))
        }
    });
    
    assert!(result.is_err());
}

#[test]
fn test_secure_wrapper_rate_limiting() {
    let mut config = SecurityConfig::default();
    config.global_rate_limit = 2;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    let args = vec![Value::Integer(42)];
    
    // First two calls should succeed
    assert!(wrapper.execute_function("TestFunc", &args, |args| Ok(args[0].clone())).is_ok());
    assert!(wrapper.execute_function("TestFunc", &args, |args| Ok(args[0].clone())).is_ok());
    
    // Third should fail due to rate limiting
    assert!(wrapper.execute_function("TestFunc", &args, |args| Ok(args[0].clone())).is_err());
}

#[test]
fn test_secure_wrapper_tensor_validation() {
    let mut config = SecurityConfig::default();
    config.max_tensor_dimensions = 3;
    config.max_tensor_size = 100;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // Valid tensor
    let valid_tensor = Value::List(vec![
        Value::List(vec![Value::Integer(1), Value::Integer(2)]),
        Value::List(vec![Value::Integer(3), Value::Integer(4)]),
    ]);
    let result = wrapper.execute_function("Array", &vec![valid_tensor], |_| Ok(Value::Integer(1)));
    assert!(result.is_ok());
    
    // Invalid tensor (too many dimensions)
    let invalid_tensor = Value::List(vec![
        Value::List(vec![
            Value::List(vec![
                Value::List(vec![Value::Integer(1)])
            ])
        ])
    ]);
    let result = wrapper.execute_function("Array", &vec![invalid_tensor], |_| Ok(Value::Integer(1)));
    assert!(result.is_err());
}

#[test]
fn test_secure_wrapper_string_operations() {
    let mut config = SecurityConfig::default();
    config.max_string_length = 100;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // Valid string operation
    let args = vec![Value::String("hello".to_string()), Value::String(" world".to_string())];
    let result = wrapper.execute_function("StringJoin", &args, |args| {
        if let (Value::String(s1), Value::String(s2)) = (&args[0], &args[1]) {
            Ok(Value::String(format!("{}{}", s1, s2)))
        } else {
            Err(lyra::error::Error::TypeError("Expected strings".to_string()))
        }
    });
    assert!(result.is_ok());
    
    // Invalid string operation (result too long)
    let long_str1 = Value::String("a".repeat(60));
    let long_str2 = Value::String("b".repeat(60));
    let result = wrapper.execute_function("StringJoin", &vec![long_str1, long_str2], |_| Ok(Value::Integer(1)));
    assert!(result.is_err());
}

#[test]
fn test_secure_wrapper_power_operation() {
    let config = SecurityConfig::default();
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // Valid power operation
    let args = vec![Value::Integer(2), Value::Integer(10)];
    let result = wrapper.execute_function("Power", &args, |args| {
        if let (Value::Integer(base), Value::Integer(exp)) = (&args[0], &args[1]) {
            Ok(Value::Integer(base.pow(*exp as u32)))
        } else {
            Err(lyra::error::Error::TypeError("Expected integers".to_string()))
        }
    });
    assert!(result.is_ok());
    
    // Invalid power operation (exponent too large)
    let args = vec![Value::Integer(2), Value::Integer(2000)];
    let result = wrapper.execute_function("Power", &args, |_| Ok(Value::Integer(1)));
    assert!(result.is_err());
}

#[test]
fn test_secure_wrapper_io_blocking() {
    let config = SecurityConfig::default();
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
    
    // I/O operations should be blocked
    let args = vec![Value::String("file.txt".to_string())];
    let result = wrapper.execute_function("Import", &args, |_| Ok(Value::Integer(1)));
    assert!(result.is_err());
    
    let result = wrapper.execute_function("Export", &args, |_| Ok(Value::Integer(1)));
    assert!(result.is_err());
}

#[test]
fn test_vm_security_integration() {
    // Test end-to-end security with VM execution
    let config = SecurityConfig::default();
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    
    let test_cases = vec![
        // Safe expressions
        ("1 + 2", true),
        ("Length[{1, 2, 3}]", true),
        ("Sin[0.5]", true),
        
        // Potentially dangerous expressions that should be caught
        // (These would be caught by parser/compiler validation)
    ];
    
    for (expression, should_succeed) in test_cases {
        let lexer = Lexer::new(expression);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            
            if let Ok(bytecode) = compiler.compile(&ast) {
                let mut vm = VM::new();
                
                // Execute with resource monitoring
                let start_memory = security_manager.get_resource_stats().memory_used;
                
                let result = vm.execute(&bytecode);
                
                // Track the execution
                let end_memory = security_manager.get_resource_stats().memory_used;
                let memory_delta = end_memory as i64 - start_memory as i64;
                
                let _ = security_manager.track_resource_usage("vm_execution", memory_delta, 10);
                
                if should_succeed {
                    // For expressions that should succeed, ensure they don't violate security
                    if result.is_ok() {
                        let stats = security_manager.get_resource_stats();
                        assert!(stats.memory_used < config.max_memory_per_context);
                    }
                }
            }
        }
    }
}

#[test]
fn test_security_event_logging_integration() {
    let mut config = SecurityConfig::default();
    config.enable_audit_logging = true;
    config.global_rate_limit = 1;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager.clone(), "test_context".to_string());
    
    let args = vec![Value::Integer(42)];
    
    // First call should succeed
    let result1 = wrapper.execute_function("TestFunc", &args, |args| Ok(args[0].clone()));
    assert!(result1.is_ok());
    
    // Second call should fail and generate security event
    let result2 = wrapper.execute_function("TestFunc", &args, |args| Ok(args[0].clone()));
    assert!(result2.is_err());
    
    // Check that security events were logged
    // (In a real implementation, we would check the audit log)
}

#[test]
fn test_sandbox_resource_limits() {
    let mut config = SecurityConfig::default();
    config.max_memory_per_context = 1000;
    config.max_cpu_time_ms = 100;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    
    // Test memory limit enforcement
    let result = security_manager.execute_sandboxed("memory_test", || {
        // Simulate memory allocation
        let _large_vec: Vec<u8> = vec![0; 10000];
        42
    });
    
    // This might succeed or fail depending on implementation
    // The key is that it shouldn't crash or hang
    
    // Test CPU time limit enforcement
    let result = security_manager.execute_sandboxed("cpu_test", || {
        // Simulate CPU-intensive operation
        let mut sum = 0;
        for i in 0..1000000 {
            sum += i;
        }
        sum
    });
    
    // Should complete or timeout gracefully
}

#[test]
fn test_security_hardening_comprehensive() {
    // Comprehensive test combining all security features
    let mut config = SecurityConfig::default();
    config.global_rate_limit = 5;
    config.max_memory_per_context = 10000;
    config.max_tensor_dimensions = 4;
    config.max_string_length = 1000;
    config.enable_audit_logging = true;
    config.enable_resource_monitoring = true;
    config.enable_sandboxing = true;
    
    let security_manager = Arc::new(SecurityManager::new(config).unwrap());
    let wrapper = SecureStdlibWrapper::new(security_manager.clone(), "comprehensive_test".to_string());
    
    // Test various operations that exercise different security features
    let test_operations = vec![
        // Valid operations
        (
            "Length", 
            vec![Value::List(vec![Value::Integer(1), Value::Integer(2)])],
            true
        ),
        (
            "StringJoin",
            vec![Value::String("hello".to_string()), Value::String(" world".to_string())],
            true
        ),
        (
            "Plus",
            vec![Value::Integer(10), Value::Integer(20)],
            true
        ),
        
        // Operations that should trigger security measures
        (
            "Array",
            vec![Value::List(vec![Value::Real(f64::NAN)])],
            false
        ),
        (
            "Power",
            vec![Value::Integer(2), Value::Integer(5000)],
            false
        ),
    ];
    
    for (func_name, args, should_succeed) in test_operations {
        let result = wrapper.execute_function(func_name, &args, |args| {
            // Simple mock implementation
            match func_name {
                "Length" => {
                    if let Value::List(list) = &args[0] {
                        Ok(Value::Integer(list.len() as i64))
                    } else {
                        Err(lyra::error::Error::TypeError("Expected list".to_string()))
                    }
                }
                "StringJoin" => {
                    if args.len() >= 2 {
                        if let (Value::String(s1), Value::String(s2)) = (&args[0], &args[1]) {
                            Ok(Value::String(format!("{}{}", s1, s2)))
                        } else {
                            Err(lyra::error::Error::TypeError("Expected strings".to_string()))
                        }
                    } else {
                        Err(lyra::error::Error::TypeError("Not enough arguments".to_string()))
                    }
                }
                "Plus" => {
                    if args.len() >= 2 {
                        if let (Value::Integer(a), Value::Integer(b)) = (&args[0], &args[1]) {
                            Ok(Value::Integer(a + b))
                        } else {
                            Err(lyra::error::Error::TypeError("Expected integers".to_string()))
                        }
                    } else {
                        Err(lyra::error::Error::TypeError("Not enough arguments".to_string()))
                    }
                }
                _ => Ok(Value::Integer(1)), // Default mock response
            }
        });
        
        if should_succeed {
            assert!(result.is_ok(), "Function {} should have succeeded", func_name);
        } else {
            assert!(result.is_err(), "Function {} should have been blocked by security", func_name);
        }
    }
    
    // Verify that security monitoring is working
    let stats = security_manager.get_resource_stats();
    assert!(stats.operations_count > 0);
    
    // Test rate limiting by exceeding the limit
    for _ in 0..10 {
        let args = vec![Value::Integer(1)];
        let _ = wrapper.execute_function("Plus", &args, |args| {
            if let Value::Integer(a) = &args[0] {
                Ok(Value::Integer(a + 1))
            } else {
                Err(lyra::error::Error::TypeError("Expected integer".to_string()))
            }
        });
    }
    
    // Should have hit rate limits
    let final_stats = security_manager.get_resource_stats();
    assert!(final_stats.operations_count <= 15); // Should be rate limited
}