//! Security wrapper for stdlib functions with input validation and resource monitoring

use crate::security::{SecurityManager, SecurityError, SecurityResult};
use crate::security::validation::{validate_input, Validatable};
use crate::security::audit::SecurityEvent;
use crate::vm::{Value, VmResult, VmError};
use std::time::Instant;
use std::sync::Arc;

/// Security-enhanced stdlib function wrapper
pub struct SecureStdlibWrapper {
    security_manager: Arc<SecurityManager>,
    context_id: String,
}

impl SecureStdlibWrapper {
    pub fn new(security_manager: Arc<SecurityManager>, context_id: String) -> Self {
        Self {
            security_manager,
            context_id,
        }
    }
    
    /// Execute a stdlib function with security checks
    pub fn execute_function<F>(
        &self,
        function_name: &str,
        args: &[Value],
        function: F,
    ) -> VmResult<Value>
    where
        F: FnOnce(&[Value]) -> VmResult<Value> + Send,
    {
        let start_time = Instant::now();
        
        // 1. Rate limiting check
        if let Err(e) = self.security_manager.check_rate_limit(function_name, Some(&self.context_id)) {
            let _ = self.security_manager.log_security_event(SecurityEvent::RateLimitExceeded {
                operation: function_name.to_string(),
                user_id: Some(self.context_id.clone()),
                limit: 0, // Will be filled by rate limiter
                timestamp: std::time::SystemTime::now(),
            });
            return Err(VmError::SecurityViolation(format!("Rate limit exceeded for {}: {}", function_name, e)));
        }
        
        // 2. Input validation
        for (i, arg) in args.iter().enumerate() {
            if let Err(e) = self.security_manager.validate_input(arg, &format!("{}[{}]", function_name, i)) {
                let _ = self.security_manager.log_security_event(SecurityEvent::InvalidInput {
                    input_type: format!("{}[{}]", function_name, i),
                    reason: format!("{}", e),
                    operation: function_name.to_string(),
                    timestamp: std::time::SystemTime::now(),
                });
                return Err(VmError::SecurityViolation(format!("Invalid input for {}: {}", function_name, e)));
            }
        }
        
        // 3. Function-specific validation
        self.validate_function_specific(function_name, args)?;
        
        // 4. Sandbox execution
        let result = self.security_manager.execute_sandboxed(&self.context_id, || {
            function(args)
        }).map_err(|e| VmError::SecurityViolation(format!("Sandbox violation in {}: {}", function_name, e)))?;
        
        // 5. Resource tracking
        let execution_time = start_time.elapsed().as_millis() as u64;
        let memory_usage = self.estimate_memory_usage(&result);
        
        if let Err(e) = self.security_manager.track_resource_usage(function_name, memory_usage, execution_time) {
            let _ = self.security_manager.log_security_event(SecurityEvent::ResourceLimitExceeded {
                resource: "function_execution".to_string(),
                current: execution_time,
                limit: 0, // Will be filled by resource monitor
                operation: function_name.to_string(),
                context: self.context_id.clone(),
                timestamp: std::time::SystemTime::now(),
            });
            return Err(VmError::SecurityViolation(format!("Resource limit exceeded in {}: {}", function_name, e)));
        }
        
        result
    }
    
    /// Function-specific validation rules
    fn validate_function_specific(&self, function_name: &str, args: &[Value]) -> VmResult<()> {
        match function_name {
            // Tensor operations
            "Array" | "Tensor" => {
                if args.is_empty() {
                    return Err(VmError::SecurityViolation("Array/Tensor requires at least one argument".to_string()));
                }
                
                // Validate nested structure depth and size
                for arg in args {
                    self.validate_tensor_structure(arg, 0)?;
                }
            }
            
            "ArrayReshape" => {
                if args.len() != 2 {
                    return Err(VmError::SecurityViolation("ArrayReshape requires exactly 2 arguments".to_string()));
                }
                
                if let Value::List(shape) = &args[1] {
                    let config = self.security_manager.config();
                    if shape.len() > config.max_tensor_dimensions {
                        return Err(VmError::SecurityViolation(
                            format!("Too many dimensions for reshape: {} > {}", 
                                   shape.len(), config.max_tensor_dimensions)
                        ));
                    }
                    
                    // Calculate total size to prevent memory bombs
                    let total_size: Result<usize, _> = shape.iter().try_fold(1usize, |acc, val| {
                        if let Value::Integer(n) = val {
                            if *n <= 0 {
                                return Err("Invalid dimension size");
                            }
                            acc.checked_mul(*n as usize).ok_or("Dimension overflow")
                        } else {
                            Err("Non-integer dimension")
                        }
                    });
                    
                    match total_size {
                        Ok(size) if size > config.max_tensor_size => {
                            return Err(VmError::SecurityViolation(
                                format!("Reshaped tensor too large: {} > {}", size, config.max_tensor_size)
                            ));
                        }
                        Err(msg) => {
                            return Err(VmError::SecurityViolation(format!("Invalid reshape dimensions: {}", msg)));
                        }
                        _ => {}
                    }
                }
            }
            
            // String operations
            "StringJoin" => {
                let config = self.security_manager.config();
                let total_length: usize = args.iter()
                    .filter_map(|arg| if let Value::String(s) = arg { Some(s.len()) } else { None })
                    .sum();
                
                if total_length > config.max_string_length {
                    return Err(VmError::SecurityViolation(
                        format!("StringJoin result too long: {} > {}", total_length, config.max_string_length)
                    ));
                }
            }
            
            "StringTake" | "StringDrop" => {
                if args.len() != 2 {
                    return Err(VmError::SecurityViolation(format!("{} requires exactly 2 arguments", function_name)));
                }
                
                if let (Value::String(s), Value::Integer(n)) = (&args[0], &args[1]) {
                    if n.abs() as usize > s.len() * 2 {
                        return Err(VmError::SecurityViolation(
                            "String operation index too large relative to string length".to_string()
                        ));
                    }
                }
            }
            
            // Math operations that could be expensive
            "Power" => {
                if args.len() != 2 {
                    return Err(VmError::SecurityViolation("Power requires exactly 2 arguments".to_string()));
                }
                
                // Prevent extremely large exponents
                if let Value::Integer(exp) = &args[1] {
                    if exp.abs() > 1000 {
                        return Err(VmError::SecurityViolation(
                            format!("Exponent too large: {}", exp)
                        ));
                    }
                } else if let Value::Real(exp) = &args[1] {
                    if exp.abs() > 1000.0 {
                        return Err(VmError::SecurityViolation(
                            format!("Exponent too large: {}", exp)
                        ));
                    }
                }
            }
            
            // Range operations
            "Range" => {
                if args.is_empty() || args.len() > 3 {
                    return Err(VmError::SecurityViolation("Range requires 1-3 arguments".to_string()));
                }
                
                // Calculate range size to prevent memory bombs
                let (start, end, step) = match args.len() {
                    1 => (1, self.extract_integer(&args[0])?, 1),
                    2 => (self.extract_integer(&args[0])?, self.extract_integer(&args[1])?, 1),
                    3 => (self.extract_integer(&args[0])?, self.extract_integer(&args[1])?, self.extract_integer(&args[2])?),
                    _ => unreachable!(),
                };
                
                if step == 0 {
                    return Err(VmError::SecurityViolation("Range step cannot be zero".to_string()));
                }
                
                let range_size = ((end - start).abs() / step.abs()) as usize;
                let config = self.security_manager.config();
                if range_size > config.max_list_length {
                    return Err(VmError::SecurityViolation(
                        format!("Range too large: {} > {}", range_size, config.max_list_length)
                    ));
                }
            }
            
            // I/O operations - require special permissions
            "Import" | "Export" | "FileOpen" => {
                // These should be blocked in sandbox by default
                return Err(VmError::SecurityViolation(
                    format!("I/O operation {} not permitted in secure context", function_name)
                ));
            }
            
            // FFT and signal processing - can be memory intensive
            "FFT" | "IFFT" | "DCT" => {
                if let Some(Value::List(data)) = args.first() {
                    let config = self.security_manager.config();
                    if data.len() > config.max_tensor_size / 2 {
                        return Err(VmError::SecurityViolation(
                            format!("FFT input too large: {} elements", data.len())
                        ));
                    }
                }
            }
            
            // Machine learning operations
            "KMeans" | "ARIMA" | "SVD" | "EigenDecomposition" => {
                // These operations can be computationally expensive
                // Additional validation based on input size
                self.validate_ml_operation_size(function_name, args)?;
            }
            
            _ => {
                // Default validation for unknown functions
                if args.len() > 100 {
                    return Err(VmError::SecurityViolation(
                        format!("Too many arguments for function {}: {}", function_name, args.len())
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate tensor structure for nested lists
    fn validate_tensor_structure(&self, value: &Value, depth: usize) -> VmResult<()> {
        const MAX_TENSOR_DEPTH: usize = 10;
        
        if depth > MAX_TENSOR_DEPTH {
            return Err(VmError::SecurityViolation(
                format!("Tensor nesting too deep: {} > {}", depth, MAX_TENSOR_DEPTH)
            ));
        }
        
        match value {
            Value::List(list) => {
                let config = self.security_manager.config();
                if list.len() > config.max_list_length {
                    return Err(VmError::SecurityViolation(
                        format!("Tensor dimension too large: {} > {}", list.len(), config.max_list_length)
                    ));
                }
                
                for item in list {
                    self.validate_tensor_structure(item, depth + 1)?;
                }
            }
            Value::Integer(n) => {
                if n.abs() > 1_000_000_000_000_000 { // 10^15
                    return Err(VmError::SecurityViolation(
                        format!("Integer value too large: {}", n)
                    ));
                }
            }
            Value::Real(f) => {
                if !f.is_finite() {
                    return Err(VmError::SecurityViolation(
                        format!("Invalid real number: {}", f)
                    ));
                }
                if f.abs() > 1e100 {
                    return Err(VmError::SecurityViolation(
                        format!("Real value too large: {}", f)
                    ));
                }
            }
            _ => {} // Other types are generally safe
        }
        
        Ok(())
    }
    
    /// Validate ML operation input sizes
    fn validate_ml_operation_size(&self, function_name: &str, args: &[Value]) -> VmResult<()> {
        let config = self.security_manager.config();
        
        for arg in args {
            if let Value::List(data) = arg {
                let total_elements = self.count_total_elements(data);
                let max_ml_elements = config.max_tensor_size / 10; // More restrictive for ML
                
                if total_elements > max_ml_elements {
                    return Err(VmError::SecurityViolation(
                        format!("ML operation {} input too large: {} elements > {}", 
                               function_name, total_elements, max_ml_elements)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Count total elements in nested list structure
    fn count_total_elements(&self, list: &[Value]) -> usize {
        list.iter().map(|item| match item {
            Value::List(inner) => self.count_total_elements(inner),
            _ => 1,
        }).sum()
    }
    
    /// Extract integer from Value with validation
    fn extract_integer(&self, value: &Value) -> VmResult<i64> {
        match value {
            Value::Integer(n) => {
                if n.abs() > 1_000_000_000 {
                    Err(VmError::SecurityViolation(format!("Integer too large: {}", n)))
                } else {
                    Ok(*n)
                }
            }
            Value::Real(f) => {
                if f.fract() == 0.0 && f.abs() <= 1_000_000_000.0 {
                    Ok(*f as i64)
                } else {
                    Err(VmError::SecurityViolation("Expected integer value".to_string()))
                }
            }
            _ => Err(VmError::SecurityViolation("Expected numeric value".to_string())),
        }
    }
    
    /// Estimate memory usage of a value
    fn estimate_memory_usage(&self, value: &VmResult<Value>) -> i64 {
        match value {
            Ok(val) => self.estimate_value_memory(val),
            Err(_) => 100, // Small overhead for error
        }
    }
    
    /// Estimate memory usage of a specific value
    fn estimate_value_memory(&self, value: &Value) -> i64 {
        match value {
            Value::Integer(_) => 8,
            Value::Real(_) => 8,
            Value::Boolean(_) => 1,
            Value::String(s) => s.len() as i64,
            Value::Symbol(s) => s.len() as i64,
            Value::List(list) => {
                24 + list.iter().map(|item| self.estimate_value_memory(item)).sum::<i64>()
            }
            Value::Function(name) => {
                name.len() as i64 + 24  // Function name plus basic overhead
            }
            Value::LyObj(_) => 1000, // Conservative estimate for foreign objects
        }
    }
}

/// Secure wrapper macro for stdlib functions
#[macro_export]
macro_rules! secure_stdlib_function {
    ($wrapper:expr, $function_name:expr, $args:expr, $function:expr) => {
        $wrapper.execute_function($function_name, $args, $function)
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::security::{SecurityManager, SecurityConfig};
    
    #[test]
    fn test_secure_wrapper_creation() {
        let config = SecurityConfig::default();
        let security_manager = Arc::new(SecurityManager::new(config).unwrap());
        let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
        
        // Test basic functionality
        let args = vec![Value::Integer(42)];
        let result = wrapper.execute_function("TestFunction", &args, |args| {
            Ok(args[0].clone())
        });
        
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_tensor_validation() {
        let config = SecurityConfig::default();
        let security_manager = Arc::new(SecurityManager::new(config).unwrap());
        let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
        
        // Valid tensor
        let valid_tensor = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ]);
        
        assert!(wrapper.validate_tensor_structure(&valid_tensor, 0).is_ok());
        
        // Invalid tensor (non-finite real)
        let invalid_tensor = Value::List(vec![
            Value::Real(f64::NAN),
        ]);
        
        assert!(wrapper.validate_tensor_structure(&invalid_tensor, 0).is_err());
    }
    
    #[test]
    fn test_function_specific_validation() {
        let config = SecurityConfig::default();
        let security_manager = Arc::new(SecurityManager::new(config).unwrap());
        let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
        
        // Valid Power operation
        let args = vec![Value::Integer(2), Value::Integer(10)];
        assert!(wrapper.validate_function_specific("Power", &args).is_ok());
        
        // Invalid Power operation (exponent too large)
        let args = vec![Value::Integer(2), Value::Integer(2000)];
        assert!(wrapper.validate_function_specific("Power", &args).is_err());
        
        // Invalid I/O operation
        let args = vec![Value::String("file.txt".to_string())];
        assert!(wrapper.validate_function_specific("Import", &args).is_err());
    }
    
    #[test]
    fn test_memory_estimation() {
        let config = SecurityConfig::default();
        let security_manager = Arc::new(SecurityManager::new(config).unwrap());
        let wrapper = SecureStdlibWrapper::new(security_manager, "test_context".to_string());
        
        // Test integer memory estimation
        let int_val = Value::Integer(42);
        assert_eq!(wrapper.estimate_value_memory(&int_val), 8);
        
        // Test string memory estimation
        let str_val = Value::String("hello".to_string());
        assert_eq!(wrapper.estimate_value_memory(&str_val), 5);
        
        // Test list memory estimation
        let list_val = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        assert_eq!(wrapper.estimate_value_memory(&list_val), 24 + 8 + 8);
    }
}