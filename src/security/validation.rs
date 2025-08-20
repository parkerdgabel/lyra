//! Input validation and sanitization for security hardening

use super::{SecurityError, SecurityResult, SecurityConfig};
use crate::vm::Value;
use std::collections::HashMap;

/// Trait for validatable inputs
pub trait Validatable {
    fn validate(&self, config: &SecurityConfig) -> SecurityResult<()>;
}

/// Validate input with specific type checking
pub fn validate_input<T: Validatable>(input: &T, input_type: &str, config: &SecurityConfig) -> SecurityResult<()> {
    input.validate(config).map_err(|e| match e {
        SecurityError::InvalidInput { reason, .. } => SecurityError::InvalidInput {
            input_type: input_type.to_string(),
            reason,
        },
        other => other,
    })
}

/// Validate tensor dimensions
pub fn validate_tensor_dimensions(dims: &[usize], config: &SecurityConfig) -> SecurityResult<()> {
    if dims.len() > config.max_tensor_dimensions {
        return Err(SecurityError::InvalidInput {
            input_type: "tensor_dimensions".to_string(),
            reason: format!("Too many dimensions: {} > {}", dims.len(), config.max_tensor_dimensions),
        });
    }
    
    let total_size: usize = dims.iter().product();
    if total_size > config.max_tensor_size {
        return Err(SecurityError::InvalidInput {
            input_type: "tensor_size".to_string(),
            reason: format!("Tensor too large: {} > {}", total_size, config.max_tensor_size),
        });
    }
    
    for (i, &dim) in dims.iter().enumerate() {
        if dim == 0 {
            return Err(SecurityError::InvalidInput {
                input_type: "tensor_dimension".to_string(),
                reason: format!("Zero dimension at index {}", i),
            });
        }
        if dim > 1_000_000_000 {
            return Err(SecurityError::InvalidInput {
                input_type: "tensor_dimension".to_string(),
                reason: format!("Dimension too large at index {}: {}", i, dim),
            });
        }
    }
    
    Ok(())
}

/// Validate string length and content
pub fn validate_string(s: &str, config: &SecurityConfig) -> SecurityResult<()> {
    if s.len() > config.max_string_length {
        return Err(SecurityError::InvalidInput {
            input_type: "string_length".to_string(),
            reason: format!("String too long: {} > {}", s.len(), config.max_string_length),
        });
    }
    
    // Check for potentially dangerous patterns
    if s.contains('\0') {
        return Err(SecurityError::InvalidInput {
            input_type: "string_content".to_string(),
            reason: "String contains null bytes".to_string(),
        });
    }
    
    // Check for extremely long lines that might cause issues
    for (line_no, line) in s.lines().enumerate() {
        if line.len() > 100_000 {
            return Err(SecurityError::InvalidInput {
                input_type: "string_content".to_string(),
                reason: format!("Line {} too long: {} characters", line_no, line.len()),
            });
        }
    }
    
    Ok(())
}

/// Validate list size and depth
pub fn validate_list<T>(list: &[T], config: &SecurityConfig) -> SecurityResult<()> {
    if list.len() > config.max_list_length {
        return Err(SecurityError::InvalidInput {
            input_type: "list_length".to_string(),
            reason: format!("List too long: {} > {}", list.len(), config.max_list_length),
        });
    }
    
    Ok(())
}

/// Validate function name
pub fn validate_function_name(name: &str, config: &SecurityConfig) -> SecurityResult<()> {
    if name.len() > config.max_string_length {
        return Err(SecurityError::InvalidInput {
            input_type: "function_name".to_string(),
            reason: format!("Function name too long: {} > {}", name.len(), config.max_string_length),
        });
    }
    
    // Check for potentially dangerous function names
    let dangerous_functions = ["System", "Run", "Execute", "Import", "Export", "FileOpen", "FileWrite"];
    if dangerous_functions.contains(&name) {
        return Err(SecurityError::InvalidInput {
            input_type: "function_name".to_string(),
            reason: format!("Potentially dangerous function: {}", name),
        });
    }
    
    Ok(())
}

/// Validate nested structure depth to prevent stack overflow
pub fn validate_nesting_depth(depth: usize) -> SecurityResult<()> {
    const MAX_NESTING_DEPTH: usize = 1000;
    
    if depth > MAX_NESTING_DEPTH {
        return Err(SecurityError::InvalidInput {
            input_type: "nesting_depth".to_string(),
            reason: format!("Nesting too deep: {} > {}", depth, MAX_NESTING_DEPTH),
        });
    }
    
    Ok(())
}

/// Validate numerical ranges to prevent overflow/underflow
pub fn validate_number_range(value: f64, min: f64, max: f64) -> SecurityResult<()> {
    if !value.is_finite() {
        return Err(SecurityError::InvalidInput {
            input_type: "number_value".to_string(),
            reason: format!("Non-finite number: {}", value),
        });
    }
    
    if value < min || value > max {
        return Err(SecurityError::InvalidInput {
            input_type: "number_range".to_string(),
            reason: format!("Number out of range: {} not in [{}, {}]", value, min, max),
        });
    }
    
    Ok(())
}

/// Validate file paths to prevent directory traversal
pub fn validate_file_path(path: &str) -> SecurityResult<()> {
    // Prevent directory traversal
    if path.contains("..") {
        return Err(SecurityError::InvalidInput {
            input_type: "file_path".to_string(),
            reason: "Path contains directory traversal".to_string(),
        });
    }
    
    // Prevent absolute paths in untrusted contexts
    if path.starts_with('/') || path.starts_with('\\') {
        return Err(SecurityError::InvalidInput {
            input_type: "file_path".to_string(),
            reason: "Absolute paths not allowed".to_string(),
        });
    }
    
    // Prevent access to system files
    let dangerous_patterns = [
        "/etc/", "/sys/", "/proc/", "/dev/",
        "c:\\windows\\", "c:\\system32\\",
    ];
    
    let path_lower = path.to_lowercase();
    for pattern in &dangerous_patterns {
        if path_lower.contains(pattern) {
            return Err(SecurityError::InvalidInput {
                input_type: "file_path".to_string(),
                reason: format!("Access to system path denied: {}", path),
            });
        }
    }
    
    Ok(())
}

/// Validate function parameters based on operation type
pub fn validate_function_params(
    function_name: &str, 
    params: &[Value], 
    config: &SecurityConfig
) -> SecurityResult<()> {
    match function_name {
        "Tensor" | "Array" | "Matrix" => {
            if params.len() > 10 {
                return Err(SecurityError::InvalidInput {
                    input_type: "function_params".to_string(),
                    reason: format!("Too many parameters for {}: {}", function_name, params.len()),
                });
            }
            
            // Validate tensor creation parameters
            for param in params {
                if let Value::List(ref list) = param {
                    validate_list(list, config)?;
                    validate_value_depth(param, 0)?;
                }
            }
        }
        
        "Range" => {
            if params.len() > 3 {
                return Err(SecurityError::InvalidInput {
                    input_type: "function_params".to_string(),
                    reason: "Range function takes at most 3 parameters".to_string(),
                });
            }
            
            // Validate range parameters
            for param in params {
                if let Value::Integer(n) = param {
                    if n.abs() > 1_000_000_000 {
                        return Err(SecurityError::InvalidInput {
                            input_type: "range_parameter".to_string(),
                            reason: format!("Range parameter too large: {}", n),
                        });
                    }
                }
            }
        }
        
        "StringJoin" | "StringSplit" => {
            for param in params {
                if let Value::String(ref s) = param {
                    validate_string(s, config)?;
                }
            }
        }
        
        "Import" | "Export" | "FileOpen" => {
            if let Some(Value::String(ref path)) = params.first() {
                validate_file_path(path)?;
            }
        }
        
        _ => {
            // Generic validation for unknown functions
            if params.len() > 100 {
                return Err(SecurityError::InvalidInput {
                    input_type: "function_params".to_string(),
                    reason: format!("Too many parameters: {}", params.len()),
                });
            }
        }
    }
    
    Ok(())
}

/// Validate AST node depth to prevent stack overflow during evaluation
fn validate_value_depth(value: &Value, current_depth: usize) -> SecurityResult<()> {
    validate_nesting_depth(current_depth)?;
    
    match value {
        Value::List(list) => {
            for item in list {
                validate_value_depth(item, current_depth + 1)?;
            }
        }
        Value::Function(_) => {
            // Function values don't contain nested structures to validate
        }
        _ => {}
    }
    
    Ok(())
}

/// Sanitize string input by removing dangerous characters
pub fn sanitize_string(input: &str) -> String {
    input
        .chars()
        .filter(|&c| c != '\0' && c.is_control() == false || c == '\n' || c == '\t')
        .collect()
}

/// Validate and sanitize user input for safe processing
pub fn validate_user_input(input: &str, config: &SecurityConfig) -> SecurityResult<String> {
    // First validate the input
    validate_string(input, config)?;
    
    // Then sanitize it
    let sanitized = sanitize_string(input);
    
    // Check for injection patterns
    let dangerous_patterns = [
        "eval(", "exec(", "system(", "shell(",
        "import os", "import sys", "__import__",
        "subprocess", "getattr", "setattr",
    ];
    
    let input_lower = sanitized.to_lowercase();
    for pattern in &dangerous_patterns {
        if input_lower.contains(pattern) {
            return Err(SecurityError::InvalidInput {
                input_type: "user_input".to_string(),
                reason: format!("Potentially dangerous pattern detected: {}", pattern),
            });
        }
    }
    
    Ok(sanitized)
}

/// Validation macros for common patterns
#[macro_export]
macro_rules! validate_params {
    ($params:expr, $config:expr, $expected:expr) => {
        if $params.len() != $expected {
            return Err(SecurityError::InvalidInput {
                input_type: "parameter_count".to_string(),
                reason: format!("Expected {} parameters, got {}", $expected, $params.len()),
            });
        }
    };
}

#[macro_export]
macro_rules! validate_tensor_dims {
    ($dims:expr, $config:expr) => {
        $crate::security::validation::validate_tensor_dimensions($dims, $config)?;
    };
}

#[macro_export]
macro_rules! validate_string_input {
    ($string:expr, $config:expr) => {
        $crate::security::validation::validate_string($string, $config)?;
    };
}

// Implement Validatable for common types
impl Validatable for Value {
    fn validate(&self, config: &SecurityConfig) -> SecurityResult<()> {
        validate_value_depth(self, 0)?;
        
        match self {
            Value::String(s) => validate_string(s, config)?,
            Value::List(list) => {
                validate_list(list, config)?;
                for item in list {
                    item.validate(config)?;
                }
            }
            Value::Function(name) => {
                validate_function_name(name, config)?;
            }
            Value::Real(f) => {
                if !f.is_finite() {
                    return Err(SecurityError::InvalidInput {
                        input_type: "real_number".to_string(),
                        reason: format!("Non-finite real number: {}", f),
                    });
                }
            }
            _ => {} // Other types are considered safe
        }
        
        Ok(())
    }
}

impl Validatable for String {
    fn validate(&self, config: &SecurityConfig) -> SecurityResult<()> {
        validate_string(self, config)
    }
}

impl<T: Validatable> Validatable for Vec<T> {
    fn validate(&self, config: &SecurityConfig) -> SecurityResult<()> {
        validate_list(self, config)?;
        for item in self {
            item.validate(config)?;
        }
        Ok(())
    }
}

impl Validatable for f64 {
    fn validate(&self, _config: &SecurityConfig) -> SecurityResult<()> {
        if !self.is_finite() {
            return Err(SecurityError::InvalidInput {
                input_type: "float".to_string(),
                reason: format!("Non-finite float: {}", self),
            });
        }
        Ok(())
    }
}

impl Validatable for usize {
    fn validate(&self, config: &SecurityConfig) -> SecurityResult<()> {
        if *self > config.max_tensor_size {
            return Err(SecurityError::InvalidInput {
                input_type: "size".to_string(),
                reason: format!("Size too large: {} > {}", self, config.max_tensor_size),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_validate_tensor_dimensions() {
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
    fn test_validate_string() {
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
    fn test_validate_list() {
        let config = SecurityConfig::default();
        
        // Valid list
        let short_list: Vec<i32> = (0..100).collect();
        assert!(validate_list(&short_list, &config).is_ok());
        
        // List too long
        let long_list: Vec<i32> = (0..config.max_list_length + 1).map(|i| i as i32).collect();
        assert!(validate_list(&long_list, &config).is_err());
    }
    
    #[test]
    fn test_validate_nesting_depth() {
        assert!(validate_nesting_depth(10).is_ok());
        assert!(validate_nesting_depth(1500).is_err());
    }
    
    #[test]
    fn test_validate_number_range() {
        assert!(validate_number_range(5.0, 0.0, 10.0).is_ok());
        assert!(validate_number_range(-1.0, 0.0, 10.0).is_err());
        assert!(validate_number_range(15.0, 0.0, 10.0).is_err());
        assert!(validate_number_range(f64::NAN, 0.0, 10.0).is_err());
        assert!(validate_number_range(f64::INFINITY, 0.0, 10.0).is_err());
    }
    
    #[test]
    fn test_validate_file_path() {
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
    fn test_validate_function_params() {
        let config = SecurityConfig::default();
        
        // Valid tensor creation
        let params = vec![
            Value::List(vec![Value::Integer(1), Value::Integer(2)]),
            Value::List(vec![Value::Integer(3), Value::Integer(4)]),
        ];
        assert!(validate_function_params("Tensor", &params, &config).is_ok());
        
        // Too many parameters
        let too_many_params: Vec<Value> = (0..20).map(|i| Value::Integer(i)).collect();
        assert!(validate_function_params("Tensor", &too_many_params, &config).is_err());
        
        // Valid range
        let range_params = vec![Value::Integer(1), Value::Integer(100)];
        assert!(validate_function_params("Range", &range_params, &config).is_ok());
        
        // Range parameter too large
        let large_range = vec![Value::Integer(2_000_000_000)];
        assert!(validate_function_params("Range", &large_range, &config).is_err());
    }
    
    #[test]
    fn test_sanitize_string() {
        assert_eq!(sanitize_string("hello\0world"), "helloworld");
        assert_eq!(sanitize_string("normal text"), "normal text");
        assert_eq!(sanitize_string("line1\nline2"), "line1\nline2");
    }
    
    #[test]
    fn test_validate_user_input() {
        let config = SecurityConfig::default();
        
        // Safe input
        assert!(validate_user_input("x + y", &config).is_ok());
        
        // Dangerous patterns
        assert!(validate_user_input("eval(malicious_code)", &config).is_err());
        assert!(validate_user_input("import os; os.system('rm -rf /')", &config).is_err());
        assert!(validate_user_input("subprocess.call(['rm', '-rf', '/'])", &config).is_err());
    }
    
    #[test]
    fn test_value_validation() {
        let config = SecurityConfig::default();
        
        // Valid values
        assert!(Value::Integer(42).validate(&config).is_ok());
        assert!(Value::String("hello".to_string()).validate(&config).is_ok());
        
        // Invalid values
        assert!(Value::Real(f64::NAN).validate(&config).is_err());
        
        let long_string = "a".repeat(config.max_string_length + 1);
        assert!(Value::String(long_string).validate(&config).is_err());
    }
}