/// Common utility functions and shared patterns across Lyra modules
/// 
/// This module contains extracted common functionality to reduce code duplication:
/// - Type checking utilities
/// - Collection manipulation helpers
/// - String and symbol processing
/// - Error handling patterns
/// - Validation helpers

use crate::vm::Value;
use crate::unified_errors::{LyraUnifiedError, LyraResult};
use std::collections::HashMap;

/// Shared type checking utilities
pub mod type_utils {
    use super::*;
    
    /// Check if a value matches a type name
    pub fn value_matches_type(value: &Value, type_name: &str) -> bool {
        match (value, type_name) {
            (Value::Integer(_), "Integer") => true,
            (Value::Real(_), "Real") => true,
            (Value::String(_), "String") => true,
            (Value::Symbol(_), "Symbol") => true,
            (Value::List(_), "List") => true,
            (Value::Function(_), "Function") => true,
            (Value::Boolean(_), "Boolean") => true,
            (Value::Missing, "Missing") => true,
            (Value::LyObj(_), "Object") => true,
            (Value::Quote(_), "Quote") => true,
            (Value::Pattern(_), "Pattern") => true,
            (Value::PureFunction { .. }, "PureFunction") => true,
            (Value::Slot { .. }, "Slot") => true,
            // Type coercion rules
            (Value::Integer(_), "Real") => true,
            _ => false,
        }
    }
    
    /// Get the canonical type name for a value
    pub fn get_type_name(value: &Value) -> &'static str {
        match value {
            Value::Integer(_) => "Integer",
            Value::Real(_) => "Real",
            Value::String(_) => "String",
            Value::Symbol(_) => "Symbol",
            Value::List(_) => "List",
            Value::Function(_) => "Function",
            Value::Boolean(_) => "Boolean",
            Value::Missing => "Missing",
            Value::Object(_) => "Object",
            Value::LyObj(_) => "LyObj",
            Value::Quote(_) => "Quote",
            Value::Pattern(_) => "Pattern",
            Value::Rule { .. } => "Rule",
            Value::PureFunction { .. } => "PureFunction",
            Value::Slot { .. } => "Slot",
        }
    }
    
    /// Check if two types are compatible (including coercion)
    pub fn types_compatible(from_type: &str, to_type: &str) -> bool {
        if from_type == to_type {
            return true;
        }
        
        // Common coercion rules
        match (from_type, to_type) {
            ("Integer", "Real") => true,
            ("List", list_type) if list_type.starts_with("List[") => true,
            _ => false,
        }
    }
    
    /// Parse complex type expressions like "List[Real]" into components
    pub fn parse_complex_type(type_expr: &str) -> Option<(String, Vec<String>)> {
        if let Some(open_bracket) = type_expr.find('[') {
            if let Some(close_bracket) = type_expr.rfind(']') {
                let base_type = type_expr[..open_bracket].to_string();
                let params_str = &type_expr[open_bracket + 1..close_bracket];
                
                let type_params: Vec<String> = params_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                
                return Some((base_type, type_params));
            }
        }
        None
    }
}

/// Collection manipulation helpers
pub mod collection_utils {
    use super::*;
    
    /// Safe index access for collections
    pub fn safe_index<T>(collection: &[T], index: i64) -> Option<&T> {
        if index < 0 {
            None
        } else {
            collection.get(index as usize)
        }
    }
    
    /// Safe mutable index access for collections
    pub fn safe_index_mut<T>(collection: &mut [T], index: i64) -> Option<&mut T> {
        if index < 0 {
            None
        } else {
            collection.get_mut(index as usize)
        }
    }
    
    /// Check if all elements in a collection satisfy a predicate
    pub fn all_match<T, F>(collection: &[T], predicate: F) -> bool 
    where 
        F: Fn(&T) -> bool
    {
        collection.iter().all(predicate)
    }
    
    /// Find the first element that matches a predicate
    pub fn find_first<T, F>(collection: &[T], predicate: F) -> Option<&T>
    where
        F: Fn(&T) -> bool
    {
        collection.iter().find(|item| predicate(item))
    }
    
    /// Group elements by a key function
    pub fn group_by<T, K, F>(collection: &[T], key_fn: F) -> HashMap<K, Vec<&T>>
    where
        K: Eq + std::hash::Hash,
        F: Fn(&T) -> K,
    {
        let mut groups = HashMap::new();
        for item in collection {
            let key = key_fn(item);
            groups.entry(key).or_insert_with(Vec::new).push(item);
        }
        groups
    }
    
    /// Safe slice operation for collections
    pub fn safe_slice<T>(collection: &[T], start: i64, end: i64) -> Option<&[T]> {
        let len = collection.len() as i64;
        
        // Handle negative indices
        let start_idx = if start < 0 { len + start } else { start };
        let end_idx = if end < 0 { len + end } else { end };
        
        // Bounds checking
        if start_idx < 0 || end_idx < start_idx || start_idx >= len {
            return None;
        }
        
        let end_clamped = end_idx.min(len);
        collection.get(start_idx as usize..end_clamped as usize)
    }
}

/// String and symbol processing utilities
pub mod string_utils {
    
    
    /// Normalize symbol names for consistent comparison
    pub fn normalize_symbol_name(name: &str) -> String {
        name.trim().to_string()
    }
    
    /// Check if a string is a valid identifier
    pub fn is_valid_identifier(name: &str) -> bool {
        if name.is_empty() {
            return false;
        }
        
        let first_char = name.chars().next().unwrap();
        if !first_char.is_alphabetic() && first_char != '_' && first_char != '$' {
            return false;
        }
        
        name.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '$')
    }
    
    /// Escape string for safe display
    pub fn escape_string(s: &str) -> String {
        s.chars()
            .map(|c| match c {
                '\n' => "\\n".to_string(),
                '\r' => "\\r".to_string(),
                '\t' => "\\t".to_string(),
                '\\' => "\\\\".to_string(),
                '"' => "\\\"".to_string(),
                c if c.is_control() => format!("\\u{{{:04x}}}", c as u32),
                c => c.to_string(),
            })
            .collect()
    }
    
    /// Parse qualified names like "Module::Function"
    pub fn parse_qualified_name(name: &str) -> (Option<String>, String) {
        if let Some(pos) = name.rfind("::") {
            let module = name[..pos].to_string();
            let function = name[pos + 2..].to_string();
            (Some(module), function)
        } else {
            (None, name.to_string())
        }
    }
    
    /// Generate unique names with counter
    pub fn make_unique_name(base_name: &str, existing_names: &std::collections::HashSet<String>) -> String {
        if !existing_names.contains(base_name) {
            return base_name.to_string();
        }
        
        let mut counter = 1;
        loop {
            let candidate = format!("{}_{}", base_name, counter);
            if !existing_names.contains(&candidate) {
                return candidate;
            }
            counter += 1;
        }
    }
}

/// Validation helpers
pub mod validation {
    use super::*;
    
    /// Validate function arity
    pub fn validate_arity(function_name: &str, expected: usize, actual: usize) -> LyraResult<()> {
        if expected != actual {
            Err(LyraUnifiedError::FunctionCall {
                function_name: function_name.to_string(),
                issue: crate::unified_errors::FunctionCallIssue::ArityMismatch { expected, actual },
                available_signatures: vec![],
            })
        } else {
            Ok(())
        }
    }
    
    /// Validate parameter types
    pub fn validate_parameter_types(
        function_name: &str,
        params: &[(String, Option<String>)],
        args: &[Value]
    ) -> LyraResult<()> {
        for (i, (param_name, expected_type_opt)) in params.iter().enumerate() {
            if let Some(expected_type) = expected_type_opt {
                if let Some(arg) = args.get(i) {
                    if !type_utils::value_matches_type(arg, expected_type) {
                        let actual_type = type_utils::get_type_name(arg);
                        return Err(LyraUnifiedError::FunctionCall {
                            function_name: function_name.to_string(),
                            issue: crate::unified_errors::FunctionCallIssue::TypeMismatch {
                                parameter: param_name.clone(),
                                expected: expected_type.clone(),
                                actual: actual_type.to_string(),
                            },
                            available_signatures: vec![],
                        });
                    }
                }
            }
        }
        Ok(())
    }
    
    /// Validate numeric range
    pub fn validate_numeric_range(
        value: f64, 
        min: Option<f64>, 
        max: Option<f64>,
        parameter_name: &str
    ) -> LyraResult<()> {
        if let Some(min_val) = min {
            if value < min_val {
                return Err(LyraUnifiedError::Validation {
                    constraint: format!("{} >= {}", parameter_name, min_val),
                    value: value.to_string(),
                    valid_range: Some(format!("[{}, {}]", 
                        min_val, 
                        max.map_or("∞".to_string(), |m| m.to_string())
                    )),
                });
            }
        }
        
        if let Some(max_val) = max {
            if value > max_val {
                return Err(LyraUnifiedError::Validation {
                    constraint: format!("{} <= {}", parameter_name, max_val),
                    value: value.to_string(),
                    valid_range: Some(format!("[{}, {}]", 
                        min.map_or("-∞".to_string(), |m| m.to_string()),
                        max_val
                    )),
                });
            }
        }
        
        Ok(())
    }
    
    /// Validate collection bounds
    pub fn validate_collection_bounds<T>(
        collection: &[T],
        index: i64,
        operation: &str
    ) -> LyraResult<()> {
        let len = collection.len();
        if index < 0 || index as usize >= len {
            Err(LyraUnifiedError::VmExecution {
                kind: crate::unified_errors::VmErrorKind::IndexOutOfBounds { 
                    index, 
                    length: len 
                },
                instruction_pointer: None,
                stack_trace: vec![operation.to_string()],
            })
        } else {
            Ok(())
        }
    }
}

/// Error handling patterns
pub mod error_utils {
    use super::*;
    
    /// Convert any error to LyraUnifiedError with context
    pub fn wrap_error<E: std::fmt::Display>(error: E, context: &str) -> LyraUnifiedError {
        LyraUnifiedError::Runtime {
            message: error.to_string(),
            context: crate::unified_errors::RuntimeContext {
                current_function: None,
                call_stack_depth: 0,
                local_variables: vec![],
                evaluation_mode: "normal".to_string(),
            },
            recoverable: true,
        }.with_context(context)
    }
    
    /// Chain multiple errors together
    pub fn chain_errors(primary: LyraUnifiedError, secondary: LyraUnifiedError, context: &str) -> LyraUnifiedError {
        primary.with_context(&format!("{}: {}", context, secondary))
    }
    
    /// Convert Result<T, E> to LyraResult<T> with context
    pub fn with_context<T, E: std::fmt::Display>(
        result: Result<T, E>, 
        context: &str
    ) -> LyraResult<T> {
        result.map_err(|e| wrap_error(e, context))
    }
}

/// Performance and caching utilities
pub mod perf_utils {
    use std::collections::HashMap;
    use std::hash::Hash;
    
    /// Simple LRU cache implementation
    pub struct LruCache<K: Hash + Eq + Clone, V: Clone> {
        capacity: usize,
        map: HashMap<K, (V, usize)>,
        access_count: usize,
    }
    
    impl<K: Hash + Eq + Clone, V: Clone> LruCache<K, V> {
        pub fn new(capacity: usize) -> Self {
            LruCache {
                capacity,
                map: HashMap::new(),
                access_count: 0,
            }
        }
        
        pub fn get(&mut self, key: &K) -> Option<V> {
            if let Some((value, access_time)) = self.map.get_mut(key) {
                self.access_count += 1;
                *access_time = self.access_count;
                Some(value.clone())
            } else {
                None
            }
        }
        
        pub fn put(&mut self, key: K, value: V) {
            self.access_count += 1;
            
            if self.map.len() >= self.capacity && !self.map.contains_key(&key) {
                // Find LRU item and remove it
                if let Some(lru_key) = self.map
                    .iter()
                    .min_by_key(|(_, (_, access))| *access)
                    .map(|(k, _)| k.clone())
                {
                    self.map.remove(&lru_key);
                }
            }
            
            self.map.insert(key, (value, self.access_count));
        }
        
        pub fn len(&self) -> usize {
            self.map.len()
        }
        
        pub fn is_empty(&self) -> bool {
            self.map.is_empty()
        }
        
        pub fn clear(&mut self) {
            self.map.clear();
            self.access_count = 0;
        }
    }
}

/// Common constants and shared values
pub mod constants {
    /// Mathematical constants
    pub const PI: f64 = std::f64::consts::PI;
    pub const E: f64 = std::f64::consts::E;
    pub const TAU: f64 = 2.0 * std::f64::consts::PI;
    
    /// System limits
    pub const MAX_STACK_DEPTH: usize = 10000;
    pub const MAX_RECURSION_DEPTH: usize = 1000;
    pub const MAX_LIST_SIZE: usize = 1_000_000;
    pub const MAX_STRING_LENGTH: usize = 100_000_000;
    
    /// Default cache sizes
    pub const DEFAULT_TYPE_CACHE_SIZE: usize = 1000;
    pub const DEFAULT_SYMBOL_CACHE_SIZE: usize = 10000;
    pub const DEFAULT_EXPRESSION_CACHE_SIZE: usize = 500;
    
    /// Common type names
    pub const BUILTIN_TYPES: &[&str] = &[
        "Integer", "Real", "String", "Symbol", "List", 
        "Function", "Boolean", "Missing", "Object", 
        "Quote", "Pattern"
    ];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_utils() {
        assert!(type_utils::value_matches_type(&Value::Integer(42), "Integer"));
        assert!(type_utils::value_matches_type(&Value::Integer(42), "Real")); // Coercion
        assert!(!type_utils::value_matches_type(&Value::String("test".to_string()), "Integer"));
        
        assert_eq!(type_utils::get_type_name(&Value::Real(3.14)), "Real");
        assert!(type_utils::types_compatible("Integer", "Real"));
        
        let (base, params) = type_utils::parse_complex_type("List[Integer]").unwrap();
        assert_eq!(base, "List");
        assert_eq!(params, vec!["Integer"]);
    }
    
    #[test]
    fn test_collection_utils() {
        let vec = vec![1, 2, 3, 4, 5];
        
        assert_eq!(collection_utils::safe_index(&vec, 2), Some(&3));
        assert_eq!(collection_utils::safe_index(&vec, -1), None);
        assert_eq!(collection_utils::safe_index(&vec, 10), None);
        
        assert!(collection_utils::all_match(&vec, |&x| x > 0));
        assert!(!collection_utils::all_match(&vec, |&x| x > 3));
        
        let slice = collection_utils::safe_slice(&vec, 1, 4).unwrap();
        assert_eq!(slice, &[2, 3, 4]);
    }
    
    #[test]
    fn test_string_utils() {
        assert!(string_utils::is_valid_identifier("validName"));
        assert!(!string_utils::is_valid_identifier("123invalid"));
        assert!(!string_utils::is_valid_identifier(""));
        
        assert_eq!(string_utils::normalize_symbol_name("  test  "), "test");
        assert_eq!(string_utils::escape_string("hello\nworld"), "hello\\nworld");
        
        let (module, function) = string_utils::parse_qualified_name("Math::Sin");
        assert_eq!(module, Some("Math".to_string()));
        assert_eq!(function, "Sin");
    }
    
    #[test]
    fn test_validation() {
        // Test arity validation
        assert!(validation::validate_arity("test", 2, 2).is_ok());
        assert!(validation::validate_arity("test", 2, 3).is_err());
        
        // Test numeric range validation
        assert!(validation::validate_numeric_range(5.0, Some(0.0), Some(10.0), "value").is_ok());
        assert!(validation::validate_numeric_range(-1.0, Some(0.0), Some(10.0), "value").is_err());
        assert!(validation::validate_numeric_range(15.0, Some(0.0), Some(10.0), "value").is_err());
    }
    
    #[test]
    fn test_lru_cache() {
        let mut cache = perf_utils::LruCache::new(2);
        
        cache.put("a", 1);
        cache.put("b", 2);
        
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"b"), Some(2));
        
        // Adding a third item should evict the LRU item
        cache.put("c", 3);
        assert_eq!(cache.len(), 2);
        
        // "a" was accessed more recently than "b", so "b" should be evicted
        assert_eq!(cache.get(&"a"), Some(1));
        assert_eq!(cache.get(&"c"), Some(3));
        assert_eq!(cache.get(&"b"), None);
    }
}