//! Foreign object utilities and patterns for Lyra standard library
//!
//! This module provides utilities and macros for implementing Foreign objects
//! consistently across all stdlib modules, including method dispatch, 
//! error handling, and common patterns.

use crate::vm::Value;
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;
use std::fmt;

/// Trait for Foreign objects that provides additional utility methods
pub trait ForeignObjectTemplate: Foreign {
    /// Get all available methods for this object type
    fn available_methods(&self) -> Vec<&'static str>;
    
    /// Get method documentation
    fn method_help(&self, method: &str) -> Option<&'static str>;
    
    /// Check if a method exists
    fn has_method(&self, method: &str) -> bool {
        self.available_methods().contains(&method)
    }
}

/// Macro for implementing Foreign trait with method dispatch
#[macro_export]
macro_rules! impl_foreign {
    ($type:ty, $type_name:expr, {
        $($method:ident($($param:ident: $param_type:ty),*) -> $return_type:ty => $body:expr),* $(,)?
    }) => {
        impl crate::foreign::Foreign for $type {
            fn type_name(&self) -> &'static str {
                $type_name
            }
            
            fn call_method(&self, method: &str, args: &[crate::vm::Value]) -> Result<crate::vm::Value, crate::foreign::ForeignError> {
                match method {
                    $(
                        stringify!($method) => {
                            // Validate argument count
                            let expected_count = impl_foreign!(@count_params $($param),*);
                            if args.len() != expected_count {
                                return Err(crate::foreign::ForeignError::InvalidArity {
                                    method: method.to_string(),
                                    expected: expected_count,
                                    actual: args.len(),
                                });
                            }
                            
                            // Extract and convert arguments
                            #[allow(unused_variables, unused_mut)]
                            let mut arg_index = 0;
                            $(
                                let $param = impl_foreign!(@extract_arg args, arg_index, $param_type, method)?;
                                arg_index += 1;
                            )*
                            
                            // Call the method implementation
                            let result: $return_type = $body;
                            impl_foreign!(@convert_result result)
                        }
                    )*
                    _ => Err(crate::foreign::ForeignError::UnknownMethod {
                        method: method.to_string(),
                        type_name: $type_name.to_string(),
                    })
                }
            }
            
            fn clone_boxed(&self) -> Box<dyn crate::foreign::Foreign> {
                Box::new(self.clone())
            }
            
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        
        impl $crate::stdlib::common::foreign_utils::ForeignObjectTemplate for $type {
            fn available_methods(&self) -> Vec<&'static str> {
                vec![$(stringify!($method)),*]
            }
            
            fn method_help(&self, method: &str) -> Option<&'static str> {
                match method {
                    $(stringify!($method) => Some(concat!(stringify!($method), "(", impl_foreign!(@param_help $($param: $param_type),*), ")"))),*
                    _ => None,
                }
            }
        }
    };
    
    // Helper macro to count parameters
    (@count_params) => { 0 };
    (@count_params $param:ident) => { 1 };
    (@count_params $param:ident, $($rest:ident),*) => { 1 + impl_foreign!(@count_params $($rest),*) };
    
    // Helper macro to extract arguments
    (@extract_arg $args:expr, $index:expr, i64, $method:expr) => {
        match &$args[$index] {
            crate::vm::Value::Integer(i) => *i,
            _ => return Err(crate::foreign::ForeignError::InvalidArgumentType {
                method: $method.to_string(),
                expected: "Integer".to_string(),
                actual: format!("{:?}", &$args[$index]),
            }),
        }
    };
    
    (@extract_arg $args:expr, $index:expr, f64, $method:expr) => {
        match &$args[$index] {
            crate::vm::Value::Real(r) => *r,
            crate::vm::Value::Integer(i) => *i as f64,
            _ => return Err(crate::foreign::ForeignError::InvalidArgumentType {
                method: $method.to_string(),
                expected: "Number".to_string(),
                actual: format!("{:?}", &$args[$index]),
            }),
        }
    };
    
    (@extract_arg $args:expr, $index:expr, String, $method:expr) => {
        match &$args[$index] {
            crate::vm::Value::String(s) => s.clone(),
            _ => return Err(crate::foreign::ForeignError::InvalidArgumentType {
                method: $method.to_string(),
                expected: "String".to_string(),
                actual: format!("{:?}", &$args[$index]),
            }),
        }
    };
    
    (@extract_arg $args:expr, $index:expr, Vec<crate::vm::Value>, $method:expr) => {
        match &$args[$index] {
            crate::vm::Value::List(l) => l.clone(),
            _ => return Err(crate::foreign::ForeignError::InvalidArgumentType {
                method: $method.to_string(),
                expected: "List".to_string(),
                actual: format!("{:?}", &$args[$index]),
            }),
        }
    };
    
    // Helper macro to convert results back to Value
    (@convert_result $result:expr) => {
        match $result {
            Ok(value) => impl_foreign!(@value_from value),
            Err(e) => Err(e),
        }
    };
    
    (@value_from $value:expr) => {{
        let val = $value;
        impl_foreign!(@to_value val)
    }};
    
    (@to_value $val:expr) => {{
        let v = $val;
        // This will need to be implemented based on the actual return type
        // For now, we'll use a placeholder
        Ok(crate::vm::Value::Integer(0)) // TODO: Implement proper conversion
    }};
    
    // Helper for parameter documentation
    (@param_help) => { "" };
    (@param_help $param:ident: $param_type:ty) => { concat!(stringify!($param), ": ", stringify!($param_type)) };
    (@param_help $param:ident: $param_type:ty, $($rest:ident: $rest_type:ty),*) => {
        concat!(stringify!($param), ": ", stringify!($param_type), ", ", impl_foreign!(@param_help $($rest: $rest_type),*))
    };
}

/// Macro for creating a Foreign object method dispatcher
#[macro_export]
macro_rules! foreign_method {
    ($self:expr, $method:expr, $args:expr, {
        $($method_name:literal => $handler:expr),* $(,)?
    }) => {
        match $method {
            $($method_name => $handler,)*
            _ => Err(crate::foreign::ForeignError::UnknownMethod {
                method: $method.to_string(),
                type_name: $self.type_name().to_string(),
            })
        }
    };
}

/// Helper function to convert a Value to a specific type safely
pub fn extract_value<T>(value: &Value) -> Result<T, ForeignError>
where
    T: for<'a> TryFrom<&'a Value>,
    for<'a> <T as TryFrom<&'a Value>>::Error: fmt::Display,
{
    T::try_from(value).map_err(|e| ForeignError::RuntimeError {
        message: format!("Type conversion error: {}", e),
    })
}

/// Helper function to validate argument count
pub fn validate_arity(method: &str, expected: usize, actual: usize) -> Result<(), ForeignError> {
    if expected != actual {
        Err(ForeignError::InvalidArity {
            method: method.to_string(),
            expected,
            actual,
        })
    } else {
        Ok(())
    }
}

/// Helper function to create a method not found error
pub fn method_not_found(type_name: &str, method: &str) -> ForeignError {
    ForeignError::UnknownMethod {
        type_name: type_name.to_string(),
        method: method.to_string(),
    }
}

/// Helper function to create an invalid argument error
pub fn invalid_argument(message: &str) -> ForeignError {
    ForeignError::InvalidArgument(message.to_string())
}

/// Helper function to create a runtime error
pub fn runtime_error(message: &str) -> ForeignError {
    ForeignError::RuntimeError {
        message: message.to_string(),
    }
}

/// Convenience function to wrap a Foreign object in a LyObj Value
pub fn wrap_foreign<T: Foreign + 'static>(object: T) -> Value {
    Value::LyObj(LyObj::new(Box::new(object)))
}

/// Convenience function to unwrap a Foreign object from a Value
pub fn unwrap_foreign<T: 'static>(value: &Value) -> Result<&T, ForeignError> {
    match value {
        Value::LyObj(obj) => {
            // Use pattern matching to access the inner Foreign object
            obj.as_foreign().as_any()
                .downcast_ref::<T>()
                .ok_or_else(|| runtime_error("Invalid object type"))
        }
        _ => Err(runtime_error("Expected object, got primitive value")),
    }
}

/// Trait for converting Rust types to Lyra Values
pub trait ToLyraValue {
    fn to_lyra_value(self) -> Value;
}

impl ToLyraValue for i64 {
    fn to_lyra_value(self) -> Value {
        Value::Integer(self)
    }
}

impl ToLyraValue for f64 {
    fn to_lyra_value(self) -> Value {
        Value::Real(self)
    }
}

impl ToLyraValue for String {
    fn to_lyra_value(self) -> Value {
        Value::String(self)
    }
}

impl ToLyraValue for &str {
    fn to_lyra_value(self) -> Value {
        Value::String(self.to_string())
    }
}

impl ToLyraValue for Vec<Value> {
    fn to_lyra_value(self) -> Value {
        Value::List(self)
    }
}

impl ToLyraValue for bool {
    fn to_lyra_value(self) -> Value {
        Value::Symbol(if self { "True".to_string() } else { "False".to_string() })
    }
}

/// Trait for converting Lyra Values to Rust types
pub trait FromLyraValue: Sized {
    fn from_lyra_value(value: &Value) -> Result<Self, ForeignError>;
}

impl FromLyraValue for i64 {
    fn from_lyra_value(value: &Value) -> Result<Self, ForeignError> {
        match value {
            Value::Integer(i) => Ok(*i),
            _ => Err(runtime_error("Expected integer")),
        }
    }
}

impl FromLyraValue for f64 {
    fn from_lyra_value(value: &Value) -> Result<Self, ForeignError> {
        match value {
            Value::Real(r) => Ok(*r),
            Value::Integer(i) => Ok(*i as f64),
            _ => Err(runtime_error("Expected number")),
        }
    }
}

impl FromLyraValue for String {
    fn from_lyra_value(value: &Value) -> Result<Self, ForeignError> {
        match value {
            Value::String(s) => Ok(s.clone()),
            _ => Err(runtime_error("Expected string")),
        }
    }
}

impl FromLyraValue for Vec<Value> {
    fn from_lyra_value(value: &Value) -> Result<Self, ForeignError> {
        match value {
            Value::List(l) => Ok(l.clone()),
            _ => Err(runtime_error("Expected list")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[derive(Debug, Clone)]
    struct TestObject {
        value: i64,
    }
    
    // Example usage of the foreign_method macro
    impl Foreign for TestObject {
        fn type_name(&self) -> &'static str {
            "TestObject"
        }
        
        fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
            foreign_method!(self, method, args, {
                "getValue" => {
                    validate_arity(method, 0, args.len())?;
                    Ok(Value::Integer(self.value))
                },
                "setValue" => {
                    validate_arity(method, 1, args.len())?;
                    let new_value = i64::from_lyra_value(&args[0])?;
                    // Note: This would need to be mutable in practice
                    Ok(Value::Integer(new_value))
                },
            })
        }
        
        fn clone_boxed(&self) -> Box<dyn Foreign> {
            Box::new(self.clone())
        }
        
        fn as_any(&self) -> &dyn Any {
            self
        }
    }
    
    #[test]
    fn test_foreign_object_utilities() {
        let obj = TestObject { value: 42 };
        let value = wrap_foreign(obj);
        
        match &value {
            Value::LyObj(_) => (),
            _ => panic!("Expected LyObj"),
        }
        
        let unwrapped = unwrap_foreign::<TestObject>(&value).unwrap();
        assert_eq!(unwrapped.value, 42);
    }
    
    #[test]
    fn test_type_conversions() {
        let int_value = Value::Integer(42);
        let extracted = i64::from_lyra_value(&int_value).unwrap();
        assert_eq!(extracted, 42);
        
        let converted = extracted.to_lyra_value();
        assert_eq!(converted, int_value);
    }
}