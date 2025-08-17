use crate::vm::Value;
use std::any::Any;
use std::fmt;
use thiserror::Error;

/// Error types for Foreign object operations
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ForeignError {
    #[error("Unknown method '{method}' for type '{type_name}'")]
    UnknownMethod {
        type_name: String,
        method: String,
    },
    
    #[error("Invalid arity for method '{method}': expected {expected}, got {actual}")]
    InvalidArity {
        method: String,
        expected: usize,
        actual: usize,
    },
    
    #[error("Invalid argument type for method '{method}': expected {expected}, got {actual}")]
    InvalidArgumentType {
        method: String,
        expected: String,
        actual: String,
    },
    
    #[error("Index out of bounds: {index} not in range {bounds}")]
    IndexOutOfBounds {
        index: String,
        bounds: String,
    },
    
    #[error("Runtime error: {message}")]
    RuntimeError {
        message: String,
    },
}

/// Trait for foreign objects that can be embedded in the VM
/// 
/// Foreign objects are opaque types implemented in the stdlib that can be
/// manipulated through method calls. This allows complex data types like
/// Table, Tensor, Series, and Dataset to live outside the VM core while
/// maintaining type safety and performance.
pub trait Foreign: fmt::Debug + Send + Sync {
    /// Get the type name for this foreign object
    fn type_name(&self) -> &'static str;
    
    /// Call a method on this foreign object
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError>;
    
    /// Clone this foreign object into a Box
    fn clone_boxed(&self) -> Box<dyn Foreign>;
    
    /// Get a reference to this object as Any for safe downcasting
    fn as_any(&self) -> &dyn Any;
    
    /// Optional: Serialize this object (for persistence)
    fn serialize(&self) -> Result<Vec<u8>, ForeignError> {
        Err(ForeignError::RuntimeError {
            message: format!("Serialization not implemented for {}", self.type_name()),
        })
    }
    
    /// Optional: Deserialize this object (for persistence)
    fn deserialize(_data: &[u8]) -> Result<Box<dyn Foreign>, ForeignError>
    where
        Self: Sized
    {
        Err(ForeignError::RuntimeError {
            message: "Deserialization not implemented".to_string(),
        })
    }
}

/// Wrapper type for Foreign objects in the VM Value system
/// 
/// LyObj provides a type-erased container for Foreign objects while
/// maintaining Clone, Debug, and PartialEq semantics through the Foreign trait.
#[derive(Debug)]
pub struct LyObj {
    inner: Box<dyn Foreign>,
}

impl LyObj {
    /// Create a new LyObj wrapper around a Foreign object
    pub fn new(foreign: Box<dyn Foreign>) -> Self {
        LyObj { inner: foreign }
    }
    
    /// Get the type name of the wrapped object
    pub fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }
    
    /// Call a method on the wrapped object
    pub fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        self.inner.call_method(method, args)
    }
    
    /// Get a reference to the wrapped Foreign object
    pub fn as_foreign(&self) -> &dyn Foreign {
        self.inner.as_ref()
    }
    
    /// Attempt to downcast to a specific type
    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.as_any().downcast_ref::<T>()
    }
    
    /// Get size information for debugging
    pub fn size_info(&self) -> String {
        format!("LyObj<{}>", self.type_name())
    }
}

impl Clone for LyObj {
    fn clone(&self) -> Self {
        LyObj {
            inner: self.inner.clone_boxed(),
        }
    }
}

impl PartialEq for LyObj {
    fn eq(&self, other: &Self) -> bool {
        // For now, we compare by type name and debug representation
        // Individual Foreign implementations can provide better equality
        // by implementing PartialEq on their concrete types
        if self.type_name() != other.type_name() {
            return false;
        }
        
        // Try to use the concrete type's PartialEq if available
        // This is a simplified approach - in practice, Foreign objects
        // should implement a custom equality method
        format!("{:?}", self.inner) == format!("{:?}", other.inner)
    }
}

impl fmt::Display for LyObj {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[...]", self.type_name())
    }
}

/// Utility functions for working with Foreign objects
impl LyObj {
    /// Check if this object supports a given method
    pub fn has_method(&self, method: &str) -> bool {
        // We can't easily check this without calling the method
        // Individual Foreign implementations might provide introspection
        self.call_method(method, &[]).is_ok() || 
        matches!(
            self.call_method(method, &[]),
            Err(ForeignError::InvalidArity { .. })
        )
    }
    
    /// Get a list of available methods (if supported by the Foreign type)
    pub fn list_methods(&self) -> Vec<String> {
        // This is a placeholder - Foreign objects could implement introspection
        match self.call_method("__methods__", &[]) {
            Ok(Value::List(methods)) => {
                methods.iter()
                    .filter_map(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect()
            }
            _ => vec![], // No introspection available
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple test Foreign implementation
    #[derive(Debug, Clone, PartialEq)]
    struct TestForeign {
        value: i64,
    }

    impl Foreign for TestForeign {
        fn type_name(&self) -> &'static str {
            "TestForeign"
        }

        fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
            match method {
                "get" => {
                    if !args.is_empty() {
                        return Err(ForeignError::InvalidArity {
                            method: method.to_string(),
                            expected: 0,
                            actual: args.len(),
                        });
                    }
                    Ok(Value::Integer(self.value))
                }
                "set" => {
                    if args.len() != 1 {
                        return Err(ForeignError::InvalidArity {
                            method: method.to_string(),
                            expected: 1,
                            actual: args.len(),
                        });
                    }
                    match &args[0] {
                        Value::Integer(n) => {
                            // Note: This would need interior mutability in practice
                            Ok(Value::Integer(*n))
                        }
                        _ => Err(ForeignError::InvalidArgumentType {
                            method: method.to_string(),
                            expected: "Integer".to_string(),
                            actual: format!("{:?}", args[0]),
                        }),
                    }
                }
                _ => Err(ForeignError::UnknownMethod {
                    type_name: self.type_name().to_string(),
                    method: method.to_string(),
                }),
            }
        }

        fn clone_boxed(&self) -> Box<dyn Foreign> {
            Box::new(self.clone())
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_foreign_trait_basic() {
        let foreign = TestForeign { value: 42 };
        
        assert_eq!(foreign.type_name(), "TestForeign");
        
        let result = foreign.call_method("get", &[]).unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_lyobj_wrapper() {
        let foreign = TestForeign { value: 123 };
        let lyobj = LyObj::new(Box::new(foreign));
        
        assert_eq!(lyobj.type_name(), "TestForeign");
        
        let result = lyobj.call_method("get", &[]).unwrap();
        assert_eq!(result, Value::Integer(123));
    }

    #[test]
    fn test_lyobj_clone_equality() {
        let foreign = TestForeign { value: 456 };
        let lyobj1 = LyObj::new(Box::new(foreign));
        let lyobj2 = lyobj1.clone();
        
        assert_eq!(lyobj1, lyobj2);
        
        // Both should work independently
        let result1 = lyobj1.call_method("get", &[]).unwrap();
        let result2 = lyobj2.call_method("get", &[]).unwrap();
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_foreign_error_types() {
        let foreign = TestForeign { value: 0 };
        
        // Test unknown method error
        let result = foreign.call_method("unknown", &[]);
        assert!(matches!(result, Err(ForeignError::UnknownMethod { .. })));
        
        // Test invalid arity error  
        let result = foreign.call_method("get", &[Value::Integer(1)]);
        assert!(matches!(result, Err(ForeignError::InvalidArity { .. })));
        
        // Test invalid argument type error
        let result = foreign.call_method("set", &[Value::String("invalid".to_string())]);
        assert!(matches!(result, Err(ForeignError::InvalidArgumentType { .. })));
    }

    #[test]
    fn test_downcast_safety() {
        let foreign = TestForeign { value: 789 };
        let lyobj = LyObj::new(Box::new(foreign));
        
        // Should successfully downcast to correct type
        let downcast = lyobj.downcast_ref::<TestForeign>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().value, 789);
        
        // Should fail to downcast to wrong type
        #[derive(Debug)]
        struct WrongType;
        let wrong_cast = lyobj.downcast_ref::<WrongType>();
        assert!(wrong_cast.is_none());
    }
}