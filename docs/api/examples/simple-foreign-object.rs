// Simple Foreign Object Example
//
// This example shows how to create a basic Foreign object that integrates
// with Lyra's VM. We'll create a simple Counter object that demonstrates
// the essential patterns.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// A simple counter that can be incremented and decremented
#[derive(Debug, Clone, PartialEq)]
pub struct Counter {
    value: i64,
}

impl Counter {
    /// Create a new counter starting at zero
    pub fn new() -> Self {
        Counter { value: 0 }
    }

    /// Create a new counter with a specific starting value
    pub fn with_value(value: i64) -> Self {
        Counter { value }
    }

    /// Get the current value
    pub fn get(&self) -> i64 {
        self.value
    }

    /// Increment by one and return the new value
    pub fn increment(&mut self) -> i64 {
        self.value += 1;
        self.value
    }

    /// Decrement by one and return the new value
    pub fn decrement(&mut self) -> i64 {
        self.value -= 1;
        self.value
    }

    /// Add a specific amount
    pub fn add(&mut self, amount: i64) -> i64 {
        self.value += amount;
        self.value
    }

    /// Reset to zero
    pub fn reset(&mut self) {
        self.value = 0;
    }
}

impl Foreign for Counter {
    fn type_name(&self) -> &'static str {
        "Counter"
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

            "add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let amount = match &args[0] {
                    Value::Integer(n) => *n,
                    Value::Real(f) => *f as i64,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer or Real".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                // Note: This creates a new Counter since Foreign trait methods take &self
                let mut new_counter = self.clone();
                let new_value = new_counter.add(amount);
                Ok(Value::LyObj(LyObj::new(Box::new(new_counter))))
            }

            "increment" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let mut new_counter = self.clone();
                new_counter.increment();
                Ok(Value::LyObj(LyObj::new(Box::new(new_counter))))
            }

            "decrement" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let mut new_counter = self.clone();
                new_counter.decrement();
                Ok(Value::LyObj(LyObj::new(Box::new(new_counter))))
            }

            "reset" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let mut new_counter = self.clone();
                new_counter.reset();
                Ok(Value::LyObj(LyObj::new(Box::new(new_counter))))
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

// Required for thread safety
unsafe impl Send for Counter {}
unsafe impl Sync for Counter {}

// Stdlib function to create Counter objects
use crate::stdlib::StdlibFunction;

pub fn create_counter(args: &[Value]) -> VmResult<Value> {
    match args {
        // Counter[] - create with default value
        [] => {
            let counter = Counter::new();
            Ok(Value::LyObj(LyObj::new(Box::new(counter))))
        }
        
        // Counter[value] - create with specific value
        [Value::Integer(value)] => {
            let counter = Counter::with_value(*value);
            Ok(Value::LyObj(LyObj::new(Box::new(counter))))
        }
        
        [Value::Real(value)] => {
            let counter = Counter::with_value(*value as i64);
            Ok(Value::LyObj(LyObj::new(Box::new(counter))))
        }
        
        _ => Err(VmError::TypeError {
            expected: "Counter[] or Counter[value]".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter_creation() {
        let counter = Counter::new();
        assert_eq!(counter.get(), 0);

        let counter = Counter::with_value(42);
        assert_eq!(counter.get(), 42);
    }

    #[test]
    fn test_counter_operations() {
        let mut counter = Counter::new();
        
        assert_eq!(counter.increment(), 1);
        assert_eq!(counter.increment(), 2);
        assert_eq!(counter.add(10), 12);
        assert_eq!(counter.decrement(), 11);
        
        counter.reset();
        assert_eq!(counter.get(), 0);
    }

    #[test]
    fn test_foreign_methods() {
        let counter = Counter::with_value(5);

        // Test get method
        let result = counter.call_method("get", &[]).unwrap();
        assert_eq!(result, Value::Integer(5));

        // Test add method
        let result = counter.call_method("add", &[Value::Integer(3)]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(new_counter) = obj.downcast_ref::<Counter>() {
                assert_eq!(new_counter.get(), 8);
            }
        }

        // Test increment method
        let result = counter.call_method("increment", &[]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(new_counter) = obj.downcast_ref::<Counter>() {
                assert_eq!(new_counter.get(), 6);
            }
        }
    }

    #[test]
    fn test_error_handling() {
        let counter = Counter::new();

        // Test unknown method
        let result = counter.call_method("unknown", &[]);
        assert!(matches!(result, Err(ForeignError::UnknownMethod { .. })));

        // Test invalid arity
        let result = counter.call_method("get", &[Value::Integer(1)]);
        assert!(matches!(result, Err(ForeignError::InvalidArity { .. })));

        // Test invalid argument type
        let result = counter.call_method("add", &[Value::String("invalid".to_string())]);
        assert!(matches!(result, Err(ForeignError::InvalidArgumentType { .. })));
    }

    #[test]
    fn test_stdlib_function() {
        // Test Counter[] 
        let result = create_counter(&[]).unwrap();
        if let Value::LyObj(obj) = result {
            assert_eq!(obj.type_name(), "Counter");
            if let Some(counter) = obj.downcast_ref::<Counter>() {
                assert_eq!(counter.get(), 0);
            }
        }

        // Test Counter[42]
        let result = create_counter(&[Value::Integer(42)]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(counter) = obj.downcast_ref::<Counter>() {
                assert_eq!(counter.get(), 42);
            }
        }
    }
}

// Example usage:
//
// // Create a counter
// let counter = Counter::new();
// let counter_value = Value::LyObj(LyObj::new(Box::new(counter)));
//
// // Use it through method calls
// if let Value::LyObj(obj) = counter_value {
//     let current = obj.call_method("get", &[])?;        // Returns 0
//     let incremented = obj.call_method("increment", &[])?; // Returns Counter with value 1
//     let added = obj.call_method("add", &[Value::Integer(5)])?; // Returns Counter with value 5
// }
//
// // Or register as a stdlib function and use in Lyra code:
// // counter = Counter[10]
// // counter.get()        (* Returns 10 *)
// // counter.increment()  (* Returns Counter with value 11 *)
// // counter.add(5)       (* Returns Counter with value 15 *)