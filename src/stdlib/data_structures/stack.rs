//! Stack implementation for Lyra
//!
//! This module provides LIFO (Last In, First Out) stack operations.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::fmt;

/// LIFO Stack implementation
#[derive(Clone)]
pub struct Stack {
    data: Vec<Value>,
}

impl Stack {
    /// Create a new empty stack
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
        }
    }
    
    /// Create a stack from a list of values (bottom to top)
    pub fn from_list(values: Vec<Value>) -> Self {
        Self {
            data: values,
        }
    }
    
    /// Push an element onto the top of the stack
    pub fn push(&mut self, value: Value) {
        self.data.push(value);
    }
    
    /// Pop and return the top element from the stack
    pub fn pop(&mut self) -> Result<Value, ForeignError> {
        self.data.pop().ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot pop from empty stack".to_string(),
        })
    }
    
    /// Peek at the top element without removing it
    pub fn top(&self) -> Result<Value, ForeignError> {
        self.data.last().cloned().ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot peek at empty stack".to_string(),
        })
    }
    
    /// Get the size of the stack
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the stack is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Convert stack to a list (bottom to top)
    pub fn to_list(&self) -> Vec<Value> {
        self.data.clone()
    }
    
    /// Clear all elements from the stack
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Foreign for Stack {
    fn type_name(&self) -> &'static str {
        "Stack"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut stack = self.clone();
        match method {
            "push" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                stack.push(args[0].clone());
                Ok(Value::LyObj(LyObj::new(Box::new(stack))))
            }
            "pop" => {
                stack.pop()
            }
            "top" => {
                stack.top()
            }
            "size" => {
                Ok(Value::Integer(stack.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(stack.is_empty()))
            }
            "toList" => {
                let list = stack.to_list();
                Ok(Value::List(list))
            }
            "clear" => {
                stack.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(stack))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

impl fmt::Display for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stack[size: {}]", self.data.len())
    }
}

impl fmt::Debug for Stack {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Stack")
            .field("data", &self.data)
            .finish()
    }
}

/// Create a new empty stack
pub fn stack(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            let stack = Stack::new();
            Ok(Value::LyObj(LyObj::new(Box::new(stack))))
        }
        1 => {
            if let Value::List(list) = &args[0] {
                let stack = Stack::from_list(list.clone());
                Ok(Value::LyObj(LyObj::new(Box::new(stack))))
            } else {
                Err(VmError::Runtime("Stack expects a list argument".to_string()))
            }
        }
        _ => Err(VmError::Runtime("Stack takes 0 or 1 arguments".to_string()))
    }
}

/// Push an element onto a stack
pub fn push(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(stack_obj) = &args[0] {
        stack_obj.call_method("push", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a stack".to_string()))
    }
}

/// Pop and return the top element from a stack
pub fn pop(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(stack_obj) = &args[0] {
        stack_obj.call_method("pop", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a stack".to_string()))
    }
}

/// Peek at the top element of a stack
pub fn stack_top(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(stack_obj) = &args[0] {
        stack_obj.call_method("top", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a stack".to_string()))
    }
}

/// Register stack functions with the standard library
pub fn register_stack_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("Stack".to_string(), stack);
    functions.insert("Push".to_string(), push);
    functions.insert("Pop".to_string(), pop);
    functions.insert("StackTop".to_string(), stack_top);
}

/// Get documentation for stack functions
pub fn get_stack_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("Stack".to_string(), "Stack[] - Create empty stack. Stack[list] - Create stack from list.".to_string());
    docs.insert("Push".to_string(), "Push[stack, element] - Push element onto top of stack.".to_string());
    docs.insert("Pop".to_string(), "Pop[stack] - Remove and return top element from stack.".to_string());
    docs.insert("StackTop".to_string(), "StackTop[stack] - Peek at top element without removing.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_stack_creation() {
        let stack = Stack::new();
        assert_eq!(stack.size(), 0);
        assert!(stack.is_empty());
    }
    
    #[test]
    fn test_stack_operations() {
        let mut stack = Stack::new();
        
        // Push elements
        stack.push(Value::Integer(1));
        stack.push(Value::Integer(2));
        stack.push(Value::Integer(3));
        
        assert_eq!(stack.size(), 3);
        assert!(!stack.is_empty());
        
        // Peek at top
        assert_eq!(stack.top().unwrap(), Value::Integer(3));
        
        // Pop elements in LIFO order
        assert_eq!(stack.pop().unwrap(), Value::Integer(3));
        assert_eq!(stack.pop().unwrap(), Value::Integer(2));
        assert_eq!(stack.pop().unwrap(), Value::Integer(1));
        
        assert!(stack.is_empty());
    }
    
    #[test]
    fn test_stack_from_list() {
        let values = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let mut stack = Stack::from_list(values);
        
        assert_eq!(stack.size(), 3);
        // Top element should be the last in the list
        assert_eq!(stack.pop().unwrap(), Value::Integer(3));
        assert_eq!(stack.pop().unwrap(), Value::Integer(2));
        assert_eq!(stack.pop().unwrap(), Value::Integer(1));
    }
}