//! Queue and Deque implementations for Lyra
//!
//! This module provides FIFO (First In, First Out) queue operations
//! and double-ended queue (deque) functionality.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// FIFO Queue implementation using VecDeque for efficient front/back operations
#[derive(Clone)]
pub struct Queue {
    data: VecDeque<Value>,
}

impl Queue {
    /// Create a new empty queue
    pub fn new() -> Self {
        Self {
            data: VecDeque::new(),
        }
    }
    
    /// Create a queue from a list of values
    pub fn from_list(values: Vec<Value>) -> Self {
        Self {
            data: values.into(),
        }
    }
    
    /// Add an element to the rear of the queue
    pub fn enqueue(&mut self, value: Value) {
        self.data.push_back(value);
    }
    
    /// Remove and return the element from the front of the queue
    pub fn dequeue(&mut self) -> Result<Value, ForeignError> {
        self.data.pop_front().ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot dequeue from empty queue".to_string(),
        })
    }
    
    /// Peek at the front element without removing it
    pub fn front(&self) -> Result<Value, ForeignError> {
        self.data.front().cloned().ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot peek at empty queue".to_string(),
        })
    }
    
    /// Peek at the rear element without removing it
    pub fn back(&self) -> Result<Value, ForeignError> {
        self.data.back().cloned().ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot peek at empty queue".to_string(),
        })
    }
    
    /// Get the size of the queue
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Convert queue to a list (front to back)
    pub fn to_list(&self) -> Vec<Value> {
        self.data.iter().cloned().collect()
    }
    
    /// Clear all elements from the queue
    pub fn clear(&mut self) {
        self.data.clear();
    }
}

impl Foreign for Queue {
    fn type_name(&self) -> &'static str {
        "Queue"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut queue = self.clone();
        match method {
            "enqueue" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                queue.enqueue(args[0].clone());
                Ok(Value::LyObj(LyObj::new(Box::new(queue))))
            }
            "dequeue" => {
                queue.dequeue()
            }
            "front" => {
                queue.front()
            }
            "back" => {
                queue.back()
            }
            "size" => {
                Ok(Value::Integer(queue.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(queue.is_empty()))
            }
            "toList" => {
                let list = queue.to_list();
                Ok(Value::List(list))
            }
            "clear" => {
                queue.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(queue))))
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

impl fmt::Display for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Queue[size: {}]", self.data.len())
    }
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue")
            .field("data", &self.data)
            .finish()
    }
}

/// Create a new empty queue
pub fn queue(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            let queue = Queue::new();
            Ok(Value::LyObj(LyObj::new(Box::new(queue))))
        }
        1 => {
            if let Value::List(list) = &args[0] {
                let queue = Queue::from_list(list.clone());
                Ok(Value::LyObj(LyObj::new(Box::new(queue))))
            } else {
                Err(VmError::Runtime("Queue expects a list argument".to_string()))
            }
        }
        _ => Err(VmError::Runtime("Queue takes 0 or 1 arguments".to_string()))
    }
}

/// Add an element to the rear of a queue
pub fn enqueue(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("enqueue", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a queue".to_string()))
    }
}

/// Remove and return the front element from a queue
pub fn dequeue(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("dequeue", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a queue".to_string()))
    }
}

/// Peek at the front element of a queue
pub fn queue_front(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("front", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a queue".to_string()))
    }
}

/// Register queue functions with the standard library
pub fn register_queue_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("Queue".to_string(), queue);
    functions.insert("Enqueue".to_string(), enqueue);
    functions.insert("Dequeue".to_string(), dequeue);
    functions.insert("QueueFront".to_string(), queue_front);
}

/// Get documentation for queue functions
pub fn get_queue_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("Queue".to_string(), "Queue[] - Create empty queue. Queue[list] - Create queue from list.".to_string());
    docs.insert("Enqueue".to_string(), "Enqueue[queue, element] - Add element to rear of queue.".to_string());
    docs.insert("Dequeue".to_string(), "Dequeue[queue] - Remove and return front element from queue.".to_string());
    docs.insert("QueueFront".to_string(), "QueueFront[queue] - Peek at front element without removing.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_queue_creation() {
        let queue = Queue::new();
        assert_eq!(queue.size(), 0);
        assert!(queue.is_empty());
    }
    
    #[test]
    fn test_queue_operations() {
        let mut queue = Queue::new();
        
        // Enqueue elements
        queue.enqueue(Value::Integer(1));
        queue.enqueue(Value::Integer(2));
        queue.enqueue(Value::Integer(3));
        
        assert_eq!(queue.size(), 3);
        assert!(!queue.is_empty());
        
        // Peek at front
        assert_eq!(queue.front().unwrap(), Value::Integer(1));
        
        // Dequeue elements in FIFO order
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(1));
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(2));
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(3));
        
        assert!(queue.is_empty());
    }
    
    #[test]
    fn test_queue_from_list() {
        let values = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let mut queue = Queue::from_list(values);
        
        assert_eq!(queue.size(), 3);
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(1));
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(2));
        assert_eq!(queue.dequeue().unwrap(), Value::Integer(3));
    }
}