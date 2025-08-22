//! Priority Queue implementation for Lyra
//!
//! This module provides priority queue data structures with custom comparators
//! and priority management. Built on the Foreign object pattern for VM integration.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::{BinaryHeap, HashMap};
use std::cmp::Ordering;
use std::fmt;

/// Priority queue item with custom priority and value
#[derive(Clone, Debug)]
struct PriorityItem {
    priority: f64,
    value: Value,
    insertion_order: usize, // For stable ordering when priorities are equal
}

impl PartialEq for PriorityItem {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.insertion_order == other.insertion_order
    }
}

impl Eq for PriorityItem {}

impl PartialOrd for PriorityItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority comes first, then by insertion order for stability
        other.priority.partial_cmp(&self.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.insertion_order.cmp(&other.insertion_order))
    }
}

/// Priority queue with custom priorities and comparators
#[derive(Clone)]
pub struct PriorityQueue {
    heap: BinaryHeap<PriorityItem>,
    next_order: usize,
    is_min_queue: bool, // True for min-priority queue, false for max-priority
}

impl PriorityQueue {
    /// Create a new priority queue
    pub fn new(is_min_queue: bool) -> Self {
        Self {
            heap: BinaryHeap::new(),
            next_order: 0,
            is_min_queue,
        }
    }

    /// Insert an item with explicit priority
    pub fn insert(&mut self, value: Value, priority: f64) -> Result<(), ForeignError> {
        let adjusted_priority = if self.is_min_queue {
            -priority  // Negate for min-queue behavior in max-heap
        } else {
            priority
        };

        let item = PriorityItem {
            priority: adjusted_priority,
            value,
            insertion_order: self.next_order,
        };
        
        self.next_order += 1;
        self.heap.push(item);
        Ok(())
    }

    /// Extract the highest (or lowest for min-queue) priority item
    pub fn extract(&mut self) -> Result<Value, ForeignError> {
        match self.heap.pop() {
            Some(item) => Ok(item.value),
            None => Err(ForeignError::RuntimeError {
                message: "Cannot extract from empty priority queue".to_string(),
            }),
        }
    }

    /// Peek at the highest priority item without removing it
    pub fn peek(&self) -> Result<Value, ForeignError> {
        match self.heap.peek() {
            Some(item) => Ok(item.value.clone()),
            None => Err(ForeignError::RuntimeError {
                message: "Cannot peek at empty priority queue".to_string(),
            }),
        }
    }

    /// Get the priority of the top item
    pub fn peek_priority(&self) -> Result<f64, ForeignError> {
        match self.heap.peek() {
            Some(item) => {
                let priority = if self.is_min_queue {
                    -item.priority
                } else {
                    item.priority
                };
                Ok(priority)
            }
            None => Err(ForeignError::RuntimeError {
                message: "Cannot peek priority of empty queue".to_string(),
            }),
        }
    }

    /// Get the size of the queue
    pub fn size(&self) -> usize {
        self.heap.len()
    }

    /// Check if the queue is empty
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Clear all items from the queue
    pub fn clear(&mut self) {
        self.heap.clear();
        self.next_order = 0;
    }

    /// Check if queue contains a specific value
    pub fn contains(&self, target: &Value) -> bool {
        self.heap.iter().any(|item| &item.value == target)
    }

    /// Convert to sorted list (by priority)
    pub fn to_sorted_list(&self) -> Vec<Value> {
        let mut items: Vec<_> = self.heap.iter().collect();
        items.sort_by(|a, b| a.cmp(b));
        items.into_iter().map(|item| item.value.clone()).collect()
    }

    /// Merge another priority queue into this one
    pub fn merge(&mut self, other: &PriorityQueue) -> Result<(), ForeignError> {
        if self.is_min_queue != other.is_min_queue {
            return Err(ForeignError::RuntimeError {
                message: "Cannot merge priority queues of different types (min vs max)".to_string(),
            });
        }

        for item in other.heap.iter() {
            let mut new_item = item.clone();
            new_item.insertion_order = self.next_order;
            self.next_order += 1;
            self.heap.push(new_item);
        }
        
        Ok(())
    }
}

impl Foreign for PriorityQueue {
    fn type_name(&self) -> &'static str {
        if self.is_min_queue {
            "MinPriorityQueue"
        } else {
            "PriorityQueue"
        }
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut queue = self.clone();
        match method {
            "insert" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let priority = match &args[1] {
                    Value::Integer(i) => *i as f64,
                    Value::Real(r) => *r,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Number".to_string(),
                        actual: "other".to_string(),
                    }),
                };

                queue.insert(args[0].clone(), priority)?;
                Ok(Value::LyObj(LyObj::new(Box::new(queue))))
            }
            "extract" => {
                queue.extract()
            }
            "peek" => {
                queue.peek()
            }
            "peekPriority" => {
                let priority = queue.peek_priority()?;
                Ok(Value::Real(priority))
            }
            "size" => {
                Ok(Value::Integer(queue.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(queue.is_empty()))
            }
            "clear" => {
                queue.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(queue))))
            }
            "contains" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                Ok(Value::Boolean(queue.contains(&args[0])))
            }
            "toList" => {
                let list = queue.to_sorted_list();
                Ok(Value::List(list))
            }
            "merge" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                
                if let Value::LyObj(obj) = &args[0] {
                    if let Some(other_queue) = obj.downcast_ref::<PriorityQueue>() {
                        queue.merge(other_queue)?;
                        return Ok(Value::LyObj(LyObj::new(Box::new(queue))));
                    }
                }
                
                Err(ForeignError::InvalidArgumentType {
                    method: method.to_string(),
                    expected: "PriorityQueue".to_string(),
                    actual: "other".to_string(),
                })
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

impl fmt::Display for PriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let queue_type = if self.is_min_queue { "MinPriorityQueue" } else { "PriorityQueue" };
        write!(f, "{}[size: {}]", queue_type, self.heap.len())
    }
}

impl fmt::Debug for PriorityQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PriorityQueue")
            .field("heap_size", &self.heap.len())
            .field("is_min_queue", &self.is_min_queue)
            .field("next_order", &self.next_order)
            .finish()
    }
}

/// Create a new max-priority queue
pub fn priority_queue(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            let queue = PriorityQueue::new(false); // Max-priority queue by default
            Ok(Value::LyObj(LyObj::new(Box::new(queue))))
        }
        1 => {
            // Check for min-queue flag
            match &args[0] {
                Value::Boolean(is_min) => {
                    let queue = PriorityQueue::new(*is_min);
                    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
                }
                _ => Err(VmError::Runtime("PriorityQueue expects boolean argument for min-queue flag".to_string()))
            }
        }
        _ => Err(VmError::Runtime("PriorityQueue takes 0 or 1 arguments".to_string()))
    }
}

/// Create a new min-priority queue
pub fn min_priority_queue(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime("MinPriorityQueue takes no arguments".to_string()));
    }
    
    let queue = PriorityQueue::new(true);
    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
}

/// Insert an item with priority into a priority queue
pub fn pq_insert(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("insert", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a priority queue".to_string()))
    }
}

/// Extract the highest priority item from a priority queue
pub fn pq_extract(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("extract", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a priority queue".to_string()))
    }
}

/// Peek at the highest priority item without removing it
pub fn pq_peek(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("peek", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a priority queue".to_string()))
    }
}

/// Get the priority of the top item
pub fn pq_peek_priority(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("peekPriority", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a priority queue".to_string()))
    }
}

/// Get the size of a priority queue
pub fn pq_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("size", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a priority queue".to_string()))
    }
}

/// Check if a priority queue is empty
pub fn pq_is_empty(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("isEmpty", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a priority queue".to_string()))
    }
}

/// Check if priority queue contains a value
pub fn pq_contains(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("contains", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a priority queue".to_string()))
    }
}

/// Merge two priority queues
pub fn pq_merge(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(queue_obj) = &args[0] {
        queue_obj.call_method("merge", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a priority queue".to_string()))
    }
}

/// Register priority queue functions with the standard library
pub fn register_priority_queue_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("PriorityQueue".to_string(), priority_queue);
    functions.insert("MinPriorityQueue".to_string(), min_priority_queue);
    functions.insert("PQInsert".to_string(), pq_insert);
    functions.insert("PQExtract".to_string(), pq_extract);
    functions.insert("PQPeek".to_string(), pq_peek);
    functions.insert("PQPeekPriority".to_string(), pq_peek_priority);
    functions.insert("PQSize".to_string(), pq_size);
    functions.insert("PQIsEmpty".to_string(), pq_is_empty);
    functions.insert("PQContains".to_string(), pq_contains);
    functions.insert("PQMerge".to_string(), pq_merge);
}

/// Get documentation for priority queue functions
pub fn get_priority_queue_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("PriorityQueue".to_string(), "PriorityQueue[] - Create empty max-priority queue. PriorityQueue[True] - Create min-priority queue.".to_string());
    docs.insert("MinPriorityQueue".to_string(), "MinPriorityQueue[] - Create empty min-priority queue.".to_string());
    docs.insert("PQInsert".to_string(), "PQInsert[pq, item, priority] - Insert item with priority. Returns new queue.".to_string());
    docs.insert("PQExtract".to_string(), "PQExtract[pq] - Remove and return highest priority item.".to_string());
    docs.insert("PQPeek".to_string(), "PQPeek[pq] - View highest priority item without removing it.".to_string());
    docs.insert("PQPeekPriority".to_string(), "PQPeekPriority[pq] - View priority of top item.".to_string());
    docs.insert("PQSize".to_string(), "PQSize[pq] - Get number of items in queue.".to_string());
    docs.insert("PQIsEmpty".to_string(), "PQIsEmpty[pq] - Check if queue is empty.".to_string());
    docs.insert("PQContains".to_string(), "PQContains[pq, item] - Check if queue contains item.".to_string());
    docs.insert("PQMerge".to_string(), "PQMerge[pq1, pq2] - Merge two priority queues of same type.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_priority_queue_creation() {
        let pq = PriorityQueue::new(false);
        assert_eq!(pq.size(), 0);
        assert!(pq.is_empty());
        assert!(!pq.is_min_queue);
    }
    
    #[test]
    fn test_min_priority_queue_creation() {
        let pq = PriorityQueue::new(true);
        assert_eq!(pq.size(), 0);
        assert!(pq.is_empty());
        assert!(pq.is_min_queue);
    }
    
    #[test]
    fn test_max_priority_queue_operations() {
        let mut pq = PriorityQueue::new(false);
        
        // Insert items with different priorities
        pq.insert(Value::String("low".to_string()), 1.0).unwrap();
        pq.insert(Value::String("high".to_string()), 3.0).unwrap();
        pq.insert(Value::String("medium".to_string()), 2.0).unwrap();
        
        assert_eq!(pq.size(), 3);
        assert!(!pq.is_empty());
        
        // Extract should return highest priority first
        assert_eq!(pq.extract().unwrap(), Value::String("high".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("medium".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("low".to_string()));
        
        assert!(pq.is_empty());
    }
    
    #[test]
    fn test_min_priority_queue_operations() {
        let mut pq = PriorityQueue::new(true);
        
        // Insert items with different priorities
        pq.insert(Value::String("low".to_string()), 1.0).unwrap();
        pq.insert(Value::String("high".to_string()), 3.0).unwrap();
        pq.insert(Value::String("medium".to_string()), 2.0).unwrap();
        
        // Extract should return lowest priority first for min-queue
        assert_eq!(pq.extract().unwrap(), Value::String("low".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("medium".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("high".to_string()));
    }
    
    #[test]
    fn test_priority_queue_peek() {
        let mut pq = PriorityQueue::new(false);
        pq.insert(Value::Integer(42), 2.5).unwrap();
        pq.insert(Value::Integer(24), 1.5).unwrap();
        
        // Peek should return highest priority without removing
        assert_eq!(pq.peek().unwrap(), Value::Integer(42));
        assert_eq!(pq.peek_priority().unwrap(), 2.5);
        assert_eq!(pq.size(), 2);
        
        // Extract should return the same item
        assert_eq!(pq.extract().unwrap(), Value::Integer(42));
        assert_eq!(pq.size(), 1);
    }
    
    #[test]
    fn test_priority_queue_contains() {
        let mut pq = PriorityQueue::new(false);
        let value = Value::String("test".to_string());
        pq.insert(value.clone(), 1.0).unwrap();
        
        assert!(pq.contains(&value));
        assert!(!pq.contains(&Value::String("other".to_string())));
    }
    
    #[test]
    fn test_priority_queue_merge() {
        let mut pq1 = PriorityQueue::new(false);
        pq1.insert(Value::Integer(1), 1.0).unwrap();
        pq1.insert(Value::Integer(3), 3.0).unwrap();
        
        let mut pq2 = PriorityQueue::new(false);
        pq2.insert(Value::Integer(2), 2.0).unwrap();
        pq2.insert(Value::Integer(4), 4.0).unwrap();
        
        pq1.merge(&pq2).unwrap();
        
        assert_eq!(pq1.size(), 4);
        // Should extract in priority order: 4, 3, 2, 1
        assert_eq!(pq1.extract().unwrap(), Value::Integer(4));
        assert_eq!(pq1.extract().unwrap(), Value::Integer(3));
        assert_eq!(pq1.extract().unwrap(), Value::Integer(2));
        assert_eq!(pq1.extract().unwrap(), Value::Integer(1));
    }
    
    #[test]
    fn test_stable_ordering() {
        let mut pq = PriorityQueue::new(false);
        
        // Insert items with same priority
        pq.insert(Value::String("first".to_string()), 1.0).unwrap();
        pq.insert(Value::String("second".to_string()), 1.0).unwrap();
        pq.insert(Value::String("third".to_string()), 1.0).unwrap();
        
        // Should extract in insertion order for same priority
        assert_eq!(pq.extract().unwrap(), Value::String("first".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("second".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("third".to_string()));
    }
    
    #[test]
    fn test_mixed_number_priorities() {
        let mut pq = PriorityQueue::new(false);
        
        // Test with integer and real priorities
        pq.insert(Value::String("int".to_string()), 5.0).unwrap();
        pq.insert(Value::String("real".to_string()), 5.5).unwrap();
        
        assert_eq!(pq.extract().unwrap(), Value::String("real".to_string()));
        assert_eq!(pq.extract().unwrap(), Value::String("int".to_string()));
    }
}