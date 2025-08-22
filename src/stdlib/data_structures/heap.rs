//! Binary Heap and Priority Queue implementations for Lyra
//!
//! This module provides efficient heap data structures with O(log n) 
//! insert/extract operations and O(n) heapify from arrays.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::fmt;

/// Binary heap implementation supporting both min-heap and max-heap operations
#[derive(Clone)]
pub struct BinaryHeap {
    data: Vec<Value>,
    is_max_heap: bool,
}

impl BinaryHeap {
    /// Create a new empty binary heap
    pub fn new(is_max_heap: bool) -> Self {
        Self {
            data: Vec::new(),
            is_max_heap,
        }
    }
    
    /// Create a binary heap from a list of values (heapify)
    pub fn from_list(values: Vec<Value>, is_max_heap: bool) -> Self {
        let mut heap = Self {
            data: values,
            is_max_heap,
        };
        
        // Heapify: start from the last parent and sift down
        if heap.data.len() > 1 {
            for i in (0..=heap.parent_index(heap.data.len() - 1)).rev() {
                heap.sift_down(i);
            }
        }
        
        heap
    }
    
    /// Compare two values according to heap type (min or max)
    fn compare(&self, a: &Value, b: &Value) -> Ordering {
        let cmp = self.value_compare(a, b);
        if self.is_max_heap {
            cmp.reverse()
        } else {
            cmp
        }
    }
    
    /// Compare two Value objects using consistent ordering
    fn value_compare(&self, a: &Value, b: &Value) -> Ordering {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
            (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
            (Value::String(a), Value::String(b)) => a.cmp(b),
            (Value::Symbol(a), Value::Symbol(b)) => a.cmp(b),
            (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
            _ => Ordering::Equal, // For unsupported types, consider them equal
        }
    }
    
    /// Get parent index
    fn parent_index(&self, i: usize) -> usize {
        (i - 1) / 2
    }
    
    /// Get left child index
    fn left_child_index(&self, i: usize) -> usize {
        2 * i + 1
    }
    
    /// Get right child index
    fn right_child_index(&self, i: usize) -> usize {
        2 * i + 2
    }
    
    /// Insert a value into the heap
    pub fn insert(&mut self, value: Value) -> Result<(), ForeignError> {
        self.data.push(value);
        if self.data.len() > 1 {
            self.sift_up(self.data.len() - 1);
        }
        Ok(())
    }
    
    /// Extract the minimum (or maximum for max-heap) element
    pub fn extract_min(&mut self) -> Result<Value, ForeignError> {
        if self.data.is_empty() {
            return Err(ForeignError::RuntimeError {
                message: "Cannot extract from empty heap".to_string(),
            });
        }
        
        let min = self.data[0].clone();
        
        if self.data.len() == 1 {
            self.data.pop();
        } else {
            // Move last element to root and sift down
            let last = self.data.pop().unwrap();
            self.data[0] = last;
            self.sift_down(0);
        }
        
        Ok(min)
    }
    
    /// Peek at the minimum (or maximum) element without removing it
    pub fn peek(&self) -> Result<Value, ForeignError> {
        if self.data.is_empty() {
            Err(ForeignError::RuntimeError {
                message: "Cannot peek at empty heap".to_string(),
            })
        } else {
            Ok(self.data[0].clone())
        }
    }
    
    /// Get the size of the heap
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Check if the heap is empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Convert heap to a sorted list (heap sort)
    pub fn to_sorted_list(&self) -> Vec<Value> {
        let mut heap_copy = self.clone();
        let mut result = Vec::with_capacity(heap_copy.size());
        
        while !heap_copy.is_empty() {
            result.push(heap_copy.extract_min().unwrap());
        }
        
        result
    }
    
    /// Sift up operation for maintaining heap property
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = self.parent_index(index);
            if self.compare(&self.data[index], &self.data[parent]) == Ordering::Greater {
                break;
            }
            self.data.swap(index, parent);
            index = parent;
        }
    }
    
    /// Sift down operation for maintaining heap property
    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = self.left_child_index(index);
            let right = self.right_child_index(index);
            let mut smallest = index;
            
            if left < self.data.len() && 
               self.compare(&self.data[left], &self.data[smallest]) == Ordering::Less {
                smallest = left;
            }
            
            if right < self.data.len() && 
               self.compare(&self.data[right], &self.data[smallest]) == Ordering::Less {
                smallest = right;
            }
            
            if smallest == index {
                break;
            }
            
            self.data.swap(index, smallest);
            index = smallest;
        }
    }
    
    /// Merge another heap into this one
    pub fn merge(&mut self, other: &BinaryHeap) -> Result<(), ForeignError> {
        if self.is_max_heap != other.is_max_heap {
            return Err(ForeignError::RuntimeError {
                message: "Cannot merge heaps of different types (min vs max)".to_string(),
            });
        }
        
        // Simple approach: add all elements and re-heapify
        self.data.extend(other.data.iter().cloned());
        
        // Re-heapify the combined data
        if self.data.len() > 1 {
            for i in (0..=self.parent_index(self.data.len() - 1)).rev() {
                self.sift_down(i);
            }
        }
        
        Ok(())
    }
}

impl Foreign for BinaryHeap {
    fn type_name(&self) -> &'static str {
        if self.is_max_heap {
            "BinaryMaxHeap"
        } else {
            "BinaryHeap"
        }
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut heap = self.clone();
        match method {
            "insert" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                heap.insert(args[0].clone())?;
                Ok(Value::LyObj(LyObj::new(Box::new(heap))))
            }
            "extractMin" | "extractMax" => {
                let min = heap.extract_min()?;
                Ok(min)
            }
            "peek" => {
                heap.peek()
            }
            "size" => {
                Ok(Value::Integer(heap.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(heap.is_empty()))
            }
            "toList" => {
                let sorted = heap.to_sorted_list();
                Ok(Value::List(sorted))
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
                    if let Some(other_heap) = obj.downcast_ref::<BinaryHeap>() {
                        heap.merge(other_heap)?;
                        return Ok(Value::LyObj(LyObj::new(Box::new(heap))));
                    }
                }
                
                Err(ForeignError::InvalidArgumentType {
                    method: method.to_string(),
                    expected: "BinaryHeap".to_string(),
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

impl fmt::Display for BinaryHeap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let heap_type = if self.is_max_heap { "MaxHeap" } else { "MinHeap" };
        write!(f, "{}[size: {}]", heap_type, self.data.len())
    }
}

impl fmt::Debug for BinaryHeap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BinaryHeap")
            .field("data", &self.data)
            .field("is_max_heap", &self.is_max_heap)
            .finish()
    }
}

/// Create a new empty min-heap
pub fn binary_heap(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            // Create empty min-heap
            let heap = BinaryHeap::new(false);
            Ok(Value::LyObj(LyObj::new(Box::new(heap))))
        }
        1 => {
            // Create heap from list
            if let Value::List(list) = &args[0] {
                let heap = BinaryHeap::from_list(list.clone(), false);
                Ok(Value::LyObj(LyObj::new(Box::new(heap))))
            } else {
                Err(VmError::Runtime("BinaryHeap expects a list argument".to_string()))
            }
        }
        _ => Err(VmError::Runtime("BinaryHeap takes 0 or 1 arguments".to_string()))
    }
}

/// Create a new empty max-heap  
pub fn binary_max_heap(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            // Create empty max-heap
            let heap = BinaryHeap::new(true);
            Ok(Value::LyObj(LyObj::new(Box::new(heap))))
        }
        1 => {
            // Create max-heap from list
            if let Value::List(list) = &args[0] {
                let heap = BinaryHeap::from_list(list.clone(), true);
                Ok(Value::LyObj(LyObj::new(Box::new(heap))))
            } else {
                Err(VmError::Runtime("BinaryMaxHeap expects a list argument".to_string()))
            }
        }
        _ => Err(VmError::Runtime("BinaryMaxHeap takes 0 or 1 arguments".to_string()))
    }
}

/// Insert an element into a heap
pub fn heap_insert(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("insert", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a heap".to_string()))
    }
}

/// Extract the minimum element from a heap
pub fn heap_extract_min(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("extractMin", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a heap".to_string()))
    }
}

/// Extract the maximum element from a max-heap
pub fn heap_extract_max(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("extractMax", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a heap".to_string()))
    }
}

/// Peek at the top element of a heap without removing it
pub fn heap_peek(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("peek", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a heap".to_string()))
    }
}

/// Get the size of a heap
pub fn heap_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("size", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a heap".to_string()))
    }
}

/// Check if a heap is empty
pub fn heap_is_empty(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("isEmpty", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a heap".to_string()))
    }
}

/// Merge two heaps
pub fn heap_merge(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(heap_obj) = &args[0] {
        heap_obj.call_method("merge", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a heap".to_string()))
    }
}

/// Register heap functions with the standard library
pub fn register_heap_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("BinaryHeap".to_string(), binary_heap);
    functions.insert("BinaryMaxHeap".to_string(), binary_max_heap);
    functions.insert("HeapInsert".to_string(), heap_insert);
    functions.insert("HeapExtractMin".to_string(), heap_extract_min);
    functions.insert("HeapExtractMax".to_string(), heap_extract_max);
    functions.insert("HeapPeek".to_string(), heap_peek);
    functions.insert("HeapSize".to_string(), heap_size);
    functions.insert("HeapIsEmpty".to_string(), heap_is_empty);
    functions.insert("HeapMerge".to_string(), heap_merge);
}

/// Get documentation for heap functions
pub fn get_heap_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("BinaryHeap".to_string(), "BinaryHeap[] - Create empty min-heap. BinaryHeap[list] - Create min-heap from list.".to_string());
    docs.insert("BinaryMaxHeap".to_string(), "BinaryMaxHeap[] - Create empty max-heap. BinaryMaxHeap[list] - Create max-heap from list.".to_string());
    docs.insert("HeapInsert".to_string(), "HeapInsert[heap, element] - Insert element into heap. Returns new heap.".to_string());
    docs.insert("HeapExtractMin".to_string(), "HeapExtractMin[heap] - Remove and return minimum element from min-heap.".to_string());
    docs.insert("HeapExtractMax".to_string(), "HeapExtractMax[heap] - Remove and return maximum element from max-heap.".to_string());
    docs.insert("HeapPeek".to_string(), "HeapPeek[heap] - View top element without removing it.".to_string());
    docs.insert("HeapSize".to_string(), "HeapSize[heap] - Get number of elements in heap.".to_string());
    docs.insert("HeapIsEmpty".to_string(), "HeapIsEmpty[heap] - Check if heap is empty.".to_string());
    docs.insert("HeapMerge".to_string(), "HeapMerge[heap1, heap2] - Merge two heaps of the same type.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_empty_heap_creation() {
        let heap = BinaryHeap::new(false);
        assert_eq!(heap.size(), 0);
        assert!(heap.is_empty());
    }
    
    #[test]
    fn test_heap_from_list() {
        let values = vec![
            Value::Integer(5),
            Value::Integer(3),
            Value::Integer(8),
            Value::Integer(1),
        ];
        let heap = BinaryHeap::from_list(values, false);
        assert_eq!(heap.size(), 4);
        assert!(!heap.is_empty());
    }
    
    #[test]
    fn test_min_heap_operations() {
        let mut heap = BinaryHeap::new(false);
        
        // Insert elements
        heap.insert(Value::Integer(5)).unwrap();
        heap.insert(Value::Integer(3)).unwrap();
        heap.insert(Value::Integer(8)).unwrap();
        heap.insert(Value::Integer(1)).unwrap();
        
        assert_eq!(heap.size(), 4);
        
        // Peek should return minimum
        assert_eq!(heap.peek().unwrap(), Value::Integer(1));
        
        // Extract elements in order
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(1));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(3));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(5));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(8));
        
        assert!(heap.is_empty());
    }
    
    #[test]
    fn test_max_heap_operations() {
        let mut heap = BinaryHeap::new(true);
        
        // Insert elements
        heap.insert(Value::Integer(5)).unwrap();
        heap.insert(Value::Integer(3)).unwrap();
        heap.insert(Value::Integer(8)).unwrap();
        heap.insert(Value::Integer(1)).unwrap();
        
        // Peek should return maximum
        assert_eq!(heap.peek().unwrap(), Value::Integer(8));
        
        // Extract elements in reverse order
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(8));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(5));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(3));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(1));
    }
    
    #[test]
    fn test_heap_merge() {
        let mut heap1 = BinaryHeap::new(false);
        heap1.insert(Value::Integer(1)).unwrap();
        heap1.insert(Value::Integer(3)).unwrap();
        
        let mut heap2 = BinaryHeap::new(false);
        heap2.insert(Value::Integer(2)).unwrap();
        heap2.insert(Value::Integer(4)).unwrap();
        
        heap1.merge(&heap2).unwrap();
        
        assert_eq!(heap1.size(), 4);
        assert_eq!(heap1.extract_min().unwrap(), Value::Integer(1));
        assert_eq!(heap1.extract_min().unwrap(), Value::Integer(2));
    }
    
    #[test]
    fn test_mixed_types() {
        let mut heap = BinaryHeap::new(false);
        heap.insert(Value::Integer(5)).unwrap();
        heap.insert(Value::Real(3.14)).unwrap();
        heap.insert(Value::Integer(2)).unwrap();
        
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(2));
        assert_eq!(heap.extract_min().unwrap(), Value::Real(3.14));
        assert_eq!(heap.extract_min().unwrap(), Value::Integer(5));
    }
}