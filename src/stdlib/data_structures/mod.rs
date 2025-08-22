//! Data Structures for Lyra Standard Library
//!
//! This module provides fundamental computer science data structures
//! including heaps, priority queues, trees, and other essential structures.
//! All data structures follow the Foreign object pattern and provide
//! Wolfram Language-style APIs.
//!
//! ## Available Data Structures
//!
//! ### Heaps and Priority Queues
//! - **BinaryHeap**: Efficient min/max heap implementation
//! - **PriorityQueue**: Priority-based queue with custom priorities
//!
//! ### Linear Structures  
//! - **Queue**: FIFO (First In, First Out) queue
//! - **Stack**: LIFO (Last In, First Out) stack
//! - **Deque**: Double-ended queue
//!
//! ### Tree Structures
//! - **BinarySearchTree**: Basic BST implementation
//! - **AVLTree**: Self-balancing BST (planned)
//! - **Trie**: Prefix tree for string operations (planned)
//!
//! ## Performance Characteristics
//!
//! All data structures are optimized for their intended use cases:
//! - Heap operations: O(log n) insert/extract, O(n) heapify
//! - Queue/Stack operations: O(1) push/pop
//! - BST operations: O(log n) average, O(n) worst case

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod heap;
pub mod queue;
pub mod stack;
pub mod priority_queue;
pub mod trees;
pub mod disjoint_set;
pub mod cache;
pub mod probabilistic;

// Re-export all data structure functions
pub use heap::*;
pub use queue::*;
pub use stack::*;
pub use priority_queue::*;
pub use trees::*;
pub use disjoint_set::*;
pub use cache::*;
pub use probabilistic::*;

/// Register all data structure functions with the standard library
pub fn register_data_structure_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    // Register heap functions
    heap::register_heap_functions(functions);
    
    // Register priority queue functions
    priority_queue::register_priority_queue_functions(functions);
    
    // Register queue functions
    queue::register_queue_functions(functions);
    
    // Register stack functions  
    stack::register_stack_functions(functions);
    
    // Register tree functions
    trees::register_tree_functions(functions);
    
    // Register disjoint set functions
    disjoint_set::register_disjoint_set_functions(functions);
    
    // Register cache functions
    cache::register_cache_functions(functions);
    
    // Register probabilistic functions
    probabilistic::register_probabilistic_functions(functions);
}

/// Get documentation for all available data structures
pub fn get_data_structure_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    
    // Add heap documentation
    docs.extend(heap::get_heap_documentation());
    
    // Add priority queue documentation
    docs.extend(priority_queue::get_priority_queue_documentation());
    
    // Add queue documentation
    docs.extend(queue::get_queue_documentation());
    
    // Add stack documentation
    docs.extend(stack::get_stack_documentation());
    
    // Add tree documentation
    docs.extend(trees::get_tree_documentation());
    
    // Add disjoint set documentation
    docs.extend(disjoint_set::get_disjoint_set_documentation());
    
    // Add cache documentation
    docs.extend(cache::get_lru_cache_documentation());
    
    // Add probabilistic documentation
    docs.extend(probabilistic::get_probabilistic_documentation());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_data_structure_registration() {
        let mut functions = HashMap::new();
        register_data_structure_functions(&mut functions);
        
        // Verify that core data structures are registered
        assert!(functions.contains_key("BinaryHeap"));
        assert!(functions.contains_key("HeapInsert"));
        assert!(functions.contains_key("HeapExtractMin"));
        
        // Verify we have a reasonable number of data structure functions
        assert!(functions.len() >= 10, "Expected at least 10 data structure functions");
    }
    
    #[test] 
    fn test_documentation_availability() {
        let docs = get_data_structure_documentation();
        
        // Verify that documentation exists for core functions
        assert!(docs.contains_key("BinaryHeap"));
        
        // Verify documentation is non-empty
        for (name, doc) in docs.iter() {
            assert!(!doc.is_empty(), "Documentation for {} is empty", name);
        }
    }
}