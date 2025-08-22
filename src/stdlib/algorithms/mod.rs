//! Core computer science algorithms for Lyra standard library
//!
//! This module provides fundamental algorithms that are expected in any
//! comprehensive programming language standard library. All algorithms
//! follow the Wolfram Language naming conventions and are implemented
//! using the Foreign object pattern for complex data structures.
//!
//! ## Algorithm Categories
//!
//! - **Sorting**: QuickSort, MergeSort, HeapSort, TimSort, RadixSort, etc.
//! - **Searching**: Binary search, interpolation search, exponential search
//! - **String Algorithms**: Pattern matching, edit distance, LCS
//! - **Compression**: Huffman coding, LZ77, run-length encoding
//!
//! All algorithms are optimized for performance while maintaining the
//! symbolic computation paradigm of Lyra.

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod sorting;
pub mod searching;
pub mod strings;
pub mod compression;

// Re-export all algorithm functions
pub use sorting::*;
pub use searching::*;
pub use strings::*;
pub use compression::*;

/// Register all algorithm functions with the standard library
pub fn register_algorithm_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    // Register sorting algorithms
    sorting::register_sorting_functions(functions);
    
    // Register searching algorithms  
    searching::register_searching_functions(functions);
    
    // Register string algorithms
    strings::register_string_algorithms(functions);
    
    // Register compression algorithms
    compression::register_compression_functions(functions);
}

/// Get documentation for all available algorithms
pub fn get_algorithm_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    
    // Add sorting documentation
    docs.extend(sorting::get_sorting_documentation());
    
    // Add searching documentation
    docs.extend(searching::get_searching_documentation());
    
    // Add string algorithm documentation
    docs.extend(strings::get_string_documentation());
    
    // Add compression documentation
    docs.extend(compression::get_compression_documentation());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_registration() {
        let mut functions = HashMap::new();
        register_algorithm_functions(&mut functions);
        
        // Verify that core algorithms are registered
        assert!(functions.contains_key("Sort"));
        assert!(functions.contains_key("QuickSort"));
        assert!(functions.contains_key("MergeSort"));
        assert!(functions.contains_key("BinarySearch"));
        
        // Verify we have a reasonable number of algorithms
        assert!(functions.len() >= 15, "Expected at least 15 core algorithms");
    }
    
    #[test]
    fn test_documentation_availability() {
        let docs = get_algorithm_documentation();
        
        // Verify that documentation exists for core functions
        assert!(docs.contains_key("Sort"));
        assert!(docs.contains_key("BinarySearch"));
        
        // Verify documentation is non-empty
        for (name, doc) in docs.iter() {
            assert!(!doc.is_empty(), "Documentation for {} is empty", name);
        }
    }
}