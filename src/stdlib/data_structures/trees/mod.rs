//! Tree data structures for Lyra standard library
//!
//! This module provides various tree data structures including binary search trees,
//! tries, and other hierarchical data structures. All implementations use the
//! Foreign object pattern for VM integration.

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod bst;
pub mod trie;

// Re-export all tree functions
pub use bst::*;
pub use trie::*;

/// Register all tree data structure functions
pub fn register_tree_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    // Register BST functions
    bst::register_bst_functions(functions);
    
    // Register Trie functions
    trie::register_trie_functions(functions);
}

/// Get documentation for all tree data structures
pub fn get_tree_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    
    // Add BST documentation
    docs.extend(bst::get_bst_documentation());
    
    // Add Trie documentation
    docs.extend(trie::get_trie_documentation());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tree_function_registration() {
        let mut functions = HashMap::new();
        register_tree_functions(&mut functions);
        
        // Verify BST functions are registered
        assert!(functions.contains_key("BST"));
        assert!(functions.contains_key("BSTInsert"));
        assert!(functions.contains_key("BSTSearch"));
        assert!(functions.contains_key("BSTDelete"));
        assert!(functions.contains_key("BSTMin"));
        assert!(functions.contains_key("BSTMax"));
        assert!(functions.contains_key("BSTInOrder"));
        assert!(functions.contains_key("BSTPreOrder"));
        assert!(functions.contains_key("BSTPostOrder"));
        assert!(functions.contains_key("BSTHeight"));
        assert!(functions.contains_key("BSTBalance"));
        
        // Should have 11 BST functions
        let bst_count = functions.keys().filter(|k| k.starts_with("BST")).count();
        assert_eq!(bst_count, 11);
    }
    
    #[test]
    fn test_tree_documentation_availability() {
        let docs = get_tree_documentation();
        
        // Verify BST documentation exists
        assert!(docs.contains_key("BST"));
        assert!(docs.contains_key("BSTInsert"));
        
        // Verify documentation is non-empty
        for (name, doc) in docs.iter() {
            assert!(!doc.is_empty(), "Documentation for {} is empty", name);
        }
    }
}