//! Binary Search Tree implementation for Lyra
//!
//! This module provides a complete BST implementation with insertion, deletion,
//! search, and traversal operations. Built using the Foreign object pattern.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::cmp::Ordering;
use std::fmt;

/// Node in the binary search tree
#[derive(Clone, Debug)]
struct BSTNode {
    value: Value,
    left: Option<Box<BSTNode>>,
    right: Option<Box<BSTNode>>,
}

impl BSTNode {
    fn new(value: Value) -> Self {
        Self {
            value,
            left: None,
            right: None,
        }
    }
    
    /// Insert a value into the subtree rooted at this node
    fn insert(&mut self, value: Value) -> Result<(), ForeignError> {
        match compare_values(&value, &self.value) {
            Ok(Ordering::Less) => {
                match &mut self.left {
                    Some(left_child) => left_child.insert(value),
                    None => {
                        self.left = Some(Box::new(BSTNode::new(value)));
                        Ok(())
                    }
                }
            }
            Ok(Ordering::Greater) => {
                match &mut self.right {
                    Some(right_child) => right_child.insert(value),
                    None => {
                        self.right = Some(Box::new(BSTNode::new(value)));
                        Ok(())
                    }
                }
            }
            Ok(Ordering::Equal) => {
                // Replace existing value (BST property maintained)
                self.value = value;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }
    
    /// Search for a value in the subtree rooted at this node
    fn search(&self, target: &Value) -> Result<bool, ForeignError> {
        match compare_values(target, &self.value) {
            Ok(Ordering::Equal) => Ok(true),
            Ok(Ordering::Less) => {
                match &self.left {
                    Some(left_child) => left_child.search(target),
                    None => Ok(false),
                }
            }
            Ok(Ordering::Greater) => {
                match &self.right {
                    Some(right_child) => right_child.search(target),
                    None => Ok(false),
                }
            }
            Err(e) => Err(e),
        }
    }
    
    /// Find the minimum value in the subtree
    fn find_min(&self) -> &Value {
        match &self.left {
            Some(left_child) => left_child.find_min(),
            None => &self.value,
        }
    }
    
    /// Find the maximum value in the subtree
    fn find_max(&self) -> &Value {
        match &self.right {
            Some(right_child) => right_child.find_max(),
            None => &self.value,
        }
    }
    
    /// Delete a value from the subtree, returning the new root
    fn delete(mut self: Box<Self>, target: &Value) -> Result<Option<Box<BSTNode>>, ForeignError> {
        match compare_values(target, &self.value) {
            Ok(Ordering::Less) => {
                if let Some(left_child) = self.left.take() {
                    self.left = left_child.delete(target)?;
                }
                Ok(Some(self))
            }
            Ok(Ordering::Greater) => {
                if let Some(right_child) = self.right.take() {
                    self.right = right_child.delete(target)?;
                }
                Ok(Some(self))
            }
            Ok(Ordering::Equal) => {
                // Node to delete found
                match (self.left.take(), self.right.take()) {
                    (None, None) => Ok(None), // Leaf node
                    (Some(left), None) => Ok(Some(left)), // Only left child
                    (None, Some(right)) => Ok(Some(right)), // Only right child
                    (Some(left), Some(right)) => {
                        // Two children: replace with inorder successor
                        let successor_value = right.find_min().clone();
                        self.value = successor_value.clone();
                        self.left = Some(left);
                        self.right = right.delete(&successor_value)?;
                        Ok(Some(self))
                    }
                }
            }
            Err(e) => Err(e),
        }
    }
    
    /// In-order traversal (left, root, right)
    fn inorder(&self, result: &mut Vec<Value>) {
        if let Some(left) = &self.left {
            left.inorder(result);
        }
        result.push(self.value.clone());
        if let Some(right) = &self.right {
            right.inorder(result);
        }
    }
    
    /// Pre-order traversal (root, left, right)
    fn preorder(&self, result: &mut Vec<Value>) {
        result.push(self.value.clone());
        if let Some(left) = &self.left {
            left.preorder(result);
        }
        if let Some(right) = &self.right {
            right.preorder(result);
        }
    }
    
    /// Post-order traversal (left, right, root)
    fn postorder(&self, result: &mut Vec<Value>) {
        if let Some(left) = &self.left {
            left.postorder(result);
        }
        if let Some(right) = &self.right {
            right.postorder(result);
        }
        result.push(self.value.clone());
    }
    
    /// Calculate height of the subtree
    fn height(&self) -> usize {
        let left_height = self.left.as_ref().map_or(0, |left| left.height());
        let right_height = self.right.as_ref().map_or(0, |right| right.height());
        1 + std::cmp::max(left_height, right_height)
    }
    
    /// Count nodes in the subtree
    fn count(&self) -> usize {
        let left_count = self.left.as_ref().map_or(0, |left| left.count());
        let right_count = self.right.as_ref().map_or(0, |right| right.count());
        1 + left_count + right_count
    }
}

/// Binary Search Tree with complete operations
#[derive(Clone)]
pub struct BST {
    root: Option<Box<BSTNode>>,
}

impl BST {
    /// Create a new empty BST
    pub fn new() -> Self {
        Self { root: None }
    }
    
    /// Insert a value into the BST
    pub fn insert(&mut self, value: Value) -> Result<(), ForeignError> {
        match &mut self.root {
            Some(root) => root.insert(value),
            None => {
                self.root = Some(Box::new(BSTNode::new(value)));
                Ok(())
            }
        }
    }
    
    /// Search for a value in the BST
    pub fn search(&self, target: &Value) -> Result<bool, ForeignError> {
        match &self.root {
            Some(root) => root.search(target),
            None => Ok(false),
        }
    }
    
    /// Delete a value from the BST
    pub fn delete(&mut self, target: &Value) -> Result<(), ForeignError> {
        if let Some(root) = self.root.take() {
            self.root = root.delete(target)?;
        }
        Ok(())
    }
    
    /// Find the minimum value in the BST
    pub fn min(&self) -> Result<Value, ForeignError> {
        match &self.root {
            Some(root) => Ok(root.find_min().clone()),
            None => Err(ForeignError::RuntimeError {
                message: "Cannot find minimum in empty BST".to_string(),
            }),
        }
    }
    
    /// Find the maximum value in the BST
    pub fn max(&self) -> Result<Value, ForeignError> {
        match &self.root {
            Some(root) => Ok(root.find_max().clone()),
            None => Err(ForeignError::RuntimeError {
                message: "Cannot find maximum in empty BST".to_string(),
            }),
        }
    }
    
    /// Get in-order traversal (sorted order)
    pub fn inorder(&self) -> Vec<Value> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            root.inorder(&mut result);
        }
        result
    }
    
    /// Get pre-order traversal
    pub fn preorder(&self) -> Vec<Value> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            root.preorder(&mut result);
        }
        result
    }
    
    /// Get post-order traversal
    pub fn postorder(&self) -> Vec<Value> {
        let mut result = Vec::new();
        if let Some(root) = &self.root {
            root.postorder(&mut result);
        }
        result
    }
    
    /// Get the height of the BST
    pub fn height(&self) -> usize {
        match &self.root {
            Some(root) => root.height(),
            None => 0,
        }
    }
    
    /// Get the number of nodes in the BST
    pub fn size(&self) -> usize {
        match &self.root {
            Some(root) => root.count(),
            None => 0,
        }
    }
    
    /// Check if the BST is empty
    pub fn is_empty(&self) -> bool {
        self.root.is_none()
    }
    
    /// Clear the BST
    pub fn clear(&mut self) {
        self.root = None;
    }
    
    /// Check if the tree is balanced (simple check: height difference <= 1)
    pub fn is_balanced(&self) -> bool {
        fn check_balance(node: &Option<Box<BSTNode>>) -> (bool, usize) {
            match node {
                None => (true, 0),
                Some(n) => {
                    let (left_balanced, left_height) = check_balance(&n.left);
                    let (right_balanced, right_height) = check_balance(&n.right);
                    
                    let balanced = left_balanced && right_balanced 
                        && (left_height as i32 - right_height as i32).abs() <= 1;
                    let height = 1 + std::cmp::max(left_height, right_height);
                    
                    (balanced, height)
                }
            }
        }
        
        check_balance(&self.root).0
    }
}

/// Compare two Values for BST ordering
fn compare_values(a: &Value, b: &Value) -> Result<Ordering, ForeignError> {
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Ok(x.cmp(y)),
        (Value::Real(x), Value::Real(y)) => Ok(x.partial_cmp(y).unwrap_or(Ordering::Equal)),
        (Value::Integer(x), Value::Real(y)) => Ok((*x as f64).partial_cmp(y).unwrap_or(Ordering::Equal)),
        (Value::Real(x), Value::Integer(y)) => Ok(x.partial_cmp(&(*y as f64)).unwrap_or(Ordering::Equal)),
        (Value::String(x), Value::String(y)) => Ok(x.cmp(y)),
        (Value::Symbol(x), Value::Symbol(y)) => Ok(x.cmp(y)),
        _ => Err(ForeignError::RuntimeError {
            message: format!("Cannot compare values of different types: {:?} and {:?}", a, b),
        }),
    }
}

impl Foreign for BST {
    fn type_name(&self) -> &'static str {
        "BST"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut bst = self.clone();
        match method {
            "insert" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                bst.insert(args[0].clone())?;
                Ok(Value::LyObj(LyObj::new(Box::new(bst))))
            }
            "search" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let found = bst.search(&args[0])?;
                Ok(Value::Boolean(found))
            }
            "delete" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                bst.delete(&args[0])?;
                Ok(Value::LyObj(LyObj::new(Box::new(bst))))
            }
            "min" => {
                let min_val = bst.min()?;
                Ok(min_val)
            }
            "max" => {
                let max_val = bst.max()?;
                Ok(max_val)
            }
            "inorder" => {
                let traversal = bst.inorder();
                Ok(Value::List(traversal))
            }
            "preorder" => {
                let traversal = bst.preorder();
                Ok(Value::List(traversal))
            }
            "postorder" => {
                let traversal = bst.postorder();
                Ok(Value::List(traversal))
            }
            "height" => {
                Ok(Value::Integer(bst.height() as i64))
            }
            "size" => {
                Ok(Value::Integer(bst.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(bst.is_empty()))
            }
            "clear" => {
                bst.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(bst))))
            }
            "isBalanced" => {
                Ok(Value::Boolean(bst.is_balanced()))
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

impl fmt::Display for BST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BST[size: {}]", self.size())
    }
}

impl fmt::Debug for BST {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BST")
            .field("size", &self.size())
            .field("height", &self.height())
            .field("balanced", &self.is_balanced())
            .finish()
    }
}

/// Create a new empty BST
pub fn bst_new(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime("BST[] takes no arguments".to_string()));
    }
    
    let bst = BST::new();
    Ok(Value::LyObj(LyObj::new(Box::new(bst))))
}

/// Insert a value into a BST
pub fn bst_insert(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("insert", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a BST".to_string()))
    }
}

/// Search for a value in a BST
pub fn bst_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("search", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a BST".to_string()))
    }
}

/// Delete a value from a BST
pub fn bst_delete(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("delete", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a BST".to_string()))
    }
}

/// Find the minimum value in a BST
pub fn bst_min(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("min", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Find the maximum value in a BST
pub fn bst_max(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("max", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Get in-order traversal of a BST (sorted order)
pub fn bst_inorder(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("inorder", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Get pre-order traversal of a BST
pub fn bst_preorder(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("preorder", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Get post-order traversal of a BST
pub fn bst_postorder(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("postorder", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Get the height of a BST
pub fn bst_height(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("height", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Check if a BST is balanced
pub fn bst_is_balanced(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(bst_obj) = &args[0] {
        bst_obj.call_method("isBalanced", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a BST".to_string()))
    }
}

/// Register BST functions with the standard library
pub fn register_bst_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("BST".to_string(), bst_new);
    functions.insert("BSTInsert".to_string(), bst_insert);
    functions.insert("BSTSearch".to_string(), bst_search);
    functions.insert("BSTDelete".to_string(), bst_delete);
    functions.insert("BSTMin".to_string(), bst_min);
    functions.insert("BSTMax".to_string(), bst_max);
    functions.insert("BSTInOrder".to_string(), bst_inorder);
    functions.insert("BSTPreOrder".to_string(), bst_preorder);
    functions.insert("BSTPostOrder".to_string(), bst_postorder);
    functions.insert("BSTHeight".to_string(), bst_height);
    functions.insert("BSTBalance".to_string(), bst_is_balanced);
}

/// Get documentation for BST functions
pub fn get_bst_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("BST".to_string(), "BST[] - Create empty binary search tree. O(1) time.".to_string());
    docs.insert("BSTInsert".to_string(), "BSTInsert[bst, value] - Insert value into BST. O(log n) average, O(n) worst.".to_string());
    docs.insert("BSTSearch".to_string(), "BSTSearch[bst, value] - Search for value in BST. O(log n) average, O(n) worst.".to_string());
    docs.insert("BSTDelete".to_string(), "BSTDelete[bst, value] - Delete value from BST. O(log n) average, O(n) worst.".to_string());
    docs.insert("BSTMin".to_string(), "BSTMin[bst] - Find minimum value in BST. O(log n) average, O(n) worst.".to_string());
    docs.insert("BSTMax".to_string(), "BSTMax[bst] - Find maximum value in BST. O(log n) average, O(n) worst.".to_string());
    docs.insert("BSTInOrder".to_string(), "BSTInOrder[bst] - In-order traversal (sorted order). O(n) time.".to_string());
    docs.insert("BSTPreOrder".to_string(), "BSTPreOrder[bst] - Pre-order traversal. O(n) time.".to_string());
    docs.insert("BSTPostOrder".to_string(), "BSTPostOrder[bst] - Post-order traversal. O(n) time.".to_string());
    docs.insert("BSTHeight".to_string(), "BSTHeight[bst] - Get height of BST. O(n) time.".to_string());
    docs.insert("BSTBalance".to_string(), "BSTBalance[bst] - Check if BST is balanced. O(n) time.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bst_creation() {
        let bst = BST::new();
        assert_eq!(bst.size(), 0);
        assert!(bst.is_empty());
        assert_eq!(bst.height(), 0);
    }
    
    #[test]
    fn test_bst_insertion_and_search() {
        let mut bst = BST::new();
        
        bst.insert(Value::Integer(5)).unwrap();
        bst.insert(Value::Integer(3)).unwrap();
        bst.insert(Value::Integer(7)).unwrap();
        bst.insert(Value::Integer(1)).unwrap();
        bst.insert(Value::Integer(9)).unwrap();
        
        assert_eq!(bst.size(), 5);
        assert!(!bst.is_empty());
        assert_eq!(bst.height(), 3);
        
        assert!(bst.search(&Value::Integer(5)).unwrap());
        assert!(bst.search(&Value::Integer(3)).unwrap());
        assert!(bst.search(&Value::Integer(7)).unwrap());
        assert!(!bst.search(&Value::Integer(4)).unwrap());
    }
    
    #[test]
    fn test_bst_min_max() {
        let mut bst = BST::new();
        
        bst.insert(Value::Integer(5)).unwrap();
        bst.insert(Value::Integer(3)).unwrap();
        bst.insert(Value::Integer(7)).unwrap();
        bst.insert(Value::Integer(1)).unwrap();
        bst.insert(Value::Integer(9)).unwrap();
        
        assert_eq!(bst.min().unwrap(), Value::Integer(1));
        assert_eq!(bst.max().unwrap(), Value::Integer(9));
    }
    
    #[test]
    fn test_bst_traversals() {
        let mut bst = BST::new();
        
        bst.insert(Value::Integer(5)).unwrap();
        bst.insert(Value::Integer(3)).unwrap();
        bst.insert(Value::Integer(7)).unwrap();
        bst.insert(Value::Integer(1)).unwrap();
        bst.insert(Value::Integer(9)).unwrap();
        
        // In-order should be sorted
        let inorder = bst.inorder();
        assert_eq!(inorder, vec![
            Value::Integer(1),
            Value::Integer(3),
            Value::Integer(5),
            Value::Integer(7),
            Value::Integer(9),
        ]);
        
        // Pre-order: root, left, right
        let preorder = bst.preorder();
        assert_eq!(preorder, vec![
            Value::Integer(5),
            Value::Integer(3),
            Value::Integer(1),
            Value::Integer(7),
            Value::Integer(9),
        ]);
        
        // Post-order: left, right, root
        let postorder = bst.postorder();
        assert_eq!(postorder, vec![
            Value::Integer(1),
            Value::Integer(3),
            Value::Integer(9),
            Value::Integer(7),
            Value::Integer(5),
        ]);
    }
    
    #[test]
    fn test_bst_deletion() {
        let mut bst = BST::new();
        
        // Build tree: 5(3(1,4),7(6,9))
        bst.insert(Value::Integer(5)).unwrap();
        bst.insert(Value::Integer(3)).unwrap();
        bst.insert(Value::Integer(7)).unwrap();
        bst.insert(Value::Integer(1)).unwrap();
        bst.insert(Value::Integer(4)).unwrap();
        bst.insert(Value::Integer(6)).unwrap();
        bst.insert(Value::Integer(9)).unwrap();
        
        assert_eq!(bst.size(), 7);
        
        // Delete leaf node
        bst.delete(&Value::Integer(1)).unwrap();
        assert_eq!(bst.size(), 6);
        assert!(!bst.search(&Value::Integer(1)).unwrap());
        
        // Delete node with one child
        bst.delete(&Value::Integer(3)).unwrap();
        assert_eq!(bst.size(), 5);
        assert!(!bst.search(&Value::Integer(3)).unwrap());
        assert!(bst.search(&Value::Integer(4)).unwrap());
        
        // Delete node with two children
        bst.delete(&Value::Integer(5)).unwrap();
        assert_eq!(bst.size(), 4);
        assert!(!bst.search(&Value::Integer(5)).unwrap());
        
        // Verify BST property is maintained
        let inorder = bst.inorder();
        let mut sorted = inorder.clone();
        sorted.sort_by(|a, b| compare_values(a, b).unwrap());
        assert_eq!(inorder, sorted);
    }
    
    #[test]
    fn test_bst_with_strings() {
        let mut bst = BST::new();
        
        bst.insert(Value::String("dog".to_string())).unwrap();
        bst.insert(Value::String("cat".to_string())).unwrap();
        bst.insert(Value::String("fish".to_string())).unwrap();
        bst.insert(Value::String("bird".to_string())).unwrap();
        
        let inorder = bst.inorder();
        assert_eq!(inorder, vec![
            Value::String("bird".to_string()),
            Value::String("cat".to_string()),
            Value::String("dog".to_string()),
            Value::String("fish".to_string()),
        ]);
    }
    
    #[test]
    fn test_bst_balance_check() {
        let mut balanced_bst = BST::new();
        balanced_bst.insert(Value::Integer(2)).unwrap();
        balanced_bst.insert(Value::Integer(1)).unwrap();
        balanced_bst.insert(Value::Integer(3)).unwrap();
        assert!(balanced_bst.is_balanced());
        
        let mut unbalanced_bst = BST::new();
        unbalanced_bst.insert(Value::Integer(1)).unwrap();
        unbalanced_bst.insert(Value::Integer(2)).unwrap();
        unbalanced_bst.insert(Value::Integer(3)).unwrap();
        unbalanced_bst.insert(Value::Integer(4)).unwrap();
        assert!(!unbalanced_bst.is_balanced());
    }
    
    #[test]
    fn test_compare_values() {
        assert_eq!(compare_values(&Value::Integer(5), &Value::Integer(3)).unwrap(), Ordering::Greater);
        assert_eq!(compare_values(&Value::Integer(3), &Value::Integer(5)).unwrap(), Ordering::Less);
        assert_eq!(compare_values(&Value::Integer(5), &Value::Integer(5)).unwrap(), Ordering::Equal);
        
        assert_eq!(compare_values(&Value::Real(5.5), &Value::Integer(5)).unwrap(), Ordering::Greater);
        assert_eq!(compare_values(&Value::Integer(5), &Value::Real(5.5)).unwrap(), Ordering::Less);
        
        assert_eq!(compare_values(&Value::String("apple".to_string()), &Value::String("banana".to_string())).unwrap(), Ordering::Less);
    }
}