//! Trie (Prefix Tree) implementation for Lyra
//!
//! This module provides a complete Trie implementation for efficient prefix-based
//! string operations. Built using the Foreign object pattern for VM integration.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::fmt;

/// Node in the Trie
#[derive(Clone, Debug)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end_of_word: bool,
    value: Option<Value>, // Associated value for this prefix/word
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end_of_word: false,
            value: None,
        }
    }
    
    /// Insert a word with associated value
    fn insert(&mut self, word: &str, value: Option<Value>) {
        let mut current = self;
        
        for ch in word.chars() {
            current = current.children.entry(ch).or_insert_with(TrieNode::new);
        }
        
        current.is_end_of_word = true;
        current.value = value;
    }
    
    /// Search for a word
    fn search(&self, word: &str) -> bool {
        if let Some(node) = self.find_node(word) {
            node.is_end_of_word
        } else {
            false
        }
    }
    
    /// Check if any word starts with the given prefix
    fn starts_with(&self, prefix: &str) -> bool {
        self.find_node(prefix).is_some()
    }
    
    /// Get value associated with a word
    fn get_value(&self, word: &str) -> Option<&Value> {
        if let Some(node) = self.find_node(word) {
            if node.is_end_of_word {
                node.value.as_ref()
            } else {
                None
            }
        } else {
            None
        }
    }
    
    /// Find node corresponding to a prefix
    fn find_node(&self, prefix: &str) -> Option<&TrieNode> {
        let mut current = self;
        
        for ch in prefix.chars() {
            match current.children.get(&ch) {
                Some(node) => current = node,
                None => return None,
            }
        }
        
        Some(current)
    }
    
    /// Delete a word from the trie
    fn delete(&mut self, word: &str) -> bool {
        self.delete_recursive(word, 0)
    }
    
    fn delete_recursive(&mut self, word: &str, index: usize) -> bool {
        if index == word.len() {
            if !self.is_end_of_word {
                return false; // Word doesn't exist
            }
            
            self.is_end_of_word = false;
            self.value = None;
            
            // Return true if this node has no children (can be deleted)
            return self.children.is_empty();
        }
        
        let chars: Vec<char> = word.chars().collect();
        let ch = chars[index];
        
        if let Some(node) = self.children.get_mut(&ch) {
            let should_delete_child = node.delete_recursive(word, index + 1);
            
            if should_delete_child {
                self.children.remove(&ch);
            }
            
            // Return true if current node should be deleted
            !self.is_end_of_word && self.children.is_empty()
        } else {
            false // Character not found
        }
    }
    
    /// Get all words with a given prefix
    fn get_words_with_prefix(&self, prefix: &str) -> Vec<String> {
        if let Some(prefix_node) = self.find_node(prefix) {
            let mut results = Vec::new();
            prefix_node.collect_words(prefix, &mut results);
            results
        } else {
            Vec::new()
        }
    }
    
    /// Collect all words from this node recursively
    fn collect_words(&self, current_prefix: &str, results: &mut Vec<String>) {
        if self.is_end_of_word {
            results.push(current_prefix.to_string());
        }
        
        for (ch, child) in &self.children {
            let new_prefix = format!("{}{}", current_prefix, ch);
            child.collect_words(&new_prefix, results);
        }
    }
    
    /// Get all key-value pairs with a given prefix
    fn get_entries_with_prefix(&self, prefix: &str) -> Vec<(String, Value)> {
        if let Some(prefix_node) = self.find_node(prefix) {
            let mut results = Vec::new();
            prefix_node.collect_entries(prefix, &mut results);
            results
        } else {
            Vec::new()
        }
    }
    
    /// Collect all key-value pairs from this node recursively
    fn collect_entries(&self, current_prefix: &str, results: &mut Vec<(String, Value)>) {
        if self.is_end_of_word {
            if let Some(ref value) = self.value {
                results.push((current_prefix.to_string(), value.clone()));
            }
        }
        
        for (ch, child) in &self.children {
            let new_prefix = format!("{}{}", current_prefix, ch);
            child.collect_entries(&new_prefix, results);
        }
    }
    
    /// Count total number of words
    fn count_words(&self) -> usize {
        let mut count = if self.is_end_of_word { 1 } else { 0 };
        
        for child in self.children.values() {
            count += child.count_words();
        }
        
        count
    }
    
    /// Get the longest common prefix of all words in the trie
    fn get_longest_common_prefix(&self) -> String {
        let mut prefix = String::new();
        let mut current = self;
        
        while current.children.len() == 1 && !current.is_end_of_word {
            let (&ch, child) = current.children.iter().next().unwrap();
            prefix.push(ch);
            current = child;
        }
        
        prefix
    }
}

/// Trie (Prefix Tree) data structure
#[derive(Clone)]
pub struct Trie {
    root: TrieNode,
}

impl Trie {
    /// Create a new empty Trie
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
        }
    }
    
    /// Insert a word into the Trie
    pub fn insert(&mut self, word: &str) -> Result<(), ForeignError> {
        if word.is_empty() {
            return Err(ForeignError::RuntimeError {
                message: "Cannot insert empty string into Trie".to_string(),
            });
        }
        
        self.root.insert(word, None);
        Ok(())
    }
    
    /// Insert a word with an associated value
    pub fn insert_with_value(&mut self, word: &str, value: Value) -> Result<(), ForeignError> {
        if word.is_empty() {
            return Err(ForeignError::RuntimeError {
                message: "Cannot insert empty string into Trie".to_string(),
            });
        }
        
        self.root.insert(word, Some(value));
        Ok(())
    }
    
    /// Search for a word in the Trie
    pub fn search(&self, word: &str) -> bool {
        if word.is_empty() {
            return false;
        }
        self.root.search(word)
    }
    
    /// Check if any word starts with the given prefix
    pub fn starts_with(&self, prefix: &str) -> bool {
        if prefix.is_empty() {
            return true; // Empty prefix matches everything
        }
        self.root.starts_with(prefix)
    }
    
    /// Get value associated with a word
    pub fn get(&self, word: &str) -> Option<Value> {
        if word.is_empty() {
            return None;
        }
        self.root.get_value(word).cloned()
    }
    
    /// Delete a word from the Trie
    pub fn delete(&mut self, word: &str) -> Result<bool, ForeignError> {
        if word.is_empty() {
            return Err(ForeignError::RuntimeError {
                message: "Cannot delete empty string from Trie".to_string(),
            });
        }
        
        Ok(self.root.delete(word))
    }
    
    /// Get all words with a given prefix
    pub fn get_words_with_prefix(&self, prefix: &str) -> Vec<String> {
        self.root.get_words_with_prefix(prefix)
    }
    
    /// Get all key-value pairs with a given prefix  
    pub fn get_entries_with_prefix(&self, prefix: &str) -> Vec<(String, Value)> {
        self.root.get_entries_with_prefix(prefix)
    }
    
    /// Get all words in the Trie
    pub fn get_all_words(&self) -> Vec<String> {
        self.get_words_with_prefix("")
    }
    
    /// Count total number of words
    pub fn size(&self) -> usize {
        self.root.count_words()
    }
    
    /// Check if Trie is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    
    /// Clear all words from the Trie
    pub fn clear(&mut self) {
        self.root = TrieNode::new();
    }
    
    /// Get longest common prefix of all words
    pub fn get_longest_common_prefix(&self) -> String {
        self.root.get_longest_common_prefix()
    }
}

impl Foreign for Trie {
    fn type_name(&self) -> &'static str {
        "Trie"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut trie = self.clone();
        match method {
            "insert" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let word = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                trie.insert(word)?;
                Ok(Value::LyObj(LyObj::new(Box::new(trie))))
            }
            "insertWithValue" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let word = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                trie.insert_with_value(word, args[1].clone())?;
                Ok(Value::LyObj(LyObj::new(Box::new(trie))))
            }
            "search" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let word = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                Ok(Value::Boolean(trie.search(word)))
            }
            "startsWith" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let prefix = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                Ok(Value::Boolean(trie.starts_with(prefix)))
            }
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let word = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                match trie.get(word) {
                    Some(value) => Ok(value),
                    None => Ok(Value::Symbol("Missing".to_string())),
                }
            }
            "delete" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let word = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let deleted = trie.delete(word)?;
                Ok(Value::Boolean(deleted))
            }
            "getWordsWithPrefix" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let prefix = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let words = trie.get_words_with_prefix(prefix);
                let values: Vec<Value> = words.into_iter().map(Value::String).collect();
                Ok(Value::List(values))
            }
            "getAllWords" => {
                let words = trie.get_all_words();
                let values: Vec<Value> = words.into_iter().map(Value::String).collect();
                Ok(Value::List(values))
            }
            "size" => {
                Ok(Value::Integer(trie.size() as i64))
            }
            "isEmpty" => {
                Ok(Value::Boolean(trie.is_empty()))
            }
            "clear" => {
                trie.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(trie))))
            }
            "longestCommonPrefix" => {
                let prefix = trie.get_longest_common_prefix();
                Ok(Value::String(prefix))
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

impl fmt::Display for Trie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Trie[size: {}]", self.size())
    }
}

impl fmt::Debug for Trie {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Trie")
            .field("size", &self.size())
            .field("is_empty", &self.is_empty())
            .finish()
    }
}

/// Create a new empty Trie
pub fn trie_new(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime("Trie[] takes no arguments".to_string()));
    }
    
    let trie = Trie::new();
    Ok(Value::LyObj(LyObj::new(Box::new(trie))))
}

/// Insert a word into a Trie
pub fn trie_insert(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("insert", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Insert a word with value into a Trie
pub fn trie_insert_with_value(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("insertWithValue", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Search for a word in a Trie
pub fn trie_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("search", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Check if any words start with prefix in a Trie
pub fn trie_starts_with(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("startsWith", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Get value associated with a word in a Trie
pub fn trie_get(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("get", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Delete a word from a Trie
pub fn trie_delete(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("delete", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Get all words with a given prefix from a Trie
pub fn trie_get_words_with_prefix(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("getWordsWithPrefix", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a Trie".to_string()))
    }
}

/// Get all words from a Trie
pub fn trie_get_all_words(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("getAllWords", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a Trie".to_string()))
    }
}

/// Get the size of a Trie
pub fn trie_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("size", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a Trie".to_string()))
    }
}

/// Check if a Trie is empty
pub fn trie_is_empty(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("isEmpty", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a Trie".to_string()))
    }
}

/// Get longest common prefix of all words in a Trie
pub fn trie_longest_common_prefix(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(trie_obj) = &args[0] {
        trie_obj.call_method("longestCommonPrefix", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a Trie".to_string()))
    }
}

/// Register Trie functions with the standard library
pub fn register_trie_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("Trie".to_string(), trie_new);
    functions.insert("TrieInsert".to_string(), trie_insert);
    functions.insert("TrieInsertWithValue".to_string(), trie_insert_with_value);
    functions.insert("TrieSearch".to_string(), trie_search);
    functions.insert("TrieStartsWith".to_string(), trie_starts_with);
    functions.insert("TrieGet".to_string(), trie_get);
    functions.insert("TrieDelete".to_string(), trie_delete);
    functions.insert("TrieWordsWithPrefix".to_string(), trie_get_words_with_prefix);
    functions.insert("TrieAllWords".to_string(), trie_get_all_words);
    functions.insert("TrieSize".to_string(), trie_size);
    functions.insert("TrieIsEmpty".to_string(), trie_is_empty);
    functions.insert("TrieLongestPrefix".to_string(), trie_longest_common_prefix);
}

/// Get documentation for Trie functions
pub fn get_trie_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("Trie".to_string(), "Trie[] - Create empty prefix tree for string operations. O(1) time.".to_string());
    docs.insert("TrieInsert".to_string(), "TrieInsert[trie, word] - Insert word into trie. O(m) where m = word length.".to_string());
    docs.insert("TrieInsertWithValue".to_string(), "TrieInsertWithValue[trie, word, value] - Insert word with associated value. O(m) time.".to_string());
    docs.insert("TrieSearch".to_string(), "TrieSearch[trie, word] - Check if word exists in trie. O(m) time.".to_string());
    docs.insert("TrieStartsWith".to_string(), "TrieStartsWith[trie, prefix] - Check if any word starts with prefix. O(m) time.".to_string());
    docs.insert("TrieGet".to_string(), "TrieGet[trie, word] - Get value associated with word. O(m) time.".to_string());
    docs.insert("TrieDelete".to_string(), "TrieDelete[trie, word] - Delete word from trie. O(m) time.".to_string());
    docs.insert("TrieWordsWithPrefix".to_string(), "TrieWordsWithPrefix[trie, prefix] - Get all words starting with prefix. O(p) time.".to_string());
    docs.insert("TrieAllWords".to_string(), "TrieAllWords[trie] - Get all words in trie. O(n) time.".to_string());
    docs.insert("TrieSize".to_string(), "TrieSize[trie] - Get number of words in trie. O(n) time.".to_string());
    docs.insert("TrieIsEmpty".to_string(), "TrieIsEmpty[trie] - Check if trie is empty. O(1) time.".to_string());
    docs.insert("TrieLongestPrefix".to_string(), "TrieLongestPrefix[trie] - Get longest common prefix of all words. O(m) time.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trie_creation() {
        let trie = Trie::new();
        assert_eq!(trie.size(), 0);
        assert!(trie.is_empty());
    }
    
    #[test]
    fn test_trie_insertion_and_search() {
        let mut trie = Trie::new();
        
        trie.insert("hello").unwrap();
        trie.insert("world").unwrap();
        trie.insert("hell").unwrap();
        trie.insert("help").unwrap();
        
        assert_eq!(trie.size(), 4);
        assert!(!trie.is_empty());
        
        assert!(trie.search("hello"));
        assert!(trie.search("world"));
        assert!(trie.search("hell"));
        assert!(trie.search("help"));
        assert!(!trie.search("he"));
        assert!(!trie.search("helper"));
    }
    
    #[test]
    fn test_trie_starts_with() {
        let mut trie = Trie::new();
        
        trie.insert("cat").unwrap();
        trie.insert("car").unwrap();
        trie.insert("card").unwrap();
        trie.insert("care").unwrap();
        trie.insert("careful").unwrap();
        
        assert!(trie.starts_with("c"));
        assert!(trie.starts_with("ca"));
        assert!(trie.starts_with("car"));
        assert!(trie.starts_with("care"));
        assert!(!trie.starts_with("dog"));
        assert!(!trie.starts_with("bat"));
    }
    
    #[test]
    fn test_trie_with_values() {
        let mut trie = Trie::new();
        
        trie.insert_with_value("apple", Value::Integer(1)).unwrap();
        trie.insert_with_value("app", Value::Integer(2)).unwrap();
        trie.insert_with_value("application", Value::Integer(3)).unwrap();
        
        assert_eq!(trie.get("apple"), Some(Value::Integer(1)));
        assert_eq!(trie.get("app"), Some(Value::Integer(2)));
        assert_eq!(trie.get("application"), Some(Value::Integer(3)));
        assert_eq!(trie.get("ap"), None);
    }
    
    #[test]
    fn test_trie_words_with_prefix() {
        let mut trie = Trie::new();
        
        trie.insert("cat").unwrap();
        trie.insert("car").unwrap();
        trie.insert("card").unwrap();
        trie.insert("care").unwrap();
        trie.insert("dog").unwrap();
        
        let mut car_words = trie.get_words_with_prefix("car");
        car_words.sort(); // For consistent testing
        
        assert_eq!(car_words, vec!["car", "card", "care"]);
        
        let ca_words = trie.get_words_with_prefix("ca");
        assert_eq!(ca_words.len(), 4); // cat, car, card, care
        
        let empty_words = trie.get_words_with_prefix("xyz");
        assert!(empty_words.is_empty());
    }
    
    #[test]
    fn test_trie_deletion() {
        let mut trie = Trie::new();
        
        trie.insert("cat").unwrap();
        trie.insert("car").unwrap();
        trie.insert("card").unwrap();
        trie.insert("care").unwrap();
        
        assert_eq!(trie.size(), 4);
        
        // Delete a leaf word
        assert!(trie.delete("card").unwrap());
        assert!(!trie.search("card"));
        assert_eq!(trie.size(), 3);
        
        // Delete a word that is prefix of others
        assert!(trie.delete("car").unwrap());
        assert!(!trie.search("car"));
        assert!(trie.search("care")); // Should still exist
        assert_eq!(trie.size(), 2);
        
        // Try to delete non-existent word
        assert!(!trie.delete("dog").unwrap());
        assert_eq!(trie.size(), 2);
    }
    
    #[test]
    fn test_trie_longest_common_prefix() {
        let mut trie = Trie::new();
        
        // Test with common prefix
        trie.insert("flower").unwrap();
        trie.insert("flow").unwrap();
        trie.insert("flight").unwrap();
        
        let prefix = trie.get_longest_common_prefix();
        assert_eq!(prefix, "fl");
        
        // Test with no common prefix
        let mut trie2 = Trie::new();
        trie2.insert("cat").unwrap();
        trie2.insert("dog").unwrap();
        
        let prefix2 = trie2.get_longest_common_prefix();
        assert_eq!(prefix2, "");
        
        // Test empty trie
        let trie3 = Trie::new();
        let prefix3 = trie3.get_longest_common_prefix();
        assert_eq!(prefix3, "");
    }
    
    #[test]
    fn test_trie_all_words() {
        let mut trie = Trie::new();
        
        trie.insert("cat").unwrap();
        trie.insert("car").unwrap();
        trie.insert("dog").unwrap();
        
        let mut all_words = trie.get_all_words();
        all_words.sort(); // For consistent testing
        
        assert_eq!(all_words, vec!["car", "cat", "dog"]);
    }
    
    #[test]
    fn test_empty_string_handling() {
        let mut trie = Trie::new();
        
        // Should return error for empty string
        assert!(trie.insert("").is_err());
        assert!(trie.insert_with_value("", Value::Integer(1)).is_err());
        assert!(trie.delete("").is_err());
        
        // But search should return false for empty string
        assert!(!trie.search(""));
        assert_eq!(trie.get(""), None);
        
        // Empty prefix should return true for starts_with
        trie.insert("test").unwrap();
        assert!(trie.starts_with(""));
    }
}