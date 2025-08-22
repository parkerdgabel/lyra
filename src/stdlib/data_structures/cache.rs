//! Cache data structures for Lyra Standard Library
//!
//! This module provides efficient cache implementations including LRU and LFU caches.
//! All implementations use the Foreign object pattern for VM integration and provide
//! O(1) access times for optimal performance.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::fmt;

// LRU Cache Implementation using HashMap + Doubly Linked List
// This provides O(1) get, put, and eviction operations

/// Node in the doubly-linked list for LRU cache
#[derive(Debug, Clone)]
struct LRUNode {
    key: Value,
    value: Value,
    prev: Option<usize>,
    next: Option<usize>,
}

impl LRUNode {
    fn new(key: Value, value: Value) -> Self {
        Self {
            key,
            value,
            prev: None,
            next: None,
        }
    }
}

/// LRU (Least Recently Used) Cache implementation
#[derive(Clone)]
pub struct LRUCache {
    capacity: usize,
    size: usize,
    nodes: Vec<LRUNode>,
    map: HashMap<String, usize>, // Key to node index mapping
    head: Option<usize>,
    tail: Option<usize>,
    free_indices: Vec<usize>,
}

impl LRUCache {
    /// Create a new LRU cache with the specified capacity
    pub fn new(capacity: usize) -> Result<Self, ForeignError> {
        if capacity == 0 {
            return Err(ForeignError::RuntimeError {
                message: "LRU cache capacity must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            capacity,
            size: 0,
            nodes: Vec::new(),
            map: HashMap::new(),
            head: None,
            tail: None,
            free_indices: Vec::new(),
        })
    }

    /// Convert a Value to a string key for internal storage
    fn value_to_key(value: &Value) -> String {
        match value {
            Value::String(s) => s.clone(),
            Value::Integer(i) => i.to_string(),
            Value::Real(r) => r.to_string(),
            Value::Symbol(s) => s.clone(),
            Value::Boolean(b) => b.to_string(),
            _ => format!("{:?}", value), // Fallback for complex types
        }
    }

    /// Get a value from the cache, updating its position to most recent
    pub fn get(&mut self, key: &Value) -> Option<Value> {
        let key_str = Self::value_to_key(key);
        
        if let Some(&node_idx) = self.map.get(&key_str) {
            // Move to front (most recent)
            self.move_to_front(node_idx);
            Some(self.nodes[node_idx].value.clone())
        } else {
            None
        }
    }

    /// Put a key-value pair into the cache
    pub fn put(&mut self, key: Value, value: Value) -> Result<(), ForeignError> {
        let key_str = Self::value_to_key(&key);

        if let Some(&node_idx) = self.map.get(&key_str) {
            // Update existing key
            self.nodes[node_idx].value = value;
            self.move_to_front(node_idx);
        } else {
            // Add new key
            if self.size >= self.capacity {
                self.evict_lru()?;
            }
            
            let node_idx = self.allocate_node();
            self.nodes[node_idx] = LRUNode::new(key, value);
            self.map.insert(key_str, node_idx);
            self.add_to_front(node_idx);
            self.size += 1;
        }

        Ok(())
    }

    /// Remove and return the least recently used item
    pub fn evict(&mut self) -> Result<Option<(Value, Value)>, ForeignError> {
        if self.size == 0 {
            return Ok(None);
        }

        self.evict_lru().map(Some)
    }

    /// Check if a key exists in the cache (without updating order)
    pub fn contains(&self, key: &Value) -> bool {
        let key_str = Self::value_to_key(key);
        self.map.contains_key(&key_str)
    }

    /// Get current size of the cache
    pub fn size(&self) -> usize {
        self.size
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.map.clear();
        self.free_indices.clear();
        self.head = None;
        self.tail = None;
        self.size = 0;
    }

    /// Get all keys in order from most recent to least recent
    pub fn keys(&self) -> Vec<Value> {
        let mut keys = Vec::new();
        let mut current = self.head;
        
        while let Some(idx) = current {
            keys.push(self.nodes[idx].key.clone());
            current = self.nodes[idx].next;
        }
        
        keys
    }

    /// Internal helper methods

    fn allocate_node(&mut self) -> usize {
        if let Some(idx) = self.free_indices.pop() {
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(LRUNode::new(Value::Symbol("dummy".to_string()), Value::Symbol("dummy".to_string())));
            idx
        }
    }

    fn deallocate_node(&mut self, idx: usize) {
        self.free_indices.push(idx);
    }

    fn move_to_front(&mut self, node_idx: usize) {
        if Some(node_idx) == self.head {
            return; // Already at front
        }

        self.remove_from_list(node_idx);
        self.add_to_front(node_idx);
    }

    fn remove_from_list(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        let prev = node.prev;
        let next = node.next;

        if let Some(prev_idx) = prev {
            self.nodes[prev_idx].next = next;
        } else {
            self.head = next;
        }

        if let Some(next_idx) = next {
            self.nodes[next_idx].prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn add_to_front(&mut self, node_idx: usize) {
        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = self.head;

        if let Some(head_idx) = self.head {
            self.nodes[head_idx].prev = Some(node_idx);
        } else {
            self.tail = Some(node_idx);
        }

        self.head = Some(node_idx);
    }

    fn evict_lru(&mut self) -> Result<(Value, Value), ForeignError> {
        let tail_idx = self.tail.ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot evict from empty cache".to_string(),
        })?;

        let tail_node = self.nodes[tail_idx].clone();
        let key_str = Self::value_to_key(&tail_node.key);

        self.remove_from_list(tail_idx);
        self.map.remove(&key_str);
        self.deallocate_node(tail_idx);
        self.size -= 1;

        Ok((tail_node.key, tail_node.value))
    }
}

impl Foreign for LRUCache {
    fn type_name(&self) -> &'static str {
        "LRUCache"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut cache = self.clone();
        match method {
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                match cache.get(&args[0]) {
                    Some(value) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        value,
                    ])),
                    None => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::Symbol("Missing".to_string()),
                    ])),
                }
            }
            "put" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                cache.put(args[0].clone(), args[1].clone())?;
                Ok(Value::LyObj(LyObj::new(Box::new(cache))))
            }
            "size" => {
                Ok(Value::Integer(cache.size() as i64))
            }
            "clear" => {
                cache.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(cache))))
            }
            "evict" => {
                match cache.evict()? {
                    Some((key, value)) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::List(vec![key, value]),
                    ])),
                    None => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::Symbol("Missing".to_string()),
                    ])),
                }
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                Ok(Value::Boolean(cache.contains(&args[0])))
            }
            "keys" => {
                let keys = cache.keys();
                Ok(Value::List(keys))
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

impl fmt::Display for LRUCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LRUCache[capacity: {}, size: {}]", self.capacity, self.size)
    }
}

impl fmt::Debug for LRUCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LRUCache")
            .field("capacity", &self.capacity)
            .field("size", &self.size)
            .finish()
    }
}

/// Create a new LRU cache with specified capacity
pub fn lru_cache_new(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    let capacity = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::Runtime("LRUCache capacity must be an integer".to_string())),
    };

    let cache = LRUCache::new(capacity).map_err(|e| VmError::Runtime(e.to_string()))?;
    Ok(Value::LyObj(LyObj::new(Box::new(cache))))
}

/// Get a value from LRU cache
pub fn lru_cache_get(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("get", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LRUCache".to_string()))
    }
}

/// Put a key-value pair into LRU cache
pub fn lru_cache_put(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("put", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LRUCache".to_string()))
    }
}

/// Get the size of LRU cache
pub fn lru_cache_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("size", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LRUCache".to_string()))
    }
}

/// Clear all items from LRU cache
pub fn lru_cache_clear(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("clear", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LRUCache".to_string()))
    }
}

/// Force evict least recently used item from cache
pub fn lru_cache_evict(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("evict", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LRUCache".to_string()))
    }
}

/// Check if key exists in LRU cache
pub fn lru_cache_contains(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("contains", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LRUCache".to_string()))
    }
}

/// Get all keys from LRU cache in order
pub fn lru_cache_keys(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("keys", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LRUCache".to_string()))
    }
}

/// Register LRU cache functions with the standard library
pub fn register_lru_cache_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("LRUCache".to_string(), lru_cache_new);
    functions.insert("LRUGet".to_string(), lru_cache_get);
    functions.insert("LRUPut".to_string(), lru_cache_put);
    functions.insert("LRUSize".to_string(), lru_cache_size);
    functions.insert("LRUClear".to_string(), lru_cache_clear);
    functions.insert("LRUEvict".to_string(), lru_cache_evict);
    functions.insert("LRUContains".to_string(), lru_cache_contains);
    functions.insert("LRUKeys".to_string(), lru_cache_keys);
}

/// Get documentation for LRU cache functions
pub fn get_lru_cache_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("LRUCache".to_string(), "LRUCache[capacity] - Create LRU cache with specified capacity. O(1) operations.".to_string());
    docs.insert("LRUGet".to_string(), "LRUGet[cache, key] - Get value and mark as most recent. Returns {newCache, value} or {newCache, Missing}. O(1) time.".to_string());
    docs.insert("LRUPut".to_string(), "LRUPut[cache, key, value] - Insert/update key-value pair. Evicts LRU if at capacity. O(1) time.".to_string());
    docs.insert("LRUSize".to_string(), "LRUSize[cache] - Get current number of items in cache. O(1) time.".to_string());
    docs.insert("LRUClear".to_string(), "LRUClear[cache] - Remove all items from cache. O(1) time.".to_string());
    docs.insert("LRUEvict".to_string(), "LRUEvict[cache] - Force eviction of LRU item. Returns {newCache, {key, value}} or {newCache, Missing}. O(1) time.".to_string());
    docs.insert("LRUContains".to_string(), "LRUContains[cache, key] - Check if key exists without updating order. O(1) time.".to_string());
    docs.insert("LRUKeys".to_string(), "LRUKeys[cache] - Get all keys in order from most recent to least recent. O(n) time.".to_string());
    docs
}

// LFU Cache Implementation using frequency-based eviction
// This provides O(1) get, put operations with LFU eviction policy

/// Node for LFU cache with frequency tracking
#[derive(Debug, Clone)]
struct LFUNode {
    key: Value,
    value: Value,
    frequency: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

impl LFUNode {
    fn new(key: Value, value: Value) -> Self {
        Self {
            key,
            value,
            frequency: 1,
            prev: None,
            next: None,
        }
    }
}

/// Frequency bucket containing doubly-linked list of nodes
#[derive(Debug, Clone)]
struct FrequencyBucket {
    head: Option<usize>,
    tail: Option<usize>,
    size: usize,
}

impl FrequencyBucket {
    fn new() -> Self {
        Self {
            head: None,
            tail: None,
            size: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.size == 0
    }
}

/// LFU (Least Frequently Used) Cache implementation
#[derive(Clone)]
pub struct LFUCache {
    capacity: usize,
    size: usize,
    nodes: Vec<LFUNode>,
    map: HashMap<String, usize>, // Key to node index mapping
    freq_buckets: HashMap<usize, FrequencyBucket>, // Frequency to bucket mapping
    min_frequency: usize,
    free_indices: Vec<usize>,
}

impl LFUCache {
    /// Create a new LFU cache with the specified capacity
    pub fn new(capacity: usize) -> Result<Self, ForeignError> {
        if capacity == 0 {
            return Err(ForeignError::RuntimeError {
                message: "LFU cache capacity must be greater than 0".to_string(),
            });
        }

        Ok(Self {
            capacity,
            size: 0,
            nodes: Vec::new(),
            map: HashMap::new(),
            freq_buckets: HashMap::new(),
            min_frequency: 0,
            free_indices: Vec::new(),
        })
    }

    /// Convert a Value to a string key for internal storage
    fn value_to_key(value: &Value) -> String {
        LRUCache::value_to_key(value) // Reuse LRU implementation
    }

    /// Get a value from the cache, updating its frequency
    pub fn get(&mut self, key: &Value) -> Option<Value> {
        let key_str = Self::value_to_key(key);
        
        if let Some(&node_idx) = self.map.get(&key_str) {
            let value = self.nodes[node_idx].value.clone();
            self.increment_frequency(node_idx);
            Some(value)
        } else {
            None
        }
    }

    /// Put a key-value pair into the cache
    pub fn put(&mut self, key: Value, value: Value) -> Result<(), ForeignError> {
        let key_str = Self::value_to_key(&key);

        if let Some(&node_idx) = self.map.get(&key_str) {
            // Update existing key
            self.nodes[node_idx].value = value;
            self.increment_frequency(node_idx);
        } else {
            // Add new key
            if self.size >= self.capacity {
                self.evict_lfu()?;
            }
            
            let node_idx = self.allocate_node();
            self.nodes[node_idx] = LFUNode::new(key, value);
            self.map.insert(key_str, node_idx);
            self.add_to_frequency_bucket(node_idx, 1);
            self.min_frequency = 1;
            self.size += 1;
        }

        Ok(())
    }

    /// Remove and return the least frequently used item
    pub fn evict(&mut self) -> Result<Option<(Value, Value)>, ForeignError> {
        if self.size == 0 {
            return Ok(None);
        }

        self.evict_lfu().map(Some)
    }

    /// Check if a key exists in the cache (without updating frequency)
    pub fn contains(&self, key: &Value) -> bool {
        let key_str = Self::value_to_key(key);
        self.map.contains_key(&key_str)
    }

    /// Get current size of the cache
    pub fn size(&self) -> usize {
        self.size
    }

    /// Clear all items from the cache
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.map.clear();
        self.freq_buckets.clear();
        self.free_indices.clear();
        self.min_frequency = 0;
        self.size = 0;
    }

    /// Get the frequency of a specific key
    pub fn frequency(&self, key: &Value) -> Option<usize> {
        let key_str = Self::value_to_key(key);
        if let Some(&node_idx) = self.map.get(&key_str) {
            Some(self.nodes[node_idx].frequency)
        } else {
            None
        }
    }

    /// Get all keys in order from least frequent to most frequent
    pub fn keys(&self) -> Vec<Value> {
        let mut keys = Vec::new();
        
        // Collect all frequencies and sort them
        let mut frequencies: Vec<usize> = self.freq_buckets.keys().cloned().collect();
        frequencies.sort();
        
        for freq in frequencies {
            if let Some(bucket) = self.freq_buckets.get(&freq) {
                let mut current = bucket.head;
                while let Some(idx) = current {
                    keys.push(self.nodes[idx].key.clone());
                    current = self.nodes[idx].next;
                }
            }
        }
        
        keys
    }

    /// Internal helper methods

    fn allocate_node(&mut self) -> usize {
        if let Some(idx) = self.free_indices.pop() {
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(LFUNode::new(Value::Symbol("dummy".to_string()), Value::Symbol("dummy".to_string())));
            idx
        }
    }

    fn deallocate_node(&mut self, idx: usize) {
        self.free_indices.push(idx);
    }

    fn increment_frequency(&mut self, node_idx: usize) {
        let old_freq = self.nodes[node_idx].frequency;
        let new_freq = old_freq + 1;
        
        // Remove from old frequency bucket
        self.remove_from_frequency_bucket(node_idx, old_freq);
        
        // Add to new frequency bucket
        self.nodes[node_idx].frequency = new_freq;
        self.add_to_frequency_bucket(node_idx, new_freq);
        
        // Update min_frequency if needed
        if old_freq == self.min_frequency && self.is_frequency_bucket_empty(old_freq) {
            self.min_frequency = new_freq;
        }
    }

    fn add_to_frequency_bucket(&mut self, node_idx: usize, frequency: usize) {
        let bucket = self.freq_buckets.entry(frequency).or_insert_with(FrequencyBucket::new);
        
        self.nodes[node_idx].prev = None;
        self.nodes[node_idx].next = bucket.head;
        
        if let Some(head_idx) = bucket.head {
            self.nodes[head_idx].prev = Some(node_idx);
        } else {
            bucket.tail = Some(node_idx);
        }
        
        bucket.head = Some(node_idx);
        bucket.size += 1;
    }

    fn remove_from_frequency_bucket(&mut self, node_idx: usize, frequency: usize) {
        let node = &self.nodes[node_idx];
        let prev = node.prev;
        let next = node.next;
        
        if let Some(bucket) = self.freq_buckets.get_mut(&frequency) {
            if let Some(prev_idx) = prev {
                self.nodes[prev_idx].next = next;
            } else {
                bucket.head = next;
            }
            
            if let Some(next_idx) = next {
                self.nodes[next_idx].prev = prev;
            } else {
                bucket.tail = prev;
            }
            
            bucket.size -= 1;
            
            if bucket.size == 0 {
                self.freq_buckets.remove(&frequency);
            }
        }
    }

    fn is_frequency_bucket_empty(&self, frequency: usize) -> bool {
        !self.freq_buckets.contains_key(&frequency)
    }

    fn evict_lfu(&mut self) -> Result<(Value, Value), ForeignError> {
        // Find the tail of the minimum frequency bucket
        let bucket = self.freq_buckets.get(&self.min_frequency)
            .ok_or_else(|| ForeignError::RuntimeError {
                message: "Cannot evict from empty cache".to_string(),
            })?;

        let tail_idx = bucket.tail.ok_or_else(|| ForeignError::RuntimeError {
            message: "Cannot evict from empty bucket".to_string(),
        })?;

        let tail_node = self.nodes[tail_idx].clone();
        let key_str = Self::value_to_key(&tail_node.key);

        self.remove_from_frequency_bucket(tail_idx, tail_node.frequency);
        self.map.remove(&key_str);
        self.deallocate_node(tail_idx);
        self.size -= 1;

        // Update min_frequency if this was the last node with min frequency
        if self.is_frequency_bucket_empty(self.min_frequency) {
            self.min_frequency += 1;
        }

        Ok((tail_node.key, tail_node.value))
    }
}

impl Foreign for LFUCache {
    fn type_name(&self) -> &'static str {
        "LFUCache"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut cache = self.clone();
        match method {
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                match cache.get(&args[0]) {
                    Some(value) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        value,
                    ])),
                    None => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::Symbol("Missing".to_string()),
                    ])),
                }
            }
            "put" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                cache.put(args[0].clone(), args[1].clone())?;
                Ok(Value::LyObj(LyObj::new(Box::new(cache))))
            }
            "size" => {
                Ok(Value::Integer(cache.size() as i64))
            }
            "clear" => {
                cache.clear();
                Ok(Value::LyObj(LyObj::new(Box::new(cache))))
            }
            "evict" => {
                match cache.evict()? {
                    Some((key, value)) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::List(vec![key, value]),
                    ])),
                    None => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(cache))),
                        Value::Symbol("Missing".to_string()),
                    ])),
                }
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                Ok(Value::Boolean(cache.contains(&args[0])))
            }
            "frequency" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                match cache.frequency(&args[0]) {
                    Some(freq) => Ok(Value::Integer(freq as i64)),
                    None => Ok(Value::Symbol("Missing".to_string())),
                }
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

impl fmt::Display for LFUCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LFUCache[capacity: {}, size: {}]", self.capacity, self.size)
    }
}

impl fmt::Debug for LFUCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LFUCache")
            .field("capacity", &self.capacity)
            .field("size", &self.size)
            .field("min_frequency", &self.min_frequency)
            .finish()
    }
}

/// Create a new LFU cache with specified capacity
pub fn lfu_cache_new(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    let capacity = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::Runtime("LFUCache capacity must be an integer".to_string())),
    };

    let cache = LFUCache::new(capacity).map_err(|e| VmError::Runtime(e.to_string()))?;
    Ok(Value::LyObj(LyObj::new(Box::new(cache))))
}

/// Get a value from LFU cache
pub fn lfu_cache_get(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("get", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LFUCache".to_string()))
    }
}

/// Put a key-value pair into LFU cache
pub fn lfu_cache_put(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("put", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LFUCache".to_string()))
    }
}

/// Get the size of LFU cache
pub fn lfu_cache_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("size", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LFUCache".to_string()))
    }
}

/// Clear all items from LFU cache
pub fn lfu_cache_clear(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("clear", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LFUCache".to_string()))
    }
}

/// Force evict least frequently used item from cache
pub fn lfu_cache_evict(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("evict", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be an LFUCache".to_string()))
    }
}

/// Check if key exists in LFU cache
pub fn lfu_cache_contains(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("contains", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LFUCache".to_string()))
    }
}

/// Get access frequency of a key in LFU cache
pub fn lfu_cache_frequency(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;

    if let Value::LyObj(cache_obj) = &args[0] {
        cache_obj.call_method("frequency", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be an LFUCache".to_string()))
    }
}

/// Register cache functions with the standard library
pub fn register_cache_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    // Register LRU functions
    register_lru_cache_functions(functions);
    
    // Register LFU functions
    functions.insert("LFUCache".to_string(), lfu_cache_new);
    functions.insert("LFUGet".to_string(), lfu_cache_get);
    functions.insert("LFUPut".to_string(), lfu_cache_put);
    functions.insert("LFUSize".to_string(), lfu_cache_size);
    functions.insert("LFUClear".to_string(), lfu_cache_clear);
    functions.insert("LFUEvict".to_string(), lfu_cache_evict);
    functions.insert("LFUContains".to_string(), lfu_cache_contains);
    functions.insert("LFUFrequency".to_string(), lfu_cache_frequency);
}

/// Get documentation for cache functions
pub fn get_cache_documentation() -> HashMap<String, String> {
    let mut docs = get_lru_cache_documentation();
    
    // Add LFU documentation
    docs.insert("LFUCache".to_string(), "LFUCache[capacity] - Create LFU cache with specified capacity. O(1) operations.".to_string());
    docs.insert("LFUGet".to_string(), "LFUGet[cache, key] - Get value and increment frequency. Returns {newCache, value} or {newCache, Missing}. O(1) time.".to_string());
    docs.insert("LFUPut".to_string(), "LFUPut[cache, key, value] - Insert/update key-value pair. Evicts LFU if at capacity. O(1) time.".to_string());
    docs.insert("LFUSize".to_string(), "LFUSize[cache] - Get current number of items in cache. O(1) time.".to_string());
    docs.insert("LFUClear".to_string(), "LFUClear[cache] - Remove all items from cache. O(1) time.".to_string());
    docs.insert("LFUEvict".to_string(), "LFUEvict[cache] - Force eviction of LFU item. Returns {newCache, {key, value}} or {newCache, Missing}. O(1) time.".to_string());
    docs.insert("LFUContains".to_string(), "LFUContains[cache, key] - Check if key exists without updating frequency. O(1) time.".to_string());
    docs.insert("LFUFrequency".to_string(), "LFUFrequency[cache, key] - Get access frequency of key. Returns frequency or Missing. O(1) time.".to_string());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru_cache_creation() {
        let cache = LRUCache::new(3).unwrap();
        assert_eq!(cache.capacity, 3);
        assert_eq!(cache.size(), 0);
    }

    #[test]
    fn test_lru_cache_put_get() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        
        assert_eq!(cache.get(&Value::String("key1".to_string())), Some(Value::Integer(1)));
        assert_eq!(cache.get(&Value::String("key2".to_string())), Some(Value::Integer(2)));
        assert_eq!(cache.size(), 2);
    }

    #[test]
    fn test_lru_cache_eviction() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        cache.put(Value::String("key3".to_string()), Value::Integer(3)).unwrap(); // Should evict key1
        
        assert_eq!(cache.get(&Value::String("key1".to_string())), None);
        assert_eq!(cache.get(&Value::String("key2".to_string())), Some(Value::Integer(2)));
        assert_eq!(cache.get(&Value::String("key3".to_string())), Some(Value::Integer(3)));
        assert_eq!(cache.size(), 2);
    }

    #[test]
    fn test_lru_cache_update_order() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        
        // Access key1 to make it most recent
        cache.get(&Value::String("key1".to_string()));
        
        // Adding key3 should evict key2 (not key1)
        cache.put(Value::String("key3".to_string()), Value::Integer(3)).unwrap();
        
        assert_eq!(cache.get(&Value::String("key1".to_string())), Some(Value::Integer(1)));
        assert_eq!(cache.get(&Value::String("key2".to_string())), None);
        assert_eq!(cache.get(&Value::String("key3".to_string())), Some(Value::Integer(3)));
    }

    #[test]
    fn test_lru_cache_contains() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        
        assert!(cache.contains(&Value::String("key1".to_string())));
        assert!(!cache.contains(&Value::String("key2".to_string())));
    }

    #[test]
    fn test_lru_cache_clear() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        
        assert_eq!(cache.size(), 2);
        cache.clear();
        assert_eq!(cache.size(), 0);
        assert!(!cache.contains(&Value::String("key1".to_string())));
    }

    #[test]
    fn test_lru_cache_keys_order() {
        let mut cache = LRUCache::new(3).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        cache.put(Value::String("key3".to_string()), Value::Integer(3)).unwrap();
        
        // Access key1 to make it most recent
        cache.get(&Value::String("key1".to_string()));
        
        let keys = cache.keys();
        assert_eq!(keys[0], Value::String("key1".to_string())); // Most recent
        assert_eq!(keys[1], Value::String("key3".to_string()));
        assert_eq!(keys[2], Value::String("key2".to_string())); // Least recent
    }

    #[test]
    fn test_lru_cache_evict_manual() {
        let mut cache = LRUCache::new(2).unwrap();
        
        cache.put(Value::String("key1".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::String("key2".to_string()), Value::Integer(2)).unwrap();
        
        let evicted = cache.evict().unwrap();
        assert!(evicted.is_some());
        
        let (key, value) = evicted.unwrap();
        assert_eq!(key, Value::String("key1".to_string())); // Least recent
        assert_eq!(value, Value::Integer(1));
        assert_eq!(cache.size(), 1);
    }

    #[test]
    fn test_lru_cache_zero_capacity_error() {
        assert!(LRUCache::new(0).is_err());
    }

    #[test]
    fn test_lru_cache_mixed_key_types() {
        let mut cache = LRUCache::new(3).unwrap();
        
        cache.put(Value::String("str_key".to_string()), Value::Integer(1)).unwrap();
        cache.put(Value::Integer(42), Value::String("int_key".to_string())).unwrap();
        cache.put(Value::Boolean(true), Value::Real(3.14)).unwrap();
        
        assert_eq!(cache.get(&Value::String("str_key".to_string())), Some(Value::Integer(1)));
        assert_eq!(cache.get(&Value::Integer(42)), Some(Value::String("int_key".to_string())));
        assert_eq!(cache.get(&Value::Boolean(true)), Some(Value::Real(3.14)));
    }
}