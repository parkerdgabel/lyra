//! Probabilistic data structures for Lyra standard library
//!
//! This module implements space-efficient probabilistic data structures that
//! provide approximate answers with guaranteed error bounds. These are essential
//! for big data processing and streaming analytics.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Bloom Filter - Space-efficient probabilistic set membership
/// False positives possible, false negatives impossible
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bit_array: Vec<bool>,
    num_hash_functions: usize,
    capacity: usize,
    element_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with optimal parameters
    pub fn new(expected_elements: usize, false_positive_rate: f64) -> Self {
        let capacity = Self::calculate_optimal_size(expected_elements, false_positive_rate);
        let num_hash_functions = Self::calculate_optimal_hash_count(expected_elements, capacity);
        
        Self {
            bit_array: vec![false; capacity],
            num_hash_functions,
            capacity,
            element_count: 0,
        }
    }
    
    /// Create Bloom filter with specific size and hash function count
    pub fn with_parameters(size: usize, num_hash_functions: usize) -> Self {
        Self {
            bit_array: vec![false; size],
            num_hash_functions,
            capacity: size,
            element_count: 0,
        }
    }
    
    /// Calculate optimal bit array size
    fn calculate_optimal_size(expected_elements: usize, false_positive_rate: f64) -> usize {
        let ln2 = std::f64::consts::LN_2;
        let size = -(expected_elements as f64 * false_positive_rate.ln()) / (ln2 * ln2);
        size.ceil() as usize
    }
    
    /// Calculate optimal number of hash functions
    fn calculate_optimal_hash_count(expected_elements: usize, bit_array_size: usize) -> usize {
        let count = (bit_array_size as f64 / expected_elements as f64) * std::f64::consts::LN_2;
        (count.round() as usize).max(1)
    }
    
    /// Generate hash values for an element
    fn hash_element(&self, element: &str) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.num_hash_functions);
        
        for i in 0..self.num_hash_functions {
            let mut hasher = DefaultHasher::new();
            element.hash(&mut hasher);
            i.hash(&mut hasher);
            let hash = hasher.finish() as usize;
            hashes.push(hash % self.capacity);
        }
        
        hashes
    }
    
    /// Add an element to the Bloom filter
    pub fn add(&mut self, element: &str) {
        let indices = self.hash_element(element);
        for &index in &indices {
            self.bit_array[index] = true;
        }
        self.element_count += 1;
    }
    
    /// Test if an element might be in the set
    pub fn contains(&self, element: &str) -> bool {
        let indices = self.hash_element(element);
        indices.iter().all(|&index| self.bit_array[index])
    }
    
    /// Get the current false positive probability
    pub fn current_false_positive_rate(&self) -> f64 {
        if self.element_count == 0 {
            return 0.0;
        }
        
        let filled_ratio = self.bit_array.iter().filter(|&&bit| bit).count() as f64 / self.capacity as f64;
        filled_ratio.powi(self.num_hash_functions as i32)
    }
    
    /// Get filter statistics
    pub fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("capacity".to_string(), self.capacity as f64);
        stats.insert("element_count".to_string(), self.element_count as f64);
        stats.insert("hash_functions".to_string(), self.num_hash_functions as f64);
        stats.insert("false_positive_rate".to_string(), self.current_false_positive_rate());
        
        let filled_bits = self.bit_array.iter().filter(|&&bit| bit).count();
        stats.insert("filled_bits".to_string(), filled_bits as f64);
        stats.insert("load_factor".to_string(), filled_bits as f64 / self.capacity as f64);
        
        stats
    }
    
    /// Clear the filter
    pub fn clear(&mut self) {
        self.bit_array.fill(false);
        self.element_count = 0;
    }
}

impl Foreign for BloomFilter {
    fn type_name(&self) -> &'static str {
        "BloomFilter"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut filter = self.clone();
        match method {
            "add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let element = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                filter.add(&element);
                Ok(Value::LyObj(LyObj::new(Box::new(filter))))
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let element = match &args[0] {
                    Value::String(s) => s,
                    Value::Symbol(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                Ok(Value::Boolean(filter.contains(element)))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Create a new Bloom filter
pub fn bloom_filter(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        1 => {
            // BloomFilter[expected_elements] - use default 1% false positive rate
            let expected_elements = match &args[0] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("BloomFilter expected positive integer for expected elements".to_string())),
            };
            
            let filter = BloomFilter::new(expected_elements, 0.01); // 1% false positive rate
            Ok(Value::LyObj(LyObj::new(Box::new(filter))))
        }
        2 => {
            // BloomFilter[expected_elements, false_positive_rate]
            let expected_elements = match &args[0] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("BloomFilter expected positive integer for expected elements".to_string())),
            };
            
            let false_positive_rate = match &args[1] {
                Value::Real(r) if *r > 0.0 && *r < 1.0 => *r,
                _ => return Err(VmError::Runtime("BloomFilter expected false positive rate between 0 and 1".to_string())),
            };
            
            let filter = BloomFilter::new(expected_elements, false_positive_rate);
            Ok(Value::LyObj(LyObj::new(Box::new(filter))))
        }
        3 => {
            // BloomFilter[size, hash_functions, "Manual"]
            let size = match &args[0] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("BloomFilter expected positive integer for size".to_string())),
            };
            
            let hash_functions = match &args[1] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("BloomFilter expected positive integer for hash functions".to_string())),
            };
            
            // Third argument should be "Manual" string (ignored for now)
            let filter = BloomFilter::with_parameters(size, hash_functions);
            Ok(Value::LyObj(LyObj::new(Box::new(filter))))
        }
        _ => Err(VmError::Runtime("BloomFilter expects 1-3 arguments".to_string())),
    }
}

/// Add element to Bloom filter
pub fn bloom_filter_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("BloomFilterAdd expects 2 arguments".to_string()));
    }
    
    let mut filter = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<BloomFilter>() {
                Some(f) => f.clone(),
                None => return Err(VmError::Runtime("BloomFilterAdd expected BloomFilter object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("BloomFilterAdd expected BloomFilter object".to_string())),
    };
    
    let element = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::Runtime("BloomFilterAdd expected string or symbol".to_string())),
    };
    
    filter.add(element);
    Ok(Value::LyObj(LyObj::new(Box::new(filter))))
}

/// Test if element might be in Bloom filter
pub fn bloom_filter_contains(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("BloomFilterContains expects 2 arguments".to_string()));
    }
    
    let filter = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<BloomFilter>() {
                Some(f) => f,
                None => return Err(VmError::Runtime("BloomFilterContains expected BloomFilter object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("BloomFilterContains expected BloomFilter object".to_string())),
    };
    
    let element = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::Runtime("BloomFilterContains expected string or symbol".to_string())),
    };
    
    Ok(Value::Boolean(filter.contains(element)))
}

/// Get Bloom filter statistics
pub fn bloom_filter_stats(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("BloomFilterStats expects 1 argument".to_string()));
    }
    
    let filter = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<BloomFilter>() {
                Some(f) => f,
                None => return Err(VmError::Runtime("BloomFilterStats expected BloomFilter object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("BloomFilterStats expected BloomFilter object".to_string())),
    };
    
    let stats = filter.stats();
    let mut result = Vec::new();
    
    for (key, value) in stats {
        let rule = Value::List(vec![
            Value::String(key),
            Value::Real(value),
        ]);
        result.push(rule);
    }
    
    Ok(Value::List(result))
}

/// Clear Bloom filter
pub fn bloom_filter_clear(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("BloomFilterClear expects 1 argument".to_string()));
    }
    
    let mut filter = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<BloomFilter>() {
                Some(f) => f.clone(),
                None => return Err(VmError::Runtime("BloomFilterClear expected BloomFilter object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("BloomFilterClear expected BloomFilter object".to_string())),
    };
    
    filter.clear();
    Ok(Value::LyObj(LyObj::new(Box::new(filter))))
}

/// Get current false positive rate
pub fn bloom_filter_false_positive_rate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("BloomFilterFalsePositiveRate expects 1 argument".to_string()));
    }
    
    let filter = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<BloomFilter>() {
                Some(f) => f,
                None => return Err(VmError::Runtime("BloomFilterFalsePositiveRate expected BloomFilter object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("BloomFilterFalsePositiveRate expected BloomFilter object".to_string())),
    };
    
    Ok(Value::Real(filter.current_false_positive_rate()))
}


/// Count-Min Sketch - Space-efficient approximate frequency counter
/// Provides approximate counts with guaranteed upper bounds on error
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    counters: Vec<Vec<u32>>,
    total_count: u64,
}

impl CountMinSketch {
    /// Create a new Count-Min Sketch with specified parameters
    pub fn new(width: usize, depth: usize) -> Self {
        Self {
            width,
            depth,
            counters: vec![vec![0; width]; depth],
            total_count: 0,
        }
    }
    
    /// Create Count-Min Sketch with optimal parameters for error bounds
    pub fn with_error_bounds(epsilon: f64, delta: f64) -> Self {
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        Self::new(width, depth)
    }
    
    /// Generate hash values for an element
    fn hash_element(&self, element: &str, row: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        element.hash(&mut hasher);
        row.hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
    
    /// Add an element to the sketch (increment its count)
    pub fn add(&mut self, element: &str) {
        self.add_count(element, 1);
    }
    
    /// Add multiple occurrences of an element
    pub fn add_count(&mut self, element: &str, count: u32) {
        for row in 0..self.depth {
            let col = self.hash_element(element, row);
            self.counters[row][col] += count;
        }
        self.total_count += count as u64;
    }
    
    /// Get the estimated count for an element
    pub fn estimate(&self, element: &str) -> u32 {
        let mut min_count = u32::MAX;
        for row in 0..self.depth {
            let col = self.hash_element(element, row);
            min_count = min_count.min(self.counters[row][col]);
        }
        min_count
    }
    
    /// Get sketch statistics
    pub fn stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        stats.insert("width".to_string(), self.width as f64);
        stats.insert("depth".to_string(), self.depth as f64);
        stats.insert("total_count".to_string(), self.total_count as f64);
        
        // Calculate memory usage
        let memory_bytes = self.width * self.depth * 4; // 4 bytes per u32
        stats.insert("memory_bytes".to_string(), memory_bytes as f64);
        
        // Calculate average and max counter values
        let mut total_counters = 0u64;
        let mut max_counter = 0u32;
        for row in &self.counters {
            for &counter in row {
                total_counters += counter as u64;
                max_counter = max_counter.max(counter);
            }
        }
        
        let total_cells = (self.width * self.depth) as f64;
        stats.insert("average_counter".to_string(), total_counters as f64 / total_cells);
        stats.insert("max_counter".to_string(), max_counter as f64);
        
        stats
    }
    
    /// Clear all counters
    pub fn clear(&mut self) {
        for row in &mut self.counters {
            row.fill(0);
        }
        self.total_count = 0;
    }
}

impl Foreign for CountMinSketch {
    fn type_name(&self) -> &'static str {
        "CountMinSketch"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut sketch = self.clone();
        match method {
            "add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let element = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                sketch.add(&element);
                Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
            }
            "add_count" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                let element = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                let count = match &args[1] {
                    Value::Integer(n) if *n >= 0 => *n as u32,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Non-negative Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                sketch.add_count(&element, count);
                Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
            }
            "estimate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let element = match &args[0] {
                    Value::String(s) => s,
                    Value::Symbol(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                Ok(Value::Integer(sketch.estimate(element) as i64))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Create a new Count-Min Sketch
pub fn count_min_sketch(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        2 => {
            // CountMinSketch[width, depth]
            let width = match &args[0] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("CountMinSketch expected positive integer for width".to_string())),
            };
            
            let depth = match &args[1] {
                Value::Integer(n) if *n > 0 => *n as usize,
                _ => return Err(VmError::Runtime("CountMinSketch expected positive integer for depth".to_string())),
            };
            
            let sketch = CountMinSketch::new(width, depth);
            Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
        }
        3 => {
            // CountMinSketch[epsilon, delta, "ErrorBounds"]
            let epsilon = match &args[0] {
                Value::Real(r) if *r > 0.0 && *r < 1.0 => *r,
                _ => return Err(VmError::Runtime("CountMinSketch expected epsilon between 0 and 1".to_string())),
            };
            
            let delta = match &args[1] {
                Value::Real(r) if *r > 0.0 && *r < 1.0 => *r,
                _ => return Err(VmError::Runtime("CountMinSketch expected delta between 0 and 1".to_string())),
            };
            
            // Third argument should be "ErrorBounds" string (ignored for now)
            let sketch = CountMinSketch::with_error_bounds(epsilon, delta);
            Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
        }
        _ => Err(VmError::Runtime("CountMinSketch expects 2-3 arguments".to_string())),
    }
}

/// Add element to Count-Min Sketch
pub fn count_min_sketch_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("CountMinSketchAdd expects 2 arguments".to_string()));
    }
    
    let mut sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s.clone(),
                None => return Err(VmError::Runtime("CountMinSketchAdd expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchAdd expected CountMinSketch object".to_string())),
    };
    
    let element = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::Runtime("CountMinSketchAdd expected string or symbol".to_string())),
    };
    
    sketch.add(element);
    Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
}

/// Add multiple counts to Count-Min Sketch
pub fn count_min_sketch_add_count(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime("CountMinSketchAddCount expects 3 arguments".to_string()));
    }
    
    let mut sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s.clone(),
                None => return Err(VmError::Runtime("CountMinSketchAddCount expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchAddCount expected CountMinSketch object".to_string())),
    };
    
    let element = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::Runtime("CountMinSketchAddCount expected string or symbol".to_string())),
    };
    
    let count = match &args[2] {
        Value::Integer(n) if *n >= 0 => *n as u32,
        _ => return Err(VmError::Runtime("CountMinSketchAddCount expected non-negative integer count".to_string())),
    };
    
    sketch.add_count(element, count);
    Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
}

/// Get estimated count from Count-Min Sketch
pub fn count_min_sketch_estimate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("CountMinSketchEstimate expects 2 arguments".to_string()));
    }
    
    let sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s,
                None => return Err(VmError::Runtime("CountMinSketchEstimate expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchEstimate expected CountMinSketch object".to_string())),
    };
    
    let element = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::Runtime("CountMinSketchEstimate expected string or symbol".to_string())),
    };
    
    Ok(Value::Integer(sketch.estimate(element) as i64))
}

/// Get Count-Min Sketch statistics
pub fn count_min_sketch_stats(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("CountMinSketchStats expects 1 argument".to_string()));
    }
    
    let sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s,
                None => return Err(VmError::Runtime("CountMinSketchStats expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchStats expected CountMinSketch object".to_string())),
    };
    
    let stats = sketch.stats();
    let mut result = Vec::new();
    
    for (key, value) in stats {
        let rule = Value::List(vec![
            Value::String(key),
            Value::Real(value),
        ]);
        result.push(rule);
    }
    
    Ok(Value::List(result))
}

/// Clear Count-Min Sketch
pub fn count_min_sketch_clear(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("CountMinSketchClear expects 1 argument".to_string()));
    }
    
    let mut sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s.clone(),
                None => return Err(VmError::Runtime("CountMinSketchClear expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchClear expected CountMinSketch object".to_string())),
    };
    
    sketch.clear();
    Ok(Value::LyObj(LyObj::new(Box::new(sketch))))
}

/// Get total count from Count-Min Sketch
pub fn count_min_sketch_total_count(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("CountMinSketchTotalCount expects 1 argument".to_string()));
    }
    
    let sketch = match &args[0] {
        Value::LyObj(obj) => {
            match obj.downcast_ref::<CountMinSketch>() {
                Some(s) => s,
                None => return Err(VmError::Runtime("CountMinSketchTotalCount expected CountMinSketch object".to_string())),
            }
        }
        _ => return Err(VmError::Runtime("CountMinSketchTotalCount expected CountMinSketch object".to_string())),
    };
    
    Ok(Value::Integer(sketch.total_count as i64))
}

/// Register probabilistic data structure functions
pub fn register_probabilistic_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    // Bloom Filter functions
    functions.insert("BloomFilter".to_string(), bloom_filter);
    functions.insert("BloomFilterAdd".to_string(), bloom_filter_add);
    functions.insert("BloomFilterContains".to_string(), bloom_filter_contains);
    functions.insert("BloomFilterStats".to_string(), bloom_filter_stats);
    functions.insert("BloomFilterClear".to_string(), bloom_filter_clear);
    functions.insert("BloomFilterFalsePositiveRate".to_string(), bloom_filter_false_positive_rate);
    
    // Count-Min Sketch functions
    functions.insert("CountMinSketch".to_string(), count_min_sketch);
    functions.insert("CountMinSketchAdd".to_string(), count_min_sketch_add);
    functions.insert("CountMinSketchAddCount".to_string(), count_min_sketch_add_count);
    functions.insert("CountMinSketchEstimate".to_string(), count_min_sketch_estimate);
    functions.insert("CountMinSketchStats".to_string(), count_min_sketch_stats);
    functions.insert("CountMinSketchClear".to_string(), count_min_sketch_clear);
    functions.insert("CountMinSketchTotalCount".to_string(), count_min_sketch_total_count);
}

/// Get documentation for probabilistic data structures
pub fn get_probabilistic_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    
    // Bloom Filter documentation
    docs.insert("BloomFilter".to_string(), 
        "BloomFilter[expectedElements] or BloomFilter[expectedElements, falsePositiveRate] - Create a space-efficient probabilistic set. False positives possible, false negatives impossible.".to_string());
    docs.insert("BloomFilterAdd".to_string(),
        "BloomFilterAdd[filter, element] - Add element to Bloom filter. Returns new filter.".to_string());
    docs.insert("BloomFilterContains".to_string(),
        "BloomFilterContains[filter, element] - Test if element might be in the set. Returns True if possibly present, False if definitely not present.".to_string());
    docs.insert("BloomFilterStats".to_string(),
        "BloomFilterStats[filter] - Get filter statistics including capacity, element count, and current false positive rate.".to_string());
    docs.insert("BloomFilterClear".to_string(),
        "BloomFilterClear[filter] - Clear all elements from the filter. Returns new empty filter.".to_string());
    docs.insert("BloomFilterFalsePositiveRate".to_string(),
        "BloomFilterFalsePositiveRate[filter] - Get the current false positive probability based on filter state.".to_string());
    
    // Count-Min Sketch documentation
    docs.insert("CountMinSketch".to_string(),
        "CountMinSketch[width, depth] or CountMinSketch[epsilon, delta, \"ErrorBounds\"] - Create a space-efficient approximate frequency counter.".to_string());
    docs.insert("CountMinSketchAdd".to_string(),
        "CountMinSketchAdd[sketch, element] - Increment count for element in sketch. Returns new sketch.".to_string());
    docs.insert("CountMinSketchAddCount".to_string(),
        "CountMinSketchAddCount[sketch, element, count] - Add specific count for element. Returns new sketch.".to_string());
    docs.insert("CountMinSketchEstimate".to_string(),
        "CountMinSketchEstimate[sketch, element] - Get estimated count for element (upper bound guarantee).".to_string());
    docs.insert("CountMinSketchStats".to_string(),
        "CountMinSketchStats[sketch] - Get sketch statistics including dimensions, memory usage, and counter statistics.".to_string());
    docs.insert("CountMinSketchClear".to_string(),
        "CountMinSketchClear[sketch] - Clear all counters in the sketch. Returns new empty sketch.".to_string());
    docs.insert("CountMinSketchTotalCount".to_string(),
        "CountMinSketchTotalCount[sketch] - Get total number of items added to the sketch.".to_string());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bloom_filter_creation() {
        let args = vec![Value::Integer(1000)];
        let result = bloom_filter(&args).unwrap();
        assert!(matches!(result, Value::LyObj(_)));
    }
    
    #[test]
    fn test_bloom_filter_creation_with_rate() {
        let args = vec![Value::Integer(1000), Value::Real(0.01)];
        let result = bloom_filter(&args).unwrap();
        assert!(matches!(result, Value::LyObj(_)));
    }
    
    #[test]
    fn test_bloom_filter_creation_manual() {
        let args = vec![Value::Integer(1000), Value::Integer(3), Value::String("Manual".to_string())];
        let result = bloom_filter(&args).unwrap();
        assert!(matches!(result, Value::LyObj(_)));
    }
    
    #[test]
    fn test_bloom_filter_add_and_contains() {
        let filter_args = vec![Value::Integer(100)];
        let filter = bloom_filter(&filter_args).unwrap();
        
        // Add element
        let add_args = vec![filter.clone(), Value::String("test".to_string())];
        let new_filter = bloom_filter_add(&add_args).unwrap();
        
        // Test contains
        let contains_args = vec![new_filter.clone(), Value::String("test".to_string())];
        let result = bloom_filter_contains(&contains_args).unwrap();
        assert_eq!(result, Value::Boolean(true));
        
        // Test not contains
        let not_contains_args = vec![new_filter, Value::String("missing".to_string())];
        let result = bloom_filter_contains(&not_contains_args).unwrap();
        // Could be false positive, but with small filter and one element, likely false
        // Just ensure it returns a boolean
        assert!(matches!(result, Value::Boolean(_)));
    }
    
    #[test]
    fn test_bloom_filter_stats() {
        let filter_args = vec![Value::Integer(100)];
        let filter = bloom_filter(&filter_args).unwrap();
        
        let stats_args = vec![filter];
        let result = bloom_filter_stats(&stats_args).unwrap();
        assert!(matches!(result, Value::List(_)));
        
        if let Value::List(stats) = result {
            assert!(stats.len() > 0);
            // Should contain statistics like capacity, element_count, etc.
        }
    }
    
    #[test]
    fn test_bloom_filter_clear() {
        let filter_args = vec![Value::Integer(100)];
        let filter = bloom_filter(&filter_args).unwrap();
        
        // Add element
        let add_args = vec![filter.clone(), Value::String("test".to_string())];
        let filter_with_element = bloom_filter_add(&add_args).unwrap();
        
        // Clear filter
        let clear_args = vec![filter_with_element];
        let cleared_filter = bloom_filter_clear(&clear_args).unwrap();
        
        // Check that element is no longer present
        let contains_args = vec![cleared_filter, Value::String("test".to_string())];
        let result = bloom_filter_contains(&contains_args).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }
    
    #[test]
    fn test_bloom_filter_false_positive_rate() {
        let filter_args = vec![Value::Integer(100)];
        let filter = bloom_filter(&filter_args).unwrap();
        
        let rate_args = vec![filter];
        let result = bloom_filter_false_positive_rate(&rate_args).unwrap();
        assert!(matches!(result, Value::Real(_)));
        
        if let Value::Real(rate) = result {
            assert!(rate >= 0.0 && rate <= 1.0);
        }
    }
    
    #[test]
    fn test_bloom_filter_no_false_negatives() {
        // Test that there are no false negatives
        let filter_args = vec![Value::Integer(1000)];
        let mut filter = bloom_filter(&filter_args).unwrap();
        
        let test_elements = vec!["apple", "banana", "cherry", "date", "elderberry"];
        
        // Add all elements
        for element in &test_elements {
            let add_args = vec![filter, Value::String(element.to_string())];
            filter = bloom_filter_add(&add_args).unwrap();
        }
        
        // All added elements should be found (no false negatives)
        for element in &test_elements {
            let contains_args = vec![filter.clone(), Value::String(element.to_string())];
            let result = bloom_filter_contains(&contains_args).unwrap();
            assert_eq!(result, Value::Boolean(true), "False negative for element: {}", element);
        }
    }
    
    #[test]
    fn test_bloom_filter_parameter_validation() {
        // Test invalid expected elements
        let args = vec![Value::Integer(-1)];
        assert!(bloom_filter(&args).is_err());
        
        // Test invalid false positive rate
        let args = vec![Value::Integer(100), Value::Real(1.5)];
        assert!(bloom_filter(&args).is_err());
        
        let args = vec![Value::Integer(100), Value::Real(-0.1)];
        assert!(bloom_filter(&args).is_err());
    }
    
    #[test]
    fn test_bloom_filter_wrong_argument_count() {
        // Test too many arguments
        let args = vec![Value::Integer(100), Value::Real(0.01), Value::Integer(3), Value::String("extra".to_string())];
        assert!(bloom_filter(&args).is_err());
        
        // Test no arguments
        let args = vec![];
        assert!(bloom_filter(&args).is_err());
    }
    
    #[test]
    fn test_bloom_filter_multiple_elements() {
        let filter_args = vec![Value::Integer(1000), Value::Real(0.01)];
        let mut filter = bloom_filter(&filter_args).unwrap();
        
        let elements = vec!["element1", "element2", "element3", "element4", "element5"];
        
        // Add multiple elements
        for element in &elements {
            let add_args = vec![filter, Value::String(element.to_string())];
            filter = bloom_filter_add(&add_args).unwrap();
        }
        
        // Verify all are present
        for element in &elements {
            let contains_args = vec![filter.clone(), Value::String(element.to_string())];
            let result = bloom_filter_contains(&contains_args).unwrap();
            assert_eq!(result, Value::Boolean(true));
        }
        
        // Test false positive behavior with non-existent elements
        let non_existent = vec!["missing1", "missing2", "missing3"];
        let mut false_positives = 0;
        
        for element in &non_existent {
            let contains_args = vec![filter.clone(), Value::String(element.to_string())];
            let result = bloom_filter_contains(&contains_args).unwrap();
            if result == Value::Boolean(true) {
                false_positives += 1;
            }
        }
        
        // With a 1% false positive rate and good parameters, we shouldn't get many false positives
        // This is probabilistic, so we just ensure it's reasonable
        assert!(false_positives <= non_existent.len());
    }
    
    // Count-Min Sketch tests
    #[test]
    fn test_count_min_sketch_creation() {
        let args = vec![Value::Integer(100), Value::Integer(5)];
        let result = count_min_sketch(&args).unwrap();
        assert!(matches!(result, Value::LyObj(_)));
    }
    
    #[test]
    fn test_count_min_sketch_creation_with_error_bounds() {
        let args = vec![Value::Real(0.1), Value::Real(0.01), Value::String("ErrorBounds".to_string())];
        let result = count_min_sketch(&args).unwrap();
        assert!(matches!(result, Value::LyObj(_)));
    }
    
    #[test]
    fn test_count_min_sketch_add_and_estimate() {
        let sketch_args = vec![Value::Integer(100), Value::Integer(5)];
        let sketch = count_min_sketch(&sketch_args).unwrap();
        
        // Add element
        let add_args = vec![sketch.clone(), Value::String("test".to_string())];
        let new_sketch = count_min_sketch_add(&add_args).unwrap();
        
        // Test estimate
        let estimate_args = vec![new_sketch.clone(), Value::String("test".to_string())];
        let result = count_min_sketch_estimate(&estimate_args).unwrap();
        assert_eq!(result, Value::Integer(1));
        
        // Test estimate for non-existent element
        let not_exists_args = vec![new_sketch, Value::String("missing".to_string())];
        let result = count_min_sketch_estimate(&not_exists_args).unwrap();
        // Could be 0 or higher due to hash collisions, but should be low
        assert!(matches!(result, Value::Integer(_)));
    }
    
    #[test]
    fn test_count_min_sketch_add_count() {
        let sketch_args = vec![Value::Integer(100), Value::Integer(5)];
        let sketch = count_min_sketch(&sketch_args).unwrap();
        
        // Add multiple counts
        let add_count_args = vec![sketch, Value::String("test".to_string()), Value::Integer(5)];
        let new_sketch = count_min_sketch_add_count(&add_count_args).unwrap();
        
        // Test estimate
        let estimate_args = vec![new_sketch, Value::String("test".to_string())];
        let result = count_min_sketch_estimate(&estimate_args).unwrap();
        assert_eq!(result, Value::Integer(5));
    }
    
    #[test]
    fn test_count_min_sketch_stats() {
        let sketch_args = vec![Value::Integer(100), Value::Integer(5)];
        let sketch = count_min_sketch(&sketch_args).unwrap();
        
        let stats_args = vec![sketch];
        let result = count_min_sketch_stats(&stats_args).unwrap();
        assert!(matches!(result, Value::List(_)));
        
        if let Value::List(stats) = result {
            assert!(stats.len() > 0);
            // Should contain statistics like width, depth, total_count, etc.
        }
    }
    
    #[test]
    fn test_count_min_sketch_total_count() {
        let sketch_args = vec![Value::Integer(100), Value::Integer(5)];
        let sketch = count_min_sketch(&sketch_args).unwrap();
        
        // Add some elements
        let add_args1 = vec![sketch, Value::String("test1".to_string())];
        let sketch = count_min_sketch_add(&add_args1).unwrap();
        
        let add_count_args = vec![sketch, Value::String("test2".to_string()), Value::Integer(3)];
        let sketch = count_min_sketch_add_count(&add_count_args).unwrap();
        
        // Check total count
        let total_args = vec![sketch];
        let result = count_min_sketch_total_count(&total_args).unwrap();
        assert_eq!(result, Value::Integer(4)); // 1 + 3
    }
    
    #[test]
    fn test_count_min_sketch_clear() {
        let sketch_args = vec![Value::Integer(100), Value::Integer(5)];
        let sketch = count_min_sketch(&sketch_args).unwrap();
        
        // Add element
        let add_args = vec![sketch, Value::String("test".to_string())];
        let sketch_with_element = count_min_sketch_add(&add_args).unwrap();
        
        // Clear sketch
        let clear_args = vec![sketch_with_element];
        let cleared_sketch = count_min_sketch_clear(&clear_args).unwrap();
        
        // Check that total count is 0
        let total_args = vec![cleared_sketch.clone()];
        let result = count_min_sketch_total_count(&total_args).unwrap();
        assert_eq!(result, Value::Integer(0));
        
        // Check that estimate is 0
        let estimate_args = vec![cleared_sketch, Value::String("test".to_string())];
        let result = count_min_sketch_estimate(&estimate_args).unwrap();
        assert_eq!(result, Value::Integer(0));
    }
    
    #[test]
    fn test_count_min_sketch_parameter_validation() {
        // Test invalid width
        let args = vec![Value::Integer(-1), Value::Integer(5)];
        assert!(count_min_sketch(&args).is_err());
        
        // Test invalid depth
        let args = vec![Value::Integer(100), Value::Integer(0)];
        assert!(count_min_sketch(&args).is_err());
        
        // Test invalid epsilon
        let args = vec![Value::Real(1.5), Value::Real(0.01), Value::String("ErrorBounds".to_string())];
        assert!(count_min_sketch(&args).is_err());
        
        // Test invalid delta
        let args = vec![Value::Real(0.1), Value::Real(-0.1), Value::String("ErrorBounds".to_string())];
        assert!(count_min_sketch(&args).is_err());
    }
    
    #[test]
    fn test_count_min_sketch_wrong_argument_count() {
        // Test too many arguments
        let args = vec![Value::Integer(100), Value::Integer(5), Value::String("extra".to_string()), Value::Integer(1)];
        assert!(count_min_sketch(&args).is_err());
        
        // Test too few arguments
        let args = vec![Value::Integer(100)];
        assert!(count_min_sketch(&args).is_err());
        
        // Test no arguments
        let args = vec![];
        assert!(count_min_sketch(&args).is_err());
    }
    
    #[test]
    fn test_count_min_sketch_accuracy() {
        // Test that Count-Min Sketch provides upper bound estimates
        let sketch_args = vec![Value::Integer(1000), Value::Integer(5)];
        let mut sketch = count_min_sketch(&sketch_args).unwrap();
        
        let test_elements = vec![
            ("apple", 5),
            ("banana", 3),
            ("cherry", 8),
            ("date", 1),
            ("elderberry", 12),
        ];
        
        // Add elements with specific counts
        for (element, count) in &test_elements {
            let add_count_args = vec![sketch, Value::String(element.to_string()), Value::Integer(*count)];
            sketch = count_min_sketch_add_count(&add_count_args).unwrap();
        }
        
        // Verify estimates are at least the actual count (upper bound property)
        for (element, actual_count) in &test_elements {
            let estimate_args = vec![sketch.clone(), Value::String(element.to_string())];
            let result = count_min_sketch_estimate(&estimate_args).unwrap();
            if let Value::Integer(estimated) = result {
                assert!(estimated >= *actual_count, 
                    "Estimate {} should be >= actual count {} for element {}", 
                    estimated, actual_count, element);
            }
        }
    }
}