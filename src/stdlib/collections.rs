//! Collections module providing essential data structures for Lyra
//!
//! This module implements fundamental collection types as stdlib functions:
//! - Set operations using Value::List with uniqueness guarantees
//! - Dictionary/HashMap operations using Foreign objects
//! - Queue operations (FIFO) using Foreign objects  
//! - Stack operations (LIFO) using Foreign objects
//!
//! All stateful collections (Dictionary, Queue, Stack) are implemented as Foreign objects
//! to maintain separation from VM core while providing efficient operations.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::collections::{HashMap, VecDeque, HashSet};
use std::any::Any;
use std::sync::{Arc, Mutex};

// =============================================================================
// Set Operations (using Value::List with uniqueness)
// =============================================================================

/// Create a set from a list by removing duplicates
pub fn set_create(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "SetCreate expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(items) => {
            let mut unique_items = Vec::new();
            let mut seen = HashSet::new();
            
            for item in items {
                if seen.insert(item.clone()) {
                    unique_items.push(item.clone());
                }
            }
            
            Ok(Value::List(unique_items))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Union of two sets
pub fn set_union(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetUnion expects 2 arguments, got {}", args.len()
        )));
    }

    match (&args[0], &args[1]) {
        (Value::List(set1), Value::List(set2)) => {
            let mut union_set = HashSet::new();
            
            // Add all items from both sets
            for item in set1.iter().chain(set2.iter()) {
                union_set.insert(item.clone());
            }
            
            Ok(Value::List(union_set.into_iter().collect()))
        }
        _ => Err(VmError::TypeError {
            expected: "List, List".to_string(),
            actual: format!("{:?}, {:?}", args[0], args[1]),
        }),
    }
}

/// Intersection of two sets
pub fn set_intersection(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetIntersection expects 2 arguments, got {}", args.len()
        )));
    }

    match (&args[0], &args[1]) {
        (Value::List(set1), Value::List(set2)) => {
            let set1_hash: HashSet<_> = set1.iter().cloned().collect();
            let set2_hash: HashSet<_> = set2.iter().cloned().collect();
            
            let intersection: Vec<Value> = set1_hash.intersection(&set2_hash).cloned().collect();
            
            Ok(Value::List(intersection))
        }
        _ => Err(VmError::TypeError {
            expected: "List, List".to_string(),
            actual: format!("{:?}, {:?}", args[0], args[1]),
        }),
    }
}

/// Difference of two sets (set1 - set2)
pub fn set_difference(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetDifference expects 2 arguments, got {}", args.len()
        )));
    }

    match (&args[0], &args[1]) {
        (Value::List(set1), Value::List(set2)) => {
            let set2_hash: HashSet<_> = set2.iter().cloned().collect();
            
            let difference: Vec<Value> = set1.iter()
                .filter(|item| !set2_hash.contains(item))
                .cloned()
                .collect();
            
            Ok(Value::List(difference))
        }
        _ => Err(VmError::TypeError {
            expected: "List, List".to_string(),
            actual: format!("{:?}, {:?}", args[0], args[1]),
        }),
    }
}

/// Check if set contains element
pub fn set_contains(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetContains expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(set) => {
            let contains = set.contains(&args[1]);
            Ok(Value::Boolean(contains))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Add element to set
pub fn set_add(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetAdd expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(set) => {
            let mut new_set = set.clone();
            if !new_set.contains(&args[1]) {
                new_set.push(args[1].clone());
            }
            Ok(Value::List(new_set))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Remove element from set
pub fn set_remove(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "SetRemove expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(set) => {
            let new_set: Vec<Value> = set.iter()
                .filter(|item| *item != &args[1])
                .cloned()
                .collect();
            Ok(Value::List(new_set))
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get set size
pub fn set_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "SetSize expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(set) => Ok(Value::Integer(set.len() as i64)),
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

// =============================================================================
// Dictionary Foreign Object Implementation
// =============================================================================

/// Dictionary Foreign object implementation using HashMap
#[derive(Debug, Clone)]
pub struct Dictionary {
    data: Arc<Mutex<HashMap<String, Value>>>,
}

impl Dictionary {
    fn new() -> Self {
        Dictionary {
            data: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn from_pairs(pairs: &[Value]) -> Result<Self, ForeignError> {
        let mut dict = Dictionary::new();
        
        for pair in pairs {
            match pair {
                Value::List(pair_items) if pair_items.len() == 2 => {
                    let key = match &pair_items[0] {
                        Value::String(s) => s.clone(),
                        Value::Symbol(s) => s.clone(),
                        v => return Err(ForeignError::TypeError {
                            expected: "String or Symbol".to_string(),
                            actual: format!("{:?}", v),
                        }),
                    };
                    
                    let value = pair_items[1].clone();
                    dict.data.lock().unwrap().insert(key, value);
                }
                _ => return Err(ForeignError::InvalidArgument(
                    "Dictionary pairs must be [key, value] lists".to_string()
                )),
            }
        }
        
        Ok(dict)
    }
}

impl Foreign for Dictionary {
    fn type_name(&self) -> &'static str {
        "Dictionary"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "get" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    v => return Err(ForeignError::TypeError {
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", v),
                    }),
                };
                
                let data = self.data.lock().unwrap();
                match data.get(&key) {
                    Some(value) => Ok(value.clone()),
                    None => {
                        if args.len() == 2 {
                            Ok(args[1].clone()) // Default value
                        } else {
                            Ok(Value::Missing) // No default
                        }
                    }
                }
            }
            "set" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    v => return Err(ForeignError::TypeError {
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", v),
                    }),
                };
                
                let value = args[1].clone();
                self.data.lock().unwrap().insert(key, value);
                Ok(Value::Missing) // Void return
            }
            "delete" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    v => return Err(ForeignError::TypeError {
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", v),
                    }),
                };
                
                let removed = self.data.lock().unwrap().remove(&key);
                Ok(if removed.is_some() { Value::Boolean(true) } else { Value::Boolean(false) })
            }
            "keys" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                let data = self.data.lock().unwrap();
                let keys: Vec<Value> = data.keys()
                    .map(|k| Value::String(k.clone()))
                    .collect();
                Ok(Value::List(keys))
            }
            "values" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                let data = self.data.lock().unwrap();
                let values: Vec<Value> = data.values().cloned().collect();
                Ok(Value::List(values))
            }
            "contains" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    v => return Err(ForeignError::TypeError {
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", v),
                    }),
                };
                
                let data = self.data.lock().unwrap();
                Ok(Value::Boolean(data.contains_key(&key)))
            }
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                let data = self.data.lock().unwrap();
                Ok(Value::Integer(data.len() as i64))
            }
            "merge" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::LyObj(other_obj) => {
                        if let Some(other_dict) = other_obj.downcast_ref::<Dictionary>() {
                            let new_dict = self.clone();
                            {
                                let other_data = other_dict.data.lock().unwrap();
                                let mut new_data = new_dict.data.lock().unwrap();
                                
                                for (key, value) in other_data.iter() {
                                    new_data.insert(key.clone(), value.clone());
                                }
                            } // Release locks before creating LyObj
                            
                            Ok(Value::LyObj(LyObj::new(Box::new(new_dict))))
                        } else {
                            Err(ForeignError::TypeError {
                                expected: "Dictionary".to_string(),
                                actual: other_obj.type_name().to_string(),
                            })
                        }
                    }
                    v => Err(ForeignError::TypeError {
                        expected: "Dictionary".to_string(),
                        actual: format!("{:?}", v),
                    }),
                }
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

// =============================================================================
// Dictionary Functions
// =============================================================================

/// Create dictionary from key-value pairs
pub fn dict_create(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "DictCreate expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::List(pairs) => {
            match Dictionary::from_pairs(pairs) {
                Ok(dict) => Ok(Value::LyObj(LyObj::new(Box::new(dict)))),
                Err(err) => Err(VmError::Runtime(format!("Dict creation failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get value by key with optional default
pub fn dict_get(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::Runtime(format!(
            "DictGet expects 2-3 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            let method_args = if args.len() == 3 {
                &args[1..3] // Key and default
            } else {
                &args[1..2] // Just key
            };
            
            match obj.call_method("get", method_args) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict get failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Set key-value pair
pub fn dict_set(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime(format!(
            "DictSet expects 3 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("set", &args[1..3]) {
                Ok(_) => Ok(args[0].clone()), // Return the dictionary
                Err(err) => Err(VmError::Runtime(format!("Dict set failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Remove key-value pair
pub fn dict_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "DictDelete expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("delete", &args[1..2]) {
                Ok(_) => Ok(args[0].clone()), // Return the dictionary
                Err(err) => Err(VmError::Runtime(format!("Dict delete failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get all keys as list
pub fn dict_keys(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "DictKeys expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("keys", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict keys failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get all values as list
pub fn dict_values(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "DictValues expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("values", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict values failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Check if key exists
pub fn dict_contains(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "DictContains expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("contains", &args[1..2]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict contains failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Merge two dictionaries
pub fn dict_merge(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "DictMerge expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("merge", &args[1..2]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict merge failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get number of key-value pairs
pub fn dict_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "DictSize expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("size", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Dict size failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Dictionary".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

// =============================================================================
// Queue Foreign Object Implementation (FIFO)
// =============================================================================

/// Queue Foreign object implementation using VecDeque
#[derive(Debug, Clone)]
pub struct Queue {
    data: Arc<Mutex<VecDeque<Value>>>,
}

impl Queue {
    fn new() -> Self {
        Queue {
            data: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
}

impl Foreign for Queue {
    fn type_name(&self) -> &'static str {
        "Queue"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "enqueue" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                self.data.lock().unwrap().push_back(args[0].clone());
                Ok(Value::Missing) // Void return
            }
            "dequeue" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.data.lock().unwrap().pop_front() {
                    Some(value) => Ok(value),
                    None => Ok(Value::Missing), // Empty queue
                }
            }
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                Ok(Value::Integer(self.data.lock().unwrap().len() as i64))
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

// =============================================================================
// Queue Functions
// =============================================================================

/// Create empty queue
pub fn queue_create(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime(format!(
            "QueueCreate expects 0 arguments, got {}", args.len()
        )));
    }

    let queue = Queue::new();
    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
}

/// Add element to back of queue
pub fn queue_enqueue(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "QueueEnqueue expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("enqueue", &args[1..2]) {
                Ok(_) => Ok(args[0].clone()), // Return the queue
                Err(err) => Err(VmError::Runtime(format!("Queue enqueue failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Queue".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Remove and return front element
pub fn queue_dequeue(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "QueueDequeue expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("dequeue", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Queue dequeue failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Queue".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get queue size
pub fn queue_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "QueueSize expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("size", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Queue size failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Queue".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

// =============================================================================
// Stack Foreign Object Implementation (LIFO)
// =============================================================================

/// Stack Foreign object implementation using Vec
#[derive(Debug, Clone)]
pub struct Stack {
    data: Arc<Mutex<Vec<Value>>>,
}

impl Stack {
    fn new() -> Self {
        Stack {
            data: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl Foreign for Stack {
    fn type_name(&self) -> &'static str {
        "Stack"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "push" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                self.data.lock().unwrap().push(args[0].clone());
                Ok(Value::Missing) // Void return
            }
            "pop" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.data.lock().unwrap().pop() {
                    Some(value) => Ok(value),
                    None => Ok(Value::Missing), // Empty stack
                }
            }
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                Ok(Value::Integer(self.data.lock().unwrap().len() as i64))
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

// =============================================================================
// Stack Functions
// =============================================================================

/// Create empty stack
pub fn stack_create(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::Runtime(format!(
            "StackCreate expects 0 arguments, got {}", args.len()
        )));
    }

    let stack = Stack::new();
    Ok(Value::LyObj(LyObj::new(Box::new(stack))))
}

/// Push element onto stack
pub fn stack_push(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!(
            "StackPush expects 2 arguments, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("push", &args[1..2]) {
                Ok(_) => Ok(args[0].clone()), // Return the stack
                Err(err) => Err(VmError::Runtime(format!("Stack push failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Stack".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Pop and return top element
pub fn stack_pop(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "StackPop expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("pop", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Stack pop failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Stack".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Get stack size
pub fn stack_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!(
            "StackSize expects 1 argument, got {}", args.len()
        )));
    }

    match &args[0] {
        Value::LyObj(obj) => {
            match obj.call_method("size", &[]) {
                Ok(value) => Ok(value),
                Err(err) => Err(VmError::Runtime(format!("Stack size failed: {}", err))),
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Stack".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_create_removes_duplicates() {
        let input = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(2),
            Value::Integer(3),
            Value::Integer(1),
        ]);
        
        let result = set_create(&[input]).unwrap();
        
        if let Value::List(items) = result {
            assert_eq!(items.len(), 3);
            assert!(items.contains(&Value::Integer(1)));
            assert!(items.contains(&Value::Integer(2)));
            assert!(items.contains(&Value::Integer(3)));
        } else {
            panic!("Expected List value for set");
        }
    }

    #[test]
    fn test_set_operations() {
        let set1 = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        let set2 = Value::List(vec![Value::Integer(2), Value::Integer(3)]);
        
        // Test union
        let union_result = set_union(&[set1.clone(), set2.clone()]).unwrap();
        if let Value::List(items) = union_result {
            assert_eq!(items.len(), 3);
        }
        
        // Test intersection
        let intersection_result = set_intersection(&[set1.clone(), set2.clone()]).unwrap();
        if let Value::List(items) = intersection_result {
            assert_eq!(items.len(), 1);
            assert!(items.contains(&Value::Integer(2)));
        }
        
        // Test difference
        let diff_result = set_difference(&[set1.clone(), set2.clone()]).unwrap();
        if let Value::List(items) = diff_result {
            assert_eq!(items.len(), 1);
            assert!(items.contains(&Value::Integer(1)));
        }
        
        // Test contains
        let contains_result = set_contains(&[set1.clone(), Value::Integer(1)]).unwrap();
        assert_eq!(contains_result, Value::Boolean(true));
        
        let not_contains_result = set_contains(&[set1.clone(), Value::Integer(4)]).unwrap();
        assert_eq!(not_contains_result, Value::Boolean(false));
    }

    #[test]
    fn test_dictionary_operations() {
        // Create dictionary
        let pairs = Value::List(vec![
            Value::List(vec![Value::String("key1".to_string()), Value::Integer(10)]),
            Value::List(vec![Value::String("key2".to_string()), Value::Integer(20)]),
        ]);
        
        let dict = dict_create(&[pairs]).unwrap();
        
        // Test get
        let get_result = dict_get(&[dict.clone(), Value::String("key1".to_string())]).unwrap();
        assert_eq!(get_result, Value::Integer(10));
        
        // Test get with default
        let get_default = dict_get(&[
            dict.clone(),
            Value::String("missing".to_string()),
            Value::Integer(-1)
        ]).unwrap();
        assert_eq!(get_default, Value::Integer(-1));
        
        // Test contains
        let contains_result = dict_contains(&[dict.clone(), Value::String("key1".to_string())]).unwrap();
        assert_eq!(contains_result, Value::Boolean(true));
        
        // Test size
        let size_result = dict_size(&[dict.clone()]).unwrap();
        assert_eq!(size_result, Value::Integer(2));
        
        // Test keys
        let keys_result = dict_keys(&[dict.clone()]).unwrap();
        if let Value::List(keys) = keys_result {
            assert_eq!(keys.len(), 2);
        }
        
        // Test values
        let values_result = dict_values(&[dict]).unwrap();
        if let Value::List(values) = values_result {
            assert_eq!(values.len(), 2);
        }
    }

    #[test]
    fn test_queue_operations() {
        // Create queue
        let queue = queue_create(&[]).unwrap();
        
        // Test enqueue
        queue_enqueue(&[queue.clone(), Value::Integer(10)]).unwrap();
        queue_enqueue(&[queue.clone(), Value::Integer(20)]).unwrap();
        
        // Test size
        let size_result = queue_size(&[queue.clone()]).unwrap();
        assert_eq!(size_result, Value::Integer(2));
        
        // Test dequeue (FIFO order)
        let first = queue_dequeue(&[queue.clone()]).unwrap();
        assert_eq!(first, Value::Integer(10));
        
        let second = queue_dequeue(&[queue.clone()]).unwrap();
        assert_eq!(second, Value::Integer(20));
        
        // Test empty queue
        let empty = queue_dequeue(&[queue]).unwrap();
        assert_eq!(empty, Value::Missing);
    }

    #[test]
    fn test_stack_operations() {
        // Create stack
        let stack = stack_create(&[]).unwrap();
        
        // Test push
        stack_push(&[stack.clone(), Value::Integer(10)]).unwrap();
        stack_push(&[stack.clone(), Value::Integer(20)]).unwrap();
        
        // Test size
        let size_result = stack_size(&[stack.clone()]).unwrap();
        assert_eq!(size_result, Value::Integer(2));
        
        // Test pop (LIFO order)
        let first = stack_pop(&[stack.clone()]).unwrap();
        assert_eq!(first, Value::Integer(20)); // Last in, first out
        
        let second = stack_pop(&[stack.clone()]).unwrap();
        assert_eq!(second, Value::Integer(10));
        
        // Test empty stack
        let empty = stack_pop(&[stack]).unwrap();
        assert_eq!(empty, Value::Missing);
    }

    #[test]
    fn test_error_handling() {
        // Test invalid argument count
        let result = set_create(&[]);
        assert!(result.is_err());
        
        // Test invalid argument type
        let result = set_create(&[Value::Integer(42)]);
        assert!(result.is_err());
        
        // Test invalid dictionary operation
        let result = dict_get(&[Value::Integer(42), Value::String("key".to_string())]);
        assert!(result.is_err());
    }
}