//! Search algorithms for Lyra standard library
//!
//! This module implements core searching algorithms with support for
//! different data types and custom comparison functions.
//!
//! ## Available Algorithms
//!
//! - **BinarySearch**: O(log n) search in sorted arrays
//! - **LinearSearch**: O(n) search in any array
//! - **InterpolationSearch**: O(log log n) for uniformly distributed data
//! - **BinarySearchFirst**: Find first occurrence in sorted array
//! - **BinarySearchLast**: Find last occurrence in sorted array

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::cmp::Ordering;

/// Default comparison function for Value types
fn default_compare(a: &Value, b: &Value) -> Ordering {
    match (a, b) {
        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
        (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Symbol(a), Value::Symbol(b)) => a.cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        _ => Ordering::Equal,
    }
}

/// Extract list from Value, returning error for non-lists
fn extract_list(value: &Value) -> Result<&Vec<Value>, VmError> {
    match value {
        Value::List(list) => Ok(list),
        _ => Err(VmError::Runtime("Expected list".to_string())),
    }
}

/// Binary search implementation - finds target in sorted list
pub fn binary_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let list = extract_list(&args[0])?;
    let target = &args[1];
    
    if list.is_empty() {
        return Ok(Value::Integer(-1));
    }
    
    let mut left = 0;
    let mut right = list.len();
    
    while left < right {
        let mid = left + (right - left) / 2;
        
        match default_compare(&list[mid], target) {
            Ordering::Less => left = mid + 1,
            Ordering::Greater => right = mid,
            Ordering::Equal => return Ok(Value::Integer(mid as i64 + 1)), // 1-indexed
        }
    }
    
    Ok(Value::Integer(-1)) // Not found
}

/// Binary search first occurrence - finds first occurrence of target
pub fn binary_search_first(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let list = extract_list(&args[0])?;
    let target = &args[1];
    
    if list.is_empty() {
        return Ok(Value::Integer(-1));
    }
    
    let mut left = 0;
    let mut right = list.len();
    let mut result = -1;
    
    while left < right {
        let mid = left + (right - left) / 2;
        
        match default_compare(&list[mid], target) {
            Ordering::Less => left = mid + 1,
            Ordering::Greater => right = mid,
            Ordering::Equal => {
                result = mid as i64 + 1; // 1-indexed
                right = mid; // Continue searching left half
            }
        }
    }
    
    Ok(Value::Integer(result))
}

/// Binary search last occurrence - finds last occurrence of target
pub fn binary_search_last(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let list = extract_list(&args[0])?;
    let target = &args[1];
    
    if list.is_empty() {
        return Ok(Value::Integer(-1));
    }
    
    let mut left = 0;
    let mut right = list.len();
    let mut result = -1;
    
    while left < right {
        let mid = left + (right - left) / 2;
        
        match default_compare(&list[mid], target) {
            Ordering::Less => left = mid + 1,
            Ordering::Greater => right = mid,
            Ordering::Equal => {
                result = mid as i64 + 1; // 1-indexed
                left = mid + 1; // Continue searching right half
            }
        }
    }
    
    Ok(Value::Integer(result))
}

/// Linear search implementation - searches through unsorted list
pub fn linear_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let list = extract_list(&args[0])?;
    let target = &args[1];
    
    for (i, item) in list.iter().enumerate() {
        if default_compare(item, target) == Ordering::Equal {
            return Ok(Value::Integer(i as i64 + 1)); // 1-indexed
        }
    }
    
    Ok(Value::Integer(-1)) // Not found
}

/// Interpolation search - O(log log n) for uniformly distributed numeric data
pub fn interpolation_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let list = extract_list(&args[0])?;
    let target = &args[1];
    
    if list.is_empty() {
        return Ok(Value::Integer(-1));
    }
    
    // Only works with numeric data
    let target_val = match target {
        Value::Integer(n) => *n as f64,
        Value::Real(r) => *r,
        _ => return linear_search(args), // Fallback to linear search
    };
    
    let mut left = 0;
    let mut right = list.len() - 1;
    
    while left <= right && right < list.len() {
        // Get numeric values for interpolation
        let left_val = match &list[left] {
            Value::Integer(n) => *n as f64,
            Value::Real(r) => *r,
            _ => return linear_search(args), // Fallback
        };
        
        let right_val = match &list[right] {
            Value::Integer(n) => *n as f64,
            Value::Real(r) => *r,
            _ => return linear_search(args), // Fallback
        };
        
        if left_val == right_val {
            if left_val == target_val {
                return Ok(Value::Integer(left as i64 + 1));
            }
            break;
        }
        
        // Interpolate position
        let pos = left + ((target_val - left_val) / (right_val - left_val) * (right - left) as f64) as usize;
        let pos = pos.min(right).max(left);
        
        let pos_val = match &list[pos] {
            Value::Integer(n) => *n as f64,
            Value::Real(r) => *r,
            _ => return linear_search(args), // Fallback
        };
        
        if pos_val == target_val {
            return Ok(Value::Integer(pos as i64 + 1)); // 1-indexed
        } else if pos_val < target_val {
            left = pos + 1;
        } else {
            if pos == 0 { break; }
            right = pos - 1;
        }
    }
    
    Ok(Value::Integer(-1)) // Not found
}

/// Register searching functions
pub fn register_searching_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("BinarySearch".to_string(), binary_search);
    functions.insert("BinarySearchFirst".to_string(), binary_search_first);
    functions.insert("BinarySearchLast".to_string(), binary_search_last);
    functions.insert("LinearSearch".to_string(), linear_search);
    functions.insert("InterpolationSearch".to_string(), interpolation_search);
}

/// Get documentation for searching functions
pub fn get_searching_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("BinarySearch".to_string(), "BinarySearch[list, target] - Search for target in sorted list using binary search. Returns 1-based index or -1 if not found.".to_string());
    docs.insert("BinarySearchFirst".to_string(), "BinarySearchFirst[list, target] - Find first occurrence of target in sorted list with duplicates.".to_string());
    docs.insert("BinarySearchLast".to_string(), "BinarySearchLast[list, target] - Find last occurrence of target in sorted list with duplicates.".to_string());
    docs.insert("LinearSearch".to_string(), "LinearSearch[list, target] - Search for target in unsorted list using linear search.".to_string());
    docs.insert("InterpolationSearch".to_string(), "InterpolationSearch[list, target] - Fast search for numeric data in uniformly distributed sorted list.".to_string());
    docs
}