//! Sorting algorithms for Lyra standard library
//!
//! This module implements comprehensive sorting algorithms following
//! Wolfram Language conventions. All sorting algorithms work with
//! heterogeneous Value types and support custom comparison functions.
//!
//! ## Available Algorithms
//!
//! - **Sort**: Default stable sort (TimSort)
//! - **QuickSort**: O(n log n) average, O(n²) worst case
//! - **MergeSort**: O(n log n) guaranteed, stable
//! - **HeapSort**: O(n log n) guaranteed, in-place
//! - **InsertionSort**: O(n²) but efficient for small arrays
//! - **SelectionSort**: O(n²) simple algorithm
//! - **RadixSort**: O(d×n) for integers and strings
//! - **CountingSort**: O(n+k) for bounded integer ranges
//! - **TimSort**: Hybrid algorithm optimized for real-world data

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::validation::{validate_args, extract_list};
use std::collections::HashMap;
use std::cmp::Ordering;

/// Comparator function type for custom sorting
pub type Comparator = Box<dyn Fn(&Value, &Value) -> Ordering + Send + Sync>;

/// Default comparison function for Lyra values
fn default_compare(a: &Value, b: &Value) -> Ordering {
    // Define a consistent type ordering: Integer < Real < String < Symbol < Boolean < Function < List < Object < LyObj < Missing < Quote < Pattern < Rule < PureFunction < Slot
    match (a, b) {
        // Same type comparisons
        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
        (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::String(a), Value::String(b)) => a.cmp(b),
        (Value::Symbol(a), Value::Symbol(b)) => a.cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        (Value::Function(a), Value::Function(b)) => a.cmp(b),
        
        // Numeric comparisons (Integer and Real are inter-comparable)
        (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(Ordering::Equal),
        
        // List comparison (lexicographic)
        (Value::List(a), Value::List(b)) => {
            for (x, y) in a.iter().zip(b.iter()) {
                match default_compare(x, y) {
                    Ordering::Equal => continue,
                    other => return other,
                }
            }
            a.len().cmp(&b.len())
        }
        
        // Mixed type comparisons - use consistent type ordering
        (Value::Integer(_), Value::Real(_)) => Ordering::Less, // Handled above
        (Value::Integer(_), _) => Ordering::Less,
        (Value::Real(_), Value::Integer(_)) => Ordering::Greater, // Handled above
        (Value::Real(_), _) => Ordering::Less,
        (Value::String(_), Value::Integer(_) | Value::Real(_)) => Ordering::Greater,
        (Value::String(_), _) => Ordering::Less,
        (Value::Symbol(_), Value::Integer(_) | Value::Real(_) | Value::String(_)) => Ordering::Greater,
        (Value::Symbol(_), _) => Ordering::Less,
        (Value::Boolean(_), Value::Integer(_) | Value::Real(_) | Value::String(_) | Value::Symbol(_)) => Ordering::Greater,
        (Value::Boolean(_), _) => Ordering::Less,
        (Value::Function(_), Value::Integer(_) | Value::Real(_) | Value::String(_) | Value::Symbol(_) | Value::Boolean(_)) => Ordering::Greater,
        (Value::Function(_), _) => Ordering::Less,
        (Value::List(_), Value::Integer(_) | Value::Real(_) | Value::String(_) | Value::Symbol(_) | Value::Boolean(_) | Value::Function(_)) => Ordering::Greater,
        (Value::List(_), _) => Ordering::Less,
        (Value::Object(_), Value::LyObj(_) | Value::Missing | Value::Quote(_) | Value::Pattern(_) | Value::Rule { .. } | Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::Object(_), _) => Ordering::Greater,
        (Value::LyObj(_), Value::Missing | Value::Quote(_) | Value::Pattern(_) | Value::Rule { .. } | Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::LyObj(_), _) => Ordering::Greater,
        
        // For complex types that can't be meaningfully compared, we just use a consistent ordering
        (Value::Missing, Value::Quote(_) | Value::Pattern(_) | Value::Rule { .. } | Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::Missing, _) => Ordering::Greater,
        (Value::Quote(_), Value::Pattern(_) | Value::Rule { .. } | Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::Quote(_), _) => Ordering::Greater,
        (Value::Pattern(_), Value::Rule { .. } | Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::Pattern(_), _) => Ordering::Greater,
        (Value::Rule { .. }, Value::PureFunction { .. } | Value::Slot { .. }) => Ordering::Less,
        (Value::Rule { .. }, _) => Ordering::Greater,
        (Value::PureFunction { .. }, Value::Slot { .. }) => Ordering::Less,
        (Value::PureFunction { .. }, _) => Ordering::Greater,
        (Value::Slot { .. }, _) => Ordering::Greater,
    }
}

/// Sort a list using the default stable sort algorithm (TimSort)
pub fn sort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let mut sorted_list = list.to_vec();
    timsort(&mut sorted_list, &default_compare);
    
    Ok(Value::List(sorted_list))
}

/// Sort a list using QuickSort algorithm
pub fn quicksort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let mut sorted_list = list.to_vec();
    quicksort(&mut sorted_list, &default_compare);
    
    Ok(Value::List(sorted_list))
}

/// Sort a list using MergeSort algorithm (stable)
pub fn mergesort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let sorted_list = mergesort(list, &default_compare);
    
    Ok(Value::List(sorted_list))
}

/// Sort a list using HeapSort algorithm
pub fn heapsort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let mut sorted_list = list.to_vec();
    heapsort(&mut sorted_list, &default_compare);
    
    Ok(Value::List(sorted_list))
}

/// Sort a list using InsertionSort algorithm
pub fn insertion_sort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let mut sorted_list = list.to_vec();
    insertion_sort(&mut sorted_list, &default_compare);
    
    Ok(Value::List(sorted_list))
}

/// Check if a list is sorted
pub fn is_sorted(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let sorted = list.windows(2).all(|pair| {
        default_compare(&pair[0], &pair[1]) != Ordering::Greater
    });
    
    Ok(Value::Symbol(if sorted { "True".to_string() } else { "False".to_string() }))
}

/// Sort a list using RadixSort algorithm (for integers and strings)
pub fn radix_sort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    // Check if all elements are integers or all are strings
    let all_integers = list.iter().all(|v| matches!(v, Value::Integer(_)));
    let all_strings = list.iter().all(|v| matches!(v, Value::String(_)));
    
    if all_integers {
        let sorted_list = radix_sort_integers(list);
        Ok(Value::List(sorted_list))
    } else if all_strings {
        let sorted_list = radix_sort_strings(list);
        Ok(Value::List(sorted_list))
    } else {
        // Fall back to default sort for mixed types
        let mut sorted_list = list.to_vec();
        timsort(&mut sorted_list, &default_compare);
        Ok(Value::List(sorted_list))
    }
}

/// Sort a list using CountingSort algorithm (for small integer ranges)
pub fn counting_sort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    // Check if all elements are integers
    if !list.iter().all(|v| matches!(v, Value::Integer(_))) {
        return Err(VmError::Runtime("CountingSort only works with integer lists".to_string()));
    }
    
    let sorted_list = counting_sort_integers(list)?;
    Ok(Value::List(sorted_list))
}

/// Sort a list using BucketSort algorithm (for uniformly distributed data)
pub fn bucket_sort_list(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    let list = extract_list(args, 0).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    // Check if all elements are numeric (integers or reals)
    if !list.iter().all(|v| matches!(v, Value::Integer(_) | Value::Real(_))) {
        return Err(VmError::Runtime("BucketSort only works with numeric lists".to_string()));
    }
    
    let sorted_list = bucket_sort_numeric(list);
    Ok(Value::List(sorted_list))
}

// Advanced sorting algorithm implementations

/// RadixSort for integers using LSD (Least Significant Digit) approach
fn radix_sort_integers(list: &[Value]) -> Vec<Value> {
    if list.is_empty() {
        return Vec::new();
    }
    
    // Extract integers and handle negative numbers
    let mut integers: Vec<i64> = list.iter().map(|v| {
        if let Value::Integer(i) = v { *i } else { 0 }
    }).collect();
    
    // Handle negative numbers by adding offset
    let min_val = *integers.iter().min().unwrap();
    let offset = if min_val < 0 { -min_val } else { 0 };
    
    for int in &mut integers {
        *int += offset;
    }
    
    // Find maximum value to determine number of digits
    let max_val = *integers.iter().max().unwrap() as u64;
    
    // Perform radix sort
    let mut exp = 1u64;
    while max_val / exp > 0 {
        radix_sort_by_digit(&mut integers, exp);
        exp *= 10;
    }
    
    // Remove offset and convert back to Values
    integers.into_iter().map(|i| Value::Integer(i - offset)).collect()
}

/// RadixSort helper function for sorting by a specific digit
fn radix_sort_by_digit(arr: &mut [i64], exp: u64) {
    let n = arr.len();
    let mut output = vec![0i64; n];
    let mut count = vec![0; 10]; // Count array for digits 0-9
    
    // Count occurrences of each digit
    for &val in arr.iter() {
        let digit = ((val as u64) / exp) % 10;
        count[digit as usize] += 1;
    }
    
    // Convert to actual positions
    for i in 1..10 {
        count[i] += count[i - 1];
    }
    
    // Build output array in stable order
    for &val in arr.iter().rev() {
        let digit = ((val as u64) / exp) % 10;
        count[digit as usize] -= 1;
        output[count[digit as usize]] = val;
    }
    
    // Copy back to original array
    arr.copy_from_slice(&output);
}

/// RadixSort for strings using MSD (Most Significant Digit) approach
fn radix_sort_strings(list: &[Value]) -> Vec<Value> {
    if list.is_empty() {
        return Vec::new();
    }
    
    let mut strings: Vec<String> = list.iter().map(|v| {
        if let Value::String(s) = v { s.clone() } else { String::new() }
    }).collect();
    
    // Find maximum length
    let max_len = strings.iter().map(|s| s.len()).max().unwrap_or(0);
    
    // Perform string radix sort from most significant character
    radix_sort_strings_recursive(&mut strings, 0, max_len);
    
    strings.into_iter().map(Value::String).collect()
}

/// Recursive helper for string radix sort
fn radix_sort_strings_recursive(strings: &mut Vec<String>, char_index: usize, max_len: usize) {
    if strings.len() <= 1 || char_index >= max_len {
        return;
    }
    
    // Use simple bucket sort for each character position
    let mut buckets: Vec<Vec<String>> = vec![Vec::new(); 256]; // ASCII characters
    
    for string in strings.drain(..) {
        let ch = if char_index < string.len() {
            string.chars().nth(char_index).unwrap_or('\0') as u8
        } else {
            0 // Treat missing characters as null
        };
        buckets[ch as usize].push(string);
    }
    
    // Recursively sort each bucket and collect results
    for bucket in &mut buckets {
        if bucket.len() > 1 {
            radix_sort_strings_recursive(bucket, char_index + 1, max_len);
        }
        strings.extend(bucket.drain(..));
    }
}

/// CountingSort for integers (works best with small ranges)
fn counting_sort_integers(list: &[Value]) -> VmResult<Vec<Value>> {
    if list.is_empty() {
        return Ok(Vec::new());
    }
    
    // Find min and max values
    let mut min_val = i64::MAX;
    let mut max_val = i64::MIN;
    
    for value in list {
        if let Value::Integer(i) = value {
            min_val = min_val.min(*i);
            max_val = max_val.max(*i);
        }
    }
    
    // Check if range is reasonable for counting sort
    let range = max_val - min_val + 1;
    if range > 1_000_000 {
        return Err(VmError::Runtime("CountingSort: Integer range too large (> 1M)".to_string()));
    }
    
    // Create count array
    let mut count = vec![0; range as usize];
    
    // Count occurrences
    for value in list {
        if let Value::Integer(i) = value {
            count[(i - min_val) as usize] += 1;
        }
    }
    
    // Build result array
    let mut result = Vec::new();
    for (i, &cnt) in count.iter().enumerate() {
        let val = min_val + i as i64;
        for _ in 0..cnt {
            result.push(Value::Integer(val));
        }
    }
    
    Ok(result)
}

/// BucketSort for numeric data (integers and reals)
fn bucket_sort_numeric(list: &[Value]) -> Vec<Value> {
    if list.is_empty() {
        return Vec::new();
    }
    
    // Convert all values to f64 for uniform processing
    let mut values: Vec<(f64, Value)> = list.iter().map(|v| match v {
        Value::Integer(i) => (*i as f64, v.clone()),
        Value::Real(r) => (*r, v.clone()),
        _ => (0.0, v.clone()),
    }).collect();
    
    // Find min and max values
    let min_val = values.iter().map(|(f, _)| *f).fold(f64::INFINITY, f64::min);
    let max_val = values.iter().map(|(f, _)| *f).fold(f64::NEG_INFINITY, f64::max);
    
    if (max_val - min_val).abs() < f64::EPSILON {
        return list.to_vec(); // All values are equal
    }
    
    // Create buckets (use sqrt(n) buckets for good performance)
    let bucket_count = (values.len() as f64).sqrt().ceil() as usize;
    let bucket_count = bucket_count.max(1);
    let mut buckets: Vec<Vec<(f64, Value)>> = vec![Vec::new(); bucket_count];
    
    // Distribute values into buckets
    let range = max_val - min_val;
    for (val, original) in values {
        let bucket_index = if val == max_val {
            bucket_count - 1
        } else {
            ((val - min_val) / range * bucket_count as f64) as usize
        };
        buckets[bucket_index].push((val, original));
    }
    
    // Sort each bucket and collect results
    let mut result = Vec::new();
    for bucket in buckets {
        let mut sorted_bucket = bucket;
        sorted_bucket.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        result.extend(sorted_bucket.into_iter().map(|(_, v)| v));
    }
    
    result
}

// Core sorting algorithm implementations

/// QuickSort implementation with median-of-three pivot selection
fn quicksort<T>(arr: &mut [T], cmp: &dyn Fn(&T, &T) -> Ordering) {
    if arr.len() <= 1 {
        return;
    }
    
    // Use insertion sort for small arrays
    if arr.len() <= 16 {
        insertion_sort(arr, cmp);
        return;
    }
    
    let pivot_index = partition_median_of_three(arr, cmp);
    let (left, right) = arr.split_at_mut(pivot_index);
    let (_pivot, right) = right.split_at_mut(1);
    
    quicksort(left, cmp);
    quicksort(right, cmp);
}

/// Partition with median-of-three pivot selection
fn partition_median_of_three<T>(arr: &mut [T], cmp: &dyn Fn(&T, &T) -> Ordering) -> usize {
    let len = arr.len();
    if len < 3 {
        return partition_hoare(arr, 0, cmp);
    }
    
    // Select median of first, middle, and last elements
    let mid = len / 2;
    let last = len - 1;
    
    // Sort the three pivot candidates
    if cmp(&arr[0], &arr[mid]) == Ordering::Greater {
        arr.swap(0, mid);
    }
    if cmp(&arr[mid], &arr[last]) == Ordering::Greater {
        arr.swap(mid, last);
        if cmp(&arr[0], &arr[mid]) == Ordering::Greater {
            arr.swap(0, mid);
        }
    }
    
    // Use middle element as pivot (move to first position)
    arr.swap(0, mid);
    partition_hoare(arr, 0, cmp)
}

/// Hoare partition scheme
fn partition_hoare<T>(arr: &mut [T], pivot_index: usize, cmp: &dyn Fn(&T, &T) -> Ordering) -> usize {
    let len = arr.len();
    if len <= 1 {
        return 0;
    }
    
    // Move pivot to end temporarily
    arr.swap(pivot_index, len - 1);
    
    let mut i = 0;
    for j in 0..len - 1 {
        if cmp(&arr[j], &arr[len - 1]) != Ordering::Greater {
            arr.swap(i, j);
            i += 1;
        }
    }
    
    // Move pivot to final position
    arr.swap(i, len - 1);
    i
}

/// MergeSort implementation (stable)
fn mergesort<T: Clone>(arr: &[T], cmp: &dyn Fn(&T, &T) -> Ordering) -> Vec<T> {
    if arr.len() <= 1 {
        return arr.to_vec();
    }
    
    let mid = arr.len() / 2;
    let left = mergesort(&arr[..mid], cmp);
    let right = mergesort(&arr[mid..], cmp);
    
    merge(left, right, cmp)
}

/// Merge two sorted arrays
fn merge<T: Clone>(left: Vec<T>, right: Vec<T>, cmp: &dyn Fn(&T, &T) -> Ordering) -> Vec<T> {
    let mut result = Vec::with_capacity(left.len() + right.len());
    let mut left_iter = left.into_iter();
    let mut right_iter = right.into_iter();
    let mut left_current = left_iter.next();
    let mut right_current = right_iter.next();
    
    loop {
        match (&left_current, &right_current) {
            (Some(l), Some(r)) => {
                if cmp(l, r) != Ordering::Greater {
                    result.push(left_current.take().unwrap());
                    left_current = left_iter.next();
                } else {
                    result.push(right_current.take().unwrap());
                    right_current = right_iter.next();
                }
            }
            (Some(_), None) => {
                result.push(left_current.take().unwrap());
                result.extend(left_iter);
                break;
            }
            (None, Some(_)) => {
                result.push(right_current.take().unwrap());
                result.extend(right_iter);
                break;
            }
            (None, None) => break,
        }
    }
    
    result
}

/// HeapSort implementation
fn heapsort<T>(arr: &mut [T], cmp: &dyn Fn(&T, &T) -> Ordering) {
    if arr.len() <= 1 {
        return;
    }
    
    // Build max heap
    for i in (0..arr.len() / 2).rev() {
        heapify(arr, i, arr.len(), cmp);
    }
    
    // Extract elements from heap
    for i in (1..arr.len()).rev() {
        arr.swap(0, i);
        heapify(arr, 0, i, cmp);
    }
}

/// Heapify a subtree rooted at index i
fn heapify<T>(arr: &mut [T], i: usize, heap_size: usize, cmp: &dyn Fn(&T, &T) -> Ordering) {
    let left = 2 * i + 1;
    let right = 2 * i + 2;
    let mut largest = i;
    
    if left < heap_size && cmp(&arr[left], &arr[largest]) == Ordering::Greater {
        largest = left;
    }
    
    if right < heap_size && cmp(&arr[right], &arr[largest]) == Ordering::Greater {
        largest = right;
    }
    
    if largest != i {
        arr.swap(i, largest);
        heapify(arr, largest, heap_size, cmp);
    }
}

/// InsertionSort implementation
fn insertion_sort<T>(arr: &mut [T], cmp: &dyn Fn(&T, &T) -> Ordering) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && cmp(&arr[j - 1], &arr[j]) == Ordering::Greater {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

/// Enhanced TimSort implementation with run detection
fn timsort<T: Clone>(arr: &mut [T], cmp: &dyn Fn(&T, &T) -> Ordering) {
    const MIN_MERGE: usize = 32;
    
    if arr.len() < 2 {
        return;
    }
    
    if arr.len() < MIN_MERGE {
        insertion_sort(arr, cmp);
        return;
    }
    
    // Find natural runs and extend short runs
    let mut runs = find_runs(arr, cmp);
    
    // Ensure minimum run length
    let min_run_len = compute_min_run_len(arr.len());
    for run in &mut runs {
        if run.len < min_run_len {
            let end = (run.start + min_run_len).min(arr.len());
            run.len = end - run.start;
            insertion_sort(&mut arr[run.start..end], cmp);
        }
    }
    
    // Merge runs using TimSort's merging strategy
    merge_runs(arr, &runs, cmp);
}

#[derive(Debug, Clone)]
struct Run {
    start: usize,
    len: usize,
}

/// Find natural runs in the array
fn find_runs<T>(arr: &[T], cmp: &dyn Fn(&T, &T) -> Ordering) -> Vec<Run> {
    let mut runs = Vec::new();
    let mut i = 0;
    
    while i < arr.len() {
        let start = i;
        
        // Find end of current run
        if i + 1 < arr.len() {
            if cmp(&arr[i], &arr[i + 1]) == Ordering::Greater {
                // Descending run - find end and reverse
                while i + 1 < arr.len() && cmp(&arr[i], &arr[i + 1]) == Ordering::Greater {
                    i += 1;
                }
            } else {
                // Ascending or equal run
                while i + 1 < arr.len() && cmp(&arr[i], &arr[i + 1]) != Ordering::Greater {
                    i += 1;
                }
            }
            i += 1;
        } else {
            i += 1;
        }
        
        runs.push(Run {
            start,
            len: i - start,
        });
    }
    
    runs
}

/// Compute minimum run length for TimSort
fn compute_min_run_len(n: usize) -> usize {
    let mut r = 0;
    let mut n = n;
    while n >= 32 {
        r |= n & 1;
        n >>= 1;
    }
    n + r
}

/// Merge runs using TimSort strategy
fn merge_runs<T: Clone>(arr: &mut [T], runs: &[Run], cmp: &dyn Fn(&T, &T) -> Ordering) {
    if runs.len() <= 1 {
        return;
    }
    
    // Simple merge strategy for now - merge adjacent runs
    for i in 1..runs.len() {
        let left_start = runs[i - 1].start;
        let left_end = left_start + runs[i - 1].len;
        let right_start = runs[i].start;
        let right_end = right_start + runs[i].len;
        
        if left_end == right_start {
            // Merge adjacent runs
            let left_part = arr[left_start..left_end].to_vec();
            let right_part = arr[right_start..right_end].to_vec();
            let merged = merge(left_part, right_part, cmp);
            
            for (j, item) in merged.into_iter().enumerate() {
                arr[left_start + j] = item;
            }
        }
    }
}

/// Register sorting functions with the standard library
pub fn register_sorting_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("Sort".to_string(), sort_list);
    functions.insert("QuickSort".to_string(), quicksort_list);
    functions.insert("MergeSort".to_string(), mergesort_list);
    functions.insert("HeapSort".to_string(), heapsort_list);
    functions.insert("InsertionSort".to_string(), insertion_sort_list);
    functions.insert("RadixSort".to_string(), radix_sort_list);
    functions.insert("CountingSort".to_string(), counting_sort_list);
    functions.insert("BucketSort".to_string(), bucket_sort_list);
    functions.insert("IsSorted".to_string(), is_sorted);
}

/// Get documentation for sorting functions
pub fn get_sorting_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    
    docs.insert("Sort".to_string(), 
        "Sort[list] - Sort a list using the default stable algorithm (enhanced TimSort). Works with numbers, strings, symbols, and lists.".to_string());
    
    docs.insert("QuickSort".to_string(),
        "QuickSort[list] - Sort using QuickSort algorithm. O(n log n) average case, uses median-of-three pivot selection.".to_string());
    
    docs.insert("MergeSort".to_string(),
        "MergeSort[list] - Sort using stable MergeSort algorithm. Guaranteed O(n log n) time complexity.".to_string());
    
    docs.insert("HeapSort".to_string(),
        "HeapSort[list] - Sort using HeapSort algorithm. Guaranteed O(n log n) time and O(1) space complexity.".to_string());
    
    docs.insert("InsertionSort".to_string(),
        "InsertionSort[list] - Sort using InsertionSort algorithm. O(n²) but efficient for small arrays.".to_string());
    
    docs.insert("RadixSort".to_string(),
        "RadixSort[list] - Sort integers or strings using Radix Sort. O(d×n) where d is number of digits/characters.".to_string());
    
    docs.insert("CountingSort".to_string(),
        "CountingSort[list] - Sort integers using Counting Sort. O(n+k) where k is the range. Best for small ranges.".to_string());
    
    docs.insert("BucketSort".to_string(),
        "BucketSort[list] - Sort numeric values using Bucket Sort. O(n) average case for uniformly distributed data.".to_string());
    
    docs.insert("IsSorted".to_string(),
        "IsSorted[list] - Check if a list is already sorted. Returns True or False.".to_string());
    
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_sort_integers() {
        let args = vec![Value::List(vec![
            Value::Integer(3),
            Value::Integer(1),
            Value::Integer(4),
            Value::Integer(1),
            Value::Integer(5),
        ])];
        
        let result = sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::Integer(1),
                Value::Integer(1),
                Value::Integer(3),
                Value::Integer(4),
                Value::Integer(5),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_sort_mixed_numbers() {
        let args = vec![Value::List(vec![
            Value::Real(3.14),
            Value::Integer(1),
            Value::Real(2.71),
            Value::Integer(4),
        ])];
        
        let result = sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            // Check that integers and reals are properly ordered
            let values: Vec<f64> = sorted.iter().map(|v| match v {
                Value::Integer(i) => *i as f64,
                Value::Real(r) => *r,
                _ => panic!("Unexpected type"),
            }).collect();
            
            assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_sort_strings() {
        let args = vec![Value::List(vec![
            Value::String("zebra".to_string()),
            Value::String("apple".to_string()),
            Value::String("banana".to_string()),
        ])];
        
        let result = sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::String("apple".to_string()),
                Value::String("banana".to_string()),
                Value::String("zebra".to_string()),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_quicksort() {
        let args = vec![Value::List(vec![
            Value::Integer(5),
            Value::Integer(2),
            Value::Integer(8),
            Value::Integer(1),
            Value::Integer(9),
        ])];
        
        let result = quicksort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(5),
                Value::Integer(8),
                Value::Integer(9),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_is_sorted() {
        let sorted_args = vec![Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ])];
        
        let result = is_sorted(&sorted_args).unwrap();
        assert_eq!(result, Value::Symbol("True".to_string()));
        
        let unsorted_args = vec![Value::List(vec![
            Value::Integer(3),
            Value::Integer(1),
            Value::Integer(2),
        ])];
        
        let result = is_sorted(&unsorted_args).unwrap();
        assert_eq!(result, Value::Symbol("False".to_string()));
    }
    
    #[test]
    fn test_empty_and_single_element_lists() {
        // Empty list
        let empty_args = vec![Value::List(vec![])];
        let result = sort_list(&empty_args).unwrap();
        assert_eq!(result, Value::List(vec![]));
        
        // Single element
        let single_args = vec![Value::List(vec![Value::Integer(42)])];
        let result = sort_list(&single_args).unwrap();
        assert_eq!(result, Value::List(vec![Value::Integer(42)]));
    }
    
    #[test]
    fn test_radix_sort_integers() {
        let args = vec![Value::List(vec![
            Value::Integer(170),
            Value::Integer(45),
            Value::Integer(75),
            Value::Integer(90),
            Value::Integer(2),
            Value::Integer(802),
            Value::Integer(24),
            Value::Integer(66),
        ])];
        
        let result = radix_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::Integer(2),
                Value::Integer(24),
                Value::Integer(45),
                Value::Integer(66),
                Value::Integer(75),
                Value::Integer(90),
                Value::Integer(170),
                Value::Integer(802),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_radix_sort_negative_integers() {
        let args = vec![Value::List(vec![
            Value::Integer(-5),
            Value::Integer(3),
            Value::Integer(-2),
            Value::Integer(1),
            Value::Integer(-10),
        ])];
        
        let result = radix_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::Integer(-10),
                Value::Integer(-5),
                Value::Integer(-2),
                Value::Integer(1),
                Value::Integer(3),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_radix_sort_strings() {
        let args = vec![Value::List(vec![
            Value::String("apple".to_string()),
            Value::String("banana".to_string()),
            Value::String("cherry".to_string()),
            Value::String("apricot".to_string()),
        ])];
        
        let result = radix_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::String("apple".to_string()),
                Value::String("apricot".to_string()),
                Value::String("banana".to_string()),
                Value::String("cherry".to_string()),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_counting_sort() {
        let args = vec![Value::List(vec![
            Value::Integer(4),
            Value::Integer(2),
            Value::Integer(2),
            Value::Integer(8),
            Value::Integer(3),
            Value::Integer(3),
            Value::Integer(1),
        ])];
        
        let result = counting_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            assert_eq!(sorted, vec![
                Value::Integer(1),
                Value::Integer(2),
                Value::Integer(2),
                Value::Integer(3),
                Value::Integer(3),
                Value::Integer(4),
                Value::Integer(8),
            ]);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_counting_sort_large_range_error() {
        // Test with large integer range that should fail
        let args = vec![Value::List(vec![
            Value::Integer(1),
            Value::Integer(2000000), // Large gap
        ])];
        
        assert!(counting_sort_list(&args).is_err());
    }
    
    #[test]
    fn test_counting_sort_non_integer_error() {
        let args = vec![Value::List(vec![
            Value::Integer(1),
            Value::String("test".to_string()),
        ])];
        
        assert!(counting_sort_list(&args).is_err());
    }
    
    #[test]
    fn test_bucket_sort() {
        let args = vec![Value::List(vec![
            Value::Real(0.897),
            Value::Real(0.565),
            Value::Real(0.656),
            Value::Real(0.1234),
            Value::Real(0.665),
            Value::Real(0.3434),
        ])];
        
        let result = bucket_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            // Check that it's sorted (exact order might vary due to floating point)
            let values: Vec<f64> = sorted.iter().map(|v| match v {
                Value::Real(r) => *r,
                _ => panic!("Expected Real value"),
            }).collect();
            
            assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_bucket_sort_mixed_numeric() {
        let args = vec![Value::List(vec![
            Value::Integer(5),
            Value::Real(2.5),
            Value::Integer(1),
            Value::Real(4.5),
        ])];
        
        let result = bucket_sort_list(&args).unwrap();
        if let Value::List(sorted) = result {
            // Check that integers and reals are properly mixed and sorted
            let values: Vec<f64> = sorted.iter().map(|v| match v {
                Value::Integer(i) => *i as f64,
                Value::Real(r) => *r,
                _ => panic!("Unexpected type"),
            }).collect();
            
            assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
            assert_eq!(values.len(), 4);
        } else {
            panic!("Expected list result");
        }
    }
    
    #[test]
    fn test_bucket_sort_non_numeric_error() {
        let args = vec![Value::List(vec![
            Value::Integer(1),
            Value::String("test".to_string()),
        ])];
        
        assert!(bucket_sort_list(&args).is_err());
    }
    
    #[test]
    fn test_all_sorting_algorithms_produce_same_result() {
        let test_data = vec![Value::Integer(5), Value::Integer(2), Value::Integer(8), Value::Integer(1), Value::Integer(9), Value::Integer(5)];
        let args = vec![Value::List(test_data)];
        
        let sort_result = sort_list(&args).unwrap();
        let quicksort_result = quicksort_list(&args).unwrap();
        let mergesort_result = mergesort_list(&args).unwrap();
        let heapsort_result = heapsort_list(&args).unwrap();
        let insertion_result = insertion_sort_list(&args).unwrap();
        let radix_result = radix_sort_list(&args).unwrap();
        let counting_result = counting_sort_list(&args).unwrap();
        let bucket_result = bucket_sort_list(&args).unwrap();
        
        // All algorithms should produce the same sorted result
        assert_eq!(sort_result, quicksort_result);
        assert_eq!(sort_result, mergesort_result);
        assert_eq!(sort_result, heapsort_result);
        assert_eq!(sort_result, insertion_result);
        assert_eq!(sort_result, radix_result);
        assert_eq!(sort_result, counting_result);
        assert_eq!(sort_result, bucket_result);
    }
    
    #[test]
    fn test_empty_lists_all_algorithms() {
        let empty_args = vec![Value::List(vec![])];
        
        // All sorting algorithms should handle empty lists
        assert_eq!(sort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(quicksort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(mergesort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(heapsort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(insertion_sort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(radix_sort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(counting_sort_list(&empty_args).unwrap(), Value::List(vec![]));
        assert_eq!(bucket_sort_list(&empty_args).unwrap(), Value::List(vec![]));
    }
}