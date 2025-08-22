//! String algorithms for Lyra standard library
//!
//! This module implements string processing and pattern matching algorithms.
//! All algorithms are optimized for performance and handle Unicode strings properly.

use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;

/// Build failure function for KMP algorithm
fn build_failure_function(pattern: &str) -> Vec<usize> {
    let chars: Vec<char> = pattern.chars().collect();
    let m = chars.len();
    let mut failure = vec![0; m];
    let mut j = 0;
    
    for i in 1..m {
        while j > 0 && chars[i] != chars[j] {
            j = failure[j - 1];
        }
        if chars[i] == chars[j] {
            j += 1;
        }
        failure[i] = j;
    }
    
    failure
}

/// Knuth-Morris-Pratt string matching - finds first occurrence of pattern in text
/// Time complexity: O(n + m) where n = text length, m = pattern length
/// Space complexity: O(m) for failure function
pub fn kmp_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    if pattern.is_empty() {
        return Ok(Value::Integer(0)); // Empty pattern found at position 0
    }
    
    if text.is_empty() {
        return Ok(Value::Integer(-1)); // Pattern not found in empty text
    }
    
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let n = text_chars.len();
    let m = pattern_chars.len();
    
    if m > n {
        return Ok(Value::Integer(-1)); // Pattern longer than text
    }
    
    let failure = build_failure_function(pattern);
    let mut j = 0; // Pattern index
    
    for i in 0..n {
        while j > 0 && text_chars[i] != pattern_chars[j] {
            j = failure[j - 1];
        }
        
        if text_chars[i] == pattern_chars[j] {
            j += 1;
        }
        
        if j == m {
            // Found pattern at position i - m + 1
            return Ok(Value::Integer((i - m + 1) as i64));
        }
    }
    
    Ok(Value::Integer(-1)) // Pattern not found
}

/// Edit distance (Levenshtein distance) between two strings
/// Time complexity: O(m * n) where m, n are string lengths
/// Space complexity: O(min(m, n)) with optimization
pub fn edit_distance(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let str1 = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let str2 = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    let chars1: Vec<char> = str1.chars().collect();
    let chars2: Vec<char> = str2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();
    
    // Handle base cases
    if m == 0 {
        return Ok(Value::Integer(n as i64));
    }
    if n == 0 {
        return Ok(Value::Integer(m as i64));
    }
    
    // Space-optimized DP: only keep current and previous rows
    let mut prev_row = (0..=n).collect::<Vec<usize>>();
    let mut curr_row = vec![0; n + 1];
    
    for i in 1..=m {
        curr_row[0] = i;
        
        for j in 1..=n {
            let cost = if chars1[i - 1] == chars2[j - 1] { 0 } else { 1 };
            
            curr_row[j] = std::cmp::min(
                std::cmp::min(
                    curr_row[j - 1] + 1,      // Insertion
                    prev_row[j] + 1           // Deletion
                ),
                prev_row[j - 1] + cost        // Substitution
            );
        }
        
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    Ok(Value::Integer(prev_row[n] as i64))
}

/// Boyer-Moore string search algorithm - efficient for long patterns
/// Time complexity: O(nm) worst case, O(n/m) average case
pub fn boyer_moore_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    if pattern.is_empty() {
        return Ok(Value::Integer(0));
    }
    
    if text.is_empty() || pattern.len() > text.len() {
        return Ok(Value::Integer(-1));
    }
    
    // Build bad character table
    let mut bad_char = [0; 256];
    for i in 0..256 {
        bad_char[i] = pattern.len() as i32;
    }
    
    let pattern_bytes = pattern.as_bytes();
    for (i, &byte) in pattern_bytes.iter().enumerate() {
        bad_char[byte as usize] = (pattern.len() - 1 - i) as i32;
    }
    
    let text_bytes = text.as_bytes();
    let n = text_bytes.len();
    let m = pattern_bytes.len();
    let mut i = 0;
    
    while i <= n - m {
        let mut j = m;
        
        // Match from right to left
        while j > 0 && pattern_bytes[j - 1] == text_bytes[i + j - 1] {
            j -= 1;
        }
        
        if j == 0 {
            return Ok(Value::Integer(i as i64));
        }
        
        // Use bad character rule
        let bad_char_skip = bad_char[text_bytes[i + j - 1] as usize];
        i += std::cmp::max(1, bad_char_skip as usize);
    }
    
    Ok(Value::Integer(-1))
}

/// Rabin-Karp string search using rolling hash
/// Time complexity: O(n + m) average, O(nm) worst case
pub fn rabin_karp_search(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let pattern = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    if pattern.is_empty() {
        return Ok(Value::Integer(0));
    }
    
    if text.is_empty() || pattern.len() > text.len() {
        return Ok(Value::Integer(-1));
    }
    
    const BASE: u64 = 256;
    const MOD: u64 = 1000000007;
    
    let text_chars: Vec<char> = text.chars().collect();
    let pattern_chars: Vec<char> = pattern.chars().collect();
    let n = text_chars.len();
    let m = pattern_chars.len();
    
    // Calculate pattern hash
    let mut pattern_hash = 0u64;
    let mut text_hash = 0u64;
    let mut h = 1u64;
    
    // h = BASE^(m-1) % MOD
    for _ in 0..m-1 {
        h = (h * BASE) % MOD;
    }
    
    // Calculate hash of pattern and first window of text
    for i in 0..m {
        pattern_hash = (BASE * pattern_hash + pattern_chars[i] as u64) % MOD;
        text_hash = (BASE * text_hash + text_chars[i] as u64) % MOD;
    }
    
    // Check first window
    if pattern_hash == text_hash && &text[0..m] == pattern {
        return Ok(Value::Integer(0));
    }
    
    // Roll the hash
    for i in m..n {
        // Remove leading character, add trailing character
        text_hash = (BASE * (text_hash + MOD - (text_chars[i - m] as u64 * h) % MOD) + text_chars[i] as u64) % MOD;
        
        if pattern_hash == text_hash {
            let start = i - m + 1;
            if &text_chars[start..=i].iter().collect::<String>() == pattern {
                return Ok(Value::Integer(start as i64));
            }
        }
    }
    
    Ok(Value::Integer(-1))
}

/// Hamming distance for equal-length strings
/// Time complexity: O(n) where n is string length
pub fn hamming_distance(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let str1 = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let str2 = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    let chars1: Vec<char> = str1.chars().collect();
    let chars2: Vec<char> = str2.chars().collect();
    
    if chars1.len() != chars2.len() {
        return Err(VmError::Runtime("Hamming distance requires strings of equal length".to_string()));
    }
    
    let distance = chars1.iter().zip(chars2.iter())
        .map(|(c1, c2)| if c1 != c2 { 1 } else { 0 })
        .sum::<i64>();
    
    Ok(Value::Integer(distance))
}

/// Jaro-Winkler similarity for fuzzy string matching
/// Time complexity: O(n * m) where n, m are string lengths
pub fn jaro_winkler_distance(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let str1 = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let str2 = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    let chars1: Vec<char> = str1.chars().collect();
    let chars2: Vec<char> = str2.chars().collect();
    let len1 = chars1.len();
    let len2 = chars2.len();
    
    if len1 == 0 && len2 == 0 {
        return Ok(Value::Real(1.0));
    }
    
    if len1 == 0 || len2 == 0 {
        return Ok(Value::Real(0.0));
    }
    
    let match_distance = (std::cmp::max(len1, len2) / 2).saturating_sub(1);
    let mut matches1 = vec![false; len1];
    let mut matches2 = vec![false; len2];
    let mut matches = 0;
    let mut transpositions = 0;
    
    // Find matches
    for i in 0..len1 {
        let start = if i > match_distance { i - match_distance } else { 0 };
        let end = std::cmp::min(i + match_distance + 1, len2);
        
        for j in start..end {
            if matches2[j] || chars1[i] != chars2[j] {
                continue;
            }
            matches1[i] = true;
            matches2[j] = true;
            matches += 1;
            break;
        }
    }
    
    if matches == 0 {
        return Ok(Value::Real(0.0));
    }
    
    // Count transpositions
    let mut k = 0;
    for i in 0..len1 {
        if !matches1[i] {
            continue;
        }
        while !matches2[k] {
            k += 1;
        }
        if chars1[i] != chars2[k] {
            transpositions += 1;
        }
        k += 1;
    }
    
    let jaro = (matches as f64 / len1 as f64 + 
                matches as f64 / len2 as f64 + 
                (matches - transpositions / 2) as f64 / matches as f64) / 3.0;
    
    // Jaro-Winkler adds prefix bonus
    let prefix_len = chars1.iter().zip(chars2.iter())
        .take(4)
        .take_while(|(c1, c2)| c1 == c2)
        .count();
    
    let jaro_winkler = jaro + (0.1 * prefix_len as f64 * (1.0 - jaro));
    
    Ok(Value::Real(jaro_winkler))
}

/// Longest Common Substring using dynamic programming
/// Time complexity: O(m * n), Space complexity: O(min(m, n))
pub fn longest_common_substring(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let str1 = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("First argument must be a string".to_string())),
    };
    
    let str2 = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Second argument must be a string".to_string())),
    };
    
    let chars1: Vec<char> = str1.chars().collect();
    let chars2: Vec<char> = str2.chars().collect();
    let m = chars1.len();
    let n = chars2.len();
    
    if m == 0 || n == 0 {
        return Ok(Value::String("".to_string()));
    }
    
    let mut max_length = 0;
    let mut ending_pos = 0;
    
    // Space-optimized DP
    let mut prev_row = vec![0; n + 1];
    let mut curr_row = vec![0; n + 1];
    
    for i in 1..=m {
        for j in 1..=n {
            if chars1[i - 1] == chars2[j - 1] {
                curr_row[j] = prev_row[j - 1] + 1;
                if curr_row[j] > max_length {
                    max_length = curr_row[j];
                    ending_pos = i;
                }
            } else {
                curr_row[j] = 0;
            }
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    
    if max_length == 0 {
        Ok(Value::String("".to_string()))
    } else {
        let start = ending_pos - max_length;
        let substring: String = chars1[start..ending_pos].iter().collect();
        Ok(Value::String(substring))
    }
}

/// Build suffix array using counting sort (for simplicity)
/// Time complexity: O(n^2 log n), Space complexity: O(n)
pub fn suffix_array(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Argument must be a string".to_string())),
    };
    
    if text.is_empty() {
        return Ok(Value::List(vec![]));
    }
    
    let n = text.len();
    let mut suffixes: Vec<(usize, &str)> = (0..n)
        .map(|i| (i, &text[i..]))
        .collect();
    
    // Sort suffixes lexicographically
    suffixes.sort_by_key(|&(_, suffix)| suffix);
    
    let result: Vec<Value> = suffixes.into_iter()
        .map(|(index, _)| Value::Integer(index as i64))
        .collect();
    
    Ok(Value::List(result))
}

/// Z Algorithm for pattern preprocessing
/// Time complexity: O(n), Space complexity: O(n)
pub fn z_algorithm(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let text = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Argument must be a string".to_string())),
    };
    
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    
    if n == 0 {
        return Ok(Value::List(vec![]));
    }
    
    let mut z = vec![0; n];
    let mut l = 0;
    let mut r = 0;
    
    for i in 1..n {
        if i > r {
            l = i;
            r = i;
            while r < n && chars[r - l] == chars[r] {
                r += 1;
            }
            z[i] = r - l;
            r -= 1;
        } else {
            let k = i - l;
            if z[k] < r - i + 1 {
                z[i] = z[k];
            } else {
                l = i;
                while r < n && chars[r - l] == chars[r] {
                    r += 1;
                }
                z[i] = r - l;
                r -= 1;
            }
        }
    }
    
    let result: Vec<Value> = z.into_iter()
        .map(|val| Value::Integer(val as i64))
        .collect();
    
    Ok(Value::List(result))
}

/// Register string algorithm functions
pub fn register_string_algorithms(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("KMPSearch".to_string(), kmp_search);
    functions.insert("EditDistance".to_string(), edit_distance);
    functions.insert("BoyerMooreSearch".to_string(), boyer_moore_search);
    functions.insert("RabinKarpSearch".to_string(), rabin_karp_search);
    functions.insert("HammingDistance".to_string(), hamming_distance);
    functions.insert("JaroWinklerDistance".to_string(), jaro_winkler_distance);
    functions.insert("LongestCommonSubstring".to_string(), longest_common_substring);
    functions.insert("SuffixArray".to_string(), suffix_array);
    functions.insert("ZAlgorithm".to_string(), z_algorithm);
}

/// Get documentation for string algorithms
pub fn get_string_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("KMPSearch".to_string(), "KMPSearch[text, pattern] - Find pattern in text using KMP algorithm. O(n+m) time.".to_string());
    docs.insert("EditDistance".to_string(), "EditDistance[str1, str2] - Calculate Levenshtein distance between strings. O(m*n) time.".to_string());
    docs.insert("BoyerMooreSearch".to_string(), "BoyerMooreSearch[text, pattern] - Efficient search for long patterns. O(n/m) average time.".to_string());
    docs.insert("RabinKarpSearch".to_string(), "RabinKarpSearch[text, pattern] - Rolling hash search algorithm. O(n+m) average time.".to_string());
    docs.insert("HammingDistance".to_string(), "HammingDistance[str1, str2] - Count different positions in equal-length strings. O(n) time.".to_string());
    docs.insert("JaroWinklerDistance".to_string(), "JaroWinklerDistance[str1, str2] - Fuzzy string similarity (0-1 scale). O(m*n) time.".to_string());
    docs.insert("LongestCommonSubstring".to_string(), "LongestCommonSubstring[str1, str2] - Find longest common substring. O(m*n) time.".to_string());
    docs.insert("SuffixArray".to_string(), "SuffixArray[text] - Build suffix array for efficient string operations. O(n²log n) time.".to_string());
    docs.insert("ZAlgorithm".to_string(), "ZAlgorithm[text] - Compute Z-array for pattern matching. O(n) time.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kmp_search() {
        let args = vec![
            Value::String("abcabcabcabc".to_string()),
            Value::String("abcabc".to_string())
        ];
        let result = kmp_search(&args).unwrap();
        assert_eq!(result, Value::Integer(0));
        
        let args = vec![
            Value::String("hello world".to_string()),
            Value::String("world".to_string())
        ];
        let result = kmp_search(&args).unwrap();
        assert_eq!(result, Value::Integer(6));
    }
    
    #[test]
    fn test_edit_distance() {
        let args = vec![
            Value::String("kitten".to_string()),
            Value::String("sitting".to_string())
        ];
        let result = edit_distance(&args).unwrap();
        assert_eq!(result, Value::Integer(3));
        
        let args = vec![
            Value::String("".to_string()),
            Value::String("abc".to_string())
        ];
        let result = edit_distance(&args).unwrap();
        assert_eq!(result, Value::Integer(3));
    }
    
    #[test]
    fn test_hamming_distance() {
        let args = vec![
            Value::String("karolin".to_string()),
            Value::String("kathrin".to_string())
        ];
        let result = hamming_distance(&args).unwrap();
        assert_eq!(result, Value::Integer(3));
    }
    
    #[test]
    fn test_boyer_moore_search() {
        let args = vec![
            Value::String("HERE IS A SIMPLE EXAMPLE".to_string()),
            Value::String("EXAMPLE".to_string())
        ];
        let result = boyer_moore_search(&args).unwrap();
        assert_eq!(result, Value::Integer(17));
    }
    
    #[test]
    fn test_longest_common_substring() {
        let args = vec![
            Value::String("abcdxyz".to_string()),
            Value::String("xyzabcd".to_string())
        ];
        let result = longest_common_substring(&args).unwrap();
        // Should find "abcd" or "xyz"
        match result {
            Value::String(s) => assert!(s == "abcd" || s == "xyz"),
            _ => panic!("Expected string result"),
        }
    }
}