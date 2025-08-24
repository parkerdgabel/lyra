//! Advanced Combinatorial Functions
//!
//! This module implements sophisticated combinatorial functions including Stirling numbers,
//! Bell numbers, Catalan numbers, and integer partitions.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Stirling number result as a Foreign object
#[derive(Debug, Clone)]
pub struct StirlingResult {
    /// Parameter n
    pub n: i64,
    /// Parameter k
    pub k: i64,
    /// Type: 1 for first kind, 2 for second kind
    pub stirling_type: i64,
    /// Computed Stirling number
    pub value: i64,
    /// Whether computation was successful
    pub success: bool,
}

impl StirlingResult {
    pub fn new(n: i64, k: i64, stirling_type: i64) -> Self {
        let mut result = Self {
            n,
            k,
            stirling_type,
            value: 0,
            success: false,
        };
        result.compute();
        result
    }
    
    fn compute(&mut self) {
        match self.stirling_type {
            1 => {
                // Stirling numbers of the first kind (unsigned)
                match stirling_first_kind(self.n, self.k) {
                    Some(val) => {
                        self.value = val;
                        self.success = true;
                    }
                    None => {
                        self.success = false;
                    }
                }
            }
            2 => {
                // Stirling numbers of the second kind
                match stirling_second_kind(self.n, self.k) {
                    Some(val) => {
                        self.value = val;
                        self.success = true;
                    }
                    None => {
                        self.success = false;
                    }
                }
            }
            _ => {
                self.success = false;
            }
        }
    }
}

impl Foreign for StirlingResult {
    fn type_name(&self) -> &'static str {
        "StirlingResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "N" => Ok(Value::Integer(self.n)),
            "K" => Ok(Value::Integer(self.k)),
            "Type" => Ok(Value::Integer(self.stirling_type)),
            "Value" => Ok(Value::Integer(self.value)),
            "Success" => Ok(Value::Integer(if self.success { 1 } else { 0 })),
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

/// Integer partitions representation
#[derive(Debug, Clone)]
pub struct IntegerPartitions {
    /// The number being partitioned
    pub n: i64,
    /// All partitions of n
    pub partitions: Vec<Vec<i64>>,
    /// Number of partitions
    pub count: i64,
}

impl IntegerPartitions {
    pub fn new(n: i64) -> Self {
        let mut result = Self {
            n,
            partitions: Vec::new(),
            count: 0,
        };
        result.generate();
        result
    }
    
    fn generate(&mut self) {
        if self.n <= 0 {
            if self.n == 0 {
                self.partitions.push(vec![]);
                self.count = 1;
            }
            return;
        }
        
        self.partitions = generate_partitions(self.n);
        self.count = self.partitions.len() as i64;
    }
}

impl Foreign for IntegerPartitions {
    fn type_name(&self) -> &'static str {
        "IntegerPartitions"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "N" => Ok(Value::Integer(self.n)),
            "Count" => Ok(Value::Integer(self.count)),
            "Partitions" => {
                let partitions: Vec<Value> = self.partitions.iter()
                    .map(|partition| {
                        let parts: Vec<Value> = partition.iter()
                            .map(|&part| Value::Integer(part))
                            .collect();
                        Value::List(parts)
                    })
                    .collect();
                Ok(Value::List(partitions))
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

/// Compute Stirling numbers of the first kind (unsigned)
/// These count the number of permutations of n elements with k cycles
pub fn stirling_first_kind(n: i64, k: i64) -> Option<i64> {
    if n < 0 || k < 0 {
        return Some(0);
    }
    
    if n == 0 && k == 0 {
        return Some(1);
    }
    
    if n == 0 || k == 0 {
        return Some(0);
    }
    
    if k > n {
        return Some(0);
    }
    
    // Use dynamic programming with recurrence:
    // S(n,k) = (n-1)*S(n-1,k) + S(n-1,k-1)
    let n_usize = n as usize;
    let k_usize = k as usize;
    
    let mut dp = vec![vec![0i64; k_usize + 1]; n_usize + 1];
    dp[0][0] = 1;
    
    for i in 1..=n_usize {
        for j in 1..=std::cmp::min(i, k_usize) {
            match dp[i-1][j].checked_mul(i as i64 - 1) {
                Some(term1) => {
                    match term1.checked_add(dp[i-1][j-1]) {
                        Some(sum) => dp[i][j] = sum,
                        None => return None, // Overflow
                    }
                }
                None => return None, // Overflow
            }
        }
    }
    
    Some(dp[n_usize][k_usize])
}

/// Compute Stirling numbers of the second kind
/// These count the number of ways to partition n objects into k non-empty subsets
pub fn stirling_second_kind(n: i64, k: i64) -> Option<i64> {
    if n < 0 || k < 0 {
        return Some(0);
    }
    
    if n == 0 && k == 0 {
        return Some(1);
    }
    
    if n == 0 || k == 0 {
        return Some(0);
    }
    
    if k > n {
        return Some(0);
    }
    
    // Use dynamic programming with recurrence:
    // S(n,k) = k*S(n-1,k) + S(n-1,k-1)
    let n_usize = n as usize;
    let k_usize = k as usize;
    
    let mut dp = vec![vec![0i64; k_usize + 1]; n_usize + 1];
    dp[0][0] = 1;
    
    for i in 1..=n_usize {
        for j in 1..=std::cmp::min(i, k_usize) {
            match dp[i-1][j].checked_mul(j as i64) {
                Some(term1) => {
                    match term1.checked_add(dp[i-1][j-1]) {
                        Some(sum) => dp[i][j] = sum,
                        None => return None, // Overflow
                    }
                }
                None => return None, // Overflow
            }
        }
    }
    
    Some(dp[n_usize][k_usize])
}

/// Compute Bell number B(n) - the number of partitions of a set with n elements
pub fn bell_number(n: i64) -> Option<i64> {
    if n < 0 {
        return Some(0);
    }
    
    if n == 0 {
        return Some(1);
    }
    
    // Bell numbers can be computed using Stirling numbers of the second kind:
    // B(n) = sum(S(n,k)) for k=0 to n
    let mut sum = 0i64;
    for k in 0..=n {
        match stirling_second_kind(n, k) {
            Some(stirling) => {
                match sum.checked_add(stirling) {
                    Some(new_sum) => sum = new_sum,
                    None => return None, // Overflow
                }
            }
            None => return None, // Overflow in Stirling computation
        }
    }
    
    Some(sum)
}

/// Compute Catalan number C(n) using the formula C(n) = (1/(n+1)) * C(2n,n)
pub fn catalan_number(n: i64) -> Option<i64> {
    if n < 0 {
        return Some(0);
    }
    
    if n == 0 {
        return Some(1);
    }
    
    // Use the formula: C(n) = C(2n,n) / (n+1)
    // which can be computed as: C(n) = (2n)! / ((n+1)! * n!)
    
    // For efficiency, use the recurrence: C(n) = sum(C(i) * C(n-1-i)) for i=0 to n-1
    let n_usize = n as usize;
    let mut catalan = vec![0i64; n_usize + 1];
    catalan[0] = 1;
    
    for i in 1..=n_usize {
        for j in 0..i {
            match catalan[j].checked_mul(catalan[i - 1 - j]) {
                Some(product) => {
                    match catalan[i].checked_add(product) {
                        Some(sum) => catalan[i] = sum,
                        None => return None, // Overflow
                    }
                }
                None => return None, // Overflow
            }
        }
    }
    
    Some(catalan[n_usize])
}

/// Generate all integer partitions of n
pub fn generate_partitions(n: i64) -> Vec<Vec<i64>> {
    if n <= 0 {
        if n == 0 {
            return vec![vec![]];
        } else {
            return vec![];
        }
    }
    
    let mut result = Vec::new();
    generate_partitions_helper(n, n, vec![], &mut result);
    result
}

/// Helper function for generating partitions using backtracking
fn generate_partitions_helper(n: i64, max_val: i64, current: Vec<i64>, result: &mut Vec<Vec<i64>>) {
    if n == 0 {
        result.push(current);
        return;
    }
    
    for i in std::cmp::min(n, max_val)..=1 {
        let mut new_current = current.clone();
        new_current.push(i);
        generate_partitions_helper(n - i, i, new_current, result);
    }
}

/// Count the number of partitions of n (partition function p(n))
pub fn partition_count(n: i64) -> i64 {
    if n <= 0 {
        return if n == 0 { 1 } else { 0 };
    }
    
    // Use dynamic programming
    let n_usize = n as usize;
    let mut dp = vec![0i64; n_usize + 1];
    dp[0] = 1;
    
    for i in 1..=n_usize {
        for j in i..=n_usize {
            dp[j] += dp[j - i];
        }
    }
    
    dp[n_usize]
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Stirling number function
/// Syntax: StirlingNumber[n, k, type]
pub fn stirling_number_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (n, k, type)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let k = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for k".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let stirling_type = match &args[2] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for type (1 or 2)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    if stirling_type != 1 && stirling_type != 2 {
        return Err(VmError::Runtime("Stirling type must be 1 or 2".to_string()));
    }
    
    let result = StirlingResult::new(n, k, stirling_type);
    // Return standardized Association instead of Foreign object
    let mut m = std::collections::HashMap::new();
    m.insert("n".to_string(), Value::Integer(result.n));
    m.insert("k".to_string(), Value::Integer(result.k));
    m.insert("type".to_string(), Value::Integer(result.stirling_type));
    m.insert("value".to_string(), Value::Integer(result.value));
    m.insert("success".to_string(), Value::Boolean(result.success));
    Ok(Value::Object(m))
}

/// Bell number function
/// Syntax: BellNumber[n]
pub fn bell_number_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    match bell_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Bell number B({}) too large", n))),
    }
}

/// Catalan number function
/// Syntax: CatalanNumber[n]
pub fn catalan_number_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    match catalan_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Catalan number C({}) too large", n))),
    }
}

/// Integer partitions function
/// Syntax: Partitions[n]
pub fn partitions_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n > 100 {
        return Err(VmError::Runtime("Partition enumeration limited to n <= 100".to_string()));
    }
    
    let result = IntegerPartitions::new(n);
    // Return standardized Association instead of Foreign object
    let mut m = std::collections::HashMap::new();
    m.insert("n".to_string(), Value::Integer(result.n));
    m.insert(
        "partitions".to_string(),
        Value::List(
            result
                .partitions
                .iter()
                .map(|p| Value::List(p.iter().cloned().map(Value::Integer).collect()))
                .collect(),
        ),
    );
    m.insert("count".to_string(), Value::Integer(result.count));
    Ok(Value::Object(m))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stirling_first_kind() {
        // Known values of unsigned Stirling numbers of the first kind
        assert_eq!(stirling_first_kind(0, 0), Some(1));
        assert_eq!(stirling_first_kind(1, 1), Some(1));
        assert_eq!(stirling_first_kind(2, 1), Some(1));
        assert_eq!(stirling_first_kind(2, 2), Some(1));
        assert_eq!(stirling_first_kind(3, 1), Some(2));
        assert_eq!(stirling_first_kind(3, 2), Some(3));
        assert_eq!(stirling_first_kind(3, 3), Some(1));
        assert_eq!(stirling_first_kind(4, 2), Some(11));
        
        // Edge cases
        assert_eq!(stirling_first_kind(5, 0), Some(0));
        assert_eq!(stirling_first_kind(0, 5), Some(0));
        assert_eq!(stirling_first_kind(3, 4), Some(0));
    }
    
    #[test]
    fn test_stirling_second_kind() {
        // Known values of Stirling numbers of the second kind
        assert_eq!(stirling_second_kind(0, 0), Some(1));
        assert_eq!(stirling_second_kind(1, 1), Some(1));
        assert_eq!(stirling_second_kind(2, 1), Some(1));
        assert_eq!(stirling_second_kind(2, 2), Some(1));
        assert_eq!(stirling_second_kind(3, 1), Some(1));
        assert_eq!(stirling_second_kind(3, 2), Some(3));
        assert_eq!(stirling_second_kind(3, 3), Some(1));
        assert_eq!(stirling_second_kind(4, 2), Some(7));
        
        // Edge cases
        assert_eq!(stirling_second_kind(5, 0), Some(0));
        assert_eq!(stirling_second_kind(0, 5), Some(0));
        assert_eq!(stirling_second_kind(3, 4), Some(0));
    }
    
    #[test]
    fn test_bell_number() {
        // Known Bell numbers
        assert_eq!(bell_number(0), Some(1));
        assert_eq!(bell_number(1), Some(1));
        assert_eq!(bell_number(2), Some(2));
        assert_eq!(bell_number(3), Some(5));
        assert_eq!(bell_number(4), Some(15));
        assert_eq!(bell_number(5), Some(52));
        
        // Verify relationship with Stirling numbers
        let n = 4;
        let bell_n = bell_number(n).unwrap();
        let mut sum = 0;
        for k in 0..=n {
            sum += stirling_second_kind(n, k).unwrap();
        }
        assert_eq!(bell_n, sum);
    }
    
    #[test]
    fn test_catalan_number() {
        // Known Catalan numbers
        assert_eq!(catalan_number(0), Some(1));
        assert_eq!(catalan_number(1), Some(1));
        assert_eq!(catalan_number(2), Some(2));
        assert_eq!(catalan_number(3), Some(5));
        assert_eq!(catalan_number(4), Some(14));
        assert_eq!(catalan_number(5), Some(42));
        assert_eq!(catalan_number(6), Some(132));
    }
    
    #[test]
    fn test_generate_partitions() {
        // Partitions of 4: {4}, {3,1}, {2,2}, {2,1,1}, {1,1,1,1}
        let partitions = generate_partitions(4);
        assert_eq!(partitions.len(), 5);
        
        // Check that all partitions sum to 4
        for partition in &partitions {
            assert_eq!(partition.iter().sum::<i64>(), 4);
        }
        
        // Test edge cases
        assert_eq!(generate_partitions(0), vec![vec![]]);
        assert_eq!(generate_partitions(1), vec![vec![1]]);
    }
    
    #[test]
    fn test_partition_count() {
        // Known partition counts
        assert_eq!(partition_count(0), 1);
        assert_eq!(partition_count(1), 1);
        assert_eq!(partition_count(2), 2);
        assert_eq!(partition_count(3), 3);
        assert_eq!(partition_count(4), 5);
        assert_eq!(partition_count(5), 7);
        assert_eq!(partition_count(6), 11);
        
        // Verify consistency with generated partitions
        for n in 1..=10 {
            let count1 = partition_count(n);
            let count2 = generate_partitions(n).len() as i64;
            assert_eq!(count1, count2);
        }
    }
}
