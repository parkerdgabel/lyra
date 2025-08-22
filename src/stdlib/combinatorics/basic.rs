//! Basic Combinatorial Functions
//!
//! This module implements fundamental combinatorial functions including binomial coefficients,
//! permutations, combinations, and multinomial coefficients with overflow protection.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Multinomial coefficient result as a Foreign object
#[derive(Debug, Clone)]
pub struct MultinomialResult {
    /// Total number of objects
    pub n: i64,
    /// Group sizes
    pub groups: Vec<i64>,
    /// Computed multinomial coefficient
    pub coefficient: i64,
    /// Whether computation was successful (no overflow)
    pub success: bool,
}

impl MultinomialResult {
    pub fn new(n: i64, groups: Vec<i64>) -> Self {
        let mut result = Self {
            n,
            groups,
            coefficient: 0,
            success: false,
        };
        result.compute();
        result
    }
    
    fn compute(&mut self) {
        // Check if groups sum to n
        let sum: i64 = self.groups.iter().sum();
        if sum != self.n {
            self.success = false;
            return;
        }
        
        // Check for negative values
        if self.n < 0 || self.groups.iter().any(|&x| x < 0) {
            self.success = false;
            return;
        }
        
        // Compute multinomial coefficient: n! / (k1! * k2! * ... * km!)
        match multinomial_coefficient(self.n, &self.groups) {
            Some(coeff) => {
                self.coefficient = coeff;
                self.success = true;
            }
            None => {
                self.success = false;
            }
        }
    }
}

impl Foreign for MultinomialResult {
    fn type_name(&self) -> &'static str {
        "MultinomialResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "N" => Ok(Value::Integer(self.n)),
            "Groups" => {
                let groups: Vec<Value> = self.groups.iter()
                    .map(|&g| Value::Integer(g))
                    .collect();
                Ok(Value::List(groups))
            }
            "Coefficient" => Ok(Value::Integer(self.coefficient)),
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

/// Compute binomial coefficient C(n, k) with overflow protection
pub fn binomial_coefficient(n: i64, k: i64) -> Option<i64> {
    if k < 0 || k > n || n < 0 {
        return Some(0);
    }
    
    if k == 0 || k == n {
        return Some(1);
    }
    
    // Use symmetry to minimize computation: C(n,k) = C(n,n-k)
    let k = if k > n - k { n - k } else { k };
    
    // Use multiplicative formula with overflow checking
    let mut result = 1i64;
    for i in 0..k {
        // Compute result * (n - i) / (i + 1)
        match result.checked_mul(n - i) {
            Some(temp) => {
                result = temp / (i + 1);
            }
            None => return None, // Overflow
        }
    }
    
    Some(result)
}

/// Compute multinomial coefficient n! / (k1! * k2! * ... * km!)
pub fn multinomial_coefficient(n: i64, groups: &[i64]) -> Option<i64> {
    if groups.is_empty() {
        return Some(1);
    }
    
    // Check if groups sum to n
    let sum: i64 = groups.iter().sum();
    if sum != n {
        return Some(0);
    }
    
    // Check for negative values
    if n < 0 || groups.iter().any(|&x| x < 0) {
        return Some(0);
    }
    
    // Compute using repeated binomial coefficients
    let mut result = 1i64;
    let mut remaining = n;
    
    for &group_size in groups {
        if group_size == 0 {
            continue;
        }
        
        match binomial_coefficient(remaining, group_size) {
            Some(coeff) => {
                match result.checked_mul(coeff) {
                    Some(new_result) => result = new_result,
                    None => return None, // Overflow
                }
                remaining -= group_size;
            }
            None => return None, // Overflow in binomial
        }
    }
    
    Some(result)
}

/// Compute number of k-permutations of n objects: P(n,k) = n!/(n-k)!
pub fn permutations(n: i64, k: i64) -> Option<i64> {
    if k < 0 || k > n || n < 0 {
        return Some(0);
    }
    
    if k == 0 {
        return Some(1);
    }
    
    // Compute n * (n-1) * ... * (n-k+1)
    let mut result = 1i64;
    for i in 0..k {
        match result.checked_mul(n - i) {
            Some(new_result) => result = new_result,
            None => return None, // Overflow
        }
    }
    
    Some(result)
}

/// Compute number of k-combinations of n objects: C(n,k) = n!/(k!(n-k)!)
pub fn combinations(n: i64, k: i64) -> Option<i64> {
    binomial_coefficient(n, k)
}

/// Compute factorial with overflow protection
pub fn factorial(n: i64) -> Option<i64> {
    if n < 0 {
        return None;
    }
    
    if n <= 1 {
        return Some(1);
    }
    
    let mut result = 1i64;
    for i in 2..=n {
        match result.checked_mul(i) {
            Some(new_result) => result = new_result,
            None => return None, // Overflow
        }
    }
    
    Some(result)
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Binomial coefficient function
/// Syntax: Binomial[n, k]
pub fn binomial_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (n, k)".to_string(),
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
    
    match binomial_coefficient(n, k) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Binomial coefficient C({}, {}) too large", n, k))),
    }
}

/// Multinomial coefficient function
/// Syntax: Multinomial[n, {k1, k2, ...}]
pub fn multinomial_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (n, groups)".to_string(),
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
    
    let groups = match &args[1] {
        Value::List(list) => {
            let mut groups = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => groups.push(*i),
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for groups".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            groups
        }
        _ => return Err(VmError::TypeError {
            expected: "List for groups".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let result = MultinomialResult::new(n, groups);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Permutations function
/// Syntax: Permutations[n, k]
pub fn permutations_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (n, k)".to_string(),
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
    
    match permutations(n, k) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Permutation P({}, {}) too large", n, k))),
    }
}

/// Combinations function
/// Syntax: Combinations[n, k]
pub fn combinations_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (n, k)".to_string(),
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
    
    match combinations(n, k) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Combination C({}, {}) too large", n, k))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_binomial_coefficient() {
        assert_eq!(binomial_coefficient(5, 0), Some(1));
        assert_eq!(binomial_coefficient(5, 1), Some(5));
        assert_eq!(binomial_coefficient(5, 2), Some(10));
        assert_eq!(binomial_coefficient(5, 3), Some(10));
        assert_eq!(binomial_coefficient(5, 4), Some(5));
        assert_eq!(binomial_coefficient(5, 5), Some(1));
        
        // Test symmetry
        assert_eq!(binomial_coefficient(10, 3), binomial_coefficient(10, 7));
        
        // Test edge cases
        assert_eq!(binomial_coefficient(0, 0), Some(1));
        assert_eq!(binomial_coefficient(5, -1), Some(0));
        assert_eq!(binomial_coefficient(5, 6), Some(0));
        assert_eq!(binomial_coefficient(-1, 2), Some(0));
    }
    
    #[test]
    fn test_permutations() {
        assert_eq!(permutations(5, 0), Some(1));
        assert_eq!(permutations(5, 1), Some(5));
        assert_eq!(permutations(5, 2), Some(20));
        assert_eq!(permutations(5, 3), Some(60));
        assert_eq!(permutations(5, 5), Some(120)); // 5!
        
        // Edge cases
        assert_eq!(permutations(0, 0), Some(1));
        assert_eq!(permutations(5, -1), Some(0));
        assert_eq!(permutations(5, 6), Some(0));
    }
    
    #[test]
    fn test_combinations() {
        assert_eq!(combinations(5, 2), Some(10));
        assert_eq!(combinations(10, 5), Some(252));
        assert_eq!(combinations(0, 0), Some(1));
        assert_eq!(combinations(5, 0), Some(1));
        
        // Test relationship with permutations
        assert_eq!(
            permutations(5, 3).unwrap() / factorial(3).unwrap(),
            combinations(5, 3).unwrap()
        );
    }
    
    #[test]
    fn test_multinomial_coefficient() {
        // Multinomial(4, [2, 1, 1]) = 4!/(2!*1!*1!) = 12
        assert_eq!(multinomial_coefficient(4, &[2, 1, 1]), Some(12));
        
        // Multinomial(6, [3, 2, 1]) = 6!/(3!*2!*1!) = 60
        assert_eq!(multinomial_coefficient(6, &[3, 2, 1]), Some(60));
        
        // Edge cases
        assert_eq!(multinomial_coefficient(5, &[5]), Some(1));
        assert_eq!(multinomial_coefficient(0, &[]), Some(1));
        assert_eq!(multinomial_coefficient(5, &[2, 2]), Some(0)); // Doesn't sum to 5
    }
    
    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0), Some(1));
        assert_eq!(factorial(1), Some(1));
        assert_eq!(factorial(5), Some(120));
        assert_eq!(factorial(10), Some(3628800));
        assert_eq!(factorial(-1), None);
    }
    
    #[test]
    fn test_multinomial_result() {
        let result = MultinomialResult::new(5, vec![2, 2, 1]);
        assert_eq!(result.n, 5);
        assert_eq!(result.groups, vec![2, 2, 1]);
        assert_eq!(result.coefficient, 30); // 5!/(2!*2!*1!) = 30
        assert!(result.success);
        
        // Test invalid case
        let invalid = MultinomialResult::new(5, vec![2, 2, 2]);
        assert!(!invalid.success); // Doesn't sum to 5
    }
}