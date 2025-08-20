//! Combinatorial Sequences
//!
//! This module implements important combinatorial sequences including Fibonacci numbers,
//! Lucas numbers, and other recurrence-based sequences.

use crate::vm::{Value, VmResult, VmError};

/// Compute Fibonacci number F(n) using matrix exponentiation for efficiency
pub fn fibonacci_number(n: i64) -> Option<i64> {
    if n < 0 {
        // Extended Fibonacci: F(-n) = (-1)^(n+1) * F(n)
        match fibonacci_number(-n) {
            Some(fib) => {
                if (-n) % 2 == 0 {
                    Some(-fib)
                } else {
                    Some(fib)
                }
            }
            None => None,
        }
    } else if n == 0 {
        Some(0)
    } else if n == 1 {
        Some(1)
    } else {
        // Use iterative approach for efficiency and overflow protection
        let mut a = 0i64;
        let mut b = 1i64;
        
        for _ in 2..=n {
            match a.checked_add(b) {
                Some(next) => {
                    a = b;
                    b = next;
                }
                None => return None, // Overflow
            }
        }
        
        Some(b)
    }
}

/// Compute Lucas number L(n)
/// Lucas numbers follow the same recurrence as Fibonacci but with different initial conditions:
/// L(0) = 2, L(1) = 1, L(n) = L(n-1) + L(n-2)
pub fn lucas_number(n: i64) -> Option<i64> {
    if n < 0 {
        // Extended Lucas: L(-n) = (-1)^n * L(n)
        match lucas_number(-n) {
            Some(lucas) => {
                if (-n) % 2 == 0 {
                    Some(lucas)
                } else {
                    Some(-lucas)
                }
            }
            None => None,
        }
    } else if n == 0 {
        Some(2)
    } else if n == 1 {
        Some(1)
    } else {
        // Use iterative approach
        let mut a = 2i64;
        let mut b = 1i64;
        
        for _ in 2..=n {
            match a.checked_add(b) {
                Some(next) => {
                    a = b;
                    b = next;
                }
                None => return None, // Overflow
            }
        }
        
        Some(b)
    }
}

/// Compute Tribonacci number T(n)
/// T(0) = 0, T(1) = 0, T(2) = 1, T(n) = T(n-1) + T(n-2) + T(n-3)
pub fn tribonacci_number(n: i64) -> Option<i64> {
    if n < 0 {
        return Some(0);
    } else if n == 0 || n == 1 {
        return Some(0);
    } else if n == 2 {
        return Some(1);
    }
    
    let mut a = 0i64;
    let mut b = 0i64;
    let mut c = 1i64;
    
    for _ in 3..=n {
        match a.checked_add(b) {
            Some(temp) => {
                match temp.checked_add(c) {
                    Some(next) => {
                        a = b;
                        b = c;
                        c = next;
                    }
                    None => return None, // Overflow
                }
            }
            None => return None, // Overflow
        }
    }
    
    Some(c)
}

/// Compute Pell number P(n)
/// P(0) = 0, P(1) = 1, P(n) = 2*P(n-1) + P(n-2)
pub fn pell_number(n: i64) -> Option<i64> {
    if n < 0 {
        return Some(0);
    } else if n == 0 {
        return Some(0);
    } else if n == 1 {
        return Some(1);
    }
    
    let mut a = 0i64;
    let mut b = 1i64;
    
    for _ in 2..=n {
        match b.checked_mul(2) {
            Some(doubled) => {
                match doubled.checked_add(a) {
                    Some(next) => {
                        a = b;
                        b = next;
                    }
                    None => return None, // Overflow
                }
            }
            None => return None, // Overflow
        }
    }
    
    Some(b)
}

/// Compute Jacobsthal number J(n)
/// J(0) = 0, J(1) = 1, J(n) = J(n-1) + 2*J(n-2)
pub fn jacobsthal_number(n: i64) -> Option<i64> {
    if n < 0 {
        return Some(0);
    } else if n == 0 {
        return Some(0);
    } else if n == 1 {
        return Some(1);
    }
    
    let mut a = 0i64;
    let mut b = 1i64;
    
    for _ in 2..=n {
        match a.checked_mul(2) {
            Some(doubled) => {
                match b.checked_add(doubled) {
                    Some(next) => {
                        a = b;
                        b = next;
                    }
                    None => return None, // Overflow
                }
            }
            None => return None, // Overflow
        }
    }
    
    Some(b)
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Fibonacci number function
/// Syntax: FibonacciNumber[n]
pub fn fibonacci_number_fn(args: &[Value]) -> VmResult<Value> {
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
    
    match fibonacci_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Fibonacci number F({}) too large", n))),
    }
}

/// Lucas number function
/// Syntax: LucasNumber[n]
pub fn lucas_number_fn(args: &[Value]) -> VmResult<Value> {
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
    
    match lucas_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Lucas number L({}) too large", n))),
    }
}

/// Tribonacci number function
/// Syntax: TribonacciNumber[n]
pub fn tribonacci_number_fn(args: &[Value]) -> VmResult<Value> {
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
    
    match tribonacci_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Tribonacci number T({}) too large", n))),
    }
}

/// Pell number function
/// Syntax: PellNumber[n]
pub fn pell_number_fn(args: &[Value]) -> VmResult<Value> {
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
    
    match pell_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Pell number P({}) too large", n))),
    }
}

/// Jacobsthal number function
/// Syntax: JacobsthalNumber[n]
pub fn jacobsthal_number_fn(args: &[Value]) -> VmResult<Value> {
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
    
    match jacobsthal_number(n) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime(format!("Jacobsthal number J({}) too large", n))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_fibonacci_number() {
        // Test basic Fibonacci sequence
        assert_eq!(fibonacci_number(0), Some(0));
        assert_eq!(fibonacci_number(1), Some(1));
        assert_eq!(fibonacci_number(2), Some(1));
        assert_eq!(fibonacci_number(3), Some(2));
        assert_eq!(fibonacci_number(4), Some(3));
        assert_eq!(fibonacci_number(5), Some(5));
        assert_eq!(fibonacci_number(6), Some(8));
        assert_eq!(fibonacci_number(7), Some(13));
        assert_eq!(fibonacci_number(10), Some(55));
        
        // Test negative Fibonacci (extended definition)
        assert_eq!(fibonacci_number(-1), Some(1));
        assert_eq!(fibonacci_number(-2), Some(-1));
        assert_eq!(fibonacci_number(-3), Some(2));
        assert_eq!(fibonacci_number(-4), Some(-3));
    }
    
    #[test]
    fn test_lucas_number() {
        // Test Lucas sequence
        assert_eq!(lucas_number(0), Some(2));
        assert_eq!(lucas_number(1), Some(1));
        assert_eq!(lucas_number(2), Some(3));
        assert_eq!(lucas_number(3), Some(4));
        assert_eq!(lucas_number(4), Some(7));
        assert_eq!(lucas_number(5), Some(11));
        assert_eq!(lucas_number(6), Some(18));
        assert_eq!(lucas_number(7), Some(29));
        
        // Test negative Lucas
        assert_eq!(lucas_number(-1), Some(-1));
        assert_eq!(lucas_number(-2), Some(3));
        assert_eq!(lucas_number(-3), Some(-4));
    }
    
    #[test]
    fn test_tribonacci_number() {
        // Test Tribonacci sequence
        assert_eq!(tribonacci_number(0), Some(0));
        assert_eq!(tribonacci_number(1), Some(0));
        assert_eq!(tribonacci_number(2), Some(1));
        assert_eq!(tribonacci_number(3), Some(1));
        assert_eq!(tribonacci_number(4), Some(2));
        assert_eq!(tribonacci_number(5), Some(4));
        assert_eq!(tribonacci_number(6), Some(7));
        assert_eq!(tribonacci_number(7), Some(13));
        assert_eq!(tribonacci_number(8), Some(24));
    }
    
    #[test]
    fn test_pell_number() {
        // Test Pell sequence
        assert_eq!(pell_number(0), Some(0));
        assert_eq!(pell_number(1), Some(1));
        assert_eq!(pell_number(2), Some(2));
        assert_eq!(pell_number(3), Some(5));
        assert_eq!(pell_number(4), Some(12));
        assert_eq!(pell_number(5), Some(29));
        assert_eq!(pell_number(6), Some(70));
    }
    
    #[test]
    fn test_jacobsthal_number() {
        // Test Jacobsthal sequence
        assert_eq!(jacobsthal_number(0), Some(0));
        assert_eq!(jacobsthal_number(1), Some(1));
        assert_eq!(jacobsthal_number(2), Some(1));
        assert_eq!(jacobsthal_number(3), Some(3));
        assert_eq!(jacobsthal_number(4), Some(5));
        assert_eq!(jacobsthal_number(5), Some(11));
        assert_eq!(jacobsthal_number(6), Some(21));
    }
    
    #[test]
    fn test_fibonacci_lucas_identity() {
        // Test the identity: L(n) = F(n-1) + F(n+1)
        for n in 1..=10 {
            let lucas_n = lucas_number(n).unwrap();
            let fib_prev = fibonacci_number(n - 1).unwrap();
            let fib_next = fibonacci_number(n + 1).unwrap();
            assert_eq!(lucas_n, fib_prev + fib_next);
        }
    }
}