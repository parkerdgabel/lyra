//! Modular Arithmetic
//!
//! This module implements advanced modular arithmetic operations including
//! discrete logarithms, primitive roots, and multiplicative orders.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::assoc;
use std::any::Any;
use std::collections::HashMap;

/// Result from discrete logarithm computation
#[derive(Debug, Clone)]
pub struct DiscreteLogResult {
    /// The discrete logarithm (exponent)
    pub log: Option<i64>,
    /// Base used
    pub base: i64,
    /// Target value
    pub target: i64,
    /// Modulus
    pub modulus: i64,
    /// Whether computation was successful
    pub found: bool,
}

impl Foreign for DiscreteLogResult {
    fn type_name(&self) -> &'static str {
        "DiscreteLogResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Log" => Ok(self.log.map(Value::Integer).unwrap_or(Value::Integer(-1))),
            "Base" => Ok(Value::Integer(self.base)),
            "Target" => Ok(Value::Integer(self.target)),
            "Modulus" => Ok(Value::Integer(self.modulus)),
            "Found" => Ok(Value::Integer(if self.found { 1 } else { 0 })),
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

/// Fast modular exponentiation: (base^exp) mod m
pub fn power_mod(mut base: i64, mut exp: i64, m: i64) -> i64 {
    if m == 1 { return 0; }
    if exp == 0 { return 1; }
    if exp < 0 {
        // Handle negative exponents by finding modular inverse
        if let Some(inv) = modular_inverse(base, m) {
            return power_mod(inv, -exp, m);
        } else {
            return 0; // No inverse exists
        }
    }
    
    let mut result = 1;
    base = ((base % m) + m) % m;
    
    while exp > 0 {
        if exp % 2 == 1 {
            result = mod_mul(result, base, m);
        }
        exp >>= 1;
        base = mod_mul(base, base, m);
    }
    
    result
}

/// Modular multiplication with overflow protection
pub fn mod_mul(a: i64, b: i64, m: i64) -> i64 {
    ((a as i128 * b as i128) % m as i128) as i64
}

/// Extended Euclidean algorithm for modular inverse
pub fn modular_inverse(a: i64, m: i64) -> Option<i64> {
    let (gcd, x, _) = extended_gcd(a, m);
    if gcd != 1 {
        None
    } else {
        Some(((x % m) + m) % m)
    }
}

/// Extended GCD returning (gcd, x, y) where ax + my = gcd
fn extended_gcd(a: i64, m: i64) -> (i64, i64, i64) {
    if a == 0 {
        (m, 0, 1)
    } else {
        let (gcd, x1, y1) = extended_gcd(m % a, a);
        let x = y1 - (m / a) * x1;
        let y = x1;
        (gcd, x, y)
    }
}

/// Baby-step giant-step algorithm for discrete logarithm
/// Finds x such that base^x ≡ target (mod m)
pub fn discrete_log_baby_giant(base: i64, target: i64, m: i64) -> DiscreteLogResult {
    let mut result = DiscreteLogResult {
        log: None,
        base,
        target,
        modulus: m,
        found: false,
    };
    
    if m <= 1 { return result; }
    
    let base = ((base % m) + m) % m;
    let target = ((target % m) + m) % m;
    
    if base == 0 {
        if target == 0 {
            result.log = Some(1);
            result.found = true;
        }
        return result;
    }
    
    if target == 1 {
        result.log = Some(0);
        result.found = true;
        return result;
    }
    
    // Choose step size as ceiling(sqrt(m))
    let n = ((m as f64).sqrt() as i64) + 1;
    
    // Baby steps: store base^j mod m for j = 0, 1, ..., n-1
    let mut baby_steps = HashMap::new();
    let mut gamma = 1;
    for j in 0..n {
        if gamma == target {
            result.log = Some(j);
            result.found = true;
            return result;
        }
        baby_steps.insert(gamma, j);
        gamma = mod_mul(gamma, base, m);
    }
    
    // Giant steps: compute target * (base^(-n))^i for i = 0, 1, ..., n-1
    if let Some(base_inv_n) = modular_inverse(power_mod(base, n, m), m) {
        let mut y = target;
        for i in 0..n {
            if let Some(&j) = baby_steps.get(&y) {
                let log = i * n + j;
                result.log = Some(log);
                result.found = true;
                return result;
            }
            y = mod_mul(y, base_inv_n, m);
        }
    }
    
    result
}

/// Test if a is a quadratic residue modulo p (odd prime)
pub fn is_quadratic_residue(a: i64, p: i64) -> bool {
    if p == 2 { return true; }
    if a % p == 0 { return true; }
    
    // Use Euler's criterion: a^((p-1)/2) ≡ 1 (mod p) iff a is QR
    power_mod(a, (p - 1) / 2, p) == 1
}

/// Find a primitive root modulo p (where p is prime)
/// A primitive root g has multiplicative order φ(p) = p-1
pub fn find_primitive_root(p: i64) -> Option<i64> {
    if p <= 1 { return None; }
    if p == 2 { return Some(1); }
    
    // For primitive roots to exist, we need p to be of the form 2, 4, p^k, or 2*p^k
    // For now, we'll assume p is prime and find a primitive root
    
    let phi = p - 1; // φ(p) for prime p
    let prime_factors = find_prime_factors(phi);
    
    for g in 2..p {
        if gcd(g, p) != 1 { continue; }
        
        let mut is_primitive = true;
        for &factor in &prime_factors {
            if power_mod(g, phi / factor, p) == 1 {
                is_primitive = false;
                break;
            }
        }
        
        if is_primitive {
            return Some(g);
        }
    }
    
    None
}

/// Find multiplicative order of a modulo n
/// Returns the smallest positive integer k such that a^k ≡ 1 (mod n)
pub fn multiplicative_order(a: i64, n: i64) -> Option<i64> {
    if n <= 1 || gcd(a, n) != 1 { return None; }
    
    let a = ((a % n) + n) % n;
    let mut order = 1;
    let mut current = a;
    
    while current != 1 && order <= n {
        current = mod_mul(current, a, n);
        order += 1;
    }
    
    if current == 1 { Some(order) } else { None }
}

/// Simple factorization to find prime factors (for small numbers)
fn find_prime_factors(mut n: i64) -> Vec<i64> {
    let mut factors = Vec::new();
    
    // Check for factor 2
    if n % 2 == 0 {
        factors.push(2);
        while n % 2 == 0 {
            n /= 2;
        }
    }
    
    // Check for odd factors
    let mut f = 3;
    while f * f <= n {
        if n % f == 0 {
            factors.push(f);
            while n % f == 0 {
                n /= f;
            }
        }
        f += 2;
    }
    
    // If n is still > 1, then it's a prime factor
    if n > 1 {
        factors.push(n);
    }
    
    factors
}

/// GCD implementation
fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Fast modular exponentiation
/// Syntax: PowerMod[a, b, m]
pub fn power_mod_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (base, exponent, modulus)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let base = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for base".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let exp = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for exponent".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let modulus = match &args[2] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for modulus".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    if modulus <= 0 {
        return Err(VmError::Runtime("Modulus must be positive".to_string()));
    }
    
    Ok(Value::Integer(power_mod(base, exp, modulus)))
}

/// Modular multiplicative inverse
/// Syntax: ModularInverse[a, m]
pub fn modular_inverse_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (a, modulus)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let a = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for a".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let m = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for modulus".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    if m <= 0 {
        return Err(VmError::Runtime("Modulus must be positive".to_string()));
    }
    
    match modular_inverse(a, m) {
        Some(inv) => Ok(Value::Integer(inv)),
        None => Err(VmError::Runtime(format!("No modular inverse exists for {} mod {}", a, m))),
    }
}

/// Discrete logarithm computation
/// Syntax: DiscreteLog[a, b, m] finds x such that a^x ≡ b (mod m)
pub fn discrete_log_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (base, target, modulus)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let base = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for base".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let target = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for target".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let modulus = match &args[2] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for modulus".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    if modulus <= 0 {
        return Err(VmError::Runtime("Modulus must be positive".to_string()));
    }
    
    let result = discrete_log_baby_giant(base, target, modulus);
    Ok(assoc(vec![
        ("log", match result.log { Some(x) => Value::Integer(x), None => Value::String("None".to_string()) }),
        ("base", Value::Integer(base)),
        ("target", Value::Integer(target)),
        ("modulus", Value::Integer(modulus)),
        ("found", Value::Boolean(result.found)),
        ("method", Value::String("BabyStepGiantStep".to_string())),
    ]))
}

/// Quadratic residue test
/// Syntax: QuadraticResidue[a, p]
pub fn quadratic_residue_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (a, p)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let a = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for a".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let p = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for p".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    if p <= 0 {
        return Err(VmError::Runtime("p must be positive".to_string()));
    }
    
    Ok(Value::Integer(if is_quadratic_residue(a, p) { 1 } else { 0 }))
}

/// Find primitive root
/// Syntax: PrimitiveRoot[p]
pub fn primitive_root_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (p)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let p = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for p".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if p <= 0 {
        return Err(VmError::Runtime("p must be positive".to_string()));
    }
    
    match find_primitive_root(p) {
        Some(root) => Ok(Value::Integer(root)),
        None => Err(VmError::Runtime(format!("No primitive root exists for {}", p))),
    }
}

/// Multiplicative order
/// Syntax: MultOrder[a, n]
pub fn mult_order_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (a, n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let a = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for a".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let n = match &args[1] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    if n <= 0 {
        return Err(VmError::Runtime("n must be positive".to_string()));
    }
    
    match multiplicative_order(a, n) {
        Some(order) => Ok(Value::Integer(order)),
        None => Err(VmError::Runtime(format!("No multiplicative order exists for {} mod {}", a, n))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_power_mod() {
        assert_eq!(power_mod(2, 10, 1000), 1024 % 1000);
        assert_eq!(power_mod(3, 4, 7), 4); // 3^4 = 81 ≡ 4 (mod 7)
        assert_eq!(power_mod(2, 0, 5), 1);
        assert_eq!(power_mod(5, 1, 7), 5);
    }
    
    #[test]
    fn test_modular_inverse() {
        assert_eq!(modular_inverse(3, 7), Some(5)); // 3 * 5 ≡ 1 (mod 7)
        assert_eq!(modular_inverse(2, 6), None); // gcd(2, 6) = 2 ≠ 1
        assert_eq!(modular_inverse(17, 101), Some(6)); // 17 * 6 ≡ 1 (mod 101)
    }
    
    #[test]
    fn test_discrete_log() {
        // 2^x ≡ 8 (mod 13) => x = 3 (since 2^3 = 8)
        let result = discrete_log_baby_giant(2, 8, 13);
        assert!(result.found);
        assert_eq!(result.log, Some(3));
        
        // 3^x ≡ 1 (mod 7) => x = 0
        let result = discrete_log_baby_giant(3, 1, 7);
        assert!(result.found);
        assert_eq!(result.log, Some(0));
    }
    
    #[test]
    fn test_quadratic_residue() {
        assert!(is_quadratic_residue(1, 7)); // 1 is always QR
        assert!(is_quadratic_residue(4, 7)); // 2^2 ≡ 4 (mod 7)
        assert!(!is_quadratic_residue(3, 7)); // 3 is not a QR mod 7
        assert!(is_quadratic_residue(9, 7)); // 3^2 ≡ 2 (mod 7), so 9 ≡ 2 is QR
    }
    
    #[test]
    fn test_primitive_root() {
        // 3 is a primitive root mod 7
        let root = find_primitive_root(7);
        assert!(root.is_some());
        let g = root.unwrap();
        assert_eq!(multiplicative_order(g, 7), Some(6)); // φ(7) = 6
    }
    
    #[test]
    fn test_multiplicative_order() {
        assert_eq!(multiplicative_order(2, 7), Some(3)); // 2^3 ≡ 1 (mod 7)
        assert_eq!(multiplicative_order(3, 7), Some(6)); // 3 is primitive root mod 7
        assert_eq!(multiplicative_order(1, 5), Some(1)); // 1^1 ≡ 1 (mod 5)
    }
    
    #[test]
    fn test_find_prime_factors() {
        assert_eq!(find_prime_factors(12), vec![2, 3]);
        assert_eq!(find_prime_factors(30), vec![2, 3, 5]);
        assert_eq!(find_prime_factors(17), vec![17]);
    }
}
