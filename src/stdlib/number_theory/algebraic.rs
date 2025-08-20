//! Algebraic Number Theory
//!
//! This module implements algorithms for algebraic numbers, continued fractions,
//! and fundamental operations in algebraic number theory.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Extended GCD result
#[derive(Debug, Clone)]
pub struct ExtendedGcdResult {
    /// Greatest common divisor
    pub gcd: i64,
    /// Coefficient x such that ax + by = gcd(a, b)
    pub x: i64,
    /// Coefficient y such that ax + by = gcd(a, b)
    pub y: i64,
}

impl Foreign for ExtendedGcdResult {
    fn type_name(&self) -> &'static str {
        "ExtendedGcdResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "GCD" => Ok(Value::Integer(self.gcd)),
            "X" => Ok(Value::Integer(self.x)),
            "Y" => Ok(Value::Integer(self.y)),
            "Coefficients" => Ok(Value::List(vec![
                Value::Integer(self.x),
                Value::Integer(self.y)
            ])),
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

/// Continued fraction representation
#[derive(Debug, Clone)]
pub struct ContinuedFraction {
    /// Integer part
    pub integer_part: i64,
    /// Periodic part (empty if terminating)
    pub periodic_part: Vec<i64>,
    /// Non-periodic part after integer
    pub non_periodic_part: Vec<i64>,
    /// Whether the fraction is finite
    pub finite: bool,
}

impl ContinuedFraction {
    pub fn new(integer_part: i64) -> Self {
        Self {
            integer_part,
            periodic_part: Vec::new(),
            non_periodic_part: Vec::new(),
            finite: true,
        }
    }
    
    /// Convert back to a rational approximation
    pub fn to_rational(&self, max_terms: usize) -> (i64, i64) {
        if self.non_periodic_part.is_empty() {
            return (self.integer_part, 1);
        }
        
        let terms: Vec<i64> = std::iter::once(self.integer_part)
            .chain(self.non_periodic_part.iter().copied())
            .take(max_terms)
            .collect();
        
        if terms.len() <= 1 {
            return (self.integer_part, 1);
        }
        
        // Use continued fraction convergents
        let mut p_prev = 1;
        let mut p_curr = terms[0];
        let mut q_prev = 0;
        let mut q_curr = 1;
        
        for &a in &terms[1..] {
            let p_next = a * p_curr + p_prev;
            let q_next = a * q_curr + q_prev;
            
            p_prev = p_curr;
            p_curr = p_next;
            q_prev = q_curr;
            q_curr = q_next;
        }
        
        (p_curr, q_curr)
    }
}

impl Foreign for ContinuedFraction {
    fn type_name(&self) -> &'static str {
        "ContinuedFraction"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "IntegerPart" => Ok(Value::Integer(self.integer_part)),
            "NonPeriodicPart" => {
                let terms: Vec<Value> = self.non_periodic_part.iter()
                    .map(|&x| Value::Integer(x))
                    .collect();
                Ok(Value::List(terms))
            }
            "PeriodicPart" => {
                let terms: Vec<Value> = self.periodic_part.iter()
                    .map(|&x| Value::Integer(x))
                    .collect();
                Ok(Value::List(terms))
            }
            "Finite" => Ok(Value::Integer(if self.finite { 1 } else { 0 })),
            "ToRational" => {
                let (num, den) = self.to_rational(20);
                Ok(Value::List(vec![Value::Integer(num), Value::Integer(den)]))
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

/// Algebraic number representation
#[derive(Debug, Clone)]
pub struct AlgebraicNumber {
    /// Minimal polynomial coefficients (highest degree first)
    pub polynomial: Vec<i64>,
    /// Rational approximation
    pub approximation: f64,
    /// Degree of the algebraic number
    pub degree: usize,
}

impl AlgebraicNumber {
    pub fn new(polynomial: Vec<i64>, approximation: f64) -> Self {
        let degree = polynomial.len().saturating_sub(1);
        Self {
            polynomial,
            approximation,
            degree,
        }
    }
    
    /// Evaluate polynomial at a given point
    pub fn evaluate_polynomial(&self, x: f64) -> f64 {
        self.polynomial.iter()
            .fold(0.0, |acc, &coeff| acc * x + coeff as f64)
    }
    
    /// Check if this is a rational number (degree 1)
    pub fn is_rational(&self) -> bool {
        self.degree == 1
    }
    
    /// Get rational representation if applicable
    pub fn to_rational(&self) -> Option<(i64, i64)> {
        if self.is_rational() && self.polynomial.len() == 2 {
            // ax + b = 0 => x = -b/a
            let a = self.polynomial[0];
            let b = self.polynomial[1];
            if a != 0 {
                Some((-b, a))
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl Foreign for AlgebraicNumber {
    fn type_name(&self) -> &'static str {
        "AlgebraicNumber"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Polynomial" => {
                let coeffs: Vec<Value> = self.polynomial.iter()
                    .map(|&c| Value::Integer(c))
                    .collect();
                Ok(Value::List(coeffs))
            }
            "Approximation" => Ok(Value::Real(self.approximation)),
            "Degree" => Ok(Value::Integer(self.degree as i64)),
            "IsRational" => Ok(Value::Integer(if self.is_rational() { 1 } else { 0 })),
            "ToRational" => {
                if let Some((num, den)) = self.to_rational() {
                    Ok(Value::List(vec![Value::Integer(num), Value::Integer(den)]))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "Not a rational number".to_string(),
                    })
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

/// Extended Euclidean algorithm
pub fn extended_gcd(a: i64, b: i64) -> ExtendedGcdResult {
    if b == 0 {
        return ExtendedGcdResult {
            gcd: a.abs(),
            x: if a >= 0 { 1 } else { -1 },
            y: 0,
        };
    }
    
    let result = extended_gcd(b, a % b);
    ExtendedGcdResult {
        gcd: result.gcd,
        x: result.y,
        y: result.x - (a / b) * result.y,
    }
}

/// Greatest common divisor for multiple numbers
pub fn gcd_multiple(numbers: &[i64]) -> i64 {
    if numbers.is_empty() { return 0; }
    if numbers.len() == 1 { return numbers[0].abs(); }
    
    let mut result = numbers[0].abs();
    for &num in &numbers[1..] {
        result = gcd_two(result, num.abs());
        if result == 1 { break; }
    }
    result
}

/// GCD for two numbers
pub fn gcd_two(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

/// Least common multiple for multiple numbers
pub fn lcm_multiple(numbers: &[i64]) -> i64 {
    if numbers.is_empty() { return 0; }
    if numbers.len() == 1 { return numbers[0].abs(); }
    
    let mut result = numbers[0].abs();
    for &num in &numbers[1..] {
        if num == 0 { return 0; }
        result = lcm_two(result, num.abs());
    }
    result
}

/// LCM for two numbers
pub fn lcm_two(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 { return 0; }
    (a / gcd_two(a, b)) * b
}

/// Chinese Remainder Theorem solver
pub fn chinese_remainder_theorem(remainders: &[i64], moduli: &[i64]) -> Option<i64> {
    if remainders.len() != moduli.len() || remainders.is_empty() {
        return None;
    }
    
    // Check that moduli are pairwise coprime
    for i in 0..moduli.len() {
        for j in i + 1..moduli.len() {
            if gcd_two(moduli[i], moduli[j]) != 1 {
                return None;
            }
        }
    }
    
    let n: i64 = moduli.iter().product();
    let mut result = 0;
    
    for i in 0..remainders.len() {
        let ni = n / moduli[i];
        let mi = mod_inverse(ni, moduli[i])?;
        result += remainders[i] * ni * mi;
    }
    
    Some(((result % n) + n) % n)
}

/// Modular inverse using extended Euclidean algorithm
pub fn mod_inverse(a: i64, m: i64) -> Option<i64> {
    let result = extended_gcd(a, m);
    if result.gcd != 1 {
        None
    } else {
        Some(((result.x % m) + m) % m)
    }
}

/// Jacobi symbol computation
pub fn jacobi_symbol(mut a: i64, mut n: i64) -> i64 {
    if n <= 0 || n % 2 == 0 { return 0; }
    
    a %= n;
    let mut result = 1;
    
    while a != 0 {
        while a % 2 == 0 {
            a /= 2;
            let n_mod_8 = n % 8;
            if n_mod_8 == 3 || n_mod_8 == 5 {
                result = -result;
            }
        }
        
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 {
            result = -result;
        }
        a %= n;
    }
    
    if n == 1 { result } else { 0 }
}

/// Continued fraction expansion for a rational number
pub fn rational_to_continued_fraction(mut num: i64, mut den: i64) -> ContinuedFraction {
    if den == 0 {
        return ContinuedFraction::new(0);
    }
    
    let mut cf = ContinuedFraction::new(num / den);
    num %= den;
    
    while num != 0 {
        std::mem::swap(&mut num, &mut den);
        let quotient = num / den;
        cf.non_periodic_part.push(quotient);
        num %= den;
    }
    
    cf.finite = true;
    cf
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Greatest common divisor for multiple arguments
/// Syntax: GCD[a, b, ...]
pub fn gcd_fn(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::TypeError {
            expected: "At least 1 argument".to_string(),
            actual: "0 arguments".to_string(),
        });
    }
    
    let mut numbers = Vec::new();
    for arg in args {
        match arg {
            Value::Integer(i) => numbers.push(*i),
            _ => return Err(VmError::TypeError {
                expected: "Integer".to_string(),
                actual: format!("{:?}", arg),
            }),
        }
    }
    
    Ok(Value::Integer(gcd_multiple(&numbers)))
}

/// Least common multiple for multiple arguments
/// Syntax: LCM[a, b, ...]
pub fn lcm_fn(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() {
        return Err(VmError::TypeError {
            expected: "At least 1 argument".to_string(),
            actual: "0 arguments".to_string(),
        });
    }
    
    let mut numbers = Vec::new();
    for arg in args {
        match arg {
            Value::Integer(i) => numbers.push(*i),
            _ => return Err(VmError::TypeError {
                expected: "Integer".to_string(),
                actual: format!("{:?}", arg),
            }),
        }
    }
    
    Ok(Value::Integer(lcm_multiple(&numbers)))
}

/// Chinese Remainder Theorem solver
/// Syntax: ChineseRemainder[{a1, a2, ...}, {m1, m2, ...}]
pub fn chinese_remainder(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (remainders, moduli)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let remainders = match &args[0] {
        Value::List(list) => {
            let mut remainders = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => remainders.push(*i),
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for remainders".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            remainders
        }
        _ => return Err(VmError::TypeError {
            expected: "List for remainders".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let moduli = match &args[1] {
        Value::List(list) => {
            let mut moduli = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => moduli.push(*i),
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for moduli".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            moduli
        }
        _ => return Err(VmError::TypeError {
            expected: "List for moduli".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    match chinese_remainder_theorem(&remainders, &moduli) {
        Some(result) => Ok(Value::Integer(result)),
        None => Err(VmError::Runtime("No solution exists (moduli not pairwise coprime)".to_string())),
    }
}

/// Jacobi symbol computation
/// Syntax: JacobiSymbol[a, n]
pub fn jacobi_symbol_fn(args: &[Value]) -> VmResult<Value> {
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
    
    Ok(Value::Integer(jacobi_symbol(a, n)))
}

/// Continued fraction expansion
/// Syntax: ContinuedFraction[x] or ContinuedFraction[num, den]
pub fn continued_fraction_fn(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        1 => {
            // Single argument - could be Real or fraction
            match &args[0] {
                Value::Real(x) => {
                    // Convert real to rational approximation first
                    let (num, den) = real_to_rational(*x, 1000000);
                    let cf = rational_to_continued_fraction(num, den);
                    Ok(Value::LyObj(LyObj::new(Box::new(cf))))
                }
                Value::Integer(i) => {
                    let cf = ContinuedFraction::new(*i);
                    Ok(Value::LyObj(LyObj::new(Box::new(cf))))
                }
                _ => Err(VmError::TypeError {
                    expected: "Real or Integer".to_string(),
                    actual: format!("{:?}", args[0]),
                }),
            }
        }
        2 => {
            // Two arguments - numerator and denominator
            let num = match &args[0] {
                Value::Integer(i) => *i,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for numerator".to_string(),
                    actual: format!("{:?}", args[0]),
                }),
            };
            
            let den = match &args[1] {
                Value::Integer(i) => *i,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for denominator".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            };
            
            let cf = rational_to_continued_fraction(num, den);
            Ok(Value::LyObj(LyObj::new(Box::new(cf))))
        }
        _ => Err(VmError::TypeError {
            expected: "1-2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

/// Create an algebraic number
/// Syntax: AlgebraicNumber[polynomial, approximation]
pub fn algebraic_number(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (polynomial, approximation)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let polynomial = match &args[0] {
        Value::List(list) => {
            let mut coeffs = Vec::new();
            for item in list {
                match item {
                    Value::Integer(i) => coeffs.push(*i),
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for polynomial".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            coeffs
        }
        _ => return Err(VmError::TypeError {
            expected: "List for polynomial".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let approximation = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real or Integer for approximation".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let algebraic = AlgebraicNumber::new(polynomial, approximation);
    Ok(Value::LyObj(LyObj::new(Box::new(algebraic))))
}

/// Compute minimal polynomial (placeholder - complex algorithm)
/// Syntax: MinimalPolynomial[α]
pub fn minimal_polynomial(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (algebraic number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(algebraic) = obj.downcast_ref::<AlgebraicNumber>() {
                // For now, return the stored polynomial
                let coeffs: Vec<Value> = algebraic.polynomial.iter()
                    .map(|&c| Value::Integer(c))
                    .collect();
                Ok(Value::List(coeffs))
            } else {
                Err(VmError::TypeError {
                    expected: "AlgebraicNumber object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "AlgebraicNumber object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Convert real number to rational approximation using continued fractions
fn real_to_rational(x: f64, max_denominator: i64) -> (i64, i64) {
    if x.is_infinite() || x.is_nan() {
        return (0, 1);
    }
    
    let mut num = x.floor() as i64;
    let mut frac = x - x.floor();
    
    if frac < 1e-15 {
        return (num, 1);
    }
    
    let mut p0 = 1;
    let mut q0 = 0;
    let mut p1 = num;
    let mut q1 = 1;
    
    while q1 <= max_denominator && frac > 1e-15 {
        frac = 1.0 / frac;
        let a = frac.floor() as i64;
        frac -= frac.floor();
        
        let p2 = a * p1 + p0;
        let q2 = a * q1 + q0;
        
        if q2 > max_denominator {
            break;
        }
        
        p0 = p1;
        q0 = q1;
        p1 = p2;
        q1 = q2;
    }
    
    (p1, q1)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_extended_gcd() {
        let result = extended_gcd(240, 46);
        assert_eq!(result.gcd, 2);
        assert_eq!(240 * result.x + 46 * result.y, result.gcd);
    }
    
    #[test]
    fn test_gcd_multiple() {
        assert_eq!(gcd_multiple(&[48, 18, 24]), 6);
        assert_eq!(gcd_multiple(&[17, 13]), 1);
        assert_eq!(gcd_multiple(&[0, 5]), 5);
    }
    
    #[test]
    fn test_lcm_multiple() {
        assert_eq!(lcm_multiple(&[4, 6]), 12);
        assert_eq!(lcm_multiple(&[2, 3, 4]), 12);
        assert_eq!(lcm_multiple(&[7, 11]), 77);
    }
    
    #[test]
    fn test_chinese_remainder_theorem() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
        // Solution: x ≡ 23 (mod 105)
        let result = chinese_remainder_theorem(&[2, 3, 2], &[3, 5, 7]);
        assert_eq!(result, Some(23));
    }
    
    #[test]
    fn test_jacobi_symbol() {
        assert_eq!(jacobi_symbol(1, 3), 1);
        assert_eq!(jacobi_symbol(2, 3), -1);
        assert_eq!(jacobi_symbol(3, 5), -1);
        assert_eq!(jacobi_symbol(4, 5), 1);
    }
    
    #[test]
    fn test_continued_fraction() {
        let cf = rational_to_continued_fraction(22, 7);
        assert_eq!(cf.integer_part, 3);
        assert_eq!(cf.non_periodic_part, vec![7]);
        assert!(cf.finite);
    }
    
    #[test]
    fn test_real_to_rational() {
        let (num, den) = real_to_rational(0.75, 1000);
        assert_eq!((num, den), (3, 4));
        
        let (num, den) = real_to_rational(1.414213, 10000);
        // Should be close to √2 approximation
        assert!((num as f64 / den as f64 - 1.414213).abs() < 1e-6);
    }
}