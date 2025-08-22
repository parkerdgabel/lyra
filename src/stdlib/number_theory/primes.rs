//! Prime Number Algorithms
//!
//! This module implements efficient algorithms for prime number operations,
//! including primality testing, prime generation, and prime-related functions.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Prime factorization result as a Foreign object
#[derive(Debug, Clone)]
pub struct PrimeFactorization {
    /// The original number that was factored
    pub number: i64,
    /// Prime factors with their exponents: [(prime, exponent), ...]
    pub factors: Vec<(i64, i64)>,
    /// Whether the factorization is complete
    pub complete: bool,
}

impl PrimeFactorization {
    pub fn new(number: i64) -> Self {
        Self {
            number,
            factors: Vec::new(),
            complete: false,
        }
    }
    
    /// Add a prime factor with its exponent
    pub fn add_factor(&mut self, prime: i64, exponent: i64) {
        self.factors.push((prime, exponent));
    }
    
    /// Check if the factorization accounts for the original number
    pub fn verify(&self) -> bool {
        let product: i64 = self.factors.iter()
            .map(|(p, e)| p.pow(*e as u32))
            .product();
        product == self.number.abs()
    }
    
    /// Get the number of distinct prime factors
    pub fn omega(&self) -> i64 {
        self.factors.len() as i64
    }
    
    /// Get the total number of prime factors (counting multiplicity)
    pub fn big_omega(&self) -> i64 {
        self.factors.iter().map(|(_, e)| e).sum()
    }
}

impl Foreign for PrimeFactorization {
    fn type_name(&self) -> &'static str {
        "PrimeFactorization"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Number" => Ok(Value::Integer(self.number)),
            "Factors" => {
                let factor_list: Vec<Value> = self.factors.iter()
                    .map(|(p, e)| {
                        if *e == 1 {
                            Value::Integer(*p)
                        } else {
                            Value::List(vec![Value::Integer(*p), Value::Integer(*e)])
                        }
                    })
                    .collect();
                Ok(Value::List(factor_list))
            }
            "Complete" => Ok(Value::Integer(if self.complete { 1 } else { 0 })),
            "Omega" => Ok(Value::Integer(self.omega())),
            "BigOmega" => Ok(Value::Integer(self.big_omega())),
            "Verify" => Ok(Value::Integer(if self.verify() { 1 } else { 0 })),
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

/// Miller-Rabin primality test
/// 
/// Probabilistic primality test with adjustable number of rounds.
/// For cryptographic applications, use at least 20 rounds.
pub fn miller_rabin_test(n: i64, rounds: usize) -> bool {
    if n < 2 { return false; }
    if n == 2 || n == 3 { return true; }
    if n % 2 == 0 { return false; }
    
    // Write n-1 as d * 2^r
    let mut d = n - 1;
    let mut r = 0;
    while d % 2 == 0 {
        d /= 2;
        r += 1;
    }
    
    // Simple LCG for deterministic testing (in production, use cryptographic RNG)
    let mut rng_state = 12345u64;
    let mut random = || {
        rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_state % (n as u64 - 2)) as i64 + 2
    };
    
    for _ in 0..rounds {
        let a = random();
        let mut x = mod_pow(a, d, n);
        
        if x == 1 || x == n - 1 {
            continue;
        }
        
        let mut composite = true;
        for _ in 0..r - 1 {
            x = mod_mul(x, x, n);
            if x == n - 1 {
                composite = false;
                break;
            }
        }
        
        if composite {
            return false;
        }
    }
    
    true
}

/// Fast modular exponentiation: (base^exp) mod m
pub fn mod_pow(mut base: i64, mut exp: i64, m: i64) -> i64 {
    if m == 1 { return 0; }
    
    let mut result = 1;
    base %= m;
    
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

/// Trial division for small primes
pub fn trial_division(mut n: i64, limit: i64) -> (Vec<(i64, i64)>, i64) {
    let mut factors = Vec::new();
    
    // Handle factor of 2
    if n % 2 == 0 {
        let mut exp = 0;
        while n % 2 == 0 {
            n /= 2;
            exp += 1;
        }
        factors.push((2, exp));
    }
    
    // Handle odd factors
    let mut p = 3;
    while p * p <= n && p <= limit {
        if n % p == 0 {
            let mut exp = 0;
            while n % p == 0 {
                n /= p;
                exp += 1;
            }
            factors.push((p, exp));
        }
        p += 2;
    }
    
    (factors, n)
}

/// Pollard's rho algorithm for factorization
pub fn pollard_rho(n: i64) -> Option<i64> {
    if n % 2 == 0 { return Some(2); }
    
    let f = |x| mod_mul(x, x, n) + 1;
    
    let mut x = 2;
    let mut y = 2;
    let mut d = 1;
    
    while d == 1 {
        x = f(x);
        y = f(f(y));
        d = gcd((x - y).abs(), n);
    }
    
    if d == n { None } else { Some(d) }
}

/// Greatest common divisor using Euclidean algorithm
pub fn gcd(mut a: i64, mut b: i64) -> i64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

/// Sieve of Eratosthenes for generating primes up to limit
pub fn sieve_of_eratosthenes(limit: usize) -> Vec<i64> {
    if limit < 2 { return Vec::new(); }
    
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    is_prime[1] = false;
    
    for i in 2..=((limit as f64).sqrt() as usize) {
        if is_prime[i] {
            for j in ((i * i)..=limit).step_by(i) {
                is_prime[j] = false;
            }
        }
    }
    
    is_prime.iter()
        .enumerate()
        .filter_map(|(i, &prime)| if prime { Some(i as i64) } else { None })
        .collect()
}

/// Euler's totient function
pub fn euler_phi(n: i64) -> i64 {
    if n <= 1 { return if n == 1 { 1 } else { 0 }; }
    
    let (factors, remaining) = trial_division(n, 1000);
    let mut result = n;
    
    // Apply formula: φ(n) = n * ∏(1 - 1/p) for each prime p dividing n
    for (p, _) in factors {
        result = result * (p - 1) / p;
    }
    
    // Handle remaining factor if it's prime
    if remaining > 1 && miller_rabin_test(remaining, 10) {
        result = result * (remaining - 1) / remaining;
    }
    
    result
}

/// Möbius function
pub fn moebius_mu(n: i64) -> i64 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    
    let (factors, remaining) = trial_division(n, 1000);
    
    // Check for square factors
    for (_, exp) in &factors {
        if *exp > 1 { return 0; }
    }
    
    let mut prime_count = factors.len();
    
    // Handle remaining factor
    if remaining > 1 {
        if miller_rabin_test(remaining, 10) {
            prime_count += 1;
        } else {
            // remaining is composite, so n has a square factor
            return 0;
        }
    }
    
    if prime_count % 2 == 0 { 1 } else { -1 }
}

/// Sum of divisors function σ_k(n)
pub fn divisor_sigma(k: i64, n: i64) -> i64 {
    if n <= 0 { return 0; }
    if n == 1 { return 1; }
    
    let (factors, remaining) = trial_division(n, 1000);
    let mut result = 1;
    
    for (p, exp) in factors {
        // σ_k(p^e) = (p^(k*(e+1)) - 1) / (p^k - 1)
        if k == 0 {
            result *= exp + 1;
        } else if k == 1 {
            result *= (p.pow((exp + 1) as u32) - 1) / (p - 1);
        } else {
            let pk = p.pow(k as u32);
            result *= (pk.pow((exp + 1) as u32) - 1) / (pk - 1);
        }
    }
    
    // Handle remaining factor if prime
    if remaining > 1 && miller_rabin_test(remaining, 10) {
        if k == 0 {
            result *= 2; // σ_0(p) = 2
        } else {
            result *= 1 + remaining.pow(k as u32); // σ_k(p) = 1 + p^k
        }
    }
    
    result
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Test if a number is prime
/// Syntax: PrimeQ[n]
pub fn prime_q(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n < 2 {
        return Ok(Value::Integer(0)); // False
    }
    
    let is_prime = miller_rabin_test(n, 20);
    Ok(Value::Integer(if is_prime { 1 } else { 0 }))
}

/// Find the next prime greater than n
/// Syntax: NextPrime[n]
pub fn next_prime(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut n = match &args[0] {
        Value::Integer(i) => *i + 1,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n < 2 { n = 2; }
    if n == 2 { return Ok(Value::Integer(2)); }
    if n % 2 == 0 { n += 1; }
    
    while !miller_rabin_test(n, 20) {
        n += 2;
    }
    
    Ok(Value::Integer(n))
}

/// Find the previous prime less than n
/// Syntax: PreviousPrime[n]
pub fn previous_prime(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut n = match &args[0] {
        Value::Integer(i) => *i - 1,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n < 2 {
        return Err(VmError::Runtime("No prime less than 2".to_string()));
    }
    if n == 2 { return Ok(Value::Integer(2)); }
    if n % 2 == 0 { n -= 1; }
    
    while n >= 3 && !miller_rabin_test(n, 20) {
        n -= 2;
    }
    
    if n < 2 {
        Err(VmError::Runtime("No prime found".to_string()))
    } else {
        Ok(Value::Integer(n))
    }
}

/// Count primes up to n using simple enumeration
/// Syntax: PrimePi[n]
pub fn prime_pi(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n < 2 { return Ok(Value::Integer(0)); }
    
    let primes = sieve_of_eratosthenes(n as usize);
    Ok(Value::Integer(primes.len() as i64))
}

/// Compute prime factorization
/// Syntax: PrimeFactorization[n]
pub fn prime_factorization(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if n.abs() < 2 {
        return Ok(Value::List(Vec::new()));
    }
    
    let mut factorization = PrimeFactorization::new(n);
    let mut remaining = n.abs();
    
    // Trial division up to 10000
    let (small_factors, rest) = trial_division(remaining, 10000);
    for (p, e) in small_factors {
        factorization.add_factor(p, e);
    }
    remaining = rest;
    
    // Use Pollard's rho for larger factors
    while remaining > 1 && !miller_rabin_test(remaining, 10) {
        if let Some(factor) = pollard_rho(remaining) {
            let mut exp = 0;
            while remaining % factor == 0 {
                remaining /= factor;
                exp += 1;
            }
            factorization.add_factor(factor, exp);
        } else {
            // Fallback to trial division
            break;
        }
    }
    
    // Handle remaining prime factor
    if remaining > 1 {
        factorization.add_factor(remaining, 1);
    }
    
    factorization.complete = true;
    Ok(Value::LyObj(LyObj::new(Box::new(factorization))))
}

/// Euler's totient function
/// Syntax: EulerPhi[n]
pub fn euler_phi_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Integer(euler_phi(n)))
}

/// Möbius function
/// Syntax: MoebiusMu[n]
pub fn moebius_mu_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (number)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let n = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Integer(moebius_mu(n)))
}

/// Sum of divisors function
/// Syntax: DivisorSigma[k, n]
pub fn divisor_sigma_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (k, n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let k = match &args[0] {
        Value::Integer(i) => *i,
        _ => return Err(VmError::TypeError {
            expected: "Integer for k".to_string(),
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
    
    Ok(Value::Integer(divisor_sigma(k, n)))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_miller_rabin() {
        assert!(miller_rabin_test(2, 10));
        assert!(miller_rabin_test(3, 10));
        assert!(!miller_rabin_test(4, 10));
        assert!(miller_rabin_test(5, 10));
        assert!(!miller_rabin_test(6, 10));
        assert!(miller_rabin_test(7, 10));
        assert!(!miller_rabin_test(8, 10));
        assert!(!miller_rabin_test(9, 10));
        assert!(!miller_rabin_test(10, 10));
        assert!(miller_rabin_test(11, 10));
    }
    
    #[test]
    fn test_trial_division() {
        let (factors, remaining) = trial_division(60, 100);
        assert_eq!(factors, vec![(2, 2), (3, 1), (5, 1)]);
        assert_eq!(remaining, 1);
    }
    
    #[test]
    fn test_euler_phi() {
        assert_eq!(euler_phi(1), 1);
        assert_eq!(euler_phi(2), 1);
        assert_eq!(euler_phi(3), 2);
        assert_eq!(euler_phi(4), 2);
        assert_eq!(euler_phi(5), 4);
        assert_eq!(euler_phi(6), 2);
        assert_eq!(euler_phi(7), 6);
        assert_eq!(euler_phi(8), 4);
        assert_eq!(euler_phi(9), 6);
        assert_eq!(euler_phi(10), 4);
    }
    
    #[test]
    fn test_moebius_mu() {
        assert_eq!(moebius_mu(1), 1);
        assert_eq!(moebius_mu(2), -1);
        assert_eq!(moebius_mu(3), -1);
        assert_eq!(moebius_mu(4), 0);  // 4 = 2^2 has square factor
        assert_eq!(moebius_mu(5), -1);
        assert_eq!(moebius_mu(6), 1);  // 6 = 2*3 has 2 distinct prime factors
        assert_eq!(moebius_mu(7), -1);
        assert_eq!(moebius_mu(8), 0);  // 8 = 2^3 has square factor
    }
    
    #[test]
    fn test_divisor_sigma() {
        // σ_0(6) = number of divisors of 6 = |{1,2,3,6}| = 4
        assert_eq!(divisor_sigma(0, 6), 4);
        
        // σ_1(6) = sum of divisors of 6 = 1+2+3+6 = 12
        assert_eq!(divisor_sigma(1, 6), 12);
        
        // σ_2(6) = sum of squares of divisors = 1+4+9+36 = 50
        assert_eq!(divisor_sigma(2, 6), 50);
    }
    
    #[test]
    fn test_prime_factorization_foreign() {
        let mut factorization = PrimeFactorization::new(60);
        factorization.add_factor(2, 2);
        factorization.add_factor(3, 1);
        factorization.add_factor(5, 1);
        factorization.complete = true;
        
        assert!(factorization.verify());
        assert_eq!(factorization.omega(), 3);
        assert_eq!(factorization.big_omega(), 4);
    }
}