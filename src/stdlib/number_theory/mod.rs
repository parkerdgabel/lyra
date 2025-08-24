//! Advanced Number Theory & Algebraic Structures
//!
//! This module implements sophisticated number-theoretic algorithms and algebraic structures
//! that form the foundation of symbolic computation in number theory, cryptography, and
//! abstract algebra.
//!
//! # Design Philosophy
//!
//! ## Symbolic Number Theory
//! - **Exact Arithmetic**: All computations maintain exactness where mathematically possible
//! - **Algorithmic Efficiency**: Implements asymptotically optimal algorithms (e.g., Miller-Rabin, Pollard's rho)
//! - **Mathematical Rigor**: Follows established number theory conventions and proofs
//! - **Cryptographic Grade**: Suitable for cryptographic applications with proper randomness
//!
//! ## Integration with Lyra Architecture
//! - **Foreign Object Pattern**: Complex number-theoretic objects (algebraic numbers, factorizations)
//! - **Symbolic Representation**: Number theory operations as mathematical expressions
//! - **Composition Support**: Functions compose naturally with other Lyra operations
//! - **Performance Optimization**: Leverages Rust's performance for computationally intensive algorithms
//!
//! # Module Organization
//!
//! ## Phase 13A: Core Number Theory (25 functions)
//!
//! ### Prime Number Algorithms (8 functions)
//! - **PrimeQ[n]**: Probabilistic primality testing using Miller-Rabin
//! - **NextPrime[n], PreviousPrime[n]**: Efficient prime navigation
//! - **PrimePi[n]**: Prime counting function with Meissel-Lehmer algorithm
//! - **PrimeFactorization[n]**: Hybrid Pollard's rho + trial division
//! - **EulerPhi[n]**: Euler's totient function via prime factorization
//! - **MoebiusMu[n]**: Möbius function computation
//! - **DivisorSigma[k, n]**: Generalized sum of divisors function
//!
//! ### Algebraic Number Theory (7 functions)
//! - **GCD[a, b, ...]**: Extended Euclidean algorithm for multiple inputs
//! - **LCM[a, b, ...]**: Least common multiple with exact computation
//! - **ChineseRemainder[{a1, a2, ...}, {m1, m2, ...}]**: Chinese Remainder Theorem
//! - **JacobiSymbol[a, n]**: Jacobi symbol computation
//! - **ContinuedFraction[x]**: Continued fraction expansion for rationals and algebraics
//! - **AlgebraicNumber[poly, approx]**: Algebraic number representation and arithmetic
//! - **MinimalPolynomial[α]**: Minimal polynomial computation for algebraic numbers
//!
//! ### Modular Arithmetic (6 functions)
//! - **PowerMod[a, b, m]**: Fast modular exponentiation using binary method
//! - **ModularInverse[a, m]**: Extended Euclidean algorithm for modular inverses
//! - **DiscreteLog[a, b, m]**: Baby-step giant-step discrete logarithm
//! - **QuadraticResidue[a, p]**: Quadratic residue testing with Legendre symbols
//! - **PrimitiveRoot[p]**: Primitive root computation for prime moduli
//! - **MultOrder[a, n]**: Multiplicative order computation
//!
//! ### Cryptographic Primitives (4 functions)
//! - **RSAGenerate[bits]**: Cryptographically secure RSA key generation
//! - **ECPoint[curve, x, y]**: Elliptic curve point arithmetic
//! - **HashFunction[data, algorithm]**: Cryptographic hash functions
//! - **RandomPrime[bits]**: Cryptographically secure prime generation
//!
//! # Usage Examples
//!
//! ## Prime Number Operations
//! ```wolfram
//! (* Test primality of large numbers *)
//! PrimeQ[2^127 - 1]  (* → True, Mersenne prime *)
//! 
//! (* Generate sequence of primes *)
//! primes = NestList[NextPrime, 2, 10]
//! 
//! (* Count primes up to a limit *)
//! PrimePi[1000]  (* → 168 *)
//! 
//! (* Factor large numbers *)
//! factors = PrimeFactorization[2^64 + 1]
//! ```
//!
//! ## Modular Arithmetic
//! ```wolfram
//! (* Fast modular exponentiation *)
//! PowerMod[2, 1000, 1009]
//! 
//! (* Solve modular equations *)
//! ModularInverse[17, 101]
//! 
//! (* Chinese Remainder Theorem *)
//! ChineseRemainder[{2, 3, 1}, {3, 4, 5}]
//! ```
//!
//! ## Algebraic Numbers
//! ```wolfram
//! (* Work with algebraic numbers *)
//! α = AlgebraicNumber[x^2 - 2, 1.414]  (* √2 *)
//! MinimalPolynomial[α]  (* → x^2 - 2 *)
//! 
//! (* Continued fractions *)
//! ContinuedFraction[GoldenRatio]  (* → {1, {1}} *)
//! ```
//!
//! ## Cryptographic Applications
//! ```wolfram
//! (* Generate RSA keypair *)
//! keypair = RSAGenerate[2048]
//! 
//! (* Elliptic curve operations *)
//! P = ECPoint[curve, x1, y1]
//! Q = ECPoint[curve, x2, y2]
//! R = P + Q  (* Point addition *)
//! ```

pub mod primes;
pub mod algebraic;
pub mod modular;
pub mod crypto;

// Re-export specific functions to avoid conflicts
pub use primes::{
    prime_q, next_prime, previous_prime, prime_pi, prime_factorization,
    euler_phi_fn, moebius_mu_fn, divisor_sigma_fn
};
pub use algebraic::*;
// Use modular functions from modular module (primary implementation)
pub use modular::{
    power_mod_fn, modular_inverse_fn, discrete_log_fn,
    quadratic_residue_fn, primitive_root_fn, mult_order_fn
};
// Use crypto functions with different names or selectively
pub use crypto::{
    rsa_generate, ec_point, hash_function, random_prime
};

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate number-theory stdlib functions
pub fn register_number_theory_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();

    // Prime number algorithms
    f.insert("PrimeQ".to_string(), primes::prime_q as fn(&[Value]) -> VmResult<Value>);
    f.insert("NextPrime".to_string(), primes::next_prime as fn(&[Value]) -> VmResult<Value>);
    f.insert("PreviousPrime".to_string(), primes::previous_prime as fn(&[Value]) -> VmResult<Value>);
    f.insert("PrimePi".to_string(), primes::prime_pi as fn(&[Value]) -> VmResult<Value>);
    f.insert("PrimeFactorization".to_string(), primes::prime_factorization as fn(&[Value]) -> VmResult<Value>);
    f.insert("EulerPhi".to_string(), primes::euler_phi_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("MoebiusMu".to_string(), primes::moebius_mu_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("DivisorSigma".to_string(), primes::divisor_sigma_fn as fn(&[Value]) -> VmResult<Value>);

    // Algebraic number theory
    f.insert("GCD".to_string(), algebraic::gcd_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("LCM".to_string(), algebraic::lcm_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("ChineseRemainder".to_string(), algebraic::chinese_remainder as fn(&[Value]) -> VmResult<Value>);
    f.insert("JacobiSymbol".to_string(), algebraic::jacobi_symbol_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContinuedFraction".to_string(), algebraic::continued_fraction_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("AlgebraicNumber".to_string(), algebraic::algebraic_number as fn(&[Value]) -> VmResult<Value>);
    f.insert("MinimalPolynomial".to_string(), algebraic::minimal_polynomial as fn(&[Value]) -> VmResult<Value>);

    // Modular arithmetic
    f.insert("PowerMod".to_string(), modular::power_mod_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("ModularInverse".to_string(), modular::modular_inverse_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("DiscreteLog".to_string(), modular::discrete_log_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("QuadraticResidue".to_string(), modular::quadratic_residue_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("PrimitiveRoot".to_string(), modular::primitive_root_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("MultOrder".to_string(), modular::mult_order_fn as fn(&[Value]) -> VmResult<Value>);

    // Cryptographic primitives
    f.insert("RSAGenerate".to_string(), crypto::rsa_generate as fn(&[Value]) -> VmResult<Value>);
    f.insert("ECPoint".to_string(), crypto::ec_point as fn(&[Value]) -> VmResult<Value>);
    f.insert("HashFunction".to_string(), crypto::hash_function as fn(&[Value]) -> VmResult<Value>);
    f.insert("RandomPrime".to_string(), crypto::random_prime as fn(&[Value]) -> VmResult<Value>);

    f
}
