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

// Re-export all public functions and types
pub use primes::*;
pub use algebraic::*;
pub use modular::*;
pub use crypto::*;