//! Advanced Combinatorics & Discrete Mathematics
//!
//! This module implements sophisticated combinatorial algorithms and discrete mathematical
//! structures that form the foundation for symbolic computation in discrete mathematics,
//! algorithmic analysis, and combinatorial optimization.
//!
//! # Design Philosophy
//!
//! ## Symbolic Combinatorial Computation
//! - **Exact Arithmetic**: All computations maintain exactness for combinatorial identities
//! - **Efficient Algorithms**: Implements optimal algorithms for large parameter ranges
//! - **Mathematical Rigor**: Follows established combinatorial conventions and identities
//! - **Overflow Protection**: Proper handling of large combinatorial numbers
//!
//! ## Integration with Lyra Architecture
//! - **Foreign Object Pattern**: Complex combinatorial objects (partitions, sequences)
//! - **Symbolic Representation**: Combinatorial operations as mathematical expressions
//! - **Composition Support**: Functions compose naturally with other Lyra operations
//! - **Performance Optimization**: Leverages dynamic programming and memoization
//!
//! # Module Organization
//!
//! ## Phase 13B: Advanced Combinatorics (8 functions)
//!
//! ### Basic Combinatorial Functions (4 functions)
//! - **Binomial[n, k]**: Binomial coefficients with Pascal's triangle optimization
//! - **Multinomial[n, {k1, k2, ...}]**: Multinomial coefficients for multiset permutations
//! - **Permutations[n, k]**: k-permutations of n objects (nPk)
//! - **Combinations[n, k]**: k-combinations of n objects (nCk)
//!
//! ### Advanced Combinatorial Functions (4 functions)
//! - **StirlingNumber[n, k, type]**: Stirling numbers of first and second kind
//! - **BellNumber[n]**: Bell numbers (number of partitions of a set)
//! - **CatalanNumber[n]**: Catalan numbers for combinatorial structures
//! - **Partitions[n]**: Integer partition enumeration and generation
//!
//! # Usage Examples
//!
//! ## Basic Combinatorial Calculations
//! ```wolfram
//! (* Binomial coefficients *)
//! Binomial[10, 3]  (* → 120 *)
//! Binomial[100, 50]  (* Large numbers with overflow protection *)
//! 
//! (* Permutations and combinations *)
//! Permutations[5, 3]  (* → 60 *)
//! Combinations[5, 3]  (* → 10 *)
//! 
//! (* Multinomial coefficients *)
//! Multinomial[10, {3, 4, 3}]  (* → 4200 *)
//! ```
//!
//! ## Advanced Combinatorial Structures
//! ```wolfram
//! (* Stirling numbers *)
//! StirlingNumber[5, 3, 1]  (* First kind: unsigned *)
//! StirlingNumber[5, 3, 2]  (* Second kind: set partitions *)
//! 
//! (* Bell numbers *)
//! BellNumber[5]  (* → 52, number of set partitions *)
//! 
//! (* Catalan numbers *)
//! CatalanNumber[4]  (* → 14, binary trees with 4 internal nodes *)
//! 
//! (* Integer partitions *)
//! partitions = Partitions[5]  (* All partitions of 5 *)
//! Length[partitions]  (* → 7 partitions *)
//! ```
//!
//! ## Combinatorial Analysis
//! ```wolfram
//! (* Generate sequences *)
//! Table[Binomial[n, k], {n, 0, 10}, {k, 0, n}]  (* Pascal's triangle *)
//! Table[CatalanNumber[n], {n, 0, 10}]  (* Catalan sequence *)
//! 
//! (* Combinatorial identities *)
//! Sum[Binomial[n, k], {k, 0, n}] == 2^n  (* Binomial theorem *)
//! Sum[StirlingNumber[n, k, 2], {k, 0, n}] == BellNumber[n]  (* Bell identity *)
//! ```
//!
//! # Mathematical Background
//!
//! ## Binomial Coefficients
//! - Pascal's triangle recurrence: C(n,k) = C(n-1,k-1) + C(n-1,k)
//! - Symmetry property: C(n,k) = C(n,n-k)
//! - Overflow-safe computation for large parameters
//!
//! ## Stirling Numbers
//! - **First Kind**: Coefficients in rising factorial expansion
//! - **Second Kind**: Number of ways to partition n objects into k non-empty subsets
//! - Recurrence relations for efficient computation
//!
//! ## Bell Numbers
//! - Bell triangle for efficient computation
//! - Exponential generating function representation
//! - Connection to set partitions and Stirling numbers
//!
//! ## Catalan Numbers
//! - Formula: C_n = (1/(n+1)) * C(2n,n)
//! - Recurrence: C_n = sum(C_i * C_(n-1-i)) for i=0 to n-1
//! - Applications: binary trees, Dyck paths, polygon triangulations

pub mod basic;
pub mod advanced;
pub mod sequences;

// Re-export all public functions and types
pub use basic::*;
pub use advanced::*;
pub use sequences::*;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate combinatorics stdlib functions
pub fn register_combinatorics_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();
    // Basic combinatorial functions
    f.insert("Binomial".to_string(), basic::binomial_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("Multinomial".to_string(), basic::multinomial_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("Permutations".to_string(), basic::permutations_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("Combinations".to_string(), basic::combinations_fn as fn(&[Value]) -> VmResult<Value>);
    // Advanced combinatorial functions
    f.insert("StirlingNumber".to_string(), advanced::stirling_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("BellNumber".to_string(), advanced::bell_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("CatalanNumber".to_string(), advanced::catalan_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("Partitions".to_string(), advanced::partitions_fn as fn(&[Value]) -> VmResult<Value>);
    // Combinatorial sequences
    f.insert("FibonacciNumber".to_string(), sequences::fibonacci_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("LucasNumber".to_string(), sequences::lucas_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("TribonacciNumber".to_string(), sequences::tribonacci_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("PellNumber".to_string(), sequences::pell_number_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("JacobsthalNumber".to_string(), sequences::jacobsthal_number_fn as fn(&[Value]) -> VmResult<Value>);
    f
}
