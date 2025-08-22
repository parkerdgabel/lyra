//! Mathematics Module
//!
//! This module provides comprehensive mathematical capabilities including:
//! - Basic arithmetic and trigonometric functions
//! - Symbolic calculus (differentiation, integration)
//! - Special functions (gamma, elliptic, error functions)
//! - Differential equations solving
//! - Interpolation and approximation
//! - Linear algebra operations
//! - Optimization algorithms
//! - Signal processing

pub mod basic;
pub mod calculus;
pub mod special;
pub mod differential;
pub mod interpolation;
pub mod linear_algebra;
pub mod optimization;
pub mod signal;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Register all mathematics functions with the standard library
pub fn register_mathematics_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();

    // Basic mathematical functions (from basic.rs)
    functions.insert("Plus".to_string(), basic::plus as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Times".to_string(), basic::times as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Divide".to_string(), basic::divide as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Power".to_string(), basic::power as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Minus".to_string(), basic::minus as fn(&[Value]) -> VmResult<Value>);
    
    // Trigonometric functions
    functions.insert("Sin".to_string(), basic::sin as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Cos".to_string(), basic::cos as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Tan".to_string(), basic::tan as fn(&[Value]) -> VmResult<Value>);
    
    // Exponential and logarithmic functions
    functions.insert("Exp".to_string(), basic::exp as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Log".to_string(), basic::log as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Sqrt".to_string(), basic::sqrt as fn(&[Value]) -> VmResult<Value>);
    
    // Other mathematical functions
    functions.insert("Abs".to_string(), basic::abs as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Sign".to_string(), basic::sign as fn(&[Value]) -> VmResult<Value>);
    
    // Test functions
    functions.insert("TestHold".to_string(), basic::test_hold as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TestHoldMultiple".to_string(), basic::test_hold_multiple as fn(&[Value]) -> VmResult<Value>);

    // Calculus functions (from calculus.rs)
    functions.insert("D".to_string(), calculus::d as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Integrate".to_string(), calculus::integrate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IntegrateDefinite".to_string(), calculus::integrate_definite as fn(&[Value]) -> VmResult<Value>);

    // Special functions (from special.rs)
    functions.insert("Pi".to_string(), special::pi_constant as fn(&[Value]) -> VmResult<Value>);
    functions.insert("E".to_string(), special::e_constant as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EulerGamma".to_string(), special::euler_gamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GoldenRatio".to_string(), special::golden_ratio as fn(&[Value]) -> VmResult<Value>);
    
    // Gamma functions
    functions.insert("Gamma".to_string(), special::gamma_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LogGamma".to_string(), special::log_gamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Digamma".to_string(), special::digamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Polygamma".to_string(), special::polygamma as fn(&[Value]) -> VmResult<Value>);
    
    // Error functions
    functions.insert("Erf".to_string(), special::erf_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Erfc".to_string(), special::erfc_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("InverseErf".to_string(), special::inverse_erf as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FresnelC".to_string(), special::fresnel_c as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FresnelS".to_string(), special::fresnel_s as fn(&[Value]) -> VmResult<Value>);
    
    // Elliptic functions
    functions.insert("EllipticK".to_string(), special::elliptic_k as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EllipticE".to_string(), special::elliptic_e as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EllipticTheta".to_string(), special::elliptic_theta as fn(&[Value]) -> VmResult<Value>);
    
    // Hypergeometric functions
    functions.insert("Hypergeometric0F1".to_string(), special::hypergeometric_0f1 as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Hypergeometric1F1".to_string(), special::hypergeometric_1f1 as fn(&[Value]) -> VmResult<Value>);
    
    // Orthogonal polynomials
    functions.insert("ChebyshevT".to_string(), special::chebyshev_t as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ChebyshevU".to_string(), special::chebyshev_u as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GegenbauerC".to_string(), special::gegenbauer_c as fn(&[Value]) -> VmResult<Value>);

    // Note: Add other modules (differential, interpolation, linear_algebra, optimization, signal)
    // as they get integrated into the mathematics module

    functions
}

// Re-export public functions (avoiding conflicts)
// Note: Only re-exporting functions that actually exist
pub use basic::{
    divide, power, sqrt, sin, cos, tan, exp, log, random_real
};
pub use calculus::*;
pub use special::*;
// pub use differential::*;
// pub use interpolation::*;
// pub use linear_algebra::*;
// pub use optimization::*;
// pub use signal::*;