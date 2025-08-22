//! Numerical Integration Methods
//!
//! This module implements fundamental numerical integration algorithms for computing
//! definite integrals. These methods are essential for scientific computing when
//! analytical integration is not possible or practical.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Results from numerical integration
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// The computed integral value
    pub value: f64,
    /// Estimated error (when available)
    pub error_estimate: f64,
    /// Number of function evaluations
    pub function_evaluations: usize,
    /// Method used for integration
    pub method: String,
    /// Whether the requested tolerance was achieved
    pub converged: bool,
}

impl Foreign for IntegrationResult {
    fn type_name(&self) -> &'static str {
        "IntegrationResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Value" => Ok(Value::Real(self.value)),
            "ErrorEstimate" => Ok(Value::Real(self.error_estimate)),
            "FunctionEvaluations" => Ok(Value::Integer(self.function_evaluations as i64)),
            "Method" => Ok(Value::String(self.method.clone())),
            "Converged" => Ok(Value::Integer(if self.converged { 1 } else { 0 })),
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

/// Helper function to evaluate function from Lyra expression
fn evaluate_function(expr: &Value, x: f64) -> Result<f64, String> {
    match expr {
        Value::List(coeffs) => {
            // Treat as polynomial coefficients [a0, a1, a2, ...] for a0 + a1*x + a2*x^2 + ...
            let mut result = 0.0;
            let mut power = 1.0;
            for coeff in coeffs {
                match coeff {
                    Value::Real(c) => {
                        result += c * power;
                        power *= x;
                    }
                    Value::Integer(c) => {
                        result += (*c as f64) * power;
                        power *= x;
                    }
                    _ => return Err("Invalid coefficient in polynomial".to_string()),
                }
            }
            Ok(result)
        }
        _ => Err("Function must be represented as polynomial coefficients".to_string()),
    }
}

/// Trapezoidal rule for numerical integration
///
/// Approximates integral using linear interpolation between function values
pub fn trapezoidal_rule(f: &Value, a: f64, b: f64, n: usize) -> Result<IntegrationResult, String> {
    if n == 0 {
        return Err("Number of intervals must be positive".to_string());
    }
    
    let h = (b - a) / (n as f64);
    let mut sum = 0.0;
    let mut function_evaluations = 0;
    
    // First and last terms have coefficient 1/2
    let fa = evaluate_function(f, a)?;
    let fb = evaluate_function(f, b)?;
    sum += 0.5 * (fa + fb);
    function_evaluations += 2;
    
    // Middle terms have coefficient 1
    for i in 1..n {
        let x = a + (i as f64) * h;
        let fx = evaluate_function(f, x)?;
        sum += fx;
        function_evaluations += 1;
    }
    
    let integral = h * sum;
    
    // Simple error estimate based on step size
    let error_estimate = (b - a).powi(3) / (12.0 * (n as f64).powi(2)) * h;
    
    Ok(IntegrationResult {
        value: integral,
        error_estimate,
        function_evaluations,
        method: "Trapezoidal".to_string(),
        converged: true,
    })
}

/// Simpson's rule for numerical integration  
///
/// Uses quadratic interpolation for higher accuracy than trapezoidal rule
pub fn simpson_rule(f: &Value, a: f64, b: f64, n: usize) -> Result<IntegrationResult, String> {
    if n == 0 || n % 2 != 0 {
        return Err("Number of intervals must be positive and even for Simpson's rule".to_string());
    }
    
    let h = (b - a) / (n as f64);
    let mut sum = 0.0;
    let mut function_evaluations = 0;
    
    // First and last terms
    let fa = evaluate_function(f, a)?;
    let fb = evaluate_function(f, b)?;
    sum += fa + fb;
    function_evaluations += 2;
    
    // Odd-indexed terms (coefficient 4)
    for i in (1..n).step_by(2) {
        let x = a + (i as f64) * h;
        let fx = evaluate_function(f, x)?;
        sum += 4.0 * fx;
        function_evaluations += 1;
    }
    
    // Even-indexed terms (coefficient 2)
    for i in (2..n).step_by(2) {
        let x = a + (i as f64) * h;
        let fx = evaluate_function(f, x)?;
        sum += 2.0 * fx;
        function_evaluations += 1;
    }
    
    let integral = (h / 3.0) * sum;
    
    // Error estimate for Simpson's rule
    let error_estimate = (b - a).powi(5) / (2880.0 * (n as f64).powi(4)) * h.powi(4);
    
    Ok(IntegrationResult {
        value: integral,
        error_estimate,
        function_evaluations,
        method: "Simpson".to_string(),
        converged: true,
    })
}

/// Romberg integration with Richardson extrapolation
///
/// Systematically improves trapezoidal estimates using extrapolation
pub fn romberg_integration(f: &Value, a: f64, b: f64, tolerance: f64, max_levels: usize) -> Result<IntegrationResult, String> {
    let mut r = vec![vec![0.0; max_levels]; max_levels];
    let mut function_evaluations = 0;
    
    // Start with trapezoidal rule for different step sizes
    for i in 0..max_levels {
        let n = 1_usize << i; // 2^i intervals
        let h = (b - a) / (n as f64);
        
        if i == 0 {
            // First trapezoidal estimate
            let fa = evaluate_function(f, a)?;
            let fb = evaluate_function(f, b)?;
            r[0][0] = 0.5 * h * (fa + fb);
            function_evaluations += 2;
        } else {
            // Use previous estimate and add midpoint values
            r[i][0] = 0.5 * r[i-1][0];
            let step = (b - a) / ((1_usize << i) as f64);
            
            for k in 1..(1_usize << (i-1)) + 1 {
                let x = a + (2*k - 1) as f64 * step;
                let fx = evaluate_function(f, x)?;
                r[i][0] += step * fx;
                function_evaluations += 1;
            }
        }
        
        // Richardson extrapolation
        for j in 1..=i {
            let factor = 4_f64.powi(j as i32);
            r[i][j] = (factor * r[i][j-1] - r[i-1][j-1]) / (factor - 1.0);
        }
        
        // Check convergence
        if i > 0 && (r[i][i] - r[i-1][i-1]).abs() < tolerance {
            return Ok(IntegrationResult {
                value: r[i][i],
                error_estimate: (r[i][i] - r[i-1][i-1]).abs(),
                function_evaluations,
                method: "Romberg".to_string(),
                converged: true,
            });
        }
    }
    
    Ok(IntegrationResult {
        value: r[max_levels-1][max_levels-1],
        error_estimate: (r[max_levels-1][max_levels-1] - r[max_levels-2][max_levels-2]).abs(),
        function_evaluations,
        method: "Romberg".to_string(),
        converged: false,
    })
}

/// Gaussian quadrature integration
///
/// Uses optimal node points and weights for polynomial integration
pub fn gauss_quadrature(f: &Value, a: f64, b: f64, n: usize) -> Result<IntegrationResult, String> {
    // Gauss-Legendre nodes and weights for different orders
    let (nodes, weights) = match n {
        1 => (vec![0.0], vec![2.0]),
        2 => (vec![-0.5773502691896257, 0.5773502691896257], vec![1.0, 1.0]),
        3 => (vec![-0.7745966692414834, 0.0, 0.7745966692414834], 
              vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4 => (vec![-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
              vec![0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
        5 => (vec![-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
              vec![0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
        _ => return Err("Gaussian quadrature only implemented for n=1,2,3,4,5".to_string()),
    };
    
    let mut integral = 0.0;
    let mut function_evaluations = 0;
    
    // Transform from [-1,1] to [a,b]
    let half_length = (b - a) / 2.0;
    let midpoint = (a + b) / 2.0;
    
    for (i, &xi) in nodes.iter().enumerate() {
        let x = midpoint + half_length * xi;
        let fx = evaluate_function(f, x)?;
        integral += weights[i] * fx;
        function_evaluations += 1;
    }
    
    integral *= half_length;
    
    // Rough error estimate (actual error depends on function derivatives)
    let error_estimate = integral * 1e-10; // Placeholder
    
    Ok(IntegrationResult {
        value: integral,
        error_estimate,
        function_evaluations,
        method: format!("Gauss-{}", n),
        converged: true,
    })
}

/// Monte Carlo integration
///
/// Uses random sampling for high-dimensional integration
pub fn monte_carlo_integration(f: &Value, a: f64, b: f64, n: usize, seed: Option<u64>) -> Result<IntegrationResult, String> {
    
    // Simple linear congruential generator for reproducible results
    let mut rng_state = seed.unwrap_or(12345);
    let mut random = || {
        rng_state = (rng_state.wrapping_mul(1664525).wrapping_add(1013904223)) % (1u64 << 32);
        rng_state as f64 / (1u64 << 32) as f64
    };
    
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut function_evaluations = 0;
    
    for _ in 0..n {
        let x = a + (b - a) * random();
        let fx = evaluate_function(f, x)?;
        sum += fx;
        sum_sq += fx * fx;
        function_evaluations += 1;
    }
    
    let integral = (b - a) * sum / (n as f64);
    
    // Error estimate based on sample variance
    let variance = (sum_sq / (n as f64) - (sum / (n as f64)).powi(2)) / (n as f64);
    let error_estimate = (b - a) * variance.sqrt();
    
    Ok(IntegrationResult {
        value: integral,
        error_estimate,
        function_evaluations,
        method: "Monte Carlo".to_string(),
        converged: true,
    })
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Trapezoidal rule integration
/// Syntax: Trapezoidal[f, a, b, n]
pub fn trapezoidal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (function, a, b, n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for a".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for b".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let n = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    match trapezoidal_rule(f, a, b, n) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Trapezoidal integration failed: {}", e))),
    }
}

/// Simpson's rule integration
/// Syntax: Simpson[f, a, b, n]
pub fn simpson(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (function, a, b, n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for a".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for b".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let n = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    match simpson_rule(f, a, b, n) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Simpson integration failed: {}", e))),
    }
}

/// Romberg integration
/// Syntax: Romberg[f, a, b, tolerance, maxLevels]
pub fn romberg(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, a, b, [tolerance], [maxLevels])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for a".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for b".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let tolerance = if args.len() > 3 {
        match &args[3] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real number for tolerance".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else {
        1e-10
    };
    
    let max_levels = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max levels".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        10
    };
    
    match romberg_integration(f, a, b, tolerance, max_levels) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Romberg integration failed: {}", e))),
    }
}

/// Gaussian quadrature integration
/// Syntax: GaussQuadrature[f, a, b, n]
pub fn gauss_quadrature_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "4 arguments (function, a, b, n)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for a".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for b".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let n = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    match gauss_quadrature(f, a, b, n) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Gauss quadrature failed: {}", e))),
    }
}

/// Monte Carlo integration
/// Syntax: MonteCarlo[f, a, b, n, seed]
pub fn monte_carlo(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "4-5 arguments (function, a, b, n, [seed])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for a".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for b".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let n = match &args[3] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for n".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    let seed = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => Some(*i as u64),
            _ => return Err(VmError::TypeError {
                expected: "Integer for seed".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        None
    };
    
    match monte_carlo_integration(f, a, b, n, seed) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Monte Carlo integration failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_trapezoidal_rule() {
        // Test integral of x^2 from 0 to 1, which should be 1/3
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = trapezoidal_rule(&f, 0.0, 1.0, 1000).unwrap();
        assert!((result.value - 1.0/3.0).abs() < 1e-3);
    }
    
    #[test]
    fn test_simpson_rule() {
        // Test integral of x^2 from 0 to 1, which should be 1/3
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = simpson_rule(&f, 0.0, 1.0, 1000).unwrap();
        assert!((result.value - 1.0/3.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_gauss_quadrature() {
        // Test integral of x^2 from 0 to 1 with 3-point Gauss quadrature
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = gauss_quadrature(&f, 0.0, 1.0, 3).unwrap();
        assert!((result.value - 1.0/3.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_monte_carlo_integration() {
        // Test integral of constant function 1 from 0 to 1, which should be 1
        let f = Value::List(vec![Value::Real(1.0)]);
        
        let result = monte_carlo_integration(&f, 0.0, 1.0, 10000, Some(12345)).unwrap();
        assert!((result.value - 1.0).abs() < 0.1); // Monte Carlo has larger error
    }
    
    #[test]
    fn test_romberg_integration() {
        // Test integral of x^2 from 0 to 1
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = romberg_integration(&f, 0.0, 1.0, 1e-10, 10).unwrap();
        assert!((result.value - 1.0/3.0).abs() < 1e-9);
        assert!(result.converged);
    }
}