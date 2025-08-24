//! Numerical Differentiation Methods
//!
//! This module implements numerical differentiation algorithms for computing
//! derivatives when analytical methods are not available or practical.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::common::result::derivative_result;
use std::any::Any;

/// Results from numerical differentiation
#[derive(Debug, Clone)]
pub struct DerivativeResult {
    /// The computed derivative value
    pub value: f64,
    /// Estimated error (when available)
    pub error_estimate: f64,
    /// Step size used for computation
    pub step_size: f64,
    /// Method used for differentiation
    pub method: String,
    /// Order of the derivative
    pub order: usize,
}

impl Foreign for DerivativeResult {
    fn type_name(&self) -> &'static str {
        "DerivativeResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Value" => Ok(Value::Real(self.value)),
            "ErrorEstimate" => Ok(Value::Real(self.error_estimate)),
            "StepSize" => Ok(Value::Real(self.step_size)),
            "Method" => Ok(Value::String(self.method.clone())),
            "Order" => Ok(Value::Integer(self.order as i64)),
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

/// Forward finite difference for first derivative
///
/// f'(x) ≈ [f(x+h) - f(x)] / h
pub fn forward_difference(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    let fx = evaluate_function(f, x)?;
    let fxh = evaluate_function(f, x + h)?;
    
    let derivative = (fxh - fx) / h;
    
    // Error estimate: O(h) truncation error
    let error_estimate = h.abs();
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Forward Difference".to_string(),
        order: 1,
    })
}

/// Backward finite difference for first derivative
///
/// f'(x) ≈ [f(x) - f(x-h)] / h
pub fn backward_difference(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    let fx = evaluate_function(f, x)?;
    let fxmh = evaluate_function(f, x - h)?;
    
    let derivative = (fx - fxmh) / h;
    
    // Error estimate: O(h) truncation error
    let error_estimate = h.abs();
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Backward Difference".to_string(),
        order: 1,
    })
}

/// Central finite difference for first derivative
///
/// f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
pub fn central_difference(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    let fxh = evaluate_function(f, x + h)?;
    let fxmh = evaluate_function(f, x - h)?;
    
    let derivative = (fxh - fxmh) / (2.0 * h);
    
    // Error estimate: O(h²) truncation error (better than forward/backward)
    let error_estimate = h * h;
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Central Difference".to_string(),
        order: 1,
    })
}

/// Five-point stencil for first derivative (higher accuracy)
///
/// f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
pub fn five_point_stencil(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    let fx2h = evaluate_function(f, x + 2.0 * h)?;
    let fxh = evaluate_function(f, x + h)?;
    let fxmh = evaluate_function(f, x - h)?;
    let fxm2h = evaluate_function(f, x - 2.0 * h)?;
    
    let derivative = (-fx2h + 8.0 * fxh - 8.0 * fxmh + fxm2h) / (12.0 * h);
    
    // Error estimate: O(h⁴) truncation error
    let error_estimate = h.powi(4);
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Five-Point Stencil".to_string(),
        order: 1,
    })
}

/// Second derivative using central differences
///
/// f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
pub fn second_derivative(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    let fxh = evaluate_function(f, x + h)?;
    let fx = evaluate_function(f, x)?;
    let fxmh = evaluate_function(f, x - h)?;
    
    let derivative = (fxh - 2.0 * fx + fxmh) / (h * h);
    
    // Error estimate: O(h²) truncation error
    let error_estimate = h * h;
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Central Difference (2nd order)".to_string(),
        order: 2,
    })
}

/// Richardson extrapolation for improved accuracy
///
/// Uses multiple step sizes to reduce truncation error
pub fn richardson_extrapolation(f: &Value, x: f64, h: f64) -> Result<DerivativeResult, String> {
    // Compute derivatives with step size h and h/2
    let d1 = central_difference(f, x, h)?;
    let d2 = central_difference(f, x, h / 2.0)?;
    
    // Richardson extrapolation formula for central differences
    // D = (4*D(h/2) - D(h)) / 3
    let derivative = (4.0 * d2.value - d1.value) / 3.0;
    
    // Error estimate based on the difference
    let error_estimate = (d1.value - d2.value).abs() / 3.0;
    
    Ok(DerivativeResult {
        value: derivative,
        error_estimate,
        step_size: h,
        method: "Richardson Extrapolation".to_string(),
        order: 1,
    })
}

/// Adaptive step size for optimal finite difference
///
/// Finds optimal step size balancing truncation and roundoff errors
pub fn adaptive_finite_difference(f: &Value, x: f64, tolerance: f64, max_iterations: usize) -> Result<DerivativeResult, String> {
    let mut h = 1e-3; // Initial step size
    let mut best_derivative = 0.0;
    let mut best_error = f64::INFINITY;
    let mut best_h = h;
    
    for _ in 0..max_iterations {
        let result = richardson_extrapolation(f, x, h)?;
        
        if result.error_estimate < best_error {
            best_derivative = result.value;
            best_error = result.error_estimate;
            best_h = h;
        }
        
        if best_error < tolerance {
            break;
        }
        
        h *= 0.5; // Reduce step size
        
        // Avoid extremely small step sizes that cause roundoff errors
        if h < 1e-15 {
            break;
        }
    }
    
    Ok(DerivativeResult {
        value: best_derivative,
        error_estimate: best_error,
        step_size: best_h,
        method: "Adaptive Richardson".to_string(),
        order: 1,
    })
}

/// Gradient computation for multivariable functions
///
/// Computes partial derivatives for all variables
pub fn gradient(f: &Value, x: &[f64], h: f64) -> Result<Vec<f64>, String> {
    let mut grad = Vec::with_capacity(x.len());
    
    // For polynomial functions, we need to handle multivariate case
    // For now, we'll treat this as univariate for each variable
    for i in 0..x.len() {
        // Create temporary function for partial derivative w.r.t. variable i
        // This is a simplified approach - real implementation would handle
        // true multivariate functions
        let mut x_plus = x.to_vec();
        let mut x_minus = x.to_vec();
        x_plus[i] += h;
        x_minus[i] -= h;
        
        // For polynomial case, evaluate at specific point
        let f_plus = evaluate_function(f, x_plus[i])?;
        let f_minus = evaluate_function(f, x_minus[i])?;
        
        let partial = (f_plus - f_minus) / (2.0 * h);
        grad.push(partial);
    }
    
    Ok(grad)
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Finite difference derivative
/// Syntax: FiniteDifference[f, x, h, method, order]
pub fn finite_difference(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "2-5 arguments (function, x, [h], [method], [order])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let x = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let h = if args.len() > 2 {
        match &args[2] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real number for h".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        1e-6
    };
    
    let method = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => s.as_str(),
            _ => return Err(VmError::TypeError {
                expected: "String for method".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else {
        "Central"
    };
    
    let order = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for order".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        1
    };
    
    let result = match (method, order) {
        ("Forward", 1) => forward_difference(f, x, h),
        ("Backward", 1) => backward_difference(f, x, h),
        ("Central", 1) => central_difference(f, x, h),
        ("FivePoint", 1) => five_point_stencil(f, x, h),
        ("Central", 2) => second_derivative(f, x, h),
        _ => return Err(VmError::Runtime(format!("Unsupported method/order combination: {}/{}", method, order))),
    };
    
    match result {
        Ok(res) => Ok(derivative_result(res.value, res.error_estimate, res.step_size, &res.method, res.order)),
        Err(e) => Err(VmError::Runtime(format!("Finite difference failed: {}", e))),
    }
}

/// Richardson extrapolation for improved derivatives
/// Syntax: RichardsonExtrapolation[f, x, h]
pub fn richardson_extrapolation_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (function, x, [h])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let x = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let h = if args.len() > 2 {
        match &args[2] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real number for h".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        1e-4
    };
    
    match richardson_extrapolation(f, x, h) {
        Ok(res) => Ok(derivative_result(res.value, res.error_estimate, res.step_size, &res.method, res.order)),
        Err(e) => Err(VmError::Runtime(format!("Richardson extrapolation failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_central_difference() {
        // Test derivative of x^2, which should be 2x
        // At x = 2, derivative should be 4
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = central_difference(&f, 2.0, 1e-6).unwrap();
        assert!((result.value - 4.0).abs() < 1e-5);
    }
    
    #[test]
    fn test_second_derivative() {
        // Test second derivative of x^2, which should be 2
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = second_derivative(&f, 1.0, 1e-4).unwrap();
        assert!((result.value - 2.0).abs() < 1e-3);
    }
    
    #[test]
    fn test_five_point_stencil() {
        // Test derivative of x^3, which should be 3x^2
        // At x = 2, derivative should be 12
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(0.0),  // x^2 term
            Value::Real(1.0),  // x^3 term
        ]);
        
        let result = five_point_stencil(&f, 2.0, 1e-4).unwrap();
        assert!((result.value - 12.0).abs() < 1e-6);
    }
    
    #[test]
    fn test_richardson_extrapolation() {
        // Test derivative of x^2 at x = 3, should be 6
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = richardson_extrapolation(&f, 3.0, 1e-3).unwrap();
        assert!((result.value - 6.0).abs() < 1e-8);
    }
    
    #[test]
    fn test_forward_vs_central() {
        // Compare forward and central differences
        let f = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(1.0),  // x term (derivative should be 1)
        ]);
        
        let forward = forward_difference(&f, 1.0, 1e-6).unwrap();
        let central = central_difference(&f, 1.0, 1e-6).unwrap();
        
        // Central difference should be more accurate
        assert!((central.value - 1.0).abs() < (forward.value - 1.0).abs());
    }
}
