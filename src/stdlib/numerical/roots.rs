//! Root Finding and Equation Solving Methods
//!
//! This module implements fundamental root finding algorithms for solving equations
//! of the form f(x) = 0. These methods are essential for scientific computing,
//! optimization, and engineering applications.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;

/// Parameters and results for root finding algorithms
#[derive(Debug, Clone)]
pub struct RootResult {
    /// The root value found
    pub root: f64,
    /// Number of iterations required
    pub iterations: usize,
    /// Final function value at the root
    pub function_value: f64,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Error estimate
    pub error_estimate: f64,
    /// Method used for finding the root
    pub method: String,
}

impl Foreign for RootResult {
    fn type_name(&self) -> &'static str {
        "RootResult"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Root" => Ok(Value::Real(self.root)),
            "Iterations" => Ok(Value::Integer(self.iterations as i64)),
            "FunctionValue" => Ok(Value::Real(self.function_value)),
            "Converged" => Ok(Value::Integer(if self.converged { 1 } else { 0 })),
            "ErrorEstimate" => Ok(Value::Real(self.error_estimate)),
            "Method" => Ok(Value::String(self.method.clone())),
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
    // For now, we'll work with simple polynomial representations
    // In a full implementation, this would evaluate arbitrary expressions
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

/// Bisection method for root finding
/// 
/// Finds a root of f(x) = 0 in the interval [a, b] where f(a) and f(b) have opposite signs
pub fn bisection_method(f: &Value, a: f64, b: f64, tolerance: f64, max_iterations: usize) -> Result<RootResult, String> {
    let mut left = a;
    let mut right = b;
    
    // Check that f(a) and f(b) have opposite signs
    let fa = evaluate_function(f, left)?;
    let fb = evaluate_function(f, right)?;
    
    if fa * fb > 0.0 {
        return Err("Function values at endpoints must have opposite signs".to_string());
    }
    
    let mut iterations = 0;
    let mut root = (left + right) / 2.0;
    let mut function_value = evaluate_function(f, root)?;
    
    while (right - left).abs() > tolerance && iterations < max_iterations {
        let fc = evaluate_function(f, root)?;
        
        if fa * fc < 0.0 {
            right = root;
        } else {
            left = root;
        }
        
        root = (left + right) / 2.0;
        function_value = evaluate_function(f, root)?;
        iterations += 1;
    }
    
    Ok(RootResult {
        root,
        iterations,
        function_value,
        converged: (right - left).abs() <= tolerance,
        error_estimate: (right - left).abs() / 2.0,
        method: "Bisection".to_string(),
    })
}

/// Newton-Raphson method for root finding
///
/// Uses both function and derivative for rapid convergence
pub fn newton_raphson_method(f: &Value, df: &Value, x0: f64, tolerance: f64, max_iterations: usize) -> Result<RootResult, String> {
    let mut x = x0;
    let mut iterations = 0;
    
    loop {
        let fx = evaluate_function(f, x)?;
        let dfx = evaluate_function(df, x)?;
        
        if dfx.abs() < 1e-15 {
            return Err("Derivative too small - potential division by zero".to_string());
        }
        
        let x_new = x - fx / dfx;
        
        if (x_new - x).abs() < tolerance || iterations >= max_iterations {
            let function_value = evaluate_function(f, x_new)?;
            return Ok(RootResult {
                root: x_new,
                iterations: iterations + 1,
                function_value,
                converged: (x_new - x).abs() < tolerance,
                error_estimate: (x_new - x).abs(),
                method: "Newton-Raphson".to_string(),
            });
        }
        
        x = x_new;
        iterations += 1;
    }
}

/// Secant method for root finding
///
/// Uses two initial points and approximates derivative using finite differences
pub fn secant_method(f: &Value, x0: f64, x1: f64, tolerance: f64, max_iterations: usize) -> Result<RootResult, String> {
    let mut x_prev = x0;
    let mut x_curr = x1;
    let mut iterations = 0;
    
    loop {
        let f_prev = evaluate_function(f, x_prev)?;
        let f_curr = evaluate_function(f, x_curr)?;
        
        if (f_curr - f_prev).abs() < 1e-15 {
            return Err("Function values too close - potential division by zero".to_string());
        }
        
        let x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev);
        
        if (x_new - x_curr).abs() < tolerance || iterations >= max_iterations {
            let function_value = evaluate_function(f, x_new)?;
            return Ok(RootResult {
                root: x_new,
                iterations: iterations + 1,
                function_value,
                converged: (x_new - x_curr).abs() < tolerance,
                error_estimate: (x_new - x_curr).abs(),
                method: "Secant".to_string(),
            });
        }
        
        x_prev = x_curr;
        x_curr = x_new;
        iterations += 1;
    }
}

/// Brent's method (hybrid approach)
///
/// Combines bisection, secant, and inverse quadratic interpolation for robust convergence
pub fn brent_method(f: &Value, a: f64, b: f64, tolerance: f64, max_iterations: usize) -> Result<RootResult, String> {
    let mut fa = evaluate_function(f, a)?;
    let mut fb = evaluate_function(f, b)?;
    
    if fa * fb > 0.0 {
        return Err("Function values at endpoints must have opposite signs".to_string());
    }
    
    let mut a_curr = a;
    let mut b_curr = b;
    let mut c = a;
    let mut fc = fa;
    
    let mut mflag = true;
    let mut iterations = 0;
    
    while fb.abs() > tolerance && (b_curr - a_curr).abs() > tolerance && iterations < max_iterations {
        let mut s;
        
        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            s = a_curr * fb * fc / ((fa - fb) * (fa - fc))
              + b_curr * fa * fc / ((fb - fa) * (fb - fc))
              + c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b_curr - fb * (b_curr - a_curr) / (fb - fa);
        }
        
        // Check if we should use bisection instead
        let condition1 = s < (3.0 * a_curr + b_curr) / 4.0 || s > b_curr;
        let condition2 = mflag && (s - b_curr).abs() >= (b_curr - c).abs() / 2.0;
        let condition3 = !mflag && (s - b_curr).abs() >= (c - a_curr).abs() / 2.0;
        let condition4 = mflag && (b_curr - c).abs() < tolerance;
        let condition5 = !mflag && (c - a_curr).abs() < tolerance;
        
        if condition1 || condition2 || condition3 || condition4 || condition5 {
            s = (a_curr + b_curr) / 2.0;
            mflag = true;
        } else {
            mflag = false;
        }
        
        let fs = evaluate_function(f, s)?;
        c = b_curr;
        fc = fb;
        
        if fa * fs < 0.0 {
            b_curr = s;
            fb = fs;
        } else {
            a_curr = s;
            fa = fs;
        }
        
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a_curr, &mut b_curr);
            std::mem::swap(&mut fa, &mut fb);
        }
        
        iterations += 1;
    }
    
    Ok(RootResult {
        root: b_curr,
        iterations,
        function_value: fb,
        converged: fb.abs() <= tolerance,
        error_estimate: (b_curr - a_curr).abs(),
        method: "Brent".to_string(),
    })
}

/// Fixed point iteration
///
/// Finds x such that g(x) = x, equivalent to finding roots of f(x) = g(x) - x
pub fn fixed_point_iteration(g: &Value, x0: f64, tolerance: f64, max_iterations: usize) -> Result<RootResult, String> {
    let mut x = x0;
    let mut iterations = 0;
    
    loop {
        let x_new = evaluate_function(g, x)?;
        
        if (x_new - x).abs() < tolerance || iterations >= max_iterations {
            // For fixed point, f(x) = g(x) - x, so function_value = g(x_new) - x_new
            let function_value = evaluate_function(g, x_new)? - x_new;
            return Ok(RootResult {
                root: x_new,
                iterations: iterations + 1,
                function_value,
                converged: (x_new - x).abs() < tolerance,
                error_estimate: (x_new - x).abs(),
                method: "Fixed Point".to_string(),
            });
        }
        
        x = x_new;
        iterations += 1;
    }
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// Bisection method root finding
/// Syntax: Bisection[f, a, b, tolerance, maxIterations]
pub fn bisection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, a, b, [tolerance], [maxIterations])".to_string(),
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
    
    let max_iterations = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        100
    };
    
    match bisection_method(f, a, b, tolerance, max_iterations) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Bisection failed: {}", e))),
    }
}

/// Newton-Raphson method root finding
/// Syntax: NewtonRaphson[f, df, x0, tolerance, maxIterations]
pub fn newton_raphson(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, derivative, x0, [tolerance], [maxIterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    let df = &args[1];
    
    let x0 = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x0".to_string(),
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
    
    let max_iterations = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        100
    };
    
    match newton_raphson_method(f, df, x0, tolerance, max_iterations) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Newton-Raphson failed: {}", e))),
    }
}

/// Secant method root finding
/// Syntax: Secant[f, x0, x1, tolerance, maxIterations]
pub fn secant(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, x0, x1, [tolerance], [maxIterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let f = &args[0];
    
    let x0 = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x0".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let x1 = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x1".to_string(),
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
    
    let max_iterations = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        100
    };
    
    match secant_method(f, x0, x1, tolerance, max_iterations) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Secant failed: {}", e))),
    }
}

/// Brent's method root finding
/// Syntax: Brent[f, a, b, tolerance, maxIterations]
pub fn brent(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, a, b, [tolerance], [maxIterations])".to_string(),
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
    
    let max_iterations = if args.len() > 4 {
        match &args[4] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[4]),
            }),
        }
    } else {
        100
    };
    
    match brent_method(f, a, b, tolerance, max_iterations) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Brent failed: {}", e))),
    }
}

/// Fixed point iteration
/// Syntax: FixedPoint[g, x0, tolerance, maxIterations]
pub fn fixed_point(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (function, x0, [tolerance], [maxIterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let g = &args[0];
    
    let x0 = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for x0".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let tolerance = if args.len() > 2 {
        match &args[2] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real number for tolerance".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        1e-10
    };
    
    let max_iterations = if args.len() > 3 {
        match &args[3] {
            Value::Integer(i) => *i as usize,
            _ => return Err(VmError::TypeError {
                expected: "Integer for max iterations".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else {
        100
    };
    
    match fixed_point_iteration(g, x0, tolerance, max_iterations) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::Runtime(format!("Fixed point iteration failed: {}", e))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_bisection_method() {
        // Test f(x) = x^2 - 4, which has roots at x = ±2
        let f = Value::List(vec![
            Value::Real(-4.0), // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = bisection_method(&f, 0.0, 3.0, 1e-10, 100).unwrap();
        assert!((result.root - 2.0).abs() < 1e-9);
        assert!(result.converged);
    }
    
    #[test]
    fn test_newton_raphson_method() {
        // Test f(x) = x^2 - 4, df(x) = 2x
        let f = Value::List(vec![
            Value::Real(-4.0), // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let df = Value::List(vec![
            Value::Real(0.0),  // constant term
            Value::Real(2.0),  // x term
        ]);
        
        let result = newton_raphson_method(&f, &df, 1.0, 1e-10, 100).unwrap();
        assert!((result.root - 2.0).abs() < 1e-9);
        assert!(result.converged);
    }
    
    #[test]
    fn test_secant_method() {
        // Test f(x) = x^2 - 4
        let f = Value::List(vec![
            Value::Real(-4.0), // constant term
            Value::Real(0.0),  // x term
            Value::Real(1.0),  // x^2 term
        ]);
        
        let result = secant_method(&f, 1.0, 3.0, 1e-10, 100).unwrap();
        assert!((result.root - 2.0).abs() < 1e-9);
        assert!(result.converged);
    }
    
    #[test]
    fn test_fixed_point_iteration() {
        // Test g(x) = (x + 2/x)/2, which has fixed point at x = sqrt(2)
        // This represents the iterative formula for computing sqrt(2)
        // For simplicity, we'll test with a linear function g(x) = 0.5*x + 1
        // which has fixed point at x = 2
        let g = Value::List(vec![
            Value::Real(1.0),  // constant term
            Value::Real(0.5),  // x term
        ]);
        
        let result = fixed_point_iteration(&g, 0.0, 1e-10, 100).unwrap();
        assert!((result.root - 2.0).abs() < 1e-9);
        assert!(result.converged);
    }
}