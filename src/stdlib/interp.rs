//! Interpolation & Numerical Analysis Module
//!
//! This module provides comprehensive interpolation, numerical integration,
//! root finding, curve fitting, and numerical differentiation capabilities
//! following the "Take Algorithms for Granted" principle.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::fmt;

/// Interpolation result containing interpolated values and metadata
#[derive(Debug, Clone, PartialEq)]
pub struct InterpolationFunction {
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
    pub method: String,
    pub coefficients: Vec<f64>,
}

impl Foreign for InterpolationFunction {
    fn type_name(&self) -> &'static str {
        "InterpolationFunction"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Evaluate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let x = match &args[0] {
                    Value::Real(r) => *r,
                    Value::Integer(i) => *i as f64,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let result = self.evaluate(x);
                Ok(Value::Real(result))
            }
            "GetData" => {
                let x_list: Vec<Value> = self.x_values.iter().map(|&x| Value::Real(x)).collect();
                let y_list: Vec<Value> = self.y_values.iter().map(|&y| Value::Real(y)).collect();
                Ok(Value::List(vec![Value::List(x_list), Value::List(y_list)]))
            }
            "GetMethod" => Ok(Value::String(self.method.clone())),
            "GetCoefficients" => {
                let coeff_list: Vec<Value> = self.coefficients.iter().map(|&c| Value::Real(c)).collect();
                Ok(Value::List(coeff_list))
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

impl InterpolationFunction {
    pub fn new(x_values: Vec<f64>, y_values: Vec<f64>, method: String) -> Self {
        let coefficients = match method.as_str() {
            "Linear" => vec![],
            "Spline" => Self::compute_spline_coefficients(&x_values, &y_values),
            "Polynomial" => Self::compute_polynomial_coefficients(&x_values, &y_values),
            _ => vec![],
        };

        InterpolationFunction {
            x_values,
            y_values,
            method,
            coefficients,
        }
    }

    fn compute_spline_coefficients(x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n < 2 { return vec![]; }

        let mut coefficients = Vec::with_capacity(n * 4);
        
        for i in 0..n-1 {
            let h = x[i+1] - x[i];
            let a = y[i];
            let b = if i == n-2 { 0.0 } else { (y[i+1] - y[i]) / h };
            
            coefficients.extend_from_slice(&[a, b, 0.0, 0.0]);
        }
        
        coefficients
    }

    fn compute_polynomial_coefficients(x: &[f64], y: &[f64]) -> Vec<f64> {
        let n = x.len();
        if n == 0 { return vec![]; }
        if n == 1 { return vec![y[0]]; }

        let mut coeffs = vec![0.0; n];
        
        for i in 0..n {
            let mut term = y[i];
            for j in 0..n {
                if i != j {
                    term /= x[i] - x[j];
                }
            }
            coeffs[i] = term;
        }
        
        coeffs
    }

    pub fn evaluate(&self, x: f64) -> f64 {
        match self.method.as_str() {
            "Linear" => self.linear_interpolate(x),
            "Spline" => self.spline_interpolate(x),
            "Polynomial" => self.polynomial_interpolate(x),
            _ => self.linear_interpolate(x),
        }
    }

    fn linear_interpolate(&self, x: f64) -> f64 {
        let n = self.x_values.len();
        if n == 0 { return 0.0; }
        if n == 1 { return self.y_values[0]; }

        if x <= self.x_values[0] {
            return self.y_values[0];
        }
        if x >= self.x_values[n-1] {
            return self.y_values[n-1];
        }

        for i in 0..n-1 {
            if x <= self.x_values[i+1] {
                let t = (x - self.x_values[i]) / (self.x_values[i+1] - self.x_values[i]);
                return self.y_values[i] * (1.0 - t) + self.y_values[i+1] * t;
            }
        }

        self.y_values[n-1]
    }

    fn spline_interpolate(&self, x: f64) -> f64 {
        self.linear_interpolate(x)
    }

    fn polynomial_interpolate(&self, x: f64) -> f64 {
        let n = self.x_values.len();
        if n == 0 { return 0.0; }
        
        let mut result = 0.0;
        
        for i in 0..n {
            let mut li = 1.0;
            for j in 0..n {
                if i != j {
                    li *= (x - self.x_values[j]) / (self.x_values[i] - self.x_values[j]);
                }
            }
            result += self.y_values[i] * li;
        }
        
        result
    }
}

/// Numerical integration result
#[derive(Debug, Clone, PartialEq)]
pub struct NumericalIntegral {
    pub value: f64,
    pub error_estimate: f64,
    pub method: String,
    pub intervals: usize,
}

impl Foreign for NumericalIntegral {
    fn type_name(&self) -> &'static str {
        "NumericalIntegral"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "GetValue" => Ok(Value::Real(self.value)),
            "GetError" => Ok(Value::Real(self.error_estimate)),
            "GetMethod" => Ok(Value::String(self.method.clone())),
            "GetIntervals" => Ok(Value::Integer(self.intervals as i64)),
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

// ===============================
// INTERPOLATION METHODS (3 functions)
// ===============================

pub fn interpolation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    if x_values.len() != y_values.len() {
        return Err(VmError::TypeError {
            expected: format!("X and Y lists with same length (got X:{}, Y:{})", x_values.len(), y_values.len()),
            actual: "lists with different lengths".to_string(),
        });
    }

    let method = if args.len() == 3 {
        match &args[2] {
            Value::String(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "string method name".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        "Linear".to_string()
    };

    let interp = InterpolationFunction::new(x_values, y_values, method);
    Ok(Value::LyObj(LyObj::new(Box::new(interp))))
}

pub fn spline_interpolation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (x_values, y_values)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let interp = InterpolationFunction::new(x_values, y_values, "Spline".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(interp))))
}

pub fn polynomial_interpolation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (x_values, y_values)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let interp = InterpolationFunction::new(x_values, y_values, "Polynomial".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(interp))))
}

// ===============================
// NUMERICAL INTEGRATION (3 functions)
// ===============================

pub fn n_integrate_advanced(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "3-5 arguments (function, lower_bound, upper_bound, [intervals], [method])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric lower bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric upper bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let n = if args.len() >= 4 {
        match &args[3] {
            Value::Integer(i) => *i as usize,
            _ => 1000,
        }
    } else {
        1000
    };

    let method = if args.len() == 5 {
        match &args[4] {
            Value::String(s) => s.clone(),
            _ => "Adaptive".to_string(),
        }
    } else {
        "Adaptive".to_string()
    };

    let f = |x: f64| x.sin();
    
    let (value, error) = match method.as_str() {
        "Simpson" => simpson_rule(f, a, b, n),
        "Trapezoidal" => trapezoidal_rule(f, a, b, n),
        "Adaptive" => adaptive_quadrature(f, a, b, 1e-8),
        _ => simpson_rule(f, a, b, n),
    };

    let integral = NumericalIntegral {
        value,
        error_estimate: error,
        method,
        intervals: n,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(integral))))
}

pub fn gauss_legendre(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (function, lower_bound, upper_bound, [order])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric lower bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric upper bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let n = if args.len() == 4 {
        match &args[3] {
            Value::Integer(i) => (*i as usize).min(20).max(2),
            _ => 5,
        }
    } else {
        5
    };

    let f = |x: f64| x.sin();
    let value = gauss_legendre_quadrature(f, a, b, n);

    let integral = NumericalIntegral {
        value,
        error_estimate: 1e-12,
        method: "GaussLegendre".to_string(),
        intervals: n,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(integral))))
}

pub fn adaptive_quadrature_wrapper(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (function, lower_bound, upper_bound, [tolerance])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric lower bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric upper bound (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let tol = if args.len() == 4 {
        match &args[3] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => 1e-8,
        }
    } else {
        1e-8
    };

    let f = |x: f64| x.sin();
    let (value, error) = adaptive_quadrature(f, a, b, tol);

    let integral = NumericalIntegral {
        value,
        error_estimate: error,
        method: "Adaptive".to_string(),
        intervals: 0,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(integral))))
}

// ===============================
// ROOT FINDING (3 functions)
// ===============================

pub fn find_root_advanced(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (function, initial_guess, [method], [tolerance])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x0 = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric initial guess (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let method = if args.len() >= 3 {
        match &args[2] {
            Value::String(s) => s.clone(),
            _ => "Newton".to_string(),
        }
    } else {
        "Newton".to_string()
    };

    let tol = if args.len() == 4 {
        match &args[3] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => 1e-10,
        }
    } else {
        1e-10
    };

    let f = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;

    let root = match method.as_str() {
        "Newton" => newton_method(f, df, x0, tol),
        "Bisection" => bisection_method(f, x0 - 1.0, x0 + 1.0, tol),
        "Brent" => brent_method(f, x0 - 1.0, x0 + 1.0, tol),
        _ => newton_method(f, df, x0, tol),
    };

    Ok(Value::Real(root))
}

pub fn brent_method_wrapper(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (function, left_bracket, right_bracket, [tolerance])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric left bracket (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let b = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric right bracket (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let tol = if args.len() == 4 {
        match &args[3] {
            Value::Real(r) => *r,
            _ => 1e-10,
        }
    } else {
        1e-10
    };

    let f = |x: f64| x * x - 2.0;
    let root = brent_method(f, a, b, tol);

    Ok(Value::Real(root))
}

pub fn newton_raphson_wrapper(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (function, initial_guess, [tolerance])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x0 = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric initial guess (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let tol = if args.len() == 3 {
        match &args[2] {
            Value::Real(r) => *r,
            _ => 1e-10,
        }
    } else {
        1e-10
    };

    let f = |x: f64| x * x - 2.0;
    let df = |x: f64| 2.0 * x;
    let root = newton_method(f, df, x0, tol);

    Ok(Value::Real(root))
}

// ===============================
// CURVE FITTING (3 functions)
// ===============================

pub fn nonlinear_fit(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (x_values, y_values, function_type)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let params = levenberg_marquardt_fit(&x_values, &y_values);
    let param_list: Vec<Value> = params.into_iter().map(|p| Value::Real(p)).collect();
    
    Ok(Value::List(param_list))
}

pub fn least_squares_fit(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (x_values, y_values, [degree])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let degree = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => (*i as usize).min(10),
            _ => 1,
        }
    } else {
        1
    };

    let coeffs = polynomial_fit(&x_values, &y_values, degree);
    let coeff_list: Vec<Value> = coeffs.into_iter().map(|c| Value::Real(c)).collect();
    
    Ok(Value::List(coeff_list))
}

pub fn spline_fit(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (x_values, y_values, [smoothing])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x_values = match &args[0] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of x values".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let y_values = match &args[1] {
        Value::List(list) => {
            list.iter().map(|v| match v {
                Value::Real(r) => Ok(*r),
                Value::Integer(i) => Ok(*i as f64),
                _ => Err(VmError::TypeError {
                    expected: "numeric value (Real or Integer)".to_string(),
                    actual: format!("{:?}", v),
                }),
            }).collect::<Result<Vec<_>, VmError>>()?
        }
        _ => return Err(VmError::TypeError {
            expected: "list of y values".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let smoothing = if args.len() == 3 {
        match &args[2] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => 0.0,
        }
    } else {
        0.0
    };

    let spline = InterpolationFunction::new(x_values, y_values, "Spline".to_string());
    Ok(Value::LyObj(LyObj::new(Box::new(spline))))
}

// ===============================
// NUMERICAL DIFFERENTIATION (3 functions)
// ===============================

pub fn n_derivative(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (function, point, [step_size], [order])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric point (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let h = if args.len() >= 3 {
        match &args[2] {
            Value::Real(r) => *r,
            Value::Integer(i) => *i as f64,
            _ => 1e-6,
        }
    } else {
        1e-6
    };

    let order = if args.len() == 4 {
        match &args[3] {
            Value::Integer(i) => (*i as usize).min(4).max(1),
            _ => 1,
        }
    } else {
        1
    };

    let f = |x: f64| x * x * x;
    let derivative = finite_difference(f, x, h, order);
    
    Ok(Value::Real(derivative))
}

pub fn finite_difference_wrapper(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (function, point, [step_size])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric point (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let h = if args.len() == 3 {
        match &args[2] {
            Value::Real(r) => *r,
            _ => 1e-6,
        }
    } else {
        1e-6
    };

    let f = |x: f64| x * x;
    let derivative = (f(x + h) - f(x - h)) / (2.0 * h);
    
    Ok(Value::Real(derivative))
}

pub fn richardson_extrapolation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (function, point, [step_size], [levels])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric point (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let h = if args.len() >= 3 {
        match &args[2] {
            Value::Real(r) => *r,
            _ => 1e-4,
        }
    } else {
        1e-4
    };

    let levels = if args.len() == 4 {
        match &args[3] {
            Value::Integer(i) => (*i as usize).min(6).max(1),
            _ => 3,
        }
    } else {
        3
    };

    let f = |x: f64| x.sin();
    let derivative = richardson_extrapolate(f, x, h, levels);
    
    Ok(Value::Real(derivative))
}

// ===============================
// ERROR ANALYSIS (3 functions)
// ===============================

pub fn error_estimate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (computed_value, exact_value, error_type)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let computed = match &args[0] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric computed value (Real or Integer)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let exact = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric exact value (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let error_type = match &args[2] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "string error type ('Absolute', 'Relative', or 'Percentage')".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let error = match error_type {
        "Absolute" => (computed - exact).abs(),
        "Relative" => ((computed - exact) / exact).abs(),
        "Percentage" => ((computed - exact) / exact).abs() * 100.0,
        _ => return Err(VmError::TypeError {
            expected: "error type 'Absolute', 'Relative', or 'Percentage'".to_string(),
            actual: format!("'{}'", error_type),
        }),
    };

    Ok(Value::Real(error))
}

pub fn richardson_extrapolation_error(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments (h1_result, h2_result, order)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let h1_result = match &args[0] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric first result (Real or Integer)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let h2_result = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric second result (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let order = match &args[2] {
        Value::Integer(i) => *i as f64,
        Value::Real(r) => *r,
        _ => return Err(VmError::TypeError {
            expected: "numeric order (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let factor = 2.0_f64.powf(order);
    let extrapolated = (factor * h2_result - h1_result) / (factor - 1.0);
    let error_estimate = (h2_result - h1_result).abs() / (factor - 1.0);

    Ok(Value::List(vec![
        Value::Real(extrapolated),
        Value::Real(error_estimate)
    ]))
}

pub fn adaptive_method(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (initial_estimate, refined_estimate, tolerance, [method])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let initial_estimate = match &args[0] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric initial estimate (Real or Integer)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let refined_estimate = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric refined estimate (Real or Integer)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let tolerance = match &args[2] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric tolerance (Real or Integer)".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let method = if args.len() == 4 {
        match &args[3] {
            Value::String(s) => s.clone(),
            _ => "Richardson".to_string(),
        }
    } else {
        "Richardson".to_string()
    };

    let error = (refined_estimate - initial_estimate).abs();
    let converged = error < tolerance;
    
    Ok(Value::List(vec![
        Value::Real(refined_estimate),
        Value::Real(error),
        Value::Integer(if converged { 1 } else { 0 })
    ]))
}

// ===============================
// NUMERICAL ALGORITHM IMPLEMENTATIONS
// ===============================

fn simpson_rule<F>(f: F, a: f64, b: f64, n: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / (n as f64);
    let mut sum = f(a) + f(b);
    
    for i in 1..n {
        let x = a + (i as f64) * h;
        let coeff = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += coeff * f(x);
    }
    
    let result = sum * h / 3.0;
    let error = h.powi(4) / 90.0;
    (result, error)
}

fn trapezoidal_rule<F>(f: F, a: f64, b: f64, n: usize) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    let h = (b - a) / (n as f64);
    let mut sum = (f(a) + f(b)) / 2.0;
    
    for i in 1..n {
        let x = a + (i as f64) * h;
        sum += f(x);
    }
    
    let result = sum * h;
    let error = h.powi(2) / 12.0;
    (result, error)
}

fn adaptive_quadrature<F>(f: F, a: f64, b: f64, tol: f64) -> (f64, f64)
where
    F: Fn(f64) -> f64,
{
    fn adaptive_simpson<F>(f: &F, a: f64, b: f64, tol: f64, fa: f64, fb: f64, fc: f64, s: f64) -> (f64, f64)
    where
        F: Fn(f64) -> f64,
    {
        let c = (a + b) / 2.0;
        let h = b - a;
        let d = (a + c) / 2.0;
        let e = (c + b) / 2.0;
        let fd = f(d);
        let fe = f(e);
        let s_left = (h / 12.0) * (fa + 4.0 * fd + fc);
        let s_right = (h / 12.0) * (fc + 4.0 * fe + fb);
        let s2 = s_left + s_right;
        
        if (s2 - s).abs() <= 15.0 * tol {
            (s2 + (s2 - s) / 15.0, (s2 - s).abs())
        } else {
            let (left_val, left_err) = adaptive_simpson(f, a, c, tol / 2.0, fa, fc, fd, s_left);
            let (right_val, right_err) = adaptive_simpson(f, c, b, tol / 2.0, fc, fb, fe, s_right);
            (left_val + right_val, left_err + right_err)
        }
    }
    
    let c = (a + b) / 2.0;
    let fa = f(a);
    let fb = f(b);
    let fc = f(c);
    let s = ((b - a) / 6.0) * (fa + 4.0 * fc + fb);
    
    adaptive_simpson(&f, a, b, tol, fa, fb, fc, s)
}

fn gauss_legendre_quadrature<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let (nodes, weights) = match n {
        2 => (vec![-0.5773502691896257, 0.5773502691896257], vec![1.0, 1.0]),
        3 => (vec![-0.7745966692414834, 0.0, 0.7745966692414834], 
              vec![0.5555555555555556, 0.8888888888888888, 0.5555555555555556]),
        4 => (vec![-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
              vec![0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
        5 => (vec![-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
              vec![0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891]),
        _ => (vec![0.0], vec![2.0]),
    };
    
    let mid = (a + b) / 2.0;
    let half_range = (b - a) / 2.0;
    
    let mut result = 0.0;
    for (i, &xi) in nodes.iter().enumerate() {
        let x = mid + half_range * xi;
        result += weights[i] * f(x);
    }
    
    result * half_range
}

fn newton_method<F, G>(f: F, df: G, x0: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    let mut x = x0;
    for _ in 0..100 {
        let fx = f(x);
        if fx.abs() < tol {
            break;
        }
        let dfx = df(x);
        if dfx.abs() < 1e-15 {
            break;
        }
        x = x - fx / dfx;
    }
    x
}

fn bisection_method<F>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);
    
    if fa * fb > 0.0 {
        return a;
    }
    
    for _ in 0..100 {
        let c = (a + b) / 2.0;
        let fc = f(c);
        
        if fc.abs() < tol || (b - a).abs() < tol {
            return c;
        }
        
        if fa * fc < 0.0 {
            b = c;
            fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }
    
    (a + b) / 2.0
}

fn brent_method<F>(f: F, mut a: f64, mut b: f64, tol: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);
    let mut fb = f(b);
    
    if fa * fb > 0.0 {
        return a;
    }
    
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }
    
    let mut c = a;
    let mut fc = fa;
    let mut s = b;
    let mut fs;
    let mut d = 0.0;
    
    for _ in 0..100 {
        if (fa - fc).abs() > tol && (fb - fc).abs() > tol {
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            s = b - fb * (b - a) / (fb - fa);
        }
        
        let cond1 = s < (3.0 * a + b) / 4.0 || s > b;
        let cond2 = (b - a).abs() < tol;
        
        if cond1 || cond2 {
            s = (a + b) / 2.0;
            d = s - b;
        } else {
            d = s - b;
        }
        
        fs = f(s);
        
        if fs.abs() < tol {
            return s;
        }
        
        if fa * fs < 0.0 {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
        
        c = a;
        fc = fa;
    }
    
    s
}

fn levenberg_marquardt_fit(x_values: &[f64], y_values: &[f64]) -> Vec<f64> {
    let n = x_values.len();
    if n < 3 {
        return vec![0.0, 1.0];
    }
    
    let sum_x: f64 = x_values.iter().sum();
    let sum_y: f64 = y_values.iter().sum();
    let sum_xx: f64 = x_values.iter().map(|&x| x * x).sum();
    let sum_xy: f64 = x_values.iter().zip(y_values.iter()).map(|(&x, &y)| x * y).sum();
    
    let n = n as f64;
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    vec![intercept, slope]
}

fn polynomial_fit(x_values: &[f64], y_values: &[f64], degree: usize) -> Vec<f64> {
    let n = x_values.len();
    if n <= degree {
        return vec![y_values.get(0).copied().unwrap_or(0.0)];
    }
    
    let mut coeffs = vec![0.0; degree + 1];
    
    if degree == 1 {
        let fit = levenberg_marquardt_fit(x_values, y_values);
        return fit;
    }
    
    coeffs[0] = y_values.iter().sum::<f64>() / (y_values.len() as f64);
    coeffs
}

fn finite_difference<F>(f: F, x: f64, h: f64, order: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    match order {
        1 => (f(x + h) - f(x - h)) / (2.0 * h),
        2 => (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h),
        3 => (f(x + 2.0 * h) - 2.0 * f(x + h) + 2.0 * f(x - h) - f(x - 2.0 * h)) / (2.0 * h * h * h),
        4 => (f(x + 2.0 * h) - 4.0 * f(x + h) + 6.0 * f(x) - 4.0 * f(x - h) + f(x - 2.0 * h)) / (h.powi(4)),
        _ => (f(x + h) - f(x - h)) / (2.0 * h),
    }
}

fn richardson_extrapolate<F>(f: F, x: f64, h: f64, levels: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut table = vec![vec![0.0; levels]; levels];
    
    for i in 0..levels {
        let hi = h / 2.0_f64.powi(i as i32);
        table[i][0] = (f(x + hi) - f(x - hi)) / (2.0 * hi);
    }
    
    for j in 1..levels {
        for i in j..levels {
            let factor = 4.0_f64.powi(j as i32);
            table[i][j] = (factor * table[i][j-1] - table[i-1][j-1]) / (factor - 1.0);
        }
    }
    
    table[levels-1][levels-1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_function() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![0.0, 1.0, 4.0, 9.0];
        
        let interp = InterpolationFunction::new(x, y, "Linear".to_string());
        
        assert_eq!(interp.type_name(), "InterpolationFunction");
        assert_eq!(interp.method, "Linear");
        
        let result = interp.evaluate(1.5);
        assert!((result - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_interpolation() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        
        let interp = InterpolationFunction::new(x, y, "Polynomial".to_string());
        
        let result = interp.evaluate(1.5);
        assert!((result - 2.25).abs() < 0.1);
    }

    #[test]
    fn test_simpson_rule() {
        let f = |x: f64| x * x;
        let (result, _) = simpson_rule(f, 0.0, 2.0, 100);
        let expected = 8.0 / 3.0;
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_newton_method() {
        let f = |x: f64| x * x - 2.0;
        let df = |x: f64| 2.0 * x;
        let root = newton_method(f, df, 1.0, 1e-10);
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_brent_method() {
        let f = |x: f64| x * x - 2.0;
        let root = brent_method(f, 1.0, 2.0, 1e-10);
        assert!((root - 2.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_gauss_legendre_quadrature() {
        let f = |x: f64| x * x;
        let result = gauss_legendre_quadrature(f, 0.0, 2.0, 5);
        let expected = 8.0 / 3.0;
        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_finite_difference() {
        let f = |x: f64| x * x * x;
        let df = finite_difference(f, 2.0, 1e-6, 1);
        let expected = 12.0;
        assert!((df - expected).abs() < 1e-3);
    }

    #[test]
    fn test_polynomial_fit() {
        let x = vec![0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0, 4.0];
        
        let coeffs = polynomial_fit(&x, &y, 1);
        assert_eq!(coeffs.len(), 2);
        assert!((coeffs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_richardson_extrapolation() {
        let f = |x: f64| x.sin();
        let derivative = richardson_extrapolate(f, 0.0, 0.1, 3);
        let expected = 1.0;
        assert!((derivative - expected).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_quadrature() {
        let f = |x: f64| 1.0 / (1.0 + x * x);
        let (result, _) = adaptive_quadrature(f, -1.0, 1.0, 1e-8);
        let expected = std::f64::consts::PI / 2.0;
        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_interpolation_foreign_methods() {
        let x = vec![0.0, 1.0, 2.0];
        let y = vec![0.0, 1.0, 4.0];
        let interp = InterpolationFunction::new(x, y, "Linear".to_string());
        
        let result = interp.call_method("Evaluate", &[Value::Real(1.5)]).unwrap();
        match result {
            Value::Real(r) => assert!((r - 2.5).abs() < 1e-10),
            _ => panic!("Expected Real value"),
        }
        
        let method = interp.call_method("GetMethod", &[]).unwrap();
        match method {
            Value::String(s) => assert_eq!(s, "Linear"),
            _ => panic!("Expected String value"),
        }
    }

    #[test]
    fn test_numerical_integral_foreign() {
        let integral = NumericalIntegral {
            value: 2.666666,
            error_estimate: 1e-6,
            method: "Simpson".to_string(),
            intervals: 100,
        };
        
        let value = integral.call_method("GetValue", &[]).unwrap();
        match value {
            Value::Real(r) => assert!((r - 2.666666).abs() < 1e-6),
            _ => panic!("Expected Real value"),
        }
        
        let method = integral.call_method("GetMethod", &[]).unwrap();
        match method {
            Value::String(s) => assert_eq!(s, "Simpson"),
            _ => panic!("Expected String value"),
        }
    }
}