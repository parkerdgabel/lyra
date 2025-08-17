//! Math functions for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};

/// Trigonometric sine function
/// Usage: Sin[0] -> 0.0, Sin[Pi/2] -> 1.0
pub fn sin(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => *n as f64,
        Value::Real(r) => *r,
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.sin()))
}

/// Trigonometric cosine function
/// Usage: Cos[0] -> 1.0, Cos[Pi] -> -1.0
pub fn cos(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => *n as f64,
        Value::Real(r) => *r,
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.cos()))
}

/// Trigonometric tangent function
/// Usage: Tan[0] -> 0.0, Tan[Pi/4] -> 1.0
pub fn tan(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => *n as f64,
        Value::Real(r) => *r,
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.tan()))
}

/// Exponential function (e^x)
/// Usage: Exp[0] -> 1.0, Exp[1] -> 2.718...
pub fn exp(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => *n as f64,
        Value::Real(r) => *r,
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.exp()))
}

/// Natural logarithm function
/// Usage: Log[1] -> 0.0, Log[E] -> 1.0
pub fn log(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => {
            if *n <= 0 {
                return Err(VmError::TypeError {
                    expected: "positive number".to_string(),
                    actual: format!("non-positive integer: {}", n),
                });
            }
            *n as f64
        }
        Value::Real(r) => {
            if *r <= 0.0 {
                return Err(VmError::TypeError {
                    expected: "positive number".to_string(),
                    actual: format!("non-positive real: {}", r),
                });
            }
            *r
        }
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.ln()))
}

/// Square root function
/// Usage: Sqrt[4] -> 2.0, Sqrt[2] -> 1.414...
pub fn sqrt(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let x = match &args[0] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative number".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as f64
        }
        Value::Real(r) => {
            if *r < 0.0 {
                return Err(VmError::TypeError {
                    expected: "non-negative number".to_string(),
                    actual: format!("negative real: {}", r),
                });
            }
            *r
        }
        _ => return Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    Ok(Value::Real(x.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;
    
    const EPSILON: f64 = 1e-10;
    
    fn assert_float_eq(actual: Value, expected: f64) {
        match actual {
            Value::Real(r) => assert!((r - expected).abs() < EPSILON, 
                "Expected {}, got {}", expected, r),
            _ => panic!("Expected Real value, got {:?}", actual),
        }
    }
    
    #[test]
    fn test_sin_basic() {
        let result = sin(&[Value::Integer(0)]).unwrap();
        assert_float_eq(result, 0.0);
        
        let result = sin(&[Value::Real(std::f64::consts::PI / 2.0)]).unwrap();
        assert_float_eq(result, 1.0);
        
        let result = sin(&[Value::Real(std::f64::consts::PI)]).unwrap();
        assert_float_eq(result, 0.0);
    }
    
    #[test]
    fn test_sin_wrong_args() {
        assert!(sin(&[]).is_err());
        assert!(sin(&[Value::Integer(1), Value::Integer(2)]).is_err());
    }
    
    #[test]
    fn test_sin_wrong_type() {
        assert!(sin(&[Value::String("not a number".to_string())]).is_err());
    }
    
    #[test]
    fn test_cos_basic() {
        let result = cos(&[Value::Integer(0)]).unwrap();
        assert_float_eq(result, 1.0);
        
        let result = cos(&[Value::Real(std::f64::consts::PI / 2.0)]).unwrap();
        assert_float_eq(result, 0.0);
        
        let result = cos(&[Value::Real(std::f64::consts::PI)]).unwrap();
        assert_float_eq(result, -1.0);
    }
    
    #[test]
    fn test_tan_basic() {
        let result = tan(&[Value::Integer(0)]).unwrap();
        assert_float_eq(result, 0.0);
        
        let result = tan(&[Value::Real(std::f64::consts::PI / 4.0)]).unwrap();
        assert_float_eq(result, 1.0);
    }
    
    #[test]
    fn test_exp_basic() {
        let result = exp(&[Value::Integer(0)]).unwrap();
        assert_float_eq(result, 1.0);
        
        let result = exp(&[Value::Integer(1)]).unwrap();
        assert_float_eq(result, std::f64::consts::E);
        
        let result = exp(&[Value::Real(2.0)]).unwrap();
        assert_float_eq(result, std::f64::consts::E * std::f64::consts::E);
    }
    
    #[test]
    fn test_log_basic() {
        let result = log(&[Value::Integer(1)]).unwrap();
        assert_float_eq(result, 0.0);
        
        let result = log(&[Value::Real(std::f64::consts::E)]).unwrap();
        assert_float_eq(result, 1.0);
        
        let result = log(&[Value::Real(std::f64::consts::E * std::f64::consts::E)]).unwrap();
        assert_float_eq(result, 2.0);
    }
    
    #[test]
    fn test_log_invalid_input() {
        assert!(log(&[Value::Integer(0)]).is_err());
        assert!(log(&[Value::Integer(-1)]).is_err());
        assert!(log(&[Value::Real(0.0)]).is_err());
        assert!(log(&[Value::Real(-1.0)]).is_err());
    }
    
    #[test]
    fn test_sqrt_basic() {
        let result = sqrt(&[Value::Integer(0)]).unwrap();
        assert_float_eq(result, 0.0);
        
        let result = sqrt(&[Value::Integer(1)]).unwrap();
        assert_float_eq(result, 1.0);
        
        let result = sqrt(&[Value::Integer(4)]).unwrap();
        assert_float_eq(result, 2.0);
        
        let result = sqrt(&[Value::Real(2.0)]).unwrap();
        assert_float_eq(result, std::f64::consts::SQRT_2);
    }
    
    #[test]
    fn test_sqrt_invalid_input() {
        assert!(sqrt(&[Value::Integer(-1)]).is_err());
        assert!(sqrt(&[Value::Real(-1.0)]).is_err());
    }
    
    #[test]
    fn test_math_functions_wrong_args() {
        let funcs = [cos, tan, exp, log, sqrt];
        for func in &funcs {
            assert!(func(&[]).is_err());
            assert!(func(&[Value::Integer(1), Value::Integer(2)]).is_err());
        }
    }
    
    #[test]
    fn test_math_functions_wrong_type() {
        let funcs = [sin, cos, tan, exp, log, sqrt];
        for func in &funcs {
            assert!(func(&[Value::String("not a number".to_string())]).is_err());
            assert!(func(&[Value::List(vec![])]).is_err());
        }
    }
}