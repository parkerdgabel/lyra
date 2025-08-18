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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
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
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", args[0]),
            })
        }
    };

    Ok(Value::Real(x.sqrt()))
}

/// Addition function for Listable attribute support
/// Usage: Plus[2, 3] -> 5, Plus[{1,2}, {3,4}] -> {4,6} (when Listable)
pub fn plus(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x + y)),
        (Value::Integer(x), Value::Real(y)) => Ok(Value::Real(*x as f64 + y)),
        (Value::Real(x), Value::Integer(y)) => Ok(Value::Real(x + *y as f64)),
        (Value::Real(x), Value::Real(y)) => Ok(Value::Real(x + y)),
        _ => Err(VmError::TypeError {
            expected: "Numbers".to_string(),
            actual: format!("{:?}, {:?}", a, b),
        }),
    }
}

/// Multiplication function for Listable attribute support
/// Usage: Times[2, 3] -> 6, Times[{1,2}, {3,4}] -> {3,8} (when Listable)
pub fn times(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x * y)),
        (Value::Integer(x), Value::Real(y)) => Ok(Value::Real(*x as f64 * y)),
        (Value::Real(x), Value::Integer(y)) => Ok(Value::Real(x * *y as f64)),
        (Value::Real(x), Value::Real(y)) => Ok(Value::Real(x * y)),
        _ => Err(VmError::TypeError {
            expected: "Numbers".to_string(),
            actual: format!("{:?}, {:?}", a, b),
        }),
    }
}

/// Division function for Listable attribute support
/// Usage: Divide[6, 2] -> 3.0, Divide[{6,8}, {2,4}] -> {3.0,2.0} (when Listable)
pub fn divide(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let divisor = match b {
        Value::Integer(y) => {
            if *y == 0 {
                return Err(VmError::TypeError {
                    expected: "non-zero divisor".to_string(),
                    actual: "zero".to_string(),
                });
            }
            *y as f64
        }
        Value::Real(y) => {
            if *y == 0.0 {
                return Err(VmError::TypeError {
                    expected: "non-zero divisor".to_string(),
                    actual: "zero".to_string(),
                });
            }
            *y
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", b),
            });
        }
    };
    
    let dividend = match a {
        Value::Integer(x) => *x as f64,
        Value::Real(x) => *x,
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", a),
            });
        }
    };
    
    Ok(Value::Real(dividend / divisor))
}

/// Power function for Listable attribute support
/// Usage: Power[2, 3] -> 8, Power[{2,3}, {2,3}] -> {4,27} (when Listable)
pub fn power(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let base = match a {
        Value::Integer(x) => *x as f64,
        Value::Real(x) => *x,
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", a),
            });
        }
    };
    
    let exponent = match b {
        Value::Integer(y) => *y as f64,
        Value::Real(y) => *y,
        _ => {
            return Err(VmError::TypeError {
                expected: "Number".to_string(),
                actual: format!("{:?}", b),
            });
        }
    };
    
    let result = base.powf(exponent);
    
    // Return integer if result is a whole number and both inputs were integers
    if matches!((a, b), (Value::Integer(_), Value::Integer(_))) && result.fract() == 0.0 {
        Ok(Value::Integer(result as i64))
    } else {
        Ok(Value::Real(result))
    }
}

/// Minus (unary negation) function for Listable attribute support
/// Usage: Minus[5] -> -5, Minus[{1,2,3}] -> {-1,-2,-3} (when Listable)
pub fn minus(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Integer(x) => Ok(Value::Integer(-x)),
        Value::Real(x) => Ok(Value::Real(-x)),
        _ => Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Test function for Hold[1] attribute - holds first argument
/// Usage: TestHold[1+1, 2+2] -> evaluates second arg but holds first
pub fn test_hold(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // For testing purposes, just return a string indicating what we received
    // First arg should be held (unevaluated), second arg should be evaluated
    Ok(Value::String(format!("TestHold[{:?}, {:?}]", args[0], args[1])))
}

/// Test function for Hold[2,3] attribute - holds second and third arguments
/// Usage: TestHoldMultiple[1+1, 2+2, 3+3, 4+4] -> evaluates args 1,4 but holds args 2,3
pub fn test_hold_multiple(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::TypeError {
            expected: "exactly 4 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // For testing purposes, return a string showing what we received
    // Args 1,4 should be evaluated, args 2,3 should be held (unevaluated)
    Ok(Value::String(format!("TestHoldMultiple[{:?}, {:?}, {:?}, {:?}]", 
        args[0], args[1], args[2], args[3])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    const EPSILON: f64 = 1e-10;

    fn assert_float_eq(actual: Value, expected: f64) {
        match actual {
            Value::Real(r) => assert!(
                (r - expected).abs() < EPSILON,
                "Expected {}, got {}",
                expected,
                r
            ),
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
