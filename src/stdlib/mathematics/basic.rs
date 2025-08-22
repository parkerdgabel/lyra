//! Math functions for the Lyra standard library

use crate::vm::{Value, VmError, VmResult};

/// Fast numeric value extraction for math operations
#[inline(always)]
fn extract_number(value: &Value) -> Result<f64, VmError> {
    match value {
        Value::Integer(n) => Ok(*n as f64),
        Value::Real(r) => Ok(*r),
        _ => Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Extract number with arity check in one operation
#[inline(always)]
fn extract_single_number(args: &[Value]) -> Result<f64, VmError> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    extract_number(&args[0])
}

/// Trigonometric sine function
/// Usage: Sin[0] -> 0.0, Sin[Pi/2] -> 1.0
pub fn sin(args: &[Value]) -> VmResult<Value> {
    let x = extract_single_number(args)?;
    Ok(Value::Real(x.sin()))
}

/// Trigonometric cosine function
/// Usage: Cos[0] -> 1.0, Cos[Pi] -> -1.0
pub fn cos(args: &[Value]) -> VmResult<Value> {
    let x = extract_single_number(args)?;
    Ok(Value::Real(x.cos()))
}

/// Trigonometric tangent function
/// Usage: Tan[0] -> 0.0, Tan[Pi/4] -> 1.0
pub fn tan(args: &[Value]) -> VmResult<Value> {
    let x = extract_single_number(args)?;
    Ok(Value::Real(x.tan()))
}

/// Exponential function (e^x)
/// Usage: Exp[0] -> 1.0, Exp[1] -> 2.718...
pub fn exp(args: &[Value]) -> VmResult<Value> {
    let x = extract_single_number(args)?;
    Ok(Value::Real(x.exp()))
}

/// Natural logarithm function
/// Usage: Log[1] -> 0.0, Log[E] -> 1.0
pub fn log(args: &[Value]) -> VmResult<Value> {
    let x = extract_single_number(args)?;
    
    // Fast validation for positive numbers
    if x <= 0.0 {
        return Err(VmError::TypeError {
            expected: "positive number".to_string(),
            actual: format!("non-positive number: {}", x),
        });
    }
    
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

/// Modulo function
/// Usage: Modulo[17, 5] -> 2, Modulo[10, 3] -> 1
pub fn modulo(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    let (a, b) = (&args[0], &args[1]);
    
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => {
            if *y == 0 {
                return Err(VmError::DivisionByZero);
            }
            Ok(Value::Integer(x % y))
        }
        (Value::Integer(x), Value::Real(y)) => {
            if *y == 0.0 {
                return Err(VmError::DivisionByZero);
            }
            Ok(Value::Real((*x as f64) % y))
        }
        (Value::Real(x), Value::Integer(y)) => {
            if *y == 0 {
                return Err(VmError::DivisionByZero);
            }
            Ok(Value::Real(x % (*y as f64)))
        }
        (Value::Real(x), Value::Real(y)) => {
            if *y == 0.0 {
                return Err(VmError::DivisionByZero);
            }
            Ok(Value::Real(x % y))
        }
        _ => Err(VmError::TypeError {
            expected: "Numbers".to_string(),
            actual: format!("{:?}, {:?}", a, b),
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

/// Absolute value function
/// Usage: Abs[-42] -> 42, Abs[3.14] -> 3.14
pub fn abs(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Integer(n) => Ok(Value::Integer(n.abs())),
        Value::Real(r) => Ok(Value::Real(r.abs())),
        _ => Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Sign function
/// Usage: Sign[-5] -> -1, Sign[0] -> 0, Sign[5] -> 1
pub fn sign(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Integer(n) => Ok(Value::Integer(n.signum())),
        Value::Real(r) => {
            if *r > 0.0 {
                Ok(Value::Integer(1))
            } else if *r < 0.0 {
                Ok(Value::Integer(-1))
            } else {
                Ok(Value::Integer(0))
            }
        }
        _ => Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Mathematical constants
/// Usage: Pi -> 3.14159..., E -> 2.71828...
pub fn pi(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(std::f64::consts::PI))
}

pub fn e(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(std::f64::consts::E))
}

/// Boolean constants
/// Usage: True -> True, False -> False
pub fn true_constant(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(true))
}

pub fn false_constant(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Boolean(false))
}

/// Special constants
pub fn infinity(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(f64::INFINITY))
}

pub fn undefined(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Symbol("Undefined".to_string()))
}

pub fn missing(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Missing)
}

/// Euler-Mascheroni constant (γ ≈ 0.5772156649)
pub fn euler_gamma(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(0.5772156649015329))
}

/// Golden ratio (φ ≈ 1.618033988749)
pub fn golden_ratio(_args: &[Value]) -> VmResult<Value> {
    Ok(Value::Real(1.618033988749894))
}

/// Convert any value to string representation
/// Usage: ToString[42] -> "42", ToString[{1,2,3}] -> "{1, 2, 3}"
pub fn to_string(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let string_repr = match &args[0] {
        Value::Integer(n) => n.to_string(),
        Value::Real(f) => {
            if f.fract() == 0.0 {
                format!("{:.1}", f)
            } else {
                f.to_string()
            }
        }
        Value::String(s) => s.clone(),
        Value::Symbol(s) => s.clone(),
        Value::Boolean(b) => if *b { "True".to_string() } else { "False".to_string() },
        Value::Missing => "Missing".to_string(),
        Value::List(items) => {
            let formatted_items: Vec<String> = items.iter().map(|v| match v {
                Value::Integer(n) => n.to_string(),
                Value::Real(f) => {
                    if f.fract() == 0.0 {
                        format!("{:.1}", f)
                    } else {
                        f.to_string()
                    }
                }
                Value::String(s) => format!("\"{}\"", s),
                Value::Symbol(s) => s.clone(),
                Value::Boolean(b) => if *b { "True".to_string() } else { "False".to_string() },
                Value::Missing => "Missing".to_string(),
                Value::Object(_) => "Object[...]".to_string(),
                Value::List(_) => "{...}".to_string(), // Nested lists simplified
                Value::Function(name) => format!("Function[{}]", name),
                Value::LyObj(obj) => format!("{}[...]", obj.type_name()),
                Value::Quote(expr) => format!("Hold[{:?}]", expr),
                Value::Pattern(pattern) => format!("{}", pattern),
                Value::Rule { lhs: _, rhs: _ } => "Rule[...]".to_string(),
                Value::PureFunction { .. } => "PureFunction[...]".to_string(),
                Value::Slot { .. } => "Slot[...]".to_string(),
            }).collect();
            format!("{{{}}}", formatted_items.join(", "))
        }
        Value::Function(name) => format!("Function[{}]", name),
        Value::Object(_) => "Object[...]".to_string(),
        Value::LyObj(obj) => format!("{}[...]", obj.type_name()),
        Value::Quote(expr) => format!("Hold[{:?}]", expr),
        Value::Pattern(pattern) => format!("{}", pattern),
        Value::Rule { lhs: _, rhs: _ } => "Rule[...]".to_string(),
        Value::PureFunction { .. } => "PureFunction[...]".to_string(),
        Value::Slot { .. } => "Slot[...]".to_string(),
    };

    Ok(Value::String(string_repr))
}

/// Conditional evaluation function
/// Usage: If[True, "yes", "no"] -> "yes", If[False, 1, 0] -> 0
pub fn if_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let condition = &args[0];
    let true_value = &args[1];
    let false_value = &args[2];

    let is_true = match condition {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => {
            return Err(VmError::TypeError {
                expected: "Boolean, Integer, or Real condition".to_string(),
                actual: format!("{:?}", condition),
            });
        }
    };

    if is_true {
        Ok(true_value.clone())
    } else {
        Ok(false_value.clone())
    }
}

/// Generate random real number between 0 and 1
/// Usage: RandomReal[] -> 0.42384... (random)
pub fn random_real(_args: &[Value]) -> VmResult<Value> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    Ok(Value::Real(rng.gen::<f64>()))
}

/// Get current date and time as string
/// Usage: DateString[] -> "2024-01-15T14:30:45"
pub fn date_string(_args: &[Value]) -> VmResult<Value> {
    use std::time::{SystemTime, UNIX_EPOCH};
    
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| VmError::Runtime("Failed to get current time".to_string()))?;
    
    // Simple ISO-like format (simplified since we don't have chrono)
    let secs = now.as_secs();
    let days_since_epoch = secs / 86400;
    let seconds_today = secs % 86400;
    let hours = seconds_today / 3600;
    let minutes = (seconds_today % 3600) / 60;
    let seconds = seconds_today % 60;
    
    // Approximate date calculation (simplified)
    let year = 1970 + (days_since_epoch / 365);
    let month = ((days_since_epoch % 365) / 30) + 1;
    let day = ((days_since_epoch % 365) % 30) + 1;
    
    Ok(Value::String(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        year, month, day, hours, minutes, seconds
    )))
}

/// Boolean NOT operation
/// Usage: Not[True] -> False, Not[False] -> True
pub fn not_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::Boolean(b) => Ok(Value::Boolean(!b)),
        Value::Integer(n) => Ok(Value::Boolean(*n == 0)),
        Value::Real(f) => Ok(Value::Boolean(*f == 0.0)),
        _ => Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Boolean AND operation
/// Usage: And[True, False] -> False, And[True, True] -> True
pub fn and_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[0] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let b = match &args[1] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(a && b))
}

/// Boolean OR operation
/// Usage: Or[True, False] -> True, Or[False, False] -> False
pub fn or_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[0] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let b = match &args[1] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(a || b))
}

/// Boolean XOR operation
/// Usage: Xor[True, False] -> True, Xor[True, True] -> False
pub fn xor_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let a = match &args[0] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let b = match &args[1] {
        Value::Boolean(b) => *b,
        Value::Integer(n) => *n != 0,
        Value::Real(f) => *f != 0.0,
        _ => return Err(VmError::TypeError {
            expected: "Boolean, Integer, or Real".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    Ok(Value::Boolean(a ^ b))
}

/// Greater than comparison
/// Usage: Greater[5, 3] -> True, Greater[2, 5] -> False
pub fn greater(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let result = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x > y,
        (Value::Integer(x), Value::Real(y)) => (*x as f64) > *y,
        (Value::Real(x), Value::Integer(y)) => *x > (*y as f64),
        (Value::Real(x), Value::Real(y)) => x > y,
        _ => {
            return Err(VmError::TypeError {
                expected: "Numbers".to_string(),
                actual: format!("{:?}, {:?}", a, b),
            });
        }
    };

    Ok(Value::Boolean(result))
}

/// Less than comparison
/// Usage: Less[3, 5] -> True, Less[5, 2] -> False
pub fn less(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let result = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x < y,
        (Value::Integer(x), Value::Real(y)) => (*x as f64) < *y,
        (Value::Real(x), Value::Integer(y)) => *x < (*y as f64),
        (Value::Real(x), Value::Real(y)) => x < y,
        _ => {
            return Err(VmError::TypeError {
                expected: "Numbers".to_string(),
                actual: format!("{:?}, {:?}", a, b),
            });
        }
    };

    Ok(Value::Boolean(result))
}

/// Equal comparison
/// Usage: Equal[5, 5] -> True, Equal[3, 5] -> False
pub fn equal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let result = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x == y,
        (Value::Integer(x), Value::Real(y)) => (*x as f64) == *y,
        (Value::Real(x), Value::Integer(y)) => *x == (*y as f64),
        (Value::Real(x), Value::Real(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::Boolean(x), Value::Boolean(y)) => x == y,
        (Value::Symbol(x), Value::Symbol(y)) => x == y,
        _ => false, // Different types are not equal
    };

    Ok(Value::Boolean(result))
}

/// Not equal comparison
/// Usage: Unequal[5, 3] -> True, Unequal[5, 5] -> False
pub fn unequal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Use the equal function and negate the result
    match equal(args)? {
        Value::Boolean(b) => Ok(Value::Boolean(!b)),
        _ => unreachable!("equal function should always return boolean"),
    }
}

/// Greater than or equal comparison
/// Usage: GreaterEqual[5, 5] -> True, GreaterEqual[3, 5] -> False
pub fn greater_equal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let result = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x >= y,
        (Value::Integer(x), Value::Real(y)) => (*x as f64) >= *y,
        (Value::Real(x), Value::Integer(y)) => *x >= (*y as f64),
        (Value::Real(x), Value::Real(y)) => x >= y,
        _ => {
            return Err(VmError::TypeError {
                expected: "Numbers".to_string(),
                actual: format!("{:?}, {:?}", a, b),
            });
        }
    };

    Ok(Value::Boolean(result))
}

/// Less than or equal comparison
/// Usage: LessEqual[3, 5] -> True, LessEqual[5, 3] -> False
pub fn less_equal(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let (a, b) = (&args[0], &args[1]);
    
    let result = match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => x <= y,
        (Value::Integer(x), Value::Real(y)) => (*x as f64) <= *y,
        (Value::Real(x), Value::Integer(y)) => *x <= (*y as f64),
        (Value::Real(x), Value::Real(y)) => x <= y,
        _ => {
            return Err(VmError::TypeError {
                expected: "Numbers".to_string(),
                actual: format!("{:?}, {:?}", a, b),
            });
        }
    };

    Ok(Value::Boolean(result))
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
