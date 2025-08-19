//! Symbolic calculus functions for the Lyra standard library
//! 
//! This module implements symbolic differentiation and integration functions
//! following Wolfram Language syntax and semantics.
//!
//! Functions provided:
//! - `D[expr, var]` - Symbolic derivative of expression with respect to variable
//! - `Integrate[expr, var]` - Symbolic indefinite integral
//! - `Integrate[expr, {var, a, b}]` - Symbolic definite integral

use crate::vm::{Value, VmError, VmResult};

/// Symbolic derivative: D[expr, var]
/// 
/// Computes the symbolic derivative of an expression with respect to a variable.
/// 
/// Examples:
/// - `D[x^2, x]` → `2*x` 
/// - `D[Sin[x], x]` → `Cos[x]`
/// - `D[x*y, x]` → `y`
/// - `D[Exp[x^2], x]` → `2*x*Exp[x^2]` (chain rule)
pub fn d(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (expression, variable)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Extract the expression and variable
    let expr = parse_expression_from_value(&args[0])?;
    let var = parse_variable_from_value(&args[1])?;
    
    // Compute symbolic derivative
    let derivative = symbolic_derivative(&expr, &var)?;
    
    // Convert back to Value
    expression_to_value(&derivative)
}

/// Symbolic indefinite integral: Integrate[expr, var] 
///
/// Computes the symbolic indefinite integral of an expression.
///
/// Examples:
/// - `Integrate[x^2, x]` → `x^3/3`
/// - `Integrate[Sin[x], x]` → `-Cos[x]`
/// - `Integrate[1/x, x]` → `Log[x]`
pub fn integrate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (expression, variable)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let expr = parse_expression_from_value(&args[0])?;
    let var = parse_variable_from_value(&args[1])?;
    
    // Compute symbolic integral
    let integral = symbolic_integral(&expr, &var)?;
    
    // Convert back to Value
    expression_to_value(&integral)
}

/// Symbolic definite integral: Integrate[expr, {var, a, b}]
///
/// Computes the definite integral over the specified bounds.
///
/// Examples:
/// - `Integrate[x^2, {x, 0, 1}]` → `1/3`
/// - `Integrate[Sin[x], {x, 0, Pi}]` → `2`
pub fn integrate_definite(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (expression, {var, a, b})".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let expr = parse_expression_from_value(&args[0])?;
    let bounds = parse_integration_bounds(&args[1])?;
    
    // First compute the indefinite integral
    let indefinite = symbolic_integral(&expr, &bounds.variable)?;
    
    // Evaluate at bounds: F(b) - F(a)
    let upper_bound = substitute_variable(&indefinite, &bounds.variable, &bounds.upper)?;
    let lower_bound = substitute_variable(&indefinite, &bounds.variable, &bounds.lower)?;
    
    // Compute difference
    let difference = subtract_expressions(&upper_bound, &lower_bound)?;
    let simplified = simplify_expression(&difference)?;
    
    expression_to_value(&simplified)
}

// Core symbolic differentiation engine
fn symbolic_derivative(expr: &SymbolicExpr, var: &str) -> VmResult<SymbolicExpr> {
    match expr {
        // Constant rule: d/dx[c] = 0
        SymbolicExpr::Constant(_) => Ok(SymbolicExpr::Constant(0.0)),
        
        // Variable rule: d/dx[x] = 1, d/dx[y] = 0
        SymbolicExpr::Variable(name) => {
            if name == var {
                Ok(SymbolicExpr::Constant(1.0))
            } else {
                Ok(SymbolicExpr::Constant(0.0))
            }
        }
        
        // Power rule: d/dx[x^n] = n*x^(n-1)
        SymbolicExpr::Power(base, exp) => {
            match (base.as_ref(), exp.as_ref()) {
                (SymbolicExpr::Variable(name), SymbolicExpr::Constant(n)) if name == var => {
                    // Simple power rule: x^n → n*x^(n-1)
                    if *n == 1.0 {
                        Ok(SymbolicExpr::Constant(1.0))
                    } else if *n == 0.0 {
                        Ok(SymbolicExpr::Constant(0.0))
                    } else {
                        let coeff = SymbolicExpr::Constant(*n);
                        let new_exp = n - 1.0;
                        
                        // Simplify x^1 to x
                        let base_term = if new_exp == 1.0 {
                            base.as_ref().clone()
                        } else if new_exp == 0.0 {
                            SymbolicExpr::Constant(1.0)
                        } else {
                            SymbolicExpr::Power(base.clone(), Box::new(SymbolicExpr::Constant(new_exp)))
                        };
                        
                        // Simplify coefficient * 1 to coefficient
                        if matches!(base_term, SymbolicExpr::Constant(1.0)) {
                            Ok(coeff)
                        } else {
                            Ok(SymbolicExpr::Multiply(Box::new(coeff), Box::new(base_term)))
                        }
                    }
                }
                _ => {
                    // General power rule with chain rule: d/dx[u^v] = u^v * (v'*ln(u) + v*u'/u)
                    let u = base.as_ref();
                    let v = exp.as_ref();
                    let u_prime = symbolic_derivative(u, var)?;
                    let v_prime = symbolic_derivative(v, var)?;
                    
                    // u^v term
                    let original = expr.clone();
                    
                    // v'*ln(u) term
                    let ln_u = SymbolicExpr::Function("Log".to_string(), vec![u.clone()]);
                    let v_prime_ln_u = SymbolicExpr::Multiply(Box::new(v_prime), Box::new(ln_u));
                    
                    // v*u'/u term
                    let v_u_prime = SymbolicExpr::Multiply(Box::new(v.clone()), Box::new(u_prime));
                    let v_u_prime_over_u = SymbolicExpr::Divide(Box::new(v_u_prime), Box::new(u.clone()));
                    
                    // Combine: u^v * (v'*ln(u) + v*u'/u)
                    let sum = SymbolicExpr::Add(Box::new(v_prime_ln_u), Box::new(v_u_prime_over_u));
                    Ok(SymbolicExpr::Multiply(Box::new(original), Box::new(sum)))
                }
            }
        }
        
        // Sum rule: d/dx[f + g] = f' + g'
        SymbolicExpr::Add(left, right) => {
            let left_derivative = symbolic_derivative(left, var)?;
            let right_derivative = symbolic_derivative(right, var)?;
            Ok(SymbolicExpr::Add(Box::new(left_derivative), Box::new(right_derivative)))
        }
        
        // Difference rule: d/dx[f - g] = f' - g'
        SymbolicExpr::Subtract(left, right) => {
            let left_derivative = symbolic_derivative(left, var)?;
            let right_derivative = symbolic_derivative(right, var)?;
            Ok(SymbolicExpr::Subtract(Box::new(left_derivative), Box::new(right_derivative)))
        }
        
        // Product rule: d/dx[f*g] = f'*g + f*g'
        SymbolicExpr::Multiply(left, right) => {
            let f = left.as_ref();
            let g = right.as_ref();
            let f_prime = symbolic_derivative(f, var)?;
            let g_prime = symbolic_derivative(g, var)?;
            
            let term1 = SymbolicExpr::Multiply(Box::new(f_prime), Box::new(g.clone()));
            let term2 = SymbolicExpr::Multiply(Box::new(f.clone()), Box::new(g_prime));
            
            Ok(SymbolicExpr::Add(Box::new(term1), Box::new(term2)))
        }
        
        // Quotient rule: d/dx[f/g] = (f'*g - f*g') / g^2
        SymbolicExpr::Divide(numerator, denominator) => {
            let f = numerator.as_ref();
            let g = denominator.as_ref();
            let f_prime = symbolic_derivative(f, var)?;
            let g_prime = symbolic_derivative(g, var)?;
            
            // f'*g - f*g'
            let term1 = SymbolicExpr::Multiply(Box::new(f_prime), Box::new(g.clone()));
            let term2 = SymbolicExpr::Multiply(Box::new(f.clone()), Box::new(g_prime));
            let numerator_result = SymbolicExpr::Subtract(Box::new(term1), Box::new(term2));
            
            // g^2
            let g_squared = SymbolicExpr::Power(Box::new(g.clone()), Box::new(SymbolicExpr::Constant(2.0)));
            
            Ok(SymbolicExpr::Divide(Box::new(numerator_result), Box::new(g_squared)))
        }
        
        // Function derivatives with chain rule
        SymbolicExpr::Function(name, args) => {
            if args.len() != 1 {
                return Err(VmError::TypeError {
                    expected: "single-argument function for differentiation".to_string(),
                    actual: format!("function {} with {} arguments", name, args.len()),
                });
            }
            
            let inner = &args[0];
            let inner_derivative = symbolic_derivative(inner, var)?;
            
            let outer_derivative = match name.as_str() {
                "Sin" => SymbolicExpr::Function("Cos".to_string(), vec![inner.clone()]),
                "Cos" => {
                    let neg_sin = SymbolicExpr::Function("Sin".to_string(), vec![inner.clone()]);
                    SymbolicExpr::Multiply(Box::new(SymbolicExpr::Constant(-1.0)), Box::new(neg_sin))
                }
                "Tan" => {
                    // d/dx[tan(x)] = sec²(x) = 1/cos²(x)
                    let cos_inner = SymbolicExpr::Function("Cos".to_string(), vec![inner.clone()]);
                    let cos_squared = SymbolicExpr::Power(Box::new(cos_inner), Box::new(SymbolicExpr::Constant(2.0)));
                    SymbolicExpr::Divide(Box::new(SymbolicExpr::Constant(1.0)), Box::new(cos_squared))
                }
                "Exp" => SymbolicExpr::Function("Exp".to_string(), vec![inner.clone()]),
                "Log" => {
                    // d/dx[log(x)] = 1/x
                    SymbolicExpr::Divide(Box::new(SymbolicExpr::Constant(1.0)), Box::new(inner.clone()))
                }
                "Sqrt" => {
                    // d/dx[sqrt(x)] = 1/(2*sqrt(x))
                    let two_sqrt = SymbolicExpr::Multiply(
                        Box::new(SymbolicExpr::Constant(2.0)),
                        Box::new(SymbolicExpr::Function("Sqrt".to_string(), vec![inner.clone()]))
                    );
                    SymbolicExpr::Divide(Box::new(SymbolicExpr::Constant(1.0)), Box::new(two_sqrt))
                }
                _ => {
                    return Err(VmError::TypeError {
                        expected: "known function (Sin, Cos, Tan, Exp, Log, Sqrt)".to_string(),
                        actual: format!("unknown function: {}", name),
                    });
                }
            };
            
            // Chain rule: multiply by derivative of inner function
            // Simplify when inner derivative is 1
            match inner_derivative {
                SymbolicExpr::Constant(1.0) => Ok(outer_derivative),
                _ => Ok(SymbolicExpr::Multiply(Box::new(outer_derivative), Box::new(inner_derivative)))
            }
        }
    }
}

// Core symbolic integration engine (basic rules)
fn symbolic_integral(expr: &SymbolicExpr, var: &str) -> VmResult<SymbolicExpr> {
    match expr {
        // ∫c dx = c*x
        SymbolicExpr::Constant(c) => {
            let x = SymbolicExpr::Variable(var.to_string());
            Ok(SymbolicExpr::Multiply(Box::new(SymbolicExpr::Constant(*c)), Box::new(x)))
        }
        
        // ∫x dx = x²/2
        SymbolicExpr::Variable(name) if name == var => {
            let x_squared = SymbolicExpr::Power(
                Box::new(SymbolicExpr::Variable(var.to_string())), 
                Box::new(SymbolicExpr::Constant(2.0))
            );
            Ok(SymbolicExpr::Divide(Box::new(x_squared), Box::new(SymbolicExpr::Constant(2.0))))
        }
        
        // ∫y dx = y*x (if y ≠ x)
        SymbolicExpr::Variable(name) => {
            let y = SymbolicExpr::Variable(name.clone());
            let x = SymbolicExpr::Variable(var.to_string());
            Ok(SymbolicExpr::Multiply(Box::new(y), Box::new(x)))
        }
        
        // Power rule: ∫x^n dx = x^(n+1)/(n+1) (n ≠ -1)
        SymbolicExpr::Power(base, exp) => {
            match (base.as_ref(), exp.as_ref()) {
                (SymbolicExpr::Variable(name), SymbolicExpr::Constant(n)) if name == var => {
                    if (*n - (-1.0)).abs() < 1e-10 {
                        // Special case: ∫1/x dx = log(x)
                        Ok(SymbolicExpr::Function("Log".to_string(), vec![base.as_ref().clone()]))
                    } else {
                        // General power rule
                        let new_exp = n + 1.0;
                        let integrated_power = SymbolicExpr::Power(
                            base.clone(), 
                            Box::new(SymbolicExpr::Constant(new_exp))
                        );
                        Ok(SymbolicExpr::Divide(
                            Box::new(integrated_power), 
                            Box::new(SymbolicExpr::Constant(new_exp))
                        ))
                    }
                }
                _ => {
                    // Complex power - not implemented
                    Err(VmError::TypeError {
                        expected: "simple power integration x^n".to_string(),
                        actual: format!("complex power: {:?}^{:?}", base, exp),
                    })
                }
            }
        }
        
        // Sum rule: ∫(f + g) dx = ∫f dx + ∫g dx
        SymbolicExpr::Add(left, right) => {
            let left_integral = symbolic_integral(left, var)?;
            let right_integral = symbolic_integral(right, var)?;
            Ok(SymbolicExpr::Add(Box::new(left_integral), Box::new(right_integral)))
        }
        
        // Difference rule: ∫(f - g) dx = ∫f dx - ∫g dx
        SymbolicExpr::Subtract(left, right) => {
            let left_integral = symbolic_integral(left, var)?;
            let right_integral = symbolic_integral(right, var)?;
            Ok(SymbolicExpr::Subtract(Box::new(left_integral), Box::new(right_integral)))
        }
        
        // Constant multiple: ∫c*f dx = c*∫f dx
        SymbolicExpr::Multiply(left, right) => {
            match (left.as_ref(), right.as_ref()) {
                (SymbolicExpr::Constant(_), _) => {
                    let integral = symbolic_integral(right, var)?;
                    Ok(SymbolicExpr::Multiply(left.clone(), Box::new(integral)))
                }
                (_, SymbolicExpr::Constant(_)) => {
                    let integral = symbolic_integral(left, var)?;
                    Ok(SymbolicExpr::Multiply(Box::new(integral), right.clone()))
                }
                _ => {
                    // General product integration is complex - not implemented
                    Err(VmError::TypeError {
                        expected: "constant multiple or simple product".to_string(),
                        actual: format!("complex product: {:?} * {:?}", left, right),
                    })
                }
            }
        }
        
        // Basic function integrals
        SymbolicExpr::Function(name, args) => {
            if args.len() != 1 {
                return Err(VmError::TypeError {
                    expected: "single-argument function".to_string(),
                    actual: format!("function with {} arguments", args.len()),
                });
            }
            
            let inner = &args[0];
            match (name.as_str(), inner) {
                // ∫sin(x) dx = -cos(x)
                ("Sin", SymbolicExpr::Variable(name)) if name == var => {
                    let cos_x = SymbolicExpr::Function("Cos".to_string(), vec![inner.clone()]);
                    Ok(SymbolicExpr::Multiply(Box::new(SymbolicExpr::Constant(-1.0)), Box::new(cos_x)))
                }
                
                // ∫cos(x) dx = sin(x)
                ("Cos", SymbolicExpr::Variable(name)) if name == var => {
                    Ok(SymbolicExpr::Function("Sin".to_string(), vec![inner.clone()]))
                }
                
                // ∫exp(x) dx = exp(x)
                ("Exp", SymbolicExpr::Variable(name)) if name == var => {
                    Ok(SymbolicExpr::Function("Exp".to_string(), vec![inner.clone()]))
                }
                
                // ∫1/x dx = log(x)
                ("Divide", _) if matches!(inner, SymbolicExpr::Variable(name) if name == var) => {
                    Ok(SymbolicExpr::Function("Log".to_string(), vec![inner.clone()]))
                }
                
                _ => {
                    Err(VmError::TypeError {
                        expected: "integrable function".to_string(),
                        actual: format!("function: {}[{:?}]", name, inner),
                    })
                }
            }
        }
        
        _ => {
            Err(VmError::TypeError {
                expected: "integrable expression".to_string(),
                actual: format!("{:?}", expr),
            })
        }
    }
}

// Internal symbolic expression representation
#[derive(Debug, Clone)]
enum SymbolicExpr {
    Constant(f64),
    Variable(String),
    Add(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Subtract(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Multiply(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Divide(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Power(Box<SymbolicExpr>, Box<SymbolicExpr>),
    Function(String, Vec<SymbolicExpr>),
}

// Integration bounds for definite integrals
#[derive(Debug)]
struct IntegrationBounds {
    variable: String,
    lower: SymbolicExpr,
    upper: SymbolicExpr,
}

// Helper functions for parsing and conversion
fn parse_expression_from_value(value: &Value) -> VmResult<SymbolicExpr> {
    match value {
        Value::Integer(n) => Ok(SymbolicExpr::Constant(*n as f64)),
        Value::Real(r) => Ok(SymbolicExpr::Constant(*r)),
        Value::Symbol(name) => Ok(SymbolicExpr::Variable(name.clone())),
        Value::List(items) if items.len() >= 2 => {
            // Parse function calls like ["+", x, y] or ["Sin", x]
            if let Value::Symbol(op) = &items[0] {
                match op.as_str() {
                    "Plus" | "+" => {
                        if items.len() != 3 {
                            return Err(VmError::TypeError {
                                expected: "binary Plus operation".to_string(),
                                actual: format!("{} arguments", items.len() - 1),
                            });
                        }
                        let left = parse_expression_from_value(&items[1])?;
                        let right = parse_expression_from_value(&items[2])?;
                        Ok(SymbolicExpr::Add(Box::new(left), Box::new(right)))
                    }
                    "Times" | "*" => {
                        if items.len() != 3 {
                            return Err(VmError::TypeError {
                                expected: "binary Times operation".to_string(),
                                actual: format!("{} arguments", items.len() - 1),
                            });
                        }
                        let left = parse_expression_from_value(&items[1])?;
                        let right = parse_expression_from_value(&items[2])?;
                        Ok(SymbolicExpr::Multiply(Box::new(left), Box::new(right)))
                    }
                    "Power" | "^" => {
                        if items.len() != 3 {
                            return Err(VmError::TypeError {
                                expected: "binary Power operation".to_string(),
                                actual: format!("{} arguments", items.len() - 1),
                            });
                        }
                        let base = parse_expression_from_value(&items[1])?;
                        let exp = parse_expression_from_value(&items[2])?;
                        Ok(SymbolicExpr::Power(Box::new(base), Box::new(exp)))
                    }
                    func_name => {
                        // Function call
                        let mut args = Vec::new();
                        for arg in &items[1..] {
                            args.push(parse_expression_from_value(arg)?);
                        }
                        Ok(SymbolicExpr::Function(func_name.to_string(), args))
                    }
                }
            } else {
                Err(VmError::TypeError {
                    expected: "symbolic expression".to_string(),
                    actual: format!("invalid expression format: {:?}", value),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "symbolic expression".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

fn parse_variable_from_value(value: &Value) -> VmResult<String> {
    match value {
        Value::Symbol(name) => Ok(name.clone()),
        _ => Err(VmError::TypeError {
            expected: "variable symbol".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

fn parse_integration_bounds(value: &Value) -> VmResult<IntegrationBounds> {
    match value {
        Value::List(items) if items.len() == 3 => {
            let var = parse_variable_from_value(&items[0])?;
            let lower = parse_expression_from_value(&items[1])?;
            let upper = parse_expression_from_value(&items[2])?;
            Ok(IntegrationBounds {
                variable: var,
                lower,
                upper,
            })
        }
        _ => Err(VmError::TypeError {
            expected: "integration bounds {var, a, b}".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

fn expression_to_value(expr: &SymbolicExpr) -> VmResult<Value> {
    match expr {
        SymbolicExpr::Constant(c) => Ok(Value::Real(*c)),
        SymbolicExpr::Variable(name) => Ok(Value::Symbol(name.clone())),
        SymbolicExpr::Add(left, right) => {
            let left_val = expression_to_value(left)?;
            let right_val = expression_to_value(right)?;
            Ok(Value::List(vec![Value::Symbol("Plus".to_string()), left_val, right_val]))
        }
        SymbolicExpr::Subtract(left, right) => {
            let left_val = expression_to_value(left)?;
            let right_val = expression_to_value(right)?;
            Ok(Value::List(vec![Value::Symbol("Subtract".to_string()), left_val, right_val]))
        }
        SymbolicExpr::Multiply(left, right) => {
            let left_val = expression_to_value(left)?;
            let right_val = expression_to_value(right)?;
            Ok(Value::List(vec![Value::Symbol("Times".to_string()), left_val, right_val]))
        }
        SymbolicExpr::Divide(left, right) => {
            let left_val = expression_to_value(left)?;
            let right_val = expression_to_value(right)?;
            Ok(Value::List(vec![Value::Symbol("Divide".to_string()), left_val, right_val]))
        }
        SymbolicExpr::Power(base, exp) => {
            let base_val = expression_to_value(base)?;
            let exp_val = expression_to_value(exp)?;
            Ok(Value::List(vec![Value::Symbol("Power".to_string()), base_val, exp_val]))
        }
        SymbolicExpr::Function(name, args) => {
            let mut result = vec![Value::Symbol(name.clone())];
            for arg in args {
                result.push(expression_to_value(arg)?);
            }
            Ok(Value::List(result))
        }
    }
}

// Helper functions for definite integration
fn substitute_variable(expr: &SymbolicExpr, var: &str, value: &SymbolicExpr) -> VmResult<SymbolicExpr> {
    match expr {
        SymbolicExpr::Constant(c) => Ok(SymbolicExpr::Constant(*c)),
        SymbolicExpr::Variable(name) => {
            if name == var {
                Ok(value.clone())
            } else {
                Ok(SymbolicExpr::Variable(name.clone()))
            }
        }
        SymbolicExpr::Add(left, right) => {
            let left_sub = substitute_variable(left, var, value)?;
            let right_sub = substitute_variable(right, var, value)?;
            Ok(SymbolicExpr::Add(Box::new(left_sub), Box::new(right_sub)))
        }
        SymbolicExpr::Subtract(left, right) => {
            let left_sub = substitute_variable(left, var, value)?;
            let right_sub = substitute_variable(right, var, value)?;
            Ok(SymbolicExpr::Subtract(Box::new(left_sub), Box::new(right_sub)))
        }
        SymbolicExpr::Multiply(left, right) => {
            let left_sub = substitute_variable(left, var, value)?;
            let right_sub = substitute_variable(right, var, value)?;
            Ok(SymbolicExpr::Multiply(Box::new(left_sub), Box::new(right_sub)))
        }
        SymbolicExpr::Divide(left, right) => {
            let left_sub = substitute_variable(left, var, value)?;
            let right_sub = substitute_variable(right, var, value)?;
            Ok(SymbolicExpr::Divide(Box::new(left_sub), Box::new(right_sub)))
        }
        SymbolicExpr::Power(base, exp) => {
            let base_sub = substitute_variable(base, var, value)?;
            let exp_sub = substitute_variable(exp, var, value)?;
            Ok(SymbolicExpr::Power(Box::new(base_sub), Box::new(exp_sub)))
        }
        SymbolicExpr::Function(name, args) => {
            let mut sub_args = Vec::new();
            for arg in args {
                sub_args.push(substitute_variable(arg, var, value)?);
            }
            Ok(SymbolicExpr::Function(name.clone(), sub_args))
        }
    }
}

fn subtract_expressions(left: &SymbolicExpr, right: &SymbolicExpr) -> VmResult<SymbolicExpr> {
    Ok(SymbolicExpr::Subtract(Box::new(left.clone()), Box::new(right.clone())))
}

fn simplify_expression(expr: &SymbolicExpr) -> VmResult<SymbolicExpr> {
    // Basic simplification - more sophisticated simplification would be needed for production
    match expr {
        SymbolicExpr::Add(left, right) => {
            match (left.as_ref(), right.as_ref()) {
                (SymbolicExpr::Constant(0.0), _) => Ok(right.as_ref().clone()),
                (_, SymbolicExpr::Constant(0.0)) => Ok(left.as_ref().clone()),
                (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) => Ok(SymbolicExpr::Constant(a + b)),
                _ => Ok(expr.clone())
            }
        }
        SymbolicExpr::Subtract(left, right) => {
            match (left.as_ref(), right.as_ref()) {
                (_, SymbolicExpr::Constant(0.0)) => Ok(left.as_ref().clone()),
                (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) => Ok(SymbolicExpr::Constant(a - b)),
                _ => Ok(expr.clone())
            }
        }
        SymbolicExpr::Multiply(left, right) => {
            match (left.as_ref(), right.as_ref()) {
                (SymbolicExpr::Constant(1.0), _) => Ok(right.as_ref().clone()),
                (_, SymbolicExpr::Constant(1.0)) => Ok(left.as_ref().clone()),
                (SymbolicExpr::Constant(0.0), _) | (_, SymbolicExpr::Constant(0.0)) => Ok(SymbolicExpr::Constant(0.0)),
                (SymbolicExpr::Constant(a), SymbolicExpr::Constant(b)) => Ok(SymbolicExpr::Constant(a * b)),
                _ => Ok(expr.clone())
            }
        }
        _ => Ok(expr.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_rule_derivative() {
        // D[x^2, x] should give 2*x
        let x_squared = Value::List(vec![
            Value::Symbol("Power".to_string()),
            Value::Symbol("x".to_string()),
            Value::Real(2.0),
        ]);
        let x = Value::Symbol("x".to_string());
        
        let result = d(&[x_squared, x]).unwrap();
        
        // Expected: 2*x
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Symbol("Times".to_string()));
                assert_eq!(items[1], Value::Real(2.0));
                assert_eq!(items[2], Value::Symbol("x".to_string()));
            }
            _ => panic!("Expected list for derivative result"),
        }
    }

    #[test]
    fn test_constant_derivative() {
        // D[5, x] should give 0
        let constant = Value::Real(5.0);
        let x = Value::Symbol("x".to_string());
        
        let result = d(&[constant, x]).unwrap();
        assert_eq!(result, Value::Real(0.0));
    }

    #[test]
    fn test_variable_derivative() {
        // D[x, x] should give 1
        let x = Value::Symbol("x".to_string());
        
        let result = d(&[x.clone(), x]).unwrap();
        assert_eq!(result, Value::Real(1.0));
    }

    #[test]
    fn test_sin_derivative() {
        // D[Sin[x], x] should give Cos[x]
        let sin_x = Value::List(vec![
            Value::Symbol("Sin".to_string()),
            Value::Symbol("x".to_string()),
        ]);
        let x = Value::Symbol("x".to_string());
        
        let result = d(&[sin_x, x]).unwrap();
        
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], Value::Symbol("Cos".to_string()));
                assert_eq!(items[1], Value::Symbol("x".to_string()));
            }
            _ => panic!("Expected function call for Sin derivative"),
        }
    }

    #[test]
    fn test_power_rule_integration() {
        // Integrate[x^2, x] should give x^3/3
        let x_squared = Value::List(vec![
            Value::Symbol("Power".to_string()),
            Value::Symbol("x".to_string()),
            Value::Real(2.0),
        ]);
        let x = Value::Symbol("x".to_string());
        
        let result = integrate(&[x_squared, x]).unwrap();
        
        // Expected: x^3/3
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Symbol("Divide".to_string()));
                // Should be x^3 / 3
            }
            _ => panic!("Expected division for integration result"),
        }
    }

    #[test]
    fn test_constant_integration() {
        // Integrate[5, x] should give 5*x
        let constant = Value::Real(5.0);
        let x = Value::Symbol("x".to_string());
        
        let result = integrate(&[constant, x]).unwrap();
        
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Symbol("Times".to_string()));
                assert_eq!(items[1], Value::Real(5.0));
                assert_eq!(items[2], Value::Symbol("x".to_string()));
            }
            _ => panic!("Expected multiplication for constant integration"),
        }
    }
}