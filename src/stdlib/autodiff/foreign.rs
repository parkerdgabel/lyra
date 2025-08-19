//! Foreign Objects for Automatic Differentiation
//!
//! This module provides foreign object implementations for dual numbers
//! that integrate with Lyra's VM.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;
use std::fmt;
use super::Dual;

/// Foreign object wrapper for dual numbers
#[derive(Debug, Clone)]
pub struct ForeignDual {
    pub dual: Dual,
}

impl ForeignDual {
    /// Create a new foreign dual number
    pub fn new(dual: Dual) -> Self {
        Self { dual }
    }
    
    /// Create a constant dual number
    pub fn constant(value: f64) -> Self {
        Self::new(Dual::constant(value))
    }
    
    /// Create a variable dual number
    pub fn variable(value: f64) -> Self {
        Self::new(Dual::variable(value))
    }
    
    /// Get the function value
    pub fn value(&self) -> f64 {
        self.dual.value()
    }
    
    /// Get the derivative value
    pub fn derivative(&self) -> f64 {
        self.dual.derivative()
    }
}

impl Foreign for ForeignDual {
    fn type_name(&self) -> &'static str {
        "Dual"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Value" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.value()))
            }
            "Derivative" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Real(self.derivative()))
            }
            "IsConstant" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.dual.is_constant()))
            }
            "IsVariable" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.dual.is_variable()))
            }
            "Sin" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let result = ForeignDual::new(self.dual.sin());
                Ok(Value::LyObj(LyObj::new(Box::new(result))))
            }
            "Cos" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let result = ForeignDual::new(self.dual.cos());
                Ok(Value::LyObj(LyObj::new(Box::new(result))))
            }
            "Exp" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let result = ForeignDual::new(self.dual.exp());
                Ok(Value::LyObj(LyObj::new(Box::new(result))))
            }
            "ReLU" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let result = ForeignDual::new(self.dual.relu());
                Ok(Value::LyObj(LyObj::new(Box::new(result))))
            }
            "Sigmoid" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let result = ForeignDual::new(self.dual.sigmoid());
                Ok(Value::LyObj(LyObj::new(Box::new(result))))
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

impl fmt::Display for ForeignDual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual[{}]", self.dual)
    }
}

/// Functions for creating autodiff foreign objects in Lyra
pub mod functions {
    use super::*;
    
    /// Create a dual number: Dual[value, derivative]
    pub fn dual(args: &[Value]) -> VmResult<Value> {
        match args.len() {
            1 => {
                // Single argument - create variable dual
                match &args[0] {
                    Value::Real(value) => {
                        let dual = ForeignDual::variable(*value);
                        Ok(Value::LyObj(LyObj::new(Box::new(dual))))
                    }
                    Value::Integer(value) => {
                        let dual = ForeignDual::variable(*value as f64);
                        Ok(Value::LyObj(LyObj::new(Box::new(dual))))
                    }
                    _ => Err(VmError::TypeError {
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            2 => {
                // Two arguments - create dual with specific value and derivative
                let value = match &args[0] {
                    Value::Real(v) => *v,
                    Value::Integer(v) => *v as f64,
                    _ => return Err(VmError::TypeError {
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let derivative = match &args[1] {
                    Value::Real(v) => *v,
                    Value::Integer(v) => *v as f64,
                    _ => return Err(VmError::TypeError {
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                let dual = ForeignDual::new(Dual::new(value, derivative));
                Ok(Value::LyObj(LyObj::new(Box::new(dual))))
            }
            _ => Err(VmError::TypeError {
                expected: "1 or 2 arguments".to_string(),
                actual: format!("{} arguments", args.len()),
            }),
        }
    }
    
    /// Create a constant dual number: DualConstant[value]
    pub fn dual_constant(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::TypeError {
                expected: "exactly 1 argument".to_string(),
                actual: format!("{} arguments", args.len()),
            });
        }
        
        let value = match &args[0] {
            Value::Real(v) => *v,
            Value::Integer(v) => *v as f64,
            _ => return Err(VmError::TypeError {
                expected: "Real or Integer".to_string(),
                actual: format!("{:?}", args[0]),
            }),
        };
        
        let dual = ForeignDual::constant(value);
        Ok(Value::LyObj(LyObj::new(Box::new(dual))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_foreign_dual_creation() {
        let dual = ForeignDual::variable(2.0);
        assert_eq!(dual.value(), 2.0);
        assert_eq!(dual.derivative(), 1.0);
        assert_eq!(dual.type_name(), "Dual");
        
        let const_dual = ForeignDual::constant(3.14);
        assert_eq!(const_dual.value(), 3.14);
        assert_eq!(const_dual.derivative(), 0.0);
    }
    
    #[test]
    fn test_foreign_dual_methods() {
        let dual = ForeignDual::variable(2.0);
        
        // Test Value method
        let result = dual.call_method("Value", &[]).unwrap();
        assert_eq!(result, Value::Real(2.0));
        
        // Test Derivative method
        let result = dual.call_method("Derivative", &[]).unwrap();
        assert_eq!(result, Value::Real(1.0));
        
        // Test IsVariable method
        let result = dual.call_method("IsVariable", &[]).unwrap();
        assert_eq!(result, Value::Boolean(true));
        
        // Test IsConstant method
        let result = dual.call_method("IsConstant", &[]).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }
    
    #[test]
    fn test_foreign_dual_transcendental() {
        let dual = ForeignDual::new(Dual::new(0.0, 1.0));
        
        // Test sine: sin(0) = 0, cos(0) = 1
        let result = dual.call_method("Sin", &[]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(sin_dual) = obj.downcast_ref::<ForeignDual>() {
                assert!((sin_dual.value() - 0.0).abs() < 1e-10);
                assert!((sin_dual.derivative() - 1.0).abs() < 1e-10);
            }
        }
        
        // Test exponential: exp(0) = 1, exp'(0) = 1
        let result = dual.call_method("Exp", &[]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(exp_dual) = obj.downcast_ref::<ForeignDual>() {
                assert!((exp_dual.value() - 1.0).abs() < 1e-10);
                assert!((exp_dual.derivative() - 1.0).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_dual_function() {
        // Test variable dual creation
        let result = functions::dual(&[Value::Real(2.0)]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(dual) = obj.downcast_ref::<ForeignDual>() {
                assert_eq!(dual.value(), 2.0);
                assert_eq!(dual.derivative(), 1.0);
            }
        }
        
        // Test dual with custom derivative
        let result = functions::dual(&[Value::Real(2.0), Value::Real(3.0)]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(dual) = obj.downcast_ref::<ForeignDual>() {
                assert_eq!(dual.value(), 2.0);
                assert_eq!(dual.derivative(), 3.0);
            }
        }
    }
    
    #[test]
    fn test_dual_constant_function() {
        let result = functions::dual_constant(&[Value::Real(3.14)]).unwrap();
        if let Value::LyObj(obj) = result {
            if let Some(dual) = obj.downcast_ref::<ForeignDual>() {
                assert_eq!(dual.value(), 3.14);
                assert_eq!(dual.derivative(), 0.0);
            }
        }
    }
}