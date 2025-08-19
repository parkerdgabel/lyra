//! Dual Number Implementation for Forward-Mode Automatic Differentiation
//!
//! Dual numbers are a mathematical extension that allows computation of derivatives
//! alongside function values. They enable forward-mode automatic differentiation
//! with exact gradients and minimal computational overhead.

use std::ops::{Add, Sub, Mul, Div, Neg};
use std::fmt;
use super::{AutodiffError, AutodiffResult};

/// Dual number for forward-mode automatic differentiation
///
/// A dual number has the form: a + b*ε where ε² = 0
/// - `value`: The function value (real part)
/// - `derivative`: The derivative value (dual part)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dual {
    /// The function value (real part)
    pub value: f64,
    /// The derivative value (dual part)
    pub derivative: f64,
}

impl Dual {
    /// Create a new dual number with given value and derivative
    pub fn new(value: f64, derivative: f64) -> Self {
        Self { value, derivative }
    }
    
    /// Create a constant dual number (derivative = 0)
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }
    
    /// Create a variable dual number (derivative = 1)
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }
    
    /// Get the function value
    pub fn value(&self) -> f64 {
        self.value
    }
    
    /// Get the derivative value
    pub fn derivative(&self) -> f64 {
        self.derivative
    }
    
    /// Check if this is a constant (derivative = 0)
    pub fn is_constant(&self) -> bool {
        self.derivative == 0.0
    }
    
    /// Check if this is a variable (derivative = 1)
    pub fn is_variable(&self) -> bool {
        self.derivative == 1.0
    }
    
    /// Apply a unary function with its derivative
    pub fn apply_unary<F, D>(self, func: F, deriv: D) -> Dual
    where
        F: FnOnce(f64) -> f64,
        D: FnOnce(f64) -> f64,
    {
        let new_value = func(self.value);
        let new_derivative = deriv(self.value) * self.derivative;
        Dual::new(new_value, new_derivative)
    }
    
    /// Apply a binary function with its partial derivatives
    pub fn apply_binary<F, D1, D2>(self, other: Dual, func: F, deriv1: D1, deriv2: D2) -> Dual
    where
        F: FnOnce(f64, f64) -> f64,
        D1: FnOnce(f64, f64) -> f64,
        D2: FnOnce(f64, f64) -> f64,
    {
        let new_value = func(self.value, other.value);
        let new_derivative = deriv1(self.value, other.value) * self.derivative
                           + deriv2(self.value, other.value) * other.derivative;
        Dual::new(new_value, new_derivative)
    }
    
    /// Power function: self^n
    pub fn powi(self, n: i32) -> Dual {
        if n == 0 {
            Dual::constant(1.0)
        } else if n == 1 {
            self
        } else {
            let new_value = self.value.powi(n);
            let new_derivative = (n as f64) * self.value.powi(n - 1) * self.derivative;
            Dual::new(new_value, new_derivative)
        }
    }
    
    /// Power function: self^other (general power)
    pub fn powf(self, other: Dual) -> AutodiffResult<Dual> {
        if self.value <= 0.0 && !other.is_constant() {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "Power with negative base and non-constant exponent".to_string(),
            });
        }
        
        if self.value == 0.0 && other.value <= 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "Zero raised to non-positive power".to_string(),
            });
        }
        
        let new_value = self.value.powf(other.value);
        
        if self.value == 0.0 {
            // Special case: 0^n where n > 0
            Ok(Dual::constant(0.0))
        } else {
            // General case: a^b
            // d/dx(a^b) = a^b * (b' * ln(a) + b * a'/a)
            let ln_a = self.value.ln();
            let new_derivative = new_value * (
                other.derivative * ln_a + 
                other.value * self.derivative / self.value
            );
            Ok(Dual::new(new_value, new_derivative))
        }
    }
    
    /// Natural exponential: e^self
    pub fn exp(self) -> Dual {
        let exp_val = self.value.exp();
        Dual::new(exp_val, exp_val * self.derivative)
    }
    
    /// Natural logarithm: ln(self)
    pub fn ln(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "Logarithm of non-positive number".to_string(),
            });
        }
        
        let new_value = self.value.ln();
        let new_derivative = self.derivative / self.value;
        Ok(Dual::new(new_value, new_derivative))
    }
    
    /// Square root: sqrt(self)
    pub fn sqrt(self) -> AutodiffResult<Dual> {
        if self.value < 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "Square root of negative number".to_string(),
            });
        }
        
        if self.value == 0.0 {
            if self.derivative == 0.0 {
                Ok(Dual::constant(0.0))
            } else {
                Err(AutodiffError::InvalidDualOperation {
                    reason: "Derivative of sqrt at zero is undefined".to_string(),
                })
            }
        } else {
            let sqrt_val = self.value.sqrt();
            let new_derivative = self.derivative / (2.0 * sqrt_val);
            Ok(Dual::new(sqrt_val, new_derivative))
        }
    }
    
    /// Sine: sin(self)
    pub fn sin(self) -> Dual {
        self.apply_unary(|x| x.sin(), |x| x.cos())
    }
    
    /// Cosine: cos(self)
    pub fn cos(self) -> Dual {
        self.apply_unary(|x| x.cos(), |x| -x.sin())
    }
    
    /// Tangent: tan(self)
    pub fn tan(self) -> Dual {
        let cos_val = self.value.cos();
        if cos_val == 0.0 {
            // Handle undefined tangent
            Dual::new(f64::INFINITY, f64::INFINITY)
        } else {
            self.apply_unary(|x| x.tan(), |x| 1.0 / (x.cos().powi(2)))
        }
    }
    
    /// Hyperbolic sine: sinh(self)
    pub fn sinh(self) -> Dual {
        self.apply_unary(|x| x.sinh(), |x| x.cosh())
    }
    
    /// Hyperbolic cosine: cosh(self)
    pub fn cosh(self) -> Dual {
        self.apply_unary(|x| x.cosh(), |x| x.sinh())
    }
    
    /// Hyperbolic tangent: tanh(self)
    pub fn tanh(self) -> Dual {
        self.apply_unary(|x| x.tanh(), |x| 1.0 / (x.cosh().powi(2)))
    }
    
    /// Absolute value: |self|
    pub fn abs(self) -> Dual {
        if self.value == 0.0 {
            // Derivative is undefined at zero, but we can use subgradient
            Dual::new(0.0, 0.0)
        } else {
            let sign = if self.value > 0.0 { 1.0 } else { -1.0 };
            Dual::new(self.value.abs(), sign * self.derivative)
        }
    }
    
    /// Minimum of two dual numbers
    pub fn min(self, other: Dual) -> Dual {
        if self.value < other.value {
            self
        } else if self.value > other.value {
            other
        } else {
            // Values are equal - use subgradient
            let min_deriv = self.derivative.min(other.derivative);
            Dual::new(self.value, min_deriv)
        }
    }
    
    /// Maximum of two dual numbers
    pub fn max(self, other: Dual) -> Dual {
        if self.value > other.value {
            self
        } else if self.value < other.value {
            other
        } else {
            // Values are equal - use subgradient
            let max_deriv = self.derivative.max(other.derivative);
            Dual::new(self.value, max_deriv)
        }
    }
    
    /// ReLU activation function: max(0, self)
    pub fn relu(self) -> Dual {
        if self.value > 0.0 {
            self
        } else {
            Dual::constant(0.0)
        }
    }
    
    /// Sigmoid activation function: 1 / (1 + e^(-self))
    pub fn sigmoid(self) -> Dual {
        let exp_neg_x = (-self).exp();
        let sigmoid_val = 1.0 / (1.0 + exp_neg_x.value);
        let sigmoid_deriv = sigmoid_val * (1.0 - sigmoid_val);
        Dual::new(sigmoid_val, sigmoid_deriv * self.derivative)
    }
}

impl Add for Dual {
    type Output = Dual;
    
    fn add(self, other: Dual) -> Dual {
        Dual::new(
            self.value + other.value,
            self.derivative + other.derivative,
        )
    }
}

impl Add<f64> for Dual {
    type Output = Dual;
    
    fn add(self, scalar: f64) -> Dual {
        Dual::new(self.value + scalar, self.derivative)
    }
}

impl Add<Dual> for f64 {
    type Output = Dual;
    
    fn add(self, dual: Dual) -> Dual {
        dual + self
    }
}

impl Sub for Dual {
    type Output = Dual;
    
    fn sub(self, other: Dual) -> Dual {
        Dual::new(
            self.value - other.value,
            self.derivative - other.derivative,
        )
    }
}

impl Sub<f64> for Dual {
    type Output = Dual;
    
    fn sub(self, scalar: f64) -> Dual {
        Dual::new(self.value - scalar, self.derivative)
    }
}

impl Sub<Dual> for f64 {
    type Output = Dual;
    
    fn sub(self, dual: Dual) -> Dual {
        Dual::new(self - dual.value, -dual.derivative)
    }
}

impl Mul for Dual {
    type Output = Dual;
    
    fn mul(self, other: Dual) -> Dual {
        Dual::new(
            self.value * other.value,
            self.value * other.derivative + self.derivative * other.value,
        )
    }
}

impl Mul<f64> for Dual {
    type Output = Dual;
    
    fn mul(self, scalar: f64) -> Dual {
        Dual::new(self.value * scalar, self.derivative * scalar)
    }
}

impl Mul<Dual> for f64 {
    type Output = Dual;
    
    fn mul(self, dual: Dual) -> Dual {
        dual * self
    }
}

impl Div for Dual {
    type Output = Dual;
    
    fn div(self, other: Dual) -> Dual {
        if other.value == 0.0 {
            // Handle division by zero
            Dual::new(f64::INFINITY, f64::INFINITY)
        } else {
            let quotient = self.value / other.value;
            let quotient_deriv = (self.derivative * other.value - self.value * other.derivative) 
                               / (other.value * other.value);
            Dual::new(quotient, quotient_deriv)
        }
    }
}

impl Div<f64> for Dual {
    type Output = Dual;
    
    fn div(self, scalar: f64) -> Dual {
        if scalar == 0.0 {
            Dual::new(f64::INFINITY, f64::INFINITY)
        } else {
            Dual::new(self.value / scalar, self.derivative / scalar)
        }
    }
}

impl Div<Dual> for f64 {
    type Output = Dual;
    
    fn div(self, dual: Dual) -> Dual {
        if dual.value == 0.0 {
            Dual::new(f64::INFINITY, f64::INFINITY)
        } else {
            Dual::new(
                self / dual.value,
                -self * dual.derivative / (dual.value * dual.value),
            )
        }
    }
}

impl Neg for Dual {
    type Output = Dual;
    
    fn neg(self) -> Dual {
        Dual::new(-self.value, -self.derivative)
    }
}

impl fmt::Display for Dual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.derivative == 0.0 {
            write!(f, "{}", self.value)
        } else if self.derivative == 1.0 {
            write!(f, "{} + ε", self.value)
        } else {
            write!(f, "{} + {}ε", self.value, self.derivative)
        }
    }
}

impl From<f64> for Dual {
    fn from(value: f64) -> Self {
        Dual::constant(value)
    }
}

impl From<i32> for Dual {
    fn from(value: i32) -> Self {
        Dual::constant(value as f64)
    }
}

impl From<Dual> for f64 {
    fn from(dual: Dual) -> Self {
        dual.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{PI, E};
    
    const EPSILON: f64 = 1e-10;
    
    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }
    
    #[test]
    fn test_dual_creation() {
        let d1 = Dual::new(3.0, 1.0);
        assert_eq!(d1.value, 3.0);
        assert_eq!(d1.derivative, 1.0);
        
        let d2 = Dual::constant(5.0);
        assert_eq!(d2.value, 5.0);
        assert_eq!(d2.derivative, 0.0);
        assert!(d2.is_constant());
        
        let d3 = Dual::variable(2.0);
        assert_eq!(d3.value, 2.0);
        assert_eq!(d3.derivative, 1.0);
        assert!(d3.is_variable());
    }
    
    #[test]
    fn test_basic_arithmetic() {
        let x = Dual::variable(3.0);
        let y = Dual::variable(2.0);
        
        // Addition: (x + y)' = x' + y' = 1 + 1 = 2
        let sum = x + y;
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.derivative, 2.0);
        
        // Subtraction: (x - y)' = x' - y' = 1 - 1 = 0
        let diff = x - y;
        assert_eq!(diff.value, 1.0);
        assert_eq!(diff.derivative, 0.0);
        
        // Multiplication: (x * y)' = x' * y + x * y' = 1 * 2 + 3 * 1 = 5
        let prod = x * y;
        assert_eq!(prod.value, 6.0);
        assert_eq!(prod.derivative, 5.0);
        
        // Division: (x / y)' = (x' * y - x * y') / y^2 = (1 * 2 - 3 * 1) / 4 = -1/4
        let quot = x / y;
        assert_eq!(quot.value, 1.5);
        assert_eq!(quot.derivative, -0.25);
    }
    
    #[test]
    fn test_scalar_operations() {
        let x = Dual::variable(3.0);
        
        let sum = x + 5.0;
        assert_eq!(sum.value, 8.0);
        assert_eq!(sum.derivative, 1.0);
        
        let prod = x * 2.0;
        assert_eq!(prod.value, 6.0);
        assert_eq!(prod.derivative, 2.0);
        
        let quot = x / 2.0;
        assert_eq!(quot.value, 1.5);
        assert_eq!(quot.derivative, 0.5);
    }
    
    #[test]
    fn test_power_functions() {
        let x = Dual::variable(2.0);
        
        // x^3, derivative should be 3*x^2 = 3*4 = 12
        let cubic = x.powi(3);
        assert_eq!(cubic.value, 8.0);
        assert_eq!(cubic.derivative, 12.0);
        
        // x^0 should be constant 1
        let zero_power = x.powi(0);
        assert_eq!(zero_power.value, 1.0);
        assert_eq!(zero_power.derivative, 0.0);
        
        // x^1 should be x itself
        let first_power = x.powi(1);
        assert_eq!(first_power.value, 2.0);
        assert_eq!(first_power.derivative, 1.0);
    }
    
    #[test]
    fn test_exponential_logarithm() {
        let x = Dual::variable(1.0);
        
        // e^x at x=1, derivative should be e^1 = e
        let exp_x = x.exp();
        assert!(approx_eq(exp_x.value, E));
        assert!(approx_eq(exp_x.derivative, E));
        
        // ln(e) = 1, derivative should be 1/e
        let e_val = Dual::variable(E);
        let ln_e = e_val.ln().unwrap();
        assert!(approx_eq(ln_e.value, 1.0));
        assert!(approx_eq(ln_e.derivative, 1.0 / E));
    }
    
    #[test]
    fn test_trigonometric() {
        let x = Dual::variable(PI / 2.0);
        
        // sin(π/2) = 1, derivative = cos(π/2) = 0
        let sin_x = x.sin();
        assert!(approx_eq(sin_x.value, 1.0));
        assert!(approx_eq(sin_x.derivative, 0.0));
        
        // cos(π/2) = 0, derivative = -sin(π/2) = -1
        let cos_x = x.cos();
        assert!(approx_eq(cos_x.value, 0.0));
        assert!(approx_eq(cos_x.derivative, -1.0));
    }
    
    #[test]
    fn test_hyperbolic() {
        let x = Dual::variable(0.0);
        
        // sinh(0) = 0, derivative = cosh(0) = 1
        let sinh_x = x.sinh();
        assert!(approx_eq(sinh_x.value, 0.0));
        assert!(approx_eq(sinh_x.derivative, 1.0));
        
        // cosh(0) = 1, derivative = sinh(0) = 0
        let cosh_x = x.cosh();
        assert!(approx_eq(cosh_x.value, 1.0));
        assert!(approx_eq(cosh_x.derivative, 0.0));
        
        // tanh(0) = 0, derivative = 1/cosh²(0) = 1
        let tanh_x = x.tanh();
        assert!(approx_eq(tanh_x.value, 0.0));
        assert!(approx_eq(tanh_x.derivative, 1.0));
    }
    
    #[test]
    fn test_activation_functions() {
        let x = Dual::variable(2.0);
        
        // ReLU(2) = 2, derivative = 1
        let relu_x = x.relu();
        assert_eq!(relu_x.value, 2.0);
        assert_eq!(relu_x.derivative, 1.0);
        
        let neg_x = Dual::variable(-1.0);
        // ReLU(-1) = 0, derivative = 0
        let relu_neg = neg_x.relu();
        assert_eq!(relu_neg.value, 0.0);
        assert_eq!(relu_neg.derivative, 0.0);
        
        // Test sigmoid
        let zero = Dual::variable(0.0);
        let sigmoid_zero = zero.sigmoid();
        assert!(approx_eq(sigmoid_zero.value, 0.5));
        assert!(approx_eq(sigmoid_zero.derivative, 0.25)); // sigmoid'(0) = 1/4
    }
    
    #[test]
    fn test_error_conditions() {
        let neg = Dual::variable(-1.0);
        
        // sqrt of negative should error
        assert!(neg.sqrt().is_err());
        
        // ln of negative should error
        assert!(neg.ln().is_err());
        
        let zero = Dual::variable(0.0);
        let neg_exp = Dual::variable(-1.0);
        
        // 0^(-1) should error
        assert!(zero.powf(neg_exp).is_err());
    }
    
    #[test]
    fn test_chain_rule() {
        // Test d/dx[sin(x²)] = cos(x²) * 2x
        let x = Dual::variable(2.0);
        let x_squared = x * x;
        let sin_x_squared = x_squared.sin();
        
        // At x = 2: sin(4), derivative = cos(4) * 4
        let expected_value = (4.0_f64).sin();
        let expected_derivative = (4.0_f64).cos() * 4.0;
        
        assert!(approx_eq(sin_x_squared.value, expected_value));
        assert!(approx_eq(sin_x_squared.derivative, expected_derivative));
    }
    
    #[test]
    fn test_display() {
        let constant = Dual::constant(3.0);
        assert_eq!(constant.to_string(), "3");
        
        let variable = Dual::variable(2.0);
        assert_eq!(variable.to_string(), "2 + ε");
        
        let general = Dual::new(1.5, 2.5);
        assert_eq!(general.to_string(), "1.5 + 2.5ε");
    }
    
    #[test]
    fn test_conversions() {
        let dual: Dual = 3.0.into();
        assert_eq!(dual.value, 3.0);
        assert_eq!(dual.derivative, 0.0);
        
        let dual: Dual = 5.into();
        assert_eq!(dual.value, 5.0);
        assert_eq!(dual.derivative, 0.0);
        
        let value: f64 = Dual::new(2.5, 1.0).into();
        assert_eq!(value, 2.5);
    }
}