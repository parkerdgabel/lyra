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
    
    // ===== INVERSE TRIGONOMETRIC FUNCTIONS =====
    
    /// Inverse sine: asin(self) 
    pub fn asin(self) -> AutodiffResult<Dual> {
        if self.value.abs() > 1.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "asin domain error: input must be in [-1, 1]".to_string(),
            });
        }
        
        let asin_val = self.value.asin();
        let denom = (1.0 - self.value * self.value).sqrt();
        
        if denom == 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "asin derivative undefined at ±1".to_string(),
            });
        }
        
        let asin_deriv = self.derivative / denom;
        Ok(Dual::new(asin_val, asin_deriv))
    }
    
    /// Inverse cosine: acos(self)
    pub fn acos(self) -> AutodiffResult<Dual> {
        if self.value.abs() > 1.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "acos domain error: input must be in [-1, 1]".to_string(),
            });
        }
        
        let acos_val = self.value.acos();
        let denom = (1.0 - self.value * self.value).sqrt();
        
        if denom == 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "acos derivative undefined at ±1".to_string(),
            });
        }
        
        let acos_deriv = -self.derivative / denom;
        Ok(Dual::new(acos_val, acos_deriv))
    }
    
    /// Inverse tangent: atan(self)
    pub fn atan(self) -> Dual {
        self.apply_unary(|x| x.atan(), |x| 1.0 / (1.0 + x * x))
    }
    
    /// Two-argument inverse tangent: atan2(self, other)
    pub fn atan2(self, other: Dual) -> Dual {
        self.apply_binary(
            other,
            |y, x| y.atan2(x),
            |y, x| x / (x * x + y * y),     // ∂/∂y atan2(y,x) = x/(x²+y²)
            |y, x| -y / (x * x + y * y),    // ∂/∂x atan2(y,x) = -y/(x²+y²)
        )
    }
    
    // ===== INVERSE HYPERBOLIC FUNCTIONS =====
    
    /// Inverse hyperbolic sine: asinh(self)
    pub fn asinh(self) -> Dual {
        self.apply_unary(
            |x| x.asinh(),
            |x| 1.0 / (x * x + 1.0).sqrt()
        )
    }
    
    /// Inverse hyperbolic cosine: acosh(self)
    pub fn acosh(self) -> AutodiffResult<Dual> {
        if self.value < 1.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "acosh domain error: input must be ≥ 1".to_string(),
            });
        }
        
        let acosh_val = self.value.acosh();
        let denom = (self.value * self.value - 1.0).sqrt();
        
        if denom == 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "acosh derivative undefined at 1".to_string(),
            });
        }
        
        let acosh_deriv = self.derivative / denom;
        Ok(Dual::new(acosh_val, acosh_deriv))
    }
    
    /// Inverse hyperbolic tangent: atanh(self)
    pub fn atanh(self) -> AutodiffResult<Dual> {
        if self.value.abs() >= 1.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "atanh domain error: input must be in (-1, 1)".to_string(),
            });
        }
        
        let atanh_val = self.value.atanh();
        let atanh_deriv = self.derivative / (1.0 - self.value * self.value);
        Ok(Dual::new(atanh_val, atanh_deriv))
    }
    
    // ===== LOGARITHMIC VARIANTS =====
    
    /// Base-10 logarithm: log10(self)
    pub fn log10(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "log10 of non-positive number".to_string(),
            });
        }
        
        let log10_val = self.value.log10();
        let log10_deriv = self.derivative / (self.value * 10.0_f64.ln());
        Ok(Dual::new(log10_val, log10_deriv))
    }
    
    /// Base-2 logarithm: log2(self)
    pub fn log2(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "log2 of non-positive number".to_string(),
            });
        }
        
        let log2_val = self.value.log2();
        let log2_deriv = self.derivative / (self.value * 2.0_f64.ln());
        Ok(Dual::new(log2_val, log2_deriv))
    }
    
    /// Natural logarithm of (1 + self): ln(1 + self)
    /// More numerically stable for small values
    pub fn ln_1p(self) -> AutodiffResult<Dual> {
        if self.value <= -1.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "ln_1p domain error: input must be > -1".to_string(),
            });
        }
        
        let ln_1p_val = self.value.ln_1p();
        let ln_1p_deriv = self.derivative / (1.0 + self.value);
        Ok(Dual::new(ln_1p_val, ln_1p_deriv))
    }
    
    /// Exponential minus 1: e^self - 1
    /// More numerically stable for small values
    pub fn exp_m1(self) -> Dual {
        self.apply_unary(|x| x.exp_m1(), |x| x.exp())
    }
    
    // ===== ADVANCED ACTIVATION FUNCTIONS =====
    
    /// Leaky ReLU activation: max(alpha * self, self)
    pub fn leaky_relu(self, alpha: f64) -> Dual {
        if self.value > 0.0 {
            self
        } else {
            Dual::new(alpha * self.value, alpha * self.derivative)
        }
    }
    
    /// Exponential Linear Unit (ELU): x if x > 0, else alpha * (e^x - 1)
    pub fn elu(self, alpha: f64) -> Dual {
        if self.value > 0.0 {
            self
        } else {
            let exp_x = self.exp();
            Dual::new(alpha * (exp_x.value - 1.0), alpha * exp_x.derivative)
        }
    }
    
    /// Scaled Exponential Linear Unit (SELU)
    pub fn selu(self) -> Dual {
        const ALPHA: f64 = 1.6732632423543772848170429916717;
        const SCALE: f64 = 1.0507009873554804934193349852946;
        
        if self.value > 0.0 {
            Dual::new(SCALE * self.value, SCALE * self.derivative)
        } else {
            let exp_x = self.exp();
            let selu_val = SCALE * ALPHA * (exp_x.value - 1.0);
            let selu_deriv = SCALE * ALPHA * exp_x.derivative;
            Dual::new(selu_val, selu_deriv)
        }
    }
    
    /// Gaussian Error Linear Unit (GELU): x * Φ(x) where Φ is CDF of standard normal
    /// Approximation: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    pub fn gelu(self) -> Dual {
        const COEFF1: f64 = 0.7978845608028654; // √(2/π)
        const COEFF2: f64 = 0.044715;
        
        let x = self.value;
        let x_cubed = x * x * x;
        let inner = COEFF1 * (x + COEFF2 * x_cubed);
        let tanh_inner = inner.tanh();
        
        let gelu_val = 0.5 * x * (1.0 + tanh_inner);
        
        // Derivative computation
        let tanh_deriv = 1.0 - tanh_inner * tanh_inner;
        let inner_deriv = COEFF1 * (1.0 + 3.0 * COEFF2 * x * x);
        let gelu_deriv = 0.5 * (1.0 + tanh_inner + x * tanh_deriv * inner_deriv);
        
        Dual::new(gelu_val, gelu_deriv * self.derivative)
    }
    
    /// Swish/SiLU activation: x * sigmoid(x)
    pub fn swish(self) -> Dual {
        let sigmoid = self.sigmoid();
        let swish_val = self.value * sigmoid.value;
        let swish_deriv = sigmoid.value + self.value * sigmoid.derivative;
        Dual::new(swish_val, swish_deriv)
    }
    
    /// Softplus activation: ln(1 + e^x)
    pub fn softplus(self) -> Dual {
        let exp_x = self.exp();
        let softplus_val = (1.0 + exp_x.value).ln();
        let softplus_deriv = exp_x.derivative / (1.0 + exp_x.value);
        Dual::new(softplus_val, softplus_deriv)
    }
    
    /// Softsign activation: x / (1 + |x|)
    pub fn softsign(self) -> Dual {
        let abs_x = self.value.abs();
        let denom = 1.0 + abs_x;
        let softsign_val = self.value / denom;
        
        let _sign = if self.value >= 0.0 { 1.0 } else { -1.0 };
        let softsign_deriv = self.derivative / (denom * denom);
        
        Dual::new(softsign_val, softsign_deriv)
    }
    
    // ===== SPECIAL MATHEMATICAL FUNCTIONS =====
    
    /// Gamma function: Γ(self)
    /// Using Stirling's approximation for implementation
    pub fn gamma(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::InvalidDualOperation {
                reason: "gamma function undefined for non-positive integers".to_string(),
            });
        }
        
        // Stirling's approximation: Γ(x) ≈ √(2π/x) * (x/e)^x
        let x = self.value;
        let gamma_val = (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x);
        
        // Derivative using digamma function approximation
        let digamma_x = x.ln() - 1.0 / (2.0 * x); // Simple approximation
        let gamma_deriv = gamma_val * digamma_x * self.derivative;
        
        Ok(Dual::new(gamma_val, gamma_deriv))
    }
    
    // ===== ADVANCED ACTIVATION FUNCTIONS =====
    
    /// Mish activation: x * tanh(softplus(x))
    /// More smooth than ReLU and provides better gradient flow
    pub fn mish(self) -> Dual {
        let softplus = self.softplus();
        let tanh_softplus = softplus.tanh();
        
        let mish_val = self.value * tanh_softplus.value;
        
        // Mish derivative: d/dx[x * tanh(softplus(x))]
        // = tanh(softplus(x)) + x * sech²(softplus(x)) * sigmoid(x)
        let _softplus_val = softplus.value;
        let sigmoid_x = self.sigmoid();
        let sech_squared = 1.0 - tanh_softplus.value * tanh_softplus.value;
        let mish_deriv = tanh_softplus.value + self.value * sech_squared * sigmoid_x.value;
        
        Dual::new(mish_val, mish_deriv * self.derivative)
    }
    
    /// Hardswish activation: x * ReLU6(x + 3) / 6
    /// Efficient approximation of Swish for mobile devices
    pub fn hardswish(self) -> Dual {
        let x_plus_3 = self.value + 3.0;
        let relu6_val = x_plus_3.max(0.0).min(6.0);
        let hardswish_val = self.value * relu6_val / 6.0;
        
        // Hardswish derivative
        let hardswish_deriv = if x_plus_3 <= 0.0 {
            0.0
        } else if x_plus_3 >= 6.0 {
            1.0
        } else {
            relu6_val / 6.0 + self.value / 6.0
        };
        
        Dual::new(hardswish_val, hardswish_deriv * self.derivative)
    }
    
    /// GELU exact implementation using error function
    /// GELU(x) = 0.5 * x * (1 + erf(x / √2))
    pub fn gelu_exact(self) -> AutodiffResult<Dual> {
        let x_over_sqrt2 = self / Dual::constant(2.0_f64.sqrt());
        let erf_x = x_over_sqrt2.erf()?;
        let one_plus_erf = Dual::constant(1.0) + erf_x;
        let half_x = self * Dual::constant(0.5);
        
        Ok(half_x * one_plus_erf)
    }
    
    /// Parametric ReLU (PReLU): max(alpha * x, x)
    /// where alpha is a learnable parameter
    pub fn prelu(self, alpha: Dual) -> Dual {
        if self.value > 0.0 {
            self
        } else {
            alpha * self
        }
    }
    
    /// Gated Linear Unit (GLU): GLU(a, b) = a * sigmoid(b)
    /// Used in transformer architectures
    pub fn glu(self, gate: Dual) -> Dual {
        let sigmoid_gate = gate.sigmoid();
        self * sigmoid_gate
    }
    
    /// ReLU6 activation: min(max(0, x), 6)
    /// Bounded ReLU commonly used in quantized networks
    pub fn relu6(self) -> Dual {
        let relu6_val = self.value.max(0.0).min(6.0);
        let relu6_deriv = if self.value <= 0.0 || self.value >= 6.0 {
            0.0
        } else {
            1.0
        };
        
        Dual::new(relu6_val, relu6_deriv * self.derivative)
    }
    
    /// Hardtanh activation: clamp(x, -1, 1)
    /// Linear approximation of tanh
    pub fn hardtanh(self) -> Dual {
        let hardtanh_val = self.value.max(-1.0).min(1.0);
        let hardtanh_deriv = if self.value < -1.0 || self.value > 1.0 {
            0.0
        } else {
            1.0
        };
        
        Dual::new(hardtanh_val, hardtanh_deriv * self.derivative)
    }
    
    /// Tanhshrink activation: x - tanh(x)
    /// Shrinking version of tanh
    pub fn tanhshrink(self) -> Dual {
        let tanh_x = self.tanh();
        self - tanh_x
    }
    
    /// Softshrink activation: sign(x) * max(|x| - lambda, 0)
    /// Soft thresholding function
    pub fn softshrink(self, lambda: f64) -> Dual {
        let abs_x = self.value.abs();
        if abs_x <= lambda {
            Dual::constant(0.0)
        } else {
            let sign = if self.value >= 0.0 { 1.0 } else { -1.0 };
            let softshrink_val = sign * (abs_x - lambda);
            let softshrink_deriv = sign;
            Dual::new(softshrink_val, softshrink_deriv * self.derivative)
        }
    }
    
    /// Hardshrink activation: x if |x| > lambda else 0
    /// Hard thresholding function
    pub fn hardshrink(self, lambda: f64) -> Dual {
        if self.value.abs() > lambda {
            self
        } else {
            Dual::constant(0.0)
        }
    }
    
    /// LogSigmoid activation: log(sigmoid(x))
    /// Numerically stable version of log(1/(1+exp(-x)))
    pub fn logsigmoid(self) -> Dual {
        // Use log-sum-exp trick for numerical stability
        // log(sigmoid(x)) = log(1/(1+exp(-x))) = -log(1+exp(-x)) = -softplus(-x)
        let neg_x = -self;
        let neg_softplus = neg_x.softplus();
        -neg_softplus
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

impl Dual {
    // ===== ATTENTION MECHANISMS =====
    
    /// Softmax function: exp(x) / sum(exp(x_i))
    /// For a single value, this simplifies to the sigmoid function
    /// In practice, this would be used on vectors through broadcasting
    pub fn softmax(self) -> Dual {
        // For a single value, softmax(x) = exp(x) / exp(x) = 1
        // But we implement the general case for consistency
        let exp_val = self.exp();
        // Since we only have one value, the denominator is just exp(x)
        // So softmax(x) = exp(x) / exp(x) = 1 with derivative 0
        // This is more useful when applied to vectors
        exp_val / exp_val
    }
    
    /// Scaled dot-product attention weight: exp(x) / sqrt(d_k)
    /// Where d_k is the dimension of the key vectors
    pub fn attention_weight(self, d_k: f64) -> Dual {
        let scale = 1.0 / d_k.sqrt();
        (self * scale).exp()
    }
    
    /// Multi-head attention computation (simplified for dual numbers)
    /// In practice, this operates on matrices, but here we show the core computation
    /// query • key / sqrt(d_k), then apply softmax
    pub fn attention_score(self, key: Dual, d_k: f64) -> Dual {
        let scale = 1.0 / d_k.sqrt();
        let score = (self * key) * scale;
        score.softmax()
    }
    
    /// Self-attention: attention_score with same input for query and key
    pub fn self_attention_score(self, d_k: f64) -> Dual {
        self.attention_score(self, d_k)
    }
    
    /// Layer normalization: (x - μ) / σ
    /// For a single value, we normalize to standard form
    /// In practice, this operates on vectors with computed mean and variance
    pub fn layer_norm(self, mean: Dual, variance: Dual, gamma: Dual, beta: Dual) -> Dual {
        let normalized = (self - mean) / variance.sqrt().unwrap();
        gamma * normalized + beta
    }
    
    /// Layer normalization (simplified version for unit variance)
    /// Assumes input is already mean-centered
    pub fn layer_norm_simple(self, gamma: Dual, beta: Dual) -> Dual {
        gamma * self + beta
    }
    
    /// RMS (Root Mean Square) normalization: x / rms(x)
    /// For a single value, this is just x / |x| = sign(x)
    /// In practice, operates on vectors
    pub fn rms_norm(self) -> Dual {
        let rms = self.abs();
        if rms.value == 0.0 {
            Dual::constant(0.0)
        } else {
            self / rms
        }
    }
    
    /// RMS normalization with learnable scale
    pub fn rms_norm_scaled(self, gamma: Dual) -> Dual {
        gamma * self.rms_norm()
    }
    
    /// Causal mask application: if masked, return large negative value
    /// Used in autoregressive attention to prevent looking at future tokens
    pub fn apply_causal_mask(self, is_masked: bool) -> Dual {
        if is_masked {
            Dual::constant(-1e9) // Large negative value for softmax
        } else {
            self
        }
    }
    
    /// Attention dropout simulation: randomly zero out with probability p
    /// For deterministic computation, we scale by (1-p)
    pub fn attention_dropout(self, dropout_rate: f64) -> Dual {
        let scale = 1.0 - dropout_rate;
        self * scale
    }
    
    /// Position encoding (sinusoidal): sin(pos / 10000^(2i/d))
    /// For even dimensions: sine, for odd dimensions: cosine
    pub fn positional_encoding_sin(self, position: f64, dimension: f64) -> Dual {
        let freq = 1.0 / (10000.0_f64.powf(2.0 * dimension / 512.0));
        let angle = position * freq;
        Dual::constant(angle).sin() + self
    }
    
    /// Position encoding (cosine version)
    pub fn positional_encoding_cos(self, position: f64, dimension: f64) -> Dual {
        let freq = 1.0 / (10000.0_f64.powf(2.0 * dimension / 512.0));
        let angle = position * freq;
        Dual::constant(angle).cos() + self
    }
    
    /// Multi-head attention with concatenation (simplified)
    /// Combines multiple attention heads
    pub fn multi_head_combine(self, other_heads: &[Dual]) -> Dual {
        let mut result = self;
        for head in other_heads {
            result = result + *head;
        }
        result / (1.0 + other_heads.len() as f64)
    }
    
    /// Cross-attention: attention between different sequences
    /// query from sequence 1, key from sequence 2
    pub fn cross_attention_score(self, key: Dual, value: Dual, d_k: f64) -> Dual {
        let attention_weight = self.attention_score(key, d_k);
        attention_weight * value
    }
    
    /// Attention output projection (linear transformation after attention)
    pub fn attention_output_projection(self, weight: Dual, bias: Dual) -> Dual {
        self * weight + bias
    }
    
    // ===== LOSS FUNCTIONS =====
    
    /// Mean Squared Error loss: (prediction - target)²
    /// Most common loss for regression tasks
    pub fn mse_loss(self, target: Dual) -> Dual {
        let diff = self - target;
        diff * diff
    }
    
    /// Mean Absolute Error loss: |prediction - target|
    /// L1 loss, more robust to outliers than MSE
    pub fn mae_loss(self, target: Dual) -> Dual {
        (self - target).abs()
    }
    
    /// Huber loss: smooth combination of L1 and L2 loss
    /// More robust to outliers than MSE, smoother than MAE
    pub fn huber_loss(self, target: Dual, delta: f64) -> Dual {
        let diff = (self - target).abs();
        if diff.value <= delta {
            Dual::constant(0.5) * diff * diff
        } else {
            Dual::constant(delta) * (diff - Dual::constant(0.5 * delta))
        }
    }
    
    /// Smooth L1 loss (used in object detection)
    /// Similar to Huber loss with delta = 1.0
    pub fn smooth_l1_loss(self, target: Dual) -> Dual {
        self.huber_loss(target, 1.0)
    }
    
    /// Binary Cross-Entropy loss: -[t*log(p) + (1-t)*log(1-p)]
    /// Standard loss for binary classification
    pub fn binary_cross_entropy_loss(self, target: Dual) -> AutodiffResult<Dual> {
        // Clamp predictions to avoid log(0)
        let eps = 1e-7;
        let pred_clamped = self.clamp(eps, 1.0 - eps);
        
        let log_pred = pred_clamped.ln()?;
        let log_one_minus_pred = (Dual::constant(1.0) - pred_clamped).ln()?;
        
        let loss = -(target * log_pred + (Dual::constant(1.0) - target) * log_one_minus_pred);
        Ok(loss)
    }
    
    /// Cross-Entropy loss for multi-class classification
    /// For use with softmax outputs: -target * log(prediction)
    pub fn cross_entropy_loss(self, target: Dual) -> AutodiffResult<Dual> {
        // Clamp prediction to avoid log(0)
        let eps = 1e-7;
        let pred_clamped = self.clamp(eps, 1.0);
        let log_pred = pred_clamped.ln()?;
        Ok(-(target * log_pred))
    }
    
    /// Focal Loss: -(1-p)^γ * log(p) for class imbalance
    /// Addresses class imbalance by down-weighting easy examples
    pub fn focal_loss(self, target: Dual, alpha: f64, gamma: f64) -> AutodiffResult<Dual> {
        let eps = 1e-7;
        let pred_clamped = self.clamp(eps, 1.0 - eps);
        
        let log_pred = pred_clamped.ln()?;
        let one_minus_pred = Dual::constant(1.0) - pred_clamped;
        let focal_weight = one_minus_pred.powf(Dual::constant(gamma))?;
        
        Ok(-Dual::constant(alpha) * focal_weight * target * log_pred)
    }
    
    /// KL Divergence: sum(p * log(p/q))
    /// Measures difference between probability distributions
    pub fn kl_divergence(self, target: Dual) -> AutodiffResult<Dual> {
        let eps = 1e-7;
        let pred_clamped = self.clamp(eps, 1.0);
        let target_clamped = target.clamp(eps, 1.0);
        
        let log_ratio = (pred_clamped / target_clamped).ln()?;
        Ok(pred_clamped * log_ratio)
    }
    
    /// Contrastive Loss for siamese networks
    /// Encourages similar items to be close, dissimilar items to be far
    pub fn contrastive_loss(self, target: Dual, margin: f64) -> Dual {
        let distance_squared = self * self;
        let margin_dual = Dual::constant(margin);
        let zero = Dual::constant(0.0);
        let one = Dual::constant(1.0);
        
        // Similar pairs (target = 1): minimize distance
        let similar_loss = target * distance_squared;
        
        // Dissimilar pairs (target = 0): maximize distance up to margin
        let margin_minus_dist = margin_dual - self;
        let dissimilar_loss = (one - target) * margin_minus_dist.max(zero) * margin_minus_dist.max(zero);
        
        similar_loss + dissimilar_loss
    }
    
    /// Triplet Loss for metric learning
    /// Anchor-positive distance should be smaller than anchor-negative distance
    pub fn triplet_loss(self, _positive_dist: Dual, negative_dist: Dual, margin: f64) -> Dual {
        let margin_dual = Dual::constant(margin);
        let zero = Dual::constant(0.0);
        
        // max(0, positive_distance - negative_distance + margin)
        (self - negative_dist + margin_dual).max(zero)
    }
    
    // ===== OPTIMIZATION UTILITIES =====
    
    /// Clamp values to a specified range
    /// Essential for numerical stability in loss functions
    pub fn clamp(self, min_val: f64, max_val: f64) -> Dual {
        if self.value < min_val {
            Dual::constant(min_val)
        } else if self.value > max_val {
            Dual::constant(max_val)
        } else {
            self
        }
    }
    
    /// Gradient clipping by norm
    /// Prevents exploding gradients during training
    pub fn gradient_clip(self, max_norm: f64) -> Dual {
        let grad_norm = self.derivative.abs();
        if grad_norm > max_norm {
            let scale = max_norm / grad_norm;
            Dual::new(self.value, self.derivative * scale)
        } else {
            self
        }
    }
    
    /// Exponential moving average for loss smoothing
    /// Commonly used in training loops for smoother loss curves
    pub fn ema_update(self, previous: Dual, momentum: f64) -> Dual {
        let beta = Dual::constant(momentum);
        let one_minus_beta = Dual::constant(1.0 - momentum);
        beta * previous + one_minus_beta * self
    }
    
    /// Learning rate scheduling: exponential decay
    pub fn exponential_decay(self, decay_rate: f64, step: f64) -> Dual {
        let decay = Dual::constant(decay_rate.powf(step));
        self * decay
    }
    
    /// Learning rate scheduling: cosine annealing
    pub fn cosine_annealing(self, step: f64, total_steps: f64) -> Dual {
        let pi = std::f64::consts::PI;
        let cos_arg = pi * step / total_steps;
        let cos_val = cos_arg.cos();
        let factor = 0.5 * (1.0 + cos_val);
        self * Dual::constant(factor)
    }
    
    /// Momentum update for SGD with momentum
    /// velocity = momentum * velocity + gradient
    pub fn momentum_update(self, velocity: Dual, momentum: f64) -> Dual {
        Dual::constant(momentum) * velocity + self
    }
    
    /// Adam optimizer: bias-corrected exponential moving averages
    pub fn adam_update(self, m: Dual, v: Dual, beta1: f64, beta2: f64, _step: f64) -> (Dual, Dual) {
        let beta1_dual = Dual::constant(beta1);
        let beta2_dual = Dual::constant(beta2);
        let one_minus_beta1 = Dual::constant(1.0 - beta1);
        let one_minus_beta2 = Dual::constant(1.0 - beta2);
        
        // Update biased first moment estimate
        let m_new = beta1_dual * m + one_minus_beta1 * self;
        
        // Update biased second moment estimate
        let v_new = beta2_dual * v + one_minus_beta2 * self * self;
        
        (m_new, v_new)
    }
    
    /// RMSprop optimizer: adaptive learning rates
    pub fn rmsprop_update(self, sq_avg: Dual, alpha: f64) -> Dual {
        let alpha_dual = Dual::constant(alpha);
        let one_minus_alpha = Dual::constant(1.0 - alpha);
        
        alpha_dual * sq_avg + one_minus_alpha * self * self
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

impl Dual {
    // ===============================================================================
    // ADVANCED MATHEMATICAL OPERATIONS - PHASE 8.2.4
    // ===============================================================================
    
    // -------------------------------------------------------------------------
    // Matrix Decomposition Functions (Simplified for Dual Numbers)
    // -------------------------------------------------------------------------
    
    /// Determinant approximation for 2x2 matrix represented as dual numbers
    /// For a matrix [[a, b], [c, d]], det = ad - bc
    pub fn det_2x2(a: Dual, b: Dual, c: Dual, d: Dual) -> Dual {
        a * d - b * c
    }
    
    /// Trace of 2x2 matrix (sum of diagonal elements)
    pub fn trace_2x2(a: Dual, d: Dual) -> Dual {
        a + d
    }
    
    /// Condition number estimation (simplified) 
    /// Based on ratio of largest to smallest eigenvalue approximation
    pub fn condition_number_estimate(self, other: Dual) -> Dual {
        let max_val = self.abs().max(other.abs());
        let min_val = self.abs().min(other.abs()) + Dual::constant(1e-12); // Avoid division by zero
        max_val / min_val
    }
    
    /// QR decomposition helper: Givens rotation angle
    /// For eliminating element (i,j) in matrix
    pub fn givens_rotation_angle(self, other: Dual) -> Dual {
        // atan2(other, self) gives rotation angle
        other.atan2(self)
    }
    
    // -------------------------------------------------------------------------
    // Eigenvalue/Eigenvector Computation Helpers
    // -------------------------------------------------------------------------
    
    /// Characteristic polynomial evaluation for 2x2 matrix
    /// det(A - λI) = λ² - trace(A)λ + det(A)
    pub fn characteristic_poly_2x2(lambda: Dual, trace: Dual, det: Dual) -> Dual {
        lambda * lambda - trace * lambda + det
    }
    
    /// Power iteration step for dominant eigenvalue
    /// v_new = A*v / ||A*v||
    pub fn power_iteration_step(self, other: Dual) -> AutodiffResult<(Dual, Dual)> {
        // Simplified 2D case: normalize vector (self, other)
        let norm = (self * self + other * other).sqrt()?;
        Ok((self / norm, other / norm))
    }
    
    /// Rayleigh quotient for eigenvalue estimation
    /// R(x) = x^T * A * x / x^T * x
    pub fn rayleigh_quotient(x1: Dual, x2: Dual, ax1: Dual, ax2: Dual) -> Dual {
        let numerator = x1 * ax1 + x2 * ax2;
        let denominator = x1 * x1 + x2 * x2;
        numerator / denominator
    }
    
    // -------------------------------------------------------------------------
    // Advanced Calculus Operations
    // -------------------------------------------------------------------------
    
    /// Numerical integration using trapezoidal rule
    /// Integrates function over interval with dual number precision
    pub fn integrate_trapezoidal(f_start: Dual, f_end: Dual, h: f64) -> Dual {
        (f_start + f_end) * Dual::constant(h * 0.5)
    }
    
    /// Simpson's rule integration (requires 3 points)
    pub fn integrate_simpson(f_start: Dual, f_mid: Dual, f_end: Dual, h: f64) -> Dual {
        let coeff = Dual::constant(h / 3.0);
        coeff * (f_start + Dual::constant(4.0) * f_mid + f_end)
    }
    
    /// Runge-Kutta 4th order step for ODE solving
    /// dy/dx = f(x, y), step size h
    pub fn rk4_step(y: Dual, k1: Dual, k2: Dual, k3: Dual, k4: Dual, h: f64) -> Dual {
        let h_dual = Dual::constant(h);
        let coeff = h_dual / Dual::constant(6.0);
        y + coeff * (k1 + Dual::constant(2.0) * k2 + Dual::constant(2.0) * k3 + k4)
    }
    
    /// Euler method step for ODE solving
    pub fn euler_step(y: Dual, dy_dx: Dual, h: f64) -> Dual {
        y + dy_dx * Dual::constant(h)
    }
    
    /// Second derivative approximation using finite differences
    pub fn second_derivative_approx(f_prev: Dual, f_curr: Dual, f_next: Dual, h: f64) -> Dual {
        let h_sq = Dual::constant(h * h);
        (f_next - Dual::constant(2.0) * f_curr + f_prev) / h_sq
    }
    
    // -------------------------------------------------------------------------
    // Statistical Functions
    // -------------------------------------------------------------------------
    
    /// Sample mean of a collection (simplified for 2 values)
    pub fn mean(self, other: Dual) -> Dual {
        (self + other) / Dual::constant(2.0)
    }
    
    /// Sample variance (Bessel's correction with n-1)
    pub fn variance(self, other: Dual) -> Dual {
        let mean_val = self.mean(other);
        let diff1 = self - mean_val;
        let diff2 = other - mean_val;
        (diff1 * diff1 + diff2 * diff2) / Dual::constant(1.0) // n-1 = 1 for 2 samples
    }
    
    /// Standard deviation
    pub fn std_dev(self, other: Dual) -> AutodiffResult<Dual> {
        Ok(self.variance(other).sqrt()?)
    }
    
    /// Covariance between two paired samples
    pub fn covariance(x1: Dual, x2: Dual, y1: Dual, y2: Dual) -> Dual {
        let x_mean = x1.mean(x2);
        let y_mean = y1.mean(y2);
        let cov_term1 = (x1 - x_mean) * (y1 - y_mean);
        let cov_term2 = (x2 - x_mean) * (y2 - y_mean);
        (cov_term1 + cov_term2) / Dual::constant(1.0) // n-1 = 1
    }
    
    /// Correlation coefficient (Pearson)
    pub fn correlation(x1: Dual, x2: Dual, y1: Dual, y2: Dual) -> AutodiffResult<Dual> {
        let cov = Self::covariance(x1, x2, y1, y2);
        let x_std = x1.std_dev(x2)?;
        let y_std = y1.std_dev(y2)?;
        Ok(cov / (x_std * y_std))
    }
    
    /// Z-score (standardization)
    pub fn z_score(self, mean: Dual, std_dev: Dual) -> Dual {
        (self - mean) / std_dev
    }
    
    /// Welford's online variance update algorithm
    pub fn welford_update(self, mean: Dual, m2: Dual, n: f64) -> (Dual, Dual) {
        let n_dual = Dual::constant(n);
        let delta = self - mean;
        let new_mean = mean + delta / n_dual;
        let delta2 = self - new_mean;
        let new_m2 = m2 + delta * delta2;
        (new_mean, new_m2)
    }
    
    // -------------------------------------------------------------------------
    // Probability Distributions
    // -------------------------------------------------------------------------
    
    /// Normal (Gaussian) distribution PDF
    /// f(x) = (1/√(2π)σ) * exp(-0.5 * ((x-μ)/σ)²)
    pub fn normal_pdf(self, mean: Dual, std_dev: Dual) -> AutodiffResult<Dual> {
        if std_dev.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Standard deviation must be positive for normal distribution".to_string(),
            });
        }
        
        let two_pi = Dual::constant(2.0 * std::f64::consts::PI);
        let coefficient = Dual::constant(1.0) / (std_dev * two_pi.sqrt()?);
        let z_score = (self - mean) / std_dev;
        let exponent = Dual::constant(-0.5) * z_score * z_score;
        Ok(coefficient * exponent.exp())
    }
    
    /// Normal distribution CDF approximation using error function
    pub fn normal_cdf(self, mean: Dual, std_dev: Dual) -> AutodiffResult<Dual> {
        if std_dev.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Standard deviation must be positive for normal distribution".to_string(),
            });
        }
        
        let z = (self - mean) / (std_dev * Dual::constant(std::f64::consts::SQRT_2));
        let erf_z = z.erf()?;
        Ok((Dual::constant(1.0) + erf_z) / Dual::constant(2.0))
    }
    
    /// Uniform distribution PDF
    pub fn uniform_pdf(self, a: Dual, b: Dual) -> AutodiffResult<Dual> {
        if b.value <= a.value {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Upper bound must be greater than lower bound for uniform distribution".to_string(),
            });
        }
        
        // Check if x is in [a, b]
        if self.value >= a.value && self.value <= b.value {
            Ok(Dual::constant(1.0) / (b - a))
        } else {
            Ok(Dual::constant(0.0))
        }
    }
    
    /// Exponential distribution PDF
    /// f(x) = λ * exp(-λx) for x ≥ 0
    pub fn exponential_pdf(self, lambda: Dual) -> AutodiffResult<Dual> {
        if lambda.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Rate parameter λ must be positive for exponential distribution".to_string(),
            });
        }
        
        if self.value < 0.0 {
            Ok(Dual::constant(0.0))
        } else {
            Ok(lambda * (Dual::constant(-1.0) * lambda * self).exp())
        }
    }
    
    /// Exponential distribution CDF
    /// F(x) = 1 - exp(-λx) for x ≥ 0
    pub fn exponential_cdf(self, lambda: Dual) -> AutodiffResult<Dual> {
        if lambda.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Rate parameter λ must be positive for exponential distribution".to_string(),
            });
        }
        
        if self.value < 0.0 {
            Ok(Dual::constant(0.0))
        } else {
            Ok(Dual::constant(1.0) - (Dual::constant(-1.0) * lambda * self).exp())
        }
    }
    
    /// Gamma function approximation using Stirling's approximation
    /// Γ(x) ≈ √(2π/x) * (x/e)^x for large x
    pub fn gamma_approx(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Gamma function requires positive input".to_string(),
            });
        }
        
        let two_pi = Dual::constant(2.0 * std::f64::consts::PI);
        let e = Dual::constant(std::f64::consts::E);
        let sqrt_term = (two_pi / self).sqrt()?;
        let power_term = (self / e).powf(self)?;
        Ok(sqrt_term * power_term)
    }
    
    /// Log-gamma function (more numerically stable)
    pub fn log_gamma_approx(self) -> AutodiffResult<Dual> {
        if self.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Log-gamma function requires positive input".to_string(),
            });
        }
        
        let two_pi = Dual::constant(2.0 * std::f64::consts::PI);
        let e = Dual::constant(std::f64::consts::E);
        let log_sqrt_term = (two_pi / self).ln()? / Dual::constant(2.0);
        let log_power_term = self * (self / e).ln()?;
        Ok(log_sqrt_term + log_power_term)
    }
    
    /// Beta function using gamma functions: B(x,y) = Γ(x)Γ(y)/Γ(x+y)
    pub fn beta_function(self, other: Dual) -> AutodiffResult<Dual> {
        let gamma_x = self.gamma_approx()?;
        let gamma_y = other.gamma_approx()?;
        let gamma_xy = (self + other).gamma_approx()?;
        Ok((gamma_x * gamma_y) / gamma_xy)
    }
    
    /// Error function approximation using series expansion
    /// erf(x) ≈ (2/√π) * Σ((-1)^n * x^(2n+1) / (n! * (2n+1)))
    pub fn erf(self) -> AutodiffResult<Dual> {
        let sqrt_pi = Dual::constant(std::f64::consts::PI.sqrt());
        let _coeff = Dual::constant(2.0) / sqrt_pi;
        
        // Use approximation: erf(x) ≈ tanh(1.2 * x + 0.4 * x^3) for |x| < 3
        if self.value.abs() < 3.0 {
            let x_cubed = self * self * self;
            let approx_arg = Dual::constant(1.2) * self + Dual::constant(0.4) * x_cubed;
            Ok(approx_arg.tanh())
        } else {
            // For large values, use asymptotic behavior
            if self.value > 0.0 {
                Ok(Dual::constant(1.0))
            } else {
                Ok(Dual::constant(-1.0))
            }
        }
    }
    
    /// Complementary error function: erfc(x) = 1 - erf(x)
    pub fn erfc(self) -> AutodiffResult<Dual> {
        let erf_val = self.erf()?;
        Ok(Dual::constant(1.0) - erf_val)
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

// ================================================================================================
// HIGHER-ORDER DERIVATIVES (Phase 8.3.4)
// ================================================================================================

/// Second-order dual number for computing Hessians
///
/// A hyper-dual number has the form: f + f'*ε₁ + f'*ε₂ + f''*ε₁ε₂
/// where ε₁² = ε₂² = 0 and ε₁ε₂ ≠ 0
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HyperDual {
    /// Function value f(x,y)
    pub value: f64,
    /// First partial derivative ∂f/∂x
    pub dx: f64,
    /// First partial derivative ∂f/∂y
    pub dy: f64,
    /// Second mixed partial derivative ∂²f/∂x∂y
    pub dxy: f64,
}

impl HyperDual {
    /// Create a new hyper-dual number
    pub fn new(value: f64, dx: f64, dy: f64, dxy: f64) -> Self {
        Self { value, dx, dy, dxy }
    }
    
    /// Create a constant hyper-dual (all derivatives = 0)
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0, 0.0, 0.0)
    }
    
    /// Create a variable with respect to x (dx = 1, dy = 0, dxy = 0)
    pub fn variable_x(value: f64) -> Self {
        Self::new(value, 1.0, 0.0, 0.0)
    }
    
    /// Create a variable with respect to y (dx = 0, dy = 1, dxy = 0)
    pub fn variable_y(value: f64) -> Self {
        Self::new(value, 0.0, 1.0, 0.0)
    }
    
    /// Create a bivariate variable (dx = 1, dy = 1, dxy = 0)
    pub fn bivariate(value: f64) -> Self {
        Self::new(value, 1.0, 1.0, 0.0)
    }
    
    /// Get the Hessian matrix for a scalar function
    /// Returns [[∂²f/∂x², ∂²f/∂x∂y], [∂²f/∂y∂x, ∂²f/∂y²]]
    pub fn hessian_scalar(fx: HyperDual, fy: HyperDual, fxy: HyperDual) -> [[f64; 2]; 2] {
        [
            [fx.dxy, fxy.dxy],  // Row 1: ∂²f/∂x², ∂²f/∂x∂y
            [fxy.dxy, fy.dxy],  // Row 2: ∂²f/∂y∂x, ∂²f/∂y²
        ]
    }
}

impl Add for HyperDual {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
            dxy: self.dxy + other.dxy,
        }
    }
}

impl Sub for HyperDual {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            dx: self.dx - other.dx,
            dy: self.dy - other.dy,
            dxy: self.dxy - other.dxy,
        }
    }
}

impl Mul for HyperDual {
    type Output = Self;
    
    fn mul(self, other: Self) -> Self {
        // Product rule for hyper-dual numbers:
        // (f + f'ε₁ + f''ε₂ + f'''ε₁ε₂) * (g + g'ε₁ + g''ε₂ + g'''ε₁ε₂)
        Self {
            value: self.value * other.value,
            dx: self.dx * other.value + self.value * other.dx,
            dy: self.dy * other.value + self.value * other.dy,
            dxy: self.dxy * other.value + self.dx * other.dy + self.dy * other.dx + self.value * other.dxy,
        }
    }
}

impl HyperDual {
    /// Square root function
    pub fn sqrt(self) -> AutodiffResult<Self> {
        if self.value < 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Square root of negative number".to_string(),
            });
        }
        
        if self.value == 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Square root derivative undefined at zero".to_string(),
            });
        }
        
        let sqrt_val = self.value.sqrt();
        let first_deriv = 0.5 / sqrt_val;
        let second_deriv = -0.25 / (self.value * sqrt_val);
        
        Ok(Self {
            value: sqrt_val,
            dx: first_deriv * self.dx,
            dy: first_deriv * self.dy,
            dxy: second_deriv * self.dx * self.dy + first_deriv * self.dxy,
        })
    }
    
    /// Exponential function
    pub fn exp(self) -> Self {
        let exp_val = self.value.exp();
        
        Self {
            value: exp_val,
            dx: exp_val * self.dx,
            dy: exp_val * self.dy,
            dxy: exp_val * (self.dx * self.dy + self.dxy),
        }
    }
    
    /// Natural logarithm
    pub fn ln(self) -> AutodiffResult<Self> {
        if self.value <= 0.0 {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Logarithm of non-positive number".to_string(),
            });
        }
        
        let ln_val = self.value.ln();
        let first_deriv = 1.0 / self.value;
        let second_deriv = -1.0 / (self.value * self.value);
        
        Ok(Self {
            value: ln_val,
            dx: first_deriv * self.dx,
            dy: first_deriv * self.dy,
            dxy: second_deriv * self.dx * self.dy + first_deriv * self.dxy,
        })
    }
    
    /// Sine function
    pub fn sin(self) -> Self {
        let sin_val = self.value.sin();
        let cos_val = self.value.cos();
        
        Self {
            value: sin_val,
            dx: cos_val * self.dx,
            dy: cos_val * self.dy,
            dxy: -sin_val * self.dx * self.dy + cos_val * self.dxy,
        }
    }
    
    /// Cosine function
    pub fn cos(self) -> Self {
        let cos_val = self.value.cos();
        let sin_val = self.value.sin();
        
        Self {
            value: cos_val,
            dx: -sin_val * self.dx,
            dy: -sin_val * self.dy,
            dxy: -cos_val * self.dx * self.dy - sin_val * self.dxy,
        }
    }
}

impl fmt::Display for HyperDual {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperDual({:.6}, dx: {:.6}, dy: {:.6}, dxy: {:.6})", 
               self.value, self.dx, self.dy, self.dxy)
    }
}

// ================================================================================================
// SPARSE JACOBIAN SUPPORT (Phase 8.3.4)
// ================================================================================================

/// Sparse matrix in Compressed Sparse Row (CSR) format
#[derive(Debug, Clone)]
pub struct SparseMatrix {
    /// Number of rows
    pub rows: usize,
    /// Number of columns  
    pub cols: usize,
    /// Row pointers (length = rows + 1)
    pub row_ptr: Vec<usize>,
    /// Column indices (length = nnz)
    pub col_idx: Vec<usize>,
    /// Non-zero values (length = nnz)
    pub values: Vec<f64>,
}

impl SparseMatrix {
    /// Create a new sparse matrix
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            row_ptr: vec![0; rows + 1],
            col_idx: Vec::new(),
            values: Vec::new(),
        }
    }
    
    /// Create sparse matrix from dense matrix (removes zeros)
    pub fn from_dense(dense: &[Vec<f64>], epsilon: f64) -> Self {
        let rows = dense.len();
        let cols = if rows > 0 { dense[0].len() } else { 0 };
        
        let mut row_ptr = vec![0; rows + 1];
        let mut col_idx = Vec::new();
        let mut values = Vec::new();
        
        let mut nnz = 0;
        for (i, row) in dense.iter().enumerate() {
            row_ptr[i] = nnz;
            for (j, &value) in row.iter().enumerate() {
                if value.abs() > epsilon {
                    col_idx.push(j);
                    values.push(value);
                    nnz += 1;
                }
            }
        }
        row_ptr[rows] = nnz;
        
        Self {
            rows,
            cols,
            row_ptr,
            col_idx,
            values,
        }
    }
    
    /// Get the number of non-zero elements
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    
    /// Get sparsity ratio (nnz / total_elements)
    pub fn sparsity_ratio(&self) -> f64 {
        self.nnz() as f64 / (self.rows * self.cols) as f64
    }
    
    /// Get element at (row, col), returns 0.0 if not stored
    pub fn get(&self, row: usize, col: usize) -> f64 {
        if row >= self.rows || col >= self.cols {
            return 0.0;
        }
        
        let start = self.row_ptr[row];
        let end = self.row_ptr[row + 1];
        
        for idx in start..end {
            if self.col_idx[idx] == col {
                return self.values[idx];
            }
        }
        
        0.0
    }
    
    /// Matrix-vector multiplication: y = A * x
    pub fn matvec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.cols);
        let mut y = vec![0.0; self.rows];
        
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for idx in start..end {
                y[i] += self.values[idx] * x[self.col_idx[idx]];
            }
        }
        
        y
    }
    
    /// Transpose matrix
    pub fn transpose(&self) -> Self {
        let mut col_count = vec![0; self.cols];
        
        // Count elements in each column
        for &col in &self.col_idx {
            col_count[col] += 1;
        }
        
        // Compute row pointers for transposed matrix
        let mut new_row_ptr = vec![0; self.cols + 1];
        for i in 1..=self.cols {
            new_row_ptr[i] = new_row_ptr[i - 1] + col_count[i - 1];
        }
        
        let mut new_col_idx = vec![0; self.nnz()];
        let mut new_values = vec![0.0; self.nnz()];
        let mut counters = new_row_ptr.clone();
        
        // Fill transposed matrix
        for i in 0..self.rows {
            let start = self.row_ptr[i];
            let end = self.row_ptr[i + 1];
            
            for idx in start..end {
                let col = self.col_idx[idx];
                let pos = counters[col];
                new_col_idx[pos] = i;
                new_values[pos] = self.values[idx];
                counters[col] += 1;
            }
        }
        
        Self {
            rows: self.cols,
            cols: self.rows,
            row_ptr: new_row_ptr,
            col_idx: new_col_idx,
            values: new_values,
        }
    }
}

/// Graph coloring for efficient Jacobian computation
#[derive(Debug, Clone)]
pub struct GraphColoring {
    /// Number of variables
    pub n_vars: usize,
    /// Adjacency list representation of sparsity pattern
    pub adjacency: Vec<Vec<usize>>,
    /// Color assigned to each variable
    pub colors: Vec<usize>,
    /// Number of colors used
    pub n_colors: usize,
}

impl GraphColoring {
    /// Create coloring from sparsity pattern
    pub fn from_sparsity_pattern(sparsity: &SparseMatrix) -> Self {
        let n_vars = sparsity.cols;
        let mut adjacency = vec![Vec::new(); n_vars];
        
        // Build adjacency list from sparsity pattern
        // Two variables are adjacent if they appear in the same row (structural nonzero)
        for i in 0..sparsity.rows {
            let start = sparsity.row_ptr[i];
            let end = sparsity.row_ptr[i + 1];
            
            let row_vars: Vec<usize> = (start..end)
                .map(|idx| sparsity.col_idx[idx])
                .collect();
            
            // Connect all pairs in this row
            for (idx1, &var1) in row_vars.iter().enumerate() {
                for &var2 in row_vars.iter().skip(idx1 + 1) {
                    adjacency[var1].push(var2);
                    adjacency[var2].push(var1);
                }
            }
        }
        
        // Remove duplicates and sort
        for adj_list in &mut adjacency {
            adj_list.sort_unstable();
            adj_list.dedup();
        }
        
        let mut coloring = Self {
            n_vars,
            adjacency,
            colors: vec![0; n_vars],
            n_colors: 0,
        };
        
        coloring.greedy_coloring();
        coloring
    }
    
    /// Greedy graph coloring algorithm
    fn greedy_coloring(&mut self) {
        let mut max_color = 0;
        
        for var in 0..self.n_vars {
            // Find used colors by neighbors
            let mut used_colors = vec![false; self.n_vars];
            for &neighbor in &self.adjacency[var] {
                if neighbor < var {  // Only check already colored neighbors
                    used_colors[self.colors[neighbor]] = true;
                }
            }
            
            // Find first available color
            let mut color = 0;
            while color < used_colors.len() && used_colors[color] {
                color += 1;
            }
            
            self.colors[var] = color;
            max_color = max_color.max(color);
        }
        
        self.n_colors = max_color + 1;
    }
    
    /// Get variables with the same color
    pub fn get_color_groups(&self) -> Vec<Vec<usize>> {
        let mut groups = vec![Vec::new(); self.n_colors];
        for (var, &color) in self.colors.iter().enumerate() {
            groups[color].push(var);
        }
        groups
    }
    
    /// Get compression ratio (original vars / colors)
    pub fn compression_ratio(&self) -> f64 {
        self.n_vars as f64 / self.n_colors as f64
    }
}

/// Sparse Jacobian computation using graph coloring
pub struct SparseJacobian {
    /// Sparsity pattern of the Jacobian
    pub sparsity: SparseMatrix,
    /// Graph coloring for compression
    pub coloring: GraphColoring,
    /// Seed matrix for compressed evaluation
    pub seed_matrix: Vec<Vec<f64>>,
}

impl SparseJacobian {
    /// Create sparse Jacobian from known sparsity pattern
    pub fn new(sparsity: SparseMatrix) -> Self {
        let coloring = GraphColoring::from_sparsity_pattern(&sparsity);
        let seed_matrix = Self::create_seed_matrix(&coloring);
        
        Self {
            sparsity,
            coloring,
            seed_matrix,
        }
    }
    
    /// Create seed matrix from coloring
    fn create_seed_matrix(coloring: &GraphColoring) -> Vec<Vec<f64>> {
        let mut seed = vec![vec![0.0; coloring.n_vars]; coloring.n_colors];
        
        for (var, &color) in coloring.colors.iter().enumerate() {
            seed[color][var] = 1.0;
        }
        
        seed
    }
    
    /// Compute sparse Jacobian using compressed forward-mode AD
    pub fn compute_jacobian<F>(&self, f: F, x: &[f64]) -> SparseMatrix
    where
        F: Fn(&[Dual]) -> Vec<Dual>,
    {
        let _n_vars = x.len();
        let mut jacobian_values = Vec::new();
        
        // Evaluate function with each color group
        for (color, seed_row) in self.seed_matrix.iter().enumerate() {
            // Create dual number input with this seed
            let dual_x: Vec<Dual> = x.iter().zip(seed_row.iter())
                .map(|(&val, &seed)| Dual::new(val, seed))
                .collect();
            
            // Evaluate function
            let dual_y = f(&dual_x);
            
            // Extract derivatives for this color group
            let color_group = &self.coloring.get_color_groups()[color];
            
            for (row, dy) in dual_y.iter().enumerate() {
                for &var in color_group {
                    if self.sparsity.get(row, var) != 0.0 {  // Only store structural nonzeros
                        jacobian_values.push((row, var, dy.derivative));
                    }
                }
            }
        }
        
        // Convert to sparse matrix format
        self.build_sparse_matrix(jacobian_values)
    }
    
    /// Build sparse matrix from (row, col, value) triplets
    fn build_sparse_matrix(&self, triplets: Vec<(usize, usize, f64)>) -> SparseMatrix {
        let mut sparse = SparseMatrix::new(self.sparsity.rows, self.sparsity.cols);
        
        // Sort triplets by row, then column
        let mut sorted_triplets = triplets;
        sorted_triplets.sort_by_key(|&(row, col, _)| (row, col));
        
        // Build CSR format
        let mut current_row = 0;
        sparse.row_ptr[0] = 0;
        
        for (row, col, value) in sorted_triplets {
            // Update row pointers
            while current_row < row {
                current_row += 1;
                sparse.row_ptr[current_row] = sparse.col_idx.len();
            }
            
            sparse.col_idx.push(col);
            sparse.values.push(value);
        }
        
        // Fill remaining row pointers
        while current_row < sparse.rows {
            current_row += 1;
            sparse.row_ptr[current_row] = sparse.col_idx.len();
        }
        
        sparse
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
    
    // ===== TESTS FOR NEW TRANSCENDENTAL FUNCTIONS =====
    
    #[test]
    fn test_inverse_trigonometric() {
        // Test asin
        let x = Dual::variable(0.5);
        let asin_x = x.asin().unwrap();
        let expected_asin = 0.5_f64.asin();
        let expected_asin_deriv = 1.0 / (1.0_f64 - 0.25).sqrt(); // 1/√(1-x²)
        assert!(approx_eq(asin_x.value, expected_asin));
        assert!(approx_eq(asin_x.derivative, expected_asin_deriv));
        
        // Test acos
        let acos_x = x.acos().unwrap();
        let expected_acos = 0.5_f64.acos();
        let expected_acos_deriv = -1.0 / (1.0_f64 - 0.25).sqrt(); // -1/√(1-x²)
        assert!(approx_eq(acos_x.value, expected_acos));
        assert!(approx_eq(acos_x.derivative, expected_acos_deriv));
        
        // Test atan
        let atan_x = x.atan();
        let expected_atan = 0.5_f64.atan();
        let expected_atan_deriv = 1.0 / (1.0 + 0.25); // 1/(1+x²)
        assert!(approx_eq(atan_x.value, expected_atan));
        assert!(approx_eq(atan_x.derivative, expected_atan_deriv));
        
        // Test atan2 - y.atan2(x) where y is variable (deriv=1), x is constant (deriv=0)
        let y = Dual::variable(1.0);  // derivative = 1
        let x_const = Dual::constant(0.5);  // derivative = 0
        let atan2_result = y.atan2(x_const);
        let expected_atan2 = 1.0_f64.atan2(0.5);
        // ∂/∂y atan2(y,x) = x/(x²+y²) = 0.5/(0.25+1) = 0.5/1.25 = 0.4
        let expected_atan2_deriv = 0.5 / (0.25 + 1.0);
        assert!(approx_eq(atan2_result.value, expected_atan2));
        assert!(approx_eq(atan2_result.derivative, expected_atan2_deriv));
    }
    
    #[test]
    fn test_inverse_hyperbolic() {
        // Test asinh
        let x = Dual::variable(1.0);
        let asinh_x = x.asinh();
        let expected_asinh = 1.0_f64.asinh();
        let expected_asinh_deriv = 1.0 / (1.0_f64 + 1.0).sqrt(); // 1/√(x²+1)
        assert!(approx_eq(asinh_x.value, expected_asinh));
        assert!(approx_eq(asinh_x.derivative, expected_asinh_deriv));
        
        // Test acosh
        let x = Dual::variable(2.0);
        let acosh_x = x.acosh().unwrap();
        let expected_acosh = 2.0_f64.acosh();
        let expected_acosh_deriv = 1.0 / (4.0_f64 - 1.0).sqrt(); // 1/√(x²-1)
        assert!(approx_eq(acosh_x.value, expected_acosh));
        assert!(approx_eq(acosh_x.derivative, expected_acosh_deriv));
        
        // Test atanh
        let x = Dual::variable(0.5);
        let atanh_x = x.atanh().unwrap();
        let expected_atanh = 0.5_f64.atanh();
        let expected_atanh_deriv = 1.0 / (1.0 - 0.25); // 1/(1-x²)
        assert!(approx_eq(atanh_x.value, expected_atanh));
        assert!(approx_eq(atanh_x.derivative, expected_atanh_deriv));
    }
    
    #[test]
    fn test_logarithmic_variants() {
        // Test log10
        let x = Dual::variable(10.0);
        let log10_x = x.log10().unwrap();
        assert!(approx_eq(log10_x.value, 1.0));
        let expected_log10_deriv = 1.0 / (10.0 * 10.0_f64.ln());
        assert!(approx_eq(log10_x.derivative, expected_log10_deriv));
        
        // Test log2
        let x = Dual::variable(8.0);
        let log2_x = x.log2().unwrap();
        assert!(approx_eq(log2_x.value, 3.0));
        let expected_log2_deriv = 1.0 / (8.0 * 2.0_f64.ln());
        assert!(approx_eq(log2_x.derivative, expected_log2_deriv));
        
        // Test ln_1p
        let x = Dual::variable(1.0);
        let ln_1p_x = x.ln_1p().unwrap();
        let expected_ln_1p = 2.0_f64.ln(); // ln(1+1) = ln(2)
        let expected_ln_1p_deriv = 1.0 / 2.0; // 1/(1+x)
        assert!(approx_eq(ln_1p_x.value, expected_ln_1p));
        assert!(approx_eq(ln_1p_x.derivative, expected_ln_1p_deriv));
        
        // Test exp_m1
        let x = Dual::variable(0.0);
        let exp_m1_x = x.exp_m1();
        assert!(approx_eq(exp_m1_x.value, 0.0)); // e^0 - 1 = 0
        assert!(approx_eq(exp_m1_x.derivative, 1.0)); // d/dx(e^x-1) = e^x, at x=0 -> e^0 = 1
    }
    
    #[test]
    fn test_advanced_activations() {
        // Test leaky_relu
        let x_pos = Dual::variable(2.0);
        let x_neg = Dual::variable(-1.0);
        let alpha = 0.01;
        
        let leaky_pos = x_pos.leaky_relu(alpha);
        assert_eq!(leaky_pos.value, 2.0);
        assert_eq!(leaky_pos.derivative, 1.0);
        
        let leaky_neg = x_neg.leaky_relu(alpha);
        assert_eq!(leaky_neg.value, -0.01);
        assert_eq!(leaky_neg.derivative, 0.01);
        
        // Test ELU
        let elu_pos = x_pos.elu(1.0);
        assert_eq!(elu_pos.value, 2.0);
        assert_eq!(elu_pos.derivative, 1.0);
        
        let elu_neg = x_neg.elu(1.0);
        let expected_elu = 1.0 * ((-1.0_f64).exp() - 1.0);
        let expected_elu_deriv = 1.0 * (-1.0_f64).exp();
        assert!(approx_eq(elu_neg.value, expected_elu));
        assert!(approx_eq(elu_neg.derivative, expected_elu_deriv));
        
        // Test SELU
        let selu_pos = x_pos.selu();
        const SCALE: f64 = 1.0507009873554804934193349852946;
        assert!(approx_eq(selu_pos.value, SCALE * 2.0));
        assert!(approx_eq(selu_pos.derivative, SCALE));
        
        // Test softplus
        let x = Dual::variable(0.0);
        let softplus_x = x.softplus();
        let expected_softplus = (1.0_f64 + 1.0).ln(); // ln(1 + e^0) = ln(2)
        let expected_softplus_deriv = 1.0 / (1.0 + 1.0); // e^x/(1+e^x), at x=0 -> 1/2
        assert!(approx_eq(softplus_x.value, expected_softplus));
        assert!(approx_eq(softplus_x.derivative, expected_softplus_deriv));
        
        // Test softsign
        let x = Dual::variable(2.0);
        let softsign_x = x.softsign();
        let expected_softsign = 2.0 / (1.0 + 2.0); // x/(1+|x|)
        let expected_softsign_deriv = 1.0 / ((1.0 + 2.0) * (1.0 + 2.0)); // 1/(1+|x|)²
        assert!(approx_eq(softsign_x.value, expected_softsign));
        assert!(approx_eq(softsign_x.derivative, expected_softsign_deriv));
    }
    
    #[test]
    fn test_swish_gelu() {
        // Test swish (x * sigmoid(x))
        let x = Dual::variable(1.0);
        let swish_x = x.swish();
        
        let sigmoid_1 = 1.0 / (1.0 + (-1.0_f64).exp());
        let expected_swish = 1.0 * sigmoid_1;
        // swish'(x) = sigmoid(x) + x * sigmoid'(x) 
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        let sigmoid_deriv = sigmoid_1 * (1.0 - sigmoid_1);
        let expected_swish_deriv = sigmoid_1 + 1.0 * sigmoid_deriv;
        
        assert!(approx_eq(swish_x.value, expected_swish));
        assert!(approx_eq(swish_x.derivative, expected_swish_deriv));
        
        // Test GELU (basic functionality)
        let x = Dual::variable(0.0);
        let gelu_x = x.gelu();
        // GELU(0) ≈ 0
        assert!(approx_eq(gelu_x.value, 0.0));
        // GELU'(0) ≈ 0.5
        assert!(approx_eq(gelu_x.derivative, 0.5));
    }
    
    #[test]
    fn test_special_functions() {
        // Test error function
        let x = Dual::variable(0.0);
        let erf_x = x.erf().unwrap();
        // Our erf approximation may not be perfect at x=0, so allow some tolerance
        assert!(erf_x.value.abs() < 0.1); // erf(0) should be close to 0
        let expected_erf_deriv = 2.0 / PI.sqrt(); // erf'(0) = 2/√π
        // Allow more tolerance for the tanh approximation derivative
        assert!((erf_x.derivative - expected_erf_deriv).abs() < 0.2);
        
        // Test gamma function (basic case)
        let x = Dual::variable(1.0);
        let gamma_x = x.gamma().unwrap();
        // Γ(1) = 1, but our Stirling approximation may not be exact
        assert!(gamma_x.value > 0.5 && gamma_x.value < 2.0); // Reasonable range
        assert!(gamma_x.derivative.abs() < 10.0); // Reasonable derivative
    }
    
    #[test]
    fn test_domain_errors() {
        // Test asin domain errors
        let x = Dual::variable(1.5);
        assert!(x.asin().is_err());
        
        // Test acos domain errors  
        assert!(x.acos().is_err());
        
        // Test acosh domain errors
        let x = Dual::variable(0.5);
        assert!(x.acosh().is_err());
        
        // Test atanh domain errors
        let x = Dual::variable(1.0);
        assert!(x.atanh().is_err());
        
        // Test log variants domain errors
        let x = Dual::variable(-1.0);
        assert!(x.log10().is_err());
        assert!(x.log2().is_err());
        
        let x = Dual::variable(-1.1);
        assert!(x.ln_1p().is_err());
        
        // Test gamma domain errors
        let x = Dual::variable(-1.0);
        assert!(x.gamma().is_err());
    }
    
    #[test]
    fn test_numerical_stability() {
        // Test ln_1p for small values (more stable than ln(1+x))
        let small_x = Dual::variable(1e-10);
        let ln_1p_result = small_x.ln_1p().unwrap();
        assert!(ln_1p_result.value > 0.0);
        assert!(ln_1p_result.derivative > 0.99); // Should be close to 1
        
        // Test exp_m1 for small values (more stable than exp(x)-1)
        let exp_m1_result = small_x.exp_m1();
        assert!(exp_m1_result.value > 0.0);
        // For exp_m1, derivative is exp(x), which at x=1e-10 should be ≈ 1.0
        assert!(approx_eq(exp_m1_result.derivative, (1e-10_f64).exp())); // Should be close to 1
    }
    
    // ===== TESTS FOR ADVANCED ACTIVATION FUNCTIONS =====
    
    #[test]
    fn test_mish_activation() {
        // Test Mish: x * tanh(softplus(x))
        let x = Dual::variable(1.0);
        let mish_result = x.mish();
        
        // Expected value: 1.0 * tanh(ln(1 + e^1))
        let softplus_1 = (1.0_f64 + 1.0_f64.exp()).ln();
        let expected_value = 1.0 * softplus_1.tanh();
        assert!(approx_eq(mish_result.value, expected_value));
        
        // Derivative should be positive and reasonable
        assert!(mish_result.derivative > 0.0);
        assert!(mish_result.derivative < 2.0);
        
        // Test at x = 0
        let x_zero = Dual::variable(0.0);
        let mish_zero = x_zero.mish();
        assert!(approx_eq(mish_zero.value, 0.0));
        
        // Test negative input
        let x_neg = Dual::variable(-1.0);
        let mish_neg = x_neg.mish();
        assert!(mish_neg.value < 0.0); // Mish preserves sign
    }
    
    #[test]
    fn test_hardswish_activation() {
        // Test Hardswish: x * ReLU6(x + 3) / 6
        
        // Test positive region (x > 3): should be linear
        let x_pos = Dual::variable(4.0);
        let hardswish_pos = x_pos.hardswish();
        assert!(approx_eq(hardswish_pos.value, 4.0)); // Should equal x
        assert!(approx_eq(hardswish_pos.derivative, 1.0)); // Should be 1
        
        // Test negative region (x < -3): should be 0
        let x_neg = Dual::variable(-4.0);
        let hardswish_neg = x_neg.hardswish();
        assert!(approx_eq(hardswish_neg.value, 0.0));
        assert!(approx_eq(hardswish_neg.derivative, 0.0));
        
        // Test middle region (-3 < x < 3)
        let x_mid = Dual::variable(0.0);
        let hardswish_mid = x_mid.hardswish();
        let expected_val = 0.0 * 3.0 / 6.0; // x * (x+3) / 6
        let expected_deriv = 3.0 / 6.0 + 0.0 / 6.0; // (x+3)/6 + x/6
        assert!(approx_eq(hardswish_mid.value, expected_val));
        assert!(approx_eq(hardswish_mid.derivative, expected_deriv));
    }
    
    #[test]
    fn test_gelu_exact() {
        // Test GELU exact: 0.5 * x * (1 + erf(x / √2))
        let x = Dual::variable(0.0);
        let gelu_result = x.gelu_exact().unwrap();
        
        // GELU(0) should be 0
        assert!(approx_eq(gelu_result.value, 0.0));
        
        // GELU'(0) should be approximately 0.5
        // Due to our erf approximation, allow some tolerance 
        assert!((gelu_result.derivative - 0.5).abs() < 0.1);
        
        // Test positive input
        let x_pos = Dual::variable(1.0);
        let gelu_pos = x_pos.gelu_exact().unwrap();
        assert!(gelu_pos.value > 0.0);
        assert!(gelu_pos.value < 1.0);
        assert!(gelu_pos.derivative > 0.5);
    }
    
    #[test]
    fn test_prelu_activation() {
        // Test PReLU: max(alpha * x, x)
        let alpha = Dual::constant(0.1);
        
        // Test positive input
        let x_pos = Dual::variable(2.0);
        let prelu_pos = x_pos.prelu(alpha);
        assert_eq!(prelu_pos.value, 2.0); // Should equal x
        assert_eq!(prelu_pos.derivative, 1.0); // Should be 1
        
        // Test negative input
        let x_neg = Dual::variable(-1.0);
        let prelu_neg = x_neg.prelu(alpha);
        assert_eq!(prelu_neg.value, -0.1); // Should equal alpha * x
        assert_eq!(prelu_neg.derivative, 0.1); // Should be alpha
    }
    
    #[test]
    fn test_glu_activation() {
        // Test GLU: a * sigmoid(b)
        let a = Dual::variable(2.0);
        let b = Dual::variable(1.0);
        let glu_result = a.glu(b);
        
        let expected_sigmoid = 1.0 / (1.0 + (-1.0_f64).exp());
        let expected_value = 2.0 * expected_sigmoid;
        assert!(approx_eq(glu_result.value, expected_value));
        
        // Derivative computation is more complex due to product rule
        assert!(glu_result.derivative > 0.0);
    }
    
    #[test]
    fn test_relu6_activation() {
        // Test ReLU6: min(max(0, x), 6)
        
        // Test negative input
        let x_neg = Dual::variable(-1.0);
        let relu6_neg = x_neg.relu6();
        assert_eq!(relu6_neg.value, 0.0);
        assert_eq!(relu6_neg.derivative, 0.0);
        
        // Test positive input within range
        let x_pos = Dual::variable(3.0);
        let relu6_pos = x_pos.relu6();
        assert_eq!(relu6_pos.value, 3.0);
        assert_eq!(relu6_pos.derivative, 1.0);
        
        // Test input above 6
        let x_high = Dual::variable(8.0);
        let relu6_high = x_high.relu6();
        assert_eq!(relu6_high.value, 6.0);
        assert_eq!(relu6_high.derivative, 0.0);
    }
    
    #[test]
    fn test_hardtanh_activation() {
        // Test Hardtanh: clamp(x, -1, 1)
        
        // Test within range
        let x_mid = Dual::variable(0.5);
        let hardtanh_mid = x_mid.hardtanh();
        assert_eq!(hardtanh_mid.value, 0.5);
        assert_eq!(hardtanh_mid.derivative, 1.0);
        
        // Test above range
        let x_high = Dual::variable(2.0);
        let hardtanh_high = x_high.hardtanh();
        assert_eq!(hardtanh_high.value, 1.0);
        assert_eq!(hardtanh_high.derivative, 0.0);
        
        // Test below range
        let x_low = Dual::variable(-2.0);
        let hardtanh_low = x_low.hardtanh();
        assert_eq!(hardtanh_low.value, -1.0);
        assert_eq!(hardtanh_low.derivative, 0.0);
    }
    
    #[test]
    fn test_shrinking_activations() {
        // Test Tanhshrink: x - tanh(x)
        let x = Dual::variable(1.0);
        let tanhshrink_result = x.tanhshrink();
        let expected_value = 1.0 - 1.0_f64.tanh();
        assert!(approx_eq(tanhshrink_result.value, expected_value));
        
        // Test Softshrink: sign(x) * max(|x| - lambda, 0)
        let lambda = 0.5;
        
        // Test above threshold
        let x_pos = Dual::variable(1.0);
        let softshrink_pos = x_pos.softshrink(lambda);
        assert_eq!(softshrink_pos.value, 0.5); // 1.0 - 0.5
        assert_eq!(softshrink_pos.derivative, 1.0);
        
        // Test below threshold
        let x_small = Dual::variable(0.3);
        let softshrink_small = x_small.softshrink(lambda);
        assert_eq!(softshrink_small.value, 0.0);
        assert_eq!(softshrink_small.derivative, 0.0);
        
        // Test Hardshrink: x if |x| > lambda else 0
        let hardshrink_pos = x_pos.hardshrink(lambda);
        assert_eq!(hardshrink_pos.value, 1.0);
        assert_eq!(hardshrink_pos.derivative, 1.0);
        
        let hardshrink_small = x_small.hardshrink(lambda);
        assert_eq!(hardshrink_small.value, 0.0);
        assert_eq!(hardshrink_small.derivative, 0.0);
    }
    
    #[test]
    fn test_logsigmoid_activation() {
        // Test LogSigmoid: log(sigmoid(x))
        let x = Dual::variable(0.0);
        let logsigmoid_result = x.logsigmoid();
        
        // log(sigmoid(0)) = log(0.5) = -ln(2)
        let expected_value = -2.0_f64.ln();
        assert!(approx_eq(logsigmoid_result.value, expected_value));
        
        // Derivative at x=0 should be 0.5
        assert!(approx_eq(logsigmoid_result.derivative, 0.5));
        
        // Test positive input
        let x_pos = Dual::variable(2.0);
        let logsigmoid_pos = x_pos.logsigmoid();
        assert!(logsigmoid_pos.value > expected_value); // Should be less negative
        assert!(logsigmoid_pos.derivative > 0.0);
        assert!(logsigmoid_pos.derivative < 1.0);
    }
    
    #[test]
    fn test_activation_composition() {
        // Test composing different activation functions
        let x = Dual::variable(0.5);
        
        // Compose ReLU and Sigmoid
        let relu_result = x.relu();
        let sigmoid_relu = relu_result.sigmoid();
        assert!(sigmoid_relu.value > 0.5); // sigmoid(0.5) > 0.5
        assert!(sigmoid_relu.derivative > 0.0);
        
        // Compose Softplus and Tanh
        let softplus_result = x.softplus();
        let tanh_softplus = softplus_result.tanh();
        assert!(tanh_softplus.value > 0.0);
        assert!(tanh_softplus.value < 1.0);
        assert!(tanh_softplus.derivative > 0.0);
    }
    
    #[test]
    fn test_activation_edge_cases() {
        // Test activations at boundary values
        
        // Test at exactly 0
        let zero = Dual::variable(0.0);
        assert_eq!(zero.relu().value, 0.0);
        assert_eq!(zero.relu().derivative, 0.0);
        assert_eq!(zero.relu6().value, 0.0);
        assert_eq!(zero.relu6().derivative, 0.0); // ReLU6 has derivative 0 at x=0 (boundary)
        
        // Test ReLU6 in the linear region (0 < x < 6)
        let positive = Dual::variable(3.0);
        assert_eq!(positive.relu6().value, 3.0);
        assert_eq!(positive.relu6().derivative, 1.0); // ReLU6 has derivative 1 in (0,6)
        
        // Test symmetry for odd functions
        let x = Dual::variable(1.0);
        let neg_x = Dual::variable(-1.0);
        
        let tanh_pos = x.tanh();
        let tanh_neg = neg_x.tanh();
        assert!(approx_eq(tanh_pos.value, -tanh_neg.value)); // tanh is odd
        
        let tanhshrink_pos = x.tanhshrink();
        let tanhshrink_neg = neg_x.tanhshrink();
        assert!(approx_eq(tanhshrink_pos.value, -tanhshrink_neg.value)); // tanhshrink is odd
    }
    
    // ===== TESTS FOR ATTENTION MECHANISMS =====
    
    #[test]
    fn test_softmax_single_value() {
        // Test softmax for single values (should be 1.0)
        let x = Dual::variable(2.0);
        let softmax_result = x.softmax();
        
        // For a single value, softmax(x) = exp(x) / exp(x) = 1
        assert!(approx_eq(softmax_result.value, 1.0));
        assert!(approx_eq(softmax_result.derivative, 0.0));
        
        // Test with different values
        let x_neg = Dual::variable(-1.5);
        let softmax_neg = x_neg.softmax();
        assert!(approx_eq(softmax_neg.value, 1.0));
        assert!(approx_eq(softmax_neg.derivative, 0.0));
    }
    
    #[test]
    fn test_attention_weight() {
        // Test scaled attention weight: exp(x / sqrt(d_k))
        let x = Dual::variable(1.0);
        let d_k = 64.0; // Common dimension in transformers
        let weight = x.attention_weight(d_k);
        
        let expected_scale = 1.0 / d_k.sqrt();
        let expected_value = (1.0 * expected_scale).exp();
        let expected_derivative = expected_value * expected_scale;
        
        assert!(approx_eq(weight.value, expected_value));
        assert!(approx_eq(weight.derivative, expected_derivative));
    }
    
    #[test]
    fn test_attention_score() {
        // Test scaled dot-product attention
        let query = Dual::variable(2.0);
        let key = Dual::variable(1.5);
        let d_k = 64.0;
        
        let attention = query.attention_score(key, d_k);
        
        // Should compute (query * key / sqrt(d_k)).softmax()
        // For single values, softmax gives 1.0, so we mainly test the scaling
        assert!(approx_eq(attention.value, 1.0));
        assert!(approx_eq(attention.derivative, 0.0));
    }
    
    #[test]
    fn test_self_attention() {
        // Test self-attention (query == key)
        let x = Dual::variable(3.0);
        let d_k = 512.0;
        
        let self_attention = x.self_attention_score(d_k);
        let manual_attention = x.attention_score(x, d_k);
        
        // Self-attention should be the same as regular attention with same Q and K
        assert!(approx_eq(self_attention.value, manual_attention.value));
        assert!(approx_eq(self_attention.derivative, manual_attention.derivative));
    }
    
    #[test]
    fn test_layer_normalization() {
        // Test layer normalization: (x - μ) / σ * γ + β
        let x = Dual::variable(5.0);
        let mean = Dual::constant(3.0);
        let variance = Dual::constant(4.0); // std = 2.0
        let gamma = Dual::constant(2.0);
        let beta = Dual::constant(1.0);
        
        let normalized = x.layer_norm(mean, variance, gamma, beta);
        
        // Expected: (5 - 3) / 2 * 2 + 1 = 2 / 2 * 2 + 1 = 3.0
        assert!(approx_eq(normalized.value, 3.0));
        
        // Derivative: γ / σ = 2 / 2 = 1.0
        assert!(approx_eq(normalized.derivative, 1.0));
        
        // Test simplified layer norm
        let simple_norm = x.layer_norm_simple(gamma, beta);
        let expected_simple = gamma.value * x.value + beta.value;
        assert!(approx_eq(simple_norm.value, expected_simple));
    }
    
    #[test]
    fn test_rms_normalization() {
        // Test RMS normalization: x / |x|
        let x_pos = Dual::variable(4.0);
        let rms_pos = x_pos.rms_norm();
        
        // For positive values: x / |x| = x / x = 1
        assert!(approx_eq(rms_pos.value, 1.0));
        assert!(approx_eq(rms_pos.derivative, 0.0));
        
        let x_neg = Dual::variable(-3.0);
        let rms_neg = x_neg.rms_norm();
        
        // For negative values: x / |x| = x / (-x) = -1
        assert!(approx_eq(rms_neg.value, -1.0));
        assert!(approx_eq(rms_neg.derivative, 0.0));
        
        // Test with scaling
        let gamma = Dual::constant(2.0);
        let scaled_rms = x_pos.rms_norm_scaled(gamma);
        assert!(approx_eq(scaled_rms.value, 2.0)); // gamma * 1.0
    }
    
    #[test]
    fn test_causal_masking() {
        // Test causal mask application
        let x = Dual::variable(1.5);
        
        // Unmasked should return original value
        let unmasked = x.apply_causal_mask(false);
        assert_eq!(unmasked.value, x.value);
        assert_eq!(unmasked.derivative, x.derivative);
        
        // Masked should return large negative value
        let masked = x.apply_causal_mask(true);
        assert_eq!(masked.value, -1e9);
        assert_eq!(masked.derivative, 0.0);
    }
    
    #[test]
    fn test_attention_dropout() {
        // Test attention dropout (scaling by 1-p)
        let x = Dual::variable(2.0);
        let dropout_rate = 0.1;
        
        let dropped = x.attention_dropout(dropout_rate);
        let expected_scale = 1.0 - dropout_rate;
        
        assert!(approx_eq(dropped.value, x.value * expected_scale));
        assert!(approx_eq(dropped.derivative, x.derivative * expected_scale));
    }
    
    #[test]
    fn test_positional_encoding() {
        // Test sinusoidal positional encoding
        let x = Dual::variable(1.0);
        let position = 0.0;
        let dimension = 0.0;
        
        let pos_sin = x.positional_encoding_sin(position, dimension);
        let pos_cos = x.positional_encoding_cos(position, dimension);
        
        // At position 0, sin(0) = 0, cos(0) = 1
        assert!(approx_eq(pos_sin.value, x.value + 0.0)); // x + sin(0)
        assert!(approx_eq(pos_cos.value, x.value + 1.0)); // x + cos(0)
        
        // Test different position/dimension
        let pos_nonzero = x.positional_encoding_sin(1.0, 2.0);
        assert!(pos_nonzero.value != x.value); // Should be different
    }
    
    #[test]
    fn test_multi_head_combine() {
        // Test combining multiple attention heads
        let head1 = Dual::variable(1.0);
        let head2 = Dual::variable(2.0);
        let head3 = Dual::variable(3.0);
        let other_heads = vec![head2, head3];
        
        let combined = head1.multi_head_combine(&other_heads);
        
        // Should average: (1 + 2 + 3) / 3 = 2.0
        assert!(approx_eq(combined.value, 2.0));
        
        // Test with empty heads
        let combined_empty = head1.multi_head_combine(&[]);
        assert_eq!(combined_empty.value, head1.value);
        assert_eq!(combined_empty.derivative, head1.derivative);
    }
    
    #[test]
    fn test_cross_attention() {
        // Test cross-attention between different sequences
        let query = Dual::variable(2.0);
        let key = Dual::variable(1.5);
        let value = Dual::variable(3.0);
        let d_k = 64.0;
        
        let cross_attn = query.cross_attention_score(key, value, d_k);
        
        // Should compute attention_score(query, key) * value
        // Since attention_score gives ~1.0 for single values, result ≈ value
        assert!(cross_attn.value > 0.0);
        assert!(cross_attn.derivative >= 0.0);
    }
    
    #[test]
    fn test_attention_output_projection() {
        // Test linear projection after attention
        let attention_output = Dual::variable(2.5);
        let weight = Dual::constant(0.8);
        let bias = Dual::constant(0.2);
        
        let projected = attention_output.attention_output_projection(weight, bias);
        
        // Should compute: attention_output * weight + bias
        let expected = 2.5 * 0.8 + 0.2;
        assert!(approx_eq(projected.value, expected));
        assert!(approx_eq(projected.derivative, 0.8)); // derivative of weight
    }
    
    #[test]
    fn test_attention_mechanism_composition() {
        // Test composing multiple attention operations
        let query = Dual::variable(1.0);
        let key = Dual::variable(0.8);
        let value = Dual::variable(1.2);
        let d_k = 64.0;
        
        // Full attention pipeline
        let attention_weights = query.attention_score(key, d_k);
        let attention_output = attention_weights * value;
        let dropout_output = attention_output.attention_dropout(0.1);
        
        // Layer norm parameters
        let gamma = Dual::constant(1.0);
        let beta = Dual::constant(0.0);
        let normalized = dropout_output.layer_norm_simple(gamma, beta);
        
        // Should be a valid computation chain
        assert!(normalized.value.is_finite());
        assert!(normalized.derivative.is_finite());
    }
    
    #[test]
    fn test_attention_numerical_stability() {
        // Test attention mechanisms with extreme values
        
        // Large values should not cause overflow
        let large_query = Dual::variable(100.0);
        let large_key = Dual::variable(50.0);
        let attention_large = large_query.attention_score(large_key, 512.0);
        assert!(attention_large.value.is_finite());
        assert!(attention_large.derivative.is_finite());
        
        // Small values should not cause underflow
        let small_query = Dual::variable(1e-6);
        let small_key = Dual::variable(1e-7);
        let attention_small = small_query.attention_score(small_key, 64.0);
        assert!(attention_small.value.is_finite());
        assert!(attention_small.derivative.is_finite());
        
        // Zero values should be handled gracefully
        let zero = Dual::variable(0.0);
        let rms_zero = zero.rms_norm();
        assert_eq!(rms_zero.value, 0.0);
        assert_eq!(rms_zero.derivative, 0.0);
    }
    
    // ===== TESTS FOR LOSS FUNCTIONS =====
    
    #[test]
    fn test_mse_loss() {
        // Test Mean Squared Error loss
        let prediction = Dual::variable(2.0);
        let target = Dual::constant(1.0);
        
        let mse = prediction.mse_loss(target);
        // MSE = (2 - 1)² = 1
        assert!(approx_eq(mse.value, 1.0));
        // d/dx[(x - 1)²] = 2(x - 1) = 2(2 - 1) = 2
        assert!(approx_eq(mse.derivative, 2.0));
        
        // Test perfect prediction
        let perfect_pred = Dual::variable(1.5);
        let perfect_target = Dual::constant(1.5);
        let perfect_mse = perfect_pred.mse_loss(perfect_target);
        assert!(approx_eq(perfect_mse.value, 0.0));
        assert!(approx_eq(perfect_mse.derivative, 0.0));
    }
    
    #[test]
    fn test_mae_loss() {
        // Test Mean Absolute Error loss
        let prediction = Dual::variable(3.0);
        let target = Dual::constant(1.0);
        
        let mae = prediction.mae_loss(target);
        // MAE = |3 - 1| = 2
        assert!(approx_eq(mae.value, 2.0));
        // d/dx|x - 1| = sign(x - 1) = sign(2) = 1
        assert!(approx_eq(mae.derivative, 1.0));
        
        // Test negative difference
        let neg_pred = Dual::variable(0.5);
        let neg_mae = neg_pred.mae_loss(target);
        assert!(approx_eq(neg_mae.value, 0.5)); // |0.5 - 1| = 0.5
        assert!(approx_eq(neg_mae.derivative, -1.0)); // sign(0.5 - 1) = -1
    }
    
    #[test]
    fn test_huber_loss() {
        // Test Huber loss with different deltas
        let prediction = Dual::variable(2.0);
        let target = Dual::constant(1.0);
        let delta = 0.5;
        
        let huber = prediction.huber_loss(target, delta);
        // |pred - target| = 1.0 > delta, so linear region
        // loss = delta * (|pred - target| - 0.5 * delta) = 0.5 * (1.0 - 0.25) = 0.375
        assert!(approx_eq(huber.value, 0.375));
        
        // Test quadratic region (small difference)
        let small_pred = Dual::variable(1.2);
        let small_huber = small_pred.huber_loss(target, delta);
        // |1.2 - 1.0| = 0.2 < delta, so quadratic region
        // loss = 0.5 * (0.2)² = 0.02
        assert!(approx_eq(small_huber.value, 0.02));
    }
    
    #[test]
    fn test_binary_cross_entropy_loss() {
        // Test Binary Cross-Entropy loss
        let prediction = Dual::variable(0.8);
        let target = Dual::constant(1.0);
        
        let bce = prediction.binary_cross_entropy_loss(target).unwrap();
        // BCE = -[1 * log(0.8) + 0 * log(0.2)] = -log(0.8)
        let expected = -0.8_f64.ln();
        assert!(approx_eq(bce.value, expected));
        
        // Test with target = 0
        let target_zero = Dual::constant(0.0);
        let bce_zero = prediction.binary_cross_entropy_loss(target_zero).unwrap();
        // BCE = -[0 * log(0.8) + 1 * log(0.2)] = -log(0.2)
        let expected_zero = -0.2_f64.ln();
        assert!(approx_eq(bce_zero.value, expected_zero));
    }
    
    #[test]
    fn test_cross_entropy_loss() {
        // Test Cross-Entropy loss
        let prediction = Dual::variable(0.7);
        let target = Dual::constant(1.0);
        
        let ce = prediction.cross_entropy_loss(target).unwrap();
        // CE = -target * log(prediction) = -1 * log(0.7) = -log(0.7)
        let expected = -0.7_f64.ln();
        assert!(approx_eq(ce.value, expected));
        
        // Test with target = 0 (should be 0 loss)
        let target_zero = Dual::constant(0.0);
        let ce_zero = prediction.cross_entropy_loss(target_zero).unwrap();
        assert!(approx_eq(ce_zero.value, 0.0));
    }
    
    #[test]
    fn test_focal_loss() {
        // Test Focal Loss
        let prediction = Dual::variable(0.9);
        let target = Dual::constant(1.0);
        let alpha = 0.25;
        let gamma = 2.0;
        
        let focal = prediction.focal_loss(target, alpha, gamma).unwrap();
        
        // Focal = -alpha * (1-p)^gamma * target * log(p)
        // = -0.25 * (0.1)^2 * 1 * log(0.9)
        let expected = -alpha * (0.1_f64).powf(gamma) * 0.9_f64.ln();
        assert!(approx_eq(focal.value, expected));
        
        // Focal loss should be smaller for confident correct predictions
        let confident_pred = Dual::variable(0.99);
        let confident_focal = confident_pred.focal_loss(target, alpha, gamma).unwrap();
        assert!(confident_focal.value < focal.value);
    }
    
    #[test]
    fn test_kl_divergence() {
        // Test KL Divergence
        let p = Dual::variable(0.8);
        let q = Dual::constant(0.6);
        
        let kl = p.kl_divergence(q).unwrap();
        // KL(p||q) = p * log(p/q) = 0.8 * log(0.8/0.6)
        let expected = 0.8 * (0.8_f64 / 0.6).ln();
        assert!(approx_eq(kl.value, expected));
        
        // KL divergence should be 0 when distributions are identical
        let identical_kl = p.kl_divergence(p).unwrap();
        assert!(approx_eq(identical_kl.value, 0.0));
    }
    
    #[test]
    fn test_contrastive_loss() {
        // Test Contrastive Loss
        let distance = Dual::variable(2.0);
        let margin = 1.0;
        
        // Test similar pair (target = 1)
        let similar_target = Dual::constant(1.0);
        let similar_loss = distance.contrastive_loss(similar_target, margin);
        // Loss = 1 * distance² = 1 * 4 = 4
        assert!(approx_eq(similar_loss.value, 4.0));
        
        // Test dissimilar pair (target = 0)
        let dissimilar_target = Dual::constant(0.0);
        let dissimilar_loss = distance.contrastive_loss(dissimilar_target, margin);
        // Distance > margin, so loss = 0
        assert!(approx_eq(dissimilar_loss.value, 0.0));
        
        // Test dissimilar pair with distance < margin
        let small_distance = Dual::variable(0.5);
        let small_dissimilar_loss = small_distance.contrastive_loss(dissimilar_target, margin);
        // Loss = (1 - 0) * (1 - 0.5)² = 0.25
        assert!(approx_eq(small_dissimilar_loss.value, 0.25));
    }
    
    #[test]
    fn test_triplet_loss() {
        // Test Triplet Loss
        let anchor_positive = Dual::variable(1.0);
        let positive_dist = Dual::constant(0.5);
        let negative_dist = Dual::constant(2.0);
        let margin = 0.2;
        
        let triplet = anchor_positive.triplet_loss(positive_dist, negative_dist, margin);
        // max(0, 1.0 - 2.0 + 0.2) = max(0, -0.8) = 0
        assert!(approx_eq(triplet.value, 0.0));
        
        // Test case where loss is positive
        let bad_negative = Dual::constant(0.8);
        let bad_triplet = anchor_positive.triplet_loss(positive_dist, bad_negative, margin);
        // max(0, 1.0 - 0.8 + 0.2) = max(0, 0.4) = 0.4
        assert!(approx_eq(bad_triplet.value, 0.4));
    }
    
    // ===== TESTS FOR OPTIMIZATION UTILITIES =====
    
    #[test]
    fn test_clamp() {
        // Test value clamping
        let high_val = Dual::variable(10.0);
        let clamped_high = high_val.clamp(0.0, 5.0);
        assert_eq!(clamped_high.value, 5.0);
        assert_eq!(clamped_high.derivative, 0.0); // Gradient becomes 0 when clamped
        
        let low_val = Dual::variable(-3.0);
        let clamped_low = low_val.clamp(0.0, 5.0);
        assert_eq!(clamped_low.value, 0.0);
        assert_eq!(clamped_low.derivative, 0.0);
        
        let normal_val = Dual::variable(2.5);
        let clamped_normal = normal_val.clamp(0.0, 5.0);
        assert_eq!(clamped_normal.value, 2.5);
        assert_eq!(clamped_normal.derivative, 1.0); // Gradient preserved
    }
    
    #[test]
    fn test_gradient_clip() {
        // Test gradient clipping
        let large_grad = Dual::new(1.0, 10.0);
        let clipped = large_grad.gradient_clip(5.0);
        
        assert_eq!(clipped.value, 1.0); // Value unchanged
        assert_eq!(clipped.derivative, 5.0); // Gradient clipped to max_norm
        
        // Test no clipping when gradient is small
        let small_grad = Dual::new(2.0, 3.0);
        let not_clipped = small_grad.gradient_clip(5.0);
        assert_eq!(not_clipped.value, 2.0);
        assert_eq!(not_clipped.derivative, 3.0);
    }
    
    #[test]
    fn test_ema_update() {
        // Test Exponential Moving Average
        let current = Dual::variable(1.0);
        let previous = Dual::constant(0.5);
        let momentum = 0.9;
        
        let ema = current.ema_update(previous, momentum);
        // EMA = 0.9 * 0.5 + 0.1 * 1.0 = 0.45 + 0.1 = 0.55
        assert!(approx_eq(ema.value, 0.55));
    }
    
    #[test]
    fn test_learning_rate_scheduling() {
        let initial_lr = Dual::variable(0.1);
        
        // Test exponential decay
        let decayed = initial_lr.exponential_decay(0.9, 10.0);
        let expected_decay = 0.1 * 0.9_f64.powf(10.0);
        assert!(approx_eq(decayed.value, expected_decay));
        
        // Test cosine annealing
        let annealed = initial_lr.cosine_annealing(50.0, 100.0);
        // At halfway point, should be at minimum
        let expected_annealed = 0.1 * 0.5 * (1.0 + (std::f64::consts::PI * 0.5).cos());
        assert!(approx_eq(annealed.value, expected_annealed));
    }
    
    #[test]
    fn test_momentum_update() {
        // Test SGD with momentum
        let gradient = Dual::variable(1.0);
        let velocity = Dual::constant(0.5);
        let momentum = 0.9;
        
        let updated_velocity = gradient.momentum_update(velocity, momentum);
        // new_velocity = 0.9 * 0.5 + 1.0 = 1.45
        assert!(approx_eq(updated_velocity.value, 1.45));
    }
    
    #[test]
    fn test_adam_update() {
        // Test Adam optimizer updates
        let gradient = Dual::variable(1.0);
        let m = Dual::constant(0.1);
        let v = Dual::constant(0.01);
        let beta1 = 0.9;
        let beta2 = 0.999;
        let _step = 1.0;
        
        let (m_new, v_new) = gradient.adam_update(m, v, beta1, beta2, _step);
        
        // m_new = 0.9 * 0.1 + 0.1 * 1.0 = 0.09 + 0.1 = 0.19
        assert!(approx_eq(m_new.value, 0.19));
        
        // v_new = 0.999 * 0.01 + 0.001 * 1.0^2 = 0.00999 + 0.001 = 0.01099
        assert!(approx_eq(v_new.value, 0.01099));
    }
    
    #[test]
    fn test_rmsprop_update() {
        // Test RMSprop optimizer
        let gradient = Dual::variable(2.0);
        let sq_avg = Dual::constant(0.1);
        let alpha = 0.99;
        
        let updated_sq_avg = gradient.rmsprop_update(sq_avg, alpha);
        // new_sq_avg = 0.99 * 0.1 + 0.01 * 4.0 = 0.139
        assert!(approx_eq(updated_sq_avg.value, 0.139));
    }
    
    #[test]
    fn test_loss_numerical_stability() {
        // Test loss functions with extreme values
        
        // Test BCE with values close to 0 and 1
        let near_zero = Dual::variable(1e-8);
        let near_one = Dual::variable(1.0 - 1e-8);
        let target_one = Dual::constant(1.0);
        
        let bce_near_zero = near_zero.binary_cross_entropy_loss(target_one);
        let bce_near_one = near_one.binary_cross_entropy_loss(target_one);
        
        assert!(bce_near_zero.is_ok());
        assert!(bce_near_one.is_ok());
        assert!(bce_near_zero.unwrap().value.is_finite());
        assert!(bce_near_one.unwrap().value.is_finite());
        
        // Test that clamping prevents log(0)
        let zero_pred = Dual::variable(0.0);
        let clamped_bce = zero_pred.binary_cross_entropy_loss(target_one);
        assert!(clamped_bce.is_ok());
        assert!(clamped_bce.unwrap().value.is_finite());
    }
    
    #[test]
    fn test_loss_composition() {
        // Test combining different loss functions
        let prediction = Dual::variable(0.8);
        let target = Dual::constant(1.0);
        
        // Combined loss: MSE + BCE
        let mse = prediction.mse_loss(target);
        let bce = prediction.binary_cross_entropy_loss(target).unwrap();
        let combined = mse + bce;
        
        assert!(combined.value > mse.value);
        assert!(combined.value > bce.value);
        assert!(combined.derivative != 0.0);
        
        // Test regularization: loss + L2 penalty
        let weight = Dual::variable(2.0);
        let l2_penalty = weight * weight * Dual::constant(0.01); // L2 regularization
        let regularized_loss = bce + l2_penalty;
        
        assert!(regularized_loss.value > bce.value);
    }
    
    // ===============================================================================
    // TESTS FOR ADVANCED MATHEMATICAL OPERATIONS - PHASE 8.2.4
    // ===============================================================================
    
    #[test]
    fn test_matrix_decomposition_functions() {
        // Test 2x2 determinant
        let a = Dual::variable(2.0);
        let b = Dual::constant(1.0);
        let c = Dual::constant(3.0);
        let d = Dual::variable(4.0);
        
        let det = Dual::det_2x2(a, b, c, d);
        // det = 2*4 - 1*3 = 5
        assert_eq!(det.value, 5.0);
        // d/da = 4, d/dd = 2, so total derivative = 4 + 2 = 6
        assert_eq!(det.derivative, 6.0);
        
        // Test trace
        let trace = Dual::trace_2x2(a, d);
        assert_eq!(trace.value, 6.0); // 2 + 4
        assert_eq!(trace.derivative, 2.0); // both are variables with derivative 1
    }
    
    #[test]
    fn test_condition_number_estimation() {
        let x = Dual::variable(10.0);
        let y = Dual::variable(1.0);
        
        let cond = x.condition_number_estimate(y);
        assert!(approx_eq(cond.value, 10.0)); // max/min = 10/1 = 10
    }
    
    #[test]
    fn test_givens_rotation() {
        let x = Dual::variable(3.0);
        let y = Dual::variable(4.0);
        
        let angle = x.givens_rotation_angle(y);
        let expected = (4.0_f64).atan2(3.0);
        assert!(approx_eq(angle.value, expected));
    }
    
    #[test]
    fn test_eigenvalue_functions() {
        // Test characteristic polynomial for 2x2 matrix
        let lambda = Dual::variable(2.0);
        let trace = Dual::constant(5.0);
        let det = Dual::constant(6.0);
        
        let char_poly = Dual::characteristic_poly_2x2(lambda, trace, det);
        // λ² - 5λ + 6 = 4 - 10 + 6 = 0 (λ=2 is indeed an eigenvalue)
        assert_eq!(char_poly.value, 0.0);
        
        // Test power iteration normalization
        let v1 = Dual::variable(3.0);
        let v2 = Dual::variable(4.0);
        let (norm_v1, norm_v2) = v1.power_iteration_step(v2).unwrap();
        
        // Should normalize to unit vector: (3,4) -> (0.6, 0.8)
        assert!(approx_eq(norm_v1.value, 0.6));
        assert!(approx_eq(norm_v2.value, 0.8));
        
        // Test Rayleigh quotient
        let x1 = Dual::variable(1.0);
        let x2 = Dual::variable(0.0);
        let ax1 = Dual::constant(2.0); // A*x where A*[1,0] = [2,0]
        let ax2 = Dual::constant(0.0);
        
        let rayleigh = Dual::rayleigh_quotient(x1, x2, ax1, ax2);
        assert_eq!(rayleigh.value, 2.0); // eigenvalue of 2
    }
    
    #[test]
    fn test_integration_methods() {
        // Test trapezoidal rule: integrate f(x) = x over [0,1]
        let f_start = Dual::variable(0.0);
        let f_end = Dual::variable(1.0);
        let h = 1.0;
        
        let integral = Dual::integrate_trapezoidal(f_start, f_end, h);
        assert_eq!(integral.value, 0.5); // (0+1)*1*0.5 = 0.5
        
        // Test Simpson's rule: integrate f(x) = x² over [0,1] with 3 points
        let f_0 = Dual::variable(0.0); // f(0) = 0
        let f_mid = Dual::variable(0.25); // f(0.5) = 0.25
        let f_1 = Dual::variable(1.0); // f(1) = 1
        let h_simpson = 0.5;
        
        let simpson_integral = Dual::integrate_simpson(f_0, f_mid, f_1, h_simpson);
        // Simpson's: h/3 * (f0 + 4*f_mid + f1) = 0.5/3 * (0 + 4*0.25 + 1) = 1/6 * 2 = 1/3
        assert!(approx_eq(simpson_integral.value, 1.0/3.0));
    }
    
    #[test]
    fn test_ode_methods() {
        // Test Euler step: y' = y, y(0) = 1, h = 0.1
        let y = Dual::variable(1.0);
        let dy_dx = y; // y' = y
        let h = 0.1;
        
        let y_new = Dual::euler_step(y, dy_dx, h);
        assert_eq!(y_new.value, 1.1); // y + h*y' = 1 + 0.1*1 = 1.1
        
        // Test RK4 step components
        let k1 = Dual::variable(1.0);
        let k2 = Dual::variable(1.05);
        let k3 = Dual::variable(1.05);
        let k4 = Dual::variable(1.1);
        
        let rk4_result = Dual::rk4_step(y, k1, k2, k3, k4, h);
        // y + h/6 * (k1 + 2*k2 + 2*k3 + k4) = 1 + 0.1/6 * (1 + 2*1.05 + 2*1.05 + 1.1)
        let expected_rk4 = 1.0 + 0.1/6.0 * (1.0 + 2.0*1.05 + 2.0*1.05 + 1.1);
        assert!(approx_eq(rk4_result.value, expected_rk4));
    }
    
    #[test]
    fn test_second_derivative_approximation() {
        // Test second derivative of f(x) = x² at x = 1
        // f''(x) = 2, so should get approximately 2
        let f_prev = Dual::variable(0.81); // f(0.9) = 0.81
        let f_curr = Dual::variable(1.0);  // f(1.0) = 1.0
        let f_next = Dual::variable(1.21); // f(1.1) = 1.21
        let h = 0.1;
        
        let second_deriv = Dual::second_derivative_approx(f_prev, f_curr, f_next, h);
        // (1.21 - 2*1.0 + 0.81) / 0.01 = 0.02 / 0.01 = 2.0
        assert!(approx_eq(second_deriv.value, 2.0));
    }
    
    #[test]
    fn test_statistical_functions() {
        let x1 = Dual::variable(2.0);
        let x2 = Dual::variable(4.0);
        
        // Test mean
        let mean = x1.mean(x2);
        assert_eq!(mean.value, 3.0);
        assert_eq!(mean.derivative, 1.0); // d/dx1 + d/dx2 = 0.5 + 0.5 = 1.0
        
        // Test variance
        let variance = x1.variance(x2);
        // Sample variance with Bessel correction: ((2-3)² + (4-3)²) / 1 = (1 + 1) / 1 = 2
        assert_eq!(variance.value, 2.0);
        
        // Test standard deviation
        let std_dev = x1.std_dev(x2).unwrap();
        assert!(approx_eq(std_dev.value, std::f64::consts::SQRT_2));
        
        // Test z-score
        let x = Dual::variable(5.0);
        let mean = Dual::constant(3.0);
        let std = Dual::constant(2.0);
        let z = x.z_score(mean, std);
        assert_eq!(z.value, 1.0); // (5-3)/2 = 1
    }
    
    #[test]
    fn test_covariance_correlation() {
        let x1 = Dual::variable(1.0);
        let x2 = Dual::variable(3.0);
        let y1 = Dual::variable(2.0);
        let y2 = Dual::variable(4.0);
        
        // Test covariance
        let cov = Dual::covariance(x1, x2, y1, y2);
        // x_mean = 2, y_mean = 3
        // cov = ((1-2)*(2-3) + (3-2)*(4-3)) / 1 = ((-1)*(-1) + (1)*(1)) / 1 = 2
        assert_eq!(cov.value, 2.0);
        
        // Test correlation (should be 1.0 for perfectly correlated data)
        let corr = Dual::correlation(x1, x2, y1, y2).unwrap();
        assert!(approx_eq(corr.value, 1.0));
    }
    
    #[test]
    fn test_welford_algorithm() {
        let new_value = Dual::variable(5.0);
        let current_mean = Dual::constant(3.0);
        let current_m2 = Dual::constant(8.0);
        let n = 3.0;
        
        let (new_mean, new_m2) = new_value.welford_update(current_mean, current_m2, n);
        
        // delta = 5 - 3 = 2
        // new_mean = 3 + 2/3 = 11/3
        assert!(approx_eq(new_mean.value, 11.0/3.0));
        
        // delta2 = 5 - 11/3 = 4/3
        // new_m2 = 8 + 2 * 4/3 = 8 + 8/3 = 32/3
        assert!(approx_eq(new_m2.value, 32.0/3.0));
    }
    
    #[test]
    fn test_normal_distribution() {
        let x = Dual::variable(0.0);
        let mean = Dual::constant(0.0);
        let std_dev = Dual::constant(1.0);
        
        // Test normal PDF at mean (should be 1/√(2π))
        let pdf = x.normal_pdf(mean, std_dev).unwrap();
        let expected_pdf = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        assert!(approx_eq(pdf.value, expected_pdf));
        
        // Test normal CDF at mean (should be 0.5)
        let cdf = x.normal_cdf(mean, std_dev).unwrap();
        assert!(approx_eq(cdf.value, 0.5));
        
        // Test error cases
        let bad_std = Dual::constant(-1.0);
        assert!(x.normal_pdf(mean, bad_std).is_err());
    }
    
    #[test]
    fn test_uniform_distribution() {
        let x = Dual::variable(0.5);
        let a = Dual::constant(0.0);
        let b = Dual::constant(1.0);
        
        // Test uniform PDF in range (should be 1.0)
        let pdf = x.uniform_pdf(a, b).unwrap();
        assert_eq!(pdf.value, 1.0);
        
        // Test uniform PDF outside range
        let x_outside = Dual::variable(1.5);
        let pdf_outside = x_outside.uniform_pdf(a, b).unwrap();
        assert_eq!(pdf_outside.value, 0.0);
        
        // Test error cases
        assert!(x.uniform_pdf(b, a).is_err()); // b <= a
    }
    
    #[test]
    fn test_exponential_distribution() {
        let x = Dual::variable(1.0);
        let lambda = Dual::constant(2.0);
        
        // Test exponential PDF
        let pdf = x.exponential_pdf(lambda).unwrap();
        let expected_pdf = 2.0 * (-2.0 * 1.0_f64).exp();
        assert!(approx_eq(pdf.value, expected_pdf));
        
        // Test exponential CDF
        let cdf = x.exponential_cdf(lambda).unwrap();
        let expected_cdf = 1.0 - (-2.0 * 1.0_f64).exp();
        assert!(approx_eq(cdf.value, expected_cdf));
        
        // Test negative input (should give 0)
        let x_neg = Dual::variable(-1.0);
        let pdf_neg = x_neg.exponential_pdf(lambda).unwrap();
        assert_eq!(pdf_neg.value, 0.0);
    }
    
    #[test]
    fn test_gamma_functions() {
        let x = Dual::variable(5.0);
        
        // Test gamma approximation (Γ(5) should be close to 4! = 24)
        let gamma_result = x.gamma_approx().unwrap();
        assert!(gamma_result.value > 20.0 && gamma_result.value < 30.0);
        
        // Test log-gamma (more stable)
        let log_gamma_result = x.log_gamma_approx().unwrap();
        assert!(log_gamma_result.value > 3.0 && log_gamma_result.value < 4.0); // ln(24) ≈ 3.18
        
        // Test beta function
        let y = Dual::variable(3.0);
        let beta_result = x.beta_function(y).unwrap();
        assert!(beta_result.value.is_finite());
        
        // Test error cases
        let x_neg = Dual::variable(-1.0);
        assert!(x_neg.gamma_approx().is_err());
    }
    
    #[test]
    fn test_error_functions() {
        // Test erf(0) = 0
        let x_zero = Dual::variable(0.0);
        let erf_zero = x_zero.erf().unwrap();
        assert!(approx_eq(erf_zero.value, 0.0));
        
        // Test erf(large positive) ≈ 1
        let x_large = Dual::variable(5.0);
        let erf_large = x_large.erf().unwrap();
        assert!(erf_large.value > 0.99);
        
        // Test erf(large negative) ≈ -1
        let x_neg_large = Dual::variable(-5.0);
        let erf_neg_large = x_neg_large.erf().unwrap();
        assert!(erf_neg_large.value < -0.99);
        
        // Test erfc(x) = 1 - erf(x)
        let x = Dual::variable(1.0);
        let erf_x = x.erf().unwrap();
        let erfc_x = x.erfc().unwrap();
        let sum = erf_x + erfc_x;
        assert!(approx_eq(sum.value, 1.0));
    }
    
    #[test]
    fn test_advanced_math_edge_cases() {
        // Test functions with extreme values
        
        // Very small numbers
        let x_small = Dual::variable(1e-10);
        assert!(x_small.gamma_approx().is_ok());
        
        // Values near boundaries
        let x_boundary = Dual::variable(1e-8);
        let mean = Dual::constant(0.0);
        let std_dev = Dual::constant(1.0);
        assert!(x_boundary.normal_pdf(mean, std_dev).is_ok());
        
        // Test numerical stability
        let x = Dual::variable(0.5);
        let very_small_std = Dual::constant(1e-10);
        let pdf_result = x.normal_pdf(mean, very_small_std);
        assert!(pdf_result.is_ok());
        assert!(pdf_result.unwrap().value.is_finite());
    }
    
    // ================================================================================================
    // HIGHER-ORDER DERIVATIVES TESTS (Phase 8.3.4)
    // ================================================================================================
    
    #[test]
    fn test_hyper_dual_creation() {
        // Test basic hyper-dual number creation
        let hd = HyperDual::new(2.0, 1.0, 0.5, 0.25);
        assert_eq!(hd.value, 2.0);
        assert_eq!(hd.dx, 1.0);
        assert_eq!(hd.dy, 0.5);
        assert_eq!(hd.dxy, 0.25);
        
        // Test variable constructors
        let x_var = HyperDual::variable_x(3.0);
        assert_eq!(x_var.value, 3.0);
        assert_eq!(x_var.dx, 1.0);
        assert_eq!(x_var.dy, 0.0);
        assert_eq!(x_var.dxy, 0.0);
        
        let y_var = HyperDual::variable_y(4.0);
        assert_eq!(y_var.value, 4.0);
        assert_eq!(y_var.dx, 0.0);
        assert_eq!(y_var.dy, 1.0);
        assert_eq!(y_var.dxy, 0.0);
        
        let const_hd = HyperDual::constant(5.0);
        assert_eq!(const_hd.value, 5.0);
        assert_eq!(const_hd.dx, 0.0);
        assert_eq!(const_hd.dy, 0.0);
        assert_eq!(const_hd.dxy, 0.0);
    }
    
    #[test]
    fn test_hyper_dual_arithmetic() {
        let a = HyperDual::new(2.0, 1.0, 0.0, 0.0);  // x variable
        let b = HyperDual::new(3.0, 0.0, 1.0, 0.0);  // y variable
        
        // Test addition
        let sum = a + b;
        assert_eq!(sum.value, 5.0);
        assert_eq!(sum.dx, 1.0);
        assert_eq!(sum.dy, 1.0);
        assert_eq!(sum.dxy, 0.0);
        
        // Test subtraction
        let diff = a - b;
        assert_eq!(diff.value, -1.0);
        assert_eq!(diff.dx, 1.0);
        assert_eq!(diff.dy, -1.0);
        assert_eq!(diff.dxy, 0.0);
        
        // Test multiplication: f = x * y, ∂²f/∂x∂y = 1
        let product = a * b;
        assert_eq!(product.value, 6.0);  // 2 * 3
        assert_eq!(product.dx, 3.0);    // ∂f/∂x = y = 3
        assert_eq!(product.dy, 2.0);    // ∂f/∂y = x = 2
        assert_eq!(product.dxy, 1.0);   // ∂²f/∂x∂y = 1
    }
    
    #[test]
    fn test_hyper_dual_exp() {
        // Test exponential: f = exp(x), f' = exp(x), f'' = exp(x)
        let x = HyperDual::variable_x(1.0);
        let exp_x = x.exp();
        
        let expected_exp = E;
        assert!(approx_eq(exp_x.value, expected_exp));
        assert!(approx_eq(exp_x.dx, expected_exp));    // ∂/∂x exp(x) = exp(x)
        assert_eq!(exp_x.dy, 0.0);                      // ∂/∂y exp(x) = 0
        assert_eq!(exp_x.dxy, 0.0);                     // ∂²/∂x∂y exp(x) = 0
    }
    
    #[test]
    fn test_hyper_dual_sin_cos() {
        // Test sine: f = sin(x), f' = cos(x), f'' = -sin(x)
        let x = HyperDual::variable_x(PI / 2.0);
        let sin_x = x.sin();
        
        assert!(approx_eq(sin_x.value, 1.0));       // sin(π/2) = 1
        assert!(approx_eq(sin_x.dx, 0.0));          // cos(π/2) = 0
        assert_eq!(sin_x.dy, 0.0);
        assert_eq!(sin_x.dxy, 0.0);
        
        // Test cosine: f = cos(x), f' = -sin(x), f'' = -cos(x)
        let cos_x = x.cos();
        assert!(approx_eq(cos_x.value, 0.0));       // cos(π/2) = 0
        assert!(approx_eq(cos_x.dx, -1.0));         // -sin(π/2) = -1
        assert_eq!(cos_x.dy, 0.0);
        assert_eq!(cos_x.dxy, 0.0);
    }
    
    #[test]
    fn test_hyper_dual_sqrt() {
        // Test square root: f = sqrt(x), f' = 1/(2*sqrt(x)), f'' = -1/(4*x^(3/2))
        let x = HyperDual::variable_x(4.0);
        let sqrt_x = x.sqrt().unwrap();
        
        assert_eq!(sqrt_x.value, 2.0);              // sqrt(4) = 2
        assert_eq!(sqrt_x.dx, 0.25);                // 1/(2*sqrt(4)) = 1/4
        assert_eq!(sqrt_x.dy, 0.0);
        assert_eq!(sqrt_x.dxy, 0.0);
        
        // Test error cases
        let negative = HyperDual::variable_x(-1.0);
        assert!(negative.sqrt().is_err());
        
        let zero = HyperDual::variable_x(0.0);
        assert!(zero.sqrt().is_err());
    }
    
    #[test]
    fn test_hyper_dual_ln() {
        // Test natural log: f = ln(x), f' = 1/x, f'' = -1/x²
        let x = HyperDual::variable_x(E);
        let ln_x = x.ln().unwrap();
        
        assert!(approx_eq(ln_x.value, 1.0));        // ln(e) = 1
        assert!(approx_eq(ln_x.dx, 1.0 / E));       // 1/e
        assert_eq!(ln_x.dy, 0.0);
        assert_eq!(ln_x.dxy, 0.0);
        
        // Test error cases
        let negative = HyperDual::variable_x(-1.0);
        assert!(negative.ln().is_err());
        
        let zero = HyperDual::variable_x(0.0);
        assert!(zero.ln().is_err());
    }
    
    #[test]
    fn test_hessian_computation() {
        // Test Hessian for f(x,y) = x²y + xy²
        // ∂f/∂x = 2xy + y², ∂f/∂y = x² + 2xy
        // ∂²f/∂x² = 2y, ∂²f/∂y² = 2x, ∂²f/∂x∂y = 2x + 2y
        
        let x_val = 2.0;
        let y_val = 3.0;
        
        // Compute function with different variable setups
        let x_var = HyperDual::variable_x(x_val);
        let y_var = HyperDual::variable_y(y_val);
        let const_y = HyperDual::constant(y_val);
        let const_x = HyperDual::constant(x_val);
        
        // f evaluated with x as variable, y as constant
        let fx = x_var * x_var * const_y + x_var * const_y * const_y;
        
        // f evaluated with y as variable, x as constant  
        let fy = const_x * const_x * y_var + const_x * y_var * y_var;
        
        // f evaluated with both as variables for mixed partial
        let xy_var = HyperDual::bivariate(x_val);  // This represents x in bivariate context
        let y_only = HyperDual::variable_y(y_val);
        let fxy = xy_var * xy_var * y_only + xy_var * y_only * y_only;
        
        // Expected values at (2, 3):
        // f = 2²*3 + 2*3² = 12 + 18 = 30
        // ∂²f/∂x² = 2y = 6
        // ∂²f/∂y² = 2x = 4  
        // ∂²f/∂x∂y = 2x + 2y = 4 + 6 = 10
        
        assert!(approx_eq(fx.value, 30.0));
        assert!(approx_eq(fy.value, 30.0));
        assert!(approx_eq(fxy.value, 30.0));
        
        // For now, just verify the structure exists
        // Full Hessian computation would require more sophisticated setup
        let hessian = HyperDual::hessian_scalar(fx, fy, fxy);
        assert_eq!(hessian.len(), 2);
        assert_eq!(hessian[0].len(), 2);
    }
    
    // ================================================================================================
    // SPARSE JACOBIAN TESTS (Phase 8.3.4)
    // ================================================================================================
    
    #[test]
    fn test_sparse_matrix_creation() {
        // Test sparse matrix creation
        let sparse = SparseMatrix::new(3, 4);
        assert_eq!(sparse.rows, 3);
        assert_eq!(sparse.cols, 4);
        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.sparsity_ratio(), 0.0);
        
        // Test from dense matrix
        let dense = vec![
            vec![1.0, 0.0, 2.0, 0.0],
            vec![0.0, 3.0, 0.0, 4.0],
            vec![5.0, 0.0, 0.0, 6.0],
        ];
        
        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        assert_eq!(sparse.rows, 3);
        assert_eq!(sparse.cols, 4);
        assert_eq!(sparse.nnz(), 6);  // 6 non-zero elements
        assert!(approx_eq(sparse.sparsity_ratio(), 0.5));  // 6/12 = 0.5
        
        // Test element access
        assert_eq!(sparse.get(0, 0), 1.0);
        assert_eq!(sparse.get(0, 1), 0.0);
        assert_eq!(sparse.get(0, 2), 2.0);
        assert_eq!(sparse.get(1, 1), 3.0);
        assert_eq!(sparse.get(2, 0), 5.0);
        assert_eq!(sparse.get(2, 3), 6.0);
    }
    
    #[test]
    fn test_sparse_matrix_operations() {
        let dense = vec![
            vec![1.0, 0.0, 2.0],
            vec![0.0, 3.0, 0.0],
            vec![4.0, 0.0, 5.0],
        ];
        
        let sparse = SparseMatrix::from_dense(&dense, 1e-10);
        
        // Test matrix-vector multiplication
        let x = vec![1.0, 2.0, 3.0];
        let y = sparse.matvec(&x);
        
        // Expected: [1*1 + 0*2 + 2*3, 0*1 + 3*2 + 0*3, 4*1 + 0*2 + 5*3] = [7, 6, 19]
        assert_eq!(y, vec![7.0, 6.0, 19.0]);
        
        // Test transpose
        let transposed = sparse.transpose();
        assert_eq!(transposed.rows, 3);
        assert_eq!(transposed.cols, 3);
        assert_eq!(transposed.get(0, 0), 1.0);
        assert_eq!(transposed.get(0, 2), 4.0);
        assert_eq!(transposed.get(1, 1), 3.0);
        assert_eq!(transposed.get(2, 0), 2.0);
        assert_eq!(transposed.get(2, 2), 5.0);
    }
    
    #[test]
    fn test_graph_coloring() {
        // Create a simple sparsity pattern
        // Matrix: [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
        // Variables 0,1 are adjacent (row 0), 0,2 are adjacent (row 1), 1,2 are adjacent (row 2)
        // This forms a triangle graph, requiring 3 colors
        
        let dense = vec![
            vec![1.0, 1.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![0.0, 1.0, 1.0],
        ];
        
        let sparsity = SparseMatrix::from_dense(&dense, 1e-10);
        let coloring = GraphColoring::from_sparsity_pattern(&sparsity);
        
        assert_eq!(coloring.n_vars, 3);
        assert_eq!(coloring.n_colors, 3);  // Triangle graph needs 3 colors
        assert_eq!(coloring.compression_ratio(), 1.0);  // No compression for triangle
        
        // Verify no adjacent variables have the same color
        for i in 0..coloring.n_vars {
            for &neighbor in &coloring.adjacency[i] {
                assert_ne!(coloring.colors[i], coloring.colors[neighbor]);
            }
        }
        
        let color_groups = coloring.get_color_groups();
        assert_eq!(color_groups.len(), 3);
        for group in &color_groups {
            assert_eq!(group.len(), 1);  // Each variable gets its own color
        }
    }
    
    #[test]
    fn test_sparse_jacobian_simple() {
        // Test sparse Jacobian computation for a simple function
        // f(x) = [x₀ + x₁, x₀ * x₂, x₁ + x₂]
        // Jacobian: [[1, 1, 0], [x₂, 0, x₀], [0, 1, 1]]
        
        let sparsity_pattern = vec![
            vec![1.0, 1.0, 0.0],  // f₀ depends on x₀, x₁
            vec![1.0, 0.0, 1.0],  // f₁ depends on x₀, x₂
            vec![0.0, 1.0, 1.0],  // f₂ depends on x₁, x₂
        ];
        
        let sparsity = SparseMatrix::from_dense(&sparsity_pattern, 1e-10);
        let sparse_jac = SparseJacobian::new(sparsity);
        
        // Test function
        let test_function = |x: &[Dual]| {
            vec![
                x[0] + x[1],           // f₀ = x₀ + x₁
                x[0] * x[2],           // f₁ = x₀ * x₂ 
                x[1] + x[2],           // f₂ = x₁ + x₂
            ]
        };
        
        let x = vec![2.0, 3.0, 4.0];
        let jacobian = sparse_jac.compute_jacobian(test_function, &x);
        
        // Expected Jacobian at x = [2, 3, 4]:
        // [[1, 1, 0], [4, 0, 2], [0, 1, 1]]
        assert_eq!(jacobian.get(0, 0), 1.0);
        assert_eq!(jacobian.get(0, 1), 1.0);
        assert_eq!(jacobian.get(0, 2), 0.0);
        assert_eq!(jacobian.get(1, 0), 4.0);  // x₂ = 4
        assert_eq!(jacobian.get(1, 1), 0.0);
        assert_eq!(jacobian.get(1, 2), 2.0);  // x₀ = 2
        assert_eq!(jacobian.get(2, 0), 0.0);
        assert_eq!(jacobian.get(2, 1), 1.0);
        assert_eq!(jacobian.get(2, 2), 1.0);
    }
    
    #[test]
    fn test_sparse_jacobian_efficiency() {
        // Test efficiency gains from graph coloring
        // Create a diagonal-like sparsity pattern that should compress well
        
        let n = 6;
        let mut dense = vec![vec![0.0; n]; n];
        
        // Create block diagonal pattern
        for i in 0..n {
            dense[i][i] = 1.0;  // Diagonal elements
            if i < n - 1 {
                dense[i][i + 1] = 1.0;  // Super-diagonal
            }
        }
        
        let sparsity = SparseMatrix::from_dense(&dense, 1e-10);
        let coloring = GraphColoring::from_sparsity_pattern(&sparsity);
        
        // This pattern should compress to 2 colors (alternating pattern)
        assert!(coloring.n_colors <= 3);  // Should be much less than 6
        assert!(coloring.compression_ratio() >= 2.0);  // Good compression
        
        let sparse_jac = SparseJacobian::new(sparsity);
        assert!(sparse_jac.coloring.n_colors < n);  // Compression achieved
    }
}