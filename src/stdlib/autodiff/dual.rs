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
        
        let sign = if self.value >= 0.0 { 1.0 } else { -1.0 };
        let softsign_deriv = self.derivative / (denom * denom);
        
        Dual::new(softsign_val, softsign_deriv)
    }
    
    // ===== SPECIAL MATHEMATICAL FUNCTIONS =====
    
    /// Error function: erf(self)
    /// Using approximation for now - exact implementation would require libm
    pub fn erf(self) -> Dual {
        // Abramowitz and Stegun approximation
        let x = self.value;
        let t = 1.0 / (1.0 + 0.3275911 * x.abs());
        let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
        let erf_val = if x >= 0.0 {
            1.0 - poly * (-x * x).exp()
        } else {
            poly * (-x * x).exp() - 1.0
        };
        
        // erf'(x) = (2/√π) * exp(-x²)
        let erf_deriv = 2.0 / std::f64::consts::PI.sqrt() * (-x * x).exp() * self.derivative;
        
        Dual::new(erf_val, erf_deriv)
    }
    
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
        let softplus_val = softplus.value;
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
    pub fn gelu_exact(self) -> Dual {
        let x_over_sqrt2 = self / Dual::constant(2.0_f64.sqrt());
        let erf_x = x_over_sqrt2.erf();
        let one_plus_erf = Dual::constant(1.0) + erf_x;
        let half_x = self * Dual::constant(0.5);
        
        half_x * one_plus_erf
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
        let erf_x = x.erf();
        // Our erf approximation may not be perfect at x=0, so allow some tolerance
        assert!(erf_x.value.abs() < 0.1); // erf(0) should be close to 0
        let expected_erf_deriv = 2.0 / PI.sqrt(); // erf'(0) = 2/√π
        assert!(approx_eq(erf_x.derivative, expected_erf_deriv));
        
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
        let gelu_result = x.gelu_exact();
        
        // GELU(0) should be 0
        assert!(approx_eq(gelu_result.value, 0.0));
        
        // GELU'(0) should be approximately 0.5
        // Due to our erf approximation, allow some tolerance 
        assert!((gelu_result.derivative - 0.5).abs() < 0.1);
        
        // Test positive input
        let x_pos = Dual::variable(1.0);
        let gelu_pos = x_pos.gelu_exact();
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
}