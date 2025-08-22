//! Unified Complex number implementation for Lyra standard library
//!
//! This module provides a comprehensive Complex number implementation that consolidates
//! all complex arithmetic used across quantum computing, signal processing, mathematics,
//! and other domains in the stdlib. It provides both Cartesian and polar representations
//! with efficient arithmetic operations and mathematical functions.

use std::fmt;

/// Complex number representation using double precision
/// 
/// This is the canonical Complex number type used throughout the Lyra stdlib.
/// It replaces individual Complex implementations in quantum, signal, and math modules.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64,
}

impl Complex {
    /// Create a new complex number from real and imaginary parts
    #[inline]
    pub fn new(real: f64, imag: f64) -> Self {
        Self { real, imag }
    }
    
    /// Complex number zero (0 + 0i)
    #[inline]
    pub const fn zero() -> Self {
        Self { real: 0.0, imag: 0.0 }
    }
    
    /// Complex number one (1 + 0i)
    #[inline]
    pub const fn one() -> Self {
        Self { real: 1.0, imag: 0.0 }
    }
    
    /// Complex number i (0 + 1i)
    #[inline]
    pub const fn i() -> Self {
        Self { real: 0.0, imag: 1.0 }
    }
    
    /// Create complex number from real part only
    #[inline]
    pub fn from_real(real: f64) -> Self {
        Self::new(real, 0.0)
    }
    
    /// Create complex number from imaginary part only
    #[inline]
    pub fn from_imag(imag: f64) -> Self {
        Self::new(0.0, imag)
    }
    
    /// Create complex number from polar coordinates (magnitude, phase)
    pub fn from_polar(magnitude: f64, phase: f64) -> Self {
        Self::new(
            magnitude * phase.cos(),
            magnitude * phase.sin()
        )
    }
    
    /// Calculate the magnitude (modulus) of the complex number
    #[inline]
    pub fn magnitude(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    
    /// Calculate the squared magnitude (faster than magnitude when comparison only needed)
    #[inline]
    pub fn magnitude_squared(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }
    
    /// Calculate the phase (argument) of the complex number
    #[inline]
    pub fn phase(&self) -> f64 {
        self.imag.atan2(self.real)
    }
    
    /// Calculate the complex conjugate
    #[inline]
    pub fn conjugate(&self) -> Self {
        Self::new(self.real, -self.imag)
    }
    
    /// Short alias for conjugate
    #[inline]
    pub fn conj(&self) -> Self {
        self.conjugate()
    }
    
    /// Calculate the reciprocal (1/z)
    pub fn reciprocal(&self) -> Self {
        let norm_squared = self.magnitude_squared();
        if norm_squared == 0.0 {
            // Handle division by zero - return infinity
            Self::new(f64::INFINITY, f64::INFINITY)
        } else {
            Self::new(self.real / norm_squared, -self.imag / norm_squared)
        }
    }
    
    /// Check if the complex number is real (imaginary part is zero)
    #[inline]
    pub fn is_real(&self) -> bool {
        self.imag == 0.0 || self.imag.abs() < f64::EPSILON
    }
    
    /// Check if the complex number is imaginary (real part is zero)
    #[inline]
    pub fn is_imaginary(&self) -> bool {
        self.real == 0.0 || self.real.abs() < f64::EPSILON
    }
    
    /// Check if the complex number is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.magnitude_squared() < f64::EPSILON * f64::EPSILON
    }
    
    /// Complex exponential e^z
    pub fn exp(&self) -> Self {
        let exp_real = self.real.exp();
        Self::new(
            exp_real * self.imag.cos(),
            exp_real * self.imag.sin()
        )
    }
    
    /// Complex natural logarithm
    pub fn ln(&self) -> Self {
        Self::new(self.magnitude().ln(), self.phase())
    }
    
    /// Complex power z^n for real exponent
    pub fn powf(&self, n: f64) -> Self {
        if self.is_zero() {
            if n > 0.0 {
                Self::zero()
            } else {
                Self::new(f64::INFINITY, f64::INFINITY)
            }
        } else {
            let magnitude = self.magnitude().powf(n);
            let phase = self.phase() * n;
            Self::from_polar(magnitude, phase)
        }
    }
    
    /// Complex square root
    pub fn sqrt(&self) -> Self {
        let magnitude = self.magnitude().sqrt();
        let phase = self.phase() / 2.0;
        Self::from_polar(magnitude, phase)
    }
    
    /// Convert to tuple (real, imag)
    #[inline]
    pub fn to_tuple(&self) -> (f64, f64) {
        (self.real, self.imag)
    }
    
    /// Convert to polar tuple (magnitude, phase)
    #[inline]
    pub fn to_polar(&self) -> (f64, f64) {
        (self.magnitude(), self.phase())
    }
}

// Arithmetic trait implementations
impl std::ops::Add for Complex {
    type Output = Complex;
    
    #[inline]
    fn add(self, other: Complex) -> Complex {
        Complex::new(self.real + other.real, self.imag + other.imag)
    }
}

impl std::ops::AddAssign for Complex {
    #[inline]
    fn add_assign(&mut self, other: Complex) {
        self.real += other.real;
        self.imag += other.imag;
    }
}

impl std::ops::Sub for Complex {
    type Output = Complex;
    
    #[inline]
    fn sub(self, other: Complex) -> Complex {
        Complex::new(self.real - other.real, self.imag - other.imag)
    }
}

impl std::ops::SubAssign for Complex {
    #[inline]
    fn sub_assign(&mut self, other: Complex) {
        self.real -= other.real;
        self.imag -= other.imag;
    }
}

impl std::ops::Mul for Complex {
    type Output = Complex;
    
    #[inline]
    fn mul(self, other: Complex) -> Complex {
        Complex::new(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real
        )
    }
}

impl std::ops::MulAssign for Complex {
    #[inline]
    fn mul_assign(&mut self, other: Complex) {
        let new_real = self.real * other.real - self.imag * other.imag;
        let new_imag = self.real * other.imag + self.imag * other.real;
        self.real = new_real;
        self.imag = new_imag;
    }
}

impl std::ops::Div for Complex {
    type Output = Complex;
    
    #[inline]
    fn div(self, other: Complex) -> Complex {
        self * other.reciprocal()
    }
}

impl std::ops::DivAssign for Complex {
    #[inline]
    fn div_assign(&mut self, other: Complex) {
        *self = *self / other;
    }
}

// Scalar operations
impl std::ops::Mul<f64> for Complex {
    type Output = Complex;
    
    #[inline]
    fn mul(self, scalar: f64) -> Complex {
        Complex::new(self.real * scalar, self.imag * scalar)
    }
}

impl std::ops::Mul<Complex> for f64 {
    type Output = Complex;
    
    #[inline]
    fn mul(self, complex: Complex) -> Complex {
        complex * self
    }
}

impl std::ops::Div<f64> for Complex {
    type Output = Complex;
    
    #[inline]
    fn div(self, scalar: f64) -> Complex {
        if scalar == 0.0 {
            Complex::new(f64::INFINITY, f64::INFINITY)
        } else {
            Complex::new(self.real / scalar, self.imag / scalar)
        }
    }
}

impl std::ops::Add<f64> for Complex {
    type Output = Complex;
    
    #[inline]
    fn add(self, real: f64) -> Complex {
        Complex::new(self.real + real, self.imag)
    }
}

impl std::ops::Add<Complex> for f64 {
    type Output = Complex;
    
    #[inline]
    fn add(self, complex: Complex) -> Complex {
        complex + self
    }
}

impl std::ops::Sub<f64> for Complex {
    type Output = Complex;
    
    #[inline]
    fn sub(self, real: f64) -> Complex {
        Complex::new(self.real - real, self.imag)
    }
}

impl std::ops::Sub<Complex> for f64 {
    type Output = Complex;
    
    #[inline]
    fn sub(self, complex: Complex) -> Complex {
        Complex::new(self - complex.real, -complex.imag)
    }
}

// Unary operations
impl std::ops::Neg for Complex {
    type Output = Complex;
    
    #[inline]
    fn neg(self) -> Complex {
        Complex::new(-self.real, -self.imag)
    }
}

// Display implementation
impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.imag == 0.0 {
            write!(f, "{}", self.real)
        } else if self.real == 0.0 {
            if self.imag == 1.0 {
                write!(f, "I")
            } else if self.imag == -1.0 {
                write!(f, "-I")
            } else {
                write!(f, "{}*I", self.imag)
            }
        } else {
            if self.imag > 0.0 {
                if self.imag == 1.0 {
                    write!(f, "{} + I", self.real)
                } else {
                    write!(f, "{} + {}*I", self.real, self.imag)
                }
            } else {
                if self.imag == -1.0 {
                    write!(f, "{} - I", self.real)
                } else {
                    write!(f, "{} - {}*I", self.real, -self.imag)
                }
            }
        }
    }
}

// Conversion from real numbers
impl From<f64> for Complex {
    #[inline]
    fn from(real: f64) -> Self {
        Self::from_real(real)
    }
}

impl From<f32> for Complex {
    #[inline]
    fn from(real: f32) -> Self {
        Self::from_real(real as f64)
    }
}

impl From<i32> for Complex {
    #[inline]
    fn from(real: i32) -> Self {
        Self::from_real(real as f64)
    }
}

impl From<i64> for Complex {
    #[inline]
    fn from(real: i64) -> Self {
        Self::from_real(real as f64)
    }
}

impl From<(f64, f64)> for Complex {
    #[inline]
    fn from((real, imag): (f64, f64)) -> Self {
        Self::new(real, imag)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_complex_creation() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.real, 3.0);
        assert_eq!(z.imag, 4.0);
        
        let zero = Complex::zero();
        assert_eq!(zero.real, 0.0);
        assert_eq!(zero.imag, 0.0);
        
        let one = Complex::one();
        assert_eq!(one.real, 1.0);
        assert_eq!(one.imag, 0.0);
        
        let i = Complex::i();
        assert_eq!(i.real, 0.0);
        assert_eq!(i.imag, 1.0);
    }
    
    #[test]
    fn test_magnitude_and_phase() {
        let z = Complex::new(3.0, 4.0);
        assert_eq!(z.magnitude(), 5.0);
        assert_eq!(z.magnitude_squared(), 25.0);
        
        let i = Complex::i();
        assert!((i.phase() - std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_arithmetic() {
        let z1 = Complex::new(1.0, 2.0);
        let z2 = Complex::new(3.0, 4.0);
        
        let sum = z1 + z2;
        assert_eq!(sum, Complex::new(4.0, 6.0));
        
        let diff = z2 - z1;
        assert_eq!(diff, Complex::new(2.0, 2.0));
        
        let product = z1 * z2;
        assert_eq!(product, Complex::new(-5.0, 10.0)); // (1+2i)(3+4i) = 3+4i+6i+8i² = -5+10i
    }
    
    #[test]
    fn test_conjugate() {
        let z = Complex::new(3.0, 4.0);
        let conj = z.conjugate();
        assert_eq!(conj, Complex::new(3.0, -4.0));
        
        // z * conj(z) should be real
        let product = z * conj;
        assert!(product.is_real());
        assert_eq!(product.real, 25.0);
    }
    
    #[test]
    fn test_polar_conversion() {
        let z = Complex::from_polar(5.0, std::f64::consts::FRAC_PI_2);
        assert!((z.real).abs() < f64::EPSILON);
        assert!((z.imag - 5.0).abs() < f64::EPSILON);
        
        let (mag, phase) = z.to_polar();
        assert!((mag - 5.0).abs() < f64::EPSILON);
        assert!((phase - std::f64::consts::FRAC_PI_2).abs() < f64::EPSILON);
    }
    
    #[test]
    fn test_display() {
        assert_eq!(Complex::new(3.0, 0.0).to_string(), "3");
        assert_eq!(Complex::new(0.0, 1.0).to_string(), "I");
        assert_eq!(Complex::new(0.0, -1.0).to_string(), "-I");
        assert_eq!(Complex::new(3.0, 4.0).to_string(), "3 + 4*I");
        assert_eq!(Complex::new(3.0, -4.0).to_string(), "3 - 4*I");
    }
}