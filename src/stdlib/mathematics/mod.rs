//! Mathematics Module
//!
//! This module provides comprehensive mathematical capabilities including:
//! - Basic arithmetic and trigonometric functions
//! - Symbolic calculus (differentiation, integration)
//! - Special functions (gamma, elliptic, error functions)
//! - Differential equations solving
//! - Interpolation and approximation
//! - Linear algebra operations
//! - Optimization algorithms
//! - Signal processing

pub mod basic;
pub mod calculus;
pub mod special;
pub mod differential;
pub mod interpolation;
pub mod linear_algebra;
pub mod optimization;
pub mod signal;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Register all mathematics functions with the standard library
pub fn register_mathematics_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();

    // Basic mathematical functions (from basic.rs)
    functions.insert("Plus".to_string(), basic::plus as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Times".to_string(), basic::times as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Divide".to_string(), basic::divide as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Power".to_string(), basic::power as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Minus".to_string(), basic::minus as fn(&[Value]) -> VmResult<Value>);
    
    // Trigonometric functions
    functions.insert("Sin".to_string(), basic::sin as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Cos".to_string(), basic::cos as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Tan".to_string(), basic::tan as fn(&[Value]) -> VmResult<Value>);
    
    // Exponential and logarithmic functions
    functions.insert("Exp".to_string(), basic::exp as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Log".to_string(), basic::log as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Sqrt".to_string(), basic::sqrt as fn(&[Value]) -> VmResult<Value>);
    
    // Other mathematical functions
    functions.insert("Abs".to_string(), basic::abs as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Sign".to_string(), basic::sign as fn(&[Value]) -> VmResult<Value>);
    
    // Test functions
    functions.insert("TestHold".to_string(), basic::test_hold as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TestHoldMultiple".to_string(), basic::test_hold_multiple as fn(&[Value]) -> VmResult<Value>);

    // Calculus functions (from calculus.rs)
    functions.insert("D".to_string(), calculus::d as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Integrate".to_string(), calculus::integrate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("IntegrateDefinite".to_string(), calculus::integrate_definite as fn(&[Value]) -> VmResult<Value>);

    // Special functions (from special.rs)
    functions.insert("Pi".to_string(), special::pi_constant as fn(&[Value]) -> VmResult<Value>);
    functions.insert("E".to_string(), special::e_constant as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EulerGamma".to_string(), special::euler_gamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GoldenRatio".to_string(), special::golden_ratio as fn(&[Value]) -> VmResult<Value>);
    
    // Gamma functions
    functions.insert("Gamma".to_string(), special::gamma_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LogGamma".to_string(), special::log_gamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Digamma".to_string(), special::digamma as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Polygamma".to_string(), special::polygamma as fn(&[Value]) -> VmResult<Value>);
    
    // Error functions
    functions.insert("Erf".to_string(), special::erf_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Erfc".to_string(), special::erfc_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("InverseErf".to_string(), special::inverse_erf as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FresnelC".to_string(), special::fresnel_c as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FresnelS".to_string(), special::fresnel_s as fn(&[Value]) -> VmResult<Value>);
    
    // Elliptic functions
    functions.insert("EllipticK".to_string(), special::elliptic_k as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EllipticE".to_string(), special::elliptic_e as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EllipticTheta".to_string(), special::elliptic_theta as fn(&[Value]) -> VmResult<Value>);
    
    // Hypergeometric functions
    functions.insert("Hypergeometric0F1".to_string(), special::hypergeometric_0f1 as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Hypergeometric1F1".to_string(), special::hypergeometric_1f1 as fn(&[Value]) -> VmResult<Value>);
    
    // Orthogonal polynomials
    functions.insert("ChebyshevT".to_string(), special::chebyshev_t as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ChebyshevU".to_string(), special::chebyshev_u as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GegenbauerC".to_string(), special::gegenbauer_c as fn(&[Value]) -> VmResult<Value>);

    // Linear Algebra functions (from linear_algebra.rs)
    functions.insert("SVD".to_string(), linear_algebra::svd as fn(&[Value]) -> VmResult<Value>);
    functions.insert("QRDecomposition".to_string(), linear_algebra::qr_decomposition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LUDecomposition".to_string(), linear_algebra::lu_decomposition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CholeskyDecomposition".to_string(), linear_algebra::cholesky_decomposition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("EigenDecomposition".to_string(), linear_algebra::eigen_decomposition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SchurDecomposition".to_string(), linear_algebra::schur_decomposition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LinearSolve".to_string(), linear_algebra::linear_solve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LeastSquares".to_string(), linear_algebra::least_squares as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PseudoInverse".to_string(), linear_algebra::pseudo_inverse as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MatrixPower".to_string(), linear_algebra::matrix_power as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MatrixFunction".to_string(), linear_algebra::matrix_function as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Tr".to_string(), linear_algebra::matrix_trace as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MatrixRank".to_string(), linear_algebra::matrix_rank as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MatrixCondition".to_string(), linear_algebra::matrix_condition as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MatrixNorm".to_string(), linear_algebra::matrix_norm as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Det".to_string(), linear_algebra::determinant as fn(&[Value]) -> VmResult<Value>);

    // Optimization functions (from optimization.rs)
    functions.insert("FindRoot".to_string(), optimization::find_root as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NewtonMethod".to_string(), optimization::newton_method_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BisectionMethod".to_string(), optimization::bisection_method_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SecantMethod".to_string(), optimization::secant_method_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Minimize".to_string(), optimization::minimize as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Maximize".to_string(), optimization::maximize as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NIntegrate".to_string(), optimization::n_integrate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GaussianQuadrature".to_string(), optimization::gaussian_quadrature as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MonteCarloIntegration".to_string(), optimization::monte_carlo_integration as fn(&[Value]) -> VmResult<Value>);

    // Signal Processing functions (from signal.rs)
    functions.insert("FFT".to_string(), signal::fft as fn(&[Value]) -> VmResult<Value>);
    functions.insert("InverseFourierTransform".to_string(), signal::ifft as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DCT".to_string(), signal::dct as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PowerSpectrum".to_string(), signal::power_spectrum as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Periodogram".to_string(), signal::periodogram as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Spectrogram".to_string(), signal::spectrogram as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PSDEstimate".to_string(), signal::psd_estimate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HammingWindow".to_string(), signal::hamming_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HanningWindow".to_string(), signal::hanning_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BlackmanWindow".to_string(), signal::blackman_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ApplyWindow".to_string(), signal::apply_window as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Convolve".to_string(), signal::convolve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("CrossCorrelation".to_string(), signal::cross_correlation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AutoCorrelation".to_string(), signal::auto_correlation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LowPassFilter".to_string(), signal::low_pass_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HighPassFilter".to_string(), signal::high_pass_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("MedianFilter".to_string(), signal::median_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HilbertTransform".to_string(), signal::hilbert_transform as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ZeroPadding".to_string(), signal::zero_padding as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PhaseUnwrap".to_string(), signal::phase_unwrap as fn(&[Value]) -> VmResult<Value>);

    // Differential Equations functions (from differential.rs)
    functions.insert("NDSolve".to_string(), differential::nd_solve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DSolve".to_string(), differential::d_solve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DEigensystem".to_string(), differential::d_eigensystem as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PDSolve".to_string(), differential::pd_solve as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LaplacianFilter".to_string(), differential::laplacian_filter as fn(&[Value]) -> VmResult<Value>);
    functions.insert("WaveEquation".to_string(), differential::wave_equation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("VectorCalculus".to_string(), differential::vector_calculus as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Gradient".to_string(), differential::gradient as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Divergence".to_string(), differential::divergence as fn(&[Value]) -> VmResult<Value>);
    functions.insert("Curl".to_string(), differential::curl as fn(&[Value]) -> VmResult<Value>);
    functions.insert("RungeKutta".to_string(), differential::runge_kutta as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdamsBashforth".to_string(), differential::adams_bashforth as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BDF".to_string(), differential::bdf as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BesselJ".to_string(), differential::bessel_j as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HermiteH".to_string(), differential::hermite_h as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LegendreP".to_string(), differential::legendre_p as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LaplaceTransform".to_string(), differential::laplace_transform as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ZTransform".to_string(), differential::z_transform as fn(&[Value]) -> VmResult<Value>);
    functions.insert("HankelTransform".to_string(), differential::hankel_transform as fn(&[Value]) -> VmResult<Value>);

    // Interpolation functions (from interpolation.rs)
    functions.insert("Interpolation".to_string(), interpolation::interpolation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SplineInterpolation".to_string(), interpolation::spline_interpolation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("PolynomialInterpolation".to_string(), interpolation::polynomial_interpolation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NIntegrateAdvanced".to_string(), interpolation::n_integrate_advanced as fn(&[Value]) -> VmResult<Value>);
    functions.insert("GaussLegendre".to_string(), interpolation::gauss_legendre as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdaptiveQuadrature".to_string(), interpolation::adaptive_quadrature_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FindRootAdvanced".to_string(), interpolation::find_root_advanced as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BrentMethod".to_string(), interpolation::brent_method_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NewtonRaphson".to_string(), interpolation::newton_raphson_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NonlinearFit".to_string(), interpolation::nonlinear_fit as fn(&[Value]) -> VmResult<Value>);
    functions.insert("LeastSquaresFit".to_string(), interpolation::least_squares_fit as fn(&[Value]) -> VmResult<Value>);
    functions.insert("SplineFit".to_string(), interpolation::spline_fit as fn(&[Value]) -> VmResult<Value>);
    functions.insert("NDerivative".to_string(), interpolation::n_derivative as fn(&[Value]) -> VmResult<Value>);
    functions.insert("FiniteDifference".to_string(), interpolation::finite_difference_wrapper as fn(&[Value]) -> VmResult<Value>);
    functions.insert("RichardsonExtrapolation".to_string(), interpolation::richardson_extrapolation as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ErrorEstimate".to_string(), interpolation::error_estimate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("AdaptiveMethod".to_string(), interpolation::adaptive_method as fn(&[Value]) -> VmResult<Value>);

    functions
}

// Re-export public functions (avoiding conflicts)
// Note: Only re-exporting functions that actually exist
pub use basic::{
    divide, power, sqrt, sin, cos, tan, exp, log, random_real
};
pub use calculus::*;
pub use special::*;
pub use differential::*;
pub use interpolation::*;
pub use linear_algebra::*;
pub use optimization::*;
pub use signal::*;