//! Numerical Methods for Scientific Computing
//!
//! This module provides comprehensive numerical methods essential for scientific computing,
//! engineering simulations, and mathematical analysis. It implements robust algorithms
//! for solving real-world problems in physics, engineering, and computational sciences.
//!
//! # Module Organization
//!
//! ## Root Finding & Equation Solving
//! - Bisection method for bracketed roots
//! - Newton-Raphson method for smooth functions
//! - Secant method when derivatives unavailable
//! - Brent's hybrid method for robust convergence
//! - Fixed point iteration for iterative solutions
//! - Polynomial root finding algorithms
//! - Nonlinear system solvers
//!
//! ## Numerical Integration & Differentiation
//! - Trapezoidal rule for basic integration
//! - Simpson's rule for higher accuracy
//! - Romberg integration with extrapolation
//! - Gaussian quadrature for optimal points
//! - Monte Carlo integration for high dimensions
//! - Finite difference derivatives
//! - Richardson extrapolation for error reduction
//!
//! ## Mesh Generation & FEM Preparation
//! - Delaunay triangulation meshes
//! - Voronoi diagram generation
//! - Uniform grid construction
//! - Adaptive mesh refinement
//! - Mesh quality assessment
//! - Mesh coarsening/refinement operations
//!
//! ## Finite Element Method Components
//! - Stiffness matrix generation
//! - Mass matrix construction
//! - Load vector assembly
//! - Shape/basis functions
//! - Gauss integration points
//! - Element assembly operations
//! - Boundary condition application
//!
//! ## ODE/PDE Solvers
//! - Runge-Kutta methods (fixed and adaptive)
//! - Implicit methods for stiff equations
//! - Crank-Nicolson for parabolic PDEs
//! - Finite difference PDE solvers
//! - Method of lines for PDEs

pub mod roots;
pub mod integration;
pub mod differentiation;
pub mod mesh;
pub mod fem;
pub mod solvers;

// Re-export all public functions
pub use roots::*;
pub use integration::*;
pub use differentiation::*;
pub use mesh::*;
pub use fem::*;
pub use solvers::*;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate numerical-related stdlib functions
pub fn register_numerical_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();

    // Root finding and equation solving
    f.insert("Bisection".to_string(), roots::bisection as fn(&[Value]) -> VmResult<Value>);
    f.insert("NewtonRaphson".to_string(), roots::newton_raphson as fn(&[Value]) -> VmResult<Value>);
    f.insert("Secant".to_string(), roots::secant as fn(&[Value]) -> VmResult<Value>);
    f.insert("Brent".to_string(), roots::brent as fn(&[Value]) -> VmResult<Value>);
    f.insert("FixedPoint".to_string(), roots::fixed_point as fn(&[Value]) -> VmResult<Value>);

    // Numerical integration
    f.insert("Trapezoidal".to_string(), integration::trapezoidal as fn(&[Value]) -> VmResult<Value>);
    f.insert("Simpson".to_string(), integration::simpson as fn(&[Value]) -> VmResult<Value>);
    f.insert("Romberg".to_string(), integration::romberg as fn(&[Value]) -> VmResult<Value>);
    f.insert("GaussQuadrature".to_string(), integration::gauss_quadrature_fn as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonteCarlo".to_string(), integration::monte_carlo as fn(&[Value]) -> VmResult<Value>);

    // Numerical differentiation
    f.insert("FiniteDifference".to_string(), differentiation::finite_difference as fn(&[Value]) -> VmResult<Value>);
    f.insert("RichardsonExtrapolation".to_string(), differentiation::richardson_extrapolation_fn as fn(&[Value]) -> VmResult<Value>);

    // Mesh generation
    f.insert("DelaunayMesh".to_string(), mesh::delaunay_mesh as fn(&[Value]) -> VmResult<Value>);
    f.insert("VoronoiMesh".to_string(), mesh::voronoi_mesh as fn(&[Value]) -> VmResult<Value>);
    f.insert("UniformMesh".to_string(), mesh::uniform_mesh as fn(&[Value]) -> VmResult<Value>);

    // Finite element components
    f.insert("StiffnessMatrix".to_string(), fem::stiffness_matrix as fn(&[Value]) -> VmResult<Value>);
    f.insert("MassMatrix".to_string(), fem::mass_matrix as fn(&[Value]) -> VmResult<Value>);
    f.insert("LoadVector".to_string(), fem::load_vector as fn(&[Value]) -> VmResult<Value>);

    // ODE/PDE solvers
    f.insert("RungeKutta4".to_string(), solvers::runge_kutta4 as fn(&[Value]) -> VmResult<Value>);
    f.insert("AdaptiveRK".to_string(), solvers::adaptive_rk as fn(&[Value]) -> VmResult<Value>);
    f.insert("CrankNicolson".to_string(), solvers::crank_nicolson as fn(&[Value]) -> VmResult<Value>);

    f
}
