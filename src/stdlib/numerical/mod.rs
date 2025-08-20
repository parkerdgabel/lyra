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