//! Sparse Matrix Operations for Lyra
//!
//! This module provides comprehensive sparse matrix support with multiple storage formats
//! optimized for different use cases. Sparse matrices are essential for scientific computing,
//! machine learning, and graph algorithms where most matrix elements are zero.
//!
//! # Storage Formats
//!
//! - **CSR (Compressed Sparse Row)**: Efficient for row-wise operations and matrix-vector products
//! - **CSC (Compressed Sparse Column)**: Efficient for column-wise operations and transposes
//! - **COO (Coordinate)**: Simple triplet format, good for matrix construction
//! - **DOK (Dictionary of Keys)**: Fast random access and incremental building
//! - **LIL (List of Lists)**: Efficient row-based modifications
//!
//! # Core Operations
//!
//! - Matrix arithmetic (addition, multiplication, transpose)
//! - Linear algebra (solve, decompositions, eigenvalues)
//! - Graph operations (Laplacian, PageRank, BFS)
//! - Format conversions and utilities

pub mod core;
pub mod csr;
pub mod csc;
pub mod coo;
pub mod dok;
pub mod lil;
pub mod operations;
pub mod solvers;
pub mod graph;

// Re-export all public functions and types
pub use core::*;
pub use csr::*;
pub use csc::*;
pub use coo::*;
pub use dok::*;
pub use lil::*;
pub use operations::*;
pub use solvers::*;
pub use graph::*;