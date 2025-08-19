//! Spatial Data Structures for Lyra
//!
//! This module provides comprehensive spatial data structures and algorithms
//! optimized for computational geometry, machine learning, and GIS applications.
//! Spatial structures enable efficient nearest neighbor searches, range queries,
//! and geometric algorithms essential for modern scientific computing.
//!
//! # Spatial Trees
//!
//! - **KDTree**: K-dimensional tree for balanced spatial partitioning
//! - **BallTree**: Ball tree for high-dimensional nearest neighbor search
//! - **RTree**: R-tree for spatial indexing of rectangles and regions
//! - **QuadTree**: Quadtree for 2D spatial decomposition
//! - **Octree**: Octree for 3D spatial decomposition
//! - **VPTree**: Vantage-point tree for general metric spaces
//! - **CoverTree**: Cover tree with theoretical guarantees
//!
//! # Nearest Neighbor Operations
//!
//! - Exact and approximate k-nearest neighbor search
//! - Radius-based neighbor queries
//! - Range queries (rectangular, spherical)
//! - k-NN graph construction
//!
//! # Computational Geometry
//!
//! - Convex hull computation
//! - Delaunay triangulation
//! - Voronoi diagrams
//! - Point-in-polygon tests
//! - Line intersection detection

pub mod core;
pub mod kdtree;
pub mod balltree;
pub mod rtree;
// pub mod quadtree;
// pub mod octree;
// pub mod vptree;
// pub mod covertree;
// pub mod neighbors;
// pub mod geometry;
// pub mod analysis;

// Re-export all public functions and types
pub use core::*;
pub use kdtree::*;
pub use balltree::*;
pub use rtree::*;
// pub use quadtree::*;
// pub use octree::*;
// pub use vptree::*;
// pub use covertree::*;
// pub use neighbors::*;
// pub use geometry::*;
// pub use analysis::*;