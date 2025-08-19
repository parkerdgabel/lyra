//! Clustering Algorithms Module
//!
//! This module provides comprehensive clustering capabilities following the
//! "Take Algorithms for Granted" principle. Includes K-means, hierarchical,
//! density-based, and advanced clustering algorithms with Foreign object support.

pub mod core;
pub mod kmeans;
pub mod hierarchical;
pub mod density;
pub mod spectral;
pub mod mixture;
pub mod advanced;

// Re-export all public functions
pub use core::*;
pub use kmeans::*;
pub use hierarchical::*;
pub use density::*;
pub use spectral::*;
pub use mixture::*;
pub use advanced::*;