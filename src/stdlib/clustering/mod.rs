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

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate clustering-related stdlib functions
pub fn register_clustering_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();
    // Core clustering infrastructure
    f.insert("ClusterData".to_string(), core::cluster_data as fn(&[Value]) -> VmResult<Value>);
    f.insert("DistanceMatrix".to_string(), core::distance_matrix as fn(&[Value]) -> VmResult<Value>);
    // K-means variants
    f.insert("KMeans".to_string(), kmeans::kmeans as fn(&[Value]) -> VmResult<Value>);
    f.insert("MiniBatchKMeans".to_string(), kmeans::mini_batch_kmeans as fn(&[Value]) -> VmResult<Value>);
    // Future: hierarchical/density/spectral/mixture registrations when ready
    f
}
