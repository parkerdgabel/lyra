//! Image Processing & Computer Vision Module
//!
//! This module provides comprehensive image processing and computer vision capabilities
//! following the "Take Algorithms for Granted" principle. Includes image I/O, filtering,
//! morphological operations, geometric transformations, and feature detection.

pub mod core;
pub mod filters;
pub mod morphology;
pub mod transform;
pub mod analysis;
pub mod utils;

// Re-export all public functions
pub use core::*;
pub use filters::*;
pub use morphology::*;
// pub use transform::*;  // TODO: Implement geometric transforms
pub use analysis::*;