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

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate image-related stdlib functions
pub fn register_image_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();
    // Core infrastructure
    f.insert("ImageImport".to_string(), core::image_import as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageExport".to_string(), core::image_export as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageInfo".to_string(), core::image_info as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageResize".to_string(), core::image_resize as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageHistogram".to_string(), core::image_histogram as fn(&[Value]) -> VmResult<Value>);
    // Filtering & enhancement
    f.insert("GaussianFilter".to_string(), filters::gaussian_filter as fn(&[Value]) -> VmResult<Value>);
    f.insert("MedianFilter".to_string(), filters::median_filter as fn(&[Value]) -> VmResult<Value>);
    f.insert("SobelFilter".to_string(), filters::sobel_filter as fn(&[Value]) -> VmResult<Value>);
    f.insert("CannyEdgeDetection".to_string(), filters::canny_edge_detection as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageRotate".to_string(), filters::image_rotate as fn(&[Value]) -> VmResult<Value>);
    // Morphological operations
    f.insert("Erosion".to_string(), morphology::erosion as fn(&[Value]) -> VmResult<Value>);
    f.insert("Dilation".to_string(), morphology::dilation as fn(&[Value]) -> VmResult<Value>);
    f.insert("Opening".to_string(), morphology::opening as fn(&[Value]) -> VmResult<Value>);
    f.insert("Closing".to_string(), morphology::closing as fn(&[Value]) -> VmResult<Value>);
    // Advanced analysis
    f.insert("AffineTransform".to_string(), analysis::affine_transform as fn(&[Value]) -> VmResult<Value>);
    f.insert("PerspectiveTransform".to_string(), analysis::perspective_transform as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContourDetection".to_string(), analysis::contour_detection as fn(&[Value]) -> VmResult<Value>);
    f.insert("FeatureDetection".to_string(), analysis::feature_detection as fn(&[Value]) -> VmResult<Value>);
    f.insert("TemplateMatching".to_string(), analysis::template_matching as fn(&[Value]) -> VmResult<Value>);
    f.insert("ImageSegmentation".to_string(), analysis::image_segmentation as fn(&[Value]) -> VmResult<Value>);
    f
}
