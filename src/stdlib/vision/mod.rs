//! Computer Vision Module
//!
//! This module provides advanced computer vision algorithms including feature detection,
//! edge detection, and geometric transformations. All data structures use the Foreign
//! object pattern for optimal integration with the VM.
//!
//! Core Computer Vision Algorithms:
//! - Feature Detection: Harris corners, SIFT descriptors, ORB features
//! - Edge Detection: Canny, Sobel, Laplacian, Prewitt operators
//! - Geometric Transforms: Affine, perspective, rotation, scaling
//! - Template Matching: Normalized cross-correlation, SIFT matching
//! - Morphological Operations: Erosion, dilation, opening, closing
//! - Filtering: Gaussian, bilateral, median filtering

pub mod features;
pub mod edges;
pub mod transforms;

// Re-export all public functions
pub use features::*;
pub use edges::*;
pub use transforms::*;

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::Value;
use std::any::Any;
use std::fmt;

/// Keypoint structure for feature detection
#[derive(Debug, Clone, PartialEq)]
pub struct KeyPoint {
    pub x: f32,           // x-coordinate
    pub y: f32,           // y-coordinate
    pub size: f32,        // keypoint scale
    pub angle: f32,       // keypoint orientation
    pub response: f32,    // detector response strength
    pub octave: u8,       // scale octave
    pub class_id: i32,    // keypoint class identifier
}

impl KeyPoint {
    pub fn new(x: f32, y: f32) -> Self {
        Self {
            x,
            y,
            size: 1.0,
            angle: -1.0,
            response: 0.0,
            octave: 0,
            class_id: -1,
        }
    }

    pub fn with_response(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            size: 1.0,
            angle: -1.0,
            response,
            octave: 0,
            class_id: -1,
        }
    }
}

/// Feature set containing keypoints and descriptors
#[derive(Debug, Clone)]
pub struct FeatureSet {
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Vec<Vec<f32>>,  // Each descriptor is a vector of features
    pub feature_type: String,        // "harris", "sift", "orb", etc.
}

impl FeatureSet {
    pub fn new(feature_type: String) -> Self {
        Self {
            keypoints: Vec::new(),
            descriptors: Vec::new(),
            feature_type,
        }
    }

    pub fn add_keypoint(&mut self, keypoint: KeyPoint) {
        self.keypoints.push(keypoint);
    }

    pub fn add_descriptor(&mut self, descriptor: Vec<f32>) {
        self.descriptors.push(descriptor);
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

impl Foreign for FeatureSet {
    fn type_name(&self) -> &'static str {
        "FeatureSet"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "FeatureType" => Ok(Value::String(self.feature_type.clone())),
            "IsEmpty" => Ok(Value::Boolean(self.is_empty())),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for FeatureSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeatureSet[{} {} features]", self.len(), self.feature_type)
    }
}

/// Edge map result from edge detection algorithms
#[derive(Debug, Clone)]
pub struct EdgeMap {
    pub edges: Vec<f32>,     // Edge strength values [0.0, 1.0]
    pub width: usize,
    pub height: usize,
    pub algorithm: String,   // "canny", "sobel", etc.
    pub threshold_low: f32,  // Low threshold used
    pub threshold_high: f32, // High threshold used
}

impl EdgeMap {
    pub fn new(width: usize, height: usize, algorithm: String) -> Self {
        Self {
            edges: vec![0.0; width * height],
            width,
            height,
            algorithm,
            threshold_low: 0.1,
            threshold_high: 0.3,
        }
    }

    pub fn get_pixel(&self, x: usize, y: usize) -> f32 {
        if x < self.width && y < self.height {
            self.edges[y * self.width + x]
        } else {
            0.0
        }
    }

    pub fn set_pixel(&mut self, x: usize, y: usize, value: f32) {
        if x < self.width && y < self.height {
            self.edges[y * self.width + x] = value.clamp(0.0, 1.0);
        }
    }
}

impl Foreign for EdgeMap {
    fn type_name(&self) -> &'static str {
        "EdgeMap"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Width" => Ok(Value::Integer(self.width as i64)),
            "Height" => Ok(Value::Integer(self.height as i64)),
            "Algorithm" => Ok(Value::String(self.algorithm.clone())),
            "GetPixel" => {
                if args.len() != 2 {
                    return Err(ForeignError::ArgumentError { expected: 2, actual: args.len() });
                }
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::TypeError { 
                        expected: "Integer".to_string(), 
                        actual: format!("{:?}", args[0]) 
                    }),
                };
                let y = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::TypeError { 
                        expected: "Integer".to_string(), 
                        actual: format!("{:?}", args[1]) 
                    }),
                };
                Ok(Value::Real(self.get_pixel(x, y) as f64))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for EdgeMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EdgeMap[{}x{} {}]", self.width, self.height, self.algorithm)
    }
}

/// Transformation matrix for geometric operations
#[derive(Debug, Clone, PartialEq)]
pub struct TransformMatrix {
    pub matrix: [[f32; 3]; 3],  // 3x3 transformation matrix
    pub transform_type: String, // "affine", "perspective", "rotation", etc.
}

impl TransformMatrix {
    /// Create identity transformation matrix
    pub fn identity() -> Self {
        Self {
            matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            transform_type: "identity".to_string(),
        }
    }

    /// Create translation matrix
    pub fn translation(dx: f32, dy: f32) -> Self {
        Self {
            matrix: [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0],
            ],
            transform_type: "translation".to_string(),
        }
    }

    /// Create rotation matrix
    pub fn rotation(angle_radians: f32) -> Self {
        let cos_a = angle_radians.cos();
        let sin_a = angle_radians.sin();
        
        Self {
            matrix: [
                [cos_a, -sin_a, 0.0],
                [sin_a,  cos_a, 0.0],
                [0.0,    0.0,   1.0],
            ],
            transform_type: "rotation".to_string(),
        }
    }

    /// Create scaling matrix
    pub fn scaling(sx: f32, sy: f32) -> Self {
        Self {
            matrix: [
                [sx,  0.0, 0.0],
                [0.0, sy,  0.0],
                [0.0, 0.0, 1.0],
            ],
            transform_type: "scaling".to_string(),
        }
    }

    /// Transform a point (x, y) using this matrix
    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let w = self.matrix[2][0] * x + self.matrix[2][1] * y + self.matrix[2][2];
        let new_x = (self.matrix[0][0] * x + self.matrix[0][1] * y + self.matrix[0][2]) / w;
        let new_y = (self.matrix[1][0] * x + self.matrix[1][1] * y + self.matrix[1][2]) / w;
        (new_x, new_y)
    }

    /// Compose this transformation with another
    pub fn compose(&self, other: &TransformMatrix) -> TransformMatrix {
        let mut result = TransformMatrix::identity();
        
        for i in 0..3 {
            for j in 0..3 {
                result.matrix[i][j] = 0.0;
                for k in 0..3 {
                    result.matrix[i][j] += self.matrix[i][k] * other.matrix[k][j];
                }
            }
        }
        
        result.transform_type = format!("{}_composed_{}", self.transform_type, other.transform_type);
        result
    }
}

impl Foreign for TransformMatrix {
    fn type_name(&self) -> &'static str {
        "TransformMatrix"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "TransformType" => Ok(Value::String(self.transform_type.clone())),
            "TransformPoint" => {
                if args.len() != 2 {
                    return Err(ForeignError::ArgumentError { expected: 2, actual: args.len() });
                }
                let x = match &args[0] {
                    Value::Integer(i) => *i as f32,
                    Value::Real(f) => *f as f32,
                    _ => return Err(ForeignError::TypeError { 
                        expected: "Number".to_string(), 
                        actual: format!("{:?}", args[0]) 
                    }),
                };
                let y = match &args[1] {
                    Value::Integer(i) => *i as f32,
                    Value::Real(f) => *f as f32,
                    _ => return Err(ForeignError::TypeError { 
                        expected: "Number".to_string(), 
                        actual: format!("{:?}", args[1]) 
                    }),
                };
                let (new_x, new_y) = self.transform_point(x, y);
                Ok(Value::List(vec![Value::Real(new_x as f64), Value::Real(new_y as f64)]))
            }
            "Compose" => {
                if args.len() != 1 {
                    return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() });
                }
                match &args[0] {
                    Value::LyObj(obj) => {
                        if let Some(other_transform) = obj.downcast_ref::<TransformMatrix>() {
                            let composed = self.compose(other_transform);
                            Ok(Value::LyObj(LyObj::new(Box::new(composed))))
                        } else {
                            Err(ForeignError::TypeError {
                                expected: "TransformMatrix".to_string(),
                                actual: obj.type_name().to_string(),
                            })
                        }
                    }
                    _ => Err(ForeignError::TypeError {
                        expected: "TransformMatrix".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for TransformMatrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TransformMatrix[{}]", self.transform_type)
    }
}

/// Feature match between two keypoints
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureMatch {
    pub query_idx: usize,    // Index in query feature set
    pub train_idx: usize,    // Index in training feature set
    pub distance: f32,       // Distance between descriptors
    pub confidence: f32,     // Match confidence score
}

impl FeatureMatch {
    pub fn new(query_idx: usize, train_idx: usize, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            distance,
            confidence: 1.0 - distance, // Simple confidence calculation
        }
    }
}

/// Set of feature matches
#[derive(Debug, Clone)]
pub struct FeatureMatches {
    pub matches: Vec<FeatureMatch>,
    pub match_type: String,  // "brute_force", "flann", etc.
}

impl FeatureMatches {
    pub fn new(match_type: String) -> Self {
        Self {
            matches: Vec::new(),
            match_type,
        }
    }

    pub fn add_match(&mut self, match_item: FeatureMatch) {
        self.matches.push(match_item);
    }

    pub fn len(&self) -> usize {
        self.matches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.matches.is_empty()
    }

    /// Filter matches by distance threshold
    pub fn filter_by_distance(&mut self, max_distance: f32) {
        self.matches.retain(|m| m.distance <= max_distance);
    }

    /// Sort matches by distance (best first)
    pub fn sort_by_distance(&mut self) {
        self.matches.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
    }
}

impl Foreign for FeatureMatches {
    fn type_name(&self) -> &'static str {
        "FeatureMatches"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "MatchType" => Ok(Value::String(self.match_type.clone())),
            "IsEmpty" => Ok(Value::Boolean(self.is_empty())),
            "FilterByDistance" => {
                if args.len() != 1 {
                    return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() });
                }
                let max_distance = match &args[0] {
                    Value::Integer(i) => *i as f32,
                    Value::Real(f) => *f as f32,
                    _ => return Err(ForeignError::TypeError { 
                        expected: "Number".to_string(), 
                        actual: format!("{:?}", args[0]) 
                    }),
                };
                let mut filtered = self.clone();
                filtered.filter_by_distance(max_distance);
                Ok(Value::LyObj(LyObj::new(Box::new(filtered))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl fmt::Display for FeatureMatches {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FeatureMatches[{} {} matches]", self.len(), self.match_type)
    }
}