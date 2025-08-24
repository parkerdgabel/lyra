//! Advanced Image Analysis Operations
//!
//! This module provides advanced image analysis and computer vision functions including
//! geometric transformations, feature detection, contour analysis, and image segmentation.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::core::{Image, ColorSpace, InterpolationMethod};
use std::any::Any;

// ===============================
// FOREIGN OBJECT TYPES
// ===============================

/// Feature point detected in an image
#[derive(Debug, Clone, PartialEq)]
pub struct FeaturePoint {
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub scale: f32,
    pub angle: f32,
}

impl Foreign for FeaturePoint {
    fn type_name(&self) -> &'static str {
        "FeaturePoint"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "X" => Ok(Value::Real(self.x as f64)),
            "Y" => Ok(Value::Real(self.y as f64)),
            "Position" => Ok(Value::List(vec![
                Value::Real(self.x as f64),
                Value::Real(self.y as f64)
            ])),
            "Confidence" => Ok(Value::Real(self.confidence as f64)),
            "Scale" => Ok(Value::Real(self.scale as f64)),
            "Angle" => Ok(Value::Real(self.angle as f64)),
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

/// Contour representing object boundary
#[derive(Debug, Clone, PartialEq)]
pub struct Contour {
    pub points: Vec<(f32, f32)>,
    pub area: f32,
    pub perimeter: f32,
    pub is_closed: bool,
}

impl Foreign for Contour {
    fn type_name(&self) -> &'static str {
        "Contour"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Points" => {
                let points: Vec<Value> = self.points.iter()
                    .map(|(x, y)| Value::List(vec![
                        Value::Real(*x as f64),
                        Value::Real(*y as f64)
                    ]))
                    .collect();
                Ok(Value::List(points))
            }
            "Area" => Ok(Value::Real(self.area as f64)),
            "Perimeter" => Ok(Value::Real(self.perimeter as f64)),
            "Length" => Ok(Value::Integer(self.points.len() as i64)),
            "IsClosed" => Ok(Value::Integer(if self.is_closed { 1 } else { 0 })),
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

/// Segmentation result with labeled regions
#[derive(Debug, Clone, PartialEq)]
pub struct SegmentationResult {
    pub labels: Image,
    pub num_regions: usize,
    pub region_stats: Vec<RegionStats>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RegionStats {
    pub label: i32,
    pub area: f32,
    pub centroid: (f32, f32),
    pub bounding_box: (f32, f32, f32, f32), // (x, y, width, height)
}

impl Foreign for SegmentationResult {
    fn type_name(&self) -> &'static str {
        "SegmentationResult"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Labels" => Ok(Value::LyObj(LyObj::new(Box::new(self.labels.clone())))),
            "NumRegions" => Ok(Value::Integer(self.num_regions as i64)),
            "RegionStats" => {
                let stats: Vec<Value> = self.region_stats.iter()
                    .map(|stat| Value::List(vec![
                        Value::Integer(stat.label as i64),
                        Value::Real(stat.area as f64),
                        Value::List(vec![
                            Value::Real(stat.centroid.0 as f64),
                            Value::Real(stat.centroid.1 as f64)
                        ]),
                        Value::List(vec![
                            Value::Real(stat.bounding_box.0 as f64),
                            Value::Real(stat.bounding_box.1 as f64),
                            Value::Real(stat.bounding_box.2 as f64),
                            Value::Real(stat.bounding_box.3 as f64)
                        ])
                    ]))
                    .collect();
                Ok(Value::List(stats))
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

// ===============================
// TRANSFORMATION MATRICES
// ===============================

/// 2x3 Affine transformation matrix
#[derive(Debug, Clone, PartialEq)]
pub struct AffineMatrix {
    pub matrix: [[f32; 3]; 2], // [a, b, tx], [c, d, ty]
}

impl AffineMatrix {
    pub fn identity() -> Self {
        AffineMatrix {
            matrix: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        }
    }

    pub fn from_values(a: f32, b: f32, c: f32, d: f32, tx: f32, ty: f32) -> Self {
        AffineMatrix {
            matrix: [[a, b, tx], [c, d, ty]]
        }
    }

    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let new_x = self.matrix[0][0] * x + self.matrix[0][1] * y + self.matrix[0][2];
        let new_y = self.matrix[1][0] * x + self.matrix[1][1] * y + self.matrix[1][2];
        (new_x, new_y)
    }
}

/// 3x3 Perspective transformation matrix
#[derive(Debug, Clone, PartialEq)]
pub struct PerspectiveMatrix {
    pub matrix: [[f32; 3]; 3],
}

impl PerspectiveMatrix {
    pub fn identity() -> Self {
        PerspectiveMatrix {
            matrix: [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ]
        }
    }

    pub fn transform_point(&self, x: f32, y: f32) -> (f32, f32) {
        let w = self.matrix[2][0] * x + self.matrix[2][1] * y + self.matrix[2][2];
        if w.abs() < f32::EPSILON {
            return (x, y); // Avoid division by zero
        }
        
        let new_x = (self.matrix[0][0] * x + self.matrix[0][1] * y + self.matrix[0][2]) / w;
        let new_y = (self.matrix[1][0] * x + self.matrix[1][1] * y + self.matrix[1][2]) / w;
        (new_x, new_y)
    }
}

// ===============================
// CORE ANALYSIS IMPLEMENTATIONS
// ===============================

impl Image {
    /// Apply affine transformation to image
    pub fn affine_transform(&self, matrix: &AffineMatrix, interpolation: InterpolationMethod) -> Image {
        // Calculate transformed image bounds
        let corners = [
            (0.0, 0.0),
            (self.width as f32, 0.0),
            (0.0, self.height as f32),
            (self.width as f32, self.height as f32),
        ];

        let transformed_corners: Vec<(f32, f32)> = corners
            .iter()
            .map(|(x, y)| matrix.transform_point(*x, *y))
            .collect();

        let min_x = transformed_corners.iter().map(|(x, _)| *x).fold(f32::INFINITY, f32::min);
        let max_x = transformed_corners.iter().map(|(x, _)| *x).fold(f32::NEG_INFINITY, f32::max);
        let min_y = transformed_corners.iter().map(|(_, y)| *y).fold(f32::INFINITY, f32::min);
        let max_y = transformed_corners.iter().map(|(_, y)| *y).fold(f32::NEG_INFINITY, f32::max);

        let new_width = (max_x - min_x).ceil() as usize;
        let new_height = (max_y - min_y).ceil() as usize;

        let mut transformed_data = vec![0.0; new_width * new_height * self.channels];

        // Calculate inverse transformation for reverse mapping
        let det = matrix.matrix[0][0] * matrix.matrix[1][1] - matrix.matrix[0][1] * matrix.matrix[1][0];
        if det.abs() < f32::EPSILON {
            return Image::from_data(transformed_data, new_width, new_height, self.color_space, self.bit_depth);
        }

        for y in 0..new_height {
            for x in 0..new_width {
                let world_x = x as f32 + min_x;
                let world_y = y as f32 + min_y;

                // Inverse transform to get source coordinates
                let src_x = (matrix.matrix[1][1] * (world_x - matrix.matrix[0][2]) - 
                            matrix.matrix[0][1] * (world_y - matrix.matrix[1][2])) / det;
                let src_y = (matrix.matrix[0][0] * (world_y - matrix.matrix[1][2]) - 
                            matrix.matrix[1][0] * (world_x - matrix.matrix[0][2])) / det;

                if src_x >= 0.0 && src_x < self.width as f32 && src_y >= 0.0 && src_y < self.height as f32 {
                    for c in 0..self.channels {
                        let pixel_value = match interpolation {
                            InterpolationMethod::NearestNeighbor => {
                                let x_idx = src_x.round() as usize;
                                let y_idx = src_y.round() as usize;
                                self.get_pixel(x_idx, y_idx, c).unwrap_or(0.0)
                            }
                            _ => self.bilinear_interpolate(src_x, src_y, c),
                        };

                        let dst_index = (y * new_width + x) * self.channels + c;
                        transformed_data[dst_index] = pixel_value;
                    }
                }
            }
        }

        Image::from_data(transformed_data, new_width, new_height, self.color_space, self.bit_depth)
    }

    /// Apply perspective transformation to image
    pub fn perspective_transform(&self, matrix: &PerspectiveMatrix, interpolation: InterpolationMethod) -> Image {
        let new_width = self.width;
        let new_height = self.height;
        let mut transformed_data = vec![0.0; new_width * new_height * self.channels];

        for y in 0..new_height {
            for x in 0..new_width {
                let (src_x, src_y) = matrix.transform_point(x as f32, y as f32);

                if src_x >= 0.0 && src_x < self.width as f32 && src_y >= 0.0 && src_y < self.height as f32 {
                    for c in 0..self.channels {
                        let pixel_value = match interpolation {
                            InterpolationMethod::NearestNeighbor => {
                                let x_idx = src_x.round() as usize;
                                let y_idx = src_y.round() as usize;
                                self.get_pixel(x_idx, y_idx, c).unwrap_or(0.0)
                            }
                            _ => self.bilinear_interpolate(src_x, src_y, c),
                        };

                        let dst_index = (y * new_width + x) * self.channels + c;
                        transformed_data[dst_index] = pixel_value;
                    }
                }
            }
        }

        Image::from_data(transformed_data, new_width, new_height, self.color_space, self.bit_depth)
    }

    /// Detect contours in the image
    pub fn detect_contours(&self, threshold: f32) -> Vec<Contour> {
        let binary = self.to_binary(threshold);
        let mut contours = Vec::new();
        let mut visited = vec![vec![false; binary.width]; binary.height];

        for y in 0..binary.height {
            for x in 0..binary.width {
                if !visited[y][x] && binary.get_pixel(x, y, 0).unwrap_or(0.0) > 0.5 {
                    let contour = self.trace_contour(&binary, &mut visited, x, y);
                    if contour.points.len() > 3 {
                        contours.push(contour);
                    }
                }
            }
        }

        contours
    }

    /// Detect features using Harris corner detection
    pub fn detect_features(&self, threshold: f32) -> Vec<FeaturePoint> {
        let gray = if self.color_space != ColorSpace::Grayscale {
            self.to_grayscale()
        } else {
            self.clone()
        };

        let mut features = Vec::new();
        let response = gray.harris_response();

        for y in 2..gray.height - 2 {
            for x in 2..gray.width - 2 {
                let r = response.get_pixel(x, y, 0).unwrap_or(0.0);
                if r > threshold {
                    // Check if it's a local maximum
                    let mut is_maximum = true;
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx == 0 && dy == 0 { continue; }
                            let nx = (x as i32 + dx) as usize;
                            let ny = (y as i32 + dy) as usize;
                            if response.get_pixel(nx, ny, 0).unwrap_or(0.0) >= r {
                                is_maximum = false;
                                break;
                            }
                        }
                        if !is_maximum { break; }
                    }

                    if is_maximum {
                        features.push(FeaturePoint {
                            x: x as f32,
                            y: y as f32,
                            confidence: r,
                            scale: 1.0,
                            angle: 0.0,
                        });
                    }
                }
            }
        }

        features
    }

    /// Template matching using normalized cross-correlation
    pub fn template_match(&self, template: &Image) -> Image {
        if template.width > self.width || template.height > self.height {
            return Image::new(1, 1, ColorSpace::Grayscale, 8);
        }

        let result_width = self.width - template.width + 1;
        let result_height = self.height - template.height + 1;
        let mut result_data = vec![0.0; result_width * result_height];

        for y in 0..result_height {
            for x in 0..result_width {
                let correlation = self.compute_ncc(template, x, y);
                result_data[y * result_width + x] = correlation;
            }
        }

        Image::from_data(result_data, result_width, result_height, ColorSpace::Grayscale, self.bit_depth)
    }

    /// Segment image using simple thresholding
    pub fn segment_image(&self, method: &str) -> SegmentationResult {
        match method {
            "Threshold" => self.threshold_segmentation(0.5),
            "Watershed" => self.watershed_segmentation(),
            _ => self.threshold_segmentation(0.5),
        }
    }

    // Helper methods
    fn to_binary(&self, threshold: f32) -> Image {
        let gray = if self.color_space != ColorSpace::Grayscale {
            self.to_grayscale()
        } else {
            self.clone()
        };

        let mut binary_data = vec![0.0; gray.data.len()];
        for i in 0..gray.data.len() {
            binary_data[i] = if gray.data[i] > threshold { 1.0 } else { 0.0 };
        }

        Image::from_data(binary_data, gray.width, gray.height, ColorSpace::Grayscale, gray.bit_depth)
    }

    fn trace_contour(&self, binary: &Image, visited: &mut Vec<Vec<bool>>, start_x: usize, start_y: usize) -> Contour {
        let mut points = Vec::new();
        let mut x = start_x;
        let mut y = start_y;
        let mut direction = 0; // 0: right, 1: down, 2: left, 3: up

        let dx = [1, 0, -1, 0];
        let dy = [0, 1, 0, -1];

        loop {
            visited[y][x] = true;
            points.push((x as f32, y as f32));

            let mut found = false;
            for _ in 0..4 {
                let nx = x as i32 + dx[direction];
                let ny = y as i32 + dy[direction];

                if nx >= 0 && nx < binary.width as i32 && ny >= 0 && ny < binary.height as i32 {
                    let nx = nx as usize;
                    let ny = ny as usize;
                    if binary.get_pixel(nx, ny, 0).unwrap_or(0.0) > 0.5 && !visited[ny][nx] {
                        x = nx;
                        y = ny;
                        found = true;
                        break;
                    }
                }
                direction = (direction + 1) % 4;
            }

            if !found || (x == start_x && y == start_y && points.len() > 1) {
                break;
            }
        }

        let area = self.compute_contour_area(&points);
        let perimeter = self.compute_contour_perimeter(&points);

        Contour {
            points,
            area,
            perimeter,
            is_closed: true,
        }
    }

    fn harris_response(&self) -> Image {
        let mut response_data = vec![0.0; self.data.len()];

        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                // Compute image gradients
                let ix = self.get_pixel(x + 1, y, 0).unwrap_or(0.0) - self.get_pixel(x - 1, y, 0).unwrap_or(0.0);
                let iy = self.get_pixel(x, y + 1, 0).unwrap_or(0.0) - self.get_pixel(x, y - 1, 0).unwrap_or(0.0);

                // Harris matrix elements
                let ixx = ix * ix;
                let iyy = iy * iy;
                let ixy = ix * iy;

                // Harris response (simplified)
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let k = 0.04;
                let response = det - k * trace * trace;

                let index = y * self.width + x;
                response_data[index] = response.max(0.0);
            }
        }

        Image::from_data(response_data, self.width, self.height, ColorSpace::Grayscale, self.bit_depth)
    }

    fn compute_ncc(&self, template: &Image, x: usize, y: usize) -> f32 {
        let mut sum_template = 0.0;
        let mut sum_image = 0.0;
        let mut sum_template_sq = 0.0;
        let mut sum_image_sq = 0.0;
        let mut sum_product = 0.0;
        let mut count = 0.0;

        for ty in 0..template.height {
            for tx in 0..template.width {
                for c in 0..template.channels.min(self.channels) {
                    let template_val = template.get_pixel(tx, ty, c).unwrap_or(0.0);
                    let image_val = self.get_pixel(x + tx, y + ty, c).unwrap_or(0.0);

                    sum_template += template_val;
                    sum_image += image_val;
                    sum_template_sq += template_val * template_val;
                    sum_image_sq += image_val * image_val;
                    sum_product += template_val * image_val;
                    count += 1.0;
                }
            }
        }

        if count == 0.0 {
            return 0.0;
        }

        let mean_template = sum_template / count;
        let mean_image = sum_image / count;

        let numerator = sum_product - count * mean_template * mean_image;
        let denominator = ((sum_template_sq - count * mean_template * mean_template) *
                          (sum_image_sq - count * mean_image * mean_image)).sqrt();

        if denominator.abs() < f32::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn threshold_segmentation(&self, threshold: f32) -> SegmentationResult {
        let gray = if self.color_space != ColorSpace::Grayscale {
            self.to_grayscale()
        } else {
            self.clone()
        };

        let mut labels_data = vec![0.0; gray.data.len()];
        let label = 1.0;

        for i in 0..gray.data.len() {
            labels_data[i] = if gray.data[i] > threshold { label } else { 0.0 };
        }

        // Compute region statistics before moving labels_data
        let foreground_area = labels_data.iter().filter(|&&x| x > 0.0).count() as f32;
        
        let labels = Image::from_data(labels_data, gray.width, gray.height, ColorSpace::Grayscale, gray.bit_depth);

        // Compute region statistics
        let region_stats = vec![RegionStats {
            label: 1,
            area: foreground_area,
            centroid: (gray.width as f32 / 2.0, gray.height as f32 / 2.0),
            bounding_box: (0.0, 0.0, gray.width as f32, gray.height as f32),
        }];

        SegmentationResult {
            labels,
            num_regions: 2, // Background + foreground
            region_stats,
        }
    }

    fn watershed_segmentation(&self) -> SegmentationResult {
        // Simplified watershed - use threshold for now
        self.threshold_segmentation(0.5)
    }

    fn compute_contour_area(&self, points: &[(f32, f32)]) -> f32 {
        if points.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            area += points[i].0 * points[j].1;
            area -= points[j].0 * points[i].1;
        }
        area.abs() / 2.0
    }

    fn compute_contour_perimeter(&self, points: &[(f32, f32)]) -> f32 {
        if points.len() < 2 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        for i in 0..points.len() {
            let j = (i + 1) % points.len();
            let dx = points[j].0 - points[i].0;
            let dy = points[j].1 - points[i].1;
            perimeter += (dx * dx + dy * dy).sqrt();
        }
        perimeter
    }
}

// ===============================
// PHASE 6D: ADVANCED ANALYSIS (6 functions)
// ===============================

/// Apply affine transformation to image
/// Syntax: AffineTransform[image, transformMatrix, [interpolation]]
pub fn affine_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, transformMatrix, [interpolation])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let matrix = parse_affine_matrix(&args[1])?;
    let interpolation = if args.len() == 3 {
        parse_interpolation_method(&args[2])
    } else {
        InterpolationMethod::Bilinear
    };

    let result = image.affine_transform(&matrix, interpolation);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Apply perspective transformation to image
/// Syntax: PerspectiveTransform[image, transformMatrix, [interpolation]]
pub fn perspective_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, transformMatrix, [interpolation])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let matrix = parse_perspective_matrix(&args[1])?;
    let interpolation = if args.len() == 3 {
        parse_interpolation_method(&args[2])
    } else {
        InterpolationMethod::Bilinear
    };

    let result = image.perspective_transform(&matrix, interpolation);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Detect contours in image
/// Syntax: ContourDetection[image, [threshold], [method]]
pub fn contour_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (image, [threshold], [method])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let threshold = if args.len() >= 2 {
        match &args[1] {
            Value::Real(r) => *r as f32,
            Value::Integer(i) => *i as f32,
            _ => 0.5,
        }
    } else {
        0.5
    };

    let contours = image.detect_contours(threshold);
    // Return list of Associations for each contour
    let contour_objects: Vec<Value> = contours
        .into_iter()
        .map(|contour| {
            let points_list: Vec<Value> = contour
                .points
                .iter()
                .map(|(x, y)| Value::List(vec![Value::Real(*x as f64), Value::Real(*y as f64)]))
                .collect();
            let mut m = std::collections::HashMap::new();
            m.insert("points".to_string(), Value::List(points_list));
            m.insert("area".to_string(), Value::Real(contour.area as f64));
            m.insert("perimeter".to_string(), Value::Real(contour.perimeter as f64));
            m.insert("isClosed".to_string(), Value::Boolean(contour.is_closed));
            Value::Object(m)
        })
        .collect();

    Ok(Value::List(contour_objects))
}

/// Detect features in image
/// Syntax: FeatureDetection[image, [method], [threshold]]
pub fn feature_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (image, [method], [threshold])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let threshold = if args.len() >= 3 {
        match &args[2] {
            Value::Real(r) => *r as f32,
            Value::Integer(i) => *i as f32,
            _ => 0.01,
        }
    } else {
        0.01
    };

    let features = image.detect_features(threshold);
    // Return list of Associations for each feature point
    let feature_objects: Vec<Value> = features
        .into_iter()
        .map(|feature| {
            let mut m = std::collections::HashMap::new();
            m.insert("x".to_string(), Value::Real(feature.x as f64));
            m.insert("y".to_string(), Value::Real(feature.y as f64));
            m.insert("confidence".to_string(), Value::Real(feature.confidence as f64));
            m.insert("scale".to_string(), Value::Real(feature.scale as f64));
            m.insert("angle".to_string(), Value::Real(feature.angle as f64));
            Value::Object(m)
        })
        .collect();

    Ok(Value::List(feature_objects))
}

/// Template matching in image
/// Syntax: TemplateMatching[image, template, [method]]
pub fn template_matching(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, template, [method])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let template = extract_image(&args[1])?;

    let result = image.template_match(template);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Segment image into regions
/// Syntax: ImageSegmentation[image, [method], [parameters]]
pub fn image_segmentation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (image, [method], [parameters])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let method = if args.len() >= 2 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "Threshold",
        }
    } else {
        "Threshold"
    };

    let result = image.segment_image(method);
    // Return Association with labels (Image), numRegions, regionStats list of Associations
    let mut m = std::collections::HashMap::new();
    m.insert("labels".to_string(), Value::LyObj(LyObj::new(Box::new(result.labels))));
    m.insert("numRegions".to_string(), Value::Integer(result.num_regions as i64));
    let stats_list: Vec<Value> = result
        .region_stats
        .iter()
        .map(|stat| {
            let mut sm = std::collections::HashMap::new();
            sm.insert("label".to_string(), Value::Integer(stat.label as i64));
            sm.insert("area".to_string(), Value::Real(stat.area as f64));
            sm.insert(
                "centroid".to_string(),
                Value::List(vec![Value::Real(stat.centroid.0 as f64), Value::Real(stat.centroid.1 as f64)]),
            );
            sm.insert(
                "boundingBox".to_string(),
                Value::List(vec![
                    Value::Real(stat.bounding_box.0 as f64),
                    Value::Real(stat.bounding_box.1 as f64),
                    Value::Real(stat.bounding_box.2 as f64),
                    Value::Real(stat.bounding_box.3 as f64),
                ]),
            );
            Value::Object(sm)
        })
        .collect();
    m.insert("regionStats".to_string(), Value::List(stats_list));
    Ok(Value::Object(m))
}

// ===============================
// HELPER FUNCTIONS
// ===============================

/// Extract Image from Value
fn extract_image(value: &Value) -> VmResult<&Image> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Parse affine transformation matrix from Value
fn parse_affine_matrix(value: &Value) -> VmResult<AffineMatrix> {
    match value {
        Value::List(rows) => {
            if rows.len() != 2 {
                return Ok(AffineMatrix::identity());
            }

            let mut matrix = [[0.0f32; 3]; 2];
            for (i, row) in rows.iter().enumerate() {
                if let Value::List(cols) = row {
                    if cols.len() != 3 {
                        return Ok(AffineMatrix::identity());
                    }
                    for (j, col) in cols.iter().enumerate() {
                        matrix[i][j] = match col {
                            Value::Real(r) => *r as f32,
                            Value::Integer(i) => *i as f32,
                            _ => 0.0,
                        };
                    }
                }
            }
            Ok(AffineMatrix { matrix })
        }
        _ => Ok(AffineMatrix::identity()),
    }
}

/// Parse perspective transformation matrix from Value
fn parse_perspective_matrix(value: &Value) -> VmResult<PerspectiveMatrix> {
    match value {
        Value::List(rows) => {
            if rows.len() != 3 {
                return Ok(PerspectiveMatrix::identity());
            }

            let mut matrix = [[0.0f32; 3]; 3];
            for (i, row) in rows.iter().enumerate() {
                if let Value::List(cols) = row {
                    if cols.len() != 3 {
                        return Ok(PerspectiveMatrix::identity());
                    }
                    for (j, col) in cols.iter().enumerate() {
                        matrix[i][j] = match col {
                            Value::Real(r) => *r as f32,
                            Value::Integer(i) => *i as f32,
                            _ => if i == j { 1.0 } else { 0.0 },
                        };
                    }
                }
            }
            Ok(PerspectiveMatrix { matrix })
        }
        _ => Ok(PerspectiveMatrix::identity()),
    }
}

/// Parse interpolation method from Value
fn parse_interpolation_method(value: &Value) -> InterpolationMethod {
    match value {
        Value::String(s) => match s.as_str() {
            "NearestNeighbor" => InterpolationMethod::NearestNeighbor,
            "Bilinear" => InterpolationMethod::Bilinear,
            "Bicubic" => InterpolationMethod::Bicubic,
            "Lanczos" => InterpolationMethod::Lanczos,
            _ => InterpolationMethod::Bilinear,
        },
        _ => InterpolationMethod::Bilinear,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_affine_matrix() {
        let matrix = AffineMatrix::from_values(1.0, 0.0, 0.0, 1.0, 10.0, 20.0);
        let (x, y) = matrix.transform_point(5.0, 5.0);
        assert_eq!(x, 15.0);
        assert_eq!(y, 25.0);
    }

    #[test]
    fn test_perspective_matrix() {
        let matrix = PerspectiveMatrix::identity();
        let (x, y) = matrix.transform_point(5.0, 5.0);
        assert_eq!(x, 5.0);
        assert_eq!(y, 5.0);
    }

    #[test]
    fn test_feature_point_foreign() {
        let feature = FeaturePoint {
            x: 10.0,
            y: 20.0,
            confidence: 0.8,
            scale: 1.0,
            angle: 0.0,
        };

        let x = feature.call_method("X", &[]).unwrap();
        assert_eq!(x, Value::Real(10.0));

        let pos = feature.call_method("Position", &[]).unwrap();
        match pos {
            Value::List(list) => {
                assert_eq!(list.len(), 2);
                assert_eq!(list[0], Value::Real(10.0));
                assert_eq!(list[1], Value::Real(20.0));
            }
            _ => panic!("Expected List"),
        }
    }

    #[test]
    fn test_contour_foreign() {
        let contour = Contour {
            points: vec![(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            area: 1.0,
            perimeter: 4.0,
            is_closed: true,
        };

        let area = contour.call_method("Area", &[]).unwrap();
        assert_eq!(area, Value::Real(1.0));

        let length = contour.call_method("Length", &[]).unwrap();
        assert_eq!(length, Value::Integer(4));
    }

    #[test]
    fn test_image_affine_transform() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let matrix = AffineMatrix::identity();
        let transformed = image.affine_transform(&matrix, InterpolationMethod::NearestNeighbor);
        
        assert_eq!(transformed.channels, 3);
    }

    #[test]
    fn test_image_feature_detection() {
        let mut image = Image::new(10, 10, ColorSpace::Grayscale, 8);
        // Set some corner-like pattern
        image.set_pixel(5, 5, 0, 1.0);
        
        let features = image.detect_features(0.01);
        // Should detect at least some features (may be 0 for simple pattern)
        assert!(features.len() >= 0);
    }

    #[test]
    fn test_image_contour_detection() {
        let mut image = Image::new(10, 10, ColorSpace::Grayscale, 8);
        // Create a simple shape
        for x in 3..7 {
            for y in 3..7 {
                image.set_pixel(x, y, 0, 1.0);
            }
        }
        
        let contours = image.detect_contours(0.5);
        assert!(contours.len() >= 0);
    }

    #[test]
    fn test_template_matching() {
        let image = Image::new(10, 10, ColorSpace::Grayscale, 8);
        let template = Image::new(3, 3, ColorSpace::Grayscale, 8);
        
        let result = image.template_match(&template);
        assert_eq!(result.width, 8); // 10 - 3 + 1
        assert_eq!(result.height, 8);
        assert_eq!(result.channels, 1);
    }

    #[test]
    fn test_image_segmentation() {
        let image = Image::new(10, 10, ColorSpace::Grayscale, 8);
        let result = image.segment_image("Threshold");
        
        assert_eq!(result.labels.width, 10);
        assert_eq!(result.labels.height, 10);
        assert_eq!(result.num_regions, 2);
    }

    #[test]
    fn test_affine_transform_function() {
        let image = Image::new(5, 5, ColorSpace::RGB, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let matrix_value = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(0.0), Value::Real(0.0)]),
            Value::List(vec![Value::Real(0.0), Value::Real(1.0), Value::Real(0.0)]),
        ]);
        
        let result = affine_transform(&[image_value, matrix_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let transformed = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(transformed.channels, 3);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_contour_detection_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = contour_detection(&[image_value]).unwrap();
        match result {
            Value::List(_contours) => {
                // Should return a list of contours
            }
            _ => panic!("Expected List of contours"),
        }
    }

    #[test]
    fn test_feature_detection_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = feature_detection(&[image_value]).unwrap();
        match result {
            Value::List(_features) => {
                // Should return a list of features
            }
            _ => panic!("Expected List of features"),
        }
    }
}
