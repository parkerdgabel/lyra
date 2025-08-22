//! Geometric Transformation Algorithms
//!
//! This module implements comprehensive geometric transformation algorithms including:
//! - Affine transformations (translation, rotation, scaling, shearing)
//! - Perspective transformations with homography
//! - Composite transformations and matrix operations
//! - Image warping and resampling with different interpolation methods
//! - Transformation estimation from point correspondences

use crate::foreign::LyObj;
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::{Image, InterpolationMethod};
use super::{TransformMatrix, KeyPoint};
use std::f32::consts::PI;

/// Parameters for image transformation
#[derive(Debug, Clone)]
pub struct TransformParams {
    pub interpolation: InterpolationMethod, // Interpolation method
    pub border_mode: BorderMode,            // How to handle borders
    pub fill_value: f32,                   // Fill value for constant border
    pub output_size: Option<(usize, usize)>, // Optional output size
}

impl Default for TransformParams {
    fn default() -> Self {
        Self {
            interpolation: InterpolationMethod::Bilinear,
            border_mode: BorderMode::Constant,
            fill_value: 0.0,
            output_size: None,
        }
    }
}

/// Border handling modes for image transformation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BorderMode {
    Constant,    // Fill with constant value
    Reflect,     // Reflect border pixels
    Wrap,        // Wrap around
    Clamp,       // Clamp to edge pixels
}

/// Homography estimation parameters
#[derive(Debug, Clone)]
pub struct HomographyParams {
    pub method: HomographyMethod,       // Estimation method
    pub max_iterations: usize,          // Maximum RANSAC iterations
    pub threshold: f32,                 // Inlier threshold
    pub confidence: f32,                // Desired confidence level
}

impl Default for HomographyParams {
    fn default() -> Self {
        Self {
            method: HomographyMethod::RANSAC,
            max_iterations: 1000,
            threshold: 3.0,
            confidence: 0.99,
        }
    }
}

/// Homography estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HomographyMethod {
    DLT,        // Direct Linear Transformation
    RANSAC,     // Random Sample Consensus
    LeastSquares, // Least squares
}

/// Apply affine transformation to image
pub fn apply_affine_transform(image: &Image, transform: &TransformMatrix, params: Option<TransformParams>) -> VmResult<Image> {
    let params = params.unwrap_or_default();
    
    // Determine output size
    let (output_width, output_height) = params.output_size.unwrap_or((image.width, image.height));
    
    let mut result = Image {
        data: vec![params.fill_value; output_width * output_height * image.channels],
        width: output_width,
        height: output_height,
        channels: image.channels,
        color_space: image.color_space,
        bit_depth: image.bit_depth,
    };
    
    // Inverse transformation matrix for backward mapping
    let inv_transform = invert_matrix(transform)?;
    
    for y in 0..output_height {
        for x in 0..output_width {
            // Transform output coordinates to input coordinates
            let (src_x, src_y) = inv_transform.transform_point(x as f32, y as f32);
            
            // Sample from source image
            for c in 0..image.channels {
                let pixel_value = sample_pixel(image, src_x, src_y, c, params.interpolation, params.border_mode, params.fill_value);
                let out_idx = (y * output_width + x) * image.channels + c;
                result.data[out_idx] = pixel_value;
            }
        }
    }
    
    Ok(result)
}

/// Apply perspective transformation to image
pub fn apply_perspective_transform(image: &Image, transform: &TransformMatrix, params: Option<TransformParams>) -> VmResult<Image> {
    let params = params.unwrap_or_default();
    
    // Determine output size
    let (output_width, output_height) = params.output_size.unwrap_or((image.width, image.height));
    
    let mut result = Image {
        data: vec![params.fill_value; output_width * output_height * image.channels],
        width: output_width,
        height: output_height,
        channels: image.channels,
        color_space: image.color_space,
        bit_depth: image.bit_depth,
    };
    
    // Inverse transformation matrix for backward mapping
    let inv_transform = invert_matrix(transform)?;
    
    for y in 0..output_height {
        for x in 0..output_width {
            // Transform output coordinates to input coordinates using perspective division
            let (src_x, src_y) = inv_transform.transform_point(x as f32, y as f32);
            
            // Sample from source image
            for c in 0..image.channels {
                let pixel_value = sample_pixel(image, src_x, src_y, c, params.interpolation, params.border_mode, params.fill_value);
                let out_idx = (y * output_width + x) * image.channels + c;
                result.data[out_idx] = pixel_value;
            }
        }
    }
    
    Ok(result)
}

/// Create affine transformation matrix from parameters
pub fn create_affine_transform(translation: (f32, f32), rotation: f32, scaling: (f32, f32), shearing: (f32, f32)) -> TransformMatrix {
    // Create individual transformation matrices
    let t_matrix = TransformMatrix::translation(translation.0, translation.1);
    let r_matrix = TransformMatrix::rotation(rotation);
    let s_matrix = TransformMatrix::scaling(scaling.0, scaling.1);
    let sh_matrix = create_shear_matrix(shearing.0, shearing.1);
    
    // Compose transformations: T * R * S * Sh
    let temp1 = t_matrix.compose(&r_matrix);
    let temp2 = temp1.compose(&s_matrix);
    temp2.compose(&sh_matrix)
}

/// Create shearing transformation matrix
pub fn create_shear_matrix(shear_x: f32, shear_y: f32) -> TransformMatrix {
    TransformMatrix {
        matrix: [
            [1.0, shear_x, 0.0],
            [shear_y, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        transform_type: "shear".to_string(),
    }
}

/// Create perspective transformation matrix from four point correspondences
pub fn create_perspective_transform(src_points: &[(f32, f32)], dst_points: &[(f32, f32)]) -> VmResult<TransformMatrix> {
    if src_points.len() != 4 || dst_points.len() != 4 {
        return Err(VmError::Runtime("Need exactly 4 point correspondences".to_string()));
    }
    
    // Set up the linear system for homography estimation (DLT)
    let mut a_matrix = vec![vec![0.0; 8]; 8];
    
    for i in 0..4 {
        let (x, y) = src_points[i];
        let (u, v) = dst_points[i];
        
        // First row for each correspondence
        let row1 = 2 * i;
        a_matrix[row1][0] = x;
        a_matrix[row1][1] = y;
        a_matrix[row1][2] = 1.0;
        a_matrix[row1][3] = 0.0;
        a_matrix[row1][4] = 0.0;
        a_matrix[row1][5] = 0.0;
        a_matrix[row1][6] = -u * x;
        a_matrix[row1][7] = -u * y;
        
        // Second row for each correspondence
        let row2 = 2 * i + 1;
        a_matrix[row2][0] = 0.0;
        a_matrix[row2][1] = 0.0;
        a_matrix[row2][2] = 0.0;
        a_matrix[row2][3] = x;
        a_matrix[row2][4] = y;
        a_matrix[row2][5] = 1.0;
        a_matrix[row2][6] = -v * x;
        a_matrix[row2][7] = -v * y;
    }
    
    let b_vector = vec![dst_points[0].0, dst_points[0].1, dst_points[1].0, dst_points[1].1,
                       dst_points[2].0, dst_points[2].1, dst_points[3].0, dst_points[3].1];
    
    // Solve the linear system (simplified - in practice would use SVD or other robust method)
    let h_params = solve_linear_system(&a_matrix, &b_vector)?;
    
    // Construct the homography matrix
    let mut transform = TransformMatrix::identity();
    transform.matrix[0][0] = h_params[0];
    transform.matrix[0][1] = h_params[1];
    transform.matrix[0][2] = h_params[2];
    transform.matrix[1][0] = h_params[3];
    transform.matrix[1][1] = h_params[4];
    transform.matrix[1][2] = h_params[5];
    transform.matrix[2][0] = h_params[6];
    transform.matrix[2][1] = h_params[7];
    transform.matrix[2][2] = 1.0;
    transform.transform_type = "perspective".to_string();
    
    Ok(transform)
}

/// Estimate homography using RANSAC
pub fn estimate_homography_ransac(src_points: &[(f32, f32)], dst_points: &[(f32, f32)], params: Option<HomographyParams>) -> VmResult<TransformMatrix> {
    let params = params.unwrap_or_default();
    
    if src_points.len() != dst_points.len() || src_points.len() < 4 {
        return Err(VmError::Runtime("Need at least 4 point correspondences".to_string()));
    }
    
    let mut best_transform = TransformMatrix::identity();
    let mut best_inlier_count = 0;
    
    for _ in 0..params.max_iterations {
        // Randomly select 4 points
        let indices = select_random_indices(src_points.len(), 4);
        let sample_src: Vec<_> = indices.iter().map(|&i| src_points[i]).collect();
        let sample_dst: Vec<_> = indices.iter().map(|&i| dst_points[i]).collect();
        
        // Estimate homography from 4 points
        if let Ok(transform) = create_perspective_transform(&sample_src, &sample_dst) {
            // Count inliers
            let inlier_count = count_inliers(&transform, src_points, dst_points, params.threshold);
            
            if inlier_count > best_inlier_count {
                best_inlier_count = inlier_count;
                best_transform = transform;
            }
            
            // Early termination if we have enough inliers
            let inlier_ratio = inlier_count as f32 / src_points.len() as f32;
            if inlier_ratio >= params.confidence {
                break;
            }
        }
    }
    
    if best_inlier_count < 4 {
        return Err(VmError::Runtime("Could not find reliable homography".to_string()));
    }
    
    best_transform.transform_type = "ransac_homography".to_string();
    Ok(best_transform)
}

/// Transform keypoints using transformation matrix
pub fn transform_keypoints(keypoints: &[KeyPoint], transform: &TransformMatrix) -> Vec<KeyPoint> {
    keypoints.iter().map(|kp| {
        let (new_x, new_y) = transform.transform_point(kp.x, kp.y);
        KeyPoint {
            x: new_x,
            y: new_y,
            size: kp.size,
            angle: kp.angle,
            response: kp.response,
            octave: kp.octave,
            class_id: kp.class_id,
        }
    }).collect()
}

/// Warp image using thin plate spline transformation
pub fn apply_thin_plate_spline_transform(image: &Image, control_points: &[(f32, f32)], target_points: &[(f32, f32)], params: Option<TransformParams>) -> VmResult<Image> {
    let params = params.unwrap_or_default();
    
    if control_points.len() != target_points.len() || control_points.len() < 3 {
        return Err(VmError::Runtime("Need at least 3 point correspondences for TPS".to_string()));
    }
    
    // Compute TPS coefficients
    let tps_transform = compute_tps_coefficients(control_points, target_points)?;
    
    // Apply transformation
    let (output_width, output_height) = params.output_size.unwrap_or((image.width, image.height));
    
    let mut result = Image {
        data: vec![params.fill_value; output_width * output_height * image.channels],
        width: output_width,
        height: output_height,
        channels: image.channels,
        color_space: image.color_space,
        bit_depth: image.bit_depth,
    };
    
    for y in 0..output_height {
        for x in 0..output_width {
            // Apply TPS transformation
            let (src_x, src_y) = apply_tps_transform(&tps_transform, x as f32, y as f32);
            
            // Sample from source image
            for c in 0..image.channels {
                let pixel_value = sample_pixel(image, src_x, src_y, c, params.interpolation, params.border_mode, params.fill_value);
                let out_idx = (y * output_width + x) * image.channels + c;
                result.data[out_idx] = pixel_value;
            }
        }
    }
    
    Ok(result)
}

/// Get transformation matrix for common operations
pub fn get_transform_matrix(transform_type: &str, params: &[f32]) -> VmResult<TransformMatrix> {
    match transform_type {
        "translation" => {
            if params.len() != 2 {
                return Err(VmError::Runtime("Translation requires 2 parameters (dx, dy)".to_string()));
            }
            Ok(TransformMatrix::translation(params[0], params[1]))
        }
        "rotation" => {
            if params.len() != 1 {
                return Err(VmError::Runtime("Rotation requires 1 parameter (angle in radians)".to_string()));
            }
            Ok(TransformMatrix::rotation(params[0]))
        }
        "scaling" => {
            if params.len() == 1 {
                Ok(TransformMatrix::scaling(params[0], params[0]))
            } else if params.len() == 2 {
                Ok(TransformMatrix::scaling(params[0], params[1]))
            } else {
                Err(VmError::Runtime("Scaling requires 1 or 2 parameters".to_string()))
            }
        }
        "shear" => {
            if params.len() != 2 {
                return Err(VmError::Runtime("Shear requires 2 parameters (shear_x, shear_y)".to_string()));
            }
            Ok(create_shear_matrix(params[0], params[1]))
        }
        _ => Err(VmError::Runtime(format!("Unknown transform type: {}", transform_type))),
    }
}

// Helper functions for transformation implementation

/// Invert a 3x3 transformation matrix
fn invert_matrix(transform: &TransformMatrix) -> VmResult<TransformMatrix> {
    let m = &transform.matrix;
    
    // Calculate determinant
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
              m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
              m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    
    if det.abs() < 1e-10 {
        return Err(VmError::Runtime("Matrix is singular and cannot be inverted".to_string()));
    }
    
    let inv_det = 1.0 / det;
    
    let mut inv_matrix = [[0.0; 3]; 3];
    
    // Calculate adjugate matrix and divide by determinant
    inv_matrix[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det;
    inv_matrix[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv_matrix[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv_matrix[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv_matrix[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv_matrix[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det;
    inv_matrix[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det;
    inv_matrix[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det;
    inv_matrix[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det;
    
    Ok(TransformMatrix {
        matrix: inv_matrix,
        transform_type: format!("inverse_{}", transform.transform_type),
    })
}

/// Sample pixel value from image with interpolation
fn sample_pixel(image: &Image, x: f32, y: f32, channel: usize, interpolation: InterpolationMethod, border_mode: BorderMode, fill_value: f32) -> f32 {
    match interpolation {
        InterpolationMethod::NearestNeighbor => sample_nearest_neighbor(image, x, y, channel, border_mode, fill_value),
        InterpolationMethod::Bilinear => sample_bilinear(image, x, y, channel, border_mode, fill_value),
        InterpolationMethod::Bicubic => sample_bicubic(image, x, y, channel, border_mode, fill_value),
        InterpolationMethod::Lanczos => sample_lanczos(image, x, y, channel, border_mode, fill_value),
    }
}

/// Nearest neighbor interpolation
fn sample_nearest_neighbor(image: &Image, x: f32, y: f32, channel: usize, border_mode: BorderMode, fill_value: f32) -> f32 {
    let xi = x.round() as i32;
    let yi = y.round() as i32;
    
    let (px, py) = handle_border_coordinates(xi, yi, image.width, image.height, border_mode);
    
    if px < 0 || px >= image.width as i32 || py < 0 || py >= image.height as i32 {
        return fill_value;
    }
    
    let idx = (py as usize * image.width + px as usize) * image.channels + channel;
    image.data.get(idx).copied().unwrap_or(fill_value)
}

/// Bilinear interpolation
fn sample_bilinear(image: &Image, x: f32, y: f32, channel: usize, border_mode: BorderMode, fill_value: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    
    // Get the four corner pixels
    let p00 = get_pixel_with_border(image, x0, y0, channel, border_mode, fill_value);
    let p10 = get_pixel_with_border(image, x1, y0, channel, border_mode, fill_value);
    let p01 = get_pixel_with_border(image, x0, y1, channel, border_mode, fill_value);
    let p11 = get_pixel_with_border(image, x1, y1, channel, border_mode, fill_value);
    
    // Bilinear interpolation
    let top = p00 * (1.0 - fx) + p10 * fx;
    let bottom = p01 * (1.0 - fx) + p11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Bicubic interpolation (simplified version)
fn sample_bicubic(image: &Image, x: f32, y: f32, channel: usize, border_mode: BorderMode, fill_value: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    
    let mut sum = 0.0;
    
    // Sample 4x4 neighborhood
    for dy in -1..3 {
        for dx in -1..3 {
            let px = x0 + dx;
            let py = y0 + dy;
            let pixel = get_pixel_with_border(image, px, py, channel, border_mode, fill_value);
            
            let weight_x = cubic_weight(fx - dx as f32);
            let weight_y = cubic_weight(fy - dy as f32);
            sum += pixel * weight_x * weight_y;
        }
    }
    
    sum
}

/// Lanczos interpolation (simplified version using 3x3 kernel)
fn sample_lanczos(image: &Image, x: f32, y: f32, channel: usize, border_mode: BorderMode, fill_value: f32) -> f32 {
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    
    let mut sum = 0.0;
    let mut weight_sum = 0.0;
    
    // Sample 3x3 neighborhood
    for dy in -1..2 {
        for dx in -1..2 {
            let px = x0 + dx;
            let py = y0 + dy;
            let pixel = get_pixel_with_border(image, px, py, channel, border_mode, fill_value);
            
            let weight_x = lanczos_weight(fx - dx as f32, 2.0);
            let weight_y = lanczos_weight(fy - dy as f32, 2.0);
            let weight = weight_x * weight_y;
            
            sum += pixel * weight;
            weight_sum += weight;
        }
    }
    
    if weight_sum > 0.0 { sum / weight_sum } else { fill_value }
}

/// Get pixel value with border handling
fn get_pixel_with_border(image: &Image, x: i32, y: i32, channel: usize, border_mode: BorderMode, fill_value: f32) -> f32 {
    let (px, py) = handle_border_coordinates(x, y, image.width, image.height, border_mode);
    
    if px < 0 || px >= image.width as i32 || py < 0 || py >= image.height as i32 {
        return fill_value;
    }
    
    let idx = (py as usize * image.width + px as usize) * image.channels + channel;
    image.data.get(idx).copied().unwrap_or(fill_value)
}

/// Handle border coordinates based on border mode
fn handle_border_coordinates(x: i32, y: i32, width: usize, height: usize, border_mode: BorderMode) -> (i32, i32) {
    match border_mode {
        BorderMode::Constant => (x, y),
        BorderMode::Reflect => {
            let px = if x < 0 {
                -x
            } else if x >= width as i32 {
                2 * (width as i32 - 1) - x
            } else {
                x
            };
            
            let py = if y < 0 {
                -y
            } else if y >= height as i32 {
                2 * (height as i32 - 1) - y
            } else {
                y
            };
            
            (px, py)
        }
        BorderMode::Wrap => {
            let px = ((x % width as i32) + width as i32) % width as i32;
            let py = ((y % height as i32) + height as i32) % height as i32;
            (px, py)
        }
        BorderMode::Clamp => {
            let px = x.max(0).min(width as i32 - 1);
            let py = y.max(0).min(height as i32 - 1);
            (px, py)
        }
    }
}

/// Cubic interpolation weight function
fn cubic_weight(t: f32) -> f32 {
    let t = t.abs();
    if t <= 1.0 {
        1.5 * t * t * t - 2.5 * t * t + 1.0
    } else if t <= 2.0 {
        -0.5 * t * t * t + 2.5 * t * t - 4.0 * t + 2.0
    } else {
        0.0
    }
}

/// Lanczos weight function
fn lanczos_weight(x: f32, a: f32) -> f32 {
    if x.abs() < 1e-10 {
        return 1.0;
    }
    if x.abs() >= a {
        return 0.0;
    }
    
    let pi_x = PI * x;
    let pi_x_a = pi_x / a;
    (a * pi_x.sin() * pi_x_a.sin()) / (pi_x * pi_x_a)
}

/// Solve linear system (simplified Gaussian elimination)
fn solve_linear_system(a: &[Vec<f32>], b: &[f32]) -> VmResult<Vec<f32>> {
    let n = a.len();
    if n != b.len() || n == 0 || a[0].len() != n {
        return Err(VmError::Runtime("Invalid matrix dimensions".to_string()));
    }
    
    let mut aug_matrix = a.clone();
    let mut b_vec = b.to_vec();
    
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug_matrix[k][i].abs() > aug_matrix[max_row][i].abs() {
                max_row = k;
            }
        }
        
        // Swap rows
        aug_matrix.swap(i, max_row);
        b_vec.swap(i, max_row);
        
        // Check for singular matrix
        if aug_matrix[i][i].abs() < 1e-10 {
            return Err(VmError::Runtime("Matrix is singular".to_string()));
        }
        
        // Eliminate column
        for k in (i + 1)..n {
            let factor = aug_matrix[k][i] / aug_matrix[i][i];
            for j in i..n {
                aug_matrix[k][j] -= factor * aug_matrix[i][j];
            }
            b_vec[k] -= factor * b_vec[i];
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b_vec[i];
        for j in (i + 1)..n {
            x[i] -= aug_matrix[i][j] * x[j];
        }
        x[i] /= aug_matrix[i][i];
    }
    
    Ok(x)
}

/// Select random indices for RANSAC
fn select_random_indices(max: usize, count: usize) -> Vec<usize> {
    // Simplified random selection - in practice would use proper RNG
    let mut indices = Vec::new();
    let step = max / count;
    for i in 0..count {
        indices.push((i * step) % max);
    }
    indices
}

/// Count inliers for homography validation
fn count_inliers(transform: &TransformMatrix, src_points: &[(f32, f32)], dst_points: &[(f32, f32)], threshold: f32) -> usize {
    let mut count = 0;
    
    for i in 0..src_points.len() {
        let (transformed_x, transformed_y) = transform.transform_point(src_points[i].0, src_points[i].1);
        let dx = transformed_x - dst_points[i].0;
        let dy = transformed_y - dst_points[i].1;
        let error = (dx * dx + dy * dy).sqrt();
        
        if error < threshold {
            count += 1;
        }
    }
    
    count
}

/// Thin Plate Spline transformation structure
#[derive(Debug, Clone)]
struct TpsTransform {
    control_points: Vec<(f32, f32)>,
    coefficients: Vec<f32>,
    affine_coeffs: [f32; 6],
}

/// Compute TPS coefficients
fn compute_tps_coefficients(control_points: &[(f32, f32)], target_points: &[(f32, f32)]) -> VmResult<TpsTransform> {
    let n = control_points.len();
    
    // Build the TPS system matrix
    let mut k_matrix = vec![vec![0.0; n + 3]; n + 3];
    let mut target_x = vec![0.0; n + 3];
    let mut target_y = vec![0.0; n + 3];
    
    // Fill K matrix (radial basis function values)
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let dx = control_points[i].0 - control_points[j].0;
                let dy = control_points[i].1 - control_points[j].1;
                let r_squared = dx * dx + dy * dy;
                k_matrix[i][j] = r_squared * r_squared.ln();
            }
        }
    }
    
    // Fill P matrix (polynomial terms)
    for i in 0..n {
        k_matrix[i][n] = 1.0;
        k_matrix[i][n + 1] = control_points[i].0;
        k_matrix[i][n + 2] = control_points[i].1;
        
        k_matrix[n][i] = 1.0;
        k_matrix[n + 1][i] = control_points[i].0;
        k_matrix[n + 2][i] = control_points[i].1;
    }
    
    // Set target points
    for i in 0..n {
        target_x[i] = target_points[i].0;
        target_y[i] = target_points[i].1;
    }
    
    // Solve for x coefficients
    let x_coeffs = solve_linear_system(&k_matrix, &target_x)?;
    let y_coeffs = solve_linear_system(&k_matrix, &target_y)?;
    
    // Separate coefficients
    let mut coefficients = Vec::new();
    coefficients.extend(&x_coeffs[0..n]);
    coefficients.extend(&y_coeffs[0..n]);
    
    let affine_coeffs = [
        x_coeffs[n], x_coeffs[n + 1], x_coeffs[n + 2],
        y_coeffs[n], y_coeffs[n + 1], y_coeffs[n + 2],
    ];
    
    Ok(TpsTransform {
        control_points: control_points.to_vec(),
        coefficients,
        affine_coeffs,
    })
}

/// Apply TPS transformation to a point
fn apply_tps_transform(tps: &TpsTransform, x: f32, y: f32) -> (f32, f32) {
    let n = tps.control_points.len();
    
    let mut new_x = tps.affine_coeffs[0] + tps.affine_coeffs[1] * x + tps.affine_coeffs[2] * y;
    let mut new_y = tps.affine_coeffs[3] + tps.affine_coeffs[4] * x + tps.affine_coeffs[5] * y;
    
    for i in 0..n {
        let dx = x - tps.control_points[i].0;
        let dy = y - tps.control_points[i].1;
        let r_squared = dx * dx + dy * dy;
        
        if r_squared > 0.0 {
            let rbf_value = r_squared * r_squared.ln();
            new_x += tps.coefficients[i] * rbf_value;
            new_y += tps.coefficients[i + n] * rbf_value;
        }
    }
    
    (new_x, new_y)
}

// Stdlib function implementations

/// Apply affine transformation - AffineTransform[image, transform, opts]
pub fn affine_transform(args: &[Value]) -> VmResult<Value> {
    match args {
        [img, trans] => {
            let image = match img {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", img)
                })
            };
            
            let image = image.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: image.type_name().to_string(),
                })?;
            
            let transform = match trans {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", trans)
                })
            };
            
            let transform = transform.downcast_ref::<TransformMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TransformMatrix".to_string(),
                    actual: transform.type_name().to_string(),
                })?;
            
            let result = apply_affine_transform(image, transform, None)?;
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        [img, trans, opts] => {
            let image = match img {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", img)
                })
            };
            
            let image = image.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: image.type_name().to_string(),
                })?;
            
            let transform = match trans {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", trans)
                })
            };
            
            let transform = transform.downcast_ref::<TransformMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TransformMatrix".to_string(),
                    actual: transform.type_name().to_string(),
                })?;
            
            // Parse options (simplified)
            let params = TransformParams::default();
            
            let result = apply_affine_transform(image, transform, Some(params))?;
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        _ => Err(VmError::Runtime("AffineTransform expects 2 or 3 arguments".to_string())),
    }
}

/// Apply perspective transformation - PerspectiveTransform[image, transform, opts]
pub fn perspective_transform(args: &[Value]) -> VmResult<Value> {
    match args {
        [img, trans] => {
            let image = match img {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", img)
                })
            };
            
            let image = image.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: image.type_name().to_string(),
                })?;
            
            let transform = match trans {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", trans)
                })
            };
            
            let transform = transform.downcast_ref::<TransformMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TransformMatrix".to_string(),
                    actual: transform.type_name().to_string(),
                })?;
            
            let result = apply_perspective_transform(image, transform, None)?;
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        [img, trans, opts] => {
            let image = match img {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", img)
                })
            };
            
            let image = image.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: image.type_name().to_string(),
                })?;
            
            let transform = match trans {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", trans)
                })
            };
            
            let transform = transform.downcast_ref::<TransformMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TransformMatrix".to_string(),
                    actual: transform.type_name().to_string(),
                })?;
            
            // Parse options (simplified)
            let params = TransformParams::default();
            
            let result = apply_perspective_transform(image, transform, Some(params))?;
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        _ => Err(VmError::Runtime("PerspectiveTransform expects 2 or 3 arguments".to_string())),
    }
}

/// Create transformation matrix - CreateTransform[type, params]
pub fn create_transform(args: &[Value]) -> VmResult<Value> {
    match args {
        [transform_type, params_list] => {
            let type_str = match transform_type {
                Value::Symbol(s) => s.as_str(),
                Value::String(s) => s.as_str(),
                _ => return Err(VmError::Runtime("Expected transform type as symbol or string".to_string())),
            };
            
            let params = match params_list {
                Value::List(list) => {
                    let mut param_values = Vec::new();
                    for val in list {
                        match val {
                            Value::Integer(i) => param_values.push(*i as f32),
                            Value::Real(f) => param_values.push(*f as f32),
                            _ => return Err(VmError::Runtime("Parameters must be numbers".to_string())),
                        }
                    }
                    param_values
                }
                _ => return Err(VmError::Runtime("Expected parameter list".to_string())),
            };
            
            let transform = get_transform_matrix(type_str, &params)?;
            Ok(Value::LyObj(LyObj::new(Box::new(transform))))
        }
        _ => Err(VmError::Runtime("CreateTransform expects 2 arguments".to_string())),
    }
}

/// Estimate homography - EstimateHomography[srcPoints, dstPoints, opts]
pub fn estimate_homography(args: &[Value]) -> VmResult<Value> {
    match args {
        [src_pts, dst_pts] => {
            let src_points = extract_point_list(src_pts)?;
            let dst_points = extract_point_list(dst_pts)?;
            
            if src_points.len() == 4 && dst_points.len() == 4 {
                let transform = create_perspective_transform(&src_points, &dst_points)?;
                Ok(Value::LyObj(LyObj::new(Box::new(transform))))
            } else {
                let transform = estimate_homography_ransac(&src_points, &dst_points, None)?;
                Ok(Value::LyObj(LyObj::new(Box::new(transform))))
            }
        }
        [src_pts, dst_pts, opts] => {
            let src_points = extract_point_list(src_pts)?;
            let dst_points = extract_point_list(dst_pts)?;
            
            // Parse options (simplified)
            let params = HomographyParams::default();
            
            let transform = estimate_homography_ransac(&src_points, &dst_points, Some(params))?;
            Ok(Value::LyObj(LyObj::new(Box::new(transform))))
        }
        _ => Err(VmError::Runtime("EstimateHomography expects 2 or 3 arguments".to_string())),
    }
}

/// Transform keypoints - TransformKeypoints[keypoints, transform]
pub fn transform_keypoints_func(args: &[Value]) -> VmResult<Value> {
    match args {
        [kp_set, trans] => {
            // Extract keypoints from FeatureSet
            let feature_set = match kp_set {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", kp_set)
                })
            };
            
            let feature_set = feature_set.downcast_ref::<super::FeatureSet>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "FeatureSet".to_string(),
                    actual: feature_set.type_name().to_string(),
                })?;
            
            let transform = match trans {
                Value::LyObj(obj) => obj,
                _ => return Err(VmError::TypeError {
                    expected: "LyObj".to_string(),
                    actual: format!("{:?}", trans)
                })
            };
            
            let transform = transform.downcast_ref::<TransformMatrix>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TransformMatrix".to_string(),
                    actual: transform.type_name().to_string(),
                })?;
            
            let transformed_keypoints = transform_keypoints(&feature_set.keypoints, transform);
            
            let mut result = super::FeatureSet::new(feature_set.feature_type.clone());
            result.keypoints = transformed_keypoints;
            result.descriptors = feature_set.descriptors.clone();
            
            Ok(Value::LyObj(LyObj::new(Box::new(result))))
        }
        _ => Err(VmError::Runtime("TransformKeypoints expects 2 arguments".to_string())),
    }
}

/// Extract point list from Value
fn extract_point_list(value: &Value) -> VmResult<Vec<(f32, f32)>> {
    match value {
        Value::List(list) => {
            let mut points = Vec::new();
            for item in list {
                match item {
                    Value::List(point) if point.len() == 2 => {
                        let x = match &point[0] {
                            Value::Integer(i) => *i as f32,
                            Value::Real(f) => *f as f32,
                            _ => return Err(VmError::Runtime("Point coordinates must be numbers".to_string())),
                        };
                        let y = match &point[1] {
                            Value::Integer(i) => *i as f32,
                            Value::Real(f) => *f as f32,
                            _ => return Err(VmError::Runtime("Point coordinates must be numbers".to_string())),
                        };
                        points.push((x, y));
                    }
                    _ => return Err(VmError::Runtime("Expected list of 2D points".to_string())),
                }
            }
            Ok(points)
        }
        _ => Err(VmError::Runtime("Expected list of points".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::image::{Image, ColorSpace};

    /// Create a simple test image
    fn create_test_image(width: usize, height: usize) -> Image {
        Image {
            data: vec![0.5; width * height],
            width,
            height,
            channels: 1,
            color_space: ColorSpace::Grayscale,
            bit_depth: 8,
        }
    }

    /// Create a test image with a pattern
    fn create_pattern_image(width: usize, height: usize) -> Image {
        let mut image = create_test_image(width, height);
        
        // Create checkerboard pattern
        for y in 0..height {
            for x in 0..width {
                let value = if (x / 8 + y / 8) % 2 == 0 { 1.0 } else { 0.0 };
                image.data[y * width + x] = value;
            }
        }
        
        image
    }

    #[test]
    fn test_transform_matrix_creation() {
        let t_matrix = TransformMatrix::translation(10.0, 20.0);
        assert_eq!(t_matrix.matrix[0][2], 10.0);
        assert_eq!(t_matrix.matrix[1][2], 20.0);
        assert_eq!(t_matrix.transform_type, "translation");
        
        let r_matrix = TransformMatrix::rotation(PI / 2.0);
        assert!((r_matrix.matrix[0][0] - 0.0).abs() < 1e-6);
        assert!((r_matrix.matrix[0][1] - (-1.0)).abs() < 1e-6);
        
        let s_matrix = TransformMatrix::scaling(2.0, 3.0);
        assert_eq!(s_matrix.matrix[0][0], 2.0);
        assert_eq!(s_matrix.matrix[1][1], 3.0);
    }

    #[test]
    fn test_matrix_composition() {
        let t_matrix = TransformMatrix::translation(5.0, 10.0);
        let s_matrix = TransformMatrix::scaling(2.0, 2.0);
        
        let composed = t_matrix.compose(&s_matrix);
        
        // Check that composition worked
        assert_eq!(composed.matrix[0][0], 2.0);
        assert_eq!(composed.matrix[1][1], 2.0);
        assert_eq!(composed.matrix[0][2], 5.0);
        assert_eq!(composed.matrix[1][2], 10.0);
    }

    #[test]
    fn test_point_transformation() {
        let transform = TransformMatrix::translation(5.0, 10.0);
        let (new_x, new_y) = transform.transform_point(1.0, 2.0);
        
        assert_eq!(new_x, 6.0);
        assert_eq!(new_y, 12.0);
        
        let rotation = TransformMatrix::rotation(PI / 2.0);
        let (rot_x, rot_y) = rotation.transform_point(1.0, 0.0);
        
        assert!((rot_x - 0.0).abs() < 1e-6);
        assert!((rot_y - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_inversion() {
        let transform = TransformMatrix::translation(5.0, 10.0);
        let inverse = invert_matrix(&transform).unwrap();
        
        assert_eq!(inverse.matrix[0][2], -5.0);
        assert_eq!(inverse.matrix[1][2], -10.0);
        
        // Test that T * T^-1 = I
        let identity = transform.compose(&inverse);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((identity.matrix[i][j] - expected).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_affine_transform_creation() {
        let transform = create_affine_transform((5.0, 10.0), PI / 4.0, (2.0, 2.0), (0.1, 0.2));
        
        // Should be a valid transformation matrix
        assert_eq!(transform.matrix[2][2], 1.0);
        
        // Test transformation of origin
        let (x, y) = transform.transform_point(0.0, 0.0);
        assert!(x != 0.0 || y != 0.0); // Should be transformed
    }

    #[test]
    fn test_shear_matrix() {
        let shear = create_shear_matrix(0.5, 0.0);
        assert_eq!(shear.matrix[0][1], 0.5);
        assert_eq!(shear.matrix[1][0], 0.0);
        assert_eq!(shear.transform_type, "shear");
        
        // Test shear transformation
        let (x, y) = shear.transform_point(2.0, 4.0);
        assert_eq!(x, 4.0); // 2 + 0.5 * 4
        assert_eq!(y, 4.0); // y unchanged
    }

    #[test]
    fn test_perspective_transform_creation() {
        let src_points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let dst_points = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        
        let transform = create_perspective_transform(&src_points, &dst_points).unwrap();
        assert_eq!(transform.transform_type, "perspective");
        
        // Test transformation of corner points
        for i in 0..4 {
            let (x, y) = transform.transform_point(src_points[i].0, src_points[i].1);
            assert!((x - dst_points[i].0).abs() < 1.0);
            assert!((y - dst_points[i].1).abs() < 1.0);
        }
    }

    #[test]
    fn test_interpolation_methods() {
        let image = create_pattern_image(32, 32);
        
        // Test different interpolation methods
        let nn_val = sample_nearest_neighbor(&image, 15.5, 15.5, 0, BorderMode::Constant, 0.0);
        let bl_val = sample_bilinear(&image, 15.5, 15.5, 0, BorderMode::Constant, 0.0);
        let bc_val = sample_bicubic(&image, 15.5, 15.5, 0, BorderMode::Constant, 0.0);
        
        // All should return valid values
        assert!(nn_val >= 0.0 && nn_val <= 1.0);
        assert!(bl_val >= 0.0 && bl_val <= 1.0);
        assert!(bc_val >= 0.0 && bc_val <= 1.0);
    }

    #[test]
    fn test_border_handling() {
        let image = create_test_image(10, 10);
        
        // Test different border modes
        let constant = get_pixel_with_border(&image, -1, -1, 0, BorderMode::Constant, -1.0);
        assert_eq!(constant, -1.0);
        
        let clamp = get_pixel_with_border(&image, -1, -1, 0, BorderMode::Clamp, -1.0);
        assert_eq!(clamp, 0.5);
        
        let reflect = get_pixel_with_border(&image, -1, 5, 0, BorderMode::Reflect, -1.0);
        assert_eq!(reflect, 0.5);
        
        let wrap = get_pixel_with_border(&image, 15, 5, 0, BorderMode::Wrap, -1.0);
        assert_eq!(wrap, 0.5);
    }

    #[test]
    fn test_affine_image_transformation() {
        let image = create_pattern_image(32, 32);
        let transform = TransformMatrix::translation(5.0, 10.0);
        
        let result = apply_affine_transform(&image, &transform, None).unwrap();
        
        assert_eq!(result.width, image.width);
        assert_eq!(result.height, image.height);
        assert_eq!(result.channels, image.channels);
    }

    #[test]
    fn test_perspective_image_transformation() {
        let image = create_pattern_image(32, 32);
        let src_points = [(0.0, 0.0), (32.0, 0.0), (32.0, 32.0), (0.0, 32.0)];
        let dst_points = [(5.0, 5.0), (27.0, 5.0), (32.0, 32.0), (0.0, 27.0)];
        
        let transform = create_perspective_transform(&src_points, &dst_points).unwrap();
        let result = apply_perspective_transform(&image, &transform, None).unwrap();
        
        assert_eq!(result.width, image.width);
        assert_eq!(result.height, image.height);
    }

    #[test]
    fn test_keypoint_transformation() {
        let keypoints = vec![
            KeyPoint::new(10.0, 20.0),
            KeyPoint::new(30.0, 40.0),
        ];
        
        let transform = TransformMatrix::scaling(2.0, 2.0);
        let transformed = transform_keypoints(&keypoints, &transform);
        
        assert_eq!(transformed.len(), 2);
        assert_eq!(transformed[0].x, 20.0);
        assert_eq!(transformed[0].y, 40.0);
        assert_eq!(transformed[1].x, 60.0);
        assert_eq!(transformed[1].y, 80.0);
    }

    #[test]
    fn test_homography_estimation() {
        let src_points = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (5.0, 5.0)];
        let dst_points = [(1.0, 1.0), (11.0, 1.0), (11.0, 11.0), (1.0, 11.0), (6.0, 6.0)];
        
        let transform = estimate_homography_ransac(&src_points, &dst_points, None).unwrap();
        assert_eq!(transform.transform_type, "ransac_homography");
    }

    #[test]
    fn test_weight_functions() {
        // Test cubic weight function
        assert!((cubic_weight(0.0) - 1.0).abs() < 1e-6);
        assert!(cubic_weight(0.5).abs() > 0.0);
        assert_eq!(cubic_weight(3.0), 0.0);
        
        // Test Lanczos weight function
        assert!((lanczos_weight(0.0, 2.0) - 1.0).abs() < 1e-6);
        assert!(lanczos_weight(1.0, 2.0).abs() > 0.0);
        assert_eq!(lanczos_weight(3.0, 2.0), 0.0);
    }

    #[test]
    fn test_linear_system_solver() {
        let a = vec![
            vec![2.0, 1.0],
            vec![1.0, 1.0],
        ];
        let b = vec![3.0, 2.0];
        
        let solution = solve_linear_system(&a, &b).unwrap();
        assert_eq!(solution.len(), 2);
        
        // Verify solution: [1, 1]
        assert!((solution[0] - 1.0).abs() < 1e-6);
        assert!((solution[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_get_transform_matrix() {
        let translation = get_transform_matrix("translation", &[5.0, 10.0]).unwrap();
        assert_eq!(translation.transform_type, "translation");
        
        let rotation = get_transform_matrix("rotation", &[PI / 2.0]).unwrap();
        assert_eq!(rotation.transform_type, "rotation");
        
        let scaling = get_transform_matrix("scaling", &[2.0, 3.0]).unwrap();
        assert_eq!(scaling.transform_type, "scaling");
        
        let uniform_scaling = get_transform_matrix("scaling", &[2.0]).unwrap();
        assert_eq!(uniform_scaling.matrix[0][0], 2.0);
        assert_eq!(uniform_scaling.matrix[1][1], 2.0);
        
        let shear = get_transform_matrix("shear", &[0.5, 0.2]).unwrap();
        assert_eq!(shear.transform_type, "shear");
    }

    #[test]
    fn test_transform_params() {
        let params = TransformParams {
            interpolation: InterpolationMethod::Bicubic,
            border_mode: BorderMode::Reflect,
            fill_value: 0.5,
            output_size: Some((64, 64)),
        };
        
        assert_eq!(params.interpolation, InterpolationMethod::Bicubic);
        assert_eq!(params.border_mode, BorderMode::Reflect);
        assert_eq!(params.fill_value, 0.5);
        assert_eq!(params.output_size, Some((64, 64)));
    }

    #[test]
    fn test_tps_transformation() {
        let image = create_pattern_image(32, 32);
        let control_points = [(5.0, 5.0), (25.0, 5.0), (15.0, 25.0)];
        let target_points = [(7.0, 7.0), (23.0, 7.0), (15.0, 23.0)];
        
        let result = apply_thin_plate_spline_transform(&image, &control_points, &target_points, None).unwrap();
        
        assert_eq!(result.width, image.width);
        assert_eq!(result.height, image.height);
    }

    #[test]
    fn test_stdlib_functions() {
        let image = create_pattern_image(32, 32);
        let transform = TransformMatrix::translation(5.0, 10.0);
        
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let transform_value = Value::LyObj(LyObj::new(Box::new(transform)));
        
        let result = affine_transform(&[image_value, transform_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let transformed_image = obj.as_any().downcast_ref::<Image>().unwrap();
                assert_eq!(transformed_image.width, 32);
                assert_eq!(transformed_image.height, 32);
            }
            _ => panic!("Expected transformed image"),
        }
    }

    #[test]
    fn test_create_transform_stdlib() {
        let type_val = Value::Symbol("translation".to_string());
        let params_val = Value::List(vec![Value::Real(5.0), Value::Real(10.0)]);
        
        let result = create_transform(&[type_val, params_val]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let transform = obj.as_any().downcast_ref::<TransformMatrix>().unwrap();
                assert_eq!(transform.transform_type, "translation");
            }
            _ => panic!("Expected TransformMatrix"),
        }
    }

    #[test]
    fn test_extract_point_list() {
        let points = Value::List(vec![
            Value::List(vec![Value::Real(1.0), Value::Real(2.0)]),
            Value::List(vec![Value::Real(3.0), Value::Real(4.0)]),
        ]);
        
        let extracted = extract_point_list(&points).unwrap();
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0], (1.0, 2.0));
        assert_eq!(extracted[1], (3.0, 4.0));
    }

    #[test]
    fn test_invalid_arguments() {
        let result = affine_transform(&[]);
        assert!(result.is_err());
        
        let result = create_transform(&[Value::Integer(42)]);
        assert!(result.is_err());
        
        let result = get_transform_matrix("invalid_transform", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_singular_matrix_error() {
        let singular_matrix = TransformMatrix {
            matrix: [
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [1.0, 2.0, 3.0],
            ],
            transform_type: "singular".to_string(),
        };
        
        let result = invert_matrix(&singular_matrix);
        assert!(result.is_err());
    }

    #[test]
    fn test_perspective_transform_invalid_points() {
        let src_points = [(0.0, 0.0), (1.0, 0.0)]; // Only 2 points
        let dst_points = [(0.0, 0.0), (2.0, 0.0)];
        
        let result = create_perspective_transform(&src_points, &dst_points);
        assert!(result.is_err());
    }
}