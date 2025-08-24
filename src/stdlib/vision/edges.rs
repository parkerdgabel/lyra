//! Edge Detection Algorithms
//!
//! This module implements comprehensive edge detection algorithms including:
//! - Canny edge detection with hysteresis thresholding
//! - Sobel edge operators (Gx, Gy, magnitude, direction)
//! - Laplacian edge detection with zero-crossing
//! - Prewitt edge operators
//! - Roberts cross-gradient operators
//! - Scharr edge operators (optimized 3x3 kernels)

use crate::foreign::LyObj;
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::Image;
use super::EdgeMap;
use std::f32::consts::PI;

/// Canny edge detection parameters
#[derive(Debug, Clone)]
pub struct CannyParams {
    pub low_threshold: f32,    // Low threshold for hysteresis
    pub high_threshold: f32,   // High threshold for hysteresis
    pub sigma: f32,            // Gaussian blur sigma
    pub kernel_size: usize,    // Gaussian kernel size
    pub suppress_non_maximum: bool, // Apply non-maximum suppression
}

impl Default for CannyParams {
    fn default() -> Self {
        Self {
            low_threshold: 0.1,
            high_threshold: 0.3,
            sigma: 1.0,
            kernel_size: 5,
            suppress_non_maximum: true,
        }
    }
}

/// Sobel edge detection parameters
#[derive(Debug, Clone)]
pub struct SobelParams {
    pub return_magnitude: bool,  // Return gradient magnitude
    pub return_direction: bool,  // Return gradient direction
    pub normalize: bool,         // Normalize output to [0, 1]
    pub threshold: Option<f32>,  // Optional threshold for edge map
}

impl Default for SobelParams {
    fn default() -> Self {
        Self {
            return_magnitude: true,
            return_direction: false,
            normalize: true,
            threshold: None,
        }
    }
}

/// Laplacian edge detection parameters
#[derive(Debug, Clone)]
pub struct LaplacianParams {
    pub kernel_size: usize,      // Kernel size (3 or 5)
    pub zero_crossing: bool,     // Detect zero crossings
    pub threshold: f32,          // Threshold for zero crossing detection
    pub sigma: Option<f32>,      // Optional Gaussian blur before Laplacian
}

impl Default for LaplacianParams {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            zero_crossing: false,
            threshold: 0.01,
            sigma: None,
        }
    }
}

/// Detect edges using Canny edge detection algorithm
pub fn detect_canny_edges(image: &Image, params: Option<CannyParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Canny edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Step 1: Apply Gaussian blur to reduce noise
    let blurred = apply_gaussian_blur(image, params.sigma, params.kernel_size)?;
    
    // Step 2: Compute gradients using Sobel operators
    let (gradient_x, gradient_y) = compute_sobel_gradients(&blurred)?;
    
    // Step 3: Compute gradient magnitude and direction
    let (magnitude, direction) = compute_gradient_magnitude_direction(&gradient_x, &gradient_y)?;
    
    // Step 4: Apply non-maximum suppression
    let suppressed = if params.suppress_non_maximum {
        apply_non_maximum_suppression_canny(&magnitude, &direction)?
    } else {
        magnitude
    };
    
    // Step 5: Apply hysteresis thresholding
    let edges = apply_hysteresis_thresholding(&suppressed, params.low_threshold, params.high_threshold)?;
    
    let mut edge_map = EdgeMap::new(width, height, "canny".to_string());
    edge_map.edges = edges;
    edge_map.threshold_low = params.low_threshold;
    edge_map.threshold_high = params.high_threshold;
    
    Ok(edge_map)
}

/// Detect edges using Sobel operators
pub fn detect_sobel_edges(image: &Image, params: Option<SobelParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Sobel edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Compute gradients using Sobel operators
    let (gradient_x, gradient_y) = compute_sobel_gradients(image)?;
    
    let edges = if params.return_magnitude {
        // Compute gradient magnitude
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        if let Some(threshold) = params.threshold {
            apply_threshold(&mut magnitude, threshold);
        }
        
        magnitude
    } else if params.return_direction {
        // Compute gradient direction
        let mut direction = vec![0.0; width * height];
        for i in 0..direction.len() {
            direction[i] = gradient_y[i].atan2(gradient_x[i]);
            // Normalize to [0, 1] range
            direction[i] = (direction[i] + PI) / (2.0 * PI);
        }
        direction
    } else {
        // Return gradient magnitude by default
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        magnitude
    };
    
    let mut edge_map = EdgeMap::new(width, height, "sobel".to_string());
    edge_map.edges = edges;
    
    Ok(edge_map)
}

/// Detect edges using Laplacian operator
pub fn detect_laplacian_edges(image: &Image, params: Option<LaplacianParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Laplacian edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Optional Gaussian blur before Laplacian
    let input_image = if let Some(sigma) = params.sigma {
        apply_gaussian_blur(image, sigma, 5)?
    } else {
        image.clone()
    };
    
    // Apply Laplacian kernel
    let laplacian = apply_laplacian_kernel(&input_image, params.kernel_size)?;
    
    let edges = if params.zero_crossing {
        // Detect zero crossings
        detect_zero_crossings(&laplacian, params.threshold)
    } else {
        // Return absolute Laplacian values
        let mut abs_laplacian: Vec<f32> = laplacian.iter().map(|&x| x.abs()).collect();
        normalize_to_unit_range(&mut abs_laplacian);
        abs_laplacian
    };
    
    let mut edge_map = EdgeMap::new(width, height, "laplacian".to_string());
    edge_map.edges = edges;
    
    Ok(edge_map)
}

/// Detect edges using Prewitt operators
pub fn detect_prewitt_edges(image: &Image, params: Option<SobelParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Prewitt edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Compute gradients using Prewitt operators
    let (gradient_x, gradient_y) = compute_prewitt_gradients(image)?;
    
    let edges = if params.return_magnitude {
        // Compute gradient magnitude
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        if let Some(threshold) = params.threshold {
            apply_threshold(&mut magnitude, threshold);
        }
        
        magnitude
    } else if params.return_direction {
        // Compute gradient direction
        let mut direction = vec![0.0; width * height];
        for i in 0..direction.len() {
            direction[i] = gradient_y[i].atan2(gradient_x[i]);
            direction[i] = (direction[i] + PI) / (2.0 * PI);
        }
        direction
    } else {
        // Return gradient magnitude by default
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        magnitude
    };
    
    let mut edge_map = EdgeMap::new(width, height, "prewitt".to_string());
    edge_map.edges = edges;
    
    Ok(edge_map)
}

/// Detect edges using Roberts cross-gradient operators
pub fn detect_roberts_edges(image: &Image, params: Option<SobelParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Roberts edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Roberts cross-gradient kernels
    let roberts_x = [[1.0, 0.0], [0.0, -1.0]];
    let roberts_y = [[0.0, 1.0], [-1.0, 0.0]];
    
    let gradient_x = apply_2x2_kernel(image, &roberts_x)?;
    let gradient_y = apply_2x2_kernel(image, &roberts_y)?;
    
    let edges = if params.return_magnitude {
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        if let Some(threshold) = params.threshold {
            apply_threshold(&mut magnitude, threshold);
        }
        
        magnitude
    } else {
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        magnitude
    };
    
    let mut edge_map = EdgeMap::new(width, height, "roberts".to_string());
    edge_map.edges = edges;
    
    Ok(edge_map)
}

/// Detect edges using Scharr operators (optimized 3x3 kernels)
pub fn detect_scharr_edges(image: &Image, params: Option<SobelParams>) -> VmResult<EdgeMap> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Scharr edge detection requires grayscale image".to_string()));
    }
    
    let width = image.width;
    let height = image.height;
    
    // Compute gradients using Scharr operators
    let (gradient_x, gradient_y) = compute_scharr_gradients(image)?;
    
    let edges = if params.return_magnitude {
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        if let Some(threshold) = params.threshold {
            apply_threshold(&mut magnitude, threshold);
        }
        
        magnitude
    } else if params.return_direction {
        let mut direction = vec![0.0; width * height];
        for i in 0..direction.len() {
            direction[i] = gradient_y[i].atan2(gradient_x[i]);
            direction[i] = (direction[i] + PI) / (2.0 * PI);
        }
        direction
    } else {
        let mut magnitude = vec![0.0; width * height];
        for i in 0..magnitude.len() {
            magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        }
        
        if params.normalize {
            normalize_to_unit_range(&mut magnitude);
        }
        
        magnitude
    };
    
    let mut edge_map = EdgeMap::new(width, height, "scharr".to_string());
    edge_map.edges = edges;
    
    Ok(edge_map)
}

// Helper functions for edge detection implementation

/// Apply Gaussian blur to image
fn apply_gaussian_blur(image: &Image, sigma: f32, kernel_size: usize) -> VmResult<Image> {
    let kernel = generate_gaussian_kernel(kernel_size, sigma);
    
    // Apply separable Gaussian filtering
    let horizontal_filtered = convolve_horizontal(image, &kernel)?;
    convolve_vertical(&horizontal_filtered, &kernel)
}

/// Generate 1D Gaussian kernel
fn generate_gaussian_kernel(size: usize, sigma: f32) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = (size / 2) as i32;
    let mut sum = 0.0;
    
    for i in 0..size {
        let x = (i as i32 - center) as f32;
        let value = (-x * x / (2.0 * sigma * sigma)).exp() / (sigma * (2.0 * PI).sqrt());
        kernel[i] = value;
        sum += value;
    }
    
    // Normalize kernel
    for val in &mut kernel {
        *val /= sum;
    }
    
    kernel
}

/// Apply horizontal convolution
fn convolve_horizontal(image: &Image, kernel: &[f32]) -> VmResult<Image> {
    let mut result = image.clone();
    let kernel_radius = kernel.len() / 2;
    
    for y in 0..image.height {
        for x in 0..image.width {
            let mut sum = 0.0;
            
            for k in 0..kernel.len() {
                let pixel_x = x as i32 + k as i32 - kernel_radius as i32;
                if pixel_x >= 0 && pixel_x < image.width as i32 {
                    sum += image.data[y * image.width + pixel_x as usize] * kernel[k];
                }
            }
            
            result.data[y * image.width + x] = sum;
        }
    }
    
    Ok(result)
}

/// Apply vertical convolution
fn convolve_vertical(image: &Image, kernel: &[f32]) -> VmResult<Image> {
    let mut result = image.clone();
    let kernel_radius = kernel.len() / 2;
    
    for y in 0..image.height {
        for x in 0..image.width {
            let mut sum = 0.0;
            
            for k in 0..kernel.len() {
                let pixel_y = y as i32 + k as i32 - kernel_radius as i32;
                if pixel_y >= 0 && pixel_y < image.height as i32 {
                    sum += image.data[pixel_y as usize * image.width + x] * kernel[k];
                }
            }
            
            result.data[y * image.width + x] = sum;
        }
    }
    
    Ok(result)
}

/// Compute Sobel gradients (Gx and Gy)
fn compute_sobel_gradients(image: &Image) -> VmResult<(Vec<f32>, Vec<f32>)> {
    let width = image.width;
    let height = image.height;
    
    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    
    let mut gradient_x = vec![0.0; width * height];
    let mut gradient_y = vec![0.0; width * height];
    
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            // Apply 3x3 kernels
            for dy in 0..3 {
                for dx in 0..3 {
                    let pixel_idx = (y + dy - 1) * width + (x + dx - 1);
                    let pixel = image.data[pixel_idx];
                    
                    gx += sobel_x[dy][dx] * pixel;
                    gy += sobel_y[dy][dx] * pixel;
                }
            }
            
            gradient_x[idx] = gx;
            gradient_y[idx] = gy;
        }
    }
    
    Ok((gradient_x, gradient_y))
}

/// Compute Prewitt gradients
fn compute_prewitt_gradients(image: &Image) -> VmResult<(Vec<f32>, Vec<f32>)> {
    let width = image.width;
    let height = image.height;
    
    // Prewitt kernels
    let prewitt_x = [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]];
    let prewitt_y = [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]];
    
    let mut gradient_x = vec![0.0; width * height];
    let mut gradient_y = vec![0.0; width * height];
    
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            for dy in 0..3 {
                for dx in 0..3 {
                    let pixel_idx = (y + dy - 1) * width + (x + dx - 1);
                    let pixel = image.data[pixel_idx];
                    
                    gx += prewitt_x[dy][dx] * pixel;
                    gy += prewitt_y[dy][dx] * pixel;
                }
            }
            
            gradient_x[idx] = gx;
            gradient_y[idx] = gy;
        }
    }
    
    Ok((gradient_x, gradient_y))
}

/// Compute Scharr gradients (optimized 3x3 kernels)
fn compute_scharr_gradients(image: &Image) -> VmResult<(Vec<f32>, Vec<f32>)> {
    let width = image.width;
    let height = image.height;
    
    // Scharr kernels (optimized for rotation invariance)
    let scharr_x = [[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]];
    let scharr_y = [[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]];
    
    let mut gradient_x = vec![0.0; width * height];
    let mut gradient_y = vec![0.0; width * height];
    
    for y in 1..height - 1 {
        for x in 1..width - 1 {
            let idx = y * width + x;
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            for dy in 0..3 {
                for dx in 0..3 {
                    let pixel_idx = (y + dy - 1) * width + (x + dx - 1);
                    let pixel = image.data[pixel_idx];
                    
                    gx += scharr_x[dy][dx] * pixel;
                    gy += scharr_y[dy][dx] * pixel;
                }
            }
            
            gradient_x[idx] = gx / 32.0; // Normalize by kernel sum
            gradient_y[idx] = gy / 32.0;
        }
    }
    
    Ok((gradient_x, gradient_y))
}

/// Apply 2x2 kernel (for Roberts operator)
fn apply_2x2_kernel(image: &Image, kernel: &[[f32; 2]; 2]) -> VmResult<Vec<f32>> {
    let width = image.width;
    let height = image.height;
    let mut result = vec![0.0; width * height];
    
    for y in 0..height - 1 {
        for x in 0..width - 1 {
            let idx = y * width + x;
            let mut sum = 0.0;
            
            for dy in 0..2 {
                for dx in 0..2 {
                    let pixel_idx = (y + dy) * width + (x + dx);
                    sum += image.data[pixel_idx] * kernel[dy][dx];
                }
            }
            
            result[idx] = sum;
        }
    }
    
    Ok(result)
}

/// Compute gradient magnitude and direction
fn compute_gradient_magnitude_direction(gradient_x: &[f32], gradient_y: &[f32]) -> VmResult<(Vec<f32>, Vec<f32>)> {
    if gradient_x.len() != gradient_y.len() {
        return Err(VmError::Runtime("Gradient arrays must have same length".to_string()));
    }
    
    let mut magnitude = vec![0.0; gradient_x.len()];
    let mut direction = vec![0.0; gradient_x.len()];
    
    for i in 0..gradient_x.len() {
        magnitude[i] = (gradient_x[i].powi(2) + gradient_y[i].powi(2)).sqrt();
        direction[i] = gradient_y[i].atan2(gradient_x[i]);
    }
    
    Ok((magnitude, direction))
}

/// Apply non-maximum suppression for Canny edge detection
fn apply_non_maximum_suppression_canny(magnitude: &[f32], direction: &[f32]) -> VmResult<Vec<f32>> {
    let len = magnitude.len();
    let mut suppressed = magnitude.to_vec();
    
    // Note: This is a simplified version. A complete implementation would
    // need to know image dimensions and handle edge pixels properly.
    // For now, we'll just apply a simple suppression.
    
    for i in 1..len - 1 {
        let angle = direction[i];
        
        // Quantize angle to 0, 45, 90, 135 degrees
        let quantized_angle = ((angle * 4.0 / PI + 0.5) as i32) % 4;
        
        // Check neighbors based on gradient direction
        let (prev_idx, next_idx) = match quantized_angle {
            0 => (i - 1, i + 1),     // Horizontal
            1 => (i - 1, i + 1),     // Diagonal (simplified)
            2 => (i - 1, i + 1),     // Vertical (simplified)
            3 => (i - 1, i + 1),     // Diagonal (simplified)
            _ => (i - 1, i + 1),
        };
        
        if prev_idx < len && next_idx < len {
            if magnitude[i] < magnitude[prev_idx] || magnitude[i] < magnitude[next_idx] {
                suppressed[i] = 0.0;
            }
        }
    }
    
    Ok(suppressed)
}

/// Apply hysteresis thresholding for Canny edge detection
fn apply_hysteresis_thresholding(magnitude: &[f32], low_threshold: f32, high_threshold: f32) -> VmResult<Vec<f32>> {
    let len = magnitude.len();
    let mut edges = vec![0.0; len];
    let mut visited = vec![false; len];
    
    // Mark strong edges
    for i in 0..len {
        if magnitude[i] >= high_threshold {
            edges[i] = 1.0;
        }
    }
    
    // Connect weak edges to strong edges
    for i in 0..len {
        if !visited[i] && magnitude[i] >= low_threshold && magnitude[i] < high_threshold {
            if has_strong_neighbor(i, &magnitude, high_threshold, len) {
                edges[i] = 1.0;
            }
            visited[i] = true;
        }
    }
    
    Ok(edges)
}

/// Check if pixel has a strong edge neighbor
fn has_strong_neighbor(idx: usize, magnitude: &[f32], high_threshold: f32, len: usize) -> bool {
    // Simplified neighbor checking - in a real implementation,
    // this would need proper 2D image dimensions
    let neighbors = [idx.saturating_sub(1), (idx + 1).min(len - 1)];
    
    for &neighbor_idx in &neighbors {
        if neighbor_idx < len && magnitude[neighbor_idx] >= high_threshold {
            return true;
        }
    }
    
    false
}

/// Apply Laplacian kernel
fn apply_laplacian_kernel(image: &Image, kernel_size: usize) -> VmResult<Vec<f32>> {
    let width = image.width;
    let height = image.height;
    
    let kernel = match kernel_size {
        3 => vec![
            vec![0.0, -1.0, 0.0],
            vec![-1.0, 4.0, -1.0],
            vec![0.0, -1.0, 0.0]
        ],
        5 => vec![
            vec![0.0, 0.0, -1.0, 0.0, 0.0],
            vec![0.0, -1.0, -2.0, -1.0, 0.0],
            vec![-1.0, -2.0, 16.0, -2.0, -1.0],
            vec![0.0, -1.0, -2.0, -1.0, 0.0],
            vec![0.0, 0.0, -1.0, 0.0, 0.0]
        ],
        _ => return Err(VmError::Runtime("Laplacian kernel size must be 3 or 5".to_string())),
    };
    
    let mut result = vec![0.0; width * height];
    let radius = kernel_size / 2;
    
    for y in radius..height - radius {
        for x in radius..width - radius {
            let mut sum = 0.0;
            
            for dy in 0..kernel_size {
                for dx in 0..kernel_size {
                    let pixel_idx = (y + dy - radius) * width + (x + dx - radius);
                    sum += image.data[pixel_idx] * kernel[dy][dx];
                }
            }
            
            result[y * width + x] = sum;
        }
    }
    
    Ok(result)
}

/// Detect zero crossings in Laplacian image
fn detect_zero_crossings(laplacian: &[f32], threshold: f32) -> Vec<f32> {
    let len = laplacian.len();
    let mut crossings = vec![0.0; len];
    
    // Simplified zero crossing detection
    for i in 1..len - 1 {
        let curr = laplacian[i];
        let prev = laplacian[i - 1];
        let next = laplacian[i + 1];
        
        // Check for zero crossing (sign change)
        if (curr * prev < 0.0 || curr * next < 0.0) && curr.abs() > threshold {
            crossings[i] = 1.0;
        }
    }
    
    crossings
}

/// Normalize values to [0, 1] range
fn normalize_to_unit_range(values: &mut [f32]) {
    if values.is_empty() {
        return;
    }
    
    let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    if max_val > min_val {
        let range = max_val - min_val;
        for val in values.iter_mut() {
            *val = (*val - min_val) / range;
        }
    }
}

/// Apply threshold to edge map
fn apply_threshold(values: &mut [f32], threshold: f32) {
    for val in values.iter_mut() {
        *val = if *val >= threshold { 1.0 } else { 0.0 };
    }
}

// Stdlib function implementations

/// Canny edge detection - CannyEdges[image, opts]
pub fn canny_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_canny_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = CannyParams::default();
            
            let edge_map = detect_canny_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("CannyEdges expects 1 or 2 arguments".to_string())),
    }
}

/// Sobel edge detection - SobelEdges[image, opts]
pub fn sobel_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_sobel_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = SobelParams::default();
            
            let edge_map = detect_sobel_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("SobelEdges expects 1 or 2 arguments".to_string())),
    }
}

/// Laplacian edge detection - LaplacianEdges[image, opts]
pub fn laplacian_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_laplacian_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = LaplacianParams::default();
            
            let edge_map = detect_laplacian_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("LaplacianEdges expects 1 or 2 arguments".to_string())),
    }
}

/// Prewitt edge detection - PrewittEdges[image, opts]
pub fn prewitt_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_prewitt_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = SobelParams::default();
            
            let edge_map = detect_prewitt_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("PrewittEdges expects 1 or 2 arguments".to_string())),
    }
}

/// Roberts edge detection - RobertsEdges[image, opts]
pub fn roberts_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_roberts_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = SobelParams::default();
            
            let edge_map = detect_roberts_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("RobertsEdges expects 1 or 2 arguments".to_string())),
    }
}

/// Scharr edge detection - ScharrEdges[image, opts]
pub fn scharr_edges(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
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
            
            let edge_map = detect_scharr_edges(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        [img, _opts] => {
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
            
            // Parse options (simplified)
            let params = SobelParams::default();
            
            let edge_map = detect_scharr_edges(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("edges".to_string(), Value::List(edge_map.edges.iter().cloned().map(|v| Value::Real(v as f64)).collect()));
            m.insert("width".to_string(), Value::Integer(edge_map.width as i64));
            m.insert("height".to_string(), Value::Integer(edge_map.height as i64));
            m.insert("algorithm".to_string(), Value::String(edge_map.algorithm.clone()));
            m.insert("thresholdLow".to_string(), Value::Real(edge_map.threshold_low as f64));
            m.insert("thresholdHigh".to_string(), Value::Real(edge_map.threshold_high as f64));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("ScharrEdges expects 1 or 2 arguments".to_string())),
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

    /// Create a test image with edges
    fn create_edge_image(width: usize, height: usize) -> Image {
        let mut image = create_test_image(width, height);
        
        // Create vertical edge in the middle
        for y in 0..height {
            for x in 0..width / 2 {
                image.data[y * width + x] = 0.0; // Left side black
            }
            for x in width / 2..width {
                image.data[y * width + x] = 1.0; // Right side white
            }
        }
        
        image
    }

    /// Create a test image with horizontal edge
    fn create_horizontal_edge_image(width: usize, height: usize) -> Image {
        let mut image = create_test_image(width, height);
        
        // Create horizontal edge in the middle
        for y in 0..height / 2 {
            for x in 0..width {
                image.data[y * width + x] = 0.0; // Top half black
            }
        }
        for y in height / 2..height {
            for x in 0..width {
                image.data[y * width + x] = 1.0; // Bottom half white
            }
        }
        
        image
    }

    #[test]
    fn test_canny_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_canny_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "canny");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
        assert_eq!(edge_map.edges.len(), 64 * 64);
        
        // Should detect the vertical edge
        let edge_count: usize = edge_map.edges.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).sum();
        assert!(edge_count > 0, "Should detect some edges");
    }

    #[test]
    fn test_canny_with_custom_params() {
        let image = create_edge_image(64, 64);
        let params = CannyParams {
            low_threshold: 0.05,
            high_threshold: 0.15,
            sigma: 1.5,
            kernel_size: 7,
            suppress_non_maximum: true,
        };
        
        let edge_map = detect_canny_edges(&image, Some(params)).unwrap();
        assert_eq!(edge_map.algorithm, "canny");
        assert_eq!(edge_map.threshold_low, 0.05);
        assert_eq!(edge_map.threshold_high, 0.15);
    }

    #[test]
    fn test_sobel_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_sobel_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "sobel");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
        
        // Should detect the vertical edge
        let edge_count: usize = edge_map.edges.iter().map(|&x| if x > 0.1 { 1 } else { 0 }).sum();
        assert!(edge_count > 0, "Should detect some edges");
    }

    #[test]
    fn test_sobel_gradient_computation() {
        let image = create_edge_image(32, 32);
        let (gradient_x, gradient_y) = compute_sobel_gradients(&image).unwrap();
        
        assert_eq!(gradient_x.len(), 32 * 32);
        assert_eq!(gradient_y.len(), 32 * 32);
        
        // For vertical edge, gradient_x should be non-zero
        let gx_sum: f32 = gradient_x.iter().map(|&x| x.abs()).sum();
        assert!(gx_sum > 0.0, "Should have non-zero x-gradient");
    }

    #[test]
    fn test_laplacian_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_laplacian_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "laplacian");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
    }

    #[test]
    fn test_laplacian_with_zero_crossing() {
        let image = create_edge_image(64, 64);
        let params = LaplacianParams {
            kernel_size: 3,
            zero_crossing: true,
            threshold: 0.01,
            sigma: Some(1.0),
        };
        
        let edge_map = detect_laplacian_edges(&image, Some(params)).unwrap();
        assert_eq!(edge_map.algorithm, "laplacian");
    }

    #[test]
    fn test_prewitt_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_prewitt_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "prewitt");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
    }

    #[test]
    fn test_roberts_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_roberts_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "roberts");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
    }

    #[test]
    fn test_scharr_edge_detection() {
        let image = create_edge_image(64, 64);
        let edge_map = detect_scharr_edges(&image, None).unwrap();
        
        assert_eq!(edge_map.algorithm, "scharr");
        assert_eq!(edge_map.width, 64);
        assert_eq!(edge_map.height, 64);
    }

    #[test]
    fn test_gradient_computation_functions() {
        let image = create_horizontal_edge_image(32, 32);
        
        // Test Sobel gradients
        let (sobel_x, sobel_y) = compute_sobel_gradients(&image).unwrap();
        let gy_sum: f32 = sobel_y.iter().map(|&x| x.abs()).sum();
        assert!(gy_sum > 0.0, "Should have non-zero y-gradient for horizontal edge");
        
        // Test Prewitt gradients
        let (prewitt_x, prewitt_y) = compute_prewitt_gradients(&image).unwrap();
        assert_eq!(prewitt_x.len(), 32 * 32);
        assert_eq!(prewitt_y.len(), 32 * 32);
        
        // Test Scharr gradients
        let (scharr_x, scharr_y) = compute_scharr_gradients(&image).unwrap();
        assert_eq!(scharr_x.len(), 32 * 32);
        assert_eq!(scharr_y.len(), 32 * 32);
    }

    #[test]
    fn test_edge_map_operations() {
        let edge_map = EdgeMap::new(32, 32, "test".to_string());
        
        assert_eq!(edge_map.width, 32);
        assert_eq!(edge_map.height, 32);
        assert_eq!(edge_map.algorithm, "test");
        assert_eq!(edge_map.edges.len(), 32 * 32);
        
        // Test pixel access
        let pixel_value = edge_map.get_pixel(15, 15);
        assert_eq!(pixel_value, 0.0);
        
        // Test out-of-bounds access
        let oob_pixel = edge_map.get_pixel(50, 50);
        assert_eq!(oob_pixel, 0.0);
    }

    #[test]
    fn test_edge_map_pixel_operations() {
        let mut edge_map = EdgeMap::new(10, 10, "test".to_string());
        
        edge_map.set_pixel(5, 5, 0.8);
        assert_eq!(edge_map.get_pixel(5, 5), 0.8);
        
        // Test clamping
        edge_map.set_pixel(3, 3, 1.5);
        assert_eq!(edge_map.get_pixel(3, 3), 1.0);
        
        edge_map.set_pixel(7, 7, -0.5);
        assert_eq!(edge_map.get_pixel(7, 7), 0.0);
    }

    #[test]
    fn test_gaussian_kernel_generation() {
        let kernel = generate_gaussian_kernel(5, 1.0);
        assert_eq!(kernel.len(), 5);
        
        // Check normalization
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check symmetry
        assert!((kernel[0] - kernel[4]).abs() < 1e-6);
        assert!((kernel[1] - kernel[3]).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_blur() {
        let image = create_test_image(32, 32);
        let blurred = apply_gaussian_blur(&image, 1.0, 5).unwrap();
        
        assert_eq!(blurred.width, image.width);
        assert_eq!(blurred.height, image.height);
        assert_eq!(blurred.data.len(), image.data.len());
    }

    #[test]
    fn test_normalize_to_unit_range() {
        let mut values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        normalize_to_unit_range(&mut values);
        
        // Check range
        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        assert!((min_val - 0.0).abs() < 1e-6);
        assert!((max_val - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_threshold_application() {
        let mut values = vec![0.1, 0.3, 0.5, 0.7, 0.9];
        apply_threshold(&mut values, 0.5);
        
        assert_eq!(values, vec![0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_laplacian_kernel_application() {
        let image = create_test_image(16, 16);
        let laplacian_3 = apply_laplacian_kernel(&image, 3).unwrap();
        let laplacian_5 = apply_laplacian_kernel(&image, 5).unwrap();
        
        assert_eq!(laplacian_3.len(), 16 * 16);
        assert_eq!(laplacian_5.len(), 16 * 16);
    }

    #[test]
    fn test_zero_crossing_detection() {
        let laplacian = vec![-1.0, -0.5, 0.1, 0.5, -0.2, 0.3, -0.1];
        let crossings = detect_zero_crossings(&laplacian, 0.05);
        
        assert_eq!(crossings.len(), laplacian.len());
        
        // Should detect zero crossings where sign changes and magnitude > threshold
        let crossing_count: usize = crossings.iter().map(|&x| if x > 0.5 { 1 } else { 0 }).sum();
        assert!(crossing_count > 0, "Should detect some zero crossings");
    }

    #[test]
    fn test_2x2_kernel_application() {
        let image = create_test_image(16, 16);
        let roberts_x = [[1.0, 0.0], [0.0, -1.0]];
        let result = apply_2x2_kernel(&image, &roberts_x).unwrap();
        
        assert_eq!(result.len(), 16 * 16);
    }

    #[test]
    fn test_gradient_magnitude_direction() {
        let gradient_x = vec![1.0, 0.0, -1.0];
        let gradient_y = vec![0.0, 1.0, 0.0];
        
        let (magnitude, direction) = compute_gradient_magnitude_direction(&gradient_x, &gradient_y).unwrap();
        
        assert_eq!(magnitude.len(), 3);
        assert_eq!(direction.len(), 3);
        
        assert!((magnitude[0] - 1.0).abs() < 1e-6);
        assert!((magnitude[1] - 1.0).abs() < 1e-6);
        assert!((magnitude[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_canny_stdlib_function() {
        let image = create_edge_image(64, 64);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = canny_edges(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let edge_map = obj.as_any().downcast_ref::<EdgeMap>().unwrap();
                assert_eq!(edge_map.algorithm, "canny");
            }
            _ => panic!("Expected EdgeMap"),
        }
    }

    #[test]
    fn test_sobel_stdlib_function() {
        let image = create_edge_image(64, 64);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = sobel_edges(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let edge_map = obj.as_any().downcast_ref::<EdgeMap>().unwrap();
                assert_eq!(edge_map.algorithm, "sobel");
            }
            _ => panic!("Expected EdgeMap"),
        }
    }

    #[test]
    fn test_laplacian_stdlib_function() {
        let image = create_edge_image(64, 64);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = laplacian_edges(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let edge_map = obj.as_any().downcast_ref::<EdgeMap>().unwrap();
                assert_eq!(edge_map.algorithm, "laplacian");
            }
            _ => panic!("Expected EdgeMap"),
        }
    }

    #[test]
    fn test_invalid_arguments() {
        let result = canny_edges(&[]);
        assert!(result.is_err());
        
        let result = canny_edges(&[Value::Integer(42)]);
        assert!(result.is_err());
        
        let result = sobel_edges(&[Value::Real(3.14)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_non_grayscale_image_error() {
        let mut image = create_test_image(32, 32);
        image.channels = 3; // RGB instead of grayscale
        
        let result = detect_canny_edges(&image, None);
        assert!(result.is_err());
        
        let result = detect_sobel_edges(&image, None);
        assert!(result.is_err());
        
        let result = detect_laplacian_edges(&image, None);
        assert!(result.is_err());
    }
}
