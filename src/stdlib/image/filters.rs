//! Image Filtering & Enhancement Operations
//!
//! This module provides comprehensive image filtering and enhancement functions
//! including Gaussian filtering, median filtering, edge detection, and rotation.

use crate::foreign::{Foreign, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::core::{Image, ColorSpace, InterpolationMethod};
use std::f32::consts::PI;

// ===============================
// PHASE 6B: FILTERING & ENHANCEMENT (5 functions)
// ===============================

/// Apply Gaussian blur filter to image
/// Syntax: GaussianFilter[image, sigma, [kernelSize]]
pub fn gaussian_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, sigma, [kernelSize])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let sigma = match &args[1] {
        Value::Real(r) => *r as f32,
        Value::Integer(i) => *i as f32,
        _ => return Err(VmError::TypeError {
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let kernel_size = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => *i as usize,
            _ => (6.0 * sigma).ceil() as usize | 1, // Ensure odd size
        }
    } else {
        (6.0 * sigma).ceil() as usize | 1 // Ensure odd size
    };

    let filtered = image.gaussian_filter(sigma, kernel_size);
    Ok(Value::LyObj(LyObj::new(Box::new(filtered))))
}

/// Apply median filter to image for noise reduction
/// Syntax: MedianFilter[image, kernelSize]
pub fn median_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (image, kernelSize)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let kernel_size = match &args[1] {
        Value::Integer(i) => {
            let size = *i as usize;
            if size % 2 == 0 {
                return Err(VmError::TypeError {
                    expected: "Odd kernel size".to_string(),
                    actual: format!("Even kernel size: {}", size),
                });
            }
            size
        }
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let filtered = image.median_filter(kernel_size);
    Ok(Value::LyObj(LyObj::new(Box::new(filtered))))
}

/// Apply Sobel edge detection filter
/// Syntax: SobelFilter[image, [direction]]
pub fn sobel_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (image, [direction])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let direction = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => match s.as_str() {
                "Horizontal" | "X" => SobelDirection::Horizontal,
                "Vertical" | "Y" => SobelDirection::Vertical,
                "Both" | "Magnitude" => SobelDirection::Both,
                _ => SobelDirection::Both,
            },
            _ => SobelDirection::Both,
        }
    } else {
        SobelDirection::Both
    };

    let filtered = image.sobel_filter(direction);
    Ok(Value::LyObj(LyObj::new(Box::new(filtered))))
}

/// Apply Canny edge detection algorithm
/// Syntax: CannyEdgeDetection[image, lowThreshold, highThreshold, [sigma]]
pub fn canny_edge_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (image, lowThreshold, highThreshold, [sigma])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let low_threshold = match &args[1] {
        Value::Real(r) => *r as f32,
        Value::Integer(i) => *i as f32,
        _ => return Err(VmError::TypeError {
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let high_threshold = match &args[2] {
        Value::Real(r) => *r as f32,
        Value::Integer(i) => *i as f32,
        _ => return Err(VmError::TypeError {
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let sigma = if args.len() == 4 {
        match &args[3] {
            Value::Real(r) => *r as f32,
            Value::Integer(i) => *i as f32,
            _ => 1.0,
        }
    } else {
        1.0
    };

    let edges = image.canny_edge_detection(low_threshold, high_threshold, sigma);
    Ok(Value::LyObj(LyObj::new(Box::new(edges))))
}

/// Rotate image by specified angle with interpolation
/// Syntax: ImageRotate[image, angle, [interpolation], [backgroundColor]]
pub fn image_rotate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (image, angle, [interpolation], [backgroundColor])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<Image>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "Image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let angle = match &args[1] {
        Value::Real(r) => *r as f32,
        Value::Integer(i) => *i as f32,
        _ => return Err(VmError::TypeError {
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let interpolation = if args.len() >= 3 {
        match &args[2] {
            Value::String(s) => match s.as_str() {
                "NearestNeighbor" => InterpolationMethod::NearestNeighbor,
                "Bilinear" => InterpolationMethod::Bilinear,
                "Bicubic" => InterpolationMethod::Bicubic,
                "Lanczos" => InterpolationMethod::Lanczos,
                _ => InterpolationMethod::Bilinear,
            },
            _ => InterpolationMethod::Bilinear,
        }
    } else {
        InterpolationMethod::Bilinear
    };

    let background_color = if args.len() == 4 {
        match &args[3] {
            Value::Real(r) => *r as f32,
            Value::Integer(i) => *i as f32,
            _ => 0.0,
        }
    } else {
        0.0
    };

    let rotated = image.rotate(angle, interpolation, background_color);
    Ok(Value::LyObj(LyObj::new(Box::new(rotated))))
}

// ===============================
// HELPER TYPES AND IMPLEMENTATIONS
// ===============================

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SobelDirection {
    Horizontal,
    Vertical,
    Both,
}

// Implementation methods for Image struct
impl Image {
    /// Apply Gaussian blur filter
    pub fn gaussian_filter(&self, sigma: f32, kernel_size: usize) -> Image {
        if sigma <= 0.0 || kernel_size < 3 {
            return self.clone();
        }

        // Generate Gaussian kernel
        let kernel = generate_gaussian_kernel(sigma, kernel_size);
        
        // Apply separable filter (horizontal then vertical)
        let horizontal_filtered = self.apply_1d_filter(&kernel, true);
        horizontal_filtered.apply_1d_filter(&kernel, false)
    }

    /// Apply median filter for noise reduction
    pub fn median_filter(&self, kernel_size: usize) -> Image {
        if kernel_size < 3 || kernel_size % 2 == 0 {
            return self.clone();
        }

        let mut filtered_data = vec![0.0; self.data.len()];
        let half_kernel = kernel_size / 2;

        for y in 0..self.height {
            for x in 0..self.width {
                for c in 0..self.channels {
                    let mut values = Vec::new();
                    
                    // Collect neighborhood values
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let ny = y as i32 + ky as i32 - half_kernel as i32;
                            let nx = x as i32 + kx as i32 - half_kernel as i32;
                            
                            if ny >= 0 && ny < self.height as i32 && nx >= 0 && nx < self.width as i32 {
                                if let Some(value) = self.get_pixel(nx as usize, ny as usize, c) {
                                    values.push(value);
                                }
                            }
                        }
                    }
                    
                    // Find median
                    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let median = if values.is_empty() {
                        0.0
                    } else {
                        values[values.len() / 2]
                    };
                    
                    let index = (y * self.width + x) * self.channels + c;
                    filtered_data[index] = median;
                }
            }
        }

        Image::from_data(filtered_data, self.width, self.height, self.color_space, self.bit_depth)
    }

    /// Apply Sobel edge detection filter
    pub fn sobel_filter(&self, direction: SobelDirection) -> Image {
        // Convert to grayscale for edge detection
        let gray = if self.color_space != ColorSpace::Grayscale {
            self.to_grayscale()
        } else {
            self.clone()
        };

        let sobel_x = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
        let sobel_y = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

        let mut result_data = vec![0.0; gray.data.len()];

        for y in 1..gray.height - 1 {
            for x in 1..gray.width - 1 {
                let mut gx = 0.0;
                let mut gy = 0.0;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let ny = y + ky - 1;
                        let nx = x + kx - 1;
                        let pixel = gray.get_pixel(nx, ny, 0).unwrap_or(0.0);
                        let kernel_idx = ky * 3 + kx;
                        
                        gx += pixel * sobel_x[kernel_idx];
                        gy += pixel * sobel_y[kernel_idx];
                    }
                }

                let magnitude = match direction {
                    SobelDirection::Horizontal => gx.abs(),
                    SobelDirection::Vertical => gy.abs(),
                    SobelDirection::Both => (gx * gx + gy * gy).sqrt(),
                };

                let index = y * gray.width + x;
                result_data[index] = magnitude.clamp(0.0, 1.0);
            }
        }

        Image::from_data(result_data, gray.width, gray.height, ColorSpace::Grayscale, gray.bit_depth)
    }

    /// Apply Canny edge detection algorithm
    pub fn canny_edge_detection(&self, low_threshold: f32, high_threshold: f32, sigma: f32) -> Image {
        // Step 1: Apply Gaussian blur
        let blurred = self.gaussian_filter(sigma, (6.0 * sigma).ceil() as usize | 1);
        
        // Step 2: Apply Sobel filter to get gradients
        let grad = blurred.sobel_filter(SobelDirection::Both);
        
        // Step 3: Apply non-maximum suppression (simplified)
        let suppressed = grad.non_maximum_suppression();
        
        // Step 4: Apply double thresholding
        suppressed.double_threshold(low_threshold, high_threshold)
    }

    /// Rotate image by specified angle
    pub fn rotate(&self, angle_degrees: f32, interpolation: InterpolationMethod, background_color: f32) -> Image {
        let angle_rad = angle_degrees * PI / 180.0;
        let cos_angle = angle_rad.cos();
        let sin_angle = angle_rad.sin();

        // Calculate new image dimensions
        let corners = [
            (0.0, 0.0),
            (self.width as f32, 0.0),
            (0.0, self.height as f32),
            (self.width as f32, self.height as f32),
        ];

        let mut min_x = f32::INFINITY;
        let mut max_x = f32::NEG_INFINITY;
        let mut min_y = f32::INFINITY;
        let mut max_y = f32::NEG_INFINITY;

        for (x, y) in corners.iter() {
            let rotated_x = x * cos_angle - y * sin_angle;
            let rotated_y = x * sin_angle + y * cos_angle;
            
            min_x = min_x.min(rotated_x);
            max_x = max_x.max(rotated_x);
            min_y = min_y.min(rotated_y);
            max_y = max_y.max(rotated_y);
        }

        let new_width = (max_x - min_x).ceil() as usize;
        let new_height = (max_y - min_y).ceil() as usize;
        let center_x = self.width as f32 / 2.0;
        let center_y = self.height as f32 / 2.0;
        let new_center_x = new_width as f32 / 2.0;
        let new_center_y = new_height as f32 / 2.0;

        let mut rotated_data = vec![background_color; new_width * new_height * self.channels];

        for y in 0..new_height {
            for x in 0..new_width {
                // Translate to center, rotate, translate back
                let rel_x = x as f32 - new_center_x;
                let rel_y = y as f32 - new_center_y;
                
                let src_x = rel_x * cos_angle + rel_y * sin_angle + center_x;
                let src_y = -rel_x * sin_angle + rel_y * cos_angle + center_y;

                for c in 0..self.channels {
                    let pixel_value = match interpolation {
                        InterpolationMethod::NearestNeighbor => {
                            let x_idx = src_x.round() as i32;
                            let y_idx = src_y.round() as i32;
                            if x_idx >= 0 && x_idx < self.width as i32 && y_idx >= 0 && y_idx < self.height as i32 {
                                self.get_pixel(x_idx as usize, y_idx as usize, c).unwrap_or(background_color)
                            } else {
                                background_color
                            }
                        }
                        InterpolationMethod::Bilinear => {
                            if src_x >= 0.0 && src_x < self.width as f32 - 1.0 && src_y >= 0.0 && src_y < self.height as f32 - 1.0 {
                                self.bilinear_interpolate(src_x, src_y, c)
                            } else {
                                background_color
                            }
                        }
                        _ => {
                            // Use bilinear for bicubic and lanczos for now
                            if src_x >= 0.0 && src_x < self.width as f32 - 1.0 && src_y >= 0.0 && src_y < self.height as f32 - 1.0 {
                                self.bilinear_interpolate(src_x, src_y, c)
                            } else {
                                background_color
                            }
                        }
                    };

                    let dst_index = (y * new_width + x) * self.channels + c;
                    rotated_data[dst_index] = pixel_value;
                }
            }
        }

        Image::from_data(rotated_data, new_width, new_height, self.color_space, self.bit_depth)
    }

    // Helper methods for filtering operations
    fn apply_1d_filter(&self, kernel: &[f32], horizontal: bool) -> Image {
        let mut filtered_data = vec![0.0; self.data.len()];
        let half_kernel = kernel.len() / 2;

        for y in 0..self.height {
            for x in 0..self.width {
                for c in 0..self.channels {
                    let mut sum = 0.0;
                    
                    for k in 0..kernel.len() {
                        let (nx, ny) = if horizontal {
                            (x as i32 + k as i32 - half_kernel as i32, y as i32)
                        } else {
                            (x as i32, y as i32 + k as i32 - half_kernel as i32)
                        };
                        
                        if nx >= 0 && nx < self.width as i32 && ny >= 0 && ny < self.height as i32 {
                            if let Some(value) = self.get_pixel(nx as usize, ny as usize, c) {
                                sum += value * kernel[k];
                            }
                        }
                    }
                    
                    let index = (y * self.width + x) * self.channels + c;
                    filtered_data[index] = sum.clamp(0.0, 1.0);
                }
            }
        }

        Image::from_data(filtered_data, self.width, self.height, self.color_space, self.bit_depth)
    }

    fn non_maximum_suppression(&self) -> Image {
        // Simplified non-maximum suppression for Canny edge detection
        let mut suppressed_data = vec![0.0; self.data.len()];
        
        for y in 1..self.height - 1 {
            for x in 1..self.width - 1 {
                let current = self.get_pixel(x, y, 0).unwrap_or(0.0);
                let left = self.get_pixel(x - 1, y, 0).unwrap_or(0.0);
                let right = self.get_pixel(x + 1, y, 0).unwrap_or(0.0);
                let top = self.get_pixel(x, y - 1, 0).unwrap_or(0.0);
                let bottom = self.get_pixel(x, y + 1, 0).unwrap_or(0.0);
                
                // Check if current pixel is local maximum
                let is_maximum = current >= left && current >= right && current >= top && current >= bottom;
                
                let index = y * self.width + x;
                suppressed_data[index] = if is_maximum { current } else { 0.0 };
            }
        }

        Image::from_data(suppressed_data, self.width, self.height, ColorSpace::Grayscale, self.bit_depth)
    }

    fn double_threshold(&self, low_threshold: f32, high_threshold: f32) -> Image {
        let mut result_data = vec![0.0; self.data.len()];
        
        for y in 0..self.height {
            for x in 0..self.width {
                let pixel = self.get_pixel(x, y, 0).unwrap_or(0.0);
                let index = y * self.width + x;
                
                if pixel >= high_threshold {
                    result_data[index] = 1.0; // Strong edge
                } else if pixel >= low_threshold {
                    result_data[index] = 0.5; // Weak edge
                } else {
                    result_data[index] = 0.0; // Not an edge
                }
            }
        }

        Image::from_data(result_data, self.width, self.height, ColorSpace::Grayscale, self.bit_depth)
    }
}

// Generate Gaussian kernel for filtering
fn generate_gaussian_kernel(sigma: f32, size: usize) -> Vec<f32> {
    let mut kernel = vec![0.0; size];
    let center = size / 2;
    let two_sigma_squared = 2.0 * sigma * sigma;
    let mut sum = 0.0;

    for i in 0..size {
        let x = i as f32 - center as f32;
        let value = (-x * x / two_sigma_squared).exp();
        kernel[i] = value;
        sum += value;
    }

    // Normalize kernel
    for value in kernel.iter_mut() {
        *value /= sum;
    }

    kernel
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_filter_creation() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let filtered = image.gaussian_filter(1.0, 5);
        
        assert_eq!(filtered.width, 10);
        assert_eq!(filtered.height, 10);
        assert_eq!(filtered.channels, 3);
    }

    #[test]
    fn test_median_filter_creation() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let filtered = image.median_filter(3);
        
        assert_eq!(filtered.width, 10);
        assert_eq!(filtered.height, 10);
        assert_eq!(filtered.channels, 3);
    }

    #[test]
    fn test_sobel_filter_creation() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let edges = image.sobel_filter(SobelDirection::Both);
        
        assert_eq!(edges.width, 10);
        assert_eq!(edges.height, 10);
        assert_eq!(edges.channels, 1); // Sobel produces grayscale output
    }

    #[test]
    fn test_canny_edge_detection() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let edges = image.canny_edge_detection(0.1, 0.3, 1.0);
        
        assert_eq!(edges.width, 10);
        assert_eq!(edges.height, 10);
        assert_eq!(edges.channels, 1);
    }

    #[test]
    fn test_image_rotation() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let rotated = image.rotate(45.0, InterpolationMethod::Bilinear, 0.0);
        
        // Rotated image should have different dimensions
        assert!(rotated.width > 10 || rotated.height > 10);
        assert_eq!(rotated.channels, 3);
    }

    #[test]
    fn test_gaussian_kernel_generation() {
        let kernel = generate_gaussian_kernel(1.0, 5);
        assert_eq!(kernel.len(), 5);
        
        // Check kernel normalization
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check symmetry
        assert!((kernel[0] - kernel[4]).abs() < 1e-6);
        assert!((kernel[1] - kernel[3]).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_filter_function() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let sigma_value = Value::Real(1.0);
        
        let result = gaussian_filter(&[image_value, sigma_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let filtered = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(filtered.width, 10);
                assert_eq!(filtered.height, 10);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_median_filter_function() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let kernel_value = Value::Integer(3);
        
        let result = median_filter(&[image_value, kernel_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let filtered = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(filtered.width, 10);
                assert_eq!(filtered.height, 10);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_sobel_filter_function() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = sobel_filter(&[image_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let edges = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(edges.width, 10);
                assert_eq!(edges.height, 10);
                assert_eq!(edges.channels, 1);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_image_rotate_function() {
        let image = Image::new(10, 10, ColorSpace::RGB, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let angle_value = Value::Real(45.0);
        
        let result = image_rotate(&[image_value, angle_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let rotated = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(rotated.channels, 3);
            }
            _ => panic!("Expected Image object"),
        }
    }
}