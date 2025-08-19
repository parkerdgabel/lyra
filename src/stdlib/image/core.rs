//! Core Image Processing - Image Type and Basic Operations
//!
//! Provides the fundamental Image foreign object and basic operations
//! like import, export, info, resize, and histogram computation.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::path::Path;

/// Color space representation for images
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ColorSpace {
    Grayscale,
    RGB,
    RGBA,
    HSV,
    HSL,
    LAB,
    XYZ,
}

impl ColorSpace {
    pub fn channels(&self) -> usize {
        match self {
            ColorSpace::Grayscale => 1,
            ColorSpace::RGB | ColorSpace::HSV | ColorSpace::HSL | ColorSpace::LAB | ColorSpace::XYZ => 3,
            ColorSpace::RGBA => 4,
        }
    }
}

/// Interpolation method for image operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterpolationMethod {
    NearestNeighbor,
    Bilinear,
    Bicubic,
    Lanczos,
}

/// Core Image type implementing Foreign trait
#[derive(Debug, Clone, PartialEq)]
pub struct Image {
    pub data: Vec<f32>,           // Normalized pixel data [0.0, 1.0]
    pub width: usize,             // Image width in pixels
    pub height: usize,            // Image height in pixels
    pub channels: usize,          // Number of color channels
    pub color_space: ColorSpace,  // Color space representation
    pub bit_depth: usize,         // Original bit depth (8, 16, 32)
}

impl Image {
    /// Create a new image with given dimensions and color space
    pub fn new(width: usize, height: usize, color_space: ColorSpace, bit_depth: usize) -> Self {
        let channels = color_space.channels();
        let data = vec![0.0; width * height * channels];
        
        Image {
            data,
            width,
            height,
            channels,
            color_space,
            bit_depth,
        }
    }

    /// Create image from raw data
    pub fn from_data(data: Vec<f32>, width: usize, height: usize, color_space: ColorSpace, bit_depth: usize) -> Self {
        let channels = color_space.channels();
        assert_eq!(data.len(), width * height * channels, "Data size mismatch");
        
        Image {
            data,
            width,
            height,
            channels,
            color_space,
            bit_depth,
        }
    }

    /// Get pixel value at coordinates (x, y, channel)
    pub fn get_pixel(&self, x: usize, y: usize, channel: usize) -> Option<f32> {
        if x >= self.width || y >= self.height || channel >= self.channels {
            return None;
        }
        let index = (y * self.width + x) * self.channels + channel;
        self.data.get(index).copied()
    }

    /// Set pixel value at coordinates (x, y, channel)
    pub fn set_pixel(&mut self, x: usize, y: usize, channel: usize, value: f32) -> bool {
        if x >= self.width || y >= self.height || channel >= self.channels {
            return false;
        }
        let index = (y * self.width + x) * self.channels + channel;
        if let Some(pixel) = self.data.get_mut(index) {
            *pixel = value.clamp(0.0, 1.0);
            true
        } else {
            false
        }
    }

    /// Get image dimensions as (width, height)
    pub fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    /// Get total number of pixels
    pub fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Convert to grayscale using luminance weights
    pub fn to_grayscale(&self) -> Image {
        if self.color_space == ColorSpace::Grayscale {
            return self.clone();
        }

        let mut gray_data = Vec::with_capacity(self.width * self.height);
        
        for y in 0..self.height {
            for x in 0..self.width {
                let gray = match self.color_space {
                    ColorSpace::RGB | ColorSpace::RGBA => {
                        let r = self.get_pixel(x, y, 0).unwrap_or(0.0);
                        let g = self.get_pixel(x, y, 1).unwrap_or(0.0);
                        let b = self.get_pixel(x, y, 2).unwrap_or(0.0);
                        // Standard RGB to grayscale conversion
                        0.299 * r + 0.587 * g + 0.114 * b
                    }
                    ColorSpace::HSV => {
                        // Use Value component for grayscale
                        self.get_pixel(x, y, 2).unwrap_or(0.0)
                    }
                    _ => {
                        // For other color spaces, use first channel
                        self.get_pixel(x, y, 0).unwrap_or(0.0)
                    }
                };
                gray_data.push(gray);
            }
        }

        Image::from_data(gray_data, self.width, self.height, ColorSpace::Grayscale, self.bit_depth)
    }

    /// Resize image using specified interpolation method
    pub fn resize(&self, new_width: usize, new_height: usize, method: InterpolationMethod) -> Image {
        if new_width == self.width && new_height == self.height {
            return self.clone();
        }

        let mut resized_data = vec![0.0; new_width * new_height * self.channels];
        
        let x_scale = (self.width as f32) / (new_width as f32);
        let y_scale = (self.height as f32) / (new_height as f32);

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = x as f32 * x_scale;
                let src_y = y as f32 * y_scale;

                for c in 0..self.channels {
                    let pixel_value = match method {
                        InterpolationMethod::NearestNeighbor => {
                            let x_idx = src_x.round() as usize;
                            let y_idx = src_y.round() as usize;
                            self.get_pixel(x_idx, y_idx, c).unwrap_or(0.0)
                        }
                        InterpolationMethod::Bilinear => {
                            self.bilinear_interpolate(src_x, src_y, c)
                        }
                        InterpolationMethod::Bicubic => {
                            self.bicubic_interpolate(src_x, src_y, c)
                        }
                        InterpolationMethod::Lanczos => {
                            self.lanczos_interpolate(src_x, src_y, c)
                        }
                    };

                    let dst_index = (y * new_width + x) * self.channels + c;
                    resized_data[dst_index] = pixel_value;
                }
            }
        }

        Image::from_data(resized_data, new_width, new_height, self.color_space, self.bit_depth)
    }

    /// Compute histogram for specified channel
    pub fn histogram(&self, channel: usize, bins: usize) -> Option<Vec<u32>> {
        if channel >= self.channels || bins == 0 {
            return None;
        }

        let mut hist = vec![0u32; bins];
        let bin_scale = bins as f32;

        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(value) = self.get_pixel(x, y, channel) {
                    let bin_index = ((value * bin_scale) as usize).min(bins - 1);
                    hist[bin_index] += 1;
                }
            }
        }

        Some(hist)
    }

    // Public interpolation methods
    pub fn bilinear_interpolate(&self, x: f32, y: f32, channel: usize) -> f32 {
        let x1 = x.floor() as usize;
        let x2 = (x1 + 1).min(self.width - 1);
        let y1 = y.floor() as usize;
        let y2 = (y1 + 1).min(self.height - 1);

        let dx = x - x1 as f32;
        let dy = y - y1 as f32;

        let p11 = self.get_pixel(x1, y1, channel).unwrap_or(0.0);
        let p21 = self.get_pixel(x2, y1, channel).unwrap_or(0.0);
        let p12 = self.get_pixel(x1, y2, channel).unwrap_or(0.0);
        let p22 = self.get_pixel(x2, y2, channel).unwrap_or(0.0);

        let top = p11 * (1.0 - dx) + p21 * dx;
        let bottom = p12 * (1.0 - dx) + p22 * dx;
        
        top * (1.0 - dy) + bottom * dy
    }

    fn bicubic_interpolate(&self, x: f32, y: f32, channel: usize) -> f32 {
        // Simplified bicubic - use bilinear for now
        self.bilinear_interpolate(x, y, channel)
    }

    fn lanczos_interpolate(&self, x: f32, y: f32, channel: usize) -> f32 {
        // Simplified Lanczos - use bilinear for now  
        self.bilinear_interpolate(x, y, channel)
    }
}

impl Foreign for Image {
    fn type_name(&self) -> &'static str {
        "Image"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Dimensions" => {
                Ok(Value::List(vec![
                    Value::Integer(self.width as i64),
                    Value::Integer(self.height as i64)
                ]))
            }
            "Width" => Ok(Value::Integer(self.width as i64)),
            "Height" => Ok(Value::Integer(self.height as i64)),
            "Channels" => Ok(Value::Integer(self.channels as i64)),
            "ColorSpace" => Ok(Value::String(format!("{:?}", self.color_space))),
            "BitDepth" => Ok(Value::Integer(self.bit_depth as i64)),
            "PixelCount" => Ok(Value::Integer(self.pixel_count() as i64)),
            "GetPixel" => {
                if args.len() != 3 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 3,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let y = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };

                let channel = match &args[2] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[2]),
                    }),
                };

                match self.get_pixel(x, y, channel) {
                    Some(value) => Ok(Value::Real(value as f64)),
                    None => Err(ForeignError::IndexOutOfBounds {
                        index: format!("({}, {}, {})", x, y, channel),
                        bounds: format!("({}, {}, {})", self.width, self.height, self.channels),
                    }),
                }
            }
            "ToGrayscale" => {
                let gray_image = self.to_grayscale();
                Ok(Value::LyObj(LyObj::new(Box::new(gray_image))))
            }
            "Resize" => {
                if args.len() < 2 || args.len() > 3 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                let new_width = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let new_height = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };

                let method = if args.len() == 3 {
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

                let resized = self.resize(new_width, new_height, method);
                Ok(Value::LyObj(LyObj::new(Box::new(resized))))
            }
            "Histogram" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let channel = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let bins = if args.len() == 2 {
                    match &args[1] {
                        Value::Integer(i) => *i as usize,
                        _ => 256,
                    }
                } else {
                    256
                };

                match self.histogram(channel, bins) {
                    Some(hist) => {
                        let hist_values: Vec<Value> = hist.into_iter()
                            .map(|count| Value::Integer(count as i64))
                            .collect();
                        Ok(Value::List(hist_values))
                    }
                    None => Err(ForeignError::RuntimeError {
                        message: format!("Invalid channel {} or bins {}", channel, bins),
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

// ===============================
// PHASE 6A: CORE INFRASTRUCTURE (5 functions)
// ===============================

/// Create a test image for development and testing
pub fn image_import(_args: &[Value]) -> VmResult<Value> {
    // For now, create a simple test pattern
    // TODO: Implement actual file loading using image crate
    let width = 100;
    let height = 100;
    let mut data = Vec::with_capacity(width * height * 3);
    
    // Create a simple RGB test pattern
    for y in 0..height {
        for x in 0..width {
            let r = (x as f32) / (width as f32);
            let g = (y as f32) / (height as f32);
            let b = ((x + y) as f32) / ((width + height) as f32);
            
            data.push(r);
            data.push(g); 
            data.push(b);
        }
    }

    let image = Image::from_data(data, width, height, ColorSpace::RGB, 8);
    Ok(Value::LyObj(LyObj::new(Box::new(image))))
}

pub fn image_export(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (image, filepath)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let _image = match &args[0] {
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

    let _filepath = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // TODO: Implement actual file saving
    Ok(Value::Integer(1)) // Success indicator
}

pub fn image_info(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (image)".to_string(),
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

    let info = vec![
        ("Width".to_string(), Value::Integer(image.width as i64)),
        ("Height".to_string(), Value::Integer(image.height as i64)),
        ("Channels".to_string(), Value::Integer(image.channels as i64)),
        ("ColorSpace".to_string(), Value::String(format!("{:?}", image.color_space))),
        ("BitDepth".to_string(), Value::Integer(image.bit_depth as i64)),
        ("PixelCount".to_string(), Value::Integer(image.pixel_count() as i64)),
    ];

    // Convert to association list
    let assoc_list: Vec<Value> = info.into_iter()
        .map(|(k, v)| Value::List(vec![Value::String(k), v]))
        .collect();

    Ok(Value::List(assoc_list))
}

pub fn image_resize(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (image, width, height, [method])".to_string(),
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

    let width = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let height = match &args[2] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let method = if args.len() == 4 {
        match &args[3] {
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

    let resized = image.resize(width, height, method);
    Ok(Value::LyObj(LyObj::new(Box::new(resized))))
}

pub fn image_histogram(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (image, [channel], [bins])".to_string(),
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

    let channel = if args.len() >= 2 {
        match &args[1] {
            Value::Integer(i) => *i as usize,
            _ => 0,
        }
    } else {
        0
    };

    let bins = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => *i as usize,
            _ => 256,
        }
    } else {
        256
    };

    match image.histogram(channel, bins) {
        Some(hist) => {
            let hist_values: Vec<Value> = hist.into_iter()
                .map(|count| Value::Integer(count as i64))
                .collect();
            Ok(Value::List(hist_values))
        }
        None => Err(VmError::TypeError {
            expected: format!("valid channel (0-{})", image.channels - 1),
            actual: format!("channel {}", channel),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let image = Image::new(100, 100, ColorSpace::RGB, 8);
        assert_eq!(image.width, 100);
        assert_eq!(image.height, 100);
        assert_eq!(image.channels, 3);
        assert_eq!(image.color_space, ColorSpace::RGB);
        assert_eq!(image.data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_pixel_access() {
        let mut image = Image::new(10, 10, ColorSpace::RGB, 8);
        
        // Set a pixel
        assert!(image.set_pixel(5, 5, 0, 0.5));
        
        // Get the pixel
        assert_eq!(image.get_pixel(5, 5, 0), Some(0.5));
        
        // Out of bounds
        assert_eq!(image.get_pixel(10, 10, 0), None);
    }

    #[test]
    fn test_image_resize() {
        let image = Image::new(100, 100, ColorSpace::RGB, 8);
        let resized = image.resize(50, 50, InterpolationMethod::NearestNeighbor);
        
        assert_eq!(resized.width, 50);
        assert_eq!(resized.height, 50);
        assert_eq!(resized.channels, 3);
    }

    #[test]
    fn test_grayscale_conversion() {
        let mut image = Image::new(10, 10, ColorSpace::RGB, 8);
        
        // Set some RGB values
        image.set_pixel(0, 0, 0, 1.0); // Red
        image.set_pixel(0, 0, 1, 0.0); // Green
        image.set_pixel(0, 0, 2, 0.0); // Blue
        
        let gray = image.to_grayscale();
        assert_eq!(gray.channels, 1);
        assert_eq!(gray.color_space, ColorSpace::Grayscale);
        
        // Check luminance calculation (0.299 * 1.0 + 0.587 * 0.0 + 0.114 * 0.0 = 0.299)
        let gray_value = gray.get_pixel(0, 0, 0).unwrap();
        assert!((gray_value - 0.299).abs() < 1e-6);
    }

    #[test]
    fn test_histogram() {
        let mut image = Image::new(10, 10, ColorSpace::Grayscale, 8);
        
        // Fill with known values
        for x in 0..10 {
            for y in 0..10 {
                image.set_pixel(x, y, 0, if (x + y) % 2 == 0 { 0.0 } else { 1.0 });
            }
        }
        
        let hist = image.histogram(0, 2).unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0] + hist[1], 100); // Total pixels
    }

    #[test]
    fn test_image_import_function() {
        let result = image_import(&[]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let image = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(image.width, 100);
                assert_eq!(image.height, 100);
                assert_eq!(image.channels, 3);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_image_foreign_methods() {
        let image = Image::new(100, 100, ColorSpace::RGB, 8);
        
        // Test Dimensions method
        let dims = image.call_method("Dimensions", &[]).unwrap();
        match dims {
            Value::List(list) => {
                assert_eq!(list.len(), 2);
                assert_eq!(list[0], Value::Integer(100));
                assert_eq!(list[1], Value::Integer(100));
            }
            _ => panic!("Expected List"),
        }
        
        // Test Width method
        let width = image.call_method("Width", &[]).unwrap();
        assert_eq!(width, Value::Integer(100));
        
        // Test ColorSpace method
        let cs = image.call_method("ColorSpace", &[]).unwrap();
        assert_eq!(cs, Value::String("RGB".to_string()));
    }
}