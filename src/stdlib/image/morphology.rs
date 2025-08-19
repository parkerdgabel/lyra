//! Morphological Operations for Images
//!
//! This module provides fundamental morphological operations for binary and grayscale
//! image processing, including erosion, dilation, opening, and closing operations.

use crate::foreign::LyObj;
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::core::{Image, ColorSpace};

// ===============================
// STRUCTURING ELEMENTS
// ===============================

/// Predefined structuring element shapes
#[derive(Debug, Clone, PartialEq)]
pub enum StructuringElement {
    /// Circular structuring element with given radius
    Circle(usize),
    /// Rectangular structuring element (width, height)
    Rectangle(usize, usize),
    /// Cross-shaped structuring element with given size
    Cross(usize),
    /// Custom structuring element with explicit kernel
    Custom(Vec<Vec<bool>>),
}

impl StructuringElement {
    /// Generate the binary kernel for this structuring element
    pub fn to_kernel(&self) -> Vec<Vec<bool>> {
        match self {
            StructuringElement::Circle(radius) => generate_circle_kernel(*radius),
            StructuringElement::Rectangle(width, height) => generate_rectangle_kernel(*width, *height),
            StructuringElement::Cross(size) => generate_cross_kernel(*size),
            StructuringElement::Custom(kernel) => kernel.clone(),
        }
    }

    /// Get the dimensions of the structuring element
    pub fn dimensions(&self) -> (usize, usize) {
        let kernel = self.to_kernel();
        (kernel[0].len(), kernel.len())
    }

    /// Get the center point of the structuring element
    pub fn center(&self) -> (usize, usize) {
        let (width, height) = self.dimensions();
        (width / 2, height / 2)
    }
}

/// Generate a circular structuring element kernel
fn generate_circle_kernel(radius: usize) -> Vec<Vec<bool>> {
    let size = 2 * radius + 1;
    let center = radius as f32;
    let mut kernel = vec![vec![false; size]; size];
    
    for y in 0..size {
        for x in 0..size {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let distance = (dx * dx + dy * dy).sqrt();
            kernel[y][x] = distance <= radius as f32;
        }
    }
    
    kernel
}

/// Generate a rectangular structuring element kernel
fn generate_rectangle_kernel(width: usize, height: usize) -> Vec<Vec<bool>> {
    vec![vec![true; width]; height]
}

/// Generate a cross-shaped structuring element kernel
fn generate_cross_kernel(size: usize) -> Vec<Vec<bool>> {
    let mut kernel = vec![vec![false; size]; size];
    let center = size / 2;
    
    // Horizontal line
    for x in 0..size {
        kernel[center][x] = true;
    }
    
    // Vertical line
    for y in 0..size {
        kernel[y][center] = true;
    }
    
    kernel
}

// ===============================
// CORE MORPHOLOGICAL OPERATIONS
// ===============================

/// Morphological operation type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MorphOperation {
    Erosion,
    Dilation,
}

impl Image {
    /// Apply erosion morphological operation
    pub fn erosion(&self, structuring_element: &StructuringElement, iterations: usize) -> Image {
        let mut result = self.clone();
        for _ in 0..iterations {
            result = result.apply_morphological_operation(MorphOperation::Erosion, structuring_element);
        }
        result
    }

    /// Apply dilation morphological operation
    pub fn dilation(&self, structuring_element: &StructuringElement, iterations: usize) -> Image {
        let mut result = self.clone();
        for _ in 0..iterations {
            result = result.apply_morphological_operation(MorphOperation::Dilation, structuring_element);
        }
        result
    }

    /// Apply opening operation (erosion followed by dilation)
    pub fn opening(&self, structuring_element: &StructuringElement, iterations: usize) -> Image {
        let eroded = self.erosion(structuring_element, iterations);
        eroded.dilation(structuring_element, iterations)
    }

    /// Apply closing operation (dilation followed by erosion)
    pub fn closing(&self, structuring_element: &StructuringElement, iterations: usize) -> Image {
        let dilated = self.dilation(structuring_element, iterations);
        dilated.erosion(structuring_element, iterations)
    }

    /// Apply a single morphological operation
    fn apply_morphological_operation(&self, operation: MorphOperation, structuring_element: &StructuringElement) -> Image {
        let kernel = structuring_element.to_kernel();
        let (se_width, se_height) = (kernel[0].len(), kernel.len());
        let (center_x, center_y) = structuring_element.center();
        
        let mut result_data = vec![0.0; self.data.len()];
        
        for y in 0..self.height {
            for x in 0..self.width {
                for c in 0..self.channels {
                    let mut values = Vec::new();
                    
                    // Apply structuring element
                    for sy in 0..se_height {
                        for sx in 0..se_width {
                            if kernel[sy][sx] {
                                let img_x = x as i32 + sx as i32 - center_x as i32;
                                let img_y = y as i32 + sy as i32 - center_y as i32;
                                
                                if img_x >= 0 && img_x < self.width as i32 && img_y >= 0 && img_y < self.height as i32 {
                                    if let Some(value) = self.get_pixel(img_x as usize, img_y as usize, c) {
                                        values.push(value);
                                    }
                                }
                            }
                        }
                    }
                    
                    let result_value = if values.is_empty() {
                        match operation {
                            MorphOperation::Erosion => 1.0, // White for erosion when no values
                            MorphOperation::Dilation => 0.0, // Black for dilation when no values
                        }
                    } else {
                        match operation {
                            MorphOperation::Erosion => {
                                // Erosion: minimum value in neighborhood
                                values.iter().fold(f32::INFINITY, |a, &b| a.min(b))
                            }
                            MorphOperation::Dilation => {
                                // Dilation: maximum value in neighborhood
                                values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
                            }
                        }
                    };
                    
                    let index = (y * self.width + x) * self.channels + c;
                    result_data[index] = result_value.clamp(0.0, 1.0);
                }
            }
        }
        
        Image::from_data(result_data, self.width, self.height, self.color_space, self.bit_depth)
    }
}

// ===============================
// PHASE 6C: MORPHOLOGICAL OPERATIONS (4 functions)
// ===============================

/// Apply erosion morphological operation to image
/// Syntax: Erosion[image, structuringElement, [iterations]]
pub fn erosion(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, structuringElement, [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let structuring_element = parse_structuring_element(&args[1])?;
    let iterations = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => (*i as usize).max(1),
            _ => 1,
        }
    } else {
        1
    };

    let result = image.erosion(&structuring_element, iterations);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Apply dilation morphological operation to image
/// Syntax: Dilation[image, structuringElement, [iterations]]
pub fn dilation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, structuringElement, [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let structuring_element = parse_structuring_element(&args[1])?;
    let iterations = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => (*i as usize).max(1),
            _ => 1,
        }
    } else {
        1
    };

    let result = image.dilation(&structuring_element, iterations);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Apply opening morphological operation (erosion followed by dilation)
/// Syntax: Opening[image, structuringElement, [iterations]]
pub fn opening(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, structuringElement, [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let structuring_element = parse_structuring_element(&args[1])?;
    let iterations = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => (*i as usize).max(1),
            _ => 1,
        }
    } else {
        1
    };

    let result = image.opening(&structuring_element, iterations);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Apply closing morphological operation (dilation followed by erosion)
/// Syntax: Closing[image, structuringElement, [iterations]]
pub fn closing(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (image, structuringElement, [iterations])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let image = extract_image(&args[0])?;
    let structuring_element = parse_structuring_element(&args[1])?;
    let iterations = if args.len() == 3 {
        match &args[2] {
            Value::Integer(i) => (*i as usize).max(1),
            _ => 1,
        }
    } else {
        1
    };

    let result = image.closing(&structuring_element, iterations);
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
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

/// Parse structuring element from Value
fn parse_structuring_element(value: &Value) -> VmResult<StructuringElement> {
    match value {
        Value::String(s) => {
            match s.as_str() {
                "Circle" => Ok(StructuringElement::Circle(1)),
                "Rectangle" => Ok(StructuringElement::Rectangle(3, 3)),
                "Cross" => Ok(StructuringElement::Cross(3)),
                _ => Ok(StructuringElement::Circle(1)), // Default fallback
            }
        }
        Value::Integer(size) => {
            let size = (*size as usize).max(1);
            Ok(StructuringElement::Circle(size))
        }
        Value::List(elements) => {
            // Parse as custom structuring element
            if elements.is_empty() {
                return Ok(StructuringElement::Circle(1));
            }
            
            // Check if it's a list of lists (2D)
            let mut kernel = Vec::new();
            for row in elements {
                match row {
                    Value::List(row_elements) => {
                        let mut kernel_row = Vec::new();
                        for element in row_elements {
                            match element {
                                Value::Integer(i) => kernel_row.push(*i != 0),
                                Value::Real(r) => kernel_row.push(*r != 0.0),
                                _ => kernel_row.push(true),
                            }
                        }
                        kernel.push(kernel_row);
                    }
                    _ => {
                        // Treat as 1D list, create rectangular element
                        let size = elements.len();
                        return Ok(StructuringElement::Rectangle(size, 1));
                    }
                }
            }
            
            if kernel.is_empty() {
                Ok(StructuringElement::Circle(1))
            } else {
                Ok(StructuringElement::Custom(kernel))
            }
        }
        _ => Ok(StructuringElement::Circle(1)), // Default fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structuring_element_circle() {
        let circle = StructuringElement::Circle(1);
        let kernel = circle.to_kernel();
        
        // Should be 3x3 for radius 1
        assert_eq!(kernel.len(), 3);
        assert_eq!(kernel[0].len(), 3);
        
        // Center should be true
        assert!(kernel[1][1]);
        
        // Check dimensions
        assert_eq!(circle.dimensions(), (3, 3));
        assert_eq!(circle.center(), (1, 1));
    }

    #[test]
    fn test_structuring_element_rectangle() {
        let rect = StructuringElement::Rectangle(3, 2);
        let kernel = rect.to_kernel();
        
        assert_eq!(kernel.len(), 2);
        assert_eq!(kernel[0].len(), 3);
        
        // All elements should be true
        for row in &kernel {
            for &cell in row {
                assert!(cell);
            }
        }
    }

    #[test]
    fn test_structuring_element_cross() {
        let cross = StructuringElement::Cross(3);
        let kernel = cross.to_kernel();
        
        assert_eq!(kernel.len(), 3);
        assert_eq!(kernel[0].len(), 3);
        
        // Check cross pattern
        assert!(kernel[1][0]); // Left
        assert!(kernel[1][1]); // Center
        assert!(kernel[1][2]); // Right
        assert!(kernel[0][1]); // Top
        assert!(kernel[2][1]); // Bottom
        
        // Corners should be false
        assert!(!kernel[0][0]);
        assert!(!kernel[0][2]);
        assert!(!kernel[2][0]);
        assert!(!kernel[2][2]);
    }

    #[test]
    fn test_structuring_element_custom() {
        let kernel = vec![
            vec![true, false, true],
            vec![false, true, false],
            vec![true, false, true],
        ];
        let custom = StructuringElement::Custom(kernel.clone());
        
        assert_eq!(custom.to_kernel(), kernel);
        assert_eq!(custom.dimensions(), (3, 3));
    }

    #[test]
    fn test_image_erosion() {
        let mut image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        
        // Create a small white square in center
        for y in 1..4 {
            for x in 1..4 {
                image.set_pixel(x, y, 0, 1.0);
            }
        }
        
        let se = StructuringElement::Circle(1);
        let eroded = image.erosion(&se, 1);
        
        // Erosion should shrink the white region
        assert_eq!(eroded.width, 5);
        assert_eq!(eroded.height, 5);
        assert_eq!(eroded.channels, 1);
    }

    #[test]
    fn test_image_dilation() {
        let mut image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        
        // Create a single white pixel in center
        image.set_pixel(2, 2, 0, 1.0);
        
        let se = StructuringElement::Circle(1);
        let dilated = image.dilation(&se, 1);
        
        // Dilation should expand the white region
        assert_eq!(dilated.width, 5);
        assert_eq!(dilated.height, 5);
        assert_eq!(dilated.channels, 1);
        
        // Center should still be white
        assert_eq!(dilated.get_pixel(2, 2, 0), Some(1.0));
    }

    #[test]
    fn test_image_opening() {
        let mut image = Image::new(7, 7, ColorSpace::Grayscale, 8);
        
        // Create a shape with noise
        for y in 1..6 {
            for x in 1..6 {
                image.set_pixel(x, y, 0, 1.0);
            }
        }
        // Add noise pixel
        image.set_pixel(0, 0, 0, 1.0);
        
        let se = StructuringElement::Circle(1);
        let opened = image.opening(&se, 1);
        
        assert_eq!(opened.width, 7);
        assert_eq!(opened.height, 7);
        assert_eq!(opened.channels, 1);
    }

    #[test]
    fn test_image_closing() {
        let mut image = Image::new(7, 7, ColorSpace::Grayscale, 8);
        
        // Create a shape with a hole
        for y in 1..6 {
            for x in 1..6 {
                image.set_pixel(x, y, 0, 1.0);
            }
        }
        // Create hole
        image.set_pixel(3, 3, 0, 0.0);
        
        let se = StructuringElement::Circle(1);
        let closed = image.closing(&se, 1);
        
        assert_eq!(closed.width, 7);
        assert_eq!(closed.height, 7);
        assert_eq!(closed.channels, 1);
    }

    #[test]
    fn test_erosion_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let se_value = Value::String("Circle".to_string());
        
        let result = erosion(&[image_value, se_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let eroded = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(eroded.width, 5);
                assert_eq!(eroded.height, 5);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_dilation_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let se_value = Value::Integer(2);
        
        let result = dilation(&[image_value, se_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let dilated = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(dilated.width, 5);
                assert_eq!(dilated.height, 5);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_opening_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let se_value = Value::String("Cross".to_string());
        let iterations_value = Value::Integer(2);
        
        let result = opening(&[image_value, se_value, iterations_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let opened = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(opened.width, 5);
                assert_eq!(opened.height, 5);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_closing_function() {
        let image = Image::new(5, 5, ColorSpace::Grayscale, 8);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        let se_value = Value::String("Rectangle".to_string());
        
        let result = closing(&[image_value, se_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let closed = obj.downcast_ref::<Image>().unwrap();
                assert_eq!(closed.width, 5);
                assert_eq!(closed.height, 5);
            }
            _ => panic!("Expected Image object"),
        }
    }

    #[test]
    fn test_parse_structuring_element() {
        // Test string parsing
        let circle = parse_structuring_element(&Value::String("Circle".to_string())).unwrap();
        assert_eq!(circle, StructuringElement::Circle(1));
        
        // Test integer parsing
        let circle2 = parse_structuring_element(&Value::Integer(3)).unwrap();
        assert_eq!(circle2, StructuringElement::Circle(3));
        
        // Test custom kernel parsing
        let kernel_value = Value::List(vec![
            Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(1)]),
            Value::List(vec![Value::Integer(0), Value::Integer(1), Value::Integer(0)]),
            Value::List(vec![Value::Integer(1), Value::Integer(0), Value::Integer(1)]),
        ]);
        let custom = parse_structuring_element(&kernel_value).unwrap();
        
        let expected_kernel = vec![
            vec![true, false, true],
            vec![false, true, false],
            vec![true, false, true],
        ];
        assert_eq!(custom, StructuringElement::Custom(expected_kernel));
    }
}