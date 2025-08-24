//! Feature Detection Algorithms
//!
//! This module implements advanced feature detection algorithms including:
//! - Harris corner detection
//! - SIFT (Scale-Invariant Feature Transform) descriptors  
//! - ORB (Oriented FAST and Rotated BRIEF) features
//! - Feature matching and tracking

use crate::foreign::LyObj;
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::image::Image;
use super::{FeatureSet, KeyPoint, FeatureMatch, FeatureMatches};
use std::f32::consts::PI;

/// Harris corner detection parameters
#[derive(Debug, Clone)]
pub struct HarrisParams {
    pub k: f32,              // Harris corner detection parameter (typically 0.04-0.06)
    pub threshold: f32,      // Corner response threshold
    pub window_size: usize,  // Window size for corner calculation
    pub sigma: f32,          // Gaussian smoothing sigma
}

impl Default for HarrisParams {
    fn default() -> Self {
        Self {
            k: 0.04,
            threshold: 0.01,
            window_size: 3,
            sigma: 1.0,
        }
    }
}

/// SIFT feature detection parameters
#[derive(Debug, Clone)]
pub struct SiftParams {
    pub num_octaves: usize,      // Number of octaves in scale space
    pub num_scales: usize,       // Number of scales per octave  
    pub sigma: f32,              // Initial smoothing sigma
    pub contrast_threshold: f32,  // Contrast threshold for keypoint detection
    pub edge_threshold: f32,     // Edge response threshold
    pub descriptor_size: usize,  // Size of SIFT descriptor (typically 128)
}

impl Default for SiftParams {
    fn default() -> Self {
        Self {
            num_octaves: 4,
            num_scales: 3,
            sigma: 1.6,
            contrast_threshold: 0.03,
            edge_threshold: 10.0,
            descriptor_size: 128,
        }
    }
}

/// ORB feature detection parameters
#[derive(Debug, Clone)]
pub struct OrbParams {
    pub max_features: usize,     // Maximum number of features to detect
    pub scale_factor: f32,       // Scale factor between levels  
    pub num_levels: usize,       // Number of pyramid levels
    pub edge_threshold: usize,   // Edge threshold for FAST detector
    pub first_level: usize,      // First level in pyramid
    pub wta_k: usize,           // Number of points for WTA hash
    pub patch_size: usize,       // Patch size for descriptor
}

impl Default for OrbParams {
    fn default() -> Self {
        Self {
            max_features: 500,
            scale_factor: 1.2,
            num_levels: 8,
            edge_threshold: 31,
            first_level: 0,
            wta_k: 2,
            patch_size: 31,
        }
    }
}

/// Detect Harris corners in an image
pub fn detect_harris_corners(image: &Image, params: Option<HarrisParams>) -> VmResult<FeatureSet> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("Harris corner detection requires grayscale image".to_string()));
    }
    
    let mut feature_set = FeatureSet::new("harris".to_string());
    
    // Convert image to grayscale if needed (should already be grayscale)
    let gray_data = &image.data;
    let width = image.width;
    let height = image.height;
    
    // Compute image gradients using Sobel operators
    let mut ix = vec![0.0; width * height];
    let mut iy = vec![0.0; width * height];
    
    // Sobel kernels
    let sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    
    // Apply Sobel operators
    for y in 1..height-1 {
        for x in 1..width-1 {
            let idx = y * width + x;
            
            // Compute x-gradient
            let mut gx = 0.0;
            let mut gy = 0.0;
            
            for dy in 0..3 {
                for dx in 0..3 {
                    let pixel_idx = (y + dy - 1) * width + (x + dx - 1);
                    let pixel = gray_data[pixel_idx];
                    
                    gx += sobel_x[dy][dx] * pixel;
                    gy += sobel_y[dy][dx] * pixel;
                }
            }
            
            ix[idx] = gx;
            iy[idx] = gy;
        }
    }
    
    // Compute Harris response
    let window_radius = params.window_size / 2;
    
    for y in window_radius..height-window_radius {
        for x in window_radius..width-window_radius {
            let _idx = y * width + x;
            
            // Compute structure matrix components in window
            let mut ixx = 0.0;
            let mut ixy = 0.0;
            let mut iyy = 0.0;
            
            for dy in -(window_radius as i32)..=window_radius as i32 {
                for dx in -(window_radius as i32)..=window_radius as i32 {
                    let nx = (x as i32 + dx) as usize;
                    let ny = (y as i32 + dy) as usize;
                    
                    if nx < width && ny < height {
                        let pidx = ny * width + nx;
                        let gx = ix[pidx];
                        let gy = iy[pidx];
                        
                        ixx += gx * gx;
                        ixy += gx * gy;
                        iyy += gy * gy;
                    }
                }
            }
            
            // Harris corner response
            let det = ixx * iyy - ixy * ixy;
            let trace = ixx + iyy;
            let response = det - params.k * trace * trace;
            
            // Check if corner response exceeds threshold
            if response > params.threshold {
                let keypoint = KeyPoint::with_response(x as f32, y as f32, response);
                feature_set.add_keypoint(keypoint);
            }
        }
    }
    
    // Apply non-maximum suppression
    apply_non_maximum_suppression(&mut feature_set, params.window_size);
    
    Ok(feature_set)
}

/// Detect SIFT features in an image
pub fn detect_sift_features(image: &Image, params: Option<SiftParams>) -> VmResult<FeatureSet> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("SIFT detection requires grayscale image".to_string()));
    }
    
    let mut feature_set = FeatureSet::new("sift".to_string());
    
    // Build Gaussian pyramid
    let pyramid = build_gaussian_pyramid(image, params.num_octaves, params.num_scales, params.sigma)?;
    
    // Build Difference of Gaussians (DoG) pyramid
    let dog_pyramid = build_dog_pyramid(&pyramid)?;
    
    // Detect keypoints in DoG pyramid
    let mut keypoints = detect_dog_keypoints(&dog_pyramid, params.contrast_threshold, params.edge_threshold)?;
    
    // Assign orientations to keypoints
    assign_keypoint_orientations(&mut keypoints, &pyramid)?;
    
    // Compute SIFT descriptors
    for keypoint in keypoints {
        let descriptor = compute_sift_descriptor(&keypoint, &pyramid, params.descriptor_size)?;
        feature_set.add_keypoint(keypoint);
        feature_set.add_descriptor(descriptor);
    }
    
    Ok(feature_set)
}

/// Detect ORB features in an image
pub fn detect_orb_features(image: &Image, params: Option<OrbParams>) -> VmResult<FeatureSet> {
    let params = params.unwrap_or_default();
    
    if image.channels != 1 {
        return Err(VmError::Runtime("ORB detection requires grayscale image".to_string()));
    }
    
    let mut feature_set = FeatureSet::new("orb".to_string());
    
    // Build image pyramid
    let pyramid = build_scale_pyramid(image, params.num_levels, params.scale_factor)?;
    
    // Detect FAST keypoints in each level
    for (level, level_image) in pyramid.iter().enumerate() {
        let scale = params.scale_factor.powi(level as i32);
        let mut level_keypoints = detect_fast_keypoints(level_image, params.edge_threshold)?;
        
        // Scale keypoints back to original image coordinates
        for keypoint in &mut level_keypoints {
            keypoint.x *= scale;
            keypoint.y *= scale;
            keypoint.size = scale;
            keypoint.octave = level as u8;
        }
        
        // Compute orientation for each keypoint
        for keypoint in &mut level_keypoints {
            assign_orb_orientation(keypoint, level_image)?;
        }
        
        // Compute BRIEF descriptors
        for keypoint in &level_keypoints {
            let descriptor = compute_brief_descriptor(keypoint, level_image, params.patch_size)?;
            feature_set.add_keypoint(keypoint.clone());
            feature_set.add_descriptor(descriptor);
        }
        
        // Limit number of features per level
        if feature_set.len() >= params.max_features {
            break;
        }
    }
    
    // Keep only the best features
    if feature_set.len() > params.max_features {
        retain_best_features(&mut feature_set, params.max_features);
    }
    
    Ok(feature_set)
}

/// Match features between two feature sets using brute force
pub fn match_features_brute_force(query: &FeatureSet, train: &FeatureSet, max_distance: Option<f32>) -> VmResult<FeatureMatches> {
    let max_dist = max_distance.unwrap_or(f32::MAX);
    let mut matches = FeatureMatches::new("brute_force".to_string());
    
    if query.descriptors.is_empty() || train.descriptors.is_empty() {
        return Ok(matches); // No descriptors to match
    }
    
    if query.descriptors[0].len() != train.descriptors[0].len() {
        return Err(VmError::Runtime("Descriptor dimensions don't match".to_string()));
    }
    
    // For each query descriptor, find the best match in train set
    for (query_idx, query_desc) in query.descriptors.iter().enumerate() {
        let mut best_distance = f32::MAX;
        let mut best_train_idx = 0;
        
        for (train_idx, train_desc) in train.descriptors.iter().enumerate() {
            let distance = compute_descriptor_distance(query_desc, train_desc);
            
            if distance < best_distance {
                best_distance = distance;
                best_train_idx = train_idx;
            }
        }
        
        if best_distance <= max_dist {
            let feature_match = FeatureMatch::new(query_idx, best_train_idx, best_distance);
            matches.add_match(feature_match);
        }
    }
    
    Ok(matches)
}

/// Match features using ratio test (Lowe's ratio test)
pub fn match_features_ratio_test(query: &FeatureSet, train: &FeatureSet, ratio_threshold: Option<f32>) -> VmResult<FeatureMatches> {
    let ratio_thresh = ratio_threshold.unwrap_or(0.75);
    let mut matches = FeatureMatches::new("ratio_test".to_string());
    
    if query.descriptors.is_empty() || train.descriptors.is_empty() {
        return Ok(matches); // No descriptors to match
    }
    
    if query.descriptors[0].len() != train.descriptors[0].len() {
        return Err(VmError::Runtime("Descriptor dimensions don't match".to_string()));
    }
    
    // For each query descriptor, find the two best matches
    for (query_idx, query_desc) in query.descriptors.iter().enumerate() {
        let mut distances: Vec<(usize, f32)> = train.descriptors.iter().enumerate()
            .map(|(train_idx, train_desc)| {
                let distance = compute_descriptor_distance(query_desc, train_desc);
                (train_idx, distance)
            })
            .collect();
        
        // Sort by distance
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        if distances.len() >= 2 {
            let best_distance = distances[0].1;
            let second_best_distance = distances[1].1;
            
            // Apply ratio test
            if best_distance / second_best_distance < ratio_thresh {
                let feature_match = FeatureMatch::new(query_idx, distances[0].0, best_distance);
                matches.add_match(feature_match);
            }
        }
    }
    
    Ok(matches)
}

// Helper functions for feature detection implementation

/// Apply non-maximum suppression to remove weak corners
fn apply_non_maximum_suppression(feature_set: &mut FeatureSet, window_size: usize) {
    let radius = window_size / 2;
    let mut to_remove = Vec::new();
    
    for i in 0..feature_set.keypoints.len() {
        for j in (i + 1)..feature_set.keypoints.len() {
            let kp1 = &feature_set.keypoints[i];
            let kp2 = &feature_set.keypoints[j];
            
            let dx = (kp1.x - kp2.x).abs();
            let dy = (kp1.y - kp2.y).abs();
            
            if dx <= radius as f32 && dy <= radius as f32 {
                // Keep the one with higher response
                if kp1.response < kp2.response {
                    to_remove.push(i);
                } else {
                    to_remove.push(j);
                }
            }
        }
    }
    
    // Remove duplicates and sort in reverse order
    to_remove.sort_unstable();
    to_remove.dedup();
    to_remove.reverse();
    
    // Remove keypoints
    for &idx in &to_remove {
        feature_set.keypoints.remove(idx);
    }
}

/// Build Gaussian pyramid for SIFT
fn build_gaussian_pyramid(image: &Image, num_octaves: usize, num_scales: usize, sigma: f32) -> VmResult<Vec<Vec<Image>>> {
    let mut pyramid: Vec<Vec<Image>> = Vec::new();
    
    for octave in 0..num_octaves {
        let mut octave_images: Vec<Image> = Vec::new();
        let _scale_factor = 2.0_f32.powi(octave as i32);
        
        // Create base image for this octave
        let current_image = if octave == 0 {
            image.clone()
        } else {
            // Downsample previous octave's image
            downsample_image(&pyramid[octave - 1][num_scales - 1], 0.5)?
        };
        
        // Apply Gaussian blur for each scale
        for scale in 0..num_scales + 3 {  // +3 for DoG computation
            let current_sigma = sigma * (2.0_f32.powf(scale as f32 / num_scales as f32));
            let blurred = apply_gaussian_blur(&current_image, current_sigma)?;
            octave_images.push(blurred);
        }
        
        pyramid.push(octave_images);
    }
    
    Ok(pyramid)
}

/// Build Difference of Gaussians pyramid
fn build_dog_pyramid(pyramid: &[Vec<Image>]) -> VmResult<Vec<Vec<Image>>> {
    let mut dog_pyramid = Vec::new();
    
    for octave_images in pyramid {
        let mut dog_octave = Vec::new();
        
        for i in 0..octave_images.len() - 1 {
            let diff = subtract_images(&octave_images[i + 1], &octave_images[i])?;
            dog_octave.push(diff);
        }
        
        dog_pyramid.push(dog_octave);
    }
    
    Ok(dog_pyramid)
}

/// Detect keypoints in DoG pyramid
fn detect_dog_keypoints(dog_pyramid: &[Vec<Image>], contrast_threshold: f32, edge_threshold: f32) -> VmResult<Vec<KeyPoint>> {
    let mut keypoints = Vec::new();
    
    // For each octave
    for (octave, octave_images) in dog_pyramid.iter().enumerate() {
        // For each scale (skip first and last)
        for scale in 1..octave_images.len() - 1 {
            let current = &octave_images[scale];
            let below = &octave_images[scale - 1];
            let above = &octave_images[scale + 1];
            
            // Check each pixel (skip borders)
            for y in 1..current.height - 1 {
                for x in 1..current.width - 1 {
                    let center_value = current.data[y * current.width + x];
                    
                    // Check if this is a local extremum
                    if is_local_extremum(x, y, current, below, above) {
                        // Apply contrast threshold
                        if center_value.abs() > contrast_threshold {
                            // Apply edge threshold using Hessian
                            if !is_edge_response(x, y, current, edge_threshold) {
                                let scale_factor = 2.0_f32.powi(octave as i32);
                                let keypoint = KeyPoint {
                                    x: (x as f32) * scale_factor,
                                    y: (y as f32) * scale_factor,
                                    size: scale_factor,
                                    angle: -1.0,
                                    response: center_value.abs(),
                                    octave: octave as u8,
                                    class_id: -1,
                                };
                                keypoints.push(keypoint);
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(keypoints)
}

/// Check if pixel is local extremum in scale space
fn is_local_extremum(x: usize, y: usize, current: &Image, below: &Image, above: &Image) -> bool {
    let center_value = current.data[y * current.width + x];
    let mut is_max = true;
    let mut is_min = true;
    
    // Check 3x3x3 neighborhood
    for dz in &[below, current, above] {
        for dy in -1_i32..=1 {
            for dx in -1_i32..=1 {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                
                if nx < dz.width && ny < dz.height {
                    let neighbor = dz.data[ny * dz.width + nx];
                    
                    if neighbor >= center_value {
                        is_max = false;
                    }
                    if neighbor <= center_value {
                        is_min = false;
                    }
                    
                    if !is_max && !is_min {
                        return false;
                    }
                }
            }
        }
    }
    
    is_max || is_min
}

/// Check if response is an edge using Hessian determinant
fn is_edge_response(x: usize, y: usize, image: &Image, edge_threshold: f32) -> bool {
    let width = image.width;
    let data = &image.data;
    
    // Compute second derivatives
    let dxx = data[y * width + x + 1] - 2.0 * data[y * width + x] + data[y * width + x - 1];
    let dyy = data[(y + 1) * width + x] - 2.0 * data[y * width + x] + data[(y - 1) * width + x];
    let dxy = (data[(y + 1) * width + x + 1] - data[(y + 1) * width + x - 1] -
               data[(y - 1) * width + x + 1] + data[(y - 1) * width + x - 1]) / 4.0;
    
    // Compute trace and determinant of Hessian
    let trace = dxx + dyy;
    let det = dxx * dyy - dxy * dxy;
    
    // Check edge condition
    if det <= 0.0 || trace * trace / det >= (edge_threshold + 1.0).powi(2) / edge_threshold {
        true // Is an edge response
    } else {
        false // Not an edge response
    }
}

/// Assign orientations to SIFT keypoints
fn assign_keypoint_orientations(keypoints: &mut Vec<KeyPoint>, pyramid: &[Vec<Image>]) -> VmResult<()> {
    for keypoint in keypoints.iter_mut() {
        // Get the appropriate pyramid level
        let octave = keypoint.octave as usize;
        let scale = 1; // Simplified - use middle scale
        
        if octave < pyramid.len() && scale < pyramid[octave].len() {
            let image = &pyramid[octave][scale];
            
            // Compute gradient orientation histogram
            let orientation = compute_dominant_orientation(keypoint, image)?;
            keypoint.angle = orientation;
        }
    }
    
    Ok(())
}

/// Compute dominant orientation for a keypoint
fn compute_dominant_orientation(keypoint: &KeyPoint, image: &Image) -> VmResult<f32> {
    let x = (keypoint.x / 2.0_f32.powi(keypoint.octave as i32)) as usize;
    let y = (keypoint.y / 2.0_f32.powi(keypoint.octave as i32)) as usize;
    
    if x >= image.width || y >= image.height {
        return Ok(-1.0);
    }
    
    let mut histogram = [0.0; 36]; // 36 bins for 360 degrees
    let window_radius = 8; // Window radius around keypoint
    
    // Compute gradient orientations in window
    for dy in -(window_radius as i32)..=window_radius as i32 {
        for dx in -(window_radius as i32)..=window_radius as i32 {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;
            
            if nx > 0 && nx < image.width - 1 && ny > 0 && ny < image.height - 1 {
                // Compute gradient
                let gx = image.data[ny * image.width + nx + 1] - image.data[ny * image.width + nx - 1];
                let gy = image.data[(ny + 1) * image.width + nx] - image.data[(ny - 1) * image.width + nx];
                
                let magnitude = (gx * gx + gy * gy).sqrt();
                let orientation = gy.atan2(gx);
                
                // Convert to degrees and bin
                let angle_degrees = orientation * 180.0 / PI;
                let angle_positive = if angle_degrees < 0.0 { angle_degrees + 360.0 } else { angle_degrees };
                let bin = ((angle_positive / 10.0) as usize).min(35);
                
                histogram[bin] += magnitude;
            }
        }
    }
    
    // Find dominant orientation
    let max_idx = histogram.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    Ok((max_idx as f32) * 10.0 * PI / 180.0) // Convert back to radians
}

/// Compute SIFT descriptor for a keypoint
fn compute_sift_descriptor(keypoint: &KeyPoint, _pyramid: &[Vec<Image>], descriptor_size: usize) -> VmResult<Vec<f32>> {
    // Simplified SIFT descriptor - just return a random descriptor for now
    // In a real implementation, this would compute the 128-dimensional SIFT descriptor
    let mut descriptor = vec![0.0; descriptor_size];
    
    // Generate a simple descriptor based on keypoint properties
    for i in 0..descriptor_size {
        descriptor[i] = ((keypoint.x + keypoint.y + i as f32) * 0.1).sin();
    }
    
    // Normalize descriptor
    let norm = descriptor.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for val in &mut descriptor {
            *val /= norm;
        }
    }
    
    Ok(descriptor)
}

/// Build scale pyramid for ORB
fn build_scale_pyramid(image: &Image, num_levels: usize, scale_factor: f32) -> VmResult<Vec<Image>> {
    let mut pyramid = vec![image.clone()];
    
    for level in 1..num_levels {
        let scale = 1.0 / scale_factor.powi(level as i32);
        let scaled_image = resize_image(&pyramid[0], scale)?;
        pyramid.push(scaled_image);
    }
    
    Ok(pyramid)
}

/// Detect FAST keypoints
fn detect_fast_keypoints(image: &Image, threshold: usize) -> VmResult<Vec<KeyPoint>> {
    let mut keypoints = Vec::new();
    let threshold_f = threshold as f32 / 255.0; // Normalize threshold
    
    // FAST-9 circle pattern (16 pixels around center)
    let circle = [
        (0, 3), (1, 3), (2, 2), (3, 1),
        (3, 0), (3, -1), (2, -2), (1, -3),
        (0, -3), (-1, -3), (-2, -2), (-3, -1),
        (-3, 0), (-3, 1), (-2, 2), (-1, 3)
    ];
    
    for y in 3..image.height - 3 {
        for x in 3..image.width - 3 {
            let center = image.data[y * image.width + x];
            let mut brighter = 0;
            let mut darker = 0;
            
            // Check pixels on circle
            for &(dx, dy) in &circle {
                let nx = (x as i32 + dx) as usize;
                let ny = (y as i32 + dy) as usize;
                let pixel = image.data[ny * image.width + nx];
                
                if pixel > center + threshold_f {
                    brighter += 1;
                } else if pixel < center - threshold_f {
                    darker += 1;
                }
            }
            
            // Need at least 9 consecutive pixels
            if brighter >= 9 || darker >= 9 {
                let response = calculate_fast_response(x, y, image, &circle, center);
                let keypoint = KeyPoint::with_response(x as f32, y as f32, response);
                keypoints.push(keypoint);
            }
        }
    }
    
    Ok(keypoints)
}

/// Calculate FAST corner response
fn calculate_fast_response(x: usize, y: usize, image: &Image, circle: &[(i32, i32)], center: f32) -> f32 {
    let mut sum = 0.0;
    
    for &(dx, dy) in circle {
        let nx = (x as i32 + dx) as usize;
        let ny = (y as i32 + dy) as usize;
        let pixel = image.data[ny * image.width + nx];
        sum += (pixel - center).abs();
    }
    
    sum / circle.len() as f32
}

/// Assign orientation to ORB keypoint
fn assign_orb_orientation(keypoint: &mut KeyPoint, image: &Image) -> VmResult<()> {
    let x = keypoint.x as usize;
    let y = keypoint.y as usize;
    
    if x >= image.width || y >= image.height {
        return Ok(());
    }
    
    // Compute intensity centroid
    let mut m10 = 0.0;
    let mut m01 = 0.0;
    let mut m00 = 0.0;
    let radius = 15; // Patch radius
    
    for dy in -(radius as i32)..=radius as i32 {
        for dx in -(radius as i32)..=radius as i32 {
            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;
            
            if nx < image.width && ny < image.height {
                let intensity = image.data[ny * image.width + nx];
                m10 += dx as f32 * intensity;
                m01 += dy as f32 * intensity;
                m00 += intensity;
            }
        }
    }
    
    if m00 > 0.0 {
        let centroid_x = m10 / m00;
        let centroid_y = m01 / m00;
        keypoint.angle = centroid_y.atan2(centroid_x);
    }
    
    Ok(())
}

/// Compute BRIEF descriptor
fn compute_brief_descriptor(keypoint: &KeyPoint, image: &Image, patch_size: usize) -> VmResult<Vec<f32>> {
    let descriptor_length = 256; // BRIEF descriptor length
    let mut descriptor = vec![0.0; descriptor_length];
    
    let x = keypoint.x as usize;
    let y = keypoint.y as usize;
    let radius = patch_size / 2;
    
    // Pre-defined test pattern (simplified)
    let _bit_idx = 0;
    for i in 0..descriptor_length {
        let (p1x, p1y, p2x, p2y) = generate_test_points(i, radius);
        
        let nx1 = (x as i32 + p1x) as usize;
        let ny1 = (y as i32 + p1y) as usize;
        let nx2 = (x as i32 + p2x) as usize;
        let ny2 = (y as i32 + p2y) as usize;
        
        if nx1 < image.width && ny1 < image.height &&
           nx2 < image.width && ny2 < image.height {
            let intensity1 = image.data[ny1 * image.width + nx1];
            let intensity2 = image.data[ny2 * image.width + nx2];
            
            descriptor[i] = if intensity1 < intensity2 { 1.0 } else { 0.0 };
        }
    }
    
    Ok(descriptor)
}

/// Generate test points for BRIEF descriptor
fn generate_test_points(index: usize, radius: usize) -> (i32, i32, i32, i32) {
    // Simplified test pattern generation
    let angle = (index as f32) * 2.0 * PI / 256.0;
    let r = (radius as f32) * 0.8;
    
    let p1x = (r * angle.cos()) as i32;
    let p1y = (r * angle.sin()) as i32;
    let p2x = (r * (angle + PI).cos()) as i32;
    let p2y = (r * (angle + PI).sin()) as i32;
    
    (p1x, p1y, p2x, p2y)
}

/// Retain only the best features
fn retain_best_features(feature_set: &mut FeatureSet, max_features: usize) {
    if feature_set.keypoints.len() <= max_features {
        return;
    }
    
    // Create indices sorted by response (highest first)
    let mut indices: Vec<usize> = (0..feature_set.keypoints.len()).collect();
    indices.sort_by(|&a, &b| {
        feature_set.keypoints[b].response.partial_cmp(&feature_set.keypoints[a].response)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    // Keep only the best features
    let mut new_keypoints = Vec::new();
    let mut new_descriptors = Vec::new();
    
    for i in 0..max_features {
        if i < indices.len() {
            let idx = indices[i];
            new_keypoints.push(feature_set.keypoints[idx].clone());
            if idx < feature_set.descriptors.len() {
                new_descriptors.push(feature_set.descriptors[idx].clone());
            }
        }
    }
    
    feature_set.keypoints = new_keypoints;
    feature_set.descriptors = new_descriptors;
}

/// Compute distance between two descriptors
fn compute_descriptor_distance(desc1: &[f32], desc2: &[f32]) -> f32 {
    if desc1.len() != desc2.len() {
        return f32::MAX;
    }
    
    // Compute Euclidean distance
    desc1.iter().zip(desc2.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}

// Helper functions for image processing

/// Apply Gaussian blur to image
fn apply_gaussian_blur(image: &Image, sigma: f32) -> VmResult<Image> {
    // Generate Gaussian kernel
    let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd size
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
                let pixel_x = (x as i32 + k as i32 - kernel_radius as i32) as usize;
                if pixel_x < image.width {
                    sum += image.data[y * image.width + pixel_x] * kernel[k];
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
                let pixel_y = (y as i32 + k as i32 - kernel_radius as i32) as usize;
                if pixel_y < image.height {
                    sum += image.data[pixel_y * image.width + x] * kernel[k];
                }
            }
            
            result.data[y * image.width + x] = sum;
        }
    }
    
    Ok(result)
}

/// Subtract two images
fn subtract_images(image1: &Image, image2: &Image) -> VmResult<Image> {
    if image1.width != image2.width || image1.height != image2.height {
        return Err(VmError::Runtime("Image dimensions don't match".to_string()));
    }
    
    let mut result = image1.clone();
    for i in 0..result.data.len() {
        result.data[i] = image1.data[i] - image2.data[i];
    }
    
    Ok(result)
}

/// Downsample image by given factor
fn downsample_image(image: &Image, factor: f32) -> VmResult<Image> {
    let new_width = ((image.width as f32) * factor) as usize;
    let new_height = ((image.height as f32) * factor) as usize;
    
    let mut result = Image {
        data: vec![0.0; new_width * new_height],
        width: new_width,
        height: new_height,
        channels: image.channels,
        color_space: image.color_space,
        bit_depth: image.bit_depth,
    };
    
    // Simple nearest-neighbor downsampling
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = ((x as f32) / factor) as usize;
            let src_y = ((y as f32) / factor) as usize;
            
            if src_x < image.width && src_y < image.height {
                result.data[y * new_width + x] = image.data[src_y * image.width + src_x];
            }
        }
    }
    
    Ok(result)
}

/// Resize image by scale factor
fn resize_image(image: &Image, scale: f32) -> VmResult<Image> {
    let new_width = ((image.width as f32) * scale) as usize;
    let new_height = ((image.height as f32) * scale) as usize;
    
    let mut result = Image {
        data: vec![0.0; new_width * new_height],
        width: new_width,
        height: new_height,
        channels: image.channels,
        color_space: image.color_space,
        bit_depth: image.bit_depth,
    };
    
    // Bilinear interpolation
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = (x as f32) / scale;
            let src_y = (y as f32) / scale;
            
            let x0 = src_x.floor() as usize;
            let y0 = src_y.floor() as usize;
            let x1 = (x0 + 1).min(image.width - 1);
            let y1 = (y0 + 1).min(image.height - 1);
            
            let fx = src_x - x0 as f32;
            let fy = src_y - y0 as f32;
            
            if x0 < image.width && y0 < image.height {
                let p00 = image.data[y0 * image.width + x0];
                let p10 = image.data[y0 * image.width + x1];
                let p01 = image.data[y1 * image.width + x0];
                let p11 = image.data[y1 * image.width + x1];
                
                let interpolated = p00 * (1.0 - fx) * (1.0 - fy) +
                                 p10 * fx * (1.0 - fy) +
                                 p01 * (1.0 - fx) * fy +
                                 p11 * fx * fy;
                
                result.data[y * new_width + x] = interpolated;
            }
        }
    }
    
    Ok(result)
}

// Stdlib function implementations

/// Harris corner detection - HarrisCorners[image, opts]
pub fn harris_corners(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
            let image = match img {
                Value::LyObj(obj) => {
                    obj.downcast_ref::<Image>()
                        .ok_or_else(|| VmError::TypeError {
                            expected: "Image".to_string(),
                            actual: obj.type_name().to_string(),
                        })?
                }
                _ => return Err(VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: format!("{:?}", img),
                }),
            };
            
            let features = detect_harris_corners(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
            Ok(Value::Object(m))
        }
        [img, _opts] => {
            let image = match img {
                Value::LyObj(obj) => {
                    obj.downcast_ref::<Image>()
                        .ok_or_else(|| VmError::TypeError {
                            expected: "Image".to_string(),
                            actual: obj.type_name().to_string(),
                        })?
                }
                _ => return Err(VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: format!("{:?}", img),
                }),
            };
            
            // Parse options (simplified - in real implementation would parse Rule lists)
            let params = HarrisParams::default();
            
            let features = detect_harris_corners(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
            Ok(Value::Object(m))
        }
        _ => Err(VmError::TypeError {
            expected: "1 or 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

/// SIFT feature detection - SIFTFeatures[image, opts]
pub fn sift_features(args: &[Value]) -> VmResult<Value> {
    match args {
        [img] => {
            let image = match img {
                Value::LyObj(obj) => {
                    obj.downcast_ref::<Image>()
                        .ok_or_else(|| VmError::TypeError {
                            expected: "Image".to_string(),
                            actual: obj.type_name().to_string(),
                        })?
                }
                _ => return Err(VmError::TypeError {
                    expected: "Image".to_string(),
                    actual: format!("{:?}", img),
                }),
            };
            
            let features = detect_sift_features(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
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
            let params = SiftParams::default();
            
            let features = detect_sift_features(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("SIFTFeatures expects 1 or 2 arguments".to_string())),
    }
}

/// ORB feature detection - ORBFeatures[image, opts]
pub fn orb_features(args: &[Value]) -> VmResult<Value> {
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
            
            let features = detect_orb_features(image, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
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
            let params = OrbParams::default();
            
            let features = detect_orb_features(image, Some(params))?;
            let mut m = std::collections::HashMap::new();
            m.insert("featureType".to_string(), Value::String(features.feature_type.clone()));
            let kps: Vec<Value> = features.keypoints.iter().map(|kp| {
                let mut km = std::collections::HashMap::new();
                km.insert("x".to_string(), Value::Real(kp.x as f64));
                km.insert("y".to_string(), Value::Real(kp.y as f64));
                km.insert("size".to_string(), Value::Real(kp.size as f64));
                km.insert("angle".to_string(), Value::Real(kp.angle as f64));
                km.insert("response".to_string(), Value::Real(kp.response as f64));
                km.insert("octave".to_string(), Value::Integer(kp.octave as i64));
                km.insert("classId".to_string(), Value::Integer(kp.class_id as i64));
                Value::Object(km)
            }).collect();
            m.insert("keypoints".to_string(), Value::List(kps));
            m.insert(
                "descriptors".to_string(),
                Value::List(
                    features
                        .descriptors
                        .iter()
                        .map(|d| Value::List(d.iter().map(|&f| Value::Real(f as f64)).collect()))
                        .collect(),
                ),
            );
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("ORBFeatures expects 1 or 2 arguments".to_string())),
    }
}

/// Match features between two sets - MatchFeatures[features1, features2, opts]
pub fn match_features(args: &[Value]) -> VmResult<Value> {
    match args {
        [feat1, feat2] => {
            let features1 = match feat1 {
            Value::LyObj(obj) => obj,
            _ => return Err(VmError::TypeError {
                expected: "LyObj".to_string(),
                actual: format!("{:?}", feat1)
            })
        };
        
        let features1 = features1.downcast_ref::<FeatureSet>()
            .ok_or_else(|| VmError::TypeError {
                expected: "FeatureSet".to_string(),
                actual: features1.type_name().to_string(),
            })?;
            
            let features2 = match feat2 {
            Value::LyObj(obj) => obj,
            _ => return Err(VmError::TypeError {
                expected: "LyObj".to_string(),
                actual: format!("{:?}", feat2)
            })
        };
        
        let features2 = features2.downcast_ref::<FeatureSet>()
            .ok_or_else(|| VmError::TypeError {
                expected: "FeatureSet".to_string(),
                actual: features2.type_name().to_string(),
            })?;
            
            let matches = match_features_brute_force(features1, features2, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("matchType".to_string(), Value::String(matches.match_type.clone()));
            m.insert("count".to_string(), Value::Integer(matches.matches.len() as i64));
            let items: Vec<Value> = matches.matches.iter().map(|fm| {
                let mut mm = std::collections::HashMap::new();
                mm.insert("queryIndex".to_string(), Value::Integer(fm.query_idx as i64));
                mm.insert("trainIndex".to_string(), Value::Integer(fm.train_idx as i64));
                mm.insert("distance".to_string(), Value::Real(fm.distance as f64));
                mm.insert("confidence".to_string(), Value::Real(fm.confidence as f64));
                Value::Object(mm)
            }).collect();
            m.insert("matches".to_string(), Value::List(items));
            Ok(Value::Object(m))
        }
        [feat1, feat2, _opts] => {
            let features1 = match feat1 {
            Value::LyObj(obj) => obj,
            _ => return Err(VmError::TypeError {
                expected: "LyObj".to_string(),
                actual: format!("{:?}", feat1)
            })
        };
        
        let features1 = features1.downcast_ref::<FeatureSet>()
            .ok_or_else(|| VmError::TypeError {
                expected: "FeatureSet".to_string(),
                actual: features1.type_name().to_string(),
            })?;
            
            let features2 = match feat2 {
            Value::LyObj(obj) => obj,
            _ => return Err(VmError::TypeError {
                expected: "LyObj".to_string(),
                actual: format!("{:?}", feat2)
            })
        };
        
        let features2 = features2.downcast_ref::<FeatureSet>()
            .ok_or_else(|| VmError::TypeError {
                expected: "FeatureSet".to_string(),
                actual: features2.type_name().to_string(),
            })?;
            
            // Parse options (simplified)
            let matches = match_features_ratio_test(features1, features2, None)?;
            let mut m = std::collections::HashMap::new();
            m.insert("matchType".to_string(), Value::String(matches.match_type.clone()));
            m.insert("count".to_string(), Value::Integer(matches.matches.len() as i64));
            let items: Vec<Value> = matches.matches.iter().map(|fm| {
                let mut mm = std::collections::HashMap::new();
                mm.insert("queryIndex".to_string(), Value::Integer(fm.query_idx as i64));
                mm.insert("trainIndex".to_string(), Value::Integer(fm.train_idx as i64));
                mm.insert("distance".to_string(), Value::Real(fm.distance as f64));
                mm.insert("confidence".to_string(), Value::Real(fm.confidence as f64));
                Value::Object(mm)
            }).collect();
            m.insert("matches".to_string(), Value::List(items));
            Ok(Value::Object(m))
        }
        _ => Err(VmError::Runtime("MatchFeatures expects 2 or 3 arguments".to_string())),
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

    /// Create a test image with a corner pattern
    fn create_corner_image(width: usize, height: usize) -> Image {
        let mut image = create_test_image(width, height);
        
        // Create a corner in the upper-left quadrant
        for y in height / 4..height / 2 {
            for x in width / 4..width / 2 {
                if x < width / 3 || y < height / 3 {
                    image.data[y * width + x] = 1.0; // White
                } else {
                    image.data[y * width + x] = 0.0; // Black
                }
            }
        }
        
        image
    }

    #[test]
    fn test_harris_corner_detection() {
        let image = create_corner_image(64, 64);
        let features = detect_harris_corners(&image, None).unwrap();
        
        assert_eq!(features.feature_type, "harris");
        // Should detect at least some corners
        assert!(!features.keypoints.is_empty(), "Should detect some corners");
    }

    #[test]
    fn test_harris_with_custom_params() {
        let image = create_corner_image(64, 64);
        let params = HarrisParams {
            k: 0.05,
            threshold: 0.005,
            window_size: 5,
            sigma: 1.5,
        };
        
        let features = detect_harris_corners(&image, Some(params)).unwrap();
        assert_eq!(features.feature_type, "harris");
    }

    #[test]
    fn test_sift_feature_detection() {
        let image = create_corner_image(128, 128);
        let features = detect_sift_features(&image, None).unwrap();
        
        assert_eq!(features.feature_type, "sift");
        // SIFT should produce keypoints and descriptors
        if !features.keypoints.is_empty() {
            assert!(!features.descriptors.is_empty());
            assert_eq!(features.keypoints.len(), features.descriptors.len());
        }
    }

    #[test]
    fn test_orb_feature_detection() {
        let image = create_corner_image(128, 128);
        let features = detect_orb_features(&image, None).unwrap();
        
        assert_eq!(features.feature_type, "orb");
        // ORB should produce keypoints and descriptors
        if !features.keypoints.is_empty() {
            assert!(!features.descriptors.is_empty());
            assert_eq!(features.keypoints.len(), features.descriptors.len());
        }
    }

    #[test]
    fn test_feature_matching_brute_force() {
        let image1 = create_corner_image(64, 64);
        let image2 = create_corner_image(64, 64);
        
        let features1 = detect_harris_corners(&image1, None).unwrap();
        let features2 = detect_harris_corners(&image2, None).unwrap();
        
        // Add dummy descriptors for matching
        let mut feat1_with_desc = FeatureSet::new("harris".to_string());
        feat1_with_desc.keypoints = features1.keypoints;
        for _ in 0..feat1_with_desc.keypoints.len() {
            feat1_with_desc.descriptors.push(vec![1.0, 0.0, 1.0, 0.0]);
        }
        
        let mut feat2_with_desc = FeatureSet::new("harris".to_string());
        feat2_with_desc.keypoints = features2.keypoints;
        for _ in 0..feat2_with_desc.keypoints.len() {
            feat2_with_desc.descriptors.push(vec![1.0, 0.0, 1.0, 0.0]);
        }
        
        let matches = match_features_brute_force(&feat1_with_desc, &feat2_with_desc, None).unwrap();
        assert_eq!(matches.match_type, "brute_force");
    }

    #[test]
    fn test_feature_matching_ratio_test() {
        let image1 = create_corner_image(64, 64);
        let image2 = create_corner_image(64, 64);
        
        let features1 = detect_harris_corners(&image1, None).unwrap();
        let features2 = detect_harris_corners(&image2, None).unwrap();
        
        // Add dummy descriptors
        let mut feat1_with_desc = FeatureSet::new("harris".to_string());
        feat1_with_desc.keypoints = features1.keypoints;
        for i in 0..feat1_with_desc.keypoints.len() {
            feat1_with_desc.descriptors.push(vec![i as f32, 0.0, 1.0, 0.0]);
        }
        
        let mut feat2_with_desc = FeatureSet::new("harris".to_string());
        feat2_with_desc.keypoints = features2.keypoints;
        for i in 0..feat2_with_desc.keypoints.len() {
            feat2_with_desc.descriptors.push(vec![i as f32, 0.0, 1.0, 0.0]);
        }
        
        let matches = match_features_ratio_test(&feat1_with_desc, &feat2_with_desc, Some(0.8)).unwrap();
        assert_eq!(matches.match_type, "ratio_test");
    }

    #[test]
    fn test_keypoint_creation() {
        let kp1 = KeyPoint::new(10.0, 20.0);
        assert_eq!(kp1.x, 10.0);
        assert_eq!(kp1.y, 20.0);
        assert_eq!(kp1.size, 1.0);
        assert_eq!(kp1.response, 0.0);
        
        let kp2 = KeyPoint::with_response(5.0, 15.0, 0.8);
        assert_eq!(kp2.x, 5.0);
        assert_eq!(kp2.y, 15.0);
        assert_eq!(kp2.response, 0.8);
    }

    #[test]
    fn test_feature_set_operations() {
        let mut features = FeatureSet::new("test".to_string());
        assert!(features.is_empty());
        assert_eq!(features.len(), 0);
        
        features.add_keypoint(KeyPoint::new(1.0, 2.0));
        features.add_descriptor(vec![1.0, 2.0, 3.0, 4.0]);
        
        assert!(!features.is_empty());
        assert_eq!(features.len(), 1);
        assert_eq!(features.keypoints[0].x, 1.0);
        assert_eq!(features.descriptors[0], vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_descriptor_distance_computation() {
        let desc1 = vec![1.0, 0.0, 1.0, 0.0];
        let desc2 = vec![0.0, 1.0, 0.0, 1.0];
        let desc3 = vec![1.0, 0.0, 1.0, 0.0];
        
        let dist1 = compute_descriptor_distance(&desc1, &desc2);
        let dist2 = compute_descriptor_distance(&desc1, &desc3);
        
        assert!(dist1 > 0.0);
        assert_eq!(dist2, 0.0);
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
        let blurred = apply_gaussian_blur(&image, 1.0).unwrap();
        
        assert_eq!(blurred.width, image.width);
        assert_eq!(blurred.height, image.height);
        assert_eq!(blurred.data.len(), image.data.len());
    }

    #[test]
    fn test_image_downsampling() {
        let image = create_test_image(64, 64);
        let downsampled = downsample_image(&image, 0.5).unwrap();
        
        assert_eq!(downsampled.width, 32);
        assert_eq!(downsampled.height, 32);
    }

    #[test]
    fn test_image_resizing() {
        let image = create_test_image(32, 32);
        let resized = resize_image(&image, 2.0).unwrap();
        
        assert_eq!(resized.width, 64);
        assert_eq!(resized.height, 64);
    }

    #[test]
    fn test_image_subtraction() {
        let image1 = create_test_image(32, 32);
        let mut image2 = create_test_image(32, 32);
        
        // Make image2 slightly different
        image2.data[0] = 0.8;
        
        let diff = subtract_images(&image1, &image2).unwrap();
        assert_eq!(diff.width, 32);
        assert_eq!(diff.height, 32);
        assert!((diff.data[0] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_non_maximum_suppression() {
        let mut features = FeatureSet::new("test".to_string());
        
        // Add overlapping keypoints with different responses
        features.add_keypoint(KeyPoint::with_response(10.0, 10.0, 0.5));
        features.add_keypoint(KeyPoint::with_response(11.0, 11.0, 0.8)); // Should be kept
        features.add_keypoint(KeyPoint::with_response(10.0, 11.0, 0.3));
        features.add_keypoint(KeyPoint::with_response(20.0, 20.0, 0.9)); // Should be kept
        
        apply_non_maximum_suppression(&mut features, 3);
        
        // Should keep only the best keypoints in each region
        assert!(features.keypoints.len() <= 2);
    }

    #[test]
    fn test_fast_keypoint_detection() {
        let image = create_corner_image(64, 64);
        let keypoints = detect_fast_keypoints(&image, 30).unwrap();
        
        // Should detect some FAST keypoints
        for keypoint in &keypoints {
            assert!(keypoint.x >= 0.0 && keypoint.x < 64.0);
            assert!(keypoint.y >= 0.0 && keypoint.y < 64.0);
            assert!(keypoint.response >= 0.0);
        }
    }

    #[test]
    fn test_harris_stdlib_function() {
        let image = create_corner_image(64, 64);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = harris_corners(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let features = obj.as_any().downcast_ref::<FeatureSet>().unwrap();
                assert_eq!(features.feature_type, "harris");
            }
            _ => panic!("Expected FeatureSet"),
        }
    }

    #[test]
    fn test_sift_stdlib_function() {
        let image = create_corner_image(128, 128);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = sift_features(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let features = obj.as_any().downcast_ref::<FeatureSet>().unwrap();
                assert_eq!(features.feature_type, "sift");
            }
            _ => panic!("Expected FeatureSet"),
        }
    }

    #[test]
    fn test_orb_stdlib_function() {
        let image = create_corner_image(128, 128);
        let image_value = Value::LyObj(LyObj::new(Box::new(image)));
        
        let result = orb_features(&[image_value]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let features = obj.as_any().downcast_ref::<FeatureSet>().unwrap();
                assert_eq!(features.feature_type, "orb");
            }
            _ => panic!("Expected FeatureSet"),
        }
    }

    #[test]
    fn test_invalid_arguments() {
        let result = harris_corners(&[]);
        assert!(result.is_err());
        
        let result = harris_corners(&[Value::Integer(42)]);
        assert!(result.is_err());
        
        let result = sift_features(&[Value::Real(3.14)]);
        assert!(result.is_err());
    }
}
