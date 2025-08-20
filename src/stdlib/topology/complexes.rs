//! Simplicial Complex Construction
//!
//! Implementation of algorithms for building simplicial complexes:
//! - Vietoris-Rips complex
//! - Čech complex  
//! - Custom simplicial complex construction

use super::{Simplex, SimplicialComplex, value_to_points};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;
use crate::stdlib::geometry::Point2D;
use std::collections::{HashMap, HashSet};

/// Build Vietoris-Rips complex from point cloud
/// All simplices whose vertices are pairwise within distance radius
pub fn vietoris_rips_complex(points: &[Point2D], radius: f64, max_dimension: usize) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    let n = points.len();
    
    // Add all vertices
    for i in 0..n {
        complex.add_simplex(Simplex::new(vec![i]));
    }
    
    // Build adjacency matrix
    let mut adjacent = vec![vec![false; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if points[i].distance_to(&points[j]) <= radius {
                adjacent[i][j] = true;
                adjacent[j][i] = true;
            }
        }
    }
    
    // Build higher-dimensional simplices using clique enumeration
    for dim in 1..=max_dimension {
        let simplices_at_prev_dim = complex.simplices_at_dimension(dim - 1).cloned().collect::<Vec<_>>();
        
        for simplex in simplices_at_prev_dim {
            // Try to extend this simplex by one vertex
            for k in 0..n {
                if !simplex.vertices.contains(&k) {
                    // Check if k is adjacent to all vertices in the simplex
                    let can_extend = simplex.vertices.iter()
                        .all(|&v| adjacent[v][k]);
                    
                    if can_extend {
                        let mut new_vertices = simplex.vertices.clone();
                        new_vertices.push(k);
                        let new_simplex = Simplex::new(new_vertices);
                        
                        if !complex.contains_simplex(&new_simplex) {
                            complex.add_simplex(new_simplex);
                        }
                    }
                }
            }
        }
    }
    
    complex
}

/// Build Čech complex from point cloud
/// All simplices whose circumsphere has radius ≤ radius
pub fn cech_complex(points: &[Point2D], radius: f64, max_dimension: usize) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    let n = points.len();
    
    // Add all vertices
    for i in 0..n {
        complex.add_simplex(Simplex::new(vec![i]));
    }
    
    // Generate all possible simplices and check Čech condition
    for dim in 1..=max_dimension {
        let combinations = generate_combinations(n, dim + 1);
        
        for combination in combinations {
            if is_cech_simplex(&combination, points, radius) {
                let simplex = Simplex::new(combination);
                complex.add_simplex(simplex);
            }
        }
    }
    
    complex
}

/// Check if a set of points forms a valid Čech simplex
fn is_cech_simplex(vertices: &[usize], points: &[Point2D], radius: f64) -> bool {
    if vertices.len() <= 1 {
        return true;
    }
    
    if vertices.len() == 2 {
        // For edges, just check distance
        return points[vertices[0]].distance_to(&points[vertices[1]]) <= 2.0 * radius;
    }
    
    // For higher dimensions, find circumcenter and check circumradius
    let circumcenter_radius = compute_circumsphere(vertices, points);
    match circumcenter_radius {
        Some((_, r)) => r <= radius,
        None => false, // Degenerate case
    }
}

/// Compute circumsphere of a set of points
fn compute_circumsphere(vertices: &[usize], points: &[Point2D]) -> Option<(Point2D, f64)> {
    match vertices.len() {
        2 => {
            // Circumcenter of two points is their midpoint
            let p1 = &points[vertices[0]];
            let p2 = &points[vertices[1]];
            let center = Point2D::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
            let radius = p1.distance_to(p2) / 2.0;
            Some((center, radius))
        }
        3 => {
            // Circumcenter of triangle
            let p1 = &points[vertices[0]];
            let p2 = &points[vertices[1]];
            let p3 = &points[vertices[2]];
            
            circumcenter_triangle(p1, p2, p3)
        }
        _ => {
            // For higher dimensions, use approximate method
            // Find center that minimizes maximum distance to all points
            let centroid = compute_centroid(vertices, points);
            let max_distance = vertices.iter()
                .map(|&i| centroid.distance_to(&points[i]))
                .fold(0.0, f64::max);
            Some((centroid, max_distance))
        }
    }
}

/// Compute circumcenter of a triangle
fn circumcenter_triangle(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> Option<(Point2D, f64)> {
    let ax = p1.x;
    let ay = p1.y;
    let bx = p2.x;
    let by = p2.y;
    let cx = p3.x;
    let cy = p3.y;

    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    
    if d.abs() < 1e-10 {
        return None; // Points are collinear
    }

    let ux = (ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by);
    let uy = (ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax);

    let center_x = ux / d;
    let center_y = uy / d;
    let center = Point2D::new(center_x, center_y);
    
    let radius = center.distance_to(p1);
    
    Some((center, radius))
}

/// Compute centroid of a set of points
fn compute_centroid(vertices: &[usize], points: &[Point2D]) -> Point2D {
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    
    for &i in vertices {
        sum_x += points[i].x;
        sum_y += points[i].y;
    }
    
    let n = vertices.len() as f64;
    Point2D::new(sum_x / n, sum_y / n)
}

/// Generate all combinations of k elements from n
fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }
    
    let mut result = Vec::new();
    generate_combinations_recursive(0, n, k, &mut vec![], &mut result);
    result
}

fn generate_combinations_recursive(
    start: usize,
    n: usize,
    k: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }
    
    for i in start..n {
        current.push(i);
        generate_combinations_recursive(i + 1, n, k, current, result);
        current.pop();
    }
}

/// Build custom simplicial complex from explicit simplex list
pub fn build_simplicial_complex(simplices: Vec<Vec<usize>>) -> SimplicialComplex {
    let mut complex = SimplicialComplex::new();
    
    for simplex_vertices in simplices {
        if !simplex_vertices.is_empty() {
            complex.add_simplex(Simplex::new(simplex_vertices));
        }
    }
    
    complex
}

/// SimplicialComplex function for Lyra
/// Usage: SimplicialComplex[vertices, simplices]
pub fn simplicial_complex_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (vertices, simplices)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Parse vertices
    let vertices = match &args[0] {
        Value::List(vertex_list) => {
            let mut vertices = Vec::new();
            for v in vertex_list {
                match v {
                    Value::Integer(i) => vertices.push(*i as usize),
                    _ => return Err(VmError::TypeError {
                        expected: "integer vertex index".to_string(),
                        actual: format!("{:?}", v),
                    }),
                }
            }
            vertices
        }
        _ => return Err(VmError::TypeError {
            expected: "list of vertex indices".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    // Parse simplices
    let simplices = match &args[1] {
        Value::List(simplex_list) => {
            let mut simplices = Vec::new();
            for s in simplex_list {
                match s {
                    Value::List(vertex_indices) => {
                        let mut simplex_vertices = Vec::new();
                        for v in vertex_indices {
                            match v {
                                Value::Integer(i) => simplex_vertices.push(*i as usize),
                                _ => return Err(VmError::TypeError {
                                    expected: "integer vertex index".to_string(),
                                    actual: format!("{:?}", v),
                                }),
                            }
                        }
                        simplices.push(simplex_vertices);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of vertex indices for simplex".to_string(),
                        actual: format!("{:?}", s),
                    }),
                }
            }
            simplices
        }
        _ => return Err(VmError::TypeError {
            expected: "list of simplices".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let complex = build_simplicial_complex(simplices);
    Ok(Value::LyObj(LyObj::new(Box::new(complex))))
}

/// VietorisRips function for Lyra
/// Usage: VietorisRips[points, radius, maxDimension]
pub fn vietoris_rips_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (points, radius, maxDimension)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    let radius = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(r) => *r as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric radius".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let max_dimension = match &args[2] {
        Value::Integer(d) => *d as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer max dimension".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    if points.len() < 2 {
        return Err(VmError::Runtime(
            "Vietoris-Rips complex requires at least 2 points".to_string()
        ));
    }

    let complex = vietoris_rips_complex(&points, radius, max_dimension);
    Ok(Value::LyObj(LyObj::new(Box::new(complex))))
}

/// CechComplex function for Lyra  
/// Usage: CechComplex[points, radius, maxDimension]
pub fn cech_complex_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (points, radius, maxDimension)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    let radius = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(r) => *r as f64,
        _ => return Err(VmError::TypeError {
            expected: "numeric radius".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let max_dimension = match &args[2] {
        Value::Integer(d) => *d as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer max dimension".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    if points.len() < 2 {
        return Err(VmError::Runtime(
            "Čech complex requires at least 2 points".to_string()
        ));
    }

    let complex = cech_complex(&points, radius, max_dimension);
    Ok(Value::LyObj(LyObj::new(Box::new(complex))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vietoris_rips_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.866), // Approximately equilateral triangle
        ];
        
        let complex = vietoris_rips_complex(&points, 1.1, 2);
        
        // Should have 3 vertices, 3 edges, and 1 triangle
        assert_eq!(complex.simplices_at_dimension(0).len(), 3);
        assert_eq!(complex.simplices_at_dimension(1).len(), 3);
        assert_eq!(complex.simplices_at_dimension(2).len(), 1);
    }

    #[test]
    fn test_cech_complex_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.866),
        ];
        
        let complex = cech_complex(&points, 0.6, 2);
        
        // Should create a valid complex
        assert!(complex.simplices_at_dimension(0).len() >= 3);
        assert!(complex.max_dimension <= 2);
    }

    #[test]
    fn test_circumcenter_triangle() {
        let p1 = Point2D::new(0.0, 0.0);
        let p2 = Point2D::new(1.0, 0.0);
        let p3 = Point2D::new(0.5, 0.866);
        
        let result = circumcenter_triangle(&p1, &p2, &p3);
        assert!(result.is_some());
        
        let (center, radius) = result.unwrap();
        
        // All three points should be approximately equidistant from center
        let d1 = center.distance_to(&p1);
        let d2 = center.distance_to(&p2);
        let d3 = center.distance_to(&p3);
        
        assert!((d1 - radius).abs() < 1e-10);
        assert!((d2 - radius).abs() < 1e-10);
        assert!((d3 - radius).abs() < 1e-10);
    }

    #[test]
    fn test_generate_combinations() {
        let combinations = generate_combinations(4, 2);
        assert_eq!(combinations.len(), 6); // C(4,2) = 6
        
        assert!(combinations.contains(&vec![0, 1]));
        assert!(combinations.contains(&vec![0, 2]));
        assert!(combinations.contains(&vec![0, 3]));
        assert!(combinations.contains(&vec![1, 2]));
        assert!(combinations.contains(&vec![1, 3]));
        assert!(combinations.contains(&vec![2, 3]));
    }

    #[test]
    fn test_build_simplicial_complex() {
        let simplices = vec![
            vec![0],
            vec![1],
            vec![2],
            vec![0, 1],
            vec![1, 2],
            vec![0, 1, 2],
        ];
        
        let complex = build_simplicial_complex(simplices);
        
        assert_eq!(complex.simplices_at_dimension(0).len(), 3);
        assert_eq!(complex.simplices_at_dimension(1).len(), 2);
        assert_eq!(complex.simplices_at_dimension(2).len(), 1);
        assert_eq!(complex.max_dimension, 2);
    }
}