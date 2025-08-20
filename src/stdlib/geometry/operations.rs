//! Advanced Geometric Operations
//!
//! Implementation of advanced geometric algorithms:
//! - Minkowski sum
//! - Geometric median (Weiszfeld's algorithm)
//! - Shape matching (Hausdorff distance)
//! - Polygon decomposition

use super::{Point2D, Polygon, GeometricShape, value_to_points, points_to_value, convex_hull::convex_hull};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;
use std::f64::consts::PI;

/// Minkowski sum of two polygons
/// Returns the convex hull of all pairwise sums of vertices
pub fn minkowski_sum(poly1: &Polygon, poly2: &Polygon) -> Polygon {
    let mut sum_points = Vec::new();
    
    for p1 in &poly1.vertices {
        for p2 in &poly2.vertices {
            sum_points.push(Point2D::new(p1.x + p2.x, p1.y + p2.y));
        }
    }
    
    // The Minkowski sum of two convex polygons is convex
    let hull_points = convex_hull(&sum_points);
    Polygon::new(hull_points)
}

/// Geometric median using Weiszfeld's algorithm
/// Finds the point that minimizes the sum of distances to all given points
pub fn geometric_median(points: &[Point2D], max_iterations: usize, tolerance: f64) -> Point2D {
    if points.is_empty() {
        return Point2D::new(0.0, 0.0);
    }
    
    if points.len() == 1 {
        return points[0].clone();
    }

    // Initialize with centroid
    let mut median = Point2D::new(
        points.iter().map(|p| p.x).sum::<f64>() / points.len() as f64,
        points.iter().map(|p| p.y).sum::<f64>() / points.len() as f64,
    );

    for _ in 0..max_iterations {
        let mut numerator_x = 0.0;
        let mut numerator_y = 0.0;
        let mut denominator = 0.0;

        for point in points {
            let distance = median.distance_to(point);
            
            if distance < tolerance {
                // Point is very close, use the point itself
                return point.clone();
            }
            
            let weight = 1.0 / distance;
            numerator_x += weight * point.x;
            numerator_y += weight * point.y;
            denominator += weight;
        }

        if denominator == 0.0 {
            break;
        }

        let new_median = Point2D::new(numerator_x / denominator, numerator_y / denominator);
        
        if median.distance_to(&new_median) < tolerance {
            return new_median;
        }
        
        median = new_median;
    }

    median
}

/// Hausdorff distance between two point sets
/// Measures the maximum distance from any point in one set to the closest point in the other
pub fn hausdorff_distance(points1: &[Point2D], points2: &[Point2D]) -> f64 {
    if points1.is_empty() || points2.is_empty() {
        return f64::INFINITY;
    }

    let max_dist_1_to_2 = points1.iter().map(|p1| {
        points2.iter().map(|p2| p1.distance_to(p2)).fold(f64::INFINITY, f64::min)
    }).fold(0.0, f64::max);

    let max_dist_2_to_1 = points2.iter().map(|p2| {
        points1.iter().map(|p1| p2.distance_to(p1)).fold(f64::INFINITY, f64::min)
    }).fold(0.0, f64::max);

    max_dist_1_to_2.max(max_dist_2_to_1)
}

/// Shape similarity using Hausdorff distance
pub fn shape_similarity(shape1: &[Point2D], shape2: &[Point2D]) -> f64 {
    let hausdorff_dist = hausdorff_distance(shape1, shape2);
    
    // Convert to similarity score (0 = identical, decreasing with distance)
    if hausdorff_dist == 0.0 {
        1.0
    } else {
        1.0 / (1.0 + hausdorff_dist)
    }
}

/// Polygon triangulation using ear clipping algorithm
pub fn triangulate_polygon(polygon: &Polygon) -> Vec<(usize, usize, usize)> {
    let vertices = &polygon.vertices;
    let n = vertices.len();
    
    if n < 3 {
        return Vec::new();
    }
    
    if n == 3 {
        return vec![(0, 1, 2)];
    }

    let mut triangles = Vec::new();
    let mut indices: Vec<usize> = (0..n).collect();

    while indices.len() > 3 {
        let mut ear_found = false;
        
        for i in 0..indices.len() {
            let prev = indices[(i + indices.len() - 1) % indices.len()];
            let curr = indices[i];
            let next = indices[(i + 1) % indices.len()];
            
            let p1 = &vertices[prev];
            let p2 = &vertices[curr];
            let p3 = &vertices[next];
            
            // Check if this forms a valid ear (convex and no points inside)
            if is_ear(p1, p2, p3, vertices, &indices) {
                triangles.push((prev, curr, next));
                indices.remove(i);
                ear_found = true;
                break;
            }
        }
        
        if !ear_found {
            // Fallback: just take the first three remaining vertices
            if indices.len() >= 3 {
                triangles.push((indices[0], indices[1], indices[2]));
                indices.remove(1);
            } else {
                break;
            }
        }
    }
    
    if indices.len() == 3 {
        triangles.push((indices[0], indices[1], indices[2]));
    }
    
    triangles
}

fn is_ear(p1: &Point2D, p2: &Point2D, p3: &Point2D, vertices: &[Point2D], indices: &[usize]) -> bool {
    // Check if triangle is convex (counter-clockwise)
    let cross = Point2D::cross_product(p1, p2, p3);
    if cross <= 0.0 {
        return false;
    }
    
    // Check if any other vertex is inside the triangle
    for &idx in indices {
        let vertex = &vertices[idx];
        if vertex == p1 || vertex == p2 || vertex == p3 {
            continue;
        }
        
        if point_in_triangle(vertex, p1, p2, p3) {
            return false;
        }
    }
    
    true
}

fn point_in_triangle(point: &Point2D, a: &Point2D, b: &Point2D, c: &Point2D) -> bool {
    let d1 = Point2D::cross_product(a, b, point);
    let d2 = Point2D::cross_product(b, c, point);
    let d3 = Point2D::cross_product(c, a, point);
    
    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);
    
    !(has_neg && has_pos)
}

/// MinkowskiSum function for Lyra
/// Usage: MinkowskiSum[{{x1, y1}, ...}, {{x2, y2}, ...}]
pub fn minkowski_sum_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (polygon1, polygon2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points1 = value_to_points(&args[0])?;
    let points2 = value_to_points(&args[1])?;
    
    if points1.len() < 3 || points2.len() < 3 {
        return Err(VmError::Runtime(
            "Minkowski sum requires polygons with at least 3 vertices each".to_string()
        ));
    }

    let poly1 = Polygon::new(points1);
    let poly2 = Polygon::new(points2);
    let result = minkowski_sum(&poly1, &poly2);
    
    Ok(points_to_value(&result.vertices))
}

/// GeometricMedian function for Lyra
/// Usage: GeometricMedian[{{x1, y1}, {x2, y2}, ...}]
pub fn geometric_median_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (points)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.is_empty() {
        return Err(VmError::Runtime(
            "Geometric median requires at least 1 point".to_string()
        ));
    }

    let median = geometric_median(&points, 1000, 1e-10);
    
    Ok(Value::List(vec![Value::Real(median.x), Value::Real(median.y)]))
}

/// ShapeMatching function for Lyra
/// Usage: ShapeMatching[{{x1, y1}, ...}, {{x2, y2}, ...}]
pub fn shape_matching_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (shape1, shape2)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points1 = value_to_points(&args[0])?;
    let points2 = value_to_points(&args[1])?;
    
    if points1.is_empty() || points2.is_empty() {
        return Err(VmError::Runtime(
            "Shape matching requires non-empty point sets".to_string()
        ));
    }

    let similarity = shape_similarity(&points1, &points2);
    let hausdorff_dist = hausdorff_distance(&points1, &points2);
    
    Ok(Value::List(vec![
        Value::Real(similarity),
        Value::Real(hausdorff_dist),
    ]))
}

/// PolygonDecomposition function for Lyra
/// Usage: PolygonDecomposition[{{x1, y1}, {x2, y2}, ...}]
pub fn polygon_decomposition_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (polygon)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.len() < 3 {
        return Err(VmError::Runtime(
            "Polygon decomposition requires at least 3 vertices".to_string()
        ));
    }

    let polygon = Polygon::new(points);
    let triangles = triangulate_polygon(&polygon);
    
    let triangle_values: Vec<Value> = triangles.into_iter().map(|(a, b, c)| {
        Value::List(vec![
            Value::List(vec![Value::Real(polygon.vertices[a].x), Value::Real(polygon.vertices[a].y)]),
            Value::List(vec![Value::Real(polygon.vertices[b].x), Value::Real(polygon.vertices[b].y)]),
            Value::List(vec![Value::Real(polygon.vertices[c].x), Value::Real(polygon.vertices[c].y)]),
        ])
    }).collect();
    
    Ok(Value::List(triangle_values))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minkowski_sum_squares() {
        let square1 = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ]);

        let square2 = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(0.5, 0.0),
            Point2D::new(0.5, 0.5),
            Point2D::new(0.0, 0.5),
        ]);

        let result = minkowski_sum(&square1, &square2);
        assert!(!result.vertices.is_empty());
        
        // The result should be larger than both input polygons
        assert!(result.area() > square1.area());
        assert!(result.area() > square2.area());
    }

    #[test]
    fn test_geometric_median_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let median = geometric_median(&points, 100, 1e-6);
        
        // Median should be somewhere in the triangle
        assert!(median.x >= 0.0 && median.x <= 1.0);
        assert!(median.y >= 0.0 && median.y <= 1.0);
    }

    #[test]
    fn test_hausdorff_distance_identical() {
        let points1 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let points2 = points1.clone();
        let distance = hausdorff_distance(&points1, &points2);
        
        assert!(distance < 1e-10);
    }

    #[test]
    fn test_triangulate_square() {
        let square = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ]);

        let triangles = triangulate_polygon(&square);
        assert_eq!(triangles.len(), 2); // Square should decompose into 2 triangles
    }

    #[test]
    fn test_shape_similarity_identical() {
        let points1 = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let similarity = shape_similarity(&points1, &points1);
        assert!((similarity - 1.0).abs() < 1e-10);
    }
}