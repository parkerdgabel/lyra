//! Convex Hull Algorithms
//!
//! Implementation of Graham scan algorithm for computing 2D convex hulls.

use super::{Point2D, GeometricShape, value_to_points};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;

/// Compute convex hull using Graham scan algorithm
/// Time complexity: O(n log n)
pub fn convex_hull(points: &[Point2D]) -> Vec<Point2D> {
    if points.len() < 3 {
        return points.to_vec();
    }

    // Find the bottom-most point (and leftmost in case of tie)
    let mut bottom_point = points[0].clone();
    for point in points.iter().skip(1) {
        if point.y < bottom_point.y || (point.y == bottom_point.y && point.x < bottom_point.x) {
            bottom_point = point.clone();
        }
    }

    // Sort points by polar angle with respect to bottom point
    let mut sorted_points: Vec<Point2D> = points.iter()
        .filter(|&p| *p != bottom_point)
        .cloned()
        .collect();

    sorted_points.sort_by(|a, b| {
        let cross = Point2D::cross_product(&bottom_point, a, b);
        if cross == 0.0 {
            // Collinear points - sort by distance
            let dist_a = bottom_point.distance_squared_to(a);
            let dist_b = bottom_point.distance_squared_to(b);
            dist_a.partial_cmp(&dist_b).unwrap()
        } else if cross > 0.0 {
            std::cmp::Ordering::Less
        } else {
            std::cmp::Ordering::Greater
        }
    });

    // Graham scan
    let mut hull = vec![bottom_point];
    
    for point in sorted_points {
        // Remove points that make a clockwise turn
        while hull.len() > 1 {
            let len = hull.len();
            if Point2D::ccw(&hull[len - 2], &hull[len - 1], &point) {
                break;
            }
            hull.pop();
        }
        hull.push(point);
    }

    hull
}

/// ConvexHull function for Lyra
/// Usage: ConvexHull[{{x1, y1}, {x2, y2}, ...}]
pub fn convex_hull_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (points)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.len() < 3 {
        return Err(VmError::Runtime(
            "Convex hull requires at least 3 points".to_string()
        ));
    }

    let hull_points = convex_hull(&points);
    let shape = GeometricShape::new_convex_hull(hull_points.clone(), points);
    
    Ok(Value::LyObj(LyObj::new(Box::new(shape))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convex_hull_square() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
            Point2D::new(0.5, 0.5), // Interior point
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 4);
        
        // Check that interior point is not in hull
        assert!(!hull.contains(&Point2D::new(0.5, 0.5)));
    }

    #[test]
    fn test_convex_hull_collinear() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(2.0, 0.0),
            Point2D::new(3.0, 0.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 2); // Only endpoints
        assert!(hull.contains(&Point2D::new(0.0, 0.0)));
        assert!(hull.contains(&Point2D::new(3.0, 0.0)));
    }

    #[test]
    fn test_convex_hull_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let hull = convex_hull(&points);
        assert_eq!(hull.len(), 3);
    }
}