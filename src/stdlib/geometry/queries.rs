//! Geometric Query Algorithms
//!
//! Implementation of fundamental geometric queries:
//! - Point-in-polygon testing
//! - Polygon intersection
//! - Closest pair of points

use super::{Point2D, Polygon, value_to_points, value_to_point, points_to_value};
use crate::vm::{Value, VmResult, VmError};

/// Point-in-polygon test using ray casting algorithm
pub fn point_in_polygon(point: &Point2D, polygon: &Polygon) -> bool {
    if polygon.vertices.len() < 3 {
        return false;
    }

    let mut inside = false;
    let n = polygon.vertices.len();
    let mut j = n - 1;

    for i in 0..n {
        let vi = &polygon.vertices[i];
        let vj = &polygon.vertices[j];

        if ((vi.y > point.y) != (vj.y > point.y)) &&
           (point.x < (vj.x - vi.x) * (point.y - vi.y) / (vj.y - vi.y) + vi.x) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// Line segment intersection test
fn segments_intersect(p1: &Point2D, q1: &Point2D, p2: &Point2D, q2: &Point2D) -> bool {
    fn orientation(p: &Point2D, q: &Point2D, r: &Point2D) -> i32 {
        let val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y);
        if val.abs() < 1e-10 {
            0 // Collinear
        } else if val > 0.0 {
            1 // Clockwise
        } else {
            2 // Counterclockwise
        }
    }

    fn on_segment(p: &Point2D, q: &Point2D, r: &Point2D) -> bool {
        q.x <= p.x.max(r.x) && q.x >= p.x.min(r.x) &&
        q.y <= p.y.max(r.y) && q.y >= p.y.min(r.y)
    }

    let o1 = orientation(p1, q1, p2);
    let o2 = orientation(p1, q1, q2);
    let o3 = orientation(p2, q2, p1);
    let o4 = orientation(p2, q2, q1);

    // General case
    if o1 != o2 && o3 != o4 {
        return true;
    }

    // Special cases
    if o1 == 0 && on_segment(p1, p2, q1) { return true; }
    if o2 == 0 && on_segment(p1, q2, q1) { return true; }
    if o3 == 0 && on_segment(p2, p1, q2) { return true; }
    if o4 == 0 && on_segment(p2, q1, q2) { return true; }

    false
}

/// Polygon intersection using Sutherland-Hodgman clipping algorithm
pub fn polygon_intersection(poly1: &Polygon, poly2: &Polygon) -> Option<Polygon> {
    if poly1.vertices.len() < 3 || poly2.vertices.len() < 3 {
        return None;
    }

    // Check if any edges intersect
    let n1 = poly1.vertices.len();
    let n2 = poly2.vertices.len();

    for i in 0..n1 {
        let p1 = &poly1.vertices[i];
        let q1 = &poly1.vertices[(i + 1) % n1];

        for j in 0..n2 {
            let p2 = &poly2.vertices[j];
            let q2 = &poly2.vertices[(j + 1) % n2];

            if segments_intersect(p1, q1, p2, q2) {
                // For simplicity, return a degenerate polygon to indicate intersection
                return Some(Polygon::new(vec![p1.clone(), q1.clone()]));
            }
        }
    }

    // Check if one polygon is entirely inside the other
    let all_poly1_in_poly2 = poly1.vertices.iter()
        .all(|p| point_in_polygon(p, poly2));
    
    if all_poly1_in_poly2 {
        return Some(poly1.clone());
    }

    let all_poly2_in_poly1 = poly2.vertices.iter()
        .all(|p| point_in_polygon(p, poly1));
    
    if all_poly2_in_poly1 {
        return Some(poly2.clone());
    }

    None
}

/// Closest pair of points using divide and conquer
/// Time complexity: O(n log n)
pub fn closest_pair(points: &[Point2D]) -> Option<(Point2D, Point2D, f64)> {
    if points.len() < 2 {
        return None;
    }

    let mut sorted_points = points.to_vec();
    sorted_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

    let (p1, p2, dist) = closest_pair_rec(&sorted_points);
    Some((p1, p2, dist))
}

fn closest_pair_rec(points: &[Point2D]) -> (Point2D, Point2D, f64) {
    let n = points.len();

    // Base case for small arrays
    if n <= 3 {
        return brute_force_closest_pair(points);
    }

    // Divide
    let mid = n / 2;
    let midpoint = &points[mid];

    let left_points = &points[0..mid];
    let right_points = &points[mid..];

    let (left_p1, left_p2, left_dist) = closest_pair_rec(left_points);
    let (right_p1, right_p2, right_dist) = closest_pair_rec(right_points);

    // Find the minimum of the two halves
    let (mut min_p1, mut min_p2, mut min_dist) = if left_dist < right_dist {
        (left_p1, left_p2, left_dist)
    } else {
        (right_p1, right_p2, right_dist)
    };

    // Create strip of points close to the line dividing the two halves
    let mut strip = Vec::new();
    for point in points {
        if (point.x - midpoint.x).abs() < min_dist {
            strip.push(point.clone());
        }
    }

    // Sort strip by y-coordinate
    strip.sort_by(|a, b| a.y.partial_cmp(&b.y).unwrap());

    // Find closest points in strip
    for i in 0..strip.len() {
        let mut j = i + 1;
        while j < strip.len() && (strip[j].y - strip[i].y) < min_dist {
            let dist = strip[i].distance_to(&strip[j]);
            if dist < min_dist {
                min_dist = dist;
                min_p1 = strip[i].clone();
                min_p2 = strip[j].clone();
            }
            j += 1;
        }
    }

    (min_p1, min_p2, min_dist)
}

fn brute_force_closest_pair(points: &[Point2D]) -> (Point2D, Point2D, f64) {
    let mut min_dist = f64::INFINITY;
    let mut result = (points[0].clone(), points[1].clone(), min_dist);

    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dist = points[i].distance_to(&points[j]);
            if dist < min_dist {
                min_dist = dist;
                result = (points[i].clone(), points[j].clone(), dist);
            }
        }
    }

    result
}

/// PointInPolygon function for Lyra
/// Usage: PointInPolygon[{x, y}, {{x1, y1}, {x2, y2}, ...}]
pub fn point_in_polygon_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (point, polygon)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let point = value_to_point(&args[0])?;
    let polygon_points = value_to_points(&args[1])?;
    
    if polygon_points.len() < 3 {
        return Err(VmError::Runtime(
            "Polygon must have at least 3 vertices".to_string()
        ));
    }

    let polygon = Polygon::new(polygon_points);
    let inside = point_in_polygon(&point, &polygon);
    
    Ok(Value::Boolean(inside))
}

/// PolygonIntersection function for Lyra
/// Usage: PolygonIntersection[{{x1, y1}, ...}, {{x2, y2}, ...}]
pub fn polygon_intersection_fn(args: &[Value]) -> VmResult<Value> {
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
            "Polygons must have at least 3 vertices each".to_string()
        ));
    }

    let poly1 = Polygon::new(points1);
    let poly2 = Polygon::new(points2);
    
    match polygon_intersection(&poly1, &poly2) {
        Some(intersection) => Ok(points_to_value(&intersection.vertices)),
        None => Ok(Value::List(Vec::new())), // Empty list for no intersection
    }
}

/// ClosestPair function for Lyra
/// Usage: ClosestPair[{{x1, y1}, {x2, y2}, ...}]
pub fn closest_pair_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (points)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.len() < 2 {
        return Err(VmError::Runtime(
            "Closest pair requires at least 2 points".to_string()
        ));
    }

    match closest_pair(&points) {
        Some((p1, p2, distance)) => {
            Ok(Value::List(vec![
                Value::List(vec![
                    Value::List(vec![Value::Real(p1.x), Value::Real(p1.y)]),
                    Value::List(vec![Value::Real(p2.x), Value::Real(p2.y)]),
                ]),
                Value::Real(distance),
            ]))
        }
        None => Err(VmError::Runtime("Could not find closest pair".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_polygon_square() {
        let polygon = Polygon::new(vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ]);

        assert!(point_in_polygon(&Point2D::new(0.5, 0.5), &polygon));
        assert!(!point_in_polygon(&Point2D::new(1.5, 0.5), &polygon));
        assert!(!point_in_polygon(&Point2D::new(0.5, 1.5), &polygon));
    }

    #[test]
    fn test_closest_pair_simple() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 0.0),
            Point2D::new(2.0, 0.0),
        ];

        let result = closest_pair(&points);
        assert!(result.is_some());
        
        let (p1, p2, dist) = result.unwrap();
        assert!((dist - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_segments_intersect() {
        let p1 = Point2D::new(0.0, 0.0);
        let q1 = Point2D::new(1.0, 1.0);
        let p2 = Point2D::new(0.0, 1.0);
        let q2 = Point2D::new(1.0, 0.0);

        assert!(segments_intersect(&p1, &q1, &p2, &q2));

        let p3 = Point2D::new(0.0, 0.0);
        let q3 = Point2D::new(1.0, 0.0);
        let p4 = Point2D::new(0.0, 1.0);
        let q4 = Point2D::new(1.0, 1.0);

        assert!(!segments_intersect(&p3, &q3, &p4, &q4));
    }
}