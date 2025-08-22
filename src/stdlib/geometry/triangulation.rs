//! Triangulation and Voronoi Algorithms
//!
//! Implementation of Delaunay triangulation using Bowyer-Watson algorithm
//! and Voronoi diagram construction.

use super::{Point2D, GeometricShape, value_to_points};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;

/// Triangle represented by three point indices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triangle {
    pub a: usize,
    pub b: usize, 
    pub c: usize,
}

impl Triangle {
    pub fn new(a: usize, b: usize, c: usize) -> Self {
        Self { a, b, c }
    }

    /// Check if triangle contains the given point index
    pub fn contains_vertex(&self, vertex: usize) -> bool {
        self.a == vertex || self.b == vertex || self.c == vertex
    }

    /// Get the circumcircle of the triangle
    pub fn circumcircle(&self, points: &[Point2D]) -> (Point2D, f64) {
        let pa = &points[self.a];
        let pb = &points[self.b];
        let pc = &points[self.c];

        let ax = pa.x;
        let ay = pa.y;
        let bx = pb.x;
        let by = pb.y;
        let cx = pc.x;
        let cy = pc.y;

        let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
        
        if d.abs() < 1e-10 {
            // Degenerate triangle - return center of bounding box
            let center_x = (ax + bx + cx) / 3.0;
            let center_y = (ay + by + cy) / 3.0;
            return (Point2D::new(center_x, center_y), f64::INFINITY);
        }

        let ux = (ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by);
        let uy = (ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax);

        let center_x = ux / d;
        let center_y = uy / d;
        let center = Point2D::new(center_x, center_y);
        
        let radius = center.distance_to(pa);
        
        (center, radius)
    }

    /// Check if a point is inside the circumcircle of this triangle
    pub fn point_in_circumcircle(&self, point: &Point2D, points: &[Point2D]) -> bool {
        let (center, radius) = self.circumcircle(points);
        center.distance_to(point) < radius - 1e-10
    }
}

/// Edge represented by two point indices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
}

impl Edge {
    pub fn new(a: usize, b: usize) -> Self {
        // Ensure consistent ordering
        if a < b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

/// Delaunay triangulation using Bowyer-Watson algorithm
pub fn delaunay_triangulation(points: &[Point2D]) -> Vec<Triangle> {
    if points.len() < 3 {
        return Vec::new();
    }

    // Create a super triangle that contains all points
    let (min_x, max_x, min_y, max_y) = points.iter().fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY),
        |(min_x, max_x, min_y, max_y), p| {
            (min_x.min(p.x), max_x.max(p.x), min_y.min(p.y), max_y.max(p.y))
        }
    );

    let dx = max_x - min_x;
    let dy = max_y - min_y;
    let delta_max = dx.max(dy);
    let mid_x = (min_x + max_x) / 2.0;
    let mid_y = (min_y + max_y) / 2.0;

    // Super triangle vertices
    let super_points = vec![
        Point2D::new(mid_x - 20.0 * delta_max, mid_y - delta_max),
        Point2D::new(mid_x, mid_y + 20.0 * delta_max),
        Point2D::new(mid_x + 20.0 * delta_max, mid_y - delta_max),
    ];

    // Combine points with super triangle
    let mut all_points = points.to_vec();
    all_points.extend(super_points);

    let n = points.len();
    let super_triangle = Triangle::new(n, n + 1, n + 2);
    let mut triangles = vec![super_triangle];

    // Add points one by one
    for (i, point) in points.iter().enumerate() {
        let mut bad_triangles = Vec::new();
        
        // Find triangles whose circumcircle contains the point
        for (j, triangle) in triangles.iter().enumerate() {
            if triangle.point_in_circumcircle(point, &all_points) {
                bad_triangles.push(j);
            }
        }

        // Find the boundary of the polygonal hole
        let mut polygon = Vec::new();
        for &bad_idx in &bad_triangles {
            let triangle = &triangles[bad_idx];
            let edges = vec![
                Edge::new(triangle.a, triangle.b),
                Edge::new(triangle.b, triangle.c),
                Edge::new(triangle.c, triangle.a),
            ];

            for edge in edges {
                let mut shared = false;
                for &other_idx in &bad_triangles {
                    if other_idx == bad_idx {
                        continue;
                    }
                    let other = &triangles[other_idx];
                    let other_edges = vec![
                        Edge::new(other.a, other.b),
                        Edge::new(other.b, other.c),
                        Edge::new(other.c, other.a),
                    ];
                    if other_edges.contains(&edge) {
                        shared = true;
                        break;
                    }
                }
                if !shared {
                    polygon.push(edge);
                }
            }
        }

        // Remove bad triangles
        let mut new_triangles = Vec::new();
        for (j, triangle) in triangles.into_iter().enumerate() {
            if !bad_triangles.contains(&j) {
                new_triangles.push(triangle);
            }
        }
        triangles = new_triangles;

        // Re-triangulate the polygonal hole
        for edge in polygon {
            triangles.push(Triangle::new(edge.a, edge.b, i));
        }
    }

    // Remove triangles that contain super triangle vertices
    triangles.retain(|triangle| {
        !triangle.contains_vertex(n) && 
        !triangle.contains_vertex(n + 1) && 
        !triangle.contains_vertex(n + 2)
    });

    triangles
}

/// Construct Voronoi diagram from Delaunay triangulation
pub fn voronoi_diagram(points: &[Point2D]) -> (Vec<Vec<Point2D>>, Vec<(Point2D, Point2D)>) {
    let triangles = delaunay_triangulation(points);
    
    // Compute circumcenters
    let circumcenters: Vec<Point2D> = triangles.iter()
        .map(|triangle| triangle.circumcircle(points).0)
        .collect();

    // For each point, find its Voronoi cell
    let mut cells = vec![Vec::new(); points.len()];
    let mut edges = Vec::new();

    // Build adjacency information
    for (i, triangle) in triangles.iter().enumerate() {
        for &vertex in &[triangle.a, triangle.b, triangle.c] {
            if vertex < points.len() {
                cells[vertex].push(circumcenters[i].clone());
            }
        }
    }

    // Sort circumcenters around each point to form proper polygons
    for (point_idx, cell) in cells.iter_mut().enumerate() {
        if cell.is_empty() {
            continue;
        }

        let center = &points[point_idx];
        cell.sort_by(|a, b| {
            let angle_a = (a.y - center.y).atan2(a.x - center.x);
            let angle_b = (b.y - center.y).atan2(b.x - center.x);
            angle_a.partial_cmp(&angle_b).unwrap()
        });
    }

    // Extract edges
    for triangle in &triangles {
        let center = triangle.circumcircle(points).0;
        
        // Find adjacent triangles and connect their circumcenters
        for other_triangle in &triangles {
            if triangle == other_triangle {
                continue;
            }
            
            // Check if they share an edge
            let shared_vertices = [triangle.a, triangle.b, triangle.c].iter()
                .filter(|&&v| other_triangle.contains_vertex(v))
                .count();
                
            if shared_vertices == 2 {
                let other_center = other_triangle.circumcircle(points).0;
                edges.push((center.clone(), other_center));
            }
        }
    }

    (cells, edges)
}

/// VoronoiDiagram function for Lyra
/// Usage: VoronoiDiagram[{{x1, y1}, {x2, y2}, ...}]
pub fn voronoi_diagram_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (points)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.len() < 3 {
        return Err(VmError::Runtime(
            "Voronoi diagram requires at least 3 points".to_string()
        ));
    }

    let (cells, edges) = voronoi_diagram(&points);
    let shape = GeometricShape::new_voronoi(points, cells, edges);
    
    Ok(Value::LyObj(LyObj::new(Box::new(shape))))
}

/// DelaunayTriangulation function for Lyra
/// Usage: DelaunayTriangulation[{{x1, y1}, {x2, y2}, ...}]
pub fn delaunay_triangulation_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (points)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let points = value_to_points(&args[0])?;
    
    if points.len() < 3 {
        return Err(VmError::Runtime(
            "Delaunay triangulation requires at least 3 points".to_string()
        ));
    }

    let triangles = delaunay_triangulation(&points);
    let triangle_tuples: Vec<(usize, usize, usize)> = triangles.iter()
        .map(|t| (t.a, t.b, t.c))
        .collect();
    
    let shape = GeometricShape::new_delaunay(points, triangle_tuples);
    
    Ok(Value::LyObj(LyObj::new(Box::new(shape))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delaunay_triangulation_square() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(1.0, 1.0),
            Point2D::new(0.0, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 2); // Square should be split into 2 triangles
    }

    #[test]
    fn test_delaunay_triangulation_triangle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let triangles = delaunay_triangulation(&points);
        assert_eq!(triangles.len(), 1);
    }

    #[test]
    fn test_circumcircle() {
        let points = vec![
            Point2D::new(0.0, 0.0),
            Point2D::new(1.0, 0.0),
            Point2D::new(0.5, 1.0),
        ];

        let triangle = Triangle::new(0, 1, 2);
        let (center, radius) = triangle.circumcircle(&points);
        
        // All three points should be approximately equidistant from center
        let d1 = center.distance_to(&points[0]);
        let d2 = center.distance_to(&points[1]);
        let d3 = center.distance_to(&points[2]);
        
        assert!((d1 - radius).abs() < 1e-10);
        assert!((d2 - radius).abs() < 1e-10);
        assert!((d3 - radius).abs() < 1e-10);
    }
}