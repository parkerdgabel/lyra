//! Computational Geometry Module
//!
//! This module provides comprehensive computational geometry algorithms including:
//! - Convex hull computation (Graham scan)
//! - Voronoi diagrams and Delaunay triangulation
//! - Geometric queries (point-in-polygon, intersections)
//! - Advanced geometric operations (Minkowski sum, shape matching)

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;

pub mod convex_hull;
pub mod triangulation;
pub mod queries;
pub mod operations;

pub use convex_hull::*;
pub use triangulation::*;
pub use queries::*;
pub use operations::*;

/// Point in 2D space
#[derive(Debug, Clone, PartialEq)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn distance_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }

    pub fn distance_squared_to(&self, other: &Point2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx * dx + dy * dy
    }

    /// Cross product for orientation test
    pub fn cross_product(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> f64 {
        (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x)
    }

    /// Check if three points are in counter-clockwise order
    pub fn ccw(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> bool {
        Self::cross_product(p1, p2, p3) > 0.0
    }

    /// Check if three points are collinear
    pub fn collinear(p1: &Point2D, p2: &Point2D, p3: &Point2D) -> bool {
        Self::cross_product(p1, p2, p3).abs() < 1e-10
    }
}

/// Polygon represented as a sequence of vertices
#[derive(Debug, Clone)]
pub struct Polygon {
    pub vertices: Vec<Point2D>,
}

impl Polygon {
    pub fn new(vertices: Vec<Point2D>) -> Self {
        Self { vertices }
    }

    pub fn area(&self) -> f64 {
        if self.vertices.len() < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        let n = self.vertices.len();
        
        for i in 0..n {
            let j = (i + 1) % n;
            area += self.vertices[i].x * self.vertices[j].y;
            area -= self.vertices[j].x * self.vertices[i].y;
        }
        
        area.abs() / 2.0
    }

    pub fn centroid(&self) -> Point2D {
        if self.vertices.is_empty() {
            return Point2D::new(0.0, 0.0);
        }

        let mut cx = 0.0;
        let mut cy = 0.0;
        
        for vertex in &self.vertices {
            cx += vertex.x;
            cy += vertex.y;
        }
        
        let n = self.vertices.len() as f64;
        Point2D::new(cx / n, cy / n)
    }
}

/// Geometric shape foreign object for complex geometric data
#[derive(Debug, Clone)]
pub struct GeometricShape {
    pub shape_type: ShapeType,
    pub data: ShapeData,
}

#[derive(Debug, Clone)]
pub enum ShapeType {
    Points,
    Polygon,
    ConvexHull,
    VoronoiDiagram,
    DelaunayTriangulation,
}

#[derive(Debug, Clone)]
pub enum ShapeData {
    Points(Vec<Point2D>),
    Polygon(Polygon),
    ConvexHull { hull: Vec<Point2D>, original_points: Vec<Point2D> },
    VoronoiDiagram { 
        sites: Vec<Point2D>, 
        cells: Vec<Vec<Point2D>>,
        edges: Vec<(Point2D, Point2D)>,
    },
    DelaunayTriangulation {
        points: Vec<Point2D>,
        triangles: Vec<(usize, usize, usize)>,
    },
}

impl GeometricShape {
    pub fn new_points(points: Vec<Point2D>) -> Self {
        Self {
            shape_type: ShapeType::Points,
            data: ShapeData::Points(points),
        }
    }

    pub fn new_polygon(polygon: Polygon) -> Self {
        Self {
            shape_type: ShapeType::Polygon,
            data: ShapeData::Polygon(polygon),
        }
    }

    pub fn new_convex_hull(hull: Vec<Point2D>, original_points: Vec<Point2D>) -> Self {
        Self {
            shape_type: ShapeType::ConvexHull,
            data: ShapeData::ConvexHull { hull, original_points },
        }
    }

    pub fn new_voronoi(sites: Vec<Point2D>, cells: Vec<Vec<Point2D>>, edges: Vec<(Point2D, Point2D)>) -> Self {
        Self {
            shape_type: ShapeType::VoronoiDiagram,
            data: ShapeData::VoronoiDiagram { sites, cells, edges },
        }
    }

    pub fn new_delaunay(points: Vec<Point2D>, triangles: Vec<(usize, usize, usize)>) -> Self {
        Self {
            shape_type: ShapeType::DelaunayTriangulation,
            data: ShapeData::DelaunayTriangulation { points, triangles },
        }
    }

    pub fn get_points(&self) -> Vec<Point2D> {
        match &self.data {
            ShapeData::Points(points) => points.clone(),
            ShapeData::Polygon(polygon) => polygon.vertices.clone(),
            ShapeData::ConvexHull { original_points, .. } => original_points.clone(),
            ShapeData::VoronoiDiagram { sites, .. } => sites.clone(),
            ShapeData::DelaunayTriangulation { points, .. } => points.clone(),
        }
    }
}

impl Foreign for GeometricShape {
    fn type_name(&self) -> &'static str {
        match self.shape_type {
            ShapeType::Points => "Points",
            ShapeType::Polygon => "Polygon", 
            ShapeType::ConvexHull => "ConvexHull",
            ShapeType::VoronoiDiagram => "VoronoiDiagram",
            ShapeType::DelaunayTriangulation => "DelaunayTriangulation",
        }
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getPoints" => {
                let points = self.get_points();
                let point_values: Vec<Value> = points.into_iter().map(|p| {
                    Value::List(vec![Value::Real(p.x), Value::Real(p.y)])
                }).collect();
                Ok(Value::List(point_values))
            }
            "getType" => {
                let type_name = match self.shape_type {
                    ShapeType::Points => "Points",
                    ShapeType::Polygon => "Polygon",
                    ShapeType::ConvexHull => "ConvexHull", 
                    ShapeType::VoronoiDiagram => "VoronoiDiagram",
                    ShapeType::DelaunayTriangulation => "DelaunayTriangulation",
                };
                Ok(Value::String(type_name.to_string()))
            }
            "area" => {
                match &self.data {
                    ShapeData::Polygon(polygon) => Ok(Value::Real(polygon.area())),
                    ShapeData::ConvexHull { hull, .. } => {
                        let polygon = Polygon::new(hull.clone());
                        Ok(Value::Real(polygon.area()))
                    }
                    _ => Err(ForeignError::RuntimeError { 
                        message: "Area not defined for this shape type".to_string() 
                    }),
                }
            }
            "centroid" => {
                match &self.data {
                    ShapeData::Polygon(polygon) => {
                        let centroid = polygon.centroid();
                        Ok(Value::List(vec![Value::Real(centroid.x), Value::Real(centroid.y)]))
                    }
                    ShapeData::Points(points) => {
                        if points.is_empty() {
                            return Ok(Value::List(vec![Value::Real(0.0), Value::Real(0.0)]));
                        }
                        let mut cx = 0.0;
                        let mut cy = 0.0;
                        for point in points {
                            cx += point.x;
                            cy += point.y;
                        }
                        let n = points.len() as f64;
                        Ok(Value::List(vec![Value::Real(cx / n), Value::Real(cy / n)]))
                    }
                    _ => Err(ForeignError::RuntimeError { 
                        message: "Centroid not defined for this shape type".to_string() 
                    }),
                }
            }
            "length" => {
                let points = self.get_points();
                Ok(Value::Integer(points.len() as i64))
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

/// Helper function to convert Value to Point2D
pub fn value_to_point(value: &Value) -> VmResult<Point2D> {
    match value {
        Value::List(coords) => {
            if coords.len() != 2 {
                return Err(VmError::TypeError {
                    expected: "2D point [x, y]".to_string(),
                    actual: format!("list with {} elements", coords.len()),
                });
            }
            
            let x = match &coords[0] {
                Value::Real(x) => *x,
                Value::Integer(x) => *x as f64,
                _ => return Err(VmError::TypeError {
                    expected: "numeric x coordinate".to_string(),
                    actual: format!("{:?}", coords[0]),
                }),
            };
            
            let y = match &coords[1] {
                Value::Real(y) => *y,
                Value::Integer(y) => *y as f64,
                _ => return Err(VmError::TypeError {
                    expected: "numeric y coordinate".to_string(),
                    actual: format!("{:?}", coords[1]),
                }),
            };
            
            Ok(Point2D::new(x, y))
        }
        _ => Err(VmError::TypeError {
            expected: "2D point [x, y]".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Helper function to convert Value to Vec<Point2D>
pub fn value_to_points(value: &Value) -> VmResult<Vec<Point2D>> {
    match value {
        Value::List(point_values) => {
            let mut points = Vec::new();
            for point_value in point_values {
                points.push(value_to_point(point_value)?);
            }
            Ok(points)
        }
        _ => Err(VmError::TypeError {
            expected: "list of 2D points".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

/// Helper function to convert points to Value
pub fn points_to_value(points: &[Point2D]) -> Value {
    let point_values: Vec<Value> = points.iter().map(|p| {
        Value::List(vec![Value::Real(p.x), Value::Real(p.y)])
    }).collect();
    Value::List(point_values)
}