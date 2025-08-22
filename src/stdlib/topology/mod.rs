//! Topological Data Analysis Module
//!
//! This module provides comprehensive topological data analysis capabilities including:
//! - Persistent homology computation
//! - Simplicial complex construction (Vietoris-Rips, Čech)
//! - Topological feature extraction
//! - Mapper algorithm for visualization

use crate::vm::{Value, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::geometry::Point2D;
use std::any::Any;
use std::collections::HashMap;

pub mod homology;
pub mod complexes;
pub mod analysis;

pub use homology::*;
pub use complexes::*;
pub use analysis::*;

/// Simplex represented by a set of vertex indices
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Simplex {
    pub vertices: Vec<usize>,
}

impl Simplex {
    pub fn new(mut vertices: Vec<usize>) -> Self {
        vertices.sort();
        Self { vertices }
    }

    pub fn dimension(&self) -> usize {
        if self.vertices.is_empty() {
            0
        } else {
            self.vertices.len() - 1
        }
    }

    /// Get all faces (sub-simplices) of this simplex
    pub fn faces(&self) -> Vec<Simplex> {
        let n = self.vertices.len();
        if n <= 1 {
            return Vec::new();
        }

        let mut faces = Vec::new();
        for i in 0..n {
            let mut face_vertices = self.vertices.clone();
            face_vertices.remove(i);
            faces.push(Simplex::new(face_vertices));
        }
        faces
    }

    /// Check if this simplex is a face of another simplex
    pub fn is_face_of(&self, other: &Simplex) -> bool {
        self.vertices.iter().all(|&v| other.vertices.contains(&v))
    }
}

/// Filtration value for persistent homology
#[derive(Debug, Clone, PartialEq)]
pub struct Filtration {
    pub simplices: Vec<(Simplex, f64)>, // (simplex, birth_time)
}

impl Filtration {
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
        }
    }

    pub fn add_simplex(&mut self, simplex: Simplex, birth_time: f64) {
        self.simplices.push((simplex, birth_time));
    }

    pub fn sort_by_filtration(&mut self) {
        self.simplices.sort_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.dimension().cmp(&b.0.dimension()))
        });
    }
}

/// Persistence interval (birth, death)
#[derive(Debug, Clone, PartialEq)]
pub struct PersistenceInterval {
    pub birth: f64,
    pub death: f64,
    pub dimension: usize,
}

impl PersistenceInterval {
    pub fn new(birth: f64, death: f64, dimension: usize) -> Self {
        Self { birth, death, dimension }
    }

    pub fn persistence(&self) -> f64 {
        if self.death == f64::INFINITY {
            f64::INFINITY
        } else {
            self.death - self.birth
        }
    }

    pub fn is_infinite(&self) -> bool {
        self.death == f64::INFINITY
    }
}

/// Persistence diagram
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub intervals: Vec<PersistenceInterval>,
    pub max_dimension: usize,
}

impl PersistenceDiagram {
    pub fn new(intervals: Vec<PersistenceInterval>) -> Self {
        let max_dimension = intervals.iter()
            .map(|interval| interval.dimension)
            .max()
            .unwrap_or(0);
        
        Self { intervals, max_dimension }
    }

    pub fn intervals_by_dimension(&self, dimension: usize) -> Vec<&PersistenceInterval> {
        self.intervals.iter()
            .filter(|interval| interval.dimension == dimension)
            .collect()
    }

    pub fn finite_intervals(&self) -> Vec<&PersistenceInterval> {
        self.intervals.iter()
            .filter(|interval| !interval.is_infinite())
            .collect()
    }

    pub fn infinite_intervals(&self) -> Vec<&PersistenceInterval> {
        self.intervals.iter()
            .filter(|interval| interval.is_infinite())
            .collect()
    }
}

/// Simplicial complex structure
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    pub vertices: Vec<usize>,
    pub simplices: HashMap<usize, Vec<Simplex>>, // dimension -> simplices
    pub max_dimension: usize,
}

impl SimplicialComplex {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            simplices: HashMap::new(),
            max_dimension: 0,
        }
    }

    pub fn add_simplex(&mut self, simplex: Simplex) {
        let dim = simplex.dimension();
        
        // Add all vertices
        for &vertex in &simplex.vertices {
            if !self.vertices.contains(&vertex) {
                self.vertices.push(vertex);
            }
        }

        // Add the simplex
        self.simplices.entry(dim).or_insert_with(Vec::new).push(simplex.clone());
        self.max_dimension = self.max_dimension.max(dim);

        // Add all faces recursively
        for face in simplex.faces() {
            if !self.contains_simplex(&face) {
                self.add_simplex(face);
            }
        }
    }

    pub fn contains_simplex(&self, simplex: &Simplex) -> bool {
        if let Some(simplices_at_dim) = self.simplices.get(&simplex.dimension()) {
            simplices_at_dim.contains(simplex)
        } else {
            false
        }
    }

    pub fn simplices_at_dimension(&self, dimension: usize) -> Vec<&Simplex> {
        self.simplices.get(&dimension)
            .map(|simplices| simplices.iter().collect())
            .unwrap_or_else(Vec::new)
    }

    pub fn num_simplices(&self) -> usize {
        self.simplices.values().map(|s| s.len()).sum()
    }

    pub fn euler_characteristic(&self) -> i64 {
        let mut chi = 0i64;
        for (dim, simplices) in &self.simplices {
            if dim % 2 == 0 {
                chi += simplices.len() as i64;
            } else {
                chi -= simplices.len() as i64;
            }
        }
        chi
    }
}

/// Topological features extracted from data
#[derive(Debug, Clone)]
pub struct TopologicalFeatures {
    pub betti_numbers: Vec<usize>,
    pub persistence_diagram: PersistenceDiagram,
    pub euler_characteristic: i64,
    pub num_components: usize,
    pub num_holes: usize,
    pub num_voids: usize,
}

impl TopologicalFeatures {
    pub fn new(
        betti_numbers: Vec<usize>,
        persistence_diagram: PersistenceDiagram,
        euler_characteristic: i64,
    ) -> Self {
        let num_components = betti_numbers.get(0).copied().unwrap_or(0);
        let num_holes = betti_numbers.get(1).copied().unwrap_or(0);
        let num_voids = betti_numbers.get(2).copied().unwrap_or(0);

        Self {
            betti_numbers,
            persistence_diagram,
            euler_characteristic,
            num_components,
            num_holes,
            num_voids,
        }
    }
}

/// Foreign object for simplicial complexes
impl Foreign for SimplicialComplex {
    fn type_name(&self) -> &'static str {
        "SimplicialComplex"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "dimension" => Ok(Value::Integer(self.max_dimension as i64)),
            "numVertices" => Ok(Value::Integer(self.vertices.len() as i64)),
            "numSimplices" => Ok(Value::Integer(self.num_simplices() as i64)),
            "eulerCharacteristic" => Ok(Value::Integer(self.euler_characteristic())),
            "bettiNumbers" => {
                let betti = compute_betti_numbers(self);
                let betti_values: Vec<Value> = betti.into_iter()
                    .map(|b| Value::Integer(b as i64))
                    .collect();
                Ok(Value::List(betti_values))
            }
            "simplicesAtDimension" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "simplicesAtDimension".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let dim = match &args[0] {
                    Value::Integer(d) => *d as usize,
                    _ => return Err(ForeignError::TypeError {
                        expected: "integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let simplices = self.simplices_at_dimension(dim);
                let simplex_values: Vec<Value> = simplices.into_iter()
                    .map(|s| {
                        let vertices: Vec<Value> = s.vertices.iter()
                            .map(|&v| Value::Integer(v as i64))
                            .collect();
                        Value::List(vertices)
                    })
                    .collect();
                Ok(Value::List(simplex_values))
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

/// Foreign object for persistence diagrams
impl Foreign for PersistenceDiagram {
    fn type_name(&self) -> &'static str {
        "PersistenceDiagram"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "intervals" => {
                let interval_values: Vec<Value> = self.intervals.iter()
                    .map(|interval| {
                        Value::List(vec![
                            Value::Real(interval.birth),
                            Value::Real(interval.death),
                            Value::Integer(interval.dimension as i64),
                        ])
                    })
                    .collect();
                Ok(Value::List(interval_values))
            }
            "intervalsByDimension" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "intervalsByDimension".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let dim = match &args[0] {
                    Value::Integer(d) => *d as usize,
                    _ => return Err(ForeignError::TypeError {
                        expected: "integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let intervals = self.intervals_by_dimension(dim);
                let interval_values: Vec<Value> = intervals.into_iter()
                    .map(|interval| {
                        Value::List(vec![
                            Value::Real(interval.birth),
                            Value::Real(interval.death),
                        ])
                    })
                    .collect();
                Ok(Value::List(interval_values))
            }
            "finiteIntervals" => {
                let intervals = self.finite_intervals();
                let interval_values: Vec<Value> = intervals.into_iter()
                    .map(|interval| {
                        Value::List(vec![
                            Value::Real(interval.birth),
                            Value::Real(interval.death),
                            Value::Integer(interval.dimension as i64),
                        ])
                    })
                    .collect();
                Ok(Value::List(interval_values))
            }
            "infiniteIntervals" => {
                let intervals = self.infinite_intervals();
                let interval_values: Vec<Value> = intervals.into_iter()
                    .map(|interval| {
                        Value::List(vec![
                            Value::Real(interval.birth),
                            Value::Integer(interval.dimension as i64),
                        ])
                    })
                    .collect();
                Ok(Value::List(interval_values))
            }
            "maxDimension" => Ok(Value::Integer(self.max_dimension as i64)),
            "numIntervals" => Ok(Value::Integer(self.intervals.len() as i64)),
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

/// Foreign object for topological features
impl Foreign for TopologicalFeatures {
    fn type_name(&self) -> &'static str {
        "TopologicalFeatures"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "bettiNumbers" => {
                let betti_values: Vec<Value> = self.betti_numbers.iter()
                    .map(|&b| Value::Integer(b as i64))
                    .collect();
                Ok(Value::List(betti_values))
            }
            "eulerCharacteristic" => Ok(Value::Integer(self.euler_characteristic)),
            "numComponents" => Ok(Value::Integer(self.num_components as i64)),
            "numHoles" => Ok(Value::Integer(self.num_holes as i64)),
            "numVoids" => Ok(Value::Integer(self.num_voids as i64)),
            "persistenceDiagram" => {
                Ok(Value::LyObj(LyObj::new(Box::new(self.persistence_diagram.clone()))))
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

/// Helper function to convert Value to vector of points
pub fn value_to_points(value: &Value) -> VmResult<Vec<Point2D>> {
    use crate::stdlib::geometry::value_to_points as geom_value_to_points;
    geom_value_to_points(value)
}

/// Helper function to compute Betti numbers from simplicial complex
pub fn compute_betti_numbers(complex: &SimplicialComplex) -> Vec<usize> {
    // Simplified Betti number computation
    // In a full implementation, this would use homology computation
    let mut betti = vec![0; complex.max_dimension + 1];
    
    // B_0 = number of connected components (simplified)
    if let Some(vertices) = complex.simplices.get(&0) {
        betti[0] = vertices.len();
    }
    
    // For higher dimensions, use Euler characteristic approximation
    let chi = complex.euler_characteristic();
    if complex.max_dimension >= 1 {
        // B_1 = number of holes (simplified estimate)
        let num_edges = complex.simplices.get(&1).map(|s| s.len()).unwrap_or(0);
        let num_vertices = complex.simplices.get(&0).map(|s| s.len()).unwrap_or(0);
        if num_edges > num_vertices {
            betti[1] = num_edges - num_vertices + 1;
        }
    }
    
    betti
}