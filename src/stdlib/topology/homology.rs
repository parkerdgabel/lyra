//! Persistent Homology Computation
//!
//! Implementation of persistent homology algorithms for topological data analysis.

use super::{Simplex, Filtration, PersistenceInterval, PersistenceDiagram, SimplicialComplex};
use crate::vm::{Value, VmResult, VmError};
use crate::foreign::LyObj;
use crate::stdlib::geometry::Point2D;
use std::collections::{HashMap, HashSet};

/// Compute persistent homology using the standard algorithm
pub fn compute_persistent_homology(filtration: &Filtration, max_dimension: usize) -> PersistenceDiagram {
    let mut intervals = Vec::new();
    
    // Build boundary matrix
    let boundary_matrix = build_boundary_matrix(filtration);
    
    // Perform matrix reduction to find persistence pairs
    let pairs = matrix_reduction(&boundary_matrix, filtration);
    
    // Convert pairs to intervals
    for (birth_idx, death_idx) in pairs {
        let birth_time = filtration.simplices[birth_idx].1;
        let death_time = if let Some(death) = death_idx {
            filtration.simplices[death].1
        } else {
            f64::INFINITY
        };
        
        let dimension = filtration.simplices[birth_idx].0.dimension();
        if dimension <= max_dimension {
            intervals.push(PersistenceInterval::new(birth_time, death_time, dimension));
        }
    }
    
    PersistenceDiagram::new(intervals)
}

/// Build boundary matrix for the filtration
fn build_boundary_matrix(filtration: &Filtration) -> Vec<Vec<bool>> {
    let n = filtration.simplices.len();
    let mut matrix = vec![vec![false; n]; n];
    
    for (i, (simplex, _)) in filtration.simplices.iter().enumerate() {
        let faces = simplex.faces();
        
        for face in faces {
            // Find the index of this face in the filtration
            for (j, (other_simplex, _)) in filtration.simplices.iter().enumerate() {
                if j < i && *other_simplex == face {
                    matrix[i][j] = true;
                    break;
                }
            }
        }
    }
    
    matrix
}

/// Matrix reduction algorithm for computing persistence pairs
fn matrix_reduction(boundary_matrix: &[Vec<bool>], filtration: &Filtration) -> Vec<(usize, Option<usize>)> {
    let n = boundary_matrix.len();
    let mut reduced_matrix = boundary_matrix.to_vec();
    let mut pairs = Vec::new();
    let mut low = vec![None; n];
    
    for j in 0..n {
        // Reduce column j
        while let Some(i) = lowest_one(&reduced_matrix[j]) {
            if let Some(k) = low[i] {
                // Add column k to column j
                for row in 0..n {
                    reduced_matrix[j][row] ^= reduced_matrix[k][row];
                }
            } else {
                low[i] = Some(j);
                pairs.push((i, Some(j)));
                break;
            }
        }
    }
    
    // Add unpaired simplices (infinite intervals)
    for i in 0..n {
        if !pairs.iter().any(|(birth, death)| death == &Some(i)) {
            pairs.push((i, None));
        }
    }
    
    pairs
}

/// Find the lowest 1 in a column
fn lowest_one(column: &[bool]) -> Option<usize> {
    for (i, &val) in column.iter().enumerate().rev() {
        if val {
            return Some(i);
        }
    }
    None
}

/// Compute Betti numbers from a simplicial complex
pub fn compute_betti_numbers_full(complex: &SimplicialComplex) -> Vec<usize> {
    let mut betti = vec![0; complex.max_dimension + 1];
    
    // For each dimension, compute rank of homology group
    for dim in 0..=complex.max_dimension {
        betti[dim] = compute_homology_rank(complex, dim);
    }
    
    betti
}

/// Compute the rank of the k-th homology group
fn compute_homology_rank(complex: &SimplicialComplex, k: usize) -> usize {
    // Simplified computation using Euler characteristic
    // In a full implementation, this would use Smith normal form
    
    if k == 0 {
        // H_0 = connected components
        estimate_connected_components(complex)
    } else if k == 1 && complex.max_dimension >= 1 {
        // H_1 = loops (simplified estimate)
        let num_edges = complex.simplices_at_dimension(1).len();
        let num_vertices = complex.simplices_at_dimension(0).len();
        let num_triangles = complex.simplices_at_dimension(2).len();
        
        // Crude estimate: edges - vertices + 1 - triangles
        if num_edges > num_vertices {
            (num_edges - num_vertices + 1).saturating_sub(num_triangles)
        } else {
            0
        }
    } else {
        // Higher dimensional homology groups (simplified)
        0
    }
}

/// Estimate number of connected components
fn estimate_connected_components(complex: &SimplicialComplex) -> usize {
    let vertices = &complex.vertices;
    let edges = complex.simplices_at_dimension(1);
    
    if vertices.is_empty() {
        return 0;
    }
    
    // Use Union-Find to count components
    let mut parent: HashMap<usize, usize> = vertices.iter().map(|&v| (v, v)).collect();
    
    fn find(parent: &mut HashMap<usize, usize>, x: usize) -> usize {
        if parent[&x] != x {
            parent.insert(x, find(parent, parent[&x]));
        }
        parent[&x]
    }
    
    fn union(parent: &mut HashMap<usize, usize>, x: usize, y: usize) {
        let px = find(parent, x);
        let py = find(parent, y);
        if px != py {
            parent.insert(px, py);
        }
    }
    
    // Connect vertices that share an edge
    for edge in edges {
        if edge.vertices.len() == 2 {
            union(&mut parent, edge.vertices[0], edge.vertices[1]);
        }
    }
    
    // Count distinct components
    let mut components = HashSet::new();
    for &vertex in vertices {
        components.insert(find(&mut parent, vertex));
    }
    
    components.len()
}

/// PersistentHomology function for Lyra
/// Usage: PersistentHomology[filtration, maxDimension]
pub fn persistent_homology_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (filtration, maxDimension)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    // Parse filtration from list of [simplex, birth_time] pairs
    let filtration_data = match &args[0] {
        Value::List(items) => {
            let mut filtration = Filtration::new();
            
            for item in items {
                match item {
                    Value::List(pair) => {
                        if pair.len() != 2 {
                            return Err(VmError::TypeError {
                                expected: "[simplex, birth_time] pair".to_string(),
                                actual: format!("list with {} elements", pair.len()),
                            });
                        }
                        
                        // Parse simplex vertices
                        let vertices = match &pair[0] {
                            Value::List(vertex_list) => {
                                let mut verts = Vec::new();
                                for v in vertex_list {
                                    match v {
                                        Value::Integer(i) => verts.push(*i as usize),
                                        _ => return Err(VmError::TypeError {
                                            expected: "integer vertex index".to_string(),
                                            actual: format!("{:?}", v),
                                        }),
                                    }
                                }
                                verts
                            }
                            _ => return Err(VmError::TypeError {
                                expected: "list of vertex indices".to_string(),
                                actual: format!("{:?}", pair[0]),
                            }),
                        };
                        
                        // Parse birth time
                        let birth_time = match &pair[1] {
                            Value::Real(t) => *t,
                            Value::Integer(t) => *t as f64,
                            _ => return Err(VmError::TypeError {
                                expected: "numeric birth time".to_string(),
                                actual: format!("{:?}", pair[1]),
                            }),
                        };
                        
                        filtration.add_simplex(Simplex::new(vertices), birth_time);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "[simplex, birth_time] pair".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            
            filtration.sort_by_filtration();
            filtration
        }
        _ => return Err(VmError::TypeError {
            expected: "list of filtration data".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let max_dimension = match &args[1] {
        Value::Integer(d) => *d as usize,
        _ => return Err(VmError::TypeError {
            expected: "integer max dimension".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let diagram = compute_persistent_homology(&filtration_data, max_dimension);
    Ok(Value::LyObj(LyObj::new(Box::new(diagram))))
}

/// BettiNumbers function for Lyra
/// Usage: BettiNumbers[complex]
pub fn betti_numbers_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (complex)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let complex = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<SimplicialComplex>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "SimplicialComplex".to_string(),
                    actual: "other LyObj type".to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "SimplicialComplex".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let betti = compute_betti_numbers_full(complex);
    let betti_values: Vec<Value> = betti.into_iter()
        .map(|b| Value::Integer(b as i64))
        .collect();
    
    Ok(Value::List(betti_values))
}

/// PersistenceDiagram function for Lyra
/// Usage: PersistenceDiagram[intervals]
pub fn persistence_diagram_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (intervals)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let intervals = match &args[0] {
        Value::List(interval_list) => {
            let mut intervals = Vec::new();
            
            for interval_data in interval_list {
                match interval_data {
                    Value::List(data) => {
                        if data.len() < 2 || data.len() > 3 {
                            return Err(VmError::TypeError {
                                expected: "[birth, death] or [birth, death, dimension]".to_string(),
                                actual: format!("list with {} elements", data.len()),
                            });
                        }
                        
                        let birth = match &data[0] {
                            Value::Real(b) => *b,
                            Value::Integer(b) => *b as f64,
                            _ => return Err(VmError::TypeError {
                                expected: "numeric birth time".to_string(),
                                actual: format!("{:?}", data[0]),
                            }),
                        };
                        
                        let death = match &data[1] {
                            Value::Real(d) => *d,
                            Value::Integer(d) => *d as f64,
                            Value::Symbol(s) if s == "Infinity" => f64::INFINITY,
                            _ => return Err(VmError::TypeError {
                                expected: "numeric death time or Infinity".to_string(),
                                actual: format!("{:?}", data[1]),
                            }),
                        };
                        
                        let dimension = if data.len() == 3 {
                            match &data[2] {
                                Value::Integer(d) => *d as usize,
                                _ => return Err(VmError::TypeError {
                                    expected: "integer dimension".to_string(),
                                    actual: format!("{:?}", data[2]),
                                }),
                            }
                        } else {
                            0 // Default dimension
                        };
                        
                        intervals.push(PersistenceInterval::new(birth, death, dimension));
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list representing interval".to_string(),
                        actual: format!("{:?}", interval_data),
                    }),
                }
            }
            
            intervals
        }
        _ => return Err(VmError::TypeError {
            expected: "list of intervals".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let diagram = PersistenceDiagram::new(intervals);
    Ok(Value::LyObj(LyObj::new(Box::new(diagram))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplex_faces() {
        let simplex = Simplex::new(vec![0, 1, 2]);
        let faces = simplex.faces();
        
        assert_eq!(faces.len(), 3);
        assert!(faces.contains(&Simplex::new(vec![1, 2])));
        assert!(faces.contains(&Simplex::new(vec![0, 2])));
        assert!(faces.contains(&Simplex::new(vec![0, 1])));
    }

    #[test]
    fn test_persistence_interval() {
        let interval = PersistenceInterval::new(1.0, 3.0, 1);
        assert_eq!(interval.persistence(), 2.0);
        assert!(!interval.is_infinite());
        
        let infinite_interval = PersistenceInterval::new(1.0, f64::INFINITY, 0);
        assert!(infinite_interval.is_infinite());
    }

    #[test]
    fn test_filtration_sorting() {
        let mut filtration = Filtration::new();
        filtration.add_simplex(Simplex::new(vec![0, 1]), 2.0);
        filtration.add_simplex(Simplex::new(vec![0]), 1.0);
        filtration.add_simplex(Simplex::new(vec![1]), 1.0);
        
        filtration.sort_by_filtration();
        
        // Should be sorted by birth time, then by dimension
        assert_eq!(filtration.simplices[0].1, 1.0);
        assert_eq!(filtration.simplices[1].1, 1.0);
        assert_eq!(filtration.simplices[2].1, 2.0);
    }

    #[test]
    fn test_boundary_matrix() {
        let mut filtration = Filtration::new();
        filtration.add_simplex(Simplex::new(vec![0]), 1.0);
        filtration.add_simplex(Simplex::new(vec![1]), 1.0);
        filtration.add_simplex(Simplex::new(vec![0, 1]), 2.0);
        
        let matrix = build_boundary_matrix(&filtration);
        
        // Edge [0,1] should have boundary containing vertices 0 and 1
        assert_eq!(matrix.len(), 3);
        assert_eq!(matrix[2][0], true);  // Edge depends on vertex 0
        assert_eq!(matrix[2][1], true);  // Edge depends on vertex 1
    }
}