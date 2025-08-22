//! Disjoint Set (Union-Find) implementation for Lyra
//!
//! This module provides a complete Union-Find data structure with path compression
//! and union by rank optimizations for near-constant amortized operations.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::stdlib::common::validation::validate_args;
use std::collections::HashMap;
use std::fmt;

/// Disjoint Set (Union-Find) data structure with path compression and union by rank
#[derive(Clone, Debug)]
pub struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
    size: Vec<usize>,
    count: usize, // Number of disjoint sets
}

impl DisjointSet {
    /// Create a new DisjointSet with n elements (0 to n-1)
    pub fn new(n: usize) -> Result<Self, ForeignError> {
        if n == 0 {
            return Err(ForeignError::RuntimeError {
                message: "DisjointSet size must be greater than 0".to_string(),
            });
        }
        
        Ok(Self {
            parent: (0..n).collect(), // Each element is its own parent initially
            rank: vec![0; n],         // All ranks start at 0
            size: vec![1; n],         // All sets have size 1 initially
            count: n,                 // n disjoint sets initially
        })
    }
    
    /// Find the root of element x with path compression
    pub fn find(&mut self, x: usize) -> Result<usize, ForeignError> {
        if x >= self.parent.len() {
            return Err(ForeignError::RuntimeError {
                message: format!("Element {} is out of bounds", x),
            });
        }
        
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x])?; // Path compression
        }
        
        Ok(self.parent[x])
    }
    
    /// Union two sets containing elements x and y
    pub fn union(&mut self, x: usize, y: usize) -> Result<bool, ForeignError> {
        let root_x = self.find(x)?;
        let root_y = self.find(y)?;
        
        if root_x == root_y {
            return Ok(false); // Already in the same set
        }
        
        // Union by rank: attach smaller rank tree under root of higher rank tree
        match self.rank[root_x].cmp(&self.rank[root_y]) {
            std::cmp::Ordering::Less => {
                self.parent[root_x] = root_y;
                self.size[root_y] += self.size[root_x];
            }
            std::cmp::Ordering::Greater => {
                self.parent[root_y] = root_x;
                self.size[root_x] += self.size[root_y];
            }
            std::cmp::Ordering::Equal => {
                self.parent[root_y] = root_x;
                self.size[root_x] += self.size[root_y];
                self.rank[root_x] += 1;
            }
        }
        
        self.count -= 1; // One fewer disjoint set
        Ok(true)
    }
    
    /// Check if two elements are in the same set
    pub fn connected(&mut self, x: usize, y: usize) -> Result<bool, ForeignError> {
        Ok(self.find(x)? == self.find(y)?)
    }
    
    /// Get the size of the set containing element x
    pub fn set_size(&mut self, x: usize) -> Result<usize, ForeignError> {
        let root = self.find(x)?;
        Ok(self.size[root])
    }
    
    /// Get the number of disjoint sets
    pub fn count_sets(&self) -> usize {
        self.count
    }
    
    /// Get the total number of elements
    pub fn total_elements(&self) -> usize {
        self.parent.len()
    }
    
    /// Get all elements in the same set as element x
    pub fn get_set_members(&mut self, x: usize) -> Result<Vec<usize>, ForeignError> {
        let root = self.find(x)?;
        let mut members = Vec::new();
        
        for i in 0..self.parent.len() {
            if self.find(i)? == root {
                members.push(i);
            }
        }
        
        Ok(members)
    }
    
    /// Get all disjoint sets as a list of lists
    pub fn get_all_sets(&mut self) -> Result<Vec<Vec<usize>>, ForeignError> {
        let mut sets_map: HashMap<usize, Vec<usize>> = HashMap::new();
        
        for i in 0..self.parent.len() {
            let root = self.find(i)?;
            sets_map.entry(root).or_insert_with(Vec::new).push(i);
        }
        
        Ok(sets_map.into_values().collect())
    }
    
    /// Get the root representatives of all disjoint sets
    pub fn get_roots(&mut self) -> Result<Vec<usize>, ForeignError> {
        let mut roots = Vec::new();
        
        for i in 0..self.parent.len() {
            if self.find(i)? == i {
                roots.push(i);
            }
        }
        
        Ok(roots)
    }
    
    /// Reset the data structure (make all elements disjoint again)
    pub fn reset(&mut self) {
        let n = self.parent.len();
        self.parent = (0..n).collect();
        self.rank = vec![0; n];
        self.size = vec![1; n];
        self.count = n;
    }
}

impl Foreign for DisjointSet {
    fn type_name(&self) -> &'static str {
        "DisjointSet"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let mut ds = self.clone();
        match method {
            "find" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let root = ds.find(x)?;
                Ok(Value::Integer(root as i64))
            }
            "union" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let y = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let united = ds.union(x, y)?;
                // Return the modified DisjointSet and whether union occurred
                Ok(Value::List(vec![
                    Value::LyObj(LyObj::new(Box::new(ds))),
                    Value::Boolean(united),
                ]))
            }
            "connected" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let y = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let connected = ds.connected(x, y)?;
                Ok(Value::Boolean(connected))
            }
            "setSize" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let size = ds.set_size(x)?;
                Ok(Value::Integer(size as i64))
            }
            "countSets" => {
                Ok(Value::Integer(ds.count_sets() as i64))
            }
            "totalElements" => {
                Ok(Value::Integer(ds.total_elements() as i64))
            }
            "getSetMembers" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let x = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "other".to_string(),
                    }),
                };
                
                let members = ds.get_set_members(x)?;
                let values: Vec<Value> = members.into_iter().map(|i| Value::Integer(i as i64)).collect();
                Ok(Value::List(values))
            }
            "getAllSets" => {
                let sets = ds.get_all_sets()?;
                let values: Vec<Value> = sets.into_iter().map(|set| {
                    let members: Vec<Value> = set.into_iter().map(|i| Value::Integer(i as i64)).collect();
                    Value::List(members)
                }).collect();
                Ok(Value::List(values))
            }
            "getRoots" => {
                let roots = ds.get_roots()?;
                let values: Vec<Value> = roots.into_iter().map(|i| Value::Integer(i as i64)).collect();
                Ok(Value::List(values))
            }
            "reset" => {
                ds.reset();
                Ok(Value::LyObj(LyObj::new(Box::new(ds))))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            })
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

impl fmt::Display for DisjointSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DisjointSet[elements: {}, sets: {}]", self.total_elements(), self.count_sets())
    }
}

/// Create a new DisjointSet with n elements
pub fn disjoint_set_new(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    let n = match &args[0] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::Runtime("Argument must be an integer".to_string())),
    };
    
    let ds = DisjointSet::new(n).map_err(|e| VmError::Runtime(e.to_string()))?;
    Ok(Value::LyObj(LyObj::new(Box::new(ds))))
}

/// Find the root of an element in a DisjointSet
pub fn disjoint_set_find(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("find", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a DisjointSet".to_string()))
    }
}

/// Union two sets in a DisjointSet
pub fn disjoint_set_union(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("union", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a DisjointSet".to_string()))
    }
}

/// Check if two elements are connected in a DisjointSet
pub fn disjoint_set_connected(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 3).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("connected", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a DisjointSet".to_string()))
    }
}

/// Get the size of the set containing an element
pub fn disjoint_set_set_size(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("setSize", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a DisjointSet".to_string()))
    }
}

/// Get the number of disjoint sets
pub fn disjoint_set_count(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("countSets", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a DisjointSet".to_string()))
    }
}

/// Get total number of elements
pub fn disjoint_set_total_elements(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("totalElements", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a DisjointSet".to_string()))
    }
}

/// Get all members of the set containing an element
pub fn disjoint_set_get_members(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 2).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("getSetMembers", &args[1..])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("First argument must be a DisjointSet".to_string()))
    }
}

/// Get all disjoint sets
pub fn disjoint_set_get_all_sets(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("getAllSets", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a DisjointSet".to_string()))
    }
}

/// Get all root representatives
pub fn disjoint_set_get_roots(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("getRoots", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a DisjointSet".to_string()))
    }
}

/// Reset the DisjointSet to initial state
pub fn disjoint_set_reset(args: &[Value]) -> VmResult<Value> {
    validate_args(args, 1).map_err(|e| VmError::Runtime(e.to_string()))?;
    
    if let Value::LyObj(ds_obj) = &args[0] {
        ds_obj.call_method("reset", &[])
            .map_err(|e| VmError::Runtime(e.to_string()))
    } else {
        Err(VmError::Runtime("Argument must be a DisjointSet".to_string()))
    }
}

/// Register DisjointSet functions with the standard library
pub fn register_disjoint_set_functions(functions: &mut HashMap<String, fn(&[Value]) -> VmResult<Value>>) {
    functions.insert("DisjointSet".to_string(), disjoint_set_new);
    functions.insert("DSFind".to_string(), disjoint_set_find);
    functions.insert("DSUnion".to_string(), disjoint_set_union);
    functions.insert("DSConnected".to_string(), disjoint_set_connected);
    functions.insert("DSSetSize".to_string(), disjoint_set_set_size);
    functions.insert("DSCount".to_string(), disjoint_set_count);
    functions.insert("DSTotalElements".to_string(), disjoint_set_total_elements);
    functions.insert("DSGetMembers".to_string(), disjoint_set_get_members);
    functions.insert("DSGetAllSets".to_string(), disjoint_set_get_all_sets);
    functions.insert("DSGetRoots".to_string(), disjoint_set_get_roots);
    functions.insert("DSReset".to_string(), disjoint_set_reset);
}

/// Get documentation for DisjointSet functions
pub fn get_disjoint_set_documentation() -> HashMap<String, String> {
    let mut docs = HashMap::new();
    docs.insert("DisjointSet".to_string(), "DisjointSet[n] - Create disjoint set with n elements (0 to n-1). O(n) time.".to_string());
    docs.insert("DSFind".to_string(), "DSFind[ds, x] - Find root of element x with path compression. O(α(n)) amortized.".to_string());
    docs.insert("DSUnion".to_string(), "DSUnion[ds, x, y] - Union sets containing x and y. Returns {newDS, wasUnited}. O(α(n)) amortized.".to_string());
    docs.insert("DSConnected".to_string(), "DSConnected[ds, x, y] - Check if x and y are in same set. O(α(n)) amortized.".to_string());
    docs.insert("DSSetSize".to_string(), "DSSetSize[ds, x] - Get size of set containing element x. O(α(n)) amortized.".to_string());
    docs.insert("DSCount".to_string(), "DSCount[ds] - Get number of disjoint sets. O(1) time.".to_string());
    docs.insert("DSTotalElements".to_string(), "DSTotalElements[ds] - Get total number of elements. O(1) time.".to_string());
    docs.insert("DSGetMembers".to_string(), "DSGetMembers[ds, x] - Get all elements in same set as x. O(n) time.".to_string());
    docs.insert("DSGetAllSets".to_string(), "DSGetAllSets[ds] - Get all disjoint sets as list of lists. O(n) time.".to_string());
    docs.insert("DSGetRoots".to_string(), "DSGetRoots[ds] - Get root representatives of all sets. O(n) time.".to_string());
    docs.insert("DSReset".to_string(), "DSReset[ds] - Reset to initial state (all elements disjoint). O(n) time.".to_string());
    docs
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_disjoint_set_creation() {
        let ds = DisjointSet::new(5).unwrap();
        assert_eq!(ds.total_elements(), 5);
        assert_eq!(ds.count_sets(), 5); // All elements initially disjoint
    }
    
    #[test]
    fn test_disjoint_set_union_find() {
        let mut ds = DisjointSet::new(5).unwrap();
        
        // Initially, all elements should be their own root
        assert_eq!(ds.find(0).unwrap(), 0);
        assert_eq!(ds.find(1).unwrap(), 1);
        assert_eq!(ds.find(2).unwrap(), 2);
        
        // Union 0 and 1
        assert!(ds.union(0, 1).unwrap()); // Should return true (union occurred)
        assert_eq!(ds.count_sets(), 4); // One fewer disjoint set
        
        // 0 and 1 should now have the same root
        assert_eq!(ds.find(0).unwrap(), ds.find(1).unwrap());
        assert!(ds.connected(0, 1).unwrap());
        
        // Union 2 and 3
        assert!(ds.union(2, 3).unwrap());
        assert_eq!(ds.count_sets(), 3);
        assert!(ds.connected(2, 3).unwrap());
        
        // 0,1 should not be connected to 2,3 yet
        assert!(!ds.connected(0, 2).unwrap());
        
        // Union the two groups
        assert!(ds.union(1, 3).unwrap());
        assert_eq!(ds.count_sets(), 2); // Now we have 2 sets: {0,1,2,3} and {4}
        
        // All of 0,1,2,3 should be connected
        assert!(ds.connected(0, 2).unwrap());
        assert!(ds.connected(1, 3).unwrap());
        assert!(ds.connected(0, 3).unwrap());
        
        // But 4 should still be separate
        assert!(!ds.connected(0, 4).unwrap());
    }
    
    #[test]
    fn test_disjoint_set_sizes() {
        let mut ds = DisjointSet::new(5).unwrap();
        
        // Initially all sets have size 1
        assert_eq!(ds.set_size(0).unwrap(), 1);
        assert_eq!(ds.set_size(1).unwrap(), 1);
        
        // Union 0 and 1
        ds.union(0, 1).unwrap();
        assert_eq!(ds.set_size(0).unwrap(), 2);
        assert_eq!(ds.set_size(1).unwrap(), 2); // Same set
        
        // Union 2, 3, 4
        ds.union(2, 3).unwrap();
        ds.union(3, 4).unwrap();
        assert_eq!(ds.set_size(2).unwrap(), 3);
        assert_eq!(ds.set_size(3).unwrap(), 3);
        assert_eq!(ds.set_size(4).unwrap(), 3);
        
        // Set containing 0,1 should still have size 2
        assert_eq!(ds.set_size(0).unwrap(), 2);
    }
    
    #[test]
    fn test_get_set_members() {
        let mut ds = DisjointSet::new(5).unwrap();
        
        // Initially each element is in its own set
        let members = ds.get_set_members(0).unwrap();
        assert_eq!(members, vec![0]);
        
        // Union some elements
        ds.union(0, 1).unwrap();
        ds.union(1, 2).unwrap();
        
        let mut members = ds.get_set_members(0).unwrap();
        members.sort(); // Sort for consistent testing
        assert_eq!(members, vec![0, 1, 2]);
        
        let mut members1 = ds.get_set_members(1).unwrap();
        members1.sort();
        assert_eq!(members1, vec![0, 1, 2]); // Same set
    }
    
    #[test]
    fn test_get_all_sets() {
        let mut ds = DisjointSet::new(4).unwrap();
        
        // Union to create sets {0,1} and {2,3}
        ds.union(0, 1).unwrap();
        ds.union(2, 3).unwrap();
        
        let mut all_sets = ds.get_all_sets().unwrap();
        
        // Sort sets and elements within sets for consistent testing
        for set in &mut all_sets {
            set.sort();
        }
        all_sets.sort_by_key(|set| set[0]); // Sort by first element
        
        assert_eq!(all_sets.len(), 2);
        assert_eq!(all_sets[0], vec![0, 1]);
        assert_eq!(all_sets[1], vec![2, 3]);
    }
    
    #[test]
    fn test_get_roots() {
        let mut ds = DisjointSet::new(5).unwrap();
        
        // Initially all elements are roots
        let mut roots = ds.get_roots().unwrap();
        roots.sort();
        assert_eq!(roots, vec![0, 1, 2, 3, 4]);
        
        // After some unions
        ds.union(0, 1).unwrap();
        ds.union(2, 3).unwrap();
        
        let roots = ds.get_roots().unwrap();
        assert_eq!(roots.len(), 3); // 3 disjoint sets remaining
    }
    
    #[test]
    fn test_reset() {
        let mut ds = DisjointSet::new(5).unwrap();
        
        // Make some unions
        ds.union(0, 1).unwrap();
        ds.union(2, 3).unwrap();
        ds.union(3, 4).unwrap();
        
        assert_eq!(ds.count_sets(), 2);
        assert!(ds.connected(0, 1).unwrap());
        
        // Reset
        ds.reset();
        
        assert_eq!(ds.count_sets(), 5);
        assert_eq!(ds.total_elements(), 5);
        assert!(!ds.connected(0, 1).unwrap());
        
        // All elements should be their own root again
        for i in 0..5 {
            assert_eq!(ds.find(i).unwrap(), i);
            assert_eq!(ds.set_size(i).unwrap(), 1);
        }
    }
    
    #[test]
    fn test_error_handling() {
        // Test creating with 0 elements
        assert!(DisjointSet::new(0).is_err());
        
        let mut ds = DisjointSet::new(3).unwrap();
        
        // Test out of bounds access
        assert!(ds.find(5).is_err());
        assert!(ds.union(0, 5).is_err());
        assert!(ds.union(5, 0).is_err());
        assert!(ds.connected(0, 5).is_err());
        assert!(ds.set_size(5).is_err());
        assert!(ds.get_set_members(5).is_err());
    }
    
    #[test]
    fn test_union_same_set() {
        let mut ds = DisjointSet::new(3).unwrap();
        
        // Union elements that are already in the same set
        ds.union(0, 1).unwrap();
        assert_eq!(ds.count_sets(), 2);
        
        // Try to union them again - should return false and not change count
        assert!(!ds.union(0, 1).unwrap());
        assert_eq!(ds.count_sets(), 2);
    }
}