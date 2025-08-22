//! Phylogenetic Analysis Module
//!
//! This module implements phylogenetic tree construction and analysis algorithms:
//! - Neighbor-joining method for building phylogenetic trees
//! - Maximum likelihood estimation
//! - Pairwise distance calculations
//! - Tree manipulation and analysis functions

use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::collections::HashMap;

/// Foreign object representing a phylogenetic tree
#[derive(Debug, Clone, PartialEq)]
pub struct PhylogeneticTree {
    pub newick: String,
    pub leaves: Vec<String>,
    pub internal_nodes: Vec<String>,
    pub branch_lengths: HashMap<String, f64>,
    pub distance_matrix: Vec<Vec<f64>>,
    pub is_rooted: bool,
    pub likelihood: Option<f64>,
}

impl Foreign for PhylogeneticTree {
    fn type_name(&self) -> &'static str {
        "PhylogeneticTree"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "newick" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.newick.clone()))
            }
            "leaves" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let leaf_values: Vec<Value> = self.leaves
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(leaf_values))
            }
            "numLeaves" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.leaves.len() as i64))
            }
            "isRooted" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_rooted))
            }
            "branchLengths" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let branch_list: Vec<Value> = self.branch_lengths
                    .values()
                    .map(|&length| Value::Real(length))
                    .collect();
                Ok(Value::List(branch_list))
            }
            "distanceMatrix" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                // Return the distance matrix as a Foreign object
                let distance_matrix_obj = DistanceMatrix {
                    matrix: self.distance_matrix.clone(),
                    taxa: self.leaves.clone(),
                };
                Ok(Value::LyObj(LyObj::new(Box::new(distance_matrix_obj))))
            }
            "likelihood" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.likelihood {
                    Some(likelihood) => Ok(Value::Real(likelihood)),
                    None => Ok(Value::Missing),
                }
            }
            "internalNodes" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let node_values: Vec<Value> = self.internal_nodes
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(node_values))
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

/// Foreign object representing a distance matrix
#[derive(Debug, Clone, PartialEq)]
pub struct DistanceMatrix {
    pub matrix: Vec<Vec<f64>>,
    pub taxa: Vec<String>,
}

impl Foreign for DistanceMatrix {
    fn type_name(&self) -> &'static str {
        "DistanceMatrix"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getDistance" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let i = match &args[0] {
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let j = match &args[1] {
                    Value::Integer(idx) => *idx as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                if i >= self.matrix.len() || j >= self.matrix[0].len() {
                    return Err(ForeignError::IndexOutOfBounds {
                        index: format!("({}, {})", i, j),
                        bounds: format!("0..{}", self.matrix.len()),
                    });
                }
                
                Ok(Value::Real(self.matrix[i][j]))
            }
            "taxa" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let taxa_values: Vec<Value> = self.taxa
                    .iter()
                    .map(|s| Value::String(s.clone()))
                    .collect();
                Ok(Value::List(taxa_values))
            }
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.matrix.len() as i64))
            }
            "matrix" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let matrix_rows: Vec<Value> = self.matrix
                    .iter()
                    .map(|row| {
                        let row_values: Vec<Value> = row.iter().map(|&val| Value::Real(val)).collect();
                        Value::List(row_values)
                    })
                    .collect();
                Ok(Value::List(matrix_rows))
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

/// Node structure for phylogenetic tree construction
#[derive(Debug, Clone)]
struct TreeNode {
    name: String,
    children: Vec<TreeNode>,
    branch_length: f64,
    is_leaf: bool,
}

impl TreeNode {
    fn new_leaf(name: String) -> Self {
        TreeNode {
            name,
            children: Vec::new(),
            branch_length: 0.0,
            is_leaf: true,
        }
    }
    
    fn new_internal(name: String, children: Vec<TreeNode>, branch_length: f64) -> Self {
        TreeNode {
            name,
            children,
            branch_length,
            is_leaf: false,
        }
    }
    
    fn to_newick(&self) -> String {
        if self.is_leaf {
            format!("{}:{:.6}", self.name, self.branch_length)
        } else {
            let child_strings: Vec<String> = self.children
                .iter()
                .map(|child| child.to_newick())
                .collect();
            
            if self.name.is_empty() {
                format!("({}):{:.6}", child_strings.join(","), self.branch_length)
            } else {
                format!("({}){}:{:.6}", child_strings.join(","), self.name, self.branch_length)
            }
        }
    }
}

/// Calculate pairwise distances between sequences using simple Hamming distance
fn calculate_distance_matrix(sequences: &[String]) -> Vec<Vec<f64>> {
    let n = sequences.len();
    let mut matrix = vec![vec![0.0; n]; n];
    
    for i in 0..n {
        for j in i..n {
            if i == j {
                matrix[i][j] = 0.0;
            } else {
                let distance = hamming_distance(&sequences[i], &sequences[j]);
                matrix[i][j] = distance;
                matrix[j][i] = distance;
            }
        }
    }
    
    matrix
}

/// Calculate Hamming distance between two sequences (proportion of differing positions)
fn hamming_distance(seq1: &str, seq2: &str) -> f64 {
    let chars1: Vec<char> = seq1.chars().collect();
    let chars2: Vec<char> = seq2.chars().collect();
    
    let max_len = chars1.len().max(chars2.len());
    if max_len == 0 {
        return 0.0;
    }
    
    let mut differences = 0;
    
    for i in 0..max_len {
        let c1 = chars1.get(i).unwrap_or(&'-');
        let c2 = chars2.get(i).unwrap_or(&'-');
        
        if c1 != c2 {
            differences += 1;
        }
    }
    
    differences as f64 / max_len as f64
}

/// Neighbor-joining algorithm implementation
fn neighbor_joining_impl(sequences: &[String]) -> PhylogeneticTree {
    let n = sequences.len();
    let mut distance_matrix = calculate_distance_matrix(sequences);
    
    // Create initial leaf nodes
    let mut nodes: Vec<TreeNode> = sequences
        .iter()
        .enumerate()
        .map(|(i, seq)| TreeNode::new_leaf(format!("seq_{}", i)))
        .collect();
    
    let mut taxa: Vec<String> = sequences
        .iter()
        .enumerate()
        .map(|(i, _)| format!("seq_{}", i))
        .collect();
    
    let mut next_internal_id = n;
    
    // NJ algorithm main loop
    while nodes.len() > 2 {
        let current_n = nodes.len();
        
        // Calculate Q matrix
        let mut q_matrix = vec![vec![0.0; current_n]; current_n];
        let mut r_values = vec![0.0; current_n];
        
        // Calculate r values (sum of distances for each node)
        for i in 0..current_n {
            r_values[i] = distance_matrix[i].iter().sum::<f64>();
        }
        
        // Calculate Q values
        for i in 0..current_n {
            for j in i + 1..current_n {
                q_matrix[i][j] = (current_n - 2) as f64 * distance_matrix[i][j] - r_values[i] - r_values[j];
                q_matrix[j][i] = q_matrix[i][j];
            }
        }
        
        // Find minimum Q value
        let mut min_q = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        
        for i in 0..current_n {
            for j in i + 1..current_n {
                if q_matrix[i][j] < min_q {
                    min_q = q_matrix[i][j];
                    min_i = i;
                    min_j = j;
                }
            }
        }
        
        // Calculate branch lengths
        let branch_length_i = 0.5 * distance_matrix[min_i][min_j] + 
            (r_values[min_i] - r_values[min_j]) / (2.0 * (current_n - 2) as f64);
        let branch_length_j = distance_matrix[min_i][min_j] - branch_length_i;
        
        // Set branch lengths for nodes being joined
        nodes[min_i].branch_length = branch_length_i.max(0.0);
        nodes[min_j].branch_length = branch_length_j.max(0.0);
        
        // Create new internal node
        let new_node_name = format!("node_{}", next_internal_id);
        let new_node = TreeNode::new_internal(
            new_node_name.clone(),
            vec![nodes[min_i].clone(), nodes[min_j].clone()],
            0.0,
        );
        
        // Calculate distances from new node to all other nodes
        let mut new_distances = vec![0.0; current_n];
        for k in 0..current_n {
            if k != min_i && k != min_j {
                new_distances[k] = 0.5 * (distance_matrix[min_i][k] + distance_matrix[min_j][k] - distance_matrix[min_i][min_j]);
            }
        }
        
        // Update distance matrix and nodes
        let mut new_matrix = vec![vec![0.0; current_n - 1]; current_n - 1];
        let mut new_nodes = Vec::new();
        let mut new_row_idx = 0;
        
        // Add the new internal node first
        new_nodes.push(new_node);
        
        // Add remaining nodes (excluding the two that were joined)
        for (old_idx, node) in nodes.iter().enumerate() {
            if old_idx != min_i && old_idx != min_j {
                new_nodes.push(node.clone());
            }
        }
        
        // Fill the new distance matrix
        for i in 0..new_nodes.len() {
            for j in 0..new_nodes.len() {
                if i == 0 { // New internal node
                    if j == 0 {
                        new_matrix[i][j] = 0.0;
                    } else {
                        let old_j = if j - 1 < min_i { j - 1 } else if j - 1 < min_j - 1 { j } else { j + 1 };
                        new_matrix[i][j] = new_distances[old_j];
                        new_matrix[j][i] = new_distances[old_j];
                    }
                } else if j > i && i > 0 && j > 0 { // Both are old nodes
                    let old_i = if i - 1 < min_i { i - 1 } else if i - 1 < min_j - 1 { i } else { i + 1 };
                    let old_j = if j - 1 < min_i { j - 1 } else if j - 1 < min_j - 1 { j } else { j + 1 };
                    new_matrix[i][j] = distance_matrix[old_i][old_j];
                    new_matrix[j][i] = distance_matrix[old_i][old_j];
                }
            }
        }
        
        nodes = new_nodes;
        distance_matrix = new_matrix;
        next_internal_id += 1;
    }
    
    // Create final root node if there are exactly 2 nodes left
    let root_node = if nodes.len() == 2 {
        let final_distance = if !distance_matrix.is_empty() && distance_matrix[0].len() > 1 {
            distance_matrix[0][1]
        } else {
            0.1
        };
        
        nodes[0].branch_length = final_distance / 2.0;
        nodes[1].branch_length = final_distance / 2.0;
        
        TreeNode::new_internal(
            String::new(), // Root has no name
            nodes,
            0.0,
        )
    } else {
        nodes.into_iter().next().unwrap_or(TreeNode::new_leaf("empty".to_string()))
    };
    
    // Generate newick string
    let newick = format!("{};", root_node.to_newick());
    
    // Extract leaves and branch lengths
    let mut leaves = Vec::new();
    let mut branch_lengths = HashMap::new();
    collect_tree_info(&root_node, &mut leaves, &mut branch_lengths);
    
    PhylogeneticTree {
        newick,
        leaves: taxa,
        internal_nodes: vec![], // Simplified for now
        branch_lengths,
        distance_matrix: calculate_distance_matrix(sequences),
        is_rooted: true,
        likelihood: None,
    }
}

/// Recursively collect tree information (leaves and branch lengths)
fn collect_tree_info(
    node: &TreeNode,
    leaves: &mut Vec<String>,
    branch_lengths: &mut HashMap<String, f64>,
) {
    if node.is_leaf {
        leaves.push(node.name.clone());
    }
    
    if !node.name.is_empty() {
        branch_lengths.insert(node.name.clone(), node.branch_length);
    }
    
    for child in &node.children {
        collect_tree_info(child, leaves, branch_lengths);
    }
}

/// Simple maximum likelihood estimation (placeholder implementation)
fn maximum_likelihood_estimation(sequences: &[String]) -> PhylogeneticTree {
    // Start with NJ tree as initial topology
    let mut tree = neighbor_joining_impl(sequences);
    
    // Simplified ML estimation - optimize branch lengths
    let likelihood = calculate_likelihood(sequences, &tree.distance_matrix);
    tree.likelihood = Some(likelihood);
    
    tree
}

/// Calculate likelihood for a given tree and sequences (simplified)
fn calculate_likelihood(sequences: &[String], distance_matrix: &[Vec<f64>]) -> f64 {
    let n = sequences.len();
    let mut log_likelihood = 0.0;
    
    // Simplified likelihood calculation based on distances
    for i in 0..n {
        for j in i + 1..n {
            let observed_distance = hamming_distance(&sequences[i], &sequences[j]);
            let expected_distance = distance_matrix[i][j];
            
            // Simple Poisson model for substitutions
            let rate = expected_distance.max(1e-10);
            log_likelihood += observed_distance * rate.ln() - rate - (1..=(observed_distance as usize)).map(|k| (k as f64).ln()).sum::<f64>();
        }
    }
    
    log_likelihood
}

/// Parse a Newick format string into a PhylogeneticTree
fn parse_newick(newick: &str) -> Result<PhylogeneticTree, String> {
    let clean_newick = newick.trim().trim_end_matches(';');
    
    if clean_newick.is_empty() {
        return Err("Empty newick string".to_string());
    }
    
    // Simple newick parser - extract leaf names
    let mut leaves = Vec::new();
    let mut current_name = String::new();
    let mut in_name = false;
    let mut in_branch_length = false;
    
    for ch in clean_newick.chars() {
        match ch {
            '(' | ')' | ',' => {
                if !current_name.is_empty() && in_name {
                    leaves.push(current_name.clone());
                    current_name.clear();
                }
                in_name = false;
                in_branch_length = false;
            }
            ':' => {
                if !current_name.is_empty() && in_name {
                    leaves.push(current_name.clone());
                    current_name.clear();
                }
                in_name = false;
                in_branch_length = true;
            }
            ' ' | '\t' | '\n' => {
                // Skip whitespace
            }
            _ => {
                if !in_branch_length && (ch.is_alphanumeric() || ch == '_') {
                    current_name.push(ch);
                    in_name = true;
                }
            }
        }
    }
    
    // Handle final name
    if !current_name.is_empty() && in_name {
        leaves.push(current_name);
    }
    
    Ok(PhylogeneticTree {
        newick: format!("{};", clean_newick),
        leaves: leaves.clone(),
        internal_nodes: vec![], // Simplified
        branch_lengths: HashMap::new(), // Would need more complex parsing
        distance_matrix: vec![], // Empty for parsed trees
        is_rooted: !clean_newick.starts_with('('),
        likelihood: None,
    })
}

// Public API functions for the Lyra stdlib

/// NeighborJoining[sequences] -> PhylogeneticTree
pub fn neighbor_joining(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequences = crate::stdlib::bioinformatics::validate_sequence_list(&args[0])?;
    
    if sequences.len() < 3 {
        return Err(VmError::Runtime("Need at least 3 sequences for neighbor-joining".to_string()));
    }
    
    for seq in &sequences {
        if seq.is_empty() {
            return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
        }
    }
    
    let tree = neighbor_joining_impl(&sequences);
    Ok(Value::LyObj(LyObj::new(Box::new(tree))))
}

/// MaximumLikelihood[sequences] -> PhylogeneticTree
pub fn maximum_likelihood(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let sequences = crate::stdlib::bioinformatics::validate_sequence_list(&args[0])?;
    
    if sequences.len() < 3 {
        return Err(VmError::Runtime("Need at least 3 sequences for maximum likelihood".to_string()));
    }
    
    for seq in &sequences {
        if seq.is_empty() {
            return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
        }
    }
    
    let tree = maximum_likelihood_estimation(&sequences);
    Ok(Value::LyObj(LyObj::new(Box::new(tree))))
}

/// PairwiseDistance[seq1, seq2] -> Real
pub fn pairwise_distance(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 2, got {}", args.len())));
    }
    
    let seq1 = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    let seq2 = crate::stdlib::bioinformatics::validate_sequence_string(&args[1])?;
    
    if seq1.is_empty() || seq2.is_empty() {
        return Err(VmError::Runtime("Sequences cannot be empty".to_string()));
    }
    
    let distance = hamming_distance(&seq1, &seq2);
    Ok(Value::Real(distance))
}

/// PhylogeneticTree[newick_string] -> PhylogeneticTree
pub fn phylogenetic_tree(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Invalid number of arguments: expected 1, got {}", args.len())));
    }
    
    let newick = crate::stdlib::bioinformatics::validate_sequence_string(&args[0])?;
    
    if newick.is_empty() {
        return Err(VmError::Runtime("Newick string cannot be empty".to_string()));
    }
    
    match parse_newick(&newick) {
        Ok(tree) => Ok(Value::LyObj(LyObj::new(Box::new(tree)))),
        Err(error) => Err(VmError::Runtime(
            format!("Error parsing Newick string: {}", error))),
    }
}