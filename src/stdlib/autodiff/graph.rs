//! Computation Graph for Reverse-Mode Automatic Differentiation
//!
//! This module implements a computation graph that tracks operations and enables
//! efficient gradient computation via backpropagation. The graph supports
//! dynamic construction during forward computation and reverse-mode gradient
//! computation with optimal memory usage.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use super::{AutodiffError, AutodiffResult, constants::MAX_GRAPH_DEPTH};

/// Unique identifier for computation graph nodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    /// Create a new unique node ID
    pub fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        NodeId(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
    
    /// Get the internal ID value
    pub fn id(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.0)
    }
}

/// Mathematical operations supported in the computation graph
#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    /// Input variable or constant
    Input { name: Option<String> },
    /// Addition: output = input1 + input2
    Add,
    /// Subtraction: output = input1 - input2
    Sub,
    /// Multiplication: output = input1 * input2
    Mul,
    /// Division: output = input1 / input2
    Div,
    /// Power: output = input1 ^ input2
    Pow,
    /// Negation: output = -input1
    Neg,
    /// Natural exponential: output = exp(input1)
    Exp,
    /// Natural logarithm: output = ln(input1)
    Ln,
    /// Square root: output = sqrt(input1)
    Sqrt,
    /// Sine: output = sin(input1)
    Sin,
    /// Cosine: output = cos(input1)
    Cos,
    /// Tangent: output = tan(input1)
    Tan,
    /// Hyperbolic sine: output = sinh(input1)
    Sinh,
    /// Hyperbolic cosine: output = cosh(input1)
    Cosh,
    /// Hyperbolic tangent: output = tanh(input1)
    Tanh,
    /// Absolute value: output = |input1|
    Abs,
    /// Minimum: output = min(input1, input2)
    Min,
    /// Maximum: output = max(input1, input2)
    Max,
    /// ReLU activation: output = max(0, input1)
    ReLU,
    /// Sigmoid activation: output = 1 / (1 + exp(-input1))
    Sigmoid,
    /// Sum reduction: output = sum(input1)
    Sum,
    /// Mean reduction: output = mean(input1)
    Mean,
}

impl Operation {
    /// Get the number of inputs required for this operation
    pub fn input_count(&self) -> usize {
        match self {
            Operation::Input { .. } => 0,
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div 
            | Operation::Pow | Operation::Min | Operation::Max => 2,
            _ => 1,
        }
    }
    
    /// Check if this operation is commutative
    pub fn is_commutative(&self) -> bool {
        matches!(self, Operation::Add | Operation::Mul | Operation::Min | Operation::Max)
    }
    
    /// Get a human-readable name for this operation
    pub fn name(&self) -> &'static str {
        match self {
            Operation::Input { .. } => "Input",
            Operation::Add => "Add",
            Operation::Sub => "Sub",
            Operation::Mul => "Mul",
            Operation::Div => "Div",
            Operation::Pow => "Pow",
            Operation::Neg => "Neg",
            Operation::Exp => "Exp",
            Operation::Ln => "Ln",
            Operation::Sqrt => "Sqrt",
            Operation::Sin => "Sin",
            Operation::Cos => "Cos",
            Operation::Tan => "Tan",
            Operation::Sinh => "Sinh",
            Operation::Cosh => "Cosh",
            Operation::Tanh => "Tanh",
            Operation::Abs => "Abs",
            Operation::Min => "Min",
            Operation::Max => "Max",
            Operation::ReLU => "ReLU",
            Operation::Sigmoid => "Sigmoid",
            Operation::Sum => "Sum",
            Operation::Mean => "Mean",
        }
    }
}

/// A node in the computation graph
#[derive(Debug, Clone)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: NodeId,
    /// The operation performed at this node
    pub operation: Operation,
    /// Input nodes that feed into this operation
    pub inputs: Vec<NodeId>,
    /// The computed value at this node
    pub value: f64,
    /// The gradient accumulated at this node during backpropagation
    pub gradient: f64,
    /// Whether this node requires gradient computation
    pub requires_grad: bool,
    /// Cached partial derivatives for efficiency
    pub partial_derivatives: Vec<f64>,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(
        operation: Operation,
        inputs: Vec<NodeId>,
        value: f64,
        requires_grad: bool,
    ) -> Self {
        let input_count = inputs.len();
        Self {
            id: NodeId::new(),
            operation,
            inputs,
            value,
            gradient: 0.0,
            requires_grad,
            partial_derivatives: vec![0.0; input_count],
        }
    }
    
    /// Create an input node (variable or constant)
    pub fn input(name: Option<String>, value: f64, requires_grad: bool) -> Self {
        Self::new(
            Operation::Input { name },
            vec![],
            value,
            requires_grad,
        )
    }
    
    /// Reset gradient to zero
    pub fn zero_grad(&mut self) {
        self.gradient = 0.0;
        self.partial_derivatives.fill(0.0);
    }
    
    /// Accumulate gradient from backpropagation
    pub fn accumulate_gradient(&mut self, grad: f64) {
        self.gradient += grad;
    }
    
    /// Check if this node is a leaf (input) node
    pub fn is_leaf(&self) -> bool {
        matches!(self.operation, Operation::Input { .. })
    }
    
    /// Get the name of an input node
    pub fn input_name(&self) -> Option<&str> {
        match &self.operation {
            Operation::Input { name } => name.as_deref(),
            _ => None,
        }
    }
}

/// Computation graph for reverse-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// All nodes in the graph, indexed by their ID
    nodes: HashMap<NodeId, GraphNode>,
    /// Topological ordering of nodes for efficient computation
    topological_order: Vec<NodeId>,
    /// Whether the graph needs re-sorting
    needs_sort: bool,
    /// Maximum depth to prevent infinite recursion
    max_depth: usize,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            topological_order: Vec::new(),
            needs_sort: false,
            max_depth: MAX_GRAPH_DEPTH,
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, mut node: GraphNode) -> NodeId {
        let node_id = node.id;
        
        // Validate input dependencies exist
        for input_id in &node.inputs {
            if !self.nodes.contains_key(input_id) {
                panic!("Input node {:?} not found in graph", input_id);
            }
        }
        
        // Compute partial derivatives for this node
        self.compute_partial_derivatives(&mut node);
        
        self.nodes.insert(node_id, node);
        self.needs_sort = true;
        
        node_id
    }
    
    /// Create and add an input node
    pub fn add_input(&mut self, name: Option<String>, value: f64, requires_grad: bool) -> NodeId {
        let node = GraphNode::input(name, value, requires_grad);
        self.add_node(node)
    }
    
    /// Create and add a binary operation node
    pub fn add_binary_op(
        &mut self,
        operation: Operation,
        left: NodeId,
        right: NodeId,
        requires_grad: bool,
    ) -> AutodiffResult<NodeId> {
        let left_node = self.get_node(left)?;
        let right_node = self.get_node(right)?;
        
        // Use NaN as placeholder - actual value computed during forward pass
        let value = f64::NAN;
        
        let node = GraphNode::new(
            operation,
            vec![left, right],
            value,
            requires_grad || left_node.requires_grad || right_node.requires_grad,
        );
        
        Ok(self.add_node(node))
    }
    
    /// Create and add a unary operation node
    pub fn add_unary_op(
        &mut self,
        operation: Operation,
        input: NodeId,
        requires_grad: bool,
    ) -> AutodiffResult<NodeId> {
        let input_node = self.get_node(input)?;
        
        // Use NaN as placeholder - actual value computed during forward pass
        let value = f64::NAN;
        
        let node = GraphNode::new(
            operation,
            vec![input],
            value,
            requires_grad || input_node.requires_grad,
        );
        
        Ok(self.add_node(node))
    }
    
    /// Get a node by ID
    pub fn get_node(&self, id: NodeId) -> AutodiffResult<&GraphNode> {
        self.nodes.get(&id).ok_or_else(|| AutodiffError::GraphError {
            reason: format!("Node {:?} not found", id),
        })
    }
    
    /// Get a mutable node by ID
    pub fn get_node_mut(&mut self, id: NodeId) -> AutodiffResult<&mut GraphNode> {
        self.nodes.get_mut(&id).ok_or_else(|| AutodiffError::GraphError {
            reason: format!("Node {:?} not found", id),
        })
    }
    
    /// Get all nodes
    pub fn nodes(&self) -> &HashMap<NodeId, GraphNode> {
        &self.nodes
    }
    
    /// Clear all gradients in the graph
    pub fn zero_grad(&mut self) {
        for node in self.nodes.values_mut() {
            node.zero_grad();
        }
    }
    
    /// Perform forward pass to compute all values
    pub fn forward(&mut self) -> AutodiffResult<()> {
        self.ensure_topological_sort()?;
        
        for &node_id in &self.topological_order {
            let node = self.nodes.get(&node_id).unwrap();
            
            if !node.is_leaf() {
                let inputs: Result<Vec<f64>, _> = node.inputs.iter()
                    .map(|&input_id| {
                        self.get_node(input_id).map(|n| n.value)
                    })
                    .collect();
                
                let input_values = inputs?;
                let new_value = match &node.operation {
                    Operation::Add => input_values[0] + input_values[1],
                    Operation::Sub => input_values[0] - input_values[1],
                    Operation::Mul => input_values[0] * input_values[1],
                    Operation::Div => {
                        if input_values[1] == 0.0 {
                            return Err(AutodiffError::GradientComputationFailed {
                                reason: "Division by zero".to_string(),
                            });
                        }
                        input_values[0] / input_values[1]
                    }
                    Operation::Pow => input_values[0].powf(input_values[1]),
                    Operation::Neg => -input_values[0],
                    Operation::Exp => input_values[0].exp(),
                    Operation::Ln => {
                        if input_values[0] <= 0.0 {
                            return Err(AutodiffError::GradientComputationFailed {
                                reason: "Logarithm of non-positive number".to_string(),
                            });
                        }
                        input_values[0].ln()
                    }
                    Operation::Sqrt => {
                        if input_values[0] < 0.0 {
                            return Err(AutodiffError::GradientComputationFailed {
                                reason: "Square root of negative number".to_string(),
                            });
                        }
                        input_values[0].sqrt()
                    }
                    Operation::Sin => input_values[0].sin(),
                    Operation::Cos => input_values[0].cos(),
                    Operation::Tan => input_values[0].tan(),
                    Operation::Sinh => input_values[0].sinh(),
                    Operation::Cosh => input_values[0].cosh(),
                    Operation::Tanh => input_values[0].tanh(),
                    Operation::Abs => input_values[0].abs(),
                    Operation::Min => input_values[0].min(input_values[1]),
                    Operation::Max => input_values[0].max(input_values[1]),
                    Operation::ReLU => input_values[0].max(0.0),
                    Operation::Sigmoid => 1.0 / (1.0 + (-input_values[0]).exp()),
                    Operation::Sum => input_values[0], // For single value
                    Operation::Mean => input_values[0], // For single value
                    Operation::Input { .. } => unreachable!("Leaf nodes should not be computed"),
                };
                
                {
                    let node = self.nodes.get_mut(&node_id).unwrap();
                    node.value = new_value;
                }
                // Compute partial derivatives after updating value
                let mut node = self.nodes.get_mut(&node_id).unwrap().clone();
                self.compute_partial_derivatives(&mut node);
                self.nodes.insert(node_id, node);
            }
        }
        
        Ok(())
    }
    
    /// Perform backward pass to compute gradients
    pub fn backward(&mut self, output_node: NodeId) -> AutodiffResult<()> {
        self.ensure_topological_sort()?;
        
        // Initialize output gradient to 1.0
        {
            let output = self.get_node_mut(output_node)?;
            output.gradient = 1.0;
        }
        
        // Get reverse topological order to avoid borrowing conflicts
        let reverse_order: Vec<NodeId> = self.topological_order.iter().rev().copied().collect();
        
        for &node_id in &reverse_order {
            let node = self.get_node(node_id)?.clone();
            
            if node.requires_grad && !node.is_leaf() {
                // Propagate gradients to input nodes
                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let grad_contribution = node.gradient * node.partial_derivatives[i];
                    
                    let input_node = self.get_node_mut(input_id)?;
                    input_node.accumulate_gradient(grad_contribution);
                }
            }
        }
        
        Ok(())
    }
    
    /// Get the gradient of a node
    pub fn get_gradient(&self, node_id: NodeId) -> AutodiffResult<f64> {
        Ok(self.get_node(node_id)?.gradient)
    }
    
    /// Get gradients for all nodes that require gradients
    pub fn get_gradients(&self) -> HashMap<NodeId, f64> {
        self.nodes.iter()
            .filter(|(_, node)| node.requires_grad)
            .map(|(&id, node)| (id, node.gradient))
            .collect()
    }
    
    /// Find nodes by name (for input nodes)
    pub fn find_nodes_by_name(&self, name: &str) -> Vec<NodeId> {
        self.nodes.iter()
            .filter_map(|(&id, node)| {
                if let Some(node_name) = node.input_name() {
                    if node_name == name {
                        Some(id)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    }
    
    /// Get statistics about the graph
    pub fn stats(&self) -> GraphStats {
        let total_nodes = self.nodes.len();
        let leaf_nodes = self.nodes.values().filter(|n| n.is_leaf()).count();
        let grad_nodes = self.nodes.values().filter(|n| n.requires_grad).count();
        
        GraphStats {
            total_nodes,
            leaf_nodes,
            internal_nodes: total_nodes - leaf_nodes,
            gradient_nodes: grad_nodes,
            max_depth: self.compute_max_depth(),
        }
    }
    
    /// Ensure the graph is topologically sorted
    fn ensure_topological_sort(&mut self) -> AutodiffResult<()> {
        if self.needs_sort {
            self.topological_sort()?;
            self.needs_sort = false;
        }
        Ok(())
    }
    
    /// Perform topological sort of the graph
    fn topological_sort(&mut self) -> AutodiffResult<()> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        
        // Initialize
        for (&node_id, node) in &self.nodes {
            in_degree.insert(node_id, node.inputs.len());
            adjacency.insert(node_id, Vec::new());
        }
        
        // Build adjacency list (reverse direction for topological sort)
        for (&node_id, node) in &self.nodes {
            for &input_id in &node.inputs {
                adjacency.get_mut(&input_id).unwrap().push(node_id);
            }
        }
        
        // Kahn's algorithm
        let mut queue: VecDeque<NodeId> = in_degree.iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(&id, _)| id)
            .collect();
        
        let mut result = Vec::new();
        
        while let Some(node_id) = queue.pop_front() {
            result.push(node_id);
            
            for &dependent_id in &adjacency[&node_id] {
                let degree = in_degree.get_mut(&dependent_id).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    queue.push_back(dependent_id);
                }
            }
        }
        
        if result.len() != self.nodes.len() {
            return Err(AutodiffError::GraphError {
                reason: "Cycle detected in computation graph".to_string(),
            });
        }
        
        self.topological_order = result;
        Ok(())
    }
    
    /// Compute partial derivatives for a node
    fn compute_partial_derivatives(&self, node: &mut GraphNode) {
        if node.is_leaf() {
            return;
        }
        
        let input_values: Vec<f64> = node.inputs.iter()
            .map(|&id| self.nodes.get(&id).unwrap().value)
            .collect();
        
        node.partial_derivatives.clear();
        
        match &node.operation {
            Operation::Add => {
                node.partial_derivatives = vec![1.0, 1.0];
            }
            Operation::Sub => {
                node.partial_derivatives = vec![1.0, -1.0];
            }
            Operation::Mul => {
                node.partial_derivatives = vec![input_values[1], input_values[0]];
            }
            Operation::Div => {
                let denom = input_values[1];
                node.partial_derivatives = vec![
                    1.0 / denom,
                    -input_values[0] / (denom * denom),
                ];
            }
            Operation::Pow => {
                let base = input_values[0];
                let exp = input_values[1];
                if base > 0.0 {
                    node.partial_derivatives = vec![
                        exp * base.powf(exp - 1.0),
                        base.powf(exp) * base.ln(),
                    ];
                } else {
                    node.partial_derivatives = vec![0.0, 0.0];
                }
            }
            Operation::Neg => {
                node.partial_derivatives = vec![-1.0];
            }
            Operation::Exp => {
                let exp_val = input_values[0].exp();
                node.partial_derivatives = vec![exp_val];
            }
            Operation::Ln => {
                node.partial_derivatives = vec![1.0 / input_values[0]];
            }
            Operation::Sqrt => {
                let sqrt_val = input_values[0].sqrt();
                node.partial_derivatives = vec![1.0 / (2.0 * sqrt_val)];
            }
            Operation::Sin => {
                node.partial_derivatives = vec![input_values[0].cos()];
            }
            Operation::Cos => {
                node.partial_derivatives = vec![-input_values[0].sin()];
            }
            Operation::Tan => {
                let cos_val = input_values[0].cos();
                node.partial_derivatives = vec![1.0 / (cos_val * cos_val)];
            }
            Operation::Sinh => {
                node.partial_derivatives = vec![input_values[0].cosh()];
            }
            Operation::Cosh => {
                node.partial_derivatives = vec![input_values[0].sinh()];
            }
            Operation::Tanh => {
                let cosh_val = input_values[0].cosh();
                node.partial_derivatives = vec![1.0 / (cosh_val * cosh_val)];
            }
            Operation::Abs => {
                let sign = if input_values[0] > 0.0 { 1.0 } else if input_values[0] < 0.0 { -1.0 } else { 0.0 };
                node.partial_derivatives = vec![sign];
            }
            Operation::Min => {
                if input_values[0] < input_values[1] {
                    node.partial_derivatives = vec![1.0, 0.0];
                } else if input_values[0] > input_values[1] {
                    node.partial_derivatives = vec![0.0, 1.0];
                } else {
                    node.partial_derivatives = vec![0.5, 0.5]; // Subgradient
                }
            }
            Operation::Max => {
                if input_values[0] > input_values[1] {
                    node.partial_derivatives = vec![1.0, 0.0];
                } else if input_values[0] < input_values[1] {
                    node.partial_derivatives = vec![0.0, 1.0];
                } else {
                    node.partial_derivatives = vec![0.5, 0.5]; // Subgradient
                }
            }
            Operation::ReLU => {
                let deriv = if input_values[0] > 0.0 { 1.0 } else { 0.0 };
                node.partial_derivatives = vec![deriv];
            }
            Operation::Sigmoid => {
                let sigmoid_val = 1.0 / (1.0 + (-input_values[0]).exp());
                let deriv = sigmoid_val * (1.0 - sigmoid_val);
                node.partial_derivatives = vec![deriv];
            }
            Operation::Sum => {
                node.partial_derivatives = vec![1.0];
            }
            Operation::Mean => {
                node.partial_derivatives = vec![1.0];
            }
            Operation::Input { .. } => {}
        }
    }
    
    /// Compute binary operation value
    fn compute_binary_value(&self, op: &Operation, left: f64, right: f64) -> AutodiffResult<f64> {
        match op {
            Operation::Add => Ok(left + right),
            Operation::Sub => Ok(left - right),
            Operation::Mul => Ok(left * right),
            Operation::Div => {
                if right == 0.0 {
                    Err(AutodiffError::GradientComputationFailed {
                        reason: "Division by zero".to_string(),
                    })
                } else {
                    Ok(left / right)
                }
            }
            Operation::Pow => Ok(left.powf(right)),
            Operation::Min => Ok(left.min(right)),
            Operation::Max => Ok(left.max(right)),
            _ => Err(AutodiffError::UnsupportedOperation {
                operation: op.name().to_string(),
                mode: "binary".to_string(),
            }),
        }
    }
    
    /// Compute unary operation value
    fn compute_unary_value(&self, op: &Operation, input: f64) -> AutodiffResult<f64> {
        match op {
            Operation::Neg => Ok(-input),
            Operation::Exp => Ok(input.exp()),
            Operation::Ln => {
                if input <= 0.0 {
                    Err(AutodiffError::GradientComputationFailed {
                        reason: "Logarithm of non-positive number".to_string(),
                    })
                } else {
                    Ok(input.ln())
                }
            }
            Operation::Sqrt => {
                if input < 0.0 {
                    Err(AutodiffError::GradientComputationFailed {
                        reason: "Square root of negative number".to_string(),
                    })
                } else {
                    Ok(input.sqrt())
                }
            }
            Operation::Sin => Ok(input.sin()),
            Operation::Cos => Ok(input.cos()),
            Operation::Tan => Ok(input.tan()),
            Operation::Sinh => Ok(input.sinh()),
            Operation::Cosh => Ok(input.cosh()),
            Operation::Tanh => Ok(input.tanh()),
            Operation::Abs => Ok(input.abs()),
            Operation::ReLU => Ok(input.max(0.0)),
            Operation::Sigmoid => Ok(1.0 / (1.0 + (-input).exp())),
            Operation::Sum => Ok(input),
            Operation::Mean => Ok(input),
            _ => Err(AutodiffError::UnsupportedOperation {
                operation: op.name().to_string(),
                mode: "unary".to_string(),
            }),
        }
    }
    
    /// Compute maximum depth of the graph
    fn compute_max_depth(&self) -> usize {
        let mut max_depth = 0;
        let mut visited = std::collections::HashSet::new();
        
        for &node_id in self.nodes.keys() {
            if !visited.contains(&node_id) {
                let depth = self.compute_depth_dfs(node_id, &mut visited, 0);
                max_depth = max_depth.max(depth);
            }
        }
        
        max_depth
    }
    
    /// Depth-first search to compute depth
    fn compute_depth_dfs(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        current_depth: usize,
    ) -> usize {
        if visited.contains(&node_id) {
            return current_depth;
        }
        
        visited.insert(node_id);
        
        let node = &self.nodes[&node_id];
        let mut max_child_depth = current_depth;
        
        for &input_id in &node.inputs {
            let child_depth = self.compute_depth_dfs(input_id, visited, current_depth + 1);
            max_child_depth = max_child_depth.max(child_depth);
        }
        
        max_child_depth
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a computation graph
#[derive(Debug, Clone)]
pub struct GraphStats {
    /// Total number of nodes
    pub total_nodes: usize,
    /// Number of leaf (input) nodes
    pub leaf_nodes: usize,
    /// Number of internal (operation) nodes
    pub internal_nodes: usize,
    /// Number of nodes requiring gradients
    pub gradient_nodes: usize,
    /// Maximum depth of the graph
    pub max_depth: usize,
}

impl std::fmt::Display for GraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Graph Stats: {} total nodes ({} leaf, {} internal), {} gradient nodes, max depth {}",
            self.total_nodes,
            self.leaf_nodes,
            self.internal_nodes,
            self.gradient_nodes,
            self.max_depth
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_node_creation() {
        let node = GraphNode::input(Some("x".to_string()), 3.0, true);
        assert_eq!(node.value, 3.0);
        assert!(node.requires_grad);
        assert!(node.is_leaf());
        assert_eq!(node.input_name(), Some("x"));
    }
    
    #[test]
    fn test_basic_graph_operations() {
        let mut graph = ComputationGraph::new();
        
        // Add input nodes: x = 2, y = 3
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let y = graph.add_input(Some("y".to_string()), 3.0, true);
        
        // Compute x + y
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        // Forward pass
        graph.forward().unwrap();
        
        let sum_node = graph.get_node(sum).unwrap();
        assert_eq!(sum_node.value, 5.0);
        
        // Backward pass
        graph.backward(sum).unwrap();
        
        // Check gradients
        assert_eq!(graph.get_gradient(x).unwrap(), 1.0);
        assert_eq!(graph.get_gradient(y).unwrap(), 1.0);
    }
    
    #[test]
    fn test_multiplication_chain() {
        let mut graph = ComputationGraph::new();
        
        // Create: f(x) = x * x * x (x^3)
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let x_squared = graph.add_binary_op(Operation::Mul, x, x, true).unwrap();
        let x_cubed = graph.add_binary_op(Operation::Mul, x_squared, x, true).unwrap();
        
        graph.forward().unwrap();
        
        let result = graph.get_node(x_cubed).unwrap();
        assert_eq!(result.value, 8.0); // 2^3 = 8
        
        graph.backward(x_cubed).unwrap();
        
        // d/dx(x^3) = 3x^2 = 3*4 = 12
        assert_eq!(graph.get_gradient(x).unwrap(), 12.0);
    }
    
    #[test]
    fn test_mixed_operations() {
        let mut graph = ComputationGraph::new();
        
        // Create: f(x, y) = sin(x * y) + exp(x)
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 0.5, true);
        
        let xy = graph.add_binary_op(Operation::Mul, x, y, true).unwrap();
        let sin_xy = graph.add_unary_op(Operation::Sin, xy, true).unwrap();
        let exp_x = graph.add_unary_op(Operation::Exp, x, true).unwrap();
        let result = graph.add_binary_op(Operation::Add, sin_xy, exp_x, true).unwrap();
        
        graph.forward().unwrap();
        
        let result_node = graph.get_node(result).unwrap();
        let expected = (1.0_f64 * 0.5).sin() + 1.0_f64.exp();
        assert!((result_node.value - expected).abs() < 1e-10);
        
        graph.backward(result).unwrap();
        
        // Check that gradients were computed (exact values depend on derivatives)
        let x_grad = graph.get_gradient(x).unwrap();
        let y_grad = graph.get_gradient(y).unwrap();
        
        // Gradients should be non-zero for this non-trivial function
        assert!(x_grad.abs() > 1e-10);
        assert!(y_grad.abs() > 1e-10);
    }
    
    #[test]
    fn test_relu_activation() {
        let mut graph = ComputationGraph::new();
        
        // Test ReLU with positive input
        let x_pos = graph.add_input(Some("x_pos".to_string()), 2.0, true);
        let relu_pos = graph.add_unary_op(Operation::ReLU, x_pos, true).unwrap();
        
        // Test ReLU with negative input
        let x_neg = graph.add_input(Some("x_neg".to_string()), -1.0, true);
        let relu_neg = graph.add_unary_op(Operation::ReLU, x_neg, true).unwrap();
        
        graph.forward().unwrap();
        
        assert_eq!(graph.get_node(relu_pos).unwrap().value, 2.0);
        assert_eq!(graph.get_node(relu_neg).unwrap().value, 0.0);
        
        graph.backward(relu_pos).unwrap();
        assert_eq!(graph.get_gradient(x_pos).unwrap(), 1.0);
        
        graph.zero_grad();
        graph.backward(relu_neg).unwrap();
        assert_eq!(graph.get_gradient(x_neg).unwrap(), 0.0);
    }
    
    #[test]
    fn test_graph_stats() {
        let mut graph = ComputationGraph::new();
        
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 2.0, false);
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        let stats = graph.stats();
        assert_eq!(stats.total_nodes, 3);
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.internal_nodes, 1);
        assert_eq!(stats.gradient_nodes, 2); // x and sum require gradients
    }
    
    #[test]
    fn test_find_nodes_by_name() {
        let mut graph = ComputationGraph::new();
        
        let x1 = graph.add_input(Some("x".to_string()), 1.0, true);
        let x2 = graph.add_input(Some("x".to_string()), 2.0, true);
        let y = graph.add_input(Some("y".to_string()), 3.0, true);
        
        let x_nodes = graph.find_nodes_by_name("x");
        assert_eq!(x_nodes.len(), 2);
        assert!(x_nodes.contains(&x1));
        assert!(x_nodes.contains(&x2));
        
        let y_nodes = graph.find_nodes_by_name("y");
        assert_eq!(y_nodes.len(), 1);
        assert!(y_nodes.contains(&y));
    }
    
    #[test]
    fn test_error_conditions() {
        let mut graph = ComputationGraph::new();
        
        let x = graph.add_input(Some("x".to_string()), -1.0, true);
        
        // Sqrt of negative should work during graph construction but fail in forward
        let sqrt_x = graph.add_unary_op(Operation::Sqrt, x, true).unwrap();
        assert!(graph.forward().is_err());
        
        // Division by zero
        let mut graph2 = ComputationGraph::new();
        let zero = graph2.add_input(Some("zero".to_string()), 0.0, true);
        let one = graph2.add_input(Some("one".to_string()), 1.0, true);
        let div = graph2.add_binary_op(Operation::Div, one, zero, true).unwrap();
        assert!(graph2.forward().is_err());
    }
}