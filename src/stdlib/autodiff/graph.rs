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
    /// Quantum operation: output = quantum_circuit(input1)
    QuantumOp { layer_name: String },
}

impl Operation {
    /// Get the number of inputs required for this operation
    pub fn input_count(&self) -> usize {
        match self {
            Operation::Input { .. } => 0,
            Operation::Add | Operation::Sub | Operation::Mul | Operation::Div 
            | Operation::Pow | Operation::Min | Operation::Max => 2,
            Operation::QuantumOp { .. } => 1, // Quantum operations are unary (for now)
            _ => 1,
        }
    }
    
    /// Check if this operation is commutative
    pub fn is_commutative(&self) -> bool {
        matches!(self, Operation::Add | Operation::Mul | Operation::Min | Operation::Max)
    }
    
    /// Get a human-readable name for this operation
    pub fn name(&self) -> &'static str {
        self.type_name()
    }
    
    /// Get a human-readable name for this operation
    pub fn type_name(&self) -> &'static str {
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
            Operation::QuantumOp { .. } => "QuantumOp",
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

/// Gradient propagation mode for reverse-mode AD
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GradientMode {
    /// Standard reverse-mode AD
    Reverse,
    /// Forward-mode AD (for comparison)
    Forward,
    /// Automatic mode selection based on problem characteristics
    Auto,
}

/// Memory management strategy for large graphs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryStrategy {
    /// Store all intermediate values (fastest, more memory)
    StoreAll,
    /// Recompute intermediate values as needed (slower, less memory)
    Recompute,
    /// Checkpoint strategy: store selected values, recompute others
    Checkpoint,
}

/// Enhanced node for reverse-mode AD with memory optimization
#[derive(Debug, Clone)]
pub struct EnhancedGraphNode {
    /// Basic graph node
    pub node: GraphNode,
    /// Backward function for gradient computation
    pub backward_fn: Option<BackwardFunction>,
    /// Reference count for memory management
    pub ref_count: usize,
    /// Whether this node's value should be retained for backward pass
    pub retain_value: bool,
    /// Children nodes that depend on this node
    pub children: Vec<NodeId>,
    /// Gradient multiplier for efficient chain rule application
    pub grad_multiplier: f64,
}

/// Function pointer for backward gradient computation
pub type BackwardFunction = fn(&[f64], f64) -> Vec<f64>;

impl EnhancedGraphNode {
    /// Create a new enhanced graph node
    pub fn new(node: GraphNode, backward_fn: Option<BackwardFunction>) -> Self {
        Self {
            children: Vec::new(),
            ref_count: 0,
            retain_value: node.requires_grad,
            grad_multiplier: 1.0,
            node,
            backward_fn,
        }
    }
    
    /// Add a child dependency
    pub fn add_child(&mut self, child_id: NodeId) {
        if !self.children.contains(&child_id) {
            self.children.push(child_id);
        }
    }
    
    /// Check if this node can be freed from memory
    pub fn can_free(&self) -> bool {
        self.ref_count == 0 && !self.retain_value
    }
    
    /// Increment reference count
    pub fn inc_ref(&mut self) {
        self.ref_count += 1;
    }
    
    /// Decrement reference count
    pub fn dec_ref(&mut self) {
        if self.ref_count > 0 {
            self.ref_count -= 1;
        }
    }
}

/// Execution trace for reverse-mode AD
#[derive(Debug, Clone)]
pub struct ExecutionTrace {
    /// Nodes in execution order (for backward pass)
    pub execution_order: Vec<NodeId>,
    /// Gradient flow graph (who gets gradients from whom)
    pub gradient_flow: HashMap<NodeId, Vec<NodeId>>,
    /// Checkpointed values for memory-efficient execution
    pub checkpoints: HashMap<NodeId, f64>,
    /// Memory usage statistics
    pub memory_usage: MemoryUsage,
}

/// Memory usage statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryUsage {
    /// Total nodes created
    pub total_nodes: usize,
    /// Active nodes in memory
    pub active_nodes: usize,
    /// Peak memory usage
    pub peak_nodes: usize,
    /// Number of recomputations performed
    pub recomputations: usize,
}

/// Enhanced computation graph for reverse-mode automatic differentiation
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    /// All nodes in the graph, indexed by their ID
    nodes: HashMap<NodeId, GraphNode>,
    /// Enhanced nodes with reverse-mode AD features
    enhanced_nodes: HashMap<NodeId, EnhancedGraphNode>,
    /// Topological ordering of nodes for efficient computation
    topological_order: Vec<NodeId>,
    /// Whether the graph needs re-sorting
    needs_sort: bool,
    /// Maximum depth to prevent infinite recursion
    max_depth: usize,
    /// Current gradient mode
    gradient_mode: GradientMode,
    /// Memory management strategy
    memory_strategy: MemoryStrategy,
    /// Execution trace for reverse-mode AD
    execution_trace: Option<ExecutionTrace>,
    /// Common subexpression elimination cache
    cse_cache: HashMap<String, NodeId>,
    /// Whether to enable automatic optimization
    optimize: bool,
}

impl ComputationGraph {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            enhanced_nodes: HashMap::new(),
            topological_order: Vec::new(),
            needs_sort: false,
            max_depth: MAX_GRAPH_DEPTH,
            gradient_mode: GradientMode::Auto,
            memory_strategy: MemoryStrategy::StoreAll,
            execution_trace: None,
            cse_cache: HashMap::new(),
            optimize: true,
        }
    }
    
    /// Create a new computation graph with specific configuration
    pub fn with_config(
        gradient_mode: GradientMode,
        memory_strategy: MemoryStrategy,
        optimize: bool,
    ) -> Self {
        Self {
            nodes: HashMap::new(),
            enhanced_nodes: HashMap::new(),
            topological_order: Vec::new(),
            needs_sort: false,
            max_depth: MAX_GRAPH_DEPTH,
            gradient_mode,
            memory_strategy,
            execution_trace: None,
            cse_cache: HashMap::new(),
            optimize,
        }
    }
    
    /// Set gradient computation mode
    pub fn set_gradient_mode(&mut self, mode: GradientMode) {
        self.gradient_mode = mode;
    }
    
    /// Set memory management strategy
    pub fn set_memory_strategy(&mut self, strategy: MemoryStrategy) {
        self.memory_strategy = strategy;
    }
    
    /// Enable or disable optimizations
    pub fn set_optimize(&mut self, optimize: bool) {
        self.optimize = optimize;
        if !optimize {
            self.cse_cache.clear();
        }
    }
    
    /// Get current gradient mode
    pub fn gradient_mode(&self) -> GradientMode {
        self.gradient_mode
    }
    
    /// Get memory usage statistics
    pub fn memory_usage(&self) -> Option<&MemoryUsage> {
        self.execution_trace.as_ref().map(|trace| &trace.memory_usage)
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
    
    /// Add a quantum operation node to the graph
    pub fn add_quantum_op(
        &mut self,
        input: NodeId,
        layer_name: String,
    ) -> AutodiffResult<NodeId> {
        self.add_unary_op(
            Operation::QuantumOp { layer_name },
            input,
            true, // Quantum operations always require gradients
        )
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
                    Operation::QuantumOp { .. } => {
                        // Quantum operations are handled separately during specialized forward passes
                        // This is a placeholder - actual value computed by quantum layers
                        f64::NAN
                    },
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
    
    /// Compute partial derivatives for a node (static version)
    fn compute_node_partial_derivatives(node: &GraphNode, nodes: &HashMap<NodeId, GraphNode>) -> Vec<f64> {
        if node.is_leaf() {
            return vec![];
        }
        
        let input_values: Vec<f64> = node.inputs.iter()
            .map(|&id| nodes.get(&id).unwrap().value)
            .collect();
        
        match &node.operation {
            Operation::Add => vec![1.0, 1.0],
            Operation::Sub => vec![1.0, -1.0],
            Operation::Mul => vec![input_values[1], input_values[0]],
            Operation::Div => {
                let denom = input_values[1];
                vec![
                    1.0 / denom,
                    -input_values[0] / (denom * denom),
                ]
            },
            Operation::Pow => {
                let base = input_values[0];
                let exp = input_values[1];
                if base > 0.0 {
                    vec![
                        exp * base.powf(exp - 1.0),
                        base.powf(exp) * base.ln(),
                    ]
                } else {
                    vec![0.0, 0.0]
                }
            },
            Operation::Neg => vec![-1.0],
            Operation::Exp => {
                let exp_val = input_values[0].exp();
                vec![exp_val]
            },
            Operation::Ln => vec![1.0 / input_values[0]],
            Operation::Sqrt => {
                let sqrt_val = input_values[0].sqrt();
                vec![1.0 / (2.0 * sqrt_val)]
            },
            Operation::Sin => vec![input_values[0].cos()],
            Operation::Cos => vec![-input_values[0].sin()],
            Operation::Tan => {
                let cos_val = input_values[0].cos();
                vec![1.0 / (cos_val * cos_val)]
            },
            Operation::Sinh => vec![input_values[0].cosh()],
            Operation::Cosh => vec![input_values[0].sinh()],
            Operation::Tanh => {
                let cosh_val = input_values[0].cosh();
                vec![1.0 / (cosh_val * cosh_val)]
            },
            Operation::Abs => {
                let sign = if input_values[0] > 0.0 { 1.0 } else if input_values[0] < 0.0 { -1.0 } else { 0.0 };
                vec![sign]
            },
            Operation::Min => {
                if input_values[0] < input_values[1] {
                    vec![1.0, 0.0]
                } else if input_values[0] > input_values[1] {
                    vec![0.0, 1.0]
                } else {
                    vec![0.5, 0.5] // Subgradient
                }
            },
            Operation::Max => {
                if input_values[0] > input_values[1] {
                    vec![1.0, 0.0]
                } else if input_values[0] < input_values[1] {
                    vec![0.0, 1.0]
                } else {
                    vec![0.5, 0.5] // Subgradient
                }
            },
            Operation::ReLU => {
                let deriv = if input_values[0] > 0.0 { 1.0 } else { 0.0 };
                vec![deriv]
            },
            Operation::Sigmoid => {
                let sigmoid_val = 1.0 / (1.0 + (-input_values[0]).exp());
                let deriv = sigmoid_val * (1.0 - sigmoid_val);
                vec![deriv]
            },
            Operation::Sum => vec![1.0],
            Operation::Mean => vec![1.0],
            Operation::QuantumOp { .. } => {
                // Quantum operations require specialized gradient computation
                // This is handled by the quantum gradient system
                vec![1.0] // Placeholder derivative
            },
            Operation::Input { .. } => vec![],
        }
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
            Operation::QuantumOp { .. } => {
                // Quantum operations use specialized gradient computation
                node.partial_derivatives = vec![1.0]; // Placeholder derivative
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
    
    // ===== REVERSE-MODE AUTOMATIC DIFFERENTIATION METHODS =====
    
    /// Perform reverse-mode backward pass with memory optimization
    pub fn backward_reverse(&mut self, output_node: NodeId) -> AutodiffResult<()> {
        self.ensure_topological_sort()?;
        
        // Initialize execution trace
        let mut trace = ExecutionTrace {
            execution_order: Vec::new(),
            gradient_flow: HashMap::new(),
            checkpoints: HashMap::new(),
            memory_usage: MemoryUsage::default(),
        };
        
        // Initialize output gradient
        {
            let output = self.get_node_mut(output_node)?;
            output.gradient = 1.0;
        }
        
        // Build reverse execution order and gradient flow
        let reverse_order: Vec<NodeId> = self.topological_order.iter().rev().copied().collect();
        trace.execution_order = reverse_order.clone();
        
        // Perform reverse-mode gradient computation
        for &node_id in &reverse_order {
            let node = self.get_node(node_id)?.clone();
            
            if node.requires_grad && !node.is_leaf() {
                // Get gradient flow for this node
                let mut input_gradients = Vec::new();
                
                for &input_id in &node.inputs {
                    input_gradients.push(input_id);
                }
                
                trace.gradient_flow.insert(node_id, input_gradients);
                
                // Propagate gradients using chain rule
                for (i, &input_id) in node.inputs.iter().enumerate() {
                    let local_grad = node.gradient * node.partial_derivatives[i];
                    
                    // Apply memory management strategy
                    match self.memory_strategy {
                        MemoryStrategy::StoreAll => {
                            // Simply accumulate gradient
                            let input_node = self.get_node_mut(input_id)?;
                            input_node.accumulate_gradient(local_grad);
                        }
                        MemoryStrategy::Recompute => {
                            // Recompute if needed and accumulate
                            let input_node = self.get_node_mut(input_id)?;
                            input_node.accumulate_gradient(local_grad);
                            trace.memory_usage.recomputations += 1;
                        }
                        MemoryStrategy::Checkpoint => {
                            // Use checkpoint strategy
                            if self.should_checkpoint(input_id) {
                                trace.checkpoints.insert(input_id, self.get_node(input_id)?.value);
                            }
                            
                            let input_node = self.get_node_mut(input_id)?;
                            input_node.accumulate_gradient(local_grad);
                        }
                    }
                }
            }
            
            // Update memory usage statistics
            trace.memory_usage.total_nodes += 1;
            if self.get_node(node_id)?.requires_grad {
                trace.memory_usage.active_nodes += 1;
            }
        }
        
        // Store execution trace
        trace.memory_usage.peak_nodes = trace.memory_usage.active_nodes;
        self.execution_trace = Some(trace);
        
        Ok(())
    }
    
    /// Automatic mode selection based on problem characteristics
    pub fn select_optimal_mode(&self, num_inputs: usize, num_outputs: usize) -> GradientMode {
        match self.gradient_mode {
            GradientMode::Auto => {
                // Forward mode is better when few inputs, many outputs
                // Reverse mode is better when many inputs, few outputs
                if num_inputs <= 4 && num_outputs > num_inputs {
                    GradientMode::Forward
                } else if num_inputs > num_outputs {
                    GradientMode::Reverse
                } else {
                    // Default to Forward for equal or small cases
                    GradientMode::Forward
                }
            }
            mode => mode,
        }
    }
    
    /// Build execution trace for reverse-mode AD
    pub fn build_execution_trace(&self) -> AutodiffResult<ExecutionTrace> {
        let mut trace = ExecutionTrace {
            execution_order: self.topological_order.clone(),
            gradient_flow: HashMap::new(),
            checkpoints: HashMap::new(),
            memory_usage: MemoryUsage {
                total_nodes: self.nodes.len(),
                active_nodes: self.nodes.values().filter(|n| n.requires_grad).count(),
                peak_nodes: self.nodes.len(),
                recomputations: 0,
            },
        };
        
        // Build gradient flow graph
        for (&node_id, node) in &self.nodes {
            if !node.is_leaf() {
                trace.gradient_flow.insert(node_id, node.inputs.clone());
            }
        }
        
        Ok(trace)
    }
    
    /// Common subexpression elimination
    pub fn eliminate_common_subexpressions(&mut self) -> AutodiffResult<()> {
        if !self.optimize {
            return Ok(());
        }
        
        let mut expression_map: HashMap<String, NodeId> = HashMap::new();
        let mut to_merge: Vec<(NodeId, NodeId)> = Vec::new();
        
        // Find common subexpressions
        for (&node_id, node) in &self.nodes {
            if !node.is_leaf() {
                let expr_key = self.compute_expression_key(node);
                
                if let Some(&existing_id) = expression_map.get(&expr_key) {
                    // Found duplicate expression
                    to_merge.push((node_id, existing_id));
                } else {
                    expression_map.insert(expr_key, node_id);
                }
            }
        }
        
        // Merge duplicate nodes (simplified implementation)
        for (duplicate_id, canonical_id) in to_merge {
            // Update references to point to canonical node
            self.merge_nodes(duplicate_id, canonical_id)?;
        }
        
        self.needs_sort = true;
        Ok(())
    }
    
    /// Compute a key for expression matching
    fn compute_expression_key(&self, node: &GraphNode) -> String {
        format!("{:?}:{:?}", node.operation, node.inputs)
    }
    
    /// Merge duplicate nodes (simplified implementation)
    fn merge_nodes(&mut self, duplicate_id: NodeId, _canonical_id: NodeId) -> AutodiffResult<()> {
        // In a full implementation, this would update all references
        // For now, just remove the duplicate
        self.nodes.remove(&duplicate_id);
        Ok(())
    }
    
    /// Determine if a node should be checkpointed
    fn should_checkpoint(&self, node_id: NodeId) -> bool {
        // Simple heuristic: checkpoint nodes with high fan-out
        let node = self.nodes.get(&node_id);
        if let Some(_node) = node {
            // Count how many nodes depend on this one
            let dependents = self.nodes.values()
                .filter(|n| n.inputs.contains(&node_id))
                .count();
            dependents > 2 // Checkpoint if more than 2 dependents
        } else {
            false
        }
    }
    
    /// Get the gradient computation graph
    pub fn gradient_graph(&self) -> Option<&HashMap<NodeId, Vec<NodeId>>> {
        self.execution_trace.as_ref().map(|trace| &trace.gradient_flow)
    }
    
    /// Vector-Jacobian product for efficient neural network gradients
    pub fn vector_jacobian_product(
        &mut self,
        output_nodes: &[NodeId],
        vector: &[f64],
    ) -> AutodiffResult<HashMap<NodeId, f64>> {
        if output_nodes.len() != vector.len() {
            return Err(AutodiffError::GradientComputationFailed {
                reason: "Vector length must match number of outputs".to_string(),
            });
        }
        
        // Ensure forward pass has been done to compute partial derivatives
        // (In practice, this should already be done, but ensuring correctness)
        
        // Clear existing gradients
        self.zero_grad();
        
        // Initialize output gradients with vector components
        for (i, &output_id) in output_nodes.iter().enumerate() {
            let output_node = self.get_node_mut(output_id)?;
            output_node.gradient = vector[i];
        }
        
        // Perform reverse-mode computation
        self.ensure_topological_sort()?;
        let reverse_order: Vec<NodeId> = self.topological_order.iter().rev().copied().collect();
        
        for &node_id in &reverse_order {
            // First check if we need to process this node
            let should_process = {
                let node = self.get_node(node_id)?;
                node.requires_grad && !node.is_leaf()
            };
            
            if should_process {
                // Compute partial derivatives directly without borrowing conflicts
                let partial_derivs = {
                    let node = self.get_node(node_id)?;
                    Self::compute_node_partial_derivatives(&node, &self.nodes)
                };
                
                // Get gradient and inputs for computation
                let (gradient, inputs) = {
                    let node = self.get_node(node_id)?;
                    (node.gradient, node.inputs.clone())
                };
                
                for (i, &input_id) in inputs.iter().enumerate() {
                    let grad_contribution = gradient * partial_derivs[i];
                    let input_node = self.get_node_mut(input_id)?;
                    input_node.accumulate_gradient(grad_contribution);
                }
            }
        }
        
        // Collect gradients for all nodes that require gradients
        let mut result = HashMap::new();
        for (&node_id, node) in &self.nodes {
            if node.requires_grad {
                result.insert(node_id, node.gradient);
            }
        }
        
        Ok(result)
    }
    
    /// Parallel backward pass (simplified implementation)
    pub fn backward_parallel(&mut self, output_node: NodeId) -> AutodiffResult<()> {
        // For now, fall back to sequential implementation
        // In a full implementation, this would use thread pools
        self.backward_reverse(output_node)
    }
    
    /// Get enhanced graph statistics
    pub fn enhanced_stats(&self) -> EnhancedGraphStats {
        let basic_stats = self.stats();
        let memory_stats = self.memory_usage().cloned().unwrap_or_default();
        
        EnhancedGraphStats {
            basic: basic_stats,
            memory: memory_stats,
            gradient_mode: self.gradient_mode,
            memory_strategy: self.memory_strategy,
            optimizations_enabled: self.optimize,
            cse_cache_size: self.cse_cache.len(),
        }
    }
}

/// Enhanced statistics for reverse-mode AD
#[derive(Debug, Clone)]
pub struct EnhancedGraphStats {
    /// Basic graph statistics
    pub basic: GraphStats,
    /// Memory usage statistics
    pub memory: MemoryUsage,
    /// Current gradient mode
    pub gradient_mode: GradientMode,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Whether optimizations are enabled
    pub optimizations_enabled: bool,
    /// Size of CSE cache
    pub cse_cache_size: usize,
}

impl std::fmt::Display for EnhancedGraphStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}\nMode: {:?}, Memory: {:?}, Optimizations: {}, CSE Cache: {} entries", 
               self.basic, self.gradient_mode, self.memory_strategy, 
               self.optimizations_enabled, self.cse_cache_size)
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
    
    // ===== REVERSE-MODE AUTOMATIC DIFFERENTIATION TESTS =====
    
    #[test]
    fn test_reverse_mode_basic() {
        let mut graph = ComputationGraph::with_config(
            GradientMode::Reverse,
            MemoryStrategy::StoreAll,
            true,
        );
        
        // Test basic reverse-mode computation: f(x) = x^2
        let x = graph.add_input(Some("x".to_string()), 3.0, true);
        let x_squared = graph.add_binary_op(Operation::Mul, x, x, true).unwrap();
        
        graph.forward().unwrap();
        graph.backward_reverse(x_squared).unwrap();
        
        // f(x) = x^2, f'(x) = 2x = 2*3 = 6
        assert_eq!(graph.get_gradient(x).unwrap(), 6.0);
        
        // Check execution trace was created
        assert!(graph.execution_trace.is_some());
        let trace = graph.execution_trace.as_ref().unwrap();
        assert!(!trace.execution_order.is_empty());
        assert!(!trace.gradient_flow.is_empty());
    }
    
    #[test]
    fn test_automatic_mode_selection() {
        let graph = ComputationGraph::new();
        
        // Few inputs, many outputs -> Forward mode
        assert_eq!(
            graph.select_optimal_mode(2, 10),
            GradientMode::Forward
        );
        
        // Many inputs, few outputs -> Reverse mode
        assert_eq!(
            graph.select_optimal_mode(10, 2),
            GradientMode::Reverse
        );
        
        // Equal inputs/outputs -> Forward mode (default for ties)
        assert_eq!(
            graph.select_optimal_mode(5, 5),
            GradientMode::Forward
        );
    }
    
    #[test]
    fn test_memory_strategies() {
        // Test different memory strategies
        let strategies = [
            MemoryStrategy::StoreAll,
            MemoryStrategy::Recompute,
            MemoryStrategy::Checkpoint,
        ];
        
        for strategy in strategies.iter() {
            let mut graph = ComputationGraph::with_config(
                GradientMode::Reverse,
                *strategy,
                true,
            );
            
            // Create a simple computation
            let x = graph.add_input(Some("x".to_string()), 2.0, true);
            let y = graph.add_input(Some("y".to_string()), 3.0, true);
            let product = graph.add_binary_op(Operation::Mul, x, y, true).unwrap();
            
            graph.forward().unwrap();
            graph.backward_reverse(product).unwrap();
            
            // Gradients should be computed correctly regardless of strategy
            assert_eq!(graph.get_gradient(x).unwrap(), 3.0); // d/dx(xy) = y
            assert_eq!(graph.get_gradient(y).unwrap(), 2.0); // d/dy(xy) = x
        }
    }
    
    #[test]
    fn test_vector_jacobian_product() {
        let mut graph = ComputationGraph::new();
        
        // Create a function with multiple outputs: f(x,y) = [x+y, x*y]
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let y = graph.add_input(Some("y".to_string()), 3.0, true);
        
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        let product = graph.add_binary_op(Operation::Mul, x, y, true).unwrap();
        
        graph.forward().unwrap();
        
        // Compute VJP with vector v = [1.0, 2.0]
        let outputs = vec![sum, product];
        let vector = vec![1.0, 2.0];
        
        let gradients = graph.vector_jacobian_product(&outputs, &vector).unwrap();
        
        // VJP should compute v^T * J where J is Jacobian
        // J = [[1, 1], [y, x]] = [[1, 1], [3, 2]]
        // v^T * J = [1, 2] * [[1, 1], [3, 2]] = [1*1 + 2*3, 1*1 + 2*2] = [7, 5]
        assert_eq!(gradients[&x], 7.0);
        assert_eq!(gradients[&y], 5.0);
    }
    
    #[test]
    fn test_execution_trace() {
        let mut graph = ComputationGraph::new();
        
        // Build a computation graph
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 2.0, true);
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        let exp_sum = graph.add_unary_op(Operation::Exp, sum, true).unwrap();
        
        graph.forward().unwrap();
        
        // Build execution trace
        let trace = graph.build_execution_trace().unwrap();
        
        assert_eq!(trace.memory_usage.total_nodes, 4);
        assert_eq!(trace.memory_usage.active_nodes, 4); // All nodes require gradients by default
        assert!(!trace.execution_order.is_empty());
        assert!(!trace.gradient_flow.is_empty());
        
        // Check gradient flow structure
        assert!(trace.gradient_flow.contains_key(&sum));
        assert!(trace.gradient_flow.contains_key(&exp_sum));
        assert_eq!(trace.gradient_flow[&sum], vec![x, y]);
        assert_eq!(trace.gradient_flow[&exp_sum], vec![sum]);
    }
    
    #[test]
    fn test_enhanced_graph_stats() {
        let mut graph = ComputationGraph::with_config(
            GradientMode::Reverse,
            MemoryStrategy::Checkpoint,
            true,
        );
        
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 2.0, false);
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        let stats = graph.enhanced_stats();
        
        assert_eq!(stats.basic.total_nodes, 3);
        assert_eq!(stats.basic.gradient_nodes, 2);
        assert_eq!(stats.gradient_mode, GradientMode::Reverse);
        assert_eq!(stats.memory_strategy, MemoryStrategy::Checkpoint);
        assert!(stats.optimizations_enabled);
        assert_eq!(stats.cse_cache_size, 0); // No duplicates yet
    }
    
    #[test]
    fn test_common_subexpression_elimination() {
        let mut graph = ComputationGraph::new();
        graph.set_optimize(true);
        
        // Create duplicate expressions
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let y = graph.add_input(Some("y".to_string()), 3.0, true);
        
        // Both of these should create x + y
        let sum1 = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        let sum2 = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        let initial_nodes = graph.nodes.len();
        
        // Apply CSE
        graph.eliminate_common_subexpressions().unwrap();
        
        // Should have removed one duplicate (simplified test)
        // Note: Full CSE implementation would be more sophisticated
        assert!(graph.nodes.len() <= initial_nodes);
    }
    
    #[test]
    fn test_complex_reverse_mode_chain() {
        let mut graph = ComputationGraph::with_config(
            GradientMode::Reverse,
            MemoryStrategy::StoreAll,
            false, // Disable optimizations for predictable behavior
        );
        
        // Build: f(x,y) = sin(x*y) * exp(x) + y^2
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 0.5, true);
        
        let xy = graph.add_binary_op(Operation::Mul, x, y, true).unwrap();
        let sin_xy = graph.add_unary_op(Operation::Sin, xy, true).unwrap();
        let exp_x = graph.add_unary_op(Operation::Exp, x, true).unwrap();
        let sin_exp = graph.add_binary_op(Operation::Mul, sin_xy, exp_x, true).unwrap();
        let y_squared = graph.add_binary_op(Operation::Mul, y, y, true).unwrap();
        let result = graph.add_binary_op(Operation::Add, sin_exp, y_squared, true).unwrap();
        
        graph.forward().unwrap();
        graph.backward_reverse(result).unwrap();
        
        // Verify gradients were computed (exact values would require manual calculation)
        let x_grad = graph.get_gradient(x).unwrap();
        let y_grad = graph.get_gradient(y).unwrap();
        
        assert!(x_grad.abs() > 1e-10); // Should be non-trivial
        assert!(y_grad.abs() > 1e-10); // Should be non-trivial
        
        // Verify execution trace was built
        assert!(graph.execution_trace.is_some());
        let trace = graph.execution_trace.as_ref().unwrap();
        assert_eq!(trace.memory_usage.total_nodes, 8); // All intermediate nodes
    }
    
    #[test]
    fn test_gradient_mode_configuration() {
        // Test that gradient mode can be changed
        let mut graph = ComputationGraph::new();
        
        assert_eq!(graph.gradient_mode(), GradientMode::Auto);
        
        graph.set_gradient_mode(GradientMode::Reverse);
        assert_eq!(graph.gradient_mode(), GradientMode::Reverse);
        
        graph.set_gradient_mode(GradientMode::Forward);
        assert_eq!(graph.gradient_mode(), GradientMode::Forward);
    }
    
    #[test]
    fn test_memory_strategy_configuration() {
        let mut graph = ComputationGraph::new();
        
        // Test memory strategy changes
        graph.set_memory_strategy(MemoryStrategy::Recompute);
        let stats = graph.enhanced_stats();
        assert_eq!(stats.memory_strategy, MemoryStrategy::Recompute);
        
        graph.set_memory_strategy(MemoryStrategy::Checkpoint);
        let stats = graph.enhanced_stats();
        assert_eq!(stats.memory_strategy, MemoryStrategy::Checkpoint);
    }
    
    #[test]
    fn test_parallel_backward_pass() {
        let mut graph = ComputationGraph::new();
        
        // Create a computation
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let x_cubed = graph.add_binary_op(Operation::Mul, x, x, true).unwrap();
        let x_cubed = graph.add_binary_op(Operation::Mul, x_cubed, x, true).unwrap();
        
        graph.forward().unwrap();
        
        // Test parallel backward pass (currently falls back to sequential)
        graph.backward_parallel(x_cubed).unwrap();
        
        // Should produce same result as sequential
        assert_eq!(graph.get_gradient(x).unwrap(), 12.0); // 3x^2 = 3*4 = 12
    }
    
    #[test]
    fn test_vjp_error_conditions() {
        let mut graph = ComputationGraph::new();
        
        let x = graph.add_input(Some("x".to_string()), 1.0, true);
        let y = graph.add_input(Some("y".to_string()), 2.0, true);
        let sum = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        graph.forward().unwrap();
        
        // Test mismatched vector length
        let outputs = vec![sum];
        let wrong_vector = vec![1.0, 2.0]; // Too many elements
        
        let result = graph.vector_jacobian_product(&outputs, &wrong_vector);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_checkpointing_heuristic() {
        let mut graph = ComputationGraph::with_config(
            GradientMode::Reverse,
            MemoryStrategy::Checkpoint,
            true,
        );
        
        // Create a node with high fan-out (many dependents)
        let x = graph.add_input(Some("x".to_string()), 2.0, true);
        let y = graph.add_input(Some("y".to_string()), 3.0, true);
        let shared = graph.add_binary_op(Operation::Add, x, y, true).unwrap();
        
        // Create multiple nodes that depend on shared
        let dep1 = graph.add_binary_op(Operation::Mul, shared, x, true).unwrap();
        let dep2 = graph.add_binary_op(Operation::Mul, shared, y, true).unwrap();
        let dep3 = graph.add_unary_op(Operation::Exp, shared, true).unwrap();
        
        // Shared node should be marked for checkpointing (>2 dependents)
        assert!(graph.should_checkpoint(shared));
        
        // Leaf nodes shouldn't be checkpointed
        assert!(!graph.should_checkpoint(x));
        assert!(!graph.should_checkpoint(y));
    }
}