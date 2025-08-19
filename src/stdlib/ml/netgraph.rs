//! NetGraph Implementation
//!
//! NetGraph represents complex neural network graphs with multiple inputs/outputs,
//! modeling the Wolfram Language NetGraph functionality for arbitrary DAG topologies.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::{Layer, Tensor};
use std::collections::{HashMap, VecDeque};
use std::fmt;

/// Node identifier in the computation graph
pub type NodeId = String;

/// Port identifier for multi-input/output nodes
pub type PortId = String;

/// Connection between nodes in the graph
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    /// Source node and optional output port
    pub from: (NodeId, Option<PortId>),
    /// Target node and optional input port  
    pub to: (NodeId, Option<PortId>),
}

impl Connection {
    /// Create a simple connection between two nodes
    pub fn simple(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: (from.into(), None),
            to: (to.into(), None),
        }
    }
    
    /// Create a connection with specific ports
    pub fn with_ports(
        from_node: impl Into<String>,
        from_port: Option<impl Into<String>>,
        to_node: impl Into<String>, 
        to_port: Option<impl Into<String>>,
    ) -> Self {
        Self {
            from: (from_node.into(), from_port.map(|p| p.into())),
            to: (to_node.into(), to_port.map(|p| p.into())),
        }
    }
}

/// Node in the NetGraph with associated layer
#[derive(Debug)]
pub struct GraphNode {
    /// Unique identifier for this node
    pub id: NodeId,
    /// The neural network layer
    pub layer: Box<dyn Layer>,
    /// Input connections (from other nodes)
    pub inputs: Vec<Connection>,
    /// Output connections (to other nodes)
    pub outputs: Vec<Connection>,
    /// Whether this is an input node to the graph
    pub is_input: bool,
    /// Whether this is an output node from the graph
    pub is_output: bool,
    /// Cached output from last forward pass
    pub cached_output: Option<Tensor>,
}

impl GraphNode {
    pub fn new(id: NodeId, layer: Box<dyn Layer>) -> Self {
        Self {
            id,
            layer,
            inputs: Vec::new(),
            outputs: Vec::new(),
            is_input: false,
            is_output: false,
            cached_output: None,
        }
    }
    
    pub fn as_input(mut self) -> Self {
        self.is_input = true;
        self
    }
    
    pub fn as_output(mut self) -> Self {
        self.is_output = true;
        self
    }
}

/// NetGraph: Complex neural network graph with arbitrary topology
/// 
/// NetGraph[{associations}, {connections}] creates a neural network with:
/// - associations: {node_id -> layer} mappings
/// - connections: list of connections between nodes
/// 
/// # Examples
/// ```rust
/// // Simple residual connection
/// let graph = NetGraphBuilder::new()
///     .add_node("input", InputLayer::new())
///     .add_node("conv1", ConvLayer::new(64, 3))
///     .add_node("conv2", ConvLayer::new(64, 3))
///     .add_node("add", AddLayer::new())
///     .add_node("output", OutputLayer::new())
///     .connect("input", "conv1")
///     .connect("conv1", "conv2")
///     .connect("conv2", "add")
///     .connect("input", "add")  // Residual connection
///     .connect("add", "output")
///     .build();
/// ```
#[derive(Debug)]
pub struct NetGraph {
    /// All nodes in the graph
    nodes: HashMap<NodeId, GraphNode>,
    /// All connections between nodes
    connections: Vec<Connection>,
    /// Topologically sorted execution order
    execution_order: Vec<NodeId>,
    /// Input nodes to the graph
    input_nodes: Vec<NodeId>,
    /// Output nodes from the graph
    output_nodes: Vec<NodeId>,
    /// Network name for debugging
    name: String,
    /// Whether the graph has been validated and compiled
    compiled: bool,
}

impl NetGraph {
    /// Create a new NetGraph from nodes and connections
    pub fn new(
        nodes: HashMap<NodeId, GraphNode>,
        connections: Vec<Connection>,
        name: String,
    ) -> MLResult<Self> {
        let mut graph = Self {
            nodes,
            connections,
            execution_order: Vec::new(),
            input_nodes: Vec::new(),
            output_nodes: Vec::new(),
            name,
            compiled: false,
        };
        
        graph.compile()?;
        Ok(graph)
    }
    
    /// Compile the graph: validate DAG property and compute execution order
    fn compile(&mut self) -> MLResult<()> {
        // Reset compilation state
        self.execution_order.clear();
        self.input_nodes.clear();
        self.output_nodes.clear();
        
        // Build adjacency lists
        self.update_node_connections()?;
        
        // Find input and output nodes
        self.identify_input_output_nodes();
        
        // Validate DAG property and compute topological order
        self.execution_order = self.topological_sort()?;
        
        self.compiled = true;
        Ok(())
    }
    
    /// Update input/output connections for each node
    fn update_node_connections(&mut self) -> MLResult<()> {
        // Clear existing connections
        for node in self.nodes.values_mut() {
            node.inputs.clear();
            node.outputs.clear();
        }
        
        // Rebuild connections
        for connection in &self.connections {
            let from_id = &connection.from.0;
            let to_id = &connection.to.0;
            
            // Verify nodes exist
            if !self.nodes.contains_key(from_id) {
                return Err(MLError::NetworkError {
                    reason: format!("Source node '{}' not found in graph", from_id),
                });
            }
            if !self.nodes.contains_key(to_id) {
                return Err(MLError::NetworkError {
                    reason: format!("Target node '{}' not found in graph", to_id),
                });
            }
            
            // Add connections to nodes
            if let Some(from_node) = self.nodes.get_mut(from_id) {
                from_node.outputs.push(connection.clone());
            }
            if let Some(to_node) = self.nodes.get_mut(to_id) {
                to_node.inputs.push(connection.clone());
            }
        }
        
        Ok(())
    }
    
    /// Identify input and output nodes based on connections
    fn identify_input_output_nodes(&mut self) {
        for (node_id, node) in &self.nodes {
            // Input nodes have no inputs or are explicitly marked
            if node.inputs.is_empty() || node.is_input {
                self.input_nodes.push(node_id.clone());
            }
            
            // Output nodes have no outputs or are explicitly marked
            if node.outputs.is_empty() || node.is_output {
                self.output_nodes.push(node_id.clone());
            }
        }
        
        // Sort for deterministic behavior
        self.input_nodes.sort();
        self.output_nodes.sort();
    }
    
    /// Perform topological sort to get execution order
    fn topological_sort(&self) -> MLResult<Vec<NodeId>> {
        let mut in_degree: HashMap<NodeId, usize> = HashMap::new();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        
        // Initialize in-degree count and adjacency list
        for node_id in self.nodes.keys() {
            in_degree.insert(node_id.clone(), 0);
            adjacency.insert(node_id.clone(), Vec::new());
        }
        
        // Build adjacency list and compute in-degrees
        for connection in &self.connections {
            let from_id = &connection.from.0;
            let to_id = &connection.to.0;
            
            adjacency.get_mut(from_id).unwrap().push(to_id.clone());
            *in_degree.get_mut(to_id).unwrap() += 1;
        }
        
        // Start with nodes that have no incoming edges
        let mut queue: VecDeque<NodeId> = in_degree
            .iter()
            .filter(|(_, &degree)| degree == 0)
            .map(|(node_id, _)| node_id.clone())
            .collect();
        
        let mut topo_order = Vec::new();
        
        while let Some(node_id) = queue.pop_front() {
            topo_order.push(node_id.clone());
            
            // Reduce in-degree for all neighbors
            for neighbor in &adjacency[&node_id] {
                let neighbor_degree = in_degree.get_mut(neighbor).unwrap();
                *neighbor_degree -= 1;
                
                if *neighbor_degree == 0 {
                    queue.push_back(neighbor.clone());
                }
            }
        }
        
        // Check for cycles
        if topo_order.len() != self.nodes.len() {
            return Err(MLError::NetworkError {
                reason: "Graph contains cycles - not a valid DAG".to_string(),
            });
        }
        
        Ok(topo_order)
    }
    
    /// Forward pass through the entire graph
    pub fn forward(&mut self, inputs: &HashMap<NodeId, Tensor>) -> MLResult<HashMap<NodeId, Tensor>> {
        if !self.compiled {
            return Err(MLError::NetworkError {
                reason: "Graph must be compiled before forward pass".to_string(),
            });
        }
        
        // Clear cached outputs
        for node in self.nodes.values_mut() {
            node.cached_output = None;
        }
        
        // Set input values
        for (input_id, input_tensor) in inputs {
            if let Some(node) = self.nodes.get_mut(input_id) {
                node.cached_output = Some(input_tensor.clone());
            } else {
                return Err(MLError::NetworkError {
                    reason: format!("Input node '{}' not found in graph", input_id),
                });
            }
        }
        
        // Execute nodes in topological order
        let execution_order = self.execution_order.clone();
        for node_id in &execution_order {
            self.execute_node(node_id)?;
        }
        
        // Collect outputs
        let mut outputs = HashMap::new();
        for output_id in &self.output_nodes {
            if let Some(node) = self.nodes.get(output_id) {
                if let Some(ref output_tensor) = node.cached_output {
                    outputs.insert(output_id.clone(), output_tensor.clone());
                } else {
                    return Err(MLError::NetworkError {
                        reason: format!("Output node '{}' has no computed value", output_id),
                    });
                }
            }
        }
        
        Ok(outputs)
    }
    
    /// Execute a single node in the graph
    fn execute_node(&mut self, node_id: &NodeId) -> MLResult<()> {
        // Skip if already computed (input nodes)
        if self.nodes[node_id].cached_output.is_some() {
            return Ok(());
        }
        
        // Collect inputs from predecessor nodes
        let input_connections = self.nodes[node_id].inputs.clone();
        
        if input_connections.is_empty() {
            return Err(MLError::NetworkError {
                reason: format!("Non-input node '{}' has no input connections", node_id),
            });
        }
        
        // For now, assume single input per node (will extend for multi-input)
        if input_connections.len() > 1 {
            return Err(MLError::NetworkError {
                reason: format!("Multi-input nodes not yet supported for node '{}'", node_id),
            });
        }
        
        let input_connection = &input_connections[0];
        let input_node_id = &input_connection.from.0;
        
        // Get input tensor from predecessor
        let input_tensor = if let Some(input_node) = self.nodes.get(input_node_id) {
            if let Some(ref tensor) = input_node.cached_output {
                tensor.clone()
            } else {
                return Err(MLError::NetworkError {
                    reason: format!("Input node '{}' has no computed output", input_node_id),
                });
            }
        } else {
            return Err(MLError::NetworkError {
                reason: format!("Input node '{}' not found", input_node_id),
            });
        };
        
        // Execute this node's layer
        let output_tensor = self.nodes[node_id].layer.forward(&input_tensor)?;
        
        // Cache the output
        self.nodes.get_mut(node_id).unwrap().cached_output = Some(output_tensor);
        
        Ok(())
    }
    
    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
    
    /// Get the number of connections in the graph
    pub fn connection_count(&self) -> usize {
        self.connections.len()
    }
    
    /// Get input node IDs
    pub fn input_nodes(&self) -> &[NodeId] {
        &self.input_nodes
    }
    
    /// Get output node IDs
    pub fn output_nodes(&self) -> &[NodeId] {
        &self.output_nodes
    }
    
    /// Get execution order
    pub fn execution_order(&self) -> &[NodeId] {
        &self.execution_order
    }
    
    /// Get graph summary
    pub fn summary(&self) -> String {
        let mut summary = format!("NetGraph Summary: {}\n", self.name);
        summary.push_str("=".repeat(50).as_str());
        summary.push('\n');
        
        summary.push_str(&format!("Nodes: {}\n", self.node_count()));
        summary.push_str(&format!("Connections: {}\n", self.connection_count()));
        summary.push_str(&format!("Input Nodes: {:?}\n", self.input_nodes));
        summary.push_str(&format!("Output Nodes: {:?}\n", self.output_nodes));
        
        summary.push_str("\nExecution Order:\n");
        summary.push_str("-".repeat(50).as_str());
        summary.push('\n');
        
        for (i, node_id) in self.execution_order.iter().enumerate() {
            if let Some(node) = self.nodes.get(node_id) {
                summary.push_str(&format!(
                    "{:2}: {:15} | Layer: {}\n",
                    i + 1,
                    node_id,
                    node.layer.name()
                ));
            }
        }
        
        summary.push_str("-".repeat(50).as_str());
        summary.push('\n');
        
        summary
    }
    
    /// Get network name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Display for NetGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Builder for convenient NetGraph construction
pub struct NetGraphBuilder {
    nodes: HashMap<NodeId, GraphNode>,
    connections: Vec<Connection>,
    name: Option<String>,
}

impl NetGraphBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            connections: Vec::new(),
            name: None,
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(mut self, id: impl Into<String>, layer: Box<dyn Layer>) -> Self {
        let node_id = id.into();
        let node = GraphNode::new(node_id.clone(), layer);
        self.nodes.insert(node_id, node);
        self
    }
    
    /// Add an input node to the graph
    pub fn add_input_node(mut self, id: impl Into<String>, layer: Box<dyn Layer>) -> Self {
        let node_id = id.into();
        let node = GraphNode::new(node_id.clone(), layer).as_input();
        self.nodes.insert(node_id, node);
        self
    }
    
    /// Add an output node to the graph
    pub fn add_output_node(mut self, id: impl Into<String>, layer: Box<dyn Layer>) -> Self {
        let node_id = id.into();
        let node = GraphNode::new(node_id.clone(), layer).as_output();
        self.nodes.insert(node_id, node);
        self
    }
    
    /// Connect two nodes
    pub fn connect(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.connections.push(Connection::simple(from, to));
        self
    }
    
    /// Connect nodes with specific ports
    pub fn connect_ports(
        mut self,
        from_node: impl Into<String>,
        from_port: Option<impl Into<String>>,
        to_node: impl Into<String>,
        to_port: Option<impl Into<String>>,
    ) -> Self {
        self.connections.push(Connection::with_ports(
            from_node, from_port, to_node, to_port,
        ));
        self
    }
    
    /// Set the graph name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Build the NetGraph
    pub fn build(self) -> MLResult<NetGraph> {
        let name = self.name.unwrap_or_else(|| "NetGraph".to_string());
        NetGraph::new(self.nodes, self.connections, name)
    }
}

impl Default for NetGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::ml::layers::{LinearLayer, ReLULayer, IdentityLayer};
    
    #[test]
    fn test_netgraph_creation() {
        let graph = NetGraphBuilder::new()
            .add_node("input", Box::new(IdentityLayer::new()))
            .add_node("linear", Box::new(LinearLayer::new(10)))
            .add_node("relu", Box::new(ReLULayer::new()))
            .add_node("output", Box::new(IdentityLayer::new()))
            .connect("input", "linear")
            .connect("linear", "relu")
            .connect("relu", "output")
            .with_name("TestGraph".to_string())
            .build()
            .unwrap();
        
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.connection_count(), 3);
        assert_eq!(graph.name(), "TestGraph");
        assert!(graph.compiled);
    }
    
    #[test]
    fn test_topological_sort() {
        let graph = NetGraphBuilder::new()
            .add_node("a", Box::new(IdentityLayer::new()))
            .add_node("b", Box::new(IdentityLayer::new()))
            .add_node("c", Box::new(IdentityLayer::new()))
            .connect("a", "b")
            .connect("b", "c")
            .build()
            .unwrap();
        
        let execution_order = graph.execution_order();
        assert_eq!(execution_order, &["a", "b", "c"]);
    }
    
    #[test]
    fn test_cycle_detection() {
        let result = NetGraphBuilder::new()
            .add_node("a", Box::new(IdentityLayer::new()))
            .add_node("b", Box::new(IdentityLayer::new()))
            .connect("a", "b")
            .connect("b", "a")  // Creates a cycle
            .build();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cycles"));
    }
    
    #[test]
    fn test_input_output_identification() {
        let graph = NetGraphBuilder::new()
            .add_node("input", Box::new(IdentityLayer::new()))
            .add_node("middle", Box::new(LinearLayer::new(5)))
            .add_node("output", Box::new(IdentityLayer::new()))
            .connect("input", "middle")
            .connect("middle", "output")
            .build()
            .unwrap();
        
        assert_eq!(graph.input_nodes(), &["input"]);
        assert_eq!(graph.output_nodes(), &["output"]);
    }
    
    #[test]
    fn test_graph_forward_pass() {
        let mut graph = NetGraphBuilder::new()
            .add_node("input", Box::new(IdentityLayer::new()))
            .add_node("linear", Box::new(LinearLayer::new(2)))
            .add_node("output", Box::new(IdentityLayer::new()))
            .connect("input", "linear")
            .connect("linear", "output")
            .build()
            .unwrap();
        
        // Create input tensor
        let input_tensor = Tensor::from_values(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), input_tensor);
        
        // Forward pass
        let outputs = graph.forward(&inputs).unwrap();
        
        assert!(outputs.contains_key("output"));
        let output_tensor = &outputs["output"];
        assert_eq!(output_tensor.shape, vec![1, 2]);  // Linear layer outputs 2 features
    }
    
    #[test]
    fn test_invalid_node_reference() {
        let result = NetGraphBuilder::new()
            .add_node("a", Box::new(IdentityLayer::new()))
            .connect("a", "nonexistent")  // Reference to non-existent node
            .build();
        
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }
    
    #[test]
    fn test_graph_summary() {
        let graph = NetGraphBuilder::new()
            .add_node("input", Box::new(IdentityLayer::new()))
            .add_node("output", Box::new(LinearLayer::new(5)))
            .connect("input", "output")
            .with_name("SummaryTest".to_string())
            .build()
            .unwrap();
        
        let summary = graph.summary();
        assert!(summary.contains("SummaryTest"));
        assert!(summary.contains("Nodes: 2"));
        assert!(summary.contains("Connections: 1"));
        assert!(summary.contains("IdentityLayer"));
        assert!(summary.contains("LinearLayer[5]"));
    }
}