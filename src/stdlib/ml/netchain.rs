//! NetChain Implementation
//!
//! NetChain represents a sequential composition of neural network layers,
//! modeling the Wolfram Language NetChain functionality.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::{Layer, Tensor};
use std::fmt;

/// NetChain: Sequential composition of neural network layers
/// 
/// NetChain[{layer1, layer2, ...}] creates a neural network where the output 
/// of layer_i is connected to the input of layer_{i+1}.
/// 
/// # Examples
/// ```rust
/// use lyra::stdlib::ml::{NetChain, LinearLayer, ReLULayer};
/// 
/// let chain = NetChain::new(vec![
///     Box::new(LinearLayer::new(128)),
///     Box::new(ReLULayer::new()),
///     Box::new(LinearLayer::new(10)),
/// ]);
/// ```
#[derive(Debug)]
pub struct NetChain {
    /// Sequential list of layers
    layers: Vec<Box<dyn Layer>>,
    /// Network name for debugging
    name: String,
    /// Whether the network has been initialized
    initialized: bool,
    /// Input shape (set after first forward pass)
    input_shape: Option<Vec<usize>>,
    /// Output shape (computed from layer composition)
    output_shape: Option<Vec<usize>>,
    /// Intermediate activations for backpropagation
    activations: Vec<Tensor>,
    /// Whether to store activations (for training mode)
    training_mode: bool,
}

impl NetChain {
    /// Create a new NetChain from a vector of layers
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        let layer_names: Vec<&str> = layers.iter().map(|l| l.name()).collect();
        let name = format!("NetChain[{}]", layer_names.join(", "));
        
        Self {
            layers,
            name,
            initialized: false,
            input_shape: None,
            output_shape: None,
            activations: Vec::new(),
            training_mode: true,  // Default to training mode
        }
    }
    
    /// Create NetChain with a name
    pub fn with_name(layers: Vec<Box<dyn Layer>>, name: String) -> Self {
        Self {
            layers,
            name,
            initialized: false,
            input_shape: None,
            output_shape: None,
            activations: Vec::new(),
            training_mode: true,  // Default to training mode
        }
    }
    
    /// Get the number of layers in the chain
    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }
    
    /// Get layer at specific index
    pub fn get_layer(&self, index: usize) -> Option<&Box<dyn Layer>> {
        self.layers.get(index)
    }
    
    /// Get mutable layer at specific index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut Box<dyn Layer>> {
        self.layers.get_mut(index)
    }
    
    /// Add a layer to the end of the chain
    pub fn append_layer(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
        self.initialized = false; // Need to reinitialize
        self.output_shape = None;
        self.activations.clear(); // Clear stored activations
    }
    
    /// Insert a layer at a specific position
    pub fn insert_layer(&mut self, index: usize, layer: Box<dyn Layer>) -> MLResult<()> {
        if index > self.layers.len() {
            return Err(MLError::NetworkError {
                reason: format!("Insert index {} out of bounds for {} layers", index, self.layers.len()),
            });
        }
        
        self.layers.insert(index, layer);
        self.initialized = false;
        self.output_shape = None;
        self.activations.clear(); // Clear stored activations
        Ok(())
    }
    
    /// Remove a layer at a specific position
    pub fn remove_layer(&mut self, index: usize) -> MLResult<Box<dyn Layer>> {
        if index >= self.layers.len() {
            return Err(MLError::NetworkError {
                reason: format!("Remove index {} out of bounds for {} layers", index, self.layers.len()),
            });
        }
        
        self.initialized = false;
        self.output_shape = None;
        self.activations.clear(); // Clear stored activations
        Ok(self.layers.remove(index))
    }
    
    /// Forward pass through the entire chain
    pub fn forward(&mut self, input: &Tensor) -> MLResult<Tensor> {
        if self.layers.is_empty() {
            return Err(MLError::NetworkError {
                reason: "Cannot perform forward pass on empty NetChain".to_string(),
            });
        }
        
        // Initialize if this is the first forward pass
        if !self.initialized {
            self.initialize_chain(input)?;
        }
        
        // Validate input shape
        if let Some(ref expected_shape) = self.input_shape {
            if input.shape != *expected_shape {
                return Err(MLError::ShapeMismatch {
                    expected: expected_shape.clone(),
                    actual: input.shape.clone(),
                });
            }
        }
        
        // Clear previous activations if in training mode
        if self.training_mode {
            self.activations.clear();
            self.activations.push(input.clone()); // Store input as first activation
        }
        
        // Sequential forward pass through all layers
        let mut current_output = input.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            match layer.forward(&current_output) {
                Ok(output) => {
                    current_output = output;
                    // Store activation for backpropagation if in training mode
                    if self.training_mode {
                        self.activations.push(current_output.clone());
                    }
                },
                Err(e) => {
                    return Err(MLError::NetworkError {
                        reason: format!("Forward pass failed at layer {} ({}): {}", i, layer.name(), e),
                    });
                }
            }
        }
        
        Ok(current_output)
    }
    
    /// Initialize the entire chain based on input shape
    fn initialize_chain(&mut self, input: &Tensor) -> MLResult<()> {
        self.input_shape = Some(input.shape.clone());
        
        // Compute shapes through the chain and initialize layers
        let mut current_shape = input.shape.clone();
        
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Initialize this layer
            layer.initialize().map_err(|e| MLError::NetworkError {
                reason: format!("Failed to initialize layer {} ({}): {}", i, layer.name(), e),
            })?;
            
            // Compute output shape for next layer
            current_shape = layer.output_shape(&current_shape).map_err(|e| MLError::NetworkError {
                reason: format!("Failed to compute output shape for layer {} ({}): {}", i, layer.name(), e),
            })?;
        }
        
        self.output_shape = Some(current_shape);
        self.initialized = true;
        Ok(())
    }
    
    /// Get all trainable parameters from all layers
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
    
    /// Get all mutable trainable parameters from all layers
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }
    
    /// Get total number of trainable parameters
    pub fn parameter_count(&self) -> usize {
        self.layers.iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }
    
    /// Get network summary as a string
    pub fn summary(&self) -> String {
        let mut summary = format!("NetChain Summary: {}\n", self.name);
        summary.push_str("=" .repeat(50).as_str());
        summary.push('\n');
        
        if let Some(ref input_shape) = self.input_shape {
            summary.push_str(&format!("Input Shape: {:?}\n", input_shape));
        } else {
            summary.push_str("Input Shape: Not determined (call forward first)\n");
        }
        
        summary.push_str("\nLayers:\n");
        summary.push_str("-".repeat(50).as_str());
        summary.push('\n');
        
        let mut current_shape = self.input_shape.clone();
        
        for (i, layer) in self.layers.iter().enumerate() {
            let param_count = layer.parameter_count();
            let output_shape = if let Some(ref shape) = current_shape {
                match layer.output_shape(shape) {
                    Ok(out_shape) => {
                        current_shape = Some(out_shape.clone());
                        format!("{:?}", out_shape)
                    }
                    Err(_) => "Unknown".to_string(),
                }
            } else {
                "Unknown".to_string()
            };
            
            summary.push_str(&format!(
                "{:2}: {:20} | Output: {:15} | Params: {:8}\n",
                i + 1,
                layer.name(),
                output_shape,
                param_count
            ));
        }
        
        summary.push_str("-".repeat(50).as_str());
        summary.push('\n');
        summary.push_str(&format!("Total Parameters: {}\n", self.parameter_count()));
        
        if let Some(ref output_shape) = self.output_shape {
            summary.push_str(&format!("Output Shape: {:?}\n", output_shape));
        } else {
            summary.push_str("Output Shape: Not determined\n");
        }
        
        summary
    }
    
    /// Get the expected input shape
    pub fn input_shape(&self) -> Option<&Vec<usize>> {
        self.input_shape.as_ref()
    }
    
    /// Get the output shape
    pub fn output_shape(&self) -> Option<&Vec<usize>> {
        self.output_shape.as_ref()
    }
    
    /// Check if the network is initialized
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
    
    /// Get network name
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Set training mode (enables activation storage for backpropagation)
    pub fn train(&mut self) {
        self.training_mode = true;
        self.activations.clear();
    }
    
    /// Set evaluation mode (disables activation storage for faster inference)
    pub fn eval(&mut self) {
        self.training_mode = false;
        self.activations.clear();
    }
    
    /// Check if the network is in training mode
    pub fn is_training(&self) -> bool {
        self.training_mode
    }
    
    /// Get stored activations (for debugging or advanced backpropagation)
    pub fn get_activations(&self) -> &[Tensor] {
        &self.activations
    }
    
    /// Perform backward pass (compute gradients with respect to parameters)
    /// This is automatically called during training via the autodiff system
    /// The gradients are stored in the Dual numbers of the parameters
    pub fn backward(&mut self, output_gradient: &Tensor) -> MLResult<Tensor> {
        if !self.training_mode {
            return Err(MLError::NetworkError {
                reason: "Cannot perform backward pass in evaluation mode. Call .train() first.".to_string(),
            });
        }
        
        if self.activations.is_empty() {
            return Err(MLError::NetworkError {
                reason: "No activations stored. Call forward() first.".to_string(),
            });
        }
        
        // For now, return the output gradient as the input gradient
        // This is a simplified implementation - in a full system we would
        // implement proper layer-wise backward passes here
        Ok(output_gradient.clone())
    }
    
    /// Clear stored gradients (reset for new training step)
    pub fn zero_grad(&mut self) {
        // Reset gradients in all parameters to zero
        for param in self.parameters_mut() {
            for dual in &mut param.data {
                *dual = crate::stdlib::autodiff::Dual::variable(dual.value());
            }
        }
    }
    
    /// Clone the network with a new name
    pub fn clone_with_name(&self, new_name: String) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| layer.clone_boxed()).collect(),
            name: new_name,
            initialized: self.initialized,
            input_shape: self.input_shape.clone(),
            output_shape: self.output_shape.clone(),
            activations: Vec::new(), // Don't clone activations
            training_mode: self.training_mode,
        }
    }
}

impl Clone for NetChain {
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|layer| layer.clone_boxed()).collect(),
            name: self.name.clone(),
            initialized: self.initialized,
            input_shape: self.input_shape.clone(),
            output_shape: self.output_shape.clone(),
            activations: Vec::new(), // Don't clone activations
            training_mode: self.training_mode,
        }
    }
}

impl fmt::Display for NetChain {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary())
    }
}

/// Convenient builder for NetChain construction
pub struct NetChainBuilder {
    layers: Vec<Box<dyn Layer>>,
    name: Option<String>,
}

impl NetChainBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            name: None,
        }
    }
    
    /// Add a layer to the chain
    pub fn add_layer(mut self, layer: Box<dyn Layer>) -> Self {
        self.layers.push(layer);
        self
    }
    
    /// Set the network name
    pub fn with_name(mut self, name: String) -> Self {
        self.name = Some(name);
        self
    }
    
    /// Build the NetChain
    pub fn build(self) -> NetChain {
        match self.name {
            Some(name) => NetChain::with_name(self.layers, name),
            None => NetChain::new(self.layers),
        }
    }
}

impl Default for NetChainBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::ml::layers::{LinearLayer, ReLULayer, SigmoidLayer};
    
    #[test]
    fn test_netchain_creation() {
        let chain = NetChain::new(vec![
            Box::new(LinearLayer::new(128)),
            Box::new(ReLULayer::new()),
            Box::new(LinearLayer::new(10)),
        ]);
        
        assert_eq!(chain.layer_count(), 3);
        assert!(!chain.is_initialized());
        assert!(chain.input_shape().is_none());
        assert!(chain.output_shape().is_none());
    }
    
    #[test]
    fn test_netchain_builder() {
        let chain = NetChainBuilder::new()
            .add_layer(Box::new(LinearLayer::new(64)))
            .add_layer(Box::new(ReLULayer::new()))
            .add_layer(Box::new(LinearLayer::new(32)))
            .add_layer(Box::new(SigmoidLayer::new()))
            .with_name("TestNetwork".to_string())
            .build();
        
        assert_eq!(chain.layer_count(), 4);
        assert_eq!(chain.name(), "TestNetwork");
    }
    
    #[test]
    fn test_netchain_forward_pass() {
        let mut chain = NetChain::new(vec![
            Box::new(LinearLayer::new(5)),
            Box::new(ReLULayer::new()),
            Box::new(LinearLayer::new(2)),
        ]);
        
        // Create input tensor [batch_size=2, features=3]
        let input = Tensor::from_values(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        ).unwrap();
        
        let output = chain.forward(&input).unwrap();
        
        // Output should be [batch_size=2, features=2]
        assert_eq!(output.shape, vec![2, 2]);
        assert!(chain.is_initialized());
        assert_eq!(chain.input_shape(), Some(&vec![2, 3]));
        assert_eq!(chain.output_shape(), Some(&vec![2, 2]));
    }
    
    #[test]
    fn test_netchain_parameter_count() {
        let chain = NetChain::new(vec![
            Box::new(LinearLayer::new(10)), // (input + 1) * 10 params, input determined at runtime
            Box::new(ReLULayer::new()),     // 0 params
            Box::new(LinearLayer::new(5)),  // (10 + 1) * 5 = 55 params
        ]);
        
        // Before initialization, parameter count is 0
        assert_eq!(chain.parameter_count(), 0);
    }
    
    #[test]
    fn test_empty_netchain_error() {
        let mut chain = NetChain::new(vec![]);
        let input = Tensor::from_values(vec![1.0, 2.0], vec![1, 2]).unwrap();
        
        assert!(chain.forward(&input).is_err());
    }
}