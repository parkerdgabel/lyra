//! Neural Network Layer Implementations
//!
//! This module provides all layer types available in the Wolfram Language ML framework,
//! including Linear, Convolution, Pooling, Recurrent, and Attention layers.

use crate::stdlib::autodiff::Dual;
use super::{MLResult, MLError};

/// Base trait for all neural network layers
pub trait Layer: std::fmt::Debug {
    /// Forward pass through the layer
    fn forward(&self, input: &Tensor) -> MLResult<Tensor>;
    
    /// Get the layer's trainable parameters
    fn parameters(&self) -> Vec<&Tensor>;
    
    /// Get mutable references to trainable parameters
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    
    /// Initialize layer parameters
    fn initialize(&mut self) -> MLResult<()>;
    
    /// Get layer name for debugging
    fn name(&self) -> &str;
    
    /// Get output shape given input shape
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>>;
    
    /// Number of trainable parameters
    fn parameter_count(&self) -> usize {
        self.parameters().iter().map(|p| p.size()).sum()
    }
    
    /// Clone the layer as a boxed trait object
    fn clone_boxed(&self) -> Box<dyn Layer>;
}

/// Tensor type for neural network computations
/// Wraps our autodiff system with shape information
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Dual number data for automatic differentiation
    pub data: Vec<Dual>,
    /// Shape of the tensor [batch_size, height, width, channels] etc.
    pub shape: Vec<usize>,
}

impl Tensor {
    /// Create a new tensor with given shape and values
    pub fn new(data: Vec<Dual>, shape: Vec<usize>) -> MLResult<Self> {
        let expected_size: usize = shape.iter().product();
        if data.len() != expected_size {
            return Err(MLError::ShapeMismatch {
                expected: vec![expected_size],
                actual: vec![data.len()],
            });
        }
        
        Ok(Self { data, shape })
    }
    
    /// Create tensor from f64 values (constants)
    pub fn from_values(values: Vec<f64>, shape: Vec<usize>) -> MLResult<Self> {
        let data = values.into_iter().map(Dual::constant).collect();
        Self::new(data, shape)
    }
    
    /// Create tensor with variables for gradients
    pub fn variables(values: Vec<f64>, shape: Vec<usize>) -> MLResult<Self> {
        let data = values.into_iter().map(Dual::variable).collect();
        Self::new(data, shape)
    }
    
    /// Create zero tensor with given shape
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data = vec![Dual::constant(0.0); size];
        Self { data, shape }
    }
    
    /// Create tensor with random normal values
    pub fn randn(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let data: Vec<Dual> = (0..size)
            .map(|_| {
                // Simple normal distribution approximation using Box-Muller
                let u1: f64 = rand::random::<f64>();
                let u2: f64 = rand::random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                Dual::variable(z * 0.1) // Scale for initialization
            })
            .collect();
        Self { data, shape }
    }
    
    /// Get tensor size (total number of elements)
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    /// Get batch size (first dimension)
    pub fn batch_size(&self) -> usize {
        self.shape.get(0).copied().unwrap_or(1)
    }
    
    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> MLResult<Self> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(MLError::ShapeMismatch {
                expected: vec![new_size],
                actual: vec![self.size()],
            });
        }
        
        Ok(Self {
            data: self.data.clone(),
            shape: new_shape,
        })
    }
    
    /// Matrix multiplication for 2D tensors
    pub fn matmul(&self, other: &Self) -> MLResult<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(MLError::InvalidLayer {
                reason: "Matrix multiplication requires 2D tensors".to_string(),
            });
        }
        
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        
        if k1 != k2 {
            return Err(MLError::ShapeMismatch {
                expected: vec![k1],
                actual: vec![k2],
            });
        }
        
        let mut result = vec![Dual::constant(0.0); m * n];
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = Dual::constant(0.0);
                for k in 0..k1 {
                    let a_val = self.data[i * k1 + k];
                    let b_val = other.data[k * n + j];
                    sum = sum + a_val * b_val;
                }
                result[i * n + j] = sum;
            }
        }
        
        Self::new(result, vec![m, n])
    }
    
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> MLResult<Self> {
        if self.shape != other.shape {
            return Err(MLError::ShapeMismatch {
                expected: self.shape.clone(),
                actual: other.shape.clone(),
            });
        }
        
        let result: Vec<Dual> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| *a + *b)
            .collect();
            
        Self::new(result, self.shape.clone())
    }
    
    /// Apply activation function element-wise
    pub fn relu(&self) -> Self {
        let result: Vec<Dual> = self.data.iter()
            .map(|&x| x.relu())
            .collect();
            
        Self {
            data: result,
            shape: self.shape.clone(),
        }
    }
    
    /// Apply sigmoid activation
    pub fn sigmoid(&self) -> Self {
        let result: Vec<Dual> = self.data.iter()
            .map(|&x| x.sigmoid())
            .collect();
            
        Self {
            data: result,
            shape: self.shape.clone(),
        }
    }
    
    /// Apply tanh activation
    pub fn tanh(&self) -> Self {
        let result: Vec<Dual> = self.data.iter()
            .map(|&x| x.tanh())
            .collect();
            
        Self {
            data: result,
            shape: self.shape.clone(),
        }
    }
}

/// LinearLayer: Trainable affine transformation (Dense/Fully-connected layer)
/// LinearLayer[n] creates a linear layer with n output units
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Output size
    pub output_size: usize,
    /// Input size (determined during first forward pass)
    pub input_size: Option<usize>,
    /// Weight matrix [input_size, output_size]
    pub weights: Option<Tensor>,
    /// Bias vector [output_size]
    pub bias: Option<Tensor>,
    /// Layer name for debugging
    pub layer_name: String,
}

impl LinearLayer {
    /// Create a new linear layer with specified output size
    pub fn new(output_size: usize) -> Self {
        Self {
            output_size,
            input_size: None,
            weights: None,
            bias: None,
            layer_name: format!("LinearLayer[{}]", output_size),
        }
    }
    
    /// Initialize weights using Xavier/Glorot initialization
    fn init_weights(&mut self, input_size: usize) -> MLResult<()> {
        self.input_size = Some(input_size);
        
        // Xavier initialization: scale = sqrt(6 / (fan_in + fan_out))
        let scale = (6.0 / (input_size + self.output_size) as f64).sqrt();
        
        // Initialize weights
        let weight_values: Vec<f64> = (0..input_size * self.output_size)
            .map(|_| {
                let u: f64 = rand::random();
                (u - 0.5) * 2.0 * scale
            })
            .collect();
        
        self.weights = Some(Tensor::variables(
            weight_values,
            vec![input_size, self.output_size],
        )?);
        
        // Initialize bias to zeros
        let bias_values = vec![0.0; self.output_size];
        self.bias = Some(Tensor::variables(bias_values, vec![self.output_size])?);
        
        Ok(())
    }
}

impl Layer for LinearLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Ensure input is 2D [batch_size, features]
        if input.shape.len() != 2 {
            return Err(MLError::InvalidLayer {
                reason: format!("LinearLayer expects 2D input, got {}D", input.shape.len()),
            });
        }
        
        let batch_size = input.shape[0];
        let input_features = input.shape[1];
        
        // Initialize weights if this is the first forward pass
        let mut layer = self.clone();
        if layer.weights.is_none() {
            layer.init_weights(input_features)?;
        }
        
        let weights = layer.weights.as_ref().unwrap();
        let bias = layer.bias.as_ref().unwrap();
        
        // Verify input size matches expected
        if let Some(expected_input_size) = self.input_size {
            if input_features != expected_input_size {
                return Err(MLError::ShapeMismatch {
                    expected: vec![expected_input_size],
                    actual: vec![input_features],
                });
            }
        }
        
        // Forward pass: output = input @ weights + bias
        let linear_output = input.matmul(weights)?;
        
        // Add bias (broadcast across batch dimension)
        let mut result_data = Vec::with_capacity(batch_size * self.output_size);
        
        for batch_idx in 0..batch_size {
            for out_idx in 0..self.output_size {
                let linear_val = linear_output.data[batch_idx * self.output_size + out_idx];
                let bias_val = bias.data[out_idx];
                result_data.push(linear_val + bias_val);
            }
        }
        
        Tensor::new(result_data, vec![batch_size, self.output_size])
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weights) = self.weights {
            params.push(weights);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weights) = self.weights {
            params.push(weights);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        if let Some(input_size) = self.input_size {
            self.init_weights(input_size)
        } else {
            // Can't initialize without knowing input size
            Ok(())
        }
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        if input_shape.len() != 2 {
            return Err(MLError::InvalidLayer {
                reason: format!("LinearLayer expects 2D input shape, got {}D", input_shape.len()),
            });
        }
        
        Ok(vec![input_shape[0], self.output_size])
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// ReLU Activation Layer
#[derive(Debug, Clone)]
pub struct ReLULayer {
    pub layer_name: String,
}

impl ReLULayer {
    pub fn new() -> Self {
        Self {
            layer_name: "ReLULayer".to_string(),
        }
    }
}

impl Layer for ReLULayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        Ok(input.relu())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new() // No trainable parameters
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        Ok(()) // Nothing to initialize
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        Ok(input_shape.to_vec()) // Same shape as input
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// Sigmoid Activation Layer
#[derive(Debug, Clone)]
pub struct SigmoidLayer {
    pub layer_name: String,
}

impl SigmoidLayer {
    pub fn new() -> Self {
        Self {
            layer_name: "SigmoidLayer".to_string(),
        }
    }
}

impl Layer for SigmoidLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        Ok(input.sigmoid())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        Ok(())
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        Ok(input_shape.to_vec())
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// Softmax Layer for probability distributions
#[derive(Debug, Clone)]
pub struct SoftmaxLayer {
    pub layer_name: String,
}

impl SoftmaxLayer {
    pub fn new() -> Self {
        Self {
            layer_name: "SoftmaxLayer".to_string(),
        }
    }
}

impl Layer for SoftmaxLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        if input.shape.len() != 2 {
            return Err(MLError::InvalidLayer {
                reason: "SoftmaxLayer expects 2D input [batch_size, features]".to_string(),
            });
        }
        
        let batch_size = input.shape[0];
        let features = input.shape[1];
        let mut result_data = Vec::with_capacity(input.data.len());
        
        // Apply softmax per batch item
        for batch_idx in 0..batch_size {
            let start_idx = batch_idx * features;
            let end_idx = start_idx + features;
            let batch_slice = &input.data[start_idx..end_idx];
            
            // Find max for numerical stability
            let max_val = batch_slice.iter()
                .map(|x| x.value())
                .fold(f64::NEG_INFINITY, f64::max);
            
            // Compute exp(x - max) for each element
            let exp_values: Vec<Dual> = batch_slice.iter()
                .map(|&x| (x - Dual::constant(max_val)).exp())
                .collect();
            
            // Compute sum of exponentials
            let sum_exp = exp_values.iter()
                .fold(Dual::constant(0.0), |acc, &x| acc + x);
            
            // Normalize by sum
            for exp_val in exp_values {
                result_data.push(exp_val / sum_exp);
            }
        }
        
        Tensor::new(result_data, input.shape.clone())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        Ok(())
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        Ok(input_shape.to_vec())
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

