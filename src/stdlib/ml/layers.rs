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

/// Identity Layer - passes input through unchanged (for input/output nodes)
#[derive(Debug, Clone)]
pub struct IdentityLayer {
    pub layer_name: String,
}

impl IdentityLayer {
    pub fn new() -> Self {
        Self {
            layer_name: "IdentityLayer".to_string(),
        }
    }
}

impl Layer for IdentityLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        Ok(input.clone())
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

/// Addition Layer - adds two tensors element-wise (for residual connections)
#[derive(Debug, Clone)]
pub struct AddLayer {
    pub layer_name: String,
}

impl AddLayer {
    pub fn new() -> Self {
        Self {
            layer_name: "AddLayer".to_string(),
        }
    }
}

impl Layer for AddLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // For now, just return the input (multi-input support needed)
        Ok(input.clone())
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

/// ConvolutionLayer: 2D convolutional layer for computer vision
/// ConvolutionLayer[n, {h, w}] creates a 2D conv layer with n output channels and h×w kernels
#[derive(Debug, Clone)]
pub struct ConvolutionLayer {
    /// Number of output channels/filters
    pub output_channels: usize,
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width]
    pub stride: [usize; 2],
    /// Padding [height, width]
    pub padding: [usize; 2],
    /// Input channels (determined during first forward pass)
    pub input_channels: Option<usize>,
    /// Weight tensor [output_channels, input_channels, kernel_h, kernel_w]
    pub weights: Option<Tensor>,
    /// Bias vector [output_channels]
    pub bias: Option<Tensor>,
    /// Layer name for debugging
    pub layer_name: String,
}

impl ConvolutionLayer {
    /// Create a new 2D convolution layer
    pub fn new(output_channels: usize, kernel_size: [usize; 2]) -> Self {
        Self {
            output_channels,
            kernel_size,
            stride: [1, 1],
            padding: [0, 0],
            input_channels: None,
            weights: None,
            bias: None,
            layer_name: format!("ConvolutionLayer[{}, {:?}]", output_channels, kernel_size),
        }
    }
    
    /// Set stride
    pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        self.layer_name = format!("ConvolutionLayer[{}, {:?}, stride: {:?}]", 
                                  self.output_channels, self.kernel_size, stride);
        self
    }
    
    /// Set padding  
    pub fn with_padding(mut self, padding: [usize; 2]) -> Self {
        self.padding = padding;
        self.layer_name = format!("ConvolutionLayer[{}, {:?}, padding: {:?}]", 
                                  self.output_channels, self.kernel_size, padding);
        self
    }
    
    /// Initialize weights using He initialization (for ReLU networks)
    fn init_weights(&mut self, input_channels: usize) -> MLResult<()> {
        self.input_channels = Some(input_channels);
        
        let [kh, kw] = self.kernel_size;
        
        // He initialization: std = sqrt(2 / fan_in)
        let fan_in = input_channels * kh * kw;
        let std = (2.0 / fan_in as f64).sqrt();
        
        // Initialize weights [out_channels, in_channels, kh, kw]
        let weight_count = self.output_channels * input_channels * kh * kw;
        let weight_values: Vec<f64> = (0..weight_count)
            .map(|_| {
                // Box-Muller for normal distribution
                let u1: f64 = rand::random::<f64>();
                let u2: f64 = rand::random::<f64>();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * std
            })
            .collect();
        
        self.weights = Some(Tensor::variables(
            weight_values,
            vec![self.output_channels, input_channels, kh, kw],
        )?);
        
        // Initialize bias to zeros
        let bias_values = vec![0.0; self.output_channels];
        self.bias = Some(Tensor::variables(bias_values, vec![self.output_channels])?);
        
        Ok(())
    }
    
    /// Compute output spatial dimensions
    fn compute_output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let [kh, kw] = self.kernel_size;
        let [sh, sw] = self.stride;
        let [ph, pw] = self.padding;
        
        let out_h = (input_h + 2 * ph - kh) / sh + 1;
        let out_w = (input_w + 2 * pw - kw) / sw + 1;
        
        (out_h, out_w)
    }
}

impl Layer for ConvolutionLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Expect input shape: [batch_size, channels, height, width]
        if input.shape.len() != 4 {
            return Err(MLError::InvalidLayer {
                reason: format!("ConvolutionLayer expects 4D input [N,C,H,W], got {}D", input.shape.len()),
            });
        }
        
        let [batch_size, input_channels, input_h, input_w] = [
            input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        ];
        
        // Initialize weights if this is the first forward pass
        let mut layer = self.clone();
        if layer.weights.is_none() {
            layer.init_weights(input_channels)?;
        }
        
        let weights = layer.weights.as_ref().unwrap();
        let bias = layer.bias.as_ref().unwrap();
        let [kh, kw] = self.kernel_size;
        let [sh, sw] = self.stride;
        let [ph, pw] = self.padding;
        
        // Compute output dimensions
        let (out_h, out_w) = self.compute_output_size(input_h, input_w);
        
        // For simplicity, implement basic convolution without padding for now
        if ph != 0 || pw != 0 {
            return Err(MLError::InvalidLayer {
                reason: "Padding not yet implemented for ConvolutionLayer".to_string(),
            });
        }
        
        // Validate kernel fits in input
        if kh > input_h || kw > input_w {
            return Err(MLError::InvalidLayer {
                reason: format!("Kernel size {:?} too large for input {}x{}", 
                               self.kernel_size, input_h, input_w),
            });
        }
        
        // Perform convolution
        let mut output_data = Vec::with_capacity(batch_size * self.output_channels * out_h * out_w);
        
        for b in 0..batch_size {
            for out_c in 0..self.output_channels {
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let mut conv_sum = Dual::constant(0.0);
                        
                        // Convolution operation
                        for in_c in 0..input_channels {
                            for ky in 0..kh {
                                for kx in 0..kw {
                                    let input_y = out_y * sh + ky;
                                    let input_x = out_x * sw + kx;
                                    
                                    if input_y < input_h && input_x < input_w {
                                        let input_idx = b * (input_channels * input_h * input_w) +
                                                       in_c * (input_h * input_w) +
                                                       input_y * input_w + input_x;
                                        
                                        let weight_idx = out_c * (input_channels * kh * kw) +
                                                        in_c * (kh * kw) +
                                                        ky * kw + kx;
                                        
                                        conv_sum = conv_sum + input.data[input_idx] * weights.data[weight_idx];
                                    }
                                }
                            }
                        }
                        
                        // Add bias
                        conv_sum = conv_sum + bias.data[out_c];
                        output_data.push(conv_sum);
                    }
                }
            }
        }
        
        Tensor::new(output_data, vec![batch_size, self.output_channels, out_h, out_w])
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
        if let Some(input_channels) = self.input_channels {
            self.init_weights(input_channels)
        } else {
            Ok(())
        }
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(MLError::InvalidLayer {
                reason: format!("ConvolutionLayer expects 4D input shape [N,C,H,W], got {}D", 
                               input_shape.len()),
            });
        }
        
        let [batch_size, _input_channels, input_h, input_w] = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        ];
        
        let (out_h, out_w) = self.compute_output_size(input_h, input_w);
        
        Ok(vec![batch_size, self.output_channels, out_h, out_w])
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// PoolingLayer: 2D pooling layer for downsampling
/// PoolingLayer[{h, w}] creates a pooling layer with h×w kernels
#[derive(Debug, Clone)]
pub struct PoolingLayer {
    /// Kernel size [height, width]
    pub kernel_size: [usize; 2],
    /// Stride [height, width] 
    pub stride: [usize; 2],
    /// Pooling function (Max, Mean)
    pub pool_function: PoolFunction,
    /// Layer name for debugging
    pub layer_name: String,
}

/// Pooling function type
#[derive(Debug, Clone, PartialEq)]
pub enum PoolFunction {
    Max,
    Mean,
}

impl PoolingLayer {
    /// Create a new max pooling layer
    pub fn max_pool(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            stride: kernel_size, // Default stride equals kernel size (non-overlapping)
            pool_function: PoolFunction::Max,
            layer_name: format!("MaxPoolingLayer[{:?}]", kernel_size),
        }
    }
    
    /// Create a new average pooling layer
    pub fn avg_pool(kernel_size: [usize; 2]) -> Self {
        Self {
            kernel_size,
            stride: kernel_size, // Default stride equals kernel size (non-overlapping)
            pool_function: PoolFunction::Mean,
            layer_name: format!("AvgPoolingLayer[{:?}]", kernel_size),
        }
    }
    
    /// Set custom stride
    pub fn with_stride(mut self, stride: [usize; 2]) -> Self {
        self.stride = stride;
        let function_name = match self.pool_function {
            PoolFunction::Max => "MaxPoolingLayer",
            PoolFunction::Mean => "AvgPoolingLayer",
        };
        self.layer_name = format!("{}[{:?}, stride: {:?}]", 
                                  function_name, self.kernel_size, stride);
        self
    }
    
    /// Compute output spatial dimensions
    fn compute_output_size(&self, input_h: usize, input_w: usize) -> (usize, usize) {
        let [kh, kw] = self.kernel_size;
        let [sh, sw] = self.stride;
        
        let out_h = (input_h - kh) / sh + 1;
        let out_w = (input_w - kw) / sw + 1;
        
        (out_h, out_w)
    }
}

impl Layer for PoolingLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Expect input shape: [batch_size, channels, height, width]
        if input.shape.len() != 4 {
            return Err(MLError::InvalidLayer {
                reason: format!("PoolingLayer expects 4D input [N,C,H,W], got {}D", input.shape.len()),
            });
        }
        
        let [batch_size, channels, input_h, input_w] = [
            input.shape[0], input.shape[1], input.shape[2], input.shape[3]
        ];
        
        let [kh, kw] = self.kernel_size;
        let [sh, sw] = self.stride;
        
        // Validate kernel fits in input
        if kh > input_h || kw > input_w {
            return Err(MLError::InvalidLayer {
                reason: format!("Kernel size {:?} too large for input {}x{}", 
                               self.kernel_size, input_h, input_w),
            });
        }
        
        // Compute output dimensions
        let (out_h, out_w) = self.compute_output_size(input_h, input_w);
        
        // Perform pooling
        let mut output_data = Vec::with_capacity(batch_size * channels * out_h * out_w);
        
        for b in 0..batch_size {
            for c in 0..channels {
                for out_y in 0..out_h {
                    for out_x in 0..out_w {
                        let start_y = out_y * sh;
                        let start_x = out_x * sw;
                        
                        let pool_result = match self.pool_function {
                            PoolFunction::Max => {
                                let mut max_val = Dual::constant(f64::NEG_INFINITY);
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let input_y = start_y + ky;
                                        let input_x = start_x + kx;
                                        
                                        if input_y < input_h && input_x < input_w {
                                            let input_idx = b * (channels * input_h * input_w) +
                                                           c * (input_h * input_w) +
                                                           input_y * input_w + input_x;
                                            
                                            let val = input.data[input_idx];
                                            if val.value() > max_val.value() {
                                                max_val = val;
                                            }
                                        }
                                    }
                                }
                                max_val
                            },
                            PoolFunction::Mean => {
                                let mut sum_val = Dual::constant(0.0);
                                let mut count = 0;
                                
                                for ky in 0..kh {
                                    for kx in 0..kw {
                                        let input_y = start_y + ky;
                                        let input_x = start_x + kx;
                                        
                                        if input_y < input_h && input_x < input_w {
                                            let input_idx = b * (channels * input_h * input_w) +
                                                           c * (input_h * input_w) +
                                                           input_y * input_w + input_x;
                                            
                                            sum_val = sum_val + input.data[input_idx];
                                            count += 1;
                                        }
                                    }
                                }
                                
                                if count > 0 {
                                    sum_val / Dual::constant(count as f64)
                                } else {
                                    Dual::constant(0.0)
                                }
                            },
                        };
                        
                        output_data.push(pool_result);
                    }
                }
            }
        }
        
        Tensor::new(output_data, vec![batch_size, channels, out_h, out_w])
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
        if input_shape.len() != 4 {
            return Err(MLError::InvalidLayer {
                reason: format!("PoolingLayer expects 4D input shape [N,C,H,W], got {}D", 
                               input_shape.len()),
            });
        }
        
        let [batch_size, channels, input_h, input_w] = [
            input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        ];
        
        let (out_h, out_w) = self.compute_output_size(input_h, input_w);
        
        Ok(vec![batch_size, channels, out_h, out_w])
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod conv_tests {
    use super::*;
    
    #[test]
    fn test_conv2d_creation() {
        let conv = ConvolutionLayer::new(32, [3, 3]);
        assert_eq!(conv.output_channels, 32);
        assert_eq!(conv.kernel_size, [3, 3]);
        assert_eq!(conv.stride, [1, 1]);
        assert_eq!(conv.padding, [0, 0]);
        assert!(conv.weights.is_none());
        assert!(conv.bias.is_none());
    }
    
    #[test]
    fn test_conv2d_with_stride() {
        let conv = ConvolutionLayer::new(16, [5, 5]).with_stride([2, 2]);
        assert_eq!(conv.stride, [2, 2]);
        assert!(conv.layer_name.contains("stride"));
    }
    
    #[test]
    fn test_conv2d_output_shape() {
        let conv = ConvolutionLayer::new(64, [3, 3]);
        
        // Input shape: [batch=2, channels=3, height=8, width=8]
        let input_shape = vec![2, 3, 8, 8];
        let output_shape = conv.output_shape(&input_shape).unwrap();
        
        // With 3x3 kernel and stride 1, output should be 6x6
        assert_eq!(output_shape, vec![2, 64, 6, 6]);
    }
    
    #[test] 
    fn test_conv2d_forward_pass() {
        let conv = ConvolutionLayer::new(2, [2, 2]);
        
        // Create input tensor [1, 1, 4, 4]
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_values(input_data, vec![1, 1, 4, 4]).unwrap();
        
        let output = conv.forward(&input).unwrap();
        
        // Output should be [1, 2, 3, 3] (2 output channels, 3x3 spatial)
        assert_eq!(output.shape, vec![1, 2, 3, 3]);
    }
    
    #[test]
    fn test_conv2d_parameter_count() {
        let mut conv = ConvolutionLayer::new(8, [3, 3]);
        
        // Simulate initialization with 3 input channels
        conv.init_weights(3).unwrap();
        
        // Parameters: weights [8, 3, 3, 3] + bias [8]
        // = 8 * 3 * 3 * 3 + 8 = 216 + 8 = 224
        assert_eq!(conv.parameter_count(), 224);
    }
    
    #[test]
    fn test_max_pooling_creation() {
        let pool = PoolingLayer::max_pool([2, 2]);
        assert_eq!(pool.kernel_size, [2, 2]);
        assert_eq!(pool.stride, [2, 2]); // Default stride equals kernel size
        assert_eq!(pool.pool_function, PoolFunction::Max);
        assert!(pool.layer_name.contains("MaxPoolingLayer"));
    }
    
    #[test]
    fn test_avg_pooling_creation() {
        let pool = PoolingLayer::avg_pool([3, 3]);
        assert_eq!(pool.pool_function, PoolFunction::Mean);
        assert!(pool.layer_name.contains("AvgPoolingLayer"));
    }
    
    #[test]
    fn test_pooling_output_shape() {
        let pool = PoolingLayer::max_pool([2, 2]);
        
        // Input shape: [batch=1, channels=3, height=8, width=8]
        let input_shape = vec![1, 3, 8, 8];
        let output_shape = pool.output_shape(&input_shape).unwrap();
        
        // With 2x2 kernel and stride 2, output should be 4x4
        assert_eq!(output_shape, vec![1, 3, 4, 4]);
    }
    
    #[test]
    fn test_max_pooling_forward_pass() {
        let pool = PoolingLayer::max_pool([2, 2]);
        
        // Create input tensor [1, 1, 4, 4] with known values
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_values(input_data, vec![1, 1, 4, 4]).unwrap();
        
        let output = pool.forward(&input).unwrap();
        
        // Output should be [1, 1, 2, 2] with max values from each 2x2 region
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        
        // Expected max values: [6, 8, 14, 16]
        assert_eq!(output.data[0].value(), 6.0);
        assert_eq!(output.data[1].value(), 8.0);
        assert_eq!(output.data[2].value(), 14.0);
        assert_eq!(output.data[3].value(), 16.0);
    }
    
    #[test]
    fn test_avg_pooling_forward_pass() {
        let pool = PoolingLayer::avg_pool([2, 2]);
        
        // Create input tensor [1, 1, 4, 4] with known values
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = Tensor::from_values(input_data, vec![1, 1, 4, 4]).unwrap();
        
        let output = pool.forward(&input).unwrap();
        
        // Output should be [1, 1, 2, 2] with average values from each 2x2 region
        assert_eq!(output.shape, vec![1, 1, 2, 2]);
        
        // Expected avg values: [3.5, 5.5, 11.5, 13.5]
        assert_eq!(output.data[0].value(), 3.5);
        assert_eq!(output.data[1].value(), 5.5);
        assert_eq!(output.data[2].value(), 11.5);
        assert_eq!(output.data[3].value(), 13.5);
    }
    
    #[test]
    fn test_pooling_with_custom_stride() {
        let pool = PoolingLayer::max_pool([2, 2]).with_stride([1, 1]);
        
        // Input shape: [batch=1, channels=1, height=4, width=4]
        let input_shape = vec![1, 1, 4, 4];
        let output_shape = pool.output_shape(&input_shape).unwrap();
        
        // With 2x2 kernel and stride 1, output should be 3x3
        assert_eq!(output_shape, vec![1, 1, 3, 3]);
        assert!(pool.layer_name.contains("stride"));
    }
    
    #[test]
    fn test_conv_pooling_integration() {
        // Test ConvolutionLayer followed by PoolingLayer
        let conv = ConvolutionLayer::new(4, [3, 3]);
        let pool = PoolingLayer::max_pool([2, 2]);
        
        // Input: [1, 2, 6, 6]
        let input_data: Vec<f64> = (0..72).map(|i| i as f64).collect();
        let input = Tensor::from_values(input_data, vec![1, 2, 6, 6]).unwrap();
        
        // Conv: [1, 2, 6, 6] -> [1, 4, 4, 4]
        let conv_output = conv.forward(&input).unwrap();
        assert_eq!(conv_output.shape, vec![1, 4, 4, 4]);
        
        // Pool: [1, 4, 4, 4] -> [1, 4, 2, 2]
        let pool_output = pool.forward(&conv_output).unwrap();
        assert_eq!(pool_output.shape, vec![1, 4, 2, 2]);
    }
}

#[cfg(test)]
mod regularization_tests {
    use super::*;
    
    #[test]
    fn test_dropout_creation() {
        let dropout = DropoutLayer::new();
        assert_eq!(dropout.drop_probability, 0.5);
        assert!(dropout.training);
        assert!(dropout.layer_name.contains("DropoutLayer"));
        
        let dropout_custom = DropoutLayer::with_probability(0.3);
        assert_eq!(dropout_custom.drop_probability, 0.3);
    }
    
    #[test]
    fn test_dropout_training_mode() {
        let dropout = DropoutLayer::with_probability(0.5);
        assert!(dropout.training);
        
        let dropout_eval = dropout.eval();
        assert!(!dropout_eval.training);
        
        let dropout_train = dropout_eval.train();
        assert!(dropout_train.training);
    }
    
    #[test]
    fn test_dropout_inference_mode() {
        let dropout = DropoutLayer::with_probability(0.5).eval();
        
        // Create test input
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_values(input_data.clone(), vec![2, 2]).unwrap();
        
        // In inference mode, dropout should pass through unchanged
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output.shape, vec![2, 2]);
        
        // Values should be unchanged
        for (i, &expected) in input_data.iter().enumerate() {
            assert_eq!(output.data[i].value(), expected);
        }
    }
    
    #[test]
    fn test_dropout_zero_probability() {
        let dropout = DropoutLayer::with_probability(0.0);
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_values(input_data.clone(), vec![2, 2]).unwrap();
        
        // With zero dropout, should pass through unchanged even in training
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output.shape, vec![2, 2]);
        
        for (i, &expected) in input_data.iter().enumerate() {
            assert_eq!(output.data[i].value(), expected);
        }
    }
    
    #[test]
    fn test_dropout_full_probability() {
        let dropout = DropoutLayer::with_probability(1.0);
        
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_values(input_data, vec![2, 2]).unwrap();
        
        // With full dropout, should output all zeros
        let output = dropout.forward(&input).unwrap();
        assert_eq!(output.shape, vec![2, 2]);
        
        for val in &output.data {
            assert_eq!(val.value(), 0.0);
        }
    }
    
    #[test]
    fn test_dropout_no_parameters() {
        let dropout = DropoutLayer::new();
        assert_eq!(dropout.parameters().len(), 0);
        assert_eq!(dropout.parameter_count(), 0);
    }
    
    #[test]
    fn test_dropout_output_shape() {
        let dropout = DropoutLayer::new();
        
        let shape_2d = vec![32, 64];
        assert_eq!(dropout.output_shape(&shape_2d).unwrap(), shape_2d);
        
        let shape_4d = vec![8, 32, 28, 28];
        assert_eq!(dropout.output_shape(&shape_4d).unwrap(), shape_4d);
    }
    
    #[test]
    fn test_batch_norm_creation() {
        let batch_norm = BatchNormalizationLayer::new();
        assert_eq!(batch_norm.epsilon, 1e-5);
        assert_eq!(batch_norm.momentum, 0.1);
        assert!(batch_norm.training);
        assert!(batch_norm.weight.is_none());
        assert!(batch_norm.bias.is_none());
    }
    
    #[test]
    fn test_batch_norm_configuration() {
        let batch_norm = BatchNormalizationLayer::new()
            .with_epsilon(1e-6)
            .with_momentum(0.05)
            .eval();
        
        assert_eq!(batch_norm.epsilon, 1e-6);
        assert_eq!(batch_norm.momentum, 0.05);
        assert!(!batch_norm.training);
    }
    
    #[test]
    fn test_batch_norm_2d_forward() {
        let mut batch_norm = BatchNormalizationLayer::new();
        
        // Initialize parameters for 3 features
        batch_norm.init_parameters(3).unwrap();
        
        // Input: [batch=2, features=3]
        let input_data = vec![
            1.0, 2.0, 3.0,  // Sample 1
            4.0, 5.0, 6.0,  // Sample 2
        ];
        let input = Tensor::from_values(input_data, vec![2, 3]).unwrap();
        
        let output = batch_norm.forward(&input).unwrap();
        assert_eq!(output.shape, vec![2, 3]);
        
        // Verify parameters were initialized
        assert!(batch_norm.weight.is_some());
        assert!(batch_norm.bias.is_some());
        assert_eq!(batch_norm.num_features, Some(3));
    }
    
    #[test]
    fn test_batch_norm_4d_forward() {
        let mut batch_norm = BatchNormalizationLayer::new();
        
        // Initialize parameters for 2 channels
        batch_norm.init_parameters(2).unwrap();
        
        // Input: [batch=1, channels=2, height=2, width=2]
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0,  // Channel 0
            5.0, 6.0, 7.0, 8.0,  // Channel 1
        ];
        let input = Tensor::from_values(input_data, vec![1, 2, 2, 2]).unwrap();
        
        let output = batch_norm.forward(&input).unwrap();
        assert_eq!(output.shape, vec![1, 2, 2, 2]);
        
        // Verify parameters were initialized for 2 channels
        assert_eq!(batch_norm.num_features, Some(2));
    }
    
    #[test]
    fn test_batch_norm_parameters() {
        let mut batch_norm = BatchNormalizationLayer::new();
        
        // Initialize with 3 features
        batch_norm.init_parameters(3).unwrap();
        
        let params = batch_norm.parameters();
        assert_eq!(params.len(), 2); // weight and bias
        
        // Check parameter shapes
        assert_eq!(params[0].shape, vec![3]); // weight
        assert_eq!(params[1].shape, vec![3]); // bias
        
        // Check parameter count
        assert_eq!(batch_norm.parameter_count(), 6); // 3 weights + 3 biases
    }
    
    #[test]
    fn test_batch_norm_initialization() {
        let mut batch_norm = BatchNormalizationLayer::new();
        batch_norm.init_parameters(4).unwrap();
        
        let weight = batch_norm.weight.as_ref().unwrap();
        let bias = batch_norm.bias.as_ref().unwrap();
        let running_mean = batch_norm.running_mean.as_ref().unwrap();
        let running_var = batch_norm.running_var.as_ref().unwrap();
        
        // Weight should be initialized to ones
        for val in &weight.data {
            assert_eq!(val.value(), 1.0);
        }
        
        // Bias should be initialized to zeros
        for val in &bias.data {
            assert_eq!(val.value(), 0.0);
        }
        
        // Running mean should be initialized to zeros
        for val in &running_mean.data {
            assert_eq!(val.value(), 0.0);
        }
        
        // Running variance should be initialized to ones
        for val in &running_var.data {
            assert_eq!(val.value(), 1.0);
        }
    }
    
    #[test]
    fn test_batch_norm_output_shape() {
        let batch_norm = BatchNormalizationLayer::new();
        
        // 2D input
        let shape_2d = vec![16, 32];
        assert_eq!(batch_norm.output_shape(&shape_2d).unwrap(), shape_2d);
        
        // 4D input
        let shape_4d = vec![8, 64, 28, 28];
        assert_eq!(batch_norm.output_shape(&shape_4d).unwrap(), shape_4d);
        
        // Invalid shapes should error
        let shape_1d = vec![32];
        assert!(batch_norm.output_shape(&shape_1d).is_err());
        
        let shape_3d = vec![8, 32, 28];
        assert!(batch_norm.output_shape(&shape_3d).is_err());
    }
    
    #[test]
    fn test_regularization_layer_integration() {
        // Test combining dropout and batch norm in a network
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_values(input_data, vec![2, 3]).unwrap();
        
        // Apply batch norm first
        let mut batch_norm = BatchNormalizationLayer::new();
        batch_norm.init_parameters(3).unwrap();
        let normalized = batch_norm.forward(&input).unwrap();
        assert_eq!(normalized.shape, vec![2, 3]);
        
        // Then apply dropout (in eval mode to get deterministic results)
        let dropout = DropoutLayer::with_probability(0.2).eval();
        let output = dropout.forward(&normalized).unwrap();
        assert_eq!(output.shape, vec![2, 3]);
        
        // In eval mode, dropout should pass through batch norm output unchanged
        for (i, val) in normalized.data.iter().enumerate() {
            assert_eq!(output.data[i].value(), val.value());
        }
    }
}

/// DropoutLayer: Regularization layer that randomly sets inputs to zero during training
/// DropoutLayer[p] sets inputs to zero with probability p during training
#[derive(Debug, Clone)]
pub struct DropoutLayer {
    /// Dropout probability (0.0 to 1.0)
    pub drop_probability: f64,
    /// Whether the layer is in training mode (dropout active) or inference mode
    pub training: bool,
    /// Layer name for debugging
    pub layer_name: String,
}

impl DropoutLayer {
    /// Create a new dropout layer with default probability 0.5
    pub fn new() -> Self {
        Self::with_probability(0.5)
    }
    
    /// Create a new dropout layer with specified probability
    pub fn with_probability(drop_probability: f64) -> Self {
        assert!(drop_probability >= 0.0 && drop_probability <= 1.0, 
                "Dropout probability must be between 0.0 and 1.0");
        
        Self {
            drop_probability,
            training: true, // Default to training mode
            layer_name: format!("DropoutLayer[{}]", drop_probability),
        }
    }
    
    /// Set training mode (dropout active)
    pub fn train(mut self) -> Self {
        self.training = true;
        self
    }
    
    /// Set evaluation mode (dropout inactive)
    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }
}

impl Layer for DropoutLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        if !self.training || self.drop_probability == 0.0 {
            // During inference or with zero dropout, pass through unchanged
            return Ok(input.clone());
        }
        
        if self.drop_probability == 1.0 {
            // Drop everything - return zeros
            return Ok(Tensor::zeros(input.shape.clone()));
        }
        
        // Apply dropout with scaling to maintain expected value
        let keep_probability = 1.0 - self.drop_probability;
        let scale = 1.0 / keep_probability;
        
        let mut output_data = Vec::with_capacity(input.data.len());
        
        for &val in &input.data {
            let random_val: f64 = rand::random();
            if random_val < self.drop_probability {
                // Drop this element
                output_data.push(Dual::constant(0.0));
            } else {
                // Keep and scale this element
                output_data.push(val * Dual::constant(scale));
            }
        }
        
        Tensor::new(output_data, input.shape.clone())
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

/// BatchNormalizationLayer: Normalizes inputs by learning mean and variance
/// BatchNormalizationLayer[] represents batch normalization with learnable parameters
#[derive(Debug, Clone)]
pub struct BatchNormalizationLayer {
    /// Number of features (channels)
    pub num_features: Option<usize>,
    /// Small constant for numerical stability
    pub epsilon: f64,
    /// Momentum for running statistics
    pub momentum: f64,
    /// Learnable scale parameter (gamma)
    pub weight: Option<Tensor>,
    /// Learnable shift parameter (beta)
    pub bias: Option<Tensor>,
    /// Running mean for inference
    pub running_mean: Option<Tensor>,
    /// Running variance for inference
    pub running_var: Option<Tensor>,
    /// Whether the layer is in training mode
    pub training: bool,
    /// Layer name for debugging
    pub layer_name: String,
}

impl BatchNormalizationLayer {
    /// Create a new batch normalization layer
    pub fn new() -> Self {
        Self {
            num_features: None,
            epsilon: 1e-5,
            momentum: 0.1,
            weight: None,
            bias: None,
            running_mean: None,
            running_var: None,
            training: true,
            layer_name: "BatchNormalizationLayer".to_string(),
        }
    }
    
    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Set momentum for running statistics
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Set training mode
    pub fn train(mut self) -> Self {
        self.training = true;
        self
    }
    
    /// Set evaluation mode
    pub fn eval(mut self) -> Self {
        self.training = false;
        self
    }
    
    /// Initialize parameters based on input shape
    fn init_parameters(&mut self, num_features: usize) -> MLResult<()> {
        self.num_features = Some(num_features);
        
        // Initialize weight (gamma) to ones
        let weight_values = vec![1.0; num_features];
        self.weight = Some(Tensor::variables(weight_values, vec![num_features])?);
        
        // Initialize bias (beta) to zeros
        let bias_values = vec![0.0; num_features];
        self.bias = Some(Tensor::variables(bias_values, vec![num_features])?);
        
        // Initialize running statistics
        let mean_values = vec![0.0; num_features];
        self.running_mean = Some(Tensor::from_values(mean_values, vec![num_features])?);
        
        let var_values = vec![1.0; num_features];
        self.running_var = Some(Tensor::from_values(var_values, vec![num_features])?);
        
        Ok(())
    }
    
    /// Compute batch statistics for training
    fn compute_batch_stats(&self, input: &Tensor) -> MLResult<(Tensor, Tensor)> {
        // Expect input shape: [N, C, H, W] for 4D or [N, C] for 2D
        let (batch_size, num_features) = match input.shape.len() {
            2 => (input.shape[0], input.shape[1]),
            4 => (input.shape[0], input.shape[1]),
            _ => return Err(MLError::InvalidLayer {
                reason: format!("BatchNorm expects 2D or 4D input, got {}D", input.shape.len()),
            }),
        };
        
        let spatial_size = input.data.len() / (batch_size * num_features);
        let total_elements = batch_size * spatial_size;
        
        // Compute mean for each channel
        let mut channel_means = vec![Dual::constant(0.0); num_features];
        
        for c in 0..num_features {
            let mut sum = Dual::constant(0.0);
            for n in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = match input.shape.len() {
                        2 => n * num_features + c,
                        4 => {
                            let h = input.shape[2];
                            let w = input.shape[3];
                            let spatial_idx = s;
                            let spatial_h = spatial_idx / w;
                            let spatial_w = spatial_idx % w;
                            n * (num_features * h * w) + c * (h * w) + spatial_h * w + spatial_w
                        },
                        _ => unreachable!(),
                    };
                    sum = sum + input.data[idx];
                }
            }
            channel_means[c] = sum / Dual::constant(total_elements as f64);
        }
        
        // Compute variance for each channel
        let mut channel_vars = vec![Dual::constant(0.0); num_features];
        
        for c in 0..num_features {
            let mut sum_sq_diff = Dual::constant(0.0);
            for n in 0..batch_size {
                for s in 0..spatial_size {
                    let idx = match input.shape.len() {
                        2 => n * num_features + c,
                        4 => {
                            let h = input.shape[2];
                            let w = input.shape[3];
                            let spatial_idx = s;
                            let spatial_h = spatial_idx / w;
                            let spatial_w = spatial_idx % w;
                            n * (num_features * h * w) + c * (h * w) + spatial_h * w + spatial_w
                        },
                        _ => unreachable!(),
                    };
                    let diff = input.data[idx] - channel_means[c];
                    sum_sq_diff = sum_sq_diff + diff * diff;
                }
            }
            channel_vars[c] = sum_sq_diff / Dual::constant(total_elements as f64);
        }
        
        let mean_tensor = Tensor::new(channel_means, vec![num_features])?;
        let var_tensor = Tensor::new(channel_vars, vec![num_features])?;
        
        Ok((mean_tensor, var_tensor))
    }
}

impl Layer for BatchNormalizationLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        // Determine number of features from input shape
        let num_features = match input.shape.len() {
            2 => input.shape[1], // [N, C]
            4 => input.shape[1], // [N, C, H, W]
            _ => return Err(MLError::InvalidLayer {
                reason: format!("BatchNorm expects 2D or 4D input, got {}D", input.shape.len()),
            }),
        };
        
        // Check if parameters are initialized
        if self.weight.is_none() {
            return Err(MLError::InvalidLayer {
                reason: "BatchNormalization parameters not initialized. Call initialize() first.".to_string(),
            });
        }
        
        let weight = self.weight.as_ref().unwrap();
        let bias = self.bias.as_ref().unwrap();
        
        let (mean, var) = if self.training {
            // Use batch statistics during training
            self.compute_batch_stats(input)?
        } else {
            // Use running statistics during inference
            let running_mean = self.running_mean.as_ref().unwrap();
            let running_var = self.running_var.as_ref().unwrap();
            (running_mean.clone(), running_var.clone())
        };
        
        // Normalize and apply learned parameters
        let mut output_data = Vec::with_capacity(input.data.len());
        
        match input.shape.len() {
            2 => {
                // 2D case: [N, C]
                let batch_size = input.shape[0];
                for n in 0..batch_size {
                    for c in 0..num_features {
                        let idx = n * num_features + c;
                        let std_dev = (var.data[c] + Dual::constant(self.epsilon)).sqrt()
                            .map_err(|e| MLError::NetworkError { 
                                reason: format!("Failed to compute standard deviation: {}", e) 
                            })?;
                        let normalized = (input.data[idx] - mean.data[c]) / std_dev;
                        let output = weight.data[c] * normalized + bias.data[c];
                        output_data.push(output);
                    }
                }
            },
            4 => {
                // 4D case: [N, C, H, W]
                let batch_size = input.shape[0];
                let height = input.shape[2];
                let width = input.shape[3];
                
                for n in 0..batch_size {
                    for c in 0..num_features {
                        for h in 0..height {
                            for w in 0..width {
                                let idx = n * (num_features * height * width) + 
                                         c * (height * width) + h * width + w;
                                let std_dev = (var.data[c] + Dual::constant(self.epsilon)).sqrt()
                                    .map_err(|e| MLError::NetworkError { 
                                        reason: format!("Failed to compute standard deviation: {}", e) 
                                    })?;
                                let normalized = (input.data[idx] - mean.data[c]) / std_dev;
                                let output = weight.data[c] * normalized + bias.data[c];
                                output_data.push(output);
                            }
                        }
                    }
                }
            },
            _ => unreachable!(),
        }
        
        Tensor::new(output_data, input.shape.clone())
    }
    
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }
    
    fn initialize(&mut self) -> MLResult<()> {
        if let Some(num_features) = self.num_features {
            self.init_parameters(num_features)
        } else {
            Ok(())
        }
    }
    
    fn name(&self) -> &str {
        &self.layer_name
    }
    
    fn output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        match input_shape.len() {
            2 | 4 => Ok(input_shape.to_vec()),
            _ => Err(MLError::InvalidLayer {
                reason: format!("BatchNorm expects 2D or 4D input shape, got {}D", input_shape.len()),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

//
// ===== SPATIAL & UTILITY LAYERS =====
//

/// FlattenLayer: Converts multi-dimensional input to 1D
/// FlattenLayer[] flattens all dimensions except the first (batch) dimension
#[derive(Debug, Clone)]
pub struct FlattenLayer {
    /// Starting dimension to flatten from (default: 1, preserving batch dimension)
    pub start_dim: usize,
    /// Ending dimension to flatten to (default: -1, meaning last dimension)
    pub end_dim: Option<usize>,
    /// Layer name for debugging
    pub layer_name: String,
}

impl FlattenLayer {
    /// Create a new flatten layer that flattens from dimension 1 onwards
    pub fn new() -> Self {
        Self {
            start_dim: 1,
            end_dim: None,
            layer_name: "FlattenLayer".to_string(),
        }
    }
    
    /// Create a flatten layer with custom start dimension
    pub fn from_dim(start_dim: usize) -> Self {
        Self {
            start_dim,
            end_dim: None,
            layer_name: format!("FlattenLayer[start_dim={}]", start_dim),
        }
    }
    
    /// Create a flatten layer with custom start and end dimensions
    pub fn from_range(start_dim: usize, end_dim: usize) -> Self {
        Self {
            start_dim,
            end_dim: Some(end_dim),
            layer_name: format!("FlattenLayer[{}:{}]", start_dim, end_dim),
        }
    }
}

impl Layer for FlattenLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        let input_shape = &input.shape;
        
        if input_shape.is_empty() {
            return Err(MLError::InvalidLayer {
                reason: "Cannot flatten tensor with empty shape".to_string(),
            });
        }
        
        if self.start_dim >= input_shape.len() {
            return Err(MLError::InvalidLayer {
                reason: format!("Start dimension {} out of bounds for {}-dimensional tensor", 
                               self.start_dim, input_shape.len()),
            });
        }
        
        let end_dim = self.end_dim.unwrap_or(input_shape.len() - 1);
        if end_dim >= input_shape.len() {
            return Err(MLError::InvalidLayer {
                reason: format!("End dimension {} out of bounds for {}-dimensional tensor", 
                               end_dim, input_shape.len()),
            });
        }
        
        if self.start_dim > end_dim {
            return Err(MLError::InvalidLayer {
                reason: format!("Start dimension {} must be <= end dimension {}", 
                               self.start_dim, end_dim),
            });
        }
        
        // Compute new shape
        let mut new_shape = Vec::new();
        
        // Keep dimensions before start_dim unchanged
        new_shape.extend_from_slice(&input_shape[..self.start_dim]);
        
        // Flatten dimensions from start_dim to end_dim
        let flattened_size: usize = input_shape[self.start_dim..=end_dim].iter().product();
        new_shape.push(flattened_size);
        
        // Keep dimensions after end_dim unchanged
        if end_dim + 1 < input_shape.len() {
            new_shape.extend_from_slice(&input_shape[end_dim + 1..]);
        }
        
        // Data remains the same, only shape changes
        Tensor::new(input.data.clone(), new_shape)
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
        if input_shape.is_empty() {
            return Err(MLError::InvalidLayer {
                reason: "Cannot flatten tensor with empty shape".to_string(),
            });
        }
        
        if self.start_dim >= input_shape.len() {
            return Err(MLError::InvalidLayer {
                reason: format!("Start dimension {} out of bounds for {}-dimensional tensor", 
                               self.start_dim, input_shape.len()),
            });
        }
        
        let end_dim = self.end_dim.unwrap_or(input_shape.len() - 1);
        if end_dim >= input_shape.len() {
            return Err(MLError::InvalidLayer {
                reason: format!("End dimension {} out of bounds for {}-dimensional tensor", 
                               end_dim, input_shape.len()),
            });
        }
        
        let mut new_shape = Vec::new();
        new_shape.extend_from_slice(&input_shape[..self.start_dim]);
        
        let flattened_size: usize = input_shape[self.start_dim..=end_dim].iter().product();
        new_shape.push(flattened_size);
        
        if end_dim + 1 < input_shape.len() {
            new_shape.extend_from_slice(&input_shape[end_dim + 1..]);
        }
        
        Ok(new_shape)
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// ReshapeLayer: Transforms tensor to a new shape
/// ReshapeLayer[{dims...}] reshapes tensor to specified dimensions
#[derive(Debug, Clone)]
pub struct ReshapeLayer {
    /// Target shape for the tensor
    pub target_shape: Vec<i32>, // -1 means infer this dimension
    /// Layer name for debugging
    pub layer_name: String,
}

impl ReshapeLayer {
    /// Create a new reshape layer with target shape
    pub fn new(target_shape: Vec<i32>) -> Self {
        Self {
            layer_name: format!("ReshapeLayer{:?}", target_shape),
            target_shape,
        }
    }
    
    /// Compute the actual output shape, resolving any -1 dimensions
    fn compute_output_shape(&self, input_shape: &[usize]) -> MLResult<Vec<usize>> {
        let total_elements: usize = input_shape.iter().product();
        
        if total_elements == 0 {
            return Err(MLError::InvalidLayer {
                reason: "Cannot reshape tensor with zero elements".to_string(),
            });
        }
        
        // Count -1 dimensions and compute known size
        let mut unknown_dim_idx = None;
        let mut known_size = 1usize;
        
        for (i, &dim) in self.target_shape.iter().enumerate() {
            if dim == -1 {
                if unknown_dim_idx.is_some() {
                    return Err(MLError::InvalidLayer {
                        reason: "Only one dimension can be -1 in reshape".to_string(),
                    });
                }
                unknown_dim_idx = Some(i);
            } else if dim <= 0 {
                return Err(MLError::InvalidLayer {
                    reason: format!("Invalid dimension {} in reshape (must be positive or -1)", dim),
                });
            } else {
                known_size *= dim as usize;
            }
        }
        
        let mut output_shape: Vec<usize> = self.target_shape.iter()
            .map(|&dim| if dim == -1 { 0 } else { dim as usize })
            .collect();
        
        // Resolve -1 dimension
        if let Some(idx) = unknown_dim_idx {
            if total_elements % known_size != 0 {
                return Err(MLError::InvalidLayer {
                    reason: format!("Cannot reshape tensor with {} elements to shape with known size {} (not divisible)", 
                                   total_elements, known_size),
                });
            }
            output_shape[idx] = total_elements / known_size;
        } else {
            // Check that shapes are compatible
            let target_elements: usize = output_shape.iter().product();
            if target_elements != total_elements {
                return Err(MLError::InvalidLayer {
                    reason: format!("Cannot reshape tensor with {} elements to shape with {} elements", 
                                   total_elements, target_elements),
                });
            }
        }
        
        Ok(output_shape)
    }
}

impl Layer for ReshapeLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        let output_shape = self.compute_output_shape(&input.shape)?;
        
        // Data remains the same, only shape changes
        Tensor::new(input.data.clone(), output_shape)
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
        self.compute_output_shape(input_shape)
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

/// PermuteLayer: Reorders tensor dimensions according to a permutation
/// PermuteLayer[{perm...}] reorders dimensions according to the permutation
#[derive(Debug, Clone)]
pub struct PermuteLayer {
    /// Permutation of dimensions (0-indexed)
    pub permutation: Vec<usize>,
    /// Layer name for debugging
    pub layer_name: String,
}

impl PermuteLayer {
    /// Create a new permute layer
    pub fn new(permutation: Vec<usize>) -> Self {
        Self {
            layer_name: format!("PermuteLayer{:?}", permutation),
            permutation,
        }
    }
    
    /// Validate that the permutation is valid for the given input shape
    fn validate_permutation(&self, input_shape: &[usize]) -> MLResult<()> {
        if self.permutation.len() != input_shape.len() {
            return Err(MLError::InvalidLayer {
                reason: format!("Permutation length {} does not match input rank {}", 
                               self.permutation.len(), input_shape.len()),
            });
        }
        
        let mut used = vec![false; input_shape.len()];
        for &dim in &self.permutation {
            if dim >= input_shape.len() {
                return Err(MLError::InvalidLayer {
                    reason: format!("Permutation dimension {} out of bounds for rank {}", 
                                   dim, input_shape.len()),
                });
            }
            if used[dim] {
                return Err(MLError::InvalidLayer {
                    reason: format!("Dimension {} appears multiple times in permutation", dim),
                });
            }
            used[dim] = true;
        }
        
        Ok(())
    }
}

impl Layer for PermuteLayer {
    fn forward(&self, input: &Tensor) -> MLResult<Tensor> {
        self.validate_permutation(&input.shape)?;
        
        // Compute output shape
        let output_shape: Vec<usize> = self.permutation.iter()
            .map(|&i| input.shape[i])
            .collect();
        
        // For now, implement a simple case where we just change the shape
        // A full implementation would properly reorder the data according to permutation
        // This is a placeholder implementation
        Tensor::new(input.data.clone(), output_shape)
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
        self.validate_permutation(input_shape)?;
        Ok(self.permutation.iter().map(|&i| input_shape[i]).collect())
    }
    
    fn clone_boxed(&self) -> Box<dyn Layer> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flatten_layer() {
        use crate::stdlib::autodiff::Dual;
        
        // Test basic flatten operation (flatten from start_dim=1)
        let flatten = FlattenLayer::from_dim(1);
        let input = Tensor::new(
            vec![Dual::constant(1.0), Dual::constant(2.0), Dual::constant(3.0), 
                 Dual::constant(4.0), Dual::constant(5.0), Dual::constant(6.0)], 
            vec![2, 3]
        ).unwrap();
        
        let result = flatten.forward(&input).unwrap();
        assert_eq!(result.shape, vec![2, 3]);
        
        // Test flatten 3D tensor
        let input_3d = Tensor::new(
            (0..8).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 2, 2]
        ).unwrap();
        
        let result_3d = flatten.forward(&input_3d).unwrap();
        assert_eq!(result_3d.shape, vec![2, 4]); // flatten from dim 1: [2, 2*2]
        
        // Test flatten with end_dim specified
        let flatten_partial = FlattenLayer::from_range(1, 2);
        let input_4d = Tensor::new(
            (0..24).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 3, 2, 2]
        ).unwrap();
        
        let result_4d = flatten_partial.forward(&input_4d).unwrap();
        assert_eq!(result_4d.shape, vec![2, 6, 2]); // flatten dims 1-2: [2, 3*2, 2]
    }

    #[test]
    fn test_flatten_layer_errors() {
        use crate::stdlib::autodiff::Dual;
        
        let flatten = FlattenLayer::from_dim(1);
        
        // Test invalid start_dim
        let flatten_invalid = FlattenLayer::from_dim(5);
        let input = Tensor::new(vec![Dual::constant(1.0), Dual::constant(2.0)], vec![1, 2]).unwrap();
        assert!(flatten_invalid.forward(&input).is_err());
        
        // Test invalid end_dim
        let flatten_invalid_end = FlattenLayer::from_range(0, 5);
        assert!(flatten_invalid_end.forward(&input).is_err());
        
        // Test 0D tensor (edge case)
        let flatten_0d = FlattenLayer::from_dim(0);
        let scalar_input = Tensor::new(vec![Dual::constant(1.0)], vec![]).unwrap();
        assert!(flatten_0d.forward(&scalar_input).is_err());
    }

    #[test]
    fn test_reshape_layer() {
        use crate::stdlib::autodiff::Dual;
        
        // Test basic reshape
        let reshape = ReshapeLayer::new(vec![3, 2]);
        let input = Tensor::new(
            (0..6).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 3]
        ).unwrap();
        
        let result = reshape.forward(&input).unwrap();
        assert_eq!(result.shape, vec![3, 2]);
        
        // Test reshape with -1 (inferred dimension)
        let reshape_infer = ReshapeLayer::new(vec![-1, 2]);
        let result_infer = reshape_infer.forward(&input).unwrap();
        assert_eq!(result_infer.shape, vec![3, 2]); // 6 elements, 2 wide -> 3 tall
        
        // Test reshape to 1D
        let reshape_1d = ReshapeLayer::new(vec![-1]);
        let result_1d = reshape_1d.forward(&input).unwrap();
        assert_eq!(result_1d.shape, vec![6]);
        
        // Test reshape 4D to 2D
        let input_4d = Tensor::new(
            (0..24).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 3, 2, 2]
        ).unwrap();
        let reshape_4d_to_2d = ReshapeLayer::new(vec![6, 4]);
        let result_4d = reshape_4d_to_2d.forward(&input_4d).unwrap();
        assert_eq!(result_4d.shape, vec![6, 4]);
    }

    #[test]
    fn test_reshape_layer_errors() {
        use crate::stdlib::autodiff::Dual;
        
        // Test incompatible element count
        let reshape = ReshapeLayer::new(vec![2, 4]); // wants 8 elements
        let input = Tensor::new(
            vec![Dual::constant(1.0), Dual::constant(2.0), Dual::constant(3.0)], 
            vec![3]
        ).unwrap(); // has 3 elements
        assert!(reshape.forward(&input).is_err());
        
        // Test multiple -1 dimensions
        let reshape_invalid = ReshapeLayer::new(vec![-1, -1]);
        let input = Tensor::new(
            (0..4).map(|i| Dual::constant(i as f64)).collect(), 
            vec![4]
        ).unwrap();
        assert!(reshape_invalid.forward(&input).is_err());
        
        // Test -1 with incompatible size
        let reshape_bad_infer = ReshapeLayer::new(vec![-1, 3]);
        let input = Tensor::new(
            (0..4).map(|i| Dual::constant(i as f64)).collect(), 
            vec![4]
        ).unwrap(); // 4 elements, can't divide by 3
        assert!(reshape_bad_infer.forward(&input).is_err());
    }

    #[test]
    fn test_permute_layer() {
        use crate::stdlib::autodiff::Dual;
        
        // Test 2D transpose (swap dimensions)
        let permute = PermuteLayer::new(vec![1, 0]);
        let input = Tensor::new(
            (0..6).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 3]
        ).unwrap();
        
        let result = permute.forward(&input).unwrap();
        assert_eq!(result.shape, vec![3, 2]); // dimensions swapped
        
        // Test 3D permutation
        let permute_3d = PermuteLayer::new(vec![2, 0, 1]);
        let input_3d = Tensor::new(
            (0..24).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 3, 4]
        ).unwrap();
        
        let result_3d = permute_3d.forward(&input_3d).unwrap();
        assert_eq!(result_3d.shape, vec![4, 2, 3]); // [2,3,4] -> [4,2,3]
        
        // Test identity permutation
        let permute_identity = PermuteLayer::new(vec![0, 1, 2]);
        let result_identity = permute_identity.forward(&input_3d).unwrap();
        assert_eq!(result_identity.shape, vec![2, 3, 4]); // unchanged
        
        // Test 4D NHWC to NCHW conversion (common in ML)
        let permute_nhwc_to_nchw = PermuteLayer::new(vec![0, 3, 1, 2]);
        let input_nhwc = Tensor::new(
            (0..96).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 4, 4, 3] // [batch, height, width, channels] = 2*4*4*3 = 96 elements
        ).unwrap();
        
        let result_nchw = permute_nhwc_to_nchw.forward(&input_nhwc).unwrap();
        assert_eq!(result_nchw.shape, vec![2, 3, 4, 4]); // [batch, channels, height, width]
    }

    #[test]
    fn test_permute_layer_errors() {
        use crate::stdlib::autodiff::Dual;
        
        // Test wrong permutation length
        let permute = PermuteLayer::new(vec![1, 0]);
        let input = Tensor::new(
            vec![Dual::constant(1.0), Dual::constant(2.0), Dual::constant(3.0)], 
            vec![3]
        ).unwrap(); // 1D tensor
        assert!(permute.forward(&input).is_err());
        
        // Test out-of-bounds dimension
        let permute_oob = PermuteLayer::new(vec![0, 3]); // dimension 3 doesn't exist
        let input_2d = Tensor::new(
            (0..4).map(|i| Dual::constant(i as f64)).collect(), 
            vec![2, 2]
        ).unwrap();
        assert!(permute_oob.forward(&input_2d).is_err());
        
        // Test duplicate dimension
        let permute_dup = PermuteLayer::new(vec![0, 0]);
        assert!(permute_dup.forward(&input_2d).is_err());
        
        // Test missing dimension
        let permute_missing = PermuteLayer::new(vec![1]); // missing dimension 0
        assert!(permute_missing.forward(&input_2d).is_err());
    }

    #[test]
    fn test_spatial_layers_output_shapes() {
        // Test output shape computation without forward pass
        
        // FlattenLayer
        let flatten = FlattenLayer::from_dim(1);
        assert_eq!(flatten.output_shape(&[2, 3, 4]).unwrap(), vec![2, 12]);
        assert_eq!(flatten.output_shape(&[5, 2, 3, 4]).unwrap(), vec![5, 24]);
        
        let flatten_partial = FlattenLayer::from_range(1, 2);
        assert_eq!(flatten_partial.output_shape(&[2, 3, 4, 5]).unwrap(), vec![2, 12, 5]);
        
        // ReshapeLayer  
        let reshape = ReshapeLayer::new(vec![6, 4]);
        assert_eq!(reshape.output_shape(&[3, 8]).unwrap(), vec![6, 4]);
        
        let reshape_infer = ReshapeLayer::new(vec![-1, 4]);
        assert_eq!(reshape_infer.output_shape(&[3, 8]).unwrap(), vec![6, 4]);
        
        // PermuteLayer
        let permute = PermuteLayer::new(vec![2, 0, 1]);
        assert_eq!(permute.output_shape(&[2, 3, 4]).unwrap(), vec![4, 2, 3]);
    }

    #[test]
    fn test_spatial_layers_names() {
        let flatten = FlattenLayer::from_dim(1);
        assert!(flatten.name().contains("FlattenLayer"));
        
        let reshape = ReshapeLayer::new(vec![2, -1]);
        assert!(reshape.name().contains("ReshapeLayer"));
        
        let permute = PermuteLayer::new(vec![1, 0]);
        assert!(permute.name().contains("PermuteLayer"));
    }

    #[test]
    fn test_spatial_layers_no_parameters() {
        // All spatial layers should have no trainable parameters
        let mut flatten = FlattenLayer::from_dim(1);
        assert!(flatten.parameters().is_empty());
        assert!(flatten.parameters_mut().is_empty());
        assert!(flatten.initialize().is_ok());
        
        let mut reshape = ReshapeLayer::new(vec![2, 3]);
        assert!(reshape.parameters().is_empty());
        assert!(reshape.parameters_mut().is_empty());
        assert!(reshape.initialize().is_ok());
        
        let mut permute = PermuteLayer::new(vec![1, 0]);
        assert!(permute.parameters().is_empty());
        assert!(permute.parameters_mut().is_empty());
        assert!(permute.initialize().is_ok());
    }

    #[test]
    fn test_spatial_layers_cloning() {
        // Test that layers can be cloned
        let flatten = FlattenLayer::from_range(2, 3);
        let _flatten_clone = flatten.clone_boxed();
        
        let reshape = ReshapeLayer::new(vec![4, -1]);
        let _reshape_clone = reshape.clone_boxed();
        
        let permute = PermuteLayer::new(vec![2, 1, 0]);
        let _permute_clone = permute.clone_boxed();
    }

    #[test]
    fn test_common_spatial_transformations() {
        use crate::stdlib::autodiff::Dual;
        
        // Test common ML spatial transformations
        
        // Image batch flattening for linear layer input
        let flatten_for_linear = FlattenLayer::from_dim(1);
        let image_batch = Tensor::new(
            (0..48).map(|i| Dual::constant(i as f64)).collect(),
            vec![4, 3, 2, 2] // [batch=4, channels=3, height=2, width=2]
        ).unwrap();
        
        let flattened = flatten_for_linear.forward(&image_batch).unwrap();
        assert_eq!(flattened.shape, vec![4, 12]); // ready for linear layer
        
        // Reshape weight matrix for different layer sizes
        let weight_reshape = ReshapeLayer::new(vec![64, -1]);
        let weights = Tensor::new(
            (0..128).map(|i| Dual::constant(i as f64)).collect(),
            vec![32, 4] // 32x4 matrix
        ).unwrap();
        
        let reshaped_weights = weight_reshape.forward(&weights).unwrap();
        assert_eq!(reshaped_weights.shape, vec![64, 2]); // 64x2 matrix
        
        // Channel-last to channel-first conversion
        let nhwc_to_nchw = PermuteLayer::new(vec![0, 3, 1, 2]);
        let nhwc_tensor = Tensor::new(
            (0..96).map(|i| Dual::constant(i as f64)).collect(),
            vec![2, 4, 4, 3] // [batch, height, width, channels]
        ).unwrap();
        
        let nchw_tensor = nhwc_to_nchw.forward(&nhwc_tensor).unwrap();
        assert_eq!(nchw_tensor.shape, vec![2, 3, 4, 4]); // [batch, channels, height, width]
    }
}

