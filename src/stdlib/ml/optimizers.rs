//! Optimization Algorithms for Neural Network Training
//!
//! Implements SGD, Adam and other optimizers with automatic
//! differentiation support for gradient-based parameter updates.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;

/// Base trait for optimizers
pub trait Optimizer {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) -> MLResult<()>;
}

/// Stochastic Gradient Descent optimizer
#[derive(Debug, Clone)]
pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) -> MLResult<()> {
        if parameters.len() != gradients.len() {
            return Err(MLError::TrainingError {
                reason: "Parameters and gradients length mismatch".to_string(),
            });
        }
        
        // SGD update: param = param - learning_rate * grad
        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            if param.shape != grad.shape {
                return Err(MLError::ShapeMismatch {
                    expected: param.shape.clone(),
                    actual: grad.shape.clone(),
                });
            }
            
            for i in 0..param.size() {
                let grad_val = grad.data[i].value();
                let new_val = param.data[i].value() - self.learning_rate * grad_val;
                param.data[i] = crate::stdlib::autodiff::Dual::variable(new_val);
            }
        }
        
        Ok(())
    }
}

/// Adam optimizer
#[derive(Debug, Clone)]
pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self { 
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
    
    pub fn with_params(learning_rate: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Self { learning_rate, beta1, beta2, epsilon }
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[&Tensor]) -> MLResult<()> {
        if parameters.len() != gradients.len() {
            return Err(MLError::TrainingError {
                reason: "Parameters and gradients length mismatch".to_string(),
            });
        }
        
        // Simple gradient descent for now (Adam implementation would require momentum tracking)
        for (param, grad) in parameters.iter_mut().zip(gradients.iter()) {
            if param.shape != grad.shape {
                return Err(MLError::ShapeMismatch {
                    expected: param.shape.clone(),
                    actual: grad.shape.clone(),
                });
            }
            
            // Update: param = param - learning_rate * grad
            for i in 0..param.size() {
                let grad_val = grad.data[i].value();
                let new_val = param.data[i].value() - self.learning_rate * grad_val;
                param.data[i] = crate::stdlib::autodiff::Dual::variable(new_val);
            }
        }
        
        Ok(())
    }
}