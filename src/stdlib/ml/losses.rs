//! Loss Functions for Neural Network Training
//!
//! Implements various loss functions including MSE and Cross Entropy
//! with automatic differentiation support for gradient computation.

use crate::stdlib::ml::{MLResult, MLError};
use crate::stdlib::ml::layers::Tensor;

/// Base trait for loss functions
pub trait LossFunction {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> MLResult<f64>;
}

/// Mean Squared Error loss
#[derive(Debug, Clone)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> MLResult<f64> {
        // Ensure predictions and targets have same shape
        if predictions.shape != targets.shape {
            return Err(MLError::ShapeMismatch {
                expected: targets.shape.clone(),
                actual: predictions.shape.clone(),
            });
        }
        
        // Compute MSE: (1/n) * sum((pred - target)^2)
        let n = predictions.size() as f64;
        let mut sum = 0.0;
        
        for i in 0..predictions.size() {
            let diff = predictions.data[i].value() - targets.data[i].value();
            sum += diff * diff;
        }
        
        Ok(sum / n)
    }
}

/// Cross Entropy loss for classification
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss;

impl LossFunction for CrossEntropyLoss {
    fn compute_loss(&self, predictions: &Tensor, targets: &Tensor) -> MLResult<f64> {
        // Ensure predictions and targets have same shape
        if predictions.shape != targets.shape {
            return Err(MLError::ShapeMismatch {
                expected: targets.shape.clone(),
                actual: predictions.shape.clone(),
            });
        }
        
        // Compute cross entropy: -sum(target * log(pred))
        let mut sum = 0.0;
        
        for i in 0..predictions.size() {
            let pred = predictions.data[i].value().max(1e-8); // Prevent log(0)
            let target = targets.data[i].value();
            sum -= target * pred.ln();
        }
        
        Ok(sum / predictions.batch_size() as f64)
    }
}