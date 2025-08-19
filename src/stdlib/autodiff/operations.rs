//! High-level Autodiff Operations
//!
//! This module provides high-level operations for automatic differentiation
//! that integrate with Lyra's expression system and VM.

use super::{AutodiffResult, GradientContext, Dual, Operation};

/// Trait for types that support automatic differentiation operations
pub trait AutodiffOps {
    /// Compute the gradient of this expression with respect to given variables
    fn gradient(&self, ctx: &mut GradientContext, variables: &[&str]) -> AutodiffResult<Vec<f64>>;
    
    /// Compute the Jacobian matrix for vector-valued functions
    fn jacobian(&self, ctx: &mut GradientContext, variables: &[&str]) -> AutodiffResult<Vec<Vec<f64>>>;
    
    /// Compute the Hessian matrix (second derivatives)
    fn hessian(&self, ctx: &mut GradientContext, variable: &str) -> AutodiffResult<Vec<Vec<f64>>>;
    
    /// Evaluate the expression and return both value and gradients
    fn value_and_grad(&self, ctx: &mut GradientContext, variables: &[&str]) -> AutodiffResult<(f64, Vec<f64>)>;
}

/// Basic arithmetic operations for dual numbers and computation graphs
pub struct ArithmeticOps;

impl ArithmeticOps {
    /// Addition operation
    pub fn add(ctx: &mut GradientContext, left_var: &str, right_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let left = ctx.get_variable(left_var)?;
                let right = ctx.get_variable(right_var)?;
                
                if let (Some(left_dual), Some(right_dual)) = (left.dual, right.dual) {
                    let result_dual = left_dual + right_dual;
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let left = ctx.get_variable(left_var)?;
                let right = ctx.get_variable(right_var)?;
                
                if let (Some(left_id), Some(right_id)) = (left.node_id, right.node_id) {
                    let result_id = ctx.graph_mut().add_binary_op(Operation::Add, left_id, right_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?; // Value will be computed in forward pass
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
    
    /// Multiplication operation
    pub fn mul(ctx: &mut GradientContext, left_var: &str, right_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let left = ctx.get_variable(left_var)?;
                let right = ctx.get_variable(right_var)?;
                
                if let (Some(left_dual), Some(right_dual)) = (left.dual, right.dual) {
                    let result_dual = left_dual * right_dual;
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let left = ctx.get_variable(left_var)?;
                let right = ctx.get_variable(right_var)?;
                
                if let (Some(left_id), Some(right_id)) = (left.node_id, right.node_id) {
                    let result_id = ctx.graph_mut().add_binary_op(Operation::Mul, left_id, right_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
    
    /// Sine operation
    pub fn sin(ctx: &mut GradientContext, input_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_dual) = input.dual {
                    let result_dual = input_dual.sin();
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_id) = input.node_id {
                    let result_id = ctx.graph_mut().add_unary_op(Operation::Sin, input_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
    
    /// Exponential operation
    pub fn exp(ctx: &mut GradientContext, input_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_dual) = input.dual {
                    let result_dual = input_dual.exp();
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_id) = input.node_id {
                    let result_id = ctx.graph_mut().add_unary_op(Operation::Exp, input_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
}

/// Numerical differentiation utilities for fallback when symbolic differentiation fails
pub struct NumericalOps;

impl NumericalOps {
    /// Compute numerical gradient using finite differences
    pub fn numerical_gradient<F>(func: F, x: f64, h: f64) -> f64
    where
        F: Fn(f64) -> f64,
    {
        (func(x + h) - func(x - h)) / (2.0 * h)
    }
    
    /// Compute numerical Jacobian for vector-valued functions
    pub fn numerical_jacobian<F>(func: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n = x.len();
        let y = func(x);
        let m = y.len();
        
        let mut jacobian = vec![vec![0.0; n]; m];
        
        for j in 0..n {
            let mut x_plus = x.to_vec();
            let mut x_minus = x.to_vec();
            
            x_plus[j] += h;
            x_minus[j] -= h;
            
            let y_plus = func(&x_plus);
            let y_minus = func(&x_minus);
            
            for i in 0..m {
                jacobian[i][j] = (y_plus[i] - y_minus[i]) / (2.0 * h);
            }
        }
        
        jacobian
    }
    
    /// Compute numerical Hessian using finite differences
    pub fn numerical_hessian<F>(func: F, x: &[f64], h: f64) -> Vec<Vec<f64>>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n = x.len();
        let mut hessian = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // Diagonal elements: second derivatives
                    let mut x_plus = x.to_vec();
                    let mut x_minus = x.to_vec();
                    
                    x_plus[i] += h;
                    x_minus[i] -= h;
                    
                    let f_plus = func(&x_plus);
                    let f_center = func(x);
                    let f_minus = func(&x_minus);
                    
                    hessian[i][j] = (f_plus - 2.0 * f_center + f_minus) / (h * h);
                } else {
                    // Off-diagonal elements: mixed partial derivatives
                    let mut x_pp = x.to_vec();
                    let mut x_pm = x.to_vec();
                    let mut x_mp = x.to_vec();
                    let mut x_mm = x.to_vec();
                    
                    x_pp[i] += h;
                    x_pp[j] += h;
                    
                    x_pm[i] += h;
                    x_pm[j] -= h;
                    
                    x_mp[i] -= h;
                    x_mp[j] += h;
                    
                    x_mm[i] -= h;
                    x_mm[j] -= h;
                    
                    let f_pp = func(&x_pp);
                    let f_pm = func(&x_pm);
                    let f_mp = func(&x_mp);
                    let f_mm = func(&x_mm);
                    
                    hessian[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
                }
            }
        }
        
        hessian
    }
}

/// Utilities for common machine learning operations
pub struct MLOps;

impl MLOps {
    /// ReLU activation function with autodiff support
    pub fn relu(ctx: &mut GradientContext, input_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_dual) = input.dual {
                    let result_dual = input_dual.relu();
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_id) = input.node_id {
                    let result_id = ctx.graph_mut().add_unary_op(Operation::ReLU, input_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
    
    /// Sigmoid activation function with autodiff support
    pub fn sigmoid(ctx: &mut GradientContext, input_var: &str, output_var: &str) -> AutodiffResult<()> {
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_dual) = input.dual {
                    let result_dual = input_dual.sigmoid();
                    ctx.register_variable(output_var.to_string(), result_dual.value(), true)?;
                    ctx.get_variable_mut(output_var)?.set_dual(result_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let input = ctx.get_variable(input_var)?;
                
                if let Some(input_id) = input.node_id {
                    let result_id = ctx.graph_mut().add_unary_op(Operation::Sigmoid, input_id, true)?;
                    ctx.register_variable(output_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(output_var)?.set_node_id(result_id);
                }
            }
        }
        Ok(())
    }
    
    /// Mean squared error loss function
    pub fn mse_loss(ctx: &mut GradientContext, pred_var: &str, target_var: &str, loss_var: &str) -> AutodiffResult<()> {
        // MSE = (pred - target)^2
        
        // First compute difference: diff = pred - target
        let diff_var = format!("{}_diff", loss_var);
        match ctx.mode() {
            super::AutodiffMode::Forward => {
                let pred = ctx.get_variable(pred_var)?;
                let target = ctx.get_variable(target_var)?;
                
                if let (Some(pred_dual), Some(target_dual)) = (pred.dual, target.dual) {
                    let diff_dual = pred_dual - target_dual;
                    let loss_dual = diff_dual * diff_dual; // Square the difference
                    
                    ctx.register_variable(loss_var.to_string(), loss_dual.value(), true)?;
                    ctx.get_variable_mut(loss_var)?.set_dual(loss_dual);
                }
            }
            super::AutodiffMode::Reverse | super::AutodiffMode::Auto => {
                let pred = ctx.get_variable(pred_var)?;
                let target = ctx.get_variable(target_var)?;
                
                if let (Some(pred_id), Some(target_id)) = (pred.node_id, target.node_id) {
                    // diff = pred - target
                    let diff_id = ctx.graph_mut().add_binary_op(Operation::Sub, pred_id, target_id, true)?;
                    
                    // loss = diff * diff
                    let loss_id = ctx.graph_mut().add_binary_op(Operation::Mul, diff_id, diff_id, true)?;
                    
                    ctx.register_variable(loss_var.to_string(), 0.0, true)?;
                    ctx.get_variable_mut(loss_var)?.set_node_id(loss_id);
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::AutodiffMode;
    
    #[test]
    fn test_arithmetic_operations_forward() {
        let mut ctx = GradientContext::forward_mode();
        
        ctx.register_variable("x".to_string(), 2.0, true).unwrap();
        ctx.register_variable("y".to_string(), 3.0, true).unwrap();
        
        // Test addition
        ArithmeticOps::add(&mut ctx, "x", "y", "sum").unwrap();
        let sum_var = ctx.get_variable("sum").unwrap();
        assert_eq!(sum_var.dual.unwrap().value(), 5.0);
        assert_eq!(sum_var.dual.unwrap().derivative(), 2.0); // dx + dy = 1 + 1 = 2
    }
    
    #[test]
    fn test_arithmetic_operations_reverse() {
        let mut ctx = GradientContext::reverse_mode();
        
        ctx.register_variable("x".to_string(), 2.0, true).unwrap();
        ctx.register_variable("y".to_string(), 3.0, true).unwrap();
        
        // Test multiplication
        ArithmeticOps::mul(&mut ctx, "x", "y", "product").unwrap();
        
        // Compute forward pass
        ctx.graph_mut().forward().unwrap();
        
        let product_node_id = ctx.get_variable("product").unwrap().node_id.unwrap();
        let product_value = ctx.graph().get_node(product_node_id).unwrap().value;
        assert_eq!(product_value, 6.0);
        
        // Compute backward pass
        ctx.backward("product").unwrap();
        
        let x_grad = ctx.get_gradient("x").unwrap();
        let y_grad = ctx.get_gradient("y").unwrap();
        
        // d/dx(x*y) = y = 3, d/dy(x*y) = x = 2
        assert_eq!(x_grad, 3.0);
        assert_eq!(y_grad, 2.0);
    }
    
    #[test]
    fn test_transcendental_functions() {
        let mut ctx = GradientContext::forward_mode();
        
        ctx.register_variable("x".to_string(), 0.0, true).unwrap();
        
        // Test sine function: sin(0) = 0, cos(0) = 1
        ArithmeticOps::sin(&mut ctx, "x", "sin_x").unwrap();
        let sin_var = ctx.get_variable("sin_x").unwrap();
        assert!((sin_var.dual.unwrap().value() - 0.0).abs() < 1e-10);
        assert!((sin_var.dual.unwrap().derivative() - 1.0).abs() < 1e-10);
        
        // Test exponential function: exp(0) = 1, exp'(0) = 1
        ArithmeticOps::exp(&mut ctx, "x", "exp_x").unwrap();
        let exp_var = ctx.get_variable("exp_x").unwrap();
        assert!((exp_var.dual.unwrap().value() - 1.0).abs() < 1e-10);
        assert!((exp_var.dual.unwrap().derivative() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_ml_operations() {
        let mut ctx = GradientContext::forward_mode();
        
        // Test ReLU with positive input
        ctx.register_variable("x_pos".to_string(), 2.0, true).unwrap();
        MLOps::relu(&mut ctx, "x_pos", "relu_pos").unwrap();
        let relu_var = ctx.get_variable("relu_pos").unwrap();
        assert_eq!(relu_var.dual.unwrap().value(), 2.0);
        assert_eq!(relu_var.dual.unwrap().derivative(), 1.0);
        
        // Test ReLU with negative input
        ctx.register_variable("x_neg".to_string(), -1.0, true).unwrap();
        MLOps::relu(&mut ctx, "x_neg", "relu_neg").unwrap();
        let relu_neg_var = ctx.get_variable("relu_neg").unwrap();
        assert_eq!(relu_neg_var.dual.unwrap().value(), 0.0);
        assert_eq!(relu_neg_var.dual.unwrap().derivative(), 0.0);
        
        // Test sigmoid at zero: sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        ctx.register_variable("x_zero".to_string(), 0.0, true).unwrap();
        MLOps::sigmoid(&mut ctx, "x_zero", "sigmoid_zero").unwrap();
        let sigmoid_var = ctx.get_variable("sigmoid_zero").unwrap();
        assert!((sigmoid_var.dual.unwrap().value() - 0.5).abs() < 1e-10);
        assert!((sigmoid_var.dual.unwrap().derivative() - 0.25).abs() < 1e-10);
    }
    
    #[test]
    fn test_mse_loss() {
        let mut ctx = GradientContext::forward_mode();
        
        ctx.register_variable("pred".to_string(), 2.0, true).unwrap();
        ctx.register_variable("target".to_string(), 1.0, false).unwrap(); // Target is constant
        
        MLOps::mse_loss(&mut ctx, "pred", "target", "loss").unwrap();
        
        let loss_var = ctx.get_variable("loss").unwrap();
        // MSE = (2 - 1)^2 = 1
        assert_eq!(loss_var.dual.unwrap().value(), 1.0);
        // d/dpred MSE = 2 * (pred - target) = 2 * (2 - 1) = 2
        assert_eq!(loss_var.dual.unwrap().derivative(), 2.0);
    }
    
    #[test]
    fn test_numerical_gradient() {
        // Test numerical gradient of x^2 at x = 3
        // f(x) = x^2, f'(x) = 2x, f'(3) = 6
        let func = |x| x * x;
        let grad = NumericalOps::numerical_gradient(func, 3.0, 1e-6);
        assert!((grad - 6.0).abs() < 1e-4);
        
        // Test numerical gradient of sin(x) at x = 0
        // f(x) = sin(x), f'(x) = cos(x), f'(0) = 1
        let func = |x: f64| x.sin();
        let grad = NumericalOps::numerical_gradient(func, 0.0, 1e-6);
        assert!((grad - 1.0).abs() < 1e-4);
    }
    
    #[test]
    fn test_numerical_jacobian() {
        // Test Jacobian of [x^2, y^2] at (1, 2)
        // J = [[2x, 0], [0, 2y]] = [[2, 0], [0, 4]]
        let func = |x: &[f64]| vec![x[0] * x[0], x[1] * x[1]];
        let jac = NumericalOps::numerical_jacobian(func, &[1.0, 2.0], 1e-6);
        
        assert!((jac[0][0] - 2.0).abs() < 1e-4); // ∂(x²)/∂x = 2x = 2
        assert!((jac[0][1] - 0.0).abs() < 1e-4); // ∂(x²)/∂y = 0
        assert!((jac[1][0] - 0.0).abs() < 1e-4); // ∂(y²)/∂x = 0
        assert!((jac[1][1] - 4.0).abs() < 1e-4); // ∂(y²)/∂y = 2y = 4
    }
}