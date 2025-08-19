//! Gradient Context for Automatic Differentiation
//!
//! This module provides the context and state management for automatic
//! differentiation operations, including variable tracking, mode selection,
//! and gradient computation coordination.

use std::collections::HashMap;
use super::{AutodiffError, AutodiffResult, ComputationGraph, NodeId, Dual};

/// Automatic differentiation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutodiffMode {
    /// Forward-mode automatic differentiation using dual numbers
    Forward,
    /// Reverse-mode automatic differentiation using computation graphs
    Reverse,
    /// Automatic mode selection based on function characteristics
    Auto,
}

impl AutodiffMode {
    /// Get a human-readable name for this mode
    pub fn name(&self) -> &'static str {
        match self {
            AutodiffMode::Forward => "forward",
            AutodiffMode::Reverse => "reverse",
            AutodiffMode::Auto => "auto",
        }
    }
}

/// A variable in the gradient computation context
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name
    pub name: String,
    /// Current value
    pub value: f64,
    /// Whether this variable requires gradient computation
    pub requires_grad: bool,
    /// Node ID in computation graph (for reverse-mode)
    pub node_id: Option<NodeId>,
    /// Dual number representation (for forward-mode)
    pub dual: Option<Dual>,
}

impl Variable {
    /// Create a new variable
    pub fn new(name: String, value: f64, requires_grad: bool) -> Self {
        Self {
            name,
            value,
            requires_grad,
            node_id: None,
            dual: None,
        }
    }
    
    /// Create a variable that requires gradients
    pub fn with_grad(name: String, value: f64) -> Self {
        Self::new(name, value, true)
    }
    
    /// Create a constant variable (no gradients)
    pub fn constant(name: String, value: f64) -> Self {
        Self::new(name, value, false)
    }
    
    /// Set the node ID for reverse-mode autodiff
    pub fn set_node_id(&mut self, node_id: NodeId) {
        self.node_id = Some(node_id);
    }
    
    /// Set the dual number for forward-mode autodiff
    pub fn set_dual(&mut self, dual: Dual) {
        self.dual = Some(dual);
    }
    
    /// Get the gradient from computation graph
    pub fn get_gradient(&self, graph: &ComputationGraph) -> AutodiffResult<f64> {
        if let Some(node_id) = self.node_id {
            graph.get_gradient(node_id)
        } else {
            Ok(0.0)
        }
    }
    
    /// Get the gradient from dual number
    pub fn get_dual_gradient(&self) -> f64 {
        self.dual.map(|d| d.derivative()).unwrap_or(0.0)
    }
}

/// Context for gradient computation
#[derive(Debug, Clone)]
pub struct GradientContext {
    /// Registered variables
    variables: HashMap<String, Variable>,
    /// Computation graph for reverse-mode AD
    graph: ComputationGraph,
    /// Current autodiff mode
    mode: AutodiffMode,
    /// Whether gradients are enabled
    gradients_enabled: bool,
}

impl GradientContext {
    /// Create a new gradient context
    pub fn new(mode: AutodiffMode) -> Self {
        Self {
            variables: HashMap::new(),
            graph: ComputationGraph::new(),
            mode,
            gradients_enabled: true,
        }
    }
    
    /// Create context with forward-mode autodiff
    pub fn forward_mode() -> Self {
        Self::new(AutodiffMode::Forward)
    }
    
    /// Create context with reverse-mode autodiff
    pub fn reverse_mode() -> Self {
        Self::new(AutodiffMode::Reverse)
    }
    
    /// Create context with automatic mode selection
    pub fn auto_mode() -> Self {
        Self::new(AutodiffMode::Auto)
    }
    
    /// Register a variable
    pub fn register_variable(&mut self, name: String, value: f64, requires_grad: bool) -> AutodiffResult<()> {
        let mut variable = Variable::new(name.clone(), value, requires_grad);
        
        match self.mode {
            AutodiffMode::Forward => {
                if requires_grad {
                    variable.set_dual(Dual::variable(value));
                } else {
                    variable.set_dual(Dual::constant(value));
                }
            }
            AutodiffMode::Reverse => {
                let node_id = self.graph.add_input(Some(name.clone()), value, requires_grad);
                variable.set_node_id(node_id);
            }
            AutodiffMode::Auto => {
                // For now, use reverse-mode as default in auto mode
                let node_id = self.graph.add_input(Some(name.clone()), value, requires_grad);
                variable.set_node_id(node_id);
            }
        }
        
        self.variables.insert(name, variable);
        Ok(())
    }
    
    /// Get a variable by name
    pub fn get_variable(&self, name: &str) -> AutodiffResult<&Variable> {
        self.variables.get(name).ok_or_else(|| AutodiffError::VariableNotFound {
            name: name.to_string(),
        })
    }
    
    /// Get a mutable variable by name
    pub fn get_variable_mut(&mut self, name: &str) -> AutodiffResult<&mut Variable> {
        self.variables.get_mut(name).ok_or_else(|| AutodiffError::VariableNotFound {
            name: name.to_string(),
        })
    }
    
    /// Get all variable names
    pub fn variable_names(&self) -> Vec<&String> {
        self.variables.keys().collect()
    }
    
    /// Get the computation graph (for reverse-mode)
    pub fn graph(&self) -> &ComputationGraph {
        &self.graph
    }
    
    /// Get mutable access to computation graph
    pub fn graph_mut(&mut self) -> &mut ComputationGraph {
        &mut self.graph
    }
    
    /// Get the current autodiff mode
    pub fn mode(&self) -> AutodiffMode {
        self.mode
    }
    
    /// Set the autodiff mode
    pub fn set_mode(&mut self, mode: AutodiffMode) {
        self.mode = mode;
    }
    
    /// Check if gradients are enabled
    pub fn gradients_enabled(&self) -> bool {
        self.gradients_enabled
    }
    
    /// Enable or disable gradient computation
    pub fn set_gradients_enabled(&mut self, enabled: bool) {
        self.gradients_enabled = enabled;
    }
    
    /// Clear all gradients
    pub fn zero_grad(&mut self) {
        self.graph.zero_grad();
        
        // Reset dual number derivatives for forward-mode
        for variable in self.variables.values_mut() {
            if let Some(dual) = variable.dual {
                variable.dual = Some(Dual::new(dual.value(), 0.0));
            }
        }
    }
    
    /// Compute gradients for all variables
    pub fn backward(&mut self, output_var: &str) -> AutodiffResult<()> {
        if !self.gradients_enabled {
            return Ok(());
        }
        
        let variable = self.get_variable(output_var)?.clone();
        
        match self.mode {
            AutodiffMode::Forward => {
                // In forward-mode, gradients are already computed
                Ok(())
            }
            AutodiffMode::Reverse | AutodiffMode::Auto => {
                if let Some(node_id) = variable.node_id {
                    self.graph.backward(node_id)
                } else {
                    Err(AutodiffError::GradientComputationFailed {
                        reason: format!("Variable '{}' not found in computation graph", output_var),
                    })
                }
            }
        }
    }
    
    /// Get gradients for all variables that require them
    pub fn get_gradients(&self) -> HashMap<String, f64> {
        let mut gradients = HashMap::new();
        
        for (name, variable) in &self.variables {
            if variable.requires_grad {
                let grad = match self.mode {
                    AutodiffMode::Forward => variable.get_dual_gradient(),
                    AutodiffMode::Reverse | AutodiffMode::Auto => {
                        variable.get_gradient(&self.graph).unwrap_or(0.0)
                    }
                };
                gradients.insert(name.clone(), grad);
            }
        }
        
        gradients
    }
    
    /// Get gradient for a specific variable
    pub fn get_gradient(&self, var_name: &str) -> AutodiffResult<f64> {
        let variable = self.get_variable(var_name)?;
        
        if !variable.requires_grad {
            return Ok(0.0);
        }
        
        match self.mode {
            AutodiffMode::Forward => Ok(variable.get_dual_gradient()),
            AutodiffMode::Reverse | AutodiffMode::Auto => variable.get_gradient(&self.graph),
        }
    }
    
    /// Get context statistics
    pub fn stats(&self) -> ContextStats {
        let total_vars = self.variables.len();
        let grad_vars = self.variables.values().filter(|v| v.requires_grad).count();
        
        ContextStats {
            total_variables: total_vars,
            gradient_variables: grad_vars,
            mode: self.mode,
            gradients_enabled: self.gradients_enabled,
            graph_stats: self.graph.stats(),
        }
    }
}

impl Default for GradientContext {
    fn default() -> Self {
        Self::auto_mode()
    }
}

/// Statistics about the gradient context
#[derive(Debug, Clone)]
pub struct ContextStats {
    /// Total number of variables
    pub total_variables: usize,
    /// Number of variables requiring gradients
    pub gradient_variables: usize,
    /// Current autodiff mode
    pub mode: AutodiffMode,
    /// Whether gradients are enabled
    pub gradients_enabled: bool,
    /// Computation graph statistics
    pub graph_stats: super::graph::GraphStats,
}

impl std::fmt::Display for ContextStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GradientContext: {} vars ({} grad), {} mode, gradients {}, {}",
            self.total_variables,
            self.gradient_variables,
            self.mode.name(),
            if self.gradients_enabled { "enabled" } else { "disabled" },
            self.graph_stats
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_variable_creation() {
        let var = Variable::with_grad("x".to_string(), 2.0);
        assert_eq!(var.name, "x");
        assert_eq!(var.value, 2.0);
        assert!(var.requires_grad);
        
        let const_var = Variable::constant("c".to_string(), 3.14);
        assert_eq!(const_var.name, "c");
        assert_eq!(const_var.value, 3.14);
        assert!(!const_var.requires_grad);
    }
    
    #[test]
    fn test_context_creation() {
        let ctx = GradientContext::forward_mode();
        assert_eq!(ctx.mode(), AutodiffMode::Forward);
        assert!(ctx.gradients_enabled());
        
        let ctx = GradientContext::reverse_mode();
        assert_eq!(ctx.mode(), AutodiffMode::Reverse);
        
        let ctx = GradientContext::auto_mode();
        assert_eq!(ctx.mode(), AutodiffMode::Auto);
    }
    
    #[test]
    fn test_variable_registration() {
        let mut ctx = GradientContext::forward_mode();
        
        ctx.register_variable("x".to_string(), 1.0, true).unwrap();
        ctx.register_variable("y".to_string(), 2.0, false).unwrap();
        
        let x = ctx.get_variable("x").unwrap();
        assert_eq!(x.name, "x");
        assert_eq!(x.value, 1.0);
        assert!(x.requires_grad);
        assert!(x.dual.is_some());
        
        let y = ctx.get_variable("y").unwrap();
        assert_eq!(y.name, "y");
        assert_eq!(y.value, 2.0);
        assert!(!y.requires_grad);
    }
    
    #[test]
    fn test_variable_not_found() {
        let ctx = GradientContext::new(AutodiffMode::Forward);
        
        let result = ctx.get_variable("nonexistent");
        assert!(result.is_err());
        
        if let Err(AutodiffError::VariableNotFound { name }) = result {
            assert_eq!(name, "nonexistent");
        } else {
            panic!("Expected VariableNotFound error");
        }
    }
    
    #[test]
    fn test_gradient_computation_forward_mode() {
        let mut ctx = GradientContext::forward_mode();
        
        ctx.register_variable("x".to_string(), 2.0, true).unwrap();
        
        // For forward-mode, gradients are available immediately
        let grad = ctx.get_gradient("x").unwrap();
        assert_eq!(grad, 1.0); // Variable dual has derivative = 1
    }
    
    #[test]
    fn test_context_stats() {
        let mut ctx = GradientContext::reverse_mode();
        
        ctx.register_variable("x".to_string(), 1.0, true).unwrap();
        ctx.register_variable("y".to_string(), 2.0, false).unwrap();
        ctx.register_variable("z".to_string(), 3.0, true).unwrap();
        
        let stats = ctx.stats();
        assert_eq!(stats.total_variables, 3);
        assert_eq!(stats.gradient_variables, 2);
        assert_eq!(stats.mode, AutodiffMode::Reverse);
        assert!(stats.gradients_enabled);
    }
    
    #[test]
    fn test_zero_grad() {
        let mut ctx = GradientContext::reverse_mode();
        
        ctx.register_variable("x".to_string(), 1.0, true).unwrap();
        
        // Zero gradients should work without error
        ctx.zero_grad();
        
        let grad = ctx.get_gradient("x").unwrap();
        assert_eq!(grad, 0.0);
    }
}