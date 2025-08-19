//! Tree-Shaking System for Lyra
//!
//! This module implements a comprehensive tree-shaking system that analyzes
//! function dependencies across the stdlib hierarchy to enable dead code elimination.

pub mod dependency_graph;
pub mod graph_analyzer;
pub mod call_analyzer;
pub mod module_deps;
pub mod usage_tracker;
pub mod eliminator;
pub mod import_analyzer;

pub use dependency_graph::{DependencyGraph, DependencyNode, DependencyEdge, DependencyType, 
                           CallContext as DependencyCallContext, FunctionMetadata, 
                           PerformanceCharacteristics, NodeUsageStats, ComplexityMetrics};
pub use graph_analyzer::{GraphAnalyzer, CentralityMetrics, BottleneckAnalysis};
pub use call_analyzer::{CallAnalyzer, FunctionInfo, CallPattern};
pub use module_deps::{ModuleDependencyAnalyzer, ModuleNode, ModuleOptimization};
pub use usage_tracker::{UsageTracker, UsageStats, FunctionUsageStats};
pub use eliminator::{
    Eliminator, EliminatorConfig, EliminationResults, EliminatedFunction, 
    PreservedFunction, EliminationAnalysis
};
pub use import_analyzer::{
    ImportAnalyzer, ImportAnalyzerConfig, ImportPattern, ImportPatternType,
    ImportOptimizationType, OptimizationSuggestion, ImportOptimizationImpact, 
    ModuleImportOptimization, ImportDependencyGraph
};

use crate::modules::registry::ModuleRegistry;
use std::collections::HashMap;

/// Tree-shaking coordinator that orchestrates the entire optimization process
pub struct TreeShaker {
    /// Usage tracking system
    usage_tracker: UsageTracker,
    
    /// Dependency graph analyzer
    dependency_graph: DependencyGraph,
    
    /// Module dependency analyzer
    module_deps: ModuleDependencyAnalyzer,
    
    /// Function call analyzer
    call_analyzer: CallAnalyzer,
    
    /// Graph analysis algorithms
    graph_analyzer: GraphAnalyzer,
    
    /// Dead code elimination engine
    eliminator: Eliminator,
    
    /// Import optimization analyzer
    import_analyzer: ImportAnalyzer,
}

impl TreeShaker {
    /// Create a new tree-shaking system
    pub fn new() -> Self {
        TreeShaker {
            usage_tracker: UsageTracker::new(),
            dependency_graph: DependencyGraph::new(),
            module_deps: ModuleDependencyAnalyzer::new(),
            call_analyzer: CallAnalyzer::new(),
            graph_analyzer: GraphAnalyzer::new(),
            eliminator: Eliminator::new(),
            import_analyzer: ImportAnalyzer::new(),
        }
    }
    
    /// Analyze stdlib dependencies and build dependency graph
    pub fn analyze_stdlib(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        // Step 1: Track all function usage patterns
        self.usage_tracker.track_stdlib_usage(module_registry)?;
        
        // Step 2: Analyze function calls and build dependency graph
        self.call_analyzer.analyze_stdlib_calls(module_registry, &mut self.dependency_graph)?;
        
        // Step 3: Analyze module-level dependencies
        self.module_deps.analyze_module_dependencies(module_registry, &mut self.dependency_graph)?;
        
        // Step 4: Run graph analysis algorithms
        self.graph_analyzer.analyze_graph(&mut self.dependency_graph)?;
        
        // Step 5: Analyze import patterns and optimize imports
        self.import_analyzer.analyze_imports(&self.dependency_graph, &self.usage_tracker, module_registry)?;
        
        Ok(())
    }
    
    /// Get the complete dependency graph
    pub fn dependency_graph(&self) -> &DependencyGraph {
        &self.dependency_graph
    }
    
    /// Get usage statistics
    pub fn usage_stats(&self) -> &UsageStats {
        self.usage_tracker.stats()
    }
    
    /// Get list of unused functions that can be eliminated
    pub fn unused_functions(&self) -> Vec<String> {
        self.graph_analyzer.find_unused_functions(&self.dependency_graph)
    }
    
    /// Get optimization recommendations
    pub fn optimization_recommendations(&self) -> Vec<OptimizationRecommendation> {
        self.graph_analyzer.generate_recommendations(&self.dependency_graph, &self.usage_tracker)
    }
    
    /// Perform dead code elimination
    pub fn eliminate_dead_code(
        &mut self, 
        module_registry: &mut ModuleRegistry
    ) -> Result<EliminationResults, TreeShakeError> {
        self.eliminator.eliminate_dead_code(
            &self.dependency_graph,
            &self.usage_tracker,
            module_registry,
        )
    }
    
    /// Preview dead code elimination without actually removing functions
    pub fn preview_elimination(&mut self) -> Result<EliminationResults, TreeShakeError> {
        let mut config = EliminatorConfig::default();
        config.preview_mode = true;
        self.eliminator = Eliminator::with_config(config);
        
        // Create a dummy module registry for preview
        let function_registry = std::sync::Arc::new(std::sync::RwLock::new(crate::linker::FunctionRegistry::new()));
        let mut dummy_registry = crate::modules::registry::ModuleRegistry::new(function_registry);
        self.eliminator.eliminate_dead_code(
            &self.dependency_graph,
            &self.usage_tracker,
            &mut dummy_registry,
        )
    }
    
    /// Analyze elimination potential without performing elimination
    pub fn analyze_elimination_potential(&self) -> Result<EliminationAnalysis, TreeShakeError> {
        self.eliminator.analyze_elimination_potential(&self.dependency_graph, &self.usage_tracker)
    }
    
    /// Configure elimination settings
    pub fn configure_elimination(&mut self, config: EliminatorConfig) {
        self.eliminator = Eliminator::with_config(config);
    }
    
    /// Get import optimization results
    pub fn get_import_optimizations(&self) -> Vec<&ModuleImportOptimization> {
        self.import_analyzer.get_all_optimizations()
    }
    
    /// Get optimized import statements for all modules
    pub fn get_optimized_imports(&self) -> Result<HashMap<String, String>, TreeShakeError> {
        self.import_analyzer.generate_optimized_imports()
    }
    
    /// Get import pattern for a specific module
    pub fn get_import_pattern(&self, module_name: &str) -> Option<&ImportPattern> {
        self.import_analyzer.get_import_pattern(module_name)
    }
    
    /// Get total import optimization impact
    pub fn get_import_optimization_impact(&self) -> ImportOptimizationImpact {
        self.import_analyzer.calculate_total_impact()
    }
    
    /// Get import dependency graph
    pub fn get_import_dependency_graph(&self) -> &ImportDependencyGraph {
        self.import_analyzer.get_import_graph()
    }
    
    /// Configure import analysis settings
    pub fn configure_import_analysis(&mut self, config: ImportAnalyzerConfig) {
        self.import_analyzer = ImportAnalyzer::with_config(config);
    }
}

/// Tree-shaking specific errors
#[derive(Debug, Clone)]
pub enum TreeShakeError {
    /// Dependency analysis failed
    DependencyAnalysisError { message: String },
    
    /// Circular dependency detected
    CircularDependency { functions: Vec<String> },
    
    /// Graph analysis failed
    GraphAnalysisError { message: String },
    
    /// Module resolution failed
    ModuleResolutionError { module: String, message: String },
}

impl std::fmt::Display for TreeShakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TreeShakeError::DependencyAnalysisError { message } => {
                write!(f, "Dependency analysis error: {}", message)
            }
            TreeShakeError::CircularDependency { functions } => {
                write!(f, "Circular dependency detected between functions: {:?}", functions)
            }
            TreeShakeError::GraphAnalysisError { message } => {
                write!(f, "Graph analysis error: {}", message)
            }
            TreeShakeError::ModuleResolutionError { module, message } => {
                write!(f, "Module resolution error in {}: {}", module, message)
            }
        }
    }
}

impl std::error::Error for TreeShakeError {}

impl From<String> for TreeShakeError {
    fn from(message: String) -> Self {
        TreeShakeError::DependencyAnalysisError { message }
    }
}

/// Optimization recommendation from tree-shaking analysis
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Type of optimization
    pub recommendation_type: OptimizationType,
    
    /// Functions affected
    pub functions: Vec<String>,
    
    /// Expected impact
    pub impact: OptimizationImpact,
    
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    
    /// Description of the recommendation
    pub description: String,
}

/// Types of optimizations that can be performed
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationType {
    /// Remove completely unused functions
    DeadCodeElimination,
    
    /// Inline small frequently-called functions
    FunctionInlining,
    
    /// Combine related functions into single modules
    ModuleConsolidation,
    
    /// Split large modules with independent functions
    ModuleSplitting,
    
    /// Optimize import chains
    ImportOptimization,
    
    /// Lazy loading for rarely used functions
    LazyLoading,
}

/// Expected impact of an optimization
#[derive(Debug, Clone)]
pub struct OptimizationImpact {
    /// Estimated binary size reduction (bytes)
    pub size_reduction: usize,
    
    /// Estimated compilation time improvement (milliseconds)
    pub compile_time_improvement: u64,
    
    /// Estimated runtime performance impact
    pub runtime_impact: RuntimeImpact,
    
    /// Number of functions affected
    pub functions_affected: usize,
}

/// Runtime performance impact
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeImpact {
    /// Improves performance
    Positive(f64),
    
    /// No significant impact
    Neutral,
    
    /// Minor performance cost
    NegligibleCost(f64),
    
    /// Performance trade-off for size benefits
    Negative(f64),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tree_shaker_creation() {
        let tree_shaker = TreeShaker::new();
        
        // Verify initial state
        assert_eq!(tree_shaker.dependency_graph().node_count(), 0);
        assert_eq!(tree_shaker.unused_functions().len(), 0);
    }

    #[test]
    fn test_optimization_recommendation() {
        let recommendation = OptimizationRecommendation {
            recommendation_type: OptimizationType::DeadCodeElimination,
            functions: vec!["unused_function".to_string()],
            impact: OptimizationImpact {
                size_reduction: 1024,
                compile_time_improvement: 50,
                runtime_impact: RuntimeImpact::Neutral,
                functions_affected: 1,
            },
            confidence: 0.95,
            description: "Remove unused function to reduce binary size".to_string(),
        };
        
        assert_eq!(recommendation.recommendation_type, OptimizationType::DeadCodeElimination);
        assert_eq!(recommendation.confidence, 0.95);
        assert_eq!(recommendation.functions.len(), 1);
    }

    #[test]
    fn test_runtime_impact_types() {
        assert_eq!(RuntimeImpact::Neutral, RuntimeImpact::Neutral);
        assert!(matches!(RuntimeImpact::Positive(0.1), RuntimeImpact::Positive(_)));
        assert!(matches!(RuntimeImpact::Negative(0.05), RuntimeImpact::Negative(_)));
    }
}