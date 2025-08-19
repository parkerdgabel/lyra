//! Import Analysis and Optimization Engine
//!
//! Analyzes import patterns and optimizes imports based on actual usage patterns
//! from the tree-shaking analysis. Enables selective imports and compile-time resolution.

use super::{DependencyGraph, UsageTracker, TreeShakeError};
use crate::modules::registry::ModuleRegistry;
use std::collections::{HashMap, HashSet};
use serde::{Serialize, Deserialize};
use std::time::SystemTime;

/// Analyzes and optimizes import patterns based on usage data
pub struct ImportAnalyzer {
    /// Configuration for import analysis
    config: ImportAnalyzerConfig,
    
    /// Import pattern database
    import_patterns: HashMap<String, ImportPattern>,
    
    /// Module import optimization results
    optimization_results: HashMap<String, ModuleImportOptimization>,
    
    /// Import dependency graph
    import_graph: ImportDependencyGraph,
    
    /// Performance metrics
    performance_metrics: ImportAnalysisMetrics,
}

/// Configuration for import analysis
#[derive(Debug, Clone)]
pub struct ImportAnalyzerConfig {
    /// Enable aggressive import optimization
    pub aggressive_optimization: bool,
    
    /// Minimum usage threshold for imports
    pub min_usage_threshold: u64,
    
    /// Enable selective import generation
    pub enable_selective_imports: bool,
    
    /// Enable dead import elimination
    pub enable_dead_import_elimination: bool,
    
    /// Maximum number of selective imports per module
    pub max_selective_imports: usize,
    
    /// Enable import bundling optimization
    pub enable_import_bundling: bool,
    
    /// Threshold for bundling imports together
    pub bundling_threshold: f64,
    
    /// Enable compile-time import resolution
    pub enable_compile_time_resolution: bool,
}

impl Default for ImportAnalyzerConfig {
    fn default() -> Self {
        ImportAnalyzerConfig {
            aggressive_optimization: false,
            min_usage_threshold: 1,
            enable_selective_imports: true,
            enable_dead_import_elimination: true,
            max_selective_imports: 20,
            enable_import_bundling: true,
            bundling_threshold: 0.8,
            enable_compile_time_resolution: true,
        }
    }
}

/// Import pattern analysis for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportPattern {
    /// Module being imported from
    pub module_name: String,
    
    /// Type of import pattern
    pub pattern_type: ImportPatternType,
    
    /// Functions actually used from this module
    pub used_functions: Vec<FunctionUsage>,
    
    /// Functions imported but never used
    pub unused_functions: Vec<String>,
    
    /// Import efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,
    
    /// Suggested optimization
    pub optimization_suggestion: OptimizationSuggestion,
    
    /// Impact of optimization
    pub optimization_impact: ImportOptimizationImpact,
    
    /// Import frequency and patterns
    pub usage_patterns: ImportUsagePatterns,
}

/// Types of import patterns
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportPatternType {
    /// Full module import (use std::math::*)
    FullModule,
    
    /// Selective function imports (use std::math::{Sin, Cos})
    Selective { functions: Vec<String> },
    
    /// Single function import (use std::math::Sin)
    Single { function: String },
    
    /// Re-export pattern (pub use std::math::*)
    ReExport,
    
    /// Conditional import (only in certain contexts)
    Conditional,
    
    /// Unused import (imported but never used)
    Unused,
}

/// Function usage within an import
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionUsage {
    /// Function name
    pub function_name: String,
    
    /// Number of times used
    pub usage_count: u64,
    
    /// Contexts where it's used
    pub usage_contexts: Vec<UsageContext>,
    
    /// Whether it's on critical path
    pub is_critical: bool,
    
    /// Performance impact of this function
    pub performance_impact: f64,
}

/// Context where a function is used
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageContext {
    /// Calling function
    pub caller: String,
    
    /// Call frequency
    pub frequency: u64,
    
    /// Whether it's in a loop
    pub in_loop: bool,
    
    /// Whether it's conditional
    pub is_conditional: bool,
}

/// Optimization suggestion for imports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    /// Type of optimization
    pub optimization_type: ImportOptimizationType,
    
    /// Suggested import statement
    pub suggested_import: String,
    
    /// Confidence in the suggestion (0.0 to 1.0)
    pub confidence: f64,
    
    /// Expected benefits
    pub expected_benefits: Vec<String>,
    
    /// Potential risks
    pub potential_risks: Vec<String>,
}

/// Types of import optimizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportOptimizationType {
    /// Convert to selective imports
    ConvertToSelective,
    
    /// Remove unused imports
    RemoveUnused,
    
    /// Bundle related imports
    BundleImports,
    
    /// Split large imports
    SplitImports,
    
    /// Lazy load infrequently used functions
    LazyLoad,
    
    /// Inline small functions
    InlineFunctions,
    
    /// No optimization needed
    NoOptimization,
}

/// Impact of optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ImportOptimizationImpact {
    /// Estimated compilation time improvement (ms)
    pub compile_time_improvement: i64,
    
    /// Estimated binary size reduction (bytes)
    pub binary_size_reduction: i64,
    
    /// Estimated memory usage reduction (bytes)
    pub memory_usage_reduction: i64,
    
    /// Estimated startup time improvement (ms)
    pub startup_time_improvement: i64,
    
    /// Number of eliminated imports
    pub eliminated_imports: usize,
    
    /// Percentage reduction in imports
    pub import_reduction_percentage: f64,
}

/// Usage patterns for imports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportUsagePatterns {
    /// Peak usage periods
    pub peak_usage: Vec<UsagePeak>,
    
    /// Usage trend over time
    pub trend: UsageTrend,
    
    /// Co-usage patterns (functions used together)
    pub co_usage_patterns: HashMap<String, Vec<String>>,
    
    /// Temporal usage patterns
    pub temporal_patterns: TemporalUsagePattern,
}

/// Peak usage detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsagePeak {
    /// Time of peak
    pub timestamp: SystemTime,
    
    /// Peak value
    pub peak_value: u64,
    
    /// Duration of peak
    pub duration_seconds: u64,
    
    /// Functions involved in peak
    pub functions: Vec<String>,
}

/// Usage trend analysis
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UsageTrend {
    /// Usage increasing over time
    Increasing { rate: f64 },
    
    /// Usage decreasing over time
    Decreasing { rate: f64 },
    
    /// Usage is stable
    Stable,
    
    /// Usage is sporadic
    Sporadic,
    
    /// Insufficient data
    Unknown,
}

/// Temporal usage patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalUsagePattern {
    /// Usage distribution by time of day
    pub hourly_distribution: Vec<u64>,
    
    /// Usage spikes
    pub spikes: Vec<UsageSpike>,
    
    /// Quiet periods (low usage)
    pub quiet_periods: Vec<QuietPeriod>,
}

/// Usage spike detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageSpike {
    /// Start time of spike
    pub start_time: SystemTime,
    
    /// Duration of spike
    pub duration_seconds: u64,
    
    /// Magnitude of spike
    pub magnitude: f64,
    
    /// Functions involved
    pub functions: Vec<String>,
}

/// Quiet period detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuietPeriod {
    /// Start time of quiet period
    pub start_time: SystemTime,
    
    /// Duration of quiet period
    pub duration_seconds: u64,
    
    /// Functions not used during this period
    pub unused_functions: Vec<String>,
}

/// Module import optimization results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleImportOptimization {
    /// Module name
    pub module_name: String,
    
    /// Original import pattern
    pub original_pattern: ImportPattern,
    
    /// Optimized import pattern
    pub optimized_pattern: ImportPattern,
    
    /// Optimization applied
    pub optimization: OptimizationSuggestion,
    
    /// Actual impact achieved
    pub actual_impact: ImportOptimizationImpact,
    
    /// Validation results
    pub validation: OptimizationValidation,
    
    /// Timestamp of optimization
    pub optimized_at: SystemTime,
}

/// Validation of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationValidation {
    /// Whether optimization is valid
    pub is_valid: bool,
    
    /// Validation errors if any
    pub errors: Vec<String>,
    
    /// Validation warnings
    pub warnings: Vec<String>,
    
    /// Performance regression detected
    pub performance_regression: Option<f64>,
    
    /// Functions that became unavailable
    pub missing_functions: Vec<String>,
}

/// Import dependency graph
#[derive(Debug, Clone, Default)]
pub struct ImportDependencyGraph {
    /// Modules and their import relationships
    pub modules: HashMap<String, ModuleImportNode>,
    
    /// Import edges between modules
    pub edges: Vec<ImportEdge>,
    
    /// Circular import detection
    pub circular_imports: Vec<Vec<String>>,
    
    /// Import resolution order
    pub resolution_order: Vec<String>,
}

/// Node in import dependency graph
#[derive(Debug, Clone)]
pub struct ModuleImportNode {
    /// Module name
    pub module_name: String,
    
    /// Modules this module imports from
    pub imports_from: HashSet<String>,
    
    /// Modules that import from this module
    pub imported_by: HashSet<String>,
    
    /// Functions exported by this module
    pub exports: HashSet<String>,
    
    /// Functions imported by this module
    pub imports: HashSet<String>,
    
    /// Import weight (usage frequency)
    pub import_weight: f64,
}

/// Edge in import dependency graph
#[derive(Debug, Clone)]
pub struct ImportEdge {
    /// Source module
    pub from_module: String,
    
    /// Target module
    pub to_module: String,
    
    /// Functions imported
    pub functions: Vec<String>,
    
    /// Import strength (usage frequency)
    pub strength: f64,
    
    /// Whether this is a critical import
    pub is_critical: bool,
}

/// Performance metrics for import analysis
#[derive(Debug, Clone, Default)]
pub struct ImportAnalysisMetrics {
    /// Time taken for analysis
    pub analysis_time: std::time::Duration,
    
    /// Memory used during analysis
    pub memory_used: usize,
    
    /// Number of patterns analyzed
    pub patterns_analyzed: usize,
    
    /// Number of optimizations found
    pub optimizations_found: usize,
    
    /// Total potential savings identified
    pub total_potential_savings: ImportOptimizationImpact,
}

impl ImportAnalyzer {
    /// Create a new import analyzer
    pub fn new() -> Self {
        ImportAnalyzer {
            config: ImportAnalyzerConfig::default(),
            import_patterns: HashMap::new(),
            optimization_results: HashMap::new(),
            import_graph: ImportDependencyGraph::default(),
            performance_metrics: ImportAnalysisMetrics::default(),
        }
    }
    
    /// Create import analyzer with custom configuration
    pub fn with_config(config: ImportAnalyzerConfig) -> Self {
        ImportAnalyzer {
            config,
            import_patterns: HashMap::new(),
            optimization_results: HashMap::new(),
            import_graph: ImportDependencyGraph::default(),
            performance_metrics: ImportAnalysisMetrics::default(),
        }
    }
    
    /// Analyze import patterns across the stdlib
    pub fn analyze_imports(
        &mut self,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
        module_registry: &ModuleRegistry,
    ) -> Result<(), TreeShakeError> {
        let start_time = std::time::Instant::now();
        
        // Step 1: Build import dependency graph
        self.build_import_graph(module_registry)?;
        
        // Step 2: Analyze import patterns for each module
        self.analyze_module_import_patterns(dependency_graph, usage_tracker)?;
        
        // Step 3: Detect optimization opportunities
        self.detect_optimization_opportunities()?;
        
        // Step 4: Generate optimization suggestions
        self.generate_optimization_suggestions()?;
        
        // Step 5: Validate optimizations
        self.validate_optimizations(dependency_graph)?;
        
        // Update metrics
        self.performance_metrics.analysis_time = start_time.elapsed();
        self.performance_metrics.patterns_analyzed = self.import_patterns.len();
        self.performance_metrics.optimizations_found = self.optimization_results.len();
        
        Ok(())
    }
    
    /// Get import pattern for a module
    pub fn get_import_pattern(&self, module_name: &str) -> Option<&ImportPattern> {
        self.import_patterns.get(module_name)
    }
    
    /// Get optimization results for a module
    pub fn get_optimization_results(&self, module_name: &str) -> Option<&ModuleImportOptimization> {
        self.optimization_results.get(module_name)
    }
    
    /// Get all optimization suggestions
    pub fn get_all_optimizations(&self) -> Vec<&ModuleImportOptimization> {
        self.optimization_results.values().collect()
    }
    
    /// Get import dependency graph
    pub fn get_import_graph(&self) -> &ImportDependencyGraph {
        &self.import_graph
    }
    
    /// Get performance metrics
    pub fn get_performance_metrics(&self) -> &ImportAnalysisMetrics {
        &self.performance_metrics
    }
    
    /// Generate optimized import statements
    pub fn generate_optimized_imports(&self) -> Result<HashMap<String, String>, TreeShakeError> {
        let mut optimized_imports = HashMap::new();
        
        for (module_name, optimization) in &self.optimization_results {
            if optimization.validation.is_valid {
                optimized_imports.insert(
                    module_name.clone(),
                    optimization.optimization.suggested_import.clone(),
                );
            }
        }
        
        Ok(optimized_imports)
    }
    
    /// Calculate total optimization impact
    pub fn calculate_total_impact(&self) -> ImportOptimizationImpact {
        let mut total_impact = ImportOptimizationImpact {
            compile_time_improvement: 0,
            binary_size_reduction: 0,
            memory_usage_reduction: 0,
            startup_time_improvement: 0,
            eliminated_imports: 0,
            import_reduction_percentage: 0.0,
        };
        
        let mut valid_optimizations = 0;
        
        for optimization in self.optimization_results.values() {
            if optimization.validation.is_valid {
                total_impact.compile_time_improvement += optimization.actual_impact.compile_time_improvement;
                total_impact.binary_size_reduction += optimization.actual_impact.binary_size_reduction;
                total_impact.memory_usage_reduction += optimization.actual_impact.memory_usage_reduction;
                total_impact.startup_time_improvement += optimization.actual_impact.startup_time_improvement;
                total_impact.eliminated_imports += optimization.actual_impact.eliminated_imports;
                valid_optimizations += 1;
            }
        }
        
        if valid_optimizations > 0 {
            total_impact.import_reduction_percentage = 
                self.optimization_results.values()
                    .filter(|opt| opt.validation.is_valid)
                    .map(|opt| opt.actual_impact.import_reduction_percentage)
                    .sum::<f64>() / valid_optimizations as f64;
        }
        
        total_impact
    }
    
    // Private implementation methods
    
    fn build_import_graph(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        // Build nodes for each module
        for namespace in module_registry.list_modules() {
            if let Some(module) = module_registry.get_module(&namespace) {
                let mut node = ModuleImportNode {
                    module_name: namespace.clone(),
                    imports_from: HashSet::new(),
                    imported_by: HashSet::new(),
                    exports: HashSet::new(),
                    imports: HashSet::new(),
                    import_weight: 0.0,
                };
                
                // Add exports
                for export in module.exports.values() {
                    node.exports.insert(export.export_name.clone());
                }
                
                self.import_graph.modules.insert(namespace.clone(), node);
            }
        }
        
        // For now, we'll build a simple import graph
        // In a real implementation, this would analyze actual import statements
        
        Ok(())
    }
    
    fn analyze_module_import_patterns(
        &mut self,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<(), TreeShakeError> {
        for module_name in dependency_graph.get_module_names() {
            let pattern = self.analyze_single_module_pattern(&module_name, dependency_graph, usage_tracker)?;
            self.import_patterns.insert(module_name.clone(), pattern);
        }
        
        Ok(())
    }
    
    fn analyze_single_module_pattern(
        &self,
        module_name: &str,
        dependency_graph: &DependencyGraph,
        usage_tracker: &UsageTracker,
    ) -> Result<ImportPattern, TreeShakeError> {
        let mut used_functions = Vec::new();
        let mut unused_functions = Vec::new();
        
        // Get all functions in this module
        let module_functions = dependency_graph.get_functions_in_module(module_name);
        
        for function_name in &module_functions {
            if let Some(stats) = usage_tracker.get_function_stats(function_name) {
                if stats.call_count > 0 {
                    used_functions.push(FunctionUsage {
                        function_name: function_name.clone(),
                        usage_count: stats.call_count,
                        usage_contexts: vec![UsageContext {
                            caller: "unknown".to_string(),
                            frequency: stats.call_count,
                            in_loop: false,
                            is_conditional: false,
                        }],
                        is_critical: stats.is_on_critical_path,
                        performance_impact: stats.performance_impact.bottleneck_score,
                    });
                } else {
                    unused_functions.push(function_name.clone());
                }
            } else {
                unused_functions.push(function_name.clone());
            }
        }
        
        // Calculate efficiency score
        let total_functions = module_functions.len();
        let used_count = used_functions.len();
        let efficiency_score = if total_functions > 0 {
            used_count as f64 / total_functions as f64
        } else {
            1.0
        };
        
        // Determine pattern type and optimization
        let (pattern_type, optimization_suggestion) = self.determine_pattern_and_optimization(
            &used_functions,
            &unused_functions,
            efficiency_score,
        );
        
        Ok(ImportPattern {
            module_name: module_name.to_string(),
            pattern_type,
            used_functions: used_functions.clone(),
            unused_functions: unused_functions.clone(),
            efficiency_score,
            optimization_suggestion,
            optimization_impact: self.estimate_optimization_impact(&used_functions, &unused_functions),
            usage_patterns: ImportUsagePatterns {
                peak_usage: Vec::new(),
                trend: UsageTrend::Unknown,
                co_usage_patterns: HashMap::new(),
                temporal_patterns: TemporalUsagePattern {
                    hourly_distribution: vec![0; 24],
                    spikes: Vec::new(),
                    quiet_periods: Vec::new(),
                },
            },
        })
    }
    
    fn determine_pattern_and_optimization(
        &self,
        used_functions: &[FunctionUsage],
        unused_functions: &[String],
        efficiency_score: f64,
    ) -> (ImportPatternType, OptimizationSuggestion) {
        let used_count = used_functions.len();
        let unused_count = unused_functions.len();
        let total_count = used_count + unused_count;
        
        // Determine current pattern
        let pattern_type = if unused_count == total_count {
            ImportPatternType::Unused
        } else if used_count == 1 {
            ImportPatternType::Single {
                function: used_functions[0].function_name.clone(),
            }
        } else if used_count <= self.config.max_selective_imports {
            ImportPatternType::Selective {
                functions: used_functions.iter().map(|f| f.function_name.clone()).collect(),
            }
        } else {
            ImportPatternType::FullModule
        };
        
        // Determine optimization
        let optimization_type = if unused_count == total_count {
            ImportOptimizationType::RemoveUnused
        } else if efficiency_score < 0.5 && used_count <= self.config.max_selective_imports {
            ImportOptimizationType::ConvertToSelective
        } else if unused_count > 0 {
            ImportOptimizationType::RemoveUnused
        } else {
            ImportOptimizationType::NoOptimization
        };
        
        let suggested_import = self.generate_suggested_import(&optimization_type, used_functions);
        
        let optimization_suggestion = OptimizationSuggestion {
            optimization_type,
            suggested_import,
            confidence: if efficiency_score < 0.3 { 0.9 } else { 0.7 },
            expected_benefits: vec!["Reduced compilation time".to_string(), "Smaller binary size".to_string()],
            potential_risks: vec!["May need to add imports later".to_string()],
        };
        
        (pattern_type, optimization_suggestion)
    }
    
    fn generate_suggested_import(&self, optimization_type: &ImportOptimizationType, used_functions: &[FunctionUsage]) -> String {
        match optimization_type {
            ImportOptimizationType::RemoveUnused => "// Remove unused import".to_string(),
            ImportOptimizationType::ConvertToSelective => {
                let function_names: Vec<String> = used_functions.iter()
                    .map(|f| f.function_name.clone())
                    .collect();
                format!("use module::{{{}}}", function_names.join(", "))
            },
            ImportOptimizationType::NoOptimization => "use module::*".to_string(),
            _ => "// Optimization needed".to_string(),
        }
    }
    
    fn estimate_optimization_impact(
        &self,
        used_functions: &[FunctionUsage],
        unused_functions: &[String],
    ) -> ImportOptimizationImpact {
        let eliminated_imports = unused_functions.len();
        let total_functions = used_functions.len() + unused_functions.len();
        
        ImportOptimizationImpact {
            compile_time_improvement: (eliminated_imports * 10) as i64,
            binary_size_reduction: (eliminated_imports * 1000) as i64,
            memory_usage_reduction: (eliminated_imports * 500) as i64,
            startup_time_improvement: (eliminated_imports * 5) as i64,
            eliminated_imports,
            import_reduction_percentage: if total_functions > 0 {
                (eliminated_imports as f64 / total_functions as f64) * 100.0
            } else {
                0.0
            },
        }
    }
    
    fn detect_optimization_opportunities(&mut self) -> Result<(), TreeShakeError> {
        // This would detect patterns and opportunities across modules
        // For now, we'll use the per-module analysis
        Ok(())
    }
    
    fn generate_optimization_suggestions(&mut self) -> Result<(), TreeShakeError> {
        for (module_name, pattern) in &self.import_patterns {
            let optimization = ModuleImportOptimization {
                module_name: module_name.clone(),
                original_pattern: pattern.clone(),
                optimized_pattern: pattern.clone(), // Would be different in real implementation
                optimization: pattern.optimization_suggestion.clone(),
                actual_impact: pattern.optimization_impact.clone(),
                validation: OptimizationValidation {
                    is_valid: true,
                    errors: Vec::new(),
                    warnings: Vec::new(),
                    performance_regression: None,
                    missing_functions: Vec::new(),
                },
                optimized_at: SystemTime::now(),
            };
            
            self.optimization_results.insert(module_name.clone(), optimization);
        }
        
        Ok(())
    }
    
    fn validate_optimizations(&mut self, dependency_graph: &DependencyGraph) -> Result<(), TreeShakeError> {
        let mut validations = HashMap::new();
        
        // First, compute all validations without mutating self
        for (module_name, optimization) in &self.optimization_results {
            let validation = self.validate_single_optimization(optimization, dependency_graph)?;
            validations.insert(module_name.clone(), validation);
        }
        
        // Then update the validations
        for (module_name, validation) in validations {
            if let Some(optimization) = self.optimization_results.get_mut(&module_name) {
                optimization.validation = validation;
            }
        }
        
        Ok(())
    }
    
    fn validate_single_optimization(
        &self,
        optimization: &ModuleImportOptimization,
        dependency_graph: &DependencyGraph,
    ) -> Result<OptimizationValidation, TreeShakeError> {
        let mut validation = OptimizationValidation {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_regression: None,
            missing_functions: Vec::new(),
        };
        
        // Check if optimization would break dependencies
        for used_function in &optimization.original_pattern.used_functions {
            if let Some(_node) = dependency_graph.get_node(&used_function.function_name) {
                // Function exists in dependency graph, so it's valid
            } else {
                validation.missing_functions.push(used_function.function_name.clone());
                validation.is_valid = false;
            }
        }
        
        if !validation.missing_functions.is_empty() {
            validation.errors.push(format!(
                "Optimization would make {} functions unavailable",
                validation.missing_functions.len()
            ));
        }
        
        Ok(validation)
    }
}

impl Default for ImportAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_analyzer_creation() {
        let analyzer = ImportAnalyzer::new();
        assert_eq!(analyzer.import_patterns.len(), 0);
        assert_eq!(analyzer.optimization_results.len(), 0);
    }

    #[test]
    fn test_import_analyzer_config() {
        let config = ImportAnalyzerConfig {
            aggressive_optimization: true,
            min_usage_threshold: 5,
            max_selective_imports: 10,
            ..Default::default()
        };
        
        let analyzer = ImportAnalyzer::with_config(config);
        assert!(analyzer.config.aggressive_optimization);
        assert_eq!(analyzer.config.min_usage_threshold, 5);
        assert_eq!(analyzer.config.max_selective_imports, 10);
    }

    #[test]
    fn test_optimization_impact_calculation() {
        let impact = ImportOptimizationImpact {
            compile_time_improvement: 100,
            binary_size_reduction: 2000,
            memory_usage_reduction: 1000,
            startup_time_improvement: 50,
            eliminated_imports: 5,
            import_reduction_percentage: 25.0,
        };
        
        assert_eq!(impact.eliminated_imports, 5);
        assert_eq!(impact.import_reduction_percentage, 25.0);
    }

    #[test]
    fn test_import_pattern_types() {
        let unused = ImportPatternType::Unused;
        let single = ImportPatternType::Single { function: "Sin".to_string() };
        let selective = ImportPatternType::Selective { functions: vec!["Sin".to_string(), "Cos".to_string()] };
        
        assert_eq!(unused, ImportPatternType::Unused);
        assert!(matches!(single, ImportPatternType::Single { .. }));
        assert!(matches!(selective, ImportPatternType::Selective { .. }));
    }

    #[test]
    fn test_optimization_types() {
        assert_eq!(ImportOptimizationType::RemoveUnused, ImportOptimizationType::RemoveUnused);
        assert_ne!(ImportOptimizationType::RemoveUnused, ImportOptimizationType::ConvertToSelective);
    }

    #[test]
    fn test_usage_trend_variants() {
        let increasing = UsageTrend::Increasing { rate: 0.1 };
        let decreasing = UsageTrend::Decreasing { rate: 0.05 };
        let stable = UsageTrend::Stable;
        
        assert!(matches!(increasing, UsageTrend::Increasing { .. }));
        assert!(matches!(decreasing, UsageTrend::Decreasing { .. }));
        assert_eq!(stable, UsageTrend::Stable);
    }

    #[test]
    fn test_import_dependency_graph() {
        let mut graph = ImportDependencyGraph::default();
        assert_eq!(graph.modules.len(), 0);
        assert_eq!(graph.edges.len(), 0);
        
        let node = ModuleImportNode {
            module_name: "std::math".to_string(),
            imports_from: HashSet::new(),
            imported_by: HashSet::new(),
            exports: HashSet::new(),
            imports: HashSet::new(),
            import_weight: 1.0,
        };
        
        graph.modules.insert("std::math".to_string(), node);
        assert_eq!(graph.modules.len(), 1);
    }

    #[test]
    fn test_optimization_validation() {
        let validation = OptimizationValidation {
            is_valid: true,
            errors: Vec::new(),
            warnings: vec!["Minor warning".to_string()],
            performance_regression: None,
            missing_functions: Vec::new(),
        };
        
        assert!(validation.is_valid);
        assert_eq!(validation.errors.len(), 0);
        assert_eq!(validation.warnings.len(), 1);
    }
}