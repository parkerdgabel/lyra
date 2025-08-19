//! Module-Level Dependency Tracking
//!
//! Tracks dependencies between modules in the stdlib hierarchy for tree-shaking optimization.

use crate::modules::Module;
use crate::modules::registry::ModuleRegistry;
use super::{DependencyGraph, TreeShakeError};
use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Serialize, Deserialize};

/// Analyzes module-level dependencies for tree-shaking
pub struct ModuleDependencyAnalyzer {
    /// Module dependency graph
    module_graph: ModuleDependencyGraph,
    
    /// Import/export analysis
    import_export_analyzer: ImportExportAnalyzer,
    
    /// Cross-module call tracking
    cross_module_calls: CrossModuleCallTracker,
    
    /// Configuration
    config: ModuleAnalyzerConfig,
}

/// Module dependency graph structure
#[derive(Debug, Clone)]
pub struct ModuleDependencyGraph {
    /// Module nodes
    modules: HashMap<String, ModuleNode>,
    
    /// Dependencies between modules (module -> modules it depends on)
    dependencies: HashMap<String, HashSet<String>>,
    
    /// Reverse dependencies (module -> modules that depend on it)
    dependents: HashMap<String, HashSet<String>>,
    
    /// Circular dependency groups
    circular_groups: Vec<Vec<String>>,
    
    /// Topological ordering of modules
    topological_order: Vec<String>,
    
    /// Module hierarchy levels
    hierarchy_levels: HashMap<String, u32>,
}

/// Information about a module in the dependency graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleNode {
    /// Module name
    pub name: String,
    
    /// Module type (stdlib, external, etc.)
    pub module_type: ModuleType,
    
    /// Functions exported by this module
    pub exports: Vec<String>,
    
    /// Functions imported from other modules
    pub imports: HashMap<String, Vec<String>>, // module -> functions
    
    /// Module size metrics
    pub size_metrics: ModuleSizeMetrics,
    
    /// Usage patterns
    pub usage_patterns: ModuleUsagePatterns,
    
    /// Optimization opportunities
    pub optimization_opportunities: Vec<ModuleOptimization>,
    
    /// Whether module is essential (cannot be removed)
    pub is_essential: bool,
    
    /// Whether module is a leaf (no dependencies)
    pub is_leaf: bool,
    
    /// Whether module is a root (nothing depends on it)
    pub is_root: bool,
}

/// Types of modules in the system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleType {
    /// Core stdlib module
    StdlibCore,
    
    /// Extended stdlib module
    StdlibExtended,
    
    /// Machine learning module
    MachineLearning,
    
    /// User-defined module
    UserDefined,
    
    /// External dependency
    External,
    
    /// Built-in language module
    Builtin,
}

/// Size metrics for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSizeMetrics {
    /// Number of functions
    pub function_count: usize,
    
    /// Total estimated size in bytes
    pub estimated_size: usize,
    
    /// Lines of code
    pub lines_of_code: usize,
    
    /// Binary size contribution
    pub binary_size_contribution: usize,
    
    /// Memory usage at runtime
    pub runtime_memory_usage: usize,
    
    /// Compilation time
    pub compilation_time: std::time::Duration,
}

/// Usage patterns for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleUsagePatterns {
    /// How often module is imported
    pub import_frequency: f64,
    
    /// Which functions are most used
    pub popular_functions: Vec<(String, u64)>,
    
    /// Which functions are never used
    pub unused_functions: Vec<String>,
    
    /// Partial import patterns (only some functions used)
    pub partial_usage: PartialUsagePattern,
    
    /// Seasonal usage patterns
    pub seasonal_patterns: Vec<f64>,
    
    /// Co-import patterns (modules imported together)
    pub co_import_patterns: HashMap<String, f64>,
}

/// Partial usage pattern for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialUsagePattern {
    /// Percentage of functions actually used
    pub usage_percentage: f64,
    
    /// Functions that could be split into separate module
    pub splittable_functions: Vec<String>,
    
    /// Functions that should stay together
    pub cohesive_groups: Vec<Vec<String>>,
    
    /// Import optimization recommendations
    pub import_optimizations: Vec<ImportOptimization>,
}

/// Import optimization recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportOptimization {
    /// Type of optimization
    pub optimization_type: ImportOptimizationType,
    
    /// Functions involved
    pub functions: Vec<String>,
    
    /// Expected size reduction
    pub size_reduction: usize,
    
    /// Confidence level
    pub confidence: f64,
    
    /// Description
    pub description: String,
}

/// Types of import optimizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImportOptimizationType {
    /// Replace full import with selective imports
    SelectiveImport,
    
    /// Combine imports from same module
    CombineImports,
    
    /// Split module to reduce import size
    SplitModule,
    
    /// Inline small frequently used functions
    InlineSmallFunctions,
    
    /// Lazy load rarely used functions
    LazyLoad,
    
    /// Replace with lighter alternative
    LighterAlternative,
}

/// Module optimization opportunity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleOptimization {
    /// Type of optimization
    pub optimization_type: ModuleOptimizationType,
    
    /// Expected impact
    pub expected_impact: OptimizationImpact,
    
    /// Implementation complexity
    pub complexity: OptimizationComplexity,
    
    /// Prerequisites
    pub prerequisites: Vec<String>,
    
    /// Risk assessment
    pub risk_level: RiskLevel,
}

/// Types of module optimizations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModuleOptimizationType {
    /// Remove unused functions
    DeadCodeElimination,
    
    /// Split large module
    ModuleSplitting,
    
    /// Merge small related modules
    ModuleMerging,
    
    /// Reorganize function layout
    FunctionReorganization,
    
    /// Extract common dependencies
    DependencyExtraction,
    
    /// Optimize import chains
    ImportChainOptimization,
}

/// Expected impact of optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationImpact {
    /// Binary size reduction (bytes)
    pub binary_size_reduction: usize,
    
    /// Compilation time reduction (ms)
    pub compilation_time_reduction: u64,
    
    /// Runtime memory reduction (bytes)
    pub memory_reduction: usize,
    
    /// Performance improvement (percentage)
    pub performance_improvement: f64,
    
    /// Maintainability impact
    pub maintainability_impact: MaintainabilityImpact,
}

/// Impact on maintainability
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaintainabilityImpact {
    /// Improves maintainability
    Positive,
    
    /// No impact on maintainability
    Neutral,
    
    /// Minor negative impact
    SlightlyNegative,
    
    /// Significant negative impact
    Negative,
}

/// Complexity of implementing optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizationComplexity {
    /// Simple to implement
    Low,
    
    /// Moderate complexity
    Medium,
    
    /// High complexity
    High,
    
    /// Very complex, requires major changes
    VeryHigh,
}

/// Risk level of optimization
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RiskLevel {
    /// Low risk
    Low,
    
    /// Medium risk
    Medium,
    
    /// High risk
    High,
    
    /// Very high risk
    Critical,
}

/// Analyzes import/export patterns
#[derive(Debug, Clone)]
pub struct ImportExportAnalyzer {
    /// Import patterns per module
    import_patterns: HashMap<String, ImportPattern>,
    
    /// Export usage tracking
    export_usage: HashMap<String, ExportUsage>,
    
    /// Cross-module relationships
    cross_module_relationships: Vec<CrossModuleRelationship>,
}

/// Import pattern for a module
#[derive(Debug, Clone)]
pub struct ImportPattern {
    /// Module doing the importing
    pub importing_module: String,
    
    /// What is being imported
    pub imports: Vec<ImportSpec>,
    
    /// Import frequency
    pub frequency: f64,
    
    /// Import necessity (how critical)
    pub necessity: ImportNecessity,
}

/// Specification of what is imported
#[derive(Debug, Clone)]
pub struct ImportSpec {
    /// Source module
    pub from_module: String,
    
    /// Specific functions (empty means full import)
    pub functions: Vec<String>,
    
    /// Import type
    pub import_type: ImportType,
    
    /// Usage frequency of this import
    pub usage_frequency: f64,
}

/// Types of imports
#[derive(Debug, Clone, PartialEq)]
pub enum ImportType {
    /// Full module import
    Full,
    
    /// Selective function import
    Selective,
    
    /// Wildcard import
    Wildcard,
    
    /// Re-export import
    ReExport,
}

/// How necessary an import is
#[derive(Debug, Clone, PartialEq)]
pub enum ImportNecessity {
    /// Absolutely required
    Essential,
    
    /// Commonly used
    Common,
    
    /// Occasionally used
    Occasional,
    
    /// Rarely used
    Rare,
    
    /// Never actually used
    Unused,
}

/// Export usage tracking
#[derive(Debug, Clone)]
pub struct ExportUsage {
    /// Function being exported
    pub function_name: String,
    
    /// Module exporting it
    pub exporting_module: String,
    
    /// Modules that import it
    pub importing_modules: Vec<String>,
    
    /// Usage frequency
    pub usage_frequency: u64,
    
    /// Whether it's a public API
    pub is_public_api: bool,
    
    /// Whether it can be removed
    pub can_be_removed: bool,
}

/// Relationship between modules
#[derive(Debug, Clone)]
pub struct CrossModuleRelationship {
    /// Source module
    pub from_module: String,
    
    /// Target module
    pub to_module: String,
    
    /// Relationship type
    pub relationship_type: RelationshipType,
    
    /// Strength of relationship (0.0 to 1.0)
    pub strength: f64,
    
    /// Functions involved in relationship
    pub involved_functions: Vec<String>,
}

/// Types of relationships between modules
#[derive(Debug, Clone, PartialEq)]
pub enum RelationshipType {
    /// Direct dependency
    DirectDependency,
    
    /// Mutual dependency (circular)
    MutualDependency,
    
    /// Transitive dependency
    TransitiveDependency,
    
    /// Optional dependency
    OptionalDependency,
    
    /// Alternative implementations
    Alternative,
    
    /// Complementary functionality
    Complementary,
}

/// Tracks function calls across module boundaries
#[derive(Debug, Clone)]
pub struct CrossModuleCallTracker {
    /// Calls from one module to another
    cross_calls: HashMap<(String, String), Vec<CrossModuleCall>>,
    
    /// Hot paths across modules
    hot_paths: Vec<CrossModulePath>,
    
    /// Module interfaces
    module_interfaces: HashMap<String, ModuleInterface>,
}

/// A function call across module boundaries
#[derive(Debug, Clone)]
pub struct CrossModuleCall {
    /// Calling function
    pub caller: String,
    
    /// Calling module
    pub caller_module: String,
    
    /// Called function
    pub callee: String,
    
    /// Called module
    pub callee_module: String,
    
    /// Call frequency
    pub frequency: u64,
    
    /// Call context
    pub context: ModuleCallContext,
    
    /// Performance impact
    pub performance_impact: f64,
}

/// Path of calls across multiple modules
#[derive(Debug, Clone)]
pub struct CrossModulePath {
    /// Modules in the path
    pub modules: Vec<String>,
    
    /// Functions in the path
    pub functions: Vec<String>,
    
    /// Frequency this path is taken
    pub frequency: u64,
    
    /// Performance characteristics
    pub performance: PathPerformance,
}

/// Performance characteristics of a cross-module path
#[derive(Debug, Clone)]
pub struct PathPerformance {
    /// Total execution time
    pub total_time: std::time::Duration,
    
    /// Module switching overhead
    pub switching_overhead: std::time::Duration,
    
    /// Memory overhead
    pub memory_overhead: usize,
    
    /// Cache impact
    pub cache_impact: f64,
}

/// Interface definition for a module
#[derive(Debug, Clone)]
pub struct ModuleInterface {
    /// Module name
    pub module_name: String,
    
    /// Public functions
    pub public_functions: Vec<String>,
    
    /// Private functions
    pub private_functions: Vec<String>,
    
    /// Interface stability
    pub stability: InterfaceStability,
    
    /// Compatibility guarantees
    pub compatibility: CompatibilityLevel,
}

/// Stability of module interface
#[derive(Debug, Clone, PartialEq)]
pub enum InterfaceStability {
    /// Interface is stable and won't change
    Stable,
    
    /// Interface is mostly stable with minor changes
    MostlyStable,
    
    /// Interface is evolving
    Evolving,
    
    /// Interface is experimental
    Experimental,
    
    /// Interface is deprecated
    Deprecated,
}

/// Compatibility level
#[derive(Debug, Clone, PartialEq)]
pub enum CompatibilityLevel {
    /// Full backward compatibility
    FullBackward,
    
    /// Source compatibility only
    SourceOnly,
    
    /// Binary compatibility only
    BinaryOnly,
    
    /// No compatibility guarantees
    None,
}

/// Call context for cross-module calls
#[derive(Debug, Clone)]
pub struct ModuleCallContext {
    /// Call stack depth
    pub depth: u32,
    
    /// Whether in performance-critical section
    pub is_critical: bool,
    
    /// Whether in error handling
    pub is_error_handling: bool,
    
    /// Loop context
    pub loop_context: Option<LoopContext>,
}

/// Loop context information
#[derive(Debug, Clone)]
pub struct LoopContext {
    /// Nesting level
    pub nesting_level: u32,
    
    /// Estimated iteration count
    pub iteration_count: Option<u64>,
    
    /// Loop type
    pub loop_type: LoopType,
}

/// Types of loops
#[derive(Debug, Clone, PartialEq)]
pub enum LoopType {
    /// Simple for loop
    For,
    
    /// While loop
    While,
    
    /// Recursive loop
    Recursive,
    
    /// Iterator-based loop
    Iterator,
}

/// Configuration for module analyzer
#[derive(Debug, Clone)]
pub struct ModuleAnalyzerConfig {
    /// Enable deep dependency analysis
    pub enable_deep_analysis: bool,
    
    /// Maximum dependency depth to analyze
    pub max_dependency_depth: u32,
    
    /// Enable cross-module call tracking
    pub enable_call_tracking: bool,
    
    /// Minimum usage threshold for reporting
    pub min_usage_threshold: f64,
    
    /// Enable optimization suggestions
    pub enable_optimization_suggestions: bool,
    
    /// Risk tolerance for optimizations
    pub risk_tolerance: RiskLevel,
}

impl Default for ModuleAnalyzerConfig {
    fn default() -> Self {
        ModuleAnalyzerConfig {
            enable_deep_analysis: true,
            max_dependency_depth: 5,
            enable_call_tracking: true,
            min_usage_threshold: 0.01,
            enable_optimization_suggestions: true,
            risk_tolerance: RiskLevel::Medium,
        }
    }
}

impl ModuleDependencyAnalyzer {
    /// Create a new module dependency analyzer
    pub fn new() -> Self {
        ModuleDependencyAnalyzer {
            module_graph: ModuleDependencyGraph::new(),
            import_export_analyzer: ImportExportAnalyzer::new(),
            cross_module_calls: CrossModuleCallTracker::new(),
            config: ModuleAnalyzerConfig::default(),
        }
    }
    
    /// Create analyzer with custom configuration
    pub fn with_config(config: ModuleAnalyzerConfig) -> Self {
        ModuleDependencyAnalyzer {
            module_graph: ModuleDependencyGraph::new(),
            import_export_analyzer: ImportExportAnalyzer::new(),
            cross_module_calls: CrossModuleCallTracker::new(),
            config,
        }
    }
    
    /// Analyze module dependencies in the registry
    pub fn analyze_module_dependencies(
        &mut self,
        module_registry: &ModuleRegistry,
        dependency_graph: &mut DependencyGraph,
    ) -> Result<(), TreeShakeError> {
        // Step 1: Build module graph
        self.build_module_graph(module_registry)?;
        
        // Step 2: Analyze import/export patterns
        self.analyze_import_export_patterns(module_registry)?;
        
        // Step 3: Track cross-module calls
        if self.config.enable_call_tracking {
            self.track_cross_module_calls(dependency_graph)?;
        }
        
        // Step 4: Detect optimization opportunities
        if self.config.enable_optimization_suggestions {
            self.detect_optimization_opportunities()?;
        }
        
        // Step 5: Update dependency graph with module info
        self.update_dependency_graph_with_module_info(dependency_graph)?;
        
        Ok(())
    }
    
    /// Get module dependency graph
    pub fn get_module_graph(&self) -> &ModuleDependencyGraph {
        &self.module_graph
    }
    
    /// Get optimization opportunities for a module
    pub fn get_module_optimizations(&self, module_name: &str) -> Vec<&ModuleOptimization> {
        if let Some(module_node) = self.module_graph.modules.get(module_name) {
            module_node.optimization_opportunities.iter().collect()
        } else {
            Vec::new()
        }
    }
    
    /// Get cross-module call hot paths
    pub fn get_hot_paths(&self) -> &Vec<CrossModulePath> {
        &self.cross_module_calls.hot_paths
    }
    
    /// Get modules that can be safely removed
    pub fn get_removable_modules(&self) -> Vec<String> {
        self.module_graph.modules.iter()
            .filter(|(_, node)| !node.is_essential && node.is_root)
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Get circular dependencies
    pub fn get_circular_dependencies(&self) -> &Vec<Vec<String>> {
        &self.module_graph.circular_groups
    }
    
    // Private implementation methods
    
    fn build_module_graph(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        // Add all modules as nodes
        for namespace in module_registry.list_modules() {
            if let Some(module) = module_registry.get_module(&namespace) {
                let module_node = self.create_module_node(&namespace, &module)?;
                self.module_graph.modules.insert(namespace.clone(), module_node);
            }
        }
        
        // Analyze dependencies between modules
        self.analyze_module_dependencies_internal()?;
        
        // Compute topological order
        self.compute_module_topological_order()?;
        
        // Detect circular dependencies
        self.detect_circular_dependencies()?;
        
        // Compute hierarchy levels
        self.compute_hierarchy_levels()?;
        
        Ok(())
    }
    
    fn create_module_node(&self, namespace: &str, module: &Module) -> Result<ModuleNode, TreeShakeError> {
        let exports: Vec<String> = module.exports.values()
            .map(|export| export.export_name.clone())
            .collect();
        
        let module_type = self.determine_module_type(namespace);
        let is_essential = self.is_essential_module(namespace);
        
        Ok(ModuleNode {
            name: namespace.to_string(),
            module_type,
            exports: exports.clone(),
            imports: HashMap::new(), // Will be filled in later
            size_metrics: self.calculate_module_size_metrics(&exports),
            usage_patterns: self.analyze_module_usage_patterns(namespace, &exports),
            optimization_opportunities: Vec::new(), // Will be filled in later
            is_essential,
            is_leaf: false, // Will be determined after dependency analysis
            is_root: false, // Will be determined after dependency analysis
        })
    }
    
    fn determine_module_type(&self, namespace: &str) -> ModuleType {
        match namespace {
            ns if ns.starts_with("std::math") || ns.starts_with("std::list") || ns.starts_with("std::string") => {
                ModuleType::StdlibCore
            }
            ns if ns.starts_with("std::tensor") || ns.starts_with("std::table") || ns.starts_with("std::rules") => {
                ModuleType::StdlibExtended
            }
            ns if ns.starts_with("std::ml") => {
                ModuleType::MachineLearning
            }
            _ => ModuleType::UserDefined,
        }
    }
    
    fn is_essential_module(&self, namespace: &str) -> bool {
        // Core stdlib modules are essential
        matches!(namespace, 
            "std::math" | "std::list" | "std::string" | "std::tensor"
        )
    }
    
    fn calculate_module_size_metrics(&self, exports: &[String]) -> ModuleSizeMetrics {
        ModuleSizeMetrics {
            function_count: exports.len(),
            estimated_size: exports.len() * 500, // Rough estimate
            lines_of_code: exports.len() * 25,   // Rough estimate
            binary_size_contribution: exports.len() * 1024,
            runtime_memory_usage: exports.len() * 256,
            compilation_time: std::time::Duration::from_millis(exports.len() as u64 * 10),
        }
    }
    
    fn analyze_module_usage_patterns(&self, namespace: &str, exports: &[String]) -> ModuleUsagePatterns {
        // This would be populated with real usage data in a full implementation
        ModuleUsagePatterns {
            import_frequency: 0.5,
            popular_functions: exports.iter().take(3)
                .map(|name| (name.clone(), 100))
                .collect(),
            unused_functions: Vec::new(),
            partial_usage: PartialUsagePattern {
                usage_percentage: 0.8,
                splittable_functions: Vec::new(),
                cohesive_groups: Vec::new(),
                import_optimizations: Vec::new(),
            },
            seasonal_patterns: vec![1.0; 12], // Monthly usage
            co_import_patterns: HashMap::new(),
        }
    }
    
    fn analyze_module_dependencies_internal(&mut self) -> Result<(), TreeShakeError> {
        // Analyze which modules depend on which others based on function calls
        // This is a simplified implementation that would be more sophisticated in practice
        
        for (module_name, module_node) in &self.module_graph.modules {
            let mut dependencies = HashSet::new();
            
            // Analyze function dependencies to determine module dependencies
            for function_name in &module_node.exports {
                let deps = self.get_function_module_dependencies(function_name);
                dependencies.extend(deps);
            }
            
            // Remove self-dependency
            dependencies.remove(module_name);
            
            self.module_graph.dependencies.insert(module_name.clone(), dependencies.clone());
            
            // Update reverse dependencies
            for dep in dependencies {
                self.module_graph.dependents.entry(dep)
                    .or_insert_with(HashSet::new)
                    .insert(module_name.clone());
            }
        }
        
        // Update leaf and root status
        for (module_name, module_node) in self.module_graph.modules.iter_mut() {
            module_node.is_leaf = self.module_graph.dependencies.get(module_name)
                .map(|deps| deps.is_empty())
                .unwrap_or(true);
            
            module_node.is_root = self.module_graph.dependents.get(module_name)
                .map(|deps| deps.is_empty())
                .unwrap_or(true);
        }
        
        Ok(())
    }
    
    fn get_function_module_dependencies(&self, function_name: &str) -> Vec<String> {
        // Simplified dependency analysis based on function names
        match function_name {
            // Math functions typically don't depend on other modules
            name if name.starts_with("Sin") || name.starts_with("Cos") || name.starts_with("Tan") => {
                vec![]
            }
            
            // String functions might depend on math for string length
            "StringJoin" => vec!["std::math".to_string()],
            
            // Tensor operations depend on math
            "Dot" | "Transpose" | "Maximum" => vec!["std::math".to_string()],
            
            // Array operations might depend on tensor operations
            "ArrayReshape" | "ArrayFlatten" => vec!["std::tensor".to_string()],
            
            // ML operations depend on tensor operations
            name if name.ends_with("Layer") => vec!["std::tensor".to_string()],
            
            // Rule operations might depend on list operations
            "ReplaceAll" | "Cases" => vec!["std::list".to_string()],
            
            _ => vec![]
        }
    }
    
    fn compute_module_topological_order(&mut self) -> Result<(), TreeShakeError> {
        let mut in_degree = HashMap::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        // Initialize in-degree count
        for module_name in self.module_graph.modules.keys() {
            in_degree.insert(module_name.clone(), 0);
        }
        
        // Calculate in-degrees
        for dependencies in self.module_graph.dependencies.values() {
            for dep in dependencies {
                *in_degree.entry(dep.clone()).or_insert(0) += 1;
            }
        }
        
        // Find modules with zero in-degree
        for (module_name, degree) in &in_degree {
            if *degree == 0 {
                queue.push_back(module_name.clone());
            }
        }
        
        // Process modules
        while let Some(module_name) = queue.pop_front() {
            result.push(module_name.clone());
            
            // Reduce in-degree of dependent modules
            if let Some(dependencies) = self.module_graph.dependencies.get(&module_name) {
                for dep in dependencies {
                    let degree = in_degree.get_mut(dep).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push_back(dep.clone());
                    }
                }
            }
        }
        
        self.module_graph.topological_order = result;
        Ok(())
    }
    
    fn detect_circular_dependencies(&mut self) -> Result<(), TreeShakeError> {
        // Use strongly connected components to find circular dependencies
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        let mut in_stack = HashSet::new();
        let mut low_link = HashMap::new();
        let mut index = HashMap::new();
        let mut current_index = 0;
        let mut sccs = Vec::new();
        
        for module_name in self.module_graph.modules.keys() {
            if !visited.contains(module_name) {
                self.tarjan_scc(
                    module_name,
                    &mut visited,
                    &mut stack,
                    &mut in_stack,
                    &mut low_link,
                    &mut index,
                    &mut current_index,
                    &mut sccs,
                );
            }
        }
        
        // Filter out single-node SCCs (no circular dependencies)
        self.module_graph.circular_groups = sccs.into_iter()
            .filter(|scc| scc.len() > 1)
            .collect();
        
        Ok(())
    }
    
    fn tarjan_scc(
        &self,
        module: &str,
        visited: &mut HashSet<String>,
        stack: &mut Vec<String>,
        in_stack: &mut HashSet<String>,
        low_link: &mut HashMap<String, usize>,
        index: &mut HashMap<String, usize>,
        current_index: &mut usize,
        sccs: &mut Vec<Vec<String>>,
    ) {
        visited.insert(module.to_string());
        index.insert(module.to_string(), *current_index);
        low_link.insert(module.to_string(), *current_index);
        *current_index += 1;
        stack.push(module.to_string());
        in_stack.insert(module.to_string());
        
        // Visit neighbors
        if let Some(dependencies) = self.module_graph.dependencies.get(module) {
            for dep in dependencies {
                if !visited.contains(dep) {
                    self.tarjan_scc(dep, visited, stack, in_stack, low_link, index, current_index, sccs);
                    let dep_low = *low_link.get(dep).unwrap();
                    let current_low = *low_link.get(module).unwrap();
                    low_link.insert(module.to_string(), current_low.min(dep_low));
                } else if in_stack.contains(dep) {
                    let dep_index = *index.get(dep).unwrap();
                    let current_low = *low_link.get(module).unwrap();
                    low_link.insert(module.to_string(), current_low.min(dep_index));
                }
            }
        }
        
        // If module is a root of SCC
        if low_link.get(module) == index.get(module) {
            let mut scc = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                in_stack.remove(&w);
                scc.push(w.clone());
                if w == module {
                    break;
                }
            }
            sccs.push(scc);
        }
    }
    
    fn compute_hierarchy_levels(&mut self) -> Result<(), TreeShakeError> {
        let mut levels = HashMap::new();
        
        // Start with leaf modules at level 0
        for (module_name, module_node) in &self.module_graph.modules {
            if module_node.is_leaf {
                levels.insert(module_name.clone(), 0);
            }
        }
        
        // Compute levels iteratively
        let mut changed = true;
        while changed {
            changed = false;
            for (module_name, dependencies) in &self.module_graph.dependencies {
                if !levels.contains_key(module_name) {
                    let max_dep_level = dependencies.iter()
                        .filter_map(|dep| levels.get(dep))
                        .max()
                        .copied();
                    
                    if let Some(max_level) = max_dep_level {
                        levels.insert(module_name.clone(), max_level + 1);
                        changed = true;
                    }
                }
            }
        }
        
        self.module_graph.hierarchy_levels = levels;
        Ok(())
    }
    
    fn analyze_import_export_patterns(&mut self, _module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        // Analyze how modules import from each other
        // This would be more sophisticated in a real implementation
        Ok(())
    }
    
    fn track_cross_module_calls(&mut self, _dependency_graph: &DependencyGraph) -> Result<(), TreeShakeError> {
        // Track function calls that cross module boundaries
        // This would analyze the dependency graph to find cross-module edges
        Ok(())
    }
    
    fn detect_optimization_opportunities(&mut self) -> Result<(), TreeShakeError> {
        // Detect various optimization opportunities for each module
        for (module_name, module_node) in self.module_graph.modules.iter_mut() {
            let mut optimizations = Vec::new();
            
            // Dead code elimination opportunity
            if !module_node.usage_patterns.unused_functions.is_empty() {
                optimizations.push(ModuleOptimization {
                    optimization_type: ModuleOptimizationType::DeadCodeElimination,
                    expected_impact: OptimizationImpact {
                        binary_size_reduction: module_node.usage_patterns.unused_functions.len() * 500,
                        compilation_time_reduction: module_node.usage_patterns.unused_functions.len() as u64 * 10,
                        memory_reduction: module_node.usage_patterns.unused_functions.len() * 256,
                        performance_improvement: 2.0,
                        maintainability_impact: MaintainabilityImpact::Positive,
                    },
                    complexity: OptimizationComplexity::Low,
                    prerequisites: vec![],
                    risk_level: RiskLevel::Low,
                });
            }
            
            // Module splitting opportunity
            if module_node.size_metrics.function_count > 20 &&
               module_node.usage_patterns.partial_usage.usage_percentage < 0.5 {
                optimizations.push(ModuleOptimization {
                    optimization_type: ModuleOptimizationType::ModuleSplitting,
                    expected_impact: OptimizationImpact {
                        binary_size_reduction: module_node.size_metrics.estimated_size / 3,
                        compilation_time_reduction: 100,
                        memory_reduction: module_node.size_metrics.runtime_memory_usage / 4,
                        performance_improvement: 5.0,
                        maintainability_impact: MaintainabilityImpact::Positive,
                    },
                    complexity: OptimizationComplexity::Medium,
                    prerequisites: vec!["dependency analysis".to_string()],
                    risk_level: RiskLevel::Medium,
                });
            }
            
            module_node.optimization_opportunities = optimizations;
        }
        
        Ok(())
    }
    
    fn update_dependency_graph_with_module_info(&self, dependency_graph: &mut DependencyGraph) -> Result<(), TreeShakeError> {
        // Update the dependency graph with module-level information
        for (module_name, dependencies) in &self.module_graph.dependencies {
            for dep_module in dependencies {
                dependency_graph.add_module_dependency(module_name.clone(), dep_module.clone());
            }
        }
        
        Ok(())
    }
}

impl ModuleDependencyGraph {
    fn new() -> Self {
        ModuleDependencyGraph {
            modules: HashMap::new(),
            dependencies: HashMap::new(),
            dependents: HashMap::new(),
            circular_groups: Vec::new(),
            topological_order: Vec::new(),
            hierarchy_levels: HashMap::new(),
        }
    }
}

impl ImportExportAnalyzer {
    fn new() -> Self {
        ImportExportAnalyzer {
            import_patterns: HashMap::new(),
            export_usage: HashMap::new(),
            cross_module_relationships: Vec::new(),
        }
    }
}

impl CrossModuleCallTracker {
    fn new() -> Self {
        CrossModuleCallTracker {
            cross_calls: HashMap::new(),
            hot_paths: Vec::new(),
            module_interfaces: HashMap::new(),
        }
    }
}

impl Default for ModuleDependencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_dependency_analyzer_creation() {
        let analyzer = ModuleDependencyAnalyzer::new();
        assert_eq!(analyzer.module_graph.modules.len(), 0);
        assert_eq!(analyzer.module_graph.dependencies.len(), 0);
    }

    #[test]
    fn test_module_type_determination() {
        let analyzer = ModuleDependencyAnalyzer::new();
        
        assert_eq!(analyzer.determine_module_type("std::math"), ModuleType::StdlibCore);
        assert_eq!(analyzer.determine_module_type("std::tensor"), ModuleType::StdlibExtended);
        assert_eq!(analyzer.determine_module_type("std::ml::core"), ModuleType::MachineLearning);
        assert_eq!(analyzer.determine_module_type("user::custom"), ModuleType::UserDefined);
    }

    #[test]
    fn test_essential_module_detection() {
        let analyzer = ModuleDependencyAnalyzer::new();
        
        assert!(analyzer.is_essential_module("std::math"));
        assert!(analyzer.is_essential_module("std::list"));
        assert!(!analyzer.is_essential_module("std::ml::layers"));
    }

    #[test]
    fn test_module_size_metrics() {
        let analyzer = ModuleDependencyAnalyzer::new();
        let exports = vec!["func1".to_string(), "func2".to_string()];
        
        let metrics = analyzer.calculate_module_size_metrics(&exports);
        assert_eq!(metrics.function_count, 2);
        assert_eq!(metrics.estimated_size, 1000);
    }

    #[test]
    fn test_function_module_dependencies() {
        let analyzer = ModuleDependencyAnalyzer::new();
        
        let deps = analyzer.get_function_module_dependencies("StringJoin");
        assert!(deps.contains(&"std::math".to_string()));
        
        let no_deps = analyzer.get_function_module_dependencies("Sin");
        assert!(no_deps.is_empty());
    }

    #[test]
    fn test_optimization_types() {
        assert_eq!(ModuleOptimizationType::DeadCodeElimination, ModuleOptimizationType::DeadCodeElimination);
        assert_ne!(ModuleOptimizationType::DeadCodeElimination, ModuleOptimizationType::ModuleSplitting);
    }

    #[test]
    fn test_risk_levels() {
        assert!(matches!(RiskLevel::Low, RiskLevel::Low));
        assert!(matches!(RiskLevel::High, RiskLevel::High));
    }

    #[test]
    fn test_import_types() {
        assert_eq!(ImportType::Full, ImportType::Full);
        assert_ne!(ImportType::Full, ImportType::Selective);
    }

    #[test]
    fn test_relationship_types() {
        assert_eq!(RelationshipType::DirectDependency, RelationshipType::DirectDependency);
        assert_ne!(RelationshipType::DirectDependency, RelationshipType::MutualDependency);
    }

    #[test]
    fn test_config_customization() {
        let config = ModuleAnalyzerConfig {
            max_dependency_depth: 3,
            enable_optimization_suggestions: false,
            ..Default::default()
        };
        
        let analyzer = ModuleDependencyAnalyzer::with_config(config);
        assert_eq!(analyzer.config.max_dependency_depth, 3);
        assert!(!analyzer.config.enable_optimization_suggestions);
    }
}