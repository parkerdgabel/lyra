//! Function Call Analysis Engine
//!
//! Analyzes function calls throughout the stdlib to build accurate dependency graphs.

use crate::modules::{Module, FunctionExport};
use crate::modules::registry::ModuleRegistry;
use super::{DependencyGraph, DependencyNode, DependencyEdge, DependencyType, DependencyCallContext, TreeShakeError};
use super::{FunctionMetadata, PerformanceCharacteristics, NodeUsageStats, ComplexityMetrics};
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::SystemTime;

/// Analyzes function calls and builds dependency relationships
pub struct CallAnalyzer {
    /// Function registry for lookup
    function_registry: HashMap<String, FunctionInfo>,
    
    /// Call patterns detected
    call_patterns: HashMap<String, CallPattern>,
    
    /// Static analysis results
    static_analysis: StaticAnalysisResults,
    
    /// Configuration
    config: CallAnalyzerConfig,
}

/// Information about a function for analysis
#[derive(Debug, Clone)]
pub struct FunctionInfo {
    /// Function name
    pub name: String,
    
    /// Module it belongs to
    pub module: String,
    
    /// Function signature
    pub signature: String,
    
    /// Function attributes
    pub attributes: Vec<String>,
    
    /// Whether function is builtin
    pub is_builtin: bool,
    
    /// Function implementation analysis
    pub implementation_info: ImplementationInfo,
}

/// Implementation analysis information
#[derive(Debug, Clone)]
pub struct ImplementationInfo {
    /// Functions this function directly calls
    pub direct_calls: Vec<String>,
    
    /// Conditional calls (may or may not be executed)
    pub conditional_calls: Vec<String>,
    
    /// Higher-order calls (through function pointers)
    pub higher_order_calls: Vec<String>,
    
    /// Types this function depends on
    pub type_dependencies: Vec<String>,
    
    /// Estimated complexity
    pub complexity_estimate: u32,
    
    /// Whether function has side effects
    pub has_side_effects: bool,
    
    /// Whether function is pure
    pub is_pure: bool,
    
    /// Memory allocation pattern
    pub allocation_pattern: AllocationPattern,
}

/// Memory allocation patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AllocationPattern {
    /// No allocations
    None,
    
    /// Constant size allocation
    Constant(usize),
    
    /// Allocation grows with input size
    Linear,
    
    /// Allocation grows quadratically
    Quadratic,
    
    /// Dynamic/unpredictable allocation
    Dynamic,
}

/// Call patterns detected in functions
#[derive(Debug, Clone)]
pub struct CallPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    
    /// Functions involved in the pattern
    pub functions: Vec<String>,
    
    /// Frequency of this pattern
    pub frequency: f64,
    
    /// Impact on performance
    pub performance_impact: PatternImpact,
}

/// Types of call patterns
#[derive(Debug, Clone, PartialEq)]
pub enum PatternType {
    /// Simple direct call
    DirectCall,
    
    /// Chain of function calls A->B->C
    CallChain,
    
    /// Recursive call pattern
    Recursion,
    
    /// Mutual recursion between functions
    MutualRecursion,
    
    /// Fan-out pattern (one function calls many)
    FanOut,
    
    /// Fan-in pattern (many functions call one)
    FanIn,
    
    /// Conditional calling pattern
    ConditionalCalls,
    
    /// Loop-based calling pattern
    LoopCalls,
    
    /// Pipeline pattern
    Pipeline,
}

/// Impact of a call pattern on performance
#[derive(Debug, Clone)]
pub struct PatternImpact {
    /// CPU impact score (0.0 to 1.0)
    pub cpu_impact: f64,
    
    /// Memory impact score (0.0 to 1.0)
    pub memory_impact: f64,
    
    /// I/O impact score (0.0 to 1.0)
    pub io_impact: f64,
    
    /// Overall performance score (0.0 to 1.0)
    pub overall_score: f64,
}

/// Results of static analysis
#[derive(Debug, Clone, Default)]
pub struct StaticAnalysisResults {
    /// Call graph adjacency list
    pub call_graph: HashMap<String, HashSet<String>>,
    
    /// Reverse call graph (who calls what)
    pub reverse_call_graph: HashMap<String, HashSet<String>>,
    
    /// Strongly connected components (cycles)
    pub sccs: Vec<Vec<String>>,
    
    /// Topological ordering
    pub topological_order: Vec<String>,
    
    /// Call depth from entry points
    pub call_depths: HashMap<String, u32>,
    
    /// Critical path analysis
    pub critical_paths: Vec<Vec<String>>,
    
    /// Bottleneck analysis
    pub bottlenecks: Vec<String>,
}

/// Configuration for call analyzer
#[derive(Debug, Clone)]
pub struct CallAnalyzerConfig {
    /// Enable deep static analysis
    pub enable_deep_analysis: bool,
    
    /// Maximum analysis depth
    pub max_analysis_depth: u32,
    
    /// Enable pattern detection
    pub enable_pattern_detection: bool,
    
    /// Minimum frequency for pattern reporting
    pub min_pattern_frequency: f64,
    
    /// Enable performance impact analysis
    pub enable_performance_analysis: bool,
    
    /// Recursion depth limit for analysis
    pub recursion_limit: u32,
}

impl Default for CallAnalyzerConfig {
    fn default() -> Self {
        CallAnalyzerConfig {
            enable_deep_analysis: true,
            max_analysis_depth: 10,
            enable_pattern_detection: true,
            min_pattern_frequency: 0.1,
            enable_performance_analysis: true,
            recursion_limit: 100,
        }
    }
}

impl CallAnalyzer {
    /// Create a new call analyzer
    pub fn new() -> Self {
        CallAnalyzer {
            function_registry: HashMap::new(),
            call_patterns: HashMap::new(),
            static_analysis: StaticAnalysisResults::default(),
            config: CallAnalyzerConfig::default(),
        }
    }
    
    /// Create call analyzer with custom configuration
    pub fn with_config(config: CallAnalyzerConfig) -> Self {
        CallAnalyzer {
            function_registry: HashMap::new(),
            call_patterns: HashMap::new(),
            static_analysis: StaticAnalysisResults::default(),
            config,
        }
    }
    
    /// Analyze stdlib function calls and build dependency graph
    pub fn analyze_stdlib_calls(
        &mut self,
        module_registry: &ModuleRegistry,
        dependency_graph: &mut DependencyGraph,
    ) -> Result<(), TreeShakeError> {
        // Step 1: Build function registry
        self.build_function_registry(module_registry)?;
        
        // Step 2: Analyze function implementations
        self.analyze_function_implementations()?;
        
        // Step 3: Build call graph
        self.build_call_graph()?;
        
        // Step 4: Detect call patterns
        if self.config.enable_pattern_detection {
            self.detect_call_patterns()?;
        }
        
        // Step 5: Populate dependency graph
        self.populate_dependency_graph(dependency_graph)?;
        
        // Step 6: Perform static analysis
        if self.config.enable_deep_analysis {
            self.perform_static_analysis(dependency_graph)?;
        }
        
        Ok(())
    }
    
    /// Get static analysis results
    pub fn get_analysis_results(&self) -> &StaticAnalysisResults {
        &self.static_analysis
    }
    
    /// Get detected call patterns
    pub fn get_call_patterns(&self) -> &HashMap<String, CallPattern> {
        &self.call_patterns
    }
    
    /// Get function information
    pub fn get_function_info(&self, function_name: &str) -> Option<&FunctionInfo> {
        self.function_registry.get(function_name)
    }
    
    /// Analyze specific function for call dependencies
    pub fn analyze_function_calls(&self, function_name: &str) -> Result<Vec<String>, TreeShakeError> {
        if let Some(function_info) = self.function_registry.get(function_name) {
            let mut calls = Vec::new();
            calls.extend(function_info.implementation_info.direct_calls.clone());
            calls.extend(function_info.implementation_info.conditional_calls.clone());
            calls.extend(function_info.implementation_info.higher_order_calls.clone());
            Ok(calls)
        } else {
            Err(TreeShakeError::DependencyAnalysisError {
                message: format!("Function '{}' not found in registry", function_name),
            })
        }
    }
    
    /// Get call depth for a function
    pub fn get_call_depth(&self, function_name: &str) -> Option<u32> {
        self.static_analysis.call_depths.get(function_name).copied()
    }
    
    /// Check if function is on critical path
    pub fn is_on_critical_path(&self, function_name: &str) -> bool {
        self.static_analysis.critical_paths.iter()
            .any(|path| path.contains(&function_name.to_string()))
    }
    
    /// Check if function is a bottleneck
    pub fn is_bottleneck(&self, function_name: &str) -> bool {
        self.static_analysis.bottlenecks.contains(&function_name.to_string())
    }
    
    // Private implementation methods
    
    fn build_function_registry(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        for namespace in module_registry.list_modules() {
            if let Some(module) = module_registry.get_module(&namespace) {
                self.register_module_functions(&namespace, &module)?;
            }
        }
        Ok(())
    }
    
    fn register_module_functions(&mut self, namespace: &str, module: &Module) -> Result<(), TreeShakeError> {
        for export in module.exports.values() {
            let function_info = self.analyze_function_export(namespace, export)?;
            self.function_registry.insert(export.export_name.clone(), function_info);
        }
        Ok(())
    }
    
    fn analyze_function_export(&self, namespace: &str, export: &FunctionExport) -> Result<FunctionInfo, TreeShakeError> {
        // Analyze the function based on its metadata
        let implementation_info = self.analyze_function_implementation(&export.export_name)?;
        
        Ok(FunctionInfo {
            name: export.export_name.clone(),
            module: namespace.to_string(),
            signature: format!("{}() -> Value", export.export_name),
            attributes: export.attributes.iter().map(|attr| format!("{:?}", attr)).collect(),
            is_builtin: true,
            implementation_info,
        })
    }
    
    fn analyze_function_implementation(&self, function_name: &str) -> Result<ImplementationInfo, TreeShakeError> {
        // This is where we would perform detailed static analysis of function bodies
        // For now, we'll use heuristics based on function names and stdlib knowledge
        
        let mut implementation_info = ImplementationInfo {
            direct_calls: Vec::new(),
            conditional_calls: Vec::new(),
            higher_order_calls: Vec::new(),
            type_dependencies: Vec::new(),
            complexity_estimate: 1,
            has_side_effects: false,
            is_pure: true,
            allocation_pattern: AllocationPattern::None,
        };
        
        // Analyze based on function categories
        match function_name {
            // Math functions - pure, no dependencies
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => {
                implementation_info.is_pure = true;
                implementation_info.has_side_effects = false;
                implementation_info.complexity_estimate = 1;
                implementation_info.allocation_pattern = AllocationPattern::None;
            }
            
            // List functions - may call other list functions
            "Length" | "Head" | "Tail" => {
                implementation_info.is_pure = true;
                implementation_info.complexity_estimate = 1;
                implementation_info.allocation_pattern = AllocationPattern::None;
            }
            
            "Append" => {
                implementation_info.is_pure = false; // Creates new list
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
            }
            
            "Flatten" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 3;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("Length".to_string());
                implementation_info.conditional_calls.push("Flatten".to_string()); // Recursive
            }
            
            "Map" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 3;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.higher_order_calls.push("Apply".to_string());
            }
            
            "Apply" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Dynamic;
            }
            
            // String functions
            "StringJoin" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("StringLength".to_string());
            }
            
            "StringLength" | "StringTake" | "StringDrop" => {
                implementation_info.is_pure = true;
                implementation_info.complexity_estimate = 1;
                implementation_info.allocation_pattern = AllocationPattern::None;
            }
            
            // Tensor operations - complex dependencies
            "Array" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
            }
            
            "ArrayDimensions" | "ArrayRank" => {
                implementation_info.is_pure = true;
                implementation_info.complexity_estimate = 1;
                implementation_info.allocation_pattern = AllocationPattern::None;
            }
            
            "ArrayReshape" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("ArrayDimensions".to_string());
            }
            
            "Dot" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 4;
                implementation_info.allocation_pattern = AllocationPattern::Quadratic;
                implementation_info.direct_calls.push("ArrayDimensions".to_string());
            }
            
            "Transpose" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 3;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("ArrayDimensions".to_string());
            }
            
            "Maximum" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
            }
            
            // Rule functions
            "MatchQ" | "Cases" | "Position" => {
                implementation_info.is_pure = true;
                implementation_info.complexity_estimate = 3;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
            }
            
            "ReplaceAll" | "ReplaceRepeated" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 4;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("MatchQ".to_string());
            }
            
            // ML functions - higher complexity
            "FlattenLayer" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Linear;
                implementation_info.direct_calls.push("ArrayReshape".to_string());
            }
            
            "Sequential" => {
                implementation_info.is_pure = false;
                implementation_info.complexity_estimate = 3;
                implementation_info.allocation_pattern = AllocationPattern::Dynamic;
                implementation_info.higher_order_calls.push("Apply".to_string());
            }
            
            _ => {
                // Default analysis for unknown functions
                implementation_info.complexity_estimate = 2;
                implementation_info.allocation_pattern = AllocationPattern::Constant(64);
            }
        }
        
        Ok(implementation_info)
    }
    
    fn analyze_function_implementations(&mut self) -> Result<(), TreeShakeError> {
        // This step would involve deeper analysis of function bodies
        // For now, we rely on the initial analysis done during registration
        Ok(())
    }
    
    fn build_call_graph(&mut self) -> Result<(), TreeShakeError> {
        // Build adjacency lists for call graph
        for (function_name, function_info) in &self.function_registry {
            let mut callees = HashSet::new();
            
            // Add direct calls
            for callee in &function_info.implementation_info.direct_calls {
                callees.insert(callee.clone());
            }
            
            // Add conditional calls
            for callee in &function_info.implementation_info.conditional_calls {
                callees.insert(callee.clone());
            }
            
            // Add higher-order calls
            for callee in &function_info.implementation_info.higher_order_calls {
                callees.insert(callee.clone());
            }
            
            self.static_analysis.call_graph.insert(function_name.clone(), callees.clone());
            
            // Build reverse call graph
            for callee in callees {
                self.static_analysis.reverse_call_graph
                    .entry(callee)
                    .or_insert_with(HashSet::new)
                    .insert(function_name.clone());
            }
        }
        
        Ok(())
    }
    
    fn detect_call_patterns(&mut self) -> Result<(), TreeShakeError> {
        // Detect various call patterns in the graph
        
        // 1. Detect recursive patterns
        self.detect_recursive_patterns()?;
        
        // 2. Detect fan-out patterns
        self.detect_fan_patterns()?;
        
        // 3. Detect call chains
        self.detect_call_chains()?;
        
        // 4. Detect pipeline patterns
        self.detect_pipeline_patterns()?;
        
        Ok(())
    }
    
    fn detect_recursive_patterns(&mut self) -> Result<(), TreeShakeError> {
        for (function_name, callees) in &self.static_analysis.call_graph {
            if callees.contains(function_name) {
                // Direct recursion
                let pattern = CallPattern {
                    pattern_type: PatternType::Recursion,
                    functions: vec![function_name.clone()],
                    frequency: 1.0,
                    performance_impact: PatternImpact {
                        cpu_impact: 0.8,
                        memory_impact: 0.6,
                        io_impact: 0.0,
                        overall_score: 0.7,
                    },
                };
                self.call_patterns.insert(format!("recursion_{}", function_name), pattern);
            }
        }
        
        Ok(())
    }
    
    fn detect_fan_patterns(&mut self) -> Result<(), TreeShakeError> {
        for (function_name, callees) in &self.static_analysis.call_graph {
            if callees.len() > 5 {
                // Fan-out pattern
                let pattern = CallPattern {
                    pattern_type: PatternType::FanOut,
                    functions: vec![function_name.clone()],
                    frequency: callees.len() as f64 / 10.0,
                    performance_impact: PatternImpact {
                        cpu_impact: 0.6,
                        memory_impact: 0.5,
                        io_impact: 0.2,
                        overall_score: 0.5,
                    },
                };
                self.call_patterns.insert(format!("fanout_{}", function_name), pattern);
            }
        }
        
        Ok(())
    }
    
    fn detect_call_chains(&mut self) -> Result<(), TreeShakeError> {
        // Detect linear call chains A -> B -> C
        for (function_name, callees) in &self.static_analysis.call_graph {
            if callees.len() == 1 {
                let callee = callees.iter().next().unwrap();
                if let Some(callee_callees) = self.static_analysis.call_graph.get(callee) {
                    if callee_callees.len() == 1 {
                        let chain = vec![function_name.clone(), callee.clone()];
                        let pattern = CallPattern {
                            pattern_type: PatternType::CallChain,
                            functions: chain,
                            frequency: 0.8,
                            performance_impact: PatternImpact {
                                cpu_impact: 0.3,
                                memory_impact: 0.3,
                                io_impact: 0.1,
                                overall_score: 0.3,
                            },
                        };
                        self.call_patterns.insert(format!("chain_{}_{}", function_name, callee), pattern);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    fn detect_pipeline_patterns(&mut self) -> Result<(), TreeShakeError> {
        // Detect pipeline patterns where data flows through multiple stages
        // This is a simplified implementation
        Ok(())
    }
    
    fn populate_dependency_graph(&self, dependency_graph: &mut DependencyGraph) -> Result<(), TreeShakeError> {
        // Add all function nodes to the dependency graph
        for (function_name, function_info) in &self.function_registry {
            let node = self.create_dependency_node(function_name, function_info)?;
            dependency_graph.add_node(node);
        }
        
        // Add dependency edges
        for (caller, function_info) in &self.function_registry {
            // Direct calls
            for callee in &function_info.implementation_info.direct_calls {
                if self.function_registry.contains_key(callee) {
                    let edge = DependencyEdge {
                        from: caller.clone(),
                        to: callee.clone(),
                        edge_type: DependencyType::DirectCall,
                        call_frequency: Some(100), // Estimated frequency
                        context: DependencyCallContext {
                            location: None,
                            call_depth: 1,
                            in_loop: false,
                            in_error_path: false,
                            conditional_probability: None,
                        },
                        is_critical: true,
                    };
                    dependency_graph.add_edge(edge)?;
                }
            }
            
            // Conditional calls
            for callee in &function_info.implementation_info.conditional_calls {
                if self.function_registry.contains_key(callee) {
                    let edge = DependencyEdge {
                        from: caller.clone(),
                        to: callee.clone(),
                        edge_type: DependencyType::ConditionalCall,
                        call_frequency: Some(50), // Lower frequency for conditional
                        context: DependencyCallContext {
                            location: None,
                            call_depth: 1,
                            in_loop: false,
                            in_error_path: false,
                            conditional_probability: Some(0.5),
                        },
                        is_critical: false,
                    };
                    dependency_graph.add_edge(edge)?;
                }
            }
            
            // Higher-order calls
            for callee in &function_info.implementation_info.higher_order_calls {
                if self.function_registry.contains_key(callee) {
                    let edge = DependencyEdge {
                        from: caller.clone(),
                        to: callee.clone(),
                        edge_type: DependencyType::HigherOrderCall,
                        call_frequency: Some(25), // Even lower frequency
                        context: DependencyCallContext {
                            location: None,
                            call_depth: 2,
                            in_loop: false,
                            in_error_path: false,
                            conditional_probability: Some(0.3),
                        },
                        is_critical: false,
                    };
                    dependency_graph.add_edge(edge)?;
                }
            }
        }
        
        Ok(())
    }
    
    fn create_dependency_node(&self, function_name: &str, function_info: &FunctionInfo) -> Result<DependencyNode, TreeShakeError> {
        let is_entry_point = self.is_entry_point_function(function_name);
        
        Ok(DependencyNode {
            name: function_name.to_string(),
            module: function_info.module.clone(),
            metadata: FunctionMetadata {
                signature: function_info.signature.clone(),
                attributes: function_info.attributes.clone(),
                documentation: None,
                is_pure: function_info.implementation_info.is_pure,
                can_inline: function_info.implementation_info.complexity_estimate <= 2,
                estimated_size: self.estimate_function_size(function_info),
                performance: PerformanceCharacteristics {
                    time_complexity: self.estimate_time_complexity(function_info),
                    space_complexity: self.estimate_space_complexity(function_info),
                    allocates_memory: !matches!(function_info.implementation_info.allocation_pattern, AllocationPattern::None),
                    performs_io: false,
                    typical_execution_time: Some(self.estimate_execution_time(function_info)),
                },
            },
            usage_stats: NodeUsageStats::default(),
            is_entry_point,
            is_dead_code: false,
            complexity: ComplexityMetrics {
                cyclomatic_complexity: function_info.implementation_info.complexity_estimate,
                parameter_count: self.estimate_parameter_count(function_name),
                local_variable_count: function_info.implementation_info.complexity_estimate * 2,
                nesting_depth: function_info.implementation_info.complexity_estimate,
                branch_count: if function_info.implementation_info.conditional_calls.is_empty() { 0 } else { 2 },
                lines_of_code: function_info.implementation_info.complexity_estimate * 5,
            },
        })
    }
    
    fn is_entry_point_function(&self, function_name: &str) -> bool {
        // Functions that are likely to be entry points (commonly used)
        matches!(function_name, 
            "Sin" | "Cos" | "Length" | "Head" | "StringJoin" | "Array" | "Dot" | "Transpose" | "Map" | "Apply"
        )
    }
    
    fn estimate_function_size(&self, function_info: &FunctionInfo) -> usize {
        match function_info.implementation_info.complexity_estimate {
            1 => 100,
            2 => 200,
            3 => 400,
            4 => 800,
            _ => 1000,
        }
    }
    
    fn estimate_time_complexity(&self, function_info: &FunctionInfo) -> String {
        match function_info.implementation_info.complexity_estimate {
            1 => "O(1)".to_string(),
            2 => "O(n)".to_string(),
            3 => "O(n log n)".to_string(),
            4 => "O(n²)".to_string(),
            _ => "O(n³)".to_string(),
        }
    }
    
    fn estimate_space_complexity(&self, function_info: &FunctionInfo) -> String {
        match function_info.implementation_info.allocation_pattern {
            AllocationPattern::None => "O(1)".to_string(),
            AllocationPattern::Constant(_) => "O(1)".to_string(),
            AllocationPattern::Linear => "O(n)".to_string(),
            AllocationPattern::Quadratic => "O(n²)".to_string(),
            AllocationPattern::Dynamic => "O(n)".to_string(),
        }
    }
    
    fn estimate_execution_time(&self, function_info: &FunctionInfo) -> u64 {
        match function_info.implementation_info.complexity_estimate {
            1 => 100,   // 100 nanoseconds
            2 => 500,   // 500 nanoseconds
            3 => 2000,  // 2 microseconds
            4 => 10000, // 10 microseconds
            _ => 50000, // 50 microseconds
        }
    }
    
    fn estimate_parameter_count(&self, function_name: &str) -> u32 {
        match function_name {
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" | "Length" | "Head" | "Tail" | "StringLength" => 1,
            "Append" | "StringJoin" | "StringTake" | "StringDrop" | "Dot" | "Maximum" | "Map" => 2,
            "Apply" | "ArrayReshape" => 3,
            _ => 1,
        }
    }
    
    fn perform_static_analysis(&mut self, dependency_graph: &mut DependencyGraph) -> Result<(), TreeShakeError> {
        // Compute strongly connected components
        self.static_analysis.sccs = dependency_graph.find_strongly_connected_components();
        
        // Compute topological order
        match dependency_graph.topological_sort() {
            Ok(order) => self.static_analysis.topological_order = order,
            Err(_) => {
                // Handle cycles - use partial ordering
                self.static_analysis.topological_order = dependency_graph.all_functions()
                    .iter().map(|s| s.to_string()).collect();
            }
        }
        
        // Compute call depths
        self.compute_call_depths(dependency_graph)?;
        
        // Identify critical paths
        self.identify_critical_paths(dependency_graph)?;
        
        // Identify bottlenecks
        self.identify_bottlenecks(dependency_graph)?;
        
        Ok(())
    }
    
    fn compute_call_depths(&mut self, dependency_graph: &DependencyGraph) -> Result<(), TreeShakeError> {
        let mut depths = HashMap::new();
        let mut queue = VecDeque::new();
        
        // Start with entry points at depth 0
        for entry_point in dependency_graph.entry_points() {
            depths.insert(entry_point.name.clone(), 0);
            queue.push_back((entry_point.name.clone(), 0));
        }
        
        // BFS to compute depths
        while let Some((function_name, depth)) = queue.pop_front() {
            for edge in dependency_graph.get_dependencies(&function_name) {
                let new_depth = depth + 1;
                let current_depth = depths.get(&edge.to).copied().unwrap_or(u32::MAX);
                
                if new_depth < current_depth {
                    depths.insert(edge.to.clone(), new_depth);
                    queue.push_back((edge.to.clone(), new_depth));
                }
            }
        }
        
        self.static_analysis.call_depths = depths;
        Ok(())
    }
    
    fn identify_critical_paths(&mut self, dependency_graph: &DependencyGraph) -> Result<(), TreeShakeError> {
        let mut critical_paths = Vec::new();
        
        // Find paths from entry points to frequently called functions
        for entry_point in dependency_graph.entry_points() {
            let paths = self.find_paths_from_entry_point(&entry_point.name, dependency_graph, 5);
            critical_paths.extend(paths);
        }
        
        self.static_analysis.critical_paths = critical_paths;
        Ok(())
    }
    
    fn find_paths_from_entry_point(
        &self,
        entry_point: &str,
        dependency_graph: &DependencyGraph,
        max_length: usize,
    ) -> Vec<Vec<String>> {
        let mut paths = Vec::new();
        let mut current_path = vec![entry_point.to_string()];
        let mut visited = HashSet::new();
        
        self.dfs_find_paths(
            entry_point,
            dependency_graph,
            &mut current_path,
            &mut visited,
            &mut paths,
            max_length,
        );
        
        paths
    }
    
    fn dfs_find_paths(
        &self,
        current: &str,
        dependency_graph: &DependencyGraph,
        current_path: &mut Vec<String>,
        visited: &mut HashSet<String>,
        paths: &mut Vec<Vec<String>>,
        max_length: usize,
    ) {
        if current_path.len() >= max_length {
            paths.push(current_path.clone());
            return;
        }
        
        visited.insert(current.to_string());
        
        for edge in dependency_graph.get_dependencies(current) {
            if !visited.contains(&edge.to) {
                current_path.push(edge.to.clone());
                self.dfs_find_paths(&edge.to, dependency_graph, current_path, visited, paths, max_length);
                current_path.pop();
            }
        }
        
        visited.remove(current);
    }
    
    fn identify_bottlenecks(&mut self, dependency_graph: &DependencyGraph) -> Result<(), TreeShakeError> {
        let mut bottlenecks = Vec::new();
        
        // Functions with high in-degree (many functions depend on them)
        for function_name in dependency_graph.all_functions() {
            let dependents = dependency_graph.get_dependents(function_name);
            if dependents.len() > 3 {
                bottlenecks.push(function_name.clone());
            }
        }
        
        self.static_analysis.bottlenecks = bottlenecks;
        Ok(())
    }
}

impl Default for CallAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_call_analyzer_creation() {
        let analyzer = CallAnalyzer::new();
        assert_eq!(analyzer.function_registry.len(), 0);
        assert_eq!(analyzer.call_patterns.len(), 0);
    }

    #[test]
    fn test_function_info_creation() {
        let analyzer = CallAnalyzer::new();
        
        let impl_info = analyzer.analyze_function_implementation("Sin").unwrap();
        assert!(impl_info.is_pure);
        assert!(!impl_info.has_side_effects);
        assert_eq!(impl_info.complexity_estimate, 1);
    }

    #[test]
    fn test_complex_function_analysis() {
        let analyzer = CallAnalyzer::new();
        
        let impl_info = analyzer.analyze_function_implementation("Flatten").unwrap();
        assert!(!impl_info.is_pure);
        assert_eq!(impl_info.complexity_estimate, 3);
        assert!(impl_info.direct_calls.contains(&"Length".to_string()));
        assert!(impl_info.conditional_calls.contains(&"Flatten".to_string()));
    }

    #[test]
    fn test_allocation_patterns() {
        let analyzer = CallAnalyzer::new();
        
        let sin_info = analyzer.analyze_function_implementation("Sin").unwrap();
        assert_eq!(sin_info.allocation_pattern, AllocationPattern::None);
        
        let append_info = analyzer.analyze_function_implementation("Append").unwrap();
        assert_eq!(append_info.allocation_pattern, AllocationPattern::Linear);
        
        let dot_info = analyzer.analyze_function_implementation("Dot").unwrap();
        assert_eq!(dot_info.allocation_pattern, AllocationPattern::Quadratic);
    }

    #[test]
    fn test_call_pattern_types() {
        assert_eq!(PatternType::DirectCall, PatternType::DirectCall);
        assert_ne!(PatternType::DirectCall, PatternType::Recursion);
    }

    #[test]
    fn test_performance_impact() {
        let impact = PatternImpact {
            cpu_impact: 0.8,
            memory_impact: 0.6,
            io_impact: 0.0,
            overall_score: 0.7,
        };
        
        assert_eq!(impact.cpu_impact, 0.8);
        assert_eq!(impact.overall_score, 0.7);
    }

    #[test]
    fn test_static_analysis_results() {
        let results = StaticAnalysisResults::default();
        assert_eq!(results.call_graph.len(), 0);
        assert_eq!(results.sccs.len(), 0);
        assert_eq!(results.topological_order.len(), 0);
    }

    #[test]
    fn test_config_customization() {
        let config = CallAnalyzerConfig {
            max_analysis_depth: 5,
            enable_pattern_detection: false,
            ..Default::default()
        };
        
        let analyzer = CallAnalyzer::with_config(config);
        assert_eq!(analyzer.config.max_analysis_depth, 5);
        assert!(!analyzer.config.enable_pattern_detection);
    }
}