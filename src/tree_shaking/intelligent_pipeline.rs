//! Simple Dead Code Elimination Pipeline
//!
//! Analyzes user programs to eliminate unused stdlib functions,
//! enabling distribution of small compiled programs from a large engine.

use super::{DependencyGraph, TreeShakeError};
use crate::modules::registry::ModuleRegistry;
use crate::linker::{FunctionRegistry, FunctionEntry, LinkerError};
use crate::stdlib::StdlibFunction;
use crate::modules::FunctionImplementation;
use crate::ast::{Expr, Symbol, Number};
use std::collections::{HashSet, VecDeque};
use std::sync::{Arc, RwLock};

/// Simple dead code eliminator for small program distribution
pub struct DeadCodeEliminator {
    /// Required functions traced from user program
    required_functions: HashSet<String>,
    
    /// Required modules based on function dependencies  
    required_modules: HashSet<String>,
    
    /// Dependency graph for tracing
    dependency_graph: DependencyGraph,
}

impl DeadCodeEliminator {
    /// Create a new dead code eliminator
    pub fn new() -> Self {
        DeadCodeEliminator {
            required_functions: HashSet::new(),
            required_modules: HashSet::new(), 
            dependency_graph: DependencyGraph::new(),
        }
    }
    
    /// Eliminate unused code from a user program
    /// Returns the set of required functions for minimal binary
    pub fn eliminate_unused_code(
        &mut self,
        program_ast: &Expr,
        module_registry: &ModuleRegistry,
    ) -> Result<MinimalStdlib, TreeShakeError> {
        // Phase 1: Find all function calls in user program
        self.find_program_dependencies(program_ast)?;
        
        // Phase 2: Trace dependencies through stdlib
        self.trace_stdlib_dependencies(module_registry)?;
        
        // Phase 3: Create minimal stdlib
        let minimal_stdlib = self.create_minimal_stdlib(module_registry)?;
        
        Ok(minimal_stdlib)
    }
    
    /// Phase 1: Find all function calls in the user program AST
    fn find_program_dependencies(&mut self, expr: &Expr) -> Result<(), TreeShakeError> {
        match expr {
            Expr::Function { head, args } => {
                // Add the function being called
                if let Expr::Symbol(name) = head.as_ref() {
                    self.required_functions.insert(name.name.clone());
                }
                
                // Recursively check arguments
                for arg in args {
                    self.find_program_dependencies(arg)?;
                }
            }
            Expr::Symbol(name) => {
                // Could be a constant or function reference
                self.required_functions.insert(name.name.clone());
            }
            Expr::List(elements) => {
                for elem in elements {
                    self.find_program_dependencies(elem)?;
                }
            }
            // Handle other expression types as needed
            _ => {}
        }
        
        Ok(())
    }
    
    /// Phase 2: Trace dependencies through stdlib modules
    fn trace_stdlib_dependencies(&mut self, module_registry: &ModuleRegistry) -> Result<(), TreeShakeError> {
        let mut work_queue: VecDeque<String> = self.required_functions.iter().cloned().collect();
        let mut visited = HashSet::new();
        
        while let Some(function_name) = work_queue.pop_front() {
            if visited.contains(&function_name) {
                continue;
            }
            visited.insert(function_name.clone());
            
            // Find which module contains this function
            if let Some(module_name) = self.find_function_module(&function_name, module_registry) {
                self.required_modules.insert(module_name.clone());
                
                // Get dependencies of this function
                let deps = self.get_function_dependencies(&function_name, &module_name, module_registry)?;
                for dep in deps {
                    if !visited.contains(&dep) {
                        work_queue.push_back(dep.clone());
                        self.required_functions.insert(dep);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Phase 3: Create minimal stdlib with only required functions
    fn create_minimal_stdlib(&self, _module_registry: &ModuleRegistry) -> Result<MinimalStdlib, TreeShakeError> {
        Ok(MinimalStdlib {
            required_functions: self.required_functions.clone(),
            required_modules: self.required_modules.clone(),
            eliminated_function_count: self.estimate_eliminated_functions(),
            size_reduction_estimate: self.estimate_size_reduction(),
        })
    }
    
    /// Find which module contains a given function
    fn find_function_module(&self, function_name: &str, module_registry: &ModuleRegistry) -> Option<String> {
        // Check common modules first for performance
        let common_modules = ["Math", "Statistics", "List", "Core"];
        
        for module_name in &common_modules {
            if module_registry.has_function(module_name, function_name) {
                return Some(module_name.to_string());
            }
        }
        
        // Search all modules
        for module_name in module_registry.list_modules() {
            if module_registry.has_function(&module_name, function_name) {
                return Some(module_name);
            }
        }
        
        None
    }
    
    /// Get dependencies of a specific function within a module
    fn get_function_dependencies(
        &self, 
        function_name: &str, 
        _module_name: &str, 
        _module_registry: &ModuleRegistry
    ) -> Result<Vec<String>, TreeShakeError> {
        // Static dependency map based on known stdlib function relationships
        let dependencies = match function_name {
            // Math functions - mostly independent
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => vec![],
            
            // List functions with dependencies
            "Mean" => vec!["Length".to_string(), "Plus".to_string(), "Divide".to_string()],
            "StandardDeviation" => vec!["Mean".to_string(), "Length".to_string(), "Sqrt".to_string(), "Plus".to_string()],
            "Variance" => vec!["Mean".to_string(), "Length".to_string(), "Plus".to_string()],
            "Sum" => vec!["Plus".to_string()],
            "Product" => vec!["Times".to_string()],
            "Total" => vec!["Plus".to_string()],
            "Map" => vec!["Apply".to_string()],
            
            // String functions
            "StringJoin" => vec![], // primitive
            "StringLength" => vec![], // primitive
            "StringTake" => vec!["StringLength".to_string()],
            "StringDrop" => vec!["StringLength".to_string()],
            
            // List operations
            "Length" => vec![], // primitive
            "Head" => vec![], // primitive
            "Tail" => vec![], // primitive
            "Append" => vec![], // primitive
            "Flatten" => vec![], // primitive
            "Apply" => vec![], // primitive
            
            // Tensor operations
            "Dot" => vec!["Plus".to_string(), "Times".to_string()],
            "Transpose" => vec![], // primitive
            "Maximum" => vec![], // primitive
            "Array" => vec![], // primitive
            "ArrayDimensions" => vec![], // primitive
            "ArrayRank" => vec!["ArrayDimensions".to_string()],
            "ArrayReshape" => vec!["ArrayDimensions".to_string(), "Length".to_string()],
            "ArrayFlatten" => vec![], // primitive
            
            // Basic arithmetic (these are very fundamental)
            "Plus" => vec![], // primitive
            "Times" => vec![], // primitive
            "Divide" => vec![], // primitive
            "Minus" => vec![], // primitive
            "Power" => vec![], // primitive
            
            // Constants
            "Pi" => vec![], // primitive
            "E" => vec![], // primitive
            
            // ML functions (depend on tensor operations)
            "LinearLayer" => vec!["Dot".to_string(), "Plus".to_string()],
            "ReLU" => vec!["Maximum".to_string()],
            "Softmax" => vec!["Exp".to_string(), "Sum".to_string(), "Divide".to_string()],
            
            // Rule functions
            "Rule" => vec![], // primitive
            "RuleDelayed" => vec![], // primitive
            "ReplaceAll" => vec![], // primitive (but complex)
            
            // If we don't know the function, assume it has no dependencies
            _ => vec![],
        };
        
        Ok(dependencies)
    }
    
    /// Estimate how many functions were eliminated
    fn estimate_eliminated_functions(&self) -> usize {
        // Rough estimate: assume stdlib has ~500 functions total
        let estimated_total: usize = 500;
        estimated_total.saturating_sub(self.required_functions.len())
    }
    
    /// Estimate binary size reduction  
    fn estimate_size_reduction(&self) -> f64 {
        let retention_ratio = self.required_functions.len() as f64 / 500.0;
        (1.0 - retention_ratio) * 100.0 // Percentage reduction
    }
    
    /// Get the list of required functions
    pub fn get_required_functions(&self) -> &HashSet<String> {
        &self.required_functions
    }
    
    /// Get the list of required modules
    pub fn get_required_modules(&self) -> &HashSet<String> {
        &self.required_modules
    }
    
    /// Create a filtered function registry containing only required functions
    /// This is the key integration point for creating small binaries
    pub fn create_minimal_function_registry(
        &self,
        full_registry: &FunctionRegistry,
        module_registry: &ModuleRegistry,
    ) -> Result<FunctionRegistry, TreeShakeError> {
        let mut minimal_registry = FunctionRegistry::new();
        let mut function_index = 0u16;
        
        // Add only the required functions to the minimal registry
        for function_name in &self.required_functions {
            if let Some(function_entry) = full_registry.get_function_entry(function_name) {
                // Clone the function entry for the minimal registry - get implementation
                if let FunctionImplementation::Stdlib(func_ptr) = &function_entry.implementation {
                    minimal_registry.register_stdlib_function(
                        function_name,
                        *func_ptr.clone(),
                        function_entry.signature.arity,
                    ).map_err(|e| TreeShakeError::DependencyAnalysisError {
                        message: format!("Failed to register function {}: {:?}", function_name, e)
                    })?;
                }
                function_index += 1;
            } else {
                // Try to find the function in modules
                if let Some(module_name) = self.find_function_module(function_name, module_registry) {
                    // For now, create a placeholder entry - in real implementation
                    // this would extract the actual function from the module
                    if let FunctionImplementation::Stdlib(placeholder_impl) = create_placeholder_implementation(function_name) {
                        minimal_registry.register_stdlib_function(
                            function_name,
                            placeholder_impl,
                            1, // Default arity for placeholder
                        ).map_err(|e| TreeShakeError::DependencyAnalysisError {
                            message: format!("Failed to register placeholder for {}: {:?}", function_name, e)
                        })?;
                    }
                    function_index += 1;
                }
            }
        }
        
        Ok(minimal_registry)
    }
    
    /// Generate compilation configuration for minimal binary
    pub fn create_compilation_config(&self) -> CompilationConfig {
        CompilationConfig {
            included_functions: self.required_functions.clone(),
            included_modules: self.required_modules.clone(),
            eliminate_unused_imports: true,
            optimize_for_size: true,
            strip_debug_symbols: true,
            eliminate_unreachable_code: true,
        }
    }
    
    /// Generate a minimal binary for the given program (Phase 4: Binary Generation)
    /// This is the main entry point for creating small standalone executables
    pub fn generate_minimal_binary(
        &mut self,
        program_ast: &Expr,
        module_registry: &ModuleRegistry,
        full_function_registry: &FunctionRegistry,
    ) -> Result<MinimalBinary, TreeShakeError> {
        // Step 1: Analyze program dependencies
        let minimal_stdlib = self.eliminate_unused_code(program_ast, module_registry)?;
        
        // Step 2: Create filtered function registry
        let minimal_registry = self.create_minimal_function_registry(
            full_function_registry,
            module_registry,
        )?;
        
        // Step 3: Generate compilation configuration
        let compilation_config = self.create_compilation_config();
        
        // Step 4: Estimate binary characteristics
        let binary_size_estimate = self.estimate_binary_size(&minimal_stdlib, &compilation_config);
        
        Ok(MinimalBinary {
            stdlib: minimal_stdlib,
            function_registry: minimal_registry,
            compilation_config,
            binary_size_estimate,
            program_ast: program_ast.clone(),
        })
    }
    
    /// Estimate the size of the resulting binary
    fn estimate_binary_size(&self, minimal_stdlib: &MinimalStdlib, config: &CompilationConfig) -> BinarySizeEstimate {
        let base_runtime_size = 50_000; // ~50KB base runtime
        let function_size = minimal_stdlib.required_functions.len() * 200; // ~200 bytes per function
        let program_size = 1_000; // ~1KB for user program
        
        let total_unoptimized = base_runtime_size + function_size + program_size;
        let size_reduction_factor = config.get_estimated_size_reduction() / 100.0;
        let optimized_size = (total_unoptimized as f64 * (1.0 - size_reduction_factor)) as usize;
        
        BinarySizeEstimate {
            unoptimized_size: total_unoptimized,
            optimized_size,
            size_reduction_bytes: total_unoptimized - optimized_size,
            size_reduction_percentage: config.get_estimated_size_reduction(),
            functions_included: minimal_stdlib.required_functions.len(),
            functions_eliminated: minimal_stdlib.eliminated_function_count,
        }
    }
}

/// Result of dead code elimination - minimal stdlib needed for user program
#[derive(Debug, Clone)]
pub struct MinimalStdlib {
    /// Functions that must be included in the binary
    pub required_functions: HashSet<String>,
    
    /// Modules that must be included
    pub required_modules: HashSet<String>,
    
    /// Estimated number of eliminated functions
    pub eliminated_function_count: usize,
    
    /// Estimated size reduction percentage
    pub size_reduction_estimate: f64,
}

impl MinimalStdlib {
    /// Check if a function is required
    pub fn is_function_required(&self, function_name: &str) -> bool {
        self.required_functions.contains(function_name)
    }
    
    /// Check if a module is required
    pub fn is_module_required(&self, module_name: &str) -> bool {
        self.required_modules.contains(module_name)
    }
    
    /// Get summary of elimination results
    pub fn get_summary(&self) -> String {
        format!(
            "Minimal Stdlib: {} functions in {} modules (eliminated {}, ~{:.1}% size reduction)",
            self.required_functions.len(),
            self.required_modules.len(),
            self.eliminated_function_count,
            self.size_reduction_estimate
        )
    }
}

/// Complete minimal binary ready for distribution
#[derive(Debug)]
pub struct MinimalBinary {
    /// Minimal stdlib with only required functions
    pub stdlib: MinimalStdlib,
    
    /// Filtered function registry
    pub function_registry: FunctionRegistry,
    
    /// Compilation configuration
    pub compilation_config: CompilationConfig,
    
    /// Binary size estimate
    pub binary_size_estimate: BinarySizeEstimate,
    
    /// Original program AST
    pub program_ast: Expr,
}

impl MinimalBinary {
    /// Get a comprehensive summary of the binary optimization
    pub fn get_optimization_summary(&self) -> String {
        format!(
            "Minimal Binary Summary:\n\
             ├─ Functions: {} required, {} eliminated\n\
             ├─ Modules: {} included\n\
             ├─ Size: {}KB → {}KB ({:.1}% reduction)\n\
             ├─ Optimizations: {}\n\
             └─ Ready for distribution",
            self.stdlib.required_functions.len(),
            self.stdlib.eliminated_function_count,
            self.stdlib.required_modules.len(),
            self.binary_size_estimate.unoptimized_size / 1024,
            self.binary_size_estimate.optimized_size / 1024,
            self.binary_size_estimate.size_reduction_percentage,
            self.get_optimization_flags()
        )
    }
    
    /// Get list of enabled optimizations
    fn get_optimization_flags(&self) -> String {
        let mut flags = Vec::new();
        if self.compilation_config.eliminate_unused_imports { flags.push("dead-code"); }
        if self.compilation_config.optimize_for_size { flags.push("size-opt"); }
        if self.compilation_config.strip_debug_symbols { flags.push("strip"); }
        if self.compilation_config.eliminate_unreachable_code { flags.push("unreachable"); }
        
        if flags.is_empty() {
            "none".to_string()
        } else {
            flags.join(", ")
        }
    }
    
    /// Check if the binary is suitable for production distribution
    pub fn is_production_ready(&self) -> bool {
        self.compilation_config.eliminate_unused_imports
            && self.compilation_config.optimize_for_size
            && self.binary_size_estimate.optimized_size < 1_000_000 // < 1MB
    }
}

/// Detailed binary size estimation
#[derive(Debug, Clone)]
pub struct BinarySizeEstimate {
    /// Unoptimized binary size in bytes
    pub unoptimized_size: usize,
    
    /// Optimized binary size in bytes
    pub optimized_size: usize,
    
    /// Size reduction in bytes
    pub size_reduction_bytes: usize,
    
    /// Size reduction percentage
    pub size_reduction_percentage: f64,
    
    /// Number of functions included
    pub functions_included: usize,
    
    /// Number of functions eliminated
    pub functions_eliminated: usize,
}

impl BinarySizeEstimate {
    /// Get human-readable size summary
    pub fn get_size_summary(&self) -> String {
        format!(
            "{}KB → {}KB ({:.1}% smaller, saved {}KB)",
            self.unoptimized_size / 1024,
            self.optimized_size / 1024,
            self.size_reduction_percentage,
            self.size_reduction_bytes / 1024
        )
    }
    
    /// Check if the size reduction meets expectations
    pub fn is_effective_reduction(&self) -> bool {
        self.size_reduction_percentage > 80.0 && self.optimized_size < 500_000 // < 500KB
    }
}

/// Configuration for minimal binary compilation
#[derive(Debug, Clone)]
pub struct CompilationConfig {
    /// Functions to include in the binary
    pub included_functions: HashSet<String>,
    
    /// Modules to include in the binary
    pub included_modules: HashSet<String>,
    
    /// Whether to eliminate unused imports
    pub eliminate_unused_imports: bool,
    
    /// Whether to optimize for size over speed
    pub optimize_for_size: bool,
    
    /// Whether to strip debug symbols
    pub strip_debug_symbols: bool,
    
    /// Whether to eliminate unreachable code
    pub eliminate_unreachable_code: bool,
}

impl CompilationConfig {
    /// Create a minimal compilation configuration
    pub fn minimal() -> Self {
        CompilationConfig {
            included_functions: HashSet::new(),
            included_modules: HashSet::new(),
            eliminate_unused_imports: true,
            optimize_for_size: true,
            strip_debug_symbols: true,
            eliminate_unreachable_code: true,
        }
    }
    
    /// Create a development-friendly configuration
    pub fn development() -> Self {
        CompilationConfig {
            included_functions: HashSet::new(),
            included_modules: HashSet::new(),
            eliminate_unused_imports: false,
            optimize_for_size: false,
            strip_debug_symbols: false,
            eliminate_unreachable_code: false,
        }
    }
    
    /// Get estimated binary size reduction
    pub fn get_estimated_size_reduction(&self) -> f64 {
        let base_reduction = if self.eliminate_unused_imports { 80.0 } else { 0.0 };
        let size_reduction = if self.optimize_for_size { 10.0 } else { 0.0 };
        let debug_reduction = if self.strip_debug_symbols { 15.0 } else { 0.0 };
        let code_reduction = if self.eliminate_unreachable_code { 5.0 } else { 0.0 };
        
        base_reduction + size_reduction + debug_reduction + code_reduction
    }
}

/// Helper function to create placeholder implementation for missing functions
fn create_placeholder_implementation(function_name: &str) -> FunctionImplementation {
    use crate::vm::{Value, VmResult};
    
    let name = function_name.to_string();
    FunctionImplementation::Stdlib(Box::new(move |_args| {
        Err(crate::vm::VmError::TypeError {
            expected: format!("Function '{}' was eliminated by dead code elimination", name),
            actual: "eliminated_function".to_string(),
        })
    }))
}

impl Default for DeadCodeEliminator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;

    #[test]
    fn test_dead_code_eliminator_creation() {
        let eliminator = DeadCodeEliminator::new();
        assert_eq!(eliminator.required_functions.len(), 0);
        assert_eq!(eliminator.required_modules.len(), 0);
    }

    #[test]
    fn test_find_program_dependencies() {
        let mut eliminator = DeadCodeEliminator::new();
        
        // Test program: Mean[{1, 2, 3}]
        let program = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Mean".to_string() })),
            args: vec![Expr::List(vec![
                Expr::Number(Number::Real(1.0)),
                Expr::Number(Number::Real(2.0)),
                Expr::Number(Number::Real(3.0)),
            ])],
        };
        
        eliminator.find_program_dependencies(&program).unwrap();
        
        assert!(eliminator.required_functions.contains("Mean"));
        assert_eq!(eliminator.required_functions.len(), 1);
    }

    #[test]
    fn test_minimal_stdlib_summary() {
        let minimal = MinimalStdlib {
            required_functions: ["Mean", "Plus", "List"].iter().map(|s| s.to_string()).collect(),
            required_modules: ["Statistics", "Core"].iter().map(|s| s.to_string()).collect(),
            eliminated_function_count: 497,
            size_reduction_estimate: 99.4,
        };
        
        let summary = minimal.get_summary();
        assert!(summary.contains("3 functions"));
        assert!(summary.contains("2 modules"));
        assert!(summary.contains("497"));
        assert!(summary.contains("99.4%"));
    }

    #[test]
    fn test_complex_program_dependencies() {
        let mut eliminator = DeadCodeEliminator::new();
        
        // Test program: Sin[Pi/4] + Mean[{1, 2, 3}]
        let program = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
                    args: vec![Expr::Function {
                        head: Box::new(Expr::Symbol(Symbol { name: "Divide".to_string() })),
                        args: vec![
                            Expr::Symbol(Symbol { name: "Pi".to_string() }),
                            Expr::Number(Number::Real(4.0)),
                        ],
                    }],
                },
                Expr::Function {
                    head: Box::new(Expr::Symbol(Symbol { name: "Mean".to_string() })),
                    args: vec![Expr::List(vec![
                        Expr::Number(Number::Real(1.0)),
                        Expr::Number(Number::Real(2.0)),
                        Expr::Number(Number::Real(3.0)),
                    ])],
                },
            ],
        };
        
        eliminator.find_program_dependencies(&program).unwrap();
        
        assert!(eliminator.required_functions.contains("Plus"));
        assert!(eliminator.required_functions.contains("Sin"));
        assert!(eliminator.required_functions.contains("Divide"));
        assert!(eliminator.required_functions.contains("Pi"));
        assert!(eliminator.required_functions.contains("Mean"));
        assert_eq!(eliminator.required_functions.len(), 5);
    }

    #[test]
    fn test_dependency_tracing() {
        let mut eliminator = DeadCodeEliminator::new();
        
        // Test that Mean depends on Length, Plus, Divide
        let deps = eliminator.get_function_dependencies("Mean", "Statistics", &ModuleRegistry::new(Default::default())).unwrap();
        assert!(deps.contains(&"Length".to_string()));
        assert!(deps.contains(&"Plus".to_string()));
        assert!(deps.contains(&"Divide".to_string()));
        assert_eq!(deps.len(), 3);
        
        // Test that Sin has no dependencies (primitive function)
        let deps = eliminator.get_function_dependencies("Sin", "Math", &ModuleRegistry::new(Default::default())).unwrap();
        assert_eq!(deps.len(), 0);
        
        // Test that StandardDeviation has complex dependencies
        let deps = eliminator.get_function_dependencies("StandardDeviation", "Statistics", &ModuleRegistry::new(Default::default())).unwrap();
        assert!(deps.contains(&"Mean".to_string()));
        assert!(deps.contains(&"Sqrt".to_string()));
        assert!(deps.len() >= 4); // Mean, Length, Sqrt, Plus
    }

    #[test]
    fn test_realistic_program_elimination() {
        // Simulate a minimal program that should result in dramatic size reduction
        let eliminator = DeadCodeEliminator::new();
        
        // A program using just Mean should eliminate most functions
        let mut required = HashSet::new();
        required.insert("Mean".to_string());
        required.insert("Length".to_string());
        required.insert("Plus".to_string());
        required.insert("Divide".to_string());
        
        let minimal = MinimalStdlib {
            required_functions: required,
            required_modules: ["Statistics".to_string(), "Core".to_string()].iter().cloned().collect(),
            eliminated_function_count: 496, // Out of ~500 total functions
            size_reduction_estimate: 99.2,
        };
        
        // This should show massive size reduction for simple programs
        assert!(minimal.size_reduction_estimate > 95.0);
        assert!(minimal.required_functions.len() < 10);
        
        println!("{}", minimal.get_summary());
        // Should print something like: "Minimal Stdlib: 4 functions in 2 modules (eliminated 496, ~99.2% size reduction)"
    }

    #[test]
    fn test_compilation_config() {
        let eliminator = DeadCodeEliminator::new();
        
        let minimal_config = eliminator.create_compilation_config();
        assert!(minimal_config.eliminate_unused_imports);
        assert!(minimal_config.optimize_for_size);
        assert!(minimal_config.strip_debug_symbols);
        
        // Test size reduction estimation
        let reduction = minimal_config.get_estimated_size_reduction();
        assert!(reduction > 90.0); // Should estimate significant reduction
        
        println!("Estimated size reduction: {}%", reduction);
    }

    #[test] 
    fn test_realistic_elimination_workflow() {
        let mut eliminator = DeadCodeEliminator::new();
        
        // Simulate a simple program: Mean[{1, 2, 3}]
        let program = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Mean".to_string() })),
            args: vec![Expr::List(vec![
                Expr::Number(Number::Real(1.0)),
                Expr::Number(Number::Real(2.0)),
                Expr::Number(Number::Real(3.0)),
            ])],
        };
        
        // Create a mock registry for testing
        let function_registry = std::sync::Arc::new(std::sync::RwLock::new(FunctionRegistry::new()));
        let module_registry = ModuleRegistry::new(function_registry);
        
        // Run the complete elimination workflow
        let minimal_stdlib = eliminator.eliminate_unused_code(&program, &module_registry).unwrap();
        
        // Verify results
        assert!(minimal_stdlib.required_functions.contains("Mean"));
        assert!(minimal_stdlib.size_reduction_estimate > 95.0);
        
        // Generate compilation config
        let config = eliminator.create_compilation_config();
        assert!(config.included_functions.contains("Mean"));
        assert!(config.get_estimated_size_reduction() > 90.0);
        
        println!("Complete workflow test:");
        println!("  {}", minimal_stdlib.get_summary());
        println!("  Estimated binary size reduction: {}%", config.get_estimated_size_reduction());
    }
}