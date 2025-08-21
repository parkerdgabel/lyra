//! Module Registry System
//!
//! Manages the registration and lookup of modules within the Lyra system.

use super::{Module, ModuleError, Version, VersionConstraint, FunctionExport, FunctionImplementation};
use crate::{
    stdlib::{StandardLibrary, StdlibFunction},
    linker::{FunctionRegistry, FunctionAttribute},
    vm::Value,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global module registry for the Lyra system
#[derive(Debug)]
pub struct ModuleRegistry {
    /// Loaded modules indexed by namespace
    modules: RwLock<HashMap<String, Arc<Module>>>,
    
    /// Unified function registry for static dispatch
    function_registry: Arc<RwLock<FunctionRegistry>>,
    
    /// Namespace resolver for function lookup
    resolver: RwLock<NamespaceResolver>,
    
    /// Standard library integration
    stdlib: Arc<StandardLibrary>,
}

impl ModuleRegistry {
    /// Create a new module registry
    pub fn new(function_registry: Arc<RwLock<FunctionRegistry>>) -> Self {
        let stdlib = Arc::new(StandardLibrary::new());
        let mut registry = ModuleRegistry {
            modules: RwLock::new(HashMap::new()),
            function_registry,
            resolver: RwLock::new(NamespaceResolver::new()),
            stdlib,
        };
        
        // Register standard library as modules
        if let Err(e) = registry.register_stdlib_modules() {
            eprintln!("Warning: Failed to register stdlib modules: {}", e);
        }
        
        registry
    }
    
    /// Register a module in the registry
    pub fn register_module(&self, namespace: &str, module: Module) -> Result<(), ModuleError> {
        // Validate module
        module.validate()?;
        
        // Check for conflicts
        if self.modules.read().unwrap().contains_key(namespace) {
            return Err(ModuleError::PackageError {
                message: format!("Module {} already registered", namespace),
            });
        }
        
        // Register all exported functions in the unified registry
        {
            let mut func_registry = self.function_registry.write().unwrap();
            for (name, export) in &module.exports {
                let qualified_name = format!("{}::{}", namespace, name);
                
                match &export.implementation {
                    FunctionImplementation::Native(func) => {
                        // Determine arity from function attributes or default to 1
                        let arity = self.infer_function_arity(name);
                        func_registry.register_stdlib_function(
                            &qualified_name,
                            *func,
                            arity,
                        ).map_err(|e| ModuleError::PackageError {
                            message: format!("Failed to register function {}: {:?}", qualified_name, e),
                        })?;
                    },
                    FunctionImplementation::Foreign { type_name, method_name } => {
                        // For foreign methods, we need to handle them through the existing foreign system
                        // This would require extending the linker to handle module-aware foreign methods
                        // For now, we'll store the information but not register in the function registry
                    },
                    FunctionImplementation::Lyra { .. } => {
                        // For Lyra-defined functions, we would need to extend the function registry
                        // to handle bytecode-based functions
                    },
                    FunctionImplementation::External { .. } => {
                        // For external functions, we would need FFI integration
                    },
                }
            }
        }
        
        // Register in namespace resolver
        self.resolver.write().unwrap().register_namespace(namespace, &module)?;
        
        // Store module
        self.modules.write().unwrap().insert(namespace.to_string(), Arc::new(module));
        
        Ok(())
    }
    
    /// Get a module by namespace
    pub fn get_module(&self, namespace: &str) -> Option<Arc<Module>> {
        self.modules.read().unwrap().get(namespace).cloned()
    }
    
    /// List all registered modules
    pub fn list_modules(&self) -> Vec<String> {
        self.modules.read().unwrap().keys().cloned().collect()
    }
    
    /// Resolve a function call to a function index
    pub fn resolve_function(&self, qualified_name: &str) -> Result<u16, ModuleError> {
        self.function_registry
            .read()
            .unwrap()
            .get_function_index(qualified_name)
            .ok_or_else(|| ModuleError::ModuleNotFound {
                name: qualified_name.to_string(),
            })
    }
    
    /// Search for modules by keyword
    pub fn search_modules(&self, query: &str) -> Vec<String> {
        let modules = self.modules.read().unwrap();
        modules
            .iter()
            .filter(|(name, module)| {
                name.contains(query) ||
                module.metadata.description.contains(query) ||
                module.metadata.keywords.iter().any(|k| k.contains(query))
            })
            .map(|(name, _)| name.clone())
            .collect()
    }
    
    /// Get module metadata
    pub fn get_module_info(&self, namespace: &str) -> Option<super::ModuleMetadata> {
        self.modules
            .read()
            .unwrap()
            .get(namespace)
            .map(|module| module.metadata.clone())
    }
    
    /// Check if a module is registered
    pub fn has_module(&self, namespace: &str) -> bool {
        self.modules.read().unwrap().contains_key(namespace)
    }
    
    /// Get all exported functions from a module
    pub fn get_module_exports(&self, namespace: &str) -> Vec<String> {
        self.modules
            .read()
            .unwrap()
            .get(namespace)
            .map(|module| module.exports.keys().cloned().collect())
            .unwrap_or_default()
    }
    
    /// Check if a module contains a specific function
    pub fn has_function(&self, namespace: &str, function_name: &str) -> bool {
        self.modules
            .read()
            .unwrap()
            .get(namespace)
            .map(|module| module.exports.contains_key(function_name))
            .unwrap_or(false)
    }
    
    /// Register standard library functions as modules
    fn register_stdlib_modules(&mut self) -> Result<(), ModuleError> {
        // Create std::math module
        let mut math_module = Module::new(super::ModuleMetadata {
            name: "std::math".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard mathematical functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["math", "trigonometry", "algebra"].into_iter().map(String::from).collect(),
            categories: vec!["mathematics".to_string()],
        });
        
        // Add math functions
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Sin", crate::stdlib::mathematics::basic::sin, vec![FunctionAttribute::Listable], 
            "Sine function - computes sin(x) for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Cos", crate::stdlib::mathematics::basic::cos, vec![FunctionAttribute::Listable],
            "Cosine function - computes cos(x) for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Tan", crate::stdlib::mathematics::basic::tan, vec![FunctionAttribute::Listable],
            "Tangent function - computes tan(x) for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Exp", crate::stdlib::mathematics::basic::exp, vec![FunctionAttribute::Listable],
            "Exponential function - computes e^x for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Log", crate::stdlib::mathematics::basic::log, vec![FunctionAttribute::Listable],
            "Natural logarithm function - computes ln(x) for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Sqrt", crate::stdlib::mathematics::basic::sqrt, vec![FunctionAttribute::Listable],
            "Square root function - computes sqrt(x) for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Plus", crate::stdlib::mathematics::basic::plus, vec![FunctionAttribute::Listable, FunctionAttribute::Orderless],
            "Addition function - computes the sum of arguments"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Times", crate::stdlib::mathematics::basic::times, vec![FunctionAttribute::Listable, FunctionAttribute::Orderless],
            "Multiplication function - computes the product of arguments"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Power", crate::stdlib::mathematics::basic::power, vec![FunctionAttribute::Listable],
            "Power function - computes base^exponent"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Divide", crate::stdlib::mathematics::basic::divide, vec![FunctionAttribute::Listable],
            "Division function - computes a/b for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "Minus", crate::stdlib::mathematics::basic::minus, vec![FunctionAttribute::Listable],
            "Subtraction function - computes a-b for numeric inputs"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "TestHold", crate::stdlib::mathematics::basic::test_hold, vec![FunctionAttribute::Hold(vec![1])],
            "Test function for Hold attribute support"
        ))?;
        math_module.add_export(super::stdlib_to_export_with_docs(
            "TestHoldMultiple", crate::stdlib::mathematics::basic::test_hold_multiple, vec![FunctionAttribute::Hold(vec![1, 2])],
            "Test function for Hold attribute with multiple arguments"
        ))?;
        
        self.register_module("std::math", math_module)?;
        
        // Create std::list module
        let mut list_module = Module::new(super::ModuleMetadata {
            name: "std::list".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard list manipulation functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["list", "array", "collection"].into_iter().map(String::from).collect(),
            categories: vec!["data-structures".to_string()],
        });
        
        // Add list functions
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Length", crate::stdlib::list::length, vec![],
            "Returns the number of elements in a list"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Head", crate::stdlib::list::head, vec![],
            "Returns the first element of a list"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Tail", crate::stdlib::list::tail, vec![],
            "Returns all elements except the first"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Append", crate::stdlib::list::append, vec![],
            "Appends an element to the end of a list"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Flatten", crate::stdlib::list::flatten, vec![],
            "Flattens nested lists into a single list"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Map", crate::stdlib::list::map, vec![],
            "Applies a function to each element of a list"
        ))?;
        list_module.add_export(super::stdlib_to_export_with_docs(
            "Apply", crate::stdlib::list::apply, vec![],
            "Applies a function to arguments"
        ))?;
        
        self.register_module("std::list", list_module)?;
        
        // Create std::string module  
        let mut string_module = Module::new(super::ModuleMetadata {
            name: "std::string".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard string manipulation functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["string", "text", "manipulation"].into_iter().map(String::from).collect(),
            categories: vec!["text-processing".to_string()],
        });
        
        // Add string functions
        string_module.add_export(super::stdlib_to_export_with_docs(
            "StringJoin", crate::stdlib::string::string_join, vec![],
            "Joins strings together with a delimiter"
        ))?;
        string_module.add_export(super::stdlib_to_export_with_docs(
            "StringLength", crate::stdlib::string::string_length, vec![],
            "Returns the length of a string"
        ))?;
        string_module.add_export(super::stdlib_to_export_with_docs(
            "StringTake", crate::stdlib::string::string_take, vec![],
            "Takes the first n characters of a string"
        ))?;
        string_module.add_export(super::stdlib_to_export_with_docs(
            "StringDrop", crate::stdlib::string::string_drop, vec![],
            "Drops the first n characters of a string"
        ))?;
        
        self.register_module("std::string", string_module)?;
        
        // Create std::tensor module
        let mut tensor_module = Module::new(super::ModuleMetadata {
            name: "std::tensor".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard tensor operations and linear algebra".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["tensor", "linear-algebra", "numpy"].into_iter().map(String::from).collect(),
            categories: vec!["linear-algebra".to_string()],
        });
        
        // Add tensor functions
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Array", crate::stdlib::tensor::array, vec![],
            "Creates a tensor from nested lists"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Dot", crate::stdlib::tensor::dot, vec![],
            "Matrix multiplication and dot product"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Transpose", crate::stdlib::tensor::transpose, vec![],
            "Transposes a matrix or tensor"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Maximum", crate::stdlib::tensor::maximum, vec![],
            "Element-wise maximum of tensors"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "ArrayDimensions", crate::stdlib::tensor::array_dimensions, vec![],
            "Returns the dimensions of a tensor"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "ArrayRank", crate::stdlib::tensor::array_rank, vec![],
            "Returns the rank (number of dimensions) of a tensor"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "ArrayReshape", crate::stdlib::tensor::array_reshape, vec![],
            "Reshapes a tensor to new dimensions"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "ArrayFlatten", crate::stdlib::tensor::array_flatten, vec![],
            "Flattens a tensor to 1D"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Sigmoid", crate::stdlib::tensor::sigmoid, vec![FunctionAttribute::Listable],
            "Sigmoid activation function"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Tanh", crate::stdlib::tensor::tanh, vec![FunctionAttribute::Listable],
            "Hyperbolic tangent activation function"
        ))?;
        tensor_module.add_export(super::stdlib_to_export_with_docs(
            "Softmax", crate::stdlib::tensor::softmax, vec![],
            "Softmax activation function"
        ))?;
        
        self.register_module("std::tensor", tensor_module)?;
        
        // Create std::ml::core module for core ML types and utilities
        let mut ml_core_module = Module::new(super::ModuleMetadata {
            name: "std::ml::core".to_string(),
            version: Version::new(0, 1, 0),
            description: "Core ML framework utilities and tensor operations".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["ml", "machine-learning", "tensor", "utilities"].into_iter().map(String::from).collect(),
            categories: vec!["machine-learning".to_string()],
        });
        
        // Add tensor utility functions to ml core
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "TensorShape", crate::stdlib::ml::wrapper::tensor_shape, vec![],
            "Returns the shape of a tensor as a list"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "TensorRank", crate::stdlib::ml::wrapper::tensor_rank, vec![],
            "Returns the number of dimensions of a tensor"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "TensorSize", crate::stdlib::ml::wrapper::tensor_size, vec![],
            "Returns the total number of elements in a tensor"
        ))?;
        
        self.register_module("std::ml::core", ml_core_module)?;
        
        // Create std::ml::layers module for neural network layers
        let mut ml_layers_module = Module::new(super::ModuleMetadata {
            name: "std::ml::layers".to_string(),
            version: Version::new(0, 1, 0),
            description: "Neural network layers for deep learning".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["ml", "neural-networks", "layers", "deep-learning"].into_iter().map(String::from).collect(),
            categories: vec!["machine-learning".to_string()],
        });
        
        // Add spatial layer functions
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "FlattenLayer", crate::stdlib::ml::wrapper::flatten_layer, vec![],
            "Flattens multi-dimensional tensors to 1D or specified dimensions"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "ReshapeLayer", crate::stdlib::ml::wrapper::reshape_layer, vec![],
            "Reshapes tensors to new dimensions (supports -1 for auto-inference)"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "PermuteLayer", crate::stdlib::ml::wrapper::permute_layer, vec![],
            "Reorders tensor dimensions according to permutation"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "TransposeLayer", crate::stdlib::ml::wrapper::transpose_layer, vec![],
            "Transposes 2D tensors (swaps first two dimensions)"
        ))?;
        
        // Add layer composition functions
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Sequential", crate::stdlib::ml::wrapper::sequential_layer, vec![],
            "Composes multiple layers sequentially"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Identity", crate::stdlib::ml::wrapper::identity_layer, vec![],
            "Identity layer that returns input unchanged"
        ))?;
        
        self.register_module("std::ml::layers", ml_layers_module)?;
        
        // Create std::rules module for pattern matching and replacement
        let mut rules_module = Module::new(super::ModuleMetadata {
            name: "std::rules".to_string(),
            version: Version::new(0, 1, 0),
            description: "Pattern matching and replacement rules".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["pattern", "rules", "matching", "replacement"].into_iter().map(String::from).collect(),
            categories: vec!["pattern-matching".to_string()],
        });
        
        // Add rule functions
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "MatchQ", crate::stdlib::rules::match_q, vec![],
            "Tests whether an expression matches a pattern"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "Cases", crate::stdlib::rules::cases, vec![],
            "Extracts elements that match a pattern"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "CountPattern", crate::stdlib::rules::count_pattern, vec![],
            "Counts elements that match a pattern"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "Position", crate::stdlib::rules::position, vec![],
            "Finds positions of elements matching a pattern"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "ReplaceAll", crate::stdlib::rules::replace_all, vec![],
            "Applies replacement rules to all matching subexpressions"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "ReplaceRepeated", crate::stdlib::rules::replace_repeated, vec![],
            "Repeatedly applies replacement rules until no more changes"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "Rule", crate::stdlib::rules::rule, vec![],
            "Creates a replacement rule"
        ))?;
        rules_module.add_export(super::stdlib_to_export_with_docs(
            "RuleDelayed", crate::stdlib::rules::rule_delayed, vec![],
            "Creates a delayed replacement rule"
        ))?;
        
        self.register_module("std::rules", rules_module)?;
        
        // Create std::table module for data manipulation
        let mut table_module = Module::new(super::ModuleMetadata {
            name: "std::table".to_string(),
            version: Version::new(0, 1, 0),
            description: "Data table and series manipulation functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["table", "data", "series", "database"].into_iter().map(String::from).collect(),
            categories: vec!["data-manipulation".to_string()],
        });
        
        // Add legacy table functions
        table_module.add_export(super::stdlib_to_export_with_docs(
            "GroupBy", crate::stdlib::table::group_by, vec![],
            "Groups table rows by specified columns"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Aggregate", crate::stdlib::table::aggregate, vec![],
            "Applies aggregation functions to grouped data"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Count", crate::stdlib::table::count, vec![],
            "Counts occurrences or rows in data"
        ))?;
        
        // Add foreign table constructors
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Table", crate::stdlib::table::table, vec![],
            "Creates a table from data"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "TableFromRows", crate::stdlib::table::table_from_rows, vec![],
            "Creates a table from row data"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "EmptyTable", crate::stdlib::table::empty_table, vec![],
            "Creates an empty table with specified schema"
        ))?;
        
        // Add foreign series constructors
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Series", crate::stdlib::table::series, vec![],
            "Creates a data series"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Range", crate::stdlib::table::range, vec![],
            "Creates a range of values"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Zeros", crate::stdlib::table::zeros, vec![],
            "Creates a series of zeros"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Ones", crate::stdlib::table::ones, vec![],
            "Creates a series of ones"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "ConstantSeries", crate::stdlib::table::constant_series, vec![],
            "Creates a series with constant values"
        ))?;
        
        // Add foreign tensor constructors
        table_module.add_export(super::stdlib_to_export_with_docs(
            "Tensor", crate::stdlib::table::tensor, vec![],
            "Creates a foreign tensor object"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "ZerosTensor", crate::stdlib::table::zeros_tensor, vec![],
            "Creates a tensor filled with zeros"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "OnesTensor", crate::stdlib::table::ones_tensor, vec![],
            "Creates a tensor filled with ones"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "EyeTensor", crate::stdlib::table::eye_tensor, vec![],
            "Creates an identity tensor"
        ))?;
        table_module.add_export(super::stdlib_to_export_with_docs(
            "RandomTensor", crate::stdlib::table::random_tensor, vec![],
            "Creates a tensor with random values"
        ))?;
        
        self.register_module("std::table", table_module)?;
        
        println!("✅ Registered {} standard library modules with complete stdlib coverage", 8);
        Ok(())
    }
    
    /// Infer function arity from name (simplified heuristic)
    fn infer_function_arity(&self, function_name: &str) -> u8 {
        match function_name {
            // 0-arity functions
            "Length" | "Head" | "Tail" | "Flatten" | "StringLength" | "EmptyTable" => 0,
            
            // 1-arity functions  
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" | "StringTake" | "StringDrop" | 
            "Array" | "Transpose" | "ArrayDimensions" | "ArrayRank" | "ArrayFlatten" |
            "Sigmoid" | "Tanh" | "Softmax" | "TestHold" | "TestHoldMultiple" |
            "Series" | "Range" | "Zeros" | "Ones" | "ZerosTensor" | "OnesTensor" |
            "EyeTensor" | "RandomTensor" | "Tensor" | "ConstantSeries" => 1,
            
            // ML 1-arity functions
            "TensorShape" | "TensorRank" | "TensorSize" | "TransposeLayer" | "Identity" | "Sequential" => 1,
            
            // 2-arity functions
            "Plus" | "Times" | "Power" | "Divide" | "Minus" | "Append" | "StringJoin" | 
            "Dot" | "Maximum" | "Map" | "Apply" | "ArrayReshape" | "MatchQ" | "Cases" |
            "CountPattern" | "Position" | "Rule" | "RuleDelayed" | "Table" | "TableFromRows" => 2,
            
            // ML 2-arity functions
            "ReshapeLayer" | "PermuteLayer" => 2,
            
            // 3-arity functions
            "ReplaceAll" | "ReplaceRepeated" | "GroupBy" | "Aggregate" | "Count" => 3,
            
            // ML variable arity functions (can take 1-3 arguments)
            "FlattenLayer" => 3,
            
            // Default to 1
            _ => 1,
        }
    }
}

/// Namespace resolution system
#[derive(Debug)]
pub struct NamespaceResolver {
    /// Mapping from namespace to module reference
    namespaces: HashMap<String, String>,  // namespace -> module_key
    
    /// Import aliases for namespaces
    aliases: HashMap<String, String>,     // alias -> full_namespace
    
    /// Wildcard imports (namespace -> imported_symbols)
    wildcard_imports: HashMap<String, Vec<String>>,
}

impl NamespaceResolver {
    pub fn new() -> Self {
        NamespaceResolver {
            namespaces: HashMap::new(),
            aliases: HashMap::new(),
            wildcard_imports: HashMap::new(),
        }
    }
    
    /// Register a namespace
    pub fn register_namespace(&mut self, namespace: &str, module: &Module) -> Result<(), ModuleError> {
        if self.namespaces.contains_key(namespace) {
            return Err(ModuleError::PackageError {
                message: format!("Namespace {} already exists", namespace),
            });
        }
        
        self.namespaces.insert(namespace.to_string(), module.metadata.name.clone());
        Ok(())
    }
    
    /// Resolve a qualified function name (e.g., "std::math::Sin")
    pub fn resolve_qualified(&self, qualified_name: &str) -> Result<String, ModuleError> {
        // Parse namespace and function
        let parts: Vec<&str> = qualified_name.split("::").collect();
        if parts.len() < 2 {
            return Err(ModuleError::InvalidModuleName {
                name: qualified_name.to_string(),
            });
        }
        
        let namespace = parts[..parts.len()-1].join("::");
        let _function_name = parts[parts.len()-1];
        
        // Check if namespace exists
        if !self.namespaces.contains_key(&namespace) {
            return Err(ModuleError::ModuleNotFound {
                name: namespace,
            });
        }
        
        Ok(qualified_name.to_string())
    }
    
    /// Resolve an unqualified function name using imports
    pub fn resolve_unqualified(&self, function_name: &str) -> Result<String, ModuleError> {
        // Check wildcard imports first
        for (namespace, symbols) in &self.wildcard_imports {
            if symbols.contains(&function_name.to_string()) {
                return Ok(format!("{}::{}", namespace, function_name));
            }
        }
        
        // Check aliases
        if let Some(full_namespace) = self.aliases.get(function_name) {
            return Ok(full_namespace.clone());
        }
        
        // Fallback to global scope (backwards compatibility)
        Ok(function_name.to_string())
    }
    
    /// Add an import alias
    pub fn add_alias(&mut self, alias: String, full_name: String) {
        self.aliases.insert(alias, full_name);
    }
    
    /// Add a wildcard import
    pub fn add_wildcard_import(&mut self, namespace: String, symbols: Vec<String>) {
        self.wildcard_imports.insert(namespace, symbols);
    }
    
    /// List all registered namespaces
    pub fn list_namespaces(&self) -> Vec<&String> {
        self.namespaces.keys().collect()
    }
    
    /// Check if a namespace is registered
    pub fn has_namespace(&self, namespace: &str) -> bool {
        self.namespaces.contains_key(namespace)
    }
    
    /// Clear all imports (for testing)
    pub fn clear_imports(&mut self) {
        self.aliases.clear();
        self.wildcard_imports.clear();
    }
}

impl Default for NamespaceResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linker::FunctionRegistry;

    #[test]
    fn test_namespace_resolver() {
        let mut resolver = NamespaceResolver::new();
        
        // Test qualified name resolution
        let metadata = super::super::ModuleMetadata {
            name: "test::module".to_string(),
            version: Version::new(1, 0, 0),
            description: "Test".to_string(),
            authors: vec![],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        };
        
        let module = Module::new(metadata);
        resolver.register_namespace("test::module", &module).unwrap();
        
        assert!(resolver.has_namespace("test::module"));
        assert!(!resolver.has_namespace("nonexistent"));
        
        // Test qualified resolution
        let result = resolver.resolve_qualified("test::module::func");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "test::module::func");
        
        // Test invalid qualified resolution
        let result = resolver.resolve_qualified("invalid");
        assert!(result.is_err());
    }

    #[test]
    fn test_module_registry() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = ModuleRegistry::new(func_registry);
        
        // Should have registered standard library modules
        assert!(registry.has_module("std::math"));
        assert!(registry.has_module("std::list"));
        assert!(registry.has_module("std::string"));
        assert!(registry.has_module("std::tensor"));
        assert!(registry.has_module("std::ml::core"));
        assert!(registry.has_module("std::ml::layers"));
        
        // Test module listing
        let modules = registry.list_modules();
        assert!(modules.len() >= 6);
        
        // Test module search
        let search_results = registry.search_modules("math");
        assert!(!search_results.is_empty());
    }

    #[test]
    fn test_module_exports() {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let registry = ModuleRegistry::new(func_registry);
        
        // Test getting module exports
        let math_exports = registry.get_module_exports("std::math");
        assert!(!math_exports.is_empty());
        assert!(math_exports.contains(&"Sin".to_string()));
        assert!(math_exports.contains(&"Cos".to_string()));
        
        let list_exports = registry.get_module_exports("std::list");
        assert!(!list_exports.is_empty());
        assert!(list_exports.contains(&"Length".to_string()));
        assert!(list_exports.contains(&"Head".to_string()));
        
        // Test ML module exports
        let ml_core_exports = registry.get_module_exports("std::ml::core");
        assert!(!ml_core_exports.is_empty());
        assert!(ml_core_exports.contains(&"TensorShape".to_string()));
        assert!(ml_core_exports.contains(&"TensorRank".to_string()));
        assert!(ml_core_exports.contains(&"TensorSize".to_string()));
        
        let ml_layers_exports = registry.get_module_exports("std::ml::layers");
        assert!(!ml_layers_exports.is_empty());
        assert!(ml_layers_exports.contains(&"FlattenLayer".to_string()));
        assert!(ml_layers_exports.contains(&"ReshapeLayer".to_string()));
        assert!(ml_layers_exports.contains(&"PermuteLayer".to_string()));
        assert!(ml_layers_exports.contains(&"TransposeLayer".to_string()));
        assert!(ml_layers_exports.contains(&"Sequential".to_string()));
        assert!(ml_layers_exports.contains(&"Identity".to_string()));
    }
}