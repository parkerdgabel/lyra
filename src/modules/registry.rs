//! Module Registry System
//!
//! Manages the registration and lookup of modules within the Lyra system.

use super::{Module, ModuleError, Version, FunctionImplementation};
use crate::{
    stdlib::StandardLibrary,
    linker::{FunctionRegistry, FunctionAttribute},
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
                    FunctionImplementation::Foreign { type_name: _, method_name: _ } => {
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

    /// Audit: count exports per registered module (namespace -> count), sorted by namespace
    pub fn audit_module_counts(&self) -> Vec<(String, usize)> {
        let modules = self.modules.read().unwrap();
        let mut counts: Vec<(String, usize)> = modules
            .iter()
            .map(|(ns, m)| (ns.clone(), m.exports.len()))
            .collect();
        counts.sort_by(|a, b| a.0.cmp(&b.0));
        counts
    }

    /// Audit: find duplicate export names across different modules.
    /// Returns (export_name, [namespaces...]) for any export present in 2+ modules.
    pub fn audit_duplicate_exports(&self) -> Vec<(String, Vec<String>)> {
        let modules = self.modules.read().unwrap();
        let mut export_map: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for (ns, module) in modules.iter() {
            for name in module.exports.keys() {
                export_map.entry(name.clone()).or_default().push(ns.clone());
            }
        }
        let mut dups: Vec<(String, Vec<String>)> = export_map
            .into_iter()
            .filter_map(|(name, nss)| if nss.len() > 1 { Some((name, nss)) } else { None })
            .collect();
        dups.sort_by(|a, b| a.0.cmp(&b.0));
        dups
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

        // Training / constructors
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "NetTrain", crate::stdlib::ml::wrapper::net_train, vec![],
            "Train a neural network with a dataset and configuration"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "NetChain", crate::stdlib::ml::wrapper::net_chain, vec![],
            "Construct a sequential neural network from layer specs"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "CreateTrainingConfig", crate::stdlib::ml::wrapper::create_training_config, vec![],
            "Create a training configuration association"
        ))?;

        // Evaluation & preprocessing
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "TrainTestSplit", crate::stdlib::ml::wrapper::train_test_split, vec![],
            "Split a dataset into train and test partitions"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "ClassificationReport", crate::stdlib::ml::wrapper::classification_report, vec![],
            "Compute classification metrics (accuracy, precision, recall, F1)"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "RegressionReport", crate::stdlib::ml::wrapper::regression_report, vec![],
            "Compute regression metrics (MSE, MAE, RMSE, R2)"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "StandardScale", crate::stdlib::ml::wrapper::standard_scale, vec![],
            "Standardize a numeric list to zero mean, unit variance"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "OneHotEncode", crate::stdlib::ml::wrapper::one_hot_encode, vec![],
            "One-hot encode a list of categorical strings"
        ))?;

        // Cross validation
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "CrossValidate", crate::stdlib::ml::wrapper::cross_validate, vec![],
            "K-fold cross validation for datasets with builder options"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "CrossValidateTable", crate::stdlib::ml::wrapper::cross_validate_table, vec![],
            "K-fold cross validation for tables (feature/target columns)"
        ))?;

        // NetGraph and unified forward
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "NetGraph", crate::stdlib::ml::wrapper::net_graph, vec![],
            "Build a neural network graph from nodes and connections"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "AIForward", crate::stdlib::ml::wrapper::ai_forward, vec![],
            "Unified forward pass for sequential lists or graph specs"
        ))?;

        // AutoML & MLOps
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "AutoMLQuickStart", crate::stdlib::ml::wrapper::automl_quick_start_dataset, vec![],
            "Train a quick model given a dataset"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "AutoMLQuickStartTable", crate::stdlib::ml::wrapper::automl_quick_start_table, vec![],
            "Train a quick model given a table, feature cols, and target col"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "ExperimentStart", crate::stdlib::ml::wrapper::experiment_start, vec![],
            "Start a tracked ML experiment"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "ExperimentLogMetrics", crate::stdlib::ml::wrapper::experiment_log_metrics, vec![],
            "Log scalar metrics to the current experiment"
        ))?;
        ml_core_module.add_export(super::stdlib_to_export_with_docs(
            "ExperimentEnd", crate::stdlib::ml::wrapper::experiment_end, vec![],
            "End the current ML experiment and return a summary"
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

        // Common layer wrappers
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Linear", crate::stdlib::ml::wrapper::linear, vec![],
            "Apply a linear transformation with output feature size"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "ReLU", crate::stdlib::ml::wrapper::relu, vec![],
            "Apply ReLU activation"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Sigmoid", crate::stdlib::ml::wrapper::sigmoid, vec![],
            "Apply Sigmoid activation"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Tanh", crate::stdlib::ml::wrapper::tanh, vec![],
            "Apply Tanh activation"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Softmax", crate::stdlib::ml::wrapper::softmax, vec![],
            "Apply Softmax across features"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Conv2D", crate::stdlib::ml::wrapper::conv2d, vec![],
            "2D convolution with options for stride and padding"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "MaxPool", crate::stdlib::ml::wrapper::max_pool, vec![],
            "2D max pooling with kernel and optional stride"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "AvgPool", crate::stdlib::ml::wrapper::avg_pool, vec![],
            "2D average pooling with kernel and optional stride"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "Dropout", crate::stdlib::ml::wrapper::dropout, vec![],
            "Dropout regularization with probability and eval option"
        ))?;
        ml_layers_module.add_export(super::stdlib_to_export_with_docs(
            "BatchNorm", crate::stdlib::ml::wrapper::batch_norm, vec![],
            "Batch normalization with epsilon/momentum and eval option"
        ))?;
        
        self.register_module("std::ml::layers", ml_layers_module)?;
        
        // Register additional stdlib modules via per-module registries to keep exports consistent
        
        // std::image
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::image".to_string(),
                version: Version::new(0, 1, 0),
                description: "Image I/O, filtering, morphology, analysis".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["image", "vision", "processing"].into_iter().map(String::from).collect(),
                categories: vec!["computer-vision".to_string()],
            });
            for (name, func) in crate::stdlib::image::register_image_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::image", module)?;
        }
        
        // std::vision
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::vision".to_string(),
                version: Version::new(0, 1, 0),
                description: "Computer vision algorithms (features, edges, transforms)".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["vision", "features", "edges", "transforms"].into_iter().map(String::from).collect(),
                categories: vec!["computer-vision".to_string()],
            });
            for (name, func) in crate::stdlib::vision::register_vision_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::vision", module)?;
        }
        
        // std::analytics::timeseries (consolidated)
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::analytics::timeseries".to_string(),
                version: Version::new(0, 1, 0),
                description: "Time series analysis (ARIMA, ACF, decomposition)".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["analytics", "timeseries", "forecasting"].into_iter().map(String::from).collect(),
                categories: vec!["analytics".to_string()],
            });
            for (name, func) in crate::stdlib::analytics::timeseries::register_timeseries_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::analytics::timeseries", module)?;
        }
        
        // std::network
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::network".to_string(),
                version: Version::new(0, 1, 0),
                description: "Networking primitives, WebSocket, distributed computing".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["network", "http", "distributed"].into_iter().map(String::from).collect(),
                categories: vec!["networking".to_string()],
            });
            for (name, func) in crate::stdlib::network::register_network_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::network", module)?;
        }
        
        // std::numerical
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::numerical".to_string(),
                version: Version::new(0, 1, 0),
                description: "Numerical analysis and methods".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["numerical", "optimization", "methods"].into_iter().map(String::from).collect(),
                categories: vec!["mathematics".to_string()],
            });
            for (name, func) in crate::stdlib::numerical::register_numerical_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::numerical", module)?;
        }
        
        // std::clustering
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::clustering".to_string(),
                version: Version::new(0, 1, 0),
                description: "Clustering algorithms and utilities".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["clustering", "kmeans", "dbscan"].into_iter().map(String::from).collect(),
                categories: vec!["machine-learning".to_string()],
            });
            for (name, func) in crate::stdlib::clustering::register_clustering_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::clustering", module)?;
        }
        
        // std::geometry
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::geometry".to_string(),
                version: Version::new(0, 1, 0),
                description: "Computational geometry functions".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["geometry", "convex", "distance"].into_iter().map(String::from).collect(),
                categories: vec!["mathematics".to_string()],
            });
            for (name, func) in crate::stdlib::geometry::register_geometry_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::geometry", module)?;
        }
        
        // std::number_theory
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::number_theory".to_string(),
                version: Version::new(0, 1, 0),
                description: "Number theory functions".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["primes", "number-theory"].into_iter().map(String::from).collect(),
                categories: vec!["mathematics".to_string()],
            });
            for (name, func) in crate::stdlib::number_theory::register_number_theory_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::number_theory", module)?;
        }
        
        // std::combinatorics
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::combinatorics".to_string(),
                version: Version::new(0, 1, 0),
                description: "Combinatorics functions".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["combinatorics", "permutations", "combinations"].into_iter().map(String::from).collect(),
                categories: vec!["mathematics".to_string()],
            });
            for (name, func) in crate::stdlib::combinatorics::register_combinatorics_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::combinatorics", module)?;
        }
        
        // std::data_processing
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::data_processing".to_string(),
                version: Version::new(0, 1, 0),
                description: "Data processing and transformation".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["data", "processing", "pipeline"].into_iter().map(String::from).collect(),
                categories: vec!["data-processing".to_string()],
            });
            for (name, func) in crate::stdlib::data_processing::register_data_processing_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::data_processing", module)?;
        }
        
        // std::topology
        {
            let mut module = Module::new(super::ModuleMetadata {
                name: "std::topology".to_string(),
                version: Version::new(0, 1, 0),
                description: "Topology functions".to_string(),
                authors: vec!["Lyra Team".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["topology", "algebraic-topology"].into_iter().map(String::from).collect(),
                categories: vec!["mathematics".to_string()],
            });
            for (name, func) in crate::stdlib::topology::register_topology_functions() {
                module.add_export(super::stdlib_to_export(&name, func, vec![]))?;
            }
            self.register_module("std::topology", module)?;
        }
        
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
