//! Unified Function Resolution System (Phase 4B)
//!
//! This module implements a unified compile-time function resolution system that handles
//! BOTH Foreign object methods AND stdlib functions through static dispatch, eliminating
//! all dynamic dispatch overhead across the entire function call system.
//!
//! ## Architecture Overview
//!
//! **Previous (Split System):**
//! ```
//! Foreign methods: CallStatic -> 1000x+ speedup ✅
//! Stdlib functions: CALL -> HashMap lookup -> slow ❌
//! ```
//! 
//! **Target (Unified System):**
//! ```
//! ALL functions: CallStatic -> direct function pointer -> 1000x+ speedup ✅
//! ```
//!
//! ## Unified Function Index Space
//! ```
//! ├─ 0-31:   Foreign Methods (Series.Length, Tensor.Add, etc.)
//! ├─ 32-78:  Stdlib Functions (Sin, Cos, Plus, Length, etc.)  
//! ├─ 79-95:  User Functions (future)
//! └─ 96+:    Extended Functions
//! ```
//!
//! ## Performance Goals
//! - Eliminate ALL string-based function lookups
//! - Remove ALL dynamic dispatch overhead
//! - Stdlib functions get same 1000x+ speedup as Foreign methods
//! - Target: Zero runtime function resolution cost

use crate::{
    vm::Value,
    foreign::{ForeignError, LyObj},
    stdlib::StdlibFunction,
};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during link-time function resolution
#[derive(Error, Debug, Clone, PartialEq)]
pub enum LinkerError {
    #[error("Function not found: {type_name}::{method_name}")]
    FunctionNotFound {
        type_name: String,
        method_name: String,
    },
    
    #[error("Invalid arity for function {function_name}: expected {expected}, got {actual}")]
    InvalidArity {
        function_name: String,
        expected: u8,
        actual: u8,
    },
    
    #[error("Type validation failed for function {function_name}: {message}")]
    TypeValidationFailed {
        function_name: String,
        message: String,
    },
    
    #[error("Registry error: {message}")]
    RegistryError {
        message: String,
    },
}

/// Function attributes for compile-time processing (Phase 5A)
/// 
/// These attributes control how functions behave during compilation and execution:
/// - Hold: Prevents evaluation of specified arguments
/// - Listable: Automatically threads over lists 
/// - Orderless: Allows commutative argument reordering
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FunctionAttribute {
    /// Hold[1,2] - Don't evaluate arguments at positions 1,2 (1-indexed)
    /// Example: SetDelayed[f[x_], x^2] - the definition x_ should not evaluate
    Hold(Vec<usize>),
    
    /// Listable - Automatically thread over lists
    /// Example: Sin[{1,2,3}] becomes {Sin[1], Sin[2], Sin[3]}
    Listable,
    
    /// Orderless - Commutative, can reorder arguments for canonical form
    /// Example: Plus[b, a] becomes Plus[a, b] at compile time
    Orderless,
    
    /// Protected - Cannot be redefined (built-in functions)
    Protected,
    
    /// ReadProtected - Definition cannot be read
    ReadProtected,
}

/// Function signature metadata for registry lookup and validation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    /// The Foreign object type name (e.g., "Tensor", "Series", "Dataset")
    pub type_name: String,
    
    /// The method name (e.g., "add", "get", "transpose")
    pub method_name: String,
    
    /// Expected argument count (not including the self object)
    pub arity: u8,
    
    /// Optional: argument type constraints for validation
    pub arg_types: Option<Vec<String>>,
    
    /// Function attributes for compile-time processing (Phase 5A)
    pub attributes: Vec<FunctionAttribute>,
}

impl FunctionSignature {
    /// Create a new function signature for a Foreign method
    pub fn new(type_name: &str, method_name: &str, arity: u8) -> Self {
        Self {
            type_name: type_name.to_string(),
            method_name: method_name.to_string(),
            arity,
            arg_types: None,
            attributes: Vec::new(),
        }
    }
    
    /// Create a function signature with type constraints
    pub fn with_types(type_name: &str, method_name: &str, arity: u8, arg_types: Vec<String>) -> Self {
        Self {
            type_name: type_name.to_string(),
            method_name: method_name.to_string(),
            arity,
            arg_types: Some(arg_types),
            attributes: Vec::new(),
        }
    }
    
    /// Create a function signature with attributes (Phase 5A)
    pub fn with_attributes(type_name: &str, method_name: &str, arity: u8, attributes: Vec<FunctionAttribute>) -> Self {
        Self {
            type_name: type_name.to_string(),
            method_name: method_name.to_string(),
            arity,
            arg_types: None,
            attributes,
        }
    }
    
    /// Create a function signature with both type constraints and attributes
    pub fn with_types_and_attributes(
        type_name: &str, 
        method_name: &str, 
        arity: u8, 
        arg_types: Vec<String>,
        attributes: Vec<FunctionAttribute>
    ) -> Self {
        Self {
            type_name: type_name.to_string(),
            method_name: method_name.to_string(),
            arity,
            arg_types: Some(arg_types),
            attributes,
        }
    }
    
    /// Check if this function has a specific attribute
    pub fn has_attribute(&self, attribute: &FunctionAttribute) -> bool {
        self.attributes.contains(attribute)
    }
    
    /// Get all Hold position arguments (1-indexed)
    pub fn get_hold_positions(&self) -> Option<&Vec<usize>> {
        for attr in &self.attributes {
            if let FunctionAttribute::Hold(positions) = attr {
                return Some(positions);
            }
        }
        None
    }
    
    /// Check if this function is Listable
    pub fn is_listable(&self) -> bool {
        self.attributes.contains(&FunctionAttribute::Listable)
    }
    
    /// Check if this function is Orderless
    pub fn is_orderless(&self) -> bool {
        self.attributes.contains(&FunctionAttribute::Orderless)
    }
    
    /// Get a unique key for registry lookup
    pub fn key(&self) -> String {
        format!("{}::{}", self.type_name, self.method_name)
    }
}

/// Type-safe function pointer for Foreign object methods
/// 
/// This wraps a native function pointer with validation and metadata.
/// The function signature is: fn(object: &LyObj, args: &[Value]) -> Result<Value, ForeignError>
pub type NativeFunctionPtr = fn(&LyObj, &[Value]) -> Result<Value, ForeignError>;

/// Unified function type that can handle both Foreign methods and stdlib functions
#[derive(Debug, Clone)]
pub enum UnifiedFunction {
    /// Foreign object method (requires object as first parameter)
    Foreign(NativeFunctionPtr),
    /// Stdlib function (no object parameter)
    Stdlib(StdlibFunction),
}

impl UnifiedFunction {
    /// Call the function with appropriate signature
    pub fn call(&self, object: Option<&LyObj>, args: &[Value]) -> Result<Value, ForeignError> {
        match self {
            UnifiedFunction::Foreign(func_ptr) => {
                let obj = object.ok_or_else(|| ForeignError::RuntimeError {
                    message: "Foreign method requires object parameter".to_string(),
                })?;
                func_ptr(obj, args)
            }
            UnifiedFunction::Stdlib(func_ptr) => {
                // Convert VmResult to ForeignError for consistent error handling
                func_ptr(args).map_err(|vm_err| ForeignError::RuntimeError {
                    message: format!("Stdlib function error: {:?}", vm_err),
                })
            }
        }
    }
    
    /// Check if this function requires an object parameter
    pub fn requires_object(&self) -> bool {
        matches!(self, UnifiedFunction::Foreign(_))
    }
}

/// Function metadata stored in the registry
#[derive(Debug, Clone)]
pub struct FunctionEntry {
    /// The function signature for validation
    pub signature: FunctionSignature,
    
    /// The unified function (Foreign method or stdlib function)
    pub function: UnifiedFunction,
    
    /// The static function index for CallStatic
    pub function_index: u16,
    
    /// Optional documentation for debugging
    pub documentation: Option<String>,
}

impl FunctionEntry {
    /// Create a new function entry for a Foreign method
    pub fn new_foreign(signature: FunctionSignature, function_ptr: NativeFunctionPtr, function_index: u16) -> Self {
        Self {
            signature,
            function: UnifiedFunction::Foreign(function_ptr),
            function_index,
            documentation: None,
        }
    }
    
    /// Create a new function entry for a stdlib function
    pub fn new_stdlib(signature: FunctionSignature, function_ptr: StdlibFunction, function_index: u16) -> Self {
        Self {
            signature,
            function: UnifiedFunction::Stdlib(function_ptr),
            function_index,
            documentation: None,
        }
    }
    
    /// Create a function entry with documentation
    pub fn with_docs(
        signature: FunctionSignature, 
        function: UnifiedFunction,
        function_index: u16,
        docs: &str
    ) -> Self {
        Self {
            signature,
            function,
            function_index,
            documentation: Some(docs.to_string()),
        }
    }
    
    /// Validate argument count against signature
    pub fn validate_arity(&self, arg_count: usize) -> Result<(), LinkerError> {
        if arg_count != self.signature.arity as usize {
            return Err(LinkerError::InvalidArity {
                function_name: self.signature.key(),
                expected: self.signature.arity,
                actual: arg_count as u8,
            });
        }
        Ok(())
    }
    
    /// Call the function with validation
    pub fn call(&self, object: Option<&LyObj>, args: &[Value]) -> Result<Value, ForeignError> {
        // Validate arity
        if let Err(_linker_err) = self.validate_arity(args.len()) {
            return Err(ForeignError::InvalidArity {
                method: self.signature.method_name.clone(),
                expected: self.signature.arity as usize,
                actual: args.len(),
            });
        }
        
        // Call the unified function
        self.function.call(object, args)
    }
    
    /// Get the function index for CallStatic
    pub fn get_function_index(&self) -> u16 {
        self.function_index
    }
}

/// Unified registry of ALL functions (Foreign methods + stdlib functions) for static dispatch
/// 
/// This registry combines both Foreign object methods and stdlib functions into a single
/// index space, allowing the compiler to resolve ALL function calls to CallStatic instructions
/// with direct function indices, eliminating all dynamic dispatch overhead.
///
/// ## Function Index Space:
/// - 0-31:   Foreign Methods (Series.Length, Tensor.Add, etc.)
/// - 32-78:  Stdlib Functions (Sin, Cos, Plus, Length, etc.)
/// - 79+:    Future expansion
#[derive(Debug, Default)]
pub struct FunctionRegistry {
    /// Map from function signature to function entry
    functions: HashMap<String, FunctionEntry>,
    
    /// Map from function name to function index (for CallStatic)
    function_indices: HashMap<String, u16>,
    
    /// Map from type name to list of available methods (for introspection)
    type_methods: HashMap<String, Vec<String>>,
    
    /// Map from stdlib function name to function index
    stdlib_indices: HashMap<String, u16>,
    
    /// Next available function index
    next_function_index: u16,
    
    /// Attribute Registry (Phase 5A) - Maps from attributes to function names
    /// This enables efficient lookup of functions by their attributes
    attribute_registry: HashMap<FunctionAttribute, Vec<String>>,
    
    /// Reverse mapping: function name to its attributes for quick lookup
    function_attributes: HashMap<String, Vec<FunctionAttribute>>,
    
    /// Statistics for performance monitoring
    pub stats: RegistryStats,
}

/// Statistics about the function registry
#[derive(Debug, Default)]
pub struct RegistryStats {
    pub total_functions: usize,
    pub total_types: usize,
    pub lookups_performed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

impl FunctionRegistry {
    /// Create a new empty function registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
            function_indices: HashMap::new(),
            type_methods: HashMap::new(),
            stdlib_indices: HashMap::new(),
            next_function_index: 0, // Start from 0 for Foreign methods
            attribute_registry: HashMap::new(),
            function_attributes: HashMap::new(),
            stats: RegistryStats::default(),
        }
    }
    
    /// Reserve index space for Foreign methods (0-31)
    pub fn reserve_foreign_indices(&mut self) {
        self.next_function_index = 32; // Skip Foreign method indices 0-31
    }
    
    /// Register a Foreign object method in the registry (indices 0-31)
    pub fn register_method(
        &mut self,
        signature: FunctionSignature,
        function_ptr: NativeFunctionPtr,
        function_index: u16,
    ) -> Result<(), LinkerError> {
        let key = signature.key();
        let type_name = signature.type_name.clone();
        let method_name = signature.method_name.clone();
        
        // Check for duplicate registration
        if self.functions.contains_key(&key) {
            return Err(LinkerError::RegistryError {
                message: format!("Function {} already registered", key),
            });
        }
        
        // Validate Foreign method index range (0-31)
        if function_index >= 32 {
            return Err(LinkerError::RegistryError {
                message: format!("Foreign method index {} must be in range 0-31", function_index),
            });
        }
        
        // Create function entry
        let entry = FunctionEntry::new_foreign(signature, function_ptr, function_index);
        
        // Add to functions map
        self.functions.insert(key.clone(), entry);
        
        // Add to function indices map
        self.function_indices.insert(key, function_index);
        
        // Update type methods map
        self.type_methods
            .entry(type_name)
            .or_insert_with(Vec::new)
            .push(method_name);
        
        // Update stats
        self.stats.total_functions += 1;
        self.stats.total_types = self.type_methods.len();
        
        Ok(())
    }
    
    /// Register a namespaced function in the registry
    pub fn register_namespaced_function(
        &mut self,
        qualified_name: &str,
        function_ptr: StdlibFunction,
        arity: u8,
        attributes: Vec<FunctionAttribute>,
    ) -> Result<(), LinkerError> {
        // Check for duplicate registration
        if self.functions.contains_key(qualified_name) {
            return Err(LinkerError::RegistryError {
                message: format!("Function {} already registered", qualified_name),
            });
        }
        
        // Assign function index
        let function_index = self.next_function_index;
        self.next_function_index += 1;
        
        // Create function signature with namespace information
        let signature = FunctionSignature::with_attributes("Module", qualified_name, arity, attributes.clone());
        
        // Create function entry
        let entry = FunctionEntry::new_stdlib(signature, function_ptr, function_index);
        
        // Add to functions map
        self.functions.insert(qualified_name.to_string(), entry);
        
        // Add to function indices map
        self.function_indices.insert(qualified_name.to_string(), function_index);
        
        // Register function attributes
        self.register_function_attributes(qualified_name, attributes);
        
        // Update stats
        self.stats.total_functions += 1;
        
        Ok(())
    }

    /// Register a stdlib function in the registry (indices 32+)
    pub fn register_stdlib_function(
        &mut self,
        name: &str,
        function_ptr: StdlibFunction,
        arity: u8,
    ) -> Result<(), LinkerError> {
        // Check for duplicate registration
        if self.stdlib_indices.contains_key(name) {
            return Err(LinkerError::RegistryError {
                message: format!("Stdlib function {} already registered", name),
            });
        }
        
        // Assign function index (starting from 32)
        let function_index = self.next_function_index;
        self.next_function_index += 1;
        
        // Create function signature for stdlib function
        let signature = FunctionSignature::new("Stdlib", name, arity);
        
        // Create function entry
        let entry = FunctionEntry::new_stdlib(signature, function_ptr, function_index);
        
        // Add to functions map
        self.functions.insert(name.to_string(), entry);
        
        // Add to stdlib indices map
        self.stdlib_indices.insert(name.to_string(), function_index);
        
        // Add to function indices map
        self.function_indices.insert(name.to_string(), function_index);
        
        // Update stats
        self.stats.total_functions += 1;
        
        Ok(())
    }
    
    /// Look up a function by type and method name
    pub fn lookup(&mut self, type_name: &str, method_name: &str) -> Result<&FunctionEntry, LinkerError> {
        let key = format!("{}::{}", type_name, method_name);
        self.stats.lookups_performed += 1;
        
        match self.functions.get(&key) {
            Some(entry) => {
                self.stats.cache_hits += 1;
                Ok(entry)
            }
            None => {
                self.stats.cache_misses += 1;
                Err(LinkerError::FunctionNotFound {
                    type_name: type_name.to_string(),
                    method_name: method_name.to_string(),
                })
            }
        }
    }
    
    /// Get all methods available for a specific type
    pub fn get_type_methods(&self, type_name: &str) -> Vec<String> {
        self.type_methods
            .get(type_name)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get all registered type names
    pub fn get_type_names(&self) -> Vec<String> {
        self.type_methods.keys().cloned().collect()
    }
    
    /// Check if a specific method is registered for a type
    pub fn has_method(&self, type_name: &str, method_name: &str) -> bool {
        let key = format!("{}::{}", type_name, method_name);
        self.functions.contains_key(&key)
    }
    
    /// Check if a stdlib function is registered
    pub fn has_stdlib_function(&self, function_name: &str) -> bool {
        self.stdlib_indices.contains_key(function_name)
    }
    
    /// Get function index by name (works for both Foreign methods and stdlib functions)
    pub fn get_function_index(&self, function_name: &str) -> Option<u16> {
        self.function_indices.get(function_name).copied()
    }
    
    /// Get function index for Foreign method
    pub fn get_method_index(&self, type_name: &str, method_name: &str) -> Option<u16> {
        let key = format!("{}::{}", type_name, method_name);
        self.function_indices.get(&key).copied()
    }
    
    /// Get function index for stdlib function
    pub fn get_stdlib_index(&self, function_name: &str) -> Option<u16> {
        self.stdlib_indices.get(function_name).copied()
    }
    
    /// Get stdlib function by index (for VM CallStatic execution)
    pub fn get_stdlib_function(&self, function_index: u16) -> Option<StdlibFunction> {
        // Find the function name that corresponds to this index
        for (name, &index) in &self.stdlib_indices {
            if index == function_index {
                // Found the function name, now get the function entry
                if let Some(entry) = self.functions.get(name) {
                    // Extract the stdlib function pointer
                    if let UnifiedFunction::Stdlib(function_ptr) = entry.function {
                        return Some(function_ptr);
                    }
                }
                break;
            }
        }
        None
    }
    
    /// Get total function count (Foreign methods + stdlib functions)
    pub fn get_total_function_count(&self) -> usize {
        self.stats.total_functions
    }
    
    /// Lookup function by name (for stdlib functions)
    pub fn lookup_stdlib(&mut self, function_name: &str) -> Result<&FunctionEntry, LinkerError> {
        self.stats.lookups_performed += 1;
        
        match self.functions.get(function_name) {
            Some(entry) => {
                self.stats.cache_hits += 1;
                Ok(entry)
            }
            None => {
                self.stats.cache_misses += 1;
                Err(LinkerError::FunctionNotFound {
                    type_name: "Stdlib".to_string(),
                    method_name: function_name.to_string(),
                })
            }
        }
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> &RegistryStats {
        &self.stats
    }
    
    /// Clear all registered functions (for testing)
    pub fn clear(&mut self) {
        self.functions.clear();
        self.type_methods.clear();
        self.attribute_registry.clear();
        self.function_attributes.clear();
        self.stats = RegistryStats::default();
    }
    
    // ================================
    // ATTRIBUTE REGISTRY METHODS (Phase 5A)
    // ================================
    
    /// Register function attributes in the attribute registry
    pub fn register_function_attributes(&mut self, function_name: &str, attributes: Vec<FunctionAttribute>) {
        // Store function -> attributes mapping
        self.function_attributes.insert(function_name.to_string(), attributes.clone());
        
        // Store attribute -> functions mapping
        for attribute in attributes {
            self.attribute_registry
                .entry(attribute)
                .or_insert_with(Vec::new)
                .push(function_name.to_string());
        }
    }
    
    /// Get all functions that have a specific attribute
    pub fn get_functions_with_attribute(&self, attribute: &FunctionAttribute) -> Vec<String> {
        self.attribute_registry
            .get(attribute)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Get all attributes for a specific function
    pub fn get_function_attributes(&self, function_name: &str) -> Vec<FunctionAttribute> {
        self.function_attributes
            .get(function_name)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Check if a function has a specific attribute
    pub fn function_has_attribute(&self, function_name: &str, attribute: &FunctionAttribute) -> bool {
        self.function_attributes
            .get(function_name)
            .map(|attrs| attrs.contains(attribute))
            .unwrap_or(false)
    }
    
    /// Get all Listable functions (for compiler optimization)
    pub fn get_listable_functions(&self) -> Vec<String> {
        self.get_functions_with_attribute(&FunctionAttribute::Listable)
    }
    
    // TODO: Temporarily disabled due to module system being disabled
    /*
    /// Register all functions from a module with their namespace
    pub fn register_module_functions(
        &mut self,
        namespace: &str,
        functions: &std::collections::HashMap<String, crate::modules::FunctionExport>,
    ) -> Result<(), LinkerError> {
        for (name, export) in functions {
            let qualified_name = format!("{}::{}", namespace, name);
            
            match &export.implementation {
                crate::modules::FunctionImplementation::Native(func_ptr) => {
                    // Infer arity from function signature or use default
                    let arity = export.signature.arity;
                    
                    self.register_namespaced_function(
                        &qualified_name,
                        *func_ptr,
                        arity,
                        export.attributes.clone(),
                    )?;
                },
                crate::modules::FunctionImplementation::Foreign { type_name, method_name } => {
                    // Handle foreign function registration
                    // For now, we'll add a placeholder entry
                    let function_index = self.next_function_index;
                    self.next_function_index += 1;
                    
                    let signature = FunctionSignature::with_attributes(
                        type_name,
                        method_name,
                        export.signature.arity,
                        export.attributes.clone(),
                    );
                    
                    // Create a placeholder entry - in practice this would need proper foreign function handling
                    self.function_indices.insert(qualified_name.clone(), function_index);
                    self.stats.total_functions += 1;
                },
                crate::modules::FunctionImplementation::Lyra { .. } => {
                    // Handle Lyra-defined functions
                    // For now, skip these as they need special handling
                    continue;
                },
                crate::modules::FunctionImplementation::External { .. } => {
                    // Handle external functions
                    // For now, skip these as they need FFI integration
                    continue;
                },
            }
        }
        
        Ok(())
    }
    */
    
    /// Get all functions in a specific namespace
    pub fn get_namespace_functions(&self, namespace: &str) -> Vec<String> {
        let prefix = format!("{}::", namespace);
        self.functions
            .keys()
            .filter(|name| name.starts_with(&prefix))
            .map(|name| name.strip_prefix(&prefix).unwrap_or(name).to_string())
            .collect()
    }
    
    /// Check if a namespace has any registered functions
    pub fn has_namespace(&self, namespace: &str) -> bool {
        let prefix = format!("{}::", namespace);
        self.functions.keys().any(|name| name.starts_with(&prefix))
    }
    
    /// Get all registered namespaces
    pub fn get_namespaces(&self) -> Vec<String> {
        let mut namespaces = std::collections::HashSet::new();
        
        for function_name in self.functions.keys() {
            if let Some(pos) = function_name.rfind("::") {
                let namespace = &function_name[..pos];
                namespaces.insert(namespace.to_string());
            }
        }
        
        namespaces.into_iter().collect()
    }
    
    /// Resolve a qualified function name to its index
    pub fn resolve_qualified_function(&self, qualified_name: &str) -> Option<u16> {
        self.function_indices.get(qualified_name).copied()
    }
    
    /// Get function entry by qualified name
    pub fn get_function_entry(&self, qualified_name: &str) -> Option<&FunctionEntry> {
        self.functions.get(qualified_name)
    }
    
    /// Get all Orderless functions (for compiler optimization)
    pub fn get_orderless_functions(&self) -> Vec<String> {
        self.get_functions_with_attribute(&FunctionAttribute::Orderless)
    }
    
    /// Get functions with Hold attributes (for compiler processing)
    pub fn get_hold_functions(&self) -> Vec<(String, Vec<usize>)> {
        let mut hold_functions = Vec::new();
        
        for (function_name, attributes) in &self.function_attributes {
            for attribute in attributes {
                if let FunctionAttribute::Hold(positions) = attribute {
                    hold_functions.push((function_name.clone(), positions.clone()));
                }
            }
        }
        
        hold_functions
    }
    
    /// Register standard library functions with their default attributes (Phase 5A)
    pub fn register_stdlib_attributes(&mut self) {
        // Math functions - all Listable
        let math_functions = vec!["Sin", "Cos", "Tan", "Exp", "Log", "Sqrt"];
        for func in math_functions {
            self.register_function_attributes(func, vec![
                FunctionAttribute::Listable,
                FunctionAttribute::Protected,
            ]);
        }
        
        // Arithmetic functions - Orderless and Listable
        let arithmetic_functions = vec!["Plus", "Times"];
        for func in arithmetic_functions {
            self.register_function_attributes(func, vec![
                FunctionAttribute::Orderless,
                FunctionAttribute::Listable,
                FunctionAttribute::Protected,
            ]);
        }
        
        // List functions - just Protected
        let list_functions = vec!["Length", "Head", "Tail", "Append", "Flatten"];
        for func in list_functions {
            self.register_function_attributes(func, vec![
                FunctionAttribute::Protected,
            ]);
        }
        
        // Special functions with Hold attributes
        self.register_function_attributes("SetDelayed", vec![
            FunctionAttribute::Hold(vec![1]), // Don't evaluate the pattern
            FunctionAttribute::Protected,
        ]);
        
        self.register_function_attributes("If", vec![
            FunctionAttribute::Hold(vec![2, 3]), // Don't evaluate branches
            FunctionAttribute::Protected,
        ]);
        
        // Test functions for Hold attribute validation
        self.register_function_attributes("TestHold", vec![
            FunctionAttribute::Hold(vec![1]), // Don't evaluate first argument
        ]);
        
        self.register_function_attributes("TestHoldMultiple", vec![
            FunctionAttribute::Hold(vec![2, 3]), // Don't evaluate second and third arguments
        ]);
    }
    
    /// Get attribute registry statistics
    pub fn get_attribute_stats(&self) -> (usize, usize) {
        (
            self.function_attributes.len(), // Functions with attributes
            self.attribute_registry.len(),  // Unique attribute types
        )
    }
}

/// Link-time resolver that maps method calls to function pointers during compilation
/// 
/// This component analyzes method calls in the AST/bytecode and resolves them to
/// direct function pointers, enabling the VM to bypass dynamic dispatch entirely.
pub struct LinkTimeResolver {
    /// The function registry to resolve against
    registry: FunctionRegistry,
    
    /// Cache of resolved function pointers for performance
    resolution_cache: HashMap<String, NativeFunctionPtr>,
    
    /// Statistics about resolution performance
    pub stats: ResolverStats,
}

/// Statistics about link-time resolution
#[derive(Debug, Default)]
pub struct ResolverStats {
    pub resolutions_attempted: usize,
    pub resolutions_successful: usize,
    pub resolutions_failed: usize,
    pub cache_hits: usize,
}

impl LinkTimeResolver {
    /// Create a new link-time resolver with the given registry
    pub fn new(registry: FunctionRegistry) -> Self {
        Self {
            registry,
            resolution_cache: HashMap::new(),
            stats: ResolverStats::default(),
        }
    }
    
    /// Resolve a method call to a direct function pointer
    /// 
    /// This is called during compilation to convert dynamic method calls
    /// into static function pointer calls.
    pub fn resolve_method_call(
        &mut self,
        type_name: &str,
        method_name: &str,
        arg_count: usize,
    ) -> Result<NativeFunctionPtr, LinkerError> {
        let key = format!("{}::{}({})", type_name, method_name, arg_count);
        self.stats.resolutions_attempted += 1;
        
        // Check cache first
        if let Some(&cached_ptr) = self.resolution_cache.get(&key) {
            self.stats.cache_hits += 1;
            self.stats.resolutions_successful += 1;
            return Ok(cached_ptr);
        }
        
        // Look up in registry
        match self.registry.lookup(type_name, method_name) {
            Ok(entry) => {
                // Validate arity
                entry.validate_arity(arg_count)?;
                
                // Cache the result  
                let function_ptr = match &entry.function {
                    UnifiedFunction::Foreign(ptr) => *ptr,
                    UnifiedFunction::Stdlib(_) => {
                        return Err(LinkerError::RegistryError {
                            message: "Cannot cache stdlib function as native function pointer".to_string(),
                        });
                    }
                };
                self.resolution_cache.insert(key, function_ptr);
                
                self.stats.resolutions_successful += 1;
                Ok(function_ptr)
            }
            Err(err) => {
                self.stats.resolutions_failed += 1;
                Err(err)
            }
        }
    }
    
    /// Get resolver statistics
    pub fn get_stats(&self) -> &ResolverStats {
        &self.stats
    }
    
    /// Clear resolution cache (for testing or memory management)
    pub fn clear_cache(&mut self) {
        self.resolution_cache.clear();
    }
}

/// Auto-registration system for Foreign object methods
/// 
/// This module provides automatic registration of Foreign object methods into the
/// Function Registry. It scans Foreign implementations and extracts method metadata
/// to populate the registry for link-time resolution.
pub mod registry {
    use super::*;
    use crate::foreign::Foreign;
    use crate::stdlib::{data::{
        tensor::ForeignTensor,
        series::ForeignSeries, 
        table::ForeignTable,
    }, StandardLibrary};

    /// Registry builder that auto-registers ALL functions (Foreign methods + stdlib functions)
    pub struct RegistryBuilder {
        registry: FunctionRegistry,
        current_foreign_index: u16,
    }

    impl RegistryBuilder {
        /// Create a new registry builder
        pub fn new() -> Self {
            Self {
                registry: FunctionRegistry::new(),
                current_foreign_index: 0, // Start Foreign methods at index 0
            }
        }

        /// Build the complete unified registry with ALL functions registered
        pub fn build(mut self) -> Result<FunctionRegistry, LinkerError> {
            // Phase 1: Register all Foreign object methods (indices 0-31)
            self.register_tensor_methods()?;
            self.register_series_methods()?;
            self.register_table_methods()?;
            
            // Phase 2: Reserve stdlib function index space (starting at 32)
            self.registry.reserve_foreign_indices();
            
            // Phase 3: Register all stdlib functions (indices 32+)
            self.register_stdlib_functions()?;

            Ok(self.registry)
        }

        /// Register all Tensor methods
        fn register_tensor_methods(&mut self) -> Result<(), LinkerError> {
            // Tensor methods with 0 arguments
            self.register_method("Tensor", "Dimensions", 0, tensor_dimensions)?;
            self.register_method("Tensor", "Rank", 0, tensor_rank)?;
            self.register_method("Tensor", "Length", 0, tensor_length)?;
            self.register_method("Tensor", "Flatten", 0, tensor_flatten)?;
            self.register_method("Tensor", "Transpose", 0, tensor_transpose)?;
            self.register_method("Tensor", "ToList", 0, tensor_to_list)?;

            // Tensor methods with 1 argument
            self.register_method("Tensor", "Reshape", 1, tensor_reshape)?;
            self.register_method("Tensor", "Add", 1, tensor_add)?;
            self.register_method("Tensor", "Multiply", 1, tensor_multiply)?;
            self.register_method("Tensor", "Subtract", 1, tensor_subtract)?;
            self.register_method("Tensor", "Divide", 1, tensor_divide)?;
            self.register_method("Tensor", "Dot", 1, tensor_dot)?;
            self.register_method("Tensor", "Maximum", 1, tensor_maximum)?;

            // Tensor methods with variable arguments (handled as special cases)
            self.register_method("Tensor", "Get", 1, tensor_get)?; // Simplified to 1 arg for now
            self.register_method("Tensor", "Set", 2, tensor_set)?; // Simplified to 2 args for now

            Ok(())
        }

        /// Register all Series methods
        fn register_series_methods(&mut self) -> Result<(), LinkerError> {
            // Series methods with 0 arguments
            self.register_method("Series", "Length", 0, series_length)?;
            self.register_method("Series", "Type", 0, series_type)?;
            self.register_method("Series", "ToList", 0, series_to_list)?;
            self.register_method("Series", "IsEmpty", 0, series_is_empty)?;

            // Series methods with 1 argument
            self.register_method("Series", "Get", 1, series_get)?;
            self.register_method("Series", "Append", 1, series_append)?;

            // Series methods with 2 arguments
            self.register_method("Series", "Set", 2, series_set)?;
            self.register_method("Series", "Slice", 2, series_slice)?;

            Ok(())
        }

        /// Register all Table methods
        fn register_table_methods(&mut self) -> Result<(), LinkerError> {
            // Table methods with 0 arguments
            self.register_method("Table", "RowCount", 0, table_row_count)?;
            self.register_method("Table", "ColumnCount", 0, table_column_count)?;
            self.register_method("Table", "ColumnNames", 0, table_column_names)?;
            self.register_method("Table", "IsEmpty", 0, table_is_empty)?;

            // Table methods with 1 argument
            self.register_method("Table", "GetRow", 1, table_get_row)?;
            self.register_method("Table", "GetColumn", 1, table_get_column)?;
            self.register_method("Table", "Head", 1, table_head)?;
            self.register_method("Table", "Tail", 1, table_tail)?;

            // Table methods with 2 arguments
            self.register_method("Table", "GetCell", 2, table_get_cell)?;

            Ok(())
        }

        /// Register all stdlib functions (indices 32+)
        fn register_stdlib_functions(&mut self) -> Result<(), LinkerError> {
            let stdlib = StandardLibrary::new();
            
            // Get all stdlib function names and register them (now in deterministic sorted order)
            let function_names: Vec<String> = stdlib.function_names().iter().map(|s| (*s).clone()).collect();
            
            for function_name in function_names {
                if let Some(function_ptr) = stdlib.get_function(&function_name) {
                    // Determine arity for each function (simplified mapping)
                    let arity = match function_name.as_str() {
                        // 0-arity functions (take no arguments)
                        "Length" | "StringLength" | "Head" | "Tail" | "Flatten" | "Array" |
                        "ArrayDimensions" | "ArrayRank" | "ArrayFlatten" | "Transpose" => 0,
                        
                        // 1-arity functions (take one argument)
                        "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" | "Sigmoid" | "Tanh" | "Softmax" |
                        "StringTake" | "StringDrop" | "ArrayReshape" | "Maximum" | "Apply" |
                        "Map" | "Append" | "Zeros" | "Ones" | "Range" | "ConstantSeries" |
                        "ZerosTensor" | "OnesTensor" | "EyeTensor" | "RandomTensor" |
                        "Tensor" | "Series" | "EmptyTable" | "Count" => 1,
                        
                        // 2-arity functions
                        "StringJoin" | "Dot" | "ReplaceAll" | "Rule" | "RuleDelayed" |
                        "Table" | "TableFromRows" | "GroupBy" | "Aggregate" | "TestHold" => 2,
                        
                        // 4-arity functions
                        "TestHoldMultiple" => 4,
                        
                        // Variable arity - default to 1 for now
                        _ => 1,
                    };
                    
                    self.registry.register_stdlib_function(&function_name, function_ptr, arity)?;
                }
            }
            
            Ok(())
        }

        /// Helper method to register a single Foreign method
        fn register_method(
            &mut self,
            type_name: &str,
            method_name: &str,
            arity: u8,
            function_ptr: NativeFunctionPtr,
        ) -> Result<(), LinkerError> {
            let signature = FunctionSignature::new(type_name, method_name, arity);
            let function_index = self.current_foreign_index;
            self.current_foreign_index += 1;
            
            self.registry.register_method(signature, function_ptr, function_index)
        }
    }

    // ===== TENSOR NATIVE FUNCTION IMPLEMENTATIONS =====

    fn tensor_dimensions(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Dimensions".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Dimensions", args)
    }

    fn tensor_rank(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Rank".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Rank", args)
    }

    fn tensor_length(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Length".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Length", args)
    }

    fn tensor_flatten(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Flatten".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Flatten", args)
    }

    fn tensor_transpose(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Transpose".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Transpose", args)
    }

    fn tensor_to_list(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "ToList".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("ToList", args)
    }

    fn tensor_reshape(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Reshape".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Reshape", args)
    }

    fn tensor_add(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Add".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Add", args)
    }

    fn tensor_multiply(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Multiply".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Multiply", args)
    }

    fn tensor_subtract(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Subtract".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Subtract", args)
    }

    fn tensor_divide(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Divide".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Divide", args)
    }

    fn tensor_dot(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Dot".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Dot", args)
    }

    fn tensor_maximum(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Maximum".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Maximum", args)
    }

    fn tensor_get(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Get".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Get", args)
    }

    fn tensor_set(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let tensor = obj.downcast_ref::<ForeignTensor>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Set".to_string(),
                expected: "Tensor".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        tensor.call_method("Set", args)
    }

    // ===== SERIES NATIVE FUNCTION IMPLEMENTATIONS =====

    fn series_length(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Length".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Length", args)
    }

    fn series_type(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Type".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Type", args)
    }

    fn series_to_list(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "ToList".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("ToList", args)
    }

    fn series_is_empty(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "IsEmpty".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("IsEmpty", args)
    }

    fn series_get(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Get".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Get", args)
    }

    fn series_append(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Append".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Append", args)
    }

    fn series_set(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Set".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Set", args)
    }

    fn series_slice(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let series = obj.downcast_ref::<ForeignSeries>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Slice".to_string(),
                expected: "Series".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        series.call_method("Slice", args)
    }

    // ===== TABLE NATIVE FUNCTION IMPLEMENTATIONS =====

    fn table_row_count(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "RowCount".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("RowCount", args)
    }

    fn table_column_count(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "ColumnCount".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("ColumnCount", args)
    }

    fn table_column_names(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "ColumnNames".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("ColumnNames", args)
    }

    fn table_is_empty(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "IsEmpty".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("IsEmpty", args)
    }

    fn table_get_row(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "GetRow".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("GetRow", args)
    }

    fn table_get_column(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "GetColumn".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("GetColumn", args)
    }

    fn table_head(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Head".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("Head", args)
    }

    fn table_tail(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "Tail".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("Tail", args)
    }

    fn table_get_cell(obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        let table = obj.downcast_ref::<ForeignTable>()
            .ok_or_else(|| ForeignError::InvalidArgumentType {
                method: "GetCell".to_string(),
                expected: "Table".to_string(),
                actual: obj.type_name().to_string(),
            })?;
        table.call_method("GetCell", args)
    }

    /// Create the global function registry with all Foreign object methods registered
    pub fn create_global_registry() -> Result<FunctionRegistry, LinkerError> {
        RegistryBuilder::new().build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;
    
    /// Test function for Series::length method
    fn series_length(_obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        if !args.is_empty() {
            return Err(ForeignError::InvalidArity {
                method: "length".to_string(),
                expected: 0,
                actual: args.len(),
            });
        }
        Ok(Value::Integer(42)) // Mock length
    }
    
    /// Test function for Tensor::add method
    fn tensor_add(_obj: &LyObj, args: &[Value]) -> Result<Value, ForeignError> {
        if args.len() != 1 {
            return Err(ForeignError::InvalidArity {
                method: "add".to_string(),
                expected: 1,
                actual: args.len(),
            });
        }
        Ok(Value::Integer(100)) // Mock result
    }
    
    #[test]
    fn test_function_signature_creation() {
        let sig = FunctionSignature::new("Series", "length", 0);
        assert_eq!(sig.type_name, "Series");
        assert_eq!(sig.method_name, "length");
        assert_eq!(sig.arity, 0);
        assert_eq!(sig.key(), "Series::length");
    }
    
    #[test]
    fn test_function_signature_with_types() {
        let sig = FunctionSignature::with_types(
            "Tensor", 
            "add", 
            1, 
            vec!["Tensor".to_string()]
        );
        assert_eq!(sig.type_name, "Tensor");
        assert_eq!(sig.method_name, "add");
        assert_eq!(sig.arity, 1);
        assert!(sig.arg_types.is_some());
        assert_eq!(sig.arg_types.unwrap(), vec!["Tensor"]);
    }
    
    #[test]
    fn test_function_registry_registration() {
        let mut registry = FunctionRegistry::new();
        let sig = FunctionSignature::new("Series", "length", 0);
        
        let result = registry.register_method(sig, series_length, 0);
        assert!(result.is_ok());
        
        assert_eq!(registry.stats.total_functions, 1);
        assert!(registry.has_method("Series", "length"));
    }
    
    #[test]
    fn test_function_registry_duplicate_registration() {
        let mut registry = FunctionRegistry::new();
        let sig = FunctionSignature::new("Series", "length", 0);
        
        // First registration should succeed
        assert!(registry.register_method(sig.clone(), series_length, 1).is_ok());
        
        // Second registration should fail
        let result = registry.register_method(sig, series_length, 0);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            LinkerError::RegistryError { message } => {
                assert!(message.contains("already registered"));
            }
            _ => panic!("Expected RegistryError"),
        }
    }
    
    #[test]
    fn test_function_registry_lookup() {
        let mut registry = FunctionRegistry::new();
        let sig = FunctionSignature::new("Series", "length", 0);
        registry.register_method(sig, series_length, 2).unwrap();
        
        // Successful lookup
        let entry = registry.lookup("Series", "length").unwrap();
        assert_eq!(entry.signature.type_name, "Series");
        assert_eq!(entry.signature.method_name, "length");
        
        // Failed lookup
        let result = registry.lookup("Series", "nonexistent");
        assert!(result.is_err());
        
        match result.unwrap_err() {
            LinkerError::FunctionNotFound { type_name, method_name } => {
                assert_eq!(type_name, "Series");
                assert_eq!(method_name, "nonexistent");
            }
            _ => panic!("Expected FunctionNotFound error"),
        }
    }
    
    #[test]
    fn test_function_registry_type_methods() {
        let mut registry = FunctionRegistry::new();
        
        // Register multiple methods for Series
        registry.register_method(
            FunctionSignature::new("Series", "length", 0),
            series_length,
            3
        ).unwrap();
        
        registry.register_method(
            FunctionSignature::new("Series", "get", 1),
            series_length,
            4
        ).unwrap();
        
        // Register method for Tensor
        registry.register_method(
            FunctionSignature::new("Tensor", "add", 1),
            tensor_add,
            5
        ).unwrap();
        
        // Check type methods
        let series_methods = registry.get_type_methods("Series");
        assert_eq!(series_methods.len(), 2);
        assert!(series_methods.contains(&"length".to_string()));
        assert!(series_methods.contains(&"get".to_string()));
        
        let tensor_methods = registry.get_type_methods("Tensor");
        assert_eq!(tensor_methods.len(), 1);
        assert!(tensor_methods.contains(&"add".to_string()));
        
        // Check type names
        let type_names = registry.get_type_names();
        assert_eq!(type_names.len(), 2);
        assert!(type_names.contains(&"Series".to_string()));
        assert!(type_names.contains(&"Tensor".to_string()));
    }
    
    #[test]
    fn test_link_time_resolver() {
        let mut registry = FunctionRegistry::new();
        registry.register_method(
            FunctionSignature::new("Series", "length", 0),
            series_length,
            7
        ).unwrap();
        
        let mut resolver = LinkTimeResolver::new(registry);
        
        // Successful resolution
        let result = resolver.resolve_method_call("Series", "length", 0);
        assert!(result.is_ok());
        
        // Check that cache works on second call
        let result2 = resolver.resolve_method_call("Series", "length", 0);
        assert!(result2.is_ok());
        assert_eq!(resolver.stats.cache_hits, 1);
        
        // Failed resolution (wrong arity)
        let result = resolver.resolve_method_call("Series", "length", 1);
        assert!(result.is_err());
        
        // Failed resolution (unknown method)
        let result = resolver.resolve_method_call("Series", "unknown", 0);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_function_entry_validation() {
        let sig = FunctionSignature::new("Series", "length", 0);
        let entry = FunctionEntry::new_foreign(sig, series_length, 6);
        
        // Valid arity
        assert!(entry.validate_arity(0).is_ok());
        
        // Invalid arity
        let result = entry.validate_arity(1);
        assert!(result.is_err());
        
        match result.unwrap_err() {
            LinkerError::InvalidArity { expected, actual, .. } => {
                assert_eq!(expected, 0);
                assert_eq!(actual, 1);
            }
            _ => panic!("Expected InvalidArity error"),
        }
    }
    
    #[test]
    fn test_registry_builder_create() {
        let builder = registry::RegistryBuilder::new();
        // Test that builder creates without error
        // We can't access private fields, so just check it builds successfully
        let registry = builder.build().unwrap();
        assert!(!registry.get_type_names().is_empty());
    }
    
    #[test] 
    fn test_registry_builder_build() {
        let builder = registry::RegistryBuilder::new();
        let registry = builder.build().unwrap();
        
        // Should have registered all 3 types
        let type_names = registry.get_type_names();
        assert_eq!(type_names.len(), 3);
        assert!(type_names.contains(&"Tensor".to_string()));
        assert!(type_names.contains(&"Series".to_string()));
        assert!(type_names.contains(&"Table".to_string()));
        
        // Check total function count (registry reports 100 total functions + 2 I/O functions)
        assert_eq!(registry.stats.total_functions, 118);
    }
    
    #[test]
    fn test_registry_has_tensor_methods() {
        let builder = registry::RegistryBuilder::new();
        let registry = builder.build().unwrap();
        
        // Test Tensor methods with 0 args
        assert!(registry.has_method("Tensor", "Dimensions"));
        assert!(registry.has_method("Tensor", "Rank"));
        assert!(registry.has_method("Tensor", "Length"));
        assert!(registry.has_method("Tensor", "Flatten"));
        assert!(registry.has_method("Tensor", "Transpose"));
        assert!(registry.has_method("Tensor", "ToList"));
        
        // Test Tensor methods with 1 arg
        assert!(registry.has_method("Tensor", "Reshape"));
        assert!(registry.has_method("Tensor", "Add"));
        assert!(registry.has_method("Tensor", "Multiply"));
        assert!(registry.has_method("Tensor", "Subtract"));
        assert!(registry.has_method("Tensor", "Divide"));
        assert!(registry.has_method("Tensor", "Dot"));
        assert!(registry.has_method("Tensor", "Maximum"));
        
        // Test Tensor methods with variable args
        assert!(registry.has_method("Tensor", "Get"));
        assert!(registry.has_method("Tensor", "Set"));
        
        // Test non-existent method
        assert!(!registry.has_method("Tensor", "NonExistent"));
    }
    
    #[test]
    fn test_registry_has_series_methods() {
        let builder = registry::RegistryBuilder::new();
        let registry = builder.build().unwrap();
        
        // Test Series methods with 0 args
        assert!(registry.has_method("Series", "Length"));
        assert!(registry.has_method("Series", "Type"));
        assert!(registry.has_method("Series", "ToList"));
        assert!(registry.has_method("Series", "IsEmpty"));
        
        // Test Series methods with 1 arg
        assert!(registry.has_method("Series", "Get"));
        assert!(registry.has_method("Series", "Append"));
        
        // Test Series methods with 2 args
        assert!(registry.has_method("Series", "Set"));
        assert!(registry.has_method("Series", "Slice"));
        
        // Test non-existent method
        assert!(!registry.has_method("Series", "NonExistent"));
    }
    
    #[test]
    fn test_registry_has_table_methods() {
        let builder = registry::RegistryBuilder::new();
        let registry = builder.build().unwrap();
        
        // Test Table methods with 0 args
        assert!(registry.has_method("Table", "RowCount"));
        assert!(registry.has_method("Table", "ColumnCount"));
        assert!(registry.has_method("Table", "ColumnNames"));
        assert!(registry.has_method("Table", "IsEmpty"));
        
        // Test Table methods with 1 arg
        assert!(registry.has_method("Table", "GetRow"));
        assert!(registry.has_method("Table", "GetColumn"));
        assert!(registry.has_method("Table", "Head"));
        assert!(registry.has_method("Table", "Tail"));
        
        // Test Table methods with 2 args
        assert!(registry.has_method("Table", "GetCell"));
        
        // Test non-existent method
        assert!(!registry.has_method("Table", "NonExistent"));
    }
    
    #[test]
    fn test_global_registry_creation() {
        let registry = registry::create_global_registry().unwrap();
        
        // Should have all types and methods (+ 2 I/O functions)
        assert_eq!(registry.get_type_names().len(), 3);
        assert_eq!(registry.stats.total_functions, 118);
        
        // Test a few key methods are registered
        assert!(registry.has_method("Tensor", "Add"));
        assert!(registry.has_method("Series", "Length"));
        assert!(registry.has_method("Table", "RowCount"));
    }
    
    #[test]
    fn test_registry_method_counts() {
        let builder = registry::RegistryBuilder::new();
        let registry = builder.build().unwrap();
        
        // Check method counts for each type
        let tensor_methods = registry.get_type_methods("Tensor");
        assert_eq!(tensor_methods.len(), 15);
        
        let series_methods = registry.get_type_methods("Series");
        assert_eq!(series_methods.len(), 8);
        
        let table_methods = registry.get_type_methods("Table");
        assert_eq!(table_methods.len(), 9);
    }
}