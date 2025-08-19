//! Type metadata system for storing and managing function type signatures
//!
//! This module provides infrastructure for:
//! - Storing function type signatures from TypedFunction expressions
//! - Looking up type information for validation
//! - Managing type aliases and constraints
//! - Source location tracking for error reporting

use super::{LyraType, TypeConstraint};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};

/// Source location information for error reporting
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File name or identifier
    pub file: String,
    /// Line number (1-based)
    pub line: usize,
    /// Column number (1-based)
    pub column: usize,
}

impl Default for SourceLocation {
    fn default() -> Self {
        SourceLocation {
            file: "<unknown>".to_string(),
            line: 1,
            column: 1,
        }
    }
}

/// Function type signature containing parameter and return type information
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionTypeSignature {
    /// Function name
    pub name: String,
    /// Parameter types (None for untyped parameters)
    pub params: Vec<(String, Option<LyraType>)>,
    /// Return type (None for untyped)
    pub return_type: Option<LyraType>,
    /// Type constraints on parameters
    pub constraints: Vec<TypeConstraint>,
    /// Source location for error reporting
    pub location: SourceLocation,
}

impl FunctionTypeSignature {
    /// Create a new function type signature
    pub fn new(name: String) -> Self {
        FunctionTypeSignature {
            name,
            params: Vec::new(),
            return_type: None,
            constraints: Vec::new(),
            location: SourceLocation::default(),
        }
    }

    /// Add a typed parameter
    pub fn add_typed_param(&mut self, name: String, param_type: LyraType) {
        self.params.push((name, Some(param_type)));
    }

    /// Add an untyped parameter
    pub fn add_untyped_param(&mut self, name: String) {
        self.params.push((name, None));
    }

    /// Set the return type
    pub fn with_return_type(mut self, return_type: LyraType) -> Self {
        self.return_type = Some(return_type);
        self
    }

    /// Add a type constraint
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.constraints.push(constraint);
    }

    /// Set the source location
    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = location;
        self
    }

    /// Get the number of parameters
    pub fn param_count(&self) -> usize {
        self.params.len()
    }

    /// Check if all parameters are typed
    pub fn is_fully_typed(&self) -> bool {
        self.params.iter().all(|(_, ty)| ty.is_some()) && self.return_type.is_some()
    }

    /// Get typed parameter names and types
    pub fn typed_params(&self) -> Vec<(&str, &LyraType)> {
        self.params
            .iter()
            .filter_map(|(name, ty)| ty.as_ref().map(|t| (name.as_str(), t)))
            .collect()
    }

    /// Get untyped parameter names
    pub fn untyped_params(&self) -> Vec<&str> {
        self.params
            .iter()
            .filter_map(|(name, ty)| if ty.is_none() { Some(name.as_str()) } else { None })
            .collect()
    }
}

/// Type alias definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TypeAlias {
    /// Alias name
    pub name: String,
    /// Target type
    pub target: LyraType,
    /// Source location
    pub location: SourceLocation,
}

/// Registry for storing and managing type metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeRegistry {
    /// Function type signatures
    functions: HashMap<String, FunctionTypeSignature>,
    /// Type aliases (e.g., Matrix = List[List[Real]])
    aliases: HashMap<String, TypeAlias>,
    /// Global type constraints
    constraints: Vec<TypeConstraint>,
}

impl TypeRegistry {
    /// Create a new empty type registry
    pub fn new() -> Self {
        TypeRegistry {
            functions: HashMap::new(),
            aliases: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Register a function type signature
    pub fn register_function(&mut self, signature: FunctionTypeSignature) {
        self.functions.insert(signature.name.clone(), signature);
    }

    /// Get a function type signature by name
    pub fn get_function(&self, name: &str) -> Option<&FunctionTypeSignature> {
        self.functions.get(name)
    }

    /// Check if a function is registered
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }

    /// Get all registered function names
    pub fn function_names(&self) -> Vec<&str> {
        self.functions.keys().map(|s| s.as_str()).collect()
    }

    /// Register a type alias
    pub fn register_alias(&mut self, alias: TypeAlias) {
        self.aliases.insert(alias.name.clone(), alias);
    }

    /// Get a type alias by name
    pub fn get_alias(&self, name: &str) -> Option<&TypeAlias> {
        self.aliases.get(name)
    }

    /// Resolve a type, following aliases
    pub fn resolve_type(&self, ty: &LyraType) -> LyraType {
        match ty {
            LyraType::Custom(name) => {
                if let Some(alias) = self.get_alias(name) {
                    // Recursively resolve in case of nested aliases
                    self.resolve_type(&alias.target)
                } else {
                    ty.clone()
                }
            }
            LyraType::List(elem_type) => {
                LyraType::List(Box::new(self.resolve_type(elem_type)))
            }
            LyraType::Tensor { element_type, shape } => {
                LyraType::Tensor {
                    element_type: Box::new(self.resolve_type(element_type)),
                    shape: shape.clone(),
                }
            }
            LyraType::Union(types) => {
                LyraType::Union(types.iter().map(|t| self.resolve_type(t)).collect())
            }
            LyraType::Tuple(types) => {
                LyraType::Tuple(types.iter().map(|t| self.resolve_type(t)).collect())
            }
            _ => ty.clone(),
        }
    }

    /// Add a global type constraint
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.constraints.push(constraint);
    }

    /// Get all global constraints
    pub fn constraints(&self) -> &[TypeConstraint] {
        &self.constraints
    }

    /// Clear all registered types (for testing)
    pub fn clear(&mut self) {
        self.functions.clear();
        self.aliases.clear();
        self.constraints.clear();
    }

    /// Get statistics about registered types
    pub fn stats(&self) -> TypeRegistryStats {
        let typed_functions = self.functions.values()
            .filter(|sig| sig.is_fully_typed())
            .count();
        
        let untyped_functions = self.functions.len() - typed_functions;

        TypeRegistryStats {
            total_functions: self.functions.len(),
            typed_functions,
            untyped_functions,
            type_aliases: self.aliases.len(),
            global_constraints: self.constraints.len(),
        }
    }
}

/// Statistics about the type registry
#[derive(Debug, Clone, PartialEq)]
pub struct TypeRegistryStats {
    pub total_functions: usize,
    pub typed_functions: usize,
    pub untyped_functions: usize,
    pub type_aliases: usize,
    pub global_constraints: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_signature_creation() {
        let mut sig = FunctionTypeSignature::new("add".to_string());
        sig.add_typed_param("x".to_string(), LyraType::Integer);
        sig.add_typed_param("y".to_string(), LyraType::Integer);
        sig = sig.with_return_type(LyraType::Integer);

        assert_eq!(sig.name, "add");
        assert_eq!(sig.param_count(), 2);
        assert!(sig.is_fully_typed());
        assert_eq!(sig.typed_params().len(), 2);
        assert_eq!(sig.untyped_params().len(), 0);
    }

    #[test]
    fn test_mixed_typed_untyped_params() {
        let mut sig = FunctionTypeSignature::new("func".to_string());
        sig.add_typed_param("x".to_string(), LyraType::Integer);
        sig.add_untyped_param("y".to_string());
        sig.add_typed_param("z".to_string(), LyraType::Real);

        assert_eq!(sig.param_count(), 3);
        assert!(!sig.is_fully_typed());
        assert_eq!(sig.typed_params().len(), 2);
        assert_eq!(sig.untyped_params().len(), 1);
        assert_eq!(sig.untyped_params()[0], "y");
    }

    #[test]
    fn test_type_registry_functions() {
        let mut registry = TypeRegistry::new();

        let mut sig = FunctionTypeSignature::new("add".to_string());
        sig.add_typed_param("x".to_string(), LyraType::Integer);
        sig.add_typed_param("y".to_string(), LyraType::Integer);
        sig = sig.with_return_type(LyraType::Integer);

        registry.register_function(sig);

        assert!(registry.has_function("add"));
        assert!(!registry.has_function("multiply"));

        let retrieved = registry.get_function("add").unwrap();
        assert_eq!(retrieved.name, "add");
        assert_eq!(retrieved.param_count(), 2);
    }

    #[test]
    fn test_type_registry_aliases() {
        let mut registry = TypeRegistry::new();

        let alias = TypeAlias {
            name: "Matrix".to_string(),
            target: LyraType::List(Box::new(LyraType::List(Box::new(LyraType::Real)))),
            location: SourceLocation::default(),
        };

        registry.register_alias(alias);

        assert!(registry.get_alias("Matrix").is_some());
        assert!(registry.get_alias("Vector").is_none());
    }

    #[test]
    fn test_type_resolution() {
        let mut registry = TypeRegistry::new();

        // Register Matrix = List[List[Real]]
        let matrix_alias = TypeAlias {
            name: "Matrix".to_string(),
            target: LyraType::List(Box::new(LyraType::List(Box::new(LyraType::Real)))),
            location: SourceLocation::default(),
        };
        registry.register_alias(matrix_alias);

        // Test resolution
        let matrix_type = LyraType::Custom("Matrix".to_string());
        let resolved = registry.resolve_type(&matrix_type);

        match resolved {
            LyraType::List(elem_type) => {
                match elem_type.as_ref() {
                    LyraType::List(inner_elem) => {
                        assert_eq!(**inner_elem, LyraType::Real);
                    }
                    _ => panic!("Expected List[Real], got {:?}", elem_type),
                }
            }
            _ => panic!("Expected List type, got {:?}", resolved),
        }
    }

    #[test]
    fn test_type_registry_stats() {
        let mut registry = TypeRegistry::new();

        // Add typed function
        let mut typed_sig = FunctionTypeSignature::new("add".to_string());
        typed_sig.add_typed_param("x".to_string(), LyraType::Integer);
        typed_sig = typed_sig.with_return_type(LyraType::Integer);
        registry.register_function(typed_sig);

        // Add untyped function
        let mut untyped_sig = FunctionTypeSignature::new("print".to_string());
        untyped_sig.add_untyped_param("value".to_string());
        registry.register_function(untyped_sig);

        // Add alias
        let alias = TypeAlias {
            name: "Vector".to_string(),
            target: LyraType::List(Box::new(LyraType::Real)),
            location: SourceLocation::default(),
        };
        registry.register_alias(alias);

        let stats = registry.stats();
        assert_eq!(stats.total_functions, 2);
        assert_eq!(stats.typed_functions, 1);
        assert_eq!(stats.untyped_functions, 1);
        assert_eq!(stats.type_aliases, 1);
    }

    #[test]
    fn test_source_location() {
        let location = SourceLocation {
            file: "test.lyra".to_string(),
            line: 10,
            column: 5,
        };

        let sig = FunctionTypeSignature::new("test".to_string())
            .with_location(location.clone());

        assert_eq!(sig.location.file, "test.lyra");
        assert_eq!(sig.location.line, 10);
        assert_eq!(sig.location.column, 5);
    }
}