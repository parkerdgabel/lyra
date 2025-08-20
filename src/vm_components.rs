// VM Components Module
// This module contains extracted components from the VM for better organization

use crate::vm::{Value, VmError, VmResult};
use crate::ast::Expr;
use crate::compiler::{SimpleFunctionSignature, EnhancedFunctionSignature};
use std::collections::HashMap;
use std::sync::Arc;
use dashmap::DashMap;

/// Symbol manager responsible for handling symbol table operations
/// and symbol-to-value bindings in the VM.
/// 
/// This component is extracted from the VM to provide:
/// - Thread-safe symbol resolution
/// - Separation of concerns between symbol management and VM execution
/// - Better testing and maintainability
/// - Clear interfaces for symbol operations
#[derive(Debug)]
pub struct SymbolManager {
    /// Symbol table - maps symbol names to indices for efficient lookup
    pub symbol_table: HashMap<String, usize>,
    
    /// Global symbol values for immediate assignments (=)
    /// Thread-safe for concurrent access
    pub global_symbols: Arc<DashMap<String, Value>>,
    
    /// Delayed symbol definitions for delayed assignments (:=)
    /// Thread-safe for concurrent access
    pub delayed_definitions: Arc<DashMap<String, Expr>>,
    
    /// User-defined function bodies
    /// Thread-safe for concurrent access
    pub user_functions: Arc<DashMap<String, Expr>>,
}

impl SymbolManager {
    /// Create a new SymbolManager
    pub fn new() -> Self {
        SymbolManager {
            symbol_table: HashMap::new(),
            global_symbols: Arc::new(DashMap::new()),
            delayed_definitions: Arc::new(DashMap::new()),
            user_functions: Arc::new(DashMap::new()),
        }
    }
    
    /// Add a symbol to the symbol table, returns its index
    pub fn add_symbol(&mut self, name: String) -> usize {
        if let Some(&index) = self.symbol_table.get(&name) {
            return index;
        }
        let index = self.symbol_table.len();
        self.symbol_table.insert(name, index);
        index
    }
    
    /// Get symbol index by name
    pub fn get_symbol_index(&self, name: &str) -> Option<usize> {
        self.symbol_table.get(name).copied()
    }
    
    /// Set immediate symbol value (= assignment)
    pub fn set_global_symbol(&self, name: String, value: Value) {
        self.global_symbols.insert(name, value);
    }
    
    /// Get immediate symbol value
    pub fn get_global_symbol(&self, name: &str) -> Option<Value> {
        self.global_symbols.get(name).map(|entry| entry.value().clone())
    }
    
    /// Set delayed symbol definition (:= assignment)
    pub fn set_delayed_definition(&self, name: String, expr: Expr) {
        self.delayed_definitions.insert(name, expr);
    }
    
    /// Get delayed symbol definition
    pub fn get_delayed_definition(&self, name: &str) -> Option<Expr> {
        self.delayed_definitions.get(name).map(|entry| entry.value().clone())
    }
    
    /// Register user function
    pub fn register_user_function(&self, name: String, body: Expr) {
        self.user_functions.insert(name, body);
    }
    
    /// Get user function body
    pub fn get_user_function(&self, name: &str) -> Option<Expr> {
        self.user_functions.get(name).map(|entry| entry.value().clone())
    }
    
    /// Check if a symbol exists in any symbol namespace
    pub fn symbol_exists(&self, name: &str) -> bool {
        self.symbol_table.contains_key(name) ||
        self.global_symbols.contains_key(name) ||
        self.delayed_definitions.contains_key(name) ||
        self.user_functions.contains_key(name)
    }
    
    /// Clear all symbols (useful for testing)
    pub fn clear(&mut self) {
        self.symbol_table.clear();
        self.global_symbols.clear();
        self.delayed_definitions.clear();
        self.user_functions.clear();
    }
}

impl Default for SymbolManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Type registry responsible for handling type metadata storage,
/// type checking, and type inference operations in the VM.
/// 
/// This component is extracted from the VM to provide:
/// - Efficient type lookup and validation
/// - Type caching for performance
/// - Separation of concerns between type management and VM execution
/// - Better testing and maintainability
/// - Thread-safe type operations
#[derive(Debug)]
pub struct TypeRegistry {
    /// Simple function type metadata (legacy support)
    pub type_metadata: HashMap<String, SimpleFunctionSignature>,
    
    /// Enhanced function type metadata with detailed type information
    /// Thread-safe for concurrent access
    pub enhanced_metadata: Arc<DashMap<String, EnhancedFunctionSignature>>,
    
    /// Type cache for frequently accessed type information
    /// Thread-safe for concurrent access
    pub type_cache: Arc<DashMap<String, TypeInfo>>,
}

/// Cached type information for efficient lookup
#[derive(Debug, Clone)]
pub struct TypeInfo {
    pub type_name: String,
    pub is_primitive: bool,
    pub is_listable: bool,
    pub element_type: Option<String>, // For List[T], Option[T], etc.
}

impl TypeRegistry {
    /// Create a new TypeRegistry
    pub fn new() -> Self {
        TypeRegistry {
            type_metadata: HashMap::new(),
            enhanced_metadata: Arc::new(DashMap::new()),
            type_cache: Arc::new(DashMap::new()),
        }
    }
    
    /// Register a simple function signature (legacy)
    pub fn register_type_signature(&mut self, signature: SimpleFunctionSignature) {
        self.type_metadata.insert(signature.name.clone(), signature);
    }
    
    /// Get a simple function signature by name (legacy)
    pub fn get_type_signature(&self, name: &str) -> Option<&SimpleFunctionSignature> {
        self.type_metadata.get(name)
    }
    
    /// Check if a function has simple type metadata (legacy)
    pub fn has_type_metadata(&self, name: &str) -> bool {
        self.type_metadata.contains_key(name)
    }
    
    /// Register an enhanced function signature
    pub fn register_enhanced_signature(&self, signature: EnhancedFunctionSignature) {
        self.enhanced_metadata.insert(signature.name.clone(), signature);
    }
    
    /// Get an enhanced function signature by name
    pub fn get_enhanced_signature(&self, name: &str) -> Option<EnhancedFunctionSignature> {
        self.enhanced_metadata.get(name).map(|entry| entry.value().clone())
    }
    
    /// Check if a function has enhanced type metadata
    pub fn has_enhanced_metadata(&self, name: &str) -> bool {
        self.enhanced_metadata.contains_key(name)
    }
    
    /// Get the type name of a value for type checking
    pub fn get_value_type_name(&self, value: &Value) -> String {
        match value {
            Value::Integer(_) => "Integer".to_string(),
            Value::Real(_) => "Real".to_string(),
            Value::String(_) => "String".to_string(),
            Value::Symbol(_) => "Symbol".to_string(),
            Value::List(_) => "List".to_string(),
            Value::Function(_) => "Function".to_string(),
            Value::Boolean(_) => "Boolean".to_string(),
            Value::Missing => "Missing".to_string(),
            Value::LyObj(_) => "Object".to_string(),
            Value::Quote(_) => "Quote".to_string(),
            Value::Pattern(_) => "Pattern".to_string(),
        }
    }
    
    /// Check if a value is compatible with an expected type (including coercion)
    pub fn is_type_compatible(&self, value: &Value, expected_type: &str) -> VmResult<bool> {
        // Handle complex type expressions
        if expected_type.contains('[') && expected_type.contains(']') {
            return self.validate_complex_type(value, expected_type);
        }
        
        let actual_type = self.get_value_type_name(value);
        
        // Exact type match
        if actual_type == expected_type {
            return Ok(true);
        }
        
        // Type coercion rules
        match (actual_type.as_str(), expected_type) {
            // Integer can be coerced to Real
            ("Integer", "Real") => Ok(true),
            // Add more coercion rules as needed
            _ => Ok(false),
        }
    }
    
    /// Validate complex type expressions like List[T], Map[K,V], etc.
    pub fn validate_complex_type(&self, value: &Value, expected_type: &str) -> VmResult<bool> {
        // Parse the complex type expression
        if let Some((base_type, type_params)) = self.parse_complex_type(expected_type) {
            match (base_type.as_str(), value) {
                ("List", Value::List(list)) => {
                    // Validate each element in the list matches the parameter type
                    if type_params.len() != 1 {
                        return Err(VmError::Runtime(format!("List type requires exactly 1 type parameter, got {}", type_params.len())));
                    }
                    let element_type = &type_params[0];
                    
                    // Check each element in the list
                    for (index, element) in list.iter().enumerate() {
                        if !self.is_type_compatible(element, element_type)? {
                            let actual_element_type = self.get_value_type_name(element);
                            return Err(VmError::Runtime(format!(
                                "Type error: List[{}] element at index {} expected {}, but found {}",
                                element_type, index, element_type, actual_element_type
                            )));
                        }
                    }
                    Ok(true)
                }
                _ => {
                    // Base type mismatch
                    Ok(false)
                }
            }
        } else {
            // Failed to parse complex type
            Err(VmError::Runtime(format!("Invalid complex type expression: {}", expected_type)))
        }
    }
    
    /// Parse complex type expressions like "List[Real]" into ("List", ["Real"])
    pub fn parse_complex_type(&self, type_expr: &str) -> Option<(String, Vec<String>)> {
        if let Some(open_bracket) = type_expr.find('[') {
            if let Some(close_bracket) = type_expr.rfind(']') {
                let base_type = type_expr[..open_bracket].to_string();
                let params_str = &type_expr[open_bracket + 1..close_bracket];
                
                // Simple parsing - split by comma and trim
                let type_params: Vec<String> = params_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                
                return Some((base_type, type_params));
            }
        }
        None
    }
    
    /// Load type metadata from compiler context
    pub fn load_from_compiler(&mut self, 
        type_metadata: HashMap<String, SimpleFunctionSignature>,
        enhanced_metadata: HashMap<String, EnhancedFunctionSignature>
    ) {
        self.type_metadata = type_metadata;
        
        // Transfer enhanced metadata to thread-safe storage
        for (name, signature) in enhanced_metadata {
            self.enhanced_metadata.insert(name, signature);
        }
    }
}

impl Default for TypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Number};

    #[test]
    fn test_symbol_manager_basic_operations() {
        let mut manager = SymbolManager::new();
        
        // Add symbols
        let index1 = manager.add_symbol("x".to_string());
        let index2 = manager.add_symbol("y".to_string());
        let index3 = manager.add_symbol("x".to_string()); // Duplicate
        
        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(index3, 0); // Should return existing index
        
        // Check symbol existence
        assert_eq!(manager.get_symbol_index("x"), Some(0));
        assert_eq!(manager.get_symbol_index("y"), Some(1));
        assert_eq!(manager.get_symbol_index("z"), None);
        
        // Test global symbols
        manager.set_global_symbol("x".to_string(), Value::Integer(42));
        assert_eq!(manager.get_global_symbol("x"), Some(Value::Integer(42)));
        
        // Test symbol existence check
        assert!(manager.symbol_exists("x"));
        assert!(manager.symbol_exists("y"));
        assert!(!manager.symbol_exists("z"));
    }
    
    #[test]
    fn test_type_registry_basic_operations() {
        let mut registry = TypeRegistry::new();
        
        // Test simple signatures
        let signature = SimpleFunctionSignature {
            name: "testFunc".to_string(),
            param_count: 2,
            is_typed: true,
        };
        
        registry.register_type_signature(signature.clone());
        assert!(registry.has_type_metadata("testFunc"));
        assert_eq!(registry.get_type_signature("testFunc"), Some(&signature));
        
        // Test enhanced signatures
        let mut enhanced_sig = EnhancedFunctionSignature::new("enhancedFunc".to_string());
        enhanced_sig.add_param("x".to_string(), Some("Integer".to_string()));
        enhanced_sig.set_return_type("Real".to_string());
        
        registry.register_enhanced_signature(enhanced_sig.clone());
        assert!(registry.has_enhanced_metadata("enhancedFunc"));
        assert_eq!(registry.get_enhanced_signature("enhancedFunc"), Some(enhanced_sig));
    }
    
    #[test]
    fn test_type_checking() {
        let registry = TypeRegistry::new();
        
        // Basic type compatibility
        assert!(registry.is_type_compatible(&Value::Integer(42), "Integer").unwrap());
        assert!(registry.is_type_compatible(&Value::Real(3.14), "Real").unwrap());
        assert!(registry.is_type_compatible(&Value::Integer(42), "Real").unwrap()); // Coercion
        assert!(!registry.is_type_compatible(&Value::String("test".to_string()), "Integer").unwrap());
        
        // Complex type validation
        let list_of_integers = Value::List(vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)]);
        assert!(registry.is_type_compatible(&list_of_integers, "List[Integer]").unwrap());
        
        let mixed_list = Value::List(vec![Value::Integer(1), Value::String("test".to_string())]);
        assert!(!registry.is_type_compatible(&mixed_list, "List[Integer]").unwrap());
    }
}