//! Namespace Resolution and Import System
//!
//! Handles qualified name parsing, import statement resolution, and scope management.

use super::{ModuleError, Module};
use super::registry::ModuleRegistry;
use std::collections::HashMap;

/// Import statement types
#[derive(Debug, Clone)]
pub enum ImportStatement {
    /// Import entire module: `import std::math`
    Module {
        namespace: String,
        alias: Option<String>,
    },
    
    /// Import specific functions: `import std::math::{Sin, Cos}`
    Specific {
        namespace: String,
        functions: Vec<String>,
        aliases: HashMap<String, String>, // function -> alias
    },
    
    /// Wildcard import: `import std::math::*`
    Wildcard {
        namespace: String,
    },
    
    /// Aliased import: `import std::math::Sin as Sine`
    Aliased {
        namespace: String,
        function: String,
        alias: String,
    },
}

/// Import resolution context for a compilation unit
#[derive(Debug, Clone)]
pub struct ImportContext {
    /// Direct function imports (function_name -> qualified_name)
    function_imports: HashMap<String, String>,
    
    /// Module imports (alias -> namespace)
    module_imports: HashMap<String, String>,
    
    /// Wildcard imports (namespace -> exported_functions)
    wildcard_imports: HashMap<String, Vec<String>>,
    
    /// Current namespace (for relative imports)
    current_namespace: Option<String>,
}

impl ImportContext {
    /// Create a new empty import context
    pub fn new() -> Self {
        ImportContext {
            function_imports: HashMap::new(),
            module_imports: HashMap::new(),
            wildcard_imports: HashMap::new(),
            current_namespace: None,
        }
    }
    
    /// Set the current namespace
    pub fn set_namespace(&mut self, namespace: String) {
        self.current_namespace = Some(namespace);
    }
    
    /// Add an import statement to the context
    pub fn add_import(&mut self, import: ImportStatement, registry: &ModuleRegistry) -> Result<(), ModuleError> {
        match import {
            ImportStatement::Module { namespace, alias } => {
                self.add_module_import(namespace, alias)?;
            },
            ImportStatement::Specific { namespace, functions, aliases } => {
                self.add_specific_imports(namespace, functions, aliases, registry)?;
            },
            ImportStatement::Wildcard { namespace } => {
                self.add_wildcard_import(namespace, registry)?;
            },
            ImportStatement::Aliased { namespace, function, alias } => {
                self.add_aliased_import(namespace, function, alias)?;
            },
        }
        Ok(())
    }
    
    /// Resolve a function name to its qualified name
    pub fn resolve_function(&self, function_name: &str) -> Option<String> {
        // Check direct function imports first
        if let Some(qualified) = self.function_imports.get(function_name) {
            return Some(qualified.clone());
        }
        
        // Check wildcard imports
        for (namespace, functions) in &self.wildcard_imports {
            if functions.contains(&function_name.to_string()) {
                return Some(format!("{}::{}", namespace, function_name));
            }
        }
        
        // Check if it's a module-qualified call (e.g., math::Sin)
        if let Some(module_name) = function_name.split("::").next() {
            if let Some(namespace) = self.module_imports.get(module_name) {
                let remaining = function_name.strip_prefix(&format!("{}::", module_name))?;
                return Some(format!("{}::{}", namespace, remaining));
            }
        }
        
        None
    }
    
    /// Get all imported function names
    pub fn imported_functions(&self) -> Vec<String> {
        let mut functions = Vec::new();
        
        // Add direct imports
        functions.extend(self.function_imports.keys().cloned());
        
        // Add wildcard imports
        for imports in self.wildcard_imports.values() {
            functions.extend(imports.clone());
        }
        
        functions.sort();
        functions.dedup();
        functions
    }
    
    /// Get all imported modules
    pub fn imported_modules(&self) -> Vec<(String, String)> {
        self.module_imports.iter()
            .map(|(alias, namespace)| (alias.clone(), namespace.clone()))
            .collect()
    }
    
    /// Check if a function is imported
    pub fn has_function(&self, function_name: &str) -> bool {
        self.resolve_function(function_name).is_some()
    }
    
    /// Clear all imports (for testing)
    pub fn clear(&mut self) {
        self.function_imports.clear();
        self.module_imports.clear();
        self.wildcard_imports.clear();
        self.current_namespace = None;
    }
    
    // Private helper methods
    
    fn add_module_import(&mut self, namespace: String, alias: Option<String>) -> Result<(), ModuleError> {
        let module_alias = alias.unwrap_or_else(|| {
            // Use the last part of the namespace as the default alias
            namespace.split("::").last().unwrap_or(&namespace).to_string()
        });
        
        if self.module_imports.contains_key(&module_alias) {
            return Err(ModuleError::PackageError {
                message: format!("Module alias '{}' already in use", module_alias),
            });
        }
        
        self.module_imports.insert(module_alias, namespace);
        Ok(())
    }
    
    fn add_specific_imports(
        &mut self,
        namespace: String,
        functions: Vec<String>,
        aliases: HashMap<String, String>,
        registry: &ModuleRegistry,
    ) -> Result<(), ModuleError> {
        // Verify the module exists
        if !registry.has_module(&namespace) {
            return Err(ModuleError::ModuleNotFound { name: namespace });
        }
        
        // Get module exports to validate function names
        let exports = registry.get_module_exports(&namespace);
        
        for function in functions {
            // Validate function exists in module
            if !exports.contains(&function) {
                return Err(ModuleError::PackageError {
                    message: format!("Function '{}' not found in module '{}'", function, namespace),
                });
            }
            
            let import_name = aliases.get(&function).unwrap_or(&function);
            let qualified_name = format!("{}::{}", namespace, function);
            
            // Check for conflicts
            if self.function_imports.contains_key(import_name) {
                return Err(ModuleError::PackageError {
                    message: format!("Function name '{}' already imported", import_name),
                });
            }
            
            self.function_imports.insert(import_name.clone(), qualified_name);
        }
        
        Ok(())
    }
    
    fn add_wildcard_import(&mut self, namespace: String, registry: &ModuleRegistry) -> Result<(), ModuleError> {
        // Verify the module exists
        if !registry.has_module(&namespace) {
            return Err(ModuleError::ModuleNotFound { name: namespace.clone() });
        }
        
        // Get all exported functions from the module
        let exports = registry.get_module_exports(&namespace);
        
        // Check for conflicts with existing imports
        for export in &exports {
            if self.function_imports.contains_key(export) {
                return Err(ModuleError::PackageError {
                    message: format!("Wildcard import conflicts with existing import: '{}'", export),
                });
            }
        }
        
        self.wildcard_imports.insert(namespace, exports);
        Ok(())
    }
    
    fn add_aliased_import(&mut self, namespace: String, function: String, alias: String) -> Result<(), ModuleError> {
        let qualified_name = format!("{}::{}", namespace, function);
        
        // Check for conflicts
        if self.function_imports.contains_key(&alias) {
            return Err(ModuleError::PackageError {
                message: format!("Alias '{}' already in use", alias),
            });
        }
        
        self.function_imports.insert(alias, qualified_name);
        Ok(())
    }
}

impl Default for ImportContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Import statement parser
pub struct ImportParser;

impl ImportParser {
    /// Parse an import statement from text
    /// 
    /// Examples:
    /// - `import std::math` -> Module import
    /// - `import std::math::{Sin, Cos}` -> Specific imports
    /// - `import std::math::*` -> Wildcard import
    /// - `import std::math::Sin as Sine` -> Aliased import
    pub fn parse(statement: &str) -> Result<ImportStatement, ModuleError> {
        let trimmed = statement.trim();
        
        if !trimmed.starts_with("import ") {
            return Err(ModuleError::ParseError {
                message: "Import statement must start with 'import'".to_string(),
            });
        }
        
        let import_part = trimmed.strip_prefix("import ").unwrap().trim();
        
        // Check for aliased import: `import std::math::Sin as Sine`
        // But make sure it's not a specific import with aliases like {Sin, Cos as Cosine}
        // Also catch incomplete aliases like "import std::math::Sin as"
        if (import_part.contains(" as ") || import_part.ends_with(" as")) && !import_part.contains('{') {
            return Self::parse_aliased_import(import_part);
        }
        
        // Check for wildcard import: `import std::math::*`
        if import_part.ends_with("::*") {
            let namespace = import_part.strip_suffix("::*").unwrap().to_string();
            return Ok(ImportStatement::Wildcard { namespace });
        }
        
        // Check for specific imports: `import std::math::{Sin, Cos}`
        if import_part.contains('{') {
            // Must also have closing brace for valid specific import
            if !import_part.contains('}') {
                return Err(ModuleError::ParseError {
                    message: "Missing closing brace in specific import".to_string(),
                });
            }
            return Self::parse_specific_imports(import_part);
        }
        
        // Default to module import: `import std::math`
        let parts: Vec<&str> = import_part.split(" as ").collect();
        let namespace = parts[0].to_string();
        let alias = if parts.len() > 1 {
            Some(parts[1].to_string())
        } else {
            None
        };
        
        Ok(ImportStatement::Module { namespace, alias })
    }
    
    fn parse_aliased_import(import_part: &str) -> Result<ImportStatement, ModuleError> {
        // Handle the case where the input ends with " as" (incomplete alias)
        if import_part.ends_with(" as") {
            return Err(ModuleError::ParseError {
                message: "Invalid aliased import syntax - missing alias after 'as'".to_string(),
            });
        }
        
        let parts: Vec<&str> = import_part.split(" as ").collect();
        if parts.len() != 2 {
            return Err(ModuleError::ParseError {
                message: "Invalid aliased import syntax - missing alias after 'as'".to_string(),
            });
        }
        
        let qualified_name = parts[0];
        let alias = parts[1].trim().to_string();
        
        // Validate that alias is not empty
        if alias.is_empty() {
            return Err(ModuleError::ParseError {
                message: "Alias cannot be empty".to_string(),
            });
        }
        
        // Parse namespace and function
        let name_parts: Vec<&str> = qualified_name.split("::").collect();
        if name_parts.len() < 2 {
            return Err(ModuleError::ParseError {
                message: "Qualified name must contain namespace".to_string(),
            });
        }
        
        let namespace = name_parts[..name_parts.len() - 1].join("::");
        let function = name_parts[name_parts.len() - 1].to_string();
        
        Ok(ImportStatement::Aliased { namespace, function, alias })
    }
    
    fn parse_specific_imports(import_part: &str) -> Result<ImportStatement, ModuleError> {
        // Find the namespace part before the '{'
        let open_brace = import_part.find('{')
            .ok_or_else(|| ModuleError::ParseError {
                message: "Missing opening brace in specific import".to_string(),
            })?;
        
        let namespace = import_part[..open_brace].trim_end_matches("::").to_string();
        
        // Extract the function list between braces
        let close_brace = import_part.rfind('}')
            .ok_or_else(|| ModuleError::ParseError {
                message: "Missing closing brace in specific import".to_string(),
            })?;
        
        let function_list = &import_part[open_brace + 1..close_brace];
        
        let mut functions = Vec::new();
        let mut aliases = HashMap::new();
        
        // Parse each function in the list
        for item in function_list.split(',') {
            let item = item.trim();
            if item.is_empty() {
                continue;
            }
            
            if item.contains(" as ") {
                // Handle aliased function: `Sin as Sine`
                let parts: Vec<&str> = item.split(" as ").collect();
                if parts.len() != 2 {
                    return Err(ModuleError::ParseError {
                        message: format!("Invalid alias syntax: {}", item),
                    });
                }
                let function_name = parts[0].trim().to_string();
                let alias = parts[1].trim().to_string();
                
                functions.push(function_name.clone());
                aliases.insert(function_name, alias);
            } else {
                // Regular function import
                functions.push(item.to_string());
            }
        }
        
        if functions.is_empty() {
            return Err(ModuleError::ParseError {
                message: "No functions specified in import".to_string(),
            });
        }
        
        Ok(ImportStatement::Specific { namespace, functions, aliases })
    }
}

/// Scope manager for nested scopes and local bindings
#[derive(Debug, Clone)]
pub struct ScopeManager {
    /// Stack of scopes (outermost to innermost)
    scopes: Vec<Scope>,
    
    /// Global import context
    import_context: ImportContext,
}

#[derive(Debug, Clone)]
struct Scope {
    /// Local variable bindings
    variables: HashMap<String, String>, // name -> type/value info
    
    /// Local function definitions
    functions: HashMap<String, String>, // name -> definition info
}

impl ScopeManager {
    /// Create a new scope manager with global import context
    pub fn new(import_context: ImportContext) -> Self {
        ScopeManager {
            scopes: vec![Scope::new()], // Global scope
            import_context,
        }
    }
    
    /// Push a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(Scope::new());
    }
    
    /// Pop the current scope
    pub fn pop_scope(&mut self) -> Result<(), ModuleError> {
        if self.scopes.len() <= 1 {
            return Err(ModuleError::PackageError {
                message: "Cannot pop global scope".to_string(),
            });
        }
        self.scopes.pop();
        Ok(())
    }
    
    /// Resolve a name in the current scope chain
    pub fn resolve_name(&self, name: &str) -> Option<String> {
        // Check local scopes first (innermost to outermost)
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.resolve_name(name) {
                return Some(binding);
            }
        }
        
        // Check import context
        self.import_context.resolve_function(name)
    }
    
    /// Define a variable in the current scope
    pub fn define_variable(&mut self, name: String, type_info: String) -> Result<(), ModuleError> {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.define_variable(name, type_info);
            Ok(())
        } else {
            Err(ModuleError::PackageError {
                message: "No active scope".to_string(),
            })
        }
    }
    
    /// Define a function in the current scope
    pub fn define_function(&mut self, name: String, definition: String) -> Result<(), ModuleError> {
        if let Some(current_scope) = self.scopes.last_mut() {
            current_scope.define_function(name, definition);
            Ok(())
        } else {
            Err(ModuleError::PackageError {
                message: "No active scope".to_string(),
            })
        }
    }
    
    /// Get the current scope depth
    pub fn scope_depth(&self) -> usize {
        self.scopes.len()
    }
    
    /// Check if a name is defined in local scopes
    pub fn is_local(&self, name: &str) -> bool {
        for scope in &self.scopes {
            if scope.has_name(name) {
                return true;
            }
        }
        false
    }
    
    /// Get all names visible in current context
    pub fn visible_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        
        // Collect from all scopes
        for scope in &self.scopes {
            names.extend(scope.all_names());
        }
        
        // Add imported functions
        names.extend(self.import_context.imported_functions());
        
        names.sort();
        names.dedup();
        names
    }
}

impl Scope {
    fn new() -> Self {
        Scope {
            variables: HashMap::new(),
            functions: HashMap::new(),
        }
    }
    
    fn resolve_name(&self, name: &str) -> Option<String> {
        if let Some(var_info) = self.variables.get(name) {
            Some(format!("variable:{}", var_info))
        } else if let Some(func_info) = self.functions.get(name) {
            Some(format!("function:{}", func_info))
        } else {
            None
        }
    }
    
    fn define_variable(&mut self, name: String, type_info: String) {
        self.variables.insert(name, type_info);
    }
    
    fn define_function(&mut self, name: String, definition: String) {
        self.functions.insert(name, definition);
    }
    
    fn has_name(&self, name: &str) -> bool {
        self.variables.contains_key(name) || self.functions.contains_key(name)
    }
    
    fn all_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        names.extend(self.variables.keys().cloned());
        names.extend(self.functions.keys().cloned());
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linker::FunctionRegistry;
    use std::sync::{Arc, RwLock};

    fn create_test_registry() -> ModuleRegistry {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        ModuleRegistry::new(func_registry)
    }

    #[test]
    fn test_import_statement_parsing() {
        // Test module import
        let import = ImportParser::parse("import std::math").unwrap();
        assert!(matches!(import, ImportStatement::Module { .. }));
        
        // Test wildcard import
        let import = ImportParser::parse("import std::math::*").unwrap();
        assert!(matches!(import, ImportStatement::Wildcard { .. }));
        
        // Test specific imports
        let import = ImportParser::parse("import std::math::{Sin, Cos}").unwrap();
        assert!(matches!(import, ImportStatement::Specific { .. }));
        
        // Test aliased import
        let import = ImportParser::parse("import std::math::Sin as Sine").unwrap();
        assert!(matches!(import, ImportStatement::Aliased { .. }));
    }

    #[test]
    fn test_import_context() {
        let registry = create_test_registry();
        let mut context = ImportContext::new();
        
        // Test module import
        let module_import = ImportStatement::Module {
            namespace: "std::math".to_string(),
            alias: Some("math".to_string()),
        };
        
        context.add_import(module_import, &registry).unwrap();
        assert_eq!(context.imported_modules().len(), 1);
        
        // Test function resolution
        context.function_imports.insert("Sin".to_string(), "std::math::Sin".to_string());
        assert_eq!(context.resolve_function("Sin"), Some("std::math::Sin".to_string()));
        assert_eq!(context.resolve_function("Unknown"), None);
    }

    #[test]
    fn test_scope_manager() {
        let import_context = ImportContext::new();
        let mut scope_manager = ScopeManager::new(import_context);
        
        // Test initial state
        assert_eq!(scope_manager.scope_depth(), 1);
        
        // Test scope operations
        scope_manager.push_scope();
        assert_eq!(scope_manager.scope_depth(), 2);
        
        scope_manager.define_variable("x".to_string(), "Integer".to_string()).unwrap();
        assert!(scope_manager.is_local("x"));
        
        scope_manager.pop_scope().unwrap();
        assert_eq!(scope_manager.scope_depth(), 1);
        assert!(!scope_manager.is_local("x"));
    }

    #[test]
    fn test_specific_import_parsing() {
        let import = ImportParser::parse("import std::math::{Sin, Cos as Cosine, Tan}").unwrap();
        
        if let ImportStatement::Specific { namespace, functions, aliases } = import {
            assert_eq!(namespace, "std::math");
            assert_eq!(functions.len(), 3);
            assert!(functions.contains(&"Sin".to_string()));
            assert!(functions.contains(&"Cos".to_string()));
            assert!(functions.contains(&"Tan".to_string()));
            assert_eq!(aliases.get("Cos"), Some(&"Cosine".to_string()));
        } else {
            panic!("Expected specific import");
        }
    }

    #[test]
    fn test_invalid_import_parsing() {
        // Test invalid syntax
        assert!(ImportParser::parse("invalid statement").is_err());
        assert!(ImportParser::parse("import").is_err());
        assert!(ImportParser::parse("import std::math::{").is_err());
        assert!(ImportParser::parse("import std::math::Sin as").is_err());
    }
}