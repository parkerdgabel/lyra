//! Standard Library for Lyra
//! 
//! This module provides the core standard library functions that are built into
//! the Lyra symbolic computation engine. Functions are organized by category:
//! - List operations
//! - String operations  
//! - Math functions
//! - Pattern matching and rules

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

pub mod list;
pub mod string;
pub mod math;
pub mod rules;

/// Standard library function signature
pub type StdlibFunction = fn(&[Value]) -> VmResult<Value>;

/// Registry of all standard library functions
#[derive(Debug)]
pub struct StandardLibrary {
    functions: HashMap<String, StdlibFunction>,
}

impl StandardLibrary {
    /// Create a new standard library with all built-in functions registered
    pub fn new() -> Self {
        let mut stdlib = StandardLibrary {
            functions: HashMap::new(),
        };
        
        // Register all function categories
        stdlib.register_list_functions();
        stdlib.register_string_functions();
        stdlib.register_math_functions();
        stdlib.register_rule_functions();
        
        stdlib
    }
    
    /// Look up a function by name
    pub fn get_function(&self, name: &str) -> Option<StdlibFunction> {
        self.functions.get(name).copied()
    }
    
    /// Register a function with the given name
    pub fn register(&mut self, name: impl Into<String>, func: StdlibFunction) {
        self.functions.insert(name.into(), func);
    }
    
    /// Get all registered function names
    pub fn function_names(&self) -> impl Iterator<Item = &String> {
        self.functions.keys()
    }
    
    // Registration functions for each category
    fn register_list_functions(&mut self) {
        self.register("Length", list::length);
        self.register("Head", list::head);
        self.register("Tail", list::tail);
        self.register("Append", list::append);
        self.register("Flatten", list::flatten);
        self.register("Map", list::map);
        self.register("Apply", list::apply);
    }
    
    fn register_string_functions(&mut self) {
        self.register("StringJoin", string::string_join);
        self.register("StringLength", string::string_length);
        self.register("StringTake", string::string_take);
        self.register("StringDrop", string::string_drop);
    }
    
    fn register_math_functions(&mut self) {
        // Basic arithmetic already handled by VM
        self.register("Sin", math::sin);
        self.register("Cos", math::cos);
        self.register("Tan", math::tan);
        self.register("Exp", math::exp);
        self.register("Log", math::log);
        self.register("Sqrt", math::sqrt);
    }
    
    fn register_rule_functions(&mut self) {
        self.register("ReplaceAll", rules::replace_all);
        self.register("Rule", rules::rule);
        self.register("RuleDelayed", rules::rule_delayed);
    }
}

impl Default for StandardLibrary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stdlib_creation() {
        let stdlib = StandardLibrary::new();
        
        // Verify key functions are registered
        assert!(stdlib.get_function("Length").is_some());
        assert!(stdlib.get_function("Head").is_some());
        assert!(stdlib.get_function("StringJoin").is_some());
        assert!(stdlib.get_function("Sin").is_some());
        assert!(stdlib.get_function("ReplaceAll").is_some());
    }
    
    #[test]
    fn test_stdlib_function_count() {
        let stdlib = StandardLibrary::new();
        let function_count = stdlib.function_names().count();
        
        // Should have at least the core functions we're implementing
        assert!(function_count >= 16); // 7 list + 4 string + 6 math + 3 rules = 20 minimum
    }
    
    #[test]
    fn test_function_lookup() {
        let stdlib = StandardLibrary::new();
        
        // Test valid lookup
        assert!(stdlib.get_function("Length").is_some());
        
        // Test invalid lookup
        assert!(stdlib.get_function("NonexistentFunction").is_none());
    }
}