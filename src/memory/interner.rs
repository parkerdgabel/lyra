//! String interning system for memory-efficient symbol management
//!
//! This module provides a high-performance string interning system optimized
//! for the common patterns in symbolic computation where symbols like "x", "y",
//! "Plus", "Times" are used repeatedly.

use std::sync::Arc;
use std::collections::HashMap;
use parking_lot::RwLock;
use crate::memory::stats::InternerStats;

/// An interned string reference that provides O(1) equality and hashing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InternedString {
    ptr: *const str,
}

unsafe impl Send for InternedString {}
unsafe impl Sync for InternedString {}

impl InternedString {
    /// Create from a static string reference
    pub fn from_static(s: &'static str) -> Self {
        Self { ptr: s as *const str }
    }
    
    /// Get the string slice
    pub fn as_str(&self) -> &str {
        unsafe { &*self.ptr }
    }
    
    /// Get the length of the interned string
    pub fn len(&self) -> usize {
        self.as_str().len()
    }
    
    /// Check if the string is empty
    pub fn is_empty(&self) -> bool {
        self.as_str().is_empty()
    }
}

impl std::fmt::Display for InternedString {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// High-performance string interning system
/// 
/// Uses a two-tier approach:
/// 1. Static interning for common symbols (O(1) lookup)
/// 2. Dynamic interning with arena allocation for other strings
pub struct StringInterner {
    /// Cache of dynamically interned strings
    dynamic_cache: RwLock<HashMap<String, InternedString>>,
    /// Arena for string storage
    arena: RwLock<Vec<String>>,
    /// Statistics for monitoring performance
    stats: RwLock<InternerStats>,
}

impl StringInterner {
    /// Common symbols that appear frequently in symbolic computation
    const COMMON_SYMBOLS: &'static [&'static str] = &[
        // Variables
        "x", "y", "z", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
        
        // Mathematical functions
        "Plus", "Times", "Power", "Subtract", "Divide", "Mod",
        "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan",
        "Exp", "Log", "Log10", "Sqrt", "Abs", "Sign",
        "Floor", "Ceiling", "Round", "Min", "Max",
        
        // Logical operations
        "And", "Or", "Not", "Equal", "Unequal", "Less", "Greater", "LessEqual", "GreaterEqual",
        
        // List operations  
        "List", "Length", "Head", "Tail", "Append", "Prepend", "Join", "Flatten", "Map", "Apply",
        
        // Tensor operations
        "Array", "Dot", "Transpose", "ArrayDimensions", "ArrayRank", "ArrayReshape", "ArrayFlatten",
        "Maximum", "Minimum", "Mean", "Sum", "Product",
        
        // Pattern matching
        "Rule", "RuleDelayed", "ReplaceAll", "Pattern", "Blank", "BlankSequence", "BlankNullSequence",
        
        // Control structures
        "If", "While", "For", "Do", "Function", "Module", "Block", "With",
        
        // Constants
        "True", "False", "Null", "Missing", "Undefined", "Pi", "E", "I", "Infinity",
        
        // Types
        "Integer", "Real", "Complex", "String", "Symbol", "Boolean", "Expression",
        
        // Attributes
        "Hold", "HoldAll", "HoldFirst", "HoldRest", "Listable", "Flat", "OneIdentity", "Orderless",
        
        // Common variable names
        "result", "value", "expr", "args", "func", "data", "input", "output", "temp", "var",
        
        // Single character strings (very common in parsing)
        " ", "\t", "\n", "\r", "(", ")", "[", "]", "{", "}", ",", ";", ":", "=", "+", "-", "*", "/", "^", "&", "|", "!", "<", ">", "?", "@", "#", "$", "%", "_", ".", "'", "\"", "\\", "`", "~"
    ];
    
    /// Create a new string interner
    pub fn new() -> Self {
        Self {
            dynamic_cache: RwLock::new(HashMap::with_capacity(1024)),
            arena: RwLock::new(Vec::with_capacity(1024)),
            stats: RwLock::new(InternerStats::default()),
        }
    }
    
    /// Intern a string, returning an efficient reference
    pub fn intern(&self, s: &str) -> InternedString {
        // First check static symbols (O(1) for common cases)
        for &static_str in Self::COMMON_SYMBOLS {
            if s == static_str {
                self.stats.write().static_hits += 1;
                return InternedString::from_static(static_str);
            }
        }
        
        // Check dynamic cache
        {
            let cache = self.dynamic_cache.read();
            if let Some(&interned) = cache.get(s) {
                self.stats.write().dynamic_hits += 1;
                return interned;
            }
        }
        
        // Need to intern a new string
        let mut cache = self.dynamic_cache.write();
        let mut arena = self.arena.write();
        let mut stats = self.stats.write();
        
        // Double-check in case another thread added it
        if let Some(&interned) = cache.get(s) {
            stats.dynamic_hits += 1;
            return interned;
        }
        
        // Add to arena and cache
        let owned = s.to_string();
        arena.push(owned);
        let last_string = arena.last().unwrap();
        let interned = InternedString { ptr: last_string.as_str() as *const str };
        
        cache.insert(s.to_string(), interned);
        stats.dynamic_misses += 1;
        stats.total_interned += 1;
        stats.total_bytes += s.len();
        
        interned
    }
    
    /// Get interning statistics
    pub fn stats(&self) -> InternerStats {
        *self.stats.read()
    }
    
    /// Get the number of dynamically interned strings
    pub fn dynamic_count(&self) -> usize {
        self.arena.read().len()
    }
    
    /// Get total memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        let arena = self.arena.read();
        let cache_size = self.dynamic_cache.read().capacity() * (std::mem::size_of::<String>() + std::mem::size_of::<InternedString>());
        let arena_size = arena.iter().map(|s| s.capacity()).sum::<usize>();
        cache_size + arena_size
    }
    
    /// Attempt to shrink memory usage
    pub fn shrink_to_fit(&self) {
        let mut arena = self.arena.write();
        arena.shrink_to_fit();
        for s in arena.iter_mut() {
            s.shrink_to_fit();
        }
    }
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_static_interning() {
        let interner = StringInterner::new();
        
        let x1 = interner.intern("x");
        let x2 = interner.intern("x");
        
        // Should be identical pointers for static strings
        assert!(std::ptr::eq(x1.as_str(), x2.as_str()));
        assert_eq!(x1.as_str(), "x");
        
        let stats = interner.stats();
        assert_eq!(stats.static_hits, 2);
        assert_eq!(stats.dynamic_misses, 0);
    }
    
    #[test]
    fn test_dynamic_interning() {
        let interner = StringInterner::new();
        
        let custom1 = interner.intern("my_custom_symbol");
        let custom2 = interner.intern("my_custom_symbol");
        
        // Should be identical pointers for dynamic strings
        assert!(std::ptr::eq(custom1.as_str(), custom2.as_str()));
        assert_eq!(custom1.as_str(), "my_custom_symbol");
        
        let stats = interner.stats();
        assert_eq!(stats.dynamic_hits, 1);
        assert_eq!(stats.dynamic_misses, 1);
        assert_eq!(stats.total_interned, 1);
    }
    
    #[test]
    fn test_mixed_interning() {
        let interner = StringInterner::new();
        
        // Mix of static and dynamic
        let _x = interner.intern("x");
        let _plus = interner.intern("Plus");
        let _custom = interner.intern("MyFunction");
        let _y = interner.intern("y");
        
        let stats = interner.stats();
        assert_eq!(stats.static_hits, 3); // x, Plus, y
        assert_eq!(stats.dynamic_misses, 1); // MyFunction
        assert_eq!(stats.total_interned, 1); // Only MyFunction goes to arena
    }
    
    #[test]
    fn test_interned_string_properties() {
        let interner = StringInterner::new();
        let interned = interner.intern("test_string");
        
        assert_eq!(interned.len(), 11);
        assert!(!interned.is_empty());
        assert_eq!(interned.to_string(), "test_string");
    }
    
    #[test]
    fn test_memory_usage() {
        let interner = StringInterner::new();
        
        let initial_usage = interner.memory_usage();
        
        // Add some strings
        interner.intern("x"); // static
        interner.intern("custom_function_name"); // dynamic
        
        let final_usage = interner.memory_usage();
        
        // Static strings shouldn't increase memory usage
        // Dynamic strings should increase it moderately
        assert!(final_usage > initial_usage);
        assert!(final_usage < initial_usage + 1024); // Reasonable overhead
    }
    
    #[test]
    fn test_common_symbols_coverage() {
        let interner = StringInterner::new();
        
        // Test that all common mathematical operations are covered
        let math_symbols = vec!["Plus", "Times", "Sin", "Cos", "Log", "Exp"];
        for symbol in math_symbols {
            interner.intern(symbol);
        }
        
        let stats = interner.stats();
        assert_eq!(stats.static_hits, 6);
        assert_eq!(stats.dynamic_misses, 0);
    }
}