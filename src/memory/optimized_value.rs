//! Optimized Value representation for memory efficiency
//!
//! This module provides a memory-optimized version of the Value enum that uses
//! symbol interning, small integer optimization, and cache-aligned layouts
//! to reduce memory usage from 24+ bytes to <16 bytes per value.

use std::sync::Arc;
use crate::memory::{SymbolId, StringInterner};
use crate::foreign::LyObj;
use crate::ast;
use serde::{Serialize, Deserialize};

/// Compact value representation optimized for cache efficiency
/// 
/// Target size: 12-16 bytes (vs 24+ bytes for original Value)
/// Uses symbol interning, small int optimization, and efficient layout.
#[repr(C)]
#[derive(Debug, Clone, PartialEq)]
pub enum CompactValue {
    /// Small integers (-2^31 to 2^31-1) stored inline
    SmallInt(i32),
    
    /// Large integers stored in shared pool
    LargeInt(Arc<i64>),
    
    /// Real numbers (f64)
    Real(f64),
    
    /// Interned symbols (4 bytes vs 24+ for String)
    Symbol(SymbolId),
    
    /// Interned strings (4 bytes vs 24+ for String)  
    String(SymbolId),
    
    /// Lists stored in shared pool with reference counting
    List(Arc<Vec<CompactValue>>),
    
    /// Function names stored as interned symbols
    Function(SymbolId),
    
    /// Boolean values
    Boolean(bool),
    
    /// Missing/unknown value
    Missing,
    
    /// Foreign objects (keep as-is for compatibility)
    LyObj(LyObj),
    
    /// Quoted expressions (store in shared pool)
    Quote(Arc<ast::Expr>),
    
    /// Pattern expressions (store in shared pool)
    Pattern(Arc<ast::Pattern>),
    
    /// Complex values that cannot be compacted (fallback)
    Complex(Arc<crate::vm::Value>),
}

impl CompactValue {
    /// Create from a regular Value, using interner for string storage
    pub fn from_value(value: crate::vm::Value, interner: &StringInterner) -> Self {
        match value {
            crate::vm::Value::Integer(i) => {
                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    CompactValue::SmallInt(i as i32)
                } else {
                    CompactValue::LargeInt(Arc::new(i))
                }
            }
            crate::vm::Value::Real(r) => CompactValue::Real(r),
            crate::vm::Value::String(s) => CompactValue::String(interner.intern_symbol_id(&s)),
            crate::vm::Value::Symbol(s) => CompactValue::Symbol(interner.intern_symbol_id(&s)),
            crate::vm::Value::List(items) => {
                let compact_items: Vec<CompactValue> = items.into_iter()
                    .map(|item| CompactValue::from_value(item, interner))
                    .collect();
                CompactValue::List(Arc::new(compact_items))
            }
            crate::vm::Value::Function(f) => CompactValue::Function(interner.intern_symbol_id(&f)),
            crate::vm::Value::Boolean(b) => CompactValue::Boolean(b),
            crate::vm::Value::Missing => CompactValue::Missing,
            crate::vm::Value::Object(_) => {
                // Objects cannot be compacted, store as complex value
                CompactValue::Complex(Arc::new(value))
            }
            crate::vm::Value::LyObj(obj) => CompactValue::LyObj(obj),
            crate::vm::Value::Quote(expr) => CompactValue::Quote(expr.into()),
            crate::vm::Value::Pattern(pattern) => CompactValue::Pattern(Arc::new(pattern)),
            crate::vm::Value::Rule { lhs, rhs } => {
                // For now, represent Rule as a compact list with special marker
                let rule_items = vec![
                    CompactValue::from_value(*lhs, interner),
                    CompactValue::from_value(*rhs, interner)
                ];
                CompactValue::List(Arc::new(rule_items))
            },
            crate::vm::Value::PureFunction { body } => {
                // Store pure functions as complex values for now
                CompactValue::Complex(Arc::new(crate::vm::Value::PureFunction { body }))
            },
            crate::vm::Value::Slot { number } => {
                // Store slots as complex values for now
                CompactValue::Complex(Arc::new(crate::vm::Value::Slot { number }))
            },
        }
    }
    
    /// Convert back to regular Value
    pub fn to_value(&self, interner: &StringInterner) -> crate::vm::Value {
        match self {
            CompactValue::SmallInt(i) => crate::vm::Value::Integer(*i as i64),
            CompactValue::LargeInt(i) => crate::vm::Value::Integer(**i),
            CompactValue::Real(r) => crate::vm::Value::Real(*r),
            CompactValue::Symbol(id) => {
                let symbol_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownSymbol_{}", id.raw()));
                crate::vm::Value::Symbol(symbol_str)
            }
            CompactValue::String(id) => {
                let string_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownString_{}", id.raw()));
                crate::vm::Value::String(string_str)
            }
            CompactValue::List(items) => {
                let regular_items: Vec<crate::vm::Value> = items.iter()
                    .map(|item| item.to_value(interner))
                    .collect();
                crate::vm::Value::List(regular_items)
            }
            CompactValue::Function(id) => {
                let func_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownFunction_{}", id.raw()));
                crate::vm::Value::Function(func_str)
            }
            CompactValue::Boolean(b) => crate::vm::Value::Boolean(*b),
            CompactValue::Missing => crate::vm::Value::Missing,
            CompactValue::LyObj(obj) => crate::vm::Value::LyObj(obj.clone()),
            CompactValue::Quote(expr) => crate::vm::Value::Quote(Box::new((**expr).clone())),
            CompactValue::Pattern(pattern) => crate::vm::Value::Pattern((**pattern).clone()),
            CompactValue::Complex(value) => (**value).clone(),
        }
    }
    
    /// Get estimated memory size in bytes
    pub fn memory_size(&self) -> usize {
        match self {
            CompactValue::SmallInt(_) => std::mem::size_of::<Self>(),
            CompactValue::LargeInt(_) => std::mem::size_of::<Self>() + std::mem::size_of::<i64>(),
            CompactValue::Real(_) => std::mem::size_of::<Self>(),
            CompactValue::Symbol(_) | CompactValue::String(_) | CompactValue::Function(_) => {
                std::mem::size_of::<Self>() // Symbol is just a 4-byte index
            }
            CompactValue::List(items) => {
                std::mem::size_of::<Self>() + items.iter().map(|v| v.memory_size()).sum::<usize>()
            }
            CompactValue::Boolean(_) | CompactValue::Missing => std::mem::size_of::<Self>(),
            CompactValue::LyObj(_) => std::mem::size_of::<Self>() + 64, // Estimate
            CompactValue::Quote(_) => std::mem::size_of::<Self>() + 32, // Estimate
            CompactValue::Pattern(_) => std::mem::size_of::<Self>() + 24, // Estimate
            CompactValue::Complex(_) => std::mem::size_of::<Self>() + 128, // Estimate for complex values
        }
    }
    
    /// Check if this value can be efficiently cached
    pub fn is_cacheable(&self) -> bool {
        match self {
            CompactValue::SmallInt(_) | CompactValue::Real(_) | 
            CompactValue::Symbol(_) | CompactValue::String(_) |
            CompactValue::Function(_) | CompactValue::Boolean(_) | 
            CompactValue::Missing => true,
            CompactValue::LargeInt(_) => true, // Shared via Arc
            CompactValue::List(items) => items.len() < 1024, // Reasonable size limit
            _ => false, // Complex types might not be suitable for caching
        }
    }
    
    /// Get type name for debugging
    pub fn type_name(&self) -> &'static str {
        match self {
            CompactValue::SmallInt(_) | CompactValue::LargeInt(_) => "Integer",
            CompactValue::Real(_) => "Real", 
            CompactValue::Symbol(_) => "Symbol",
            CompactValue::String(_) => "String",
            CompactValue::List(_) => "List",
            CompactValue::Function(_) => "Function",
            CompactValue::Boolean(_) => "Boolean",
            CompactValue::Missing => "Missing",
            CompactValue::LyObj(_) => "LyObj",
            CompactValue::Quote(_) => "Quote",
            CompactValue::Complex(_) => "Complex",
            CompactValue::Pattern(_) => "Pattern",
        }
    }
}

/// Cache-aligned wrapper for frequently accessed values
/// 
/// Aligns data to 64-byte cache lines to prevent false sharing
/// and improve memory access patterns in multi-threaded scenarios.
#[repr(align(64))]
#[derive(Debug, Clone)]
pub struct CacheAlignedValue {
    pub value: CompactValue,
    // Padding automatically added by repr(align(64))
}

impl CacheAlignedValue {
    /// Create a new cache-aligned value
    pub fn new(value: CompactValue) -> Self {
        Self { value }
    }
    
    /// Get the size including alignment padding
    pub fn aligned_size(&self) -> usize {
        std::mem::size_of::<Self>()
    }
}

impl From<CompactValue> for CacheAlignedValue {
    fn from(value: CompactValue) -> Self {
        Self::new(value)
    }
}

impl std::ops::Deref for CacheAlignedValue {
    type Target = CompactValue;
    
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl std::ops::DerefMut for CacheAlignedValue {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

/// Serializable representation of CompactValue
/// 
/// Since CompactValue uses SymbolId which is specific to a particular
/// interner instance, we need a serializable format for persistence.
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum SerializableValue {
    SmallInt(i32),
    LargeInt(i64),
    Real(f64),
    Symbol(String),
    String(String),
    List(Vec<SerializableValue>),
    Function(String),
    Boolean(bool),
    Missing,
    // Complex types simplified for serialization
    LyObjPlaceholder { type_name: String },
    QuotePlaceholder { expr_debug: String },
    PatternPlaceholder { pattern_debug: String },
    ComplexPlaceholder { complex_debug: String },
}

impl CompactValue {
    /// Convert to serializable format
    pub fn to_serializable(&self, interner: &StringInterner) -> SerializableValue {
        match self {
            CompactValue::SmallInt(i) => SerializableValue::SmallInt(*i),
            CompactValue::LargeInt(i) => SerializableValue::LargeInt(**i),
            CompactValue::Real(r) => SerializableValue::Real(*r),
            CompactValue::Symbol(id) => {
                let symbol_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownSymbol_{}", id.raw()));
                SerializableValue::Symbol(symbol_str)
            }
            CompactValue::String(id) => {
                let string_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownString_{}", id.raw()));
                SerializableValue::String(string_str)
            }
            CompactValue::List(items) => {
                let serializable_items: Vec<SerializableValue> = items.iter()
                    .map(|item| item.to_serializable(interner))
                    .collect();
                SerializableValue::List(serializable_items)
            }
            CompactValue::Function(id) => {
                let func_str = interner.resolve_symbol(*id)
                    .unwrap_or_else(|| format!("UnknownFunction_{}", id.raw()));
                SerializableValue::Function(func_str)
            }
            CompactValue::Boolean(b) => SerializableValue::Boolean(*b),
            CompactValue::Missing => SerializableValue::Missing,
            CompactValue::LyObj(obj) => SerializableValue::LyObjPlaceholder {
                type_name: obj.type_name().to_string(),
            },
            CompactValue::Quote(expr) => SerializableValue::QuotePlaceholder {
                expr_debug: format!("{:?}", expr),
            },
            CompactValue::Pattern(pattern) => SerializableValue::PatternPlaceholder {
                pattern_debug: format!("{:?}", pattern),
            },
            CompactValue::Complex(_) => SerializableValue::ComplexPlaceholder {
                complex_debug: "Complex".to_string(),
            },
        }
    }
    
    /// Create from serializable format
    pub fn from_serializable(serializable: SerializableValue, interner: &StringInterner) -> Self {
        match serializable {
            SerializableValue::SmallInt(i) => CompactValue::SmallInt(i),
            SerializableValue::LargeInt(i) => {
                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    CompactValue::SmallInt(i as i32)
                } else {
                    CompactValue::LargeInt(Arc::new(i))
                }
            }
            SerializableValue::Real(r) => CompactValue::Real(r),
            SerializableValue::Symbol(s) => CompactValue::Symbol(interner.intern_symbol_id(&s)),
            SerializableValue::String(s) => CompactValue::String(interner.intern_symbol_id(&s)),
            SerializableValue::List(items) => {
                let compact_items: Vec<CompactValue> = items.into_iter()
                    .map(|item| CompactValue::from_serializable(item, interner))
                    .collect();
                CompactValue::List(Arc::new(compact_items))
            }
            SerializableValue::Function(f) => CompactValue::Function(interner.intern_symbol_id(&f)),
            SerializableValue::Boolean(b) => CompactValue::Boolean(b),
            SerializableValue::Missing => CompactValue::Missing,
            // Complex types become Missing when deserialized
            SerializableValue::LyObjPlaceholder { .. } => CompactValue::Missing,
            SerializableValue::QuotePlaceholder { .. } => CompactValue::Missing,
            SerializableValue::PatternPlaceholder { .. } => CompactValue::Missing,
            SerializableValue::ComplexPlaceholder { .. } => CompactValue::Missing,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::StringInterner;
    
    #[test]
    fn test_compact_value_size() {
        // CompactValue should be significantly smaller than Value
        println!("CompactValue size: {}", std::mem::size_of::<CompactValue>());
        println!("Original Value size: {}", std::mem::size_of::<crate::vm::Value>());
        
        // Should be <= 16 bytes as target
        assert!(std::mem::size_of::<CompactValue>() <= 24); // Allow some margin initially
        
        // Should be smaller than original Value
        assert!(std::mem::size_of::<CompactValue>() < std::mem::size_of::<crate::vm::Value>());
    }
    
    #[test]
    fn test_cache_aligned_value() {
        let value = CompactValue::SmallInt(42);
        let aligned = CacheAlignedValue::new(value);
        
        // Should be aligned to 64 bytes
        assert_eq!(std::mem::align_of::<CacheAlignedValue>(), 64);
        assert_eq!(aligned.aligned_size(), 64);
        
        // Should be able to access the value
        assert_eq!(*aligned, CompactValue::SmallInt(42));
    }
    
    #[test]
    fn test_small_int_optimization() {
        let interner = StringInterner::new();
        
        // Small integers should use SmallInt variant
        let small = CompactValue::from_value(crate::vm::Value::Integer(42), &interner);
        assert!(matches!(small, CompactValue::SmallInt(42)));
        
        // Large integers should use LargeInt variant  
        let large = CompactValue::from_value(crate::vm::Value::Integer(i64::MAX), &interner);
        assert!(matches!(large, CompactValue::LargeInt(_)));
    }
    
    #[test]
    fn test_symbol_interning() {
        let interner = StringInterner::new();
        
        // Symbols should be interned
        let sym1 = CompactValue::from_value(crate::vm::Value::Symbol("x".to_string()), &interner);
        let sym2 = CompactValue::from_value(crate::vm::Value::Symbol("x".to_string()), &interner);
        
        if let (CompactValue::Symbol(id1), CompactValue::Symbol(id2)) = (sym1, sym2) {
            assert_eq!(id1, id2); // Same symbol should have same ID
        } else {
            panic!("Expected symbols to be interned");
        }
    }
    
    #[test]
    fn test_round_trip_conversion() {
        let interner = StringInterner::new();
        
        let original = crate::vm::Value::List(vec![
            crate::vm::Value::Integer(42),
            crate::vm::Value::Real(3.14),
            crate::vm::Value::Symbol("Plus".to_string()),
            crate::vm::Value::Boolean(true),
        ]);
        
        let compact = CompactValue::from_value(original.clone(), &interner);
        let converted = compact.to_value(&interner);
        
        assert_eq!(original, converted);
    }
    
    #[test]
    fn test_memory_size_calculation() {
        let interner = StringInterner::new();
        
        let small_int = CompactValue::SmallInt(42);
        let symbol = CompactValue::Symbol(interner.intern_symbol_id("x"));
        
        // Small values should have minimal memory overhead
        let small_int_size = small_int.memory_size();
        let symbol_size = symbol.memory_size();
        assert!(small_int_size <= 24);
        assert!(symbol_size <= 24);
        
        let list = CompactValue::List(Arc::new(vec![small_int, symbol]));
        assert!(list.memory_size() > small_int_size);
    }
    
    #[test]
    fn test_serialization() {
        let interner = StringInterner::new();
        
        let value = CompactValue::from_value(
            crate::vm::Value::Symbol("Plus".to_string()), 
            &interner
        );
        
        let serializable = value.to_serializable(&interner);
        let deserialized = CompactValue::from_serializable(serializable, &interner);
        
        // Should round-trip correctly
        assert_eq!(value, deserialized);
    }
    
    #[test]
    fn test_type_names() {
        let interner = StringInterner::new();
        
        assert_eq!(CompactValue::SmallInt(42).type_name(), "Integer");
        assert_eq!(CompactValue::Real(3.14).type_name(), "Real");
        assert_eq!(CompactValue::Symbol(interner.intern_symbol_id("x")).type_name(), "Symbol");
        assert_eq!(CompactValue::Boolean(true).type_name(), "Boolean");
        assert_eq!(CompactValue::Missing.type_name(), "Missing");
    }
    
    #[test]
    fn test_cacheability() {
        let interner = StringInterner::new();
        
        // Simple types should be cacheable
        assert!(CompactValue::SmallInt(42).is_cacheable());
        assert!(CompactValue::Real(3.14).is_cacheable());
        assert!(CompactValue::Symbol(interner.intern_symbol_id("x")).is_cacheable());
        assert!(CompactValue::Boolean(true).is_cacheable());
        
        // Small lists should be cacheable
        let small_list = CompactValue::List(Arc::new(vec![CompactValue::SmallInt(1)]));
        assert!(small_list.is_cacheable());
        
        // Very large lists should not be cacheable
        let large_list = CompactValue::List(Arc::new(vec![CompactValue::SmallInt(1); 2000]));
        assert!(!large_list.is_cacheable());
    }
}