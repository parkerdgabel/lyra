//! Managed value system for memory-efficient representation
//!
//! This module provides a compressed, memory-efficient representation of Lyra values
//! that reduces memory usage through union-based storage and tagged pointers.

use std::fmt;
use crate::vm::{Value, VmResult, VmError};
use crate::memory::InternedString;

/// Compact value tag for discriminating union data
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueTag {
    Integer = 0,
    Real = 1,
    String = 2,
    Symbol = 3,
    Boolean = 4,
    Missing = 5,
    LyObj = 6,
    List = 7,
    Quote = 8,
    Pattern = 9,
}

/// Union data storage for different value types
/// Using safer shared pointer approach instead of raw pointers
#[repr(C)]
#[derive(Copy, Clone)]
pub union ValueData {
    pub integer: i64,
    pub real: f64,
    pub string: InternedString,
    pub symbol: InternedString,
    pub boolean: bool,
    pub lyobj_index: usize, // Index into shared object pool
    pub list_index: usize,  // Index into shared list pool  
    pub quote_index: usize, // Index into shared expression pool
    pub pattern_index: usize, // Index into shared pattern pool
}

/// Memory-efficient value representation using tagged unions
/// 
/// Reduces memory overhead from ~32 bytes (Value enum) to ~16 bytes per value
/// while maintaining type safety through compile-time tag checking.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ManagedValue {
    pub tag: ValueTag,
    pub data: ValueData,
}

impl ManagedValue {
    /// Create a managed integer value
    pub fn integer(value: i64) -> Self {
        Self {
            tag: ValueTag::Integer,
            data: ValueData { integer: value },
        }
    }
    
    /// Create a managed real value
    pub fn real(value: f64) -> Self {
        Self {
            tag: ValueTag::Real,
            data: ValueData { real: value },
        }
    }
    
    /// Create a managed string value
    pub fn string(value: InternedString) -> Self {
        Self {
            tag: ValueTag::String,
            data: ValueData { string: value },
        }
    }
    
    /// Create a managed symbol value
    pub fn symbol(value: InternedString) -> Self {
        Self {
            tag: ValueTag::Symbol,
            data: ValueData { symbol: value },
        }
    }
    
    /// Create a managed boolean value
    pub fn boolean(value: bool) -> Self {
        Self {
            tag: ValueTag::Boolean,
            data: ValueData { boolean: value },
        }
    }
    
    /// Create a missing value
    pub fn missing() -> Self {
        Self {
            tag: ValueTag::Missing,
            data: ValueData { integer: 0 }, // Zero-initialized
        }
    }
    
    /// Convert from a regular Value to ManagedValue
    pub fn from_value(value: Value, interner: &crate::memory::StringInterner) -> VmResult<Self> {
        match value {
            Value::Integer(i) => Ok(Self::integer(i)),
            Value::Real(r) => Ok(Self::real(r)),
            Value::String(s) => Ok(Self::string(interner.intern(&s))),
            Value::Symbol(s) => Ok(Self::symbol(interner.intern(&s))),
            Value::Boolean(b) => Ok(Self::boolean(b)),
            Value::Missing => Ok(Self::missing()),
            Value::Function(f) => Ok(Self::symbol(interner.intern(&f))),
            _ => Err(VmError::TypeError {
                expected: "manageable value type".to_string(),
                actual: format!("{:?}", value),
            }),
        }
    }
    
    /// Convert back to a regular Value
    pub fn to_value(&self) -> VmResult<Value> {
        match self.tag {
            ValueTag::Integer => Ok(Value::Integer(unsafe { self.data.integer })),
            ValueTag::Real => Ok(Value::Real(unsafe { self.data.real })),
            ValueTag::String => Ok(Value::String(unsafe { self.data.string.as_str().to_string() })),
            ValueTag::Symbol => Ok(Value::Symbol(unsafe { self.data.symbol.as_str().to_string() })),
            ValueTag::Boolean => Ok(Value::Boolean(unsafe { self.data.boolean })),
            ValueTag::Missing => Ok(Value::Missing),
            _ => Err(VmError::TypeError {
                expected: "convertible value type".to_string(),
                actual: format!("ManagedValue with tag {:?}", self.tag),
            }),
        }
    }
    
    /// Get the memory size of this value
    pub fn memory_size(&self) -> usize {
        match self.tag {
            ValueTag::Integer | ValueTag::Real | ValueTag::Boolean | ValueTag::Missing => {
                std::mem::size_of::<Self>()
            }
            ValueTag::String | ValueTag::Symbol => {
                std::mem::size_of::<Self>() + unsafe { self.data.string.len() }
            }
            ValueTag::LyObj => {
                std::mem::size_of::<Self>() + 64 // Estimate for foreign object overhead
            }
            ValueTag::List => {
                std::mem::size_of::<Self>() + 128 // Estimate for list overhead
            }
            ValueTag::Quote | ValueTag::Pattern => {
                std::mem::size_of::<Self>() + 32 // Estimate for AST node overhead
            }
        }
    }
    
    /// Check if this value can be safely recycled
    pub fn is_recyclable(&self) -> bool {
        match self.tag {
            ValueTag::Integer | ValueTag::Real | ValueTag::Boolean | ValueTag::Missing => true,
            ValueTag::String | ValueTag::Symbol => unsafe { self.data.string.len() < 256 }, // Small strings only
            _ => false, // Complex types need careful handling
        }
    }
}

// Clone is now derived automatically

impl PartialEq for ManagedValue {
    fn eq(&self, other: &Self) -> bool {
        if self.tag != other.tag {
            return false;
        }
        
        match self.tag {
            ValueTag::Integer => unsafe { self.data.integer == other.data.integer },
            ValueTag::Real => unsafe { 
                // Handle NaN equality properly
                let a = self.data.real;
                let b = other.data.real;
                (a.is_nan() && b.is_nan()) || (a == b)
            },
            ValueTag::String | ValueTag::Symbol => unsafe {
                std::ptr::eq(self.data.string.as_str(), other.data.string.as_str())
            },
            ValueTag::Boolean => unsafe { self.data.boolean == other.data.boolean },
            ValueTag::Missing => true,
            _ => false, // Complex types need specialized comparison
        }
    }
}

impl fmt::Debug for ManagedValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.tag {
            ValueTag::Integer => write!(f, "ManagedValue::Integer({})", unsafe { self.data.integer }),
            ValueTag::Real => write!(f, "ManagedValue::Real({})", unsafe { self.data.real }),
            ValueTag::String => write!(f, "ManagedValue::String(\"{}\")", unsafe { self.data.string.as_str() }),
            ValueTag::Symbol => write!(f, "ManagedValue::Symbol({})", unsafe { self.data.symbol.as_str() }),
            ValueTag::Boolean => write!(f, "ManagedValue::Boolean({})", unsafe { self.data.boolean }),
            ValueTag::Missing => write!(f, "ManagedValue::Missing"),
            ValueTag::LyObj => write!(f, "ManagedValue::LyObj(<object>)"),
            ValueTag::List => write!(f, "ManagedValue::List(<list>)"),
            ValueTag::Quote => write!(f, "ManagedValue::Quote(<expr>)"),
            ValueTag::Pattern => write!(f, "ManagedValue::Pattern(<pattern>)"),
        }
    }
}

// Note: Removed MemoryManaged trait implementation due to Send+Sync requirements
// The trait uses raw pointers which aren't thread-safe. We'll use index-based approach instead.

/// Pool-allocated vector for managed values
pub struct ManagedVec<T> {
    data: Vec<T>,
    pool_id: usize,
}

impl<T> ManagedVec<T> {
    pub fn new(pool_id: usize) -> Self {
        Self {
            data: Vec::new(),
            pool_id,
        }
    }
    
    pub fn with_capacity(capacity: usize, pool_id: usize) -> Self {
        Self {
            data: Vec::with_capacity(capacity),
            pool_id,
        }
    }
    
    pub fn push(&mut self, value: T) {
        self.data.push(value);
    }
    
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.data.iter()
    }
    
    pub fn pool_id(&self) -> usize {
        self.pool_id
    }
}

impl<T> std::ops::Index<usize> for ManagedVec<T> {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::IndexMut<usize> for ManagedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::StringInterner;
    
    #[test]
    fn test_managed_value_size() {
        // ManagedValue should be smaller than Value
        assert!(std::mem::size_of::<ManagedValue>() <= 16);
        assert!(std::mem::size_of::<ManagedValue>() < std::mem::size_of::<Value>());
    }
    
    #[test]
    fn test_integer_managed_value() {
        let value = ManagedValue::integer(42);
        assert_eq!(value.tag, ValueTag::Integer);
        assert_eq!(unsafe { value.data.integer }, 42);
        
        let converted = value.to_value().unwrap();
        assert_eq!(converted, Value::Integer(42));
    }
    
    #[test]
    fn test_real_managed_value() {
        let value = ManagedValue::real(3.14159);
        assert_eq!(value.tag, ValueTag::Real);
        assert!((unsafe { value.data.real } - 3.14159).abs() < 1e-10);
        
        let converted = value.to_value().unwrap();
        assert_eq!(converted, Value::Real(3.14159));
    }
    
    #[test]
    fn test_string_managed_value() {
        let interner = StringInterner::new();
        let interned = interner.intern("hello");
        let value = ManagedValue::string(interned);
        
        assert_eq!(value.tag, ValueTag::String);
        assert_eq!(unsafe { value.data.string.as_str() }, "hello");
        
        let converted = value.to_value().unwrap();
        assert_eq!(converted, Value::String("hello".to_string()));
    }
    
    #[test]
    fn test_boolean_managed_value() {
        let value_true = ManagedValue::boolean(true);
        let value_false = ManagedValue::boolean(false);
        
        assert_eq!(value_true.tag, ValueTag::Boolean);
        assert_eq!(value_false.tag, ValueTag::Boolean);
        assert_eq!(unsafe { value_true.data.boolean }, true);
        assert_eq!(unsafe { value_false.data.boolean }, false);
    }
    
    #[test]
    fn test_missing_managed_value() {
        let value = ManagedValue::missing();
        assert_eq!(value.tag, ValueTag::Missing);
        
        let converted = value.to_value().unwrap();
        assert_eq!(converted, Value::Missing);
    }
    
    #[test]
    fn test_managed_value_equality() {
        let val1 = ManagedValue::integer(42);
        let val2 = ManagedValue::integer(42);
        let val3 = ManagedValue::integer(24);
        
        assert_eq!(val1, val2);
        assert_ne!(val1, val3);
        
        let interner = StringInterner::new();
        let str1 = ManagedValue::string(interner.intern("test"));
        let str2 = ManagedValue::string(interner.intern("test"));
        assert_eq!(str1, str2); // Same interned string
    }
    
    #[test]
    fn test_memory_size_calculation() {
        let int_val = ManagedValue::integer(42);
        let real_val = ManagedValue::real(3.14);
        
        assert_eq!(int_val.memory_size(), std::mem::size_of::<ManagedValue>());
        assert_eq!(real_val.memory_size(), std::mem::size_of::<ManagedValue>());
        
        let interner = StringInterner::new();
        let str_val = ManagedValue::string(interner.intern("hello"));
        assert!(str_val.memory_size() > std::mem::size_of::<ManagedValue>());
    }
    
    #[test]
    fn test_recyclable_check() {
        let int_val = ManagedValue::integer(42);
        let real_val = ManagedValue::real(3.14);
        let bool_val = ManagedValue::boolean(true);
        let missing_val = ManagedValue::missing();
        
        assert!(int_val.is_recyclable());
        assert!(real_val.is_recyclable());
        assert!(bool_val.is_recyclable());
        assert!(missing_val.is_recyclable());
        
        let interner = StringInterner::new();
        let short_str = ManagedValue::string(interner.intern("hi"));
        assert!(short_str.is_recyclable());
    }
    
    #[test]
    fn test_from_value_conversion() {
        let interner = StringInterner::new();
        
        let original = Value::Integer(123);
        let managed = ManagedValue::from_value(original.clone(), &interner).unwrap();
        let converted = managed.to_value().unwrap();
        assert_eq!(original, converted);
        
        let original = Value::String("test".to_string());
        let managed = ManagedValue::from_value(original, &interner).unwrap();
        let converted = managed.to_value().unwrap();
        assert_eq!(converted, Value::String("test".to_string()));
    }
    
    #[test]
    fn test_managed_vec() {
        let mut vec = ManagedVec::new(1);
        assert!(vec.is_empty());
        assert_eq!(vec.pool_id(), 1);
        
        vec.push(ManagedValue::integer(1));
        vec.push(ManagedValue::integer(2));
        
        assert_eq!(vec.len(), 2);
        assert_eq!(vec[0].tag, ValueTag::Integer);
        assert_eq!(unsafe { vec[0].data.integer }, 1);
    }
}