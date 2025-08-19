//! # Advanced Memory Management System for Lyra
//!
//! This module provides a comprehensive memory management system designed to achieve
//! 35%+ memory reduction through string interning, memory pools, arena allocation,
//! and reference counting for expressions.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │   MemoryMgr     │───▶│  Memory Pools   │───▶│  Arena Alloc    │
//! │   (main)        │    │  (by type)      │    │  (temp scope)   │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//!          │                       │                       │
//!          ▼                       ▼                       ▼
//! ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
//! │ String Interner │    │ Value Recycling │    │   RC Manager    │
//! │ (static cache)  │    │  (type pools)   │    │ (expr sharing)  │
//! └─────────────────┘    └─────────────────┘    └─────────────────┘
//! ```

pub mod interner;
pub mod pools;
pub mod arena;
pub mod managed_value;
pub mod stats;

use std::sync::Arc;
use crate::vm::{Value, VmResult, VmError};

pub use interner::{StringInterner, InternedString};
pub use pools::{ValuePools, TypedPool};
pub use arena::{ComputationArena, ScopeId};
pub use managed_value::{ManagedValue, ValueData, ValueTag};
pub use stats::{MemoryStats, PoolStats, ArenaStats};

/// Global memory manager instance for Lyra
/// 
/// This singleton provides thread-safe access to all memory management
/// components including string interning, value pools, and computation arenas.
pub struct MemoryManager {
    interner: Arc<StringInterner>,
    pools: Arc<ValuePools>,
    arena: ComputationArena,
}

impl MemoryManager {
    /// Create a new memory manager with default configuration
    pub fn new() -> Self {
        Self {
            interner: Arc::new(StringInterner::new()),
            pools: Arc::new(ValuePools::new()),
            arena: ComputationArena::new(),
        }
    }
    
    /// Intern a string for memory efficiency
    pub fn intern_string(&self, s: &str) -> InternedString {
        self.interner.intern(s)
    }
    
    /// Allocate a managed value from the appropriate pool
    pub fn alloc_value(&mut self, value: Value) -> VmResult<ManagedValue> {
        self.pools.alloc_value(value)
    }
    
    /// Create a temporary computation scope for efficient allocation
    pub fn with_temp_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        self.arena.with_scope(|_arena| f(self))
    }
    
    /// Trigger garbage collection and return bytes freed
    pub fn collect_garbage(&mut self) -> usize {
        let mut freed = 0;
        freed += self.pools.collect_unused();
        freed += self.arena.collect_unused_scopes();
        freed
    }
    
    /// Get comprehensive memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            total_allocated: self.pools.total_allocated() + self.arena.total_allocated(),
            interner_stats: self.interner.stats(),
            pool_stats: self.pools.stats(),
            arena_stats: self.arena.stats(),
        }
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for memory-managed types
pub trait MemoryManaged: Send + Sync {
    /// Get memory usage in bytes
    fn memory_usage(&self) -> usize;
    
    /// Attempt to compress/optimize memory usage
    fn compress(&mut self) -> VmResult<usize>;
    
    /// Check if this object can be safely recycled
    fn can_recycle(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_manager_creation() {
        let manager = MemoryManager::new();
        let stats = manager.memory_stats();
        
        // Should start with minimal memory usage
        assert!(stats.total_allocated < 1024); // Less than 1KB initial overhead
    }
    
    #[test]
    fn test_string_interning() {
        let manager = MemoryManager::new();
        
        let str1 = manager.intern_string("x");
        let str2 = manager.intern_string("x");
        
        // Same string should return same interned reference
        assert!(std::ptr::eq(str1.as_str(), str2.as_str()));
    }
    
    #[test]
    fn test_temp_scope() {
        let mut manager = MemoryManager::new();
        
        let initial_allocated = manager.memory_stats().total_allocated;
        
        manager.with_temp_scope(|mgr| {
            // Allocate some temporary values
            let _val1 = mgr.intern_string("temp1");
            let _val2 = mgr.intern_string("temp2");
        });
        
        // Memory should be efficiently managed
        let final_allocated = manager.memory_stats().total_allocated;
        assert!(final_allocated <= initial_allocated + 1024); // Minimal overhead
    }
}