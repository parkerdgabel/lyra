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
pub mod compact_pools;
pub mod arena;
pub mod managed_value;
pub mod optimized_value;
pub mod validation;
pub mod stats;

use std::sync::Arc;
use crate::vm::{Value, VmResult};

pub use interner::{StringInterner, InternedString, SymbolId};
pub use pools::{ValuePools, TypedPool};
pub use compact_pools::CompactValuePools;
pub use arena::{ComputationArena, ScopeId};
pub use managed_value::{ManagedValue, ValueData, ValueTag};
pub use optimized_value::{CompactValue, CacheAlignedValue, SerializableValue};
pub use validation::{validate_memory_optimizations, performance_comparison, MemoryValidationReport};
pub use stats::{MemoryStats, PoolStats, ArenaStats};

/// Global memory manager instance for Lyra
/// 
/// This singleton provides thread-safe access to all memory management
/// components including string interning, value pools, and computation arenas.
pub struct MemoryManager {
    interner: Arc<StringInterner>,
    pools: Arc<ValuePools>,
    compact_pools: Arc<CompactValuePools>,
    arena: ComputationArena,
}

impl MemoryManager {
    /// Create a new memory manager with default configuration
    pub fn new() -> Self {
        Self {
            interner: Arc::new(StringInterner::new()),
            pools: Arc::new(ValuePools::new()),
            compact_pools: Arc::new(CompactValuePools::new()),
            arena: ComputationArena::new(),
        }
    }
    
    /// Intern a string for memory efficiency (legacy method)
    pub fn intern_string(&self, s: &str) -> InternedString {
        self.interner.intern(s)
    }
    
    /// NEW: Intern a symbol and return compact ID (4 bytes vs 16+ for String)
    pub fn intern_symbol(&self, s: &str) -> SymbolId {
        self.interner.intern_symbol_id(s)
    }
    
    /// NEW: Resolve a symbol ID back to string
    pub fn resolve_symbol(&self, id: SymbolId) -> Option<String> {
        self.interner.resolve_symbol(id)
    }
    
    /// Allocate a managed value from the appropriate pool (legacy)
    pub fn alloc_value(&mut self, value: Value) -> VmResult<ManagedValue> {
        self.pools.alloc_value(value)
    }
    
    /// NEW: Convert regular Value to optimized CompactValue
    pub fn compact_value(&self, value: Value) -> CompactValue {
        CompactValue::from_value(value, &self.interner)
    }
    
    /// NEW: Allocate a CompactValue using optimized pools
    pub fn alloc_compact_value(&self, value: CompactValue) -> CompactValue {
        self.compact_pools.alloc_value(value)
    }
    
    /// NEW: Recycle a CompactValue for memory efficiency
    pub fn recycle_compact_value(&self, value: &CompactValue) {
        self.compact_pools.recycle_value(value)
    }
    
    /// Create a temporary computation scope for efficient allocation
    pub fn with_temp_scope<T>(&mut self, f: impl FnOnce(&mut Self) -> T) -> T {
        // TODO: Fix borrow checker issue with arena scope
        f(self) // Temporary workaround
    }
    
    /// Trigger garbage collection and return bytes freed
    pub fn collect_garbage(&mut self) -> usize {
        let mut freed = 0;
        freed += self.pools.collect_unused();
        freed += self.compact_pools.collect_unused();
        freed += self.arena.collect_unused_scopes();
        freed
    }
    
    /// Get comprehensive memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        let mut combined_pool_stats = self.pools.stats();
        let compact_stats = self.compact_pools.stats();
        
        // Merge compact pool stats
        for (key, stats) in compact_stats {
            combined_pool_stats.insert(format!("compact_{}", key), stats);
        }
        
        MemoryStats {
            total_allocated: self.pools.total_allocated() 
                + self.compact_pools.total_memory_usage() 
                + self.arena.total_allocated()
                + self.interner.memory_usage(),
            interner_stats: self.interner.stats(),
            pool_stats: combined_pool_stats,
            arena_stats: self.arena.stats(),
        }
    }
    
    /// NEW: Generate detailed efficiency report
    pub fn efficiency_report(&self) -> String {
        let mut report = String::new();
        
        report.push_str("=== Lyra Memory Management Efficiency Report ===\n\n");
        
        // Overall statistics
        let stats = self.memory_stats();
        report.push_str(&stats.format_summary());
        report.push_str("\n");
        
        // Compact value pools efficiency
        report.push_str("=== CompactValue Pools ===\n");
        report.push_str(&self.compact_pools.efficiency_report());
        report.push_str("\n");
        
        // String interning efficiency
        let interner_stats = self.interner.stats();
        report.push_str(&format!(
            "=== String Interning ===\n\
             Symbol count: {}\n\
             Memory usage: {:.2} KB\n\
             Hit ratio: {:.1}%\n\
             Average length: {:.1} chars\n\n",
            self.interner.symbol_count(),
            self.interner.memory_usage() as f64 / 1024.0,
            interner_stats.hit_ratio() * 100.0,
            interner_stats.average_string_length()
        ));
        
        report
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
        
        println!("Total allocated: {}", stats.total_allocated);
        
        // Should start with reasonable memory usage (very generous limit for now)
        assert!(stats.total_allocated < 1024 * 1024); // Less than 1MB initial overhead
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