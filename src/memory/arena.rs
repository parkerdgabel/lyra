//! Arena allocation for temporary computations
//!
//! This module provides scoped arena allocation that allows efficient
//! allocation of temporary values during symbolic computation with
//! automatic cleanup when scopes end.

use std::collections::HashMap;
use parking_lot::RwLock;
use crate::memory::{ManagedValue, stats::ArenaStats};

/// Unique identifier for allocation scopes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScopeId(u64);

impl ScopeId {
    fn new(id: u64) -> Self {
        Self(id)
    }
    
    pub fn as_u64(&self) -> u64 {
        self.0
    }
}

/// Arena allocation scope for temporary values
pub struct ArenaScope {
    /// Unique identifier for this scope
    id: ScopeId,
    /// Values allocated in this scope
    allocated_values: Vec<ManagedValue>,
    /// Nested scopes within this scope
    child_scopes: Vec<ScopeId>,
    /// Parent scope (if any)
    parent: Option<ScopeId>,
    /// Memory usage tracking
    allocated_bytes: usize,
}

impl ArenaScope {
    fn new(id: ScopeId, parent: Option<ScopeId>) -> Self {
        Self {
            id,
            allocated_values: Vec::new(),
            child_scopes: Vec::new(),
            parent,
            allocated_bytes: 0,
        }
    }
    
    /// Allocate a value in this scope
    fn alloc(&mut self, value: ManagedValue) -> &ManagedValue {
        self.allocated_bytes += value.memory_size();
        self.allocated_values.push(value);
        self.allocated_values.last().unwrap()
    }
    
    /// Get memory usage for this scope
    fn memory_usage(&self) -> usize {
        self.allocated_bytes
    }
    
    /// Get number of allocated values
    fn value_count(&self) -> usize {
        self.allocated_values.len()
    }
    
    /// Check if this scope is empty
    fn is_empty(&self) -> bool {
        self.allocated_values.is_empty() && self.child_scopes.is_empty()
    }
}

/// Computation arena for scoped temporary allocation
/// 
/// Provides efficient allocation of temporary values with automatic
/// cleanup when computation scopes end. Particularly useful for
/// pattern matching, rule application, and expression evaluation.
pub struct ComputationArena {
    /// All active scopes
    scopes: RwLock<HashMap<ScopeId, ArenaScope>>,
    /// Current active scope
    current_scope: RwLock<Option<ScopeId>>,
    /// Next scope ID
    next_scope_id: RwLock<u64>,
    /// Statistics for monitoring
    stats: RwLock<ArenaStats>,
}

impl ComputationArena {
    /// Create a new computation arena
    pub fn new() -> Self {
        Self {
            scopes: RwLock::new(HashMap::new()),
            current_scope: RwLock::new(None),
            next_scope_id: RwLock::new(1),
            stats: RwLock::new(ArenaStats::default()),
        }
    }
    
    /// Create a new scope and make it active
    pub fn push_scope(&self) -> ScopeId {
        let mut next_id = self.next_scope_id.write();
        let scope_id = ScopeId::new(*next_id);
        *next_id += 1;
        drop(next_id);
        
        let parent = *self.current_scope.read();
        let scope = ArenaScope::new(scope_id, parent);
        
        // Add to parent's children if there is a parent
        if let Some(parent_id) = parent {
            let mut scopes = self.scopes.write();
            if let Some(parent_scope) = scopes.get_mut(&parent_id) {
                parent_scope.child_scopes.push(scope_id);
            }
        }
        
        // Register new scope
        {
            let mut scopes = self.scopes.write();
            scopes.insert(scope_id, scope);
        }
        
        // Make it current
        {
            let mut current = self.current_scope.write();
            *current = Some(scope_id);
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_scopes += 1;
            stats.active_scopes += 1;
            if stats.active_scopes > stats.peak_scopes {
                stats.peak_scopes = stats.active_scopes;
            }
        }
        
        scope_id
    }
    
    /// End a scope and clean up its allocations
    pub fn pop_scope(&self, scope_id: ScopeId) -> usize {
        let mut freed_bytes = 0;
        let parent_scope_id;
        
        // Remove scope and get its memory usage and parent
        {
            let mut scopes = self.scopes.write();
            if let Some(scope) = scopes.remove(&scope_id) {
                freed_bytes = scope.memory_usage();
                parent_scope_id = scope.parent;
                
                // Remove from parent's children
                if let Some(parent_id) = scope.parent {
                    if let Some(parent_scope) = scopes.get_mut(&parent_id) {
                        parent_scope.child_scopes.retain(|&id| id != scope_id);
                    }
                }
            } else {
                parent_scope_id = None;
            }
        }
        
        // Update current scope to parent
        {
            let mut current = self.current_scope.write();
            if *current == Some(scope_id) {
                *current = parent_scope_id;
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.active_scopes = stats.active_scopes.saturating_sub(1);
            stats.total_freed += freed_bytes;
        }
        
        freed_bytes
    }
    
    /// Allocate a value in the current scope
    pub fn alloc(&self, value: ManagedValue) -> Option<ScopeId> {
        let current_scope_id = *self.current_scope.read();
        
        if let Some(scope_id) = current_scope_id {
            let mut scopes = self.scopes.write();
            if let Some(scope) = scopes.get_mut(&scope_id) {
                // Get memory size before moving value
                let memory_size = value.memory_size();
                scope.alloc(value);
                
                // Update global statistics
                let mut stats = self.stats.write();
                stats.total_allocated += memory_size;
                
                return Some(scope_id);
            }
        }
        
        None
    }
    
    /// Execute a function within a temporary scope
    pub fn with_scope<T>(&self, f: impl FnOnce(ScopeId) -> T) -> T {
        let scope_id = self.push_scope();
        let result = f(scope_id);
        self.pop_scope(scope_id);
        result
    }
    
    /// Get current scope ID
    pub fn current_scope(&self) -> Option<ScopeId> {
        *self.current_scope.read()
    }
    
    /// Get total memory allocated in all scopes
    pub fn total_allocated(&self) -> usize {
        let scopes = self.scopes.read();
        scopes.values().map(|scope| scope.memory_usage()).sum()
    }
    
    /// Get memory usage for a specific scope
    pub fn scope_memory_usage(&self, scope_id: ScopeId) -> Option<usize> {
        let scopes = self.scopes.read();
        scopes.get(&scope_id).map(|scope| scope.memory_usage())
    }
    
    /// Get number of active scopes
    pub fn active_scope_count(&self) -> usize {
        self.scopes.read().len()
    }
    
    /// Collect unused scopes (garbage collection)
    pub fn collect_unused_scopes(&self) -> usize {
        let mut freed_bytes = 0;
        
        // Find empty scopes to clean up
        let scope_ids_to_remove: Vec<ScopeId> = {
            let scopes = self.scopes.read();
            scopes.values()
                .filter(|scope| scope.is_empty())
                .map(|scope| scope.id)
                .collect()
        };
        
        // Remove empty scopes
        for scope_id in scope_ids_to_remove {
            freed_bytes += self.pop_scope(scope_id);
        }
        
        // Update GC statistics
        {
            let mut stats = self.stats.write();
            stats.gc_cycles += 1;
            stats.last_gc_freed = freed_bytes;
        }
        
        freed_bytes
    }
    
    /// Get comprehensive arena statistics
    pub fn stats(&self) -> ArenaStats {
        self.stats.read().clone()
    }
    
    /// Clear all scopes (for testing/cleanup)
    pub fn clear(&self) {
        let scope_ids: Vec<ScopeId> = {
            let scopes = self.scopes.read();
            scopes.keys().copied().collect()
        };
        
        for scope_id in scope_ids {
            self.pop_scope(scope_id);
        }
        
        *self.current_scope.write() = None;
    }
    
    /// Get debug information about scope hierarchy
    pub fn debug_scope_tree(&self) -> String {
        let scopes = self.scopes.read();
        let mut output = String::from("Arena Scope Tree:\n");
        
        // Find root scopes (no parent)
        let root_scopes: Vec<&ArenaScope> = scopes.values()
            .filter(|scope| scope.parent.is_none())
            .collect();
        
        for root in root_scopes {
            self.debug_scope_recursive(root, &*scopes, &mut output, 0);
        }
        
        if output == "Arena Scope Tree:\n" {
            output.push_str("  (no active scopes)\n");
        }
        
        output
    }
    
    fn debug_scope_recursive(
        &self,
        scope: &ArenaScope,
        all_scopes: &HashMap<ScopeId, ArenaScope>,
        output: &mut String,
        depth: usize,
    ) {
        let indent = "  ".repeat(depth);
        output.push_str(&format!(
            "{}Scope {:?}: {} values, {} bytes\n",
            indent,
            scope.id,
            scope.value_count(),
            scope.memory_usage()
        ));
        
        // Recursively print children
        for &child_id in &scope.child_scopes {
            if let Some(child_scope) = all_scopes.get(&child_id) {
                self.debug_scope_recursive(child_scope, all_scopes, output, depth + 1);
            }
        }
    }
}

impl Default for ComputationArena {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII scope guard for automatic cleanup
pub struct ScopeGuard<'a> {
    arena: &'a ComputationArena,
    scope_id: ScopeId,
}

impl<'a> ScopeGuard<'a> {
    pub fn new(arena: &'a ComputationArena) -> Self {
        let scope_id = arena.push_scope();
        Self { arena, scope_id }
    }
    
    pub fn scope_id(&self) -> ScopeId {
        self.scope_id
    }
}

impl<'a> Drop for ScopeGuard<'a> {
    fn drop(&mut self) {
        self.arena.pop_scope(self.scope_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ManagedValue;
    
    #[test]
    fn test_arena_scope_creation() {
        let arena = ComputationArena::new();
        
        assert_eq!(arena.active_scope_count(), 0);
        assert!(arena.current_scope().is_none());
        
        let scope1 = arena.push_scope();
        assert_eq!(arena.active_scope_count(), 1);
        assert_eq!(arena.current_scope(), Some(scope1));
        
        let scope2 = arena.push_scope();
        assert_eq!(arena.active_scope_count(), 2);
        assert_eq!(arena.current_scope(), Some(scope2));
        
        arena.pop_scope(scope2);
        assert_eq!(arena.active_scope_count(), 1);
        assert_eq!(arena.current_scope(), Some(scope1));
        
        arena.pop_scope(scope1);
        assert_eq!(arena.active_scope_count(), 0);
    }
    
    #[test]
    fn test_arena_allocation() {
        let arena = ComputationArena::new();
        
        let initial_allocated = arena.total_allocated();
        
        let _scope_guard = arena.with_scope(|_scope_id| {
            let value = ManagedValue::integer(42);
            arena.alloc(value);
            
            let allocated_after = arena.total_allocated();
            assert!(allocated_after > initial_allocated);
        });
        
        // After scope ends, memory should be cleaned up
        let final_allocated = arena.total_allocated();
        assert_eq!(final_allocated, initial_allocated);
    }
    
    #[test]
    fn test_nested_scopes() {
        let arena = ComputationArena::new();
        
        let scope1 = arena.push_scope();
        arena.alloc(ManagedValue::integer(1));
        
        let scope2 = arena.push_scope();
        arena.alloc(ManagedValue::integer(2));
        
        let scope3 = arena.push_scope();
        arena.alloc(ManagedValue::integer(3));
        
        assert_eq!(arena.active_scope_count(), 3);
        
        // Pop in reverse order
        arena.pop_scope(scope3);
        assert_eq!(arena.active_scope_count(), 2);
        assert_eq!(arena.current_scope(), Some(scope2));
        
        arena.pop_scope(scope2);
        assert_eq!(arena.active_scope_count(), 1);
        assert_eq!(arena.current_scope(), Some(scope1));
        
        arena.pop_scope(scope1);
        assert_eq!(arena.active_scope_count(), 0);
    }
    
    #[test]
    fn test_with_scope_helper() {
        let arena = ComputationArena::new();
        
        let result = arena.with_scope(|scope_id| {
            assert!(arena.current_scope() == Some(scope_id));
            arena.alloc(ManagedValue::real(3.14));
            42
        });
        
        assert_eq!(result, 42);
        assert_eq!(arena.active_scope_count(), 0);
        assert_eq!(arena.total_allocated(), 0);
    }
    
    #[test]
    fn test_scope_guard() {
        let arena = ComputationArena::new();
        
        {
            let _guard = ScopeGuard::new(&arena);
            assert_eq!(arena.active_scope_count(), 1);
            
            arena.alloc(ManagedValue::integer(123));
            assert!(arena.total_allocated() > 0);
        } // Guard drops here
        
        assert_eq!(arena.active_scope_count(), 0);
        assert_eq!(arena.total_allocated(), 0);
    }
    
    #[test]
    fn test_memory_tracking() {
        let arena = ComputationArena::new();
        
        let scope1 = arena.push_scope();
        arena.alloc(ManagedValue::integer(42));
        let usage1 = arena.scope_memory_usage(scope1).unwrap();
        assert!(usage1 > 0);
        
        arena.alloc(ManagedValue::real(3.14));
        let usage2 = arena.scope_memory_usage(scope1).unwrap();
        assert!(usage2 > usage1);
        
        arena.pop_scope(scope1);
        assert!(arena.scope_memory_usage(scope1).is_none());
    }
    
    #[test]
    fn test_garbage_collection() {
        let arena = ComputationArena::new();
        
        // Create and immediately clean up some scopes
        for _ in 0..10 {
            arena.with_scope(|_| {
                // Empty scope - should be collectable
            });
        }
        
        let freed = arena.collect_unused_scopes();
        assert_eq!(freed, 0); // Empty scopes have no memory to free
        
        let stats = arena.stats();
        assert_eq!(stats.gc_cycles, 1);
    }
    
    #[test]
    fn test_statistics() {
        let arena = ComputationArena::new();
        
        let initial_stats = arena.stats();
        assert_eq!(initial_stats.total_scopes, 0);
        assert_eq!(initial_stats.active_scopes, 0);
        
        arena.with_scope(|_| {
            arena.alloc(ManagedValue::integer(42));
            arena.alloc(ManagedValue::real(3.14));
        });
        
        let final_stats = arena.stats();
        assert_eq!(final_stats.total_scopes, 1);
        assert_eq!(final_stats.active_scopes, 0); // Cleaned up after scope
        assert!(final_stats.total_allocated > 0);
        assert!(final_stats.total_freed > 0);
        assert!(final_stats.efficiency() > 0.0);
    }
    
    #[test]
    fn test_debug_scope_tree() {
        let arena = ComputationArena::new();
        
        let empty_tree = arena.debug_scope_tree();
        assert!(empty_tree.contains("(no active scopes)"));
        
        let scope1 = arena.push_scope();
        arena.alloc(ManagedValue::integer(1));
        
        let scope2 = arena.push_scope();
        arena.alloc(ManagedValue::integer(2));
        
        let tree_with_scopes = arena.debug_scope_tree();
        assert!(tree_with_scopes.contains("Scope"));
        assert!(tree_with_scopes.contains("values"));
        assert!(tree_with_scopes.contains("bytes"));
        
        println!("Debug tree:\n{}", tree_with_scopes);
        
        arena.pop_scope(scope2);
        arena.pop_scope(scope1);
    }
    
    #[test]
    fn test_concurrent_scope_access() {
        use std::sync::Arc;
        use std::thread;
        
        let arena = Arc::new(ComputationArena::new());
        let arena_clone = arena.clone();
        
        let handle = thread::spawn(move || {
            arena_clone.with_scope(|_| {
                arena_clone.alloc(ManagedValue::integer(42));
            });
        });
        
        arena.with_scope(|_| {
            arena.alloc(ManagedValue::real(3.14));
        });
        
        handle.join().unwrap();
        
        // Both threads should have completed successfully
        assert_eq!(arena.active_scope_count(), 0);
    }
}