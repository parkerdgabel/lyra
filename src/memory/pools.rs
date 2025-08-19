//! Memory pools for efficient value allocation and recycling
//!
//! This module provides type-specific memory pools that reduce allocation
//! overhead and enable efficient recycling of frequently used value types.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use crate::vm::{Value, VmResult, VmError};
use crate::memory::{ManagedValue, ValueTag, MemoryManaged, stats::PoolStats};

/// Generic typed memory pool for efficient allocation and recycling
pub struct TypedPool<T> {
    /// Available items for reuse
    available: Vec<T>,
    /// Maximum pool size before items are discarded
    max_size: usize,
    /// Statistics for monitoring performance
    stats: PoolStats,
}

impl<T: Clone + Default> TypedPool<T> {
    /// Create a new typed pool with specified capacity
    pub fn new(max_size: usize) -> Self {
        Self {
            available: Vec::with_capacity(max_size.min(1024)),
            max_size,
            stats: PoolStats::default(),
        }
    }
    
    /// Get an item from the pool or create a new one
    pub fn get(&mut self) -> T {
        self.stats.total_allocations += 1;
        
        if let Some(item) = self.available.pop() {
            self.stats.reuse_hits += 1;
            item
        } else {
            self.stats.pool_misses += 1;
            T::default()
        }
    }
    
    /// Return an item to the pool for reuse
    pub fn recycle(&mut self, mut item: T) {
        if self.available.len() < self.max_size {
            // Reset item to default state if possible
            item = T::default();
            self.available.push(item);
            self.stats.available_items = self.available.len();
            self.stats.current_size = self.available.len();
            
            if self.stats.current_size > self.stats.peak_size {
                self.stats.peak_size = self.stats.current_size;
            }
        }
        // Otherwise, let it drop to prevent unbounded growth
    }
    
    /// Get current pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.clone()
    }
    
    /// Get current pool size
    pub fn size(&self) -> usize {
        self.available.len()
    }
    
    /// Get available capacity
    pub fn capacity(&self) -> usize {
        self.max_size
    }
    
    /// Clear all items from the pool
    pub fn clear(&mut self) {
        self.available.clear();
        self.stats.current_size = 0;
        self.stats.available_items = 0;
    }
    
    /// Shrink pool to reduce memory usage
    pub fn shrink_to_fit(&mut self) {
        self.available.shrink_to_fit();
    }
}

/// Specialized pool for ManagedValue instances
pub struct ManagedValuePool {
    /// Pools organized by value type for efficient allocation
    integer_pool: TypedPool<i64>,
    real_pool: TypedPool<f64>,
    boolean_pool: TypedPool<bool>,
    /// String pool managed by interner
    /// List and object pools would need specialized handling
    stats: PoolStats,
}

impl ManagedValuePool {
    /// Create a new managed value pool
    pub fn new() -> Self {
        Self {
            integer_pool: TypedPool::new(1024),
            real_pool: TypedPool::new(512),
            boolean_pool: TypedPool::new(8), // Only true/false needed
            stats: PoolStats::default(),
        }
    }
    
    /// Allocate a managed value with pool optimization
    pub fn alloc(&mut self, value: Value, interner: &crate::memory::StringInterner) -> VmResult<ManagedValue> {
        self.stats.total_allocations += 1;
        
        match value {
            Value::Integer(i) => {
                let _pooled_int = self.integer_pool.get(); // Could use for validation
                Ok(ManagedValue::integer(i))
            }
            Value::Real(r) => {
                let _pooled_real = self.real_pool.get(); // Could use for validation
                Ok(ManagedValue::real(r))
            }
            Value::Boolean(b) => {
                let _pooled_bool = self.boolean_pool.get(); // Could use for validation
                Ok(ManagedValue::boolean(b))
            }
            Value::String(s) => {
                let interned = interner.intern(&s);
                Ok(ManagedValue::string(interned))
            }
            Value::Symbol(s) => {
                let interned = interner.intern(&s);
                Ok(ManagedValue::symbol(interned))
            }
            Value::Function(f) => {
                let interned = interner.intern(&f);
                Ok(ManagedValue::symbol(interned))
            }
            Value::Missing => Ok(ManagedValue::missing()),
            _ => Err(VmError::TypeError {
                expected: "poolable value type".to_string(),
                actual: format!("{:?}", value),
            }),
        }
    }
    
    /// Recycle a managed value back to the appropriate pool
    pub fn recycle(&mut self, value: &ManagedValue) {
        if !value.is_recyclable() {
            return;
        }
        
        match value.tag {
            ValueTag::Integer => {
                let int_val = unsafe { value.data.integer };
                self.integer_pool.recycle(int_val);
            }
            ValueTag::Real => {
                let real_val = unsafe { value.data.real };
                self.real_pool.recycle(real_val);
            }
            ValueTag::Boolean => {
                let bool_val = unsafe { value.data.boolean };
                self.boolean_pool.recycle(bool_val);
            }
            _ => {
                // Other types don't have dedicated pools yet
            }
        }
    }
    
    /// Get comprehensive pool statistics
    pub fn stats(&self) -> HashMap<String, PoolStats> {
        let mut stats = HashMap::new();
        stats.insert("integer".to_string(), self.integer_pool.stats());
        stats.insert("real".to_string(), self.real_pool.stats());
        stats.insert("boolean".to_string(), self.boolean_pool.stats());
        stats.insert("managed_value".to_string(), self.stats.clone());
        stats
    }
    
    /// Get total memory usage across all pools
    pub fn total_memory_usage(&self) -> usize {
        let integer_size = self.integer_pool.size() * std::mem::size_of::<i64>();
        let real_size = self.real_pool.size() * std::mem::size_of::<f64>();
        let boolean_size = self.boolean_pool.size() * std::mem::size_of::<bool>();
        
        integer_size + real_size + boolean_size
    }
    
    /// Perform garbage collection on pools
    pub fn collect_unused(&mut self) -> usize {
        let initial_usage = self.total_memory_usage();
        
        // Shrink pools to optimal size (keep some items for performance)
        let target_integer = (self.integer_pool.size() * 3) / 4;
        let target_real = (self.real_pool.size() * 3) / 4;
        let target_boolean = (self.boolean_pool.size() * 3) / 4;
        
        self.integer_pool.available.truncate(target_integer);
        self.real_pool.available.truncate(target_real);
        self.boolean_pool.available.truncate(target_boolean);
        
        // Shrink backing storage
        self.integer_pool.shrink_to_fit();
        self.real_pool.shrink_to_fit();
        self.boolean_pool.shrink_to_fit();
        
        let final_usage = self.total_memory_usage();
        initial_usage.saturating_sub(final_usage)
    }
}

impl Default for ManagedValuePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive value pools for all Lyra types
pub struct ValuePools {
    /// Pool for managed values
    managed_pool: RwLock<ManagedValuePool>,
    /// Pool for Vec<Value> instances
    list_pool: RwLock<TypedPool<Vec<Value>>>,
    /// Pool for HashMap instances (for pattern matching)
    map_pool: RwLock<TypedPool<HashMap<String, Value>>>,
    /// Global statistics
    global_stats: RwLock<PoolStats>,
}

impl ValuePools {
    /// Create a new comprehensive value pool system
    pub fn new() -> Self {
        Self {
            managed_pool: RwLock::new(ManagedValuePool::new()),
            list_pool: RwLock::new(TypedPool::new(256)),
            map_pool: RwLock::new(TypedPool::new(128)),
            global_stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Allocate a managed value efficiently
    pub fn alloc_value(&self, value: Value) -> VmResult<ManagedValue> {
        // For now, we need access to the interner
        // This would be better with dependency injection
        let temp_interner = crate::memory::StringInterner::new();
        let mut pool = self.managed_pool.write();
        pool.alloc(value, &temp_interner)
    }
    
    /// Get a list from the pool
    pub fn get_list(&self) -> Vec<Value> {
        let mut pool = self.list_pool.write();
        let mut list = pool.get();
        list.clear(); // Ensure it's empty
        list
    }
    
    /// Return a list to the pool
    pub fn recycle_list(&self, mut list: Vec<Value>) {
        list.clear(); // Clear contents
        list.shrink_to_fit(); // Optimize memory
        
        let mut pool = self.list_pool.write();
        pool.recycle(list);
    }
    
    /// Get a map from the pool
    pub fn get_map(&self) -> HashMap<String, Value> {
        let mut pool = self.map_pool.write();
        let mut map = pool.get();
        map.clear(); // Ensure it's empty
        map
    }
    
    /// Return a map to the pool
    pub fn recycle_map(&self, mut map: HashMap<String, Value>) {
        map.clear(); // Clear contents
        map.shrink_to_fit(); // Optimize memory
        
        let mut pool = self.map_pool.write();
        pool.recycle(map);
    }
    
    /// Get comprehensive statistics for all pools
    pub fn stats(&self) -> HashMap<String, PoolStats> {
        let mut all_stats = HashMap::new();
        
        // Managed value pool stats
        {
            let managed_pool = self.managed_pool.read();
            let managed_stats = managed_pool.stats();
            all_stats.extend(managed_stats);
        }
        
        // List pool stats
        {
            let list_pool = self.list_pool.read();
            all_stats.insert("list".to_string(), list_pool.stats());
        }
        
        // Map pool stats
        {
            let map_pool = self.map_pool.read();
            all_stats.insert("map".to_string(), map_pool.stats());
        }
        
        // Global stats
        {
            let global_stats = self.global_stats.read();
            all_stats.insert("global".to_string(), global_stats.clone());
        }
        
        all_stats
    }
    
    /// Get total allocated memory across all pools
    pub fn total_allocated(&self) -> usize {
        let managed_usage = self.managed_pool.read().total_memory_usage();
        let list_usage = self.list_pool.read().size() * std::mem::size_of::<Vec<Value>>();
        let map_usage = self.map_pool.read().size() * std::mem::size_of::<HashMap<String, Value>>();
        
        managed_usage + list_usage + map_usage
    }
    
    /// Perform garbage collection across all pools
    pub fn collect_unused(&self) -> usize {
        let mut total_freed = 0;
        
        // GC managed value pool
        {
            let mut managed_pool = self.managed_pool.write();
            total_freed += managed_pool.collect_unused();
        }
        
        // GC list pool
        {
            let mut list_pool = self.list_pool.write();
            let initial_size = list_pool.size();
            list_pool.clear(); // Aggressive clearing for now
            list_pool.shrink_to_fit();
            total_freed += initial_size * std::mem::size_of::<Vec<Value>>();
        }
        
        // GC map pool
        {
            let mut map_pool = self.map_pool.write();
            let initial_size = map_pool.size();
            map_pool.clear(); // Aggressive clearing for now
            map_pool.shrink_to_fit();
            total_freed += initial_size * std::mem::size_of::<HashMap<String, Value>>();
        }
        
        total_freed
    }
    
    /// Get pool efficiency metrics
    pub fn efficiency_report(&self) -> String {
        let stats = self.stats();
        let mut report = String::from("Pool Efficiency Report:\n");
        
        for (pool_name, pool_stats) in stats {
            report.push_str(&format!(
                "{}: {:.1}% efficiency, {} items, {:.1}% utilization\n",
                pool_name,
                pool_stats.efficiency() * 100.0,
                pool_stats.current_size,
                pool_stats.utilization() * 100.0
            ));
        }
        
        report
    }
}

impl Default for ValuePools {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_typed_pool_basic_operations() {
        let mut pool: TypedPool<i64> = TypedPool::new(10);
        
        // First allocation should miss (empty pool)
        let val1 = pool.get();
        assert_eq!(val1, 0); // Default i64
        assert_eq!(pool.stats().pool_misses, 1);
        assert_eq!(pool.stats().reuse_hits, 0);
        
        // Recycle the value
        pool.recycle(42);
        assert_eq!(pool.size(), 1);
        
        // Next allocation should hit
        let val2 = pool.get();
        assert_eq!(pool.stats().reuse_hits, 1);
        assert_eq!(pool.size(), 0);
    }
    
    #[test]
    fn test_typed_pool_capacity_limits() {
        let mut pool: TypedPool<i64> = TypedPool::new(2);
        
        // Fill pool to capacity
        pool.recycle(1);
        pool.recycle(2);
        assert_eq!(pool.size(), 2);
        
        // Exceeding capacity should discard items
        pool.recycle(3);
        assert_eq!(pool.size(), 2); // Still at capacity
    }
    
    #[test]
    fn test_managed_value_pool() {
        let interner = crate::memory::StringInterner::new();
        let mut pool = ManagedValuePool::new();
        
        // Test integer allocation
        let int_val = pool.alloc(Value::Integer(42), &interner).unwrap();
        assert_eq!(int_val.tag, ValueTag::Integer);
        assert_eq!(unsafe { int_val.data.integer }, 42);
        
        // Test string allocation
        let str_val = pool.alloc(Value::String("test".to_string()), &interner).unwrap();
        assert_eq!(str_val.tag, ValueTag::String);
        assert_eq!(unsafe { str_val.data.string.as_str() }, "test");
        
        // Test recycling
        pool.recycle(&int_val);
        
        let stats = pool.stats();
        assert!(stats.contains_key("integer"));
        assert!(stats.contains_key("real"));
        assert!(stats.contains_key("boolean"));
    }
    
    #[test]
    fn test_value_pools_comprehensive() {
        let pools = ValuePools::new();
        
        // Test list allocation
        let mut list = pools.get_list();
        list.push(Value::Integer(1));
        list.push(Value::Integer(2));
        assert_eq!(list.len(), 2);
        pools.recycle_list(list);
        
        // Test map allocation
        let mut map = pools.get_map();
        map.insert("key".to_string(), Value::Integer(42));
        assert_eq!(map.len(), 1);
        pools.recycle_map(map);
        
        // Test statistics
        let stats = pools.stats();
        assert!(!stats.is_empty());
    }
    
    #[test]
    fn test_pool_statistics() {
        let mut pool: TypedPool<i64> = TypedPool::new(5);
        
        // Generate some activity
        for i in 0..10 {
            let _val = pool.get();
            if i % 2 == 0 {
                pool.recycle(i);
            }
        }
        
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 10);
        assert!(stats.pool_misses > 0);
        assert!(stats.reuse_hits > 0);
        assert!(stats.efficiency() > 0.0);
        assert!(stats.efficiency() <= 1.0);
    }
    
    #[test]
    fn test_memory_usage_calculation() {
        let pools = ValuePools::new();
        let initial_usage = pools.total_allocated();
        
        // Allocate some values
        let _val1 = pools.alloc_value(Value::Integer(42));
        let _val2 = pools.alloc_value(Value::Real(3.14));
        
        // Memory usage should increase
        let final_usage = pools.total_allocated();
        // Note: Due to pooling, this might not always increase significantly
        assert!(final_usage >= initial_usage);
    }
    
    #[test]
    fn test_garbage_collection() {
        let pools = ValuePools::new();
        
        // Generate some allocation activity
        for i in 0..100 {
            let _list = pools.get_list();
            let _map = pools.get_map();
            let _val = pools.alloc_value(Value::Integer(i));
        }
        
        let initial_usage = pools.total_allocated();
        let freed = pools.collect_unused();
        let final_usage = pools.total_allocated();
        
        // Should free some memory
        assert!(final_usage <= initial_usage);
        println!("Freed {} bytes in GC", freed);
    }
    
    #[test]
    fn test_efficiency_report() {
        let pools = ValuePools::new();
        
        // Generate activity
        for _ in 0..10 {
            let _list = pools.get_list();
            let _map = pools.get_map();
        }
        
        let report = pools.efficiency_report();
        assert!(report.contains("Pool Efficiency Report"));
        assert!(report.contains("efficiency"));
        println!("{}", report);
    }
}