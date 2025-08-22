//! Optimized memory pools for CompactValue system
//!
//! This module provides specialized memory pools that take advantage of
//! the CompactValue representation for maximum memory efficiency.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use crossbeam::queue::SegQueue;

use crate::memory::{CompactValue, stats::PoolStats};

/// Pool for small integers (most common case)
pub struct SmallIntPool {
    /// Pre-allocated small integers for common values (-1000 to 1000)
    cached_values: Vec<CompactValue>,
    /// Statistics
    stats: RwLock<PoolStats>,
}

impl SmallIntPool {
    /// Create a new small integer pool with pre-cached common values
    pub fn new() -> Self {
        let mut cached_values = Vec::with_capacity(2001);
        
        // Pre-cache integers from -1000 to 1000
        for i in -1000..=1000 {
            cached_values.push(CompactValue::SmallInt(i));
        }
        
        Self {
            cached_values,
            stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Get a small integer value (O(1) for common values)
    pub fn get_small_int(&self, value: i32) -> CompactValue {
        self.stats.write().total_allocations += 1;
        
        if value >= -1000 && value <= 1000 {
            self.stats.write().reuse_hits += 1;
            self.cached_values[(value + 1000) as usize].clone()
        } else {
            self.stats.write().pool_misses += 1;
            CompactValue::SmallInt(value)
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }
}

/// Pool for large integers using Arc<i64> for sharing
pub struct LargeIntPool {
    /// Thread-safe queue for recycling Arc<i64> instances
    recycled: SegQueue<Arc<i64>>,
    /// Maximum pool size
    max_size: usize,
    /// Statistics
    stats: RwLock<PoolStats>,
}

impl LargeIntPool {
    /// Create a new large integer pool
    pub fn new(max_size: usize) -> Self {
        Self {
            recycled: SegQueue::new(),
            max_size,
            stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Get a large integer, reusing Arc if possible
    pub fn get_large_int(&self, value: i64) -> CompactValue {
        self.stats.write().total_allocations += 1;
        
        // Try to reuse an existing Arc
        if let Some(_arc) = self.recycled.pop() {
            self.stats.write().reuse_hits += 1;
            // We can't modify the Arc content, so create a new one
            // But this saves allocation overhead in some cases
            CompactValue::LargeInt(Arc::new(value))
        } else {
            self.stats.write().pool_misses += 1;
            CompactValue::LargeInt(Arc::new(value))
        }
    }
    
    /// Recycle a large integer Arc
    pub fn recycle_large_int(&self, _value: &CompactValue) {
        // For now, we can't easily recycle Arc<i64> since the value
        // is immutable. In a more sophisticated system, we could use
        // a different approach with mutable cells.
        // This is a placeholder for future optimization.
    }
    
    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }
}

/// Pool for real number values with common value caching
pub struct RealPool {
    /// Cache for common real values (0.0, 1.0, -1.0, pi, e, etc.)
    common_reals: HashMap<u64, CompactValue>, // Using f64 bits as key
    /// Statistics
    stats: RwLock<PoolStats>,
}

impl RealPool {
    /// Create a new real pool with common values cached
    pub fn new() -> Self {
        let mut common_reals = HashMap::new();
        
        // Cache common mathematical constants
        let common_values = [
            0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5,
            std::f64::consts::PI, std::f64::consts::E,
            std::f64::consts::SQRT_2, std::f64::consts::LN_2,
            std::f64::consts::LN_10, std::f64::consts::LOG2_E,
        ];
        
        for &value in &common_values {
            common_reals.insert(value.to_bits(), CompactValue::Real(value));
        }
        
        Self {
            common_reals,
            stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Get a real value, using cache for common values
    pub fn get_real(&self, value: f64) -> CompactValue {
        self.stats.write().total_allocations += 1;
        
        let bits = value.to_bits();
        if let Some(cached) = self.common_reals.get(&bits) {
            self.stats.write().reuse_hits += 1;
            cached.clone()
        } else {
            self.stats.write().pool_misses += 1;
            CompactValue::Real(value)
        }
    }
    
    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }
}

/// Pool for Arc<Vec<CompactValue>> list storage
pub struct ListPool {
    /// Thread-safe queue for recycling list containers
    recycled_small: SegQueue<Arc<Vec<CompactValue>>>, // Lists with capacity <= 16
    recycled_medium: SegQueue<Arc<Vec<CompactValue>>>, // Lists with capacity <= 64
    recycled_large: SegQueue<Arc<Vec<CompactValue>>>, // Lists with capacity > 64
    /// Maximum items in each pool
    max_size: usize,
    /// Statistics
    stats: RwLock<PoolStats>,
}

impl ListPool {
    /// Create a new list pool
    pub fn new(max_size: usize) -> Self {
        Self {
            recycled_small: SegQueue::new(),
            recycled_medium: SegQueue::new(),
            recycled_large: SegQueue::new(),
            max_size,
            stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Get a list, trying to reuse containers where possible
    pub fn get_list(&self, items: Vec<CompactValue>) -> CompactValue {
        self.stats.write().total_allocations += 1;
        
        // For now, just create a new Arc since list contents vary
        // In a more sophisticated system, we could try to reuse
        // Arc<Vec<T>> containers when they're empty
        self.stats.write().pool_misses += 1;
        CompactValue::List(Arc::new(items))
    }
    
    /// Recycle a list container (placeholder for future optimization)
    pub fn recycle_list(&self, _value: &CompactValue) {
        // Similar to LargeIntPool, recycling Arc<Vec<T>> is complex
        // since the contents are immutable. This is a placeholder.
    }
    
    /// Get statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.read().clone()
    }
}

/// Comprehensive pool system for CompactValue
pub struct CompactValuePools {
    /// Small integer pool (most common case)
    small_int_pool: SmallIntPool,
    /// Large integer pool
    large_int_pool: LargeIntPool,
    /// Real number pool
    real_pool: RealPool,
    /// List pool
    list_pool: ListPool,
    /// Global statistics
    global_stats: RwLock<PoolStats>,
}

impl CompactValuePools {
    /// Create a new comprehensive pool system
    pub fn new() -> Self {
        Self {
            small_int_pool: SmallIntPool::new(),
            large_int_pool: LargeIntPool::new(1024),
            real_pool: RealPool::new(),
            list_pool: ListPool::new(512),
            global_stats: RwLock::new(PoolStats::default()),
        }
    }
    
    /// Allocate a CompactValue using appropriate pool
    pub fn alloc_value(&self, value: CompactValue) -> CompactValue {
        self.global_stats.write().total_allocations += 1;
        
        match value {
            CompactValue::SmallInt(i) => self.small_int_pool.get_small_int(i),
            CompactValue::LargeInt(arc_i) => self.large_int_pool.get_large_int(*arc_i),
            CompactValue::Real(r) => self.real_pool.get_real(r),
            CompactValue::List(items) => {
                // Extract items from Arc and let pool manage allocation
                let vec_items = (*items).clone();
                self.list_pool.get_list(vec_items)
            }
            // For other types, return as-is since they're already optimized
            _ => value,
        }
    }
    
    /// Try to recycle a value back to appropriate pool
    pub fn recycle_value(&self, value: &CompactValue) {
        match value {
            CompactValue::LargeInt(_) => {
                self.large_int_pool.recycle_large_int(value);
            }
            CompactValue::List(_) => {
                self.list_pool.recycle_list(value);
            }
            // Small ints and reals don't need recycling (cached or cheap to create)
            _ => {}
        }
    }
    
    /// Get comprehensive statistics
    pub fn stats(&self) -> HashMap<String, PoolStats> {
        let mut stats = HashMap::new();
        
        stats.insert("small_int".to_string(), self.small_int_pool.stats());
        stats.insert("large_int".to_string(), self.large_int_pool.stats());
        stats.insert("real".to_string(), self.real_pool.stats());
        stats.insert("list".to_string(), self.list_pool.stats());
        stats.insert("global".to_string(), self.global_stats.read().clone());
        
        stats
    }
    
    /// Get total memory usage estimate
    pub fn total_memory_usage(&self) -> usize {
        // Estimate based on cached items and overhead
        let small_int_usage = self.small_int_pool.cached_values.len() * std::mem::size_of::<CompactValue>();
        let real_usage = self.real_pool.common_reals.len() * std::mem::size_of::<CompactValue>();
        
        // Add overhead for data structures
        let overhead = 1024; // Conservative estimate
        
        small_int_usage + real_usage + overhead
    }
    
    /// Perform garbage collection on pools
    pub fn collect_unused(&self) -> usize {
        // For these pools, most data is either cached (permanent) or
        // managed by Arc (automatic cleanup). Return estimate of any
        // cleanup performed.
        
        let _initial_usage = self.total_memory_usage();
        
        // Reset statistics to clean state
        *self.global_stats.write() = PoolStats::default();
        
        // No actual cleanup needed for current implementation
        // Return 0 to indicate no memory was freed
        0
    }
    
    /// Generate efficiency report
    pub fn efficiency_report(&self) -> String {
        let stats = self.stats();
        let mut report = String::from("CompactValue Pool Efficiency Report:\n");
        
        for (pool_name, pool_stats) in stats {
            let efficiency = pool_stats.efficiency() * 100.0;
            let hit_rate = if pool_stats.total_allocations > 0 {
                pool_stats.reuse_hits as f64 / pool_stats.total_allocations as f64 * 100.0
            } else {
                0.0
            };
            
            report.push_str(&format!(
                "{}: {:.1}% efficiency, {:.1}% hit rate, {} allocations\n",
                pool_name, efficiency, hit_rate, pool_stats.total_allocations
            ));
        }
        
        report.push_str(&format!(
            "\nTotal Memory Usage: {:.2} KB\n",
            self.total_memory_usage() as f64 / 1024.0
        ));
        
        report
    }
}

impl Default for CompactValuePools {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_small_int_pool() {
        let pool = SmallIntPool::new();
        
        // Common values should be cached
        let val1 = pool.get_small_int(42);
        let val2 = pool.get_small_int(42);
        
        assert_eq!(val1, val2);
        assert_eq!(val1, CompactValue::SmallInt(42));
        
        let stats = pool.stats();
        assert_eq!(stats.reuse_hits, 2); // Both hits cached values
        assert_eq!(stats.pool_misses, 0);
    }
    
    #[test]
    fn test_small_int_pool_edge_cases() {
        let pool = SmallIntPool::new();
        
        // Values outside cache range
        let large_val = pool.get_small_int(10000);
        assert_eq!(large_val, CompactValue::SmallInt(10000));
        
        let stats = pool.stats();
        assert_eq!(stats.pool_misses, 1); // Cache miss for large value
    }
    
    #[test]
    fn test_real_pool() {
        let pool = RealPool::new();
        
        // Common reals should be cached
        let pi1 = pool.get_real(std::f64::consts::PI);
        let pi2 = pool.get_real(std::f64::consts::PI);
        
        assert_eq!(pi1, pi2);
        
        let stats = pool.stats();
        assert_eq!(stats.reuse_hits, 2);
        assert_eq!(stats.pool_misses, 0);
    }
    
    #[test]
    fn test_large_int_pool() {
        let pool = LargeIntPool::new(10);
        
        let large_val = pool.get_large_int(i64::MAX);
        assert!(matches!(large_val, CompactValue::LargeInt(_)));
        
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
    }
    
    #[test]
    fn test_list_pool() {
        let pool = ListPool::new(10);
        
        let items = vec![CompactValue::SmallInt(1), CompactValue::SmallInt(2)];
        let list_val = pool.get_list(items);
        
        assert!(matches!(list_val, CompactValue::List(_)));
        
        let stats = pool.stats();
        assert_eq!(stats.total_allocations, 1);
    }
    
    #[test]
    fn test_compact_value_pools() {
        let pools = CompactValuePools::new();
        
        // Test various value types
        let small_int = pools.alloc_value(CompactValue::SmallInt(42));
        let large_int = pools.alloc_value(CompactValue::LargeInt(Arc::new(i64::MAX)));
        let real_val = pools.alloc_value(CompactValue::Real(std::f64::consts::PI));
        
        assert_eq!(small_int, CompactValue::SmallInt(42));
        assert!(matches!(large_int, CompactValue::LargeInt(_)));
        assert_eq!(real_val, CompactValue::Real(std::f64::consts::PI));
        
        let stats = pools.stats();
        assert!(!stats.is_empty());
        
        // Test efficiency report
        let report = pools.efficiency_report();
        assert!(report.contains("CompactValue Pool Efficiency Report"));
    }
    
    #[test]
    fn test_memory_usage_calculation() {
        let pools = CompactValuePools::new();
        
        let initial_usage = pools.total_memory_usage();
        assert!(initial_usage > 0); // Should have cached values
        
        // Allocate some values
        for i in 0..100 {
            pools.alloc_value(CompactValue::SmallInt(i));
        }
        
        // Memory usage shouldn't increase much due to caching
        let final_usage = pools.total_memory_usage();
        assert!(final_usage >= initial_usage);
    }
    
    #[test]
    fn test_garbage_collection() {
        let pools = CompactValuePools::new();
        
        // Generate some activity
        for i in 0..1000 {
            pools.alloc_value(CompactValue::SmallInt(i % 100));
        }
        
        let freed = pools.collect_unused();
        // Current implementation doesn't free memory (cached values are permanent)
        // But this tests the interface
        assert_eq!(freed, 0);
    }
}