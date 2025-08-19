//! Memory management statistics and monitoring

use std::collections::HashMap;

/// Statistics for string interning performance
#[derive(Debug, Clone, Copy, Default)]
pub struct InternerStats {
    /// Number of times a static symbol was found
    pub static_hits: u64,
    /// Number of times a dynamic symbol was found in cache
    pub dynamic_hits: u64,
    /// Number of times a new string had to be interned
    pub dynamic_misses: u64,
    /// Total number of strings interned dynamically
    pub total_interned: u64,
    /// Total bytes used by interned strings
    pub total_bytes: usize,
}

impl InternerStats {
    /// Calculate hit ratio for performance monitoring
    pub fn hit_ratio(&self) -> f64 {
        let total_requests = self.static_hits + self.dynamic_hits + self.dynamic_misses;
        if total_requests == 0 {
            0.0
        } else {
            (self.static_hits + self.dynamic_hits) as f64 / total_requests as f64
        }
    }
    
    /// Get average string length
    pub fn average_string_length(&self) -> f64 {
        if self.total_interned == 0 {
            0.0
        } else {
            self.total_bytes as f64 / self.total_interned as f64
        }
    }
}

/// Statistics for a typed memory pool
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total allocations from this pool
    pub total_allocations: u64,
    /// Current number of items in pool
    pub current_size: usize,
    /// Maximum size the pool has reached
    pub peak_size: usize,
    /// Number of items currently available for reuse
    pub available_items: usize,
    /// Total bytes allocated by this pool
    pub total_bytes: usize,
    /// Number of times pool was exhausted and had to allocate
    pub pool_misses: u64,
    /// Number of times an item was successfully reused
    pub reuse_hits: u64,
}

impl PoolStats {
    /// Calculate pool efficiency (reuse ratio)
    pub fn efficiency(&self) -> f64 {
        let total = self.reuse_hits + self.pool_misses;
        if total == 0 {
            0.0
        } else {
            self.reuse_hits as f64 / total as f64
        }
    }
    
    /// Get memory utilization ratio
    pub fn utilization(&self) -> f64 {
        if self.peak_size == 0 {
            0.0
        } else {
            self.current_size as f64 / self.peak_size as f64
        }
    }
}

/// Statistics for arena allocation
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total number of scopes created
    pub total_scopes: u64,
    /// Currently active scopes
    pub active_scopes: usize,
    /// Peak number of concurrent scopes
    pub peak_scopes: usize,
    /// Total bytes allocated in arenas
    pub total_allocated: usize,
    /// Total bytes freed when scopes ended
    pub total_freed: usize,
    /// Number of garbage collection cycles
    pub gc_cycles: u64,
    /// Bytes freed in last GC cycle
    pub last_gc_freed: usize,
}

impl ArenaStats {
    /// Calculate memory efficiency (freed/allocated ratio)
    pub fn efficiency(&self) -> f64 {
        if self.total_allocated == 0 {
            0.0
        } else {
            self.total_freed as f64 / self.total_allocated as f64
        }
    }
    
    /// Get current memory usage
    pub fn current_usage(&self) -> usize {
        self.total_allocated.saturating_sub(self.total_freed)
    }
}

/// Comprehensive memory management statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory allocated across all systems
    pub total_allocated: usize,
    /// String interning statistics
    pub interner_stats: InternerStats,
    /// Per-pool statistics
    pub pool_stats: HashMap<String, PoolStats>,
    /// Arena allocation statistics
    pub arena_stats: ArenaStats,
}

impl MemoryStats {
    /// Calculate overall memory efficiency
    pub fn overall_efficiency(&self) -> f64 {
        // Weighted average of pool and arena efficiency
        let pool_efficiency: f64 = self.pool_stats.values()
            .map(|stats| stats.efficiency())
            .sum::<f64>() / self.pool_stats.len().max(1) as f64;
        
        let arena_efficiency = self.arena_stats.efficiency();
        
        (pool_efficiency + arena_efficiency) / 2.0
    }
    
    /// Get total current memory usage
    pub fn current_usage(&self) -> usize {
        let pool_usage: usize = self.pool_stats.values()
            .map(|stats| stats.total_bytes)
            .sum();
        
        pool_usage + self.arena_stats.current_usage()
    }
    
    /// Format statistics for human-readable display
    pub fn format_summary(&self) -> String {
        format!(
            "Memory Statistics:\n\
             Total Allocated: {:.2} MB\n\
             Current Usage: {:.2} MB\n\
             Overall Efficiency: {:.1}%\n\
             \n\
             String Interning:\n\
             - Hit Ratio: {:.1}%\n\
             - Total Interned: {}\n\
             - Avg Length: {:.1} chars\n\
             \n\
             Arena Allocation:\n\
             - Active Scopes: {}\n\
             - Efficiency: {:.1}%\n\
             - GC Cycles: {}\n\
             \n\
             Pool Summary: {} pools active",
            self.total_allocated as f64 / 1024.0 / 1024.0,
            self.current_usage() as f64 / 1024.0 / 1024.0,
            self.overall_efficiency() * 100.0,
            self.interner_stats.hit_ratio() * 100.0,
            self.interner_stats.total_interned,
            self.interner_stats.average_string_length(),
            self.arena_stats.active_scopes,
            self.arena_stats.efficiency() * 100.0,
            self.arena_stats.gc_cycles,
            self.pool_stats.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_interner_stats() {
        let mut stats = InternerStats::default();
        stats.static_hits = 80;
        stats.dynamic_hits = 15;
        stats.dynamic_misses = 5;
        stats.total_interned = 5;
        stats.total_bytes = 50;
        
        assert_eq!(stats.hit_ratio(), 0.95); // 95% hit rate
        assert_eq!(stats.average_string_length(), 10.0); // 50 bytes / 5 strings
    }
    
    #[test]
    fn test_pool_stats() {
        let mut stats = PoolStats::default();
        stats.reuse_hits = 90;
        stats.pool_misses = 10;
        stats.current_size = 75;
        stats.peak_size = 100;
        
        assert_eq!(stats.efficiency(), 0.9); // 90% reuse rate
        assert_eq!(stats.utilization(), 0.75); // 75% utilization
    }
    
    #[test]
    fn test_arena_stats() {
        let mut stats = ArenaStats::default();
        stats.total_allocated = 1000;
        stats.total_freed = 800;
        
        assert_eq!(stats.efficiency(), 0.8); // 80% freed
        assert_eq!(stats.current_usage(), 200); // 200 bytes still allocated
    }
    
    #[test]
    fn test_memory_stats_summary() {
        let stats = MemoryStats {
            total_allocated: 1024 * 1024, // 1 MB
            interner_stats: InternerStats {
                static_hits: 100,
                dynamic_hits: 0,
                dynamic_misses: 0,
                total_interned: 10,
                total_bytes: 100,
            },
            arena_stats: ArenaStats {
                active_scopes: 5,
                total_allocated: 512 * 1024,
                total_freed: 256 * 1024,
                gc_cycles: 3,
                ..Default::default()
            },
            pool_stats: HashMap::new(),
        };
        
        let summary = stats.format_summary();
        assert!(summary.contains("1.00 MB")); // Total allocated
        assert!(summary.contains("100.0%")); // Hit ratio (all static hits)
        assert!(summary.contains("5")); // Active scopes
    }
}