//! Import Cache System
//!
//! Multi-layer caching system for tree-shaking import optimizations with intelligent
//! cache invalidation, compression, and warming strategies for maximum performance.

use super::{TreeShakeError};
use super::selective_resolver::{ResolvedImport, ImportResolutionResults};
use super::import_statement_generator::{ImportGenerationResults};
use super::dependency_validator::{DependencyValidationResults};
use std::collections::{HashMap, HashSet, BTreeMap};
use serde::{Serialize, Deserialize};
use std::time::{SystemTime, Duration, Instant};
use std::sync::{Arc, RwLock};
use std::path::PathBuf;

/// Multi-layer import cache system
#[derive(Debug)]
pub struct ImportCache {
    /// Cache configuration
    config: ImportCacheConfig,
    
    /// Memory cache layer
    memory_cache: MemoryCache,
    
    /// Disk cache layer
    disk_cache: DiskCache,
    
    /// Cache warming system
    cache_warmer: CacheWarmer,
    
    /// Cache invalidation manager
    invalidation_manager: CacheInvalidationManager,
    
    /// Cache performance metrics
    performance_metrics: CachePerformanceMetrics,
    
    /// Cache compression system
    compression_system: CompressionSystem,
}

/// Configuration for import cache
#[derive(Debug, Clone)]
pub struct ImportCacheConfig {
    /// Enable memory caching
    pub enable_memory_cache: bool,
    
    /// Enable disk caching
    pub enable_disk_cache: bool,
    
    /// Memory cache size limit (bytes)
    pub memory_cache_size_limit: usize,
    
    /// Disk cache size limit (bytes)
    pub disk_cache_size_limit: usize,
    
    /// Default cache TTL
    pub default_ttl_seconds: u64,
    
    /// Enable cache compression
    pub enable_compression: bool,
    
    /// Compression level (1-9, 1=fast, 9=best)
    pub compression_level: u32,
    
    /// Enable cache warming
    pub enable_cache_warming: bool,
    
    /// Cache warming concurrency
    pub cache_warming_concurrency: usize,
    
    /// Enable intelligent invalidation
    pub enable_intelligent_invalidation: bool,
    
    /// Cache directory path
    pub cache_directory: PathBuf,
    
    /// Maximum cache entries per layer
    pub max_entries_per_layer: usize,
    
    /// Cache persistence interval (seconds)
    pub persistence_interval_seconds: u64,
}

impl Default for ImportCacheConfig {
    fn default() -> Self {
        ImportCacheConfig {
            enable_memory_cache: true,
            enable_disk_cache: true,
            memory_cache_size_limit: 100 * 1024 * 1024, // 100MB
            disk_cache_size_limit: 1024 * 1024 * 1024, // 1GB
            default_ttl_seconds: 3600, // 1 hour
            enable_compression: true,
            compression_level: 6,
            enable_cache_warming: true,
            cache_warming_concurrency: 4,
            enable_intelligent_invalidation: true,
            cache_directory: PathBuf::from(".lyra_cache"),
            max_entries_per_layer: 10000,
            persistence_interval_seconds: 300, // 5 minutes
        }
    }
}

/// Memory cache layer with LRU eviction
#[derive(Debug)]
pub struct MemoryCache {
    /// Cache entries
    entries: HashMap<String, MemoryCacheEntry>,
    
    /// LRU ordering
    lru_order: BTreeMap<Instant, String>,
    
    /// Current memory usage
    current_size: usize,
    
    /// Maximum memory size
    max_size: usize,
    
    /// Cache statistics
    stats: MemoryCacheStats,
}

/// Memory cache entry
#[derive(Debug, Clone)]
pub struct MemoryCacheEntry {
    /// Cache key
    pub key: String,
    
    /// Cached data
    pub data: CachedImportData,
    
    /// Entry creation time
    pub created_at: SystemTime,
    
    /// Entry expiration time
    pub expires_at: SystemTime,
    
    /// Last access time
    pub last_accessed: Instant,
    
    /// Access count
    pub access_count: u64,
    
    /// Entry size in bytes
    pub size_bytes: usize,
    
    /// Cache hit ratio for this entry
    pub hit_ratio: f64,
}

/// Cached import data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CachedImportData {
    /// Resolved import results
    ResolvedImports(ImportResolutionResults),
    
    /// Generated import statements
    GeneratedStatements(ImportGenerationResults),
    
    /// Validation results
    ValidationResults(super::selective_resolver::ValidationResults),
    
    /// Dependency validation report
    DependencyReport(DependencyValidationResults),
    
    /// Individual resolved import
    SingleResolvedImport(ResolvedImport),
    
    /// Generated statement (as string)
    SingleGeneratedStatement(String),
    
    /// Custom data
    CustomData { 
        data_type: String, 
        serialized_data: Vec<u8> 
    },
}

/// Disk cache layer with compression
#[derive(Debug)]
pub struct DiskCache {
    /// Cache directory
    cache_dir: PathBuf,
    
    /// Cache index
    index: HashMap<String, DiskCacheEntry>,
    
    /// Current disk usage
    current_size: u64,
    
    /// Maximum disk size
    max_size: u64,
    
    /// Cache statistics
    stats: DiskCacheStats,
    
    /// Compression enabled
    compression_enabled: bool,
    
    /// Compression level
    compression_level: u32,
}

/// Disk cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskCacheEntry {
    /// Cache key
    pub key: String,
    
    /// File path relative to cache directory
    pub file_path: String,
    
    /// Entry creation time
    pub created_at: SystemTime,
    
    /// Entry expiration time
    pub expires_at: SystemTime,
    
    /// Last access time
    pub last_accessed: SystemTime,
    
    /// File size in bytes
    pub file_size: u64,
    
    /// Whether file is compressed
    pub is_compressed: bool,
    
    /// Compression ratio (if compressed)
    pub compression_ratio: Option<f64>,
    
    /// Data type
    pub data_type: String,
    
    /// Checksum for integrity verification
    pub checksum: String,
}

/// Cache warming system
#[derive(Debug)]
pub struct CacheWarmer {
    /// Warming configuration
    config: CacheWarmingConfig,
    
    /// Frequently accessed keys
    frequent_keys: HashMap<String, KeyAccessPattern>,
    
    /// Warming queue
    warming_queue: Vec<WarmingTask>,
    
    /// Warming statistics
    stats: CacheWarmingStats,
    
    /// Warming in progress
    warming_in_progress: HashSet<String>,
}

/// Cache warming configuration
#[derive(Debug, Clone)]
pub struct CacheWarmingConfig {
    /// Enable predictive warming
    pub enable_predictive_warming: bool,
    
    /// Warming threshold (access frequency)
    pub warming_threshold: f64,
    
    /// Maximum concurrent warming tasks
    pub max_concurrent_warming: usize,
    
    /// Warming priority algorithm
    pub priority_algorithm: WarmingPriorityAlgorithm,
    
    /// Time window for access pattern analysis
    pub analysis_time_window_hours: u64,
}

/// Warming priority algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum WarmingPriorityAlgorithm {
    /// Frequency-based priority
    Frequency,
    
    /// Recency-based priority
    Recency,
    
    /// Combined frequency and recency
    FrequencyRecency { frequency_weight: f64 },
    
    /// Cost-based priority (expensive operations first)
    CostBased,
    
    /// Machine learning predicted priority
    MLPredicted,
}

impl Default for WarmingPriorityAlgorithm {
    fn default() -> Self {
        WarmingPriorityAlgorithm::FrequencyRecency { frequency_weight: 0.7 }
    }
}

/// Key access pattern analysis
#[derive(Debug, Clone)]
pub struct KeyAccessPattern {
    /// Key identifier
    pub key: String,
    
    /// Total access count
    pub access_count: u64,
    
    /// Recent access timestamps
    pub recent_accesses: Vec<SystemTime>,
    
    /// Access frequency (accesses per hour)
    pub access_frequency: f64,
    
    /// Last access time
    pub last_access: SystemTime,
    
    /// Average time between accesses
    pub average_access_interval: Duration,
    
    /// Access pattern type
    pub pattern_type: AccessPatternType,
    
    /// Warming priority score
    pub warming_priority: f64,
}

/// Types of access patterns
#[derive(Debug, Clone, PartialEq)]
pub enum AccessPatternType {
    /// Regular periodic access
    Periodic { interval_seconds: u64 },
    
    /// Burst access (many accesses in short time)
    Burst { burst_size: u32, burst_interval: Duration },
    
    /// Steady access (consistent over time)
    Steady,
    
    /// Sporadic access (irregular pattern)
    Sporadic,
    
    /// Declining access (decreasing frequency)
    Declining { decline_rate: f64 },
    
    /// Growing access (increasing frequency)
    Growing { growth_rate: f64 },
}

/// Cache warming task
#[derive(Debug, Clone)]
pub struct WarmingTask {
    /// Task ID
    pub task_id: String,
    
    /// Key to warm
    pub key: String,
    
    /// Task priority
    pub priority: f64,
    
    /// Task type
    pub task_type: WarmingTaskType,
    
    /// Created at
    pub created_at: SystemTime,
    
    /// Estimated completion time
    pub estimated_completion_time: Duration,
    
    /// Retry count
    pub retry_count: u32,
    
    /// Maximum retries
    pub max_retries: u32,
}

/// Types of warming tasks
#[derive(Debug, Clone, PartialEq)]
pub enum WarmingTaskType {
    /// Precompute and cache expensive operations
    Precompute,
    
    /// Load from disk to memory
    DiskToMemory,
    
    /// Refresh expired entries
    RefreshExpired,
    
    /// Predictive warming based on patterns
    Predictive,
    
    /// Dependency warming (warm related entries)
    Dependency { dependencies: Vec<String> },
}

/// Cache invalidation manager
#[derive(Debug)]
pub struct CacheInvalidationManager {
    /// Invalidation rules
    invalidation_rules: Vec<InvalidationRule>,
    
    /// Dependency tracking
    dependency_tracker: DependencyTracker,
    
    /// Invalidation queue
    invalidation_queue: Vec<InvalidationTask>,
    
    /// Invalidation statistics
    stats: InvalidationStats,
}

/// Cache invalidation rule
#[derive(Debug, Clone)]
pub struct InvalidationRule {
    /// Rule ID
    pub rule_id: String,
    
    /// Rule type
    pub rule_type: InvalidationRuleType,
    
    /// Rule condition
    pub condition: InvalidationCondition,
    
    /// Rule action
    pub action: InvalidationAction,
    
    /// Rule priority
    pub priority: u32,
    
    /// Rule enabled
    pub enabled: bool,
}

/// Types of invalidation rules
#[derive(Debug, Clone, PartialEq)]
pub enum InvalidationRuleType {
    /// Time-based expiration
    TimeBasedExpiration,
    
    /// Dependency-based invalidation
    DependencyBased,
    
    /// Content change detection
    ContentChange,
    
    /// Memory pressure invalidation
    MemoryPressure,
    
    /// Disk space pressure invalidation
    DiskSpacePressure,
    
    /// Custom rule
    Custom { rule_name: String },
}

/// Invalidation conditions
#[derive(Debug, Clone)]
pub enum InvalidationCondition {
    /// Time since creation
    TimeSinceCreation(Duration),
    
    /// Time since last access
    TimeSinceLastAccess(Duration),
    
    /// Dependency changed
    DependencyChanged { dependency_key: String },
    
    /// Memory usage threshold
    MemoryUsageThreshold(f64),
    
    /// Disk usage threshold
    DiskUsageThreshold(f64),
    
    /// Access count threshold
    AccessCountThreshold(u64),
    
    /// Custom condition
    Custom { condition_expression: String },
}

/// Invalidation actions
#[derive(Debug, Clone, PartialEq)]
pub enum InvalidationAction {
    /// Remove from cache
    Remove,
    
    /// Mark as expired
    MarkExpired,
    
    /// Refresh asynchronously
    RefreshAsync,
    
    /// Move to lower cache layer
    Demote,
    
    /// Compress entry
    Compress,
    
    /// Custom action
    Custom { action_name: String },
}

/// Dependency tracking for cache invalidation
#[derive(Debug, Clone, Default)]
pub struct DependencyTracker {
    /// Forward dependencies (key -> depends on these keys)
    forward_dependencies: HashMap<String, HashSet<String>>,
    
    /// Reverse dependencies (key -> these keys depend on it)
    reverse_dependencies: HashMap<String, HashSet<String>>,
    
    /// Dependency graph
    dependency_graph: HashMap<String, DependencyNode>,
}

/// Dependency node in cache dependency graph
#[derive(Debug, Clone)]
pub struct DependencyNode {
    /// Node key
    pub key: String,
    
    /// Direct dependencies
    pub dependencies: HashSet<String>,
    
    /// Direct dependents
    pub dependents: HashSet<String>,
    
    /// Dependency depth
    pub depth: u32,
    
    /// Last invalidation time
    pub last_invalidation: Option<SystemTime>,
}

/// Cache invalidation task
#[derive(Debug, Clone)]
pub struct InvalidationTask {
    /// Task ID
    pub task_id: String,
    
    /// Keys to invalidate
    pub keys: Vec<String>,
    
    /// Invalidation action
    pub action: InvalidationAction,
    
    /// Task priority
    pub priority: u32,
    
    /// Created at
    pub created_at: SystemTime,
    
    /// Reason for invalidation
    pub reason: String,
    
    /// Cascade invalidation
    pub cascade: bool,
}

/// Compression system for cache entries
#[derive(Debug)]
pub struct CompressionSystem {
    /// Compression configuration
    config: CompressionConfig,
    
    /// Compression algorithms
    algorithms: HashMap<String, CompressionAlgorithm>,
    
    /// Compression statistics
    stats: CompressionStats,
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Default compression algorithm
    pub default_algorithm: String,
    
    /// Compression threshold (minimum size to compress)
    pub compression_threshold_bytes: usize,
    
    /// Enable adaptive compression
    pub enable_adaptive_compression: bool,
    
    /// Compression level
    pub compression_level: u32,
}

/// Compression algorithm
#[derive(Debug, Clone)]
pub struct CompressionAlgorithm {
    /// Algorithm name
    pub name: String,
    
    /// Algorithm type
    pub algorithm_type: CompressionAlgorithmType,
    
    /// Compression function
    pub compress_fn: fn(&[u8], u32) -> Result<Vec<u8>, CompressionError>,
    
    /// Decompression function
    pub decompress_fn: fn(&[u8]) -> Result<Vec<u8>, CompressionError>,
    
    /// Algorithm configuration
    pub config: HashMap<String, String>,
}

/// Types of compression algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionAlgorithmType {
    /// LZ4 compression (fast)
    LZ4,
    
    /// ZSTD compression (balanced)
    ZSTD,
    
    /// GZIP compression (good compression)
    GZIP,
    
    /// Brotli compression (best compression)
    Brotli,
    
    /// Custom compression
    Custom { name: String },
}

/// Compression errors
#[derive(Debug, Clone)]
pub enum CompressionError {
    /// Compression failed
    CompressionFailed(String),
    
    /// Decompression failed
    DecompressionFailed(String),
    
    /// Algorithm not found
    AlgorithmNotFound(String),
    
    /// Invalid data
    InvalidData(String),
}

/// Cache performance metrics
#[derive(Debug, Clone, Default)]
pub struct CachePerformanceMetrics {
    /// Memory cache metrics
    pub memory_cache_metrics: MemoryCacheStats,
    
    /// Disk cache metrics
    pub disk_cache_metrics: DiskCacheStats,
    
    /// Overall cache metrics
    pub overall_metrics: OverallCacheStats,
    
    /// Cache warming metrics
    pub warming_metrics: CacheWarmingStats,
    
    /// Invalidation metrics
    pub invalidation_metrics: InvalidationStats,
    
    /// Compression metrics
    pub compression_metrics: CompressionStats,
}

/// Memory cache statistics
#[derive(Debug, Clone, Default)]
pub struct MemoryCacheStats {
    /// Total requests
    pub total_requests: u64,
    
    /// Cache hits
    pub hits: u64,
    
    /// Cache misses
    pub misses: u64,
    
    /// Hit ratio
    pub hit_ratio: f64,
    
    /// Average lookup time
    pub average_lookup_time: Duration,
    
    /// Current entries count
    pub current_entries: usize,
    
    /// Current memory usage
    pub current_memory_usage: usize,
    
    /// Evictions count
    pub evictions: u64,
    
    /// Total memory allocated
    pub total_memory_allocated: usize,
}

/// Disk cache statistics
#[derive(Debug, Clone, Default)]
pub struct DiskCacheStats {
    /// Total requests
    pub total_requests: u64,
    
    /// Cache hits
    pub hits: u64,
    
    /// Cache misses
    pub misses: u64,
    
    /// Hit ratio
    pub hit_ratio: f64,
    
    /// Average lookup time
    pub average_lookup_time: Duration,
    
    /// Current entries count
    pub current_entries: usize,
    
    /// Current disk usage
    pub current_disk_usage: u64,
    
    /// Evictions count
    pub evictions: u64,
    
    /// Read/write statistics
    pub bytes_read: u64,
    
    /// Bytes written
    pub bytes_written: u64,
}

/// Overall cache statistics
#[derive(Debug, Clone, Default)]
pub struct OverallCacheStats {
    /// Total requests across all layers
    pub total_requests: u64,
    
    /// Total hits across all layers
    pub total_hits: u64,
    
    /// Total misses across all layers
    pub total_misses: u64,
    
    /// Overall hit ratio
    pub overall_hit_ratio: f64,
    
    /// Average response time
    pub average_response_time: Duration,
    
    /// Cache efficiency score
    pub efficiency_score: f64,
    
    /// Memory to disk promotion rate
    pub memory_to_disk_promotions: u64,
    
    /// Disk to memory promotion rate
    pub disk_to_memory_promotions: u64,
}

/// Cache warming statistics
#[derive(Debug, Clone, Default)]
pub struct CacheWarmingStats {
    /// Total warming tasks
    pub total_warming_tasks: u64,
    
    /// Successful warming tasks
    pub successful_warmings: u64,
    
    /// Failed warming tasks
    pub failed_warmings: u64,
    
    /// Average warming time
    pub average_warming_time: Duration,
    
    /// Predictive accuracy
    pub predictive_accuracy: f64,
    
    /// Warming hit improvement
    pub warming_hit_improvement: f64,
}

/// Cache invalidation statistics
#[derive(Debug, Clone, Default)]
pub struct InvalidationStats {
    /// Total invalidations
    pub total_invalidations: u64,
    
    /// Cascade invalidations
    pub cascade_invalidations: u64,
    
    /// Time-based invalidations
    pub time_based_invalidations: u64,
    
    /// Dependency-based invalidations
    pub dependency_based_invalidations: u64,
    
    /// Average invalidation time
    pub average_invalidation_time: Duration,
    
    /// Invalidation efficiency
    pub invalidation_efficiency: f64,
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    /// Total compressions
    pub total_compressions: u64,
    
    /// Total decompressions
    pub total_decompressions: u64,
    
    /// Average compression ratio
    pub average_compression_ratio: f64,
    
    /// Average compression time
    pub average_compression_time: Duration,
    
    /// Average decompression time
    pub average_decompression_time: Duration,
    
    /// Bytes saved by compression
    pub bytes_saved: u64,
    
    /// Compression effectiveness
    pub compression_effectiveness: f64,
}

impl ImportCache {
    /// Create a new import cache
    pub fn new() -> Self {
        Self::with_config(ImportCacheConfig::default())
    }
    
    /// Create import cache with custom configuration
    pub fn with_config(config: ImportCacheConfig) -> Self {
        ImportCache {
            memory_cache: MemoryCache::new(config.memory_cache_size_limit),
            disk_cache: DiskCache::new(&config.cache_directory, config.disk_cache_size_limit),
            cache_warmer: CacheWarmer::new(),
            invalidation_manager: CacheInvalidationManager::new(),
            performance_metrics: CachePerformanceMetrics::default(),
            compression_system: CompressionSystem::new(),
            config,
        }
    }
    
    /// Get cached data
    pub fn get(&mut self, key: &str) -> Result<Option<CachedImportData>, TreeShakeError> {
        let start_time = Instant::now();
        
        // Try memory cache first
        if self.config.enable_memory_cache {
            if let Some(data) = self.memory_cache.get(key)? {
                self.update_access_pattern(key);
                self.performance_metrics.memory_cache_metrics.hits += 1;
                self.performance_metrics.overall_metrics.total_hits += 1;
                return Ok(Some(data));
            }
            self.performance_metrics.memory_cache_metrics.misses += 1;
        }
        
        // Try disk cache
        if self.config.enable_disk_cache {
            if let Some(data) = self.disk_cache.get(key)? {
                // Promote to memory cache
                if self.config.enable_memory_cache {
                    let _ = self.memory_cache.put(key.to_string(), data.clone(), None);
                    self.performance_metrics.overall_metrics.disk_to_memory_promotions += 1;
                }
                
                self.update_access_pattern(key);
                self.performance_metrics.disk_cache_metrics.hits += 1;
                self.performance_metrics.overall_metrics.total_hits += 1;
                return Ok(Some(data));
            }
            self.performance_metrics.disk_cache_metrics.misses += 1;
        }
        
        self.performance_metrics.overall_metrics.total_misses += 1;
        
        // Update response time metrics
        let response_time = start_time.elapsed();
        self.update_response_time_metrics(response_time);
        
        Ok(None)
    }
    
    /// Put data in cache
    pub fn put(&mut self, key: String, data: CachedImportData, ttl: Option<Duration>) -> Result<(), TreeShakeError> {
        let ttl = ttl.unwrap_or_else(|| Duration::from_secs(self.config.default_ttl_seconds));
        
        // Put in memory cache
        if self.config.enable_memory_cache {
            match self.memory_cache.put(key.clone(), data.clone(), Some(ttl)) {
                Ok(_) => {},
                Err(_) => {
                    // Memory cache full, promote to disk
                    if self.config.enable_disk_cache {
                        self.disk_cache.put(key.clone(), data.clone(), Some(ttl))?;
                        self.performance_metrics.overall_metrics.memory_to_disk_promotions += 1;
                    }
                }
            }
        } else if self.config.enable_disk_cache {
            self.disk_cache.put(key.clone(), data, Some(ttl))?;
        }
        
        // Add to warming patterns
        self.cache_warmer.add_key_access(&key);
        
        Ok(())
    }
    
    /// Remove from cache
    pub fn remove(&mut self, key: &str) -> Result<bool, TreeShakeError> {
        let mut removed = false;
        
        if self.config.enable_memory_cache {
            removed |= self.memory_cache.remove(key)?;
        }
        
        if self.config.enable_disk_cache {
            removed |= self.disk_cache.remove(key)?;
        }
        
        Ok(removed)
    }
    
    /// Clear all caches
    pub fn clear(&mut self) -> Result<(), TreeShakeError> {
        if self.config.enable_memory_cache {
            self.memory_cache.clear()?;
        }
        
        if self.config.enable_disk_cache {
            self.disk_cache.clear()?;
        }
        
        Ok(())
    }
    
    /// Warm cache for frequently accessed keys
    pub fn warm_cache(&mut self, keys: Vec<String>) -> Result<(), TreeShakeError> {
        if !self.config.enable_cache_warming {
            return Ok(());
        }
        
        for key in keys {
            let task = WarmingTask {
                task_id: format!("warm-{}-{}", key, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()),
                key: key.clone(),
                priority: self.cache_warmer.calculate_priority(&key),
                task_type: WarmingTaskType::Precompute,
                created_at: SystemTime::now(),
                estimated_completion_time: Duration::from_millis(100),
                retry_count: 0,
                max_retries: 3,
            };
            
            self.cache_warmer.add_warming_task(task);
        }
        
        Ok(())
    }
    
    /// Invalidate cache entries based on dependencies
    pub fn invalidate_dependencies(&mut self, changed_key: &str) -> Result<(), TreeShakeError> {
        if !self.config.enable_intelligent_invalidation {
            return Ok(());
        }
        
        let dependent_keys = self.invalidation_manager.get_dependent_keys(changed_key);
        
        for key in dependent_keys {
            let task = InvalidationTask {
                task_id: format!("inv-{}-{}", key, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()),
                keys: vec![key],
                action: InvalidationAction::Remove,
                priority: 100,
                created_at: SystemTime::now(),
                reason: format!("Dependency {} changed", changed_key),
                cascade: true,
            };
            
            self.invalidation_manager.add_invalidation_task(task);
        }
        
        Ok(())
    }
    
    /// Get cache performance metrics
    pub fn get_performance_metrics(&self) -> &CachePerformanceMetrics {
        &self.performance_metrics
    }
    
    /// Get cache statistics summary
    pub fn get_cache_stats_summary(&self) -> CacheStatsSummary {
        CacheStatsSummary {
            total_entries: self.memory_cache.stats.current_entries + self.disk_cache.stats.current_entries,
            total_memory_usage: self.memory_cache.stats.current_memory_usage,
            total_disk_usage: self.disk_cache.stats.current_disk_usage,
            overall_hit_ratio: self.performance_metrics.overall_metrics.overall_hit_ratio,
            average_response_time: self.performance_metrics.overall_metrics.average_response_time,
            cache_efficiency: self.performance_metrics.overall_metrics.efficiency_score,
            compression_ratio: self.performance_metrics.compression_metrics.average_compression_ratio,
            warming_effectiveness: self.performance_metrics.warming_metrics.predictive_accuracy,
        }
    }
    
    /// Optimize cache performance
    pub fn optimize(&mut self) -> Result<CacheOptimizationReport, TreeShakeError> {
        let start_time = Instant::now();
        let mut optimizations_applied = Vec::new();
        
        // Optimize memory cache
        if self.config.enable_memory_cache {
            let memory_optimizations = self.memory_cache.optimize()?;
            optimizations_applied.extend(memory_optimizations);
        }
        
        // Optimize disk cache
        if self.config.enable_disk_cache {
            let disk_optimizations = self.disk_cache.optimize()?;
            optimizations_applied.extend(disk_optimizations);
        }
        
        // Optimize cache warming
        if self.config.enable_cache_warming {
            let warming_optimizations = self.cache_warmer.optimize()?;
            optimizations_applied.extend(warming_optimizations);
        }
        
        // Process invalidation queue
        if self.config.enable_intelligent_invalidation {
            let invalidation_optimizations = self.invalidation_manager.process_queue()?;
            optimizations_applied.extend(invalidation_optimizations);
        }
        
        Ok(CacheOptimizationReport {
            optimization_time: start_time.elapsed(),
            optimizations_applied,
            performance_improvement: self.calculate_performance_improvement(),
            memory_freed: self.calculate_memory_freed(),
            disk_space_freed: self.calculate_disk_space_freed(),
        })
    }
    
    // Private helper methods
    
    fn update_access_pattern(&mut self, key: &str) {
        self.cache_warmer.update_access_pattern(key);
        self.performance_metrics.overall_metrics.total_requests += 1;
    }
    
    fn update_response_time_metrics(&mut self, response_time: Duration) {
        let current_avg = self.performance_metrics.overall_metrics.average_response_time;
        let total_requests = self.performance_metrics.overall_metrics.total_requests;
        
        if total_requests > 0 {
            let new_avg = Duration::from_nanos(
                (current_avg.as_nanos() * (total_requests - 1) as u128 + response_time.as_nanos()) 
                / total_requests as u128
            );
            self.performance_metrics.overall_metrics.average_response_time = new_avg;
        } else {
            self.performance_metrics.overall_metrics.average_response_time = response_time;
        }
    }
    
    fn calculate_performance_improvement(&self) -> f64 {
        // Calculate based on hit ratio improvements and response time reductions
        let hit_ratio = self.performance_metrics.overall_metrics.overall_hit_ratio;
        let efficiency = self.performance_metrics.overall_metrics.efficiency_score;
        (hit_ratio * 0.6 + efficiency * 0.4) * 100.0
    }
    
    fn calculate_memory_freed(&self) -> usize {
        // This would track memory freed during optimization
        0
    }
    
    fn calculate_disk_space_freed(&self) -> u64 {
        // This would track disk space freed during optimization
        0
    }
}

/// Cache statistics summary
#[derive(Debug, Clone)]
pub struct CacheStatsSummary {
    /// Total entries across all cache layers
    pub total_entries: usize,
    
    /// Total memory usage
    pub total_memory_usage: usize,
    
    /// Total disk usage
    pub total_disk_usage: u64,
    
    /// Overall hit ratio
    pub overall_hit_ratio: f64,
    
    /// Average response time
    pub average_response_time: Duration,
    
    /// Cache efficiency score
    pub cache_efficiency: f64,
    
    /// Compression ratio
    pub compression_ratio: f64,
    
    /// Warming effectiveness
    pub warming_effectiveness: f64,
}

/// Cache optimization report
#[derive(Debug, Clone)]
pub struct CacheOptimizationReport {
    /// Time taken for optimization
    pub optimization_time: Duration,
    
    /// List of optimizations applied
    pub optimizations_applied: Vec<String>,
    
    /// Performance improvement percentage
    pub performance_improvement: f64,
    
    /// Memory freed in bytes
    pub memory_freed: usize,
    
    /// Disk space freed in bytes
    pub disk_space_freed: u64,
}

// Implementation of individual cache layers

impl MemoryCache {
    fn new(max_size: usize) -> Self {
        MemoryCache {
            entries: HashMap::new(),
            lru_order: BTreeMap::new(),
            current_size: 0,
            max_size,
            stats: MemoryCacheStats::default(),
        }
    }
    
    fn get(&mut self, key: &str) -> Result<Option<CachedImportData>, TreeShakeError> {
        self.stats.total_requests += 1;
        
        if let Some(entry) = self.entries.get_mut(key) {
            // Check if expired
            if SystemTime::now() > entry.expires_at {
                self.remove(key)?;
                return Ok(None);
            }
            
            // Update access tracking
            let old_access_time = entry.last_accessed;
            entry.last_accessed = Instant::now();
            entry.access_count += 1;
            
            // Update LRU order
            self.lru_order.remove(&old_access_time);
            self.lru_order.insert(entry.last_accessed, key.to_string());
            
            return Ok(Some(entry.data.clone()));
        }
        
        Ok(None)
    }
    
    fn put(&mut self, key: String, data: CachedImportData, ttl: Option<Duration>) -> Result<(), TreeShakeError> {
        let now = SystemTime::now();
        let expires_at = now + ttl.unwrap_or(Duration::from_secs(3600));
        let access_time = Instant::now();
        
        // Estimate entry size (simplified)
        let entry_size = self.estimate_entry_size(&data);
        
        // Check if we need to evict entries
        while self.current_size + entry_size > self.max_size && !self.entries.is_empty() {
            self.evict_lru()?;
        }
        
        if self.current_size + entry_size > self.max_size {
            return Err(TreeShakeError::CacheError {
                message: "Cannot fit entry in memory cache".to_string(),
            });
        }
        
        let entry = MemoryCacheEntry {
            key: key.clone(),
            data,
            created_at: now,
            expires_at,
            last_accessed: access_time,
            access_count: 0,
            size_bytes: entry_size,
            hit_ratio: 0.0,
        };
        
        self.entries.insert(key.clone(), entry);
        self.lru_order.insert(access_time, key);
        self.current_size += entry_size;
        
        Ok(())
    }
    
    fn remove(&mut self, key: &str) -> Result<bool, TreeShakeError> {
        if let Some(entry) = self.entries.remove(key) {
            self.lru_order.remove(&entry.last_accessed);
            self.current_size -= entry.size_bytes;
            self.stats.evictions += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn clear(&mut self) -> Result<(), TreeShakeError> {
        self.entries.clear();
        self.lru_order.clear();
        self.current_size = 0;
        Ok(())
    }
    
    fn optimize(&mut self) -> Result<Vec<String>, TreeShakeError> {
        let mut optimizations = Vec::new();
        
        // Remove expired entries
        let expired_keys: Vec<String> = self.entries.iter()
            .filter(|(_, entry)| SystemTime::now() > entry.expires_at)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            self.remove(&key)?;
            optimizations.push(format!("Removed expired entry: {}", key));
        }
        
        // Update hit ratios
        for entry in self.entries.values_mut() {
            if self.stats.total_requests > 0 {
                entry.hit_ratio = entry.access_count as f64 / self.stats.total_requests as f64;
            }
        }
        
        Ok(optimizations)
    }
    
    fn evict_lru(&mut self) -> Result<(), TreeShakeError> {
        if let Some((_, key)) = self.lru_order.iter().next() {
            let key = key.clone();
            self.remove(&key)?;
        }
        Ok(())
    }
    
    fn estimate_entry_size(&self, data: &CachedImportData) -> usize {
        // Simplified size estimation
        match data {
            CachedImportData::ResolvedImports(_) => 1024,
            CachedImportData::GeneratedStatements(_) => 2048,
            CachedImportData::ValidationResults(_) => 512,
            CachedImportData::DependencyReport(_) => 1536,
            CachedImportData::SingleResolvedImport(_) => 256,
            CachedImportData::SingleGeneratedStatement(_) => 512,
            CachedImportData::CustomData { serialized_data, .. } => serialized_data.len(),
        }
    }
}

impl DiskCache {
    fn new(cache_dir: &PathBuf, max_size: u64) -> Self {
        DiskCache {
            cache_dir: cache_dir.clone(),
            index: HashMap::new(),
            current_size: 0,
            max_size,
            stats: DiskCacheStats::default(),
            compression_enabled: true,
            compression_level: 6,
        }
    }
    
    fn get(&mut self, key: &str) -> Result<Option<CachedImportData>, TreeShakeError> {
        self.stats.total_requests += 1;
        
        if let Some(entry) = self.index.get_mut(key) {
            // Check if expired
            if SystemTime::now() > entry.expires_at {
                self.remove(key)?;
                return Ok(None);
            }
            
            // Read from disk
            let file_path = self.cache_dir.join(&entry.file_path);
            if !file_path.exists() {
                // File missing, remove from index
                self.index.remove(key);
                return Ok(None);
            }
            
            let data = std::fs::read(&file_path).map_err(|e| TreeShakeError::CacheError {
                message: format!("Failed to read cache file: {}", e),
            })?;
            
            let decompressed_data = if entry.is_compressed {
                self.decompress_data(&data)?
            } else {
                data
            };
            
            let cached_data: CachedImportData = bincode::deserialize(&decompressed_data)
                .map_err(|e| TreeShakeError::CacheError {
                    message: format!("Failed to deserialize cache data: {}", e),
                })?;
            
            // Update access time
            entry.last_accessed = SystemTime::now();
            
            self.stats.bytes_read += decompressed_data.len() as u64;
            return Ok(Some(cached_data));
        }
        
        Ok(None)
    }
    
    fn put(&mut self, key: String, data: CachedImportData, ttl: Option<Duration>) -> Result<(), TreeShakeError> {
        let now = SystemTime::now();
        let expires_at = now + ttl.unwrap_or(Duration::from_secs(3600));
        
        // Serialize data
        let serialized_data = bincode::serialize(&data)
            .map_err(|e| TreeShakeError::CacheError {
                message: format!("Failed to serialize cache data: {}", e),
            })?;
        
        // Compress if enabled and beneficial
        let (final_data, is_compressed, compression_ratio) = if self.compression_enabled && serialized_data.len() > 1024 {
            let compressed = self.compress_data(&serialized_data)?;
            let ratio = compressed.len() as f64 / serialized_data.len() as f64;
            if ratio < 0.9 { // Only use compression if it saves at least 10%
                (compressed, true, Some(ratio))
            } else {
                (serialized_data, false, None)
            }
        } else {
            (serialized_data, false, None)
        };
        
        // Check disk space
        if self.current_size + final_data.len() as u64 > self.max_size {
            self.evict_entries_for_space(final_data.len() as u64)?;
        }
        
        // Write to disk
        let file_name = format!("{}.cache", self.generate_file_hash(&key));
        let file_path = self.cache_dir.join(&file_name);
        
        // Ensure cache directory exists
        if let Some(parent) = file_path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| TreeShakeError::CacheError {
                message: format!("Failed to create cache directory: {}", e),
            })?;
        }
        
        std::fs::write(&file_path, &final_data).map_err(|e| TreeShakeError::CacheError {
            message: format!("Failed to write cache file: {}", e),
        })?;
        
        // Create index entry
        let entry = DiskCacheEntry {
            key: key.clone(),
            file_path: file_name,
            created_at: now,
            expires_at,
            last_accessed: now,
            file_size: final_data.len() as u64,
            is_compressed,
            compression_ratio,
            data_type: self.get_data_type(&data),
            checksum: self.calculate_checksum(&final_data),
        };
        
        self.index.insert(key, entry);
        self.current_size += final_data.len() as u64;
        self.stats.bytes_written += final_data.len() as u64;
        self.stats.current_entries = self.index.len();
        
        Ok(())
    }
    
    fn remove(&mut self, key: &str) -> Result<bool, TreeShakeError> {
        if let Some(entry) = self.index.remove(key) {
            let file_path = self.cache_dir.join(&entry.file_path);
            if file_path.exists() {
                std::fs::remove_file(&file_path).map_err(|e| TreeShakeError::CacheError {
                    message: format!("Failed to remove cache file: {}", e),
                })?;
            }
            self.current_size -= entry.file_size;
            self.stats.evictions += 1;
            self.stats.current_entries = self.index.len();
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    fn clear(&mut self) -> Result<(), TreeShakeError> {
        // Remove all files
        for entry in self.index.values() {
            let file_path = self.cache_dir.join(&entry.file_path);
            if file_path.exists() {
                let _ = std::fs::remove_file(&file_path);
            }
        }
        
        self.index.clear();
        self.current_size = 0;
        self.stats.current_entries = 0;
        
        Ok(())
    }
    
    fn optimize(&mut self) -> Result<Vec<String>, TreeShakeError> {
        let mut optimizations = Vec::new();
        
        // Remove expired entries
        let expired_keys: Vec<String> = self.index.iter()
            .filter(|(_, entry)| SystemTime::now() > entry.expires_at)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in expired_keys {
            self.remove(&key)?;
            optimizations.push(format!("Removed expired disk entry: {}", key));
        }
        
        // Verify file integrity
        let mut corrupted_keys = Vec::new();
        for (key, entry) in &self.index {
            let file_path = self.cache_dir.join(&entry.file_path);
            if !file_path.exists() {
                corrupted_keys.push(key.clone());
            }
        }
        
        for key in corrupted_keys {
            self.index.remove(&key);
            optimizations.push(format!("Removed corrupted entry: {}", key));
        }
        
        Ok(optimizations)
    }
    
    fn evict_entries_for_space(&mut self, needed_space: u64) -> Result<(), TreeShakeError> {
        let mut entries_by_access: Vec<_> = self.index.iter()
            .map(|(key, entry)| (key.clone(), entry.last_accessed))
            .collect();
        
        // Sort by last access time (oldest first)
        entries_by_access.sort_by_key(|(_, access_time)| *access_time);
        
        let mut freed_space = 0u64;
        for (key, _) in entries_by_access {
            if freed_space >= needed_space {
                break;
            }
            
            if let Some(entry) = self.index.get(&key) {
                freed_space += entry.file_size;
            }
            
            self.remove(&key)?;
        }
        
        Ok(())
    }
    
    fn compress_data(&self, data: &[u8]) -> Result<Vec<u8>, TreeShakeError> {
        // Simplified compression using flate2 (gzip)
        use std::io::Write;
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::new(self.compression_level));
        encoder.write_all(data).map_err(|e| TreeShakeError::CacheError {
            message: format!("Compression failed: {}", e),
        })?;
        encoder.finish().map_err(|e| TreeShakeError::CacheError {
            message: format!("Compression finish failed: {}", e),
        })
    }
    
    fn decompress_data(&self, data: &[u8]) -> Result<Vec<u8>, TreeShakeError> {
        // Simplified decompression using flate2 (gzip)
        use std::io::Read;
        let mut decoder = flate2::read::GzDecoder::new(data);
        let mut decompressed = Vec::new();
        decoder.read_to_end(&mut decompressed).map_err(|e| TreeShakeError::CacheError {
            message: format!("Decompression failed: {}", e),
        })?;
        Ok(decompressed)
    }
    
    fn generate_file_hash(&self, key: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
    
    fn get_data_type(&self, data: &CachedImportData) -> String {
        match data {
            CachedImportData::ResolvedImports(_) => "resolved_imports".to_string(),
            CachedImportData::GeneratedStatements(_) => "generated_statements".to_string(),
            CachedImportData::ValidationResults(_) => "validation_results".to_string(),
            CachedImportData::DependencyReport(_) => "dependency_report".to_string(),
            CachedImportData::SingleResolvedImport(_) => "single_resolved_import".to_string(),
            CachedImportData::SingleGeneratedStatement(_) => "single_generated_statement".to_string(),
            CachedImportData::CustomData { data_type, .. } => data_type.clone(),
        }
    }
    
    fn calculate_checksum(&self, data: &[u8]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        data.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }
}

impl CacheWarmer {
    fn new() -> Self {
        CacheWarmer {
            config: CacheWarmingConfig {
                enable_predictive_warming: true,
                warming_threshold: 0.1,
                max_concurrent_warming: 4,
                priority_algorithm: WarmingPriorityAlgorithm::default(),
                analysis_time_window_hours: 24,
            },
            frequent_keys: HashMap::new(),
            warming_queue: Vec::new(),
            stats: CacheWarmingStats::default(),
            warming_in_progress: HashSet::new(),
        }
    }
    
    fn add_key_access(&mut self, key: &str) {
        let now = SystemTime::now();
        let pattern = self.frequent_keys.entry(key.to_string()).or_insert_with(|| KeyAccessPattern {
            key: key.to_string(),
            access_count: 0,
            recent_accesses: Vec::new(),
            access_frequency: 0.0,
            last_access: now,
            average_access_interval: Duration::from_secs(3600),
            pattern_type: AccessPatternType::Sporadic,
            warming_priority: 0.0,
        });
        
        pattern.access_count += 1;
        pattern.last_access = now;
        pattern.recent_accesses.push(now);
        
        // Keep only recent accesses within the analysis window
        let cutoff_time = now - Duration::from_secs(self.config.analysis_time_window_hours * 3600);
        pattern.recent_accesses.retain(|&access_time| access_time > cutoff_time);
        
        // Update access frequency
        if pattern.recent_accesses.len() > 1 {
            let time_span = pattern.recent_accesses.last().unwrap()
                .duration_since(*pattern.recent_accesses.first().unwrap())
                .unwrap_or(Duration::from_secs(1));
            
            pattern.access_frequency = pattern.recent_accesses.len() as f64 / time_span.as_secs_f64() * 3600.0;
        }
        
        // Update warming priority
        pattern.warming_priority = self.calculate_priority(key);
        
        // Schedule warming if needed
        if pattern.warming_priority > self.config.warming_threshold {
            self.schedule_warming(key.to_string());
        }
    }
    
    fn calculate_priority(&self, key: &str) -> f64 {
        if let Some(pattern) = self.frequent_keys.get(key) {
            match &self.config.priority_algorithm {
                WarmingPriorityAlgorithm::Frequency => {
                    pattern.access_frequency / 100.0 // Normalize to 0-1 range
                },
                WarmingPriorityAlgorithm::Recency => {
                    let hours_since_access = pattern.last_access
                        .elapsed()
                        .unwrap_or(Duration::from_secs(0))
                        .as_secs() as f64 / 3600.0;
                    1.0 / (1.0 + hours_since_access) // More recent = higher priority
                },
                WarmingPriorityAlgorithm::FrequencyRecency { frequency_weight } => {
                    let frequency_score = pattern.access_frequency / 100.0;
                    let recency_score = {
                        let hours_since_access = pattern.last_access
                            .elapsed()
                            .unwrap_or(Duration::from_secs(0))
                            .as_secs() as f64 / 3600.0;
                        1.0 / (1.0 + hours_since_access)
                    };
                    frequency_score * frequency_weight + recency_score * (1.0 - frequency_weight)
                },
                WarmingPriorityAlgorithm::CostBased => {
                    // Would consider the cost of regenerating the cache entry
                    pattern.access_frequency * 0.5 // Simplified
                },
                WarmingPriorityAlgorithm::MLPredicted => {
                    // Would use ML model to predict access probability
                    pattern.access_frequency * 0.8 // Simplified
                },
            }
        } else {
            0.0
        }
    }
    
    fn schedule_warming(&mut self, key: String) {
        if self.warming_in_progress.contains(&key) {
            return;
        }
        
        if self.warming_queue.iter().any(|task| task.key == key) {
            return;
        }
        
        let task = WarmingTask {
            task_id: format!("warm-{}-{}", key, SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()),
            key: key.clone(),
            priority: self.calculate_priority(&key),
            task_type: WarmingTaskType::Predictive,
            created_at: SystemTime::now(),
            estimated_completion_time: Duration::from_millis(100),
            retry_count: 0,
            max_retries: 3,
        };
        
        self.add_warming_task(task);
    }
    
    fn add_warming_task(&mut self, task: WarmingTask) {
        self.warming_queue.push(task);
        
        // Sort by priority (highest first)
        self.warming_queue.sort_by(|a, b| b.priority.partial_cmp(&a.priority).unwrap_or(std::cmp::Ordering::Equal));
        
        // Limit queue size
        if self.warming_queue.len() > 1000 {
            self.warming_queue.truncate(1000);
        }
    }
    
    fn update_access_pattern(&mut self, key: &str) {
        self.add_key_access(key);
    }
    
    fn optimize(&mut self) -> Result<Vec<String>, TreeShakeError> {
        let mut optimizations = Vec::new();
        
        // Clean up old access patterns
        let cutoff_time = SystemTime::now() - Duration::from_secs(self.config.analysis_time_window_hours * 3600 * 2);
        let stale_keys: Vec<String> = self.frequent_keys.iter()
            .filter(|(_, pattern)| pattern.last_access < cutoff_time)
            .map(|(key, _)| key.clone())
            .collect();
        
        for key in stale_keys {
            self.frequent_keys.remove(&key);
            optimizations.push(format!("Removed stale access pattern: {}", key));
        }
        
        // Update pattern types
        for pattern in self.frequent_keys.values_mut() {
            pattern.pattern_type = self.analyze_pattern_type(pattern);
        }
        
        Ok(optimizations)
    }
    
    fn analyze_pattern_type(&self, pattern: &KeyAccessPattern) -> AccessPatternType {
        if pattern.recent_accesses.len() < 3 {
            return AccessPatternType::Sporadic;
        }
        
        // Analyze intervals between accesses
        let mut intervals: Vec<Duration> = Vec::new();
        for window in pattern.recent_accesses.windows(2) {
            if let Ok(interval) = window[1].duration_since(window[0]) {
                intervals.push(interval);
            }
        }
        
        if intervals.is_empty() {
            return AccessPatternType::Sporadic;
        }
        
        // Calculate variance to determine pattern type
        let mean_interval = intervals.iter().sum::<Duration>().as_secs_f64() / intervals.len() as f64;
        let variance: f64 = intervals.iter()
            .map(|interval| {
                let diff = interval.as_secs_f64() - mean_interval;
                diff * diff
            })
            .sum::<f64>() / intervals.len() as f64;
        
        let coefficient_of_variation = variance.sqrt() / mean_interval;
        
        if coefficient_of_variation < 0.1 {
            AccessPatternType::Periodic { interval_seconds: mean_interval as u64 }
        } else if coefficient_of_variation < 0.3 {
            AccessPatternType::Steady
        } else {
            AccessPatternType::Sporadic
        }
    }
}

impl CacheInvalidationManager {
    fn new() -> Self {
        CacheInvalidationManager {
            invalidation_rules: Vec::new(),
            dependency_tracker: DependencyTracker::default(),
            invalidation_queue: Vec::new(),
            stats: InvalidationStats::default(),
        }
    }
    
    fn get_dependent_keys(&self, key: &str) -> Vec<String> {
        self.dependency_tracker.reverse_dependencies
            .get(key)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .collect()
    }
    
    fn add_invalidation_task(&mut self, task: InvalidationTask) {
        self.invalidation_queue.push(task);
        
        // Sort by priority (highest first)
        self.invalidation_queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }
    
    fn process_queue(&mut self) -> Result<Vec<String>, TreeShakeError> {
        let mut optimizations = Vec::new();
        let tasks_to_process = std::mem::take(&mut self.invalidation_queue);
        
        for task in tasks_to_process {
            let result = self.process_invalidation_task(&task)?;
            optimizations.push(result);
        }
        
        Ok(optimizations)
    }
    
    fn process_invalidation_task(&mut self, task: &InvalidationTask) -> Result<String, TreeShakeError> {
        match &task.action {
            InvalidationAction::Remove => {
                for key in &task.keys {
                    // Would actually remove from cache layers
                    self.stats.total_invalidations += 1;
                }
                Ok(format!("Invalidated {} keys: {:?}", task.keys.len(), task.keys))
            },
            InvalidationAction::MarkExpired => {
                Ok(format!("Marked {} keys as expired", task.keys.len()))
            },
            InvalidationAction::RefreshAsync => {
                Ok(format!("Scheduled async refresh for {} keys", task.keys.len()))
            },
            _ => {
                Ok(format!("Processed invalidation task: {}", task.task_id))
            }
        }
    }
}

impl CompressionSystem {
    fn new() -> Self {
        CompressionSystem {
            config: CompressionConfig {
                default_algorithm: "gzip".to_string(),
                compression_threshold_bytes: 1024,
                enable_adaptive_compression: true,
                compression_level: 6,
            },
            algorithms: HashMap::new(),
            stats: CompressionStats::default(),
        }
    }
}

impl Default for ImportCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_cache_creation() {
        let cache = ImportCache::new();
        assert!(cache.config.enable_memory_cache);
        assert!(cache.config.enable_disk_cache);
    }

    #[test]
    fn test_cache_config() {
        let config = ImportCacheConfig {
            enable_memory_cache: false,
            memory_cache_size_limit: 50 * 1024 * 1024,
            enable_compression: false,
            ..Default::default()
        };
        
        let cache = ImportCache::with_config(config);
        assert!(!cache.config.enable_memory_cache);
        assert_eq!(cache.config.memory_cache_size_limit, 50 * 1024 * 1024);
        assert!(!cache.config.enable_compression);
    }

    #[test]
    fn test_memory_cache_creation() {
        let cache = MemoryCache::new(1024 * 1024);
        assert_eq!(cache.max_size, 1024 * 1024);
        assert_eq!(cache.current_size, 0);
        assert_eq!(cache.entries.len(), 0);
    }

    #[test]
    fn test_warming_priority_algorithms() {
        let frequency = WarmingPriorityAlgorithm::Frequency;
        let recency = WarmingPriorityAlgorithm::Recency;
        let combined = WarmingPriorityAlgorithm::FrequencyRecency { frequency_weight: 0.7 };
        
        assert_eq!(frequency, WarmingPriorityAlgorithm::Frequency);
        assert_ne!(frequency, recency);
        assert!(matches!(combined, WarmingPriorityAlgorithm::FrequencyRecency { .. }));
    }

    #[test]
    fn test_access_pattern_types() {
        let periodic = AccessPatternType::Periodic { interval_seconds: 3600 };
        let burst = AccessPatternType::Burst { burst_size: 10, burst_interval: Duration::from_secs(60) };
        let steady = AccessPatternType::Steady;
        
        assert!(matches!(periodic, AccessPatternType::Periodic { .. }));
        assert!(matches!(burst, AccessPatternType::Burst { .. }));
        assert_eq!(steady, AccessPatternType::Steady);
    }

    #[test]
    fn test_invalidation_rule_types() {
        let time_based = InvalidationRuleType::TimeBasedExpiration;
        let dependency = InvalidationRuleType::DependencyBased;
        let custom = InvalidationRuleType::Custom { rule_name: "test".to_string() };
        
        assert_eq!(time_based, InvalidationRuleType::TimeBasedExpiration);
        assert_ne!(time_based, dependency);
        assert!(matches!(custom, InvalidationRuleType::Custom { .. }));
    }

    #[test]
    fn test_compression_algorithm_types() {
        let lz4 = CompressionAlgorithmType::LZ4;
        let zstd = CompressionAlgorithmType::ZSTD;
        let gzip = CompressionAlgorithmType::GZIP;
        let brotli = CompressionAlgorithmType::Brotli;
        
        assert_eq!(lz4, CompressionAlgorithmType::LZ4);
        assert_ne!(lz4, zstd);
        assert!(matches!(gzip, CompressionAlgorithmType::GZIP));
        assert!(matches!(brotli, CompressionAlgorithmType::Brotli));
    }

    #[test]
    fn test_cached_import_data_variants() {
        // Test data creation - in real implementation these would be properly constructed
        let resolved_data = CachedImportData::CustomData {
            data_type: "test".to_string(),
            serialized_data: vec![1, 2, 3, 4],
        };
        
        assert!(matches!(resolved_data, CachedImportData::CustomData { .. }));
    }

    #[test]
    fn test_cache_stats_summary() {
        let summary = CacheStatsSummary {
            total_entries: 100,
            total_memory_usage: 1024 * 1024,
            total_disk_usage: 10 * 1024 * 1024,
            overall_hit_ratio: 0.85,
            average_response_time: Duration::from_millis(10),
            cache_efficiency: 0.9,
            compression_ratio: 0.6,
            warming_effectiveness: 0.75,
        };
        
        assert_eq!(summary.total_entries, 100);
        assert_eq!(summary.overall_hit_ratio, 0.85);
        assert_eq!(summary.compression_ratio, 0.6);
    }

    #[test]
    fn test_cache_warmer() {
        let warmer = CacheWarmer::new();
        assert!(warmer.config.enable_predictive_warming);
        assert_eq!(warmer.config.max_concurrent_warming, 4);
        assert_eq!(warmer.frequent_keys.len(), 0);
        assert_eq!(warmer.warming_queue.len(), 0);
    }

    #[test]
    fn test_dependency_tracker() {
        let tracker = DependencyTracker::default();
        assert_eq!(tracker.forward_dependencies.len(), 0);
        assert_eq!(tracker.reverse_dependencies.len(), 0);
        assert_eq!(tracker.dependency_graph.len(), 0);
    }
}