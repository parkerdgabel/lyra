//! # Lyra Concurrency System
//! 
//! A high-performance actor-based concurrency system designed for symbolic computation
//! with work-stealing schedulers, parallel pattern matching, and lock-free data structures.
//! 
//! This module targets 10-100x speedup on multi-core systems through:
//! - Actor-based concurrent evaluation 
//! - Work-stealing task scheduler
//! - Parallel pattern matching
//! - Lock-free data structures
//! - NUMA-aware memory management

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::runtime::Runtime;
use crate::vm::{Value, VmResult};
use crate::error::Result;

pub mod actor;
pub mod scheduler;
pub mod pattern_parallel;
pub mod eval_parallel;
pub mod data_structures;
pub mod vm_integration;

pub use actor::{Actor, ActorSystem, ActorMessage, ActorHandle};
pub use scheduler::{WorkStealingScheduler, Task, TaskPriority};
pub use pattern_parallel::{ParallelPatternMatcher, PatternCache};
pub use eval_parallel::{ParallelEvaluator, EvaluationContext};
pub use data_structures::{LockFreeValue, ConcurrentSymbolTable, ThreadSafeVmState};
pub use vm_integration::{ConcurrentLyraVM, ConcurrentVmFactory, PerformanceStats, BenchmarkResult};

/// Configuration for the concurrency system
#[derive(Debug, Clone)]
pub struct ConcurrencyConfig {
    /// Number of worker threads for the work-stealing scheduler
    pub worker_threads: usize,
    /// Maximum number of tasks in a worker's local queue
    pub max_local_queue_size: usize,
    /// Maximum number of tasks in the global queue
    pub max_global_queue_size: usize,
    /// Enable NUMA-aware scheduling
    pub numa_aware: bool,
    /// Pattern cache size for parallel pattern matching
    pub pattern_cache_size: usize,
    /// Maximum depth for parallel evaluation recursion
    pub max_parallel_depth: usize,
    /// Threshold for switching from sequential to parallel evaluation
    pub parallel_threshold: usize,
}

impl Default for ConcurrencyConfig {
    fn default() -> Self {
        let num_cpus = num_cpus::get();
        Self {
            worker_threads: num_cpus,
            max_local_queue_size: 1024,
            max_global_queue_size: 8192,
            numa_aware: true,
            pattern_cache_size: 4096,
            max_parallel_depth: 32,
            parallel_threshold: 100,
        }
    }
}

/// Statistics for monitoring concurrency performance
#[derive(Debug)]
pub struct ConcurrencyStats {
    /// Total number of tasks executed
    pub tasks_executed: AtomicUsize,
    /// Number of successful work steals
    pub work_steals: AtomicUsize,
    /// Number of failed work steal attempts
    pub failed_steals: AtomicUsize,
    /// Number of parallel pattern matches performed
    pub parallel_patterns: AtomicUsize,
    /// Number of parallel evaluations performed
    pub parallel_evaluations: AtomicUsize,
    /// Cache hit rate for pattern matching
    pub pattern_cache_hits: AtomicUsize,
    /// Cache miss rate for pattern matching
    pub pattern_cache_misses: AtomicUsize,
}

impl Clone for ConcurrencyStats {
    fn clone(&self) -> Self {
        Self {
            tasks_executed: AtomicUsize::new(self.tasks_executed.load(Ordering::Relaxed)),
            work_steals: AtomicUsize::new(self.work_steals.load(Ordering::Relaxed)),
            failed_steals: AtomicUsize::new(self.failed_steals.load(Ordering::Relaxed)),
            parallel_patterns: AtomicUsize::new(self.parallel_patterns.load(Ordering::Relaxed)),
            parallel_evaluations: AtomicUsize::new(self.parallel_evaluations.load(Ordering::Relaxed)),
            pattern_cache_hits: AtomicUsize::new(self.pattern_cache_hits.load(Ordering::Relaxed)),
            pattern_cache_misses: AtomicUsize::new(self.pattern_cache_misses.load(Ordering::Relaxed)),
        }
    }
}

impl Default for ConcurrencyStats {
    fn default() -> Self {
        Self {
            tasks_executed: AtomicUsize::new(0),
            work_steals: AtomicUsize::new(0),
            failed_steals: AtomicUsize::new(0),
            parallel_patterns: AtomicUsize::new(0),
            parallel_evaluations: AtomicUsize::new(0),
            pattern_cache_hits: AtomicUsize::new(0),
            pattern_cache_misses: AtomicUsize::new(0),
        }
    }
}

impl ConcurrencyStats {
    /// Get work stealing efficiency as a percentage
    pub fn work_steal_efficiency(&self) -> f64 {
        let steals = self.work_steals.load(Ordering::Relaxed) as f64;
        let failed = self.failed_steals.load(Ordering::Relaxed) as f64;
        let total = steals + failed;
        
        if total > 0.0 {
            (steals / total) * 100.0
        } else {
            0.0
        }
    }
    
    /// Get pattern cache hit rate as a percentage
    pub fn pattern_cache_hit_rate(&self) -> f64 {
        let hits = self.pattern_cache_hits.load(Ordering::Relaxed) as f64;
        let misses = self.pattern_cache_misses.load(Ordering::Relaxed) as f64;
        let total = hits + misses;
        
        if total > 0.0 {
            (hits / total) * 100.0
        } else {
            0.0
        }
    }
    
    /// Reset all statistics
    pub fn reset(&self) {
        self.tasks_executed.store(0, Ordering::Relaxed);
        self.work_steals.store(0, Ordering::Relaxed);
        self.failed_steals.store(0, Ordering::Relaxed);
        self.parallel_patterns.store(0, Ordering::Relaxed);
        self.parallel_evaluations.store(0, Ordering::Relaxed);
        self.pattern_cache_hits.store(0, Ordering::Relaxed);
        self.pattern_cache_misses.store(0, Ordering::Relaxed);
    }
}

/// Main concurrency coordinator that manages all concurrent components
pub struct ConcurrencySystem {
    /// System configuration
    config: ConcurrencyConfig,
    /// Performance statistics
    stats: Arc<ConcurrencyStats>,
    /// Work-stealing scheduler
    scheduler: Arc<WorkStealingScheduler>,
    /// Actor system for managing concurrent computation units
    actor_system: Arc<ActorSystem>,
    /// Parallel pattern matcher
    pattern_matcher: Arc<ParallelPatternMatcher>,
    /// Parallel evaluator
    evaluator: Arc<ParallelEvaluator>,
    /// Tokio runtime for async operations
    runtime: Arc<Runtime>,
}

impl ConcurrencySystem {
    /// Create a new concurrency system with default configuration
    pub fn new() -> Result<Self> {
        Self::with_config(ConcurrencyConfig::default())
    }
    
    /// Create a new concurrency system with custom configuration
    pub fn with_config(config: ConcurrencyConfig) -> Result<Self> {
        let stats = Arc::new(ConcurrencyStats::default());
        
        // Create Tokio runtime
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .worker_threads(config.worker_threads)
                .enable_all()
                .build()
                .map_err(|e| crate::error::Error::Runtime { message: format!("Failed to create async runtime: {}", e) })?
        );
        
        // Create work-stealing scheduler
        let scheduler = Arc::new(WorkStealingScheduler::new(
            config.clone(),
            Arc::clone(&stats),
        )?);
        
        // Create actor system
        let actor_system = Arc::new(ActorSystem::new(
            Arc::clone(&scheduler),
            Arc::clone(&stats),
        )?);
        
        // Create parallel pattern matcher
        let pattern_matcher = Arc::new(ParallelPatternMatcher::new(
            config.clone(),
            Arc::clone(&stats),
        )?);
        
        // Create parallel evaluator  
        let evaluator = Arc::new(ParallelEvaluator::new(
            config.clone(),
            Arc::clone(&stats),
            Arc::clone(&scheduler),
        )?);
        
        Ok(Self {
            config,
            stats,
            scheduler,
            actor_system,
            pattern_matcher,
            evaluator,
            runtime,
        })
    }
    
    /// Start the concurrency system
    pub fn start(&self) -> Result<()> {
        self.scheduler.start()?;
        self.actor_system.start()?;
        Ok(())
    }
    
    /// Stop the concurrency system gracefully
    pub async fn stop(&self) -> Result<()> {
        self.actor_system.stop().await?;
        self.scheduler.stop()?;
        Ok(())
    }
    
    /// Get a reference to the work-stealing scheduler
    pub fn scheduler(&self) -> &Arc<WorkStealingScheduler> {
        &self.scheduler
    }
    
    /// Get a reference to the actor system
    pub fn actor_system(&self) -> &Arc<ActorSystem> {
        &self.actor_system
    }
    
    /// Get a reference to the parallel pattern matcher
    pub fn pattern_matcher(&self) -> &Arc<ParallelPatternMatcher> {
        &self.pattern_matcher
    }
    
    /// Get a reference to the parallel evaluator
    pub fn evaluator(&self) -> &Arc<ParallelEvaluator> {
        &self.evaluator
    }
    
    /// Get current performance statistics
    pub fn stats(&self) -> &Arc<ConcurrencyStats> {
        &self.stats
    }
    
    /// Get system configuration
    pub fn config(&self) -> &ConcurrencyConfig {
        &self.config
    }
    
    /// Execute a task using the parallel evaluator
    pub async fn execute_parallel<F>(&self, task: F) -> VmResult<Value>
    where
        F: FnOnce() -> VmResult<Value> + Send + 'static,
    {
        self.evaluator.evaluate_async(task).await
    }
    
    /// Execute pattern matching in parallel
    pub fn match_patterns_parallel(
        &self,
        _expression: &Value,
        patterns: &[crate::ast::Pattern],
    ) -> VmResult<Vec<crate::pattern_matcher::MatchResult>> {
        // TODO: Fix thread safety issues with ParallelPatternMatcher
        // Issue: match_parallel requires &mut self but pattern_matcher is in Arc
        // For now, return placeholder result
        Ok(vec![crate::pattern_matcher::MatchResult::Failure { 
            reason: "Parallel pattern matching not yet implemented".to_string() 
        }; patterns.len()])
    }
}

impl Default for ConcurrencySystem {
    fn default() -> Self {
        Self::new().expect("Failed to create default concurrency system")
    }
}

/// Trait for objects that can be executed concurrently
pub trait ConcurrentExecutable: Send + Sync {
    type Output: Send;
    type Error: Send;
    
    /// Execute the task concurrently
    fn execute(&self) -> std::result::Result<Self::Output, Self::Error>;
    
    /// Get the priority of this task
    fn priority(&self) -> TaskPriority {
        TaskPriority::Normal
    }
    
    /// Check if this task can be executed in parallel with others
    fn is_parallelizable(&self) -> bool {
        true
    }
    
    /// Get an estimate of the computational cost of this task
    fn cost_estimate(&self) -> usize {
        1
    }
}

/// Error types specific to the concurrency system
#[derive(Debug, thiserror::Error)]
pub enum ConcurrencyError {
    #[error("Scheduler error: {0}")]
    Scheduler(String),
    
    #[error("Actor system error: {0}")]
    ActorSystem(String),
    
    #[error("Pattern matching error: {0}")]
    PatternMatching(String),
    
    #[error("Evaluation error: {0}")]
    Evaluation(String),
    
    #[error("Runtime error: {0}")]
    Runtime(String),
    
    #[error("Configuration error: {0}")]
    Configuration(String),
}

impl From<ConcurrencyError> for crate::error::Error {
    fn from(err: ConcurrencyError) -> Self {
        crate::error::Error::Runtime { message: err.to_string() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_concurrency_config_default() {
        let config = ConcurrencyConfig::default();
        assert!(config.worker_threads > 0);
        assert!(config.max_local_queue_size > 0);
        assert!(config.max_global_queue_size > 0);
    }
    
    #[test]
    fn test_concurrency_stats() {
        let stats = ConcurrencyStats::default();
        assert_eq!(stats.work_steal_efficiency(), 0.0);
        assert_eq!(stats.pattern_cache_hit_rate(), 0.0);
        
        stats.work_steals.store(70, Ordering::Relaxed);
        stats.failed_steals.store(30, Ordering::Relaxed);
        assert_eq!(stats.work_steal_efficiency(), 70.0);
        
        stats.pattern_cache_hits.store(80, Ordering::Relaxed);
        stats.pattern_cache_misses.store(20, Ordering::Relaxed);
        assert_eq!(stats.pattern_cache_hit_rate(), 80.0);
    }
    
    #[tokio::test]
    async fn test_concurrency_system_creation() {
        let system = ConcurrencySystem::new();
        assert!(system.is_ok());
        
        let system = system.unwrap();
        assert_eq!(system.config().worker_threads, num_cpus::get());
    }
}
