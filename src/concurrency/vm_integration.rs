//! # VM Integration for Concurrency
//! 
//! Integrates the concurrency system with the existing Lyra VM,
//! providing concurrent execution capabilities while maintaining compatibility.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Instant;
use tokio::sync::RwLock;

use crate::vm::{VirtualMachine, Value, VmResult, VmError};
use crate::bytecode::{Instruction, OpCode};
use crate::compiler::Compiler;
use crate::ast::{Expr, Pattern};
use crate::pattern_matcher::{MatchResult, MatchingContext};

use super::{
    ConcurrencySystem, ConcurrencyConfig, ConcurrencyStats,
    ActorSystem, ParallelPatternMatcher, ParallelEvaluator,
    WorkStealingScheduler, EvaluationContext, TaskPriority,
    data_structures::{ThreadSafeVmState, ConcurrentVirtualMachine},
};

/// Enhanced VM with concurrency capabilities
pub struct ConcurrentLyraVM {
    /// Original sequential VM for compatibility
    sequential_vm: VirtualMachine,
    /// Concurrent VM for parallel execution
    concurrent_vm: ConcurrentVirtualMachine,
    /// Concurrency system
    concurrency_system: Arc<ConcurrencySystem>,
    /// Configuration
    config: ConcurrencyConfig,
    /// Whether concurrent execution is enabled
    concurrent_enabled: bool,
    /// Parallel execution threshold
    parallel_threshold: usize,
}

impl ConcurrentLyraVM {
    /// Create a new concurrent Lyra VM
    pub fn new() -> Result<Self, crate::error::Error> {
        let config = ConcurrencyConfig::default();
        let concurrency_system = Arc::new(ConcurrencySystem::with_config(config.clone())?);
        let sequential_vm = VirtualMachine::new();
        let concurrent_vm = ConcurrentVirtualMachine::new(concurrency_system.stats().clone());
        
        Ok(Self {
            sequential_vm,
            concurrent_vm,
            concurrency_system,
            config,
            concurrent_enabled: true,
            parallel_threshold: 100,
        })
    }
    
    /// Create with custom configuration
    pub fn with_config(config: ConcurrencyConfig) -> Result<Self, crate::error::Error> {
        let concurrency_system = Arc::new(ConcurrencySystem::with_config(config.clone())?);
        let sequential_vm = VirtualMachine::new();
        let concurrent_vm = ConcurrentVirtualMachine::new(concurrency_system.stats().clone());
        
        Ok(Self {
            sequential_vm,
            concurrent_vm,
            concurrency_system,
            config,
            concurrent_enabled: true,
            parallel_threshold: config.parallel_threshold,
        })
    }
    
    /// Start the concurrency system
    pub fn start(&self) -> Result<(), crate::error::Error> {
        self.concurrency_system.start().map_err(|e| e.into())
    }
    
    /// Stop the concurrency system
    pub fn stop(&self) -> Result<(), crate::error::Error> {
        self.concurrency_system.stop().map_err(|e| e.into())
    }
    
    /// Load bytecode into both VMs
    pub fn load(&mut self, code: Vec<Instruction>, constants: Vec<Value>) {
        self.sequential_vm.load(code.clone(), constants.clone());
        self.concurrent_vm.state().load(code, constants);
    }
    
    /// Execute with automatic concurrency selection
    pub async fn execute_auto(&mut self, expression: &Expr) -> VmResult<Value> {
        if self.should_use_concurrent_execution(expression) {
            self.execute_concurrent(expression).await
        } else {
            self.execute_sequential(expression)
        }
    }
    
    /// Execute using sequential VM
    pub fn execute_sequential(&mut self, expression: &Expr) -> VmResult<Value> {
        // Compile expression to bytecode and execute
        let mut compiler = Compiler::new();
        let bytecode = compiler.compile_expression(expression)
            .map_err(|_| VmError::TypeError {
                expected: "valid bytecode".to_string(),
                actual: "compilation failed".to_string(),
            })?;
        
        self.sequential_vm.load(bytecode.instructions, bytecode.constants);
        self.sequential_vm.run()
    }
    
    /// Execute using concurrent VM
    pub async fn execute_concurrent(&self, expression: &Expr) -> VmResult<Value> {
        let context = EvaluationContext::new();
        
        // Use the parallel evaluator
        self.concurrency_system
            .evaluator()
            .evaluate_parallel(expression, &context)
    }
    
    /// Execute pattern matching concurrently
    pub fn match_patterns_concurrent(
        &self,
        expression: &Value,
        patterns: &[Pattern],
    ) -> VmResult<Vec<MatchResult>> {
        self.concurrency_system
            .pattern_matcher()
            .match_parallel(expression, patterns)
    }
    
    /// Execute a batch of expressions in parallel
    pub async fn execute_batch_parallel(
        &self,
        expressions: &[Expr],
    ) -> VmResult<Vec<Value>> {
        let context = EvaluationContext::new();
        
        // Use rayon for parallel execution
        use rayon::prelude::*;
        
        let results: Result<Vec<_>, _> = expressions
            .par_iter()
            .map(|expr| {
                // Create a blocking task for each expression
                self.concurrency_system
                    .evaluator()
                    .evaluate_parallel(expr, &context)
            })
            .collect();
        
        results
    }
    
    /// Check if concurrent execution should be used
    fn should_use_concurrent_execution(&self, expression: &Expr) -> bool {
        if !self.concurrent_enabled {
            return false;
        }
        
        match expression {
            Expr::List(items) => items.len() >= self.parallel_threshold,
            Expr::Function { head: _, args } => args.len() >= self.parallel_threshold,
            _ => false,
        }
    }
    
    /// Enable or disable concurrent execution
    pub fn set_concurrent_enabled(&mut self, enabled: bool) {
        self.concurrent_enabled = enabled;
    }
    
    /// Set parallel execution threshold
    pub fn set_parallel_threshold(&mut self, threshold: usize) {
        self.parallel_threshold = threshold;
    }
    
    /// Get performance statistics
    pub fn get_performance_stats(&self) -> PerformanceStats {
        let concurrency_stats = self.concurrency_system.stats();
        let scheduler_stats = self.concurrency_system.scheduler().stats();
        let cache_stats = self.concurrency_system.pattern_matcher().cache_stats();
        
        PerformanceStats {
            tasks_executed: concurrency_stats.tasks_executed.load(std::sync::atomic::Ordering::Relaxed),
            work_steals: concurrency_stats.work_steals.load(std::sync::atomic::Ordering::Relaxed),
            failed_steals: concurrency_stats.failed_steals.load(std::sync::atomic::Ordering::Relaxed),
            parallel_patterns: concurrency_stats.parallel_patterns.load(std::sync::atomic::Ordering::Relaxed),
            parallel_evaluations: concurrency_stats.parallel_evaluations.load(std::sync::atomic::Ordering::Relaxed),
            pattern_cache_hit_rate: cache_stats.hit_rate,
            work_steal_efficiency: concurrency_stats.work_steal_efficiency(),
            worker_count: scheduler_stats.worker_count,
            global_queue_size: scheduler_stats.global_queue_size,
        }
    }
    
    /// Get a reference to the concurrency system
    pub fn concurrency_system(&self) -> &Arc<ConcurrencySystem> {
        &self.concurrency_system
    }
    
    /// Get a reference to the sequential VM
    pub fn sequential_vm(&mut self) -> &mut VirtualMachine {
        &mut self.sequential_vm
    }
    
    /// Get a reference to the concurrent VM
    pub fn concurrent_vm(&self) -> &ConcurrentVirtualMachine {
        &self.concurrent_vm
    }
    
    /// Benchmark concurrent vs sequential execution
    pub async fn benchmark_execution(&mut self, expression: &Expr, iterations: usize) -> BenchmarkResult {
        let mut sequential_times = Vec::new();
        let mut concurrent_times = Vec::new();
        
        // Benchmark sequential execution
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = self.execute_sequential(expression);
            sequential_times.push(start.elapsed());
        }
        
        // Benchmark concurrent execution
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = self.execute_concurrent(expression).await;
            concurrent_times.push(start.elapsed());
        }
        
        let sequential_avg = sequential_times.iter().sum::<std::time::Duration>() / iterations as u32;
        let concurrent_avg = concurrent_times.iter().sum::<std::time::Duration>() / iterations as u32;
        
        let speedup = if concurrent_avg.as_nanos() > 0 {
            sequential_avg.as_nanos() as f64 / concurrent_avg.as_nanos() as f64
        } else {
            1.0
        };
        
        BenchmarkResult {
            sequential_avg_time: sequential_avg,
            concurrent_avg_time: concurrent_avg,
            speedup,
            iterations,
        }
    }
}

impl Default for ConcurrentLyraVM {
    fn default() -> Self {
        Self::new().expect("Failed to create default ConcurrentLyraVM")
    }
}

/// Performance statistics for the concurrent VM
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Total tasks executed
    pub tasks_executed: usize,
    /// Successful work steals
    pub work_steals: usize,
    /// Failed work steal attempts
    pub failed_steals: usize,
    /// Parallel pattern matches
    pub parallel_patterns: usize,
    /// Parallel evaluations
    pub parallel_evaluations: usize,
    /// Pattern cache hit rate
    pub pattern_cache_hit_rate: f64,
    /// Work stealing efficiency
    pub work_steal_efficiency: f64,
    /// Number of worker threads
    pub worker_count: usize,
    /// Global queue size
    pub global_queue_size: usize,
}

impl PerformanceStats {
    /// Calculate overall efficiency score (0-100)
    pub fn efficiency_score(&self) -> f64 {
        let cache_weight = 0.3;
        let steal_weight = 0.3;
        let utilization_weight = 0.4;
        
        let cache_score = self.pattern_cache_hit_rate;
        let steal_score = self.work_steal_efficiency;
        let utilization_score = if self.tasks_executed > 0 {
            (self.parallel_evaluations as f64 / self.tasks_executed as f64) * 100.0
        } else {
            0.0
        };
        
        (cache_score * cache_weight + 
         steal_score * steal_weight + 
         utilization_score * utilization_weight)
    }
}

/// Benchmark results comparing sequential vs concurrent execution
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Average sequential execution time
    pub sequential_avg_time: std::time::Duration,
    /// Average concurrent execution time
    pub concurrent_avg_time: std::time::Duration,
    /// Speedup factor (sequential_time / concurrent_time)
    pub speedup: f64,
    /// Number of iterations
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Check if concurrent execution provides significant speedup
    pub fn is_concurrent_beneficial(&self) -> bool {
        self.speedup > 1.1 // At least 10% improvement
    }
    
    /// Get speedup as a percentage
    pub fn speedup_percentage(&self) -> f64 {
        (self.speedup - 1.0) * 100.0
    }
}

/// Factory for creating optimized VM instances
pub struct ConcurrentVmFactory;

impl ConcurrentVmFactory {
    /// Create a VM optimized for mathematical computation
    pub fn create_math_optimized() -> Result<ConcurrentLyraVM, crate::error::Error> {
        let config = ConcurrencyConfig {
            worker_threads: num_cpus::get(),
            parallel_threshold: 50, // Lower threshold for math operations
            pattern_cache_size: 8192, // Larger cache for mathematical patterns
            numa_aware: true,
            ..Default::default()
        };
        
        ConcurrentLyraVM::with_config(config)
    }
    
    /// Create a VM optimized for symbolic manipulation
    pub fn create_symbolic_optimized() -> Result<ConcurrentLyraVM, crate::error::Error> {
        let config = ConcurrencyConfig {
            worker_threads: num_cpus::get() * 2, // More threads for symbolic work
            parallel_threshold: 20, // Lower threshold for complex expressions
            pattern_cache_size: 16384, // Large cache for patterns
            max_parallel_depth: 64, // Deeper parallelization
            ..Default::default()
        };
        
        ConcurrentLyraVM::with_config(config)
    }
    
    /// Create a VM optimized for list processing
    pub fn create_list_optimized() -> Result<ConcurrentLyraVM, crate::error::Error> {
        let config = ConcurrencyConfig {
            worker_threads: num_cpus::get(),
            parallel_threshold: 10, // Very low threshold for lists
            pattern_cache_size: 4096,
            max_local_queue_size: 2048, // Larger queues for list items
            ..Default::default()
        };
        
        ConcurrentLyraVM::with_config(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Expr;
    
    #[tokio::test]
    async fn test_concurrent_vm_creation() {
        let vm = ConcurrentLyraVM::new();
        assert!(vm.is_ok());
        
        let mut vm = vm.unwrap();
        assert!(vm.start().is_ok());
        assert!(vm.stop().is_ok());
    }
    
    #[tokio::test]
    async fn test_sequential_execution() {
        let mut vm = ConcurrentLyraVM::new().unwrap();
        vm.start().unwrap();
        
        let expr = Expr::Number(crate::ast::Number::Integer(42));
        let result = vm.execute_sequential(&expr);
        assert!(result.is_ok());
        
        vm.stop().unwrap();
    }
    
    #[tokio::test]
    async fn test_concurrent_execution() {
        let vm = ConcurrentLyraVM::new().unwrap();
        vm.start().unwrap();
        
        let expr = Expr::Number(crate::ast::Number::Integer(42));
        let result = vm.execute_concurrent(&expr).await;
        assert!(result.is_ok());
        
        vm.stop().unwrap();
    }
    
    #[tokio::test]
    async fn test_batch_execution() {
        let vm = ConcurrentLyraVM::new().unwrap();
        vm.start().unwrap();
        
        let expressions = vec![
            Expr::Number(crate::ast::Number::Integer(1)),
            Expr::Number(crate::ast::Number::Integer(2)),
            Expr::Number(crate::ast::Number::Integer(3)),
        ];
        
        let results = vm.execute_batch_parallel(&expressions).await;
        assert!(results.is_ok());
        
        let results = results.unwrap();
        assert_eq!(results.len(), 3);
        
        vm.stop().unwrap();
    }
    
    #[test]
    fn test_performance_stats() {
        let stats = PerformanceStats {
            tasks_executed: 1000,
            work_steals: 80,
            failed_steals: 20,
            parallel_patterns: 500,
            parallel_evaluations: 800,
            pattern_cache_hit_rate: 85.0,
            work_steal_efficiency: 80.0,
            worker_count: 8,
            global_queue_size: 50,
        };
        
        let efficiency = stats.efficiency_score();
        assert!(efficiency > 0.0 && efficiency <= 100.0);
    }
    
    #[test]
    fn test_vm_factory() {
        let math_vm = ConcurrentVmFactory::create_math_optimized();
        assert!(math_vm.is_ok());
        
        let symbolic_vm = ConcurrentVmFactory::create_symbolic_optimized();
        assert!(symbolic_vm.is_ok());
        
        let list_vm = ConcurrentVmFactory::create_list_optimized();
        assert!(list_vm.is_ok());
    }
    
    #[tokio::test]
    async fn test_auto_execution_selection() {
        let mut vm = ConcurrentLyraVM::new().unwrap();
        vm.start().unwrap();
        
        // Small expression should use sequential
        let small_expr = Expr::Integer(42);
        let result = vm.execute_auto(&small_expr).await;
        assert!(result.is_ok());
        
        // Large list should potentially use concurrent
        let large_list = Expr::List(vec![Expr::Number(crate::ast::Number::Integer(1)); 200]);
        let result = vm.execute_auto(&large_list).await;
        assert!(result.is_ok());
        
        vm.stop().unwrap();
    }
}