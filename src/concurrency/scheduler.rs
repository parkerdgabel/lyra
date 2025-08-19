//! # Work-Stealing Task Scheduler
//! 
//! High-performance work-stealing scheduler optimized for symbolic computation workloads.
//! Features lock-free queues, NUMA-aware scheduling, and dynamic load balancing.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use crossbeam_deque::{Injector, Stealer, Worker as CrossbeamWorker};
use crossbeam_utils::Backoff;
use parking_lot::{Mutex, RwLock};
use once_cell::sync::Lazy;

use crate::vm::{Value, VmResult, VmError};
use super::{ConcurrencyConfig, ConcurrencyStats, ConcurrencyError, ConcurrentExecutable};

/// Priority levels for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 1,
    Normal = 2,
    High = 3,
    Critical = 4,
}

/// A task that can be executed by the scheduler
pub struct Task {
    /// Unique task ID
    pub id: TaskId,
    /// Task priority
    pub priority: TaskPriority,
    /// The actual computation to perform
    pub computation: Box<dyn ConcurrentExecutable<Output = Value, Error = VmError> + Send>,
    /// When the task was created
    pub created_at: Instant,
    /// Dependencies that must complete before this task
    pub dependencies: Vec<TaskId>,
    /// Whether this task can be stolen by other workers
    pub stealable: bool,
    /// Estimated cost for load balancing
    pub cost_estimate: usize,
}

// Ensure Task can be safely sent between threads
unsafe impl Send for Task {}
unsafe impl Sync for Task {}

/// Unique identifier for tasks
pub type TaskId = usize;

/// Result of task execution
#[derive(Debug)]
pub struct TaskResult {
    /// Task ID
    pub id: TaskId,
    /// Execution result
    pub result: VmResult<Value>,
    /// Execution time
    pub duration: Duration,
    /// Worker ID that executed the task
    pub worker_id: WorkerId,
}

/// Unique identifier for worker threads
pub type WorkerId = usize;

/// Statistics for individual workers
#[derive(Debug, Default)]
pub struct WorkerStats {
    /// Tasks executed by this worker
    pub tasks_executed: AtomicUsize,
    /// Tasks stolen from this worker
    pub tasks_stolen: AtomicUsize,
    /// Tasks this worker stole from others
    pub tasks_stolen_from_others: AtomicUsize,
    /// Total execution time
    pub total_execution_time: AtomicUsize, // In microseconds
    /// Number of failed steal attempts
    pub failed_steals: AtomicUsize,
    /// Current queue size
    pub queue_size: AtomicUsize,
}

impl WorkerStats {
    /// Get average task execution time in microseconds
    pub fn average_execution_time(&self) -> f64 {
        let total_time = self.total_execution_time.load(Ordering::Relaxed) as f64;
        let task_count = self.tasks_executed.load(Ordering::Relaxed) as f64;
        
        if task_count > 0.0 {
            total_time / task_count
        } else {
            0.0
        }
    }
    
    /// Get steal success rate as a percentage
    pub fn steal_success_rate(&self) -> f64 {
        let stolen = self.tasks_stolen_from_others.load(Ordering::Relaxed) as f64;
        let failed = self.failed_steals.load(Ordering::Relaxed) as f64;
        let total = stolen + failed;
        
        if total > 0.0 {
            (stolen / total) * 100.0
        } else {
            0.0
        }
    }
}

/// A worker thread that executes tasks
struct Worker {
    /// Worker ID
    id: WorkerId,
    /// Local task queue (LIFO for cache locality)
    local_queue: CrossbeamWorker<Task>,
    /// Stealer for this worker's queue
    stealer: crossbeam_deque::Stealer<Task>,
    /// Global task queue for load balancing
    global_queue: Arc<Injector<Task>>,
    /// Stealers from other workers
    other_stealers: Vec<Stealer<Task>>,
    /// Configuration
    config: ConcurrencyConfig,
    /// Statistics
    stats: Arc<ConcurrencyStats>,
    /// Worker-specific statistics
    worker_stats: Arc<WorkerStats>,
    /// Thread handle
    handle: Option<JoinHandle<()>>,
    /// Whether the worker should continue running
    running: Arc<AtomicBool>,
    /// NUMA node for this worker (if NUMA-aware)
    numa_node: Option<usize>,
}

impl Worker {
    /// Create a new worker
    fn new(
        id: WorkerId,
        global_queue: Arc<Injector<Task>>,
        config: ConcurrencyConfig,
        stats: Arc<ConcurrencyStats>,
        numa_node: Option<usize>,
    ) -> Self {
        let local_queue = CrossbeamWorker::new_fifo();
        let stealer = local_queue.stealer();
        
        Self {
            id,
            local_queue,
            stealer,
            global_queue,
            other_stealers: Vec::new(),
            config,
            stats,
            worker_stats: Arc::new(WorkerStats::default()),
            handle: None,
            running: Arc::new(AtomicBool::new(false)),
            numa_node,
        }
    }
    
    /// Start the worker thread
    fn start(&mut self, other_stealers: Vec<Stealer<Task>>) {
        self.other_stealers = other_stealers;
        self.running.store(true, Ordering::Relaxed);
        
        let id = self.id;
        let local_queue = self.local_queue.clone();
        let global_queue = Arc::clone(&self.global_queue);
        let other_stealers = self.other_stealers.clone();
        let config = self.config.clone();
        let stats = Arc::clone(&self.stats);
        let worker_stats = Arc::clone(&self.worker_stats);
        let running = Arc::clone(&self.running);
        let numa_node = self.numa_node;
        
        let handle = thread::Builder::new()
            .name(format!("lyra-worker-{}", id))
            .spawn(move || {
                // Set thread affinity if NUMA-aware
                if let Some(node) = numa_node {
                    Self::set_thread_affinity(node);
                }
                
                let backoff = Backoff::new();
                
                while running.load(Ordering::Relaxed) {
                    // Try to find a task to execute
                    if let Some(task) = Self::find_task(&local_queue, &global_queue, &other_stealers, &worker_stats) {
                        let start_time = Instant::now();
                        
                        // Execute the task
                        let result = task.computation.execute();
                        let duration = start_time.elapsed();
                        
                        // Update statistics
                        worker_stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
                        worker_stats.total_execution_time.fetch_add(
                            duration.as_micros() as usize,
                            Ordering::Relaxed,
                        );
                        stats.tasks_executed.fetch_add(1, Ordering::Relaxed);
                        
                        // Reset backoff on successful work
                        backoff.reset();
                    } else {
                        // No work found, back off
                        backoff.snooze();
                        
                        // If we've backed off too much, yield the thread
                        if backoff.is_completed() {
                            thread::yield_now();
                        }
                    }
                }
            })
            .expect("Failed to spawn worker thread");
        
        self.handle = Some(handle);
    }
    
    /// Try to find a task to execute
    fn find_task(
        local_queue: &CrossbeamWorker<Task>,
        global_queue: &Injector<Task>,
        other_stealers: &[Stealer<Task>],
        worker_stats: &WorkerStats,
    ) -> Option<Task> {
        // 1. Try local queue first (LIFO for cache locality)
        if let Some(task) = local_queue.pop() {
            worker_stats.queue_size.fetch_sub(1, Ordering::Relaxed);
            return Some(task);
        }
        
        // 2. Try global queue
        if let Some(task) = global_queue.steal().success() {
            return Some(task);
        }
        
        // 3. Try stealing from other workers
        for stealer in other_stealers {
            if let Some(task) = stealer.steal().success() {
                worker_stats.tasks_stolen_from_others.fetch_add(1, Ordering::Relaxed);
                return Some(task);
            } else {
                worker_stats.failed_steals.fetch_add(1, Ordering::Relaxed);
            }
        }
        
        None
    }
    
    /// Set thread affinity for NUMA awareness
    fn set_thread_affinity(_numa_node: usize) {
        // Platform-specific implementation would go here
        // For now, this is a no-op
    }
    
    /// Push a task to the local queue
    fn push_local(&self, task: Task) {
        self.local_queue.push(task);
        self.worker_stats.queue_size.fetch_add(1, Ordering::Relaxed);
    }
    
    /// Stop the worker
    fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

/// Work-stealing scheduler for concurrent task execution
pub struct WorkStealingScheduler {
    /// Configuration
    config: ConcurrencyConfig,
    /// Global task queue
    global_queue: Arc<Injector<Task>>,
    /// Worker threads
    workers: RwLock<Vec<Worker>>,
    /// Whether the scheduler is running
    running: AtomicBool,
    /// Next task ID
    next_task_id: AtomicUsize,
    /// Statistics
    stats: Arc<ConcurrencyStats>,
    /// Task dependency tracking
    dependency_graph: Arc<Mutex<DependencyGraph>>,
}

/// Tracks task dependencies
#[derive(Debug, Default)]
struct DependencyGraph {
    /// Maps task ID to its dependencies
    dependencies: std::collections::HashMap<TaskId, Vec<TaskId>>,
    /// Maps task ID to tasks that depend on it
    dependents: std::collections::HashMap<TaskId, Vec<TaskId>>,
    /// Completed tasks
    completed: std::collections::HashSet<TaskId>,
}

impl DependencyGraph {
    /// Add a dependency relationship
    fn add_dependency(&mut self, task_id: TaskId, depends_on: TaskId) {
        self.dependencies.entry(task_id).or_default().push(depends_on);
        self.dependents.entry(depends_on).or_default().push(task_id);
    }
    
    /// Mark a task as completed and return newly ready tasks
    fn complete_task(&mut self, task_id: TaskId) -> Vec<TaskId> {
        self.completed.insert(task_id);
        
        let mut ready_tasks = Vec::new();
        
        if let Some(dependents) = self.dependents.get(&task_id) {
            for &dependent in dependents {
                if self.is_ready(dependent) {
                    ready_tasks.push(dependent);
                }
            }
        }
        
        ready_tasks
    }
    
    /// Check if a task is ready to execute (all dependencies completed)
    fn is_ready(&self, task_id: TaskId) -> bool {
        if let Some(deps) = self.dependencies.get(&task_id) {
            deps.iter().all(|&dep| self.completed.contains(&dep))
        } else {
            true // No dependencies
        }
    }
}

impl WorkStealingScheduler {
    /// Create a new work-stealing scheduler
    pub fn new(
        config: ConcurrencyConfig,
        stats: Arc<ConcurrencyStats>,
    ) -> Result<Self, ConcurrencyError> {
        let global_queue = Arc::new(Injector::new());
        let workers = RwLock::new(Vec::new());
        
        Ok(Self {
            config,
            global_queue,
            workers,
            running: AtomicBool::new(false),
            next_task_id: AtomicUsize::new(1),
            stats,
            dependency_graph: Arc::new(Mutex::new(DependencyGraph::default())),
        })
    }
    
    /// Start the scheduler
    pub fn start(&self) -> Result<(), ConcurrencyError> {
        if self.running.swap(true, Ordering::Relaxed) {
            return Err(ConcurrencyError::Scheduler("Scheduler already running".to_string()));
        }
        
        let mut workers = self.workers.write();
        
        // Create worker threads
        for i in 0..self.config.worker_threads {
            let numa_node = if self.config.numa_aware {
                Some(i % Self::get_numa_node_count())
            } else {
                None
            };
            
            let mut worker = Worker::new(
                i,
                Arc::clone(&self.global_queue),
                self.config.clone(),
                Arc::clone(&self.stats),
                numa_node,
            );
            
            workers.push(worker);
        }
        
        // Collect stealers
        let stealers: Vec<_> = workers.iter().map(|w| w.stealer.clone()).collect();
        
        // Start each worker with access to other workers' stealers
        for (i, worker) in workers.iter_mut().enumerate() {
            let other_stealers: Vec<_> = stealers.iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(_, stealer)| stealer.clone())
                .collect();
            
            worker.start(other_stealers);
        }
        
        Ok(())
    }
    
    /// Stop the scheduler
    pub fn stop(&self) -> Result<(), ConcurrencyError> {
        if !self.running.swap(false, Ordering::Relaxed) {
            return Ok(()); // Already stopped
        }
        
        let mut workers = self.workers.write();
        for worker in workers.iter_mut() {
            worker.stop();
        }
        workers.clear();
        
        Ok(())
    }
    
    /// Submit a task for execution
    pub fn submit<T>(&self, computation: T) -> Result<TaskId, ConcurrencyError>
    where
        T: ConcurrentExecutable<Output = Value, Error = VmError> + 'static,
    {
        if !self.running.load(Ordering::Relaxed) {
            return Err(ConcurrencyError::Scheduler("Scheduler not running".to_string()));
        }
        
        let task_id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        
        let task = Task {
            id: task_id,
            priority: computation.priority(),
            computation: Box::new(computation),
            created_at: Instant::now(),
            dependencies: Vec::new(),
            stealable: computation.is_parallelizable(),
            cost_estimate: computation.cost_estimate(),
        };
        
        // Try to push to a worker's local queue first for cache locality
        if let Some(worker) = self.find_least_loaded_worker() {
            worker.push_local(task);
        } else {
            // Fall back to global queue
            self.global_queue.push(task);
        }
        
        Ok(task_id)
    }
    
    /// Submit a task with dependencies
    pub fn submit_with_dependencies<T>(
        &self,
        computation: T,
        dependencies: Vec<TaskId>,
    ) -> Result<TaskId, ConcurrencyError>
    where
        T: ConcurrentExecutable<Output = Value, Error = VmError> + 'static,
    {
        let task_id = self.submit(computation)?;
        
        // Add dependencies to the graph
        let mut graph = self.dependency_graph.lock();
        for dep in dependencies {
            graph.add_dependency(task_id, dep);
        }
        
        Ok(task_id)
    }
    
    /// Find the worker with the smallest queue
    fn find_least_loaded_worker(&self) -> Option<&Worker> {
        let workers = self.workers.read();
        workers.iter()
            .min_by_key(|w| w.worker_stats.queue_size.load(Ordering::Relaxed))
    }
    
    /// Get the number of NUMA nodes on the system
    fn get_numa_node_count() -> usize {
        // Platform-specific implementation would go here
        // For now, assume 2 NUMA nodes
        2
    }
    
    /// Get scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        let workers = self.workers.read();
        let worker_stats: Vec<_> = workers.iter()
            .map(|w| WorkerStatsSnapshot {
                worker_id: w.id,
                tasks_executed: w.worker_stats.tasks_executed.load(Ordering::Relaxed),
                tasks_stolen: w.worker_stats.tasks_stolen.load(Ordering::Relaxed),
                tasks_stolen_from_others: w.worker_stats.tasks_stolen_from_others.load(Ordering::Relaxed),
                average_execution_time: w.worker_stats.average_execution_time(),
                steal_success_rate: w.worker_stats.steal_success_rate(),
                queue_size: w.worker_stats.queue_size.load(Ordering::Relaxed),
            })
            .collect();
        
        SchedulerStats {
            worker_count: workers.len(),
            global_queue_size: self.global_queue.len(),
            worker_stats,
            total_tasks_executed: self.stats.tasks_executed.load(Ordering::Relaxed),
            total_work_steals: self.stats.work_steals.load(Ordering::Relaxed),
            total_failed_steals: self.stats.failed_steals.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of scheduler statistics
#[derive(Debug)]
pub struct SchedulerStats {
    /// Number of worker threads
    pub worker_count: usize,
    /// Current global queue size
    pub global_queue_size: usize,
    /// Per-worker statistics
    pub worker_stats: Vec<WorkerStatsSnapshot>,
    /// Total tasks executed across all workers
    pub total_tasks_executed: usize,
    /// Total successful work steals
    pub total_work_steals: usize,
    /// Total failed work steal attempts
    pub total_failed_steals: usize,
}

/// Snapshot of worker statistics
#[derive(Debug)]
pub struct WorkerStatsSnapshot {
    /// Worker ID
    pub worker_id: WorkerId,
    /// Tasks executed by this worker
    pub tasks_executed: usize,
    /// Tasks stolen from this worker
    pub tasks_stolen: usize,
    /// Tasks this worker stole from others
    pub tasks_stolen_from_others: usize,
    /// Average task execution time in microseconds
    pub average_execution_time: f64,
    /// Steal success rate as a percentage
    pub steal_success_rate: f64,
    /// Current queue size
    pub queue_size: usize,
}

/// Simple computation for testing
pub struct SimpleComputation {
    pub value: i64,
    pub priority: TaskPriority,
}

impl ConcurrentExecutable for SimpleComputation {
    type Output = Value;
    type Error = VmError;
    
    fn execute(&self) -> Result<Self::Output, Self::Error> {
        // Simulate some work
        thread::sleep(Duration::from_micros(10));
        Ok(Value::Integer(self.value * 2))
    }
    
    fn priority(&self) -> TaskPriority {
        self.priority
    }
    
    fn cost_estimate(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;
    
    #[test]
    fn test_scheduler_creation() {
        let config = ConcurrencyConfig::default();
        let stats = Arc::new(ConcurrencyStats::default());
        
        let scheduler = WorkStealingScheduler::new(config, stats);
        assert!(scheduler.is_ok());
    }
    
    #[test]
    fn test_scheduler_start_stop() {
        let config = ConcurrencyConfig {
            worker_threads: 2,
            ..Default::default()
        };
        let stats = Arc::new(ConcurrencyStats::default());
        
        let scheduler = WorkStealingScheduler::new(config, stats).unwrap();
        
        assert!(scheduler.start().is_ok());
        assert!(scheduler.start().is_err()); // Already running
        
        thread::sleep(Duration::from_millis(100));
        
        assert!(scheduler.stop().is_ok());
        assert!(scheduler.stop().is_ok()); // Already stopped
    }
    
    #[test]
    fn test_task_submission() {
        let config = ConcurrencyConfig {
            worker_threads: 1,
            ..Default::default()
        };
        let stats = Arc::new(ConcurrencyStats::default());
        
        let scheduler = WorkStealingScheduler::new(config, stats).unwrap();
        scheduler.start().unwrap();
        
        let computation = SimpleComputation {
            value: 21,
            priority: TaskPriority::Normal,
        };
        
        let task_id = scheduler.submit(computation);
        assert!(task_id.is_ok());
        
        thread::sleep(Duration::from_millis(100));
        
        scheduler.stop().unwrap();
    }
    
    #[test]
    fn test_scheduler_stats() {
        let config = ConcurrencyConfig {
            worker_threads: 2,
            ..Default::default()
        };
        let stats = Arc::new(ConcurrencyStats::default());
        
        let scheduler = WorkStealingScheduler::new(config, stats).unwrap();
        scheduler.start().unwrap();
        
        let scheduler_stats = scheduler.stats();
        assert_eq!(scheduler_stats.worker_count, 2);
        assert_eq!(scheduler_stats.worker_stats.len(), 2);
        
        scheduler.stop().unwrap();
    }
}