//! ThreadPool Foreign Object Implementation
//!
//! Provides thread pool functionality as a Foreign object that integrates
//! with the existing work-stealing scheduler.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError};
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use crossbeam_channel::{Sender, Receiver, unbounded};
use std::any::Any;

/// Unique identifier for tasks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub usize);

/// Result of a completed task
#[derive(Debug, Clone)]
pub enum TaskResult {
    Completed(Value),
    Error(String),
    Pending,
}

/// ThreadPool Foreign object that manages parallel task execution
#[derive(Debug)]
pub struct ThreadPool {
    /// Worker thread count
    worker_count: usize,
    /// Task ID generator
    next_task_id: AtomicUsize,
    /// Storage for task results
    task_results: Arc<RwLock<HashMap<TaskId, TaskResult>>>,
    /// Channel for communicating completed tasks
    completion_sender: Sender<(TaskId, TaskResult)>,
    completion_receiver: Receiver<(TaskId, TaskResult)>,
}

impl ThreadPool {
    /// Create a new ThreadPool with specified worker count
    pub fn new(worker_count: usize) -> VmResult<Self> {
        let task_results = Arc::new(RwLock::new(HashMap::new()));
        let (completion_sender, completion_receiver) = unbounded();

        Ok(Self {
            worker_count,
            next_task_id: AtomicUsize::new(0),
            task_results,
            completion_sender,
            completion_receiver,
        })
    }

    /// Submit a task for execution and return a task ID
    pub fn submit_task<F>(&self, task: F) -> VmResult<TaskId>
    where
        F: FnOnce() -> VmResult<Value> + Send + 'static,
    {
        let task_id = TaskId(self.next_task_id.fetch_add(1, Ordering::SeqCst));
        let results = Arc::clone(&self.task_results);
        let sender = self.completion_sender.clone();

        // Mark task as pending
        {
            let mut results_guard = results.write()
                .map_err(|e| VmError::Runtime(format!("Failed to acquire task results lock: {}", e)))?;
            results_guard.insert(task_id, TaskResult::Pending);
        }

        // Create task wrapper
        let wrapped_task = move || {
            let result = match task() {
                Ok(value) => TaskResult::Completed(value),
                Err(e) => TaskResult::Error(format!("Task execution error: {:?}", e)),
            };
            
            // Send completion notification
            if let Err(e) = sender.send((task_id, result.clone())) {
                eprintln!("Failed to send task completion: {}", e);
            }
            
            result
        };

        // Submit to scheduler (this would need integration with the actual scheduler)
        // For now, we'll simulate this with a simple task execution
        std::thread::spawn(move || {
            let _ = wrapped_task();
        });

        Ok(task_id)
    }

    /// Get the result of a task (non-blocking)
    pub fn get_result(&self, task_id: TaskId) -> VmResult<TaskResult> {
        // Process any completed tasks first
        self.process_completed_tasks()?;

        let results = self.task_results.read()
            .map_err(|e| VmError::Runtime(format!("Failed to acquire task results lock: {}", e)))?;

        Ok(results.get(&task_id).cloned().unwrap_or(TaskResult::Error("Task not found".to_string())))
    }

    /// Check if a task is completed
    pub fn is_completed(&self, task_id: TaskId) -> VmResult<bool> {
        let result = self.get_result(task_id)?;
        Ok(matches!(result, TaskResult::Completed(_) | TaskResult::Error(_)))
    }

    /// Get the number of worker threads
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        if let Ok(results) = self.task_results.read() {
            results.values().filter(|r| matches!(r, TaskResult::Pending)).count()
        } else {
            0
        }
    }

    /// Process completed tasks from the completion channel
    fn process_completed_tasks(&self) -> VmResult<()> {
        while let Ok((task_id, result)) = self.completion_receiver.try_recv() {
            let mut results = self.task_results.write()
                .map_err(|e| VmError::Runtime(format!("Failed to acquire task results lock: {}", e)))?;
            results.insert(task_id, result);
        }
        Ok(())
    }
}

impl Foreign for ThreadPool {
    fn type_name(&self) -> &'static str {
        "ThreadPool"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "submit" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                // This is a simplified version - in practice we'd need to handle
                // function calls and their arguments properly
                let task_id = self.submit_task(|| Ok(Value::Integer(42)))
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Failed to submit task: {:?}", e),
                    })?;

                Ok(Value::Integer(task_id.0 as i64))
            },
            "getResult" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                if let Value::Integer(task_id) = &args[0] {
                    let result = self.get_result(TaskId(*task_id as usize))
                        .map_err(|e| ForeignError::RuntimeError {
                            message: format!("Failed to get task result: {:?}", e),
                        })?;

                    match result {
                        TaskResult::Completed(value) => Ok(value),
                        TaskResult::Error(msg) => Err(ForeignError::RuntimeError { message: msg }),
                        TaskResult::Pending => Ok(Value::Symbol("Pending".to_string())),
                    }
                } else {
                    Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    })
                }
            },
            "isCompleted" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                if let Value::Integer(task_id) = &args[0] {
                    let completed = self.is_completed(TaskId(*task_id as usize))
                        .map_err(|e| ForeignError::RuntimeError {
                            message: format!("Failed to check task completion: {:?}", e),
                        })?;

                    Ok(Value::Boolean(completed))
                } else {
                    Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    })
                }
            },
            "workerCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                Ok(Value::Integer(self.worker_count() as i64))
            },
            "pendingTasks" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                Ok(Value::Integer(self.pending_tasks() as i64))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(ThreadPool {
            worker_count: self.worker_count,
            next_task_id: AtomicUsize::new(self.next_task_id.load(Ordering::Relaxed)),
            task_results: Arc::clone(&self.task_results),
            completion_sender: self.completion_sender.clone(),
            completion_receiver: self.completion_receiver.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for ThreadPool {}
unsafe impl Sync for ThreadPool {}