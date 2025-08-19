use crate::vm::{Value, VmError, VmResult};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;
use std::thread;
use std::sync::{Arc, Mutex, Condvar};
use std::collections::VecDeque;
use crossbeam_channel::{bounded, unbounded, Sender, Receiver, RecvError, SendError, TryRecvError, TrySendError};

/// Future foreign object - represents an async computation result
#[derive(Debug, Clone, PartialEq)]
pub struct AsyncFuture {
    /// The resolved value (for now, immediately resolved)
    value: Value,
    // Future could have additional fields like status, thread_id, etc.
}

impl AsyncFuture {
    /// Create a new resolved Future with the given value
    pub fn resolved(value: Value) -> Self {
        AsyncFuture { value }
    }
    
    /// Get the resolved value
    pub fn get_value(&self) -> &Value {
        &self.value
    }
}

impl Foreign for AsyncFuture {
    fn type_name(&self) -> &'static str {
        "Future"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "resolve" => {
                // Return the resolved value
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(self.value.clone())
            }
            "isResolved" => {
                // For now, all futures are immediately resolved
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(true))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Work item for ThreadPool - represents a task to be executed
#[derive(Debug, Clone)]
struct WorkItem {
    id: usize,
    function: Value,
    args: Vec<Value>,
}

/// ThreadPool foreign object - manages worker threads for concurrent execution
#[derive(Debug)]
pub struct ThreadPool {
    worker_count: usize,
    task_queue: Arc<Mutex<VecDeque<WorkItem>>>,
    result_map: Arc<Mutex<std::collections::HashMap<usize, Value>>>,
    task_available: Arc<Condvar>,
    next_task_id: Arc<Mutex<usize>>,
    shutdown: Arc<Mutex<bool>>,
    // Store worker handles as Option so we can take them in drop
    workers: Arc<Mutex<Vec<Option<thread::JoinHandle<()>>>>>,
}

impl ThreadPool {
    /// Create a new ThreadPool with the specified number of workers
    pub fn new(worker_count: usize) -> Self {
        let task_queue = Arc::new(Mutex::new(VecDeque::new()));
        let result_map = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let task_available = Arc::new(Condvar::new());
        let next_task_id = Arc::new(Mutex::new(0));
        let shutdown = Arc::new(Mutex::new(false));
        let mut worker_handles = Vec::new();

        // Spawn worker threads
        for worker_id in 0..worker_count {
            let task_queue = Arc::clone(&task_queue);
            let result_map = Arc::clone(&result_map);
            let task_available = Arc::clone(&task_available);
            let shutdown = Arc::clone(&shutdown);

            let worker = thread::spawn(move || {
                loop {
                    // Wait for work or shutdown signal
                    let work_item = {
                        let mut queue = task_queue.lock().unwrap();
                        while queue.is_empty() && !*shutdown.lock().unwrap() {
                            queue = task_available.wait(queue).unwrap();
                        }
                        
                        if *shutdown.lock().unwrap() {
                            break;
                        }
                        
                        queue.pop_front()
                    };

                    if let Some(work) = work_item {
                        // Execute the work item
                        let result = Self::execute_work_item(&work);
                        
                        // Store the result
                        {
                            let mut results = result_map.lock().unwrap();
                            results.insert(work.id, result);
                        }
                    }
                }
                println!("Worker {} shutting down", worker_id);
            });

            worker_handles.push(Some(worker));
        }

        let workers = Arc::new(Mutex::new(worker_handles));

        ThreadPool {
            worker_count,
            task_queue,
            result_map,
            task_available,
            next_task_id,
            shutdown,
            workers,
        }
    }

    /// Execute a work item (simplified version for now)
    fn execute_work_item(work: &WorkItem) -> Value {
        // For now, just return a simple computation result
        // In a full implementation, this would invoke the VM or interpreter
        match &work.function {
            Value::Function(name) if name == "Add" => {
                // Simple addition function
                if work.args.len() == 2 {
                    match (&work.args[0], &work.args[1]) {
                        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a + b),
                        (Value::Real(a), Value::Real(b)) => Value::Real(a + b),
                        _ => Value::String(format!("Error: Cannot add {:?} and {:?}", work.args[0], work.args[1])),
                    }
                } else {
                    Value::String("Error: Add requires exactly 2 arguments".to_string())
                }
            }
            Value::Function(name) if name == "Multiply" => {
                // Simple multiplication function
                if work.args.len() == 2 {
                    match (&work.args[0], &work.args[1]) {
                        (Value::Integer(a), Value::Integer(b)) => Value::Integer(a * b),
                        (Value::Real(a), Value::Real(b)) => Value::Real(a * b),
                        _ => Value::String(format!("Error: Cannot multiply {:?} and {:?}", work.args[0], work.args[1])),
                    }
                } else {
                    Value::String("Error: Multiply requires exactly 2 arguments".to_string())
                }
            }
            _ => Value::String(format!("Error: Unknown function {:?}", work.function)),
        }
    }

    /// Submit a task to the thread pool and return a task ID
    pub fn submit_task(&self, function: Value, args: Vec<Value>) -> usize {
        let task_id = {
            let mut id_counter = self.next_task_id.lock().unwrap();
            let id = *id_counter;
            *id_counter += 1;
            id
        };

        let work_item = WorkItem {
            id: task_id,
            function,
            args,
        };

        {
            let mut queue = self.task_queue.lock().unwrap();
            queue.push_back(work_item);
        }

        // Notify workers that work is available
        self.task_available.notify_one();

        task_id
    }

    /// Get the result of a completed task (non-blocking)
    pub fn get_result(&self, task_id: usize) -> Option<Value> {
        let mut results = self.result_map.lock().unwrap();
        results.remove(&task_id)
    }

    /// Check if a task is completed
    pub fn is_completed(&self, task_id: usize) -> bool {
        let results = self.result_map.lock().unwrap();
        results.contains_key(&task_id)
    }

    /// Get the number of worker threads
    pub fn worker_count(&self) -> usize {
        self.worker_count
    }

    /// Get the number of pending tasks
    pub fn pending_tasks(&self) -> usize {
        let queue = self.task_queue.lock().unwrap();
        queue.len()
    }
}

impl Foreign for ThreadPool {
    fn type_name(&self) -> &'static str {
        "ThreadPool"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "submit" => {
                if args.len() < 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let function = args[0].clone();
                let task_args = if args.len() > 1 {
                    args[1..].to_vec()
                } else {
                    Vec::new()
                };
                
                let task_id = self.submit_task(function, task_args);
                Ok(Value::Integer(task_id as i64))
            }
            "getResult" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::Integer(task_id) => {
                        if let Some(result) = self.get_result(*task_id as usize) {
                            Ok(result)
                        } else {
                            Ok(Value::Missing)
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "Other".to_string(),
                    }),
                }
            }
            "isCompleted" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match &args[0] {
                    Value::Integer(task_id) => {
                        let completed = self.is_completed(*task_id as usize);
                        Ok(Value::Boolean(completed))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: "Other".to_string(),
                    }),
                }
            }
            "workerCount" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.worker_count() as i64))
            }
            "pendingTasks" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.pending_tasks() as i64))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        // ThreadPool cannot be cloned due to thread handles
        // Return a new ThreadPool with the same configuration
        Box::new(ThreadPool::new(self.worker_count))
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // Signal shutdown
        {
            let mut shutdown = self.shutdown.lock().unwrap();
            *shutdown = true;
        }
        
        // Wake up all workers
        self.task_available.notify_all();
        
        // Join all worker threads
        if let Ok(mut workers) = self.workers.lock() {
            for worker_handle in workers.iter_mut() {
                if let Some(handle) = worker_handle.take() {
                    let _ = handle.join(); // Ignore join errors
                }
            }
        }
    }
}

/// ThreadPool[worker_count] - Creates a new ThreadPool with specified number of workers
pub fn create_thread_pool(args: &[Value]) -> VmResult<Value> {
    let worker_count = if args.len() == 0 {
        // Default to 4 threads (reasonable default)
        4
    } else if args.len() == 1 {
        match &args[0] {
            Value::Integer(count) => {
                if *count <= 0 {
                    return Err(VmError::Runtime("ThreadPool worker count must be positive".to_string()));
                }
                *count as usize
            }
            _ => return Err(VmError::Runtime("ThreadPool worker count must be an integer".to_string())),
        }
    } else {
        return Err(VmError::Runtime("ThreadPool requires 0 or 1 argument (worker count)".to_string()));
    };

    let thread_pool = ThreadPool::new(worker_count);
    let ly_obj = LyObj::new(Box::new(thread_pool));
    Ok(Value::LyObj(ly_obj))
}

/// Promise[value] - Creates a Future that resolves to the given value
pub fn promise(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Promise requires exactly 1 argument".to_string()));
    }
    
    // Create a Future foreign object that immediately resolves to the given value
    let future = AsyncFuture::resolved(args[0].clone());
    let ly_obj = LyObj::new(Box::new(future));
    Ok(Value::LyObj(ly_obj))
}

/// Await[future] - Resolves a Future to its contained value
pub fn await_future(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Await requires exactly 1 argument".to_string()));
    }
    
    match &args[0] {
        Value::LyObj(ly_obj) => {
            // Check if this is a Future object
            if ly_obj.type_name() == "Future" {
                // Call the resolve method to get the value
                match ly_obj.call_method("resolve", &[]) {
                    Ok(value) => Ok(value),
                    Err(foreign_err) => Err(VmError::Runtime(format!("Failed to resolve Future: {}", foreign_err))),
                }
            } else {
                Err(VmError::Runtime(format!("Await requires a Future object, got {}", ly_obj.type_name())))
            }
        }
        // If not a LyObj, check if it might be a value that's immediately available
        other => Ok(other.clone()),
    }
}

/// AsyncFunction[function] - Wraps a regular function to return Futures
pub fn async_function(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("AsyncFunction requires exactly 1 argument".to_string()));
    }
    
    // For now, we'll create a simple async wrapper
    // In practice, this would integrate with the function compilation system
    match &args[0] {
        Value::Function(func_name) => {
            // Return a wrapped function that produces Futures
            // This is a simplified implementation
            Ok(Value::Function(format!("Async{}", func_name)))
        }
        _ => Err(VmError::Runtime("AsyncFunction requires a function argument".to_string())),
    }
}

/// Calculate optimal chunk size for work distribution
fn calculate_optimal_chunk_size(data_size: usize, worker_count: usize) -> usize {
    if data_size <= worker_count {
        return 1; // One item per worker maximum
    }
    
    // Target: each worker gets 2-4 chunks for good load balancing
    let target_chunks_per_worker = 3;
    let total_target_chunks = worker_count * target_chunks_per_worker;
    
    let chunk_size = (data_size + total_target_chunks - 1) / total_target_chunks; // Ceiling division
    std::cmp::max(chunk_size, 1) // Minimum chunk size is 1
}

/// Split data into chunks for optimal parallel processing
fn split_into_chunks(data: &[Value], chunk_size: usize) -> Vec<Vec<Value>> {
    let mut chunks = Vec::new();
    let mut current_chunk = Vec::new();
    
    for item in data {
        current_chunk.push(item.clone());
        
        if current_chunk.len() >= chunk_size {
            chunks.push(current_chunk);
            current_chunk = Vec::new();
        }
    }
    
    // Add remaining items as the last chunk
    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }
    
    chunks
}

/// Parallel[{function, list}] or Parallel[{function, list}, threadpool] - Execute function over list in parallel
pub fn parallel(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::Runtime("Parallel requires 1 or 2 arguments: {function, list} and optional ThreadPool".to_string()));
    }

    // Get or create ThreadPool
    let thread_pool = if args.len() == 2 {
        // Use provided ThreadPool
        match &args[1] {
            Value::LyObj(ly_obj) if ly_obj.type_name() == "ThreadPool" => ly_obj.clone(),
            _ => return Err(VmError::Runtime("Second argument to Parallel must be a ThreadPool".to_string())),
        }
    } else {
        // Create default ThreadPool
        let default_pool = ThreadPool::new(4);
        LyObj::new(Box::new(default_pool))
    };

    match &args[0] {
        Value::List(function_and_data) if function_and_data.len() == 2 => {
            let function = function_and_data[0].clone();
            
            match &function_and_data[1] {
                Value::List(data_list) => {
                    // Get ThreadPool worker count for optimal work distribution
                    let worker_count = match thread_pool.call_method("workerCount", &[]) {
                        Ok(Value::Integer(count)) => count as usize,
                        Ok(_) => 4, // Default fallback
                        Err(_) => 4, // Default fallback
                    };
                    
                    // Adaptive work splitting based on data size and worker count
                    let data_size = data_list.len();
                    let optimal_chunk_size = calculate_optimal_chunk_size(data_size, worker_count);
                    
                    let mut task_ids = Vec::new();
                    
                    if optimal_chunk_size > 1 && data_size > worker_count * 2 {
                        // For large datasets, split into chunks for better load balancing
                        let chunks = split_into_chunks(data_list, optimal_chunk_size);
                        
                        for chunk in chunks {
                            // Submit chunk processing task
                            let chunk_value = Value::List(chunk);
                            let submit_args = vec![function.clone(), chunk_value];
                            match thread_pool.call_method("submit", &submit_args) {
                                Ok(Value::Integer(task_id)) => task_ids.push(task_id),
                                Ok(_) => return Err(VmError::Runtime("ThreadPool submit returned unexpected type".to_string())),
                                Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to submit chunk task to ThreadPool: {}", foreign_err))),
                            }
                        }
                    } else {
                        // For smaller datasets, submit individual items
                        for item in data_list {
                            // Submit task: apply function to each item
                            // ThreadPool.submit expects (function, args...) format
                            let submit_args = vec![function.clone(), item.clone()];
                            match thread_pool.call_method("submit", &submit_args) {
                                Ok(Value::Integer(task_id)) => task_ids.push(task_id),
                                Ok(_) => return Err(VmError::Runtime("ThreadPool submit returned unexpected type".to_string())),
                                Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to submit task to ThreadPool: {}", foreign_err))),
                            }
                        }
                    }

                    // Collect results from all tasks
                    let mut results = Vec::new();
                    let is_chunked = optimal_chunk_size > 1 && data_size > worker_count * 2;
                    
                    for task_id in task_ids {
                        // Wait for task completion and get result
                        loop {
                            match thread_pool.call_method("isCompleted", &[Value::Integer(task_id)]) {
                                Ok(Value::Boolean(true)) => {
                                    // Task completed, get result
                                    match thread_pool.call_method("getResult", &[Value::Integer(task_id)]) {
                                        Ok(result) if result != Value::Missing => {
                                            if is_chunked {
                                                // If we processed chunks, flatten the results
                                                match result {
                                                    Value::List(chunk_results) => {
                                                        results.extend(chunk_results);
                                                    }
                                                    other => results.push(other),
                                                }
                                            } else {
                                                // Individual task results
                                                results.push(result);
                                            }
                                            break;
                                        }
                                        Ok(Value::Missing) => {
                                            // Task not ready yet, continue waiting
                                            continue;
                                        }
                                        Ok(result) => {
                                            if is_chunked {
                                                match result {
                                                    Value::List(chunk_results) => {
                                                        results.extend(chunk_results);
                                                    }
                                                    other => results.push(other),
                                                }
                                            } else {
                                                results.push(result);
                                            }
                                            break;
                                        }
                                        Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to get task result: {}", foreign_err))),
                                    }
                                }
                                Ok(Value::Boolean(false)) => {
                                    // Task not completed yet, wait a bit
                                    std::thread::sleep(std::time::Duration::from_millis(1));
                                    continue;
                                }
                                Ok(_) => return Err(VmError::Runtime("ThreadPool isCompleted returned unexpected type".to_string())),
                                Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to check task completion: {}", foreign_err))),
                            }
                        }
                    }

                    Ok(Value::List(results))
                }
                _ => Err(VmError::Runtime("Parallel requires second element to be a list of data".to_string())),
            }
        }
        Value::List(legacy_futures) => {
            // Legacy support: Parallel[{future1, future2, ...}] - Execute futures in parallel
            let mut results = Vec::new();
            for future in legacy_futures {
                match future {
                    Value::LyObj(ly_obj) if ly_obj.type_name() == "Future" => {
                        // Resolve the Future
                        match ly_obj.call_method("resolve", &[]) {
                            Ok(value) => results.push(value),
                            Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to resolve Future in Parallel: {}", foreign_err))),
                        }
                    }
                    // If not a future, just include the value
                    other => results.push(other.clone()),
                }
            }
            Ok(Value::List(results))
        }
        _ => Err(VmError::Runtime("Parallel requires either {function, list} or list of futures".to_string())),
    }
}

/// All[{future1, future2, ...}] - Wait for all futures to complete
pub fn all_futures(args: &[Value]) -> VmResult<Value> {
    // For now, same as Parallel - resolve all immediately
    parallel(args)
}

/// Any[{future1, future2, ...}] - Return the first completed future
pub fn any_future(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Any requires exactly 1 argument (list of futures)".to_string()));
    }
    
    match &args[0] {
        Value::List(futures) => {
            if futures.is_empty() {
                return Err(VmError::Runtime("Any requires at least one future".to_string()));
            }
            
            // For now, just return the first future's result
            match &futures[0] {
                Value::LyObj(ly_obj) if ly_obj.type_name() == "Future" => {
                    // Resolve the first Future
                    match ly_obj.call_method("resolve", &[]) {
                        Ok(value) => Ok(value),
                        Err(foreign_err) => Err(VmError::Runtime(format!("Failed to resolve Future in Any: {}", foreign_err))),
                    }
                }
                other => Ok(other.clone()),
            }
        }
        _ => Err(VmError::Runtime("Any requires a list argument".to_string())),
    }
}

/// Channel Foreign Object - enables thread-safe message passing
#[derive(Debug)]
pub struct Channel {
    sender: Sender<Value>,
    receiver: Receiver<Value>,
    capacity: Option<usize>, // None for unbounded, Some(n) for bounded
    is_closed: Arc<Mutex<bool>>,
}

impl Channel {
    /// Create a new unbounded channel
    pub fn unbounded() -> Self {
        let (sender, receiver) = unbounded();
        Channel {
            sender,
            receiver,
            capacity: None,
            is_closed: Arc::new(Mutex::new(false)),
        }
    }

    /// Create a new bounded channel with specified capacity
    pub fn bounded(capacity: usize) -> Self {
        let (sender, receiver) = bounded(capacity);
        Channel {
            sender,
            receiver,
            capacity: Some(capacity),
            is_closed: Arc::new(Mutex::new(false)),
        }
    }

    /// Send a value through the channel (non-blocking)
    pub fn send(&self, value: Value) -> Result<(), String> {
        if *self.is_closed.lock().unwrap() {
            return Err("Channel is closed".to_string());
        }

        match self.sender.send(value) {
            Ok(()) => Ok(()),
            Err(SendError(_)) => Err("Channel receiver has been dropped".to_string()),
        }
    }

    /// Try to send a value through the channel (non-blocking, returns immediately)
    pub fn try_send(&self, value: Value) -> Result<(), String> {
        if *self.is_closed.lock().unwrap() {
            return Err("Channel is closed".to_string());
        }

        match self.sender.try_send(value) {
            Ok(()) => Ok(()),
            Err(TrySendError::Full(_)) => Err("Channel is full".to_string()),
            Err(TrySendError::Disconnected(_)) => Err("Channel receiver has been dropped".to_string()),
        }
    }

    /// Receive a value from the channel (blocking)
    pub fn receive(&self) -> Result<Value, String> {
        match self.receiver.recv() {
            Ok(value) => Ok(value),
            Err(RecvError) => {
                if *self.is_closed.lock().unwrap() {
                    Err("Channel is closed and empty".to_string())
                } else {
                    Err("Channel sender has been dropped".to_string())
                }
            }
        }
    }

    /// Try to receive a value from the channel (non-blocking)
    pub fn try_receive(&self) -> Result<Value, String> {
        match self.receiver.try_recv() {
            Ok(value) => Ok(value),
            Err(TryRecvError::Empty) => Err("Channel is empty".to_string()),
            Err(TryRecvError::Disconnected) => {
                if *self.is_closed.lock().unwrap() {
                    Err("Channel is closed and empty".to_string())
                } else {
                    Err("Channel sender has been dropped".to_string())
                }
            }
        }
    }

    /// Close the channel
    pub fn close(&self) {
        let mut closed = self.is_closed.lock().unwrap();
        *closed = true;
    }

    /// Check if the channel is closed
    pub fn is_closed(&self) -> bool {
        *self.is_closed.lock().unwrap()
    }

    /// Get the channel capacity (None for unbounded)
    pub fn capacity(&self) -> Option<usize> {
        self.capacity
    }

    /// Get the current number of messages in the channel
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }
}

impl Foreign for Channel {
    fn type_name(&self) -> &'static str {
        "Channel"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "send" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match self.send(args[0].clone()) {
                    Ok(()) => Ok(Value::Boolean(true)),
                    Err(err) => Err(ForeignError::RuntimeError {
                        message: err,
                    }),
                }
            }
            "trySend" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                match self.try_send(args[0].clone()) {
                    Ok(()) => Ok(Value::Boolean(true)),
                    Err(err) => Err(ForeignError::RuntimeError {
                        message: err,
                    }),
                }
            }
            "receive" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.receive() {
                    Ok(value) => Ok(value),
                    Err(err) => Err(ForeignError::RuntimeError {
                        message: err,
                    }),
                }
            }
            "tryReceive" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.try_receive() {
                    Ok(value) => Ok(value),
                    Err(err) => Err(ForeignError::RuntimeError {
                        message: err,
                    }),
                }
            }
            "close" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                self.close();
                Ok(Value::Boolean(true))
            }
            "isClosed" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                Ok(Value::Boolean(self.is_closed()))
            }
            "capacity" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.capacity() {
                    Some(cap) => Ok(Value::Integer(cap as i64)),
                    None => Ok(Value::Missing), // Unbounded channel
                }
            }
            "len" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                Ok(Value::Integer(self.len() as i64))
            }
            "isEmpty" => {
                if args.len() != 0 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                Ok(Value::Boolean(self.is_empty()))
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        // Channels can be cloned - both sender and receiver can be shared
        Box::new(Channel {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
            capacity: self.capacity,
            is_closed: Arc::clone(&self.is_closed),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Channel[] - Creates a new unbounded channel
pub fn create_channel(args: &[Value]) -> VmResult<Value> {
    if args.len() != 0 {
        return Err(VmError::Runtime("Channel[] takes no arguments for unbounded channel".to_string()));
    }

    let channel = Channel::unbounded();
    let ly_obj = LyObj::new(Box::new(channel));
    Ok(Value::LyObj(ly_obj))
}

/// BoundedChannel[capacity] - Creates a new bounded channel with specified capacity
pub fn create_bounded_channel(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("BoundedChannel requires exactly 1 argument (capacity)".to_string()));
    }

    let capacity = match &args[0] {
        Value::Integer(cap) => {
            if *cap <= 0 {
                return Err(VmError::Runtime("Channel capacity must be positive".to_string()));
            }
            *cap as usize
        }
        _ => return Err(VmError::Runtime("Channel capacity must be an integer".to_string())),
    };

    let channel = Channel::bounded(capacity);
    let ly_obj = LyObj::new(Box::new(channel));
    Ok(Value::LyObj(ly_obj))
}

/// Send[channel, value] - Send a value through a channel
pub fn channel_send(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Send requires exactly 2 arguments (channel, value)".to_string()));
    }

    match &args[0] {
        Value::LyObj(ly_obj) if ly_obj.type_name() == "Channel" => {
            match ly_obj.call_method("send", &[args[1].clone()]) {
                Ok(result) => Ok(result),
                Err(foreign_err) => Err(VmError::Runtime(format!("Send failed: {}", foreign_err))),
            }
        }
        _ => Err(VmError::Runtime("First argument to Send must be a Channel".to_string())),
    }
}

/// Receive[channel] - Receive a value from a channel (blocking)
pub fn channel_receive(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("Receive requires exactly 1 argument (channel)".to_string()));
    }

    match &args[0] {
        Value::LyObj(ly_obj) if ly_obj.type_name() == "Channel" => {
            match ly_obj.call_method("receive", &[]) {
                Ok(result) => Ok(result),
                Err(foreign_err) => Err(VmError::Runtime(format!("Receive failed: {}", foreign_err))),
            }
        }
        _ => Err(VmError::Runtime("Argument to Receive must be a Channel".to_string())),
    }
}

/// TrySend[channel, value] - Try to send a value through a channel (non-blocking)
pub fn channel_try_send(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("TrySend requires exactly 2 arguments (channel, value)".to_string()));
    }

    match &args[0] {
        Value::LyObj(ly_obj) if ly_obj.type_name() == "Channel" => {
            match ly_obj.call_method("trySend", &[args[1].clone()]) {
                Ok(result) => Ok(result),
                Err(foreign_err) => Err(VmError::Runtime(format!("TrySend failed: {}", foreign_err))),
            }
        }
        _ => Err(VmError::Runtime("First argument to TrySend must be a Channel".to_string())),
    }
}

/// TryReceive[channel] - Try to receive a value from a channel (non-blocking)
pub fn channel_try_receive(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("TryReceive requires exactly 1 argument (channel)".to_string()));
    }

    match &args[0] {
        Value::LyObj(ly_obj) if ly_obj.type_name() == "Channel" => {
            match ly_obj.call_method("tryReceive", &[]) {
                Ok(result) => Ok(result),
                Err(foreign_err) => Err(VmError::Runtime(format!("TryReceive failed: {}", foreign_err))),
            }
        }
        _ => Err(VmError::Runtime("Argument to TryReceive must be a Channel".to_string())),
    }
}

/// ChannelClose[channel] - Close a channel
pub fn channel_close(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("ChannelClose requires exactly 1 argument (channel)".to_string()));
    }

    match &args[0] {
        Value::LyObj(ly_obj) if ly_obj.type_name() == "Channel" => {
            match ly_obj.call_method("close", &[]) {
                Ok(result) => Ok(result),
                Err(foreign_err) => Err(VmError::Runtime(format!("ChannelClose failed: {}", foreign_err))),
            }
        }
        _ => Err(VmError::Runtime("Argument to ChannelClose must be a Channel".to_string())),
    }
}

/// ParallelMap[function, list] - Optimized parallel map operation
pub fn parallel_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("ParallelMap requires exactly 2 arguments (function, list)".to_string()));
    }

    let function = args[0].clone();
    
    match &args[1] {
        Value::List(data_list) => {
            // Use the enhanced parallel function with automatic load balancing
            let parallel_args = vec![Value::List(vec![function, args[1].clone()])];
            parallel(&parallel_args)
        }
        _ => Err(VmError::Runtime("ParallelMap requires second argument to be a list".to_string())),
    }
}

/// ParallelReduce[function, list] - Parallel reduction with tree-like computation
pub fn parallel_reduce(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("ParallelReduce requires exactly 2 arguments (function, list)".to_string()));
    }

    let function = args[0].clone();
    
    match &args[1] {
        Value::List(data_list) => {
            if data_list.is_empty() {
                return Ok(Value::Missing);
            }
            
            if data_list.len() == 1 {
                return Ok(data_list[0].clone());
            }
            
            // Create a default ThreadPool for reduction
            let thread_pool = ThreadPool::new(4);
            let thread_pool_obj = LyObj::new(Box::new(thread_pool));
            
            // Implement tree-like parallel reduction
            let mut current_level = data_list.clone();
            
            while current_level.len() > 1 {
                let mut next_level = Vec::new();
                let mut task_ids = Vec::new();
                
                // Process pairs in parallel
                let mut i = 0;
                while i < current_level.len() {
                    if i + 1 < current_level.len() {
                        // Process pair
                        let submit_args = vec![function.clone(), current_level[i].clone(), current_level[i + 1].clone()];
                        match thread_pool_obj.call_method("submit", &submit_args) {
                            Ok(Value::Integer(task_id)) => task_ids.push(task_id),
                            Ok(_) => return Err(VmError::Runtime("ThreadPool submit returned unexpected type".to_string())),
                            Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to submit reduction task: {}", foreign_err))),
                        }
                        i += 2;
                    } else {
                        // Odd element, carry to next level
                        next_level.push(current_level[i].clone());
                        i += 1;
                    }
                }
                
                // Collect results from this level
                for task_id in task_ids {
                    loop {
                        match thread_pool_obj.call_method("isCompleted", &[Value::Integer(task_id)]) {
                            Ok(Value::Boolean(true)) => {
                                match thread_pool_obj.call_method("getResult", &[Value::Integer(task_id)]) {
                                    Ok(result) if result != Value::Missing => {
                                        next_level.push(result);
                                        break;
                                    }
                                    Ok(Value::Missing) => continue,
                                    Ok(result) => {
                                        next_level.push(result);
                                        break;
                                    }
                                    Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to get reduction result: {}", foreign_err))),
                                }
                            }
                            Ok(Value::Boolean(false)) => {
                                std::thread::sleep(std::time::Duration::from_millis(1));
                                continue;
                            }
                            Ok(_) => return Err(VmError::Runtime("ThreadPool isCompleted returned unexpected type".to_string())),
                            Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to check reduction task completion: {}", foreign_err))),
                        }
                    }
                }
                
                current_level = next_level;
            }
            
            Ok(current_level[0].clone())
        }
        _ => Err(VmError::Runtime("ParallelReduce requires second argument to be a list".to_string())),
    }
}

/// Pipeline[channels, functions] - Create a processing pipeline using channels
pub fn pipeline(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Pipeline requires exactly 2 arguments (channels, functions)".to_string()));
    }

    match (&args[0], &args[1]) {
        (Value::List(channels), Value::List(functions)) => {
            if channels.len() != functions.len() + 1 {
                return Err(VmError::Runtime("Pipeline requires N+1 channels for N functions".to_string()));
            }
            
            // Create ThreadPool for pipeline stages
            let thread_pool = ThreadPool::new(functions.len());
            let thread_pool_obj = LyObj::new(Box::new(thread_pool));
            
            let mut stage_task_ids = Vec::new();
            
            // Start each pipeline stage
            for (i, function) in functions.iter().enumerate() {
                let input_channel = channels[i].clone();
                let output_channel = channels[i + 1].clone();
                
                // Submit pipeline stage task
                let submit_args = vec![function.clone(), input_channel, output_channel];
                match thread_pool_obj.call_method("submit", &submit_args) {
                    Ok(Value::Integer(task_id)) => stage_task_ids.push(task_id),
                    Ok(_) => return Err(VmError::Runtime("ThreadPool submit returned unexpected type".to_string())),
                    Err(foreign_err) => return Err(VmError::Runtime(format!("Failed to submit pipeline stage: {}", foreign_err))),
                }
            }
            
            // Return the pipeline control object (list of task IDs and final output channel)
            let pipeline_info = vec![
                Value::List(stage_task_ids.into_iter().map(|id| Value::Integer(id)).collect()),
                channels[channels.len() - 1].clone(), // Final output channel
            ];
            
            Ok(Value::List(pipeline_info))
        }
        _ => Err(VmError::Runtime("Pipeline requires lists of channels and functions".to_string())),
    }
}