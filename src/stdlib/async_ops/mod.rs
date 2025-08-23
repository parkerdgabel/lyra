//! Async Operations Module for Lyra Standard Library
//!
//! This module provides production-ready async/concurrency functionality as Foreign objects.
//! It bridges the existing concurrency system with the stdlib function registry.
//!
//! # Architecture
//! - Foreign objects for ThreadPool, Channel, Future, etc.
//! - Zero VM pollution - all complex types stay outside VM core
//! - Integration with work-stealing scheduler (when available)
//! - Complete thread safety and memory safety

use crate::vm::{Value, VmResult};
use crate::foreign::LyObj;

// Module organization - using real implementations
pub mod thread_pool;
pub mod channel;
pub mod future;
pub mod parallel;

// Re-export real types
pub use thread_pool::ThreadPool;
pub use channel::{Channel, BoundedChannel};
pub use future::{Future, Promise};
pub use parallel::{ParallelMap, ParallelReduce};

/// TODO: Initialize the global concurrency system when the module is re-enabled
/// For now, we'll create simplified implementations without the full scheduler

/// ThreadPool constructor function
pub fn thread_pool(args: &[Value]) -> VmResult<Value> {
    match args.len() {
        0 => {
            // Default ThreadPool with 4 workers
            let pool = ThreadPool::new(4)?;
            Ok(Value::LyObj(LyObj::new(Box::new(pool))))
        },
        1 => {
            // ThreadPool with specified worker count
            if let Value::Integer(worker_count) = &args[0] {
                let pool = ThreadPool::new(*worker_count as usize)?;
                Ok(Value::LyObj(LyObj::new(Box::new(pool))))
            } else {
                Err(crate::vm::VmError::TypeError {
                    expected: "Integer".to_string(),
                    actual: format!("{:?}", args[0]),
                })
            }
        },
        _ => Err(crate::vm::VmError::ArityError {
            function_name: "ThreadPool".to_string(),
            expected: 1,
            actual: args.len(),
        }),
    }
}

/// Channel constructor function (unbounded)
pub fn channel(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(crate::vm::VmError::ArityError {
            function_name: "Channel".to_string(),
            expected: 0,
            actual: args.len(),
        });
    }
    
    let channel = Channel::new()?;
    Ok(Value::LyObj(LyObj::new(Box::new(channel))))
}

/// BoundedChannel constructor function
pub fn bounded_channel(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "BoundedChannel".to_string(),
            expected: 1,
            actual: args.len(),
        });
    }
    
    if let Value::Integer(capacity) = &args[0] {
        let channel = BoundedChannel::new(*capacity as usize)?;
        Ok(Value::LyObj(LyObj::new(Box::new(channel))))
    } else {
        Err(crate::vm::VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        })
    }
}

/// Placeholder async functions - to be implemented when concurrency system is enabled

/// Send function for channels
pub fn send(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "Send".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }
    
    // Extract channel and value
    let channel = &args[0];
    let value = &args[1];
    
    // Call the send method on the channel Foreign object
    if let Value::LyObj(obj) = channel {
        obj.call_method("send", &[value.clone()])
            .map_err(|e| crate::vm::VmError::Runtime(format!("Send failed: {:?}", e)))
    } else {
        Err(crate::vm::VmError::TypeError {
            expected: "Channel".to_string(),
            actual: format!("{:?}", channel),
        })
    }
}

/// Receive function for channels
pub fn receive(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "Receive".to_string(),
            expected: 1,
            actual: args.len(),
        });
    }
    
    // Extract channel
    let channel = &args[0];
    
    // Call the receive method on the channel Foreign object
    if let Value::LyObj(obj) = channel {
        obj.call_method("receive", &[])
            .map_err(|e| crate::vm::VmError::Runtime(format!("Receive failed: {:?}", e)))
    } else {
        Err(crate::vm::VmError::TypeError {
            expected: "Channel".to_string(),
            actual: format!("{:?}", channel),
        })
    }
}

/// Promise constructor function
pub fn promise(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "Promise".to_string(),
            expected: 1,
            actual: args.len(),
        });
    }
    
    // Create a Promise with resolved value
    let promise = Promise::resolved(args[0].clone())?;
    Ok(Value::LyObj(LyObj::new(Box::new(promise))))
}

/// Await function for futures
pub fn await_future(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "Await".to_string(),
            expected: 1,
            actual: args.len(),
        });
    }
    
    // Extract future or promise
    let future_obj = &args[0];
    
    // Call the await method on the Future/Promise Foreign object
    if let Value::LyObj(obj) = future_obj {
        // Try to get the result if it's completed
        obj.call_method("await", &[])
            .map_err(|e| crate::vm::VmError::Runtime(format!("Await failed: {:?}", e)))
    } else {
        Err(crate::vm::VmError::TypeError {
            expected: "Future or Promise".to_string(),
            actual: format!("{:?}", future_obj),
        })
    }
}

/// ParallelMap function
pub fn parallel_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "ParallelMap".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }
    
    let function = &args[0];
    let list = &args[1];
    
    // Use the implementation from parallel.rs
    parallel::parallel_map_implementation(function, list)
}

/// ParallelReduce function
pub fn parallel_reduce(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(crate::vm::VmError::ArityError {
            function_name: "ParallelReduce".to_string(),
            expected: 2,
            actual: args.len(),
        });
    }
    
    let function = &args[0];
    let list = &args[1];
    
    // Use the implementation from parallel.rs
    parallel::parallel_reduce_implementation(function, list)
}

/// Get the list of all async function names and their implementations
pub fn get_async_functions() -> Vec<(&'static str, fn(&[Value]) -> VmResult<Value>)> {
    vec![
        ("ThreadPool", thread_pool),
        ("Channel", channel),
        ("BoundedChannel", bounded_channel),
        ("Send", send),
        ("Receive", receive),
        ("Promise", promise),
        ("Await", await_future),
        ("ParallelMap", parallel_map),
        ("ParallelReduce", parallel_reduce),
    ]
}