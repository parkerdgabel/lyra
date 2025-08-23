//! Simplified async operations for initial compilation
//!
//! This provides basic stubs for async operations while we work on the full implementation.

use crate::vm::Value;
use crate::foreign::{Foreign, ForeignError};
use std::any::Any;

/// Minimal ThreadPool stub for compilation
#[derive(Debug)]
pub struct SimpleThreadPool {
    worker_count: usize,
}

impl SimpleThreadPool {
    pub fn new(worker_count: usize) -> Self {
        Self { worker_count }
    }
}

impl Foreign for SimpleThreadPool {
    fn type_name(&self) -> &'static str {
        "ThreadPool"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "workerCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.worker_count as i64))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(Self {
            worker_count: self.worker_count,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for SimpleThreadPool {}
unsafe impl Sync for SimpleThreadPool {}

/// Minimal Channel stub for compilation
#[derive(Debug)]
pub struct SimpleChannel;

impl SimpleChannel {
    pub fn new() -> Self {
        Self
    }
}

impl Foreign for SimpleChannel {
    fn type_name(&self) -> &'static str {
        "Channel"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        Err(ForeignError::UnknownMethod {
            type_name: self.type_name().to_string(),
            method: method.to_string(),
        })
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(Self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for SimpleChannel {}
unsafe impl Sync for SimpleChannel {}