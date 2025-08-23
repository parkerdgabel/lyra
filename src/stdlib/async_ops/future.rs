//! Future and Promise Foreign Object Implementation
//!
//! Provides async computation results as Foreign objects.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError};
use std::sync::{Arc, Mutex};
use std::any::Any;

/// Future represents an async computation result
#[derive(Debug)]
pub struct Future {
    value: Arc<Mutex<Option<VmResult<Value>>>>,
}

impl Future {
    /// Create a new pending future
    pub fn new() -> Self {
        Self {
            value: Arc::new(Mutex::new(None)),
        }
    }

    /// Create a future that's already resolved
    pub fn resolved(value: Value) -> Self {
        Self {
            value: Arc::new(Mutex::new(Some(Ok(value)))),
        }
    }

    /// Create a future that's already rejected
    pub fn rejected(error: VmError) -> Self {
        Self {
            value: Arc::new(Mutex::new(Some(Err(error)))),
        }
    }

    /// Check if the future is completed
    pub fn is_completed(&self) -> bool {
        if let Ok(guard) = self.value.lock() {
            guard.is_some()
        } else {
            false
        }
    }

    /// Get the result if completed
    pub fn get_result(&self) -> Option<VmResult<Value>> {
        if let Ok(guard) = self.value.lock() {
            // Clone the inner value, not the guard
            guard.as_ref().map(|result| match result {
                Ok(value) => Ok(value.clone()),
                Err(_) => Err(VmError::Runtime("Future error".to_string())),
            })
        } else {
            None
        }
    }
}

impl Foreign for Future {
    fn type_name(&self) -> &'static str {
        "Future"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "isCompleted" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_completed()))
            },
            "getResult" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                match self.get_result() {
                    Some(Ok(value)) => Ok(value),
                    Some(Err(e)) => Err(ForeignError::RuntimeError {
                        message: format!("Future failed: {:?}", e),
                    }),
                    None => Ok(Value::Symbol("Pending".to_string())),
                }
            },
            "await" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                // For await, we block until result is available or return error if pending
                match self.get_result() {
                    Some(Ok(value)) => Ok(value),
                    Some(Err(e)) => Err(ForeignError::RuntimeError {
                        message: format!("Future failed: {:?}", e),
                    }),
                    None => Err(ForeignError::RuntimeError {
                        message: "Future is still pending".to_string(),
                    }),
                }
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(Future {
            value: Arc::clone(&self.value),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for Future {}
unsafe impl Sync for Future {}

/// Promise allows manual control over Future completion
#[derive(Debug)]
pub struct Promise {
    future: Future,
}

impl Promise {
    /// Create a new Promise with its associated Future
    pub fn new() -> Self {
        Self {
            future: Future::new(),
        }
    }

    /// Create an already resolved Promise
    pub fn resolved(value: Value) -> VmResult<Self> {
        Ok(Self {
            future: Future::resolved(value),
        })
    }

    /// Resolve the Promise with a value
    pub fn resolve(&self, value: Value) -> VmResult<()> {
        if let Ok(mut guard) = self.future.value.lock() {
            if guard.is_some() {
                return Err(VmError::Runtime( "Promise already resolved".to_string()
                ));
            }
            *guard = Some(Ok(value));
            Ok(())
        } else {
            Err(VmError::Runtime(
                "Failed to acquire promise lock".to_string()
            ))
        }
    }

    /// Reject the Promise with an error
    pub fn reject(&self, error: VmError) -> VmResult<()> {
        if let Ok(mut guard) = self.future.value.lock() {
            if guard.is_some() {
                return Err(VmError::Runtime( "Promise already resolved".to_string()
                ));
            }
            *guard = Some(Err(error));
            Ok(())
        } else {
            Err(VmError::Runtime(
                "Failed to acquire promise lock".to_string()
            ))
        }
    }

    /// Get the associated Future
    pub fn future(&self) -> &Future {
        &self.future
    }
}

impl Foreign for Promise {
    fn type_name(&self) -> &'static str {
        "Promise"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "resolve" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                self.resolve(args[0].clone())
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Resolve failed: {:?}", e),
                    })?;

                Ok(Value::Symbol("Ok".to_string()))
            },
            "reject" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let error_msg = match &args[0] {
                    Value::String(s) => s.clone(),
                    other => format!("{:?}", other),
                };

                let error = VmError::Runtime(error_msg);
                self.reject(error)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Reject failed: {:?}", e),
                    })?;

                Ok(Value::Symbol("Ok".to_string()))
            },
            "future" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                // This would need to return the Future as a LyObj
                // For now, return a placeholder
                Ok(Value::Symbol("Future".to_string()))
            },
            "await" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                // Delegate to the future's await method
                match self.future.get_result() {
                    Some(Ok(value)) => Ok(value),
                    Some(Err(e)) => Err(ForeignError::RuntimeError {
                        message: format!("Promise failed: {:?}", e),
                    }),
                    None => Err(ForeignError::RuntimeError {
                        message: "Promise is still pending".to_string(),
                    }),
                }
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(Promise {
            future: Future {
                value: Arc::clone(&self.future.value),
            },
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for Promise {}
unsafe impl Sync for Promise {}