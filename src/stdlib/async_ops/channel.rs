//! Channel Foreign Object Implementation
//!
//! Provides thread-safe message passing channels as Foreign objects.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError};
use crossbeam_channel::{Sender, Receiver, unbounded, bounded};
use std::any::Any;

/// Unbounded Channel Foreign object
#[derive(Debug)]
pub struct Channel {
    sender: Sender<Value>,
    receiver: Receiver<Value>,
}

impl Channel {
    /// Create a new unbounded channel
    pub fn new() -> VmResult<Self> {
        let (sender, receiver) = unbounded();
        Ok(Self { sender, receiver })
    }

    /// Send a value (non-blocking for unbounded channels)
    pub fn send(&self, value: Value) -> VmResult<()> {
        self.sender.send(value)
            .map_err(|_| VmError::Runtime(
                "Channel is closed".to_string()))
    }

    /// Receive a value (blocking)
    pub fn receive(&self) -> VmResult<Value> {
        self.receiver.recv()
            .map_err(|_| VmError::Runtime(
                "Channel is closed and empty".to_string()))
    }

    /// Try to send a value (non-blocking)
    pub fn try_send(&self, value: Value) -> VmResult<bool> {
        match self.sender.try_send(value) {
            Ok(_) => Ok(true),
            Err(crossbeam_channel::TrySendError::Full(_)) => Ok(false),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(VmError::Runtime(
                    "Channel is closed".to_string()))
            }
        }
    }

    /// Try to receive a value (non-blocking)
    pub fn try_receive(&self) -> VmResult<Option<Value>> {
        match self.receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(VmError::Runtime(
                    "Channel is closed and empty".to_string()))
            }
        }
    }

    /// Get the number of messages in the channel
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Close the channel
    pub fn close(&self) {
        // Dropping the sender will close the channel
        // We can't actually drop it here, but we can simulate closure
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

                self.send(args[0].clone())
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Send failed: {:?}", e),
                    })?;

                Ok(Value::Symbol("Ok".to_string()))
            },
            "receive" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                self.receive()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Receive failed: {:?}", e),
                    })
            },
            "trySend" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let success = self.try_send(args[0].clone())
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("TrySend failed: {:?}", e),
                    })?;

                Ok(Value::Boolean(success))
            },
            "tryReceive" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let result = self.try_receive()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("TryReceive failed: {:?}", e),
                    })?;

                match result {
                    Some(value) => Ok(value),
                    None => Ok(Value::Symbol("Empty".to_string())),
                }
            },
            "len" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.len() as i64))
            },
            "isEmpty" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_empty()))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(Channel {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for Channel {}
unsafe impl Sync for Channel {}

/// Bounded Channel Foreign object
#[derive(Debug)]
pub struct BoundedChannel {
    sender: Sender<Value>,
    receiver: Receiver<Value>,
    capacity: usize,
}

impl BoundedChannel {
    /// Create a new bounded channel with specified capacity
    pub fn new(capacity: usize) -> VmResult<Self> {
        let (sender, receiver) = bounded(capacity);
        Ok(Self { sender, receiver, capacity })
    }

    /// Send a value (blocking if channel is full)
    pub fn send(&self, value: Value) -> VmResult<()> {
        self.sender.send(value)
            .map_err(|_| VmError::Runtime(
                "Channel is closed".to_string()))
    }

    /// Receive a value (blocking)
    pub fn receive(&self) -> VmResult<Value> {
        self.receiver.recv()
            .map_err(|_| VmError::Runtime(
                "Channel is closed and empty".to_string()))
    }

    /// Try to send a value (non-blocking)
    pub fn try_send(&self, value: Value) -> VmResult<bool> {
        match self.sender.try_send(value) {
            Ok(_) => Ok(true),
            Err(crossbeam_channel::TrySendError::Full(_)) => Ok(false),
            Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                Err(VmError::Runtime(
                    "Channel is closed".to_string()))
            }
        }
    }

    /// Try to receive a value (non-blocking)
    pub fn try_receive(&self) -> VmResult<Option<Value>> {
        match self.receiver.try_recv() {
            Ok(value) => Ok(Some(value)),
            Err(crossbeam_channel::TryRecvError::Empty) => Ok(None),
            Err(crossbeam_channel::TryRecvError::Disconnected) => {
                Err(VmError::Runtime(
                    "Channel is closed and empty".to_string()))
            }
        }
    }

    /// Get the capacity of the channel
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the number of messages in the channel
    pub fn len(&self) -> usize {
        self.receiver.len()
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    /// Check if the channel is full
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }
}

impl Foreign for BoundedChannel {
    fn type_name(&self) -> &'static str {
        "BoundedChannel"
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

                self.send(args[0].clone())
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Send failed: {:?}", e),
                    })?;

                Ok(Value::Symbol("Ok".to_string()))
            },
            "receive" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                self.receive()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("Receive failed: {:?}", e),
                    })
            },
            "trySend" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let success = self.try_send(args[0].clone())
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("TrySend failed: {:?}", e),
                    })?;

                Ok(Value::Boolean(success))
            },
            "tryReceive" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }

                let result = self.try_receive()
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("TryReceive failed: {:?}", e),
                    })?;

                match result {
                    Some(value) => Ok(value),
                    None => Ok(Value::Symbol("Empty".to_string())),
                }
            },
            "capacity" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.capacity() as i64))
            },
            "len" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.len() as i64))
            },
            "isEmpty" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_empty()))
            },
            "isFull" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_full()))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(BoundedChannel {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
            capacity: self.capacity,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

unsafe impl Send for BoundedChannel {}
unsafe impl Sync for BoundedChannel {}