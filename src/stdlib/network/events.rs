//! Event-Driven Architecture and Stream Processing
//!
//! This module implements event streams, publish/subscribe patterns, and message queues
//! as symbolic objects for real-time data processing in distributed systems.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};
use std::sync::{Arc, Mutex};

/// Event data structure representing a single event
#[derive(Debug, Clone)]
pub struct Event {
    /// Event ID (unique identifier)
    pub id: String,
    /// Event type/topic
    pub event_type: String,
    /// Event data payload
    pub data: Value,
    /// Event metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when event was created
    pub timestamp: SystemTime,
    /// Source of the event
    pub source: String,
}

impl Event {
    /// Create a new event
    pub fn new(event_type: String, data: Value, source: String) -> Self {
        let id = format!("event_{}", SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_nanos());
        
        Self {
            id,
            event_type,
            data,
            metadata: HashMap::new(),
            timestamp: SystemTime::now(),
            source,
        }
    }
    
    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Check if event matches pattern
    pub fn matches_pattern(&self, pattern: &HashMap<String, String>) -> bool {
        for (key, value) in pattern {
            match key.as_str() {
                "type" => {
                    if &self.event_type != value {
                        return false;
                    }
                }
                "source" => {
                    if &self.source != value {
                        return false;
                    }
                }
                _ => {
                    // Check metadata
                    if let Some(meta_value) = self.metadata.get(key) {
                        if meta_value != value {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
            }
        }
        true
    }
}

impl Foreign for Event {
    fn type_name(&self) -> &'static str {
        "Event"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "ID" => Ok(Value::String(self.id.clone())),
            "Type" => Ok(Value::String(self.event_type.clone())),
            "Data" => Ok(self.data.clone()),
            "Source" => Ok(Value::String(self.source.clone())),
            "Timestamp" => {
                let timestamp = self.timestamp.duration_since(UNIX_EPOCH)
                    .unwrap_or_default().as_secs_f64();
                Ok(Value::Real(timestamp))
            }
            "Metadata" => {
                let entries: Vec<Value> = self.metadata.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
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

/// Event stream for real-time event processing
#[derive(Debug, Clone)]
pub struct EventStream {
    /// Stream URL or identifier
    pub url: String,
    /// Pattern for filtering events
    pub pattern: HashMap<String, String>,
    /// Buffer size for event storage
    pub buffer_size: usize,
    /// Event buffer
    pub events: Arc<Mutex<VecDeque<Event>>>,
    /// Stream statistics
    pub events_received: u64,
    pub events_filtered: u64,
    /// Stream state
    pub is_active: bool,
    pub created_at: SystemTime,
    /// Stream metadata
    pub metadata: HashMap<String, String>,
}

impl EventStream {
    /// Create a new event stream
    pub fn new(url: String) -> Self {
        Self {
            url,
            pattern: HashMap::new(),
            buffer_size: 1000,
            events: Arc::new(Mutex::new(VecDeque::new())),
            events_received: 0,
            events_filtered: 0,
            is_active: false,
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Set filter pattern
    pub fn with_pattern(mut self, pattern: HashMap<String, String>) -> Self {
        self.pattern = pattern;
        self
    }
    
    /// Set buffer size
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }
    
    /// Add an event to the stream
    pub fn add_event(&mut self, event: Event) -> Result<(), String> {
        if !event.matches_pattern(&self.pattern) {
            self.events_filtered += 1;
            return Ok(());
        }
        
        let mut events = self.events.lock()
            .map_err(|e| format!("Failed to lock events buffer: {}", e))?;
        
        // Ensure buffer doesn't exceed max size
        while events.len() >= self.buffer_size {
            events.pop_front();
        }
        
        events.push_back(event);
        self.events_received += 1;
        
        Ok(())
    }
    
    /// Get the next event from the stream
    pub fn get_event(&mut self) -> Result<Option<Event>, String> {
        let mut events = self.events.lock()
            .map_err(|e| format!("Failed to lock events buffer: {}", e))?;
        
        Ok(events.pop_front())
    }
    
    /// Get all available events
    pub fn get_all_events(&mut self) -> Result<Vec<Event>, String> {
        let mut events = self.events.lock()
            .map_err(|e| format!("Failed to lock events buffer: {}", e))?;
        
        let all_events: Vec<Event> = events.drain(..).collect();
        Ok(all_events)
    }
    
    /// Check if stream has events
    pub fn has_events(&self) -> bool {
        if let Ok(events) = self.events.lock() {
            !events.is_empty()
        } else {
            false
        }
    }
    
    /// Get number of buffered events
    pub fn event_count(&self) -> usize {
        if let Ok(events) = self.events.lock() {
            events.len()
        } else {
            0
        }
    }
    
    /// Start the stream
    pub fn start(&mut self) {
        self.is_active = true;
    }
    
    /// Stop the stream
    pub fn stop(&mut self) {
        self.is_active = false;
    }
}

impl Foreign for EventStream {
    fn type_name(&self) -> &'static str {
        "EventStream"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "URL" => Ok(Value::String(self.url.clone())),
            "IsActive" => Ok(Value::Integer(if self.is_active { 1 } else { 0 })),
            "BufferSize" => Ok(Value::Integer(self.buffer_size as i64)),
            "EventCount" => Ok(Value::Integer(self.event_count() as i64)),
            "EventsReceived" => Ok(Value::Integer(self.events_received as i64)),
            "EventsFiltered" => Ok(Value::Integer(self.events_filtered as i64)),
            "HasEvents" => Ok(Value::Integer(if self.has_events() { 1 } else { 0 })),
            "Pattern" => {
                let entries: Vec<Value> = self.pattern.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
            }
            "CreatedAt" => {
                let timestamp = self.created_at.duration_since(UNIX_EPOCH)
                    .unwrap_or_default().as_secs_f64();
                Ok(Value::Real(timestamp))
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

/// Message queue for persistent async messaging
#[derive(Debug, Clone)]
pub struct MessageQueue {
    /// Queue name/identifier
    pub name: String,
    /// Message buffer
    pub messages: Arc<Mutex<VecDeque<Value>>>,
    /// Queue configuration
    pub max_size: usize,
    pub is_persistent: bool,
    /// Queue statistics
    pub messages_sent: u64,
    pub messages_received: u64,
    /// Queue state
    pub is_active: bool,
    pub created_at: SystemTime,
    /// Queue metadata
    pub metadata: HashMap<String, String>,
}

impl MessageQueue {
    /// Create a new message queue
    pub fn new(name: String) -> Self {
        Self {
            name,
            messages: Arc::new(Mutex::new(VecDeque::new())),
            max_size: 10000,
            is_persistent: false,
            messages_sent: 0,
            messages_received: 0,
            is_active: true,
            created_at: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Send a message to the queue
    pub fn send(&mut self, message: Value) -> Result<(), String> {
        if !self.is_active {
            return Err("Queue is not active".to_string());
        }
        
        let mut messages = self.messages.lock()
            .map_err(|e| format!("Failed to lock message queue: {}", e))?;
        
        // Check queue size limit
        if messages.len() >= self.max_size {
            return Err(format!("Queue is full (max size: {})", self.max_size));
        }
        
        messages.push_back(message);
        self.messages_sent += 1;
        
        Ok(())
    }
    
    /// Receive a message from the queue
    pub fn receive(&mut self) -> Result<Option<Value>, String> {
        let mut messages = self.messages.lock()
            .map_err(|e| format!("Failed to lock message queue: {}", e))?;
        
        if let Some(message) = messages.pop_front() {
            self.messages_received += 1;
            Ok(Some(message))
        } else {
            Ok(None)
        }
    }
    
    /// Get queue size
    pub fn size(&self) -> usize {
        if let Ok(messages) = self.messages.lock() {
            messages.len()
        } else {
            0
        }
    }
    
    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.size() == 0
    }
    
    /// Clear all messages
    pub fn clear(&mut self) -> Result<(), String> {
        let mut messages = self.messages.lock()
            .map_err(|e| format!("Failed to lock message queue: {}", e))?;
        
        messages.clear();
        Ok(())
    }
}

impl Foreign for MessageQueue {
    fn type_name(&self) -> &'static str {
        "MessageQueue"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Name" => Ok(Value::String(self.name.clone())),
            "Size" => Ok(Value::Integer(self.size() as i64)),
            "IsEmpty" => Ok(Value::Integer(if self.is_empty() { 1 } else { 0 })),
            "IsActive" => Ok(Value::Integer(if self.is_active { 1 } else { 0 })),
            "MaxSize" => Ok(Value::Integer(self.max_size as i64)),
            "IsPersistent" => Ok(Value::Integer(if self.is_persistent { 1 } else { 0 })),
            "MessagesSent" => Ok(Value::Integer(self.messages_sent as i64)),
            "MessagesReceived" => Ok(Value::Integer(self.messages_received as i64)),
            "CreatedAt" => {
                let timestamp = self.created_at.duration_since(UNIX_EPOCH)
                    .unwrap_or_default().as_secs_f64();
                Ok(Value::Real(timestamp))
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

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// EventStream[url, pattern] - Create event stream
pub fn event_stream(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (url, [pattern])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let url = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut stream = EventStream::new(url);
    
    // Apply pattern if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(pattern_list) => {
                let mut pattern = HashMap::new();
                for item in pattern_list {
                    if let Value::List(pair) = item {
                        if pair.len() == 2 {
                            if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                                pattern.insert(key.clone(), value.clone());
                            }
                        }
                    }
                }
                stream = stream.with_pattern(pattern);
            }
            _ => return Err(VmError::TypeError {
                expected: "List of pattern pairs".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(stream))))
}

/// EventSubscribe[stream, pattern] - Subscribe to events
pub fn event_subscribe(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (stream, pattern)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(stream) = obj.downcast_ref::<EventStream>() {
                let mut new_stream = stream.clone();
                
                // Parse pattern
                match &args[1] {
                    Value::List(pattern_list) => {
                        let mut pattern = HashMap::new();
                        for item in pattern_list {
                            if let Value::List(pair) = item {
                                if pair.len() == 2 {
                                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                                        pattern.insert(key.clone(), value.clone());
                                    }
                                }
                            }
                        }
                        new_stream.pattern = pattern;
                        new_stream.start();
                        Ok(Value::LyObj(LyObj::new(Box::new(new_stream))))
                    }
                    _ => Err(VmError::TypeError {
                        expected: "List of pattern pairs".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "EventStream object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "EventStream object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// EventPublish[stream, event] - Publish event to stream
pub fn event_publish(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (stream, event_type, data, [source])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let event_type = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for event type".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let data = args[2].clone();
    
    let source = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "String for source".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else {
        "lyra".to_string()
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(stream) = obj.downcast_ref::<EventStream>() {
                let mut new_stream = stream.clone();
                let event = Event::new(event_type, data, source);
                
                match new_stream.add_event(event) {
                    Ok(()) => Ok(Value::LyObj(LyObj::new(Box::new(new_stream)))),
                    Err(e) => Err(VmError::Runtime(format!("Failed to publish event: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "EventStream object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "EventStream object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// MessageQueue[name, options] - Create message queue
pub fn message_queue(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (name, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for queue name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut queue = MessageQueue::new(name);
    
    // Apply options if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(options) => {
                for option in options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "maxSize" => {
                                        if let Value::Integer(size) = value {
                                            queue.max_size = *size as usize;
                                        }
                                    }
                                    "persistent" => {
                                        if let Value::Integer(flag) = value {
                                            queue.is_persistent = *flag != 0;
                                        }
                                    }
                                    _ => {
                                        // Add to metadata
                                        if let Value::String(val) = value {
                                            queue.metadata.insert(key.clone(), val.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            _ => return Err(VmError::TypeError {
                expected: "List of option pairs".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
}

/// NetworkChannel[endpoint] - Create network-backed channel
pub fn network_channel(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (endpoint)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let endpoint = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for endpoint".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // For now, create a message queue that represents a network channel
    // In a full implementation, this would create a distributed channel
    // that can send messages across network boundaries
    let mut queue = MessageQueue::new(format!("network_channel_{}", endpoint));
    queue.metadata.insert("type".to_string(), "network_channel".to_string());
    queue.metadata.insert("endpoint".to_string(), endpoint);
    
    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
}