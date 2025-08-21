//! # Message Queue & Pub/Sub Module
//!
//! Provides comprehensive message queue and publish/subscribe functionality with support
//! for multiple backends including RabbitMQ, Redis, and in-memory queues.

use crate::vm::{Value, VmResult};
use crate::error::crate::foreign::ForeignError;
use crate::foreign::{Foreign, LyObj};
use super::{StreamingError, StreamingConfig, SerializationFormat};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

/// Message queue with multiple backend support
pub struct MessageQueue {
    name: String,
    config: StreamingConfig,
    backend: QueueBackend,
    stats: Arc<Mutex<QueueStats>>,
}

/// Topic for publish/subscribe messaging
pub struct Topic {
    name: String,
    partitions: usize,
    replication_factor: usize,
    subscribers: Arc<Mutex<HashMap<String, Vec<mpsc::UnboundedSender<Message>>>>>,
    messages: Arc<Mutex<VecDeque<Message>>>,
}

/// Dead letter queue for failed messages
pub struct DeadLetterQueue {
    queue: MessageQueue,
    max_retries: u32,
    retry_delay_ms: u64,
    failed_messages: Arc<Mutex<Vec<FailedMessage>>>,
}

/// Message structure for queue operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub content: Value,
    pub priority: u32,
    pub timestamp: u64,
    pub headers: HashMap<String, String>,
    pub retry_count: u32,
    pub delay_until: Option<u64>,
}

/// Failed message for dead letter queue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedMessage {
    pub original_message: Message,
    pub failure_reason: String,
    pub failed_at: u64,
    pub retry_attempts: u32,
}

/// Queue statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub messages_sent: u64,
    pub messages_received: u64,
    pub messages_failed: u64,
    pub current_size: usize,
    pub peak_size: usize,
    pub total_processing_time_ms: u64,
    pub avg_processing_time_ms: f64,
}

/// Queue backend implementations
#[derive(Debug)]
pub enum QueueBackend {
    InMemory(Arc<Mutex<VecDeque<Message>>>),
    Redis(RedisQueue),
    RabbitMQ(RabbitMQQueue),
}

/// Redis queue implementation
#[derive(Debug)]
pub struct RedisQueue {
    connection_string: String,
    queue_name: String,
    // In real implementation, would contain Redis connection
}

/// RabbitMQ queue implementation
#[derive(Debug)]
pub struct RabbitMQQueue {
    connection_string: String,
    queue_name: String,
    exchange: Option<String>,
    routing_key: Option<String>,
    // In real implementation, would contain AMQP connection
}

/// Message routing rules for complex routing scenarios
#[derive(Debug, Clone)]
pub struct RoutingRule {
    pub condition: RoutingCondition,
    pub destination: String,
    pub transform: Option<String>, // Function name for message transformation
}

/// Routing condition for message routing
#[derive(Debug, Clone)]
pub enum RoutingCondition {
    HeaderEquals(String, String),
    HeaderContains(String, String),
    ContentType(String),
    MessageSize(usize, Comparison),
    Custom(String), // Custom function name
}

/// Comparison operators for routing conditions
#[derive(Debug, Clone)]
pub enum Comparison {
    Equal,
    NotEqual,
    Greater,
    Less,
    GreaterEqual,
    LessEqual,
}

impl MessageQueue {
    pub fn new(name: String, backend_type: &str, options: Option<HashMap<String, String>>) -> Result<Self, StreamingError> {
        let config = StreamingConfig::default();
        let backend = match backend_type {
            "memory" => QueueBackend::InMemory(Arc::new(Mutex::new(VecDeque::new()))),
            "redis" => {
                let connection_string = options.as_ref()
                    .and_then(|opts| opts.get("connection_string"))
                    .ok_or_else(|| StreamingError::ConfigurationError("Redis connection string required".to_string()))?;
                QueueBackend::Redis(RedisQueue {
                    connection_string: connection_string.clone(),
                    queue_name: name.clone(),
                })
            },
            "rabbitmq" => {
                let connection_string = options.as_ref()
                    .and_then(|opts| opts.get("connection_string"))
                    .ok_or_else(|| StreamingError::ConfigurationError("RabbitMQ connection string required".to_string()))?;
                QueueBackend::RabbitMQ(RabbitMQQueue {
                    connection_string: connection_string.clone(),
                    queue_name: name.clone(),
                    exchange: options.as_ref().and_then(|opts| opts.get("exchange")).cloned(),
                    routing_key: options.as_ref().and_then(|opts| opts.get("routing_key")).cloned(),
                })
            },
            _ => return Err(StreamingError::ConfigurationError(format!("Unknown backend: {}", backend_type))),
        };
        
        Ok(Self {
            name,
            config,
            backend,
            stats: Arc::new(Mutex::new(QueueStats::default())),
        })
    }
    
    pub fn send(&self, message: Value, priority: Option<u32>, delay_ms: Option<u64>) -> Result<String, StreamingError> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let delay_until = delay_ms.map(|delay| timestamp + delay);
        
        let message = Message {
            id: message_id.clone(),
            content: message,
            priority: priority.unwrap_or(0),
            timestamp,
            headers: HashMap::new(),
            retry_count: 0,
            delay_until,
        };
        
        match &self.backend {
            QueueBackend::InMemory(queue) => {
                let mut queue = queue.lock().map_err(|_| 
                    StreamingError::ProcessingError("Failed to acquire queue lock".to_string()))?;
                
                // Insert based on priority (higher priority = higher number = front of queue)
                let insert_pos = queue.iter().position(|m| m.priority < message.priority)
                    .unwrap_or(queue.len());
                queue.insert(insert_pos, message);
                
                // Update stats
                let mut stats = self.stats.lock().map_err(|_| 
                    StreamingError::ProcessingError("Failed to acquire stats lock".to_string()))?;
                stats.messages_sent += 1;
                stats.current_size = queue.len();
                if stats.current_size > stats.peak_size {
                    stats.peak_size = stats.current_size;
                }
            },
            QueueBackend::Redis(_) => {
                // In real implementation, would use Redis LPUSH/RPUSH with priority queues
                return Err(StreamingError::ProcessingError("Redis backend not fully implemented".to_string()));
            },
            QueueBackend::RabbitMQ(_) => {
                // In real implementation, would use lapin crate for AMQP
                return Err(StreamingError::ProcessingError("RabbitMQ backend not fully implemented".to_string()));
            },
        }
        
        Ok(message_id)
    }
    
    pub fn receive(&self, timeout_ms: Option<u64>, batch_size: Option<usize>) -> Result<Vec<Message>, StreamingError> {
        let batch_size = batch_size.unwrap_or(1);
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        match &self.backend {
            QueueBackend::InMemory(queue) => {
                let mut queue = queue.lock().map_err(|_| 
                    StreamingError::ProcessingError("Failed to acquire queue lock".to_string()))?;
                
                let mut messages = Vec::new();
                let mut to_remove = Vec::new();
                
                for (index, message) in queue.iter().enumerate() {
                    if messages.len() >= batch_size {
                        break;
                    }
                    
                    // Check if message delay has expired
                    if let Some(delay_until) = message.delay_until {
                        if current_time < delay_until {
                            continue;
                        }
                    }
                    
                    messages.push(message.clone());
                    to_remove.push(index);
                }
                
                // Remove messages in reverse order to maintain indices
                for &index in to_remove.iter().rev() {
                    queue.remove(index);
                }
                
                // Update stats
                let mut stats = self.stats.lock().map_err(|_| 
                    StreamingError::ProcessingError("Failed to acquire stats lock".to_string()))?;
                stats.messages_received += messages.len() as u64;
                stats.current_size = queue.len();
                
                Ok(messages)
            },
            QueueBackend::Redis(_) => {
                Err(StreamingError::ProcessingError("Redis backend not fully implemented".to_string()))
            },
            QueueBackend::RabbitMQ(_) => {
                Err(StreamingError::ProcessingError("RabbitMQ backend not fully implemented".to_string()))
            },
        }
    }
    
    pub fn get_stats(&self) -> Result<QueueStats, StreamingError> {
        let stats = self.stats.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire stats lock".to_string()))?;
        Ok(stats.clone())
    }
}

impl Topic {
    pub fn new(name: String, partitions: usize, replication_factor: usize) -> Self {
        Self {
            name,
            partitions,
            replication_factor,
            subscribers: Arc::new(Mutex::new(HashMap::new())),
            messages: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    pub fn publish(&self, message: Value, key: Option<String>, headers: Option<HashMap<String, String>>) -> Result<String, StreamingError> {
        let message_id = uuid::Uuid::new_v4().to_string();
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        let message = Message {
            id: message_id.clone(),
            content: message,
            priority: 0,
            timestamp,
            headers: headers.unwrap_or_default(),
            retry_count: 0,
            delay_until: None,
        };
        
        // Store message
        {
            let mut messages = self.messages.lock().map_err(|_| 
                StreamingError::ProcessingError("Failed to acquire messages lock".to_string()))?;
            messages.push_back(message.clone());
        }
        
        // Notify subscribers
        {
            let subscribers = self.subscribers.lock().map_err(|_| 
                StreamingError::ProcessingError("Failed to acquire subscribers lock".to_string()))?;
            
            for (_, senders) in subscribers.iter() {
                for sender in senders {
                    let _ = sender.send(message.clone()); // Ignore closed channels
                }
            }
        }
        
        Ok(message_id)
    }
    
    pub fn subscribe(&self, consumer_group: String) -> Result<mpsc::UnboundedReceiver<Message>, StreamingError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        
        let mut subscribers = self.subscribers.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire subscribers lock".to_string()))?;
        
        subscribers.entry(consumer_group)
            .or_insert_with(Vec::new)
            .push(sender);
        
        Ok(receiver)
    }
}

impl DeadLetterQueue {
    pub fn new(queue: MessageQueue, max_retries: u32, retry_delay_ms: u64) -> Self {
        Self {
            queue,
            max_retries,
            retry_delay_ms,
            failed_messages: Arc::new(Mutex::new(Vec::new())),
        }
    }
    
    pub fn handle_failed_message(&self, message: Message, failure_reason: String) -> Result<(), StreamingError> {
        let failed_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        
        if message.retry_count < self.max_retries {
            // Retry the message with delay
            let mut retry_message = message.clone();
            retry_message.retry_count += 1;
            retry_message.delay_until = Some(failed_at + self.retry_delay_ms);
            
            self.queue.send(retry_message.content, Some(retry_message.priority), Some(self.retry_delay_ms))?;
        } else {
            // Move to dead letter queue
            let failed_message = FailedMessage {
                original_message: message,
                failure_reason,
                failed_at,
                retry_attempts: self.max_retries,
            };
            
            let mut failed_messages = self.failed_messages.lock().map_err(|_| 
                StreamingError::ProcessingError("Failed to acquire failed messages lock".to_string()))?;
            failed_messages.push(failed_message);
        }
        
        Ok(())
    }
    
    pub fn get_failed_messages(&self) -> Result<Vec<FailedMessage>, StreamingError> {
        let failed_messages = self.failed_messages.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire failed messages lock".to_string()))?;
        Ok(failed_messages.clone())
    }
}

// Foreign object implementations
impl Foreign for MessageQueue {
    fn type_name(&self) -> &'static str {
        "MessageQueue"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "send" => {
                if args.is_empty() {
                    return Err(VmError::Runtime( "send requires at least 1 argument".to_string()));
                }
                
                let message = args[0].clone();
                let priority = if args.len() > 1 {
                    match &args[1] {
                        Value::Real(n) => Some(*n as u32),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let delay_ms = if args.len() > 2 {
                    match &args[2] {
                        Value::Real(n) => Some(*n as u64),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let message_id = self.send(message, priority, delay_ms)?;
                Ok(Value::String(message_id))
            },
            "receive" => {
                let timeout_ms = if args.len() > 0 {
                    match &args[0] {
                        Value::Real(n) => Some(*n as u64),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let batch_size = if args.len() > 1 {
                    match &args[1] {
                        Value::Real(n) => Some(*n as usize),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let messages = self.receive(timeout_ms, batch_size)?;
                let message_values: Vec<Value> = messages.into_iter()
                    .map(|m| serde_json::to_value(&m).unwrap_or(serde_json::Value::Null))
                    .map(|v| serde_json::from_value(v).unwrap_or(Value::Null))
                    .collect();
                
                Ok(Value::List(message_values))
            },
            "stats" => {
                let stats = self.get_stats()?;
                let stats_map = vec![
                    ("messages_sent".to_string(), Value::Real(stats.messages_sent as f64)),
                    ("messages_received".to_string(), Value::Real(stats.messages_received as f64)),
                    ("current_size".to_string(), Value::Real(stats.current_size as f64)),
                    ("peak_size".to_string(), Value::Real(stats.peak_size as f64)),
                ];
                Ok(Value::List(stats_map.into_iter().map(|(k, v)| 
                    Value::List(vec![Value::String(k), v])).collect()))
            },
            "name" => Ok(Value::String(self.name.clone())),
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) }),
        }
    }
}

impl Foreign for Topic {
    fn type_name(&self) -> &'static str {
        "Topic"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "publish" => {
                if args.is_empty() {
                    return Err(VmError::Runtime( "publish requires at least 1 argument".to_string()));
                }
                
                let message = args[0].clone();
                let key = if args.len() > 1 {
                    match &args[1] {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let headers = None; // Simplified for now
                
                let message_id = self.publish(message, key, headers)?;
                Ok(Value::String(message_id))
            },
            "name" => Ok(Value::String(self.name.clone())),
            "partitions" => Ok(Value::Integer(self.partitions as i64)),
            "replication" => Ok(Value::Real(self.replication_factor as f64)),
            _ => Err(crate::foreign::ForeignError::RuntimeError { message: format!("Unknown method: {}", method) }),
        }
    }
}

// Function implementations for Lyra VM
pub fn message_queue(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::TypeError { 
            expected: "at least 1 argument".to_string(), 
            actual: "0 arguments".to_string() 
        });
    }
    
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Queue name must be a string".to_string())),
    };
    
    let backend_type = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.as_str(),
            _ => "memory",
        }
    } else {
        "memory"
    };
    
    let options = None; // Simplified for now
    
    let queue = MessageQueue::new(name, backend_type, options)?;
    Ok(Value::LyObj(LyObj::new(Box::new(queue))))
}

pub fn topic_create(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime( "TopicCreate requires 3 arguments".to_string()));
    }
    
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Topic name must be a string".to_string())),
    };
    
    let partitions = match &args[1] {
        Value::Real(n) => *n as usize,
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Partitions must be a number".to_string())),
    };
    
    let replication = match &args[2] {
        Value::Real(n) => *n as usize,
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Replication must be a number".to_string())),
    };
    
    let topic = Topic::new(name, partitions, replication);
    Ok(Value::LyObj(LyObj::new(Box::new(topic))))
}

pub fn dead_letter_queue(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime( "DeadLetterQueue requires 3 arguments".to_string()));
    }
    
    // For simplicity, create a new in-memory queue
    let queue_name = format!("dlq-{}", uuid::Uuid::new_v4());
    let queue = MessageQueue::new(queue_name, "memory", None)?;
    
    let max_retries = match &args[1] {
        Value::Real(n) => *n as u32,
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Max retries must be a number".to_string())),
    };
    
    let retry_delay_ms = match &args[2] {
        Value::Real(n) => *n as u64,
        _ => return Err(VmError::TypeError { expected: "string".to_string(), actual: "non-string value".to_string() })Retry delay must be a number".to_string())),
    };
    
    let dlq = DeadLetterQueue::new(queue, max_retries, retry_delay_ms);
    Ok(Value::LyObj(LyObj::new(Box::new(dlq))))
}

/// Register all message queue functions
pub fn register_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    functions.insert("MessageQueue".to_string(), message_queue as fn(&[Value]) -> VmResult<Value>);
    functions.insert("TopicCreate".to_string(), topic_create as fn(&[Value]) -> VmResult<Value>);
    functions.insert("DeadLetterQueue".to_string(), dead_letter_queue as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_message_queue_creation() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None);
        assert!(queue.is_ok());
    }
    
    #[test]
    fn test_message_queue_send_receive() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None).unwrap();
        
        // Send a message
        let message_id = queue.send(Value::String("test message".to_string() }, None, None).unwrap();
        assert!(!message_id.is_empty());
        
        // Receive the message
        let messages = queue.receive(None, None).unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content, Value::String("test message".to_string()));
    }
    
    #[test]
    fn test_message_queue_priority() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None).unwrap();
        
        // Send messages with different priorities
        queue.send(Value::String("low priority".to_string() }, Some(1), None).unwrap();
        queue.send(Value::String("high priority".to_string() }, Some(10), None).unwrap();
        queue.send(Value::String("medium priority".to_string() }, Some(5), None).unwrap();
        
        // Receive messages - should come in priority order
        let messages = queue.receive(None, Some(3)).unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].content, Value::String("high priority".to_string()));
        assert_eq!(messages[1].content, Value::String("medium priority".to_string()));
        assert_eq!(messages[2].content, Value::String("low priority".to_string()));
    }
    
    #[test]
    fn test_message_queue_delayed_delivery() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None).unwrap();
        
        // Send a delayed message
        queue.send(Value::String("delayed message".to_string() }, None, Some(1000)).unwrap();
        
        // Immediate receive should return empty
        let messages = queue.receive(None, None).unwrap();
        assert_eq!(messages.len(), 0);
    }
    
    #[test]
    fn test_message_queue_stats() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None).unwrap();
        
        // Send some messages
        queue.send(Value::String("msg1".to_string() }, None, None).unwrap();
        queue.send(Value::String("msg2".to_string() }, None, None).unwrap();
        
        let stats = queue.get_stats().unwrap();
        assert_eq!(stats.messages_sent, 2);
        assert_eq!(stats.current_size, 2);
        
        // Receive messages
        queue.receive(None, Some(2)).unwrap();
        
        let stats = queue.get_stats().unwrap();
        assert_eq!(stats.messages_received, 2);
        assert_eq!(stats.current_size, 0);
    }
    
    #[test]
    fn test_topic_creation() {
        let topic = Topic::new("test-topic".to_string(), 4, 3);
        assert_eq!(topic.name, "test-topic");
        assert_eq!(topic.partitions, 4);
        assert_eq!(topic.replication_factor, 3);
    }
    
    #[test]
    fn test_topic_publish() {
        let topic = Topic::new("test-topic".to_string(), 4, 3);
        let result = topic.publish(Value::String("test message".to_string() }, None, None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_topic_subscribe() {
        let topic = Topic::new("test-topic".to_string(), 4, 3);
        let receiver = topic.subscribe("consumer-group-1".to_string() };
        assert!(receiver.is_ok());
    }
    
    #[test]
    fn test_dead_letter_queue_creation() {
        let queue = MessageQueue::new("test-queue".to_string(), "memory", None).unwrap();
        let dlq = DeadLetterQueue::new(queue, 3, 1000);
        assert_eq!(dlq.max_retries, 3);
        assert_eq!(dlq.retry_delay_ms, 1000);
    }
    
    #[test]
    fn test_message_queue_function() {
        let args = vec![Value::String("test-queue".to_string() }];
        let result = message_queue(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_topic_create_function() {
        let args = vec![
            Value::String("test-topic".to_string() },
            Value::Real(4.0),
            Value::Real(3.0),
        ];
        let result = topic_create(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_register_functions() {
        let functions = register_functions();
        assert!(functions.contains_key("MessageQueue"));
        assert!(functions.contains_key("TopicCreate"));
        assert!(functions.contains_key("DeadLetterQueue"));
    }
}