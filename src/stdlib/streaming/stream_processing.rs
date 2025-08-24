//! # Stream Processing Module
//!
//! Provides advanced stream processing capabilities including windowed aggregations,
//! stream joins, complex event processing, and backpressure management.

use crate::vm::{Value, VmResult};
use crate::error::LyraError;
use crate::foreign::{Foreign, LyObj};
use super::{StreamingError, StreamingConfig, StreamSource, StreamSink};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use serde::{Serialize, Deserialize};

/// Windowed aggregation processor for stream data
pub struct WindowAggregate {
    window_type: WindowType,
    window_size: Duration,
    aggregation: AggregationType,
    data_buffer: Arc<Mutex<VecDeque<TimestampedValue>>>,
    results: Arc<Mutex<VecDeque<AggregationResult>>>,
    watermark: Arc<Mutex<u64>>,
}

/// Stream join processor for combining multiple streams
pub struct StreamJoin {
    left_stream_buffer: Arc<Mutex<BTreeMap<u64, Vec<JoinValue>>>>,
    right_stream_buffer: Arc<Mutex<BTreeMap<u64, Vec<JoinValue>>>>,
    join_key: String,
    window_size: Duration,
    results: Arc<Mutex<VecDeque<JoinResult>>>,
}

/// Complex Event Processing engine for pattern matching
pub struct ComplexEventProcessor {
    patterns: Arc<Mutex<Vec<EventPattern>>>,
    event_buffer: Arc<Mutex<VecDeque<ProcessingEvent>>>,
    pattern_state: Arc<Mutex<HashMap<String, PatternState>>>,
    actions: Arc<Mutex<HashMap<String, EventAction>>>,
}

/// Stream reducer for folding operations
pub struct StreamReducer {
    reduce_function: Arc<dyn Fn(Value, Value) -> Result<Value, StreamingError> + Send + Sync>,
    accumulator: Arc<Mutex<Value>>,
    window_size: Option<Duration>,
    buffer: Arc<Mutex<VecDeque<TimestampedValue>>>,
}

/// Stream transformer for data transformation
pub struct StreamTransformer {
    transform_function: Arc<dyn Fn(Value) -> Result<Value, StreamingError> + Send + Sync>,
    parallelism: usize,
    input_buffer: Arc<Mutex<VecDeque<Value>>>,
    output_buffer: Arc<Mutex<VecDeque<Value>>>,
}

/// Watermark strategy for handling late data
pub struct WatermarkStrategy {
    lateness_threshold: Duration,
    current_watermark: Arc<Mutex<u64>>,
    max_event_time: Arc<Mutex<u64>>,
}

/// Stream checkpoint manager for fault tolerance
pub struct StreamCheckpoint {
    interval: Duration,
    storage: CheckpointStorage,
    last_checkpoint: Arc<Mutex<u64>>,
    checkpoint_data: Arc<Mutex<HashMap<String, Value>>>,
}

/// Backpressure controller for flow control
pub struct BackpressureController {
    strategy: BackpressureStrategy,
    thresholds: BackpressureThresholds,
    current_pressure: Arc<Mutex<f64>>,
    metrics: Arc<Mutex<BackpressureMetrics>>,
}

/// Window types for aggregation
#[derive(Debug, Clone)]
pub enum WindowType {
    Tumbling,   // Fixed, non-overlapping windows
    Sliding,    // Overlapping windows
    Session,    // Data-driven windows based on activity gaps
    Global,     // Single window for all data
}

/// Aggregation types
#[derive(Debug, Clone)]
pub enum AggregationType {
    Count,
    Sum,
    Average,
    Min,
    Max,
    First,
    Last,
    Custom(String), // Custom aggregation function name
}

/// Timestamped value for stream processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimestampedValue {
    pub value: Value,
    pub timestamp: u64,
    pub event_time: Option<u64>,
}

/// Aggregation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationResult {
    pub window_start: u64,
    pub window_end: u64,
    pub result: Value,
    pub count: usize,
}

/// Value for stream joins
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinValue {
    pub value: Value,
    pub join_key: String,
    pub timestamp: u64,
}

/// Join result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoinResult {
    pub left_value: Value,
    pub right_value: Value,
    pub join_key: String,
    pub timestamp: u64,
}

/// Event for complex event processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingEvent {
    pub event_type: String,
    pub data: Value,
    pub timestamp: u64,
    pub source: Option<String>,
}

/// Event pattern for CEP
#[derive(Debug, Clone)]
pub struct EventPattern {
    pub id: String,
    pub sequence: Vec<PatternElement>,
    pub within: Option<Duration>,
    pub action: String, // Action function name
}

/// Pattern element
#[derive(Debug, Clone)]
pub struct PatternElement {
    pub event_type: String,
    pub condition: Option<String>, // Condition function name
    pub optional: bool,
}

/// Pattern matching state
#[derive(Debug, Clone)]
pub struct PatternState {
    pub matched_events: Vec<ProcessingEvent>,
    pub current_position: usize,
    pub start_time: u64,
}

/// Event action
pub type EventAction = Box<dyn Fn(Vec<ProcessingEvent>) -> Result<Value, StreamingError> + Send + Sync>;

/// Checkpoint storage
#[derive(Debug, Clone)]
pub enum CheckpointStorage {
    InMemory,
    FileSystem(String), // Directory path
    Database(String),   // Connection string
    ObjectStorage(String), // S3/GCS path
}

/// Backpressure strategies
#[derive(Debug, Clone)]
pub enum BackpressureStrategy {
    Block,      // Block upstream when buffers are full
    Drop,       // Drop oldest messages
    Sample,     // Sample messages at reduced rate
    Adaptive,   // Dynamically adjust based on conditions
}

/// Backpressure thresholds
#[derive(Debug, Clone)]
pub struct BackpressureThresholds {
    pub warning_level: f64,   // 0.0 - 1.0
    pub critical_level: f64,  // 0.0 - 1.0
    pub recovery_level: f64,  // 0.0 - 1.0
}

/// Backpressure metrics
#[derive(Debug, Clone, Default)]
pub struct BackpressureMetrics {
    pub buffer_utilization: f64,
    pub throughput_rate: f64,
    pub latency_p95: f64,
    pub dropped_messages: u64,
    pub blocked_duration_ms: u64,
}

impl WindowAggregate {
    pub fn new(window_type: WindowType, window_size: Duration, aggregation: AggregationType) -> Self {
        Self {
            window_type,
            window_size,
            aggregation,
            data_buffer: Arc::new(Mutex::new(VecDeque::new())),
            results: Arc::new(Mutex::new(VecDeque::new())),
            watermark: Arc::new(Mutex::new(0)),
        }
    }
    
    pub fn process(&self, value: TimestampedValue) -> Result<Vec<AggregationResult>, StreamingError> {
        let mut buffer = self.data_buffer.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire buffer lock".to_string() })?;
        
        buffer.push_back(value);
        
        // Trigger window computation based on watermark
        let current_watermark = *self.watermark.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire watermark lock".to_string() })?;
        
        let mut triggered_windows = Vec::new();
        
        match self.window_type {
            WindowType::Tumbling => {
                triggered_windows.extend(self.compute_tumbling_windows(&buffer, current_watermark)?);
            },
            WindowType::Sliding => {
                triggered_windows.extend(self.compute_sliding_windows(&buffer, current_watermark)?);
            },
            WindowType::Session => {
                triggered_windows.extend(self.compute_session_windows(&buffer)?);
            },
            WindowType::Global => {
                triggered_windows.extend(self.compute_global_window(&buffer)?);
            },
        }
        
        // Store results
        let mut results = self.results.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire results lock".to_string() })?;
        
        for result in &triggered_windows {
            results.push_back(result.clone());
        }
        
        Ok(triggered_windows)
    }
    
    fn compute_tumbling_windows(&self, buffer: &VecDeque<TimestampedValue>, watermark: u64) -> Result<Vec<AggregationResult>, StreamingError> {
        let window_size_ms = self.window_size.as_millis() as u64;
        let mut results = Vec::new();
        
        // Group values by window
        let mut windows: BTreeMap<u64, Vec<&TimestampedValue>> = BTreeMap::new();
        
        for value in buffer {
            let event_time = value.event_time.unwrap_or(value.timestamp);
            if event_time <= watermark {
                let window_start = (event_time / window_size_ms) * window_size_ms;
                windows.entry(window_start).or_insert_with(Vec::new).push(value);
            }
        }
        
        // Compute aggregations for complete windows
        for (window_start, values) in windows {
            let window_end = window_start + window_size_ms;
            if window_end <= watermark {
                let count = values.len();
                let result = self.compute_aggregation(&values)?;
                results.push(AggregationResult {
                    window_start,
                    window_end,
                    result,
                    count,
                });
            }
        }
        
        Ok(results)
    }
    
    fn compute_sliding_windows(&self, buffer: &VecDeque<TimestampedValue>, watermark: u64) -> Result<Vec<AggregationResult>, StreamingError> {
        // Simplified sliding window implementation
        // In practice, would maintain multiple overlapping windows
        self.compute_tumbling_windows(buffer, watermark)
    }
    
    fn compute_session_windows(&self, buffer: &VecDeque<TimestampedValue>) -> Result<Vec<AggregationResult>, StreamingError> {
        let gap_threshold_ms = self.window_size.as_millis() as u64;
        let mut results = Vec::new();
        
        if buffer.is_empty() {
            return Ok(results);
        }
        
        let mut session_start = buffer[0].timestamp;
        let mut session_values = Vec::new();
        let mut last_timestamp = session_start;
        
        for value in buffer {
            let event_time = value.event_time.unwrap_or(value.timestamp);
            
            if event_time - last_timestamp > gap_threshold_ms {
                // Close current session
                if !session_values.is_empty() {
                    let count = session_values.len();
                    let result = self.compute_aggregation(&session_values)?;
                    results.push(AggregationResult {
                        window_start: session_start,
                        window_end: last_timestamp,
                        result,
                        count,
                    });
                }
                
                // Start new session
                session_start = event_time;
                session_values = vec![value];
            } else {
                session_values.push(value);
            }
            
            last_timestamp = event_time;
        }
        
        Ok(results)
    }
    
    fn compute_global_window(&self, buffer: &VecDeque<TimestampedValue>) -> Result<Vec<AggregationResult>, StreamingError> {
        if buffer.is_empty() {
            return Ok(Vec::new());
        }
        
        let values: Vec<&TimestampedValue> = buffer.iter().collect();
        let result = self.compute_aggregation(&values)?;
        
        Ok(vec![AggregationResult {
            window_start: 0,
            window_end: u64::MAX,
            result,
            count: buffer.len(),
        }])
    }
    
    fn compute_aggregation(&self, values: &[&TimestampedValue]) -> Result<Value, StreamingError> {
        if values.is_empty() {
            return Ok(Value::Null);
        }
        
        match self.aggregation {
            AggregationType::Count => Ok(Value::Real(values.len() as f64)),
            AggregationType::Sum => {
                let mut sum = 0.0;
                for value in values {
                    if let Value::Real(n) = &value.value {
                        sum += n;
                    }
                }
                Ok(Value::Real(sum))
            },
            AggregationType::Average => {
                let mut sum = 0.0;
                let mut count = 0;
                for value in values {
                    if let Value::Real(n) = &value.value {
                        sum += n;
                        count += 1;
                    }
                }
                Ok(if count > 0 { Value::Real(sum / count as f64) } else { Value::Null })
            },
            AggregationType::Min => {
                values.iter()
                    .filter_map(|v| match &v.value {
                        Value::Real(n) => Some(*n),
                        _ => None,
                    })
                    .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(Value::Real)
                    .unwrap_or(Value::Null)
                    .into()
            },
            AggregationType::Max => {
                values.iter()
                    .filter_map(|v| match &v.value {
                        Value::Real(n) => Some(*n),
                        _ => None,
                    })
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(Value::Real)
                    .unwrap_or(Value::Null)
                    .into()
            },
            AggregationType::First => Ok(values.first().unwrap().value.clone()),
            AggregationType::Last => Ok(values.last().unwrap().value.clone()),
            AggregationType::Custom(_) => {
                // Would call custom aggregation function
                Err(StreamingError::ProcessingError("Custom aggregation not implemented".to_string() })
            },
        }
    }
    
    pub fn update_watermark(&self, watermark: u64) -> Result<(), StreamingError> {
        let mut current_watermark = self.watermark.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire watermark lock".to_string() })?;
        *current_watermark = watermark;
        Ok(())
    }
}

impl StreamJoin {
    pub fn new(join_key: String, window_size: Duration) -> Self {
        Self {
            left_stream_buffer: Arc::new(Mutex::new(BTreeMap::new())),
            right_stream_buffer: Arc::new(Mutex::new(BTreeMap::new())),
            join_key,
            window_size,
            results: Arc::new(Mutex::new(VecDeque::new())),
        }
    }
    
    pub fn process_left(&self, value: JoinValue) -> Result<Vec<JoinResult>, StreamingError> {
        self.process_stream_value(value, true)
    }
    
    pub fn process_right(&self, value: JoinValue) -> Result<Vec<JoinResult>, StreamingError> {
        self.process_stream_value(value, false)
    }
    
    fn process_stream_value(&self, value: JoinValue, is_left: bool) -> Result<Vec<JoinResult>, StreamingError> {
        let window_size_ms = self.window_size.as_millis() as u64;
        let mut results = Vec::new();
        
        // Add to appropriate buffer
        if is_left {
            let mut left_buffer = self.left_stream_buffer.lock().map_err(|_| 
                StreamingError::ProcessingError("Failed to acquire left buffer lock".to_string() })?;
            left_buffer.entry(value.timestamp).or_insert_with(Vec::new).push(value.clone());
        } else {
            let mut right_buffer = self.right_stream_buffer.lock().map_err(|_| 
                StreamingError::ProcessingError("Failed to acquire right buffer lock".to_string() })?;
            right_buffer.entry(value.timestamp).or_insert_with(Vec::new).push(value.clone());
        }
        
        // Find matching values in the join window
        let window_start = value.timestamp.saturating_sub(window_size_ms);
        let window_end = value.timestamp + window_size_ms;
        
        let left_buffer = self.left_stream_buffer.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire left buffer lock".to_string() })?;
        let right_buffer = self.right_stream_buffer.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire right buffer lock".to_string() })?;
        
        // Perform join based on which stream the new value came from
        if is_left {
            // New left value, find matching right values
            for (&_timestamp, right_values) in right_buffer.range(window_start..=window_end) {
                for right_value in right_values {
                    if right_value.join_key == value.join_key {
                        results.push(JoinResult {
                            left_value: value.value.clone(),
                            right_value: right_value.value.clone(),
                            join_key: value.join_key.clone(),
                            timestamp: std::cmp::max(value.timestamp, right_value.timestamp),
                        });
                    }
                }
            }
        } else {
            // New right value, find matching left values
            for (&_timestamp, left_values) in left_buffer.range(window_start..=window_end) {
                for left_value in left_values {
                    if left_value.join_key == value.join_key {
                        results.push(JoinResult {
                            left_value: left_value.value.clone(),
                            right_value: value.value.clone(),
                            join_key: value.join_key.clone(),
                            timestamp: std::cmp::max(left_value.timestamp, value.timestamp),
                        });
                    }
                }
            }
        }
        
        // Store results
        let mut result_buffer = self.results.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire results lock".to_string() })?;
        
        for result in &results {
            result_buffer.push_back(result.clone());
        }
        
        Ok(results)
    }
}

impl ComplexEventProcessor {
    pub fn new() -> Self {
        Self {
            patterns: Arc::new(Mutex::new(Vec::new())),
            event_buffer: Arc::new(Mutex::new(VecDeque::new())),
            pattern_state: Arc::new(Mutex::new(HashMap::new())),
            actions: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn add_pattern(&self, pattern: EventPattern) -> Result<(), StreamingError> {
        let mut patterns = self.patterns.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire patterns lock".to_string() })?;
        patterns.push(pattern);
        Ok(())
    }
    
    pub fn process_event(&self, event: ProcessingEvent) -> Result<Vec<Value>, StreamingError> {
        let mut buffer = self.event_buffer.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire buffer lock".to_string() })?;
        buffer.push_back(event.clone());
        
        let patterns = self.patterns.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire patterns lock".to_string() })?;
        
        let mut results = Vec::new();
        
        // Check each pattern against the event buffer
        for pattern in patterns.iter() {
            if let Some(_matched_events) = self.match_pattern(pattern, &buffer)? {
                // Execute action
                if let Some(action_result) = self.get_action(&pattern.action)? {
                    results.push(action_result);
                }
            }
        }
        
        Ok(results)
    }
    
    fn match_pattern(&self, pattern: &EventPattern, buffer: &VecDeque<ProcessingEvent>) -> Result<Option<Vec<ProcessingEvent>>, StreamingError> {
        // Simplified pattern matching - in practice would use more sophisticated CEP algorithms
        let mut matched_events = Vec::new();
        let mut pattern_position = 0;
        
        for event in buffer {
            if pattern_position < pattern.sequence.len() {
                let pattern_element = &pattern.sequence[pattern_position];
                
                if event.event_type == pattern_element.event_type {
                    matched_events.push(event.clone());
                    pattern_position += 1;
                    
                    if pattern_position == pattern.sequence.len() {
                        // Check timing constraints
                        if let Some(within) = pattern.within {
                            let time_span = matched_events.last().unwrap().timestamp - matched_events.first().unwrap().timestamp;
                            if time_span <= within.as_millis() as u64 {
                                return Ok(Some(matched_events));
                            }
                        } else {
                            return Ok(Some(matched_events));
                        }
                    }
                }
            }
        }
        
        Ok(None)
    }
    
    fn get_action(&self, _action_name: &str) -> Result<Option<Value>, StreamingError> {
        // Simplified implementation - in practice would execute actual action
        Ok(Some(Value::String("action_executed".to_string() }))
    }
}

impl BackpressureController {
    pub fn new(strategy: BackpressureStrategy, thresholds: BackpressureThresholds) -> Self {
        Self {
            strategy,
            thresholds,
            current_pressure: Arc::new(Mutex::new(0.0)),
            metrics: Arc::new(Mutex::new(BackpressureMetrics::default())),
        }
    }
    
    pub fn update_pressure(&self, buffer_size: usize, buffer_capacity: usize) -> Result<(), StreamingError> {
        let pressure = buffer_size as f64 / buffer_capacity as f64;
        
        let mut current_pressure = self.current_pressure.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire pressure lock".to_string() })?;
        *current_pressure = pressure;
        
        let mut metrics = self.metrics.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire metrics lock".to_string() })?;
        metrics.buffer_utilization = pressure;
        
        Ok(())
    }
    
    pub fn should_apply_backpressure(&self) -> Result<bool, StreamingError> {
        let pressure = *self.current_pressure.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire pressure lock".to_string() })?;
        
        Ok(pressure >= self.thresholds.warning_level)
    }
    
    pub fn get_metrics(&self) -> Result<BackpressureMetrics, StreamingError> {
        let metrics = self.metrics.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire metrics lock".to_string() })?;
        Ok(metrics.clone())
    }
}

// Foreign object implementations
impl Foreign for WindowAggregate {
    fn type_name(&self) -> &'static str {
        "WindowAggregate"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "process" => {
                if args.len() < 2 {
                    return Err(LyraError::Runtime { message: "process requires 2 arguments".to_string() });
                }
                
                let value = args[0].clone();
                let timestamp = match &args[1] {
                    Value::Real(n) => *n as u64,
                    _ => return Err(LyraError::TypeError("Timestamp must be a number".to_string() }),
                };
                
                let timestamped_value = TimestampedValue {
                    value,
                    timestamp,
                    event_time: None,
                };
                
                let results = self.process(timestamped_value)?;
                // Return list of Associations for aggregation results
                let result_values: Vec<Value> = results
                    .into_iter()
                    .map(|r| {
                        let mut m = HashMap::new();
                        m.insert("windowStart".to_string(), Value::Real(r.window_start as f64));
                        m.insert("windowEnd".to_string(), Value::Real(r.window_end as f64));
                        m.insert("result".to_string(), r.result);
                        m.insert("count".to_string(), Value::Integer(r.count as i64));
                        Value::Object(m)
                    })
                    .collect();
                Ok(Value::List(result_values))
            },
            "updateWatermark" => {
                if args.is_empty() {
                    return Err(LyraError::Runtime { message: "updateWatermark requires 1 argument".to_string() });
                }
                
                let watermark = match &args[0] {
                    Value::Real(n) => *n as u64,
                    _ => return Err(LyraError::TypeError("Watermark must be a number".to_string() }),
                };
                
                self.update_watermark(watermark)?;
                Ok(Value::Boolean(true))
            },
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for StreamJoin {
    fn type_name(&self) -> &'static str {
        "StreamJoin"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "processLeft" => {
                if args.len() < 3 {
                    return Err(LyraError::Runtime { message: "processLeft requires 3 arguments".to_string() });
                }
                
                let value = args[0].clone();
                let join_key = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Err(LyraError::TypeError("Join key must be a string".to_string() }),
                };
                let timestamp = match &args[2] {
                    Value::Real(n) => *n as u64,
                    _ => return Err(LyraError::TypeError("Timestamp must be a number".to_string() }),
                };
                
                let join_value = JoinValue {
                    value,
                    join_key,
                    timestamp,
                };
                
                let results = self.process_left(join_value)?;
                let result_values: Vec<Value> = results
                    .into_iter()
                    .map(|r| {
                        let mut m = HashMap::new();
                        m.insert("left".to_string(), r.left_value);
                        m.insert("right".to_string(), r.right_value);
                        m.insert("joinKey".to_string(), Value::String(r.join_key));
                        m.insert("timestamp".to_string(), Value::Real(r.timestamp as f64));
                        Value::Object(m)
                    })
                    .collect();
                Ok(Value::List(result_values))
            },
            "processRight" => {
                if args.len() < 3 {
                    return Err(LyraError::Runtime { message: "processRight requires 3 arguments".to_string() });
                }
                
                let value = args[0].clone();
                let join_key = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Err(LyraError::TypeError("Join key must be a string".to_string() }),
                };
                let timestamp = match &args[2] {
                    Value::Real(n) => *n as u64,
                    _ => return Err(LyraError::TypeError("Timestamp must be a number".to_string() }),
                };
                
                let join_value = JoinValue {
                    value,
                    join_key,
                    timestamp,
                };
                
                let results = self.process_right(join_value)?;
                let result_values: Vec<Value> = results
                    .into_iter()
                    .map(|r| {
                        let mut m = HashMap::new();
                        m.insert("left".to_string(), r.left_value);
                        m.insert("right".to_string(), r.right_value);
                        m.insert("joinKey".to_string(), Value::String(r.join_key));
                        m.insert("timestamp".to_string(), Value::Real(r.timestamp as f64));
                        Value::Object(m)
                    })
                    .collect();
                Ok(Value::List(result_values))
            },
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for ComplexEventProcessor {
    fn type_name(&self) -> &'static str {
        "ComplexEventProcessor"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "processEvent" => {
                if args.len() < 3 {
                    return Err(LyraError::Runtime { message: "processEvent requires 3 arguments".to_string() });
                }
                
                let event_type = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(LyraError::TypeError("Event type must be a string".to_string() }),
                };
                
                let data = args[1].clone();
                
                let timestamp = match &args[2] {
                    Value::Real(n) => *n as u64,
                    _ => return Err(LyraError::TypeError("Timestamp must be a number".to_string() }),
                };
                
                let event = ProcessingEvent {
                    event_type,
                    data,
                    timestamp,
                    source: None,
                };
                
                let results = self.process_event(event)?;
                Ok(Value::List(results))
            },
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for BackpressureController {
    fn type_name(&self) -> &'static str {
        "BackpressureController"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "updatePressure" => {
                if args.len() < 2 {
                    return Err(LyraError::Runtime { message: "updatePressure requires 2 arguments".to_string() });
                }
                
                let buffer_size = match &args[0] {
                    Value::Real(n) => *n as usize,
                    _ => return Err(LyraError::TypeError("Buffer size must be a number".to_string() }),
                };
                
                let buffer_capacity = match &args[1] {
                    Value::Real(n) => *n as usize,
                    _ => return Err(LyraError::TypeError("Buffer capacity must be a number".to_string() }),
                };
                
                self.update_pressure(buffer_size, buffer_capacity)?;
                Ok(Value::Boolean(true))
            },
            "shouldApplyBackpressure" => {
                let should_apply = self.should_apply_backpressure()?;
                Ok(Value::Boolean(should_apply))
            },
            "getMetrics" => {
                let metrics = self.get_metrics()?;
                let mut m = HashMap::new();
                m.insert("bufferUtilization".to_string(), Value::Real(metrics.buffer_utilization));
                m.insert("throughputRate".to_string(), Value::Real(metrics.throughput_rate));
                m.insert("latencyP95".to_string(), Value::Real(metrics.latency_p95));
                m.insert("droppedMessages".to_string(), Value::Integer(metrics.dropped_messages as i64));
                m.insert("blockedDurationMs".to_string(), Value::Integer(metrics.blocked_duration_ms as i64));
                Ok(Value::Object(m))
            },
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

// Function implementations for Lyra VM
pub fn window_aggregate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(LyraError::Runtime { message: "WindowAggregate requires 3 arguments".to_string() });
    }
    
    let window_type = match &args[0] {
        Value::String(s) => match s.as_str() {
            "tumbling" => WindowType::Tumbling,
            "sliding" => WindowType::Sliding,
            "session" => WindowType::Session,
            "global" => WindowType::Global,
            _ => return Err(LyraError::ValueError("Unknown window type".to_string() }),
        },
        _ => return Err(LyraError::TypeError("Window type must be a string".to_string() }),
    };
    
    let window_size = match &args[1] {
        Value::Real(n) => Duration::from_millis(*n as u64),
        _ => return Err(LyraError::TypeError("Window size must be a number".to_string() }),
    };
    
    let aggregation = match &args[2] {
        Value::String(s) => match s.as_str() {
            "count" => AggregationType::Count,
            "sum" => AggregationType::Sum,
            "average" => AggregationType::Average,
            "min" => AggregationType::Min,
            "max" => AggregationType::Max,
            "first" => AggregationType::First,
            "last" => AggregationType::Last,
            _ => AggregationType::Custom(s.clone()),
        },
        _ => return Err(LyraError::TypeError("Aggregation type must be a string".to_string() }),
    };
    
    let window_agg = WindowAggregate::new(window_type, window_size, aggregation);
    Ok(Value::LyObj(LyObj::new(Box::new(window_agg))))
}

pub fn stream_join(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Runtime { message: "StreamJoin requires 2 arguments".to_string() });
    }
    
    let join_key = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("Join key must be a string".to_string() }),
    };
    
    let window_size = match &args[1] {
        Value::Real(n) => Duration::from_millis(*n as u64),
        _ => return Err(LyraError::TypeError("Window size must be a number".to_string() }),
    };
    
    let join = StreamJoin::new(join_key, window_size);
    Ok(Value::LyObj(LyObj::new(Box::new(join))))
}

pub fn complex_event_processing(_args: &[Value]) -> VmResult<Value> {
    let cep = ComplexEventProcessor::new();
    Ok(Value::LyObj(LyObj::new(Box::new(cep))))
}

pub fn backpressure_control(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(LyraError::Runtime { message: "BackpressureControl requires at least 1 argument".to_string() });
    }
    
    let strategy = match &args[0] {
        Value::String(s) => match s.as_str() {
            "block" => BackpressureStrategy::Block,
            "drop" => BackpressureStrategy::Drop,
            "sample" => BackpressureStrategy::Sample,
            "adaptive" => BackpressureStrategy::Adaptive,
            _ => return Err(LyraError::ValueError("Unknown backpressure strategy".to_string() }),
        },
        _ => return Err(LyraError::TypeError("Strategy must be a string".to_string() }),
    };
    
    let thresholds = BackpressureThresholds {
        warning_level: 0.7,
        critical_level: 0.9,
        recovery_level: 0.5,
    };
    
    let controller = BackpressureController::new(strategy, thresholds);
    Ok(Value::LyObj(LyObj::new(Box::new(controller))))
}

/// Register all stream processing functions
pub fn register_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    functions.insert("WindowAggregate".to_string(), window_aggregate as fn(&[Value]) -> VmResult<Value>);
    functions.insert("StreamJoin".to_string(), stream_join as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ComplexEventProcessing".to_string(), complex_event_processing as fn(&[Value]) -> VmResult<Value>);
    functions.insert("BackpressureControl".to_string(), backpressure_control as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_window_aggregate_creation() {
        let window_agg = WindowAggregate::new(
            WindowType::Tumbling,
            Duration::from_secs(60),
            AggregationType::Sum,
        );
        
        assert!(matches!(window_agg.window_type, WindowType::Tumbling));
        assert_eq!(window_agg.window_size, Duration::from_secs(60));
    }
    
    #[test]
    fn test_window_aggregate_sum() {
        let window_agg = WindowAggregate::new(
            WindowType::Global,
            Duration::from_secs(60),
            AggregationType::Sum,
        );
        
        let value1 = TimestampedValue {
            value: Value::Real(10.0),
            timestamp: 1000,
            event_time: None,
        };
        
        let value2 = TimestampedValue {
            value: Value::Real(20.0),
            timestamp: 2000,
            event_time: None,
        };
        
        window_agg.process(value1).unwrap();
        let results = window_agg.process(value2).unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result, Value::Real(30.0));
    }
    
    #[test]
    fn test_stream_join_creation() {
        let join = StreamJoin::new("user_id".to_string(), Duration::from_secs(300));
        assert_eq!(join.join_key, "user_id");
        assert_eq!(join.window_size, Duration::from_secs(300));
    }
    
    #[test]
    fn test_stream_join_process() {
        let join = StreamJoin::new("key1".to_string(), Duration::from_secs(300));
        
        let left_value = JoinValue {
            value: Value::String("left_data".to_string() },
            join_key: "key1".to_string(),
            timestamp: 1000,
        };
        
        let right_value = JoinValue {
            value: Value::String("right_data".to_string() },
            join_key: "key1".to_string(),
            timestamp: 1100,
        };
        
        join.process_left(left_value).unwrap();
        let results = join.process_right(right_value).unwrap();
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].join_key, "key1");
    }
    
    #[test]
    fn test_complex_event_processor_creation() {
        let cep = ComplexEventProcessor::new();
        // Just test that it creates successfully
        assert_eq!(cep.name(), "ComplexEventProcessor");
    }
    
    #[test]
    fn test_backpressure_controller_creation() {
        let thresholds = BackpressureThresholds {
            warning_level: 0.7,
            critical_level: 0.9,
            recovery_level: 0.5,
        };
        
        let controller = BackpressureController::new(BackpressureStrategy::Block, thresholds);
        controller.update_pressure(700, 1000).unwrap();
        
        let should_apply = controller.should_apply_backpressure().unwrap();
        assert!(should_apply); // 700/1000 = 0.7 >= 0.7 warning level
    }
    
    #[test]
    fn test_window_aggregate_function() {
        let args = vec![
            Value::String("tumbling".to_string() },
            Value::Real(60000.0), // 60 seconds
            Value::String("sum".to_string() },
        ];
        
        let result = window_aggregate(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_stream_join_function() {
        let args = vec![
            Value::String("user_id".to_string() },
            Value::Real(300000.0), // 5 minutes
        ];
        
        let result = stream_join(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_register_functions() {
        let functions = register_functions();
        assert!(functions.contains_key("WindowAggregate"));
        assert!(functions.contains_key("StreamJoin"));
        assert!(functions.contains_key("ComplexEventProcessing"));
        assert!(functions.contains_key("BackpressureControl"));
    }
}
