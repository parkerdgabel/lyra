//! WebSocket Operations and Real-time Communication
//!
//! This module implements WebSocket functionality as symbolic objects that enable
//! real-time bidirectional communication within Lyra's network-transparent architecture.

use super::core::{NetworkEndpoint, NetworkAuth};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// WebSocket connection state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WebSocketState {
    /// Connection is being established
    Connecting,
    /// Connection is open and ready for communication
    Open,
    /// Connection is being closed
    Closing,
    /// Connection is closed
    Closed,
    /// Connection failed
    Failed,
}

impl std::fmt::Display for WebSocketState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WebSocketState::Connecting => write!(f, "Connecting"),
            WebSocketState::Open => write!(f, "Open"),
            WebSocketState::Closing => write!(f, "Closing"),
            WebSocketState::Closed => write!(f, "Closed"),
            WebSocketState::Failed => write!(f, "Failed"),
        }
    }
}

/// WebSocket message types
#[derive(Debug, Clone)]
pub enum WebSocketMessage {
    /// Text message
    Text(String),
    /// Binary message
    Binary(Vec<u8>),
    /// Ping frame
    Ping(Vec<u8>),
    /// Pong frame (response to ping)
    Pong(Vec<u8>),
    /// Close frame
    Close(Option<String>),
}

impl WebSocketMessage {
    /// Get message type as string
    pub fn message_type(&self) -> &'static str {
        match self {
            WebSocketMessage::Text(_) => "Text",
            WebSocketMessage::Binary(_) => "Binary",
            WebSocketMessage::Ping(_) => "Ping",
            WebSocketMessage::Pong(_) => "Pong",
            WebSocketMessage::Close(_) => "Close",
        }
    }
    
    /// Get message data size
    pub fn data_size(&self) -> usize {
        match self {
            WebSocketMessage::Text(s) => s.len(),
            WebSocketMessage::Binary(b) => b.len(),
            WebSocketMessage::Ping(b) => b.len(),
            WebSocketMessage::Pong(b) => b.len(),
            WebSocketMessage::Close(s) => s.as_ref().map(|s| s.len()).unwrap_or(0),
        }
    }
    
    /// Extract text content if available
    pub fn text_content(&self) -> Result<String, String> {
        match self {
            WebSocketMessage::Text(s) => Ok(s.clone()),
            WebSocketMessage::Close(Some(s)) => Ok(s.clone()),
            _ => Err("Message is not text type".to_string()),
        }
    }
    
    /// Extract binary content if available
    pub fn binary_content(&self) -> Result<Vec<u8>, String> {
        match self {
            WebSocketMessage::Binary(b) => Ok(b.clone()),
            WebSocketMessage::Ping(b) => Ok(b.clone()),
            WebSocketMessage::Pong(b) => Ok(b.clone()),
            WebSocketMessage::Text(s) => Ok(s.as_bytes().to_vec()),
            _ => Err("Message has no binary content".to_string()),
        }
    }
}

/// WebSocket connection configuration and state
#[derive(Debug, Clone)]
pub struct WebSocket {
    /// WebSocket URL (ws:// or wss://)
    pub url: String,
    /// Current connection state
    pub state: WebSocketState,
    /// Authentication configuration
    pub auth: NetworkAuth,
    /// Connection timeout
    pub timeout: Duration,
    /// Maximum message size
    pub max_message_size: usize,
    /// Custom headers for handshake
    pub headers: HashMap<String, String>,
    /// Connection metadata
    pub metadata: HashMap<String, String>,
    /// Message buffer for incoming messages
    pub message_buffer: Vec<WebSocketMessage>,
    /// Statistics
    pub messages_sent: u64,
    pub messages_received: u64,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    /// Timestamps
    pub created_at: SystemTime,
    pub connected_at: Option<SystemTime>,
    pub last_activity: SystemTime,
}

impl WebSocket {
    /// Create a new WebSocket connection
    pub fn new(url: String) -> Self {
        Self {
            url,
            state: WebSocketState::Closed,
            auth: NetworkAuth::None,
            timeout: Duration::from_secs(30),
            max_message_size: 1024 * 1024, // 1MB default
            headers: HashMap::new(),
            metadata: HashMap::new(),
            message_buffer: Vec::new(),
            messages_sent: 0,
            messages_received: 0,
            bytes_sent: 0,
            bytes_received: 0,
            created_at: SystemTime::now(),
            connected_at: None,
            last_activity: SystemTime::now(),
        }
    }
    
    /// Set authentication
    pub fn with_auth(mut self, auth: NetworkAuth) -> Self {
        self.auth = auth;
        self
    }
    
    /// Set timeout
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Add custom header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
    
    /// Set maximum message size
    pub fn with_max_message_size(mut self, size: usize) -> Self {
        self.max_message_size = size;
        self
    }
    
    /// Connect to WebSocket (placeholder implementation)
    pub fn connect(&mut self) -> Result<(), String> {
        if self.state != WebSocketState::Closed {
            return Err(format!("Cannot connect from state: {}", self.state));
        }
        
        self.state = WebSocketState::Connecting;
        
        // Placeholder connection logic
        // In a real implementation, this would establish the WebSocket connection
        
        self.state = WebSocketState::Open;
        self.connected_at = Some(SystemTime::now());
        self.last_activity = SystemTime::now();
        
        Ok(())
    }
    
    /// Send a message (placeholder implementation)
    pub fn send_message(&mut self, message: WebSocketMessage) -> Result<(), String> {
        if self.state != WebSocketState::Open {
            return Err(format!("Cannot send message in state: {}", self.state));
        }
        
        let message_size = message.data_size();
        if message_size > self.max_message_size {
            return Err(format!("Message size {} exceeds maximum {}", message_size, self.max_message_size));
        }
        
        // Placeholder send logic
        // In real implementation, would send message over WebSocket
        
        self.messages_sent += 1;
        self.bytes_sent += message_size as u64;
        self.last_activity = SystemTime::now();
        
        Ok(())
    }
    
    /// Receive a message (placeholder implementation)
    pub fn receive_message(&mut self) -> Result<Option<WebSocketMessage>, String> {
        if self.state != WebSocketState::Open {
            return Err(format!("Cannot receive message in state: {}", self.state));
        }
        
        // Placeholder receive logic
        // In real implementation, would read from WebSocket
        
        // Simulate receiving a message occasionally
        if self.message_buffer.is_empty() && self.messages_received < 3 {
            let message = WebSocketMessage::Text(format!(
                "Simulated message {} from {}", 
                self.messages_received + 1,
                self.url
            ));
            
            self.messages_received += 1;
            self.bytes_received += message.data_size() as u64;
            self.last_activity = SystemTime::now();
            
            Ok(Some(message))
        } else {
            Ok(None)
        }
    }
    
    /// Send ping frame
    pub fn ping(&mut self, data: Option<Vec<u8>>) -> Result<(), String> {
        let ping_data = data.unwrap_or_else(|| b"ping".to_vec());
        let ping_message = WebSocketMessage::Ping(ping_data);
        self.send_message(ping_message)
    }
    
    /// Close the connection
    pub fn close(&mut self, reason: Option<String>) -> Result<(), String> {
        if matches!(self.state, WebSocketState::Closed | WebSocketState::Failed) {
            return Ok(());
        }
        
        self.state = WebSocketState::Closing;
        
        // Send close frame if connection is open
        if self.state == WebSocketState::Open {
            let close_message = WebSocketMessage::Close(reason);
            let _ = self.send_message(close_message);
        }
        
        self.state = WebSocketState::Closed;
        Ok(())
    }
    
    /// Check if connection is open
    pub fn is_open(&self) -> bool {
        self.state == WebSocketState::Open
    }
    
    /// Check if connection is closed
    pub fn is_closed(&self) -> bool {
        matches!(self.state, WebSocketState::Closed | WebSocketState::Failed)
    }
    
    /// Get connection uptime
    pub fn uptime(&self) -> Duration {
        self.connected_at
            .map(|connected| connected.elapsed().unwrap_or_default())
            .unwrap_or_default()
    }
    
    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed().unwrap_or_default()
    }
}

impl Foreign for WebSocket {
    fn type_name(&self) -> &'static str {
        "WebSocket"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "URL" => Ok(Value::String(self.url.clone())),
            "State" => Ok(Value::String(self.state.to_string())),
            "IsOpen" => Ok(Value::Integer(if self.is_open() { 1 } else { 0 })),
            "IsClosed" => Ok(Value::Integer(if self.is_closed() { 1 } else { 0 })),
            "Timeout" => Ok(Value::Real(self.timeout.as_secs_f64())),
            "MaxMessageSize" => Ok(Value::Integer(self.max_message_size as i64)),
            "MessagesSent" => Ok(Value::Integer(self.messages_sent as i64)),
            "MessagesReceived" => Ok(Value::Integer(self.messages_received as i64)),
            "BytesSent" => Ok(Value::Integer(self.bytes_sent as i64)),
            "BytesReceived" => Ok(Value::Integer(self.bytes_received as i64)),
            "Uptime" => Ok(Value::Real(self.uptime().as_secs_f64())),
            "IdleTime" => Ok(Value::Real(self.idle_time().as_secs_f64())),
            "Headers" => {
                let entries: Vec<Value> = self.headers.iter()
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
            "ConnectedAt" => {
                if let Some(connected) = self.connected_at {
                    let timestamp = connected.duration_since(UNIX_EPOCH)
                        .unwrap_or_default().as_secs_f64();
                    Ok(Value::Real(timestamp))
                } else {
                    Ok(Value::Integer(0))
                }
            }
            "LastActivity" => {
                let timestamp = self.last_activity.duration_since(UNIX_EPOCH)
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

/// WebSocket message as a symbolic object
#[derive(Debug, Clone)]
pub struct WebSocketMessageObj {
    pub message: WebSocketMessage,
    pub timestamp: SystemTime,
    pub source_url: String,
}

impl WebSocketMessageObj {
    pub fn new(message: WebSocketMessage, source_url: String) -> Self {
        Self {
            message,
            timestamp: SystemTime::now(),
            source_url,
        }
    }
}

impl Foreign for WebSocketMessageObj {
    fn type_name(&self) -> &'static str {
        "WebSocketMessage"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Type" => Ok(Value::String(self.message.message_type().to_string())),
            "Size" => Ok(Value::Integer(self.message.data_size() as i64)),
            "SourceURL" => Ok(Value::String(self.source_url.clone())),
            "TextContent" => {
                match self.message.text_content() {
                    Ok(text) => Ok(Value::String(text)),
                    Err(e) => Err(ForeignError::RuntimeError { message: e }),
                }
            }
            "BinaryContent" => {
                match self.message.binary_content() {
                    Ok(data) => {
                        let values: Vec<Value> = data.into_iter()
                            .map(|b| Value::Integer(b as i64))
                            .collect();
                        Ok(Value::List(values))
                    }
                    Err(e) => Err(ForeignError::RuntimeError { message: e }),
                }
            }
            "Timestamp" => {
                let timestamp = self.timestamp.duration_since(UNIX_EPOCH)
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

/// Create a WebSocket connection
/// Syntax: WebSocket[url], WebSocket[url, options]
pub fn websocket(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (url, [options])".to_string(),
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
    
    let mut ws = WebSocket::new(url);
    
    // Apply options if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(options) => {
                for option in options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "timeout" => {
                                        if let Value::Real(seconds) = value {
                                            ws = ws.with_timeout(Duration::from_secs_f64(*seconds));
                                        }
                                    }
                                    "maxMessageSize" => {
                                        if let Value::Integer(size) = value {
                                            ws = ws.with_max_message_size(*size as usize);
                                        }
                                    }
                                    _ => {
                                        // Add to metadata for unknown options
                                        if let Value::String(val) = value {
                                            ws.metadata.insert(key.clone(), val.clone());
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
    
    Ok(Value::LyObj(LyObj::new(Box::new(ws))))
}

/// Connect to a WebSocket
/// Syntax: WebSocketConnect[websocket]
pub fn websocket_connect(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (websocket)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(ws) = obj.downcast_ref::<WebSocket>() {
                let mut ws_clone = ws.clone();
                match ws_clone.connect() {
                    Ok(()) => Ok(Value::LyObj(LyObj::new(Box::new(ws_clone)))),
                    Err(e) => Err(VmError::Runtime(format!("WebSocket connection failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "WebSocket object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "WebSocket object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Send a message to WebSocket
/// Syntax: WebSocketSend[websocket, message]
pub fn websocket_send(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (websocket, message)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let message = match &args[1] {
        Value::String(s) => WebSocketMessage::Text(s.clone()),
        Value::List(bytes) => {
            let mut binary_data = Vec::new();
            for byte_val in bytes {
                match byte_val {
                    Value::Integer(i) => binary_data.push(*i as u8),
                    _ => return Err(VmError::TypeError {
                        expected: "List of integers for binary data".to_string(),
                        actual: format!("{:?}", byte_val),
                    }),
                }
            }
            WebSocketMessage::Binary(binary_data)
        }
        _ => return Err(VmError::TypeError {
            expected: "String or List of bytes for message".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(ws) = obj.downcast_ref::<WebSocket>() {
                let mut ws_clone = ws.clone();
                match ws_clone.send_message(message) {
                    Ok(()) => Ok(Value::LyObj(LyObj::new(Box::new(ws_clone)))),
                    Err(e) => Err(VmError::Runtime(format!("WebSocket send failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "WebSocket object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "WebSocket object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Receive a message from WebSocket
/// Syntax: WebSocketReceive[websocket]
pub fn websocket_receive(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (websocket)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(ws) = obj.downcast_ref::<WebSocket>() {
                let mut ws_clone = ws.clone();
                match ws_clone.receive_message() {
                    Ok(Some(message)) => {
                        let msg_obj = WebSocketMessageObj::new(message, ws_clone.url.clone());
                        Ok(Value::List(vec![
                            Value::LyObj(LyObj::new(Box::new(ws_clone))),
                            Value::LyObj(LyObj::new(Box::new(msg_obj))),
                        ]))
                    }
                    Ok(None) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(ws_clone))),
                        Value::String("Missing".to_string()),
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("WebSocket receive failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "WebSocket object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "WebSocket object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Close a WebSocket connection
/// Syntax: WebSocketClose[websocket], WebSocketClose[websocket, reason]
pub fn websocket_close(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (websocket, [reason])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let reason = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => Some(s.clone()),
            _ => return Err(VmError::TypeError {
                expected: "String for close reason".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        None
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(ws) = obj.downcast_ref::<WebSocket>() {
                let mut ws_clone = ws.clone();
                match ws_clone.close(reason) {
                    Ok(()) => Ok(Value::LyObj(LyObj::new(Box::new(ws_clone)))),
                    Err(e) => Err(VmError::Runtime(format!("WebSocket close failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "WebSocket object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "WebSocket object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Send ping to WebSocket
/// Syntax: WebSocketPing[websocket], WebSocketPing[websocket, data]
pub fn websocket_ping(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (websocket, [data])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let ping_data = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => Some(s.as_bytes().to_vec()),
            Value::List(bytes) => {
                let mut binary_data = Vec::new();
                for byte_val in bytes {
                    match byte_val {
                        Value::Integer(i) => binary_data.push(*i as u8),
                        _ => return Err(VmError::TypeError {
                            expected: "List of integers for ping data".to_string(),
                            actual: format!("{:?}", byte_val),
                        }),
                    }
                }
                Some(binary_data)
            }
            _ => return Err(VmError::TypeError {
                expected: "String or List of bytes for ping data".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        None
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(ws) = obj.downcast_ref::<WebSocket>() {
                let mut ws_clone = ws.clone();
                match ws_clone.ping(ping_data) {
                    Ok(()) => Ok(Value::LyObj(LyObj::new(Box::new(ws_clone)))),
                    Err(e) => Err(VmError::Runtime(format!("WebSocket ping failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "WebSocket object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "WebSocket object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_websocket_creation() {
        let ws = WebSocket::new("wss://echo.websocket.org/".to_string());
        assert_eq!(ws.url, "wss://echo.websocket.org/");
        assert_eq!(ws.state, WebSocketState::Closed);
        assert!(!ws.is_open());
        assert!(ws.is_closed());
    }
    
    #[test]
    fn test_websocket_configuration() {
        let ws = WebSocket::new("ws://localhost:8080/".to_string())
            .with_timeout(Duration::from_secs(60))
            .with_max_message_size(2048)
            .with_header("Authorization".to_string(), "Bearer token".to_string());
        
        assert_eq!(ws.timeout, Duration::from_secs(60));
        assert_eq!(ws.max_message_size, 2048);
        assert!(ws.headers.contains_key("Authorization"));
    }
    
    #[test]
    fn test_websocket_connection() {
        let mut ws = WebSocket::new("ws://test.example/".to_string());
        assert_eq!(ws.state, WebSocketState::Closed);
        
        ws.connect().unwrap();
        assert_eq!(ws.state, WebSocketState::Open);
        assert!(ws.is_open());
        assert!(!ws.is_closed());
        assert!(ws.connected_at.is_some());
    }
    
    #[test]
    fn test_websocket_messaging() {
        let mut ws = WebSocket::new("ws://test.example/".to_string());
        ws.connect().unwrap();
        
        let message = WebSocketMessage::Text("Hello WebSocket!".to_string());
        ws.send_message(message).unwrap();
        
        assert_eq!(ws.messages_sent, 1);
        assert!(ws.bytes_sent > 0);
        
        let received = ws.receive_message().unwrap();
        assert!(received.is_some());
        assert_eq!(ws.messages_received, 1);
    }
    
    #[test]
    fn test_websocket_message_types() {
        let text_msg = WebSocketMessage::Text("Hello".to_string());
        assert_eq!(text_msg.message_type(), "Text");
        assert_eq!(text_msg.data_size(), 5);
        assert_eq!(text_msg.text_content().unwrap(), "Hello");
        
        let binary_msg = WebSocketMessage::Binary(vec![1, 2, 3, 4]);
        assert_eq!(binary_msg.message_type(), "Binary");
        assert_eq!(binary_msg.data_size(), 4);
        assert_eq!(binary_msg.binary_content().unwrap(), vec![1, 2, 3, 4]);
    }
    
    #[test]
    fn test_websocket_ping_pong() {
        let mut ws = WebSocket::new("ws://test.example/".to_string());
        ws.connect().unwrap();
        
        ws.ping(Some(b"test ping".to_vec())).unwrap();
        assert_eq!(ws.messages_sent, 1);
    }
    
    #[test]
    fn test_websocket_close() {
        let mut ws = WebSocket::new("ws://test.example/".to_string());
        ws.connect().unwrap();
        assert!(ws.is_open());
        
        ws.close(Some("Test close".to_string())).unwrap();
        assert!(ws.is_closed());
    }
    
    #[test]
    fn test_websocket_interface_function() {
        let args = vec![Value::String("wss://echo.websocket.org/".to_string())];
        let result = websocket(&args).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                assert_eq!(obj.inner().type_name(), "WebSocket");
            }
            _ => panic!("Expected WebSocket object"),
        }
    }
}