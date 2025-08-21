//! # Real-time API Module
//!
//! Provides WebSocket, Server-Sent Events (SSE), gRPC, and other real-time API patterns
//! for building modern real-time applications.

use crate::vm::{Value, VmResult};
use crate::error::LyraError;
use crate::foreign::{Foreign, LyObj};
use super::{StreamingError, StreamingConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::net::TcpListener;
use futures_util::SinkExt;
use serde::{Serialize, Deserialize};

/// WebSocket server for real-time bidirectional communication
pub struct WebSocketServer {
    port: u16,
    path: String,
    handlers: Arc<Mutex<HashMap<String, WebSocketHandler>>>,
    connections: Arc<Mutex<HashMap<String, WebSocketConnection>>>,
    rooms: Arc<Mutex<HashMap<String, Vec<String>>>>, // room_id -> [connection_ids]
}

/// WebSocket client for connecting to WebSocket servers
pub struct WebSocketClient {
    url: String,
    protocols: Vec<String>,
    connection: Arc<Mutex<Option<WebSocketConnection>>>,
    config: StreamingConfig,
}

/// Server-Sent Events endpoint for one-way streaming
pub struct ServerSentEvents {
    endpoint: String,
    connections: Arc<Mutex<HashMap<String, SSEConnection>>>,
    format: SSEFormat,
}

/// gRPC server implementation
pub struct GRPCServer {
    port: u16,
    services: Arc<Mutex<HashMap<String, GRPCService>>>,
    config: StreamingConfig,
}

/// gRPC client implementation
pub struct GRPCClient {
    server_address: String,
    service: String,
    config: StreamingConfig,
}

/// Connection pool for managing multiple connections
pub struct ConnectionPool {
    pool_type: ConnectionType,
    max_size: usize,
    connections: Arc<Mutex<Vec<Box<dyn Connection + Send + Sync>>>>,
    health_check_interval: Duration,
}

/// WebSocket connection representation
#[derive(Debug, Clone)]
pub struct WebSocketConnection {
    pub id: String,
    pub user_id: Option<String>,
    pub rooms: Vec<String>,
    pub connected_at: u64,
    pub last_ping: u64,
    // In real implementation, would contain actual WebSocket stream
}

/// Server-Sent Events connection
#[derive(Debug, Clone)]
pub struct SSEConnection {
    pub id: String,
    pub user_id: Option<String>,
    pub connected_at: u64,
    pub last_event_id: Option<String>,
    // In real implementation, would contain HTTP response stream
}

/// WebSocket message handler
pub type WebSocketHandler = Box<dyn Fn(WebSocketMessage) -> Result<Option<WebSocketMessage>, StreamingError> + Send + Sync>;

/// WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    pub message_type: String,
    pub data: Value,
    pub from: Option<String>,
    pub to: Option<String>,
    pub room: Option<String>,
    pub timestamp: u64,
}

/// Server-Sent Events message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSEMessage {
    pub event_type: Option<String>,
    pub data: String,
    pub id: Option<String>,
    pub retry: Option<u32>,
}

/// gRPC service definition
#[derive(Debug, Clone)]
pub struct GRPCService {
    pub name: String,
    pub methods: HashMap<String, GRPCMethod>,
}

/// gRPC method definition
#[derive(Debug, Clone)]
pub struct GRPCMethod {
    pub name: String,
    pub input_type: String,
    pub output_type: String,
    pub streaming: GRPCStreamType,
}

/// gRPC streaming types
#[derive(Debug, Clone)]
pub enum GRPCStreamType {
    Unary,
    ClientStreaming,
    ServerStreaming,
    Bidirectional,
}

/// SSE format options
#[derive(Debug, Clone)]
pub enum SSEFormat {
    PlainText,
    Json,
    Custom(String), // Custom format string
}

/// Connection types for connection pool
#[derive(Debug, Clone)]
pub enum ConnectionType {
    WebSocket,
    HTTP,
    GRPC,
    Database,
}

/// Generic connection trait
pub trait Connection {
    fn id(&self) -> &str;
    fn is_healthy(&self) -> bool;
    fn ping(&mut self) -> Result<(), StreamingError>;
    fn close(&mut self) -> Result<(), StreamingError>;
}

impl WebSocketServer {
    pub fn new(port: u16, path: String) -> Self {
        Self {
            port,
            path,
            handlers: Arc::new(Mutex::new(HashMap::new())),
            connections: Arc::new(Mutex::new(HashMap::new())),
            rooms: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn add_handler(&self, event_type: String, handler: WebSocketHandler) -> Result<(), StreamingError> {
        let mut handlers = self.handlers.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire handlers lock".to_string()))?;
        handlers.insert(event_type, handler);
        Ok(())
    }
    
    pub fn broadcast(&self, message: WebSocketMessage, room: Option<String>) -> Result<usize, StreamingError> {
        let connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        
        let rooms = self.rooms.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire rooms lock".to_string()))?;
        
        let target_connections: Vec<String> = if let Some(room_id) = room {
            rooms.get(&room_id).cloned().unwrap_or_default()
        } else {
            connections.keys().cloned().collect()
        };
        
        let mut sent_count = 0;
        for connection_id in target_connections {
            if connections.contains_key(&connection_id) {
                // In real implementation, would send message through WebSocket
                sent_count += 1;
            }
        }
        
        Ok(sent_count)
    }
    
    pub fn join_room(&self, connection_id: String, room_id: String) -> Result<(), StreamingError> {
        let mut rooms = self.rooms.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire rooms lock".to_string()))?;
        
        rooms.entry(room_id).or_insert_with(Vec::new).push(connection_id);
        Ok(())
    }
    
    pub fn leave_room(&self, connection_id: &str, room_id: &str) -> Result<(), StreamingError> {
        let mut rooms = self.rooms.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire rooms lock".to_string()))?;
        
        if let Some(room_connections) = rooms.get_mut(room_id) {
            room_connections.retain(|id| id != connection_id);
            if room_connections.is_empty() {
                rooms.remove(room_id);
            }
        }
        
        Ok(())
    }
    
    pub fn get_connection_count(&self) -> Result<usize, StreamingError> {
        let connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        Ok(connections.len())
    }
}

impl WebSocketClient {
    pub fn new(url: String, protocols: Vec<String>, config: Option<StreamingConfig>) -> Self {
        Self {
            url,
            protocols,
            connection: Arc::new(Mutex::new(None)),
            config: config.unwrap_or_default(),
        }
    }
    
    pub fn connect(&self) -> Result<(), StreamingError> {
        // In real implementation, would establish WebSocket connection
        let connection = WebSocketConnection {
            id: uuid::Uuid::new_v4().to_string(),
            user_id: None,
            rooms: Vec::new(),
            connected_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            last_ping: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        };
        
        let mut conn = self.connection.lock().map_err(|_| 
            StreamingError::ConnectionFailed("Failed to acquire connection lock".to_string()))?;
        *conn = Some(connection);
        
        Ok(())
    }
    
    pub fn send(&self, message: WebSocketMessage) -> Result<(), StreamingError> {
        let connection = self.connection.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connection lock".to_string()))?;
        
        if connection.is_none() {
            return Err(StreamingError::ConnectionFailed("Not connected".to_string()));
        }
        
        // In real implementation, would send message through WebSocket
        Ok(())
    }
    
    pub fn disconnect(&self) -> Result<(), StreamingError> {
        let mut connection = self.connection.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connection lock".to_string()))?;
        *connection = None;
        Ok(())
    }
}

impl ServerSentEvents {
    pub fn new(endpoint: String, format: SSEFormat) -> Self {
        Self {
            endpoint,
            connections: Arc::new(Mutex::new(HashMap::new())),
            format,
        }
    }
    
    pub fn send_event(&self, event: SSEMessage, target_user: Option<String>) -> Result<usize, StreamingError> {
        let connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        
        let mut sent_count = 0;
        for (_, connection) in connections.iter() {
            if let Some(ref target) = target_user {
                if connection.user_id.as_ref() != Some(target) {
                    continue;
                }
            }
            
            // In real implementation, would send SSE event
            sent_count += 1;
        }
        
        Ok(sent_count)
    }
    
    pub fn add_connection(&self, connection: SSEConnection) -> Result<(), StreamingError> {
        let mut connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        connections.insert(connection.id.clone(), connection);
        Ok(())
    }
    
    pub fn remove_connection(&self, connection_id: &str) -> Result<(), StreamingError> {
        let mut connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        connections.remove(connection_id);
        Ok(())
    }
}

impl GRPCServer {
    pub fn new(port: u16, config: Option<StreamingConfig>) -> Self {
        Self {
            port,
            services: Arc::new(Mutex::new(HashMap::new())),
            config: config.unwrap_or_default(),
        }
    }
    
    pub fn add_service(&self, service: GRPCService) -> Result<(), StreamingError> {
        let mut services = self.services.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire services lock".to_string()))?;
        services.insert(service.name.clone(), service);
        Ok(())
    }
    
    pub fn start(&self) -> Result<(), StreamingError> {
        // In real implementation, would start gRPC server
        Ok(())
    }
}

impl GRPCClient {
    pub fn new(server_address: String, service: String, config: Option<StreamingConfig>) -> Self {
        Self {
            server_address,
            service,
            config: config.unwrap_or_default(),
        }
    }
    
    pub fn call(&self, method: &str, request: Value) -> Result<Value, StreamingError> {
        // In real implementation, would make gRPC call
        Ok(Value::String("gRPC response".to_string()))
    }
}

impl ConnectionPool {
    pub fn new(pool_type: ConnectionType, max_size: usize, health_check_interval: Duration) -> Self {
        Self {
            pool_type,
            max_size,
            connections: Arc::new(Mutex::new(Vec::new())),
            health_check_interval,
        }
    }
    
    pub fn get_connection(&self) -> Result<Option<String>, StreamingError> {
        let connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        
        // Find a healthy connection
        for conn in connections.iter() {
            if conn.is_healthy() {
                return Ok(Some(conn.id().to_string()));
            }
        }
        
        Ok(None)
    }
    
    pub fn health_check(&self) -> Result<usize, StreamingError> {
        let mut connections = self.connections.lock().map_err(|_| 
            StreamingError::ProcessingError("Failed to acquire connections lock".to_string()))?;
        
        let mut healthy_count = 0;
        connections.retain(|conn| {
            if conn.is_healthy() {
                healthy_count += 1;
                true
            } else {
                false
            }
        });
        
        Ok(healthy_count)
    }
}

// Foreign object implementations
impl Foreign for WebSocketServer {
    fn type_name(&self) -> &'static str {
        "WebSocketServer"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "broadcast" => {
                if args.is_empty() {
                    return Err(LyraError::Runtime { message: "broadcast requires at least 1 argument".to_string()));
                }
                
                let message = WebSocketMessage {
                    message_type: "broadcast".to_string(),
                    data: args[0].clone(),
                    from: None,
                    to: None,
                    room: if args.len() > 1 {
                        match &args[1] {
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    },
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                let sent_count = self.broadcast(message, message.room.clone())?;
                Ok(Value::Integer(sent_count as i64))
            },
            "connectionCount" => {
                let count = self.get_connection_count()?;
                Ok(Value::Integer(count as i64))
            },
            "port" => Ok(Value::Integer(self.port as i64)),
            "path" => Ok(Value::String(self.path.clone())),
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for WebSocketClient {
    fn type_name(&self) -> &'static str {
        "WebSocketClient"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "connect" => {
                self.connect()?;
                Ok(Value::Boolean(true))
            },
            "send" => {
                if args.is_empty() {
                    return Err(LyraError::Runtime { message: "send requires 1 argument".to_string()));
                }
                
                let message = WebSocketMessage {
                    message_type: "message".to_string(),
                    data: args[0].clone(),
                    from: None,
                    to: None,
                    room: None,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_millis() as u64,
                };
                
                self.send(message)?;
                Ok(Value::Boolean(true))
            },
            "disconnect" => {
                self.disconnect()?;
                Ok(Value::Boolean(true))
            },
            "url" => Ok(Value::String(self.url.clone())),
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for ServerSentEvents {
    fn type_name(&self) -> &'static str {
        "ServerSentEvents"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "send" => {
                if args.is_empty() {
                    return Err(LyraError::Runtime { message: "send requires at least 1 argument".to_string()));
                }
                
                let data = match &args[0] {
                    Value::String(s) => s.clone(),
                    v => serde_json::to_string(v).unwrap_or_default(),
                };
                
                let event = SSEMessage {
                    event_type: if args.len() > 1 {
                        match &args[1] {
                            Value::String(s) => Some(s.clone()),
                            _ => None,
                        }
                    } else {
                        None
                    },
                    data,
                    id: None,
                    retry: None,
                };
                
                let sent_count = self.send_event(event, None)?;
                Ok(Value::Integer(sent_count as i64))
            },
            "endpoint" => Ok(Value::String(self.endpoint.clone())),
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for GRPCServer {
    fn type_name(&self) -> &'static str {
        "GRPCServer"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "start" => {
                self.start()?;
                Ok(Value::Boolean(true))
            },
            "port" => Ok(Value::Integer(self.port as i64)),
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

impl Foreign for GRPCClient {
    fn type_name(&self) -> &'static str {
        "GRPCClient"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, LyraError> {
        match method {
            "call" => {
                if args.len() < 2 {
                    return Err(LyraError::Runtime { message: "call requires 2 arguments".to_string()));
                }
                
                let method_name = match &args[0] {
                    Value::String(s) => s.as_str(),
                    _ => return Err(LyraError::TypeError("Method name must be a string".to_string())),
                };
                
                let request = args[1].clone();
                let response = self.call(method_name, request)?;
                Ok(response)
            },
            "service" => Ok(Value::String(self.service.clone())),
            "address" => Ok(Value::String(self.server_address.clone())),
            _ => Err(LyraError::Runtime { message: format!("Unknown method: {}", method))),
        }
    }
}

// Function implementations for Lyra VM
pub fn websocket_server(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Runtime { message: "WebSocketServer requires 2 arguments".to_string()));
    }
    
    let port = match &args[0] {
        Value::Real(n) => *n as u16,
        _ => return Err(LyraError::TypeError("Port must be a number".to_string())),
    };
    
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("Path must be a string".to_string())),
    };
    
    let server = WebSocketServer::new(port, path);
    Ok(Value::LyObj(LyObj::new(Box::new(server))))
}

pub fn websocket_client(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(LyraError::Runtime { message: "WebSocketClient requires at least 1 argument".to_string()));
    }
    
    let url = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("URL must be a string".to_string())),
    };
    
    let protocols = if args.len() > 1 {
        match &args[1] {
            Value::List(list) => {
                list.iter().map(|v| match v {
                    Value::String(s) => Ok(s.clone()),
                    _ => Err(LyraError::TypeError("Protocol must be a string".to_string())),
                }).collect::<Result<Vec<_>, _>>()?
            },
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    
    let client = WebSocketClient::new(url, protocols, None);
    Ok(Value::LyObj(LyObj::new(Box::new(client))))
}

pub fn server_sent_events(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(LyraError::Runtime { message: "ServerSentEvents requires at least 1 argument".to_string()));
    }
    
    let endpoint = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("Endpoint must be a string".to_string())),
    };
    
    let format = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => match s.as_str() {
                "json" => SSEFormat::Json,
                "text" => SSEFormat::PlainText,
                _ => SSEFormat::Custom(s.clone()),
            },
            _ => SSEFormat::Json,
        }
    } else {
        SSEFormat::Json
    };
    
    let sse = ServerSentEvents::new(endpoint, format);
    Ok(Value::LyObj(LyObj::new(Box::new(sse))))
}

pub fn grpc_server(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(LyraError::Runtime { message: "gRPCServer requires at least 1 argument".to_string()));
    }
    
    let port = match &args[0] {
        Value::Real(n) => *n as u16,
        _ => return Err(LyraError::TypeError("Port must be a number".to_string())),
    };
    
    let server = GRPCServer::new(port, None);
    Ok(Value::LyObj(LyObj::new(Box::new(server))))
}

pub fn grpc_client(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Runtime { message: "gRPCClient requires 2 arguments".to_string()));
    }
    
    let server_address = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("Server address must be a string".to_string())),
    };
    
    let service = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(LyraError::TypeError("Service must be a string".to_string())),
    };
    
    let client = GRPCClient::new(server_address, service, None);
    Ok(Value::LyObj(LyObj::new(Box::new(client))))
}

pub fn connection_pool(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(LyraError::Runtime { message: "ConnectionPool requires 2 arguments".to_string()));
    }
    
    let pool_type = match &args[0] {
        Value::String(s) => match s.as_str() {
            "websocket" => ConnectionType::WebSocket,
            "http" => ConnectionType::HTTP,
            "grpc" => ConnectionType::GRPC,
            "database" => ConnectionType::Database,
            _ => return Err(LyraError::ValueError("Unknown connection type".to_string())),
        },
        _ => return Err(LyraError::TypeError("Connection type must be a string".to_string())),
    };
    
    let max_size = match &args[1] {
        Value::Real(n) => *n as usize,
        _ => return Err(LyraError::TypeError("Max size must be a number".to_string())),
    };
    
    let health_check_interval = Duration::from_secs(30); // Default
    
    let pool = ConnectionPool::new(pool_type, max_size, health_check_interval);
    Ok(Value::LyObj(LyObj::new(Box::new(pool))))
}

/// Register all real-time API functions
pub fn register_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = HashMap::new();
    
    functions.insert("WebSocketServer".to_string(), websocket_server as fn(&[Value]) -> VmResult<Value>);
    functions.insert("WebSocketClient".to_string(), websocket_client as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ServerSentEvents".to_string(), server_sent_events as fn(&[Value]) -> VmResult<Value>);
    functions.insert("gRPCServer".to_string(), grpc_server as fn(&[Value]) -> VmResult<Value>);
    functions.insert("gRPCClient".to_string(), grpc_client as fn(&[Value]) -> VmResult<Value>);
    functions.insert("ConnectionPool".to_string(), connection_pool as fn(&[Value]) -> VmResult<Value>);
    
    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_websocket_server_creation() {
        let server = WebSocketServer::new(8080, "/ws".to_string() };
        assert_eq!(server.port, 8080);
        assert_eq!(server.path, "/ws");
    }
    
    #[test]
    fn test_websocket_client_creation() {
        let client = WebSocketClient::new("ws://localhost:8080".to_string(), vec![], None);
        assert_eq!(client.url, "ws://localhost:8080");
    }
    
    #[test]
    fn test_websocket_client_connect() {
        let client = WebSocketClient::new("ws://localhost:8080".to_string(), vec![], None);
        let result = client.connect();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_server_sent_events_creation() {
        let sse = ServerSentEvents::new("/events".to_string(), SSEFormat::Json);
        assert_eq!(sse.endpoint, "/events");
    }
    
    #[test]
    fn test_grpc_server_creation() {
        let server = GRPCServer::new(50051, None);
        assert_eq!(server.port, 50051);
    }
    
    #[test]
    fn test_grpc_client_creation() {
        let client = GRPCClient::new("localhost:50051".to_string(), "TestService".to_string(), None);
        assert_eq!(client.server_address, "localhost:50051");
        assert_eq!(client.service, "TestService");
    }
    
    #[test]
    fn test_connection_pool_creation() {
        let pool = ConnectionPool::new(ConnectionType::WebSocket, 10, Duration::from_secs(30));
        assert_eq!(pool.max_size, 10);
    }
    
    #[test]
    fn test_websocket_server_broadcast() {
        let server = WebSocketServer::new(8080, "/ws".to_string() };
        
        let message = WebSocketMessage {
            message_type: "test".to_string(),
            data: Value::String("hello".to_string() },
            from: None,
            to: None,
            room: None,
            timestamp: 0,
        };
        
        let result = server.broadcast(message, None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_websocket_server_join_room() {
        let server = WebSocketServer::new(8080, "/ws".to_string() };
        let result = server.join_room("conn1".to_string(), "room1".to_string() };
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_sse_send_event() {
        let sse = ServerSentEvents::new("/events".to_string(), SSEFormat::Json);
        
        let event = SSEMessage {
            event_type: Some("test".to_string() },
            data: "test data".to_string(),
            id: None,
            retry: None,
        };
        
        let result = sse.send_event(event, None);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_websocket_server_function() {
        let args = vec![
            Value::Real(8080.0),
            Value::String("/ws".to_string() },
        ];
        let result = websocket_server(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_websocket_client_function() {
        let args = vec![
            Value::String("ws://localhost:8080".to_string() },
            Value::List(vec![Value::String("protocol1".to_string() }]),
        ];
        let result = websocket_client(&args);
        assert!(result.is_ok());
        
        match result.unwrap() {
            Value::LyObj(_) => {}, // Expected
            _ => panic!("Expected LyObj"),
        }
    }
    
    #[test]
    fn test_register_functions() {
        let functions = register_functions();
        assert!(functions.contains_key("WebSocketServer"));
        assert!(functions.contains_key("WebSocketClient"));
        assert!(functions.contains_key("ServerSentEvents"));
        assert!(functions.contains_key("gRPCServer"));
        assert!(functions.contains_key("gRPCClient"));
        assert!(functions.contains_key("ConnectionPool"));
    }
}