use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use axum::{
    extract::{ws::WebSocketUpgrade, Query, State},
    response::Response,
    routing::get,
    Router,
};
use tokio_tungstenite::tungstenite::Message;

pub mod session;
pub mod protocol;
pub mod auth;
pub mod security;

use session::{ReplSession, SessionManager};
use protocol::{WebSocketMessage, MessageType};
use auth::AuthManager;
use security::SecurityMiddleware;

/// Configuration for the WebSocket REPL server
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Server host address
    pub host: String,
    /// Server port
    pub port: u16,
    /// Maximum number of concurrent sessions
    pub max_sessions: usize,
    /// Session timeout in seconds
    pub session_timeout: u64,
    /// Enable authentication
    pub require_auth: bool,
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Maximum requests per minute per client
    pub rate_limit_rpm: u32,
    /// SSL/TLS configuration
    pub ssl_config: Option<SslConfig>,
}

/// SSL/TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SslConfig {
    pub cert_file: String,
    pub key_file: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_sessions: 100,
            session_timeout: 3600, // 1 hour
            require_auth: false,
            enable_rate_limiting: true,
            rate_limit_rpm: 60,
            ssl_config: None,
        }
    }
}

/// The main WebSocket REPL server
pub struct WebSocketReplServer {
    config: ServerConfig,
    session_manager: Arc<SessionManager>,
    auth_manager: Arc<AuthManager>,
    security_middleware: Arc<SecurityMiddleware>,
    shutdown_tx: broadcast::Sender<()>,
}

impl WebSocketReplServer {
    /// Create a new WebSocket REPL server
    pub fn new(config: ServerConfig) -> Self {
        let (shutdown_tx, _) = broadcast::channel(1);
        
        Self {
            session_manager: Arc::new(SessionManager::new(
                config.max_sessions,
                config.session_timeout,
            )),
            auth_manager: Arc::new(AuthManager::new(config.require_auth)),
            security_middleware: Arc::new(SecurityMiddleware::new(
                config.enable_rate_limiting,
                config.rate_limit_rpm,
            )),
            config,
            shutdown_tx,
        }
    }

    /// Start the WebSocket server
    pub async fn start(&self) -> Result<(), Box<dyn std::error::Error>> {
        let app = self.create_app();
        let addr = SocketAddr::from(([127, 0, 0, 1], self.config.port));

        println!("🚀 Lyra WebSocket REPL Server starting on {}", addr);
        println!("📊 Max sessions: {}", self.config.max_sessions);
        println!("🔒 Authentication: {}", if self.config.require_auth { "enabled" } else { "disabled" });
        println!("🛡️  Rate limiting: {}", if self.config.enable_rate_limiting { "enabled" } else { "disabled" });

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create the Axum application with routes
    fn create_app(&self) -> Router {
        let app_state = AppState {
            session_manager: self.session_manager.clone(),
            auth_manager: self.auth_manager.clone(),
            security_middleware: self.security_middleware.clone(),
            shutdown_tx: self.shutdown_tx.clone(),
        };

        Router::new()
            .route("/ws", get(websocket_handler))
            .route("/health", get(health_check))
            .route("/sessions", get(list_sessions))
            .route("/metrics", get(server_metrics))
            .with_state(app_state)
    }

    /// Gracefully shutdown the server
    pub async fn shutdown(&self) -> Result<(), Box<dyn std::error::Error>> {
        println!("🛑 Shutting down WebSocket REPL server...");
        
        // Signal all sessions to close
        let _ = self.shutdown_tx.send(());
        
        // Wait for sessions to close
        self.session_manager.shutdown_all().await;
        
        println!("✅ Server shutdown complete");
        Ok(())
    }
}

/// Shared application state
#[derive(Clone)]
struct AppState {
    session_manager: Arc<SessionManager>,
    auth_manager: Arc<AuthManager>,
    security_middleware: Arc<SecurityMiddleware>,
    shutdown_tx: broadcast::Sender<()>,
}

/// WebSocket connection handler
async fn websocket_handler(
    ws: WebSocketUpgrade,
    Query(params): Query<HashMap<String, String>>,
    State(state): State<AppState>,
) -> Response {
    ws.on_upgrade(move |socket| handle_websocket_connection(socket, params, state))
}

/// Handle individual WebSocket connections
async fn handle_websocket_connection(
    socket: axum::extract::ws::WebSocket,
    params: HashMap<String, String>,
    state: AppState,
) {
    use axum::extract::ws::Message;
    use futures_util::{SinkExt, StreamExt};

    // Extract authentication token if provided
    let auth_token = params.get("token").cloned();
    
    // Authenticate the connection
    if let Err(e) = state.auth_manager.authenticate(auth_token.as_deref()).await {
        println!("❌ Authentication failed: {}", e);
        return;
    }

    // Apply security middleware
    let client_ip = "127.0.0.1"; // TODO: Extract real client IP
    if let Err(e) = state.security_middleware.check_rate_limit(client_ip).await {
        println!("❌ Rate limit exceeded for {}: {}", client_ip, e);
        return;
    }

    // Create a new REPL session
    let session_id = Uuid::new_v4();
    let session = match state.session_manager.create_session(session_id).await {
        Ok(session) => session,
        Err(e) => {
            println!("❌ Failed to create session: {}", e);
            return;
        }
    };

    println!("✅ New WebSocket connection established: {}", session_id);

    let (mut ws_sender, mut ws_receiver) = socket.split();
    let mut shutdown_rx = state.shutdown_tx.subscribe();

    // Send welcome message
    let welcome_msg = protocol::WebSocketMessage {
        id: Uuid::new_v4(),
        message_type: MessageType::SessionInfo,
        data: serde_json::json!({
            "session_id": session_id,
            "message": "Welcome to Lyra WebSocket REPL!",
            "features": ["code_execution", "session_persistence", "export"]
        }),
        timestamp: chrono::Utc::now(),
    };

    if let Ok(msg_json) = serde_json::to_string(&welcome_msg) {
        let _ = ws_sender.send(Message::Text(msg_json)).await;
    }

    // Main message processing loop
    loop {
        tokio::select! {
            // Handle incoming WebSocket messages
            msg = ws_receiver.next() => {
                match msg {
                    Some(Ok(Message::Text(text))) => {
                        if let Err(e) = handle_text_message(&text, &session, &mut ws_sender).await {
                            println!("❌ Error handling message: {}", e);
                        }
                    }
                    Some(Ok(Message::Binary(_))) => {
                        // Binary messages not supported yet
                        let error_msg = protocol::WebSocketMessage::error(
                            "Binary messages not supported".to_string()
                        );
                        if let Ok(msg_json) = serde_json::to_string(&error_msg) {
                            let _ = ws_sender.send(Message::Text(msg_json)).await;
                        }
                    }
                    Some(Ok(Message::Close(_))) => {
                        println!("🔌 WebSocket connection closed: {}", session_id);
                        break;
                    }
                    Some(Err(e)) => {
                        println!("❌ WebSocket error: {}", e);
                        break;
                    }
                    None => break,
                }
            }
            
            // Handle server shutdown
            _ = shutdown_rx.recv() => {
                println!("🛑 Server shutdown signal received for session {}", session_id);
                let _ = ws_sender.send(Message::Close(None)).await;
                break;
            }
        }
    }

    // Cleanup session
    state.session_manager.remove_session(&session_id).await;
    println!("🧹 Session {} cleaned up", session_id);
}

/// Handle text messages from WebSocket clients
async fn handle_text_message(
    text: &str,
    session: &Arc<ReplSession>,
    ws_sender: &mut futures_util::stream::SplitSink<axum::extract::ws::WebSocket, Message>,
) -> Result<(), Box<dyn std::error::Error>> {
    use futures_util::SinkExt;

    // Parse the incoming message
    let message: protocol::WebSocketMessage = serde_json::from_str(text)?;

    let response = match message.message_type {
        MessageType::Execute => {
            // Execute code in the REPL session
            if let Some(code) = message.data.get("code").and_then(|v| v.as_str()) {
                session.execute_code(code).await
            } else {
                protocol::WebSocketMessage::error("Missing 'code' field".to_string())
            }
        }
        MessageType::GetHistory => {
            // Get session history
            session.get_history().await
        }
        MessageType::Export => {
            // Export session to various formats
            if let Some(format) = message.data.get("format").and_then(|v| v.as_str()) {
                session.export_session(format).await
            } else {
                protocol::WebSocketMessage::error("Missing 'format' field".to_string())
            }
        }
        MessageType::GetSession => {
            // Get session information
            session.get_session_info().await
        }
        _ => {
            protocol::WebSocketMessage::error(format!("Unsupported message type: {:?}", message.message_type))
        }
    };

    // Send response back to client
    if let Ok(response_json) = serde_json::to_string(&response) {
        ws_sender.send(Message::Text(response_json)).await?;
    }

    Ok(())
}

/// Health check endpoint
async fn health_check() -> &'static str {
    "OK"
}

/// List active sessions endpoint
async fn list_sessions(
    State(state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, axum::http::StatusCode> {
    let sessions = state.session_manager.list_sessions().await;
    Ok(axum::Json(serde_json::json!({
        "sessions": sessions,
        "count": sessions.len()
    })))
}

/// Server metrics endpoint
async fn server_metrics(
    State(state): State<AppState>,
) -> Result<axum::Json<serde_json::Value>, axum::http::StatusCode> {
    let metrics = serde_json::json!({
        "active_sessions": state.session_manager.session_count().await,
        "max_sessions": state.session_manager.max_sessions(),
        "uptime_seconds": 0, // TODO: Track server uptime
        "total_connections": 0, // TODO: Track total connections
        "memory_usage": 0, // TODO: Track memory usage
    });
    
    Ok(axum::Json(metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_config_default() {
        let config = ServerConfig::default();
        assert_eq!(config.host, "127.0.0.1");
        assert_eq!(config.port, 8080);
        assert_eq!(config.max_sessions, 100);
        assert!(!config.require_auth);
        assert!(config.enable_rate_limiting);
    }

    #[tokio::test]
    async fn test_server_creation() {
        let config = ServerConfig::default();
        let server = WebSocketReplServer::new(config);
        
        // Test server can be created without panicking
        assert_eq!(server.config.port, 8080);
    }
}