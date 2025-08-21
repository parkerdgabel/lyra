use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// WebSocket message types for REPL communication
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// Execute code in the REPL
    Execute,
    /// Result of code execution
    ExecuteResult,
    /// Get session history
    GetHistory,
    /// Session history response
    History,
    /// Export session to various formats
    Export,
    /// Export result
    ExportResult,
    /// Get session information
    GetSession,
    /// Session information response
    SessionInfo,
    /// Error message
    Error,
    /// Heartbeat/ping message
    Ping,
    /// Heartbeat/pong response
    Pong,
    /// Session state update
    StateUpdate,
    /// Variable definitions
    Variables,
    /// Completion request
    Complete,
    /// Completion response
    Completion,
    /// Documentation request
    Help,
    /// Documentation response
    Documentation,
}

/// WebSocket message structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketMessage {
    /// Unique message identifier
    pub id: Uuid,
    /// Message type
    pub message_type: MessageType,
    /// Message payload
    pub data: serde_json::Value,
    /// Message timestamp
    pub timestamp: DateTime<Utc>,
}

impl WebSocketMessage {
    /// Create a new message
    pub fn new(message_type: MessageType, data: serde_json::Value) -> Self {
        Self {
            id: Uuid::new_v4(),
            message_type,
            data,
            timestamp: Utc::now(),
        }
    }

    /// Create an error message
    pub fn error(error: String) -> Self {
        Self::new(
            MessageType::Error,
            serde_json::json!({
                "error": error,
                "timestamp": Utc::now()
            }),
        )
    }

    /// Create a pong response
    pub fn pong() -> Self {
        Self::new(
            MessageType::Pong,
            serde_json::json!({
                "timestamp": Utc::now()
            }),
        )
    }

    /// Create an execute message
    pub fn execute(code: String) -> Self {
        Self::new(
            MessageType::Execute,
            serde_json::json!({
                "code": code
            }),
        )
    }

    /// Create an execute result message
    pub fn execute_result(output: String, execution_time_ms: u64, session_id: Uuid) -> Self {
        Self::new(
            MessageType::ExecuteResult,
            serde_json::json!({
                "output": output,
                "execution_time_ms": execution_time_ms,
                "success": true,
                "session_id": session_id
            }),
        )
    }

    /// Create an execute error message
    pub fn execute_error(error: String, execution_time_ms: u64, session_id: Uuid) -> Self {
        Self::new(
            MessageType::Error,
            serde_json::json!({
                "error": error,
                "execution_time_ms": execution_time_ms,
                "success": false,
                "session_id": session_id
            }),
        )
    }

    /// Create a completion request
    pub fn complete(code: String, cursor_position: usize) -> Self {
        Self::new(
            MessageType::Complete,
            serde_json::json!({
                "code": code,
                "cursor_position": cursor_position
            }),
        )
    }

    /// Create a completion response
    pub fn completion(completions: Vec<CompletionItem>) -> Self {
        Self::new(
            MessageType::Completion,
            serde_json::json!({
                "completions": completions
            }),
        )
    }

    /// Create a help request
    pub fn help(symbol: String) -> Self {
        Self::new(
            MessageType::Help,
            serde_json::json!({
                "symbol": symbol
            }),
        )
    }

    /// Create a documentation response
    pub fn documentation(symbol: String, docs: Documentation) -> Self {
        Self::new(
            MessageType::Documentation,
            serde_json::json!({
                "symbol": symbol,
                "documentation": docs
            }),
        )
    }

    /// Create an export request
    pub fn export(format: String, options: Option<serde_json::Value>) -> Self {
        Self::new(
            MessageType::Export,
            serde_json::json!({
                "format": format,
                "options": options
            }),
        )
    }

    /// Validate message structure
    pub fn validate(&self) -> Result<(), String> {
        match self.message_type {
            MessageType::Execute => {
                if !self.data.get("code").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false) {
                    return Err("Execute message must contain non-empty 'code' field".to_string());
                }
            }
            MessageType::Complete => {
                if !self.data.get("code").and_then(|v| v.as_str()).is_some() {
                    return Err("Complete message must contain 'code' field".to_string());
                }
                if !self.data.get("cursor_position").and_then(|v| v.as_u64()).is_some() {
                    return Err("Complete message must contain 'cursor_position' field".to_string());
                }
            }
            MessageType::Export => {
                if !self.data.get("format").and_then(|v| v.as_str()).is_some() {
                    return Err("Export message must contain 'format' field".to_string());
                }
            }
            MessageType::Help => {
                if !self.data.get("symbol").and_then(|v| v.as_str()).is_some() {
                    return Err("Help message must contain 'symbol' field".to_string());
                }
            }
            _ => {}
        }
        Ok(())
    }
}

/// Code completion item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionItem {
    /// The text to insert
    pub text: String,
    /// Display text (may be different from insert text)
    pub display_text: Option<String>,
    /// Completion kind
    pub kind: CompletionKind,
    /// Documentation for this completion
    pub documentation: Option<String>,
    /// Detail information
    pub detail: Option<String>,
    /// Sort priority (lower numbers are sorted first)
    pub sort_priority: i32,
}

/// Types of completions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionKind {
    Function,
    Variable,
    Constant,
    Keyword,
    Operator,
    Module,
    Type,
    Method,
    Property,
    Field,
    Snippet,
}

/// Documentation for symbols
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Documentation {
    /// Brief description
    pub summary: String,
    /// Detailed description
    pub description: Option<String>,
    /// Function signature
    pub signature: Option<String>,
    /// Parameters
    pub parameters: Vec<Parameter>,
    /// Return type information
    pub returns: Option<String>,
    /// Usage examples
    pub examples: Vec<Example>,
    /// Related functions/symbols
    pub see_also: Vec<String>,
}

/// Parameter documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Parameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: Option<String>,
    /// Parameter description
    pub description: String,
    /// Whether the parameter is optional
    pub optional: bool,
    /// Default value if optional
    pub default_value: Option<String>,
}

/// Usage example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Example {
    /// Example description
    pub title: String,
    /// Example code
    pub code: String,
    /// Expected output
    pub output: Option<String>,
    /// Additional explanation
    pub explanation: Option<String>,
}

/// Session state update message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    /// Session ID
    pub session_id: Uuid,
    /// Variable updates
    pub variables: Option<std::collections::HashMap<String, serde_json::Value>>,
    /// Memory usage
    pub memory_usage: Option<u64>,
    /// Execution count
    pub execution_count: Option<u64>,
    /// Last execution time
    pub last_execution_time: Option<u64>,
}

/// Protocol version information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProtocolVersion {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
    /// Patch version
    pub patch: u32,
    /// Supported features
    pub features: Vec<String>,
}

impl ProtocolVersion {
    /// Current protocol version
    pub fn current() -> Self {
        Self {
            major: 1,
            minor: 0,
            patch: 0,
            features: vec![
                "code_execution".to_string(),
                "session_persistence".to_string(),
                "export_jupyter".to_string(),
                "export_html".to_string(),
                "export_latex".to_string(),
                "export_json".to_string(),
                "completion".to_string(),
                "documentation".to_string(),
                "history".to_string(),
                "variables".to_string(),
            ],
        }
    }

    /// Check if this version is compatible with another
    pub fn is_compatible(&self, other: &ProtocolVersion) -> bool {
        self.major == other.major && self.minor >= other.minor
    }
}

/// Message authentication for secure connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageAuth {
    /// Message signature
    pub signature: String,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Nonce for replay protection
    pub nonce: String,
}

/// Connection handshake message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeMessage {
    /// Protocol version
    pub protocol_version: ProtocolVersion,
    /// Client identification
    pub client_id: String,
    /// Authentication token
    pub auth_token: Option<String>,
    /// Requested features
    pub requested_features: Vec<String>,
}

/// Handshake response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    /// Whether handshake was successful
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
    /// Server protocol version
    pub protocol_version: ProtocolVersion,
    /// Session ID assigned to this connection
    pub session_id: Uuid,
    /// Enabled features
    pub enabled_features: Vec<String>,
    /// Server information
    pub server_info: ServerInfo,
}

/// Server information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerInfo {
    /// Server name
    pub name: String,
    /// Server version
    pub version: String,
    /// Lyra language version
    pub lyra_version: String,
    /// Maximum session timeout
    pub max_session_timeout: u64,
    /// Supported export formats
    pub export_formats: Vec<String>,
}

impl Default for ServerInfo {
    fn default() -> Self {
        Self {
            name: "Lyra WebSocket REPL Server".to_string(),
            version: "1.0.0".to_string(),
            lyra_version: "0.1.0".to_string(),
            max_session_timeout: 86400, // 24 hours
            export_formats: vec![
                "jupyter".to_string(),
                "html".to_string(),
                "latex".to_string(),
                "json".to_string(),
            ],
        }
    }
}

/// Helper function to create standard completion items
pub fn create_function_completion(name: &str, signature: &str, description: &str) -> CompletionItem {
    CompletionItem {
        text: name.to_string(),
        display_text: Some(signature.to_string()),
        kind: CompletionKind::Function,
        documentation: Some(description.to_string()),
        detail: Some(signature.to_string()),
        sort_priority: 1,
    }
}

/// Helper function to create variable completion items
pub fn create_variable_completion(name: &str, var_type: &str) -> CompletionItem {
    CompletionItem {
        text: name.to_string(),
        display_text: Some(format!("{}: {}", name, var_type)),
        kind: CompletionKind::Variable,
        documentation: Some(format!("Variable of type {}", var_type)),
        detail: Some(var_type.to_string()),
        sort_priority: 2,
    }
}

/// Helper function to create constant completion items
pub fn create_constant_completion(name: &str, value: &str, description: &str) -> CompletionItem {
    CompletionItem {
        text: name.to_string(),
        display_text: Some(format!("{} = {}", name, value)),
        kind: CompletionKind::Constant,
        documentation: Some(description.to_string()),
        detail: Some(value.to_string()),
        sort_priority: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = WebSocketMessage::execute("2 + 2".to_string());
        assert_eq!(msg.message_type, MessageType::Execute);
        assert_eq!(msg.data["code"], "2 + 2");
    }

    #[test]
    fn test_message_validation() {
        let mut msg = WebSocketMessage::execute("2 + 2".to_string());
        assert!(msg.validate().is_ok());

        // Test invalid execute message
        msg.data = serde_json::json!({"code": ""});
        assert!(msg.validate().is_err());

        // Test invalid completion message
        msg.message_type = MessageType::Complete;
        msg.data = serde_json::json!({"code": "test"});
        assert!(msg.validate().is_err());
    }

    #[test]
    fn test_protocol_version() {
        let v1 = ProtocolVersion::current();
        let v2 = ProtocolVersion {
            major: 1,
            minor: 1,
            patch: 0,
            features: vec![],
        };
        
        assert!(v2.is_compatible(&v1));
        assert!(!v1.is_compatible(&v2));
    }

    #[test]
    fn test_completion_items() {
        let func_completion = create_function_completion(
            "Sin",
            "Sin[x]",
            "Computes the sine of x"
        );
        assert_eq!(func_completion.kind, CompletionKind::Function);
        assert_eq!(func_completion.text, "Sin");

        let var_completion = create_variable_completion("x", "Number");
        assert_eq!(var_completion.kind, CompletionKind::Variable);
        assert_eq!(var_completion.text, "x");

        let const_completion = create_constant_completion("Pi", "3.14159...", "Mathematical constant π");
        assert_eq!(const_completion.kind, CompletionKind::Constant);
        assert_eq!(const_completion.text, "Pi");
    }

    #[test]
    fn test_error_message() {
        let error_msg = WebSocketMessage::error("Test error".to_string());
        assert_eq!(error_msg.message_type, MessageType::Error);
        assert_eq!(error_msg.data["error"], "Test error");
    }

    #[test]
    fn test_documentation() {
        let doc = Documentation {
            summary: "Computes sine".to_string(),
            description: Some("Mathematical sine function".to_string()),
            signature: Some("Sin[x]".to_string()),
            parameters: vec![Parameter {
                name: "x".to_string(),
                param_type: Some("Number".to_string()),
                description: "Input value".to_string(),
                optional: false,
                default_value: None,
            }],
            returns: Some("Number".to_string()),
            examples: vec![Example {
                title: "Basic usage".to_string(),
                code: "Sin[Pi/2]".to_string(),
                output: Some("1".to_string()),
                explanation: Some("Sine of π/2 is 1".to_string()),
            }],
            see_also: vec!["Cos".to_string(), "Tan".to_string()],
        };

        assert_eq!(doc.summary, "Computes sine");
        assert_eq!(doc.parameters.len(), 1);
        assert_eq!(doc.examples.len(), 1);
    }
}