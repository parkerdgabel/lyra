use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::vm::{VirtualMachine, Value};
use crate::parser::Parser;
use crate::lexer::Lexer;
use crate::compiler::Compiler;
use super::protocol::{WebSocketMessage, MessageType};

/// A single REPL session with persistent state
pub struct ReplSession {
    pub id: Uuid,
    pub created_at: DateTime<Utc>,
    pub last_activity: Arc<RwLock<DateTime<Utc>>>,
    pub vm: Arc<Mutex<VirtualMachine>>,
    pub history: Arc<RwLock<Vec<HistoryEntry>>>,
    pub variables: Arc<RwLock<HashMap<String, Value>>>,
    pub session_data: Arc<RwLock<SessionData>>,
}

/// Session metadata and configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionData {
    pub name: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    pub settings: SessionSettings,
}

/// Per-session settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    pub timeout_seconds: u64,
    pub max_history_entries: usize,
    pub auto_save: bool,
    pub export_formats: Vec<String>,
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            timeout_seconds: 3600, // 1 hour
            max_history_entries: 1000,
            auto_save: true,
            export_formats: vec![
                "jupyter".to_string(),
                "html".to_string(),
                "latex".to_string(),
            ],
        }
    }
}

/// A single entry in the session history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub input: String,
    pub output: String,
    pub execution_time_ms: u64,
    pub success: bool,
    pub error_message: Option<String>,
}

impl ReplSession {
    /// Create a new REPL session
    pub fn new(id: Uuid) -> Self {
        Self {
            id,
            created_at: Utc::now(),
            last_activity: Arc::new(RwLock::new(Utc::now())),
            vm: Arc::new(Mutex::new(VirtualMachine::new())),
            history: Arc::new(RwLock::new(Vec::new())),
            variables: Arc::new(RwLock::new(HashMap::new())),
            session_data: Arc::new(RwLock::new(SessionData {
                name: None,
                description: None,
                tags: Vec::new(),
                settings: SessionSettings::default(),
            })),
        }
    }

    /// Execute code in this session
    pub async fn execute_code(&self, code: &str) -> WebSocketMessage {
        let start_time = Instant::now();
        self.update_activity().await;

        let mut vm = self.vm.lock().await;
        let execution_result = self.run_code(&mut vm, code).await;
        let execution_time = start_time.elapsed().as_millis() as u64;

        let (output, success, error_message) = match execution_result {
            Ok(result) => (result, true, None),
            Err(error) => (String::new(), false, Some(error)),
        };

        // Create history entry
        let history_entry = HistoryEntry {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            input: code.to_string(),
            output: output.clone(),
            execution_time_ms: execution_time,
            success,
            error_message: error_message.clone(),
        };

        // Add to history
        let mut history = self.history.write().await;
        history.push(history_entry);

        // Limit history size
        let max_entries = {
            let session_data = self.session_data.read().await;
            session_data.settings.max_history_entries
        };
        
        let current_len = history.len();
        if current_len > max_entries {
            history.drain(0..(current_len - max_entries));
        }

        // Update variables from VM state
        self.sync_variables(&vm).await;

        drop(vm);
        drop(history);

        // Create response message
        if success {
            WebSocketMessage {
                id: Uuid::new_v4(),
                message_type: MessageType::ExecuteResult,
                data: serde_json::json!({
                    "output": output,
                    "execution_time_ms": execution_time,
                    "success": true,
                    "session_id": self.id
                }),
                timestamp: Utc::now(),
            }
        } else {
            WebSocketMessage {
                id: Uuid::new_v4(),
                message_type: MessageType::Error,
                data: serde_json::json!({
                    "error": error_message.unwrap_or_else(|| "Unknown error".to_string()),
                    "execution_time_ms": execution_time,
                    "success": false,
                    "session_id": self.id
                }),
                timestamp: Utc::now(),
            }
        }
    }

    /// Get session history
    pub async fn get_history(&self) -> WebSocketMessage {
        self.update_activity().await;
        
        let history = self.history.read().await;
        
        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::History,
            data: serde_json::json!({
                "history": *history,
                "count": history.len(),
                "session_id": self.id
            }),
            timestamp: Utc::now(),
        }
    }

    /// Export session to various formats
    pub async fn export_session(&self, format: &str) -> WebSocketMessage {
        self.update_activity().await;

        match format.to_lowercase().as_str() {
            "jupyter" => self.export_to_jupyter().await,
            "html" => self.export_to_html().await,
            "latex" => self.export_to_latex().await,
            "json" => self.export_to_json().await,
            _ => WebSocketMessage::error(format!("Unsupported export format: {}", format)),
        }
    }

    /// Get session information
    pub async fn get_session_info(&self) -> WebSocketMessage {
        self.update_activity().await;
        
        let session_data = self.session_data.read().await;
        let history = self.history.read().await;
        let variables = self.variables.read().await;
        
        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::SessionInfo,
            data: serde_json::json!({
                "id": self.id,
                "created_at": self.created_at,
                "last_activity": *self.last_activity.read().await,
                "history_count": history.len(),
                "variable_count": variables.len(),
                "session_data": *session_data
            }),
            timestamp: Utc::now(),
        }
    }

    /// Check if session has timed out
    pub async fn is_expired(&self) -> bool {
        let last_activity = *self.last_activity.read().await;
        let session_data = self.session_data.read().await;
        let timeout = Duration::from_secs(session_data.settings.timeout_seconds);
        
        Utc::now().signed_duration_since(last_activity).to_std().unwrap_or(Duration::ZERO) > timeout
    }

    /// Update last activity timestamp
    async fn update_activity(&self) {
        let mut last_activity = self.last_activity.write().await;
        *last_activity = Utc::now();
    }

    /// Execute code using the VM
    async fn run_code(&self, vm: &mut VirtualMachine, code: &str) -> Result<String, String> {
        // Tokenize
        let mut lexer = Lexer::new(code);
        let tokens = lexer.tokenize().map_err(|e| format!("Lexer error: {:?}", e))?;

        // Parse
        let mut parser = Parser::new(tokens);
        let ast = parser.parse().map_err(|e| format!("Parser error: {:?}", e))?;

        // Compile
        let mut compiler = Compiler::new();
        compiler.compile_program(&ast).map_err(|e| format!("Compiler error: {:?}", e))?;

        // Load and execute
        vm.load(compiler.context.code, compiler.context.constants);
        let result = vm.execute().map_err(|e| format!("Runtime error: {:?}", e))?;

        // Format result
        Ok(format!("{:?}", result))
    }

    /// Sync variables from VM state
    async fn sync_variables(&self, vm: &VirtualMachine) {
        let mut variables = self.variables.write().await;
        variables.clear();
        
        // Extract variables from VM - this is a simplified implementation
        // In a real implementation, you'd need access to VM's variable storage
        // For now, we'll just track that variables exist
        variables.insert("_last_result".to_string(), Value::Integer(0));
    }

    /// Export session to Jupyter notebook format
    async fn export_to_jupyter(&self) -> WebSocketMessage {
        let history = self.history.read().await;
        
        let mut cells = Vec::new();
        for entry in history.iter() {
            // Code cell
            cells.push(serde_json::json!({
                "cell_type": "code",
                "execution_count": null,
                "metadata": {},
                "outputs": if entry.success {
                    vec![serde_json::json!({
                        "output_type": "execute_result",
                        "data": {
                            "text/plain": [entry.output]
                        },
                        "metadata": {},
                        "execution_count": null
                    })]
                } else {
                    vec![serde_json::json!({
                        "output_type": "error",
                        "ename": "LyraError",
                        "evalue": entry.error_message.as_ref().unwrap_or(&"Unknown error".to_string()),
                        "traceback": []
                    })]
                },
                "source": [entry.input]
            }));
        }

        let notebook = serde_json::json!({
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Lyra",
                    "language": "lyra",
                    "name": "lyra"
                },
                "language_info": {
                    "name": "lyra",
                    "version": "0.1.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        });

        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::ExportResult,
            data: serde_json::json!({
                "format": "jupyter",
                "content": notebook,
                "filename": format!("lyra_session_{}.ipynb", self.id),
                "session_id": self.id
            }),
            timestamp: Utc::now(),
        }
    }

    /// Export session to HTML format
    async fn export_to_html(&self) -> WebSocketMessage {
        let history = self.history.read().await;
        let session_data = self.session_data.read().await;
        
        let mut html_content = String::new();
        html_content.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html_content.push_str("<title>Lyra REPL Session</title>\n");
        html_content.push_str("<meta charset=\"utf-8\">\n");
        html_content.push_str("<style>\n");
        html_content.push_str(r#"
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 2rem; }
.repl-history { max-width: 800px; }
.repl-entry { margin: 1rem 0; padding: 1rem; border: 1px solid #ddd; border-radius: 4px; }
.entry-header { font-weight: bold; color: #666; margin-bottom: 0.5rem; }
.code-input { background: #f5f5f5; padding: 0.5rem; margin: 0.5rem 0; border-radius: 3px; font-family: monospace; }
.code-output { background: #e8f5e8; padding: 0.5rem; margin: 0.5rem 0; border-radius: 3px; font-family: monospace; }
.code-error { background: #ffe8e8; padding: 0.5rem; margin: 0.5rem 0; border-radius: 3px; font-family: monospace; color: #d32f2f; }
"#);
        html_content.push_str("</style>\n");
        html_content.push_str("</head>\n<body>\n");
        
        html_content.push_str(&format!("<h1>Lyra REPL Session {}</h1>\n", self.id));
        html_content.push_str(&format!("<p>Created: {}</p>\n", self.created_at));
        html_content.push_str(&format!("<p>Total entries: {}</p>\n", history.len()));
        
        if let Some(name) = &session_data.name {
            html_content.push_str(&format!("<p>Name: {}</p>\n", name));
        }
        
        html_content.push_str("<div class=\"repl-history\">\n");
        
        for (i, entry) in history.iter().enumerate() {
            html_content.push_str(&format!("<div class=\"repl-entry\">\n"));
            html_content.push_str(&format!("<div class=\"entry-header\">In [{}]: {}</div>\n", i + 1, entry.timestamp));
            html_content.push_str(&format!("<div class=\"code-input\">{}</div>\n", html_escape::encode_text(&entry.input)));
            
            if entry.success {
                html_content.push_str(&format!("<div class=\"code-output\">{}</div>\n", html_escape::encode_text(&entry.output)));
            } else {
                html_content.push_str(&format!("<div class=\"code-error\">{}</div>\n", 
                    html_escape::encode_text(entry.error_message.as_ref().unwrap_or(&"Unknown error".to_string()))));
            }
            
            html_content.push_str("</div>\n");
        }
        
        html_content.push_str("</div>\n</body>\n</html>");

        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::ExportResult,
            data: serde_json::json!({
                "format": "html",
                "content": html_content,
                "filename": format!("lyra_session_{}.html", self.id),
                "session_id": self.id
            }),
            timestamp: Utc::now(),
        }
    }

    /// Export session to LaTeX format
    async fn export_to_latex(&self) -> WebSocketMessage {
        let history = self.history.read().await;
        
        let mut latex_content = String::new();
        latex_content.push_str("\\documentclass{article}\n");
        latex_content.push_str("\\usepackage[utf8]{inputenc}\n");
        latex_content.push_str("\\usepackage{listings}\n");
        latex_content.push_str("\\usepackage{xcolor}\n");
        latex_content.push_str("\\usepackage{amsmath}\n");
        latex_content.push_str("\\usepackage{amsfonts}\n");
        latex_content.push_str("\n");
        latex_content.push_str("\\lstdefinelanguage{Lyra}{\n");
        latex_content.push_str("  keywords={If, While, For, Function, Module, True, False},\n");
        latex_content.push_str("  sensitive=true,\n");
        latex_content.push_str("  comment=[l]{//},\n");
        latex_content.push_str("  string=[b]\",\n");
        latex_content.push_str("}\n");
        latex_content.push_str("\n");
        latex_content.push_str("\\begin{document}\n");
        latex_content.push_str(&format!("\\title{{Lyra REPL Session {}}}\n", self.id));
        latex_content.push_str("\\author{Lyra WebSocket Server}\n");
        latex_content.push_str(&format!("\\date{{{}}}\n", self.created_at.format("%Y-%m-%d %H:%M:%S UTC")));
        latex_content.push_str("\\maketitle\n\n");
        
        for (i, entry) in history.iter().enumerate() {
            latex_content.push_str(&format!("\\subsection{{Entry {}}}\n", i + 1));
            latex_content.push_str("\\textbf{Input:}\n");
            latex_content.push_str("\\begin{lstlisting}[language=Lyra]\n");
            latex_content.push_str(&entry.input);
            latex_content.push_str("\n\\end{lstlisting}\n\n");
            
            latex_content.push_str("\\textbf{Output:}\n");
            if entry.success {
                latex_content.push_str("\\begin{verbatim}\n");
                latex_content.push_str(&entry.output);
                latex_content.push_str("\n\\end{verbatim}\n\n");
            } else {
                latex_content.push_str("\\textcolor{red}{\\textbf{Error:}}\n");
                latex_content.push_str("\\begin{verbatim}\n");
                latex_content.push_str(entry.error_message.as_ref().unwrap_or(&"Unknown error".to_string()));
                latex_content.push_str("\n\\end{verbatim}\n\n");
            }
        }
        
        latex_content.push_str("\\end{document}\n");

        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::ExportResult,
            data: serde_json::json!({
                "format": "latex",
                "content": latex_content,
                "filename": format!("lyra_session_{}.tex", self.id),
                "session_id": self.id
            }),
            timestamp: Utc::now(),
        }
    }

    /// Export session to JSON format
    async fn export_to_json(&self) -> WebSocketMessage {
        let history = self.history.read().await;
        let session_data = self.session_data.read().await;
        
        let export_data = serde_json::json!({
            "session_id": self.id,
            "created_at": self.created_at,
            "last_activity": *self.last_activity.read().await,
            "session_data": *session_data,
            "history": *history
        });

        WebSocketMessage {
            id: Uuid::new_v4(),
            message_type: MessageType::ExportResult,
            data: serde_json::json!({
                "format": "json",
                "content": export_data,
                "filename": format!("lyra_session_{}.json", self.id),
                "session_id": self.id
            }),
            timestamp: Utc::now(),
        }
    }
}

/// Manages multiple REPL sessions
pub struct SessionManager {
    sessions: Arc<RwLock<HashMap<Uuid, Arc<ReplSession>>>>,
    max_sessions: usize,
    default_timeout: u64,
}

impl SessionManager {
    /// Create a new session manager
    pub fn new(max_sessions: usize, default_timeout: u64) -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            max_sessions,
            default_timeout,
        }
    }

    /// Create a new session
    pub async fn create_session(&self, id: Uuid) -> Result<Arc<ReplSession>, String> {
        let mut sessions = self.sessions.write().await;
        
        if sessions.len() >= self.max_sessions {
            return Err("Maximum number of sessions reached".to_string());
        }
        
        let session = Arc::new(ReplSession::new(id));
        sessions.insert(id, session.clone());
        
        Ok(session)
    }

    /// Get a session by ID
    pub async fn get_session(&self, id: &Uuid) -> Option<Arc<ReplSession>> {
        let sessions = self.sessions.read().await;
        sessions.get(id).cloned()
    }

    /// Remove a session
    pub async fn remove_session(&self, id: &Uuid) {
        let mut sessions = self.sessions.write().await;
        sessions.remove(id);
    }

    /// List all active sessions
    pub async fn list_sessions(&self) -> Vec<Uuid> {
        let sessions = self.sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// Get the number of active sessions
    pub async fn session_count(&self) -> usize {
        let sessions = self.sessions.read().await;
        sessions.len()
    }

    /// Get maximum sessions
    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }

    /// Clean up expired sessions
    pub async fn cleanup_expired_sessions(&self) {
        let mut sessions = self.sessions.write().await;
        let mut expired_ids = Vec::new();
        
        for (id, session) in sessions.iter() {
            if session.is_expired().await {
                expired_ids.push(*id);
            }
        }
        
        for id in expired_ids {
            sessions.remove(&id);
            println!("🧹 Cleaned up expired session: {}", id);
        }
    }

    /// Shutdown all sessions
    pub async fn shutdown_all(&self) {
        let mut sessions = self.sessions.write().await;
        sessions.clear();
        println!("🛑 All sessions shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_session_creation() {
        let id = Uuid::new_v4();
        let session = ReplSession::new(id);
        assert_eq!(session.id, id);
        
        let history = session.history.read().await;
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_session_manager() {
        let manager = SessionManager::new(10, 3600);
        let id = Uuid::new_v4();
        
        let session = manager.create_session(id).await.unwrap();
        assert_eq!(session.id, id);
        
        let retrieved = manager.get_session(&id).await;
        assert!(retrieved.is_some());
        
        manager.remove_session(&id).await;
        let removed = manager.get_session(&id).await;
        assert!(removed.is_none());
    }

    #[tokio::test]
    async fn test_session_manager_max_sessions() {
        let manager = SessionManager::new(2, 3600);
        
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        
        assert!(manager.create_session(id1).await.is_ok());
        assert!(manager.create_session(id2).await.is_ok());
        assert!(manager.create_session(id3).await.is_err());
    }
}