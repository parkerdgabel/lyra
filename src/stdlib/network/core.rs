//! Core Network Primitives and Abstractions
//!
//! This module implements the fundamental building blocks for network operations in Lyra,
//! designed as symbolic objects that can be composed, analyzed, and transformed.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// HTTP methods as symbolic values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
    TRACE,
}

impl From<&str> for HttpMethod {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "GET" => HttpMethod::GET,
            "POST" => HttpMethod::POST,
            "PUT" => HttpMethod::PUT,
            "DELETE" => HttpMethod::DELETE,
            "PATCH" => HttpMethod::PATCH,
            "HEAD" => HttpMethod::HEAD,
            "OPTIONS" => HttpMethod::OPTIONS,
            "TRACE" => HttpMethod::TRACE,
            _ => HttpMethod::GET, // Default fallback
        }
    }
}

impl std::fmt::Display for HttpMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HttpMethod::GET => write!(f, "GET"),
            HttpMethod::POST => write!(f, "POST"),
            HttpMethod::PUT => write!(f, "PUT"),
            HttpMethod::DELETE => write!(f, "DELETE"),
            HttpMethod::PATCH => write!(f, "PATCH"),
            HttpMethod::HEAD => write!(f, "HEAD"),
            HttpMethod::OPTIONS => write!(f, "OPTIONS"),
            HttpMethod::TRACE => write!(f, "TRACE"),
        }
    }
}

/// Network endpoint abstraction for service location
#[derive(Debug, Clone)]
pub struct NetworkEndpoint {
    /// Host or IP address
    pub host: String,
    /// Port number
    pub port: u16,
    /// Protocol (http, https, ws, wss, tcp, udp)
    pub protocol: String,
    /// Optional path component
    pub path: Option<String>,
    /// Additional endpoint metadata
    pub metadata: HashMap<String, String>,
}

impl NetworkEndpoint {
    /// Create a new network endpoint
    pub fn new(host: String, port: u16, protocol: String) -> Self {
        Self {
            host,
            port,
            protocol,
            path: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create endpoint from URL string
    pub fn from_url(url: &str) -> Result<Self, String> {
        // Simple URL parsing - in production would use a proper URL parser
        let url = url.trim();
        
        // Extract protocol
        let (protocol, rest) = if let Some(pos) = url.find("://") {
            (url[..pos].to_string(), &url[pos + 3..])
        } else {
            return Err("Invalid URL: missing protocol".to_string());
        };
        
        // Extract host and port
        let (host_port, path) = if let Some(pos) = rest.find('/') {
            (&rest[..pos], Some(rest[pos..].to_string()))
        } else {
            (rest, None)
        };
        
        let (host, port) = if let Some(pos) = host_port.find(':') {
            let host = host_port[..pos].to_string();
            let port_str = &host_port[pos + 1..];
            let port = port_str.parse::<u16>()
                .map_err(|_| format!("Invalid port: {}", port_str))?;
            (host, port)
        } else {
            let default_port = match protocol.as_str() {
                "http" => 80,
                "https" => 443,
                "ws" => 80,
                "wss" => 443,
                "ftp" => 21,
                _ => 80,
            };
            (host_port.to_string(), default_port)
        };
        
        Ok(Self {
            host,
            port,
            protocol,
            path,
            metadata: HashMap::new(),
        })
    }
    
    /// Convert endpoint to URL string
    pub fn to_url(&self) -> String {
        let base = format!("{}://{}:{}", self.protocol, self.host, self.port);
        if let Some(ref path) = self.path {
            format!("{}{}", base, path)
        } else {
            base
        }
    }
    
    /// Add metadata to endpoint
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

impl Foreign for NetworkEndpoint {
    fn type_name(&self) -> &'static str {
        "NetworkEndpoint"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Host" => Ok(Value::String(self.host.clone())),
            "Port" => Ok(Value::Integer(self.port as i64)),
            "Protocol" => Ok(Value::String(self.protocol.clone())),
            "Path" => Ok(self.path.as_ref()
                .map(|p| Value::String(p.clone()))
                .unwrap_or(Value::String("".to_string()))),
            "URL" => Ok(Value::String(self.to_url())),
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

/// Network authentication abstraction
#[derive(Debug, Clone)]
pub enum NetworkAuth {
    /// No authentication
    None,
    /// Basic HTTP authentication
    Basic { username: String, password: String },
    /// Bearer token authentication
    Bearer { token: String },
    /// API key authentication
    ApiKey { key: String, header: String },
    /// OAuth 2.0 authentication
    OAuth2 { token: String, refresh_token: Option<String> },
    /// Custom authentication scheme
    Custom { scheme: String, credentials: HashMap<String, String> },
}

impl NetworkAuth {
    /// Apply authentication to headers
    pub fn apply_to_headers(&self, headers: &mut HashMap<String, String>) {
        match self {
            NetworkAuth::None => {}
            NetworkAuth::Basic { username, password } => {
                let credentials = format!("{}:{}", username, password);
                let encoded = base64_encode(&credentials);
                headers.insert("Authorization".to_string(), format!("Basic {}", encoded));
            }
            NetworkAuth::Bearer { token } => {
                headers.insert("Authorization".to_string(), format!("Bearer {}", token));
            }
            NetworkAuth::ApiKey { key, header } => {
                headers.insert(header.clone(), key.clone());
            }
            NetworkAuth::OAuth2 { token, .. } => {
                headers.insert("Authorization".to_string(), format!("Bearer {}", token));
            }
            NetworkAuth::Custom { credentials, .. } => {
                for (key, value) in credentials {
                    headers.insert(key.clone(), value.clone());
                }
            }
        }
    }
    
    /// Get authentication type as string
    pub fn auth_type(&self) -> &'static str {
        match self {
            NetworkAuth::None => "None",
            NetworkAuth::Basic { .. } => "Basic",
            NetworkAuth::Bearer { .. } => "Bearer",
            NetworkAuth::ApiKey { .. } => "ApiKey",
            NetworkAuth::OAuth2 { .. } => "OAuth2",
            NetworkAuth::Custom { .. } => "Custom",
        }
    }
}

/// Simple base64 encoding (placeholder - would use proper implementation)
fn base64_encode(input: &str) -> String {
    // Simplified base64 encoding - in production would use base64 crate
    let bytes = input.as_bytes();
    let mut result = String::new();
    
    for chunk in bytes.chunks(3) {
        let mut buf = [0u8; 3];
        for (i, &byte) in chunk.iter().enumerate() {
            buf[i] = byte;
        }
        
        let b1 = buf[0] >> 2;
        let b2 = ((buf[0] & 0x03) << 4) | (buf[1] >> 4);
        let b3 = ((buf[1] & 0x0f) << 2) | (buf[2] >> 6);
        let b4 = buf[2] & 0x3f;
        
        result.push(base64_char(b1));
        result.push(base64_char(b2));
        result.push(if chunk.len() > 1 { base64_char(b3) } else { '=' });
        result.push(if chunk.len() > 2 { base64_char(b4) } else { '=' });
    }
    
    result
}

fn base64_char(b: u8) -> char {
    match b {
        0..=25 => (b'A' + b) as char,
        26..=51 => (b'a' + (b - 26)) as char,
        52..=61 => (b'0' + (b - 52)) as char,
        62 => '+',
        63 => '/',
        _ => '=',
    }
}

impl Foreign for NetworkAuth {
    fn type_name(&self) -> &'static str {
        "NetworkAuth"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Type" => Ok(Value::String(self.auth_type().to_string())),
            "IsNone" => Ok(Value::Integer(if matches!(self, NetworkAuth::None) { 1 } else { 0 })),
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

/// Symbolic network request representation
#[derive(Debug, Clone)]
pub struct NetworkRequest {
    /// Target URL or endpoint
    pub url: String,
    /// HTTP method
    pub method: HttpMethod,
    /// Request headers
    pub headers: HashMap<String, String>,
    /// Request body (if any)
    pub body: Option<Vec<u8>>,
    /// Authentication configuration
    pub auth: NetworkAuth,
    /// Request timeout
    pub timeout: Duration,
    /// Number of retry attempts
    pub retries: u32,
    /// Request metadata and configuration
    pub metadata: HashMap<String, String>,
    /// Timestamp when request was created
    pub created_at: SystemTime,
}

impl NetworkRequest {
    /// Create a new network request
    pub fn new(url: String, method: HttpMethod) -> Self {
        Self {
            url,
            method,
            headers: HashMap::new(),
            body: None,
            auth: NetworkAuth::None,
            timeout: Duration::from_secs(30),
            retries: 3,
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
        }
    }
    
    /// Add header to request
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
    
    /// Set request body
    pub fn with_body(mut self, body: Vec<u8>) -> Self {
        self.body = Some(body);
        self
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
    
    /// Set retry count
    pub fn with_retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    /// Get effective headers (including auth)
    pub fn effective_headers(&self) -> HashMap<String, String> {
        let mut headers = self.headers.clone();
        self.auth.apply_to_headers(&mut headers);
        
        // Add default headers if not present
        if !headers.contains_key("User-Agent") {
            headers.insert("User-Agent".to_string(), "Lyra/1.0".to_string());
        }
        
        if self.body.is_some() && !headers.contains_key("Content-Length") {
            let length = self.body.as_ref().map(|b| b.len()).unwrap_or(0);
            headers.insert("Content-Length".to_string(), length.to_string());
        }
        
        headers
    }
}

impl Foreign for NetworkRequest {
    fn type_name(&self) -> &'static str {
        "NetworkRequest"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "URL" => Ok(Value::String(self.url.clone())),
            "Method" => Ok(Value::String(self.method.to_string())),
            "Headers" => {
                let headers = self.effective_headers();
                let entries: Vec<Value> = headers.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
            }
            "HasBody" => Ok(Value::Integer(if self.body.is_some() { 1 } else { 0 })),
            "BodySize" => Ok(Value::Integer(
                self.body.as_ref().map(|b| b.len()).unwrap_or(0) as i64
            )),
            "Timeout" => Ok(Value::Real(self.timeout.as_secs_f64())),
            "Retries" => Ok(Value::Integer(self.retries as i64)),
            "AuthType" => Ok(Value::String(self.auth.auth_type().to_string())),
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

/// Symbolic network response representation
#[derive(Debug, Clone)]
pub struct NetworkResponse {
    /// HTTP status code
    pub status: u16,
    /// Response headers
    pub headers: HashMap<String, String>,
    /// Response body
    pub body: Vec<u8>,
    /// Content type
    pub content_type: String,
    /// Response time in milliseconds
    pub response_time: f64,
    /// Final URL (after redirects)
    pub final_url: String,
    /// Whether response was from cache
    pub from_cache: bool,
    /// Response metadata
    pub metadata: HashMap<String, String>,
    /// Timestamp when response was received
    pub received_at: SystemTime,
}

impl NetworkResponse {
    /// Create a new network response
    pub fn new(status: u16, headers: HashMap<String, String>, body: Vec<u8>) -> Self {
        let content_type = headers.get("content-type")
            .or_else(|| headers.get("Content-Type"))
            .unwrap_or(&"application/octet-stream".to_string())
            .clone();
        
        Self {
            status,
            headers,
            body,
            content_type,
            response_time: 0.0,
            final_url: String::new(),
            from_cache: false,
            metadata: HashMap::new(),
            received_at: SystemTime::now(),
        }
    }
    
    /// Check if response was successful (2xx status)
    pub fn is_success(&self) -> bool {
        self.status >= 200 && self.status < 300
    }
    
    /// Check if response is a redirect (3xx status)
    pub fn is_redirect(&self) -> bool {
        self.status >= 300 && self.status < 400
    }
    
    /// Check if response is a client error (4xx status)
    pub fn is_client_error(&self) -> bool {
        self.status >= 400 && self.status < 500
    }
    
    /// Check if response is a server error (5xx status)
    pub fn is_server_error(&self) -> bool {
        self.status >= 500 && self.status < 600
    }
    
    /// Get response body as string (assumes UTF-8)
    pub fn body_as_string(&self) -> Result<String, String> {
        String::from_utf8(self.body.clone())
            .map_err(|e| format!("Invalid UTF-8 in response body: {}", e))
    }
    
    /// Check if response is JSON
    pub fn is_json(&self) -> bool {
        self.content_type.contains("application/json") || 
        self.content_type.contains("text/json")
    }
    
    /// Check if response is HTML
    pub fn is_html(&self) -> bool {
        self.content_type.contains("text/html")
    }
    
    /// Check if response is XML
    pub fn is_xml(&self) -> bool {
        self.content_type.contains("application/xml") || 
        self.content_type.contains("text/xml")
    }
}

impl Foreign for NetworkResponse {
    fn type_name(&self) -> &'static str {
        "NetworkResponse"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Status" => Ok(Value::Integer(self.status as i64)),
            "IsSuccess" => Ok(Value::Integer(if self.is_success() { 1 } else { 0 })),
            "IsError" => Ok(Value::Integer(if !self.is_success() { 1 } else { 0 })),
            "IsRedirect" => Ok(Value::Integer(if self.is_redirect() { 1 } else { 0 })),
            "IsClientError" => Ok(Value::Integer(if self.is_client_error() { 1 } else { 0 })),
            "IsServerError" => Ok(Value::Integer(if self.is_server_error() { 1 } else { 0 })),
            "Headers" => {
                let entries: Vec<Value> = self.headers.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
            }
            "Body" => self.body_as_string()
                .map(|s| Value::String(s))
                .map_err(|e| ForeignError::RuntimeError { message: e }),
            "BodySize" => Ok(Value::Integer(self.body.len() as i64)),
            "ContentType" => Ok(Value::String(self.content_type.clone())),
            "IsJSON" => Ok(Value::Integer(if self.is_json() { 1 } else { 0 })),
            "IsHTML" => Ok(Value::Integer(if self.is_html() { 1 } else { 0 })),
            "IsXML" => Ok(Value::Integer(if self.is_xml() { 1 } else { 0 })),
            "ResponseTime" => Ok(Value::Real(self.response_time)),
            "FinalURL" => Ok(Value::String(self.final_url.clone())),
            "FromCache" => Ok(Value::Integer(if self.from_cache { 1 } else { 0 })),
            "ReceivedAt" => {
                let timestamp = self.received_at.duration_since(UNIX_EPOCH)
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

/// Create a NetworkEndpoint from host, port, and protocol
/// Syntax: NetworkEndpoint[host, port, protocol]
pub fn network_endpoint(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (host, port, [protocol])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let host = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for host".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let port = match &args[1] {
        Value::Integer(i) => *i as u16,
        _ => return Err(VmError::TypeError {
            expected: "Integer for port".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let protocol = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "String for protocol".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    } else {
        "http".to_string()
    };
    
    let endpoint = NetworkEndpoint::new(host, port, protocol);
    Ok(Value::LyObj(LyObj::new(Box::new(endpoint))))
}

/// Create a NetworkRequest from URL and method
/// Syntax: NetworkRequest[url, method, headers, body]
pub fn network_request(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "1-4 arguments (url, [method], [headers], [body])".to_string(),
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
    
    let method = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => HttpMethod::from(s.as_str()),
            _ => return Err(VmError::TypeError {
                expected: "String for method".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        HttpMethod::GET
    };
    
    let mut request = NetworkRequest::new(url, method);
    
    // Add headers if provided
    if args.len() > 2 {
        match &args[2] {
            Value::List(headers) => {
                for header in headers {
                    match header {
                        Value::List(pair) if pair.len() == 2 => {
                            let key = match &pair[0] {
                                Value::String(s) => s.clone(),
                                _ => continue,
                            };
                            let value = match &pair[1] {
                                Value::String(s) => s.clone(),
                                _ => continue,
                            };
                            request = request.with_header(key, value);
                        }
                        _ => continue,
                    }
                }
            }
            _ => return Err(VmError::TypeError {
                expected: "List of header pairs for headers".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    }
    
    // Add body if provided
    if args.len() > 3 {
        match &args[3] {
            Value::String(s) => {
                request = request.with_body(s.as_bytes().to_vec());
            }
            _ => return Err(VmError::TypeError {
                expected: "String for body".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(request))))
}

/// Create NetworkAuth object
/// Syntax: NetworkAuth[type, credentials]
pub fn network_auth(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (type, [credentials])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let auth_type = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "String for auth type".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let auth = match auth_type.to_lowercase().as_str() {
        "none" => NetworkAuth::None,
        "basic" => {
            if args.len() < 2 {
                return Err(VmError::TypeError {
                    expected: "Credentials required for Basic auth".to_string(),
                    actual: "No credentials provided".to_string(),
                });
            }
            match &args[1] {
                Value::List(creds) if creds.len() == 2 => {
                    let username = match &creds[0] {
                        Value::String(s) => s.clone(),
                        _ => return Err(VmError::TypeError {
                            expected: "String for username".to_string(),
                            actual: format!("{:?}", creds[0]),
                        }),
                    };
                    let password = match &creds[1] {
                        Value::String(s) => s.clone(),
                        _ => return Err(VmError::TypeError {
                            expected: "String for password".to_string(),
                            actual: format!("{:?}", creds[1]),
                        }),
                    };
                    NetworkAuth::Basic { username, password }
                }
                _ => return Err(VmError::TypeError {
                    expected: "List of [username, password] for Basic auth".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            }
        }
        "bearer" => {
            if args.len() < 2 {
                return Err(VmError::TypeError {
                    expected: "Token required for Bearer auth".to_string(),
                    actual: "No token provided".to_string(),
                });
            }
            match &args[1] {
                Value::String(token) => NetworkAuth::Bearer { token: token.clone() },
                _ => return Err(VmError::TypeError {
                    expected: "String for Bearer token".to_string(),
                    actual: format!("{:?}", args[1]),
                }),
            }
        }
        _ => return Err(VmError::Runtime(format!("Unsupported auth type: {}", auth_type))),
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(auth))))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_network_endpoint_creation() {
        let endpoint = NetworkEndpoint::new(
            "example.com".to_string(),
            80,
            "http".to_string()
        );
        
        assert_eq!(endpoint.host, "example.com");
        assert_eq!(endpoint.port, 80);
        assert_eq!(endpoint.protocol, "http");
        assert_eq!(endpoint.to_url(), "http://example.com:80");
    }
    
    #[test]
    fn test_network_endpoint_from_url() {
        let endpoint = NetworkEndpoint::from_url("https://api.example.com:8443/v1/data").unwrap();
        
        assert_eq!(endpoint.host, "api.example.com");
        assert_eq!(endpoint.port, 8443);
        assert_eq!(endpoint.protocol, "https");
        assert_eq!(endpoint.path, Some("/v1/data".to_string()));
    }
    
    #[test]
    fn test_network_request_creation() {
        let request = NetworkRequest::new(
            "https://api.example.com/data".to_string(),
            HttpMethod::GET
        )
        .with_header("Content-Type".to_string(), "application/json".to_string())
        .with_auth(NetworkAuth::Bearer { token: "secret".to_string() });
        
        assert_eq!(request.url, "https://api.example.com/data");
        assert_eq!(request.method, HttpMethod::GET);
        assert!(request.headers.contains_key("Content-Type"));
        
        let effective_headers = request.effective_headers();
        assert!(effective_headers.contains_key("Authorization"));
        assert!(effective_headers["Authorization"].starts_with("Bearer"));
    }
    
    #[test]
    fn test_network_response_properties() {
        let mut headers = HashMap::new();
        headers.insert("content-type".to_string(), "application/json".to_string());
        
        let response = NetworkResponse::new(
            200,
            headers,
            br#"{"status": "ok"}"#.to_vec()
        );
        
        assert!(response.is_success());
        assert!(!response.is_client_error());
        assert!(response.is_json());
        assert_eq!(response.body_as_string().unwrap(), r#"{"status": "ok"}"#);
    }
    
    #[test]
    fn test_basic_auth_encoding() {
        let auth = NetworkAuth::Basic {
            username: "user".to_string(),
            password: "pass".to_string()
        };
        
        let mut headers = HashMap::new();
        auth.apply_to_headers(&mut headers);
        
        assert!(headers.contains_key("Authorization"));
        assert!(headers["Authorization"].starts_with("Basic"));
    }
}