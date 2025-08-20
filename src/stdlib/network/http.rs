//! HTTP Operations and URL Processing
//!
//! This module implements HTTP-specific networking operations that build on the core
//! network primitives, providing URLRead, URLWrite, and related HTTP functionality.

use super::core::{NetworkRequest, NetworkResponse, HttpMethod, NetworkAuth};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::time::{Duration, Instant, SystemTime};
use tokio::runtime::Runtime;
use reqwest::{Client, Method, RequestBuilder};
use url::Url;

/// HTTP client configuration and state
#[derive(Debug, Clone)]
pub struct HttpClient {
    /// Default timeout for requests
    pub default_timeout: Duration,
    /// Default retry count
    pub default_retries: u32,
    /// Whether to follow redirects
    pub follow_redirects: bool,
    /// Maximum number of redirects to follow
    pub max_redirects: u32,
    /// Default headers to include in all requests
    pub default_headers: HashMap<String, String>,
    /// Client configuration metadata
    pub metadata: HashMap<String, String>,
}

impl HttpClient {
    /// Create a new HTTP client with default configuration
    pub fn new() -> Self {
        let mut default_headers = HashMap::new();
        default_headers.insert("User-Agent".to_string(), "Lyra/1.0".to_string());
        
        Self {
            default_timeout: Duration::from_secs(30),
            default_retries: 3,
            follow_redirects: true,
            max_redirects: 10,
            default_headers,
            metadata: HashMap::new(),
        }
    }
    
    /// Execute a network request using reqwest HTTP client
    pub fn execute(&self, request: &NetworkRequest) -> Result<NetworkResponse, String> {
        let rt = Runtime::new().map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        
        rt.block_on(async { self.execute_async(request).await })
    }
    
    /// Execute a network request asynchronously
    pub async fn execute_async(&self, request: &NetworkRequest) -> Result<NetworkResponse, String> {
        let start_time = Instant::now();
        
        // Build reqwest client with configuration
        let client = Client::builder()
            .timeout(request.timeout)
            .redirect(if self.follow_redirects {
                reqwest::redirect::Policy::limited(self.max_redirects as usize)
            } else {
                reqwest::redirect::Policy::none()
            })
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;
        
        // Convert HttpMethod to reqwest::Method
        let method = match request.method {
            HttpMethod::GET => Method::GET,
            HttpMethod::POST => Method::POST,
            HttpMethod::PUT => Method::PUT,
            HttpMethod::DELETE => Method::DELETE,
            HttpMethod::PATCH => Method::PATCH,
            HttpMethod::HEAD => Method::HEAD,
            HttpMethod::OPTIONS => Method::OPTIONS,
            HttpMethod::TRACE => Method::TRACE,
        };
        
        // Parse URL
        let url = Url::parse(&request.url)
            .map_err(|e| format!("Invalid URL '{}': {}", request.url, e))?;
        
        // Build request
        let mut req_builder = client.request(method, url);
        
        // Add headers
        let effective_headers = request.effective_headers();
        for (key, value) in effective_headers {
            req_builder = req_builder.header(&key, &value);
        }
        
        // Add body if present
        if let Some(ref body) = request.body {
            req_builder = req_builder.body(body.clone());
        }
        
        // Execute request with retries
        let mut last_error = String::new();
        for attempt in 0..=request.retries {
            match req_builder.try_clone() {
                Some(cloned_builder) => {
                    match cloned_builder.send().await {
                        Ok(response) => {
                            let response_time = start_time.elapsed().as_secs_f64() * 1000.0;
                            return self.convert_response(response, response_time, &request.url).await;
                        }
                        Err(e) => {
                            last_error = format!("HTTP request failed (attempt {}): {}", attempt + 1, e);
                            if attempt < request.retries {
                                // Exponential backoff: 100ms, 200ms, 400ms, etc.
                                let backoff_ms = 100 * (1 << attempt);
                                tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                            }
                        }
                    }
                }
                None => {
                    return Err("Request body is not cloneable for retry".to_string());
                }
            }
        }
        
        Err(last_error)
    }
    
    /// Convert reqwest Response to NetworkResponse
    async fn convert_response(
        &self, 
        response: reqwest::Response, 
        response_time: f64,
        original_url: &str
    ) -> Result<NetworkResponse, String> {
        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        
        // Extract headers
        let mut headers = HashMap::new();
        for (name, value) in response.headers() {
            if let Ok(value_str) = value.to_str() {
                headers.insert(name.to_string(), value_str.to_string());
            }
        }
        
        // Read body
        let body = response.bytes().await
            .map_err(|e| format!("Failed to read response body: {}", e))?
            .to_vec();
        
        let mut network_response = NetworkResponse::new(status, headers, body);
        network_response.response_time = response_time;
        network_response.final_url = final_url;
        network_response.from_cache = false; // reqwest doesn't expose cache info easily
        
        Ok(network_response)
    }
    
    /// Execute multiple requests in parallel using async/await
    pub fn execute_parallel(&self, requests: &[NetworkRequest]) -> Vec<Result<NetworkResponse, String>> {
        let rt = Runtime::new().expect("Failed to create tokio runtime");
        
        rt.block_on(async {
            let futures: Vec<_> = requests.iter()
                .map(|req| self.execute_async(req))
                .collect();
            
            futures::future::join_all(futures).await
        })
    }
}

impl Default for HttpClient {
    fn default() -> Self {
        Self::new()
    }
}

impl Foreign for HttpClient {
    fn type_name(&self) -> &'static str {
        "HttpClient"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "DefaultTimeout" => Ok(Value::Real(self.default_timeout.as_secs_f64())),
            "DefaultRetries" => Ok(Value::Integer(self.default_retries as i64)),
            "FollowRedirects" => Ok(Value::Integer(if self.follow_redirects { 1 } else { 0 })),
            "MaxRedirects" => Ok(Value::Integer(self.max_redirects as i64)),
            "DefaultHeaders" => {
                let entries: Vec<Value> = self.default_headers.iter()
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

/// URL streaming abstraction for handling large responses
#[derive(Debug, Clone)]
pub struct URLStream {
    /// Source URL
    pub url: String,
    /// Stream buffer size
    pub buffer_size: usize,
    /// Stream position
    pub position: u64,
    /// Whether stream is open
    pub is_open: bool,
    /// Stream metadata
    pub metadata: HashMap<String, String>,
}

impl URLStream {
    /// Create a new URL stream
    pub fn new(url: String) -> Self {
        Self {
            url,
            buffer_size: 8192,
            position: 0,
            is_open: false,
            metadata: HashMap::new(),
        }
    }
    
    /// Open the stream (placeholder)
    pub fn open(&mut self) -> Result<(), String> {
        // Placeholder implementation
        self.is_open = true;
        Ok(())
    }
    
    /// Read next chunk from stream (placeholder)
    pub fn read_chunk(&mut self) -> Result<Vec<u8>, String> {
        if !self.is_open {
            return Err("Stream is not open".to_string());
        }
        
        // Placeholder: return simulated data
        let chunk = format!("Chunk {} from {}\n", self.position, self.url);
        self.position += chunk.len() as u64;
        
        // Simulate end of stream after a few chunks
        if self.position > 1000 {
            Ok(Vec::new()) // Empty chunk indicates end of stream
        } else {
            Ok(chunk.into_bytes())
        }
    }
    
    /// Close the stream
    pub fn close(&mut self) {
        self.is_open = false;
    }
}

impl Foreign for URLStream {
    fn type_name(&self) -> &'static str {
        "URLStream"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "URL" => Ok(Value::String(self.url.clone())),
            "BufferSize" => Ok(Value::Integer(self.buffer_size as i64)),
            "Position" => Ok(Value::Integer(self.position as i64)),
            "IsOpen" => Ok(Value::Integer(if self.is_open { 1 } else { 0 })),
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

/// URLRead - Fetch data from URLs
/// Syntax: URLRead[url], URLRead[request], URLRead[{url1, url2, ...}]
pub fn url_read(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (url, request, or list of urls)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let client = HttpClient::new();
    
    match &args[0] {
        // Single URL string
        Value::String(url) => {
            let request = NetworkRequest::new(url.clone(), HttpMethod::GET);
            match client.execute(&request) {
                Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                Err(e) => Err(VmError::Runtime(format!("HTTP request failed: {}", e))),
            }
        }
        
        // NetworkRequest object
        Value::LyObj(obj) => {
            if let Some(request) = obj.downcast_ref::<NetworkRequest>() {
                match client.execute(request) {
                    Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                    Err(e) => Err(VmError::Runtime(format!("HTTP request failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "NetworkRequest object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        
        // List of URLs (parallel processing)
        Value::List(urls) => {
            let mut requests = Vec::new();
            
            for url in urls {
                match url {
                    Value::String(url_str) => {
                        requests.push(NetworkRequest::new(url_str.clone(), HttpMethod::GET));
                    }
                    Value::LyObj(obj) => {
                        if let Some(request) = obj.downcast_ref::<NetworkRequest>() {
                            requests.push(request.clone());
                        } else {
                            return Err(VmError::TypeError {
                                expected: "NetworkRequest object in list".to_string(),
                                actual: "Different object type".to_string(),
                            });
                        }
                    }
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "String URL or NetworkRequest in list".to_string(),
                            actual: format!("{:?}", url),
                        });
                    }
                }
            }
            
            let results = client.execute_parallel(&requests);
            let responses: Result<Vec<Value>, VmError> = results.into_iter()
                .map(|result| match result {
                    Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                    Err(e) => Err(VmError::Runtime(format!("HTTP request failed: {}", e))),
                })
                .collect();
            
            responses.map(Value::List)
        }
        
        _ => Err(VmError::TypeError {
            expected: "String URL, NetworkRequest, or List of URLs".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// URLWrite - Send data to URLs
/// Syntax: URLWrite[url, data], URLWrite[request]
pub fn url_write(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (url or request, [data])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let client = HttpClient::new();
    
    let request = match (&args[0], args.get(1)) {
        // URL string with data
        (Value::String(url), Some(Value::String(data))) => {
            NetworkRequest::new(url.clone(), HttpMethod::POST)
                .with_body(data.as_bytes().to_vec())
                .with_header("Content-Type".to_string(), "text/plain".to_string())
        }
        
        // NetworkRequest object
        (Value::LyObj(obj), _) => {
            if let Some(request) = obj.downcast_ref::<NetworkRequest>() {
                request.clone()
            } else {
                return Err(VmError::TypeError {
                    expected: "NetworkRequest object".to_string(),
                    actual: "Different object type".to_string(),
                });
            }
        }
        
        _ => {
            return Err(VmError::TypeError {
                expected: "String URL with data or NetworkRequest".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };
    
    match client.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP request failed: {}", e))),
    }
}

/// URLStream - Create streaming connection to URL
/// Syntax: URLStream[url], URLStream[request]
pub fn url_stream(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (url or request)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let url = match &args[0] {
        Value::String(url) => url.clone(),
        Value::LyObj(obj) => {
            if let Some(request) = obj.downcast_ref::<NetworkRequest>() {
                request.url.clone()
            } else {
                return Err(VmError::TypeError {
                    expected: "NetworkRequest object".to_string(),
                    actual: "Different object type".to_string(),
                });
            }
        }
        _ => {
            return Err(VmError::TypeError {
                expected: "String URL or NetworkRequest".to_string(),
                actual: format!("{:?}", args[0]),
            });
        }
    };
    
    let stream = URLStream::new(url);
    Ok(Value::LyObj(LyObj::new(Box::new(stream))))
}

/// NetworkPing - Test network connectivity
/// Syntax: NetworkPing[host], NetworkPing[host, count]
pub fn network_ping(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (host, [count])".to_string(),
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
    
    let count = if args.len() > 1 {
        match &args[1] {
            Value::Integer(i) => *i as u32,
            _ => return Err(VmError::TypeError {
                expected: "Integer for count".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    } else {
        4
    };
    
    // Placeholder ping implementation
    let mut results = Vec::new();
    for i in 0..count {
        // Simulate ping with varying latency
        let latency = 10.0 + (i as f64 * 2.5) + (host.len() as f64 * 0.1);
        let success = !host.contains("unreachable");
        
        let result = if success {
            Value::List(vec![
                Value::Integer(i as i64 + 1),
                Value::Real(latency),
                Value::Integer(1), // success
            ])
        } else {
            Value::List(vec![
                Value::Integer(i as i64 + 1),
                Value::Real(-1.0), // timeout
                Value::Integer(0), // failure
            ])
        };
        
        results.push(result);
    }
    
    Ok(Value::List(results))
}

/// DNSResolve - Resolve hostname to IP addresses
/// Syntax: DNSResolve[hostname]
pub fn dns_resolve(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (hostname)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let hostname = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for hostname".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Placeholder DNS resolution
    let ips = if hostname == "localhost" {
        vec!["127.0.0.1", "::1"]
    } else if hostname == "example.com" {
        vec!["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]
    } else if hostname.contains("google") {
        vec!["142.250.191.14", "2607:f8b0:4004:c1b::71"]
    } else {
        // Simulate failed resolution
        vec![]
    };
    
    if ips.is_empty() {
        Err(VmError::Runtime(format!("DNS resolution failed for {}", hostname)))
    } else {
        let ip_values: Vec<Value> = ips.iter()
            .map(|ip| Value::String(ip.to_string()))
            .collect();
        Ok(Value::List(ip_values))
    }
}

/// HttpClient - Create HTTP client with configuration
/// Syntax: HttpClient[config]
pub fn http_client(args: &[Value]) -> VmResult<Value> {
    if args.len() > 1 {
        return Err(VmError::TypeError {
            expected: "0-1 arguments ([config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut client = HttpClient::new();
    
    // Apply configuration if provided
    if let Some(Value::List(config)) = args.get(0) {
        for item in config {
            if let Value::List(pair) = item {
                if pair.len() == 2 {
                    if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                        match key.as_str() {
                            "timeout" => {
                                if let Value::Real(seconds) = value {
                                    client.default_timeout = Duration::from_secs_f64(*seconds);
                                }
                            }
                            "retries" => {
                                if let Value::Integer(count) = value {
                                    client.default_retries = *count as u32;
                                }
                            }
                            "followRedirects" => {
                                if let Value::Integer(flag) = value {
                                    client.follow_redirects = *flag != 0;
                                }
                            }
                            _ => {
                                // Add to metadata for unknown config
                                if let Value::String(val) = value {
                                    client.metadata.insert(key.clone(), val.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(client))))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_http_client_creation() {
        let client = HttpClient::new();
        assert_eq!(client.default_timeout, Duration::from_secs(30));
        assert_eq!(client.default_retries, 3);
        assert!(client.follow_redirects);
        assert!(client.default_headers.contains_key("User-Agent"));
    }
    
    #[test]
    fn test_http_client_execute() {
        let client = HttpClient::new();
        let request = NetworkRequest::new(
            "https://example.com/json".to_string(),
            HttpMethod::GET
        );
        
        let response = client.execute(&request).unwrap();
        assert!(response.is_success());
        assert!(response.is_json());
        assert!(response.body_as_string().unwrap().contains("Hello from Lyra"));
    }
    
    #[test]
    fn test_url_stream_creation() {
        let stream = URLStream::new("https://example.com/stream".to_string());
        assert_eq!(stream.url, "https://example.com/stream");
        assert_eq!(stream.buffer_size, 8192);
        assert!(!stream.is_open);
    }
    
    #[test]
    fn test_parallel_requests() {
        let client = HttpClient::new();
        let requests = vec![
            NetworkRequest::new("https://example.com/1".to_string(), HttpMethod::GET),
            NetworkRequest::new("https://example.com/2".to_string(), HttpMethod::GET),
        ];
        
        let results = client.execute_parallel(&requests);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.is_ok()));
    }
}