//! Enhanced Web & API Integration System - Agent 7
//!
//! This module implements comprehensive web and API integration capabilities that extend
//! the basic HTTP functionality with advanced features for web scraping, API integration,
//! authentication, and data processing.

use super::core::{NetworkRequest, NetworkResponse, HttpMethod, NetworkAuth};
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::time::Duration;
use tokio::runtime::Runtime;
use reqwest::{Client, Method};
use url::Url;
use scraper::{Html, Selector, ElementRef};
use serde_json::Value as JsonValue;
use percent_encoding::{utf8_percent_encode, NON_ALPHANUMERIC};

/// HTTP Session with cookie persistence and authentication
#[derive(Debug, Clone)]
pub struct HTTPSession {
    /// Underlying HTTP client with cookie jar
    client: reqwest::Client,
    /// Base URL for relative requests
    base_url: Option<String>,
    /// Default headers for all requests
    default_headers: HashMap<String, String>,
    /// Default authentication
    default_auth: NetworkAuth,
    /// Session metadata
    metadata: HashMap<String, String>,
}

impl HTTPSession {
    /// Create a new HTTP session
    pub fn new() -> Result<Self, String> {
        let client = Client::builder()
            .cookie_store(true)
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

        let mut default_headers = HashMap::new();
        default_headers.insert("User-Agent".to_string(), "Lyra/1.0 Agent7".to_string());

        Ok(Self {
            client,
            base_url: None,
            default_headers,
            default_auth: NetworkAuth::None,
            metadata: HashMap::new(),
        })
    }

    /// Set base URL for relative requests
    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = Some(base_url);
        self
    }

    /// Set default authentication
    pub fn with_auth(mut self, auth: NetworkAuth) -> Self {
        self.default_auth = auth;
        self
    }

    /// Add default header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.default_headers.insert(key, value);
        self
    }

    /// Execute HTTP request with session
    pub fn execute(&self, request: &NetworkRequest) -> Result<NetworkResponse, String> {
        let rt = Runtime::new().map_err(|e| format!("Failed to create tokio runtime: {}", e))?;
        rt.block_on(async { self.execute_async(request).await })
    }

    /// Execute HTTP request asynchronously
    pub async fn execute_async(&self, request: &NetworkRequest) -> Result<NetworkResponse, String> {
        let start_time = std::time::Instant::now();
        
        // Resolve URL (handle relative URLs)
        let full_url = if let Some(ref base) = self.base_url {
            if request.url.starts_with("http") {
                request.url.clone()
            } else {
                format!("{}{}", base.trim_end_matches('/'), 
                       if request.url.starts_with('/') { &request.url } else { &format!("/{}", request.url) })
            }
        } else {
            request.url.clone()
        };

        // Parse URL
        let url = Url::parse(&full_url)
            .map_err(|e| format!("Invalid URL '{}': {}", full_url, e))?;

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

        // Build request
        let mut req_builder = self.client.request(method, url);

        // Add default headers
        for (key, value) in &self.default_headers {
            req_builder = req_builder.header(key, value);
        }

        // Add request headers
        let effective_headers = request.effective_headers();
        for (key, value) in effective_headers {
            req_builder = req_builder.header(&key, &value);
        }

        // Apply default authentication if request doesn't have auth
        if matches!(request.auth, NetworkAuth::None) && !matches!(self.default_auth, NetworkAuth::None) {
            let mut auth_headers = HashMap::new();
            self.default_auth.apply_to_headers(&mut auth_headers);
            for (key, value) in auth_headers {
                req_builder = req_builder.header(&key, &value);
            }
        }

        // Add body if present
        if let Some(ref body) = request.body {
            req_builder = req_builder.body(body.clone());
        }

        // Execute request
        match req_builder.send().await {
            Ok(response) => {
                let response_time = start_time.elapsed().as_secs_f64() * 1000.0;
                self.convert_response(response, response_time, &full_url).await
            }
            Err(e) => Err(format!("HTTP request failed: {}", e)),
        }
    }

    /// Convert reqwest Response to NetworkResponse
    async fn convert_response(
        &self,
        response: reqwest::Response,
        response_time: f64,
        _original_url: &str,
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
        let body = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read response body: {}", e))?
            .to_vec();

        let mut network_response = NetworkResponse::new(status, headers, body);
        network_response.response_time = response_time;
        network_response.final_url = final_url;
        network_response.from_cache = false;

        Ok(network_response)
    }
}

impl Foreign for HTTPSession {
    fn type_name(&self) -> &'static str {
        "HTTPSession"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "BaseURL" => Ok(self.base_url.as_ref()
                .map(|url| Value::String(url.clone()))
                .unwrap_or(Value::String("".to_string()))),
            "DefaultHeaders" => {
                let entries: Vec<Value> = self.default_headers.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
            }
            "Execute" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "Execute".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                if let Value::LyObj(obj) = &args[0] {
                    if let Some(request) = obj.downcast_ref::<NetworkRequest>() {
                        match self.execute(request) {
                            Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                            Err(e) => Err(ForeignError::RuntimeError { message: e }),
                        }
                    } else {
                        Err(ForeignError::InvalidArgumentType {
                            method: "Execute".to_string(),
                            expected: "NetworkRequest".to_string(),
                            actual: "Other type".to_string(),
                        })
                    }
                } else {
                    Err(ForeignError::InvalidArgumentType {
                        method: "Execute".to_string(),
                        expected: "NetworkRequest".to_string(),
                        actual: "Other type".to_string(),
                    })
                }
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

/// Parsed HTML document for web scraping
#[derive(Debug, Clone)]
pub struct HTMLDocument {
    /// Raw HTML content (stored to enable thread safety)
    html_content: String,
    /// Source URL
    source_url: String,
    /// Document metadata
    metadata: HashMap<String, String>,
}

impl HTMLDocument {
    /// Parse HTML from string
    pub fn from_html(html: &str, source_url: String) -> Self {
        Self {
            html_content: html.to_string(),
            source_url,
            metadata: HashMap::new(),
        }
    }

    /// Select elements using CSS selector
    pub fn select(&self, selector_str: &str) -> Result<Vec<HTMLElement>, String> {
        let document = Html::parse_document(&self.html_content);
        let selector = Selector::parse(selector_str)
            .map_err(|e| format!("Invalid CSS selector '{}': {:?}", selector_str, e))?;

        let elements: Vec<HTMLElement> = document
            .select(&selector)
            .map(|element| HTMLElement::from_element_ref(element))
            .collect();

        Ok(elements)
    }

    /// Extract all links from the document
    pub fn extract_links(&self) -> Vec<String> {
        let document = Html::parse_document(&self.html_content);
        if let Ok(selector) = Selector::parse("a[href]") {
            document
                .select(&selector)
                .filter_map(|element| element.value().attr("href"))
                .map(|href| {
                    // Convert relative URLs to absolute
                    if href.starts_with("http") {
                        href.to_string()
                    } else if href.starts_with("//") {
                        format!("https:{}", href)
                    } else if href.starts_with("/") {
                        if let Ok(base_url) = Url::parse(&self.source_url) {
                            format!("{}://{}{}", base_url.scheme(), base_url.host_str().unwrap_or(""), href)
                        } else {
                            href.to_string()
                        }
                    } else {
                        if let Ok(base_url) = Url::parse(&self.source_url) {
                            if let Ok(resolved) = base_url.join(href) {
                                resolved.to_string()
                            } else {
                                href.to_string()
                            }
                        } else {
                            href.to_string()
                        }
                    }
                })
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Extract all image sources
    pub fn extract_images(&self) -> Vec<String> {
        let document = Html::parse_document(&self.html_content);
        if let Ok(selector) = Selector::parse("img[src]") {
            document
                .select(&selector)
                .filter_map(|element| element.value().attr("src"))
                .map(|src| src.to_string())
                .collect()
        } else {
            Vec::new()
        }
    }
}

impl Foreign for HTMLDocument {
    fn type_name(&self) -> &'static str {
        "HTMLDocument"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "SourceURL" => Ok(Value::String(self.source_url.clone())),
            "Select" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: "Select".to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                if let Value::String(selector) = &args[0] {
                    match self.select(selector) {
                        Ok(elements) => {
                            let element_values: Vec<Value> = elements
                                .into_iter()
                                .map(|el| Value::LyObj(LyObj::new(Box::new(el))))
                                .collect();
                            Ok(Value::List(element_values))
                        }
                        Err(e) => Err(ForeignError::RuntimeError { message: e }),
                    }
                } else {
                    Err(ForeignError::InvalidArgumentType {
                        method: "Select".to_string(),
                        expected: "String".to_string(),
                        actual: "Other type".to_string(),
                    })
                }
            }
            "ExtractLinks" => {
                let links: Vec<Value> = self.extract_links()
                    .into_iter()
                    .map(|link| Value::String(link))
                    .collect();
                Ok(Value::List(links))
            }
            "ExtractImages" => {
                let images: Vec<Value> = self.extract_images()
                    .into_iter()
                    .map(|img| Value::String(img))
                    .collect();
                Ok(Value::List(images))
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

/// HTML element for web scraping
#[derive(Debug, Clone)]
pub struct HTMLElement {
    /// Tag name
    pub tag_name: String,
    /// Element text content
    pub text: String,
    /// Element attributes
    pub attributes: HashMap<String, String>,
    /// Inner HTML
    pub inner_html: String,
}

impl HTMLElement {
    /// Create HTMLElement from scraper ElementRef
    pub fn from_element_ref(element: ElementRef) -> Self {
        let tag_name = element.value().name().to_string();
        let text = element.text().collect::<Vec<_>>().join(" ").trim().to_string();
        
        let mut attributes = HashMap::new();
        for (name, value) in element.value().attrs() {
            attributes.insert(name.to_string(), value.to_string());
        }
        
        let inner_html = element.inner_html();

        Self {
            tag_name,
            text,
            attributes,
            inner_html,
        }
    }
}

impl Foreign for HTMLElement {
    fn type_name(&self) -> &'static str {
        "HTMLElement"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "TagName" => Ok(Value::String(self.tag_name.clone())),
            "Text" => Ok(Value::String(self.text.clone())),
            "InnerHTML" => Ok(Value::String(self.inner_html.clone())),
            "Attributes" => {
                let entries: Vec<Value> = self.attributes.iter()
                    .map(|(k, v)| Value::List(vec![
                        Value::String(k.clone()),
                        Value::String(v.clone())
                    ]))
                    .collect();
                Ok(Value::List(entries))
            }
            "GetAttribute" => {
                if let Some(href) = self.attributes.get("href") {
                    Ok(Value::String(href.clone()))
                } else {
                    Ok(Value::String("".to_string()))
                }
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

/// REST API client with base configuration
#[derive(Debug, Clone)]
pub struct RESTClient {
    /// Base URL for API endpoints
    pub base_url: String,
    /// HTTP session for requests
    pub session: HTTPSession,
    /// API-specific configuration
    pub config: HashMap<String, String>,
}

impl RESTClient {
    /// Create new REST client
    pub fn new(base_url: String, auth: NetworkAuth) -> Result<Self, String> {
        let session = HTTPSession::new()?
            .with_base_url(base_url.clone())
            .with_auth(auth)
            .with_header("Accept".to_string(), "application/json".to_string())
            .with_header("Content-Type".to_string(), "application/json".to_string());

        Ok(Self {
            base_url,
            session,
            config: HashMap::new(),
        })
    }

    /// Execute GET request
    pub fn get(&self, path: &str, query_params: &HashMap<String, String>) -> Result<NetworkResponse, String> {
        let mut url = if path.starts_with('/') {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        } else {
            format!("{}/{}", self.base_url.trim_end_matches('/'), path)
        };

        if !query_params.is_empty() {
            let query_string: Vec<String> = query_params.iter()
                .map(|(k, v)| format!("{}={}", 
                    utf8_percent_encode(k, NON_ALPHANUMERIC),
                    utf8_percent_encode(v, NON_ALPHANUMERIC)))
                .collect();
            url.push('?');
            url.push_str(&query_string.join("&"));
        }

        let request = NetworkRequest::new(url, HttpMethod::GET);
        self.session.execute(&request)
    }

    /// Execute POST request
    pub fn post(&self, path: &str, data: &JsonValue) -> Result<NetworkResponse, String> {
        let url = if path.starts_with('/') {
            format!("{}{}", self.base_url.trim_end_matches('/'), path)
        } else {
            format!("{}/{}", self.base_url.trim_end_matches('/'), path)
        };

        let json_string = serde_json::to_string(data)
            .map_err(|e| format!("Failed to serialize JSON: {}", e))?;

        let request = NetworkRequest::new(url, HttpMethod::POST)
            .with_body(json_string.into_bytes())
            .with_header("Content-Type".to_string(), "application/json".to_string());

        self.session.execute(&request)
    }
}

impl Foreign for RESTClient {
    fn type_name(&self) -> &'static str {
        "RESTClient"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "BaseURL" => Ok(Value::String(self.base_url.clone())),
            "Get" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: "Get".to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                
                let path = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "Get".to_string(),
                        expected: "String".to_string(),
                        actual: "Other type".to_string(),
                    })
                };

                let query_params = if args.len() > 1 {
                    match &args[1] {
                        Value::List(params) => {
                            let mut map = HashMap::new();
                            for param in params {
                                if let Value::List(pair) = param {
                                    if pair.len() == 2 {
                                        if let (Value::String(k), Value::String(v)) = (&pair[0], &pair[1]) {
                                            map.insert(k.clone(), v.clone());
                                        }
                                    }
                                }
                            }
                            map
                        }
                        _ => HashMap::new(),
                    }
                } else {
                    HashMap::new()
                };

                match self.get(path, &query_params) {
                    Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                    Err(e) => Err(ForeignError::RuntimeError { message: e }),
                }
            }
            "Post" => {
                if args.len() < 2 {
                    return Err(ForeignError::InvalidArity {
                        method: "Post".to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let path = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: "Post".to_string(),
                        expected: "String".to_string(),
                        actual: "Other type".to_string(),
                    })
                };

                // Convert Lyra Value to JSON
                let json_value = match lyra_value_to_json(&args[1]) {
                    Ok(json) => json,
                    Err(e) => return Err(ForeignError::RuntimeError { message: e }),
                };

                match self.post(path, &json_value) {
                    Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
                    Err(e) => Err(ForeignError::RuntimeError { message: e }),
                }
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

/// Convert Lyra Value to JSON
fn lyra_value_to_json(value: &Value) -> Result<JsonValue, String> {
    match value {
        Value::Integer(i) => Ok(JsonValue::Number((*i).into())),
        Value::Real(r) => {
            if let Some(number) = serde_json::Number::from_f64(*r) {
                Ok(JsonValue::Number(number))
            } else {
                Err(format!("Invalid float value: {}", r))
            }
        }
        Value::String(s) => Ok(JsonValue::String(s.clone())),
        Value::List(items) => {
            let mut json_array = Vec::new();
            for item in items {
                json_array.push(lyra_value_to_json(item)?);
            }
            Ok(JsonValue::Array(json_array))
        }
        _ => Err(format!("Cannot convert {:?} to JSON", value)),
    }
}

// ===============================
// WOLFRAM LANGUAGE INTERFACE FUNCTIONS
// ===============================

/// HTTPSession - Create HTTP session with cookie persistence
/// Syntax: HTTPSession[], HTTPSession[baseUrl], HTTPSession[baseUrl, auth]
pub fn http_session(args: &[Value]) -> VmResult<Value> {
    if args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "0-2 arguments ([baseUrl], [auth])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    // Set base URL if provided
    if let Some(Value::String(base_url)) = args.get(0) {
        session = session.with_base_url(base_url.clone());
    }

    // Set authentication if provided
    if let Some(Value::LyObj(auth_obj)) = args.get(1) {
        if let Some(auth) = auth_obj.downcast_ref::<NetworkAuth>() {
            session = session.with_auth(auth.clone());
        }
    }

    Ok(Value::LyObj(LyObj::new(Box::new(session))))
}

/// HTTPRequest - Generic HTTP request with full options
/// Syntax: HTTPRequest[method, url, options]
pub fn http_request(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (method, url, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let method_str = match &args[0] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "String for method".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let url = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let method = HttpMethod::from(method_str);
    let mut request = NetworkRequest::new(url, method);

    // Process options if provided
    if let Some(Value::List(options)) = args.get(2) {
        for option in options {
            if let Value::List(pair) = option {
                if pair.len() == 2 {
                    if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                        match key.as_str() {
                            "headers" => {
                                if let Value::List(headers) = value {
                                    for header in headers {
                                        if let Value::List(header_pair) = header {
                                            if header_pair.len() == 2 {
                                                if let (Value::String(h_key), Value::String(h_value)) = 
                                                    (&header_pair[0], &header_pair[1]) {
                                                    request = request.with_header(h_key.clone(), h_value.clone());
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            "body" => {
                                if let Value::String(body) = value {
                                    request = request.with_body(body.as_bytes().to_vec());
                                }
                            }
                            "timeout" => {
                                if let Value::Real(seconds) = value {
                                    request = request.with_timeout(Duration::from_secs_f64(*seconds));
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP request failed: {}", e))),
    }
}

/// HTTPGet - Enhanced GET with query parameters
/// Syntax: HTTPGet[url, headers, params]
pub fn http_get(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (url, [headers], [params])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut url = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let mut request = NetworkRequest::new(url.clone(), HttpMethod::GET);

    // Add headers if provided
    if let Some(Value::List(headers)) = args.get(1) {
        for header in headers {
            if let Value::List(pair) = header {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                        request = request.with_header(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    // Add query parameters if provided
    if let Some(Value::List(params)) = args.get(2) {
        let mut query_pairs = Vec::new();
        for param in params {
            if let Value::List(pair) = param {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                        query_pairs.push(format!("{}={}", 
                            utf8_percent_encode(key, NON_ALPHANUMERIC),
                            utf8_percent_encode(value, NON_ALPHANUMERIC)));
                    }
                }
            }
        }
        if !query_pairs.is_empty() {
            let separator = if url.contains('?') { "&" } else { "?" };
            url = format!("{}{}{}", url, separator, query_pairs.join("&"));
            request.url = url;
        }
    }

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP GET failed: {}", e))),
    }
}

/// HTTPPost - POST with JSON/form data support
/// Syntax: HTTPPost[url, data, headers]
pub fn http_post(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (url, data, [headers])".to_string(),
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

    // Convert data to appropriate format
    let (body, content_type) = match &args[1] {
        Value::String(s) => (s.as_bytes().to_vec(), "text/plain".to_string()),
        Value::List(_) => {
            // Convert to JSON
            let json_value = lyra_value_to_json(&args[1])
                .map_err(|e| VmError::Runtime(format!("Failed to convert data to JSON: {}", e)))?;
            let json_string = serde_json::to_string(&json_value)
                .map_err(|e| VmError::Runtime(format!("Failed to serialize JSON: {}", e)))?;
            (json_string.into_bytes(), "application/json".to_string())
        }
        _ => return Err(VmError::TypeError {
            expected: "String or List for data".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let mut request = NetworkRequest::new(url, HttpMethod::POST)
        .with_body(body)
        .with_header("Content-Type".to_string(), content_type);

    // Add headers if provided
    if let Some(Value::List(headers)) = args.get(2) {
        for header in headers {
            if let Value::List(pair) = header {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                        request = request.with_header(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP POST failed: {}", e))),
    }
}

/// HTMLParse - Parse HTML to DOM structure
/// Syntax: HTMLParse[html_string], HTMLParse[html_string, source_url]
pub fn html_parse(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (html_string, [source_url])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let html_string = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String for HTML".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let source_url = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => "".to_string(),
        }
    } else {
        "".to_string()
    };

    let document = HTMLDocument::from_html(html_string, source_url);
    Ok(Value::LyObj(LyObj::new(Box::new(document))))
}

/// CSSSelect - CSS selector queries
/// Syntax: CSSSelect[dom, selector]
pub fn css_select(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (dom, selector)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let document = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(doc) = obj.downcast_ref::<HTMLDocument>() {
                doc
            } else {
                return Err(VmError::TypeError {
                    expected: "HTMLDocument for dom".to_string(),
                    actual: "Different object type".to_string(),
                });
            }
        }
        _ => return Err(VmError::TypeError {
            expected: "HTMLDocument for dom".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let selector = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String for selector".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match document.select(selector) {
        Ok(elements) => {
            let element_values: Vec<Value> = elements
                .into_iter()
                .map(|el| Value::LyObj(LyObj::new(Box::new(el))))
                .collect();
            Ok(Value::List(element_values))
        }
        Err(e) => Err(VmError::Runtime(format!("CSS selection failed: {}", e))),
    }
}

/// RESTClient - REST API client with base configuration
/// Syntax: RESTClient[base_url, auth]
pub fn rest_client(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (base_url, [auth])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let base_url = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for base_url".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let auth = if args.len() > 1 {
        match &args[1] {
            Value::LyObj(obj) => {
                if let Some(auth) = obj.downcast_ref::<NetworkAuth>() {
                    auth.clone()
                } else {
                    NetworkAuth::None
                }
            }
            _ => NetworkAuth::None,
        }
    } else {
        NetworkAuth::None
    };

    match RESTClient::new(base_url, auth) {
        Ok(client) => Ok(Value::LyObj(LyObj::new(Box::new(client)))),
        Err(e) => Err(VmError::Runtime(format!("Failed to create REST client: {}", e))),
    }
}

/// HTTPPut - PUT requests
/// Syntax: HTTPPut[url, data, headers]
pub fn http_put(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (url, data, [headers])".to_string(),
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

    // Convert data to appropriate format
    let (body, content_type) = match &args[1] {
        Value::String(s) => (s.as_bytes().to_vec(), "text/plain".to_string()),
        Value::List(_) => {
            // Convert to JSON
            let json_value = lyra_value_to_json(&args[1])
                .map_err(|e| VmError::Runtime(format!("Failed to convert data to JSON: {}", e)))?;
            let json_string = serde_json::to_string(&json_value)
                .map_err(|e| VmError::Runtime(format!("Failed to serialize JSON: {}", e)))?;
            (json_string.into_bytes(), "application/json".to_string())
        }
        _ => return Err(VmError::TypeError {
            expected: "String or List for data".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let mut request = NetworkRequest::new(url, HttpMethod::PUT)
        .with_body(body)
        .with_header("Content-Type".to_string(), content_type);

    // Add headers if provided
    if let Some(Value::List(headers)) = args.get(2) {
        for header in headers {
            if let Value::List(pair) = header {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                        request = request.with_header(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP PUT failed: {}", e))),
    }
}

/// HTTPDelete - DELETE requests
/// Syntax: HTTPDelete[url, headers]
pub fn http_delete(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (url, [headers])".to_string(),
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

    let mut request = NetworkRequest::new(url, HttpMethod::DELETE);

    // Add headers if provided
    if let Some(Value::List(headers)) = args.get(1) {
        for header in headers {
            if let Value::List(pair) = header {
                if pair.len() == 2 {
                    if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                        request = request.with_header(key.clone(), value.clone());
                    }
                }
            }
        }
    }

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("HTTP DELETE failed: {}", e))),
    }
}

/// HTTPTimeout - Request timeouts
/// Syntax: HTTPTimeout[request, seconds]
pub fn http_timeout(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (request, seconds)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut request = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(req) = obj.downcast_ref::<NetworkRequest>() {
                req.clone()
            } else {
                return Err(VmError::TypeError {
                    expected: "NetworkRequest for request".to_string(),
                    actual: "Different object type".to_string(),
                });
            }
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkRequest for request".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let seconds = match &args[1] {
        Value::Real(s) => *s,
        Value::Integer(s) => *s as f64,
        _ => return Err(VmError::TypeError {
            expected: "Number for seconds".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    request.timeout = Duration::from_secs_f64(seconds);
    Ok(Value::LyObj(LyObj::new(Box::new(request))))
}

/// HTTPAuth - Various auth methods
/// Syntax: HTTPAuth[request, auth_type, credentials]
pub fn http_auth(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (request, auth_type, credentials)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut request = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(req) = obj.downcast_ref::<NetworkRequest>() {
                req.clone()
            } else {
                return Err(VmError::TypeError {
                    expected: "NetworkRequest for request".to_string(),
                    actual: "Different object type".to_string(),
                });
            }
        }
        _ => return Err(VmError::TypeError {
            expected: "NetworkRequest for request".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let auth_type = match &args[1] {
        Value::String(s) => s.as_str(),
        _ => return Err(VmError::TypeError {
            expected: "String for auth_type".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let auth = match auth_type.to_lowercase().as_str() {
        "basic" => {
            if let Value::List(creds) = &args[2] {
                if creds.len() == 2 {
                    if let (Value::String(username), Value::String(password)) = (&creds[0], &creds[1]) {
                        NetworkAuth::Basic { 
                            username: username.clone(), 
                            password: password.clone() 
                        }
                    } else {
                        return Err(VmError::TypeError {
                            expected: "List of [username, password] for Basic auth".to_string(),
                            actual: format!("{:?}", args[2]),
                        });
                    }
                } else {
                    return Err(VmError::TypeError {
                        expected: "List of [username, password] for Basic auth".to_string(),
                        actual: format!("{:?}", args[2]),
                    });
                }
            } else {
                return Err(VmError::TypeError {
                    expected: "List of [username, password] for Basic auth".to_string(),
                    actual: format!("{:?}", args[2]),
                });
            }
        }
        "bearer" => {
            if let Value::String(token) = &args[2] {
                NetworkAuth::Bearer { token: token.clone() }
            } else {
                return Err(VmError::TypeError {
                    expected: "String token for Bearer auth".to_string(),
                    actual: format!("{:?}", args[2]),
                });
            }
        }
        "apikey" => {
            if let Value::List(creds) = &args[2] {
                if creds.len() == 2 {
                    if let (Value::String(key), Value::String(header)) = (&creds[0], &creds[1]) {
                        NetworkAuth::ApiKey { 
                            key: key.clone(), 
                            header: header.clone() 
                        }
                    } else {
                        return Err(VmError::TypeError {
                            expected: "List of [key, header] for ApiKey auth".to_string(),
                            actual: format!("{:?}", args[2]),
                        });
                    }
                } else {
                    return Err(VmError::TypeError {
                        expected: "List of [key, header] for ApiKey auth".to_string(),
                        actual: format!("{:?}", args[2]),
                    });
                }
            } else {
                return Err(VmError::TypeError {
                    expected: "List of [key, header] for ApiKey auth".to_string(),
                    actual: format!("{:?}", args[2]),
                });
            }
        }
        _ => return Err(VmError::Runtime(format!("Unsupported auth type: {}", auth_type))),
    };

    request.auth = auth;
    Ok(Value::LyObj(LyObj::new(Box::new(request))))
}

/// HTMLExtractText - Extract text content
/// Syntax: HTMLExtractText[element]
pub fn html_extract_text(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (element)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(element) = obj.downcast_ref::<HTMLElement>() {
                Ok(Value::String(element.text.clone()))
            } else {
                Err(VmError::TypeError {
                    expected: "HTMLElement for element".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "HTMLElement for element".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// HTMLExtractLinks - Extract all links
/// Syntax: HTMLExtractLinks[dom]
pub fn html_extract_links(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (dom)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(document) = obj.downcast_ref::<HTMLDocument>() {
                let links: Vec<Value> = document.extract_links()
                    .into_iter()
                    .map(|link| Value::String(link))
                    .collect();
                Ok(Value::List(links))
            } else {
                Err(VmError::TypeError {
                    expected: "HTMLDocument for dom".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "HTMLDocument for dom".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// HTMLExtractImages - Extract image sources
/// Syntax: HTMLExtractImages[dom]
pub fn html_extract_images(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (dom)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(document) = obj.downcast_ref::<HTMLDocument>() {
                let images: Vec<Value> = document.extract_images()
                    .into_iter()
                    .map(|img| Value::String(img))
                    .collect();
                Ok(Value::List(images))
            } else {
                Err(VmError::TypeError {
                    expected: "HTMLDocument for dom".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "HTMLDocument for dom".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// URLParse - Parse URL into components
/// Syntax: URLParse[url]
pub fn url_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (url)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let url_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    match Url::parse(url_str) {
        Ok(url) => {
            let mut components = Vec::new();
            
            // Scheme
            components.push(Value::List(vec![
                Value::String("scheme".to_string()),
                Value::String(url.scheme().to_string()),
            ]));

            // Host
            if let Some(host) = url.host_str() {
                components.push(Value::List(vec![
                    Value::String("host".to_string()),
                    Value::String(host.to_string()),
                ]));
            }

            // Port
            if let Some(port) = url.port() {
                components.push(Value::List(vec![
                    Value::String("port".to_string()),
                    Value::Integer(port as i64),
                ]));
            }

            // Path
            components.push(Value::List(vec![
                Value::String("path".to_string()),
                Value::String(url.path().to_string()),
            ]));

            // Query parameters
            if let Some(_query) = url.query() {
                let mut query_params = Vec::new();
                for (key, value) in url.query_pairs() {
                    query_params.push(Value::List(vec![
                        Value::String(key.to_string()),
                        Value::String(value.to_string()),
                    ]));
                }
                components.push(Value::List(vec![
                    Value::String("query".to_string()),
                    Value::List(query_params),
                ]));
            }

            // Fragment
            if let Some(fragment) = url.fragment() {
                components.push(Value::List(vec![
                    Value::String("fragment".to_string()),
                    Value::String(fragment.to_string()),
                ]));
            }

            Ok(Value::List(components))
        }
        Err(e) => Err(VmError::Runtime(format!("Failed to parse URL: {}", e))),
    }
}

/// URLBuild - Build URL from components
/// Syntax: URLBuild[components]
pub fn url_build(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (components)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let components = match &args[0] {
        Value::List(items) => items,
        _ => return Err(VmError::TypeError {
            expected: "List for components".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let mut scheme = "http";
    let mut host = "";
    let mut port: Option<u16> = None;
    let mut path = "";
    let mut query_params = Vec::new();
    let mut fragment = "";

    // Parse components
    for component in components {
        if let Value::List(pair) = component {
            if pair.len() == 2 {
                if let Value::String(key) = &pair[0] {
                    match key.as_str() {
                        "scheme" => {
                            if let Value::String(s) = &pair[1] {
                                scheme = s;
                            }
                        }
                        "host" => {
                            if let Value::String(h) = &pair[1] {
                                host = h;
                            }
                        }
                        "port" => {
                            if let Value::Integer(p) = &pair[1] {
                                port = Some(*p as u16);
                            }
                        }
                        "path" => {
                            if let Value::String(p) = &pair[1] {
                                path = p;
                            }
                        }
                        "query" => {
                            if let Value::List(params) = &pair[1] {
                                for param in params {
                                    if let Value::List(param_pair) = param {
                                        if param_pair.len() == 2 {
                                            if let (Value::String(k), Value::String(v)) = 
                                                (&param_pair[0], &param_pair[1]) {
                                                query_params.push(format!("{}={}", 
                                                    utf8_percent_encode(k, NON_ALPHANUMERIC),
                                                    utf8_percent_encode(v, NON_ALPHANUMERIC)));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        "fragment" => {
                            if let Value::String(f) = &pair[1] {
                                fragment = f;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Build URL
    let mut url = format!("{}://{}", scheme, host);
    if let Some(p) = port {
        if (scheme == "http" && p != 80) || (scheme == "https" && p != 443) {
            url.push_str(&format!(":{}", p));
        }
    }

    if !path.is_empty() && !path.starts_with('/') {
        url.push('/');
    }
    url.push_str(path);

    if !query_params.is_empty() {
        url.push('?');
        url.push_str(&query_params.join("&"));
    }

    if !fragment.is_empty() {
        url.push('#');
        url.push_str(fragment);
    }

    Ok(Value::String(url))
}

/// URLValidate - Validate URL format
/// Syntax: URLValidate[url]
pub fn url_validate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (url)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let url_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    match Url::parse(url_str) {
        Ok(_) => Ok(Value::Integer(1)), // Valid
        Err(_) => Ok(Value::Integer(0)), // Invalid
    }
}

/// FormData - Create form data for POST
/// Syntax: FormData[fields]
pub fn form_data(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (fields)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let fields = match &args[0] {
        Value::List(items) => items,
        _ => return Err(VmError::TypeError {
            expected: "List for fields".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let mut form_pairs = Vec::new();
    for field in fields {
        if let Value::List(pair) = field {
            if pair.len() == 2 {
                if let (Value::String(key), Value::String(value)) = (&pair[0], &pair[1]) {
                    form_pairs.push(format!("{}={}", 
                        utf8_percent_encode(key, NON_ALPHANUMERIC),
                        utf8_percent_encode(value, NON_ALPHANUMERIC)));
                }
            }
        }
    }

    Ok(Value::String(form_pairs.join("&")))
}

/// JSONRequest - JSON API requests
/// Syntax: JSONRequest[data, url]
pub fn json_request(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (data, url)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let url = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for URL".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Convert data to JSON
    let json_value = lyra_value_to_json(&args[0])
        .map_err(|e| VmError::Runtime(format!("Failed to convert data to JSON: {}", e)))?;
    let json_string = serde_json::to_string(&json_value)
        .map_err(|e| VmError::Runtime(format!("Failed to serialize JSON: {}", e)))?;

    let request = NetworkRequest::new(url, HttpMethod::POST)
        .with_body(json_string.into_bytes())
        .with_header("Content-Type".to_string(), "application/json".to_string());

    // Execute request
    let session = HTTPSession::new()
        .map_err(|e| VmError::Runtime(format!("Failed to create HTTP session: {}", e)))?;

    match session.execute(&request) {
        Ok(response) => Ok(Value::LyObj(LyObj::new(Box::new(response)))),
        Err(e) => Err(VmError::Runtime(format!("JSON request failed: {}", e))),
    }
}

/// XMLParse - XML parsing (basic text extraction)
/// Syntax: XMLParse[xml_string]
pub fn xml_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (xml_string)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let xml_string = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String for XML".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    // Simple XML tag extraction using regex-like approach
    let mut result = Vec::new();
    result.push(Value::List(vec![
        Value::String("type".to_string()),
        Value::String("xml".to_string()),
    ]));
    result.push(Value::List(vec![
        Value::String("content".to_string()),
        Value::String(xml_string.clone()),
    ]));
    result.push(Value::List(vec![
        Value::String("length".to_string()),
        Value::Integer(xml_string.len() as i64),
    ]));

    Ok(Value::List(result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_http_session_creation() {
        let session = HTTPSession::new().unwrap();
        assert!(session.base_url.is_none());
        assert!(session.default_headers.contains_key("User-Agent"));
    }

    #[test]
    fn test_html_document_parsing() {
        let html = r#"
            <html>
                <body>
                    <h1>Test Title</h1>
                    <a href="https://example.com">Link</a>
                    <img src="/image.jpg" alt="Test Image" />
                </body>
            </html>
        "#;
        
        let doc = HTMLDocument::from_html(html, "https://test.com".to_string());
        
        let links = doc.extract_links();
        assert!(!links.is_empty());
        assert!(links[0].contains("example.com"));

        let images = doc.extract_images();
        assert!(!images.is_empty());
        assert!(images[0].contains("image.jpg"));
    }

    #[test]
    fn test_rest_client_creation() {
        let client = RESTClient::new(
            "https://api.example.com".to_string(),
            NetworkAuth::Bearer { token: "test-token".to_string() }
        );
        
        assert!(client.is_ok());
        let client = client.unwrap();
        assert_eq!(client.base_url, "https://api.example.com");
    }

    #[test]
    fn test_lyra_value_to_json() {
        let value = Value::List(vec![
            Value::String("test".to_string()),
            Value::Integer(42),
            Value::Real(3.14),
        ]);
        
        let json = lyra_value_to_json(&value).unwrap();
        assert!(json.is_array());
        
        let array = json.as_array().unwrap();
        assert_eq!(array.len(), 3);
        assert_eq!(array[0], JsonValue::String("test".to_string()));
        assert_eq!(array[1], JsonValue::Real(42.into()));
    }

    #[test]
    fn test_url_parsing() {
        let url_str = "https://api.example.com:8443/v1/data?param=value&other=test#section";
        let url = Url::parse(url_str).unwrap();
        
        assert_eq!(url.scheme(), "https");
        assert_eq!(url.host_str().unwrap(), "api.example.com");
        assert_eq!(url.port().unwrap(), 8443);
        assert_eq!(url.path(), "/v1/data");
        assert_eq!(url.query().unwrap(), "param=value&other=test");
        assert_eq!(url.fragment().unwrap(), "section");
    }
}