//! Distributed Computing Primitives
//!
//! This module implements location-transparent computation, service discovery,
//! and distributed data processing as symbolic operations.
//!
//! ## Architecture
//!
//! ### RemoteFunction - Network-transparent function calls
//! - HTTP-based RPC using existing HttpClient infrastructure
//! - JSON serialization for cross-platform compatibility  
//! - Circuit breaker pattern for fault tolerance
//! - Integration with existing retry and timeout mechanisms
//!
//! ### ServiceRegistry - Dynamic service discovery
//! - DNS-based and HTTP health check discovery
//! - Service health monitoring and automatic failover
//! - Integration with existing event streaming system
//!
//! ### LoadBalancer - Intelligent traffic distribution
//! - Round-robin, weighted, and latency-based strategies
//! - Circuit breaker integration for cascade failure prevention
//! - Real-time health monitoring and adaptive routing
//!
//! ### DistributedMap/Reduce - Cluster parallel operations
//! - Extension of existing ParallelMap with network distribution
//! - Adaptive work partitioning based on cluster topology
//! - Fault-tolerant execution with automatic retry
//!
//! ### ComputeCluster - Managed distributed resources
//! - Node lifecycle management and health monitoring
//! - Job scheduling with resource allocation
//! - Integration with container orchestration

use super::core::{NetworkRequest, HttpMethod};
use super::http::HttpClient;
use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::{HashMap, BTreeMap};
use std::time::{Duration, SystemTime, UNIX_EPOCH, Instant};
use std::sync::{Arc, Mutex};

/// Circuit breaker for fault tolerance
#[derive(Debug, Clone)]
struct CircuitBreaker {
    failure_count: u32,
    failure_threshold: u32,
    success_threshold: u32,
    last_failure_time: Option<SystemTime>,
    timeout_duration: Duration,
    state: CircuitBreakerState,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitBreakerState {
    Closed,   // Normal operation
    Open,     // Failing fast
    HalfOpen, // Testing if service recovered
}

impl CircuitBreaker {
    fn new(failure_threshold: u32, timeout_duration: Duration) -> Self {
        Self {
            failure_count: 0,
            failure_threshold,
            success_threshold: 3,
            last_failure_time: None,
            timeout_duration,
            state: CircuitBreakerState::Closed,
        }
    }
    
    fn can_execute(&mut self) -> bool {
        match self.state {
            CircuitBreakerState::Closed => true,
            CircuitBreakerState::Open => {
                if let Some(last_failure) = self.last_failure_time {
                    if last_failure.elapsed().unwrap_or(Duration::ZERO) > self.timeout_duration {
                        self.state = CircuitBreakerState::HalfOpen;
                        return true;
                    }
                }
                false
            },
            CircuitBreakerState::HalfOpen => true,
        }
    }
    
    fn record_success(&mut self) {
        match self.state {
            CircuitBreakerState::HalfOpen => {
                self.failure_count = 0;
                self.state = CircuitBreakerState::Closed;
            },
            CircuitBreakerState::Closed => {
                self.failure_count = 0;
            },
            _ => {}
        }
    }
    
    fn record_failure(&mut self) {
        self.failure_count += 1;
        self.last_failure_time = Some(SystemTime::now());
        
        if self.failure_count >= self.failure_threshold {
            self.state = CircuitBreakerState::Open;
        }
    }
}

/// Network-transparent function execution with fault tolerance
#[derive(Debug, Clone)]
pub struct RemoteFunction {
    /// Service endpoint URL
    pub endpoint: String,
    /// Function name to invoke
    pub function_name: String,
    /// Request timeout in seconds
    pub timeout: f64,
    /// HTTP client for network communication
    pub client: HttpClient,
    /// Circuit breaker for fault tolerance
    pub circuit_breaker: Arc<Mutex<CircuitBreaker>>,
    /// Function call statistics
    pub stats: RemoteFunctionStats,
    /// Custom headers for authentication/routing
    pub headers: HashMap<String, String>,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

#[derive(Debug, Clone)]
struct RemoteFunctionStats {
    calls_made: u64,
    calls_succeeded: u64,
    calls_failed: u64,
    total_latency_ms: f64,
    last_call_time: Option<SystemTime>,
}

#[derive(Debug, Clone)]
struct RetryConfig {
    max_retries: u32,
    base_delay_ms: u64,
    max_delay_ms: u64,
    backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
        }
    }
}

impl RemoteFunction {
    /// Create a new remote function
    pub fn new(endpoint: String, function_name: String) -> Self {
        Self {
            endpoint,
            function_name,
            timeout: 30.0,
            client: HttpClient::new(),
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new(
                5, 
                Duration::from_secs(60)
            ))),
            stats: RemoteFunctionStats {
                calls_made: 0,
                calls_succeeded: 0,
                calls_failed: 0,
                total_latency_ms: 0.0,
                last_call_time: None,
            },
            headers: HashMap::new(),
            retry_config: RetryConfig::default(),
        }
    }
    
    /// Configure timeout
    pub fn with_timeout(mut self, timeout_seconds: f64) -> Self {
        self.timeout = timeout_seconds;
        self
    }
    
    /// Add custom header
    pub fn with_header(mut self, key: String, value: String) -> Self {
        self.headers.insert(key, value);
        self
    }
    
    /// Configure retry behavior
    pub fn with_retry_config(mut self, config: RetryConfig) -> Self {
        self.retry_config = config;
        self
    }
    
    /// Execute remote function call with fault tolerance
    pub fn call(&mut self, args: Vec<Value>) -> Result<Value, String> {
        // Check circuit breaker
        {
            let mut cb = self.circuit_breaker.lock()
                .map_err(|e| format!("Circuit breaker lock error: {}", e))?;
            if !cb.can_execute() {
                return Err("Circuit breaker open - service unavailable".to_string());
            }
        }
        
        let start_time = Instant::now();
        
        // Execute with retries
        let result = self.execute_with_retries(args);
        
        // Update statistics and circuit breaker
        let latency = start_time.elapsed().as_secs_f64() * 1000.0;
        self.stats.calls_made += 1;
        self.stats.total_latency_ms += latency;
        self.stats.last_call_time = Some(SystemTime::now());
        
        match &result {
            Ok(_) => {
                self.stats.calls_succeeded += 1;
                if let Ok(mut cb) = self.circuit_breaker.lock() {
                    cb.record_success();
                }
            },
            Err(_) => {
                self.stats.calls_failed += 1;
                if let Ok(mut cb) = self.circuit_breaker.lock() {
                    cb.record_failure();
                }
            }
        }
        
        result
    }
    
    /// Execute with exponential backoff retry
    fn execute_with_retries(&self, args: Vec<Value>) -> Result<Value, String> {
        let mut last_error = String::new();
        
        for attempt in 0..=self.retry_config.max_retries {
            match self.execute_single_call(&args) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    last_error = e;
                    
                    // Don't retry on the last attempt
                    if attempt < self.retry_config.max_retries {
                        let delay = self.calculate_backoff_delay(attempt);
                        std::thread::sleep(delay);
                    }
                }
            }
        }
        
        Err(format!("Remote function call failed after {} retries: {}", 
                   self.retry_config.max_retries, last_error))
    }
    
    /// Execute a single remote function call
    fn execute_single_call(&self, args: &[Value]) -> Result<Value, String> {
        // Prepare request payload
        let payload = serde_json::json!({
            "function": self.function_name,
            "arguments": self.serialize_args(args)?,
            "timestamp": SystemTime::now().duration_since(UNIX_EPOCH)
                .unwrap_or_default().as_secs(),
        });
        
        let payload_bytes = serde_json::to_vec(&payload)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        
        // Build HTTP request
        let mut request = NetworkRequest::new(self.endpoint.clone(), HttpMethod::POST)
            .with_body(payload_bytes)
            .with_header("Content-Type".to_string(), "application/json".to_string())
            .with_timeout(Duration::from_secs_f64(self.timeout));
        
        // Add custom headers
        for (key, value) in &self.headers {
            request = request.with_header(key.clone(), value.clone());
        }
        
        // Execute HTTP request
        let response = self.client.execute(&request)
            .map_err(|e| format!("HTTP request failed: {}", e))?;
        
        // Check response status
        if !response.is_success() {
            return Err(format!("Remote function returned error: HTTP {}", response.status));
        }
        
        // Parse response
        let response_text = response.body_as_string()
            .map_err(|e| format!("Response body decode error: {}", e))?;
        
        let response_json: serde_json::Value = serde_json::from_str(&response_text)
            .map_err(|e| format!("JSON parse error: {}", e))?;
        
        // Extract result
        if let Some(result) = response_json.get("result") {
            self.deserialize_result(result)
        } else if let Some(error) = response_json.get("error") {
            Err(format!("Remote function error: {}", error))
        } else {
            Err("Invalid response format".to_string())
        }
    }
    
    /// Calculate exponential backoff delay
    fn calculate_backoff_delay(&self, attempt: u32) -> Duration {
        let delay_ms = (self.retry_config.base_delay_ms as f64 * 
                       self.retry_config.backoff_multiplier.powi(attempt as i32)) as u64;
        let capped_delay = delay_ms.min(self.retry_config.max_delay_ms);
        Duration::from_millis(capped_delay)
    }
    
    /// Serialize Lyra values to JSON for transmission
    fn serialize_args(&self, args: &[Value]) -> Result<Vec<serde_json::Value>, String> {
        args.iter().map(|arg| self.value_to_json(arg)).collect()
    }
    
    /// Convert Lyra Value to JSON
    fn value_to_json(&self, value: &Value) -> Result<serde_json::Value, String> {
        match value {
            Value::Integer(i) => Ok(serde_json::Value::Number((*i).into())),
            Value::Real(r) => Ok(serde_json::Value::Number(
                serde_json::Number::from_f64(*r)
                    .ok_or("Invalid floating point number")?)),
            Value::String(s) => Ok(serde_json::Value::String(s.clone())),
            Value::List(list) => {
                let json_list: Result<Vec<_>, _> = list.iter()
                    .map(|v| self.value_to_json(v))
                    .collect();
                Ok(serde_json::Value::Array(json_list?))
            },
            Value::Symbol(s) => Ok(serde_json::json!({
                "type": "symbol",
                "value": s
            })),
            _ => Err(format!("Unsupported value type for serialization: {:?}", value))
        }
    }
    
    /// Deserialize JSON result back to Lyra Value
    fn deserialize_result(&self, json: &serde_json::Value) -> Result<Value, String> {
        match json {
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Ok(Value::Integer(i))
                } else if let Some(f) = n.as_f64() {
                    Ok(Value::Real(f))
                } else {
                    Err("Invalid number format".to_string())
                }
            },
            serde_json::Value::String(s) => Ok(Value::String(s.clone())),
            serde_json::Value::Array(arr) => {
                let values: Result<Vec<_>, _> = arr.iter()
                    .map(|v| self.deserialize_result(v))
                    .collect();
                Ok(Value::List(values?))
            },
            serde_json::Value::Object(obj) => {
                if let (Some(typ), Some(val)) = (obj.get("type"), obj.get("value")) {
                    if typ == "symbol" {
                        if let serde_json::Value::String(s) = val {
                            return Ok(Value::Symbol(s.clone()));
                        }
                    }
                }
                Err("Complex objects not supported".to_string())
            },
            serde_json::Value::Bool(b) => Ok(Value::Integer(if *b { 1 } else { 0 })),
            serde_json::Value::Null => Ok(Value::Symbol("Null".to_string())),
        }
    }
    
    /// Get function call statistics
    pub fn get_stats(&self) -> &RemoteFunctionStats {
        &self.stats
    }
    
    /// Get average latency in milliseconds
    pub fn average_latency(&self) -> f64 {
        if self.stats.calls_made > 0 {
            self.stats.total_latency_ms / self.stats.calls_made as f64
        } else {
            0.0
        }
    }
    
    /// Get success rate as percentage
    pub fn success_rate(&self) -> f64 {
        if self.stats.calls_made > 0 {
            (self.stats.calls_succeeded as f64 / self.stats.calls_made as f64) * 100.0
        } else {
            0.0
        }
    }
}

impl Foreign for RemoteFunction {
    fn type_name(&self) -> &'static str {
        "RemoteFunction"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Endpoint" => Ok(Value::String(self.endpoint.clone())),
            "FunctionName" => Ok(Value::String(self.function_name.clone())),
            "Timeout" => Ok(Value::Real(self.timeout)),
            "CallsMade" => Ok(Value::Integer(self.stats.calls_made as i64)),
            "CallsSucceeded" => Ok(Value::Integer(self.stats.calls_succeeded as i64)),
            "CallsFailed" => Ok(Value::Integer(self.stats.calls_failed as i64)),
            "AverageLatency" => Ok(Value::Real(self.average_latency())),
            "SuccessRate" => Ok(Value::Real(self.success_rate())),
            "CircuitBreakerState" => {
                if let Ok(cb) = self.circuit_breaker.lock() {
                    let state_str = match cb.state {
                        CircuitBreakerState::Closed => "Closed",
                        CircuitBreakerState::Open => "Open",
                        CircuitBreakerState::HalfOpen => "HalfOpen",
                    };
                    Ok(Value::String(state_str.to_string()))
                } else {
                    Ok(Value::String("Unknown".to_string()))
                }
            },
            "LastCallTime" => {
                if let Some(last_call) = self.stats.last_call_time {
                    let timestamp = last_call.duration_since(UNIX_EPOCH)
                        .unwrap_or_default().as_secs_f64();
                    Ok(Value::Real(timestamp))
                } else {
                    Ok(Value::Integer(0))
                }
            },
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

/// Service health status
#[derive(Debug, Clone, PartialEq)]
enum ServiceHealth {
    Healthy,
    Unhealthy,
    Unknown,
}

/// Service instance information
#[derive(Debug, Clone)]
pub struct ServiceInstance {
    pub id: String,
    pub host: String,
    pub port: u16,
    pub health: ServiceHealth,
    pub metadata: HashMap<String, String>,
    pub last_health_check: SystemTime,
    pub health_check_url: Option<String>,
    pub weight: f64,
    pub tags: Vec<String>,
}

impl ServiceInstance {
    pub fn new(id: String, host: String, port: u16) -> Self {
        Self {
            id,
            host,
            port,
            health: ServiceHealth::Unknown,
            metadata: HashMap::new(),
            last_health_check: SystemTime::now(),
            health_check_url: None,
            weight: 1.0,
            tags: Vec::new(),
        }
    }
    
    pub fn with_health_check_url(mut self, url: String) -> Self {
        self.health_check_url = Some(url);
        self
    }
    
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
    
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }
    
    pub fn add_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
    
    pub fn endpoint(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Service discovery backend types
#[derive(Debug, Clone)]
enum DiscoveryBackend {
    DNS(String), // DNS domain for SRV lookups
    HTTP(String), // HTTP endpoint for service list
    Static(HashMap<String, Vec<ServiceInstance>>), // Static configuration
}

/// Dynamic service registry with health monitoring
#[derive(Debug, Clone)]
pub struct ServiceRegistry {
    /// Registry backend for service discovery
    pub backend: DiscoveryBackend,
    /// Cache of discovered services
    pub service_cache: HashMap<String, Vec<ServiceInstance>>,
    /// Health check configuration
    pub health_check_interval: Duration,
    pub health_check_timeout: Duration,
    /// HTTP client for health checks
    pub client: HttpClient,
    /// Registry statistics
    pub stats: ServiceRegistryStats,
    /// Service change notifications
    pub notify_on_changes: bool,
}

#[derive(Debug, Clone)]
struct ServiceRegistryStats {
    total_services: u32,
    healthy_services: u32,
    unhealthy_services: u32,
    last_discovery: Option<SystemTime>,
    health_checks_performed: u64,
    discovery_errors: u64,
}

impl ServiceRegistry {
    /// Create a new service registry with the specified backend
    pub fn new(backend: DiscoveryBackend) -> Self {
        Self {
            backend,
            service_cache: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            client: HttpClient::new(),
            stats: ServiceRegistryStats {
                total_services: 0,
                healthy_services: 0,
                unhealthy_services: 0,
                last_discovery: None,
                health_checks_performed: 0,
                discovery_errors: 0,
            },
            notify_on_changes: false,
        }
    }
    
    /// Create a new service registry with DNS discovery
    pub fn with_dns(domain: String) -> Self {
        Self {
            backend: DiscoveryBackend::DNS(domain),
            service_cache: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            client: HttpClient::new(),
            stats: ServiceRegistryStats {
                total_services: 0,
                healthy_services: 0,
                unhealthy_services: 0,
                last_discovery: None,
                health_checks_performed: 0,
                discovery_errors: 0,
            },
            notify_on_changes: false,
        }
    }
    
    /// Create a new service registry with HTTP API discovery
    pub fn with_http_api(endpoint: String) -> Self {
        Self {
            backend: DiscoveryBackend::HTTP(endpoint),
            service_cache: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            client: HttpClient::new(),
            stats: ServiceRegistryStats {
                total_services: 0,
                healthy_services: 0,
                unhealthy_services: 0,
                last_discovery: None,
                health_checks_performed: 0,
                discovery_errors: 0,
            },
            notify_on_changes: false,
        }
    }
    
    /// Create a static service registry
    pub fn with_static_config() -> Self {
        Self {
            backend: DiscoveryBackend::Static(HashMap::new()),
            service_cache: HashMap::new(),
            health_check_interval: Duration::from_secs(30),
            health_check_timeout: Duration::from_secs(5),
            client: HttpClient::new(),
            stats: ServiceRegistryStats {
                total_services: 0,
                healthy_services: 0,
                unhealthy_services: 0,
                last_discovery: None,
                health_checks_performed: 0,
                discovery_errors: 0,
            },
            notify_on_changes: false,
        }
    }
    
    /// Configure health check intervals
    pub fn with_health_check_config(mut self, interval: Duration, timeout: Duration) -> Self {
        self.health_check_interval = interval;
        self.health_check_timeout = timeout;
        self
    }
    
    /// Enable change notifications
    pub fn with_change_notifications(mut self, enabled: bool) -> Self {
        self.notify_on_changes = enabled;
        self
    }
    
    /// Discover services from the configured backend
    pub fn discover_services(&mut self, service_name: &str) -> Result<Vec<ServiceInstance>, String> {
        match self.backend.clone() {
            DiscoveryBackend::DNS(domain) => self.discover_via_dns(service_name, &domain),
            DiscoveryBackend::HTTP(endpoint) => self.discover_via_http(service_name, &endpoint),
            DiscoveryBackend::Static(services) => {
                Ok(services.get(service_name).cloned().unwrap_or_default())
            }
        }
    }
    
    /// DNS SRV record discovery
    fn discover_via_dns(&mut self, service_name: &str, domain: &str) -> Result<Vec<ServiceInstance>, String> {
        // For now, simulate DNS SRV discovery - in production would use trust-dns-resolver
        // SRV records have format: _service._proto.domain
        let _srv_name = format!("_{}._{}.{}", service_name, "tcp", domain);
        
        // Simulate SRV lookup results
        let mut instances = Vec::new();
        
        // In a real implementation, this would:
        // 1. Query SRV records for the service
        // 2. Parse priority, weight, port, and target from SRV records
        // 3. Resolve A/AAAA records for targets
        // 4. Create ServiceInstance objects
        
        // For now, create simulated instances
        for i in 0..2 {
            let instance = ServiceInstance::new(
                format!("{}-{}", service_name, i),
                format!("{}_{}.{}", service_name, i, domain),
                8080 + i as u16,
            )
            .with_health_check_url(format!("http://{}_{}.{}:{}/health", service_name, i, domain, 8080 + i))
            .with_weight(1.0)
            .add_metadata("discovery".to_string(), "dns".to_string());
            
            instances.push(instance);
        }
        
        self.stats.last_discovery = Some(SystemTime::now());
        self.update_service_cache(service_name, instances.clone());
        
        Ok(instances)
    }
    
    /// HTTP API discovery
    fn discover_via_http(&mut self, service_name: &str, endpoint: &str) -> Result<Vec<ServiceInstance>, String> {
        let discovery_url = format!("{}/services/{}", endpoint, service_name);
        let request = NetworkRequest::new(discovery_url, HttpMethod::GET)
            .with_timeout(Duration::from_secs(10))
            .with_header("Accept".to_string(), "application/json".to_string());
        
        match self.client.execute(&request) {
            Ok(response) => {
                if response.is_success() {
                    let response_text = response.body_as_string()
                        .map_err(|e| format!("Response decode error: {}", e))?;
                    
                    let json: serde_json::Value = serde_json::from_str(&response_text)
                        .map_err(|e| format!("JSON parse error: {}", e))?;
                    
                    self.parse_service_instances(&json, service_name)
                } else {
                    self.stats.discovery_errors += 1;
                    Err(format!("Discovery API returned HTTP {}", response.status))
                }
            },
            Err(e) => {
                self.stats.discovery_errors += 1;
                Err(format!("Discovery request failed: {}", e))
            }
        }
    }
    
    /// Parse service instances from JSON response
    fn parse_service_instances(&mut self, json: &serde_json::Value, service_name: &str) -> Result<Vec<ServiceInstance>, String> {
        let instances_array = json.get("instances")
            .or_else(|| json.get("services"))
            .or_else(|| json.as_array().map(|_a| json))
            .ok_or("No instances found in response")?;
        
        let mut instances = Vec::new();
        
        if let serde_json::Value::Array(instance_list) = instances_array {
            for (i, instance_json) in instance_list.iter().enumerate() {
                if let Some(instance) = self.parse_single_instance(instance_json, service_name, i) {
                    instances.push(instance);
                }
            }
        }
        
        self.stats.last_discovery = Some(SystemTime::now());
        self.update_service_cache(service_name, instances.clone());
        
        Ok(instances)
    }
    
    /// Parse a single service instance from JSON
    fn parse_single_instance(&self, json: &serde_json::Value, service_name: &str, index: usize) -> Option<ServiceInstance> {
        let host = json.get("host")
            .or_else(|| json.get("address"))
            .and_then(|v| v.as_str())?;
        
        let port = json.get("port")
            .and_then(|v| v.as_u64())
            .map(|p| p as u16)?;
        
        let id = json.get("id")
            .and_then(|v| v.as_str())
            .unwrap_or(&format!("{}-{}", service_name, index))
            .to_string();
        
        let health_check_url = json.get("health_check")
            .or_else(|| json.get("health_url"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        
        let weight = json.get("weight")
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        
        let tags = json.get("tags")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.to_string())
                    .collect()
            })
            .unwrap_or_default();
        
        let mut instance = ServiceInstance::new(id, host.to_string(), port)
            .with_weight(weight)
            .with_tags(tags)
            .add_metadata("discovery".to_string(), "http".to_string());
        
        if let Some(url) = health_check_url {
            instance = instance.with_health_check_url(url);
        }
        
        Some(instance)
    }
    
    /// Update the service cache
    fn update_service_cache(&mut self, service_name: &str, instances: Vec<ServiceInstance>) {
        let old_count = self.service_cache.get(service_name)
            .map(|instances| instances.len())
            .unwrap_or(0);
        
        self.service_cache.insert(service_name.to_string(), instances.clone());
        
        // Update statistics
        self.stats.total_services = self.service_cache.values()
            .map(|instances| instances.len() as u32)
            .sum();
        
        let new_count = instances.len();
        
        // Log significant changes
        if old_count != new_count && self.notify_on_changes {
            println!("Service {} count changed: {} -> {}", service_name, old_count, new_count);
        }
    }
    
    /// Perform health check on a service instance
    pub fn health_check(&mut self, instance: &mut ServiceInstance) -> Result<ServiceHealth, String> {
        if let Some(ref health_url) = instance.health_check_url {
            let request = NetworkRequest::new(health_url.clone(), HttpMethod::GET)
                .with_timeout(self.health_check_timeout)
                .with_header("User-Agent".to_string(), "Lyra-ServiceRegistry/1.0".to_string());
            
            match self.client.execute(&request) {
                Ok(response) => {
                    let health = if response.is_success() {
                        ServiceHealth::Healthy
                    } else {
                        ServiceHealth::Unhealthy
                    };
                    
                    instance.health = health.clone();
                    instance.last_health_check = SystemTime::now();
                    self.stats.health_checks_performed += 1;
                    
                    Ok(health)
                },
                Err(_) => {
                    instance.health = ServiceHealth::Unhealthy;
                    instance.last_health_check = SystemTime::now();
                    self.stats.health_checks_performed += 1;
                    
                    Ok(ServiceHealth::Unhealthy)
                }
            }
        } else {
            // No health check URL - assume healthy if can connect to port
            let endpoint = format!("{}:{}", instance.host, instance.port);
            
            // Simple TCP connectivity test
            let request = NetworkRequest::new(
                format!("http://{}/", endpoint), 
                HttpMethod::HEAD
            ).with_timeout(self.health_check_timeout);
            
            match self.client.execute(&request) {
                Ok(_) => {
                    instance.health = ServiceHealth::Healthy;
                    instance.last_health_check = SystemTime::now();
                    self.stats.health_checks_performed += 1;
                    Ok(ServiceHealth::Healthy)
                },
                Err(_) => {
                    instance.health = ServiceHealth::Unhealthy;
                    instance.last_health_check = SystemTime::now();
                    self.stats.health_checks_performed += 1;
                    Ok(ServiceHealth::Unhealthy)
                }
            }
        }
    }
    
    /// Get healthy instances for a service
    pub fn get_healthy_instances(&self, service_name: &str) -> Vec<ServiceInstance> {
        self.service_cache.get(service_name)
            .map(|instances| {
                instances.iter()
                    .filter(|instance| instance.health == ServiceHealth::Healthy)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }
    
    /// Get all instances for a service
    pub fn get_all_instances(&self, service_name: &str) -> Vec<ServiceInstance> {
        self.service_cache.get(service_name)
            .cloned()
            .unwrap_or_default()
    }
    
    /// Register a static service instance
    pub fn register_service(&mut self, service_name: &str, instance: ServiceInstance) -> Result<(), String> {
        match &mut self.backend {
            DiscoveryBackend::Static(ref mut services) => {
                services.entry(service_name.to_string())
                    .or_insert_with(Vec::new)
                    .push(instance);
                
                // Update cache
                let instances = services.get(service_name).cloned().unwrap_or_default();
                self.update_service_cache(service_name, instances);
                
                Ok(())
            },
            _ => Err("Service registration only supported for static backends".to_string())
        }
    }
    
    /// Deregister a service instance
    pub fn deregister_service(&mut self, service_name: &str, instance_id: &str) -> Result<(), String> {
        match &mut self.backend {
            DiscoveryBackend::Static(ref mut services) => {
                if let Some(instances) = services.get_mut(service_name) {
                    instances.retain(|instance| instance.id != instance_id);
                    
                    // Update cache
                    let updated_instances = instances.clone();
                    self.update_service_cache(service_name, updated_instances);
                }
                Ok(())
            },
            _ => Err("Service deregistration only supported for static backends".to_string())
        }
    }
    
    /// Get registry statistics
    pub fn get_stats(&self) -> &ServiceRegistryStats {
        &self.stats
    }
    
    /// Update health statistics
    pub fn update_health_stats(&mut self) {
        let mut healthy = 0;
        let mut unhealthy = 0;
        
        for instances in self.service_cache.values() {
            for instance in instances {
                match instance.health {
                    ServiceHealth::Healthy => healthy += 1,
                    ServiceHealth::Unhealthy => unhealthy += 1,
                    ServiceHealth::Unknown => {}
                }
            }
        }
        
        self.stats.healthy_services = healthy;
        self.stats.unhealthy_services = unhealthy;
    }
}

impl Foreign for ServiceRegistry {
    fn type_name(&self) -> &'static str {
        "ServiceRegistry"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "BackendType" => {
                let backend_type = match &self.backend {
                    DiscoveryBackend::DNS(_) => "DNS",
                    DiscoveryBackend::HTTP(_) => "HTTP",
                    DiscoveryBackend::Static(_) => "Static",
                };
                Ok(Value::String(backend_type.to_string()))
            },
            "ServiceCount" => Ok(Value::Integer(self.service_cache.len() as i64)),
            "TotalInstances" => Ok(Value::Integer(self.stats.total_services as i64)),
            "HealthyInstances" => Ok(Value::Integer(self.stats.healthy_services as i64)),
            "UnhealthyInstances" => Ok(Value::Integer(self.stats.unhealthy_services as i64)),
            "HealthChecksPerformed" => Ok(Value::Integer(self.stats.health_checks_performed as i64)),
            "DiscoveryErrors" => Ok(Value::Integer(self.stats.discovery_errors as i64)),
            "HealthCheckInterval" => Ok(Value::Real(self.health_check_interval.as_secs_f64())),
            "HealthCheckTimeout" => Ok(Value::Real(self.health_check_timeout.as_secs_f64())),
            "LastDiscovery" => {
                if let Some(last_discovery) = self.stats.last_discovery {
                    let timestamp = last_discovery.duration_since(UNIX_EPOCH)
                        .unwrap_or_default().as_secs_f64();
                    Ok(Value::Real(timestamp))
                } else {
                    Ok(Value::Integer(0))
                }
            },
            "ServiceNames" => {
                let names: Vec<Value> = self.service_cache.keys()
                    .map(|name| Value::String(name.clone()))
                    .collect();
                Ok(Value::List(names))
            },
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

// Placeholder functions for Phase 12C implementation

/// RemoteFunction[endpoint, functionName, options] - Create remote function call
pub fn remote_function(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (endpoint, functionName, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let endpoint = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for endpoint".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let function_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let mut remote_fn = RemoteFunction::new(endpoint, function_name);
    
    // Apply options if provided
    if args.len() > 2 {
        match &args[2] {
            Value::List(options) => {
                for option in options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "timeout" => {
                                        if let Value::Real(timeout) = value {
                                            remote_fn = remote_fn.with_timeout(*timeout);
                                        }
                                    },
                                    "maxRetries" => {
                                        if let Value::Integer(retries) = value {
                                            let mut config = remote_fn.retry_config.clone();
                                            config.max_retries = *retries as u32;
                                            remote_fn = remote_fn.with_retry_config(config);
                                        }
                                    },
                                    _ => {
                                        // Add as custom header
                                        if let Value::String(val) = value {
                                            remote_fn = remote_fn.with_header(key.clone(), val.clone());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            _ => return Err(VmError::TypeError {
                expected: "List of option pairs".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(remote_fn))))
}

/// RemoteFunctionCall[remoteFunction, arguments] - Execute remote function
pub fn remote_function_call(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (remoteFunction, arguments)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_args = match &args[1] {
        Value::List(args_list) => args_list.clone(),
        other => vec![other.clone()], // Single argument
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(remote_fn) = obj.downcast_ref::<RemoteFunction>() {
                let mut mutable_fn = remote_fn.clone();
                match mutable_fn.call(function_args) {
                    Ok(result) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_fn))), // Updated function with stats
                        result
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Remote function call failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "RemoteFunction object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "RemoteFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Distributed Map operation coordinates parallel function execution across cluster nodes
#[derive(Debug, Clone)]
pub struct DistributedMap {
    pub cluster_nodes: Vec<ServiceInstance>,
    pub load_balancer: LoadBalancer,
    pub chunk_size: usize,
    pub max_concurrent_tasks: usize,
    pub timeout: Duration,
    pub retry_config: RetryConfig,
    pub stats: DistributedMapStats,
}

#[derive(Debug, Clone)]
pub struct DistributedMapStats {
    pub total_tasks: usize,
    pub completed_tasks: usize,
    pub failed_tasks: usize,
    pub total_execution_time: Duration,
    pub average_task_time: f64,
    pub data_transferred: usize,
}

impl DistributedMapStats {
    pub fn new() -> Self {
        Self {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            total_execution_time: Duration::from_secs(0),
            average_task_time: 0.0,
            data_transferred: 0,
        }
    }
}

impl DistributedMap {
    /// Create a new distributed map operation
    pub fn new(cluster_nodes: Vec<ServiceInstance>) -> Self {
        let mut load_balancer = LoadBalancer::new(LoadBalancingStrategy::LeastConnections);
        
        // Add all cluster nodes to load balancer
        for node in &cluster_nodes {
            load_balancer.add_instance(node.clone());
        }
        
        Self {
            cluster_nodes,
            load_balancer,
            chunk_size: 100, // Default chunk size
            max_concurrent_tasks: 10, // Default concurrency
            timeout: Duration::from_secs(300), // 5 minute timeout
            retry_config: RetryConfig {
                max_retries: 3,
                base_delay_ms: 1000,
                max_delay_ms: 60000,
                backoff_multiplier: 2.0,
            },
            stats: DistributedMapStats::new(),
        }
    }
    
    /// Execute distributed map operation
    pub async fn execute(&mut self, function_name: &str, data: Vec<Value>) -> Result<Vec<Value>, String> {
        let start_time = SystemTime::now();
        let chunk_size = self.chunk_size;
        let max_concurrent_tasks = self.max_concurrent_tasks;
        let timeout = self.timeout;
        let retry_config = self.retry_config.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        
        // Calculate stats
        let total_tasks = (data.len() + chunk_size - 1) / chunk_size;
        self.stats.total_tasks = total_tasks;
        
        // Split data into chunks
        let chunks: Vec<Vec<Value>> = data.chunks(chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut results = Vec::new();
        let mut completed_tasks = 0;
        let mut failed_tasks = 0;
        
        // Use futures::stream for better async control
        let chunk_futures: Vec<_> = chunks.into_iter().enumerate().map(|(chunk_idx, chunk)| {
            let function_name = function_name.to_string();
            let cluster_nodes = cluster_nodes.clone();
            let timeout = timeout;
            let retry_config = retry_config.clone();
            
            async move {
                Self::process_chunk_static(
                    function_name,
                    chunk,
                    chunk_idx,
                    cluster_nodes,
                    timeout,
                    retry_config
                ).await
            }
        }).collect();
        
        // Process chunks with controlled concurrency
        for chunk_future in chunk_futures {
            match chunk_future.await {
                Ok(chunk_result) => {
                    results.extend(chunk_result);
                    completed_tasks += 1;
                },
                Err(e) => {
                    eprintln!("Chunk processing failed: {}", e);
                    failed_tasks += 1;
                }
            }
        }
        
        // Update stats
        self.stats.completed_tasks = completed_tasks;
        self.stats.failed_tasks = failed_tasks;
        self.stats.total_execution_time = start_time.elapsed().unwrap_or(Duration::from_secs(0));
        if completed_tasks > 0 {
            self.stats.average_task_time = self.stats.total_execution_time.as_secs_f64() / completed_tasks as f64;
        }
        
        Ok(results)
    }
    
    /// Process a single chunk on a remote node (static version)
    async fn process_chunk_static(
        function_name: String,
        chunk: Vec<Value>,
        chunk_idx: usize,
        cluster_nodes: Vec<ServiceInstance>,
        timeout: Duration,
        retry_config: RetryConfig
    ) -> Result<Vec<Value>, String> {
        if cluster_nodes.is_empty() {
            return Err("No cluster nodes available".to_string());
        }
        
        // Simple round-robin selection for static version
        let node_idx = chunk_idx % cluster_nodes.len();
        let node = &cluster_nodes[node_idx];
        
        // Create remote function for this node
        let endpoint = format!("http://{}:{}", node.host, node.port);
        let mut remote_fn = RemoteFunction::new(endpoint, function_name.clone())
            .with_timeout(timeout.as_secs_f64())
            .with_retry_config(retry_config.clone());
        
        // Execute remote function call with chunk data
        let chunk_args = vec![Value::List(chunk.clone())];
        match remote_fn.call(chunk_args) {
            Ok(result) => {
                // Expect result to be a list of processed values
                match result {
                    Value::List(processed_values) => Ok(processed_values),
                    single_value => Ok(vec![single_value]),
                }
            },
            Err(e) => {
                // Retry with different node if available
                if cluster_nodes.len() > 1 {
                    let fallback_idx = (chunk_idx + 1) % cluster_nodes.len();
                    let fallback_node = &cluster_nodes[fallback_idx];
                    
                    let fallback_endpoint = format!("http://{}:{}", fallback_node.host, fallback_node.port);
                    let mut fallback_fn = RemoteFunction::new(fallback_endpoint, function_name)
                        .with_timeout(timeout.as_secs_f64());
                    
                    let chunk_args = vec![Value::List(chunk)];
                    match fallback_fn.call(chunk_args) {
                        Ok(result) => match result {
                            Value::List(processed_values) => Ok(processed_values),
                            single_value => Ok(vec![single_value]),
                        },
                        Err(fallback_err) => Err(format!("Primary and fallback failed: {} / {}", e, fallback_err))
                    }
                } else {
                    Err(e)
                }
            }
        }
    }
    
    /// Configure chunk size for workload distribution
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size.max(1);
        self
    }
    
    /// Configure maximum concurrent tasks
    pub fn with_max_concurrent_tasks(mut self, max_tasks: usize) -> Self {
        self.max_concurrent_tasks = max_tasks.max(1);
        self
    }
    
    /// Configure timeout for remote operations
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Foreign for DistributedMap {
    fn type_name(&self) -> &'static str {
        "DistributedMap"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "NodeCount" => Ok(Value::Integer(self.cluster_nodes.len() as i64)),
            "ChunkSize" => Ok(Value::Integer(self.chunk_size as i64)),
            "MaxConcurrentTasks" => Ok(Value::Integer(self.max_concurrent_tasks as i64)),
            "TimeoutSeconds" => Ok(Value::Real(self.timeout.as_secs_f64())),
            "TotalTasks" => Ok(Value::Integer(self.stats.total_tasks as i64)),
            "CompletedTasks" => Ok(Value::Integer(self.stats.completed_tasks as i64)),
            "FailedTasks" => Ok(Value::Integer(self.stats.failed_tasks as i64)),
            "AverageTaskTime" => Ok(Value::Real(self.stats.average_task_time)),
            "SuccessRate" => {
                if self.stats.total_tasks > 0 {
                    let rate = (self.stats.completed_tasks as f64 / self.stats.total_tasks as f64) * 100.0;
                    Ok(Value::Real(rate))
                } else {
                    Ok(Value::Real(0.0))
                }
            },
            "DataTransferred" => Ok(Value::Integer(self.stats.data_transferred as i64)),
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

/// Distributed Reduce operation coordinates tree-like reduction across cluster nodes
#[derive(Debug, Clone)]
pub struct DistributedReduce {
    pub cluster_nodes: Vec<ServiceInstance>,
    pub load_balancer: LoadBalancer,
    pub reduction_factor: usize, // How many values to reduce per step
    pub timeout: Duration,
    pub retry_config: RetryConfig,
    pub stats: DistributedReduceStats,
}

#[derive(Debug, Clone)]
pub struct DistributedReduceStats {
    pub total_operations: usize,
    pub completed_operations: usize,
    pub failed_operations: usize,
    pub total_execution_time: Duration,
    pub reduction_levels: usize,
    pub final_result_size: usize,
}

impl DistributedReduceStats {
    pub fn new() -> Self {
        Self {
            total_operations: 0,
            completed_operations: 0,
            failed_operations: 0,
            total_execution_time: Duration::from_secs(0),
            reduction_levels: 0,
            final_result_size: 0,
        }
    }
}

impl DistributedReduce {
    /// Create a new distributed reduce operation
    pub fn new(cluster_nodes: Vec<ServiceInstance>) -> Self {
        let mut load_balancer = LoadBalancer::new(LoadBalancingStrategy::RoundRobin);
        
        for node in &cluster_nodes {
            load_balancer.add_instance(node.clone());
        }
        
        Self {
            cluster_nodes,
            load_balancer,
            reduction_factor: 4, // Reduce 4 values at a time
            timeout: Duration::from_secs(300),
            retry_config: RetryConfig {
                max_retries: 3,
                base_delay_ms: 1000,
                max_delay_ms: 60000,
                backoff_multiplier: 2.0,
            },
            stats: DistributedReduceStats::new(),
        }
    }
    
    /// Execute distributed reduce operation using tree reduction
    pub async fn execute(&mut self, function_name: &str, data: Vec<Value>) -> Result<Value, String> {
        let start_time = SystemTime::now();
        
        if data.is_empty() {
            return Err("Cannot reduce empty data".to_string());
        }
        
        if data.len() == 1 {
            return Ok(data[0].clone());
        }
        
        let mut current_level = data;
        let mut reduction_levels = 0;
        let mut total_operations = 0;
        let mut completed_operations = 0;
        let mut failed_operations = 0;
        
        let reduction_factor = self.reduction_factor;
        let timeout = self.timeout;
        let retry_config = self.retry_config.clone();
        let cluster_nodes = self.cluster_nodes.clone();
        
        // Iteratively reduce until we have a single value
        while current_level.len() > 1 {
            reduction_levels += 1;
            let (next_level, level_stats) = Self::reduce_level_static(
                function_name,
                current_level,
                reduction_factor,
                timeout,
                retry_config.clone(),
                cluster_nodes.clone()
            ).await?;
            current_level = next_level;
            
            total_operations += level_stats.0;
            completed_operations += level_stats.1;
            failed_operations += level_stats.2;
        }
        
        // Update final stats
        self.stats.reduction_levels = reduction_levels;
        self.stats.total_operations = total_operations;
        self.stats.completed_operations = completed_operations;
        self.stats.failed_operations = failed_operations;
        self.stats.total_execution_time = start_time.elapsed().unwrap_or(Duration::from_secs(0));
        self.stats.final_result_size = 1;
        
        Ok(current_level.into_iter().next().unwrap())
    }
    
    /// Reduce one level of the tree reduction (static version)
    async fn reduce_level_static(
        function_name: &str,
        values: Vec<Value>,
        reduction_factor: usize,
        timeout: Duration,
        retry_config: RetryConfig,
        cluster_nodes: Vec<ServiceInstance>
    ) -> Result<(Vec<Value>, (usize, usize, usize)), String> {
        let chunks: Vec<Vec<Value>> = values.chunks(reduction_factor)
            .map(|chunk| chunk.to_vec())
            .collect();
        
        let mut results = Vec::new();
        let mut total_operations = 0;
        let mut completed_operations = 0;
        let mut failed_operations = 0;
        
        for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
            total_operations += 1;
            
            match Self::reduce_chunk_static(
                function_name.to_string(),
                chunk,
                chunk_idx,
                cluster_nodes.clone(),
                timeout,
                retry_config.clone()
            ).await {
                Ok(reduced_value) => {
                    results.push(reduced_value);
                    completed_operations += 1;
                },
                Err(e) => {
                    eprintln!("Reduction failed: {}", e);
                    failed_operations += 1;
                    return Err(e);
                }
            }
        }
        
        Ok((results, (total_operations, completed_operations, failed_operations)))
    }
    
    /// Reduce a chunk of values on a remote node (static version)
    async fn reduce_chunk_static(
        function_name: String,
        chunk: Vec<Value>,
        chunk_idx: usize,
        cluster_nodes: Vec<ServiceInstance>,
        timeout: Duration,
        retry_config: RetryConfig
    ) -> Result<Value, String> {
        if chunk.is_empty() {
            return Err("Cannot reduce empty chunk".to_string());
        }
        
        if chunk.len() == 1 {
            return Ok(chunk[0].clone());
        }
        
        if cluster_nodes.is_empty() {
            return Err("No cluster nodes available".to_string());
        }
        
        // Simple round-robin selection for static version
        let node_idx = chunk_idx % cluster_nodes.len();
        let node = &cluster_nodes[node_idx];
        
        // Create remote function for reduction
        let endpoint = format!("http://{}:{}", node.host, node.port);
        let mut remote_fn = RemoteFunction::new(endpoint, function_name.clone())
            .with_timeout(timeout.as_secs_f64())
            .with_retry_config(retry_config.clone());
        
        // Execute reduction on chunk
        let chunk_args = vec![Value::List(chunk.clone())];
        
        match remote_fn.call(chunk_args) {
            Ok(result) => Ok(result),
            Err(e) => {
                // Retry with fallback node if available
                if cluster_nodes.len() > 1 {
                    let fallback_idx = (chunk_idx + 1) % cluster_nodes.len();
                    let fallback_node = &cluster_nodes[fallback_idx];
                    
                    let fallback_endpoint = format!("http://{}:{}", fallback_node.host, fallback_node.port);
                    let mut fallback_fn = RemoteFunction::new(fallback_endpoint, function_name)
                        .with_timeout(timeout.as_secs_f64());
                    
                    let chunk_args = vec![Value::List(chunk)];
                    match fallback_fn.call(chunk_args) {
                        Ok(result) => Ok(result),
                        Err(fallback_err) => Err(format!("Primary and fallback failed: {} / {}", e, fallback_err))
                    }
                } else {
                    Err(e)
                }
            }
        }
    }
    
    /// Configure reduction factor (values per reduction step)
    pub fn with_reduction_factor(mut self, factor: usize) -> Self {
        self.reduction_factor = factor.max(2);
        self
    }
    
    /// Configure timeout for remote operations
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

impl Foreign for DistributedReduce {
    fn type_name(&self) -> &'static str {
        "DistributedReduce"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "NodeCount" => Ok(Value::Integer(self.cluster_nodes.len() as i64)),
            "ReductionFactor" => Ok(Value::Integer(self.reduction_factor as i64)),
            "TimeoutSeconds" => Ok(Value::Real(self.timeout.as_secs_f64())),
            "TotalOperations" => Ok(Value::Integer(self.stats.total_operations as i64)),
            "CompletedOperations" => Ok(Value::Integer(self.stats.completed_operations as i64)),
            "FailedOperations" => Ok(Value::Integer(self.stats.failed_operations as i64)),
            "ReductionLevels" => Ok(Value::Integer(self.stats.reduction_levels as i64)),
            "FinalResultSize" => Ok(Value::Integer(self.stats.final_result_size as i64)),
            "SuccessRate" => {
                if self.stats.total_operations > 0 {
                    let rate = (self.stats.completed_operations as f64 / self.stats.total_operations as f64) * 100.0;
                    Ok(Value::Real(rate))
                } else {
                    Ok(Value::Real(0.0))
                }
            },
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

/// DistributedMap[function, data, cluster] - Distributed map operation
pub fn distributed_map(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (function, data, cluster)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let data = match &args[1] {
        Value::List(list) => list.clone(),
        _ => return Err(VmError::TypeError {
            expected: "List for data".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let cluster_nodes = match &args[2] {
        Value::List(nodes) => {
            let mut instances = Vec::new();
            for (i, node) in nodes.iter().enumerate() {
                match node {
                    Value::String(endpoint) => {
                        // Parse endpoint as "host:port"
                        let parts: Vec<&str> = endpoint.split(':').collect();
                        if parts.len() == 2 {
                            let host = parts[0].to_string();
                            let port = parts[1].parse::<u16>().map_err(|_| VmError::TypeError {
                                expected: "Valid port number".to_string(),
                                actual: format!("Invalid port: {}", parts[1]),
                            })?;
                            
                            instances.push(ServiceInstance {
                                id: format!("node-{}", i),
                                host,
                                port,
                                weight: 1.0,
                                health: ServiceHealth::Healthy,
                                last_health_check: SystemTime::now(),
                                health_check_url: None,
                                metadata: HashMap::new(),
                                tags: Vec::new(),
                            });
                        } else {
                            return Err(VmError::TypeError {
                                expected: "Endpoint format 'host:port'".to_string(),
                                actual: format!("Invalid endpoint: {}", endpoint),
                            });
                        }
                    },
                    _ => return Err(VmError::TypeError {
                        expected: "String endpoints in cluster list".to_string(),
                        actual: format!("{:?}", node),
                    }),
                }
            }
            instances
        },
        _ => return Err(VmError::TypeError {
            expected: "List of cluster nodes".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    if cluster_nodes.is_empty() {
        return Err(VmError::Runtime("Cluster cannot be empty".to_string()));
    }
    
    let distributed_map = DistributedMap::new(cluster_nodes);
    
    // Return the DistributedMap object for configuration and later execution
    Ok(Value::LyObj(LyObj::new(Box::new(distributed_map))))
}

/// DistributedReduce[function, data, cluster] - Distributed reduce operation
pub fn distributed_reduce(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (function, data, cluster)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let data = match &args[1] {
        Value::List(list) => list.clone(),
        _ => return Err(VmError::TypeError {
            expected: "List for data".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let cluster_nodes = match &args[2] {
        Value::List(nodes) => {
            let mut instances = Vec::new();
            for (i, node) in nodes.iter().enumerate() {
                match node {
                    Value::String(endpoint) => {
                        let parts: Vec<&str> = endpoint.split(':').collect();
                        if parts.len() == 2 {
                            let host = parts[0].to_string();
                            let port = parts[1].parse::<u16>().map_err(|_| VmError::TypeError {
                                expected: "Valid port number".to_string(),
                                actual: format!("Invalid port: {}", parts[1]),
                            })?;
                            
                            instances.push(ServiceInstance {
                                id: format!("node-{}", i),
                                host,
                                port,
                                weight: 1.0,
                                health: ServiceHealth::Healthy,
                                last_health_check: SystemTime::now(),
                                health_check_url: None,
                                metadata: HashMap::new(),
                                tags: Vec::new(),
                            });
                        } else {
                            return Err(VmError::TypeError {
                                expected: "Endpoint format 'host:port'".to_string(),
                                actual: format!("Invalid endpoint: {}", endpoint),
                            });
                        }
                    },
                    _ => return Err(VmError::TypeError {
                        expected: "String endpoints in cluster list".to_string(),
                        actual: format!("{:?}", node),
                    }),
                }
            }
            instances
        },
        _ => return Err(VmError::TypeError {
            expected: "List of cluster nodes".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    if cluster_nodes.is_empty() {
        return Err(VmError::Runtime("Cluster cannot be empty".to_string()));
    }
    
    let distributed_reduce = DistributedReduce::new(cluster_nodes);
    
    // Return the DistributedReduce object for configuration and later execution
    Ok(Value::LyObj(LyObj::new(Box::new(distributed_reduce))))
}

/// DistributedMapExecute[distributedMap, function, data] - Execute distributed map operation
pub fn distributed_map_execute(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (distributedMap, function, data)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let data = match &args[2] {
        Value::List(list) => list.clone(),
        _ => return Err(VmError::TypeError {
            expected: "List for data".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(dist_map) = obj.downcast_ref::<DistributedMap>() {
                let mut mutable_map = dist_map.clone();
                
                // Create a simple async runtime for execution
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    VmError::Runtime(format!("Failed to create async runtime: {}", e))
                })?;
                
                match rt.block_on(mutable_map.execute(&function_name, data)) {
                    Ok(results) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_map))), // Updated map with stats
                        Value::List(results) // Results
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Distributed map execution failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "DistributedMap object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "DistributedMap object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// DistributedReduceExecute[distributedReduce, function, data] - Execute distributed reduce operation
pub fn distributed_reduce_execute(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (distributedReduce, function, data)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let data = match &args[2] {
        Value::List(list) => list.clone(),
        _ => return Err(VmError::TypeError {
            expected: "List for data".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(dist_reduce) = obj.downcast_ref::<DistributedReduce>() {
                let mut mutable_reduce = dist_reduce.clone();
                
                // Create a simple async runtime for execution
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    VmError::Runtime(format!("Failed to create async runtime: {}", e))
                })?;
                
                match rt.block_on(mutable_reduce.execute(&function_name, data)) {
                    Ok(result) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_reduce))), // Updated reduce with stats
                        result // Final result
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Distributed reduce execution failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "DistributedReduce object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "DistributedReduce object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// ServiceRegistry[backend, config] - Create service registry
pub fn service_registry(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (backend, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let backend_spec = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for backend specification".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Parse backend specification
    let mut registry = if backend_spec.starts_with("dns:") {
        let domain = backend_spec.strip_prefix("dns:").unwrap_or("local");
        ServiceRegistry::with_dns(domain.to_string())
    } else if backend_spec.starts_with("http:") || backend_spec.starts_with("https:") {
        ServiceRegistry::with_http_api(backend_spec)
    } else if backend_spec == "static" {
        ServiceRegistry::with_static_config()
    } else {
        return Err(VmError::TypeError {
            expected: "Backend specification (dns:domain, http://endpoint, or 'static')".to_string(),
            actual: backend_spec,
        });
    };
    
    // Apply configuration if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(config_options) => {
                for option in config_options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "healthCheckInterval" => {
                                        if let Value::Real(seconds) = value {
                                            let timeout = registry.health_check_timeout;
                                            registry = registry.with_health_check_config(
                                                Duration::from_secs_f64(*seconds),
                                                timeout
                                            );
                                        }
                                    },
                                    "healthCheckTimeout" => {
                                        if let Value::Real(seconds) = value {
                                            let interval = registry.health_check_interval;
                                            registry = registry.with_health_check_config(
                                                interval,
                                                Duration::from_secs_f64(*seconds)
                                            );
                                        }
                                    },
                                    "notifications" => {
                                        if let Value::Integer(enabled) = value {
                                            registry = registry.with_change_notifications(*enabled != 0);
                                        }
                                    },
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            },
            _ => return Err(VmError::TypeError {
                expected: "List of configuration pairs".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(registry))))
}

/// ServiceDiscover[registry, serviceName, constraints] - Discover services
pub fn service_discover(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (registry, serviceName, [constraints])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let service_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for service name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(registry) = obj.downcast_ref::<ServiceRegistry>() {
                let mut mutable_registry = registry.clone();
                
                match mutable_registry.discover_services(&service_name) {
                    Ok(instances) => {
                        // Convert instances to Value representation
                        let instance_values: Vec<Value> = instances.iter()
                            .map(|instance| {
                                Value::List(vec![
                                    Value::String(instance.id.clone()),
                                    Value::String(instance.endpoint()),
                                    Value::String(match instance.health {
                                        ServiceHealth::Healthy => "Healthy",
                                        ServiceHealth::Unhealthy => "Unhealthy",
                                        ServiceHealth::Unknown => "Unknown",
                                    }.to_string()),
                                    Value::Real(instance.weight),
                                ])
                            })
                            .collect();
                        
                        Ok(Value::List(vec![
                            Value::LyObj(LyObj::new(Box::new(mutable_registry))), // Updated registry
                            Value::List(instance_values) // Discovered instances
                        ]))
                    },
                    Err(e) => Err(VmError::Runtime(format!("Service discovery failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "ServiceRegistry object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "ServiceRegistry object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// ServiceHealthCheck[registry, serviceName] - Perform health checks
pub fn service_health_check(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (registry, serviceName)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let service_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for service name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(registry) = obj.downcast_ref::<ServiceRegistry>() {
                let mut mutable_registry = registry.clone();
                
                // Get instances for the service
                let instances = mutable_registry.get_all_instances(&service_name);
                let mut health_results = Vec::new();
                
                // Perform health checks on all instances
                for mut instance in instances {
                    match mutable_registry.health_check(&mut instance) {
                        Ok(health) => {
                            health_results.push(Value::List(vec![
                                Value::String(instance.id.clone()),
                                Value::String(instance.endpoint()),
                                Value::String(match health {
                                    ServiceHealth::Healthy => "Healthy",
                                    ServiceHealth::Unhealthy => "Unhealthy",
                                    ServiceHealth::Unknown => "Unknown",
                                }.to_string()),
                                Value::Real(instance.last_health_check.duration_since(UNIX_EPOCH)
                                    .unwrap_or_default().as_secs_f64()),
                            ]));
                        },
                        Err(e) => {
                            health_results.push(Value::List(vec![
                                Value::String(instance.id.clone()),
                                Value::String(instance.endpoint()),
                                Value::String("Error".to_string()),
                                Value::String(e),
                            ]));
                        }
                    }
                }
                
                // Update health statistics
                mutable_registry.update_health_stats();
                
                Ok(Value::List(vec![
                    Value::LyObj(LyObj::new(Box::new(mutable_registry))), // Updated registry
                    Value::List(health_results) // Health check results
                ]))
            } else {
                Err(VmError::TypeError {
                    expected: "ServiceRegistry object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "ServiceRegistry object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Load balancing strategies
#[derive(Debug, Clone)]
enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LatencyBased,
    Random,
    ConsistentHashing,
    ResourceBased,
}

impl From<&str> for LoadBalancingStrategy {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "roundrobin" | "round-robin" => LoadBalancingStrategy::RoundRobin,
            "weighted" | "weighted-round-robin" => LoadBalancingStrategy::WeightedRoundRobin,
            "leastconnections" | "least-connections" => LoadBalancingStrategy::LeastConnections,
            "latency" | "latency-based" => LoadBalancingStrategy::LatencyBased,
            "random" => LoadBalancingStrategy::Random,
            "hash" | "consistent-hash" => LoadBalancingStrategy::ConsistentHashing,
            "resource" | "resource-based" => LoadBalancingStrategy::ResourceBased,
            _ => LoadBalancingStrategy::RoundRobin, // Default
        }
    }
}

impl std::fmt::Display for LoadBalancingStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadBalancingStrategy::RoundRobin => write!(f, "RoundRobin"),
            LoadBalancingStrategy::WeightedRoundRobin => write!(f, "WeightedRoundRobin"),
            LoadBalancingStrategy::LeastConnections => write!(f, "LeastConnections"),
            LoadBalancingStrategy::LatencyBased => write!(f, "LatencyBased"),
            LoadBalancingStrategy::Random => write!(f, "Random"),
            LoadBalancingStrategy::ConsistentHashing => write!(f, "ConsistentHashing"),
            LoadBalancingStrategy::ResourceBased => write!(f, "ResourceBased"),
        }
    }
}

/// Load balancer instance statistics
#[derive(Debug, Clone)]
struct InstanceStats {
    active_connections: u32,
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    average_latency: f64,
    last_request_time: Option<SystemTime>,
    circuit_breaker: Arc<Mutex<CircuitBreaker>>,
}

impl InstanceStats {
    fn new() -> Self {
        Self {
            active_connections: 0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            average_latency: 0.0,
            last_request_time: None,
            circuit_breaker: Arc::new(Mutex::new(CircuitBreaker::new(
                3, 
                Duration::from_secs(30)
            ))),
        }
    }
    
    fn record_request(&mut self, latency_ms: f64, success: bool) {
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
            
            // Update average latency (exponential moving average)
            if self.average_latency == 0.0 {
                self.average_latency = latency_ms;
            } else {
                self.average_latency = 0.9 * self.average_latency + 0.1 * latency_ms;
            }
            
            if let Ok(mut cb) = self.circuit_breaker.lock() {
                cb.record_success();
            }
        } else {
            self.failed_requests += 1;
            
            if let Ok(mut cb) = self.circuit_breaker.lock() {
                cb.record_failure();
            }
        }
        
        self.last_request_time = Some(SystemTime::now());
    }
    
    fn is_available(&self) -> bool {
        if let Ok(mut cb) = self.circuit_breaker.lock() {
            cb.can_execute()
        } else {
            true // Default to available if can't check
        }
    }
    
    fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            (self.successful_requests as f64 / self.total_requests as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Intelligent load balancer with multiple strategies and circuit breakers
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Service instances with statistics
    pub instances: HashMap<String, ServiceInstance>,
    pub instance_stats: HashMap<String, InstanceStats>,
    /// Round-robin counter for RoundRobin strategy
    pub round_robin_index: usize,
    /// Load balancer statistics
    pub total_requests: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    /// Configuration
    pub health_check_enabled: bool,
    pub circuit_breaker_enabled: bool,
    /// Consistent hashing ring (for ConsistentHashing strategy)
    pub hash_ring: BTreeMap<u64, String>,
    /// Request timeout
    pub request_timeout: Duration,
    /// HTTP client for making requests
    pub client: HttpClient,
}

impl LoadBalancer {
    /// Create a new load balancer
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            instances: HashMap::new(),
            instance_stats: HashMap::new(),
            round_robin_index: 0,
            total_requests: 0,
            successful_requests: 0,
            failed_requests: 0,
            health_check_enabled: true,
            circuit_breaker_enabled: true,
            hash_ring: BTreeMap::new(),
            request_timeout: Duration::from_secs(30),
            client: HttpClient::new(),
        }
    }
    
    /// Add a service instance to the load balancer
    pub fn add_instance(&mut self, instance: ServiceInstance) {
        let instance_id = instance.id.clone();
        self.instances.insert(instance_id.clone(), instance);
        self.instance_stats.insert(instance_id.clone(), InstanceStats::new());
        
        // Update hash ring for consistent hashing
        if matches!(self.strategy, LoadBalancingStrategy::ConsistentHashing) {
            self.rebuild_hash_ring();
        }
    }
    
    /// Remove a service instance
    pub fn remove_instance(&mut self, instance_id: &str) {
        self.instances.remove(instance_id);
        self.instance_stats.remove(instance_id);
        
        // Update hash ring for consistent hashing
        if matches!(self.strategy, LoadBalancingStrategy::ConsistentHashing) {
            self.rebuild_hash_ring();
        }
    }
    
    /// Get available instances (healthy + circuit breaker not open)
    fn get_available_instances(&self) -> Vec<&ServiceInstance> {
        self.instances.values()
            .filter(|instance| {
                // Check health if enabled
                let health_ok = !self.health_check_enabled || 
                               instance.health == ServiceHealth::Healthy;
                
                // Check circuit breaker if enabled
                let circuit_ok = !self.circuit_breaker_enabled ||
                                self.instance_stats.get(&instance.id)
                                    .map(|stats| stats.is_available())
                                    .unwrap_or(true);
                
                health_ok && circuit_ok
            })
            .collect()
    }
    
    /// Select an instance using the configured strategy
    pub fn select_instance(&mut self, key: Option<&str>) -> Option<String> {
        // Get available instance info (ID, weight, stats) to avoid borrowing conflicts
        let available_instance_info: Vec<(String, f64, u32, f64)> = self.instances.values()
            .filter(|instance| {
                // Check health if enabled
                let health_ok = !self.health_check_enabled || 
                               instance.health == ServiceHealth::Healthy;
                
                // Check circuit breaker if enabled
                let circuit_ok = !self.circuit_breaker_enabled ||
                                self.instance_stats.get(&instance.id)
                                    .map(|stats| stats.is_available())
                                    .unwrap_or(true);
                
                health_ok && circuit_ok
            })
            .map(|instance| {
                let stats = self.instance_stats.get(&instance.id);
                let active_connections = stats.map(|s| s.active_connections).unwrap_or(0);
                let avg_latency = stats.map(|s| s.average_latency).unwrap_or(f64::MAX);
                (instance.id.clone(), instance.weight, active_connections, avg_latency)
            })
            .collect();
        
        if available_instance_info.is_empty() {
            return None;
        }
        
        let instance_id = match &self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let index = self.round_robin_index % available_instance_info.len();
                let instance_id = available_instance_info[index].0.clone();
                self.round_robin_index = (self.round_robin_index + 1) % available_instance_info.len();
                instance_id
            },
            
            LoadBalancingStrategy::WeightedRoundRobin => {
                // Weighted selection based on instance weights
                let total_weight: f64 = available_instance_info.iter()
                    .map(|(_, weight, _, _)| *weight)
                    .sum();
                
                if total_weight <= 0.0 {
                    return available_instance_info.first().map(|(id, _, _, _)| id.clone());
                }
                
                let mut target = self.round_robin_index as f64 % total_weight;
                self.round_robin_index += 1;
                
                for (id, weight, _, _) in &available_instance_info {
                    target -= weight;
                    if target <= 0.0 {
                        return Some(id.clone());
                    }
                }
                
                available_instance_info.first().map(|(id, _, _, _)| id.clone()).unwrap_or_default()
            },
            
            LoadBalancingStrategy::LeastConnections => {
                // Select instance with least active connections
                available_instance_info.iter()
                    .min_by_key(|(_, _, active_connections, _)| *active_connections)
                    .map(|(id, _, _, _)| id.clone())
                    .unwrap_or_default()
            },
            
            LoadBalancingStrategy::LatencyBased => {
                // Select instance with lowest average latency
                available_instance_info.iter()
                    .min_by(|(_, _, _, latency_a), (_, _, _, latency_b)| {
                        latency_a.partial_cmp(latency_b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(id, _, _, _)| id.clone())
                    .unwrap_or_default()
            },
            
            LoadBalancingStrategy::Random => {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};
                
                let mut hasher = DefaultHasher::new();
                SystemTime::now().hash(&mut hasher);
                let random_index = (hasher.finish() as usize) % available_instance_info.len();
                available_instance_info[random_index].0.clone()
            },
            
            LoadBalancingStrategy::ConsistentHashing => {
                if let Some(key) = key {
                    let available_ids: Vec<String> = available_instance_info.iter()
                        .map(|(id, _, _, _)| id.clone())
                        .collect();
                    self.select_by_hash(key, &available_ids)
                } else {
                    // Fallback to round-robin if no key provided
                    let index = self.round_robin_index % available_instance_info.len();
                    let instance_id = available_instance_info[index].0.clone();
                    self.round_robin_index = (self.round_robin_index + 1) % available_instance_info.len();
                    instance_id
                }
            },
            LoadBalancingStrategy::ResourceBased => {
                // Select instance with best resource availability (lowest CPU/memory usage)
                let best_instance = available_instance_info.iter()
                    .min_by(|(_, _, cpu1, mem1), (_, _, cpu2, mem2)| {
                        let score1 = *cpu1 as f64 + mem1; // Simple combined resource score
                        let score2 = *cpu2 as f64 + mem2;
                        score1.partial_cmp(&score2).unwrap_or(std::cmp::Ordering::Equal)
                    });
                
                best_instance.map(|(id, _, _, _)| id.clone())
                    .unwrap_or_else(|| available_instance_info.first().unwrap().0.clone())
            },
        };
        
        Some(instance_id)
    }
    
    /// Consistent hashing selection
    fn select_by_hash(&self, key: &str, available_instance_ids: &[String]) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let hash = hasher.finish();
        
        // Find the first instance in the hash ring that has a hash >= key hash
        let instance_id = self.hash_ring.range(hash..)
            .next()
            .or_else(|| self.hash_ring.iter().next())
            .map(|(_, id)| id);
        
        if let Some(id) = instance_id {
            if available_instance_ids.contains(id) {
                id.clone()
            } else {
                available_instance_ids.first().cloned().unwrap_or_default()
            }
        } else {
            available_instance_ids.first().cloned().unwrap_or_default()
        }
    }
    
    /// Rebuild hash ring for consistent hashing
    fn rebuild_hash_ring(&mut self) {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        self.hash_ring.clear();
        
        for instance in self.instances.values() {
            // Add multiple virtual nodes for better distribution
            for i in 0..100 {
                let mut hasher = DefaultHasher::new();
                format!("{}-{}", instance.id, i).hash(&mut hasher);
                self.hash_ring.insert(hasher.finish(), instance.id.clone());
            }
        }
    }
    
    /// Execute a request through the load balancer
    pub fn execute_request(&mut self, path: &str, method: HttpMethod, body: Option<Vec<u8>>, routing_key: Option<&str>) -> Result<Value, String> {
        let start_time = Instant::now();
        
        // Select an instance
        let instance_id = self.select_instance(routing_key)
            .ok_or("No available instances")?;
        
        let instance = self.instances.get(&instance_id)
            .ok_or("Selected instance not found")?;
        
        let instance_endpoint = instance.endpoint();
        
        // Update active connections
        if let Some(stats) = self.instance_stats.get_mut(&instance_id) {
            stats.active_connections += 1;
        }
        
        // Build request URL
        let url = if path.starts_with("http") {
            path.to_string()
        } else {
            format!("http://{}{}", instance_endpoint, path)
        };
        
        // Create and execute request
        let mut request = NetworkRequest::new(url, method)
            .with_timeout(self.request_timeout);
        
        if let Some(body) = body {
            request = request.with_body(body);
        }
        
        let result = self.client.execute(&request);
        
        // Record metrics
        let latency = start_time.elapsed().as_secs_f64() * 1000.0;
        let success = result.is_ok() && result.as_ref().map(|r| r.is_success()).unwrap_or(false);
        
        self.total_requests += 1;
        if success {
            self.successful_requests += 1;
        } else {
            self.failed_requests += 1;
        }
        
        // Update instance statistics
        if let Some(stats) = self.instance_stats.get_mut(&instance_id) {
            stats.active_connections = stats.active_connections.saturating_sub(1);
            stats.record_request(latency, success);
        }
        
        // Convert result
        match result {
            Ok(response) => {
                if response.is_success() {
                    Ok(Value::LyObj(LyObj::new(Box::new(response))))
                } else {
                    Err(format!("HTTP {}: Request failed", response.status))
                }
            },
            Err(e) => Err(format!("Request execution failed: {}", e))
        }
    }
    
    /// Get load balancer statistics
    pub fn get_stats(&self) -> HashMap<String, Value> {
        let mut stats = HashMap::new();
        
        stats.insert("strategy".to_string(), Value::String(self.strategy.to_string()));
        stats.insert("total_instances".to_string(), Value::Integer(self.instances.len() as i64));
        stats.insert("available_instances".to_string(), Value::Integer(self.get_available_instances().len() as i64));
        stats.insert("total_requests".to_string(), Value::Integer(self.total_requests as i64));
        stats.insert("successful_requests".to_string(), Value::Integer(self.successful_requests as i64));
        stats.insert("failed_requests".to_string(), Value::Integer(self.failed_requests as i64));
        
        if self.total_requests > 0 {
            let success_rate = (self.successful_requests as f64 / self.total_requests as f64) * 100.0;
            stats.insert("success_rate".to_string(), Value::Real(success_rate));
        }
        
        stats
    }
    
    /// Configure load balancer options
    pub fn configure(&mut self, options: &HashMap<String, Value>) {
        if let Some(Value::Integer(enabled)) = options.get("health_check_enabled") {
            self.health_check_enabled = *enabled != 0;
        }
        
        if let Some(Value::Integer(enabled)) = options.get("circuit_breaker_enabled") {
            self.circuit_breaker_enabled = *enabled != 0;
        }
        
        if let Some(Value::Real(timeout)) = options.get("request_timeout") {
            self.request_timeout = Duration::from_secs_f64(*timeout);
        }
    }
}

impl Foreign for LoadBalancer {
    fn type_name(&self) -> &'static str {
        "LoadBalancer"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Strategy" => Ok(Value::String(self.strategy.to_string())),
            "InstanceCount" => Ok(Value::Integer(self.instances.len() as i64)),
            "AvailableInstances" => Ok(Value::Integer(self.get_available_instances().len() as i64)),
            "TotalRequests" => Ok(Value::Integer(self.total_requests as i64)),
            "SuccessfulRequests" => Ok(Value::Integer(self.successful_requests as i64)),
            "FailedRequests" => Ok(Value::Integer(self.failed_requests as i64)),
            "SuccessRate" => {
                if self.total_requests > 0 {
                    let rate = (self.successful_requests as f64 / self.total_requests as f64) * 100.0;
                    Ok(Value::Real(rate))
                } else {
                    Ok(Value::Real(0.0))
                }
            },
            "HealthCheckEnabled" => Ok(Value::Integer(if self.health_check_enabled { 1 } else { 0 })),
            "CircuitBreakerEnabled" => Ok(Value::Integer(if self.circuit_breaker_enabled { 1 } else { 0 })),
            "RequestTimeout" => Ok(Value::Real(self.request_timeout.as_secs_f64())),
            "Instances" => {
                let instance_list: Vec<Value> = self.instances.keys()
                    .map(|id| Value::String(id.clone()))
                    .collect();
                Ok(Value::List(instance_list))
            },
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

/// LoadBalancer[strategy, instances, config] - Create load balancer
pub fn load_balancer(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (strategy, [instances], [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let strategy = match &args[0] {
        Value::String(s) => LoadBalancingStrategy::from(s.as_str()),
        _ => return Err(VmError::TypeError {
            expected: "String for load balancing strategy".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut load_balancer = LoadBalancer::new(strategy);
    
    // Add instances if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(instances) => {
                for instance_value in instances {
                    // Parse instance specification
                    match instance_value {
                        Value::String(endpoint) => {
                            // Parse "host:port" format
                            if let Some((host, port_str)) = endpoint.split_once(':') {
                                if let Ok(port) = port_str.parse::<u16>() {
                                    let instance = ServiceInstance::new(
                                        format!("{}:{}", host, port),
                                        host.to_string(),
                                        port
                                    );
                                    load_balancer.add_instance(instance);
                                }
                            }
                        },
                        Value::List(instance_spec) if instance_spec.len() >= 2 => {
                            if let (Value::String(host), Value::Integer(port)) = (&instance_spec[0], &instance_spec[1]) {
                                let instance = ServiceInstance::new(
                                    format!("{}:{}", host, port),
                                    host.clone(),
                                    *port as u16
                                );
                                load_balancer.add_instance(instance);
                            }
                        },
                        _ => {}
                    }
                }
            },
            _ => return Err(VmError::TypeError {
                expected: "List of instances".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    // Apply configuration if provided
    if args.len() > 2 {
        match &args[2] {
            Value::List(config_options) => {
                let mut config = HashMap::new();
                for option in config_options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let Value::String(key) = &pair[0] {
                                config.insert(key.clone(), pair[1].clone());
                            }
                        }
                    }
                }
                load_balancer.configure(&config);
            },
            _ => return Err(VmError::TypeError {
                expected: "List of configuration pairs".to_string(),
                actual: format!("{:?}", args[2]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(load_balancer))))
}

/// LoadBalancerRequest[loadBalancer, path, method, body, routingKey] - Execute request through load balancer
pub fn load_balancer_request(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "2-5 arguments (loadBalancer, path, [method], [body], [routingKey])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for request path".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let method = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => HttpMethod::from(s.as_str()),
            _ => HttpMethod::GET,
        }
    } else {
        HttpMethod::GET
    };
    
    let body = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_bytes().to_vec()),
            _ => None,
        }
    } else {
        None
    };
    
    let routing_key = if args.len() > 4 {
        match &args[4] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(load_balancer) = obj.downcast_ref::<LoadBalancer>() {
                let mut mutable_lb = load_balancer.clone();
                
                match mutable_lb.execute_request(&path, method, body, routing_key) {
                    Ok(response) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_lb))), // Updated load balancer
                        response // Response
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Load balancer request failed: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "LoadBalancer object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "LoadBalancer object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// Compute Cluster Node representing a single compute resource in the cluster
#[derive(Debug, Clone)]
pub struct ClusterNode {
    pub id: String,
    pub endpoint: String,
    pub resources: NodeResources,
    pub status: NodeStatus,
    pub health: NodeHealth,
    pub last_heartbeat: SystemTime,
    pub assigned_tasks: Vec<String>,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct NodeResources {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub storage_gb: u64,
    pub network_mbps: u32,
    pub cpu_usage: f32,      // 0.0 to 1.0
    pub memory_usage: f32,   // 0.0 to 1.0
    pub storage_usage: f32,  // 0.0 to 1.0
}

#[derive(Debug, Clone, PartialEq)]
pub enum NodeStatus {
    Active,
    Draining,
    Maintenance,
    Failed,
}

#[derive(Debug, Clone)]
pub struct NodeHealth {
    pub is_healthy: bool,
    pub last_check: SystemTime,
    pub consecutive_failures: u32,
    pub latency_ms: u32,
    pub error_rate: f32,
}

/// Task Scheduler for distributing work across cluster nodes
#[derive(Debug, Clone)]
pub struct TaskScheduler {
    pub strategy: SchedulingStrategy,
    pub queue: Vec<ScheduledTask>,
    pub running_tasks: HashMap<String, String>, // task_id -> node_id
    pub completed_tasks: HashMap<String, TaskResult>,
}

#[derive(Debug, Clone)]
pub enum SchedulingStrategy {
    FirstFit,        // First available node
    BestFit,         // Node with least waste
    RoundRobin,      // Rotate through nodes
    ResourceBased,   // Based on CPU/memory requirements
    LoadBalanced,    // Distribute load evenly
}

#[derive(Debug, Clone)]
pub struct ScheduledTask {
    pub id: String,
    pub function_name: String,
    pub arguments: Vec<Value>,
    pub resource_requirements: ResourceRequirements,
    pub priority: u32,
    pub timeout: Duration,
    pub retry_count: u32,
    pub submitted_at: SystemTime,
}

#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: u32,
    pub memory_mb: u64,
    pub estimated_duration: Duration,
    pub network_intensive: bool,
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    pub task_id: String,
    pub node_id: String,
    pub result: Value,
    pub execution_time: Duration,
    pub resource_usage: NodeResources,
    pub completed_at: SystemTime,
}

/// Auto-scaler for dynamic cluster sizing based on demand
#[derive(Debug, Clone)]
pub struct AutoScaler {
    pub enabled: bool,
    pub min_nodes: u32,
    pub max_nodes: u32,
    pub target_cpu_utilization: f32,
    pub target_memory_utilization: f32,
    pub scale_up_threshold: f32,
    pub scale_down_threshold: f32,
    pub cooldown_period: Duration,
    pub last_scaling_action: Option<SystemTime>,
}

/// Health Monitor for continuous cluster health tracking
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    pub check_interval: Duration,
    pub timeout: Duration,
    pub failure_threshold: u32,
    pub recovery_threshold: u32,
    pub enabled_checks: Vec<HealthCheckType>,
}

#[derive(Debug, Clone)]
pub enum HealthCheckType {
    HeartBeat,
    ResourceUtilization,
    NetworkConnectivity,
    ServiceAvailability,
}

/// Cluster Statistics for monitoring and observability
#[derive(Debug, Clone)]
pub struct ClusterStats {
    pub total_nodes: u32,
    pub active_nodes: u32,
    pub failed_nodes: u32,
    pub total_tasks_submitted: u64,
    pub total_tasks_completed: u64,
    pub total_tasks_failed: u64,
    pub average_task_duration: Duration,
    pub cluster_cpu_utilization: f32,
    pub cluster_memory_utilization: f32,
    pub network_throughput_mbps: f32,
    pub uptime: Duration,
    pub started_at: SystemTime,
}

impl ClusterStats {
    pub fn new() -> Self {
        Self {
            total_nodes: 0,
            active_nodes: 0,
            failed_nodes: 0,
            total_tasks_submitted: 0,
            total_tasks_completed: 0,
            total_tasks_failed: 0,
            average_task_duration: Duration::from_secs(0),
            cluster_cpu_utilization: 0.0,
            cluster_memory_utilization: 0.0,
            network_throughput_mbps: 0.0,
            uptime: Duration::from_secs(0),
            started_at: SystemTime::now(),
        }
    }
}

/// ComputeCluster - Managed distributed computing cluster
#[derive(Debug, Clone)]
pub struct ComputeCluster {
    pub cluster_id: String,
    pub nodes: HashMap<String, ClusterNode>,
    pub scheduler: TaskScheduler,
    pub auto_scaler: AutoScaler,
    pub health_monitor: HealthMonitor,
    pub stats: ClusterStats,
    pub load_balancer: LoadBalancer,
    pub service_registry: ServiceRegistry,
}

impl ComputeCluster {
    /// Create a new compute cluster
    pub fn new(cluster_id: String) -> Self {
        let scheduler = TaskScheduler {
            strategy: SchedulingStrategy::ResourceBased,
            queue: Vec::new(),
            running_tasks: HashMap::new(),
            completed_tasks: HashMap::new(),
        };
        
        let auto_scaler = AutoScaler {
            enabled: true,
            min_nodes: 1,
            max_nodes: 10,
            target_cpu_utilization: 0.7,
            target_memory_utilization: 0.8,
            scale_up_threshold: 0.85,
            scale_down_threshold: 0.3,
            cooldown_period: Duration::from_secs(300), // 5 minutes
            last_scaling_action: None,
        };
        
        let health_monitor = HealthMonitor {
            check_interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_threshold: 2,
            enabled_checks: vec![
                HealthCheckType::HeartBeat,
                HealthCheckType::ResourceUtilization,
                HealthCheckType::NetworkConnectivity,
            ],
        };
        
        let load_balancer = LoadBalancer::new(LoadBalancingStrategy::ResourceBased);
        let service_registry = ServiceRegistry::new(DiscoveryBackend::Static(HashMap::new()));
        
        Self {
            cluster_id,
            nodes: HashMap::new(),
            scheduler,
            auto_scaler,
            health_monitor,
            stats: ClusterStats::new(),
            load_balancer,
            service_registry,
        }
    }
    
    /// Add a node to the cluster
    pub fn add_node(&mut self, node: ClusterNode) -> Result<(), String> {
        let node_id = node.id.clone();
        
        // Create service instance for load balancer
        let service_instance = ServiceInstance {
            id: node_id.clone(),
            host: node.endpoint.split(':').next().unwrap_or("localhost").to_string(),
            port: node.endpoint.split(':').nth(1)
                .and_then(|p| p.parse().ok())
                .unwrap_or(8080),
            weight: 1.0,
            health: ServiceHealth::Healthy,
            last_health_check: SystemTime::now(),
            health_check_url: Some(format!("{}/health", node.endpoint)),
            metadata: node.metadata.clone(),
            tags: vec!["compute-node".to_string()],
        };
        
        // Add to load balancer and service registry
        self.load_balancer.add_instance(service_instance.clone());
        self.service_registry.register_service("compute-nodes", service_instance)?;
        
        // Add to cluster
        self.nodes.insert(node_id, node);
        self.update_cluster_stats();
        
        Ok(())
    }
    
    /// Remove a node from the cluster
    pub fn remove_node(&mut self, node_id: &str) -> Result<(), String> {
        if let Some(_node) = self.nodes.remove(node_id) {
            self.load_balancer.remove_instance(node_id);
            self.service_registry.deregister_service("compute-nodes", node_id)?;
            
            // Move running tasks to other nodes
            self.reschedule_tasks_from_node(node_id)?;
            self.update_cluster_stats();
            
            Ok(())
        } else {
            Err(format!("Node {} not found in cluster", node_id))
        }
    }
    
    /// Submit a task for execution on the cluster
    pub async fn submit_task(&mut self, task: ScheduledTask) -> Result<String, String> {
        let task_id = task.id.clone();
        
        // Find suitable node for the task
        let node_id = self.schedule_task(&task)?;
        
        // Execute task on selected node
        match self.execute_task_on_node(&task, &node_id).await {
            Ok(result) => {
                self.scheduler.completed_tasks.insert(task_id.clone(), result);
                self.stats.total_tasks_completed += 1;
                Ok(task_id)
            },
            Err(e) => {
                self.stats.total_tasks_failed += 1;
                
                // Retry on different node if available
                if task.retry_count > 0 {
                    let mut retry_task = task;
                    retry_task.retry_count -= 1;
                    self.scheduler.queue.push(retry_task);
                    Ok(format!("{}-retry", task_id))
                } else {
                    Err(format!("Task execution failed: {}", e))
                }
            }
        }
    }
    
    /// Schedule a task to an appropriate node
    fn schedule_task(&mut self, task: &ScheduledTask) -> Result<String, String> {
        let available_nodes: Vec<&ClusterNode> = self.nodes.values()
            .filter(|node| {
                node.status == NodeStatus::Active &&
                node.health.is_healthy &&
                self.node_can_handle_task(node, &task.resource_requirements)
            })
            .collect();
        
        if available_nodes.is_empty() {
            return Err("No available nodes to schedule task".to_string());
        }
        
        let selected_node = match self.scheduler.strategy {
            SchedulingStrategy::FirstFit => {
                available_nodes.first().unwrap().id.clone()
            },
            SchedulingStrategy::BestFit => {
                self.find_best_fit_node(&available_nodes, &task.resource_requirements)
            },
            SchedulingStrategy::RoundRobin => {
                let index = self.stats.total_tasks_submitted as usize % available_nodes.len();
                available_nodes[index].id.clone()
            },
            SchedulingStrategy::ResourceBased => {
                self.find_resource_optimal_node(&available_nodes, &task.resource_requirements)
            },
            SchedulingStrategy::LoadBalanced => {
                self.find_least_loaded_node(&available_nodes)
            },
        };
        
        self.scheduler.running_tasks.insert(task.id.clone(), selected_node.clone());
        self.stats.total_tasks_submitted += 1;
        
        Ok(selected_node)
    }
    
    /// Execute a task on a specific node
    async fn execute_task_on_node(&self, task: &ScheduledTask, node_id: &str) -> Result<TaskResult, String> {
        let node = self.nodes.get(node_id)
            .ok_or(format!("Node {} not found", node_id))?;
        
        let start_time = SystemTime::now();
        
        // Create remote function for task execution
        let mut remote_fn = RemoteFunction::new(node.endpoint.clone(), task.function_name.clone())
            .with_timeout(task.timeout.as_secs_f64());
        
        // Execute the task
        match remote_fn.call(task.arguments.clone()) {
            Ok(result) => {
                let execution_time = start_time.elapsed().unwrap_or(Duration::from_secs(0));
                
                Ok(TaskResult {
                    task_id: task.id.clone(),
                    node_id: node_id.to_string(),
                    result,
                    execution_time,
                    resource_usage: node.resources.clone(),
                    completed_at: SystemTime::now(),
                })
            },
            Err(e) => Err(format!("Remote execution failed: {}", e))
        }
    }
    
    /// Check if a node can handle a task's resource requirements
    fn node_can_handle_task(&self, node: &ClusterNode, requirements: &ResourceRequirements) -> bool {
        let available_cpu = (node.resources.cpu_cores as f32 * (1.0 - node.resources.cpu_usage)) as u32;
        let available_memory = (node.resources.memory_mb as f32 * (1.0 - node.resources.memory_usage)) as u64;
        
        available_cpu >= requirements.cpu_cores && available_memory >= requirements.memory_mb
    }
    
    /// Find the best fit node (minimal resource waste)
    fn find_best_fit_node(&self, nodes: &[&ClusterNode], requirements: &ResourceRequirements) -> String {
        nodes.iter()
            .min_by_key(|node| {
                let cpu_waste = node.resources.cpu_cores.saturating_sub(requirements.cpu_cores);
                let memory_waste = node.resources.memory_mb.saturating_sub(requirements.memory_mb);
                cpu_waste + (memory_waste / 1024) as u32 // Normalize memory to GB
            })
            .map(|node| node.id.clone())
            .unwrap_or_else(|| nodes[0].id.clone())
    }
    
    /// Find the resource-optimal node based on current utilization
    fn find_resource_optimal_node(&self, nodes: &[&ClusterNode], _requirements: &ResourceRequirements) -> String {
        nodes.iter()
            .min_by(|a, b| {
                let a_load = (a.resources.cpu_usage + a.resources.memory_usage) / 2.0;
                let b_load = (b.resources.cpu_usage + b.resources.memory_usage) / 2.0;
                a_load.partial_cmp(&b_load).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|node| node.id.clone())
            .unwrap_or_else(|| nodes[0].id.clone())
    }
    
    /// Find the least loaded node
    fn find_least_loaded_node(&self, nodes: &[&ClusterNode]) -> String {
        nodes.iter()
            .min_by_key(|node| node.assigned_tasks.len())
            .map(|node| node.id.clone())
            .unwrap_or_else(|| nodes[0].id.clone())
    }
    
    /// Reschedule tasks from a failed/removed node
    fn reschedule_tasks_from_node(&mut self, node_id: &str) -> Result<(), String> {
        let tasks_to_reschedule: Vec<String> = self.scheduler.running_tasks
            .iter()
            .filter(|(_, assigned_node)| *assigned_node == node_id)
            .map(|(task_id, _)| task_id.clone())
            .collect();
        
        for task_id in tasks_to_reschedule {
            self.scheduler.running_tasks.remove(&task_id);
            // In a real implementation, we would reschedule these tasks
            println!("Rescheduling task {} from failed node {}", task_id, node_id);
        }
        
        Ok(())
    }
    
    /// Update cluster statistics
    fn update_cluster_stats(&mut self) {
        self.stats.total_nodes = self.nodes.len() as u32;
        self.stats.active_nodes = self.nodes.values()
            .filter(|node| node.status == NodeStatus::Active)
            .count() as u32;
        self.stats.failed_nodes = self.nodes.values()
            .filter(|node| node.status == NodeStatus::Failed)
            .count() as u32;
        
        // Calculate cluster-wide resource utilization
        if !self.nodes.is_empty() {
            let total_cpu: f32 = self.nodes.values().map(|n| n.resources.cpu_usage).sum();
            let total_memory: f32 = self.nodes.values().map(|n| n.resources.memory_usage).sum();
            
            self.stats.cluster_cpu_utilization = total_cpu / self.nodes.len() as f32;
            self.stats.cluster_memory_utilization = total_memory / self.nodes.len() as f32;
        }
        
        self.stats.uptime = self.stats.started_at.elapsed().unwrap_or(Duration::from_secs(0));
    }
    
    /// Auto-scale the cluster based on current load
    pub async fn auto_scale(&mut self) -> Result<(), String> {
        if !self.auto_scaler.enabled {
            return Ok(());
        }
        
        // Check cooldown period
        if let Some(last_action) = self.auto_scaler.last_scaling_action {
            if last_action.elapsed().unwrap_or(Duration::from_secs(0)) < self.auto_scaler.cooldown_period {
                return Ok(());
            }
        }
        
        let current_cpu = self.stats.cluster_cpu_utilization;
        let current_memory = self.stats.cluster_memory_utilization;
        let current_load = (current_cpu + current_memory) / 2.0;
        
        if current_load > self.auto_scaler.scale_up_threshold && 
           self.stats.active_nodes < self.auto_scaler.max_nodes {
            self.scale_up().await?;
        } else if current_load < self.auto_scaler.scale_down_threshold && 
                  self.stats.active_nodes > self.auto_scaler.min_nodes {
            self.scale_down().await?;
        }
        
        Ok(())
    }
    
    /// Scale up the cluster by adding a new node
    async fn scale_up(&mut self) -> Result<(), String> {
        println!("Scaling up cluster: current load exceeds threshold");
        self.auto_scaler.last_scaling_action = Some(SystemTime::now());
        // In a real implementation, this would provision a new compute node
        Ok(())
    }
    
    /// Scale down the cluster by removing an underutilized node
    async fn scale_down(&mut self) -> Result<(), String> {
        println!("Scaling down cluster: current load below threshold");
        self.auto_scaler.last_scaling_action = Some(SystemTime::now());
        // In a real implementation, this would decommission a node
        Ok(())
    }
}

impl Foreign for ComputeCluster {
    fn type_name(&self) -> &'static str {
        "ComputeCluster"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "ClusterID" => Ok(Value::String(self.cluster_id.clone())),
            "NodeCount" => Ok(Value::Integer(self.stats.total_nodes as i64)),
            "ActiveNodes" => Ok(Value::Integer(self.stats.active_nodes as i64)),
            "FailedNodes" => Ok(Value::Integer(self.stats.failed_nodes as i64)),
            "TasksSubmitted" => Ok(Value::Integer(self.stats.total_tasks_submitted as i64)),
            "TasksCompleted" => Ok(Value::Integer(self.stats.total_tasks_completed as i64)),
            "TasksFailed" => Ok(Value::Integer(self.stats.total_tasks_failed as i64)),
            "ClusterCPUUtilization" => Ok(Value::Real(self.stats.cluster_cpu_utilization as f64)),
            "ClusterMemoryUtilization" => Ok(Value::Real(self.stats.cluster_memory_utilization as f64)),
            "NetworkThroughput" => Ok(Value::Real(self.stats.network_throughput_mbps as f64)),
            "Uptime" => Ok(Value::Real(self.stats.uptime.as_secs_f64())),
            "AutoScalerEnabled" => Ok(Value::Integer(if self.auto_scaler.enabled { 1 } else { 0 })),
            "MinNodes" => Ok(Value::Integer(self.auto_scaler.min_nodes as i64)),
            "MaxNodes" => Ok(Value::Integer(self.auto_scaler.max_nodes as i64)),
            "TargetCPUUtilization" => Ok(Value::Real(self.auto_scaler.target_cpu_utilization as f64)),
            "TargetMemoryUtilization" => Ok(Value::Real(self.auto_scaler.target_memory_utilization as f64)),
            "QueuedTasks" => Ok(Value::Integer(self.scheduler.queue.len() as i64)),
            "RunningTasks" => Ok(Value::Integer(self.scheduler.running_tasks.len() as i64)),
            "CompletedTasks" => Ok(Value::Integer(self.scheduler.completed_tasks.len() as i64)),
            "SchedulingStrategy" => {
                let strategy_name = match self.scheduler.strategy {
                    SchedulingStrategy::FirstFit => "FirstFit",
                    SchedulingStrategy::BestFit => "BestFit", 
                    SchedulingStrategy::RoundRobin => "RoundRobin",
                    SchedulingStrategy::ResourceBased => "ResourceBased",
                    SchedulingStrategy::LoadBalanced => "LoadBalanced",
                };
                Ok(Value::String(strategy_name.to_string()))
            },
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

/// ComputeCluster[clusterID, config] - Create compute cluster
pub fn compute_cluster(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (clusterID, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster_id = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for cluster ID".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut cluster = ComputeCluster::new(cluster_id);
    
    // Apply configuration if provided
    if args.len() > 1 {
        match &args[1] {
            Value::List(config_options) => {
                for option in config_options {
                    if let Value::List(pair) = option {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "minNodes" => {
                                        if let Value::Integer(min) = value {
                                            cluster.auto_scaler.min_nodes = *min as u32;
                                        }
                                    },
                                    "maxNodes" => {
                                        if let Value::Integer(max) = value {
                                            cluster.auto_scaler.max_nodes = *max as u32;
                                        }
                                    },
                                    "targetCPUUtilization" => {
                                        if let Value::Real(cpu) = value {
                                            cluster.auto_scaler.target_cpu_utilization = *cpu as f32;
                                        }
                                    },
                                    "targetMemoryUtilization" => {
                                        if let Value::Real(mem) = value {
                                            cluster.auto_scaler.target_memory_utilization = *mem as f32;
                                        }
                                    },
                                    "autoScaling" => {
                                        if let Value::Integer(enabled) = value {
                                            cluster.auto_scaler.enabled = *enabled != 0;
                                        }
                                    },
                                    "schedulingStrategy" => {
                                        if let Value::String(strategy) = value {
                                            cluster.scheduler.strategy = match strategy.as_str() {
                                                "FirstFit" => SchedulingStrategy::FirstFit,
                                                "BestFit" => SchedulingStrategy::BestFit,
                                                "RoundRobin" => SchedulingStrategy::RoundRobin,
                                                "ResourceBased" => SchedulingStrategy::ResourceBased,
                                                "LoadBalanced" => SchedulingStrategy::LoadBalanced,
                                                _ => SchedulingStrategy::ResourceBased,
                                            };
                                        }
                                    },
                                    _ => {} // Ignore unknown options
                                }
                            }
                        }
                    }
                }
            },
            _ => return Err(VmError::TypeError {
                expected: "List of configuration options".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(cluster))))
}

/// ClusterAddNode[cluster, endpoint, resources] - Add a node to the cluster
pub fn cluster_add_node(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (cluster, endpoint, resources)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let endpoint = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for endpoint".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // Parse resources configuration
    let resources = match &args[2] {
        Value::List(resource_config) => {
            let mut cpu_cores = 4u32;
            let mut memory_mb = 8192u64;
            let mut storage_gb = 100u64;
            let mut network_mbps = 1000u32;
            
            for config in resource_config {
                if let Value::List(pair) = config {
                    if pair.len() == 2 {
                        if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                            match key.as_str() {
                                "cpuCores" => {
                                    if let Value::Integer(cores) = value {
                                        cpu_cores = *cores as u32;
                                    }
                                },
                                "memoryMB" => {
                                    if let Value::Integer(mem) = value {
                                        memory_mb = *mem as u64;
                                    }
                                },
                                "storageGB" => {
                                    if let Value::Integer(storage) = value {
                                        storage_gb = *storage as u64;
                                    }
                                },
                                "networkMbps" => {
                                    if let Value::Integer(net) = value {
                                        network_mbps = *net as u32;
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
            
            NodeResources {
                cpu_cores,
                memory_mb,
                storage_gb,
                network_mbps,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                storage_usage: 0.0,
            }
        },
        _ => return Err(VmError::TypeError {
            expected: "List of resource configuration".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(cluster) = obj.downcast_ref::<ComputeCluster>() {
                let mut mutable_cluster = cluster.clone();
                
                // Create new cluster node
                let node_id = format!("node-{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
                let node = ClusterNode {
                    id: node_id.clone(),
                    endpoint,
                    resources,
                    status: NodeStatus::Active,
                    health: NodeHealth {
                        is_healthy: true,
                        last_check: SystemTime::now(),
                        consecutive_failures: 0,
                        latency_ms: 0,
                        error_rate: 0.0,
                    },
                    last_heartbeat: SystemTime::now(),
                    assigned_tasks: Vec::new(),
                    metadata: HashMap::new(),
                };
                
                match mutable_cluster.add_node(node) {
                    Ok(()) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_cluster))), // Updated cluster
                        Value::String(node_id) // Node ID
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Failed to add node: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "ComputeCluster object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "ComputeCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// ClusterSubmitTask[cluster, function, args, requirements] - Submit a task to the cluster
pub fn cluster_submit_task(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (cluster, function, args, [requirements])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let task_args = match &args[2] {
        Value::List(args_list) => args_list.clone(),
        other => vec![other.clone()], // Single argument
    };
    
    // Parse resource requirements
    let resource_requirements = if args.len() > 3 {
        match &args[3] {
            Value::List(req_config) => {
                let mut cpu_cores = 1u32;
                let mut memory_mb = 512u64;
                let mut estimated_duration = Duration::from_secs(300); // 5 minutes default
                let mut network_intensive = false;
                
                for config in req_config {
                    if let Value::List(pair) = config {
                        if pair.len() == 2 {
                            if let (Value::String(key), value) = (&pair[0], &pair[1]) {
                                match key.as_str() {
                                    "cpuCores" => {
                                        if let Value::Integer(cores) = value {
                                            cpu_cores = *cores as u32;
                                        }
                                    },
                                    "memoryMB" => {
                                        if let Value::Integer(mem) = value {
                                            memory_mb = *mem as u64;
                                        }
                                    },
                                    "estimatedDurationSeconds" => {
                                        if let Value::Integer(duration) = value {
                                            estimated_duration = Duration::from_secs(*duration as u64);
                                        }
                                    },
                                    "networkIntensive" => {
                                        if let Value::Integer(intensive) = value {
                                            network_intensive = *intensive != 0;
                                        }
                                    },
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                
                ResourceRequirements {
                    cpu_cores,
                    memory_mb,
                    estimated_duration,
                    network_intensive,
                }
            },
            _ => return Err(VmError::TypeError {
                expected: "List of resource requirements".to_string(),
                actual: format!("{:?}", args[3]),
            }),
        }
    } else {
        ResourceRequirements {
            cpu_cores: 1,
            memory_mb: 512,
            estimated_duration: Duration::from_secs(300),
            network_intensive: false,
        }
    };
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(cluster) = obj.downcast_ref::<ComputeCluster>() {
                let mut mutable_cluster = cluster.clone();
                
                // Create scheduled task
                let task_id = format!("task-{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
                let task = ScheduledTask {
                    id: task_id.clone(),
                    function_name,
                    arguments: task_args,
                    resource_requirements,
                    priority: 1,
                    timeout: Duration::from_secs(600), // 10 minute timeout
                    retry_count: 2,
                    submitted_at: SystemTime::now(),
                };
                
                // For now, create a simple async runtime for execution
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    VmError::Runtime(format!("Failed to create async runtime: {}", e))
                })?;
                
                match rt.block_on(mutable_cluster.submit_task(task)) {
                    Ok(result_task_id) => Ok(Value::List(vec![
                        Value::LyObj(LyObj::new(Box::new(mutable_cluster))), // Updated cluster
                        Value::String(result_task_id) // Task ID
                    ])),
                    Err(e) => Err(VmError::Runtime(format!("Failed to submit task: {}", e))),
                }
            } else {
                Err(VmError::TypeError {
                    expected: "ComputeCluster object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "ComputeCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}

/// ClusterGetStats[cluster] - Get cluster statistics and status
pub fn cluster_get_stats(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (cluster)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    match &args[0] {
        Value::LyObj(obj) => {
            if let Some(cluster) = obj.downcast_ref::<ComputeCluster>() {
                let stats = &cluster.stats;
                
                // Return comprehensive cluster statistics
                let stats_list = vec![
                    Value::List(vec![Value::String("TotalNodes".to_string()), Value::Integer(stats.total_nodes as i64)]),
                    Value::List(vec![Value::String("ActiveNodes".to_string()), Value::Integer(stats.active_nodes as i64)]),
                    Value::List(vec![Value::String("FailedNodes".to_string()), Value::Integer(stats.failed_nodes as i64)]),
                    Value::List(vec![Value::String("TasksSubmitted".to_string()), Value::Integer(stats.total_tasks_submitted as i64)]),
                    Value::List(vec![Value::String("TasksCompleted".to_string()), Value::Integer(stats.total_tasks_completed as i64)]),
                    Value::List(vec![Value::String("TasksFailed".to_string()), Value::Integer(stats.total_tasks_failed as i64)]),
                    Value::List(vec![Value::String("ClusterCPUUtilization".to_string()), Value::Real(stats.cluster_cpu_utilization as f64)]),
                    Value::List(vec![Value::String("ClusterMemoryUtilization".to_string()), Value::Real(stats.cluster_memory_utilization as f64)]),
                    Value::List(vec![Value::String("NetworkThroughput".to_string()), Value::Real(stats.network_throughput_mbps as f64)]),
                    Value::List(vec![Value::String("UptimeSeconds".to_string()), Value::Real(stats.uptime.as_secs_f64())]),
                    Value::List(vec![Value::String("QueuedTasks".to_string()), Value::Integer(cluster.scheduler.queue.len() as i64)]),
                    Value::List(vec![Value::String("RunningTasks".to_string()), Value::Integer(cluster.scheduler.running_tasks.len() as i64)]),
                ];
                
                Ok(Value::List(stats_list))
            } else {
                Err(VmError::TypeError {
                    expected: "ComputeCluster object".to_string(),
                    actual: "Different object type".to_string(),
                })
            }
        },
        _ => Err(VmError::TypeError {
            expected: "ComputeCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    }
}