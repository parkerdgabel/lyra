use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Security middleware for WebSocket connections
pub struct SecurityMiddleware {
    rate_limiter: Arc<RateLimiter>,
    request_monitor: Arc<RequestMonitor>,
    config: SecurityConfig,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable rate limiting
    pub enable_rate_limiting: bool,
    /// Maximum requests per minute per client
    pub requests_per_minute: u32,
    /// Enable request monitoring
    pub enable_monitoring: bool,
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Maximum concurrent connections per IP
    pub max_connections_per_ip: u32,
    /// Enable IP whitelisting
    pub enable_ip_whitelist: bool,
    /// Whitelisted IP addresses
    pub ip_whitelist: Vec<String>,
    /// Enable IP blacklisting
    pub enable_ip_blacklist: bool,
    /// Blacklisted IP addresses
    pub ip_blacklist: Vec<String>,
    /// Suspicious behavior detection
    pub enable_anomaly_detection: bool,
    /// Rate limit burst tolerance
    pub rate_limit_burst: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_rate_limiting: true,
            requests_per_minute: 60,
            enable_monitoring: true,
            max_message_size: 1024 * 1024, // 1MB
            max_connections_per_ip: 10,
            enable_ip_whitelist: false,
            ip_whitelist: vec![],
            enable_ip_blacklist: false,
            ip_blacklist: vec![],
            enable_anomaly_detection: true,
            rate_limit_burst: 10,
        }
    }
}

/// Rate limiter implementation
pub struct RateLimiter {
    clients: Arc<RwLock<HashMap<String, ClientRateInfo>>>,
    config: SecurityConfig,
}

/// Rate limiting information for a client
#[derive(Debug, Clone)]
struct ClientRateInfo {
    /// Number of requests in current window
    request_count: u32,
    /// Window start time
    window_start: Instant,
    /// Burst tokens available
    burst_tokens: u32,
    /// Last request time
    last_request: Instant,
    /// Total requests ever
    total_requests: u64,
}

impl ClientRateInfo {
    fn new(burst_tokens: u32) -> Self {
        let now = Instant::now();
        Self {
            request_count: 0,
            window_start: now,
            burst_tokens,
            last_request: now,
            total_requests: 0,
        }
    }
}

impl RateLimiter {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            clients: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Check if a client is within rate limits
    pub async fn check_rate_limit(&self, client_id: &str) -> Result<(), SecurityError> {
        if !self.config.enable_rate_limiting {
            return Ok(());
        }

        let mut clients = self.clients.write().await;
        let now = Instant::now();
        
        let client_info = clients.entry(client_id.to_string()).or_insert_with(|| {
            ClientRateInfo::new(self.config.rate_limit_burst)
        });

        // Check if we need to reset the rate limit window (1 minute)
        if now.duration_since(client_info.window_start) >= Duration::from_secs(60) {
            client_info.request_count = 0;
            client_info.window_start = now;
            client_info.burst_tokens = self.config.rate_limit_burst;
        }

        // Check burst tokens first (for handling short bursts)
        if client_info.burst_tokens > 0 {
            client_info.burst_tokens -= 1;
            client_info.request_count += 1;
            client_info.last_request = now;
            client_info.total_requests += 1;
            return Ok(());
        }

        // Check regular rate limit
        if client_info.request_count >= self.config.requests_per_minute {
            return Err(SecurityError::RateLimitExceeded {
                limit: self.config.requests_per_minute,
                window_seconds: 60,
                retry_after: 60 - now.duration_since(client_info.window_start).as_secs(),
            });
        }

        // Replenish burst tokens gradually
        let time_since_last = now.duration_since(client_info.last_request).as_secs();
        if time_since_last > 0 {
            let tokens_to_add = std::cmp::min(
                time_since_last as u32 / 10, // Add 1 token every 10 seconds
                self.config.rate_limit_burst - client_info.burst_tokens,
            );
            client_info.burst_tokens += tokens_to_add;
        }

        client_info.request_count += 1;
        client_info.last_request = now;
        client_info.total_requests += 1;

        Ok(())
    }

    /// Get rate limit status for a client
    pub async fn get_rate_limit_status(&self, client_id: &str) -> Option<RateLimitStatus> {
        let clients = self.clients.read().await;
        clients.get(client_id).map(|info| {
            let now = Instant::now();
            let window_remaining = 60 - now.duration_since(info.window_start).as_secs();
            
            RateLimitStatus {
                requests_remaining: self.config.requests_per_minute.saturating_sub(info.request_count),
                burst_tokens_remaining: info.burst_tokens,
                window_reset_seconds: window_remaining,
                total_requests: info.total_requests,
            }
        })
    }

    /// Clean up old client entries
    pub async fn cleanup_old_entries(&self) {
        let mut clients = self.clients.write().await;
        let now = Instant::now();
        
        clients.retain(|_, info| {
            now.duration_since(info.last_request) < Duration::from_secs(3600) // Keep for 1 hour
        });
    }
}

/// Rate limit status information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitStatus {
    pub requests_remaining: u32,
    pub burst_tokens_remaining: u32,
    pub window_reset_seconds: u64,
    pub total_requests: u64,
}

/// Request monitoring and anomaly detection
pub struct RequestMonitor {
    connections: Arc<RwLock<HashMap<String, ConnectionInfo>>>,
    suspicious_ips: Arc<RwLock<HashMap<String, SuspiciousActivity>>>,
    config: SecurityConfig,
}

/// Connection information for monitoring
#[derive(Debug, Clone)]
struct ConnectionInfo {
    /// Number of active connections
    active_connections: u32,
    /// Connection start times
    connection_times: Vec<Instant>,
    /// Last activity
    last_activity: Instant,
    /// Total messages sent
    total_messages: u64,
    /// Average message size
    avg_message_size: f64,
}

/// Suspicious activity tracking
#[derive(Debug, Clone)]
struct SuspiciousActivity {
    /// Rapid connection attempts
    rapid_connections: u32,
    /// Large message attempts
    large_messages: u32,
    /// Rate limit violations
    rate_limit_violations: u32,
    /// First suspicious activity time
    first_suspicious: Instant,
    /// Last suspicious activity time
    last_suspicious: Instant,
    /// Whether IP is currently blocked
    is_blocked: bool,
}

impl RequestMonitor {
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            suspicious_ips: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Record a new connection
    pub async fn record_connection(&self, client_ip: &str) -> Result<(), SecurityError> {
        if !self.config.enable_monitoring {
            return Ok(());
        }

        // Check IP whitelist/blacklist
        self.check_ip_restrictions(client_ip)?;

        let mut connections = self.connections.write().await;
        let now = Instant::now();
        
        let conn_info = connections.entry(client_ip.to_string()).or_insert_with(|| {
            ConnectionInfo {
                active_connections: 0,
                connection_times: Vec::new(),
                last_activity: now,
                total_messages: 0,
                avg_message_size: 0.0,
            }
        });

        // Check for too many connections from same IP
        if conn_info.active_connections >= self.config.max_connections_per_ip {
            return Err(SecurityError::TooManyConnections {
                ip: client_ip.to_string(),
                limit: self.config.max_connections_per_ip,
            });
        }

        // Check for rapid connection attempts (anomaly detection)
        if self.config.enable_anomaly_detection {
            conn_info.connection_times.push(now);
            
            // Keep only recent connections (last 5 minutes)
            conn_info.connection_times.retain(|&time| {
                now.duration_since(time) < Duration::from_secs(300)
            });

            // Check for suspicious rapid connections
            if conn_info.connection_times.len() > 20 {
                self.record_suspicious_activity(client_ip, SuspiciousActivityType::RapidConnections).await;
                return Err(SecurityError::SuspiciousActivity {
                    reason: "Too many rapid connection attempts".to_string(),
                });
            }
        }

        conn_info.active_connections += 1;
        conn_info.last_activity = now;

        Ok(())
    }

    /// Record a connection closure
    pub async fn record_disconnection(&self, client_ip: &str) {
        let mut connections = self.connections.write().await;
        if let Some(conn_info) = connections.get_mut(client_ip) {
            conn_info.active_connections = conn_info.active_connections.saturating_sub(1);
        }
    }

    /// Record a message and check for anomalies
    pub async fn record_message(&self, client_ip: &str, message_size: usize) -> Result<(), SecurityError> {
        if !self.config.enable_monitoring {
            return Ok(());
        }

        // Check message size
        if message_size > self.config.max_message_size {
            if self.config.enable_anomaly_detection {
                self.record_suspicious_activity(client_ip, SuspiciousActivityType::LargeMessage).await;
            }
            return Err(SecurityError::MessageTooLarge {
                size: message_size,
                limit: self.config.max_message_size,
            });
        }

        let mut connections = self.connections.write().await;
        if let Some(conn_info) = connections.get_mut(client_ip) {
            conn_info.total_messages += 1;
            conn_info.last_activity = Instant::now();
            
            // Update average message size
            let total_size = conn_info.avg_message_size * (conn_info.total_messages - 1) as f64 + message_size as f64;
            conn_info.avg_message_size = total_size / conn_info.total_messages as f64;
        }

        Ok(())
    }

    /// Check IP restrictions (whitelist/blacklist)
    fn check_ip_restrictions(&self, client_ip: &str) -> Result<(), SecurityError> {
        // Check blacklist first
        if self.config.enable_ip_blacklist && self.config.ip_blacklist.contains(&client_ip.to_string()) {
            return Err(SecurityError::IpBlocked {
                ip: client_ip.to_string(),
                reason: "IP is blacklisted".to_string(),
            });
        }

        // Check whitelist if enabled
        if self.config.enable_ip_whitelist && !self.config.ip_whitelist.contains(&client_ip.to_string()) {
            return Err(SecurityError::IpBlocked {
                ip: client_ip.to_string(),
                reason: "IP is not whitelisted".to_string(),
            });
        }

        Ok(())
    }

    /// Record suspicious activity
    async fn record_suspicious_activity(&self, client_ip: &str, activity_type: SuspiciousActivityType) {
        let mut suspicious = self.suspicious_ips.write().await;
        let now = Instant::now();
        
        let activity = suspicious.entry(client_ip.to_string()).or_insert_with(|| {
            SuspiciousActivity {
                rapid_connections: 0,
                large_messages: 0,
                rate_limit_violations: 0,
                first_suspicious: now,
                last_suspicious: now,
                is_blocked: false,
            }
        });

        match activity_type {
            SuspiciousActivityType::RapidConnections => activity.rapid_connections += 1,
            SuspiciousActivityType::LargeMessage => activity.large_messages += 1,
            SuspiciousActivityType::RateLimitViolation => activity.rate_limit_violations += 1,
        }

        activity.last_suspicious = now;

        // Auto-block if too much suspicious activity
        let total_suspicious = activity.rapid_connections + activity.large_messages + activity.rate_limit_violations;
        if total_suspicious > 10 && !activity.is_blocked {
            activity.is_blocked = true;
            println!("🚨 Auto-blocked suspicious IP: {} (total violations: {})", client_ip, total_suspicious);
        }
    }

    /// Check if an IP is blocked due to suspicious activity
    pub async fn is_ip_blocked(&self, client_ip: &str) -> bool {
        let suspicious = self.suspicious_ips.read().await;
        suspicious.get(client_ip).map(|activity| activity.is_blocked).unwrap_or(false)
    }

    /// Get connection statistics
    pub async fn get_connection_stats(&self) -> ConnectionStats {
        let connections = self.connections.read().await;
        let suspicious = self.suspicious_ips.read().await;
        
        let total_connections: u32 = connections.values().map(|info| info.active_connections).sum();
        let total_messages: u64 = connections.values().map(|info| info.total_messages).sum();
        let blocked_ips = suspicious.values().filter(|activity| activity.is_blocked).count();
        
        ConnectionStats {
            total_active_connections: total_connections,
            unique_ips: connections.len(),
            total_messages,
            blocked_ips,
            suspicious_ips: suspicious.len(),
        }
    }
}

/// Types of suspicious activity
#[derive(Debug, Clone)]
enum SuspiciousActivityType {
    RapidConnections,
    LargeMessage,
    RateLimitViolation,
}

/// Connection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStats {
    pub total_active_connections: u32,
    pub unique_ips: usize,
    pub total_messages: u64,
    pub blocked_ips: usize,
    pub suspicious_ips: usize,
}

impl SecurityMiddleware {
    pub fn new(enable_rate_limiting: bool, requests_per_minute: u32) -> Self {
        let config = SecurityConfig {
            enable_rate_limiting,
            requests_per_minute,
            ..Default::default()
        };

        Self {
            rate_limiter: Arc::new(RateLimiter::new(config.clone())),
            request_monitor: Arc::new(RequestMonitor::new(config.clone())),
            config,
        }
    }

    pub fn with_config(config: SecurityConfig) -> Self {
        Self {
            rate_limiter: Arc::new(RateLimiter::new(config.clone())),
            request_monitor: Arc::new(RequestMonitor::new(config.clone())),
            config,
        }
    }

    /// Check rate limits for a client
    pub async fn check_rate_limit(&self, client_id: &str) -> Result<(), SecurityError> {
        self.rate_limiter.check_rate_limit(client_id).await
    }

    /// Record a new connection
    pub async fn record_connection(&self, client_ip: &str) -> Result<(), SecurityError> {
        // Check if IP is blocked
        if self.request_monitor.is_ip_blocked(client_ip).await {
            return Err(SecurityError::IpBlocked {
                ip: client_ip.to_string(),
                reason: "IP blocked due to suspicious activity".to_string(),
            });
        }

        self.request_monitor.record_connection(client_ip).await
    }

    /// Record a disconnection
    pub async fn record_disconnection(&self, client_ip: &str) {
        self.request_monitor.record_disconnection(client_ip).await;
    }

    /// Record a message
    pub async fn record_message(&self, client_ip: &str, message_size: usize) -> Result<(), SecurityError> {
        self.request_monitor.record_message(client_ip, message_size).await
    }

    /// Get security statistics
    pub async fn get_security_stats(&self) -> SecurityStats {
        let connection_stats = self.request_monitor.get_connection_stats().await;
        
        SecurityStats {
            connection_stats,
            rate_limiting_enabled: self.config.enable_rate_limiting,
            monitoring_enabled: self.config.enable_monitoring,
            max_message_size: self.config.max_message_size,
            requests_per_minute_limit: self.config.requests_per_minute,
        }
    }

    /// Cleanup old entries
    pub async fn cleanup(&self) {
        self.rate_limiter.cleanup_old_entries().await;
        // Request monitor cleanup would go here if implemented
    }
}

/// Overall security statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityStats {
    pub connection_stats: ConnectionStats,
    pub rate_limiting_enabled: bool,
    pub monitoring_enabled: bool,
    pub max_message_size: usize,
    pub requests_per_minute_limit: u32,
}

/// Security-related errors
#[derive(Debug, thiserror::Error)]
pub enum SecurityError {
    #[error("Rate limit exceeded: {limit} requests per {window_seconds} seconds. Retry after {retry_after} seconds")]
    RateLimitExceeded {
        limit: u32,
        window_seconds: u64,
        retry_after: u64,
    },
    #[error("Too many connections from IP {ip}: limit is {limit}")]
    TooManyConnections {
        ip: String,
        limit: u32,
    },
    #[error("Message too large: {size} bytes, limit is {limit} bytes")]
    MessageTooLarge {
        size: usize,
        limit: usize,
    },
    #[error("IP {ip} is blocked: {reason}")]
    IpBlocked {
        ip: String,
        reason: String,
    },
    #[error("Suspicious activity detected: {reason}")]
    SuspiciousActivity {
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration as TokioDuration};

    #[tokio::test]
    async fn test_rate_limiter() {
        let config = SecurityConfig {
            enable_rate_limiting: true,
            requests_per_minute: 5,
            rate_limit_burst: 2,
            ..Default::default()
        };
        
        let rate_limiter = RateLimiter::new(config);
        
        // Should allow initial requests due to burst tokens
        assert!(rate_limiter.check_rate_limit("client1").await.is_ok());
        assert!(rate_limiter.check_rate_limit("client1").await.is_ok());
        
        // Should still allow a few more due to regular limit
        assert!(rate_limiter.check_rate_limit("client1").await.is_ok());
        assert!(rate_limiter.check_rate_limit("client1").await.is_ok());
        assert!(rate_limiter.check_rate_limit("client1").await.is_ok());
        
        // Should now be rate limited
        assert!(rate_limiter.check_rate_limit("client1").await.is_err());
    }

    #[tokio::test]
    async fn test_request_monitor() {
        let config = SecurityConfig {
            enable_monitoring: true,
            max_connections_per_ip: 2,
            max_message_size: 1000,
            ..Default::default()
        };
        
        let monitor = RequestMonitor::new(config);
        
        // Should allow first connection
        assert!(monitor.record_connection("192.168.1.1").await.is_ok());
        assert!(monitor.record_connection("192.168.1.1").await.is_ok());
        
        // Should reject third connection
        assert!(monitor.record_connection("192.168.1.1").await.is_err());
        
        // Should allow message within size limit
        assert!(monitor.record_message("192.168.1.1", 500).await.is_ok());
        
        // Should reject message over size limit
        assert!(monitor.record_message("192.168.1.1", 2000).await.is_err());
    }

    #[tokio::test]
    async fn test_security_middleware() {
        let middleware = SecurityMiddleware::new(true, 10);
        
        // Test connection recording
        assert!(middleware.record_connection("127.0.0.1").await.is_ok());
        
        // Test rate limiting
        assert!(middleware.check_rate_limit("client1").await.is_ok());
        
        // Test message recording
        assert!(middleware.record_message("127.0.0.1", 100).await.is_ok());
        
        // Test stats
        let stats = middleware.get_security_stats().await;
        assert!(stats.rate_limiting_enabled);
        assert!(stats.monitoring_enabled);
    }

    #[tokio::test]
    async fn test_ip_restrictions() {
        let config = SecurityConfig {
            enable_ip_blacklist: true,
            ip_blacklist: vec!["192.168.1.100".to_string()],
            ..Default::default()
        };
        
        let monitor = RequestMonitor::new(config);
        
        // Should allow normal IP
        assert!(monitor.record_connection("192.168.1.1").await.is_ok());
        
        // Should block blacklisted IP
        assert!(monitor.record_connection("192.168.1.100").await.is_err());
    }
}