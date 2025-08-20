//! Rate limiting implementation for DoS prevention

use super::{SecurityError, SecurityResult, SecurityConfig};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Token bucket for rate limiting
#[derive(Debug)]
struct TokenBucket {
    capacity: u64,
    tokens: u64,
    refill_rate: u64,
    last_refill: Instant,
    window_size: Duration,
}

impl TokenBucket {
    fn new(capacity: u64, refill_rate: u64, window_size: Duration) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
            window_size,
        }
    }
    
    fn try_consume(&mut self, tokens: u64) -> bool {
        self.refill();
        
        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }
    
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill);
        
        if elapsed >= self.window_size {
            // Refill bucket
            let new_tokens = (elapsed.as_secs() * self.refill_rate).min(self.capacity);
            self.tokens = (self.tokens + new_tokens).min(self.capacity);
            self.last_refill = now;
        }
    }
    
    fn get_wait_time(&mut self, tokens: u64) -> Option<Duration> {
        self.refill();
        
        if self.tokens >= tokens {
            None
        } else {
            let needed_tokens = tokens - self.tokens;
            let wait_seconds = (needed_tokens + self.refill_rate - 1) / self.refill_rate;
            Some(Duration::from_secs(wait_seconds))
        }
    }
}

/// Rate limiter using token bucket algorithm
pub struct RateLimiter {
    global_bucket: Arc<RwLock<TokenBucket>>,
    operation_buckets: Arc<RwLock<HashMap<String, TokenBucket>>>,
    user_buckets: Arc<RwLock<HashMap<String, HashMap<String, TokenBucket>>>>,
    config: SecurityConfig,
}

impl RateLimiter {
    pub fn new(config: &SecurityConfig) -> SecurityResult<Self> {
        let global_bucket = Arc::new(RwLock::new(TokenBucket::new(
            config.global_rate_limit,
            config.global_rate_limit,
            Duration::from_secs(1),
        )));
        
        let mut operation_buckets = HashMap::new();
        for (operation, limit) in &config.operation_rate_limits {
            operation_buckets.insert(
                operation.clone(),
                TokenBucket::new(*limit, *limit, Duration::from_secs(1))
            );
        }
        
        Ok(Self {
            global_bucket,
            operation_buckets: Arc::new(RwLock::new(operation_buckets)),
            user_buckets: Arc::new(RwLock::new(HashMap::new())),
            config: config.clone(),
        })
    }
    
    pub fn check_rate(&self, operation: &str, user_id: Option<&str>) -> SecurityResult<()> {
        // Check global rate limit
        {
            let mut global_bucket = self.global_bucket.write().unwrap();
            if !global_bucket.try_consume(1) {
                return Err(SecurityError::RateLimitExceeded {
                    operation: "global".to_string(),
                    limit: self.config.global_rate_limit,
                    window: 1,
                });
            }
        }
        
        // Check operation-specific rate limit
        {
            let mut operation_buckets = self.operation_buckets.write().unwrap();
            if let Some(bucket) = operation_buckets.get_mut(operation) {
                if !bucket.try_consume(1) {
                    let limit = self.config.operation_rate_limits.get(operation).copied().unwrap_or(0);
                    return Err(SecurityError::RateLimitExceeded {
                        operation: operation.to_string(),
                        limit,
                        window: 1,
                    });
                }
            }
        }
        
        // Check user-specific rate limit if user_id provided
        if let Some(user_id) = user_id {
            let mut user_buckets = self.user_buckets.write().unwrap();
            let user_operations = user_buckets.entry(user_id.to_string()).or_insert_with(HashMap::new);
            
            let user_limit = self.config.operation_rate_limits.get(operation).copied().unwrap_or(100);
            let bucket = user_operations.entry(operation.to_string()).or_insert_with(|| {
                TokenBucket::new(user_limit, user_limit, Duration::from_secs(1))
            });
            
            if !bucket.try_consume(1) {
                return Err(SecurityError::RateLimitExceeded {
                    operation: format!("user_{}_{}", user_id, operation),
                    limit: user_limit,
                    window: 1,
                });
            }
        }
        
        Ok(())
    }
    
    pub fn get_wait_time(&self, operation: &str, user_id: Option<&str>) -> Option<Duration> {
        // Check global bucket first
        let global_wait = {
            let mut global_bucket = self.global_bucket.write().unwrap();
            global_bucket.get_wait_time(1)
        };
        
        if global_wait.is_some() {
            return global_wait;
        }
        
        // Check operation bucket
        let operation_wait = {
            let mut operation_buckets = self.operation_buckets.write().unwrap();
            operation_buckets.get_mut(operation)
                .and_then(|bucket| bucket.get_wait_time(1))
        };
        
        if operation_wait.is_some() {
            return operation_wait;
        }
        
        // Check user bucket
        if let Some(user_id) = user_id {
            let mut user_buckets = self.user_buckets.write().unwrap();
            if let Some(user_operations) = user_buckets.get_mut(user_id) {
                if let Some(bucket) = user_operations.get_mut(operation) {
                    return bucket.get_wait_time(1);
                }
            }
        }
        
        None
    }
    
    pub fn reset_limits(&self) -> SecurityResult<()> {
        // Reset global bucket
        {
            let mut global_bucket = self.global_bucket.write().unwrap();
            *global_bucket = TokenBucket::new(
                self.config.global_rate_limit,
                self.config.global_rate_limit,
                Duration::from_secs(1),
            );
        }
        
        // Reset operation buckets
        {
            let mut operation_buckets = self.operation_buckets.write().unwrap();
            for (operation, limit) in &self.config.operation_rate_limits {
                operation_buckets.insert(
                    operation.clone(),
                    TokenBucket::new(*limit, *limit, Duration::from_secs(1))
                );
            }
        }
        
        // Reset user buckets
        {
            let mut user_buckets = self.user_buckets.write().unwrap();
            user_buckets.clear();
        }
        
        Ok(())
    }
    
    pub fn get_bucket_status(&self, operation: &str, user_id: Option<&str>) -> (u64, u64) {
        if let Some(user_id) = user_id {
            let user_buckets = self.user_buckets.read().unwrap();
            if let Some(user_operations) = user_buckets.get(user_id) {
                if let Some(bucket) = user_operations.get(operation) {
                    return (bucket.tokens, bucket.capacity);
                }
            }
        }
        
        let operation_buckets = self.operation_buckets.read().unwrap();
        if let Some(bucket) = operation_buckets.get(operation) {
            (bucket.tokens, bucket.capacity)
        } else {
            (0, 0)
        }
    }
}

/// Rate limiting macro for convenient use in stdlib functions
#[macro_export]
macro_rules! check_rate_limit {
    ($security_manager:expr, $operation:expr) => {
        $security_manager.check_rate_limit($operation, None)?;
    };
    ($security_manager:expr, $operation:expr, $user_id:expr) => {
        $security_manager.check_rate_limit($operation, Some($user_id))?;
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    
    #[test]
    fn test_token_bucket_basic() {
        let mut bucket = TokenBucket::new(10, 10, Duration::from_secs(1));
        
        // Should be able to consume initial tokens
        assert!(bucket.try_consume(5));
        assert!(bucket.try_consume(5));
        assert!(!bucket.try_consume(1)); // Bucket is empty
    }
    
    #[test]
    fn test_token_bucket_refill() {
        let mut bucket = TokenBucket::new(10, 10, Duration::from_millis(100));
        
        // Consume all tokens
        assert!(bucket.try_consume(10));
        assert!(!bucket.try_consume(1));
        
        // Wait for refill
        thread::sleep(Duration::from_millis(150));
        assert!(bucket.try_consume(10));
    }
    
    #[test]
    fn test_rate_limiter_global() {
        let mut config = SecurityConfig::default();
        config.global_rate_limit = 2;
        
        let limiter = RateLimiter::new(&config).unwrap();
        
        assert!(limiter.check_rate("test_op", None).is_ok());
        assert!(limiter.check_rate("test_op", None).is_ok());
        assert!(limiter.check_rate("test_op", None).is_err()); // Should exceed global limit
    }
    
    #[test]
    fn test_rate_limiter_operation_specific() {
        let mut config = SecurityConfig::default();
        config.operation_rate_limits.insert("limited_op".to_string(), 1);
        
        let limiter = RateLimiter::new(&config).unwrap();
        
        assert!(limiter.check_rate("limited_op", None).is_ok());
        assert!(limiter.check_rate("limited_op", None).is_err()); // Should exceed operation limit
    }
    
    #[test]
    fn test_rate_limiter_user_specific() {
        let mut config = SecurityConfig::default();
        config.operation_rate_limits.insert("user_op".to_string(), 2);
        
        let limiter = RateLimiter::new(&config).unwrap();
        
        // User1 should be able to use their quota
        assert!(limiter.check_rate("user_op", Some("user1")).is_ok());
        assert!(limiter.check_rate("user_op", Some("user1")).is_ok());
        assert!(limiter.check_rate("user_op", Some("user1")).is_err());
        
        // User2 should have their own quota
        assert!(limiter.check_rate("user_op", Some("user2")).is_ok());
        assert!(limiter.check_rate("user_op", Some("user2")).is_ok());
        assert!(limiter.check_rate("user_op", Some("user2")).is_err());
    }
    
    #[test]
    fn test_get_wait_time() {
        let mut config = SecurityConfig::default();
        config.operation_rate_limits.insert("test_op".to_string(), 1);
        
        let limiter = RateLimiter::new(&config).unwrap();
        
        // Consume the token
        assert!(limiter.check_rate("test_op", None).is_ok());
        
        // Should need to wait
        let wait_time = limiter.get_wait_time("test_op", None);
        assert!(wait_time.is_some());
        assert!(wait_time.unwrap() <= Duration::from_secs(1));
    }
    
    #[test]
    fn test_reset_limits() {
        let mut config = SecurityConfig::default();
        config.global_rate_limit = 1;
        
        let limiter = RateLimiter::new(&config).unwrap();
        
        // Consume the token
        assert!(limiter.check_rate("test_op", None).is_ok());
        assert!(limiter.check_rate("test_op", None).is_err());
        
        // Reset and try again
        assert!(limiter.reset_limits().is_ok());
        assert!(limiter.check_rate("test_op", None).is_ok());
    }
}