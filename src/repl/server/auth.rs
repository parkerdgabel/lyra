use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Authentication manager for WebSocket connections
pub struct AuthManager {
    enabled: bool,
    tokens: Arc<RwLock<HashMap<String, AuthToken>>>,
    api_keys: Arc<RwLock<HashMap<String, ApiKey>>>,
    config: AuthConfig,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// Enable JWT token authentication
    pub enable_jwt: bool,
    /// Enable API key authentication
    pub enable_api_keys: bool,
    /// Token expiration time in seconds
    pub token_expiry_seconds: u64,
    /// Maximum number of active tokens
    pub max_tokens: usize,
    /// JWT secret key (in production, this should be loaded from environment)
    pub jwt_secret: String,
    /// Require authentication for all connections
    pub require_auth: bool,
}

impl Default for AuthConfig {
    fn default() -> Self {
        Self {
            enable_jwt: true,
            enable_api_keys: true,
            token_expiry_seconds: 3600, // 1 hour
            max_tokens: 1000,
            jwt_secret: "lyra-websocket-secret-change-in-production".to_string(),
            require_auth: false,
        }
    }
}

/// Authentication token
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthToken {
    /// Token ID
    pub id: String,
    /// User ID
    pub user_id: String,
    /// Token expiration time
    pub expires_at: DateTime<Utc>,
    /// Token permissions
    pub permissions: Vec<Permission>,
    /// Session limits
    pub session_limits: SessionLimits,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: DateTime<Utc>,
}

/// API key for programmatic access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKey {
    /// API key ID
    pub id: String,
    /// API key name
    pub name: String,
    /// Key hash (never store raw keys)
    pub key_hash: String,
    /// User ID
    pub user_id: String,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Session limits
    pub session_limits: SessionLimits,
    /// Created timestamp
    pub created_at: DateTime<Utc>,
    /// Last used timestamp
    pub last_used: DateTime<Utc>,
    /// Whether the key is active
    pub active: bool,
}

/// Permission levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Permission {
    /// Execute code
    Execute,
    /// Read session data
    ReadSession,
    /// Write session data
    WriteSession,
    /// Export sessions
    Export,
    /// Manage multiple sessions
    ManageSessions,
    /// Administrative access
    Admin,
}

/// Session limits for authentication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionLimits {
    /// Maximum concurrent sessions
    pub max_concurrent_sessions: u32,
    /// Maximum session duration in seconds
    pub max_session_duration: u64,
    /// Maximum executions per session
    pub max_executions_per_session: u32,
    /// Rate limit: requests per minute
    pub requests_per_minute: u32,
}

impl Default for SessionLimits {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 5,
            max_session_duration: 7200, // 2 hours
            max_executions_per_session: 1000,
            requests_per_minute: 60,
        }
    }
}

/// JWT claims structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JwtClaims {
    /// Subject (user ID)
    pub sub: String,
    /// Expiration time
    pub exp: u64,
    /// Issued at
    pub iat: u64,
    /// Issuer
    pub iss: String,
    /// Permissions
    pub permissions: Vec<Permission>,
    /// Session limits
    pub session_limits: SessionLimits,
}

impl AuthManager {
    /// Create a new authentication manager
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            tokens: Arc::new(RwLock::new(HashMap::new())),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            config: AuthConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AuthConfig) -> Self {
        Self {
            enabled: config.require_auth,
            tokens: Arc::new(RwLock::new(HashMap::new())),
            api_keys: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Authenticate a connection
    pub async fn authenticate(&self, token: Option<&str>) -> Result<AuthContext, AuthError> {
        if !self.enabled {
            // Authentication disabled, allow all connections
            return Ok(AuthContext::anonymous());
        }

        let token = token.ok_or(AuthError::MissingToken)?;

        // Try JWT authentication first
        if self.config.enable_jwt {
            if let Ok(claims) = self.verify_jwt(token).await {
                return Ok(AuthContext::from_jwt_claims(claims));
            }
        }

        // Try API key authentication
        if self.config.enable_api_keys {
            if let Ok(api_key) = self.verify_api_key(token).await {
                return Ok(AuthContext::from_api_key(api_key));
            }
        }

        Err(AuthError::InvalidToken)
    }

    /// Verify JWT token
    async fn verify_jwt(&self, token: &str) -> Result<JwtClaims, AuthError> {
        // In a real implementation, you would use a proper JWT library like `jsonwebtoken`
        // For now, we'll do a simplified implementation
        
        // Check if token exists in our token store
        let tokens = self.tokens.read().await;
        
        if let Some(auth_token) = tokens.get(token) {
            if auth_token.expires_at > Utc::now() {
                let claims = JwtClaims {
                    sub: auth_token.user_id.clone(),
                    exp: auth_token.expires_at.timestamp() as u64,
                    iat: auth_token.created_at.timestamp() as u64,
                    iss: "lyra-websocket-server".to_string(),
                    permissions: auth_token.permissions.clone(),
                    session_limits: auth_token.session_limits.clone(),
                };
                
                // Update last used timestamp
                drop(tokens);
                let mut tokens = self.tokens.write().await;
                if let Some(token) = tokens.get_mut(token) {
                    token.last_used = Utc::now();
                }
                
                return Ok(claims);
            } else {
                // Token expired, remove it
                drop(tokens);
                let mut tokens = self.tokens.write().await;
                tokens.remove(token);
                return Err(AuthError::TokenExpired);
            }
        }

        Err(AuthError::InvalidToken)
    }

    /// Verify API key
    async fn verify_api_key(&self, key: &str) -> Result<ApiKey, AuthError> {
        let api_keys = self.api_keys.read().await;
        
        // Hash the provided key and compare with stored hashes
        let key_hash = self.hash_api_key(key);
        
        for api_key in api_keys.values() {
            if api_key.key_hash == key_hash && api_key.active {
                let mut result = api_key.clone();
                
                // Update last used timestamp
                drop(api_keys);
                let mut api_keys = self.api_keys.write().await;
                if let Some(key) = api_keys.get_mut(&api_key.id) {
                    key.last_used = Utc::now();
                    result.last_used = Utc::now();
                }
                
                return Ok(result);
            }
        }

        Err(AuthError::InvalidToken)
    }

    /// Create a new authentication token
    pub async fn create_token(&self, user_id: String, permissions: Vec<Permission>) -> Result<String, AuthError> {
        let mut tokens = self.tokens.write().await;
        
        if tokens.len() >= self.config.max_tokens {
            return Err(AuthError::TooManyTokens);
        }

        let token_id = Uuid::new_v4().to_string();
        let now = Utc::now();
        let expires_at = now + chrono::Duration::seconds(self.config.token_expiry_seconds as i64);

        let auth_token = AuthToken {
            id: token_id.clone(),
            user_id,
            expires_at,
            permissions,
            session_limits: SessionLimits::default(),
            created_at: now,
            last_used: now,
        };

        tokens.insert(token_id.clone(), auth_token);
        Ok(token_id)
    }

    /// Create a new API key
    pub async fn create_api_key(
        &self,
        name: String,
        user_id: String,
        permissions: Vec<Permission>,
    ) -> Result<(String, String), AuthError> {
        let key_id = Uuid::new_v4().to_string();
        let raw_key = format!("lyra_{}_{}", key_id, Uuid::new_v4().simple());
        let key_hash = self.hash_api_key(&raw_key);

        let api_key = ApiKey {
            id: key_id.clone(),
            name,
            key_hash,
            user_id,
            permissions,
            session_limits: SessionLimits::default(),
            created_at: Utc::now(),
            last_used: Utc::now(),
            active: true,
        };

        let mut api_keys = self.api_keys.write().await;
        api_keys.insert(key_id.clone(), api_key);

        Ok((key_id, raw_key))
    }

    /// Revoke a token
    pub async fn revoke_token(&self, token: &str) -> Result<(), AuthError> {
        let mut tokens = self.tokens.write().await;
        tokens.remove(token).ok_or(AuthError::TokenNotFound)?;
        Ok(())
    }

    /// Revoke an API key
    pub async fn revoke_api_key(&self, key_id: &str) -> Result<(), AuthError> {
        let mut api_keys = self.api_keys.write().await;
        if let Some(api_key) = api_keys.get_mut(key_id) {
            api_key.active = false;
            Ok(())
        } else {
            Err(AuthError::KeyNotFound)
        }
    }

    /// Clean up expired tokens
    pub async fn cleanup_expired_tokens(&self) {
        let mut tokens = self.tokens.write().await;
        let now = Utc::now();
        
        tokens.retain(|_, token| token.expires_at > now);
    }

    /// Hash an API key (simplified implementation)
    fn hash_api_key(&self, key: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Get authentication statistics
    pub async fn get_auth_stats(&self) -> AuthStats {
        let tokens = self.tokens.read().await;
        let api_keys = self.api_keys.read().await;
        
        AuthStats {
            active_tokens: tokens.len(),
            active_api_keys: api_keys.values().filter(|k| k.active).count(),
            total_api_keys: api_keys.len(),
            expired_tokens: tokens.values().filter(|t| t.expires_at <= Utc::now()).count(),
        }
    }
}

/// Authentication context for a connection
#[derive(Debug, Clone)]
pub struct AuthContext {
    pub user_id: String,
    pub permissions: Vec<Permission>,
    pub session_limits: SessionLimits,
    pub is_authenticated: bool,
}

impl AuthContext {
    /// Create an anonymous context (no authentication)
    pub fn anonymous() -> Self {
        Self {
            user_id: "anonymous".to_string(),
            permissions: vec![Permission::Execute, Permission::ReadSession, Permission::Export],
            session_limits: SessionLimits::default(),
            is_authenticated: false,
        }
    }

    /// Create context from JWT claims
    pub fn from_jwt_claims(claims: JwtClaims) -> Self {
        Self {
            user_id: claims.sub,
            permissions: claims.permissions,
            session_limits: claims.session_limits,
            is_authenticated: true,
        }
    }

    /// Create context from API key
    pub fn from_api_key(api_key: ApiKey) -> Self {
        Self {
            user_id: api_key.user_id,
            permissions: api_key.permissions,
            session_limits: api_key.session_limits,
            is_authenticated: true,
        }
    }

    /// Check if the context has a specific permission
    pub fn has_permission(&self, permission: &Permission) -> bool {
        self.permissions.contains(permission)
    }

    /// Check if the context can execute code
    pub fn can_execute(&self) -> bool {
        self.has_permission(&Permission::Execute)
    }

    /// Check if the context can export sessions
    pub fn can_export(&self) -> bool {
        self.has_permission(&Permission::Export)
    }

    /// Check if the context can manage sessions
    pub fn can_manage_sessions(&self) -> bool {
        self.has_permission(&Permission::ManageSessions)
    }
}

/// Authentication statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthStats {
    pub active_tokens: usize,
    pub active_api_keys: usize,
    pub total_api_keys: usize,
    pub expired_tokens: usize,
}

/// Authentication errors
#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("Missing authentication token")]
    MissingToken,
    #[error("Invalid authentication token")]
    InvalidToken,
    #[error("Authentication token has expired")]
    TokenExpired,
    #[error("Too many active tokens")]
    TooManyTokens,
    #[error("Token not found")]
    TokenNotFound,
    #[error("API key not found")]
    KeyNotFound,
    #[error("Permission denied")]
    PermissionDenied,
    #[error("Rate limit exceeded")]
    RateLimitExceeded,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_manager_disabled() {
        let auth_manager = AuthManager::new(false);
        let result = auth_manager.authenticate(None).await;
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert!(!context.is_authenticated);
        assert_eq!(context.user_id, "anonymous");
    }

    #[tokio::test]
    async fn test_token_creation_and_verification() {
        let auth_manager = AuthManager::new(true);
        
        let permissions = vec![Permission::Execute, Permission::ReadSession];
        let token = auth_manager.create_token("user123".to_string(), permissions.clone()).await.unwrap();
        
        let result = auth_manager.authenticate(Some(&token)).await;
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert!(context.is_authenticated);
        assert_eq!(context.user_id, "user123");
        assert!(context.has_permission(&Permission::Execute));
        assert!(context.has_permission(&Permission::ReadSession));
        assert!(!context.has_permission(&Permission::Admin));
    }

    #[tokio::test]
    async fn test_api_key_creation_and_verification() {
        let auth_manager = AuthManager::new(true);
        
        let permissions = vec![Permission::Execute, Permission::Export];
        let (key_id, raw_key) = auth_manager.create_api_key(
            "test-key".to_string(),
            "user456".to_string(),
            permissions.clone(),
        ).await.unwrap();
        
        let result = auth_manager.authenticate(Some(&raw_key)).await;
        assert!(result.is_ok());
        
        let context = result.unwrap();
        assert!(context.is_authenticated);
        assert_eq!(context.user_id, "user456");
        assert!(context.can_execute());
        assert!(context.can_export());
    }

    #[tokio::test]
    async fn test_token_revocation() {
        let auth_manager = AuthManager::new(true);
        
        let token = auth_manager.create_token("user789".to_string(), vec![Permission::Execute]).await.unwrap();
        
        // Token should work initially
        let result = auth_manager.authenticate(Some(&token)).await;
        assert!(result.is_ok());
        
        // Revoke the token
        auth_manager.revoke_token(&token).await.unwrap();
        
        // Token should no longer work
        let result = auth_manager.authenticate(Some(&token)).await;
        assert!(result.is_err());
    }

    #[test]
    fn test_auth_context_permissions() {
        let context = AuthContext {
            user_id: "test".to_string(),
            permissions: vec![Permission::Execute, Permission::ReadSession],
            session_limits: SessionLimits::default(),
            is_authenticated: true,
        };
        
        assert!(context.can_execute());
        assert!(!context.can_export());
        assert!(!context.can_manage_sessions());
        assert!(context.has_permission(&Permission::ReadSession));
    }
}