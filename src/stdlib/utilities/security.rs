//! Security Module for Lyra Utilities
//!
//! This module provides comprehensive security functionality including:
//! - Cryptographic operations (hashing, encryption, key management)
//! - Security wrappers for stdlib functions
//! - Input validation and resource monitoring
//! - Audit logging and security event tracking

pub mod crypto;
pub mod wrappers;

// Re-export main functions for convenience
pub use crypto::*;
pub use wrappers::*;

use crate::vm::{Value, VmResult};

/// Register all security functions
pub fn register_security_functions() -> std::collections::HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut functions = std::collections::HashMap::new();
    
    // Cryptographic functions from crypto module
    functions.extend(crypto::register_crypto_functions());
    
    // Security wrapper functions (currently provides infrastructure only)
    functions.extend(wrappers::register_security_wrapper_functions());
    
    functions
}