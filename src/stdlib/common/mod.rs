//! Common utilities and shared infrastructure for Lyra standard library
//!
//! This module provides shared infrastructure used across multiple stdlib domains:
//! - Unified Complex number implementation
//! - Error handling utilities and conversions
//! - Parameter validation macros
//! - Foreign object utilities and patterns

pub mod complex;
pub mod errors;
pub mod validation;
pub mod foreign_utils;

// Re-export commonly used types
pub use complex::Complex;
pub use errors::{ErrorConversion, UnifiedError};
pub use validation::{validate_args, validate_type, ValidationError};
pub use foreign_utils::ForeignObjectTemplate;