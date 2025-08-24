//! Time Series Analysis Module
//!
//! This module provides comprehensive time series analysis capabilities following the
//! "Take Algorithms for Granted" principle. Includes ARIMA/SARIMA models, forecasting
//! methods, decomposition, stationarity testing, and advanced econometric models.

pub mod core;
pub mod arima;
pub mod forecasting;
// pub mod analysis;
// pub mod advanced;

// Re-export all public functions
pub use core::*;
pub use arima::*;
pub use forecasting::*;
// pub use analysis::*;
// pub use advanced::*;

/// Temporary alias to prefer analytics timeseries as canonical entry
/// Consumers can import `std::timeseries::advanced` to access consolidated APIs.
pub use crate::stdlib::analytics::timeseries as advanced;
