//! Advanced Analytics & Statistics Module
//!
//! This module provides comprehensive statistical analysis, time series analysis,
//! business intelligence, and data mining capabilities for the Lyra symbolic computation engine.
//!
//! # Phase 15B: Advanced Analytics & Statistics Implementation
//!
//! ## Core Architecture
//! - Statistical Analysis: Regression, ANOVA, hypothesis testing, correlation analysis
//! - Time Series Analysis: ARIMA modeling, decomposition, forecasting, anomaly detection
//! - Business Intelligence: KPI calculation, cohort analysis, funnel analysis, retention metrics
//! - Data Mining: Clustering, classification, association rules, ensemble methods
//!
//! ## Foreign Object Pattern
//! All complex analytics types are implemented as Foreign objects following the established
//! pattern to maintain VM simplicity and thread safety.

pub mod statistics;
pub mod timeseries;
pub mod business_intelligence;
pub mod data_mining;

// Re-export key functions for easier access
pub use statistics::*;
pub use timeseries::*;
pub use business_intelligence::*;
pub use data_mining::*;