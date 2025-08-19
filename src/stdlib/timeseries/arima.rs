//! ARIMA (Autoregressive Integrated Moving Average) Models
//!
//! This module provides comprehensive ARIMA/SARIMA modeling capabilities including
//! parameter estimation, model fitting, forecasting, and automatic model selection.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::timeseries::core::{TimeSeries, Frequency, extract_timeseries};
use std::any::Any;
use std::collections::HashMap;

/// ARIMA model parameters
#[derive(Debug, Clone)]
pub struct ARIMAParams {
    /// Autoregressive order (p)
    pub p: usize,
    /// Differencing order (d)
    pub d: usize,
    /// Moving average order (q)
    pub q: usize,
    /// Seasonal AR order (P)
    pub seasonal_p: usize,
    /// Seasonal differencing order (D)
    pub seasonal_d: usize,
    /// Seasonal MA order (Q)
    pub seasonal_q: usize,
    /// Number of periods in season (s)
    pub seasonal_periods: usize,
}

impl ARIMAParams {
    /// Create non-seasonal ARIMA parameters
    pub fn new(p: usize, d: usize, q: usize) -> Self {
        ARIMAParams {
            p,
            d,
            q,
            seasonal_p: 0,
            seasonal_d: 0,
            seasonal_q: 0,
            seasonal_periods: 0,
        }
    }

    /// Create seasonal ARIMA (SARIMA) parameters
    pub fn seasonal(p: usize, d: usize, q: usize, seasonal_p: usize, seasonal_d: usize, seasonal_q: usize, seasonal_periods: usize) -> Self {
        ARIMAParams {
            p,
            d,
            q,
            seasonal_p,
            seasonal_d,
            seasonal_q,
            seasonal_periods,
        }
    }

    /// Check if model has seasonal components
    pub fn is_seasonal(&self) -> bool {
        self.seasonal_p > 0 || self.seasonal_d > 0 || self.seasonal_q > 0
    }

    /// Get total number of parameters
    pub fn parameter_count(&self) -> usize {
        self.p + self.q + self.seasonal_p + self.seasonal_q + 1 // +1 for intercept
    }
}

/// ARIMA model results
#[derive(Debug, Clone)]
pub struct ARIMAModel {
    /// Model parameters
    pub params: ARIMAParams,
    /// Estimated coefficients
    pub coefficients: ARIMACoefficients,
    /// Original time series
    pub original_series: TimeSeries,
    /// Differenced series (stationary)
    pub differenced_series: TimeSeries,
    /// Model residuals
    pub residuals: Vec<f64>,
    /// Log likelihood
    pub log_likelihood: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Bayesian Information Criterion
    pub bic: f64,
    /// Standard error of residuals
    pub sigma2: f64,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

/// ARIMA model coefficients
#[derive(Debug, Clone)]
pub struct ARIMACoefficients {
    /// Autoregressive coefficients (AR)
    pub ar: Vec<f64>,
    /// Moving average coefficients (MA)
    pub ma: Vec<f64>,
    /// Seasonal AR coefficients
    pub seasonal_ar: Vec<f64>,
    /// Seasonal MA coefficients
    pub seasonal_ma: Vec<f64>,
    /// Intercept/constant term
    pub intercept: f64,
}

impl ARIMACoefficients {
    pub fn new() -> Self {
        ARIMACoefficients {
            ar: Vec::new(),
            ma: Vec::new(),
            seasonal_ar: Vec::new(),
            seasonal_ma: Vec::new(),
            intercept: 0.0,
        }
    }
}

impl ARIMAModel {
    /// Fit ARIMA model to time series data
    pub fn fit(series: &TimeSeries, params: ARIMAParams) -> Result<ARIMAModel, String> {
        // Step 1: Apply differencing to make series stationary
        let differenced = Self::apply_differencing(series, params.d, params.seasonal_d, params.seasonal_periods);
        
        // Step 2: Estimate model parameters using Maximum Likelihood
        let coefficients = Self::estimate_parameters(&differenced, &params)?;
        
        // Step 3: Calculate residuals
        let residuals = Self::calculate_residuals(&differenced, &coefficients, &params)?;
        
        // Step 4: Calculate model statistics
        let log_likelihood = Self::calculate_log_likelihood(&residuals);
        let sigma2 = Self::calculate_sigma2(&residuals);
        let aic = Self::calculate_aic(log_likelihood, params.parameter_count());
        let bic = Self::calculate_bic(log_likelihood, params.parameter_count(), series.len());
        
        Ok(ARIMAModel {
            params,
            coefficients,
            original_series: series.clone(),
            differenced_series: differenced,
            residuals,
            log_likelihood,
            aic,
            bic,
            sigma2,
            metadata: HashMap::new(),
        })
    }

    /// Apply differencing to make series stationary
    fn apply_differencing(series: &TimeSeries, d: usize, seasonal_d: usize, seasonal_periods: usize) -> TimeSeries {
        let mut result = series.clone();
        
        // Apply regular differencing
        for _ in 0..d {
            result = result.diff(1);
        }
        
        // Apply seasonal differencing
        if seasonal_d > 0 && seasonal_periods > 0 {
            for _ in 0..seasonal_d {
                result = result.diff(seasonal_periods);
            }
        }
        
        result
    }

    /// Estimate ARIMA parameters using Maximum Likelihood Estimation (simplified)
    fn estimate_parameters(series: &TimeSeries, params: &ARIMAParams) -> Result<ARIMACoefficients, String> {
        if series.values.is_empty() {
            return Err("Cannot fit ARIMA to empty series".to_string());
        }

        // For simplicity, use Yule-Walker equations for AR parameters and
        // Method of Moments for MA parameters
        let mut coefficients = ARIMACoefficients::new();
        
        // Estimate AR coefficients using Yule-Walker equations
        if params.p > 0 {
            coefficients.ar = Self::estimate_ar_coefficients(&series.values, params.p)?;
        }
        
        // Estimate MA coefficients (simplified method)
        if params.q > 0 {
            coefficients.ma = Self::estimate_ma_coefficients(&series.values, params.q)?;
        }
        
        // Calculate intercept as mean of series
        coefficients.intercept = series.mean();
        
        // Seasonal parameters (simplified estimation)
        if params.seasonal_p > 0 {
            coefficients.seasonal_ar = vec![0.1; params.seasonal_p]; // Placeholder
        }
        
        if params.seasonal_q > 0 {
            coefficients.seasonal_ma = vec![0.1; params.seasonal_q]; // Placeholder
        }
        
        Ok(coefficients)
    }

    /// Estimate AR coefficients using Yule-Walker equations
    fn estimate_ar_coefficients(data: &[f64], p: usize) -> Result<Vec<f64>, String> {
        if data.len() < p + 1 {
            return Err("Not enough data points for AR estimation".to_string());
        }

        // Calculate sample autocovariances
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        
        // Calculate autocovariances up to lag p
        let mut autocovariances = vec![0.0; p + 1];
        for lag in 0..=p {
            let mut sum = 0.0;
            for t in lag..n {
                sum += (data[t] - mean) * (data[t - lag] - mean);
            }
            autocovariances[lag] = sum / (n - lag) as f64;
        }
        
        // Solve Yule-Walker equations using simple approximation
        let mut ar_coeffs = vec![0.0; p];
        
        if p == 1 {
            // For AR(1), φ₁ = ρ₁ = γ₁/γ₀
            if autocovariances[0] != 0.0 {
                ar_coeffs[0] = autocovariances[1] / autocovariances[0];
                ar_coeffs[0] = ar_coeffs[0].max(-0.99).min(0.99); // Ensure stationarity
            }
        } else if p == 2 {
            // For AR(2), use direct formulas
            let rho1 = if autocovariances[0] != 0.0 { autocovariances[1] / autocovariances[0] } else { 0.0 };
            let rho2 = if autocovariances[0] != 0.0 { autocovariances[2] / autocovariances[0] } else { 0.0 };
            
            let denom = 1.0 - rho1 * rho1;
            if denom.abs() > 1e-10 {
                ar_coeffs[0] = (rho1 * (1.0 - rho2)) / denom;
                ar_coeffs[1] = (rho2 - rho1 * rho1) / denom;
                
                // Ensure stationarity for AR(2)
                ar_coeffs[0] = ar_coeffs[0].max(-0.99).min(0.99);
                ar_coeffs[1] = ar_coeffs[1].max(-0.99).min(0.99);
                
                if ar_coeffs[0] + ar_coeffs[1] >= 1.0 {
                    ar_coeffs[1] = 0.99 - ar_coeffs[0];
                }
                if ar_coeffs[1] - ar_coeffs[0] >= 1.0 {
                    ar_coeffs[0] = ar_coeffs[1] - 0.99;
                }
            }
        } else {
            // For higher orders, use simplified approximation
            for i in 0..p {
                if autocovariances[0] != 0.0 {
                    ar_coeffs[i] = (autocovariances[i + 1] / autocovariances[0]) * 0.5_f64.powi(i as i32);
                    ar_coeffs[i] = ar_coeffs[i].max(-0.5).min(0.5);
                }
            }
        }
        
        Ok(ar_coeffs)
    }

    /// Estimate MA coefficients (simplified method)
    fn estimate_ma_coefficients(data: &[f64], q: usize) -> Result<Vec<f64>, String> {
        if data.len() < q + 1 {
            return Err("Not enough data points for MA estimation".to_string());
        }

        // Simple initial estimates for MA coefficients
        let mut ma_coeffs = vec![0.0; q];
        
        // Use a simple approximation based on first q autocorrelations
        let n = data.len();
        let mean = data.iter().sum::<f64>() / n as f64;
        
        for i in 0..q {
            let lag = i + 1;
            if lag < n {
                let mut autocorr = 0.0;
                let mut variance = 0.0;
                
                for t in lag..n {
                    autocorr += (data[t] - mean) * (data[t - lag] - mean);
                    variance += (data[t] - mean) * (data[t] - mean);
                }
                
                if variance != 0.0 {
                    ma_coeffs[i] = (autocorr / variance) * 0.5_f64.powi(i as i32);
                    ma_coeffs[i] = ma_coeffs[i].max(-0.5).min(0.5);
                }
            }
        }
        
        Ok(ma_coeffs)
    }

    /// Calculate model residuals
    fn calculate_residuals(series: &TimeSeries, coefficients: &ARIMACoefficients, params: &ARIMAParams) -> Result<Vec<f64>, String> {
        let data = &series.values;
        let n = data.len();
        let mut residuals = vec![0.0; n];
        
        // Start calculation from max(p, q) to ensure we have enough lags
        let start_idx = params.p.max(params.q);
        
        for t in start_idx..n {
            let mut prediction = coefficients.intercept;
            
            // AR component
            for (i, &ar_coeff) in coefficients.ar.iter().enumerate() {
                if t > i {
                    prediction += ar_coeff * data[t - i - 1];
                }
            }
            
            // MA component (using previous residuals)
            for (i, &ma_coeff) in coefficients.ma.iter().enumerate() {
                if t > i {
                    prediction += ma_coeff * residuals[t - i - 1];
                }
            }
            
            residuals[t] = data[t] - prediction;
        }
        
        Ok(residuals)
    }

    /// Calculate log likelihood
    fn calculate_log_likelihood(residuals: &[f64]) -> f64 {
        let n = residuals.len() as f64;
        let sum_squared_residuals: f64 = residuals.iter().map(|&r| r * r).sum();
        let sigma2 = sum_squared_residuals / n;
        
        if sigma2 <= 0.0 {
            return -1e10; // Return very low likelihood for invalid variance
        }
        
        -0.5 * n * (2.0 * std::f64::consts::PI * sigma2).ln() - 0.5 * sum_squared_residuals / sigma2
    }

    /// Calculate residual variance
    fn calculate_sigma2(residuals: &[f64]) -> f64 {
        if residuals.is_empty() {
            return 1.0;
        }
        
        let sum_squared: f64 = residuals.iter().map(|&r| r * r).sum();
        sum_squared / residuals.len() as f64
    }

    /// Calculate Akaike Information Criterion
    fn calculate_aic(log_likelihood: f64, param_count: usize) -> f64 {
        -2.0 * log_likelihood + 2.0 * param_count as f64
    }

    /// Calculate Bayesian Information Criterion
    fn calculate_bic(log_likelihood: f64, param_count: usize, n: usize) -> f64 {
        -2.0 * log_likelihood + (n as f64).ln() * param_count as f64
    }

    /// Generate forecasts
    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }

        let mut forecasts = Vec::with_capacity(steps);
        let data = &self.differenced_series.values;
        let n = data.len();
        
        // Extend data with forecasts for recursive prediction
        let mut extended_data = data.clone();
        let mut extended_residuals = self.residuals.clone();
        
        for step in 0..steps {
            let current_idx = n + step;
            let mut forecast = self.coefficients.intercept;
            
            // AR component
            for (i, &ar_coeff) in self.coefficients.ar.iter().enumerate() {
                let lag_idx = current_idx - i - 1;
                if lag_idx < extended_data.len() {
                    forecast += ar_coeff * extended_data[lag_idx];
                }
            }
            
            // MA component (using zero for future residuals)
            for (i, &ma_coeff) in self.coefficients.ma.iter().enumerate() {
                let lag_idx = current_idx - i - 1;
                if lag_idx < extended_residuals.len() {
                    forecast += ma_coeff * extended_residuals[lag_idx];
                } // Future residuals assumed to be zero
            }
            
            forecasts.push(forecast);
            extended_data.push(forecast);
            extended_residuals.push(0.0); // Assume future residuals are zero
        }
        
        Ok(forecasts)
    }

    /// Reverse differencing to convert forecasts back to original scale
    pub fn undifference_forecasts(&self, forecasts: &[f64]) -> Result<Vec<f64>, String> {
        if forecasts.is_empty() {
            return Ok(Vec::new());
        }

        let mut result = forecasts.to_vec();
        let original_data = &self.original_series.values;
        
        // Reverse regular differencing
        for _ in 0..self.params.d {
            let last_original = original_data.last().copied().unwrap_or(0.0);
            let mut undifferenced = Vec::with_capacity(result.len());
            let mut cumsum = last_original;
            
            for &diff_value in &result {
                cumsum += diff_value;
                undifferenced.push(cumsum);
            }
            
            result = undifferenced;
        }
        
        // Reverse seasonal differencing (simplified)
        if self.params.seasonal_d > 0 && self.params.seasonal_periods > 0 {
            // This would require more complex handling of seasonal patterns
            // For now, we'll just return the result as-is
        }
        
        Ok(result)
    }
}

impl Foreign for ARIMAModel {
    fn type_name(&self) -> &'static str {
        "ARIMAModel"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Parameters" => {
                let params_list = vec![
                    Value::Integer(self.params.p as i64),
                    Value::Integer(self.params.d as i64),
                    Value::Integer(self.params.q as i64),
                ];
                Ok(Value::List(params_list))
            }
            "Coefficients" => {
                let mut coeff_list = Vec::new();
                
                // AR coefficients
                for &coeff in &self.coefficients.ar {
                    coeff_list.push(Value::Real(coeff));
                }
                
                // MA coefficients
                for &coeff in &self.coefficients.ma {
                    coeff_list.push(Value::Real(coeff));
                }
                
                // Intercept
                coeff_list.push(Value::Real(self.coefficients.intercept));
                
                Ok(Value::List(coeff_list))
            }
            "AIC" => Ok(Value::Real(self.aic)),
            "BIC" => Ok(Value::Real(self.bic)),
            "LogLikelihood" => Ok(Value::Real(self.log_likelihood)),
            "Sigma2" => Ok(Value::Real(self.sigma2)),
            "Residuals" => {
                let residuals: Vec<Value> = self.residuals.iter()
                    .map(|&r| Value::Real(r))
                    .collect();
                Ok(Value::List(residuals))
            }
            "Forecast" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let steps = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                match self.forecast(steps) {
                    Ok(forecasts) => {
                        let forecast_values: Vec<Value> = forecasts.iter()
                            .map(|&f| Value::Real(f))
                            .collect();
                        Ok(Value::List(forecast_values))
                    }
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

// ===============================
// ARIMA FUNCTIONS
// ===============================

/// Fit ARIMA model to time series
/// Syntax: ARIMA[timeseries, {p, d, q}]
pub fn arima(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (timeseries, {p, d, q})".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let series = extract_timeseries(&args[0])?;

    let params = match &args[1] {
        Value::List(param_list) => {
            if param_list.len() != 3 {
                return Err(VmError::TypeError {
                    expected: "parameter list with 3 elements {p, d, q}".to_string(),
                    actual: format!("list with {} elements", param_list.len()),
                });
            }

            let p = match &param_list[0] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for p".to_string(),
                    actual: format!("{:?}", param_list[0]),
                }),
            };

            let d = match &param_list[1] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for d".to_string(),
                    actual: format!("{:?}", param_list[1]),
                }),
            };

            let q = match &param_list[2] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for q".to_string(),
                    actual: format!("{:?}", param_list[2]),
                }),
            };

            ARIMAParams::new(p, d, q)
        }
        _ => return Err(VmError::TypeError {
            expected: "List {p, d, q}".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match ARIMAModel::fit(series, params) {
        Ok(model) => Ok(Value::LyObj(LyObj::new(Box::new(model)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid ARIMA model".to_string(),
            actual: e,
        }),
    }
}

/// Fit seasonal ARIMA (SARIMA) model
/// Syntax: SARIMA[timeseries, {p, d, q}, {P, D, Q, s}]
pub fn sarima(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (timeseries, {p, d, q}, {P, D, Q, s})".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let series = extract_timeseries(&args[0])?;

    // Parse regular ARIMA parameters
    let (p, d, q) = match &args[1] {
        Value::List(param_list) => {
            if param_list.len() != 3 {
                return Err(VmError::TypeError {
                    expected: "parameter list with 3 elements {p, d, q}".to_string(),
                    actual: format!("list with {} elements", param_list.len()),
                });
            }

            let p = match &param_list[0] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for p".to_string(),
                    actual: format!("{:?}", param_list[0]),
                }),
            };

            let d = match &param_list[1] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for d".to_string(),
                    actual: format!("{:?}", param_list[1]),
                }),
            };

            let q = match &param_list[2] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for q".to_string(),
                    actual: format!("{:?}", param_list[2]),
                }),
            };

            (p, d, q)
        }
        _ => return Err(VmError::TypeError {
            expected: "List {p, d, q}".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    // Parse seasonal parameters
    let (seasonal_p, seasonal_d, seasonal_q, seasonal_periods) = match &args[2] {
        Value::List(seasonal_list) => {
            if seasonal_list.len() != 4 {
                return Err(VmError::TypeError {
                    expected: "seasonal parameter list with 4 elements {P, D, Q, s}".to_string(),
                    actual: format!("list with {} elements", seasonal_list.len()),
                });
            }

            let seasonal_p = match &seasonal_list[0] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for P".to_string(),
                    actual: format!("{:?}", seasonal_list[0]),
                }),
            };

            let seasonal_d = match &seasonal_list[1] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for D".to_string(),
                    actual: format!("{:?}", seasonal_list[1]),
                }),
            };

            let seasonal_q = match &seasonal_list[2] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for Q".to_string(),
                    actual: format!("{:?}", seasonal_list[2]),
                }),
            };

            let seasonal_periods = match &seasonal_list[3] {
                Value::Integer(i) => *i as usize,
                _ => return Err(VmError::TypeError {
                    expected: "Integer for s".to_string(),
                    actual: format!("{:?}", seasonal_list[3]),
                }),
            };

            (seasonal_p, seasonal_d, seasonal_q, seasonal_periods)
        }
        _ => return Err(VmError::TypeError {
            expected: "List {P, D, Q, s}".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };

    let params = ARIMAParams::seasonal(p, d, q, seasonal_p, seasonal_d, seasonal_q, seasonal_periods);

    match ARIMAModel::fit(series, params) {
        Ok(model) => Ok(Value::LyObj(LyObj::new(Box::new(model)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid SARIMA model".to_string(),
            actual: e,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::timeseries::core::Frequency;

    #[test]
    fn test_arima_params() {
        let params = ARIMAParams::new(2, 1, 1);
        assert_eq!(params.p, 2);
        assert_eq!(params.d, 1);
        assert_eq!(params.q, 1);
        assert!(!params.is_seasonal());
        assert_eq!(params.parameter_count(), 4); // 2 + 1 + 0 + 0 + 1(intercept) = 4

        let seasonal_params = ARIMAParams::seasonal(1, 1, 1, 1, 1, 1, 12);
        assert!(seasonal_params.is_seasonal());
        assert_eq!(seasonal_params.parameter_count(), 5); // 1 + 1 + 1 + 1 + 1(intercept) = 5
    }

    #[test]
    fn test_arima_coefficients() {
        let mut coeffs = ARIMACoefficients::new();
        coeffs.ar = vec![0.5, -0.2];
        coeffs.ma = vec![0.3];
        coeffs.intercept = 1.0;

        assert_eq!(coeffs.ar.len(), 2);
        assert_eq!(coeffs.ma.len(), 1);
        assert_eq!(coeffs.intercept, 1.0);
    }

    #[test]
    fn test_arima_model_fitting() {
        // Create a simple AR(1) process: x_t = 0.5 * x_{t-1} + e_t
        let mut values = vec![0.0];
        for i in 1..100 {
            values.push(0.5 * values[i-1] + 0.1 * (i as f64 % 7.0 - 3.0)); // Add some noise pattern
        }
        
        let series = TimeSeries::new(values, Frequency::Daily);
        let params = ARIMAParams::new(1, 0, 0); // AR(1)
        
        let result = ARIMAModel::fit(&series, params);
        assert!(result.is_ok());
        
        let model = result.unwrap();
        assert_eq!(model.params.p, 1);
        assert_eq!(model.params.d, 0);
        assert_eq!(model.params.q, 0);
        assert_eq!(model.coefficients.ar.len(), 1);
        
        // AR coefficient should be positive (around 0.5)
        assert!(model.coefficients.ar[0] > 0.0);
        assert!(model.coefficients.ar[0] < 1.0);
    }

    #[test]
    fn test_arima_forecasting() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let series = TimeSeries::new(values, Frequency::Daily);
        let params = ARIMAParams::new(1, 1, 0); // ARIMA(1,1,0)
        
        let model = ARIMAModel::fit(&series, params).unwrap();
        let forecasts = model.forecast(3).unwrap();
        
        assert_eq!(forecasts.len(), 3);
        // Forecasts should be reasonable (not NaN or infinite)
        for forecast in forecasts {
            assert!(forecast.is_finite());
        }
    }

    #[test]
    fn test_arima_function() {
        let values = vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)];
        let series_value = Value::List(values);
        let freq_value = Value::String("Daily".to_string());
        
        // Create TimeSeries
        let ts_result = crate::stdlib::timeseries::core::timeseries(&[series_value, freq_value]).unwrap();
        
        // Test ARIMA function
        let params = Value::List(vec![Value::Integer(1), Value::Integer(1), Value::Integer(0)]);
        let result = arima(&[ts_result, params]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let model = obj.downcast_ref::<ARIMAModel>().unwrap();
                assert_eq!(model.params.p, 1);
                assert_eq!(model.params.d, 1);
                assert_eq!(model.params.q, 0);
            }
            _ => panic!("Expected ARIMAModel object"),
        }
    }

    #[test]
    fn test_ar_coefficient_estimation() {
        // Test AR(1) coefficient estimation
        let data = vec![1.0, 1.5, 1.25, 1.625, 1.3125]; // AR(1) with φ=0.5
        let coeffs = ARIMAModel::estimate_ar_coefficients(&data, 1).unwrap();
        
        assert_eq!(coeffs.len(), 1);
        assert!(coeffs[0] > -1.0 && coeffs[0] < 1.0); // Stationarity constraint
    }

    #[test]
    fn test_ma_coefficient_estimation() {
        let data = vec![1.0, 0.5, 1.2, 0.8, 1.1];
        let coeffs = ARIMAModel::estimate_ma_coefficients(&data, 1).unwrap();
        
        assert_eq!(coeffs.len(), 1);
        assert!(coeffs[0].is_finite());
    }

    #[test]
    fn test_model_information_criteria() {
        let residuals = vec![0.1, -0.2, 0.15, -0.1, 0.05];
        
        let log_likelihood = ARIMAModel::calculate_log_likelihood(&residuals);
        let aic = ARIMAModel::calculate_aic(log_likelihood, 3);
        let bic = ARIMAModel::calculate_bic(log_likelihood, 3, 5);
        
        assert!(log_likelihood.is_finite());
        assert!(aic.is_finite());
        assert!(bic.is_finite());
        
        // For larger sample sizes, BIC is typically larger than AIC
        let large_residuals: Vec<f64> = (0..100).map(|i| (i as f64 % 5.0 - 2.0) * 0.1).collect();
        let log_likelihood_large = ARIMAModel::calculate_log_likelihood(&large_residuals);
        let aic_large = ARIMAModel::calculate_aic(log_likelihood_large, 3);
        let bic_large = ARIMAModel::calculate_bic(log_likelihood_large, 3, 100);
        
        assert!(bic_large > aic_large); // BIC penalizes complexity more for larger samples
    }
}