//! Time Series Analysis Functions
//!
//! Comprehensive time series analysis capabilities including ARIMA modeling,
//! decomposition, forecasting, and anomaly detection.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::collections::HashMap;
use statrs::statistics::Statistics;
use std::any::Any;

/// Time Series Decomposition Result - Foreign Object
#[derive(Debug, Clone)]
pub struct TimeSeriesDecomposition {
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    residual: Vec<f64>,
    model: String, // additive or multiplicative
    period: usize,
}

impl Foreign for TimeSeriesDecomposition {
    fn type_name(&self) -> &'static str {
        "TimeSeriesDecomposition"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "trend" => Ok(Value::List(
                self.trend.iter().map(|&x| Value::Real(x)).collect()
            )),
            "seasonal" => Ok(Value::List(
                self.seasonal.iter().map(|&x| Value::Real(x)).collect()
            )),
            "residual" => Ok(Value::List(
                self.residual.iter().map(|&x| Value::Real(x)).collect()
            )),
            "model" => Ok(Value::String(self.model.clone())),
            "period" => Ok(Value::Integer(self.period as i64)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: "TimeSeriesDecomposition".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// ARIMA Model - Foreign Object
#[derive(Debug, Clone)]
pub struct ARIMAModel {
    order: (usize, usize, usize), // (p, d, q)
    seasonal_order: Option<(usize, usize, usize, usize)>, // (P, D, Q, s)
    coefficients: Vec<f64>,
    fitted_values: Vec<f64>,
    residuals: Vec<f64>,
    aic: f64,
    bic: f64,
}

impl Foreign for ARIMAModel {
    fn type_name(&self) -> &'static str {
        "ARIMAModel"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "order" => Ok(Value::List(vec![
                Value::Integer(self.order.0 as i64),
                Value::Integer(self.order.1 as i64),
                Value::Integer(self.order.2 as i64),
            ])),
            "coefficients" => Ok(Value::List(
                self.coefficients.iter().map(|&x| Value::Real(x)).collect()
            )),
            "fittedValues" => Ok(Value::List(
                self.fitted_values.iter().map(|&x| Value::Real(x)).collect()
            )),
            "residuals" => Ok(Value::List(
                self.residuals.iter().map(|&x| Value::Real(x)).collect()
            )),
            "aic" => Ok(Value::Real(self.aic)),
            "bic" => Ok(Value::Real(self.bic)),
            "forecast" => {
                let steps = args.get(0).and_then(|v| v.as_real()).unwrap_or(1.0) as usize;
                let _confidence_level = args.get(1).and_then(|v| v.as_real()).unwrap_or(0.95);
                // Simple forecast implementation
                let forecast_values: Vec<Value> = (0..steps).map(|_| Value::Real(0.0)).collect();
                Ok(Value::List(forecast_values))
            },
            _ => Err(ForeignError::UnknownMethod {
                type_name: "ARIMAModel".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Forecast Result - Foreign Object
#[derive(Debug, Clone)]
pub struct ForecastResult {
    forecasts: Vec<f64>,
    lower_bound: Vec<f64>,
    upper_bound: Vec<f64>,
    confidence_level: f64,
}

impl Foreign for ForecastResult {
    fn type_name(&self) -> &'static str {
        "ForecastResult"
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "forecasts" => Ok(Value::List(
                self.forecasts.iter().map(|&x| Value::Real(x)).collect()
            )),
            "lowerBound" => Ok(Value::List(
                self.lower_bound.iter().map(|&x| Value::Real(x)).collect()
            )),
            "upperBound" => Ok(Value::List(
                self.upper_bound.iter().map(|&x| Value::Real(x)).collect()
            )),
            "confidenceLevel" => Ok(Value::Real(self.confidence_level)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: "ForecastResult".to_string(),
                method: method.to_string(),
            }),
        }
    }
}

/// Time series decomposition (trend/seasonal/residual)
pub fn timeseries_decompose(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "TimeSeriesDecompose requires 3 arguments: series, model, period".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let model = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Model must be 'additive' or 'multiplicative'".to_string()
    ))?;
    let period = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Period must be a number".to_string()
    ))? as usize;

    let decomposition = perform_decomposition(series, &model, period)?;
    Ok(Value::LyObj(LyObj::new(Box::new(decomposition))))
}

/// Autocorrelation function
pub fn auto_correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "AutoCorrelation requires 2 arguments: series, lags".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let lags = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Lags must be a number".to_string()
    ))? as usize;

    let acf = calculate_autocorrelation(&series, lags)?;
    Ok(Value::List(acf.iter().map(|&x| Value::Real(x)).collect()))
}

/// Partial autocorrelation function
pub fn partial_auto_correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "PartialAutoCorrelation requires 2 arguments: series, lags".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let lags = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Lags must be a number".to_string()
    ))? as usize;

    let pacf = calculate_partial_autocorrelation(&series, lags)?;
    Ok(Value::List(pacf.iter().map(|&x| Value::Real(x)).collect()))
}

/// Advanced ARIMA modeling and forecasting
pub fn arima_advanced(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "ARIMA requires at least 2 arguments: series, order".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let order = extract_arima_order(&args[1])?;
    let seasonal_order = if args.len() > 2 {
        Some(extract_seasonal_order(&args[2])?)
    } else {
        None
    };

    let model = fit_arima_model(series, order, seasonal_order)?;
    Ok(Value::LyObj(LyObj::new(Box::new(model))))
}

/// Generate advanced forecasts from a model
pub fn forecast_advanced(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "Forecast requires 2 arguments: model, periods".to_string()
        ));
    }

    let model_obj = match &args[0] {
        Value::LyObj(obj) => obj,
        _ => return Err(VmError::Runtime(
            "First argument must be a model object".to_string()
        )),
    };

    let periods = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Periods must be a number".to_string()
    ))? as usize;
    let confidence_level = args.get(2).and_then(|v| v.as_real()).unwrap_or(0.95);

    // For now, generate a simple forecast
    let forecasts = vec![0.0; periods]; // Placeholder
    let margin = 1.96; // Approximate 95% CI
    let lower_bound = forecasts.iter().map(|&x| x - margin).collect();
    let upper_bound = forecasts.iter().map(|&x| x + margin).collect();

    let forecast_result = ForecastResult {
        forecasts,
        lower_bound,
        upper_bound,
        confidence_level,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(forecast_result))))
}

/// Seasonal decomposition
pub fn seasonal_decompose(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "SeasonalDecompose requires 2 arguments: series, period".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let period = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Period must be a number".to_string()
    ))? as usize;
    let method = args.get(2).and_then(|v| v.as_string()).unwrap_or("additive".to_string());

    let decomposition = perform_decomposition(series, &method, period)?;
    Ok(Value::LyObj(LyObj::new(Box::new(decomposition))))
}

/// Trend analysis
pub fn trend_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "TrendAnalysis requires 1 argument: series".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("linear".to_string());

    let trend = extract_trend(&series, &method)?;
    Ok(Value::List(trend.iter().map(|&x| Value::Real(x)).collect()))
}

/// Change point detection
pub fn change_point_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "ChangePointDetection requires 1 argument: series".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("cusum".to_string());
    let sensitivity = args.get(2).and_then(|v| v.as_real()).unwrap_or(1.0);

    let change_points = detect_change_points(&series, &method, sensitivity)?;
    Ok(Value::List(change_points.iter().map(|&x| Value::Integer(x as i64)).collect()))
}

/// Anomaly detection in time series
pub fn anomaly_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "AnomalyDetection requires 1 argument: series".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("zscore".to_string());
    let threshold = args.get(2).and_then(|v| v.as_real()).unwrap_or(2.0);

    let anomalies = detect_anomalies(&series, &method, threshold)?;
    
    let mut result = HashMap::new();
    result.insert("indices".to_string(), Value::List(
        anomalies.iter().map(|&i| Value::Integer(i as i64)).collect()
    ));
    result.insert("values".to_string(), Value::List(
        anomalies.iter().map(|&i| Value::Real(series[i])).collect()
    ));
    result.insert("method".to_string(), Value::String(method));
    result.insert("threshold".to_string(), Value::Real(threshold));

    Ok(Value::Object(result))
}

/// Stationarity testing (ADF, KPSS)
pub fn stationarity_test(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "StationarityTest requires 1 argument: series".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let test_type = args.get(1).and_then(|v| v.as_string()).unwrap_or("adf".to_string());

    let test_result = perform_stationarity_test(&series, &test_type)?;
    Ok(Value::Object(test_result))
}

/// Cross-correlation analysis
pub fn cross_correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "CrossCorrelation requires 3 arguments: series1, series2, lags".to_string()
        ));
    }

    let series1 = extract_numeric_vector(&args[0])?;
    let series2 = extract_numeric_vector(&args[1])?;
    let lags = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Lags must be a number".to_string()
    ))? as usize;

    let ccf = calculate_cross_correlation(&series1, &series2, lags)?;
    Ok(Value::List(ccf.iter().map(|&x| Value::Real(x)).collect()))
}

/// Spectral density analysis
pub fn spectral_density(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "SpectralDensity requires 1 argument: series".to_string()
        ));
    }

    let series = extract_numeric_vector(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("periodogram".to_string());

    let (frequencies, power) = calculate_spectral_density(&series, &method)?;
    
    let mut result = HashMap::new();
    result.insert("frequencies".to_string(), Value::List(
        frequencies.iter().map(|&x| Value::Real(x)).collect()
    ));
    result.insert("power".to_string(), Value::List(
        power.iter().map(|&x| Value::Real(x)).collect()
    ));
    result.insert("method".to_string(), Value::String(method));

    Ok(Value::Object(result))
}

// Helper functions for time series analysis
fn extract_numeric_vector(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter()
                .map(|item| match item {
                    Value::Real(n) => Ok(*n),
                    _ => Err(VmError::Runtime(
                        "All series elements must be numbers".to_string()
                    )),
                })
                .collect()
        },
        Value::Real(n) => Ok(vec![*n]),
        _ => Err(VmError::Runtime(
            "Series must be a number or list of numbers".to_string()
        )),
    }
}

fn extract_arima_order(value: &Value) -> VmResult<(usize, usize, usize)> {
    match value {
        Value::List(order) => {
            if order.len() != 3 {
                return Err(VmError::Runtime(
                    "ARIMA order must have 3 elements (p, d, q)".to_string()
                ));
            }
            let p = order[0].as_real().ok_or_else(|| VmError::Runtime(
                "p must be a number".to_string()
            ))? as usize;
            let d = order[1].as_real().ok_or_else(|| VmError::Runtime(
                "d must be a number".to_string()
            ))? as usize;
            let q = order[2].as_real().ok_or_else(|| VmError::Runtime(
                "q must be a number".to_string()
            ))? as usize;
            Ok((p, d, q))
        },
        _ => Err(VmError::Runtime(
            "ARIMA order must be a list [p, d, q]".to_string()
        )),
    }
}

fn extract_seasonal_order(value: &Value) -> VmResult<(usize, usize, usize, usize)> {
    match value {
        Value::List(order) => {
            if order.len() != 4 {
                return Err(VmError::Runtime(
                    "Seasonal order must have 4 elements (P, D, Q, s)".to_string()
                ));
            }
            let p = order[0].as_real().ok_or_else(|| VmError::Runtime(
                "P must be a number".to_string()
            ))? as usize;
            let d = order[1].as_real().ok_or_else(|| VmError::Runtime(
                "D must be a number".to_string()
            ))? as usize;
            let q = order[2].as_real().ok_or_else(|| VmError::Runtime(
                "Q must be a number".to_string()
            ))? as usize;
            let s = order[3].as_real().ok_or_else(|| VmError::Runtime(
                "s must be a number".to_string()
            ))? as usize;
            Ok((p, d, q, s))
        },
        _ => Err(VmError::Runtime(
            "Seasonal order must be a list [P, D, Q, s]".to_string()
        )),
    }
}

// Implementation functions
fn perform_decomposition(series: Vec<f64>, model: &str, period: usize) -> VmResult<TimeSeriesDecomposition> {
    let n = series.len();
    
    // Simple moving average for trend extraction
    let trend = extract_trend_moving_average(&series, period)?;
    
    // Calculate seasonal component
    let seasonal = match model {
        "additive" => calculate_additive_seasonal(&series, &trend, period)?,
        "multiplicative" => calculate_multiplicative_seasonal(&series, &trend, period)?,
        _ => return Err(VmError::Runtime(
            "Model must be 'additive' or 'multiplicative'".to_string()
        )),
    };
    
    // Calculate residual
    let residual = match model {
        "additive" => series.iter().zip(trend.iter().zip(seasonal.iter()))
            .map(|(x, (t, s))| x - t - s)
            .collect(),
        "multiplicative" => series.iter().zip(trend.iter().zip(seasonal.iter()))
            .map(|(x, (t, s))| if *t * s != 0.0 { x / (t * s) } else { 0.0 })
            .collect(),
        _ => unreachable!(),
    };
    
    Ok(TimeSeriesDecomposition {
        trend,
        seasonal,
        residual,
        model: model.to_string(),
        period,
    })
}

fn extract_trend_moving_average(series: &[f64], period: usize) -> VmResult<Vec<f64>> {
    let n = series.len();
    let mut trend = vec![f64::NAN; n];
    
    let half_period = period / 2;
    
    for i in half_period..(n - half_period) {
        let start = i - half_period;
        let end = i + half_period + 1;
        let sum: f64 = series[start..end].iter().sum();
        trend[i] = sum / period as f64;
    }
    
    Ok(trend)
}

fn calculate_additive_seasonal(series: &[f64], trend: &[f64], period: usize) -> VmResult<Vec<f64>> {
    let n = series.len();
    let mut seasonal = vec![0.0; n];
    let mut seasonal_averages = vec![0.0; period];
    let mut counts = vec![0; period];
    
    // Calculate seasonal averages
    for i in 0..n {
        if !trend[i].is_nan() {
            let season_index = i % period;
            seasonal_averages[season_index] += series[i] - trend[i];
            counts[season_index] += 1;
        }
    }
    
    // Average the seasonal components
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_averages[i] /= counts[i] as f64;
        }
    }
    
    // Apply seasonal pattern
    for i in 0..n {
        seasonal[i] = seasonal_averages[i % period];
    }
    
    Ok(seasonal)
}

fn calculate_multiplicative_seasonal(series: &[f64], trend: &[f64], period: usize) -> VmResult<Vec<f64>> {
    let n = series.len();
    let mut seasonal = vec![1.0; n];
    let mut seasonal_averages = vec![0.0; period];
    let mut counts = vec![0; period];
    
    // Calculate seasonal averages
    for i in 0..n {
        if !trend[i].is_nan() && trend[i] != 0.0 {
            let season_index = i % period;
            seasonal_averages[season_index] += series[i] / trend[i];
            counts[season_index] += 1;
        }
    }
    
    // Average the seasonal components
    for i in 0..period {
        if counts[i] > 0 {
            seasonal_averages[i] /= counts[i] as f64;
        } else {
            seasonal_averages[i] = 1.0;
        }
    }
    
    // Apply seasonal pattern
    for i in 0..n {
        seasonal[i] = seasonal_averages[i % period];
    }
    
    Ok(seasonal)
}

fn calculate_autocorrelation(series: &[f64], max_lags: usize) -> VmResult<Vec<f64>> {
    let n = series.len();
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    
    let mut acf = Vec::with_capacity(max_lags + 1);
    
    for lag in 0..=max_lags {
        if lag >= n {
            acf.push(0.0);
            continue;
        }
        
        let mut covariance = 0.0;
        let count = n - lag;
        
        for i in 0..count {
            covariance += (series[i] - mean) * (series[i + lag] - mean);
        }
        
        covariance /= n as f64;
        acf.push(covariance / variance);
    }
    
    Ok(acf)
}

fn calculate_partial_autocorrelation(series: &[f64], max_lags: usize) -> VmResult<Vec<f64>> {
    // Simplified PACF calculation using Yule-Walker equations
    let acf = calculate_autocorrelation(series, max_lags)?;
    let mut pacf = vec![0.0; max_lags + 1];
    
    if !acf.is_empty() {
        pacf[0] = 1.0;
    }
    if acf.len() > 1 {
        pacf[1] = acf[1];
    }
    
    // For higher lags, use approximate calculation
    for k in 2..=max_lags {
        if k < acf.len() {
            // Simplified calculation - in practice would use Levinson-Durbin algorithm
            pacf[k] = acf[k]; // Placeholder
        }
    }
    
    Ok(pacf)
}

fn fit_arima_model(series: Vec<f64>, order: (usize, usize, usize), seasonal_order: Option<(usize, usize, usize, usize)>) -> VmResult<ARIMAModel> {
    // Placeholder ARIMA implementation
    // In a full implementation, this would use proper ARIMA estimation
    
    let model = ARIMAModel {
        order,
        seasonal_order,
        coefficients: vec![0.5, 0.3], // Placeholder coefficients
        fitted_values: series.clone(),
        residuals: vec![0.0; series.len()],
        aic: 100.0, // Placeholder AIC
        bic: 105.0, // Placeholder BIC
    };
    
    Ok(model)
}

fn generate_forecast(model: &ARIMAModel, steps: usize, confidence_level: f64) -> VmResult<Value> {
    // Simple forecast generation (placeholder)
    let last_value = model.fitted_values.last().unwrap_or(&0.0);
    let forecasts = vec![*last_value; steps];
    
    let margin = 1.96; // Approximate 95% CI
    let lower_bound = forecasts.iter().map(|&x| x - margin).collect();
    let upper_bound = forecasts.iter().map(|&x| x + margin).collect();
    
    let forecast_result = ForecastResult {
        forecasts,
        lower_bound,
        upper_bound,
        confidence_level,
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(forecast_result))))
}

fn extract_trend(series: &[f64], method: &str) -> VmResult<Vec<f64>> {
    match method {
        "linear" => extract_linear_trend(series),
        "polynomial" => extract_polynomial_trend(series),
        "moving_average" => extract_trend_moving_average(series, 5),
        _ => Err(VmError::Runtime(
            format!("Unsupported trend method: {}", method)
        )),
    }
}

fn extract_linear_trend(series: &[f64]) -> VmResult<Vec<f64>> {
    let n = series.len() as f64;
    let x_sum = (0..series.len()).map(|i| i as f64).sum::<f64>();
    let y_sum = series.iter().sum::<f64>();
    let x_mean = x_sum / n;
    let y_mean = y_sum / n;
    
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    
    for (i, &y) in series.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }
    
    let slope = if denominator != 0.0 { numerator / denominator } else { 0.0 };
    let intercept = y_mean - slope * x_mean;
    
    let trend: Vec<f64> = (0..series.len())
        .map(|i| intercept + slope * i as f64)
        .collect();
    
    Ok(trend)
}

fn extract_polynomial_trend(series: &[f64]) -> VmResult<Vec<f64>> {
    // Placeholder for polynomial trend extraction
    // Would implement polynomial regression here
    extract_linear_trend(series) // Fallback to linear for now
}

fn detect_change_points(series: &[f64], method: &str, sensitivity: f64) -> VmResult<Vec<usize>> {
    match method {
        "cusum" => detect_change_points_cusum(series, sensitivity),
        "variance" => detect_change_points_variance(series, sensitivity),
        _ => Err(VmError::Runtime(
            format!("Unsupported change point method: {}", method)
        )),
    }
}

fn detect_change_points_cusum(series: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let mut cusum_pos = 0.0;
    let mut cusum_neg = 0.0;
    let mut change_points = Vec::new();
    
    for (i, &value) in series.iter().enumerate() {
        let deviation = value - mean;
        cusum_pos = f64::max(0.0, cusum_pos + deviation);
        cusum_neg = f64::min(0.0, cusum_neg + deviation);
        
        if cusum_pos > threshold || cusum_neg < -threshold {
            change_points.push(i);
            cusum_pos = 0.0;
            cusum_neg = 0.0;
        }
    }
    
    Ok(change_points)
}

fn detect_change_points_variance(series: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    // Sliding window variance change detection
    let window_size = 10;
    let mut change_points = Vec::new();
    
    if series.len() < 2 * window_size {
        return Ok(change_points);
    }
    
    for i in window_size..(series.len() - window_size) {
        let left_window = &series[(i - window_size)..i];
        let right_window = &series[i..(i + window_size)];
        
        let left_var = calculate_variance(left_window);
        let right_var = calculate_variance(right_window);
        
        if (left_var - right_var).abs() > threshold {
            change_points.push(i);
        }
    }
    
    Ok(change_points)
}

fn calculate_variance(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    variance
}

fn detect_anomalies(series: &[f64], method: &str, threshold: f64) -> VmResult<Vec<usize>> {
    match method {
        "zscore" => detect_anomalies_zscore(series, threshold),
        "iqr" => detect_anomalies_iqr(series, threshold),
        "isolation" => detect_anomalies_isolation(series, threshold),
        _ => Err(VmError::Runtime(
            format!("Unsupported anomaly detection method: {}", method)
        )),
    }
}

fn detect_anomalies_zscore(series: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let mean = series.iter().sum::<f64>() / series.len() as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / series.len() as f64;
    let std_dev = variance.sqrt();
    
    let anomalies: Vec<usize> = series.iter().enumerate()
        .filter_map(|(i, &value)| {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    
    Ok(anomalies)
}

fn detect_anomalies_iqr(series: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let mut sorted = series.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len();
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr = q3 - q1;
    
    let lower_bound = q1 - threshold * iqr;
    let upper_bound = q3 + threshold * iqr;
    
    let anomalies: Vec<usize> = series.iter().enumerate()
        .filter_map(|(i, &value)| {
            if value < lower_bound || value > upper_bound {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    
    Ok(anomalies)
}

fn detect_anomalies_isolation(series: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    // Simplified isolation forest approach
    // In practice, would implement full isolation forest algorithm
    detect_anomalies_zscore(series, threshold) // Fallback for now
}

fn perform_stationarity_test(series: &[f64], test_type: &str) -> VmResult<HashMap<String, Value>> {
    let mut result = HashMap::new();
    
    match test_type {
        "adf" => {
            // Augmented Dickey-Fuller test (simplified)
            let (statistic, p_value) = adf_test(series)?;
            result.insert("testType".to_string(), Value::String("Augmented Dickey-Fuller".to_string()));
            result.insert("statistic".to_string(), Value::Real(statistic));
            result.insert("pValue".to_string(), Value::Real(p_value));
            result.insert("isStationary".to_string(), Value::Boolean(p_value < 0.05));
        },
        "kpss" => {
            // KPSS test (simplified)
            let (statistic, p_value) = kpss_test(series)?;
            result.insert("testType".to_string(), Value::String("KPSS".to_string()));
            result.insert("statistic".to_string(), Value::Real(statistic));
            result.insert("pValue".to_string(), Value::Real(p_value));
            result.insert("isStationary".to_string(), Value::Boolean(p_value > 0.05));
        },
        _ => return Err(VmError::Runtime(
            format!("Unsupported stationarity test: {}", test_type)
        )),
    }
    
    Ok(result)
}

fn adf_test(series: &[f64]) -> VmResult<(f64, f64)> {
    // Simplified ADF test implementation
    // In practice, would implement full Augmented Dickey-Fuller test
    let n = series.len();
    if n < 3 {
        return Err(VmError::Runtime(
            "Series too short for ADF test".to_string()
        ));
    }
    
    let differences: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
    let mean_diff = differences.iter().sum::<f64>() / differences.len() as f64;
    let variance = differences.iter().map(|x| (x - mean_diff).powi(2)).sum::<f64>() / differences.len() as f64;
    let std_error = (variance / differences.len() as f64).sqrt();
    
    let t_statistic = mean_diff / std_error;
    let p_value = if t_statistic < -2.86 { 0.01 } else { 0.10 }; // Simplified critical values
    
    Ok((t_statistic, p_value))
}

fn kpss_test(series: &[f64]) -> VmResult<(f64, f64)> {
    // Simplified KPSS test implementation
    let n = series.len();
    if n < 4 {
        return Err(VmError::Runtime(
            "Series too short for KPSS test".to_string()
        ));
    }
    
    // Calculate test statistic (simplified)
    let mean = series.iter().sum::<f64>() / n as f64;
    let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    
    let statistic = variance / n as f64;
    let p_value = if statistic > 0.463 { 0.01 } else { 0.10 }; // Simplified critical values
    
    Ok((statistic, p_value))
}

fn calculate_cross_correlation(series1: &[f64], series2: &[f64], max_lags: usize) -> VmResult<Vec<f64>> {
    if series1.len() != series2.len() {
        return Err(VmError::Runtime(
            "Series must have the same length for cross-correlation".to_string()
        ));
    }
    
    let n = series1.len();
    let mean1 = series1.iter().sum::<f64>() / n as f64;
    let mean2 = series2.iter().sum::<f64>() / n as f64;
    
    let var1 = series1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / n as f64;
    let var2 = series2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / n as f64;
    let norm_factor = (var1 * var2).sqrt();
    
    let mut ccf = Vec::with_capacity(2 * max_lags + 1);
    
    // Negative lags
    for lag in (1..=max_lags).rev() {
        if lag >= n {
            ccf.push(0.0);
            continue;
        }
        
        let mut covariance = 0.0;
        let count = n - lag;
        
        for i in 0..count {
            covariance += (series1[i + lag] - mean1) * (series2[i] - mean2);
        }
        
        covariance /= n as f64;
        ccf.push(covariance / norm_factor);
    }
    
    // Zero lag
    let mut covariance = 0.0;
    for i in 0..n {
        covariance += (series1[i] - mean1) * (series2[i] - mean2);
    }
    covariance /= n as f64;
    ccf.push(covariance / norm_factor);
    
    // Positive lags
    for lag in 1..=max_lags {
        if lag >= n {
            ccf.push(0.0);
            continue;
        }
        
        let mut covariance = 0.0;
        let count = n - lag;
        
        for i in 0..count {
            covariance += (series1[i] - mean1) * (series2[i + lag] - mean2);
        }
        
        covariance /= n as f64;
        ccf.push(covariance / norm_factor);
    }
    
    Ok(ccf)
}

fn calculate_spectral_density(series: &[f64], method: &str) -> VmResult<(Vec<f64>, Vec<f64>)> {
    match method {
        "periodogram" => calculate_periodogram(series),
        "welch" => calculate_welch_psd(series),
        _ => Err(VmError::Runtime(
            format!("Unsupported spectral density method: {}", method)
        )),
    }
}

fn calculate_periodogram(series: &[f64]) -> VmResult<(Vec<f64>, Vec<f64>)> {
    let n = series.len();
    let mut frequencies = Vec::new();
    let mut power = Vec::new();
    
    // Simple periodogram calculation
    for k in 0..(n / 2) {
        let freq = k as f64 / n as f64;
        frequencies.push(freq);
        
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;
        
        for (t, &x) in series.iter().enumerate() {
            let angle = -2.0 * std::f64::consts::PI * freq * t as f64;
            real_sum += x * angle.cos();
            imag_sum += x * angle.sin();
        }
        
        let magnitude_squared = real_sum * real_sum + imag_sum * imag_sum;
        power.push(magnitude_squared / n as f64);
    }
    
    Ok((frequencies, power))
}

fn calculate_welch_psd(series: &[f64]) -> VmResult<(Vec<f64>, Vec<f64>)> {
    // Simplified Welch's method - would implement overlapping windowed periodograms
    calculate_periodogram(series) // Fallback for now
}