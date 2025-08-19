//! Time Series Forecasting Methods
//!
//! This module provides various forecasting techniques including exponential smoothing,
//! Holt-Winters, moving averages, and trend decomposition.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use crate::stdlib::timeseries::core::{TimeSeries, Frequency, extract_timeseries};
use std::any::Any;
use std::collections::HashMap;

/// Exponential Smoothing model parameters
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingParams {
    /// Smoothing parameter for level (0 < alpha <= 1)
    pub alpha: f64,
    /// Smoothing parameter for trend (0 <= beta <= 1)
    pub beta: Option<f64>,
    /// Smoothing parameter for seasonality (0 <= gamma <= 1)
    pub gamma: Option<f64>,
    /// Number of seasonal periods
    pub seasonal_periods: Option<usize>,
    /// Additive or multiplicative seasonality
    pub seasonal_type: SeasonalType,
}

/// Type of seasonality
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeasonalType {
    Additive,
    Multiplicative,
    None,
}

/// Exponential Smoothing model results
#[derive(Debug, Clone)]
pub struct ExponentialSmoothingModel {
    /// Model parameters
    pub params: ExponentialSmoothingParams,
    /// Original time series
    pub original_series: TimeSeries,
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Level component
    pub level: Vec<f64>,
    /// Trend component (if applicable)
    pub trend: Option<Vec<f64>>,
    /// Seasonal component (if applicable)
    pub seasonal: Option<Vec<f64>>,
    /// Model residuals
    pub residuals: Vec<f64>,
    /// Sum of squared errors
    pub sse: f64,
    /// Akaike Information Criterion
    pub aic: f64,
    /// Model metadata
    pub metadata: HashMap<String, String>,
}

impl ExponentialSmoothingModel {
    /// Fit exponential smoothing model to time series
    pub fn fit(series: &TimeSeries, params: ExponentialSmoothingParams) -> Result<ExponentialSmoothingModel, String> {
        if series.values.is_empty() {
            return Err("Cannot fit model to empty series".to_string());
        }

        let data = &series.values;
        let n = data.len();
        
        // Initialize components
        let mut level = vec![0.0; n];
        let mut trend = if params.beta.is_some() { Some(vec![0.0; n]) } else { None };
        let mut seasonal = if params.gamma.is_some() && params.seasonal_periods.is_some() { 
            Some(vec![0.0; n]) 
        } else { 
            None 
        };
        let mut fitted_values = vec![0.0; n];

        // Initialize first values
        level[0] = data[0];
        
        if let Some(ref mut trend_vec) = trend {
            if n > 1 {
                trend_vec[0] = data[1] - data[0];
            }
        }

        if let (Some(ref mut seasonal_vec), Some(s)) = (&mut seasonal, params.seasonal_periods) {
            if s > 0 && n >= s {
                // Initialize seasonal components
                Self::initialize_seasonal_components(data, seasonal_vec, s, params.seasonal_type);
            }
        }

        fitted_values[0] = level[0];

        // Apply exponential smoothing
        for t in 1..n {
            let previous_fitted = fitted_values[t - 1];
            let actual = data[t];

            // Update level
            let level_update = match (&seasonal, params.seasonal_periods) {
                (Some(ref seasonal_vec), Some(s)) if t >= s => {
                    match params.seasonal_type {
                        SeasonalType::Additive => actual - seasonal_vec[t - s],
                        SeasonalType::Multiplicative => {
                            if seasonal_vec[t - s] != 0.0 {
                                actual / seasonal_vec[t - s]
                            } else {
                                actual
                            }
                        }
                        SeasonalType::None => actual,
                    }
                }
                _ => actual,
            };

            level[t] = params.alpha * level_update + (1.0 - params.alpha) * 
                (level[t - 1] + trend.as_ref().map(|t_vec| t_vec[t - 1]).unwrap_or(0.0));

            // Update trend
            if let (Some(ref mut trend_vec), Some(beta)) = (&mut trend, params.beta) {
                trend_vec[t] = beta * (level[t] - level[t - 1]) + (1.0 - beta) * trend_vec[t - 1];
            }

            // Update seasonal
            if let (Some(ref mut seasonal_vec), Some(gamma), Some(s)) = 
                (&mut seasonal, params.gamma, params.seasonal_periods) {
                if t >= s {
                    let seasonal_update = match params.seasonal_type {
                        SeasonalType::Additive => actual - level[t],
                        SeasonalType::Multiplicative => {
                            if level[t] != 0.0 {
                                actual / level[t]
                            } else {
                                1.0
                            }
                        }
                        SeasonalType::None => 0.0,
                    };
                    seasonal_vec[t] = gamma * seasonal_update + (1.0 - gamma) * seasonal_vec[t - s];
                }
            }

            // Calculate fitted value
            fitted_values[t] = level[t] + trend.as_ref().map(|t_vec| t_vec[t]).unwrap_or(0.0);
            
            if let (Some(ref seasonal_vec), Some(s)) = (&seasonal, params.seasonal_periods) {
                if t >= s {
                    match params.seasonal_type {
                        SeasonalType::Additive => fitted_values[t] += seasonal_vec[t - s],
                        SeasonalType::Multiplicative => fitted_values[t] *= seasonal_vec[t - s],
                        SeasonalType::None => {},
                    }
                }
            }
        }

        // Calculate residuals and fit statistics
        let residuals: Vec<f64> = data.iter().zip(fitted_values.iter())
            .map(|(&actual, &fitted)| actual - fitted)
            .collect();

        let sse = residuals.iter().map(|&r| r * r).sum();
        let param_count = Self::count_parameters(&params);
        let aic = (n as f64) * ((sse / n as f64) as f64).ln() + 2.0 * param_count as f64;

        Ok(ExponentialSmoothingModel {
            params,
            original_series: series.clone(),
            fitted_values,
            level,
            trend,
            seasonal,
            residuals,
            sse,
            aic,
            metadata: HashMap::new(),
        })
    }

    /// Initialize seasonal components
    fn initialize_seasonal_components(data: &[f64], seasonal: &mut [f64], s: usize, seasonal_type: SeasonalType) {
        if s == 0 || data.len() < s {
            return;
        }

        // Simple initialization: use average seasonal differences
        for i in 0..s {
            let mut seasonal_sum = 0.0;
            let mut count = 0;
            
            for period_start in (i..data.len()).step_by(s) {
                if period_start + s <= data.len() {
                    let period_avg = data[period_start..period_start + s].iter().sum::<f64>() / s as f64;
                    match seasonal_type {
                        SeasonalType::Additive => seasonal_sum += data[period_start] - period_avg,
                        SeasonalType::Multiplicative => {
                            if period_avg != 0.0 {
                                seasonal_sum += data[period_start] / period_avg;
                            } else {
                                seasonal_sum += 1.0;
                            }
                        }
                        SeasonalType::None => seasonal_sum += 0.0,
                    }
                    count += 1;
                }
            }
            
            if count > 0 {
                seasonal[i] = seasonal_sum / count as f64;
            } else {
                seasonal[i] = match seasonal_type {
                    SeasonalType::Additive => 0.0,
                    SeasonalType::Multiplicative => 1.0,
                    SeasonalType::None => 0.0,
                };
            }
        }

        // Propagate initial seasonal values
        for i in s..seasonal.len() {
            seasonal[i] = seasonal[i % s];
        }
    }

    /// Count the number of parameters in the model
    fn count_parameters(params: &ExponentialSmoothingParams) -> usize {
        let mut count = 1; // alpha
        if params.beta.is_some() {
            count += 1;
        }
        if params.gamma.is_some() {
            count += 1;
        }
        count
    }

    /// Generate forecasts
    pub fn forecast(&self, steps: usize) -> Result<Vec<f64>, String> {
        if steps == 0 {
            return Ok(Vec::new());
        }

        let mut forecasts = Vec::with_capacity(steps);
        let n = self.level.len();
        
        if n == 0 {
            return Err("Cannot forecast from empty model".to_string());
        }

        let last_level = self.level[n - 1];
        let last_trend = self.trend.as_ref().map(|t| t[n - 1]).unwrap_or(0.0);

        for h in 1..=steps {
            let mut forecast = last_level + (h as f64) * last_trend;
            
            // Add seasonal component if present
            if let (Some(ref seasonal_vec), Some(s)) = (&self.seasonal, self.params.seasonal_periods) {
                if s > 0 {
                    let seasonal_index = (n - 1 + h) % s;
                    let seasonal_factor = seasonal_vec[seasonal_index.min(seasonal_vec.len() - 1)];
                    
                    match self.params.seasonal_type {
                        SeasonalType::Additive => forecast += seasonal_factor,
                        SeasonalType::Multiplicative => forecast *= seasonal_factor,
                        SeasonalType::None => {},
                    }
                }
            }
            
            forecasts.push(forecast);
        }

        Ok(forecasts)
    }
}

impl Foreign for ExponentialSmoothingModel {
    fn type_name(&self) -> &'static str {
        "ExponentialSmoothingModel"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Alpha" => Ok(Value::Real(self.params.alpha)),
            "Beta" => Ok(Value::Real(self.params.beta.unwrap_or(0.0))),
            "Gamma" => Ok(Value::Real(self.params.gamma.unwrap_or(0.0))),
            "AIC" => Ok(Value::Real(self.aic)),
            "SSE" => Ok(Value::Real(self.sse)),
            "FittedValues" => {
                let fitted: Vec<Value> = self.fitted_values.iter()
                    .map(|&f| Value::Real(f))
                    .collect();
                Ok(Value::List(fitted))
            }
            "Residuals" => {
                let residuals: Vec<Value> = self.residuals.iter()
                    .map(|&r| Value::Real(r))
                    .collect();
                Ok(Value::List(residuals))
            }
            "Level" => {
                let level: Vec<Value> = self.level.iter()
                    .map(|&l| Value::Real(l))
                    .collect();
                Ok(Value::List(level))
            }
            "Trend" => {
                if let Some(ref trend) = self.trend {
                    let trend_values: Vec<Value> = trend.iter()
                        .map(|&t| Value::Real(t))
                        .collect();
                    Ok(Value::List(trend_values))
                } else {
                    Ok(Value::List(Vec::new()))
                }
            }
            "Seasonal" => {
                if let Some(ref seasonal) = self.seasonal {
                    let seasonal_values: Vec<Value> = seasonal.iter()
                        .map(|&s| Value::Real(s))
                        .collect();
                    Ok(Value::List(seasonal_values))
                } else {
                    Ok(Value::List(Vec::new()))
                }
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

/// Moving Average result
#[derive(Debug, Clone)]
pub struct MovingAverageResult {
    /// Original time series
    pub original_series: TimeSeries,
    /// Moving average values
    pub ma_values: Vec<f64>,
    /// Window size
    pub window: usize,
    /// Type of moving average
    pub ma_type: MovingAverageType,
}

/// Type of moving average
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MovingAverageType {
    Simple,
    Exponential,
    Weighted,
}

impl MovingAverageResult {
    /// Calculate simple moving average
    pub fn simple(series: &TimeSeries, window: usize) -> Result<MovingAverageResult, String> {
        if window == 0 || window > series.len() {
            return Err("Invalid window size".to_string());
        }

        let data = &series.values;
        let mut ma_values = Vec::new();

        for i in 0..data.len() {
            if i + 1 >= window {
                let start = i + 1 - window;
                let end = i + 1;
                let sum: f64 = data[start..end].iter().sum();
                ma_values.push(sum / window as f64);
            } else {
                ma_values.push(f64::NAN); // Not enough data points
            }
        }

        Ok(MovingAverageResult {
            original_series: series.clone(),
            ma_values,
            window,
            ma_type: MovingAverageType::Simple,
        })
    }

    /// Calculate exponential moving average
    pub fn exponential(series: &TimeSeries, alpha: f64) -> Result<MovingAverageResult, String> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err("Alpha must be between 0 and 1".to_string());
        }

        let data = &series.values;
        let mut ma_values = Vec::with_capacity(data.len());

        if data.is_empty() {
            return Ok(MovingAverageResult {
                original_series: series.clone(),
                ma_values,
                window: 0,
                ma_type: MovingAverageType::Exponential,
            });
        }

        // Initialize with first value
        ma_values.push(data[0]);

        // Calculate EMA
        for i in 1..data.len() {
            let ema = alpha * data[i] + (1.0 - alpha) * ma_values[i - 1];
            ma_values.push(ema);
        }

        Ok(MovingAverageResult {
            original_series: series.clone(),
            ma_values,
            window: 0, // EMA doesn't have a fixed window
            ma_type: MovingAverageType::Exponential,
        })
    }
}

impl Foreign for MovingAverageResult {
    fn type_name(&self) -> &'static str {
        "MovingAverageResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Values" => {
                let values: Vec<Value> = self.ma_values.iter()
                    .map(|&v| if v.is_nan() { Value::String("Missing".to_string()) } else { Value::Real(v) })
                    .collect();
                Ok(Value::List(values))
            }
            "Window" => Ok(Value::Integer(self.window as i64)),
            "Type" => Ok(Value::String(format!("{:?}", self.ma_type))),
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
// FORECASTING FUNCTIONS
// ===============================

/// Exponential smoothing forecasting
/// Syntax: ExponentialSmoothing[timeseries, alpha, [beta], [gamma], [seasonal_periods]]
pub fn exponential_smoothing(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "2-5 arguments (timeseries, alpha, [beta], [gamma], [seasonal_periods])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let series = extract_timeseries(&args[0])?;

    let alpha = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for alpha".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    if alpha <= 0.0 || alpha > 1.0 {
        return Err(VmError::TypeError {
            expected: "alpha between 0 and 1".to_string(),
            actual: format!("alpha = {}", alpha),
        });
    }

    let beta = if args.len() >= 3 {
        match &args[2] {
            Value::Real(r) => Some(*r),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    } else {
        None
    };

    let gamma = if args.len() >= 4 {
        match &args[3] {
            Value::Real(r) => Some(*r),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    } else {
        None
    };

    let seasonal_periods = if args.len() == 5 {
        match &args[4] {
            Value::Integer(i) => Some(*i as usize),
            _ => None,
        }
    } else {
        None
    };

    let seasonal_type = if gamma.is_some() && seasonal_periods.is_some() {
        SeasonalType::Additive
    } else {
        SeasonalType::None
    };

    let params = ExponentialSmoothingParams {
        alpha,
        beta,
        gamma,
        seasonal_periods,
        seasonal_type,
    };

    match ExponentialSmoothingModel::fit(series, params) {
        Ok(model) => Ok(Value::LyObj(LyObj::new(Box::new(model)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid exponential smoothing model".to_string(),
            actual: e,
        }),
    }
}

/// Simple moving average
/// Syntax: MovingAverage[timeseries, window]
pub fn moving_average(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (timeseries, window)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let series = extract_timeseries(&args[0])?;

    let window = match &args[1] {
        Value::Integer(i) => *i as usize,
        _ => return Err(VmError::TypeError {
            expected: "Integer for window size".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match MovingAverageResult::simple(series, window) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid moving average calculation".to_string(),
            actual: e,
        }),
    }
}

/// Exponential moving average
/// Syntax: ExponentialMovingAverage[timeseries, alpha]
pub fn exponential_moving_average(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (timeseries, alpha)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let series = extract_timeseries(&args[0])?;

    let alpha = match &args[1] {
        Value::Real(r) => *r,
        Value::Integer(i) => *i as f64,
        _ => return Err(VmError::TypeError {
            expected: "Real number for alpha".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    match MovingAverageResult::exponential(series, alpha) {
        Ok(result) => Ok(Value::LyObj(LyObj::new(Box::new(result)))),
        Err(e) => Err(VmError::TypeError {
            expected: "valid exponential moving average calculation".to_string(),
            actual: e,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_smoothing_params() {
        let params = ExponentialSmoothingParams {
            alpha: 0.3,
            beta: Some(0.1),
            gamma: None,
            seasonal_periods: None,
            seasonal_type: SeasonalType::None,
        };

        assert_eq!(params.alpha, 0.3);
        assert_eq!(params.beta, Some(0.1));
        assert_eq!(params.gamma, None);
        assert_eq!(ExponentialSmoothingModel::count_parameters(&params), 2);
    }

    #[test]
    fn test_simple_exponential_smoothing() {
        let values = vec![10.0, 12.0, 13.0, 12.0, 14.0, 16.0, 15.0, 17.0];
        let series = TimeSeries::new(values, Frequency::Daily);
        
        let params = ExponentialSmoothingParams {
            alpha: 0.3,
            beta: None,
            gamma: None,
            seasonal_periods: None,
            seasonal_type: SeasonalType::None,
        };

        let model = ExponentialSmoothingModel::fit(&series, params).unwrap();
        
        assert_eq!(model.level.len(), 8);
        assert!(model.trend.is_none());
        assert!(model.seasonal.is_none());
        assert_eq!(model.residuals.len(), 8);
        assert!(model.sse > 0.0);
    }

    #[test]
    fn test_double_exponential_smoothing() {
        let values = vec![10.0, 11.0, 12.5, 14.0, 15.5, 17.0, 18.5, 20.0];
        let series = TimeSeries::new(values, Frequency::Daily);
        
        let params = ExponentialSmoothingParams {
            alpha: 0.3,
            beta: Some(0.1),
            gamma: None,
            seasonal_periods: None,
            seasonal_type: SeasonalType::None,
        };

        let model = ExponentialSmoothingModel::fit(&series, params).unwrap();
        
        assert_eq!(model.level.len(), 8);
        assert!(model.trend.is_some());
        assert!(model.seasonal.is_none());
        
        // Test forecasting
        let forecasts = model.forecast(3).unwrap();
        assert_eq!(forecasts.len(), 3);
        assert!(forecasts[0] > 20.0); // Should continue the trend
    }

    #[test]
    fn test_moving_average_simple() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let series = TimeSeries::new(values, Frequency::Daily);
        
        let result = MovingAverageResult::simple(&series, 3).unwrap();
        
        assert_eq!(result.ma_values.len(), 8);
        assert!(result.ma_values[0].is_nan()); // Not enough data
        assert!(result.ma_values[1].is_nan()); // Not enough data
        assert_eq!(result.ma_values[2], 2.0); // (1+2+3)/3
        assert_eq!(result.ma_values[3], 3.0); // (2+3+4)/3
        assert_eq!(result.window, 3);
    }

    #[test]
    fn test_moving_average_exponential() {
        let values = vec![10.0, 12.0, 11.0, 13.0, 15.0];
        let series = TimeSeries::new(values, Frequency::Daily);
        
        let result = MovingAverageResult::exponential(&series, 0.3).unwrap();
        
        assert_eq!(result.ma_values.len(), 5);
        assert_eq!(result.ma_values[0], 10.0); // First value
        assert!((result.ma_values[1] - 10.6).abs() < 1e-10); // 0.3*12 + 0.7*10
        assert_eq!(result.ma_type, MovingAverageType::Exponential);
    }

    #[test]
    fn test_exponential_smoothing_function() {
        let values = vec![Value::Real(10.0), Value::Real(12.0), Value::Real(11.0), Value::Real(13.0)];
        let series_value = Value::List(values);
        let freq_value = Value::String("Daily".to_string());
        
        // Create TimeSeries
        let ts_result = crate::stdlib::timeseries::core::timeseries(&[series_value, freq_value]).unwrap();
        
        // Test ExponentialSmoothing function
        let alpha = Value::Real(0.3);
        let result = exponential_smoothing(&[ts_result, alpha]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let model = obj.downcast_ref::<ExponentialSmoothingModel>().unwrap();
                assert_eq!(model.params.alpha, 0.3);
                assert!(model.params.beta.is_none());
            }
            _ => panic!("Expected ExponentialSmoothingModel object"),
        }
    }

    #[test]
    fn test_moving_average_function() {
        let values = vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)];
        let series_value = Value::List(values);
        let freq_value = Value::String("Daily".to_string());
        
        // Create TimeSeries
        let ts_result = crate::stdlib::timeseries::core::timeseries(&[series_value, freq_value]).unwrap();
        
        // Test MovingAverage function
        let window = Value::Integer(3);
        let result = moving_average(&[ts_result, window]).unwrap();
        
        match result {
            Value::LyObj(obj) => {
                let ma_result = obj.downcast_ref::<MovingAverageResult>().unwrap();
                assert_eq!(ma_result.window, 3);
                assert_eq!(ma_result.ma_type, MovingAverageType::Simple);
            }
            _ => panic!("Expected MovingAverageResult object"),
        }
    }
}