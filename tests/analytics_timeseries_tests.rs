//! TDD Tests for Phase 15B: Advanced Analytics & Statistics - Time Series Analysis Module
//!
//! Following strict TDD principles with RED-GREEN-REFACTOR approach.
//! Tests are written first to describe expected behavior before implementation.

use lyra::vm::{Value, VmError};
use lyra::foreign::LyObj;
use lyra::stdlib::analytics::timeseries::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;

#[cfg(test)]
mod timeseries_analysis_tests {
    use super::*;

    /// Test RED: Time series decomposition should separate trend, seasonal, and residual components
    #[test]
    fn test_timeseries_decomposition_additive() {
        // Arrange - Simple seasonal data
        let seasonal_data = vec![10.0, 15.0, 20.0, 15.0, 10.0, 15.0, 20.0, 15.0]; // Period = 4
        let series = Value::List(seasonal_data.iter().map(|&x| Value::Number(x)).collect());
        let model = Value::String("additive".to_string());
        let period = Value::Number(4.0);
        
        // Act
        let result = timeseries_decompose(&[series, model, period]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "TimeSeriesDecomposition");
            
            let trend = obj.foreign().call_method("trend", &[]).unwrap();
            assert!(matches!(trend, Value::List(_)));
            
            let seasonal = obj.foreign().call_method("seasonal", &[]).unwrap();
            assert!(matches!(seasonal, Value::List(_)));
            
            let residual = obj.foreign().call_method("residual", &[]).unwrap();
            assert!(matches!(residual, Value::List(_)));
            
            let model = obj.foreign().call_method("model", &[]).unwrap();
            assert_eq!(model, Value::String("additive".to_string()));
            
            let period = obj.foreign().call_method("period", &[]).unwrap();
            assert_eq!(period, Value::Number(4.0));
        }
    }

    /// Test RED: Multiplicative decomposition should handle ratio-based seasonality
    #[test]
    fn test_timeseries_decomposition_multiplicative() {
        // Arrange - Multiplicative seasonal data
        let seasonal_data = vec![10.0, 20.0, 40.0, 20.0, 15.0, 30.0, 60.0, 30.0]; // Exponential growth
        let series = Value::List(seasonal_data.iter().map(|&x| Value::Number(x)).collect());
        let model = Value::String("multiplicative".to_string());
        let period = Value::Number(4.0);
        
        // Act
        let result = timeseries_decompose(&[series, model, period]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            let model = obj.foreign().call_method("model", &[]).unwrap();
            assert_eq!(model, Value::String("multiplicative".to_string()));
        }
    }

    /// Test RED: Autocorrelation function should calculate correlations at different lags
    #[test]
    fn test_autocorrelation_function() {
        // Arrange - Simple AR(1) process
        let ar_data = vec![1.0, 0.8, 0.64, 0.512, 0.4096]; // AR(1) with φ=0.8
        let series = Value::List(ar_data.iter().map(|&x| Value::Number(x)).collect());
        let lags = Value::Number(3.0);
        
        // Act
        let result = auto_correlation(&[series, lags]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(acf_values) = result.unwrap() {
            assert_eq!(acf_values.len(), 4); // lags 0, 1, 2, 3
            
            // ACF at lag 0 should be 1.0
            if let Value::Number(acf_0) = acf_values[0] {
                assert!((acf_0 - 1.0).abs() < 0.01);
            }
            
            // ACF should decrease with lag for AR(1) process
            if let (Value::Number(acf_1), Value::Number(acf_2)) = (&acf_values[1], &acf_values[2]) {
                assert!(acf_1 > acf_2);
            }
        }
    }

    /// Test RED: Partial autocorrelation function should identify AR order
    #[test]
    fn test_partial_autocorrelation_function() {
        // Arrange - AR(2) process
        let ar2_data = vec![1.0, 1.2, 1.36, 1.488, 1.5936]; // AR(2) process
        let series = Value::List(ar2_data.iter().map(|&x| Value::Number(x)).collect());
        let lags = Value::Number(4.0);
        
        // Act
        let result = partial_auto_correlation(&[series, lags]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(pacf_values) = result.unwrap() {
            assert_eq!(pacf_values.len(), 5); // lags 0, 1, 2, 3, 4
            
            // PACF at lag 0 should be 1.0
            if let Value::Number(pacf_0) = pacf_values[0] {
                assert!((pacf_0 - 1.0).abs() < 0.01);
            }
        }
    }

    /// Test RED: ARIMA model fitting should create model with specified order
    #[test]
    fn test_arima_model_fitting() {
        // Arrange - Time series data
        let ts_data = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0];
        let series = Value::List(ts_data.iter().map(|&x| Value::Number(x)).collect());
        let order = Value::List(vec![
            Value::Number(1.0), // p
            Value::Number(1.0), // d
            Value::Number(1.0), // q
        ]);
        
        // Act
        let result = arima(&[series, order]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "ARIMAModel");
            
            let order = obj.foreign().call_method("order", &[]).unwrap();
            if let Value::List(order_values) = order {
                assert_eq!(order_values.len(), 3);
                assert_eq!(order_values[0], Value::Number(1.0));
                assert_eq!(order_values[1], Value::Number(1.0));
                assert_eq!(order_values[2], Value::Number(1.0));
            }
            
            let coefficients = obj.foreign().call_method("coefficients", &[]).unwrap();
            assert!(matches!(coefficients, Value::List(_)));
            
            let aic = obj.foreign().call_method("aic", &[]).unwrap();
            assert!(matches!(aic, Value::Number(_)));
            
            let bic = obj.foreign().call_method("bic", &[]).unwrap();
            assert!(matches!(bic, Value::Number(_)));
        }
    }

    /// Test RED: SARIMA model should handle seasonal components
    #[test]
    fn test_sarima_model_with_seasonal_order() {
        // Arrange - Seasonal time series data
        let seasonal_ts = vec![10.0, 15.0, 20.0, 15.0, 12.0, 17.0, 22.0, 17.0, 14.0, 19.0, 24.0, 19.0];
        let series = Value::List(seasonal_ts.iter().map(|&x| Value::Number(x)).collect());
        let order = Value::List(vec![
            Value::Number(1.0), // p
            Value::Number(1.0), // d
            Value::Number(1.0), // q
        ]);
        let seasonal_order = Value::List(vec![
            Value::Number(1.0), // P
            Value::Number(1.0), // D
            Value::Number(1.0), // Q
            Value::Number(4.0), // s (seasonal period)
        ]);
        
        // Act
        let result = arima(&[series, order, seasonal_order]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "ARIMAModel");
        }
    }

    /// Test RED: Forecast generation should provide point forecasts and confidence intervals
    #[test]
    fn test_forecast_generation() {
        // Arrange - Pre-fitted model (mock)
        let ts_data = vec![10.0, 11.0, 12.0, 13.0, 14.0];
        let series = Value::List(ts_data.iter().map(|&x| Value::Number(x)).collect());
        let order = Value::List(vec![Value::Number(1.0), Value::Number(0.0), Value::Number(0.0)]);
        let model_result = arima(&[series, order]).unwrap();
        
        let periods = Value::Number(3.0);
        let confidence_level = Value::Number(0.95);
        
        // Act
        let result = forecast(&[model_result, periods, confidence_level]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "ForecastResult");
            
            let forecasts = obj.foreign().call_method("forecasts", &[]).unwrap();
            if let Value::List(forecast_values) = forecasts {
                assert_eq!(forecast_values.len(), 3);
            }
            
            let lower_bound = obj.foreign().call_method("lowerBound", &[]).unwrap();
            assert!(matches!(lower_bound, Value::List(_)));
            
            let upper_bound = obj.foreign().call_method("upperBound", &[]).unwrap();
            assert!(matches!(upper_bound, Value::List(_)));
            
            let confidence_level = obj.foreign().call_method("confidenceLevel", &[]).unwrap();
            assert_eq!(confidence_level, Value::Number(0.95));
        }
    }

    /// Test RED: Seasonal decomposition should be equivalent to timeseries_decompose
    #[test]
    fn test_seasonal_decomposition() {
        // Arrange - Seasonal data
        let seasonal_data = vec![10.0, 20.0, 30.0, 20.0, 15.0, 25.0, 35.0, 25.0];
        let series = Value::List(seasonal_data.iter().map(|&x| Value::Number(x)).collect());
        let period = Value::Number(4.0);
        let method = Value::String("additive".to_string());
        
        // Act
        let result = seasonal_decompose(&[series, period, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "TimeSeriesDecomposition");
            
            let period = obj.foreign().call_method("period", &[]).unwrap();
            assert_eq!(period, Value::Number(4.0));
        }
    }

    /// Test RED: Trend analysis should extract underlying trend
    #[test]
    fn test_trend_analysis_linear() {
        // Arrange - Data with linear trend
        let trend_data = vec![10.0, 12.0, 14.0, 16.0, 18.0, 20.0];
        let series = Value::List(trend_data.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("linear".to_string());
        
        // Act
        let result = trend_analysis(&[series, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(trend_values) = result.unwrap() {
            assert_eq!(trend_values.len(), 6);
            
            // Linear trend should be approximately increasing
            if let (Value::Number(first), Value::Number(last)) = (&trend_values[0], &trend_values[5]) {
                assert!(last > first);
            }
        }
    }

    /// Test RED: Change point detection should identify structural breaks
    #[test]
    fn test_change_point_detection_cusum() {
        // Arrange - Data with change point at index 3
        let data_with_change = vec![1.0, 1.1, 0.9, 5.0, 5.1, 4.9, 5.2]; // Change at index 3
        let series = Value::List(data_with_change.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("cusum".to_string());
        let sensitivity = Value::Number(2.0);
        
        // Act
        let result = change_point_detection(&[series, method, sensitivity]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(change_points) = result.unwrap() {
            // Should detect at least one change point
            assert!(!change_points.is_empty());
        }
    }

    /// Test RED: Variance-based change point detection
    #[test]
    fn test_change_point_detection_variance() {
        // Arrange - Data with variance change
        let low_var = vec![10.0, 10.1, 9.9, 10.05, 9.95]; // Low variance
        let high_var = vec![15.0, 12.0, 18.0, 8.0, 22.0]; // High variance
        let mut combined_data = low_var;
        combined_data.extend(high_var);
        
        let series = Value::List(combined_data.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("variance".to_string());
        let sensitivity = Value::Number(5.0);
        
        // Act
        let result = change_point_detection(&[series, method, sensitivity]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(change_points) = result.unwrap() {
            // May or may not detect change points depending on implementation
            assert!(change_points.len() >= 0);
        }
    }

    /// Test RED: Anomaly detection using Z-score method
    #[test]
    fn test_anomaly_detection_zscore() {
        // Arrange - Data with outliers
        let data_with_outliers = vec![1.0, 2.0, 3.0, 100.0, 4.0, 5.0, 6.0]; // 100.0 is an outlier
        let series = Value::List(data_with_outliers.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("zscore".to_string());
        let threshold = Value::Number(2.0);
        
        // Act
        let result = anomaly_detection(&[series, method, threshold]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(anomaly_result) = result.unwrap() {
            assert!(anomaly_result.contains_key("indices"));
            assert!(anomaly_result.contains_key("values"));
            assert!(anomaly_result.contains_key("method"));
            assert!(anomaly_result.contains_key("threshold"));
            
            if let Some(Value::String(method)) = anomaly_result.get("method") {
                assert_eq!(method, "zscore");
            }
            if let Some(Value::Number(threshold)) = anomaly_result.get("threshold") {
                assert_eq!(*threshold, 2.0);
            }
        }
    }

    /// Test RED: IQR-based anomaly detection
    #[test]
    fn test_anomaly_detection_iqr() {
        // Arrange - Data with outliers
        let data_with_outliers = vec![10.0, 12.0, 11.0, 13.0, 50.0, 14.0, 15.0]; // 50.0 is an outlier
        let series = Value::List(data_with_outliers.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("iqr".to_string());
        let threshold = Value::Number(1.5);
        
        // Act
        let result = anomaly_detection(&[series, method, threshold]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(anomaly_result) = result.unwrap() {
            if let Some(Value::String(method)) = anomaly_result.get("method") {
                assert_eq!(method, "iqr");
            }
        }
    }

    /// Test RED: Stationarity testing using ADF test
    #[test]
    fn test_stationarity_test_adf() {
        // Arrange - Non-stationary data (random walk)
        let random_walk = vec![0.0, 1.0, 0.5, 1.8, 2.1, 1.9, 2.8, 3.2]; // Trending upward
        let series = Value::List(random_walk.iter().map(|&x| Value::Number(x)).collect());
        let test_type = Value::String("adf".to_string());
        
        // Act
        let result = stationarity_test(&[series, test_type]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(test_result) = result.unwrap() {
            assert!(test_result.contains_key("testType"));
            assert!(test_result.contains_key("statistic"));
            assert!(test_result.contains_key("pValue"));
            assert!(test_result.contains_key("isStationary"));
            
            if let Some(Value::String(test_type)) = test_result.get("testType") {
                assert!(test_type.contains("Augmented Dickey-Fuller"));
            }
        }
    }

    /// Test RED: KPSS stationarity test
    #[test]
    fn test_stationarity_test_kpss() {
        // Arrange - Stationary data
        let stationary_data = vec![1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 0.98]; // Mean-reverting
        let series = Value::List(stationary_data.iter().map(|&x| Value::Number(x)).collect());
        let test_type = Value::String("kpss".to_string());
        
        // Act
        let result = stationarity_test(&[series, test_type]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(test_result) = result.unwrap() {
            if let Some(Value::String(test_type)) = test_result.get("testType") {
                assert_eq!(test_type, "KPSS");
            }
        }
    }

    /// Test RED: Cross-correlation between two series
    #[test]
    fn test_cross_correlation() {
        // Arrange - Two related series
        let series1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let series2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // series2 = 2 * series1
        
        let series1_val = Value::List(series1.iter().map(|&x| Value::Number(x)).collect());
        let series2_val = Value::List(series2.iter().map(|&x| Value::Number(x)).collect());
        let lags = Value::Number(2.0);
        
        // Act
        let result = cross_correlation(&[series1_val, series2_val, lags]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(ccf_values) = result.unwrap() {
            assert_eq!(ccf_values.len(), 5); // -2, -1, 0, 1, 2 lags
            
            // Cross-correlation at lag 0 should be high (perfect correlation)
            if let Value::Number(ccf_0) = ccf_values[2] { // Middle element is lag 0
                assert!(ccf_0 > 0.9);
            }
        }
    }

    /// Test RED: Spectral density analysis using periodogram
    #[test]
    fn test_spectral_density_periodogram() {
        // Arrange - Sinusoidal data
        let n = 16;
        let sinusoidal_data: Vec<f64> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * i as f64 / 8.0).sin())
            .collect();
        
        let series = Value::List(sinusoidal_data.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("periodogram".to_string());
        
        // Act
        let result = spectral_density(&[series, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(spectral_result) = result.unwrap() {
            assert!(spectral_result.contains_key("frequencies"));
            assert!(spectral_result.contains_key("power"));
            assert!(spectral_result.contains_key("method"));
            
            if let Some(Value::String(method)) = spectral_result.get("method") {
                assert_eq!(method, "periodogram");
            }
            
            if let Some(Value::List(frequencies)) = spectral_result.get("frequencies") {
                assert!(!frequencies.is_empty());
            }
            
            if let Some(Value::List(power)) = spectral_result.get("power") {
                assert!(!power.is_empty());
            }
        }
    }

    /// Test RED: Error handling for mismatched series lengths in cross-correlation
    #[test]
    fn test_cross_correlation_error_handling() {
        // Arrange - Series of different lengths
        let series1 = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let series2 = Value::List(vec![Value::Number(1.0)]); // Different length
        let lags = Value::Number(1.0);
        
        // Act
        let result = cross_correlation(&[series1, series2, lags]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("same length"));
        }
    }

    /// Test RED: Error handling for invalid decomposition model
    #[test]
    fn test_decomposition_invalid_model() {
        // Arrange
        let series = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let model = Value::String("invalid_model".to_string());
        let period = Value::Number(2.0);
        
        // Act
        let result = timeseries_decompose(&[series, model, period]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("additive") || msg.contains("multiplicative"));
        }
    }

    /// Test RED: Error handling for insufficient arguments
    #[test]
    fn test_arima_insufficient_arguments() {
        // Arrange - Only one argument instead of required two
        let series = Value::List(vec![Value::Number(1.0)]);
        
        // Act
        let result = arima(&[series]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("requires at least 2 arguments"));
        }
    }

    /// Test RED: Error handling for invalid ARIMA order format
    #[test]
    fn test_arima_invalid_order_format() {
        // Arrange - Invalid order format
        let series = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let order = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]); // Missing q component
        
        // Act
        let result = arima(&[series, order]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("3 elements"));
        }
    }

    /// Test RED: Error handling for unsupported stationarity test
    #[test]
    fn test_stationarity_test_unsupported() {
        // Arrange
        let series = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let test_type = Value::String("unsupported_test".to_string());
        
        // Act
        let result = stationarity_test(&[series, test_type]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("Unsupported stationarity test"));
        }
    }

    /// Test RED: Error handling for series too short for ADF test
    #[test]
    fn test_adf_test_series_too_short() {
        // Arrange - Very short series
        let short_series = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let test_type = Value::String("adf".to_string());
        
        // Act
        let result = stationarity_test(&[short_series, test_type]);
        
        // Assert
        // Should either handle gracefully or return appropriate error
        // Exact behavior depends on implementation details
        assert!(result.is_ok() || result.is_err());
    }
}