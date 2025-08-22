//! Risk Management and Metrics
//!
//! Implementation of various risk metrics including:
//! - Value at Risk (VaR) using parametric and historical methods
//! - Conditional VaR (Expected Shortfall)
//! - Sharpe ratio and information ratio
//! - Beta and correlation analysis
//! - Risk-adjusted performance measures

use super::{extract_real, extract_list_of_reals, SQRT_TRADING_DAYS};
use crate::vm::{Value, VmResult, VmError};

/// Calculate Value at Risk using historical method
pub fn historical_var(returns: &[f64], confidence_level: f64) -> Result<f64, VmError> {
    if returns.is_empty() {
        return Err(VmError::Runtime("Expected non-empty returns list".to_string()));
    }

    if confidence_level <= 0.0 || confidence_level >= 1.0 {
        return Err(VmError::Runtime(format!(
            "Confidence level must be between 0 and 1, got {}",
            confidence_level
        )));
    }

    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let index = ((1.0 - confidence_level) * (returns.len() as f64)).floor() as usize;
    let var = -sorted_returns[index.min(sorted_returns.len() - 1)]; // VaR is positive loss

    Ok(var)
}

/// Calculate parametric VaR assuming normal distribution
pub fn parametric_var(returns: &[f64], confidence_level: f64) -> Result<f64, VmError> {
    if returns.is_empty() {
        return Err(VmError::Runtime("Expected non-empty returns list".to_string()));
    }

    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns.iter()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    let std_dev = variance.sqrt();

    // Z-score for given confidence level (approximation)
    let z_score = match confidence_level {
        x if x >= 0.99 => 2.326,   // 99%
        x if x >= 0.95 => 1.645,   // 95%
        x if x >= 0.90 => 1.282,   // 90%
        _ => inverse_normal_cdf(confidence_level),
    };

    let var = -(mean - z_score * std_dev); // VaR is positive loss
    Ok(var)
}

/// Approximate inverse normal CDF
fn inverse_normal_cdf(p: f64) -> f64 {
    // Beasley-Springer-Moro algorithm approximation
    let a = vec![
        0.0,
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = vec![
        0.0,
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = vec![
        0.0,
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = vec![
        0.0,
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        return (((((c[6] * q + c[5]) * q + c[4]) * q + c[3]) * q + c[2]) * q + c[1]) * q + c[0]
               / ((((d[4] * q + d[3]) * q + d[2]) * q + d[1]) * q + 1.0);
    }

    if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        return (((((a[6] * r + a[5]) * r + a[4]) * r + a[3]) * r + a[2]) * r + a[1]) * r + a[0] * q
               / (((((b[5] * r + b[4]) * r + b[3]) * r + b[2]) * r + b[1]) * r + 1.0);
    }

    // Rational approximation for upper region
    let q = (-2.0 * (1.0 - p).ln()).sqrt();
    -(((((c[6] * q + c[5]) * q + c[4]) * q + c[3]) * q + c[2]) * q + c[1]) * q + c[0]
     / ((((d[4] * q + d[3]) * q + d[2]) * q + d[1]) * q + 1.0)
}

/// Calculate Conditional VaR (Expected Shortfall)
pub fn conditional_var(returns: &[f64], confidence_level: f64) -> Result<f64, VmError> {
    let var_threshold = historical_var(returns, confidence_level)?;

    let tail_losses: Vec<f64> = returns.iter()
        .filter(|&&r| -r >= var_threshold)
        .map(|&r| -r)
        .collect();

    if tail_losses.is_empty() {
        return Ok(var_threshold);
    }

    let cvar = tail_losses.iter().sum::<f64>() / tail_losses.len() as f64;
    Ok(cvar)
}

/// Calculate Sharpe ratio
pub fn sharpe_ratio_calc(returns: &[f64], risk_free_rate: f64) -> Result<f64, VmError> {
    if returns.is_empty() {
        return Err(VmError::Runtime("Expected non-empty returns list".to_string()));
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate;

    let variance = returns.iter()
        .map(|x| (x - mean_return).powi(2))
        .sum::<f64>() / (returns.len() - 1) as f64;
    let std_dev = variance.sqrt();

    if std_dev == 0.0 {
        return Err(VmError::Runtime("Zero volatility in Sharpe ratio calculation".to_string()));
    }

    // Annualize if needed (assuming daily returns)
    let sharpe = (excess_return * SQRT_TRADING_DAYS) / (std_dev * SQRT_TRADING_DAYS);
    Ok(sharpe)
}

/// Calculate beta coefficient
pub fn beta_calc(asset_returns: &[f64], market_returns: &[f64]) -> Result<f64, VmError> {
    if asset_returns.len() != market_returns.len() {
        return Err(VmError::Runtime("Expected equal length return series".to_string()));
    }

    if asset_returns.is_empty() {
        return Err(VmError::Runtime("Expected non-empty returns lists".to_string()));
    }

    let asset_mean = asset_returns.iter().sum::<f64>() / asset_returns.len() as f64;
    let market_mean = market_returns.iter().sum::<f64>() / market_returns.len() as f64;

    let mut covariance = 0.0;
    let mut market_variance = 0.0;

    for i in 0..asset_returns.len() {
        let asset_dev = asset_returns[i] - asset_mean;
        let market_dev = market_returns[i] - market_mean;
        
        covariance += asset_dev * market_dev;
        market_variance += market_dev * market_dev;
    }

    covariance /= (asset_returns.len() - 1) as f64;
    market_variance /= (asset_returns.len() - 1) as f64;

    if market_variance == 0.0 {
        return Err(VmError::Runtime("Zero market variance in beta calculation".to_string()));
    }

    let beta = covariance / market_variance;
    Ok(beta)
}

/// Calculate correlation coefficient
pub fn correlation_calc(x: &[f64], y: &[f64]) -> Result<f64, VmError> {
    if x.len() != y.len() {
        return Err(VmError::Runtime("Expected equal length series".to_string()));
    }

    if x.is_empty() {
        return Err(VmError::Runtime("Expected non-empty series".to_string()));
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dev_x = x[i] - mean_x;
        let dev_y = y[i] - mean_y;
        
        numerator += dev_x * dev_y;
        sum_sq_x += dev_x * dev_x;
        sum_sq_y += dev_y * dev_y;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        return Err(VmError::Runtime("Zero variance in correlation calculation".to_string()));
    }

    let correlation = numerator / denominator;
    Ok(correlation)
}

/// Calculate Treynor ratio
pub fn treynor_ratio_calc(returns: &[f64], market_returns: &[f64], risk_free_rate: f64) -> Result<f64, VmError> {
    let beta = beta_calc(returns, market_returns)?;
    
    if beta == 0.0 {
        return Err(VmError::Runtime("Zero beta in Treynor ratio calculation".to_string()));
    }

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let excess_return = mean_return - risk_free_rate;
    
    let treynor = excess_return / beta;
    Ok(treynor)
}

/// Calculate Information ratio
pub fn information_ratio_calc(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> Result<f64, VmError> {
    if portfolio_returns.len() != benchmark_returns.len() {
        return Err(VmError::Runtime("Expected equal length return series".to_string()));
    }

    let active_returns: Vec<f64> = portfolio_returns.iter()
        .zip(benchmark_returns.iter())
        .map(|(p, b)| p - b)
        .collect();

    let mean_active_return = active_returns.iter().sum::<f64>() / active_returns.len() as f64;
    
    let tracking_error_variance = active_returns.iter()
        .map(|r| (r - mean_active_return).powi(2))
        .sum::<f64>() / (active_returns.len() - 1) as f64;
    
    let tracking_error = tracking_error_variance.sqrt();

    if tracking_error == 0.0 {
        return Err(VmError::Runtime("Zero tracking error in Information ratio calculation".to_string()));
    }

    let info_ratio = mean_active_return / tracking_error;
    Ok(info_ratio)
}

/// Calculate Maximum Drawdown
pub fn max_drawdown_calc(prices: &[f64]) -> Result<f64, VmError> {
    if prices.is_empty() {
        return Err(VmError::Runtime("Expected non-empty price series".to_string()));
    }

    let mut peak = prices[0];
    let mut max_dd = 0.0;

    for &price in prices.iter() {
        if price > peak {
            peak = price;
        }
        
        let drawdown = (peak - price) / peak;
        if drawdown > max_dd {
            max_dd = drawdown;
        }
    }

    Ok(max_dd)
}

/// VM integration functions

/// Value at Risk wrapper function
pub fn value_at_risk(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let returns = extract_list_of_reals(&args[0], "returns")?;
    let confidence_level = extract_real(&args[1], "confidence level")?;

    let var = historical_var(&returns, confidence_level)?;
    Ok(Value::Real(var))
}

/// Conditional VaR wrapper function
pub fn conditional_var_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let returns = extract_list_of_reals(&args[0], "returns")?;
    let confidence_level = extract_real(&args[1], "confidence level")?;

    let cvar = conditional_var(&returns, confidence_level)?;
    Ok(Value::Real(cvar))
}

/// Sharpe ratio wrapper function
pub fn sharpe_ratio(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let returns = extract_list_of_reals(&args[0], "returns")?;
    let risk_free_rate = extract_real(&args[1], "risk-free rate")?;

    let sharpe = sharpe_ratio_calc(&returns, risk_free_rate)?;
    Ok(Value::Real(sharpe))
}

/// Beta calculation wrapper function
pub fn beta_calculation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let asset_returns = extract_list_of_reals(&args[0], "asset returns")?;
    let market_returns = extract_list_of_reals(&args[1], "market returns")?;

    let beta = beta_calc(&asset_returns, &market_returns)?;
    Ok(Value::Real(beta))
}

/// Correlation calculation wrapper function
pub fn correlation_calculation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let series1 = extract_list_of_reals(&args[0], "series1")?;
    let series2 = extract_list_of_reals(&args[1], "series2")?;

    let correlation = correlation_calc(&series1, &series2)?;
    Ok(Value::Real(correlation))
}

/// Treynor ratio wrapper function
pub fn treynor_ratio(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime(format!("Expected 3 arguments, got {}", args.len())));
    }

    let returns = extract_list_of_reals(&args[0], "returns")?;
    let market_returns = extract_list_of_reals(&args[1], "market returns")?;
    let risk_free_rate = extract_real(&args[2], "risk-free rate")?;

    let treynor = treynor_ratio_calc(&returns, &market_returns, risk_free_rate)?;
    Ok(Value::Real(treynor))
}

/// Information ratio wrapper function
pub fn information_ratio(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let portfolio_returns = extract_list_of_reals(&args[0], "portfolio returns")?;
    let benchmark_returns = extract_list_of_reals(&args[1], "benchmark returns")?;

    let info_ratio = information_ratio_calc(&portfolio_returns, &benchmark_returns)?;
    Ok(Value::Real(info_ratio))
}

/// Maximum drawdown wrapper function
pub fn max_drawdown(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(format!("Expected 1 argument, got {}", args.len())));
    }

    let prices = extract_list_of_reals(&args[0], "price series")?;
    let max_dd = max_drawdown_calc(&prices)?;
    Ok(Value::Real(max_dd))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_historical_var() {
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.04, 0.01];
        let var = historical_var(&returns, 0.95).unwrap();
        
        // VaR should be positive (representing loss)
        assert!(var > 0.0, "VaR should be positive");
        assert!(var <= 0.05, "VaR should not exceed maximum loss");
    }

    #[test]
    fn test_conditional_var() {
        let returns = vec![-0.05, -0.02, 0.01, 0.03, -0.01, 0.02, -0.03, 0.04, -0.04, 0.01];
        let cvar = conditional_var(&returns, 0.95).unwrap();
        let var = historical_var(&returns, 0.95).unwrap();
        
        // CVaR should be >= VaR
        assert!(cvar >= var, "CVaR should be greater than or equal to VaR");
    }

    #[test]
    fn test_sharpe_ratio() {
        let returns = vec![0.08, 0.12, 0.15, 0.09, 0.11];
        let risk_free_rate = 0.03;
        
        let sharpe = sharpe_ratio_calc(&returns, risk_free_rate).unwrap();
        assert!(sharpe > 0.0, "Sharpe ratio should be positive for good returns");
    }

    #[test]
    fn test_beta_calculation() {
        // Asset that moves perfectly with market should have beta = 1
        let asset_returns = vec![0.05, 0.10, 0.02, -0.03, 0.08];
        let market_returns = vec![0.05, 0.10, 0.02, -0.03, 0.08];
        
        let beta = beta_calc(&asset_returns, &market_returns).unwrap();
        assert!(approx_eq(beta, 1.0, 0.1), "Beta should be ~1.0 for perfectly correlated asset");
    }

    #[test]
    fn test_correlation() {
        // Perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let corr = correlation_calc(&x, &y).unwrap();
        assert!(approx_eq(corr, 1.0, EPSILON), "Perfect positive correlation should be 1.0");
        
        // Perfect negative correlation
        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr_neg = correlation_calc(&x, &z).unwrap();
        assert!(approx_eq(corr_neg, -1.0, EPSILON), "Perfect negative correlation should be -1.0");
    }

    #[test]
    fn test_max_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 95.0, 120.0, 90.0, 115.0];
        let max_dd = max_drawdown_calc(&prices).unwrap();
        
        // Maximum drawdown from peak of 120 to trough of 90 = 25%
        assert!(approx_eq(max_dd, 0.25, 0.01), "Maximum drawdown should be 25%");
    }

    #[test]
    fn test_treynor_ratio() {
        let returns = vec![0.08, 0.12, 0.15, 0.09, 0.11];
        let market_returns = vec![0.06, 0.10, 0.12, 0.07, 0.09];
        let risk_free_rate = 0.03;
        
        let treynor = treynor_ratio_calc(&returns, &market_returns, risk_free_rate).unwrap();
        assert!(treynor > 0.0, "Treynor ratio should be positive for good returns");
    }

    #[test]
    fn test_information_ratio() {
        let portfolio_returns = vec![0.08, 0.12, 0.15, 0.09, 0.11];
        let benchmark_returns = vec![0.06, 0.10, 0.12, 0.07, 0.09];
        
        let info_ratio = information_ratio_calc(&portfolio_returns, &benchmark_returns).unwrap();
        // Portfolio outperforming benchmark should have positive IR
        assert!(info_ratio > 0.0, "Information ratio should be positive for outperforming portfolio");
    }

    #[test]
    fn test_vm_integration_var() {
        let returns = vec![
            Value::Real(-0.05), Value::Real(-0.02), Value::Real(0.01), 
            Value::Real(0.03), Value::Real(-0.01)
        ];
        let args = vec![Value::List(returns), Value::Real(0.95)];
        
        let result = value_at_risk(&args).unwrap();
        if let Value::Real(var) = result {
            assert!(var > 0.0, "VaR should be positive");
        } else {
            panic!("Expected Real value");
        }
    }

    #[test]
    fn test_vm_integration_sharpe() {
        let returns = vec![
            Value::Real(0.08), Value::Real(0.12), Value::Real(0.15), 
            Value::Real(0.09), Value::Real(0.11)
        ];
        let args = vec![Value::List(returns), Value::Real(0.03)];
        
        let result = sharpe_ratio(&args).unwrap();
        if let Value::Real(sharpe) = result {
            assert!(sharpe > 0.0, "Sharpe ratio should be positive");
        } else {
            panic!("Expected Real value");
        }
    }
}