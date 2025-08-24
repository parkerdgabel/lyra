//! Portfolio Optimization and Management
//!
//! Implementation of portfolio optimization techniques including:
//! - Markowitz mean-variance optimization
//! - CAPM (Capital Asset Pricing Model)
//! - Bond pricing and duration calculations
//! - Efficient frontier construction
//! - Portfolio performance metrics

use super::{extract_real, extract_list_of_reals, extract_matrix, extract_integer};
use crate::vm::{Value, VmResult, VmError};

/// Calculate portfolio expected return
pub fn portfolio_expected_return(weights: &[f64], returns_matrix: &[Vec<f64>]) -> f64 {
    if weights.len() != returns_matrix.len() {
        return 0.0;
    }

    let mut portfolio_return = 0.0;
    
    for (i, &weight) in weights.iter().enumerate() {
        let asset_mean = returns_matrix[i].iter().sum::<f64>() / returns_matrix[i].len() as f64;
        portfolio_return += weight * asset_mean;
    }
    
    portfolio_return
}

/// Calculate portfolio variance
pub fn portfolio_variance(weights: &[f64], returns_matrix: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    if n != returns_matrix.len() {
        return 0.0;
    }

    // Calculate covariance matrix
    let covariance_matrix = calculate_covariance_matrix(returns_matrix);
    
    let mut variance = 0.0;
    
    // Portfolio variance = w^T * Sigma * w
    for i in 0..n {
        for j in 0..n {
            variance += weights[i] * weights[j] * covariance_matrix[i][j];
        }
    }
    
    variance
}

/// Calculate covariance matrix from returns
pub fn calculate_covariance_matrix(returns_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = returns_matrix.len();
    let mut covariance_matrix = vec![vec![0.0; n]; n];
    
    // Calculate means
    let means: Vec<f64> = returns_matrix.iter()
        .map(|returns| returns.iter().sum::<f64>() / returns.len() as f64)
        .collect();
    
    let num_observations = returns_matrix[0].len();
    
    for i in 0..n {
        for j in 0..n {
            let mut covariance = 0.0;
            
            for t in 0..num_observations {
                let dev_i = returns_matrix[i][t] - means[i];
                let dev_j = returns_matrix[j][t] - means[j];
                covariance += dev_i * dev_j;
            }
            
            covariance_matrix[i][j] = covariance / (num_observations - 1) as f64;
        }
    }
    
    covariance_matrix
}

/// Markowitz mean-variance optimization
pub fn markowitz_mean_variance_optimization(
    returns_matrix: &[Vec<f64>], 
    target_return: f64
) -> Result<Vec<f64>, VmError> {
    let n = returns_matrix.len();
    if n == 0 {
        return Err(VmError::Runtime("Expected non-empty returns matrix".to_string()));
    }

    // For simplicity, we'll use a quadratic programming approximation
    // In practice, you'd use a proper QP solver like OSQP or similar
    
    let expected_returns: Vec<f64> = returns_matrix.iter()
        .map(|returns| returns.iter().sum::<f64>() / returns.len() as f64)
        .collect();
    
    let covariance_matrix = calculate_covariance_matrix(returns_matrix);
    
    // Simple analytical solution for 2-asset case
    if n == 2 {
        return solve_two_asset_optimization(&expected_returns, &covariance_matrix, target_return);
    }
    
    // For multi-asset case, use iterative approach
    solve_multi_asset_optimization(&expected_returns, &covariance_matrix, target_return)
}

/// Solve optimization for two-asset portfolio analytically
fn solve_two_asset_optimization(
    expected_returns: &[f64],
    _covariance_matrix: &[Vec<f64>],
    target_return: f64,
) -> Result<Vec<f64>, VmError> {
    let mu1 = expected_returns[0];
    let mu2 = expected_returns[1];
    
    if (mu2 - mu1).abs() < 1e-10 {
        return Ok(vec![0.5, 0.5]); // Equal weights if returns are the same
    }
    
    let w1 = (target_return - mu2) / (mu1 - mu2);
    let _w2 = 1.0 - w1;
    
    // Ensure weights are valid
    let w1 = w1.max(0.0).min(1.0);
    let w2 = 1.0 - w1;
    
    Ok(vec![w1, w2])
}

/// Solve optimization for multi-asset portfolio using iterative method
fn solve_multi_asset_optimization(
    expected_returns: &[f64],
    covariance_matrix: &[Vec<f64>],
    target_return: f64,
) -> Result<Vec<f64>, VmError> {
    let n = expected_returns.len();
    let mut weights = vec![1.0 / n as f64; n]; // Start with equal weights
    
    // Simple gradient descent approach
    let learning_rate = 0.01;
    let max_iterations = 1000;
    let tolerance = 1e-8;
    
    for iteration in 0..max_iterations {
        let current_return = weights.iter()
            .zip(expected_returns.iter())
            .map(|(w, r)| w * r)
            .sum::<f64>();
        
        let return_error = current_return - target_return;
        
        if return_error.abs() < tolerance {
            break;
        }
        
        // Gradient of portfolio variance
        let mut variance_gradient = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                variance_gradient[i] += 2.0 * weights[j] * covariance_matrix[i][j];
            }
        }
        
        // Update weights using gradient descent with return constraint
        let mut new_weights = vec![0.0; n];
        let mut weight_sum = 0.0;
        
        for i in 0..n {
            // Adjust weight based on return target and variance gradient
            let return_adjustment = if return_error.abs() > tolerance {
                learning_rate * return_error * (expected_returns[i] - current_return)
            } else {
                0.0
            };
            
            new_weights[i] = (weights[i] + return_adjustment - learning_rate * variance_gradient[i])
                .max(0.0); // Non-negativity constraint
            
            weight_sum += new_weights[i];
        }
        
        // Normalize weights to sum to 1
        if weight_sum > 0.0 {
            for w in &mut new_weights {
                *w /= weight_sum;
            }
        } else {
            // Fallback to equal weights
            new_weights = vec![1.0 / n as f64; n];
        }
        
        // Check for convergence
        let weight_change: f64 = weights.iter()
            .zip(new_weights.iter())
            .map(|(old, new)| (old - new).abs())
            .sum();
        
        weights = new_weights;
        
        if weight_change < tolerance {
            break;
        }
        
        if iteration == max_iterations - 1 {
            return Err(VmError::Runtime("Portfolio optimization did not converge".to_string()));
        }
    }
    
    Ok(weights)
}

/// CAPM model calculation
pub fn capm_expected_return(risk_free_rate: f64, beta: f64, market_return: f64) -> f64 {
    risk_free_rate + beta * (market_return - risk_free_rate)
}

/// Bond pricing calculation
pub fn bond_pricing(
    face_value: f64,
    coupon_rate: f64,
    yield_to_maturity: f64,
    time_to_maturity: f64,
    payment_frequency: i32,
) -> f64 {
    let periods = (time_to_maturity * payment_frequency as f64).floor() as i32;
    let periodic_coupon = face_value * coupon_rate / payment_frequency as f64;
    let periodic_yield = yield_to_maturity / payment_frequency as f64;
    
    let mut price = 0.0;
    
    // Present value of coupon payments
    for period in 1..=periods {
        let discount_factor = (1.0 + periodic_yield).powi(period);
        price += periodic_coupon / discount_factor;
    }
    
    // Present value of face value
    if periods > 0 {
        let discount_factor = (1.0 + periodic_yield).powi(periods);
        price += face_value / discount_factor;
    }
    
    price
}

/// Bond duration calculation (Macaulay duration)
pub fn bond_duration(
    face_value: f64,
    coupon_rate: f64,
    yield_to_maturity: f64,
    time_to_maturity: f64,
    payment_frequency: i32,
) -> f64 {
    let periods = (time_to_maturity * payment_frequency as f64).floor() as i32;
    let periodic_coupon = face_value * coupon_rate / payment_frequency as f64;
    let periodic_yield = yield_to_maturity / payment_frequency as f64;
    
    let bond_price = bond_pricing(face_value, coupon_rate, yield_to_maturity, time_to_maturity, payment_frequency);
    
    let mut weighted_time = 0.0;
    
    // Weighted time of coupon payments
    for period in 1..=periods {
        let time = period as f64 / payment_frequency as f64;
        let discount_factor = (1.0 + periodic_yield).powi(period);
        let present_value = periodic_coupon / discount_factor;
        weighted_time += time * present_value;
    }
    
    // Weighted time of face value payment
    if periods > 0 {
        let time = periods as f64 / payment_frequency as f64;
        let discount_factor = (1.0 + periodic_yield).powi(periods);
        let present_value = face_value / discount_factor;
        weighted_time += time * present_value;
    }
    
    weighted_time / bond_price
}

/// Bond convexity calculation
pub fn bond_convexity(
    face_value: f64,
    coupon_rate: f64,
    yield_to_maturity: f64,
    time_to_maturity: f64,
    payment_frequency: i32,
) -> f64 {
    let periods = (time_to_maturity * payment_frequency as f64).floor() as i32;
    let periodic_coupon = face_value * coupon_rate / payment_frequency as f64;
    let periodic_yield = yield_to_maturity / payment_frequency as f64;
    
    let bond_price = bond_pricing(face_value, coupon_rate, yield_to_maturity, time_to_maturity, payment_frequency);
    
    let mut convexity_sum = 0.0;
    
    // Convexity from coupon payments
    for period in 1..=periods {
        let t = period as f64;
        let discount_factor = (1.0 + periodic_yield).powi(period);
        let present_value = periodic_coupon / discount_factor;
        convexity_sum += present_value * t * (t + 1.0);
    }
    
    // Convexity from face value payment
    if periods > 0 {
        let t = periods as f64;
        let discount_factor = (1.0 + periodic_yield).powi(periods);
        let present_value = face_value / discount_factor;
        convexity_sum += present_value * t * (t + 1.0);
    }
    
    convexity_sum / (bond_price * (1.0 + periodic_yield).powi(2) * (payment_frequency as f64).powi(2))
}

/// Construct efficient frontier
pub fn construct_efficient_frontier(
    returns_matrix: &[Vec<f64>], 
    num_points: i32
) -> Result<Vec<(f64, f64, Vec<f64>)>, VmError> {
    let expected_returns: Vec<f64> = returns_matrix.iter()
        .map(|returns| returns.iter().sum::<f64>() / returns.len() as f64)
        .collect();
    
    let min_return = expected_returns.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_return = expected_returns.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    let mut frontier_points = Vec::new();
    
    for i in 0..num_points {
        let target_return = min_return + (max_return - min_return) * (i as f64) / (num_points - 1) as f64;
        
        match markowitz_mean_variance_optimization(returns_matrix, target_return) {
            Ok(weights) => {
                let variance = portfolio_variance(&weights, returns_matrix);
                let risk = variance.sqrt();
                frontier_points.push((target_return, risk, weights));
            }
            Err(_) => {
                // Skip points that don't converge
                continue;
            }
        }
    }
    
    Ok(frontier_points)
}

/// Yield curve interpolation using linear interpolation
pub fn yield_curve_interpolation_calc(
    maturities: &[f64],
    yields: &[f64],
    target_maturity: f64,
) -> Result<f64, VmError> {
    if maturities.len() != yields.len() {
        return Err(VmError::Runtime("Expected equal length maturity and yield arrays".to_string()));
    }

    if maturities.is_empty() {
        return Err(VmError::Runtime("Expected non-empty arrays".to_string()));
    }

    // Find bracketing points
    let mut left_idx = 0;
    let mut right_idx = maturities.len() - 1;

    for i in 0..maturities.len() {
        if maturities[i] <= target_maturity {
            left_idx = i;
        }
        if maturities[i] >= target_maturity && right_idx == maturities.len() - 1 {
            right_idx = i;
        }
    }

    // Exact match
    if maturities[left_idx] == target_maturity {
        return Ok(yields[left_idx]);
    }

    // Extrapolation cases
    if target_maturity < maturities[left_idx] {
        return Ok(yields[left_idx]);
    }
    if target_maturity > maturities[right_idx] {
        return Ok(yields[right_idx]);
    }

    // Linear interpolation
    if left_idx == right_idx {
        return Ok(yields[left_idx]);
    }

    let t1 = maturities[left_idx];
    let t2 = maturities[right_idx];
    let y1 = yields[left_idx];
    let y2 = yields[right_idx];

    let interpolated_yield = y1 + (y2 - y1) * (target_maturity - t1) / (t2 - t1);
    Ok(interpolated_yield)
}

// VM integration functions

/// Markowitz optimization wrapper function
pub fn markowitz_optimization(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let returns_matrix = extract_matrix(&args[0], "returns matrix")?;
    let target_return = extract_real(&args[1], "target return")?;

    let optimal_weights = markowitz_mean_variance_optimization(&returns_matrix, target_return)?;
    
    Ok(Value::List(optimal_weights.into_iter().map(Value::Real).collect()))
}

/// CAPM model wrapper function
pub fn capm_model(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime(format!("Expected 3 arguments, got {}", args.len())));
    }

    let risk_free_rate = extract_real(&args[0], "risk-free rate")?;
    let beta = extract_real(&args[1], "beta")?;
    let market_return = extract_real(&args[2], "market return")?;

    let expected_return = capm_expected_return(risk_free_rate, beta, market_return);
    Ok(Value::Real(expected_return))
}

/// Bond price wrapper function
pub fn bond_price(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::Runtime(format!("Expected 4 arguments, got {}", args.len())));
    }

    let face_value = extract_real(&args[0], "face value")?;
    let coupon_rate = extract_real(&args[1], "coupon rate")?;
    let yield_to_maturity = extract_real(&args[2], "yield to maturity")?;
    let time_to_maturity = extract_real(&args[3], "time to maturity")?;

    let price = bond_pricing(face_value, coupon_rate, yield_to_maturity, time_to_maturity, 2);
    Ok(Value::Real(price))
}

/// Bond duration wrapper function
pub fn bond_duration_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::Runtime(format!("Expected 4 arguments, got {}", args.len())));
    }

    let face_value = extract_real(&args[0], "face value")?;
    let coupon_rate = extract_real(&args[1], "coupon rate")?;
    let yield_to_maturity = extract_real(&args[2], "yield to maturity")?;
    let time_to_maturity = extract_real(&args[3], "time to maturity")?;

    let duration = bond_duration(face_value, coupon_rate, yield_to_maturity, time_to_maturity, 2);
    Ok(Value::Real(duration))
}

/// Bond convexity wrapper function
pub fn bond_convexity_fn(args: &[Value]) -> VmResult<Value> {
    if args.len() != 4 {
        return Err(VmError::Runtime(format!("Expected 4 arguments, got {}", args.len())));
    }

    let face_value = extract_real(&args[0], "face value")?;
    let coupon_rate = extract_real(&args[1], "coupon rate")?;
    let yield_to_maturity = extract_real(&args[2], "yield to maturity")?;
    let time_to_maturity = extract_real(&args[3], "time to maturity")?;

    let convexity = bond_convexity(face_value, coupon_rate, yield_to_maturity, time_to_maturity, 2);
    Ok(Value::Real(convexity))
}

/// Efficient frontier wrapper function
pub fn efficient_frontier(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let returns_matrix = extract_matrix(&args[0], "returns matrix")?;
    let num_points = extract_integer(&args[1], "number of points")? as i32;

    let frontier_points = construct_efficient_frontier(&returns_matrix, num_points)?;
    
    let result: Vec<Value> = frontier_points.into_iter()
        .map(|(ret, risk, weights)| {
            Value::List(vec![
                Value::Real(ret),
                Value::Real(risk),
                Value::List(weights.into_iter().map(Value::Real).collect()),
            ])
        })
        .collect();
    
    Ok(Value::List(result))
}

/// Portfolio performance wrapper function
pub fn portfolio_performance(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime(format!("Expected 2 arguments, got {}", args.len())));
    }

    let weights = extract_list_of_reals(&args[0], "weights")?;
    let returns_matrix = extract_matrix(&args[1], "returns matrix")?;

    let expected_return = portfolio_expected_return(&weights, &returns_matrix);
    let variance = portfolio_variance(&weights, &returns_matrix);
    let risk = variance.sqrt();
    
    // Simple Sharpe ratio calculation (assuming 0 risk-free rate)
    let sharpe_ratio = if risk > 0.0 { expected_return / risk } else { 0.0 };

    Ok(Value::List(vec![
        Value::Real(expected_return),
        Value::Real(risk),
        Value::Real(sharpe_ratio),
    ]))
}

/// Yield curve interpolation wrapper function
pub fn yield_curve_interpolation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::Runtime(format!("Expected 3 arguments, got {}", args.len())));
    }

    let maturities = extract_list_of_reals(&args[0], "maturities")?;
    let yields = extract_list_of_reals(&args[1], "yields")?;
    let target_maturity = extract_real(&args[2], "target maturity")?;

    let interpolated_yield = yield_curve_interpolation_calc(&maturities, &yields, target_maturity)?;
    Ok(Value::Real(interpolated_yield))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_portfolio_expected_return() {
        let weights = vec![0.6, 0.4];
        let returns_matrix = vec![
            vec![0.08, 0.12, 0.10],
            vec![0.06, 0.08, 0.07],
        ];
        
        let expected_return = portfolio_expected_return(&weights, &returns_matrix);
        let expected = 0.6 * 0.10 + 0.4 * 0.07; // 0.088
        
        assert!(approx_eq(expected_return, expected, EPSILON));
    }

    #[test]
    fn test_portfolio_variance() {
        let weights = vec![0.5, 0.5];
        let returns_matrix = vec![
            vec![0.10, 0.12],
            vec![0.08, 0.06],
        ];
        
        let variance = portfolio_variance(&weights, &returns_matrix);
        assert!(variance >= 0.0, "Variance should be non-negative");
    }

    #[test]
    fn test_covariance_matrix() {
        let returns_matrix = vec![
            vec![0.10, 0.12, 0.08],
            vec![0.08, 0.06, 0.10],
        ];
        
        let cov_matrix = calculate_covariance_matrix(&returns_matrix);
        
        assert_eq!(cov_matrix.len(), 2);
        assert_eq!(cov_matrix[0].len(), 2);
        
        // Diagonal elements should be variances (positive)
        assert!(cov_matrix[0][0] > 0.0);
        assert!(cov_matrix[1][1] > 0.0);
        
        // Matrix should be symmetric
        assert!(approx_eq(cov_matrix[0][1], cov_matrix[1][0], EPSILON));
    }

    #[test]
    fn test_two_asset_optimization() {
        let expected_returns = vec![0.10, 0.08];
        let covariance_matrix = vec![
            vec![0.04, 0.01],
            vec![0.01, 0.02],
        ];
        
        let optimal_weights = solve_two_asset_optimization(&expected_returns, &covariance_matrix, 0.09)
            .unwrap();
        
        // Weights should sum to 1
        let weight_sum: f64 = optimal_weights.iter().sum();
        assert!(approx_eq(weight_sum, 1.0, EPSILON));
        
        // All weights should be non-negative
        for &weight in &optimal_weights {
            assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_capm_model() {
        let risk_free_rate = 0.03;
        let beta = 1.2;
        let market_return = 0.12;
        
        let expected_return = capm_expected_return(risk_free_rate, beta, market_return);
        let expected = 0.03 + 1.2 * (0.12 - 0.03); // 0.138
        
        assert!(approx_eq(expected_return, expected, EPSILON));
    }

    #[test]
    fn test_bond_pricing() {
        let price = bond_pricing(100.0, 0.05, 0.06, 5.0, 2);
        
        // Bond with coupon rate lower than yield should trade at discount
        assert!(price < 100.0, "Bond should trade at discount when coupon < yield");
        assert!(price > 0.0, "Bond price should be positive");
    }

    #[test]
    fn test_bond_duration() {
        let duration = bond_duration(100.0, 0.05, 0.06, 5.0, 2);
        
        // Duration should be positive and less than maturity
        assert!(duration > 0.0, "Duration should be positive");
        assert!(duration < 5.0, "Duration should be less than maturity for coupon bond");
    }

    #[test]
    fn test_bond_convexity() {
        let convexity = bond_convexity(100.0, 0.05, 0.06, 5.0, 2);
        
        // Convexity should be positive
        assert!(convexity > 0.0, "Convexity should be positive");
    }

    #[test]
    fn test_yield_curve_interpolation() {
        let maturities = vec![1.0, 2.0, 5.0, 10.0];
        let yields = vec![0.02, 0.025, 0.03, 0.035];
        
        // Interpolate at 3 years
        let interpolated = yield_curve_interpolation_calc(&maturities, &yields, 3.0).unwrap();
        
        // Should be between 2.5% and 3.0%
        assert!(interpolated > 0.025 && interpolated < 0.03, 
               "Interpolated yield should be between bracketing points");
    }

    #[test]
    fn test_efficient_frontier() {
        let returns_matrix = vec![
            vec![0.08, 0.12, 0.10],
            vec![0.06, 0.08, 0.07],
        ];
        
        let frontier = construct_efficient_frontier(&returns_matrix, 5).unwrap();
        
        assert!(frontier.len() <= 5, "Should have at most 5 frontier points");
        
        // Check that returns are in ascending order
        for i in 1..frontier.len() {
            assert!(frontier[i].0 >= frontier[i-1].0, 
                   "Frontier returns should be in ascending order");
        }
    }

    #[test]
    fn test_vm_integration_capm() {
        let args = vec![
            Value::Real(0.03),
            Value::Real(1.2),
            Value::Real(0.12),
        ];
        
        let result = capm_model(&args).unwrap();
        if let Value::Real(expected_return) = result {
            assert!(approx_eq(expected_return, 0.138, 0.001));
        } else {
            panic!("Expected Real value");
        }
    }

    #[test]
    fn test_vm_integration_bond_price() {
        let args = vec![
            Value::Real(100.0),
            Value::Real(0.05),
            Value::Real(0.06),
            Value::Real(5.0),
        ];
        
        let result = bond_price(&args).unwrap();
        if let Value::Real(price) = result {
            assert!(price > 0.0 && price < 100.0, "Bond should trade at discount");
        } else {
            panic!("Expected Real value");
        }
    }
}
