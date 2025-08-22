//! Comprehensive Test Suite for Financial Mathematics Module
//!
//! Following TDD principles - these tests define the expected behavior
//! before implementation. All tests should fail initially (RED phase).

use lyra::stdlib::finance::{pricing::*, risk::*, portfolio::*};
use lyra::vm::{Value, VmError};

/// Test helper to create a floating point value
fn float_val(f: f64) -> Value {
    Value::Real(f)
}

/// Test helper to create an integer value
fn int_val(i: i64) -> Value {
    Value::Integer(i)
}

/// Test helper to create a list value
fn list_val(items: Vec<Value>) -> Value {
    Value::List(items)
}

/// Test helper for approximate equality of floating point values
fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
    (a - b).abs() < epsilon
}

#[cfg(test)]
mod pricing_tests {
    use super::*;

    #[test]
    fn test_black_scholes_call_option() {
        // Test Black-Scholes pricing for European call option
        // Parameters: S=100, K=100, T=1, r=0.05, sigma=0.2
        let args = vec![
            float_val(100.0), // Current stock price
            float_val(100.0), // Strike price
            float_val(1.0),   // Time to maturity (years)
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            Value::Symbol("Call".to_string()), // Option type
        ];
        
        let result = black_scholes(&args).expect("Black-Scholes calculation failed");
        
        if let Value::Real(price) = result {
            // Expected price approximately 10.45
            assert!(approx_eq(price, 10.45, 0.1), 
                "Black-Scholes call price should be ~10.45, got {}", price);
        } else {
            panic!("Expected Real value from Black-Scholes");
        }
    }

    #[test]
    fn test_black_scholes_put_option() {
        // Test Black-Scholes pricing for European put option
        let args = vec![
            float_val(100.0), // Current stock price
            float_val(100.0), // Strike price
            float_val(1.0),   // Time to maturity
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            Value::Symbol("Put".to_string()), // Option type
        ];
        
        let result = black_scholes(&args).expect("Black-Scholes put calculation failed");
        
        if let Value::Real(price) = result {
            // Expected price approximately 5.57
            assert!(approx_eq(price, 5.57, 0.1),
                "Black-Scholes put price should be ~5.57, got {}", price);
        } else {
            panic!("Expected Real value from Black-Scholes put");
        }
    }

    #[test]
    fn test_binomial_tree_pricing() {
        // Test binomial tree option pricing
        let args = vec![
            float_val(100.0), // Current stock price
            float_val(100.0), // Strike price  
            float_val(1.0),   // Time to maturity
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            int_val(100),     // Number of steps
            Value::Symbol("Call".to_string()), // Option type
        ];
        
        let result = binomial_tree(&args).expect("Binomial tree calculation failed");
        
        if let Value::Real(price) = result {
            // Should converge to Black-Scholes price
            assert!(approx_eq(price, 10.45, 0.2),
                "Binomial tree price should converge to ~10.45, got {}", price);
        } else {
            panic!("Expected Real value from binomial tree");
        }
    }

    #[test]
    fn test_monte_carlo_option_pricing() {
        // Test Monte Carlo option pricing
        let args = vec![
            float_val(100.0), // Current stock price
            float_val(100.0), // Strike price
            float_val(1.0),   // Time to maturity
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            int_val(10000),   // Number of simulations
            Value::Symbol("Call".to_string()), // Option type
        ];
        
        let result = monte_carlo_option(&args).expect("Monte Carlo calculation failed");
        
        if let Value::Real(price) = result {
            // Should be close to Black-Scholes price (within Monte Carlo error)
            assert!(approx_eq(price, 10.45, 0.5),
                "Monte Carlo price should be close to ~10.45, got {}", price);
        } else {
            panic!("Expected Real value from Monte Carlo");
        }
    }

    #[test]
    fn test_bond_pricing() {
        // Test bond pricing with yield
        let args = vec![
            float_val(100.0), // Face value
            float_val(0.05),  // Coupon rate
            float_val(0.06),  // Yield to maturity
            float_val(5.0),   // Time to maturity
        ];
        
        let result = bond_price(&args).expect("Bond pricing calculation failed");
        
        if let Value::Real(price) = result {
            // Expected price approximately 95.79
            assert!(approx_eq(price, 95.79, 0.1),
                "Bond price should be ~95.79, got {}", price);
        } else {
            panic!("Expected Real value from bond pricing");
        }
    }
}

#[cfg(test)]
mod risk_metrics_tests {
    use super::*;

    #[test]
    fn test_value_at_risk() {
        // Test VaR calculation
        let returns = vec![0.02, -0.01, 0.015, -0.025, 0.01, -0.005, 0.03, -0.02];
        let args = vec![
            list_val(returns.into_iter().map(float_val).collect()),
            float_val(0.05), // 95% confidence level
        ];
        
        let result = value_at_risk(&args).expect("VaR calculation failed");
        
        if let Value::Real(var) = result {
            assert!(var < 0.0, "VaR should be negative, got {}", var);
            assert!(var > -0.05, "VaR should not be too extreme, got {}", var);
        } else {
            panic!("Expected Real value from VaR");
        }
    }

    #[test]
    fn test_conditional_var() {
        // Test Conditional VaR (Expected Shortfall)
        let returns = vec![0.02, -0.01, 0.015, -0.025, 0.01, -0.005, 0.03, -0.02];
        let args = vec![
            list_val(returns.into_iter().map(float_val).collect()),
            float_val(0.05), // 95% confidence level
        ];
        
        let result = conditional_var_fn(&args).expect("CVaR calculation failed");
        
        if let Value::Real(cvar) = result {
            assert!(cvar < 0.0, "CVaR should be negative, got {}", cvar);
        } else {
            panic!("Expected Real value from CVaR");
        }
    }

    #[test]
    fn test_sharpe_ratio() {
        // Test Sharpe ratio calculation
        let returns = vec![0.08, 0.12, 0.15, 0.09, 0.11];
        let args = vec![
            list_val(returns.into_iter().map(float_val).collect()),
            float_val(0.03), // Risk-free rate
        ];
        
        let result = sharpe_ratio(&args).expect("Sharpe ratio calculation failed");
        
        if let Value::Real(sharpe) = result {
            assert!(sharpe > 0.0, "Sharpe ratio should be positive for good returns, got {}", sharpe);
        } else {
            panic!("Expected Real value from Sharpe ratio");
        }
    }

    #[test]
    fn test_beta_calculation() {
        // Test beta calculation (asset vs market)
        let asset_returns = vec![0.08, 0.12, 0.15, 0.09, 0.11];
        let market_returns = vec![0.06, 0.10, 0.12, 0.07, 0.09];
        let args = vec![
            list_val(asset_returns.into_iter().map(float_val).collect()),
            list_val(market_returns.into_iter().map(float_val).collect()),
        ];
        
        let result = beta_calculation(&args).expect("Beta calculation failed");
        
        if let Value::Real(beta) = result {
            assert!(beta > 0.0, "Beta should be positive for correlated assets, got {}", beta);
        } else {
            panic!("Expected Real value from beta");
        }
    }

    #[test]
    fn test_greeks_calculation() {
        // Test Greeks calculation for options
        let args = vec![
            float_val(100.0), // Current stock price
            float_val(100.0), // Strike price
            float_val(1.0),   // Time to maturity
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            Value::Symbol("Call".to_string()), // Option type
        ];
        
        let result = greeks_calculation(&args).expect("Greeks calculation failed");
        
        if let Value::List(greeks) = result {
            assert_eq!(greeks.len(), 5, "Should return 5 Greeks: Delta, Gamma, Theta, Vega, Rho");
            
            // All Greeks should be Real values
            for greek in &greeks {
                assert!(matches!(greek, Value::Real(_)), "All Greeks should be Real values");
            }
        } else {
            panic!("Expected List of Greeks");
        }
    }
}

#[cfg(test)]
mod portfolio_tests {
    use super::*;

    #[test]
    fn test_portfolio_optimization() {
        // Test Markowitz portfolio optimization
        let returns = vec![
            vec![0.08, 0.12, 0.15, 0.09, 0.11], // Asset 1
            vec![0.06, 0.10, 0.12, 0.07, 0.09], // Asset 2
            vec![0.04, 0.08, 0.10, 0.05, 0.07], // Asset 3
        ];
        
        let returns_matrix = list_val(
            returns.into_iter()
                .map(|asset_returns| list_val(asset_returns.into_iter().map(float_val).collect()))
                .collect()
        );
        
        let args = vec![
            returns_matrix,
            float_val(0.10), // Target return
        ];
        
        let result = markowitz_optimization(&args).expect("Portfolio optimization failed");
        
        if let Value::List(weights) = result {
            // Weights should sum to 1.0
            let total_weight: f64 = weights.iter()
                .filter_map(|w| if let Value::Real(weight) = w { Some(*weight) } else { None })
                .sum();
            assert!(approx_eq(total_weight, 1.0, 0.01), 
                "Portfolio weights should sum to 1.0, got {}", total_weight);
        } else {
            panic!("Expected List of portfolio weights");
        }
    }

    #[test]
    fn test_capm_model() {
        // Test Capital Asset Pricing Model
        let args = vec![
            float_val(0.03),  // Risk-free rate
            float_val(1.2),   // Beta
            float_val(0.12),  // Market return
        ];
        
        let result = capm_model(&args).expect("CAPM calculation failed");
        
        if let Value::Real(expected_return) = result {
            // CAPM: R = Rf + β(Rm - Rf) = 0.03 + 1.2(0.12 - 0.03) = 0.138
            assert!(approx_eq(expected_return, 0.138, 0.001),
                "CAPM expected return should be ~0.138, got {}", expected_return);
        } else {
            panic!("Expected Real value from CAPM");
        }
    }

    #[test]
    fn test_efficient_frontier() {
        // Test efficient frontier calculation
        let returns = vec![
            vec![0.08, 0.12, 0.15, 0.09, 0.11], // Asset 1
            vec![0.06, 0.10, 0.12, 0.07, 0.09], // Asset 2
        ];
        
        let returns_matrix = list_val(
            returns.into_iter()
                .map(|asset_returns| list_val(asset_returns.into_iter().map(float_val).collect()))
                .collect()
        );
        
        let args = vec![
            returns_matrix,
            int_val(10), // Number of points on frontier
        ];
        
        let result = efficient_frontier(&args).expect("Efficient frontier calculation failed");
        
        if let Value::List(frontier_points) = result {
            assert_eq!(frontier_points.len(), 10, "Should return 10 frontier points");
            
            // Each point should be a list of [return, risk, weights...]
            for point in &frontier_points {
                if let Value::List(point_data) = point {
                    assert!(point_data.len() >= 2, "Each point should have return, risk, and weights");
                } else {
                    panic!("Each frontier point should be a list");
                }
            }
        } else {
            panic!("Expected List of efficient frontier points");
        }
    }

    #[test]
    fn test_portfolio_performance() {
        // Test portfolio performance metrics
        let weights = vec![0.4, 0.6];
        let returns = vec![
            vec![0.08, 0.12, 0.15, 0.09, 0.11], // Asset 1
            vec![0.06, 0.10, 0.12, 0.07, 0.09], // Asset 2
        ];
        
        let returns_matrix = list_val(
            returns.into_iter()
                .map(|asset_returns| list_val(asset_returns.into_iter().map(float_val).collect()))
                .collect()
        );
        
        let args = vec![
            list_val(weights.into_iter().map(float_val).collect()),
            returns_matrix,
        ];
        
        let result = portfolio_performance(&args).expect("Portfolio performance calculation failed");
        
        if let Value::List(performance) = result {
            assert_eq!(performance.len(), 3, "Should return [return, risk, sharpe_ratio]");
            
            // All performance metrics should be Real values
            for metric in &performance {
                assert!(matches!(metric, Value::Real(_)), "All metrics should be Real values");
            }
        } else {
            panic!("Expected List of performance metrics");
        }
    }

    #[test]
    fn test_yield_curve_interpolation() {
        // Test yield curve interpolation
        let maturities = vec![0.5, 1.0, 2.0, 5.0, 10.0];
        let yields = vec![0.02, 0.025, 0.03, 0.035, 0.04];
        let target_maturity = 3.0;
        
        let args = vec![
            list_val(maturities.into_iter().map(float_val).collect()),
            list_val(yields.into_iter().map(float_val).collect()),
            float_val(target_maturity),
        ];
        
        let result = yield_curve_interpolation(&args).expect("Yield curve interpolation failed");
        
        if let Value::Real(interpolated_yield) = result {
            assert!(interpolated_yield > 0.03 && interpolated_yield < 0.035,
                "Interpolated yield should be between 3% and 3.5%, got {}", interpolated_yield);
        } else {
            panic!("Expected Real value from yield curve interpolation");
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_option_portfolio_hedge() {
        // Test hedging a stock position with options
        let stock_position = 1000.0; // Long 1000 shares
        let stock_price = 100.0;
        let hedge_strike = 95.0; // Put option strike
        
        // Calculate put option delta to determine hedge ratio
        let put_args = vec![
            float_val(stock_price),
            float_val(hedge_strike),
            float_val(0.25),  // 3 months
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            Value::Symbol("Put".to_string()),
        ];
        
        let greeks = greeks_calculation(&put_args).expect("Greeks calculation failed");
        
        if let Value::List(greeks_list) = greeks {
            if let Value::Real(delta) = &greeks_list[0] {
                // Hedge ratio should be negative delta times stock position
                let hedge_ratio = -delta * stock_position;
                assert!(hedge_ratio > 0.0, "Put hedge ratio should be positive, got {}", hedge_ratio);
            }
        }
    }

    #[test] 
    fn test_bond_duration_convexity() {
        // Test bond duration and convexity calculations
        let args = vec![
            float_val(100.0), // Face value
            float_val(0.05),  // Coupon rate
            float_val(0.06),  // Yield
            float_val(5.0),   // Maturity
        ];
        
        let duration_result = bond_duration_fn(&args).expect("Duration calculation failed");
        let convexity_result = bond_convexity_fn(&args).expect("Convexity calculation failed");
        
        if let (Value::Real(duration), Value::Real(convexity)) = (duration_result, convexity_result) {
            assert!(duration > 0.0 && duration < 5.0, "Duration should be between 0 and maturity");
            assert!(convexity > 0.0, "Convexity should be positive");
        }
    }

    #[test]
    fn test_financial_instrument_foreign_objects() {
        // Test that financial instruments are properly wrapped as Foreign objects
        let option_args = vec![
            float_val(100.0), // Spot
            float_val(100.0), // Strike  
            float_val(1.0),   // Maturity
            float_val(0.05),  // Risk-free rate
            float_val(0.2),   // Volatility
            Value::Symbol("Call".to_string()),
        ];
        
        let option_result = create_option(&option_args).expect("Option creation failed");
        
        // Should return a Foreign object wrapped in LyObj
        assert!(matches!(option_result, Value::LyObj(_)), "Option should be wrapped as LyObj");
    }
}

// All functions are now implemented in the finance module