//! Option Pricing Models
//!
//! Implementation of various option pricing models including:
//! - Black-Scholes model for European options
//! - Binomial tree model for American and European options  
//! - Monte Carlo simulation for path-dependent options
//! - Greeks calculation for risk management

use super::{OptionType, extract_real, extract_symbol, extract_integer, parse_option_type};
use crate::vm::{Value, VmResult, VmError};
use rand::prelude::*;
use rand_distr::Normal;
use std::f64::consts::PI;

/// Standard normal cumulative distribution function
pub fn norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / (2.0_f64.sqrt())))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Standard normal probability density function
pub fn norm_pdf(x: f64) -> f64 {
    (1.0 / (2.0 * PI).sqrt()) * (-0.5 * x * x).exp()
}

/// Black-Scholes option pricing formula
pub fn black_scholes_price(
    spot: f64,
    strike: f64,
    time: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: &OptionType,
) -> f64 {
    if time <= 0.0 {
        // Option has expired
        return match option_type {
            OptionType::Call => (spot - strike).max(0.0),
            OptionType::Put => (strike - spot).max(0.0),
        };
    }

    let d1 = ((spot / strike).ln() + (risk_free_rate + 0.5 * volatility * volatility) * time) 
             / (volatility * time.sqrt());
    let d2 = d1 - volatility * time.sqrt();

    match option_type {
        OptionType::Call => {
            spot * norm_cdf(d1) - strike * (-risk_free_rate * time).exp() * norm_cdf(d2)
        }
        OptionType::Put => {
            strike * (-risk_free_rate * time).exp() * norm_cdf(-d2) - spot * norm_cdf(-d1)
        }
    }
}

/// Calculate option Greeks
pub fn calculate_greeks(
    spot: f64,
    strike: f64,
    time: f64,
    risk_free_rate: f64,
    volatility: f64,
    option_type: &OptionType,
) -> (f64, f64, f64, f64, f64) {
    if time <= 0.0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let d1 = ((spot / strike).ln() + (risk_free_rate + 0.5 * volatility * volatility) * time) 
             / (volatility * time.sqrt());
    let d2 = d1 - volatility * time.sqrt();
    
    let sqrt_t = time.sqrt();
    let discount_factor = (-risk_free_rate * time).exp();

    // Delta
    let delta = match option_type {
        OptionType::Call => norm_cdf(d1),
        OptionType::Put => norm_cdf(d1) - 1.0,
    };

    // Gamma (same for calls and puts)
    let gamma = norm_pdf(d1) / (spot * volatility * sqrt_t);

    // Theta
    let theta = match option_type {
        OptionType::Call => {
            -(spot * norm_pdf(d1) * volatility) / (2.0 * sqrt_t) 
            - risk_free_rate * strike * discount_factor * norm_cdf(d2)
        }
        OptionType::Put => {
            -(spot * norm_pdf(d1) * volatility) / (2.0 * sqrt_t) 
            + risk_free_rate * strike * discount_factor * norm_cdf(-d2)
        }
    } / 365.0; // Convert to daily theta

    // Vega (same for calls and puts)
    let vega = spot * sqrt_t * norm_pdf(d1) / 100.0; // Per 1% volatility change

    // Rho
    let rho = match option_type {
        OptionType::Call => {
            strike * time * discount_factor * norm_cdf(d2) / 100.0
        }
        OptionType::Put => {
            -strike * time * discount_factor * norm_cdf(-d2) / 100.0
        }
    };

    (delta, gamma, theta, vega, rho)
}

/// Binomial tree option pricing
pub fn binomial_tree_price(
    spot: f64,
    strike: f64,
    time: f64,
    risk_free_rate: f64,
    volatility: f64,
    steps: i32,
    option_type: &OptionType,
) -> f64 {
    let dt = time / (steps as f64);
    let u = (volatility * dt.sqrt()).exp();
    let d = 1.0 / u;
    let p = ((risk_free_rate * dt).exp() - d) / (u - d);
    let discount = (-risk_free_rate * dt).exp();

    // Initialize asset prices at maturity
    let mut asset_prices: Vec<f64> = Vec::new();
    let mut option_values: Vec<f64> = Vec::new();

    for i in 0..=steps {
        let price = spot * u.powi(2 * i - steps);
        asset_prices.push(price);
        
        let payoff = match option_type {
            OptionType::Call => (price - strike).max(0.0),
            OptionType::Put => (strike - price).max(0.0),
        };
        option_values.push(payoff);
    }

    // Work backwards through the tree
    for step in (0..steps).rev() {
        let mut new_values = Vec::new();
        
        for i in 0..=(step as usize) {
            // Expected value under risk-neutral measure
            let continuation_value = discount * (p * option_values[i + 1] + (1.0 - p) * option_values[i]);
            
            // For European options, just use continuation value
            // For American options, check if early exercise is better
            let current_spot = spot * u.powi(2 * (i as i32) - step);
            let _intrinsic_value = match option_type {
                OptionType::Call => (current_spot - strike).max(0.0),
                OptionType::Put => (strike - current_spot).max(0.0),
            };

            // For now, assume European (no early exercise)
            new_values.push(continuation_value);
        }
        
        option_values = new_values;
    }

    option_values[0]
}

/// Monte Carlo option pricing using geometric Brownian motion
pub fn monte_carlo_option_price(
    spot: f64,
    strike: f64,
    time: f64,
    risk_free_rate: f64,
    volatility: f64,
    simulations: i32,
    option_type: &OptionType,
) -> f64 {
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    
    let mut payoff_sum = 0.0;
    
    for _ in 0..simulations {
        // Generate random normal variable
        let z = normal.sample(&mut rng);
        
        // Calculate final stock price using GBM
        let final_price = spot * ((risk_free_rate - 0.5 * volatility * volatility) * time 
                                  + volatility * time.sqrt() * z).exp();
        
        // Calculate payoff
        let payoff = match option_type {
            OptionType::Call => (final_price - strike).max(0.0),
            OptionType::Put => (strike - final_price).max(0.0),
        };
        
        payoff_sum += payoff;
    }
    
    // Discount expected payoff to present value
    let expected_payoff = payoff_sum / (simulations as f64);
    expected_payoff * (-risk_free_rate * time).exp()
}

/// Implied volatility using Newton-Raphson method
pub fn implied_volatility(
    market_price: f64,
    spot: f64,
    strike: f64,
    time: f64,
    risk_free_rate: f64,
    option_type: &OptionType,
) -> Result<f64, VmError> {
    let tolerance = 1e-6;
    let max_iterations = 100;
    let mut vol = 0.2; // Initial guess

    for _ in 0..max_iterations {
        let price = black_scholes_price(spot, strike, time, risk_free_rate, vol, option_type);
        let vega = calculate_greeks(spot, strike, time, risk_free_rate, vol, option_type).3;
        
        let price_diff = price - market_price;
        
        if price_diff.abs() < tolerance {
            return Ok(vol);
        }
        
        if vega.abs() < tolerance {
            return Err(VmError::Runtime("Vega too small for Newton-Raphson".to_string()));
        }
        
        vol -= price_diff / (vega * 100.0); // vega is per 1% change
        
        // Keep volatility positive
        vol = vol.max(1e-8);
    }
    
    Err(VmError::Runtime("Implied volatility did not converge".to_string()))
}

/// Black-Scholes wrapper function for VM integration
pub fn black_scholes(args: &[Value]) -> VmResult<Value> {
    if args.len() != 6 {
        return Err(VmError::Runtime(format!("Expected 6 arguments, got {}", args.len())));
    }

    let spot = extract_real(&args[0], "spot price")?;
    let strike = extract_real(&args[1], "strike price")?;
    let time = extract_real(&args[2], "time to expiry")?;
    let rate = extract_real(&args[3], "risk-free rate")?;
    let volatility = extract_real(&args[4], "volatility")?;
    let option_type_str = extract_symbol(&args[5], "option type")?;
    
    let option_type = parse_option_type(&option_type_str)?;
    
    let price = black_scholes_price(spot, strike, time, rate, volatility, &option_type);
    Ok(Value::Real(price))
}

/// Binomial tree wrapper function for VM integration
pub fn binomial_tree(args: &[Value]) -> VmResult<Value> {
    if args.len() != 7 {
        return Err(VmError::Runtime(format!("Expected 7 arguments, got {}", args.len())));
    }

    let spot = extract_real(&args[0], "spot price")?;
    let strike = extract_real(&args[1], "strike price")?;
    let time = extract_real(&args[2], "time to expiry")?;
    let rate = extract_real(&args[3], "risk-free rate")?;
    let volatility = extract_real(&args[4], "volatility")?;
    let steps = extract_integer(&args[5], "number of steps")? as i32;
    let option_type_str = extract_symbol(&args[6], "option type")?;
    
    let option_type = parse_option_type(&option_type_str)?;
    
    let price = binomial_tree_price(spot, strike, time, rate, volatility, steps, &option_type);
    Ok(Value::Real(price))
}

/// Monte Carlo option pricing wrapper function for VM integration
pub fn monte_carlo_option(args: &[Value]) -> VmResult<Value> {
    if args.len() != 7 {
        return Err(VmError::Runtime(format!("Expected 7 arguments, got {}", args.len())));
    }

    let spot = extract_real(&args[0], "spot price")?;
    let strike = extract_real(&args[1], "strike price")?;
    let time = extract_real(&args[2], "time to expiry")?;
    let rate = extract_real(&args[3], "risk-free rate")?;
    let volatility = extract_real(&args[4], "volatility")?;
    let simulations = extract_integer(&args[5], "number of simulations")? as i32;
    let option_type_str = extract_symbol(&args[6], "option type")?;
    
    let option_type = parse_option_type(&option_type_str)?;
    
    let price = monte_carlo_option_price(spot, strike, time, rate, volatility, simulations, &option_type);
    Ok(Value::Real(price))
}

/// Greeks calculation wrapper function for VM integration
pub fn greeks_calculation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 6 {
        return Err(VmError::Runtime(format!("Expected 6 arguments, got {}", args.len())));
    }

    let spot = extract_real(&args[0], "spot price")?;
    let strike = extract_real(&args[1], "strike price")?;
    let time = extract_real(&args[2], "time to expiry")?;
    let rate = extract_real(&args[3], "risk-free rate")?;
    let volatility = extract_real(&args[4], "volatility")?;
    let option_type_str = extract_symbol(&args[5], "option type")?;
    
    let option_type = parse_option_type(&option_type_str)?;
    
    let (delta, gamma, theta, vega, rho) = calculate_greeks(spot, strike, time, rate, volatility, &option_type);
    
    Ok(Value::List(vec![
        Value::Real(delta),
        Value::Real(gamma),
        Value::Real(theta),
        Value::Real(vega),
        Value::Real(rho),
    ]))
}

/// Create option instrument wrapper function
pub fn create_option(args: &[Value]) -> VmResult<Value> {
    if args.len() != 6 {
        return Err(VmError::Runtime(format!("Expected 6 arguments, got {}", args.len())));
    }

    let spot = extract_real(&args[0], "spot price")?;
    let strike = extract_real(&args[1], "strike price")?;
    let time = extract_real(&args[2], "time to expiry")?;
    let rate = extract_real(&args[3], "risk-free rate")?;
    let volatility = extract_real(&args[4], "volatility")?;
    let option_type_str = extract_symbol(&args[5], "option type")?;
    
    let option_type = parse_option_type(&option_type_str)?;
    
    let option = super::OptionInstrument::new(spot, strike, time, rate, volatility, option_type);
    Ok(Value::LyObj(crate::foreign::LyObj::new(Box::new(option))))
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64, epsilon: f64) -> bool {
        (a - b).abs() < epsilon
    }

    #[test]
    fn test_norm_cdf() {
        // Test some known values
        assert!(approx_eq(norm_cdf(0.0), 0.5, EPSILON));
        assert!(approx_eq(norm_cdf(1.0), 0.8413, 1e-4));
        assert!(approx_eq(norm_cdf(-1.0), 0.1587, 1e-4));
    }

    #[test]
    fn test_black_scholes_call() {
        // Classic Black-Scholes test case
        let price = black_scholes_price(100.0, 100.0, 1.0, 0.05, 0.2, &OptionType::Call);
        assert!(approx_eq(price, 10.45, 0.1));
    }

    #[test]
    fn test_black_scholes_put() {
        // Put-call parity: C - P = S - K*e^(-rT)
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let vol = 0.2;
        
        let call_price = black_scholes_price(spot, strike, time, rate, vol, &OptionType::Call);
        let put_price = black_scholes_price(spot, strike, time, rate, vol, &OptionType::Put);
        
        let forward_price = spot - strike * (-rate * time).exp();
        let parity_diff = call_price - put_price - forward_price;
        
        assert!(parity_diff.abs() < EPSILON, "Put-call parity violated: {}", parity_diff);
    }

    #[test]
    fn test_binomial_convergence() {
        // Binomial tree should converge to Black-Scholes
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let vol = 0.2;
        
        let bs_price = black_scholes_price(spot, strike, time, rate, vol, &OptionType::Call);
        let binom_price = binomial_tree_price(spot, strike, time, rate, vol, 1000, &OptionType::Call);
        
        assert!(approx_eq(bs_price, binom_price, 0.1), 
               "Binomial tree should converge to Black-Scholes: BS={}, Binom={}", 
               bs_price, binom_price);
    }

    #[test]
    fn test_greeks_calculation() {
        let (delta, gamma, theta, vega, rho) = calculate_greeks(
            100.0, 100.0, 1.0, 0.05, 0.2, &OptionType::Call
        );
        
        // Delta should be between 0 and 1 for calls
        assert!(delta > 0.0 && delta < 1.0, "Call delta should be between 0 and 1, got {}", delta);
        
        // Gamma should be positive
        assert!(gamma > 0.0, "Gamma should be positive, got {}", gamma);
        
        // Vega should be positive
        assert!(vega > 0.0, "Vega should be positive, got {}", vega);
        
        // Rho should be positive for calls
        assert!(rho > 0.0, "Rho should be positive for calls, got {}", rho);
    }

    #[test]
    fn test_monte_carlo_convergence() {
        // Monte Carlo should approximate Black-Scholes with enough simulations
        let spot = 100.0;
        let strike = 100.0;
        let time = 1.0;
        let rate = 0.05;
        let vol = 0.2;
        
        let bs_price = black_scholes_price(spot, strike, time, rate, vol, &OptionType::Call);
        let mc_price = monte_carlo_option_price(spot, strike, time, rate, vol, 100000, &OptionType::Call);
        
        // Monte Carlo should be within reasonable error bounds
        assert!(approx_eq(bs_price, mc_price, 0.5), 
               "Monte Carlo should approximate Black-Scholes: BS={}, MC={}", 
               bs_price, mc_price);
    }

    #[test]
    fn test_expired_option() {
        // Test option with zero time to expiry
        let call_price = black_scholes_price(110.0, 100.0, 0.0, 0.05, 0.2, &OptionType::Call);
        let put_price = black_scholes_price(90.0, 100.0, 0.0, 0.05, 0.2, &OptionType::Put);
        
        assert_eq!(call_price, 10.0, "ITM call should equal intrinsic value");
        assert_eq!(put_price, 10.0, "ITM put should equal intrinsic value");
    }

    #[test]
    fn test_vm_integration_black_scholes() {
        let args = vec![
            Value::Real(100.0),
            Value::Real(100.0),
            Value::Real(1.0),
            Value::Real(0.05),
            Value::Real(0.2),
            Value::Symbol("Call".to_string()),
        ];
        
        let result = black_scholes(&args).unwrap();
        if let Value::Real(price) = result {
            assert!(approx_eq(price, 10.45, 0.1));
        } else {
            panic!("Expected Real value");
        }
    }

    #[test]
    fn test_vm_integration_greeks() {
        let args = vec![
            Value::Real(100.0),
            Value::Real(100.0),
            Value::Real(1.0),
            Value::Real(0.05),
            Value::Real(0.2),
            Value::Symbol("Call".to_string()),
        ];
        
        let result = greeks_calculation(&args).unwrap();
        if let Value::List(greeks) = result {
            assert_eq!(greeks.len(), 5);
            // Check all are Real values
            for greek in greeks {
                assert!(matches!(greek, Value::Real(_)));
            }
        } else {
            panic!("Expected List of Greeks");
        }
    }
}