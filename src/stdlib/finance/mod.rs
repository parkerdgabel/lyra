//! Financial Mathematics Module
//!
//! Comprehensive financial mathematics library with option pricing, risk analysis,
//! and portfolio optimization capabilities. All financial instruments are implemented
//! as Foreign objects using the LyObj wrapper pattern.

use crate::vm::{Value, VmError};
use crate::foreign::Foreign;

// Sub-modules
pub mod pricing;
pub mod risk;  
pub mod portfolio;

// Re-export main functions
pub use pricing::*;
pub use risk::*;
pub use portfolio::*;

/// Mathematical constants for financial calculations
pub const TRADING_DAYS_PER_YEAR: f64 = 252.0;
pub const SQRT_TRADING_DAYS: f64 = 15.87401052; // sqrt(252)

/// Financial instrument types as Foreign objects
#[derive(Debug, Clone)]
pub struct OptionInstrument {
    pub spot_price: f64,
    pub strike_price: f64,
    pub time_to_expiry: f64,
    pub risk_free_rate: f64,
    pub volatility: f64,
    pub option_type: OptionType,
    pub exercise_style: ExerciseStyle,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OptionType {
    Call,
    Put,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExerciseStyle {
    European,
    American,
}

impl Foreign for OptionInstrument {
    fn type_name(&self) -> &'static str {
        "OptionInstrument"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "price" => self.calculate_price(args),
            "greeks" => self.calculate_greeks(args),
            "impliedVolatility" => self.implied_volatility(args),
            "getSpotPrice" => Ok(Value::Real(self.spot_price)),
            "getStrike" => Ok(Value::Real(self.strike_price)),
            "getTimeToExpiry" => Ok(Value::Real(self.time_to_expiry)),
            "getVolatility" => Ok(Value::Real(self.volatility)),
            "getRiskFreeRate" => Ok(Value::Real(self.risk_free_rate)),
            "getType" => Ok(Value::Symbol(match self.option_type {
                OptionType::Call => "Call".to_string(),
                OptionType::Put => "Put".to_string(),
            })),
            _ => Err(crate::foreign::ForeignError::UnknownMethod {
                type_name: "OptionInstrument".to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl OptionInstrument {
    pub fn new(
        spot_price: f64,
        strike_price: f64,
        time_to_expiry: f64,
        risk_free_rate: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> Self {
        Self {
            spot_price,
            strike_price,
            time_to_expiry,
            risk_free_rate,
            volatility,
            option_type,
            exercise_style: ExerciseStyle::European,
        }
    }

    pub fn calculate_price(&self, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        // Default to Black-Scholes pricing
        match args.len() {
            0 => {
                let price = pricing::black_scholes_price(
                    self.spot_price,
                    self.strike_price,
                    self.time_to_expiry,
                    self.risk_free_rate,
                    self.volatility,
                    &self.option_type,
                );
                Ok(Value::Real(price))
            }
            1 => {
                // Pricing method specified
                if let Value::Symbol(method) = &args[0] {
                    match method.as_str() {
                        "BlackScholes" => {
                            let price = pricing::black_scholes_price(
                                self.spot_price,
                                self.strike_price,
                                self.time_to_expiry,
                                self.risk_free_rate,
                                self.volatility,
                                &self.option_type,
                            );
                            Ok(Value::Real(price))
                        }
                        "BinomialTree" => {
                            let steps = 100; // Default steps
                            let price = pricing::binomial_tree_price(
                                self.spot_price,
                                self.strike_price,
                                self.time_to_expiry,
                                self.risk_free_rate,
                                self.volatility,
                                steps,
                                &self.option_type,
                            );
                            Ok(Value::Real(price))
                        }
                        "MonteCarlo" => {
                            let simulations = 10000; // Default simulations
                            let price = pricing::monte_carlo_option_price(
                                self.spot_price,
                                self.strike_price,
                                self.time_to_expiry,
                                self.risk_free_rate,
                                self.volatility,
                                simulations,
                                &self.option_type,
                            );
                            Ok(Value::Real(price))
                        }
                        _ => Err(crate::foreign::ForeignError::InvalidArgument(format!("Unknown pricing method: {}", method))),
                    }
                } else {
                    Err(crate::foreign::ForeignError::TypeError {
                        expected: "Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    })
                }
            }
            _ => Err(crate::foreign::ForeignError::ArgumentError {
                expected: 1,
                actual: args.len(),
            }),
        }
    }

    pub fn calculate_greeks(&self, _args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        let greeks = pricing::calculate_greeks(
            self.spot_price,
            self.strike_price,
            self.time_to_expiry,
            self.risk_free_rate,
            self.volatility,
            &self.option_type,
        );
        
        Ok(Value::List(vec![
            Value::Real(greeks.0), // Delta
            Value::Real(greeks.1), // Gamma
            Value::Real(greeks.2), // Theta
            Value::Real(greeks.3), // Vega
            Value::Real(greeks.4), // Rho
        ]))
    }

    pub fn implied_volatility(&self, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        if args.len() != 1 {
            return Err(crate::foreign::ForeignError::ArgumentError {
                expected: 1,
                actual: args.len(),
            });
        }

        let market_price = match &args[0] {
            Value::Real(price) => *price,
            Value::Integer(price) => *price as f64,
            _ => return Err(crate::foreign::ForeignError::TypeError {
                expected: "Real or Integer".to_string(),
                actual: format!("{:?}", args[0]),
            }),
        };

        let implied_vol = pricing::implied_volatility(
            market_price,
            self.spot_price,
            self.strike_price,
            self.time_to_expiry,
            self.risk_free_rate,
            &self.option_type,
        ).map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;

        Ok(Value::Real(implied_vol))
    }
}

/// Bond instrument as Foreign object
#[derive(Debug, Clone)]
pub struct BondInstrument {
    pub face_value: f64,
    pub coupon_rate: f64,
    pub yield_to_maturity: f64,
    pub time_to_maturity: f64,
    pub payment_frequency: i32, // payments per year
}

impl Foreign for BondInstrument {
    fn type_name(&self) -> &'static str {
        "BondInstrument"
    }

    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "price" => Ok(Value::Real(self.calculate_price())),
            "duration" => Ok(Value::Real(self.calculate_duration())),
            "convexity" => Ok(Value::Real(self.calculate_convexity())),
            "yieldToMaturity" => Ok(Value::Real(self.yield_to_maturity)),
            "modifiedDuration" => Ok(Value::Real(self.calculate_modified_duration())),
            "dollarDuration" => Ok(Value::Real(self.calculate_dollar_duration())),
            "getFaceValue" => Ok(Value::Real(self.face_value)),
            "getCouponRate" => Ok(Value::Real(self.coupon_rate)),
            "getTimeToMaturity" => Ok(Value::Real(self.time_to_maturity)),
            _ => Err(crate::foreign::ForeignError::UnknownMethod {
                type_name: "BondInstrument".to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl BondInstrument {
    pub fn new(face_value: f64, coupon_rate: f64, yield_to_maturity: f64, time_to_maturity: f64) -> Self {
        Self {
            face_value,
            coupon_rate,
            yield_to_maturity,
            time_to_maturity,
            payment_frequency: 2, // Semi-annual by default
        }
    }

    fn calculate_price(&self) -> f64 {
        portfolio::bond_pricing(
            self.face_value,
            self.coupon_rate,
            self.yield_to_maturity,
            self.time_to_maturity,
            self.payment_frequency,
        )
    }

    fn calculate_duration(&self) -> f64 {
        portfolio::bond_duration(
            self.face_value,
            self.coupon_rate,
            self.yield_to_maturity,
            self.time_to_maturity,
            self.payment_frequency,
        )
    }

    fn calculate_convexity(&self) -> f64 {
        portfolio::bond_convexity(
            self.face_value,
            self.coupon_rate,
            self.yield_to_maturity,
            self.time_to_maturity,
            self.payment_frequency,
        )
    }

    fn calculate_modified_duration(&self) -> f64 {
        let duration = self.calculate_duration();
        duration / (1.0 + self.yield_to_maturity / (self.payment_frequency as f64))
    }

    fn calculate_dollar_duration(&self) -> f64 {
        let price = self.calculate_price();
        let modified_duration = self.calculate_modified_duration();
        price * modified_duration / 100.0
    }
}

/// Portfolio instrument as Foreign object
#[derive(Debug, Clone)]
pub struct PortfolioInstrument {
    pub weights: Vec<f64>,
    pub assets: Vec<String>,
    pub returns_matrix: Vec<Vec<f64>>,
    pub risk_free_rate: f64,
}

impl Foreign for PortfolioInstrument {
    fn type_name(&self) -> &'static str {
        "PortfolioInstrument"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        match method {
            "expectedReturn" => Ok(Value::Real(self.calculate_expected_return())),
            "variance" => Ok(Value::Real(self.calculate_variance())),
            "volatility" => Ok(Value::Real(self.calculate_volatility())),
            "sharpeRatio" => Ok(Value::Real(self.calculate_sharpe_ratio())),
            "performance" => self.calculate_performance(),
            "optimize" => self.optimize_portfolio(args),
            "getWeights" => Ok(Value::List(self.weights.iter().map(|w| Value::Real(*w)).collect())),
            "getAssets" => Ok(Value::List(self.assets.iter().map(|a| Value::Symbol(a.clone())).collect())),
            "rebalance" => self.rebalance_portfolio(args),
            _ => Err(crate::foreign::ForeignError::UnknownMethod {
                type_name: "PortfolioInstrument".to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl PortfolioInstrument {
    pub fn new(weights: Vec<f64>, returns_matrix: Vec<Vec<f64>>, risk_free_rate: f64) -> Self {
        let assets = (0..weights.len()).map(|i| format!("Asset{}", i + 1)).collect();
        Self {
            weights,
            assets,
            returns_matrix,
            risk_free_rate,
        }
    }

    fn calculate_expected_return(&self) -> f64 {
        portfolio::portfolio_expected_return(&self.weights, &self.returns_matrix)
    }

    fn calculate_variance(&self) -> f64 {
        portfolio::portfolio_variance(&self.weights, &self.returns_matrix)
    }

    fn calculate_volatility(&self) -> f64 {
        self.calculate_variance().sqrt()
    }

    fn calculate_sharpe_ratio(&self) -> f64 {
        let expected_return = self.calculate_expected_return();
        let volatility = self.calculate_volatility();
        (expected_return - self.risk_free_rate) / volatility
    }

    fn calculate_performance(&self) -> Result<Value, crate::foreign::ForeignError> {
        let expected_return = self.calculate_expected_return();
        let volatility = self.calculate_volatility();
        let sharpe_ratio = self.calculate_sharpe_ratio();

        Ok(Value::List(vec![
            Value::Real(expected_return),
            Value::Real(volatility),
            Value::Real(sharpe_ratio),
        ]))
    }

    fn optimize_portfolio(&self, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        if args.len() != 1 {
            return Err(crate::foreign::ForeignError::ArgumentError {
                expected: 1,
                actual: args.len(),
            });
        }

        let target_return = match &args[0] {
            Value::Real(ret) => *ret,
            Value::Integer(ret) => *ret as f64,
            _ => return Err(crate::foreign::ForeignError::TypeError {
                expected: "Real or Integer".to_string(),
                actual: format!("{:?}", args[0]),
            }),
        };

        let optimal_weights = portfolio::markowitz_mean_variance_optimization(
            &self.returns_matrix,
            target_return,
        ).map_err(|e| crate::foreign::ForeignError::RuntimeError { message: e.to_string() })?;

        Ok(Value::List(optimal_weights.into_iter().map(Value::Real).collect()))
    }

    fn rebalance_portfolio(&self, args: &[Value]) -> Result<Value, crate::foreign::ForeignError> {
        if args.len() != 1 {
            return Err(crate::foreign::ForeignError::ArgumentError {
                expected: 1,
                actual: args.len(),
            });
        }

        let new_weights = match &args[0] {
            Value::List(weights) => {
                let mut result = Vec::new();
                for weight in weights {
                    match weight {
                        Value::Real(w) => result.push(*w),
                        Value::Integer(w) => result.push(*w as f64),
                        _ => return Err(crate::foreign::ForeignError::TypeError {
                            expected: "List of Real or Integer".to_string(),
                            actual: format!("List containing {:?}", weight),
                        }),
                    }
                }
                result
            }
            _ => return Err(crate::foreign::ForeignError::TypeError {
                expected: "List".to_string(),
                actual: format!("{:?}", args[0]),
            }),
        };

        // Validate weights sum to 1.0
        let weight_sum: f64 = new_weights.iter().sum();
        if (weight_sum - 1.0).abs() > 1e-6 {
            return Err(crate::foreign::ForeignError::InvalidArgument(format!(
                "Portfolio weights must sum to 1.0, got {}",
                weight_sum
            )));
        }

        // Create new portfolio with updated weights
        let mut new_portfolio = self.clone();
        new_portfolio.weights = new_weights;

        Ok(Value::LyObj(crate::foreign::LyObj::new(Box::new(new_portfolio))))
    }
}

/// Utility functions for argument parsing
pub fn extract_real(value: &Value, name: &str) -> Result<f64, VmError> {
    match value {
        Value::Real(r) => Ok(*r),
        Value::Integer(i) => Ok(*i as f64),
        _ => Err(VmError::TypeError {
            expected: format!("{} as Real or Integer", name),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_symbol(value: &Value, name: &str) -> Result<String, VmError> {
    match value {
        Value::Symbol(s) => Ok(s.clone()),
        _ => Err(VmError::TypeError {
            expected: format!("{} as Symbol", name),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_integer(value: &Value, name: &str) -> Result<i64, VmError> {
    match value {
        Value::Integer(i) => Ok(*i),
        Value::Real(r) => Ok(*r as i64),
        _ => Err(VmError::TypeError {
            expected: format!("{} as Integer or Real", name),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_list_of_reals(value: &Value, name: &str) -> Result<Vec<f64>, VmError> {
    match value {
        Value::List(items) => {
            let mut result = Vec::new();
            for item in items {
                result.push(extract_real(item, name)?);
            }
            Ok(result)
        }
        _ => Err(VmError::TypeError {
            expected: format!("{} as List", name),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_matrix(value: &Value, name: &str) -> Result<Vec<Vec<f64>>, VmError> {
    match value {
        Value::List(rows) => {
            let mut result = Vec::new();
            for row in rows {
                result.push(extract_list_of_reals(row, name)?);
            }
            Ok(result)
        }
        _ => Err(VmError::TypeError {
            expected: format!("{} as List of Lists", name),
            actual: format!("{:?}", value),
        }),
    }
}

/// Parse option type from symbol
pub fn parse_option_type(symbol: &str) -> Result<OptionType, VmError> {
    match symbol.to_lowercase().as_str() {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        _ => Err(VmError::Runtime(format!(
            "Invalid option type: {}. Must be 'Call' or 'Put'",
            symbol
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_instrument_creation() {
        let option = OptionInstrument::new(100.0, 100.0, 1.0, 0.05, 0.2, OptionType::Call);
        
        assert_eq!(option.spot_price, 100.0);
        assert_eq!(option.strike_price, 100.0);
        assert_eq!(option.time_to_expiry, 1.0);
        assert_eq!(option.risk_free_rate, 0.05);
        assert_eq!(option.volatility, 0.2);
        assert_eq!(option.option_type, OptionType::Call);
    }

    #[test]
    fn test_bond_instrument_creation() {
        let bond = BondInstrument::new(100.0, 0.05, 0.06, 5.0);
        
        assert_eq!(bond.face_value, 100.0);
        assert_eq!(bond.coupon_rate, 0.05);
        assert_eq!(bond.yield_to_maturity, 0.06);
        assert_eq!(bond.time_to_maturity, 5.0);
        assert_eq!(bond.payment_frequency, 2);
    }

    #[test]
    fn test_portfolio_instrument_creation() {
        let weights = vec![0.6, 0.4];
        let returns = vec![vec![0.08, 0.12], vec![0.06, 0.10]];
        let portfolio = PortfolioInstrument::new(weights, returns, 0.03);
        
        assert_eq!(portfolio.weights, vec![0.6, 0.4]);
        assert_eq!(portfolio.risk_free_rate, 0.03);
        assert_eq!(portfolio.assets.len(), 2);
    }

    #[test]
    fn test_option_type_parsing() {
        assert_eq!(parse_option_type("Call").unwrap(), OptionType::Call);
        assert_eq!(parse_option_type("call").unwrap(), OptionType::Call);
        assert_eq!(parse_option_type("Put").unwrap(), OptionType::Put);
        assert_eq!(parse_option_type("put").unwrap(), OptionType::Put);
        
        assert!(parse_option_type("Invalid").is_err());
    }

    #[test]
    fn test_value_extraction() {
        assert_eq!(extract_real(&Value::Real(3.14), "pi").unwrap(), 3.14);
        assert_eq!(extract_real(&Value::Integer(42), "answer").unwrap(), 42.0);
        assert!(extract_real(&Value::Symbol("not_a_number".to_string()), "test").is_err());

        assert_eq!(extract_symbol(&Value::Symbol("Call".to_string()), "type").unwrap(), "Call");
        assert!(extract_symbol(&Value::Real(3.14), "test").is_err());

        let list = Value::List(vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)]);
        let extracted = extract_list_of_reals(&list, "test").unwrap();
        assert_eq!(extracted, vec![1.0, 2.0, 3.0]);
    }
}