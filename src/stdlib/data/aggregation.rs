//! Unified Aggregation Functions for Lyra Standard Library
//!
//! This module provides a centralized, consistent interface for aggregation operations
//! across all data processing modules (table, data_processing, streaming).

use crate::vm::Value;
use crate::foreign::{Foreign, ForeignError};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;

/// Core aggregation function types that can be applied to data
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    /// Count the number of non-null values
    Count,
    /// Sum all numeric values
    Sum,
    /// Calculate arithmetic mean
    Mean,
    /// Find minimum value
    Min,
    /// Find maximum value 
    Max,
    /// Return first value
    First,
    /// Return last value
    Last,
    /// Standard deviation
    StdDev,
    /// Variance
    Variance,
    /// Median value
    Median,
    /// Custom aggregation function by name
    Custom(String),
}

/// Aggregation specification that includes the function and optional column target
#[derive(Debug, Clone)]
pub struct AggregationSpec {
    /// The aggregation function to apply
    pub function: AggregationFunction,
    /// The column name to aggregate (None means aggregate all/count)
    pub column: Option<String>,
    /// Alias for the result column
    pub alias: Option<String>,
}

/// Container for multiple aggregation specifications
#[derive(Debug, Clone)]
pub struct AggregationSet {
    specs: Vec<AggregationSpec>,
}

impl AggregationFunction {
    /// Apply the aggregation function to a list of values
    pub fn apply(&self, values: &[Value]) -> Value {
        if values.is_empty() {
            return match self {
                AggregationFunction::Count => Value::Integer(0),
                _ => Value::Missing,
            };
        }

        match self {
            AggregationFunction::Count => Value::Integer(values.len() as i64),
            
            AggregationFunction::Sum => {
                let sum: f64 = values.iter()
                    .filter_map(|v| self.extract_numeric(v))
                    .sum();
                if sum.fract() == 0.0 && sum.abs() <= i64::MAX as f64 {
                    Value::Integer(sum as i64)
                } else {
                    Value::Real(sum)
                }
            }
            
            AggregationFunction::Mean => {
                let numeric_values: Vec<f64> = values.iter()
                    .filter_map(|v| self.extract_numeric(v))
                    .collect();
                
                if numeric_values.is_empty() {
                    Value::Missing
                } else {
                    let mean = numeric_values.iter().sum::<f64>() / numeric_values.len() as f64;
                    Value::Real(mean)
                }
            }
            
            AggregationFunction::Min => {
                values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(_) | Value::Real(_) => Some(v),
                        _ => None,
                    })
                    .min_by(|a, b| self.compare_values(a, b))
                    .cloned()
                    .unwrap_or(Value::Missing)
            }
            
            AggregationFunction::Max => {
                values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(_) | Value::Real(_) => Some(v),
                        _ => None,
                    })
                    .max_by(|a, b| self.compare_values(a, b))
                    .cloned()
                    .unwrap_or(Value::Missing)
            }
            
            AggregationFunction::First => values.first().cloned().unwrap_or(Value::Missing),
            
            AggregationFunction::Last => values.last().cloned().unwrap_or(Value::Missing),
            
            AggregationFunction::StdDev => {
                let numeric_values: Vec<f64> = values.iter()
                    .filter_map(|v| self.extract_numeric(v))
                    .collect();
                
                if numeric_values.len() < 2 {
                    Value::Missing
                } else {
                    let mean = numeric_values.iter().sum::<f64>() / numeric_values.len() as f64;
                    let variance = numeric_values.iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>() / (numeric_values.len() - 1) as f64;
                    Value::Real(variance.sqrt())
                }
            }
            
            AggregationFunction::Variance => {
                let numeric_values: Vec<f64> = values.iter()
                    .filter_map(|v| self.extract_numeric(v))
                    .collect();
                
                if numeric_values.len() < 2 {
                    Value::Missing
                } else {
                    let mean = numeric_values.iter().sum::<f64>() / numeric_values.len() as f64;
                    let variance = numeric_values.iter()
                        .map(|v| (v - mean).powi(2))
                        .sum::<f64>() / (numeric_values.len() - 1) as f64;
                    Value::Real(variance)
                }
            }
            
            AggregationFunction::Median => {
                let mut numeric_values: Vec<f64> = values.iter()
                    .filter_map(|v| self.extract_numeric(v))
                    .collect();
                
                if numeric_values.is_empty() {
                    Value::Missing
                } else {
                    numeric_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    let len = numeric_values.len();
                    if len % 2 == 0 {
                        let median = (numeric_values[len / 2 - 1] + numeric_values[len / 2]) / 2.0;
                        Value::Real(median)
                    } else {
                        Value::Real(numeric_values[len / 2])
                    }
                }
            }
            
            AggregationFunction::Custom(_function_name) => {
                // Custom aggregations would be implemented by calling registered functions
                // For now, return the first value as a placeholder
                values.first().cloned().unwrap_or(Value::Missing)
            }
        }
    }

    /// Extract numeric value from a Value for mathematical operations
    fn extract_numeric(&self, value: &Value) -> Option<f64> {
        match value {
            Value::Integer(i) => Some(*i as f64),
            Value::Real(f) => Some(*f),
            _ => None,
        }
    }

    /// Compare two values for min/max operations
    fn compare_values(&self, a: &Value, b: &Value) -> std::cmp::Ordering {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
            (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
            _ => std::cmp::Ordering::Equal,
        }
    }

    /// Get a human-readable name for the aggregation function
    pub fn name(&self) -> &str {
        match self {
            AggregationFunction::Count => "count",
            AggregationFunction::Sum => "sum",
            AggregationFunction::Mean => "mean",
            AggregationFunction::Min => "min",
            AggregationFunction::Max => "max",
            AggregationFunction::First => "first",
            AggregationFunction::Last => "last",
            AggregationFunction::StdDev => "stddev",
            AggregationFunction::Variance => "variance",
            AggregationFunction::Median => "median",
            AggregationFunction::Custom(name) => name,
        }
    }

    /// Parse aggregation function from string
    pub fn from_string(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "count" => Some(AggregationFunction::Count),
            "sum" => Some(AggregationFunction::Sum),
            "mean" | "avg" | "average" => Some(AggregationFunction::Mean),
            "min" => Some(AggregationFunction::Min),
            "max" => Some(AggregationFunction::Max),
            "first" => Some(AggregationFunction::First),
            "last" => Some(AggregationFunction::Last),
            "stddev" | "std" => Some(AggregationFunction::StdDev),
            "variance" | "var" => Some(AggregationFunction::Variance),
            "median" => Some(AggregationFunction::Median),
            _ => Some(AggregationFunction::Custom(s.to_string())),
        }
    }
}

impl fmt::Display for AggregationFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl AggregationSpec {
    /// Create a new aggregation specification
    pub fn new(function: AggregationFunction, column: Option<String>, alias: Option<String>) -> Self {
        Self {
            function,
            column,
            alias,
        }
    }

    /// Create a simple aggregation spec for a specific column
    pub fn for_column(function: AggregationFunction, column: String) -> Self {
        Self::new(function, Some(column), None)
    }

    /// Create a simple aggregation spec without a specific column (e.g., Count)
    pub fn simple(function: AggregationFunction) -> Self {
        Self::new(function, None, None)
    }

    /// Get the result column name (alias if provided, otherwise generated)
    pub fn result_column_name(&self) -> String {
        if let Some(alias) = &self.alias {
            alias.clone()
        } else if let Some(column) = &self.column {
            format!("{}_{}", self.function.name(), column)
        } else {
            self.function.name().to_string()
        }
    }

    /// Apply this aggregation to extracted column values
    pub fn apply(&self, values: &[Value]) -> Value {
        self.function.apply(values)
    }
}

impl AggregationSet {
    /// Create a new empty aggregation set
    pub fn new() -> Self {
        Self {
            specs: Vec::new(),
        }
    }

    /// Add an aggregation specification
    pub fn add(&mut self, spec: AggregationSpec) {
        self.specs.push(spec);
    }

    /// Get all aggregation specifications
    pub fn specs(&self) -> &[AggregationSpec] {
        &self.specs
    }

    /// Create from a vector of specs
    pub fn from_specs(specs: Vec<AggregationSpec>) -> Self {
        Self { specs }
    }

    /// Create from a HashMap of name -> function mappings (for backward compatibility)
    pub fn from_hashmap(functions: HashMap<String, AggregationFunction>) -> Self {
        let specs = functions.into_iter()
            .map(|(alias, function)| AggregationSpec::new(function, None, Some(alias)))
            .collect();
        Self { specs }
    }
}

impl Default for AggregationSet {
    fn default() -> Self {
        Self::new()
    }
}

// Foreign object implementations for VM integration

impl Foreign for AggregationFunction {
    fn type_name(&self) -> &'static str {
        "AggregationFunction"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let values = match &args[0] {
                    Value::List(v) => v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                Ok(self.apply(values))
            }
            "name" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.name().to_string()))
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

impl Foreign for AggregationSpec {
    fn type_name(&self) -> &'static str {
        "AggregationSpec"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let values = match &args[0] {
                    Value::List(v) => v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                Ok(self.apply(values))
            }
            "resultColumnName" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.result_column_name()))
            }
            "function" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.function.name().to_string()))
            }
            "column" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(self.column.as_ref()
                    .map(|c| Value::String(c.clone()))
                    .unwrap_or(Value::Missing))
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

impl Foreign for AggregationSet {
    fn type_name(&self) -> &'static str {
        "AggregationSet"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "add" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                // This would need to modify the set, but Foreign objects are typically immutable
                // In practice, this would create a new AggregationSet
                Ok(Value::Boolean(true))
            }
            "count" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.specs.len() as i64))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregation_function_apply() {
        let values = vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::Real(4.5),
        ];

        // Test Count
        assert_eq!(AggregationFunction::Count.apply(&values), Value::Integer(4));

        // Test Sum  
        assert_eq!(AggregationFunction::Sum.apply(&values), Value::Real(10.5));

        // Test Mean
        assert_eq!(AggregationFunction::Mean.apply(&values), Value::Real(2.625));

        // Test Min
        assert_eq!(AggregationFunction::Min.apply(&values), Value::Integer(1));

        // Test Max
        assert_eq!(AggregationFunction::Max.apply(&values), Value::Real(4.5));

        // Test First
        assert_eq!(AggregationFunction::First.apply(&values), Value::Integer(1));

        // Test Last
        assert_eq!(AggregationFunction::Last.apply(&values), Value::Real(4.5));
    }

    #[test]
    fn test_aggregation_function_empty_values() {
        let values = vec![];

        assert_eq!(AggregationFunction::Count.apply(&values), Value::Integer(0));
        assert_eq!(AggregationFunction::Sum.apply(&values), Value::Missing);
        assert_eq!(AggregationFunction::Mean.apply(&values), Value::Missing);
    }

    #[test]
    fn test_aggregation_spec() {
        let spec = AggregationSpec::for_column(AggregationFunction::Sum, "price".to_string());
        assert_eq!(spec.result_column_name(), "sum_price");

        let spec_with_alias = AggregationSpec::new(
            AggregationFunction::Mean,
            Some("price".to_string()),
            Some("avg_price".to_string())
        );
        assert_eq!(spec_with_alias.result_column_name(), "avg_price");
    }

    #[test]
    fn test_aggregation_function_from_string() {
        assert_eq!(AggregationFunction::from_string("count"), Some(AggregationFunction::Count));
        assert_eq!(AggregationFunction::from_string("sum"), Some(AggregationFunction::Sum));
        assert_eq!(AggregationFunction::from_string("avg"), Some(AggregationFunction::Mean));
        assert_eq!(AggregationFunction::from_string("MEAN"), Some(AggregationFunction::Mean));
    }

    #[test]
    fn test_median_calculation() {
        // Odd number of values
        let values_odd = vec![
            Value::Integer(1),
            Value::Integer(3),
            Value::Integer(2),
            Value::Integer(5),
            Value::Integer(4),
        ];
        assert_eq!(AggregationFunction::Median.apply(&values_odd), Value::Real(3.0));

        // Even number of values
        let values_even = vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
            Value::Integer(4),
        ];
        assert_eq!(AggregationFunction::Median.apply(&values_even), Value::Real(2.5));
    }

    #[test]
    fn test_standard_deviation() {
        let values = vec![
            Value::Integer(2),
            Value::Integer(4),
            Value::Integer(4),
            Value::Integer(4),
            Value::Integer(5),
            Value::Integer(5),
            Value::Integer(7),
            Value::Integer(9),
        ];
        
        let result = AggregationFunction::StdDev.apply(&values);
        if let Value::Real(stddev) = result {
            // Should be approximately 2.138
            assert!((stddev - 2.138).abs() < 0.01);
        } else {
            panic!("Expected Real value for standard deviation");
        }
    }
}