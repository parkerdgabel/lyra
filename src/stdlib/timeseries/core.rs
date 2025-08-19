//! Core Time Series Data Structures and Operations
//!
//! Provides the fundamental TimeSeries foreign object and basic time series
//! operations like creation, indexing, and basic transformations.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;

/// Frequency specification for time series data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Frequency {
    /// Yearly data
    Annual,
    /// Quarterly data
    Quarterly,
    /// Monthly data
    Monthly,
    /// Weekly data
    Weekly,
    /// Daily data
    Daily,
    /// Hourly data
    Hourly,
    /// Data at specified intervals in seconds
    Seconds(i64),
    /// Irregular/unspecified frequency
    Irregular,
}

impl Frequency {
    /// Get the number of periods per year for this frequency
    pub fn periods_per_year(&self) -> Option<f64> {
        match self {
            Frequency::Annual => Some(1.0),
            Frequency::Quarterly => Some(4.0),
            Frequency::Monthly => Some(12.0),
            Frequency::Weekly => Some(52.0),
            Frequency::Daily => Some(365.25),
            Frequency::Hourly => Some(8766.0), // 365.25 * 24
            Frequency::Seconds(s) => Some(31557600.0 / (*s as f64)), // seconds in year / interval
            Frequency::Irregular => None,
        }
    }
}

/// Core TimeSeries type implementing Foreign trait
#[derive(Debug, Clone)]
pub struct TimeSeries {
    /// Time series values
    pub values: Vec<f64>,
    /// Optional time index (Unix timestamps or relative indices)
    pub index: Option<Vec<f64>>,
    /// Frequency of the time series
    pub frequency: Frequency,
    /// Series name/identifier
    pub name: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Start time for indexed series
    pub start_time: Option<f64>,
}

impl TimeSeries {
    /// Create a new time series from values
    pub fn new(values: Vec<f64>, frequency: Frequency) -> Self {
        TimeSeries {
            values,
            index: None,
            frequency,
            name: "TimeSeries".to_string(),
            metadata: HashMap::new(),
            start_time: None,
        }
    }

    /// Create a time series with custom index
    pub fn with_index(values: Vec<f64>, index: Vec<f64>, frequency: Frequency) -> Self {
        if values.len() != index.len() {
            panic!("Values and index must have the same length");
        }
        TimeSeries {
            values,
            index: Some(index),
            frequency,
            name: "TimeSeries".to_string(),
            metadata: HashMap::new(),
            start_time: None,
        }
    }

    /// Create a time series with a start time and regular frequency
    pub fn with_start_time(values: Vec<f64>, frequency: Frequency, start_time: f64) -> Self {
        TimeSeries {
            values,
            index: None,
            frequency,
            name: "TimeSeries".to_string(),
            metadata: HashMap::new(),
            start_time: Some(start_time),
        }
    }

    /// Get the length of the time series
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if the time series is empty
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Get value at specific index
    pub fn get(&self, index: usize) -> Option<f64> {
        self.values.get(index).copied()
    }

    /// Get a slice of values
    pub fn slice(&self, start: usize, end: usize) -> Option<TimeSeries> {
        if start >= self.values.len() || end > self.values.len() || start >= end {
            return None;
        }

        let sliced_values = self.values[start..end].to_vec();
        let sliced_index = if let Some(ref idx) = self.index {
            Some(idx[start..end].to_vec())
        } else {
            None
        };

        Some(TimeSeries {
            values: sliced_values,
            index: sliced_index,
            frequency: self.frequency,
            name: format!("{}_slice", self.name),
            metadata: self.metadata.clone(),
            start_time: self.start_time,
        })
    }

    /// Set metadata
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Get metadata
    pub fn get_metadata(&self, key: &str) -> Option<&String> {
        self.metadata.get(key)
    }

    /// Calculate basic statistics
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            return 0.0;
        }
        self.values.iter().sum::<f64>() / self.values.len() as f64
    }

    /// Calculate variance
    pub fn variance(&self) -> f64 {
        if self.values.len() <= 1 {
            return 0.0;
        }
        let mean = self.mean();
        let sum_sq_diff: f64 = self.values.iter()
            .map(|x| (x - mean).powi(2))
            .sum();
        sum_sq_diff / (self.values.len() - 1) as f64
    }

    /// Calculate standard deviation
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Calculate differences (first difference for stationarity)
    pub fn diff(&self, lag: usize) -> TimeSeries {
        if lag >= self.values.len() {
            return TimeSeries::new(Vec::new(), self.frequency);
        }

        let diff_values: Vec<f64> = self.values.iter()
            .skip(lag)
            .enumerate()
            .map(|(i, &val)| val - self.values[i])
            .collect();

        let diff_index = if let Some(ref idx) = self.index {
            Some(idx[lag..].to_vec())
        } else {
            None
        };

        TimeSeries {
            values: diff_values,
            index: diff_index,
            frequency: self.frequency,
            name: format!("{}_diff_{}", self.name, lag),
            metadata: self.metadata.clone(),
            start_time: self.start_time,
        }
    }

    /// Apply lag operation (shift values by n periods)
    pub fn lag(&self, n: usize) -> TimeSeries {
        if n >= self.values.len() {
            return TimeSeries::new(Vec::new(), self.frequency);
        }

        let lagged_values = self.values[..self.values.len() - n].to_vec();
        let lagged_index = if let Some(ref idx) = self.index {
            Some(idx[n..].to_vec())
        } else {
            None
        };

        TimeSeries {
            values: lagged_values,
            index: lagged_index,
            frequency: self.frequency,
            name: format!("{}_lag_{}", self.name, n),
            metadata: self.metadata.clone(),
            start_time: self.start_time,
        }
    }
}

impl Foreign for TimeSeries {
    fn type_name(&self) -> &'static str {
        "TimeSeries"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Length" => Ok(Value::Integer(self.len() as i64)),
            "Values" => {
                let vals: Vec<Value> = self.values.iter()
                    .map(|&v| Value::Real(v))
                    .collect();
                Ok(Value::List(vals))
            }
            "Frequency" => Ok(Value::String(format!("{:?}", self.frequency))),
            "Name" => Ok(Value::String(self.name.clone())),
            "Mean" => Ok(Value::Real(self.mean())),
            "Variance" => Ok(Value::Real(self.variance())),
            "StandardDeviation" => Ok(Value::Real(self.std_dev())),
            "Get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let index = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                match self.get(index) {
                    Some(value) => Ok(Value::Real(value)),
                    None => Err(ForeignError::IndexOutOfBounds {
                        index: index.to_string(),
                        bounds: format!("0..{}", self.len()),
                    }),
                }
            }
            "Slice" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }

                let start = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let end = match &args[1] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };

                match self.slice(start, end) {
                    Some(sliced) => Ok(Value::LyObj(LyObj::new(Box::new(sliced)))),
                    None => Err(ForeignError::IndexOutOfBounds {
                        index: format!("{}..{}", start, end),
                        bounds: format!("0..{}", self.len()),
                    }),
                }
            }
            "Diff" => {
                let lag = if args.len() == 1 {
                    match &args[0] {
                        Value::Integer(i) => *i as usize,
                        _ => 1,
                    }
                } else {
                    1
                };

                let diff_series = self.diff(lag);
                Ok(Value::LyObj(LyObj::new(Box::new(diff_series))))
            }
            "Lag" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }

                let n = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };

                let lagged_series = self.lag(n);
                Ok(Value::LyObj(LyObj::new(Box::new(lagged_series))))
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
// CORE TIME SERIES FUNCTIONS
// ===============================

/// Create a TimeSeries from a list of values
/// Syntax: TimeSeries[values, frequency]
pub fn timeseries(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (values, [frequency])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let values = match &args[0] {
        Value::List(list) => {
            let mut vals = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => vals.push(*r),
                    Value::Integer(i) => vals.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric list".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            vals
        }
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let frequency = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => match s.as_str() {
                "Annual" | "Yearly" => Frequency::Annual,
                "Quarterly" => Frequency::Quarterly,
                "Monthly" => Frequency::Monthly,
                "Weekly" => Frequency::Weekly,
                "Daily" => Frequency::Daily,
                "Hourly" => Frequency::Hourly,
                "Irregular" => Frequency::Irregular,
                _ => Frequency::Irregular,
            },
            Value::Integer(i) => Frequency::Seconds(*i),
            _ => Frequency::Irregular,
        }
    } else {
        Frequency::Irregular
    };

    let ts = TimeSeries::new(values, frequency);
    Ok(Value::LyObj(LyObj::new(Box::new(ts))))
}

/// Create a TimeSeries with a time index
/// Syntax: TimeSeriesWithIndex[values, index, frequency]
pub fn timeseries_with_index(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (values, index, [frequency])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let values = match &args[0] {
        Value::List(list) => {
            let mut vals = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => vals.push(*r),
                    Value::Integer(i) => vals.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric list".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            vals
        }
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };

    let index = match &args[1] {
        Value::List(list) => {
            let mut idx = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => idx.push(*r),
                    Value::Integer(i) => idx.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "numeric list".to_string(),
                        actual: format!("list containing {:?}", item),
                    }),
                }
            }
            idx
        }
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };

    let frequency = if args.len() == 3 {
        match &args[2] {
            Value::String(s) => match s.as_str() {
                "Annual" | "Yearly" => Frequency::Annual,
                "Quarterly" => Frequency::Quarterly,
                "Monthly" => Frequency::Monthly,
                "Weekly" => Frequency::Weekly,
                "Daily" => Frequency::Daily,
                "Hourly" => Frequency::Hourly,
                "Irregular" => Frequency::Irregular,
                _ => Frequency::Irregular,
            },
            Value::Integer(i) => Frequency::Seconds(*i),
            _ => Frequency::Irregular,
        }
    } else {
        Frequency::Irregular
    };

    if values.len() != index.len() {
        return Err(VmError::TypeError {
            expected: "values and index of same length".to_string(),
            actual: format!("values length: {}, index length: {}", values.len(), index.len()),
        });
    }

    let ts = TimeSeries::with_index(values, index, frequency);
    Ok(Value::LyObj(LyObj::new(Box::new(ts))))
}

/// Extract TimeSeries from Value
pub fn extract_timeseries(value: &Value) -> VmResult<&TimeSeries> {
    match value {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TimeSeries>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TimeSeries".to_string(),
                    actual: obj.type_name().to_string(),
                })
        }
        _ => Err(VmError::TypeError {
            expected: "TimeSeries".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeseries_creation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(values.clone(), Frequency::Daily);
        
        assert_eq!(ts.len(), 5);
        assert_eq!(ts.values, values);
        assert_eq!(ts.frequency, Frequency::Daily);
        assert!(!ts.is_empty());
    }

    #[test]
    fn test_timeseries_with_index() {
        let values = vec![10.0, 20.0, 30.0];
        let index = vec![1.0, 2.0, 3.0];
        let ts = TimeSeries::with_index(values.clone(), index.clone(), Frequency::Monthly);
        
        assert_eq!(ts.len(), 3);
        assert_eq!(ts.values, values);
        assert_eq!(ts.index, Some(index));
        assert_eq!(ts.frequency, Frequency::Monthly);
    }

    #[test]
    fn test_timeseries_basic_operations() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(values, Frequency::Daily);
        
        // Test get
        assert_eq!(ts.get(0), Some(1.0));
        assert_eq!(ts.get(2), Some(3.0));
        assert_eq!(ts.get(10), None);
        
        // Test statistics
        assert_eq!(ts.mean(), 3.0);
        assert_eq!(ts.variance(), 2.5);
        assert!((ts.std_dev() - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_timeseries_slice() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(values, Frequency::Daily);
        
        let sliced = ts.slice(1, 4).unwrap();
        assert_eq!(sliced.values, vec![2.0, 3.0, 4.0]);
        assert_eq!(sliced.len(), 3);
        
        // Test out of bounds
        assert!(ts.slice(0, 10).is_none());
        assert!(ts.slice(3, 2).is_none());
    }

    #[test]
    fn test_timeseries_diff() {
        let values = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        let ts = TimeSeries::new(values, Frequency::Daily);
        
        let diff = ts.diff(1);
        assert_eq!(diff.values, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(diff.len(), 4);
    }

    #[test]
    fn test_timeseries_lag() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(values, Frequency::Daily);
        
        let lagged = ts.lag(2);
        assert_eq!(lagged.values, vec![1.0, 2.0, 3.0]);
        assert_eq!(lagged.len(), 3);
    }

    #[test]
    fn test_frequency_periods_per_year() {
        assert_eq!(Frequency::Annual.periods_per_year(), Some(1.0));
        assert_eq!(Frequency::Quarterly.periods_per_year(), Some(4.0));
        assert_eq!(Frequency::Monthly.periods_per_year(), Some(12.0));
        assert_eq!(Frequency::Weekly.periods_per_year(), Some(52.0));
        assert_eq!(Frequency::Daily.periods_per_year(), Some(365.25));
        assert_eq!(Frequency::Hourly.periods_per_year(), Some(8766.0));
        assert_eq!(Frequency::Irregular.periods_per_year(), None);
    }

    #[test]
    fn test_timeseries_function() {
        let values = vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)];
        let list_value = Value::List(values);
        let freq_value = Value::String("Daily".to_string());
        
        let result = timeseries(&[list_value, freq_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let ts = obj.downcast_ref::<TimeSeries>().unwrap();
                assert_eq!(ts.values, vec![1.0, 2.0, 3.0]);
                assert_eq!(ts.frequency, Frequency::Daily);
            }
            _ => panic!("Expected TimeSeries object"),
        }
    }

    #[test]
    fn test_timeseries_with_index_function() {
        let values = vec![Value::Real(10.0), Value::Real(20.0)];
        let index = vec![Value::Real(1.0), Value::Real(2.0)];
        let list_values = Value::List(values);
        let list_index = Value::List(index);
        let freq_value = Value::String("Monthly".to_string());
        
        let result = timeseries_with_index(&[list_values, list_index, freq_value]).unwrap();
        match result {
            Value::LyObj(obj) => {
                let ts = obj.downcast_ref::<TimeSeries>().unwrap();
                assert_eq!(ts.values, vec![10.0, 20.0]);
                assert_eq!(ts.index, Some(vec![1.0, 2.0]));
                assert_eq!(ts.frequency, Frequency::Monthly);
            }
            _ => panic!("Expected TimeSeries object"),
        }
    }

    #[test]
    fn test_timeseries_foreign_methods() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ts = TimeSeries::new(values, Frequency::Daily);
        
        // Test Length method
        let length = ts.call_method("Length", &[]).unwrap();
        assert_eq!(length, Value::Integer(5));
        
        // Test Mean method
        let mean = ts.call_method("Mean", &[]).unwrap();
        assert_eq!(mean, Value::Real(3.0));
        
        // Test Get method
        let get_result = ts.call_method("Get", &[Value::Integer(2)]).unwrap();
        assert_eq!(get_result, Value::Real(3.0));
        
        // Test Frequency method
        let freq = ts.call_method("Frequency", &[]).unwrap();
        assert_eq!(freq, Value::String("Daily".to_string()));
    }
}