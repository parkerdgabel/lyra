//! Statistical functions for the Lyra standard library
//! 
//! This module implements comprehensive statistical functions following
//! Wolfram Language syntax and semantics.
//!
//! Functions provided:
//! - Descriptive Statistics: Mean, Variance, StandardDeviation, Median, Mode
//! - Probability Distributions: RandomReal, RandomInteger, NormalDistribution
//! - Statistical Tests: Correlation, Covariance
//! - Data Generation: Range, RandomSample

use crate::vm::{Value, VmError, VmResult};
use std::collections::HashMap;
use rand::Rng;

/// Mean[list] - Arithmetic mean of a list of numbers
/// 
/// Examples:
/// - `Mean[{1, 2, 3, 4, 5}]` → `3.0`
/// - `Mean[{1.0, 2.5, 3.5}]` → `2.33...`
pub fn mean(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let numbers = extract_numeric_list(&args[0])?;
    
    if numbers.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    let sum: f64 = numbers.iter().sum();
    let mean = sum / numbers.len() as f64;
    
    Ok(Value::Real(mean))
}

/// Variance[list] - Sample variance of a list of numbers
/// 
/// Uses the unbiased sample variance formula: Var = Σ(x - μ)² / (n - 1)
/// 
/// Examples:
/// - `Variance[{1, 2, 3, 4, 5}]` → `2.5`
/// - `Variance[{2, 4, 6, 8}]` → `6.66...`
pub fn variance(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let numbers = extract_numeric_list(&args[0])?;
    
    if numbers.len() < 2 {
        return Err(VmError::TypeError {
            expected: "list with at least 2 elements".to_string(),
            actual: format!("list with {} elements", numbers.len()),
        });
    }
    
    // Calculate mean
    let sum: f64 = numbers.iter().sum();
    let mean = sum / numbers.len() as f64;
    
    // Calculate variance
    let variance_sum: f64 = numbers.iter()
        .map(|x| (x - mean).powi(2))
        .sum();
    
    let variance = variance_sum / (numbers.len() - 1) as f64; // Sample variance (n-1)
    
    Ok(Value::Real(variance))
}

/// StandardDeviation[list] - Sample standard deviation of a list of numbers
/// 
/// StandardDeviation = √Variance
/// 
/// Examples:
/// - `StandardDeviation[{1, 2, 3, 4, 5}]` → `1.58...`
/// - `StandardDeviation[{2, 4, 6, 8}]` → `2.58...`
pub fn standard_deviation(args: &[Value]) -> VmResult<Value> {
    let variance_result = variance(args)?;
    
    match variance_result {
        Value::Real(var) => Ok(Value::Real(var.sqrt())),
        _ => Err(VmError::TypeError {
            expected: "numeric variance".to_string(),
            actual: format!("{:?}", variance_result),
        })
    }
}

/// Median[list] - Median value of a list of numbers
/// 
/// For odd length: middle value
/// For even length: average of two middle values
/// 
/// Examples:
/// - `Median[{1, 2, 3, 4, 5}]` → `3.0`
/// - `Median[{1, 2, 3, 4}]` → `2.5`
/// - `Median[{5, 1, 3, 9, 2}]` → `3.0`
pub fn median(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut numbers = extract_numeric_list(&args[0])?;
    
    if numbers.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    // Sort the numbers
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = numbers.len();
    
    let median = if n % 2 == 1 {
        // Odd length: return middle element
        numbers[n / 2]
    } else {
        // Even length: return average of two middle elements
        (numbers[n / 2 - 1] + numbers[n / 2]) / 2.0
    };
    
    Ok(Value::Real(median))
}

/// Mode[list] - Most frequently occurring value(s) in a list
/// 
/// Returns a list of the most common value(s). If there's a tie, returns all tied values.
/// 
/// Examples:
/// - `Mode[{1, 2, 2, 3, 4}]` → `{2}`
/// - `Mode[{1, 1, 2, 2, 3}]` → `{1, 2}`
/// - `Mode[{1, 2, 3}]` → `{1, 2, 3}` (all equally common)
pub fn mode(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let list = match &args[0] {
        Value::List(items) => items,
        _ => return Err(VmError::TypeError {
            expected: "list".to_string(),
            actual: format!("{:?}", args[0]),
        })
    };
    
    if list.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    // Count occurrences of each value
    let mut counts = HashMap::new();
    for item in list {
        let count = counts.entry(format!("{:?}", item)).or_insert(0);
        *count += 1;
    }
    
    // Find maximum count
    let max_count = *counts.values().max().unwrap();
    
    // Collect all values with maximum count
    let mut modes = Vec::new();
    for item in list {
        let key = format!("{:?}", item);
        if counts[&key] == max_count && !modes.contains(item) {
            modes.push(item.clone());
        }
    }
    
    Ok(Value::List(modes))
}

/// Quantile[list, q] - q-th quantile of a list of numbers
/// 
/// q should be between 0 and 1 (inclusive)
/// 
/// Examples:
/// - `Quantile[{1, 2, 3, 4, 5}, 0.5]` → `3.0` (median)
/// - `Quantile[{1, 2, 3, 4, 5}, 0.25]` → `2.0` (first quartile)
/// - `Quantile[{1, 2, 3, 4, 5}, 0.75]` → `4.0` (third quartile)
pub fn quantile(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (list, quantile)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let mut numbers = extract_numeric_list(&args[0])?;
    let q = extract_single_number(&args[1])?;
    
    if !(0.0..=1.0).contains(&q) {
        return Err(VmError::TypeError {
            expected: "quantile between 0 and 1".to_string(),
            actual: format!("{}", q),
        });
    }
    
    if numbers.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    // Sort the numbers
    numbers.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = numbers.len();
    
    // Calculate position (using R-6 quantile method)
    let pos = q * (n - 1) as f64;
    let lower_index = pos.floor() as usize;
    let upper_index = pos.ceil() as usize;
    let fraction = pos - pos.floor();
    
    let quantile_value = if lower_index == upper_index {
        numbers[lower_index]
    } else {
        numbers[lower_index] * (1.0 - fraction) + numbers[upper_index] * fraction
    };
    
    Ok(Value::Real(quantile_value))
}

/// RandomReal[] - Generate random real number between 0 and 1
/// RandomReal[{min, max}] - Generate random real number between min and max
/// 
/// Examples:
/// - `RandomReal[]` → `0.7341...`
/// - `RandomReal[{-1, 1}]` → `-0.234...`
/// - `RandomReal[{0, 10}]` → `7.42...`
pub fn random_real(args: &[Value]) -> VmResult<Value> {
    let mut rng = rand::thread_rng();
    
    match args.len() {
        0 => {
            // RandomReal[] - between 0 and 1
            Ok(Value::Real(rng.gen::<f64>()))
        }
        1 => {
            // RandomReal[{min, max}]
            match &args[0] {
                Value::List(bounds) if bounds.len() == 2 => {
                    let min = extract_single_number(&bounds[0])?;
                    let max = extract_single_number(&bounds[1])?;
                    
                    if min >= max {
                        return Err(VmError::TypeError {
                            expected: "min < max".to_string(),
                            actual: format!("min={}, max={}", min, max),
                        });
                    }
                    
                    let random_val = rng.gen::<f64>() * (max - min) + min;
                    Ok(Value::Real(random_val))
                }
                _ => Err(VmError::TypeError {
                    expected: "list of two numbers {min, max}".to_string(),
                    actual: format!("{:?}", args[0]),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "0 or 1 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        })
    }
}

/// RandomInteger[n] - Generate random integer between 0 and n-1
/// RandomInteger[{min, max}] - Generate random integer between min and max (inclusive)
/// 
/// Examples:
/// - `RandomInteger[10]` → `7`
/// - `RandomInteger[{-5, 5}]` → `-2`
/// - `RandomInteger[{100, 200}]` → `157`
pub fn random_integer(args: &[Value]) -> VmResult<Value> {
    let mut rng = rand::thread_rng();
    
    match args.len() {
        1 => {
            match &args[0] {
                // RandomInteger[n] - between 0 and n-1
                Value::Integer(n) => {
                    if *n <= 0 {
                        return Err(VmError::TypeError {
                            expected: "positive integer".to_string(),
                            actual: format!("{}", n),
                        });
                    }
                    let random_val = rng.gen_range(0..*n);
                    Ok(Value::Integer(random_val))
                }
                // RandomInteger[{min, max}]
                Value::List(bounds) if bounds.len() == 2 => {
                    let min = match bounds[0] {
                        Value::Integer(n) => n,
                        Value::Real(r) => r as i64,
                        _ => return Err(VmError::TypeError {
                            expected: "integer".to_string(),
                            actual: format!("{:?}", bounds[0]),
                        })
                    };
                    let max = match bounds[1] {
                        Value::Integer(n) => n,
                        Value::Real(r) => r as i64,
                        _ => return Err(VmError::TypeError {
                            expected: "integer".to_string(),
                            actual: format!("{:?}", bounds[1]),
                        })
                    };
                    
                    if min > max {
                        return Err(VmError::TypeError {
                            expected: "min <= max".to_string(),
                            actual: format!("min={}, max={}", min, max),
                        });
                    }
                    
                    let random_val = rng.gen_range(min..=max);
                    Ok(Value::Integer(random_val))
                }
                _ => Err(VmError::TypeError {
                    expected: "integer or list of two integers {min, max}".to_string(),
                    actual: format!("{:?}", args[0]),
                })
            }
        }
        _ => Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        })
    }
}

/// Correlation[list1, list2] - Pearson correlation coefficient between two lists
/// 
/// Returns a value between -1 and 1:
/// - 1: perfect positive correlation
/// - 0: no correlation
/// - -1: perfect negative correlation
/// 
/// Examples:
/// - `Correlation[{1, 2, 3}, {1, 2, 3}]` → `1.0`
/// - `Correlation[{1, 2, 3}, {3, 2, 1}]` → `-1.0`
pub fn correlation(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (two lists)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = extract_numeric_list(&args[0])?;
    let y = extract_numeric_list(&args[1])?;
    
    if x.len() != y.len() {
        return Err(VmError::TypeError {
            expected: "lists of equal length".to_string(),
            actual: format!("lengths {} and {}", x.len(), y.len()),
        });
    }
    
    if x.len() < 2 {
        return Err(VmError::TypeError {
            expected: "lists with at least 2 elements".to_string(),
            actual: format!("lists with {} elements", x.len()),
        });
    }
    
    let n = x.len() as f64;
    
    // Calculate means
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    // Calculate correlation components
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    
    for i in 0..x.len() {
        let diff_x = x[i] - mean_x;
        let diff_y = y[i] - mean_y;
        
        numerator += diff_x * diff_y;
        sum_sq_x += diff_x * diff_x;
        sum_sq_y += diff_y * diff_y;
    }
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        // One or both lists have zero variance
        Ok(Value::Real(0.0))
    } else {
        let correlation = numerator / denominator;
        Ok(Value::Real(correlation))
    }
}

/// Covariance[list1, list2] - Sample covariance between two lists
/// 
/// Uses the unbiased sample covariance formula: Cov = Σ(x - μₓ)(y - μᵧ) / (n - 1)
/// 
/// Examples:
/// - `Covariance[{1, 2, 3}, {1, 2, 3}]` → `1.0`
/// - `Covariance[{1, 2, 3}, {3, 2, 1}]` → `-1.0`
pub fn covariance(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments (two lists)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let x = extract_numeric_list(&args[0])?;
    let y = extract_numeric_list(&args[1])?;
    
    if x.len() != y.len() {
        return Err(VmError::TypeError {
            expected: "lists of equal length".to_string(),
            actual: format!("lengths {} and {}", x.len(), y.len()),
        });
    }
    
    if x.len() < 2 {
        return Err(VmError::TypeError {
            expected: "lists with at least 2 elements".to_string(),
            actual: format!("lists with {} elements", x.len()),
        });
    }
    
    let n = x.len() as f64;
    
    // Calculate means
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    // Calculate covariance
    let covariance_sum: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    let covariance = covariance_sum / (x.len() - 1) as f64; // Sample covariance (n-1)
    
    Ok(Value::Real(covariance))
}

/// Min[list] - Minimum value in a list of numbers
/// 
/// Examples:
/// - `Min[{3, 1, 4, 1, 5}]` → `1`
/// - `Min[{-2.5, 0, 3.7}]` → `-2.5`
pub fn min(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let numbers = extract_numeric_list(&args[0])?;
    
    if numbers.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    let min_val = numbers.iter()
        .fold(f64::INFINITY, |acc, &x| acc.min(x));
    
    Ok(Value::Real(min_val))
}

/// Max[list] - Maximum value in a list of numbers
/// 
/// Examples:
/// - `Max[{3, 1, 4, 1, 5}]` → `5`
/// - `Max[{-2.5, 0, 3.7}]` → `3.7`
pub fn max(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let numbers = extract_numeric_list(&args[0])?;
    
    if numbers.is_empty() {
        return Err(VmError::TypeError {
            expected: "non-empty list".to_string(),
            actual: "empty list".to_string(),
        });
    }
    
    let max_val = numbers.iter()
        .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    
    Ok(Value::Real(max_val))
}

/// Total[list] - Sum of all elements in a list
/// 
/// Examples:
/// - `Total[{1, 2, 3, 4, 5}]` → `15`
/// - `Total[{1.5, 2.5, 3.0}]` → `7.0`
pub fn total(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument (list of numbers)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }

    let numbers = extract_numeric_list(&args[0])?;
    let sum = numbers.iter().sum::<f64>();
    
    Ok(Value::Real(sum))
}

// Helper functions

fn extract_numeric_list(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            let mut numbers = Vec::new();
            for item in items {
                match item {
                    Value::Integer(n) => numbers.push(*n as f64),
                    Value::Real(r) => numbers.push(*r),
                    _ => return Err(VmError::TypeError {
                        expected: "list of numbers".to_string(),
                        actual: format!("list containing {:?}", item),
                    })
                }
            }
            Ok(numbers)
        }
        _ => Err(VmError::TypeError {
            expected: "list".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

fn extract_single_number(value: &Value) -> VmResult<f64> {
    match value {
        Value::Integer(n) => Ok(*n as f64),
        Value::Real(r) => Ok(*r),
        _ => Err(VmError::TypeError {
            expected: "number".to_string(),
            actual: format!("{:?}", value),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        let list = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        
        let result = mean(&[list]).unwrap();
        assert_eq!(result, Value::Real(3.0));
    }

    #[test]
    fn test_mean_mixed_types() {
        let list = Value::List(vec![
            Value::Integer(1), Value::Real(2.5), Value::Integer(3)
        ]);
        
        let result = mean(&[list]).unwrap();
        assert!((extract_single_number(&result).unwrap() - 2.166666666666667).abs() < 1e-10);
    }

    #[test]
    fn test_variance_basic() {
        let list = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        
        let result = variance(&[list]).unwrap();
        assert_eq!(result, Value::Real(2.5));
    }

    #[test]
    fn test_standard_deviation() {
        let list = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        
        let result = standard_deviation(&[list]).unwrap();
        assert!((extract_single_number(&result).unwrap() - 1.5811388300841898).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd_length() {
        let list = Value::List(vec![
            Value::Real(5.0), Value::Real(1.0), Value::Real(3.0), Value::Real(9.0), Value::Real(2.0)
        ]);
        
        let result = median(&[list]).unwrap();
        assert_eq!(result, Value::Real(3.0));
    }

    #[test]
    fn test_median_even_length() {
        let list = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0)
        ]);
        
        let result = median(&[list]).unwrap();
        assert_eq!(result, Value::Real(2.5));
    }

    #[test]
    fn test_mode_single() {
        let list = Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(2), Value::Integer(3)
        ]);
        
        let result = mode(&[list]).unwrap();
        match result {
            Value::List(modes) => {
                assert_eq!(modes.len(), 1);
                assert_eq!(modes[0], Value::Integer(2));
            }
            _ => panic!("Expected list result for mode")
        }
    }

    #[test]
    fn test_quantile_median() {
        let list = Value::List(vec![
            Value::Real(1.0), Value::Real(2.0), Value::Real(3.0), Value::Real(4.0), Value::Real(5.0)
        ]);
        
        let result = quantile(&[list, Value::Real(0.5)]).unwrap();
        assert_eq!(result, Value::Real(3.0));
    }

    #[test]
    fn test_random_real_range() {
        let bounds = Value::List(vec![Value::Real(-1.0), Value::Real(1.0)]);
        let result = random_real(&[bounds]).unwrap();
        
        if let Value::Real(val) = result {
            assert!(val >= -1.0 && val <= 1.0);
        } else {
            panic!("Expected real number from RandomReal");
        }
    }

    #[test]
    fn test_random_integer_range() {
        let bounds = Value::List(vec![Value::Integer(-5), Value::Integer(5)]);
        let result = random_integer(&[bounds]).unwrap();
        
        if let Value::Integer(val) = result {
            assert!(val >= -5 && val <= 5);
        } else {
            panic!("Expected integer from RandomInteger");
        }
    }

    #[test]
    fn test_correlation_perfect_positive() {
        let x = Value::List(vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)]);
        let y = Value::List(vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)]);
        
        let result = correlation(&[x, y]).unwrap();
        assert!((extract_single_number(&result).unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_perfect_negative() {
        let x = Value::List(vec![Value::Real(1.0), Value::Real(2.0), Value::Real(3.0)]);
        let y = Value::List(vec![Value::Real(3.0), Value::Real(2.0), Value::Real(1.0)]);
        
        let result = correlation(&[x, y]).unwrap();
        assert!((extract_single_number(&result).unwrap() - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let list = Value::List(vec![
            Value::Real(3.0), Value::Real(1.0), Value::Real(4.0), Value::Real(1.0), Value::Real(5.0)
        ]);
        
        let min_result = min(&[list.clone()]).unwrap();
        assert_eq!(min_result, Value::Real(1.0));
        
        let max_result = max(&[list]).unwrap();
        assert_eq!(max_result, Value::Real(5.0));
    }

    #[test]
    fn test_total() {
        let list = Value::List(vec![
            Value::Integer(1), Value::Integer(2), Value::Integer(3), Value::Integer(4), Value::Integer(5)
        ]);
        
        let result = total(&[list]).unwrap();
        assert_eq!(result, Value::Real(15.0));
    }
}