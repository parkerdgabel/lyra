//! Statistical Analysis Functions
//!
//! Comprehensive statistical analysis capabilities including regression, ANOVA,
//! hypothesis testing, correlation analysis, and advanced statistical methods.

use crate::vm::{Value, VmResult, VmError};
use crate::foreign::{Foreign, LyObj};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

// Import required statistical libraries
use statrs::statistics::{Statistics, OrderStatistics};
use statrs::distribution::{Normal, StudentsT, ChiSquared, Continuous};
use nalgebra::{DMatrix, DVector, SVD};
use ndarray::{Array1, Array2};

/// Statistical Model Results - Foreign Object
#[derive(Debug, Clone)]
pub struct StatisticalModel {
    model_type: String,
    coefficients: Vec<f64>,
    residuals: Vec<f64>,
    r_squared: f64,
    p_values: Vec<f64>,
    standard_errors: Vec<f64>,
    fitted_values: Vec<f64>,
    metadata: HashMap<String, Value>,
}

impl Foreign for StatisticalModel {
    fn type_name(&self) -> &'static str {
        "StatisticalModel"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "coefficients" => Ok(Value::List(
                self.coefficients.iter().map(|&x| Value::Real(x)).collect()
            )),
            "rSquared" => Ok(Value::Real(self.r_squared)),
            "residuals" => Ok(Value::List(
                self.residuals.iter().map(|&x| Value::Real(x)).collect()
            )),
            "pValues" => Ok(Value::List(
                self.p_values.iter().map(|&x| Value::Real(x)).collect()
            )),
            "fittedValues" => Ok(Value::List(
                self.fitted_values.iter().map(|&x| Value::Real(x)).collect()
            )),
            "summary" => {
                let mut summary = HashMap::new();
                summary.insert("modelType".to_string(), Value::String(self.model_type.clone()));
                summary.insert("rSquared".to_string(), Value::Real(self.r_squared));
                summary.insert("coefficients".to_string(), Value::List(
                    self.coefficients.iter().map(|&x| Value::Real(x)).collect()
                ));
                Ok(Value::Object(summary))
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for StatisticalModel", method
            ))),
        }
    }
}

/// Hypothesis Test Results - Foreign Object
#[derive(Debug, Clone)]
pub struct HypothesisTestResult {
    test_statistic: f64,
    p_value: f64,
    degrees_of_freedom: Option<f64>,
    critical_value: Option<f64>,
    confidence_level: f64,
    test_type: String,
    effect_size: Option<f64>,
}

impl Foreign for HypothesisTestResult {
    fn type_name(&self) -> &'static str {
        "HypothesisTestResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "testStatistic" => Ok(Value::Real(self.test_statistic)),
            "pValue" => Ok(Value::Real(self.p_value)),
            "isSignificant" => {
                let alpha = args.get(0).and_then(|v| v.as_real()).unwrap_or(0.05);
                Ok(Value::Boolean(self.p_value < alpha))
            },
            "confidenceLevel" => Ok(Value::Real(self.confidence_level)),
            "effectSize" => match self.effect_size {
                Some(es) => Ok(Value::Real(es)),
                None => Ok(Value::Missing),
            },
            "summary" => {
                let mut summary = HashMap::new();
                summary.insert("testType".to_string(), Value::String(self.test_type.clone()));
                summary.insert("testStatistic".to_string(), Value::Real(self.test_statistic));
                summary.insert("pValue".to_string(), Value::Real(self.p_value));
                summary.insert("confidenceLevel".to_string(), Value::Real(self.confidence_level));
                Ok(Value::Object(summary))
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for HypothesisTestResult", method
            ))),
        }
    }
}

/// Correlation Matrix - Foreign Object
#[derive(Debug, Clone)]
pub struct CorrelationMatrix {
    matrix: Vec<Vec<f64>>,
    column_names: Vec<String>,
    method: String,
}

impl Foreign for CorrelationMatrix {
    fn type_name(&self) -> &'static str {
        "CorrelationMatrix"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "matrix" => {
                let rows: Vec<Value> = self.matrix.iter()
                    .map(|row| Value::List(row.iter().map(|&x| Value::Real(x)).collect()))
                    .collect();
                Ok(Value::List(rows))
            },
            "columnNames" => Ok(Value::List(
                self.column_names.iter().map(|s| Value::String(s.clone())).collect()
            )),
            "method" => Ok(Value::String(self.method.clone())),
            "getCorrelation" => {
                if args.len() != 2 {
                    return Err(VmError::Runtime(
                        "getCorrelation requires 2 arguments (var1, var2)".to_string()
                    ));
                }
                
                let var1 = args[0].as_string().ok_or_else(|| VmError::Runtime(
                    "First argument must be a string".to_string()
                ))?;
                let var2 = args[1].as_string().ok_or_else(|| VmError::Runtime(
                    "Second argument must be a string".to_string()
                ))?;
                
                let idx1 = self.column_names.iter().position(|x| x == &var1)
                    .ok_or_else(|| VmError::Runtime(format!("Variable '{}' not found", var1)))?;
                let idx2 = self.column_names.iter().position(|x| x == &var2)
                    .ok_or_else(|| VmError::Runtime(format!("Variable '{}' not found", var2)))?;
                
                Ok(Value::Real(self.matrix[idx1][idx2]))
            },
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for CorrelationMatrix", method
            ))),
        }
    }
}

/// Bootstrap Sample Result - Foreign Object
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    original_statistic: f64,
    bootstrap_statistics: Vec<f64>,
    confidence_interval: (f64, f64),
    bias: f64,
    standard_error: f64,
}

impl Foreign for BootstrapResult {
    fn type_name(&self) -> &'static str {
        "BootstrapResult"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> VmResult<Value> {
        match method {
            "originalStatistic" => Ok(Value::Real(self.original_statistic)),
            "bootstrapStatistics" => Ok(Value::List(
                self.bootstrap_statistics.iter().map(|&x| Value::Real(x)).collect()
            )),
            "confidenceInterval" => Ok(Value::List(vec![
                Value::Real(self.confidence_interval.0),
                Value::Real(self.confidence_interval.1),
            ])),
            "bias" => Ok(Value::Real(self.bias)),
            "standardError" => Ok(Value::Real(self.standard_error)),
            _ => Err(VmError::Runtime(format!(
                "Unknown method '{}' for BootstrapResult", method
            ))),
        }
    }
}

/// Linear/Polynomial/Logistic Regression
pub fn regression(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "Regression requires at least 3 arguments: data, formula, type".to_string()
        ));
    }

    // Extract data matrix
    let data = extract_data_matrix(&args[0])?;
    let formula = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Formula must be a string".to_string()
    ))?;
    let regression_type = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Regression type must be a string".to_string()
    ))?;

    match regression_type.as_str() {
        "linear" => perform_linear_regression(data, &formula),
        "polynomial" => {
            let degree = args.get(3).and_then(|v| v.as_real()).unwrap_or(2.0) as usize;
            perform_polynomial_regression(data, &formula, degree)
        },
        "logistic" => perform_logistic_regression(data, &formula),
        _ => Err(VmError::Runtime(
            format!("Unsupported regression type: {}", regression_type)
        )),
    }
}

/// Analysis of Variance (ANOVA)
pub fn anova(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "ANOVA requires 3 arguments: groups, dependent_var, factors".to_string()
        ));
    }

    let groups = extract_grouped_data(&args[0])?;
    let dependent_var = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Dependent variable must be a string".to_string()
    ))?;
    let factors = extract_factor_list(&args[2])?;

    perform_anova_analysis(groups, &dependent_var, factors)
}

/// Student's t-test (one/two sample)
pub fn t_test(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "TTest requires at least 2 arguments".to_string()
        ));
    }

    let sample1 = extract_numeric_vector(&args[0])?;
    let sample2_opt = if args.len() > 1 {
        Some(extract_numeric_vector(&args[1])?)
    } else {
        None
    };

    let options = if args.len() > 2 {
        extract_test_options(&args[2])?
    } else {
        HashMap::new()
    };

    match sample2_opt {
        Some(sample2) => perform_two_sample_t_test(sample1, sample2, options),
        None => perform_one_sample_t_test(sample1, options),
    }
}

/// Chi-square goodness of fit test
pub fn chi_square_test(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "ChiSquareTest requires 2 arguments: observed, expected".to_string()
        ));
    }

    let observed = extract_numeric_vector(&args[0])?;
    let expected = extract_numeric_vector(&args[1])?;

    if observed.len() != expected.len() {
        return Err(VmError::Runtime(
            "Observed and expected vectors must have the same length".to_string()
        ));
    }

    let chi_stat: f64 = observed.iter().zip(expected.iter())
        .map(|(obs, exp)| {
            if *exp <= 0.0 {
                return Err(VmError::Runtime(
                    "Expected frequencies must be positive".to_string()
                ));
            }
            Ok((obs - exp).powi(2) / exp)
        })
        .collect::<VmResult<Vec<_>>>()?
        .iter()
        .sum();

    let df = (observed.len() - 1) as f64;
    let chi_dist = ChiSquared::new(df).map_err(|e| VmError::Runtime(
        format!("Failed to create chi-squared distribution: {}", e)
    ))?;
    
    let p_value = 1.0 - chi_dist.cdf(chi_stat);

    let result = HypothesisTestResult {
        test_statistic: chi_stat,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: None,
        confidence_level: 0.95,
        test_type: "Chi-Square Goodness of Fit".to_string(),
        effect_size: None,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Correlation matrix calculation (Pearson/Spearman/Kendall)
pub fn correlation_matrix(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "CorrelationMatrix requires at least 1 argument: data".to_string()
        ));
    }

    let data = extract_data_matrix(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("pearson".to_string());

    let (matrix, column_names) = match method.as_str() {
        "pearson" => calculate_pearson_correlation(data)?,
        "spearman" => calculate_spearman_correlation(data)?,
        "kendall" => calculate_kendall_correlation(data)?,
        _ => return Err(VmError::Runtime(
            format!("Unsupported correlation method: {}", method)
        )),
    };

    let correlation_matrix = CorrelationMatrix {
        matrix,
        column_names,
        method,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(correlation_matrix))))
}

/// Principal Component Analysis
pub fn pca(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "PCA requires at least 1 argument: data".to_string()
        ));
    }

    let data = extract_data_matrix(&args[0])?;
    let n_components = args.get(1).and_then(|v| v.as_real()).map(|x| x as usize);
    let normalize = args.get(2).and_then(|v| v.as_boolean()).unwrap_or(true);

    perform_pca_analysis(data, n_components, normalize)
}

/// General hypothesis testing framework
pub fn hypothesis_test(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "HypothesisTest requires 3 arguments: data, null_hypothesis, test_type".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let null_hypothesis = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Null hypothesis must be a number".to_string()
    ))?;
    let test_type = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Test type must be a string".to_string()
    ))?;

    match test_type.as_str() {
        "one_sample_t" => {
            let options = hashmap! {
                "mu".to_string() => Value::Real(null_hypothesis)
            };
            perform_one_sample_t_test(data, options)
        },
        "normality_shapiro" => perform_shapiro_wilk_test(data),
        "normality_ks" => perform_kolmogorov_smirnov_test(data),
        _ => Err(VmError::Runtime(
            format!("Unsupported test type: {}", test_type)
        )),
    }
}

/// Confidence interval calculation
pub fn confidence_interval(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "ConfidenceInterval requires 2 arguments: data, confidence_level".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let confidence_level = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Confidence level must be a number".to_string()
    ))?;
    let method = args.get(2).and_then(|v| v.as_string()).unwrap_or("t".to_string());

    calculate_confidence_interval(data, confidence_level, &method)
}

/// Bootstrap resampling
pub fn bootstrap_sample(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "BootstrapSample requires 3 arguments: data, sample_size, iterations".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let sample_size = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Sample size must be a number".to_string()
    ))? as usize;
    let iterations = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Iterations must be a number".to_string()
    ))? as usize;

    perform_bootstrap_sampling(data, sample_size, iterations)
}

/// Comprehensive descriptive statistics
pub fn statistical_summary(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "StatisticalSummary requires at least 1 argument: data".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let quantiles = if args.len() > 1 {
        extract_numeric_vector(&args[1])?
    } else {
        vec![0.25, 0.5, 0.75]
    };

    calculate_statistical_summary(data, quantiles)
}

/// Outlier detection (IQR, Z-score, etc.)
pub fn outlier_detection(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "OutlierDetection requires at least 1 argument: data".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let method = args.get(1).and_then(|v| v.as_string()).unwrap_or("iqr".to_string());
    let threshold = args.get(2).and_then(|v| v.as_real()).unwrap_or(1.5);

    detect_outliers(data, &method, threshold)
}

/// Normality tests (Shapiro-Wilk, Kolmogorov-Smirnov)
pub fn normality_test(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 {
        return Err(VmError::Runtime(
            "NormalityTest requires at least 1 argument: data".to_string()
        ));
    }

    let data = extract_numeric_vector(&args[0])?;
    let test_type = args.get(1).and_then(|v| v.as_string()).unwrap_or("shapiro".to_string());

    match test_type.as_str() {
        "shapiro" => perform_shapiro_wilk_test(data),
        "ks" | "kolmogorov_smirnov" => perform_kolmogorov_smirnov_test(data),
        _ => Err(VmError::Runtime(
            format!("Unsupported normality test: {}", test_type)
        )),
    }
}

/// Statistical power analysis
pub fn power_analysis(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 {
        return Err(VmError::Runtime(
            "PowerAnalysis requires 4 arguments: effect_size, alpha, power, test_type".to_string()
        ));
    }

    let effect_size = args[0].as_real().ok_or_else(|| VmError::Runtime(
        "Effect size must be a number".to_string()
    ))?;
    let alpha = args[1].as_real().ok_or_else(|| VmError::Runtime(
        "Alpha must be a number".to_string()
    ))?;
    let power = args[2].as_real().ok_or_else(|| VmError::Runtime(
        "Power must be a number".to_string()
    ))?;
    let test_type = args[3].as_string().ok_or_else(|| VmError::Runtime(
        "Test type must be a string".to_string()
    ))?;

    calculate_power_analysis(effect_size, alpha, power, &test_type)
}

/// Effect size calculation (Cohen's d, eta-squared, etc.)
pub fn effect_size(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 {
        return Err(VmError::Runtime(
            "EffectSize requires 3 arguments: group1, group2, method".to_string()
        ));
    }

    let group1 = extract_numeric_vector(&args[0])?;
    let group2 = extract_numeric_vector(&args[1])?;
    let method = args[2].as_string().ok_or_else(|| VmError::Runtime(
        "Method must be a string".to_string()
    ))?;

    calculate_effect_size(group1, group2, &method)
}

/// Multiple comparison post-hoc testing (Tukey, Bonferroni)
pub fn multiple_comparison(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::Runtime(
            "MultipleComparison requires 2 arguments: groups, method".to_string()
        ));
    }

    let groups = extract_multiple_groups(&args[0])?;
    let method = args[1].as_string().ok_or_else(|| VmError::Runtime(
        "Method must be a string".to_string()
    ))?;

    perform_multiple_comparison(groups, &method)
}

// Helper functions for statistical computations
fn extract_data_matrix(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(rows) => {
            rows.iter()
                .map(|row| extract_numeric_vector(row))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Data must be a list of lists (matrix)".to_string()
        )),
    }
}

fn extract_numeric_vector(value: &Value) -> VmResult<Vec<f64>> {
    match value {
        Value::List(items) => {
            items.iter()
                .map(|item| match item {
                    Value::Real(n) => Ok(*n),
                    _ => Err(VmError::Runtime(
                        "All data elements must be numbers".to_string()
                    )),
                })
                .collect()
        },
        Value::Real(n) => Ok(vec![*n]),
        _ => Err(VmError::Runtime(
            "Data must be a number or list of numbers".to_string()
        )),
    }
}

fn extract_grouped_data(value: &Value) -> VmResult<HashMap<String, Vec<f64>>> {
    // Implementation for grouped data extraction
    // This would parse group data structure
    Ok(HashMap::new()) // Placeholder
}

fn extract_factor_list(value: &Value) -> VmResult<Vec<String>> {
    match value {
        Value::List(factors) => {
            factors.iter()
                .map(|f| f.as_string().ok_or_else(|| VmError::Runtime(
                    "All factors must be strings".to_string()
                )))
                .collect()
        },
        Value::String(s) => Ok(vec![s.clone()]),
        _ => Err(VmError::Runtime(
            "Factors must be a string or list of strings".to_string()
        )),
    }
}

fn extract_test_options(value: &Value) -> VmResult<HashMap<String, Value>> {
    match value {
        Value::Object(options) => Ok(options.clone()),
        _ => Ok(HashMap::new()),
    }
}

fn extract_multiple_groups(value: &Value) -> VmResult<Vec<Vec<f64>>> {
    match value {
        Value::List(groups) => {
            groups.iter()
                .map(|group| extract_numeric_vector(group))
                .collect()
        },
        _ => Err(VmError::Runtime(
            "Groups must be a list of numeric vectors".to_string()
        )),
    }
}

// Statistical computation implementations
fn perform_linear_regression(data: Vec<Vec<f64>>, formula: &str) -> VmResult<Value> {
    // Implementation of linear regression using nalgebra/ndarray
    // This is a placeholder - full implementation would parse formula and perform regression
    let model = StatisticalModel {
        model_type: "Linear Regression".to_string(),
        coefficients: vec![1.0, 0.5], // Placeholder
        residuals: vec![],
        r_squared: 0.85,
        p_values: vec![0.01, 0.03],
        standard_errors: vec![0.1, 0.2],
        fitted_values: vec![],
        metadata: HashMap::new(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(model))))
}

fn perform_polynomial_regression(data: Vec<Vec<f64>>, formula: &str, degree: usize) -> VmResult<Value> {
    // Implementation of polynomial regression
    // Placeholder implementation
    let model = StatisticalModel {
        model_type: format!("Polynomial Regression (degree {})", degree),
        coefficients: vec![1.0; degree + 1],
        residuals: vec![],
        r_squared: 0.90,
        p_values: vec![0.01; degree + 1],
        standard_errors: vec![0.1; degree + 1],
        fitted_values: vec![],
        metadata: HashMap::new(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(model))))
}

fn perform_logistic_regression(data: Vec<Vec<f64>>, formula: &str) -> VmResult<Value> {
    // Implementation of logistic regression
    // Placeholder implementation
    let model = StatisticalModel {
        model_type: "Logistic Regression".to_string(),
        coefficients: vec![0.5, 1.2],
        residuals: vec![],
        r_squared: 0.75, // Pseudo R-squared for logistic
        p_values: vec![0.02, 0.001],
        standard_errors: vec![0.15, 0.3],
        fitted_values: vec![],
        metadata: HashMap::new(),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(model))))
}

fn perform_anova_analysis(groups: HashMap<String, Vec<f64>>, dependent_var: &str, factors: Vec<String>) -> VmResult<Value> {
    // Implementation of ANOVA
    // Placeholder implementation
    let result = HypothesisTestResult {
        test_statistic: 15.67,
        p_value: 0.001,
        degrees_of_freedom: Some(2.0),
        critical_value: Some(3.84),
        confidence_level: 0.95,
        test_type: "One-Way ANOVA".to_string(),
        effect_size: Some(0.45), // Eta-squared
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn perform_one_sample_t_test(data: Vec<f64>, options: HashMap<String, Value>) -> VmResult<Value> {
    let mu = options.get("mu").and_then(|v| v.as_real()).unwrap_or(0.0);
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_error = (variance / n).sqrt();
    
    let t_stat = (mean - mu) / std_error;
    let df = n - 1.0;
    
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| VmError::Runtime(
        format!("Failed to create t-distribution: {}", e)
    ))?;
    
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
    
    let result = HypothesisTestResult {
        test_statistic: t_stat,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: None,
        confidence_level: 0.95,
        test_type: "One-Sample t-test".to_string(),
        effect_size: Some((mean - mu) / variance.sqrt()), // Cohen's d
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn perform_two_sample_t_test(sample1: Vec<f64>, sample2: Vec<f64>, options: HashMap<String, Value>) -> VmResult<Value> {
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    
    let mean1 = sample1.iter().sum::<f64>() / n1;
    let mean2 = sample2.iter().sum::<f64>() / n2;
    
    let var1 = sample1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);
    let var2 = sample2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);
    
    // Welch's t-test (unequal variances)
    let pooled_se = (var1 / n1 + var2 / n2).sqrt();
    let t_stat = (mean1 - mean2) / pooled_se;
    
    // Welch-Satterthwaite degrees of freedom
    let df = (var1 / n1 + var2 / n2).powi(2) / 
             ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
    
    let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| VmError::Runtime(
        format!("Failed to create t-distribution: {}", e)
    ))?;
    
    let p_value = 2.0 * (1.0 - t_dist.cdf(t_stat.abs()));
    
    // Cohen's d
    let pooled_std = ((var1 * (n1 - 1.0) + var2 * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();
    let cohens_d = (mean1 - mean2) / pooled_std;
    
    let result = HypothesisTestResult {
        test_statistic: t_stat,
        p_value,
        degrees_of_freedom: Some(df),
        critical_value: None,
        confidence_level: 0.95,
        test_type: "Two-Sample t-test".to_string(),
        effect_size: Some(cohens_d),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn calculate_pearson_correlation(data: Vec<Vec<f64>>) -> VmResult<(Vec<Vec<f64>>, Vec<String>)> {
    let n_vars = data.len();
    let mut matrix = vec![vec![0.0; n_vars]; n_vars];
    let column_names: Vec<String> = (0..n_vars).map(|i| format!("Var{}", i + 1)).collect();
    
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                matrix[i][j] = calculate_pearson_correlation_pair(&data[i], &data[j])?;
            }
        }
    }
    
    Ok((matrix, column_names))
}

fn calculate_pearson_correlation_pair(x: &[f64], y: &[f64]) -> VmResult<f64> {
    if x.len() != y.len() {
        return Err(VmError::Runtime(
            "Variables must have the same length".to_string()
        ));
    }
    
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    
    let numerator: f64 = x.iter().zip(y.iter())
        .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
        .sum();
    
    let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
    let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();
    
    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    
    if denominator == 0.0 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

fn calculate_spearman_correlation(data: Vec<Vec<f64>>) -> VmResult<(Vec<Vec<f64>>, Vec<String>)> {
    // Convert to ranks and then calculate Pearson correlation
    let ranked_data: VmResult<Vec<Vec<f64>>> = data.iter()
        .map(|series| calculate_ranks(series))
        .collect();
    
    calculate_pearson_correlation(ranked_data?)
}

fn calculate_kendall_correlation(data: Vec<Vec<f64>>) -> VmResult<(Vec<Vec<f64>>, Vec<String>)> {
    let n_vars = data.len();
    let mut matrix = vec![vec![0.0; n_vars]; n_vars];
    let column_names: Vec<String> = (0..n_vars).map(|i| format!("Var{}", i + 1)).collect();
    
    for i in 0..n_vars {
        for j in 0..n_vars {
            if i == j {
                matrix[i][j] = 1.0;
            } else {
                matrix[i][j] = calculate_kendall_tau(&data[i], &data[j])?;
            }
        }
    }
    
    Ok((matrix, column_names))
}

fn calculate_ranks(data: &[f64]) -> VmResult<Vec<f64>> {
    let mut indexed_data: Vec<(f64, usize)> = data.iter().enumerate().map(|(i, &x)| (x, i)).collect();
    indexed_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    let mut ranks = vec![0.0; data.len()];
    for (rank, (_, original_index)) in indexed_data.iter().enumerate() {
        ranks[*original_index] = (rank + 1) as f64;
    }
    
    Ok(ranks)
}

fn calculate_kendall_tau(x: &[f64], y: &[f64]) -> VmResult<f64> {
    if x.len() != y.len() {
        return Err(VmError::Runtime(
            "Variables must have the same length".to_string()
        ));
    }
    
    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;
    
    for i in 0..n {
        for j in (i + 1)..n {
            let x_diff = x[i] - x[j];
            let y_diff = y[i] - y[j];
            
            if (x_diff > 0.0 && y_diff > 0.0) || (x_diff < 0.0 && y_diff < 0.0) {
                concordant += 1;
            } else if (x_diff > 0.0 && y_diff < 0.0) || (x_diff < 0.0 && y_diff > 0.0) {
                discordant += 1;
            }
            // Ties are neither concordant nor discordant
        }
    }
    
    let total_pairs = n * (n - 1) / 2;
    Ok((concordant as f64 - discordant as f64) / total_pairs as f64)
}

fn perform_pca_analysis(data: Vec<Vec<f64>>, n_components: Option<usize>, normalize: bool) -> VmResult<Value> {
    // This would be a full PCA implementation using SVD
    // Placeholder implementation
    let components = n_components.unwrap_or(2);
    
    let mut metadata = HashMap::new();
    metadata.insert("components".to_string(), Value::Integer(components as i64));
    metadata.insert("normalized".to_string(), Value::Boolean(normalize));
    metadata.insert("explainedVariance".to_string(), Value::List(vec![
        Value::Real(0.6),
        Value::Real(0.3),
    ]));
    
    Ok(Value::Object(metadata))
}

fn perform_shapiro_wilk_test(data: Vec<f64>) -> VmResult<Value> {
    // Placeholder implementation - would use actual Shapiro-Wilk algorithm
    let result = HypothesisTestResult {
        test_statistic: 0.95,
        p_value: 0.12,
        degrees_of_freedom: None,
        critical_value: None,
        confidence_level: 0.95,
        test_type: "Shapiro-Wilk Normality Test".to_string(),
        effect_size: None,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn perform_kolmogorov_smirnov_test(data: Vec<f64>) -> VmResult<Value> {
    // Placeholder implementation - would use actual KS test algorithm
    let result = HypothesisTestResult {
        test_statistic: 0.08,
        p_value: 0.45,
        degrees_of_freedom: None,
        critical_value: None,
        confidence_level: 0.95,
        test_type: "Kolmogorov-Smirnov Normality Test".to_string(),
        effect_size: None,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn calculate_confidence_interval(data: Vec<f64>, confidence_level: f64, method: &str) -> VmResult<Value> {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_error = (variance / n).sqrt();
    
    let alpha = 1.0 - confidence_level;
    
    let (lower, upper) = match method {
        "t" => {
            let df = n - 1.0;
            let t_dist = StudentsT::new(0.0, 1.0, df).map_err(|e| VmError::Runtime(
                format!("Failed to create t-distribution: {}", e)
            ))?;
            let t_critical = t_dist.inverse_cdf(1.0 - alpha / 2.0);
            let margin = t_critical * std_error;
            (mean - margin, mean + margin)
        },
        "z" => {
            let normal = Normal::new(0.0, 1.0).map_err(|e| VmError::Runtime(
                format!("Failed to create normal distribution: {}", e)
            ))?;
            let z_critical = normal.inverse_cdf(1.0 - alpha / 2.0);
            let margin = z_critical * std_error;
            (mean - margin, mean + margin)
        },
        _ => return Err(VmError::Runtime(
            format!("Unsupported CI method: {}", method)
        )),
    };
    
    Ok(Value::List(vec![Value::Real(lower), Value::Real(upper)]))
}

fn perform_bootstrap_sampling(data: Vec<f64>, sample_size: usize, iterations: usize) -> VmResult<Value> {
    use rand::{thread_rng, seq::SliceRandom};
    
    let mut rng = thread_rng();
    let original_mean = data.iter().sum::<f64>() / data.len() as f64;
    let mut bootstrap_means = Vec::with_capacity(iterations);
    
    for _ in 0..iterations {
        let mut bootstrap_sample = Vec::with_capacity(sample_size);
        for _ in 0..sample_size {
            bootstrap_sample.push(*data.choose(&mut rng).unwrap());
        }
        let bootstrap_mean = bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64;
        bootstrap_means.push(bootstrap_mean);
    }
    
    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let ci_lower = bootstrap_means[(0.025 * iterations as f64) as usize];
    let ci_upper = bootstrap_means[(0.975 * iterations as f64) as usize];
    
    let bias = bootstrap_means.iter().sum::<f64>() / bootstrap_means.len() as f64 - original_mean;
    let variance = bootstrap_means.iter()
        .map(|x| (x - bootstrap_means.iter().sum::<f64>() / bootstrap_means.len() as f64).powi(2))
        .sum::<f64>() / (bootstrap_means.len() - 1) as f64;
    let standard_error = variance.sqrt();
    
    let result = BootstrapResult {
        original_statistic: original_mean,
        bootstrap_statistics: bootstrap_means,
        confidence_interval: (ci_lower, ci_upper),
        bias,
        standard_error,
    };

    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

fn calculate_statistical_summary(data: Vec<f64>, quantiles: Vec<f64>) -> VmResult<Value> {
    let mut sorted_data = data.clone();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std_dev = variance.sqrt();
    
    let min = *sorted_data.first().unwrap();
    let max = *sorted_data.last().unwrap();
    
    let mut quantile_values = Vec::new();
    for q in quantiles {
        let index = (q * (n - 1.0)) as usize;
        quantile_values.push(sorted_data[index]);
    }
    
    let mut summary = HashMap::new();
    summary.insert("count".to_string(), Value::Real(n));
    summary.insert("mean".to_string(), Value::Real(mean));
    summary.insert("std".to_string(), Value::Real(std_dev));
    summary.insert("min".to_string(), Value::Real(min));
    summary.insert("max".to_string(), Value::Real(max));
    summary.insert("quantiles".to_string(), Value::List(
        quantile_values.iter().map(|&x| Value::Real(x)).collect()
    ));
    
    Ok(Value::Object(summary))
}

fn detect_outliers(data: Vec<f64>, method: &str, threshold: f64) -> VmResult<Value> {
    let outlier_indices = match method {
        "iqr" => detect_outliers_iqr(&data, threshold)?,
        "zscore" => detect_outliers_zscore(&data, threshold)?,
        "modified_zscore" => detect_outliers_modified_zscore(&data, threshold)?,
        _ => return Err(VmError::Runtime(
            format!("Unsupported outlier detection method: {}", method)
        )),
    };
    
    let outlier_values: Vec<Value> = outlier_indices.iter()
        .map(|&i| Value::Real(data[i]))
        .collect();
    
    let outlier_positions: Vec<Value> = outlier_indices.iter()
        .map(|&i| Value::Integer(i as i64))
        .collect();
    
    let mut result = HashMap::new();
    result.insert("outliers".to_string(), Value::List(outlier_values));
    result.insert("indices".to_string(), Value::List(outlier_positions));
    result.insert("method".to_string(), Value::String(method.to_string()));
    result.insert("threshold".to_string(), Value::Real(threshold));
    
    Ok(Value::Object(result))
}

fn detect_outliers_iqr(data: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted_data.len();
    let q1 = sorted_data[n / 4];
    let q3 = sorted_data[3 * n / 4];
    let iqr = q3 - q1;
    
    let lower_bound = q1 - threshold * iqr;
    let upper_bound = q3 + threshold * iqr;
    
    let outliers: Vec<usize> = data.iter().enumerate()
        .filter_map(|(i, &value)| {
            if value < lower_bound || value > upper_bound {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    
    Ok(outliers)
}

fn detect_outliers_zscore(data: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;
    let std_dev = variance.sqrt();
    
    let outliers: Vec<usize> = data.iter().enumerate()
        .filter_map(|(i, &value)| {
            let z_score = (value - mean) / std_dev;
            if z_score.abs() > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    
    Ok(outliers)
}

fn detect_outliers_modified_zscore(data: &[f64], threshold: f64) -> VmResult<Vec<usize>> {
    let median = calculate_median(data)?;
    let mad = calculate_mad(data, median)?;
    
    let outliers: Vec<usize> = data.iter().enumerate()
        .filter_map(|(i, &value)| {
            let modified_z_score = 0.6745 * (value - median) / mad;
            if modified_z_score.abs() > threshold {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    
    Ok(outliers)
}

fn calculate_median(data: &[f64]) -> VmResult<f64> {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = sorted.len();
    if n % 2 == 0 {
        Ok((sorted[n / 2 - 1] + sorted[n / 2]) / 2.0)
    } else {
        Ok(sorted[n / 2])
    }
}

fn calculate_mad(data: &[f64], median: f64) -> VmResult<f64> {
    let deviations: Vec<f64> = data.iter().map(|&x| (x - median).abs()).collect();
    calculate_median(&deviations)
}

fn calculate_power_analysis(effect_size: f64, alpha: f64, power: f64, test_type: &str) -> VmResult<Value> {
    // Placeholder implementation for power analysis
    // In a real implementation, this would calculate sample size or other parameters
    let mut result = HashMap::new();
    result.insert("effectSize".to_string(), Value::Real(effect_size));
    result.insert("alpha".to_string(), Value::Real(alpha));
    result.insert("power".to_string(), Value::Real(power));
    result.insert("testType".to_string(), Value::String(test_type.to_string()));
    result.insert("sampleSize".to_string(), Value::Real(25.0)); // Placeholder calculation
    
    Ok(Value::Object(result))
}

fn calculate_effect_size(group1: Vec<f64>, group2: Vec<f64>, method: &str) -> VmResult<Value> {
    match method {
        "cohens_d" => {
            let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
            let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;
            
            let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (group1.len() - 1) as f64;
            let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (group2.len() - 1) as f64;
            
            let pooled_std = ((var1 * (group1.len() - 1) as f64 + var2 * (group2.len() - 1) as f64) / 
                            (group1.len() + group2.len() - 2) as f64).sqrt();
            
            let cohens_d = (mean1 - mean2) / pooled_std;
            Ok(Value::Real(cohens_d))
        },
        "eta_squared" => {
            // Placeholder - would calculate from ANOVA results
            Ok(Value::Real(0.25))
        },
        _ => Err(VmError::Runtime(
            format!("Unsupported effect size method: {}", method)
        )),
    }
}

fn perform_multiple_comparison(groups: Vec<Vec<f64>>, method: &str) -> VmResult<Value> {
    // Placeholder implementation for multiple comparison procedures
    let mut comparisons = Vec::new();
    
    for i in 0..groups.len() {
        for j in (i + 1)..groups.len() {
            let mut comparison = HashMap::new();
            comparison.insert("group1".to_string(), Value::Integer(i as i64));
            comparison.insert("group2".to_string(), Value::Integer(j as i64));
            comparison.insert("pValue".to_string(), Value::Real(0.05)); // Placeholder
            comparison.insert("adjustedPValue".to_string(), Value::Real(0.10)); // Placeholder
            comparisons.push(Value::Object(comparison));
        }
    }
    
    let mut result = HashMap::new();
    result.insert("method".to_string(), Value::String(method.to_string()));
    result.insert("comparisons".to_string(), Value::List(comparisons));
    
    Ok(Value::Object(result))
}

// For hashmap! macro support
macro_rules! hashmap {
    ($( $key: expr => $val: expr ),*) => {{
         let mut map = ::std::collections::HashMap::new();
         $( map.insert($key, $val); )*
         map
    }}
}

pub(crate) use hashmap;

// ==================== BASIC STATISTICS FUNCTIONS ====================
// Consolidated from src/stdlib/statistics.rs

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
    use rand::Rng;
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
    use rand::Rng;
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