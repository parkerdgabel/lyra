//! TDD Tests for Phase 15B: Advanced Analytics & Statistics - Statistical Analysis Module
//!
//! Following strict TDD principles with RED-GREEN-REFACTOR approach.
//! Tests are written first to describe expected behavior before implementation.

use lyra::vm::{Value, VmError};
use lyra::foreign::LyObj;
use lyra::stdlib::analytics::statistics::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;

#[cfg(test)]
mod statistical_analysis_tests {
    use super::*;

    /// Test RED: Linear regression with simple data should return a StatisticalModel
    #[test]
    fn test_linear_regression_basic() {
        // Arrange - Simple linear data: y = 2x + 1
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![3.0, 5.0, 7.0, 9.0, 11.0];
        let data_matrix = vec![x_data, y_data];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let formula = Value::String("y ~ x".to_string());
        let regression_type = Value::String("linear".to_string());
        
        // Act
        let result = regression(&[data, formula, regression_type]);
        
        // Assert
        assert!(result.is_ok());
        let model = result.unwrap();
        
        // Verify it's a StatisticalModel Foreign object
        if let Value::LyObj(obj) = model {
            let typename = obj.foreign().typename();
            assert_eq!(typename, "StatisticalModel");
            
            // Test model methods
            let coefficients = obj.foreign().call_method("coefficients", &[]).unwrap();
            assert!(matches!(coefficients, Value::List(_)));
            
            let r_squared = obj.foreign().call_method("rSquared", &[]).unwrap();
            assert!(matches!(r_squared, Value::Number(_)));
        } else {
            panic!("Expected LyObj with StatisticalModel");
        }
    }

    /// Test RED: Polynomial regression should accept degree parameter
    #[test]
    fn test_polynomial_regression_with_degree() {
        // Arrange - Quadratic data
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let y_data = vec![1.0, 4.0, 9.0, 16.0]; // y = x^2
        let data_matrix = vec![x_data, y_data];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let formula = Value::String("y ~ x".to_string());
        let regression_type = Value::String("polynomial".to_string());
        let degree = Value::Number(2.0);
        
        // Act
        let result = regression(&[data, formula, regression_type, degree]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            let summary = obj.foreign().call_method("summary", &[]).unwrap();
            if let Value::Object(summary_map) = summary {
                assert!(summary_map.contains_key("modelType"));
                if let Some(Value::String(model_type)) = summary_map.get("modelType") {
                    assert!(model_type.contains("Polynomial"));
                    assert!(model_type.contains("degree 2"));
                }
            }
        }
    }

    /// Test RED: Logistic regression should handle binary classification
    #[test]
    fn test_logistic_regression() {
        // Arrange - Binary classification data
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_data = vec![0.0, 0.0, 1.0, 1.0, 1.0]; // Binary outcomes
        let data_matrix = vec![x_data, y_data];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let formula = Value::String("y ~ x".to_string());
        let regression_type = Value::String("logistic".to_string());
        
        // Act
        let result = regression(&[data, formula, regression_type]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            let summary = obj.foreign().call_method("summary", &[]).unwrap();
            if let Value::Object(summary_map) = summary {
                if let Some(Value::String(model_type)) = summary_map.get("modelType") {
                    assert_eq!(model_type, "Logistic Regression");
                }
            }
        }
    }

    /// Test RED: ANOVA should analyze variance between groups
    #[test]
    fn test_anova_one_way() {
        // Arrange - Three groups with different means
        let groups = HashMap::from([
            ("group1".to_string(), vec![1.0, 2.0, 3.0]),
            ("group2".to_string(), vec![4.0, 5.0, 6.0]),
            ("group3".to_string(), vec![7.0, 8.0, 9.0]),
        ]);
        
        let groups_value = Value::Object(HashMap::new()); // Simplified for test
        let dependent_var = Value::String("response".to_string());
        let factors = Value::List(vec![Value::String("group".to_string())]);
        
        // Act
        let result = anova(&[groups_value, dependent_var, factors]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "HypothesisTestResult");
            
            let test_statistic = obj.foreign().call_method("testStatistic", &[]).unwrap();
            assert!(matches!(test_statistic, Value::Number(_)));
            
            let p_value = obj.foreign().call_method("pValue", &[]).unwrap();
            assert!(matches!(p_value, Value::Number(_)));
        }
    }

    /// Test RED: One-sample t-test should test against population mean
    #[test]
    fn test_one_sample_t_test() {
        // Arrange - Sample data that should be significantly different from 0
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        
        // Act
        let result = t_test(&[sample]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "HypothesisTestResult");
            
            let test_statistic = obj.foreign().call_method("testStatistic", &[]).unwrap();
            assert!(matches!(test_statistic, Value::Number(_)));
            
            let p_value = obj.foreign().call_method("pValue", &[]).unwrap();
            assert!(matches!(p_value, Value::Number(_)));
            
            let effect_size = obj.foreign().call_method("effectSize", &[]).unwrap();
            assert!(matches!(effect_size, Value::Number(_)));
        }
    }

    /// Test RED: Two-sample t-test should compare two groups
    #[test]
    fn test_two_sample_t_test() {
        // Arrange - Two samples with different means
        let sample1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sample2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        
        let sample1_val = Value::List(sample1.iter().map(|&x| Value::Number(x)).collect());
        let sample2_val = Value::List(sample2.iter().map(|&x| Value::Number(x)).collect());
        
        // Act
        let result = t_test(&[sample1_val, sample2_val]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "HypothesisTestResult");
            
            // Test significance checking
            let alpha = Value::Number(0.05);
            let is_significant = obj.foreign().call_method("isSignificant", &[alpha]).unwrap();
            assert!(matches!(is_significant, Value::Boolean(_)));
        }
    }

    /// Test RED: Chi-square test should validate goodness of fit
    #[test]
    fn test_chi_square_goodness_of_fit() {
        // Arrange - Observed vs expected frequencies
        let observed = vec![10.0, 15.0, 8.0, 12.0];
        let expected = vec![11.25, 11.25, 11.25, 11.25]; // Equal expected frequencies
        
        let observed_val = Value::List(observed.iter().map(|&x| Value::Number(x)).collect());
        let expected_val = Value::List(expected.iter().map(|&x| Value::Number(x)).collect());
        
        // Act
        let result = chi_square_test(&[observed_val, expected_val]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "HypothesisTestResult");
            
            let summary = obj.foreign().call_method("summary", &[]).unwrap();
            if let Value::Object(summary_map) = summary {
                if let Some(Value::String(test_type)) = summary_map.get("testType") {
                    assert!(test_type.contains("Chi-Square"));
                }
            }
        }
    }

    /// Test RED: Correlation matrix should calculate Pearson correlations by default
    #[test]
    fn test_pearson_correlation_matrix() {
        // Arrange - Two perfectly correlated variables
        let var1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var2 = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // var2 = 2 * var1
        let data_matrix = vec![var1, var2];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        
        // Act
        let result = correlation_matrix(&[data]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "CorrelationMatrix");
            
            let method = obj.foreign().call_method("method", &[]).unwrap();
            assert_eq!(method, Value::String("pearson".to_string()));
            
            let matrix = obj.foreign().call_method("matrix", &[]).unwrap();
            assert!(matches!(matrix, Value::List(_)));
            
            // Test specific correlation lookup
            let var1_name = Value::String("Var1".to_string());
            let var2_name = Value::String("Var2".to_string());
            let correlation = obj.foreign().call_method("getCorrelation", &[var1_name, var2_name]).unwrap();
            assert!(matches!(correlation, Value::Number(_)));
        }
    }

    /// Test RED: Spearman correlation should handle rank-based correlation
    #[test]
    fn test_spearman_correlation_matrix() {
        // Arrange - Non-linear but monotonic relationship
        let var1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let var2 = vec![1.0, 4.0, 9.0, 16.0, 25.0]; // var2 = var1^2
        let data_matrix = vec![var1, var2];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let method = Value::String("spearman".to_string());
        
        // Act
        let result = correlation_matrix(&[data, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            let method = obj.foreign().call_method("method", &[]).unwrap();
            assert_eq!(method, Value::String("spearman".to_string()));
        }
    }

    /// Test RED: PCA should perform dimensionality reduction
    #[test]
    fn test_pca_analysis() {
        // Arrange - Multi-dimensional data
        let data_matrix = vec![
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 4.0, 6.0, 8.0],
            vec![3.0, 6.0, 9.0, 12.0],
        ];
        
        let data = Value::List(
            data_matrix.iter()
                .map(|row| Value::List(row.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let n_components = Value::Number(2.0);
        let normalize = Value::Boolean(true);
        
        // Act
        let result = pca(&[data, n_components, normalize]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(pca_result) = result.unwrap() {
            assert!(pca_result.contains_key("components"));
            assert!(pca_result.contains_key("explainedVariance"));
            assert!(pca_result.contains_key("normalized"));
        }
    }

    /// Test RED: Bootstrap sampling should generate confidence intervals
    #[test]
    fn test_bootstrap_sampling() {
        // Arrange - Sample data
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let sample_size = Value::Number(5.0);
        let iterations = Value::Number(100.0);
        
        // Act
        let result = bootstrap_sample(&[data, sample_size, iterations]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "BootstrapResult");
            
            let original_statistic = obj.foreign().call_method("originalStatistic", &[]).unwrap();
            assert!(matches!(original_statistic, Value::Number(_)));
            
            let confidence_interval = obj.foreign().call_method("confidenceInterval", &[]).unwrap();
            assert!(matches!(confidence_interval, Value::List(_)));
            
            let bias = obj.foreign().call_method("bias", &[]).unwrap();
            assert!(matches!(bias, Value::Number(_)));
            
            let standard_error = obj.foreign().call_method("standardError", &[]).unwrap();
            assert!(matches!(standard_error, Value::Number(_)));
        }
    }

    /// Test RED: Statistical summary should provide comprehensive descriptive statistics
    #[test]
    fn test_statistical_summary() {
        // Arrange - Sample data with known statistics
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let quantiles = Value::List(vec![
            Value::Number(0.25),
            Value::Number(0.5),
            Value::Number(0.75),
        ]);
        
        // Act
        let result = statistical_summary(&[data, quantiles]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(summary) = result.unwrap() {
            assert!(summary.contains_key("count"));
            assert!(summary.contains_key("mean"));
            assert!(summary.contains_key("std"));
            assert!(summary.contains_key("min"));
            assert!(summary.contains_key("max"));
            assert!(summary.contains_key("quantiles"));
            
            // Verify specific values
            if let Some(Value::Number(count)) = summary.get("count") {
                assert_eq!(*count, 5.0);
            }
            if let Some(Value::Number(mean)) = summary.get("mean") {
                assert_eq!(*mean, 3.0);
            }
        }
    }

    /// Test RED: Outlier detection using IQR method
    #[test]
    fn test_outlier_detection_iqr() {
        // Arrange - Data with outliers
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100.0 is an outlier
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("iqr".to_string());
        let threshold = Value::Number(1.5);
        
        // Act
        let result = outlier_detection(&[data, method, threshold]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(outlier_result) = result.unwrap() {
            assert!(outlier_result.contains_key("outliers"));
            assert!(outlier_result.contains_key("indices"));
            assert!(outlier_result.contains_key("method"));
            assert!(outlier_result.contains_key("threshold"));
            
            if let Some(Value::String(method)) = outlier_result.get("method") {
                assert_eq!(method, "iqr");
            }
        }
    }

    /// Test RED: Z-score outlier detection
    #[test]
    fn test_outlier_detection_zscore() {
        // Arrange - Data with outliers
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 50.0]; // 50.0 is an outlier
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("zscore".to_string());
        let threshold = Value::Number(2.0);
        
        // Act
        let result = outlier_detection(&[data, method, threshold]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(outlier_result) = result.unwrap() {
            if let Some(Value::String(method)) = outlier_result.get("method") {
                assert_eq!(method, "zscore");
            }
        }
    }

    /// Test RED: Normality test using Shapiro-Wilk
    #[test]
    fn test_shapiro_wilk_normality_test() {
        // Arrange - Approximately normal data
        let sample_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let test_type = Value::String("shapiro".to_string());
        
        // Act
        let result = normality_test(&[data, test_type]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::LyObj(obj) = result.unwrap() {
            assert_eq!(obj.foreign().typename(), "HypothesisTestResult");
            
            let summary = obj.foreign().call_method("summary", &[]).unwrap();
            if let Value::Object(summary_map) = summary {
                if let Some(Value::String(test_type)) = summary_map.get("testType") {
                    assert!(test_type.contains("Shapiro-Wilk"));
                }
            }
        }
    }

    /// Test RED: Confidence interval calculation
    #[test]
    fn test_confidence_interval() {
        // Arrange - Sample data
        let sample_data = vec![10.0, 12.0, 14.0, 16.0, 18.0];
        let data = Value::List(sample_data.iter().map(|&x| Value::Number(x)).collect());
        let confidence_level = Value::Number(0.95);
        let method = Value::String("t".to_string());
        
        // Act
        let result = confidence_interval(&[data, confidence_level, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::List(ci) = result.unwrap() {
            assert_eq!(ci.len(), 2);
            assert!(matches!(ci[0], Value::Number(_)));
            assert!(matches!(ci[1], Value::Number(_)));
        }
    }

    /// Test RED: Effect size calculation (Cohen's d)
    #[test]
    fn test_effect_size_cohens_d() {
        // Arrange - Two groups with known effect size
        let group1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let group2 = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // Mean difference = 2.0
        
        let group1_val = Value::List(group1.iter().map(|&x| Value::Number(x)).collect());
        let group2_val = Value::List(group2.iter().map(|&x| Value::Number(x)).collect());
        let method = Value::String("cohens_d".to_string());
        
        // Act
        let result = effect_size(&[group1_val, group2_val, method]);
        
        // Assert
        assert!(result.is_ok());
        assert!(matches!(result.unwrap(), Value::Number(_)));
    }

    /// Test RED: Power analysis for sample size calculation
    #[test]
    fn test_power_analysis() {
        // Arrange - Standard power analysis parameters
        let effect_size = Value::Number(0.5); // Medium effect size
        let alpha = Value::Number(0.05);
        let power = Value::Number(0.8);
        let test_type = Value::String("two_sample_t".to_string());
        
        // Act
        let result = power_analysis(&[effect_size, alpha, power, test_type]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(power_result) = result.unwrap() {
            assert!(power_result.contains_key("effectSize"));
            assert!(power_result.contains_key("alpha"));
            assert!(power_result.contains_key("power"));
            assert!(power_result.contains_key("testType"));
            assert!(power_result.contains_key("sampleSize"));
        }
    }

    /// Test RED: Multiple comparison procedures
    #[test]
    fn test_multiple_comparison_tukey() {
        // Arrange - Multiple groups for comparison
        let groups = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        
        let groups_val = Value::List(
            groups.iter()
                .map(|group| Value::List(group.iter().map(|&x| Value::Number(x)).collect()))
                .collect()
        );
        let method = Value::String("tukey".to_string());
        
        // Act
        let result = multiple_comparison(&[groups_val, method]);
        
        // Assert
        assert!(result.is_ok());
        if let Value::Object(comparison_result) = result.unwrap() {
            assert!(comparison_result.contains_key("method"));
            assert!(comparison_result.contains_key("comparisons"));
            
            if let Some(Value::String(method)) = comparison_result.get("method") {
                assert_eq!(method, "tukey");
            }
            if let Some(Value::List(comparisons)) = comparison_result.get("comparisons") {
                assert!(!comparisons.is_empty());
            }
        }
    }

    /// Test RED: Error handling for invalid inputs
    #[test]
    fn test_regression_error_handling() {
        // Arrange - Invalid arguments (too few)
        let data = Value::List(vec![]);
        
        // Act
        let result = regression(&[data]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("requires at least 3 arguments"));
        }
    }

    /// Test RED: Chi-square test error handling for mismatched arrays
    #[test]
    fn test_chi_square_error_handling() {
        // Arrange - Mismatched observed and expected arrays
        let observed = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
        let expected = Value::List(vec![Value::Number(1.0)]); // Different length
        
        // Act
        let result = chi_square_test(&[observed, expected]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("same length"));
        }
    }

    /// Test RED: Correlation matrix error handling for invalid method
    #[test]
    fn test_correlation_matrix_invalid_method() {
        // Arrange
        let data = Value::List(vec![Value::List(vec![Value::Number(1.0)])]);
        let method = Value::String("invalid_method".to_string());
        
        // Act
        let result = correlation_matrix(&[data, method]);
        
        // Assert
        assert!(result.is_err());
        if let Err(VmError::RuntimeError(msg)) = result {
            assert!(msg.contains("Unsupported correlation method"));
        }
    }
}