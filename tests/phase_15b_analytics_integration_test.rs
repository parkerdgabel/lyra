//! Phase 15B: Advanced Analytics & Statistics Integration Test
//!
//! Comprehensive integration test validating the complete analytics system
//! including statistical analysis, time series analysis, business intelligence,
//! and data mining capabilities.

use lyra::vm::{Value, VmError};
use lyra::foreign::LyObj;
use lyra::stdlib::StandardLibrary;
use std::collections::HashMap;

#[cfg(test)]
mod analytics_integration_tests {
    use super::*;

    /// Test comprehensive integration of all analytics modules
    #[test]
    fn test_analytics_module_integration() {
        let stdlib = StandardLibrary::new();
        
        // Verify all 45+ analytics functions are registered
        let function_names = stdlib.function_names();
        
        // Statistical Analysis Functions (15 functions)
        assert!(stdlib.get_function("Regression").is_some());
        assert!(stdlib.get_function("ANOVA").is_some());
        assert!(stdlib.get_function("TTest").is_some());
        assert!(stdlib.get_function("ChiSquareTest").is_some());
        assert!(stdlib.get_function("CorrelationMatrix").is_some());
        assert!(stdlib.get_function("PCA").is_some());
        assert!(stdlib.get_function("HypothesisTest").is_some());
        assert!(stdlib.get_function("ConfidenceInterval").is_some());
        assert!(stdlib.get_function("BootstrapSample").is_some());
        assert!(stdlib.get_function("StatisticalSummary").is_some());
        assert!(stdlib.get_function("OutlierDetection").is_some());
        assert!(stdlib.get_function("NormalityTest").is_some());
        assert!(stdlib.get_function("PowerAnalysis").is_some());
        assert!(stdlib.get_function("EffectSize").is_some());
        assert!(stdlib.get_function("MultipleComparison").is_some());
        
        // Time Series Analysis Functions (12 functions)
        assert!(stdlib.get_function("TimeSeriesDecompose").is_some());
        assert!(stdlib.get_function("AutoCorrelation").is_some());
        assert!(stdlib.get_function("PartialAutoCorrelation").is_some());
        assert!(stdlib.get_function("ARIMA").is_some());
        assert!(stdlib.get_function("Forecast").is_some());
        assert!(stdlib.get_function("SeasonalDecompose").is_some());
        assert!(stdlib.get_function("TrendAnalysis").is_some());
        assert!(stdlib.get_function("ChangePointDetection").is_some());
        assert!(stdlib.get_function("AnomalyDetection").is_some());
        assert!(stdlib.get_function("StationarityTest").is_some());
        assert!(stdlib.get_function("CrossCorrelation").is_some());
        assert!(stdlib.get_function("SpectralDensity").is_some());
        
        // Business Intelligence Functions (10 functions)
        assert!(stdlib.get_function("KPI").is_some());
        assert!(stdlib.get_function("CohortAnalysis").is_some());
        assert!(stdlib.get_function("FunnelAnalysis").is_some());
        assert!(stdlib.get_function("RetentionAnalysis").is_some());
        assert!(stdlib.get_function("LTV").is_some());
        assert!(stdlib.get_function("Churn").is_some());
        assert!(stdlib.get_function("Segmentation").is_some());
        assert!(stdlib.get_function("ABTestAnalysis").is_some());
        assert!(stdlib.get_function("AttributionModel").is_some());
        assert!(stdlib.get_function("Dashboard").is_some());
        
        // Data Mining Functions (8 functions)
        assert!(stdlib.get_function("Clustering").is_some());
        assert!(stdlib.get_function("Classification").is_some());
        assert!(stdlib.get_function("AssociationRules").is_some());
        assert!(stdlib.get_function("DecisionTree").is_some());
        assert!(stdlib.get_function("RandomForest").is_some());
        assert!(stdlib.get_function("SVM").is_some());
        assert!(stdlib.get_function("NeuralNetwork").is_some());
        assert!(stdlib.get_function("EnsembleMethod").is_some());
        
        println!("✅ All 45 analytics functions successfully registered");
    }

    /// Test statistical analysis workflow
    #[test]
    fn test_statistical_analysis_workflow() {
        let stdlib = StandardLibrary::new();
        
        // Test regression analysis
        if let Some(regression_fn) = stdlib.get_function("Regression") {
            let data = create_sample_regression_data();
            let formula = Value::String("y ~ x".to_string());
            let regression_type = Value::String("linear".to_string());
            
            let result = regression_fn(&[data, formula, regression_type]);
            assert!(result.is_ok(), "Regression should execute successfully");
            
            if let Ok(Value::LyObj(model)) = result {
                assert_eq!(model.foreign().typename(), "StatisticalModel");
                
                // Test model methods
                let coefficients = model.foreign().call_method("coefficients", &[]);
                assert!(coefficients.is_ok());
                
                let r_squared = model.foreign().call_method("rSquared", &[]);
                assert!(r_squared.is_ok());
            }
        }
        
        // Test correlation matrix
        if let Some(correlation_fn) = stdlib.get_function("CorrelationMatrix") {
            let data = create_sample_correlation_data();
            let method = Value::String("pearson".to_string());
            
            let result = correlation_fn(&[data, method]);
            assert!(result.is_ok(), "CorrelationMatrix should execute successfully");
            
            if let Ok(Value::LyObj(matrix)) = result {
                assert_eq!(matrix.foreign().typename(), "CorrelationMatrix");
                
                let matrix_data = matrix.foreign().call_method("matrix", &[]);
                assert!(matrix_data.is_ok());
            }
        }
        
        println!("✅ Statistical analysis workflow validated");
    }

    /// Test time series analysis workflow
    #[test]
    fn test_timeseries_analysis_workflow() {
        let stdlib = StandardLibrary::new();
        
        // Test time series decomposition
        if let Some(decompose_fn) = stdlib.get_function("TimeSeriesDecompose") {
            let series = create_sample_timeseries_data();
            let model = Value::String("additive".to_string());
            let period = Value::Number(4.0);
            
            let result = decompose_fn(&[series, model, period]);
            assert!(result.is_ok(), "TimeSeriesDecompose should execute successfully");
            
            if let Ok(Value::LyObj(decomposition)) = result {
                assert_eq!(decomposition.foreign().typename(), "TimeSeriesDecomposition");
                
                let trend = decomposition.foreign().call_method("trend", &[]);
                assert!(trend.is_ok());
                
                let seasonal = decomposition.foreign().call_method("seasonal", &[]);
                assert!(seasonal.is_ok());
                
                let residual = decomposition.foreign().call_method("residual", &[]);
                assert!(residual.is_ok());
            }
        }
        
        // Test ARIMA modeling
        if let Some(arima_fn) = stdlib.get_function("ARIMA") {
            let series = create_sample_timeseries_data();
            let order = Value::List(vec![
                Value::Number(1.0),
                Value::Number(1.0),
                Value::Number(1.0),
            ]);
            
            let result = arima_fn(&[series, order]);
            assert!(result.is_ok(), "ARIMA should execute successfully");
            
            if let Ok(Value::LyObj(model)) = result {
                assert_eq!(model.foreign().typename(), "ARIMAModel");
                
                let coefficients = model.foreign().call_method("coefficients", &[]);
                assert!(coefficients.is_ok());
                
                let aic = model.foreign().call_method("aic", &[]);
                assert!(aic.is_ok());
            }
        }
        
        println!("✅ Time series analysis workflow validated");
    }

    /// Test business intelligence workflow
    #[test]
    fn test_business_intelligence_workflow() {
        let stdlib = StandardLibrary::new();
        
        // Test KPI calculation
        if let Some(kpi_fn) = stdlib.get_function("KPI") {
            let data = create_sample_kpi_data();
            let metric = Value::String("conversion_rate".to_string());
            let target = Value::Number(5.0);
            let period = Value::String("monthly".to_string());
            
            let result = kpi_fn(&[data, metric, target, period]);
            assert!(result.is_ok(), "KPI should execute successfully");
            
            if let Ok(Value::LyObj(kpi)) = result {
                assert_eq!(kpi.foreign().typename(), "KPI");
                
                let value = kpi.foreign().call_method("value", &[]);
                assert!(value.is_ok());
                
                let status = kpi.foreign().call_method("status", &[]);
                assert!(status.is_ok());
            }
        }
        
        // Test A/B test analysis
        if let Some(ab_test_fn) = stdlib.get_function("ABTestAnalysis") {
            let control = Value::List(vec![
                Value::Number(0.05), Value::Number(0.04), Value::Number(0.06),
                Value::Number(0.05), Value::Number(0.04),
            ]);
            let treatment = Value::List(vec![
                Value::Number(0.07), Value::Number(0.08), Value::Number(0.06),
                Value::Number(0.09), Value::Number(0.07),
            ]);
            let metric = Value::String("conversion_rate".to_string());
            let alpha = Value::Number(0.05);
            
            let result = ab_test_fn(&[control, treatment, metric, alpha]);
            assert!(result.is_ok(), "ABTestAnalysis should execute successfully");
            
            if let Ok(Value::LyObj(ab_result)) = result {
                assert_eq!(ab_result.foreign().typename(), "ABTestResult");
                
                let lift = ab_result.foreign().call_method("lift", &[]);
                assert!(lift.is_ok());
                
                let p_value = ab_result.foreign().call_method("pValue", &[]);
                assert!(p_value.is_ok());
            }
        }
        
        println!("✅ Business intelligence workflow validated");
    }

    /// Test data mining workflow
    #[test]
    fn test_data_mining_workflow() {
        let stdlib = StandardLibrary::new();
        
        // Test clustering
        if let Some(clustering_fn) = stdlib.get_function("Clustering") {
            let data = create_sample_clustering_data();
            let algorithm = Value::String("kmeans".to_string());
            let k = Value::Number(3.0);
            
            let result = clustering_fn(&[data, algorithm, k]);
            assert!(result.is_ok(), "Clustering should execute successfully");
            
            if let Ok(Value::LyObj(clusters)) = result {
                assert_eq!(clusters.foreign().typename(), "ClusteringResult");
                
                let labels = clusters.foreign().call_method("labels", &[]);
                assert!(labels.is_ok());
                
                let centroids = clusters.foreign().call_method("centroids", &[]);
                assert!(centroids.is_ok());
                
                let silhouette = clusters.foreign().call_method("silhouetteScore", &[]);
                assert!(silhouette.is_ok());
            }
        }
        
        // Test association rules
        if let Some(assoc_fn) = stdlib.get_function("AssociationRules") {
            let transactions = create_sample_transaction_data();
            let min_support = Value::Number(0.1);
            let min_confidence = Value::Number(0.5);
            
            let result = assoc_fn(&[transactions, min_support, min_confidence]);
            assert!(result.is_ok(), "AssociationRules should execute successfully");
            
            if let Ok(Value::LyObj(rules)) = result {
                assert_eq!(rules.foreign().typename(), "AssociationRulesResult");
                
                let rules_list = rules.foreign().call_method("rules", &[]);
                assert!(rules_list.is_ok());
                
                let top_rules = rules.foreign().call_method("topRules", &[Value::Number(5.0)]);
                assert!(top_rules.is_ok());
            }
        }
        
        println!("✅ Data mining workflow validated");
    }

    /// Test end-to-end analytics pipeline
    #[test]
    fn test_end_to_end_analytics_pipeline() {
        let stdlib = StandardLibrary::new();
        
        // Simulate a complete analytics workflow
        println!("🚀 Running end-to-end analytics pipeline...");
        
        // 1. Data preparation and statistical summary
        if let Some(summary_fn) = stdlib.get_function("StatisticalSummary") {
            let data = create_sample_dataset();
            let quantiles = Value::List(vec![
                Value::Number(0.25), Value::Number(0.5), Value::Number(0.75)
            ]);
            
            let result = summary_fn(&[data, quantiles]);
            assert!(result.is_ok(), "Statistical summary should work");
            println!("  ✅ Step 1: Data summary completed");
        }
        
        // 2. Outlier detection
        if let Some(outlier_fn) = stdlib.get_function("OutlierDetection") {
            let data = create_sample_dataset();
            let method = Value::String("zscore".to_string());
            let threshold = Value::Number(2.0);
            
            let result = outlier_fn(&[data, method, threshold]);
            assert!(result.is_ok(), "Outlier detection should work");
            println!("  ✅ Step 2: Outlier detection completed");
        }
        
        // 3. Time series analysis
        if let Some(decompose_fn) = stdlib.get_function("TimeSeriesDecompose") {
            let series = create_sample_timeseries_data();
            let model = Value::String("additive".to_string());
            let period = Value::Number(4.0);
            
            let result = decompose_fn(&[series, model, period]);
            assert!(result.is_ok(), "Time series decomposition should work");
            println!("  ✅ Step 3: Time series analysis completed");
        }
        
        // 4. Customer segmentation
        if let Some(segment_fn) = stdlib.get_function("Segmentation") {
            let data = create_sample_customer_data();
            let features = Value::List(vec![
                Value::String("recency".to_string()),
                Value::String("frequency".to_string()),
                Value::String("monetary".to_string()),
            ]);
            let method = Value::String("kmeans".to_string());
            let k = Value::Number(4.0);
            
            let result = segment_fn(&[data, features, method, k]);
            assert!(result.is_ok(), "Customer segmentation should work");
            println!("  ✅ Step 4: Customer segmentation completed");
        }
        
        // 5. Dashboard creation
        if let Some(dashboard_fn) = stdlib.get_function("Dashboard") {
            let metrics = Value::List(vec![
                Value::String("revenue".to_string()),
                Value::String("conversion_rate".to_string()),
                Value::String("customer_acquisition_cost".to_string()),
            ]);
            
            let result = dashboard_fn(&[metrics]);
            assert!(result.is_ok(), "Dashboard creation should work");
            println!("  ✅ Step 5: Dashboard creation completed");
        }
        
        println!("🎉 End-to-end analytics pipeline completed successfully!");
    }

    /// Test error handling across analytics modules
    #[test]
    fn test_analytics_error_handling() {
        let stdlib = StandardLibrary::new();
        
        // Test invalid arguments for statistical functions
        if let Some(regression_fn) = stdlib.get_function("Regression") {
            let result = regression_fn(&[Value::String("invalid".to_string())]);
            assert!(result.is_err(), "Should handle invalid arguments gracefully");
        }
        
        // Test invalid correlation method
        if let Some(correlation_fn) = stdlib.get_function("CorrelationMatrix") {
            let data = create_sample_correlation_data();
            let method = Value::String("invalid_method".to_string());
            
            let result = correlation_fn(&[data, method]);
            assert!(result.is_err(), "Should reject invalid correlation methods");
        }
        
        // Test mismatched data for cross-correlation
        if let Some(cross_corr_fn) = stdlib.get_function("CrossCorrelation") {
            let series1 = Value::List(vec![Value::Number(1.0), Value::Number(2.0)]);
            let series2 = Value::List(vec![Value::Number(1.0)]); // Different length
            let lags = Value::Number(1.0);
            
            let result = cross_corr_fn(&[series1, series2, lags]);
            assert!(result.is_err(), "Should reject mismatched series lengths");
        }
        
        println!("✅ Error handling validation completed");
    }

    // Helper functions to create sample data
    fn create_sample_regression_data() -> Value {
        Value::List(vec![
            Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)]),
            Value::List(vec![Value::Number(2.0), Value::Number(4.0), Value::Number(6.0)]),
        ])
    }

    fn create_sample_correlation_data() -> Value {
        Value::List(vec![
            Value::List(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)]),
            Value::List(vec![Value::Number(2.0), Value::Number(4.0), Value::Number(6.0)]),
        ])
    }

    fn create_sample_timeseries_data() -> Value {
        Value::List(vec![
            Value::Number(10.0), Value::Number(15.0), Value::Number(20.0), Value::Number(15.0),
            Value::Number(12.0), Value::Number(17.0), Value::Number(22.0), Value::Number(17.0),
        ])
    }

    fn create_sample_kpi_data() -> Value {
        let mut data = HashMap::new();
        data.insert("conversions".to_string(), Value::List(vec![
            Value::Number(50.0), Value::Number(55.0), Value::Number(60.0)
        ]));
        data.insert("visitors".to_string(), Value::List(vec![
            Value::Number(1000.0), Value::Number(1100.0), Value::Number(1200.0)
        ]));
        Value::Object(data)
    }

    fn create_sample_clustering_data() -> Value {
        Value::List(vec![
            Value::List(vec![Value::Number(1.0), Value::Number(2.0)]),
            Value::List(vec![Value::Number(2.0), Value::Number(3.0)]),
            Value::List(vec![Value::Number(8.0), Value::Number(9.0)]),
            Value::List(vec![Value::Number(9.0), Value::Number(10.0)]),
        ])
    }

    fn create_sample_transaction_data() -> Value {
        Value::List(vec![
            Value::List(vec![
                Value::String("bread".to_string()),
                Value::String("butter".to_string()),
                Value::String("milk".to_string()),
            ]),
            Value::List(vec![
                Value::String("bread".to_string()),
                Value::String("butter".to_string()),
            ]),
            Value::List(vec![
                Value::String("milk".to_string()),
                Value::String("cookies".to_string()),
            ]),
        ])
    }

    fn create_sample_dataset() -> Value {
        Value::List(vec![
            Value::Number(1.0), Value::Number(2.0), Value::Number(3.0),
            Value::Number(4.0), Value::Number(5.0), Value::Number(6.0),
            Value::Number(7.0), Value::Number(8.0), Value::Number(9.0),
            Value::Number(10.0),
        ])
    }

    fn create_sample_customer_data() -> Value {
        Value::List(vec![
            create_customer_record(30, 5, 250.0),
            create_customer_record(15, 8, 480.0),
            create_customer_record(60, 2, 120.0),
            create_customer_record(10, 12, 720.0),
        ])
    }

    fn create_customer_record(recency: i32, frequency: i32, monetary: f64) -> Value {
        let mut record = HashMap::new();
        record.insert("recency".to_string(), Value::Number(recency as f64));
        record.insert("frequency".to_string(), Value::Number(frequency as f64));
        record.insert("monetary".to_string(), Value::Number(monetary));
        Value::Object(record)
    }
}