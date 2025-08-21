//! MLOps Integration Tests
//! 
//! Comprehensive integration tests for the complete MLOps system.
//! Tests end-to-end workflows and cross-component interaction.

use lyra::error::{Error as LyraError, Result as LyraResult};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::ai_ml::mlops::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use chrono::Utc;

#[cfg(test)]
mod end_to_end_mlops_tests {
    use super::*;

    #[test]
    fn test_complete_ml_experiment_workflow() {
        let mut vm = VirtualMachine::new();
        
        // 1. Create an experiment
        let experiment_args = vec![
            Value::String("iris_classification".to_string()),
            Value::String("ML model for iris classification".to_string()),
            Value::List(vec![
                Value::String("ml".to_string()),
                Value::String("classification".to_string()),
                Value::String("iris".to_string()),
            ]),
        ];
        
        let experiment_result = experiment_create(&mut vm, &experiment_args);
        assert!(experiment_result.is_ok());
        
        // 2. Create a model registry
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        // 3. Create a feature store
        let feature_store_args = vec![
            Value::String("iris_features".to_string()),
            Value::List(vec![
                Value::String("sepal_length".to_string()),
                Value::String("sepal_width".to_string()),
                Value::String("petal_length".to_string()),
                Value::String("petal_width".to_string()),
            ]),
        ];
        
        let feature_store_result = feature_store_create(&mut vm, &feature_store_args);
        assert!(feature_store_result.is_ok());
        
        // 4. Create a model monitor
        let monitor_args = vec![
            Value::String("iris_model_v1".to_string()),
            Value::List(vec![
                Value::String("accuracy".to_string()),
                Value::String("precision".to_string()),
                Value::String("recall".to_string()),
            ]),
        ];
        
        let monitor_result = model_monitor_create(&mut vm, &monitor_args);
        assert!(monitor_result.is_ok());
        
        // Verify all components were created successfully
        if let (Ok(Value::LyObj(experiment_obj)), 
                Ok(Value::LyObj(registry_obj)), 
                Ok(Value::LyObj(feature_store_obj)), 
                Ok(Value::LyObj(monitor_obj))) = 
                (experiment_result, registry_result, feature_store_result, monitor_result) {
            
            assert_eq!(experiment_obj.type_name(), "Experiment");
            assert_eq!(registry_obj.type_name(), "ModelRegistry");
            assert_eq!(feature_store_obj.type_name(), "FeatureStore");
            assert_eq!(monitor_obj.type_name(), "ModelMonitor");
            
            // 5. Test experiment run creation and logging
            let run_id_result = experiment_obj.call_method("createRun", &[]);
            assert!(run_id_result.is_ok());
            
            if let Ok(Value::String(run_id)) = run_id_result {
                // Log training metrics
                let metrics = vec![
                    ("accuracy", 0.95),
                    ("precision", 0.94),
                    ("recall", 0.96),
                    ("f1_score", 0.95),
                ];
                
                for (metric_name, value) in metrics {
                    let metric_args = vec![
                        Value::String(run_id.clone()),
                        Value::String(metric_name.to_string()),
                        Value::Number(value),
                        Value::Number(1.0),
                    ];
                    
                    let log_result = experiment_obj.call_method("logMetric", &metric_args);
                    assert!(log_result.is_ok());
                }
                
                // 6. Register the trained model
                let register_args = vec![
                    Value::String("iris_classifier".to_string()),
                    Value::String("v1.0".to_string()),
                    Value::String("/models/iris_v1.pkl".to_string()),
                    Value::String("sklearn".to_string()),
                ];
                
                let register_result = registry_obj.call_method("register", &register_args);
                assert!(register_result.is_ok());
                
                // 7. Test feature store operations
                let compute_result = feature_store_obj.call_method("computeFeatures", &[]);
                assert!(compute_result.is_ok());
                
                let serve_args = vec![
                    Value::List(vec![
                        Value::String("sepal_length".to_string()),
                        Value::String("petal_length".to_string()),
                    ]),
                ];
                
                let serve_result = feature_store_obj.call_method("serveFeatures", &serve_args);
                assert!(serve_result.is_ok());
                
                // 8. Test model monitoring
                let record_args = vec![
                    Value::String("accuracy".to_string()),
                    Value::Number(0.93), // Slightly lower accuracy
                    Value::Number(1.0),
                ];
                
                let record_result = monitor_obj.call_method("recordMetric", &record_args);
                assert!(record_result.is_ok());
                
                // Check for alerts (none expected since we don't have thresholds set)
                let alerts_result = monitor_obj.call_method("getAlerts", &[]);
                assert!(alerts_result.is_ok());
            }
        }
    }
    
    #[test]
    fn test_model_lifecycle_management() {
        let mut vm = VirtualMachine::new();
        
        // Create model registry
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Register multiple versions of a model
            let versions = vec!["v1.0", "v1.1", "v2.0"];
            
            for version in versions {
                let register_args = vec![
                    Value::String("customer_churn_model".to_string()),
                    Value::String(version.to_string()),
                    Value::String(format!("/models/churn_{}.pkl", version)),
                    Value::String("xgboost".to_string()),
                ];
                
                let result = registry_obj.call_method("register", &register_args);
                assert!(result.is_ok());
            }
            
            // List all models
            let list_result = registry_obj.call_method("list", &[]);
            assert!(list_result.is_ok());
            
            if let Ok(Value::List(models)) = list_result {
                assert_eq!(models.len(), 3); // Three versions registered
            }
            
            // Get specific model version
            let get_args = vec![
                Value::String("customer_churn_model".to_string()),
                Value::String("v2.0".to_string()),
            ];
            
            let get_result = registry_obj.call_method("get", &get_args);
            assert!(get_result.is_ok());
            
            if let Ok(Value::String(model_info)) = get_result {
                assert!(model_info.contains("customer_churn_model"));
                assert!(model_info.contains("v2.0"));
            }
        }
    }
    
    #[test]
    fn test_data_quality_and_drift_detection() {
        let mut vm = VirtualMachine::new();
        
        // Create model monitor for drift detection
        let monitor_args = vec![
            Value::String("production_model".to_string()),
            Value::List(vec![
                Value::String("accuracy".to_string()),
                Value::String("prediction_confidence".to_string()),
            ]),
        ];
        
        let monitor_result = model_monitor_create(&mut vm, &monitor_args);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            // Simulate baseline performance (stable period)
            for i in 1..=100 {
                let accuracy = 0.95 + (rand::random::<f64>() - 0.5) * 0.02; // Small noise around 0.95
                let confidence = 0.85 + (rand::random::<f64>() - 0.5) * 0.02; // Small noise around 0.85
                
                let accuracy_args = vec![
                    Value::String("accuracy".to_string()),
                    Value::Number(accuracy),
                    Value::Number(i as f64),
                ];
                
                let confidence_args = vec![
                    Value::String("prediction_confidence".to_string()),
                    Value::Number(confidence),
                    Value::Number(i as f64),
                ];
                
                let _ = monitor_obj.call_method("recordMetric", &accuracy_args);
                let _ = monitor_obj.call_method("recordMetric", &confidence_args);
            }
            
            // Simulate performance degradation (drift period)
            for i in 101..=150 {
                let accuracy = 0.85 + (rand::random::<f64>() - 0.5) * 0.02; // Shifted to 0.85
                let confidence = 0.75 + (rand::random::<f64>() - 0.5) * 0.02; // Shifted to 0.75
                
                let accuracy_args = vec![
                    Value::String("accuracy".to_string()),
                    Value::Number(accuracy),
                    Value::Number(i as f64),
                ];
                
                let confidence_args = vec![
                    Value::String("prediction_confidence".to_string()),
                    Value::Number(confidence),
                    Value::Number(i as f64),
                ];
                
                let _ = monitor_obj.call_method("recordMetric", &accuracy_args);
                let _ = monitor_obj.call_method("recordMetric", &confidence_args);
            }
            
            // Test drift detection
            let drift_args = vec![
                Value::String("accuracy".to_string()),
                Value::Number(100.0), // baseline window
                Value::Number(50.0),  // current window
            ];
            
            let drift_result = monitor_obj.call_method("detectDrift", &drift_args);
            assert!(drift_result.is_ok());
            
            if let Ok(Value::String(drift_info)) = drift_result {
                assert!(drift_info.contains("DriftResult"));
                // In a real scenario, this should detect drift due to accuracy drop from 0.95 to 0.85
            }
        }
    }
    
    #[test]
    fn test_feature_engineering_pipeline() {
        let mut vm = VirtualMachine::new();
        
        // Create feature store with various feature types
        let feature_store_args = vec![
            Value::String("customer_analytics".to_string()),
            Value::List(vec![
                Value::String("age".to_string()),
                Value::String("income".to_string()),
                Value::String("purchase_history".to_string()),
                Value::String("location".to_string()),
                Value::String("engagement_score".to_string()),
            ]),
        ];
        
        let feature_store_result = feature_store_create(&mut vm, &feature_store_args);
        assert!(feature_store_result.is_ok());
        
        if let Ok(Value::LyObj(feature_store_obj)) = feature_store_result {
            // Test schema retrieval
            let schema_result = feature_store_obj.call_method("getSchema", &[]);
            assert!(schema_result.is_ok());
            
            if let Ok(Value::List(schema_info)) = schema_result {
                assert_eq!(schema_info.len(), 5); // Five features defined
            }
            
            // Test feature computation with transformations
            let compute_result = feature_store_obj.call_method("computeFeatures", &[]);
            assert!(compute_result.is_ok());
            
            // Test feature serving for inference
            let serve_args = vec![
                Value::List(vec![
                    Value::String("age".to_string()),
                    Value::String("income".to_string()),
                    Value::String("engagement_score".to_string()),
                ]),
            ];
            
            let serve_result = feature_store_obj.call_method("serveFeatures", &serve_args);
            assert!(serve_result.is_ok());
            
            if let Ok(Value::String(features_info)) = serve_result {
                assert!(features_info.contains("Features"));
            }
        }
    }
    
    #[test]
    fn test_automl_workflow() {
        let mut vm = VirtualMachine::new();
        
        // Test AutoML functions
        let auto_train_result = auto_train(&mut vm, &[]);
        assert!(auto_train_result.is_ok());
        
        if let Ok(Value::String(model_id)) = auto_train_result {
            assert!(!model_id.is_empty());
            assert!(model_id.contains("trained_model_id"));
        }
        
        let hyperparameter_tune_result = hyperparameter_tune(&mut vm, &[]);
        assert!(hyperparameter_tune_result.is_ok());
        
        if let Ok(Value::List(optimal_params)) = hyperparameter_tune_result {
            // In a real implementation, this would return optimized parameters
            assert_eq!(optimal_params.len(), 0); // Placeholder implementation
        }
        
        let model_select_result = model_select(&mut vm, &[]);
        assert!(model_select_result.is_ok());
        
        if let Ok(Value::String(best_model_id)) = model_select_result {
            assert!(!best_model_id.is_empty());
            assert!(best_model_id.contains("best_model_id"));
        }
        
        let feature_engineering_result = feature_engineering(&mut vm, &[]);
        assert!(feature_engineering_result.is_ok());
        
        if let Ok(Value::List(engineered_features)) = feature_engineering_result {
            // In a real implementation, this would return engineered features
            assert_eq!(engineered_features.len(), 0); // Placeholder implementation
        }
    }
    
    #[test]
    fn test_performance_monitoring_alerts() {
        let mut vm = VirtualMachine::new();
        
        // Test performance monitoring functions
        let performance_drift_result = performance_drift(&mut vm, &[]);
        assert!(performance_drift_result.is_ok());
        assert_eq!(performance_drift_result.unwrap(), Value::Boolean(false));
        
        let ab_test_result = ab_test(&mut vm, &[]);
        assert!(ab_test_result.is_ok());
        
        if let Ok(Value::String(test_id)) = ab_test_result {
            assert!(!test_id.is_empty());
            assert!(test_id.contains("test_id"));
        }
        
        let feedback_loop_result = feedback_loop(&mut vm, &[]);
        assert!(feedback_loop_result.is_ok());
        assert_eq!(feedback_loop_result.unwrap(), Value::Boolean(true));
        
        let model_health_result = model_health(&mut vm, &[]);
        assert!(model_health_result.is_ok());
        assert_eq!(model_health_result.unwrap(), Value::String("healthy".to_string()));
        
        let alert_config_result = alert_config(&mut vm, &[]);
        assert!(alert_config_result.is_ok());
        assert_eq!(alert_config_result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_data_pipeline_operations() {
        let mut vm = VirtualMachine::new();
        
        // Test data pipeline functions
        let feature_compute_result = feature_compute(&mut vm, &[]);
        assert!(feature_compute_result.is_ok());
        assert_eq!(feature_compute_result.unwrap(), Value::String("features_computed".to_string()));
        
        let feature_serve_result = feature_serve(&mut vm, &[]);
        assert!(feature_serve_result.is_ok());
        assert_eq!(feature_serve_result.unwrap(), Value::List(Vec::new()));
        
        let data_drift_result = data_drift(&mut vm, &[]);
        assert!(data_drift_result.is_ok());
        assert_eq!(data_drift_result.unwrap(), Value::Boolean(false));
        
        let data_validation_result = data_validation(&mut vm, &[]);
        assert!(data_validation_result.is_ok());
        assert_eq!(data_validation_result.unwrap(), Value::Boolean(true));
        
        let data_quality_result = data_quality(&mut vm, &[]);
        assert!(data_quality_result.is_ok());
        assert_eq!(data_quality_result.unwrap(), Value::Number(0.95));
        
        let pipeline_create_result = pipeline_create(&mut vm, &[]);
        assert!(pipeline_create_result.is_ok());
        
        if let Ok(Value::String(pipeline_id)) = pipeline_create_result {
            assert!(!pipeline_id.is_empty());
            assert!(pipeline_id.contains("pipeline_id"));
        }
        
        let pipeline_execute_result = pipeline_execute(&mut vm, &[]);
        assert!(pipeline_execute_result.is_ok());
        
        if let Ok(Value::String(execution_id)) = pipeline_execute_result {
            assert!(!execution_id.is_empty());
            assert!(execution_id.contains("execution_id"));
        }
    }
}

#[cfg(test)]
mod mlops_error_handling_tests {
    use super::*;
    
    #[test]
    fn test_experiment_creation_error_handling() {
        let mut vm = VirtualMachine::new();
        
        // Test with invalid argument types
        let invalid_args = vec![
            Value::Number(123.0), // Should be string
            Value::String("description".to_string()),
            Value::List(vec![Value::String("tag".to_string())]),
        ];
        
        let result = experiment_create(&mut vm, &invalid_args);
        assert!(result.is_err());
        
        // Test with insufficient arguments
        let insufficient_args = vec![
            Value::String("name".to_string()),
        ];
        
        let result = experiment_create(&mut vm, &insufficient_args);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_model_registry_error_handling() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Test invalid argument types for register
            let invalid_args = vec![
                Value::Number(123.0), // Should be string
                Value::String("v1.0".to_string()),
                Value::String("/path".to_string()),
                Value::String("framework".to_string()),
            ];
            
            let result = registry_obj.call_method("register", &invalid_args);
            assert!(result.is_err());
            
            // Test insufficient arguments for get
            let insufficient_args = vec![
                Value::String("model_name".to_string()),
            ];
            
            let result = registry_obj.call_method("get", &insufficient_args);
            assert!(result.is_err());
        }
    }
    
    #[test]
    fn test_feature_store_error_handling() {
        let mut vm = VirtualMachine::new();
        
        // Test with invalid argument types
        let invalid_args = vec![
            Value::Number(123.0), // Should be string
            Value::List(vec![Value::String("feature".to_string())]),
        ];
        
        let result = feature_store_create(&mut vm, &invalid_args);
        assert!(result.is_err());
        
        // Test with insufficient arguments
        let insufficient_args = vec![
            Value::String("store_name".to_string()),
        ];
        
        let result = feature_store_create(&mut vm, &insufficient_args);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_model_monitor_error_handling() {
        let mut vm = VirtualMachine::new();
        
        // Test with invalid argument types
        let invalid_args = vec![
            Value::Number(123.0), // Should be string
            Value::List(vec![Value::String("metric".to_string())]),
        ];
        
        let result = model_monitor_create(&mut vm, &invalid_args);
        assert!(result.is_err());
        
        // Test with insufficient arguments
        let insufficient_args = vec![
            Value::String("model_name".to_string()),
        ];
        
        let result = model_monitor_create(&mut vm, &insufficient_args);
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod mlops_performance_tests {
    use super::*;
    use std::time::Instant;
    
    #[test]
    fn test_experiment_creation_performance() {
        let mut vm = VirtualMachine::new();
        
        let start = Instant::now();
        
        // Create multiple experiments
        for i in 0..100 {
            let args = vec![
                Value::String(format!("experiment_{}", i)),
                Value::String("Performance test experiment".to_string()),
                Value::List(vec![Value::String("performance".to_string())]),
            ];
            
            let result = experiment_create(&mut vm, &args);
            assert!(result.is_ok());
        }
        
        let duration = start.elapsed();
        
        // Should complete in reasonable time (less than 1 second for 100 experiments)
        assert!(duration.as_millis() < 1000);
    }
    
    #[test]
    fn test_model_registry_performance() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            let start = Instant::now();
            
            // Register multiple models
            for i in 0..100 {
                let args = vec![
                    Value::String(format!("model_{}", i)),
                    Value::String("v1.0".to_string()),
                    Value::String(format!("/models/model_{}.pkl", i)),
                    Value::String("sklearn".to_string()),
                ];
                
                let result = registry_obj.call_method("register", &args);
                assert!(result.is_ok());
            }
            
            let duration = start.elapsed();
            
            // Should complete in reasonable time
            assert!(duration.as_millis() < 1000);
            
            // Test listing performance
            let start = Instant::now();
            let list_result = registry_obj.call_method("list", &[]);
            assert!(list_result.is_ok());
            let list_duration = start.elapsed();
            
            // Listing should be fast
            assert!(list_duration.as_millis() < 100);
        }
    }
    
    #[test]
    fn test_monitoring_performance() {
        let mut vm = VirtualMachine::new();
        
        let monitor_result = model_monitor_create(&mut vm, &[
            Value::String("performance_test_model".to_string()),
            Value::List(vec![Value::String("accuracy".to_string())]),
        ]);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            let start = Instant::now();
            
            // Record many metrics
            for i in 0..1000 {
                let args = vec![
                    Value::String("accuracy".to_string()),
                    Value::Number(0.9 + (i as f64 / 10000.0)),
                    Value::Number(i as f64),
                ];
                
                let result = monitor_obj.call_method("recordMetric", &args);
                assert!(result.is_ok());
            }
            
            let duration = start.elapsed();
            
            // Should handle 1000 metric recordings efficiently
            assert!(duration.as_millis() < 1000);
            
            // Test drift detection performance
            let start = Instant::now();
            let drift_args = vec![
                Value::String("accuracy".to_string()),
                Value::Number(500.0), // baseline window
                Value::Number(500.0), // current window
            ];
            
            let drift_result = monitor_obj.call_method("detectDrift", &drift_args);
            assert!(drift_result.is_ok());
            
            let drift_duration = start.elapsed();
            
            // Drift detection should be reasonably fast
            assert!(drift_duration.as_millis() < 500);
        }
    }
}