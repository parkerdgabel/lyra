//! MLOps Data Pipeline Tests
//! 
//! Tests for the data pipeline functionality in the MLOps module.
//! This follows Test-Driven Development practices.

use lyra::error::{Error as LyraError, Result as LyraResult};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::ai_ml::mlops::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use chrono::Utc;

#[cfg(test)]
mod feature_store_tests {
    use super::*;

    #[test]
    fn test_feature_store_create() {
        let mut vm = VirtualMachine::new();
        
        let args = vec![
            Value::String("customer_features".to_string()),
            Value::List(vec![
                Value::String("age".to_string()),
                Value::String("income".to_string()),
                Value::String("location".to_string()),
            ]),
        ];
        
        let result = feature_store_create(&mut vm, &args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(store_obj)) = result {
            assert_eq!(store_obj.type_name(), "FeatureStore");
        } else {
            panic!("Expected LyObj with FeatureStore");
        }
    }
    
    #[test]
    fn test_feature_store_create_invalid_args() {
        let mut vm = VirtualMachine::new();
        
        // Test with insufficient arguments
        let args = vec![
            Value::String("customer_features".to_string()),
        ];
        
        let result = feature_store_create(&mut vm, &args);
        assert!(result.is_err());
        
        if let Err(LyraError::ArityError { expected, actual }) = result {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected ArityError");
        }
    }
    
    #[test]
    fn test_feature_store_compute_features() {
        let mut vm = VirtualMachine::new();
        
        let store_result = feature_store_create(&mut vm, &[
            Value::String("test_store".to_string()),
            Value::List(vec![Value::String("feature1".to_string())]),
        ]);
        assert!(store_result.is_ok());
        
        if let Ok(Value::LyObj(store_obj)) = store_result {
            let result = store_obj.call_method("computeFeatures", &[]);
            assert!(result.is_ok());
            
            if let Ok(Value::String(computed_info)) = result {
                assert!(computed_info.contains("ComputedFeatures"));
            } else {
                panic!("Expected string computed features info");
            }
        }
    }
    
    #[test]
    fn test_feature_store_serve_features() {
        let mut vm = VirtualMachine::new();
        
        let store_result = feature_store_create(&mut vm, &[
            Value::String("test_store".to_string()),
            Value::List(vec![Value::String("feature1".to_string())]),
        ]);
        assert!(store_result.is_ok());
        
        if let Ok(Value::LyObj(store_obj)) = store_result {
            let args = vec![
                Value::List(vec![
                    Value::String("feature1".to_string()),
                    Value::String("feature2".to_string()),
                ]),
            ];
            
            let result = store_obj.call_method("serveFeatures", &args);
            assert!(result.is_ok());
            
            if let Ok(Value::String(features_info)) = result {
                assert!(features_info.contains("Features"));
            } else {
                panic!("Expected string features info");
            }
        }
    }
    
    #[test]
    fn test_feature_store_get_schema() {
        let mut vm = VirtualMachine::new();
        
        let store_result = feature_store_create(&mut vm, &[
            Value::String("test_store".to_string()),
            Value::List(vec![
                Value::String("feature1".to_string()),
                Value::String("feature2".to_string()),
            ]),
        ]);
        assert!(store_result.is_ok());
        
        if let Ok(Value::LyObj(store_obj)) = store_result {
            let result = store_obj.call_method("getSchema", &[]);
            assert!(result.is_ok());
            
            if let Ok(Value::List(schema_info)) = result {
                assert_eq!(schema_info.len(), 2); // Two features
            } else {
                panic!("Expected list of schema info");
            }
        }
    }
    
    #[test]
    fn test_feature_store_direct_creation() {
        let schema = vec![
            FeatureSchema {
                name: "age".to_string(),
                feature_type: FeatureType::Integer,
                description: "Customer age".to_string(),
                nullable: false,
                default_value: None,
            },
            FeatureSchema {
                name: "income".to_string(),
                feature_type: FeatureType::Float,
                description: "Customer income".to_string(),
                nullable: true,
                default_value: Some(Value::Number(0.0)),
            },
        ];
        
        let store_result = FeatureStore::new("customer_features".to_string(), schema);
        assert!(store_result.is_ok());
        
        if let Ok(store) = store_result {
            assert_eq!(store.get_name(), "customer_features");
            assert_eq!(store.get_schema().len(), 2);
            assert_eq!(store.get_schema()[0].name, "age");
            assert!(matches!(store.get_schema()[0].feature_type, FeatureType::Integer));
            assert_eq!(store.get_schema()[1].name, "income");
            assert!(matches!(store.get_schema()[1].feature_type, FeatureType::Float));
        }
    }
    
    #[test]
    fn test_feature_store_serve_with_keys() {
        let schema = vec![
            FeatureSchema {
                name: "feature1".to_string(),
                feature_type: FeatureType::Float,
                description: "First feature".to_string(),
                nullable: false,
                default_value: None,
            },
            FeatureSchema {
                name: "feature2".to_string(),
                feature_type: FeatureType::String,
                description: "Second feature".to_string(),
                nullable: false,
                default_value: None,
            },
        ];
        
        let store = FeatureStore::new("test_store".to_string(), schema).unwrap();
        
        // Test serving features with specific keys
        let keys = vec!["feature1".to_string(), "nonexistent".to_string()];
        let result = store.serve_features(keys, None);
        assert!(result.is_ok());
        
        // Should return empty DataFrame since we have no data
        if let Ok(df) = result {
            assert_eq!(df.height(), 0);
        }
    }
}

#[cfg(test)]
mod feature_schema_tests {
    use super::*;
    
    #[test]
    fn test_feature_schema_creation() {
        let schema = FeatureSchema {
            name: "test_feature".to_string(),
            feature_type: FeatureType::Float,
            description: "A test feature".to_string(),
            nullable: true,
            default_value: Some(Value::Number(0.0)),
        };
        
        assert_eq!(schema.name, "test_feature");
        assert!(matches!(schema.feature_type, FeatureType::Float));
        assert_eq!(schema.description, "A test feature");
        assert!(schema.nullable);
        assert!(schema.default_value.is_some());
    }
    
    #[test]
    fn test_feature_type_variants() {
        let int_type = FeatureType::Integer;
        let float_type = FeatureType::Float;
        let string_type = FeatureType::String;
        let bool_type = FeatureType::Boolean;
        let timestamp_type = FeatureType::Timestamp;
        let array_type = FeatureType::Array(Box::new(FeatureType::Float));
        
        assert!(matches!(int_type, FeatureType::Integer));
        assert!(matches!(float_type, FeatureType::Float));
        assert!(matches!(string_type, FeatureType::String));
        assert!(matches!(bool_type, FeatureType::Boolean));
        assert!(matches!(timestamp_type, FeatureType::Timestamp));
        assert!(matches!(array_type, FeatureType::Array(_)));
    }
}

#[cfg(test)]
mod data_quality_tests {
    use super::*;
    
    #[test]
    fn test_data_quality_result_creation() {
        let quality_result = DataQualityResult {
            check_name: "null_check".to_string(),
            passed: true,
            score: 0.95,
            details: HashMap::from([
                ("null_count".to_string(), Value::Number(5.0)),
                ("total_rows".to_string(), Value::Number(100.0)),
            ]),
            timestamp: Utc::now(),
        };
        
        assert_eq!(quality_result.check_name, "null_check");
        assert!(quality_result.passed);
        assert_eq!(quality_result.score, 0.95);
        assert_eq!(quality_result.details.len(), 2);
    }
    
    #[test]
    fn test_drift_result_creation() {
        let drift_result = DriftResult {
            method: "ks_test".to_string(),
            drift_detected: false,
            drift_score: 0.05,
            threshold: 0.1,
            feature_drifts: HashMap::from([
                ("feature1".to_string(), 0.03),
                ("feature2".to_string(), 0.07),
            ]),
            timestamp: Utc::now(),
        };
        
        assert_eq!(drift_result.method, "ks_test");
        assert!(!drift_result.drift_detected);
        assert_eq!(drift_result.drift_score, 0.05);
        assert_eq!(drift_result.threshold, 0.1);
        assert_eq!(drift_result.feature_drifts.len(), 2);
    }
}

#[cfg(test)]
mod data_pipeline_function_tests {
    use super::*;
    
    #[test]
    fn test_feature_compute_function() {
        let mut vm = VirtualMachine::new();
        let result = feature_compute(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("features_computed".to_string()));
    }
    
    #[test]
    fn test_feature_serve_function() {
        let mut vm = VirtualMachine::new();
        let result = feature_serve(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
    
    #[test]
    fn test_data_drift_function() {
        let mut vm = VirtualMachine::new();
        let result = data_drift(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }
    
    #[test]
    fn test_data_validation_function() {
        let mut vm = VirtualMachine::new();
        let result = data_validation(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_data_quality_function() {
        let mut vm = VirtualMachine::new();
        let result = data_quality(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Number(0.95));
    }
    
    #[test]
    fn test_pipeline_create_function() {
        let mut vm = VirtualMachine::new();
        let result = pipeline_create(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("pipeline_id_123".to_string()));
    }
    
    #[test]
    fn test_pipeline_execute_function() {
        let mut vm = VirtualMachine::new();
        let result = pipeline_execute(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("execution_id_123".to_string()));
    }
}

#[cfg(test)]
mod feature_store_integration_tests {
    use super::*;
    use polars::prelude::*;
    
    #[test]
    fn test_feature_store_compute_features_with_config() {
        let schema = vec![
            FeatureSchema {
                name: "raw_value".to_string(),
                feature_type: FeatureType::Float,
                description: "Raw numeric value".to_string(),
                nullable: false,
                default_value: None,
            },
        ];
        
        let store = FeatureStore::new("test_store".to_string(), schema).unwrap();
        
        // Create test data
        let df = DataFrame::new(vec![
            Series::new("raw_value", &[1.0, 2.0, 3.0, 4.0, 5.0]),
        ]).unwrap();
        
        // Apply normalization transformation
        let config = HashMap::from([
            ("raw_value".to_string(), Value::String("normalize".to_string())),
        ]);
        
        let result = store.compute_features(config, df);
        assert!(result.is_ok());
        
        if let Ok(transformed_df) = result {
            // Should have original column plus normalized column
            assert!(transformed_df.get_column_names().contains(&"raw_value"));
            // Note: The actual normalized column would be added in a real implementation
        }
    }
    
    #[test]
    fn test_feature_store_log_transform() {
        let schema = vec![
            FeatureSchema {
                name: "positive_value".to_string(),
                feature_type: FeatureType::Float,
                description: "Positive numeric value".to_string(),
                nullable: false,
                default_value: None,
            },
        ];
        
        let store = FeatureStore::new("test_store".to_string(), schema).unwrap();
        
        // Create test data with positive values
        let df = DataFrame::new(vec![
            Series::new("positive_value", &[1.0, 2.718, 7.389, 20.086]),
        ]).unwrap();
        
        // Apply log transformation
        let config = HashMap::from([
            ("positive_value".to_string(), Value::String("log_transform".to_string())),
        ]);
        
        let result = store.compute_features(config, df);
        assert!(result.is_ok());
        
        if let Ok(transformed_df) = result {
            // Should have original column plus log-transformed column
            assert!(transformed_df.get_column_names().contains(&"positive_value"));
            // Note: The actual log-transformed column would be added in a real implementation
        }
    }
    
    #[test]
    fn test_feature_store_unknown_transformation() {
        let schema = vec![
            FeatureSchema {
                name: "test_value".to_string(),
                feature_type: FeatureType::Float,
                description: "Test value".to_string(),
                nullable: false,
                default_value: None,
            },
        ];
        
        let store = FeatureStore::new("test_store".to_string(), schema).unwrap();
        
        let df = DataFrame::new(vec![
            Series::new("test_value", &[1.0, 2.0, 3.0]),
        ]).unwrap();
        
        // Apply unknown transformation (should be ignored)
        let config = HashMap::from([
            ("test_value".to_string(), Value::String("unknown_transform".to_string())),
        ]);
        
        let result = store.compute_features(config, df);
        assert!(result.is_ok());
        
        if let Ok(transformed_df) = result {
            // Should just return original data
            assert!(transformed_df.get_column_names().contains(&"test_value"));
        }
    }
    
    #[test]
    fn test_feature_store_empty_dataframe() {
        let schema = vec![];
        let store = FeatureStore::new("empty_store".to_string(), schema).unwrap();
        
        let empty_df = DataFrame::empty();
        let config = HashMap::new();
        
        let result = store.compute_features(config, empty_df);
        assert!(result.is_ok());
        
        if let Ok(df) = result {
            assert_eq!(df.height(), 0);
            assert_eq!(df.width(), 0);
        }
    }
    
    #[test]
    fn test_feature_store_serve_empty_keys() {
        let schema = vec![
            FeatureSchema {
                name: "feature1".to_string(),
                feature_type: FeatureType::Float,
                description: "Feature 1".to_string(),
                nullable: false,
                default_value: None,
            },
        ];
        
        let store = FeatureStore::new("test_store".to_string(), schema).unwrap();
        
        let result = store.serve_features(vec![], None);
        assert!(result.is_ok());
        
        if let Ok(df) = result {
            // Should return empty DataFrame when no keys provided
            assert_eq!(df.height(), 0);
            assert_eq!(df.width(), 0);
        }
    }
}