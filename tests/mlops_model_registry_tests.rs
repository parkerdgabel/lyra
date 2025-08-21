//! MLOps Model Registry Tests
//! 
//! Tests for the model registry functionality in the MLOps module.
//! This follows Test-Driven Development practices.

use lyra::error::{Error as LyraError, Result as LyraResult};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::ai_ml::mlops::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use chrono::Utc;

#[cfg(test)]
mod model_registry_tests {
    use super::*;

    #[test]
    fn test_model_registry_create() {
        let mut vm = VirtualMachine::new();
        
        let result = model_registry_create(&mut vm, &[]);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = result {
            assert_eq!(registry_obj.type_name(), "ModelRegistry");
        } else {
            panic!("Expected LyObj with ModelRegistry");
        }
    }
    
    #[test]
    fn test_model_registry_register_model() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Test registering a model
            let args = vec![
                Value::String("test_model".to_string()),
                Value::String("v1.0".to_string()),
                Value::String("/path/to/model".to_string()),
                Value::String("sklearn".to_string()),
            ];
            
            let register_result = registry_obj.call_method("register", &args);
            assert!(register_result.is_ok());
            
            if let Ok(Value::Boolean(success)) = register_result {
                assert!(success);
            } else {
                panic!("Expected boolean success result");
            }
        }
    }
    
    #[test]
    fn test_model_registry_register_invalid_args() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Test with insufficient arguments
            let args = vec![
                Value::String("test_model".to_string()),
            ];
            
            let register_result = registry_obj.call_method("register", &args);
            assert!(register_result.is_err());
            
            if let Err(LyraError::ArityError { expected, actual }) = register_result {
                assert_eq!(expected, 4);
                assert_eq!(actual, 1);
            } else {
                panic!("Expected ArityError");
            }
        }
    }
    
    #[test]
    fn test_model_registry_get_model() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Register a model first
            let register_args = vec![
                Value::String("test_model".to_string()),
                Value::String("v1.0".to_string()),
                Value::String("/path/to/model".to_string()),
                Value::String("sklearn".to_string()),
            ];
            
            let _register_result = registry_obj.call_method("register", &register_args);
            
            // Now get the model
            let get_args = vec![
                Value::String("test_model".to_string()),
                Value::String("v1.0".to_string()),
            ];
            
            let get_result = registry_obj.call_method("get", &get_args);
            assert!(get_result.is_ok());
            
            if let Ok(Value::String(model_info)) = get_result {
                assert!(model_info.contains("test_model"));
                assert!(model_info.contains("v1.0"));
            } else {
                panic!("Expected string model info");
            }
        }
    }
    
    #[test]
    fn test_model_registry_list_models() {
        let mut vm = VirtualMachine::new();
        
        let registry_result = model_registry_create(&mut vm, &[]);
        assert!(registry_result.is_ok());
        
        if let Ok(Value::LyObj(registry_obj)) = registry_result {
            // Initially should have no models
            let list_result = registry_obj.call_method("list", &[]);
            assert!(list_result.is_ok());
            
            if let Ok(Value::List(models)) = list_result {
                assert_eq!(models.len(), 0);
            } else {
                panic!("Expected list of models");
            }
            
            // Register a model
            let register_args = vec![
                Value::String("test_model".to_string()),
                Value::String("v1.0".to_string()),
                Value::String("/path/to/model".to_string()),
                Value::String("sklearn".to_string()),
            ];
            
            let _register_result = registry_obj.call_method("register", &register_args);
            
            // Now should have one model
            let list_result = registry_obj.call_method("list", &[]);
            assert!(list_result.is_ok());
            
            if let Ok(Value::List(models)) = list_result {
                assert_eq!(models.len(), 1);
            } else {
                panic!("Expected list of models");
            }
        }
    }
    
    #[test]
    fn test_model_registry_direct_creation() {
        let registry_result = ModelRegistry::new();
        assert!(registry_result.is_ok());
        
        if let Ok(registry) = registry_result {
            // Test registering a model directly
            let model_info = ModelVersionInfo {
                name: "test_model".to_string(),
                version: "v1.0".to_string(),
                stage: ModelStage::Development,
                path: "/path/to/model".to_string(),
                framework: "sklearn".to_string(),
                metrics: HashMap::from([("accuracy".to_string(), 0.95)]),
                metadata: HashMap::new(),
                created_at: Utc::now(),
                updated_at: Utc::now(),
            };
            
            let register_result = registry.register_model(model_info);
            assert!(register_result.is_ok());
            
            // Test getting the model
            let get_result = registry.get_model("test_model", "v1.0");
            assert!(get_result.is_ok());
            
            if let Ok(model) = get_result {
                assert_eq!(model.name, "test_model");
                assert_eq!(model.version, "v1.0");
                assert_eq!(model.framework, "sklearn");
                assert!(matches!(model.stage, ModelStage::Development));
                assert_eq!(model.metrics.get("accuracy"), Some(&0.95));
            }
        }
    }
    
    #[test]
    fn test_model_promotion() {
        let registry = ModelRegistry::new().unwrap();
        
        // Register a model
        let model_info = ModelVersionInfo {
            name: "test_model".to_string(),
            version: "v1.0".to_string(),
            stage: ModelStage::Development,
            path: "/path/to/model".to_string(),
            framework: "sklearn".to_string(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        registry.register_model(model_info).unwrap();
        
        // Promote to staging
        let promote_result = registry.promote_model("test_model", "v1.0", ModelStage::Staging);
        assert!(promote_result.is_ok());
        
        // Check that stage was updated
        let model = registry.get_model("test_model", "v1.0").unwrap();
        assert!(matches!(model.stage, ModelStage::Staging));
    }
    
    #[test]
    fn test_model_promotion_nonexistent() {
        let registry = ModelRegistry::new().unwrap();
        
        // Try to promote non-existent model
        let promote_result = registry.promote_model("nonexistent", "v1.0", ModelStage::Production);
        assert!(promote_result.is_err());
    }
    
    #[test]
    fn test_model_list_with_stage_filter() {
        let registry = ModelRegistry::new().unwrap();
        
        // Register models in different stages
        let model1 = ModelVersionInfo {
            name: "model1".to_string(),
            version: "v1.0".to_string(),
            stage: ModelStage::Development,
            path: "/path/to/model1".to_string(),
            framework: "sklearn".to_string(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        let model2 = ModelVersionInfo {
            name: "model2".to_string(),
            version: "v1.0".to_string(),
            stage: ModelStage::Production,
            path: "/path/to/model2".to_string(),
            framework: "tensorflow".to_string(),
            metrics: HashMap::new(),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        registry.register_model(model1).unwrap();
        registry.register_model(model2).unwrap();
        
        // List all models
        let all_models = registry.list_models(None);
        assert_eq!(all_models.len(), 2);
        
        // List only production models
        let prod_models = registry.list_models(Some(ModelStage::Production));
        assert_eq!(prod_models.len(), 1);
        assert_eq!(prod_models[0].name, "model2");
        
        // List only development models
        let dev_models = registry.list_models(Some(ModelStage::Development));
        assert_eq!(dev_models.len(), 1);
        assert_eq!(dev_models[0].name, "model1");
    }
    
    #[test]
    fn test_multiple_model_versions() {
        let registry = ModelRegistry::new().unwrap();
        
        // Register multiple versions of the same model
        let model_v1 = ModelVersionInfo {
            name: "test_model".to_string(),
            version: "v1.0".to_string(),
            stage: ModelStage::Production,
            path: "/path/to/model_v1".to_string(),
            framework: "sklearn".to_string(),
            metrics: HashMap::from([("accuracy".to_string(), 0.90)]),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        let model_v2 = ModelVersionInfo {
            name: "test_model".to_string(),
            version: "v2.0".to_string(),
            stage: ModelStage::Staging,
            path: "/path/to/model_v2".to_string(),
            framework: "sklearn".to_string(),
            metrics: HashMap::from([("accuracy".to_string(), 0.95)]),
            metadata: HashMap::new(),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        registry.register_model(model_v1).unwrap();
        registry.register_model(model_v2).unwrap();
        
        // Check both versions exist
        let v1 = registry.get_model("test_model", "v1.0").unwrap();
        let v2 = registry.get_model("test_model", "v2.0").unwrap();
        
        assert_eq!(v1.version, "v1.0");
        assert_eq!(v2.version, "v2.0");
        assert!(matches!(v1.stage, ModelStage::Production));
        assert!(matches!(v2.stage, ModelStage::Staging));
        assert_eq!(v1.metrics.get("accuracy"), Some(&0.90));
        assert_eq!(v2.metrics.get("accuracy"), Some(&0.95));
        
        // List should show both versions
        let all_models = registry.list_models(None);
        assert_eq!(all_models.len(), 2);
    }
}

#[cfg(test)]
mod model_version_info_tests {
    use super::*;
    
    #[test]
    fn test_model_version_info_creation() {
        let model_info = ModelVersionInfo {
            name: "test_model".to_string(),
            version: "v1.0".to_string(),
            stage: ModelStage::Development,
            path: "/path/to/model".to_string(),
            framework: "pytorch".to_string(),
            metrics: HashMap::from([
                ("accuracy".to_string(), 0.95),
                ("f1_score".to_string(), 0.92),
            ]),
            metadata: HashMap::from([
                ("description".to_string(), Value::String("Test model".to_string())),
                ("author".to_string(), Value::String("ML Engineer".to_string())),
            ]),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        
        assert_eq!(model_info.name, "test_model");
        assert_eq!(model_info.version, "v1.0");
        assert_eq!(model_info.framework, "pytorch");
        assert!(matches!(model_info.stage, ModelStage::Development));
        assert_eq!(model_info.metrics.len(), 2);
        assert_eq!(model_info.metadata.len(), 2);
    }
    
    #[test]
    fn test_model_stage_variants() {
        let dev_stage = ModelStage::Development;
        let staging_stage = ModelStage::Staging;
        let prod_stage = ModelStage::Production;
        let archived_stage = ModelStage::Archived;
        
        // Test that stages can be compared
        assert!(matches!(dev_stage, ModelStage::Development));
        assert!(matches!(staging_stage, ModelStage::Staging));
        assert!(matches!(prod_stage, ModelStage::Production));
        assert!(matches!(archived_stage, ModelStage::Archived));
    }
}

#[cfg(test)]
mod model_registry_function_tests {
    use super::*;
    
    #[test]
    fn test_model_register_function() {
        let mut vm = VirtualMachine::new();
        let result = model_register(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_model_version_function() {
        let mut vm = VirtualMachine::new();
        let result = model_version(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("v1.0".to_string()));
    }
    
    #[test]
    fn test_model_promote_function() {
        let mut vm = VirtualMachine::new();
        let result = model_promote(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_model_deploy_function() {
        let mut vm = VirtualMachine::new();
        let result = model_deploy(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("deployment_id_123".to_string()));
    }
    
    #[test]
    fn test_model_retire_function() {
        let mut vm = VirtualMachine::new();
        let result = model_retire(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_model_search_function() {
        let mut vm = VirtualMachine::new();
        let result = model_search(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
    
    #[test]
    fn test_model_lineage_function() {
        let mut vm = VirtualMachine::new();
        let result = model_lineage(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
    
    #[test]
    fn test_model_metrics_function() {
        let mut vm = VirtualMachine::new();
        let result = model_metrics(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
}