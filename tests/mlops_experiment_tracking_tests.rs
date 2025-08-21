//! MLOps Experiment Tracking Tests
//! 
//! Tests for the experiment tracking functionality in the MLOps module.
//! This follows Test-Driven Development practices.

use lyra::error::{Error as LyraError, Result as LyraResult};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::ai_ml::mlops::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use chrono::Utc;

#[cfg(test)]
mod experiment_tracking_tests {
    use super::*;

    #[test]
    fn test_experiment_create_basic() {
        let mut vm = VirtualMachine::new();
        
        let args = vec![
            Value::String("test_experiment".to_string()),
            Value::String("A test experiment".to_string()),
            Value::List(vec![
                Value::String("ml".to_string()),
                Value::String("test".to_string()),
            ]),
        ];
        
        let result = experiment_create(&mut vm, &args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(experiment_obj)) = result {
            assert_eq!(experiment_obj.type_name(), "Experiment");
        } else {
            panic!("Expected LyObj with Experiment");
        }
    }
    
    #[test]
    fn test_experiment_create_invalid_args() {
        let mut vm = VirtualMachine::new();
        
        // Test with insufficient arguments
        let args = vec![
            Value::String("test_experiment".to_string()),
        ];
        
        let result = experiment_create(&mut vm, &args);
        assert!(result.is_err());
        
        if let Err(LyraError::ArityError { expected, actual }) = result {
            assert_eq!(expected, 3);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected ArityError");
        }
    }
    
    #[test]
    fn test_experiment_create_invalid_types() {
        let mut vm = VirtualMachine::new();
        
        let args = vec![
            Value::Number(123.0), // Should be string
            Value::String("A test experiment".to_string()),
            Value::List(vec![Value::String("ml".to_string())]),
        ];
        
        let result = experiment_create(&mut vm, &args);
        assert!(result.is_err());
        
        if let Err(LyraError::TypeError(_)) = result {
            // Expected behavior
        } else {
            panic!("Expected TypeError");
        }
    }
    
    #[test]
    fn test_experiment_run_creation() {
        let mut vm = VirtualMachine::new();
        
        // First create an experiment
        let args = vec![
            Value::String("test_experiment".to_string()),
            Value::String("A test experiment".to_string()),
            Value::List(vec![Value::String("ml".to_string())]),
        ];
        
        let experiment_result = experiment_create(&mut vm, &args);
        assert!(experiment_result.is_ok());
        
        if let Ok(Value::LyObj(experiment_obj)) = experiment_result {
            // Test creating a run
            let run_args = vec![];
            let run_result = experiment_obj.call_method("createRun", &run_args);
            assert!(run_result.is_ok());
            
            if let Ok(Value::String(run_id)) = run_result {
                assert!(!run_id.is_empty());
                // UUID format check (36 characters with hyphens)
                assert_eq!(run_id.len(), 36);
            } else {
                panic!("Expected string run ID");
            }
        }
    }
    
    #[test]
    fn test_experiment_metric_logging() {
        let mut vm = VirtualMachine::new();
        
        // Create experiment
        let args = vec![
            Value::String("test_experiment".to_string()),
            Value::String("A test experiment".to_string()),
            Value::List(vec![Value::String("ml".to_string())]),
        ];
        
        let experiment_result = experiment_create(&mut vm, &args);
        assert!(experiment_result.is_ok());
        
        if let Ok(Value::LyObj(experiment_obj)) = experiment_result {
            // Create a run
            let run_result = experiment_obj.call_method("createRun", &[]);
            assert!(run_result.is_ok());
            
            if let Ok(Value::String(run_id)) = run_result {
                // Log a metric
                let metric_args = vec![
                    Value::String(run_id),
                    Value::String("accuracy".to_string()),
                    Value::Number(0.95),
                    Value::Number(1.0), // step
                ];
                
                let metric_result = experiment_obj.call_method("logMetric", &metric_args);
                assert!(metric_result.is_ok());
                
                if let Ok(Value::Boolean(success)) = metric_result {
                    assert!(success);
                } else {
                    panic!("Expected boolean success result");
                }
            }
        }
    }
    
    #[test]
    fn test_experiment_metric_logging_invalid_run() {
        let mut vm = VirtualMachine::new();
        
        // Create experiment
        let args = vec![
            Value::String("test_experiment".to_string()),
            Value::String("A test experiment".to_string()),
            Value::List(vec![Value::String("ml".to_string())]),
        ];
        
        let experiment_result = experiment_create(&mut vm, &args);
        assert!(experiment_result.is_ok());
        
        if let Ok(Value::LyObj(experiment_obj)) = experiment_result {
            // Try to log metric to non-existent run
            let metric_args = vec![
                Value::String("nonexistent_run_id".to_string()),
                Value::String("accuracy".to_string()),
                Value::Number(0.95),
            ];
            
            let metric_result = experiment_obj.call_method("logMetric", &metric_args);
            assert!(metric_result.is_err());
        }
    }
    
    #[test]
    fn test_experiment_get_runs() {
        let mut vm = VirtualMachine::new();
        
        // Create experiment
        let args = vec![
            Value::String("test_experiment".to_string()),
            Value::String("A test experiment".to_string()),
            Value::List(vec![Value::String("ml".to_string())]),
        ];
        
        let experiment_result = experiment_create(&mut vm, &args);
        assert!(experiment_result.is_ok());
        
        if let Ok(Value::LyObj(experiment_obj)) = experiment_result {
            // Initially should have no runs
            let runs_result = experiment_obj.call_method("getRuns", &[]);
            assert!(runs_result.is_ok());
            
            if let Ok(Value::List(runs)) = runs_result {
                assert_eq!(runs.len(), 0);
            } else {
                panic!("Expected list of runs");
            }
            
            // Create a run
            let _run_result = experiment_obj.call_method("createRun", &[]);
            
            // Now should have one run
            let runs_result = experiment_obj.call_method("getRuns", &[]);
            assert!(runs_result.is_ok());
            
            if let Ok(Value::List(runs)) = runs_result {
                assert_eq!(runs.len(), 1);
            } else {
                panic!("Expected list of runs");
            }
        }
    }
    
    #[test]
    fn test_experiment_config_creation() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "A test experiment".to_string(),
            tags: vec!["ml".to_string(), "test".to_string()],
            created_at: Utc::now(),
            created_by: "test_user".to_string(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(config.name, "test_experiment");
        assert_eq!(config.description, "A test experiment");
        assert_eq!(config.tags.len(), 2);
        assert_eq!(config.created_by, "test_user");
    }
    
    #[test]
    fn test_experiment_direct_creation() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "A test experiment".to_string(),
            tags: vec!["ml".to_string()],
            created_at: Utc::now(),
            created_by: "test_user".to_string(),
            metadata: HashMap::new(),
        };
        
        let experiment_result = Experiment::new(config);
        assert!(experiment_result.is_ok());
        
        if let Ok(experiment) = experiment_result {
            assert_eq!(experiment.get_config().name, "test_experiment");
            
            // Test creating a run
            let run_id = experiment.create_run(HashMap::new(), Vec::new());
            assert!(run_id.is_ok());
            
            if let Ok(run_id_str) = run_id {
                // Test logging a metric
                let metric_result = experiment.log_metric(&run_id_str, "accuracy", 0.95, 1);
                assert!(metric_result.is_ok());
                
                // Test getting runs
                let runs = experiment.get_runs();
                assert_eq!(runs.len(), 1);
                assert_eq!(runs[0].run_id, run_id_str);
                
                // Check that metric was logged
                assert!(runs[0].metrics.contains_key("accuracy"));
                assert_eq!(runs[0].metrics["accuracy"].len(), 1);
                assert_eq!(runs[0].metrics["accuracy"][0].value, 0.95);
                assert_eq!(runs[0].metrics["accuracy"][0].step, 1);
            }
        }
    }
    
    #[test]
    fn test_run_status_updates() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "A test experiment".to_string(),
            tags: vec!["ml".to_string()],
            created_at: Utc::now(),
            created_by: "test_user".to_string(),
            metadata: HashMap::new(),
        };
        
        let experiment = Experiment::new(config).unwrap();
        let run_id = experiment.create_run(HashMap::new(), Vec::new()).unwrap();
        
        // Update run status to completed
        let status_result = experiment.update_run_status(&run_id, RunStatus::Completed, Some("Run completed successfully".to_string()));
        assert!(status_result.is_ok());
        
        // Check that status was updated
        let runs = experiment.get_runs();
        assert_eq!(runs.len(), 1);
        if let RunStatus::Completed = runs[0].status {
            // Expected
        } else {
            panic!("Expected Completed status");
        }
        
        assert!(runs[0].end_time.is_some());
        assert!(runs[0].parameters.contains_key("status_message"));
    }
    
    #[test]
    fn test_artifact_management() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "A test experiment".to_string(),
            tags: vec!["ml".to_string()],
            created_at: Utc::now(),
            created_by: "test_user".to_string(),
            metadata: HashMap::new(),
        };
        
        let experiment = Experiment::new(config).unwrap();
        let run_id = experiment.create_run(HashMap::new(), Vec::new()).unwrap();
        
        // Add an artifact
        let artifact = ArtifactInfo {
            name: "model.pkl".to_string(),
            path: "/tmp/model.pkl".to_string(),
            artifact_type: "model".to_string(),
            size: 1024,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };
        
        let artifact_result = experiment.add_artifact(&run_id, artifact);
        assert!(artifact_result.is_ok());
        
        // Check that artifact was added
        let runs = experiment.get_runs();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].artifacts.len(), 1);
        assert_eq!(runs[0].artifacts[0].name, "model.pkl");
        assert_eq!(runs[0].artifacts[0].artifact_type, "model");
    }
    
    #[test]
    fn test_multiple_metrics_same_run() {
        let config = ExperimentConfig {
            name: "test_experiment".to_string(),
            description: "A test experiment".to_string(),
            tags: vec!["ml".to_string()],
            created_at: Utc::now(),
            created_by: "test_user".to_string(),
            metadata: HashMap::new(),
        };
        
        let experiment = Experiment::new(config).unwrap();
        let run_id = experiment.create_run(HashMap::new(), Vec::new()).unwrap();
        
        // Log multiple metrics
        experiment.log_metric(&run_id, "accuracy", 0.95, 1).unwrap();
        experiment.log_metric(&run_id, "loss", 0.05, 1).unwrap();
        experiment.log_metric(&run_id, "accuracy", 0.96, 2).unwrap(); // Same metric, different step
        
        // Check metrics
        let runs = experiment.get_runs();
        assert_eq!(runs.len(), 1);
        
        let run = &runs[0];
        assert_eq!(run.metrics.len(), 2); // accuracy and loss
        assert_eq!(run.metrics["accuracy"].len(), 2); // Two accuracy values
        assert_eq!(run.metrics["loss"].len(), 1); // One loss value
        
        // Check accuracy progression
        assert_eq!(run.metrics["accuracy"][0].value, 0.95);
        assert_eq!(run.metrics["accuracy"][0].step, 1);
        assert_eq!(run.metrics["accuracy"][1].value, 0.96);
        assert_eq!(run.metrics["accuracy"][1].step, 2);
    }
}

#[cfg(test)]
mod experiment_function_tests {
    use super::*;
    
    #[test]
    fn test_experiment_log_function() {
        let mut vm = VirtualMachine::new();
        let result = experiment_log(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_experiment_artifact_function() {
        let mut vm = VirtualMachine::new();
        let result = experiment_artifact(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_experiment_compare_function() {
        let mut vm = VirtualMachine::new();
        let result = experiment_compare(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
    
    #[test]
    fn test_experiment_list_function() {
        let mut vm = VirtualMachine::new();
        let result = experiment_list(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::List(Vec::new()));
    }
    
    #[test]
    fn test_run_create_function() {
        let mut vm = VirtualMachine::new();
        let result = run_create(&mut vm, &[]);
        assert!(result.is_ok());
        
        if let Ok(Value::String(run_id)) = result {
            assert!(!run_id.is_empty());
            assert_eq!(run_id.len(), 36); // UUID format
        } else {
            panic!("Expected string run ID");
        }
    }
    
    #[test]
    fn test_run_log_function() {
        let mut vm = VirtualMachine::new();
        let result = run_log(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_run_status_function() {
        let mut vm = VirtualMachine::new();
        let result = run_status(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_experiment_visualize_function() {
        let mut vm = VirtualMachine::new();
        let result = experiment_visualize(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("visualization_created".to_string()));
    }
}

#[cfg(test)]
mod run_data_tests {
    use super::*;
    
    #[test]
    fn test_run_data_creation() {
        let run_data = RunData {
            run_id: "test_run_123".to_string(),
            experiment_id: "test_experiment".to_string(),
            status: RunStatus::Running,
            start_time: Utc::now(),
            end_time: None,
            metrics: HashMap::new(),
            parameters: HashMap::new(),
            artifacts: Vec::new(),
            tags: vec!["test".to_string()],
        };
        
        assert_eq!(run_data.run_id, "test_run_123");
        assert_eq!(run_data.experiment_id, "test_experiment");
        assert!(matches!(run_data.status, RunStatus::Running));
        assert!(run_data.end_time.is_none());
        assert_eq!(run_data.tags.len(), 1);
    }
    
    #[test]
    fn test_metric_value_creation() {
        let metric = MetricValue {
            value: 0.95,
            step: 100,
            timestamp: Utc::now(),
        };
        
        assert_eq!(metric.value, 0.95);
        assert_eq!(metric.step, 100);
    }
    
    #[test]
    fn test_artifact_info_creation() {
        let artifact = ArtifactInfo {
            name: "model.pkl".to_string(),
            path: "/path/to/model.pkl".to_string(),
            artifact_type: "model".to_string(),
            size: 2048,
            created_at: Utc::now(),
            metadata: HashMap::new(),
        };
        
        assert_eq!(artifact.name, "model.pkl");
        assert_eq!(artifact.artifact_type, "model");
        assert_eq!(artifact.size, 2048);
    }
}

#[cfg(test)]
mod error_handling_tests {
    use super::*;
    
    #[test]
    fn test_mlops_error_creation() {
        let error = MLOpsError::ExperimentNotFound("test_experiment".to_string());
        assert_eq!(error.to_string(), "Experiment not found: test_experiment");
        
        let error = MLOpsError::ModelNotFound {
            name: "test_model".to_string(),
            version: "v1.0".to_string(),
        };
        assert_eq!(error.to_string(), "Model not found: test_model version v1.0");
        
        let error = MLOpsError::DataValidationFailed("Invalid data format".to_string());
        assert_eq!(error.to_string(), "Data validation failed: Invalid data format");
    }
}