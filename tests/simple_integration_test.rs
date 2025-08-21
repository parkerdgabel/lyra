//! Simple integration test for Phase 17B functions
//!
//! This test focuses specifically on testing the new integration capabilities
//! without dependencies on the broader codebase that may have compilation issues.

#[cfg(test)]
mod tests {
    use lyra::vm::{Value, VmError};
    use lyra::stdlib::integrations::version_control::*;
    use lyra::stdlib::integrations::cicd::*;
    use lyra::stdlib::integrations::optimization::*;
    use std::collections::HashMap;

    #[test]
    fn test_git_repository_function() {
        let args = vec![
            Value::String("/path/to/repo".to_string()),
            Value::String("https://github.com/user/repo.git".to_string()),
            Value::List(vec![
                Value::String("token".to_string()),
                Value::String("ghp_xxxx".to_string()),
            ]),
        ];

        let result = git_repository(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "GitRepository");
        } else {
            panic!("Expected GitRepository object");
        }
    }

    #[test]
    fn test_pipeline_create_function() {
        let args = vec![
            Value::String("test-pipeline".to_string()),
            Value::List(vec![
                Value::String("build".to_string()),
                Value::String("test".to_string()),
                Value::String("deploy".to_string()),
            ]),
            Value::List(vec![
                Value::String("on_push".to_string()),
                Value::String("main".to_string()),
            ]),
            Value::List(vec![]), // options
        ];

        let result = pipeline_create(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "CiCdPipeline");
        } else {
            panic!("Expected CiCdPipeline object");
        }
    }

    #[test]
    fn test_resource_usage_function() {
        let args = vec![
            Value::List(vec![
                Value::String("service1".to_string()),
                Value::String("service2".to_string()),
            ]),
            Value::List(vec![
                Value::String("start".to_string()),
                Value::String("2024-01-01".to_string()),
                Value::String("end".to_string()),
                Value::String("2024-01-31".to_string()),
            ]),
            Value::String("daily".to_string()),
        ];

        let result = resource_usage(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "ResourceUsage");
        } else {
            panic!("Expected ResourceUsage object");
        }
    }

    #[test]
    fn test_build_stage_function() {
        let args = vec![
            Value::String("test-pipeline".to_string()),
            Value::List(vec![
                Value::String("dockerfile".to_string()),
                Value::String("Dockerfile".to_string()),
                Value::String("context".to_string()),
                Value::String(".".to_string()),
            ]),
            Value::String("production".to_string()),
        ];

        let result = build_stage(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "BuildStage");
        } else {
            panic!("Expected BuildStage object");
        }
    }

    #[test]
    fn test_cost_analysis_function() {
        let args = vec![
            Value::String("usage_data".to_string()),
            Value::String("aws".to_string()),
            Value::List(vec![
                Value::String("team".to_string()),
                Value::String("engineering".to_string()),
            ]),
        ];

        let result = cost_analysis(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "CostAnalysis");
        } else {
            panic!("Expected CostAnalysis object");
        }
    }

    #[test]
    fn test_wrong_argument_count() {
        // Test GitRepository with wrong number of arguments
        let args = vec![
            Value::String("/path/to/repo".to_string()),
            // Missing required arguments
        ];

        let result = git_repository(&args);
        assert!(result.is_err());
        
        if let Err(VmError::ArgumentError { expected, actual }) = result {
            assert_eq!(expected, 3);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected ArgumentError");
        }
    }

    #[test]
    fn test_wrong_argument_type() {
        // Test GitRepository with wrong argument type
        let args = vec![
            Value::Integer(123), // Should be String
            Value::String("https://github.com/user/repo.git".to_string()),
            Value::List(vec![]),
        ];

        let result = git_repository(&args);
        assert!(result.is_err());
        
        if let Err(VmError::TypeError { expected, actual: _ }) = result {
            assert_eq!(expected, "String");
        } else {
            panic!("Expected TypeError");
        }
    }
}