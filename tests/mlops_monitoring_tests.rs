//! MLOps Performance Monitoring Tests
//! 
//! Tests for the performance monitoring functionality in the MLOps module.
//! This follows Test-Driven Development practices.

use lyra::error::{Error as LyraError, Result as LyraResult};
use lyra::vm::{Value, VirtualMachine};
use lyra::stdlib::ai_ml::mlops::*;
use pretty_assertions::assert_eq;
use std::collections::HashMap;
use chrono::Utc;

#[cfg(test)]
mod model_monitor_tests {
    use super::*;

    #[test]
    fn test_model_monitor_create() {
        let mut vm = VirtualMachine::new();
        
        let args = vec![
            Value::String("test_model".to_string()),
            Value::List(vec![
                Value::String("accuracy".to_string()),
                Value::String("latency".to_string()),
                Value::String("throughput".to_string()),
            ]),
        ];
        
        let result = model_monitor_create(&mut vm, &args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = result {
            assert_eq!(monitor_obj.type_name(), "ModelMonitor");
        } else {
            panic!("Expected LyObj with ModelMonitor");
        }
    }
    
    #[test]
    fn test_model_monitor_create_invalid_args() {
        let mut vm = VirtualMachine::new();
        
        // Test with insufficient arguments
        let args = vec![
            Value::String("test_model".to_string()),
        ];
        
        let result = model_monitor_create(&mut vm, &args);
        assert!(result.is_err());
        
        if let Err(LyraError::ArityError { expected, actual }) = result {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected ArityError");
        }
    }
    
    #[test]
    fn test_model_monitor_record_metric() {
        let mut vm = VirtualMachine::new();
        
        let monitor_result = model_monitor_create(&mut vm, &[
            Value::String("test_model".to_string()),
            Value::List(vec![Value::String("accuracy".to_string())]),
        ]);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            let args = vec![
                Value::String("accuracy".to_string()),
                Value::Number(0.95),
                Value::Number(1.0), // step
            ];
            
            let result = monitor_obj.call_method("recordMetric", &args);
            assert!(result.is_ok());
            
            if let Ok(Value::Boolean(success)) = result {
                assert!(success);
            } else {
                panic!("Expected boolean success result");
            }
        }
    }
    
    #[test]
    fn test_model_monitor_record_metric_invalid_args() {
        let mut vm = VirtualMachine::new();
        
        let monitor_result = model_monitor_create(&mut vm, &[
            Value::String("test_model".to_string()),
            Value::List(vec![Value::String("accuracy".to_string())]),
        ]);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            // Test with insufficient arguments
            let args = vec![
                Value::String("accuracy".to_string()),
            ];
            
            let result = monitor_obj.call_method("recordMetric", &args);
            assert!(result.is_err());
            
            if let Err(LyraError::ArityError { expected, actual }) = result {
                assert_eq!(expected, 2);
                assert_eq!(actual, 1);
            } else {
                panic!("Expected ArityError");
            }
        }
    }
    
    #[test]
    fn test_model_monitor_detect_drift() {
        let mut vm = VirtualMachine::new();
        
        let monitor_result = model_monitor_create(&mut vm, &[
            Value::String("test_model".to_string()),
            Value::List(vec![Value::String("accuracy".to_string())]),
        ]);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            // First record some metrics to have enough data
            for i in 1..=200 {
                let value = 0.9 + (i as f64 / 1000.0); // Gradually increasing accuracy
                let args = vec![
                    Value::String("accuracy".to_string()),
                    Value::Number(value),
                    Value::Number(i as f64),
                ];
                let _ = monitor_obj.call_method("recordMetric", &args);
            }
            
            // Now test drift detection
            let args = vec![
                Value::String("accuracy".to_string()),
                Value::Number(100.0), // baseline window
                Value::Number(50.0),  // current window
            ];
            
            let result = monitor_obj.call_method("detectDrift", &args);
            assert!(result.is_ok());
            
            if let Ok(Value::String(drift_info)) = result {
                assert!(drift_info.contains("DriftResult"));
            } else {
                panic!("Expected string drift result info");
            }
        }
    }
    
    #[test]
    fn test_model_monitor_get_alerts() {
        let mut vm = VirtualMachine::new();
        
        let monitor_result = model_monitor_create(&mut vm, &[
            Value::String("test_model".to_string()),
            Value::List(vec![Value::String("accuracy".to_string())]),
        ]);
        assert!(monitor_result.is_ok());
        
        if let Ok(Value::LyObj(monitor_obj)) = monitor_result {
            let result = monitor_obj.call_method("getAlerts", &[]);
            assert!(result.is_ok());
            
            if let Ok(Value::List(alerts)) = result {
                // Initially should have no alerts
                assert_eq!(alerts.len(), 0);
            } else {
                panic!("Expected list of alerts");
            }
        }
    }
    
    #[test]
    fn test_model_monitor_direct_creation() {
        let config = MonitoringConfig {
            model_name: "test_model".to_string(),
            metrics: vec!["accuracy".to_string(), "latency".to_string()],
            thresholds: HashMap::from([
                ("accuracy".to_string(), 0.9),
                ("latency".to_string(), 100.0),
            ]),
            alert_channels: vec!["email".to_string(), "slack".to_string()],
            check_interval: 60,
        };
        
        let monitor_result = ModelMonitor::new(config);
        assert!(monitor_result.is_ok());
        
        if let Ok(monitor) = monitor_result {
            // Test recording a metric
            let record_result = monitor.record_metric("accuracy", 0.95, 1);
            assert!(record_result.is_ok());
            
            // Test getting metric history
            let history = monitor.get_metric_history("accuracy");
            assert!(history.is_some());
            
            if let Some(metrics) = history {
                assert_eq!(metrics.len(), 1);
                assert_eq!(metrics[0].value, 0.95);
                assert_eq!(metrics[0].step, 1);
            }
        }
    }
    
    #[test]
    fn test_model_monitor_threshold_alerts() {
        let config = MonitoringConfig {
            model_name: "test_model".to_string(),
            metrics: vec!["accuracy".to_string()],
            thresholds: HashMap::from([
                ("accuracy".to_string(), 0.9), // Threshold for accuracy
            ]),
            alert_channels: vec!["email".to_string()],
            check_interval: 60,
        };
        
        let monitor = ModelMonitor::new(config).unwrap();
        
        // Record metric below threshold - should trigger alert
        let result = monitor.record_metric("accuracy", 0.85, 1);
        assert!(result.is_ok());
        
        // Check that alert was generated
        let alerts = monitor.get_alerts();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].model_name, "test_model");
        assert_eq!(alerts[0].metric, "accuracy");
        assert_eq!(alerts[0].current_value, 0.85);
        assert_eq!(alerts[0].threshold, 0.9);
        assert!(matches!(alerts[0].severity, AlertSeverity::Medium));
    }
    
    #[test]
    fn test_model_monitor_drift_detection() {
        let config = MonitoringConfig {
            model_name: "test_model".to_string(),
            metrics: vec!["accuracy".to_string()],
            thresholds: HashMap::new(),
            alert_channels: vec![],
            check_interval: 60,
        };
        
        let monitor = ModelMonitor::new(config).unwrap();
        
        // Record baseline metrics (stable values around 0.9)
        for i in 1..=100 {
            let value = 0.9 + (rand::random::<f64>() - 0.5) * 0.02; // Small noise around 0.9
            let _ = monitor.record_metric("accuracy", value, i);
        }
        
        // Record current metrics (shifted to 0.8)
        for i in 101..=150 {
            let value = 0.8 + (rand::random::<f64>() - 0.5) * 0.02; // Small noise around 0.8
            let _ = monitor.record_metric("accuracy", value, i);
        }
        
        // Detect drift
        let drift_result = monitor.detect_drift("accuracy", 100, 50);
        assert!(drift_result.is_ok());
        
        if let Ok(drift) = drift_result {
            assert_eq!(drift.method, "mean_difference");
            assert!(drift.drift_detected); // Should detect significant shift from 0.9 to 0.8
            assert!(drift.drift_score > 0.1); // Should be greater than 10% threshold
            assert_eq!(drift.threshold, 0.1);
            assert!(drift.feature_drifts.contains_key("accuracy"));
        }
    }
    
    #[test]
    fn test_model_monitor_insufficient_data_for_drift() {
        let config = MonitoringConfig {
            model_name: "test_model".to_string(),
            metrics: vec!["accuracy".to_string()],
            thresholds: HashMap::new(),
            alert_channels: vec![],
            check_interval: 60,
        };
        
        let monitor = ModelMonitor::new(config).unwrap();
        
        // Record only a few metrics (insufficient for drift detection)
        monitor.record_metric("accuracy", 0.9, 1).unwrap();
        monitor.record_metric("accuracy", 0.91, 2).unwrap();
        
        // Try to detect drift with windows requiring more data
        let drift_result = monitor.detect_drift("accuracy", 100, 50);
        assert!(drift_result.is_err());
        
        if let Err(e) = drift_result {
            assert!(e.to_string().contains("Insufficient data"));
        }
    }
    
    #[test]
    fn test_model_monitor_nonexistent_metric() {
        let config = MonitoringConfig {
            model_name: "test_model".to_string(),
            metrics: vec!["accuracy".to_string()],
            thresholds: HashMap::new(),
            alert_channels: vec![],
            check_interval: 60,
        };
        
        let monitor = ModelMonitor::new(config).unwrap();
        
        // Try to detect drift for non-existent metric
        let drift_result = monitor.detect_drift("nonexistent_metric", 10, 5);
        assert!(drift_result.is_err());
        
        if let Err(e) = drift_result {
            assert!(e.to_string().contains("Metric not found"));
        }
    }
}

#[cfg(test)]
mod monitoring_config_tests {
    use super::*;
    
    #[test]
    fn test_monitoring_config_creation() {
        let config = MonitoringConfig {
            model_name: "production_model".to_string(),
            metrics: vec![
                "accuracy".to_string(),
                "precision".to_string(),
                "recall".to_string(),
                "f1_score".to_string(),
            ],
            thresholds: HashMap::from([
                ("accuracy".to_string(), 0.95),
                ("precision".to_string(), 0.90),
                ("recall".to_string(), 0.85),
                ("f1_score".to_string(), 0.88),
            ]),
            alert_channels: vec![
                "email:ml-team@company.com".to_string(),
                "slack:#ml-alerts".to_string(),
                "webhook:https://api.company.com/alerts".to_string(),
            ],
            check_interval: 300, // 5 minutes
        };
        
        assert_eq!(config.model_name, "production_model");
        assert_eq!(config.metrics.len(), 4);
        assert_eq!(config.thresholds.len(), 4);
        assert_eq!(config.alert_channels.len(), 3);
        assert_eq!(config.check_interval, 300);
    }
    
    #[test]
    fn test_performance_alert_creation() {
        let alert = PerformanceAlert {
            alert_id: "alert_123".to_string(),
            model_name: "test_model".to_string(),
            metric: "accuracy".to_string(),
            current_value: 0.85,
            threshold: 0.9,
            severity: AlertSeverity::High,
            timestamp: Utc::now(),
            description: "Model accuracy dropped below threshold".to_string(),
        };
        
        assert_eq!(alert.alert_id, "alert_123");
        assert_eq!(alert.model_name, "test_model");
        assert_eq!(alert.metric, "accuracy");
        assert_eq!(alert.current_value, 0.85);
        assert_eq!(alert.threshold, 0.9);
        assert!(matches!(alert.severity, AlertSeverity::High));
        assert!(!alert.description.is_empty());
    }
    
    #[test]
    fn test_alert_severity_variants() {
        let low = AlertSeverity::Low;
        let medium = AlertSeverity::Medium;
        let high = AlertSeverity::High;
        let critical = AlertSeverity::Critical;
        
        assert!(matches!(low, AlertSeverity::Low));
        assert!(matches!(medium, AlertSeverity::Medium));
        assert!(matches!(high, AlertSeverity::High));
        assert!(matches!(critical, AlertSeverity::Critical));
    }
}

#[cfg(test)]
mod monitoring_function_tests {
    use super::*;
    
    #[test]
    fn test_performance_drift_function() {
        let mut vm = VirtualMachine::new();
        let result = performance_drift(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }
    
    #[test]
    fn test_ab_test_function() {
        let mut vm = VirtualMachine::new();
        let result = ab_test(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("test_id_123".to_string()));
    }
    
    #[test]
    fn test_feedback_loop_function() {
        let mut vm = VirtualMachine::new();
        let result = feedback_loop(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
    
    #[test]
    fn test_model_health_function() {
        let mut vm = VirtualMachine::new();
        let result = model_health(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("healthy".to_string()));
    }
    
    #[test]
    fn test_alert_config_function() {
        let mut vm = VirtualMachine::new();
        let result = alert_config(&mut vm, &[]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }
}

#[cfg(test)]
mod metric_value_tests {
    use super::*;
    
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
    fn test_metric_value_ordering() {
        let metric1 = MetricValue {
            value: 0.9,
            step: 50,
            timestamp: Utc::now(),
        };
        
        let metric2 = MetricValue {
            value: 0.95,
            step: 100,
            timestamp: Utc::now(),
        };
        
        assert!(metric1.step < metric2.step);
        assert!(metric1.value < metric2.value);
    }
}

#[cfg(test)]
mod drift_result_tests {
    use super::*;
    
    #[test]
    fn test_drift_result_creation() {
        let drift_result = DriftResult {
            method: "kolmogorov_smirnov".to_string(),
            drift_detected: true,
            drift_score: 0.15,
            threshold: 0.1,
            feature_drifts: HashMap::from([
                ("feature1".to_string(), 0.12),
                ("feature2".to_string(), 0.18),
                ("feature3".to_string(), 0.05),
            ]),
            timestamp: Utc::now(),
        };
        
        assert_eq!(drift_result.method, "kolmogorov_smirnov");
        assert!(drift_result.drift_detected);
        assert_eq!(drift_result.drift_score, 0.15);
        assert_eq!(drift_result.threshold, 0.1);
        assert_eq!(drift_result.feature_drifts.len(), 3);
        
        // Check that drift_score exceeds threshold
        assert!(drift_result.drift_score > drift_result.threshold);
        
        // Check individual feature drifts
        assert_eq!(drift_result.feature_drifts.get("feature1"), Some(&0.12));
        assert_eq!(drift_result.feature_drifts.get("feature2"), Some(&0.18));
        assert_eq!(drift_result.feature_drifts.get("feature3"), Some(&0.05));
    }
    
    #[test]
    fn test_drift_result_no_drift_detected() {
        let drift_result = DriftResult {
            method: "ks_test".to_string(),
            drift_detected: false,
            drift_score: 0.05,
            threshold: 0.1,
            feature_drifts: HashMap::from([
                ("stable_feature".to_string(), 0.05),
            ]),
            timestamp: Utc::now(),
        };
        
        assert!(!drift_result.drift_detected);
        assert!(drift_result.drift_score < drift_result.threshold);
    }
}