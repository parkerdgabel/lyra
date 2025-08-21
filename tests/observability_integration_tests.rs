//! Comprehensive Integration Tests for Phase 16B: Production Observability & Monitoring
//!
//! This test suite validates all observability functions across telemetry collection,
//! monitoring & alerting, performance profiling, and code instrumentation.

use lyra::vm::{Value, VM};
use lyra::stdlib::StandardLibrary;
use std::collections::HashMap;

#[cfg(test)]
mod telemetry_tests {
    use super::*;

    #[test]
    fn test_metrics_collector_creation_and_usage() {
        let stdlib = StandardLibrary::new();
        let metrics_collector = stdlib.get_function("MetricsCollector").unwrap();
        
        // Test counter metric creation
        let args = vec![
            Value::String("http_requests".to_string()),
            Value::String("counter".to_string()),
            Value::List(vec![
                Value::List(vec![Value::String("endpoint".to_string()), Value::String("/api".to_string())]),
                Value::List(vec![Value::String("method".to_string()), Value::String("GET".to_string())]),
            ]),
        ];
        
        let result = metrics_collector(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(collector)) = result {
            // Test incrementing the counter
            let increment_result = collector.call_method("increment", &[Value::Integer(1)]);
            assert!(increment_result.is_ok());
            assert_eq!(increment_result.unwrap(), Value::Boolean(true));
            
            // Test getting metric name
            let name_result = collector.call_method("name", &[]);
            assert!(name_result.is_ok());
            assert_eq!(name_result.unwrap(), Value::String("http_requests".to_string()));
        } else {
            panic!("Expected LyObj containing MetricsCollector");
        }
    }

    #[test]
    fn test_gauge_metric_operations() {
        let stdlib = StandardLibrary::new();
        let metrics_collector = stdlib.get_function("MetricsCollector").unwrap();
        
        // Test gauge metric creation
        let args = vec![
            Value::String("memory_usage".to_string()),
            Value::String("gauge".to_string()),
            Value::List(vec![]),
        ];
        
        let result = metrics_collector(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(gauge)) = result {
            // Test setting gauge value
            let set_result = gauge.call_method("set", &[Value::Real(1024.0)]);
            assert!(set_result.is_ok());
            assert_eq!(set_result.unwrap(), Value::Boolean(true));
            
            // Test gauge type
            let type_result = gauge.call_method("type", &[]);
            assert!(type_result.is_ok());
            assert_eq!(type_result.unwrap(), Value::String("Gauge".to_string()));
        } else {
            panic!("Expected LyObj containing gauge MetricsCollector");
        }
    }

    #[test]
    fn test_histogram_metric_operations() {
        let stdlib = StandardLibrary::new();
        let metrics_collector = stdlib.get_function("MetricsCollector").unwrap();
        
        // Test histogram metric creation
        let args = vec![
            Value::String("request_duration".to_string()),
            Value::String("histogram".to_string()),
            Value::List(vec![]),
        ];
        
        let result = metrics_collector(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(histogram)) = result {
            // Test recording histogram value
            let record_result = histogram.call_method("record", &[Value::Real(123.45)]);
            assert!(record_result.is_ok());
            assert_eq!(record_result.unwrap(), Value::Boolean(true));
            
            // Test histogram type
            let type_result = histogram.call_method("type", &[]);
            assert!(type_result.is_ok());
            assert_eq!(type_result.unwrap(), Value::String("Histogram".to_string()));
        } else {
            panic!("Expected LyObj containing histogram MetricsCollector");
        }
    }

    #[test]
    fn test_log_aggregator_functionality() {
        let stdlib = StandardLibrary::new();
        let log_aggregator = stdlib.get_function("LogAggregator").unwrap();
        
        // Test log aggregator creation
        let args = vec![
            Value::String("info".to_string()),
            Value::String("json".to_string()),
            Value::List(vec![Value::String("stdout".to_string())]),
            Value::List(vec![]),
        ];
        
        let result = log_aggregator(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(aggregator)) = result {
            // Test logging an event
            let log_result = aggregator.call_method("log", &[
                Value::String("info".to_string()),
                Value::String("Test log message".to_string()),
                Value::List(vec![
                    Value::List(vec![Value::String("service".to_string()), Value::String("test".to_string())]),
                ]),
            ]);
            assert!(log_result.is_ok());
            assert_eq!(log_result.unwrap(), Value::Boolean(true));
            
            // Test getting log count
            let count_result = aggregator.call_method("count", &[]);
            assert!(count_result.is_ok());
            assert_eq!(count_result.unwrap(), Value::Integer(1));
            
            // Test filtering logs
            let filter_result = aggregator.call_method("filter", &[Value::String("info".to_string())]);
            assert!(filter_result.is_ok());
            if let Ok(Value::List(logs)) = filter_result {
                assert_eq!(logs.len(), 1);
            }
        } else {
            panic!("Expected LyObj containing LogAggregator");
        }
    }

    #[test]
    fn test_distributed_tracing_workflow() {
        let stdlib = StandardLibrary::new();
        let distributed_tracing = stdlib.get_function("DistributedTracing").unwrap();
        
        // Test tracer creation
        let args = vec![
            Value::String("user-service".to_string()),
            Value::List(vec![
                Value::List(vec![Value::String("version".to_string()), Value::String("1.0".to_string())]),
            ]),
        ];
        
        let result = distributed_tracing(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(tracer)) = result {
            // Test creating a span
            let span_result = tracer.call_method("createSpan", &[Value::String("process_request".to_string())]);
            assert!(span_result.is_ok());
            
            if let Ok(Value::LyObj(span)) = span_result {
                // Test adding event to span
                let event_result = span.call_method("addEvent", &[
                    Value::String("validation_complete".to_string()),
                    Value::List(vec![
                        Value::List(vec![Value::String("duration_ms".to_string()), Value::String("45".to_string())]),
                    ]),
                ]);
                assert!(event_result.is_ok());
                
                // Test setting span attribute
                let attr_result = span.call_method("setAttribute", &[
                    Value::String("user_id".to_string()),
                    Value::String("12345".to_string()),
                ]);
                assert!(attr_result.is_ok());
                
                // Test finishing span
                let finish_result = span.call_method("finish", &[]);
                assert!(finish_result.is_ok());
            }
        } else {
            panic!("Expected LyObj containing DistributedTracing");
        }
    }

    #[test]
    fn test_telemetry_export_functionality() {
        let stdlib = StandardLibrary::new();
        let telemetry_export = stdlib.get_function("TelemetryExport").unwrap();
        
        // Test exporter creation
        let args = vec![
            Value::List(vec![Value::String("metrics".to_string()), Value::String("traces".to_string())]),
            Value::String("otlp".to_string()),
            Value::String("http://localhost:4317".to_string()),
        ];
        
        let result = telemetry_export(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(exporter)) = result {
            // Test export operation
            let export_result = exporter.call_method("export", &[]);
            assert!(export_result.is_ok());
            assert_eq!(export_result.unwrap(), Value::Boolean(true));
            
            // Test getting status
            let status_result = exporter.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::List(status_list)) = status_result {
                assert!(!status_list.is_empty());
            }
        } else {
            panic!("Expected LyObj containing TelemetryExporter");
        }
    }
}

#[cfg(test)]
mod monitoring_tests {
    use super::*;

    #[test]
    fn test_service_health_monitoring() {
        let stdlib = StandardLibrary::new();
        let service_health = stdlib.get_function("ServiceHealth").unwrap();
        
        // Test service health creation
        let args = vec![
            Value::String("user-service".to_string()),
            Value::List(vec![]),
            Value::List(vec![
                Value::List(vec![Value::String("availability".to_string()), Value::String("99.9".to_string())]),
            ]),
        ];
        
        let result = service_health(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(health)) = result {
            // Test running health checks
            let check_result = health.call_method("runChecks", &[]);
            assert!(check_result.is_ok());
            assert_eq!(check_result.unwrap(), Value::Boolean(true));
            
            // Test getting status
            let status_result = health.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::String(status_json)) = status_result {
                assert!(!status_json.is_empty());
                assert!(status_json.contains("overall_health"));
            }
        } else {
            panic!("Expected LyObj containing ServiceHealth");
        }
    }

    #[test]
    fn test_health_check_execution() {
        let stdlib = StandardLibrary::new();
        let health_check = stdlib.get_function("HealthCheck").unwrap();
        
        // Test health check creation
        let args = vec![
            Value::String("http://localhost:8080/health".to_string()),
            Value::Integer(5),
            Value::Integer(200),
        ];
        
        let result = health_check(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(check)) = result {
            // Test executing health check
            let exec_result = check.call_method("execute", &[]);
            assert!(exec_result.is_ok());
            if let Ok(Value::String(result_json)) = exec_result {
                assert!(!result_json.is_empty());
                assert!(result_json.contains("timestamp"));
            }
            
            // Test getting endpoint
            let endpoint_result = check.call_method("endpoint", &[]);
            assert!(endpoint_result.is_ok());
            assert_eq!(endpoint_result.unwrap(), Value::String("http://localhost:8080/health".to_string()));
        } else {
            panic!("Expected LyObj containing HealthCheck");
        }
    }

    #[test]
    fn test_alert_manager_functionality() {
        let stdlib = StandardLibrary::new();
        let alert_manager = stdlib.get_function("AlertManager").unwrap();
        
        // Test alert manager creation
        let args = vec![
            Value::List(vec![]),
            Value::List(vec![]),
            Value::List(vec![
                Value::List(vec![Value::String("escalation_time".to_string()), Value::String("900".to_string())]),
            ]),
        ];
        
        let result = alert_manager(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(manager)) = result {
            // Test firing an alert
            let fire_result = manager.call_method("fireAlert", &[
                Value::String("high_cpu_usage".to_string()),
                Value::String("CPU usage above 80%".to_string()),
            ]);
            assert!(fire_result.is_ok());
            assert_eq!(fire_result.unwrap(), Value::Boolean(true));
            
            // Test getting active alerts
            let alerts_result = manager.call_method("activeAlerts", &[]);
            assert!(alerts_result.is_ok());
            if let Ok(Value::List(alerts)) = alerts_result {
                assert_eq!(alerts.len(), 1);
            }
            
            // Test resolving alert
            let resolve_result = manager.call_method("resolveAlert", &[Value::String("high_cpu_usage".to_string())]);
            assert!(resolve_result.is_ok());
            assert_eq!(resolve_result.unwrap(), Value::Boolean(true));
        } else {
            panic!("Expected LyObj containing AlertManager");
        }
    }

    #[test]
    fn test_slo_tracker_operations() {
        let stdlib = StandardLibrary::new();
        let slo_tracker = stdlib.get_function("SLOTracker").unwrap();
        
        // Test SLO tracker creation
        let args = vec![
            Value::String("user-service".to_string()),
            Value::List(vec![]),
            Value::Real(0.1), // 10% error budget
        ];
        
        let result = slo_tracker(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(tracker)) = result {
            // Test recording measurements
            let record_result = tracker.call_method("recordMeasurement", &[
                Value::String("availability".to_string()),
                Value::Real(99.95),
            ]);
            // This should fail because no SLO is defined, which is expected
            assert!(record_result.is_err());
            
            // Test getting error budget
            let budget_result = tracker.call_method("errorBudget", &[]);
            assert!(budget_result.is_ok());
            if let Ok(Value::Real(budget)) = budget_result {
                assert!(budget >= 0.0 && budget <= 1.0);
            }
        } else {
            panic!("Expected LyObj containing SLOTracker");
        }
    }

    #[test]
    fn test_heartbeat_monitoring() {
        let stdlib = StandardLibrary::new();
        let heartbeat = stdlib.get_function("Heartbeat").unwrap();
        
        // Test heartbeat creation
        let args = vec![
            Value::String("worker-service".to_string()),
            Value::Integer(30), // 30 second interval
            Value::Integer(60), // 60 second timeout
        ];
        
        let result = heartbeat(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(heartbeat_monitor)) = result {
            // Test ping operation
            let ping_result = heartbeat_monitor.call_method("ping", &[]);
            assert!(ping_result.is_ok());
            assert_eq!(ping_result.unwrap(), Value::Boolean(true));
            
            // Test getting status
            let status_result = heartbeat_monitor.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::List(status_list)) = status_result {
                assert!(!status_list.is_empty());
            }
            
            // Test missed count (should be 0 after ping)
            let missed_result = heartbeat_monitor.call_method("missedCount", &[]);
            assert!(missed_result.is_ok());
            assert_eq!(missed_result.unwrap(), Value::Integer(0));
        } else {
            panic!("Expected LyObj containing Heartbeat");
        }
    }

    #[test]
    fn test_alert_rule_creation() {
        let stdlib = StandardLibrary::new();
        let alert_rule = stdlib.get_function("AlertRule").unwrap();
        
        // Test alert rule creation
        let args = vec![
            Value::String("cpu_usage > 80".to_string()),
            Value::Real(80.0),
            Value::Integer(300), // 5 minutes
            Value::String("warning".to_string()),
        ];
        
        let result = alert_rule(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::String(rule_json)) = result {
            assert!(!rule_json.is_empty());
            assert!(rule_json.contains("condition"));
            assert!(rule_json.contains("threshold"));
            assert!(rule_json.contains("severity"));
        } else {
            panic!("Expected String containing alert rule JSON");
        }
    }

    #[test]
    fn test_notification_channel_creation() {
        let stdlib = StandardLibrary::new();
        let notification_channel = stdlib.get_function("NotificationChannel").unwrap();
        
        // Test notification channel creation
        let args = vec![
            Value::String("slack".to_string()),
            Value::List(vec![
                Value::List(vec![Value::String("webhook".to_string()), Value::String("https://hooks.slack.com/...".to_string())]),
            ]),
            Value::List(vec![
                Value::List(vec![Value::String("template".to_string()), Value::String("Alert: {{message}}".to_string())]),
            ]),
        ];
        
        let result = notification_channel(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::String(channel_json)) = result {
            assert!(!channel_json.is_empty());
            assert!(channel_json.contains("channel_type"));
            assert!(channel_json.contains("slack"));
        } else {
            panic!("Expected String containing notification channel JSON");
        }
    }
}

#[cfg(test)]
mod profiling_tests {
    use super::*;

    #[test]
    fn test_cpu_profiler_lifecycle() {
        let stdlib = StandardLibrary::new();
        let profiler_start = stdlib.get_function("ProfilerStart").unwrap();
        
        // Test CPU profiler creation
        let args = vec![
            Value::String("cpu".to_string()),
            Value::Integer(10), // 10 seconds
            Value::Integer(100), // 100Hz sampling
        ];
        
        let result = profiler_start(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(profiler)) = result {
            // Test starting profiler
            let start_result = profiler.call_method("start", &[]);
            assert!(start_result.is_ok());
            assert_eq!(start_result.unwrap(), Value::Boolean(true));
            
            // Test adding sample
            let sample_result = profiler.call_method("addSample", &[]);
            assert!(sample_result.is_ok());
            
            // Test getting status
            let status_result = profiler.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::List(status_list)) = status_result {
                assert!(!status_list.is_empty());
            }
            
            // Test stopping profiler
            let stop_result = profiler.call_method("stop", &[]);
            assert!(stop_result.is_ok());
            if let Ok(Value::String(profile_data)) = stop_result {
                assert!(!profile_data.is_empty());
                assert!(profile_data.contains("duration"));
            }
        } else {
            panic!("Expected LyObj containing CPUProfiler");
        }
    }

    #[test]
    fn test_memory_profiler_operations() {
        let stdlib = StandardLibrary::new();
        let memory_profiler = stdlib.get_function("MemoryProfiler").unwrap();
        
        // Test memory profiler creation
        let args = vec![
            Value::Integer(std::process::id() as i64),
            Value::Integer(1), // 1 second interval
            Value::Boolean(true), // heap analysis
        ];
        
        let result = memory_profiler(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(profiler)) = result {
            // Test taking snapshot
            let snapshot_result = profiler.call_method("takeSnapshot", &[]);
            assert!(snapshot_result.is_ok());
            if let Ok(Value::String(snapshot_json)) = snapshot_result {
                assert!(!snapshot_json.is_empty());
                assert!(snapshot_json.contains("timestamp"));
                assert!(snapshot_json.contains("total_memory"));
            }
            
            // Test getting snapshots
            let snapshots_result = profiler.call_method("snapshots", &[]);
            assert!(snapshots_result.is_ok());
            if let Ok(Value::List(snapshots)) = snapshots_result {
                assert_eq!(snapshots.len(), 1);
            }
        } else {
            panic!("Expected LyObj containing MemoryProfiler");
        }
    }

    #[test]
    fn test_latency_tracker_workflow() {
        let stdlib = StandardLibrary::new();
        let latency_tracker = stdlib.get_function("LatencyTracker").unwrap();
        
        // Test latency tracker creation
        let args = vec![
            Value::String("database_query".to_string()),
            Value::List(vec![Value::Real(50.0), Value::Real(95.0), Value::Real(99.0)]),
            Value::List(vec![Value::Real(1.0), Value::Real(10.0), Value::Real(100.0)]),
        ];
        
        let result = latency_tracker(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(tracker)) = result {
            // Test starting timer
            let start_result = tracker.call_method("startTimer", &[Value::String("query_1".to_string())]);
            assert!(start_result.is_ok());
            if let Ok(Value::String(timer_id)) = start_result {
                assert_eq!(timer_id, "query_1");
                
                // Test ending timer
                let end_result = tracker.call_method("endTimer", &[
                    Value::String("query_1".to_string()),
                    Value::Boolean(true),
                ]);
                assert!(end_result.is_ok());
                if let Ok(Value::Real(latency)) = end_result {
                    assert!(latency >= 0.0);
                }
            }
            
            // Test recording latency directly
            let record_result = tracker.call_method("record", &[
                Value::Real(25.5),
                Value::Boolean(true),
            ]);
            assert!(record_result.is_ok());
            
            // Test getting summary
            let summary_result = tracker.call_method("summary", &[]);
            assert!(summary_result.is_ok());
            if let Ok(Value::List(summary)) = summary_result {
                assert!(!summary.is_empty());
            }
        } else {
            panic!("Expected LyObj containing LatencyTracker");
        }
    }

    #[test]
    fn test_throughput_monitor_operations() {
        let stdlib = StandardLibrary::new();
        let throughput_monitor = stdlib.get_function("ThroughputMonitor").unwrap();
        
        // Test throughput monitor creation
        let args = vec![
            Value::String("/api/users".to_string()),
            Value::Integer(60), // 60 second window
            Value::String("average".to_string()),
        ];
        
        let result = throughput_monitor(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(monitor)) = result {
            // Test recording requests
            for i in 0..5 {
                let record_result = monitor.call_method("recordRequest", &[
                    Value::Boolean(true), // successful
                    Value::Integer(1024 + i * 100), // response size
                ]);
                assert!(record_result.is_ok());
            }
            
            // Test getting throughput metrics
            let throughput_result = monitor.call_method("throughput", &[]);
            assert!(throughput_result.is_ok());
            if let Ok(Value::List(metrics)) = throughput_result {
                assert!(!metrics.is_empty());
            }
            
            // Test reset
            let reset_result = monitor.call_method("reset", &[]);
            assert!(reset_result.is_ok());
            assert_eq!(reset_result.unwrap(), Value::Boolean(true));
        } else {
            panic!("Expected LyObj containing ThroughputMonitor");
        }
    }

    #[test]
    fn test_resource_monitor_system_metrics() {
        let stdlib = StandardLibrary::new();
        let resource_monitor = stdlib.get_function("ResourceMonitor").unwrap();
        
        // Test resource monitor creation
        let args = vec![
            Value::String("test-system".to_string()),
            Value::List(vec![
                Value::String("cpu".to_string()),
                Value::String("memory".to_string()),
                Value::String("disk".to_string()),
            ]),
            Value::List(vec![]),
        ];
        
        let result = resource_monitor(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(monitor)) = result {
            // Test collecting metrics
            let collect_result = monitor.call_method("collect", &[]);
            assert!(collect_result.is_ok());
            if let Ok(Value::String(reading_json)) = collect_result {
                assert!(!reading_json.is_empty());
                assert!(reading_json.contains("timestamp"));
                assert!(reading_json.contains("cpu_usage"));
            }
            
            // Test getting latest reading
            let latest_result = monitor.call_method("latest", &[]);
            assert!(latest_result.is_ok());
            
            // Test getting history
            let history_result = monitor.call_method("history", &[Value::Integer(10)]);
            assert!(history_result.is_ok());
            if let Ok(Value::List(history)) = history_result {
                assert_eq!(history.len(), 1);
            }
        } else {
            panic!("Expected LyObj containing ResourceMonitor");
        }
    }

    #[test]
    fn test_performance_baseline_calculation() {
        let stdlib = StandardLibrary::new();
        let performance_baseline = stdlib.get_function("PerformanceBaseline").unwrap();
        
        // Test performance baseline creation
        let args = vec![
            Value::List(vec![
                Value::String("cpu".to_string()),
                Value::String("memory".to_string()),
                Value::String("latency".to_string()),
            ]),
            Value::List(vec![]),
            Value::Boolean(true), // anomaly detection
        ];
        
        let result = performance_baseline(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::List(baseline_data)) = result {
            assert!(!baseline_data.is_empty());
            // Should contain baseline metrics
            let contains_cpu = baseline_data.iter().any(|item| {
                if let Value::List(pair) = item {
                    if let (Some(Value::String(key)), _) = (pair.get(0), pair.get(1)) {
                        key.contains("baseline_cpu")
                    } else {
                        false
                    }
                } else {
                    false
                }
            });
            assert!(contains_cpu);
        } else {
            panic!("Expected List containing baseline data");
        }
    }
}

#[cfg(test)]
mod instrumentation_tests {
    use super::*;

    #[test]
    fn test_call_stack_capture() {
        let stdlib = StandardLibrary::new();
        let call_stack = stdlib.get_function("CallStack").unwrap();
        
        // Test call stack creation
        let args = vec![
            Value::Integer(1), // thread ID
            Value::Integer(32), // depth
            Value::Boolean(true), // symbolication
        ];
        
        let result = call_stack(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(stack)) = result {
            // Test capturing stack
            let capture_result = stack.call_method("capture", &[]);
            assert!(capture_result.is_ok());
            if let Ok(Value::List(frames)) = capture_result {
                assert!(!frames.is_empty());
            }
            
            // Test formatting
            let format_result = stack.call_method("format", &[Value::String("text".to_string())]);
            assert!(format_result.is_ok());
            if let Ok(Value::String(formatted)) = format_result {
                assert!(!formatted.is_empty());
            }
            
            // Test getting depth
            let depth_result = stack.call_method("depth", &[]);
            assert!(depth_result.is_ok());
            
            // Test getting thread ID
            let thread_result = stack.call_method("threadId", &[]);
            assert!(thread_result.is_ok());
            assert_eq!(thread_result.unwrap(), Value::Integer(1));
        } else {
            panic!("Expected LyObj containing CallStack");
        }
    }

    #[test]
    fn test_heap_dump_analysis() {
        let stdlib = StandardLibrary::new();
        let heap_dump = stdlib.get_function("HeapDump").unwrap();
        
        // Test heap dump creation
        let args = vec![
            Value::String("json".to_string()),
            Value::Boolean(false), // no compression
            Value::Boolean(true), // analysis
        ];
        
        let result = heap_dump(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(dump)) = result {
            // Test capturing heap dump
            let capture_result = dump.call_method("capture", &[]);
            assert!(capture_result.is_ok());
            if let Ok(Value::String(analysis_json)) = capture_result {
                assert!(!analysis_json.is_empty());
                assert!(analysis_json.contains("total_size"));
            }
            
            // Test analysis
            let analyze_result = dump.call_method("analyze", &[]);
            assert!(analyze_result.is_ok());
            if let Ok(Value::String(report)) = analyze_result {
                assert!(!report.is_empty());
                assert!(report.contains("Heap Analysis Report"));
            }
            
            // Test getting size
            let size_result = dump.call_method("size", &[]);
            assert!(size_result.is_ok());
            
            // Test getting object count
            let count_result = dump.call_method("objectCount", &[]);
            assert!(count_result.is_ok());
        } else {
            panic!("Expected LyObj containing HeapDump");
        }
    }

    #[test]
    fn test_thread_dump_analysis() {
        let stdlib = StandardLibrary::new();
        let thread_dump = stdlib.get_function("ThreadDump").unwrap();
        
        // Test thread dump creation
        let args = vec![
            Value::Integer(std::process::id() as i64),
            Value::Boolean(true), // analysis
            Value::Boolean(true), // deadlock detection
        ];
        
        let result = thread_dump(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(dump)) = result {
            // Test capturing thread dump
            let capture_result = dump.call_method("capture", &[]);
            assert!(capture_result.is_ok());
            if let Ok(Value::List(threads)) = capture_result {
                assert!(!threads.is_empty());
            }
            
            // Test analysis
            let analyze_result = dump.call_method("analyze", &[]);
            assert!(analyze_result.is_ok());
            if let Ok(Value::String(analysis)) = analyze_result {
                assert!(!analysis.is_empty());
                assert!(analysis.contains("Thread Dump Analysis"));
            }
            
            // Test deadlock detection
            let deadlock_result = dump.call_method("deadlocks", &[]);
            assert!(deadlock_result.is_ok());
            if let Ok(Value::List(deadlocks)) = deadlock_result {
                // Should be empty (no deadlocks in test)
                assert!(deadlocks.is_empty());
            }
        } else {
            panic!("Expected LyObj containing ThreadDump");
        }
    }

    #[test]
    fn test_deadlock_detector_functionality() {
        let stdlib = StandardLibrary::new();
        let deadlock_detector = stdlib.get_function("DeadlockDetector").unwrap();
        
        // Test deadlock detector creation
        let args = vec![
            Value::List(vec![
                Value::String("mutex_a".to_string()),
                Value::String("mutex_b".to_string()),
            ]),
            Value::Integer(5000), // 5 second timeout
            Value::String("timeout".to_string()),
        ];
        
        let result = deadlock_detector(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(detector)) = result {
            // Test scanning for deadlocks
            let scan_result = detector.call_method("scan", &[]);
            assert!(scan_result.is_ok());
            if let Ok(Value::List(deadlocks)) = scan_result {
                // Should be empty (no deadlocks detected)
                assert!(deadlocks.is_empty());
            }
            
            // Test adding monitor
            let monitor_result = detector.call_method("addMonitor", &[Value::String("mutex_c".to_string())]);
            assert!(monitor_result.is_ok());
            
            // Test getting status
            let status_result = detector.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::List(status)) = status_result {
                assert!(!status.is_empty());
            }
        } else {
            panic!("Expected LyObj containing DeadlockDetector");
        }
    }

    #[test]
    fn test_debug_breakpoint_functionality() {
        let stdlib = StandardLibrary::new();
        let debug_breakpoint = stdlib.get_function("DebugBreakpoint").unwrap();
        
        // Test breakpoint creation
        let args = vec![
            Value::String("x > 10".to_string()),
            Value::String("log".to_string()),
            Value::Boolean(false), // not temporary
        ];
        
        let result = debug_breakpoint(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(breakpoint)) = result {
            // Test evaluating breakpoint
            let eval_result = breakpoint.call_method("evaluate", &[
                Value::List(vec![
                    Value::List(vec![Value::String("x".to_string()), Value::Integer(15)]),
                ]),
            ]);
            assert!(eval_result.is_ok());
            
            // Test getting hit count
            let hit_result = breakpoint.call_method("hitCount", &[]);
            assert!(hit_result.is_ok());
            
            // Test disabling breakpoint
            let disable_result = breakpoint.call_method("disable", &[]);
            assert!(disable_result.is_ok());
            assert_eq!(disable_result.unwrap(), Value::Boolean(true));
            
            // Test enabling breakpoint
            let enable_result = breakpoint.call_method("enable", &[]);
            assert!(enable_result.is_ok());
            assert_eq!(enable_result.unwrap(), Value::Boolean(true));
            
            // Test getting status
            let status_result = breakpoint.call_method("status", &[]);
            assert!(status_result.is_ok());
            if let Ok(Value::List(status)) = status_result {
                assert!(!status.is_empty());
            }
        } else {
            panic!("Expected LyObj containing DebugBreakpoint");
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_observability_workflow() {
        let stdlib = StandardLibrary::new();
        
        // Create metrics collector
        let metrics_collector = stdlib.get_function("MetricsCollector").unwrap();
        let counter_result = metrics_collector(&[
            Value::String("requests_total".to_string()),
            Value::String("counter".to_string()),
            Value::List(vec![]),
        ]);
        assert!(counter_result.is_ok());
        
        // Create service health monitor
        let service_health = stdlib.get_function("ServiceHealth").unwrap();
        let health_result = service_health(&[
            Value::String("api-service".to_string()),
            Value::List(vec![]),
            Value::List(vec![]),
        ]);
        assert!(health_result.is_ok());
        
        // Create latency tracker
        let latency_tracker = stdlib.get_function("LatencyTracker").unwrap();
        let tracker_result = latency_tracker(&[
            Value::String("api_requests".to_string()),
            Value::List(vec![Value::Real(95.0), Value::Real(99.0)]),
            Value::List(vec![]),
        ]);
        assert!(tracker_result.is_ok());
        
        // Create distributed tracer
        let distributed_tracing = stdlib.get_function("DistributedTracing").unwrap();
        let tracer_result = distributed_tracing(&[
            Value::String("api-service".to_string()),
            Value::List(vec![]),
        ]);
        assert!(tracer_result.is_ok());
        
        // All components should be successfully created
        assert!(counter_result.is_ok());
        assert!(health_result.is_ok());
        assert!(tracker_result.is_ok());
        assert!(tracer_result.is_ok());
    }

    #[test]
    fn test_observability_error_handling() {
        let stdlib = StandardLibrary::new();
        
        // Test metrics collector with invalid arguments
        let metrics_collector = stdlib.get_function("MetricsCollector").unwrap();
        let invalid_result = metrics_collector(&[Value::String("incomplete".to_string())]);
        assert!(invalid_result.is_err());
        
        // Test health check with missing endpoint
        let health_check = stdlib.get_function("HealthCheck").unwrap();
        let invalid_health = health_check(&[]);
        assert!(invalid_health.is_err());
        
        // Test latency tracker with missing operation name
        let latency_tracker = stdlib.get_function("LatencyTracker").unwrap();
        let invalid_tracker = latency_tracker(&[]);
        assert!(invalid_tracker.is_err());
    }

    #[test]
    fn test_observability_function_count() {
        let stdlib = StandardLibrary::new();
        
        // Count observability functions
        let function_names = stdlib.function_names();
        let observability_functions: Vec<&String> = function_names.iter()
            .filter(|name| {
                name.starts_with("Metric") ||
                name.starts_with("Log") ||
                name.starts_with("Trace") ||
                name.starts_with("Service") ||
                name.starts_with("Health") ||
                name.starts_with("Alert") ||
                name.starts_with("SLO") ||
                name.starts_with("Heartbeat") ||
                name.starts_with("Incident") ||
                name.starts_with("Status") ||
                name.starts_with("Profiler") ||
                name.starts_with("Memory") ||
                name.starts_with("CPU") ||
                name.starts_with("Latency") ||
                name.starts_with("Throughput") ||
                name.starts_with("Resource") ||
                name.starts_with("Performance") ||
                name.starts_with("CallStack") ||
                name.starts_with("HeapDump") ||
                name.starts_with("ThreadDump") ||
                name.starts_with("DeadlockDetector") ||
                name.starts_with("DebugBreakpoint") ||
                name.starts_with("OpenTelemetry") ||
                name.starts_with("TelemetryExport") ||
                name.starts_with("DistributedTracing") ||
                name.starts_with("SpanEvent") ||
                name.starts_with("NotificationChannel")
            })
            .collect();
        
        // Should have all 35 observability functions
        assert!(observability_functions.len() >= 35, "Expected at least 35 observability functions, found {}", observability_functions.len());
    }
}