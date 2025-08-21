//! Production Observability & Monitoring Module
//!
//! This module provides comprehensive observability capabilities for production systems,
//! including metrics collection, distributed tracing, health monitoring, performance profiling,
//! and debugging instrumentation.
//!
//! # Phase 16B: Production Observability & Monitoring Implementation
//!
//! ## Core Architecture
//! - **Telemetry Collection**: Metrics, logs, and distributed tracing
//! - **Monitoring & Alerting**: Service health, SLO tracking, and incident management
//! - **Performance Profiling**: CPU/memory profiling, latency tracking, resource monitoring
//! - **Code Instrumentation**: Debug tools, call stack analysis, deadlock detection
//!
//! ## Foreign Object Pattern
//! All observability systems are implemented as Foreign objects following the established
//! pattern to maintain VM simplicity and thread safety.
//!
//! ## Technology Integration
//! - OpenTelemetry for metrics, traces, and logs
//! - Prometheus for metrics collection and alerting
//! - pprof for CPU/memory profiling
//! - System monitoring via sysinfo
//! - Notification channels (email, Slack, webhooks)
//!
//! ## Performance Characteristics
//! - Minimal overhead (< 1% CPU, < 100MB memory)
//! - High reliability and fault tolerance
//! - Asynchronous data collection and export
//! - Production-grade observability with comprehensive coverage

pub mod telemetry;
pub mod monitoring;
pub mod profiling;
pub mod instrumentation;

// Re-export all observability functions for easier access
pub use telemetry::*;
pub use monitoring::*;
pub use profiling::*;
pub use instrumentation::*;

use crate::vm::Value;
use crate::foreign::ForeignError;
use std::collections::HashMap;

/// Register all observability functions with the VM
pub fn register_observability_functions() -> HashMap<String, fn(&[Value]) -> Result<Value, ForeignError>> {
    let mut functions = HashMap::new();

    // Telemetry Collection Functions (12 functions)
    functions.insert("MetricsCollector".to_string(), telemetry::metrics_collector as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("MetricIncrement".to_string(), telemetry::metric_increment as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("MetricGauge".to_string(), telemetry::metric_gauge as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("MetricHistogram".to_string(), telemetry::metric_histogram as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("LogAggregator".to_string(), telemetry::log_aggregator as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("LogEvent".to_string(), telemetry::log_event as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("DistributedTracing".to_string(), telemetry::distributed_tracing as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("TraceSpan".to_string(), telemetry::trace_span as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("SpanEvent".to_string(), telemetry::span_event as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("TelemetryExport".to_string(), telemetry::telemetry_export as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("MetricQuery".to_string(), telemetry::metric_query as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("OpenTelemetry".to_string(), telemetry::open_telemetry as fn(&[Value]) -> Result<Value, ForeignError>);

    // Monitoring & Alerting Functions (10 functions)
    functions.insert("ServiceHealth".to_string(), monitoring::service_health as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("HealthCheck".to_string(), monitoring::health_check as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("AlertManager".to_string(), monitoring::alert_manager as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("AlertRule".to_string(), monitoring::alert_rule as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("NotificationChannel".to_string(), monitoring::notification_channel as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("SLOTracker".to_string(), monitoring::slo_tracker as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("Heartbeat".to_string(), monitoring::heartbeat as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("ServiceDependency".to_string(), monitoring::service_dependency as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("IncidentManagement".to_string(), monitoring::incident_management as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("StatusPage".to_string(), monitoring::status_page as fn(&[Value]) -> Result<Value, ForeignError>);

    // Performance Profiling Functions (8 functions)
    functions.insert("ProfilerStart".to_string(), profiling::profiler_start as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("ProfilerStop".to_string(), profiling::profiler_stop as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("MemoryProfiler".to_string(), profiling::memory_profiler as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("CPUProfiler".to_string(), profiling::cpu_profiler as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("LatencyTracker".to_string(), profiling::latency_tracker as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("ThroughputMonitor".to_string(), profiling::throughput_monitor as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("ResourceMonitor".to_string(), profiling::resource_monitor as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("PerformanceBaseline".to_string(), profiling::performance_baseline as fn(&[Value]) -> Result<Value, ForeignError>);

    // Code Instrumentation Functions (5 functions)
    functions.insert("CallStack".to_string(), instrumentation::call_stack as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("HeapDump".to_string(), instrumentation::heap_dump as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("ThreadDump".to_string(), instrumentation::thread_dump as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("DeadlockDetector".to_string(), instrumentation::deadlock_detector as fn(&[Value]) -> Result<Value, ForeignError>);
    functions.insert("DebugBreakpoint".to_string(), instrumentation::debug_breakpoint as fn(&[Value]) -> Result<Value, ForeignError>);

    functions
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    #[test]
    fn test_function_registration() {
        let functions = register_observability_functions();
        
        // Verify all 35 functions are registered
        assert_eq!(functions.len(), 35);
        
        // Test telemetry functions
        assert!(functions.contains_key("MetricsCollector"));
        assert!(functions.contains_key("MetricIncrement"));
        assert!(functions.contains_key("LogAggregator"));
        assert!(functions.contains_key("DistributedTracing"));
        
        // Test monitoring functions
        assert!(functions.contains_key("ServiceHealth"));
        assert!(functions.contains_key("HealthCheck"));
        assert!(functions.contains_key("AlertManager"));
        assert!(functions.contains_key("SLOTracker"));
        
        // Test profiling functions
        assert!(functions.contains_key("ProfilerStart"));
        assert!(functions.contains_key("MemoryProfiler"));
        assert!(functions.contains_key("LatencyTracker"));
        assert!(functions.contains_key("ResourceMonitor"));
        
        // Test instrumentation functions
        assert!(functions.contains_key("CallStack"));
        assert!(functions.contains_key("HeapDump"));
        assert!(functions.contains_key("ThreadDump"));
        assert!(functions.contains_key("DeadlockDetector"));
    }

    #[test]
    fn test_metrics_collector_creation() {
        let functions = register_observability_functions();
        let metrics_collector = functions.get("MetricsCollector").unwrap();
        
        let args = vec![
            Value::String("http_requests".to_string()),
            Value::String("counter".to_string()),
            Value::List(vec![]),
        ];
        
        let result = metrics_collector(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(_collector)) = result {
            // Successfully created a metrics collector
        } else {
            panic!("Expected LyObj containing MetricsCollector");
        }
    }

    #[test]
    fn test_service_health_creation() {
        let functions = register_observability_functions();
        let service_health = functions.get("ServiceHealth").unwrap();
        
        let args = vec![
            Value::String("user-service".to_string()),
            Value::List(vec![]),
            Value::List(vec![]),
        ];
        
        let result = service_health(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(_health)) = result {
            // Successfully created a service health monitor
        } else {
            panic!("Expected LyObj containing ServiceHealth");
        }
    }

    #[test]
    fn test_latency_tracker_creation() {
        let functions = register_observability_functions();
        let latency_tracker = functions.get("LatencyTracker").unwrap();
        
        let args = vec![
            Value::String("database_query".to_string()),
            Value::List(vec![
                Value::Real(50.0),
                Value::Real(95.0),
                Value::Real(99.0),
            ]),
            Value::List(vec![
                Value::Real(1.0),
                Value::Real(10.0),
                Value::Real(100.0),
            ]),
        ];
        
        let result = latency_tracker(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(_tracker)) = result {
            // Successfully created a latency tracker
        } else {
            panic!("Expected LyObj containing LatencyTracker");
        }
    }

    #[test]
    fn test_call_stack_creation() {
        let functions = register_observability_functions();
        let call_stack = functions.get("CallStack").unwrap();
        
        let args = vec![
            Value::Integer(1),
            Value::Integer(32),
            Value::Boolean(true),
        ];
        
        let result = call_stack(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(_stack)) = result {
            // Successfully created a call stack
        } else {
            panic!("Expected LyObj containing CallStack");
        }
    }

    #[test]
    fn test_error_handling() {
        let functions = register_observability_functions();
        let metrics_collector = functions.get("MetricsCollector").unwrap();
        
        // Test with insufficient arguments
        let args = vec![Value::String("incomplete".to_string())];
        let result = metrics_collector(&args);
        assert!(result.is_err());
        
        if let Err(ForeignError::ArgumentError { expected, actual }) = result {
            assert_eq!(expected, 2);
            assert_eq!(actual, 1);
        } else {
            panic!("Expected ArgumentError");
        }
    }
}