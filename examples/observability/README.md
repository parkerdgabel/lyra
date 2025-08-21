# Lyra Observability Examples

This directory contains comprehensive examples demonstrating the production observability and monitoring capabilities implemented in Phase 16B.

## Overview

The observability system provides the three pillars of observability:
- **Metrics**: Counters, gauges, and histograms for quantitative monitoring
- **Logs**: Structured logging with context and correlation
- **Traces**: Distributed tracing for request flow analysis

Plus advanced monitoring, alerting, profiling, and debugging capabilities.

## Examples

### 1. Basic Telemetry (`basic_telemetry.lyra`)

Demonstrates fundamental telemetry collection:
- Creating metrics collectors (counter, gauge, histogram)
- Setting up structured logging
- Implementing distributed tracing
- Exporting telemetry data

**Key Functions Demonstrated:**
- `MetricsCollector`, `MetricIncrement`, `MetricGauge`, `MetricHistogram`
- `LogAggregator`, `LogEvent`
- `DistributedTracing`, `TraceSpan`, `SpanEvent`
- `TelemetryExport`

**Run Example:**
```bash
cargo run examples/observability/basic_telemetry.lyra
```

### 2. Service Monitoring (`service_monitoring.lyra`)

Shows comprehensive service health monitoring and alerting:
- Service health checks
- SLO/SLI tracking with error budgets
- Alert management with escalation
- Notification channels (Slack, email, PagerDuty)
- Incident management workflows

**Key Functions Demonstrated:**
- `ServiceHealth`, `HealthCheck`
- `AlertManager`, `AlertRule`, `NotificationChannel`
- `SLOTracker`, `Heartbeat`
- `ServiceDependency`, `IncidentManagement`, `StatusPage`

**Run Example:**
```bash
cargo run examples/observability/service_monitoring.lyra
```

### 3. Performance Profiling (`performance_profiling.lyra`)

Demonstrates performance analysis and optimization:
- CPU and memory profiling
- Latency tracking with percentiles
- Throughput monitoring
- System resource monitoring
- Performance baseline calculation

**Key Functions Demonstrated:**
- `ProfilerStart`, `ProfilerStop`, `CPUProfiler`
- `MemoryProfiler`, `LatencyTracker`
- `ThroughputMonitor`, `ResourceMonitor`
- `PerformanceBaseline`

**Run Example:**
```bash
cargo run examples/observability/performance_profiling.lyra
```

### 4. Debugging Instrumentation (`debugging_instrumentation.lyra`)

Shows advanced debugging and instrumentation:
- Call stack capture and analysis
- Heap dump analysis
- Thread dump with deadlock detection
- Conditional debugging breakpoints

**Key Functions Demonstrated:**
- `CallStack`, `HeapDump`, `ThreadDump`
- `DeadlockDetector`, `DebugBreakpoint`

**Run Example:**
```bash
cargo run examples/observability/debugging_instrumentation.lyra
```

## Production Integration

### Prometheus Integration

```wolfram
(* Export metrics to Prometheus *)
prometheus_exporter = TelemetryExport[
    {"metrics"},
    "prometheus", 
    "http://prometheus:9090/api/v1/write"
]

(* Prometheus-compatible metric names *)
http_requests = MetricsCollector["lyra_http_requests_total", "counter", {
    {"method", "GET"},
    {"endpoint", "/api/evaluate"}
}]
```

### Grafana Dashboards

Set up Grafana dashboards using the exported metrics:
- Request rate and latency
- Error rate and success rate
- Resource utilization (CPU, memory, disk)
- SLO compliance and error budget

### ELK Stack Integration

```wolfram
(* Configure structured logging for Elasticsearch *)
elk_logger = LogAggregator["info", "json", {
    "elasticsearch://elasticsearch:9200/lyra-logs"
}, {
    {"@timestamp", "auto"},
    {"service", "lyra-engine"},
    {"environment", "production"}
}]
```

### Jaeger Tracing

```wolfram
(* Export traces to Jaeger *)
jaeger_exporter = TelemetryExport[
    {"traces"},
    "jaeger",
    "http://jaeger:14268/api/traces"
]
```

## Best Practices

### 1. Monitoring Strategy
- Monitor the "Golden Signals": latency, traffic, errors, saturation
- Define clear SLIs (Service Level Indicators) and SLOs (Service Level Objectives)
- Set up proper alerting thresholds to avoid alert fatigue
- Link alerts to runbooks for efficient incident response

### 2. Performance Optimization
- Use sampling for high-volume metrics to reduce overhead
- Batch telemetry data for efficient network usage
- Enable compression for large data transfers
- Implement local buffering to handle network interruptions

### 3. Security Considerations
- Sanitize sensitive data before logging
- Use encrypted transport for all telemetry data
- Implement proper access controls for monitoring systems
- Ensure compliance with data protection regulations

### 4. Operational Excellence
- Maintain comprehensive documentation
- Train teams on monitoring tools and procedures
- Regularly review and update monitoring configurations
- Integrate monitoring with incident response workflows

## Troubleshooting

### Common Issues

1. **High Memory Usage**: Use `MemoryProfiler` to identify memory leaks
2. **Performance Degradation**: Use `LatencyTracker` and `CPUProfiler` for analysis
3. **Service Unavailability**: Check `ServiceHealth` and `AlertManager` for issues
4. **Deadlocks**: Use `DeadlockDetector` to identify and resolve deadlocks

### Debug Commands

```wolfram
(* Quick health check *)
health = ServiceHealth["my-service", {}, {}]
health.runChecks()

(* Memory analysis *)
profiler = MemoryProfiler[ProcessID[], 1, True]
snapshot = profiler.takeSnapshot()
analysis = profiler.analyze()

(* Performance check *)
tracker = LatencyTracker["operation", {95, 99}, {}]
tracker.record(125.5, True)
summary = tracker.summary()
```

## Advanced Usage

### Custom Metrics

```wolfram
(* Business-specific metrics *)
user_registrations = MetricsCollector["user_registrations_total", "counter", {}]
active_sessions = MetricsCollector["active_sessions", "gauge", {}]
computation_complexity = MetricsCollector["computation_complexity", "histogram", {}]
```

### Multi-Service Tracing

```wolfram
(* Trace across multiple services *)
user_service_tracer = DistributedTracing["user-service", {}]
computation_service_tracer = DistributedTracing["computation-service", {}]
storage_service_tracer = DistributedTracing["storage-service", {}]
```

### Automated Incident Response

```wolfram
(* Automated response to critical alerts *)
critical_alert_rule = AlertRule["service_down", 1.0, 0, "critical"]
auto_restart_channel = NotificationChannel["webhook", {
    {"url", "https://automation.company.com/restart-service"}
}, {}]
```

## Resources

- [Lyra Observability Documentation](../docs/PHASE_16B_OBSERVABILITY_IMPLEMENTATION.md)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Site Reliability Engineering Book](https://sre.google/sre-book/table-of-contents/)

## Support

For questions or issues with the observability system:
1. Check the comprehensive test suite in `tests/observability_integration_tests.rs`
2. Review the implementation documentation
3. Examine the Foreign object implementations in `src/stdlib/observability/`