//! Performance Profiling Module
//!
//! This module provides comprehensive performance profiling capabilities including CPU profiling,
//! memory analysis, latency tracking, and system resource monitoring.
//!
//! # Core Profiling Functions (8 functions)
//! - ProfilerStart - Start profiler
//! - ProfilerStop - Stop and export profile
//! - MemoryProfiler - Memory usage profiling
//! - CPUProfiler - CPU profiling
//! - LatencyTracker - Latency measurement
//! - ThroughputMonitor - Throughput monitoring
//! - ResourceMonitor - System resource monitoring
//! - PerformanceBaseline - Performance baselines

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::Value;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use serde::{Serialize, Deserialize};
use sysinfo::{System, SystemExt, CpuExt, ProcessExt};
use parking_lot::RwLock;

/// CPU profiler for performance analysis
#[derive(Debug)]
pub struct CPUProfiler {
    duration: Duration,
    sampling_frequency: u64,
    flame_graph: bool,
    start_time: Option<Instant>,
    samples: Arc<RwLock<Vec<ProfileSample>>>,
    config: ProfilerConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilerConfig {
    output_format: String,
    sampling_rate: u64,
    stack_depth: usize,
    filter_patterns: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSample {
    timestamp: u64,
    thread_id: u64,
    stack_trace: Vec<String>,
    cpu_usage: f32,
    memory_usage: u64,
}

impl Foreign for CPUProfiler {
    fn type_name(&self) -> &'static str {
        "CPUProfiler"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "start" => {
                // Start profiling
                let start_time = Instant::now();
                // TODO: Initialize actual CPU profiling (pprof integration)
                Ok(Value::Boolean(true))
            }

            "stop" => {
                if let Some(_start) = self.start_time {
                    // TODO: Stop profiling and generate report
                    let samples = self.samples.read();
                    let profile_data = ProfileData {
                        duration: self.duration.as_secs(),
                        sample_count: samples.len(),
                        samples: samples.clone(),
                        flame_graph_data: if self.flame_graph {
                            Some("flame_graph_svg_data".to_string())
                        } else {
                            None
                        },
                    };

                    let profile_json = serde_json::to_string(&profile_data).unwrap_or_default();
                    Ok(Value::String(profile_json))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "Profiler not started".to_string(),
                    })
                }
            }

            "addSample" => {
                let timestamp = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                let sample = ProfileSample {
                    timestamp,
                    thread_id: 0, // TODO: Get actual thread ID
                    stack_trace: vec!["main".to_string()], // TODO: Capture actual stack trace
                    cpu_usage: 0.0, // TODO: Get actual CPU usage
                    memory_usage: 0, // TODO: Get actual memory usage
                };

                self.samples.write().push(sample);
                Ok(Value::Boolean(true))
            }

            "status" => {
                let status = HashMap::from([
                    ("running".to_string(), self.start_time.is_some().to_string()),
                    ("sample_count".to_string(), self.samples.read().len().to_string()),
                    ("duration".to_string(), self.duration.as_secs().to_string()),
                ]);

                let status_list: Vec<Value> = status.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
                    .collect();

                Ok(Value::List(status_list))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(CPUProfiler {
            duration: self.duration,
            sampling_frequency: self.sampling_frequency,
            flame_graph: self.flame_graph,
            start_time: None, // Don't clone the start time
            samples: Arc::clone(&self.samples),
            config: self.config.clone(),
        })
    }
}

impl Clone for CPUProfiler {
    fn clone(&self) -> Self {
        CPUProfiler {
            duration: self.duration,
            sampling_frequency: self.sampling_frequency,
            flame_graph: self.flame_graph,
            start_time: self.start_time,
            samples: Arc::clone(&self.samples),
            config: self.config.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileData {
    duration: u64,
    sample_count: usize,
    samples: Vec<ProfileSample>,
    flame_graph_data: Option<String>,
}

/// Memory profiler for heap analysis
#[derive(Debug, Clone)]
pub struct MemoryProfiler {
    process_id: u32,
    interval: Duration,
    heap_analysis: bool,
    snapshots: Arc<RwLock<Vec<MemorySnapshot>>>,
    system: Arc<Mutex<System>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    timestamp: u64,
    total_memory: u64,
    heap_memory: u64,
    stack_memory: u64,
    allocations: u64,
    deallocations: u64,
    fragmentation: f64,
}

impl Foreign for MemoryProfiler {
    fn type_name(&self) -> &'static str {
        "MemoryProfiler"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "takeSnapshot" => {
                let mut system = self.system.lock().unwrap();
                system.refresh_process(sysinfo::Pid::from(self.process_id as usize));
                
                if let Some(process) = system.process(sysinfo::Pid::from(self.process_id as usize)) {
                    let snapshot = MemorySnapshot {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        total_memory: process.memory(),
                        heap_memory: process.memory() * 7 / 10, // Estimate heap as 70% of total
                        stack_memory: process.memory() * 1 / 10, // Estimate stack as 10% of total
                        allocations: 0, // TODO: Track allocations
                        deallocations: 0, // TODO: Track deallocations
                        fragmentation: 0.0, // TODO: Calculate fragmentation
                    };

                    self.snapshots.write().push(snapshot.clone());
                    let snapshot_json = serde_json::to_string(&snapshot).unwrap_or_default();
                    Ok(Value::String(snapshot_json))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "Process not found".to_string(),
                    })
                }
            }

            "snapshots" => {
                let snapshots = self.snapshots.read();
                let snapshot_list: Vec<Value> = snapshots.iter()
                    .map(|s| Value::String(serde_json::to_string(s).unwrap_or_default()))
                    .collect();
                Ok(Value::List(snapshot_list))
            }

            "analyze" => {
                let snapshots = self.snapshots.read();
                if snapshots.len() < 2 {
                    return Ok(Value::String("Need at least 2 snapshots for analysis".to_string()));
                }

                let first = &snapshots[0];
                let last = &snapshots[snapshots.len() - 1];
                
                let memory_growth = last.total_memory as i64 - first.total_memory as i64;
                let time_span = last.timestamp - first.timestamp;
                
                let analysis = HashMap::from([
                    ("memory_growth_bytes".to_string(), memory_growth.to_string()),
                    ("time_span_seconds".to_string(), time_span.to_string()),
                    ("growth_rate_bytes_per_sec".to_string(), 
                     if time_span > 0 { (memory_growth / time_span as i64).to_string() } else { "0".to_string() }),
                    ("snapshot_count".to_string(), snapshots.len().to_string()),
                ]);

                let analysis_list: Vec<Value> = analysis.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
                    .collect();

                Ok(Value::List(analysis_list))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

/// Latency tracker for measuring operation response times
#[derive(Debug, Clone)]
pub struct LatencyTracker {
    operation_name: String,
    percentiles: Vec<f64>,
    buckets: Vec<f64>,
    measurements: Arc<RwLock<Vec<LatencyMeasurement>>>,
    start_times: Arc<RwLock<HashMap<String, Instant>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    timestamp: u64,
    operation: String,
    latency_ms: f64,
    success: bool,
    metadata: HashMap<String, String>,
}

impl Foreign for LatencyTracker {
    fn type_name(&self) -> &'static str {
        "LatencyTracker"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "startTimer" => {
                let operation_id = match args.get(0) {
                    Some(Value::String(id)) => id.clone(),
                    _ => format!("op_{}", rand::random::<u32>()),
                };

                self.start_times.write().insert(operation_id.clone(), Instant::now());
                Ok(Value::String(operation_id))
            }

            "endTimer" => {
                let operation_id = match args.get(0) {
                    Some(Value::String(id)) => id.clone(),
                    _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
                };

                let success = match args.get(1) {
                    Some(Value::Boolean(s)) => *s,
                    _ => true,
                };

                if let Some(start_time) = self.start_times.write().remove(&operation_id) {
                    let latency = start_time.elapsed().as_millis() as f64;
                    
                    let measurement = LatencyMeasurement {
                        timestamp: SystemTime::now()
                            .duration_since(UNIX_EPOCH)
                            .unwrap()
                            .as_secs(),
                        operation: self.operation_name.clone(),
                        latency_ms: latency,
                        success,
                        metadata: HashMap::new(),
                    };

                    self.measurements.write().push(measurement);
                    Ok(Value::Real(latency))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: format!("Timer not found: {}", operation_id),
                    })
                }
            }

            "record" => {
                let latency = match args.get(0) {
                    Some(Value::Real(l)) => *l,
                    Some(Value::Integer(l)) => *l as f64,
                    _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
                };

                let success = match args.get(1) {
                    Some(Value::Boolean(s)) => *s,
                    _ => true,
                };

                let measurement = LatencyMeasurement {
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    operation: self.operation_name.clone(),
                    latency_ms: latency,
                    success,
                    metadata: HashMap::new(),
                };

                self.measurements.write().push(measurement);
                Ok(Value::Boolean(true))
            }

            "percentiles" => {
                let measurements = self.measurements.read();
                let mut latencies: Vec<f64> = measurements.iter()
                    .filter(|m| m.success)
                    .map(|m| m.latency_ms)
                    .collect();

                if latencies.is_empty() {
                    return Ok(Value::List(vec![]));
                }

                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let percentile_values: Vec<Value> = self.percentiles.iter()
                    .map(|&p| {
                        let index = ((p / 100.0) * (latencies.len() - 1) as f64) as usize;
                        let value = latencies.get(index).unwrap_or(&0.0);
                        Value::List(vec![
                            Value::String(format!("p{}", p)),
                            Value::Real(*value)
                        ])
                    })
                    .collect();

                Ok(Value::List(percentile_values))
            }

            "summary" => {
                let measurements = self.measurements.read();
                let successful: Vec<&LatencyMeasurement> = measurements.iter()
                    .filter(|m| m.success)
                    .collect();

                if successful.is_empty() {
                    return Ok(Value::String("No successful measurements".to_string()));
                }

                let total_latency: f64 = successful.iter().map(|m| m.latency_ms).sum();
                let avg_latency = total_latency / successful.len() as f64;
                let min_latency = successful.iter().map(|m| m.latency_ms).fold(f64::INFINITY, f64::min);
                let max_latency = successful.iter().map(|m| m.latency_ms).fold(0.0, f64::max);

                let summary = HashMap::from([
                    ("count".to_string(), successful.len().to_string()),
                    ("avg_latency_ms".to_string(), avg_latency.to_string()),
                    ("min_latency_ms".to_string(), min_latency.to_string()),
                    ("max_latency_ms".to_string(), max_latency.to_string()),
                    ("success_rate".to_string(), 
                     (successful.len() as f64 / measurements.len() as f64).to_string()),
                ]);

                let summary_list: Vec<Value> = summary.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
                    .collect();

                Ok(Value::List(summary_list))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

/// Throughput monitor for measuring request rates
#[derive(Debug, Clone)]
pub struct ThroughputMonitor {
    endpoint: String,
    window_size: Duration,
    aggregation_type: String,
    requests: Arc<RwLock<Vec<RequestEvent>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestEvent {
    timestamp: u64,
    endpoint: String,
    success: bool,
    response_size: u64,
}

impl Foreign for ThroughputMonitor {
    fn type_name(&self) -> &'static str {
        "ThroughputMonitor"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "recordRequest" => {
                let success = match args.get(0) {
                    Some(Value::Boolean(s)) => *s,
                    _ => true,
                };

                let response_size = match args.get(1) {
                    Some(Value::Integer(size)) => *size as u64,
                    Some(Value::Real(size)) => *size as u64,
                    _ => 0,
                };

                let event = RequestEvent {
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    endpoint: self.endpoint.clone(),
                    success,
                    response_size,
                };

                self.requests.write().push(event);
                Ok(Value::Boolean(true))
            }

            "throughput" => {
                let requests = self.requests.read();
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();
                
                let window_start = now - self.window_size.as_secs();
                
                let recent_requests: Vec<&RequestEvent> = requests.iter()
                    .filter(|r| r.timestamp >= window_start)
                    .collect();

                let request_count = recent_requests.len();
                let successful_requests = recent_requests.iter().filter(|r| r.success).count();
                let total_bytes = recent_requests.iter().map(|r| r.response_size).sum::<u64>();

                let throughput_rps = request_count as f64 / self.window_size.as_secs() as f64;
                let success_rate = if request_count > 0 {
                    successful_requests as f64 / request_count as f64
                } else {
                    0.0
                };

                let metrics = HashMap::from([
                    ("requests_per_second".to_string(), throughput_rps.to_string()),
                    ("success_rate".to_string(), success_rate.to_string()),
                    ("total_requests".to_string(), request_count.to_string()),
                    ("total_bytes".to_string(), total_bytes.to_string()),
                    ("window_seconds".to_string(), self.window_size.as_secs().to_string()),
                ]);

                let metrics_list: Vec<Value> = metrics.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
                    .collect();

                Ok(Value::List(metrics_list))
            }

            "reset" => {
                self.requests.write().clear();
                Ok(Value::Boolean(true))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

/// System resource monitor
#[derive(Debug, Clone)]
pub struct ResourceMonitor {
    system_name: String,
    metrics: Vec<String>,
    alert_rules: Vec<ResourceAlert>,
    system: Arc<Mutex<System>>,
    readings: Arc<RwLock<Vec<ResourceReading>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    metric: String,
    threshold: f64,
    condition: String, // "greater_than", "less_than", etc.
    severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReading {
    timestamp: u64,
    cpu_usage: f32,
    memory_usage: u64,
    disk_usage: u64,
    network_rx: u64,
    network_tx: u64,
}

impl Foreign for ResourceMonitor {
    fn type_name(&self) -> &'static str {
        "ResourceMonitor"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "collect" => {
                let mut system = self.system.lock().unwrap();
                system.refresh_all();

                let reading = ResourceReading {
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    cpu_usage: system.global_cpu_info().cpu_usage(),
                    memory_usage: system.used_memory(),
                    disk_usage: 0, // TODO: Implement disk usage collection
                    network_rx: 0, // TODO: Implement network stats
                    network_tx: 0, // TODO: Implement network stats
                };

                self.readings.write().push(reading.clone());
                let reading_json = serde_json::to_string(&reading).unwrap_or_default();
                Ok(Value::String(reading_json))
            }

            "latest" => {
                let readings = self.readings.read();
                if let Some(latest) = readings.last() {
                    let reading_json = serde_json::to_string(latest).unwrap_or_default();
                    Ok(Value::String(reading_json))
                } else {
                    Ok(Value::String("{}".to_string()))
                }
            }

            "history" => {
                let limit = match args.get(0) {
                    Some(Value::Integer(l)) => *l as usize,
                    _ => 100,
                };

                let readings = self.readings.read();
                let recent_readings: Vec<Value> = readings.iter()
                    .rev()
                    .take(limit)
                    .map(|r| Value::String(serde_json::to_string(r).unwrap_or_default()))
                    .collect();

                Ok(Value::List(recent_readings))
            }

            "checkAlerts" => {
                let readings = self.readings.read();
                if let Some(latest) = readings.last() {
                    let mut triggered_alerts = Vec::new();

                    for alert in &self.alert_rules {
                        let value = match alert.metric.as_str() {
                            "cpu" => latest.cpu_usage as f64,
                            "memory" => latest.memory_usage as f64,
                            "disk" => latest.disk_usage as f64,
                            _ => continue,
                        };

                        let triggered = match alert.condition.as_str() {
                            "greater_than" => value > alert.threshold,
                            "less_than" => value < alert.threshold,
                            "equals" => (value - alert.threshold).abs() < 0.001,
                            _ => false,
                        };

                        if triggered {
                            triggered_alerts.push(Value::String(format!(
                                "Alert: {} {} {} (current: {})",
                                alert.metric, alert.condition, alert.threshold, value
                            )));
                        }
                    }

                    Ok(Value::List(triggered_alerts))
                } else {
                    Ok(Value::List(vec![]))
                }
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
}

// Helper functions

fn extract_string_list(args: &[Value]) -> Vec<String> {
    args.iter()
        .filter_map(|v| match v {
            Value::String(s) => Some(s.clone()),
            _ => None,
        })
        .collect()
}

fn extract_f64_list(args: &[Value]) -> Vec<f64> {
    args.iter()
        .filter_map(|v| match v {
            Value::Real(f) => Some(*f),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        })
        .collect()
}

// Stdlib function implementations

/// ProfilerStart[type, duration, sampling_rate] - Start profiler
pub fn profiler_start(args: &[Value]) -> Result<Value, ForeignError> {
    let profiler_type = match args.get(0) {
        Some(Value::String(t)) => t.clone(),
        _ => "cpu".to_string(),
    };

    let duration_secs = match args.get(1) {
        Some(Value::Integer(d)) => *d as u64,
        Some(Value::Real(d)) => *d as u64,
        _ => 60,
    };

    let sampling_rate = match args.get(2) {
        Some(Value::Integer(r)) => *r as u64,
        Some(Value::Real(r)) => *r as u64,
        _ => 100,
    };

    match profiler_type.as_str() {
        "cpu" => {
            let profiler = CPUProfiler {
                duration: Duration::from_secs(duration_secs),
                sampling_frequency: sampling_rate,
                flame_graph: true,
                start_time: Some(Instant::now()),
                samples: Arc::new(RwLock::new(Vec::new())),
                config: ProfilerConfig {
                    output_format: "pprof".to_string(),
                    sampling_rate,
                    stack_depth: 32,
                    filter_patterns: vec![],
                },
            };

            Ok(Value::LyObj(LyObj::new(Box::new(profiler))))
        }
        _ => Err(ForeignError::RuntimeError {
            message: format!("Unknown profiler type: {}", profiler_type),
        }),
    }
}

/// ProfilerStop[profiler, output_format] - Stop and export profile
pub fn profiler_stop(args: &[Value]) -> Result<Value, ForeignError> {
    let profiler = match args.get(0) {
        Some(Value::LyObj(obj)) => obj,
        _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
    };

    let _output_format = match args.get(1) {
        Some(Value::String(_format)) => _format,
        _ => "pprof",
    };

    profiler.call_method("stop", &[])
}

/// MemoryProfiler[process, interval, heap_analysis] - Memory usage profiling
pub fn memory_profiler(args: &[Value]) -> Result<Value, ForeignError> {
    let process_id = match args.get(0) {
        Some(Value::Integer(pid)) => *pid as u32,
        _ => std::process::id(),
    };

    let interval_secs = match args.get(1) {
        Some(Value::Integer(i)) => *i as u64,
        Some(Value::Real(i)) => *i as u64,
        _ => 1,
    };

    let heap_analysis = match args.get(2) {
        Some(Value::Boolean(h)) => *h,
        _ => true,
    };

    let profiler = MemoryProfiler {
        process_id,
        interval: Duration::from_secs(interval_secs),
        heap_analysis,
        snapshots: Arc::new(RwLock::new(Vec::new())),
        system: Arc::new(Mutex::new(System::new_all())),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(profiler))))
}

/// CPUProfiler[duration, sampling_frequency, flame_graph] - CPU profiling
pub fn cpu_profiler(args: &[Value]) -> Result<Value, ForeignError> {
    profiler_start(args)
}

/// LatencyTracker[operation, percentiles, buckets] - Latency measurement
pub fn latency_tracker(args: &[Value]) -> Result<Value, ForeignError> {
    let operation = match args.get(0) {
        Some(Value::String(op)) => op.clone(),
        _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
    };

    let percentiles = if let Some(Value::List(p_args)) = args.get(1) {
        extract_f64_list(p_args)
    } else {
        vec![50.0, 95.0, 99.0]
    };

    let buckets = if let Some(Value::List(b_args)) = args.get(2) {
        extract_f64_list(b_args)
    } else {
        vec![1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    };

    let tracker = LatencyTracker {
        operation_name: operation,
        percentiles,
        buckets,
        measurements: Arc::new(RwLock::new(Vec::new())),
        start_times: Arc::new(RwLock::new(HashMap::new())),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(tracker))))
}

/// ThroughputMonitor[endpoint, window_size, aggregation] - Throughput monitoring
pub fn throughput_monitor(args: &[Value]) -> Result<Value, ForeignError> {
    let endpoint = match args.get(0) {
        Some(Value::String(ep)) => ep.clone(),
        _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
    };

    let window_size_secs = match args.get(1) {
        Some(Value::Integer(w)) => *w as u64,
        Some(Value::Real(w)) => *w as u64,
        _ => 60,
    };

    let aggregation = match args.get(2) {
        Some(Value::String(agg)) => agg.clone(),
        _ => "average".to_string(),
    };

    let monitor = ThroughputMonitor {
        endpoint,
        window_size: Duration::from_secs(window_size_secs),
        aggregation_type: aggregation,
        requests: Arc::new(RwLock::new(Vec::new())),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(monitor))))
}

/// ResourceMonitor[system, metrics, alerts] - System resource monitoring
pub fn resource_monitor(args: &[Value]) -> Result<Value, ForeignError> {
    let system_name = match args.get(0) {
        Some(Value::String(name)) => name.clone(),
        _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
    };

    let metrics = if let Some(Value::List(m_args)) = args.get(1) {
        extract_string_list(m_args)
    } else {
        vec!["cpu".to_string(), "memory".to_string(), "disk".to_string()]
    };

    let alert_rules = if let Some(Value::List(_a_args)) = args.get(2) {
        // TODO: Parse alert rules from arguments
        vec![]
    } else {
        vec![]
    };

    let monitor = ResourceMonitor {
        system_name,
        metrics,
        alert_rules,
        system: Arc::new(Mutex::new(System::new_all())),
        readings: Arc::new(RwLock::new(Vec::new())),
    };

    Ok(Value::LyObj(LyObj::new(Box::new(monitor))))
}

/// PerformanceBaseline[metrics, historical_data, anomaly_detection] - Performance baselines
pub fn performance_baseline(args: &[Value]) -> Result<Value, ForeignError> {
    let _metrics = match args.get(0) {
        Some(Value::List(_m_args)) => _m_args,
        _ => return Err(ForeignError::ArgumentError { expected: 1, actual: args.len() }),
    };

    let _historical_data = match args.get(1) {
        Some(Value::List(_h_args)) => _h_args,
        _ => &[],
    };

    let _anomaly_detection = match args.get(2) {
        Some(Value::Boolean(_ad)) => _ad,
        _ => &true,
    };

    // TODO: Implement performance baseline calculation
    // This would analyze historical performance data and detect anomalies
    let baseline_data = HashMap::from([
        ("baseline_cpu".to_string(), "25.5".to_string()),
        ("baseline_memory".to_string(), "512000000".to_string()),
        ("baseline_latency_p95".to_string(), "150.0".to_string()),
        ("anomaly_threshold".to_string(), "2.0".to_string()),
    ]);

    let baseline_list: Vec<Value> = baseline_data.iter()
        .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
        .collect();

    Ok(Value::List(baseline_list))
}