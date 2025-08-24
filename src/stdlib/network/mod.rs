//! Network Computing & Distributed Abstractions
//!
//! This module implements comprehensive networking capabilities that integrate seamlessly
//! with Lyra's symbolic computation philosophy. Network operations are treated as symbolic
//! expressions that can be composed, analyzed, and optimized.
//!
//! # Design Philosophy
//!
//! ## Network-Transparent Symbolic Computation
//! - **Network operations as symbolic expressions** that can be composed and analyzed
//! - **Location-transparent computation** where functions run seamlessly across distributed systems
//! - **Event streams as data structures** that can be manipulated symbolically
//! - **Services as first-class objects** that can be discovered, composed, and analyzed
//! - **Network topology as computational data** for mathematical reasoning
//!
//! ## Integration with Lyra's Architecture
//! - **Foreign Object Pattern**: Complex network state lives outside VM
//! - **Async by Default**: All operations integrate with existing ThreadPool/Channel/Future system
//! - **Symbolic Representation**: Network configurations, requests, and responses are symbolic data
//! - **Functional Composition**: Network operations can be chained and transformed functionally
//!
//! # Module Organization
//!
//! ## Phase 12A: Core Network Primitives
//! - **NetworkRequest/NetworkResponse**: Symbolic HTTP operations
//! - **WebSocket**: Real-time bidirectional communication
//! - **NetworkEndpoint**: Service location abstraction
//! - **URLRead/URLWrite**: Basic network I/O with automatic retries
//! - **Network Diagnostics**: Ping, DNS resolution, connectivity testing
//!
//! ## Phase 12B: Event-Driven & Streaming Architecture
//! - **EventStream**: Infinite sequences of network events
//! - **EventSubscribe/EventPublish**: Pattern-based pub/sub messaging
//! - **MessageQueue**: Persistent async messaging integration
//! - **NetworkChannel**: Network-backed channels using existing Channel system
//! - **Event Processing**: Stream aggregation, windowing, filtering
//!
//! ## Phase 12C: Distributed Computing Primitives
//! - **RemoteFunction**: Network-transparent function calls
//! - **DistributedMap/DistributedReduce**: Parallel operations across nodes
//! - **ServiceRegistry**: Automatic service discovery and registration
//! - **LoadBalancer**: Traffic distribution strategies
//! - **ComputeCluster**: Managed distributed compute resources
//!
//! ## Phase 12D: Network Analysis & Topology
//! - **NetworkGraph**: Network topology modeling and analysis
//! - **NetworkFlow**: Flow analysis algorithms and optimization
//! - **NetworkMetrics**: Centrality, clustering, performance analysis
//! - **NetworkMonitor**: Continuous monitoring and anomaly detection
//! - **Performance Analysis**: Latency, throughput, bottleneck identification
//!
//! ## Phase 12E: Cloud & Infrastructure Integration
//! - **CloudFunction**: Serverless execution on major cloud providers
//! - **CloudStorage**: Object storage abstraction (S3, GCS, Azure)
//! - **ContainerRun**: Container execution and orchestration
//! - **KubernetesService**: K8s deployment and management
//! - **Cloud APIs**: Provider-agnostic cloud service integration
//!
//! # Usage Examples
//!
//! ## Basic HTTP Operations
//! ```wolfram
//! (* Symbolic HTTP requests *)
//! request = NetworkRequest["https://api.example.com/data", "GET", {"Auth" -> token}]
//! response = URLRead[request]
//! data = response["Body"] // JSON["Property"]
//!
//! (* Parallel requests *)
//! results = URLRead[{request1, request2, request3}]  (* Automatic parallelization *)
//! ```
//!
//! ## Event Streaming
//! ```wolfram
//! (* Real-time event processing *)
//! stream = EventStream["ws://events.example.com", {"type" -> "update"}]
//! filtered = EventSubscribe[stream, {"severity" -> "critical"}]
//! aggregated = EventReduce[filtered, Add, 0]
//! ```
//!
//! ## Distributed Computation
//! ```wolfram
//! (* Location-transparent function calls *)
//! result1 = ComputeFunction[data]                          (* local *)
//! result2 = RemoteFunction[node, ComputeFunction][data]    (* remote *)
//! result3 = DistributedMap[ComputeFunction, bigData, cluster]  (* parallel *)
//! ```
//!
//! ## Service Discovery & Load Balancing
//! ```wolfram
//! (* Automatic service discovery *)
//! services = ServiceDiscover["compute-service", {"version" -> "2.0"}]
//! balancer = LoadBalancer[services, "RoundRobin"]
//! result = balancer.call("processData", data)
//! ```
//!
//! ## Network Analysis
//! ```wolfram
//! (* Network topology as symbolic data *)
//! graph = NetworkGraph[ServiceConnections[cluster]]
//! metrics = NetworkMetrics[graph, {"centrality", "clustering", "paths"}]
//! bottlenecks = NetworkBottlenecks[graph, trafficModel]
//! optimized = OptimizeTopology[graph, constraints]
//! ```

pub mod core;
pub mod http;
pub mod websocket;
pub mod events;
pub mod distributed;
pub mod analysis;
pub mod cloud;
pub mod web;
pub mod graphql;

// Re-export all public functions and types
pub use core::*;
pub use http::*;
pub use websocket::*;
pub use events::*;
pub use distributed::*;
pub use analysis::*;
pub use cloud::*;
pub use web::*;
pub use graphql::*;

use crate::vm::{Value, VmResult};
use std::collections::HashMap;

/// Registration helper to consolidate network-related stdlib functions
pub fn register_network_functions() -> HashMap<String, fn(&[Value]) -> VmResult<Value>> {
    let mut f = HashMap::new();

    // Phase 12A: Core Network Primitives
    f.insert("NetworkEndpoint".to_string(), network_endpoint as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetworkRequest".to_string(), network_request as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetworkAuth".to_string(), network_auth as fn(&[Value]) -> VmResult<Value>);
    f.insert("URLRead".to_string(), url_read as fn(&[Value]) -> VmResult<Value>);
    f.insert("URLWrite".to_string(), url_write as fn(&[Value]) -> VmResult<Value>);
    f.insert("URLStream".to_string(), url_stream as fn(&[Value]) -> VmResult<Value>);
    f.insert("HTTPRetry".to_string(), http_retry as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetworkPing".to_string(), network_ping as fn(&[Value]) -> VmResult<Value>);
    f.insert("DNSResolve".to_string(), dns_resolve as fn(&[Value]) -> VmResult<Value>);
    f.insert("HttpClient".to_string(), http_client as fn(&[Value]) -> VmResult<Value>);

    // WebSocket operations
    f.insert("WebSocket".to_string(), websocket as fn(&[Value]) -> VmResult<Value>);
    f.insert("WebSocketConnect".to_string(), websocket_connect as fn(&[Value]) -> VmResult<Value>);
    f.insert("WebSocketSend".to_string(), websocket_send as fn(&[Value]) -> VmResult<Value>);
    f.insert("WebSocketReceive".to_string(), websocket_receive as fn(&[Value]) -> VmResult<Value>);
    f.insert("WebSocketClose".to_string(), websocket_close as fn(&[Value]) -> VmResult<Value>);
    f.insert("WebSocketPing".to_string(), websocket_ping as fn(&[Value]) -> VmResult<Value>);

    // Phase 12B: Event-Driven Architecture
    f.insert("EventStream".to_string(), event_stream as fn(&[Value]) -> VmResult<Value>);
    f.insert("EventSubscribe".to_string(), event_subscribe as fn(&[Value]) -> VmResult<Value>);
    f.insert("EventPublish".to_string(), event_publish as fn(&[Value]) -> VmResult<Value>);
    f.insert("MessageQueue".to_string(), message_queue as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetworkChannel".to_string(), network_channel as fn(&[Value]) -> VmResult<Value>);

    // Phase 12C: Distributed Computing
    f.insert("RemoteFunction".to_string(), remote_function as fn(&[Value]) -> VmResult<Value>);
    f.insert("RemoteFunctionCall".to_string(), remote_function_call as fn(&[Value]) -> VmResult<Value>);
    f.insert("DistributedMap".to_string(), distributed_map as fn(&[Value]) -> VmResult<Value>);
    f.insert("DistributedMapExecute".to_string(), distributed_map_execute as fn(&[Value]) -> VmResult<Value>);
    f.insert("DistributedReduce".to_string(), distributed_reduce as fn(&[Value]) -> VmResult<Value>);
    f.insert("DistributedReduceExecute".to_string(), distributed_reduce_execute as fn(&[Value]) -> VmResult<Value>);
    f.insert("ServiceRegistry".to_string(), service_registry as fn(&[Value]) -> VmResult<Value>);
    f.insert("ServiceDiscover".to_string(), service_discover as fn(&[Value]) -> VmResult<Value>);
    f.insert("ServiceHealthCheck".to_string(), service_health_check as fn(&[Value]) -> VmResult<Value>);
    f.insert("LoadBalancer".to_string(), load_balancer as fn(&[Value]) -> VmResult<Value>);
    f.insert("LoadBalancerRequest".to_string(), load_balancer_request as fn(&[Value]) -> VmResult<Value>);
    f.insert("ComputeCluster".to_string(), compute_cluster as fn(&[Value]) -> VmResult<Value>);
    f.insert("ClusterAddNode".to_string(), cluster_add_node as fn(&[Value]) -> VmResult<Value>);
    f.insert("ClusterSubmitTask".to_string(), cluster_submit_task as fn(&[Value]) -> VmResult<Value>);
    f.insert("ClusterGetStats".to_string(), cluster_get_stats as fn(&[Value]) -> VmResult<Value>);

    // Phase 12D: Network Analysis
    f.insert("NetworkGraph".to_string(), network_graph as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphAddNode".to_string(), graph_add_node as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphAddEdge".to_string(), graph_add_edge as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphShortestPath".to_string(), graph_shortest_path as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphMST".to_string(), graph_mst as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphComponents".to_string(), graph_components as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphMetrics".to_string(), graph_metrics as fn(&[Value]) -> VmResult<Value>);

    // Centrality and community analysis
    f.insert("NetworkCentrality".to_string(), network_centrality as fn(&[Value]) -> VmResult<Value>);
    f.insert("CommunityDetection".to_string(), community_detection as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphDiameter".to_string(), graph_diameter as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphDensity".to_string(), graph_density as fn(&[Value]) -> VmResult<Value>);
    f.insert("ClusteringCoefficient".to_string(), clustering_coefficient as fn(&[Value]) -> VmResult<Value>);

    // Network flow algorithms
    f.insert("NetworkFlow".to_string(), network_flow as fn(&[Value]) -> VmResult<Value>);
    f.insert("MinimumCut".to_string(), minimum_cut as fn(&[Value]) -> VmResult<Value>);
    f.insert("FlowDecomposition".to_string(), flow_decomposition as fn(&[Value]) -> VmResult<Value>);
    f.insert("FlowBottlenecks".to_string(), flow_bottlenecks as fn(&[Value]) -> VmResult<Value>);
    f.insert("MaxFlowValue".to_string(), max_flow_value as fn(&[Value]) -> VmResult<Value>);

    // Network monitoring and diagnostics
    f.insert("NetworkMonitor".to_string(), network_monitor as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonitorStart".to_string(), monitor_start as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonitorStop".to_string(), monitor_stop as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonitorGetMetrics".to_string(), monitor_get_metrics as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonitorSetAlerts".to_string(), monitor_set_alerts as fn(&[Value]) -> VmResult<Value>);
    f.insert("MonitorPing".to_string(), monitor_ping as fn(&[Value]) -> VmResult<Value>);
    f.insert("NetworkBottlenecks".to_string(), network_bottlenecks as fn(&[Value]) -> VmResult<Value>);
    f.insert("OptimizeTopology".to_string(), optimize_topology as fn(&[Value]) -> VmResult<Value>);

    // Phase 12E: Cloud Integration
    f.insert("CloudFunction".to_string(), cloud_function as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionDeploy".to_string(), cloud_function_deploy as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionInvoke".to_string(), cloud_function_invoke as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionUpdate".to_string(), cloud_function_update as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionLogs".to_string(), cloud_function_logs as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionMetrics".to_string(), cloud_function_metrics as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudFunctionDelete".to_string(), cloud_function_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudStorage".to_string(), cloud_storage as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerRun".to_string(), container_run as fn(&[Value]) -> VmResult<Value>);
    f.insert("KubernetesService".to_string(), kubernetes_service as fn(&[Value]) -> VmResult<Value>);
    f.insert("KubernetesDeploy".to_string(), kubernetes_deploy as fn(&[Value]) -> VmResult<Value>);
    f.insert("DeploymentScale".to_string(), deployment_scale as fn(&[Value]) -> VmResult<Value>);
    f.insert("RollingUpdate".to_string(), rolling_update as fn(&[Value]) -> VmResult<Value>);
    f.insert("ConfigMapCreate".to_string(), configmap_create as fn(&[Value]) -> VmResult<Value>);
    f.insert("ServiceExpose".to_string(), service_expose as fn(&[Value]) -> VmResult<Value>);
    f.insert("PodLogs".to_string(), pod_logs as fn(&[Value]) -> VmResult<Value>);
    f.insert("ResourceGet".to_string(), resource_get as fn(&[Value]) -> VmResult<Value>);
    f.insert("ResourceDelete".to_string(), resource_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudDeploy".to_string(), cloud_deploy as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudMonitor".to_string(), cloud_monitor as fn(&[Value]) -> VmResult<Value>);

    // Cloud Storage API Functions
    f.insert("CloudUpload".to_string(), cloud_upload as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudDownload".to_string(), cloud_download as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudList".to_string(), cloud_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudDelete".to_string(), cloud_delete as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudMetadata".to_string(), cloud_metadata as fn(&[Value]) -> VmResult<Value>);
    f.insert("CloudPresignedURL".to_string(), cloud_presigned_url as fn(&[Value]) -> VmResult<Value>);

    // Docker Container API Functions
    f.insert("ContainerStop".to_string(), container_stop as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerLogs".to_string(), container_logs as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerInspect".to_string(), container_inspect as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerExec".to_string(), container_exec as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerList".to_string(), container_list as fn(&[Value]) -> VmResult<Value>);
    f.insert("ContainerPull".to_string(), container_pull as fn(&[Value]) -> VmResult<Value>);

    // GraphQL client
    f.insert("GraphQLClient".to_string(), graphql_client as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphQLQuery".to_string(), graphql_query as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphQLIntrospect".to_string(), graphql_introspect as fn(&[Value]) -> VmResult<Value>);
    f.insert("GraphQLQueryResponse".to_string(), graphql_query_response as fn(&[Value]) -> VmResult<Value>);

    f
}
