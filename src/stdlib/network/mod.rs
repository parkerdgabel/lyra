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

// Re-export all public functions and types
pub use core::*;
pub use http::*;
pub use websocket::*;
pub use events::*;
pub use distributed::*;
pub use analysis::*;
pub use cloud::*;
pub use web::*;