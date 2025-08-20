//! Cloud & Infrastructure Integration

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;
use std::process::{Command, Stdio};
use async_trait::async_trait;
use bytes::Bytes;
use std::time::SystemTime;

/// Cloud storage provider abstraction
#[async_trait]
pub trait CloudProvider: Send + Sync + std::fmt::Debug {
    async fn upload(&self, path: &str, data: Bytes) -> Result<CloudObject, CloudError>;
    async fn download(&self, path: &str) -> Result<Bytes, CloudError>;
    async fn list(&self, prefix: Option<&str>) -> Result<Vec<CloudObject>, CloudError>;
    async fn delete(&self, path: &str) -> Result<(), CloudError>;
    async fn metadata(&self, path: &str) -> Result<CloudObject, CloudError>;
    async fn presigned_url(&self, path: &str, expiry_seconds: u64) -> Result<String, CloudError>;
    fn provider_name(&self) -> &'static str;
}

/// Cloud storage error types
#[derive(Debug, Clone)]
pub enum CloudError {
    NotFound(String),
    AccessDenied(String),
    NetworkError(String),
    InvalidConfiguration(String),
    ProviderError(String),
}

/// Cloud object metadata
#[derive(Debug, Clone)]
pub struct CloudObject {
    pub path: String,
    pub size: u64,
    pub content_type: String,
    pub last_modified: SystemTime,
    pub etag: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Unified cloud storage interface
#[derive(Debug)]
pub struct CloudStorage {
    pub bucket_name: String,
    pub provider: Arc<dyn CloudProvider>,
    pub config: CloudStorageConfig,
}

/// Cloud storage configuration
#[derive(Debug, Clone)]
pub struct CloudStorageConfig {
    pub region: String,
    pub retry_count: u32,
    pub timeout_seconds: u64,
    pub encryption_enabled: bool,
    pub versioning_enabled: bool,
}

impl Foreign for CloudStorage {
    fn type_name(&self) -> &'static str {
        "CloudStorage"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "BucketName" => Ok(Value::String(self.bucket_name.clone())),
            "Provider" => Ok(Value::String(self.provider.provider_name().to_string())),
            "Region" => Ok(Value::String(self.config.region.clone())),
            "RetryCount" => Ok(Value::Integer(self.config.retry_count as i64)),
            "TimeoutSeconds" => Ok(Value::Integer(self.config.timeout_seconds as i64)),
            "EncryptionEnabled" => Ok(Value::Boolean(self.config.encryption_enabled)),
            "VersioningEnabled" => Ok(Value::Boolean(self.config.versioning_enabled)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(CloudStorage {
            bucket_name: self.bucket_name.clone(),
            provider: Arc::clone(&self.provider),
            config: self.config.clone(),
        })
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Mock provider for testing/placeholder purposes
#[derive(Debug)]
pub struct MockProvider {
    name: String,
    bucket: String,
}

#[async_trait]
impl CloudProvider for MockProvider {
    async fn upload(&self, path: &str, data: Bytes) -> Result<CloudObject, CloudError> {
        Ok(CloudObject {
            path: path.to_string(),
            size: data.len() as u64,
            content_type: "application/octet-stream".to_string(),
            last_modified: SystemTime::now(),
            etag: Some("mock-etag".to_string()),
            metadata: HashMap::new(),
        })
    }
    
    async fn download(&self, _path: &str) -> Result<Bytes, CloudError> {
        Ok(Bytes::from("mock data"))
    }
    
    async fn list(&self, _prefix: Option<&str>) -> Result<Vec<CloudObject>, CloudError> {
        Ok(vec![
            CloudObject {
                path: "example/file.txt".to_string(),
                size: 1024,
                content_type: "text/plain".to_string(),
                last_modified: SystemTime::now(),
                etag: Some("mock-etag-1".to_string()),
                metadata: HashMap::new(),
            }
        ])
    }
    
    async fn delete(&self, _path: &str) -> Result<(), CloudError> {
        Ok(())
    }
    
    async fn metadata(&self, path: &str) -> Result<CloudObject, CloudError> {
        Ok(CloudObject {
            path: path.to_string(),
            size: 1024,
            content_type: "application/octet-stream".to_string(),
            last_modified: SystemTime::now(),
            etag: Some("mock-etag".to_string()),
            metadata: HashMap::new(),
        })
    }
    
    async fn presigned_url(&self, path: &str, expiry_seconds: u64) -> Result<String, CloudError> {
        Ok(format!("https://mock-{}.com/{}/{}?expires={}", 
                  self.name.replace(" ", "-").to_lowercase(), 
                  self.bucket, path, expiry_seconds))
    }
    
    fn provider_name(&self) -> &'static str {
        "Mock Provider"
    }
}

/// Placeholder structs for other cloud services
#[derive(Debug, Clone)]
pub struct CloudFunction {
    pub name: String,
    pub provider: String,
    pub runtime: String,
    pub memory_mb: u32,
    pub timeout_seconds: u32,
}

impl Foreign for CloudFunction {
    fn type_name(&self) -> &'static str { "CloudFunction" }
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Name" => Ok(Value::String(self.name.clone())),
            "Provider" => Ok(Value::String(self.provider.clone())),
            "Runtime" => Ok(Value::String(self.runtime.clone())),
            "MemoryMB" => Ok(Value::Integer(self.memory_mb as i64)),
            "TimeoutSeconds" => Ok(Value::Integer(self.timeout_seconds as i64)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

#[derive(Debug, Clone)]
pub struct Container {
    pub image: String,
    pub name: String,
    pub ports: Vec<u16>,
    pub environment: HashMap<String, String>,
}

impl Foreign for Container {
    fn type_name(&self) -> &'static str { "Container" }
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Image" => Ok(Value::String(self.image.clone())),
            "Name" => Ok(Value::String(self.name.clone())),
            "PortCount" => Ok(Value::Integer(self.ports.len() as i64)),
            "EnvironmentCount" => Ok(Value::Integer(self.environment.len() as i64)),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

/// Docker client for container operations
#[derive(Debug, Clone)]
pub struct DockerClient {
    pub docker_host: String,
    pub api_version: String,
}

impl DockerClient {
    pub fn new() -> Self {
        DockerClient {
            docker_host: "unix:///var/run/docker.sock".to_string(),
            api_version: "1.41".to_string(),
        }
    }

    /// Execute docker command via CLI
    pub fn exec_docker_command(&self, args: &[&str]) -> Result<String, DockerError> {
        let output = Command::new("docker")
            .args(args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| DockerError::CommandFailed(format!("Failed to execute docker: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(DockerError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ))
        }
    }

    /// Pull container image
    pub fn pull_image(&self, image: &str) -> Result<String, DockerError> {
        self.exec_docker_command(&["pull", image])
    }

    /// Run container with configuration
    pub fn run_container(&self, config: &ContainerConfig) -> Result<String, DockerError> {
        let mut args = vec!["run", "-d"];
        let mut port_strs = Vec::new();
        let mut env_strs = Vec::new();
        let mut vol_strs = Vec::new();
        
        // Add name if specified
        if let Some(name) = &config.name {
            args.extend(&["--name", name]);
        }
        
        // Add port mappings
        for port_map in &config.port_mappings {
            let port_str = format!("{}:{}", port_map.host_port, port_map.container_port);
            port_strs.push(port_str);
        }
        for port_str in &port_strs {
            args.extend(&["-p", port_str]);
        }
        
        // Add environment variables
        for (key, value) in &config.environment {
            let env_str = format!("{}={}", key, value);
            env_strs.push(env_str);
        }
        for env_str in &env_strs {
            args.extend(&["-e", env_str]);
        }
        
        // Add volume mounts
        for volume in &config.volumes {
            let vol_str = format!("{}:{}", volume.host_path, volume.container_path);
            vol_strs.push(vol_str);
        }
        for vol_str in &vol_strs {
            args.extend(&["-v", vol_str]);
        }
        
        // Add resource limits
        if let Some(memory) = &config.memory_limit {
            args.extend(&["-m", memory]);
        }
        if let Some(cpus) = &config.cpu_limit {
            args.extend(&["--cpus", cpus]);
        }
        
        // Add image
        args.push(&config.image);
        
        // Add command if specified
        if let Some(cmd) = &config.command {
            args.extend(cmd.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        }
        
        self.exec_docker_command(&args)
    }

    /// Stop container
    pub fn stop_container(&self, container_id: &str) -> Result<String, DockerError> {
        self.exec_docker_command(&["stop", container_id])
    }

    /// Remove container
    pub fn remove_container(&self, container_id: &str, force: bool) -> Result<String, DockerError> {
        if force {
            self.exec_docker_command(&["rm", "-f", container_id])
        } else {
            self.exec_docker_command(&["rm", container_id])
        }
    }

    /// Get container logs
    pub fn get_logs(&self, container_id: &str, tail: Option<u32>) -> Result<String, DockerError> {
        let mut args = vec!["logs"];
        let tail_str;
        if let Some(n) = tail {
            tail_str = n.to_string();
            args.extend(&["--tail", &tail_str]);
        }
        args.push(container_id);
        self.exec_docker_command(&args)
    }

    /// Inspect container
    pub fn inspect_container(&self, container_id: &str) -> Result<String, DockerError> {
        self.exec_docker_command(&["inspect", container_id])
    }

    /// List running containers
    pub fn list_containers(&self, all: bool) -> Result<String, DockerError> {
        if all {
            self.exec_docker_command(&["ps", "-a"])
        } else {
            self.exec_docker_command(&["ps"])
        }
    }

    /// Execute command in container
    pub fn exec_in_container(&self, container_id: &str, command: &[String]) -> Result<String, DockerError> {
        let mut args = vec!["exec", container_id];
        args.extend(command.iter().map(|s| s.as_str()).collect::<Vec<_>>());
        self.exec_docker_command(&args)
    }
}

/// Docker container configuration
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    pub image: String,
    pub name: Option<String>,
    pub command: Option<Vec<String>>,
    pub environment: HashMap<String, String>,
    pub port_mappings: Vec<PortMapping>,
    pub volumes: Vec<VolumeMount>,
    pub memory_limit: Option<String>,
    pub cpu_limit: Option<String>,
    pub network_mode: Option<String>,
    pub restart_policy: Option<String>,
}

#[derive(Debug, Clone)]
pub struct PortMapping {
    pub host_port: u16,
    pub container_port: u16,
    pub protocol: String, // tcp, udp
}

#[derive(Debug, Clone)]
pub struct VolumeMount {
    pub host_path: String,
    pub container_path: String,
    pub read_only: bool,
}

/// Docker operation errors
#[derive(Debug, Clone)]
pub enum DockerError {
    CommandFailed(String),
    ContainerNotFound(String),
    ImageNotFound(String),
    InvalidConfiguration(String),
    NetworkError(String),
}

/// Docker container runtime information
#[derive(Debug, Clone)]
pub struct ContainerRuntime {
    pub container_id: String,
    pub image: String,
    pub status: ContainerStatus,
    pub created: SystemTime,
    pub ports: Vec<PortMapping>,
    pub environment: HashMap<String, String>,
    pub docker_client: DockerClient,
}

#[derive(Debug, Clone)]
pub enum ContainerStatus {
    Running,
    Stopped,
    Paused,
    Restarting,
    Removing,
    Dead,
    Created,
}

impl Foreign for ContainerRuntime {
    fn type_name(&self) -> &'static str {
        "ContainerRuntime"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "ContainerID" => Ok(Value::String(self.container_id.clone())),
            "Image" => Ok(Value::String(self.image.clone())),
            "Status" => Ok(Value::String(format!("{:?}", self.status))),
            "PortCount" => Ok(Value::Integer(self.ports.len() as i64)),
            "EnvironmentCount" => Ok(Value::Integer(self.environment.len() as i64)),
            
            "Stop" => {
                match self.docker_client.stop_container(&self.container_id) {
                    Ok(_) => Ok(Value::Boolean(true)),
                    Err(_) => Ok(Value::Boolean(false)),
                }
            },
            
            "Remove" => {
                let force = if args.len() > 0 {
                    matches!(args[0], Value::Boolean(true))
                } else {
                    false
                };
                match self.docker_client.remove_container(&self.container_id, force) {
                    Ok(_) => Ok(Value::Boolean(true)),
                    Err(_) => Ok(Value::Boolean(false)),
                }
            },
            
            "Logs" => {
                let tail = if args.len() > 0 {
                    match &args[0] {
                        Value::Integer(n) => Some(*n as u32),
                        _ => None,
                    }
                } else {
                    Some(100) // Default to last 100 lines
                };
                match self.docker_client.get_logs(&self.container_id, tail) {
                    Ok(logs) => Ok(Value::String(logs)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to get logs: {:?}", e))),
                }
            },
            
            "Inspect" => {
                match self.docker_client.inspect_container(&self.container_id) {
                    Ok(inspect_json) => Ok(Value::String(inspect_json)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to inspect: {:?}", e))),
                }
            },
            
            "Exec" => {
                if args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: 0,
                    });
                }
                let command = match &args[0] {
                    Value::String(cmd) => vec![cmd.clone()],
                    Value::List(items) => {
                        let mut cmd_parts = Vec::new();
                        for item in items {
                            if let Value::String(s) = item {
                                cmd_parts.push(s.clone());
                            }
                        }
                        cmd_parts
                    },
                    _ => return Err(ForeignError::InvalidArgument("Command must be string or list of strings".to_string())),
                };
                
                match self.docker_client.exec_in_container(&self.container_id, &command) {
                    Ok(output) => Ok(Value::String(output)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to exec: {:?}", e))),
                }
            },
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// =============================================================================
// Cloud Storage Functions
// =============================================================================

pub fn cloud_function(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (name, provider, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for function name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let provider = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for provider".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let function = CloudFunction {
        name, provider,
        runtime: "python3.9".to_string(),
        memory_mb: 128,
        timeout_seconds: 30,
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(function))))
}

pub fn cloud_storage(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (bucket, provider, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let bucket_name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for bucket name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let provider_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for provider".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let config = CloudStorageConfig {
        region: "us-east-1".to_string(),
        retry_count: 3,
        timeout_seconds: 30,
        encryption_enabled: false,
        versioning_enabled: false,
    };
    
    let provider: Arc<dyn CloudProvider> = match provider_name.as_str() {
        "s3" | "aws" => Arc::new(MockProvider { 
            name: "AWS S3".to_string(),
            bucket: bucket_name.clone(),
        }),
        "gcs" | "google" => Arc::new(MockProvider { 
            name: "Google Cloud Storage".to_string(),
            bucket: bucket_name.clone(),
        }),
        "azure" => Arc::new(MockProvider { 
            name: "Azure Blob Storage".to_string(),
            bucket: bucket_name.clone(),
        }),
        _ => return Err(VmError::TypeError {
            expected: "Provider name: s3, gcs, or azure".to_string(),
            actual: provider_name,
        }),
    };
    
    let storage = CloudStorage { bucket_name, provider, config };
    Ok(Value::LyObj(LyObj::new(Box::new(storage))))
}

pub fn cloud_upload(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "3 arguments (storage, path, data)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let _storage = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<CloudStorage>()
            .ok_or_else(|| VmError::TypeError {
                expected: "CloudStorage object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "CloudStorage object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for path".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let data = match &args[2] {
        Value::String(s) => Bytes::from(s.as_bytes().to_vec()),
        _ => return Err(VmError::TypeError {
            expected: "String for data".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let metadata_list = vec![
        Value::Rule {
            lhs: Box::new(Value::String("Path".to_string())),
            rhs: Box::new(Value::String(path)),
        },
        Value::Rule {
            lhs: Box::new(Value::String("Size".to_string())),
            rhs: Box::new(Value::Integer(data.len() as i64)),
        },
    ];
    
    Ok(Value::List(metadata_list))
}

pub fn cloud_download(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (storage, path)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    Ok(Value::String("Downloaded content".to_string()))
}

pub fn cloud_list(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (storage, [prefix])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mock_objects = vec![
        Value::List(vec![
            Value::Rule {
                lhs: Box::new(Value::String("Path".to_string())),
                rhs: Box::new(Value::String("example/file1.txt".to_string())),
            },
            Value::Rule {
                lhs: Box::new(Value::String("Size".to_string())),
                rhs: Box::new(Value::Integer(1024)),
            },
        ]),
    ];
    
    Ok(Value::List(mock_objects))
}

pub fn cloud_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (storage, path)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    Ok(Value::Boolean(true))
}

pub fn cloud_metadata(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (storage, path)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for path".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let metadata = vec![
        Value::Rule {
            lhs: Box::new(Value::String("Path".to_string())),
            rhs: Box::new(Value::String(path)),
        },
        Value::Rule {
            lhs: Box::new(Value::String("Size".to_string())),
            rhs: Box::new(Value::Integer(1024)),
        },
    ];
    
    Ok(Value::List(metadata))
}

pub fn cloud_presigned_url(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (storage, path, [expiry_seconds])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let storage = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<CloudStorage>()
            .ok_or_else(|| VmError::TypeError {
                expected: "CloudStorage object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "CloudStorage object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for path".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let expiry_seconds = if args.len() == 3 {
        match &args[2] {
            Value::Integer(n) => *n as u64,
            _ => 3600,
        }
    } else {
        3600
    };
    
    let url = format!("https://{}-{}.example.com/{}/{}?expires={}", 
                     storage.provider.provider_name().replace(" ", "-").to_lowercase(),
                     storage.bucket_name, 
                     storage.bucket_name, 
                     path, 
                     expiry_seconds);
    
    Ok(Value::String(url))
}

pub fn container_run(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (image, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let image = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Parse configuration if provided
    let mut config = ContainerConfig {
        image: image.clone(),
        name: None,
        command: None,
        environment: HashMap::new(),
        port_mappings: Vec::new(),
        volumes: Vec::new(),
        memory_limit: None,
        cpu_limit: None,
        network_mode: None,
        restart_policy: None,
    };
    
    if args.len() == 2 {
        match &args[1] {
            Value::List(config_items) => {
                for item in config_items {
                    match item {
                        Value::Rule { lhs, rhs } => {
                            if let Value::String(key) = lhs.as_ref() {
                                match key.as_str() {
                                    "Name" => {
                                        if let Value::String(name) = rhs.as_ref() {
                                            config.name = Some(name.clone());
                                        }
                                    },
                                    "Command" => {
                                        match rhs.as_ref() {
                                            Value::String(cmd) => config.command = Some(vec![cmd.clone()]),
                                            Value::List(items) => {
                                                let mut cmd_parts = Vec::new();
                                                for cmd_item in items {
                                                    if let Value::String(s) = cmd_item {
                                                        cmd_parts.push(s.clone());
                                                    }
                                                }
                                                config.command = Some(cmd_parts);
                                            },
                                            _ => {}
                                        }
                                    },
                                    "Memory" => {
                                        if let Value::String(mem) = rhs.as_ref() {
                                            config.memory_limit = Some(mem.clone());
                                        }
                                    },
                                    "CPU" => {
                                        if let Value::String(cpu) = rhs.as_ref() {
                                            config.cpu_limit = Some(cpu.clone());
                                        }
                                    },
                                    "Ports" => {
                                        if let Value::List(port_items) = rhs.as_ref() {
                                            for port_item in port_items {
                                                if let Value::String(port_spec) = port_item {
                                                    // Parse "host:container" format
                                                    let parts: Vec<&str> = port_spec.split(':').collect();
                                                    if parts.len() == 2 {
                                                        if let (Ok(host), Ok(container)) = (parts[0].parse::<u16>(), parts[1].parse::<u16>()) {
                                                            config.port_mappings.push(PortMapping {
                                                                host_port: host,
                                                                container_port: container,
                                                                protocol: "tcp".to_string(),
                                                            });
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    "Environment" => {
                                        if let Value::List(env_items) = rhs.as_ref() {
                                            for env_item in env_items {
                                                if let Value::Rule { lhs: env_key, rhs: env_value } = env_item {
                                                    if let (Value::String(k), Value::String(v)) = (env_key.as_ref(), env_value.as_ref()) {
                                                        config.environment.insert(k.clone(), v.clone());
                                                    }
                                                }
                                            }
                                        }
                                    },
                                    _ => {}
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            _ => return Err(VmError::TypeError {
                expected: "List of configuration rules".to_string(),
                actual: format!("{:?}", args[1]),
            }),
        }
    }
    
    // Create Docker client and run container
    let docker_client = DockerClient::new();
    
    match docker_client.run_container(&config) {
        Ok(container_output) => {
            // Parse container ID from output (typically first 12 characters)
            let container_id = container_output.trim().to_string();
            
            let runtime = ContainerRuntime {
                container_id: container_id.clone(),
                image: config.image.clone(),
                status: ContainerStatus::Running,
                created: SystemTime::now(),
                ports: config.port_mappings.clone(),
                environment: config.environment.clone(),
                docker_client,
            };
            
            Ok(Value::LyObj(LyObj::new(Box::new(runtime))))
        },
        Err(e) => Err(VmError::Runtime(format!("Failed to run container: {:?}", e))),
    }
}

/// ContainerStop[container] - Stop running container
pub fn container_stop(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (container)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let container = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<ContainerRuntime>()
            .ok_or_else(|| VmError::TypeError {
                expected: "ContainerRuntime object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "ContainerRuntime object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    match container.docker_client.stop_container(&container.container_id) {
        Ok(_) => Ok(Value::Boolean(true)),
        Err(e) => {
            eprintln!("Failed to stop container: {:?}", e);
            Ok(Value::Boolean(false))
        }
    }
}

/// ContainerLogs[container, options] - Retrieve container logs
pub fn container_logs(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (container, [options])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let container = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<ContainerRuntime>()
            .ok_or_else(|| VmError::TypeError {
                expected: "ContainerRuntime object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "ContainerRuntime object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let tail = if args.len() == 2 {
        match &args[1] {
            Value::Integer(n) => Some(*n as u32),
            _ => Some(100),
        }
    } else {
        Some(100)
    };
    
    match container.docker_client.get_logs(&container.container_id, tail) {
        Ok(logs) => Ok(Value::String(logs)),
        Err(e) => Err(VmError::Runtime(format!("Failed to get logs: {:?}", e))),
    }
}

/// ContainerInspect[container] - Get container metadata/status
pub fn container_inspect(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (container)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let container = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<ContainerRuntime>()
            .ok_or_else(|| VmError::TypeError {
                expected: "ContainerRuntime object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "ContainerRuntime object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    match container.docker_client.inspect_container(&container.container_id) {
        Ok(inspect_json) => Ok(Value::String(inspect_json)),
        Err(e) => Err(VmError::Runtime(format!("Failed to inspect container: {:?}", e))),
    }
}

/// ContainerExec[container, command] - Execute commands in container
pub fn container_exec(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (container, command)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let container = match &args[0] {
        Value::LyObj(obj) => obj.downcast_ref::<ContainerRuntime>()
            .ok_or_else(|| VmError::TypeError {
                expected: "ContainerRuntime object".to_string(),
                actual: format!("{:?}", args[0]),
            })?,
        _ => return Err(VmError::TypeError {
            expected: "ContainerRuntime object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let command = match &args[1] {
        Value::String(cmd) => vec![cmd.clone()],
        Value::List(items) => {
            let mut cmd_parts = Vec::new();
            for item in items {
                if let Value::String(s) = item {
                    cmd_parts.push(s.clone());
                }
            }
            cmd_parts
        },
        _ => return Err(VmError::TypeError {
            expected: "String or list of strings for command".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    match container.docker_client.exec_in_container(&container.container_id, &command) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to execute command: {:?}", e))),
    }
}

/// ContainerList[] - List all containers
pub fn container_list(args: &[Value]) -> VmResult<Value> {
    if args.len() > 1 {
        return Err(VmError::TypeError {
            expected: "0-1 arguments ([show_all])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let show_all = if args.len() == 1 {
        matches!(args[0], Value::Boolean(true))
    } else {
        false
    };
    
    let docker_client = DockerClient::new();
    match docker_client.list_containers(show_all) {
        Ok(container_list) => Ok(Value::String(container_list)),
        Err(e) => Err(VmError::Runtime(format!("Failed to list containers: {:?}", e))),
    }
}

/// ContainerPull[image] - Pull container image
pub fn container_pull(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (image)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let image = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for image".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let docker_client = DockerClient::new();
    match docker_client.pull_image(&image) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to pull image: {:?}", e))),
    }
}

pub fn kubernetes_service(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("KubernetesService not yet implemented".to_string()))
}

pub fn cloud_deploy(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("CloudDeploy not yet implemented".to_string()))
}

pub fn cloud_monitor(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("CloudMonitor not yet implemented".to_string()))
}