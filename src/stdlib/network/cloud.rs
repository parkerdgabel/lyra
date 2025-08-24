//! Cloud & Infrastructure Integration

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::any::Any;
use std::collections::{HashMap, HashSet};
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
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
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

/// AWS Lambda client for serverless function operations
#[derive(Debug, Clone)]
pub struct LambdaClient {
    pub region: String,
    pub profile: Option<String>,
}

impl LambdaClient {
    pub fn new(region: String, profile: Option<String>) -> Self {
        LambdaClient { region, profile }
    }

    /// Execute AWS CLI command for Lambda operations
    pub fn exec_aws_lambda_command(&self, args: &[&str]) -> Result<String, LambdaError> {
        let mut cmd_args = vec!["lambda"];
        cmd_args.extend(args);
        cmd_args.extend(&["--region", &self.region]);
        
        if let Some(profile) = &self.profile {
            cmd_args.extend(&["--profile", profile]);
        }

        let output = Command::new("aws")
            .args(&cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| LambdaError::CommandFailed(format!("Failed to execute aws: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(LambdaError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ))
        }
    }

    /// Create Lambda function
    pub fn create_function(&self, config: &FunctionConfig) -> Result<String, LambdaError> {
        let memory_str = config.memory_mb.to_string();
        let timeout_str = config.timeout_seconds.to_string();
        
        // Create owned strings for lifetime management
        let zip_file_str;
        let code_spec_str;
        let image_uri_str;
        let env_string;
        
        let mut args = vec![
            "create-function",
            "--function-name", &config.name,
            "--runtime", &config.runtime,
            "--role", &config.role_arn,
            "--handler", &config.handler,
            "--memory-size", &memory_str,
            "--timeout", &timeout_str,
        ];

        // Add code source 
        match &config.code_source {
            CodeSource::ZipFile(path) => {
                zip_file_str = format!("fileb://{}", path);
                args.extend(&["--zip-file", &zip_file_str]);
            },
            CodeSource::S3Bucket { bucket, key, version } => {
                code_spec_str = if let Some(v) = version {
                    format!("S3Bucket={},S3Key={},S3ObjectVersion={}", bucket, key, v)
                } else {
                    format!("S3Bucket={},S3Key={}", bucket, key)
                };
                args.extend(&["--code", &code_spec_str]);
            },
            CodeSource::ImageUri(uri) => {
                image_uri_str = format!("ImageUri={}", uri);
                args.extend(&["--code", &image_uri_str]);
                args.extend(&["--package-type", "Image"]);
            },
        }

        // Add environment variables
        if !config.environment.is_empty() {
            let env_vars: Vec<String> = config.environment.iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            env_string = format!("Variables={{{}}}", env_vars.join(","));
            args.extend(&["--environment", &env_string]);
        }

        self.exec_aws_lambda_command(&args)
    }

    /// Invoke Lambda function
    pub fn invoke_function(&self, function_name: &str, payload: Option<&str>, invocation_type: InvocationType) -> Result<String, LambdaError> {
        let mut args = vec![
            "invoke",
            "--function-name", function_name,
        ];

        // Add invocation type
        let invocation_str = match invocation_type {
            InvocationType::RequestResponse => "RequestResponse",
            InvocationType::Event => "Event",
            InvocationType::DryRun => "DryRun",
        };
        args.extend(&["--invocation-type", invocation_str]);

        // Add payload if provided
        if let Some(payload_data) = payload {
            args.extend(&["--payload", payload_data]);
        }

        // Output file for response
        args.push("/tmp/lambda_response.json");

        self.exec_aws_lambda_command(&args)
    }

    /// Delete Lambda function
    pub fn delete_function(&self, function_name: &str) -> Result<String, LambdaError> {
        self.exec_aws_lambda_command(&[
            "delete-function",
            "--function-name", function_name,
        ])
    }

    /// Update Lambda function code
    pub fn update_function_code(&self, function_name: &str, code_source: &str) -> Result<String, LambdaError> {
        let zip_file_str;
        let image_uri_str;
        
        let args = if code_source.ends_with(".zip") {
            zip_file_str = format!("fileb://{}", code_source);
            vec![
                "update-function-code",
                "--function-name", function_name,
                "--zip-file", &zip_file_str,
            ]
        } else if code_source.starts_with("s3://") {
            vec![
                "update-function-code", 
                "--function-name", function_name,
                "--s3-bucket", code_source,
                "--s3-key", "lambda-deployment.zip",
            ]
        } else {
            image_uri_str = format!("ImageUri={}", code_source);
            vec![
                "update-function-code",
                "--function-name", function_name,
                "--image-uri", &image_uri_str,
            ]
        };

        self.exec_aws_lambda_command(&args)
    }

    /// Get CloudWatch logs for Lambda function  
    pub fn get_function_logs(&self, function_name: &str, hours: u32) -> Result<String, LambdaError> {
        let log_group = format!("/aws/lambda/{}", function_name);
        let start_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_millis() - (hours as u128 * 60 * 60 * 1000);
        
        let start_time_str = start_time.to_string();
        
        let args = vec![
            "logs", "filter-log-events",
            "--log-group-name", &log_group,
            "--start-time", &start_time_str,
            "--query", "events[*].[timestamp,message]",
            "--output", "text",
        ];

        self.exec_aws_lambda_command(&args)
    }

    /// Get function performance metrics
    pub fn get_function_metrics(&self, function_name: &str) -> Result<String, LambdaError> {
        let end_time = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let start_time = end_time - 3600; // Last hour
        
        let start_time_str = format!("{}000", start_time); // CloudWatch uses milliseconds
        let end_time_str = format!("{}000", end_time);
        let dimensions_str = format!("Name=FunctionName,Value={}", function_name);
        
        let args = vec![
            "cloudwatch", "get-metric-statistics",
            "--namespace", "AWS/Lambda",
            "--metric-name", "Invocations",
            "--dimensions", &dimensions_str,
            "--statistics", "Sum",
            "--start-time", &start_time_str,
            "--end-time", &end_time_str,
            "--period", "300",
        ];

        self.exec_aws_lambda_command(&args)
    }

    /// Get function logs
    pub fn get_logs(&self, function_name: &str, start_time: Option<&str>) -> Result<String, LambdaError> {
        let log_group = format!("/aws/lambda/{}", function_name);
        let mut cmd_args = vec![
            "logs", "filter-log-events",
            "--log-group-name", &log_group,
            "--region", &self.region,
        ];

        if let Some(start) = start_time {
            cmd_args.extend(&["--start-time", start]);
        }

        let output = Command::new("aws")
            .args(&cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| LambdaError::CommandFailed(format!("Failed to get logs: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(LambdaError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ))
        }
    }
}

/// Lambda function configuration
#[derive(Debug, Clone)]
pub struct FunctionConfig {
    pub name: String,
    pub runtime: String,
    pub handler: String,
    pub role_arn: String,
    pub code_source: CodeSource,
    pub environment: HashMap<String, String>,
    pub memory_mb: u32,
    pub timeout_seconds: u32,
}

/// Lambda function code source
#[derive(Debug, Clone)]
pub enum CodeSource {
    ZipFile(String),               // Local zip file path
    S3Bucket {                     // S3 bucket source
        bucket: String,
        key: String,
        version: Option<String>,
    },
    ImageUri(String),              // Container image URI
}

/// Lambda invocation type
#[derive(Debug, Clone)]
pub enum InvocationType {
    RequestResponse,  // Synchronous invocation
    Event,           // Asynchronous invocation
    DryRun,          // Validate parameters and verify IAM permissions
}

/// Lambda operation errors
#[derive(Debug, Clone)]
pub enum LambdaError {
    CommandFailed(String),
    FunctionNotFound(String),
    InvalidConfiguration(String),
    DeploymentFailed(String),
    InvocationFailed(String),
}

/// Cloud Function with serverless capabilities
#[derive(Debug, Clone)]
pub struct CloudFunction {
    pub name: String,
    pub provider: String,
    pub runtime: String,
    pub memory_mb: u32,
    pub timeout_seconds: u32,
    pub handler: String,
    pub role_arn: Option<String>,
    pub environment: HashMap<String, String>,
    pub lambda_client: Option<LambdaClient>,
    pub status: FunctionStatus,
    pub last_modified: SystemTime,
}

#[derive(Debug, Clone)]
pub enum FunctionStatus {
    Pending,
    Active,
    Inactive,
    Failed,
}

impl Foreign for CloudFunction {
    fn type_name(&self) -> &'static str { "CloudFunction" }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Name" => Ok(Value::String(self.name.clone())),
            "Provider" => Ok(Value::String(self.provider.clone())),
            "Runtime" => Ok(Value::String(self.runtime.clone())),
            "MemoryMB" => Ok(Value::Integer(self.memory_mb as i64)),
            "TimeoutSeconds" => Ok(Value::Integer(self.timeout_seconds as i64)),
            "Handler" => Ok(Value::String(self.handler.clone())),
            "Status" => Ok(Value::String(format!("{:?}", self.status))),
            
            "Invoke" => {
                if let Some(client) = &self.lambda_client {
                    let payload = if args.len() > 0 {
                        match &args[0] {
                            Value::String(s) => Some(s.as_str()),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    
                    let invocation_type = if args.len() > 1 {
                        match &args[1] {
                            Value::String(s) if s == "Event" => InvocationType::Event,
                            Value::String(s) if s == "DryRun" => InvocationType::DryRun,
                            _ => InvocationType::RequestResponse,
                        }
                    } else {
                        InvocationType::RequestResponse
                    };
                    
                    match client.invoke_function(&self.name, payload, invocation_type) {
                        Ok(result) => Ok(Value::String(result)),
                        Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to invoke: {:?}", e))),
                    }
                } else {
                    Err(ForeignError::InvalidArgument("No Lambda client available".to_string()))
                }
            },
            
            "Delete" => {
                if let Some(client) = &self.lambda_client {
                    match client.delete_function(&self.name) {
                        Ok(_) => Ok(Value::Boolean(true)),
                        Err(_) => Ok(Value::Boolean(false)),
                    }
                } else {
                    Ok(Value::Boolean(false))
                }
            },
            
            "Logs" => {
                if let Some(client) = &self.lambda_client {
                    let start_time = if args.len() > 0 {
                        match &args[0] {
                            Value::String(s) => Some(s.as_str()),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    
                    match client.get_logs(&self.name, start_time) {
                        Ok(logs) => Ok(Value::String(logs)),
                        Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to get logs: {:?}", e))),
                    }
                } else {
                    Err(ForeignError::InvalidArgument("No Lambda client available".to_string()))
                }
            },
            
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
        name, 
        provider,
        runtime: "python3.9".to_string(),
        memory_mb: 128,
        timeout_seconds: 30,
        handler: "lambda_function.lambda_handler".to_string(),
        role_arn: None,
        environment: HashMap::new(),
        lambda_client: None,
        status: FunctionStatus::Pending,
        last_modified: SystemTime::now(),
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

// =============================================================================
// Kubernetes Integration
// =============================================================================

/// Kubernetes client for cluster operations
#[derive(Debug, Clone)]
pub struct KubernetesClient {
    pub kubeconfig_path: String,
    pub current_context: String,
    pub timeout_seconds: u32,
}

impl KubernetesClient {
    pub fn new(kubeconfig_path: String, context: Option<String>) -> Self {
        KubernetesClient {
            kubeconfig_path,
            current_context: context.unwrap_or_else(|| "default".to_string()),
            timeout_seconds: 60,
        }
    }

    /// Execute kubectl command with proper error handling
    pub fn exec_kubectl_command(&self, args: &[&str]) -> Result<String, KubernetesError> {
        let mut cmd_args = vec!["--kubeconfig", &self.kubeconfig_path];
        if !self.current_context.is_empty() && self.current_context != "default" {
            cmd_args.extend(&["--context", &self.current_context]);
        }
        cmd_args.extend(args);

        let output = Command::new("kubectl")
            .args(&cmd_args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| KubernetesError::CommandFailed(format!("Failed to execute kubectl: {}", e)))?;

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(KubernetesError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string()
            ))
        }
    }

    /// Apply YAML manifest to cluster
    pub fn apply_manifest(&self, manifest_path: &str, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let mut args = vec!["apply", "-f", manifest_path];
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        self.exec_kubectl_command(&args)
    }

    /// Delete resource by name and type
    pub fn delete_resource(&self, resource_type: &str, resource_name: &str, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let mut args = vec!["delete", resource_type, resource_name];
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        self.exec_kubectl_command(&args)
    }

    /// Scale deployment
    pub fn scale_deployment(&self, deployment_name: &str, replicas: u32, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let replica_str = replicas.to_string();
        let mut args = vec!["scale", "deployment", deployment_name, "--replicas", &replica_str];
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        self.exec_kubectl_command(&args)
    }

    /// Get pod logs
    pub fn get_pod_logs(&self, pod_name: &str, container: Option<&str>, namespace: Option<&str>, tail: Option<u32>) -> Result<String, KubernetesError> {
        let mut args: Vec<String> = vec!["logs".to_string(), pod_name.to_string()];
        if let Some(c) = container {
            args.extend(["-c".to_string(), c.to_string()]);
        }
        if let Some(ns) = namespace {
            args.extend(["-n".to_string(), ns.to_string()]);
        }
        if let Some(lines) = tail {
            args.extend(["--tail".to_string(), lines.to_string()]);
        }
        let arg_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
        self.exec_kubectl_command(&arg_refs)
    }

    /// Get resource information
    pub fn get_resource(&self, resource_type: &str, resource_name: Option<&str>, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let mut args = vec!["get", resource_type];
        if let Some(name) = resource_name {
            args.push(name);
        }
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        args.extend(&["-o", "yaml"]);
        self.exec_kubectl_command(&args)
    }

    /// Create configmap from data
    pub fn create_configmap(&self, name: &str, data: &HashMap<String, String>, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let mut args = vec!["create", "configmap", name];
        
        // Add data as --from-literal arguments
        let mut literal_args = Vec::new();
        for (key, value) in data {
            let literal = format!("{}={}", key, value);
            literal_args.push(literal);
        }
        
        for literal in &literal_args {
            args.extend(&["--from-literal", literal]);
        }
        
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        
        self.exec_kubectl_command(&args)
    }

    /// Expose deployment as service
    pub fn expose_deployment(&self, deployment_name: &str, port: u16, target_port: Option<u16>, service_type: &str, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let port_str = port.to_string();
        let mut args_vec = vec![
            "expose".to_string(),
            "deployment".to_string(),
            deployment_name.to_string(),
            "--port".to_string(),
            port_str,
            "--type".to_string(),
            service_type.to_string(),
        ];
        
        if let Some(tp) = target_port {
            args_vec.push("--target-port".to_string());
            args_vec.push(tp.to_string());
        }
        
        if let Some(ns) = namespace {
            args_vec.push("-n".to_string());
            args_vec.push(ns.to_string());
        }
        
        let args: Vec<&str> = args_vec.iter().map(String::as_str).collect();
        self.exec_kubectl_command(&args)
    }

    /// Rolling update deployment image
    pub fn rolling_update(&self, deployment_name: &str, image: &str, namespace: Option<&str>) -> Result<String, KubernetesError> {
        let image_spec = format!("{}={}", deployment_name, image);
        let deployment_spec = format!("deployment/{}", deployment_name);
        let mut args = vec!["set", "image", deployment_spec.as_str(), image_spec.as_str()];
        
        if let Some(ns) = namespace {
            args.extend(&["-n", ns]);
        }
        
        self.exec_kubectl_command(&args)
    }
}

/// Kubernetes cluster configuration
#[derive(Debug, Clone)]
pub struct KubernetesCluster {
    pub name: String,
    pub kubeconfig_path: String,
    pub current_context: String,
    pub namespaces: HashSet<String>,
    pub client: KubernetesClient,
    pub status: ClusterStatus,
    pub created: SystemTime,
}

#[derive(Debug, Clone)]
pub enum ClusterStatus {
    Connected,
    Disconnected,
    Error(String),
}

/// Kubernetes resource types
#[derive(Debug, Clone)]
pub enum KubernetesResource {
    Pod { name: String, namespace: String, status: String },
    Service { name: String, namespace: String, port: u16 },
    Deployment { name: String, namespace: String, replicas: u32 },
    ConfigMap { name: String, namespace: String, data: HashMap<String, String> },
    Secret { name: String, namespace: String, data_keys: Vec<String> },
}

/// Kubernetes operation errors
#[derive(Debug, Clone)]
pub enum KubernetesError {
    CommandFailed(String),
    ClusterUnreachable(String),
    ResourceNotFound(String),
    InvalidManifest(String),
    PermissionDenied(String),
    NamespaceError(String),
}

impl Foreign for KubernetesCluster {
    fn type_name(&self) -> &'static str { "KubernetesCluster" }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Name" => Ok(Value::String(self.name.clone())),
            "Context" => Ok(Value::String(self.current_context.clone())),
            "Status" => Ok(Value::String(format!("{:?}", self.status))),
            "NamespaceCount" => Ok(Value::Integer(self.namespaces.len() as i64)),
            
            "Apply" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let manifest_path = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgument("Manifest path must be string".to_string())),
                };
                
                let namespace = if args.len() > 1 {
                    match &args[1] {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    }
                } else {
                    None
                };
                
                match self.client.apply_manifest(&manifest_path, namespace) {
                    Ok(output) => Ok(Value::String(output)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to apply manifest: {:?}", e))),
                }
            },
            
            "Scale" => {
                if args.len() != 3 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 3,
                        actual: args.len(),
                    });
                }
                
                let deployment = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgument("Deployment name must be string".to_string())),
                };
                
                let replicas = match &args[1] {
                    Value::Integer(n) => *n as u32,
                    _ => return Err(ForeignError::InvalidArgument("Replicas must be integer".to_string())),
                };
                
                let namespace = match &args[2] {
                    Value::String(s) => Some(s.as_str()),
                    _ => None,
                };
                
                match self.client.scale_deployment(&deployment, replicas, namespace) {
                    Ok(output) => Ok(Value::String(output)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to scale deployment: {:?}", e))),
                }
            },
            
            "Logs" => {
                if args.len() < 1 || args.len() > 4 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let pod_name = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgument("Pod name must be string".to_string())),
                };
                
                let container = if args.len() > 1 {
                    match &args[1] {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let namespace = if args.len() > 2 {
                    match &args[2] {
                        Value::String(s) => Some(s.as_str()),
                        _ => None,
                    }
                } else {
                    None
                };
                
                let tail = if args.len() > 3 {
                    match &args[3] {
                        Value::Integer(n) => Some(*n as u32),
                        _ => None,
                    }
                } else {
                    Some(100)
                };
                
                match self.client.get_pod_logs(&pod_name, container, namespace, tail) {
                    Ok(logs) => Ok(Value::String(logs)),
                    Err(e) => Err(ForeignError::InvalidArgument(format!("Failed to get logs: {:?}", e))),
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

pub fn kubernetes_service(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (name, [kubeconfig_path], [context])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for cluster name".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let kubeconfig_path = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => std::env::var("KUBECONFIG").unwrap_or_else(|_| format!("{}/.kube/config", std::env::var("HOME").unwrap_or_default())),
        }
    } else {
        std::env::var("KUBECONFIG").unwrap_or_else(|_| format!("{}/.kube/config", std::env::var("HOME").unwrap_or_default()))
    };
    
    let context = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.clone()),
            _ => None,
        }
    } else {
        None
    };
    
    let client = KubernetesClient::new(kubeconfig_path.clone(), context.clone());
    
    let cluster = KubernetesCluster {
        name,
        kubeconfig_path,
        current_context: context.unwrap_or_else(|| "default".to_string()),
        namespaces: HashSet::new(),
        client,
        status: ClusterStatus::Connected,
        created: SystemTime::now(),
    };
    
    Ok(Value::LyObj(LyObj::new(Box::new(cluster))))
}

// =============================================================================
// Kubernetes API Functions
// =============================================================================

/// KubernetesDeploy[cluster, manifest_path, namespace] - Deploy YAML manifest
pub fn kubernetes_deploy(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (cluster, manifest_path, [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let manifest_path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for manifest path".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let namespace = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.apply_manifest(&manifest_path, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to deploy manifest: {:?}", e))),
    }
}

/// DeploymentScale[cluster, deployment_name, replicas, namespace] - Scale deployment
pub fn deployment_scale(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (cluster, deployment, replicas, [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let deployment = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for deployment name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let replicas = match &args[2] {
        Value::Integer(n) => *n as u32,
        _ => return Err(VmError::TypeError {
            expected: "Integer for replica count".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.scale_deployment(&deployment, replicas, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to scale deployment: {:?}", e))),
    }
}

/// RollingUpdate[cluster, deployment, image, namespace] - Update deployment image
pub fn rolling_update(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (cluster, deployment, image, [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let deployment = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for deployment name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let image = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for image name".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.rolling_update(&deployment, &image, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to update deployment: {:?}", e))),
    }
}

/// ConfigMapCreate[cluster, name, data, namespace] - Create ConfigMap
pub fn configmap_create(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (cluster, name, data, [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for ConfigMap name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let mut data = HashMap::new();
    match &args[2] {
        Value::List(items) => {
            for item in items {
                if let Value::Rule { lhs, rhs } = item {
                    if let (Value::String(k), Value::String(v)) = (lhs.as_ref(), rhs.as_ref()) {
                        data.insert(k.clone(), v.clone());
                    }
                }
            }
        },
        _ => return Err(VmError::TypeError {
            expected: "List of rules for ConfigMap data".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    }
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.create_configmap(&name, &data, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to create ConfigMap: {:?}", e))),
    }
}

/// ServiceExpose[cluster, deployment, port, service_type, namespace] - Expose deployment as service
pub fn service_expose(args: &[Value]) -> VmResult<Value> {
    if args.len() < 4 || args.len() > 6 {
        return Err(VmError::TypeError {
            expected: "4-6 arguments (cluster, deployment, port, service_type, [target_port], [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let deployment = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for deployment name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let port = match &args[2] {
        Value::Integer(n) => *n as u16,
        _ => return Err(VmError::TypeError {
            expected: "Integer for service port".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let service_type = match &args[3] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for service type".to_string(),
            actual: format!("{:?}", args[3]),
        }),
    };
    
    let target_port = if args.len() > 4 {
        match &args[4] {
            Value::Integer(n) => Some(*n as u16),
            _ => None,
        }
    } else {
        None
    };
    
    let namespace = if args.len() > 5 {
        match &args[5] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.expose_deployment(&deployment, port, target_port, &service_type, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to expose service: {:?}", e))),
    }
}

/// PodLogs[cluster, pod_name, container, namespace, tail] - Get pod logs
pub fn pod_logs(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 5 {
        return Err(VmError::TypeError {
            expected: "2-5 arguments (cluster, pod_name, [container], [namespace], [tail])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let pod_name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for pod name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let container = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    let tail = if args.len() > 4 {
        match &args[4] {
            Value::Integer(n) => Some(*n as u32),
            _ => Some(100),
        }
    } else {
        Some(100)
    };
    
    match cluster.client.get_pod_logs(&pod_name, container, namespace, tail) {
        Ok(logs) => Ok(Value::String(logs)),
        Err(e) => Err(VmError::Runtime(format!("Failed to get pod logs: {:?}", e))),
    }
}

/// ResourceGet[cluster, resource_type, resource_name, namespace] - Get resource information
pub fn resource_get(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "2-4 arguments (cluster, resource_type, [resource_name], [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let resource_type = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for resource type".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let resource_name = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.get_resource(&resource_type, resource_name, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to get resource: {:?}", e))),
    }
}

/// ResourceDelete[cluster, resource_type, resource_name, namespace] - Delete resource
pub fn resource_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() < 3 || args.len() > 4 {
        return Err(VmError::TypeError {
            expected: "3-4 arguments (cluster, resource_type, resource_name, [namespace])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let cluster = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<KubernetesCluster>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "KubernetesCluster object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "KubernetesCluster object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let resource_type = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for resource type".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let resource_name = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for resource name".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    let namespace = if args.len() > 3 {
        match &args[3] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    match cluster.client.delete_resource(&resource_type, &resource_name, namespace) {
        Ok(output) => Ok(Value::String(output)),
        Err(e) => Err(VmError::Runtime(format!("Failed to delete resource: {:?}", e))),
    }
}

pub fn cloud_deploy(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("CloudDeploy not yet implemented".to_string()))
}

pub fn cloud_monitor(_args: &[Value]) -> VmResult<Value> {
    Err(VmError::Runtime("CloudMonitor not yet implemented".to_string()))
}

// =============================================================================
// CloudFunction API Functions  
// =============================================================================

/// CloudFunctionDeploy[function, code_source] - Deploy Lambda function
pub fn cloud_function_deploy(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "2-3 arguments (function, code_source, [config])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(), 
                    actual: format!("{:?}", args[0]),
                })?
                .clone()
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let code_source = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for code source (zip file path or S3 bucket)".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // Parse configuration if provided
    if args.len() == 3 {
        if let Value::List(config_rules) = &args[2] {
            for rule in config_rules {
                if let Value::Rule { lhs, rhs } = rule {
                    if let Value::String(key) = lhs.as_ref() {
                        match key.as_str() {
                            "Runtime" => {
                                if let Value::String(runtime) = rhs.as_ref() {
                                    function.runtime = runtime.clone();
                                }
                            },
                            "Memory" => {
                                if let Value::Integer(mem) = rhs.as_ref() {
                                    function.memory_mb = *mem as u32;
                                }
                            },
                            "Timeout" => {
                                if let Value::Integer(timeout) = rhs.as_ref() {
                                    function.timeout_seconds = *timeout as u32;
                                }
                            },
                            "Handler" => {
                                if let Value::String(handler) = rhs.as_ref() {
                                    function.handler = handler.clone();
                                }
                            },
                            "Role" => {
                                if let Value::String(role) = rhs.as_ref() {
                                    function.role_arn = Some(role.clone());
                                }
                            },
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    
    // Create Lambda client and deploy function
    let client = LambdaClient::new("us-east-1".to_string(), None);
    
    let code = if code_source.ends_with(".zip") {
        CodeSource::ZipFile(code_source)
    } else if code_source.starts_with("s3://") {
        CodeSource::S3Bucket { 
            bucket: code_source.clone(), 
            key: "lambda-deployment.zip".to_string(),
            version: None,
        }
    } else {
        CodeSource::ImageUri(code_source) 
    };
    
    let config = FunctionConfig {
        name: function.name.clone(),
        runtime: function.runtime.clone(),
        handler: function.handler.clone(),
        memory_mb: function.memory_mb,
        timeout_seconds: function.timeout_seconds,
        role_arn: function.role_arn.clone().unwrap_or_else(|| "arn:aws:iam::123456789012:role/lambda-execution-role".to_string()),
        environment: function.environment.clone(),
        code_source: code,
    };
    
    match client.create_function(&config) {
        Ok(_arn) => {
            function.lambda_client = Some(client);
            function.status = FunctionStatus::Active;
            function.last_modified = SystemTime::now();
            
            Ok(Value::LyObj(LyObj::new(Box::new(function))))
        },
        Err(e) => Err(VmError::Runtime(format!("Failed to deploy function: {:?}", e))),
    }
}

/// CloudFunctionInvoke[function, payload, invocation_type] - Invoke Lambda function
pub fn cloud_function_invoke(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 3 {
        return Err(VmError::TypeError {
            expected: "1-3 arguments (function, [payload], [invocation_type])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let payload = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => Some(s.as_str()),
            _ => None,
        }
    } else {
        None
    };
    
    let invocation_type = if args.len() > 2 {
        match &args[2] {
            Value::String(s) => match s.as_str() {
                "Event" => InvocationType::Event,
                "DryRun" => InvocationType::DryRun,
                _ => InvocationType::RequestResponse,
            },
            _ => InvocationType::RequestResponse,
        }
    } else {
        InvocationType::RequestResponse
    };
    
    if let Some(client) = &function.lambda_client {
        match client.invoke_function(&function.name, payload, invocation_type) {
            Ok(result) => Ok(Value::String(result)),
            Err(e) => Err(VmError::Runtime(format!("Failed to invoke function: {:?}", e))),
        }
    } else {
        Err(VmError::Runtime("No Lambda client available - function may not be deployed".to_string()))
    }
}

/// CloudFunctionUpdate[function, code_source] - Update Lambda function code
pub fn cloud_function_update(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (function, code_source)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let mut function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
                .clone()
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let code_source = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err(VmError::TypeError {
            expected: "String for code source".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    if let Some(client) = &function.lambda_client {
        match client.update_function_code(&function.name, &code_source) {
            Ok(_) => {
                function.last_modified = SystemTime::now();
                Ok(Value::LyObj(LyObj::new(Box::new(function))))
            },
            Err(e) => Err(VmError::Runtime(format!("Failed to update function: {:?}", e))),
        }
    } else {
        Err(VmError::Runtime("No Lambda client available".to_string()))
    }
}

/// CloudFunctionLogs[function, hours] - Get CloudWatch logs for function
pub fn cloud_function_logs(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "1-2 arguments (function, [hours])".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let hours = if args.len() > 1 {
        match &args[1] {
            Value::Integer(h) => *h as u32,
            _ => 1,
        }
    } else {
        1
    };
    
    if let Some(client) = &function.lambda_client {
        match client.get_function_logs(&function.name, hours) {
            Ok(logs) => Ok(Value::String(logs)),
            Err(e) => Err(VmError::Runtime(format!("Failed to get logs: {:?}", e))),
        }
    } else {
        Err(VmError::Runtime("No Lambda client available".to_string()))
    }
}

/// CloudFunctionMetrics[function] - Get function performance metrics
pub fn cloud_function_metrics(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (function)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if let Some(client) = &function.lambda_client {
        match client.get_function_metrics(&function.name) {
            Ok(metrics) => Ok(Value::String(metrics)),
            Err(e) => Err(VmError::Runtime(format!("Failed to get metrics: {:?}", e))),
        }
    } else {
        Err(VmError::Runtime("No Lambda client available".to_string()))
    }
}

/// CloudFunctionDelete[function] - Delete Lambda function
pub fn cloud_function_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (function)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let function = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CloudFunction>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CloudFunction object".to_string(),
                    actual: format!("{:?}", args[0]),
                })?
        },
        _ => return Err(VmError::TypeError {
            expected: "CloudFunction object".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if let Some(client) = &function.lambda_client {
        match client.delete_function(&function.name) {
            Ok(_) => Ok(Value::Boolean(true)),
            Err(_) => Ok(Value::Boolean(false)),
        }
    } else {
        Ok(Value::Boolean(false))
    }
}
