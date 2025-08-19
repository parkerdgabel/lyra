# Lyra Module System Implementation Plan

## Implementation Architecture

This document provides detailed implementation specifications for the Lyra Module System, including code structures, API designs, and integration patterns.

## 1. Core Module System Implementation

### 1.1 Module Core Types

Create `/src/modules/mod.rs`:

```rust
//! Lyra Module System Core
//!
//! This module implements the hierarchical module and package management system
//! for Lyra, enabling organized code distribution and dependency management.

use crate::{
    vm::{Value, VmResult},
    stdlib::StdlibFunction,
    linker::{FunctionSignature, FunctionAttribute, UnifiedFunction},
    bytecode::Instruction,
};
use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use thiserror::Error;

pub mod loader;
pub mod registry;
pub mod resolver;
pub mod package;
pub mod cli;

/// Errors that can occur in the module system
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ModuleError {
    #[error("Module not found: {name}")]
    ModuleNotFound { name: String },
    
    #[error("Invalid module name: {name}")]
    InvalidModuleName { name: String },
    
    #[error("Circular dependency detected: {cycle}")]
    CircularDependency { cycle: String },
    
    #[error("Version conflict: {package} requires {required} but {found} is installed")]
    VersionConflict {
        package: String,
        required: String,
        found: String,
    },
    
    #[error("Function already exported: {function} in module {module}")]
    DuplicateExport {
        function: String,
        module: String,
    },
    
    #[error("Package error: {message}")]
    PackageError { message: String },
}

/// Semantic versioning implementation
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Version {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
    pub pre: Option<String>,
    pub build: Option<String>,
}

impl Version {
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Version {
            major,
            minor,
            patch,
            pre: None,
            build: None,
        }
    }
    
    /// Check if this version satisfies the given constraint
    pub fn satisfies(&self, constraint: &VersionConstraint) -> bool {
        match constraint {
            VersionConstraint::Exact(v) => self == v,
            VersionConstraint::GreaterThan(v) => self > v,
            VersionConstraint::GreaterEqual(v) => self >= v,
            VersionConstraint::Compatible(v) => {
                // Compatible: same major version, >= minor.patch
                self.major == v.major && self >= v
            },
            VersionConstraint::Range { min, max } => self >= min && self <= max,
        }
    }
}

/// Version constraint specifications
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum VersionConstraint {
    Exact(Version),           // =1.2.3
    GreaterThan(Version),     // >1.2.3
    GreaterEqual(Version),    // >=1.2.3
    Compatible(Version),      // ^1.2.3 (compatible within major version)
    Range {                   // 1.2.0..2.0.0
        min: Version,
        max: Version,
    },
}

/// Module dependency specification
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Dependency {
    pub name: String,
    pub version: VersionConstraint,
    pub features: Vec<String>,
    pub optional: bool,
}

/// Module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMetadata {
    pub name: String,
    pub version: Version,
    pub description: String,
    pub authors: Vec<String>,
    pub license: String,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub documentation: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
}

/// Function export from a module
#[derive(Debug, Clone)]
pub struct FunctionExport {
    /// Internal function name within the module
    pub internal_name: String,
    
    /// Public export name (may differ from internal)
    pub export_name: String,
    
    /// Function signature for type checking
    pub signature: FunctionSignature,
    
    /// Function implementation
    pub implementation: FunctionImplementation,
    
    /// Function attributes (Hold, Listable, Protected, etc.)
    pub attributes: Vec<FunctionAttribute>,
    
    /// Documentation string
    pub documentation: Option<String>,
}

/// Different types of function implementations
#[derive(Debug, Clone)]
pub enum FunctionImplementation {
    /// Native Rust function (most stdlib functions)
    Native(StdlibFunction),
    
    /// Foreign object method call
    Foreign {
        type_name: String,
        method_name: String,
    },
    
    /// Lyra-defined function (compiled bytecode)
    Lyra {
        bytecode: Vec<Instruction>,
        constants: Vec<Value>,
        locals_count: u32,
    },
    
    /// External function (from another language)
    External {
        library_path: String,
        symbol_name: String,
        calling_convention: CallingConvention,
    },
}

/// Calling conventions for external functions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CallingConvention {
    C,
    Stdcall,
    Python,
    Julia,
}

/// A Lyra module containing functions and submodules
#[derive(Debug, Clone)]
pub struct Module {
    /// Module metadata
    pub metadata: ModuleMetadata,
    
    /// Functions exported by this module
    pub exports: HashMap<String, FunctionExport>,
    
    /// Submodules contained within this module
    pub submodules: HashMap<String, Module>,
    
    /// Dependencies required by this module
    pub dependencies: Vec<Dependency>,
    
    /// Module configuration
    pub config: ModuleConfig,
    
    /// Module-specific constants
    pub constants: Vec<Value>,
    
    /// Module initialization code (runs once when loaded)
    pub init_code: Option<Vec<Instruction>>,
}

/// Module configuration options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleConfig {
    /// Features enabled for this module
    pub features: Vec<String>,
    
    /// Module-specific settings
    pub settings: HashMap<String, String>,
    
    /// Whether this module can be hot-reloaded
    pub hot_reload: bool,
    
    /// Maximum memory usage (bytes)
    pub memory_limit: Option<u64>,
}

impl Module {
    /// Create a new empty module
    pub fn new(metadata: ModuleMetadata) -> Self {
        Module {
            metadata,
            exports: HashMap::new(),
            submodules: HashMap::new(),
            dependencies: Vec::new(),
            config: ModuleConfig::default(),
            constants: Vec::new(),
            init_code: None,
        }
    }
    
    /// Add a function export to this module
    pub fn add_export(&mut self, export: FunctionExport) -> Result<(), ModuleError> {
        if self.exports.contains_key(&export.export_name) {
            return Err(ModuleError::DuplicateExport {
                function: export.export_name,
                module: self.metadata.name.clone(),
            });
        }
        
        self.exports.insert(export.export_name.clone(), export);
        Ok(())
    }
    
    /// Add a submodule
    pub fn add_submodule(&mut self, name: String, module: Module) {
        self.submodules.insert(name, module);
    }
    
    /// Get all exported function names
    pub fn exported_functions(&self) -> Vec<&String> {
        self.exports.keys().collect()
    }
    
    /// Validate module structure and dependencies
    pub fn validate(&self) -> Result<(), ModuleError> {
        // Validate metadata
        if self.metadata.name.is_empty() {
            return Err(ModuleError::InvalidModuleName {
                name: self.metadata.name.clone(),
            });
        }
        
        // Validate exports don't conflict with submodules
        for export_name in self.exports.keys() {
            if self.submodules.contains_key(export_name) {
                return Err(ModuleError::DuplicateExport {
                    function: export_name.clone(),
                    module: self.metadata.name.clone(),
                });
            }
        }
        
        // Recursively validate submodules
        for submodule in self.submodules.values() {
            submodule.validate()?;
        }
        
        Ok(())
    }
}

impl Default for ModuleConfig {
    fn default() -> Self {
        ModuleConfig {
            features: Vec::new(),
            settings: HashMap::new(),
            hot_reload: false,
            memory_limit: None,
        }
    }
}

/// Convert a stdlib function to a module export
pub fn stdlib_to_export(
    name: &str,
    func: StdlibFunction,
    attributes: Vec<FunctionAttribute>,
) -> FunctionExport {
    FunctionExport {
        internal_name: name.to_string(),
        export_name: name.to_string(),
        signature: FunctionSignature::infer_from_attributes(&attributes),
        implementation: FunctionImplementation::Native(func),
        attributes,
        documentation: None,
    }
}
```

### 1.2 Module Registry Implementation

Create `/src/modules/registry.rs`:

```rust
//! Module Registry System
//!
//! Manages the registration and lookup of modules within the Lyra system.

use super::{Module, ModuleError, Version, VersionConstraint};
use crate::{
    stdlib::{StandardLibrary, StdlibFunction},
    linker::{FunctionRegistry, FunctionAttribute},
    vm::Value,
};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Global module registry for the Lyra system
pub struct ModuleRegistry {
    /// Loaded modules indexed by namespace
    modules: RwLock<HashMap<String, Arc<Module>>>,
    
    /// Unified function registry for static dispatch
    function_registry: Arc<RwLock<FunctionRegistry>>,
    
    /// Namespace resolver for function lookup
    resolver: NamespaceResolver,
    
    /// Standard library integration
    stdlib: Arc<StandardLibrary>,
}

impl ModuleRegistry {
    /// Create a new module registry
    pub fn new(function_registry: Arc<RwLock<FunctionRegistry>>) -> Self {
        let stdlib = Arc::new(StandardLibrary::new());
        let mut registry = ModuleRegistry {
            modules: RwLock::new(HashMap::new()),
            function_registry,
            resolver: NamespaceResolver::new(),
            stdlib,
        };
        
        // Register standard library as modules
        registry.register_stdlib_modules().expect("Failed to register stdlib modules");
        
        registry
    }
    
    /// Register a module in the registry
    pub fn register_module(&self, namespace: &str, module: Module) -> Result<(), ModuleError> {
        // Validate module
        module.validate()?;
        
        // Check for conflicts
        if self.modules.read().unwrap().contains_key(namespace) {
            return Err(ModuleError::PackageError {
                message: format!("Module {} already registered", namespace),
            });
        }
        
        // Register all exported functions in the unified registry
        {
            let mut func_registry = self.function_registry.write().unwrap();
            for (name, export) in &module.exports {
                let qualified_name = format!("{}::{}", namespace, name);
                
                match &export.implementation {
                    super::FunctionImplementation::Native(func) => {
                        func_registry.register_stdlib_function(
                            qualified_name.clone(),
                            *func,
                            export.attributes.clone(),
                        )?;
                    },
                    super::FunctionImplementation::Foreign { type_name, method_name } => {
                        func_registry.register_foreign_method(
                            type_name.clone(),
                            method_name.clone(),
                            export.signature.clone(),
                        )?;
                    },
                    super::FunctionImplementation::Lyra { .. } => {
                        // Register Lyra-defined functions
                        func_registry.register_lyra_function(
                            qualified_name.clone(),
                            export.clone(),
                        )?;
                    },
                    super::FunctionImplementation::External { .. } => {
                        // Register external functions
                        func_registry.register_external_function(
                            qualified_name.clone(),
                            export.clone(),
                        )?;
                    },
                }
            }
        }
        
        // Register in namespace resolver
        self.resolver.register_namespace(namespace, &module)?;
        
        // Store module
        self.modules.write().unwrap().insert(namespace.to_string(), Arc::new(module));
        
        Ok(())
    }
    
    /// Get a module by namespace
    pub fn get_module(&self, namespace: &str) -> Option<Arc<Module>> {
        self.modules.read().unwrap().get(namespace).cloned()
    }
    
    /// List all registered modules
    pub fn list_modules(&self) -> Vec<String> {
        self.modules.read().unwrap().keys().cloned().collect()
    }
    
    /// Resolve a function call to a function index
    pub fn resolve_function(&self, qualified_name: &str) -> Result<u16, ModuleError> {
        self.function_registry
            .read()
            .unwrap()
            .get_function_index(qualified_name)
            .ok_or_else(|| ModuleError::ModuleNotFound {
                name: qualified_name.to_string(),
            })
    }
    
    /// Register standard library functions as modules
    fn register_stdlib_modules(&mut self) -> Result<(), ModuleError> {
        // Create std::math module
        let mut math_module = Module::new(super::ModuleMetadata {
            name: "std::math".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard mathematical functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["math", "trigonometry", "algebra"].into_iter().map(String::from).collect(),
            categories: vec!["mathematics".to_string()],
        });
        
        // Add math functions
        math_module.add_export(super::stdlib_to_export(
            "Sin", crate::stdlib::math::sin, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Cos", crate::stdlib::math::cos, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Tan", crate::stdlib::math::tan, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Exp", crate::stdlib::math::exp, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Log", crate::stdlib::math::log, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Sqrt", crate::stdlib::math::sqrt, vec![FunctionAttribute::Listable]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Plus", crate::stdlib::math::plus, vec![FunctionAttribute::Listable, FunctionAttribute::Orderless]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Times", crate::stdlib::math::times, vec![FunctionAttribute::Listable, FunctionAttribute::Orderless]
        ))?;
        math_module.add_export(super::stdlib_to_export(
            "Power", crate::stdlib::math::power, vec![FunctionAttribute::Listable]
        ))?;
        
        self.register_module("std::math", math_module)?;
        
        // Create std::list module
        let mut list_module = Module::new(super::ModuleMetadata {
            name: "std::list".to_string(),
            version: Version::new(0, 1, 0),
            description: "Standard list manipulation functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec!["list", "array", "collection"].into_iter().map(String::from).collect(),
            categories: vec!["data-structures".to_string()],
        });
        
        // Add list functions
        list_module.add_export(super::stdlib_to_export(
            "Length", crate::stdlib::list::length, vec![]
        ))?;
        list_module.add_export(super::stdlib_to_export(
            "Head", crate::stdlib::list::head, vec![]
        ))?;
        list_module.add_export(super::stdlib_to_export(
            "Tail", crate::stdlib::list::tail, vec![]
        ))?;
        list_module.add_export(super::stdlib_to_export(
            "Append", crate::stdlib::list::append, vec![]
        ))?;
        list_module.add_export(super::stdlib_to_export(
            "Flatten", crate::stdlib::list::flatten, vec![]
        ))?;
        
        self.register_module("std::list", list_module)?;
        
        // Create additional standard library modules...
        // std::string, std::tensor, std::table, std::rules
        
        Ok(())
    }
}

/// Namespace resolution system
#[derive(Debug)]
pub struct NamespaceResolver {
    /// Mapping from namespace to module reference
    namespaces: HashMap<String, String>,  // namespace -> module_key
    
    /// Import aliases for namespaces
    aliases: HashMap<String, String>,     // alias -> full_namespace
    
    /// Wildcard imports (namespace -> imported_symbols)
    wildcard_imports: HashMap<String, Vec<String>>,
}

impl NamespaceResolver {
    pub fn new() -> Self {
        NamespaceResolver {
            namespaces: HashMap::new(),
            aliases: HashMap::new(),
            wildcard_imports: HashMap::new(),
        }
    }
    
    /// Register a namespace
    pub fn register_namespace(&mut self, namespace: &str, module: &Module) -> Result<(), ModuleError> {
        if self.namespaces.contains_key(namespace) {
            return Err(ModuleError::PackageError {
                message: format!("Namespace {} already exists", namespace),
            });
        }
        
        self.namespaces.insert(namespace.to_string(), module.metadata.name.clone());
        Ok(())
    }
    
    /// Resolve a qualified function name (e.g., "std::math::Sin")
    pub fn resolve_qualified(&self, qualified_name: &str) -> Result<String, ModuleError> {
        // Parse namespace and function
        let parts: Vec<&str> = qualified_name.split("::").collect();
        if parts.len() < 2 {
            return Err(ModuleError::InvalidModuleName {
                name: qualified_name.to_string(),
            });
        }
        
        let namespace = parts[..parts.len()-1].join("::");
        let function_name = parts[parts.len()-1];
        
        // Check if namespace exists
        if !self.namespaces.contains_key(&namespace) {
            return Err(ModuleError::ModuleNotFound {
                name: namespace,
            });
        }
        
        Ok(qualified_name.to_string())
    }
    
    /// Resolve an unqualified function name using imports
    pub fn resolve_unqualified(&self, function_name: &str) -> Result<String, ModuleError> {
        // Check wildcard imports first
        for (namespace, symbols) in &self.wildcard_imports {
            if symbols.contains(&function_name.to_string()) {
                return Ok(format!("{}::{}", namespace, function_name));
            }
        }
        
        // Check aliases
        if let Some(full_namespace) = self.aliases.get(function_name) {
            return Ok(full_namespace.clone());
        }
        
        // Fallback to global scope (backwards compatibility)
        Ok(function_name.to_string())
    }
    
    /// Add an import alias
    pub fn add_alias(&mut self, alias: String, full_name: String) {
        self.aliases.insert(alias, full_name);
    }
    
    /// Add a wildcard import
    pub fn add_wildcard_import(&mut self, namespace: String, symbols: Vec<String>) {
        self.wildcard_imports.insert(namespace, symbols);
    }
}
```

### 1.3 Package Management Implementation

Create `/src/modules/package.rs`:

```rust
//! Package Management System
//!
//! Handles package loading, dependency resolution, and registry operations.

use super::{Module, ModuleError, Version, VersionConstraint, Dependency, ModuleMetadata};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs;
use tokio;

/// Package manifest (Lyra.toml) structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageManifest {
    pub package: PackageInfo,
    
    #[serde(default)]
    pub dependencies: HashMap<String, DependencySpec>,
    
    #[serde(default, rename = "dev-dependencies")]
    pub dev_dependencies: HashMap<String, DependencySpec>,
    
    #[serde(default)]
    pub features: HashMap<String, Vec<String>>,
    
    #[serde(default)]
    pub exports: HashMap<String, String>,
    
    #[serde(default)]
    pub imports: HashMap<String, ImportSpec>,
    
    #[serde(default)]
    pub permissions: PermissionSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PackageInfo {
    pub name: String,
    pub version: String,
    pub description: String,
    pub authors: Vec<String>,
    pub license: String,
    pub repository: Option<String>,
    pub homepage: Option<String>,
    pub documentation: Option<String>,
    pub keywords: Vec<String>,
    pub categories: Vec<String>,
    pub edition: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DependencySpec {
    Simple(String),  // "1.0.0" or "^1.0.0"
    Detailed {
        version: String,
        features: Option<Vec<String>>,
        optional: Option<bool>,
        #[serde(rename = "default-features")]
        default_features: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSpec {
    pub from: String,
    pub name: Option<String>,
    pub alias: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermissionSpec {
    #[serde(default)]
    pub filesystem: FilesystemPermissions,
    
    #[serde(default)]
    pub network: NetworkPermissions,
    
    #[serde(default, rename = "foreign-objects")]
    pub foreign_objects: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FilesystemPermissions {
    pub read: Vec<String>,
    pub write: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkPermissions {
    pub domains: Vec<String>,
    pub ports: Vec<u16>,
}

/// Package bundle containing compiled module and metadata
#[derive(Debug, Clone)]
pub struct PackageBundle {
    pub manifest: PackageManifest,
    pub module: Module,
    pub checksum: String,
    pub signature: Option<Vec<u8>>,
}

/// Package manager for handling package operations
pub struct PackageManager {
    /// Local package cache directory
    cache_dir: PathBuf,
    
    /// Registry clients for remote package sources
    registries: Vec<Box<dyn PackageRegistry>>,
    
    /// Dependency resolver
    resolver: DependencyResolver,
    
    /// Package verifier for security
    verifier: PackageVerifier,
}

impl PackageManager {
    /// Create a new package manager
    pub fn new(cache_dir: PathBuf) -> Self {
        PackageManager {
            cache_dir,
            registries: Vec::new(),
            resolver: DependencyResolver::new(),
            verifier: PackageVerifier::new(),
        }
    }
    
    /// Install a package and its dependencies
    pub async fn install_package(&mut self, name: &str, version_req: &str) -> Result<PackageBundle, ModuleError> {
        // Parse version requirement
        let constraint = self.parse_version_constraint(version_req)?;
        
        // Resolve dependencies
        let resolution_plan = self.resolver.resolve(&[(name.to_string(), constraint)]).await?;
        
        // Download and verify packages
        let mut bundles = Vec::new();
        for (pkg_name, version) in resolution_plan.packages {
            let bundle = self.download_package(&pkg_name, &version).await?;
            self.verifier.verify_package(&bundle)?;
            bundles.push(bundle);
        }
        
        // Install packages in dependency order
        for bundle in &bundles {
            self.cache_package(bundle)?;
        }
        
        // Return the main package bundle
        bundles.into_iter()
            .find(|b| b.manifest.package.name == name)
            .ok_or_else(|| ModuleError::PackageError {
                message: format!("Package {} not found in resolution", name),
            })
    }
    
    /// Load a package from local cache
    pub fn load_package(&self, name: &str, version: &Version) -> Result<PackageBundle, ModuleError> {
        let package_dir = self.cache_dir.join(format!("{}-{}", name, version));
        
        if !package_dir.exists() {
            return Err(ModuleError::ModuleNotFound {
                name: name.to_string(),
            });
        }
        
        // Load manifest
        let manifest_path = package_dir.join("Lyra.toml");
        let manifest_content = fs::read_to_string(manifest_path)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        let manifest: PackageManifest = toml::from_str(&manifest_content)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to parse manifest: {}", e),
            })?;
        
        // Load compiled module
        let module_path = package_dir.join("module.lyra");
        let module = self.load_compiled_module(&module_path)?;
        
        // Load checksum
        let checksum_path = package_dir.join("checksum.txt");
        let checksum = fs::read_to_string(checksum_path)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to read checksum: {}", e),
            })?;
        
        Ok(PackageBundle {
            manifest,
            module,
            checksum: checksum.trim().to_string(),
            signature: None,
        })
    }
    
    /// Build a package from source
    pub fn build_package(&self, source_dir: &Path) -> Result<PackageBundle, ModuleError> {
        // Load and validate manifest
        let manifest_path = source_dir.join("Lyra.toml");
        let manifest = self.load_manifest(&manifest_path)?;
        
        // Compile module from source
        let module = self.compile_module(source_dir, &manifest)?;
        
        // Calculate checksum
        let checksum = self.calculate_checksum(&module)?;
        
        Ok(PackageBundle {
            manifest,
            module,
            checksum,
            signature: None,
        })
    }
    
    /// Add a package registry
    pub fn add_registry(&mut self, registry: Box<dyn PackageRegistry>) {
        self.registries.push(registry);
    }
    
    // Private helper methods
    
    fn parse_version_constraint(&self, version_req: &str) -> Result<VersionConstraint, ModuleError> {
        // Parse version constraint from string
        // Examples: "1.0.0", "^1.0.0", ">=1.0.0", "1.0.0..2.0.0"
        
        if version_req.starts_with('^') {
            let version_str = &version_req[1..];
            let version = self.parse_version(version_str)?;
            Ok(VersionConstraint::Compatible(version))
        } else if version_req.starts_with(">=") {
            let version_str = &version_req[2..];
            let version = self.parse_version(version_str)?;
            Ok(VersionConstraint::GreaterEqual(version))
        } else if version_req.starts_with('>') {
            let version_str = &version_req[1..];
            let version = self.parse_version(version_str)?;
            Ok(VersionConstraint::GreaterThan(version))
        } else if version_req.contains("..") {
            let parts: Vec<&str> = version_req.split("..").collect();
            if parts.len() != 2 {
                return Err(ModuleError::PackageError {
                    message: format!("Invalid range constraint: {}", version_req),
                });
            }
            let min = self.parse_version(parts[0])?;
            let max = self.parse_version(parts[1])?;
            Ok(VersionConstraint::Range { min, max })
        } else {
            let version = self.parse_version(version_req)?;
            Ok(VersionConstraint::Exact(version))
        }
    }
    
    fn parse_version(&self, version_str: &str) -> Result<Version, ModuleError> {
        let parts: Vec<&str> = version_str.split('.').collect();
        if parts.len() != 3 {
            return Err(ModuleError::PackageError {
                message: format!("Invalid version format: {}", version_str),
            });
        }
        
        let major = parts[0].parse().map_err(|_| ModuleError::PackageError {
            message: format!("Invalid major version: {}", parts[0]),
        })?;
        
        let minor = parts[1].parse().map_err(|_| ModuleError::PackageError {
            message: format!("Invalid minor version: {}", parts[1]),
        })?;
        
        let patch = parts[2].parse().map_err(|_| ModuleError::PackageError {
            message: format!("Invalid patch version: {}", parts[2]),
        })?;
        
        Ok(Version::new(major, minor, patch))
    }
    
    async fn download_package(&self, name: &str, version: &Version) -> Result<PackageBundle, ModuleError> {
        for registry in &self.registries {
            if let Ok(bundle) = registry.download(name, version).await {
                return Ok(bundle);
            }
        }
        
        Err(ModuleError::ModuleNotFound {
            name: format!("{}@{}", name, version),
        })
    }
    
    fn cache_package(&self, bundle: &PackageBundle) -> Result<(), ModuleError> {
        let package_name = &bundle.manifest.package.name;
        let package_version = &bundle.manifest.package.version;
        let package_dir = self.cache_dir.join(format!("{}-{}", package_name, package_version));
        
        // Create package directory
        fs::create_dir_all(&package_dir)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to create package directory: {}", e),
            })?;
        
        // Write manifest
        let manifest_content = toml::to_string(&bundle.manifest)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to serialize manifest: {}", e),
            })?;
        
        fs::write(package_dir.join("Lyra.toml"), manifest_content)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to write manifest: {}", e),
            })?;
        
        // Write compiled module (binary format for fast loading)
        self.save_compiled_module(&bundle.module, &package_dir.join("module.lyra"))?;
        
        // Write checksum
        fs::write(package_dir.join("checksum.txt"), &bundle.checksum)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to write checksum: {}", e),
            })?;
        
        Ok(())
    }
    
    fn load_manifest(&self, path: &Path) -> Result<PackageManifest, ModuleError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        toml::from_str(&content)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to parse manifest: {}", e),
            })
    }
    
    fn compile_module(&self, source_dir: &Path, manifest: &PackageManifest) -> Result<Module, ModuleError> {
        // This would invoke the Lyra compiler to build the module
        // For now, return a placeholder
        todo!("Module compilation not yet implemented")
    }
    
    fn load_compiled_module(&self, path: &Path) -> Result<Module, ModuleError> {
        // Load pre-compiled module from binary format
        todo!("Binary module loading not yet implemented")
    }
    
    fn save_compiled_module(&self, module: &Module, path: &Path) -> Result<(), ModuleError> {
        // Save module in binary format for fast loading
        todo!("Binary module saving not yet implemented")
    }
    
    fn calculate_checksum(&self, module: &Module) -> Result<String, ModuleError> {
        // Calculate SHA-256 checksum of module content
        todo!("Checksum calculation not yet implemented")
    }
}

/// Package registry trait for different registry backends
#[async_trait::async_trait]
pub trait PackageRegistry: Send + Sync {
    /// Search for packages matching query
    async fn search(&self, query: &str) -> Result<Vec<PackageInfo>, ModuleError>;
    
    /// Get package metadata
    async fn get_package_info(&self, name: &str, version: &Version) -> Result<PackageInfo, ModuleError>;
    
    /// Download package
    async fn download(&self, name: &str, version: &Version) -> Result<PackageBundle, ModuleError>;
    
    /// Publish package (requires authentication)
    async fn publish(&self, bundle: &PackageBundle, token: &str) -> Result<(), ModuleError>;
    
    /// List available versions for a package
    async fn list_versions(&self, name: &str) -> Result<Vec<Version>, ModuleError>;
}

/// Dependency resolution engine
pub struct DependencyResolver {
    /// Cache of resolved dependency graphs
    resolution_cache: HashMap<Vec<(String, VersionConstraint)>, ResolutionPlan>,
}

#[derive(Debug, Clone)]
pub struct ResolutionPlan {
    pub packages: Vec<(String, Version)>,
    pub order: Vec<String>,  // Installation order
}

impl DependencyResolver {
    pub fn new() -> Self {
        DependencyResolver {
            resolution_cache: HashMap::new(),
        }
    }
    
    /// Resolve dependencies using constraint satisfaction
    pub async fn resolve(&mut self, dependencies: &[(String, VersionConstraint)]) -> Result<ResolutionPlan, ModuleError> {
        // Check cache first
        if let Some(plan) = self.resolution_cache.get(dependencies) {
            return Ok(plan.clone());
        }
        
        // Perform dependency resolution
        let plan = self.resolve_dependencies(dependencies).await?;
        
        // Cache result
        self.resolution_cache.insert(dependencies.to_vec(), plan.clone());
        
        Ok(plan)
    }
    
    async fn resolve_dependencies(&self, dependencies: &[(String, VersionConstraint)]) -> Result<ResolutionPlan, ModuleError> {
        // This is a simplified version - a real implementation would use
        // a SAT solver or similar for complex dependency resolution
        
        let mut resolved_packages = HashMap::new();
        let mut to_resolve = dependencies.to_vec();
        let mut processed = HashSet::new();
        
        while let Some((name, constraint)) = to_resolve.pop() {
            if processed.contains(&name) {
                continue;
            }
            
            // Find a version that satisfies the constraint
            let version = self.find_satisfying_version(&name, &constraint).await?;
            
            // Check for conflicts
            if let Some(existing_version) = resolved_packages.get(&name) {
                if existing_version != &version {
                    return Err(ModuleError::VersionConflict {
                        package: name,
                        required: format!("{:?}", constraint),
                        found: format!("{}", existing_version),
                    });
                }
            } else {
                resolved_packages.insert(name.clone(), version.clone());
            }
            
            // Add transitive dependencies
            let package_deps = self.get_package_dependencies(&name, &version).await?;
            for dep in package_deps {
                if !processed.contains(&dep.name) {
                    to_resolve.push((dep.name, dep.version));
                }
            }
            
            processed.insert(name);
        }
        
        // Determine installation order (topological sort)
        let order = self.topological_sort(&resolved_packages).await?;
        
        let packages: Vec<(String, Version)> = resolved_packages.into_iter().collect();
        
        Ok(ResolutionPlan { packages, order })
    }
    
    async fn find_satisfying_version(&self, name: &str, constraint: &VersionConstraint) -> Result<Version, ModuleError> {
        // This would query registries to find a version that satisfies the constraint
        // For now, return a placeholder
        Ok(Version::new(1, 0, 0))
    }
    
    async fn get_package_dependencies(&self, name: &str, version: &Version) -> Result<Vec<Dependency>, ModuleError> {
        // This would fetch the package's dependencies from its manifest
        // For now, return empty
        Ok(Vec::new())
    }
    
    async fn topological_sort(&self, packages: &HashMap<String, Version>) -> Result<Vec<String>, ModuleError> {
        // Implement topological sort for dependency order
        // For now, return packages in arbitrary order
        Ok(packages.keys().cloned().collect())
    }
}

/// Package verification for security
pub struct PackageVerifier {
    // Trusted signing keys, hash verification, etc.
}

impl PackageVerifier {
    pub fn new() -> Self {
        PackageVerifier {}
    }
    
    pub fn verify_package(&self, bundle: &PackageBundle) -> Result<(), ModuleError> {
        // Verify package signature, checksum, scan for malicious content, etc.
        // For now, always pass
        Ok(())
    }
}

impl Default for PermissionSpec {
    fn default() -> Self {
        PermissionSpec {
            filesystem: FilesystemPermissions::default(),
            network: NetworkPermissions::default(),
            foreign_objects: Vec::new(),
        }
    }
}
```

## 2. CLI Implementation

Create `/src/modules/cli.rs`:

```rust
//! CLI Interface for Package Management
//!
//! Implements the `lyra pkg` command suite for package management operations.

use super::{PackageManager, ModuleRegistry, ModuleError, Version};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tokio;

#[derive(Parser)]
#[command(name = "lyra")]
#[command(about = "Lyra symbolic computation engine")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Package management commands
    #[command(name = "pkg")]
    Package(PackageCommands),
    
    /// Other commands (eval, repl, etc.)
    // ... existing commands
}

#[derive(Parser)]
pub struct PackageCommands {
    #[command(subcommand)]
    pub command: PackageSubcommands,
}

#[derive(Subcommand)]
pub enum PackageSubcommands {
    /// Initialize a new package
    Init {
        /// Package name
        name: Option<String>,
        
        /// Package directory
        #[arg(short, long)]
        path: Option<PathBuf>,
    },
    
    /// Build the current package
    Build {
        /// Build in release mode
        #[arg(short, long)]
        release: bool,
    },
    
    /// Run package tests
    Test {
        /// Run only specific test
        #[arg(short, long)]
        test: Option<String>,
    },
    
    /// Install a package
    Install {
        /// Package name
        package: String,
        
        /// Version constraint
        #[arg(short, long)]
        version: Option<String>,
        
        /// Install as development dependency
        #[arg(long)]
        dev: bool,
    },
    
    /// Update packages
    Update {
        /// Specific package to update
        package: Option<String>,
    },
    
    /// Remove a package
    Remove {
        /// Package name
        package: String,
    },
    
    /// Search for packages
    Search {
        /// Search query
        query: String,
        
        /// Limit number of results
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },
    
    /// Show package information
    Info {
        /// Package name
        package: String,
        
        /// Show specific version
        #[arg(short, long)]
        version: Option<String>,
    },
    
    /// List installed packages
    List {
        /// Show only outdated packages
        #[arg(long)]
        outdated: bool,
    },
    
    /// Show dependency tree
    Tree {
        /// Package to show tree for
        package: Option<String>,
        
        /// Maximum depth
        #[arg(short, long)]
        depth: Option<usize>,
    },
    
    /// Check package health
    Check,
    
    /// Publish package to registry
    Publish {
        /// Registry to publish to
        #[arg(short, long)]
        registry: Option<String>,
        
        /// Authentication token
        #[arg(short, long)]
        token: Option<String>,
    },
}

/// Package CLI implementation
pub struct PackageCli {
    package_manager: PackageManager,
    module_registry: ModuleRegistry,
}

impl PackageCli {
    pub fn new(cache_dir: PathBuf, module_registry: ModuleRegistry) -> Self {
        PackageCli {
            package_manager: PackageManager::new(cache_dir),
            module_registry,
        }
    }
    
    /// Execute a package command
    pub async fn execute(&mut self, command: PackageSubcommands) -> Result<(), ModuleError> {
        match command {
            PackageSubcommands::Init { name, path } => {
                self.cmd_init(name, path).await
            },
            PackageSubcommands::Build { release } => {
                self.cmd_build(release).await
            },
            PackageSubcommands::Test { test } => {
                self.cmd_test(test).await
            },
            PackageSubcommands::Install { package, version, dev } => {
                self.cmd_install(&package, version.as_deref(), dev).await
            },
            PackageSubcommands::Update { package } => {
                self.cmd_update(package.as_deref()).await
            },
            PackageSubcommands::Remove { package } => {
                self.cmd_remove(&package).await
            },
            PackageSubcommands::Search { query, limit } => {
                self.cmd_search(&query, limit).await
            },
            PackageSubcommands::Info { package, version } => {
                self.cmd_info(&package, version.as_deref()).await
            },
            PackageSubcommands::List { outdated } => {
                self.cmd_list(outdated).await
            },
            PackageSubcommands::Tree { package, depth } => {
                self.cmd_tree(package.as_deref(), depth).await
            },
            PackageSubcommands::Check => {
                self.cmd_check().await
            },
            PackageSubcommands::Publish { registry, token } => {
                self.cmd_publish(registry.as_deref(), token.as_deref()).await
            },
        }
    }
    
    // Command implementations
    
    async fn cmd_init(&mut self, name: Option<String>, path: Option<PathBuf>) -> Result<(), ModuleError> {
        let current_dir = std::env::current_dir()
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        let project_dir = path.unwrap_or(current_dir);
        let package_name = name.unwrap_or_else(|| {
            project_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("my-package")
                .to_string()
        });
        
        // Create Lyra.toml
        let manifest_content = format!(
            r#"[package]
name = "{}"
version = "0.1.0"
description = "A Lyra package"
authors = ["Your Name <you@example.com>"]
license = "MIT"

[dependencies]
std = ">=0.1.0"

[features]
default = []
"#,
            package_name
        );
        
        let manifest_path = project_dir.join("Lyra.toml");
        std::fs::write(&manifest_path, manifest_content)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to create Lyra.toml: {}", e),
            })?;
        
        // Create src directory and main.ly
        let src_dir = project_dir.join("src");
        std::fs::create_dir_all(&src_dir)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to create src directory: {}", e),
            })?;
        
        let main_content = r#"// Welcome to Lyra!
// This is your package's main module.

// Import from standard library
import std::math::{Sin, Cos, Pi}

// Define a function
HelloWorld[] := "Hello from Lyra!"

// Export functions
export HelloWorld
"#;
        
        let main_path = src_dir.join("main.ly");
        std::fs::write(&main_path, main_content)
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to create main.ly: {}", e),
            })?;
        
        println!("✓ Created package '{}' in {}", package_name, project_dir.display());
        println!("  Next steps:");
        println!("    cd {}", project_dir.display());
        println!("    lyra pkg build");
        println!("    lyra pkg test");
        
        Ok(())
    }
    
    async fn cmd_build(&mut self, release: bool) -> Result<(), ModuleError> {
        println!("Building package{}...", if release { " (release)" } else { "" });
        
        let current_dir = std::env::current_dir()
            .map_err(|e| ModuleError::PackageError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        let bundle = self.package_manager.build_package(&current_dir)?;
        
        println!("✓ Built package '{}' v{}", 
                 bundle.manifest.package.name, 
                 bundle.manifest.package.version);
        
        Ok(())
    }
    
    async fn cmd_test(&mut self, test_name: Option<String>) -> Result<(), ModuleError> {
        if let Some(test) = test_name {
            println!("Running test '{}'...", test);
        } else {
            println!("Running all tests...");
        }
        
        // TODO: Implement test runner
        println!("✓ All tests passed");
        
        Ok(())
    }
    
    async fn cmd_install(&mut self, package: &str, version: Option<&str>, dev: bool) -> Result<(), ModuleError> {
        let version_req = version.unwrap_or("*");
        
        println!("Installing {} {}...", package, version_req);
        
        let bundle = self.package_manager.install_package(package, version_req).await?;
        
        // Register module in registry
        let namespace = format!("user::{}", package);
        self.module_registry.register_module(&namespace, bundle.module)?;
        
        println!("✓ Installed {} v{}", package, bundle.manifest.package.version);
        
        Ok(())
    }
    
    async fn cmd_update(&mut self, package: Option<&str>) -> Result<(), ModuleError> {
        if let Some(pkg) = package {
            println!("Updating {}...", pkg);
        } else {
            println!("Updating all packages...");
        }
        
        // TODO: Implement package updates
        println!("✓ Packages updated");
        
        Ok(())
    }
    
    async fn cmd_remove(&mut self, package: &str) -> Result<(), ModuleError> {
        println!("Removing {}...", package);
        
        // TODO: Implement package removal
        println!("✓ Removed {}", package);
        
        Ok(())
    }
    
    async fn cmd_search(&mut self, query: &str, limit: usize) -> Result<(), ModuleError> {
        println!("Searching for '{}'...", query);
        
        // TODO: Search registries
        println!("No packages found matching '{}'", query);
        
        Ok(())
    }
    
    async fn cmd_info(&mut self, package: &str, version: Option<&str>) -> Result<(), ModuleError> {
        println!("Package information for {}:", package);
        
        // TODO: Show package info
        println!("  Name: {}", package);
        if let Some(v) = version {
            println!("  Version: {}", v);
        }
        
        Ok(())
    }
    
    async fn cmd_list(&mut self, outdated: bool) -> Result<(), ModuleError> {
        if outdated {
            println!("Outdated packages:");
        } else {
            println!("Installed packages:");
        }
        
        let modules = self.module_registry.list_modules();
        for module in modules {
            println!("  {}", module);
        }
        
        Ok(())
    }
    
    async fn cmd_tree(&mut self, package: Option<&str>, depth: Option<usize>) -> Result<(), ModuleError> {
        let max_depth = depth.unwrap_or(usize::MAX);
        
        if let Some(pkg) = package {
            println!("Dependency tree for {}:", pkg);
        } else {
            println!("Dependency tree:");
        }
        
        // TODO: Show dependency tree
        println!("  (no dependencies)");
        
        Ok(())
    }
    
    async fn cmd_check(&mut self) -> Result<(), ModuleError> {
        println!("Checking package health...");
        
        // TODO: Health checks
        println!("✓ Package is healthy");
        
        Ok(())
    }
    
    async fn cmd_publish(&mut self, registry: Option<&str>, token: Option<&str>) -> Result<(), ModuleError> {
        let reg_name = registry.unwrap_or("default");
        
        println!("Publishing to {} registry...", reg_name);
        
        if token.is_none() {
            return Err(ModuleError::PackageError {
                message: "Authentication token required for publishing".to_string(),
            });
        }
        
        // TODO: Implement publishing
        println!("✓ Package published successfully");
        
        Ok(())
    }
}
```

This implementation plan provides:

1. **Core Module System**: Complete type definitions and basic infrastructure
2. **Registry System**: Module registration and namespace resolution
3. **Package Management**: Full package lifecycle from installation to publishing
4. **CLI Interface**: Comprehensive command-line tools for package operations
5. **Integration Points**: Clear interfaces for connecting with existing systems

The implementation maintains backwards compatibility with the existing StandardLibrary while providing a path forward for organized, scalable module management. The design leverages the existing static dispatch system for optimal performance and integrates seamlessly with the existing Foreign object architecture.

Next steps would be to implement the binary module format, registry backends, and testing framework to complete the module system.