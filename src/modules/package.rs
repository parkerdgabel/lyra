//! Package Management System
//!
//! Handles package loading, dependency resolution, and registry operations.

use super::{Module, ModuleError, Version, VersionConstraint, Dependency, ModuleMetadata};
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::fs;

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
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        let manifest: PackageManifest = toml::from_str(&manifest_content)
            .map_err(|e| ModuleError::ParseError {
                message: format!("Failed to parse manifest: {}", e),
            })?;
        
        // Load compiled module
        let module_path = package_dir.join("module.lyra");
        let module = self.load_compiled_module(&module_path)?;
        
        // Load checksum
        let checksum_path = package_dir.join("checksum.txt");
        let checksum = fs::read_to_string(checksum_path)
            .map_err(|e| ModuleError::IoError {
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
    
    /// List installed packages
    pub fn list_installed(&self) -> Result<Vec<PackageInfo>, ModuleError> {
        let mut packages = Vec::new();
        
        if !self.cache_dir.exists() {
            return Ok(packages);
        }
        
        for entry in fs::read_dir(&self.cache_dir)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })? {
            let entry = entry.map_err(|e| ModuleError::IoError { message: e.to_string() })?;
            let path = entry.path();
            
            if path.is_dir() {
                let manifest_path = path.join("Lyra.toml");
                if manifest_path.exists() {
                    if let Ok(manifest) = self.load_manifest(&manifest_path) {
                        packages.push(manifest.package);
                    }
                }
            }
        }
        
        Ok(packages)
    }
    
    /// Remove an installed package
    pub fn remove_package(&self, name: &str, version: &Version) -> Result<(), ModuleError> {
        let package_dir = self.cache_dir.join(format!("{}-{}", name, version));
        
        if package_dir.exists() {
            fs::remove_dir_all(package_dir)
                .map_err(|e| ModuleError::IoError {
                    message: format!("Failed to remove package: {}", e),
                })?;
        }
        
        Ok(())
    }
    
    // Private helper methods
    
    fn parse_version_constraint(&self, version_req: &str) -> Result<VersionConstraint, ModuleError> {
        // Parse version constraint from string
        // Examples: "1.0.0", "^1.0.0", ">=1.0.0", "1.0.0..2.0.0"
        
        if version_req == "*" {
            return Ok(VersionConstraint::GreaterEqual(Version::new(0, 0, 0)));
        }
        
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
                return Err(ModuleError::ParseError {
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
            return Err(ModuleError::ParseError {
                message: format!("Invalid version format: {}", version_str),
            });
        }
        
        let major = parts[0].parse().map_err(|_| ModuleError::ParseError {
            message: format!("Invalid major version: {}", parts[0]),
        })?;
        
        let minor = parts[1].parse().map_err(|_| ModuleError::ParseError {
            message: format!("Invalid minor version: {}", parts[1]),
        })?;
        
        let patch = parts[2].parse().map_err(|_| ModuleError::ParseError {
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
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create package directory: {}", e),
            })?;
        
        // Write manifest
        let manifest_content = toml::to_string(&bundle.manifest)
            .map_err(|e| ModuleError::ParseError {
                message: format!("Failed to serialize manifest: {}", e),
            })?;
        
        fs::write(package_dir.join("Lyra.toml"), manifest_content)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to write manifest: {}", e),
            })?;
        
        // Write compiled module (binary format for fast loading)
        self.save_compiled_module(&bundle.module, &package_dir.join("module.lyra"))?;
        
        // Write checksum
        fs::write(package_dir.join("checksum.txt"), &bundle.checksum)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to write checksum: {}", e),
            })?;
        
        Ok(())
    }
    
    pub fn load_manifest(&self, path: &Path) -> Result<PackageManifest, ModuleError> {
        let content = fs::read_to_string(path)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        toml::from_str(&content)
            .map_err(|e| ModuleError::ParseError {
                message: format!("Failed to parse manifest: {}", e),
            })
    }
    
    fn compile_module(&self, source_dir: &Path, manifest: &PackageManifest) -> Result<Module, ModuleError> {
        // Create a basic module from the manifest for now
        // In a full implementation, this would compile Lyra source code
        let metadata = ModuleMetadata {
            name: manifest.package.name.clone(),
            version: self.parse_version(&manifest.package.version)?,
            description: manifest.package.description.clone(),
            authors: manifest.package.authors.clone(),
            license: manifest.package.license.clone(),
            repository: manifest.package.repository.clone(),
            homepage: manifest.package.homepage.clone(),
            documentation: manifest.package.documentation.clone(),
            keywords: manifest.package.keywords.clone(),
            categories: manifest.package.categories.clone(),
        };
        
        Ok(Module::new(metadata))
    }
    
    fn load_compiled_module(&self, path: &Path) -> Result<Module, ModuleError> {
        // For now, create an empty module
        // In a full implementation, this would deserialize a binary module format
        let metadata = ModuleMetadata {
            name: "loaded_module".to_string(),
            version: Version::new(1, 0, 0),
            description: "Loaded module".to_string(),
            authors: vec![],
            license: "Unknown".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        };
        
        Ok(Module::new(metadata))
    }
    
    fn save_compiled_module(&self, module: &Module, path: &Path) -> Result<(), ModuleError> {
        // For now, just create an empty file
        // In a full implementation, this would serialize the module to a binary format
        fs::write(path, b"")
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to save module: {}", e),
            })?;
        
        Ok(())
    }
    
    fn calculate_checksum(&self, module: &Module) -> Result<String, ModuleError> {
        // For now, return a simple checksum
        // In a full implementation, this would calculate SHA-256 of module content
        Ok("sha256:0000000000000000000000000000000000000000000000000000000000000000".to_string())
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

/// Local file system registry implementation
pub struct LocalRegistry {
    root_path: PathBuf,
}

impl LocalRegistry {
    pub fn new(root_path: PathBuf) -> Self {
        LocalRegistry { root_path }
    }
}

#[async_trait::async_trait]
impl PackageRegistry for LocalRegistry {
    async fn search(&self, query: &str) -> Result<Vec<PackageInfo>, ModuleError> {
        let mut results = Vec::new();
        
        if !self.root_path.exists() {
            return Ok(results);
        }
        
        for entry in fs::read_dir(&self.root_path)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })? {
            let entry = entry.map_err(|e| ModuleError::IoError { message: e.to_string() })?;
            let path = entry.path();
            
            if path.is_dir() {
                let manifest_path = path.join("Lyra.toml");
                if manifest_path.exists() {
                    let content = fs::read_to_string(manifest_path)
                        .map_err(|e| ModuleError::IoError { message: e.to_string() })?;
                    
                    if let Ok(manifest) = toml::from_str::<PackageManifest>(&content) {
                        if manifest.package.name.contains(query) || 
                           manifest.package.description.contains(query) ||
                           manifest.package.keywords.iter().any(|k| k.contains(query)) {
                            results.push(manifest.package);
                        }
                    }
                }
            }
        }
        
        Ok(results)
    }
    
    async fn get_package_info(&self, name: &str, version: &Version) -> Result<PackageInfo, ModuleError> {
        let package_dir = self.root_path.join(format!("{}-{}", name, version));
        let manifest_path = package_dir.join("Lyra.toml");
        
        if !manifest_path.exists() {
            return Err(ModuleError::ModuleNotFound { name: name.to_string() });
        }
        
        let content = fs::read_to_string(manifest_path)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })?;
        
        let manifest: PackageManifest = toml::from_str(&content)
            .map_err(|e| ModuleError::ParseError { message: e.to_string() })?;
        
        Ok(manifest.package)
    }
    
    async fn download(&self, name: &str, version: &Version) -> Result<PackageBundle, ModuleError> {
        Err(ModuleError::PackageError {
            message: "Local registry does not support downloads".to_string(),
        })
    }
    
    async fn publish(&self, bundle: &PackageBundle, _token: &str) -> Result<(), ModuleError> {
        let package_name = &bundle.manifest.package.name;
        let package_version = &bundle.manifest.package.version;
        let package_dir = self.root_path.join(format!("{}-{}", package_name, package_version));
        
        fs::create_dir_all(&package_dir)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })?;
        
        let manifest_content = toml::to_string(&bundle.manifest)
            .map_err(|e| ModuleError::ParseError { message: e.to_string() })?;
        
        fs::write(package_dir.join("Lyra.toml"), manifest_content)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })?;
        
        Ok(())
    }
    
    async fn list_versions(&self, name: &str) -> Result<Vec<Version>, ModuleError> {
        let mut versions = Vec::new();
        
        if !self.root_path.exists() {
            return Ok(versions);
        }
        
        for entry in fs::read_dir(&self.root_path)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })? {
            let entry = entry.map_err(|e| ModuleError::IoError { message: e.to_string() })?;
            let dir_name = entry.file_name().to_string_lossy().to_string();
            
            if dir_name.starts_with(&format!("{}-", name)) {
                if let Some(version_str) = dir_name.strip_prefix(&format!("{}-", name)) {
                    if let Ok(version) = version_str.parse::<String>().and_then(|v| {
                        let parts: Vec<&str> = v.split('.').collect();
                        if parts.len() == 3 {
                            Ok(Version::new(
                                parts[0].parse().map_err(|_| "parse error")?,
                                parts[1].parse().map_err(|_| "parse error")?,
                                parts[2].parse().map_err(|_| "parse error")?,
                            ))
                        } else {
                            Err("invalid format")
                        }
                    }) {
                        versions.push(version);
                    }
                }
            }
        }
        
        versions.sort();
        Ok(versions)
    }
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
        // For now, return a default version that satisfies the constraint
        match constraint {
            VersionConstraint::Exact(v) => Ok(v.clone()),
            VersionConstraint::GreaterThan(v) => Ok(Version::new(v.major, v.minor, v.patch + 1)),
            VersionConstraint::GreaterEqual(v) => Ok(v.clone()),
            VersionConstraint::Compatible(v) => Ok(v.clone()),
            VersionConstraint::Range { min, .. } => Ok(min.clone()),
        }
    }
    
    async fn get_package_dependencies(&self, name: &str, version: &Version) -> Result<Vec<Dependency>, ModuleError> {
        // For now, return empty dependencies
        // In a real implementation, this would fetch the package's dependencies from its manifest
        Ok(Vec::new())
    }
    
    async fn topological_sort(&self, packages: &HashMap<String, Version>) -> Result<Vec<String>, ModuleError> {
        // For now, return packages in arbitrary order
        // In a real implementation, this would perform topological sort based on dependencies
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
        // For now, always pass verification
        // In a real implementation, this would:
        // - Verify package signature
        // - Check hash integrity
        // - Scan for malicious content
        // - Validate permissions
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_version_parsing() {
        let manager = PackageManager::new(PathBuf::new());
        
        assert!(manager.parse_version("1.2.3").is_ok());
        assert!(manager.parse_version("invalid").is_err());
        
        let version = manager.parse_version("1.2.3").unwrap();
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
    }

    #[test]
    fn test_version_constraint_parsing() {
        let manager = PackageManager::new(PathBuf::new());
        
        let constraint = manager.parse_version_constraint("^1.2.3").unwrap();
        assert!(matches!(constraint, VersionConstraint::Compatible(_)));
        
        let constraint = manager.parse_version_constraint(">=1.2.3").unwrap();
        assert!(matches!(constraint, VersionConstraint::GreaterEqual(_)));
        
        let constraint = manager.parse_version_constraint("1.2.3").unwrap();
        assert!(matches!(constraint, VersionConstraint::Exact(_)));
    }

    #[test]
    fn test_package_manifest_serialization() {
        let manifest = PackageManifest {
            package: PackageInfo {
                name: "test-package".to_string(),
                version: "1.0.0".to_string(),
                description: "A test package".to_string(),
                authors: vec!["Test Author".to_string()],
                license: "MIT".to_string(),
                repository: None,
                homepage: None,
                documentation: None,
                keywords: vec!["test".to_string()],
                categories: vec!["testing".to_string()],
                edition: None,
            },
            dependencies: HashMap::new(),
            dev_dependencies: HashMap::new(),
            features: HashMap::new(),
            exports: HashMap::new(),
            imports: HashMap::new(),
            permissions: PermissionSpec::default(),
        };
        
        let toml_str = toml::to_string(&manifest).unwrap();
        let parsed: PackageManifest = toml::from_str(&toml_str).unwrap();
        
        assert_eq!(parsed.package.name, "test-package");
        assert_eq!(parsed.package.version, "1.0.0");
    }

    #[test]
    fn test_local_registry() {
        let temp_dir = TempDir::new().unwrap();
        let registry = LocalRegistry::new(temp_dir.path().to_path_buf());
        
        // Test empty registry
        let rt = tokio::runtime::Runtime::new().unwrap();
        let result = rt.block_on(registry.search("test"));
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dependency_resolver() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut resolver = DependencyResolver::new();
        
        let deps = vec![
            ("test-package".to_string(), VersionConstraint::Exact(Version::new(1, 0, 0))),
        ];
        
        let result = rt.block_on(resolver.resolve(&deps));
        assert!(result.is_ok());
        
        let plan = result.unwrap();
        assert_eq!(plan.packages.len(), 1);
        assert_eq!(plan.packages[0].0, "test-package");
    }
}