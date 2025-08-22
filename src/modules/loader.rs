//! Module Loading System
//!
//! Handles loading and initialization of modules from various sources.

use super::{Module, ModuleError, ModuleMetadata, Version};
use super::package::PackageManager;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Module loading sources
#[derive(Debug, Clone)]
pub enum ModuleSource {
    /// Load from local file system
    LocalPath(PathBuf),
    
    /// Load from package registry
    Registry {
        name: String,
        version: Option<Version>,
    },
    
    /// Load from memory (for built-in modules)
    Memory(Module),
    
    /// Load from URL (future extension)
    Url(String),
}

/// Module loader configuration
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Maximum module cache size
    pub max_cache_size: usize,
    
    /// Enable hot reloading for development
    pub hot_reload: bool,
    
    /// Trusted module sources
    pub trusted_sources: Vec<String>,
    
    /// Module search paths
    pub search_paths: Vec<PathBuf>,
    
    /// Enable module verification
    pub verify_modules: bool,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        LoaderConfig {
            max_cache_size: 100,
            hot_reload: false,
            trusted_sources: vec!["std".to_string()],
            search_paths: vec![PathBuf::from("./modules"), PathBuf::from("~/.lyra/modules")],
            verify_modules: true,
        }
    }
}

/// Module loader with caching and dependency resolution
pub struct ModuleLoader {
    /// Configuration
    config: LoaderConfig,
    
    /// Module cache (namespace -> module)
    cache: Arc<RwLock<HashMap<String, Arc<Module>>>>,
    
    /// Package manager for registry loading
    package_manager: Arc<RwLock<PackageManager>>,
    
    /// Module metadata cache
    metadata_cache: Arc<RwLock<HashMap<String, ModuleMetadata>>>,
    
    /// Loading statistics
    stats: Arc<RwLock<LoaderStats>>,
}

/// Loading statistics
#[derive(Debug, Default, Clone)]
pub struct LoaderStats {
    pub loads_attempted: usize,
    pub loads_successful: usize,
    pub loads_failed: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub total_modules_loaded: usize,
}

impl ModuleLoader {
    /// Create a new module loader
    pub fn new(config: LoaderConfig, package_manager: PackageManager) -> Self {
        ModuleLoader {
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            package_manager: Arc::new(RwLock::new(package_manager)),
            metadata_cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(LoaderStats::default())),
        }
    }
    
    /// Load a module from the specified source
    pub async fn load_module(&self, namespace: &str, source: ModuleSource) -> Result<Arc<Module>, ModuleError> {
        // Update stats
        self.stats.write().unwrap().loads_attempted += 1;
        
        // Check cache first
        if let Some(cached_module) = self.get_cached_module(namespace) {
            self.stats.write().unwrap().cache_hits += 1;
            return Ok(cached_module);
        }
        
        self.stats.write().unwrap().cache_misses += 1;
        
        // Load module based on source type
        let module = match source {
            ModuleSource::LocalPath(path) => {
                self.load_from_path(&path).await?
            },
            ModuleSource::Registry { name, version } => {
                self.load_from_registry(&name, version).await?
            },
            ModuleSource::Memory(module) => {
                module
            },
            ModuleSource::Url(url) => {
                self.load_from_url(&url).await?
            },
        };
        
        // Validate module
        module.validate()?;
        
        // Cache the module
        let module_arc = Arc::new(module);
        self.cache_module(namespace, module_arc.clone());
        
        // Update stats
        self.stats.write().unwrap().loads_successful += 1;
        self.stats.write().unwrap().total_modules_loaded += 1;
        
        Ok(module_arc)
    }
    
    /// Load module metadata without loading the full module
    pub async fn load_metadata(&self, namespace: &str, source: ModuleSource) -> Result<ModuleMetadata, ModuleError> {
        // Check metadata cache first
        if let Some(cached_metadata) = self.metadata_cache.read().unwrap().get(namespace) {
            return Ok(cached_metadata.clone());
        }
        
        // Load metadata based on source type
        let metadata = match source {
            ModuleSource::LocalPath(path) => {
                self.load_metadata_from_path(&path).await?
            },
            ModuleSource::Registry { name, version } => {
                self.load_metadata_from_registry(&name, version).await?
            },
            ModuleSource::Memory(module) => {
                module.metadata
            },
            ModuleSource::Url(url) => {
                self.load_metadata_from_url(&url).await?
            },
        };
        
        // Cache metadata
        self.metadata_cache.write().unwrap().insert(namespace.to_string(), metadata.clone());
        
        Ok(metadata)
    }
    
    /// Load multiple modules with dependency resolution
    pub async fn load_modules_with_dependencies(&self, modules: Vec<(String, ModuleSource)>) -> Result<Vec<Arc<Module>>, ModuleError> {
        let mut loaded_modules = Vec::new();
        let mut dependency_graph = HashMap::new();
        
        // Build dependency graph
        for (namespace, source) in modules {
            let metadata = self.load_metadata(&namespace, source.clone()).await?;
            dependency_graph.insert(namespace.clone(), (source, metadata.clone()));
        }
        
        // Topological sort for dependency order
        let load_order = self.topological_sort(&dependency_graph)?;
        
        // Load modules in dependency order
        for namespace in load_order {
            if let Some((source, _metadata)) = dependency_graph.get(&namespace) {
                let module = self.load_module(&namespace, source.clone()).await?;
                loaded_modules.push(module);
            }
        }
        
        Ok(loaded_modules)
    }
    
    /// Reload a module (for hot reloading)
    pub async fn reload_module(&self, namespace: &str, source: ModuleSource) -> Result<Arc<Module>, ModuleError> {
        if !self.config.hot_reload {
            return Err(ModuleError::PackageError {
                message: "Hot reloading is disabled".to_string(),
            });
        }
        
        // Remove from cache
        self.cache.write().unwrap().remove(namespace);
        self.metadata_cache.write().unwrap().remove(namespace);
        
        // Load fresh module
        self.load_module(namespace, source).await
    }
    
    /// Check if a module is cached
    pub fn is_cached(&self, namespace: &str) -> bool {
        self.cache.read().unwrap().contains_key(namespace)
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> LoaderStats {
        (*self.stats.read().unwrap()).clone()
    }
    
    /// Clear module cache
    pub fn clear_cache(&self) {
        self.cache.write().unwrap().clear();
        self.metadata_cache.write().unwrap().clear();
    }
    
    /// Get all cached module namespaces
    pub fn cached_modules(&self) -> Vec<String> {
        self.cache.read().unwrap().keys().cloned().collect()
    }
    
    /// Search for modules in configured search paths
    pub fn search_modules(&self, pattern: &str) -> Vec<ModuleSource> {
        let mut sources = Vec::new();
        
        for search_path in &self.config.search_paths {
            if let Ok(entries) = std::fs::read_dir(search_path) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        let name = path.file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("");
                        
                        if name.contains(pattern) {
                            sources.push(ModuleSource::LocalPath(path));
                        }
                    }
                }
            }
        }
        
        sources
    }
    
    // Private helper methods
    
    fn get_cached_module(&self, namespace: &str) -> Option<Arc<Module>> {
        self.cache.read().unwrap().get(namespace).cloned()
    }
    
    fn cache_module(&self, namespace: &str, module: Arc<Module>) {
        let mut cache = self.cache.write().unwrap();
        
        // Check cache size limit
        if cache.len() >= self.config.max_cache_size {
            // Remove oldest entry (simple FIFO for now)
            if let Some(oldest_key) = cache.keys().next().cloned() {
                cache.remove(&oldest_key);
            }
        }
        
        cache.insert(namespace.to_string(), module);
    }
    
    async fn load_from_path(&self, path: &Path) -> Result<Module, ModuleError> {
        // Check if path exists
        if !path.exists() {
            return Err(ModuleError::ModuleNotFound {
                name: path.to_string_lossy().to_string(),
            });
        }
        
        // Look for Lyra.toml in the directory
        let manifest_path = path.join("Lyra.toml");
        if !manifest_path.exists() {
            return Err(ModuleError::PackageError {
                message: format!("No Lyra.toml found in {}", path.display()),
            });
        }
        
        // Load and parse manifest
        let manifest_content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        let package_manifest: super::package::PackageManifest = toml::from_str(&manifest_content)
            .map_err(|e| ModuleError::ParseError {
                message: format!("Failed to parse manifest: {}", e),
            })?;
        
        // Create module metadata
        let metadata = ModuleMetadata {
            name: package_manifest.package.name.clone(),
            version: self.parse_version(&package_manifest.package.version)?,
            description: package_manifest.package.description,
            authors: package_manifest.package.authors,
            license: package_manifest.package.license,
            repository: package_manifest.package.repository,
            homepage: package_manifest.package.homepage,
            documentation: package_manifest.package.documentation,
            keywords: package_manifest.package.keywords,
            categories: package_manifest.package.categories,
        };
        
        // Create module
        let module = Module::new(metadata);
        
        // TODO: Compile and load source files from src/ directory
        // For now, create an empty module
        
        Ok(module)
    }
    
    async fn load_from_registry(&self, name: &str, version: Option<Version>) -> Result<Module, ModuleError> {
        let version_req = version.map(|v| v.to_string()).unwrap_or_else(|| "*".to_string());
        
        let bundle = self.package_manager.write().unwrap()
            .install_package(name, &version_req).await?;
        
        Ok(bundle.module)
    }
    
    async fn load_from_url(&self, url: &str) -> Result<Module, ModuleError> {
        // For now, return an error as URL loading is not implemented
        Err(ModuleError::PackageError {
            message: format!("URL loading not yet implemented: {}", url),
        })
    }
    
    async fn load_metadata_from_path(&self, path: &Path) -> Result<ModuleMetadata, ModuleError> {
        let manifest_path = path.join("Lyra.toml");
        if !manifest_path.exists() {
            return Err(ModuleError::PackageError {
                message: format!("No Lyra.toml found in {}", path.display()),
            });
        }
        
        let manifest_content = std::fs::read_to_string(&manifest_path)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to read manifest: {}", e),
            })?;
        
        let package_manifest: super::package::PackageManifest = toml::from_str(&manifest_content)
            .map_err(|e| ModuleError::ParseError {
                message: format!("Failed to parse manifest: {}", e),
            })?;
        
        Ok(ModuleMetadata {
            name: package_manifest.package.name,
            version: self.parse_version(&package_manifest.package.version)?,
            description: package_manifest.package.description,
            authors: package_manifest.package.authors,
            license: package_manifest.package.license,
            repository: package_manifest.package.repository,
            homepage: package_manifest.package.homepage,
            documentation: package_manifest.package.documentation,
            keywords: package_manifest.package.keywords,
            categories: package_manifest.package.categories,
        })
    }
    
    async fn load_metadata_from_registry(&self, name: &str, version: Option<Version>) -> Result<ModuleMetadata, ModuleError> {
        let version_obj = version.unwrap_or_else(|| Version::new(1, 0, 0));
        
        // For now, create dummy metadata
        // In a real implementation, this would query the registry for metadata
        Ok(ModuleMetadata {
            name: name.to_string(),
            version: version_obj,
            description: format!("Module {} from registry", name),
            authors: vec![],
            license: "Unknown".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        })
    }
    
    async fn load_metadata_from_url(&self, url: &str) -> Result<ModuleMetadata, ModuleError> {
        Err(ModuleError::PackageError {
            message: format!("URL metadata loading not yet implemented: {}", url),
        })
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
    
    fn topological_sort(&self, dependency_graph: &HashMap<String, (ModuleSource, ModuleMetadata)>) -> Result<Vec<String>, ModuleError> {
        // For now, return modules in arbitrary order
        // In a real implementation, this would perform proper topological sorting
        Ok(dependency_graph.keys().cloned().collect())
    }
}

/// Built-in module factory for standard library modules
pub struct BuiltinModuleFactory;

impl BuiltinModuleFactory {
    /// Create standard library modules
    pub fn create_stdlib_modules() -> Vec<(String, Module)> {
        let mut modules = Vec::new();
        
        // std::math module
        let math_metadata = ModuleMetadata {
            name: "std::math".to_string(),
            version: Version::new(0, 1, 0),
            description: "Mathematical functions and constants".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: Some("https://github.com/lyra-lang/lyra".to_string()),
            homepage: Some("https://lyra-lang.org".to_string()),
            documentation: Some("https://docs.lyra-lang.org/std/math".to_string()),
            keywords: vec!["math", "trigonometry", "algebra", "calculus"].into_iter().map(String::from).collect(),
            categories: vec!["mathematics".to_string()],
        };
        let math_module = Module::new(math_metadata);
        modules.push(("std::math".to_string(), math_module));
        
        // std::list module
        let list_metadata = ModuleMetadata {
            name: "std::list".to_string(),
            version: Version::new(0, 1, 0),
            description: "List manipulation and processing functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: Some("https://github.com/lyra-lang/lyra".to_string()),
            homepage: Some("https://lyra-lang.org".to_string()),
            documentation: Some("https://docs.lyra-lang.org/std/list".to_string()),
            keywords: vec!["list", "array", "collection", "sequence"].into_iter().map(String::from).collect(),
            categories: vec!["data-structures".to_string()],
        };
        let list_module = Module::new(list_metadata);
        modules.push(("std::list".to_string(), list_module));
        
        // std::string module
        let string_metadata = ModuleMetadata {
            name: "std::string".to_string(),
            version: Version::new(0, 1, 0),
            description: "String manipulation and text processing functions".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: Some("https://github.com/lyra-lang/lyra".to_string()),
            homepage: Some("https://lyra-lang.org".to_string()),
            documentation: Some("https://docs.lyra-lang.org/std/string".to_string()),
            keywords: vec!["string", "text", "manipulation", "processing"].into_iter().map(String::from).collect(),
            categories: vec!["text-processing".to_string()],
        };
        let string_module = Module::new(string_metadata);
        modules.push(("std::string".to_string(), string_module));
        
        // std::tensor module
        let tensor_metadata = ModuleMetadata {
            name: "std::tensor".to_string(),
            version: Version::new(0, 1, 0),
            description: "Tensor operations and linear algebra".to_string(),
            authors: vec!["Lyra Team".to_string()],
            license: "MIT".to_string(),
            repository: Some("https://github.com/lyra-lang/lyra".to_string()),
            homepage: Some("https://lyra-lang.org".to_string()),
            documentation: Some("https://docs.lyra-lang.org/std/tensor".to_string()),
            keywords: vec!["tensor", "linear-algebra", "matrix", "numpy"].into_iter().map(String::from).collect(),
            categories: vec!["linear-algebra", "scientific-computing"].into_iter().map(String::from).collect(),
        };
        let tensor_module = Module::new(tensor_metadata);
        modules.push(("std::tensor".to_string(), tensor_module));
        
        modules
    }
    
    /// Create a specific builtin module by name
    pub fn create_module(name: &str) -> Option<Module> {
        let modules = Self::create_stdlib_modules();
        modules.into_iter()
            .find(|(module_name, _)| module_name == name)
            .map(|(_, module)| module)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn create_test_package_manager() -> PackageManager {
        let temp_dir = TempDir::new().unwrap();
        PackageManager::new(temp_dir.path().to_path_buf())
    }

    #[test]
    fn test_loader_creation() {
        let config = LoaderConfig::default();
        let package_manager = create_test_package_manager();
        let loader = ModuleLoader::new(config, package_manager);
        
        assert_eq!(loader.cached_modules().len(), 0);
        assert!(!loader.is_cached("test"));
    }

    #[test]
    fn test_builtin_module_factory() {
        let modules = BuiltinModuleFactory::create_stdlib_modules();
        assert_eq!(modules.len(), 4);
        
        let module_names: Vec<_> = modules.iter().map(|(name, _)| name.as_str()).collect();
        assert!(module_names.contains(&"std::math"));
        assert!(module_names.contains(&"std::list"));
        assert!(module_names.contains(&"std::string"));
        assert!(module_names.contains(&"std::tensor"));
    }

    #[test]
    fn test_specific_builtin_module() {
        let math_module = BuiltinModuleFactory::create_module("std::math");
        assert!(math_module.is_some());
        
        let module = math_module.unwrap();
        assert_eq!(module.metadata.name, "std::math");
        assert!(module.metadata.description.contains("Mathematical"));
    }

    #[test]
    fn test_module_source_types() {
        let local_source = ModuleSource::LocalPath(PathBuf::from("/path/to/module"));
        let registry_source = ModuleSource::Registry {
            name: "test-module".to_string(),
            version: Some(Version::new(1, 0, 0)),
        };
        
        assert!(matches!(local_source, ModuleSource::LocalPath(_)));
        assert!(matches!(registry_source, ModuleSource::Registry { .. }));
    }

    #[test]
    fn test_loader_stats() {
        let config = LoaderConfig::default();
        let package_manager = create_test_package_manager();
        let loader = ModuleLoader::new(config, package_manager);
        
        let stats = loader.get_stats();
        assert_eq!(stats.loads_attempted, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.total_modules_loaded, 0);
    }
}