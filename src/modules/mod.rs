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
pub mod compiler_integration;

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
    
    #[error("IO error: {message}")]
    IoError { message: String },
    
    #[error("Parse error: {message}")]
    ParseError { message: String },
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

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)?;
        if let Some(pre) = &self.pre {
            write!(f, "-{}", pre)?;
        }
        if let Some(build) = &self.build {
            write!(f, "+{}", build)?;
        }
        Ok(())
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
    
    /// Get a qualified name for this module
    pub fn qualified_name(&self) -> &str {
        &self.metadata.name
    }
    
    /// Check if this module has a specific export
    pub fn has_export(&self, name: &str) -> bool {
        self.exports.contains_key(name)
    }
    
    /// Get an export by name
    pub fn get_export(&self, name: &str) -> Option<&FunctionExport> {
        self.exports.get(name)
    }
    
    /// Get all submodule names
    pub fn submodule_names(&self) -> Vec<&String> {
        self.submodules.keys().collect()
    }
    
    /// Get a submodule by name
    pub fn get_submodule(&self, name: &str) -> Option<&Module> {
        self.submodules.get(name)
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

/// Create a function export with documentation
pub fn stdlib_to_export_with_docs(
    name: &str,
    func: StdlibFunction,
    attributes: Vec<FunctionAttribute>,
    docs: &str,
) -> FunctionExport {
    FunctionExport {
        internal_name: name.to_string(),
        export_name: name.to_string(),
        signature: FunctionSignature::infer_from_attributes(&attributes),
        implementation: FunctionImplementation::Native(func),
        attributes,
        documentation: Some(docs.to_string()),
    }
}

// Add a helper extension to FunctionSignature for attributes inference
impl FunctionSignature {
    /// Create a function signature by inferring arity from attributes
    pub fn infer_from_attributes(attributes: &[FunctionAttribute]) -> Self {
        // For now, create a basic signature - this would be expanded with proper type inference
        FunctionSignature::new("Stdlib", "Unknown", 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_creation() {
        let version = Version::new(1, 2, 3);
        assert_eq!(version.major, 1);
        assert_eq!(version.minor, 2);
        assert_eq!(version.patch, 3);
        assert_eq!(version.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_satisfies() {
        let version = Version::new(1, 2, 3);
        
        assert!(version.satisfies(&VersionConstraint::Exact(Version::new(1, 2, 3))));
        assert!(!version.satisfies(&VersionConstraint::Exact(Version::new(1, 2, 4))));
        
        assert!(version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 3))));
        assert!(version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 2))));
        assert!(!version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 4))));
        
        assert!(version.satisfies(&VersionConstraint::Compatible(Version::new(1, 2, 2))));
        assert!(!version.satisfies(&VersionConstraint::Compatible(Version::new(2, 0, 0))));
    }

    #[test]
    fn test_module_creation() {
        let metadata = ModuleMetadata {
            name: "test".to_string(),
            version: Version::new(1, 0, 0),
            description: "Test module".to_string(),
            authors: vec!["Test Author".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        };
        
        let module = Module::new(metadata);
        assert_eq!(module.metadata.name, "test");
        assert_eq!(module.metadata.version, Version::new(1, 0, 0));
        assert!(module.exports.is_empty());
        assert!(module.submodules.is_empty());
    }

    #[test]
    fn test_module_validation() {
        let metadata = ModuleMetadata {
            name: "test".to_string(),
            version: Version::new(1, 0, 0),
            description: "Test module".to_string(),
            authors: vec!["Test Author".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        };
        
        let module = Module::new(metadata);
        assert!(module.validate().is_ok());
        
        // Test invalid name
        let invalid_metadata = ModuleMetadata {
            name: "".to_string(),
            version: Version::new(1, 0, 0),
            description: "Test module".to_string(),
            authors: vec!["Test Author".to_string()],
            license: "MIT".to_string(),
            repository: None,
            homepage: None,
            documentation: None,
            keywords: vec![],
            categories: vec![],
        };
        
        let invalid_module = Module::new(invalid_metadata);
        assert!(invalid_module.validate().is_err());
    }
}