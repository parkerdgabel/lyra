//! Integration tests for the Lyra module system
//!
//! Tests the complete module system including package management, namespace resolution,
//! and CLI functionality.

use lyra::modules::*;
use lyra::modules::registry::ModuleRegistry;
use lyra::modules::package::{PackageManager, LocalRegistry};
use lyra::modules::cli::{PackageCli, PackageCommand, default_cache_dir};
use lyra::modules::resolver::{ImportParser, ImportContext, ImportStatement};
use lyra::modules::loader::{ModuleLoader, LoaderConfig, ModuleSource, BuiltinModuleFactory};
use lyra::linker::FunctionRegistry;
use std::sync::{Arc, RwLock};
use tempfile::TempDir;
use std::path::PathBuf;

/// Test the core module creation and validation
#[test]
fn test_module_creation() {
    let metadata = ModuleMetadata {
        name: "test::module".to_string(),
        version: Version::new(1, 0, 0),
        description: "Test module".to_string(),
        authors: vec!["Test Author".to_string()],
        license: "MIT".to_string(),
        repository: None,
        homepage: None,
        documentation: None,
        keywords: vec!["test".to_string()],
        categories: vec!["testing".to_string()],
    };
    
    let module = Module::new(metadata);
    assert_eq!(module.metadata.name, "test::module");
    assert_eq!(module.metadata.version, Version::new(1, 0, 0));
    assert!(module.exports.is_empty());
    assert!(module.validate().is_ok());
}

/// Test module registry functionality
#[test]
fn test_module_registry() {
    let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
    let registry = ModuleRegistry::new(func_registry);
    
    // Should have registered standard library modules
    assert!(registry.has_module("std::math"));
    assert!(registry.has_module("std::list"));
    assert!(registry.has_module("std::string"));
    assert!(registry.has_module("std::tensor"));
    
    // Test module listing
    let modules = registry.list_modules();
    assert!(modules.len() >= 4);
    
    // Test module search
    let search_results = registry.search_modules("math");
    assert!(!search_results.is_empty());
    
    // Test module exports
    let math_exports = registry.get_module_exports("std::math");
    assert!(!math_exports.is_empty());
    assert!(math_exports.contains(&"Sin".to_string()));
    assert!(math_exports.contains(&"Cos".to_string()));
}

/// Test package manager basic functionality
#[test]
fn test_package_manager() {
    let temp_dir = TempDir::new().unwrap();
    let package_manager = PackageManager::new(temp_dir.path().to_path_buf());
    
    // Test version constraint parsing
    let version_constraint = "^1.2.3";
    // Note: parse_version_constraint is private, so we can't test it directly
    // In a real implementation, we'd make it public or test through public APIs
    
    // Test manifest creation
    let manifest = package::PackageManifest {
        package: package::PackageInfo {
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
        dependencies: std::collections::HashMap::new(),
        dev_dependencies: std::collections::HashMap::new(),
        features: std::collections::HashMap::new(),
        exports: std::collections::HashMap::new(),
        imports: std::collections::HashMap::new(),
        permissions: package::PermissionSpec::default(),
    };
    
    // Test manifest serialization
    let toml_str = toml::to_string(&manifest).unwrap();
    let parsed: package::PackageManifest = toml::from_str(&toml_str).unwrap();
    assert_eq!(parsed.package.name, "test-package");
}

/// Test import statement parsing
#[test]
fn test_import_parsing() {
    // Test module import
    let import = ImportParser::parse("import std::math").unwrap();
    match import {
        ImportStatement::Module { namespace, alias } => {
            assert_eq!(namespace, "std::math");
            assert_eq!(alias, None);
        },
        _ => panic!("Expected module import"),
    }
    
    // Test wildcard import
    let import = ImportParser::parse("import std::math::*").unwrap();
    match import {
        ImportStatement::Wildcard { namespace } => {
            assert_eq!(namespace, "std::math");
        },
        _ => panic!("Expected wildcard import"),
    }
    
    // Test specific imports
    let import = ImportParser::parse("import std::math::{Sin, Cos}").unwrap();
    match import {
        ImportStatement::Specific { namespace, functions, .. } => {
            assert_eq!(namespace, "std::math");
            assert_eq!(functions.len(), 2);
            assert!(functions.contains(&"Sin".to_string()));
            assert!(functions.contains(&"Cos".to_string()));
        },
        _ => panic!("Expected specific import"),
    }
    
    // Test aliased import
    let import = ImportParser::parse("import std::math::Sin as Sine").unwrap();
    match import {
        ImportStatement::Aliased { namespace, function, alias } => {
            assert_eq!(namespace, "std::math");
            assert_eq!(function, "Sin");
            assert_eq!(alias, "Sine");
        },
        _ => panic!("Expected aliased import"),
    }
}

/// Test import context and resolution
#[test]
fn test_import_context() {
    let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
    let registry = ModuleRegistry::new(func_registry);
    let mut context = ImportContext::new();
    
    // Test module import
    let module_import = ImportStatement::Module {
        namespace: "std::math".to_string(),
        alias: Some("math".to_string()),
    };
    
    context.add_import(module_import, &registry).unwrap();
    assert_eq!(context.imported_modules().len(), 1);
    
    // Test specific import
    let specific_import = ImportStatement::Specific {
        namespace: "std::math".to_string(),
        functions: vec!["Sin".to_string(), "Cos".to_string()],
        aliases: std::collections::HashMap::new(),
    };
    
    context.add_import(specific_import, &registry).unwrap();
    
    // Test function resolution
    assert_eq!(context.resolve_function("Sin"), Some("std::math::Sin".to_string()));
    assert_eq!(context.resolve_function("Cos"), Some("std::math::Cos".to_string()));
    assert_eq!(context.resolve_function("Unknown"), None);
}

/// Test CLI command parsing
#[test]
fn test_cli_command_parsing() {
    use lyra::modules::cli::parse_package_command;
    
    // Test init command
    let args = vec!["init".to_string(), "my-package".to_string()];
    let cmd = parse_package_command(&args).unwrap();
    assert!(matches!(cmd, PackageCommand::Init { .. }));
    
    // Test install command
    let args = vec!["install".to_string(), "some-package".to_string()];
    let cmd = parse_package_command(&args).unwrap();
    assert!(matches!(cmd, PackageCommand::Install { .. }));
    
    // Test build command
    let args = vec!["build".to_string()];
    let cmd = parse_package_command(&args).unwrap();
    assert!(matches!(cmd, PackageCommand::Build { .. }));
    
    // Test invalid command
    let args = vec!["invalid".to_string()];
    assert!(parse_package_command(&args).is_err());
}

/// Test module loader functionality
#[test]
fn test_module_loader() {
    let temp_dir = TempDir::new().unwrap();
    let config = LoaderConfig::default();
    let package_manager = PackageManager::new(temp_dir.path().to_path_buf());
    let loader = ModuleLoader::new(config, package_manager);
    
    // Test initial state
    assert_eq!(loader.cached_modules().len(), 0);
    assert!(!loader.is_cached("test"));
    
    // Test stats
    let stats = loader.get_stats();
    assert_eq!(stats.loads_attempted, 0);
    assert_eq!(stats.cache_hits, 0);
}

/// Test builtin module factory
#[test]
fn test_builtin_modules() {
    let modules = BuiltinModuleFactory::create_stdlib_modules();
    assert_eq!(modules.len(), 4);
    
    let module_names: Vec<_> = modules.iter().map(|(name, _)| name.as_str()).collect();
    assert!(module_names.contains(&"std::math"));
    assert!(module_names.contains(&"std::list"));
    assert!(module_names.contains(&"std::string"));
    assert!(module_names.contains(&"std::tensor"));
    
    // Test specific module creation
    let math_module = BuiltinModuleFactory::create_module("std::math");
    assert!(math_module.is_some());
    
    let module = math_module.unwrap();
    assert_eq!(module.metadata.name, "std::math");
    assert!(module.metadata.description.contains("Mathematical"));
}

/// Test version constraints and resolution
#[test]
fn test_version_constraints() {
    let version = Version::new(1, 2, 3);
    
    // Test exact version
    assert!(version.satisfies(&VersionConstraint::Exact(Version::new(1, 2, 3))));
    assert!(!version.satisfies(&VersionConstraint::Exact(Version::new(1, 2, 4))));
    
    // Test greater or equal
    assert!(version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 3))));
    assert!(version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 2))));
    assert!(!version.satisfies(&VersionConstraint::GreaterEqual(Version::new(1, 2, 4))));
    
    // Test compatible version
    assert!(version.satisfies(&VersionConstraint::Compatible(Version::new(1, 2, 2))));
    assert!(!version.satisfies(&VersionConstraint::Compatible(Version::new(2, 0, 0))));
    
    // Test range
    assert!(version.satisfies(&VersionConstraint::Range {
        min: Version::new(1, 0, 0),
        max: Version::new(2, 0, 0),
    }));
    assert!(!version.satisfies(&VersionConstraint::Range {
        min: Version::new(2, 0, 0),
        max: Version::new(3, 0, 0),
    }));
}

/// Test local registry functionality
#[test]
fn test_local_registry() {
    let temp_dir = TempDir::new().unwrap();
    let registry = LocalRegistry::new(temp_dir.path().to_path_buf());
    
    // Test empty registry search
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(registry.search("test"));
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
    
    // Test version listing for non-existent package
    let versions = rt.block_on(registry.list_versions("nonexistent"));
    assert!(versions.is_ok());
    assert!(versions.unwrap().is_empty());
}

/// Test package CLI initialization
#[test]
fn test_package_cli() {
    let temp_dir = TempDir::new().unwrap();
    let cli = PackageCli::new(temp_dir.path().to_path_buf());
    
    // CLI should be created successfully
    // We can't easily test async methods in a simple unit test,
    // but we can verify the CLI was created without panicking
}

/// Test error handling
#[test]
fn test_error_handling() {
    // Test invalid module name
    let metadata = ModuleMetadata {
        name: "".to_string(), // Invalid empty name
        version: Version::new(1, 0, 0),
        description: "Test".to_string(),
        authors: vec![],
        license: "MIT".to_string(),
        repository: None,
        homepage: None,
        documentation: None,
        keywords: vec![],
        categories: vec![],
    };
    
    let module = Module::new(metadata);
    assert!(module.validate().is_err());
    
    // Test invalid import syntax
    assert!(ImportParser::parse("invalid statement").is_err());
    assert!(ImportParser::parse("import").is_err());
    
    // Test version parsing errors
    // These would be tested through public APIs once available
}

/// Test namespace integration with function registry
#[test]
fn test_namespace_function_registry() {
    let mut func_registry = FunctionRegistry::new();
    
    // Test namespace functions
    func_registry.register_stdlib_function("Sin", crate::stdlib::math::sin, 1).unwrap();
    func_registry.register_namespaced_function(
        "std::math::Sin",
        crate::stdlib::math::sin,
        1,
        vec![crate::linker::FunctionAttribute::Listable],
    ).unwrap();
    
    // Test namespace queries
    assert!(func_registry.has_namespace("std::math"));
    assert!(!func_registry.has_namespace("nonexistent"));
    
    let namespaces = func_registry.get_namespaces();
    assert!(namespaces.contains(&"std::math".to_string()));
    
    let math_functions = func_registry.get_namespace_functions("std::math");
    assert!(math_functions.contains(&"Sin".to_string()));
}

/// Test end-to-end module system workflow
#[test]
fn test_end_to_end_workflow() {
    // Create a module registry with function registry
    let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
    let mut module_registry = ModuleRegistry::new(func_registry.clone());
    
    // Create a custom test module
    let metadata = ModuleMetadata {
        name: "test::custom".to_string(),
        version: Version::new(1, 0, 0),
        description: "Custom test module".to_string(),
        authors: vec!["Test Author".to_string()],
        license: "MIT".to_string(),
        repository: None,
        homepage: None,
        documentation: None,
        keywords: vec!["test".to_string()],
        categories: vec!["testing".to_string()],
    };
    
    let mut custom_module = Module::new(metadata);
    
    // Add a test export
    let test_export = FunctionExport {
        internal_name: "test_function".to_string(),
        export_name: "TestFunction".to_string(),
        signature: crate::linker::FunctionSignature::new("Module", "TestFunction", 1),
        implementation: FunctionImplementation::Native(crate::stdlib::math::sin), // Using sin as placeholder
        attributes: vec![],
        documentation: Some("A test function".to_string()),
    };
    
    custom_module.add_export(test_export).unwrap();
    
    // Register the module
    module_registry.register_module("test::custom", custom_module).unwrap();
    
    // Verify registration
    assert!(module_registry.has_module("test::custom"));
    
    let custom_exports = module_registry.get_module_exports("test::custom");
    assert!(custom_exports.contains(&"TestFunction".to_string()));
    
    // Test import context with the new module
    let mut import_context = ImportContext::new();
    let specific_import = ImportStatement::Specific {
        namespace: "test::custom".to_string(),
        functions: vec!["TestFunction".to_string()],
        aliases: std::collections::HashMap::new(),
    };
    
    import_context.add_import(specific_import, &module_registry).unwrap();
    
    // Test function resolution
    assert_eq!(
        import_context.resolve_function("TestFunction"),
        Some("test::custom::TestFunction".to_string())
    );
}

/// Test module system performance characteristics
#[test]
fn test_module_performance() {
    let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
    let registry = ModuleRegistry::new(func_registry);
    
    // Measure time for large number of lookups
    let start = std::time::Instant::now();
    
    for _ in 0..1000 {
        let _ = registry.has_module("std::math");
        let _ = registry.get_module_exports("std::math");
        let _ = registry.search_modules("math");
    }
    
    let elapsed = start.elapsed();
    
    // Should complete quickly (under 100ms for 1000 operations)
    assert!(elapsed.as_millis() < 100, "Module operations took too long: {:?}", elapsed);
}

/// Test concurrent access to module registry
#[test]
fn test_concurrent_access() {
    use std::sync::Arc;
    use std::thread;
    
    let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
    let registry = Arc::new(ModuleRegistry::new(func_registry));
    
    let mut handles = vec![];
    
    // Spawn multiple threads accessing the registry
    for i in 0..10 {
        let registry_clone = Arc::clone(&registry);
        let handle = thread::spawn(move || {
            // Each thread performs various read operations
            let modules = registry_clone.list_modules();
            assert!(!modules.is_empty());
            
            let has_math = registry_clone.has_module("std::math");
            assert!(has_math);
            
            let exports = registry_clone.get_module_exports("std::math");
            assert!(!exports.is_empty());
            
            let search_results = registry_clone.search_modules("math");
            assert!(!search_results.is_empty());
            
            i // Return thread id for verification
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        let thread_id = handle.join().unwrap();
        assert!(thread_id < 10);
    }
}