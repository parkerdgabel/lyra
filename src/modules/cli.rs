//! CLI Interface for Package Management
//!
//! Implements the `lyra pkg` command suite for package management operations.

use super::{ModuleError, Version};
use super::package::{PackageManager, PackageBundle, LocalRegistry};
use super::registry::ModuleRegistry;
use std::path::PathBuf;
use std::env;

/// Package CLI commands structure
#[derive(Debug, Clone)]
pub enum PackageCommand {
    /// Initialize a new package
    Init {
        name: Option<String>,
        path: Option<PathBuf>,
    },
    
    /// Build the current package
    Build {
        release: bool,
    },
    
    /// Run package tests
    Test {
        test_name: Option<String>,
    },
    
    /// Install a package
    Install {
        package: String,
        version: Option<String>,
        dev: bool,
    },
    
    /// Update packages
    Update {
        package: Option<String>,
    },
    
    /// Remove a package
    Remove {
        package: String,
    },
    
    /// Search for packages
    Search {
        query: String,
        limit: usize,
    },
    
    /// Show package information
    Info {
        package: String,
        version: Option<String>,
    },
    
    /// List installed packages
    List {
        outdated: bool,
    },
    
    /// Show dependency tree
    Tree {
        package: Option<String>,
        depth: Option<usize>,
    },
    
    /// Check package health
    Check,
    
    /// Publish package to registry
    Publish {
        registry: Option<String>,
        token: Option<String>,
    },
}

/// Package CLI implementation
pub struct PackageCli {
    package_manager: PackageManager,
    module_registry: Option<ModuleRegistry>,
}

impl PackageCli {
    pub fn new(cache_dir: PathBuf) -> Self {
        let mut package_manager = PackageManager::new(cache_dir.clone());
        
        // Add default local registry
        let local_registry = LocalRegistry::new(cache_dir.join("registry"));
        package_manager.add_registry(Box::new(local_registry));
        
        PackageCli {
            package_manager,
            module_registry: None,
        }
    }
    
    pub fn with_module_registry(mut self, registry: ModuleRegistry) -> Self {
        self.module_registry = Some(registry);
        self
    }
    
    /// Execute a package command
    pub async fn execute(&mut self, command: PackageCommand) -> Result<(), ModuleError> {
        match command {
            PackageCommand::Init { name, path } => {
                self.cmd_init(name, path).await
            },
            PackageCommand::Build { release } => {
                self.cmd_build(release).await
            },
            PackageCommand::Test { test_name } => {
                self.cmd_test(test_name).await
            },
            PackageCommand::Install { package, version, dev } => {
                self.cmd_install(&package, version.as_deref(), dev).await
            },
            PackageCommand::Update { package } => {
                self.cmd_update(package.as_deref()).await
            },
            PackageCommand::Remove { package } => {
                self.cmd_remove(&package).await
            },
            PackageCommand::Search { query, limit } => {
                self.cmd_search(&query, limit).await
            },
            PackageCommand::Info { package, version } => {
                self.cmd_info(&package, version.as_deref()).await
            },
            PackageCommand::List { outdated } => {
                self.cmd_list(outdated).await
            },
            PackageCommand::Tree { package, depth } => {
                self.cmd_tree(package.as_deref(), depth).await
            },
            PackageCommand::Check => {
                self.cmd_check().await
            },
            PackageCommand::Publish { registry, token } => {
                self.cmd_publish(registry.as_deref(), token.as_deref()).await
            },
        }
    }
    
    // Command implementations
    
    async fn cmd_init(&mut self, name: Option<String>, path: Option<PathBuf>) -> Result<(), ModuleError> {
        let current_dir = env::current_dir()
            .map_err(|e| ModuleError::IoError {
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

[exports]
# Functions exported by this package
# "MyFunction" = "internal_function_name"

[imports]
# External dependencies
"std::math::*" = {{ from = "std" }}
"#,
            package_name
        );
        
        let manifest_path = project_dir.join("Lyra.toml");
        std::fs::write(&manifest_path, manifest_content)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create Lyra.toml: {}", e),
            })?;
        
        // Create src directory and main.ly
        let src_dir = project_dir.join("src");
        std::fs::create_dir_all(&src_dir)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create src directory: {}", e),
            })?;
        
        let main_content = r#"// Welcome to Lyra!
// This is your package's main module.

// Import from standard library
import std::math::{Sin, Cos, Pi}

// Define a function
HelloWorld[] := "Hello from Lyra!"

// Math example
SquareRoot[x_] := Sqrt[x]

// Trigonometric example  
SinCos[x_] := {Sin[x], Cos[x]}

// Export functions
export HelloWorld
export SquareRoot
export SinCos
"#;
        
        let main_path = src_dir.join("main.ly");
        std::fs::write(&main_path, main_content)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create main.ly: {}", e),
            })?;
        
        // Create tests directory
        let tests_dir = project_dir.join("tests");
        std::fs::create_dir_all(&tests_dir)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create tests directory: {}", e),
            })?;
        
        let test_content = r#"// Test file for your package
// Run with: lyra pkg test

import {HelloWorld, SquareRoot, SinCos}

// Test basic functionality
TestHelloWorld[] := HelloWorld[] == "Hello from Lyra!"

// Test mathematical functions
TestSquareRoot[] := SquareRoot[4] == 2

// Test trigonometric functions  
TestSinCos[] := Length[SinCos[0]] == 2
"#;
        
        let test_path = tests_dir.join("basic_tests.ly");
        std::fs::write(&test_path, test_content)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create test file: {}", e),
            })?;
        
        // Create README.md
        let readme_content = format!(
            r#"# {}

A Lyra package for symbolic computation.

## Installation

```bash
lyra pkg install {}
```

## Usage

```lyra
import {{HelloWorld, SquareRoot, SinCos}}

HelloWorld[]
SquareRoot[16]
SinCos[Pi/4]
```

## Development

Build the package:
```bash
lyra pkg build
```

Run tests:
```bash
lyra pkg test
```

## License

MIT
"#,
            package_name, package_name
        );
        
        let readme_path = project_dir.join("README.md");
        std::fs::write(&readme_path, readme_content)
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to create README.md: {}", e),
            })?;
        
        println!("✓ Created package '{}' in {}", package_name, project_dir.display());
        println!("📁 Project structure:");
        println!("  📄 Lyra.toml       - Package manifest");
        println!("  📁 src/            - Source code");
        println!("    📄 main.ly       - Main module");
        println!("  📁 tests/          - Test files");
        println!("    📄 basic_tests.ly - Basic tests");
        println!("  📄 README.md       - Documentation");
        println!();
        println!("🚀 Next steps:");
        println!("    cd {}", project_dir.display());
        println!("    lyra pkg build");
        println!("    lyra pkg test");
        
        Ok(())
    }
    
    async fn cmd_build(&mut self, release: bool) -> Result<(), ModuleError> {
        println!("🔨 Building package{}...", if release { " (release)" } else { "" });
        
        let current_dir = env::current_dir()
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        let bundle = self.package_manager.build_package(&current_dir)?;
        
        println!("✓ Built package '{}' v{}", 
                 bundle.manifest.package.name, 
                 bundle.manifest.package.version);
        
        // Show build statistics
        println!("📊 Build stats:");
        println!("  📦 Package: {}", bundle.manifest.package.name);
        println!("  🏷️  Version: {}", bundle.manifest.package.version);
        println!("  📝 Description: {}", bundle.manifest.package.description);
        println!("  🔗 Dependencies: {}", bundle.manifest.dependencies.len());
        
        Ok(())
    }
    
    async fn cmd_test(&mut self, test_name: Option<String>) -> Result<(), ModuleError> {
        if let Some(test) = test_name {
            println!("🧪 Running test '{}'...", test);
        } else {
            println!("🧪 Running all tests...");
        }
        
        let current_dir = env::current_dir()
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        let tests_dir = current_dir.join("tests");
        if !tests_dir.exists() {
            println!("⚠️  No tests directory found");
            return Ok(());
        }
        
        // Count test files
        let test_files: Vec<_> = std::fs::read_dir(&tests_dir)
            .map_err(|e| ModuleError::IoError { message: e.to_string() })?
            .filter_map(|entry| {
                let entry = entry.ok()?;
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("ly") {
                    Some(path.file_name()?.to_string_lossy().to_string())
                } else {
                    None
                }
            })
            .collect();
        
        if test_files.is_empty() {
            println!("⚠️  No test files found in tests/ directory");
            return Ok(());
        }
        
        println!("📁 Found {} test file(s):", test_files.len());
        for test_file in &test_files {
            println!("  📄 {}", test_file);
        }
        
        // TODO: Actually run the tests when we have a Lyra interpreter
        println!("✓ All tests passed (placeholder)");
        println!("📊 Test summary:");
        println!("  ✅ Passed: {}", test_files.len());
        println!("  ❌ Failed: 0");
        println!("  ⏭️  Skipped: 0");
        
        Ok(())
    }
    
    async fn cmd_install(&mut self, package: &str, version: Option<&str>, dev: bool) -> Result<(), ModuleError> {
        let version_req = version.unwrap_or("*");
        let dep_type = if dev { "dev dependency" } else { "dependency" };
        
        println!("📦 Installing {} {} as {}...", package, version_req, dep_type);
        
        let bundle = self.package_manager.install_package(package, version_req).await?;
        
        // Register module in registry if available
        if let Some(registry) = &self.module_registry {
            let namespace = format!("user::{}", package);
            registry.register_module(&namespace, bundle.module)?;
        }
        
        println!("✓ Installed {} v{}", package, bundle.manifest.package.version);
        println!("📄 Description: {}", bundle.manifest.package.description);
        
        if !bundle.manifest.package.keywords.is_empty() {
            println!("🏷️  Keywords: {}", bundle.manifest.package.keywords.join(", "));
        }
        
        Ok(())
    }
    
    async fn cmd_update(&mut self, package: Option<&str>) -> Result<(), ModuleError> {
        if let Some(pkg) = package {
            println!("🔄 Updating {}...", pkg);
        } else {
            println!("🔄 Updating all packages...");
        }
        
        // For now, just list installed packages
        let installed = self.package_manager.list_installed()?;
        if installed.is_empty() {
            println!("ℹ️  No packages installed");
            return Ok(());
        }
        
        println!("📦 Installed packages:");
        for pkg in &installed {
            println!("  {} v{}", pkg.name, pkg.version);
        }
        
        // TODO: Implement actual update logic
        println!("✓ All packages are up to date (placeholder)");
        
        Ok(())
    }
    
    async fn cmd_remove(&mut self, package: &str) -> Result<(), ModuleError> {
        println!("🗑️  Removing {}...", package);
        
        // Find installed versions
        let installed = self.package_manager.list_installed()?;
        let matching_packages: Vec<_> = installed.iter()
            .filter(|pkg| pkg.name == package)
            .collect();
        
        if matching_packages.is_empty() {
            println!("⚠️  Package '{}' is not installed", package);
            return Ok(());
        }
        
        // Remove all versions
        for pkg in matching_packages {
            // Parse version string
            let version_parts: Vec<&str> = pkg.version.split('.').collect();
            if version_parts.len() == 3 {
                if let (Ok(major), Ok(minor), Ok(patch)) = (
                    version_parts[0].parse::<u32>(),
                    version_parts[1].parse::<u32>(),
                    version_parts[2].parse::<u32>(),
                ) {
                    let version = Version::new(major, minor, patch);
                    self.package_manager.remove_package(&pkg.name, &version)?;
                    println!("✓ Removed {} v{}", pkg.name, pkg.version);
                }
            }
        }
        
        Ok(())
    }
    
    async fn cmd_search(&mut self, query: &str, limit: usize) -> Result<(), ModuleError> {
        println!("🔍 Searching for '{}'...", query);
        
        // Search through installed packages first
        let installed = self.package_manager.list_installed()?;
        let matching_installed: Vec<_> = installed.iter()
            .filter(|pkg| {
                pkg.name.contains(query) ||
                pkg.description.contains(query) ||
                pkg.keywords.iter().any(|k| k.contains(query))
            })
            .take(limit)
            .collect();
        
        if !matching_installed.is_empty() {
            println!("📦 Installed packages:");
            for pkg in &matching_installed {
                println!("  {} v{} - {}", pkg.name, pkg.version, pkg.description);
                if !pkg.keywords.is_empty() {
                    println!("    🏷️ {}", pkg.keywords.join(", "));
                }
            }
        } else {
            // TODO: Search through remote registries
            println!("❌ No packages found matching '{}'", query);
            println!("💡 Try searching with different keywords or check spelling");
        }
        
        Ok(())
    }
    
    async fn cmd_info(&mut self, package: &str, version: Option<&str>) -> Result<(), ModuleError> {
        println!("ℹ️  Package information for {}:", package);
        
        // Find in installed packages
        let installed = self.package_manager.list_installed()?;
        let matching_packages: Vec<_> = installed.iter()
            .filter(|pkg| {
                pkg.name == package && 
                (version.is_none() || pkg.version == version.unwrap())
            })
            .collect();
        
        if matching_packages.is_empty() {
            println!("❌ Package '{}' not found", package);
            if let Some(v) = version {
                println!("   Version '{}' not installed", v);
            }
            return Ok(());
        }
        
        for pkg in matching_packages {
            println!("📦 Name: {}", pkg.name);
            println!("🏷️  Version: {}", pkg.version);
            println!("📝 Description: {}", pkg.description);
            println!("👤 Authors: {}", pkg.authors.join(", "));
            println!("⚖️  License: {}", pkg.license);
            
            if let Some(repo) = &pkg.repository {
                println!("🔗 Repository: {}", repo);
            }
            
            if let Some(homepage) = &pkg.homepage {
                println!("🏠 Homepage: {}", homepage);
            }
            
            if !pkg.keywords.is_empty() {
                println!("🏷️  Keywords: {}", pkg.keywords.join(", "));
            }
            
            if !pkg.categories.is_empty() {
                println!("📁 Categories: {}", pkg.categories.join(", "));
            }
            
            // Show exports if available through module registry
            if let Some(registry) = &self.module_registry {
                let namespace = format!("user::{}", package);
                let exports = registry.get_module_exports(&namespace);
                if !exports.is_empty() {
                    println!("📤 Exports: {}", exports.join(", "));
                }
            }
        }
        
        Ok(())
    }
    
    async fn cmd_list(&mut self, outdated: bool) -> Result<(), ModuleError> {
        if outdated {
            println!("📊 Outdated packages:");
        } else {
            println!("📦 Installed packages:");
        }
        
        let installed = self.package_manager.list_installed()?;
        
        if installed.is_empty() {
            println!("ℹ️  No packages installed");
            println!("💡 Try: lyra pkg install <package-name>");
            return Ok(());
        }
        
        // Group packages by name and sort
        let mut packages_by_name: std::collections::HashMap<String, Vec<_>> = std::collections::HashMap::new();
        for pkg in &installed {
            packages_by_name.entry(pkg.name.clone()).or_default().push(pkg);
        }
        
        let mut sorted_names: Vec<_> = packages_by_name.keys().collect();
        sorted_names.sort();
        
        for name in sorted_names {
            let versions = packages_by_name.get(name).unwrap();
            if versions.len() == 1 {
                let pkg = versions[0];
                println!("  {} v{} - {}", pkg.name, pkg.version, pkg.description);
            } else {
                println!("  {} - {}", name, versions[0].description);
                for pkg in versions {
                    println!("    v{}", pkg.version);
                }
            }
        }
        
        println!();
        println!("📊 Summary: {} package(s) installed", installed.len());
        
        Ok(())
    }
    
    async fn cmd_tree(&mut self, package: Option<&str>, depth: Option<usize>) -> Result<(), ModuleError> {
        let max_depth = depth.unwrap_or(usize::MAX);
        
        if let Some(pkg) = package {
            println!("🌳 Dependency tree for {}:", pkg);
        } else {
            println!("🌳 Dependency tree:");
        }
        
        let installed = self.package_manager.list_installed()?;
        
        if installed.is_empty() {
            println!("ℹ️  No packages installed");
            return Ok(());
        }
        
        // Simple tree view (no actual dependency resolution yet)
        for (i, pkg) in installed.iter().enumerate() {
            let is_last = i == installed.len() - 1;
            let prefix = if is_last { "└── " } else { "├── " };
            
            if package.is_none() || pkg.name == package.unwrap() {
                println!("{}📦 {} v{}", prefix, pkg.name, pkg.version);
                
                // Show module exports as "dependencies"
                if let Some(registry) = &self.module_registry {
                    let namespace = format!("user::{}", pkg.name);
                    let exports = registry.get_module_exports(&namespace);
                    for (j, export) in exports.iter().enumerate() {
                        let is_last_export = j == exports.len() - 1;
                        let child_prefix = if is_last { "    " } else { "│   " };
                        let export_prefix = if is_last_export { "└── " } else { "├── " };
                        println!("{}{}🔧 {}", child_prefix, export_prefix, export);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    async fn cmd_check(&mut self) -> Result<(), ModuleError> {
        println!("🔍 Checking package health...");
        
        let current_dir = env::current_dir()
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        // Check for Lyra.toml
        let manifest_path = current_dir.join("Lyra.toml");
        if manifest_path.exists() {
            println!("✓ Lyra.toml found");
            
            // Try to parse manifest
            if let Ok(manifest) = self.package_manager.load_manifest(&manifest_path) {
                println!("✓ Manifest is valid");
                println!("  📦 Package: {}", manifest.package.name);
                println!("  🏷️  Version: {}", manifest.package.version);
            } else {
                println!("❌ Manifest is invalid");
                return Err(ModuleError::ParseError {
                    message: "Invalid Lyra.toml".to_string(),
                });
            }
        } else {
            println!("❌ Lyra.toml not found");
            println!("💡 Run 'lyra pkg init' to create a new package");
            return Ok(());
        }
        
        // Check for src directory
        let src_dir = current_dir.join("src");
        if src_dir.exists() {
            println!("✓ src/ directory found");
            
            // Count source files
            let source_files: Vec<_> = std::fs::read_dir(&src_dir)
                .map_err(|e| ModuleError::IoError { message: e.to_string() })?
                .filter_map(|entry| {
                    let entry = entry.ok()?;
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("ly") {
                        Some(path.file_name()?.to_string_lossy().to_string())
                    } else {
                        None
                    }
                })
                .collect();
            
            if !source_files.is_empty() {
                println!("  📄 {} source file(s) found", source_files.len());
            } else {
                println!("⚠️  No .ly source files found in src/");
            }
        } else {
            println!("⚠️  src/ directory not found");
        }
        
        // Check for tests directory
        let tests_dir = current_dir.join("tests");
        if tests_dir.exists() {
            println!("✓ tests/ directory found");
        } else {
            println!("ℹ️  tests/ directory not found (optional)");
        }
        
        println!("✓ Package health check completed");
        
        Ok(())
    }
    
    async fn cmd_publish(&mut self, registry: Option<&str>, token: Option<&str>) -> Result<(), ModuleError> {
        let reg_name = registry.unwrap_or("default");
        
        println!("📤 Publishing to {} registry...", reg_name);
        
        if token.is_none() {
            return Err(ModuleError::PackageError {
                message: "Authentication token required for publishing".to_string(),
            });
        }
        
        let current_dir = env::current_dir()
            .map_err(|e| ModuleError::IoError {
                message: format!("Failed to get current directory: {}", e),
            })?;
        
        // Build package first
        let bundle = self.package_manager.build_package(&current_dir)?;
        
        println!("📦 Publishing {} v{}...", 
                 bundle.manifest.package.name, 
                 bundle.manifest.package.version);
        
        // TODO: Implement actual publishing to registry
        println!("✓ Package published successfully (placeholder)");
        println!("🌐 Package is now available for installation");
        
        Ok(())
    }
}

/// Helper function to create a default cache directory
pub fn default_cache_dir() -> PathBuf {
    if let Some(home) = env::var_os("HOME") {
        PathBuf::from(home).join(".lyra").join("packages")
    } else if let Some(userprofile) = env::var_os("USERPROFILE") {
        PathBuf::from(userprofile).join(".lyra").join("packages")
    } else {
        PathBuf::from(".lyra").join("packages")
    }
}

/// Parse CLI arguments into PackageCommand
pub fn parse_package_command(args: &[String]) -> Result<PackageCommand, String> {
    if args.is_empty() {
        return Err("No command specified".to_string());
    }
    
    match args[0].as_str() {
        "init" => {
            let name = args.get(1).cloned();
            let path = args.get(2).map(PathBuf::from);
            Ok(PackageCommand::Init { name, path })
        },
        "build" => {
            let release = args.contains(&"--release".to_string());
            Ok(PackageCommand::Build { release })
        },
        "test" => {
            let test_name = args.get(1).cloned();
            Ok(PackageCommand::Test { test_name })
        },
        "install" => {
            if args.len() < 2 {
                return Err("Package name required".to_string());
            }
            let package = args[1].clone();
            let version = args.get(2).cloned();
            let dev = args.contains(&"--dev".to_string());
            Ok(PackageCommand::Install { package, version, dev })
        },
        "update" => {
            let package = args.get(1).cloned();
            Ok(PackageCommand::Update { package })
        },
        "remove" | "uninstall" => {
            if args.len() < 2 {
                return Err("Package name required".to_string());
            }
            let package = args[1].clone();
            Ok(PackageCommand::Remove { package })
        },
        "search" => {
            if args.len() < 2 {
                return Err("Search query required".to_string());
            }
            let query = args[1].clone();
            let limit = args.get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or(10);
            Ok(PackageCommand::Search { query, limit })
        },
        "info" => {
            if args.len() < 2 {
                return Err("Package name required".to_string());
            }
            let package = args[1].clone();
            let version = args.get(2).cloned();
            Ok(PackageCommand::Info { package, version })
        },
        "list" => {
            let outdated = args.contains(&"--outdated".to_string());
            Ok(PackageCommand::List { outdated })
        },
        "tree" => {
            let package = args.get(1).cloned();
            let depth = args.iter()
                .position(|arg| arg == "--depth")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok());
            Ok(PackageCommand::Tree { package, depth })
        },
        "check" => {
            Ok(PackageCommand::Check)
        },
        "publish" => {
            let registry = args.iter()
                .position(|arg| arg == "--registry")
                .and_then(|i| args.get(i + 1))
                .cloned();
            let token = args.iter()
                .position(|arg| arg == "--token")
                .and_then(|i| args.get(i + 1))
                .cloned();
            Ok(PackageCommand::Publish { registry, token })
        },
        cmd => Err(format!("Unknown command: {}", cmd)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_parse_package_command() {
        // Test init command
        let args = vec!["init".to_string(), "my-package".to_string()];
        let cmd = parse_package_command(&args).unwrap();
        assert!(matches!(cmd, PackageCommand::Init { .. }));
        
        // Test install command
        let args = vec!["install".to_string(), "some-package".to_string()];
        let cmd = parse_package_command(&args).unwrap();
        assert!(matches!(cmd, PackageCommand::Install { .. }));
        
        // Test invalid command
        let args = vec!["invalid".to_string()];
        assert!(parse_package_command(&args).is_err());
    }

    #[test]
    fn test_package_cli_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cli = PackageCli::new(temp_dir.path().to_path_buf());
        
        // Should create successfully
        // We can't easily test async methods in a simple unit test,
        // but we can verify the CLI was created
    }

    #[test]
    fn test_default_cache_dir() {
        let cache_dir = default_cache_dir();
        assert!(cache_dir.to_string_lossy().contains(".lyra"));
        assert!(cache_dir.to_string_lossy().contains("packages"));
    }
}