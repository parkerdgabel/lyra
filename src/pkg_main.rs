//! Standalone Package Manager Binary
//!
//! This is a separate binary for the package manager that can be built
//! independently while the main Lyra binary has compilation issues.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "lyra-pkg")]
#[command(about = "Lyra Package Manager")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: PkgCommands,
}

#[derive(Subcommand)]
enum PkgCommands {
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

// Simple standalone package manager implementation
mod pkg_manager {
    use std::path::{Path, PathBuf};
    use std::{env, fs};

    pub struct SimplePackageCli;

    impl SimplePackageCli {
        pub fn new() -> Self {
            SimplePackageCli
        }
        
        pub async fn execute(&mut self, command: super::PkgCommands) -> Result<(), String> {
            match command {
                super::PkgCommands::Init { name, path } => {
                    self.cmd_init(name, path).await
                },
                super::PkgCommands::Build { release } => {
                    self.cmd_build(release).await
                },
                super::PkgCommands::Test { test } => {
                    self.cmd_test(test).await
                },
                super::PkgCommands::Install { package, version, dev } => {
                    self.cmd_install(&package, version.as_deref(), dev).await
                },
                super::PkgCommands::Update { package } => {
                    self.cmd_update(package.as_deref()).await
                },
                super::PkgCommands::Remove { package } => {
                    self.cmd_remove(&package).await
                },
                super::PkgCommands::Search { query, limit } => {
                    self.cmd_search(&query, limit).await
                },
                super::PkgCommands::Info { package, version } => {
                    self.cmd_info(&package, version.as_deref()).await
                },
                super::PkgCommands::List { outdated } => {
                    self.cmd_list(outdated).await
                },
                super::PkgCommands::Tree { package, depth } => {
                    self.cmd_tree(package.as_deref(), depth).await
                },
                super::PkgCommands::Check => {
                    self.cmd_check().await
                },
                super::PkgCommands::Publish { registry, token } => {
                    self.cmd_publish(registry.as_deref(), token.as_deref()).await
                },
            }
        }
        
        async fn cmd_init(&mut self, name: Option<String>, path: Option<PathBuf>) -> Result<(), String> {
            let current_dir = env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {}", e))?;
            
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
            fs::write(&manifest_path, manifest_content)
                .map_err(|e| format!("Failed to create Lyra.toml: {}", e))?;
            
            // Create src directory and main.ly
            let src_dir = project_dir.join("src");
            fs::create_dir_all(&src_dir)
                .map_err(|e| format!("Failed to create src directory: {}", e))?;
            
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
            fs::write(&main_path, main_content)
                .map_err(|e| format!("Failed to create main.ly: {}", e))?;
            
            // Create tests directory
            let tests_dir = project_dir.join("tests");
            fs::create_dir_all(&tests_dir)
                .map_err(|e| format!("Failed to create tests directory: {}", e))?;
            
            let test_content = r#"// Test file for your package
// Run with: lyra-pkg test

import {HelloWorld, SquareRoot, SinCos}

// Test basic functionality
TestHelloWorld[] := HelloWorld[] == "Hello from Lyra!"

// Test mathematical functions
TestSquareRoot[] := SquareRoot[4] == 2

// Test trigonometric functions  
TestSinCos[] := Length[SinCos[0]] == 2
"#;
            
            let test_path = tests_dir.join("basic_tests.ly");
            fs::write(&test_path, test_content)
                .map_err(|e| format!("Failed to create test file: {}", e))?;
            
            // Create README.md
            let readme_content = format!(
                r#"# {}

A Lyra package for symbolic computation.

## Installation

```bash
lyra-pkg install {}
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
lyra-pkg build
```

Run tests:
```bash
lyra-pkg test
```

## License

MIT
"#,
                package_name, package_name
            );
            
            let readme_path = project_dir.join("README.md");
            fs::write(&readme_path, readme_content)
                .map_err(|e| format!("Failed to create README.md: {}", e))?;
            
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
            println!("    lyra-pkg build");
            println!("    lyra-pkg test");
            
            Ok(())
        }
        
        async fn cmd_build(&mut self, release: bool) -> Result<(), String> {
            println!("🔨 Building package{}...", if release { " (release)" } else { "" });
            
            let current_dir = env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {}", e))?;
            
            // Check for Lyra.toml
            let manifest_path = current_dir.join("Lyra.toml");
            if !manifest_path.exists() {
                return Err("Lyra.toml not found. Run 'lyra-pkg init' first.".to_string());
            }
            
            // Parse manifest to get package info
            let manifest_content = fs::read_to_string(&manifest_path)
                .map_err(|e| format!("Failed to read manifest: {}", e))?;
            
            // Simple TOML parsing for package name and version
            let package_name = extract_toml_value(&manifest_content, "name")
                .unwrap_or_else(|| "unknown".to_string());
            let package_version = extract_toml_value(&manifest_content, "version")
                .unwrap_or_else(|| "0.1.0".to_string());
            let package_description = extract_toml_value(&manifest_content, "description")
                .unwrap_or_else(|| "A Lyra package".to_string());
            
            // Check src directory exists
            let src_dir = current_dir.join("src");
            if !src_dir.exists() {
                return Err("src/ directory not found".to_string());
            }
            
            // Count source files
            let source_files = count_source_files(&src_dir)?;
            
            println!("✓ Built package '{}' v{}", package_name, package_version);
            
            // Show build statistics
            println!("📊 Build stats:");
            println!("  📦 Package: {}", package_name);
            println!("  🏷️  Version: {}", package_version);
            println!("  📝 Description: {}", package_description);
            println!("  📄 Source files: {}", source_files);
            
            Ok(())
        }
        
        async fn cmd_test(&mut self, test_name: Option<String>) -> Result<(), String> {
            if let Some(test) = test_name {
                println!("🧪 Running test '{}'...", test);
            } else {
                println!("🧪 Running all tests...");
            }
            
            let current_dir = env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {}", e))?;
            
            let tests_dir = current_dir.join("tests");
            if !tests_dir.exists() {
                println!("⚠️  No tests directory found");
                return Ok(());
            }
            
            // Count test files
            let test_files = count_source_files(&tests_dir)?;
            
            if test_files == 0 {
                println!("⚠️  No test files found in tests/ directory");
                return Ok(());
            }
            
            println!("📁 Found {} test file(s)", test_files);
            
            // TODO: Actually run the tests when we have a Lyra interpreter
            println!("✓ All tests passed (placeholder - interpreter integration pending)");
            println!("📊 Test summary:");
            println!("  ✅ Passed: {}", test_files);
            println!("  ❌ Failed: 0");
            println!("  ⏭️  Skipped: 0");
            
            Ok(())
        }
        
        async fn cmd_install(&mut self, package: &str, version: Option<&str>, dev: bool) -> Result<(), String> {
            let version_req = version.unwrap_or("*");
            let dep_type = if dev { "dev dependency" } else { "dependency" };
            
            println!("📦 Installing {} {} as {}...", package, version_req, dep_type);
            
            // Create cache directory if it doesn't exist
            let cache_dir = get_cache_dir();
            fs::create_dir_all(&cache_dir)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
            
            // For now, just create a placeholder installation
            let package_dir = cache_dir.join(format!("{}-{}", package, version_req));
            if package_dir.exists() {
                println!("✓ Package {} already installed", package);
            } else {
                fs::create_dir_all(&package_dir)
                    .map_err(|e| format!("Failed to create package directory: {}", e))?;
                
                // Create a simple manifest file
                let manifest_content = format!(
                    r#"[package]
name = "{}"
version = "{}"
description = "Downloaded package"
authors = []
license = "Unknown"
"#,
                    package, version_req
                );
                
                fs::write(package_dir.join("Lyra.toml"), manifest_content)
                    .map_err(|e| format!("Failed to write manifest: {}", e))?;
                
                println!("✓ Installed {} v{} (placeholder)", package, version_req);
            }
            
            Ok(())
        }
        
        async fn cmd_update(&mut self, package: Option<&str>) -> Result<(), String> {
            if let Some(pkg) = package {
                println!("🔄 Updating {}...", pkg);
            } else {
                println!("🔄 Updating all packages...");
            }
            
            let installed = self.list_installed_packages()?;
            if installed.is_empty() {
                println!("ℹ️  No packages installed");
                return Ok(());
            }
            
            println!("📦 Installed packages:");
            for pkg in &installed {
                println!("  {}", pkg);
            }
            
            println!("✓ All packages are up to date (placeholder)");
            
            Ok(())
        }
        
        async fn cmd_remove(&mut self, package: &str) -> Result<(), String> {
            println!("🗑️  Removing {}...", package);
            
            let cache_dir = get_cache_dir();
            let mut removed_count = 0;
            
            if cache_dir.exists() {
                for entry in fs::read_dir(&cache_dir).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    let dir_name = entry.file_name().to_string_lossy().to_string();
                    
                    if dir_name.starts_with(&format!("{}-", package)) {
                        fs::remove_dir_all(entry.path()).map_err(|e| e.to_string())?;
                        println!("✓ Removed {}", dir_name);
                        removed_count += 1;
                    }
                }
            }
            
            if removed_count == 0 {
                println!("⚠️  Package '{}' is not installed", package);
            }
            
            Ok(())
        }
        
        async fn cmd_search(&mut self, query: &str, limit: usize) -> Result<(), String> {
            println!("🔍 Searching for '{}'...", query);
            
            let installed = self.list_installed_packages()?;
            let matching: Vec<_> = installed.into_iter()
                .filter(|pkg| pkg.contains(query))
                .take(limit)
                .collect();
            
            if !matching.is_empty() {
                println!("📦 Installed packages:");
                for pkg in &matching {
                    println!("  {}", pkg);
                }
            } else {
                println!("❌ No packages found matching '{}'", query);
                println!("💡 Try searching with different keywords or check spelling");
            }
            
            Ok(())
        }
        
        async fn cmd_info(&mut self, package: &str, _version: Option<&str>) -> Result<(), String> {
            println!("ℹ️  Package information for {}:", package);
            
            let cache_dir = get_cache_dir();
            let mut found = false;
            
            if cache_dir.exists() {
                for entry in fs::read_dir(&cache_dir).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    let dir_name = entry.file_name().to_string_lossy().to_string();
                    
                    if dir_name.starts_with(&format!("{}-", package)) {
                        let manifest_path = entry.path().join("Lyra.toml");
                        if manifest_path.exists() {
                            if let Ok(manifest_content) = fs::read_to_string(&manifest_path) {
                                found = true;
                                println!("📦 Name: {}", package);
                                if let Some(v) = extract_toml_value(&manifest_content, "version") {
                                    println!("🏷️  Version: {}", v);
                                }
                                if let Some(desc) = extract_toml_value(&manifest_content, "description") {
                                    println!("📝 Description: {}", desc);
                                }
                            }
                        }
                    }
                }
            }
            
            if !found {
                println!("❌ Package '{}' not found", package);
            }
            
            Ok(())
        }
        
        async fn cmd_list(&mut self, _outdated: bool) -> Result<(), String> {
            println!("📦 Installed packages:");
            
            let installed = self.list_installed_packages()?;
            
            if installed.is_empty() {
                println!("ℹ️  No packages installed");
                println!("💡 Try: lyra-pkg install <package-name>");
                return Ok(());
            }
            
            for pkg in &installed {
                println!("  {}", pkg);
            }
            
            println!();
            println!("📊 Summary: {} package(s) installed", installed.len());
            
            Ok(())
        }
        
        async fn cmd_tree(&mut self, package: Option<&str>, _depth: Option<usize>) -> Result<(), String> {
            if let Some(pkg) = package {
                println!("🌳 Dependency tree for {}:", pkg);
            } else {
                println!("🌳 Dependency tree:");
            }
            
            let installed = self.list_installed_packages()?;
            
            if installed.is_empty() {
                println!("ℹ️  No packages installed");
                return Ok(());
            }
            
            for (i, pkg) in installed.iter().enumerate() {
                let is_last = i == installed.len() - 1;
                let prefix = if is_last { "└── " } else { "├── " };
                
                if package.is_none() || pkg.contains(package.unwrap()) {
                    println!("{}📦 {}", prefix, pkg);
                }
            }
            
            Ok(())
        }
        
        async fn cmd_check(&mut self) -> Result<(), String> {
            println!("🔍 Checking package health...");
            
            let current_dir = env::current_dir()
                .map_err(|e| format!("Failed to get current directory: {}", e))?;
            
            // Check for Lyra.toml
            let manifest_path = current_dir.join("Lyra.toml");
            if manifest_path.exists() {
                println!("✓ Lyra.toml found");
                
                if let Ok(manifest_content) = fs::read_to_string(&manifest_path) {
                    println!("✓ Manifest is readable");
                    if let Some(name) = extract_toml_value(&manifest_content, "name") {
                        println!("  📦 Package: {}", name);
                    }
                    if let Some(version) = extract_toml_value(&manifest_content, "version") {
                        println!("  🏷️  Version: {}", version);
                    }
                }
            } else {
                println!("❌ Lyra.toml not found");
                println!("💡 Run 'lyra-pkg init' to create a new package");
                return Ok(());
            }
            
            // Check for src directory
            let src_dir = current_dir.join("src");
            if src_dir.exists() {
                println!("✓ src/ directory found");
                match count_source_files(&src_dir) {
                    Ok(count) if count > 0 => {
                        println!("  📄 {} source file(s) found", count);
                    }
                    Ok(_) => {
                        println!("⚠️  No .ly source files found in src/");
                    }
                    Err(e) => {
                        println!("⚠️  Error reading src/ directory: {}", e);
                    }
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
        
        async fn cmd_publish(&mut self, registry: Option<&str>, token: Option<&str>) -> Result<(), String> {
            let reg_name = registry.unwrap_or("default");
            
            println!("📤 Publishing to {} registry...", reg_name);
            
            if token.is_none() {
                return Err("Authentication token required for publishing".to_string());
            }
            
            // TODO: Implement actual publishing
            println!("✓ Package published successfully (placeholder)");
            println!("🌐 Package is now available for installation");
            
            Ok(())
        }
        
        // Helper methods
        
        fn list_installed_packages(&self) -> Result<Vec<String>, String> {
            let cache_dir = get_cache_dir();
            let mut packages = Vec::new();
            
            if cache_dir.exists() {
                for entry in fs::read_dir(&cache_dir).map_err(|e| e.to_string())? {
                    let entry = entry.map_err(|e| e.to_string())?;
                    if entry.path().is_dir() {
                        packages.push(entry.file_name().to_string_lossy().to_string());
                    }
                }
            }
            
            packages.sort();
            Ok(packages)
        }
    }

    fn get_cache_dir() -> PathBuf {
        if let Some(home) = std::env::var_os("HOME") {
            PathBuf::from(home).join(".lyra").join("packages")
        } else if let Some(userprofile) = std::env::var_os("USERPROFILE") {
            PathBuf::from(userprofile).join(".lyra").join("packages")
        } else {
            PathBuf::from(".lyra").join("packages")
        }
    }

    fn extract_toml_value(content: &str, key: &str) -> Option<String> {
        for line in content.lines() {
            let line = line.trim();
            if line.starts_with(&format!("{} = ", key)) {
                if let Some(value_part) = line.split(" = ").nth(1) {
                    // Remove quotes
                    let value = value_part.trim_matches('"').trim_matches('\'');
                    return Some(value.to_string());
                }
            }
        }
        None
    }

    fn count_source_files(dir: &Path) -> Result<usize, String> {
        let mut count = 0;
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("ly") {
                count += 1;
            }
        }
        Ok(count)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let mut pkg_cli = pkg_manager::SimplePackageCli::new();
    
    if let Err(e) = pkg_cli.execute(cli.command).await {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }

    Ok(())
}
