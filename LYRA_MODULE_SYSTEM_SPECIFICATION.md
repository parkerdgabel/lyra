# Lyra Module System & Packaging Architecture

## Executive Summary

The Lyra Module System is designed to provide a comprehensive package management and namespacing solution that extends the existing StandardLibrary architecture while maintaining compatibility and performance. The system introduces hierarchical namespaces, dependency management, and a package registry for distributing and consuming Lyra packages.

## Current State Analysis

### Existing Architecture Strengths

1. **Unified Function Resolution**: The system already has a sophisticated function resolution mechanism with static dispatch via `CALL_STATIC` instructions
2. **Foreign Object System**: Clean separation between VM core and complex data types (Tensor, Series, Dataset, etc.)
3. **Performance Focus**: 1000x+ speedup achieved through static dispatch over dynamic HashMap lookups
4. **StandardLibrary Registry**: Well-organized function registration system with categories (math, list, string, rules, tensor, table)

### Current Limitations

1. **Flat Namespace**: All functions exist in global scope (e.g., `Sin`, `Length`, `Append`)
2. **No Module System**: Cannot organize related functions into logical groupings
3. **No Package Management**: No way to share, distribute, or manage third-party packages
4. **No Versioning**: Cannot handle multiple versions of the same functionality
5. **No Dependency Resolution**: Cannot express or resolve dependencies between modules

## Module System Architecture

### 1. Namespace Design

#### Hierarchical Namespace Structure
```
std::math::Sin          # Standard library math functions
std::list::Length       # Standard library list operations
std::tensor::Dot        # Standard library tensor operations
user::mylib::MyFunc     # User-defined package functions
thirdparty::ml::Neural  # Third-party ML package
```

#### Namespace Resolution Rules
- **Absolute paths**: `std::math::Sin[x]` - fully qualified
- **Relative imports**: `import std::math::*` then `Sin[x]`
- **Aliased imports**: `import std::math::Sin as Sine` then `Sine[x]`
- **Module imports**: `import std::math` then `math::Sin[x]`

### 2. Module Manifest Format (Lyra.toml)

```toml
[package]
name = "neural-networks"
version = "1.2.3"
description = "Neural network utilities for Lyra"
authors = ["Jane Doe <jane@example.com>"]
license = "MIT"
repository = "https://github.com/username/lyra-neural"
edition = "2024"

[dependencies]
std = ">=0.1.0"
linear-algebra = "^2.0.0"
statistics = { version = "1.0", features = ["advanced"] }

[dev-dependencies]
test-utils = "0.5.0"

[features]
default = ["basic"]
basic = []
advanced = ["gpu-acceleration"]
gpu-acceleration = []

[exports]
# Functions exported by this package
"NeuralNetwork" = "neural::network::create"
"Train" = "neural::training::train"
"Predict" = "neural::inference::predict"

[imports]
# External dependencies
"std::math::*" = { from = "std" }
"MatMul" = { from = "linear-algebra", name = "matrix_multiply" }
```

### 3. Module Structure

#### Core Module Type
```rust
/// A Lyra module containing functions, types, and submodules
#[derive(Debug, Clone)]
pub struct Module {
    /// Module metadata
    pub metadata: ModuleMetadata,
    
    /// Exported functions from this module
    pub exports: HashMap<String, FunctionExport>,
    
    /// Submodules contained within this module
    pub submodules: HashMap<String, Module>,
    
    /// Dependencies required by this module
    pub dependencies: Vec<Dependency>,
    
    /// Module-level configuration
    pub config: ModuleConfig,
}

/// Module metadata
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub name: String,
    pub version: Version,
    pub description: String,
    pub authors: Vec<String>,
    pub license: String,
    pub repository: Option<String>,
}

/// Function export definition
#[derive(Debug, Clone)]
pub struct FunctionExport {
    /// Internal function name
    pub internal_name: String,
    
    /// Exported function name
    pub export_name: String,
    
    /// Function signature and metadata
    pub signature: FunctionSignature,
    
    /// Function implementation
    pub function: FunctionImplementation,
    
    /// Function attributes (Hold, Listable, etc.)
    pub attributes: Vec<FunctionAttribute>,
}

/// Function implementation types
#[derive(Debug, Clone)]
pub enum FunctionImplementation {
    /// Native Rust function
    Native(StdlibFunction),
    
    /// Foreign object method
    Foreign {
        type_name: String,
        method_name: String,
    },
    
    /// Lyra-defined function (compiled bytecode)
    Lyra {
        bytecode: Vec<Instruction>,
        constants: Vec<Value>,
    },
}
```

### 4. Package Registry Architecture

#### Local Registry Structure
```
~/.lyra/packages/
├── index.json              # Package index
├── cache/                  # Downloaded package cache
│   ├── neural-networks-1.2.3/
│   └── linear-algebra-2.0.1/
└── config.toml            # Registry configuration
```

#### Remote Registry API
```rust
/// Package registry interface
pub trait PackageRegistry {
    /// Search for packages matching query
    async fn search(&self, query: &str) -> Result<Vec<PackageInfo>, RegistryError>;
    
    /// Get package metadata
    async fn get_package(&self, name: &str, version: &Version) -> Result<PackageMetadata, RegistryError>;
    
    /// Download package
    async fn download(&self, name: &str, version: &Version) -> Result<PackageBundle, RegistryError>;
    
    /// Publish package
    async fn publish(&self, package: &PackageBundle, token: &str) -> Result<(), RegistryError>;
    
    /// Get package dependencies
    async fn resolve_dependencies(&self, deps: &[Dependency]) -> Result<DependencyGraph, RegistryError>;
}
```

### 5. CLI Interface

#### Package Management Commands
```bash
# Package management
lyra pkg init                          # Initialize new package
lyra pkg build                         # Build current package
lyra pkg test                          # Run package tests
lyra pkg publish                       # Publish to registry
lyra pkg install neural-networks       # Install package
lyra pkg update                        # Update dependencies
lyra pkg remove neural-networks        # Remove package

# Package discovery
lyra pkg search "neural network"       # Search packages
lyra pkg info neural-networks          # Show package info
lyra pkg list                          # List installed packages
lyra pkg outdated                      # Show outdated packages

# Dependency management
lyra pkg add linear-algebra            # Add dependency
lyra pkg add linear-algebra@2.0.1      # Add specific version
lyra pkg tree                          # Show dependency tree
lyra pkg check                         # Check dependency health
```

#### Module Usage in REPL
```lyra
# Import entire module
import std::math
math::Sin[3.14159/2]

# Import specific functions
import std::math::{Sin, Cos, Tan}
Sin[0] + Cos[0]

# Import with alias
import std::math::Sin as Sine
Sine[3.14159/6]

# Wildcard import (use sparingly)
import std::math::*
Sin[0] + Cos[0] + Tan[0]
```

### 6. Integration with Existing Systems

#### StandardLibrary Migration
```rust
/// Extended StandardLibrary with module support
pub struct ModularStandardLibrary {
    /// Traditional flat function registry (backwards compatibility)
    legacy_functions: HashMap<String, StdlibFunction>,
    
    /// Module registry for organized functions
    modules: HashMap<String, Module>,
    
    /// Namespace resolver
    resolver: NamespaceResolver,
    
    /// Function index space for static dispatch
    unified_registry: FunctionRegistry,
}

impl ModularStandardLibrary {
    /// Register a module with the standard library
    pub fn register_module(&mut self, namespace: &str, module: Module) -> Result<(), ModuleError> {
        // Validate namespace
        self.validate_namespace(namespace)?;
        
        // Register all exported functions in unified registry
        for (name, export) in module.exports.iter() {
            let qualified_name = format!("{}::{}", namespace, name);
            let function_index = self.unified_registry.register_function(
                qualified_name.clone(),
                export.function.clone(),
                export.attributes.clone(),
            )?;
            
            // Maintain backwards compatibility for std library functions
            if namespace.starts_with("std::") {
                self.legacy_functions.insert(name.clone(), export.function.as_stdlib());
            }
        }
        
        self.modules.insert(namespace.to_string(), module);
        Ok(())
    }
}
```

#### Compiler Integration
```rust
/// Enhanced compiler with module awareness
impl Compiler {
    /// Compile function call with namespace resolution
    fn compile_function_call(&mut self, namespace: Option<&str>, function: &str, args: &[Expr]) -> CompilerResult<()> {
        // Resolve function in namespace context
        let resolved_function = if let Some(ns) = namespace {
            self.resolver.resolve_qualified(ns, function)?
        } else {
            self.resolver.resolve_unqualified(function)?
        };
        
        // Compile arguments
        for arg in args {
            self.compile_expr(arg)?;
        }
        
        // Generate CALL_STATIC with resolved function index
        let function_index = resolved_function.get_function_index();
        self.context.emit(OpCode::CALL_STATIC, ((function_index as u32) << 8) | (args.len() as u32))?;
        
        Ok(())
    }
}
```

### 7. Version Resolution Algorithm

#### Semantic Versioning Support
- **Patch versions** (1.0.0 -> 1.0.1): Backwards compatible bug fixes
- **Minor versions** (1.0.0 -> 1.1.0): Backwards compatible new features  
- **Major versions** (1.0.0 -> 2.0.0): Breaking changes

#### Dependency Resolution Strategy
```rust
/// Dependency resolution using constraint satisfaction
pub struct DependencyResolver {
    /// Available packages in registry
    available_packages: HashMap<String, Vec<Version>>,
    
    /// Cached resolution results
    resolution_cache: HashMap<DependencySet, ResolutionResult>,
}

impl DependencyResolver {
    /// Resolve dependencies using backtracking with optimization
    pub fn resolve(&self, dependencies: &[Dependency]) -> Result<ResolutionPlan, ResolutionError> {
        // 1. Parse version constraints
        let constraints = self.parse_constraints(dependencies)?;
        
        // 2. Find latest compatible versions
        let candidates = self.find_candidates(&constraints)?;
        
        // 3. Check for conflicts
        let resolved = self.resolve_conflicts(candidates)?;
        
        // 4. Validate transitive dependencies
        let plan = self.validate_transitive_deps(resolved)?;
        
        Ok(plan)
    }
}
```

### 8. Security and Sandboxing

#### Package Verification
```rust
/// Package security verification
pub struct PackageVerifier {
    /// Trusted signing keys
    trusted_keys: Vec<PublicKey>,
    
    /// Package hash verification
    hash_verifier: HashVerifier,
}

impl PackageVerifier {
    /// Verify package authenticity and integrity
    pub fn verify_package(&self, package: &PackageBundle) -> Result<VerificationResult, SecurityError> {
        // 1. Verify cryptographic signature
        self.verify_signature(&package.signature, &package.content)?;
        
        // 2. Check hash integrity
        self.hash_verifier.verify_hash(&package.content, &package.expected_hash)?;
        
        // 3. Scan for malicious content (future)
        self.scan_content(&package.content)?;
        
        Ok(VerificationResult::Trusted)
    }
}
```

#### Capability-Based Permissions
```toml
# Package permissions in Lyra.toml
[permissions]
filesystem = { read = ["./data"], write = ["./output"] }
network = { domains = ["api.example.com"] }
foreign-objects = ["Tensor", "Series"]  # Allowed foreign object types
```

### 9. Migration Strategy

#### Phase 1: Foundation (Weeks 1-2)
1. **Implement Module Core Types**
   - `Module`, `ModuleMetadata`, `FunctionExport` structs
   - Basic namespace parsing and validation
   - Module loading and registration

2. **Extend Existing Registry**
   - Add namespace support to `FunctionRegistry`
   - Maintain backwards compatibility with flat function names
   - Update compiler for qualified function calls

#### Phase 2: Package Management (Weeks 3-4)
1. **CLI Interface**
   - Implement `lyra pkg` commands
   - Package initialization and building
   - Local package registry

2. **Package Format**
   - Lyra.toml parsing and validation
   - Package bundling and extraction
   - Version resolution algorithm

#### Phase 3: Advanced Features (Weeks 5-6)
1. **Remote Registry**
   - HTTP-based package registry
   - Authentication and publishing
   - Dependency resolution with caching

2. **Security & Quality**
   - Package verification and signing
   - Permission system
   - Package testing framework

### 10. Integration Points

#### Type System Integration (Workstream 3)
```rust
/// Module-aware type checking
impl TypeChecker {
    /// Check function call with module context
    fn check_function_call(&mut self, module: &str, function: &str, args: &[TypedExpr]) -> TypeResult<Type> {
        let qualified_name = format!("{}::{}", module, function);
        let function_sig = self.module_registry.get_function_signature(&qualified_name)?;
        
        // Validate argument types against signature
        self.validate_arguments(&function_sig, args)?;
        
        Ok(function_sig.return_type.clone())
    }
}
```

#### Memory Management Integration (Workstream 2)
```rust
/// Module loading with memory management
impl ModuleLoader {
    /// Load module with efficient memory usage
    fn load_module(&self, path: &Path) -> Result<Module, LoadError> {
        // Use memory-mapped files for large modules
        let mapped_file = MemoryMap::new(path)?;
        
        // Parse module with zero-copy where possible
        let module = self.parser.parse_module(&mapped_file)?;
        
        // Register with memory manager for cleanup
        self.memory_manager.register_module(module.clone());
        
        Ok(module)
    }
}
```

#### Serialization Integration (Workstream 6)
```rust
/// Module serialization for caching
impl Serialize for Module {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Efficient binary serialization for fast loading
        let mut state = serializer.serialize_struct("Module", 5)?;
        state.serialize_field("metadata", &self.metadata)?;
        state.serialize_field("exports", &self.exports)?;
        state.serialize_field("submodules", &self.submodules)?;
        state.serialize_field("dependencies", &self.dependencies)?;
        state.serialize_field("config", &self.config)?;
        state.end()
    }
}
```

## Performance Considerations

### Function Resolution Optimization
1. **Compile-Time Resolution**: Resolve all qualified function calls at compile time
2. **Index Space Extension**: Extend unified function index to accommodate modules (0-31: Foreign, 32-1023: Stdlib, 1024+: User modules)
3. **Namespace Caching**: Cache resolved namespace lookups for performance
4. **Lazy Loading**: Load modules only when first accessed

### Memory Efficiency
1. **Shared Function Registry**: Multiple modules can reference the same function implementation
2. **Copy-on-Write Modules**: Share common module data between instances
3. **Compressed Package Storage**: Use compression for package storage and transmission

## Future Extensions

### 1. Hot Module Reloading
```rust
/// Support for hot reloading during development
impl ModuleRegistry {
    /// Reload module and update all references
    pub fn hot_reload(&mut self, module_name: &str) -> Result<(), ReloadError> {
        // Unload existing module
        self.unload_module(module_name)?;
        
        // Reload from disk
        let new_module = self.load_module_from_disk(module_name)?;
        
        // Update all references atomically
        self.update_references(module_name, new_module)?;
        
        Ok(())
    }
}
```

### 2. Module Introspection
```lyra
# Runtime module introspection
ModuleInfo[std::math]                    # Get module metadata
ModuleFunctions[std::math]               # List all functions
ModuleDependencies[neural::networks]     # Show dependencies
```

### 3. Cross-Language Integration
```toml
# Support for modules written in other languages
[foreign-modules]
python = { path = "./python_module.py", interface = "python-bridge" }
julia = { path = "./julia_module.jl", interface = "julia-bridge" }
```

## Testing Strategy

### Unit Tests
- Module loading and parsing
- Namespace resolution
- Dependency resolution algorithms
- Function registration and lookup

### Integration Tests
- End-to-end package installation
- Module import and usage
- CLI command functionality
- Registry interaction

### Performance Tests
- Function resolution benchmarks
- Module loading performance
- Memory usage profiling
- Large dependency graph resolution

## Conclusion

The Lyra Module System provides a comprehensive foundation for organizing, distributing, and managing Lyra packages while maintaining the performance characteristics and architectural principles of the existing system. The design prioritizes backwards compatibility, performance, and developer experience while providing a path for future growth and ecosystem development.

The modular architecture integrates seamlessly with existing workstreams (Type System, Memory Management, Serialization, Concurrency) and provides clear extension points for future enhancements. The implementation strategy ensures a smooth migration path while delivering immediate value to Lyra developers.