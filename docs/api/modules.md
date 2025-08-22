# Module System API Documentation

## Overview

Lyra's module system provides hierarchical organization, dependency management, and controlled access to functionality. This document covers the API for creating modules, registering functions, managing dependencies, and integrating with the VM and standard library.

## Architecture

The module system is built on several core components:

```rust
// Core module structure
pub struct Module {
    pub metadata: ModuleMetadata,           // Version, authors, description
    pub exports: HashMap<String, FunctionExport>,  // Public functions
    pub submodules: HashMap<String, Module>, // Nested modules
    pub dependencies: Vec<Dependency>,       // Required modules
    pub config: ModuleConfig,               // Features and settings
    pub constants: Vec<Value>,              // Module constants
    pub init_code: Option<Vec<Instruction>>, // Initialization code
}

// Function exports
pub struct FunctionExport {
    pub internal_name: String,              // Internal function name
    pub export_name: String,                // Public export name
    pub signature: FunctionSignature,       // Type signature
    pub implementation: FunctionImplementation, // Function body
    pub attributes: Vec<FunctionAttribute>, // Function properties
    pub documentation: Option<String>,      // Documentation string
}
```

## Creating Modules

### Basic Module Creation

```rust
use crate::modules::{Module, ModuleMetadata, Version};

// Create module metadata
let metadata = ModuleMetadata {
    name: "MyModule".to_string(),
    version: Version::new(1, 0, 0),
    description: "A sample Lyra module".to_string(),
    authors: vec!["Developer Name".to_string()],
    license: "MIT".to_string(),
    repository: Some("https://github.com/user/mymodule".to_string()),
    homepage: None,
    documentation: None,
    keywords: vec!["math".to_string(), "computation".to_string()],
    categories: vec!["mathematics".to_string()],
};

// Create the module
let mut module = Module::new(metadata);
```

### Adding Function Exports

```rust
use crate::modules::{FunctionExport, FunctionImplementation, stdlib_to_export};
use crate::stdlib::StdlibFunction;
use crate::linker::FunctionAttribute;

// Define a custom function
pub fn my_square_function(args: &[Value]) -> VmResult<Value> {
    match args {
        [Value::Integer(n)] => Ok(Value::Integer(n * n)),
        [Value::Real(f)] => Ok(Value::Real(f * f)),
        _ => Err(VmError::TypeError {
            expected: "single numeric argument".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

// Create function export
let square_export = stdlib_to_export_with_docs(
    "Square",
    my_square_function,
    vec![FunctionAttribute::Listable, FunctionAttribute::NumericFunction],
    "Computes the square of a number: Square[x] returns x^2"
);

// Add to module
module.add_export(square_export)?;
```

### Advanced Function Implementations

```rust
use crate::modules::{FunctionImplementation, CallingConvention};

// Native Rust function (most common)
let native_impl = FunctionImplementation::Native(my_custom_function);

// Foreign object method call
let foreign_impl = FunctionImplementation::Foreign {
    type_name: "TimeSeries".to_string(),
    method_name: "resample".to_string(),
};

// Lyra-defined function (compiled from Lyra code)
let lyra_impl = FunctionImplementation::Lyra {
    bytecode: vec![/* compiled instructions */],
    constants: vec![Value::Integer(42)],
    locals_count: 2,
};

// External C function
let external_impl = FunctionImplementation::External {
    library_path: "/usr/lib/libmath.so".to_string(),
    symbol_name: "pow".to_string(),
    calling_convention: CallingConvention::C,
};

// Create export with custom implementation
let custom_export = FunctionExport {
    internal_name: "internal_pow".to_string(),
    export_name: "Power".to_string(),
    signature: FunctionSignature::new("Math", "Power", 2),
    implementation: external_impl,
    attributes: vec![FunctionAttribute::NumericFunction],
    documentation: Some("External power function".to_string()),
};
```

## Submodules and Hierarchical Organization

### Creating Submodules

```rust
// Create a submodule for linear algebra functions
let linalg_metadata = ModuleMetadata {
    name: "LinearAlgebra".to_string(),
    version: Version::new(1, 0, 0),
    description: "Linear algebra operations".to_string(),
    authors: vec!["Math Team".to_string()],
    license: "MIT".to_string(),
    repository: None,
    homepage: None,
    documentation: None,
    keywords: vec!["linear algebra".to_string(), "matrices".to_string()],
    categories: vec!["mathematics".to_string()],
};

let mut linalg_module = Module::new(linalg_metadata);

// Add functions to submodule
let dot_product_export = stdlib_to_export(
    "Dot",
    vector_dot_product,
    vec![FunctionAttribute::NumericFunction]
);
linalg_module.add_export(dot_product_export)?;

// Add submodule to main module
module.add_submodule("LinearAlgebra".to_string(), linalg_module);
```

### Nested Module Access

```rust
// Access nested modules
if let Some(linalg) = module.get_submodule("LinearAlgebra") {
    if let Some(dot_fn) = linalg.get_export("Dot") {
        println!("Found Dot function: {}", dot_fn.export_name);
    }
}

// Get all submodules
let submodule_names = module.submodule_names();
for name in submodule_names {
    println!("Submodule: {}", name);
}
```

## Dependencies and Version Management

### Defining Dependencies

```rust
use crate::modules::{Dependency, VersionConstraint, Version};

// Exact version dependency
let exact_dep = Dependency {
    name: "Mathematics".to_string(),
    version: VersionConstraint::Exact(Version::new(2, 1, 0)),
    features: vec![],
    optional: false,
};

// Compatible version (^1.2.0 - same major version)
let compat_dep = Dependency {
    name: "Statistics".to_string(),
    version: VersionConstraint::Compatible(Version::new(1, 2, 0)),
    features: vec!["advanced".to_string()],
    optional: false,
};

// Version range
let range_dep = Dependency {
    name: "DataProcessing".to_string(),
    version: VersionConstraint::Range {
        min: Version::new(1, 0, 0),
        max: Version::new(2, 0, 0),
    },
    features: vec![],
    optional: true,
};

// Add dependencies to module
module.dependencies = vec![exact_dep, compat_dep, range_dep];
```

### Version Constraint Checking

```rust
// Check if a version satisfies constraints
let current_version = Version::new(1, 2, 5);

let constraints = vec![
    VersionConstraint::Compatible(Version::new(1, 2, 0)),
    VersionConstraint::GreaterEqual(Version::new(1, 0, 0)),
];

for constraint in constraints {
    if current_version.satisfies(&constraint) {
        println!("Version {} satisfies {:?}", current_version, constraint);
    }
}
```

## Module Configuration

### Feature Flags

```rust
use crate::modules::ModuleConfig;
use std::collections::HashMap;

// Create module configuration
let mut config = ModuleConfig {
    features: vec!["gpu".to_string(), "parallel".to_string()],
    settings: HashMap::new(),
    hot_reload: true,
    memory_limit: Some(1024 * 1024 * 100), // 100MB
};

// Add settings
config.settings.insert("precision".to_string(), "double".to_string());
config.settings.insert("thread_count".to_string(), "4".to_string());

// Apply to module
module.config = config;

// Check features at runtime
fn optimized_computation(args: &[Value]) -> VmResult<Value> {
    if module.config.features.contains(&"gpu".to_string()) {
        gpu_computation(args)
    } else if module.config.features.contains(&"parallel".to_string()) {
        parallel_computation(args)
    } else {
        sequential_computation(args)
    }
}
```

### Conditional Compilation

```rust
// Feature-gated function exports
fn register_gpu_functions(module: &mut Module) -> Result<(), ModuleError> {
    if module.config.features.contains(&"gpu".to_string()) {
        let gpu_export = stdlib_to_export(
            "GPUMultiply",
            gpu_matrix_multiply,
            vec![FunctionAttribute::NumericFunction]
        );
        module.add_export(gpu_export)?;
    }
    Ok(())
}
```

## Module Registry and Loading

### Module Registry

```rust
use crate::modules::registry::ModuleRegistry;

// Create a module registry
let mut registry = ModuleRegistry::new();

// Register a module
registry.register_module("MyModule".to_string(), module)?;

// Load a module by name
if let Some(loaded_module) = registry.get_module("MyModule") {
    println!("Loaded module: {}", loaded_module.metadata.name);
}

// List all registered modules
let module_names = registry.list_modules();
for name in module_names {
    println!("Available module: {}", name);
}
```

### Module Resolution

```rust
use crate::modules::resolver::ModuleResolver;

// Create resolver with search paths
let resolver = ModuleResolver::new(vec![
    "/usr/local/lib/lyra/modules".to_string(),
    "./modules".to_string(),
    "~/.lyra/modules".to_string(),
]);

// Resolve module path
let module_path = resolver.resolve_module("Mathematics")?;
println!("Found module at: {}", module_path);

// Load module from path
let module = resolver.load_module(&module_path)?;
```

### Dynamic Module Loading

```rust
use crate::modules::loader::ModuleLoader;

// Create module loader
let loader = ModuleLoader::new();

// Load module from file
let module = loader.load_from_file("/path/to/module.lyra")?;

// Load module from string
let module_source = r#"
    (* Module: SimpleUtils *)
    Version: 1.0.0
    Authors: ["Developer"]
    Description: "Simple utility functions"
    
    Square[x_] := x^2
    Double[x_] := 2*x
"#;

let module = loader.load_from_string(module_source)?;
```

## Stdlib Function Registration

### Bulk Registration

```rust
use crate::stdlib::StdlibFunction;

// Define stdlib functions for a module
pub fn register_math_functions() -> Vec<(&'static str, StdlibFunction)> {
    vec![
        ("Sin", stdlib_sin),
        ("Cos", stdlib_cos),
        ("Tan", stdlib_tan),
        ("Log", stdlib_log),
        ("Exp", stdlib_exp),
        ("Sqrt", stdlib_sqrt),
    ]
}

// Register all functions in a module
pub fn create_math_module() -> Result<Module, ModuleError> {
    let metadata = ModuleMetadata {
        name: "Mathematics".to_string(),
        version: Version::new(1, 0, 0),
        description: "Core mathematical functions".to_string(),
        authors: vec!["Lyra Team".to_string()],
        license: "MIT".to_string(),
        repository: None,
        homepage: None,
        documentation: None,
        keywords: vec!["math".to_string()],
        categories: vec!["mathematics".to_string()],
    };
    
    let mut module = Module::new(metadata);
    
    // Register all math functions
    for (name, function) in register_math_functions() {
        let export = stdlib_to_export_with_docs(
            name,
            function,
            vec![FunctionAttribute::NumericFunction, FunctionAttribute::Listable],
            &format!("Mathematical function: {}", name)
        );
        module.add_export(export)?;
    }
    
    Ok(module)
}
```

### Type-Aware Function Registration

```rust
use crate::types::{LyraType, FunctionAttribute};
use crate::linker::FunctionSignature;

// Register with explicit type signatures
pub fn register_typed_function(
    module: &mut Module,
    name: &str,
    function: StdlibFunction,
    input_types: Vec<LyraType>,
    output_type: LyraType,
    attributes: Vec<FunctionAttribute>,
) -> Result<(), ModuleError> {
    
    let signature = FunctionSignature {
        module_name: module.metadata.name.clone(),
        function_name: name.to_string(),
        arity: input_types.len(),
        input_types: Some(input_types),
        output_type: Some(output_type),
    };
    
    let export = FunctionExport {
        internal_name: name.to_string(),
        export_name: name.to_string(),
        signature,
        implementation: FunctionImplementation::Native(function),
        attributes,
        documentation: None,
    };
    
    module.add_export(export)
}

// Usage
register_typed_function(
    &mut module,
    "Plus",
    stdlib_plus,
    vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
    LyraType::TypeVar(0),
    vec![FunctionAttribute::Associative, FunctionAttribute::Commutative]
)?;
```

## Module Initialization

### Initialization Code

```rust
use crate::bytecode::{Instruction, OpCode};

// Create initialization code
let init_instructions = vec![
    Instruction::new(OpCode::LoadConstant(0)),  // Load pi constant
    Instruction::new(OpCode::StoreGlobal("Pi".to_string())),
    Instruction::new(OpCode::LoadConstant(1)),  // Load e constant  
    Instruction::new(OpCode::StoreGlobal("E".to_string())),
];

// Add constants
module.constants = vec![
    Value::Real(std::f64::consts::PI),
    Value::Real(std::f64::consts::E),
];

// Set initialization code
module.init_code = Some(init_instructions);
```

### Module Initialization Hook

```rust
impl Module {
    /// Run module initialization code
    pub fn initialize(&self, vm: &mut Vm) -> VmResult<()> {
        if let Some(ref init_code) = self.init_code {
            // Set up constants
            vm.add_constants(&self.constants);
            
            // Execute initialization
            vm.execute(init_code)?;
        }
        Ok(())
    }
    
    /// Post-initialization setup
    pub fn post_init(&self, registry: &mut ModuleRegistry) -> Result<(), ModuleError> {
        // Register global symbols, set up cross-module references, etc.
        for (name, export) in &self.exports {
            registry.register_global_function(name.clone(), export.clone());
        }
        Ok(())
    }
}
```

## Cross-Module Communication

### Module Dependencies

```rust
// Resolve and validate dependencies
pub fn resolve_dependencies(
    module: &Module,
    registry: &ModuleRegistry
) -> Result<Vec<Module>, ModuleError> {
    let mut resolved = Vec::new();
    
    for dependency in &module.dependencies {
        if let Some(dep_module) = registry.get_module(&dependency.name) {
            // Check version compatibility
            if !dep_module.metadata.version.satisfies(&dependency.version) {
                return Err(ModuleError::VersionConflict {
                    package: dependency.name.clone(),
                    required: format!("{:?}", dependency.version),
                    found: dep_module.metadata.version.to_string(),
                });
            }
            resolved.push(dep_module.clone());
        } else if !dependency.optional {
            return Err(ModuleError::ModuleNotFound {
                name: dependency.name.clone(),
            });
        }
    }
    
    Ok(resolved)
}
```

### Function Imports

```rust
// Import functions from dependencies
pub fn import_functions(
    module: &Module,
    dependencies: &[Module]
) -> HashMap<String, FunctionExport> {
    let mut imports = HashMap::new();
    
    for dep in dependencies {
        for (name, export) in &dep.exports {
            // Qualified import: ModuleName.FunctionName
            let qualified_name = format!("{}.{}", dep.metadata.name, name);
            imports.insert(qualified_name, export.clone());
        }
    }
    
    imports
}

// Selective imports
pub fn selective_import(
    from_module: &str,
    functions: &[&str],
    registry: &ModuleRegistry
) -> Result<HashMap<String, FunctionExport>, ModuleError> {
    let module = registry.get_module(from_module)
        .ok_or_else(|| ModuleError::ModuleNotFound { name: from_module.to_string() })?;
    
    let mut imports = HashMap::new();
    
    for &func_name in functions {
        if let Some(export) = module.get_export(func_name) {
            imports.insert(func_name.to_string(), export.clone());
        } else {
            return Err(ModuleError::ModuleNotFound { 
                name: format!("{}::{}", from_module, func_name) 
            });
        }
    }
    
    Ok(imports)
}
```

## Hot Reloading

### Module Watching

```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

pub struct ModuleWatcher {
    watched_modules: HashMap<String, String>,  // name -> path
    reload_sender: mpsc::Sender<String>,
}

impl ModuleWatcher {
    pub fn new() -> (Self, mpsc::Receiver<String>) {
        let (tx, rx) = mpsc::channel();
        (ModuleWatcher {
            watched_modules: HashMap::new(),
            reload_sender: tx,
        }, rx)
    }
    
    pub fn watch_module(&mut self, name: String, path: String) {
        self.watched_modules.insert(name.clone(), path.clone());
        
        let sender = self.reload_sender.clone();
        thread::spawn(move || {
            let mut last_modified = std::fs::metadata(&path)
                .and_then(|m| m.modified())
                .unwrap_or_else(|_| std::time::SystemTime::UNIX_EPOCH);
            
            loop {
                thread::sleep(Duration::from_secs(1));
                
                if let Ok(metadata) = std::fs::metadata(&path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > last_modified {
                            let _ = sender.send(name.clone());
                            last_modified = modified;
                        }
                    }
                }
            }
        });
    }
}
```

### Hot Reload Implementation

```rust
pub struct HotReloadManager {
    registry: ModuleRegistry,
    loader: ModuleLoader,
    watcher: ModuleWatcher,
    reload_receiver: mpsc::Receiver<String>,
}

impl HotReloadManager {
    pub fn new() -> Self {
        let (watcher, receiver) = ModuleWatcher::new();
        HotReloadManager {
            registry: ModuleRegistry::new(),
            loader: ModuleLoader::new(),
            watcher,
            reload_receiver: receiver,
        }
    }
    
    pub fn process_reloads(&mut self) -> Result<Vec<String>, ModuleError> {
        let mut reloaded = Vec::new();
        
        while let Ok(module_name) = self.reload_receiver.try_recv() {
            if let Some(old_module) = self.registry.get_module(&module_name) {
                if old_module.config.hot_reload {
                    self.reload_module(&module_name)?;
                    reloaded.push(module_name);
                }
            }
        }
        
        Ok(reloaded)
    }
    
    fn reload_module(&mut self, name: &str) -> Result<(), ModuleError> {
        // Find module path
        if let Some(path) = self.watcher.watched_modules.get(name) {
            // Load new version
            let new_module = self.loader.load_from_file(path)?;
            
            // Validate compatibility
            self.validate_reload_compatibility(name, &new_module)?;
            
            // Replace in registry
            self.registry.register_module(name.to_string(), new_module)?;
            
            println!("Hot reloaded module: {}", name);
        }
        
        Ok(())
    }
    
    fn validate_reload_compatibility(
        &self,
        name: &str,
        new_module: &Module
    ) -> Result<(), ModuleError> {
        if let Some(old_module) = self.registry.get_module(name) {
            // Check that public API is preserved
            for export_name in old_module.exported_functions() {
                if !new_module.has_export(export_name) {
                    return Err(ModuleError::PackageError {
                        message: format!(
                            "Hot reload failed: function {} removed from module {}",
                            export_name, name
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}
```

## Module Validation and Security

### Module Validation

```rust
impl Module {
    /// Validate module structure and consistency
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
        
        // Validate function signatures
        for (name, export) in &self.exports {
            if export.export_name != *name {
                return Err(ModuleError::PackageError {
                    message: format!(
                        "Export name mismatch: key '{}' vs export '{}'",
                        name, export.export_name
                    ),
                });
            }
        }
        
        // Recursively validate submodules
        for submodule in self.submodules.values() {
            submodule.validate()?;
        }
        
        Ok(())
    }
    
    /// Check for circular dependencies
    pub fn check_circular_dependencies(
        &self,
        registry: &ModuleRegistry,
        visited: &mut HashSet<String>
    ) -> Result<(), ModuleError> {
        if visited.contains(&self.metadata.name) {
            return Err(ModuleError::CircularDependency {
                cycle: format!("Circular dependency involving {}", self.metadata.name),
            });
        }
        
        visited.insert(self.metadata.name.clone());
        
        for dependency in &self.dependencies {
            if let Some(dep_module) = registry.get_module(&dependency.name) {
                dep_module.check_circular_dependencies(registry, visited)?;
            }
        }
        
        visited.remove(&self.metadata.name);
        Ok(())
    }
}
```

### Security Validation

```rust
use crate::security::SecurityPolicy;

pub fn validate_module_security(
    module: &Module,
    policy: &SecurityPolicy
) -> Result<(), ModuleError> {
    // Check module origin
    if let Some(repo) = &module.metadata.repository {
        if !policy.is_trusted_repository(repo) {
            return Err(ModuleError::SecurityViolation {
                message: format!("Untrusted repository: {}", repo),
            });
        }
    }
    
    // Check function implementations
    for export in module.exports.values() {
        match &export.implementation {
            FunctionImplementation::External { library_path, .. } => {
                if !policy.is_allowed_library(library_path) {
                    return Err(ModuleError::SecurityViolation {
                        message: format!("Unsafe library: {}", library_path),
                    });
                }
            }
            FunctionImplementation::Lyra { bytecode, .. } => {
                // Validate bytecode doesn't contain unsafe operations
                validate_bytecode_safety(bytecode, policy)?;
            }
            _ => {} // Native and Foreign implementations are trusted
        }
    }
    
    Ok(())
}
```

## Testing Modules

### Unit Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_module_creation() {
        let metadata = ModuleMetadata {
            name: "TestModule".to_string(),
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
        assert_eq!(module.metadata.name, "TestModule");
        assert!(module.exports.is_empty());
        assert!(module.submodules.is_empty());
    }
    
    #[test]
    fn test_function_export() {
        let mut module = create_test_module();
        
        let export = stdlib_to_export(
            "TestFunction",
            test_function,
            vec![FunctionAttribute::Pure]
        );
        
        assert!(module.add_export(export).is_ok());
        assert!(module.has_export("TestFunction"));
    }
    
    #[test]
    fn test_dependency_resolution() {
        let mut registry = ModuleRegistry::new();
        let base_module = create_base_module();
        let dependent_module = create_dependent_module();
        
        registry.register_module("Base".to_string(), base_module).unwrap();
        registry.register_module("Dependent".to_string(), dependent_module).unwrap();
        
        let deps = resolve_dependencies(
            registry.get_module("Dependent").unwrap(),
            &registry
        ).unwrap();
        
        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].metadata.name, "Base");
    }
}
```

### Integration Testing

```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_module_loading_and_execution() {
        let mut vm = Vm::new();
        let mut registry = ModuleRegistry::new();
        
        // Create and register module
        let module = create_math_module().unwrap();
        registry.register_module("Math".to_string(), module).unwrap();
        
        // Initialize module in VM
        let math_module = registry.get_module("Math").unwrap();
        math_module.initialize(&mut vm).unwrap();
        
        // Test function execution
        let result = vm.call_function("Sin", &[Value::Real(0.0)]).unwrap();
        assert_eq!(result, Value::Real(0.0));
    }
    
    #[test]
    fn test_hot_reload() {
        let mut manager = HotReloadManager::new();
        
        // Create temporary module file
        let temp_file = "/tmp/test_module.lyra";
        std::fs::write(temp_file, "TestFunc[x_] := x + 1").unwrap();
        
        // Load and watch module
        let module = manager.loader.load_from_file(temp_file).unwrap();
        manager.registry.register_module("Test".to_string(), module).unwrap();
        manager.watcher.watch_module("Test".to_string(), temp_file.to_string());
        
        // Modify file
        std::fs::write(temp_file, "TestFunc[x_] := x + 2").unwrap();
        
        // Process reloads
        std::thread::sleep(Duration::from_secs(2));
        let reloaded = manager.process_reloads().unwrap();
        assert!(reloaded.contains(&"Test".to_string()));
        
        // Clean up
        std::fs::remove_file(temp_file).unwrap();
    }
}
```

## Best Practices

### Module Design Principles

1. **Single Responsibility**: Each module should have a clear, focused purpose
2. **Stable APIs**: Minimize breaking changes in public interfaces
3. **Clear Dependencies**: Explicitly declare all dependencies
4. **Documentation**: Provide comprehensive documentation for all exports
5. **Versioning**: Use semantic versioning for module releases

### Performance Optimization

```rust
// Lazy loading for large modules
pub struct LazyModule {
    metadata: ModuleMetadata,
    loader: Box<dyn Fn() -> Result<Module, ModuleError>>,
    cached: Option<Module>,
}

impl LazyModule {
    pub fn get_or_load(&mut self) -> Result<&Module, ModuleError> {
        if self.cached.is_none() {
            self.cached = Some((self.loader)()?);
        }
        Ok(self.cached.as_ref().unwrap())
    }
}
```

### Error Handling

```rust
// Graceful error handling in module operations
pub fn safe_module_operation<F, T>(
    module: &Module,
    operation: F
) -> Result<T, ModuleError>
where
    F: FnOnce(&Module) -> Result<T, ModuleError>,
{
    match operation(module) {
        Ok(result) => Ok(result),
        Err(e) => {
            eprintln!("Module operation failed in '{}': {}", module.metadata.name, e);
            Err(e)
        }
    }
}
```

## Migration Guide

### Upgrading Module Versions

1. **Update metadata** with new version number
2. **Validate API compatibility** for breaking changes  
3. **Update dependencies** to compatible versions
4. **Test thoroughly** with existing dependent modules
5. **Document changes** in changelog

### Converting Stdlib to Modules

1. **Group related functions** into logical modules
2. **Define clear module boundaries**
3. **Extract dependencies** between function groups
4. **Create module metadata** and documentation
5. **Migrate registration code** to use module system

## Conclusion

Lyra's module system provides a robust foundation for organizing code, managing dependencies, and building scalable applications. The hierarchical structure, version management, and hot-reload capabilities enable developers to build complex systems while maintaining modularity and maintainability.

The tight integration with the VM, stdlib, and type system ensures that modules work seamlessly with all aspects of the Lyra environment, providing a unified development experience for both library authors and application developers.