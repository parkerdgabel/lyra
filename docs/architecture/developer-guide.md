# Developer Guide

## Overview

This guide provides comprehensive information for developers working on the Lyra symbolic computation engine. It covers architectural patterns, implementation guidelines, testing strategies, and best practices for extending the system.

## Architecture Patterns

### Foreign Object Pattern

The Foreign Object Pattern is the primary extension mechanism for adding complex functionality to Lyra without polluting the VM core.

#### Implementing a Foreign Object

**Step 1: Define Your Type**
```rust
use crate::vm::Value;
use crate::foreign::{Foreign, ForeignError, LyObj};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct MyCustomType {
    data: Vec<f64>,
    metadata: std::collections::HashMap<String, String>,
}

impl MyCustomType {
    pub fn new(data: Vec<f64>) -> Self {
        Self {
            data,
            metadata: std::collections::HashMap::new(),
        }
    }
    
    pub fn size(&self) -> usize {
        self.data.len()
    }
    
    pub fn get(&self, index: usize) -> Option<f64> {
        self.data.get(index).copied()
    }
    
    pub fn set_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}
```

**Step 2: Implement the Foreign Trait**
```rust
impl Foreign for MyCustomType {
    fn type_name(&self) -> &'static str {
        "MyCustomType"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "size" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.size() as i64))
            }
            
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let index = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.get(index) {
                    Some(value) => Ok(Value::Real(value)),
                    None => Err(ForeignError::IndexOutOfBounds {
                        index: index.to_string(),
                        bounds: format!("0..{}", self.size()),
                    }),
                }
            }
            
            "setMetadata" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let value = match &args[1] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[1]),
                    }),
                };
                
                // Note: This creates a new instance since Foreign trait methods are immutable
                let mut new_instance = self.clone();
                new_instance.set_metadata(key, value);
                Ok(Value::LyObj(LyObj::new(Box::new(new_instance))))
            }
            
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
    
    // Optional: Implement serialization
    fn serialize(&self) -> Result<Vec<u8>, ForeignError> {
        serde_json::to_vec(self)
            .map_err(|e| ForeignError::RuntimeError {
                message: format!("Serialization failed: {}", e),
            })
    }
}
```

**Step 3: Create Stdlib Constructor Function**
```rust
use crate::vm::{VmResult, VmError};

pub fn create_my_custom_type(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime(
            "MyCustomType requires exactly one argument".to_string()
        ));
    }
    
    let data = match &args[0] {
        Value::List(list) => {
            let mut vec_data = Vec::new();
            for item in list {
                match item {
                    Value::Real(r) => vec_data.push(*r),
                    Value::Integer(i) => vec_data.push(*i as f64),
                    _ => return Err(VmError::TypeError {
                        expected: "Real or Integer".to_string(),
                        actual: format!("{:?}", item),
                    }),
                }
            }
            vec_data
        }
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let custom_type = MyCustomType::new(data);
    Ok(Value::LyObj(LyObj::new(Box::new(custom_type))))
}
```

**Step 4: Register in Standard Library**
```rust
use crate::stdlib::StandardLibrary;

impl StandardLibrary {
    pub fn register_my_custom_type(&mut self) {
        self.register_function("MyCustomType", create_my_custom_type);
    }
}
```

#### Advanced Foreign Object Patterns

**Thread-Safe Foreign Objects:**
```rust
use std::sync::{Arc, Mutex};

#[derive(Debug)]
pub struct ThreadSafeCounter {
    value: Arc<Mutex<i64>>,
}

impl ThreadSafeCounter {
    pub fn new(initial: i64) -> Self {
        Self {
            value: Arc::new(Mutex::new(initial)),
        }
    }
}

impl Foreign for ThreadSafeCounter {
    fn type_name(&self) -> &'static str { "ThreadSafeCounter" }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "increment" => {
                let mut value = self.value.lock().unwrap();
                *value += 1;
                Ok(Value::Integer(*value))
            }
            
            "get" => {
                let value = self.value.lock().unwrap();
                Ok(Value::Integer(*value))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(ThreadSafeCounter {
            value: Arc::clone(&self.value),
        })
    }
    
    fn as_any(&self) -> &dyn Any { self }
}
```

**Resource Management in Foreign Objects:**
```rust
use std::sync::Arc;

#[derive(Debug)]
pub struct FileHandle {
    path: String,
    handle: Arc<Mutex<Option<std::fs::File>>>,
}

impl Drop for FileHandle {
    fn drop(&mut self) {
        // Ensure file is closed when object is dropped
        let mut handle = self.handle.lock().unwrap();
        if let Some(file) = handle.take() {
            drop(file);  // Explicit close
            println!("File {} closed", self.path);
        }
    }
}

impl Foreign for FileHandle {
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "close" => {
                let mut handle = self.handle.lock().unwrap();
                if handle.take().is_some() {
                    Ok(Value::Boolean(true))
                } else {
                    Ok(Value::Boolean(false))  // Already closed
                }
            }
            
            "isOpen" => {
                let handle = self.handle.lock().unwrap();
                Ok(Value::Boolean(handle.is_some()))
            }
            
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
    
    // ... other trait methods
}
```

### Static Dispatch Pattern

For high-performance functions that are called frequently, implement static dispatch:

**Step 1: Implement Static Function**
```rust
use crate::vm::{Value, VmResult, VmError};

pub fn fast_add_static(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("Add requires exactly 2 arguments".to_string()));
    }
    
    match (&args[0], &args[1]) {
        (Value::Integer(a), Value::Integer(b)) => {
            // Handle integer overflow
            match a.checked_add(*b) {
                Some(result) => Ok(Value::Integer(result)),
                None => Ok(Value::Real(*a as f64 + *b as f64)),
            }
        }
        (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
        (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(*a as f64 + b)),
        (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a + *b as f64)),
        _ => Err(VmError::TypeError {
            expected: "Number".to_string(),
            actual: format!("({:?}, {:?})", args[0], args[1]),
        }),
    }
}
```

**Step 2: Register Static Function**
```rust
use crate::linker::registry::FunctionRegistry;

impl FunctionRegistry {
    pub fn register_fast_arithmetic(&mut self) {
        self.register_static("FastAdd", fast_add_static);
        self.register_static("+", fast_add_static);  // Operator alias
    }
}
```

### Memory Management Patterns

#### Using Memory Arenas

**Temporary Computation Scope:**
```rust
use crate::memory::{ComputationArena, ScopeId};

pub fn complex_computation(args: &[Value]) -> VmResult<Value> {
    let arena = ComputationArena::current();
    let scope = arena.create_scope();
    
    // All allocations in this scope will be freed automatically
    let temp_results = arena.alloc_vec_in_scope(&scope);
    
    for arg in args {
        let processed = expensive_processing(arg)?;
        temp_results.push(processed);
    }
    
    let final_result = combine_results(&temp_results)?;
    
    // Scope automatically freed here
    Ok(final_result)
}
```

#### Symbol Interning Usage

**Efficient Symbol Operations:**
```rust
use crate::memory::{StringInterner, SymbolId};

pub struct SymbolicExpression {
    operator: SymbolId,
    operands: Vec<SymbolId>,
    interner: Arc<StringInterner>,
}

impl SymbolicExpression {
    pub fn new(operator: &str, operands: &[&str], interner: Arc<StringInterner>) -> Self {
        Self {
            operator: interner.intern_symbol_id(operator),
            operands: operands.iter()
                .map(|s| interner.intern_symbol_id(s))
                .collect(),
            interner,
        }
    }
    
    pub fn operator_name(&self) -> Option<String> {
        self.interner.resolve_symbol(self.operator)
    }
    
    // O(1) symbol comparison
    pub fn is_operator(&self, name: &str) -> bool {
        let name_id = self.interner.intern_symbol_id(name);
        self.operator == name_id
    }
}
```

## Testing Strategies

### Unit Testing

**Testing Foreign Objects:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;
    
    #[test]
    fn test_my_custom_type_creation() {
        let data = vec![1.0, 2.0, 3.0];
        let custom_type = MyCustomType::new(data.clone());
        
        assert_eq!(custom_type.size(), 3);
        assert_eq!(custom_type.get(0), Some(1.0));
        assert_eq!(custom_type.get(1), Some(2.0));
        assert_eq!(custom_type.get(2), Some(3.0));
        assert_eq!(custom_type.get(3), None);
    }
    
    #[test]
    fn test_foreign_object_methods() {
        let custom_type = MyCustomType::new(vec![1.0, 2.0, 3.0]);
        let ly_obj = LyObj::new(Box::new(custom_type));
        
        // Test size method
        let size_result = ly_obj.call_method("size", &[]).unwrap();
        assert_eq!(size_result, Value::Integer(3));
        
        // Test get method
        let get_result = ly_obj.call_method("get", &[Value::Integer(1)]).unwrap();
        assert_eq!(get_result, Value::Real(2.0));
        
        // Test error handling
        let error_result = ly_obj.call_method("get", &[Value::Integer(10)]);
        assert!(error_result.is_err());
    }
    
    #[test]
    fn test_method_error_handling() {
        let custom_type = MyCustomType::new(vec![1.0]);
        let ly_obj = LyObj::new(Box::new(custom_type));
        
        // Test invalid method
        match ly_obj.call_method("nonexistent", &[]) {
            Err(ForeignError::UnknownMethod { .. }) => (),
            _ => panic!("Expected UnknownMethod error"),
        }
        
        // Test invalid arity
        match ly_obj.call_method("size", &[Value::Integer(1)]) {
            Err(ForeignError::InvalidArity { .. }) => (),
            _ => panic!("Expected InvalidArity error"),
        }
        
        // Test invalid argument type
        match ly_obj.call_method("get", &[Value::String("invalid".to_string())]) {
            Err(ForeignError::InvalidArgumentType { .. }) => (),
            _ => panic!("Expected InvalidArgumentType error"),
        }
    }
}
```

### Integration Testing

**End-to-End Testing:**
```rust
#[cfg(test)]
mod integration_tests {
    use super::*;
    use crate::vm::VirtualMachine;
    use crate::compiler::Compiler;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    
    #[test]
    fn test_custom_type_integration() {
        let mut vm = VirtualMachine::new();
        let mut compiler = Compiler::new();
        
        // Register custom type
        vm.stdlib.register_my_custom_type();
        
        // Test creation and usage
        let source = r#"
            data = MyCustomType[{1.0, 2.0, 3.0}]
            size = data.size()
            value = data.get(1)
        "#;
        
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize().unwrap();
        let mut parser = Parser::new(tokens);
        let expressions = parser.parse_program().unwrap();
        
        for expr in expressions {
            compiler.compile_expr(&expr).unwrap();
        }
        
        vm.load(compiler.context.code, compiler.context.constants);
        let result = vm.run().unwrap();
        
        // Verify results through variable lookup
        assert_eq!(vm.get_variable("size"), Some(Value::Integer(3)));
        assert_eq!(vm.get_variable("value"), Some(Value::Real(2.0)));
    }
}
```

### Property-Based Testing

**Using QuickCheck for Robust Testing:**
```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use quickcheck::{quickcheck, TestResult};
    
    #[quickcheck]
    fn prop_custom_type_size_consistency(data: Vec<f64>) -> bool {
        let custom_type = MyCustomType::new(data.clone());
        custom_type.size() == data.len()
    }
    
    #[quickcheck]
    fn prop_get_method_bounds_safety(data: Vec<f64>, index: usize) -> TestResult {
        if data.is_empty() {
            return TestResult::discard();
        }
        
        let custom_type = MyCustomType::new(data.clone());
        let ly_obj = LyObj::new(Box::new(custom_type));
        
        let result = ly_obj.call_method("get", &[Value::Integer(index as i64)]);
        
        if index < data.len() {
            // Should succeed and return correct value
            match result {
                Ok(Value::Real(value)) => TestResult::from_bool(value == data[index]),
                _ => TestResult::failed(),
            }
        } else {
            // Should fail with bounds error
            match result {
                Err(ForeignError::IndexOutOfBounds { .. }) => TestResult::passed(),
                _ => TestResult::failed(),
            }
        }
    }
}
```

### Concurrency Testing

**Testing Thread Safety:**
```rust
#[cfg(test)]
mod concurrency_tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    
    #[test]
    fn test_thread_safe_counter() {
        let counter = ThreadSafeCounter::new(0);
        let ly_obj = Arc::new(LyObj::new(Box::new(counter)));
        
        let handles: Vec<_> = (0..10).map(|_| {
            let obj = Arc::clone(&ly_obj);
            thread::spawn(move || {
                for _ in 0..100 {
                    obj.call_method("increment", &[]).unwrap();
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
        
        let final_value = ly_obj.call_method("get", &[]).unwrap();
        assert_eq!(final_value, Value::Integer(1000));
    }
    
    #[test]
    fn test_concurrent_foreign_object_access() {
        let custom_type = MyCustomType::new(vec![1.0; 1000]);
        let ly_obj = Arc::new(LyObj::new(Box::new(custom_type)));
        
        let handles: Vec<_> = (0..10).map(|thread_id| {
            let obj = Arc::clone(&ly_obj);
            thread::spawn(move || {
                for i in 0..100 {
                    let index = (thread_id * 100 + i) % 1000;
                    let result = obj.call_method("get", &[Value::Integer(index as i64)]);
                    assert!(result.is_ok());
                }
            })
        }).collect();
        
        for handle in handles {
            handle.join().unwrap();
        }
    }
}
```

## Performance Optimization

### Profiling Foreign Objects

**Benchmark Your Implementation:**
```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};
    
    fn bench_custom_type_creation(c: &mut Criterion) {
        let data = vec![1.0; 1000];
        
        c.bench_function("MyCustomType creation", |b| {
            b.iter(|| {
                let custom_type = MyCustomType::new(black_box(data.clone()));
                black_box(custom_type)
            })
        });
    }
    
    fn bench_method_calls(c: &mut Criterion) {
        let custom_type = MyCustomType::new(vec![1.0; 1000]);
        let ly_obj = LyObj::new(Box::new(custom_type));
        
        c.bench_function("method call get", |b| {
            b.iter(|| {
                let result = ly_obj.call_method("get", &[black_box(Value::Integer(500))]);
                black_box(result)
            })
        });
        
        c.bench_function("method call size", |b| {
            b.iter(|| {
                let result = ly_obj.call_method("size", &[]);
                black_box(result)
            })
        });
    }
    
    criterion_group!(benches, bench_custom_type_creation, bench_method_calls);
    criterion_main!(benches);
}
```

### Memory Optimization

**Minimize Allocations:**
```rust
use crate::memory::ValuePools;

#[derive(Debug)]
pub struct OptimizedCustomType {
    data: Vec<f64>,
    // Use object pooling for frequently created objects
    pool: Arc<ValuePools>,
}

impl OptimizedCustomType {
    pub fn new_pooled(data: Vec<f64>, pool: Arc<ValuePools>) -> Self {
        Self { data, pool }
    }
    
    pub fn create_result(&self, value: f64) -> Value {
        // Use pool for value creation
        self.pool.alloc_real(value)
            .unwrap_or_else(|_| Value::Real(value))
    }
}
```

**Cache-Friendly Data Structures:**
```rust
#[repr(C)]  // Ensure predictable memory layout
#[derive(Debug)]
pub struct CacheOptimizedType {
    // Hot data first (frequently accessed)
    size: usize,
    capacity: usize,
    
    // Cold data last (metadata, rarely accessed)
    metadata: std::collections::HashMap<String, String>,
    
    // Data array - align for SIMD operations
    #[repr(align(32))]
    data: Vec<f64>,
}
```

### Error Handling Best Practices

**Comprehensive Error Types:**
```rust
#[derive(Debug, Clone, PartialEq)]
pub enum CustomTypeError {
    IndexOutOfBounds { index: usize, size: usize },
    InvalidData { reason: String },
    ResourceExhausted { resource: String },
    SerializationError { message: String },
}

impl From<CustomTypeError> for ForeignError {
    fn from(error: CustomTypeError) -> Self {
        match error {
            CustomTypeError::IndexOutOfBounds { index, size } => {
                ForeignError::IndexOutOfBounds {
                    index: index.to_string(),
                    bounds: format!("0..{}", size),
                }
            }
            CustomTypeError::InvalidData { reason } => {
                ForeignError::RuntimeError {
                    message: format!("Invalid data: {}", reason),
                }
            }
            // ... other conversions
        }
    }
}
```

**Error Context and Recovery:**
```rust
impl Foreign for MyCustomType {
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        let result = match method {
            "get" => self.safe_get(args),
            "set" => self.safe_set(args),
            _ => return Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        };
        
        result.map_err(|e| {
            // Add context to errors
            log::error!("Method {} failed on {}: {:?}", method, self.type_name(), e);
            e.into()
        })
    }
    
    fn safe_get(&self, args: &[Value]) -> Result<Value, CustomTypeError> {
        // Validate arguments
        if args.len() != 1 {
            return Err(CustomTypeError::InvalidData {
                reason: format!("Expected 1 argument, got {}", args.len()),
            });
        }
        
        let index = match &args[0] {
            Value::Integer(i) if *i >= 0 => *i as usize,
            Value::Integer(i) => return Err(CustomTypeError::IndexOutOfBounds {
                index: *i as usize,
                size: self.data.len(),
            }),
            _ => return Err(CustomTypeError::InvalidData {
                reason: "Index must be a non-negative integer".to_string(),
            }),
        };
        
        self.data.get(index)
            .map(|&value| Value::Real(value))
            .ok_or(CustomTypeError::IndexOutOfBounds {
                index,
                size: self.data.len(),
            })
    }
}
```

## Debugging and Tooling

### Debug Support

**Custom Debug Implementations:**
```rust
use std::fmt;

impl fmt::Debug for MyCustomType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MyCustomType")
            .field("size", &self.data.len())
            .field("data_preview", &self.data.iter().take(5).collect::<Vec<_>>())
            .field("metadata_keys", &self.metadata.keys().collect::<Vec<_>>())
            .finish()
    }
}
```

**Logging Integration:**
```rust
use log::{debug, info, warn, error};

impl Foreign for MyCustomType {
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        debug!("Calling method {} on {} with {} args", method, self.type_name(), args.len());
        
        let start_time = std::time::Instant::now();
        let result = self.call_method_impl(method, args);
        let duration = start_time.elapsed();
        
        match &result {
            Ok(_) => debug!("Method {} completed in {:?}", method, duration),
            Err(e) => warn!("Method {} failed in {:?}: {:?}", method, duration, e),
        }
        
        result
    }
}
```

### Development Tools

**Helper Macros for Foreign Objects:**
```rust
macro_rules! foreign_method {
    ($self:ident, $method:literal, $args:ident, $expected_args:expr, $body:block) => {
        if $args.len() != $expected_args {
            return Err(ForeignError::InvalidArity {
                method: $method.to_string(),
                expected: $expected_args,
                actual: $args.len(),
            });
        }
        $body
    };
}

// Usage
impl Foreign for MyCustomType {
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "get" => foreign_method!(self, "get", args, 1, {
                let index = extract_integer(&args[0])?;
                Ok(Value::Real(self.data[index as usize]))
            }),
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }
}
```

**Type-Safe Argument Extraction:**
```rust
pub fn extract_integer(value: &Value) -> Result<i64, ForeignError> {
    match value {
        Value::Integer(i) => Ok(*i),
        _ => Err(ForeignError::InvalidArgumentType {
            method: "".to_string(),  // Fill in context
            expected: "Integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_real(value: &Value) -> Result<f64, ForeignError> {
    match value {
        Value::Real(r) => Ok(*r),
        Value::Integer(i) => Ok(*i as f64),
        _ => Err(ForeignError::InvalidArgumentType {
            method: "".to_string(),
            expected: "Real or Integer".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

pub fn extract_string(value: &Value) -> Result<String, ForeignError> {
    match value {
        Value::String(s) => Ok(s.clone()),
        _ => Err(ForeignError::InvalidArgumentType {
            method: "".to_string(),
            expected: "String".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}
```

## Contributing Guidelines

### Code Style

**Rust Formatting:**
```toml
# rustfmt.toml
max_width = 100
hard_tabs = false
tab_spaces = 4
newline_style = "Unix"
use_small_heuristics = "Default"
reorder_imports = true
reorder_modules = true
remove_nested_parens = true
edition = "2021"
```

**Documentation Standards:**
```rust
/// A custom data type that demonstrates the Foreign Object Pattern.
/// 
/// This type stores a vector of floating-point numbers and provides
/// methods for accessing and manipulating the data through the Lyra
/// symbolic computation system.
/// 
/// # Examples
/// 
/// ```wolfram
/// (* Create a new instance *)
/// data = MyCustomType[{1.0, 2.0, 3.0}]
/// 
/// (* Get the size *)
/// size = data.size()  (* Returns 3 *)
/// 
/// (* Access elements *)
/// value = data.get(1)  (* Returns 2.0 *)
/// ```
/// 
/// # Thread Safety
/// 
/// This type is thread-safe for read operations but requires external
/// synchronization for write operations.
#[derive(Debug, Clone)]
pub struct MyCustomType {
    /// The internal data storage
    data: Vec<f64>,
    /// Metadata associated with this instance
    metadata: std::collections::HashMap<String, String>,
}
```

### Pull Request Guidelines

**Checklist for PRs:**
- [ ] All tests pass (`cargo test`)
- [ ] Code is formatted (`cargo fmt`)
- [ ] No clippy warnings (`cargo clippy`)
- [ ] Documentation updated for API changes
- [ ] Performance impact assessed
- [ ] Memory safety verified
- [ ] Thread safety considerations documented

**PR Template:**
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Foreign Object Changes
- [ ] New Foreign object type added
- [ ] Existing Foreign object modified
- [ ] Thread safety considerations addressed
- [ ] Memory management verified

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests run
- [ ] Memory tests run

## Performance Impact
Describe any performance implications

## Breaking Changes
List any breaking changes

## Documentation
- [ ] Code comments updated
- [ ] API documentation updated
- [ ] Architecture documentation updated
```

### Release Process

**Version Compatibility:**
- Foreign object API changes require major version bump
- New Foreign object types can be minor version
- Bug fixes in Foreign objects are patch version

**Performance Regression Testing:**
```bash
# Run performance regression tests
cargo bench --bench foreign_object_benchmarks
cargo bench --bench integration_benchmarks

# Compare with baseline
cargo install cargo-criterion
cargo criterion --compare baseline
```

## References

- [Foreign Object Pattern ADR](ADRs/001-foreign-object-pattern.md)
- [Zero VM Pollution ADR](ADRs/004-zero-vm-pollution.md)
- [Static Dispatch Design ADR](ADRs/002-static-dispatch-design.md)
- [Performance Tuning Guide](performance-tuning.md)
- [Threading Model Documentation](threading-model.md)
- [Example Foreign Objects](../src/stdlib/)
- [Test Examples](../tests/foreign_tests.rs)