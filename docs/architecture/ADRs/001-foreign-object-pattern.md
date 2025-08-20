# ADR-001: Foreign Object Pattern Rationale

## Status
Accepted

## Context

The Lyra symbolic computation engine faces a critical architectural challenge: how to support complex data types and operations (tensors, datasets, async operations, ML models) while maintaining VM simplicity and performance. Traditional approaches would either:

1. **Bloat the VM Core**: Add all types to the `Value` enum, causing performance degradation and complexity explosion
2. **Limit Functionality**: Restrict the system to only basic symbolic operations
3. **Multiple Type Systems**: Create separate systems that don't integrate well

The core tension is between **VM simplicity** (essential for symbolic computation performance) and **extensibility** (required for a practical computational system).

## Decision

Implement the **Foreign Object Pattern** using the `LyObj` wrapper and `Foreign` trait:

```rust
pub trait Foreign: fmt::Debug + Send + Sync {
    fn type_name(&self) -> &'static str;
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError>;
    fn clone_boxed(&self) -> Box<dyn Foreign>;
    fn as_any(&self) -> &dyn Any;
}

pub enum Value {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String),
    Boolean(bool),
    Missing,
    LyObj(LyObj),           // Single entry point for all complex types
    Quote(Box<Expr>),
    Pattern(Pattern),
}
```

## Rationale

### Performance Benefits
- **Minimal VM Types**: VM `Value` enum remains small and cache-friendly
- **Zero Cost Abstraction**: No performance penalty for basic symbolic operations
- **Efficient Dispatch**: Method calls use trait objects with virtual dispatch
- **Memory Locality**: Related operations stay within the Foreign object's memory space

### Architectural Benefits
- **Clean Separation**: Complex logic lives outside VM core
- **Type Safety**: Rust's type system enforces correct method signatures
- **Extensibility**: New types can be added without touching VM code
- **Maintainability**: Domain experts can implement types without VM knowledge

### Integration Benefits
- **Seamless VM Integration**: Foreign objects work naturally with existing VM operations
- **Pattern Matching**: Foreign objects participate in pattern matching system
- **Error Handling**: Consistent error propagation through VM error system
- **Serialization**: Optional serialization support for persistence

## Implementation

### Core Components

**Foreign Trait**: Defines the interface all complex types must implement
```rust
impl Foreign for AsyncFuture {
    fn type_name(&self) -> &'static str { "Future" }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "resolve" => Ok(self.value.clone()),
            "isCompleted" => Ok(Value::Boolean(true)),
            _ => Err(ForeignError::UnknownMethod { 
                type_name: "Future".to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any { self }
}
```

**LyObj Wrapper**: Type-erased container that maintains VM compatibility
```rust
impl LyObj {
    pub fn new(foreign: Box<dyn Foreign>) -> Self {
        LyObj { inner: foreign }
    }
    
    pub fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        self.inner.call_method(method, args)
    }
}
```

**VM Integration**: Seamless method calls through bytecode
```rust
// VM handles method calls transparently
match instruction {
    OpCode::CallMethod => {
        let obj = self.stack.pop()?;
        if let Value::LyObj(ly_obj) = obj {
            let result = ly_obj.call_method(method, &args)?;
            self.stack.push(result);
        }
    }
}
```

### Current Implementations

1. **Data Types**: Table, Tensor, Series, Dataset, Schema
2. **Async Operations**: Future, Promise, ThreadPool, Channel
3. **ML Components**: Neural networks, optimizers, loss functions
4. **I/O Operations**: File readers, network handlers

## Consequences

### Positive

- **Maintained VM Performance**: Core symbolic operations run at full speed
- **Unlimited Extensibility**: Any Rust type can become a Foreign object
- **Type Safety**: Compile-time guarantees for method signatures
- **Clean Architecture**: Clear boundaries between VM and stdlib
- **Memory Efficiency**: Large objects stay out of VM stack/heap
- **Thread Safety**: Foreign trait requires Send + Sync

### Negative

- **Indirect Method Calls**: Virtual dispatch has small overhead vs direct calls
- **Boxing Overhead**: All Foreign objects are heap-allocated
- **Learning Curve**: Developers must understand trait object patterns
- **Debug Complexity**: Type-erased objects harder to debug

### Mitigation Strategies

- **Performance**: Hot paths use direct VM operations, not Foreign objects
- **Memory**: Foreign objects pool and reuse expensive allocations
- **Debugging**: Comprehensive error messages and type information
- **Documentation**: Clear patterns and examples for new implementations

## Alternatives Considered

### 1. Direct VM Integration
**Approach**: Add complex types directly to `Value` enum
```rust
pub enum Value {
    Integer(i64),
    Tensor(ArrayD<f64>),     // Direct integration
    Future(AsyncFuture),     // VM pollution
    Dataset(DataFrame),      // Complexity explosion
}
```
**Rejected Because**:
- VM performance degradation (larger enum, more match arms)
- Circular dependencies between VM and stdlib
- Impossible to add types without VM changes

### 2. Separate Type Systems
**Approach**: Create parallel systems for complex types
**Rejected Because**:
- Poor integration with symbolic computation
- Duplicate pattern matching and evaluation logic
- User confusion about which system to use

### 3. Generic VM with Type Parameters
**Approach**: Make VM generic over value types
**Rejected Because**:
- Rust compilation complexity
- Loss of dynamic typing capability
- Incompatible with symbolic computation requirements

## Implementation Examples

### Creating a Foreign Object
```rust
pub struct MyComplexType {
    data: Vec<f64>,
    metadata: HashMap<String, Value>,
}

impl Foreign for MyComplexType {
    fn type_name(&self) -> &'static str { "MyComplexType" }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "size" => Ok(Value::Integer(self.data.len() as i64)),
            "get" => {
                let index = extract_integer(&args[0])?;
                Ok(Value::Real(self.data[index as usize]))
            }
            _ => Err(ForeignError::UnknownMethod { 
                type_name: "MyComplexType".to_string(), 
                method: method.to_string() 
            })
        }
    }
    
    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }
    
    fn as_any(&self) -> &dyn Any { self }
}
```

### Using in Stdlib
```rust
pub fn create_my_type(args: &[Value]) -> VmResult<Value> {
    let my_type = MyComplexType::new(args)?;
    Ok(Value::LyObj(LyObj::new(Box::new(my_type))))
}
```

### User Code
```wolfram
(* Create instance *)
obj = MyComplexType[{1.0, 2.0, 3.0}]

(* Call methods *)
size = obj.size()        (* → 3 *)
value = obj.get(1)       (* → 2.0 *)
```

## Validation

The Foreign Object Pattern has been successfully validated through:

1. **Performance Tests**: VM performance maintained for symbolic operations
2. **Complexity Tests**: Successfully implemented 15+ complex types
3. **Integration Tests**: Seamless interaction with pattern matching and typing
4. **User Tests**: Intuitive API for both implementers and users

## Future Considerations

- **JIT Optimization**: Specialize hot Foreign method calls
- **Memory Pools**: Shared allocation strategies for Foreign objects
- **Serialization Standard**: Consistent serialization across all Foreign types
- **Reflection**: Runtime type information for better debugging

## References

- [VM Design Principles](../system-architecture.md#vm-design)
- [Performance Benchmarks](../../PERFORMANCE.md)
- [Foreign Trait Implementation](../../src/foreign.rs)
- [Example Foreign Objects](../../src/stdlib/)