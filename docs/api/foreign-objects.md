# Foreign Object API Documentation (Lightweight)

## Quick Start

Foreign Objects let you add complex data types to Lyra without modifying the VM core. This guide shows you the essentials to get started quickly.

## Core Concept

1. **`Foreign` trait**: Your custom type implements this interface
2. **`LyObj` wrapper**: Wraps your type for the VM 
3. **Method calls**: Users call methods on your objects from Lyra code

```rust
// Your custom type
#[derive(Debug, Clone)]
pub struct MyType { /* your data */ }

// Implement Foreign trait
impl Foreign for MyType {
    fn type_name(&self) -> &'static str { "MyType" }
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> { /* ... */ }
    fn clone_boxed(&self) -> Box<dyn Foreign> { Box::new(self.clone()) }
    fn as_any(&self) -> &dyn Any { self }
}

// Use in VM
let obj = Value::LyObj(LyObj::new(Box::new(MyType::new())));
```

## Step-by-Step Example

See `examples/simple-foreign-object.rs` for a complete working example. Here's the basic pattern:

```rust
#[derive(Debug, Clone)]
pub struct Counter {
    value: i64,
}

impl Foreign for Counter {
    fn type_name(&self) -> &'static str {
        "Counter"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "get" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.value))
            }
            "increment" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let mut new_counter = self.clone();
                new_counter.value += 1;
                Ok(Value::LyObj(LyObj::new(Box::new(new_counter))))
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
}

// Required for thread safety
unsafe impl Send for Counter {}
unsafe impl Sync for Counter {}
```

## Using LyObj Wrapper

```rust
// Create your object
let counter = Counter::new();

// Wrap for VM
let lyobj = LyObj::new(Box::new(counter));

// Use as VM Value
let value = Value::LyObj(lyobj);

// Call methods
if let Value::LyObj(obj) = &value {
    let result = obj.call_method("get", &[])?;  // Returns current value
    let incremented = obj.call_method("increment", &[])?;  // Returns new Counter
}
```

## Register with Stdlib

```rust
use crate::stdlib::StdlibFunction;

pub fn create_counter(args: &[Value]) -> VmResult<Value> {
    match args {
        [] => {
            let counter = Counter::new();
            Ok(Value::LyObj(LyObj::new(Box::new(counter))))
        }
        [Value::Integer(value)] => {
            let counter = Counter::with_value(*value);
            Ok(Value::LyObj(LyObj::new(Box::new(counter))))
        }
        _ => Err(VmError::TypeError {
            expected: "Counter[] or Counter[value]".to_string(),
            actual: format!("{} arguments", args.len()),
        }),
    }
}

// Register with stdlib
stdlib.register("Counter", create_counter);
```

## Essential Error Handling

```rust
fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
    match method {
        "myMethod" => {
            // 1. Check argument count
            if args.len() != 1 {
                return Err(ForeignError::InvalidArity {
                    method: method.to_string(),
                    expected: 1,
                    actual: args.len(),
                });
            }
            
            // 2. Check argument type
            let value = match &args[0] {
                Value::Integer(n) => *n,
                _ => return Err(ForeignError::InvalidArgumentType {
                    method: method.to_string(),
                    expected: "Integer".to_string(),
                    actual: format!("{:?}", args[0]),
                }),
            };
            
            // 3. Do your work
            Ok(Value::Integer(value * 2))
        }
        _ => Err(ForeignError::UnknownMethod {
            type_name: self.type_name().to_string(),
            method: method.to_string(),
        }),
    }
}
```

## Thread Safety (Required)

```rust
// Always add these for thread safety
unsafe impl Send for Counter {}
unsafe impl Sync for Counter {}
```

## Quick Checklist

✅ **Must implement**:
- `type_name()` - return your type name
- `call_method()` - handle method calls  
- `clone_boxed()` - usually `Box::new(self.clone())`
- `as_any()` - usually `self`
- `Send + Sync` - add unsafe impl for both

✅ **Best practices**:
- Validate argument counts and types
- Return meaningful errors
- Use `Clone` for immutable operations
- Test thoroughly with edge cases

## Usage in Lyra Code

After registration, users can use your objects:

```wolfram
(* Create objects *)
counter = Counter[10]
counter2 = Counter[]

(* Call methods *)
value = counter.get()          (* Returns 10 *)
newCounter = counter.increment() (* Returns Counter with value 11 *)

(* Chain operations *)
result = Counter[5].increment().increment().get()  (* Returns 7 *)
```

## Complete Examples

- **Simple Counter**: `examples/simple-foreign-object.rs`
- **Advanced TimeSeries**: `examples/foreign-object-example.rs` 
- **VM Integration**: `examples/vm-integration.rs`
- **Function Registration**: `examples/function-registration.rs`

These examples show real working code you can adapt for your own Foreign objects.