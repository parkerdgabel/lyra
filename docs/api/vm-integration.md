# VM Integration API Documentation (Lightweight)

## Quick Start

Lyra's VM is a stack-based execution engine for symbolic computation. This guide shows you the essentials for integrating with the VM.

## Core Concepts

1. **Values**: All data in the VM is represented as `Value` enum variants
2. **Stack**: Push values, call functions, pop results
3. **Errors**: Use `VmResult<T>` for error handling
4. **Functions**: Register custom functions via stdlib

## Essential Value Types

```rust
pub enum Value {
    Integer(i64),        // 42
    Real(f64),           // 3.14
    String(String),      // "hello"
    Symbol(String),      // x, y, MySymbol
    Boolean(bool),       // true, false
    Missing,             // undefined/unknown
    List(Vec<Value>),    // {1, 2, 3}
    Object(HashMap<String, Value>), // {"key" -> value}
    LyObj(LyObj),        // Foreign objects
    // ... plus symbolic computation types
}
```

## Creating Values

```rust
// Basic types
let num = Value::Integer(42);
let real = Value::Real(3.14);
let text = Value::String("Hello".to_string());
let symbol = Value::Symbol("x".to_string());
let flag = Value::Boolean(true);
let unknown = Value::Missing;

// Collections
let list = Value::List(vec![
    Value::Integer(1),
    Value::Integer(2),
    Value::Integer(3),
]);

let mut obj = HashMap::new();
obj.insert("name".to_string(), Value::String("Alice".to_string()));
obj.insert("age".to_string(), Value::Integer(30));
let object = Value::Object(obj);
```

## Basic VM Usage

```rust
use crate::vm::{VirtualMachine, Value, VmResult};

// Create VM
let mut vm = VirtualMachine::new();

// Push values onto stack
vm.push(Value::Integer(42));
vm.push(Value::String("hello".to_string()));

// Pop values from stack
let value = vm.pop()?;  // Returns "hello"
let number = vm.pop()?; // Returns 42

// Check stack depth
let depth = vm.stack.len();

// Peek without removing
if let Some(top) = vm.stack.last() {
    println!("Top: {:?}", top);
}
```

## Value Pattern Matching

```rust
// Check what type a value is
let val = Value::Integer(42);

match val {
    Value::Integer(n) => println!("Integer: {}", n),
    Value::Real(f) => println!("Real: {}", f),
    Value::String(s) => println!("String: {}", s),
    Value::List(items) => println!("List with {} items", items.len()),
    Value::Missing => println!("Missing value"),
    _ => println!("Other type"),
}

// Extract values safely
if let Value::Integer(n) = val {
    println!("Got integer: {}", n);
}

// Work with lists
if let Value::List(items) = &some_value {
    for item in items {
        // Process each item
    }
}
```

## Error Handling

```rust
pub type VmResult<T> = Result<T, VmError>;

// Common errors
pub enum VmError {
    StackUnderflow,                    // Pop from empty stack
    TypeError { expected: String, actual: String },  // Wrong type
    DivisionByZero,                    // Math error
    IndexError { index: i64, length: usize },        // Out of bounds
    Runtime(String),                   // General runtime error
    // ... others
}
```

## Essential Error Patterns

```rust
// Always use VmResult for functions that can fail
fn safe_operation(value: &Value) -> VmResult<Value> {
    match value {
        Value::Integer(n) => Ok(Value::Integer(n * 2)),
        Value::Real(f) => Ok(Value::Real(f * 2.0)),
        _ => Err(VmError::TypeError {
            expected: "number".to_string(),
            actual: format!("{:?}", value),
        }),
    }
}

// Use ? operator to propagate errors
fn chain_operations() -> VmResult<Value> {
    let val = safe_operation(&Value::Integer(5))?;
    let result = safe_operation(&val)?;
    Ok(result)
}

// Handle errors with match
match safe_operation(&some_value) {
    Ok(result) => println!("Success: {:?}", result),
    Err(VmError::TypeError { expected, actual }) => {
        println!("Type error: expected {}, got {}", expected, actual);
    }
    Err(e) => println!("Other error: {:?}", e),
}
```

## Calling Functions

```rust
// Get a stdlib function and call it
let mut vm = VirtualMachine::new();

if let Some(plus_fn) = vm.stdlib.get_function("Plus") {
    let args = vec![Value::Integer(2), Value::Integer(3)];
    let result = plus_fn(&args)?;  // Returns Integer(5)
}
```

## Working with Symbols

```rust
// Set global symbol values
vm.global_symbols.insert("x".to_string(), Value::Integer(42));

// Get symbol values
if let Some(x_val) = vm.global_symbols.get("x") {
    println!("x = {:?}", x_val);
}
```

## Common Patterns

```rust
// Validate function arguments
fn my_function(args: &[Value]) -> VmResult<Value> {
    // Check argument count
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Extract and validate arguments
    let a = match &args[0] {
        Value::Integer(n) => *n,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let b = match &args[1] {
        Value::Integer(n) => *n,
        _ => return Err(VmError::TypeError {
            expected: "Integer".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    // Do computation
    Ok(Value::Integer(a + b))
}

// Process lists safely
fn process_list(list_val: &Value) -> VmResult<Vec<Value>> {
    let items = match list_val {
        Value::List(items) => items,
        _ => return Err(VmError::TypeError {
            expected: "List".to_string(),
            actual: format!("{:?}", list_val),
        }),
    };
    
    let mut results = Vec::new();
    for item in items {
        // Process each item
        let processed = process_item(item)?;
        results.push(processed);
    }
    Ok(results)
}
```

## Quick Reference

### Essential VM Operations
- `vm.push(value)` - Add value to stack
- `vm.pop()?` - Remove and return top value
- `vm.stack.len()` - Get stack size
- `vm.global_symbols.insert(name, value)` - Set symbol

### Essential Value Types
- `Value::Integer(42)` - Numbers
- `Value::String("text".to_string())` - Text
- `Value::List(vec![...])` - Collections
- `Value::Missing` - Undefined values

### Essential Error Handling
- Use `VmResult<T>` for return types
- Use `?` operator to propagate errors
- Match on specific error types when needed
- Always validate inputs

## Complete Examples

See these files for working examples:
- **Basic VM Usage**: `examples/vm-integration.rs`
- **Function Registration**: `examples/function-registration.rs`
- **Foreign Objects**: `examples/simple-foreign-object.rs`