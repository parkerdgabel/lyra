// VM Integration Example
//
// This example shows the basic patterns for integrating with Lyra's VM:
// - Creating and manipulating Values
// - Using the VM stack
// - Calling functions
// - Error handling

use crate::vm::{Value, VirtualMachine, VmResult, VmError};
use std::collections::HashMap;

/// Basic VM usage examples
pub fn basic_vm_usage() -> VmResult<()> {
    let mut vm = VirtualMachine::new();

    // Creating different types of Values
    let integer_val = Value::Integer(42);
    let real_val = Value::Real(3.14);
    let string_val = Value::String("Hello, Lyra!".to_string());
    let symbol_val = Value::Symbol("x".to_string());
    let boolean_val = Value::Boolean(true);
    let missing_val = Value::Missing;

    // Working with lists
    let list_val = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
    ]);

    // Working with objects (dictionaries)
    let mut object = HashMap::new();
    object.insert("name".to_string(), Value::String("Alice".to_string()));
    object.insert("age".to_string(), Value::Integer(30));
    let object_val = Value::Object(object);

    // Push values onto the VM stack
    vm.push(integer_val);
    vm.push(real_val);
    vm.push(string_val);

    println!("Stack depth: {}", vm.stack.len());

    // Pop values from the stack
    let popped = vm.pop()?;
    println!("Popped value: {:?}", popped);

    // Peek at the top of the stack
    if let Some(top) = vm.stack.last() {
        println!("Top of stack: {:?}", top);
    }

    Ok(())
}

/// Working with Value types and conversions
pub fn value_operations() -> VmResult<()> {
    // Type checking
    let val = Value::Integer(42);
    assert!(matches!(val, Value::Integer(_)));
    
    // Value conversions
    let num_val = Value::Integer(42);
    if let Value::Integer(n) = num_val {
        println!("Integer value: {}", n);
    }

    // Working with lists
    let list = Value::List(vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
    ]);

    if let Value::List(items) = &list {
        for (i, item) in items.iter().enumerate() {
            println!("Item {}: {:?}", i, item);
        }
    }

    // Working with objects
    let mut object = HashMap::new();
    object.insert("x".to_string(), Value::Integer(10));
    object.insert("y".to_string(), Value::Integer(20));
    let obj_val = Value::Object(object);

    if let Value::Object(obj) = &obj_val {
        if let Some(x_val) = obj.get("x") {
            println!("x value: {:?}", x_val);
        }
    }

    Ok(())
}

/// Function calling patterns
pub fn function_calling() -> VmResult<()> {
    let mut vm = VirtualMachine::new();

    // Example: Calling a simple math function
    // This simulates: Plus[2, 3]
    
    // Push arguments onto stack (in reverse order due to stack nature)
    vm.push(Value::Integer(3));
    vm.push(Value::Integer(2));
    
    // Call the function through stdlib
    let args = vec![
        vm.pop()?,
        vm.pop()?,
    ];
    
    if let Some(plus_fn) = vm.stdlib.get_function("Plus") {
        let result = plus_fn(&args)?;
        vm.push(result);
        
        println!("Result of Plus[2, 3]: {:?}", vm.pop()?);
    }

    Ok(())
}

/// Error handling patterns
pub fn error_handling_examples() -> VmResult<()> {
    let mut vm = VirtualMachine::new();

    // Example 1: Stack underflow error
    let result = vm.pop();
    match result {
        Ok(value) => println!("Got value: {:?}", value),
        Err(VmError::StackUnderflow) => println!("Stack was empty!"),
        Err(e) => println!("Other error: {:?}", e),
    }

    // Example 2: Type error handling
    fn safe_add(a: &Value, b: &Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(x), Value::Integer(y)) => Ok(Value::Integer(x + y)),
            (Value::Real(x), Value::Real(y)) => Ok(Value::Real(x + y)),
            (Value::Integer(x), Value::Real(y)) => Ok(Value::Real(*x as f64 + y)),
            (Value::Real(x), Value::Integer(y)) => Ok(Value::Real(x + *y as f64)),
            _ => Err(VmError::TypeError {
                expected: "numeric types".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    let val1 = Value::Integer(5);
    let val2 = Value::String("hello".to_string());
    
    match safe_add(&val1, &val2) {
        Ok(result) => println!("Addition result: {:?}", result),
        Err(VmError::TypeError { expected, actual }) => {
            println!("Type error: expected {}, got {}", expected, actual);
        }
        Err(e) => println!("Other error: {:?}", e),
    }

    Ok(())
}

/// Working with symbols and global state
pub fn symbol_operations() -> VmResult<()> {
    let mut vm = VirtualMachine::new();

    // Set a global symbol value
    vm.global_symbols.insert("x".to_string(), Value::Integer(42));
    vm.global_symbols.insert("message".to_string(), Value::String("Hello".to_string()));

    // Retrieve symbol values
    if let Some(x_val) = vm.global_symbols.get("x") {
        println!("Value of x: {:?}", x_val);
    }

    // Working with delayed definitions (:=)
    // This would normally be done through the compiler, but we can simulate it
    let delayed_expr = crate::ast::Expr::Number(crate::ast::Number::Integer(100));
    vm.delayed_definitions.insert("y".to_string(), delayed_expr);

    Ok(())
}

/// Batch processing multiple values
pub fn batch_processing() -> VmResult<()> {
    let values = vec![
        Value::Integer(1),
        Value::Integer(2),
        Value::Integer(3),
        Value::Integer(4),
        Value::Integer(5),
    ];

    // Process each value (square it)
    let mut results = Vec::new();
    
    for value in values {
        match value {
            Value::Integer(n) => {
                results.push(Value::Integer(n * n));
            }
            Value::Real(f) => {
                results.push(Value::Real(f * f));
            }
            _ => {
                // Skip non-numeric values or handle differently
                results.push(Value::Missing);
            }
        }
    }

    println!("Squared values: {:?}", results);
    Ok(())
}

/// Working with Foreign objects
pub fn foreign_object_usage() -> VmResult<()> {
    use crate::foreign::LyObj;
    
    // Assume we have a Counter object from the simple-foreign-object example
    // This shows how to work with Foreign objects in the VM
    
    // Create a foreign object (this would typically be done through stdlib functions)
    // let counter = Counter::new();
    // let foreign_value = Value::LyObj(LyObj::new(Box::new(counter)));
    
    // For this example, we'll simulate it
    let foreign_value = Value::Missing; // Placeholder
    
    match foreign_value {
        Value::LyObj(obj) => {
            println!("Foreign object type: {}", obj.type_name());
            
            // Call methods on the foreign object
            let result = obj.call_method("get", &[])?;
            println!("Method call result: {:?}", result);
            
            // Downcast to concrete type if needed
            // if let Some(concrete) = obj.downcast_ref::<Counter>() {
            //     println!("Direct access: {}", concrete.get());
            // }
        }
        _ => println!("Not a foreign object"),
    }

    Ok(())
}

/// Practical example: Building a simple calculator
pub fn calculator_example() -> VmResult<()> {
    let mut vm = VirtualMachine::new();

    // Define a simple calculator function
    fn calculate(operation: &str, a: i64, b: i64) -> VmResult<Value> {
        match operation {
            "add" => Ok(Value::Integer(a + b)),
            "subtract" => Ok(Value::Integer(a - b)),
            "multiply" => Ok(Value::Integer(a * b)),
            "divide" => {
                if b == 0 {
                    Err(VmError::DivisionByZero)
                } else {
                    Ok(Value::Real(a as f64 / b as f64))
                }
            }
            _ => Err(VmError::Runtime(format!("Unknown operation: {}", operation))),
        }
    }

    // Use the calculator
    let operations = vec![
        ("add", 10, 5),
        ("subtract", 10, 3),
        ("multiply", 4, 7),
        ("divide", 15, 3),
        ("divide", 10, 0), // This will cause an error
    ];

    for (op, a, b) in operations {
        match calculate(op, a, b) {
            Ok(result) => println!("{} {} {} = {:?}", a, op, b, result),
            Err(VmError::DivisionByZero) => println!("{} {} {} = Error: Division by zero", a, op, b),
            Err(e) => println!("{} {} {} = Error: {:?}", a, op, b, e),
        }
    }

    Ok(())
}

/// Memory-efficient value creation
pub fn efficient_value_creation() -> VmResult<()> {
    // When you know the size, pre-allocate
    let mut large_list = Vec::with_capacity(1000);
    for i in 0..1000 {
        large_list.push(Value::Integer(i));
    }
    let list_value = Value::List(large_list);

    // Use move semantics to avoid clones
    fn process_list(list: Value) -> VmResult<usize> {
        match list {
            Value::List(items) => Ok(items.len()),
            _ => Err(VmError::TypeError {
                expected: "List".to_string(),
                actual: "other".to_string(),
            }),
        }
    }

    let length = process_list(list_value)?;
    println!("List length: {}", length);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_vm_operations() {
        let mut vm = VirtualMachine::new();
        
        // Test push and pop
        vm.push(Value::Integer(42)).unwrap();
        let result = vm.pop().unwrap();
        assert_eq!(result, Value::Integer(42));
    }

    #[test]
    fn test_value_type_checking() {
        let val = Value::Integer(42);
        assert!(matches!(val, Value::Integer(_)));
        
        let val = Value::String("hello".to_string());
        assert!(matches!(val, Value::String(_)));
    }

    #[test]
    fn test_error_handling() {
        let mut vm = VirtualMachine::new();
        
        // Test stack underflow
        let result = vm.pop();
        assert!(matches!(result, Err(VmError::StackUnderflow)));
    }
}

// Usage Summary:
//
// 1. Create a VM instance: let mut vm = VirtualMachine::new();
// 2. Create Values: Value::Integer(42), Value::String("hello".to_string()), etc.
// 3. Use the stack: vm.push(value), vm.pop(), vm.peek()
// 4. Handle errors with VmResult and match statements
// 5. Work with symbols through vm.global_symbols
// 6. Call stdlib functions through vm.stdlib.get_function()
// 7. Always validate inputs and handle edge cases
//
// Key patterns:
// - Use match statements for Value pattern matching
// - Always handle VmResult errors properly
// - Pre-allocate collections when size is known
// - Use references to avoid unnecessary clones
// - Validate arguments before processing