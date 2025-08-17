# Lyra Examples

This directory contains example scripts that demonstrate various features of the Lyra programming language.

## Running Examples

To run any example, use the `lyra run` command:

```bash
# Run basic arithmetic examples
lyra run examples/basic_arithmetic.lyra

# Run list operations
lyra run examples/list_operations.lyra

# Run mathematical functions
lyra run examples/mathematical_functions.lyra

# Run string operations  
lyra run examples/string_operations.lyra

# Run complex expressions
lyra run examples/complex_expressions.lyra
```

## Example Files

### `basic_arithmetic.lyra`
Demonstrates basic arithmetic operations including:
- Addition, subtraction, multiplication, division
- Exponentiation and order of operations
- Working with integers and real numbers

### `list_operations.lyra`
Shows list manipulation capabilities:
- Creating lists with mixed types
- List functions: Length, Head, Tail, Append, Flatten
- Working with nested and empty lists

### `mathematical_functions.lyra`
Covers mathematical functions:
- Trigonometric functions: Sin, Cos, Tan
- Exponential and logarithmic functions: Exp, Log
- Square root and power operations

### `string_operations.lyra`
Demonstrates string handling:
- String creation and length calculation
- String joining and manipulation
- Taking and dropping characters
- Unicode support

### `complex_expressions.lyra`
Shows advanced usage with nested function calls:
- Combining multiple operations
- Performance test cases
- Real-world-like computations

## Exploring with REPL

You can also explore these examples interactively using the REPL:

```bash
lyra repl
```

In the REPL, try typing:
- `help` - Show available commands
- `functions` - List all available functions  
- `examples` - Show example expressions
- `exit` - Quit the REPL

## Building Examples

To check syntax without running:

```bash
lyra build examples/basic_arithmetic.lyra
```

## Viewing Intermediate Representation

To see how expressions are parsed and compiled:

```bash
lyra dump-ir examples/basic_arithmetic.lyra
```

This shows the AST, bytecode, constants, and symbols for educational purposes.