# Lyra: Symbolic Computation Engine

A production-grade symbolic computation engine inspired by the Wolfram Language, built in Rust with a focus on performance, safety, and developer experience.

## Features

- **Wolfram-Inspired Syntax**: Use familiar syntax like `f[x, y]`, `{1, 2, 3}`, and `x_ -> x^2`
- **Pattern Matching**: Full support for patterns (`_`, `__`, `___`) with typed constraints
- **Symbolic Computing**: Manipulate mathematical expressions symbolically
- **Fast Parser**: Hand-optimized recursive descent parser with comprehensive error reporting
- **Test-Driven Development**: 71 comprehensive tests ensuring reliability

## Language Syntax

### Basic Expressions
```
2 + 3 * 4         (* Arithmetic with proper precedence *)
f[x, y]           (* Function calls *)
{1, 2, 3}         (* Lists *)
"hello"           (* Strings *)
```

### Patterns and Rules
```
x_                (* Blank pattern *)
x_Integer         (* Typed pattern *)
x -> x^2          (* Rule *)
x :> RandomReal[] (* Delayed rule *)
expr /. rule      (* Apply rule to expression *)
```

### Function Definitions
```
f[x_] = x^2       (* Immediate definition *)
f[x_] := x + 1    (* Delayed definition *)
```

## Quick Start

### Prerequisites
- Rust 1.70 or later
- Cargo (comes with Rust)

### Installation
```bash
git clone https://github.com/parkerdgabel/lyra.git
cd lyra
cargo build --release
```

### Running Tests
```bash
cargo test
```

### Usage
```bash
# Interactive REPL (coming soon)
cargo run -- repl

# Run a script file (coming soon)
cargo run -- run script.lyra

# Show help
cargo run -- --help
```

## Development Status

### ✅ Completed (Week 1)
- [x] Language grammar specification
- [x] Complete lexer with tokenization
- [x] Full expression parser 
- [x] AST design and implementation
- [x] Comprehensive test suite (71 tests)
- [x] Error handling and reporting

### 🚧 In Progress (Week 2)
- [ ] Bytecode design and VM implementation
- [ ] Compiler (AST → Bytecode)
- [ ] Head table and attributes system
- [ ] Basic arithmetic evaluation

### 📋 Planned (Weeks 3-4)
- [ ] Standard library (Math, Lists, Strings)
- [ ] Pattern matching and rule system
- [ ] CLI interface with REPL
- [ ] Import system and tree-shaking
- [ ] Performance optimization

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Source    │───▶│    Lexer    │───▶│   Parser    │
│    Code     │    │  (Tokens)   │    │   (AST)     │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     VM      │◀───│  Compiler   │◀───│  Desugarer  │
│ (Execution) │    │ (Bytecode)  │    │ (Core AST)  │
└─────────────┘    └─────────────┘    └─────────────┘
```

## Examples

### Mathematical Expressions
```lyra
(* Basic arithmetic *)
result = 2 + 3 * 4 ^ 2    (* 50 *)

(* Function definition *)
square[x_] = x^2
square[5]                 (* 25 *)

(* Pattern matching *)
factorial[0] = 1
factorial[n_Integer] := n * factorial[n - 1]
```

### Data Manipulation
```lyra
(* Lists *)
numbers = {1, 2, 3, 4, 5}
doubled = Map[#*2&, numbers]

(* Rules and replacement *)
expr = a + b + c
result = expr /. {a -> 1, b -> 2, c -> 3}  (* 6 *)
```

## Contributing

This project follows strict Test-Driven Development practices. See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the Wolfram Language and Mathematica
- Built with ❤️ in Rust