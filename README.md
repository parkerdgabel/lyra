# Lyra: Symbolic Computation Engine

A production-grade symbolic computation engine inspired by the Wolfram Language, built in Rust with a focus on performance, safety, and developer experience.

## Features

- **Wolfram-Inspired Syntax**: Use familiar syntax like `f[x, y]`, `{1, 2, 3}`, and `Sin[Pi/2]`
- **Full Runtime System**: Complete VM with bytecode compilation and execution
- **Rich Standard Library**: Math functions, list operations, string manipulation, and rules
- **Interactive REPL**: Full-featured command-line interface with help system
- **Fast Parser**: Hand-optimized recursive descent parser with comprehensive error reporting
- **Developer Experience**: Enhanced error messages, syntax highlighting, and debugging tools
- **Test-Driven Development**: 237+ comprehensive tests with snapshot testing for reliability

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

#### Interactive REPL
```bash
# Start the REPL
cargo run -- repl

# In the REPL, try:
lyra[1]> 2 + 3 * 4
Out[1]= 14

lyra[2]> Sin[3.14159 / 2]
Out[2]= 1.0

lyra[3]> Length[{1, 2, 3, 4, 5}]
Out[3]= 5

lyra[4]> help
# Shows available commands and syntax help

lyra[5]> functions  
# Lists all available functions

lyra[6]> examples
# Shows example expressions
```

#### Running Script Files
```bash
# Run a .lyra script file
cargo run -- run examples/basic_arithmetic.lyra

# Compile without running (syntax check)
cargo run -- build examples/basic_arithmetic.lyra

# Show intermediate representation (for debugging)
cargo run -- dump-ir examples/basic_arithmetic.lyra
```

#### Command Line Help
```bash
# Show all available commands
cargo run -- --help

# Show version
cargo run -- --version
```

## Development Status

### ✅ Week 1 - Syntax & Parser (Complete)
- [x] Language grammar specification  
- [x] Complete lexer with tokenization
- [x] Full expression parser with modern Wolfram syntax
- [x] AST design and implementation
- [x] Comprehensive test suite
- [x] Error handling and reporting

### ✅ Week 2 - VM & Compiler Core (Complete)
- [x] Bytecode format and instruction set
- [x] Stack-based virtual machine implementation
- [x] Compiler (AST → Bytecode) with optimization
- [x] Head table and attributes system
- [x] Arithmetic and function call evaluation

### ✅ Week 3 - Standard Library & Linking (Complete)
- [x] Mathematical functions (Sin, Cos, Tan, Exp, Log, Sqrt)
- [x] List operations (Length, Head, Tail, Append, Flatten)
- [x] String manipulation (StringJoin, StringLength, StringTake, StringDrop)
- [x] Rule and pattern system basics
- [x] Tree-shaking and dependency resolution

### ✅ Week 4 - CLI & Polish (Complete)
- [x] Full CLI interface with clap
- [x] Interactive REPL with rustyline integration
- [x] File execution (run, build, dump-ir commands)
- [x] Enhanced error messages with line numbers and suggestions
- [x] Example scripts and documentation
- [x] Performance benchmarks with Criterion
- [x] Snapshot testing with insta for regression protection
- [x] Code quality improvements (clippy, formatting)

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
# Basic arithmetic
2 + 3 * 4           # 14
2^8                 # 256
Sqrt[16]            # 4.0

# Mathematical functions  
Sin[3.14159 / 2]    # 1.0
Cos[0]              # 1.0
Log[Exp[1]]         # 1.0
Sqrt[3^2 + 4^2]     # 5.0
```

### List Operations
```lyra
# Creating and manipulating lists
{1, 2, 3, 4, 5}                    # {1, 2, 3, 4, 5}
Length[{1, 2, 3, 4, 5}]           # 5
Head[{10, 20, 30}]                # 10
Tail[{10, 20, 30}]                # {20, 30}
Append[{1, 2, 3}, 4]              # {1, 2, 3, 4}
```

### String Manipulation
```lyra
# String operations
"Hello, World!"                        # "Hello, World!"
StringLength["Hello"]                   # 5
StringJoin["Hello", " ", "World"]       # "Hello World"  
StringTake["Programming", 4]            # "Prog"
StringDrop["Programming", 4]            # "ramming"
```

### Complex Expressions
```lyra
# Combining operations
Length[Append[{1, 2, 3}, Sqrt[16]]]           # 4
StringLength[StringJoin["Result: ", "42"]]     # 10
Sin[3.14159/4]^2 + Cos[3.14159/4]^2          # 1.0
((2 + 3) * (4 + 5)) / ((6 + 7) - (8 - 9))   # 3.214286
```

### Getting Started
See the `examples/` directory for complete working examples:
- `examples/basic_arithmetic.lyra` - Basic math operations  
- `examples/mathematical_functions.lyra` - Trigonometric and exponential functions
- `examples/list_operations.lyra` - List manipulation examples
- `examples/string_operations.lyra` - String processing examples
- `examples/complex_expressions.lyra` - Advanced nested expressions

## Testing and Quality Assurance

Lyra follows strict Test-Driven Development practices with comprehensive testing:

### Running Tests
```bash
# Run all tests (237+ unit tests + integration tests)
cargo test

# Run specific test suites
cargo test lexer     # Lexer tests
cargo test parser    # Parser tests
cargo test compiler  # Compiler tests
cargo test vm        # VM tests
cargo test stdlib    # Standard library tests

# Run snapshot tests (regression protection)
cargo test snapshot

# Check code formatting and lints
cargo fmt --check
cargo clippy -- -D warnings
```

### Performance Benchmarks
```bash
# Run performance benchmarks
cargo bench

# View benchmark reports (generates HTML reports)
open target/criterion/report/index.html
```

### Code Quality
- **237+ comprehensive tests** covering all major components
- **Snapshot testing** with insta for regression protection
- **Performance benchmarks** with Criterion 
- **100% clippy clean** with strict warnings
- **Formatted code** with rustfmt
- **Enhanced error messages** with line numbers and suggestions

## Contributing

This project follows strict Test-Driven Development practices. See [CLAUDE.md](CLAUDE.md) for detailed development guidelines.

### Development Workflow
1. Write failing tests first (RED)
2. Implement minimal code to pass tests (GREEN)  
3. Refactor while keeping tests green (REFACTOR)
4. Ensure all quality gates pass before committing

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Inspired by the Wolfram Language and Mathematica
- Built with ❤️ in Rust