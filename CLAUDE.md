# Lyra Development Guidelines

## Test-Driven Development (TDD) Requirements

**CRITICAL: ALL changes to this codebase MUST follow strict Test-Driven Development practices.**

### TDD Process
1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to make the test pass
3. **REFACTOR**: Clean up code while keeping tests green

### Before Making Any Code Changes
- [ ] Write comprehensive tests that describe the expected behavior
- [ ] Ensure tests fail initially (RED phase)
- [ ] Implement only enough code to make tests pass (GREEN phase)
- [ ] Refactor if needed while maintaining green tests

### Test Requirements
- **Unit Tests**: Every function/method must have unit tests
- **Integration Tests**: Components must be tested together
- **Snapshot Tests**: Use `insta` for complex output verification
- **Property Tests**: Where applicable, test with random inputs

### Running Tests
```bash
# Run all tests
cargo test

# Run tests with coverage
cargo test --all-features

# Update snapshots
cargo insta review
```

### Development Commands
```bash
# Check formatting
cargo fmt --check

# Run clippy lints
cargo clippy -- -D warnings

# Build project
cargo build

# Run CLI
cargo run -- --help
```

### Architecture Overview

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

### Module Structure
- `lexer`: Tokenization of source code
- `parser`: AST generation from tokens
- `ast`: AST node definitions and utilities
- `compiler`: Bytecode generation from AST
- `vm`: Virtual machine for bytecode execution
- `runtime`: Built-in functions and standard library
- `error`: Error types and handling

### Language Syntax (Wolfram-Inspired)
- Function calls: `f[x, y]`
- Lists: `{1, 2, 3}`
- Patterns: `x_`, `x__`, `x_Integer`
- Rules: `x -> x^2`, `x :> RandomReal[]`
- Replacement: `expr /. rule`
- Definitions: `f[x_] = x^2`, `f[x_] := RandomReal[]`

### Test Organization
```
tests/
├── lexer/          # Lexer unit tests
├── parser/         # Parser unit tests
├── compiler/       # Compiler unit tests
├── vm/             # VM unit tests
├── integration/    # End-to-end tests
└── snapshots/      # Insta snapshot files
```

### Quality Gates
Before any commit:
1. All tests must pass (`cargo test`)
2. Code must be formatted (`cargo fmt`)
3. No clippy warnings (`cargo clippy`)
4. Documentation must be updated if APIs change

### Performance Guidelines
- Optimize only after correctness is established
- Use benchmarks to measure performance improvements
- Profile before optimizing
- Maintain test coverage during optimization

### Error Handling
- Use `Result<T, Error>` for fallible operations
- Provide clear error messages with context
- Test error cases thoroughly
- Use `thiserror` for error definitions

Remember: **Tests first, implementation second, always verify tests pass before proceeding.**