# Contributing to Lyra

Thank you for your interest in contributing to Lyra! This document provides guidelines and information for contributors.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Process](#development-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for all contributors. By participating in this project, you agree to abide by our community standards:

- **Be respectful** to all community members
- **Be constructive** in discussions and feedback
- **Be collaborative** and open to different perspectives
- **Be patient** with new contributors
- **Report inappropriate behavior** to the maintainers

## Getting Started

### Prerequisites

- **Rust 1.70+** with `cargo`, `rustfmt`, and `clippy`
- **Git** for version control
- **Familiarity** with Rust, symbolic computation, or language implementation (helpful but not required)

### Development Setup

1. **Fork and clone** the repository:
```bash
git clone https://github.com/YOUR_USERNAME/lyra.git
cd lyra
git remote add upstream https://github.com/parkerdgabel/lyra.git
```

2. **Install development dependencies**:
```bash
cargo install cargo-criterion cargo-tarpaulin
```

3. **Verify your setup**:
```bash
cargo test
cargo bench --no-run
cargo fmt --check
cargo clippy -- -D warnings
```

### Project Structure

```
lyra/
├── src/
│   ├── ast.rs          # Abstract Syntax Tree definitions
│   ├── compiler.rs     # Bytecode compiler
│   ├── lexer.rs        # Tokenization
│   ├── parser.rs       # AST generation
│   ├── vm.rs          # Virtual machine
│   ├── stdlib/        # Standard library modules
│   ├── repl/          # REPL implementation
│   └── types/         # Type system
├── tests/             # Integration tests
├── benches/           # Performance benchmarks
├── examples/          # Language examples
└── docs/             # Documentation
```

## Development Process

### Test-Driven Development (TDD) - MANDATORY

**ALL changes MUST follow strict TDD practices:**

1. **RED**: Write a failing test first
2. **GREEN**: Write minimal code to make the test pass  
3. **REFACTOR**: Clean up code while keeping tests green

```bash
# Write your test first
cargo test your_new_test -- --nocapture

# Implement the feature
# Edit source files...

# Verify the test passes
cargo test your_new_test

# Run all tests
cargo test
```

### Quality Gates - ZERO TOLERANCE

Before ANY commit, ALL of these MUST pass:

```bash
# 1. All tests pass
cargo test

# 2. Code is formatted
cargo fmt

# 3. No clippy warnings
cargo clippy -- -D warnings

# 4. Benchmarks compile
cargo bench --no-run
```

**We have ZERO TOLERANCE for compilation errors or warnings.**

### Branching Strategy

- **`main`** - Production-ready code, always stable
- **`develop`** - Integration branch for new features
- **Feature branches** - `feature/your-feature-name`
- **Bugfix branches** - `bugfix/issue-description`
- **Documentation** - `docs/topic-name`

### Development Workflow

1. **Create a feature branch**:
```bash
git checkout -b feature/your-feature-name
```

2. **Write tests first** (TDD requirement):
```bash
# Add test to appropriate test file
# tests/ for integration tests
# src/ for unit tests
```

3. **Implement your changes**:
```bash
# Edit source files
# Follow existing code style
# Add documentation
```

4. **Verify quality gates**:
```bash
make check  # or run commands manually
```

5. **Commit with descriptive message**:
```bash
git commit -m "feat: add symbolic differentiation for polynomials

- Implement derivative calculation for polynomial expressions
- Add comprehensive test coverage for edge cases
- Update documentation with usage examples
- Resolves #123"
```

## Testing Guidelines

### Test Categories

#### Unit Tests
- **Location**: Same file as implementation with `#[cfg(test)]`
- **Purpose**: Test individual functions and components
- **Coverage**: Every function must have unit tests

#### Integration Tests  
- **Location**: `tests/` directory
- **Purpose**: Test component interactions
- **Naming**: `test_feature_name.rs`

#### Snapshot Tests
- **Tool**: `insta` crate for output verification
- **Purpose**: Complex output testing
- **Update**: `cargo insta review`

#### Property Tests
- **Tool**: `quickcheck` or custom generators
- **Purpose**: Test with randomized inputs
- **Coverage**: Mathematical operations, edge cases

#### Performance Tests
- **Location**: `benches/` directory
- **Purpose**: Regression detection and optimization validation
- **Requirement**: New features need performance benchmarks

### Test Writing Standards

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_function_name_normal_case() {
        // Arrange
        let input = create_test_input();
        let expected = expected_output();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected);
    }
    
    #[test]
    fn test_function_name_edge_case() {
        // Test edge cases like empty input, boundary values, etc.
    }
    
    #[test] 
    #[should_panic(expected = "specific error message")]
    fn test_function_name_error_case() {
        // Test error conditions
    }
}
```

## Documentation Standards

### Documentation Requirements

- **All public APIs** must have comprehensive documentation
- **Examples** must be included for all public functions
- **Module-level docs** for each module explaining purpose
- **Type documentation** for complex types and Foreign objects

### Documentation Style

```rust
/// Brief one-line summary of what the function does.
/// 
/// More detailed explanation of the function's purpose,
/// behavior, and any important details.
/// 
/// # Arguments
/// 
/// * `param_name` - Description of parameter
/// * `other_param` - Description of other parameter
/// 
/// # Returns
/// 
/// Description of return value
/// 
/// # Examples
/// 
/// ```
/// use lyra::function_name;
/// 
/// let result = function_name(42);
/// assert_eq!(result, expected_value);
/// ```
/// 
/// # Panics
/// 
/// Description of when this function panics (if applicable)
/// 
/// # Errors
/// 
/// Description of error conditions (if applicable)
pub fn function_name(param_name: Type) -> ReturnType {
    // implementation
}
```

### README Updates

When adding new features, update relevant documentation:
- **README.md** - Major features or changes to getting started
- **docs/language-reference.md** - Language syntax changes
- **examples/** - Add examples demonstrating new features
- **CHANGELOG.md** - Document changes (see next section)

## Code Style

### Rust Style Guide

Follow the [official Rust Style Guide](https://doc.rust-lang.org/style-guide/):

```bash
# Format all code
cargo fmt

# Check style issues
cargo fmt -- --check
```

### Lyra-Specific Conventions

#### Naming
- **Functions**: `snake_case` for Rust, `PascalCase` for Wolfram-style functions
- **Variables**: `snake_case`
- **Constants**: `SCREAMING_SNAKE_CASE`
- **Types**: `PascalCase`
- **Modules**: `snake_case`

#### Code Organization
- **One concept per file** when possible
- **Related functionality grouped** in modules
- **Tests alongside implementation** in same file with `#[cfg(test)]`
- **Foreign objects** in separate files when complex

#### Error Handling
```rust
// Use Result<T, E> for fallible operations
pub fn fallible_operation() -> VmResult<Value> {
    // implementation
}

// Use descriptive error messages
return Err(VmError::TypeError { 
    expected: "integer".to_string(), 
    actual: format!("{:?}", value) 
});
```

#### Performance Considerations
- **Avoid unnecessary allocations** in hot paths
- **Use `&str` instead of `String`** when possible
- **Profile before optimizing** complex code
- **Document performance characteristics** of algorithms

### Foreign Object Pattern

When adding complex functionality:

```rust
// Foreign objects isolate complexity from VM core
#[derive(Debug, Clone)]
pub struct MyComplexType {
    // complex state
}

impl Foreign for MyComplexType {
    fn type_name(&self) -> &'static str {
        "MyComplexType"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        // method implementations
    }
    
    // ... other required methods
}
```

## Submitting Changes

### Pull Request Process

1. **Ensure your branch is up to date**:
```bash
git fetch upstream
git rebase upstream/main
```

2. **Create a pull request**:
   - Use descriptive title: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`
   - Reference related issues: "Closes #123"
   - Describe changes and motivation
   - Include testing information

3. **PR Template** (automatically added):
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Benchmarks updated if performance changes

## Documentation
- [ ] Documentation updated for new features
- [ ] Examples added or updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** in integration environment
4. **Documentation review** for clarity
5. **Final approval** and merge

### Commit Message Standards

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

body

footer
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Adding or fixing tests
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

Examples:
```
feat(stdlib): add symbolic differentiation support

Implement automatic differentiation for polynomial expressions
with support for multivariable calculus.

Closes #123
```

## Community

### Communication Channels

- **GitHub Issues** - Bug reports, feature requests
- **GitHub Discussions** - General questions, ideas, showcase
- **Pull Requests** - Code contributions and reviews
- **Email** - lyra-lang@example.com for private inquiries

### Getting Help

- **Documentation** - Check docs/ first
- **Examples** - Look at examples/ for usage patterns
- **Issues** - Search existing issues before creating new ones
- **Discussions** - Ask questions in GitHub Discussions

### Recognition

Contributors are recognized in:
- **AUTHORS.md** - All contributors
- **Release notes** - Major contributions
- **Documentation** - Contributors to specific features

## Areas for Contribution

### Beginner Friendly
- **Documentation improvements**
- **Example programs**
- **Test coverage expansion**
- **Bug fixes with clear reproduction steps**

### Intermediate
- **Standard library functions**
- **REPL enhancements**
- **Performance optimizations**
- **IDE integration**

### Advanced
- **VM optimizations**
- **Compiler enhancements**
- **Type system improvements**
- **Concurrency features**

### Current Priorities

Check [GitHub Issues](https://github.com/parkerdgabel/lyra/issues) with labels:
- `good first issue` - Beginner friendly
- `help wanted` - Community contributions welcome
- `priority: high` - Important for next release

## Questions?

Don't hesitate to ask! We're here to help new contributors succeed:

- **Open a Discussion** for questions
- **Join the conversation** on existing issues
- **Reach out** to maintainers directly

Thank you for contributing to Lyra! 🚀