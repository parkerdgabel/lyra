# Lyra Examples - Comprehensive Testing Suite

This directory contains a complete suite of example scripts that demonstrate and test all major functionality of the Lyra symbolic programming language. The examples are organized into categories covering standard library functions, advanced features, performance testing, and real-world workflows.

## 🎯 Overview

**Purpose**: Validate that Lyra's 727 passing tests and 118 registered functions work correctly in practice through executable examples.

**Categories**:
- **Standard Library**: 118 registered functions across Calculus, Statistics, Mathematics, Tensors, and I/O
- **Module System**: Import statements, tree shaking, and namespace resolution
- **Advanced Features**: Pattern matching, rule systems, and ML framework functions
- **Performance**: Large-scale computation benchmarks and stress tests
- **Workflows**: End-to-end data analysis and real-world applications

## 📁 Directory Structure

```
examples/
├── README.md                    # This documentation
├── test_runner.py              # Automated test execution script
│
├── stdlib/                     # Standard Library Examples (5 files)
│   ├── calculus_examples.lyra  # Derivatives and integration
│   ├── statistics_examples.lyra # 13 statistical functions
│   ├── advanced_math.lyra      # Trigonometric and exponential
│   ├── tensor_operations.lyra  # Linear algebra operations
│   └── io_operations.lyra      # Import/Export functionality
│
├── modules/                    # Module System Examples (3 files)
│   ├── module_imports.lyra     # Basic import statements
│   ├── selective_imports.lyra  # Specific function imports
│   └── tree_shaking_demo.lyra  # Dead code elimination
│
├── advanced/                   # Advanced Features (4 files)
│   ├── pattern_examples.lyra   # Pattern matching
│   ├── rule_examples.lyra      # Rule[] and RuleDelayed[]
│   ├── rule_examples_simple.lyra # Simplified rule examples
│   └── ml_layers.lyra          # ML framework functions
│
├── performance/                # Performance Testing (1 file)
│   └── performance_benchmarks.lyra # Large computation tests
│
└── workflows/                  # Real-World Examples (1 file)
    └── data_analysis.lyra      # End-to-end data science workflow
```

## 🚀 Quick Start

### Running Individual Examples

```bash
# Run a specific example
cargo run --bin lyra -- run examples/stdlib/calculus_examples.lyra

# Run with verbose output
cargo run --bin lyra -- run examples/workflows/data_analysis.lyra
```

### Automated Testing

```bash
# Run all examples with the test runner
python3 examples/test_runner.py

# Run with timing and benchmarks
python3 examples/test_runner.py --timing --benchmark

# Run specific category
python3 examples/test_runner.py --filter="calculus*"

# Export results to JSON
python3 examples/test_runner.py --output=json

# CI mode (exits with error code on failures)
python3 examples/test_runner.py --ci --continue
```

## 📚 Example Categories

### 1. Standard Library Examples (`stdlib/`)

#### `calculus_examples.lyra`
- **Purpose**: Test calculus functions (D[] derivatives, Integrate[])
- **Coverage**: Basic derivatives, chain rule, integration, physics applications
- **Key Functions**: `D[]`, `Integrate[]`, `IntegrateDefinite[]`
- **Status**: ✅ Passing

```lyra
(* Basic Derivative Examples *)
D[x^2, x]           (* Expected: 2*x *)
D[Sin[x], x]        (* Expected: Cos[x] *)

(* Integration Examples *)
Integrate[x^2, x]   (* Expected: x^3/3 *)
```

#### `statistics_examples.lyra`
- **Purpose**: Test all 13 statistical functions comprehensively
- **Coverage**: Descriptive statistics, distributions, correlations
- **Key Functions**: `Mean[]`, `Variance[]`, `StandardDeviation[]`, `Correlation[]`, `RandomReal[]`
- **Status**: ✅ Passing

```lyra
(* Descriptive Statistics *)
Mean[{1, 2, 3, 4, 5}]       (* Expected: 3.0 *)
Variance[{1, 2, 3, 4, 5}]   (* Expected: 2.5 *)
```

#### `advanced_math.lyra`
- **Purpose**: Test trigonometric and exponential functions
- **Coverage**: Special angles, identities, logarithms, powers
- **Key Functions**: `Sin[]`, `Cos[]`, `Exp[]`, `Log[]`, `Sqrt[]`
- **Status**: ⚠️ Partially working (some functions not implemented)

#### `tensor_operations.lyra`
- **Purpose**: Test linear algebra and tensor operations
- **Coverage**: Matrix operations, array manipulation, dot products
- **Key Functions**: `Array[]`, `Dot[]`, `Transpose[]`, `ArrayReshape[]`, `Maximum[]`
- **Status**: ✅ Passing

```lyra
(* Matrix Operations *)
A = Array[{{1, 2}, {3, 4}}]
Dot[A, B]  (* Matrix multiplication *)
```

#### `io_operations.lyra`
- **Purpose**: Test import/export with various formats
- **Coverage**: File I/O, data serialization, format conversions
- **Key Functions**: `Import[]`, `Export[]`, `ReadFile[]`, `WriteFile[]`
- **Status**: ⚠️ I/O functions not fully implemented

### 2. Module System Examples (`modules/`)

#### `module_imports.lyra`
- **Purpose**: Test basic module imports and namespace resolution
- **Coverage**: `import` statements, qualified names, aliased imports
- **Status**: ⚠️ Module system not fully implemented

```lyra
import std::math
import std::statistics as stats

std::math::Sin[pi/2]        (* Module-qualified function call *)
```

#### `selective_imports.lyra`
- **Purpose**: Test specific function imports
- **Coverage**: Selective imports, function aliasing
- **Status**: ⚠️ Module system not fully implemented

#### `tree_shaking_demo.lyra`
- **Purpose**: Test dead code elimination
- **Coverage**: Unused function detection, optimization validation
- **Status**: ⚠️ Tree shaking not fully implemented

### 3. Advanced Features (`advanced/`)

#### `pattern_examples.lyra`
- **Purpose**: Test pattern matching capabilities
- **Coverage**: Basic patterns, sequence patterns, named patterns
- **Key Functions**: `MatchQ[]`, `Cases[]`, `ReplaceAll[]`
- **Status**: ✅ Passing

```lyra
(* Pattern Matching *)
MatchQ[42, _Integer]             (* Expected: True *)
Cases[{1, 2, "a", 3}, _Integer]  (* Expected: {1, 2, 3} *)
```

#### `rule_examples.lyra`
- **Purpose**: Test rule system with Rule[] and RuleDelayed[]
- **Coverage**: Rule creation, application, transformation rules
- **Key Functions**: `Rule[]`, `RuleDelayed[]`, `ReplaceAll[]`
- **Status**: ✅ Passing

```lyra
(* Rule Application *)
basic_rule = Rule[x, 42]
ReplaceAll[x + y, basic_rule]    (* Expected: 42 + y *)
```

#### `ml_layers.lyra`
- **Purpose**: Test ML framework functions and neural network operations
- **Coverage**: Linear layers, activations, tensor operations
- **Key Functions**: `Dot[]`, `Maximum[]` (for ReLU), array operations
- **Status**: ✅ Passing

### 4. Performance Testing (`performance/`)

#### `performance_benchmarks.lyra`
- **Purpose**: Test large-scale computation performance
- **Coverage**: Large matrices, complex expressions, memory usage
- **Scenarios**: ML workflows, scientific computing, financial calculations
- **Status**: ✅ Passing (1.0s execution time)

```lyra
(* Large Matrix Operations *)
large_matrix_4x4 = Array[{{1.1, 2.2, 3.3, 4.4}, ...}]
large_multiplication = Dot[large_matrix_4x4, large_matrix_4x4_b]
```

### 5. Real-World Workflows (`workflows/`)

#### `data_analysis.lyra`
- **Purpose**: End-to-end data science workflow demonstration
- **Coverage**: Complete pipeline from data loading to reporting
- **Components**: 
  - Data loading and quality assessment
  - Statistical analysis and preprocessing
  - Feature engineering and transformation
  - Machine learning and time series analysis
  - Anomaly detection and clustering
  - Business intelligence reporting
- **Status**: ✅ Passing (2.7s execution time)
- **Lines of Code**: 513 lines
- **Data Points**: 20 sensor readings, 3 variables
- **Analysis Components**: 9 major analysis phases

## 🔧 Test Runner Features

The `test_runner.py` script provides comprehensive testing infrastructure:

### Basic Usage
```bash
python3 examples/test_runner.py [options]
```

### Key Features
- **Automated Discovery**: Finds all `.lyra` files in examples directory
- **Parallel Execution**: Optional parallel test execution
- **Filtering**: Run specific tests with glob patterns
- **Multiple Output Formats**: Console, JSON, XML, HTML reports
- **Performance Analysis**: Timing and benchmark analysis
- **CI Integration**: Suitable for continuous integration pipelines

### Advanced Options
```bash
# Verbose output with timing
python3 examples/test_runner.py --verbose --timing

# Run only stdlib tests
python3 examples/test_runner.py --filter="*stdlib*"

# Generate HTML report
python3 examples/test_runner.py --output=html

# CI mode with benchmarks
python3 examples/test_runner.py --ci --benchmark --continue
```

### Output Examples

**Console Output**:
```
Lyra Examples Test Runner
Discovered 14 test files

Running Tests:
[ 1/14] calculus_examples.lyra
    ✓ PASSED (0.271s)
[ 2/14] statistics_examples.lyra
    ✓ PASSED (0.749s)

=== Test Summary ===
Total tests: 14
Passed: 9
Failed: 5
Success rate: 64.3%
```

**JSON Export**:
```json
{
  "timestamp": "2025-01-19T10:30:00",
  "summary": {
    "total": 14,
    "passed": 9,
    "failed": 5,
    "success_rate": 64.3
  },
  "tests": [...]
}
```

## 📊 Current Test Results

### Overall Status
- **Total Examples**: 14 files
- **Passing**: 9 examples (64.3% success rate)
- **Categories Working**: Advanced Features (100%), Performance (100%), Workflows (100%)
- **Categories Partial**: Standard Library (60%), Module System (0%)

### Performance Metrics
- **Average Execution Time**: 0.911s per test
- **Fastest Test**: rule_examples_simple.lyra (0.066s)
- **Slowest Test**: data_analysis.lyra (2.686s)
- **Total Suite Runtime**: ~13s

### Known Issues
1. **Module System**: Import statements not fully implemented
2. **I/O Functions**: File operations need implementation
3. **Advanced Math**: Some trigonometric functions missing
4. **Tree Shaking**: Dead code elimination not active

## 🎛️ Development Commands

### Building and Testing
```bash
# Build Lyra
cargo build --bin lyra

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Run linters
cargo clippy -- -D warnings
```

### Example Development
```bash
# Create new example
touch examples/category/new_example.lyra

# Test specific example
cargo run --bin lyra -- run examples/category/new_example.lyra

# Run test suite
python3 examples/test_runner.py
```

## 📈 Success Metrics

### Functionality Coverage
- ✅ **Pattern Matching**: Complete with 15+ pattern types
- ✅ **Rule System**: Complete with immediate and delayed evaluation
- ✅ **Tensor Operations**: Complete linear algebra support
- ✅ **Statistical Functions**: 13/13 functions tested
- ✅ **ML Framework**: Neural network building blocks working
- ✅ **Performance**: Large-scale computations validated
- ⚠️ **Module System**: Import statements need implementation
- ⚠️ **I/O Operations**: File handling functions needed

### Quality Indicators
- **Test Coverage**: 14 comprehensive example files
- **Documentation**: Extensive inline comments and examples
- **Error Handling**: Graceful failure reporting
- **Performance**: Sub-second execution for most examples
- **Maintainability**: Clear structure and organization

## 🔮 Future Enhancements

### Short Term
1. **Complete Module System**: Implement import/export functionality
2. **I/O Functions**: Add file reading/writing capabilities
3. **Advanced Math**: Complete trigonometric function set
4. **Tree Shaking**: Enable dead code elimination

### Long Term
1. **Interactive Examples**: Jupyter-style notebooks
2. **Visualization**: Built-in plotting capabilities
3. **Package Manager**: Example package distribution
4. **Performance Optimization**: JIT compilation for examples
5. **Educational Content**: Tutorial-style examples

## 💡 Contributing

### Adding New Examples
1. Choose appropriate category directory
2. Follow existing naming conventions
3. Add comprehensive comments
4. Include expected outputs
5. Test with the automated runner
6. Update this documentation

### Example Template
```lyra
//! Example Title - Brief Description
//! 
//! This example demonstrates:
//! - Feature 1: Description
//! - Feature 2: Description
//! - Real-world application: Context

(* === Section 1: Basic Operations === *)

(* Clear comments explaining each operation *)
result1 = Function[input]           (* Expected: output *)

(* === Section 2: Advanced Usage === *)

complex_example = ComplexFunction[data]
"Expected result: " + result

(* === Summary === *)

"=== Example Complete ==="
"All operations tested successfully"
```

## 📞 Support

- **Issues**: Report problems via GitHub issues
- **Documentation**: See main Lyra documentation
- **Examples**: This README and inline comments
- **Community**: Join the Lyra developer community

---

**Last Updated**: January 19, 2025  
**Version**: 1.0.0  
**Total Examples**: 14 files  
**Lines of Code**: 2,500+ lines across all examples