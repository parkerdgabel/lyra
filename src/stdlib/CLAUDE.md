# Lyra Standard Library

## Overview

The Lyra Standard Library provides a comprehensive set of functions for symbolic computation, tensor operations, and mathematical computations. All functions are designed to be compatible with Wolfram Language syntax and semantics.

## Module Structure

```
src/stdlib/
├── mod.rs              # Standard library registry and exports
├── list.rs             # List operations (Length, Head, Tail, Map, Apply)
├── string.rs           # String operations (StringJoin, StringLength, etc.)
├── math.rs             # Mathematical functions (Sin, Cos, Exp, etc.)
├── rules.rs            # Pattern matching and replacement rules
├── tensor.rs           # Tensor operations and linear algebra
├── sparse/             # Sparse matrix operations
│   ├── core.rs         # Sparse matrix core infrastructure
│   ├── coo.rs          # Coordinate (COO) format implementation
│   └── operations.rs   # Matrix arithmetic (add, multiply, transpose)
├── spatial/            # Spatial data structures and algorithms
│   ├── core.rs         # Spatial indexing infrastructure
│   ├── kdtree.rs       # K-Dimensional Tree for low-dimensional spaces
│   └── balltree.rs     # Ball Tree for high-dimensional spaces
└── CLAUDE.md           # This documentation file
```

## Function Categories

### 1. List Operations (7 functions)
- `Length[list]` - Get the length of a list
- `Head[list]` - Get the first element
- `Tail[list]` - Get all elements except the first
- `Append[list, element]` - Add element to the end
- `Flatten[list]` - Flatten nested lists
- `Map[function, list]` - Apply function to each element
- `Apply[function, arguments]` - Apply function to arguments

### 2. String Operations (4 functions)
- `StringJoin[str1, str2, ...]` - Concatenate strings
- `StringLength[string]` - Get string length
- `StringTake[string, n]` - Take first n characters
- `StringDrop[string, n]` - Drop first n characters

### 3. Mathematical Functions (6 functions)
- `Sin[x]`, `Cos[x]`, `Tan[x]` - Trigonometric functions
- `Exp[x]`, `Log[x]` - Exponential and logarithmic functions
- `Sqrt[x]` - Square root

### 4. Rule Operations (3 functions)
- `Rule[lhs, rhs]` - Create replacement rule
- `RuleDelayed[lhs, rhs]` - Create delayed replacement rule
- `ReplaceAll[expr, rule]` - Apply replacement rules

### 5. Sparse Matrix Operations (7 functions)
- `SparseAdd[matrix1, matrix2]` - Add two sparse matrices
- `SparseMultiply[matrix1, matrix2]` - Multiply two sparse matrices  
- `SparseTranspose[matrix]` - Transpose a sparse matrix
- `SparseShape[matrix]` - Get matrix dimensions
- `SparseNNZ[matrix]` - Get number of non-zero elements
- `SparseDensity[matrix]` - Get matrix density (nnz / total_elements)
- `SparseToDense[matrix]` - Convert to dense representation

### 6. Spatial Data Structures (2 functions)
- `KDTree[points, metric, leafSize]` - Create K-Dimensional Tree for nearest neighbor searches
- `BallTree[points, metric, leafSize]` - Create Ball Tree for high-dimensional spaces

### 7. Tensor Operations (8 functions)

#### Basic Tensor Operations (5 functions)
- `Array[list]` - Create tensor from nested lists
- `ArrayDimensions[tensor]` - Get tensor dimensions
- `ArrayRank[tensor]` - Get number of dimensions
- `ArrayReshape[tensor, shape]` - Reshape tensor
- `ArrayFlatten[tensor]` - Flatten to 1D

#### Linear Algebra Operations (3 functions)
- `Dot[a, b]` - Matrix/vector multiplication
- `Transpose[matrix]` - Matrix transpose
- `Maximum[a, b]` - Element-wise maximum (for ReLU)

## Tensor Operations Detailed Guide

### Creating Tensors

```wolfram
(* 1D tensor from list *)
Array[{1, 2, 3, 4}]

(* 2D tensor from nested lists *)
Array[{{1, 2}, {3, 4}}]

(* 3D tensor *)
Array[{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}]
```

### Tensor Information

```wolfram
tensor = Array[{{1, 2, 3}, {4, 5, 6}}]
ArrayDimensions[tensor]  (* → {2, 3} *)
ArrayRank[tensor]        (* → 2 *)
```

### Tensor Manipulation

```wolfram
(* Reshape 2x3 matrix to 3x2 *)
tensor = Array[{{1, 2, 3}, {4, 5, 6}}]
ArrayReshape[tensor, {3, 2}]  (* → {{1, 2}, {3, 4}, {5, 6}} *)

(* Flatten to 1D *)
ArrayFlatten[tensor]  (* → {1, 2, 3, 4, 5, 6} *)
```

### Linear Algebra Operations

#### Matrix Multiplication
```wolfram
(* Vector dot product *)
Dot[{1, 2, 3}, {4, 5, 6}]  (* → 32 *)

(* Matrix-vector multiplication *)
matrix = Array[{{1, 2}, {3, 4}}]
vector = Array[{1, 0}]
Dot[matrix, vector]  (* → {1, 3} *)

(* Matrix-matrix multiplication *)
A = Array[{{1, 2}, {3, 4}}]
B = Array[{{5, 6}, {7, 8}}]
Dot[A, B]  (* → {{19, 22}, {43, 50}} *)
```

#### Transpose Operations
```wolfram
(* 2D matrix transpose *)
matrix = Array[{{1, 2, 3}, {4, 5, 6}}]
Transpose[matrix]  (* → {{1, 4}, {2, 5}, {3, 6}} *)

(* 1D vector becomes column vector *)
vector = Array[{1, 2, 3}]
Transpose[vector]  (* → {{1}, {2}, {3}} *)

(* Row vector to column vector *)
row = Array[{{1, 2, 3}}]  (* 1x3 *)
Transpose[row]  (* → {{1}, {2}, {3}} - 3x1 *)
```

#### Element-wise Maximum (ReLU Activation)
```wolfram
(* Basic ReLU operation *)
input = Array[{-2, -1, 0, 1, 2}]
Maximum[input, 0]  (* → {0, 0, 0, 1, 2} *)

(* 2D tensor ReLU *)
matrix = Array[{{-1, 0}, {1, 2}}]
Maximum[matrix, 0]  (* → {{0, 0}, {1, 2}} *)

(* Broadcasting with different shapes *)
matrix = Array[{{1, 2}, {3, 4}}]
vector = Array[{0, 1}]
Maximum[matrix, vector]  (* → {{1, 2}, {3, 4}} *)
```

## Neural Network Building Blocks

The tensor operations provide the foundation for neural networks:

### 1. Linear Layer Forward Pass
```wolfram
(* weights: [input_size, output_size], input: [input_size] *)
weights = Array[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}]  (* 3x2 *)
bias = Array[{0.1, 0.2}]  (* 2 *)
input = Array[{1.0, 2.0, 3.0}]  (* 3 *)

(* Forward pass: output = weights^T * input + bias *)
linear_output = Dot[Transpose[weights], input] + bias
```

### 2. ReLU Activation
```wolfram
(* Apply ReLU to any tensor *)
activated = Maximum[linear_output, 0]
```

### 3. Matrix Operations for Backpropagation
```wolfram
(* Gradient computation uses transpose and dot products *)
gradient = Dot[Transpose[weights], error_signal]
```

## Broadcasting Rules

Lyra implements NumPy-style broadcasting for tensor operations:

1. **Same Shape**: Direct element-wise operation
2. **Scalar + Tensor**: Scalar is broadcast to tensor shape
3. **Compatible Shapes**: Dimensions are aligned from the right
   - `[2, 3] + [3]` → `[2, 3]` (vector broadcast across rows)
   - `[2, 1] + [3]` → `[2, 3]` (both dimensions broadcast)

### Broadcasting Examples
```wolfram
(* Scalar broadcasting *)
Array[{{1, 2}, {3, 4}}] + 10  (* → {{11, 12}, {13, 14}} *)

(* Vector broadcasting *)
matrix = Array[{{1, 2, 3}, {4, 5, 6}}]  (* 2x3 *)
vector = Array[{10, 20, 30}]            (* 3 *)
matrix + vector  (* → {{11, 22, 33}, {14, 25, 36}} *)
```

## Error Handling

All functions provide comprehensive error checking:

- **Type Errors**: Incompatible argument types
- **Shape Errors**: Incompatible tensor shapes for operations
- **Argument Count**: Wrong number of arguments
- **Division by Zero**: Arithmetic safety checks

## Performance Characteristics

- **Memory Efficient**: Uses ndarray backend for zero-copy operations
- **SIMD Optimized**: Leverages Rust's ndarray SIMD capabilities
- **Broadcasting**: Efficient broadcasting without memory duplication
- **Type Safety**: Compile-time guarantees for tensor operations

## Integration with VM

All tensor operations integrate seamlessly with Lyra's virtual machine:

```wolfram
(* Complex expressions work naturally *)
result = Dot[Transpose[Array[{{1, 2}, {3, 4}}]], Array[{1, 0}]]
Maximum[result, 0]  (* Chained operations *)
```

## Week 7 Enhancements: New Functionality

### Enhanced Map/Apply Functions

The Map and Apply functions have been completely reimplemented to support both built-in functions and pure functions:

```wolfram
(* Map with built-in functions *)
Map[Sin, {0, 1.57, 3.14}]  (* → {0, 0.99999, -0.00001} *)
Map[Plus[1], {1, 2, 3}]    (* → {2, 3, 4} *)
Map[Length, {{1, 2}, {3, 4, 5}}]  (* → {2, 3} *)

(* Map with pure functions *)
Map[#^2 &, {1, 2, 3, 4}]   (* → {1, 4, 9, 16} *)
Map[# + 10 &, {1, 2, 3}]   (* → {11, 12, 13} *)

(* Apply operations *)
Apply[Plus, {1, 2, 3, 4, 5}]  (* → 15 *)
Apply[Times, {2, 3, 4}]       (* → 24 *)
Apply[Max, {5, 2, 8, 1}]      (* → 8 *)
```

### Sparse Matrix Operations

Complete sparse matrix infrastructure with COO format and basic operations:

```wolfram
(* Create sparse matrices from triplets *)
matrix1 = SparseFromTriplets[{{0, 0, 1.0}, {1, 1, 2.0}}, {2, 2}]
matrix2 = SparseFromTriplets[{{0, 1, 3.0}, {1, 0, 4.0}}, {2, 2}]

(* Matrix operations *)
sum = SparseAdd[matrix1, matrix2]
product = SparseMultiply[matrix1, matrix2]
transposed = SparseTranspose[matrix1]

(* Matrix properties *)
SparseShape[matrix1]     (* → {2, 2} *)
SparseNNZ[matrix1]       (* → 2 *)
SparseDensity[matrix1]   (* → 0.5 *)

(* Convert to dense *)
dense = SparseToDense[matrix1]  (* → {{1.0, 0.0}, {0.0, 2.0}} *)
```

### Spatial Data Structures

High-performance spatial indexing for nearest neighbor searches:

```wolfram
(* Create KD-Tree for low-dimensional data *)
points2D = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}, {7.0, 8.0}}
kdtree = KDTree[points2D, "Euclidean", 10]

(* Create Ball Tree for high-dimensional data *)
pointsHD = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
balltree = BallTree[pointsHD, "Euclidean", 5]

(* Spatial queries (via method calls) *)
kdtree.KNearestNeighbors[{2.0, 3.0}, 2]     (* Find 2 closest points *)
kdtree.RadiusNeighbors[{2.0, 3.0}, 3.0]     (* Find points within radius *)
kdtree.RangeQuery[{0, 0}, {4, 4}]           (* Find points in rectangle *)
```

## Testing Coverage

The standard library has comprehensive test coverage:
- **62 tensor operation tests** (all passing)
- **Map/Apply function tests** (built-in and pure function support)
- **7 sparse matrix operation tests** (COO format, arithmetic, properties)
- **Spatial data structure tests** (KD-Tree and Ball-Tree construction and queries)
- **Edge case handling** (division by zero, incompatible shapes, empty data)
- **Integration tests** with VM
- **Snapshot testing** for consistent output formatting

## Future Extensions

Planned additions for neural network support:
- Activation functions (Sigmoid, Tanh, Softmax)
- Layer abstractions (LinearLayer, ActivationLayer)
- Network composition (NetChain, NetGraph)
- Loss functions (MSE, CrossEntropy)
- Training infrastructure

## Development Guidelines

1. **Test-Driven Development**: All new functions must have comprehensive tests
2. **Wolfram Compatibility**: Follow Wolfram Language naming and semantics
3. **Error Handling**: Provide clear, actionable error messages
4. **Documentation**: Include examples and edge cases
5. **Performance**: Benchmark critical operations for optimization opportunities