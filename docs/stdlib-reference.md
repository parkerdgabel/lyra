# Lyra Standard Library Reference

## Table of Contents

1. [Core Mathematics](#core-mathematics)
2. [List Operations](#list-operations)
3. [String Operations](#string-operations)
4. [Collections](#collections)
5. [System Operations](#system-operations)
6. [Logical Operations](#logical-operations)
7. [Constants](#constants)
8. [Comparison Functions](#comparison-functions)
9. [Utility Functions](#utility-functions)

---

## Core Mathematics

Basic arithmetic, trigonometric, and mathematical functions.

### Arithmetic Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Plus** | `Plus[a, b]` | Addition of two numbers | `Plus[2, 3]` → `5` |
| **Times** | `Times[a, b]` | Multiplication of two numbers | `Times[3, 4]` → `12` |
| **Divide** | `Divide[a, b]` | Division of two numbers | `Divide[6, 2]` → `3.0` |
| **Power** | `Power[base, exponent]` | Exponentiation | `Power[2, 3]` → `8` |
| **Minus** | `Minus[x]` | Unary negation | `Minus[5]` → `-5` |
| **Modulo** | `Modulo[a, b]` | Modulo operation | `Modulo[17, 5]` → `2` |
| **Abs** | `Abs[x]` | Absolute value | `Abs[-42]` → `42` |
| **Sign** | `Sign[x]` | Sign of number (-1, 0, 1) | `Sign[-5]` → `-1` |

### Trigonometric Functions

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Sin** | `Sin[x]` | Sine function | `Sin[0]` → `0.0` |
| **Cos** | `Cos[x]` | Cosine function | `Cos[0]` → `1.0` |
| **Tan** | `Tan[x]` | Tangent function | `Tan[0]` → `0.0` |

### Exponential and Logarithmic

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Exp** | `Exp[x]` | Exponential function (e^x) | `Exp[0]` → `1.0` |
| **Log** | `Log[x]` | Natural logarithm | `Log[1]` → `0.0` |
| **Sqrt** | `Sqrt[x]` | Square root | `Sqrt[4]` → `2.0` |

---

## List Operations

Functions for manipulating and working with lists.

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Length** | `Length[list]` | Get length of list | `Length[{1, 2, 3}]` → `3` |
| **Head** | `Head[list]` | Get first element | `Head[{1, 2, 3}]` → `1` |
| **Tail** | `Tail[list]` | Get all elements except first | `Tail[{1, 2, 3}]` → `{2, 3}` |
| **Append** | `Append[list, element]` | Add element to end | `Append[{1, 2}, 3]` → `{1, 2, 3}` |
| **Flatten** | `Flatten[list]` | Flatten nested lists one level | `Flatten[{{1, 2}, {3, 4}}]` → `{1, 2, 3, 4}` |
| **Total** | `Total[list]` | Sum all elements | `Total[{1, 2, 3, 4}]` → `10` |

---

## String Operations

Functions for string manipulation and processing.

### Basic String Functions

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **StringJoin** | `StringJoin[str1, str2, ...]` | Concatenate strings | `StringJoin["Hello", " ", "World"]` → `"Hello World"` |
| **StringLength** | `StringLength[string]` | Get string length | `StringLength["Hello"]` → `5` |
| **StringTake** | `StringTake[string, n]` | Take first n characters | `StringTake["Hello", 3]` → `"Hel"` |
| **StringDrop** | `StringDrop[string, n]` | Drop first n characters | `StringDrop["Hello", 2]` → `"llo"` |

### Advanced String Functions

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **StringSplit** | `StringSplit[string, delimiter]` | Split string by delimiter | `StringSplit["a,b,c", ","]` → `{"a", "b", "c"}` |
| **StringReplace** | `StringReplace[string, pattern, replacement]` | Replace pattern in string | `StringReplace["hello", "l", "x"]` → `"hexxo"` |
| **StringContains** | `StringContains[string, substring]` | Check if string contains substring | `StringContains["hello", "ell"]` → `True` |
| **StringTrim** | `StringTrim[string]` | Remove whitespace from ends | `StringTrim[" hello "]` → `"hello"` |
| **ToUpperCase** | `ToUpperCase[string]` | Convert to uppercase | `ToUpperCase["hello"]` → `"HELLO"` |
| **ToLowerCase** | `ToLowerCase[string]` | Convert to lowercase | `ToLowerCase["HELLO"]` → `"hello"` |

---

## Collections

Advanced data structures for efficient operations.

### Set Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **SetCreate** | `SetCreate[list]` | Create set from list (removes duplicates) | `SetCreate[{1, 2, 2, 3}]` → `{1, 2, 3}` |
| **SetUnion** | `SetUnion[set1, set2]` | Union of two sets | `SetUnion[{1, 2}, {2, 3}]` → `{1, 2, 3}` |
| **SetIntersection** | `SetIntersection[set1, set2]` | Intersection of sets | `SetIntersection[{1, 2}, {2, 3}]` → `{2}` |
| **SetDifference** | `SetDifference[set1, set2]` | Set difference (set1 - set2) | `SetDifference[{1, 2}, {2, 3}]` → `{1}` |
| **SetContains** | `SetContains[set, element]` | Check if set contains element | `SetContains[{1, 2, 3}, 2]` → `True` |

### Dictionary Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **DictCreate** | `DictCreate[pairs]` | Create dictionary from key-value pairs | `DictCreate[{{"key1", 10}, {"key2", 20}}]` |
| **DictGet** | `DictGet[dict, key, default]` | Get value by key with optional default | `DictGet[dict, "key1"]` |
| **DictSet** | `DictSet[dict, key, value]` | Set key-value pair | `DictSet[dict, "key1", 42]` |
| **DictKeys** | `DictKeys[dict]` | Get all keys as list | `DictKeys[dict]` → `{"key1", "key2"}` |
| **DictSize** | `DictSize[dict]` | Get number of key-value pairs | `DictSize[dict]` → `2` |

### Queue and Stack Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **QueueCreate** | `QueueCreate[]` | Create empty FIFO queue | `QueueCreate[]` |
| **QueueEnqueue** | `QueueEnqueue[queue, element]` | Add element to back of queue | `QueueEnqueue[queue, 42]` |
| **QueueDequeue** | `QueueDequeue[queue]` | Remove and return front element | `QueueDequeue[queue]` |
| **StackCreate** | `StackCreate[]` | Create empty LIFO stack | `StackCreate[]` |
| **StackPush** | `StackPush[stack, element]` | Push element onto stack | `StackPush[stack, 42]` |
| **StackPop** | `StackPop[stack]` | Pop and return top element | `StackPop[stack]` |

---

## System Operations

Functions for interacting with the operating system and environment.

### Environment Variables

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Environment** | `Environment[name]` | Get environment variable | `Environment["PATH"]` |
| **SetEnvironment** | `SetEnvironment[name, value]` | Set environment variable | `SetEnvironment["MY_VAR", "value"]` |
| **EnvironmentList** | `EnvironmentList[]` | List all environment variables | `EnvironmentList[]` |

### File System

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **FileExists** | `FileExists[path]` | Check if file exists | `FileExists["/path/to/file.txt"]` → `True` |
| **DirectoryExists** | `DirectoryExists[path]` | Check if directory exists | `DirectoryExists["/path/to/dir"]` → `True` |
| **FileSize** | `FileSize[path]` | Get file size in bytes | `FileSize["/path/to/file.txt"]` → `1024` |
| **CopyFile** | `CopyFile[source, destination]` | Copy file | `CopyFile["file1.txt", "file2.txt"]` |
| **DeleteFile** | `DeleteFile[path]` | Delete file | `DeleteFile["/path/to/file.txt"]` |

### Path Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **PathJoin** | `PathJoin[part1, part2, ...]` | Join path components | `PathJoin["home", "user", "docs"]` |
| **FileName** | `FileName[path]` | Extract filename from path | `FileName["/path/to/file.txt"]` → `"file.txt"` |
| **FileExtension** | `FileExtension[path]` | Extract file extension | `FileExtension["file.txt"]` → `".txt"` |
| **AbsolutePath** | `AbsolutePath[path]` | Get absolute path | `AbsolutePath["./file.txt"]` |

### Process Management

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **RunCommand** | `RunCommand[command, args]` | Execute command and get output | `RunCommand["echo", {"hello"}]` |
| **CurrentPID** | `CurrentPID[]` | Get current process ID | `CurrentPID[]` → `1234` |
| **ProcessExists** | `ProcessExists[pid]` | Check if process exists | `ProcessExists[1234]` → `True` |

---

## Logical Operations

Boolean logic and comparison functions.

### Boolean Operations

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **And** | `And[a, b]` | Logical AND | `And[True, False]` → `False` |
| **Or** | `Or[a, b]` | Logical OR | `Or[True, False]` → `True` |
| **Not** | `Not[x]` | Logical NOT | `Not[True]` → `False` |
| **Xor** | `Xor[a, b]` | Logical XOR | `Xor[True, True]` → `False` |

---

## Comparison Functions

Functions for comparing values.

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **Equal** | `Equal[a, b]` | Test equality | `Equal[5, 5]` → `True` |
| **Unequal** | `Unequal[a, b]` | Test inequality | `Unequal[5, 3]` → `True` |
| **Greater** | `Greater[a, b]` | Test if a > b | `Greater[5, 3]` → `True` |
| **GreaterEqual** | `GreaterEqual[a, b]` | Test if a >= b | `GreaterEqual[5, 5]` → `True` |
| **Less** | `Less[a, b]` | Test if a < b | `Less[3, 5]` → `True` |
| **LessEqual** | `LessEqual[a, b]` | Test if a <= b | `LessEqual[3, 5]` → `True` |

---

## Constants

Mathematical and system constants.

| Constant | Value | Description |
|----------|-------|-------------|
| **Pi** | `3.14159...` | The mathematical constant π |
| **E** | `2.71828...` | Euler's number (base of natural logarithm) |
| **EulerGamma** | `0.57721...` | Euler-Mascheroni constant γ |
| **GoldenRatio** | `1.61803...` | Golden ratio φ |
| **True** | `True` | Boolean true value |
| **False** | `False` | Boolean false value |
| **Infinity** | `∞` | Positive infinity |

---

## Utility Functions

Utility and conversion functions.

| Function | Signature | Description | Example |
|----------|-----------|-------------|---------|
| **ToString** | `ToString[value]` | Convert any value to string | `ToString[42]` → `"42"` |
| **If** | `If[condition, trueValue, falseValue]` | Conditional evaluation | `If[True, "yes", "no"]` → `"yes"` |
| **RandomReal** | `RandomReal[]` | Generate random real between 0 and 1 | `RandomReal[]` → `0.42384...` |
| **DateString** | `DateString[]` | Get current date and time | `DateString[]` → `"2024-01-15T14:30:45"` |

---

## Advanced Mathematics

For more advanced mathematical functions, Lyra includes extensive modules for:

### Special Functions
- **Gamma functions**: `Gamma`, `LogGamma`, `Digamma`, `Polygamma`
- **Error functions**: `Erf`, `Erfc`, `InverseErf`, `FresnelC`, `FresnelS`
- **Elliptic functions**: `EllipticK`, `EllipticE`, `EllipticTheta`
- **Hypergeometric functions**: `Hypergeometric0F1`, `Hypergeometric1F1`
- **Orthogonal polynomials**: `ChebyshevT`, `ChebyshevU`, `GegenbauerC`

### Linear Algebra
- **Matrix decompositions**: `SVD`, `QRDecomposition`, `LUDecomposition`, `CholeskyDecomposition`
- **Eigenvalue computations**: `EigenDecomposition`, `SchurDecomposition`
- **Linear systems**: `LinearSolve`, `LeastSquares`, `PseudoInverse`
- **Matrix operations**: `MatrixPower`, `MatrixTrace`, `Determinant`

### Signal Processing
- **Transforms**: `FFT`, `IFFT`, `DCT`, `PowerSpectrum`
- **Filtering**: `LowPassFilter`, `HighPassFilter`, `MedianFilter`
- **Analysis**: `Periodogram`, `Spectrogram`, `CrossCorrelation`

### Graph Theory
- **Graph algorithms**: `DepthFirstSearch`, `BreadthFirstSearch`, `Dijkstra`
- **Graph properties**: `ConnectedComponents`, `GraphProperties`
- **Network analysis**: `NetworkCentrality`, `CommunityDetection`

### Statistics
- **Descriptive statistics**: `Mean`, `Variance`, `StandardDeviation`, `Median`
- **Advanced analysis**: `Regression`, `ANOVA`, `TTest`, `PCA`
- **Random generation**: `RandomInteger`, `BootstrapSample`

## Usage Notes

1. **Function Names**: All function names use PascalCase (e.g., `StringJoin`, `FileExists`)
2. **Arguments**: Functions use square brackets for arguments: `Function[arg1, arg2]`
3. **Lists**: Lists are denoted with curly braces: `{1, 2, 3}`
4. **Strings**: Strings use double quotes: `"Hello World"`
5. **Error Handling**: Functions return appropriate error messages for invalid inputs
6. **Type Conversion**: Functions handle automatic type conversions where appropriate (e.g., integer to real)

## Examples

### Data Processing Pipeline
```wolfram
(* Read data, process, and analyze *)
data = Import["data.csv"]
filtered = DataFilter[data, GreaterThan[100]]
summary = StatisticalSummary[filtered]
Export["results.json", summary]
```

### Mathematical Computation
```wolfram
(* Complex mathematical operations *)
matrix = {{1, 2}, {3, 4}}
eigenvals = EigenDecomposition[matrix]
result = MatrixPower[matrix, 2]
```

### String Processing
```wolfram
(* Text manipulation *)
text = "Hello, World! This is a test."
words = StringSplit[text, " "]
upper = Map[ToUpperCase, words]
joined = StringJoin[upper, " "]
```

This reference covers the most essential and frequently used functions in the Lyra standard library. For a complete list of all 2,938+ available functions, refer to the full API documentation or use the `Help[]` function within Lyra.