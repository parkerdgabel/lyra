# Lyra Language Reference

## Table of Contents

1. [Introduction](#introduction)
2. [Lexical Structure](#lexical-structure)
3. [Data Types](#data-types)
4. [Expressions](#expressions)
5. [Patterns](#patterns)
6. [Rules and Transformations](#rules-and-transformations)
7. [Modern Syntax Extensions](#modern-syntax-extensions)
8. [Operator Precedence](#operator-precedence)
9. [Examples](#examples)

## Introduction

Lyra is a symbolic computation language inspired by the Wolfram Language, designed for mathematical and symbolic computation. It combines traditional Wolfram syntax with modern programming language features like pipelines, arrow functions, and structured data types.

### Key Features

- **Wolfram-compatible syntax**: `f[x, y]`, `{1, 2, 3}`, patterns, and rules
- **Modern extensions**: Pipelines (`|>`), arrow functions (`=>`), associations (`<||>`), ranges (`;;`)
- **Advanced patterns**: Typed patterns, predicates, conditions, alternatives
- **Symbolic computation**: Mathematical expressions as first-class values
- **Functional programming**: Higher-order functions, pattern matching, transformations

### Design Philosophy

Lyra prioritizes:
1. **Expressiveness**: Rich syntax for mathematical and symbolic operations
2. **Readability**: Clear, intuitive syntax for complex expressions
3. **Composability**: Functions and expressions that compose naturally
4. **Performance**: Efficient parsing and evaluation of symbolic expressions

## Lexical Structure

### Comments

```lyra
(* This is a block comment *)
(* Block comments 
   can span multiple lines
   and (* can be nested *) *)

// This is a line comment (to end of line)
```

### Identifiers and Symbols

```lyra
x                    (* Simple symbol *)
myVariable           (* CamelCase symbol *)
price_usd           (* Underscore in middle *)
$TemporaryVariable  (* Dollar prefix *)
f123                (* Numbers allowed *)
```

### Context and Namespaces

```lyra
std`net`http`Get    (* Context symbol *)
System`Plus         (* System context *)
mypackage`myfunction (* Custom context *)
```

### Whitespace

- Spaces, tabs, and newlines generally ignored
- Significant for symbol boundaries
- Implicit multiplication: `2 x` means `2 * x`

## Data Types

### Numbers

#### Integers
```lyra
42                  (* Integer *)
-17                 (* Negative integer *)
0                   (* Zero *)
```

#### Real Numbers
```lyra
3.14                (* Real number *)
2.5e-3              (* Scientific notation *)
.5                  (* Leading decimal point *)
1.23E+10            (* Capital E notation *)
```

#### Extended Number Types
```lyra
123n                (* BigInt - arbitrary precision *)
1.23d100            (* BigDecimal - arbitrary precision decimal *)
16^^FF              (* Hex notation - base 16 *)
2^^1010             (* Binary notation - base 2 *)
8^^777              (* Octal notation - base 8 *)
3/4                 (* Rational number - exact fraction *)
```

### Strings

#### Simple Strings
```lyra
"hello world"       (* Basic string *)
"line1\nline2"      (* With escape sequences *)
"quote: \"text\""   (* Escaped quotes *)
```

#### String Interpolation
```lyra
"Hello #{name}!"                    (* Variable interpolation *)
"Result: #{2 + 3}"                  (* Expression interpolation *)
"User #{user.name} has #{user.age} years"  (* Complex expressions *)
```

### Lists
```lyra
{}                  (* Empty list *)
{1, 2, 3}          (* Simple list *)
{a, b, c}          (* Symbol list *)
{f[x], g[y], h[z]} (* Function list *)
{{1, 2}, {3, 4}}   (* Nested lists *)
```

### Associations (Key-Value Maps)
```lyra
<||>                           (* Empty association *)
<|"name" -> "Ada"|>           (* Single pair *)
<|"name" -> "Ada", "age" -> 37|>  (* Multiple pairs *)
<|1 -> "one", 2 -> "two"|>    (* Numeric keys *)
<|f[x] -> g[y], a + b -> c|>  (* Expression keys and values *)
```

## Expressions

### Function Calls
```lyra
f[x]                (* Single argument *)
f[x, y]             (* Multiple arguments *)
f[]                 (* No arguments *)
Plus[1, 2, 3]       (* Variable arguments *)
```

### Arithmetic Operations
```lyra
a + b               (* Addition *)
a - b               (* Subtraction *)
a * b               (* Multiplication *)
a / b               (* Division *)
a ^ b               (* Power *)
-a                  (* Unary minus *)
+a                  (* Unary plus *)
```

### Comparison Operations
```lyra
a == b              (* Equal *)
a != b              (* Not equal *)
a < b               (* Less than *)
a <= b              (* Less than or equal *)
a > b               (* Greater than *)
a >= b              (* Greater than or equal *)
```

### Logical Operations
```lyra
a && b              (* Logical AND *)
a || b              (* Logical OR *)
!a                  (* Logical NOT *)
```

### Part Access
```lyra
list[[1]]           (* First element *)
list[[1, 2]]        (* Multiple indices *)
list[[-1]]          (* Last element *)
```

## Patterns

Patterns are used for matching and destructuring expressions.

### Basic Patterns

#### Blank Patterns
```lyra
_                   (* Matches any single expression *)
__                  (* Matches zero or more expressions *)
___                 (* Matches zero or more expressions (same as __) *)
```

#### Named Patterns
```lyra
x_                  (* Named blank - matches any expression, binds to x *)
x__                 (* Named sequence - matches sequence, binds to x *)
x___                (* Named null sequence - matches sequence, binds to x *)
```

#### Typed Patterns (Legacy)
```lyra
_Integer            (* Matches any integer *)
_Real               (* Matches any real number *)
_String             (* Matches any string *)
x_Integer           (* Named typed pattern *)
```

### Modern Pattern Extensions

#### Typed Patterns (Modern)
```lyra
x:_Integer          (* Modern typed pattern syntax *)
x:_Real             (* Matches real numbers *)
x:_String           (* Matches strings *)
x:_List             (* Matches lists *)
```

#### Predicate Patterns
```lyra
x_?Positive         (* Matches positive numbers *)
x_?EvenQ            (* Matches even numbers *)
x_?NumberQ          (* Matches any number *)
```

#### Alternative Patterns
```lyra
x | y | z           (* Matches x, y, or z *)
_Integer | _Real    (* Matches integers or reals *)
```

#### Conditional Patterns
```lyra
x_ /; x > 0         (* Matches x where x > 0 *)
x_Integer /; EvenQ[x]  (* Matches even integers *)
```

### Complex Pattern Examples
```lyra
f[x_, y_] /; x > y  (* Function pattern with condition *)
{x_, y__}           (* List with first element and rest *)
<|"key" -> value_|> (* Association pattern *)
```

## Rules and Transformations

### Rules
```lyra
x -> x^2            (* Immediate rule *)
x :> RandomReal[]   (* Delayed rule - RHS evaluated each time *)
```

### Rule Application
```lyra
expr /. rule        (* Apply rule to expression *)
expr /. {rule1, rule2}  (* Apply multiple rules *)
```

### Assignments (Definitions)
```lyra
x = 5               (* Immediate assignment *)
x := RandomReal[]   (* Delayed assignment *)
f[x_] = x^2         (* Function definition *)
f[x_] := Print[x]   (* Delayed function definition *)
```

### Example Transformations
```lyra
(* Define square function *)
square[x_] = x^2

(* Apply to expression *)
square[5]           (* Result: 25 *)

(* Pattern replacement *)
expr = a + b + c
expr /. {a -> 1, b -> 2, c -> 3}  (* Result: 6 *)

(* Conditional rules *)
abs[x_] := x /; x >= 0
abs[x_] := -x /; x < 0
```

## Modern Syntax Extensions

### Pipelines
```lyra
x |> f              (* Equivalent to f[x] *)
x |> f |> g         (* Equivalent to g[f[x]] *)
x |> f |> g[y]      (* Equivalent to g[f[x], y] *)

(* Complex pipeline *)
data |> Map[square] |> Select[# > 10&] |> Length
```

### Arrow Functions
```lyra
() => 42            (* No parameters *)
(x) => x + 1        (* Single parameter *)
(x, y) => x * y     (* Multiple parameters *)

(* In pipelines *)
{1, 2, 3} |> Map[(x) => x^2]  (* {1, 4, 9} *)

(* Nested arrow functions *)
(x) => (y) => x + y (* Curried function *)
```

### Dot-Call Syntax
```lyra
obj.method[args]    (* Equivalent to method[obj, args] *)
list.map[f]         (* Equivalent to map[list, f] *)
data.filter[pred].sort[]  (* Chained calls *)

(* In pipelines *)
data |> obj.process[config] |> result.format[]
```

### Range Syntax
```lyra
1;;10               (* Range from 1 to 10 *)
0;;1;;0.1           (* Range with step: 0, 0.1, 0.2, ..., 1.0 *)
-5;;5;;2            (* Range: -5, -3, -1, 1, 3, 5 *)

(* Usage examples *)
Range[1;;10]        (* Generate list: {1, 2, 3, ..., 10} *)
Sum[i^2, {i, 1;;100}]  (* Sum of squares *)
```

### Postfix and Prefix Operators
```lyra
x // f              (* Postfix: equivalent to f[x] *)
f @ x               (* Prefix: equivalent to f[x] *)
x // f // g         (* Chained postfix: g[f[x]] *)
```

## Operator Precedence

From highest to lowest precedence:

1. `[[]]` - Part access
2. `[]` - Function calls, `.method[]` - Dot calls  
3. `//` - Postfix application
4. `@` - Prefix application
5. `^` - Power (right associative)
6. `+`, `-` - Unary operators
7. `*`, `/` - Multiplication, division
8. `+`, `-` - Addition, subtraction  
9. `<`, `<=`, `>`, `>=` - Relational operators
10. `==`, `!=` - Equality operators
11. `|` - Alternative patterns
12. `&&` - Logical AND
13. `||` - Logical OR
14. `/;` - Conditional patterns
15. `->`, `:>` - Rules
16. `/.` - ReplaceAll
17. `|>` - Pipeline
18. `=`, `:=` - Assignment
19. `=>` - Arrow functions

### Precedence Examples
```lyra
2 + 3 * 4           (* 14, not 20 *)
2^3^2               (* 2^(3^2) = 512, right associative *)
f[x] |> g |> h[y]   (* h[g[f[x]], y] *)
x |> f /. rule      (* (x |> f) /. rule *)
```

## Examples

### Basic Mathematical Operations
```lyra
(* Arithmetic *)
result = 2 + 3 * 4^2    (* 50 *)
area = Pi * r^2         (* Circle area *)

(* Function definitions *)
factorial[0] = 1
factorial[n_Integer] := n * factorial[n - 1]

square[x_] = x^2
double[x_] := 2 * x
```

### Data Manipulation
```lyra
(* Lists *)
numbers = {1, 2, 3, 4, 5}
squares = Map[square, numbers]      (* {1, 4, 9, 16, 25} *)
evens = Select[numbers, EvenQ]      (* {2, 4} *)

(* Associations *)
person = <|"name" -> "Ada", "age" -> 37, "city" -> "London"|>
name = person[["name"]]             (* "Ada" *)
```

### Modern Syntax in Action
```lyra
(* Pipeline processing *)
result = {1, 2, 3, 4, 5}
       |> Map[(x) => x^2]
       |> Select[# > 5&]
       |> Length                    (* 3 *)

(* Data transformation *)
data = <|"values" -> {1, 2, 3, 4, 5}|>
processed = data
          |> #[["values"]]&
          |> Map[(x) => x * 2]
          |> Select[# > 5&]          (* {6, 8, 10} *)

(* Function composition *)
pipeline = (x) => x |> square |> (y) => y + 1
result = pipeline[5]                (* 26 *)
```

### Pattern Matching Examples
```lyra
(* Fibonacci with patterns *)
fib[0] = 0
fib[1] = 1  
fib[n_Integer] := fib[n-1] + fib[n-2]

(* List processing patterns *)
first[{x_, ___}] := x
rest[{_, x___}] := {x}
isEmpty[{}] := True
isEmpty[_] := False

(* Conditional patterns *)
sign[x_ /; x > 0] := 1
sign[x_ /; x < 0] := -1
sign[0] := 0
```

### Advanced Examples
```lyra
(* Symbolic differentiation rules *)
D[x_, x_] = 1
D[c_?NumberQ, x_] := 0
D[u_ + v_, x_] := D[u, x] + D[v, x]
D[u_ * v_, x_] := D[u, x] * v + u * D[v, x]
D[x_^n_?NumberQ, x_] := n * x^(n-1)

(* Data processing pipeline *)
processData[data_] := data
  |> Select[#[["status"]] == "active"&]
  |> Map[#[["value"]]&]
  |> Select[NumberQ]
  |> Sort
  |> Take[10]

(* Complex transformations *)
normalize[expr_] := expr /. {
  a_*x_ + b_*x_ :> (a + b)*x,
  x_^m_ * x_^n_ :> x^(m + n),
  x_*0 :> 0,
  x_*1 :> x
}
```

This reference covers the complete Lyra language syntax and semantics. For implementation details and API documentation, see the [API Documentation](api.md).