# Lyra User Guide

## Getting Started

Welcome to Lyra, a modern symbolic computation engine inspired by the Wolfram Language. This guide will help you understand and use Lyra's powerful syntax for mathematical and symbolic computation.

### What is Lyra?

Lyra is designed for:
- **Symbolic mathematics**: Working with mathematical expressions as data
- **Pattern matching**: Sophisticated matching and transformation rules  
- **Functional programming**: Higher-order functions and composition
- **Data manipulation**: Modern syntax for working with structured data

### Installation

```bash
git clone https://github.com/parkerdgabel/lyra.git
cd lyra
cargo build --release
```

### Your First Lyra Expressions

Let's start with some basic expressions:

```lyra
(* Basic arithmetic *)
2 + 3 * 4           (* Result: 14 *)
2^10                (* Result: 1024 *)

(* Function calls *)
f[x, y]             (* Call function f with arguments x and y *)
Plus[1, 2, 3, 4]    (* Add multiple numbers *)

(* Lists *)
{1, 2, 3, 4, 5}     (* A list of numbers *)
{a, b, c}           (* A list of symbols *)
```

## Core Concepts

### 1. Everything is an Expression

In Lyra, everything is an expression that can be manipulated symbolically:

```lyra
(* Numbers are expressions *)
42
3.14159
3/4                 (* Exact rational *)

(* Symbols are expressions *)
x
myVariable
price_usd

(* Function calls are expressions *)
f[x]
Sin[Pi/4]
Plus[a, b, c]

(* Even complex structures are expressions *)
{1, 2, 3}
<|"name" -> "Ada", "age" -> 37|>
```

### 2. Functions Use Square Brackets

Unlike many programming languages, Lyra uses square brackets for function calls:

```lyra
(* Function calls *)
f[x]                (* Call f with argument x *)
g[x, y, z]          (* Call g with three arguments *)
h[]                 (* Call h with no arguments *)

(* Mathematical functions *)
Sin[Pi/2]           (* Sine function *)
Sqrt[16]            (* Square root *)
Log[E]              (* Natural logarithm *)
```

### 3. Lists Use Curly Braces

Lists are fundamental data structures in Lyra:

```lyra
(* Empty list *)
{}

(* Number list *)
{1, 2, 3, 4, 5}

(* Mixed types *)
{1, "hello", x, f[y]}

(* Nested lists *)
{{1, 2}, {3, 4}, {5, 6}}
```

## Working with Patterns

Patterns are one of Lyra's most powerful features, allowing you to match and destructure expressions.

### Basic Patterns

```lyra
(* Blank patterns *)
_                   (* Matches any single expression *)
__                  (* Matches zero or more expressions *)
___                 (* Matches zero or more expressions *)

(* Named patterns *)
x_                  (* Matches anything, binds to x *)
x__                 (* Matches sequence, binds to x *)
```

### Function Definitions with Patterns

```lyra
(* Define a function using patterns *)
square[x_] = x^2

(* Use it *)
square[5]           (* Result: 25 *)
square[y]           (* Result: y^2 *)

(* Multiple patterns *)
factorial[0] = 1
factorial[n_] := n * factorial[n - 1]
```

### Advanced Patterns

```lyra
(* Typed patterns (legacy syntax) *)
f[x_Integer] := "Got an integer"
f[x_Real] := "Got a real number"
f[x_String] := "Got a string"

(* Modern typed patterns *)
g[x:_Integer] := x^2
g[x:_String] := StringLength[x]

(* Predicate patterns *)
positive[x_?Positive] := x
positive[x_] := 0

(* Conditional patterns *)
fibonacci[n_ /; n >= 0] := If[n <= 1, n, fibonacci[n-1] + fibonacci[n-2]]
```

## Rules and Transformations

Rules let you transform expressions by specifying patterns and replacements.

### Basic Rules

```lyra
(* Simple rule *)
x -> x^2            (* Replace x with x^2 *)

(* Apply a rule *)
expr = a + b + c
expr /. a -> 1      (* Replace a with 1: 1 + b + c *)

(* Multiple rules *)
expr /. {a -> 1, b -> 2, c -> 3}  (* Result: 6 *)
```

### Delayed Rules

```lyra
(* Immediate rule - RHS evaluated once *)
x -> RandomReal[]

(* Delayed rule - RHS evaluated each time *)
x :> RandomReal[]   (* Different random number each time *)
```

### Complex Transformations

```lyra
(* Simplification rules *)
simplify = {
  x_ + 0 -> x,
  x_ * 0 -> 0,
  x_ * 1 -> x,
  x_^0 -> 1,
  x_^1 -> x
}

(* Apply to expression *)
expr = a*1 + b*0 + c^1
expr /. simplify    (* Result: a + c *)
```

## Modern Syntax Features

Lyra extends traditional Wolfram syntax with modern programming features.

### Pipelines

Pipelines provide a clean way to chain operations:

```lyra
(* Traditional nested calls *)
h[g[f[x]]]

(* Pipeline syntax *)
x |> f |> g |> h

(* Real example *)
{1, 2, 3, 4, 5}
  |> Map[square]
  |> Select[# > 5&]
  |> Length         (* Result: 3 *)
```

### Arrow Functions

Arrow functions provide concise function syntax:

```lyra
(* Traditional function *)
Function[x, x^2]

(* Arrow function *)
(x) => x^2

(* Multiple parameters *)
(x, y) => x + y

(* In pipelines *)
{1, 2, 3} |> Map[(x) => x * 2]  (* {2, 4, 6} *)
```

### Associations (Key-Value Maps)

Associations provide structured data:

```lyra
(* Create an association *)
person = <|"name" -> "Ada", "age" -> 37, "city" -> "London"|>

(* Access values *)
person[["name"]]    (* "Ada" *)
person[["age"]]     (* 37 *)

(* Complex associations *)
data = <|
  "users" -> {
    <|"name" -> "Alice", "score" -> 85|>,
    <|"name" -> "Bob", "score" -> 92|>
  },
  "settings" -> <|"theme" -> "dark", "lang" -> "en"|>
|>
```

### Dot-Call Syntax

Dot-call syntax provides object-oriented style calls:

```lyra
(* Traditional *)
Map[f, list]

(* Dot-call syntax *)
list.Map[f]

(* Chaining *)
data
  .filter[predicate]
  .map[transform]
  .sort[]
```

### Range Syntax

Ranges provide concise sequence notation:

```lyra
1;;10               (* Range from 1 to 10 *)
0;;1;;0.1           (* Range with step *)

(* Use in functions *)
Sum[i^2, {i, 1;;100}]  (* Sum of squares from 1 to 100 *)
```

## Practical Examples

### Example 1: Data Processing

```lyra
(* Sample data *)
sales = {
  <|"date" -> "2024-01-01", "amount" -> 150, "region" -> "North"|>,
  <|"date" -> "2024-01-02", "amount" -> 200, "region" -> "South"|>,
  <|"date" -> "2024-01-03", "amount" -> 175, "region" -> "North"|>,
  <|"date" -> "2024-01-04", "amount" -> 225, "region" -> "South"|>
}

(* Process with pipeline *)
northSales = sales
  |> Select[#[["region"]] == "North"&]
  |> Map[#[["amount"]]&]
  |> Total

(* Result: 325 *)
```

### Example 2: Mathematical Functions

```lyra
(* Define derivative rules *)
D[x_, x_] = 1
D[c_?NumberQ, x_] := 0
D[u_ + v_, x_] := D[u, x] + D[v, x]
D[u_ * v_, x_] := D[u, x] * v + u * D[v, x]
D[x_^n_?NumberQ, x_] := n * x^(n-1)

(* Use the derivative *)
expr = x^3 + 2*x^2 + x + 5
D[expr, x]          (* Result: 3*x^2 + 4*x + 1 *)
```

### Example 3: Functional Programming

```lyra
(* Higher-order functions *)
twice[f_] := (x) => f[f[x]]
addOne = (x) => x + 1
addTwo = twice[addOne]

addTwo[5]           (* Result: 7 *)

(* Function composition *)
compose[f_, g_] := (x) => f[g[x]]
square = (x) => x^2
increment = (x) => x + 1
squareIncrement = compose[square, increment]

squareIncrement[4]  (* Result: 25 *)
```

### Example 4: Pattern Matching

```lyra
(* Tree structure processing *)
treeSize[Leaf[_]] := 1
treeSize[Node[left_, right_]] := treeSize[left] + treeSize[right] + 1

(* Example tree *)
tree = Node[
  Node[Leaf[1], Leaf[2]], 
  Leaf[3]
]

treeSize[tree]      (* Result: 5 *)
```

### Example 5: String Processing

```lyra
(* String interpolation *)
name = "World"
greeting = "Hello #{name}!"  (* "Hello World!" *)

(* Template processing *)
template[name_, age_] := "User #{name} is #{age} years old"
message = template["Alice", 25]  (* "User Alice is 25 years old" *)
```

## Best Practices

### 1. Use Descriptive Names

```lyra
(* Good *)
calculateArea[radius_] := Pi * radius^2
processUserData[users_] := users |> Select[#[["active"]]&]

(* Avoid *)
f[x_] := Pi * x^2
g[u_] := u |> Select[#[["active"]]&]
```

### 2. Leverage Patterns

```lyra
(* Use patterns for clear function definitions *)
isEmpty[{}] := True
isEmpty[_] := False

first[{x_, ___}] := x
rest[{_, x___}] := {x}
```

### 3. Use Modern Syntax

```lyra
(* Pipeline style is often clearer *)
result = data
  |> Select[criteria]
  |> Map[transform]
  |> Sort

(* Instead of nested calls *)
result = Sort[Map[transform, Select[data, criteria]]]
```

### 4. Combine Features

```lyra
(* Modern syntax with traditional patterns *)
processItems[items_List] := items
  |> Select[validItem]
  |> Map[(item) => item.transform[config]]
  |> GroupBy[#[["category"]]&]
```

## Common Patterns

### Working with Lists

```lyra
(* Map over list *)
{1, 2, 3} |> Map[(x) => x^2]

(* Filter list *)
{1, 2, 3, 4, 5} |> Select[EvenQ]

(* Reduce list *)
{1, 2, 3, 4, 5} |> Fold[Plus, 0]

(* Transform and filter *)
numbers = 1;;20
result = numbers
  |> Map[(x) => x^2]
  |> Select[# > 50&]
  |> Take[5]
```

### Working with Associations

```lyra
(* Extract values *)
users |> Map[#[["name"]]&]

(* Filter by criteria *)
users |> Select[#[["age"]] > 18&]

(* Transform structure *)
users |> Map[user => <|
  "fullName" -> user[["firstName"]] <> " " <> user[["lastName"]],
  "adult" -> user[["age"]] >= 18
|>]
```

### Building DSLs

```lyra
(* Query DSL *)
query[table_][Select[criteria_]][OrderBy[field_]][Take[n_]] := 
  table
    |> Select[criteria]
    |> SortBy[field]
    |> Take[n]

(* Usage *)
result = query[users][Select[#[["active"]]&]][OrderBy["name"]][Take[10]]
```

## Debugging Tips

### 1. Use Print for Debugging

```lyra
(* Add Print statements in pipelines *)
result = data
  |> (Print["After input: ", #]; #)&
  |> Map[transform]
  |> (Print["After map: ", #]; #)&
  |> Select[criteria]
```

### 2. Test Patterns Incrementally

```lyra
(* Test pattern matching step by step *)
testPattern[expr_] := {
  "original" -> expr,
  "matches" -> MatchQ[expr, pattern],
  "cases" -> Cases[expr, pattern]
}
```

### 3. Use Simple Examples

```lyra
(* Start with simple cases *)
factorial[0] = 1
factorial[1] = 1
factorial[2] = 2  (* Test specific cases first *)
factorial[n_] := n * factorial[n-1]  (* Then general case *)
```

## Next Steps

Now that you understand Lyra's core concepts:

1. **Practice with examples**: Try the examples in this guide
2. **Read the Language Reference**: See [language-reference.md](language-reference.md) for complete syntax
3. **Check the API docs**: See [api.md](api.md) for implementation details
4. **Build something**: Create your own symbolic computation projects

Lyra combines the power of symbolic computation with modern syntax for an expressive and productive programming experience. Happy computing!