# Lyra Quick Start Guide

**Get up and running with Lyra in 15 minutes!**

Lyra is a modern symbolic computation language designed for data science, mathematics, and scientific computing. This guide will get you productive quickly with practical examples and common workflows.

## 🚀 Getting Started

### Installation

**Prerequisites:**
- [Rust 1.70+](https://rustup.rs/) 
- Git

**Quick Install:**
```bash
# Clone and build Lyra
git clone https://github.com/parkerdgabel/lyra.git
cd lyra
cargo build --release

# Start the interactive REPL
cargo run --bin lyra -- repl
```

### First Steps - The REPL

Lyra's REPL (Read-Eval-Print Loop) is your gateway to interactive computing:

```lyra
Lyra v1.0 - Symbolic Computation Language
Type ?help for assistance, ?exit to quit

In[1]:= 2 + 2
Out[1]= 4

In[2]:= "Hello, World!"
Out[2]= "Hello, World!"
```

**REPL Essentials:**
- Press `Tab` for auto-completion
- Use `↑/↓` arrows for command history  
- Type `?functionName` for help on any function
- Use `??keyword` for fuzzy search and browsing
- Exit with `?exit` or `Ctrl+C`

## ⚡ Core Concepts

### Everything is an Expression

In Lyra, everything you type is an expression that gets evaluated:

```lyra
(* Numbers *)
42                    (* → 42 *)
3.14159              (* → 3.14159 *)

(* Symbols *)
x                    (* → x (unevaluated symbol) *)
Pi                   (* → π (mathematical constant) *)

(* Function calls use square brackets *)
Sin[Pi/2]            (* → 1 *)
Sqrt[16]             (* → 4 *)
```

### Variables and Assignment

```lyra
(* Immediate assignment (=) - evaluate once and store *)
x = 5 + 3            (* x becomes 8 *)
name = "Lyra"        (* name becomes "Lyra" *)

(* Delayed assignment (:=) - evaluate each time used *)
current_time := DateString[]  (* New timestamp each access *)
random := RandomReal[]        (* New random number each time *)
```

## 📊 Essential Examples

### 1. Basic Mathematics

```lyra
(* Arithmetic follows standard precedence *)
2 + 3 * 4            (* → 14 *)
(2 + 3) * 4          (* → 20 *)
2^10                 (* → 1024 *)

(* Mathematical functions *)
Sin[Pi/2]            (* → 1 *)
Log[E]               (* → 1 *)
Sqrt[144]            (* → 12 *)
Abs[-42]             (* → 42 *)

(* Complex expressions *)
Sqrt[Sin[Pi/4]^2 + Cos[Pi/4]^2]  (* → 1 *)
```

### 2. Working with Lists

```lyra
(* Create lists with curly braces *)
numbers = {1, 2, 3, 4, 5}        (* List of numbers *)
mixed = {1, "hello", 3.14, True} (* Mixed types allowed *)

(* List operations *)
Length[numbers]      (* → 5 *)
Head[numbers]        (* → 1 (first element) *)  
Tail[numbers]        (* → {2, 3, 4, 5} (rest) *)
Append[numbers, 6]   (* → {1, 2, 3, 4, 5, 6} *)

(* Functional programming on lists *)
Map[Square, numbers]           (* → {1, 4, 9, 16, 25} *)
Select[numbers, # > 3 &]       (* → {4, 5} *)
Total[numbers]                 (* → 15 *)
```

### 3. String Operations

```lyra
(* String creation and manipulation *)
greeting = "Hello, World!"
name = "Lyra"

(* String operations *)
StringLength[greeting]         (* → 13 *)
greeting <> " from " <> name   (* → "Hello, World! from Lyra" *)
StringTake[greeting, 5]        (* → "Hello" *)
StringReplace[greeting, "World", "Universe"] (* → "Hello, Universe!" *)

(* String processing *)
words = {"Data", "Science", "with", "Lyra"}
StringJoin[words, " "]         (* → "Data Science with Lyra" *)
```

### 4. Pattern Matching

Pattern matching is one of Lyra's most powerful features:

```lyra
(* Basic patterns *)
MatchQ[42, _]           (* → True (blank matches anything) *)
MatchQ[42, _Integer]    (* → True (type-constrained pattern) *)
MatchQ["hi", _String]   (* → True *)
MatchQ[{1,2,3}, {___}]  (* → True (sequence pattern) *)

(* Named patterns capture values *)
Replace[42, x_ -> x^2]  (* → 1764 *)
Replace[{1,2,3}, {x_, y_, z_} -> x + y + z]  (* → 6 *)

(* Pattern-based function definitions *)
factorial[0] = 1
factorial[n_] := n * factorial[n - 1]
factorial[5]            (* → 120 *)
```

### 5. Functions and Rules

```lyra
(* Define functions with patterns *)
square[x_] = x^2
double[x_] = 2 * x
add[x_, y_] = x + y

(* Use your functions *)
square[5]              (* → 25 *)
double[7]              (* → 14 *)
add[3, 4]              (* → 7 *)

(* Pure functions (anonymous) *)
Map[#^2 &, {1, 2, 3}]  (* → {1, 4, 9} *)
Select[{1, 2, 3, 4, 5}, # > 3 &]  (* → {4, 5} *)

(* Transformation rules *)
expr = x + 2*y + x^2
expr /. x -> 3         (* → 3 + 2*y + 9 *)
expr /. {x -> 1, y -> 2}  (* → 1 + 4 + 1 = 6 *)
```

### 6. Data Processing Workflow

```lyra
(* Create sample data *)
data = Table[{i, Sin[i/10], Cos[i/10]}, {i, 1, 100}]

(* Extract columns *)
x_values = data[[All, 1]]      (* First column *)
sin_values = data[[All, 2]]    (* Second column *)
cos_values = data[[All, 3]]    (* Third column *)

(* Statistical analysis *)
Mean[sin_values]               (* Average of sine values *)
StandardDeviation[cos_values]  (* Standard deviation *)
Min[x_values]                  (* Minimum value *)
Max[x_values]                  (* Maximum value *)

(* Filter and transform *)
filtered = Select[data, #[[2]] > 0.5 &]  (* Sin > 0.5 *)
scaled = Map[{#[[1]], 10*#[[2]], 10*#[[3]]} &, data]
```

### 7. Mathematical Expressions

```lyra
(* Symbolic mathematics *)
expr = x^2 + 2*x + 1
D[expr, x]              (* → 2*x + 2 (derivative) *)
Integrate[x^2, x]       (* → x^3/3 (integral) *)
Solve[x^2 - 5*x + 6 == 0, x]  (* → {x = 2, x = 3} *)

(* Series expansion *)
Series[Sin[x], {x, 0, 5}]  (* Taylor series *)
Series[Exp[x], {x, 0, 4}]  (* → 1 + x + x^2/2 + x^3/6 + x^4/24 + O[x]^5 *)
```

### 8. Plotting and Visualization

```lyra
(* Basic plotting *)
Plot[x^2, {x, -5, 5}]                    (* Parabola *)
Plot[{Sin[x], Cos[x]}, {x, 0, 2*Pi}]     (* Multiple functions *)

(* Parametric plots *)
ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2*Pi}]  (* Circle *)

(* Data visualization *)
data = RandomReal[{0, 1}, 100]
Histogram[data]                          (* Histogram *)
ListPlot[data]                          (* Scatter plot *)
```

### 9. Working with Tables

```lyra
(* Create structured data *)
sales_data = Table[
  {
    "Month" -> {"Jan", "Feb", "Mar", "Apr"}[[i]],
    "Sales" -> RandomInteger[{1000, 5000}],
    "Profit" -> RandomReal[{0.1, 0.3}]
  }, 
  {i, 1, 4}
]

(* Query data *)
sales_data[[All, "Sales"]]              (* Extract sales column *)
Select[sales_data, #["Sales"] > 3000 &] (* Filter high sales *)
Total[sales_data[[All, "Sales"]]]       (* Sum all sales *)
```

### 10. File Operations

```lyra
(* Read and write data *)
Export["data.csv", {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}]
imported_data = Import["data.csv"]

(* JSON operations *)
json_data = {"name" -> "John", "age" -> 30, "city" -> "NYC"}
Export["person.json", json_data]
person = Import["person.json"]
```

## 🔍 Getting Help

Lyra has a comprehensive help system:

### Basic Help Commands

```lyra
(* Get help on any function *)
?Sin                    (* Show Sin function documentation *)
?Map                    (* Show Map function help *)
??math                  (* Browse mathematical functions *)
??string               (* Find all string functions *)

(* Search for functions *)
??integrate             (* Find integration-related functions *)
??random               (* Find random number functions *)
??plot                 (* Find plotting functions *)

(* Get function signature and examples *)
?RandomReal            (* Shows usage: RandomReal[], RandomReal[{min,max}] *)
?StringReplace         (* Shows pattern replacement examples *)
```

### Help Categories

```lyra
??basic                 (* Basic arithmetic and operations *)
??list                  (* List manipulation functions *)
??math                  (* Mathematical functions *)
??string               (* String processing *)
??pattern              (* Pattern matching *)
??data                 (* Data processing *)
??plot                 (* Visualization *)
??stats                (* Statistics *)
??calculus             (* Calculus operations *)
??linear               (* Linear algebra *)
```

### Interactive Discovery

```lyra
(* Tab completion *)
Str<Tab>               (* Shows: StringJoin, StringLength, StringTake, ... *)
Plot<Tab>              (* Shows: Plot, PlotRange, PlotStyle, ... *)

(* Function information *)
Information[StringJoin] (* Detailed function documentation *)
Options[Plot]          (* Show all plot options *)
```

## 🛠️ Common Workflows

### Data Analysis Pipeline

```lyra
(* 1. Import data *)
data = Import["dataset.csv"]

(* 2. Explore structure *)
Dimensions[data]        (* Size of dataset *)
Head[data, 5]          (* First 5 rows *)
Keys[data[[1]]]        (* Column names *)

(* 3. Clean data *)
cleaned = DeleteMissing[data]
filtered = Select[cleaned, NumericQ[#["value"]] &]

(* 4. Analyze *)
summary_stats = {
  "mean" -> Mean[filtered[[All, "value"]]],
  "std" -> StandardDeviation[filtered[[All, "value"]]],
  "count" -> Length[filtered]
}

(* 5. Visualize *)
Histogram[filtered[[All, "value"]]]
```

### Mathematical Modeling

```lyra
(* Define model *)
model[x_, a_, b_, c_] := a*x^2 + b*x + c

(* Generate sample data *)
true_params = {a -> 2, b -> -1, c -> 0.5}
x_data = Range[0, 10, 0.5]
y_data = Map[model[#, 2, -1, 0.5] + RandomReal[{-0.1, 0.1}] &, x_data]

(* Fit model to data *)
fitted = FindFit[Transpose[{x_data, y_data}], model[x, a, b, c], {a, b, c}, x]

(* Visualize results *)
Show[
  ListPlot[Transpose[{x_data, y_data}]],
  Plot[model[x, a, b, c] /. fitted, {x, 0, 10}]
]
```

### Custom Function Development

```lyra
(* Define reusable functions *)
standardize[data_List] := (data - Mean[data])/StandardDeviation[data]

outliers[data_List] := Module[{q1, q3, iqr},
  q1 = Quantile[data, 0.25];
  q3 = Quantile[data, 0.75];
  iqr = q3 - q1;
  Select[data, #< q1 - 1.5*iqr || # > q3 + 1.5*iqr &]
]

(* Use your functions *)
sample_data = RandomReal[NormalDistribution[0, 1], 1000]
standardized = standardize[sample_data]
outlier_points = outliers[sample_data]
```

## 🎯 Next Steps

### Explore Advanced Features

```lyra
(* Parallel computing *)
ParallelMap[expensiveFunction, largeDataset]

(* Machine learning *)
model = LinearLayer[10, 1]
trained = TrainModel[model, trainingData]

(* Advanced mathematics *)
DSolve[y'[x] == y[x], y[x], x]  (* Differential equations *)
Minimize[x^2 + y^2, {x, y}]     (* Optimization *)

(* Database integration *)
db = OpenDatabase["postgresql://localhost/mydb"]
Query[db, "SELECT * FROM users WHERE age > 21"]
```

### Learn More

- **Examples**: Explore `examples/` directory for comprehensive examples
- **Documentation**: Check `docs/` for detailed guides
- **Language Reference**: See `docs/language-reference.md`
- **API Documentation**: Browse `docs/api/` for developer information

### Key Resources

```lyra
(* Built-in tutorials *)
??tutorial              (* Interactive tutorials *)
??examples             (* Browse example code *)

(* Community and development *)
??github               (* Link to GitHub repository *)
??contribute           (* How to contribute *)
```

## 🐛 Troubleshooting

### Common Issues

**Syntax Errors:**
```lyra
(* Wrong: f(x, y) - uses parentheses *)
f(x, y)

(* Correct: f[x, y] - uses square brackets *)
f[x, y]
```

**Assignment vs Evaluation:**
```lyra
(* Immediate evaluation *)
x = RandomReal[]       (* x gets a specific value *)
x                      (* Always returns same value *)

(* Delayed evaluation *)
y := RandomReal[]      (* y re-evaluates each time *)
y                      (* Returns new random value *)
```

**List vs Function Call:**
```lyra
{1, 2, 3}             (* List of three elements *)
f[1, 2, 3]            (* Function call with three arguments *)
```

### Getting Unstuck

```lyra
(* Check what went wrong *)
$Failed               (* Last failed expression *)
?$MessageList         (* Recent error messages *)

(* Reset if needed *)
Clear[x]              (* Clear specific variable *)
ClearAll["Global`*"] (* Clear all user variables *)
```

---

**You're Ready to Go!** 

This guide covers the essentials to get you productive with Lyra. The key to mastery is experimentation - try the examples, modify them, and explore the extensive help system. Lyra's power comes from combining simple concepts in sophisticated ways.

**Happy computing with Lyra!** 🚀