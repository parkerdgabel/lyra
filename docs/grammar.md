# Lyra Language Grammar Specification

## Overview
Lyra follows modern Wolfram Language syntax conventions with square brackets for function calls, braces for lists, and advanced pattern matching. It extends traditional WL syntax with modern features like pipelines, dot-call syntax, and associations.

## Complete EBNF Grammar

```ebnf
(* Root production *)
program = { statement } ;

statement = definition | expression ;

(* Definitions *)
definition = set_definition | set_delayed_definition ;
set_definition = lhs "=" rhs ;
set_delayed_definition = lhs ":=" rhs ;

(* Expressions with modern extensions *)
expression = pipeline_expr ;

pipeline_expr = replacement_expr { "|>" replacement_expr } ;

replacement_expr = rule_expr [ "/." rule_expr ] ;

rule_expr = conditional_expr { ( "->" | ":>" ) conditional_expr } ;

conditional_expr = or_expr [ "/;" or_expr ] ;

or_expr = and_expr { "||" and_expr } ;

and_expr = alternative_expr { "&&" alternative_expr } ;

alternative_expr = equality_expr { "|" equality_expr } ;

equality_expr = relational_expr { ( "==" | "!=" ) relational_expr } ;

relational_expr = additive_expr { ( "<" | "<=" | ">" | ">=" ) additive_expr } ;

additive_expr = multiplicative_expr { ( "+" | "-" ) multiplicative_expr } ;

multiplicative_expr = power_expr { ( "*" | "/" ) power_expr } ;

power_expr = unary_expr [ "^" power_expr ] ;

unary_expr = ( "+" | "-" | "!" | "@" ) unary_expr | postfix_expr ;

postfix_expr = primary_expr { postfix_op } ;

postfix_op = function_call | part_access | dot_call | postfix_apply ;

function_call = "[" [ expression_list ] "]" ;

part_access = "[[" expression_list "]]" ;

dot_call = "." symbol_name "[" [ expression_list ] "]" ;

postfix_apply = "//" ;

primary_expr = number | string | interpolated_string | symbol | 
               association | list | range | arrow_function | "(" expression ")" ;

(* Modern literal types *)
number = integer | real | rational | big_integer | big_decimal | hex_integer ;

integer = digit { digit } ;

real = digit { digit } "." digit { digit } [ exponent ] ;

rational = integer "/" integer ;

big_integer = integer "n" ;

big_decimal = real "d" digit { digit } ;

hex_integer = integer "^^" hex_digit { hex_digit } ;

exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } ;

(* String types *)
string = '"' { string_char } '"' ;

interpolated_string = '"' { interpolation_part } '"' ;

interpolation_part = string_char | interpolation_expr ;

interpolation_expr = "#{" expression "}" ;

string_char = ? any character except '"' ? | '\"' ;

(* Symbol and context notation *)
symbol = [ context ] symbol_name [ pattern_suffix ] ;

context = symbol_name "`" { symbol_name "`" } ;

symbol_name = ( letter | "$" ) { letter | digit | "$" | "_" } ;

(* Modern pattern extensions *)
pattern_suffix = blank_pattern | typed_pattern | predicate_pattern ;

blank_pattern = "_" [ symbol_name ] | "__" [ symbol_name ] | "___" [ symbol_name ] ;

typed_pattern = ":" expression ;

predicate_pattern = "?" expression ;

(* Modern data structures *)
association = "<|" [ pair_list ] "|>" ;

pair_list = pair { "," pair } ;

pair = expression "->" expression ;

list = "{" [ expression_list ] "}" ;

range = expression ";;" expression [ ";;" expression ] ;

arrow_function = "(" [ parameter_list ] ")" "=>" expression ;

parameter_list = symbol_name { "," symbol_name } ;

expression_list = expression { "," expression } ;

(* Pattern definitions *)
pattern = blank_pattern | typed_pattern | predicate_pattern | 
          alternative_pattern | conditional_pattern | named_pattern ;

alternative_pattern = pattern { "|" pattern } ;

conditional_pattern = pattern "/;" expression ;

named_pattern = symbol_name pattern ;

(* Basic character classes *)
letter = "a" | "b" | ... | "z" | "A" | "B" | ... | "Z" ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
hex_digit = digit | "A" | "B" | "C" | "D" | "E" | "F" | "a" | "b" | "c" | "d" | "e" | "f" ;

(* Definition patterns *)
lhs = symbol | function_pattern ;
rhs = expression ;

function_pattern = symbol "[" [ pattern_list ] "]" ;
pattern_list = pattern { "," pattern } ;
```

## Operator Precedence (highest to lowest)

1. `[[]]` (Part access)
2. `[]` (Function call), `.method[]` (Dot call)
3. `//` (Postfix apply)
4. `@` (Prefix apply)
5. `^` (Power) - right associative
6. `+`, `-` (Unary)
7. `*`, `/` (Multiplication, Division)
8. `+`, `-` (Addition, Subtraction)
9. `<`, `<=`, `>`, `>=` (Relational)
10. `==`, `!=` (Equality)
11. `|` (Alternative patterns)
12. `&&` (And)
13. `||` (Or)
14. `/;` (Conditional patterns)
15. `->`, `:>` (Rules)
16. `/.` (ReplaceAll)
17. `|>` (Pipeline)
18. `=`, `:=` (Assignment)
19. `=>` (Arrow functions)

## Syntax Examples

### Basic Expressions
```
2 + 3 * 4         (* 14 *)
f[x, y]           (* Function call *)
{1, 2, 3}         (* List *)
"hello"           (* String *)
123n              (* BigInt *)
16^^FF            (* Hex number *)
3/4               (* Rational *)
```

### Modern Data Structures
```
<|"name" -> "Ada", "age" -> 37|>    (* Association *)
1;;10                               (* Range *)
0;;1;;0.1                          (* Range with step *)
"Hello #{name}!"                   (* String interpolation *)
```

### Pipelines and Modern Operators
```
x |> f |> g[2]                     (* Pipeline *)
obj.method[args]                   (* Dot call *)
f @ x                              (* Prefix apply *)
x // f                             (* Postfix apply *)
(x) => x + 1                       (* Arrow function *)
```

### Advanced Patterns
```
x_                (* Blank pattern *)
x_Integer         (* Typed pattern - legacy *)
x:_Integer        (* Typed pattern - modern *)
x_?Positive       (* Predicate pattern *)
x_ /; x > 0       (* Conditional pattern *)
x | y | z         (* Alternative pattern *)
```

### Rules and Replacement
```
x -> x^2          (* Rule *)
x :> RandomReal[] (* RuleDelayed *)
expr /. rule      (* ReplaceAll *)
```

### Function Definitions
```
f[x_] = x^2       (* Immediate definition *)
f[x_] := Print[x] (* Delayed definition *)
square = (x) => x^2                (* Arrow function *)
```

### Context and Namespaces
```
std`net`http`Get                   (* Context symbol *)
Using["std`data`"];               (* Import *)
```

### Complex Modern Expressions
```
data = <|"values" -> {1, 2, 3, 4}|>
     |> #["values"]&
     |> Map[(x) => x^2]
     |> Select[# > 5&]

result = numbers
       |> Filter[EvenQ]
       |> Map[x -> x * 2]
       |> data.transform[(x) => x + offset]
```

## Lexical Conventions

### Comments
- `(* comment *)` - Block comments (can be nested)  
- `// comment` - Line comments (to end of line)

### Whitespace
- Spaces, tabs, newlines are generally ignored
- Implicit multiplication by juxtaposition: `2 x` means `2 * x`
- Newlines can separate expressions (semicolon optional)

### Reserved Words
None - all identifiers are valid symbols.

### Special Characters
- `[]` - Function calls
- `{}` - Lists  
- `<||>` - Associations
- `()` - Grouping, arrow function parameters
- `_` - Pattern markers
- `->`, `:>` - Rules
- `/.` - ReplaceAll
- `=`, `:=` - Definitions
- `[[]]` - Part access
- `|>` - Pipeline
- `=>` - Arrow functions
- `@` - Prefix apply
- `//` - Postfix apply
- `;;` - Range
- `/;` - Conditional patterns
- `:` - Type annotations
- `?` - Predicate patterns
- `` ` `` - Context separators

### Number Formats
- **Integer**: `42`, `-17`
- **Real**: `3.14`, `2.5e-3`, `.5`
- **Rational**: `3/4` (exact fractions)
- **BigInt**: `123n` (arbitrary precision)
- **BigDecimal**: `1.23d100` (arbitrary precision decimal)
- **Hex**: `16^^FF`, `2^^1010` (base notation)

### String Types
- **Simple**: `"hello world"`
- **Interpolated**: `"Hello #{name}!"` → `StringJoin["Hello ", name, "!"]`
- **Escape sequences**: `\"`, `\\`, `\n`, `\t`, `\r`

## Desugaring Rules

1. **Infix operators** become function calls:
   - `a + b` → `Plus[a, b]`
   - `a * b` → `Times[a, b]`
   - `a ^ b` → `Power[a, b]`

2. **Modern operators**:
   - `x |> f` → `f[x]`
   - `x |> f |> g` → `g[f[x]]`
   - `obj.method[args]` → `method[obj, args]`
   - `f @ x` → `f[x]`
   - `x // f` → `f[x]`

3. **Implicit multiplication**:
   - `2 x` → `Times[2, x]`
   - `f[x] g[y]` → `Times[f[x], g[y]]`

4. **String operations**:
   - `a <> b` → `StringJoin[a, b]`
   - `"#{expr}"` → `StringJoin[ToString[expr]]`

5. **Range expansion**:
   - `1;;10` → `Range[1, 10]`
   - `0;;1;;0.1` → `Range[0, 1, 0.1]`

6. **Pattern transformations**:
   - `x_?test` → `Pattern[x, Condition[Blank[], test]]`
   - `pat /; cond` → `Condition[pat, cond]`
   - `x:_Integer` → `Pattern[x, Blank[Integer]]`

7. **Arrow functions**:
   - `(x) => expr` → `Function[{x}, expr]`
   - `(x, y) => expr` → `Function[{x, y}, expr]`

8. **Associations**:
   - `<|a->b, c->d|>` → `Association[Rule[a,b], Rule[c,d]]`

9. **Comparison chains**:
   - `a < b < c` → `Less[a, b, c]`

10. **Unary operators**:
    - `-x` → `Times[-1, x]`
    - `!x` → `Not[x]`
    - `@f` → `f` (prefix form)