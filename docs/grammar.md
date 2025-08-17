# Lyra Language Grammar Specification

## Overview
Lyra follows Wolfram Language syntax conventions with square brackets for function calls, braces for lists, and a rich pattern matching system.

## EBNF Grammar

```ebnf
(* Root production *)
program = { statement } ;

statement = definition | expression ;

(* Definitions *)
definition = set_definition | set_delayed_definition ;
set_definition = lhs "=" rhs ;
set_delayed_definition = lhs ":=" rhs ;

(* Expressions *)
expression = replacement_expr ;

replacement_expr = rule_expr [ "/." rule_expr ] ;

rule_expr = or_expr { ( "->" | ":>" ) or_expr } ;

or_expr = and_expr { "||" and_expr } ;

and_expr = equality_expr { "&&" equality_expr } ;

equality_expr = relational_expr { ( "==" | "!=" ) relational_expr } ;

relational_expr = additive_expr { ( "<" | "<=" | ">" | ">=" ) additive_expr } ;

additive_expr = multiplicative_expr { ( "+" | "-" ) multiplicative_expr } ;

multiplicative_expr = power_expr { ( "*" | "/" ) power_expr } ;

power_expr = unary_expr [ "^" power_expr ] ;

unary_expr = ( "+" | "-" | "!" ) unary_expr | postfix_expr ;

postfix_expr = primary_expr { postfix_op } ;

postfix_op = function_call | part_access ;

function_call = "[" [ expression_list ] "]" ;

part_access = "[[" expression_list "]]" ;

primary_expr = number | string | symbol | list | "(" expression ")" ;

(* Literals *)
number = integer | real ;
integer = digit { digit } ;
real = digit { digit } "." digit { digit } [ exponent ] ;
exponent = ( "e" | "E" ) [ "+" | "-" ] digit { digit } ;

string = '"' { string_char } '"' ;
string_char = ? any character except '"' ? | '\"' ;

symbol = symbol_name [ pattern_suffix ] ;
symbol_name = ( letter | "$" ) { letter | digit | "$" } ;
pattern_suffix = "_" [ symbol_name ] | "__" [ symbol_name ] | "___" [ symbol_name ] ;

list = "{" [ expression_list ] "}" ;

expression_list = expression { "," expression } ;

(* Basic character classes *)
letter = "a" | "b" | ... | "z" | "A" | "B" | ... | "Z" ;
digit = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;

(* Left-hand side for definitions *)
lhs = symbol | function_pattern ;
rhs = expression ;

function_pattern = symbol "[" [ pattern_list ] "]" ;
pattern_list = pattern { "," pattern } ;
pattern = symbol | blank_pattern ;
blank_pattern = "_" [ symbol_name ] ;
```

## Operator Precedence (highest to lowest)

1. `[[]]` (Part access)
2. `[]` (Function call)
3. `^` (Power) - right associative
4. `+`, `-` (Unary)
5. `*`, `/` (Multiplication, Division)
6. `+`, `-` (Addition, Subtraction)
7. `<`, `<=`, `>`, `>=` (Relational)
8. `==`, `!=` (Equality)
9. `&&` (And)
10. `||` (Or)
11. `->`, `:>` (Rules)
12. `/.` (ReplaceAll)
13. `=`, `:=` (Assignment)

## Syntax Examples

### Basic Expressions
```
2 + 3 * 4         (* 14 *)
f[x, y]           (* Function call *)
{1, 2, 3}         (* List *)
"hello"           (* String *)
```

### Patterns
```
x_                (* Blank pattern *)
x_Integer         (* Typed pattern *)
x__               (* BlankSequence *)
x___              (* BlankNullSequence *)
```

### Rules and Replacement
```
x -> x^2          (* Rule *)
x :> RandomReal[] (* RuleDelayed *)
expr /. rule      (* ReplaceAll *)
```

### Definitions
```
f[x_] = x^2       (* Immediate definition *)
f[x_] := Print[x] (* Delayed definition *)
```

### Complex Expressions
```
f[g[x, y], {1, 2, 3}]
{x, y, z} /. {x -> 1, y -> 2}
f[x_Integer] := x + 1
```

## Lexical Conventions

### Comments
- `(* comment *)` - Block comments (can be nested)

### Whitespace
- Spaces, tabs, newlines are generally ignored
- Implicit multiplication by juxtaposition: `2 x` means `2 * x`

### Reserved Words
None - all identifiers are valid symbols.

### Special Characters
- `[]` - Function calls
- `{}` - Lists
- `()` - Grouping
- `_` - Pattern markers
- `->`, `:>` - Rules
- `/.` - ReplaceAll
- `=`, `:=` - Definitions
- `[[]]` - Part access

## Desugaring Rules

1. Infix operators become function calls:
   - `a + b` → `Plus[a, b]`
   - `a * b` → `Times[a, b]`
   - `a ^ b` → `Power[a, b]`

2. Implicit multiplication:
   - `2 x` → `Times[2, x]`
   - `f[x] g[y]` → `Times[f[x], g[y]]`

3. String concatenation:
   - `a <> b` → `StringJoin[a, b]`

4. Comparison chains:
   - `a < b < c` → `Less[a, b, c]`

5. Unary operators:
   - `-x` → `Times[-1, x]`
   - `!x` → `Not[x]`