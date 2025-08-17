# Lyra API Documentation

## Overview

This document provides comprehensive API documentation for the Lyra symbolic computation engine. The API is organized into three main modules: lexer, parser, and AST.

## Module: `lexer`

The lexer module provides tokenization functionality for Lyra source code.

### Types

#### `Token`
Represents a single token in the source code.

```rust
pub struct Token {
    pub kind: TokenKind,
    pub position: usize,
    pub length: usize,
}
```

**Fields:**
- `kind`: The type of token (see `TokenKind`)
- `position`: Byte position in the source where the token starts
- `length`: Length of the token in bytes

#### `TokenKind`
Enumeration of all possible token types.

```rust
pub enum TokenKind {
    // Literals
    Integer(i64),
    Real(f64),
    Rational(i64, i64),
    Complex(f64, f64),
    BigInt(String),
    BigDecimal(String),
    HexInteger(String),
    String(String),
    InterpolatedString(Vec<InterpolationPart>),
    Symbol(String),
    ContextSymbol(String),
    
    // Operators
    Plus, Minus, Times, Divide, Power,
    
    // Comparison
    Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual,
    
    // Logical
    And, Or, Not,
    
    // Assignment and Rules
    Set, SetDelayed, Rule, RuleDelayed, ReplaceAll,
    
    // Modern operators
    Pipeline, Postfix, Prefix, Arrow, Range, Condition, Alternative,
    
    // Grouping
    LeftParen, RightParen, LeftBracket, RightBracket,
    LeftBrace, RightBrace,
    
    // Associations
    LeftAssoc, RightAssoc,
    
    // Part access
    LeftDoubleBracket, RightDoubleBracket,
    
    // Separators
    Comma, Semicolon, Dot, Colon, Question,
    
    // Patterns
    Blank, BlankSequence, BlankNullSequence,
    
    // Special
    StringJoin, Backtick,
    
    // Whitespace and comments
    Whitespace, Comment(String),
    
    // End of input
    Eof,
}
```

#### `Lexer`
The main lexer struct for tokenizing source code.

```rust
pub struct Lexer<'a> {
    input: &'a str,
    position: usize,
    current_char: Option<char>,
}
```

### Methods

#### `Lexer::new(input: &str) -> Self`
Creates a new lexer for the given input string.

**Parameters:**
- `input`: The source code string to tokenize

**Returns:** A new `Lexer` instance

#### `Lexer::tokenize(&mut self) -> Result<Vec<Token>>`
Tokenizes the entire input and returns a vector of tokens.

**Returns:** 
- `Ok(Vec<Token>)`: Successfully tokenized input
- `Err(Error)`: Lexical error encountered

**Example:**
```rust
let mut lexer = Lexer::new("f[x] = x^2");
let tokens = lexer.tokenize()?;
```

## Module: `parser`

The parser module provides parsing functionality to convert tokens into an Abstract Syntax Tree (AST).

### Types

#### `Parser`
The main parser struct for building ASTs from tokens.

```rust
pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
}
```

### Methods

#### `Parser::new(tokens: Vec<Token>) -> Self`
Creates a new parser with the given tokens.

**Parameters:**
- `tokens`: Vector of tokens from the lexer

**Returns:** A new `Parser` instance

#### `Parser::parse(&mut self) -> Result<Vec<Expr>>`
Parses the tokens into a vector of expressions.

**Returns:**
- `Ok(Vec<Expr>)`: Successfully parsed expressions
- `Err(Error)`: Parse error encountered

**Example:**
```rust
let mut lexer = Lexer::new("x |> f |> g[2]");
let tokens = lexer.tokenize()?;
let mut parser = Parser::new(tokens);
let ast = parser.parse()?;
```

#### `Parser::parse_expression(&mut self) -> Result<Expr>`
Parses a single expression from the current position.

**Returns:**
- `Ok(Expr)`: Successfully parsed expression
- `Err(Error)`: Parse error encountered

## Module: `ast`

The AST module defines the Abstract Syntax Tree structures for representing Lyra expressions.

### Types

#### `Expr`
The main expression type representing all possible Lyra expressions.

```rust
pub enum Expr {
    Symbol(Symbol),
    Number(Number),
    String(String),
    List(Vec<Expr>),
    Function { head: Box<Expr>, args: Vec<Expr> },
    Pattern(Pattern),
    Rule { lhs: Box<Expr>, rhs: Box<Expr>, delayed: bool },
    Assignment { lhs: Box<Expr>, rhs: Box<Expr>, delayed: bool },
    Replace { expr: Box<Expr>, rules: Box<Expr> },
    
    // Modern syntax extensions
    Association(Vec<(Expr, Expr)>),
    Pipeline { stages: Vec<Expr> },
    DotCall { object: Box<Expr>, method: String, args: Vec<Expr> },
    Range { start: Box<Expr>, end: Box<Expr>, step: Option<Box<Expr>> },
    ArrowFunction { params: Vec<String>, body: Box<Expr> },
    InterpolatedString(Vec<InterpolationPart>),
}
```

#### `Symbol`
Represents a symbolic identifier.

```rust
pub struct Symbol {
    pub name: String,
}
```

#### `Number`
Represents numeric literals.

```rust
pub enum Number {
    Integer(i64),
    Real(f64),
}
```

#### `Pattern`
Represents pattern matching constructs.

```rust
pub enum Pattern {
    Blank { head: Option<String> },
    BlankSequence { head: Option<String> },
    BlankNullSequence { head: Option<String> },
    Named { name: String, pattern: Box<Pattern> },
    
    // Modern pattern extensions
    Typed { name: String, type_pattern: Box<Expr> },
    Predicate { pattern: Box<Pattern>, test: Box<Expr> },
    Alternative { patterns: Vec<Pattern> },
    Conditional { pattern: Box<Pattern>, condition: Box<Expr> },
}
```

### Constructor Methods

The `Expr` type provides convenient constructor methods:

#### Basic Constructors

```rust
impl Expr {
    pub fn symbol(name: impl Into<String>) -> Self
    pub fn integer(value: i64) -> Self
    pub fn real(value: f64) -> Self
    pub fn string(value: impl Into<String>) -> Self
    pub fn list(items: Vec<Expr>) -> Self
    pub fn function(head: Expr, args: Vec<Expr>) -> Self
    pub fn rule(lhs: Expr, rhs: Expr, delayed: bool) -> Self
    pub fn assignment(lhs: Expr, rhs: Expr, delayed: bool) -> Self
    pub fn replace(expr: Expr, rules: Expr) -> Self
}
```

#### Pattern Constructors

```rust
impl Expr {
    pub fn blank(head: Option<String>) -> Self
    pub fn blank_sequence(head: Option<String>) -> Self
    pub fn blank_null_sequence(head: Option<String>) -> Self
    pub fn typed_pattern(name: impl Into<String>, type_pattern: Expr) -> Self
    pub fn predicate_pattern(pattern: Pattern, test: Expr) -> Self
    pub fn alternative_pattern(patterns: Vec<Pattern>) -> Self
    pub fn conditional_pattern(pattern: Pattern, condition: Expr) -> Self
}
```

#### Modern Syntax Constructors

```rust
impl Expr {
    pub fn association(pairs: Vec<(Expr, Expr)>) -> Self
    pub fn pipeline(stages: Vec<Expr>) -> Self
    pub fn dot_call(object: Expr, method: impl Into<String>, args: Vec<Expr>) -> Self
    pub fn range(start: Expr, end: Expr, step: Option<Expr>) -> Self
    pub fn arrow_function(params: Vec<String>, body: Expr) -> Self
    pub fn interpolated_string(parts: Vec<InterpolationPart>) -> Self
}
```

### Display Implementation

All AST types implement `Display` for pretty-printing:

```rust
// Example usage
let expr = Expr::pipeline(vec![
    Expr::symbol("x"),
    Expr::symbol("f"),
    Expr::function(Expr::symbol("g"), vec![Expr::integer(2)])
]);
println!("{}", expr); // Output: x |> f |> g[2]
```

## Error Handling

All parsing operations return `Result<T, Error>` where `Error` is defined as:

```rust
pub enum Error {
    Lexer { message: String, position: usize },
    Parser { message: String, position: usize },
}
```

### Error Types

- **Lexer errors**: Invalid characters, unterminated strings, malformed numbers
- **Parser errors**: Unexpected tokens, missing brackets, invalid syntax

## Usage Examples

### Basic Parsing

```rust
use lyra::{Lexer, Parser};

fn parse_expression(input: &str) -> Result<Vec<Expr>, Error> {
    let mut lexer = Lexer::new(input);
    let tokens = lexer.tokenize()?;
    let mut parser = Parser::new(tokens);
    parser.parse()
}

// Parse a simple expression
let ast = parse_expression("f[x, y]")?;

// Parse modern syntax
let ast = parse_expression("data |> Map[(x) => x^2] |> Select[# > 10&]")?;
```

### Building AST Programmatically

```rust
use lyra::ast::Expr;

// Build: f[x] = x^2
let definition = Expr::assignment(
    Expr::function(
        Expr::symbol("f"),
        vec![Expr::blank(None)]
    ),
    Expr::function(
        Expr::symbol("Power"),
        vec![Expr::symbol("x"), Expr::integer(2)]
    ),
    false // not delayed
);

println!("{}", definition); // Output: f[_] = Power[x, 2]
```

### Modern Syntax Examples

```rust
// Association: <|"name" -> "Ada", "age" -> 37|>
let person = Expr::association(vec![
    (Expr::string("name"), Expr::string("Ada")),
    (Expr::string("age"), Expr::integer(37))
]);

// Pipeline: x |> f |> g[2]
let pipeline = Expr::pipeline(vec![
    Expr::symbol("x"),
    Expr::symbol("f"),
    Expr::function(Expr::symbol("g"), vec![Expr::integer(2)])
]);

// Arrow function: (x, y) => x + y
let arrow_fn = Expr::arrow_function(
    vec!["x".to_string(), "y".to_string()],
    Expr::function(
        Expr::symbol("Plus"),
        vec![Expr::symbol("x"), Expr::symbol("y")]
    )
);

// Range: 1;;10;;2
let range = Expr::range(
    Expr::integer(1),
    Expr::integer(10),
    Some(Expr::integer(2))
);
```

## Performance Considerations

- The lexer processes input in a single pass with O(n) time complexity
- The parser uses recursive descent with proper precedence handling
- Memory usage is proportional to the size of the input and AST depth
- All string operations use efficient Rust `String` types
- Token vectors are pre-allocated to minimize reallocations

## Thread Safety

- `Lexer` and `Parser` are not `Send` or `Sync` due to internal state
- `Expr`, `Token`, and other data types are safe to share between threads
- For concurrent parsing, create separate lexer/parser instances per thread

## Version Compatibility

This API is designed for Lyra 0.1.x and follows semantic versioning:
- Patch versions (0.1.x) maintain full API compatibility
- Minor versions may add new features while maintaining backward compatibility
- Major versions may introduce breaking changes