use crate::bytecode::{Instruction, OpCode};
use crate::foreign::LyObj;
use crate::linker::{registry::create_global_registry, FunctionRegistry};
use crate::stdlib::StandardLibrary;
use crate::ast::{Expr, Symbol, Number};
// Tensor operations imported when needed
use std::collections::HashMap;
use thiserror::Error;
use serde::{Serialize, Deserialize, Serializer, Deserializer};
// ArrayD imported when needed for tensor operations

#[derive(Error, Debug, Serialize, Deserialize)]
pub enum VmError {
    #[error("Stack underflow")]
    StackUnderflow,
    #[error("Invalid instruction pointer: {0}")]
    InvalidInstructionPointer(usize),
    #[error("Invalid constant index: {0}")]
    InvalidConstantIndex(usize),
    #[error("Invalid symbol index: {0}")]
    InvalidSymbolIndex(usize),
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Type error: expected {expected}, got {actual}")]
    TypeError { expected: String, actual: String },
    #[error("Wrong number of arguments: {function_name} expects {expected}, got {actual}")]
    ArityError { function_name: String, expected: usize, actual: usize },
    #[error("Argument type error in {function_name}: parameter {param_index} expects {expected}, got {actual}")]
    ArgumentTypeError { 
        function_name: String, 
        param_index: usize, 
        expected: String, 
        actual: String 
    },
    #[error("Unknown function: {function_name}")]
    UnknownFunction { function_name: String },
    #[error("Call stack overflow")]
    CallStackOverflow,
    #[error("Cannot call non-function value")]
    NotCallable,
    #[error("Index {index} out of bounds for length {length}")]
    IndexError { index: i64, length: usize },
    #[error("Runtime error: {0}")]
    Runtime(String),
    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

pub type VmResult<T> = std::result::Result<T, VmError>;

/// Conversion from ForeignError to VmError for spatial module compatibility
impl From<crate::foreign::ForeignError> for VmError {
    fn from(err: crate::foreign::ForeignError) -> Self {
        VmError::Runtime(err.to_string())
    }
}

/// A value that can be stored on the VM stack
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Real(f64),
    /// Exact rational number (numerator, denominator); denominator != 0
    Rational(i64, i64),
    /// Arbitrary-precision style real placeholder: value with precision bits
    BigReal { value: f64, precision: u32 },
    /// Complex number over the numeric tower; components preserve exactness
    Complex { re: Box<Value>, im: Box<Value> },
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String), // Built-in function name for now
    Boolean(bool),
    Missing,            // Missing/unknown value (distinct from Null)
    Object(HashMap<String, Value>), // Object/dictionary type for structured data
    LyObj(LyObj),       // Foreign object wrapper for complex types (Series, Table, Dataset, Schema, Tensor)
    /// Unified expression node: Head[args...] represented as values
    Expr { head: Box<Value>, args: Vec<Value> },
    Quote(Box<crate::ast::Expr>), // Unevaluated expression for Hold attributes
    Pattern(crate::ast::Pattern), // Pattern expressions for pattern matching
    Rule { lhs: Box<Value>, rhs: Box<Value> }, // Rule expressions for transformations
    PureFunction { body: Box<Value> }, // Pure function with slot placeholders
    Slot { number: Option<usize> }, // Slot placeholder (#, #1, #2, etc.)
}

// Use simpler derive-based serialization with a custom implementation for LyObj
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
enum ValueSerde {
    Integer(i64),
    Real(f64),
    Rational(i64, i64),
    BigReal { value: f64, precision: u32 },
    Complex { re: Box<Value>, im: Box<Value> },
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String),
    Boolean(bool),
    Missing,
    Object(HashMap<String, Value>),
    LyObjPlaceholder { type_name: String }, // Simplified LyObj representation  
    Expr { head: Box<Value>, args: Vec<Value> },
    Quote(Box<crate::ast::Expr>),
    Pattern(crate::ast::Pattern),
    Rule { lhs: Box<Value>, rhs: Box<Value> },
    PureFunction { body: Box<Value> },
    Slot { number: Option<usize> },
}

// Custom serialization for Value enum to handle LyObj
impl Serialize for Value {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let serde_value = match self {
            Value::Integer(n) => ValueSerde::Integer(*n),
            Value::Real(f) => ValueSerde::Real(*f),
            Value::Rational(n, d) => ValueSerde::Rational(*n, *d),
            Value::BigReal { value, precision } => ValueSerde::BigReal { value: *value, precision: *precision },
            Value::Complex { re, im } => ValueSerde::Complex { re: re.clone(), im: im.clone() },
            Value::String(s) => ValueSerde::String(s.clone()),
            Value::Symbol(s) => ValueSerde::Symbol(s.clone()),
            Value::List(items) => ValueSerde::List(items.clone()),
            Value::Function(name) => ValueSerde::Function(name.clone()),
            Value::Boolean(b) => ValueSerde::Boolean(*b),
            Value::Missing => ValueSerde::Missing,
            Value::Object(obj) => ValueSerde::Object(obj.clone()),
            Value::LyObj(obj) => ValueSerde::LyObjPlaceholder { 
                type_name: obj.type_name().to_string() 
            },
            Value::Expr { head, args } => ValueSerde::Expr { head: head.clone(), args: args.clone() },
            Value::Quote(expr) => ValueSerde::Quote(expr.clone()),
            Value::Pattern(pat) => ValueSerde::Pattern(pat.clone()),
            Value::Rule { lhs, rhs } => ValueSerde::Rule { lhs: lhs.clone(), rhs: rhs.clone() },
            Value::PureFunction { body } => ValueSerde::PureFunction { body: body.clone() },
            Value::Slot { number } => ValueSerde::Slot { number: *number },
        };
        serde_value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Value {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let serde_value = ValueSerde::deserialize(deserializer)?;
        let value = match serde_value {
            ValueSerde::Integer(n) => Value::Integer(n),
            ValueSerde::Real(f) => Value::Real(f),
            ValueSerde::Rational(n, d) => Value::Rational(n, d),
            ValueSerde::BigReal { value, precision } => Value::BigReal { value, precision },
            ValueSerde::Complex { re, im } => Value::Complex { re, im },
            ValueSerde::String(s) => Value::String(s),
            ValueSerde::Symbol(s) => Value::Symbol(s),
            ValueSerde::List(items) => Value::List(items),
            ValueSerde::Function(name) => Value::Function(name),
            ValueSerde::Boolean(b) => Value::Boolean(b),
            ValueSerde::Missing => Value::Missing,
            ValueSerde::Object(obj) => Value::Object(obj),
            ValueSerde::LyObjPlaceholder { type_name: _ } => {
                // For now, deserialize LyObj placeholders as Missing
                // Real implementation would need registry-based reconstruction
                Value::Missing
            }
            ValueSerde::Expr { head, args } => Value::Expr { head, args },
            ValueSerde::Quote(expr) => Value::Quote(expr),
            ValueSerde::Pattern(pat) => Value::Pattern(pat),
            ValueSerde::Rule { lhs, rhs } => Value::Rule { lhs, rhs },
            ValueSerde::PureFunction { body } => Value::PureFunction { body },
            ValueSerde::Slot { number } => Value::Slot { number },
        };
        Ok(value)
    }
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Integer(n) => {
                0u8.hash(state);
                n.hash(state);
            },
            Value::Real(f) => {
                1u8.hash(state);
                // Hash the bit representation of f64 for consistent hashing
                f.to_bits().hash(state);
            },
            Value::Rational(n, d) => {
                2u8.hash(state);
                n.hash(state);
                d.hash(state);
            },
            Value::BigReal { value, precision } => {
                3u8.hash(state);
                value.to_bits().hash(state);
                precision.hash(state);
            },
            Value::Complex { re, im } => {
                4u8.hash(state);
                re.hash(state);
                im.hash(state);
            },
            Value::String(s) => {
                5u8.hash(state);
                s.hash(state);
            },
            Value::Symbol(s) => {
                6u8.hash(state);
                s.hash(state);
            },
            Value::List(items) => {
                7u8.hash(state);
                items.hash(state);
            },
            Value::Function(name) => {
                8u8.hash(state);
                name.hash(state);
            },
            Value::Boolean(b) => {
                9u8.hash(state);
                b.hash(state);
            },
            Value::Missing => {
                10u8.hash(state);
            },
            Value::Object(obj) => {
                11u8.hash(state);
                // Fast path: hash size first for quick differentiation
                obj.len().hash(state);
                
                // Hash entries without allocation (deterministic order)
                let mut keys: Vec<_> = obj.keys().collect();
                keys.sort_unstable();
                for key in keys {
                    key.hash(state);
                    if let Some(value) = obj.get(key) {
                        value.hash(state);
                    }
                }
            },
            Value::LyObj(obj) => {
                12u8.hash(state);
                // Fast hash using just type name (avoid expensive debug formatting)
                obj.type_name().hash(state);
                // Use a simple hash based on the object's type and debug representation
                // This is a fallback until proper object identity is implemented
                format!("{:?}", obj).hash(state);
            },
            Value::Expr { head, args } => {
                13u8.hash(state);
                head.hash(state);
                args.hash(state);
            },
            Value::Quote(expr) => {
                14u8.hash(state);
                // Use pointer address for AST expressions to avoid expensive formatting
                (expr.as_ref() as *const _ as usize).hash(state);
            },
            Value::Pattern(pattern) => {
                15u8.hash(state);
                // Pattern has proper Hash implementation, use it directly
                pattern.hash(state);
            },
            Value::Rule { lhs, rhs } => {
                16u8.hash(state);
                lhs.hash(state);
                rhs.hash(state);
            },
            Value::PureFunction { body } => {
                17u8.hash(state);
                body.hash(state);
            },
            Value::Slot { number } => {
                18u8.hash(state);
                number.hash(state);
            },
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Real(a), Value::Real(b)) => a == b,
            (Value::Rational(an, ad), Value::Rational(bn, bd)) => an == bn && ad == bd,
            (Value::BigReal { value: av, precision: ap }, Value::BigReal { value: bv, precision: bp }) => av == bv && ap == bp,
            (Value::Complex { re: ar, im: ai }, Value::Complex { re: br, im: bi }) => ar == br && ai == bi,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Missing, Value::Missing) => true,
            (Value::Object(a), Value::Object(b)) => a == b,
            (Value::LyObj(a), Value::LyObj(b)) => a == b,
            (Value::Expr { head: ah, args: aa }, Value::Expr { head: bh, args: ba }) => ah == bh && aa == ba,
            (Value::Quote(a), Value::Quote(b)) => {
                // Compare AST expressions structurally
                format!("{:?}", a) == format!("{:?}", b)
            },
            (Value::Pattern(a), Value::Pattern(b)) => {
                // Compare Pattern expressions structurally
                a == b
            },
            _ => false,
        }
    }
}

impl Value {
    /// Check if this value is cheap to clone (no heap allocations)
    #[inline(always)]
    pub fn is_cheap_clone(&self) -> bool {
        matches!(self, 
            Value::Integer(_) | 
            Value::Real(_) | 
            Value::Boolean(_) | 
            Value::Missing |
            Value::Slot { .. }
        )
    }
    
    /// Get an estimate of memory usage for this value
    pub fn memory_size(&self) -> usize {
        match self {
            Value::Integer(_) => 8,
            Value::Real(_) => 8,
            Value::Rational(_, _) => 16,
            Value::BigReal { .. } => 16,
            Value::Complex { re, im } => re.memory_size() + im.memory_size(),
            Value::Boolean(_) => 1,
            Value::Missing => 0,
            Value::String(s) => s.len() + std::mem::size_of::<String>(),
            Value::Symbol(s) => s.len() + std::mem::size_of::<String>(),
            Value::List(items) => {
                std::mem::size_of::<Vec<Value>>() + 
                items.iter().map(|v| v.memory_size()).sum::<usize>()
            }
            Value::Function(name) => name.len() + std::mem::size_of::<String>(),
            Value::Object(obj) => {
                std::mem::size_of::<HashMap<String, Value>>() +
                obj.iter().map(|(k, v)| k.len() + v.memory_size()).sum::<usize>()
            }
            Value::LyObj(_) => std::mem::size_of::<crate::foreign::LyObj>(),
            Value::Expr { head, args } => {
                head.memory_size() + std::mem::size_of::<Vec<Value>>() + args.iter().map(|v| v.memory_size()).sum::<usize>()
            }
            Value::Quote(_) => std::mem::size_of::<Box<crate::ast::Expr>>(),
            Value::Pattern(_) => std::mem::size_of::<crate::ast::Pattern>(),
            Value::Rule { lhs, rhs } => lhs.memory_size() + rhs.memory_size(),
            Value::PureFunction { body } => body.memory_size(),
            Value::Slot { .. } => std::mem::size_of::<Option<usize>>(),
        }
    }
}

impl Value {
    // Convenience constructors for new numeric forms
    pub fn complex(re: Value, im: Value) -> Value { Value::Complex { re: Box::new(re), im: Box::new(im) } }
    pub fn rational(n: i64, d: i64) -> Value { Value::Rational(n, d) }
    pub fn bigreal(value: f64, precision: u32) -> Value { Value::BigReal { value, precision } }
    pub fn expr(head: Value, args: Vec<Value>) -> Value { Value::Expr { head: Box::new(head), args } }
}

/// Call frame for function calls
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub return_address: usize,
    pub function_name: String,
    pub local_count: usize,
}

/// Virtual Machine state - focused on symbolic computation
#[derive(Debug)]
pub struct VirtualMachine {
    /// Instruction pointer
    pub ip: usize,
    /// Value stack
    pub stack: Vec<Value>,
    /// Call stack for function calls
    pub call_stack: Vec<CallFrame>,
    /// Constant pool
    pub constants: Vec<Value>,
    /// Symbol table (name -> index)
    pub symbols: HashMap<String, usize>,
    /// Global symbol values for immediate assignments (=)
    pub global_symbols: HashMap<String, Value>,
    /// Delayed symbol definitions for delayed assignments (:=)
    pub delayed_definitions: HashMap<String, crate::ast::Expr>,
    /// Type metadata for user-defined functions
    pub type_metadata: HashMap<String, crate::compiler::SimpleFunctionSignature>,
    /// Enhanced type metadata for user-defined functions
    pub enhanced_metadata: HashMap<String, crate::compiler::EnhancedFunctionSignature>,
    /// User-defined function bodies
    pub user_functions: HashMap<String, crate::ast::Expr>,
    /// Bytecode instructions
    pub code: Vec<Instruction>,
    /// Maximum call stack depth
    pub max_call_depth: usize,
    /// Standard library functions
    pub stdlib: StandardLibrary,
    /// Function registry for static dispatch
    pub registry: FunctionRegistry,
}

impl VirtualMachine {
    /// Create a new virtual machine
    pub fn new() -> Self {
        VirtualMachine {
            ip: 0,
            stack: Vec::new(),
            call_stack: Vec::new(),
            constants: Vec::new(),
            symbols: HashMap::new(),
            global_symbols: HashMap::new(),
            delayed_definitions: HashMap::new(),
            type_metadata: HashMap::new(),
            enhanced_metadata: HashMap::new(),
            user_functions: HashMap::new(),
            code: Vec::new(),
            max_call_depth: 1000,
            stdlib: StandardLibrary::new(),
            registry: create_global_registry().expect("Failed to create function registry"),
        }
    }

    /// Load bytecode and constants into the VM
    pub fn load(&mut self, code: Vec<Instruction>, constants: Vec<Value>) {
        self.code = code;
        self.constants = constants;
        self.ip = 0;
        self.stack.clear();
        self.call_stack.clear();
    }

    /// Add a constant to the constant pool, returns its index
    pub fn add_constant(&mut self, value: Value) -> usize {
        self.constants.push(value);
        self.constants.len() - 1
    }

    /// Add a symbol to the symbol table, returns its index
    pub fn add_symbol(&mut self, name: String) -> usize {
        if let Some(&index) = self.symbols.get(&name) {
            return index;
        }
        let index = self.symbols.len();
        self.symbols.insert(name, index);
        index
    }

    /// Push a value onto the stack
    pub fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> VmResult<Value> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    /// Peek at the top value on the stack without removing it
    pub fn peek(&self) -> VmResult<&Value> {
        self.stack.last().ok_or(VmError::StackUnderflow)
    }

    /// Get current instruction
    pub fn current_instruction(&self) -> VmResult<&Instruction> {
        self.code
            .get(self.ip)
            .ok_or(VmError::InvalidInstructionPointer(self.ip))
    }

    /// Execute the current program
    pub fn run(&mut self) -> VmResult<Value> {
        loop {
            if self.ip >= self.code.len() {
                break;
            }

            self.step()?;
        }

        // Return the top value on the stack, or a default if empty
        if self.stack.is_empty() {
            Ok(Value::Integer(0)) // Default return value
        } else {
            Ok(self.stack.last().unwrap().clone())
        }
    }

    /// Legacy compatibility wrapper for run
    pub fn execute(&mut self) -> VmResult<Value> {
        self.run()
    }

    /// Execute a single instruction
    pub fn step(&mut self) -> VmResult<()> {
        let instruction = *self.current_instruction()?;

        match instruction.opcode {
            OpCode::LDC => {
                let operand = instruction.operand;
                
                // Heuristic: if operand >= constants.len(), treat as immediate integer
                // This handles the case where small integers are embedded directly
                if operand as usize >= self.constants.len() {
                    // Treat as immediate integer value
                    self.push(Value::Integer(operand as i64));
                } else {
                    // Treat as constant pool index
                    let const_index = operand as usize;
                    let value = self.constants[const_index].clone();
                    
                    // Check if this is a symbol and if it has a resolved value
                    if let Value::Symbol(symbol_name) = &value {
                        // Check immediate assignments first
                        if let Some(resolved_value) = self.global_symbols.get(symbol_name) {
                            self.push(resolved_value.clone());
                        }
                        // Check delayed assignments
                        else if let Some(delayed_expr) = self.delayed_definitions.get(symbol_name) {
                            // For delayed assignments, we need to compile and evaluate the expression
                            // For now, store the expression and mark for evaluation
                            // TODO: Implement delayed expression evaluation
                            self.push(Value::Quote(Box::new(delayed_expr.clone())));
                        }
                        // No assignment found, push the symbol as-is
                        else {
                            self.push(value);
                        }
                    } else {
                        // Not a symbol, push the value as-is
                        self.push(value);
                    }
                }
                self.ip += 1;
            }
            OpCode::LoadQuote => {
                // LoadQuote expects the AST expression to be stored in constants pool
                let const_index = instruction.operand as usize;
                if const_index >= self.constants.len() {
                    return Err(VmError::InvalidConstantIndex(const_index));
                }
                
                // The constant should contain a Quote value with the AST expression
                let quote_value = self.constants[const_index].clone();
                self.push(quote_value);
                self.ip += 1;
            }
            OpCode::POP => {
                self.pop()?;
                self.ip += 1;
            }
            OpCode::DUP => {
                let value = self.peek()?.clone();
                self.push(value);
                self.ip += 1;
            }
            OpCode::ADD => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.add_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::SUB => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.sub_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::MUL => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.mul_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::DIV => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.div_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::POW => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.power_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::JMP => {
                self.ip = instruction.operand as usize;
            }
            OpCode::JIF => {
                let condition = self.pop()?;
                if self.is_falsy(&condition) {
                    self.ip = instruction.operand as usize;
                } else {
                    self.ip += 1;
                }
            }
            OpCode::RET => {
                if let Some(frame) = self.call_stack.pop() {
                    self.ip = frame.return_address;
                } else {
                    return Ok(()); // End of program
                }
            }

            OpCode::CallUser => {
                // Call user-defined function with type validation
                let (function_name_index, argc) = {
                    let function_name_index = (instruction.operand >> 8) as u16;
                    let argc = (instruction.operand & 0xFF) as u8;
                    (function_name_index, argc)
                };
                
                // Pop arguments from stack
                let mut args = Vec::with_capacity(argc as usize);
                for _ in 0..argc {
                    args.push(self.pop()?);
                }
                args.reverse(); // Stack pops in reverse order
                
                // Get function name from constants
                let function_name = match &self.constants[function_name_index as usize] {
                    Value::Symbol(name) => name.clone(),
                    _ => {
                        return Err(VmError::TypeError {
                            expected: "function name symbol".to_string(),
                            actual: format!("constant at index {}", function_name_index),
                        });
                    }
                };
                
                // Perform runtime type validation
                self.validate_user_function_call(&function_name, &args)?;
                
                // Execute user-defined function
                let result = self.execute_user_function(&function_name, &args)?;
                self.push(result);
                
                self.ip += 1;
            }
            
            OpCode::LDL => {
                // Load local variable (placeholder implementation)
                let _local_index = instruction.operand as usize;
                // For now, return Missing - full implementation needs local variable stack
                self.push(Value::Missing);
                self.ip += 1;
            }
            OpCode::STL => {
                // Store local variable (placeholder implementation)
                let _local_index = instruction.operand as usize;
                let _value = self.pop()?;
                // For now, just discard - full implementation needs local variable stack
                self.ip += 1;
            }
            OpCode::STS => {
                // Store symbol value (immediate assignment)
                let symbol_index = instruction.operand as usize;
                let value = self.pop()?;
                
                // Get symbol name from constants pool (should be a Symbol value)
                if let Some(Value::Symbol(symbol_name)) = self.constants.get(symbol_index) {
                    self.global_symbols.insert(symbol_name.clone(), value);
                } else {
                    return Err(VmError::Runtime(format!(
                        "Invalid symbol index {} for STS instruction", 
                        symbol_index
                    )));
                }
                self.ip += 1;
            }
            OpCode::STSD => {
                // Store symbol delayed (delayed assignment)
                let symbol_index = instruction.operand as usize;
                let expr_value = self.pop()?;
                
                // Get symbol name from constants pool
                if let Some(Value::Symbol(symbol_name)) = self.constants.get(symbol_index) {
                    // The expression should be stored as a Quote value (unevaluated Expr)
                    if let Value::Quote(expr) = expr_value {
                        self.delayed_definitions.insert(symbol_name.clone(), *expr);
                    } else {
                        return Err(VmError::Runtime(format!(
                            "Expected Quote value for delayed assignment, got {:?}",
                            expr_value
                        )));
                    }
                } else {
                    return Err(VmError::Runtime(format!(
                        "Invalid symbol index {} for STSD instruction", 
                        symbol_index
                    )));
                }
                self.ip += 1;
            }
            OpCode::NEWLIST => {
                // Create new list from n stack items
                let count = instruction.operand as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.pop()?);
                }
                items.reverse(); // Items were popped in reverse order
                self.push(Value::List(items));
                self.ip += 1;
            }
            OpCode::NEWASSOC => {
                // Create new associative array from 2n stack items (key-value pairs)
                let pair_count = instruction.operand as usize;
                let mut pairs = Vec::with_capacity(pair_count);
                for _ in 0..pair_count {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    pairs.push((key, value));
                }
                pairs.reverse(); // Pairs were popped in reverse order
                // For now, convert to a simple list representation
                // Full implementation would use a proper associative data structure
                let assoc_list: Vec<Value> = pairs.into_iter()
                    .flat_map(|(k, v)| vec![k, v])
                    .collect();
                self.push(Value::List(assoc_list));
                self.ip += 1;
            }
            OpCode::SYS => {
                // System call (placeholder implementation)
                let (_sys_op, argc) = {
                    let sys_op = (instruction.operand >> 8) as u16;
                    let argc = (instruction.operand & 0xFF) as u8;
                    (sys_op, argc)
                };
                
                // Pop arguments
                let mut _args = Vec::with_capacity(argc as usize);
                for _ in 0..argc {
                    _args.push(self.pop()?);
                }
                
                // For now, just return Missing - full implementation needs system call registry
                self.push(Value::Missing);
                self.ip += 1;
            }
            
            OpCode::CallStatic => {
                // Decode function index and argument count
                let (function_index, argc) = {
                    let function_index = (instruction.operand >> 8) as u16;
                    let argc = (instruction.operand & 0xFF) as u8;
                    (function_index, argc)
                };
                
                // Pop arguments from stack
                let mut args = Vec::with_capacity(argc as usize);
                for _ in 0..argc {
                    args.push(self.pop()?);
                }
                args.reverse(); // Stack pops in reverse order
                
                // Dispatch based on function index range
                if function_index >= 1000 {
                    // Pure function application (indices 1000+): pop pure function and apply slot substitution
                    let function_value = self.pop()?; // Pop the pure function from stack
                    
                    match function_value {
                        Value::PureFunction { .. } => {
                            // Use the slot substitution algorithm from pure_function module
                            match crate::pure_function::substitute_slots(&function_value, &args) {
                                Ok(result) => {
                                    self.push(result);
                                }
                                Err(substitution_error) => {
                                    return Err(VmError::Runtime(format!(
                                        "Pure function application failed: {}",
                                        substitution_error
                                    )));
                                }
                            }
                        }
                        _ => {
                            return Err(VmError::TypeError {
                                expected: "PureFunction value for pure function call".to_string(),
                                actual: format!("{:?}", function_value),
                            });
                        }
                    }
                } else if function_index < 32 {
                    // Foreign methods (indices 0-31): need object + method call
                    let obj = self.pop()?; // Pop the object
                    
                    if let Value::LyObj(lyobj) = obj {
                        // Map function_index to method name (temporary approach)
                        let method_name = match function_index {
                            0 => "Length",
                            1 => "Type", 
                            2 => "ToList",
                            3 => "IsEmpty",
                            4 => "Get",
                            5 => "Append",
                            6 => "Set",
                            7 => "Slice",
                            8 => "Shape",
                            9 => "Columns",
                            10 => "Rows",
                            _ => {
                                return Err(VmError::TypeError {
                                    expected: "valid method index".to_string(),
                                    actual: format!("index {}", function_index),
                                });
                            }
                        };
                        
                        // Get the object type for registry lookup
                        let type_name = lyobj.type_name();
                        
                        // Use the Function Registry to look up and call the method
                        match self.registry.lookup(type_name, method_name) {
                            Ok(function_entry) => {
                                match function_entry.call(Some(&lyobj), &args) {
                                    Ok(result) => {
                                        self.push(result);
                                    }
                                    Err(foreign_error) => {
                                        return Err(VmError::TypeError {
                                            expected: "successful method call".to_string(),
                                            actual: format!("Foreign error: {:?}", foreign_error),
                                        });
                                    }
                                }
                            }
                            Err(linker_error) => {
                                return Err(VmError::TypeError {
                                    expected: format!("method {}::{}", type_name, method_name),
                                    actual: format!("Linker error: {:?}", linker_error),
                                });
                            }
                        }
                    } else {
                        return Err(VmError::TypeError {
                            expected: "LyObj for method call".to_string(),
                            actual: format!("{:?}", obj),
                        });
                    }
                } else {
                    // Stdlib functions (indices 32+): direct function call
                    match self.registry.get_stdlib_function(function_index) {
                        Some(function) => {
                            match function(&args) {
                                Ok(result) => {
                                    self.push(result);
                                }
                                Err(vm_error) => {
                                    return Err(vm_error);
                                }
                            }
                        }
                        None => {
                            return Err(VmError::TypeError {
                                expected: "valid stdlib function".to_string(),
                                actual: format!("index {}", function_index),
                            });
                        }
                    }
                }
                
                self.ip += 1;
            }
            
            OpCode::MapCallStatic => {
                // MAP_CallStatic: Apply function to lists for Listable attributes
                let (function_index, argc) = {
                    let function_index = (instruction.operand >> 8) as u16;
                    let argc = (instruction.operand & 0xFF) as u8;
                    (function_index, argc)
                };
                
                // Pop arguments from stack
                let mut args = Vec::with_capacity(argc as usize);
                for _ in 0..argc {
                    args.push(self.pop()?);
                }
                args.reverse(); // Stack pops in reverse order
                
                // Determine if we need to handle object method call (Foreign) vs stdlib function
                let needs_object = function_index < 32;
                let mut object = None;
                
                if needs_object {
                    // For Foreign methods, pop the object too
                    object = Some(self.pop()?);
                }
                
                // Check which arguments are lists and determine the result list length
                let mut list_indices = Vec::new();
                let mut max_list_length = 1; // Minimum length for scalar broadcasting
                
                for (i, arg) in args.iter().enumerate() {
                    if let Value::List(list_items) = arg {
                        list_indices.push(i);
                        max_list_length = max_list_length.max(list_items.len());
                    }
                }
                
                // If no lists found, this is an error - MAP_CallStatic should only be used with Listable functions that have list arguments
                if list_indices.is_empty() {
                    return Err(VmError::TypeError {
                        expected: "at least one list argument for Listable function".to_string(),
                        actual: "no list arguments found".to_string(),
                    });
                }
                
                // Build result list by applying function element-wise
                let mut result_list = Vec::with_capacity(max_list_length);
                
                for element_index in 0..max_list_length {
                    // Build arguments for this element position
                    let mut element_args = Vec::with_capacity(args.len());
                    
                    for (_arg_index, arg) in args.iter().enumerate() {
                        match arg {
                            Value::List(list_items) => {
                                // Get element at this position, or last element if list is shorter (broadcasting)
                                let item_index = element_index.min(list_items.len().saturating_sub(1));
                                element_args.push(list_items[item_index].clone());
                            }
                            _ => {
                                // Scalar argument - broadcast to all positions
                                element_args.push(arg.clone());
                            }
                        }
                    }
                    
                    // Call the function for this element position
                    let element_result = if needs_object {
                        // Foreign method call
                        if let Some(Value::LyObj(ref lyobj)) = object {
                            // Map function_index to method name (same mapping as CallStatic)
                            let method_name = match function_index {
                                0 => "Length", 1 => "Type", 2 => "ToList", 3 => "IsEmpty",
                                4 => "Get", 5 => "Append", 6 => "Set", 7 => "Slice",
                                8 => "Shape", 9 => "Columns", 10 => "Rows",
                                _ => {
                                    return Err(VmError::TypeError {
                                        expected: "valid method index".to_string(),
                                        actual: format!("index {}", function_index),
                                    });
                                }
                            };
                            
                            let type_name = lyobj.type_name();
                            match self.registry.lookup(type_name, method_name) {
                                Ok(function_entry) => {
                                    match function_entry.call(Some(lyobj), &element_args) {
                                        Ok(result) => result,
                                        Err(foreign_error) => {
                                            return Err(VmError::TypeError {
                                                expected: "successful method call".to_string(),
                                                actual: format!("Foreign error: {:?}", foreign_error),
                                            });
                                        }
                                    }
                                }
                                Err(linker_error) => {
                                    return Err(VmError::TypeError {
                                        expected: format!("method {}::{}", type_name, method_name),
                                        actual: format!("Linker error: {:?}", linker_error),
                                    });
                                }
                            }
                        } else {
                            return Err(VmError::TypeError {
                                expected: "LyObj for method call".to_string(),
                                actual: format!("{:?}", object),
                            });
                        }
                    } else {
                        // Stdlib function call
                        match self.registry.get_stdlib_function(function_index) {
                            Some(function) => {
                                match function(&element_args) {
                                    Ok(result) => result,
                                    Err(vm_error) => return Err(vm_error),
                                }
                            }
                            None => {
                                return Err(VmError::TypeError {
                                    expected: "valid stdlib function".to_string(),
                                    actual: format!("index {}", function_index),
                                });
                            }
                        }
                    };
                    
                    result_list.push(element_result);
                }
                
                // Push the result list onto the stack
                self.push(Value::List(result_list));
                self.ip += 1;
            }
        }

        Ok(())
    }

    /// Add two values
    fn add_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Scalar arithmetic
        match (&a, &b) {
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 + b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a + b as f64)),
                _ => unreachable!(),
            },
            _ => {
                // If we can't evaluate numerically, create a symbolic Plus expression
                Ok(self.create_symbolic_function("Plus", &[a, b]))
            }
        }
    }

    /// Subtract two values
    fn sub_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Scalar arithmetic
        match (&a, &b) {
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 - b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a - b as f64)),
                _ => unreachable!(),
            },
            _ => {
                // If we can't evaluate numerically, create a symbolic Minus expression
                Ok(self.create_symbolic_function("Minus", &[a, b]))
            }
        }
    }

    /// Multiply two values
    fn mul_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Scalar arithmetic
        match (&a, &b) {
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 * b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a * b as f64)),
                _ => unreachable!(),
            },
            _ => {
                // If we can't evaluate numerically, create a symbolic Times expression
                Ok(self.create_symbolic_function("Times", &[a, b]))
            }
        }
    }

    /// Divide two values
    fn div_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Scalar arithmetic
        match (&a, &b) {
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => {
                    if b == 0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        // Integer division returns a real number for precision
                        Ok(Value::Real(a as f64 / b as f64))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => {
                    if b == 0.0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a / b))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => {
                    if b == 0.0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a as f64 / b))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => {
                    if b == 0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a / b as f64))
                    }
                }
                _ => unreachable!(),
            },
            _ => {
                // If we can't evaluate numerically, create a symbolic Divide expression
                Ok(self.create_symbolic_function("Divide", &[a, b]))
            }
        }
    }

    /// Calculate power of two values
    fn power_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Scalar arithmetic
        match (&a, &b) {
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => {
                    if b >= 0 {
                        // Use integer power for non-negative exponents
                        Ok(Value::Integer(a.pow(b as u32)))
                    } else {
                        // Negative exponents result in real numbers
                        Ok(Value::Real((a as f64).powf(b as f64)))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a.powf(b))),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real((a as f64).powf(b))),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a.powf(b as f64))),
                _ => unreachable!(),
            },
            _ => {
                // If we can't evaluate numerically, create a symbolic Power expression
                Ok(self.create_symbolic_function("Power", &[a, b]))
            }
        }
    }

    /// Convert a Value to an Expr for symbolic computation
    fn value_to_expr(&self, value: &Value) -> Expr {
        match value {
            Value::Integer(i) => Expr::Number(Number::Integer(*i)),
            Value::Real(r) => Expr::Number(Number::Real(*r)),
            Value::String(s) => Expr::String(s.clone()),
            Value::Symbol(s) => Expr::Symbol(Symbol { name: s.clone() }),
            Value::List(items) => {
                let expr_items: Vec<Expr> = items.iter().map(|item| self.value_to_expr(item)).collect();
                Expr::List(expr_items)
            }
            Value::Quote(expr) => *expr.clone(),
            _ => {
                // For complex types, create a symbolic representation
                Expr::Symbol(Symbol { name: format!("{:?}", value) })
            }
        }
    }

    /// Create a symbolic function call when arithmetic can't be evaluated
    fn create_symbolic_function(&self, function_name: &str, args: &[Value]) -> Value {
        let expr_args: Vec<Expr> = args.iter().map(|arg| self.value_to_expr(arg)).collect();
        let function_expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: function_name.to_string() })),
            args: expr_args,
        };
        Value::Quote(Box::new(function_expr))
    }

    /// Check if a value is considered "falsy" for conditionals
    fn is_falsy(&self, value: &Value) -> bool {
        match value {
            Value::Boolean(false) => true,
            Value::Integer(0) => true,
            Value::Real(f) if *f == 0.0 => true,
            Value::Missing => true,
            Value::List(items) if items.is_empty() => true,
            _ => false,
        }
    }

    /// Load type metadata and function definitions from compiler context
    pub fn load_type_metadata(&mut self, 
        type_metadata: HashMap<String, crate::compiler::SimpleFunctionSignature>,
        enhanced_metadata: HashMap<String, crate::compiler::EnhancedFunctionSignature>,
        user_functions: HashMap<String, crate::ast::Expr>
    ) {
        self.type_metadata = type_metadata;
        self.enhanced_metadata = enhanced_metadata;
        self.user_functions = user_functions;
    }

    /// Validate user function call parameters and arity
    fn validate_user_function_call(&self, function_name: &str, args: &[Value]) -> VmResult<()> {
        // Check if we have metadata for this function
        if let Some(enhanced_sig) = self.enhanced_metadata.get(function_name) {
            // Validate arity
            if enhanced_sig.params.len() != args.len() {
                return Err(VmError::TypeError {
                    expected: format!("function {} with {} parameters", function_name, enhanced_sig.params.len()),
                    actual: format!("call with {} arguments", args.len()),
                });
            }

            // Validate each typed parameter
            for (i, (param_name, expected_type_opt)) in enhanced_sig.params.iter().enumerate() {
                if let Some(expected_type) = expected_type_opt {
                    let actual_arg = &args[i];
                    
                    // Check if type matches or can be coerced
                    if !self.is_type_compatible(actual_arg, expected_type)? {
                        return Err(VmError::TypeError {
                            expected: format!("parameter {} of type {}", param_name, expected_type),
                            actual: format!("found {:?}", self.get_value_type_name(actual_arg)),
                        });
                    }
                }
                // Untyped parameters (None) are not validated
            }
        }
        // If no metadata found, function is completely untyped - no validation
        
        Ok(())
    }

    /// Execute user-defined function with parameter binding
    fn execute_user_function(&self, function_name: &str, args: &[Value]) -> VmResult<Value> {
        // Get the function AST from user_functions
        let function_ast = self.user_functions.get(function_name)
            .ok_or_else(|| VmError::Runtime(format!("User function '{}' not found", function_name)))?;
        
        // Function AST is directly the function body (RHS), we need to get parameter names from metadata
        let param_names: Vec<String> = if let Some(metadata) = self.enhanced_metadata.get(function_name) {
            metadata.params.iter().map(|(name, _)| name.clone()).collect()
        } else {
            return Err(VmError::Runtime(format!("No metadata found for function '{}'", function_name)));
        };
        
        let function_body = function_ast;
        
        // Check parameter count
        if param_names.len() != args.len() {
            return Err(VmError::Runtime(format!(
                "Arity mismatch: function '{}' expects {} parameters, got {}", 
                function_name, param_names.len(), args.len()
            )));
        }
        
        // Create parameter bindings (for now, we'll use simple substitution)
        // In a full implementation, we'd need a proper environment/context stack
        let result = self.evaluate_expression_with_bindings(function_body, &param_names, args)?;
        
        // Validate return type if function has return type annotation
        if let Some(metadata) = self.enhanced_metadata.get(function_name) {
            if let Some(return_type) = &metadata.return_type {
                // Check if result matches expected return type
                if !self.is_type_compatible(&result, return_type)? {
                    let actual_type = self.get_value_type_name(&result);
                    return Err(VmError::Runtime(format!(
                        "Return type mismatch: function '{}' declared to return {}, but returned {}",
                        function_name, return_type, actual_type
                    )));
                }
            }
        }
        
        Ok(result)
    }
    
    /// Evaluate an expression with parameter bindings
    fn evaluate_expression_with_bindings(&self, expr: &crate::ast::Expr, param_names: &[String], param_values: &[Value]) -> VmResult<Value> {
        match expr {
            // Basic literals
            crate::ast::Expr::Number(number) => {
                match number {
                    crate::ast::Number::Integer(i) => Ok(Value::Integer(*i)),
                    crate::ast::Number::Real(r) => Ok(Value::Real(*r)),
                }
            }
            crate::ast::Expr::String(s) => Ok(Value::String(s.clone())),
            
            // Symbol lookup with parameter binding
            crate::ast::Expr::Symbol(symbol) => {
                // Check if this symbol is a bound parameter
                if let Some(index) = param_names.iter().position(|name| name == &symbol.name) {
                    Ok(param_values[index].clone())
                } else {
                    // For now, treat as unbound symbol - return the symbol itself
                    Ok(Value::Symbol(symbol.name.clone()))
                }
            }
            
            // Function calls
            crate::ast::Expr::Function { head, args } => {
                match head.as_ref() {
                    crate::ast::Expr::Symbol(func_symbol) => {
                        // Evaluate arguments
                        let evaluated_args: Result<Vec<Value>, VmError> = args.iter()
                            .map(|arg| self.evaluate_expression_with_bindings(arg, param_names, param_values))
                            .collect();
                        let evaluated_args = evaluated_args?;
                        
                        // Handle built-in functions
                        match func_symbol.name.as_str() {
                            "Plus" | "+" => {
                                if evaluated_args.len() == 2 {
                                    self.add_values(evaluated_args[0].clone(), evaluated_args[1].clone())
                                } else {
                                    Err(VmError::Runtime("Plus requires exactly 2 arguments".to_string()))
                                }
                            }
                            "Times" | "*" => {
                                if evaluated_args.len() == 2 {
                                    self.mul_values(evaluated_args[0].clone(), evaluated_args[1].clone())
                                } else {
                                    Err(VmError::Runtime("Times requires exactly 2 arguments".to_string()))
                                }
                            }
                            "Minus" | "-" => {
                                if evaluated_args.len() == 2 {
                                    self.sub_values(evaluated_args[0].clone(), evaluated_args[1].clone())
                                } else {
                                    Err(VmError::Runtime("Minus requires exactly 2 arguments".to_string()))
                                }
                            }
                            "Length" => {
                                if evaluated_args.len() == 1 {
                                    match &evaluated_args[0] {
                                        Value::List(list) => Ok(Value::Real(list.len() as f64)),
                                        _ => Err(VmError::Runtime("Length requires a list argument".to_string()))
                                    }
                                } else {
                                    Err(VmError::Runtime("Length requires exactly 1 argument".to_string()))
                                }
                            }
                            _ => {
                                // For now, return a placeholder for other functions
                                Err(VmError::Runtime(format!("Function '{}' not implemented in interpreter", func_symbol.name)))
                            }
                        }
                    }
                    _ => Err(VmError::Runtime("Invalid function head".to_string()))
                }
            }
            
            // Lists
            crate::ast::Expr::List(elements) => {
                let evaluated_elements: Result<Vec<Value>, VmError> = elements.iter()
                    .map(|elem| self.evaluate_expression_with_bindings(elem, param_names, param_values))
                    .collect();
                Ok(Value::List(evaluated_elements?))
            }
            
            _ => {
                // For other expressions, return a placeholder
                Err(VmError::Runtime(format!("Expression type not implemented in interpreter: {:?}", expr)))
            }
        }
    }

    /// Check if a value is compatible with an expected type (including coercion)
    fn is_type_compatible(&self, value: &Value, expected_type: &str) -> VmResult<bool> {
        // Handle complex type expressions
        if expected_type.contains('[') && expected_type.contains(']') {
            return self.validate_complex_type(value, expected_type);
        }
        
        let actual_type = self.get_value_type_name(value);
        
        // Exact type match
        if actual_type == expected_type {
            return Ok(true);
        }
        
        // Type coercion rules
        match (actual_type.as_str(), expected_type) {
            // Integer can be coerced to Real
            ("Integer", "Real") => Ok(true),
            // Add more coercion rules as needed
            _ => Ok(false),
        }
    }
    
    /// Validate complex type expressions like List[T], Map[K,V], etc.
    fn validate_complex_type(&self, value: &Value, expected_type: &str) -> VmResult<bool> {
        // Parse the complex type expression
        if let Some((base_type, type_params)) = self.parse_complex_type(expected_type) {
            match (base_type.as_str(), value) {
                ("List", Value::List(list)) => {
                    // Validate each element in the list matches the parameter type
                    if type_params.len() != 1 {
                        return Err(VmError::Runtime(format!("List type requires exactly 1 type parameter, got {}", type_params.len())));
                    }
                    let element_type = &type_params[0];
                    
                    // Check each element in the list
                    for (index, element) in list.iter().enumerate() {
                        if !self.is_type_compatible(element, element_type)? {
                            let actual_element_type = self.get_value_type_name(element);
                            return Err(VmError::Runtime(format!(
                                "Type error: List[{}] element at index {} expected {}, but found {}",
                                element_type, index, element_type, actual_element_type
                            )));
                        }
                    }
                    Ok(true)
                }
                // Map types would be handled via LyObj for now
                // ("Map", Value::LyObj(_)) => { ... }
                _ => {
                    // Base type mismatch
                    Ok(false)
                }
            }
        } else {
            // Failed to parse complex type
            Err(VmError::Runtime(format!("Invalid complex type expression: {}", expected_type)))
        }
    }
    
    /// Parse complex type expressions like "List[Real]" into ("List", ["Real"])
    fn parse_complex_type(&self, type_expr: &str) -> Option<(String, Vec<String>)> {
        if let Some(open_bracket) = type_expr.find('[') {
            if let Some(close_bracket) = type_expr.rfind(']') {
                let base_type = type_expr[..open_bracket].to_string();
                let params_str = &type_expr[open_bracket + 1..close_bracket];
                
                // Simple parsing - split by comma and trim
                let type_params: Vec<String> = params_str
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                
                return Some((base_type, type_params));
            }
        }
        None
    }

    /// Get the type name of a value for type checking
    fn get_value_type_name(&self, value: &Value) -> String {
        match value {
            Value::Integer(_) => "Integer".to_string(),
            Value::Real(_) => "Real".to_string(),
            Value::String(_) => "String".to_string(),
            Value::Symbol(_) => "Symbol".to_string(),
            Value::List(_) => "List".to_string(), // TODO: Support List[T] type checking
            Value::Function(_) => "Function".to_string(),
            Value::Boolean(_) => "Boolean".to_string(),
            Value::Missing => "Missing".to_string(),
            Value::Object(_) => "Object".to_string(),
            Value::LyObj(_) => "LyObj".to_string(),
            Value::Quote(_) => "Quote".to_string(),
            Value::Pattern(_) => "Pattern".to_string(),
            Value::Rule { .. } => "Rule".to_string(),
            Value::PureFunction { .. } => "PureFunction".to_string(),
            Value::Slot { .. } => "Slot".to_string(),
        }
    }
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl Value {
    /// Get the value as a real number if possible
    pub fn as_real(&self) -> Option<f64> {
        match self {
            Value::Real(r) => Some(*r),
            Value::Integer(i) => Some(*i as f64),
            _ => None,
        }
    }
    
    /// Get the value as a string if possible
    pub fn as_string(&self) -> Option<String> {
        match self {
            Value::String(s) => Some(s.clone()),
            Value::Symbol(s) => Some(s.clone()),
            _ => None,
        }
    }
    
    /// Get the value as an integer if possible
    pub fn as_integer(&self) -> Option<i64> {
        match self {
            Value::Integer(i) => Some(*i),
            Value::Real(r) if r.fract() == 0.0 => Some(*r as i64),
            _ => None,
        }
    }
    
    /// Get the value as a boolean if possible
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Boolean(b) => Some(*b),
            _ => None,
        }
    }
    
    /// Get the value as a list if possible
    pub fn as_list(&self) -> Option<&Vec<Value>> {
        match self {
            Value::List(list) => Some(list),
            _ => None,
        }
    }
    
    /// Get the value as an object if possible
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(obj) => Some(obj),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vm_creation() {
        let vm = VirtualMachine::new();
        assert_eq!(vm.stack.len(), 0);
        assert_eq!(vm.call_stack.len(), 0);
        assert_eq!(vm.ip, 0);
    }

    #[test]
    fn test_stack_operations() {
        let mut vm = VirtualMachine::new();
        
        // Test push and pop
        vm.push(Value::Integer(42));
        assert_eq!(vm.stack.len(), 1);
        
        let value = vm.pop().unwrap();
        assert_eq!(value, Value::Integer(42));
        assert_eq!(vm.stack.len(), 0);
        
        // Test underflow
        assert!(vm.pop().is_err());
    }

    #[test]
    fn test_add_values() {
        let vm = VirtualMachine::new();
        
        // Test integer addition
        let result = vm.add_values(Value::Integer(5), Value::Integer(3)).unwrap();
        assert_eq!(result, Value::Integer(8));
        
        // Test mixed addition
        let result = vm.add_values(Value::Integer(5), Value::Real(3.5)).unwrap();
        assert_eq!(result, Value::Real(8.5));
        
        // Test missing propagation
        let result = vm.add_values(Value::Integer(5), Value::Missing).unwrap();
        assert_eq!(result, Value::Missing);
    }
}
