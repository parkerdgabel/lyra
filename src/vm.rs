use crate::bytecode::{Instruction, OpCode};
use crate::foreign::LyObj;
use crate::linker::{registry::create_global_registry, FunctionRegistry};
use crate::stdlib::StandardLibrary;
use crate::stdlib::tensor::{tensor_add, tensor_sub, tensor_mul, tensor_div, tensor_pow};
use std::collections::HashMap;
use thiserror::Error;
use ndarray::ArrayD;

#[derive(Error, Debug)]
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
    #[error("Call stack overflow")]
    CallStackOverflow,
    #[error("Cannot call non-function value")]
    NotCallable,
    #[error("Index {index} out of bounds for length {length}")]
    IndexError { index: i64, length: usize },
}

pub type VmResult<T> = std::result::Result<T, VmError>;

/// A value that can be stored on the VM stack
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String), // Built-in function name for now
    Boolean(bool),
    Missing,            // Missing/unknown value (distinct from Null)
    LyObj(LyObj),       // Foreign object wrapper for complex types (Series, Table, Dataset, Schema, Tensor)
    Quote(Box<crate::ast::Expr>), // Unevaluated expression for Hold attributes
    Pattern(crate::ast::Pattern), // Pattern expressions for pattern matching
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
            Value::String(s) => {
                2u8.hash(state);
                s.hash(state);
            },
            Value::Symbol(s) => {
                3u8.hash(state);
                s.hash(state);
            },
            Value::List(items) => {
                4u8.hash(state);
                items.hash(state);
            },
            Value::Function(name) => {
                5u8.hash(state);
                name.hash(state);
            },
            Value::Boolean(b) => {
                6u8.hash(state);
                b.hash(state);
            },
            Value::Missing => {
                7u8.hash(state);
            },
            Value::LyObj(obj) => {
                8u8.hash(state);
                // Hash type name and debug representation for now
                // Foreign objects could implement custom hashing
                obj.type_name().hash(state);
                format!("{:?}", obj).hash(state);
            },
            Value::Quote(expr) => {
                10u8.hash(state);
                // Hash the debug representation of the AST expression
                format!("{:?}", expr).hash(state);
            },
            Value::Pattern(pattern) => {
                11u8.hash(state);
                // Hash the debug representation of the Pattern
                format!("{:?}", pattern).hash(state);
            },
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Real(a), Value::Real(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Missing, Value::Missing) => true,
            (Value::LyObj(a), Value::LyObj(b)) => a == b,
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
                    self.push(value);
                }
                self.ip += 1;
            }
            OpCode::LOAD_QUOTE => {
                // LOAD_QUOTE expects the AST expression to be stored in constants pool
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
                // Store symbol value (placeholder implementation)
                let _symbol_index = instruction.operand as usize;
                let _value = self.pop()?;
                // For now, just discard - full implementation needs symbol table
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
                let (sys_op, argc) = {
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
            
            OpCode::CALL_STATIC => {
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
                if function_index < 32 {
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
            
            OpCode::MAP_CALL_STATIC => {
                // MAP_CALL_STATIC: Apply function to lists for Listable attributes
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
                
                // If no lists found, this is an error - MAP_CALL_STATIC should only be used with Listable functions that have list arguments
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
                            // Map function_index to method name (same mapping as CALL_STATIC)
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
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
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
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
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
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
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
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
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
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
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
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::OpCode;

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