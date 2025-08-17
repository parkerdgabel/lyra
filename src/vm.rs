use crate::bytecode::{Instruction, OpCode};
use crate::stdlib::StandardLibrary;
use std::collections::HashMap;
use thiserror::Error;

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
}

pub type VmResult<T> = std::result::Result<T, VmError>;

/// A value that can be stored on the VM stack
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String), // Built-in function name for now
    Boolean(bool),
}

/// Call frame for function calls
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub return_address: usize,
    pub function_name: String,
    pub local_count: usize,
}

/// Virtual Machine state
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

            let instruction = *self.current_instruction()?;

            // Check for halt instruction before executing
            if matches!(instruction.opcode, OpCode::Halt) {
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
            OpCode::LoadConst => {
                let const_index = instruction.operand as usize;
                if const_index >= self.constants.len() {
                    return Err(VmError::InvalidConstantIndex(const_index));
                }
                let value = self.constants[const_index].clone();
                self.push(value);
                self.ip += 1;
            }
            OpCode::LoadSymbol => {
                let symbol_index = instruction.operand as usize;
                // For now, just push the symbol name as a Symbol value
                // In a real implementation, this would look up the symbol's value
                let symbol_name = format!("symbol_{}", symbol_index);
                self.push(Value::Symbol(symbol_name));
                self.ip += 1;
            }
            OpCode::Push => {
                // Push immediate value (for small integers)
                let value = Value::Integer(instruction.operand as i64);
                self.push(value);
                self.ip += 1;
            }
            OpCode::Pop => {
                self.pop()?;
                self.ip += 1;
            }
            OpCode::Dup => {
                let value = self.peek()?.clone();
                self.push(value);
                self.ip += 1;
            }
            OpCode::Add => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.add_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::Sub => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.sub_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::Mul => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.mul_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::Div => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.div_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::Power => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.power_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::Jump => {
                self.ip = instruction.operand as usize;
            }
            OpCode::JumpIfFalse => {
                let condition = self.pop()?;
                if self.is_falsy(&condition) {
                    self.ip = instruction.operand as usize;
                } else {
                    self.ip += 1;
                }
            }
            OpCode::Call => {
                let arg_count = instruction.operand as usize;

                // Pop function name from stack
                let function_name = match self.pop()? {
                    Value::Function(name) => name,
                    Value::Symbol(name) => name,
                    other => {
                        return Err(VmError::TypeError {
                            expected: "Function or Symbol".to_string(),
                            actual: format!("{:?}", other),
                        })
                    }
                };

                // Pop arguments from stack (in reverse order)
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(self.pop()?);
                }
                args.reverse(); // Arguments were pushed in reverse order

                // Try to call stdlib function
                if let Some(func) = self.stdlib.get_function(&function_name) {
                    let result = func(&args)?;
                    self.push(result);
                } else {
                    return Err(VmError::TypeError {
                        expected: format!("known function, got: {}", function_name),
                        actual: "unknown function".to_string(),
                    });
                }

                self.ip += 1;
            }
            OpCode::Return => {
                if let Some(frame) = self.call_stack.pop() {
                    self.ip = frame.return_address;
                } else {
                    return Ok(()); // End of program
                }
            }
            OpCode::Halt => {
                return Ok(());
            }
        }

        Ok(())
    }

    /// Add two values
    fn add_values(&self, a: Value, b: Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
            (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 + b)),
            (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a + b as f64)),
            (a, b) => Err(VmError::TypeError {
                expected: "numeric".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Subtract two values
    fn sub_values(&self, a: Value, b: Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
            (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 - b)),
            (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a - b as f64)),
            (a, b) => Err(VmError::TypeError {
                expected: "numeric".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Multiply two values
    fn mul_values(&self, a: Value, b: Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
            (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 * b)),
            (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a * b as f64)),
            (a, b) => Err(VmError::TypeError {
                expected: "numeric".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Divide two values
    fn div_values(&self, a: Value, b: Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => {
                if b == 0 {
                    Err(VmError::DivisionByZero)
                } else {
                    Ok(Value::Real(a as f64 / b as f64))
                }
            }
            (Value::Real(a), Value::Real(b)) => {
                if b == 0.0 {
                    Err(VmError::DivisionByZero)
                } else {
                    Ok(Value::Real(a / b))
                }
            }
            (Value::Integer(a), Value::Real(b)) => {
                if b == 0.0 {
                    Err(VmError::DivisionByZero)
                } else {
                    Ok(Value::Real(a as f64 / b))
                }
            }
            (Value::Real(a), Value::Integer(b)) => {
                if b == 0 {
                    Err(VmError::DivisionByZero)
                } else {
                    Ok(Value::Real(a / b as f64))
                }
            }
            (a, b) => Err(VmError::TypeError {
                expected: "numeric".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Raise a to the power of b
    fn power_values(&self, a: Value, b: Value) -> VmResult<Value> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => {
                if b >= 0 {
                    Ok(Value::Integer(a.pow(b as u32)))
                } else {
                    Ok(Value::Real((a as f64).powf(b as f64)))
                }
            }
            (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a.powf(b))),
            (Value::Integer(a), Value::Real(b)) => Ok(Value::Real((a as f64).powf(b))),
            (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a.powf(b as f64))),
            (a, b) => Err(VmError::TypeError {
                expected: "numeric".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Check if a value is falsy (for conditional jumps)
    fn is_falsy(&self, value: &Value) -> bool {
        match value {
            Value::Boolean(b) => !b,
            Value::Integer(0) => true,
            Value::Real(f) => *f == 0.0,
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
        assert_eq!(vm.ip, 0);
        assert!(vm.stack.is_empty());
        assert!(vm.call_stack.is_empty());
        assert!(vm.constants.is_empty());
        assert!(vm.symbols.is_empty());
        assert!(vm.code.is_empty());
    }

    #[test]
    fn test_stack_operations() {
        let mut vm = VirtualMachine::new();

        // Test push
        vm.push(Value::Integer(42));
        assert_eq!(vm.stack.len(), 1);

        // Test peek
        assert_eq!(vm.peek().unwrap(), &Value::Integer(42));
        assert_eq!(vm.stack.len(), 1); // Peek doesn't remove

        // Test pop
        let value = vm.pop().unwrap();
        assert_eq!(value, Value::Integer(42));
        assert!(vm.stack.is_empty());

        // Test pop on empty stack
        assert!(vm.pop().is_err());
    }

    #[test]
    fn test_constant_pool() {
        let mut vm = VirtualMachine::new();

        let index1 = vm.add_constant(Value::Integer(42));
        let index2 = vm.add_constant(Value::String("hello".to_string()));

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(vm.constants[0], Value::Integer(42));
        assert_eq!(vm.constants[1], Value::String("hello".to_string()));
    }

    #[test]
    fn test_symbol_table() {
        let mut vm = VirtualMachine::new();

        let index1 = vm.add_symbol("x".to_string());
        let index2 = vm.add_symbol("y".to_string());
        let index3 = vm.add_symbol("x".to_string()); // Duplicate

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(index3, 0); // Should return existing index
    }

    #[test]
    fn test_load_const_instruction() {
        let mut vm = VirtualMachine::new();
        let const_index = vm.add_constant(Value::Integer(42));

        let instruction = Instruction::new(OpCode::LoadConst, const_index as u32).unwrap();
        vm.load(vec![instruction], vm.constants.clone());

        vm.step().unwrap();

        assert_eq!(vm.ip, 1);
        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_push_instruction() {
        let mut vm = VirtualMachine::new();

        let instruction = Instruction::new(OpCode::Push, 42).unwrap();
        vm.load(vec![instruction], vec![]);

        vm.step().unwrap();

        assert_eq!(vm.ip, 1);
        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_dup_instruction() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 42).unwrap(),
            Instruction::new(OpCode::Dup, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 42
        vm.step().unwrap(); // Dup

        assert_eq!(vm.stack.len(), 2);
        assert_eq!(vm.stack[0], Value::Integer(42));
        assert_eq!(vm.stack[1], Value::Integer(42));
    }

    #[test]
    fn test_pop_instruction() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 42).unwrap(),
            Instruction::new(OpCode::Push, 24).unwrap(),
            Instruction::new(OpCode::Pop, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 42
        vm.step().unwrap(); // Push 24
        vm.step().unwrap(); // Pop 24

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_arithmetic_add() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 2).unwrap(),
            Instruction::new(OpCode::Push, 3).unwrap(),
            Instruction::new(OpCode::Add, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Add

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(5));
    }

    #[test]
    fn test_arithmetic_sub() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 5).unwrap(),
            Instruction::new(OpCode::Push, 3).unwrap(),
            Instruction::new(OpCode::Sub, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 5
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Sub

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(2));
    }

    #[test]
    fn test_arithmetic_mul() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 3).unwrap(),
            Instruction::new(OpCode::Push, 4).unwrap(),
            Instruction::new(OpCode::Mul, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Push 4
        vm.step().unwrap(); // Mul

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(12));
    }

    #[test]
    fn test_arithmetic_div() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 8).unwrap(),
            Instruction::new(OpCode::Push, 2).unwrap(),
            Instruction::new(OpCode::Div, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 8
        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Div

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Real(4.0));
    }

    #[test]
    fn test_arithmetic_power() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 2).unwrap(),
            Instruction::new(OpCode::Push, 3).unwrap(),
            Instruction::new(OpCode::Power, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Power

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(8));
    }

    #[test]
    fn test_division_by_zero() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::Push, 5).unwrap(),
            Instruction::new(OpCode::Push, 0).unwrap(),
            Instruction::new(OpCode::Div, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 5
        vm.step().unwrap(); // Push 0
        let result = vm.step(); // Div

        assert!(result.is_err());
        match result.unwrap_err() {
            VmError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error"),
        }
    }

    #[test]
    fn test_jump_instruction() {
        let mut vm = VirtualMachine::new();

        let instruction = Instruction::new(OpCode::Jump, 5).unwrap();
        vm.load(vec![instruction], vec![]);

        vm.step().unwrap();

        assert_eq!(vm.ip, 5);
    }

    #[test]
    fn test_jump_if_false_true_condition() {
        let mut vm = VirtualMachine::new();
        let true_index = vm.add_constant(Value::Boolean(true));

        let program = vec![
            Instruction::new(OpCode::LoadConst, true_index as u32).unwrap(),
            Instruction::new(OpCode::JumpIfFalse, 5).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load true
        vm.step().unwrap(); // JumpIfFalse

        assert_eq!(vm.ip, 2); // Should not jump
    }

    #[test]
    fn test_jump_if_false_false_condition() {
        let mut vm = VirtualMachine::new();
        let false_index = vm.add_constant(Value::Boolean(false));

        let program = vec![
            Instruction::new(OpCode::LoadConst, false_index as u32).unwrap(),
            Instruction::new(OpCode::JumpIfFalse, 5).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load false
        vm.step().unwrap(); // JumpIfFalse

        assert_eq!(vm.ip, 5); // Should jump
    }

    #[test]
    fn test_simple_program() {
        let mut vm = VirtualMachine::new();

        // Program: push 2, push 3, add, halt
        let program = vec![
            Instruction::new(OpCode::Push, 2).unwrap(),
            Instruction::new(OpCode::Push, 3).unwrap(),
            Instruction::new(OpCode::Add, 0).unwrap(),
            Instruction::new(OpCode::Halt, 0).unwrap(),
        ];

        vm.load(program, vec![]);
        let result = vm.run().unwrap();

        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_mixed_types_arithmetic() {
        let mut vm = VirtualMachine::new();
        let int_index = vm.add_constant(Value::Integer(2));
        let real_index = vm.add_constant(Value::Real(3.5));

        let program = vec![
            Instruction::new(OpCode::LoadConst, int_index as u32).unwrap(),
            Instruction::new(OpCode::LoadConst, real_index as u32).unwrap(),
            Instruction::new(OpCode::Add, 0).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load 2
        vm.step().unwrap(); // Load 3.5
        vm.step().unwrap(); // Add

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Real(5.5));
    }

    #[test]
    fn test_type_error() {
        let mut vm = VirtualMachine::new();
        let string_index = vm.add_constant(Value::String("hello".to_string()));
        let int_index = vm.add_constant(Value::Integer(42));

        let program = vec![
            Instruction::new(OpCode::LoadConst, string_index as u32).unwrap(),
            Instruction::new(OpCode::LoadConst, int_index as u32).unwrap(),
            Instruction::new(OpCode::Add, 0).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load "hello"
        vm.step().unwrap(); // Load 42
        let result = vm.step(); // Add

        assert!(result.is_err());
        match result.unwrap_err() {
            VmError::TypeError { .. } => {}
            _ => panic!("Expected TypeError"),
        }
    }
}
