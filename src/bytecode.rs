use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BytecodeError {
    #[error("Invalid opcode: {0}")]
    InvalidOpcode(u8),
    #[error("Invalid operand: {0}")]
    InvalidOperand(u32),
}

pub type Result<T> = std::result::Result<T, BytecodeError>;

/// 32-bit instruction encoding: 8-bit opcode + 24-bit operand
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instruction {
    pub opcode: OpCode,
    pub operand: u32, // 24-bit operand (0..16777215)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpCode {
    // Stack operations
    LoadConst = 0x01,  // Load constant from constant pool
    LoadSymbol = 0x02, // Load symbol from symbol table
    Push = 0x03,       // Push immediate value
    Pop = 0x04,        // Pop and discard top value
    Dup = 0x05,        // Duplicate top value

    // Arithmetic operations
    Add = 0x10,   // Pop two values, push sum
    Sub = 0x11,   // Pop two values, push difference (b - a)
    Mul = 0x12,   // Pop two values, push product
    Div = 0x13,   // Pop two values, push quotient (b / a)
    Power = 0x14, // Pop two values, push power (b ^ a)

    // Function calls
    Call = 0x20,   // Call function with n arguments
    Return = 0x21, // Return from function

    // Control flow
    Jump = 0x30,        // Unconditional jump
    JumpIfFalse = 0x31, // Jump if top of stack is false

    // Halt
    Halt = 0xFF, // Stop execution
}

impl Instruction {
    /// Create a new instruction with the given opcode and operand
    pub fn new(opcode: OpCode, operand: u32) -> Result<Self> {
        if operand > 0xFFFFFF {
            return Err(BytecodeError::InvalidOperand(operand));
        }
        Ok(Instruction { opcode, operand })
    }

    /// Encode instruction to 32-bit integer
    pub fn encode(&self) -> u32 {
        ((self.opcode as u32) << 24) | (self.operand & 0xFFFFFF)
    }

    /// Decode 32-bit integer to instruction
    pub fn decode(encoded: u32) -> Result<Self> {
        let opcode_byte = (encoded >> 24) as u8;
        let operand = encoded & 0xFFFFFF;

        let opcode = OpCode::from_u8(opcode_byte)?;
        Ok(Instruction { opcode, operand })
    }
}

impl OpCode {
    pub fn from_u8(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(OpCode::LoadConst),
            0x02 => Ok(OpCode::LoadSymbol),
            0x03 => Ok(OpCode::Push),
            0x04 => Ok(OpCode::Pop),
            0x05 => Ok(OpCode::Dup),
            0x10 => Ok(OpCode::Add),
            0x11 => Ok(OpCode::Sub),
            0x12 => Ok(OpCode::Mul),
            0x13 => Ok(OpCode::Div),
            0x14 => Ok(OpCode::Power),
            0x20 => Ok(OpCode::Call),
            0x21 => Ok(OpCode::Return),
            0x30 => Ok(OpCode::Jump),
            0x31 => Ok(OpCode::JumpIfFalse),
            0xFF => Ok(OpCode::Halt),
            _ => Err(BytecodeError::InvalidOpcode(value)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_creation() {
        let inst = Instruction::new(OpCode::LoadConst, 42).unwrap();
        assert_eq!(inst.opcode, OpCode::LoadConst);
        assert_eq!(inst.operand, 42);
    }

    #[test]
    fn test_instruction_creation_invalid_operand() {
        let result = Instruction::new(OpCode::LoadConst, 0x1000000); // > 24-bit max
        assert!(result.is_err());
        match result.unwrap_err() {
            BytecodeError::InvalidOperand(0x1000000) => {}
            _ => panic!("Expected InvalidOperand error"),
        }
    }

    #[test]
    fn test_instruction_encoding() {
        let inst = Instruction::new(OpCode::LoadConst, 42).unwrap();
        let encoded = inst.encode();

        // OpCode::LoadConst = 0x01, operand = 42
        // Expected: 0x01 << 24 | 42 = 0x0100002A
        assert_eq!(encoded, 0x0100002A);
    }

    #[test]
    fn test_instruction_decoding() {
        let encoded = 0x0100002A; // LoadConst with operand 42
        let inst = Instruction::decode(encoded).unwrap();

        assert_eq!(inst.opcode, OpCode::LoadConst);
        assert_eq!(inst.operand, 42);
    }

    #[test]
    fn test_instruction_roundtrip() {
        let original = Instruction::new(OpCode::Add, 0x123456).unwrap();
        let encoded = original.encode();
        let decoded = Instruction::decode(encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_opcode_from_u8_valid() {
        assert_eq!(OpCode::from_u8(0x01).unwrap(), OpCode::LoadConst);
        assert_eq!(OpCode::from_u8(0x10).unwrap(), OpCode::Add);
        assert_eq!(OpCode::from_u8(0xFF).unwrap(), OpCode::Halt);
    }

    #[test]
    fn test_opcode_from_u8_invalid() {
        let result = OpCode::from_u8(0x99);
        assert!(result.is_err());
        match result.unwrap_err() {
            BytecodeError::InvalidOpcode(0x99) => {}
            _ => panic!("Expected InvalidOpcode error"),
        }
    }

    #[test]
    fn test_all_opcodes_roundtrip() {
        let opcodes = vec![
            OpCode::LoadConst,
            OpCode::LoadSymbol,
            OpCode::Push,
            OpCode::Pop,
            OpCode::Dup,
            OpCode::Add,
            OpCode::Sub,
            OpCode::Mul,
            OpCode::Div,
            OpCode::Power,
            OpCode::Call,
            OpCode::Return,
            OpCode::Jump,
            OpCode::JumpIfFalse,
            OpCode::Halt,
        ];

        for opcode in opcodes {
            let inst = Instruction::new(opcode, 0x123).unwrap();
            let encoded = inst.encode();
            let decoded = Instruction::decode(encoded).unwrap();
            assert_eq!(inst, decoded);
        }
    }

    #[test]
    fn test_max_operand_value() {
        let max_operand = 0xFFFFFF; // 24-bit max
        let inst = Instruction::new(OpCode::Push, max_operand).unwrap();
        assert_eq!(inst.operand, max_operand);

        let encoded = inst.encode();
        let decoded = Instruction::decode(encoded).unwrap();
        assert_eq!(decoded.operand, max_operand);
    }

    #[test]
    fn test_instruction_size() {
        // Ensure instruction fits in 32 bits
        assert_eq!(std::mem::size_of::<Instruction>(), 8); // opcode + operand
        assert_eq!(std::mem::size_of::<OpCode>(), 1);
    }

    // ===== MINIMAL OPCODE SET TESTS (TDD Phase 2A) =====
    // These tests define the expected behavior of the new minimal opcode set
    // They should fail initially until we implement the new architecture

    #[test]
    fn test_minimal_opcode_count() {
        // Test that we have exactly 18 opcodes in the minimal set
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        assert_eq!(minimal_opcodes.len(), 18, "Minimal opcode set must have exactly 18 opcodes");
        
        // Verify each category has the expected count
        let loads_stores = minimal_opcodes.iter().filter(|op| op.is_load_store()).count();
        assert_eq!(loads_stores, 4, "Should have 4 load/store opcodes (LDS removed)");
        
        let aggregates = minimal_opcodes.iter().filter(|op| op.is_aggregate()).count();
        assert_eq!(aggregates, 2, "Should have 2 aggregate opcodes");
        
        let math = minimal_opcodes.iter().filter(|op| op.is_math()).count();
        assert_eq!(math, 5, "Should have 5 math opcodes (NEG removed)");
        
        let control = minimal_opcodes.iter().filter(|op| op.is_control()).count();
        assert_eq!(control, 2, "Should have 2 control opcodes (NOP removed)");
        
        let calls = minimal_opcodes.iter().filter(|op| op.is_call()).count();
        assert_eq!(calls, 2, "Should have 2 call opcodes");
        
        let stack = minimal_opcodes.iter().filter(|op| matches!(op, MinimalOpCode::POP | MinimalOpCode::DUP)).count();
        assert_eq!(stack, 2, "Should have 2 stack opcodes (SWAP removed)");
        
        let system = minimal_opcodes.iter().filter(|op| matches!(op, MinimalOpCode::SYS)).count();
        assert_eq!(system, 1, "Should have 1 system opcode");
    }

    #[test]
    fn test_minimal_opcode_encoding() {
        // Test that minimal opcodes encode/decode correctly
        let test_cases = vec![
            (MinimalOpCode::LDC, 42),
            (MinimalOpCode::ADD, 0),
            (MinimalOpCode::CALL, 3), // func_id encoded in high bits, argc in low bits
            (MinimalOpCode::SYS, 5),
        ];
        
        for (opcode, operand) in test_cases {
            let inst = MinimalInstruction::new(opcode, operand).unwrap();
            let encoded = inst.encode();
            let decoded = MinimalInstruction::decode(encoded).unwrap();
            
            assert_eq!(inst.opcode, decoded.opcode);
            assert_eq!(inst.operand, decoded.operand);
        }
    }

    #[test]
    fn test_minimal_call_encoding() {
        // Test special encoding for CALL opcode: func_id in high 16 bits, argc in low 8 bits
        let func_id = 1234;
        let argc = 5;
        
        let call_inst = MinimalInstruction::new_call(func_id, argc).unwrap();
        assert_eq!(call_inst.opcode, MinimalOpCode::CALL);
        
        let (decoded_func_id, decoded_argc) = call_inst.decode_call();
        assert_eq!(decoded_func_id, func_id);
        assert_eq!(decoded_argc, argc);
    }

    #[test]
    fn test_minimal_sys_encoding() {
        // Test SYS opcode encoding: sys_op in high 16 bits, argc in low 8 bits
        let sys_op = 42; // Arbitrary system call ID
        let argc = 3;
        
        let sys_inst = MinimalInstruction::new_sys(sys_op, argc).unwrap();
        assert_eq!(sys_inst.opcode, MinimalOpCode::SYS);
        
        let (decoded_sys_op, decoded_argc) = sys_inst.decode_sys();
        assert_eq!(decoded_sys_op, sys_op);
        assert_eq!(decoded_argc, argc);
    }

    #[test]
    fn test_minimal_opcodes_are_minimal() {
        // Test that we removed all complex opcodes that should be handled by stdlib
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        // Should not have any pattern matching opcodes
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Pattern")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Match")));
        
        // Should not have any table-specific opcodes
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Table")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Group")));
        
        // Should not have any complex attribute opcodes
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Hold")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Listable")));
    }

    #[test]
    fn test_minimal_comparison_opcodes() {
        // Test that comparison operations are simplified to core math
        // More complex comparisons should be handled by stdlib functions
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        // Should have basic math that can be used to build comparisons
        assert!(minimal_opcodes.contains(&MinimalOpCode::SUB)); // For implementing EQ via subtraction
        assert!(minimal_opcodes.contains(&MinimalOpCode::ADD));
        assert!(minimal_opcodes.contains(&MinimalOpCode::MUL));
        assert!(minimal_opcodes.contains(&MinimalOpCode::DIV));
        assert!(minimal_opcodes.contains(&MinimalOpCode::POW));
        
        // Complex comparison should be library functions, not opcodes
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "GT"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "LT"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "GE"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "LE"));
    }

    #[test]
    fn test_minimal_aggregate_construction() {
        // Test opcodes for building lists and associative arrays
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&MinimalOpCode::NEWLIST));
        assert!(minimal_opcodes.contains(&MinimalOpCode::NEWASSOC));
        
        // These should be the only aggregate construction opcodes
        let aggregate_count = minimal_opcodes.iter()
            .filter(|op| op.name().contains("NEW"))
            .count();
        assert_eq!(aggregate_count, 2);
    }

    #[test]
    fn test_stack_discipline_opcodes() {
        // Test that we have essential stack manipulation opcodes
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&MinimalOpCode::POP));
        assert!(minimal_opcodes.contains(&MinimalOpCode::DUP));
        
        // These should be sufficient for all stack manipulations (SWAP removed)
        let stack_ops = minimal_opcodes.iter()
            .filter(|op| matches!(op, MinimalOpCode::POP | MinimalOpCode::DUP))
            .count();
        assert_eq!(stack_ops, 2);
    }

    #[test] 
    fn test_load_store_completeness() {
        // Test that we have all necessary load/store operations
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        // Constant and literal loading
        assert!(minimal_opcodes.contains(&MinimalOpCode::LDC)); // Load constant (includes symbols)
        assert!(minimal_opcodes.contains(&MinimalOpCode::LDL)); // Load local
        assert!(minimal_opcodes.contains(&MinimalOpCode::STL)); // Store local
        
        // Symbol operations (only store needed - load via LDC)
        assert!(minimal_opcodes.contains(&MinimalOpCode::STS)); // Store symbol
        
        let load_store_count = minimal_opcodes.iter()
            .filter(|op| op.is_load_store())
            .count();
        assert_eq!(load_store_count, 4);
    }

    #[test]
    fn test_control_flow_minimal() {
        // Test minimal control flow opcodes
        let minimal_opcodes = MinimalOpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&MinimalOpCode::JMP));  // Unconditional jump
        assert!(minimal_opcodes.contains(&MinimalOpCode::JIF));  // Jump if true
        
        // Should not have complex control flow - those become library functions
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Loop")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("While")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("For")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("NOP"))); // NOP removed
    }

    #[test]
    fn test_opcode_size_efficiency() {
        // Test that minimal opcodes fit efficiently in instruction encoding
        assert_eq!(std::mem::size_of::<MinimalOpCode>(), 1); // Should be u8
        assert_eq!(std::mem::size_of::<MinimalInstruction>(), 8); // opcode + operand
        
        // All opcodes should fit in 8 bits
        let all_opcodes = MinimalOpCode::all_opcodes();
        for opcode in all_opcodes {
            let byte_value = opcode as u8;
            assert!(byte_value <= 255);
            
            // Should be able to roundtrip through byte encoding
            let decoded = MinimalOpCode::from_u8(byte_value).unwrap();
            assert_eq!(decoded, opcode);
        }
    }

    // Define the new minimal opcode types that will be implemented (exactly 18)
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    #[repr(u8)]
    enum MinimalOpCode {
        // Load/Store (4) - LDS removed: symbols loaded via LDC
        LDC = 0x01,   // Load constant from pool (includes symbols)
        LDL = 0x02,   // Load local variable
        STL = 0x03,   // Store local variable  
        STS = 0x04,   // Store symbol value
        
        // Aggregates (2)
        NEWLIST = 0x10,  // Create new list from n stack items
        NEWASSOC = 0x11, // Create new associative array from 2n stack items
        
        // Math (5) - Reduced by 1: NEG can be implemented as 0 - x
        ADD = 0x20,   // Add two values
        SUB = 0x21,   // Subtract two values  
        MUL = 0x22,   // Multiply two values
        DIV = 0x23,   // Divide two values
        POW = 0x24,   // Power of two values
        
        // Control (2) - Reduced by 1: NOP rarely needed
        JMP = 0x30,   // Unconditional jump
        JIF = 0x31,   // Jump if true
        
        // Calls (2)
        CALL = 0x40,  // Call function (func_id, argc encoded in operand)
        RET = 0x41,   // Return from function
        
        // Stack (2) - Reduced by 1: SWAP can be implemented with DUP operations
        POP = 0x50,   // Pop and discard
        DUP = 0x51,   // Duplicate top
        
        // System (1)
        SYS = 0x60,   // System call (sys_op, argc encoded in operand)
    }

    impl MinimalOpCode {
        fn all_opcodes() -> Vec<MinimalOpCode> {
            vec![
                Self::LDC, Self::LDL, Self::STL, Self::STS,
                Self::NEWLIST, Self::NEWASSOC,
                Self::ADD, Self::SUB, Self::MUL, Self::DIV, Self::POW,
                Self::JMP, Self::JIF,
                Self::CALL, Self::RET,
                Self::POP, Self::DUP,
                Self::SYS,
            ]
        }
        
        fn is_load_store(&self) -> bool {
            matches!(self, Self::LDC | Self::LDL | Self::STL | Self::STS)
        }
        
        fn is_aggregate(&self) -> bool {
            matches!(self, Self::NEWLIST | Self::NEWASSOC)
        }
        
        fn is_math(&self) -> bool {
            matches!(self, Self::ADD | Self::SUB | Self::MUL | Self::DIV | Self::POW)
        }
        
        fn is_control(&self) -> bool {
            matches!(self, Self::JMP | Self::JIF)
        }
        
        fn is_call(&self) -> bool {
            matches!(self, Self::CALL | Self::RET)
        }
        
        fn name(&self) -> &'static str {
            match self {
                Self::LDC => "LDC", Self::LDL => "LDL", Self::STL => "STL", 
                Self::STS => "STS",
                Self::NEWLIST => "NEWLIST", Self::NEWASSOC => "NEWASSOC",
                Self::ADD => "ADD", Self::SUB => "SUB", Self::MUL => "MUL", 
                Self::DIV => "DIV", Self::POW => "POW",
                Self::JMP => "JMP", Self::JIF => "JIF",
                Self::CALL => "CALL", Self::RET => "RET",
                Self::POP => "POP", Self::DUP => "DUP",
                Self::SYS => "SYS",
            }
        }
        
        fn from_u8(value: u8) -> Result<Self> {
            match value {
                0x01 => Ok(Self::LDC), 0x02 => Ok(Self::LDL), 0x03 => Ok(Self::STL),
                0x04 => Ok(Self::STS),
                0x10 => Ok(Self::NEWLIST), 0x11 => Ok(Self::NEWASSOC),
                0x20 => Ok(Self::ADD), 0x21 => Ok(Self::SUB), 0x22 => Ok(Self::MUL),
                0x23 => Ok(Self::DIV), 0x24 => Ok(Self::POW),
                0x30 => Ok(Self::JMP), 0x31 => Ok(Self::JIF),
                0x40 => Ok(Self::CALL), 0x41 => Ok(Self::RET),
                0x50 => Ok(Self::POP), 0x51 => Ok(Self::DUP),
                0x60 => Ok(Self::SYS),
                _ => Err(BytecodeError::InvalidOpcode(value)),
            }
        }
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MinimalInstruction {
        opcode: MinimalOpCode,
        operand: u32,
    }

    impl MinimalInstruction {
        fn new(opcode: MinimalOpCode, operand: u32) -> Result<Self> {
            if operand > 0xFFFFFF {
                return Err(BytecodeError::InvalidOperand(operand));
            }
            Ok(Self { opcode, operand })
        }
        
        fn new_call(func_id: u16, argc: u8) -> Result<Self> {
            let operand = ((func_id as u32) << 8) | (argc as u32);
            Self::new(MinimalOpCode::CALL, operand)
        }
        
        fn new_sys(sys_op: u16, argc: u8) -> Result<Self> {
            let operand = ((sys_op as u32) << 8) | (argc as u32);
            Self::new(MinimalOpCode::SYS, operand)
        }
        
        fn decode_call(&self) -> (u16, u8) {
            let func_id = (self.operand >> 8) as u16;
            let argc = (self.operand & 0xFF) as u8;
            (func_id, argc)
        }
        
        fn decode_sys(&self) -> (u16, u8) {
            let sys_op = (self.operand >> 8) as u16;
            let argc = (self.operand & 0xFF) as u8;
            (sys_op, argc)
        }
        
        fn encode(&self) -> u32 {
            ((self.opcode as u32) << 24) | (self.operand & 0xFFFFFF)
        }
        
        fn decode(encoded: u32) -> Result<Self> {
            let opcode_byte = (encoded >> 24) as u8;
            let operand = encoded & 0xFFFFFF;
            
            let opcode = MinimalOpCode::from_u8(opcode_byte)?;
            Ok(Self { opcode, operand })
        }
    }
}
