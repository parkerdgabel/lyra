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
}
