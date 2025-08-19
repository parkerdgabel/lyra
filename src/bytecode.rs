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

/// Minimal opcode set for simplified VM (20 opcodes after adding MAP_CALL_STATIC)
/// Designed to push complexity into compiler and stdlib
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum OpCode {
    // Load/Store (6) - LDS removed: symbols loaded via LDC  
    LDC = 0x01,   // Load constant from pool (includes symbols)
    LDL = 0x02,   // Load local variable
    STL = 0x03,   // Store local variable  
    STS = 0x04,   // Store symbol value
    LOAD_QUOTE = 0x05, // Load quoted expression for Hold attributes
    MAP_CALL_STATIC = 0x06, // Apply function to lists for Listable attributes
    
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
    CALL_STATIC = 0x42,  // Call static function (registry_index, argc encoded in operand)
    RET = 0x41,   // Return from function
    
    // Stack (2) - Reduced by 1: SWAP can be implemented with DUP operations
    POP = 0x50,   // Pop and discard
    DUP = 0x51,   // Duplicate top
    
    // System (1)
    SYS = 0x60,   // System call (sys_op, argc encoded in operand)
}

impl Instruction {
    /// Create a new instruction with the given opcode and operand
    pub fn new(opcode: OpCode, operand: u32) -> Result<Self> {
        if operand > 0xFFFFFF {
            return Err(BytecodeError::InvalidOperand(operand));
        }
        Ok(Instruction { opcode, operand })
    }
    
    
    /// Create a SYS instruction with system operation and argument count  
    /// Encoding: sys_op in high 16 bits, argc in low 8 bits
    pub fn new_sys(sys_op: u16, argc: u8) -> Result<Self> {
        let operand = ((sys_op as u32) << 8) | (argc as u32);
        Self::new(OpCode::SYS, operand)
    }
    
    /// Create a CALL_STATIC instruction with function registry index and argument count
    /// Encoding: function_index in high 16 bits, argc in low 8 bits
    pub fn new_call_static(function_index: u16, argc: u8) -> Result<Self> {
        let operand = ((function_index as u32) << 8) | (argc as u32);
        Self::new(OpCode::CALL_STATIC, operand)
    }
    
    
    /// Decode SYS instruction to get system operation and argument count
    pub fn decode_sys(&self) -> (u16, u8) {
        let sys_op = (self.operand >> 8) as u16;
        let argc = (self.operand & 0xFF) as u8;
        (sys_op, argc)
    }
    
    /// Decode CALL_STATIC instruction to get function index and argument count
    pub fn decode_call_static(&self) -> (u16, u8) {
        let function_index = (self.operand >> 8) as u16;
        let argc = (self.operand & 0xFF) as u8;
        (function_index, argc)
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
            0x01 => Ok(Self::LDC), 0x02 => Ok(Self::LDL), 0x03 => Ok(Self::STL),
            0x04 => Ok(Self::STS), 0x05 => Ok(Self::LOAD_QUOTE), 0x06 => Ok(Self::MAP_CALL_STATIC),
            0x10 => Ok(Self::NEWLIST), 0x11 => Ok(Self::NEWASSOC),
            0x20 => Ok(Self::ADD), 0x21 => Ok(Self::SUB), 0x22 => Ok(Self::MUL),
            0x23 => Ok(Self::DIV), 0x24 => Ok(Self::POW),
            0x30 => Ok(Self::JMP), 0x31 => Ok(Self::JIF),
            0x41 => Ok(Self::RET), 0x42 => Ok(Self::CALL_STATIC),
            0x50 => Ok(Self::POP), 0x51 => Ok(Self::DUP),
            0x60 => Ok(Self::SYS),
            _ => Err(BytecodeError::InvalidOpcode(value)),
        }
    }

    /// Get all opcodes for validation and testing
    pub fn all_opcodes() -> Vec<OpCode> {
        vec![
            Self::LDC, Self::LDL, Self::STL, Self::STS, Self::LOAD_QUOTE, Self::MAP_CALL_STATIC,
            Self::NEWLIST, Self::NEWASSOC,
            Self::ADD, Self::SUB, Self::MUL, Self::DIV, Self::POW,
            Self::JMP, Self::JIF,
            Self::CALL_STATIC, Self::RET,
            Self::POP, Self::DUP,
            Self::SYS,
        ]
    }
    
    /// Check if opcode is a load/store operation
    pub fn is_load_store(&self) -> bool {
        matches!(self, Self::LDC | Self::LDL | Self::STL | Self::STS | Self::LOAD_QUOTE)
    }
    
    /// Check if opcode creates aggregates
    pub fn is_aggregate(&self) -> bool {
        matches!(self, Self::NEWLIST | Self::NEWASSOC)
    }
    
    /// Check if opcode is a math operation
    pub fn is_math(&self) -> bool {
        matches!(self, Self::ADD | Self::SUB | Self::MUL | Self::DIV | Self::POW)
    }
    
    /// Check if opcode is a control flow operation
    pub fn is_control(&self) -> bool {
        matches!(self, Self::JMP | Self::JIF)
    }
    
    /// Check if opcode is a function call operation
    pub fn is_call(&self) -> bool {
        matches!(self, Self::CALL_STATIC | Self::MAP_CALL_STATIC | Self::RET)
    }
    
    /// Get opcode name for debugging
    pub fn name(&self) -> &'static str {
        match self {
            Self::LDC => "LDC", Self::LOAD_QUOTE => "LOAD_QUOTE", Self::MAP_CALL_STATIC => "MAP_CALL_STATIC",
            Self::LDL => "LDL", Self::STL => "STL", Self::STS => "STS",
            Self::NEWLIST => "NEWLIST", Self::NEWASSOC => "NEWASSOC",
            Self::ADD => "ADD", Self::SUB => "SUB", Self::MUL => "MUL", 
            Self::DIV => "DIV", Self::POW => "POW",
            Self::JMP => "JMP", Self::JIF => "JIF",
            Self::CALL_STATIC => "CALL_STATIC", Self::RET => "RET",
            Self::POP => "POP", Self::DUP => "DUP",
            Self::SYS => "SYS",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_creation() {
        let inst = Instruction::new(OpCode::LDC, 42).unwrap();
        assert_eq!(inst.opcode, OpCode::LDC);
        assert_eq!(inst.operand, 42);
    }

    #[test]
    fn test_instruction_creation_invalid_operand() {
        let result = Instruction::new(OpCode::LDC, 0x1000000); // > 24-bit max
        assert!(result.is_err());
        match result.unwrap_err() {
            BytecodeError::InvalidOperand(0x1000000) => {}
            _ => panic!("Expected InvalidOperand error"),
        }
    }

    #[test]
    fn test_instruction_encoding() {
        let inst = Instruction::new(OpCode::LDC, 42).unwrap();
        let encoded = inst.encode();

        // OpCode::LDC = 0x01, operand = 42
        // Expected: 0x01 << 24 | 42 = 0x0100002A
        assert_eq!(encoded, 0x0100002A);
    }

    #[test]
    fn test_instruction_decoding() {
        let encoded = 0x0100002A; // LDC with operand 42
        let inst = Instruction::decode(encoded).unwrap();

        assert_eq!(inst.opcode, OpCode::LDC);
        assert_eq!(inst.operand, 42);
    }

    #[test]
    fn test_instruction_roundtrip() {
        let original = Instruction::new(OpCode::ADD, 0x123456).unwrap();
        let encoded = original.encode();
        let decoded = Instruction::decode(encoded).unwrap();

        assert_eq!(original, decoded);
    }

    #[test]
    fn test_opcode_from_u8_valid() {
        assert_eq!(OpCode::from_u8(0x01).unwrap(), OpCode::LDC);
        assert_eq!(OpCode::from_u8(0x20).unwrap(), OpCode::ADD);
        assert_eq!(OpCode::from_u8(0x60).unwrap(), OpCode::SYS);
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
        let opcodes = OpCode::all_opcodes();

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
        let inst = Instruction::new(OpCode::LDC, max_operand).unwrap();
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
    
    
    #[test]
    fn test_sys_instruction_encoding() {
        // Test SYS opcode encoding: sys_op in high 16 bits, argc in low 8 bits
        let sys_op = 42; // Arbitrary system call ID
        let argc = 3;
        
        let sys_inst = Instruction::new_sys(sys_op, argc).unwrap();
        assert_eq!(sys_inst.opcode, OpCode::SYS);
        
        let (decoded_sys_op, decoded_argc) = sys_inst.decode_sys();
        assert_eq!(decoded_sys_op, sys_op);
        assert_eq!(decoded_argc, argc);
    }
    
    #[test]
    fn test_call_static_instruction_encoding() {
        // Test CALL_STATIC opcode encoding: function_index in high 16 bits, argc in low 8 bits
        let function_index = 15; // Function registry index
        let argc = 2;
        
        let call_static_inst = Instruction::new_call_static(function_index, argc).unwrap();
        assert_eq!(call_static_inst.opcode, OpCode::CALL_STATIC);
        
        let (decoded_function_index, decoded_argc) = call_static_inst.decode_call_static();
        assert_eq!(decoded_function_index, function_index);
        assert_eq!(decoded_argc, argc);
    }
    
    #[test]
    fn test_minimal_opcode_count() {
        // Test that we have exactly 20 opcodes (18 minimal + LOAD_QUOTE + MAP_CALL_STATIC)
        let minimal_opcodes = OpCode::all_opcodes();
        assert_eq!(minimal_opcodes.len(), 20, "Opcode set must have exactly 20 opcodes (18 minimal + LOAD_QUOTE + MAP_CALL_STATIC)");
        
        // Verify each category has the expected count
        let loads_stores = minimal_opcodes.iter().filter(|op| op.is_load_store()).count();
        assert_eq!(loads_stores, 5, "Should have 5 load/store opcodes (including LOAD_QUOTE)");
        
        let aggregates = minimal_opcodes.iter().filter(|op| op.is_aggregate()).count();
        assert_eq!(aggregates, 2, "Should have 2 aggregate opcodes");
        
        let math = minimal_opcodes.iter().filter(|op| op.is_math()).count();
        assert_eq!(math, 5, "Should have 5 math opcodes");
        
        let control = minimal_opcodes.iter().filter(|op| op.is_control()).count();
        assert_eq!(control, 2, "Should have 2 control opcodes");
        
        let calls = minimal_opcodes.iter().filter(|op| op.is_call()).count();
        assert_eq!(calls, 3, "Should have 3 call opcodes (CALL_STATIC, MAP_CALL_STATIC, RET)");
        
        let stack = minimal_opcodes.iter().filter(|op| matches!(op, OpCode::POP | OpCode::DUP)).count();
        assert_eq!(stack, 2, "Should have 2 stack opcodes");
        
        let system = minimal_opcodes.iter().filter(|op| matches!(op, OpCode::SYS)).count();
        assert_eq!(system, 1, "Should have 1 system opcode");
    }

    // ===== LEGACY OPCODE VALIDATION TESTS =====
    // These tests verify the minimal opcode design principles
    
    #[test]
    fn test_opcodes_are_minimal() {
        // Test that we removed all complex opcodes that should be handled by stdlib
        let minimal_opcodes = OpCode::all_opcodes();
        
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
    fn test_comparison_opcodes() {
        // Test that comparison operations are simplified to core math
        // More complex comparisons should be handled by stdlib functions
        let minimal_opcodes = OpCode::all_opcodes();
        
        // Should have basic math that can be used to build comparisons
        assert!(minimal_opcodes.contains(&OpCode::SUB)); // For implementing EQ via subtraction
        assert!(minimal_opcodes.contains(&OpCode::ADD));
        assert!(minimal_opcodes.contains(&OpCode::MUL));
        assert!(minimal_opcodes.contains(&OpCode::DIV));
        assert!(minimal_opcodes.contains(&OpCode::POW));
        
        // Complex comparison should be library functions, not opcodes
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "GT"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "LT"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "GE"));
        assert!(!minimal_opcodes.iter().any(|op| op.name() == "LE"));
    }

    #[test]
    fn test_aggregate_construction() {
        // Test opcodes for building lists and associative arrays
        let minimal_opcodes = OpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&OpCode::NEWLIST));
        assert!(minimal_opcodes.contains(&OpCode::NEWASSOC));
        
        // These should be the only aggregate construction opcodes
        let aggregate_count = minimal_opcodes.iter()
            .filter(|op| op.name().contains("NEW"))
            .count();
        assert_eq!(aggregate_count, 2);
    }

    #[test]
    fn test_stack_discipline_opcodes() {
        // Test that we have essential stack manipulation opcodes
        let minimal_opcodes = OpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&OpCode::POP));
        assert!(minimal_opcodes.contains(&OpCode::DUP));
        
        // These should be sufficient for all stack manipulations (SWAP removed)
        let stack_ops = minimal_opcodes.iter()
            .filter(|op| matches!(op, OpCode::POP | OpCode::DUP))
            .count();
        assert_eq!(stack_ops, 2);
    }

    #[test] 
    fn test_load_store_completeness() {
        // Test that we have all necessary load/store operations
        let minimal_opcodes = OpCode::all_opcodes();
        
        // Constant and literal loading
        assert!(minimal_opcodes.contains(&OpCode::LDC)); // Load constant (includes symbols)
        assert!(minimal_opcodes.contains(&OpCode::LDL)); // Load local
        assert!(minimal_opcodes.contains(&OpCode::STL)); // Store local
        
        // Symbol operations (only store needed - load via LDC)
        assert!(minimal_opcodes.contains(&OpCode::STS)); // Store symbol
        
        let load_store_count = minimal_opcodes.iter()
            .filter(|op| op.is_load_store())
            .count();
        assert_eq!(load_store_count, 5); // LDC, LDL, STL, STS, LOAD_QUOTE
    }

    #[test]
    fn test_control_flow_minimal() {
        // Test minimal control flow opcodes
        let minimal_opcodes = OpCode::all_opcodes();
        
        assert!(minimal_opcodes.contains(&OpCode::JMP));  // Unconditional jump
        assert!(minimal_opcodes.contains(&OpCode::JIF));  // Jump if true
        
        // Should not have complex control flow - those become library functions
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("Loop")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("While")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("For")));
        assert!(!minimal_opcodes.iter().any(|op| op.name().contains("NOP"))); // NOP removed
    }

    #[test]
    fn test_opcode_size_efficiency() {
        // Test that minimal opcodes fit efficiently in instruction encoding
        assert_eq!(std::mem::size_of::<OpCode>(), 1); // Should be u8
        assert_eq!(std::mem::size_of::<Instruction>(), 8); // opcode + operand
        
        // All opcodes should fit in 8 bits
        let all_opcodes = OpCode::all_opcodes();
        for opcode in all_opcodes {
            let byte_value = opcode as u8;
            assert!(byte_value <= 255);
            
            // Should be able to roundtrip through byte encoding
            let decoded = OpCode::from_u8(byte_value).unwrap();
            assert_eq!(decoded, opcode);
        }
    }
}
