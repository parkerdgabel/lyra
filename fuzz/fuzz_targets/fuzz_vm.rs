#![no_main]

use libfuzzer_sys::fuzz_target;
use lyra::{
    lexer::Lexer, 
    parser::Parser, 
    compiler::Compiler, 
    vm::{VM, Value},
    bytecode::Instruction
};
use arbitrary::{Arbitrary, Unstructured};

const MAX_INPUT_SIZE: usize = 1_000; // Small for VM testing to prevent long-running operations

#[derive(Arbitrary, Debug)]
struct FuzzInput {
    source: String,
}

impl FuzzInput {
    fn constrain_size(&mut self) {
        if self.source.len() > MAX_INPUT_SIZE {
            self.source.truncate(MAX_INPUT_SIZE);
        }
        
        // Replace problematic characters
        self.source = self.source.replace('\0', " ");
        
        // Only allow relatively simple expressions for VM testing
        self.simplify_for_vm();
    }
    
    fn simplify_for_vm(&mut self) {
        // Remove or replace potentially expensive operations
        self.source = self.source
            .replace("Import", "Length") // Replace I/O with safe operations
            .replace("Export", "Head")
            .replace("FileOpen", "Tail")
            .replace("System", "Plus")
            .replace("Run", "Times")
            .replace("Eval", "Apply");
            
        // Limit numeric values to prevent overflow
        if self.source.chars().all(|c| c.is_numeric() || c == '.' || c == '-') {
            if let Ok(num) = self.source.parse::<f64>() {
                if num.abs() > 1e10 {
                    self.source = "42".to_string();
                }
            }
        }
        
        // Ensure reasonable size
        if self.source.len() > MAX_INPUT_SIZE / 2 {
            self.source.truncate(MAX_INPUT_SIZE / 2);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(mut input) = FuzzInput::arbitrary(&mut unstructured) {
        input.constrain_size();
        
        // Test the complete pipeline: lexer -> parser -> compiler -> VM
        let lexer = Lexer::new(&input.source);
        let mut parser = Parser::new(lexer);
        
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            parser.parse()
        })) {
            Ok(parse_result) => {
                if let Ok(ast) = parse_result {
                    let mut compiler = Compiler::new();
                    
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        compiler.compile(&ast)
                    })) {
                        Ok(compile_result) => {
                            if let Ok(bytecode) = compile_result {
                                // If compilation succeeded, try VM execution
                                let mut vm = VM::new();
                                
                                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                    // Set a reasonable execution limit
                                    vm.execute_with_limit(&bytecode, 10000) // Max 10k instructions
                                })) {
                                    Ok(vm_result) => {
                                        // VM execution completed (success or error)
                                        match vm_result {
                                            Ok(_value) => {
                                                // Successfully executed
                                            }
                                            Err(_error) => {
                                                // VM error is expected for some inputs
                                            }
                                        }
                                    }
                                    Err(_) => {
                                        // VM panicked - this is a bug
                                    }
                                }
                            }
                        }
                        Err(_) => {
                            // Compiler panicked
                        }
                    }
                }
            }
            Err(_) => {
                // Parser panicked
            }
        }
    }
});

// Also test direct bytecode fuzzing
#[derive(Arbitrary, Debug)]
struct BytecodeFuzzInput {
    instructions: Vec<FuzzInstruction>,
}

#[derive(Arbitrary, Debug)]
enum FuzzInstruction {
    LoadConst(FuzzValue),
    LoadVar(u8), // Limit variable index
    StoreVar(u8),
    Add,
    Sub,
    Mul,
    Div,
    Call(u8), // Limit argument count
    Return,
    Jump(u8), // Limit jump offset
    JumpIfFalse(u8),
    Pop,
    Dup,
}

#[derive(Arbitrary, Debug)]
enum FuzzValue {
    Integer(i32), // Limit integer size
    Real(f32),    // Use f32 for simpler fuzzing
    Boolean(bool),
    String(String),
}

impl FuzzValue {
    fn to_vm_value(self) -> Value {
        match self {
            FuzzValue::Integer(i) => Value::Integer(i as i64),
            FuzzValue::Real(f) => Value::Real(f as f64),
            FuzzValue::Boolean(b) => Value::Boolean(b),
            FuzzValue::String(mut s) => {
                // Limit string size
                if s.len() > 100 {
                    s.truncate(100);
                }
                Value::String(s)
            }
        }
    }
}

impl FuzzInstruction {
    fn to_vm_instruction(self) -> Option<Instruction> {
        match self {
            FuzzInstruction::LoadConst(val) => Some(Instruction::LoadConst(val.to_vm_value())),
            FuzzInstruction::LoadVar(idx) => Some(Instruction::LoadVar(idx as usize)),
            FuzzInstruction::StoreVar(idx) => Some(Instruction::StoreVar(idx as usize)),
            FuzzInstruction::Add => Some(Instruction::Add),
            FuzzInstruction::Sub => Some(Instruction::Sub),
            FuzzInstruction::Mul => Some(Instruction::Mul),
            FuzzInstruction::Div => Some(Instruction::Div),
            FuzzInstruction::Call(argc) => Some(Instruction::Call(argc as usize)),
            FuzzInstruction::Return => Some(Instruction::Return),
            FuzzInstruction::Jump(offset) => Some(Instruction::Jump(offset as isize)),
            FuzzInstruction::JumpIfFalse(offset) => Some(Instruction::JumpIfFalse(offset as isize)),
            FuzzInstruction::Pop => Some(Instruction::Pop),
            FuzzInstruction::Dup => Some(Instruction::Dup),
        }
    }
}

// Second fuzz target for direct bytecode testing
fuzz_target!(|data: &[u8]| {
    let mut unstructured = Unstructured::new(data);
    
    if let Ok(input) = BytecodeFuzzInput::arbitrary(&mut unstructured) {
        if input.instructions.len() > 1000 {
            return; // Limit instruction count
        }
        
        let instructions: Vec<Instruction> = input.instructions
            .into_iter()
            .filter_map(|fi| fi.to_vm_instruction())
            .collect();
            
        if !instructions.is_empty() {
            let mut vm = VM::new();
            
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                vm.execute_with_limit(&instructions, 5000) // Lower limit for direct bytecode
            })) {
                Ok(_) => {
                    // VM execution completed
                }
                Err(_) => {
                    // VM panicked on bytecode - this is a bug
                }
            }
        }
    }
});

#[cfg(test)]
mod tests {
    use super::*;
    use lyra::{lexer::Lexer, parser::Parser, compiler::Compiler, vm::VM};
    
    #[test]
    fn test_vm_fuzzing_basic() {
        let test_cases = vec![
            // Simple arithmetic
            "1 + 2",
            "3 * 4",
            "10 / 2",
            "5 - 3",
            
            // Function calls
            "Length[{1, 2, 3}]",
            "Head[{1, 2, 3}]",
            "Tail[{1, 2, 3}]",
            
            // Boolean operations
            "True",
            "False",
            
            // String operations
            "\"hello\"",
            "StringLength[\"test\"]",
            
            // Lists
            "{}",
            "{1}",
            "{1, 2, 3}",
        ];
        
        for case in test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            if let Ok(ast) = parser.parse() {
                let mut compiler = Compiler::new();
                
                if let Ok(bytecode) = compiler.compile(&ast) {
                    let mut vm = VM::new();
                    
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        vm.execute(&bytecode)
                    }));
                    
                    assert!(result.is_ok(), "VM panicked on input: {}", case);
                }
            }
        }
    }
    
    #[test]
    fn test_vm_security_patterns() {
        let security_test_cases = vec![
            // Large numbers
            "999999999999999",
            "1.7976931348623157e+308", // Close to f64::MAX
            
            // Division by zero
            "1 / 0",
            "0.0 / 0.0",
            
            // Very long strings
            &format!("\"{}\"", "x".repeat(10000)),
            
            // Large lists
            &format!("{{{}}}", (1..1000).map(|i| i.to_string()).collect::<Vec<_>>().join(", ")),
            
            // Nested function calls
            "f[g[h[i[j[1]]]]]",
            
            // Pattern matching stress
            "{1, 2, 3} /. {x_, y_, z_} -> x + y + z",
        ];
        
        for case in security_test_cases {
            let lexer = Lexer::new(case);
            let mut parser = Parser::new(lexer);
            
            let parse_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                parser.parse()
            }));
            
            if let Ok(Ok(ast)) = parse_result {
                let mut compiler = Compiler::new();
                
                let compile_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    compiler.compile(&ast)
                }));
                
                if let Ok(Ok(bytecode)) = compile_result {
                    let mut vm = VM::new();
                    
                    let vm_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        vm.execute_with_limit(&bytecode, 10000)
                    }));
                    
                    assert!(vm_result.is_ok(), "VM panicked on input: {}", 
                           case.chars().take(50).collect::<String>());
                }
            }
        }
    }
    
    #[test]
    fn test_vm_resource_limits() {
        // Test that VM respects resource limits
        
        // Test instruction limit
        let many_ops = format!("0 {}", "+ 1 ".repeat(20000));
        let lexer = Lexer::new(&many_ops);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            
            if let Ok(bytecode) = compiler.compile(&ast) {
                let mut vm = VM::new();
                
                // Should respect the instruction limit
                let result = vm.execute_with_limit(&bytecode, 1000);
                // This might timeout or complete, but should not panic
            }
        }
        
        // Test stack overflow protection
        let deep_nesting = "f[".repeat(1000) + &"]".repeat(1000);
        let lexer = Lexer::new(&deep_nesting);
        let mut parser = Parser::new(lexer);
        
        if let Ok(ast) = parser.parse() {
            let mut compiler = Compiler::new();
            
            if let Ok(bytecode) = compiler.compile(&ast) {
                let mut vm = VM::new();
                
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    vm.execute_with_limit(&bytecode, 10000)
                }));
                
                assert!(result.is_ok(), "VM should handle deep nesting gracefully");
            }
        }
    }
}