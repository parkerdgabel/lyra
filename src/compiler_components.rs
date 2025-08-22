// Compiler Components Module
// This module contains extracted components from the compiler for better organization

use crate::ast::Expr;
use crate::bytecode::{Instruction, OpCode};
use crate::compiler::{CompilerError, CompilerResult, EnhancedFunctionSignature};
use std::collections::HashMap;

/// Attribute processor responsible for handling function attributes like Hold, Listable, etc.
/// 
/// This component handles:
/// - Attribute parsing and validation
/// - Attribute-specific code generation
/// - Listable function transformations
/// - Hold expression handling
/// - Protected function enforcement
#[derive(Debug)]
pub struct AttributeProcessor {
    /// Cached attribute information for efficient lookup
    pub attribute_cache: HashMap<String, FunctionAttributes>,
}

/// Function attributes that affect compilation and execution
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionAttributes {
    pub hold: Vec<usize>,      // Which arguments to hold (0-indexed)
    pub listable: bool,        // Whether function is listable (threads over lists)
    pub orderless: bool,       // Whether argument order doesn't matter
    pub protected: bool,       // Whether function is protected from modification
    pub flat: bool,            // Whether function flattens its arguments
    pub one_identity: bool,    // Whether f[x] == x for single arguments
}

impl FunctionAttributes {
    pub fn new() -> Self {
        FunctionAttributes {
            hold: Vec::new(),
            listable: false,
            orderless: false,
            protected: false,
            flat: false,
            one_identity: false,
        }
    }
    
    pub fn with_listable(mut self) -> Self {
        self.listable = true;
        self
    }
    
    pub fn with_hold(mut self, positions: Vec<usize>) -> Self {
        self.hold = positions;
        self
    }
    
    pub fn with_protected(mut self) -> Self {
        self.protected = true;
        self
    }
}

impl Default for FunctionAttributes {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributeProcessor {
    /// Create a new AttributeProcessor
    pub fn new() -> Self {
        let mut processor = AttributeProcessor {
            attribute_cache: HashMap::new(),
        };
        
        // Initialize with stdlib function attributes
        processor.initialize_stdlib_attributes();
        processor
    }
    
    /// Initialize standard library function attributes
    fn initialize_stdlib_attributes(&mut self) {
        // Mathematical functions with listable attribute
        let listable_functions = vec![
            "Plus", "Times", "Minus", "Divide", "Power",
            "Sin", "Cos", "Tan", "Exp", "Log", "Sqrt",
            "Abs", "Floor", "Ceiling", "Round"
        ];
        
        for func in listable_functions {
            self.attribute_cache.insert(
                func.to_string(),
                FunctionAttributes::new().with_listable()
            );
        }
        
        // Functions with hold attributes
        self.attribute_cache.insert(
            "Hold".to_string(),
            FunctionAttributes::new().with_hold(vec![0])
        );
        
        self.attribute_cache.insert(
            "HoldAll".to_string(),
            FunctionAttributes::new().with_hold(vec![0, 1, 2, 3, 4]) // Hold first 5 args
        );
        
        // Protected functions
        let protected_functions = vec![
            "List", "Plus", "Times", "Power", "Equal", "Set", "SetDelayed"
        ];
        
        for func in protected_functions {
            if let Some(attrs) = self.attribute_cache.get_mut(func) {
                attrs.protected = true;
            } else {
                self.attribute_cache.insert(
                    func.to_string(),
                    FunctionAttributes::new().with_protected()
                );
            }
        }
    }
    
    /// Get attributes for a function
    pub fn get_attributes(&self, function_name: &str) -> Option<&FunctionAttributes> {
        self.attribute_cache.get(function_name)
    }
    
    /// Register custom attributes for a function
    pub fn register_attributes(&mut self, function_name: String, attributes: FunctionAttributes) {
        self.attribute_cache.insert(function_name, attributes);
    }
    
    /// Check if a function has a specific attribute
    pub fn has_attribute(&self, function_name: &str, attribute: &str) -> bool {
        if let Some(attrs) = self.get_attributes(function_name) {
            match attribute {
                "Listable" => attrs.listable,
                "Protected" => attrs.protected,
                "Orderless" => attrs.orderless,
                "Flat" => attrs.flat,
                "OneIdentity" => attrs.one_identity,
                "Hold" => !attrs.hold.is_empty(),
                _ => false,
            }
        } else {
            false
        }
    }
    
    /// Generate appropriate instruction based on function attributes
    pub fn generate_call_instruction(&self, function_name: &str, _arg_count: usize) -> OpCode {
        if let Some(attrs) = self.get_attributes(function_name) {
            if attrs.listable {
                // Use MAP_CallStatic for listable functions
                OpCode::MapCallStatic
            } else {
                // Use regular CallStatic for non-listable functions
                OpCode::CallStatic
            }
        } else {
            // Default to regular call for unknown functions
            OpCode::CallStatic
        }
    }
    
    /// Check if any arguments should be held (not evaluated)
    pub fn should_hold_argument(&self, function_name: &str, arg_index: usize) -> bool {
        if let Some(attrs) = self.get_attributes(function_name) {
            attrs.hold.contains(&arg_index)
        } else {
            false
        }
    }
}

impl Default for AttributeProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Type inference engine for analyzing and inferring types in expressions
/// 
/// This component handles:
/// - Expression type inference
/// - Type compatibility checking
/// - Generic type resolution
/// - Type error detection and reporting
#[derive(Debug)]
pub struct TypeInference {
    /// Type inference cache for performance
    pub inference_cache: HashMap<String, String>,
    /// Known type signatures for user functions
    pub function_signatures: HashMap<String, EnhancedFunctionSignature>,
}

impl TypeInference {
    /// Create a new TypeInference engine
    pub fn new() -> Self {
        TypeInference {
            inference_cache: HashMap::new(),
            function_signatures: HashMap::new(),
        }
    }
    
    /// Register a function signature for type inference
    pub fn register_function_signature(&mut self, signature: EnhancedFunctionSignature) {
        self.function_signatures.insert(signature.name.clone(), signature);
    }
    
    /// Infer the type of an expression
    pub fn infer_expression_type(&self, expr: &Expr) -> Option<String> {
        match expr {
            Expr::Number(num) => match num {
                crate::ast::Number::Integer(_) => Some("Integer".to_string()),
                crate::ast::Number::Real(_) => Some("Real".to_string()),
            },
            Expr::String(_) => Some("String".to_string()),
            Expr::Symbol(_) => Some("Symbol".to_string()),
            Expr::List(elements) => {
                if elements.is_empty() {
                    Some("List".to_string())
                } else {
                    // Try to infer element type for homogeneous lists
                    let first_type = self.infer_expression_type(&elements[0])?;
                    let all_same_type = elements.iter()
                        .all(|elem| self.infer_expression_type(elem) == Some(first_type.clone()));
                    
                    if all_same_type {
                        Some(format!("List[{}]", first_type))
                    } else {
                        Some("List".to_string())
                    }
                }
            },
            Expr::Function { head, args } => {
                if let Expr::Symbol(func_symbol) = head.as_ref() {
                    self.infer_function_call_type(&func_symbol.name, args)
                } else {
                    None
                }
            },
            _ => None,
        }
    }
    
    /// Infer the return type of a function call
    pub fn infer_function_call_type(&self, function_name: &str, args: &[Expr]) -> Option<String> {
        // Check user-defined functions first
        if let Some(signature) = self.function_signatures.get(function_name) {
            return signature.return_type.clone();
        }
        
        // Handle built-in functions
        match function_name {
            "Plus" | "Times" | "Minus" | "Divide" => {
                // Arithmetic operations preserve or promote types
                if args.len() == 2 {
                    let left_type = self.infer_expression_type(&args[0])?;
                    let right_type = self.infer_expression_type(&args[1])?;
                    
                    match (left_type.as_str(), right_type.as_str()) {
                        ("Integer", "Integer") => Some("Integer".to_string()),
                        ("Real", _) | (_, "Real") => Some("Real".to_string()),
                        _ => Some("Real".to_string()), // Default to Real for mixed arithmetic
                    }
                } else {
                    None
                }
            },
            "Power" => Some("Real".to_string()), // Power usually returns Real
            "Length" => Some("Integer".to_string()),
            "Head" | "Tail" | "First" | "Last" => {
                // These functions return elements from lists
                if let Some(arg_type) = self.infer_expression_type(&args[0]) {
                    if arg_type.starts_with("List[") && arg_type.ends_with(']') {
                        // Extract element type from List[T]
                        let element_type = &arg_type[5..arg_type.len()-1];
                        Some(element_type.to_string())
                    } else {
                        Some("Object".to_string())
                    }
                } else {
                    None
                }
            },
            "Sin" | "Cos" | "Tan" | "Exp" | "Log" | "Sqrt" => Some("Real".to_string()),
            "Equal" | "Less" | "Greater" => Some("Boolean".to_string()),
            _ => None,
        }
    }
    
    /// Validate that arguments match expected parameter types
    pub fn validate_argument_types(&self, function_name: &str, args: &[Expr]) -> CompilerResult<()> {
        if let Some(signature) = self.function_signatures.get(function_name) {
            // Check arity
            if signature.params.len() != args.len() {
                return Err(CompilerError::InvalidArity {
                    function: function_name.to_string(),
                    expected: signature.params.len(),
                    actual: args.len(),
                });
            }
            
            // Check each parameter type
            for (i, (param_name, expected_type_opt)) in signature.params.iter().enumerate() {
                if let Some(expected_type) = expected_type_opt {
                    if let Some(actual_type) = self.infer_expression_type(&args[i]) {
                        if !self.is_type_compatible(&actual_type, expected_type) {
                            return Err(CompilerError::UnsupportedExpression(format!(
                                "Type mismatch for parameter {}: expected {}, got {}",
                                param_name, expected_type, actual_type
                            )));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Check if two types are compatible (including coercion)
    pub fn is_type_compatible(&self, actual: &str, expected: &str) -> bool {
        if actual == expected {
            return true;
        }
        
        // Type coercion rules
        match (actual, expected) {
            ("Integer", "Real") => true,
            ("List", list_type) if list_type.starts_with("List[") => true,
            _ => false,
        }
    }
    
    /// Clear the inference cache
    pub fn clear_cache(&mut self) {
        self.inference_cache.clear();
    }
}

impl Default for TypeInference {
    fn default() -> Self {
        Self::new()
    }
}

/// Core bytecode generator for expressions
/// 
/// This component handles:
/// - Basic expression compilation to bytecode
/// - Instruction emission and optimization
/// - Constant pool management
/// - Symbol table management during compilation
#[derive(Debug)]
pub struct CodeGenerator {
    /// Generated bytecode instructions
    pub code: Vec<Instruction>,
    /// Current position in code for jumps
    pub position: usize,
}

impl CodeGenerator {
    /// Create a new CodeGenerator
    pub fn new() -> Self {
        CodeGenerator {
            code: Vec::new(),
            position: 0,
        }
    }
    
    /// Emit an instruction
    pub fn emit(&mut self, opcode: OpCode, operand: u32) -> CompilerResult<()> {
        let instruction = Instruction::new(opcode, operand)
            .map_err(|_| CompilerError::UnsupportedExpression("Invalid instruction".to_string()))?;
        self.code.push(instruction);
        self.position += 1;
        Ok(())
    }
    
    /// Get the current code position (for jumps)
    pub fn current_position(&self) -> usize {
        self.position
    }
    
    /// Emit a jump instruction and return the position for patching
    pub fn emit_jump(&mut self, opcode: OpCode) -> CompilerResult<usize> {
        let jump_pos = self.current_position();
        self.emit(opcode, 0)?; // Placeholder operand
        Ok(jump_pos)
    }
    
    /// Patch a jump instruction with the target address
    pub fn patch_jump(&mut self, jump_pos: usize, target: usize) -> CompilerResult<()> {
        if jump_pos < self.code.len() {
            let instruction = Instruction::new(self.code[jump_pos].opcode, target as u32)
                .map_err(|_| CompilerError::UnsupportedExpression("Invalid jump target".to_string()))?;
            self.code[jump_pos] = instruction;
            Ok(())
        } else {
            Err(CompilerError::UnsupportedExpression("Invalid jump position".to_string()))
        }
    }
    
    /// Get the generated code
    pub fn code(&self) -> &[Instruction] {
        &self.code
    }
    
    /// Take ownership of the generated code
    pub fn take_code(self) -> Vec<Instruction> {
        self.code
    }
    
    /// Clear the generated code
    pub fn clear(&mut self) {
        self.code.clear();
        self.position = 0;
    }
    
    /// Optimize the generated bytecode (placeholder for future optimizations)
    pub fn optimize(&mut self) {
        // Future optimizations:
        // - Dead code elimination
        // - Constant folding
        // - Jump optimization
        // - Instruction combining
    }
}

impl Default for CodeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Number, Symbol};

    #[test]
    fn test_attribute_processor() {
        let processor = AttributeProcessor::new();
        
        // Test listable attributes
        assert!(processor.has_attribute("Plus", "Listable"));
        assert!(processor.has_attribute("Sin", "Listable"));
        assert!(!processor.has_attribute("Length", "Listable"));
        
        // Test protected attributes
        assert!(processor.has_attribute("Plus", "Protected"));
        assert!(processor.has_attribute("List", "Protected"));
        
        // Test call instruction generation
        assert_eq!(processor.generate_call_instruction("Plus", 2), OpCode::MapCallStatic);
        assert_eq!(processor.generate_call_instruction("Length", 1), OpCode::CallStatic);
    }
    
    #[test]
    fn test_type_inference() {
        let inference = TypeInference::new();
        
        // Test basic type inference
        assert_eq!(
            inference.infer_expression_type(&Expr::Number(Number::Integer(42))),
            Some("Integer".to_string())
        );
        assert_eq!(
            inference.infer_expression_type(&Expr::Number(Number::Real(3.14))),
            Some("Real".to_string())
        );
        assert_eq!(
            inference.infer_expression_type(&Expr::String("test".to_string())),
            Some("String".to_string())
        );
        
        // Test list type inference
        let int_list = Expr::List(vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
        ]);
        assert_eq!(
            inference.infer_expression_type(&int_list),
            Some("List[Integer]".to_string())
        );
        
        // Test function call type inference
        let plus_call = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(1)),
                Expr::Number(Number::Integer(2)),
            ],
        };
        assert_eq!(
            inference.infer_expression_type(&plus_call),
            Some("Integer".to_string())
        );
    }
    
    #[test]
    fn test_code_generator() {
        let mut generator = CodeGenerator::new();
        
        // Test basic instruction emission
        generator.emit(OpCode::LDC, 42).unwrap();
        generator.emit(OpCode::LDC, 24).unwrap();
        generator.emit(OpCode::ADD, 0).unwrap();
        
        assert_eq!(generator.code.len(), 3);
        assert_eq!(generator.current_position(), 3);
        
        // Test jump emission and patching
        let jump_pos = generator.emit_jump(OpCode::JMP).unwrap();
        generator.emit(OpCode::LDC, 1).unwrap();
        generator.patch_jump(jump_pos, generator.current_position()).unwrap();
        
        assert_eq!(generator.code[jump_pos].operand, generator.current_position() as u32);
    }
    
    #[test]
    fn test_type_compatibility() {
        let inference = TypeInference::new();
        
        assert!(inference.is_type_compatible("Integer", "Integer"));
        assert!(inference.is_type_compatible("Integer", "Real"));
        assert!(!inference.is_type_compatible("Real", "Integer"));
        assert!(inference.is_type_compatible("List", "List[Integer]"));
        assert!(!inference.is_type_compatible("String", "Integer"));
    }
    
    #[test]
    fn test_function_attributes() {
        let attrs = FunctionAttributes::new()
            .with_listable()
            .with_protected()
            .with_hold(vec![0, 1]);
        
        assert!(attrs.listable);
        assert!(attrs.protected);
        assert_eq!(attrs.hold, vec![0, 1]);
        assert!(!attrs.orderless);
    }
}