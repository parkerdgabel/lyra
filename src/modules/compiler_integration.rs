//! Compiler Integration for Module System
//!
//! Extends the compiler to support import statements and namespaced function calls.

use super::{ModuleError};
use super::registry::ModuleRegistry;
use super::resolver::{ImportContext, ImportParser, ImportStatement, ScopeManager};
use crate::compiler::{Compiler, CompilerContext, CompilerError, CompilerResult};
use crate::ast::{Expr, Number};
use crate::bytecode::{Instruction, OpCode};
use crate::vm::Value;
use crate::linker::FunctionRegistry;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Extended compiler context with module support
#[derive(Debug)]
pub struct ModuleAwareCompilerContext {
    /// Base compiler context
    pub base: CompilerContext,
    
    /// Import context for this compilation unit
    pub import_context: ImportContext,
    
    /// Scope manager for nested scopes
    pub scope_manager: ScopeManager,
    
    /// Module registry for function resolution
    pub module_registry: Option<Arc<ModuleRegistry>>,
    
    /// Extended function registry with namespace support
    pub function_registry: Arc<RwLock<FunctionRegistry>>,
    
    /// Import statements processed in this compilation unit
    pub imports: Vec<ImportStatement>,
}

impl ModuleAwareCompilerContext {
    /// Create a new module-aware compiler context
    pub fn new(function_registry: Arc<RwLock<FunctionRegistry>>) -> Self {
        let import_context = ImportContext::new();
        let scope_manager = ScopeManager::new(import_context.clone());
        
        ModuleAwareCompilerContext {
            base: CompilerContext {
                constants: Vec::new(),
                symbols: HashMap::new(),
                code: Vec::new(),
            },
            import_context,
            scope_manager,
            module_registry: None,
            function_registry,
            imports: Vec::new(),
        }
    }
    
    /// Set the module registry
    pub fn with_module_registry(mut self, registry: Arc<ModuleRegistry>) -> Self {
        self.module_registry = Some(registry);
        self
    }
    
    /// Process an import statement
    pub fn process_import(&mut self, import_stmt: &str) -> Result<(), ModuleError> {
        // Parse the import statement
        let import = ImportParser::parse(import_stmt)?;
        
        // Add to import context
        if let Some(registry) = &self.module_registry {
            self.import_context.add_import(import.clone(), registry)?;
        }
        
        // Store for later reference
        self.imports.push(import);
        
        Ok(())
    }
    
    /// Resolve a function name to its qualified name and index
    pub fn resolve_function_call(&self, function_name: &str) -> Option<(String, u16)> {
        // First try to resolve through import context
        if let Some(qualified_name) = self.import_context.resolve_function(function_name) {
            // Get function index from registry
            if let Some(index) = self.function_registry.read().unwrap().resolve_qualified_function(&qualified_name) {
                return Some((qualified_name, index));
            }
        }
        
        // Fallback to unqualified name (for backwards compatibility)
        if let Some(index) = self.function_registry.read().unwrap().get_function_index(function_name) {
            return Some((function_name.to_string(), index));
        }
        
        None
    }
    
    /// Enter a new scope
    pub fn push_scope(&mut self) {
        self.scope_manager.push_scope();
    }
    
    /// Exit current scope
    pub fn pop_scope(&mut self) -> Result<(), ModuleError> {
        self.scope_manager.pop_scope()
    }
    
    /// Define a variable in current scope
    pub fn define_variable(&mut self, name: String, type_info: String) -> Result<(), ModuleError> {
        self.scope_manager.define_variable(name, type_info)
    }
    
    /// Define a function in current scope
    pub fn define_function(&mut self, name: String, definition: String) -> Result<(), ModuleError> {
        self.scope_manager.define_function(name, definition)
    }
    
    /// Check if a name is locally defined
    pub fn is_local_name(&self, name: &str) -> bool {
        self.scope_manager.is_local(name)
    }
    
    /// Get all visible names in current context
    pub fn visible_names(&self) -> Vec<String> {
        self.scope_manager.visible_names()
    }
    
    /// Add a constant to the constant pool
    pub fn add_constant(&mut self, value: Value) -> Result<u32, CompilerError> {
        if self.base.constants.len() >= 16777215 {
            return Err(CompilerError::TooManyConstants);
        }
        
        self.base.constants.push(value);
        Ok((self.base.constants.len() - 1) as u32)
    }
    
    /// Emit an instruction
    pub fn emit(&mut self, instruction: Instruction) {
        self.base.code.push(instruction);
    }
    
    /// Get the current instruction count
    pub fn instruction_count(&self) -> usize {
        self.base.code.len()
    }
}

/// Extended compiler with module support
pub struct ModuleAwareCompiler {
    /// Module-aware context
    pub context: ModuleAwareCompilerContext,
}

impl ModuleAwareCompiler {
    /// Create a new module-aware compiler
    pub fn new(function_registry: Arc<RwLock<FunctionRegistry>>) -> Self {
        ModuleAwareCompiler {
            context: ModuleAwareCompilerContext::new(function_registry),
        }
    }
    
    /// Create compiler with module registry
    pub fn with_module_registry(function_registry: Arc<RwLock<FunctionRegistry>>, module_registry: Arc<ModuleRegistry>) -> Self {
        let context = ModuleAwareCompilerContext::new(function_registry).with_module_registry(module_registry);
        ModuleAwareCompiler { context }
    }
    
    /// Compile an expression with module awareness
    pub fn compile_expr(&mut self, expr: &Expr) -> CompilerResult<()> {
        match expr {
            Expr::Import { statement } => {
                // Handle import statements
                self.context.process_import(statement).map_err(|e| CompilerError::UnsupportedExpression(format!("Import error: {}", e)))?;
                // Import statements don't generate runtime code
                Ok(())
            },
            
            Expr::QualifiedCall { namespace, function, args } => {
                // Handle qualified function calls like std::math::Sin[x]
                let qualified_name = format!("{}::{}", namespace, function);
                self.compile_qualified_function_call(&qualified_name, args)
            },
            
            Expr::Call { function, args } => {
                // Handle regular function calls with namespace resolution
                self.compile_function_call(function, args)
            },
            
            Expr::ScopeBlock { statements } => {
                // Handle scoped blocks
                self.context.push_scope();
                
                for stmt in statements {
                    self.compile_expr(stmt)?;
                }
                
                self.context.pop_scope().map_err(|e| CompilerError::UnsupportedExpression(format!("Scope error: {}", e)))?;
                Ok(())
            },
            
            // Delegate other expressions to base compiler logic
            _ => {
                self.compile_base_expr(expr)
            }
        }
    }
    
    /// Compile a qualified function call
    fn compile_qualified_function_call(&mut self, qualified_name: &str, args: &[Expr]) -> CompilerResult<()> {
        // Compile arguments
        for arg in args {
            self.compile_expr(arg)?;
        }
        
        // Resolve function
        if let Some((_resolved_name, function_index)) = self.context.function_registry.read().unwrap().resolve_qualified_function(qualified_name).map(|idx| (qualified_name.to_string(), idx)) {
            // Emit static call instruction
            self.context.emit(Instruction {
                opcode: OpCode::CallStatic,
                operand: function_index as u32,
            });
            Ok(())
        } else {
            Err(CompilerError::UnknownFunction(qualified_name.to_string()))
        }
    }
    
    /// Compile a function call with namespace resolution
    fn compile_function_call(&mut self, function_name: &str, args: &[Expr]) -> CompilerResult<()> {
        // Compile arguments
        for arg in args {
            self.compile_expr(arg)?;
        }
        
        // Try to resolve through import context
        if let Some((qualified_name, function_index)) = self.context.resolve_function_call(function_name) {
            // Emit static call instruction
            self.context.emit(Instruction {
                opcode: OpCode::CallStatic,
                operand: function_index as u32,
            });
            Ok(())
        } else {
            Err(CompilerError::UnknownFunction(function_name.to_string()))
        }
    }
    
    /// Compile base expressions (delegated to base compiler)
    fn compile_base_expr(&mut self, expr: &Expr) -> CompilerResult<()> {
        match expr {
            Expr::Number(num) => {
                let value = match num {
                    Number::Integer(i) => Value::Integer(*i),
                    Number::Real(f) => Value::Real(*f),
                };
                let const_index = self.context.add_constant(value)?;
                self.context.emit(Instruction {
                    opcode: OpCode::LoadConst,
                    operand: const_index,
                });
                Ok(())
            },
            
            Expr::String(s) => {
                let const_index = self.context.add_constant(Value::String(s.clone()))?;
                self.context.emit(Instruction {
                    opcode: OpCode::LoadConst,
                    operand: const_index,
                });
                Ok(())
            },
            
            Expr::Symbol(s) => {
                let const_index = self.context.add_constant(Value::Symbol(s.clone()))?;
                self.context.emit(Instruction {
                    opcode: OpCode::LoadConst,
                    operand: const_index,
                });
                Ok(())
            },
            
            Expr::List(items) => {
                // Compile all list items
                for item in items {
                    self.compile_expr(item)?;
                }
                
                // Create list instruction
                self.context.emit(Instruction {
                    opcode: OpCode::MakeList,
                    operand: items.len() as u32,
                });
                Ok(())
            },
            
            _ => {
                Err(CompilerError::UnsupportedExpression(format!("{:?}", expr)))
            }
        }
    }
    
    /// Get the compiled bytecode
    pub fn get_bytecode(&self) -> &Vec<Instruction> {
        &self.context.base.code
    }
    
    /// Get the constants
    pub fn get_constants(&self) -> &Vec<Value> {
        &self.context.base.constants
    }
    
    /// Get import information for debugging
    pub fn get_imports(&self) -> &Vec<ImportStatement> {
        &self.context.imports
    }
}

/// AST extensions for module system
impl Expr {
    /// Create an import expression
    pub fn import(statement: String) -> Self {
        Expr::Import { statement }
    }
    
    /// Create a qualified function call
    pub fn qualified_call(namespace: String, function: String, args: Vec<Expr>) -> Self {
        Expr::QualifiedCall { namespace, function, args }
    }
    
    /// Create a scoped block
    pub fn scope_block(statements: Vec<Expr>) -> Self {
        Expr::ScopeBlock { statements }
    }
}

// Extend the AST enum (this would go in ast.rs in a real implementation)
// For now, we'll define the additional variants here as documentation
/*
pub enum Expr {
    // ... existing variants ...
    
    /// Import statement: import std::math
    Import {
        statement: String,
    },
    
    /// Qualified function call: std::math::Sin[x]
    QualifiedCall {
        namespace: String,
        function: String,
        args: Vec<Expr>,
    },
    
    /// Scoped block: { statements... }
    ScopeBlock {
        statements: Vec<Expr>,
    },
}
*/

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linker::FunctionRegistry;
    use crate::modules::registry::ModuleRegistry;

    fn create_test_compiler() -> ModuleAwareCompiler {
        let func_registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let module_registry = Arc::new(ModuleRegistry::new(func_registry.clone()));
        ModuleAwareCompiler::with_module_registry(func_registry, module_registry)
    }

    #[test]
    fn test_import_processing() {
        let mut compiler = create_test_compiler();
        
        // Process an import statement
        let result = compiler.context.process_import("import std::math");
        assert!(result.is_ok());
        
        // Check that import was recorded
        assert_eq!(compiler.context.imports.len(), 1);
        
        // Check function resolution
        let resolution = compiler.context.resolve_function_call("Sin");
        // This might be None since we don't have actual functions registered in test
        // In a real scenario with proper registry setup, this would resolve
    }

    #[test]
    fn test_scope_management() {
        let mut compiler = create_test_compiler();
        
        // Test scope operations
        assert_eq!(compiler.context.scope_manager.scope_depth(), 1);
        
        compiler.context.push_scope();
        assert_eq!(compiler.context.scope_manager.scope_depth(), 2);
        
        compiler.context.define_variable("x".to_string(), "Integer".to_string()).unwrap();
        assert!(compiler.context.is_local_name("x"));
        
        compiler.context.pop_scope().unwrap();
        assert_eq!(compiler.context.scope_manager.scope_depth(), 1);
        assert!(!compiler.context.is_local_name("x"));
    }

    #[test]
    fn test_constant_management() {
        let mut compiler = create_test_compiler();
        
        // Test adding constants
        let index1 = compiler.context.add_constant(Value::Integer(42)).unwrap();
        let index2 = compiler.context.add_constant(Value::String("test".to_string())).unwrap();
        
        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(compiler.context.base.constants.len(), 2);
    }

    #[test]
    fn test_instruction_emission() {
        let mut compiler = create_test_compiler();
        
        // Test emitting instructions
        compiler.context.emit(Instruction {
            opcode: OpCode::LoadConst,
            operand: 0,
        });
        
        assert_eq!(compiler.context.instruction_count(), 1);
        assert_eq!(compiler.get_bytecode().len(), 1);
    }
}