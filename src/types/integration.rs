use super::{
    LyraType, TensorShape, TypeError, TypeResult, TypeEnvironment, TypeScheme,
    TypeInferenceEngine, TypeChecker, infer_expression_type, check_expression_safety
};
use crate::{
    ast::{Expr, Number},
    compiler::{Compiler, CompilerError, CompilerResult},
    vm::{Value, VirtualMachine, VmError},
    error::Error,
};
use std::collections::HashMap;

/// Extended compiler with type checking capabilities
#[derive(Debug)]
pub struct TypedCompiler {
    /// Base compiler
    compiler: Compiler,
    /// Type checker
    type_checker: TypeChecker,
    /// Whether to enable compile-time type checking
    type_checking_enabled: bool,
    /// Type annotations for compiled expressions
    type_annotations: HashMap<String, LyraType>,
}

impl TypedCompiler {
    /// Create a new typed compiler
    pub fn new() -> Self {
        TypedCompiler {
            compiler: Compiler::new(),
            type_checker: TypeChecker::new(),
            type_checking_enabled: true,
            type_annotations: HashMap::new(),
        }
    }
    
    /// Create a typed compiler with strict type checking
    pub fn new_strict() -> Self {
        TypedCompiler {
            compiler: Compiler::new(),
            type_checker: TypeChecker::new_strict(),
            type_checking_enabled: true,
            type_annotations: HashMap::new(),
        }
    }
    
    /// Create a typed compiler with type checking disabled
    pub fn new_without_type_checking() -> Self {
        TypedCompiler {
            compiler: Compiler::new(),
            type_checker: TypeChecker::new(),
            type_checking_enabled: false,
            type_annotations: HashMap::new(),
        }
    }
    
    /// Enable or disable type checking
    pub fn set_type_checking(&mut self, enabled: bool) {
        self.type_checking_enabled = enabled;
    }
    
    /// Compile an expression with type checking
    pub fn compile_expr_typed(&mut self, expr: &Expr) -> CompilerResult<LyraType> {
        // First, perform type checking if enabled
        let expr_type = if self.type_checking_enabled {
            let env = TypeEnvironment::new();
            self.type_checker.check_expression(expr, &env)
                .map_err(|type_err| CompilerError::UnsupportedExpression(format!("Type error: {}", type_err)))?
        } else {
            // If type checking is disabled, still infer types for annotations
            infer_expression_type(expr)
                .unwrap_or(LyraType::Unknown)
        };
        
        // Store type annotation
        let expr_key = format!("{:?}", expr);
        self.type_annotations.insert(expr_key, expr_type.clone());
        
        // Compile the expression
        self.compiler.compile_expr(expr)?;
        
        Ok(expr_type)
    }
    
    /// Compile a program with type checking
    pub fn compile_program_typed(&mut self, expressions: &[Expr]) -> CompilerResult<Vec<LyraType>> {
        let mut types = Vec::new();
        
        for expr in expressions {
            let expr_type = self.compile_expr_typed(expr)?;
            types.push(expr_type);
        }
        
        Ok(types)
    }
    
    /// Get the type annotation for an expression
    pub fn get_type_annotation(&self, expr: &Expr) -> Option<&LyraType> {
        let expr_key = format!("{:?}", expr);
        self.type_annotations.get(&expr_key)
    }
    
    /// Create a VM with type information
    pub fn into_typed_vm(self) -> TypedVirtualMachine {
        let vm = self.compiler.into_vm();
        TypedVirtualMachine {
            vm,
            type_annotations: self.type_annotations,
            runtime_type_checking: false,
        }
    }
    
    /// Convenience method to compile and evaluate with type checking
    pub fn eval_typed(expr: &Expr) -> Result<(Value, LyraType), Error> {
        let mut compiler = TypedCompiler::new();
        let expr_type = compiler.compile_expr_typed(expr)
            .map_err(|e| Error::Compilation { message: e.to_string() })?;
        
        let mut vm = compiler.into_typed_vm();
        let value = vm.run()
            .map_err(|e| Error::Runtime { message: e.to_string() })?;
        
        Ok((value, expr_type))
    }
}

impl Default for TypedCompiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Virtual machine with type annotations and runtime type checking
#[derive(Debug)]
pub struct TypedVirtualMachine {
    /// Base VM
    vm: VirtualMachine,
    /// Type annotations from compilation
    type_annotations: HashMap<String, LyraType>,
    /// Whether to perform runtime type checking
    runtime_type_checking: bool,
}

impl TypedVirtualMachine {
    /// Create a new typed VM
    pub fn new() -> Self {
        TypedVirtualMachine {
            vm: VirtualMachine::new(),
            type_annotations: HashMap::new(),
            runtime_type_checking: false,
        }
    }
    
    /// Enable or disable runtime type checking
    pub fn set_runtime_type_checking(&mut self, enabled: bool) {
        self.runtime_type_checking = enabled;
    }
    
    /// Run the VM with optional runtime type checking
    pub fn run(&mut self) -> Result<Value, VmError> {
        if self.runtime_type_checking {
            self.run_with_type_checking()
        } else {
            self.vm.run()
        }
    }
    
    /// Run with runtime type checking
    fn run_with_type_checking(&mut self) -> Result<Value, VmError> {
        // For now, just run normally and validate the result
        let result = self.vm.run()?;
        
        // TODO: Implement runtime type validation
        // This would involve checking that runtime values match their expected types
        
        Ok(result)
    }
    
    /// Get type annotation for a value
    pub fn get_value_type(&self, value: &Value) -> LyraType {
        value_to_type(value)
    }
    
    /// Load bytecode with type annotations
    pub fn load_with_types(&mut self, code: Vec<crate::bytecode::Instruction>, constants: Vec<Value>, types: HashMap<String, LyraType>) {
        self.vm.load(code, constants);
        self.type_annotations = types;
    }
    
    /// Get all type annotations
    pub fn type_annotations(&self) -> &HashMap<String, LyraType> {
        &self.type_annotations
    }
    
    /// Delegate other VM methods
    pub fn push(&mut self, value: Value) {
        self.vm.push(value);
    }
    
    pub fn pop(&mut self) -> Result<Value, VmError> {
        self.vm.pop()
    }
    
    pub fn peek(&self) -> Result<&Value, VmError> {
        self.vm.peek()
    }
}

impl Default for TypedVirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a runtime value to its type
pub fn value_to_type(value: &Value) -> LyraType {
    match value {
        Value::Integer(_) => LyraType::Integer,
        Value::Real(_) => LyraType::Real,
        Value::String(_) => LyraType::String,
        Value::Symbol(_) => LyraType::Symbol,
        Value::Boolean(_) => LyraType::Boolean,
        Value::Missing => LyraType::Unknown,
        Value::List(elements) => {
            if elements.is_empty() {
                LyraType::List(Box::new(LyraType::Unknown))
            } else {
                let first_type = value_to_type(&elements[0]);
                LyraType::List(Box::new(first_type))
            }
        }
        Value::Function(_) => {
            // For now, treat function values as unknown function types
            LyraType::Function {
                params: vec![LyraType::Unknown],
                return_type: Box::new(LyraType::Unknown),
            }
        }
        Value::LyObj(obj) => {
            // Map foreign object types to Lyra types
            match obj.type_name() {
                "Series" => LyraType::List(Box::new(LyraType::Unknown)),
                "Table" => LyraType::List(Box::new(LyraType::List(Box::new(LyraType::Unknown)))),
                "Tensor" => LyraType::Tensor {
                    element_type: Box::new(LyraType::Unknown),
                    shape: None,
                },
                _ => LyraType::Unknown,
            }
        }
        Value::Quote(_) => LyraType::Unknown, // Quoted expressions have unknown type
        Value::Pattern(_) => LyraType::Pattern(Box::new(LyraType::Unknown)),
    }
}

/// Convert a Lyra type to a runtime value (for type representation)
pub fn type_to_value(ty: &LyraType) -> Value {
    match ty {
        LyraType::Integer => Value::Symbol("Integer".to_string()),
        LyraType::Real => Value::Symbol("Real".to_string()),
        LyraType::String => Value::Symbol("String".to_string()),
        LyraType::Boolean => Value::Symbol("Boolean".to_string()),
        LyraType::Symbol => Value::Symbol("Symbol".to_string()),
        LyraType::Unit => Value::Symbol("Unit".to_string()),
        LyraType::Unknown => Value::Symbol("Unknown".to_string()),
        LyraType::List(elem_type) => {
            Value::Function(format!("List[{}]", elem_type))
        }
        LyraType::Tensor { element_type, shape } => {
            if let Some(shape) = shape {
                Value::Function(format!("Tensor[{}, {:?}]", element_type, shape.dimensions))
            } else {
                Value::Function(format!("Tensor[{}]", element_type))
            }
        }
        LyraType::Function { params, return_type } => {
            let param_strs: Vec<String> = params.iter().map(|p| p.to_string()).collect();
            Value::Function(format!("({}) -> {}", param_strs.join(", "), return_type))
        }
        LyraType::Pattern(inner) => {
            Value::Function(format!("Pattern[{}]", inner))
        }
        LyraType::Rule { lhs_type, rhs_type } => {
            Value::Function(format!("Rule[{} -> {}]", lhs_type, rhs_type))
        }
        LyraType::TypeVar(var) => Value::Symbol(format!("α{}", var)),
        LyraType::Error(msg) => Value::String(format!("TypeError: {}", msg)),
    }
}

/// Type-aware expression evaluation
pub fn eval_with_type(expr: &Expr) -> Result<(Value, LyraType), Error> {
    TypedCompiler::eval_typed(expr)
}

/// Check if a value matches a type at runtime
pub fn value_matches_type(value: &Value, expected_type: &LyraType) -> bool {
    let actual_type = value_to_type(value);
    types_compatible(&actual_type, expected_type)
}

/// Check if two types are compatible (allowing for reasonable conversions)
fn types_compatible(actual: &LyraType, expected: &LyraType) -> bool {
    match (actual, expected) {
        // Exact matches
        (a, b) if a == b => true,
        
        // Type variables match anything
        (_, LyraType::TypeVar(_)) | (LyraType::TypeVar(_), _) => true,
        
        // Unknown types match anything
        (LyraType::Unknown, _) | (_, LyraType::Unknown) => true,
        
        // Numeric compatibility
        (LyraType::Integer, LyraType::Real) | (LyraType::Real, LyraType::Integer) => true,
        
        // List compatibility (covariant)
        (LyraType::List(actual_elem), LyraType::List(expected_elem)) => {
            types_compatible(actual_elem, expected_elem)
        }
        
        // Tensor compatibility
        (
            LyraType::Tensor { element_type: actual_elem, shape: actual_shape },
            LyraType::Tensor { element_type: expected_elem, shape: expected_shape },
        ) => {
            types_compatible(actual_elem, expected_elem) &&
            shapes_compatible(actual_shape, expected_shape)
        }
        
        // Function compatibility (contravariant in parameters, covariant in return type)
        (
            LyraType::Function { params: actual_params, return_type: actual_ret },
            LyraType::Function { params: expected_params, return_type: expected_ret },
        ) => {
            actual_params.len() == expected_params.len() &&
            actual_params.iter().zip(expected_params.iter()).all(|(a, e)| types_compatible(e, a)) &&
            types_compatible(actual_ret, expected_ret)
        }
        
        _ => false,
    }
}

/// Check if tensor shapes are compatible
fn shapes_compatible(actual: &Option<TensorShape>, expected: &Option<TensorShape>) -> bool {
    match (actual, expected) {
        (None, _) | (_, None) => true, // Unknown shapes are compatible with anything
        (Some(actual), Some(expected)) => actual.broadcast_compatible(expected),
    }
}

/// Create a type environment with stdlib functions
pub fn create_stdlib_type_environment() -> TypeEnvironment {
    let mut env = TypeEnvironment::new();
    
    // Add built-in type signatures
    // These would typically be loaded from the stdlib module
    
    // Arithmetic functions
    let binary_numeric = TypeScheme::new(
        vec![0],
        LyraType::Function {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
            return_type: Box::new(LyraType::TypeVar(0)),
        }
    );
    
    env.insert("Plus".to_string(), binary_numeric.clone());
    env.insert("Minus".to_string(), binary_numeric.clone());
    env.insert("Times".to_string(), binary_numeric.clone());
    
    // Mathematical functions
    let real_to_real = TypeScheme::monomorphic(LyraType::Function {
        params: vec![LyraType::Real],
        return_type: Box::new(LyraType::Real),
    });
    
    env.insert("Sin".to_string(), real_to_real.clone());
    env.insert("Cos".to_string(), real_to_real.clone());
    env.insert("Sqrt".to_string(), real_to_real);
    
    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Symbol, Expr};

    #[test]
    fn test_typed_compiler() {
        let mut compiler = TypedCompiler::new();
        
        // Test compiling a simple expression
        let expr = Expr::Number(Number::Integer(42));
        let expr_type = compiler.compile_expr_typed(&expr).unwrap();
        assert_eq!(expr_type, LyraType::Integer);
        
        // Test type annotation retrieval
        let annotation = compiler.get_type_annotation(&expr);
        assert_eq!(annotation, Some(&LyraType::Integer));
    }

    #[test]
    fn test_typed_vm() {
        let mut vm = TypedVirtualMachine::new();
        
        // Test basic VM operations
        vm.push(Value::Integer(42));
        let value = vm.pop().unwrap();
        assert_eq!(value, Value::Integer(42));
        
        // Test type inference from value
        let value_type = vm.get_value_type(&value);
        assert_eq!(value_type, LyraType::Integer);
    }

    #[test]
    fn test_value_to_type_conversion() {
        assert_eq!(value_to_type(&Value::Integer(42)), LyraType::Integer);
        assert_eq!(value_to_type(&Value::Real(3.14)), LyraType::Real);
        assert_eq!(value_to_type(&Value::String("hello".to_string())), LyraType::String);
        assert_eq!(value_to_type(&Value::Boolean(true)), LyraType::Boolean);
        
        let list_value = Value::List(vec![Value::Integer(1), Value::Integer(2)]);
        assert_eq!(value_to_type(&list_value), LyraType::List(Box::new(LyraType::Integer)));
    }

    #[test]
    fn test_type_to_value_conversion() {
        assert_eq!(type_to_value(&LyraType::Integer), Value::Symbol("Integer".to_string()));
        assert_eq!(type_to_value(&LyraType::Real), Value::Symbol("Real".to_string()));
        
        let list_type = LyraType::List(Box::new(LyraType::Integer));
        assert_eq!(type_to_value(&list_type), Value::Function("List[Integer]".to_string()));
    }

    #[test]
    fn test_value_type_matching() {
        let value = Value::Integer(42);
        assert!(value_matches_type(&value, &LyraType::Integer));
        assert!(value_matches_type(&value, &LyraType::Real)); // Numeric compatibility
        assert!(!value_matches_type(&value, &LyraType::String));
        
        // Type variables match anything
        assert!(value_matches_type(&value, &LyraType::TypeVar(0)));
        
        // Unknown types match anything
        assert!(value_matches_type(&value, &LyraType::Unknown));
    }

    #[test]
    fn test_type_compatibility() {
        // Exact matches
        assert!(types_compatible(&LyraType::Integer, &LyraType::Integer));
        
        // Numeric compatibility
        assert!(types_compatible(&LyraType::Integer, &LyraType::Real));
        assert!(types_compatible(&LyraType::Real, &LyraType::Integer));
        
        // Type variable compatibility
        assert!(types_compatible(&LyraType::Integer, &LyraType::TypeVar(0)));
        assert!(types_compatible(&LyraType::TypeVar(0), &LyraType::Real));
        
        // List compatibility
        let list_int = LyraType::List(Box::new(LyraType::Integer));
        let list_real = LyraType::List(Box::new(LyraType::Real));
        assert!(types_compatible(&list_int, &list_real));
        
        // Incompatible types
        assert!(!types_compatible(&LyraType::Integer, &LyraType::String));
        assert!(!types_compatible(&LyraType::Boolean, &LyraType::Real));
    }

    #[test]
    fn test_eval_with_type() {
        let expr = Expr::Number(Number::Integer(42));
        let (value, ty) = eval_with_type(&expr).unwrap();
        
        assert_eq!(value, Value::Integer(42));
        assert_eq!(ty, LyraType::Integer);
    }

    #[test]
    fn test_function_type_compatibility() {
        let func1 = LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
        };
        
        let func2 = LyraType::Function {
            params: vec![LyraType::Real], // Contravariant
            return_type: Box::new(LyraType::Real),
        };
        
        // func1 is compatible with func2 (can accept Integer where Real is expected)
        assert!(types_compatible(&func1, &func2));
    }

    #[test]
    fn test_tensor_type_compatibility() {
        let tensor1 = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::vector(10)),
        };
        
        let tensor2 = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::scalar()),
        };
        
        // These should be compatible due to broadcasting
        assert!(types_compatible(&tensor1, &tensor2));
    }

    #[test]
    fn test_stdlib_environment() {
        let env = create_stdlib_type_environment();
        
        // Check that stdlib functions are present
        assert!(env.contains("Plus"));
        assert!(env.contains("Sin"));
        assert!(env.contains("Cos"));
        
        // Check a function signature
        let plus_scheme = env.get("Plus").unwrap();
        assert!(!plus_scheme.quantified_vars.is_empty()); // Should be polymorphic
    }

    #[test]
    fn test_typed_compiler_without_checking() {
        let mut compiler = TypedCompiler::new_without_type_checking();
        
        // Even with type checking disabled, compilation should work
        let expr = Expr::Number(Number::Integer(42));
        let expr_type = compiler.compile_expr_typed(&expr).unwrap();
        
        // Type should still be inferred for annotations
        assert!(matches!(expr_type, LyraType::Integer | LyraType::Unknown));
    }

    #[test]
    fn test_typed_program_compilation() {
        let mut compiler = TypedCompiler::new();
        
        let expressions = vec![
            Expr::Number(Number::Integer(42)),
            Expr::String("hello".to_string()),
        ];
        
        let types = compiler.compile_program_typed(&expressions).unwrap();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0], LyraType::Integer);
        assert_eq!(types[1], LyraType::String);
    }
}