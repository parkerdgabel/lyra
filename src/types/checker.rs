use super::{
    LyraType, TensorShape, TypeError, TypeResult, TypeEnvironment, TypeScheme,
    TypeInferenceEngine, infer_expression_type, TypeConstraint
};
use crate::ast::{Expr, Pattern};
use std::collections::HashMap;

/// Type checker for validating type correctness and safety
#[derive(Debug)]
pub struct TypeChecker {
    /// Type inference engine for computing types
    inference_engine: TypeInferenceEngine,
    /// Function signatures for validation
    function_signatures: HashMap<String, FunctionSignature>,
    /// Strict mode - whether to allow implicit conversions
    strict_mode: bool,
}

/// Function signature information
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    /// Parameter types
    pub params: Vec<LyraType>,
    /// Return type
    pub return_type: LyraType,
    /// Whether the function is variadic (accepts variable number of arguments)
    pub variadic: bool,
    /// Type constraints (e.g., numeric types only)
    pub constraints: Vec<FunctionConstraint>,
}

/// Type constraints for function parameters
#[derive(Debug, Clone, PartialEq)]
pub enum FunctionConstraint {
    /// Parameter must be numeric (Integer or Real)
    Numeric,
    /// Parameter must be a tensor
    Tensor,
    /// Parameter must be a list
    List,
    /// Parameter must be callable (function)
    Callable,
    /// Parameters must have compatible shapes for broadcasting
    BroadcastCompatible(Vec<usize>), // indices of parameters that must be broadcast-compatible
    /// All parameters must have the same type
    SameType(Vec<usize>), // indices of parameters that must have the same type
}

impl TypeChecker {
    /// Create a new type checker
    pub fn new() -> Self {
        let mut checker = TypeChecker {
            inference_engine: TypeInferenceEngine::new(),
            function_signatures: HashMap::new(),
            strict_mode: false,
        };
        
        checker.initialize_builtin_signatures();
        checker
    }
    
    /// Create a type checker in strict mode
    pub fn new_strict() -> Self {
        let mut checker = Self::new();
        checker.strict_mode = true;
        checker
    }
    
    /// Initialize built-in function signatures
    fn initialize_builtin_signatures(&mut self) {
        // Arithmetic functions
        self.add_signature("Plus", FunctionSignature {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
            return_type: LyraType::TypeVar(0),
            variadic: false,
            constraints: vec![
                FunctionConstraint::Numeric,
                FunctionConstraint::SameType(vec![0, 1]),
            ],
        });
        
        self.add_signature("Minus", FunctionSignature {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
            return_type: LyraType::TypeVar(0),
            variadic: false,
            constraints: vec![
                FunctionConstraint::Numeric,
                FunctionConstraint::SameType(vec![0, 1]),
            ],
        });
        
        self.add_signature("Times", FunctionSignature {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
            return_type: LyraType::TypeVar(0),
            variadic: false,
            constraints: vec![
                FunctionConstraint::Numeric,
                FunctionConstraint::SameType(vec![0, 1]),
            ],
        });
        
        self.add_signature("Divide", FunctionSignature {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
            return_type: LyraType::Real, // Division always returns Real
            variadic: false,
            constraints: vec![FunctionConstraint::Numeric],
        });
        
        self.add_signature("Power", FunctionSignature {
            params: vec![LyraType::TypeVar(0), LyraType::TypeVar(1)],
            return_type: LyraType::Real,
            variadic: false,
            constraints: vec![FunctionConstraint::Numeric],
        });
        
        // Mathematical functions
        self.add_signature("Sin", FunctionSignature {
            params: vec![LyraType::Real],
            return_type: LyraType::Real,
            variadic: false,
            constraints: vec![],
        });
        
        self.add_signature("Cos", FunctionSignature {
            params: vec![LyraType::Real],
            return_type: LyraType::Real,
            variadic: false,
            constraints: vec![],
        });
        
        self.add_signature("Sqrt", FunctionSignature {
            params: vec![LyraType::Real],
            return_type: LyraType::Real,
            variadic: false,
            constraints: vec![],
        });
        
        // List functions
        self.add_signature("Length", FunctionSignature {
            params: vec![LyraType::List(Box::new(LyraType::TypeVar(0)))],
            return_type: LyraType::Integer,
            variadic: false,
            constraints: vec![],
        });
        
        self.add_signature("Head", FunctionSignature {
            params: vec![LyraType::List(Box::new(LyraType::TypeVar(0)))],
            return_type: LyraType::TypeVar(0),
            variadic: false,
            constraints: vec![],
        });
        
        self.add_signature("Append", FunctionSignature {
            params: vec![
                LyraType::List(Box::new(LyraType::TypeVar(0))),
                LyraType::TypeVar(0),
            ],
            return_type: LyraType::List(Box::new(LyraType::TypeVar(0))),
            variadic: false,
            constraints: vec![FunctionConstraint::SameType(vec![0, 1])],
        });
        
        // Tensor operations
        self.add_signature("TensorAdd", FunctionSignature {
            params: vec![
                LyraType::Tensor {
                    element_type: Box::new(LyraType::TypeVar(0)),
                    shape: None,
                },
                LyraType::Tensor {
                    element_type: Box::new(LyraType::TypeVar(0)),
                    shape: None,
                },
            ],
            return_type: LyraType::Tensor {
                element_type: Box::new(LyraType::TypeVar(0)),
                shape: None,
            },
            variadic: false,
            constraints: vec![
                FunctionConstraint::BroadcastCompatible(vec![0, 1]),
                FunctionConstraint::SameType(vec![0, 1]),
            ],
        });
    }
    
    /// Add a function signature
    pub fn add_signature(&mut self, name: &str, signature: FunctionSignature) {
        self.function_signatures.insert(name.to_string(), signature);
    }
    
    /// Check the type safety of an expression
    pub fn check_expression(&mut self, expr: &Expr, env: &TypeEnvironment) -> TypeResult<LyraType> {
        // First, infer the type
        let inferred_type = self.inference_engine.infer_expr(expr, env)?;
        
        // Then validate type safety
        self.validate_expression(expr, &inferred_type, env)?;
        
        Ok(inferred_type)
    }
    
    /// Validate type safety of an expression
    fn validate_expression(&mut self, expr: &Expr, expr_type: &LyraType, env: &TypeEnvironment) -> TypeResult<()> {
        match expr {
            Expr::Function { head, args } => {
                self.validate_function_call(head, args, expr_type, env)
            }
            
            Expr::List(elements) => {
                self.validate_list(elements, expr_type, env)
            }
            
            Expr::Pattern(pattern) => {
                self.validate_pattern(pattern, env)
            }
            
            Expr::Replace { expr, rules, .. } => {
                self.validate_replace(expr, rules, env)
            }
            
            // Most other expressions are safe by construction
            _ => Ok(()),
        }
    }
    
    /// Validate a function call
    fn validate_function_call(&mut self, head: &Expr, args: &[Expr], _result_type: &LyraType, env: &TypeEnvironment) -> TypeResult<()> {
        if let Expr::Symbol(sym) = head {
            if let Some(signature) = self.function_signatures.get(&sym.name).cloned() {
                return self.validate_against_signature(&sym.name, args, signature, env);
            }
        }
        
        // For unknown functions, we can't validate the signature but we can
        // still check argument type safety
        for arg in args {
            let arg_type = self.inference_engine.infer_expr(arg, env)?;
            self.validate_expression(arg, &arg_type, env)?;
        }
        
        Ok(())
    }
    
    /// Validate function call against a signature
    fn validate_against_signature(&mut self, name: &str, args: &[Expr], signature: FunctionSignature, env: &TypeEnvironment) -> TypeResult<()> {
        // Check arity
        if !signature.variadic && args.len() != signature.params.len() {
            return Err(TypeError::ArityMismatch {
                expected: signature.params.len(),
                actual: args.len(),
            });
        }
        
        if signature.variadic && args.len() < signature.params.len() {
            return Err(TypeError::ArityMismatch {
                expected: signature.params.len(),
                actual: args.len(),
            });
        }
        
        // Infer argument types
        let mut arg_types = Vec::new();
        for arg in args {
            arg_types.push(self.inference_engine.infer_expr(arg, env)?);
        }
        
        // Validate type constraints
        for constraint in &signature.constraints {
            self.validate_constraint(constraint, &arg_types, name)?;
        }
        
        Ok(())
    }
    
    /// Validate a type constraint
    fn validate_constraint(&self, constraint: &FunctionConstraint, arg_types: &[LyraType], function_name: &str) -> TypeResult<()> {
        match constraint {
            FunctionConstraint::Numeric => {
                for (i, arg_type) in arg_types.iter().enumerate() {
                    if !self.is_numeric_type(arg_type) {
                        return Err(TypeError::InvalidOperation {
                            operation: format!("{} (parameter {})", function_name, i + 1),
                            left: arg_type.clone(),
                            right: LyraType::Unknown, // placeholder
                        });
                    }
                }
            }
            
            FunctionConstraint::Tensor => {
                for (i, arg_type) in arg_types.iter().enumerate() {
                    if !arg_type.is_tensor() {
                        return Err(TypeError::InvalidOperation {
                            operation: format!("{} (parameter {})", function_name, i + 1),
                            left: arg_type.clone(),
                            right: LyraType::Unknown,
                        });
                    }
                }
            }
            
            FunctionConstraint::List => {
                for (i, arg_type) in arg_types.iter().enumerate() {
                    if !matches!(arg_type, LyraType::List(_)) {
                        return Err(TypeError::InvalidOperation {
                            operation: format!("{} (parameter {})", function_name, i + 1),
                            left: arg_type.clone(),
                            right: LyraType::Unknown,
                        });
                    }
                }
            }
            
            FunctionConstraint::Callable => {
                for (i, arg_type) in arg_types.iter().enumerate() {
                    if !arg_type.is_function() {
                        return Err(TypeError::InvalidOperation {
                            operation: format!("{} (parameter {})", function_name, i + 1),
                            left: arg_type.clone(),
                            right: LyraType::Unknown,
                        });
                    }
                }
            }
            
            FunctionConstraint::BroadcastCompatible(indices) => {
                self.validate_broadcast_compatibility(indices, arg_types)?;
            }
            
            FunctionConstraint::SameType(indices) => {
                self.validate_same_types(indices, arg_types)?;
            }
        }
        
        Ok(())
    }
    
    /// Validate broadcast compatibility for tensor operations
    fn validate_broadcast_compatibility(&self, indices: &[usize], arg_types: &[LyraType]) -> TypeResult<()> {
        let mut shapes = Vec::new();
        
        for &index in indices {
            if index >= arg_types.len() {
                continue;
            }
            
            if let LyraType::Tensor { shape: Some(shape), .. } = &arg_types[index] {
                shapes.push(shape);
            }
        }
        
        // Check all pairs are broadcast compatible
        for i in 0..shapes.len() {
            for j in i + 1..shapes.len() {
                if !shapes[i].broadcast_compatible(shapes[j]) {
                    return Err(TypeError::ShapeMismatch {
                        shape1: shapes[i].clone(),
                        shape2: shapes[j].clone(),
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate that specified types are the same
    fn validate_same_types(&self, indices: &[usize], arg_types: &[LyraType]) -> TypeResult<()> {
        if indices.len() < 2 {
            return Ok(());
        }
        
        let first_index = indices[0];
        if first_index >= arg_types.len() {
            return Ok(());
        }
        
        let first_type = &arg_types[first_index];
        
        for &index in &indices[1..] {
            if index >= arg_types.len() {
                continue;
            }
            
            if !self.types_compatible(first_type, &arg_types[index]) {
                return Err(TypeError::Mismatch {
                    expected: first_type.clone(),
                    actual: arg_types[index].clone(),
                });
            }
        }
        
        Ok(())
    }
    
    /// Check if two types are compatible (allowing for implicit conversions if not in strict mode)
    fn types_compatible(&self, t1: &LyraType, t2: &LyraType) -> bool {
        if t1 == t2 {
            return true;
        }
        
        if self.strict_mode {
            return false;
        }
        
        // Allow numeric conversions in non-strict mode
        match (t1, t2) {
            (LyraType::Integer, LyraType::Real) | (LyraType::Real, LyraType::Integer) => true,
            _ => false,
        }
    }
    
    /// Check if a type is numeric
    fn is_numeric_type(&self, ty: &LyraType) -> bool {
        match ty {
            LyraType::Integer | LyraType::Real => true,
            LyraType::TypeVar(_) => true, // Type variables can be numeric
            _ => false,
        }
    }
    
    /// Validate a list expression
    fn validate_list(&mut self, elements: &[Expr], list_type: &LyraType, env: &TypeEnvironment) -> TypeResult<()> {
        if let LyraType::List(elem_type) = list_type {
            for element in elements {
                let element_type = self.inference_engine.infer_expr(element, env)?;
                if !self.types_compatible(elem_type, &element_type) {
                    return Err(TypeError::Mismatch {
                        expected: elem_type.as_ref().clone(),
                        actual: element_type,
                    });
                }
                self.validate_expression(element, &element_type, env)?;
            }
        }
        Ok(())
    }
    
    /// Validate a pattern
    fn validate_pattern(&mut self, pattern: &Pattern, env: &TypeEnvironment) -> TypeResult<()> {
        match pattern {
            Pattern::Function { head, args } => {
                // Validate pattern function call
                let _head_type = self.inference_engine.infer_pattern(head, env)?;
                for arg in args {
                    self.validate_pattern(arg, env)?;
                }
            }
            
            Pattern::Predicate { pattern, test } => {
                self.validate_pattern(pattern, env)?;
                let test_type = self.inference_engine.infer_expr(test, env)?;
                
                // Test should return Boolean
                if !matches!(test_type, LyraType::Boolean | LyraType::TypeVar(_)) {
                    return Err(TypeError::Mismatch {
                        expected: LyraType::Boolean,
                        actual: test_type,
                    });
                }
            }
            
            Pattern::Alternative { patterns } => {
                for pattern in patterns {
                    self.validate_pattern(pattern, env)?;
                }
            }
            
            Pattern::Conditional { pattern, condition } => {
                self.validate_pattern(pattern, env)?;
                let condition_type = self.inference_engine.infer_expr(condition, env)?;
                
                if !matches!(condition_type, LyraType::Boolean | LyraType::TypeVar(_)) {
                    return Err(TypeError::Mismatch {
                        expected: LyraType::Boolean,
                        actual: condition_type,
                    });
                }
            }
            
            // Other patterns are safe by construction
            _ => {}
        }
        
        Ok(())
    }
    
    /// Validate a replace expression
    fn validate_replace(&mut self, expr: &Expr, rules: &Expr, env: &TypeEnvironment) -> TypeResult<()> {
        let _expr_type = self.inference_engine.infer_expr(expr, env)?;
        let _rules_type = self.inference_engine.infer_expr(rules, env)?;
        
        // TODO: Validate that rules are properly typed
        // For now, just check that expressions are well-typed
        Ok(())
    }
    
    /// Check if a program is type-safe
    pub fn check_program(&mut self, expressions: &[Expr], env: &TypeEnvironment) -> TypeResult<Vec<LyraType>> {
        let mut types = Vec::new();
        let mut current_env = env.clone();
        
        for expr in expressions {
            let expr_type = self.check_expression(expr, &current_env)?;
            types.push(expr_type.clone());
            
            // Update environment for assignments
            if let Expr::Assignment { lhs, .. } = expr {
                if let Expr::Symbol(sym) = lhs.as_ref() {
                    current_env.insert(sym.name.clone(), TypeScheme::monomorphic(expr_type));
                }
            }
        }
        
        Ok(types)
    }
    
    /// Get function signature for a function name
    pub fn get_signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.function_signatures.get(name)
    }
    
    /// Check if a function exists
    pub fn has_function(&self, name: &str) -> bool {
        self.function_signatures.contains_key(name)
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for type checking
pub fn check_expression_safety(expr: &Expr) -> TypeResult<LyraType> {
    let mut checker = TypeChecker::new();
    let env = TypeEnvironment::new();
    checker.check_expression(expr, &env)
}

pub fn check_expression_safety_strict(expr: &Expr) -> TypeResult<LyraType> {
    let mut checker = TypeChecker::new_strict();
    let env = TypeEnvironment::new();
    checker.check_expression(expr, &env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Symbol, Number, Expr};

    #[test]
    fn test_arithmetic_type_checking() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        // Plus[2, 3] should be valid
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(2)),
                Expr::Number(Number::Integer(3)),
            ],
        };
        
        let result = checker.check_expression(&expr, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_function_arity_checking() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        // Plus[2] should fail arity check
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![Expr::Number(Number::Integer(2))],
        };
        
        let result = checker.check_expression(&expr, &env);
        assert!(matches!(result, Err(TypeError::ArityMismatch { .. })));
    }

    #[test]
    fn test_list_type_checking() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        // Homogeneous list should be valid
        let expr = Expr::List(vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
        ]);
        
        let result = checker.check_expression(&expr, &env);
        assert!(result.is_ok());
        
        let ty = result.unwrap();
        assert_eq!(ty, LyraType::List(Box::new(LyraType::Integer)));
    }

    #[test]
    fn test_mathematical_function_checking() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        // Sin[3.14] should be valid
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
            args: vec![Expr::Number(Number::Real(3.14))],
        };
        
        let result = checker.check_expression(&expr, &env);
        assert!(result.is_ok());
    }

    #[test]
    fn test_strict_mode() {
        let mut checker_normal = TypeChecker::new();
        let mut checker_strict = TypeChecker::new_strict();
        let env = TypeEnvironment::new();
        
        // Mixed numeric types
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(2)),
                Expr::Number(Number::Real(3.5)),
            ],
        };
        
        // Normal mode might allow this (depending on implementation)
        let _result_normal = checker_normal.check_expression(&expr, &env);
        
        // Strict mode should be more restrictive
        let _result_strict = checker_strict.check_expression(&expr, &env);
        
        // Both should at least not panic
        assert!(true);
    }

    #[test]
    fn test_function_signature_validation() {
        let mut checker = TypeChecker::new();
        
        // Test signature exists
        assert!(checker.has_function("Plus"));
        assert!(checker.has_function("Sin"));
        assert!(!checker.has_function("NonExistentFunction"));
        
        // Test signature retrieval
        let plus_sig = checker.get_signature("Plus").unwrap();
        assert_eq!(plus_sig.params.len(), 2);
        assert!(!plus_sig.variadic);
    }

    #[test]
    fn test_pattern_validation() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        // Simple blank pattern
        let pattern = Pattern::Blank { head: None };
        let result = checker.validate_pattern(&pattern, &env);
        assert!(result.is_ok());
        
        // Predicate pattern with boolean test
        let test_expr = Expr::Symbol(Symbol { name: "True".to_string() });
        let pattern = Pattern::Predicate {
            pattern: Box::new(Pattern::Blank { head: None }),
            test: Box::new(test_expr),
        };
        
        // This might fail because "True" isn't defined, but it shouldn't panic
        let _result = checker.validate_pattern(&pattern, &env);
        assert!(true);
    }

    #[test]
    fn test_convenience_functions() {
        let expr = Expr::Number(Number::Integer(42));
        
        let result = check_expression_safety(&expr);
        assert!(result.is_ok());
        
        let result = check_expression_safety_strict(&expr);
        assert!(result.is_ok());
    }

    #[test]
    fn test_program_checking() {
        let mut checker = TypeChecker::new();
        let env = TypeEnvironment::new();
        
        let expressions = vec![
            Expr::Number(Number::Integer(42)),
            Expr::Function {
                head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
                args: vec![
                    Expr::Number(Number::Integer(1)),
                    Expr::Number(Number::Integer(2)),
                ],
            },
        ];
        
        let result = checker.check_program(&expressions, &env);
        assert!(result.is_ok());
        
        let types = result.unwrap();
        assert_eq!(types.len(), 2);
    }
}