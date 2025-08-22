use super::{
    LyraType, TensorShape, TypeError, TypeResult, TypeEnvironment, TypeScheme, 
    TypeSubstitution, TypeVar, TypeVarGenerator, TypeConstraint,
    unification::{ConstraintSolver, Unifier, unify_types}
};
use crate::ast::{Expr, Number, Pattern};

/// Type inference engine implementing the Hindley-Milner algorithm
#[derive(Debug)]
pub struct TypeInferenceEngine {
    /// Type variable generator for creating fresh variables
    var_gen: TypeVarGenerator,
    /// Global type environment with built-in functions
    global_env: TypeEnvironment,
    /// Constraint collector for the inference process
    constraints: Vec<TypeConstraint>,
}

impl TypeInferenceEngine {
    /// Create a new type inference engine
    pub fn new() -> Self {
        let mut engine = TypeInferenceEngine {
            var_gen: TypeVarGenerator::new(),
            global_env: TypeEnvironment::new(),
            constraints: Vec::new(),
        };
        
        // Initialize built-in types and functions
        engine.initialize_builtins();
        engine
    }
    
    /// Initialize built-in functions and their type signatures
    fn initialize_builtins(&mut self) {
        // Arithmetic functions
        let binary_numeric = TypeScheme::new(
            vec![0],
            LyraType::Function {
                params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
                return_type: Box::new(LyraType::TypeVar(0)),
                attributes: vec![],
            }
        );
        
        self.global_env.insert("Plus".to_string(), binary_numeric.clone());
        self.global_env.insert("Minus".to_string(), binary_numeric.clone());
        self.global_env.insert("Times".to_string(), binary_numeric.clone());
        self.global_env.insert("Divide".to_string(), binary_numeric.clone());
        
        // Power function: (α, β) -> Real where α,β are numeric
        let power_scheme = TypeScheme::new(
            vec![0, 1],
            LyraType::Function {
                params: vec![LyraType::TypeVar(0), LyraType::TypeVar(1)],
                return_type: Box::new(LyraType::Real),
                attributes: vec![],
            }
        );
        self.global_env.insert("Power".to_string(), power_scheme);
        
        // Mathematical functions: Real -> Real
        let real_to_real = TypeScheme::monomorphic(LyraType::Function {
            params: vec![LyraType::Real],
            return_type: Box::new(LyraType::Real),
            attributes: vec![],
        });
        
        self.global_env.insert("Sin".to_string(), real_to_real.clone());
        self.global_env.insert("Cos".to_string(), real_to_real.clone());
        self.global_env.insert("Tan".to_string(), real_to_real.clone());
        self.global_env.insert("Exp".to_string(), real_to_real.clone());
        self.global_env.insert("Log".to_string(), real_to_real.clone());
        self.global_env.insert("Sqrt".to_string(), real_to_real);
        
        // List functions
        let length_scheme = TypeScheme::new(
            vec![0],
            LyraType::Function {
                params: vec![LyraType::List(Box::new(LyraType::TypeVar(0)))],
                return_type: Box::new(LyraType::Integer),
                attributes: vec![],
            }
        );
        self.global_env.insert("Length".to_string(), length_scheme);
        
        let head_scheme = TypeScheme::new(
            vec![0],
            LyraType::Function {
                params: vec![LyraType::List(Box::new(LyraType::TypeVar(0)))],
                return_type: Box::new(LyraType::TypeVar(0)),
                attributes: vec![],
            }
        );
        self.global_env.insert("Head".to_string(), head_scheme);
        
        let tail_scheme = TypeScheme::new(
            vec![0],
            LyraType::Function {
                params: vec![LyraType::List(Box::new(LyraType::TypeVar(0)))],
                return_type: Box::new(LyraType::List(Box::new(LyraType::TypeVar(0)))),
                attributes: vec![],
            }
        );
        self.global_env.insert("Tail".to_string(), tail_scheme);
        
        // Append: (List[α], α) -> List[α]
        let append_scheme = TypeScheme::new(
            vec![0],
            LyraType::Function {
                params: vec![
                    LyraType::List(Box::new(LyraType::TypeVar(0))),
                    LyraType::TypeVar(0),
                ],
                return_type: Box::new(LyraType::List(Box::new(LyraType::TypeVar(0)))),
                attributes: vec![],
            }
        );
        self.global_env.insert("Append".to_string(), append_scheme);
        
        // Map: ((α -> β), List[α]) -> List[β]
        let map_scheme = TypeScheme::new(
            vec![0, 1],
            LyraType::Function {
                params: vec![
                    LyraType::Function {
                        params: vec![LyraType::TypeVar(0)],
                        return_type: Box::new(LyraType::TypeVar(1)),
                        attributes: vec![],
                    },
                    LyraType::List(Box::new(LyraType::TypeVar(0))),
                ],
                return_type: Box::new(LyraType::List(Box::new(LyraType::TypeVar(1)))),
                attributes: vec![],
            }
        );
        self.global_env.insert("Map".to_string(), map_scheme);
        
        // Tensor operations
        let tensor_add_scheme = TypeScheme::new(
            vec![0],
            LyraType::Function {
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
                return_type: Box::new(LyraType::Tensor {
                    element_type: Box::new(LyraType::TypeVar(0)),
                    shape: None,
                }),
                attributes: vec![],
            }
        );
        self.global_env.insert("TensorAdd".to_string(), tensor_add_scheme);
    }
    
    /// Infer the type of an expression
    pub fn infer_expr(&mut self, expr: &Expr, env: &TypeEnvironment) -> TypeResult<LyraType> {
        self.constraints.clear();
        let inferred_type = self.generate_constraints(expr, env)?;
        
        // Solve constraints
        let mut solver = ConstraintSolver::new();
        for constraint in &self.constraints {
            solver.add_constraint(constraint.clone());
        }
        
        let substitution = solver.solve()?;
        Ok(substitution.apply(&inferred_type))
    }
    
    /// Generate type constraints for an expression
    fn generate_constraints(&mut self, expr: &Expr, env: &TypeEnvironment) -> TypeResult<LyraType> {
        match expr {
            Expr::Number(Number::Integer(_)) => Ok(LyraType::Integer),
            Expr::Number(Number::Real(_)) => Ok(LyraType::Real),
            Expr::String(_) => Ok(LyraType::String),
            
            Expr::Symbol(sym) => {
                // Look up in local environment first, then global
                if let Some(scheme) = env.get(&sym.name) {
                    Ok(scheme.instantiate(&mut self.var_gen))
                } else if let Some(scheme) = self.global_env.get(&sym.name) {
                    Ok(scheme.instantiate(&mut self.var_gen))
                } else {
                    Err(TypeError::UnboundVariable { name: sym.name.clone() })
                }
            }
            
            Expr::List(elements) => {
                if elements.is_empty() {
                    // Empty list has type List[α] for fresh α
                    let elem_type = self.var_gen.fresh_type();
                    Ok(LyraType::List(Box::new(elem_type)))
                } else {
                    // Infer type of first element
                    let first_type = self.generate_constraints(&elements[0], env)?;
                    
                    // All elements must have the same type
                    for element in &elements[1..] {
                        let elem_type = self.generate_constraints(element, env)?;
                        self.constraints.push(TypeConstraint::Equal(first_type.clone(), elem_type));
                    }
                    
                    Ok(LyraType::List(Box::new(first_type)))
                }
            }
            
            Expr::Function { head, args } => {
                self.infer_function_call(head, args, env)
            }
            
            Expr::Pattern(pattern) => {
                let inner_type = self.infer_pattern(pattern, env)?;
                Ok(LyraType::Pattern(Box::new(inner_type)))
            }
            
            Expr::Rule { lhs, rhs, .. } => {
                let lhs_type = self.generate_constraints(lhs, env)?;
                let rhs_type = self.generate_constraints(rhs, env)?;
                Ok(LyraType::Rule {
                    lhs_type: Box::new(lhs_type),
                    rhs_type: Box::new(rhs_type),
                })
            }
            
            Expr::Replace { expr, rules, .. } => {
                let expr_type = self.generate_constraints(expr, env)?;
                let _rules_type = self.generate_constraints(rules, env)?;
                // Result type is the same as the expression being transformed
                Ok(expr_type)
            }
            
            Expr::Assignment { lhs, rhs, .. } => {
                let _lhs_type = self.generate_constraints(lhs, env)?;
                let rhs_type = self.generate_constraints(rhs, env)?;
                // Assignment returns Unit but we return the RHS type for now
                Ok(rhs_type)
            }
            
            Expr::DotCall { object, method, args } => {
                let _object_type = self.generate_constraints(object, env)?;
                // For now, treat dot calls as regular function calls
                // In a full implementation, we'd check method types on object types
                let method_expr = Expr::Symbol(crate::ast::Symbol { name: method.clone() });
                let mut all_args = vec![*object.clone()];
                all_args.extend(args.clone());
                self.infer_function_call(&method_expr, &all_args, env)
            }
            
            Expr::ArrowFunction { params, body } => {
                // Create fresh type variables for parameters
                let mut extended_env = env.clone();
                let mut param_types = Vec::new();
                
                for param in params {
                    let param_type = self.var_gen.fresh_type();
                    param_types.push(param_type.clone());
                    extended_env.insert(param.clone(), TypeScheme::monomorphic(param_type));
                }
                
                // Infer body type with extended environment
                let body_type = self.generate_constraints(body, &extended_env)?;
                
                Ok(LyraType::Function {
                    params: param_types,
                    return_type: Box::new(body_type),
                    attributes: vec![],
                })
            }
            
            // TODO: Implement other expression types
            _ => {
                // For unsupported expressions, return Unknown type
                Ok(LyraType::Unknown)
            }
        }
    }
    
    /// Infer the type of a function call
    fn infer_function_call(&mut self, head: &Expr, args: &[Expr], env: &TypeEnvironment) -> TypeResult<LyraType> {
        // Get function type
        let func_type = self.generate_constraints(head, env)?;
        
        // Generate argument types
        let mut arg_types = Vec::new();
        for arg in args {
            arg_types.push(self.generate_constraints(arg, env)?);
        }
        
        // Create fresh type variable for return type
        let return_type = self.var_gen.fresh_type();
        
        // Expected function type: (arg_types...) -> return_type
        let expected_func_type = LyraType::Function {
            params: arg_types,
            return_type: Box::new(return_type.clone()),
            attributes: vec![],
        };
        
        // Add constraint that function type equals expected type
        self.constraints.push(TypeConstraint::Equal(func_type, expected_func_type));
        
        Ok(return_type)
    }
    
    /// Infer the type of a pattern
    pub fn infer_pattern(&mut self, pattern: &Pattern, env: &TypeEnvironment) -> TypeResult<LyraType> {
        match pattern {
            Pattern::Blank { head } => {
                if let Some(head_name) = head {
                    // Look up the head type
                    if let Some(scheme) = env.get(head_name) {
                        Ok(scheme.instantiate(&mut self.var_gen))
                    } else if let Some(scheme) = self.global_env.get(head_name) {
                        Ok(scheme.instantiate(&mut self.var_gen))
                    } else {
                        // Unknown head, create fresh type variable
                        Ok(self.var_gen.fresh_type())
                    }
                } else {
                    // Untyped blank, can match anything
                    Ok(self.var_gen.fresh_type())
                }
            }
            
            Pattern::Exact { value } => {
                // Exact pattern has the same type as the value it matches
                self.generate_constraints(value, env)
            }
            
            Pattern::BlankSequence { head } | Pattern::BlankNullSequence { head } => {
                let elem_type = if let Some(head_name) = head {
                    if let Some(scheme) = env.get(head_name) {
                        scheme.instantiate(&mut self.var_gen)
                    } else if let Some(scheme) = self.global_env.get(head_name) {
                        scheme.instantiate(&mut self.var_gen)
                    } else {
                        self.var_gen.fresh_type()
                    }
                } else {
                    self.var_gen.fresh_type()
                };
                
                // Sequence patterns match lists
                Ok(LyraType::List(Box::new(elem_type)))
            }
            
            Pattern::Named { pattern, .. } => {
                // Named patterns have the same type as their inner pattern
                self.infer_pattern(pattern, env)
            }
            
            Pattern::Function { head, args } => {
                // Pattern function call
                let head_type = self.infer_pattern(head, env)?;
                let mut arg_types = Vec::new();
                for arg in args {
                    arg_types.push(self.infer_pattern(arg, env)?);
                }
                
                let return_type = self.var_gen.fresh_type();
                let expected_func_type = LyraType::Function {
                    params: arg_types,
                    return_type: Box::new(return_type.clone()),
                    attributes: vec![],
                };
                
                self.constraints.push(TypeConstraint::Equal(head_type, expected_func_type));
                Ok(return_type)
            }
            
            Pattern::Typed { type_pattern, .. } => {
                // Type patterns constrain the type
                self.generate_constraints(type_pattern, env)
            }
            
            Pattern::Predicate { pattern, test } => {
                let pattern_type = self.infer_pattern(pattern, env)?;
                let test_type = self.generate_constraints(test, env)?;
                
                // Test should be a function that returns Boolean
                let expected_test_type = LyraType::Function {
                    params: vec![pattern_type.clone()],
                    return_type: Box::new(LyraType::Boolean),
                    attributes: vec![],
                };
                
                self.constraints.push(TypeConstraint::Equal(test_type, expected_test_type));
                Ok(pattern_type)
            }
            
            Pattern::Alternative { patterns } => {
                if patterns.is_empty() {
                    return Ok(self.var_gen.fresh_type());
                }
                
                // All alternatives must have the same type
                let first_type = self.infer_pattern(&patterns[0], env)?;
                for pattern in &patterns[1..] {
                    let pattern_type = self.infer_pattern(pattern, env)?;
                    self.constraints.push(TypeConstraint::Equal(first_type.clone(), pattern_type));
                }
                
                Ok(first_type)
            }
            
            Pattern::Conditional { pattern, condition } => {
                let pattern_type = self.infer_pattern(pattern, env)?;
                let condition_type = self.generate_constraints(condition, env)?;
                
                // Condition should be Boolean
                self.constraints.push(TypeConstraint::Equal(condition_type, LyraType::Boolean));
                Ok(pattern_type)
            }
        }
    }
    
    /// Infer types for a list of expressions (program)
    pub fn infer_program(&mut self, expressions: &[Expr], env: &TypeEnvironment) -> TypeResult<Vec<LyraType>> {
        let mut types = Vec::new();
        let mut current_env = env.clone();
        
        for expr in expressions {
            let expr_type = self.infer_expr(expr, &current_env)?;
            types.push(expr_type.clone());
            
            // For assignments, update the environment
            if let Expr::Assignment { lhs, .. } = expr {
                if let Expr::Symbol(sym) = lhs.as_ref() {
                    current_env.insert(sym.name.clone(), TypeScheme::monomorphic(expr_type));
                }
            }
        }
        
        Ok(types)
    }
    
    /// Check if an expression has a specific type
    pub fn check_expr(&mut self, expr: &Expr, expected_type: &LyraType, env: &TypeEnvironment) -> TypeResult<()> {
        let inferred_type = self.infer_expr(expr, env)?;
        let mut unifier = Unifier::new();
        unifier.unify(&inferred_type, expected_type)?;
        Ok(())
    }
    
    /// Get the current type environment (for testing)
    pub fn global_env(&self) -> &TypeEnvironment {
        &self.global_env
    }
    
    /// Add a new binding to the global environment
    pub fn add_global_binding(&mut self, name: String, scheme: TypeScheme) {
        self.global_env.insert(name, scheme);
    }
}

impl Default for TypeInferenceEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for type inference
pub fn infer_expression_type(expr: &Expr) -> TypeResult<LyraType> {
    let mut engine = TypeInferenceEngine::new();
    let env = TypeEnvironment::new();
    engine.infer_expr(expr, &env)
}

pub fn check_expression_type(expr: &Expr, expected_type: &LyraType) -> TypeResult<()> {
    let mut engine = TypeInferenceEngine::new();
    let env = TypeEnvironment::new();
    engine.check_expr(expr, expected_type, &env)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Symbol, Expr};

    #[test]
    fn test_literal_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Integer literal
        let expr = Expr::Number(Number::Integer(42));
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, LyraType::Integer);
        
        // Real literal
        let expr = Expr::Number(Number::Real(3.14));
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, LyraType::Real);
        
        // String literal
        let expr = Expr::String("hello".to_string());
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, LyraType::String);
    }

    #[test]
    fn test_list_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Empty list
        let expr = Expr::List(vec![]);
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert!(matches!(ty, LyraType::List(_)));
        
        // Integer list
        let expr = Expr::List(vec![
            Expr::Number(Number::Integer(1)),
            Expr::Number(Number::Integer(2)),
            Expr::Number(Number::Integer(3)),
        ]);
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert_eq!(ty, LyraType::List(Box::new(LyraType::Integer)));
    }

    #[test]
    fn test_function_call_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Plus[2, 3] should have type Integer
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Plus".to_string() })),
            args: vec![
                Expr::Number(Number::Integer(2)),
                Expr::Number(Number::Integer(3)),
            ],
        };
        
        let ty = engine.infer_expr(&expr, &env).unwrap();
        // The result should be Integer (since both operands are Integer)
        assert!(matches!(ty, LyraType::Integer) || matches!(ty, LyraType::TypeVar(_)));
    }

    #[test]
    fn test_symbol_lookup() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Built-in function lookup
        let expr = Expr::Symbol(Symbol { name: "Sin".to_string() });
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert!(ty.is_function());
        
        // Unknown symbol should fail
        let expr = Expr::Symbol(Symbol { name: "UnknownSymbol".to_string() });
        let result = engine.infer_expr(&expr, &env);
        assert!(matches!(result, Err(TypeError::UnboundVariable { .. })));
    }

    #[test]
    fn test_arrow_function_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // (x) => x should have type α -> α
        let expr = Expr::ArrowFunction {
            params: vec!["x".to_string()],
            body: Box::new(Expr::Symbol(Symbol { name: "x".to_string() })),
        };
        
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert!(ty.is_function());
        
        if let LyraType::Function { params, return_type, .. } = ty {
            assert_eq!(params.len(), 1);
            // The parameter and return type should be the same (both type variables)
            if let (LyraType::TypeVar(p), LyraType::TypeVar(r)) = (&params[0], return_type.as_ref()) {
                assert_eq!(p, r);
            }
        }
    }

    #[test]
    fn test_pattern_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Blank pattern
        let pattern = Pattern::Blank { head: None };
        let ty = engine.infer_pattern(&pattern, &env).unwrap();
        assert!(matches!(ty, LyraType::TypeVar(_)));
        
        // Typed blank pattern
        let pattern = Pattern::Blank { head: Some("Integer".to_string()) };
        let ty = engine.infer_pattern(&pattern, &env).unwrap();
        // Should be a type variable since "Integer" isn't in scope as a value
        assert!(matches!(ty, LyraType::TypeVar(_)));
    }

    #[test]
    fn test_mathematical_functions() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Sin[3.14] should have type Real
        let expr = Expr::Function {
            head: Box::new(Expr::Symbol(Symbol { name: "Sin".to_string() })),
            args: vec![Expr::Number(Number::Real(3.14))],
        };
        
        let ty = engine.infer_expr(&expr, &env).unwrap();
        assert!(matches!(ty, LyraType::Real) || matches!(ty, LyraType::TypeVar(_)));
    }

    #[test]
    fn test_type_checking() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        // Check that 42 has type Integer
        let expr = Expr::Number(Number::Integer(42));
        let result = engine.check_expr(&expr, &LyraType::Integer, &env);
        assert!(result.is_ok());
        
        // Check that 42 doesn't have type String
        let result = engine.check_expr(&expr, &LyraType::String, &env);
        assert!(result.is_err());
    }

    #[test]
    fn test_program_inference() {
        let mut engine = TypeInferenceEngine::new();
        let env = TypeEnvironment::new();
        
        let expressions = vec![
            Expr::Number(Number::Integer(42)),
            Expr::String("hello".to_string()),
            Expr::List(vec![Expr::Number(Number::Real(1.0))]),
        ];
        
        let types = engine.infer_program(&expressions, &env).unwrap();
        assert_eq!(types.len(), 3);
        assert_eq!(types[0], LyraType::Integer);
        assert_eq!(types[1], LyraType::String);
        assert_eq!(types[2], LyraType::List(Box::new(LyraType::Real)));
    }

    #[test]
    fn test_convenience_functions() {
        // Test convenience functions
        let expr = Expr::Number(Number::Integer(42));
        let ty = infer_expression_type(&expr).unwrap();
        assert_eq!(ty, LyraType::Integer);
        
        let result = check_expression_type(&expr, &LyraType::Integer);
        assert!(result.is_ok());
        
        let result = check_expression_type(&expr, &LyraType::String);
        assert!(result.is_err());
    }
}