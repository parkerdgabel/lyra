use super::{LyraType, TensorShape, TypeConstraint, TypeError, TypeResult, TypeSubstitution, TypeVar};
use std::collections::VecDeque;

/// Unification algorithm for the Hindley-Milner type system
#[derive(Debug, Clone)]
pub struct Unifier {
    substitution: TypeSubstitution,
}

impl Unifier {
    pub fn new() -> Self {
        Unifier {
            substitution: TypeSubstitution::new(),
        }
    }

    /// Unify two types, returning the most general unifier (MGU)
    pub fn unify(&mut self, t1: &LyraType, t2: &LyraType) -> TypeResult<()> {
        self.unify_types(t1, t2)
    }

    /// Get the current substitution
    pub fn substitution(&self) -> &TypeSubstitution {
        &self.substitution
    }

    /// Take the substitution, consuming the unifier
    pub fn into_substitution(self) -> TypeSubstitution {
        self.substitution
    }

    /// Internal unification algorithm
    fn unify_types(&mut self, t1: &LyraType, t2: &LyraType) -> TypeResult<()> {
        // Apply current substitution to both types first
        let t1 = self.substitution.apply(t1);
        let t2 = self.substitution.apply(t2);

        match (&t1, &t2) {
            // Same types unify trivially
            (LyraType::Integer, LyraType::Integer) => Ok(()),
            (LyraType::Real, LyraType::Real) => Ok(()),
            (LyraType::String, LyraType::String) => Ok(()),
            (LyraType::Boolean, LyraType::Boolean) => Ok(()),
            (LyraType::Symbol, LyraType::Symbol) => Ok(()),
            (LyraType::Unit, LyraType::Unit) => Ok(()),
            (LyraType::Unknown, _) => Ok(()), // Unknown unifies with anything
            (_, LyraType::Unknown) => Ok(()),

            // Type variable unification
            (LyraType::TypeVar(var), ty) | (ty, LyraType::TypeVar(var)) => {
                self.unify_var(*var, ty)
            }

            // List types
            (LyraType::List(elem1), LyraType::List(elem2)) => {
                self.unify_types(elem1, elem2)
            }

            // Tensor types
            (
                LyraType::Tensor {
                    element_type: elem1,
                    shape: shape1,
                },
                LyraType::Tensor {
                    element_type: elem2,
                    shape: shape2,
                },
            ) => {
                // Unify element types
                self.unify_types(elem1, elem2)?;
                
                // Check shape compatibility
                self.unify_tensor_shapes(shape1, shape2)
            }

            // Function types
            (
                LyraType::Function {
                    params: params1,
                    return_type: ret1,
                },
                LyraType::Function {
                    params: params2,
                    return_type: ret2,
                },
            ) => {
                // Check arity
                if params1.len() != params2.len() {
                    return Err(TypeError::ArityMismatch {
                        expected: params1.len(),
                        actual: params2.len(),
                    });
                }

                // Unify parameter types
                for (p1, p2) in params1.iter().zip(params2.iter()) {
                    self.unify_types(p1, p2)?;
                }

                // Unify return types
                self.unify_types(ret1, ret2)
            }

            // Pattern types
            (LyraType::Pattern(inner1), LyraType::Pattern(inner2)) => {
                self.unify_types(inner1, inner2)
            }

            // Rule types
            (
                LyraType::Rule {
                    lhs_type: lhs1,
                    rhs_type: rhs1,
                },
                LyraType::Rule {
                    lhs_type: lhs2,
                    rhs_type: rhs2,
                },
            ) => {
                self.unify_types(lhs1, lhs2)?;
                self.unify_types(rhs1, rhs2)
            }

            // Error types always fail unification unless they're the same error
            (LyraType::Error(msg1), LyraType::Error(msg2)) if msg1 == msg2 => Ok(()),
            (LyraType::Error(_), _) | (_, LyraType::Error(_)) => Err(TypeError::Mismatch {
                expected: t1.clone(),
                actual: t2.clone(),
            }),

            // All other combinations fail
            _ => Err(TypeError::Mismatch {
                expected: t1.clone(),
                actual: t2.clone(),
            }),
        }
    }

    /// Unify a type variable with a type
    fn unify_var(&mut self, var: TypeVar, ty: &LyraType) -> TypeResult<()> {
        match ty {
            LyraType::TypeVar(other_var) if var == *other_var => {
                // Same variable unifies with itself
                Ok(())
            }
            _ => {
                // Occurs check: ensure var doesn't occur in ty
                if self.occurs_check(var, ty) {
                    return Err(TypeError::OccursCheck {
                        var,
                        ty: ty.clone(),
                    });
                }

                // Add substitution and propagate
                let new_substitution = TypeSubstitution::from_single(var, ty.clone());
                self.substitution = self.substitution.compose(&new_substitution);
                Ok(())
            }
        }
    }

    /// Occurs check: verify that a type variable doesn't occur in a type
    fn occurs_check(&self, var: TypeVar, ty: &LyraType) -> bool {
        match ty {
            LyraType::TypeVar(other_var) => var == *other_var,
            LyraType::List(elem_type) => self.occurs_check(var, elem_type),
            LyraType::Tensor { element_type, .. } => self.occurs_check(var, element_type),
            LyraType::Function { params, return_type } => {
                params.iter().any(|p| self.occurs_check(var, p))
                    || self.occurs_check(var, return_type)
            }
            LyraType::Pattern(inner) => self.occurs_check(var, inner),
            LyraType::Rule { lhs_type, rhs_type } => {
                self.occurs_check(var, lhs_type) || self.occurs_check(var, rhs_type)
            }
            _ => false,
        }
    }

    /// Unify tensor shapes
    fn unify_tensor_shapes(
        &self,
        shape1: &Option<TensorShape>,
        shape2: &Option<TensorShape>,
    ) -> TypeResult<()> {
        match (shape1, shape2) {
            (None, None) => Ok(()),
            (None, Some(_)) | (Some(_), None) => {
                // One has unknown shape, the other has known shape - they can unify
                Ok(())
            }
            (Some(s1), Some(s2)) => {
                if s1.broadcast_compatible(s2) {
                    Ok(())
                } else {
                    Err(TypeError::ShapeMismatch {
                        shape1: s1.clone(),
                        shape2: s2.clone(),
                    })
                }
            }
        }
    }
}

impl Default for Unifier {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeSubstitution {
    /// Create a substitution with a single mapping
    pub fn from_single(var: TypeVar, ty: LyraType) -> Self {
        let mut substitution = TypeSubstitution::new();
        substitution.insert(var, ty);
        substitution
    }
}

/// Constraint solver for type inference
#[derive(Debug)]
pub struct ConstraintSolver {
    constraints: VecDeque<TypeConstraint>,
    unifier: Unifier,
}

impl ConstraintSolver {
    pub fn new() -> Self {
        ConstraintSolver {
            constraints: VecDeque::new(),
            unifier: Unifier::new(),
        }
    }

    /// Add a constraint to be solved
    pub fn add_constraint(&mut self, constraint: TypeConstraint) {
        self.constraints.push_back(constraint);
    }

    /// Solve all constraints, returning the final substitution
    pub fn solve(mut self) -> TypeResult<TypeSubstitution> {
        while let Some(constraint) = self.constraints.pop_front() {
            self.solve_constraint(constraint)?;
        }
        Ok(self.unifier.into_substitution())
    }

    /// Solve a single constraint
    fn solve_constraint(&mut self, constraint: TypeConstraint) -> TypeResult<()> {
        match constraint {
            TypeConstraint::Equal(t1, t2) => {
                self.unifier.unify(&t1, &t2)
            }
            TypeConstraint::Subtype(t1, t2) => {
                // For now, treat subtype constraints as equality constraints
                // In a more sophisticated system, we'd implement proper subtyping
                self.unifier.unify(&t1, &t2)
            }
            TypeConstraint::Instance(ty, scheme) => {
                // This would require instantiation of the scheme and unification
                // For now, we'll handle this in the type inference engine
                Err(TypeError::AmbiguousType)
            }
        }
    }
}

impl Default for ConstraintSolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for common unification operations
pub fn unify_types(t1: &LyraType, t2: &LyraType) -> TypeResult<TypeSubstitution> {
    let mut unifier = Unifier::new();
    unifier.unify(t1, t2)?;
    Ok(unifier.into_substitution())
}

/// Unify a list of type pairs
pub fn unify_many(pairs: &[(LyraType, LyraType)]) -> TypeResult<TypeSubstitution> {
    let mut unifier = Unifier::new();
    for (t1, t2) in pairs {
        unifier.unify(t1, t2)?;
    }
    Ok(unifier.into_substitution())
}

/// Check if two types can be unified without computing the substitution
pub fn can_unify(t1: &LyraType, t2: &LyraType) -> bool {
    unify_types(t1, t2).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_unification() {
        // Integer unifies with Integer
        let result = unify_types(&LyraType::Integer, &LyraType::Integer);
        assert!(result.is_ok());
        
        // Integer doesn't unify with Real
        let result = unify_types(&LyraType::Integer, &LyraType::Real);
        assert!(result.is_err());
    }

    #[test]
    fn test_type_var_unification() {
        // Type variable unifies with any type
        let t1 = LyraType::TypeVar(0);
        let t2 = LyraType::Integer;
        
        let substitution = unify_types(&t1, &t2).unwrap();
        assert_eq!(substitution.get(0), Some(LyraType::Integer));
        
        // Two different type variables unify
        let t1 = LyraType::TypeVar(0);
        let t2 = LyraType::TypeVar(1);
        let substitution = unify_types(&t1, &t2).unwrap();
        assert!(substitution.get(0).is_some() || substitution.get(1).is_some());
    }

    #[test]
    fn test_occurs_check() {
        // α → List[α] should fail occurs check
        let var_type = LyraType::TypeVar(0);
        let list_type = LyraType::List(Box::new(LyraType::TypeVar(0)));
        
        let result = unify_types(&var_type, &list_type);
        assert!(matches!(result, Err(TypeError::OccursCheck { var: 0, .. })));
    }

    #[test]
    fn test_function_unification() {
        // (Integer -> Real) unifies with (Integer -> Real)
        let f1 = LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
        };
        let f2 = LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
        };
        
        let result = unify_types(&f1, &f2);
        assert!(result.is_ok());
        
        // Functions with different arities don't unify
        let f3 = LyraType::Function {
            params: vec![LyraType::Integer, LyraType::Real],
            return_type: Box::new(LyraType::Real),
        };
        
        let result = unify_types(&f1, &f3);
        assert!(matches!(result, Err(TypeError::ArityMismatch { .. })));
    }

    #[test]
    fn test_list_unification() {
        // List[Integer] unifies with List[Integer]
        let l1 = LyraType::List(Box::new(LyraType::Integer));
        let l2 = LyraType::List(Box::new(LyraType::Integer));
        
        let result = unify_types(&l1, &l2);
        assert!(result.is_ok());
        
        // List[α] unifies with List[Integer]
        let l3 = LyraType::List(Box::new(LyraType::TypeVar(0)));
        
        let substitution = unify_types(&l3, &l1).unwrap();
        assert_eq!(substitution.get(0), Some(LyraType::Integer));
    }

    #[test]
    fn test_tensor_unification() {
        let t1 = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::vector(10)),
        };
        let t2 = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::vector(10)),
        };
        
        // Same tensors unify
        let result = unify_types(&t1, &t2);
        assert!(result.is_ok());
        
        // Different element types don't unify
        let t3 = LyraType::Tensor {
            element_type: Box::new(LyraType::Integer),
            shape: Some(TensorShape::vector(10)),
        };
        
        let result = unify_types(&t1, &t3);
        assert!(result.is_err());
        
        // Incompatible shapes don't unify
        let t4 = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::matrix(3, 4)),
        };
        
        let result = unify_types(&t1, &t4);
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_unification() {
        // (α -> β) unifies with (Integer -> Real)
        let f1 = LyraType::Function {
            params: vec![LyraType::TypeVar(0)],
            return_type: Box::new(LyraType::TypeVar(1)),
        };
        let f2 = LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
        };
        
        let substitution = unify_types(&f1, &f2).unwrap();
        assert_eq!(substitution.get(0), Some(LyraType::Integer));
        assert_eq!(substitution.get(1), Some(LyraType::Real));
    }

    #[test]
    fn test_constraint_solver() {
        let mut solver = ConstraintSolver::new();
        
        // Add constraints: α = Integer, β = List[α]
        solver.add_constraint(TypeConstraint::Equal(
            LyraType::TypeVar(0),
            LyraType::Integer,
        ));
        solver.add_constraint(TypeConstraint::Equal(
            LyraType::TypeVar(1),
            LyraType::List(Box::new(LyraType::TypeVar(0))),
        ));
        
        let substitution = solver.solve().unwrap();
        assert_eq!(substitution.get(0), Some(LyraType::Integer));
        assert_eq!(substitution.get(1), Some(LyraType::List(Box::new(LyraType::Integer))));
    }

    #[test]
    fn test_substitution_composition() {
        let mut sub1 = TypeSubstitution::new();
        sub1.insert(0, LyraType::TypeVar(1));
        
        let mut sub2 = TypeSubstitution::new();
        sub2.insert(1, LyraType::Integer);
        
        let composed = sub1.compose(&sub2);
        
        // After composition: 0 -> TypeVar(1) (from sub1), 1 -> Integer (from sub2)
        // But when we apply the composed substitution to TypeVar(0), we get TypeVar(1)
        // which should not be further substituted automatically in this test
        let result = composed.apply(&LyraType::TypeVar(0));
        assert_eq!(result, LyraType::TypeVar(1));
        
        // However, if we want the full transitive closure, we need to apply substitutions iteratively
        let sub1_then_sub2 = composed.apply(&LyraType::TypeVar(1));
        assert_eq!(sub1_then_sub2, LyraType::Integer);
    }

    #[test]
    fn test_can_unify() {
        assert!(can_unify(&LyraType::Integer, &LyraType::Integer));
        assert!(!can_unify(&LyraType::Integer, &LyraType::String));
        assert!(can_unify(&LyraType::TypeVar(0), &LyraType::Integer));
        assert!(can_unify(&LyraType::Unknown, &LyraType::Real));
    }

    #[test]
    fn test_unify_many() {
        let pairs = vec![
            (LyraType::TypeVar(0), LyraType::Integer),
            (LyraType::TypeVar(1), LyraType::List(Box::new(LyraType::TypeVar(0)))),
        ];
        
        let substitution = unify_many(&pairs).unwrap();
        assert_eq!(substitution.get(0), Some(LyraType::Integer));
        assert_eq!(substitution.get(1), Some(LyraType::List(Box::new(LyraType::Integer))));
    }
}