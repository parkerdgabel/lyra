use std::collections::{HashMap, HashSet};
use std::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod unification;
pub mod inference;
pub mod checker;
pub mod integration;
pub mod metadata;

pub use unification::*;
pub use inference::*;
pub use checker::*;
pub use integration::*;
pub use metadata::*;

/// Type variable identifier for generic types (α, β, γ, etc.)
pub type TypeVar = u32;

/// Function attributes for symbolic computation and optimization
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FunctionAttribute {
    /// Function holds its arguments (delayed evaluation)
    Hold,
    /// Function is listable (automatically threads over lists)
    Listable,
    /// Function is pure (no side effects)
    Pure,
    /// Function is associative: f[f[a,b],c] = f[a,f[b,c]]
    Associative,
    /// Function is commutative: f[a,b] = f[b,a]
    Commutative,
    /// Function has arbitrary precision
    NumericFunction,
    /// Function is protected (cannot be redefined)
    Protected,
    /// Custom attribute with name and optional parameters
    Custom(String, Vec<String>),
}

/// Type classes for constraining type variables (similar to Haskell type classes)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TypeClass {
    /// Numeric types (Integer, Real, Complex, Rational)
    Numeric,
    /// Ordered types (can be compared with <, >, etc.)
    Ordered,
    /// Equality comparable types
    Equatable, 
    /// Types that support addition
    Additive,
    /// Types that support multiplication
    Multiplicative,
    /// Types that can be iterated over
    Iterable,
    /// Types that can be indexed
    Indexable,
    /// Custom type class
    Custom(String),
}

/// Shape information for tensors
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape {
    pub dimensions: Vec<usize>,
}

impl TensorShape {
    pub fn new(dimensions: Vec<usize>) -> Self {
        TensorShape { dimensions }
    }
    
    pub fn scalar() -> Self {
        TensorShape { dimensions: vec![] }
    }
    
    pub fn vector(length: usize) -> Self {
        TensorShape { dimensions: vec![length] }
    }
    
    pub fn matrix(rows: usize, cols: usize) -> Self {
        TensorShape { dimensions: vec![rows, cols] }
    }
    
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }
    
    pub fn total_elements(&self) -> usize {
        self.dimensions.iter().product()
    }
    
    /// Check if two shapes are compatible for broadcasting
    pub fn broadcast_compatible(&self, other: &TensorShape) -> bool {
        let mut iter1 = self.dimensions.iter().rev();
        let mut iter2 = other.dimensions.iter().rev();
        
        loop {
            match (iter1.next(), iter2.next()) {
                (None, None) => return true,
                (Some(&d1), Some(&d2)) => {
                    if d1 != d2 && d1 != 1 && d2 != 1 {
                        return false;
                    }
                }
                (Some(&d), None) | (None, Some(&d)) => {
                    if d != 1 {
                        // Broadcasting is still possible
                    }
                }
            }
        }
    }
    
    /// Get the result shape after broadcasting
    pub fn broadcast_result(&self, other: &TensorShape) -> Option<TensorShape> {
        if !self.broadcast_compatible(other) {
            return None;
        }
        
        let max_rank = self.rank().max(other.rank());
        let mut result_dims = Vec::with_capacity(max_rank);
        
        for i in 0..max_rank {
            let dim1 = self.dimensions.get(self.rank().saturating_sub(i + 1)).copied().unwrap_or(1);
            let dim2 = other.dimensions.get(other.rank().saturating_sub(i + 1)).copied().unwrap_or(1);
            
            result_dims.push(dim1.max(dim2));
        }
        
        result_dims.reverse();
        Some(TensorShape::new(result_dims))
    }
}

/// Core type system for Lyra
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LyraType {
    /// Integer type
    Integer,
    /// Real number type  
    Real,
    /// Complex number type
    Complex,
    /// Rational number type
    Rational,
    /// String type
    String,
    /// Boolean type
    Boolean,
    /// Symbol type for symbolic computation
    Symbol,
    /// List type with element type
    List(Box<LyraType>),
    /// Tensor type with element type and shape
    Tensor {
        element_type: Box<LyraType>,
        shape: Option<TensorShape>,
    },
    /// Function type with parameter types and return type
    Function {
        params: Vec<LyraType>,
        return_type: Box<LyraType>,
        /// Function attributes for symbolic computation
        attributes: Vec<FunctionAttribute>,
    },
    /// Pattern type for pattern matching
    Pattern(Box<LyraType>),
    /// Rule type for transformation rules
    Rule {
        lhs_type: Box<LyraType>,
        rhs_type: Box<LyraType>,
    },
    /// Union type for sum types (T | U)
    Union(Vec<LyraType>),
    /// Tuple type for product types
    Tuple(Vec<LyraType>),
    /// Association type (key-value pairs)
    Association {
        key_type: Box<LyraType>,
        value_type: Box<LyraType>,
    },
    /// Type variable for generic types
    TypeVar(TypeVar),
    /// Constrained type variable with type class constraints
    ConstrainedTypeVar(TypeVar, Vec<TypeClass>),
    /// Unknown type for gradual typing
    Unknown,
    /// Any type (top type) 
    Any,
    /// Never type (bottom type)
    Never,
    /// Unit type for expressions that don't return a value
    Unit,
    /// Module type for namespacing
    Module(String),
    /// Effect type for algebraic effects (future)
    Effect(String),
    /// Custom type for user-defined types and aliases
    Custom(String),
}

impl LyraType {
    /// Check if this type is a numeric type
    pub fn is_numeric(&self) -> bool {
        matches!(self, 
            LyraType::Integer | 
            LyraType::Real | 
            LyraType::Complex | 
            LyraType::Rational
        )
    }
    
    /// Check if this type is a tensor type
    pub fn is_tensor(&self) -> bool {
        matches!(self, LyraType::Tensor { .. })
    }
    
    /// Check if this type is a function type
    pub fn is_function(&self) -> bool {
        matches!(self, LyraType::Function { .. })
    }
    
    /// Check if this type is a collection type
    pub fn is_collection(&self) -> bool {
        matches!(self, 
            LyraType::List(_) | 
            LyraType::Tuple(_) | 
            LyraType::Association { .. }
        )
    }
    
    /// Check if this type is gradually typed (Unknown or Any)
    pub fn is_gradual(&self) -> bool {
        matches!(self, LyraType::Unknown | LyraType::Any)
    }
    
    /// Check if this type satisfies a type class constraint
    pub fn satisfies_constraint(&self, constraint: &TypeClass) -> bool {
        match constraint {
            TypeClass::Numeric => self.is_numeric(),
            TypeClass::Ordered => matches!(self, 
                LyraType::Integer | LyraType::Real | 
                LyraType::String | LyraType::Boolean
            ),
            TypeClass::Equatable => !matches!(self, LyraType::Never),
            TypeClass::Additive => self.is_numeric() || matches!(self, 
                LyraType::String | LyraType::List(_)
            ),
            TypeClass::Multiplicative => self.is_numeric(),
            TypeClass::Iterable => matches!(self, 
                LyraType::List(_) | LyraType::Tuple(_) | 
                LyraType::Association { .. } | LyraType::String
            ),
            TypeClass::Indexable => matches!(self, 
                LyraType::List(_) | LyraType::Tuple(_) | 
                LyraType::Association { .. } | LyraType::String |
                LyraType::Tensor { .. }
            ),
            TypeClass::Custom(_) => false, // TODO: implement custom type classes
        }
    }
    
    /// Check if this type contains type variables
    pub fn contains_type_vars(&self) -> bool {
        match self {
            LyraType::TypeVar(_) | LyraType::ConstrainedTypeVar(_, _) => true,
            LyraType::List(elem_type) => elem_type.contains_type_vars(),
            LyraType::Tensor { element_type, .. } => element_type.contains_type_vars(),
            LyraType::Function { params, return_type, .. } => {
                params.iter().any(|p| p.contains_type_vars()) || return_type.contains_type_vars()
            }
            LyraType::Pattern(inner) => inner.contains_type_vars(),
            LyraType::Rule { lhs_type, rhs_type } => {
                lhs_type.contains_type_vars() || rhs_type.contains_type_vars()
            }
            LyraType::Union(types) | LyraType::Tuple(types) => {
                types.iter().any(|t| t.contains_type_vars())
            }
            LyraType::Association { key_type, value_type } => {
                key_type.contains_type_vars() || value_type.contains_type_vars()
            }
            LyraType::Custom(_) => false,
            _ => false,
        }
    }
    
    /// Get all type variables used in this type
    pub fn type_vars(&self) -> HashSet<TypeVar> {
        let mut vars = HashSet::new();
        self.collect_type_vars(&mut vars);
        vars
    }
    
    fn collect_type_vars(&self, vars: &mut HashSet<TypeVar>) {
        match self {
            LyraType::TypeVar(var) | LyraType::ConstrainedTypeVar(var, _) => {
                vars.insert(*var);
            }
            LyraType::List(elem_type) => elem_type.collect_type_vars(vars),
            LyraType::Tensor { element_type, .. } => element_type.collect_type_vars(vars),
            LyraType::Function { params, return_type, .. } => {
                for param in params {
                    param.collect_type_vars(vars);
                }
                return_type.collect_type_vars(vars);
            }
            LyraType::Pattern(inner) => inner.collect_type_vars(vars),
            LyraType::Rule { lhs_type, rhs_type } => {
                lhs_type.collect_type_vars(vars);
                rhs_type.collect_type_vars(vars);
            }
            LyraType::Union(types) | LyraType::Tuple(types) => {
                for ty in types {
                    ty.collect_type_vars(vars);
                }
            }
            LyraType::Association { key_type, value_type } => {
                key_type.collect_type_vars(vars);
                value_type.collect_type_vars(vars);
            }
            LyraType::Custom(_) => {
                // Custom types don't contain type variables directly
            }
            _ => {}
        }
    }
    
    /// Apply a type substitution to this type
    pub fn substitute(&self, substitution: &TypeSubstitution) -> LyraType {
        match self {
            LyraType::TypeVar(var) => {
                substitution.get(*var).unwrap_or_else(|| self.clone())
            }
            LyraType::ConstrainedTypeVar(var, constraints) => {
                substitution.get(*var).unwrap_or_else(|| 
                    LyraType::ConstrainedTypeVar(*var, constraints.clone())
                )
            }
            LyraType::List(elem_type) => {
                LyraType::List(Box::new(elem_type.substitute(substitution)))
            }
            LyraType::Tensor { element_type, shape } => LyraType::Tensor {
                element_type: Box::new(element_type.substitute(substitution)),
                shape: shape.clone(),
            },
            LyraType::Function { params, return_type, attributes } => LyraType::Function {
                params: params.iter().map(|p| p.substitute(substitution)).collect(),
                return_type: Box::new(return_type.substitute(substitution)),
                attributes: attributes.clone(),
            },
            LyraType::Pattern(inner) => {
                LyraType::Pattern(Box::new(inner.substitute(substitution)))
            }
            LyraType::Rule { lhs_type, rhs_type } => LyraType::Rule {
                lhs_type: Box::new(lhs_type.substitute(substitution)),
                rhs_type: Box::new(rhs_type.substitute(substitution)),
            },
            LyraType::Union(types) => LyraType::Union(
                types.iter().map(|t| t.substitute(substitution)).collect()
            ),
            LyraType::Tuple(types) => LyraType::Tuple(
                types.iter().map(|t| t.substitute(substitution)).collect()
            ),
            LyraType::Association { key_type, value_type } => LyraType::Association {
                key_type: Box::new(key_type.substitute(substitution)),
                value_type: Box::new(value_type.substitute(substitution)),
            },
            LyraType::Custom(_) => self.clone(),
            _ => self.clone(),
        }
    }
    
    /// Create a type scheme from this type by quantifying over all type variables
    pub fn generalize(&self, environment: &TypeEnvironment) -> TypeScheme {
        let type_vars = self.type_vars();
        let env_vars = environment.type_vars();
        let quantified_vars: Vec<TypeVar> = type_vars.difference(&env_vars).copied().collect();
        
        TypeScheme {
            quantified_vars,
            body: self.clone(),
        }
    }
}

impl fmt::Display for LyraType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LyraType::Integer => write!(f, "Integer"),
            LyraType::Real => write!(f, "Real"),
            LyraType::Complex => write!(f, "Complex"),
            LyraType::Rational => write!(f, "Rational"),
            LyraType::String => write!(f, "String"),
            LyraType::Boolean => write!(f, "Boolean"),
            LyraType::Symbol => write!(f, "Symbol"),
            LyraType::List(elem_type) => write!(f, "List[{}]", elem_type),
            LyraType::Tensor { element_type, shape } => {
                if let Some(shape) = shape {
                    write!(f, "Tensor[{}, {:?}]", element_type, shape.dimensions)
                } else {
                    write!(f, "Tensor[{}]", element_type)
                }
            }
            LyraType::Function { params, return_type, .. } => {
                write!(f, "(")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", param)?;
                }
                write!(f, ") -> {}", return_type)
            }
            LyraType::Pattern(inner) => write!(f, "Pattern[{}]", inner),
            LyraType::Rule { lhs_type, rhs_type } => {
                write!(f, "Rule[{} -> {}]", lhs_type, rhs_type)
            }
            LyraType::Union(types) => {
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 { write!(f, " | ")?; }
                    write!(f, "{}", ty)?;
                }
                Ok(())
            }
            LyraType::Tuple(types) => {
                write!(f, "(")?;
                for (i, ty) in types.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", ty)?;
                }
                write!(f, ")")
            }
            LyraType::Association { key_type, value_type } => {
                write!(f, "Association[{}, {}]", key_type, value_type)
            }
            LyraType::TypeVar(var) => write!(f, "α{}", var),
            LyraType::ConstrainedTypeVar(var, constraints) => {
                write!(f, "α{}", var)?;
                if !constraints.is_empty() {
                    write!(f, " where ")?;
                    for (i, constraint) in constraints.iter().enumerate() {
                        if i > 0 { write!(f, " & ")?; }
                        write!(f, "{:?}", constraint)?;
                    }
                }
                Ok(())
            }
            LyraType::Unknown => write!(f, "?"),
            LyraType::Any => write!(f, "Any"),
            LyraType::Never => write!(f, "Never"),
            LyraType::Unit => write!(f, "()"),
            LyraType::Module(name) => write!(f, "Module[{}]", name),
            LyraType::Effect(name) => write!(f, "Effect[{}]", name),
            LyraType::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// Type scheme for polymorphic types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TypeScheme {
    pub quantified_vars: Vec<TypeVar>,
    pub body: LyraType,
}

impl TypeScheme {
    pub fn new(quantified_vars: Vec<TypeVar>, body: LyraType) -> Self {
        TypeScheme { quantified_vars, body }
    }
    
    /// Create a monomorphic type scheme
    pub fn monomorphic(ty: LyraType) -> Self {
        TypeScheme {
            quantified_vars: Vec::new(),
            body: ty,
        }
    }
    
    /// Instantiate this type scheme with fresh type variables
    pub fn instantiate(&self, var_gen: &mut TypeVarGenerator) -> LyraType {
        if self.quantified_vars.is_empty() {
            return self.body.clone();
        }
        
        let mut substitution = TypeSubstitution::new();
        for &var in &self.quantified_vars {
            let fresh_var = var_gen.next();
            substitution.insert(var, LyraType::TypeVar(fresh_var));
        }
        
        self.body.substitute(&substitution)
    }
}

impl fmt::Display for TypeScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.quantified_vars.is_empty() {
            write!(f, "{}", self.body)
        } else {
            write!(f, "∀")?;
            for (i, var) in self.quantified_vars.iter().enumerate() {
                if i > 0 {
                    write!(f, " ")?;
                }
                write!(f, "α{}", var)?;
            }
            write!(f, ". {}", self.body)
        }
    }
}

/// Type substitution mapping type variables to types
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeSubstitution {
    mapping: HashMap<TypeVar, LyraType>,
}

impl TypeSubstitution {
    pub fn new() -> Self {
        TypeSubstitution {
            mapping: HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, var: TypeVar, ty: LyraType) {
        self.mapping.insert(var, ty);
    }
    
    pub fn get(&self, var: TypeVar) -> Option<LyraType> {
        self.mapping.get(&var).cloned()
    }
    
    pub fn contains(&self, var: TypeVar) -> bool {
        self.mapping.contains_key(&var)
    }
    
    pub fn is_empty(&self) -> bool {
        self.mapping.is_empty()
    }
    
    /// Compose two substitutions: apply `other` first, then `self`
    pub fn compose(&self, other: &TypeSubstitution) -> TypeSubstitution {
        let mut result = TypeSubstitution::new();
        
        // Apply self to other's mappings (other is applied first)
        for (&var, ty) in &other.mapping {
            result.insert(var, ty.substitute(self));
        }
        
        // Add self's mappings (self is applied second)
        for (&var, ty) in &self.mapping {
            result.insert(var, ty.clone());
        }
        
        result
    }
    
    /// Apply this substitution to a type
    pub fn apply(&self, ty: &LyraType) -> LyraType {
        ty.substitute(self)
    }
}

/// Type environment for variable bindings
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TypeEnvironment {
    bindings: HashMap<String, TypeScheme>,
}

impl TypeEnvironment {
    pub fn new() -> Self {
        TypeEnvironment {
            bindings: HashMap::new(),
        }
    }
    
    pub fn insert(&mut self, name: String, scheme: TypeScheme) {
        self.bindings.insert(name, scheme);
    }
    
    pub fn get(&self, name: &str) -> Option<&TypeScheme> {
        self.bindings.get(name)
    }
    
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
    
    /// Get all type variables used in this environment
    pub fn type_vars(&self) -> HashSet<TypeVar> {
        let mut vars = HashSet::new();
        for scheme in self.bindings.values() {
            scheme.body.collect_type_vars(&mut vars);
        }
        vars
    }
    
    /// Apply a substitution to all types in this environment
    pub fn substitute(&self, substitution: &TypeSubstitution) -> TypeEnvironment {
        let mut result = TypeEnvironment::new();
        for (name, scheme) in &self.bindings {
            let new_body = scheme.body.substitute(substitution);
            result.insert(name.clone(), TypeScheme::new(scheme.quantified_vars.clone(), new_body));
        }
        result
    }
    
    /// Create a new environment extending this one
    pub fn extend(&self, name: String, scheme: TypeScheme) -> TypeEnvironment {
        let mut result = self.clone();
        result.insert(name, scheme);
        result
    }
}

/// Type variable generator for creating fresh type variables
#[derive(Debug, Clone)]
pub struct TypeVarGenerator {
    counter: TypeVar,
}

impl TypeVarGenerator {
    pub fn new() -> Self {
        TypeVarGenerator { counter: 0 }
    }
    
    pub fn next(&mut self) -> TypeVar {
        let var = self.counter;
        self.counter += 1;
        var
    }
    
    pub fn fresh_type(&mut self) -> LyraType {
        LyraType::TypeVar(self.next())
    }
}

impl Default for TypeVarGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Type constraint for the constraint solver
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeConstraint {
    /// Equality constraint: t1 = t2
    Equal(LyraType, LyraType),
    /// Subtype constraint: t1 <: t2 (for future use)
    Subtype(LyraType, LyraType),
    /// Instance constraint: t is an instance of scheme
    Instance(LyraType, TypeScheme),
}

/// Type errors that can occur during inference
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum TypeError {
    #[error("Type mismatch: expected {expected}, found {actual}")]
    Mismatch { expected: LyraType, actual: LyraType },
    
    #[error("Occurs check failed: type variable α{var} occurs in {ty}")]
    OccursCheck { var: TypeVar, ty: LyraType },
    
    #[error("Unbound variable: {name}")]
    UnboundVariable { name: String },
    
    #[error("Function arity mismatch: expected {expected} arguments, found {actual}")]
    ArityMismatch { expected: usize, actual: usize },
    
    #[error("Shape mismatch: cannot broadcast {shape1:?} with {shape2:?}")]
    ShapeMismatch { shape1: TensorShape, shape2: TensorShape },
    
    #[error("Invalid operation: {operation} not supported for types {left} and {right}")]
    InvalidOperation { operation: String, left: LyraType, right: LyraType },
    
    #[error("Ambiguous type: could not infer type for expression")]
    AmbiguousType,
    
    #[error("Infinite type: {description}")]
    InfiniteType { description: String },
}

pub type TypeResult<T> = Result<T, TypeError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_types() {
        assert!(LyraType::Integer.is_numeric());
        assert!(LyraType::Real.is_numeric());
        assert!(!LyraType::String.is_numeric());
        
        let tensor = LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: Some(TensorShape::vector(10)),
        };
        assert!(tensor.is_tensor());
        
        let function = LyraType::Function {
            params: vec![LyraType::Integer, LyraType::Real],
            return_type: Box::new(LyraType::Real),
            attributes: vec![],
        };
        assert!(function.is_function());
    }

    #[test]
    fn test_type_vars() {
        let mut gen = TypeVarGenerator::new();
        let var1 = gen.next();
        let var2 = gen.next();
        assert_ne!(var1, var2);
        
        let ty = LyraType::Function {
            params: vec![LyraType::TypeVar(var1), LyraType::Integer],
            return_type: Box::new(LyraType::TypeVar(var2)),
            attributes: vec![],
        };
        
        assert!(ty.contains_type_vars());
        let vars = ty.type_vars();
        assert!(vars.contains(&var1));
        assert!(vars.contains(&var2));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_type_substitution() {
        let mut substitution = TypeSubstitution::new();
        substitution.insert(0, LyraType::Integer);
        substitution.insert(1, LyraType::Real);
        
        let ty = LyraType::Function {
            params: vec![LyraType::TypeVar(0)],
            return_type: Box::new(LyraType::TypeVar(1)),
            attributes: vec![],
        };
        
        let result = ty.substitute(&substitution);
        let expected = LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
            attributes: vec![],
        };
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_tensor_shapes() {
        let shape1 = TensorShape::vector(10);
        let shape2 = TensorShape::matrix(3, 4);
        
        assert_eq!(shape1.rank(), 1);
        assert_eq!(shape2.rank(), 2);
        assert_eq!(shape1.total_elements(), 10);
        assert_eq!(shape2.total_elements(), 12);
        
        // Test broadcasting compatibility
        let scalar = TensorShape::scalar();
        assert!(scalar.broadcast_compatible(&shape1));
        assert!(shape1.broadcast_compatible(&scalar));
        
        let shape3 = TensorShape::new(vec![1, 4]);
        let shape4 = TensorShape::new(vec![3, 1]);
        assert!(shape3.broadcast_compatible(&shape4));
        
        let result = shape3.broadcast_result(&shape4).unwrap();
        assert_eq!(result.dimensions, vec![3, 4]);
    }

    #[test]
    fn test_type_environment() {
        let mut env = TypeEnvironment::new();
        let scheme = TypeScheme::monomorphic(LyraType::Integer);
        
        env.insert("x".to_string(), scheme.clone());
        assert!(env.contains("x"));
        assert_eq!(env.get("x"), Some(&scheme));
        assert!(!env.contains("y"));
        
        let extended = env.extend("y".to_string(), TypeScheme::monomorphic(LyraType::Real));
        assert!(extended.contains("x"));
        assert!(extended.contains("y"));
    }

    #[test]
    fn test_type_scheme() {
        let mut gen = TypeVarGenerator::new();
        let var = gen.next();
        
        let scheme = TypeScheme::new(
            vec![var],
            LyraType::Function {
                params: vec![LyraType::TypeVar(var)],
                return_type: Box::new(LyraType::TypeVar(var)),
                attributes: vec![],
            }
        );
        
        let instance1 = scheme.instantiate(&mut gen);
        let instance2 = scheme.instantiate(&mut gen);
        
        // Should be structurally the same but with different type variables
        assert_ne!(instance1, instance2);
        assert!(instance1.contains_type_vars());
        assert!(instance2.contains_type_vars());
    }

    #[test]
    fn test_substitution_composition() {
        let mut sub1 = TypeSubstitution::new();
        sub1.insert(0, LyraType::TypeVar(1));
        sub1.insert(2, LyraType::Integer);
        
        let mut sub2 = TypeSubstitution::new();
        sub2.insert(1, LyraType::Real);
        sub2.insert(3, LyraType::String);
        
        let composed = sub1.compose(&sub2);
        
        // sub2 first, then sub1: 
        // 1 -> Real (from sub2, then sub1 applied: Real)
        // 3 -> String (from sub2, then sub1 applied: String)  
        // 0 -> TypeVar(1) (from sub1)
        // 2 -> Integer (from sub1)
        assert_eq!(composed.get(0), Some(LyraType::TypeVar(1)));
        assert_eq!(composed.get(1), Some(LyraType::Real));
        assert_eq!(composed.get(2), Some(LyraType::Integer));
        assert_eq!(composed.get(3), Some(LyraType::String));
    }

    #[test]
    fn test_type_display() {
        let integer = LyraType::Integer;
        assert_eq!(format!("{}", integer), "Integer");
        
        let list = LyraType::List(Box::new(LyraType::Real));
        assert_eq!(format!("{}", list), "List[Real]");
        
        let function = LyraType::Function {
            params: vec![LyraType::Integer, LyraType::Real],
            return_type: Box::new(LyraType::Boolean),
            attributes: vec![],
        };
        assert_eq!(format!("{}", function), "(Integer, Real) -> Boolean");
        
        let type_var = LyraType::TypeVar(42);
        assert_eq!(format!("{}", type_var), "α42");
    }
}