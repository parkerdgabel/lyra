# Type System API Documentation

## Overview

Lyra features a sophisticated gradual type system that combines static type checking with runtime flexibility. The type system supports type inference, pattern matching integration, constraint solving, and symbolic computation. This document provides comprehensive API documentation for developers working with Lyra's type system.

## Architecture

The type system is built on several core components:

```rust
// Core type representation
pub enum LyraType {
    // Primitive types
    Integer, Real, Complex, Rational, String, Boolean, Symbol,
    
    // Collection types
    List(Box<LyraType>),
    Tuple(Vec<LyraType>),
    Association { key_type: Box<LyraType>, value_type: Box<LyraType> },
    
    // Tensor types with shape information
    Tensor { element_type: Box<LyraType>, shape: Option<TensorShape> },
    
    // Function types with attributes
    Function { 
        params: Vec<LyraType>, 
        return_type: Box<LyraType>, 
        attributes: Vec<FunctionAttribute> 
    },
    
    // Pattern and rule types
    Pattern(Box<LyraType>),
    Rule { lhs_type: Box<LyraType>, rhs_type: Box<LyraType> },
    
    // Type variables for generics
    TypeVar(TypeVar),
    ConstrainedTypeVar(TypeVar, Vec<TypeClass>),
    
    // Gradual typing
    Unknown,  // Gradually typed
    Any,      // Top type
    Never,    // Bottom type
    Unit,     // Unit type
    
    // Advanced types
    Union(Vec<LyraType>),     // Sum types
    Module(String),           // Module types
    Effect(String),           // Effect types (future)
    Custom(String),           // User-defined types
}
```

## Type Construction

### Basic Types

```rust
use crate::types::LyraType;

// Primitive types
let int_type = LyraType::Integer;
let real_type = LyraType::Real;
let string_type = LyraType::String;
let bool_type = LyraType::Boolean;
let symbol_type = LyraType::Symbol;

// Gradual types
let unknown_type = LyraType::Unknown;  // For gradual typing
let any_type = LyraType::Any;         // Top type
let never_type = LyraType::Never;     // Bottom type
let unit_type = LyraType::Unit;       // No return value
```

### Collection Types

```rust
// List types
let int_list = LyraType::List(Box::new(LyraType::Integer));
let string_list = LyraType::List(Box::new(LyraType::String));

// Nested lists
let matrix_type = LyraType::List(Box::new(
    LyraType::List(Box::new(LyraType::Real))
));

// Tuple types
let point_2d = LyraType::Tuple(vec![LyraType::Real, LyraType::Real]);
let mixed_tuple = LyraType::Tuple(vec![
    LyraType::String,
    LyraType::Integer,
    LyraType::Boolean
]);

// Association types (key-value pairs)
let string_to_int = LyraType::Association {
    key_type: Box::new(LyraType::String),
    value_type: Box::new(LyraType::Integer),
};
```

### Tensor Types

```rust
use crate::types::TensorShape;

// Tensor with known shape
let vector_type = LyraType::Tensor {
    element_type: Box::new(LyraType::Real),
    shape: Some(TensorShape::vector(100)),
};

let matrix_type = LyraType::Tensor {
    element_type: Box::new(LyraType::Complex),
    shape: Some(TensorShape::matrix(28, 28)),
};

// Tensor with unknown shape
let flexible_tensor = LyraType::Tensor {
    element_type: Box::new(LyraType::Integer),
    shape: None,
};

// Higher-dimensional tensors
let tensor_3d = LyraType::Tensor {
    element_type: Box::new(LyraType::Real),
    shape: Some(TensorShape::new(vec![10, 20, 30])),
};
```

### Function Types

```rust
use crate::types::FunctionAttribute;

// Simple function: Integer -> Integer
let simple_fn = LyraType::Function {
    params: vec![LyraType::Integer],
    return_type: Box::new(LyraType::Integer),
    attributes: vec![],
};

// Binary operation: (Real, Real) -> Real
let binary_op = LyraType::Function {
    params: vec![LyraType::Real, LyraType::Real],
    return_type: Box::new(LyraType::Real),
    attributes: vec![
        FunctionAttribute::Associative,
        FunctionAttribute::Commutative,
    ],
};

// Listable function with Hold attribute
let listable_fn = LyraType::Function {
    params: vec![LyraType::Any],
    return_type: Box::new(LyraType::Any),
    attributes: vec![
        FunctionAttribute::Hold,
        FunctionAttribute::Listable,
    ],
};

// Higher-order function: (Integer -> Real) -> List[Real]
let higher_order = LyraType::Function {
    params: vec![
        LyraType::Function {
            params: vec![LyraType::Integer],
            return_type: Box::new(LyraType::Real),
            attributes: vec![],
        }
    ],
    return_type: Box::new(LyraType::List(Box::new(LyraType::Real))),
    attributes: vec![],
};
```

### Generic Types with Type Variables

```rust
use crate::types::{TypeVar, TypeVarGenerator, TypeClass};

let mut gen = TypeVarGenerator::new();

// Generic identity function: α -> α
let var_a = gen.next();
let identity_type = LyraType::Function {
    params: vec![LyraType::TypeVar(var_a)],
    return_type: Box::new(LyraType::TypeVar(var_a)),
    attributes: vec![],
};

// Constrained generic: where α: Numeric => α -> α  
let var_b = gen.next();
let numeric_fn = LyraType::Function {
    params: vec![LyraType::ConstrainedTypeVar(var_b, vec![TypeClass::Numeric])],
    return_type: Box::new(LyraType::TypeVar(var_b)),
    attributes: vec![],
};

// Generic list operations: where α: Equatable => List[α] -> Boolean
let var_c = gen.next();
let list_predicate = LyraType::Function {
    params: vec![LyraType::List(Box::new(
        LyraType::ConstrainedTypeVar(var_c, vec![TypeClass::Equatable])
    ))],
    return_type: Box::new(LyraType::Boolean),
    attributes: vec![],
};
```

## Type Checking and Inference

### Type Environment

```rust
use crate::types::{TypeEnvironment, TypeScheme};

// Create a type environment
let mut env = TypeEnvironment::new();

// Add variable bindings
env.insert("x".to_string(), TypeScheme::monomorphic(LyraType::Integer));
env.insert("f".to_string(), TypeScheme::new(
    vec![0, 1],  // Quantified variables α0, α1
    LyraType::Function {
        params: vec![LyraType::TypeVar(0)],
        return_type: Box::new(LyraType::TypeVar(1)),
        attributes: vec![],
    }
));

// Query the environment
if let Some(scheme) = env.get("f") {
    println!("Type of f: {}", scheme);
}

// Check if variable is bound
assert!(env.contains("x"));
assert!(!env.contains("y"));
```

### Type Inference

```rust
use crate::types::{TypeInferenceEngine, TypeConstraint};

// Create inference engine
let mut inference = TypeInferenceEngine::new();

// Infer types for expressions
let expr = /* some AST expression */;
let inferred_type = inference.infer_type(&expr, &env)?;

// Add constraints
inference.add_constraint(TypeConstraint::Equal(
    LyraType::TypeVar(0),
    LyraType::Integer
));

// Solve constraints
let solution = inference.solve_constraints()?;
let final_type = solution.apply(&inferred_type);
```

### Type Checking

```rust
use crate::types::TypeChecker;

let checker = TypeChecker::new();

// Check if two types are compatible
let can_assign = checker.is_assignable(&LyraType::Integer, &LyraType::Real)?;
assert!(can_assign);  // Integer can be assigned to Real

// Check function application
let function_type = LyraType::Function {
    params: vec![LyraType::Integer, LyraType::String],
    return_type: Box::new(LyraType::Boolean),
    attributes: vec![],
};

let arg_types = vec![LyraType::Integer, LyraType::String];
let result_type = checker.check_application(&function_type, &arg_types)?;
assert_eq!(result_type, LyraType::Boolean);
```

## Type Operations

### Type Queries

```rust
impl LyraType {
    // Type classification
    pub fn is_numeric(&self) -> bool {
        matches!(self, 
            LyraType::Integer | 
            LyraType::Real | 
            LyraType::Complex | 
            LyraType::Rational
        )
    }
    
    pub fn is_collection(&self) -> bool {
        matches!(self, 
            LyraType::List(_) | 
            LyraType::Tuple(_) | 
            LyraType::Association { .. }
        )
    }
    
    pub fn is_tensor(&self) -> bool {
        matches!(self, LyraType::Tensor { .. })
    }
    
    pub fn is_function(&self) -> bool {
        matches!(self, LyraType::Function { .. })
    }
    
    pub fn is_gradual(&self) -> bool {
        matches!(self, LyraType::Unknown | LyraType::Any)
    }
    
    pub fn contains_type_vars(&self) -> bool {
        matches!(self, LyraType::TypeVar(_) | LyraType::ConstrainedTypeVar(_, _))
        // ... or recursively check nested types
    }
}

// Usage examples
let tensor_type = LyraType::Tensor {
    element_type: Box::new(LyraType::Real),
    shape: Some(TensorShape::matrix(3, 3)),
};

assert!(tensor_type.is_tensor());
assert!(!tensor_type.is_numeric());
assert!(!tensor_type.contains_type_vars());
```

### Type Constraints

```rust
use crate::types::TypeClass;

impl LyraType {
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
            TypeClass::Custom(name) => {
                // Check custom type class registry
                false  // Placeholder
            },
        }
    }
}

// Usage
let list_type = LyraType::List(Box::new(LyraType::Integer));
assert!(list_type.satisfies_constraint(&TypeClass::Iterable));
assert!(list_type.satisfies_constraint(&TypeClass::Indexable));
assert!(!list_type.satisfies_constraint(&TypeClass::Numeric));
```

### Type Substitution

```rust
use crate::types::TypeSubstitution;

// Create a substitution
let mut subst = TypeSubstitution::new();
subst.insert(0, LyraType::Integer);  // α0 := Integer
subst.insert(1, LyraType::String);   // α1 := String

// Apply substitution to a type
let generic_type = LyraType::Function {
    params: vec![LyraType::TypeVar(0)],
    return_type: Box::new(LyraType::TypeVar(1)),
    attributes: vec![],
};

let concrete_type = generic_type.substitute(&subst);
// Result: Integer -> String

// Compose substitutions
let mut subst2 = TypeSubstitution::new();
subst2.insert(2, LyraType::Real);

let composed = subst.compose(&subst2);
```

## Tensor Shape System

### Shape Operations

```rust
use crate::types::TensorShape;

// Create shapes
let vector = TensorShape::vector(10);
let matrix = TensorShape::matrix(3, 4);
let tensor_3d = TensorShape::new(vec![2, 3, 4]);

// Shape queries
assert_eq!(vector.rank(), 1);
assert_eq!(matrix.rank(), 2);
assert_eq!(tensor_3d.total_elements(), 24);

// Broadcasting compatibility
let shape1 = TensorShape::new(vec![3, 1]);
let shape2 = TensorShape::new(vec![1, 4]);

assert!(shape1.broadcast_compatible(&shape2));

if let Some(result_shape) = shape1.broadcast_result(&shape2) {
    assert_eq!(result_shape.dimensions, vec![3, 4]);
}
```

### Tensor Type Operations

```rust
// Create tensor types with shape constraints
fn create_matrix_multiply_signature() -> LyraType {
    LyraType::Function {
        params: vec![
            LyraType::Tensor {
                element_type: Box::new(LyraType::Real),
                shape: Some(TensorShape::new(vec![0, 0])), // Symbolic dimensions
            },
            LyraType::Tensor {
                element_type: Box::new(LyraType::Real),
                shape: Some(TensorShape::new(vec![0, 0])),
            },
        ],
        return_type: Box::new(LyraType::Tensor {
            element_type: Box::new(LyraType::Real),
            shape: None,  // Result shape computed at runtime
        }),
        attributes: vec![],
    }
}
```

## Pattern Matching Integration

### Pattern Types

```rust
use crate::ast::Pattern;

// Convert patterns to types for type checking
fn pattern_to_type(pattern: &Pattern) -> LyraType {
    match pattern {
        Pattern::Blank { head: None } => LyraType::Any,
        Pattern::Blank { head: Some(type_name) } => {
            match type_name.as_str() {
                "Integer" => LyraType::Integer,
                "Real" => LyraType::Real,
                "String" => LyraType::String,
                _ => LyraType::Custom(type_name.clone()),
            }
        }
        Pattern::BlankSequence { head } => {
            let element_type = head.as_ref()
                .map(|h| pattern_to_type(&Pattern::Blank { head: Some(h.clone()) }))
                .unwrap_or(LyraType::Any);
            LyraType::List(Box::new(element_type))
        }
        Pattern::Named { name: _, pattern } => pattern_to_type(pattern),
        Pattern::Typed { name: _, type_pattern } => {
            // Convert type expression to LyraType
            expr_to_type(type_pattern)
        }
        // ... other pattern types
    }
}

// Type-aware pattern matching
fn check_pattern_match(pattern: &Pattern, value_type: &LyraType) -> bool {
    let pattern_type = pattern_to_type(pattern);
    // Check if value_type is assignable to pattern_type
    type_checker.is_assignable(value_type, &pattern_type)
}
```

### Rule Type Checking

```rust
// Check that rule types are consistent
fn check_rule_type(lhs_type: &LyraType, rhs_type: &LyraType) -> Result<(), TypeError> {
    // The RHS type should be assignable to what the LHS expects
    if !type_checker.is_assignable(rhs_type, lhs_type) {
        return Err(TypeError::Mismatch {
            expected: lhs_type.clone(),
            actual: rhs_type.clone(),
        });
    }
    Ok(())
}

// Rule type construction
let rule_type = LyraType::Rule {
    lhs_type: Box::new(LyraType::Pattern(Box::new(LyraType::Integer))),
    rhs_type: Box::new(LyraType::Integer),
};
```

## Advanced Type Features

### Union Types

```rust
// Create union types for sum types
let int_or_string = LyraType::Union(vec![
    LyraType::Integer,
    LyraType::String,
]);

let optional_int = LyraType::Union(vec![
    LyraType::Integer,
    LyraType::Unit,  // Represents None/null
]);

// Type operations on unions
impl LyraType {
    pub fn union_contains(&self, other: &LyraType) -> bool {
        match self {
            LyraType::Union(types) => types.contains(other),
            _ => self == other,
        }
    }
    
    pub fn union_with(self, other: LyraType) -> LyraType {
        match (self, other) {
            (LyraType::Union(mut types), other_type) => {
                if !types.contains(&other_type) {
                    types.push(other_type);
                }
                LyraType::Union(types)
            }
            (self_type, LyraType::Union(mut types)) => {
                if !types.contains(&self_type) {
                    types.insert(0, self_type);
                }
                LyraType::Union(types)
            }
            (self_type, other_type) if self_type == other_type => self_type,
            (self_type, other_type) => {
                LyraType::Union(vec![self_type, other_type])
            }
        }
    }
}
```

### Custom Types and Aliases

```rust
// Define custom types
pub struct CustomTypeRegistry {
    types: HashMap<String, LyraType>,
    aliases: HashMap<String, LyraType>,
}

impl CustomTypeRegistry {
    pub fn register_type(&mut self, name: String, definition: LyraType) {
        self.types.insert(name, definition);
    }
    
    pub fn register_alias(&mut self, alias: String, target: LyraType) {
        self.aliases.insert(alias, target);
    }
    
    pub fn resolve_type(&self, name: &str) -> Option<&LyraType> {
        self.types.get(name)
            .or_else(|| self.aliases.get(name))
    }
}

// Usage
let mut registry = CustomTypeRegistry::new();

// Register a Point2D type alias
registry.register_alias("Point2D".to_string(), 
    LyraType::Tuple(vec![LyraType::Real, LyraType::Real])
);

// Register a custom algebraic data type
registry.register_type("Maybe".to_string(),
    LyraType::Union(vec![
        LyraType::Unit,     // Nothing
        LyraType::TypeVar(0) // Just(α)
    ])
);
```

### Effect Types (Future Feature)

```rust
// Effect types for tracking computational effects
let io_effect = LyraType::Effect("IO".to_string());
let async_effect = LyraType::Effect("Async".to_string());

// Function with effects: Integer -> IO[String]
let io_function = LyraType::Function {
    params: vec![LyraType::Integer],
    return_type: Box::new(LyraType::Custom("IO[String]".to_string())),
    attributes: vec![],
};
```

## Type System Integration

### With VM Values

```rust
use crate::vm::Value;

// Infer types from runtime values
fn infer_value_type(value: &Value) -> LyraType {
    match value {
        Value::Integer(_) => LyraType::Integer,
        Value::Real(_) => LyraType::Real,
        Value::String(_) => LyraType::String,
        Value::Symbol(_) => LyraType::Symbol,
        Value::Boolean(_) => LyraType::Boolean,
        Value::Missing => LyraType::Unit,
        Value::List(items) => {
            if items.is_empty() {
                LyraType::List(Box::new(LyraType::Unknown))
            } else {
                // Infer element type from first element (or union of all)
                let element_type = infer_value_type(&items[0]);
                LyraType::List(Box::new(element_type))
            }
        }
        Value::LyObj(obj) => LyraType::Custom(obj.type_name().to_string()),
        // ... other value types
        _ => LyraType::Unknown,
    }
}

// Type-guided value construction
fn construct_typed_value(ty: &LyraType, default: bool) -> Option<Value> {
    match ty {
        LyraType::Integer => Some(if default { Value::Integer(0) } else { Value::Missing }),
        LyraType::Real => Some(if default { Value::Real(0.0) } else { Value::Missing }),
        LyraType::String => Some(if default { Value::String(String::new()) } else { Value::Missing }),
        LyraType::Boolean => Some(if default { Value::Boolean(false) } else { Value::Missing }),
        LyraType::List(elem_type) => {
            if default {
                Some(Value::List(Vec::new()))
            } else {
                Some(Value::Missing)
            }
        }
        _ => None,
    }
}
```

### With Stdlib Functions

```rust
// Type signatures for stdlib functions
use crate::stdlib::StdlibFunction;

pub struct TypedStdlibFunction {
    pub name: &'static str,
    pub function: StdlibFunction,
    pub type_signature: LyraType,
}

// Define typed stdlib functions
pub fn get_typed_stdlib_functions() -> Vec<TypedStdlibFunction> {
    vec![
        TypedStdlibFunction {
            name: "Plus",
            function: stdlib_plus,
            type_signature: LyraType::Function {
                params: vec![LyraType::TypeVar(0), LyraType::TypeVar(0)],
                return_type: Box::new(LyraType::TypeVar(0)),
                attributes: vec![
                    FunctionAttribute::Associative,
                    FunctionAttribute::Commutative,
                    FunctionAttribute::Listable,
                ],
            },
        },
        TypedStdlibFunction {
            name: "Map",
            function: stdlib_map,
            type_signature: LyraType::Function {
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
            },
        },
        // ... more functions
    ]
}
```

## Error Handling

### Type Errors

```rust
use crate::types::TypeError;

// Common type error patterns
fn handle_type_error(result: Result<LyraType, TypeError>) {
    match result {
        Ok(ty) => println!("Inferred type: {}", ty),
        Err(TypeError::Mismatch { expected, actual }) => {
            eprintln!("Type mismatch: expected {}, got {}", expected, actual);
        }
        Err(TypeError::UnboundVariable { name }) => {
            eprintln!("Unbound variable: {}", name);
        }
        Err(TypeError::ArityMismatch { expected, actual }) => {
            eprintln!("Wrong number of arguments: expected {}, got {}", expected, actual);
        }
        Err(TypeError::OccursCheck { var, ty }) => {
            eprintln!("Infinite type: α{} occurs in {}", var, ty);
        }
        Err(TypeError::ShapeMismatch { shape1, shape2 }) => {
            eprintln!("Shape mismatch: {:?} vs {:?}", shape1, shape2);
        }
        Err(err) => eprintln!("Type error: {}", err),
    }
}
```

### Gradual Typing Error Recovery

```rust
// Gradual typing allows fallback to Unknown type
fn infer_with_gradual_fallback(expr: &Expr, env: &TypeEnvironment) -> LyraType {
    match infer_type(expr, env) {
        Ok(ty) => ty,
        Err(TypeError::UnboundVariable { .. }) => LyraType::Unknown,
        Err(TypeError::AmbiguousType) => LyraType::Unknown,
        Err(_) => LyraType::Never,  // Hard error
    }
}
```

## Performance Optimization

### Type Caching

```rust
use std::collections::HashMap;

pub struct TypeCache {
    expression_types: HashMap<u64, LyraType>,  // Hash -> Type
    substitution_cache: HashMap<(TypeVar, LyraType), LyraType>,
}

impl TypeCache {
    pub fn get_or_infer<F>(&mut self, expr_hash: u64, infer_fn: F) -> LyraType
    where
        F: FnOnce() -> LyraType,
    {
        self.expression_types
            .entry(expr_hash)
            .or_insert_with(infer_fn)
            .clone()
    }
}
```

### Efficient Type Comparison

```rust
impl LyraType {
    /// Fast structural equality check
    pub fn structurally_equal(&self, other: &LyraType) -> bool {
        // Use discriminant comparison first for speed
        std::mem::discriminant(self) == std::mem::discriminant(other) &&
        self == other
    }
    
    /// Compute hash for efficient lookup
    pub fn type_hash(&self) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}
```

## Testing

### Type System Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_type_operations() {
        let int_type = LyraType::Integer;
        let real_type = LyraType::Real;
        
        assert!(int_type.is_numeric());
        assert!(real_type.is_numeric());
        assert!(!int_type.is_collection());
        
        assert!(int_type.satisfies_constraint(&TypeClass::Numeric));
        assert!(int_type.satisfies_constraint(&TypeClass::Ordered));
    }
    
    #[test]
    fn test_type_substitution() {
        let mut subst = TypeSubstitution::new();
        subst.insert(0, LyraType::Integer);
        
        let generic_list = LyraType::List(Box::new(LyraType::TypeVar(0)));
        let concrete_list = generic_list.substitute(&subst);
        
        assert_eq!(concrete_list, LyraType::List(Box::new(LyraType::Integer)));
    }
    
    #[test]
    fn test_tensor_broadcasting() {
        let shape1 = TensorShape::new(vec![3, 1]);
        let shape2 = TensorShape::new(vec![1, 4]);
        
        assert!(shape1.broadcast_compatible(&shape2));
        
        let result = shape1.broadcast_result(&shape2).unwrap();
        assert_eq!(result.dimensions, vec![3, 4]);
    }
    
    #[test]
    fn test_function_type_checking() {
        let fn_type = LyraType::Function {
            params: vec![LyraType::Integer, LyraType::String],
            return_type: Box::new(LyraType::Boolean),
            attributes: vec![],
        };
        
        assert!(fn_type.is_function());
        
        let checker = TypeChecker::new();
        let arg_types = vec![LyraType::Integer, LyraType::String];
        let result = checker.check_application(&fn_type, &arg_types).unwrap();
        
        assert_eq!(result, LyraType::Boolean);
    }
}
```

### Property-Based Testing

```rust
#[cfg(test)]
mod property_tests {
    use super::*;
    use quickcheck::*;
    
    #[quickcheck]
    fn type_substitution_identity(ty: LyraType) -> bool {
        let empty_subst = TypeSubstitution::new();
        ty.substitute(&empty_subst) == ty
    }
    
    #[quickcheck]
    fn substitution_composition_associative(
        s1: TypeSubstitution,
        s2: TypeSubstitution, 
        s3: TypeSubstitution
    ) -> bool {
        let left = s1.compose(&s2).compose(&s3);
        let right = s1.compose(&s2.compose(&s3));
        // Check that they produce equivalent results
        left == right
    }
    
    #[quickcheck]
    fn tensor_broadcasting_symmetric(shape1: TensorShape, shape2: TensorShape) -> bool {
        shape1.broadcast_compatible(&shape2) == shape2.broadcast_compatible(&shape1)
    }
}
```

## Best Practices

### Type Design Principles

1. **Make illegal states unrepresentable**: Use the type system to prevent runtime errors
2. **Prefer composition over inheritance**: Use product and sum types
3. **Use gradual typing judiciously**: Fall back to `Unknown` only when necessary
4. **Leverage type inference**: Let the system infer types when possible
5. **Document type constraints**: Use type classes to express requirements

### Performance Guidelines

1. **Cache frequently computed types**
2. **Use structural sharing for type substitution**  
3. **Prefer early type checking over runtime validation**
4. **Use discriminant-based fast paths for type comparison**
5. **Minimize allocation in type operations**

### Error Handling

1. **Provide precise error messages with context**
2. **Use gradual typing for error recovery**
3. **Fail fast for type mismatches in critical paths**
4. **Collect multiple errors when possible**
5. **Suggest fixes for common type errors**

## Migration Guide

### Adding Types to Existing Code

1. **Start with gradual typing** using `Unknown` types
2. **Add type annotations incrementally**  
3. **Use type inference to fill gaps**
4. **Validate with comprehensive tests**
5. **Refactor to use more precise types over time**

### Extending the Type System

1. **Add new type variants to `LyraType` enum**
2. **Implement all required trait methods**
3. **Add type class constraints as needed**
4. **Update type checker and inference engine**
5. **Add comprehensive tests for new features**

## Conclusion

Lyra's type system provides a powerful foundation for symbolic computation with static type safety and runtime flexibility. By leveraging gradual typing, constraint solving, and advanced features like tensor shapes and effect types, developers can build robust, high-performance applications while maintaining the flexibility needed for symbolic computation.

The type system seamlessly integrates with pattern matching, the VM, and stdlib functions, providing a unified approach to type-safe symbolic computation. Through careful use of type inference, constraint solving, and error recovery, Lyra achieves both correctness and usability.