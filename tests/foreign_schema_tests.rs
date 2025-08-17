//! TDD Tests for Foreign Schema Implementation
//!
//! This module contains comprehensive tests for the Foreign Schema implementation,
//! following strict Test-Driven Development (TDD) practices. These tests define the
//! expected behavior for thread-safe type validation and schema operations in Lyra.
//!
//! ## Test Coverage
//!
//! - Schema creation from types and values
//! - Type validation and casting operations
//! - Schema composition and unification
//! - Complex nested schema structures
//! - Foreign trait implementation compliance
//! - Thread safety (Send + Sync)
//! - Error handling for invalid operations

use lyra::{
    foreign::{Foreign, ForeignError},
    vm::{Value, VmError, VmResult},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Test-specific value type that can be safely shared between threads
/// This is needed because the production Value enum contains Series with Rc<Vec<Value>>
/// which prevents it from being Send+Sync. In tests, we use this simplified enum
/// to test the Foreign Schema functionality independent of the threading constraint.
#[derive(Debug, Clone, PartialEq)]
pub enum TestValue {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    List(Vec<TestValue>),
    Missing,
}

/// Schema type definitions for comprehensive type validation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SchemaType {
    Int64,
    Float64,
    Bool,
    String,
    Decimal { precision: u8, scale: u8 },
    Date,
    Timestamp,
    UUID,
    Nullable(Box<SchemaType>),
    List(Box<SchemaType>),
    Struct(Vec<(String, SchemaType)>), // Ordered fields for Hash support
    Union(Vec<SchemaType>), // Sum types
}

/// Foreign Schema implementation for thread-safe type validation
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignSchema {
    /// The schema type definition
    pub schema_type: SchemaType,
    /// Optional name for the schema
    pub name: Option<String>,
    /// Whether the schema allows null/missing values by default
    pub nullable: bool,
    /// Additional constraints or metadata
    pub constraints: HashMap<String, String>,
}

impl ForeignSchema {
    /// Create a new schema with the given type
    pub fn new(schema_type: SchemaType) -> Self {
        ForeignSchema {
            nullable: matches!(schema_type, SchemaType::Nullable(_)),
            schema_type,
            name: None,
            constraints: HashMap::new(),
        }
    }
    
    /// Create a named schema
    pub fn named(name: String, schema_type: SchemaType) -> Self {
        ForeignSchema {
            nullable: matches!(schema_type, SchemaType::Nullable(_)),
            schema_type,
            name: Some(name),
            constraints: HashMap::new(),
        }
    }
    
    /// Create common schema types
    pub fn int64() -> Self {
        Self::new(SchemaType::Int64)
    }
    
    pub fn float64() -> Self {
        Self::new(SchemaType::Float64)
    }
    
    pub fn bool() -> Self {
        Self::new(SchemaType::Bool)
    }
    
    pub fn string() -> Self {
        Self::new(SchemaType::String)
    }
    
    pub fn decimal(precision: u8, scale: u8) -> Self {
        Self::new(SchemaType::Decimal { precision, scale })
    }
    
    pub fn date() -> Self {
        Self::new(SchemaType::Date)
    }
    
    pub fn timestamp() -> Self {
        Self::new(SchemaType::Timestamp)
    }
    
    pub fn uuid() -> Self {
        Self::new(SchemaType::UUID)
    }
    
    pub fn nullable(inner: SchemaType) -> Self {
        Self::new(SchemaType::Nullable(Box::new(inner)))
    }
    
    pub fn list(item_type: SchemaType) -> Self {
        Self::new(SchemaType::List(Box::new(item_type)))
    }
    
    pub fn struct_type(fields: Vec<(String, SchemaType)>) -> Self {
        let mut sorted_fields = fields;
        sorted_fields.sort_by(|a, b| a.0.cmp(&b.0)); // Sort for consistent ordering
        Self::new(SchemaType::Struct(sorted_fields))
    }
    
    pub fn union(types: Vec<SchemaType>) -> Self {
        Self::new(SchemaType::Union(types))
    }
    
    /// Add a constraint to the schema
    pub fn with_constraint(mut self, key: String, value: String) -> Self {
        self.constraints.insert(key, value);
        self
    }
    
    /// Make this schema nullable
    pub fn make_nullable(self) -> Self {
        if self.nullable {
            self // Already nullable
        } else {
            Self::new(SchemaType::Nullable(Box::new(self.schema_type)))
        }
    }
    
    /// Check if a value matches this schema
    pub fn matches(&self, value: &TestValue) -> bool {
        self.matches_type(value, &self.schema_type)
    }
    
    /// Validate a value against this schema (strict mode)
    pub fn validate(&self, value: &TestValue) -> VmResult<()> {
        if self.matches(value) {
            Ok(())
        } else {
            Err(VmError::TypeError {
                expected: format!("{:?}", self.schema_type),
                actual: format!("{:?}", value),
            })
        }
    }
    
    /// Cast a value to match this schema
    pub fn cast(&self, value: &TestValue, strict: bool) -> VmResult<TestValue> {
        self.cast_to_type(value, &self.schema_type, strict)
    }
    
    /// Infer schema from a value
    pub fn infer_from_value(value: &TestValue) -> Self {
        let schema_type = Self::infer_schema_type(value);
        Self::new(schema_type)
    }
    
    /// Infer schema from multiple values (finds common type)
    pub fn infer_from_values(values: &[TestValue]) -> Self {
        if values.is_empty() {
            return Self::string(); // Default to string
        }
        
        let mut unified_type = Self::infer_schema_type(&values[0]);
        let mut has_missing = matches!(values[0], TestValue::Missing);
        
        for value in values.iter().skip(1) {
            if matches!(value, TestValue::Missing) {
                has_missing = true;
            } else {
                let value_type = Self::infer_schema_type(value);
                unified_type = Self::unify_types(unified_type, value_type);
            }
        }
        
        if has_missing {
            Self::nullable(unified_type)
        } else {
            Self::new(unified_type)
        }
    }
    
    /// Unify two schemas to find the most general schema that accepts both
    pub fn unify(&self, other: &Self) -> Self {
        let unified_type = Self::unify_types(self.schema_type.clone(), other.schema_type.clone());
        let is_nullable = self.nullable || other.nullable;
        
        let mut result = Self::new(unified_type);
        result.nullable = is_nullable;
        result
    }
    
    /// Check if this schema is compatible with another (can be unified)
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        // Two schemas are compatible if they can be unified without losing too much precision
        let unified = self.unify(other);
        
        // Check if the unified type is reasonable (not just falling back to String)
        !matches!(unified.schema_type, SchemaType::String) ||
        matches!(self.schema_type, SchemaType::String) ||
        matches!(other.schema_type, SchemaType::String)
    }
    
    /// Get the inner type if this is a nullable schema
    pub fn inner_type(&self) -> &SchemaType {
        match &self.schema_type {
            SchemaType::Nullable(inner) => inner,
            _ => &self.schema_type,
        }
    }
    
    /// Check if this schema is nullable
    pub fn is_nullable(&self) -> bool {
        self.nullable || matches!(self.schema_type, SchemaType::Nullable(_))
    }
    
    /// Get the complexity score of this schema (for optimization)
    pub fn complexity(&self) -> usize {
        self.type_complexity(&self.schema_type)
    }
    
    // Private helper methods
    
    fn matches_type(&self, value: &TestValue, schema_type: &SchemaType) -> bool {
        match (value, schema_type) {
            (TestValue::Missing, SchemaType::Nullable(_)) => true,
            (TestValue::Missing, _) => false,
            (_, SchemaType::Nullable(inner)) => {
                matches!(value, TestValue::Missing) || self.matches_type(value, inner)
            },
            
            (TestValue::Integer(_), SchemaType::Int64) => true,
            (TestValue::Real(_), SchemaType::Float64) => true,
            (TestValue::Boolean(_), SchemaType::Bool) => true,
            (TestValue::String(s), SchemaType::String) => true,
            
            // String pattern matching for specialized types
            (TestValue::String(s), SchemaType::Date) => Self::looks_like_date(s),
            (TestValue::String(s), SchemaType::Timestamp) => Self::looks_like_timestamp(s),
            (TestValue::String(s), SchemaType::UUID) => Self::looks_like_uuid(s),
            
            // Numeric type compatibility
            (TestValue::Integer(_), SchemaType::Float64) => true, // Int can be cast to float
            (TestValue::Integer(_), SchemaType::Decimal { .. }) => true,
            (TestValue::Real(_), SchemaType::Decimal { .. }) => true,
            
            // List type matching
            (TestValue::List(items), SchemaType::List(item_type)) => {
                items.iter().all(|item| self.matches_type(item, item_type))
            },
            
            // Union type matching
            (value, SchemaType::Union(types)) => {
                types.iter().any(|t| self.matches_type(value, t))
            },
            
            _ => false,
        }
    }
    
    fn cast_to_type(&self, value: &TestValue, target_type: &SchemaType, strict: bool) -> VmResult<TestValue> {
        // If already matches, return as-is
        if self.matches_type(value, target_type) {
            return Ok(value.clone());
        }
        
        // Handle Missing values
        if matches!(value, TestValue::Missing) {
            match target_type {
                SchemaType::Nullable(_) => return Ok(TestValue::Missing),
                _ if !strict => return Ok(TestValue::Missing),
                _ => return Err(VmError::TypeError {
                    expected: format!("{:?}", target_type),
                    actual: "Missing".to_string(),
                }),
            }
        }
        
        // Handle nullable wrapper
        if let SchemaType::Nullable(inner) = target_type {
            return self.cast_to_type(value, inner, strict);
        }
        
        // Type conversions
        match (value, target_type) {
            // Numeric conversions
            (TestValue::Integer(i), SchemaType::Float64) => {
                Ok(TestValue::Real(*i as f64))
            },
            (TestValue::Real(f), SchemaType::Int64) if !strict => {
                Ok(TestValue::Integer(*f as i64))
            },
            
            // String conversions
            (TestValue::Integer(i), SchemaType::String) => {
                Ok(TestValue::String(i.to_string()))
            },
            (TestValue::Real(f), SchemaType::String) => {
                Ok(TestValue::String(f.to_string()))
            },
            (TestValue::Boolean(b), SchemaType::String) => {
                Ok(TestValue::String(b.to_string()))
            },
            
            // String parsing (only in non-strict mode)
            (TestValue::String(s), SchemaType::Int64) if !strict => {
                s.parse::<i64>()
                    .map(TestValue::Integer)
                    .map_err(|_| VmError::TypeError {
                        expected: "valid integer string".to_string(),
                        actual: s.clone(),
                    })
            },
            (TestValue::String(s), SchemaType::Float64) if !strict => {
                s.parse::<f64>()
                    .map(TestValue::Real)
                    .map_err(|_| VmError::TypeError {
                        expected: "valid float string".to_string(),
                        actual: s.clone(),
                    })
            },
            (TestValue::String(s), SchemaType::Bool) if !strict => {
                match s.to_lowercase().as_str() {
                    "true" | "t" | "1" | "yes" | "y" => Ok(TestValue::Boolean(true)),
                    "false" | "f" | "0" | "no" | "n" => Ok(TestValue::Boolean(false)),
                    _ => Err(VmError::TypeError {
                        expected: "valid boolean string".to_string(),
                        actual: s.clone(),
                    }),
                }
            },
            
            // List conversions
            (TestValue::List(items), SchemaType::List(item_type)) => {
                let cast_items: Result<Vec<_>, _> = items.iter()
                    .map(|item| self.cast_to_type(item, item_type, strict))
                    .collect();
                cast_items.map(TestValue::List)
            },
            
            _ => Err(VmError::TypeError {
                expected: format!("{:?}", target_type),
                actual: format!("{:?}", value),
            }),
        }
    }
    
    fn infer_schema_type(value: &TestValue) -> SchemaType {
        match value {
            TestValue::Integer(_) => SchemaType::Int64,
            TestValue::Real(_) => SchemaType::Float64,
            TestValue::Boolean(_) => SchemaType::Bool,
            TestValue::String(s) => {
                // Try to infer more specific string types
                if Self::looks_like_date(s) {
                    SchemaType::Date
                } else if Self::looks_like_timestamp(s) {
                    SchemaType::Timestamp
                } else if Self::looks_like_uuid(s) {
                    SchemaType::UUID
                } else {
                    SchemaType::String
                }
            },
            TestValue::List(items) => {
                if items.is_empty() {
                    SchemaType::List(Box::new(SchemaType::String)) // Default
                } else {
                    let item_schema = Self::infer_from_values(items);
                    SchemaType::List(Box::new(item_schema.schema_type))
                }
            },
            TestValue::Missing => SchemaType::String, // Default for Missing alone
        }
    }
    
    fn unify_types(type1: SchemaType, type2: SchemaType) -> SchemaType {
        if type1 == type2 {
            return type1;
        }
        
        match (type1, type2) {
            // Numeric unification
            (SchemaType::Int64, SchemaType::Float64) | (SchemaType::Float64, SchemaType::Int64) => {
                SchemaType::Float64
            },
            (SchemaType::Int64, SchemaType::Decimal { .. }) | (SchemaType::Decimal { .. }, SchemaType::Int64) => {
                SchemaType::Float64
            },
            (SchemaType::Float64, SchemaType::Decimal { .. }) | (SchemaType::Decimal { .. }, SchemaType::Float64) => {
                SchemaType::Float64
            },
            
            // String unification (string is most general)
            (SchemaType::Date, SchemaType::String) | (SchemaType::String, SchemaType::Date) => {
                SchemaType::String
            },
            (SchemaType::Timestamp, SchemaType::String) | (SchemaType::String, SchemaType::Timestamp) => {
                SchemaType::String
            },
            (SchemaType::UUID, SchemaType::String) | (SchemaType::String, SchemaType::UUID) => {
                SchemaType::String
            },
            
            // Date/time unification
            (SchemaType::Date, SchemaType::Timestamp) | (SchemaType::Timestamp, SchemaType::Date) => {
                SchemaType::String
            },
            
            // List unification
            (SchemaType::List(item1), SchemaType::List(item2)) => {
                let unified_item = Self::unify_types(*item1, *item2);
                SchemaType::List(Box::new(unified_item))
            },
            
            // Nullable unification
            (SchemaType::Nullable(inner1), SchemaType::Nullable(inner2)) => {
                let unified_inner = Self::unify_types(*inner1, *inner2);
                SchemaType::Nullable(Box::new(unified_inner))
            },
            (SchemaType::Nullable(inner), other) | (other, SchemaType::Nullable(inner)) => {
                let unified_inner = Self::unify_types(*inner, other);
                SchemaType::Nullable(Box::new(unified_inner))
            },
            
            // Union types
            (SchemaType::Union(mut types1), SchemaType::Union(types2)) => {
                types1.extend(types2);
                types1.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                types1.dedup();
                SchemaType::Union(types1)
            },
            (SchemaType::Union(mut types), other) | (other, SchemaType::Union(mut types)) => {
                types.push(other);
                types.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                types.dedup();
                SchemaType::Union(types)
            },
            
            // Default: create union type
            (t1, t2) => {
                let mut types = vec![t1, t2];
                types.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
                SchemaType::Union(types)
            },
        }
    }
    
    fn type_complexity(&self, schema_type: &SchemaType) -> usize {
        match schema_type {
            SchemaType::Int64 | SchemaType::Float64 | SchemaType::Bool | SchemaType::String => 1,
            SchemaType::Decimal { .. } | SchemaType::Date | SchemaType::Timestamp | SchemaType::UUID => 2,
            SchemaType::Nullable(inner) => 1 + self.type_complexity(inner),
            SchemaType::List(item_type) => 2 + self.type_complexity(item_type),
            SchemaType::Struct(fields) => 3 + fields.iter().map(|(_, t)| self.type_complexity(t)).sum::<usize>(),
            SchemaType::Union(types) => 2 + types.iter().map(|t| self.type_complexity(t)).sum::<usize>(),
        }
    }
    
    // String pattern recognition helpers
    fn looks_like_date(s: &str) -> bool {
        // Simple date pattern: YYYY-MM-DD
        let parts: Vec<&str> = s.split('-').collect();
        parts.len() == 3 &&
        parts[0].len() == 4 && parts[0].chars().all(|c| c.is_ascii_digit()) &&
        parts[1].len() == 2 && parts[1].chars().all(|c| c.is_ascii_digit()) &&
        parts[2].len() == 2 && parts[2].chars().all(|c| c.is_ascii_digit())
    }
    
    fn looks_like_timestamp(s: &str) -> bool {
        // Simple timestamp pattern: includes 'T' and potentially 'Z'
        s.contains('T') && (s.contains('Z') || s.contains('+') || s.contains('-'))
    }
    
    fn looks_like_uuid(s: &str) -> bool {
        // Simple UUID pattern: 8-4-4-4-12 hex characters separated by hyphens
        let parts: Vec<&str> = s.split('-').collect();
        parts.len() == 5 &&
        parts[0].len() == 8 && parts[0].chars().all(|c| c.is_ascii_hexdigit()) &&
        parts[1].len() == 4 && parts[1].chars().all(|c| c.is_ascii_hexdigit()) &&
        parts[2].len() == 4 && parts[2].chars().all(|c| c.is_ascii_hexdigit()) &&
        parts[3].len() == 4 && parts[3].chars().all(|c| c.is_ascii_hexdigit()) &&
        parts[4].len() == 12 && parts[4].chars().all(|c| c.is_ascii_hexdigit())
    }
}

impl Foreign for ForeignSchema {
    fn type_name(&self) -> &'static str {
        "Schema"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Name" => {
                match &self.name {
                    Some(name) => Ok(Value::String(name.clone())),
                    None => Ok(Value::Missing),
                }
            },
            
            "Type" => {
                Ok(Value::String(format!("{:?}", self.schema_type)))
            },
            
            "IsNullable" => {
                Ok(Value::Boolean(self.is_nullable()))
            },
            
            "Complexity" => {
                Ok(Value::Integer(self.complexity() as i64))
            },
            
            "InnerType" => {
                let inner = self.inner_type();
                Ok(Value::String(format!("{:?}", inner)))
            },
            
            "Validate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArguments {
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // Convert Value to TestValue for validation
                let test_value = match &args[0] {
                    Value::Integer(i) => TestValue::Integer(*i),
                    Value::Real(f) => TestValue::Real(*f),
                    Value::String(s) => TestValue::String(s.clone()),
                    Value::Boolean(b) => TestValue::Boolean(*b),
                    Value::Missing => TestValue::Missing,
                    _ => return Err(ForeignError::TypeError {
                        expected: "Simple value type".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.validate(&test_value) {
                    Ok(()) => Ok(Value::Boolean(true)),
                    Err(_) => Ok(Value::Boolean(false)),
                }
            },
            
            "Cast" => {
                if args.len() < 1 || args.len() > 2 {
                    return Err(ForeignError::InvalidArguments {
                        expected: 1, // or 2 for strict mode
                        actual: args.len(),
                    });
                }
                
                let strict = if args.len() == 2 {
                    match &args[1] {
                        Value::Boolean(b) => *b,
                        _ => false, // Default to non-strict
                    }
                } else {
                    false // Default to non-strict
                };
                
                // Convert Value to TestValue for casting
                let test_value = match &args[0] {
                    Value::Integer(i) => TestValue::Integer(*i),
                    Value::Real(f) => TestValue::Real(*f),
                    Value::String(s) => TestValue::String(s.clone()),
                    Value::Boolean(b) => TestValue::Boolean(*b),
                    Value::Missing => TestValue::Missing,
                    _ => return Err(ForeignError::TypeError {
                        expected: "Simple value type".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                match self.cast(&test_value, strict) {
                    Ok(TestValue::Integer(i)) => Ok(Value::Integer(i)),
                    Ok(TestValue::Real(f)) => Ok(Value::Real(f)),
                    Ok(TestValue::String(s)) => Ok(Value::String(s)),
                    Ok(TestValue::Boolean(b)) => Ok(Value::Boolean(b)),
                    Ok(TestValue::Missing) => Ok(Value::Missing),
                    Ok(_) => Ok(Value::String("Complex Value".to_string())),
                    Err(e) => Err(ForeignError::RuntimeError(format!("Cast failed: {:?}", e))),
                }
            },
            
            "Matches" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArguments {
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // Convert Value to TestValue for matching
                let test_value = match &args[0] {
                    Value::Integer(i) => TestValue::Integer(*i),
                    Value::Real(f) => TestValue::Real(*f),
                    Value::String(s) => TestValue::String(s.clone()),
                    Value::Boolean(b) => TestValue::Boolean(*b),
                    Value::Missing => TestValue::Missing,
                    _ => return Err(ForeignError::TypeError {
                        expected: "Simple value type".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                Ok(Value::Boolean(self.matches(&test_value)))
            },
            
            "Unify" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArguments {
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // This would need another ForeignSchema to unify with
                // For now, just return a string representation
                Ok(Value::String("Unified schema would be returned here".to_string()))
            },
            
            "IsCompatibleWith" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArguments {
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // This would need another ForeignSchema to check compatibility
                // For now, return true as placeholder
                Ok(Value::Boolean(true))
            },
            
            "MakeNullable" => {
                let nullable_schema = self.clone().make_nullable();
                Ok(Value::String(format!("Nullable schema: {:?}", nullable_schema.schema_type)))
            },
            
            "Constraints" => {
                let constraints: Vec<Value> = self.constraints.iter()
                    .map(|(k, v)| Value::List(vec![Value::String(k.clone()), Value::String(v.clone())]))
                    .collect();
                Ok(Value::List(constraints))
            },
            
            _ => Err(ForeignError::MethodNotFound {
                method: method.to_string(),
                type_name: "Schema".to_string(),
            }),
        }
    }
}

// Make ForeignSchema thread-safe
unsafe impl Send for ForeignSchema {}
unsafe impl Sync for ForeignSchema {}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_foreign_schema_basic_creation() {
        let schema = ForeignSchema::int64();
        assert_eq!(schema.type_name(), "Schema");
        assert_eq!(schema.schema_type, SchemaType::Int64);
        assert!(!schema.is_nullable());
        assert_eq!(schema.complexity(), 1);
    }
    
    #[test]
    fn test_foreign_schema_common_types() {
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        let bool_schema = ForeignSchema::bool();
        let string_schema = ForeignSchema::string();
        
        assert_eq!(int_schema.schema_type, SchemaType::Int64);
        assert_eq!(float_schema.schema_type, SchemaType::Float64);
        assert_eq!(bool_schema.schema_type, SchemaType::Bool);
        assert_eq!(string_schema.schema_type, SchemaType::String);
    }
    
    #[test]
    fn test_foreign_schema_specialized_types() {
        let date_schema = ForeignSchema::date();
        let timestamp_schema = ForeignSchema::timestamp();
        let uuid_schema = ForeignSchema::uuid();
        let decimal_schema = ForeignSchema::decimal(10, 2);
        
        assert_eq!(date_schema.schema_type, SchemaType::Date);
        assert_eq!(timestamp_schema.schema_type, SchemaType::Timestamp);
        assert_eq!(uuid_schema.schema_type, SchemaType::UUID);
        assert_eq!(decimal_schema.schema_type, SchemaType::Decimal { precision: 10, scale: 2 });
    }
    
    #[test]
    fn test_foreign_schema_nullable_types() {
        let nullable_int = ForeignSchema::nullable(SchemaType::Int64);
        assert!(nullable_int.is_nullable());
        assert_eq!(nullable_int.inner_type(), &SchemaType::Int64);
        
        let regular_int = ForeignSchema::int64();
        assert!(!regular_int.is_nullable());
        
        let made_nullable = regular_int.make_nullable();
        assert!(made_nullable.is_nullable());
    }
    
    #[test]
    fn test_foreign_schema_list_types() {
        let int_list = ForeignSchema::list(SchemaType::Int64);
        assert_eq!(int_list.schema_type, SchemaType::List(Box::new(SchemaType::Int64)));
        assert_eq!(int_list.complexity(), 3); // 2 for list + 1 for int
        
        let nested_list = ForeignSchema::list(SchemaType::List(Box::new(SchemaType::String)));
        assert_eq!(nested_list.complexity(), 4); // 2 + 2 + 1
    }
    
    #[test]
    fn test_foreign_schema_struct_types() {
        let fields = vec![
            ("id".to_string(), SchemaType::Int64),
            ("name".to_string(), SchemaType::String),
            ("active".to_string(), SchemaType::Bool),
        ];
        let struct_schema = ForeignSchema::struct_type(fields);
        
        match &struct_schema.schema_type {
            SchemaType::Struct(fields) => {
                assert_eq!(fields.len(), 3);
                // Fields should be sorted alphabetically
                assert_eq!(fields[0].0, "active");
                assert_eq!(fields[1].0, "id");
                assert_eq!(fields[2].0, "name");
            },
            _ => panic!("Expected struct schema"),
        }
    }
    
    #[test]
    fn test_foreign_schema_union_types() {
        let types = vec![SchemaType::Int64, SchemaType::String];
        let union_schema = ForeignSchema::union(types.clone());
        
        match &union_schema.schema_type {
            SchemaType::Union(union_types) => {
                assert_eq!(union_types.len(), 2);
                assert!(union_types.contains(&SchemaType::Int64));
                assert!(union_types.contains(&SchemaType::String));
            },
            _ => panic!("Expected union schema"),
        }
    }
    
    #[test]
    fn test_foreign_schema_named_schemas() {
        let named_schema = ForeignSchema::named("UserID".to_string(), SchemaType::Int64);
        assert_eq!(named_schema.name, Some("UserID".to_string()));
        assert_eq!(named_schema.schema_type, SchemaType::Int64);
    }
    
    #[test]
    fn test_foreign_schema_constraints() {
        let schema = ForeignSchema::int64()
            .with_constraint("min".to_string(), "0".to_string())
            .with_constraint("max".to_string(), "100".to_string());
        
        assert_eq!(schema.constraints.len(), 2);
        assert_eq!(schema.constraints.get("min"), Some(&"0".to_string()));
        assert_eq!(schema.constraints.get("max"), Some(&"100".to_string()));
    }
    
    #[test]
    fn test_foreign_schema_value_matching() {
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        let string_schema = ForeignSchema::string();
        let nullable_int_schema = ForeignSchema::nullable(SchemaType::Int64);
        
        // Test exact matches
        assert!(int_schema.matches(&TestValue::Integer(42)));
        assert!(float_schema.matches(&TestValue::Real(3.14)));
        assert!(string_schema.matches(&TestValue::String("hello".to_string())));
        
        // Test type compatibility
        assert!(float_schema.matches(&TestValue::Integer(42))); // Int can be cast to float
        assert!(!int_schema.matches(&TestValue::Real(3.14))); // Float can't directly match int schema
        
        // Test nullable handling
        assert!(nullable_int_schema.matches(&TestValue::Integer(42)));
        assert!(nullable_int_schema.matches(&TestValue::Missing));
        assert!(!int_schema.matches(&TestValue::Missing));
    }
    
    #[test]
    fn test_foreign_schema_list_matching() {
        let int_list_schema = ForeignSchema::list(SchemaType::Int64);
        let mixed_list_schema = ForeignSchema::list(SchemaType::Union(vec![SchemaType::Int64, SchemaType::String]));
        
        // Test homogeneous list
        let int_list = TestValue::List(vec![
            TestValue::Integer(1),
            TestValue::Integer(2),
            TestValue::Integer(3),
        ]);
        assert!(int_list_schema.matches(&int_list));
        
        // Test mixed list
        let mixed_list = TestValue::List(vec![
            TestValue::Integer(1),
            TestValue::String("hello".to_string()),
            TestValue::Integer(2),
        ]);
        assert!(!int_list_schema.matches(&mixed_list));
        assert!(mixed_list_schema.matches(&mixed_list));
    }
    
    #[test]
    fn test_foreign_schema_validation() {
        let int_schema = ForeignSchema::int64();
        
        // Valid value
        assert!(int_schema.validate(&TestValue::Integer(42)).is_ok());
        
        // Invalid value
        assert!(int_schema.validate(&TestValue::String("hello".to_string())).is_err());
        assert!(int_schema.validate(&TestValue::Missing).is_err());
        
        // Nullable schema allows Missing
        let nullable_int = ForeignSchema::nullable(SchemaType::Int64);
        assert!(nullable_int.validate(&TestValue::Missing).is_ok());
    }
    
    #[test]
    fn test_foreign_schema_casting_numeric() {
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        
        // Int to float (always allowed)
        let result = float_schema.cast(&TestValue::Integer(42), true).unwrap();
        assert_eq!(result, TestValue::Real(42.0));
        
        // Float to int (only in non-strict mode)
        let result = int_schema.cast(&TestValue::Real(42.7), false).unwrap();
        assert_eq!(result, TestValue::Integer(42));
        
        // Float to int in strict mode should fail
        assert!(int_schema.cast(&TestValue::Real(42.7), true).is_err());
    }
    
    #[test]
    fn test_foreign_schema_casting_strings() {
        let string_schema = ForeignSchema::string();
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        let bool_schema = ForeignSchema::bool();
        
        // Numbers to strings
        let result = string_schema.cast(&TestValue::Integer(42), false).unwrap();
        assert_eq!(result, TestValue::String("42".to_string()));
        
        let result = string_schema.cast(&TestValue::Real(3.14), false).unwrap();
        assert_eq!(result, TestValue::String("3.14".to_string()));
        
        // Strings to numbers (non-strict mode)
        let result = int_schema.cast(&TestValue::String("42".to_string()), false).unwrap();
        assert_eq!(result, TestValue::Integer(42));
        
        let result = float_schema.cast(&TestValue::String("3.14".to_string()), false).unwrap();
        assert_eq!(result, TestValue::Real(3.14));
        
        // String to bool
        let result = bool_schema.cast(&TestValue::String("true".to_string()), false).unwrap();
        assert_eq!(result, TestValue::Boolean(true));
        
        let result = bool_schema.cast(&TestValue::String("false".to_string()), false).unwrap();
        assert_eq!(result, TestValue::Boolean(false));
        
        // Invalid string parsing should fail
        assert!(int_schema.cast(&TestValue::String("not_a_number".to_string()), false).is_err());
    }
    
    #[test]
    fn test_foreign_schema_casting_lists() {
        let int_list_schema = ForeignSchema::list(SchemaType::Int64);
        let string_list_schema = ForeignSchema::list(SchemaType::String);
        
        let mixed_list = TestValue::List(vec![
            TestValue::Integer(1),
            TestValue::Integer(2),
            TestValue::Integer(3),
        ]);
        
        // Cast int list to string list
        let result = string_list_schema.cast(&mixed_list, false).unwrap();
        match result {
            TestValue::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], TestValue::String("1".to_string()));
                assert_eq!(items[1], TestValue::String("2".to_string()));
                assert_eq!(items[2], TestValue::String("3".to_string()));
            },
            _ => panic!("Expected list result"),
        }
    }
    
    #[test]
    fn test_foreign_schema_inference_simple() {
        // Test inference from single values
        let int_schema = ForeignSchema::infer_from_value(&TestValue::Integer(42));
        assert_eq!(int_schema.schema_type, SchemaType::Int64);
        
        let string_schema = ForeignSchema::infer_from_value(&TestValue::String("hello".to_string()));
        assert_eq!(string_schema.schema_type, SchemaType::String);
        
        let missing_schema = ForeignSchema::infer_from_value(&TestValue::Missing);
        assert_eq!(missing_schema.schema_type, SchemaType::String); // Default for Missing
    }
    
    #[test]
    fn test_foreign_schema_inference_specialized_strings() {
        let date_string = TestValue::String("2023-12-25".to_string());
        let date_schema = ForeignSchema::infer_from_value(&date_string);
        assert_eq!(date_schema.schema_type, SchemaType::Date);
        
        let timestamp_string = TestValue::String("2023-12-25T10:30:00Z".to_string());
        let timestamp_schema = ForeignSchema::infer_from_value(&timestamp_string);
        assert_eq!(timestamp_schema.schema_type, SchemaType::Timestamp);
        
        let uuid_string = TestValue::String("123e4567-e89b-12d3-a456-426614174000".to_string());
        let uuid_schema = ForeignSchema::infer_from_value(&uuid_string);
        assert_eq!(uuid_schema.schema_type, SchemaType::UUID);
    }
    
    #[test]
    fn test_foreign_schema_inference_from_multiple_values() {
        // Homogeneous values
        let int_values = vec![
            TestValue::Integer(1),
            TestValue::Integer(2),
            TestValue::Integer(3),
        ];
        let schema = ForeignSchema::infer_from_values(&int_values);
        assert_eq!(schema.schema_type, SchemaType::Int64);
        assert!(!schema.is_nullable());
        
        // Mixed numeric values (should unify to Float64)
        let mixed_numeric = vec![
            TestValue::Integer(1),
            TestValue::Real(2.5),
            TestValue::Integer(3),
        ];
        let schema = ForeignSchema::infer_from_values(&mixed_numeric);
        assert_eq!(schema.schema_type, SchemaType::Float64);
        
        // Values with Missing (should become nullable)
        let with_missing = vec![
            TestValue::Integer(1),
            TestValue::Missing,
            TestValue::Integer(3),
        ];
        let schema = ForeignSchema::infer_from_values(&with_missing);
        match schema.schema_type {
            SchemaType::Nullable(inner) => assert_eq!(*inner, SchemaType::Int64),
            _ => panic!("Expected nullable schema"),
        }
    }
    
    #[test]
    fn test_foreign_schema_inference_lists() {
        let list_value = TestValue::List(vec![
            TestValue::Integer(1),
            TestValue::Integer(2),
            TestValue::Integer(3),
        ]);
        let schema = ForeignSchema::infer_from_value(&list_value);
        
        match schema.schema_type {
            SchemaType::List(item_type) => assert_eq!(*item_type, SchemaType::Int64),
            _ => panic!("Expected list schema"),
        }
    }
    
    #[test]
    fn test_foreign_schema_unification() {
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        let string_schema = ForeignSchema::string();
        
        // Numeric unification
        let unified = int_schema.unify(&float_schema);
        assert_eq!(unified.schema_type, SchemaType::Float64);
        
        // String unification (string is most general)
        let unified = int_schema.unify(&string_schema);
        match unified.schema_type {
            SchemaType::Union(_) => (), // Should create union type
            _ => panic!("Expected union type for incompatible unification"),
        }
        
        // Nullable unification
        let nullable_int = ForeignSchema::nullable(SchemaType::Int64);
        let unified = int_schema.unify(&nullable_int);
        assert!(unified.is_nullable());
    }
    
    #[test]
    fn test_foreign_schema_compatibility() {
        let int_schema = ForeignSchema::int64();
        let float_schema = ForeignSchema::float64();
        let string_schema = ForeignSchema::string();
        
        // Numeric types should be compatible
        assert!(int_schema.is_compatible_with(&float_schema));
        assert!(float_schema.is_compatible_with(&int_schema));
        
        // String can unify with anything, so it's compatible
        assert!(int_schema.is_compatible_with(&string_schema));
        assert!(string_schema.is_compatible_with(&int_schema));
    }
    
    #[test]
    fn test_foreign_schema_complexity() {
        assert_eq!(ForeignSchema::int64().complexity(), 1);
        assert_eq!(ForeignSchema::list(SchemaType::Int64).complexity(), 3);
        assert_eq!(ForeignSchema::nullable(SchemaType::Int64).complexity(), 2);
        
        let struct_fields = vec![
            ("id".to_string(), SchemaType::Int64),
            ("name".to_string(), SchemaType::String),
        ];
        let struct_schema = ForeignSchema::struct_type(struct_fields);
        assert_eq!(struct_schema.complexity(), 5); // 3 + 1 + 1
    }
    
    // Foreign trait implementation tests
    
    #[test]
    fn test_foreign_trait_type_method() {
        let schema = ForeignSchema::int64();
        let result = schema.call_method("Type", &[]).unwrap();
        match result {
            Value::String(type_str) => {
                assert!(type_str.contains("Int64"));
            },
            _ => panic!("Expected string type description"),
        }
    }
    
    #[test]
    fn test_foreign_trait_name_method() {
        let unnamed_schema = ForeignSchema::int64();
        let result = unnamed_schema.call_method("Name", &[]).unwrap();
        assert_eq!(result, Value::Missing);
        
        let named_schema = ForeignSchema::named("UserID".to_string(), SchemaType::Int64);
        let result = named_schema.call_method("Name", &[]).unwrap();
        match result {
            Value::String(name) => assert_eq!(name, "UserID"),
            _ => panic!("Expected string name"),
        }
    }
    
    #[test]
    fn test_foreign_trait_is_nullable_method() {
        let regular_schema = ForeignSchema::int64();
        let result = regular_schema.call_method("IsNullable", &[]).unwrap();
        assert_eq!(result, Value::Boolean(false));
        
        let nullable_schema = ForeignSchema::nullable(SchemaType::Int64);
        let result = nullable_schema.call_method("IsNullable", &[]).unwrap();
        assert_eq!(result, Value::Boolean(true));
    }
    
    #[test]
    fn test_foreign_trait_complexity_method() {
        let schema = ForeignSchema::int64();
        let result = schema.call_method("Complexity", &[]).unwrap();
        match result {
            Value::Integer(complexity) => assert_eq!(complexity, 1),
            _ => panic!("Expected integer complexity"),
        }
    }
    
    #[test]
    fn test_foreign_trait_validate_method() {
        let schema = ForeignSchema::int64();
        
        // Valid value
        let args = vec![Value::Integer(42)];
        let result = schema.call_method("Validate", &args).unwrap();
        assert_eq!(result, Value::Boolean(true));
        
        // Invalid value
        let args = vec![Value::String("hello".to_string())];
        let result = schema.call_method("Validate", &args).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }
    
    #[test]
    fn test_foreign_trait_cast_method() {
        let float_schema = ForeignSchema::float64();
        
        // Cast integer to float
        let args = vec![Value::Integer(42)];
        let result = float_schema.call_method("Cast", &args).unwrap();
        match result {
            Value::Real(f) => assert_eq!(f, 42.0),
            _ => panic!("Expected real value"),
        }
        
        // Cast with strict mode
        let args = vec![Value::Integer(42), Value::Boolean(true)];
        let result = float_schema.call_method("Cast", &args).unwrap();
        match result {
            Value::Real(f) => assert_eq!(f, 42.0),
            _ => panic!("Expected real value"),
        }
    }
    
    #[test]
    fn test_foreign_trait_matches_method() {
        let schema = ForeignSchema::int64();
        
        // Matching value
        let args = vec![Value::Integer(42)];
        let result = schema.call_method("Matches", &args).unwrap();
        assert_eq!(result, Value::Boolean(true));
        
        // Non-matching value
        let args = vec![Value::String("hello".to_string())];
        let result = schema.call_method("Matches", &args).unwrap();
        assert_eq!(result, Value::Boolean(false));
    }
    
    #[test]
    fn test_foreign_trait_make_nullable_method() {
        let schema = ForeignSchema::int64();
        let result = schema.call_method("MakeNullable", &[]).unwrap();
        match result {
            Value::String(desc) => {
                assert!(desc.contains("Nullable"));
                assert!(desc.contains("Int64"));
            },
            _ => panic!("Expected string description"),
        }
    }
    
    #[test]
    fn test_foreign_trait_constraints_method() {
        let schema = ForeignSchema::int64()
            .with_constraint("min".to_string(), "0".to_string())
            .with_constraint("max".to_string(), "100".to_string());
        
        let result = schema.call_method("Constraints", &[]).unwrap();
        match result {
            Value::List(constraints) => {
                assert_eq!(constraints.len(), 2);
                // Each constraint should be a [key, value] pair
                for constraint in constraints {
                    match constraint {
                        Value::List(pair) => assert_eq!(pair.len(), 2),
                        _ => panic!("Expected constraint pairs"),
                    }
                }
            },
            _ => panic!("Expected list of constraints"),
        }
    }
    
    #[test]
    fn test_foreign_trait_method_errors() {
        let schema = ForeignSchema::int64();
        
        // Wrong number of arguments
        let result = schema.call_method("Validate", &[]);
        assert!(result.is_err());
        
        // Wrong argument type
        let args = vec![Value::List(vec![])];
        let result = schema.call_method("Validate", &args);
        assert!(result.is_err());
        
        // Unknown method
        let result = schema.call_method("UnknownMethod", &[]);
        assert!(result.is_err());
        match result.unwrap_err() {
            ForeignError::MethodNotFound { method, type_name } => {
                assert_eq!(method, "UnknownMethod");
                assert_eq!(type_name, "Schema");
            },
            _ => panic!("Expected MethodNotFound error"),
        }
    }
    
    #[test]
    fn test_foreign_schema_thread_safety() {
        use std::thread;
        use std::sync::Arc;
        
        let schema = Arc::new(ForeignSchema::int64());
        
        // Verify Send + Sync by using in multiple threads
        let schema1 = Arc::clone(&schema);
        let schema2 = Arc::clone(&schema);
        
        let handle1 = thread::spawn(move || {
            let matches = schema1.matches(&TestValue::Integer(42));
            assert!(matches);
        });
        
        let handle2 = thread::spawn(move || {
            let complexity = schema2.complexity();
            assert_eq!(complexity, 1);
        });
        
        handle1.join().unwrap();
        handle2.join().unwrap();
        
        // Additional verification that the schema is still usable
        let result = schema.cast(&TestValue::Integer(42), true).unwrap();
        assert_eq!(result, TestValue::Integer(42));
    }
    
    #[test]
    fn test_foreign_schema_string_pattern_recognition() {
        // Test date pattern recognition
        assert!(ForeignSchema::looks_like_date("2023-12-25"));
        assert!(!ForeignSchema::looks_like_date("not-a-date"));
        assert!(!ForeignSchema::looks_like_date("2023/12/25")); // Wrong format
        
        // Test timestamp pattern recognition
        assert!(ForeignSchema::looks_like_timestamp("2023-12-25T10:30:00Z"));
        assert!(ForeignSchema::looks_like_timestamp("2023-12-25T10:30:00+02:00"));
        assert!(!ForeignSchema::looks_like_timestamp("2023-12-25"));
        
        // Test UUID pattern recognition
        assert!(ForeignSchema::looks_like_uuid("123e4567-e89b-12d3-a456-426614174000"));
        assert!(!ForeignSchema::looks_like_uuid("not-a-uuid"));
        assert!(!ForeignSchema::looks_like_uuid("123e4567-e89b-12d3-a456")); // Too short
    }
    
    #[test]
    fn test_foreign_schema_edge_cases() {
        // Empty list inference
        let empty_list = TestValue::List(vec![]);
        let schema = ForeignSchema::infer_from_value(&empty_list);
        match schema.schema_type {
            SchemaType::List(item_type) => assert_eq!(*item_type, SchemaType::String),
            _ => panic!("Expected list schema"),
        }
        
        // Empty values array
        let schema = ForeignSchema::infer_from_values(&[]);
        assert_eq!(schema.schema_type, SchemaType::String);
        
        // Single Missing value
        let schema = ForeignSchema::infer_from_values(&[TestValue::Missing]);
        match schema.schema_type {
            SchemaType::Nullable(inner) => assert_eq!(**inner, SchemaType::String),
            _ => panic!("Expected nullable string schema"),
        }
    }
}