use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::collections::HashMap;

/// Schema type constructors for data validation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SchemaType {
    Int64,
    Float64,
    Bool,
    String,
    Decimal { precision: u8, scale: u8 }, // Decimal[p,s]
    Date,                                 // Date values
    Timestamp,                            // Timestamp values
    UUID,                                 // UUID values
    Nullable(Box<SchemaType>),
    List(Box<SchemaType>),
    Struct(Vec<(String, SchemaType)>), // Use Vec instead of HashMap for Hash support
}

/// Foreign Schema implementation - thread-safe Schema for type validation and inference
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignSchema {
    pub schema_type: SchemaType,
}

impl ForeignSchema {
    /// Create a new Schema
    pub fn new(schema_type: SchemaType) -> Self {
        ForeignSchema { schema_type }
    }

    /// Create common schema types
    pub fn int64() -> Self {
        ForeignSchema::new(SchemaType::Int64)
    }

    pub fn float64() -> Self {
        ForeignSchema::new(SchemaType::Float64)
    }

    pub fn bool() -> Self {
        ForeignSchema::new(SchemaType::Bool)
    }

    pub fn string() -> Self {
        ForeignSchema::new(SchemaType::String)
    }

    pub fn decimal(precision: u8, scale: u8) -> Self {
        ForeignSchema::new(SchemaType::Decimal { precision, scale })
    }

    pub fn date() -> Self {
        ForeignSchema::new(SchemaType::Date)
    }

    pub fn timestamp() -> Self {
        ForeignSchema::new(SchemaType::Timestamp)
    }

    pub fn uuid() -> Self {
        ForeignSchema::new(SchemaType::UUID)
    }

    pub fn nullable(inner: SchemaType) -> Self {
        ForeignSchema::new(SchemaType::Nullable(Box::new(inner)))
    }

    pub fn list(item_type: SchemaType) -> Self {
        ForeignSchema::new(SchemaType::List(Box::new(item_type)))
    }

    pub fn struct_type(fields: HashMap<String, SchemaType>) -> Self {
        let mut field_vec: Vec<(String, SchemaType)> = fields.into_iter().collect();
        field_vec.sort_by(|a, b| a.0.cmp(&b.0)); // Sort for consistent ordering
        ForeignSchema::new(SchemaType::Struct(field_vec))
    }

    /// Infer schema from a value
    pub fn infer_from_value(value: &Value) -> Self {
        let schema_type = Self::infer_schema_type(value);
        ForeignSchema::new(schema_type)
    }

    /// Infer schema from a list of values (like a column)
    pub fn infer_from_values(values: &[Value]) -> Self {
        if values.is_empty() {
            return ForeignSchema::string(); // Default for empty
        }

        // Find the most general type that can accommodate all values
        let mut inferred_type = None;
        let mut has_missing = false;

        for value in values {
            if matches!(value, Value::Missing) {
                has_missing = true;
                continue;
            }

            let value_type = Self::infer_schema_type(value);

            if inferred_type.is_none() {
                inferred_type = Some(value_type);
            } else {
                inferred_type = Some(Self::unify_types(inferred_type.unwrap(), value_type));
            }
        }

        let final_type = inferred_type.unwrap_or(SchemaType::String);

        if has_missing {
            ForeignSchema::new(SchemaType::Nullable(Box::new(final_type)))
        } else {
            ForeignSchema::new(final_type)
        }
    }

    /// Infer the schema type for a single value
    fn infer_schema_type(value: &Value) -> SchemaType {
        match value {
            Value::Integer(_) => SchemaType::Int64,
            Value::Real(_) => SchemaType::Float64,
            Value::Boolean(_) => SchemaType::Bool,
            Value::String(s) => {
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
            Value::List(items) => {
                if items.is_empty() {
                    SchemaType::List(Box::new(SchemaType::String)) // Default
                } else {
                    // Infer the item type from all items
                    let item_schema = Self::infer_from_values(items);
                    SchemaType::List(Box::new(item_schema.schema_type))
                }
            },
            Value::Missing => SchemaType::String, // Missing alone defaults to string
            Value::LyObj(obj) => {
                // For Foreign objects, delegate to their type name
                match obj.type_name() {
                    "Series" => {
                        // Would need to call a method to get series type info
                        // For now, default to a nullable string list
                        SchemaType::List(Box::new(SchemaType::Nullable(Box::new(SchemaType::String))))
                    },
                    "Table" => {
                        // Tables would need to infer struct type from columns
                        SchemaType::Struct(Vec::new())
                    },
                    "Dataset" => {
                        // Datasets would analyze nested structure
                        SchemaType::Struct(Vec::new())
                    },
                    "Tensor" => {
                        // Tensors are numeric arrays
                        SchemaType::List(Box::new(SchemaType::Float64))
                    },
                    _ => SchemaType::String, // Default for unknown Foreign types
                }
            },
            _ => SchemaType::String, // Default fallback
        }
    }

    /// Unify two schema types to find the most general type that accommodates both
    fn unify_types(type1: SchemaType, type2: SchemaType) -> SchemaType {
        if type1 == type2 {
            return type1;
        }

        match (type1, type2) {
            // Numeric type unification
            (SchemaType::Int64, SchemaType::Float64) | (SchemaType::Float64, SchemaType::Int64) => {
                SchemaType::Float64 // Float64 is more general than Int64
            },
            (SchemaType::Int64, SchemaType::Decimal { .. }) | (SchemaType::Decimal { .. }, SchemaType::Int64) => {
                SchemaType::Float64 // Simplify to Float64 for now
            },
            (SchemaType::Float64, SchemaType::Decimal { .. }) | (SchemaType::Decimal { .. }, SchemaType::Float64) => {
                SchemaType::Float64
            },

            // String type unification
            (SchemaType::Date, SchemaType::String) | (SchemaType::String, SchemaType::Date) => {
                SchemaType::String // String is more general
            },
            (SchemaType::Timestamp, SchemaType::String) | (SchemaType::String, SchemaType::Timestamp) => {
                SchemaType::String
            },
            (SchemaType::UUID, SchemaType::String) | (SchemaType::String, SchemaType::UUID) => {
                SchemaType::String
            },
            (SchemaType::Date, SchemaType::Timestamp) | (SchemaType::Timestamp, SchemaType::Date) => {
                SchemaType::String // Different date formats, generalize to string
            },

            // List unification
            (SchemaType::List(item1), SchemaType::List(item2)) => {
                let unified_item = Self::unify_types(*item1, *item2);
                SchemaType::List(Box::new(unified_item))
            },

            // When all else fails, use String as the most general type
            _ => SchemaType::String,
        }
    }

    /// Helper to check if a string looks like a date
    fn looks_like_date(s: &str) -> bool {
        // Simple heuristic: YYYY-MM-DD format
        s.len() == 10 && s.chars().nth(4) == Some('-') && s.chars().nth(7) == Some('-')
    }

    /// Helper to check if a string looks like a timestamp
    fn looks_like_timestamp(s: &str) -> bool {
        // Simple heuristic: contains both date and time parts
        s.contains(' ') && s.len() >= 19 && Self::looks_like_date(&s[..10])
    }

    /// Helper to check if a string looks like a UUID
    fn looks_like_uuid(s: &str) -> bool {
        // Simple heuristic: 8-4-4-4-12 format
        s.len() == 36 && s.matches('-').count() == 4
    }

    /// Validate a value against this schema
    pub fn validate(&self, value: &Value) -> VmResult<()> {
        if self.matches_type(value, &self.schema_type) {
            Ok(())
        } else {
            Err(VmError::TypeError {
                expected: format!("{:?}", self.schema_type),
                actual: format!("{:?}", value),
            })
        }
    }

    /// Cast a value to match this schema (with type conversion)
    pub fn cast(&self, value: &Value, strict: bool) -> VmResult<Value> {
        self.cast_to_type(value, &self.schema_type, strict)
    }

    /// Cast a value to a specific schema type
    fn cast_to_type(&self, value: &Value, target_type: &SchemaType, strict: bool) -> VmResult<Value> {
        // If value already matches, return as-is
        if self.matches_type(value, target_type) {
            return Ok(value.clone());
        }

        // Handle Missing values
        if matches!(value, Value::Missing) {
            match target_type {
                SchemaType::Nullable(_) => Ok(Value::Missing),
                _ if !strict => Ok(Value::Missing), // Lenient mode allows Missing anywhere
                _ => Err(VmError::TypeError {
                    expected: format!("{:?}", target_type),
                    actual: "Missing".to_string(),
                }),
            }
        } else {
            self.perform_cast(value, target_type, strict)
        }
    }

    /// Perform the actual type casting
    fn perform_cast(&self, value: &Value, target_type: &SchemaType, strict: bool) -> VmResult<Value> {
        match (value, target_type) {
            // Numeric conversions
            (Value::Integer(i), SchemaType::Float64) => Ok(Value::Real(*i as f64)),
            (Value::Integer(i), SchemaType::Decimal { .. }) => Ok(Value::Real(*i as f64)),
            (Value::Real(f), SchemaType::Int64) if !strict => Ok(Value::Integer(*f as i64)),
            (Value::Real(_), SchemaType::Decimal { .. }) => Ok(value.clone()), // Accept as-is for now

            // String conversions
            (Value::Integer(i), SchemaType::String) => Ok(Value::String(i.to_string())),
            (Value::Real(f), SchemaType::String) => Ok(Value::String(f.to_string())),
            (Value::Boolean(b), SchemaType::String) => Ok(Value::String(b.to_string())),

            // Parse from string
            (Value::String(s), SchemaType::Int64) if !strict => {
                s.parse::<i64>()
                    .map(Value::Integer)
                    .map_err(|_| VmError::TypeError {
                        expected: "parseable integer".to_string(),
                        actual: format!("string '{}'", s),
                    })
            },
            (Value::String(s), SchemaType::Float64) if !strict => {
                s.parse::<f64>()
                    .map(Value::Real)
                    .map_err(|_| VmError::TypeError {
                        expected: "parseable float".to_string(),
                        actual: format!("string '{}'", s),
                    })
            },
            (Value::String(s), SchemaType::Bool) if !strict => {
                match s.to_lowercase().as_str() {
                    "true" | "t" | "1" | "yes" | "y" => Ok(Value::Boolean(true)),
                    "false" | "f" | "0" | "no" | "n" => Ok(Value::Boolean(false)),
                    _ => Err(VmError::TypeError {
                        expected: "parseable boolean".to_string(),
                        actual: format!("string '{}'", s),
                    }),
                }
            },

            // Handle nullable types
            (_, SchemaType::Nullable(inner)) => self.cast_to_type(value, inner, strict),

            // List casting
            (Value::List(items), SchemaType::List(item_type)) => {
                let mut cast_items = Vec::new();
                for item in items {
                    cast_items.push(self.cast_to_type(item, item_type, strict)?);
                }
                Ok(Value::List(cast_items))
            },

            // Default: no conversion possible
            _ => Err(VmError::TypeError {
                expected: format!("{:?}", target_type),
                actual: format!("{:?}", value),
            }),
        }
    }

    /// Validate and cast a value (convenience method)
    pub fn validate_and_cast(&self, value: &Value, strict: bool) -> VmResult<Value> {
        // Try validation first
        if self.validate(value).is_ok() {
            Ok(value.clone())
        } else {
            // If validation fails, try casting
            self.cast(value, strict)
        }
    }

    /// Check if a value matches a schema type
    fn matches_type(&self, value: &Value, schema_type: &SchemaType) -> bool {
        match (value, schema_type) {
            (Value::Integer(_), SchemaType::Int64) => true,
            (Value::Real(_), SchemaType::Float64) => true,
            (Value::Boolean(_), SchemaType::Bool) => true,
            (Value::String(_), SchemaType::String) => true,
            (Value::Real(_), SchemaType::Decimal { .. }) => true, // For now, accept floats as decimals
            (Value::String(s), SchemaType::Date) => {
                // Basic date validation - check if string looks like a date
                self.is_valid_date_string(s)
            },
            (Value::String(s), SchemaType::Timestamp) => {
                // Basic timestamp validation
                self.is_valid_timestamp_string(s)
            },
            (Value::String(s), SchemaType::UUID) => {
                // Basic UUID validation
                self.is_valid_uuid_string(s)
            },
            (_, SchemaType::Nullable(inner)) => {
                // Null is represented as Missing in our system
                matches!(value, Value::Missing) || self.matches_type(value, inner)
            },
            (Value::List(items), SchemaType::List(item_type)) => {
                items.iter().all(|item| self.matches_type(item, item_type))
            },
            (Value::List(items), SchemaType::Struct(fields)) => {
                // For struct validation, we expect a list of key-value pairs
                // This is a simple implementation - could be improved
                self.validate_struct_items(items, fields)
            },
            _ => false,
        }
    }

    /// Validate date string format (simple implementation)
    fn is_valid_date_string(&self, s: &str) -> bool {
        // Simple validation: YYYY-MM-DD format
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 3 {
            return false;
        }

        // Check year (4 digits)
        if parts[0].len() != 4 || !parts[0].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }

        // Check month (01-12)
        if parts[1].len() != 2 || !parts[1].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if let Ok(month) = parts[1].parse::<u32>() {
            if month < 1 || month > 12 {
                return false;
            }
        } else {
            return false;
        }

        // Check day (01-31)
        if parts[2].len() != 2 || !parts[2].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if let Ok(day) = parts[2].parse::<u32>() {
            if day < 1 || day > 31 {
                return false;
            }
        } else {
            return false;
        }

        true
    }

    /// Validate timestamp string format (simple implementation)
    fn is_valid_timestamp_string(&self, s: &str) -> bool {
        // Simple validation: YYYY-MM-DD HH:MM:SS format
        if !s.contains(' ') || s.len() < 19 {
            return false;
        }

        let parts: Vec<&str> = s.splitn(2, ' ').collect();
        if parts.len() != 2 {
            return false;
        }

        // Check date part
        if !self.is_valid_date_string(parts[0]) {
            return false;
        }

        // Check time part (HH:MM:SS)
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() != 3 {
            return false;
        }

        // Validate hours (00-23)
        if time_parts[0].len() != 2 || !time_parts[0].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if let Ok(hour) = time_parts[0].parse::<u32>() {
            if hour > 23 {
                return false;
            }
        } else {
            return false;
        }

        // Validate minutes (00-59)
        if time_parts[1].len() != 2 || !time_parts[1].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if let Ok(minute) = time_parts[1].parse::<u32>() {
            if minute > 59 {
                return false;
            }
        } else {
            return false;
        }

        // Validate seconds (00-59)
        if time_parts[2].len() != 2 || !time_parts[2].chars().all(|c| c.is_ascii_digit()) {
            return false;
        }
        if let Ok(second) = time_parts[2].parse::<u32>() {
            if second > 59 {
                return false;
            }
        } else {
            return false;
        }

        true
    }

    /// Validate UUID string format (simple implementation)
    fn is_valid_uuid_string(&self, s: &str) -> bool {
        // Simple validation: 8-4-4-4-12 format
        if s.len() != 36 || s.matches('-').count() != 4 {
            return false;
        }

        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 5 || 
           parts[0].len() != 8 || 
           parts[1].len() != 4 || 
           parts[2].len() != 4 || 
           parts[3].len() != 4 || 
           parts[4].len() != 12 {
            return false;
        }

        // Check all parts contain only hex digits
        for part in parts {
            if !part.chars().all(|c| c.is_ascii_hexdigit()) {
                return false;
            }
        }

        true
    }

    /// Validate struct items (simple implementation)
    fn validate_struct_items(&self, items: &[Value], _fields: &[(String, SchemaType)]) -> bool {
        // Simple implementation - for now, just check it's a list
        // In a full implementation, we'd validate each field matches expected type
        !items.is_empty()
    }

    /// Convert to type name string
    pub fn type_name_string(&self) -> String {
        match &self.schema_type {
            SchemaType::Int64 => "Integer".to_string(),
            SchemaType::Float64 => "Real".to_string(),
            SchemaType::Bool => "Boolean".to_string(),
            SchemaType::String => "String".to_string(),
            SchemaType::Decimal { precision, scale } => format!("Decimal[{},{}]", precision, scale),
            SchemaType::Date => "Date".to_string(),
            SchemaType::Timestamp => "Timestamp".to_string(),
            SchemaType::UUID => "UUID".to_string(),
            SchemaType::Nullable(inner) => {
                let inner_schema = ForeignSchema::new((**inner).clone());
                format!("Nullable[{}]", inner_schema.type_name_string())
            },
            SchemaType::List(item_type) => {
                let item_schema = ForeignSchema::new((**item_type).clone());
                format!("List[{}]", item_schema.type_name_string())
            },
            SchemaType::Struct(fields) => {
                let field_strs: Vec<String> = fields.iter()
                    .map(|(name, schema_type)| {
                        let type_schema = ForeignSchema::new(schema_type.clone());
                        format!("{}: {}", name, type_schema.type_name_string())
                    })
                    .collect();
                format!("Struct[{}]", field_strs.join(", "))
            },
        }
    }
}

impl Foreign for ForeignSchema {
    fn type_name(&self) -> &'static str {
        "Schema"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Type" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.type_name_string()))
            }
            "Validate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match self.validate(&args[0]) {
                    Ok(()) => Ok(Value::Boolean(true)),
                    Err(_) => Ok(Value::Boolean(false)),
                }
            }
            "Cast" => {
                if args.len() != 1 && args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let strict = if args.len() == 2 {
                    match &args[1] {
                        Value::Boolean(b) => *b,
                        _ => return Err(ForeignError::InvalidArgumentType {
                            method: method.to_string(),
                            expected: "Boolean".to_string(),
                            actual: format!("{:?}", args[1]),
                        }),
                    }
                } else {
                    false // Default to non-strict
                };
                
                match self.cast(&args[0], strict) {
                    Ok(cast_value) => Ok(cast_value),
                    Err(e) => Err(ForeignError::RuntimeError {
                        message: format!("Cast operation error: {}", e),
                    }),
                }
            }
            "InferFromValue" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let inferred_schema = ForeignSchema::infer_from_value(&args[0]);
                Ok(Value::LyObj(LyObj::new(Box::new(inferred_schema))))
            }
            "InferFromValues" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::List(values) => {
                        let inferred_schema = ForeignSchema::infer_from_values(values);
                        Ok(Value::LyObj(LyObj::new(Box::new(inferred_schema))))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            _ => Err(ForeignError::UnknownMethod {
                type_name: self.type_name().to_string(),
                method: method.to_string(),
            }),
        }
    }

    fn clone_boxed(&self) -> Box<dyn Foreign> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}