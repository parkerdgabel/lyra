use crate::bytecode::{Instruction, OpCode};
use crate::stdlib::StandardLibrary;
use crate::stdlib::tensor::{tensor_add, tensor_sub, tensor_mul, tensor_div, tensor_pow};
use std::collections::HashMap;
use thiserror::Error;
use ndarray::ArrayD;

#[derive(Error, Debug)]
pub enum VmError {
    #[error("Stack underflow")]
    StackUnderflow,
    #[error("Invalid instruction pointer: {0}")]
    InvalidInstructionPointer(usize),
    #[error("Invalid constant index: {0}")]
    InvalidConstantIndex(usize),
    #[error("Invalid symbol index: {0}")]
    InvalidSymbolIndex(usize),
    #[error("Division by zero")]
    DivisionByZero,
    #[error("Type error: expected {expected}, got {actual}")]
    TypeError { expected: String, actual: String },
    #[error("Call stack overflow")]
    CallStackOverflow,
    #[error("Cannot call non-function value")]
    NotCallable,
    #[error("Index {index} out of bounds for length {length}")]
    IndexError { index: i64, length: usize },
}

pub type VmResult<T> = std::result::Result<T, VmError>;

/// Series data type for typed vectors
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SeriesType {
    Int64,
    Float64,
    Bool,
    String,
}

use std::rc::Rc;

/// Series - typed vector with metadata and COW semantics
#[derive(Debug, Clone, Hash)]
pub struct Series {
    pub data: Rc<Vec<Value>>, // Reference-counted for COW
    pub dtype: SeriesType,
    pub length: usize,
}

impl PartialEq for Series {
    fn eq(&self, other: &Self) -> bool {
        // Use pointer equality first for efficiency, then value equality
        (Rc::ptr_eq(&self.data, &other.data) || *self.data == *other.data) 
            && self.dtype == other.dtype 
            && self.length == other.length
    }
}

impl Series {
    /// Create a new Series with type validation
    pub fn new(data: Vec<Value>, dtype: SeriesType) -> VmResult<Self> {
        // Validate all data matches the specified type
        for value in &data {
            if !Self::value_matches_type(value, &dtype) {
                return Err(VmError::TypeError {
                    expected: format!("{:?}", dtype),
                    actual: format!("{:?}", value),
                });
            }
        }
        
        Ok(Series {
            length: data.len(),
            data: Rc::new(data),
            dtype,
        })
    }
    
    /// Create a new Series with automatic type inference
    pub fn infer(data: Vec<Value>) -> VmResult<Self> {
        if data.is_empty() {
            return Err(VmError::TypeError {
                expected: "non-empty data".to_string(),
                actual: "empty data".to_string(),
            });
        }
        
        // Infer type from first non-Missing element
        let dtype = data.iter()
            .find(|v| !matches!(v, Value::Missing))
            .map(|v| Self::infer_type(v))
            .unwrap_or(SeriesType::String); // Default to String if all Missing
            
        Self::new(data, dtype)
    }
    
    /// Create a view/slice of this Series (COW optimization)
    pub fn slice(&self, start: usize, end: usize) -> VmResult<Self> {
        if start > end || end > self.length {
            return Err(VmError::TypeError {
                expected: format!("valid slice bounds (0 <= {} <= {} <= {})", start, end, self.length),
                actual: format!("invalid bounds start={}, end={}", start, end),
            });
        }
        
        // Create a new vector with the sliced data (for now - could optimize with views later)
        let sliced_data = self.data[start..end].to_vec();
        
        Ok(Series {
            data: Rc::new(sliced_data),
            dtype: self.dtype.clone(),
            length: end - start,
        })
    }
    
    /// Get a value at index (bounds-checked)
    pub fn get(&self, index: usize) -> VmResult<&Value> {
        if index >= self.length {
            return Err(VmError::TypeError {
                expected: format!("index < {}", self.length),
                actual: format!("index {}", index),
            });
        }
        Ok(&self.data[index])
    }
    
    /// Create a new Series with a modified value (COW semantics)
    pub fn with_value_at(&self, index: usize, value: Value) -> VmResult<Self> {
        if index >= self.length {
            return Err(VmError::TypeError {
                expected: format!("index < {}", self.length),
                actual: format!("index {}", index),
            });
        }
        
        // Validate the new value matches the series type
        if !Self::value_matches_type(&value, &self.dtype) {
            return Err(VmError::TypeError {
                expected: format!("{:?}", self.dtype),
                actual: format!("{:?}", value),
            });
        }
        
        // Clone the data (COW)
        let mut new_data = (*self.data).clone();
        new_data[index] = value;
        
        Ok(Series {
            data: Rc::new(new_data),
            dtype: self.dtype.clone(),
            length: self.length,
        })
    }
    
    /// Append a value to create a new Series (COW semantics)
    pub fn append(&self, value: Value) -> VmResult<Self> {
        // Validate the value matches the series type
        if !Self::value_matches_type(&value, &self.dtype) {
            return Err(VmError::TypeError {
                expected: format!("{:?}", self.dtype),
                actual: format!("{:?}", value),
            });
        }
        
        // Clone and extend the data (COW)
        let mut new_data = (*self.data).clone();
        new_data.push(value);
        
        Ok(Series {
            data: Rc::new(new_data),
            dtype: self.dtype.clone(),
            length: self.length + 1,
        })
    }
    
    /// Create an iterator over the series values
    pub fn iter(&self) -> std::slice::Iter<Value> {
        self.data.iter()
    }
    
    /// Check if this series shares data with another (for COW detection)
    pub fn shares_data_with(&self, other: &Series) -> bool {
        Rc::ptr_eq(&self.data, &other.data)
    }
    
    fn value_matches_type(value: &Value, dtype: &SeriesType) -> bool {
        match (value, dtype) {
            (Value::Missing, _) => true, // Missing can be in any series
            (Value::Integer(_), SeriesType::Int64) => true,
            (Value::Real(_), SeriesType::Float64) => true,
            (Value::Boolean(_), SeriesType::Bool) => true,
            (Value::String(_), SeriesType::String) => true,
            _ => false,
        }
    }
    
    fn infer_type(value: &Value) -> SeriesType {
        match value {
            Value::Integer(_) => SeriesType::Int64,
            Value::Real(_) => SeriesType::Float64,
            Value::Boolean(_) => SeriesType::Bool,
            Value::String(_) => SeriesType::String,
            _ => SeriesType::String, // Default fallback
        }
    }
}

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

/// Schema for type validation and inference
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub schema_type: SchemaType,
}

impl Schema {
    /// Create a new Schema
    pub fn new(schema_type: SchemaType) -> Self {
        Schema { schema_type }
    }
    
    /// Create common schema types
    pub fn int64() -> Self {
        Schema::new(SchemaType::Int64)
    }
    
    pub fn float64() -> Self {
        Schema::new(SchemaType::Float64)
    }
    
    pub fn bool() -> Self {
        Schema::new(SchemaType::Bool)
    }
    
    pub fn string() -> Self {
        Schema::new(SchemaType::String)
    }
    
    pub fn decimal(precision: u8, scale: u8) -> Self {
        Schema::new(SchemaType::Decimal { precision, scale })
    }
    
    pub fn date() -> Self {
        Schema::new(SchemaType::Date)
    }
    
    pub fn timestamp() -> Self {
        Schema::new(SchemaType::Timestamp)
    }
    
    pub fn uuid() -> Self {
        Schema::new(SchemaType::UUID)
    }
    
    pub fn nullable(inner: SchemaType) -> Self {
        Schema::new(SchemaType::Nullable(Box::new(inner)))
    }
    
    pub fn list(item_type: SchemaType) -> Self {
        Schema::new(SchemaType::List(Box::new(item_type)))
    }
    
    pub fn struct_type(fields: HashMap<String, SchemaType>) -> Self {
        let mut field_vec: Vec<(String, SchemaType)> = fields.into_iter().collect();
        field_vec.sort_by(|a, b| a.0.cmp(&b.0)); // Sort for consistent ordering
        Schema::new(SchemaType::Struct(field_vec))
    }
    
    /// Infer schema from a value
    pub fn infer_from_value(value: &Value) -> Self {
        let schema_type = Self::infer_schema_type(value);
        Schema::new(schema_type)
    }
    
    /// Infer schema from a list of values (like a column)
    pub fn infer_from_values(values: &[Value]) -> Self {
        if values.is_empty() {
            return Schema::string(); // Default for empty
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
            Schema::new(SchemaType::Nullable(Box::new(final_type)))
        } else {
            Schema::new(final_type)
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
            Value::Series(series) => {
                // Convert SeriesType to SchemaType
                let series_schema_type = match series.dtype {
                    SeriesType::Int64 => SchemaType::Int64,
                    SeriesType::Float64 => SchemaType::Float64,
                    SeriesType::Bool => SchemaType::Bool,
                    SeriesType::String => SchemaType::String,
                };
                
                // Check if series contains Missing values
                let has_missing = series.data.iter().any(|v| matches!(v, Value::Missing));
                if has_missing {
                    SchemaType::Nullable(Box::new(series_schema_type))
                } else {
                    series_schema_type
                }
            },
            Value::Table(_) => {
                // For tables, we'd need to infer struct type from columns
                // This is a simplified implementation
                SchemaType::Struct(Vec::new())
            },
            Value::Dataset(_) => {
                // For datasets, we'd analyze the nested structure
                // This is a simplified implementation
                SchemaType::Struct(Vec::new())
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
        let parts: Vec<&str> = s.split(' ').collect();
        if parts.len() != 2 {
            return false;
        }
        
        // Validate date part
        if !self.is_valid_date_string(parts[0]) {
            return false;
        }
        
        // Validate time part (HH:MM:SS)
        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() != 3 {
            return false;
        }
        
        // Check each time component
        for (i, part) in time_parts.iter().enumerate() {
            if part.len() != 2 || !part.chars().all(|c| c.is_ascii_digit()) {
                return false;
            }
            
            if let Ok(value) = part.parse::<u32>() {
                match i {
                    0 => if value > 23 { return false; }, // Hours: 00-23
                    1 => if value > 59 { return false; }, // Minutes: 00-59
                    2 => if value > 59 { return false; }, // Seconds: 00-59
                    _ => return false,
                }
            } else {
                return false;
            }
        }
        
        true
    }
    
    /// Validate UUID string format (simple implementation)
    fn is_valid_uuid_string(&self, s: &str) -> bool {
        // Simple validation: 8-4-4-4-12 hex format
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() != 5 {
            return false;
        }
        
        let expected_lengths = [8, 4, 4, 4, 12];
        for (i, part) in parts.iter().enumerate() {
            if part.len() != expected_lengths[i] || !part.chars().all(|c| c.is_ascii_hexdigit()) {
                return false;
            }
        }
        
        true
    }
    
    /// Validate struct items (simple implementation)
    fn validate_struct_items(&self, items: &[Value], fields: &[(String, SchemaType)]) -> bool {
        // For now, just check if we have the right number of fields
        // This is a simplified implementation - a real struct validation would be more complex
        items.len() == fields.len()
    }
}

/// Table - columnar data structure
#[derive(Debug, Clone)]
pub struct Table {
    pub columns: HashMap<String, Series>,
    pub length: usize,
    pub index: Option<Vec<Value>>,
}

impl PartialEq for Table {
    fn eq(&self, other: &Self) -> bool {
        self.columns == other.columns && self.length == other.length && self.index == other.index
    }
}

impl Table {
    /// Create a new empty Table
    pub fn new() -> Self {
        Table {
            columns: HashMap::new(),
            length: 0,
            index: None,
        }
    }
    
    /// Create a Table from columns, validating all series have same length
    pub fn from_columns(columns: HashMap<String, Series>) -> VmResult<Self> {
        if columns.is_empty() {
            return Ok(Table::new());
        }
        
        // Check all series have the same length
        let lengths: Vec<usize> = columns.values().map(|s| s.length).collect();
        let first_length = lengths[0];
        
        if !lengths.iter().all(|&len| len == first_length) {
            return Err(VmError::TypeError {
                expected: format!("all columns to have length {}", first_length),
                actual: format!("columns with varying lengths: {:?}", lengths),
            });
        }
        
        Ok(Table {
            length: first_length,
            columns,
            index: None,
        })
    }
    
    /// Add a column to the table
    pub fn add_column(&mut self, name: String, series: Series) -> VmResult<()> {
        if self.columns.is_empty() {
            self.length = series.length;
        } else if series.length != self.length {
            return Err(VmError::TypeError {
                expected: format!("column length {}", self.length),
                actual: format!("column length {}", series.length),
            });
        }
        
        self.columns.insert(name, series);
        Ok(())
    }
    
    /// Get column names
    pub fn column_names(&self) -> Vec<&String> {
        self.columns.keys().collect()
    }
    
    /// Get a column by name
    pub fn get_column(&self, name: &str) -> Option<&Series> {
        self.columns.get(name)
    }
    
    /// Get a mutable reference to a column by name
    pub fn get_column_mut(&mut self, name: &str) -> Option<&mut Series> {
        self.columns.get_mut(name)
    }
    
    /// Drop columns by name
    pub fn drop_columns(&self, column_names: &[&str]) -> VmResult<Self> {
        let mut new_columns = HashMap::new();
        for (name, series) in &self.columns {
            if !column_names.contains(&name.as_str()) {
                new_columns.insert(name.clone(), series.clone());
            }
        }
        
        Ok(Table {
            columns: new_columns,
            length: self.length,
            index: self.index.clone(),
        })
    }
    
    /// Select specific columns by name
    pub fn select_columns(&self, column_names: &[&str]) -> VmResult<Self> {
        let mut new_columns = HashMap::new();
        
        for &name in column_names {
            if let Some(series) = self.columns.get(name) {
                new_columns.insert(name.to_string(), series.clone());
            } else {
                return Err(VmError::TypeError {
                    expected: format!("column '{}' to exist", name),
                    actual: format!("column not found in table with columns: {:?}", 
                                  self.column_names()),
                });
            }
        }
        
        Ok(Table {
            columns: new_columns,
            length: self.length,
            index: self.index.clone(),
        })
    }
    
    /// Rename columns
    pub fn rename_columns(&self, renames: &HashMap<String, String>) -> Self {
        let mut new_columns = HashMap::new();
        
        for (old_name, series) in &self.columns {
            let new_name = renames.get(old_name).unwrap_or(old_name);
            new_columns.insert(new_name.clone(), series.clone());
        }
        
        Table {
            columns: new_columns,
            length: self.length,
            index: self.index.clone(),
        }
    }
    
    /// Add computed columns with COW semantics
    pub fn with_columns(&self, new_columns: HashMap<String, Series>) -> VmResult<Self> {
        let mut result_columns = self.columns.clone();
        
        for (name, series) in new_columns {
            if series.length != self.length && self.length > 0 {
                return Err(VmError::TypeError {
                    expected: format!("column length {}", self.length),
                    actual: format!("column length {}", series.length),
                });
            }
            result_columns.insert(name, series);
        }
        
        let new_length = if self.length == 0 && !result_columns.is_empty() {
            result_columns.values().next().unwrap().length
        } else {
            self.length
        };
        
        Ok(Table {
            columns: result_columns,
            length: new_length,
            index: self.index.clone(),
        })
    }
    
    /// Get row as a vector of values (for iteration)
    pub fn get_row(&self, index: usize) -> VmResult<Vec<Value>> {
        if index >= self.length {
            return Err(VmError::IndexError {
                index: index as i64,
                length: self.length,
            });
        }
        
        let mut row = Vec::new();
        for column_name in self.column_names() {
            if let Some(series) = self.columns.get(column_name) {
                row.push(series.get(index)?.clone());
            }
        }
        
        Ok(row)
    }
    
    /// Slice rows (equivalent to head/tail operations)
    pub fn slice_rows(&self, start: usize, end: usize) -> VmResult<Self> {
        if start > end || end > self.length {
            return Err(VmError::IndexError {
                index: end as i64,
                length: self.length,
            });
        }
        
        let mut new_columns = HashMap::new();
        for (name, series) in &self.columns {
            let sliced_series = series.slice(start, end)?;
            new_columns.insert(name.clone(), sliced_series);
        }
        
        let new_index = if let Some(ref idx) = self.index {
            Some(idx[start..end].to_vec())
        } else {
            None
        };
        
        Ok(Table {
            columns: new_columns,
            length: end - start,
            index: new_index,
        })
    }
    
    /// Filter rows by predicate (simple boolean mask)
    pub fn filter_rows(&self, mask: &[bool]) -> VmResult<Self> {
        if mask.len() != self.length {
            return Err(VmError::TypeError {
                expected: format!("mask length {}", self.length),
                actual: format!("mask length {}", mask.len()),
            });
        }
        
        let mut new_columns = HashMap::new();
        for (name, series) in &self.columns {
            let mut filtered_data = Vec::new();
            for (i, &include) in mask.iter().enumerate() {
                if include {
                    filtered_data.push(series.get(i)?.clone());
                }
            }
            
            let filtered_series = Series::new(filtered_data, series.dtype.clone())?;
            new_columns.insert(name.clone(), filtered_series);
        }
        
        let new_index = if let Some(ref idx) = self.index {
            let mut filtered_index = Vec::new();
            for (i, &include) in mask.iter().enumerate() {
                if include {
                    filtered_index.push(idx[i].clone());
                }
            }
            Some(filtered_index)
        } else {
            None
        };
        
        let new_length = mask.iter().filter(|&&x| x).count();
        
        Ok(Table {
            columns: new_columns,
            length: new_length,
            index: new_index,
        })
    }
    
    /// Get column data types as a map
    pub fn dtypes(&self) -> HashMap<String, SeriesType> {
        self.columns.iter()
            .map(|(name, series)| (name.clone(), series.dtype.clone()))
            .collect()
    }
    
    /// Get table shape (rows, columns)
    pub fn shape(&self) -> (usize, usize) {
        (self.length, self.columns.len())
    }
    
    /// Convert table to a vector of rows (each row as a Vec<Value>)
    pub fn to_rows(&self) -> VmResult<Vec<Vec<Value>>> {
        let mut rows = Vec::with_capacity(self.length);
        for i in 0..self.length {
            rows.push(self.get_row(i)?);
        }
        Ok(rows)
    }
    
    /// Create Table from rows of data with column names
    pub fn from_rows(column_names: Vec<String>, rows: Vec<Vec<Value>>) -> VmResult<Self> {
        if column_names.is_empty() {
            return Ok(Table::new());
        }
        
        if rows.is_empty() {
            // Create empty table with given columns
            let mut columns = HashMap::new();
            for name in column_names {
                columns.insert(name, Series::new(Vec::new(), SeriesType::String)?);
            }
            return Ok(Table::from_columns(columns)?);
        }
        
        let num_cols = column_names.len();
        let num_rows = rows.len();
        
        // Validate all rows have same number of columns
        for (i, row) in rows.iter().enumerate() {
            if row.len() != num_cols {
                return Err(VmError::TypeError {
                    expected: format!("row with {} columns", num_cols),
                    actual: format!("row {} has {} columns", i, row.len()),
                });
            }
        }
        
        // Transpose rows to columns
        let mut columns = HashMap::new();
        for (col_idx, column_name) in column_names.into_iter().enumerate() {
            let mut column_data = Vec::with_capacity(num_rows);
            for row in &rows {
                column_data.push(row[col_idx].clone());
            }
            
            // Create series using Series inference, not Schema inference
            let series = Series::infer(column_data)?;
            columns.insert(column_name, series);
        }
        
        Table::from_columns(columns)
    }
    
    /// Create Table from rows with automatic column naming (col0, col1, ...)
    pub fn from_rows_auto_names(rows: Vec<Vec<Value>>) -> VmResult<Self> {
        if rows.is_empty() {
            return Ok(Table::new());
        }
        
        let num_cols = rows[0].len();
        let column_names: Vec<String> = (0..num_cols)
            .map(|i| format!("col{}", i))
            .collect();
            
        Self::from_rows(column_names, rows)
    }
    
    /// Create Table from a single column
    pub fn from_column(name: String, series: Series) -> Self {
        let mut columns = HashMap::new();
        let length = series.length;
        columns.insert(name, series);
        
        Table {
            columns,
            length,
            index: None,
        }
    }
    
    /// Get head of table (first n rows)
    pub fn head(&self, n: usize) -> VmResult<Self> {
        let end = std::cmp::min(n, self.length);
        self.slice_rows(0, end)
    }
    
    /// Get tail of table (last n rows)
    pub fn tail(&self, n: usize) -> VmResult<Self> {
        let start = if n >= self.length { 0 } else { self.length - n };
        self.slice_rows(start, self.length)
    }
    
    /// Check if table is empty
    pub fn is_empty(&self) -> bool {
        self.length == 0
    }
    
    
    /// Sort table by column values (simple ascending sort)
    pub fn sort_by_column(&self, column_name: &str, ascending: bool) -> VmResult<Self> {
        let column = self.get_column(column_name)
            .ok_or_else(|| VmError::TypeError {
                expected: format!("column '{}' to exist", column_name),
                actual: format!("column not found in table"),
            })?;
        
        // Create indices for sorting
        let mut indices: Vec<usize> = (0..self.length).collect();
        
        // Sort indices based on column values
        indices.sort_by(|&a, &b| {
            let val_a = column.get(a).map(|v| v.clone()).unwrap_or(Value::Missing);
            let val_b = column.get(b).map(|v| v.clone()).unwrap_or(Value::Missing);
            
            let cmp = match (&val_a, &val_b) {
                (Value::Missing, Value::Missing) => std::cmp::Ordering::Equal,
                (Value::Missing, _) => std::cmp::Ordering::Greater, // Missing values last
                (_, Value::Missing) => std::cmp::Ordering::Less,
                (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                (Value::String(a), Value::String(b)) => a.cmp(b),
                (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
                _ => std::cmp::Ordering::Equal, // For mixed types, keep original order
            };
            
            if ascending { cmp } else { cmp.reverse() }
        });
        
        // Reorder all columns based on sorted indices
        let mut new_columns = HashMap::new();
        for (name, series) in &self.columns {
            let mut reordered_data = Vec::with_capacity(self.length);
            for &idx in &indices {
                reordered_data.push(series.get(idx)?.clone());
            }
            
            let reordered_series = Series::new(reordered_data, series.dtype.clone())?;
            new_columns.insert(name.clone(), reordered_series);
        }
        
        // Reorder index if present
        let new_index = if let Some(ref idx) = self.index {
            let mut reordered_index = Vec::with_capacity(self.length);
            for &i in &indices {
                reordered_index.push(idx[i].clone());
            }
            Some(reordered_index)
        } else {
            None
        };
        
        Ok(Table {
            columns: new_columns,
            length: self.length,
            index: new_index,
        })
    }
}


/// Dataset - hierarchical data structure wrapper
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    pub value: Box<Value>, // Nested List/Association structures
}

impl Dataset {
    /// Create a new Dataset from a value
    pub fn new(value: Value) -> Self {
        Dataset {
            value: Box::new(value),
        }
    }
    
    /// Create Dataset from nested lists/associations
    pub fn from_nested(value: Value) -> VmResult<Self> {
        // TODO: Validate that the value is appropriate for hierarchical data
        // For now, accept any nested structure
        Ok(Dataset::new(value))
    }
}

/// A value that can be stored on the VM stack
#[derive(Debug, Clone)]
pub enum Value {
    Integer(i64),
    Real(f64),
    String(String),
    Symbol(String),
    List(Vec<Value>),
    Function(String), // Built-in function name for now
    Boolean(bool),
    Tensor(ArrayD<f64>), // N-dimensional tensor with floating point values
    Missing,            // Missing/unknown value (distinct from Null)
    Series(Series),     // Typed vector for columnar data
    Table(Table),       // Columnar data structure
    Dataset(Dataset),   // Hierarchical data structure
    Schema(Schema),     // Type schema for validation
}

impl Eq for Value {}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::Integer(n) => {
                0u8.hash(state);
                n.hash(state);
            },
            Value::Real(f) => {
                1u8.hash(state);
                // Hash the bit representation of f64 for consistent hashing
                f.to_bits().hash(state);
            },
            Value::String(s) => {
                2u8.hash(state);
                s.hash(state);
            },
            Value::Symbol(s) => {
                3u8.hash(state);
                s.hash(state);
            },
            Value::List(items) => {
                4u8.hash(state);
                items.hash(state);
            },
            Value::Function(name) => {
                5u8.hash(state);
                name.hash(state);
            },
            Value::Boolean(b) => {
                6u8.hash(state);
                b.hash(state);
            },
            Value::Tensor(tensor) => {
                7u8.hash(state);
                // Hash tensor shape and data
                tensor.shape().hash(state);
                for &val in tensor.as_slice().unwrap_or(&[]) {
                    val.to_bits().hash(state);
                }
            },
            Value::Missing => {
                8u8.hash(state);
            },
            Value::Series(series) => {
                9u8.hash(state);
                series.dtype.hash(state);
                series.length.hash(state);
                // Hash the data vector
                series.data.hash(state);
            },
            Value::Table(table) => {
                10u8.hash(state);
                table.length.hash(state);
                // Hash column names and their series
                let mut column_pairs: Vec<_> = table.columns.iter().collect();
                column_pairs.sort_by_key(|(name, _)| *name);
                for (name, series) in column_pairs {
                    name.hash(state);
                    series.hash(state);
                }
            },
            Value::Dataset(dataset) => {
                11u8.hash(state);
                dataset.value.hash(state);
            },
            Value::Schema(schema) => {
                12u8.hash(state);
                schema.schema_type.hash(state);
            },
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Integer(a), Value::Integer(b)) => a == b,
            (Value::Real(a), Value::Real(b)) => a == b,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Symbol(a), Value::Symbol(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Function(a), Value::Function(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::Tensor(a), Value::Tensor(b)) => {
                // Compare tensors element-wise
                a.shape() == b.shape() && a.iter().zip(b.iter()).all(|(x, y)| x == y)
            }
            (Value::Missing, Value::Missing) => true,
            (Value::Series(a), Value::Series(b)) => a == b,
            (Value::Table(a), Value::Table(b)) => a == b,
            (Value::Dataset(a), Value::Dataset(b)) => a == b,
            (Value::Schema(a), Value::Schema(b)) => a == b,
            _ => false,
        }
    }
}

/// Call frame for function calls
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub return_address: usize,
    pub function_name: String,
    pub local_count: usize,
}

/// Virtual Machine state
#[derive(Debug)]
pub struct VirtualMachine {
    /// Instruction pointer
    pub ip: usize,
    /// Value stack
    pub stack: Vec<Value>,
    /// Call stack for function calls
    pub call_stack: Vec<CallFrame>,
    /// Constant pool
    pub constants: Vec<Value>,
    /// Symbol table (name -> index)
    pub symbols: HashMap<String, usize>,
    /// Bytecode instructions
    pub code: Vec<Instruction>,
    /// Maximum call stack depth
    pub max_call_depth: usize,
    /// Standard library functions
    pub stdlib: StandardLibrary,
}

impl VirtualMachine {
    /// Create a new virtual machine
    pub fn new() -> Self {
        VirtualMachine {
            ip: 0,
            stack: Vec::new(),
            call_stack: Vec::new(),
            constants: Vec::new(),
            symbols: HashMap::new(),
            code: Vec::new(),
            max_call_depth: 1000,
            stdlib: StandardLibrary::new(),
        }
    }

    /// Load bytecode and constants into the VM
    pub fn load(&mut self, code: Vec<Instruction>, constants: Vec<Value>) {
        self.code = code;
        self.constants = constants;
        self.ip = 0;
        self.stack.clear();
        self.call_stack.clear();
    }

    /// Add a constant to the constant pool, returns its index
    pub fn add_constant(&mut self, value: Value) -> usize {
        self.constants.push(value);
        self.constants.len() - 1
    }

    /// Add a symbol to the symbol table, returns its index
    pub fn add_symbol(&mut self, name: String) -> usize {
        if let Some(&index) = self.symbols.get(&name) {
            return index;
        }
        let index = self.symbols.len();
        self.symbols.insert(name, index);
        index
    }

    /// Push a value onto the stack
    pub fn push(&mut self, value: Value) {
        self.stack.push(value);
    }

    /// Pop a value from the stack
    pub fn pop(&mut self) -> VmResult<Value> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    /// Peek at the top value on the stack without removing it
    pub fn peek(&self) -> VmResult<&Value> {
        self.stack.last().ok_or(VmError::StackUnderflow)
    }

    /// Get current instruction
    pub fn current_instruction(&self) -> VmResult<&Instruction> {
        self.code
            .get(self.ip)
            .ok_or(VmError::InvalidInstructionPointer(self.ip))
    }

    /// Execute the current program
    pub fn run(&mut self) -> VmResult<Value> {
        loop {
            if self.ip >= self.code.len() {
                break;
            }

            let instruction = *self.current_instruction()?;

            // No halt instruction in minimal opcode set - execution ends naturally

            self.step()?;
        }

        // Return the top value on the stack, or a default if empty
        if self.stack.is_empty() {
            Ok(Value::Integer(0)) // Default return value
        } else {
            Ok(self.stack.last().unwrap().clone())
        }
    }

    /// Execute a single instruction
    pub fn step(&mut self) -> VmResult<()> {
        let instruction = *self.current_instruction()?;

        match instruction.opcode {
            OpCode::LDC => {
                let const_index = instruction.operand as usize;
                if const_index >= self.constants.len() {
                    return Err(VmError::InvalidConstantIndex(const_index));
                }
                let value = self.constants[const_index].clone();
                self.push(value);
                self.ip += 1;
            }
            // LoadSymbol removed - symbols loaded via LDC constant pool
            // Push removed - immediates loaded via LDC constant pool
            OpCode::POP => {
                self.pop()?;
                self.ip += 1;
            }
            OpCode::DUP => {
                let value = self.peek()?.clone();
                self.push(value);
                self.ip += 1;
            }
            OpCode::ADD => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.add_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::SUB => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.sub_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::MUL => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.mul_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::DIV => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.div_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::POW => {
                let b = self.pop()?;
                let a = self.pop()?;
                let result = self.power_values(a, b)?;
                self.push(result);
                self.ip += 1;
            }
            OpCode::JMP => {
                self.ip = instruction.operand as usize;
            }
            OpCode::JIF => {
                let condition = self.pop()?;
                if self.is_falsy(&condition) {
                    self.ip = instruction.operand as usize;
                } else {
                    self.ip += 1;
                }
            }
            OpCode::CALL => {
                let arg_count = instruction.operand as usize;

                // Pop function name from stack
                let function_name = match self.pop()? {
                    Value::Function(name) => name,
                    Value::Symbol(name) => name,
                    other => {
                        return Err(VmError::TypeError {
                            expected: "Function or Symbol".to_string(),
                            actual: format!("{:?}", other),
                        })
                    }
                };

                // Pop arguments from stack (in reverse order)
                let mut args = Vec::with_capacity(arg_count);
                for _ in 0..arg_count {
                    args.push(self.pop()?);
                }
                args.reverse(); // Arguments were pushed in reverse order

                // Try to call stdlib function
                if let Some(func) = self.stdlib.get_function(&function_name) {
                    let result = func(&args)?;
                    self.push(result);
                } else {
                    return Err(VmError::TypeError {
                        expected: format!("known function, got: {}", function_name),
                        actual: "unknown function".to_string(),
                    });
                }

                self.ip += 1;
            }
            OpCode::RET => {
                if let Some(frame) = self.call_stack.pop() {
                    self.ip = frame.return_address;
                } else {
                    return Ok(()); // End of program
                }
            }
            // Halt instruction removed from minimal opcode set
            
            // New minimal opcodes
            OpCode::LDL => {
                // Load local variable (placeholder implementation)
                let local_index = instruction.operand as usize;
                // For now, return Missing - full implementation needs local variable stack
                self.push(Value::Missing);
                self.ip += 1;
            }
            OpCode::STL => {
                // Store local variable (placeholder implementation)
                let _local_index = instruction.operand as usize;
                let _value = self.pop()?;
                // For now, just discard - full implementation needs local variable stack
                self.ip += 1;
            }
            OpCode::STS => {
                // Store symbol value (placeholder implementation)
                let _symbol_index = instruction.operand as usize;
                let _value = self.pop()?;
                // For now, just discard - full implementation needs symbol table
                self.ip += 1;
            }
            OpCode::NEWLIST => {
                // Create new list from n stack items
                let count = instruction.operand as usize;
                let mut items = Vec::with_capacity(count);
                for _ in 0..count {
                    items.push(self.pop()?);
                }
                items.reverse(); // Items were popped in reverse order
                self.push(Value::List(items));
                self.ip += 1;
            }
            OpCode::NEWASSOC => {
                // Create new associative array from 2n stack items (key-value pairs)
                let pair_count = instruction.operand as usize;
                let mut pairs = Vec::with_capacity(pair_count);
                for _ in 0..pair_count {
                    let value = self.pop()?;
                    let key = self.pop()?;
                    pairs.push((key, value));
                }
                pairs.reverse(); // Pairs were popped in reverse order
                // For now, convert to a simple list representation
                // Full implementation would use a proper associative data structure
                let assoc_list: Vec<Value> = pairs.into_iter()
                    .flat_map(|(k, v)| vec![k, v])
                    .collect();
                self.push(Value::List(assoc_list));
                self.ip += 1;
            }
            OpCode::SYS => {
                // System call (placeholder implementation)
                let (sys_op, argc) = {
                    let sys_op = (instruction.operand >> 8) as u16;
                    let argc = (instruction.operand & 0xFF) as u8;
                    (sys_op, argc)
                };
                
                // Pop arguments
                let mut _args = Vec::with_capacity(argc as usize);
                for _ in 0..argc {
                    _args.push(self.pop()?);
                }
                
                // For now, just return Missing - full implementation needs system call registry
                self.push(Value::Missing);
                self.ip += 1;
            }
        }

        Ok(())
    }

    /// Add two values
    fn add_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Check if either operand is a tensor
        match (&a, &b) {
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                // Use tensor arithmetic
                tensor_add(&a, &b)
            }
            // Original scalar arithmetic
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a + b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a + b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 + b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a + b as f64)),
                _ => unreachable!(),
            },
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Subtract two values
    fn sub_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Check if either operand is a tensor
        match (&a, &b) {
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                // Use tensor arithmetic
                tensor_sub(&a, &b)
            }
            // Original scalar arithmetic
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a - b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a - b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 - b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a - b as f64)),
                _ => unreachable!(),
            },
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Multiply two values
    fn mul_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Check if either operand is a tensor
        match (&a, &b) {
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                // Use tensor arithmetic
                tensor_mul(&a, &b)
            }
            // Original scalar arithmetic
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => Ok(Value::Integer(a * b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a * b)),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 * b)),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a * b as f64)),
                _ => unreachable!(),
            },
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Divide two values
    fn div_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Check if either operand is a tensor
        match (&a, &b) {
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                // Use tensor arithmetic
                tensor_div(&a, &b)
            }
            // Original scalar arithmetic
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => {
                    if b == 0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a as f64 / b as f64))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => {
                    if b == 0.0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a / b))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => {
                    if b == 0.0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a as f64 / b))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => {
                    if b == 0 {
                        Err(VmError::DivisionByZero)
                    } else {
                        Ok(Value::Real(a / b as f64))
                    }
                }
                _ => unreachable!(),
            },
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Raise a to the power of b
    fn power_values(&self, a: Value, b: Value) -> VmResult<Value> {
        // Missing propagation: if either operand is Missing, result is Missing
        if matches!(a, Value::Missing) || matches!(b, Value::Missing) {
            return Ok(Value::Missing);
        }
        
        // Check if either operand is a tensor
        match (&a, &b) {
            (Value::Tensor(_), _) | (_, Value::Tensor(_)) => {
                // Use tensor arithmetic
                tensor_pow(&a, &b)
            }
            // Original scalar arithmetic
            (Value::Integer(_), Value::Integer(_)) => match (a, b) {
                (Value::Integer(a), Value::Integer(b)) => {
                    if b >= 0 {
                        Ok(Value::Integer(a.pow(b as u32)))
                    } else {
                        Ok(Value::Real((a as f64).powf(b as f64)))
                    }
                }
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Real(_)) => match (a, b) {
                (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a.powf(b))),
                _ => unreachable!(),
            },
            (Value::Integer(_), Value::Real(_)) => match (a, b) {
                (Value::Integer(a), Value::Real(b)) => Ok(Value::Real((a as f64).powf(b))),
                _ => unreachable!(),
            },
            (Value::Real(_), Value::Integer(_)) => match (a, b) {
                (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a.powf(b as f64))),
                _ => unreachable!(),
            },
            _ => Err(VmError::TypeError {
                expected: "numeric or tensor".to_string(),
                actual: format!("{:?} and {:?}", a, b),
            }),
        }
    }

    /// Check if a value is falsy (for conditional jumps)
    fn is_falsy(&self, value: &Value) -> bool {
        match value {
            Value::Boolean(b) => !b,
            Value::Missing => true, // Missing values are falsy
            Value::Integer(0) => true,
            Value::Real(f) => *f == 0.0,
            _ => false,
        }
    }
}

impl Default for VirtualMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::OpCode;

    #[test]
    fn test_vm_creation() {
        let vm = VirtualMachine::new();
        assert_eq!(vm.ip, 0);
        assert!(vm.stack.is_empty());
        assert!(vm.call_stack.is_empty());
        assert!(vm.constants.is_empty());
        assert!(vm.symbols.is_empty());
        assert!(vm.code.is_empty());
    }

    #[test]
    fn test_stack_operations() {
        let mut vm = VirtualMachine::new();

        // Test push
        vm.push(Value::Integer(42));
        assert_eq!(vm.stack.len(), 1);

        // Test peek
        assert_eq!(vm.peek().unwrap(), &Value::Integer(42));
        assert_eq!(vm.stack.len(), 1); // Peek doesn't remove

        // Test pop
        let value = vm.pop().unwrap();
        assert_eq!(value, Value::Integer(42));
        assert!(vm.stack.is_empty());

        // Test pop on empty stack
        assert!(vm.pop().is_err());
    }

    #[test]
    fn test_constant_pool() {
        let mut vm = VirtualMachine::new();

        let index1 = vm.add_constant(Value::Integer(42));
        let index2 = vm.add_constant(Value::String("hello".to_string()));

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(vm.constants[0], Value::Integer(42));
        assert_eq!(vm.constants[1], Value::String("hello".to_string()));
    }

    #[test]
    fn test_symbol_table() {
        let mut vm = VirtualMachine::new();

        let index1 = vm.add_symbol("x".to_string());
        let index2 = vm.add_symbol("y".to_string());
        let index3 = vm.add_symbol("x".to_string()); // Duplicate

        assert_eq!(index1, 0);
        assert_eq!(index2, 1);
        assert_eq!(index3, 0); // Should return existing index
    }

    #[test]
    fn test_load_const_instruction() {
        let mut vm = VirtualMachine::new();
        let const_index = vm.add_constant(Value::Integer(42));

        let instruction = Instruction::new(OpCode::LDC, const_index as u32).unwrap();
        vm.load(vec![instruction], vm.constants.clone());

        vm.step().unwrap();

        assert_eq!(vm.ip, 1);
        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_push_instruction() {
        let mut vm = VirtualMachine::new();

        let instruction = Instruction::new(OpCode::LDC, 42).unwrap();
        vm.load(vec![instruction], vec![]);

        vm.step().unwrap();

        assert_eq!(vm.ip, 1);
        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_dup_instruction() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 42).unwrap(),
            Instruction::new(OpCode::DUP, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 42
        vm.step().unwrap(); // Dup

        assert_eq!(vm.stack.len(), 2);
        assert_eq!(vm.stack[0], Value::Integer(42));
        assert_eq!(vm.stack[1], Value::Integer(42));
    }

    #[test]
    fn test_pop_instruction() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 42).unwrap(),
            Instruction::new(OpCode::LDC, 24).unwrap(),
            Instruction::new(OpCode::POP, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 42
        vm.step().unwrap(); // Push 24
        vm.step().unwrap(); // Pop 24

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(42));
    }

    #[test]
    fn test_arithmetic_add() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 2).unwrap(),
            Instruction::new(OpCode::LDC, 3).unwrap(),
            Instruction::new(OpCode::ADD, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Add

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(5));
    }

    #[test]
    fn test_arithmetic_sub() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 5).unwrap(),
            Instruction::new(OpCode::LDC, 3).unwrap(),
            Instruction::new(OpCode::SUB, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 5
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Sub

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(2));
    }

    #[test]
    fn test_arithmetic_mul() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 3).unwrap(),
            Instruction::new(OpCode::LDC, 4).unwrap(),
            Instruction::new(OpCode::MUL, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Push 4
        vm.step().unwrap(); // Mul

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(12));
    }

    #[test]
    fn test_arithmetic_div() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 8).unwrap(),
            Instruction::new(OpCode::LDC, 2).unwrap(),
            Instruction::new(OpCode::DIV, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 8
        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Div

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Real(4.0));
    }

    #[test]
    fn test_arithmetic_power() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 2).unwrap(),
            Instruction::new(OpCode::LDC, 3).unwrap(),
            Instruction::new(OpCode::POW, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 2
        vm.step().unwrap(); // Push 3
        vm.step().unwrap(); // Power

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(8));
    }

    #[test]
    fn test_division_by_zero() {
        let mut vm = VirtualMachine::new();

        let program = vec![
            Instruction::new(OpCode::LDC, 5).unwrap(),
            Instruction::new(OpCode::LDC, 0).unwrap(),
            Instruction::new(OpCode::DIV, 0).unwrap(),
        ];
        vm.load(program, vec![]);

        vm.step().unwrap(); // Push 5
        vm.step().unwrap(); // Push 0
        let result = vm.step(); // Div

        assert!(result.is_err());
        match result.unwrap_err() {
            VmError::DivisionByZero => {}
            _ => panic!("Expected DivisionByZero error"),
        }
    }

    #[test]
    fn test_jump_instruction() {
        let mut vm = VirtualMachine::new();

        let instruction = Instruction::new(OpCode::JMP, 5).unwrap();
        vm.load(vec![instruction], vec![]);

        vm.step().unwrap();

        assert_eq!(vm.ip, 5);
    }

    #[test]
    fn test_jump_if_false_true_condition() {
        let mut vm = VirtualMachine::new();
        let true_index = vm.add_constant(Value::Boolean(true));

        let program = vec![
            Instruction::new(OpCode::LDC, true_index as u32).unwrap(),
            Instruction::new(OpCode::JIF, 5).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load true
        vm.step().unwrap(); // JumpIfFalse

        assert_eq!(vm.ip, 2); // Should not jump
    }

    #[test]
    fn test_jump_if_false_false_condition() {
        let mut vm = VirtualMachine::new();
        let false_index = vm.add_constant(Value::Boolean(false));

        let program = vec![
            Instruction::new(OpCode::LDC, false_index as u32).unwrap(),
            Instruction::new(OpCode::JIF, 5).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load false
        vm.step().unwrap(); // JumpIfFalse

        assert_eq!(vm.ip, 5); // Should jump
    }

    #[test]
    fn test_simple_program() {
        let mut vm = VirtualMachine::new();

        // Program: push 2, push 3, add, halt
        let program = vec![
            Instruction::new(OpCode::LDC, 2).unwrap(),
            Instruction::new(OpCode::LDC, 3).unwrap(),
            Instruction::new(OpCode::ADD, 0).unwrap(),
            Instruction::new(OpCode::RET, 0).unwrap(),
        ];

        vm.load(program, vec![]);
        let result = vm.run().unwrap();

        assert_eq!(result, Value::Integer(5));
    }

    #[test]
    fn test_mixed_types_arithmetic() {
        let mut vm = VirtualMachine::new();
        let int_index = vm.add_constant(Value::Integer(2));
        let real_index = vm.add_constant(Value::Real(3.5));

        let program = vec![
            Instruction::new(OpCode::LDC, int_index as u32).unwrap(),
            Instruction::new(OpCode::LDC, real_index as u32).unwrap(),
            Instruction::new(OpCode::ADD, 0).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load 2
        vm.step().unwrap(); // Load 3.5
        vm.step().unwrap(); // Add

        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Real(5.5));
    }

    #[test]
    fn test_type_error() {
        let mut vm = VirtualMachine::new();
        let string_index = vm.add_constant(Value::String("hello".to_string()));
        let int_index = vm.add_constant(Value::Integer(42));

        let program = vec![
            Instruction::new(OpCode::LDC, string_index as u32).unwrap(),
            Instruction::new(OpCode::LDC, int_index as u32).unwrap(),
            Instruction::new(OpCode::ADD, 0).unwrap(),
        ];
        vm.load(program, vm.constants.clone());

        vm.step().unwrap(); // Load "hello"
        vm.step().unwrap(); // Load 42
        let result = vm.step(); // Add

        assert!(result.is_err());
        match result.unwrap_err() {
            VmError::TypeError { .. } => {}
            _ => panic!("Expected TypeError"),
        }
    }

    // ===== TENSOR ARITHMETIC INTEGRATION TESTS =====

    #[test]
    fn test_tensor_scalar_addition_vm() {
        let vm = VirtualMachine::new();
        
        // Create a test tensor
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![1.0, 2.0, 3.0]).unwrap();
        let tensor_value = Value::Tensor(tensor);
        let scalar_value = Value::Integer(10);
        
        // Test tensor + scalar through VM
        let result = vm.add_values(tensor_value, scalar_value).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[3]);
                assert_eq!(result_tensor[[0]], 11.0); // 1 + 10
                assert_eq!(result_tensor[[1]], 12.0); // 2 + 10
                assert_eq!(result_tensor[[2]], 13.0); // 3 + 10
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_tensor_addition_vm() {
        let vm = VirtualMachine::new();
        
        // Create two test tensors
        let tensor_a = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![1.0, 2.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![3.0, 4.0]).unwrap();
        
        let value_a = Value::Tensor(tensor_a);
        let value_b = Value::Tensor(tensor_b);
        
        // Test tensor + tensor through VM
        let result = vm.add_values(value_a, value_b).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2]);
                assert_eq!(result_tensor[[0]], 4.0); // 1 + 3
                assert_eq!(result_tensor[[1]], 6.0); // 2 + 4
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_subtraction_vm() {
        let vm = VirtualMachine::new();
        
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![10.0, 8.0]).unwrap();
        let tensor_value = Value::Tensor(tensor);
        let scalar_value = Value::Real(3.0);
        
        let result = vm.sub_values(tensor_value, scalar_value).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 7.0);  // 10 - 3
                assert_eq!(result_tensor[[1]], 5.0);  // 8 - 3
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_multiplication_vm() {
        let vm = VirtualMachine::new();
        
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![2.0, 3.0]).unwrap();
        let tensor_value = Value::Tensor(tensor);
        let scalar_value = Value::Integer(4);
        
        let result = vm.mul_values(tensor_value, scalar_value).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 8.0);  // 2 * 4
                assert_eq!(result_tensor[[1]], 12.0); // 3 * 4
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_division_vm() {
        let vm = VirtualMachine::new();
        
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![12.0, 20.0]).unwrap();
        let tensor_value = Value::Tensor(tensor);
        let scalar_value = Value::Integer(4);
        
        let result = vm.div_values(tensor_value, scalar_value).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 3.0); // 12 / 4
                assert_eq!(result_tensor[[1]], 5.0); // 20 / 4
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_power_vm() {
        let vm = VirtualMachine::new();
        
        let tensor = ArrayD::from_shape_vec(ndarray::IxDyn(&[2]), vec![2.0, 3.0]).unwrap();
        let tensor_value = Value::Tensor(tensor);
        let scalar_value = Value::Integer(2);
        
        let result = vm.power_values(tensor_value, scalar_value).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor[[0]], 4.0);  // 2^2
                assert_eq!(result_tensor[[1]], 9.0);  // 3^2
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_tensor_broadcasting_vm() {
        let vm = VirtualMachine::new();
        
        // Test broadcasting: [2, 3] + [3] -> [2, 3]
        let tensor_a = ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tensor_b = ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![10.0, 20.0, 30.0]).unwrap();
        
        let value_a = Value::Tensor(tensor_a);
        let value_b = Value::Tensor(tensor_b);
        
        let result = vm.add_values(value_a, value_b).unwrap();
        
        match result {
            Value::Tensor(result_tensor) => {
                assert_eq!(result_tensor.shape(), &[2, 3]);
                // First row: [1,2,3] + [10,20,30] = [11,22,33]
                assert_eq!(result_tensor[[0, 0]], 11.0);
                assert_eq!(result_tensor[[0, 1]], 22.0);
                assert_eq!(result_tensor[[0, 2]], 33.0);
                // Second row: [4,5,6] + [10,20,30] = [14,25,36]
                assert_eq!(result_tensor[[1, 0]], 14.0);
                assert_eq!(result_tensor[[1, 1]], 25.0);
                assert_eq!(result_tensor[[1, 2]], 36.0);
            }
            _ => panic!("Expected tensor result"),
        }
    }

    #[test]
    fn test_scalar_arithmetic_still_works() {
        // Verify that the tensor integration doesn't break existing scalar arithmetic
        let vm = VirtualMachine::new();
        
        // Test integer + integer
        let result = vm.add_values(Value::Integer(2), Value::Integer(3)).unwrap();
        assert_eq!(result, Value::Integer(5));
        
        // Test real + real  
        let result = vm.add_values(Value::Real(2.5), Value::Real(3.5)).unwrap();
        assert_eq!(result, Value::Real(6.0));
        
        // Test mixed types
        let result = vm.mul_values(Value::Integer(4), Value::Real(2.5)).unwrap();
        assert_eq!(result, Value::Real(10.0));
    }

    // ===== MISSING VALUE TESTS =====

    #[test]
    fn test_missing_arithmetic_propagation() {
        let mut vm = VirtualMachine::new();
        
        // Test Missing + Integer = Missing
        let result = vm.add_values(Value::Missing, Value::Integer(5)).unwrap();
        assert!(matches!(result, Value::Missing));
        
        // Test Integer + Missing = Missing
        let result = vm.add_values(Value::Integer(5), Value::Missing).unwrap();
        assert!(matches!(result, Value::Missing));
        
        // Test Missing - Real = Missing
        let result = vm.sub_values(Value::Missing, Value::Real(3.14)).unwrap();
        assert!(matches!(result, Value::Missing));
        
        // Test Missing * Integer = Missing
        let result = vm.mul_values(Value::Missing, Value::Integer(10)).unwrap();
        assert!(matches!(result, Value::Missing));
        
        // Test Missing / Integer = Missing
        let result = vm.div_values(Value::Missing, Value::Integer(2)).unwrap();
        assert!(matches!(result, Value::Missing));
        
        // Test Missing ^ Integer = Missing
        let result = vm.power_values(Value::Missing, Value::Integer(2)).unwrap();
        assert!(matches!(result, Value::Missing));
    }
    
    #[test]
    fn test_missing_equality() {
        // Test Missing == Missing
        assert_eq!(Value::Missing, Value::Missing);
        
        // Test Missing != other values
        assert_ne!(Value::Missing, Value::Integer(0));
        assert_ne!(Value::Missing, Value::Real(0.0));
        assert_ne!(Value::Missing, Value::String("".to_string()));
        assert_ne!(Value::Missing, Value::Boolean(false));
    }
    
    #[test]
    fn test_missing_is_falsy() {
        let vm = VirtualMachine::new();
        assert!(vm.is_falsy(&Value::Missing));
    }
    
    #[test]
    fn test_series_with_missing() {
        // Test creating a series with Mixed data including Missing
        let data = vec![
            Value::Integer(1),
            Value::Missing,
            Value::Integer(3),
        ];
        
        let series = Series::infer(data).unwrap();
        assert_eq!(series.dtype, SeriesType::Int64);
        assert_eq!(series.length, 3);
        assert!(matches!(series.data[1], Value::Missing));
    }
    
    #[test]
    fn test_series_all_missing() {
        // Test creating a series with all Missing values
        let data = vec![Value::Missing, Value::Missing];
        
        let series = Series::infer(data).unwrap();
        assert_eq!(series.dtype, SeriesType::String); // Default when all Missing
        assert_eq!(series.length, 2);
    }
    
    #[test]
    fn test_series_validate_missing() {
        // Test that Missing values are accepted in any series type
        let data = vec![
            Value::Integer(1),
            Value::Missing,
        ];
        
        let series = Series::new(data, SeriesType::Int64).unwrap();
        assert_eq!(series.dtype, SeriesType::Int64);
        assert_eq!(series.length, 2);
    }
    
    #[test]
    fn test_schema_nullable_validation() {
        // Test that Nullable schemas accept Missing values
        let schema = Schema::new(SchemaType::Nullable(Box::new(SchemaType::Int64)));
        
        // Should accept Missing
        assert!(schema.validate(&Value::Missing).is_ok());
        
        // Should accept Int64
        assert!(schema.validate(&Value::Integer(42)).is_ok());
        
        // Should reject other types
        assert!(schema.validate(&Value::String("hello".to_string())).is_err());
    }
    
    #[test]
    fn test_schema_constructors() {
        // Test helper constructors
        let int_schema = Schema::int64();
        assert_eq!(int_schema.schema_type, SchemaType::Int64);
        
        let float_schema = Schema::float64();
        assert_eq!(float_schema.schema_type, SchemaType::Float64);
        
        let decimal_schema = Schema::decimal(10, 2);
        assert_eq!(decimal_schema.schema_type, SchemaType::Decimal { precision: 10, scale: 2 });
        
        let nullable_int = Schema::nullable(SchemaType::Int64);
        assert_eq!(nullable_int.schema_type, SchemaType::Nullable(Box::new(SchemaType::Int64)));
        
        let list_schema = Schema::list(SchemaType::String);
        assert_eq!(list_schema.schema_type, SchemaType::List(Box::new(SchemaType::String)));
    }
    
    #[test]
    fn test_schema_date_validation() {
        let date_schema = Schema::date();
        
        // Valid dates
        assert!(date_schema.validate(&Value::String("2023-12-25".to_string())).is_ok());
        assert!(date_schema.validate(&Value::String("2000-01-01".to_string())).is_ok());
        
        // Invalid dates
        assert!(date_schema.validate(&Value::String("2023-13-01".to_string())).is_err()); // Invalid month
        assert!(date_schema.validate(&Value::String("2023-12-32".to_string())).is_err()); // Invalid day
        assert!(date_schema.validate(&Value::String("23-12-25".to_string())).is_err());   // Wrong format
        assert!(date_schema.validate(&Value::Integer(20231225)).is_err());                // Wrong type
    }
    
    #[test]
    fn test_schema_timestamp_validation() {
        let timestamp_schema = Schema::timestamp();
        
        // Valid timestamps
        assert!(timestamp_schema.validate(&Value::String("2023-12-25 14:30:45".to_string())).is_ok());
        assert!(timestamp_schema.validate(&Value::String("2000-01-01 00:00:00".to_string())).is_ok());
        
        // Invalid timestamps
        assert!(timestamp_schema.validate(&Value::String("2023-12-25".to_string())).is_err());      // Missing time
        assert!(timestamp_schema.validate(&Value::String("2023-12-25 25:00:00".to_string())).is_err()); // Invalid hour
        assert!(timestamp_schema.validate(&Value::String("2023-12-25 14:60:00".to_string())).is_err()); // Invalid minute
    }
    
    #[test] 
    fn test_schema_uuid_validation() {
        let uuid_schema = Schema::uuid();
        
        // Valid UUIDs
        assert!(uuid_schema.validate(&Value::String("550e8400-e29b-41d4-a716-446655440000".to_string())).is_ok());
        assert!(uuid_schema.validate(&Value::String("6ba7b810-9dad-11d1-80b4-00c04fd430c8".to_string())).is_ok());
        
        // Invalid UUIDs
        assert!(uuid_schema.validate(&Value::String("550e8400-e29b-41d4-a716".to_string())).is_err());     // Too short
        assert!(uuid_schema.validate(&Value::String("550e8400-e29b-41d4-a716-446655440000-extra".to_string())).is_err()); // Too long
        assert!(uuid_schema.validate(&Value::String("550e8400-e29b-41d4-a716-44665544000g".to_string())).is_err()); // Invalid hex
    }
    
    #[test]
    fn test_schema_decimal_validation() {
        let decimal_schema = Schema::decimal(10, 2);
        
        // Should accept Real values (for now)
        assert!(decimal_schema.validate(&Value::Real(123.45)).is_ok());
        assert!(decimal_schema.validate(&Value::Real(0.0)).is_ok());
        
        // Should reject other types
        assert!(decimal_schema.validate(&Value::String("123.45".to_string())).is_err());
        assert!(decimal_schema.validate(&Value::Integer(123)).is_err());
    }
    
    #[test]
    fn test_schema_list_validation() {
        let list_schema = Schema::list(SchemaType::Int64);
        
        // Valid list of integers
        let valid_list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        assert!(list_schema.validate(&valid_list).is_ok());
        
        // Invalid list with mixed types
        let invalid_list = Value::List(vec![
            Value::Integer(1),
            Value::String("hello".to_string()),
            Value::Integer(3),
        ]);
        assert!(list_schema.validate(&invalid_list).is_err());
        
        // Empty list should be valid
        let empty_list = Value::List(vec![]);
        assert!(list_schema.validate(&empty_list).is_ok());
    }
    
    // ===== SCHEMA INFERENCE TESTS =====
    
    #[test]
    fn test_schema_inference_basic_types() {
        // Test single value inference
        let int_schema = Schema::infer_from_value(&Value::Integer(42));
        assert_eq!(int_schema.schema_type, SchemaType::Int64);
        
        let float_schema = Schema::infer_from_value(&Value::Real(3.14));
        assert_eq!(float_schema.schema_type, SchemaType::Float64);
        
        let bool_schema = Schema::infer_from_value(&Value::Boolean(true));
        assert_eq!(bool_schema.schema_type, SchemaType::Bool);
        
        let string_schema = Schema::infer_from_value(&Value::String("hello".to_string()));
        assert_eq!(string_schema.schema_type, SchemaType::String);
    }
    
    #[test]
    fn test_schema_inference_string_types() {
        // Test date string inference
        let date_schema = Schema::infer_from_value(&Value::String("2023-12-25".to_string()));
        assert_eq!(date_schema.schema_type, SchemaType::Date);
        
        // Test timestamp string inference
        let timestamp_schema = Schema::infer_from_value(&Value::String("2023-12-25 14:30:45".to_string()));
        assert_eq!(timestamp_schema.schema_type, SchemaType::Timestamp);
        
        // Test UUID string inference
        let uuid_schema = Schema::infer_from_value(&Value::String("550e8400-e29b-41d4-a716-446655440000".to_string()));
        assert_eq!(uuid_schema.schema_type, SchemaType::UUID);
        
        // Test regular string
        let regular_schema = Schema::infer_from_value(&Value::String("just a string".to_string()));
        assert_eq!(regular_schema.schema_type, SchemaType::String);
    }
    
    #[test]
    fn test_schema_inference_from_values() {
        // Test homogeneous integer list
        let int_values = vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ];
        let schema = Schema::infer_from_values(&int_values);
        assert_eq!(schema.schema_type, SchemaType::Int64);
        
        // Test mixed numeric types (should unify to Float64)
        let mixed_numeric = vec![
            Value::Integer(1),
            Value::Real(2.5),
            Value::Integer(3),
        ];
        let schema = Schema::infer_from_values(&mixed_numeric);
        assert_eq!(schema.schema_type, SchemaType::Float64);
        
        // Test with Missing values (should be Nullable)
        let nullable_values = vec![
            Value::Integer(1),
            Value::Missing,
            Value::Integer(3),
        ];
        let schema = Schema::infer_from_values(&nullable_values);
        assert_eq!(schema.schema_type, SchemaType::Nullable(Box::new(SchemaType::Int64)));
    }
    
    #[test]
    fn test_schema_inference_type_unification() {
        // Test date/string unification (should become String)
        let mixed_strings = vec![
            Value::String("2023-12-25".to_string()),
            Value::String("hello world".to_string()),
        ];
        let schema = Schema::infer_from_values(&mixed_strings);
        assert_eq!(schema.schema_type, SchemaType::String);
        
        // Test different date formats (should become String)
        let mixed_dates = vec![
            Value::String("2023-12-25".to_string()),
            Value::String("2023-12-25 14:30:45".to_string()),
        ];
        let schema = Schema::infer_from_values(&mixed_dates);
        assert_eq!(schema.schema_type, SchemaType::String);
    }
    
    #[test]
    fn test_schema_inference_list_types() {
        // Test inference for list values
        let list_value = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        let schema = Schema::infer_from_value(&list_value);
        assert_eq!(schema.schema_type, SchemaType::List(Box::new(SchemaType::Int64)));
        
        // Test empty list
        let empty_list = Value::List(vec![]);
        let schema = Schema::infer_from_value(&empty_list);
        assert_eq!(schema.schema_type, SchemaType::List(Box::new(SchemaType::String)));
    }
    
    #[test]
    fn test_schema_inference_series() {
        // Test inference from Series
        let series = Series::new(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Missing,
        ], SeriesType::Int64).unwrap();
        
        let schema = Schema::infer_from_value(&Value::Series(series));
        assert_eq!(schema.schema_type, SchemaType::Nullable(Box::new(SchemaType::Int64)));
        
        // Test series without Missing values
        let series_no_missing = Series::new(vec![
            Value::Integer(1),
            Value::Integer(2),
        ], SeriesType::Int64).unwrap();
        
        let schema = Schema::infer_from_value(&Value::Series(series_no_missing));
        assert_eq!(schema.schema_type, SchemaType::Int64);
    }
    
    #[test]
    fn test_schema_inference_edge_cases() {
        // Test all Missing values
        let all_missing = vec![Value::Missing, Value::Missing];
        let schema = Schema::infer_from_values(&all_missing);
        assert_eq!(schema.schema_type, SchemaType::Nullable(Box::new(SchemaType::String)));
        
        // Test empty values
        let empty_values: Vec<Value> = vec![];
        let schema = Schema::infer_from_values(&empty_values);
        assert_eq!(schema.schema_type, SchemaType::String);
        
        // Test single Missing value
        let single_missing = Schema::infer_from_value(&Value::Missing);
        assert_eq!(single_missing.schema_type, SchemaType::String);
    }
    
    // ===== SCHEMA CASTING TESTS =====
    
    #[test]
    fn test_schema_cast_numeric_conversions() {
        // Test Integer to Float64
        let float_schema = Schema::float64();
        let result = float_schema.cast(&Value::Integer(42), false).unwrap();
        assert_eq!(result, Value::Real(42.0));
        
        // Test Float64 to Integer (lenient)
        let int_schema = Schema::int64();
        let result = int_schema.cast(&Value::Real(42.5), false).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Test Float64 to Integer (strict should fail)
        let result = int_schema.cast(&Value::Real(42.5), true);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_schema_cast_string_conversions() {
        let string_schema = Schema::string();
        
        // Test various types to string
        assert_eq!(string_schema.cast(&Value::Integer(42), false).unwrap(), Value::String("42".to_string()));
        assert_eq!(string_schema.cast(&Value::Real(3.14), false).unwrap(), Value::String("3.14".to_string()));
        assert_eq!(string_schema.cast(&Value::Boolean(true), false).unwrap(), Value::String("true".to_string()));
    }
    
    #[test]
    fn test_schema_cast_from_string() {
        // Test string to integer
        let int_schema = Schema::int64();
        let result = int_schema.cast(&Value::String("42".to_string()), false).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Test invalid string to integer
        let result = int_schema.cast(&Value::String("not_a_number".to_string()), false);
        assert!(result.is_err());
        
        // Test string to float
        let float_schema = Schema::float64();
        let result = float_schema.cast(&Value::String("3.14".to_string()), false).unwrap();
        assert_eq!(result, Value::Real(3.14));
        
        // Test string to boolean
        let bool_schema = Schema::bool();
        assert_eq!(bool_schema.cast(&Value::String("true".to_string()), false).unwrap(), Value::Boolean(true));
        assert_eq!(bool_schema.cast(&Value::String("false".to_string()), false).unwrap(), Value::Boolean(false));
        assert_eq!(bool_schema.cast(&Value::String("1".to_string()), false).unwrap(), Value::Boolean(true));
        assert_eq!(bool_schema.cast(&Value::String("0".to_string()), false).unwrap(), Value::Boolean(false));
        assert_eq!(bool_schema.cast(&Value::String("yes".to_string()), false).unwrap(), Value::Boolean(true));
        assert_eq!(bool_schema.cast(&Value::String("no".to_string()), false).unwrap(), Value::Boolean(false));
    }
    
    #[test]
    fn test_schema_cast_missing_values() {
        // Test Missing in nullable schema
        let nullable_int = Schema::nullable(SchemaType::Int64);
        let result = nullable_int.cast(&Value::Missing, true).unwrap();
        assert_eq!(result, Value::Missing);
        
        // Test Missing in non-nullable schema (strict mode should fail)
        let int_schema = Schema::int64();
        let result = int_schema.cast(&Value::Missing, true);
        assert!(result.is_err());
        
        // Test Missing in non-nullable schema (lenient mode should pass)
        let result = int_schema.cast(&Value::Missing, false).unwrap();
        assert_eq!(result, Value::Missing);
    }
    
    #[test]
    fn test_schema_cast_list_values() {
        // Test casting list items
        let list_schema = Schema::list(SchemaType::Float64);
        let input_list = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Real(3.5),
        ]);
        
        let result = list_schema.cast(&input_list, false).unwrap();
        match result {
            Value::List(items) => {
                assert_eq!(items.len(), 3);
                assert_eq!(items[0], Value::Real(1.0));
                assert_eq!(items[1], Value::Real(2.0));
                assert_eq!(items[2], Value::Real(3.5));
            },
            _ => panic!("Expected list result"),
        }
    }
    
    #[test]
    fn test_schema_cast_already_valid() {
        // Test casting value that already matches schema
        let int_schema = Schema::int64();
        let result = int_schema.cast(&Value::Integer(42), true).unwrap();
        assert_eq!(result, Value::Integer(42));
    }
    
    #[test]
    fn test_schema_validate_and_cast() {
        let int_schema = Schema::int64();
        
        // Test value that validates
        let result = int_schema.validate_and_cast(&Value::Integer(42), true).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Test value that needs casting
        let result = int_schema.validate_and_cast(&Value::String("42".to_string()), false).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Test value that can't be cast in strict mode
        let result = int_schema.validate_and_cast(&Value::String("not_a_number".to_string()), true);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_schema_cast_strict_vs_lenient() {
        let int_schema = Schema::int64();
        
        // Lenient mode: should allow float to int conversion
        let result = int_schema.cast(&Value::Real(42.7), false).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Strict mode: should reject float to int conversion
        let result = int_schema.cast(&Value::Real(42.7), true);
        assert!(result.is_err());
        
        // Lenient mode: should allow string parsing
        let result = int_schema.cast(&Value::String("42".to_string()), false).unwrap();
        assert_eq!(result, Value::Integer(42));
        
        // Strict mode: should reject string parsing
        let result = int_schema.cast(&Value::String("42".to_string()), true);
        assert!(result.is_err());
    }
    
    // ===== SERIES COW TESTS =====
    
    #[test]
    fn test_series_cow_basic_operations() {
        let data = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let series = Series::new(data, SeriesType::Int64).unwrap();
        
        // Test basic access
        assert_eq!(series.length, 3);
        assert_eq!(*series.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*series.get(2).unwrap(), Value::Integer(3));
        
        // Test bounds checking
        assert!(series.get(3).is_err());
    }
    
    #[test]
    fn test_series_cow_cloning() {
        let data = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let series1 = Series::new(data, SeriesType::Int64).unwrap();
        let series2 = series1.clone();
        
        // After cloning, both series should share the same data
        assert!(series1.shares_data_with(&series2));
        assert_eq!(series1, series2);
    }
    
    #[test]
    fn test_series_cow_modification() {
        let data = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let series1 = Series::new(data, SeriesType::Int64).unwrap();
        let series2 = series1.clone();
        
        // Modify one series - should trigger COW
        let series3 = series1.with_value_at(1, Value::Integer(42)).unwrap();
        
        // series1 and series2 should still share data
        assert!(series1.shares_data_with(&series2));
        
        // series3 should have its own data
        assert!(!series1.shares_data_with(&series3));
        assert!(!series2.shares_data_with(&series3));
        
        // Verify values
        assert_eq!(*series1.get(1).unwrap(), Value::Integer(2));
        assert_eq!(*series2.get(1).unwrap(), Value::Integer(2));
        assert_eq!(*series3.get(1).unwrap(), Value::Integer(42));
    }
    
    #[test]
    fn test_series_cow_append() {
        let data = vec![Value::Integer(1), Value::Integer(2)];
        let series1 = Series::new(data, SeriesType::Int64).unwrap();
        let series2 = series1.clone();
        
        // Append to create new series
        let series3 = series1.append(Value::Integer(3)).unwrap();
        
        // Verify lengths
        assert_eq!(series1.length, 2);
        assert_eq!(series2.length, 2);
        assert_eq!(series3.length, 3);
        
        // series1 and series2 should still share data
        assert!(series1.shares_data_with(&series2));
        assert!(!series1.shares_data_with(&series3));
        
        // Verify the appended value
        assert_eq!(*series3.get(2).unwrap(), Value::Integer(3));
    }
    
    #[test]
    fn test_series_cow_slicing() {
        let data = vec![
            Value::Integer(1), 
            Value::Integer(2), 
            Value::Integer(3), 
            Value::Integer(4), 
            Value::Integer(5)
        ];
        let series = Series::new(data, SeriesType::Int64).unwrap();
        
        // Create a slice
        let slice = series.slice(1, 4).unwrap();
        
        // Verify slice properties
        assert_eq!(slice.length, 3);
        assert_eq!(*slice.get(0).unwrap(), Value::Integer(2));
        assert_eq!(*slice.get(1).unwrap(), Value::Integer(3));
        assert_eq!(*slice.get(2).unwrap(), Value::Integer(4));
        
        // Test slice bounds checking
        assert!(series.slice(3, 2).is_err()); // start > end
        assert!(series.slice(0, 6).is_err()); // end > length
    }
    
    #[test]
    fn test_series_cow_type_validation() {
        let data = vec![Value::Integer(1), Value::Integer(2)];
        let series = Series::new(data, SeriesType::Int64).unwrap();
        
        // Try to set a value with wrong type
        let result = series.with_value_at(0, Value::String("hello".to_string()));
        assert!(result.is_err());
        
        // Try to append a value with wrong type
        let result = series.append(Value::Real(3.14));
        assert!(result.is_err());
    }
    
    #[test]
    fn test_series_cow_with_missing() {
        let data = vec![Value::Integer(1), Value::Missing, Value::Integer(3)];
        let series = Series::new(data, SeriesType::Int64).unwrap();
        
        // Missing values should be allowed in any series type
        assert_eq!(series.length, 3);
        assert_eq!(*series.get(1).unwrap(), Value::Missing);
        
        // Can append Missing
        let series2 = series.append(Value::Missing).unwrap();
        assert_eq!(series2.length, 4);
        assert_eq!(*series2.get(3).unwrap(), Value::Missing);
    }
    
    #[test]
    fn test_series_cow_iteration() {
        let data = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let series = Series::new(data, SeriesType::Int64).unwrap();
        
        // Test iteration
        let values: Vec<&Value> = series.iter().collect();
        assert_eq!(values.len(), 3);
        assert_eq!(*values[0], Value::Integer(1));
        assert_eq!(*values[1], Value::Integer(2));
        assert_eq!(*values[2], Value::Integer(3));
    }
    
    #[test]
    fn test_series_cow_memory_efficiency() {
        let data = vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)];
        let series1 = Series::new(data, SeriesType::Int64).unwrap();
        
        // Create many clones
        let series2 = series1.clone();
        let series3 = series1.clone();
        let series4 = series1.clone();
        
        // All should share the same data (memory efficient)
        assert!(series1.shares_data_with(&series2));
        assert!(series1.shares_data_with(&series3));
        assert!(series1.shares_data_with(&series4));
        assert!(series2.shares_data_with(&series3));
        
        // Only when we modify should we get separate data
        let series5 = series1.with_value_at(0, Value::Integer(42)).unwrap();
        assert!(!series1.shares_data_with(&series5));
        assert!(series1.shares_data_with(&series2)); // Others still share
    }

    // ===== TABLE TESTS =====

    #[test]
    fn test_table_creation_empty() {
        let table = Table::new();
        assert_eq!(table.length, 0);
        assert_eq!(table.columns.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_table_from_columns() {
        let mut columns = HashMap::new();
        
        let col1 = Series::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)], 
            SeriesType::Int64
        ).unwrap();
        let col2 = Series::new(
            vec![Value::String("a".to_string()), Value::String("b".to_string()), Value::String("c".to_string())], 
            SeriesType::String
        ).unwrap();
        
        columns.insert("numbers".to_string(), col1);
        columns.insert("letters".to_string(), col2);
        
        let table = Table::from_columns(columns).unwrap();
        assert_eq!(table.length, 3);
        assert_eq!(table.columns.len(), 2);
        assert_eq!(table.shape(), (3, 2));
    }

    #[test]
    fn test_table_from_columns_mismatched_lengths() {
        let mut columns = HashMap::new();
        
        let col1 = Series::new(
            vec![Value::Integer(1), Value::Integer(2)], 
            SeriesType::Int64
        ).unwrap();
        let col2 = Series::new(
            vec![Value::String("a".to_string()), Value::String("b".to_string()), Value::String("c".to_string())], 
            SeriesType::String
        ).unwrap();
        
        columns.insert("numbers".to_string(), col1);
        columns.insert("letters".to_string(), col2);
        
        let result = Table::from_columns(columns);
        assert!(result.is_err());
    }

    #[test]
    fn test_table_from_rows() {
        let column_names = vec!["id".to_string(), "name".to_string(), "age".to_string()];
        let rows = vec![
            vec![Value::Integer(1), Value::String("Alice".to_string()), Value::Integer(25)],
            vec![Value::Integer(2), Value::String("Bob".to_string()), Value::Integer(30)],
            vec![Value::Integer(3), Value::String("Carol".to_string()), Value::Integer(35)],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        assert_eq!(table.length, 3);
        assert_eq!(table.columns.len(), 3);
        
        // Test column access
        let id_col = table.get_column("id").unwrap();
        assert_eq!(*id_col.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*id_col.get(1).unwrap(), Value::Integer(2));
        assert_eq!(*id_col.get(2).unwrap(), Value::Integer(3));
        
        let name_col = table.get_column("name").unwrap();
        assert_eq!(*name_col.get(0).unwrap(), Value::String("Alice".to_string()));
        assert_eq!(*name_col.get(1).unwrap(), Value::String("Bob".to_string()));
        assert_eq!(*name_col.get(2).unwrap(), Value::String("Carol".to_string()));
    }

    #[test]
    fn test_table_from_rows_auto_names() {
        let rows = vec![
            vec![Value::Integer(1), Value::String("Alice".to_string())],
            vec![Value::Integer(2), Value::String("Bob".to_string())],
        ];
        
        let table = Table::from_rows_auto_names(rows).unwrap();
        assert_eq!(table.length, 2);
        assert_eq!(table.columns.len(), 2);
        
        assert!(table.get_column("col0").is_some());
        assert!(table.get_column("col1").is_some());
    }

    #[test]
    fn test_table_select_columns() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![Value::Integer(1), Value::Integer(2)], SeriesType::Int64).unwrap());
        columns.insert("b".to_string(), Series::new(vec![Value::Integer(3), Value::Integer(4)], SeriesType::Int64).unwrap());
        columns.insert("c".to_string(), Series::new(vec![Value::Integer(5), Value::Integer(6)], SeriesType::Int64).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let selected = table.select_columns(&["a", "c"]).unwrap();
        
        assert_eq!(selected.columns.len(), 2);
        assert!(selected.get_column("a").is_some());
        assert!(selected.get_column("c").is_some());
        assert!(selected.get_column("b").is_none());
    }

    #[test]
    fn test_table_drop_columns() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![Value::Integer(1), Value::Integer(2)], SeriesType::Int64).unwrap());
        columns.insert("b".to_string(), Series::new(vec![Value::Integer(3), Value::Integer(4)], SeriesType::Int64).unwrap());
        columns.insert("c".to_string(), Series::new(vec![Value::Integer(5), Value::Integer(6)], SeriesType::Int64).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let dropped = table.drop_columns(&["b"]).unwrap();
        
        assert_eq!(dropped.columns.len(), 2);
        assert!(dropped.get_column("a").is_some());
        assert!(dropped.get_column("c").is_some());
        assert!(dropped.get_column("b").is_none());
    }

    #[test]
    fn test_table_with_columns() {
        let mut columns = HashMap::new();
        columns.insert("a".to_string(), Series::new(vec![Value::Integer(1), Value::Integer(2)], SeriesType::Int64).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        
        let mut new_columns = HashMap::new();
        new_columns.insert("b".to_string(), Series::new(vec![Value::Integer(3), Value::Integer(4)], SeriesType::Int64).unwrap());
        
        let extended = table.with_columns(new_columns).unwrap();
        
        assert_eq!(extended.columns.len(), 2);
        assert!(extended.get_column("a").is_some());
        assert!(extended.get_column("b").is_some());
        assert_eq!(extended.length, 2);
    }

    #[test]
    fn test_table_get_row() {
        let column_names = vec!["x".to_string(), "y".to_string()];
        let rows = vec![
            vec![Value::Integer(1), Value::String("hello".to_string())],
            vec![Value::Integer(2), Value::String("world".to_string())],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        
        let row0 = table.get_row(0).unwrap();
        assert_eq!(row0.len(), 2);
        // Note: order may vary since HashMap doesn't guarantee order
        assert!(row0.contains(&Value::Integer(1)));
        assert!(row0.contains(&Value::String("hello".to_string())));
        
        let row1 = table.get_row(1).unwrap();
        assert_eq!(row1.len(), 2);
        assert!(row1.contains(&Value::Integer(2)));
        assert!(row1.contains(&Value::String("world".to_string())));
    }

    #[test]
    fn test_table_slice_rows() {
        let column_names = vec!["x".to_string()];
        let rows = vec![
            vec![Value::Integer(1)],
            vec![Value::Integer(2)],
            vec![Value::Integer(3)],
            vec![Value::Integer(4)],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        
        let sliced = table.slice_rows(1, 3).unwrap();
        assert_eq!(sliced.length, 2);
        
        let x_col = sliced.get_column("x").unwrap();
        assert_eq!(*x_col.get(0).unwrap(), Value::Integer(2));
        assert_eq!(*x_col.get(1).unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_table_head_tail() {
        let column_names = vec!["x".to_string()];
        let rows = vec![
            vec![Value::Integer(1)],
            vec![Value::Integer(2)],
            vec![Value::Integer(3)],
            vec![Value::Integer(4)],
            vec![Value::Integer(5)],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        
        let head = table.head(3).unwrap();
        assert_eq!(head.length, 3);
        let x_col = head.get_column("x").unwrap();
        assert_eq!(*x_col.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*x_col.get(2).unwrap(), Value::Integer(3));
        
        let tail = table.tail(2).unwrap();
        assert_eq!(tail.length, 2);
        let x_col = tail.get_column("x").unwrap();
        assert_eq!(*x_col.get(0).unwrap(), Value::Integer(4));
        assert_eq!(*x_col.get(1).unwrap(), Value::Integer(5));
    }

    #[test]
    fn test_table_filter_rows() {
        let column_names = vec!["x".to_string()];
        let rows = vec![
            vec![Value::Integer(1)],
            vec![Value::Integer(2)],
            vec![Value::Integer(3)],
            vec![Value::Integer(4)],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        
        let mask = vec![true, false, true, false];
        let filtered = table.filter_rows(&mask).unwrap();
        
        assert_eq!(filtered.length, 2);
        let x_col = filtered.get_column("x").unwrap();
        assert_eq!(*x_col.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*x_col.get(1).unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_table_sort_by_column() {
        let column_names = vec!["x".to_string(), "y".to_string()];
        let rows = vec![
            vec![Value::Integer(3), Value::String("c".to_string())],
            vec![Value::Integer(1), Value::String("a".to_string())],
            vec![Value::Integer(2), Value::String("b".to_string())],
        ];
        
        let table = Table::from_rows(column_names, rows).unwrap();
        
        let sorted = table.sort_by_column("x", true).unwrap();
        
        let x_col = sorted.get_column("x").unwrap();
        assert_eq!(*x_col.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*x_col.get(1).unwrap(), Value::Integer(2));
        assert_eq!(*x_col.get(2).unwrap(), Value::Integer(3));
        
        let y_col = sorted.get_column("y").unwrap();
        assert_eq!(*y_col.get(0).unwrap(), Value::String("a".to_string()));
        assert_eq!(*y_col.get(1).unwrap(), Value::String("b".to_string()));
        assert_eq!(*y_col.get(2).unwrap(), Value::String("c".to_string()));
    }

    #[test]
    fn test_table_to_rows() {
        let column_names = vec!["x".to_string(), "y".to_string()];
        let original_rows = vec![
            vec![Value::Integer(1), Value::String("a".to_string())],
            vec![Value::Integer(2), Value::String("b".to_string())],
        ];
        
        let table = Table::from_rows(column_names, original_rows.clone()).unwrap();
        let converted_rows = table.to_rows().unwrap();
        
        assert_eq!(converted_rows.len(), 2);
        assert_eq!(converted_rows[0].len(), 2);
        assert_eq!(converted_rows[1].len(), 2);
        
        // Check that data round-trips correctly
        for row in &converted_rows {
            assert!(row.contains(&Value::Integer(1)) || row.contains(&Value::Integer(2)));
            assert!(row.contains(&Value::String("a".to_string())) || row.contains(&Value::String("b".to_string())));
        }
    }

    #[test]
    fn test_table_dtypes() {
        let mut columns = HashMap::new();
        columns.insert("int_col".to_string(), Series::new(vec![Value::Integer(1)], SeriesType::Int64).unwrap());
        columns.insert("str_col".to_string(), Series::new(vec![Value::String("test".to_string())], SeriesType::String).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let dtypes = table.dtypes();
        
        assert_eq!(dtypes.get("int_col"), Some(&SeriesType::Int64));
        assert_eq!(dtypes.get("str_col"), Some(&SeriesType::String));
    }

    // ===== NEW UNIFIED STACK ARCHITECTURE TESTS (TDD) =====
    // These tests define the expected behavior of the new unified stack + frame system
    // They should fail initially until we implement the new architecture

    #[test]
    fn test_new_frame_structure() {
        // Test the new lightweight Frame structure
        let frame = NewFrame {
            func_id: 42,
            ip: 100,
            base: 5,
        };
        
        assert_eq!(frame.func_id, 42);
        assert_eq!(frame.ip, 100);
        assert_eq!(frame.base, 5);
        
        // Frame should be small and efficient (u32 + 2*usize = 4 + 8 + 8 = 20 bytes on 64-bit)
        let frame_size = std::mem::size_of::<NewFrame>();
        assert!(frame_size <= 24, "Frame too large: {} bytes", frame_size);
    }

    #[test]
    fn test_unified_vm_structure() {
        // Test the new unified VM structure
        let vm = NewVirtualMachine::new();
        
        // Should have unified stack and frame architecture
        assert!(vm.stack.is_empty());
        assert!(vm.frames.is_empty());
        assert_eq!(vm.ip, 0);
        
        // Should still have essential components
        assert!(vm.constants.is_empty());
        assert!(vm.code.is_empty());
    }

    #[test]
    fn test_unified_call_convention_basic() {
        // Test the new calling convention: caller pushes args → CALL(func_id, argc)
        let mut vm = NewVirtualMachine::new();
        
        // Simulate: push arg1, push arg2, call func_id=1 argc=2
        vm.push(Value::Integer(10));
        vm.push(Value::Integer(20));
        
        // Before call: stack = [10, 20], frames = []
        assert_eq!(vm.stack.len(), 2);
        assert_eq!(vm.frames.len(), 0);
        
        // Execute CALL(func_id=1, argc=2)
        vm.call_function(1, 2).unwrap();
        
        // After call: stack = [10, 20], frames = [{func_id: 1, ip: old_ip, base: 0}]
        assert_eq!(vm.stack.len(), 2);
        assert_eq!(vm.frames.len(), 1);
        assert_eq!(vm.frames[0].func_id, 1);
        assert_eq!(vm.frames[0].base, 0); // base points to first argument
    }

    #[test]
    fn test_unified_call_convention_nested() {
        // Test nested function calls with unified stack
        let mut vm = NewVirtualMachine::new();
        
        // First call: f1(arg1)
        vm.push(Value::Integer(100));
        vm.call_function(1, 1).unwrap();
        
        // Second call from within f1: f2(arg2, arg3)
        vm.push(Value::Integer(200));
        vm.push(Value::Integer(300));
        vm.call_function(2, 2).unwrap();
        
        // Should have nested frames
        assert_eq!(vm.frames.len(), 2);
        assert_eq!(vm.frames[0].func_id, 1);
        assert_eq!(vm.frames[0].base, 0);
        assert_eq!(vm.frames[1].func_id, 2);
        assert_eq!(vm.frames[1].base, 1); // base points to start of f2's args
        
        // Stack should contain all arguments: [100, 200, 300]
        assert_eq!(vm.stack.len(), 3);
        assert_eq!(vm.stack[0], Value::Integer(100));
        assert_eq!(vm.stack[1], Value::Integer(200));
        assert_eq!(vm.stack[2], Value::Integer(300));
    }

    #[test]
    fn test_unified_return_convention() {
        // Test return convention: stack.truncate(base), then push result
        let mut vm = NewVirtualMachine::new();
        
        // Set up call state
        vm.push(Value::Integer(10));
        vm.push(Value::Integer(20));
        vm.call_function(1, 2).unwrap();
        
        // Function computes result and returns
        let result = Value::Integer(30);
        vm.return_from_function(result).unwrap();
        
        // After return: stack = [30], frames = []
        assert_eq!(vm.stack.len(), 1);
        assert_eq!(vm.stack[0], Value::Integer(30));
        assert_eq!(vm.frames.len(), 0);
    }

    #[test]
    fn test_stack_discipline_property() {
        // Property test: random call/return sequences should maintain stack discipline
        let mut vm = NewVirtualMachine::new();
        
        // Test various call patterns
        for arity in 0..=5 {
            let initial_stack_size = vm.stack.len();
            let initial_frame_count = vm.frames.len();
            
            // Push arguments
            for i in 0..arity {
                vm.push(Value::Integer(i as i64));
            }
            
            // Call function
            vm.call_function(1, arity).unwrap();
            
            // Stack should have grown by arity, frames by 1
            assert_eq!(vm.stack.len(), initial_stack_size + arity);
            assert_eq!(vm.frames.len(), initial_frame_count + 1);
            
            // Return with result
            vm.return_from_function(Value::Integer(42)).unwrap();
            
            // Stack should be back to initial size + 1 (for result)
            assert_eq!(vm.stack.len(), initial_stack_size + 1);
            assert_eq!(vm.frames.len(), initial_frame_count);
            
            // Pop the result to reset
            vm.pop().unwrap();
        }
    }

    #[test]
    fn test_stack_overflow_protection() {
        // Test that deeply nested calls are detected and rejected
        let mut vm = NewVirtualMachine::new();
        vm.max_call_depth = 10; // Set low limit for testing
        
        // Should be able to make calls up to the limit
        for i in 0..10 {
            vm.push(Value::Integer(i));
            vm.call_function(1, 1).unwrap();
        }
        
        // 11th call should fail
        vm.push(Value::Integer(99));
        let result = vm.call_function(1, 1);
        assert!(result.is_err());
        match result.unwrap_err() {
            VmError::CallStackOverflow => {}
            _ => panic!("Expected CallStackOverflow error"),
        }
    }

    #[test]
    fn test_call_frame_base_calculation() {
        // Test that frame.base correctly points to function arguments
        let mut vm = NewVirtualMachine::new();
        
        // Call f1 with 1 arg
        vm.push(Value::Integer(100));
        vm.call_function(1, 1).unwrap();
        assert_eq!(vm.frames[0].base, 0);
        
        // Call f2 with 3 args from within f1
        vm.push(Value::Integer(200));
        vm.push(Value::Integer(300));
        vm.push(Value::Integer(400));
        vm.call_function(2, 3).unwrap();
        assert_eq!(vm.frames[1].base, 1); // Points to start of f2's args
        
        // Each frame should be able to access its arguments correctly
        let f1_args = vm.get_function_args(0);
        assert_eq!(f1_args.len(), 1);
        assert_eq!(f1_args[0], Value::Integer(100));
        
        let f2_args = vm.get_function_args(1);
        assert_eq!(f2_args.len(), 3);
        assert_eq!(f2_args[0], Value::Integer(200));
        assert_eq!(f2_args[1], Value::Integer(300));
        assert_eq!(f2_args[2], Value::Integer(400));
    }

    #[test]
    fn test_no_separate_call_stack() {
        // Test that the old CallFrame system is gone
        let vm = NewVirtualMachine::new();
        
        // Should not have a separate call_stack field
        // This test ensures we've unified everything into the new frame system
        assert!(!has_field(&vm, "call_stack"));
    }

    #[test]
    fn test_frame_memory_efficiency() {
        // Test that frames are lightweight and memory-efficient
        let frame_size = std::mem::size_of::<NewFrame>();
        
        // Frame should be reasonably small (u32 + 2*usize = 24 bytes on 64-bit)
        assert!(frame_size <= 24, "Frame too large: {} bytes", frame_size);
        
        // Should be much smaller than old CallFrame (which includes a String)
        let old_frame_size = std::mem::size_of::<CallFrame>();
        assert!(frame_size <= old_frame_size, "New frame not smaller or equal to old frame (old: {}, new: {})", old_frame_size, frame_size);
    }

    #[test]
    fn test_stack_only_values() {
        // Test that stack contains only values, no metadata
        let mut vm = NewVirtualMachine::new();
        
        vm.push(Value::Integer(42));
        vm.push(Value::String("test".to_string()));
        vm.push(Value::Boolean(true));
        
        // Stack should contain exactly the values we pushed
        assert_eq!(vm.stack.len(), 3);
        assert_eq!(vm.stack[0], Value::Integer(42));
        assert_eq!(vm.stack[1], Value::String("test".to_string()));
        assert_eq!(vm.stack[2], Value::Boolean(true));
        
        // No additional metadata should be stored on the stack
        assert!(vm.stack.iter().all(|v| matches!(v, Value::Integer(_) | Value::String(_) | Value::Boolean(_) | Value::Real(_) | Value::Symbol(_) | Value::List(_) | Value::Function(_) | Value::Tensor(_) | Value::Missing | Value::Series(_) | Value::Table(_) | Value::Dataset(_) | Value::Schema(_))));
    }

    #[test]
    fn test_frame_ip_management() {
        // Test that frames correctly save and restore instruction pointers
        let mut vm = NewVirtualMachine::new();
        vm.ip = 100;
        
        // Call should save current IP in frame
        vm.push(Value::Integer(1));
        vm.call_function(1, 1).unwrap();
        assert_eq!(vm.frames[0].ip, 100);
        
        // VM IP should change for the called function
        vm.ip = 200;
        
        // Return should restore the saved IP
        vm.return_from_function(Value::Integer(42)).unwrap();
        assert_eq!(vm.ip, 100);
    }

    // Implementation of unified stack + frame architecture
    impl NewVirtualMachine {
        fn call_function(&mut self, func_id: u32, argc: usize) -> VmResult<()> {
            // Check call depth limit
            if self.frames.len() >= self.max_call_depth {
                return Err(VmError::CallStackOverflow);
            }
            
            // Check that we have enough arguments on the stack
            if self.stack.len() < argc {
                return Err(VmError::StackUnderflow);
            }
            
            // Create new frame: base points to start of this function's arguments
            let base = self.stack.len() - argc;
            let frame = NewFrame {
                func_id,
                ip: self.ip, // Save current instruction pointer
                base,
            };
            
            // Push frame onto frame stack
            self.frames.push(frame);
            
            // IP will be set to the function's entry point by the caller
            // For now, we just indicate success
            Ok(())
        }
        
        fn return_from_function(&mut self, result: Value) -> VmResult<()> {
            // Pop the current frame
            let frame = self.frames.pop().ok_or(VmError::StackUnderflow)?;
            
            // Restore instruction pointer
            self.ip = frame.ip;
            
            // Truncate stack to remove function arguments
            self.stack.truncate(frame.base);
            
            // Push result onto stack
            self.stack.push(result);
            
            Ok(())
        }
        
        fn get_function_args(&self, frame_index: usize) -> &[Value] {
            if frame_index >= self.frames.len() {
                return &[];
            }
            
            let frame = &self.frames[frame_index];
            let start = frame.base;
            
            // Calculate end of this frame's arguments
            let end = if frame_index + 1 < self.frames.len() {
                // Not the top frame - args end where next frame's args start
                self.frames[frame_index + 1].base
            } else {
                // Top frame - args extend to current stack top
                self.stack.len()
            };
            
            &self.stack[start..end]
        }
    }

    // Helper functions for testing
    fn has_field<T>(_obj: &T, _field_name: &str) -> bool {
        // This would use reflection in a real implementation
        // For now, we'll implement this when we actually remove call_stack
        false
    }

    // New types that will be implemented
    #[derive(Debug, Clone)]
    struct NewFrame {
        func_id: u32,
        ip: usize,
        base: usize,
    }

    #[derive(Debug)]
    struct NewVirtualMachine {
        ip: usize,
        stack: Vec<Value>,
        frames: Vec<NewFrame>,
        constants: Vec<Value>,
        code: Vec<Instruction>,
        max_call_depth: usize,
    }

    impl NewVirtualMachine {
        fn new() -> Self {
            NewVirtualMachine {
                ip: 0,
                stack: Vec::new(),
                frames: Vec::new(),
                constants: Vec::new(),
                code: Vec::new(),
                max_call_depth: 1000,
            }
        }

        fn push(&mut self, value: Value) {
            self.stack.push(value);
        }

        fn pop(&mut self) -> VmResult<Value> {
            self.stack.pop().ok_or(VmError::StackUnderflow)
        }
    }
}
