use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmResult},
};
use std::any::Any;

/// Foreign Dataset implementation - hierarchical data structure wrapper
/// Thread-safe Dataset for complex nested data structures
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignDataset {
    pub value: Box<Value>, // Nested List/Association structures
}

impl ForeignDataset {
    /// Create a new Dataset from a value
    pub fn new(value: Value) -> Self {
        ForeignDataset {
            value: Box::new(value),
        }
    }

    /// Create Dataset from nested lists/associations
    pub fn from_nested(value: Value) -> VmResult<Self> {
        // TODO: Validate that the value is appropriate for hierarchical data
        // For now, accept any nested structure
        Ok(ForeignDataset::new(value))
    }

    /// Get the underlying value
    pub fn get_value(&self) -> &Value {
        &self.value
    }

    /// Extract a clone of the underlying value
    pub fn to_value(&self) -> Value {
        (*self.value).clone()
    }

    /// Check if the dataset is empty
    pub fn is_empty(&self) -> bool {
        match &*self.value {
            Value::List(items) => items.is_empty(),
            Value::Missing => true,
            _ => false,
        }
    }

    /// Get the size/length of the dataset
    pub fn size(&self) -> usize {
        match &*self.value {
            Value::List(items) => items.len(),
            Value::Missing => 0,
            _ => 1, // Single value has size 1
        }
    }

    /// Convert to a list representation if possible
    pub fn to_list(&self) -> Option<Vec<Value>> {
        match &*self.value {
            Value::List(items) => Some(items.clone()),
            _ => None,
        }
    }

    /// Create a dataset from a list of values
    pub fn from_list(values: Vec<Value>) -> Self {
        ForeignDataset::new(Value::List(values))
    }

    /// Create an empty dataset
    pub fn empty() -> Self {
        ForeignDataset::new(Value::List(Vec::new()))
    }

    /// Flatten nested structures to a single level (simple implementation)
    pub fn flatten(&self) -> Self {
        let flattened_value = self.flatten_value(&self.value);
        ForeignDataset::new(flattened_value)
    }

    /// Helper function to recursively flatten values
    fn flatten_value(&self, value: &Value) -> Value {
        match value {
            Value::List(items) => {
                let mut flattened_items = Vec::new();
                for item in items {
                    match item {
                        Value::List(nested_items) => {
                            // Recursively flatten nested lists
                            flattened_items.extend(nested_items.iter().cloned());
                        }
                        _ => flattened_items.push(item.clone()),
                    }
                }
                Value::List(flattened_items)
            }
            _ => value.clone(), // Non-list values remain as-is
        }
    }

    /// Get a string representation of the dataset structure
    pub fn describe(&self) -> String {
        match &*self.value {
            Value::List(items) => {
                format!("Dataset with {} items of type List", items.len())
            }
            Value::Missing => "Empty dataset".to_string(),
            _ => format!("Dataset with single value of type {:?}", self.value),
        }
    }

    /// Filter the dataset based on a predicate (simplified implementation)
    pub fn filter<F>(&self, predicate: F) -> Self 
    where
        F: Fn(&Value) -> bool,
    {
        match &*self.value {
            Value::List(items) => {
                let filtered_items: Vec<Value> = items.iter()
                    .filter(|item| predicate(item))
                    .cloned()
                    .collect();
                ForeignDataset::new(Value::List(filtered_items))
            }
            _ => {
                // For non-list values, apply predicate to the single value
                if predicate(&self.value) {
                    self.clone()
                } else {
                    ForeignDataset::empty()
                }
            }
        }
    }

    /// Map values in the dataset (simplified implementation)
    pub fn map<F>(&self, mapper: F) -> Self 
    where
        F: Fn(&Value) -> Value,
    {
        match &*self.value {
            Value::List(items) => {
                let mapped_items: Vec<Value> = items.iter()
                    .map(|item| mapper(item))
                    .collect();
                ForeignDataset::new(Value::List(mapped_items))
            }
            _ => {
                // For non-list values, apply mapper to the single value
                let mapped_value = mapper(&self.value);
                ForeignDataset::new(mapped_value)
            }
        }
    }

    /// Append a value to the dataset
    pub fn append(&self, value: Value) -> Self {
        match &*self.value {
            Value::List(items) => {
                let mut new_items = items.clone();
                new_items.push(value);
                ForeignDataset::new(Value::List(new_items))
            }
            _ => {
                // Convert single value to list and append
                let new_items = vec![(*self.value).clone(), value];
                ForeignDataset::new(Value::List(new_items))
            }
        }
    }

    /// Prepend a value to the dataset
    pub fn prepend(&self, value: Value) -> Self {
        match &*self.value {
            Value::List(items) => {
                let mut new_items = vec![value];
                new_items.extend(items.iter().cloned());
                ForeignDataset::new(Value::List(new_items))
            }
            _ => {
                // Convert single value to list and prepend
                let new_items = vec![value, (*self.value).clone()];
                ForeignDataset::new(Value::List(new_items))
            }
        }
    }
}

impl Foreign for ForeignDataset {
    fn type_name(&self) -> &'static str {
        "Dataset"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "Size" | "Length" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.size() as i64))
            }
            "IsEmpty" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.is_empty()))
            }
            "ToList" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                match self.to_list() {
                    Some(list) => Ok(Value::List(list)),
                    None => Err(ForeignError::RuntimeError {
                        message: "Dataset cannot be converted to list".to_string(),
                    }),
                }
            }
            "ToValue" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(self.to_value())
            }
            "Describe" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.describe()))
            }
            "Flatten" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let flattened = self.flatten();
                Ok(Value::LyObj(LyObj::new(Box::new(flattened))))
            }
            "Append" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let new_dataset = self.append(args[0].clone());
                Ok(Value::LyObj(LyObj::new(Box::new(new_dataset))))
            }
            "Prepend" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                let new_dataset = self.prepend(args[0].clone());
                Ok(Value::LyObj(LyObj::new(Box::new(new_dataset))))
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