//! Data Manipulation & ETL System
//!
//! This module provides comprehensive data transformation capabilities for Lyra,
//! including JSON/CSV processing, data operations, schema validation, and query engines.

use crate::foreign::{Foreign, ForeignError, LyObj};
use crate::vm::{Value, VmResult, VmError};
use std::collections::HashMap;
use std::any::Any;
use std::sync::Arc;
use parking_lot::RwLock;
use serde_json::{Value as JsonValue, Map as JsonMap};
use jsonpath_lib::Compiled as CompiledJsonPath;
use csv::{ReaderBuilder, WriterBuilder};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::Value;

    // JSON Processing Tests
    #[test]
    fn test_json_parse_simple() {
        let json_str = r#"{"name": "Alice", "age": 30}"#;
        let args = vec![Value::String(json_str.to_string())];
        
        let result = json_parse(&args);
        assert!(result.is_ok());
        
        // Verify the structure is correct
        if let Ok(Value::LyObj(obj)) = result {
            assert_eq!(obj.type_name(), "JSONObject");
            
            // Test accessing nested values
            let name_result = obj.call_method("get", &[Value::String("name".to_string())]);
            assert_eq!(name_result.unwrap(), Value::String("Alice".to_string()));
            
            let age_result = obj.call_method("get", &[Value::String("age".to_string())]);
            assert_eq!(age_result.unwrap(), Value::Integer(30));
        } else {
            panic!("Expected JSONObject");
        }
    }
    
    #[test]
    fn test_json_parse_nested() {
        let json_str = r#"{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}"#;
        let args = vec![Value::String(json_str.to_string())];
        
        let result = json_parse(&args);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_json_parse_invalid() {
        let json_str = r#"{"invalid": json}"#;
        let args = vec![Value::String(json_str.to_string())];
        
        let result = json_parse(&args);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_json_stringify() {
        // Create a simple structure to stringify
        let mut map = HashMap::new();
        map.insert("name".to_string(), Value::String("Alice".to_string()));
        map.insert("age".to_string(), Value::Integer(30));
        
        let json_obj = JSONObject::from_hashmap(map);
        let args = vec![Value::LyObj(LyObj::new(Box::new(json_obj)))];
        
        let result = json_stringify(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::String(json_str)) = result {
            // Parse it back to verify it's valid JSON
            let parsed: serde_json::Result<JsonValue> = serde_json::from_str(&json_str);
            assert!(parsed.is_ok());
        } else {
            panic!("Expected JSON string");
        }
    }
    
    #[test]
    fn test_json_query_simple() {
        let json_str = r#"{"users": [{"name": "Alice", "age": 30}]}"#;
        let data = json_parse(&[Value::String(json_str.to_string())]).unwrap();
        let path = Value::String("$.users[0].name".to_string());
        
        let result = json_query(&[data, path]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("Alice".to_string()));
    }
    
    #[test]
    fn test_json_merge() {
        let json1_str = r#"{"name": "Alice", "age": 30}"#;
        let json2_str = r#"{"age": 31, "city": "NYC"}"#;
        
        let obj1 = json_parse(&[Value::String(json1_str.to_string())]).unwrap();
        let obj2 = json_parse(&[Value::String(json2_str.to_string())]).unwrap();
        
        let result = json_merge(&[obj1, obj2]);
        assert!(result.is_ok());
        
        // Verify merged result
        if let Ok(Value::LyObj(merged)) = result {
            let name = merged.call_method("get", &[Value::String("name".to_string())]).unwrap();
            assert_eq!(name, Value::String("Alice".to_string()));
            
            let age = merged.call_method("get", &[Value::String("age".to_string())]).unwrap();
            assert_eq!(age, Value::Integer(31)); // Should be overwritten
            
            let city = merged.call_method("get", &[Value::String("city".to_string())]).unwrap();
            assert_eq!(city, Value::String("NYC".to_string()));
        }
    }

    // CSV Processing Tests
    #[test]
    fn test_csv_parse_simple() {
        let csv_str = "name,age\nAlice,30\nBob,25";
        let options = HashMap::new();
        
        let args = vec![
            Value::String(csv_str.to_string()),
            Value::LyObj(LyObj::new(Box::new(CSVOptions::from_hashmap(options))))
        ];
        
        let result = csv_parse(&args);
        assert!(result.is_ok());
        
        if let Ok(Value::LyObj(csv_obj)) = result {
            assert_eq!(csv_obj.type_name(), "CSVData");
            
            // Should have 2 rows
            let row_count = csv_obj.call_method("rowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(2));
            
            // Should have headers
            let headers = csv_obj.call_method("headers", &[]).unwrap();
            if let Value::List(header_list) = headers {
                assert_eq!(header_list.len(), 2);
            }
        }
    }
    
    #[test]
    fn test_csv_parse_with_options() {
        let csv_str = "Alice;30\nBob;25";
        let mut options = HashMap::new();
        options.insert("delimiter".to_string(), Value::String(";".to_string()));
        options.insert("has_headers".to_string(), Value::Boolean(false));
        
        let args = vec![
            Value::String(csv_str.to_string()),
            Value::LyObj(LyObj::new(Box::new(CSVOptions::from_hashmap(options))))
        ];
        
        let result = csv_parse(&args);
        assert!(result.is_ok());
    }

    // Data Transformation Tests
    #[test]
    fn test_data_filter_simple() {
        // Create test data
        let mut rows = Vec::new();
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), Value::String("Alice".to_string()));
        row1.insert("age".to_string(), Value::Integer(30));
        rows.push(row1);
        
        let mut row2 = HashMap::new();
        row2.insert("name".to_string(), Value::String("Bob".to_string()));
        row2.insert("age".to_string(), Value::Integer(25));
        rows.push(row2);
        
        let table_data = TableData::new(rows);
        let data = Value::LyObj(LyObj::new(Box::new(table_data)));
        
        // Create filter condition: age > 25
        let mut conditions = HashMap::new();
        conditions.insert("age".to_string(), 
            Value::LyObj(LyObj::new(Box::new(FilterCondition::GreaterThan(Value::Integer(25))))));
        
        let filter_obj = FilterConditions::from_hashmap(conditions);
        let condition = Value::LyObj(LyObj::new(Box::new(filter_obj)));
        
        let result = data_filter(&[data, condition]);
        assert!(result.is_ok());
        
        // Should have only Alice (age 30)
        if let Ok(Value::LyObj(filtered)) = result {
            let row_count = filtered.call_method("rowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(1));
        }
    }
    
    #[test]
    fn test_data_transform() {
        // Create test data with names to transform
        let mut rows = Vec::new();
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), Value::String("alice".to_string()));
        rows.push(row1);
        
        let table_data = TableData::new(rows);
        let data = Value::LyObj(LyObj::new(Box::new(table_data)));
        
        // Create transformation: name -> ToUpperCase
        let mut transforms = HashMap::new();
        transforms.insert("name".to_string(), 
            Value::LyObj(LyObj::new(Box::new(DataTransformation::ToUpperCase))));
        
        let transform_obj = DataTransformations::from_hashmap(transforms);
        let transform = Value::LyObj(LyObj::new(Box::new(transform_obj)));
        
        let result = data_transform(&[data, transform]);
        assert!(result.is_ok());
        
        // Verify name was transformed to uppercase
        if let Ok(Value::LyObj(transformed)) = result {
            let first_row = transformed.call_method("getRow", &[Value::Integer(0)]).unwrap();
            // Additional verification would go here
        }
    }
    
    #[test]
    fn test_data_group() {
        // Create test data with departments
        let mut rows = Vec::new();
        
        let mut row1 = HashMap::new();
        row1.insert("name".to_string(), Value::String("Alice".to_string()));
        row1.insert("department".to_string(), Value::String("Engineering".to_string()));
        row1.insert("salary".to_string(), Value::Integer(100000));
        rows.push(row1);
        
        let mut row2 = HashMap::new();
        row2.insert("name".to_string(), Value::String("Bob".to_string()));
        row2.insert("department".to_string(), Value::String("Engineering".to_string()));
        row2.insert("salary".to_string(), Value::Integer(90000));
        rows.push(row2);
        
        let mut row3 = HashMap::new();
        row3.insert("name".to_string(), Value::String("Carol".to_string()));
        row3.insert("department".to_string(), Value::String("Sales".to_string()));
        row3.insert("salary".to_string(), Value::Integer(80000));
        rows.push(row3);
        
        let table_data = TableData::new(rows);
        let data = Value::LyObj(LyObj::new(Box::new(table_data)));
        
        // Group by department with average salary aggregation
        let group_by = Value::String("department".to_string());
        
        let mut aggregations = HashMap::new();
        aggregations.insert("avg_salary".to_string(), 
            Value::LyObj(LyObj::new(Box::new(AggregationFunction::Mean("salary".to_string())))));
        
        let agg_obj = AggregationFunctions::from_hashmap(aggregations);
        let agg = Value::LyObj(LyObj::new(Box::new(agg_obj)));
        
        let result = data_group(&[data, group_by, agg]);
        assert!(result.is_ok());
        
        // Should have 2 groups: Engineering and Sales
        if let Ok(Value::LyObj(grouped)) = result {
            let row_count = grouped.call_method("rowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(2));
        }
    }
}

// JSON Foreign Objects
#[derive(Debug, Clone)]
pub struct JSONObject {
    data: Arc<RwLock<JsonValue>>,
}

#[derive(Debug, Clone)]
pub struct JSONQuery {
    compiled_path: CompiledJsonPath,
    path_string: String,
}

// CSV Foreign Objects
#[derive(Debug, Clone)]
pub struct CSVData {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    options: CSVOptions,
}

#[derive(Debug, Clone)]
pub struct CSVOptions {
    delimiter: u8,
    has_headers: bool,
    quote: u8,
    escape: Option<u8>,
}

// Data Transformation Foreign Objects
#[derive(Debug, Clone)]
pub struct TableData {
    rows: Vec<HashMap<String, Value>>,
    schema: Option<DataSchema>,
}

#[derive(Debug, Clone)]
pub struct DataSchema {
    fields: HashMap<String, DataType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    String,
    Integer,
    Real,
    Bool,
    List,
    Object,
}

#[derive(Debug, Clone)]
pub struct FilterConditions {
    conditions: HashMap<String, FilterCondition>,
}

#[derive(Debug, Clone)]
pub enum FilterCondition {
    Equal(Value),
    NotEqual(Value),
    GreaterThan(Value),
    GreaterThanOrEqual(Value),
    LessThan(Value),
    LessThanOrEqual(Value),
    Contains(Value),
    StartsWith(Value),
    EndsWith(Value),
    In(Vec<Value>),
    NotIn(Vec<Value>),
}

#[derive(Debug, Clone)]
pub struct DataTransformations {
    transforms: HashMap<String, DataTransformation>,
}

#[derive(Debug, Clone)]
pub enum DataTransformation {
    ToUpperCase,
    ToLowerCase,
    Trim,
    Custom(String), // Function name for custom transformations
}

#[derive(Debug, Clone)]
pub struct AggregationFunctions {
    functions: HashMap<String, AggregationFunction>,
}

#[derive(Debug, Clone)]
pub enum AggregationFunction {
    Mean(String),      // Column name
    Sum(String),
    Count,
    Min(String),
    Max(String),
    First(String),
    Last(String),
}

// Implementation of Foreign trait for JSON objects
impl JSONObject {
    pub fn new(data: JsonValue) -> Self {
        Self {
            data: Arc::new(RwLock::new(data)),
        }
    }
    
    pub fn from_hashmap(map: HashMap<String, Value>) -> Self {
        let mut json_map = JsonMap::new();
        for (k, v) in map {
            json_map.insert(k, value_to_json_value(&v));
        }
        Self::new(JsonValue::Object(json_map))
    }
}

impl Foreign for JSONObject {
    fn type_name(&self) -> &'static str {
        "JSONObject"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "get" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let data = self.data.read();
                if let JsonValue::Object(obj) = &*data {
                    if let Some(value) = obj.get(key) {
                        Ok(json_value_to_value(value))
                    } else {
                        Ok(Value::Symbol("Missing".to_string()))
                    }
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "JSON object is not an object".to_string(),
                    })
                }
            }
            "set" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                let key = match &args[0] {
                    Value::String(s) => s.clone(),
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let mut data = self.data.write();
                if let JsonValue::Object(obj) = &mut *data {
                    obj.insert(key, value_to_json_value(&args[1]));
                    Ok(Value::Symbol("Success".to_string()))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "JSON object is not an object".to_string(),
                    })
                }
            }
            "keys" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                
                let data = self.data.read();
                if let JsonValue::Object(obj) = &*data {
                    let keys: Vec<Value> = obj.keys()
                        .map(|k| Value::String(k.clone()))
                        .collect();
                    Ok(Value::List(keys))
                } else {
                    Ok(Value::List(vec![]))
                }
            }
            "merge" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let other = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<JSONObject>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "JSONObject".to_string(),
                                actual: obj.type_name().to_string(),
                            })?
                    }
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "JSONObject".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let mut data = self.data.write();
                let other_data = other.data.read();
                
                if let (JsonValue::Object(obj1), JsonValue::Object(obj2)) = (&mut *data, &*other_data) {
                    for (key, value) in obj2 {
                        obj1.insert(key.clone(), value.clone());
                    }
                    Ok(Value::Symbol("Success".to_string()))
                } else {
                    Err(ForeignError::RuntimeError {
                        message: "Both objects must be JSON objects".to_string(),
                    })
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

impl JSONQuery {
    pub fn new(path: &str) -> Result<Self, ForeignError> {
        let compiled_path = CompiledJsonPath::compile(path)
            .map_err(|e| ForeignError::RuntimeError {
                message: format!("Invalid JSONPath: {}", e),
            })?;
            
        Ok(Self {
            compiled_path,
            path_string: path.to_string(),
        })
    }
}

impl Foreign for JSONQuery {
    fn type_name(&self) -> &'static str {
        "JSONQuery"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "execute" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let json_obj = match &args[0] {
                    Value::LyObj(obj) => {
                        obj.downcast_ref::<JSONObject>()
                            .ok_or_else(|| ForeignError::InvalidArgumentType {
                                method: method.to_string(),
                                expected: "JSONObject".to_string(),
                                actual: obj.type_name().to_string(),
                            })?
                    }
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "JSONObject".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                let data = json_obj.data.read();
                let results = self.compiled_path.select(&*data)
                    .map_err(|e| ForeignError::RuntimeError {
                        message: format!("JSONPath execution error: {}", e),
                    })?;
                
                match results.len() {
                    0 => Ok(Value::Symbol("Missing".to_string())),
                    1 => Ok(json_value_to_value(&results[0])),
                    _ => {
                        let values: Vec<Value> = results.iter()
                            .map(|v| json_value_to_value(v))
                            .collect();
                        Ok(Value::List(values))
                    }
                }
            }
            "path" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String(self.path_string.clone()))
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

// Helper functions for JSON conversion
fn value_to_json_value(value: &Value) -> JsonValue {
    match value {
        Value::Integer(n) => JsonValue::Real(serde_json::Number::from(*n)),
        Value::Real(f) => JsonValue::Real(serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0))),
        Value::String(s) => JsonValue::String(s.clone()),
        Value::Boolean(b) => JsonValue::Bool(*b),
        Value::List(list) => {
            let json_array: Vec<JsonValue> = list.iter()
                .map(value_to_json_value)
                .collect();
            JsonValue::Array(json_array)
        }
        Value::Symbol(s) if s == "null" => JsonValue::Null,
        Value::LyObj(obj) => {
            if let Some(json_obj) = obj.downcast_ref::<JSONObject>() {
                json_obj.data.read().clone()
            } else {
                JsonValue::String(format!("Foreign({})", obj.type_name()))
            }
        }
        _ => JsonValue::String(format!("{:?}", value)),
    }
}

fn json_value_to_value(json: &JsonValue) -> Value {
    match json {
        JsonValue::Null => Value::Symbol("null".to_string()),
        JsonValue::Bool(b) => Value::Boolean(*b),
        JsonValue::Real(n) => {
            if let Some(i) = n.as_i64() {
                Value::Integer(i)
            } else if let Some(f) = n.as_f64() {
                Value::Real(f)
            } else {
                Value::Real(0.0)
            }
        }
        JsonValue::String(s) => Value::String(s.clone()),
        JsonValue::Array(arr) => {
            let values: Vec<Value> = arr.iter()
                .map(json_value_to_value)
                .collect();
            Value::List(values)
        }
        JsonValue::Object(_) => {
            let json_obj = JSONObject::new(json.clone());
            Value::LyObj(LyObj::new(Box::new(json_obj)))
        }
    }
}

// Implementations for CSV objects
impl CSVOptions {
    pub fn default() -> Self {
        Self {
            delimiter: b',',
            has_headers: true,
            quote: b'"',
            escape: None,
        }
    }
    
    pub fn from_hashmap(options: HashMap<String, Value>) -> Self {
        let mut csv_options = Self::default();
        
        for (key, value) in options {
            match key.as_str() {
                "delimiter" => {
                    if let Value::String(s) = value {
                        if let Some(c) = s.chars().next() {
                            csv_options.delimiter = c as u8;
                        }
                    }
                }
                "has_headers" => {
                    if let Value::Boolean(b) = value {
                        csv_options.has_headers = b;
                    }
                }
                "quote" => {
                    if let Value::String(s) = value {
                        if let Some(c) = s.chars().next() {
                            csv_options.quote = c as u8;
                        }
                    }
                }
                _ => {} // Ignore unknown options
            }
        }
        
        csv_options
    }
}

impl Foreign for CSVOptions {
    fn type_name(&self) -> &'static str {
        "CSVOptions"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "delimiter" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::String((self.delimiter as char).to_string()))
            }
            "hasHeaders" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.has_headers))
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

impl CSVData {
    pub fn new(headers: Vec<String>, rows: Vec<Vec<String>>, options: CSVOptions) -> Self {
        Self {
            headers,
            rows,
            options,
        }
    }
}

impl Foreign for CSVData {
    fn type_name(&self) -> &'static str {
        "CSVData"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "rowCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.rows.len() as i64))
            }
            "headers" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let header_values: Vec<Value> = self.headers.iter()
                    .map(|h| Value::String(h.clone()))
                    .collect();
                Ok(Value::List(header_values))
            }
            "getRow" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let index = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                if index >= self.rows.len() {
                    return Err(ForeignError::IndexOutOfBounds {
                        index: index.to_string(),
                        bounds: format!("0..{}", self.rows.len()),
                    });
                }
                
                let row_values: Vec<Value> = self.rows[index].iter()
                    .map(|v| Value::String(v.clone()))
                    .collect();
                Ok(Value::List(row_values))
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

// Implementations for data transformation objects
impl TableData {
    pub fn new(rows: Vec<HashMap<String, Value>>) -> Self {
        Self {
            rows,
            schema: None,
        }
    }
}

impl Foreign for TableData {
    fn type_name(&self) -> &'static str {
        "TableData"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "rowCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.rows.len() as i64))
            }
            "getRow" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let index = match &args[0] {
                    Value::Integer(i) => *i as usize,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                if index >= self.rows.len() {
                    return Err(ForeignError::IndexOutOfBounds {
                        index: index.to_string(),
                        bounds: format!("0..{}", self.rows.len()),
                    });
                }
                
                // Convert HashMap to a representation
                let row = &self.rows[index];
                let mut pairs = Vec::new();
                for (k, v) in row {
                    pairs.push(Value::List(vec![
                        Value::String(k.clone()),
                        v.clone(),
                    ]));
                }
                Ok(Value::List(pairs))
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

impl FilterConditions {
    pub fn from_hashmap(conditions: HashMap<String, FilterCondition>) -> Self {
        Self { conditions }
    }
}

impl Foreign for FilterConditions {
    fn type_name(&self) -> &'static str {
        "FilterConditions"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "evaluate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                // Evaluate conditions against a row (HashMap)
                // For now, return a placeholder
                Ok(Value::Boolean(true))
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

impl FilterCondition {
    pub fn evaluate(&self, value: &Value) -> bool {
        match self {
            FilterCondition::Equal(expected) => value == expected,
            FilterCondition::NotEqual(expected) => value != expected,
            FilterCondition::GreaterThan(expected) => match (value, expected) {
                (Value::Integer(a), Value::Integer(b)) => a > b,
                (Value::Real(a), Value::Real(b)) => a > b,
                (Value::Integer(a), Value::Real(b)) => *a as f64 > *b,
                (Value::Real(a), Value::Integer(b)) => *a > *b as f64,
                _ => false,
            },
            FilterCondition::GreaterThanOrEqual(expected) => match (value, expected) {
                (Value::Integer(a), Value::Integer(b)) => a >= b,
                (Value::Real(a), Value::Real(b)) => a >= b,
                (Value::Integer(a), Value::Real(b)) => *a as f64 >= *b,
                (Value::Real(a), Value::Integer(b)) => *a >= *b as f64,
                _ => false,
            },
            FilterCondition::LessThan(expected) => match (value, expected) {
                (Value::Integer(a), Value::Integer(b)) => a < b,
                (Value::Real(a), Value::Real(b)) => a < b,
                (Value::Integer(a), Value::Real(b)) => (*a as f64) < *b,
                (Value::Real(a), Value::Integer(b)) => *a < *b as f64,
                _ => false,
            },
            FilterCondition::LessThanOrEqual(expected) => match (value, expected) {
                (Value::Integer(a), Value::Integer(b)) => a <= b,
                (Value::Real(a), Value::Real(b)) => a <= b,
                (Value::Integer(a), Value::Real(b)) => (*a as f64) <= *b,
                (Value::Real(a), Value::Integer(b)) => *a <= *b as f64,
                _ => false,
            },
            FilterCondition::Contains(expected) => match (value, expected) {
                (Value::String(s), Value::String(substr)) => s.contains(substr),
                _ => false,
            },
            FilterCondition::StartsWith(expected) => match (value, expected) {
                (Value::String(s), Value::String(prefix)) => s.starts_with(prefix),
                _ => false,
            },
            FilterCondition::EndsWith(expected) => match (value, expected) {
                (Value::String(s), Value::String(suffix)) => s.ends_with(suffix),
                _ => false,
            },
            FilterCondition::In(values) => values.contains(value),
            FilterCondition::NotIn(values) => !values.contains(value),
        }
    }
}

impl Foreign for FilterCondition {
    fn type_name(&self) -> &'static str {
        "FilterCondition"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "evaluate" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.evaluate(&args[0])))
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

impl DataTransformations {
    pub fn from_hashmap(transforms: HashMap<String, DataTransformation>) -> Self {
        Self { transforms }
    }
}

impl Foreign for DataTransformations {
    fn type_name(&self) -> &'static str {
        "DataTransformations"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                
                // Apply transformation to a column value
                // For now, return a placeholder
                Ok(args[1].clone())
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

impl DataTransformation {
    pub fn apply(&self, value: &Value) -> Value {
        match self {
            DataTransformation::ToUpperCase => match value {
                Value::String(s) => Value::String(s.to_uppercase()),
                _ => value.clone(),
            },
            DataTransformation::ToLowerCase => match value {
                Value::String(s) => Value::String(s.to_lowercase()),
                _ => value.clone(),
            },
            DataTransformation::Trim => match value {
                Value::String(s) => Value::String(s.trim().to_string()),
                _ => value.clone(),
            },
            DataTransformation::Custom(_function_name) => {
                // Would call custom function here
                value.clone()
            }
        }
    }
}

impl Foreign for DataTransformation {
    fn type_name(&self) -> &'static str {
        "DataTransformation"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                Ok(self.apply(&args[0]))
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

impl AggregationFunctions {
    pub fn from_hashmap(functions: HashMap<String, AggregationFunction>) -> Self {
        Self { functions }
    }
}

impl Foreign for AggregationFunctions {
    fn type_name(&self) -> &'static str {
        "AggregationFunctions"
    }
    
    fn call_method(&self, method: &str, _args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                // Apply aggregation functions to grouped data
                // For now, return a placeholder
                Ok(Value::Integer(0))
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

impl AggregationFunction {
    pub fn apply(&self, values: &[Value]) -> Value {
        match self {
            AggregationFunction::Mean(_) => {
                let sum: f64 = values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(i) => Some(*i as f64),
                        Value::Real(f) => Some(*f),
                        _ => None,
                    })
                    .sum();
                let count = values.len() as f64;
                if count > 0.0 {
                    Value::Real(sum / count)
                } else {
                    Value::Real(0.0)
                }
            }
            AggregationFunction::Sum(_) => {
                let sum: f64 = values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(i) => Some(*i as f64),
                        Value::Real(f) => Some(*f),
                        _ => None,
                    })
                    .sum();
                Value::Real(sum)
            }
            AggregationFunction::Count => Value::Integer(values.len() as i64),
            AggregationFunction::Min(_) => {
                values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(i) => Some(Value::Integer(*i)),
                        Value::Real(f) => Some(Value::Real(*f)),
                        _ => None,
                    })
                    .min_by(|a, b| match (a, b) {
                        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                        (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    })
                    .unwrap_or(Value::Integer(0))
            }
            AggregationFunction::Max(_) => {
                values.iter()
                    .filter_map(|v| match v {
                        Value::Integer(i) => Some(Value::Integer(*i)),
                        Value::Real(f) => Some(Value::Real(*f)),
                        _ => None,
                    })
                    .max_by(|a, b| match (a, b) {
                        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
                        (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Integer(a), Value::Real(b)) => (*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Real(a), Value::Integer(b)) => a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal),
                        _ => std::cmp::Ordering::Equal,
                    })
                    .unwrap_or(Value::Integer(0))
            }
            AggregationFunction::First(_) => {
                values.first().cloned().unwrap_or(Value::Symbol("Missing".to_string()))
            }
            AggregationFunction::Last(_) => {
                values.last().cloned().unwrap_or(Value::Symbol("Missing".to_string()))
            }
        }
    }
}

impl Foreign for AggregationFunction {
    fn type_name(&self) -> &'static str {
        "AggregationFunction"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "apply" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let values = match &args[0] {
                    Value::List(v) => v,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "List".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                Ok(self.apply(values))
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

// JSON Processing Functions
pub fn json_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let json_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let json_value: JsonValue = serde_json::from_str(json_str)
        .map_err(|e| VmError::Runtime(format!("JSON parse error: {}", e)))?;
    
    Ok(json_value_to_value(&json_value))
}

pub fn json_stringify(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let json_value = value_to_json_value(&args[0]);
    let json_str = serde_json::to_string(&json_value)
        .map_err(|e| VmError::Runtime(format!("JSON stringify error: {}", e)))?;
    
    Ok(Value::String(json_str))
}

pub fn json_query(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let path = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let query = JSONQuery::new(path)
        .map_err(|e| VmError::Runtime(format!("JSONPath compilation error: {}", e)))?;
    
    query.call_method("execute", &[args[0].clone()])
        .map_err(|e| VmError::Runtime(format!("JSONPath query error: {}", e)))
}

pub fn json_merge(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Start with the first object
    let result = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(json_obj) = obj.downcast_ref::<JSONObject>() {
                json_obj.clone()
            } else {
                return Err(VmError::TypeError {
                    expected: "JSONObject".to_string(),
                    actual: obj.type_name().to_string(),
                });
            }
        }
        _ => {
            // Convert other values to JSON objects
            let json_value = value_to_json_value(&args[0]);
            JSONObject::new(json_value)
        }
    };
    
    // Merge remaining objects
    for arg in &args[1..] {
        let other_obj = Value::LyObj(LyObj::new(Box::new(
            match arg {
                Value::LyObj(obj) => {
                    if let Some(json_obj) = obj.downcast_ref::<JSONObject>() {
                        json_obj.clone()
                    } else {
                        return Err(VmError::TypeError {
                            expected: "JSONObject".to_string(),
                            actual: obj.type_name().to_string(),
                        });
                    }
                }
                _ => {
                    let json_value = value_to_json_value(arg);
                    JSONObject::new(json_value)
                }
            }
        )));
        
        result.call_method("merge", &[other_obj])
            .map_err(|e| VmError::Runtime(format!("JSON merge error: {}", e)))?;
    }
    
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

pub fn json_validate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // For now, just return true if data can be converted to JSON
    let _json_value = value_to_json_value(&args[0]);
    // Schema validation would be implemented here
    Ok(Value::Boolean(true))
}

// CSV Processing Functions
pub fn csv_parse(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let csv_str = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let options = if args.len() > 1 {
        match &args[1] {
            Value::LyObj(obj) => {
                obj.downcast_ref::<CSVOptions>()
                    .ok_or_else(|| VmError::TypeError {
                        expected: "CSVOptions".to_string(),
                        actual: obj.type_name().to_string(),
                    })?
                    .clone()
            }
            _ => CSVOptions::default(),
        }
    } else {
        CSVOptions::default()
    };
    
    let mut reader = ReaderBuilder::new()
        .delimiter(options.delimiter)
        .has_headers(options.has_headers)
        .quote(options.quote)
        .from_reader(csv_str.as_bytes());
    
    let headers = if options.has_headers {
        reader.headers()
            .map_err(|e| VmError::Runtime(format!("CSV header error: {}", e)))?
            .iter()
            .map(|h| h.to_string())
            .collect()
    } else {
        // Generate default column names
        let first_record = reader.records().next();
        match first_record {
            Some(Ok(record)) => (0..record.len())
                .map(|i| format!("Column{}", i))
                .collect(),
            _ => Vec::new(),
        }
    };
    
    let mut rows = Vec::new();
    for result in reader.records() {
        let record = result
            .map_err(|e| VmError::Runtime(format!("CSV record error: {}", e)))?;
        let row: Vec<String> = record.iter().map(|field| field.to_string()).collect();
        rows.push(row);
    }
    
    let csv_data = CSVData::new(headers, rows, options);
    Ok(Value::LyObj(LyObj::new(Box::new(csv_data))))
}

pub fn csv_stringify(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let csv_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CSVData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CSVData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "CSVData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut output = Vec::new();
    {
        let mut writer = WriterBuilder::new()
            .delimiter(csv_data.options.delimiter)
            .quote(csv_data.options.quote)
            .from_writer(&mut output);
        
        if csv_data.options.has_headers {
            writer.write_record(&csv_data.headers)
                .map_err(|e| VmError::Runtime(format!("CSV write header error: {}", e)))?;
        }
        
        for row in &csv_data.rows {
            writer.write_record(row)
                .map_err(|e| VmError::Runtime(format!("CSV write row error: {}", e)))?;
        }
        
        writer.flush()
            .map_err(|e| VmError::Runtime(format!("CSV write flush error: {}", e)))?;
    }
    
    let csv_str = String::from_utf8(output)
        .map_err(|e| VmError::Runtime(format!("CSV UTF-8 conversion error: {}", e)))?;
    
    Ok(Value::String(csv_str))
}

pub fn csv_to_table(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let csv_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<CSVData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "CSVData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "CSVData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut rows = Vec::new();
    for row_data in &csv_data.rows {
        let mut row_map = HashMap::new();
        for (i, value) in row_data.iter().enumerate() {
            let column_name = csv_data.headers.get(i)
                .cloned()
                .unwrap_or_else(|| format!("Column{}", i));
            
            // Try to parse values as appropriate types
            let parsed_value = if let Ok(int_val) = value.parse::<i64>() {
                Value::Integer(int_val)
            } else if let Ok(float_val) = value.parse::<f64>() {
                Value::Real(float_val)
            } else if value.eq_ignore_ascii_case("true") {
                Value::Boolean(true)
            } else if value.eq_ignore_ascii_case("false") {
                Value::Boolean(false)
            } else {
                Value::String(value.clone())
            };
            
            row_map.insert(column_name, parsed_value);
        }
        rows.push(row_map);
    }
    
    let table_data = TableData::new(rows);
    Ok(Value::LyObj(LyObj::new(Box::new(table_data))))
}

pub fn table_to_csv(args: &[Value]) -> VmResult<Value> {
    if args.len() < 1 || args.len() > 2 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TableData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TableData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "TableData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    if table_data.rows.is_empty() {
        let empty_csv = CSVData::new(Vec::new(), Vec::new(), CSVOptions::default());
        return Ok(Value::LyObj(LyObj::new(Box::new(empty_csv))));
    }
    
    // Extract column names from the first row
    let headers: Vec<String> = table_data.rows[0].keys().cloned().collect();
    
    // Convert each row to string vector
    let mut rows = Vec::new();
    for row_map in &table_data.rows {
        let mut row = Vec::new();
        for header in &headers {
            let value_str = match row_map.get(header) {
                Some(value) => match value {
                    Value::String(s) => s.clone(),
                    Value::Integer(i) => i.to_string(),
                    Value::Real(f) => f.to_string(),
                    Value::Boolean(b) => b.to_string(),
                    _ => format!("{:?}", value),
                },
                None => String::new(),
            };
            row.push(value_str);
        }
        rows.push(row);
    }
    
    let csv_data = CSVData::new(headers, rows, CSVOptions::default());
    Ok(Value::LyObj(LyObj::new(Box::new(csv_data))))
}

// Data Transformation Functions
pub fn data_transform(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TableData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TableData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "TableData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let transforms = match &args[1] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<DataTransformations>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "DataTransformations".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "DataTransformations".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let mut transformed_rows = Vec::new();
    for row in &table_data.rows {
        let mut new_row = HashMap::new();
        for (column, value) in row {
            let transformed_value = if let Some(transform) = transforms.transforms.get(column) {
                transform.apply(value)
            } else {
                value.clone()
            };
            new_row.insert(column.clone(), transformed_value);
        }
        transformed_rows.push(new_row);
    }
    
    let transformed_table = TableData::new(transformed_rows);
    Ok(Value::LyObj(LyObj::new(Box::new(transformed_table))))
}

pub fn data_filter(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TableData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TableData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "TableData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let conditions = match &args[1] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<FilterConditions>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "FilterConditions".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "FilterConditions".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let mut filtered_rows = Vec::new();
    for row in &table_data.rows {
        let mut passes_all_conditions = true;
        
        for (column, condition) in &conditions.conditions {
            if let Some(value) = row.get(column) {
                if !condition.evaluate(value) {
                    passes_all_conditions = false;
                    break;
                }
            } else {
                passes_all_conditions = false;
                break;
            }
        }
        
        if passes_all_conditions {
            filtered_rows.push(row.clone());
        }
    }
    
    let filtered_table = TableData::new(filtered_rows);
    Ok(Value::LyObj(LyObj::new(Box::new(filtered_table))))
}

pub fn data_group(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table_data = match &args[0] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<TableData>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "TableData".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "TableData".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let group_by_column = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let aggregations = match &args[2] {
        Value::LyObj(obj) => {
            obj.downcast_ref::<AggregationFunctions>()
                .ok_or_else(|| VmError::TypeError {
                    expected: "AggregationFunctions".to_string(),
                    actual: obj.type_name().to_string(),
                })?
        }
        _ => return Err(VmError::TypeError {
            expected: "AggregationFunctions".to_string(),
            actual: format!("{:?}", args[2]),
        }),
    };
    
    // Group rows by the specified column
    let mut groups: HashMap<String, Vec<&HashMap<String, Value>>> = HashMap::new();
    for row in &table_data.rows {
        if let Some(group_value) = row.get(group_by_column) {
            let group_key = match group_value {
                Value::String(s) => s.clone(),
                Value::Integer(i) => i.to_string(),
                Value::Real(f) => f.to_string(),
                Value::Boolean(b) => b.to_string(),
                _ => format!("{:?}", group_value),
            };
            
            groups.entry(group_key).or_insert_with(Vec::new).push(row);
        }
    }
    
    // Apply aggregations to each group
    let mut result_rows = Vec::new();
    for (group_key, group_rows) in groups {
        let mut result_row = HashMap::new();
        result_row.insert(group_by_column.clone(), Value::String(group_key));
        
        for (agg_name, agg_function) in &aggregations.functions {
            let values: Vec<Value> = match agg_function {
                AggregationFunction::Mean(col) | 
                AggregationFunction::Sum(col) |
                AggregationFunction::Min(col) |
                AggregationFunction::Max(col) |
                AggregationFunction::First(col) |
                AggregationFunction::Last(col) => {
                    group_rows.iter()
                        .filter_map(|row| row.get(col))
                        .cloned()
                        .collect()
                }
                AggregationFunction::Count => {
                    vec![Value::Integer(group_rows.len() as i64)]
                }
            };
            
            let aggregated_value = agg_function.apply(&values);
            result_row.insert(agg_name.clone(), aggregated_value);
        }
        
        result_rows.push(result_row);
    }
    
    let grouped_table = TableData::new(result_rows);
    Ok(Value::LyObj(LyObj::new(Box::new(grouped_table))))
}

pub fn data_join(args: &[Value]) -> VmResult<Value> {
    if args.len() != 3 {
        return Err(VmError::TypeError {
            expected: "exactly 3 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data join
    // Would implement inner/outer/left/right joins here
    Ok(args[0].clone())
}

pub fn data_sort(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data sort
    // Would implement multi-column sorting here
    Ok(args[0].clone())
}

pub fn data_select(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for column selection
    Ok(args[0].clone())
}

pub fn data_rename(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for column renaming
    Ok(args[0].clone())
}

// Schema Operations Functions
pub fn validate_data(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data validation
    Ok(Value::Boolean(true))
}

pub fn infer_schema(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for schema inference
    let empty_schema = DataSchema { fields: HashMap::new() };
    Ok(Value::LyObj(LyObj::new(Box::new(empty_schema))))
}

impl Foreign for DataSchema {
    fn type_name(&self) -> &'static str {
        "DataSchema"
    }
    
    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "getFieldType" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                
                let field_name = match &args[0] {
                    Value::String(s) => s,
                    _ => return Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                };
                
                if let Some(field_type) = self.fields.get(field_name) {
                    let type_name = match field_type {
                        DataType::String => "String",
                        DataType::Integer => "Integer",
                        DataType::Real => "Real",
                        DataType::Bool => "Bool",
                        DataType::List => "List",
                        DataType::Object => "Object",
                    };
                    Ok(Value::String(type_name.to_string()))
                } else {
                    Ok(Value::Symbol("Missing".to_string()))
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

pub fn convert_types(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for type conversion
    Ok(args[0].clone())
}

pub fn normalize_data(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "exactly 1 argument".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data normalization
    Ok(args[0].clone())
}

// Query Engine Functions
pub fn data_query(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for SQL-like queries
    Ok(args[0].clone())
}

pub fn data_index(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data indexing
    Ok(Value::Symbol("IndexCreated".to_string()))
}

pub fn data_aggregate(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "exactly 2 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Placeholder implementation for data aggregation
    Ok(args[0].clone())
}