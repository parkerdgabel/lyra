use lyra::{
    foreign::{Foreign, ForeignError, LyObj},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::collections::HashMap;

/// Test-only Value type that is Send + Sync for TDD tests
#[derive(Debug, Clone, PartialEq)]
pub enum TestValue {
    Integer(i64),
    Real(f64),
    String(String),
    Boolean(bool),
    List(Vec<TestValue>),
}

impl TestValue {
    /// Convert TestValue to real Value for integration
    pub fn to_value(&self) -> Value {
        match self {
            TestValue::Integer(i) => Value::Integer(*i),
            TestValue::Real(f) => Value::Real(*f),
            TestValue::String(s) => Value::String(s.clone()),
            TestValue::Boolean(b) => Value::Boolean(*b),
            TestValue::List(list) => Value::List(list.iter().map(|v| v.to_value()).collect()),
        }
    }
    
    /// Convert real Value to TestValue for testing
    pub fn from_value(value: &Value) -> Self {
        match value {
            Value::Integer(i) => TestValue::Integer(*i),
            Value::Real(f) => TestValue::Real(*f),
            Value::String(s) => TestValue::String(s.clone()),
            Value::Boolean(b) => TestValue::Boolean(*b),
            Value::List(list) => TestValue::List(list.iter().map(|v| TestValue::from_value(v)).collect()),
            _ => TestValue::String(format!("Unsupported: {:?}", value)),
        }
    }
}

/// Simple thread-safe series for Foreign Table implementation
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignSeries {
    pub data: Vec<TestValue>,
    pub length: usize,
}

impl ForeignSeries {
    pub fn new(data: Vec<TestValue>) -> Self {
        let length = data.len();
        ForeignSeries { data, length }
    }
    
    pub fn from_values(data: Vec<Value>) -> Self {
        let test_data: Vec<TestValue> = data.iter().map(|v| TestValue::from_value(v)).collect();
        ForeignSeries::new(test_data)
    }
    
    pub fn get(&self, index: usize) -> VmResult<&TestValue> {
        if index >= self.length {
            return Err(VmError::IndexError {
                index: index as i64,
                length: self.length,
            });
        }
        Ok(&self.data[index])
    }
    
    pub fn slice(&self, start: usize, end: usize) -> VmResult<Self> {
        if start > end || end > self.length {
            return Err(VmError::IndexError {
                index: end as i64,
                length: self.length,
            });
        }
        Ok(ForeignSeries::new(self.data[start..end].to_vec()))
    }
}

/// Foreign Table implementation that will replace Value::Table
/// This struct mirrors the existing Table but implements Foreign trait
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignTable {
    pub columns: HashMap<String, ForeignSeries>,
    pub length: usize,
    pub index: Option<Vec<TestValue>>,
}

impl ForeignTable {
    /// Create a new empty Table
    pub fn new() -> Self {
        ForeignTable {
            columns: HashMap::new(),
            length: 0,
            index: None,
        }
    }
    
    /// Create a Table from columns, validating all series have same length
    pub fn from_columns(columns: HashMap<String, ForeignSeries>) -> VmResult<Self> {
        if columns.is_empty() {
            return Ok(ForeignTable::new());
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
        
        Ok(ForeignTable {
            length: first_length,
            columns,
            index: None,
        })
    }
    
    /// Create Table from rows with column names
    pub fn from_rows(column_names: Vec<String>, rows: Vec<Vec<TestValue>>) -> VmResult<Self> {
        if rows.is_empty() {
            return Ok(ForeignTable::new());
        }
        
        let num_rows = rows.len();
        let num_cols = column_names.len();
        
        // Validate all rows have same length
        for (row_idx, row) in rows.iter().enumerate() {
            if row.len() != num_cols {
                return Err(VmError::TypeError {
                    expected: format!("row with {} columns", num_cols),
                    actual: format!("row {} has {} columns", row_idx, row.len()),
                });
            }
        }
        
        let mut columns = HashMap::new();
        
        // Build columns from rows
        for (col_idx, column_name) in column_names.into_iter().enumerate() {
            let mut column_data = Vec::with_capacity(num_rows);
            for row in &rows {
                column_data.push(row[col_idx].clone());
            }
            
            // Create series from column data
            let series = ForeignSeries::new(column_data);
            columns.insert(column_name, series);
        }
        
        ForeignTable::from_columns(columns)
    }
    
    /// Get column names
    pub fn column_names(&self) -> Vec<&String> {
        self.columns.keys().collect()
    }
    
    /// Get a column by name
    pub fn get_column(&self, name: &str) -> Option<&ForeignSeries> {
        self.columns.get(name)
    }
    
    /// Get a row by index
    pub fn get_row(&self, index: usize) -> VmResult<Vec<TestValue>> {
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
        
        Ok(ForeignTable {
            columns: new_columns,
            length: end - start,
            index: new_index,
        })
    }
}

impl Foreign for ForeignTable {
    fn type_name(&self) -> &'static str {
        "Table"
    }

    fn call_method(&self, method: &str, args: &[Value]) -> Result<Value, ForeignError> {
        match method {
            "RowCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.length as i64))
            }
            "ColumnCount" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Integer(self.columns.len() as i64))
            }
            "ColumnNames" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                let names: Vec<Value> = self.column_names()
                    .iter()
                    .map(|name| Value::String((*name).clone()))
                    .collect();
                Ok(Value::List(names))
            }
            "GetCell" => {
                if args.len() != 2 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 2,
                        actual: args.len(),
                    });
                }
                match (&args[0], &args[1]) {
                    (Value::Integer(row), Value::Integer(col)) => {
                        let row_idx = *row as usize;
                        let col_idx = *col as usize;
                        
                        if row_idx >= self.length {
                            return Err(ForeignError::IndexOutOfBounds {
                                index: format!("row {}", row),
                                bounds: format!("0..{}", self.length),
                            });
                        }
                        
                        let column_names: Vec<&String> = self.column_names();
                        if col_idx >= column_names.len() {
                            return Err(ForeignError::IndexOutOfBounds {
                                index: format!("column {}", col),
                                bounds: format!("0..{}", column_names.len()),
                            });
                        }
                        
                        let column_name = column_names[col_idx];
                        let series = self.columns.get(column_name).unwrap();
                        let test_value = series.get(row_idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Series access error: {}", e),
                        })?;
                        
                        Ok(test_value.to_value())
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args),
                    }),
                }
            }
            "GetRow" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(row) => {
                        let row_idx = *row as usize;
                        let test_row_values = self.get_row(row_idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Row access error: {}", e),
                        })?;
                        let row_values: Vec<Value> = test_row_values.iter().map(|v| v.to_value()).collect();
                        Ok(Value::List(row_values))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "GetColumn" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::String(col_name) | Value::Symbol(col_name) => {
                        if let Some(series) = self.get_column(col_name) {
                            let column_values: Vec<Value> = (0..series.length)
                                .map(|i| series.get(i).unwrap().to_value())
                                .collect();
                            Ok(Value::List(column_values))
                        } else {
                            Err(ForeignError::RuntimeError {
                                message: format!("Column '{}' not found", col_name),
                            })
                        }
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "String or Symbol".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "Head" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(n) => {
                        let n_rows = (*n as usize).min(self.length);
                        let head_table = self.slice_rows(0, n_rows).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Head operation error: {}", e),
                        })?;
                        Ok(Value::LyObj(LyObj::new(Box::new(head_table))))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "Tail" => {
                if args.len() != 1 {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 1,
                        actual: args.len(),
                    });
                }
                match &args[0] {
                    Value::Integer(n) => {
                        let n_rows = *n as usize;
                        let start = if n_rows >= self.length { 0 } else { self.length - n_rows };
                        let tail_table = self.slice_rows(start, self.length).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Tail operation error: {}", e),
                        })?;
                        Ok(Value::LyObj(LyObj::new(Box::new(tail_table))))
                    }
                    _ => Err(ForeignError::InvalidArgumentType {
                        method: method.to_string(),
                        expected: "Integer".to_string(),
                        actual: format!("{:?}", args[0]),
                    }),
                }
            }
            "IsEmpty" => {
                if !args.is_empty() {
                    return Err(ForeignError::InvalidArity {
                        method: method.to_string(),
                        expected: 0,
                        actual: args.len(),
                    });
                }
                Ok(Value::Boolean(self.length == 0))
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

// ==========================================
// TDD Tests for Foreign Table Implementation
// ==========================================

/// Helper function to create a simple test table
fn create_test_table() -> ForeignTable {
    let mut columns = HashMap::new();
    
    columns.insert("id".to_string(), ForeignSeries::new(
        vec![TestValue::Integer(1), TestValue::Integer(2), TestValue::Integer(3)]
    ));
    
    columns.insert("name".to_string(), ForeignSeries::new(
        vec![
            TestValue::String("Alice".to_string()),
            TestValue::String("Bob".to_string()),
            TestValue::String("Charlie".to_string())
        ]
    ));
    
    columns.insert("age".to_string(), ForeignSeries::new(
        vec![TestValue::Integer(25), TestValue::Integer(30), TestValue::Integer(35)]
    ));
    
    ForeignTable::from_columns(columns).unwrap()
}

// ==========================================
// Basic Foreign Table Tests
// ==========================================

#[test]
fn test_foreign_table_creation() {
    let table = create_test_table();
    
    assert_eq!(table.length, 3);
    assert_eq!(table.columns.len(), 3);
    assert!(table.get_column("id").is_some());
    assert!(table.get_column("name").is_some());
    assert!(table.get_column("age").is_some());
}

#[test]
fn test_foreign_table_empty() {
    let table = ForeignTable::new();
    
    assert_eq!(table.length, 0);
    assert_eq!(table.columns.len(), 0);
    assert!(table.index.is_none());
}

#[test]
fn test_foreign_table_from_rows() {
    let column_names = vec!["x".to_string(), "y".to_string()];
    let rows = vec![
        vec![TestValue::Integer(1), TestValue::Integer(10)],
        vec![TestValue::Integer(2), TestValue::Integer(20)],
        vec![TestValue::Integer(3), TestValue::Integer(30)],
    ];
    
    let table = ForeignTable::from_rows(column_names, rows).unwrap();
    
    assert_eq!(table.length, 3);
    assert_eq!(table.columns.len(), 2);
    
    let x_col = table.get_column("x").unwrap();
    assert_eq!(*x_col.get(0).unwrap(), TestValue::Integer(1));
    assert_eq!(*x_col.get(1).unwrap(), TestValue::Integer(2));
    assert_eq!(*x_col.get(2).unwrap(), TestValue::Integer(3));
}

#[test]
fn test_foreign_table_get_row() {
    let table = create_test_table();
    
    let row0 = table.get_row(0).unwrap();
    assert_eq!(row0.len(), 3);
    assert!(row0.contains(&TestValue::Integer(1))); // id
    assert!(row0.contains(&TestValue::String("Alice".to_string()))); // name
    assert!(row0.contains(&TestValue::Integer(25))); // age
    
    let row1 = table.get_row(1).unwrap();
    assert!(row1.contains(&TestValue::Integer(2)));
    assert!(row1.contains(&TestValue::String("Bob".to_string())));
    assert!(row1.contains(&TestValue::Integer(30)));
}

#[test]
fn test_foreign_table_slice_rows() {
    let table = create_test_table();
    
    let sliced = table.slice_rows(1, 3).unwrap();
    assert_eq!(sliced.length, 2);
    
    let id_col = sliced.get_column("id").unwrap();
    assert_eq!(*id_col.get(0).unwrap(), TestValue::Integer(2));
    assert_eq!(*id_col.get(1).unwrap(), TestValue::Integer(3));
}

// ==========================================
// Foreign Trait Method Tests
// ==========================================

#[test]
fn test_foreign_table_row_count_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("RowCount", &[]).unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_foreign_table_column_count_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("ColumnCount", &[]).unwrap();
    assert_eq!(result, Value::Integer(3));
}

#[test]
fn test_foreign_table_column_names_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("ColumnNames", &[]).unwrap();
    match result {
        Value::List(names) => {
            assert_eq!(names.len(), 3);
            let name_strings: Vec<String> = names.iter()
                .filter_map(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    _ => None,
                })
                .collect();
            assert!(name_strings.contains(&"id".to_string()));
            assert!(name_strings.contains(&"name".to_string()));
            assert!(name_strings.contains(&"age".to_string()));
        }
        _ => panic!("Expected List result"),
    }
}

#[test]
fn test_foreign_table_get_cell_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    // Test GetCell[table, 0, 0] (row 0, column 0)
    let result = table_obj.call_method("GetCell", &[Value::Integer(0), Value::Integer(0)]).unwrap();
    // The exact value depends on column ordering, but should be one of the first row values
    assert!(matches!(result, Value::Integer(_) | Value::String(_)));
    
    // Test GetCell[table, 1, 1] (row 1, column 1)
    let result = table_obj.call_method("GetCell", &[Value::Integer(1), Value::Integer(1)]).unwrap();
    assert!(matches!(result, Value::Integer(_) | Value::String(_)));
}

#[test]
fn test_foreign_table_get_row_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("GetRow", &[Value::Integer(0)]).unwrap();
    match result {
        Value::List(row) => {
            assert_eq!(row.len(), 3);
            assert!(row.contains(&Value::Integer(1))); // id
            assert!(row.contains(&Value::String("Alice".to_string()))); // name
            assert!(row.contains(&Value::Integer(25))); // age
        }
        _ => panic!("Expected List result"),
    }
}

#[test]
fn test_foreign_table_get_column_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("GetColumn", &[Value::String("id".to_string())]).unwrap();
    match result {
        Value::List(column) => {
            assert_eq!(column.len(), 3);
            assert_eq!(column[0], Value::Integer(1));
            assert_eq!(column[1], Value::Integer(2));
            assert_eq!(column[2], Value::Integer(3));
        }
        _ => panic!("Expected List result"),
    }
}

#[test]
fn test_foreign_table_head_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("Head", &[Value::Integer(2)]).unwrap();
    match result {
        Value::LyObj(head_table) => {
            let row_count = head_table.call_method("RowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(2));
        }
        _ => panic!("Expected LyObj result"),
    }
}

#[test]
fn test_foreign_table_tail_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("Tail", &[Value::Integer(2)]).unwrap();
    match result {
        Value::LyObj(tail_table) => {
            let row_count = tail_table.call_method("RowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(2));
        }
        _ => panic!("Expected LyObj result"),
    }
}

#[test]
fn test_foreign_table_is_empty_method() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("IsEmpty", &[]).unwrap();
    assert_eq!(result, Value::Boolean(false));
    
    // Test with empty table
    let empty_table = ForeignTable::new();
    let empty_obj = LyObj::new(Box::new(empty_table));
    
    let result = empty_obj.call_method("IsEmpty", &[]).unwrap();
    assert_eq!(result, Value::Boolean(true));
}

// ==========================================
// Error Handling Tests
// ==========================================

#[test]
fn test_foreign_table_method_error_handling() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    // Test unknown method
    let result = table_obj.call_method("UnknownMethod", &[]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::UnknownMethod { method, .. } => {
            assert_eq!(method, "UnknownMethod");
        }
        _ => panic!("Expected UnknownMethod error"),
    }
    
    // Test invalid arity
    let result = table_obj.call_method("RowCount", &[Value::Integer(1)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::InvalidArity { method, expected, actual } => {
            assert_eq!(method, "RowCount");
            assert_eq!(expected, 0);
            assert_eq!(actual, 1);
        }
        _ => panic!("Expected InvalidArity error"),
    }
    
    // Test index out of bounds
    let result = table_obj.call_method("GetCell", &[Value::Integer(10), Value::Integer(0)]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::IndexOutOfBounds { .. } => {}
        _ => panic!("Expected IndexOutOfBounds error"),
    }
    
    // Test invalid argument type
    let result = table_obj.call_method("GetRow", &[Value::String("invalid".to_string())]);
    assert!(result.is_err());
    match result.unwrap_err() {
        ForeignError::InvalidArgumentType { method, expected, .. } => {
            assert_eq!(method, "GetRow");
            assert_eq!(expected, "Integer");
        }
        _ => panic!("Expected InvalidArgumentType error"),
    }
}

// ==========================================
// Integration Tests
// ==========================================

#[test]
fn test_foreign_table_value_integration() {
    let table = create_test_table();
    let table_value = Value::LyObj(LyObj::new(Box::new(table)));
    
    // Test that it can be cloned
    let cloned_value = table_value.clone();
    assert_eq!(table_value, cloned_value);
    
    // Test that it can be pattern matched
    match table_value {
        Value::LyObj(obj) => {
            assert_eq!(obj.type_name(), "Table");
        }
        _ => panic!("Expected LyObj variant"),
    }
}

#[test]
fn test_foreign_table_nested_operations() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    // Test nested method calls: Head[table, 2] then RowCount on result
    let head_result = table_obj.call_method("Head", &[Value::Integer(2)]).unwrap();
    
    match head_result {
        Value::LyObj(head_table) => {
            let row_count = head_table.call_method("RowCount", &[]).unwrap();
            assert_eq!(row_count, Value::Integer(2));
            
            // Test getting a cell from the head table
            let cell_value = head_table.call_method("GetCell", &[Value::Integer(0), Value::Integer(0)]).unwrap();
            assert!(matches!(cell_value, Value::Integer(_) | Value::String(_)));
        }
        _ => panic!("Expected LyObj result from Head method"),
    }
}

// ==========================================
// Performance Tests
// ==========================================

#[test]
fn test_foreign_table_method_call_performance() {
    // Create a larger table for performance testing
    let mut columns = HashMap::new();
    
    let ids: Vec<TestValue> = (0..1000).map(|i| TestValue::Integer(i)).collect();
    let names: Vec<TestValue> = (0..1000).map(|i| TestValue::String(format!("name{}", i))).collect();
    
    columns.insert("id".to_string(), ForeignSeries::new(ids));
    columns.insert("name".to_string(), ForeignSeries::new(names));
    
    let table = ForeignTable::from_columns(columns).unwrap();
    let table_obj = LyObj::new(Box::new(table));
    
    let start = std::time::Instant::now();
    
    // Perform many method calls
    for i in 0..100 {
        let _result = table_obj.call_method("GetCell", &[
            Value::Integer(i % 1000),
            Value::Integer(0)
        ]).unwrap();
    }
    
    let duration = start.elapsed();
    
    // Should complete 100 method calls quickly (< 10ms)
    assert!(duration.as_millis() < 10, "Table method calls too slow: {:?}", duration);
}

// ==========================================
// Edge Cases and Boundary Tests
// ==========================================

#[test]
fn test_foreign_table_empty_operations() {
    let empty_table = ForeignTable::new();
    let table_obj = LyObj::new(Box::new(empty_table));
    
    // Test operations on empty table
    let result = table_obj.call_method("RowCount", &[]).unwrap();
    assert_eq!(result, Value::Integer(0));
    
    let result = table_obj.call_method("ColumnCount", &[]).unwrap();
    assert_eq!(result, Value::Integer(0));
    
    let result = table_obj.call_method("IsEmpty", &[]).unwrap();
    assert_eq!(result, Value::Boolean(true));
    
    // Operations that should error on empty table
    let result = table_obj.call_method("GetCell", &[Value::Integer(0), Value::Integer(0)]);
    assert!(result.is_err());
    
    let result = table_obj.call_method("GetRow", &[Value::Integer(0)]);
    assert!(result.is_err());
}

#[test]
fn test_foreign_table_single_row() {
    let column_names = vec!["single".to_string()];
    let rows = vec![vec![TestValue::Integer(42)]];
    
    let table = ForeignTable::from_rows(column_names, rows).unwrap();
    let table_obj = LyObj::new(Box::new(table));
    
    let result = table_obj.call_method("RowCount", &[]).unwrap();
    assert_eq!(result, Value::Integer(1));
    
    let result = table_obj.call_method("GetCell", &[Value::Integer(0), Value::Integer(0)]).unwrap();
    assert_eq!(result, Value::Integer(42));
}

#[test]
fn test_foreign_table_type_safety() {
    let table = create_test_table();
    let table_obj = LyObj::new(Box::new(table));
    
    // Verify type name
    assert_eq!(table_obj.type_name(), "Table");
    
    // Verify safe downcasting
    let foreign_ref = table_obj.as_foreign();
    let table_ref = foreign_ref.as_any().downcast_ref::<ForeignTable>();
    assert!(table_ref.is_some());
    
    let table_ref = table_ref.unwrap();
    assert_eq!(table_ref.length, 3);
    assert_eq!(table_ref.columns.len(), 3);
}

// ==========================================
// Constructor Function Tests (for future implementation)
// ==========================================

#[test]
fn test_foreign_table_construction_patterns() {
    // Test various construction patterns that will be used by stdlib functions
    
    // From columns
    let mut columns = HashMap::new();
    columns.insert("test".to_string(), ForeignSeries::new(
        vec![TestValue::Integer(1), TestValue::Integer(2)]
    ));
    
    let table = ForeignTable::from_columns(columns).unwrap();
    assert_eq!(table.length, 2);
    
    // From rows
    let table = ForeignTable::from_rows(
        vec!["x".to_string(), "y".to_string()],
        vec![
            vec![TestValue::Integer(1), TestValue::Integer(2)],
            vec![TestValue::Integer(3), TestValue::Integer(4)],
        ]
    ).unwrap();
    assert_eq!(table.length, 2);
    assert_eq!(table.columns.len(), 2);
}