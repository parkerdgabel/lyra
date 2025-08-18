use crate::{
    foreign::{Foreign, ForeignError, LyObj},
    stdlib::data::{ForeignSeries, SeriesType},
    vm::{Value, VmError, VmResult},
};
use std::any::Any;
use std::collections::HashMap;

/// Foreign Table implementation that replaces Value::Table
/// This struct mirrors the existing Table but is Send + Sync for Foreign objects
#[derive(Debug, Clone, PartialEq)]
pub struct ForeignTable {
    pub columns: HashMap<String, ForeignSeries>,
    pub length: usize,
    pub index: Option<Vec<String>>, // Simplified index for now to avoid Value threading issues
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
    pub fn from_rows(column_names: Vec<String>, rows: Vec<Vec<Value>>) -> VmResult<Self> {
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
            
            // Infer column type from data
            let series = ForeignSeries::infer(column_data).map_err(|_| VmError::TypeError {
                expected: "valid series data".to_string(),
                actual: "invalid column data".to_string(),
            })?;
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
        
        let new_index = None; // Simplified for now
        
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
                        let value = series.get(row_idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Series access error: {}", e),
                        })?;
                        
                        Ok(value.clone())
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
                        let row_values = self.get_row(row_idx).map_err(|e| ForeignError::RuntimeError {
                            message: format!("Row access error: {}", e),
                        })?;
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
                                .map(|i| series.get(i).unwrap().clone())
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