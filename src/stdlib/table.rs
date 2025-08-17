use crate::vm::{Value, Table, Series, VmError, VmResult};
use std::collections::HashMap;

/// GroupBy - handles grouped operations on tables
#[derive(Debug, Clone)]
pub struct GroupBy {
    pub table: Table,
    pub group_columns: Vec<String>,
    pub groups: HashMap<Vec<Value>, Vec<usize>>, // group_key -> row_indices
}

impl GroupBy {
    /// Create a new GroupBy from a table and grouping columns
    pub fn new(table: Table, group_columns: Vec<String>) -> VmResult<Self> {
        // Validate that all group columns exist
        for col_name in &group_columns {
            if !table.columns.contains_key(col_name) {
                return Err(VmError::TypeError {
                    expected: format!("column '{}' to exist", col_name),
                    actual: format!("column not found in table with columns: {:?}", 
                                  table.column_names()),
                });
            }
        }
        
        // Build groups using hash-based algorithm
        let mut groups: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
        
        for row_idx in 0..table.length {
            // Extract group key values for this row
            let mut group_key = Vec::with_capacity(group_columns.len());
            for col_name in &group_columns {
                if let Some(series) = table.columns.get(col_name) {
                    group_key.push(series.get(row_idx)?.clone());
                }
            }
            
            // Add row index to the appropriate group
            groups.entry(group_key).or_insert_with(Vec::new).push(row_idx);
        }
        
        Ok(GroupBy {
            table,
            group_columns,
            groups,
        })
    }
    
    /// Get the number of groups
    pub fn group_count(&self) -> usize {
        self.groups.len()
    }
    
    /// Get group keys
    pub fn group_keys(&self) -> Vec<&Vec<Value>> {
        self.groups.keys().collect()
    }
    
    /// Get row indices for a specific group
    pub fn get_group(&self, group_key: &[Value]) -> Option<&Vec<usize>> {
        self.groups.get(group_key)
    }
    
    /// Apply aggregation function to all groups
    pub fn agg(&self, aggregations: HashMap<String, AggregationFunction>) -> VmResult<Table> {
        let mut result_columns: HashMap<String, Series> = HashMap::new();
        
        // Add group columns to result
        for (i, group_col) in self.group_columns.iter().enumerate() {
            let mut group_values = Vec::with_capacity(self.groups.len());
            for group_key in self.groups.keys() {
                if i < group_key.len() {
                    group_values.push(group_key[i].clone());
                }
            }
            
            let group_series = Series::infer(group_values)?;
            result_columns.insert(group_col.clone(), group_series);
        }
        
        // Apply aggregations to each specified column
        for (col_name, agg_func) in aggregations {
            if !self.table.columns.contains_key(&col_name) {
                return Err(VmError::TypeError {
                    expected: format!("column '{}' to exist", col_name),
                    actual: format!("column not found in table"),
                });
            }
            
            let mut agg_values = Vec::with_capacity(self.groups.len());
            let source_series = self.table.columns.get(&col_name).unwrap();
            
            for group_indices in self.groups.values() {
                let group_result = self.apply_aggregation(source_series, group_indices, &agg_func)?;
                agg_values.push(group_result);
            }
            
            let result_col_name = format!("{}_{}", col_name, agg_func.name());
            let agg_series = Series::infer(agg_values)?;
            result_columns.insert(result_col_name, agg_series);
        }
        
        Table::from_columns(result_columns)
    }
    
    /// Apply a single aggregation function to a group
    fn apply_aggregation(&self, series: &Series, indices: &[usize], func: &AggregationFunction) -> VmResult<Value> {
        match func {
            AggregationFunction::Count => {
                Ok(Value::Integer(indices.len() as i64))
            },
            AggregationFunction::Sum => {
                let mut sum = 0.0;
                let mut has_values = false;
                
                for &idx in indices {
                    let value = series.get(idx)?;
                    match value {
                        Value::Integer(n) => {
                            sum += *n as f64;
                            has_values = true;
                        },
                        Value::Real(f) => {
                            sum += f;
                            has_values = true;
                        },
                        Value::Missing => {}, // Skip missing values
                        _ => return Err(VmError::TypeError {
                            expected: "numeric value".to_string(),
                            actual: format!("{:?}", value),
                        }),
                    }
                }
                
                if has_values {
                    Ok(Value::Real(sum))
                } else {
                    Ok(Value::Missing)
                }
            },
            AggregationFunction::Mean => {
                let mut sum = 0.0;
                let mut count = 0;
                
                for &idx in indices {
                    let value = series.get(idx)?;
                    match value {
                        Value::Integer(n) => {
                            sum += *n as f64;
                            count += 1;
                        },
                        Value::Real(f) => {
                            sum += f;
                            count += 1;
                        },
                        Value::Missing => {}, // Skip missing values
                        _ => return Err(VmError::TypeError {
                            expected: "numeric value".to_string(),
                            actual: format!("{:?}", value),
                        }),
                    }
                }
                
                if count > 0 {
                    Ok(Value::Real(sum / count as f64))
                } else {
                    Ok(Value::Missing)
                }
            },
            AggregationFunction::Min => {
                let mut min_val: Option<Value> = None;
                
                for &idx in indices {
                    let value = series.get(idx)?;
                    if !matches!(value, Value::Missing) {
                        match &min_val {
                            None => min_val = Some(value.clone()),
                            Some(current_min) => {
                                if Self::compare_values(value, current_min)? < 0 {
                                    min_val = Some(value.clone());
                                }
                            }
                        }
                    }
                }
                
                Ok(min_val.unwrap_or(Value::Missing))
            },
            AggregationFunction::Max => {
                let mut max_val: Option<Value> = None;
                
                for &idx in indices {
                    let value = series.get(idx)?;
                    if !matches!(value, Value::Missing) {
                        match &max_val {
                            None => max_val = Some(value.clone()),
                            Some(current_max) => {
                                if Self::compare_values(value, current_max)? > 0 {
                                    max_val = Some(value.clone());
                                }
                            }
                        }
                    }
                }
                
                Ok(max_val.unwrap_or(Value::Missing))
            },
            AggregationFunction::First => {
                if indices.is_empty() {
                    Ok(Value::Missing)
                } else {
                    Ok(series.get(indices[0])?.clone())
                }
            },
            AggregationFunction::Last => {
                if indices.is_empty() {
                    Ok(Value::Missing)
                } else {
                    Ok(series.get(indices[indices.len() - 1])?.clone())
                }
            },
        }
    }
    
    /// Compare two values for ordering (used in Min/Max)
    fn compare_values(a: &Value, b: &Value) -> VmResult<i32> {
        match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => Ok(a.cmp(b) as i32),
            (Value::Integer(a), Value::Real(b)) => Ok((*a as f64).partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (Value::Real(a), Value::Integer(b)) => Ok(a.partial_cmp(&(*b as f64)).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (Value::Real(a), Value::Real(b)) => Ok(a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) as i32),
            (Value::String(a), Value::String(b)) => Ok(a.cmp(b) as i32),
            (Value::Boolean(a), Value::Boolean(b)) => Ok(a.cmp(b) as i32),
            _ => Err(VmError::TypeError {
                expected: "comparable values".to_string(),
                actual: format!("cannot compare {:?} and {:?}", a, b),
            }),
        }
    }
}

/// Aggregation functions for GroupBy operations
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationFunction {
    Count,
    Sum,
    Mean,
    Min,
    Max,
    First,
    Last,
}

impl AggregationFunction {
    pub fn name(&self) -> &'static str {
        match self {
            AggregationFunction::Count => "count",
            AggregationFunction::Sum => "sum",
            AggregationFunction::Mean => "mean",
            AggregationFunction::Min => "min",
            AggregationFunction::Max => "max",
            AggregationFunction::First => "first",
            AggregationFunction::Last => "last",
        }
    }
}

/// Standard library functions for table operations

/// GroupBy function - create grouped table for aggregation
pub fn group_by(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::TypeError {
            expected: "at least 2 arguments (table, group_columns...)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table = match &args[0] {
        Value::Table(t) => t.clone(),
        _ => return Err(VmError::TypeError {
            expected: "Table".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let mut group_columns = Vec::new();
    for arg in &args[1..] {
        match arg {
            Value::String(col_name) => group_columns.push(col_name.clone()),
            Value::Symbol(col_name) => group_columns.push(col_name.clone()),
            _ => return Err(VmError::TypeError {
                expected: "String or Symbol column name".to_string(),
                actual: format!("{:?}", arg),
            }),
        }
    }
    
    let groupby = GroupBy::new(table, group_columns)?;
    
    // For now, return a placeholder - we'll need to enhance the VM to support GroupBy values
    // or immediately perform aggregation
    Ok(Value::String(format!("GroupBy[groups: {}]", groupby.group_count())))
}

/// Aggregation function - apply aggregations to a table or grouped table
pub fn aggregate(args: &[Value]) -> VmResult<Value> {
    if args.len() < 2 {
        return Err(VmError::TypeError {
            expected: "at least 2 arguments (table, agg_spec...)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table = match &args[0] {
        Value::Table(t) => t.clone(),
        _ => return Err(VmError::TypeError {
            expected: "Table".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Simple aggregation without grouping for now
    let mut aggregations = HashMap::new();
    
    // Parse aggregation specifications from remaining args
    // For now, assume format: Agg[column, function]
    for arg in &args[1..] {
        match arg {
            Value::List(agg_spec) if agg_spec.len() == 2 => {
                let column = match &agg_spec[0] {
                    Value::String(s) => s.clone(),
                    Value::Symbol(s) => s.clone(),
                    _ => return Err(VmError::TypeError {
                        expected: "column name".to_string(),
                        actual: format!("{:?}", agg_spec[0]),
                    }),
                };
                
                let function = match &agg_spec[1] {
                    Value::Symbol(f) => match f.as_str() {
                        "count" | "Count" => AggregationFunction::Count,
                        "sum" | "Sum" => AggregationFunction::Sum,
                        "mean" | "Mean" => AggregationFunction::Mean,
                        "min" | "Min" => AggregationFunction::Min,
                        "max" | "Max" => AggregationFunction::Max,
                        "first" | "First" => AggregationFunction::First,
                        "last" | "Last" => AggregationFunction::Last,
                        _ => return Err(VmError::TypeError {
                            expected: "valid aggregation function".to_string(),
                            actual: f.clone(),
                        }),
                    },
                    _ => return Err(VmError::TypeError {
                        expected: "aggregation function name".to_string(),
                        actual: format!("{:?}", agg_spec[1]),
                    }),
                };
                
                aggregations.insert(column, function);
            },
            _ => return Err(VmError::TypeError {
                expected: "aggregation specification {column, function}".to_string(),
                actual: format!("{:?}", arg),
            }),
        }
    }
    
    // For simple table aggregation, create a single-group GroupBy
    let dummy_groupby = GroupBy {
        table: table.clone(),
        group_columns: vec![],
        groups: {
            let mut groups = HashMap::new();
            groups.insert(vec![], (0..table.length).collect());
            groups
        },
    };
    
    let result = dummy_groupby.agg(aggregations)?;
    Ok(Value::Table(result))
}

/// Count function - count non-missing values in a column
pub fn count(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (table, column)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let table = match &args[0] {
        Value::Table(t) => t,
        _ => return Err(VmError::TypeError {
            expected: "Table".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    let column_name = match &args[1] {
        Value::String(s) => s,
        Value::Symbol(s) => s,
        _ => return Err(VmError::TypeError {
            expected: "String or Symbol column name".to_string(),
            actual: format!("{:?}", args[1]),
        }),
    };
    
    let series = table.get_column(column_name)
        .ok_or_else(|| VmError::TypeError {
            expected: format!("column '{}' to exist", column_name),
            actual: "column not found".to_string(),
        })?;
    
    let mut count = 0;
    for i in 0..series.length {
        let value = series.get(i)?;
        if !matches!(value, Value::Missing) {
            count += 1;
        }
    }
    
    Ok(Value::Integer(count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vm::{SeriesType};

    #[test]
    fn test_groupby_creation() {
        let mut columns = HashMap::new();
        columns.insert("group".to_string(), Series::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), Series::new(
            vec![Value::Integer(10), Value::Integer(20), Value::Integer(15)],
            SeriesType::Int64
        ).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let groupby = GroupBy::new(table, vec!["group".to_string()]).unwrap();
        
        assert_eq!(groupby.group_count(), 2);
        assert!(groupby.get_group(&[Value::String("A".to_string())]).is_some());
        assert!(groupby.get_group(&[Value::String("B".to_string())]).is_some());
    }

    #[test]
    fn test_groupby_aggregation() {
        let mut columns = HashMap::new();
        columns.insert("group".to_string(), Series::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), Series::new(
            vec![Value::Integer(10), Value::Integer(20), Value::Integer(15)],
            SeriesType::Int64
        ).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let groupby = GroupBy::new(table, vec!["group".to_string()]).unwrap();
        
        let mut aggregations = HashMap::new();
        aggregations.insert("value".to_string(), AggregationFunction::Sum);
        
        let result = groupby.agg(aggregations).unwrap();
        assert_eq!(result.length, 2);
        assert!(result.get_column("group").is_some());
        assert!(result.get_column("value_sum").is_some());
    }

    #[test]
    fn test_aggregation_with_missing_values() {
        let mut columns = HashMap::new();
        columns.insert("group".to_string(), Series::new(
            vec![Value::String("A".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), Series::new(
            vec![Value::Integer(10), Value::Missing],
            SeriesType::Int64
        ).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let groupby = GroupBy::new(table, vec!["group".to_string()]).unwrap();
        
        let mut aggregations = HashMap::new();
        aggregations.insert("value".to_string(), AggregationFunction::Mean);
        
        let result = groupby.agg(aggregations).unwrap();
        let mean_col = result.get_column("value_mean").unwrap();
        
        // Should be 10.0 (missing values excluded)
        assert_eq!(*mean_col.get(0).unwrap(), Value::Real(10.0));
    }

    #[test]
    fn test_count_function() {
        let mut columns = HashMap::new();
        columns.insert("test".to_string(), Series::new(
            vec![Value::Integer(1), Value::Missing, Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        
        let table = Table::from_columns(columns).unwrap();
        let args = vec![
            Value::Table(table),
            Value::String("test".to_string()),
        ];
        
        let result = count(&args).unwrap();
        assert_eq!(result, Value::Integer(2)); // 2 non-missing values
    }
}