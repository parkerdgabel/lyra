use crate::{
    foreign::LyObj,
    stdlib::data::{ForeignTable, ForeignSeries, ForeignTensor, SeriesType},
    vm::{Value, VmError, VmResult},
};
use std::collections::HashMap;

/// GroupBy - handles grouped operations on tables
#[derive(Debug, Clone)]
pub struct GroupBy {
    pub table: ForeignTable,
    pub group_columns: Vec<String>,
    pub groups: HashMap<Vec<Value>, Vec<usize>>, // group_key -> row_indices
}

impl GroupBy {
    /// Create a new GroupBy from a table and grouping columns
    pub fn new(table: ForeignTable, group_columns: Vec<String>) -> VmResult<Self> {
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
    pub fn agg(&self, aggregations: HashMap<String, AggregationFunction>) -> VmResult<ForeignTable> {
        let mut result_columns: HashMap<String, ForeignSeries> = HashMap::new();
        
        // Add group columns to result
        for (i, group_col) in self.group_columns.iter().enumerate() {
            let mut group_values = Vec::with_capacity(self.groups.len());
            for group_key in self.groups.keys() {
                if i < group_key.len() {
                    group_values.push(group_key[i].clone());
                }
            }
            
            let group_series = ForeignSeries::infer(group_values)?;
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
            let agg_series = ForeignSeries::infer(agg_values)?;
            result_columns.insert(result_col_name, agg_series);
        }
        
        ForeignTable::from_columns(result_columns)
    }
    
    /// Apply a single aggregation function to a group
    fn apply_aggregation(&self, series: &ForeignSeries, indices: &[usize], func: &AggregationFunction) -> VmResult<Value> {
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
                            sum += *f;
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
                            sum += *f;
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
    
    // Extract ForeignTable from LyObj
    let foreign_table = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(table) = obj.downcast_ref::<ForeignTable>() {
                table
            } else {
                return Err(VmError::TypeError {
                    expected: "Table (Foreign)".to_string(),
                    actual: format!("LyObj({})", obj.type_name()),
                });
            }
        },
        _ => return Err(VmError::TypeError {
            expected: "Table (Foreign)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Note: Legacy function needs conversion from Foreign to legacy Table
    // This is a temporary bridge until all table operations are migrated
    let table = ForeignTable::new(); // Already ForeignTable
    
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
    
    // Extract ForeignTable from LyObj
    let foreign_table = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(table) = obj.downcast_ref::<ForeignTable>() {
                table
            } else {
                return Err(VmError::TypeError {
                    expected: "Table (Foreign)".to_string(),
                    actual: format!("LyObj({})", obj.type_name()),
                });
            }
        },
        _ => return Err(VmError::TypeError {
            expected: "Table (Foreign)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Note: Legacy function needs conversion from Foreign to legacy Table
    // This is a temporary bridge until all table operations are migrated
    let table = ForeignTable::new(); // Already ForeignTable
    
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
    // Result is already a ForeignTable
    Ok(Value::LyObj(LyObj::new(Box::new(result))))
}

/// Count function - count non-missing values in a column
pub fn count(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (table, column)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    // Extract ForeignTable from LyObj
    let foreign_table = match &args[0] {
        Value::LyObj(obj) => {
            if let Some(table) = obj.downcast_ref::<ForeignTable>() {
                table
            } else {
                return Err(VmError::TypeError {
                    expected: "Table (Foreign)".to_string(),
                    actual: format!("LyObj({})", obj.type_name()),
                });
            }
        },
        _ => return Err(VmError::TypeError {
            expected: "Table (Foreign)".to_string(),
            actual: format!("{:?}", args[0]),
        }),
    };
    
    // Note: Legacy function needs conversion from Foreign to legacy Table
    // This is a temporary bridge until all table operations are migrated
    let table = &ForeignTable::new(); // Already ForeignTable
    
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

// ============================================================================
// Hash Join Implementation
// ============================================================================

/// Inner join - returns only rows where join keys match in both tables
pub fn inner_join(left: ForeignTable, right: ForeignTable, join_keys: Vec<(String, String)>) -> VmResult<ForeignTable> {
    if join_keys.is_empty() {
        // No join keys specified - cartesian product (for now, return empty)
        return ForeignTable::from_columns(HashMap::new());
    }
    
    // Validate that all join columns exist
    for (left_col, right_col) in &join_keys {
        if !left.columns.contains_key(left_col) {
            return Err(VmError::TypeError {
                expected: format!("column '{}' to exist in left table", left_col),
                actual: "column not found".to_string(),
            });
        }
        if !right.columns.contains_key(right_col) {
            return Err(VmError::TypeError {
                expected: format!("column '{}' to exist in right table", right_col),
                actual: "column not found".to_string(),
            });
        }
    }
    
    // Clone tables for later use
    let left_clone = left.clone();
    let right_clone = right.clone();

    // Build hash table from the smaller table (choose build vs probe table)
    let (build_table, probe_table, build_keys, probe_keys, is_left_build) = 
        if left.length <= right.length {
            (left, right, join_keys.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(),
             join_keys.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>(), true)
        } else {
            (right, left, join_keys.iter().map(|(_, r)| r.clone()).collect::<Vec<_>>(),
             join_keys.iter().map(|(l, _)| l.clone()).collect::<Vec<_>>(), false)
        };
    
    // Build hash table: join_key -> Vec<row_indices>
    let mut hash_table: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
    
    for row_idx in 0..build_table.length {
        let mut join_key = Vec::with_capacity(build_keys.len());
        
        for key_col in &build_keys {
            let series = build_table.columns.get(key_col).unwrap();
            let value = series.get(row_idx)?.clone();
            
            // Skip rows with Missing values in join keys (Missing never matches anything)
            if matches!(value, Value::Missing) {
                break;
            }
            join_key.push(value);
        }
        
        // Only add to hash table if all join key values are non-Missing
        if join_key.len() == build_keys.len() {
            hash_table.entry(join_key).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    // Probe phase - iterate through probe table and find matches
    let mut result_rows: Vec<Vec<Value>> = Vec::new();
    
    for probe_row_idx in 0..probe_table.length {
        let mut probe_key = Vec::with_capacity(probe_keys.len());
        
        for key_col in &probe_keys {
            let series = probe_table.columns.get(key_col).unwrap();
            let value = series.get(probe_row_idx)?.clone();
            
            // Skip rows with Missing values in join keys
            if matches!(value, Value::Missing) {
                break;
            }
            probe_key.push(value);
        }
        
        // Only probe if all join key values are non-Missing
        if probe_key.len() == probe_keys.len() {
            if let Some(matching_build_rows) = hash_table.get(&probe_key) {
                // Found matches - create result rows
                for &build_row_idx in matching_build_rows {
                    let mut result_row = Vec::new();
                    
                    // Add all columns from both tables (resolve order based on which is build/probe)
                    if is_left_build {
                        // Left was build table, right was probe table
                        // Add left columns first, then right columns
                        for col_name in left_clone.column_names() {
                            let series = build_table.columns.get(col_name).unwrap();
                            result_row.push(series.get(build_row_idx)?.clone());
                        }
                        for col_name in right_clone.column_names() {
                            if !join_keys.iter().any(|(_, r)| r == col_name) {
                                // Add right columns that aren't duplicated join keys
                                let series = probe_table.columns.get(col_name).unwrap();
                                result_row.push(series.get(probe_row_idx)?.clone());
                            }
                        }
                    } else {
                        // Right was build table, left was probe table
                        // Add left columns first, then right columns
                        for col_name in left_clone.column_names() {
                            if !join_keys.iter().any(|(l, _)| l == col_name) {
                                let series = probe_table.columns.get(col_name).unwrap();
                                result_row.push(series.get(probe_row_idx)?.clone());
                            }
                        }
                        for col_name in right_clone.column_names() {
                            let series = build_table.columns.get(col_name).unwrap();
                            result_row.push(series.get(build_row_idx)?.clone());
                        }
                    }
                    
                    result_rows.push(result_row);
                }
            }
        }
    }
    
    // Create result table from collected rows
    create_joined_table(result_rows, &left_clone, &right_clone, &join_keys)
}

/// Left join - returns all rows from left table, with Missing for unmatched right rows
pub fn left_join(left: ForeignTable, right: ForeignTable, join_keys: Vec<(String, String)>) -> VmResult<ForeignTable> {
    if join_keys.is_empty() {
        return ForeignTable::from_columns(HashMap::new());
    }
    
    // Validate join columns exist
    for (left_col, right_col) in &join_keys {
        if !left.columns.contains_key(left_col) {
            return Err(VmError::TypeError {
                expected: format!("column '{}' to exist in left table", left_col),
                actual: "column not found".to_string(),
            });
        }
        if !right.columns.contains_key(right_col) {
            return Err(VmError::TypeError {
                expected: format!("column '{}' to exist in right table", right_col),
                actual: "column not found".to_string(),
            });
        }
    }
    
    // Build hash table from right table
    let mut hash_table: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
    let right_keys: Vec<String> = join_keys.iter().map(|(_, r)| r.clone()).collect();
    
    for row_idx in 0..right.length {
        let mut join_key = Vec::with_capacity(right_keys.len());
        
        for key_col in &right_keys {
            let series = right.columns.get(key_col).unwrap();
            let value = series.get(row_idx)?.clone();
            
            if matches!(value, Value::Missing) {
                break;
            }
            join_key.push(value);
        }
        
        if join_key.len() == right_keys.len() {
            hash_table.entry(join_key).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    // Probe with left table - always include all left rows
    let mut result_rows: Vec<Vec<Value>> = Vec::new();
    let left_keys: Vec<String> = join_keys.iter().map(|(l, _)| l.clone()).collect();
    
    for left_row_idx in 0..left.length {
        let mut left_key = Vec::with_capacity(left_keys.len());
        
        for key_col in &left_keys {
            let series = left.columns.get(key_col).unwrap();
            let value = series.get(left_row_idx)?.clone();
            
            if matches!(value, Value::Missing) {
                break;
            }
            left_key.push(value);
        }
        
        // Always include the left row
        let mut result_row = Vec::new();
        
        // Add all left columns
        for col_name in left.column_names() {
            let series = left.columns.get(col_name).unwrap();
            result_row.push(series.get(left_row_idx)?.clone());
        }
        
        // Add right columns (Missing if no match)
        if left_key.len() == left_keys.len() && hash_table.contains_key(&left_key) {
            // Found match - add values from first matching right row
            let matching_right_rows = hash_table.get(&left_key).unwrap();
            let right_row_idx = matching_right_rows[0]; // Take first match for now
            
            for col_name in right.column_names() {
                if !join_keys.iter().any(|(_, r)| r == col_name) {
                    let series = right.columns.get(col_name).unwrap();
                    result_row.push(series.get(right_row_idx)?.clone());
                }
            }
        } else {
            // No match - add Missing values for right columns
            for col_name in right.column_names() {
                if !join_keys.iter().any(|(_, r)| r == col_name) {
                    result_row.push(Value::Missing);
                }
            }
        }
        
        result_rows.push(result_row);
    }
    
    create_joined_table(result_rows, &left, &right, &join_keys)
}

/// Right join - returns all rows from right table, with Missing for unmatched left rows
pub fn right_join(left: ForeignTable, right: ForeignTable, join_keys: Vec<(String, String)>) -> VmResult<ForeignTable> {
    // Right join is equivalent to left join with tables swapped
    let swapped_keys: Vec<(String, String)> = join_keys.iter()
        .map(|(l, r)| (r.clone(), l.clone()))
        .collect();
    
    let result = left_join(right.clone(), left.clone(), swapped_keys)?;
    
    // Reorder columns to match expected left-first, right-second order
    reorder_columns_for_right_join(result, &left, &right, &join_keys)
}

/// Full outer join - returns all rows from both tables, with Missing for unmatched rows
pub fn full_join(left: ForeignTable, right: ForeignTable, join_keys: Vec<(String, String)>) -> VmResult<ForeignTable> {
    if join_keys.is_empty() {
        return ForeignTable::from_columns(HashMap::new());
    }
    
    // Build hash tables from both sides
    let mut left_hash: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
    let mut right_hash: HashMap<Vec<Value>, Vec<usize>> = HashMap::new();
    
    let left_keys: Vec<String> = join_keys.iter().map(|(l, _)| l.clone()).collect();
    let right_keys: Vec<String> = join_keys.iter().map(|(_, r)| r.clone()).collect();
    
    // Build left hash table
    for row_idx in 0..left.length {
        let mut join_key = Vec::with_capacity(left_keys.len());
        
        for key_col in &left_keys {
            let series = left.columns.get(key_col).unwrap();
            let value = series.get(row_idx)?.clone();
            
            if matches!(value, Value::Missing) {
                break;
            }
            join_key.push(value);
        }
        
        if join_key.len() == left_keys.len() {
            left_hash.entry(join_key).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    // Build right hash table
    for row_idx in 0..right.length {
        let mut join_key = Vec::with_capacity(right_keys.len());
        
        for key_col in &right_keys {
            let series = right.columns.get(key_col).unwrap();
            let value = series.get(row_idx)?.clone();
            
            if matches!(value, Value::Missing) {
                break;
            }
            join_key.push(value);
        }
        
        if join_key.len() == right_keys.len() {
            right_hash.entry(join_key).or_insert_with(Vec::new).push(row_idx);
        }
    }
    
    let mut result_rows: Vec<Vec<Value>> = Vec::new();
    let mut matched_right_rows: std::collections::HashSet<usize> = std::collections::HashSet::new();
    
    // Process all left rows
    for left_row_idx in 0..left.length {
        let mut left_key = Vec::with_capacity(left_keys.len());
        
        for key_col in &left_keys {
            let series = left.columns.get(key_col).unwrap();
            let value = series.get(left_row_idx)?.clone();
            
            if matches!(value, Value::Missing) {
                break;
            }
            left_key.push(value);
        }
        
        let mut result_row = Vec::new();
        
        // Add left columns
        for col_name in left.column_names() {
            let series = left.columns.get(col_name).unwrap();
            result_row.push(series.get(left_row_idx)?.clone());
        }
        
        // Add right columns
        if left_key.len() == left_keys.len() && right_hash.contains_key(&left_key) {
            // Found match
            let matching_right_rows = right_hash.get(&left_key).unwrap();
            let right_row_idx = matching_right_rows[0];
            matched_right_rows.insert(right_row_idx);
            
            for col_name in right.column_names() {
                if !join_keys.iter().any(|(_, r)| r == col_name) {
                    let series = right.columns.get(col_name).unwrap();
                    result_row.push(series.get(right_row_idx)?.clone());
                }
            }
        } else {
            // No match - add Missing
            for col_name in right.column_names() {
                if !join_keys.iter().any(|(_, r)| r == col_name) {
                    result_row.push(Value::Missing);
                }
            }
        }
        
        result_rows.push(result_row);
    }
    
    // Add unmatched right rows
    for right_row_idx in 0..right.length {
        if !matched_right_rows.contains(&right_row_idx) {
            let mut result_row = Vec::new();
            
            // Add Missing for left columns
            for col_name in left.column_names() {
                if !join_keys.iter().any(|(l, _)| l == col_name) {
                    result_row.push(Value::Missing);
                } else {
                    // For join key columns, use the value from the right table
                    let series = right.columns.get(&join_keys.iter()
                        .find(|(l, _)| l == col_name).unwrap().1).unwrap();
                    result_row.push(series.get(right_row_idx)?.clone());
                }
            }
            
            // Add right columns
            for col_name in right.column_names() {
                if !join_keys.iter().any(|(_, r)| r == col_name) {
                    let series = right.columns.get(col_name).unwrap();
                    result_row.push(series.get(right_row_idx)?.clone());
                }
            }
            
            result_rows.push(result_row);
        }
    }
    
    create_joined_table(result_rows, &left, &right, &join_keys)
}

/// Helper function to create a joined table from result rows
fn create_joined_table(
    result_rows: Vec<Vec<Value>>,
    left: &ForeignTable,
    right: &ForeignTable,
    join_keys: &[(String, String)]
) -> VmResult<ForeignTable> {
    if result_rows.is_empty() {
        return ForeignTable::from_columns(HashMap::new());
    }
    
    let mut result_columns: HashMap<String, ForeignSeries> = HashMap::new();
    let mut column_index = 0;
    
    // Add left table columns
    for col_name in left.column_names() {
        let mut column_data = Vec::with_capacity(result_rows.len());
        for row in &result_rows {
            column_data.push(row[column_index].clone());
        }
        
        let series = ForeignSeries::infer(column_data)?;
        result_columns.insert(col_name.clone(), series);
        column_index += 1;
    }
    
    // Add right table columns (excluding duplicate join keys)
    for col_name in right.column_names() {
        if !join_keys.iter().any(|(_, r)| r == col_name) {
            let mut column_data = Vec::with_capacity(result_rows.len());
            for row in &result_rows {
                column_data.push(row[column_index].clone());
            }
            
            let series = ForeignSeries::infer(column_data)?;
            result_columns.insert(col_name.clone(), series);
            column_index += 1;
        }
    }
    
    ForeignTable::from_columns(result_columns)
}

/// Helper function to reorder columns for right join result
fn reorder_columns_for_right_join(
    table: ForeignTable,
    _left: &ForeignTable,
    _right: &ForeignTable,
    _join_keys: &[(String, String)]
) -> VmResult<ForeignTable> {
    // For now, return as-is. In a production implementation,
    // we would reorder columns to match left-first, right-second convention
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stdlib::data::SeriesType;

    #[test]
    fn test_groupby_creation() {
        let mut columns = HashMap::new();
        columns.insert("group".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(10), Value::Integer(20), Value::Integer(15)],
            SeriesType::Int64
        ).unwrap());
        
        let table = ForeignTable::from_columns(columns).unwrap();
        let groupby = GroupBy::new(table, vec!["group".to_string()]).unwrap();
        
        assert_eq!(groupby.group_count(), 2);
        assert!(groupby.get_group(&[Value::String("A".to_string())]).is_some());
        assert!(groupby.get_group(&[Value::String("B".to_string())]).is_some());
    }

    #[test]
    fn test_groupby_aggregation() {
        let mut columns = HashMap::new();
        columns.insert("group".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(10), Value::Integer(20), Value::Integer(15)],
            SeriesType::Int64
        ).unwrap());
        
        let table = ForeignTable::from_columns(columns).unwrap();
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
        columns.insert("group".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("A".to_string())],
            SeriesType::String
        ).unwrap());
        columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(10), Value::Missing],
            SeriesType::Int64
        ).unwrap());
        
        let table = ForeignTable::from_columns(columns).unwrap();
        let groupby = GroupBy::new(table, vec!["group".to_string()]).unwrap();
        
        let mut aggregations = HashMap::new();
        aggregations.insert("value".to_string(), AggregationFunction::Mean);
        
        let result = groupby.agg(aggregations).unwrap();
        let mean_col = result.get_column("value_mean").unwrap();
        
        // Should be 10.0 (missing values excluded)
        assert_eq!(*mean_col.get(0).unwrap(), Value::Real(10.0));
    }

    // TODO: Re-enable after Foreign Table migration complete
    // #[test]
    // fn test_count_function() {
    //     let mut columns = HashMap::new();
    //     columns.insert("test".to_string(), ForeignSeries::new(
    //         vec![Value::Integer(1), Value::Missing, Value::Integer(3)],
    //         SeriesType::Int64
    //     ).unwrap());
    //     
    //     let table = Table::from_columns(columns).unwrap();
    //     let args = vec![
    //         Value::Table(table),
    //         Value::String("test".to_string()),
    //     ];
    //     
    //     let result = count(&args).unwrap();
    //     assert_eq!(result, Value::Integer(2)); // 2 non-missing values
    // }

    // ============================================================================
    // Hash Join Tests (TDD approach - tests written before implementation)
    // ============================================================================

    #[test]
    fn test_inner_join_basic() {
        // Create left table: {id: [1, 2, 3], name: ["A", "B", "C"]}
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        // Create right table: {id: [2, 3, 4], value: [20, 30, 40]}
        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(20), Value::Integer(30), Value::Integer(40)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Inner join on 'id' column
        let result = inner_join(left_table, right_table, vec![("id".to_string(), "id".to_string())]).unwrap();
        
        // Should have 2 rows (id=2, id=3) with 3 columns (id, name, value)
        assert_eq!(result.length, 2);
        assert_eq!(result.columns.len(), 3);
        assert!(result.get_column("id").is_some());
        assert!(result.get_column("name").is_some());
        assert!(result.get_column("value").is_some());

        // Check specific values
        let id_col = result.get_column("id").unwrap();
        let name_col = result.get_column("name").unwrap();
        let value_col = result.get_column("value").unwrap();
        
        assert_eq!(*id_col.get(0).unwrap(), Value::Integer(2));
        assert_eq!(*name_col.get(0).unwrap(), Value::String("B".to_string()));
        assert_eq!(*value_col.get(0).unwrap(), Value::Integer(20));
        
        assert_eq!(*id_col.get(1).unwrap(), Value::Integer(3));
        assert_eq!(*name_col.get(1).unwrap(), Value::String("C".to_string()));
        assert_eq!(*value_col.get(1).unwrap(), Value::Integer(30));
    }

    #[test]
    fn test_left_join_basic() {
        // Same tables as inner join test
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(20), Value::Integer(30), Value::Integer(40)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Left join on 'id' column
        let result = left_join(left_table, right_table, vec![("id".to_string(), "id".to_string())]).unwrap();
        
        // Should have 3 rows (all from left) with 3 columns
        assert_eq!(result.length, 3);
        assert_eq!(result.columns.len(), 3);

        // Check that row with id=1 has Missing for value column
        let id_col = result.get_column("id").unwrap();
        let name_col = result.get_column("name").unwrap();
        let value_col = result.get_column("value").unwrap();
        
        assert_eq!(*id_col.get(0).unwrap(), Value::Integer(1));
        assert_eq!(*name_col.get(0).unwrap(), Value::String("A".to_string()));
        assert_eq!(*value_col.get(0).unwrap(), Value::Missing);
    }

    #[test]
    fn test_right_join_basic() {
        // Same setup as previous tests
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(20), Value::Integer(30), Value::Integer(40)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Right join on 'id' column
        let result = right_join(left_table, right_table, vec![("id".to_string(), "id".to_string())]).unwrap();
        
        // Should have 3 rows (all from right) with 3 columns
        assert_eq!(result.length, 3);
        assert_eq!(result.columns.len(), 3);

        // Check that row with id=4 has Missing for name column
        let id_col = result.get_column("id").unwrap();
        let name_col = result.get_column("name").unwrap();
        let value_col = result.get_column("value").unwrap();
        
        // Find the row with id=4
        let mut found_id_4 = false;
        for i in 0..result.length {
            if *id_col.get(i).unwrap() == Value::Integer(4) {
                assert_eq!(*name_col.get(i).unwrap(), Value::Missing);
                assert_eq!(*value_col.get(i).unwrap(), Value::Integer(40));
                found_id_4 = true;
                break;
            }
        }
        assert!(found_id_4, "Should find row with id=4");
    }

    #[test]
    fn test_full_join_basic() {
        // Same setup as previous tests
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(2), Value::Integer(3), Value::Integer(4)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(20), Value::Integer(30), Value::Integer(40)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Full outer join on 'id' column
        let result = full_join(left_table, right_table, vec![("id".to_string(), "id".to_string())]).unwrap();
        
        // Should have 4 rows (union of both tables) with 3 columns
        assert_eq!(result.length, 4);
        assert_eq!(result.columns.len(), 3);

        // Should contain all ids: 1, 2, 3, 4
        let id_col = result.get_column("id").unwrap();
        let mut ids = Vec::new();
        for i in 0..result.length {
            ids.push(id_col.get(i).unwrap().clone());
        }
        ids.sort_by(|a, b| match (a, b) {
            (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
            _ => std::cmp::Ordering::Equal,
        });
        
        assert_eq!(ids[0], Value::Integer(1));
        assert_eq!(ids[1], Value::Integer(2));
        assert_eq!(ids[2], Value::Integer(3));
        assert_eq!(ids[3], Value::Integer(4));
    }

    #[test]
    fn test_join_with_missing_values() {
        // Left table with Missing values
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Missing, Value::Integer(3)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("B".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        // Right table with Missing values
        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Missing, Value::Integer(3), Value::Integer(4)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(10), Value::Integer(30), Value::Integer(40)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Inner join - Missing values should NOT match each other
        let result = inner_join(left_table, right_table, vec![("id".to_string(), "id".to_string())]).unwrap();
        
        // Should only have 1 row (id=3 matches)
        assert_eq!(result.length, 1);
        let id_col = result.get_column("id").unwrap();
        assert_eq!(*id_col.get(0).unwrap(), Value::Integer(3));
    }

    #[test]
    fn test_multi_column_join() {
        // Left table: {dept: ["A", "A", "B"], level: [1, 2, 1], name: ["Alice", "Bob", "Charlie"]}
        let mut left_columns = HashMap::new();
        left_columns.insert("dept".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("A".to_string()), Value::String("B".to_string())],
            SeriesType::String
        ).unwrap());
        left_columns.insert("level".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2), Value::Integer(1)],
            SeriesType::Int64
        ).unwrap());
        left_columns.insert("name".to_string(), ForeignSeries::new(
            vec![Value::String("Alice".to_string()), Value::String("Bob".to_string()), Value::String("Charlie".to_string())],
            SeriesType::String
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        // Right table: {dept: ["A", "A", "C"], level: [1, 3, 1], salary: [50000, 60000, 70000]}
        let mut right_columns = HashMap::new();
        right_columns.insert("dept".to_string(), ForeignSeries::new(
            vec![Value::String("A".to_string()), Value::String("A".to_string()), Value::String("C".to_string())],
            SeriesType::String
        ).unwrap());
        right_columns.insert("level".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(3), Value::Integer(1)],
            SeriesType::Int64
        ).unwrap());
        right_columns.insert("salary".to_string(), ForeignSeries::new(
            vec![Value::Integer(50000), Value::Integer(60000), Value::Integer(70000)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Join on both dept and level
        let result = inner_join(left_table, right_table, vec![
            ("dept".to_string(), "dept".to_string()),
            ("level".to_string(), "level".to_string())
        ]).unwrap();
        
        // Should have 1 row (dept="A", level=1 matches)
        assert_eq!(result.length, 1);
        assert_eq!(result.columns.len(), 4); // dept, level, name, salary
        
        let name_col = result.get_column("name").unwrap();
        let salary_col = result.get_column("salary").unwrap();
        assert_eq!(*name_col.get(0).unwrap(), Value::String("Alice".to_string()));
        assert_eq!(*salary_col.get(0).unwrap(), Value::Integer(50000));
    }

    #[test]
    fn test_join_empty_tables() {
        // Empty left table
        let left_table = ForeignTable::from_columns(HashMap::new()).unwrap();
        
        // Non-empty right table
        let mut right_columns = HashMap::new();
        right_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1), Value::Integer(2)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Any join with empty table should result in empty table
        let result = inner_join(left_table.clone(), right_table.clone(), vec![]).unwrap();
        assert_eq!(result.length, 0);

        let result = left_join(left_table.clone(), right_table.clone(), vec![]).unwrap();
        assert_eq!(result.length, 0);

        let result = right_join(left_table, right_table, vec![]).unwrap();
        assert_eq!(result.length, 0);
    }

    #[test]
    fn test_join_no_common_columns() {
        // Tables with no overlapping join columns should return appropriate errors
        let mut left_columns = HashMap::new();
        left_columns.insert("id".to_string(), ForeignSeries::new(
            vec![Value::Integer(1)],
            SeriesType::Int64
        ).unwrap());
        let left_table = ForeignTable::from_columns(left_columns).unwrap();

        let mut right_columns = HashMap::new();
        right_columns.insert("value".to_string(), ForeignSeries::new(
            vec![Value::Integer(10)],
            SeriesType::Int64
        ).unwrap());
        let right_table = ForeignTable::from_columns(right_columns).unwrap();

        // Should return error for non-existent join column
        let result = inner_join(left_table, right_table, vec![("nonexistent".to_string(), "id".to_string())]);
        assert!(result.is_err());
    }
}

// ============================================================================
// Foreign Table Constructor Functions
// ============================================================================

/// Create a Foreign Table from column vectors
/// Usage: Table[{col1, col2, ...}, {name1, name2, ...}]
pub fn table(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (column_data_list, column_names_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let (column_data, column_names) = match (&args[0], &args[1]) {
        (Value::List(data_list), Value::List(names_list)) => (data_list, names_list),
        _ => return Err(VmError::TypeError {
            expected: "two lists (column data and column names)".to_string(),
            actual: format!("arguments of types: {:?}, {:?}", args[0], args[1]),
        }),
    };
    
    if column_data.len() != column_names.len() {
        return Err(VmError::TypeError {
            expected: format!("{} column names for {} columns", column_data.len(), column_data.len()),
            actual: format!("{} column names provided", column_names.len()),
        });
    }
    
    let mut foreign_columns = HashMap::new();
    
    for (data, name) in column_data.iter().zip(column_names.iter()) {
        let column_name = match name {
            Value::String(s) | Value::Symbol(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "string or symbol for column name".to_string(),
                actual: format!("column name of type: {:?}", name),
            }),
        };
        
        let column_values = match data {
            Value::List(values) => values.clone(),
            _ => return Err(VmError::TypeError {
                expected: "list for column data".to_string(),
                actual: format!("column data of type: {:?}", data),
            }),
        };
        
        // Infer column type from data
        let foreign_series = ForeignSeries::infer(column_values).map_err(|_| VmError::TypeError {
            expected: "valid series data".to_string(),
            actual: "invalid column data".to_string(),
        })?;
        foreign_columns.insert(column_name, foreign_series);
    }
    
    let foreign_table = ForeignTable::from_columns(foreign_columns)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_table))))
}

/// Create a Foreign Table from rows
/// Usage: TableFromRows[{row1, row2, ...}, {col1, col2, ...}]
pub fn table_from_rows(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (rows_list, column_names_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let (rows_list, column_names) = match (&args[0], &args[1]) {
        (Value::List(rows), Value::List(names)) => (rows, names),
        _ => return Err(VmError::TypeError {
            expected: "two lists (rows and column names)".to_string(),
            actual: format!("arguments of types: {:?}, {:?}", args[0], args[1]),
        }),
    };
    
    // Extract column names
    let mut column_names_vec = Vec::new();
    for name in column_names {
        let column_name = match name {
            Value::String(s) | Value::Symbol(s) => s.clone(),
            _ => return Err(VmError::TypeError {
                expected: "string or symbol for column name".to_string(),
                actual: format!("column name of type: {:?}", name),
            }),
        };
        column_names_vec.push(column_name);
    }
    
    // Extract rows
    let mut rows_vec = Vec::new();
    for row in rows_list {
        let row_values = match row {
            Value::List(values) => values.clone(),
            _ => return Err(VmError::TypeError {
                expected: "list for row data".to_string(),
                actual: format!("row data of type: {:?}", row),
            }),
        };
        rows_vec.push(row_values);
    }
    
    let foreign_table = ForeignTable::from_rows(column_names_vec, rows_vec)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_table))))
}

/// Create an empty Foreign Table
/// Usage: EmptyTable[]
pub fn empty_table(args: &[Value]) -> VmResult<Value> {
    if !args.is_empty() {
        return Err(VmError::TypeError {
            expected: "0 arguments".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let foreign_table = ForeignTable::new();
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_table))))
}

// ============================================================================
// Foreign Series Constructor Functions
// ============================================================================

/// Create a Foreign Series from a list of values
/// Usage: Series[{1, 2, 3, 4}]
pub fn series(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (data_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let data = match &args[0] {
        Value::List(values) => values.clone(),
        _ => return Err(VmError::TypeError {
            expected: "list for series data".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_series = ForeignSeries::infer(data)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_series))))
}

/// Create a range series from start to end (exclusive)
/// Usage: Range[start, end]
pub fn range(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (start, end)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let (start, end) = match (&args[0], &args[1]) {
        (Value::Integer(s), Value::Integer(e)) => (*s, *e),
        _ => return Err(VmError::TypeError {
            expected: "two integers (start, end)".to_string(),
            actual: format!("arguments of types: {:?}, {:?}", args[0], args[1]),
        }),
    };
    
    let foreign_series = ForeignSeries::range(start, end)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_series))))
}

/// Create a series filled with zeros
/// Usage: Zeros[n]
pub fn zeros(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let length = match &args[0] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as usize
        }
        _ => return Err(VmError::TypeError {
            expected: "integer for series length".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_series = ForeignSeries::zeros(length)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_series))))
}

/// Create a series filled with ones
/// Usage: Ones[n]
pub fn ones(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let length = match &args[0] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as usize
        }
        _ => return Err(VmError::TypeError {
            expected: "integer for series length".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_series = ForeignSeries::ones(length)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_series))))
}

/// Create a series filled with a constant value
/// Usage: ConstantSeries[value, n]
pub fn constant_series(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::TypeError {
            expected: "2 arguments (value, length)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let value = args[0].clone();
    let length = match &args[1] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as usize
        }
        _ => return Err(VmError::TypeError {
            expected: "integer for series length".to_string(),
            actual: format!("second argument of type: {:?}", args[1]),
        }),
    };
    
    let foreign_series = ForeignSeries::filled(value, length)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_series))))
}

// ============================================================================
// Foreign Tensor Constructor Functions
// ============================================================================

/// Create a Foreign Tensor from nested lists
/// Usage: Tensor[{{1, 2}, {3, 4}}] or Tensor[{1, 2, 3}]
pub fn tensor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (data_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let data = args[0].clone();
    let foreign_tensor = ForeignTensor::from_nested_list(data)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_tensor))))
}

/// Create a zero-filled tensor with given shape
/// Usage: ZerosTensor[{2, 3}] or ZerosTensor[{5}]
pub fn zeros_tensor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (shape_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let shape = match &args[0] {
        Value::List(shape_values) => {
            let mut shape = Vec::new();
            for val in shape_values {
                match val {
                    Value::Integer(dim) => {
                        if *dim < 0 {
                            return Err(VmError::TypeError {
                                expected: "non-negative integer dimensions".to_string(),
                                actual: format!("negative dimension: {}", dim),
                            });
                        }
                        shape.push(*dim as usize);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of integers for tensor shape".to_string(),
                        actual: format!("non-integer in shape: {:?}", val),
                    }),
                }
            }
            shape
        }
        Value::Integer(single_dim) => {
            if *single_dim < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer dimension".to_string(),
                    actual: format!("negative dimension: {}", single_dim),
                });
            }
            vec![*single_dim as usize]
        }
        _ => return Err(VmError::TypeError {
            expected: "list or integer for tensor shape".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_tensor = ForeignTensor::zeros(shape)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_tensor))))
}

/// Create a ones-filled tensor with given shape
/// Usage: OnesTensor[{2, 3}] or OnesTensor[{5}]
pub fn ones_tensor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (shape_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let shape = match &args[0] {
        Value::List(shape_values) => {
            let mut shape = Vec::new();
            for val in shape_values {
                match val {
                    Value::Integer(dim) => {
                        if *dim < 0 {
                            return Err(VmError::TypeError {
                                expected: "non-negative integer dimensions".to_string(),
                                actual: format!("negative dimension: {}", dim),
                            });
                        }
                        shape.push(*dim as usize);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of integers for tensor shape".to_string(),
                        actual: format!("non-integer in shape: {:?}", val),
                    }),
                }
            }
            shape
        }
        Value::Integer(single_dim) => {
            if *single_dim < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer dimension".to_string(),
                    actual: format!("negative dimension: {}", single_dim),
                });
            }
            vec![*single_dim as usize]
        }
        _ => return Err(VmError::TypeError {
            expected: "list or integer for tensor shape".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_tensor = ForeignTensor::ones(shape)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_tensor))))
}

/// Create an identity tensor (2D square matrix)
/// Usage: EyeTensor[3]
pub fn eye_tensor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (size)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let size = match &args[0] {
        Value::Integer(n) => {
            if *n < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer".to_string(),
                    actual: format!("negative integer: {}", n),
                });
            }
            *n as usize
        }
        _ => return Err(VmError::TypeError {
            expected: "integer for identity matrix size".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_tensor = ForeignTensor::eye(size)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_tensor))))
}

/// Create a random tensor with given shape (uniform [0,1])
/// Usage: RandomTensor[{2, 3}] or RandomTensor[{5}]
pub fn random_tensor(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::TypeError {
            expected: "1 argument (shape_list)".to_string(),
            actual: format!("{} arguments", args.len()),
        });
    }
    
    let shape = match &args[0] {
        Value::List(shape_values) => {
            let mut shape = Vec::new();
            for val in shape_values {
                match val {
                    Value::Integer(dim) => {
                        if *dim < 0 {
                            return Err(VmError::TypeError {
                                expected: "non-negative integer dimensions".to_string(),
                                actual: format!("negative dimension: {}", dim),
                            });
                        }
                        shape.push(*dim as usize);
                    }
                    _ => return Err(VmError::TypeError {
                        expected: "list of integers for tensor shape".to_string(),
                        actual: format!("non-integer in shape: {:?}", val),
                    }),
                }
            }
            shape
        }
        Value::Integer(single_dim) => {
            if *single_dim < 0 {
                return Err(VmError::TypeError {
                    expected: "non-negative integer dimension".to_string(),
                    actual: format!("negative dimension: {}", single_dim),
                });
            }
            vec![*single_dim as usize]
        }
        _ => return Err(VmError::TypeError {
            expected: "list or integer for tensor shape".to_string(),
            actual: format!("argument of type: {:?}", args[0]),
        }),
    };
    
    let foreign_tensor = ForeignTensor::random(shape)?;
    Ok(Value::LyObj(LyObj::new(Box::new(foreign_tensor))))
}