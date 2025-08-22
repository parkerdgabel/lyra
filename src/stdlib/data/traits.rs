//! Unified Data Traits for Lyra Standard Library
//!
//! This module defines common traits and interfaces that provide a consistent
//! way to work with different data structures (tables, series, tensors, etc.)

use crate::vm::Value;
use crate::foreign::ForeignError;
use crate::stdlib::data::aggregation::{AggregationSpec, AggregationSet};
use std::collections::HashMap;

/// Core trait for columnar data structures (tables, data frames, etc.)
pub trait Tabular {
    /// Get the number of rows in the data structure
    fn row_count(&self) -> usize;
    
    /// Get the column names
    fn column_names(&self) -> Vec<String>;
    
    /// Get the number of columns
    fn column_count(&self) -> usize {
        self.column_names().len()
    }
    
    /// Check if a column exists
    fn has_column(&self, name: &str) -> bool {
        self.column_names().contains(&name.to_string())
    }
    
    /// Get a specific row as a HashMap of column name -> value
    fn get_row(&self, index: usize) -> Result<HashMap<String, Value>, ForeignError>;
    
    /// Get a specific column as a vector of values
    fn get_column(&self, name: &str) -> Result<Vec<Value>, ForeignError>;
    
    /// Get a specific cell value
    fn get_cell(&self, row: usize, column: &str) -> Result<Value, ForeignError> {
        let column_data = self.get_column(column)?;
        column_data.get(row)
            .cloned()
            .ok_or_else(|| ForeignError::IndexOutOfBounds {
                index: row.to_string(),
                bounds: format!("0..{}", column_data.len()),
            })
    }
    
    /// Check if the data structure is empty
    fn is_empty(&self) -> bool {
        self.row_count() == 0
    }
}

/// Trait for data structures that support filtering operations
pub trait Filterable: Tabular {
    /// Filter rows based on a predicate function
    fn filter<F>(&self, predicate: F) -> Result<Box<dyn Tabular>, ForeignError>
    where
        F: Fn(&HashMap<String, Value>) -> bool;
    
    /// Filter rows where a specific column matches a value
    fn filter_by_column(&self, column: &str, value: &Value) -> Result<Box<dyn Tabular>, ForeignError> {
        self.filter(|row| {
            row.get(column)
                .map(|v| v == value)
                .unwrap_or(false)
        })
    }
    
    /// Filter rows where a specific column contains a value (for string columns)
    fn filter_contains(&self, column: &str, substring: &str) -> Result<Box<dyn Tabular>, ForeignError> {
        self.filter(|row| {
            row.get(column)
                .and_then(|v| match v {
                    Value::String(s) => Some(s.contains(substring)),
                    _ => None,
                })
                .unwrap_or(false)
        })
    }
}

/// Trait for data structures that support grouping and aggregation
pub trait Groupable: Tabular {
    /// Group by one or more columns and apply aggregations
    fn group_by(&self, columns: &[String], aggregations: &AggregationSet) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Group by a single column and apply aggregations
    fn group_by_single(&self, column: &str, aggregations: &AggregationSet) -> Result<Box<dyn Tabular>, ForeignError> {
        self.group_by(&[column.to_string()], aggregations)
    }
}

/// Trait for data structures that support sorting
pub trait Sortable: Tabular {
    /// Sort by one or more columns
    fn sort_by(&self, columns: &[String], ascending: &[bool]) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Sort by a single column in ascending order
    fn sort_by_column(&self, column: &str) -> Result<Box<dyn Tabular>, ForeignError> {
        self.sort_by(&[column.to_string()], &[true])
    }
    
    /// Sort by a single column in descending order
    fn sort_by_column_desc(&self, column: &str) -> Result<Box<dyn Tabular>, ForeignError> {
        self.sort_by(&[column.to_string()], &[false])
    }
}

/// Trait for data structures that support joining with other data structures
pub trait Joinable: Tabular {
    /// Inner join with another tabular data structure
    fn inner_join(&self, other: &dyn Tabular, left_key: &str, right_key: &str) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Left join with another tabular data structure
    fn left_join(&self, other: &dyn Tabular, left_key: &str, right_key: &str) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Right join with another tabular data structure
    fn right_join(&self, other: &dyn Tabular, left_key: &str, right_key: &str) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Full outer join with another tabular data structure
    fn full_join(&self, other: &dyn Tabular, left_key: &str, right_key: &str) -> Result<Box<dyn Tabular>, ForeignError>;
}

/// Trait for data structures that support column transformations
pub trait Transformable: Tabular {
    /// Add a new column computed from existing columns
    fn add_column<F>(&self, name: &str, compute: F) -> Result<Box<dyn Tabular>, ForeignError>
    where
        F: Fn(&HashMap<String, Value>) -> Value;
    
    /// Remove a column
    fn remove_column(&self, name: &str) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Rename a column
    fn rename_column(&self, old_name: &str, new_name: &str) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Select only specific columns
    fn select_columns(&self, columns: &[String]) -> Result<Box<dyn Tabular>, ForeignError>;
    
    /// Transform values in a specific column
    fn transform_column<F>(&self, column: &str, transform: F) -> Result<Box<dyn Tabular>, ForeignError>
    where
        F: Fn(&Value) -> Value;
}

/// Core trait for series data structures (1-dimensional arrays)
pub trait Series {
    /// Get the length of the series
    fn len(&self) -> usize;
    
    /// Check if the series is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get a value by index
    fn get(&self, index: usize) -> Result<Value, ForeignError>;
    
    /// Get all values as a vector
    fn values(&self) -> Vec<Value>;
    
    /// Get the data type of the series (if homogeneous)
    fn dtype(&self) -> Option<SeriesType>;
    
    /// Get the name of the series (if named)
    fn name(&self) -> Option<String>;
}

/// Types that a series can contain
#[derive(Debug, Clone, PartialEq)]
pub enum SeriesType {
    Integer,
    Real,
    String,
    Boolean,
    Mixed,
}

/// Trait for series that support mathematical operations
pub trait NumericSeries: Series {
    /// Calculate sum of all values
    fn sum(&self) -> Result<Value, ForeignError>;
    
    /// Calculate mean of all values
    fn mean(&self) -> Result<Value, ForeignError>;
    
    /// Find minimum value
    fn min(&self) -> Result<Value, ForeignError>;
    
    /// Find maximum value
    fn max(&self) -> Result<Value, ForeignError>;
    
    /// Calculate standard deviation
    fn std(&self) -> Result<Value, ForeignError>;
    
    /// Calculate variance
    fn var(&self) -> Result<Value, ForeignError>;
}

/// Trait for series that support filtering
pub trait FilterableSeries: Series {
    /// Filter values based on a predicate
    fn filter<F>(&self, predicate: F) -> Result<Box<dyn Series>, ForeignError>
    where
        F: Fn(&Value) -> bool;
    
    /// Filter values that are greater than a threshold
    fn filter_gt(&self, threshold: &Value) -> Result<Box<dyn Series>, ForeignError> {
        self.filter(|v| match (v, threshold) {
            (Value::Integer(a), Value::Integer(b)) => a > b,
            (Value::Real(a), Value::Real(b)) => a > b,
            (Value::Integer(a), Value::Real(b)) => (*a as f64) > *b,
            (Value::Real(a), Value::Integer(b)) => *a > (*b as f64),
            _ => false,
        })
    }
    
    /// Filter values that are less than a threshold
    fn filter_lt(&self, threshold: &Value) -> Result<Box<dyn Series>, ForeignError> {
        self.filter(|v| match (v, threshold) {
            (Value::Integer(a), Value::Integer(b)) => a < b,
            (Value::Real(a), Value::Real(b)) => a < b,
            (Value::Integer(a), Value::Real(b)) => (*a as f64) < *b,
            (Value::Real(a), Value::Integer(b)) => *a < (*b as f64),
            _ => false,
        })
    }
}

/// Core trait for tensor data structures (multi-dimensional arrays)
pub trait Tensor {
    /// Get the shape of the tensor (dimensions)
    fn shape(&self) -> Vec<usize>;
    
    /// Get the number of dimensions (rank)
    fn rank(&self) -> usize {
        self.shape().len()
    }
    
    /// Get the total number of elements
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
    
    /// Check if the tensor is empty
    fn is_empty(&self) -> bool {
        self.size() == 0
    }
    
    /// Get a value by multi-dimensional index
    fn get(&self, indices: &[usize]) -> Result<Value, ForeignError>;
    
    /// Reshape the tensor to new dimensions
    fn reshape(&self, new_shape: &[usize]) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Flatten to a 1-dimensional tensor
    fn flatten(&self) -> Result<Box<dyn Tensor>, ForeignError>;
}

/// Trait for tensors that support mathematical operations
pub trait NumericTensor: Tensor {
    /// Element-wise addition with another tensor
    fn add(&self, other: &dyn Tensor) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Element-wise subtraction with another tensor
    fn sub(&self, other: &dyn Tensor) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Element-wise multiplication with another tensor
    fn mul(&self, other: &dyn Tensor) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Element-wise division with another tensor
    fn div(&self, other: &dyn Tensor) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Matrix multiplication (if applicable)
    fn matmul(&self, other: &dyn Tensor) -> Result<Box<dyn Tensor>, ForeignError>;
    
    /// Transpose (for 2D tensors)
    fn transpose(&self) -> Result<Box<dyn Tensor>, ForeignError>;
}

/// Trait for data structures that can be serialized/deserialized
pub trait Serializable {
    /// Serialize to a Value that can be used by the VM
    fn to_value(&self) -> Result<Value, ForeignError>;
    
    /// Deserialize from a VM Value
    fn from_value(value: &Value) -> Result<Box<dyn Serializable>, ForeignError>
    where
        Self: Sized;
}

/// Trait for data structures that support I/O operations
pub trait DataIO {
    /// Export to JSON format
    fn to_json(&self) -> Result<String, ForeignError>;
    
    /// Export to CSV format (if applicable)
    fn to_csv(&self) -> Result<String, ForeignError>;
    
    /// Import from JSON string
    fn from_json(json: &str) -> Result<Box<dyn DataIO>, ForeignError>
    where
        Self: Sized;
    
    /// Import from CSV string
    fn from_csv(csv: &str) -> Result<Box<dyn DataIO>, ForeignError>
    where
        Self: Sized;
}

/// Marker trait for data structures that are thread-safe
pub trait ThreadSafe: Send + Sync {}

/// Comprehensive trait that combines all data manipulation capabilities
pub trait DataStructure: Tabular + Filterable + Groupable + Sortable + Joinable + Transformable + Serializable + DataIO + ThreadSafe {
    /// Get a human-readable description of the data structure
    fn describe(&self) -> String {
        format!("{} rows × {} columns", self.row_count(), self.column_count())
    }
    
    /// Get basic statistics about the data structure
    fn info(&self) -> HashMap<String, Value> {
        let mut info = HashMap::new();
        info.insert("rows".to_string(), Value::Integer(self.row_count() as i64));
        info.insert("columns".to_string(), Value::Integer(self.column_count() as i64));
        info.insert("column_names".to_string(), Value::List(
            self.column_names().into_iter().map(Value::String).collect()
        ));
        info
    }
}

/// Utility functions for working with data structures
pub mod utils {
    use super::*;
    
    /// Check if two data structures have compatible schemas for joining
    pub fn can_join(left: &dyn Tabular, right: &dyn Tabular, left_key: &str, right_key: &str) -> bool {
        left.has_column(left_key) && right.has_column(right_key)
    }
    
    /// Get common columns between two data structures
    pub fn common_columns(left: &dyn Tabular, right: &dyn Tabular) -> Vec<String> {
        let left_cols: std::collections::HashSet<String> = left.column_names().into_iter().collect();
        let right_cols: std::collections::HashSet<String> = right.column_names().into_iter().collect();
        
        left_cols.intersection(&right_cols).cloned().collect()
    }
    
    /// Validate that a shape is valid for tensor operations
    pub fn validate_tensor_shape(shape: &[usize]) -> bool {
        !shape.is_empty() && shape.iter().all(|&dim| dim > 0)
    }
    
    /// Calculate the flat index for multi-dimensional tensor access
    pub fn calculate_flat_index(indices: &[usize], shape: &[usize]) -> Option<usize> {
        if indices.len() != shape.len() {
            return None;
        }
        
        let mut flat_index = 0;
        let mut stride = 1;
        
        for i in (0..indices.len()).rev() {
            if indices[i] >= shape[i] {
                return None;
            }
            flat_index += indices[i] * stride;
            stride *= shape[i];
        }
        
        Some(flat_index)
    }
    
    /// Convert a flat index back to multi-dimensional indices
    pub fn calculate_multi_index(flat_index: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut remaining = flat_index;
        
        for i in (0..shape.len()).rev() {
            let stride: usize = shape[i+1..].iter().product();
            indices[i] = remaining / stride;
            remaining %= stride;
        }
        
        indices
    }
    
    /// Check if two shapes are compatible for broadcasting
    pub fn can_broadcast(shape1: &[usize], shape2: &[usize]) -> bool {
        let max_len = shape1.len().max(shape2.len());
        
        for i in 0..max_len {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).copied().unwrap_or(1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).copied().unwrap_or(1);
            
            if dim1 != dim2 && dim1 != 1 && dim2 != 1 {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate the resulting shape after broadcasting
    pub fn broadcast_shape(shape1: &[usize], shape2: &[usize]) -> Option<Vec<usize>> {
        if !can_broadcast(shape1, shape2) {
            return None;
        }
        
        let max_len = shape1.len().max(shape2.len());
        let mut result = Vec::with_capacity(max_len);
        
        for i in 0..max_len {
            let dim1 = shape1.get(shape1.len().wrapping_sub(i + 1)).copied().unwrap_or(1);
            let dim2 = shape2.get(shape2.len().wrapping_sub(i + 1)).copied().unwrap_or(1);
            
            result.push(dim1.max(dim2));
        }
        
        result.reverse();
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::utils::*;
    
    #[test]
    fn test_calculate_flat_index() {
        // 2D array: 3x4
        let shape = vec![3, 4];
        assert_eq!(calculate_flat_index(&[0, 0], &shape), Some(0));
        assert_eq!(calculate_flat_index(&[0, 1], &shape), Some(1));
        assert_eq!(calculate_flat_index(&[1, 0], &shape), Some(4));
        assert_eq!(calculate_flat_index(&[2, 3], &shape), Some(11));
        assert_eq!(calculate_flat_index(&[3, 0], &shape), None); // Out of bounds
    }
    
    #[test]
    fn test_calculate_multi_index() {
        // 2D array: 3x4
        let shape = vec![3, 4];
        assert_eq!(calculate_multi_index(0, &shape), vec![0, 0]);
        assert_eq!(calculate_multi_index(1, &shape), vec![0, 1]);
        assert_eq!(calculate_multi_index(4, &shape), vec![1, 0]);
        assert_eq!(calculate_multi_index(11, &shape), vec![2, 3]);
    }
    
    #[test]
    fn test_can_broadcast() {
        assert!(can_broadcast(&[3, 4], &[4]));
        assert!(can_broadcast(&[3, 4], &[1, 4]));
        assert!(can_broadcast(&[3, 1], &[1, 4]));
        assert!(!can_broadcast(&[3, 4], &[3, 5]));
        assert!(can_broadcast(&[1], &[3, 4, 5]));
    }
    
    #[test]
    fn test_broadcast_shape() {
        assert_eq!(broadcast_shape(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[3, 4], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shape(&[3, 4], &[3, 5]), None);
        assert_eq!(broadcast_shape(&[1], &[3, 4, 5]), Some(vec![3, 4, 5]));
    }
}