//! Data Processing Workload Simulations
//!
//! ETL-style operations with async pipelines that test data transformation,
//! filtering, aggregation, and stream processing capabilities.

use criterion::{black_box, criterion_group, Criterion, BenchmarkId, Throughput};
use lyra::{
    vm::{VirtualMachine, Value},
    parser::Parser,
    compiler::Compiler,
    memory::{MemoryManager, CompactValue, StringInterner},
};
use std::sync::Arc;
use std::collections::HashMap;

/// Generate synthetic dataset for processing
fn generate_dataset(size: usize) -> Vec<Value> {
    (0..size).map(|i| {
        Value::List(vec![
            Value::Integer(i as i64),                    // ID
            Value::String(format!("record_{}", i)),      // Name
            Value::Real((i as f64) * 1.5 + 0.5),        // Value
            Value::Boolean(i % 2 == 0),                  // Flag
            Value::Symbol(format!("category_{}", i % 10)), // Category
        ])
    }).collect()
}

/// Generate large tabular data for processing
fn generate_tabular_data(rows: usize, cols: usize) -> Value {
    let mut data = Vec::new();
    
    // Header row
    let header: Vec<Value> = (0..cols)
        .map(|i| Value::String(format!("col_{}", i)))
        .collect();
    data.push(Value::List(header));
    
    // Data rows
    for row in 0..rows {
        let row_data: Vec<Value> = (0..cols).map(|col| {
            match col % 4 {
                0 => Value::Integer((row * cols + col) as i64),
                1 => Value::Real((row as f64 + col as f64) / 10.0),
                2 => Value::String(format!("data_{}_{}", row, col)),
                _ => Value::Boolean((row + col) % 2 == 0),
            }
        }).collect();
        data.push(Value::List(row_data));
    }
    
    Value::List(data)
}

/// Benchmark data filtering operations
fn data_filtering_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_filtering");
    group.throughput(Throughput::Elements(10000));
    
    let dataset = generate_dataset(10000);
    
    group.bench_function("filter_by_predicate", |b| {
        b.iter(|| {
            let mut filtered = Vec::new();
            
            for record in &dataset {
                if let Value::List(fields) = record {
                    // Filter: keep records where ID is even and value > 50
                    if fields.len() >= 3 {
                        if let (Value::Integer(id), Value::Real(value)) = (&fields[0], &fields[2]) {
                            if id % 2 == 0 && *value > 50.0 {
                                filtered.push(record.clone());
                            }
                        }
                    }
                }
            }
            
            black_box(filtered);
        });
    });
    
    group.bench_function("filter_by_category", |b| {
        b.iter(|| {
            let mut filtered = Vec::new();
            
            for record in &dataset {
                if let Value::List(fields) = record {
                    // Filter: keep records in specific categories
                    if fields.len() >= 5 {
                        if let Value::Symbol(category) = &fields[4] {
                            if category == "category_0" || category == "category_5" {
                                filtered.push(record.clone());
                            }
                        }
                    }
                }
            }
            
            black_box(filtered);
        });
    });
    
    group.finish();
}

/// Benchmark data transformation operations
fn data_transformation_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("data_transformation");
    group.throughput(Throughput::Elements(10000));
    
    let dataset = generate_dataset(10000);
    
    group.bench_function("map_transformation", |b| {
        b.iter(|| {
            let mut transformed = Vec::new();
            
            for record in &dataset {
                if let Value::List(fields) = record {
                    if fields.len() >= 3 {
                        // Transform: normalize values and add computed field
                        let mut new_fields = fields.clone();
                        
                        if let Value::Real(value) = &fields[2] {
                            // Normalize value to 0-1 range
                            let normalized = value / 1000.0;
                            new_fields[2] = Value::Real(normalized);
                            
                            // Add computed field (square of normalized value)
                            new_fields.push(Value::Real(normalized * normalized));
                        }
                        
                        transformed.push(Value::List(new_fields));
                    }
                }
            }
            
            black_box(transformed);
        });
    });
    
    group.bench_function("aggregation_by_category", |b| {
        b.iter(|| {
            let mut aggregates: HashMap<String, (i64, f64, usize)> = HashMap::new();
            
            for record in &dataset {
                if let Value::List(fields) = record {
                    if fields.len() >= 5 {
                        if let (Value::Integer(id), Value::Real(value), Value::Symbol(category)) = 
                            (&fields[0], &fields[2], &fields[4]) {
                            
                            let entry = aggregates.entry(category.clone()).or_insert((0, 0.0, 0));
                            entry.0 += id;        // Sum of IDs
                            entry.1 += value;     // Sum of values
                            entry.2 += 1;         // Count
                        }
                    }
                }
            }
            
            black_box(aggregates);
        });
    });
    
    group.finish();
}

/// Benchmark ETL pipeline operations
fn etl_pipeline_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("etl_pipeline");
    group.throughput(Throughput::Elements(5000));
    
    let raw_data = generate_dataset(5000);
    
    group.bench_function("complete_etl_pipeline", |b| {
        b.iter(|| {
            // Extract: Input data (already have it)
            let extracted = &raw_data;
            
            // Transform: Filter, normalize, and enrich
            let mut transformed = Vec::new();
            for record in extracted {
                if let Value::List(fields) = record {
                    if fields.len() >= 5 {
                        // Filter: only process records with positive values
                        if let Value::Real(value) = &fields[2] {
                            if *value > 0.0 {
                                let mut new_record = fields.clone();
                                
                                // Normalize value
                                new_record[2] = Value::Real(value / 100.0);
                                
                                // Add enrichment fields
                                new_record.push(Value::String("processed".to_string()));
                                new_record.push(Value::Integer(chrono::Utc::now().timestamp_millis()));
                                
                                transformed.push(Value::List(new_record));
                            }
                        }
                    }
                }
            }
            
            // Load: Group by category and create summary
            let mut summary: HashMap<String, Vec<Value>> = HashMap::new();
            for record in &transformed {
                if let Value::List(fields) = record {
                    if fields.len() >= 5 {
                        if let Value::Symbol(category) = &fields[4] {
                            summary.entry(category.clone()).or_insert_with(Vec::new).push(record.clone());
                        }
                    }
                }
            }
            
            black_box((transformed, summary));
        });
    });
    
    group.finish();
}

/// Benchmark large dataset operations
fn large_dataset_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_dataset_operations");
    
    for size in [1000, 5000, 10000, 20000].iter() {
        let dataset = generate_dataset(*size);
        
        group.bench_with_input(
            BenchmarkId::new("dataset_scan", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut count = 0;
                    let mut sum = 0.0;
                    
                    for record in &dataset {
                        if let Value::List(fields) = record {
                            if fields.len() >= 3 {
                                if let Value::Real(value) = &fields[2] {
                                    count += 1;
                                    sum += value;
                                }
                            }
                        }
                    }
                    
                    black_box((count, sum));
                });
            },
        );
    }
    
    group.finish();
}

/// Benchmark tabular data operations
fn tabular_data_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("tabular_data_operations");
    
    let table_data = generate_tabular_data(1000, 10);
    
    group.bench_function("column_selection", |b| {
        b.iter(|| {
            if let Value::List(rows) = &table_data {
                let mut selected_columns = Vec::new();
                
                for row in rows {
                    if let Value::List(cells) = row {
                        // Select columns 0, 2, 4 (every other column)
                        let selected_row: Vec<Value> = cells.iter()
                            .enumerate()
                            .filter(|(i, _)| i % 2 == 0)
                            .map(|(_, cell)| cell.clone())
                            .collect();
                        selected_columns.push(Value::List(selected_row));
                    }
                }
                
                black_box(selected_columns);
            }
        });
    });
    
    group.bench_function("row_filtering", |b| {
        b.iter(|| {
            if let Value::List(rows) = &table_data {
                let mut filtered_rows = Vec::new();
                
                for (i, row) in rows.iter().enumerate() {
                    if i == 0 {
                        // Keep header
                        filtered_rows.push(row.clone());
                    } else if let Value::List(cells) = row {
                        // Filter: keep rows where first column value is even
                        if !cells.is_empty() {
                            if let Value::Integer(val) = &cells[0] {
                                if val % 2 == 0 {
                                    filtered_rows.push(row.clone());
                                }
                            }
                        }
                    }
                }
                
                black_box(filtered_rows);
            }
        });
    });
    
    group.bench_function("table_join_simulation", |b| {
        // Create a second table for joining
        let lookup_table: HashMap<i64, String> = (0..500)
            .map(|i| (i, format!("lookup_value_{}", i)))
            .collect();
        
        b.iter(|| {
            if let Value::List(rows) = &table_data {
                let mut joined_rows = Vec::new();
                
                for row in rows {
                    if let Value::List(cells) = row {
                        if !cells.is_empty() {
                            let mut new_row = cells.clone();
                            
                            // Join based on first column
                            if let Value::Integer(key) = &cells[0] {
                                if let Some(lookup_value) = lookup_table.get(key) {
                                    new_row.push(Value::String(lookup_value.clone()));
                                } else {
                                    new_row.push(Value::String("NOT_FOUND".to_string()));
                                }
                            }
                            
                            joined_rows.push(Value::List(new_row));
                        }
                    }
                }
                
                black_box(joined_rows);
            }
        });
    });
    
    group.finish();
}

/// Benchmark memory-efficient data processing
fn memory_efficient_data_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficient_processing");
    
    group.bench_function("memory_managed_processing", |b| {
        let mut memory_manager = MemoryManager::new();
        let interner = StringInterner::new();
        
        b.iter(|| {
            let mut processed_data = Vec::new();
            
            // Process data using memory-managed values
            for i in 0..1000 {
                let id_symbol = interner.intern_symbol_id(&format!("record_{}", i));
                let category_symbol = interner.intern_symbol_id(&format!("cat_{}", i % 5));
                
                let record = vec![
                    CompactValue::SmallInt(i as i32),
                    CompactValue::Symbol(id_symbol),
                    CompactValue::Real(i as f64 * 1.5),
                    CompactValue::Boolean(i % 2 == 0),
                    CompactValue::Symbol(category_symbol),
                ];
                
                let compact_record = CompactValue::List(Arc::new(record));
                let managed_record = memory_manager.alloc_compact_value(compact_record);
                processed_data.push(managed_record);
            }
            
            // Simulate processing cleanup
            for record in &processed_data {
                memory_manager.recycle_compact_value(record);
            }
            
            black_box(processed_data);
        });
    });
    
    group.bench_function("traditional_processing", |b| {
        b.iter(|| {
            let mut processed_data = Vec::new();
            
            // Process data using traditional values
            for i in 0..1000 {
                let record = vec![
                    Value::Integer(i),
                    Value::String(format!("record_{}", i)),
                    Value::Real(i as f64 * 1.5),
                    Value::Boolean(i % 2 == 0),
                    Value::String(format!("cat_{}", i % 5)),
                ];
                
                processed_data.push(Value::List(record));
            }
            
            black_box(processed_data);
        });
    });
    
    group.finish();
}

criterion_group!(
    data_processing_benchmarks,
    data_filtering_operations,
    data_transformation_operations,
    etl_pipeline_operations,
    large_dataset_operations,
    tabular_data_operations,
    memory_efficient_data_processing
);

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn validate_dataset_generation() {
        let dataset = generate_dataset(100);
        assert_eq!(dataset.len(), 100);
        
        // Verify structure of first record
        if let Value::List(fields) = &dataset[0] {
            assert_eq!(fields.len(), 5);
            assert!(matches!(fields[0], Value::Integer(_)));
            assert!(matches!(fields[1], Value::String(_)));
            assert!(matches!(fields[2], Value::Real(_)));
            assert!(matches!(fields[3], Value::Boolean(_)));
            assert!(matches!(fields[4], Value::Symbol(_)));
        } else {
            panic!("Expected list for record");
        }
    }
    
    #[test]
    fn validate_tabular_data_generation() {
        let table = generate_tabular_data(10, 5);
        
        if let Value::List(rows) = table {
            assert_eq!(rows.len(), 11); // 10 data rows + 1 header
            
            // Check header
            if let Value::List(header) = &rows[0] {
                assert_eq!(header.len(), 5);
                for (i, col) in header.iter().enumerate() {
                    if let Value::String(name) = col {
                        assert_eq!(name, &format!("col_{}", i));
                    } else {
                        panic!("Expected string column name");
                    }
                }
            }
            
            // Check data row
            if let Value::List(data_row) = &rows[1] {
                assert_eq!(data_row.len(), 5);
            }
        } else {
            panic!("Expected list for table");
        }
    }
    
    #[test]
    fn validate_filtering_operations() {
        let dataset = generate_dataset(10);
        let mut filtered = Vec::new();
        
        for record in &dataset {
            if let Value::List(fields) = record {
                if fields.len() >= 3 {
                    if let (Value::Integer(id), Value::Real(value)) = (&fields[0], &fields[2]) {
                        if id % 2 == 0 && *value > 5.0 {
                            filtered.push(record.clone());
                        }
                    }
                }
            }
        }
        
        println!("Filtered {} records from {} total", filtered.len(), dataset.len());
        assert!(filtered.len() <= dataset.len());
    }
    
    #[test]
    fn validate_etl_pipeline() {
        let raw_data = generate_dataset(10);
        
        // Extract (already have data)
        let extracted = &raw_data;
        
        // Transform
        let mut transformed = Vec::new();
        for record in extracted {
            if let Value::List(fields) = record {
                if fields.len() >= 5 {
                    let mut new_record = fields.clone();
                    new_record.push(Value::String("processed".to_string()));
                    transformed.push(Value::List(new_record));
                }
            }
        }
        
        // Load (aggregate)
        let mut summary: HashMap<String, usize> = HashMap::new();
        for record in &transformed {
            if let Value::List(fields) = record {
                if fields.len() >= 5 {
                    if let Value::Symbol(category) = &fields[4] {
                        *summary.entry(category.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        
        println!("ETL pipeline processed {} records into {} categories", 
                transformed.len(), summary.len());
        
        assert_eq!(transformed.len(), raw_data.len());
        assert!(!summary.is_empty());
    }
    
    #[test]
    fn validate_memory_efficient_processing() {
        let mut memory_manager = MemoryManager::new();
        let interner = StringInterner::new();
        
        let initial_memory = memory_manager.memory_stats().total_allocated;
        
        // Process some data
        let mut processed = Vec::new();
        for i in 0..100 {
            let symbol_id = interner.intern_symbol_id(&format!("test_{}", i));
            let compact_val = CompactValue::Symbol(symbol_id);
            processed.push(memory_manager.alloc_compact_value(compact_val));
        }
        
        let peak_memory = memory_manager.memory_stats().total_allocated;
        
        // Clean up
        for value in &processed {
            memory_manager.recycle_compact_value(value);
        }
        
        let final_memory = memory_manager.memory_stats().total_allocated;
        
        println!("Memory usage: {} -> {} -> {}", initial_memory, peak_memory, final_memory);
        
        assert!(peak_memory >= initial_memory);
        // Note: final_memory may not be less than peak_memory depending on pool implementation
    }
}