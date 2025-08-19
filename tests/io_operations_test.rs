//! Comprehensive I/O operations tests for Import/Export functionality
//! 
//! Tests multi-format support, streaming operations, error handling, and integration with VM.

use lyra::io::{import, export};
use lyra::vm::{Value, VmError};
use std::fs::{File, create_dir_all, remove_file, remove_dir_all};
use std::io::Write;
use std::path::Path;
use tempfile::tempdir;

#[test]
fn test_import_export_json() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.json");
    
    // Test data
    let test_data = Value::List(vec![
        Value::Integer(42),
        Value::String("hello".to_string()),
        Value::Boolean(true),
    ]);
    
    // Export to JSON
    let export_args = vec![
        test_data.clone(),
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("JSON".to_string()),
    ];
    
    let export_result = export(&export_args);
    assert!(export_result.is_ok());
    
    // Import from JSON
    let import_args = vec![
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("JSON".to_string()),
    ];
    
    let import_result = import(&import_args);
    assert!(import_result.is_ok());
    
    // Basic verification (actual JSON parsing is simplified in current implementation)
    match import_result.unwrap() {
        Value::List(_) => {}, // Expected for JSON array
        _ => panic!("Expected list from JSON import"),
    }
}

#[test]
fn test_import_export_csv() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.csv");
    
    // Test data - CSV requires list of lists
    let test_data = Value::List(vec![
        Value::List(vec![
            Value::String("Name".to_string()),
            Value::String("Age".to_string()),
            Value::String("Score".to_string()),
        ]),
        Value::List(vec![
            Value::String("Alice".to_string()),
            Value::Integer(25),
            Value::Real(95.5),
        ]),
        Value::List(vec![
            Value::String("Bob".to_string()),
            Value::Integer(30),
            Value::Real(87.2),
        ]),
    ]);
    
    // Export to CSV
    let export_args = vec![
        test_data.clone(),
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("CSV".to_string()),
    ];
    
    let export_result = export(&export_args);
    assert!(export_result.is_ok());
    
    // Verify file exists and has content
    assert!(file_path.exists());
    let content = std::fs::read_to_string(&file_path).unwrap();
    assert!(content.contains("Name,Age,Score"));
    assert!(content.contains("Alice,25,95.5"));
    
    // Import from CSV
    let import_args = vec![
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("CSV".to_string()),
    ];
    
    let import_result = import(&import_args);
    assert!(import_result.is_ok());
    
    match import_result.unwrap() {
        Value::List(rows) => {
            assert!(rows.len() >= 2); // At least header + 1 data row
        },
        _ => panic!("Expected list from CSV import"),
    }
}

#[test]
fn test_import_export_text() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    
    // Test data
    let test_data = Value::String("Hello, World!\nThis is a test file.\nLine 3.".to_string());
    
    // Export to text
    let export_args = vec![
        test_data.clone(),
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("TEXT".to_string()),
    ];
    
    let export_result = export(&export_args);
    assert!(export_result.is_ok());
    
    // Import from text
    let import_args = vec![
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("TEXT".to_string()),
    ];
    
    let import_result = import(&import_args);
    assert!(import_result.is_ok());
    
    match import_result.unwrap() {
        Value::String(content) => {
            assert!(content.contains("Hello, World!"));
            assert!(content.contains("test file"));
        },
        _ => panic!("Expected string from text import"),
    }
}

#[test]
fn test_import_export_binary() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.bin");
    
    // Test data - binary requires list of integers 0-255
    let test_data = Value::List(vec![
        Value::Integer(72),  // 'H'
        Value::Integer(101), // 'e'
        Value::Integer(108), // 'l'
        Value::Integer(108), // 'l'
        Value::Integer(111), // 'o'
    ]);
    
    // Export to binary
    let export_args = vec![
        test_data.clone(),
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("BINARY".to_string()),
    ];
    
    let export_result = export(&export_args);
    assert!(export_result.is_ok());
    
    // Import from binary
    let import_args = vec![
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("BINARY".to_string()),
    ];
    
    let import_result = import(&import_args);
    assert!(import_result.is_ok());
    
    match import_result.unwrap() {
        Value::List(bytes) => {
            assert_eq!(bytes.len(), 5);
            assert_eq!(bytes[0], Value::Integer(72));
            assert_eq!(bytes[4], Value::Integer(111));
        },
        _ => panic!("Expected list from binary import"),
    }
}

#[test]
fn test_format_auto_detection() {
    let temp_dir = tempdir().unwrap();
    
    // Create test files with different extensions
    let json_path = temp_dir.path().join("data.json");
    let csv_path = temp_dir.path().join("data.csv");
    let txt_path = temp_dir.path().join("data.txt");
    
    // Create simple test files
    std::fs::write(&json_path, "[1, 2, 3]").unwrap();
    std::fs::write(&csv_path, "a,b,c\n1,2,3").unwrap();
    std::fs::write(&txt_path, "Hello World").unwrap();
    
    // Test auto-detection (without explicit format)
    let json_args = vec![Value::String(json_path.to_string_lossy().to_string())];
    let csv_args = vec![Value::String(csv_path.to_string_lossy().to_string())];
    let txt_args = vec![Value::String(txt_path.to_string_lossy().to_string())];
    
    // All should succeed with auto-detection
    assert!(import(&json_args).is_ok());
    assert!(import(&csv_args).is_ok());
    assert!(import(&txt_args).is_ok());
}

#[test]
fn test_error_handling() {
    // Test file not found
    let import_args = vec![Value::String("/nonexistent/path/file.txt".to_string())];
    let result = import(&import_args);
    assert!(result.is_err());
    
    match result.err().unwrap() {
        VmError::TypeError { actual, .. } => {
            assert!(actual.contains("File not found") || actual.contains("I/O error"));
        },
        _ => panic!("Expected TypeError for file not found"),
    }
    
    // Test invalid argument count
    let result = import(&[]);
    assert!(result.is_err());
    
    let result = export(&[Value::String("data".to_string())]);
    assert!(result.is_err());
    
    // Test invalid argument types
    let result = import(&[Value::Integer(42)]);
    assert!(result.is_err());
    
    let result = export(&[
        Value::String("data".to_string()),
        Value::Integer(42), // Invalid filename type
    ]);
    assert!(result.is_err());
    
    // Test invalid format
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.txt");
    std::fs::write(&file_path, "test").unwrap();
    
    let result = import(&[
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("INVALID_FORMAT".to_string()),
    ]);
    assert!(result.is_err());
}

#[test]
fn test_csv_export_validation() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.csv");
    
    // Test invalid data for CSV (must be list of lists)
    let invalid_data = Value::String("Not a list".to_string());
    
    let export_args = vec![
        invalid_data,
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("CSV".to_string()),
    ];
    
    let result = export(&export_args);
    assert!(result.is_err());
}

#[test]
fn test_binary_export_validation() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("test.bin");
    
    // Test invalid data for binary (must be list of integers 0-255)
    let invalid_data = Value::List(vec![
        Value::Integer(256), // Out of range
    ]);
    
    let export_args = vec![
        invalid_data,
        Value::String(file_path.to_string_lossy().to_string()),
        Value::String("BINARY".to_string()),
    ];
    
    let result = export(&export_args);
    assert!(result.is_err());
}

#[test]
fn test_directory_creation() {
    let temp_dir = tempdir().unwrap();
    let nested_path = temp_dir.path().join("nested").join("directory").join("test.txt");
    
    let test_data = Value::String("Test content".to_string());
    
    // Export should create parent directories
    let export_args = vec![
        test_data,
        Value::String(nested_path.to_string_lossy().to_string()),
        Value::String("TEXT".to_string()),
    ];
    
    let result = export(&export_args);
    assert!(result.is_ok());
    assert!(nested_path.exists());
}

// Note: parse_format is internal, so we test it indirectly through import/export

#[test]
fn test_empty_file_handling() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("empty.txt");
    
    // Create empty file
    File::create(&file_path).unwrap();
    
    // Import should handle empty files gracefully
    let import_args = vec![Value::String(file_path.to_string_lossy().to_string())];
    let result = import(&import_args);
    
    assert!(result.is_ok());
    match result.unwrap() {
        Value::String(content) => assert!(content.is_empty()),
        _ => panic!("Expected empty string for empty text file"),
    }
}

#[test]
fn test_large_text_file() {
    let temp_dir = tempdir().unwrap();
    let file_path = temp_dir.path().join("large.txt");
    
    // Create a reasonably large text file
    let large_content = "Hello World!\n".repeat(1000);
    std::fs::write(&file_path, &large_content).unwrap();
    
    // Import should handle large files
    let import_args = vec![Value::String(file_path.to_string_lossy().to_string())];
    let result = import(&import_args);
    
    assert!(result.is_ok());
    match result.unwrap() {
        Value::String(content) => {
            assert_eq!(content.len(), large_content.len());
            assert!(content.contains("Hello World!"));
        },
        _ => panic!("Expected string for text file"),
    }
}

// Note: format detection is tested indirectly through auto-detection in import/export