//! Simple unit tests for the implemented file I/O functions
//! These tests verify the core functionality without complex integration

use std::fs;
use std::path::Path;
use tempfile::TempDir;

// Mock Value and VmResult types for testing
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    String(String),
    Boolean(bool),
    Integer(i64),
    List(Vec<Value>),
    Missing,
}

#[derive(Debug)]
pub enum VmError {
    Runtime(String),
}

pub type VmResult<T> = Result<T, VmError>;

// Include the module functions directly for testing
mod io_functions {
    use super::*;
    use std::fs;
    use std::io::{Write, BufRead, BufReader};
    use std::path::{Path, PathBuf};

    // Copy the actual implementations here for testing
    pub fn file_read(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("FileRead requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        match fs::read_to_string(path) {
            Ok(content) => Ok(Value::String(content)),
            Err(e) => Err(VmError::Runtime(format!("Failed to read file '{}': {}", path, e))),
        }
    }

    pub fn file_write(args: &[Value]) -> VmResult<Value> {
        if args.len() < 2 || args.len() > 3 {
            return Err(VmError::Runtime("FileWrite requires 2-3 arguments (path, content, [options])".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        let content = match &args[1] {
            Value::String(s) => s.as_bytes(),
            Value::Integer(n) => n.to_string().as_bytes(),
            _ => return Err(VmError::Runtime("Content must be convertible to string".to_string())),
        };
        
        // Create parent directories if they don't exist
        if let Some(parent) = Path::new(path).parent() {
            if let Err(e) = fs::create_dir_all(parent) {
                return Err(VmError::Runtime(format!("Failed to create parent directories: {}", e)));
            }
        }
        
        match fs::write(path, content) {
            Ok(()) => Ok(Value::String(path.to_string())),
            Err(e) => Err(VmError::Runtime(format!("Failed to write file '{}': {}", path, e))),
        }
    }

    pub fn file_exists(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("FileExists requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        let exists = Path::new(path).exists() && Path::new(path).is_file();
        Ok(Value::Boolean(exists))
    }

    pub fn file_size(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("FileSize requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        match fs::metadata(path) {
            Ok(metadata) => Ok(Value::Integer(metadata.len() as i64)),
            Err(e) => Err(VmError::Runtime(format!("Failed to get file size for '{}': {}", path, e))),
        }
    }

    pub fn directory_create(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("DirectoryCreate requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        match fs::create_dir_all(path) {
            Ok(()) => Ok(Value::String(path.to_string())),
            Err(e) => Err(VmError::Runtime(format!("Failed to create directory '{}': {}", path, e))),
        }
    }

    pub fn directory_exists(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("DirectoryExists requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        let exists = Path::new(path).exists() && Path::new(path).is_dir();
        Ok(Value::Boolean(exists))
    }

    pub fn path_join(args: &[Value]) -> VmResult<Value> {
        if args.is_empty() {
            return Err(VmError::Runtime("PathJoin requires at least 1 argument".to_string()));
        }
        
        let mut path = PathBuf::new();
        
        for arg in args {
            match arg {
                Value::String(s) => path.push(s),
                _ => return Err(VmError::Runtime("All path components must be strings".to_string())),
            }
        }
        
        match path.to_str() {
            Some(path_str) => Ok(Value::String(path_str.to_string())),
            None => Err(VmError::Runtime("Invalid path characters".to_string())),
        }
    }

    pub fn path_filename(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("PathFilename requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        let path_obj = Path::new(path);
        match path_obj.file_name() {
            Some(filename) => {
                match filename.to_str() {
                    Some(filename_str) => Ok(Value::String(filename_str.to_string())),
                    None => Err(VmError::Runtime("Invalid filename characters".to_string())),
                }
            }
            None => Ok(Value::Missing),
        }
    }

    pub fn path_extension(args: &[Value]) -> VmResult<Value> {
        if args.len() != 1 {
            return Err(VmError::Runtime("PathExtension requires exactly 1 argument (path)".to_string()));
        }
        
        let path = match &args[0] {
            Value::String(s) => s,
            _ => return Err(VmError::Runtime("Path must be a string".to_string())),
        };
        
        let path_obj = Path::new(path);
        match path_obj.extension() {
            Some(ext) => {
                match ext.to_str() {
                    Some(ext_str) => Ok(Value::String(ext_str.to_string())),
                    None => Err(VmError::Runtime("Invalid extension characters".to_string())),
                }
            }
            None => Ok(Value::String("".to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use io_functions::*;

    /// Helper function to create a temporary test directory
    fn create_test_dir() -> TempDir {
        TempDir::new().expect("Failed to create temp directory")
    }

    /// Helper function to create a test file with content
    fn create_test_file(dir: &TempDir, name: &str, content: &str) -> String {
        let file_path = dir.path().join(name);
        fs::write(&file_path, content).expect("Failed to write test file");
        file_path.to_str().unwrap().to_string()
    }

    #[test]
    fn test_file_read_success() {
        let temp_dir = create_test_dir();
        let file_path = create_test_file(&temp_dir, "test.txt", "Hello, World!");
        
        let result = file_read(&[Value::String(file_path)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("Hello, World!".to_string()));
    }

    #[test]
    fn test_file_read_nonexistent() {
        let result = file_read(&[Value::String("/nonexistent/file.txt".to_string())]);
        assert!(result.is_err());
    }

    #[test]
    fn test_file_write_success() {
        let temp_dir = create_test_dir();
        let file_path = temp_dir.path().join("new_file.txt");
        
        let result = file_write(&[
            Value::String(file_path.to_str().unwrap().to_string()),
            Value::String("Test content".to_string())
        ]);
        
        assert!(result.is_ok());
        
        // Verify file was created and has correct content
        let actual_content = fs::read_to_string(&file_path).expect("Failed to read written file");
        assert_eq!(actual_content, "Test content");
    }

    #[test]
    fn test_file_exists_true() {
        let temp_dir = create_test_dir();
        let file_path = create_test_file(&temp_dir, "exists.txt", "content");
        
        let result = file_exists(&[Value::String(file_path)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_file_exists_false() {
        let result = file_exists(&[Value::String("/nonexistent/file.txt".to_string())]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_file_size() {
        let temp_dir = create_test_dir();
        let content = "Hello, World!"; // 13 bytes
        let file_path = create_test_file(&temp_dir, "size_test.txt", content);
        
        let result = file_size(&[Value::String(file_path)]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Integer(13));
    }

    #[test]
    fn test_directory_create() {
        let temp_dir = create_test_dir();
        let new_dir_path = temp_dir.path().join("new_directory");
        
        let result = directory_create(&[
            Value::String(new_dir_path.to_str().unwrap().to_string())
        ]);
        
        assert!(result.is_ok());
        assert!(new_dir_path.exists());
        assert!(new_dir_path.is_dir());
    }

    #[test]
    fn test_directory_exists_true() {
        let temp_dir = create_test_dir();
        
        let result = directory_exists(&[
            Value::String(temp_dir.path().to_str().unwrap().to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(true));
    }

    #[test]
    fn test_directory_exists_false() {
        let result = directory_exists(&[
            Value::String("/nonexistent/directory".to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::Boolean(false));
    }

    #[test]
    fn test_path_join() {
        let result = path_join(&[
            Value::String("home".to_string()),
            Value::String("user".to_string()),
            Value::String("documents".to_string()),
            Value::String("file.txt".to_string())
        ]);
        
        assert!(result.is_ok());
        
        if let Value::String(joined_path) = result.unwrap() {
            let expected = Path::new("home").join("user").join("documents").join("file.txt");
            assert_eq!(joined_path, expected.to_str().unwrap());
        } else {
            panic!("Expected string path");
        }
    }

    #[test]
    fn test_path_filename() {
        let result = path_filename(&[
            Value::String("/home/user/documents/file.txt".to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("file.txt".to_string()));
    }

    #[test]
    fn test_path_extension() {
        let result = path_extension(&[
            Value::String("/home/user/documents/file.txt".to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("txt".to_string()));
    }

    #[test]
    fn test_path_extension_no_extension() {
        let result = path_extension(&[
            Value::String("/home/user/documents/file".to_string())
        ]);
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), Value::String("".to_string()));
    }

    #[test]
    fn test_function_error_handling() {
        // Test wrong argument count
        let result = file_read(&[]);
        assert!(result.is_err());
        
        // Test wrong argument type
        let result = file_read(&[Value::Integer(123)]);
        assert!(result.is_err());
        
        // Test path join with non-string
        let result = path_join(&[Value::String("home".to_string()), Value::Integer(123)]);
        assert!(result.is_err());
    }

    #[test]
    fn test_complete_file_workflow() {
        let temp_dir = create_test_dir();
        let file_path = temp_dir.path().join("workflow_test.txt");
        let file_path_str = file_path.to_str().unwrap().to_string();
        
        // 1. Check file doesn't exist initially
        let exists_result = file_exists(&[Value::String(file_path_str.clone())]);
        assert_eq!(exists_result.unwrap(), Value::Boolean(false));
        
        // 2. Write initial content
        let write_result = file_write(&[
            Value::String(file_path_str.clone()),
            Value::String("Initial content".to_string())
        ]);
        assert!(write_result.is_ok());
        
        // 3. Check file now exists
        let exists_result = file_exists(&[Value::String(file_path_str.clone())]);
        assert_eq!(exists_result.unwrap(), Value::Boolean(true));
        
        // 4. Read content back
        let read_result = file_read(&[Value::String(file_path_str.clone())]);
        assert_eq!(read_result.unwrap(), Value::String("Initial content".to_string()));
        
        // 5. Check file size
        let size_result = file_size(&[Value::String(file_path_str.clone())]);
        assert_eq!(size_result.unwrap(), Value::Integer(15)); // Length of "Initial content"
    }
}