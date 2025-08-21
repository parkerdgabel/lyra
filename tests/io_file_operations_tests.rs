//! Tests for Lyra file I/O operations
//! 
//! These tests verify the file, directory, and path operations 
//! that will be implemented in the stdlib io module.

use std::fs;
use std::path::Path;
use tempfile::TempDir;

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

// Note: These tests will verify the underlying functionality that the stdlib functions will use
// The actual stdlib function tests will be written once the functions are implemented

#[cfg(test)]
mod core_functionality_tests {
    use super::*;

    #[test]
    fn test_file_read_functionality() {
        let temp_dir = create_test_dir();
        let file_path = create_test_file(&temp_dir, "test.txt", "Hello, World!");
        
        // Test the core functionality that FileRead will use
        let content = fs::read_to_string(&file_path).expect("Should read file");
        assert_eq!(content, "Hello, World!");
    }

    #[test]
    fn test_file_write_functionality() {
        let temp_dir = create_test_dir();
        let file_path = temp_dir.path().join("new_file.txt");
        
        // Test the core functionality that FileWrite will use
        let content = "Test content";
        fs::write(&file_path, content).expect("Should write file");
        
        let actual_content = fs::read_to_string(&file_path).expect("Should read written file");
        assert_eq!(actual_content, content);
    }

    #[test]
    fn test_file_append_functionality() {
        let temp_dir = create_test_dir();
        let file_path = create_test_file(&temp_dir, "append_test.txt", "initial");
        
        // Test the core functionality that FileAppend will use
        use std::fs::OpenOptions;
        use std::io::Write;
        
        let mut file = OpenOptions::new()
            .append(true)
            .open(&file_path)
            .expect("Should open file for append");
        
        file.write_all(b" appended").expect("Should append content");
        drop(file);
        
        let content = fs::read_to_string(&file_path).expect("Should read appended file");
        assert_eq!(content, "initial appended");
    }

    #[test]
    fn test_file_exists_functionality() {
        let temp_dir = create_test_dir();
        let existing_file = create_test_file(&temp_dir, "exists.txt", "content");
        let nonexistent_file = temp_dir.path().join("nonexistent.txt");
        
        // Test the core functionality that FileExists will use
        assert!(Path::new(&existing_file).exists());
        assert!(!nonexistent_file.exists());
    }

    #[test]
    fn test_file_size_functionality() {
        let temp_dir = create_test_dir();
        let content = "Hello, World!"; // 13 bytes
        let file_path = create_test_file(&temp_dir, "size_test.txt", content);
        
        // Test the core functionality that FileSize will use
        let metadata = fs::metadata(&file_path).expect("Should get file metadata");
        assert_eq!(metadata.len(), 13);
    }

    #[test]
    fn test_file_delete_functionality() {
        let temp_dir = create_test_dir();
        let file_path = create_test_file(&temp_dir, "delete_me.txt", "content");
        
        // Verify file exists before deletion
        assert!(Path::new(&file_path).exists());
        
        // Test the core functionality that FileDelete will use
        fs::remove_file(&file_path).expect("Should delete file");
        
        // Verify file no longer exists
        assert!(!Path::new(&file_path).exists());
    }

    #[test]
    fn test_file_copy_functionality() {
        let temp_dir = create_test_dir();
        let source_path = create_test_file(&temp_dir, "source.txt", "content to copy");
        let dest_path = temp_dir.path().join("destination.txt");
        
        // Test the core functionality that FileCopy will use
        fs::copy(&source_path, &dest_path).expect("Should copy file");
        
        // Verify destination file exists and has same content
        assert!(dest_path.exists());
        let dest_content = fs::read_to_string(&dest_path).expect("Should read copied file");
        assert_eq!(dest_content, "content to copy");
    }

    #[test]
    fn test_directory_create_functionality() {
        let temp_dir = create_test_dir();
        let new_dir_path = temp_dir.path().join("new_directory");
        
        // Test the core functionality that DirectoryCreate will use
        fs::create_dir_all(&new_dir_path).expect("Should create directory");
        
        assert!(new_dir_path.exists());
        assert!(new_dir_path.is_dir());
    }

    #[test]
    fn test_directory_exists_functionality() {
        let temp_dir = create_test_dir();
        let nonexistent_dir = temp_dir.path().join("nonexistent");
        
        // Test the core functionality that DirectoryExists will use
        assert!(temp_dir.path().is_dir());
        assert!(!nonexistent_dir.exists());
    }

    #[test]
    fn test_directory_list_functionality() {
        let temp_dir = create_test_dir();
        
        // Create some files and directories
        create_test_file(&temp_dir, "file1.txt", "content1");
        create_test_file(&temp_dir, "file2.txt", "content2");
        fs::create_dir(temp_dir.path().join("subdir")).expect("Should create subdir");
        
        // Test the core functionality that DirectoryList will use
        let entries: Vec<_> = fs::read_dir(temp_dir.path())
            .expect("Should read directory")
            .map(|entry| entry.expect("Should get directory entry"))
            .collect();
        
        assert_eq!(entries.len(), 3);
        
        let entry_names: Vec<String> = entries
            .iter()
            .map(|entry| entry.file_name().to_string_lossy().to_string())
            .collect();
        
        assert!(entry_names.contains(&"file1.txt".to_string()));
        assert!(entry_names.contains(&"file2.txt".to_string()));
        assert!(entry_names.contains(&"subdir".to_string()));
    }

    #[test]
    fn test_path_operations_functionality() {
        let test_path = Path::new("/home/user/documents/file.txt");
        
        // Test the core functionality that path operations will use
        assert_eq!(test_path.file_name().unwrap().to_str().unwrap(), "file.txt");
        assert_eq!(test_path.extension().unwrap().to_str().unwrap(), "txt");
        assert_eq!(test_path.parent().unwrap().to_str().unwrap(), "/home/user/documents");
        
        let joined = Path::new("home").join("user").join("file.txt");
        assert_eq!(joined.to_str().unwrap(), "home/user/file.txt");
        
        let components: Vec<_> = test_path.components().collect();
        assert!(components.len() > 0);
    }
}