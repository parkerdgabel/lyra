//! I/O functions for the Lyra standard library
//! 
//! This module provides comprehensive file I/O operations including:
//! - File operations: read, write, append, exists, size, delete, copy
//! - Directory operations: create, delete, exists, list, size
//! - Path operations: join, split, parent, filename, extension, absolute
//! - Import/Export functionality with comprehensive format support

use crate::vm::{Value, VmResult, VmError};
use std::fs;
use std::io::{Write, BufRead, BufReader};
use std::path::{Path, PathBuf};

/// Import[filename] - Load data from external file
/// Import[filename, format] - Load data with specific format
/// 
/// Examples:
/// - `Import["data.json"]` → Load JSON data with auto-detection
/// - `Import["data.csv", "CSV"]` → Load CSV with explicit format
/// - `Import["config.txt"]` → Load text file
pub fn import(args: &[Value]) -> VmResult<Value> {
    crate::io::import(args)
}

/// Export[data, filename] - Save data to external file  
/// Export[data, filename, format] - Save data with specific format
/// 
/// Examples:
/// - `Export[data, "output.json"]` → Save as JSON with auto-detection
/// - `Export[data, "output.csv", "CSV"]` → Save as CSV with explicit format
/// - `Export[text, "output.txt"]` → Save as plain text
pub fn export(args: &[Value]) -> VmResult<Value> {
    crate::io::export(args)
}

// ================================================================================
// FILE OPERATIONS
// ================================================================================

/// FileRead[path] - Read entire file content as string
/// 
/// Arguments:
/// - path: String - Path to the file to read
/// 
/// Returns:
/// - String containing the file content
/// 
/// Examples:
/// - `FileRead["data.txt"]` → Read entire file as string
/// - `FileRead["/path/to/file.log"]` → Read log file content
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

/// FileReadLines[path] - Read file as list of lines
/// 
/// Arguments:
/// - path: String - Path to the file to read
/// 
/// Returns:
/// - List of strings, one per line
/// 
/// Examples:
/// - `FileReadLines["config.txt"]` → Read file as list of lines
/// - `FileReadLines["data.csv"]` → Read CSV file line by line
pub fn file_read_lines(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("FileReadLines requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    match fs::File::open(path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let lines: Result<Vec<Value>, _> = reader
                .lines()
                .map(|line| line.map(Value::String).map_err(|e| 
                    VmError::Runtime(format!("Failed to read line: {}", e))
                ))
                .collect();
            
            match lines {
                Ok(line_list) => Ok(Value::List(line_list)),
                Err(e) => Err(e),
            }
        }
        Err(e) => Err(VmError::Runtime(format!("Failed to open file '{}': {}", path, e))),
    }
}

/// FileWrite[path, content] - Write content to file
/// FileWrite[path, content, options] - Write content to file with options
/// 
/// Arguments:
/// - path: String - Path to the file to write
/// - content: String - Content to write to the file
/// - options: String (optional) - Write options ("overwrite" (default), "create_new")
/// 
/// Returns:
/// - String containing the written file path
/// 
/// Examples:
/// - `FileWrite["output.txt", "Hello, World!"]` → Write string to file
/// - `FileWrite["new.txt", data, "create_new"]` → Write only if file doesn't exist
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
        Value::Real(f) => f.to_string().as_bytes(),
        Value::Boolean(b) => b.to_string().as_bytes(),
        _ => return Err(VmError::Runtime("Content must be convertible to string".to_string())),
    };
    
    let options = if args.len() == 3 {
        match &args[2] {
            Value::String(s) => s.as_str(),
            _ => return Err(VmError::Runtime("Options must be a string".to_string())),
        }
    } else {
        "overwrite"
    };
    
    // Create parent directories if they don't exist
    if let Some(parent) = Path::new(path).parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return Err(VmError::Runtime(format!("Failed to create parent directories: {}", e)));
        }
    }
    
    let result = match options {
        "create_new" => {
            if Path::new(path).exists() {
                return Err(VmError::Runtime(format!("File '{}' already exists", path)));
            }
            fs::write(path, content)
        }
        "overwrite" | _ => fs::write(path, content),
    };
    
    match result {
        Ok(()) => Ok(Value::String(path.to_string())),
        Err(e) => Err(VmError::Runtime(format!("Failed to write file '{}': {}", path, e))),
    }
}

/// FileAppend[path, content] - Append content to file
/// 
/// Arguments:
/// - path: String - Path to the file to append to
/// - content: String - Content to append to the file
/// 
/// Returns:
/// - String containing the file path
/// 
/// Examples:
/// - `FileAppend["log.txt", "New log entry\n"]` → Append to log file
/// - `FileAppend["data.csv", "new,row,data\n"]` → Append new row to CSV
pub fn file_append(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("FileAppend requires exactly 2 arguments (path, content)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let content = match &args[1] {
        Value::String(s) => s.as_bytes(),
        Value::Integer(n) => n.to_string().as_bytes(),
        Value::Real(f) => f.to_string().as_bytes(),
        Value::Boolean(b) => b.to_string().as_bytes(),
        _ => return Err(VmError::Runtime("Content must be convertible to string".to_string())),
    };
    
    match fs::OpenOptions::new().create(true).append(true).open(path) {
        Ok(mut file) => {
            match file.write_all(content) {
                Ok(()) => Ok(Value::String(path.to_string())),
                Err(e) => Err(VmError::Runtime(format!("Failed to append to file '{}': {}", path, e))),
            }
        }
        Err(e) => Err(VmError::Runtime(format!("Failed to open file '{}' for appending: {}", path, e))),
    }
}

/// FileExists[path] - Check if file exists
/// 
/// Arguments:
/// - path: String - Path to check
/// 
/// Returns:
/// - Boolean indicating whether the file exists
/// 
/// Examples:
/// - `FileExists["data.txt"]` → True if file exists, False otherwise
/// - `FileExists["/path/to/file"]` → Check if file exists at absolute path
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

/// FileSize[path] - Get file size in bytes
/// 
/// Arguments:
/// - path: String - Path to the file
/// 
/// Returns:
/// - Integer representing file size in bytes
/// 
/// Examples:
/// - `FileSize["data.txt"]` → Get size of file in bytes
/// - `FileSize["/path/to/large_file.bin"]` → Check size of binary file
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

/// FileDelete[path] - Delete a file
/// 
/// Arguments:
/// - path: String - Path to the file to delete
/// 
/// Returns:
/// - Boolean indicating success
/// 
/// Examples:
/// - `FileDelete["temp.txt"]` → Delete temporary file
/// - `FileDelete["/path/to/old_file.log"]` → Delete old log file
pub fn file_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("FileDelete requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    if !Path::new(path).exists() {
        return Err(VmError::Runtime(format!("File '{}' does not exist", path)));
    }
    
    if !Path::new(path).is_file() {
        return Err(VmError::Runtime(format!("'{}' is not a file", path)));
    }
    
    match fs::remove_file(path) {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to delete file '{}': {}", path, e))),
    }
}

/// FileCopy[source, destination] - Copy file
/// 
/// Arguments:
/// - source: String - Path to the source file
/// - destination: String - Path to the destination file
/// 
/// Returns:
/// - String containing the destination path
/// 
/// Examples:
/// - `FileCopy["source.txt", "backup.txt"]` → Copy file to backup
/// - `FileCopy["/data/file.csv", "/backup/file.csv"]` → Copy to different directory
pub fn file_copy(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("FileCopy requires exactly 2 arguments (source, destination)".to_string()));
    }
    
    let source = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Source path must be a string".to_string())),
    };
    
    let destination = match &args[1] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Destination path must be a string".to_string())),
    };
    
    if !Path::new(source).exists() {
        return Err(VmError::Runtime(format!("Source file '{}' does not exist", source)));
    }
    
    if !Path::new(source).is_file() {
        return Err(VmError::Runtime(format!("Source '{}' is not a file", source)));
    }
    
    // Create parent directories for destination if they don't exist
    if let Some(parent) = Path::new(destination).parent() {
        if let Err(e) = fs::create_dir_all(parent) {
            return Err(VmError::Runtime(format!("Failed to create destination directories: {}", e)));
        }
    }
    
    match fs::copy(source, destination) {
        Ok(_) => Ok(Value::String(destination.to_string())),
        Err(e) => Err(VmError::Runtime(format!("Failed to copy '{}' to '{}': {}", source, destination, e))),
    }
}

// ================================================================================
// DIRECTORY OPERATIONS
// ================================================================================

/// DirectoryCreate[path] - Create directory (including parents)
/// 
/// Arguments:
/// - path: String - Path to the directory to create
/// 
/// Returns:
/// - String containing the created directory path
/// 
/// Examples:
/// - `DirectoryCreate["new_folder"]` → Create directory in current location
/// - `DirectoryCreate["/path/to/nested/dirs"]` → Create nested directories
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

/// DirectoryDelete[path, recursive] - Delete directory
/// 
/// Arguments:
/// - path: String - Path to the directory to delete
/// - recursive: Boolean - Whether to delete non-empty directories
/// 
/// Returns:
/// - Boolean indicating success
/// 
/// Examples:
/// - `DirectoryDelete["empty_folder", False]` → Delete empty directory
/// - `DirectoryDelete["folder_with_files", True]` → Delete directory and all contents
pub fn directory_delete(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("DirectoryDelete requires exactly 2 arguments (path, recursive)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let recursive = match &args[1] {
        Value::Boolean(b) => *b,
        _ => return Err(VmError::Runtime("Recursive flag must be a boolean".to_string())),
    };
    
    if !Path::new(path).exists() {
        return Err(VmError::Runtime(format!("Directory '{}' does not exist", path)));
    }
    
    if !Path::new(path).is_dir() {
        return Err(VmError::Runtime(format!("'{}' is not a directory", path)));
    }
    
    let result = if recursive {
        fs::remove_dir_all(path)
    } else {
        fs::remove_dir(path)
    };
    
    match result {
        Ok(()) => Ok(Value::Boolean(true)),
        Err(e) => Err(VmError::Runtime(format!("Failed to delete directory '{}': {}", path, e))),
    }
}

/// DirectoryExists[path] - Check if directory exists
/// 
/// Arguments:
/// - path: String - Path to check
/// 
/// Returns:
/// - Boolean indicating whether the directory exists
/// 
/// Examples:
/// - `DirectoryExists["folder"]` → True if directory exists
/// - `DirectoryExists["/path/to/dir"]` → Check if directory exists at absolute path
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

/// DirectoryList[path] - List directory contents
/// DirectoryList[path, pattern] - List directory contents with pattern filter
/// 
/// Arguments:
/// - path: String - Path to the directory to list
/// - pattern: String (optional) - Glob pattern to filter results
/// 
/// Returns:
/// - List of strings containing directory entry names
/// 
/// Examples:
/// - `DirectoryList["."]` → List current directory contents
/// - `DirectoryList["/path/to/dir", "*.txt"]` → List only .txt files
pub fn directory_list(args: &[Value]) -> VmResult<Value> {
    if args.is_empty() || args.len() > 2 {
        return Err(VmError::Runtime("DirectoryList requires 1-2 arguments (path, [pattern])".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let pattern = if args.len() == 2 {
        match &args[1] {
            Value::String(s) => Some(s.as_str()),
            _ => return Err(VmError::Runtime("Pattern must be a string".to_string())),
        }
    } else {
        None
    };
    
    if !Path::new(path).exists() {
        return Err(VmError::Runtime(format!("Directory '{}' does not exist", path)));
    }
    
    if !Path::new(path).is_dir() {
        return Err(VmError::Runtime(format!("'{}' is not a directory", path)));
    }
    
    match fs::read_dir(path) {
        Ok(entries) => {
            let mut entry_names = Vec::new();
            
            for entry in entries {
                match entry {
                    Ok(dir_entry) => {
                        let name = dir_entry.file_name().to_string_lossy().to_string();
                        
                        // Apply pattern filter if provided
                        if let Some(pat) = pattern {
                            if simple_glob_match(&name, pat) {
                                entry_names.push(Value::String(name));
                            }
                        } else {
                            entry_names.push(Value::String(name));
                        }
                    }
                    Err(e) => return Err(VmError::Runtime(format!("Failed to read directory entry: {}", e))),
                }
            }
            
            entry_names.sort_by(|a, b| {
                if let (Value::String(s1), Value::String(s2)) = (a, b) {
                    s1.cmp(s2)
                } else {
                    std::cmp::Ordering::Equal
                }
            });
            
            Ok(Value::List(entry_names))
        }
        Err(e) => Err(VmError::Runtime(format!("Failed to read directory '{}': {}", path, e))),
    }
}

/// DirectorySize[path] - Get total size of directory and all contents
/// 
/// Arguments:
/// - path: String - Path to the directory
/// 
/// Returns:
/// - Integer representing total size in bytes
/// 
/// Examples:
/// - `DirectorySize["."]` → Get size of current directory
/// - `DirectorySize["/large/data/directory"]` → Get size of data directory
pub fn directory_size(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("DirectorySize requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    if !Path::new(path).exists() {
        return Err(VmError::Runtime(format!("Directory '{}' does not exist", path)));
    }
    
    if !Path::new(path).is_dir() {
        return Err(VmError::Runtime(format!("'{}' is not a directory", path)));
    }
    
    match calculate_directory_size(Path::new(path)) {
        Ok(size) => Ok(Value::Integer(size as i64)),
        Err(e) => Err(VmError::Runtime(format!("Failed to calculate directory size for '{}': {}", path, e))),
    }
}

/// DirectoryWatch[path, callback] - Watch directory for changes (placeholder)
/// 
/// Arguments:
/// - path: String - Path to the directory to watch
/// - callback: Function - Callback function to call on changes
/// 
/// Returns:
/// - String indicating watch has been set up
/// 
/// Note: This is a placeholder implementation for future file system watching
pub fn directory_watch(args: &[Value]) -> VmResult<Value> {
    if args.len() != 2 {
        return Err(VmError::Runtime("DirectoryWatch requires exactly 2 arguments (path, callback)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    if !Path::new(path).exists() {
        return Err(VmError::Runtime(format!("Directory '{}' does not exist", path)));
    }
    
    if !Path::new(path).is_dir() {
        return Err(VmError::Runtime(format!("'{}' is not a directory", path)));
    }
    
    // Placeholder implementation - in a real system this would set up file system watching
    Ok(Value::String(format!("Directory watch set up for '{}'", path)))
}

// ================================================================================
// PATH OPERATIONS
// ================================================================================

/// PathJoin[parts...] - Join path components
/// 
/// Arguments:
/// - parts: Variable number of String arguments - Path components to join
/// 
/// Returns:
/// - String containing the joined path
/// 
/// Examples:
/// - `PathJoin["home", "user", "documents"]` → Join path components
/// - `PathJoin["/root", "folder", "file.txt"]` → Create full path
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

/// PathSplit[path] - Split path into components
/// 
/// Arguments:
/// - path: String - Path to split
/// 
/// Returns:
/// - List of strings containing path components
/// 
/// Examples:
/// - `PathSplit["/home/user/file.txt"]` → Split into components
/// - `PathSplit["relative/path/file.txt"]` → Split relative path
pub fn path_split(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("PathSplit requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let path_obj = Path::new(path);
    let components: Vec<Value> = path_obj
        .components()
        .map(|component| Value::String(component.as_os_str().to_string_lossy().to_string()))
        .collect();
    
    Ok(Value::List(components))
}

/// PathParent[path] - Get parent directory
/// 
/// Arguments:
/// - path: String - Path to get parent of
/// 
/// Returns:
/// - String containing parent directory path, or Missing if no parent
/// 
/// Examples:
/// - `PathParent["/home/user/file.txt"]` → Returns "/home/user"
/// - `PathParent["file.txt"]` → Returns "."
pub fn path_parent(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("PathParent requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let path_obj = Path::new(path);
    match path_obj.parent() {
        Some(parent) => {
            match parent.to_str() {
                Some(parent_str) => Ok(Value::String(parent_str.to_string())),
                None => Err(VmError::Runtime("Invalid parent path characters".to_string())),
            }
        }
        None => Ok(Value::Missing),
    }
}

/// PathFilename[path] - Get filename from path
/// 
/// Arguments:
/// - path: String - Path to extract filename from
/// 
/// Returns:
/// - String containing filename, or Missing if no filename
/// 
/// Examples:
/// - `PathFilename["/home/user/file.txt"]` → Returns "file.txt"
/// - `PathFilename["/home/user/"]` → Returns Missing
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

/// PathExtension[path] - Get file extension
/// 
/// Arguments:
/// - path: String - Path to extract extension from
/// 
/// Returns:
/// - String containing file extension (without dot), or empty string if no extension
/// 
/// Examples:
/// - `PathExtension["/home/user/file.txt"]` → Returns "txt"
/// - `PathExtension["file"]` → Returns ""
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

/// PathAbsolute[path] - Convert to absolute path
/// 
/// Arguments:
/// - path: String - Path to convert to absolute
/// 
/// Returns:
/// - String containing absolute path
/// 
/// Examples:
/// - `PathAbsolute["./file.txt"]` → Returns absolute path to file.txt
/// - `PathAbsolute["../data"]` → Returns absolute path to data directory
pub fn path_absolute(args: &[Value]) -> VmResult<Value> {
    if args.len() != 1 {
        return Err(VmError::Runtime("PathAbsolute requires exactly 1 argument (path)".to_string()));
    }
    
    let path = match &args[0] {
        Value::String(s) => s,
        _ => return Err(VmError::Runtime("Path must be a string".to_string())),
    };
    
    let path_obj = Path::new(path);
    match path_obj.canonicalize() {
        Ok(absolute_path) => {
            match absolute_path.to_str() {
                Some(abs_str) => Ok(Value::String(abs_str.to_string())),
                None => Err(VmError::Runtime("Invalid absolute path characters".to_string())),
            }
        }
        Err(e) => {
            // If canonicalize fails (e.g., path doesn't exist), try converting to absolute manually
            match std::env::current_dir() {
                Ok(current_dir) => {
                    let absolute_path = current_dir.join(path_obj);
                    match absolute_path.to_str() {
                        Some(abs_str) => Ok(Value::String(abs_str.to_string())),
                        None => Err(VmError::Runtime("Invalid absolute path characters".to_string())),
                    }
                }
                Err(_) => Err(VmError::Runtime(format!("Failed to convert to absolute path: {}", e))),
            }
        }
    }
}

// ================================================================================
// HELPER FUNCTIONS
// ================================================================================

// Helper function for simple glob pattern matching
fn simple_glob_match(text: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    
    if pattern.starts_with("*.") {
        let extension = &pattern[2..];
        return text.ends_with(extension);
    }
    
    if pattern.ends_with("*") {
        let prefix = &pattern[..pattern.len()-1];
        return text.starts_with(prefix);
    }
    
    // Exact match
    text == pattern
}

// Helper function to calculate directory size recursively
fn calculate_directory_size(path: &Path) -> Result<u64, std::io::Error> {
    let mut total_size = 0;
    
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        
        if metadata.is_dir() {
            total_size += calculate_directory_size(&entry.path())?;
        } else {
            total_size += metadata.len();
        }
    }
    
    Ok(total_size)
}